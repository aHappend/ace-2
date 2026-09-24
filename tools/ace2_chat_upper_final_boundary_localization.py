#!/usr/bin/env python3
"""Localize first-token quality across the upper decoder and final boundaries."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import Tensor
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_full_model_fixed_point as fixed_point
from tools.ace2_chat_all_residual_reference import (
    FROZEN_SCALES,
    MODEL_REPOSITORY,
    ROPE_MECHANISM,
    exact_process_argv,
    frozen_ranges,
    top_k,
    utc_now,
    validate_frozen_scale_binding,
)
from tools.ace2_chat_demo import (
    REVISION,
    TOKENIZER_SNAPSHOT,
    canonical_bytes,
    load_tokenizer,
    sha256_file,
    tokenize_prompt,
    write_atomic,
)
from tools.ace2_chat_layer_prefix_substitution import (
    PrefixSubstitution,
    relative_l2,
    saturation,
)


PROMPT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Mode:
    name: str
    exact_prefix_layers: int
    exact_final_norm: bool = False
    exact_head_input: str | None = None


MODES = (
    Mode("exact_through_layer21", 22),
    Mode("exact_through_layer22", 23),
    Mode("exact_through_layer23", 24),
    Mode("exact_final_norm_quantized_fixed_head", 24, True),
    Mode("bf16_head_on_quantized_exact_norm", 24, True, "quantized"),
    Mode("full_bf16_upper_final", 24, True, "exact"),
)


def parse_prompts(values: list[str]) -> dict[str, str]:
    if not values:
        values = [
            "hello_world=Hello world",
            "blue=Write one word: blue",
        ]
    prompts: dict[str, str] = {}
    for value in values:
        name, separator, prompt = value.partition("=")
        name = name.strip()
        prompt = prompt.strip()
        if not separator or not PROMPT_NAME.fullmatch(name):
            raise RuntimeError("--prompt must use a unique NAME=TEXT with a lowercase name")
        if not prompt:
            raise RuntimeError(f"prompt {name!r} is empty")
        if name in prompts:
            raise RuntimeError(f"duplicate prompt name: {name}")
        prompts[name] = prompt
    return prompts


def oracle_rank(logits: Tensor, token_id: int) -> int:
    selected = logits[token_id]
    return int((logits > selected).sum()) + 1


def top_k_overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    left_ids = {int(item["token_id"]) for item in left}
    right_ids = {int(item["token_id"]) for item in right}
    return len(left_ids & right_ids)


def logits_record(
    tokenizer: Any,
    logits: Tensor,
    oracle_logits: Tensor,
    *,
    top_k_count: int,
) -> dict[str, Any]:
    candidates = top_k(tokenizer, logits, top_k_count)
    oracle_candidates = top_k(tokenizer, oracle_logits, top_k_count)
    oracle_token_id = int(oracle_candidates[0]["token_id"])
    return {
        "selected_token_id": int(candidates[0]["token_id"]),
        "selected_piece": candidates[0]["decoded_piece"],
        "top_k": candidates,
        "relative_l2_vs_bf16_oracle": relative_l2(oracle_logits, logits),
        "oracle_token_rank": oracle_rank(logits, oracle_token_id),
        "top_k_token_overlap_with_oracle": top_k_overlap(
            candidates, oracle_candidates
        ),
    }


class FinalNormSubstitution:
    def __init__(self, fixed_norm: Any, source_norm: Any) -> None:
        self.fixed_norm = fixed_norm
        self.source_norm = source_norm
        self.handle: Any | None = None
        self.record: dict[str, Any] | None = None
        self.exact_output: Tensor | None = None
        self.quantized_dequantized: Tensor | None = None

    def install(self) -> None:
        def hook(
            module: Any,
            args: tuple[Tensor, ...],
            fixed_output: Tensor,
        ) -> Tensor:
            if not args:
                raise RuntimeError("final RMSNorm hook lacks hidden states")
            exact_output = self.source_norm(args[0])
            quantized = fixed_point.quantize_int8(exact_output, module.output_scale)
            fixed_dequantized = fixed_output.to(exact_output.dtype) * module.output_scale
            quantized_dequantized = (
                quantized.to(exact_output.dtype) * module.output_scale
            )
            self.exact_output = exact_output.detach().clone()
            self.quantized_dequantized = quantized_dequantized.detach().clone()
            self.record = {
                "fixed_shadow_vs_exact_relative_l2": relative_l2(
                    exact_output, fixed_dequantized
                ),
                "exact_quantization_vs_exact_relative_l2": relative_l2(
                    exact_output, quantized_dequantized
                ),
                "output_scale": float(module.output_scale),
                "fixed_shadow_saturation": saturation(fixed_output),
                "exact_quantized_saturation": saturation(quantized),
            }
            return quantized.to(exact_output.dtype)

        self.handle = self.fixed_norm.register_forward_hook(hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class HeadSubstitution:
    def __init__(
        self,
        fixed_head: Any,
        source_head: Any,
        norm_substitution: FinalNormSubstitution,
        source_input: str,
    ) -> None:
        if source_input not in {"quantized", "exact"}:
            raise ValueError("source head input must be quantized or exact")
        self.fixed_head = fixed_head
        self.source_head = source_head
        self.norm_substitution = norm_substitution
        self.source_input = source_input
        self.handle: Any | None = None
        self.record: dict[str, Any] | None = None

    def install(self) -> None:
        def hook(
            _module: Any,
            _args: tuple[Tensor, ...],
            fixed_output: Tensor,
        ) -> Tensor:
            source_hidden = (
                self.norm_substitution.quantized_dequantized
                if self.source_input == "quantized"
                else self.norm_substitution.exact_output
            )
            if source_hidden is None:
                raise RuntimeError("LM head executed before final RMSNorm capture")
            exact_output = self.source_head(source_hidden)
            self.record = {
                "source_input": self.source_input,
                "fixed_shadow_vs_bf16_head_relative_l2": relative_l2(
                    exact_output, fixed_output
                ),
            }
            return exact_output

        self.handle = self.fixed_head.register_forward_hook(hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def final_prefix_record(
    substitution: PrefixSubstitution,
    prompt_name: str,
    boundary_layer: int,
) -> dict[str, Any]:
    selected = [
        record
        for record in substitution.records
        if record["prompt"] == prompt_name
        and int(record["layer_index"]) == boundary_layer
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one layer-{boundary_layer} boundary record, got {len(selected)}"
        )
    return selected[0]


def run_mode(
    fixed_model: Any,
    source_model: Any,
    tokenizer: Any,
    prompt_name: str,
    input_ids: Tensor,
    oracle_logits: Tensor,
    mode: Mode,
    *,
    top_k_count: int,
) -> dict[str, Any]:
    prefix = PrefixSubstitution(
        fixed_model, source_model, mode.exact_prefix_layers
    )
    prefix.set_step(prompt_name, 0)
    norm_substitution: FinalNormSubstitution | None = None
    head_substitution: HeadSubstitution | None = None
    prefix.install()
    try:
        if mode.exact_final_norm:
            norm_substitution = FinalNormSubstitution(
                fixed_model.model.norm, source_model.model.norm
            )
            norm_substitution.install()
        if mode.exact_head_input is not None:
            if norm_substitution is None:
                raise AssertionError("exact LM head requires exact final RMSNorm")
            head_substitution = HeadSubstitution(
                fixed_model.lm_head,
                source_model.lm_head,
                norm_substitution,
                mode.exact_head_input,
            )
            head_substitution.install()
        with torch.inference_mode():
            logits = fixed_model(input_ids=input_ids, use_cache=False).logits[0, -1]
    finally:
        if head_substitution is not None:
            head_substitution.remove()
        if norm_substitution is not None:
            norm_substitution.remove()
        prefix.remove()

    record = logits_record(
        tokenizer,
        logits,
        oracle_logits,
        top_k_count=top_k_count,
    )
    record.update(
        {
            "mode": mode.name,
            "exact_prefix_layers": mode.exact_prefix_layers,
            "newly_exact_boundary": (
                f"model.layers.{mode.exact_prefix_layers - 1}"
                if not mode.exact_final_norm
                else (
                    "lm_head"
                    if mode.exact_head_input is not None
                    else "model.norm"
                )
            ),
            "prefix_boundary": final_prefix_record(
                prefix, prompt_name, mode.exact_prefix_layers - 1
            ),
            "final_norm_boundary": (
                norm_substitution.record if norm_substitution is not None else None
            ),
            "lm_head_boundary": (
                head_substitution.record if head_substitution is not None else None
            ),
        }
    )
    return record


def add_stage_deltas(records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records):
        if index == 0:
            record["delta_vs_previous_stage"] = None
            continue
        previous = records[index - 1]
        record["delta_vs_previous_stage"] = {
            "selected_token_changed": record["selected_token_id"]
            != previous["selected_token_id"],
            "oracle_token_rank_change": record["oracle_token_rank"]
            - previous["oracle_token_rank"],
            "relative_l2_to_oracle_change": (
                record["relative_l2_vs_bf16_oracle"]
                - previous["relative_l2_vs_bf16_oracle"]
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        metavar="NAME=TEXT",
        help="repeatable named prompt; defaults to hello_world and blue",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 64:
        raise RuntimeError("--top-k must be in 1..64")
    prompts = parse_prompts(args.prompt)

    started = time.perf_counter()
    scales = json.loads(FROZEN_SCALES.read_text(encoding="utf-8"))
    ranges, operator_ranges = frozen_ranges(scales)
    tokenizer = load_tokenizer()
    tokenizations = {
        name: tokenize_prompt(tokenizer, prompt, "")
        for name, prompt in prompts.items()
    }
    source_model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    fixed_model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    fixed_point.replace_linears(
        fixed_model, ranges, rope_diagnostic_mechanism=ROPE_MECHANISM
    )
    fixed_point.replace_fixed_operators(
        fixed_model,
        operator_ranges,
        rope_diagnostic_mechanism=ROPE_MECHANISM,
        all_projection_residual_fusion=True,
    )
    scale_binding = validate_frozen_scale_binding(fixed_model, scales)

    prompt_results: dict[str, Any] = {}
    oracle_agreement = True
    for prompt_name, tokenization in tokenizations.items():
        input_ids = torch.tensor(
            [tokenization["chat_template_token_ids"]], dtype=torch.long
        )
        with torch.inference_mode():
            oracle_logits = source_model(
                input_ids=input_ids, use_cache=False
            ).logits[0, -1]
        oracle_top_k = top_k(tokenizer, oracle_logits, args.top_k)
        records = [
            run_mode(
                fixed_model,
                source_model,
                tokenizer,
                prompt_name,
                input_ids,
                oracle_logits,
                mode,
                top_k_count=args.top_k,
            )
            for mode in MODES
        ]
        add_stage_deltas(records)
        full_exact = records[-1]
        prompt_oracle_agreement = bool(
            full_exact["selected_token_id"] == int(oracle_top_k[0]["token_id"])
            and full_exact["relative_l2_vs_bf16_oracle"] <= 1e-6
            and [item["token_id"] for item in full_exact["top_k"]]
            == [item["token_id"] for item in oracle_top_k]
        )
        oracle_agreement = oracle_agreement and prompt_oracle_agreement
        prompt_results[prompt_name] = {
            "prompt": prompts[prompt_name],
            "tokenization": tokenization,
            "bf16_oracle": {
                "selected_token_id": int(oracle_top_k[0]["token_id"]),
                "selected_piece": oracle_top_k[0]["decoded_piece"],
                "top_k": oracle_top_k,
            },
            "stages": records,
            "full_exact_chain_matches_direct_bf16_oracle": prompt_oracle_agreement,
        }
        print(
            "ACE2_UPPER_FINAL_BOUNDARY_PROMPT "
            f"prompt={prompt_name} "
            f"tokens={[item['selected_token_id'] for item in records]} "
            f"oracle={oracle_top_k[0]['token_id']} "
            f"oracle_match={prompt_oracle_agreement}",
            flush=True,
        )

    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "chat_v2_upper_final_first_token_boundary_localization",
        "contract": {
            "acceptance_candidate": False,
            "first_token_only": True,
            "reference_only_shadow_diagnostic": True,
            "does_not_invoke_verilated_shell": True,
            "does_not_modify_rtl_or_ppa": True,
            "semantics": (
                "Each stage executes the frozen W4A8 model as a shadow and replaces "
                "successive outputs with BF16 reference outputs. The final two stages "
                "separate W4 LM-head error from final-normalization output quantization."
            ),
        },
        "invocation": {"argv": exact_process_argv(), "cwd": str(Path.cwd())},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": REVISION,
            "dtype": "bfloat16",
            "quantization": "frozen W4A8 software shadow",
        },
        "mode_order": [mode.name for mode in MODES],
        "prompts": prompt_results,
        "scale_authentication": scale_binding,
        "artifacts": {
            "frozen_scales": {
                "path": str(FROZEN_SCALES.relative_to(ROOT)),
                "sha256": sha256_file(FROZEN_SCALES),
            },
            "fixed_point_source": {
                "path": "tools/ace2_full_model_fixed_point.py",
                "sha256": sha256_file(
                    ROOT / "tools/ace2_full_model_fixed_point.py"
                ),
            },
            "prefix_substitution_source": {
                "path": "tools/ace2_chat_layer_prefix_substitution.py",
                "sha256": sha256_file(
                    ROOT / "tools/ace2_chat_layer_prefix_substitution.py"
                ),
            },
            "runner_source": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "full_exact_chain_matches_direct_bf16_oracle": oracle_agreement,
        "wall_seconds": time.perf_counter() - started,
        "status": (
            "PASS_UPPER_FINAL_BOUNDARY_LOCALIZATION_COMPLETE"
            if oracle_agreement
            else "FAIL_UPPER_FINAL_BOUNDARY_ORACLE_MISMATCH"
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, canonical_bytes(result))
    print(
        "ACE2_UPPER_FINAL_BOUNDARY_RESULT "
        f"status={result['status']} wall_seconds={result['wall_seconds']:.6f} "
        f"output={output}",
        flush=True,
    )
    return 0 if oracle_agreement else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_UPPER_FINAL_BOUNDARY_FAIL detail={error}", file=sys.stderr)
        raise
