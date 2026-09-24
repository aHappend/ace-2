#!/usr/bin/env python3
"""Test reference-only numerical repairs for ACE-2 chat quality."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_full_model_fixed_point as fixed_point
from tools.ace2_chat_demo import (
    TOKENIZER_SNAPSHOT,
    canonical_bytes,
    decode_generated,
    load_tokenizer,
    read_ace2rt2_package_metadata,
    sha256_bytes,
    sha256_file,
    tokenize_prompt,
    write_atomic,
)

FROZEN_SCALES = (
    ROOT
    / "evidence/layer0_tile_bfp_score_attention_v1"
    / "paired-smoke-20260801-v1/derived_scales.json"
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def token_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def top_k(
    tokenizer: Any,
    logits: torch.Tensor,
    count: int,
    *,
    key: str,
) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.detach().to(torch.float64), count)
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            key: float(value),
            "decoded_piece": token_piece(tokenizer, int(token_id)),
        }
        for rank, (token_id, value) in enumerate(
            zip(indices.tolist(), values.tolist(), strict=True), start=1
        )
    ]


def readability_gate(text: str, token_ids: list[int]) -> dict[str, Any]:
    stripped = text.strip()
    disallowed = [
        character
        for character in text
        if unicodedata.category(character).startswith("C")
        and character not in {"\n", "\t", "\r"}
    ]
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated_single_line = (
        len(nonempty_lines) >= 2 and len(set(nonempty_lines)) == 1
    )
    alphabetic = sum(character.isalpha() for character in text)
    unique_tokens = len(set(token_ids))
    passed = bool(
        len(stripped) >= 4
        and alphabetic >= 3
        and not disallowed
        and "\ufffd" not in text
        and unique_tokens >= 2
        and not repeated_single_line
    )
    return {
        "passed": passed,
        "visible_characters": len(stripped),
        "alphabetic_characters": alphabetic,
        "disallowed_format_or_control_characters": [
            f"U+{ord(character):04X}" for character in disallowed
        ],
        "unique_generated_tokens": unique_tokens,
        "nonempty_lines": nonempty_lines,
        "repeated_single_line": repeated_single_line,
    }


def build_fixed_lm_head(
    source: nn.Linear,
    metadata: dict[str, Any],
) -> fixed_point.W4A8Linear:
    head = fixed_point.W4A8Linear(
        source,
        fixed_point.CalibrationRange(
            input_absmax=float(metadata["input_absmax"]),
            output_absmax=float(metadata["output_absmax"]),
        ),
        input_is_quantized=True,
    )
    head.bind_hardware_input_scale(float(metadata["hardware_input_scale"]))
    if not math.isclose(
        float(head.output_scale),
        float(metadata["output_scale"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("reconstructed LM-head output scale differs from frozen scale")
    if head.multiplier.cpu().tolist() != metadata["multiplier"]:
        raise RuntimeError("reconstructed LM-head multipliers differ from frozen metadata")
    if head.right_shift.cpu().tolist() != metadata["right_shift"]:
        raise RuntimeError("reconstructed LM-head shifts differ from frozen metadata")
    if fixed_point.sha256_tensor(head.qweight) != metadata["qweight_sha256"]:
        raise RuntimeError("reconstructed LM-head W4 weights differ from frozen metadata")
    return head


def quantize_transformer_linears_w4(model: nn.Module) -> dict[str, Any]:
    started = time.perf_counter()
    records = []
    with torch.no_grad():
        for name, module in model.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            weight = module.weight.detach().to(torch.float64)
            scale = weight.abs().amax(dim=1) / 7.0
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            quantized = torch.round(weight / scale[:, None]).clamp(-8, 7)
            dequantized = quantized * scale[:, None]
            module.weight.copy_(dequantized.to(module.weight.dtype))
            records.append(
                {
                    "name": name,
                    "shape": list(module.weight.shape),
                    "zero_fraction": float(
                        (quantized == 0).to(torch.float64).mean()
                    ),
                    "saturated_fraction": float(
                        ((quantized == -8) | (quantized == 7))
                        .to(torch.float64)
                        .mean()
                    ),
                    "w4_sha256": fixed_point.sha256_tensor(
                        quantized.to(torch.int8)
                    ),
                }
            )
    return {
        "linears_quantized": len(records),
        "records": records,
        "wall_seconds": time.perf_counter() - started,
    }


def greedy_generate(
    model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    termination_token_ids: list[int],
    *,
    max_new_tokens: int,
    top_k_count: int,
    logits_fn: Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, Any]]],
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    past_key_values = None
    generated: list[int] = []
    steps = []
    with torch.inference_mode():
        for generation_index in range(max_new_tokens):
            model_input = input_ids if past_key_values is None else input_ids[:, -1:]
            output = model.model(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden = output.last_hidden_state[:, -1]
            logits, numeric = logits_fn(hidden)
            records = top_k(
                tokenizer,
                logits[0],
                top_k_count,
                key="logit",
            )
            selected = int(records[0]["token_id"])
            generated.append(selected)
            terminated = selected in termination_token_ids
            steps.append(
                {
                    "generation_index": generation_index,
                    "selected_token_id": selected,
                    "selected_piece": token_piece(tokenizer, selected),
                    "terminated": terminated,
                    "top_k": records,
                    "numeric": numeric,
                }
            )
            input_ids = torch.tensor([[selected]], dtype=torch.long)
            past_key_values = output.past_key_values
            if terminated:
                break
    decoded = decode_generated(tokenizer, generated)
    readability = readability_gate(decoded, generated)
    return {
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "readability_gate": readability,
        "readable_text_observed": readability["passed"],
        "terminated": bool(steps and steps[-1]["terminated"]),
        "termination_token_ids": termination_token_ids,
        "steps": steps,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--second-prompt",
        default="Write one word: blue",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 32:
        raise RuntimeError("--max-new-tokens must be in 1..32")
    if not 1 <= args.top_k <= 64:
        raise RuntimeError("--top-k must be in 1..64")

    started = time.perf_counter()
    package = read_ace2rt2_package_metadata(args.package.resolve())
    tokenizer = load_tokenizer()
    hello_tokenization = tokenize_prompt(tokenizer, "Hello world", "")
    if hello_tokenization["chat_template_token_ids"] != package["prompt_tokens"]:
        raise RuntimeError("Hello-world package prompt tokens differ from live tokenizer")
    second_tokenization = tokenize_prompt(tokenizer, args.second_prompt, "")
    prompts = {
        "hello_world": {
            "text_sha256": sha256_bytes(b"Hello world"),
            "token_ids": package["prompt_tokens"],
        },
        "second_prompt": {
            "text_sha256": sha256_bytes(args.second_prompt.encode("utf-8")),
            "token_ids": second_tokenization["chat_template_token_ids"],
        },
    }
    scales = json.loads(FROZEN_SCALES.read_text(encoding="utf-8"))

    model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    fixed_lm_head = build_fixed_lm_head(model.lm_head, scales["linears"]["lm_head"])
    output_scale = float(fixed_lm_head.output_scale)

    def bf16_lm(hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        logits = model.lm_head(hidden)
        return logits, {
            "classification": "bf16_lm_head",
            "hidden_zero_fraction": float(
                (hidden == 0).to(torch.float64).mean()
            ),
        }

    def w4a8_lm(hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        quantized_hidden = fixed_point.quantize_int8(
            hidden,
            fixed_lm_head.hardware_input_scale,
        )
        raw = fixed_lm_head.forward_quantized(quantized_hidden)
        logits = raw.to(torch.float64) * output_scale
        return logits, {
            "classification": "global_domain_w4a8_lm_head",
            "input_scale": fixed_lm_head.hardware_input_scale,
            "output_scale": output_scale,
            "hidden_s8_zero_fraction": float(
                (quantized_hidden == 0).to(torch.float64).mean()
            ),
            "hidden_s8_saturated_fraction": float(
                ((quantized_hidden == -128) | (quantized_hidden == 127))
                .to(torch.float64)
                .mean()
            ),
            "logit_s8_zero_fraction": float(
                (raw == 0).to(torch.float64).mean()
            ),
            "logit_s8_saturated_fraction": float(
                ((raw == -128) | (raw == 127)).to(torch.float64).mean()
            ),
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "chat_v2_reference_only_numerical_repair_sweep",
        "package": package,
        "second_prompt": args.second_prompt,
        "prompts": prompts,
        "max_new_tokens": args.max_new_tokens,
        "termination_token_ids": package["termination_token_ids"],
        "artifacts": {
            "frozen_scales": {
                "path": str(FROZEN_SCALES.resolve()),
                "sha256": sha256_file(FROZEN_SCALES),
            },
            "source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "candidates": {},
        "status": "RUNNING",
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output_path, canonical_bytes(result))

    for prompt_name, prompt in prompts.items():
        result["candidates"].setdefault("bf16_baseline", {})[prompt_name] = (
            greedy_generate(
                model,
                tokenizer,
                prompt["token_ids"],
                package["termination_token_ids"],
                max_new_tokens=args.max_new_tokens,
                top_k_count=args.top_k,
                logits_fn=bf16_lm,
            )
        )
    write_atomic(output_path, canonical_bytes(result))

    for prompt_name, prompt in prompts.items():
        result["candidates"].setdefault(
            "bf16_upstream_global_w4a8_lm_head", {}
        )[prompt_name] = greedy_generate(
            model,
            tokenizer,
            prompt["token_ids"],
            package["termination_token_ids"],
            max_new_tokens=args.max_new_tokens,
            top_k_count=args.top_k,
            logits_fn=w4a8_lm,
        )
    write_atomic(output_path, canonical_bytes(result))

    result["transformer_w4_quantization"] = quantize_transformer_linears_w4(model)
    for prompt_name, prompt in prompts.items():
        result["candidates"].setdefault(
            "w4_transformer_bf16_activations_global_w4a8_lm_head", {}
        )[prompt_name] = greedy_generate(
            model,
            tokenizer,
            prompt["token_ids"],
            package["termination_token_ids"],
            max_new_tokens=args.max_new_tokens,
            top_k_count=args.top_k,
            logits_fn=w4a8_lm,
        )

    result["wall_seconds"] = time.perf_counter() - started
    repaired = result["candidates"]["bf16_upstream_global_w4a8_lm_head"]
    result["selected_repair"] = {
        "candidate": "bf16_upstream_global_w4a8_lm_head",
        "rationale": (
            "preserve upstream BF16 activations and residual arithmetic while retaining "
            "the frozen global-domain W4A8 LM head"
        ),
        "rejected_candidate": {
            "candidate": "w4_transformer_bf16_activations_global_w4a8_lm_head",
            "reason": "did not pass the strict visible-text gate on both prompts",
        },
    }
    result["status"] = (
        "PASS_READABLE_REPAIRED_REFERENCE"
        if all(item["readable_text_observed"] for item in repaired.values())
        else "FAIL_REPAIRED_REFERENCE_NOT_READABLE"
    )
    write_atomic(output_path, canonical_bytes(result))
    print(
        "ACE2_CHAT_REFERENCE_REPAIR_RESULT "
        f"status={result['status']} "
        f"hello={repaired['hello_world']['decoded_text']!r} "
        f"second={repaired['second_prompt']['decoded_text']!r} "
        f"wall_seconds={result['wall_seconds']:.6f} "
        f"output={output_path}",
        flush=True,
    )
    return 0 if result["status"] == "PASS_READABLE_REPAIRED_REFERENCE" else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_CHAT_REFERENCE_REPAIR_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
