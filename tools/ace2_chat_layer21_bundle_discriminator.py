#!/usr/bin/env python3
"""Bounded Layer-21 attention-versus-MLP exact-residual discriminator."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
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
    prompt_coherence,
    sha256_bytes,
    summarize_execution,
    token_piece,
    top_k,
    utc_now,
    validate_frozen_scale_binding,
)
from tools.ace2_chat_demo import (
    REVISION,
    TOKENIZER_SNAPSHOT,
    canonical_bytes,
    decode_generated,
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
from tools.ace2_chat_reference_repairs import readability_gate


PREFIX21 = (
    ROOT
    / "build/ace2_chat_diagnostics/layer-prefix-substitution-p21-two-prompts-20260805.json"
)
PREFIX22 = (
    ROOT
    / "build/ace2_chat_diagnostics/layer-prefix-substitution-p22-eight-token-two-prompts-20260805.json"
)
LAYER_INDEX = 21


class Layer21BundleDiscriminator:
    def __init__(
        self,
        fixed_model: Any,
        source_model: Any,
        candidate: str,
    ) -> None:
        self.fixed_model = fixed_model
        self.source_layer = source_model.model.layers[LAYER_INDEX]
        self.fixed_layer = fixed_model.model.layers[LAYER_INDEX]
        self.candidate = candidate
        self.prompt_name = ""
        self.generation_index = -1
        self.records: list[dict[str, Any]] = []
        self.handles = []
        self.current: dict[str, Tensor] | None = None

    def install(self) -> None:
        def layer_pre_hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            del module
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                if not args:
                    raise RuntimeError("Layer-21 hook lacks hidden states")
                hidden_states = args[0]
            source_kwargs = dict(kwargs)
            source_kwargs.pop("hidden_states", None)
            captured: dict[str, Tensor] = {}

            def capture_exact_attention_residual(
                _module: Any,
                norm_args: tuple[Any, ...],
            ) -> None:
                if not norm_args:
                    raise RuntimeError("source post-attention norm lacks input")
                captured["exact_attention_residual"] = norm_args[0].detach().clone()

            handle = self.source_layer.post_attention_layernorm.register_forward_pre_hook(
                capture_exact_attention_residual
            )
            try:
                exact_output = self.source_layer(hidden_states, **source_kwargs)
            finally:
                handle.remove()
            if not isinstance(exact_output, Tensor):
                raise TypeError("source Layer-21 output must be a tensor")
            if "exact_attention_residual" not in captured:
                raise RuntimeError("source Layer-21 attention residual was not captured")
            self.current = {
                "layer_input": hidden_states.detach().clone(),
                "exact_attention_residual": captured["exact_attention_residual"],
                "exact_layer_output": exact_output.detach().clone(),
            }

        def post_attention_norm_pre_hook(
            module: Any,
            args: tuple[Any, ...],
        ) -> tuple[Tensor]:
            del module
            if self.current is None or not args:
                raise RuntimeError("fixed Layer-21 attention boundary lacks context")
            fixed_attention_residual = args[0]
            self.current["fixed_attention_residual"] = (
                fixed_attention_residual.detach().clone()
            )
            if self.candidate == "exact_attention_residual":
                return (self.current["exact_attention_residual"],)
            return (fixed_attention_residual,)

        def layer_hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            fixed_output: Tensor,
        ) -> Tensor:
            del module, args, kwargs
            if self.current is None:
                raise RuntimeError("fixed Layer-21 output lacks discriminator context")
            if not isinstance(fixed_output, Tensor):
                raise TypeError("fixed Layer-21 output must be a tensor")
            fixed_attention_residual = self.current["fixed_attention_residual"]
            exact_attention_residual = self.current["exact_attention_residual"]
            exact_layer_output = self.current["exact_layer_output"]

            if self.candidate == "exact_mlp_residual":
                normalized = self.source_layer.post_attention_layernorm(
                    fixed_attention_residual
                )
                exact_mlp = self.source_layer.mlp(normalized)
                candidate_output = fixed_attention_residual + exact_mlp
                fixed_shadow_output = fixed_output
            else:
                candidate_output = fixed_output
                fixed_shadow_output = None

            attention_q = fixed_point.quantize_int8(
                exact_attention_residual,
                float(self.fixed_layer.post_attention_scale),
            )
            next_scale = float(
                self.fixed_model.model.layers[LAYER_INDEX + 1].input_scale
            )
            next_q = fixed_point.quantize_int8(candidate_output, next_scale)
            record = {
                "prompt": self.prompt_name,
                "generation_index": self.generation_index,
                "candidate": self.candidate,
                "layer_index": LAYER_INDEX,
                "elements": candidate_output.numel(),
                "fixed_attention_residual_vs_exact_relative_l2": relative_l2(
                    exact_attention_residual, fixed_attention_residual
                ),
                "candidate_attention_residual_vs_exact_relative_l2": (
                    0.0
                    if self.candidate == "exact_attention_residual"
                    else relative_l2(
                        exact_attention_residual, fixed_attention_residual
                    )
                ),
                "candidate_layer_output_vs_exact_relative_l2": relative_l2(
                    exact_layer_output, candidate_output
                ),
                "fixed_shadow_layer_output_vs_exact_relative_l2": (
                    relative_l2(exact_layer_output, fixed_shadow_output)
                    if fixed_shadow_output is not None
                    else None
                ),
                "exact_attention_residual_quantized_for_fixed_mlp": {
                    "scale": float(self.fixed_layer.post_attention_scale),
                    "relative_l2_error": relative_l2(
                        exact_attention_residual,
                        attention_q.to(exact_attention_residual.dtype)
                        * float(self.fixed_layer.post_attention_scale),
                    ),
                    "saturation": saturation(attention_q),
                },
                "candidate_output_quantized_for_layer22": {
                    "scale": next_scale,
                    "relative_l2_error": relative_l2(
                        candidate_output,
                        next_q.to(candidate_output.dtype) * next_scale,
                    ),
                    "saturation": saturation(next_q),
                },
            }
            self.records.append(record)
            self.current = None
            return candidate_output

        self.handles.extend(
            [
                self.fixed_layer.register_forward_pre_hook(
                    layer_pre_hook, with_kwargs=True
                ),
                self.fixed_layer.post_attention_layernorm.register_forward_pre_hook(
                    post_attention_norm_pre_hook
                ),
                self.fixed_layer.register_forward_hook(
                    layer_hook, with_kwargs=True
                ),
            ]
        )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def set_step(self, prompt_name: str, generation_index: int) -> None:
        self.prompt_name = prompt_name
        self.generation_index = generation_index


def greedy_generate_with_fast_fail(
    model: Any,
    tokenizer: Any,
    prompt_name: str,
    prompt_tokens: list[int],
    termination_token_ids: set[int],
    *,
    max_new_tokens: int,
    top_k_count: int,
    prefix: PrefixSubstitution,
    discriminator: Layer21BundleDiscriminator,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    generated: list[int] = []
    steps = []
    fast_fail = None
    with torch.inference_mode():
        for generation_index in range(max_new_tokens):
            prefix.set_step(prompt_name, generation_index)
            discriminator.set_step(prompt_name, generation_index)
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
            candidates = top_k(tokenizer, logits, top_k_count)
            selected = int(candidates[0]["token_id"])
            piece = token_piece(tokenizer, selected)
            generated.append(selected)
            terminated = selected in termination_token_ids
            steps.append(
                {
                    "generation_index": generation_index,
                    "selected_token_id": selected,
                    "selected_piece": piece,
                    "terminated": terminated,
                    "top_k": candidates,
                }
            )
            if generation_index == 0 and not any(
                character.isalpha() for character in piece
            ):
                fast_fail = {
                    "triggered": True,
                    "reason": "first selected token contains no alphabetic character",
                    "selected_token_id": selected,
                    "selected_piece": piece,
                }
                break
            input_ids = torch.cat(
                [input_ids, torch.tensor([[selected]], dtype=torch.long)], dim=1
            )
            if terminated:
                break
    decoded = decode_generated(tokenizer, generated)
    readability = readability_gate(decoded, generated)
    return {
        "prompt_token_count": len(prompt_tokens),
        "prompt_tokens_sha256": sha256_bytes(
            torch.tensor(prompt_tokens, dtype=torch.int32).numpy().tobytes()
        ),
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "readability_gate": readability,
        "coherence_gate": prompt_coherence(prompt_name, decoded, readability),
        "fast_fail": fast_fail or {"triggered": False},
        "terminated": bool(steps and steps[-1]["terminated"]),
        "steps": steps,
        "wall_seconds": time.perf_counter() - started,
    }


def aggregate_boundary(
    records: list[dict[str, Any]], prompt_name: str
) -> dict[str, Any]:
    selected = [record for record in records if record["prompt"] == prompt_name]
    if not selected:
        raise RuntimeError(f"Layer-21 discriminator produced no records for {prompt_name}")
    return {
        "calls": len(selected),
        "first_generation_step": selected[0],
        "fixed_attention_residual_vs_exact_relative_l2_max": max(
            record["fixed_attention_residual_vs_exact_relative_l2"]
            for record in selected
        ),
        "candidate_layer_output_vs_exact_relative_l2_max": max(
            record["candidate_layer_output_vs_exact_relative_l2"]
            for record in selected
        ),
        "fixed_shadow_layer_output_vs_exact_relative_l2_max": (
            max(
                record["fixed_shadow_layer_output_vs_exact_relative_l2"]
                for record in selected
            )
            if selected[0]["fixed_shadow_layer_output_vs_exact_relative_l2"]
            is not None
            else None
        ),
        "exact_attention_residual_quantization": {
            "relative_l2_error_max": max(
                record["exact_attention_residual_quantized_for_fixed_mlp"][
                    "relative_l2_error"
                ]
                for record in selected
            ),
            "positive_saturations": sum(
                record["exact_attention_residual_quantized_for_fixed_mlp"][
                    "saturation"
                ]["positive"]
                for record in selected
            ),
            "negative_saturations": sum(
                record["exact_attention_residual_quantized_for_fixed_mlp"][
                    "saturation"
                ]["negative"]
                for record in selected
            ),
        },
        "layer22_input_quantization": {
            "relative_l2_error_max": max(
                record["candidate_output_quantized_for_layer22"][
                    "relative_l2_error"
                ]
                for record in selected
            ),
            "positive_saturations": sum(
                record["candidate_output_quantized_for_layer22"]["saturation"][
                    "positive"
                ]
                for record in selected
            ),
            "negative_saturations": sum(
                record["candidate_output_quantized_for_layer22"]["saturation"][
                    "negative"
                ]
                for record in selected
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=("exact_attention_residual", "exact_mlp_residual"),
        required=True,
    )
    parser.add_argument("--second-prompt", default="Write one word: blue")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    started = time.perf_counter()
    scales = json.loads(FROZEN_SCALES.read_text(encoding="utf-8"))
    ranges, operator_ranges = frozen_ranges(scales)
    tokenizer = load_tokenizer()
    tokenizations = {
        "hello_world": tokenize_prompt(tokenizer, "Hello world", ""),
        "second_prompt": tokenize_prompt(tokenizer, args.second_prompt, ""),
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
    runtime = fixed_model.ace2_all_projection_residual_fusion_runtime
    prefix = PrefixSubstitution(fixed_model, source_model, LAYER_INDEX)
    discriminator = Layer21BundleDiscriminator(
        fixed_model, source_model, args.candidate
    )
    prefix.install()
    discriminator.install()

    termination_token_ids = {
        int(tokenizer.eos_token_id),
        int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
    }
    generations = {}
    try:
        for prompt_name, tokenization in tokenizations.items():
            generations[prompt_name] = greedy_generate_with_fast_fail(
                fixed_model,
                tokenizer,
                prompt_name,
                tokenization["chat_template_token_ids"],
                termination_token_ids,
                max_new_tokens=args.max_new_tokens,
                top_k_count=args.top_k,
                prefix=prefix,
                discriminator=discriminator,
            )
            generation = generations[prompt_name]
            print(
                "ACE2_LAYER21_BUNDLE_PROMPT "
                f"candidate={args.candidate} prompt={prompt_name} "
                f"first={generation['generated_token_ids'][0]} "
                f"decoded={generation['decoded_text']!r} "
                f"readable={generation['readability_gate']['passed']} "
                f"semantic={generation['coherence_gate']['passed']} "
                f"fast_fail={generation['fast_fail']['triggered']}",
                flush=True,
            )
    finally:
        discriminator.remove()
        prefix.remove()

    boundary = {
        prompt_name: aggregate_boundary(discriminator.records, prompt_name)
        for prompt_name in generations
    }
    execution = summarize_execution(runtime)
    readable = all(
        generation["readability_gate"]["passed"]
        for generation in generations.values()
    )
    semantic = all(
        generation["coherence_gate"]["passed"]
        for generation in generations.values()
    )
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "chat_v2_reference_only_layer21_bundle_discriminator",
        "contract": {
            "acceptance_candidate": False,
            "diagnostic_bf16_substitution": True,
            "conditioned_exact_prefix_layers": 21,
            "candidate": args.candidate,
            "layer_index": LAYER_INDEX,
            "fast_fail_rule": (
                "Stop a prompt after generation step 0 when the selected token piece "
                "contains no alphabetic character."
            ),
            "readability_and_semantic_gates_are_independent": True,
            "semantics": (
                "All fixed layers execute. Layers 0 through 20 are replaced at their "
                "complete output boundary. Layer 21 injects either the exact attention "
                "residual before the fixed MLP or the exact MLP residual conditioned on "
                "the fixed attention residual. Layers 22 and 23 remain frozen W4A8."
            ),
        },
        "invocation": {"argv": exact_process_argv(), "cwd": str(Path.cwd())},
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": REVISION,
            "snapshot": str(TOKENIZER_SNAPSHOT.resolve()),
        },
        "prompts": {
            "hello_world": "Hello world",
            "second_prompt": args.second_prompt,
        },
        "generations": generations,
        "boundary": boundary,
        "scale_authentication": scale_binding,
        "shadow_residual_execution": execution,
        "two_prompt_readability_passed": readable,
        "two_prompt_semantic_gate_passed": semantic,
        "status": (
            "PASS_LAYER21_BUNDLE_TWO_PROMPT_READABILITY"
            if readable
            else "FAIL_LAYER21_BUNDLE_TWO_PROMPT_READABILITY"
        ),
        "semantic_status": (
            "PASS_LAYER21_BUNDLE_TWO_PROMPT_SEMANTIC_GATE"
            if semantic
            else "FAIL_LAYER21_BUNDLE_TWO_PROMPT_SEMANTIC_GATE"
        ),
        "artifacts": {
            "prefix21": {"path": str(PREFIX21), "sha256": sha256_file(PREFIX21)},
            "prefix22": {"path": str(PREFIX22), "sha256": sha256_file(PREFIX22)},
            "frozen_scales": {
                "path": str(FROZEN_SCALES),
                "sha256": sha256_file(FROZEN_SCALES),
            },
            "fixed_point_source": {
                "path": str(ROOT / "tools/ace2_full_model_fixed_point.py"),
                "sha256": sha256_file(ROOT / "tools/ace2_full_model_fixed_point.py"),
            },
            "runner_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "wall_seconds": time.perf_counter() - started,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, canonical_bytes(result))
    print(
        "ACE2_LAYER21_BUNDLE_RESULT "
        f"status={result['status']} semantic_status={result['semantic_status']} "
        f"candidate={args.candidate} wall_seconds={result['wall_seconds']:.6f} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_LAYER21_BUNDLE_FAIL detail={error}", file=sys.stderr)
        raise
