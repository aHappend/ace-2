#!/usr/bin/env python3
"""Reference-only exact BF16 layer-prefix substitution diagnostics."""

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
    greedy_generate,
    summarize_execution,
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


BASELINE = (
    ROOT
    / "build/ace2_chat_diagnostics/all-residual-scale32-two-prompts-20260805.json"
)


def relative_l2(reference: Tensor, candidate: Tensor) -> float:
    reference64 = reference.detach().to(torch.float64)
    candidate64 = candidate.detach().to(torch.float64)
    denominator = torch.linalg.vector_norm(reference64)
    numerator = torch.linalg.vector_norm(candidate64 - reference64)
    return float(numerator / denominator) if float(denominator) else float(numerator)


def saturation(value: Tensor) -> dict[str, int]:
    return {
        "elements": value.numel(),
        "negative": int((value == -128).sum()),
        "positive": int((value == 127).sum()),
    }


class PrefixSubstitution:
    def __init__(
        self,
        fixed_model: Any,
        source_model: Any,
        prefix_layers: int,
    ) -> None:
        self.fixed_model = fixed_model
        self.source_model = source_model
        self.prefix_layers = prefix_layers
        self.prompt_name = ""
        self.generation_index = -1
        self.records: list[dict[str, Any]] = []
        self.handles = []

    def install(self) -> None:
        for layer_index in range(self.prefix_layers):
            fixed_layer = self.fixed_model.model.layers[layer_index]
            source_layer = self.source_model.model.layers[layer_index]

            def hook(
                module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                fixed_output: Tensor,
                *,
                layer_index: int = layer_index,
                source_layer: Any = source_layer,
            ) -> Tensor:
                del module
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is None:
                    if not args:
                        raise RuntimeError("decoder-layer hook lacks hidden states")
                    hidden_states = args[0]
                source_kwargs = dict(kwargs)
                source_kwargs.pop("hidden_states", None)
                exact_output = source_layer(hidden_states, **source_kwargs)
                if not isinstance(fixed_output, Tensor) or not isinstance(
                    exact_output, Tensor
                ):
                    raise TypeError("decoder layers must return tensors")

                next_layer = layer_index + 1
                consumer = None
                if next_layer < len(self.fixed_model.model.layers):
                    consumer_scale = float(
                        self.fixed_model.model.layers[next_layer].input_scale
                    )
                    quantized = fixed_point.quantize_int8(
                        exact_output, consumer_scale
                    )
                    dequantized = quantized.to(exact_output.dtype) * consumer_scale
                    consumer = {
                        "layer_index": next_layer,
                        "input_scale": consumer_scale,
                        "relative_l2_error": relative_l2(
                            exact_output, dequantized
                        ),
                        "saturation": saturation(quantized),
                    }

                self.records.append(
                    {
                        "prompt": self.prompt_name,
                        "generation_index": self.generation_index,
                        "layer_index": layer_index,
                        "elements": exact_output.numel(),
                        "fixed_shadow_vs_exact_relative_l2": relative_l2(
                            exact_output, fixed_output
                        ),
                        "injected_vs_exact_relative_l2": relative_l2(
                            exact_output, exact_output
                        ),
                        "next_w4a8_consumer_quantization": consumer,
                    }
                )
                return exact_output

            self.handles.append(
                fixed_layer.register_forward_hook(hook, with_kwargs=True)
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def set_step(self, prompt_name: str, generation_index: int) -> None:
        self.prompt_name = prompt_name
        self.generation_index = generation_index


def greedy_generate_with_context(
    model: Any,
    tokenizer: Any,
    prompt_name: str,
    prompt_tokens: list[int],
    termination_token_ids: set[int],
    *,
    max_new_tokens: int,
    top_k_count: int,
    substitution: PrefixSubstitution,
) -> dict[str, Any]:
    original_forward = model.forward
    call_index = 0

    def contextual_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_index
        substitution.set_step(prompt_name, call_index)
        result = original_forward(*args, **kwargs)
        call_index += 1
        return result

    model.forward = contextual_forward
    try:
        result = greedy_generate(
            model,
            tokenizer,
            prompt_name,
            prompt_tokens,
            termination_token_ids,
            max_new_tokens=max_new_tokens,
            top_k_count=top_k_count,
        )
    finally:
        model.forward = original_forward
    return result


def aggregate_boundary(
    records: list[dict[str, Any]], prompt_name: str, layer_index: int
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if record["prompt"] == prompt_name
        and record["layer_index"] == layer_index
    ]
    if not selected:
        raise RuntimeError("substitution boundary produced no records")
    consumer_records = [
        record["next_w4a8_consumer_quantization"]
        for record in selected
        if record["next_w4a8_consumer_quantization"] is not None
    ]
    return {
        "calls": len(selected),
        "first_generation_step": selected[0],
        "fixed_shadow_vs_exact_relative_l2_max": max(
            record["fixed_shadow_vs_exact_relative_l2"] for record in selected
        ),
        "injected_vs_exact_relative_l2_max": max(
            record["injected_vs_exact_relative_l2"] for record in selected
        ),
        "next_w4a8_consumer": (
            {
                "relative_l2_error_max": max(
                    record["relative_l2_error"] for record in consumer_records
                ),
                "positive_saturations": sum(
                    record["saturation"]["positive"]
                    for record in consumer_records
                ),
                "negative_saturations": sum(
                    record["saturation"]["negative"]
                    for record in consumer_records
                ),
                "elements": sum(
                    record["saturation"]["elements"]
                    for record in consumer_records
                ),
            }
            if consumer_records
            else None
        ),
    }


def token_delta(baseline: list[int], candidate: list[int]) -> dict[str, Any]:
    first = next(
        (
            index
            for index in range(max(len(baseline), len(candidate)))
            if index >= len(baseline)
            or index >= len(candidate)
            or baseline[index] != candidate[index]
        ),
        None,
    )
    return {
        "changed": first is not None,
        "first_difference_generation_index": first,
        "baseline": baseline,
        "candidate": candidate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-layers", type=int, default=1)
    parser.add_argument("--second-prompt", default="Write one word: blue")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.prefix_layers <= 24:
        raise RuntimeError("--prefix-layers must be in 1..24")

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
    substitution = PrefixSubstitution(
        fixed_model, source_model, args.prefix_layers
    )
    substitution.install()

    termination_token_ids = {
        int(tokenizer.eos_token_id),
        int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
    }
    generations = {}
    try:
        for prompt_name, tokenization in tokenizations.items():
            generations[prompt_name] = greedy_generate_with_context(
                fixed_model,
                tokenizer,
                prompt_name,
                tokenization["chat_template_token_ids"],
                termination_token_ids,
                max_new_tokens=args.max_new_tokens,
                top_k_count=args.top_k,
                substitution=substitution,
            )
            print(
                "ACE2_PREFIX_SUBSTITUTION_PROMPT "
                f"prefix={args.prefix_layers} prompt={prompt_name} "
                f"first={generations[prompt_name]['generated_token_ids'][0]} "
                f"decoded={generations[prompt_name]['decoded_text']!r} "
                f"coherent={generations[prompt_name]['coherence_gate']['passed']}",
                flush=True,
            )
    finally:
        substitution.remove()

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    boundary_layer = args.prefix_layers - 1
    boundary = {
        prompt_name: aggregate_boundary(
            substitution.records, prompt_name, boundary_layer
        )
        for prompt_name in generations
    }
    execution = summarize_execution(runtime)
    token_deltas = {
        prompt_name: token_delta(
            baseline["generations"][prompt_name]["generated_token_ids"],
            generation["generated_token_ids"],
        )
        for prompt_name, generation in generations.items()
    }
    coherent = all(
        generation["coherence_gate"]["passed"]
        for generation in generations.values()
    )
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "chat_v2_reference_only_exact_bf16_layer_prefix_substitution",
        "contract": {
            "acceptance_candidate": False,
            "diagnostic_bf16_substitution": True,
            "w4a8_downstream_preserved": True,
            "substituted_prefix_layers": args.prefix_layers,
            "boundary_layer": boundary_layer,
            "semantics": (
                "Each frozen W4A8 layer still executes as a shadow, then its complete "
                "output is replaced by the exact source BF16 layer output before the "
                "next layer. All layers after the prefix remain frozen W4A8."
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
        "token_deltas_vs_frozen_baseline": token_deltas,
        "scale_authentication": scale_binding,
        "shadow_residual_execution": execution,
        "artifacts": {
            "baseline": {"path": str(BASELINE), "sha256": sha256_file(BASELINE)},
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
        "two_prompt_coherence_passed": coherent,
        "status": (
            "PASS_EXACT_PREFIX_TWO_PROMPT_COHERENCE"
            if coherent
            else "FAIL_EXACT_PREFIX_TWO_PROMPT_COHERENCE"
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, canonical_bytes(result))
    print(
        "ACE2_PREFIX_SUBSTITUTION_RESULT "
        f"status={result['status']} prefix={args.prefix_layers} "
        f"wall_seconds={result['wall_seconds']:.6f} output={output}",
        flush=True,
    )
    return 0 if coherent else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_PREFIX_SUBSTITUTION_FAIL detail={error}", file=sys.stderr)
        raise
