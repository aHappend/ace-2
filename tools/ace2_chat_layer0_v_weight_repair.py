#!/usr/bin/env python3
"""Run one reference-only layer-0 V-weight upper-bound repair on two prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
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


QUALITY_CONFIG = ROOT / "benchmark" / "quality" / "QUALITY_CONFIG.json"
SPEC = ROOT / "design" / "SPEC.md"
CHIP_SCOPE = ROOT / "design" / "CHIP_SCOPE.json"
FIXED_POINT_SOURCE = ROOT / "tools" / "ace2_full_model_fixed_point.py"
ALL_RESIDUAL_SOURCE = ROOT / "tools" / "ace2_chat_all_residual_reference.py"
READABILITY_SOURCE = ROOT / "tools" / "ace2_chat_reference_repairs.py"


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "bytes": resolved.stat().st_size,
        "path": resolved.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(resolved),
    }


def source_binding(paths: list[Path]) -> dict[str, Any]:
    sources = [artifact(path) for path in paths]
    digest, count = fixed_point.hash_records(
        f"{item['path']}\0{item['bytes']}\0{item['sha256']}" for item in sources
    )
    return {
        "ordered_source_hash_list_sha256": digest,
        "source_count": count,
        "sources": sources,
        "status": "bound_to_current_reference_sources",
    }


def relative_l2(reference: Tensor, candidate: Tensor) -> float:
    reference64 = reference.to(torch.float64)
    candidate64 = candidate.to(torch.float64)
    denominator = torch.linalg.vector_norm(reference64)
    numerator = torch.linalg.vector_norm(candidate64 - reference64)
    return float(numerator / denominator) if float(denominator) else float(numerator)


def raw_tensor_sha256(value: Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def token_delta(baseline: list[int], repaired: list[int]) -> dict[str, Any]:
    first_difference = next(
        (
            index
            for index in range(max(len(baseline), len(repaired)))
            if index >= len(baseline)
            or index >= len(repaired)
            or baseline[index] != repaired[index]
        ),
        None,
    )
    return {
        "baseline_generated_token_ids": baseline,
        "repaired_generated_token_ids": repaired,
        "changed": first_difference is not None,
        "first_difference_generation_index": first_difference,
    }


def install_layer0_exact_source_v_weight(
    model: Any,
    source_weight: Tensor,
    source_bias: Tensor | None,
) -> tuple[fixed_point.W4A8Linear, dict[str, int]]:
    projection = model.model.layers[0].self_attn.v_proj
    if not isinstance(projection, fixed_point.W4A8Linear):
        raise TypeError("layer-0 V projection is not W4A8Linear")
    if tuple(source_weight.shape) != (projection.out_features, projection.in_features):
        raise RuntimeError("layer-0 source V weight shape differs after replacement")

    stats = {
        "calls": 0,
        "input_elements": 0,
        "output_elements": 0,
        "positive_saturations": 0,
        "negative_saturations": 0,
    }
    exact_weight = source_weight.detach().to(torch.float64).contiguous()
    exact_bias = (
        None
        if source_bias is None
        else source_bias.detach().to(torch.float64).contiguous()
    )

    def exact_source_forward(self: fixed_point.W4A8Linear, inputs: Tensor) -> Tensor:
        if inputs.dtype == torch.int8:
            qinput = inputs
        elif inputs.is_floating_point():
            if not torch.all(torch.isfinite(inputs)):
                raise ValueError("layer-0 V repair input must be finite")
            if torch.any(inputs < -128) or torch.any(inputs > 127):
                raise ValueError("layer-0 V repair input is outside signed int8")
            if not torch.equal(inputs, torch.round(inputs)):
                raise ValueError("layer-0 V repair input must contain exact integers")
            qinput = inputs.to(torch.int8)
        else:
            raise TypeError("layer-0 V repair input must be signed int8 or exact float")

        real_input = qinput.to(torch.float64) * self.hardware_input_scale
        output = torch.matmul(real_input, exact_weight.transpose(0, 1))
        if exact_bias is not None:
            output = output + exact_bias
        pre_round = output / self.output_scale_per_channel
        rounded = torch.round(pre_round)
        stats["calls"] += 1
        stats["input_elements"] += qinput.numel()
        stats["output_elements"] += rounded.numel()
        stats["positive_saturations"] += int((rounded > 127).sum())
        stats["negative_saturations"] += int((rounded < -128).sum())
        return rounded.clamp(-128, 127).to(torch.int8)

    projection.forward_hardware_input = MethodType(exact_source_forward, projection)
    return projection, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--second-prompt", default="Write one word: blue")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 32:
        raise RuntimeError("--max-new-tokens must be in 1..32")
    if not 1 <= args.top_k <= 64:
        raise RuntimeError("--top-k must be in 1..64")

    started = time.perf_counter()
    baseline_path = args.baseline.resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected_prompts = {
        "hello_world": "Hello world",
        "second_prompt": args.second_prompt,
    }
    if baseline.get("prompts") != expected_prompts:
        raise RuntimeError("baseline prompts differ from the requested two-prompt repair")
    if baseline.get("model", {}).get("revision") != REVISION:
        raise RuntimeError("baseline model revision differs from the frozen Base revision")

    scales = json.loads(FROZEN_SCALES.read_text(encoding="utf-8"))
    ranges, operator_ranges = frozen_ranges(scales)
    tokenizer = load_tokenizer()
    tokenizations = {
        name: tokenize_prompt(tokenizer, prompt, "")
        for name, prompt in expected_prompts.items()
    }
    model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    source_v = model.model.layers[0].self_attn.v_proj
    if not isinstance(source_v, nn.Linear):
        raise TypeError("source layer-0 V projection is not nn.Linear")
    source_weight = source_v.weight.detach().clone()
    source_bias = None if source_v.bias is None else source_v.bias.detach().clone()
    source_weight_sha256 = raw_tensor_sha256(source_weight)
    source_bias_sha256 = (
        None if source_bias is None else raw_tensor_sha256(source_bias)
    )

    fixed_point.replace_linears(
        model,
        ranges,
        rope_diagnostic_mechanism=ROPE_MECHANISM,
    )
    fixed_point.replace_fixed_operators(
        model,
        operator_ranges,
        rope_diagnostic_mechanism=ROPE_MECHANISM,
        all_projection_residual_fusion=True,
    )
    scale_binding = validate_frozen_scale_binding(model, scales)
    projection = model.model.layers[0].self_attn.v_proj
    if not isinstance(projection, fixed_point.W4A8Linear):
        raise TypeError("fixed layer-0 V projection is not W4A8Linear")
    baseline_dequantized_weight = (
        projection.qweight.to(torch.float64) * projection.weight_scale[:, None]
    )
    approximation = {
        "baseline_qweight_sha256": fixed_point.sha256_tensor(projection.qweight),
        "baseline_weight_scale_sha256": fixed_point.sha256_tensor(
            projection.weight_scale
        ),
        "baseline_w4_relative_l2_vs_source": relative_l2(
            source_weight,
            baseline_dequantized_weight,
        ),
        "source_bias_present": source_bias is not None,
        "source_bias_sha256": source_bias_sha256,
        "source_weight_sha256": source_weight_sha256,
    }
    repaired_projection, repair_stats = install_layer0_exact_source_v_weight(
        model,
        source_weight,
        source_bias,
    )
    if repaired_projection is not projection:
        raise AssertionError("layer-0 V repair replaced the metadata-bearing module")

    runtime = model.ace2_all_projection_residual_fusion_runtime
    if not isinstance(runtime, fixed_point.AllProjectionResidualFusionRuntime):
        raise RuntimeError("all-projection residual fusion runtime was not installed")
    termination_token_ids = {
        int(tokenizer.eos_token_id),
        int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
    }
    generations = {}
    for prompt_name, tokenization in tokenizations.items():
        generations[prompt_name] = greedy_generate(
            model,
            tokenizer,
            prompt_name,
            tokenization["chat_template_token_ids"],
            termination_token_ids,
            max_new_tokens=args.max_new_tokens,
            top_k_count=args.top_k,
        )
        print(
            "ACE2_LAYER0_V_REPAIR_PROMPT "
            f"prompt={prompt_name} "
            f"decoded={generations[prompt_name]['decoded_text']!r} "
            f"coherent={generations[prompt_name]['coherence_gate']['passed']}",
            flush=True,
        )

    execution = summarize_execution(runtime)
    coherent = all(
        generation["coherence_gate"]["passed"]
        for generation in generations.values()
    )
    deltas = {
        prompt_name: token_delta(
            baseline["generations"][prompt_name]["generated_token_ids"],
            generation["generated_token_ids"],
        )
        for prompt_name, generation in generations.items()
    }
    source_paths = [
        QUALITY_CONFIG,
        SPEC,
        CHIP_SCOPE,
        FIXED_POINT_SOURCE,
        ALL_RESIDUAL_SOURCE,
        READABILITY_SOURCE,
        Path(__file__),
    ]
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "chat_v2_reference_only_layer0_exact_source_v_weight_upper_bound",
        "contract": {
            "in_contract_w4_repair": False,
            "packing_or_image_changed": False,
            "preserved_boundaries": [
                "frozen signed-int8 layer-0 RMSNorm/V input",
                "frozen layer-0 V output scale",
                "ties-to-even signed-int8 V output requantization",
                "all non-layer-0-V W4A8 operators and residual joins",
            ],
            "purpose": (
                "upper-bound causal counterfactual for the localized layer-0 V weight "
                "approximation; it is not accelerator completion or a packable W4 image"
            ),
            "resolved_w4_scale_granularity": (
                "one symmetric scale per output channel across the complete projection "
                "reduction; 128-element blocks are packing/tiled-MAC boundaries"
            ),
        },
        "invocation": {
            "argv": exact_process_argv(),
            "cwd": str(Path.cwd().resolve()),
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": REVISION,
            "snapshot": str(TOKENIZER_SNAPSHOT.resolve()),
            "dtype_at_load": "bfloat16",
        },
        "prompts": expected_prompts,
        "baseline": artifact(baseline_path),
        "layer0_v_weight_repair": {
            "approximation_removed": approximation,
            "execution": repair_stats,
        },
        "generations": generations,
        "token_deltas_vs_baseline": deltas,
        "scale_authentication_before_repair": scale_binding,
        "residual_execution": execution,
        "bindings": {
            "diagnostic_rtl": {
                "acceptance_claimed": False,
                "rtl_hash": None,
                "status": "not_applicable_reference_only_diagnostic",
            },
            "reference_sources": source_binding(source_paths),
            "frozen_scales": artifact(FROZEN_SCALES),
        },
        "two_prompt_coherence_passed": coherent,
        "status": (
            "PASS_LAYER0_V_WEIGHT_UPPER_BOUND_TWO_PROMPT_COHERENCE"
            if coherent
            else "FAIL_LAYER0_V_WEIGHT_UPPER_BOUND_TWO_PROMPT_COHERENCE"
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output_path, canonical_bytes(result))
    print(
        "ACE2_LAYER0_V_WEIGHT_REPAIR_RESULT "
        f"status={result['status']} "
        f"hello_changed={deltas['hello_world']['changed']} "
        f"second_changed={deltas['second_prompt']['changed']} "
        f"passes={execution['completed_passes']} "
        f"wall_seconds={result['wall_seconds']:.6f} "
        f"output={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_LAYER0_V_WEIGHT_REPAIR_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
