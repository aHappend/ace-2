#!/usr/bin/env python3
"""Localize frozen layer-0 RMSNorm, Q/K projection, and RoPE divergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from ace2_full_model_fixed_point import (
    FixedAttention,
    FixedRMSNorm,
    PROMPT_MANIFEST,
    QUALITY_CONFIG,
    ROOT,
    calibrate,
    fixed_rope_raw_with_saturation,
    hash_records,
    hash_token_sequences,
    load_contracts,
    replace_fixed_operators,
    replace_linears,
    seed_everything,
    selected_texts,
    sha256_file,
    tokenize_prompts,
    utc_now,
    validate_runtime,
)
from localize_quality_divergence import compare_tensor, write_sha256s


def capture_layer0_inputs(
    model: nn.Module,
    input_ids: Tensor,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    captured: dict[str, Any] = {}
    layer = model.model.layers[0]

    def capture_norm(
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: Tensor,
    ) -> None:
        captured["norm"] = output.detach()

    def capture_attention(
        _module: nn.Module,
        _args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        position_embeddings = kwargs.get("position_embeddings")
        if (
            not isinstance(position_embeddings, tuple)
            or len(position_embeddings) != 2
            or not all(isinstance(value, Tensor) for value in position_embeddings)
        ):
            raise RuntimeError("layer-0 attention did not receive rotary position embeddings")
        captured["position_embeddings"] = tuple(
            value.detach() for value in position_embeddings
        )

    hooks = [
        layer.input_layernorm.register_forward_hook(capture_norm),
        layer.self_attn.register_forward_pre_hook(capture_attention, with_kwargs=True),
    ]
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    if set(captured) != {"norm", "position_embeddings"}:
        raise RuntimeError(f"incomplete layer-0 capture: {sorted(captured)}")
    return captured["norm"], captured["position_embeddings"]


@torch.inference_mode()
def baseline_boundaries(
    attention: nn.Module,
    norm_output: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
) -> dict[str, Tensor]:
    input_shape = norm_output.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query = attention.q_proj(norm_output).view(hidden_shape).transpose(1, 2)
    key = attention.k_proj(norm_output).view(hidden_shape).transpose(1, 2)
    query_rope, key_rope = apply_rotary_pos_emb(
        query,
        key,
        *position_embeddings,
    )
    return {
        "layer0.input_rmsnorm": norm_output,
        "layer0.q_projection": query,
        "layer0.k_projection": key,
        "layer0.q_post_rope": query_rope,
        "layer0.k_post_rope": key_rope,
    }


@torch.inference_mode()
def fixed_boundaries(
    attention: FixedAttention,
    norm: FixedRMSNorm,
    norm_output: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    input_shape = norm_output.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query = (
        attention.q_proj.forward_hardware_input(norm_output)
        .view(hidden_shape)
        .transpose(1, 2)
    )
    key = (
        attention.k_proj.forward_hardware_input(norm_output)
        .view(hidden_shape)
        .transpose(1, 2)
    )
    cos, sin = position_embeddings
    binding = attention.scale_binding
    query_rope, query_saturations = fixed_rope_raw_with_saturation(
        query,
        binding.conversion_q9,
        cos,
        sin,
    )
    key_rope, key_saturations = fixed_rope_raw_with_saturation(
        key,
        binding.conversion_q9,
        cos,
        sin,
    )
    boundaries = {
        "layer0.input_rmsnorm": norm_output.to(torch.float64) * norm.output_scale,
        "layer0.q_projection": (
            query.to(torch.float64) * attention.q_proj.output_scale
        ),
        "layer0.k_projection": key.to(torch.float64) * attention.k_proj.output_scale,
        "layer0.q_post_rope": query_rope.to(torch.float64)
        * binding.query_output_scale,
        "layer0.k_post_rope": key_rope.to(torch.float64) * binding.key_output_scale,
    }
    metadata = {
        "input_rmsnorm_output_scale": norm.output_scale,
        "q_projection": {
            "hardware_input_scale": attention.q_proj.hardware_input_scale,
            "output_scale": attention.q_proj.output_scale,
            "raw_absmax": int(query.to(torch.int64).abs().amax()),
        },
        "k_projection": {
            "hardware_input_scale": attention.k_proj.hardware_input_scale,
            "output_scale": attention.k_proj.output_scale,
            "raw_absmax": int(key.to(torch.int64).abs().amax()),
        },
        "rope": {
            "conversion_q9": binding.conversion_q9,
            "conversion_scale_realized": binding.conversion_q9 / float(1 << 9),
            "conversion_scale_requested": binding.conversion_scale,
            "query_output_scale": binding.query_output_scale,
            "key_output_scale": binding.key_output_scale,
            "output_scale_product": (
                binding.query_output_scale * binding.key_output_scale
            ),
            "score_multiplier": int(attention.score_multiplier.item()),
            "score_right_shift": int(attention.score_right_shift.item()),
            "query_saturation": {
                "elements": query.numel(),
                "fraction": query_saturations / query.numel(),
                "saturated_elements": query_saturations,
            },
            "key_saturation": {
                "elements": key.numel(),
                "fraction": key_saturations / key.numel(),
                "saturated_elements": key_saturations,
            },
        },
    }
    return boundaries, metadata


def run(output_dir: Path) -> dict[str, Any]:
    manifest, config, rtl_binding = load_contracts(require_rtl_binding=False)
    versions = validate_runtime(config)
    seed_everything(config)
    output_dir.mkdir(parents=True, exist_ok=False)

    calibration_spec = manifest["datasets"]["c4_calibration"]
    evaluation_spec = manifest["datasets"]["c4_en_512"]
    calibration_text = selected_texts(calibration_spec, limit=1)
    evaluation_text = selected_texts(evaluation_spec, limit=1)
    model_spec = manifest["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
    )
    calibration_prompt = tokenize_prompts(
        tokenizer,
        calibration_text,
        calibration_spec["token_limit"],
    )[0][:, :32]
    evaluation_prompt = tokenize_prompts(
        tokenizer,
        evaluation_text,
        evaluation_spec["token_limit"],
    )[0][:, :32]

    model = AutoModelForCausalLM.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision != model_spec["revision"]:
        raise RuntimeError(
            f"model resolved to {resolved_revision}, expected {model_spec['revision']}"
        )
    ranges, operator_ranges = calibrate(model, [calibration_prompt])

    seed_everything(config)
    baseline_norm, baseline_position = capture_layer0_inputs(model, evaluation_prompt)
    baseline = baseline_boundaries(
        model.model.layers[0].self_attn,
        baseline_norm,
        baseline_position,
    )

    replace_linears(model, ranges)
    replace_fixed_operators(model, operator_ranges)
    layer = model.model.layers[0]
    if not isinstance(layer.input_layernorm, FixedRMSNorm) or not isinstance(
        layer.self_attn, FixedAttention
    ):
        raise TypeError("layer-0 fixed operators were not installed")
    seed_everything(config)
    fixed_norm, fixed_position = capture_layer0_inputs(model, evaluation_prompt)
    fixed, metadata = fixed_boundaries(
        layer.self_attn,
        layer.input_layernorm,
        fixed_norm,
        fixed_position,
    )

    comparisons = {
        name: compare_tensor(
            baseline[name].detach().to(device="cpu", dtype=torch.float64),
            fixed[name].detach().to(device="cpu", dtype=torch.float64),
        )
        for name in baseline
    }
    query_absmax = comparisons["layer0.q_post_rope"]["reference_absmax"]
    key_absmax = comparisons["layer0.k_post_rope"]["reference_absmax"]
    integer_max = config["activation_quantization"]["integer_max"]
    required_query_scale = query_absmax / integer_max
    required_key_scale = key_absmax / integer_max
    realized_query_scale = metadata["rope"]["query_output_scale"]
    realized_key_scale = metadata["rope"]["key_output_scale"]
    realized_product = realized_query_scale * realized_key_scale
    feasibility = {
        "global_signed_int8_range_covered": (
            required_query_scale <= realized_query_scale
            and required_key_scale <= realized_key_scale
        ),
        "observed_query_absmax": query_absmax,
        "observed_key_absmax": key_absmax,
        "required_query_output_scale": required_query_scale,
        "required_key_output_scale": required_key_scale,
        "realized_query_output_scale": realized_query_scale,
        "realized_key_output_scale": realized_key_scale,
        "realized_output_scale_product": realized_product,
        "score_multiplier": metadata["rope"]["score_multiplier"],
        "score_right_shift": metadata["rope"]["score_right_shift"],
        "basis": (
            "Distinct Q/K signed-int8 scales must each cover the observed maxima; "
            "the attention-score multiplier consumes their realized product."
        ),
    }

    calibration_record_sha256, calibration_record_count = hash_records(calibration_text)
    evaluation_record_sha256, evaluation_record_count = hash_records(evaluation_text)
    calibration_token_sha256, calibration_sequences, calibration_tokens = (
        hash_token_sequences([calibration_prompt])
    )
    evaluation_token_sha256, evaluation_sequences, evaluation_tokens = (
        hash_token_sequences([evaluation_prompt])
    )
    source_paths = [
        PROMPT_MANIFEST,
        QUALITY_CONFIG,
        ROOT / "benchmark" / "quality" / "RTL_BINDING.json",
        ROOT / "tools" / "ace2_full_model_fixed_point.py",
        Path(__file__),
    ]
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "diagnostic_layer0_boundary_localization_not_acceptance_evidence",
        "gate_passed": False,
        "model": {**model_spec, "resolved_revision": resolved_revision},
        "input_observations": {
            "calibration": {
                "record_count": calibration_record_count,
                "record_sha256": calibration_record_sha256,
                "token_sequence_count": calibration_sequences,
                "token_sequence_sha256": calibration_token_sha256,
                "token_count": calibration_tokens,
            },
            "evaluation": {
                "record_count": evaluation_record_count,
                "record_sha256": evaluation_record_sha256,
                "token_sequence_count": evaluation_sequences,
                "token_sequence_sha256": evaluation_token_sha256,
                "token_count": evaluation_tokens,
            },
        },
        "boundary_order": list(comparisons),
        "comparisons": comparisons,
        "fixed_metadata": metadata,
        "scale_only_feasibility": feasibility,
        "contract": {
            "historical_rtl_binding": rtl_binding,
            "candidate_rtl_binding_required_for_acceptance": True,
            "attention_scale_derivation": config["full_model_scope"][
                "attention_scale_derivation"
            ],
        },
        "sources": [
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
        "runtime": {
            "packages": versions,
            "torch": torch.__version__,
        },
    }
    output_path = output_dir / "results.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sha256s = write_sha256s(output_dir)
    print(
        "ACE2_LAYER0_BOUNDARY_LOCALIZATION "
        f"q_rope_relative_l2={comparisons['layer0.q_post_rope']['relative_l2_error']:.9f} "
        f"k_rope_relative_l2={comparisons['layer0.k_post_rope']['relative_l2_error']:.9f} "
        f"range_covered={int(feasibility['global_signed_int8_range_covered'])} "
        f"sha256s_sha256={sha256_file(sha256s)}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    if ROOT.resolve() not in args.output_dir.resolve().parents:
        raise SystemExit("--output-dir must be below the repository root")
    run(args.output_dir)


if __name__ == "__main__":
    main()
