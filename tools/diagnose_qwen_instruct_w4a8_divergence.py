#!/usr/bin/env python3
"""Localize and repair the first decisive Option-B W4A8 activation collapse."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import ace2_full_model_fixed_point as fixed_point
from ace2_quality_contracts import ceil_scale32_from_float, scale32_ratio
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    ROOT,
    SNAPSHOT,
    canonical_bytes,
    file_record,
    load_json,
    require,
    sha256_bytes,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
)
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM, reconstruct_fixed_model


GROUP_LANES = 128
FIRST_DECISIVE_BOUNDARY = "model.layers.2.mlp"


def require_project_python() -> dict[str, Any]:
    expected = ROOT / ".venv/bin/python"
    observed = Path(sys.executable)
    require(expected.is_file(), f"project Python is missing: {expected}")
    require(os.path.samefile(observed, expected), f"wrong Python executable: {observed}")
    return {
        "bound_entrypoint": "./.venv/bin/python",
        "bound_entrypoint_absolute": str(expected),
        "resolved_executable": str(expected.resolve()),
        "sys_executable": str(observed),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": verify_versions(),
    }


def tensor_output(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, Tensor):
                return item
    raise TypeError(f"module output does not contain a tensor: {type(value).__name__}")


def capture_first_layers(model: nn.Module, input_ids: Tensor) -> tuple[dict[str, Tensor], Tensor]:
    captured: dict[str, Tensor] = {}
    hooks: list[Any] = []

    def capture_pre(name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            captured[name] = tensor_output(inputs).detach().cpu().clone()

        return hook

    def capture_output(name: str, *, rmsnorm: bool = False, keep_raw: bool = False):
        def hook(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = tensor_output(output).detach().cpu().clone()
            if keep_raw:
                captured[f"{name}.raw"] = value
            if rmsnorm and hasattr(module, "output_scale"):
                value = value.to(torch.float64) * float(module.output_scale)
            captured[name] = value

        return hook

    for index in range(4):
        layer = model.model.layers[index]
        prefix = f"model.layers.{index}"
        hooks.extend(
            [
                layer.register_forward_pre_hook(capture_pre(f"{prefix}.input")),
                layer.input_layernorm.register_forward_hook(
                    capture_output(f"{prefix}.input_layernorm", rmsnorm=True)
                ),
                layer.self_attn.register_forward_hook(
                    capture_output(f"{prefix}.self_attn")
                ),
                layer.post_attention_layernorm.register_forward_pre_hook(
                    capture_pre(f"{prefix}.post_attention_residual")
                ),
                layer.post_attention_layernorm.register_forward_hook(
                    capture_output(
                        f"{prefix}.post_attention_layernorm",
                        rmsnorm=True,
                        keep_raw=index == 2,
                    )
                ),
                layer.mlp.register_forward_hook(capture_output(f"{prefix}.mlp")),
                layer.register_forward_hook(capture_output(f"{prefix}.output")),
            ]
        )
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=False)
    for hook in hooks:
        hook.remove()
    return captured, output.logits[0, -1].detach().cpu().to(torch.float64)


def final_position_stats(value: Tensor) -> dict[str, Any]:
    final = value[0, -1].to(torch.float64)
    return {
        "absmax": float(final.abs().amax()),
        "l2": float(torch.linalg.vector_norm(final)),
        "shape": list(final.shape),
        "unique_values": int(torch.unique(final).numel()),
        "zero_fraction": float(torch.eq(final, 0).to(torch.float64).mean()),
    }


def comparison(reference: Tensor, candidate: Tensor) -> dict[str, Any]:
    left = reference[0, -1].to(torch.float64)
    right = candidate[0, -1].to(torch.float64)
    difference = right - left
    left_l2 = torch.linalg.vector_norm(left)
    right_l2 = torch.linalg.vector_norm(right)
    denominator = left_l2 * right_l2
    return {
        "cosine_similarity": (
            None if float(denominator) == 0.0 else float(torch.dot(left, right) / denominator)
        ),
        "relative_l2_error": (
            None if float(left_l2) == 0.0 else float(torch.linalg.vector_norm(difference) / left_l2)
        ),
    }


def boundary_order() -> list[str]:
    names: list[str] = []
    for index in range(4):
        prefix = f"model.layers.{index}"
        names.extend(
            [
                f"{prefix}.input_layernorm",
                f"{prefix}.self_attn",
                f"{prefix}.post_attention_residual",
                f"{prefix}.post_attention_layernorm",
                f"{prefix}.mlp",
                f"{prefix}.output",
            ]
        )
    return names


def silu_product_q9_21(gate: Tensor, up: Tensor, gate_scale: float, up_scale: float) -> Tensor:
    gate_q6_9 = torch.round(gate.to(torch.float64) * gate_scale * (1 << 9))
    up_q6_9 = torch.round(up.to(torch.float64) * up_scale * (1 << 9))
    gate_q6_9 = gate_q6_9.clamp(-32768, 32767).to(torch.int64)
    up_q6_9 = up_q6_9.clamp(-32768, 32767).to(torch.int64)
    table_index = torch.bitwise_right_shift(gate_q6_9, 6).clamp(-64, 64) + 64
    table = torch.tensor(fixed_point.SILU_LUT, dtype=torch.int64, device=gate.device)
    return table[table_index] * up_q6_9


def dynamic_group_requantize(product_q9_21: Tensor) -> tuple[Tensor, Tensor, list[list[int]], list[list[float]]]:
    require(product_q9_21.shape[-1] % GROUP_LANES == 0, "MLP width is not 128-lane grouped")
    raw = torch.empty_like(product_q9_21, dtype=torch.int8)
    dequantized = torch.empty_like(product_q9_21, dtype=torch.float64)
    records: list[list[int]] = []
    scales: list[list[float]] = []
    for token in range(product_q9_21.shape[1]):
        token_records: list[int] = []
        token_scales: list[float] = []
        for start in range(0, product_q9_21.shape[-1], GROUP_LANES):
            stop = start + GROUP_LANES
            product = product_q9_21[0, token, start:stop]
            real = product.to(torch.float64) / float(1 << 21)
            absmax = float(real.abs().amax())
            if absmax == 0.0:
                scale = math.ldexp(1.0, -24)
                record = ceil_scale32_from_float(scale)
                quantized = torch.zeros_like(product, dtype=torch.int8)
            else:
                record = ceil_scale32_from_float(absmax / 127.0)
                numerator, denominator = scale32_ratio(record)
                scale = numerator / denominator
                quantized = torch.round(real / scale).clamp(-127, 127).to(torch.int8)
            raw[0, token, start:stop] = quantized
            dequantized[0, token, start:stop] = quantized.to(torch.float64) * scale
            token_records.append(record)
            token_scales.append(scale)
        records.append(token_records)
        scales.append(token_scales)
    return raw, dequantized, records, scales


def quantize_linears_w4_weight_only(model: nn.Module) -> int:
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, nn.Linear):
                continue
            weight = module.weight.detach().to(torch.float64)
            scale = weight.abs().amax(dim=1) / 7.0
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            quantized = torch.round(weight / scale[:, None]).clamp(-8, 7)
            module.weight.copy_((quantized * scale[:, None]).to(module.weight.dtype))
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    environment = require_project_python()
    source_contract = verify_source_contract()
    source_identity = verify_source_snapshot()
    scales_path = CALIBRATION_DIR / "derived_scales.json"
    scales = load_json(scales_path)

    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT, local_files_only=True, trust_remote_code=False
    )
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    require(isinstance(input_ids, Tensor) and input_ids.ndim == 2, "tokenization differs")

    source_model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    source_boundaries, source_logits = capture_first_layers(source_model, input_ids)

    fixed_model = reconstruct_fixed_model(scales)
    fixed_boundaries, fixed_logits = capture_first_layers(fixed_model, input_ids)

    comparisons: dict[str, Any] = {}
    first_decisive = None
    for name in boundary_order():
        reference_stats = final_position_stats(source_boundaries[name])
        candidate_stats = final_position_stats(fixed_boundaries[name])
        comparisons[name] = {
            "bf16": reference_stats,
            "w4a8": candidate_stats,
            **comparison(source_boundaries[name], fixed_boundaries[name]),
        }
        if (
            first_decisive is None
            and reference_stats["zero_fraction"] < 0.01
            and candidate_stats["zero_fraction"] >= 0.99
        ):
            first_decisive = name

    source_hidden = source_boundaries["model.layers.2.post_attention_layernorm"]
    source_mlp = source_model.model.layers[2].mlp
    with torch.inference_mode():
        source_gate = source_mlp.gate_proj(source_hidden)
        source_up = source_mlp.up_proj(source_hidden)
        source_gated = F.silu(source_gate) * source_up

    fixed_hidden_raw = fixed_boundaries["model.layers.2.post_attention_layernorm.raw"]
    fixed_mlp = fixed_model.model.layers[2].mlp
    with torch.inference_mode():
        fixed_gate = fixed_mlp.gate_proj.forward_hardware_input(fixed_hidden_raw)
        fixed_up = fixed_mlp.up_proj.forward_hardware_input(fixed_hidden_raw)
        static_raw = fixed_point.fixed_silu_gate_raw(
            fixed_gate,
            fixed_up,
            fixed_mlp.gate_proj.output_scale,
            fixed_mlp.up_proj.output_scale,
            fixed_mlp.silu_multiplier,
            fixed_mlp.silu_right_shift,
        )
    product = silu_product_q9_21(
        fixed_gate,
        fixed_up,
        fixed_mlp.gate_proj.output_scale,
        fixed_mlp.up_proj.output_scale,
    )
    fixed_wide = product.to(torch.float64) / float(1 << 21)
    static_dequantized = static_raw.to(torch.float64) * fixed_mlp.down_proj.input_scale
    dynamic_raw, dynamic_dequantized, dynamic_records, dynamic_scales = (
        dynamic_group_requantize(product)
    )

    source_abs = source_gated[0].detach().abs().to(torch.float64)
    source_flat_index = int(source_abs.argmax())
    source_position = source_flat_index // source_abs.shape[1]
    source_channel = source_flat_index % source_abs.shape[1]
    final_dynamic_scales = dynamic_scales[-1]
    sink_dynamic_scales = dynamic_scales[0]

    tied_embedding_lm_head = (
        source_model.config.tie_word_embeddings
        and source_model.lm_head.weight.data_ptr()
        == source_model.model.embed_tokens.weight.data_ptr()
    )
    weight_only_linear_count = quantize_linears_w4_weight_only(source_model)
    with torch.inference_mode():
        weight_only_logits = source_model(input_ids=input_ids, use_cache=False).logits[
            0, -1
        ].detach().cpu().to(torch.float64)

    static_error = comparison(fixed_wide, static_dequantized)
    dynamic_error = comparison(fixed_wide, dynamic_dequantized)
    boundary_checks = {
        "bf16_and_w4a8_first_tokens_diverge": int(source_logits.argmax()) != int(fixed_logits.argmax()),
        "earliest_decisive_boundary_is_layer2_mlp": first_decisive == FIRST_DECISIVE_BOUNDARY,
        "frozen_static_silu_gate_final_position_is_all_zero": final_position_stats(static_raw)["zero_fraction"] == 1.0,
        "dynamic_group_repair_final_position_is_nonzero": final_position_stats(dynamic_raw)["zero_fraction"] < 1.0,
        "dynamic_group_repair_reduces_boundary_error": (
            dynamic_error["relative_l2_error"] is not None
            and static_error["relative_l2_error"] is not None
            and dynamic_error["relative_l2_error"] < static_error["relative_l2_error"]
        ),
        "sink_outlier_is_position_zero_im_start": (
            source_position == 0
            and int(input_ids[0, source_position]) == 151644
        ),
    }
    boundary_passed = all(boundary_checks.values())
    weight_only_first_token_changed = (
        int(weight_only_logits.argmax()) != int(source_logits.argmax())
    )
    result = {
        "schema_version": 2,
        "classification": "qwen_instruct_option_b_first_decisive_w4a8_divergence_and_boundary_repair",
        "status": (
            "PASS_LOCALIZED_AND_REPAIRED_BOUNDARY_FULL_ORACLE_STILL_BLOCKED"
            if boundary_passed
            else "FAIL_BOUNDARY_REPAIR_DIAGNOSTIC"
        ),
        "environment": environment,
        "model": source_identity,
        "prompt": {
            "chat_template_token_count": int(input_ids.shape[1]),
            "system_sha256": sha256_bytes(args.system_prompt.encode()),
            "user_sha256": sha256_bytes(args.prompt.encode()),
        },
        "first_step": {
            "bf16_token_id": int(source_logits.argmax()),
            "w4a8_token_id": int(fixed_logits.argmax()),
        },
        "localization": {
            "definition": "first execution-ordered boundary with <1% BF16 zeros and at least 99% W4A8 zeros at the generation position",
            "first_decisive_boundary": first_decisive,
            "comparisons": comparisons,
        },
        "root_cause": {
            "operator": "model.layers.2.mlp.fixed_silu_gate_raw_to_down_proj_input",
            "classification": "tensor_wide_static_activation_scale_polluted_by_position0_sink_token_outlier",
            "frozen_down_projection_input_absmax": float(
                scales["linears"]["model.layers.2.mlp.down_proj"]["input_absmax"]
            ),
            "frozen_down_projection_input_scale": float(fixed_mlp.down_proj.input_scale),
            "source_gated_global_absmax": float(source_abs.amax()),
            "source_gated_global_absmax_position": source_position,
            "source_gated_global_absmax_channel": source_channel,
            "source_gated_global_absmax_token_id": int(input_ids[0, source_position]),
            "source_gated_final_position_absmax": float(source_abs[-1].amax()),
            "outlier_to_final_absmax_ratio": float(source_abs.amax() / source_abs[-1].amax()),
        },
        "repair": {
            "scope": "layer2_silu_gate_output_boundary_only",
            "mechanism": "per_token_128_lane_dynamic_Scale32_signed_A8_requantization",
            "weight_quantization_changed": False,
            "frozen_v1_artifacts_mutated": False,
            "static": {
                "final_position": final_position_stats(static_dequantized),
                "relative_l2_error_to_fixed_wide": static_error["relative_l2_error"],
            },
            "dynamic": {
                "final_position": final_position_stats(dynamic_dequantized),
                "relative_l2_error_to_fixed_wide": dynamic_error["relative_l2_error"],
                "groups_per_token": len(final_dynamic_scales),
                "final_position_scale_min": min(final_dynamic_scales),
                "final_position_scale_max": max(final_dynamic_scales),
                "sink_position_scale_min": min(sink_dynamic_scales),
                "sink_position_scale_max": max(sink_dynamic_scales),
                "final_position_scale32_records": dynamic_records[-1],
            },
            "integration_status": "boundary_repaired_only_mixed_scale_down_projection_and_recurrent_propagation_pending",
        },
        "independent_remaining_blocker": {
            "classification": "prompt_dependent_w4_weight_only_token_sensitivity_requiring_separate_causal_localization",
            "indicator_status": (
                "W4_WEIGHT_ONLY_FIRST_TOKEN_CHANGED"
                if weight_only_first_token_changed
                else "W4_WEIGHT_ONLY_FIRST_TOKEN_UNCHANGED"
            ),
            "linears_quantized": weight_only_linear_count,
            "lm_head_shares_model_embed_tokens": tied_embedding_lm_head,
            "in_place_lm_head_quantization_also_changes_embedding_weight": tied_embedding_lm_head,
            "bf16_first_token_id": int(source_logits.argmax()),
            "w4_weight_only_first_token_id": int(weight_only_logits.argmax()),
            "first_token_changed": weight_only_first_token_changed,
            "affects_boundary_repair_verdict": False,
            "numerical_repair_applied": False,
        },
        "verdicts": {
            "boundary_repair": {
                "passed": boundary_passed,
                "checks": boundary_checks,
            },
            "weight_only_token_sensitivity": {
                "classification": "prompt_dependent_non_acceptance_indicator",
                "observed": weight_only_first_token_changed,
                "affects_boundary_repair_verdict": False,
            },
        },
        "artifacts": {
            "source_contract": file_record(ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"),
            "derived_scales": file_record(scales_path),
            "diagnostic_source": file_record(Path(__file__)),
        },
        "scope_guards": {
            "accelerator_executed": False,
            "demo_retargeted": False,
            "network_access_performed": False,
            "ppa_executed": False,
            "source_contract_status": source_contract["status"],
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(result))
    print(
        "ACE2_QWEN_INSTRUCT_DIVERGENCE "
        f"status={result['status']} boundary={first_decisive} "
        f"bf16={result['first_step']['bf16_token_id']} "
        f"w4a8={result['first_step']['w4a8_token_id']} output={output.relative_to(ROOT)}"
    )
    return 0 if boundary_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
