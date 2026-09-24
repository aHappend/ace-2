#!/usr/bin/env python3
"""Freeze the selected Option-B v2 numerical overlay and Scale32 boundary witness."""

from __future__ import annotations

import argparse
import math
import os
import platform
import re
import struct
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ace2_quality_contracts import (
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    round_divide_even_signed,
    scale32_ratio,
    unpack_scale32,
)
from diagnose_qwen_instruct_w4a8_divergence import (
    capture_first_layers,
    dynamic_group_requantize,
    silu_product_q9_21,
)
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    IMAGE_DIR,
    REPOSITORY,
    REVISION,
    ROOT,
    SNAPSHOT,
    canonical_bytes,
    file_record,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
)
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM, reconstruct_fixed_model


TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
SELECTED_POLICY_ID = "layer23_down_per_32"
SELECTED_OPERATOR = "model.layers.23.mlp.down_proj"
SELECTED_GROUP_LANES = 32
BOUNDARY_GROUP_LANES = 128
BOUNDARY_PROMPT = "Name one color and one bird, using exactly two words."
BOUNDARY_RESULT = (
    ROOT
    / "build/option-b-weight-localization-20260806/"
    "independent_prompt_boundary_separation.json"
)
SOURCE_CONTRACT = ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"


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


def packed_s4(value: Tensor) -> bytes:
    flat = value.detach().reshape(-1).to(torch.int16).cpu()
    require(flat.numel() % 2 == 0, "signed-int4 payload has odd element count")
    packed = torch.bitwise_or(
        flat[0::2] & 0xF,
        torch.bitwise_left_shift(flat[1::2] & 0xF, 4),
    )
    return packed.to(torch.uint8).numpy().tobytes(order="C")


def grouped_w4(weight: Tensor, group_lanes: int) -> tuple[Tensor, Tensor]:
    value = weight.detach().to(torch.float64)
    require(value.shape[1] % group_lanes == 0, "selected weight width is not grouped")
    grouped = value.reshape(value.shape[0], -1, group_lanes)
    scale = grouped.abs().amax(dim=2) / 7.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    qweight = torch.round(grouped / scale[:, :, None]).clamp(-8, 7).to(torch.int8)
    return qweight.reshape_as(value), scale


def scale32_bytes(scales: Tensor) -> tuple[bytes, list[int]]:
    records = [
        ceil_scale32_from_float(float(value))
        for value in scales.detach().cpu().reshape(-1).tolist()
    ]
    return b"".join(struct.pack("<I", record) for record in records), records


def float64_bytes(scales: Tensor) -> bytes:
    return b"".join(
        struct.pack("<d", float(value))
        for value in scales.detach().cpu().reshape(-1).tolist()
    )


def exact_mixed_scale32_boundary(
    fixed_model: nn.Module,
    input_ids: Tensor,
) -> dict[str, Any]:
    fixed_boundaries, _fixed_logits = capture_first_layers(fixed_model, input_ids)
    hidden_raw = fixed_boundaries["model.layers.2.post_attention_layernorm.raw"]
    mlp = fixed_model.model.layers[2].mlp
    with torch.inference_mode():
        gate = mlp.gate_proj.forward_hardware_input(hidden_raw)
        up = mlp.up_proj.forward_hardware_input(hidden_raw)
    product = silu_product_q9_21(
        gate,
        up,
        mlp.gate_proj.output_scale,
        mlp.up_proj.output_scale,
    )
    dynamic_raw, dynamic_dequantized, dynamic_records, dynamic_scales = (
        dynamic_group_requantize(product)
    )
    final_qinput = dynamic_raw[0, -1]
    final_input_records = dynamic_records[-1]
    final_input_scales = dynamic_scales[-1]
    down = mlp.down_proj
    require(down.in_features == 4864 and down.out_features == 896, "layer-2 down geometry differs")
    require(len(final_input_records) == 38, "layer-2 dynamic group count differs")
    require(down.bias_accumulator is None, "layer-2 down projection unexpectedly has bias")

    qinput_groups = final_qinput.reshape(-1, BOUNDARY_GROUP_LANES).to(torch.int64)
    qweight_groups = down.qweight.reshape(
        down.out_features, -1, BOUNDARY_GROUP_LANES
    ).to(torch.int64)
    accumulators = torch.empty(
        (qinput_groups.shape[0], down.out_features), dtype=torch.int64
    )
    for group in range(qinput_groups.shape[0]):
        accumulators[group] = (
            qweight_groups[:, group, :] * qinput_groups[group][None, :]
        ).sum(dim=1)
    require(
        not bool(torch.any(accumulators < -(1 << 31)))
        and not bool(torch.any(accumulators >= (1 << 31))),
        "layer-2 group accumulator exceeds signed-32",
    )

    weight_scales = down.weight_scale.detach().cpu().to(torch.float64)
    accumulator_records: list[list[int]] = []
    for input_scale in final_input_scales:
        accumulator_records.append(
            [
                ceil_scale32_from_float(float(input_scale) * float(weight_scale))
                for weight_scale in weight_scales.tolist()
            ]
        )
    wide_numerators: list[int] = []
    wide_denominators: list[int] = []
    maximum_absolute_numerator = 0
    maximum_denominator = 0
    common_exponents: list[int] = []
    accumulator_rows = accumulators.tolist()
    for output_lane in range(down.out_features):
        source_records = [
            accumulator_records[group][output_lane]
            for group in range(len(accumulator_records))
        ]
        unpacked = [unpack_scale32(record) for record in source_records]
        common_exp = min(value[1] for value in unpacked)
        numerator = 0
        for group, (significand, exponent) in enumerate(unpacked):
            numerator += (
                int(accumulator_rows[group][output_lane])
                * significand
                * (1 << (exponent - common_exp))
            )
        denominator = 1 << (15 - common_exp)
        wide_numerators.append(numerator)
        wide_denominators.append(denominator)
        maximum_absolute_numerator = max(maximum_absolute_numerator, abs(numerator))
        maximum_denominator = max(maximum_denominator, denominator)
        common_exponents.append(common_exp)

    maximum_ratio_numerator = 0
    maximum_ratio_denominator = 1
    for numerator, denominator in zip(
        wide_numerators, wide_denominators, strict=True
    ):
        magnitude = abs(numerator)
        if (
            magnitude * maximum_ratio_denominator
            > maximum_ratio_numerator * denominator
        ):
            maximum_ratio_numerator = magnitude
            maximum_ratio_denominator = denominator
    require(maximum_ratio_numerator > 0, "dynamic layer-2 down output is all zero")
    destination_record = ceil_scale32_from_ratio(
        maximum_ratio_numerator, maximum_ratio_denominator * 127
    )
    destination_numerator, destination_denominator = scale32_ratio(
        destination_record
    )

    outputs: list[int] = []
    positive_saturations = 0
    negative_saturations = 0
    for numerator, denominator in zip(
        wide_numerators, wide_denominators, strict=True
    ):
        requant_numerator = numerator * destination_denominator
        requant_denominator = denominator * destination_numerator
        rounded = round_divide_even_signed(
            requant_numerator, requant_denominator
        )
        positive_saturations += int(rounded > 127)
        negative_saturations += int(rounded < -127)
        outputs.append(max(-127, min(127, rounded)))
    output = torch.tensor(outputs, dtype=torch.int8)

    exact_scale_values = torch.tensor(
        [
            [scale32_ratio(record)[0] / scale32_ratio(record)[1] for record in row]
            for row in accumulator_records
        ],
        dtype=torch.float64,
    )
    float_sum = (accumulators.to(torch.float64) * exact_scale_values).sum(dim=0)
    float_output = torch.round(
        float_sum / (destination_numerator / destination_denominator)
    ).clamp(-127, 127).to(torch.int8)
    require(torch.equal(output, float_output), "exact mixed-Scale32 output differs from one-round reference")

    dequantized_output = output.to(torch.float64) * (
        destination_numerator / destination_denominator
    )
    proxy_weight = down.qweight.to(torch.float64) * weight_scales[:, None]
    proxy_output = torch.nn.functional.linear(
        dynamic_dequantized[0, -1].to(torch.float64), proxy_weight
    )
    proxy_norm = torch.linalg.vector_norm(proxy_output)
    relative_error_to_proxy = float(
        torch.linalg.vector_norm(dequantized_output - proxy_output) / proxy_norm
    )
    exact_wide = torch.tensor(
        [
            numerator / denominator
            for numerator, denominator in zip(
                wide_numerators, wide_denominators, strict=True
            )
        ],
        dtype=torch.float64,
    )
    exact_wide_norm = torch.linalg.vector_norm(exact_wide)
    relative_error_to_exact_wide = float(
        torch.linalg.vector_norm(dequantized_output - exact_wide) / exact_wide_norm
    )
    relative_error_bound = float(
        math.sqrt(exact_wide.numel())
        * (destination_numerator / destination_denominator)
        / 2.0
        / exact_wide_norm
    )

    frozen_destination_record = ceil_scale32_from_float(float(down.output_scale))
    frozen_numerator, frozen_denominator = scale32_ratio(
        frozen_destination_record
    )
    frozen_outputs = []
    for numerator, denominator in zip(
        wide_numerators, wide_denominators, strict=True
    ):
        rounded = round_divide_even_signed(
            numerator * frozen_denominator,
            denominator * frozen_numerator,
        )
        frozen_outputs.append(max(-127, min(127, rounded)))
    frozen_output = torch.tensor(frozen_outputs, dtype=torch.int8)
    dynamic_nonzero_fraction = float(
        torch.ne(output, 0).to(torch.float64).mean()
    )
    frozen_nonzero_fraction = float(
        torch.ne(frozen_output, 0).to(torch.float64).mean()
    )
    require(dynamic_nonzero_fraction > 0.0, "dynamic destination scale still collapses layer-2 down output")
    require(
        relative_error_to_exact_wide <= relative_error_bound + 1e-15,
        "dynamic destination scale exceeds its signed-A8 quantization-error bound",
    )
    require(
        relative_error_to_exact_wide < 1.0,
        "dynamic destination scale does not improve on the frozen all-zero output",
    )
    return {
        "status": "PASS_EXACT_ONE_ROUND_DYNAMIC_DESTINATION_SCALE32_LAYER2_DOWN_PROJECTION",
        "prompt_sha256": sha256_bytes(BOUNDARY_PROMPT.encode()),
        "chat_template_token_count": int(input_ids.shape[1]),
        "activation_group_lanes": BOUNDARY_GROUP_LANES,
        "activation_groups": len(final_input_records),
        "output_lanes": down.out_features,
        "group_accumulator_count": accumulators.numel(),
        "group_accumulator_signed32_safe": True,
        "accumulator_scale32_record_count": len(final_input_records) * down.out_features,
        "frozen_destination_scale32_record": frozen_destination_record,
        "frozen_destination_nonzero_fraction": frozen_nonzero_fraction,
        "dynamic_destination_scale32_record": destination_record,
        "dynamic_destination_scale": destination_numerator / destination_denominator,
        "one_round_exact_reference_match": True,
        "dynamic_output_nonzero_fraction": dynamic_nonzero_fraction,
        "positive_saturations": positive_saturations,
        "negative_saturations": negative_saturations,
        "maximum_absolute_numerator": maximum_absolute_numerator,
        "maximum_denominator": maximum_denominator,
        "common_exponent_minimum": min(common_exponents),
        "common_exponent_maximum": max(common_exponents),
        "relative_l2_error_to_exact_mixed_scale32_wide": relative_error_to_exact_wide,
        "relative_l2_signed_a8_error_bound": relative_error_bound,
        "relative_l2_error_to_float64_grouped_w4_proxy": relative_error_to_proxy,
        "final_activation_scale32_records": final_input_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(
        TIMESTAMP.fullmatch(args.generated_at_utc) is not None,
        "timestamp must use YYYY-MM-DDTHH:MM:SSZ",
    )
    environment = require_project_python()
    source_contract = verify_source_contract()
    source_identity = verify_source_snapshot()
    selection_path = args.selection.resolve()
    selection = load_json(selection_path)
    require(
        selection.get("status") == "PASS_UNIQUE_SEQUENCE_FIRST_WEIGHT_POLICY_SELECTED",
        "weight-policy selection did not pass",
    )
    selected = selection.get("selected", {})
    require(
        selected.get("policy_id") == SELECTED_POLICY_ID
        and selected.get("operator") == SELECTED_OPERATOR
        and selected.get("input_group_lanes") == SELECTED_GROUP_LANES,
        "selected weight policy differs",
    )
    boundary_result = load_json(BOUNDARY_RESULT)
    require(
        boundary_result.get("status")
        == "PASS_LOCALIZED_AND_REPAIRED_BOUNDARY_FULL_ORACLE_STILL_BLOCKED",
        "boundary diagnostic status differs",
    )
    require(
        boundary_result.get("repair", {}).get("mechanism")
        == "per_token_128_lane_dynamic_Scale32_signed_A8_requantization",
        "boundary repair mechanism differs",
    )
    require(
        boundary_result.get("prompt", {}).get("user_sha256")
        == sha256_bytes(BOUNDARY_PROMPT.encode()),
        "boundary prompt identity differs",
    )

    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"candidate output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    source_model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    selected_module = dict(source_model.named_modules()).get(SELECTED_OPERATOR)
    require(isinstance(selected_module, nn.Linear), "selected source linear is missing")
    qweight, weight_scales = grouped_w4(
        selected_module.weight, SELECTED_GROUP_LANES
    )
    require(
        tuple(weight_scales.shape) == (896, 152),
        "selected grouped scale geometry differs",
    )
    reconstructed = qweight.to(torch.float64).reshape(896, 152, 32) * weight_scales[:, :, None]
    reference_weight = selected_module.weight.detach().to(torch.float64).reshape_as(reconstructed)
    weight_error = float(
        torch.linalg.vector_norm(reconstructed - reference_weight)
        / torch.linalg.vector_norm(reference_weight)
    )

    qweight_path = output_dir / "layer23_down_proj_qweight.s4"
    float_scale_path = output_dir / "layer23_down_proj_weight_scales.f64le"
    scale32_path = output_dir / "layer23_down_proj_weight_scales.scale32le"
    qweight_path.write_bytes(packed_s4(qweight))
    float_scale_path.write_bytes(float64_bytes(weight_scales))
    encoded_scale32, weight_scale32_records = scale32_bytes(weight_scales)
    scale32_path.write_bytes(encoded_scale32)

    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT, local_files_only=True, trust_remote_code=False
    )
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": BOUNDARY_PROMPT},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    require(isinstance(input_ids, Tensor) and input_ids.ndim == 2, "boundary tokenization differs")
    scales = load_json(CALIBRATION_DIR / "derived_scales.json")
    fixed_model = reconstruct_fixed_model(scales)
    boundary_integration = exact_mixed_scale32_boundary(fixed_model, input_ids)

    scale32_values = torch.tensor(
        [
            scale32_ratio(record)[0] / scale32_ratio(record)[1]
            for record in weight_scale32_records
        ],
        dtype=torch.float64,
    ).reshape_as(weight_scales)
    scale32_inflation = scale32_values / weight_scales
    contract = {
        "schema_version": 1,
        "contract_id": "qwen-instruct-option-b-v2-numerical-candidate-v1",
        "classification": "frozen_option_b_v2_numerical_candidate_full_oracle_pending",
        "generated_at_utc": args.generated_at_utc,
        "status": "FROZEN_NUMERICAL_CANDIDATE_BOUNDARY_INTEGRATED_FULL_ORACLE_PENDING",
        "model": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "source": source_identity,
        },
        "weight_policy": {
            "policy_id": SELECTED_POLICY_ID,
            "operator": SELECTED_OPERATOR,
            "weight_bits": 4,
            "signed_integer_range": [-8, 7],
            "input_group_lanes": SELECTED_GROUP_LANES,
            "groups_per_output": 152,
            "output_lanes": 896,
            "scale_count": int(weight_scales.numel()),
            "float64_proxy_relative_l2_weight_error": weight_error,
            "scale32_ceiling_inflation_minimum": float(scale32_inflation.min()),
            "scale32_ceiling_inflation_maximum": float(scale32_inflation.max()),
            "multiply_accumulate_count_ratio": 1.0,
            "additional_group_combines": 896 * 151,
            "selection": file_record(selection_path),
        },
        "layer2_dynamic_boundary": {
            "mechanism": "per_token_128_lane_dynamic_Scale32_signed_A8_requantization",
            "scope": "model.layers.2.mlp.fixed_silu_gate_raw_to_down_proj_input",
            "selected_weight_policy_operator_is_disjoint": True,
            "exact_mixed_scale32_down_projection": boundary_integration,
            "boundary_diagnostic": file_record(BOUNDARY_RESULT),
        },
        "combined_numerical_contract": {
            "layer2": "38 signed-A8 activation groups per token; each group dot product uses one exact Scale32 accumulator record per output lane; all 38 terms are exponent-aligned, one per-token destination Scale32 is derived from the exact wide output absmax, and the result is rounded once",
            "layer23": "152 signed-W4 input groups per output lane for down_proj; packed weights are unchanged in count and grouped scale metadata is frozen by this candidate",
            "rounding": "round-to-nearest ties-to-even only after exact mixed-scale accumulation",
            "saturation": "signed int8 after the single destination requantization",
            "recurrent_scale_propagation": "pending through residual, RMSNorm, intervening projections, final norm, and LM head",
        },
        "frozen_overlay_artifacts": {
            "grouped_qweight": file_record(qweight_path),
            "float64_weight_scales": file_record(float_scale_path),
            "scale32_weight_scales": file_record(scale32_path),
        },
        "source_artifacts": {
            "source_contract": file_record(SOURCE_CONTRACT),
            "v1_identity_manifest": file_record(
                IMAGE_DIR / "option_b_identity_manifest.json"
            ),
            "v1_checksum_manifest": file_record(IMAGE_DIR / "OPTION_B_SHA256SUMS"),
            "freeze_source": file_record(Path(__file__)),
        },
        "acceptance_boundary": {
            "full_w4a8_oracle_passed": False,
            "response_gate_passed_by_combined_candidate": False,
            "accelerator_reference_agreement_passed": False,
            "demo_retarget_authorized": False,
            "stage1_complete": False,
        },
        "scope_guards": {
            "frozen_v1_artifacts_mutated": False,
            "base_artifacts_mutated": False,
            "rtl_mutated": False,
            "accelerator_executed": False,
            "demo_retargeted": False,
            "network_access_performed": False,
            "ppa_executed": False,
            "source_contract_status": source_contract["status"],
        },
        "environment": environment,
    }
    contract_path = output_dir / "numerical_candidate_contract.json"
    contract_path.write_bytes(canonical_bytes(contract))
    names = [
        qweight_path.name,
        float_scale_path.name,
        scale32_path.name,
        contract_path.name,
    ]
    sums_path = output_dir / "OPTION_B_V2_SHA256SUMS"
    sums_path.write_text(
        "\n".join(f"{sha256_file(output_dir / name)}  {name}" for name in sorted(names))
        + "\n",
        encoding="utf-8",
    )
    print(
        "ACE2_QWEN_INSTRUCT_OPTION_B_V2_CANDIDATE_FROZEN "
        f"policy={SELECTED_POLICY_ID} boundary={boundary_integration['status']} "
        f"contract_sha256={sha256_file(contract_path)} output_dir={output_dir.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
