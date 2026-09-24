#!/usr/bin/env python3
"""Offline integer-reference test for layer-2 dynamic Scale32 recurrence."""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import AutoTokenizer

import ace2_full_model_fixed_point as fixed_point
from ace2_quality_contracts import (
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    round_divide_even_signed,
    scale32_ratio,
    unpack_scale32,
)
from ace2_rmsnorm_reference import reference_rmsnorm
from diagnose_qwen_instruct_w4a8_divergence import (
    capture_first_layers,
    dynamic_group_requantize,
    silu_product_q9_21,
)
from discriminate_qwen_instruct_w4_weight_policies import PROMPTS, chat_input
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
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


HIDDEN_SIZE = 896
GROUP_LANES = 128
SIGNED32_MIN = -(1 << 31)
SIGNED32_MAX = (1 << 31) - 1
SIGNED96_MIN = -(1 << 95)
SIGNED96_MAX = (1 << 95) - 1
UNSIGNED64_MAX = (1 << 64) - 1
CONTRACT_ID = "qwen-instruct-layer2-dynamic-scale32-to-layer3-rmsnorm-v1"
DISCRIMINATION_SOURCE = ROOT / "tools/discriminate_qwen_instruct_w4_weight_policies.py"


def environment() -> dict[str, Any]:
    expected = ROOT / ".venv/bin/python"
    observed = Path(sys.executable)
    require(expected.is_file(), f"project Python is missing: {expected}")
    require(os.path.samefile(observed, expected), f"wrong Python executable: {observed}")
    return {
        "bound_entrypoint": "./.venv/bin/python",
        "sys_executable": str(observed),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": verify_versions(),
    }


def predeclared_contract() -> dict[str, Any]:
    prompts = [
        {
            "case_id": case.case_id,
            "prompt_sha256": sha256_bytes(case.prompt.encode()),
        }
        for case in PROMPTS
    ]
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "scope": "offline_final_prefill_token_only_across_unchanged_nine_prompts",
        "prompt_binding": {
            "source": file_record(DISCRIMINATION_SOURCE),
            "case_count": 9,
            "cases": prompts,
            "system_prompt_sha256": sha256_bytes(DEFAULT_SYSTEM.encode()),
        },
        "mechanism": {
            "layer2_mlp_input": "per-token 128-lane signed-A8 groups with normalized Scale32",
            "layer2_down_projection": "38 signed-32 group dot products per output lane; exact Scale32 exponent alignment; no intermediate rounding",
            "layer2_residual": "signed-A8 post-attention residual with its frozen Scale32",
            "fusion_destination": "one per-token normalized Scale32: smallest legal Scale32 greater than or equal to exact fused absmax/127",
            "layer3_rmsnorm": "the exact fusion Scale32 record is forwarded unchanged; scalar scale cancels before integer RMS normalization",
        },
        "rounding": {
            "fusion": "signed round-to-nearest ties-to-even once after exact residual fusion",
            "rms_mean_square": "add 448 then integer-divide by 896",
            "rms_root": "positive ceil integer square root",
            "rms_inverse": "floor(2^30/root)",
            "rms_output": "signed round-to-nearest ties-to-even after shift by 38",
        },
        "widths": {
            "activation_and_residual": "signed-8",
            "weight": "signed-4 stored in signed-8",
            "group_dot": "signed-32",
            "fusion_numerator": "signed-96",
            "fusion_denominator": "unsigned-64",
            "rms_sumsq": "unsigned-48",
            "rms_gain": "signed-16 Q7.8",
            "rms_inverse": "unsigned-31 Q1.30",
            "rms_product": "signed-64",
            "outputs": "signed-8",
        },
        "overflow": "fail_closed_before_any_narrowing; final outputs saturate to [-128,127]",
        "pass_criteria": {
            "all_nine_prompts_required": True,
            "fusion_exact_scalar_integer_reference_match": True,
            "fusion_scale32_forwarded_unchanged_to_layer3_rmsnorm": True,
            "layer3_rmsnorm_exact_scalar_integer_reference_match": True,
            "all_declared_width_checks_pass": True,
            "fusion_and_rmsnorm_outputs_each_contain_at_least_one_nonzero_lane": True,
            "fusion_relative_l2_error_not_above_derived_half_lsb_bound": True,
        },
        "containment": {
            "policy_or_gate_changed": False,
            "demo_retargeted": False,
            "rtl_mutated": False,
            "ppa_executed": False,
            "stage2_entered": False,
            "network_access_permitted": False,
        },
        "test_source": file_record(Path(__file__)),
    }


def write_contract(path: Path) -> None:
    contract = predeclared_contract()
    wrapper = {
        "contract": contract,
        "contract_sha256": sha256_bytes(canonical_bytes(contract)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(wrapper))


def load_bound_contract(path: Path) -> tuple[dict[str, Any], str]:
    wrapper = load_json(path)
    require(set(wrapper) == {"contract", "contract_sha256"}, "contract wrapper differs")
    contract = wrapper["contract"]
    digest = sha256_bytes(canonical_bytes(contract))
    require(digest == wrapper["contract_sha256"], "predeclared contract digest differs")
    require(contract == predeclared_contract(), "live test or prompt binding differs from predeclaration")
    return contract, digest


def scalar_rne(numerator: int, denominator: int) -> int:
    require(denominator > 0, "non-positive scalar denominator")
    quotient, remainder = divmod(abs(numerator), denominator)
    doubled = remainder * 2
    quotient += int(doubled > denominator or (doubled == denominator and quotient & 1))
    return -quotient if numerator < 0 else quotient


def mixed_down_and_fuse(
    final_qinput: Tensor,
    input_records: list[int],
    residual_s8: Tensor,
    residual_record: int,
    down: Any,
) -> tuple[Tensor, int, dict[str, Any]]:
    qinput_groups = final_qinput.reshape(-1, GROUP_LANES).to(torch.int64)
    qweight_groups = down.qweight.reshape(down.out_features, -1, GROUP_LANES).to(torch.int64)
    dots = torch.empty((qinput_groups.shape[0], down.out_features), dtype=torch.int64)
    for group in range(qinput_groups.shape[0]):
        dots[group] = (qweight_groups[:, group, :] * qinput_groups[group][None, :]).sum(dim=1)
    require(bool(torch.all(dots >= SIGNED32_MIN) and torch.all(dots <= SIGNED32_MAX)), "group dot exceeds signed-32")

    weight_scales = down.weight_scale.detach().cpu().to(torch.float64).tolist()
    accumulator_records = [
        [ceil_scale32_from_float(scale32_ratio(input_record)[0] / scale32_ratio(input_record)[1] * float(weight_scale)) for weight_scale in weight_scales]
        for input_record in input_records
    ]
    residual_sig, residual_exp = unpack_scale32(residual_record)
    dot_rows = dots.tolist()
    fused: list[tuple[int, int]] = []
    max_num, max_den = 0, 1
    max_abs_numerator = 0
    max_denominator = 0
    for lane in range(HIDDEN_SIZE):
        unpacked = [unpack_scale32(row[lane]) for row in accumulator_records]
        common_exp = min([residual_exp, *[value[1] for value in unpacked]])
        numerator = int(residual_s8[lane]) * residual_sig * (1 << (residual_exp - common_exp))
        for group, (significand, exponent) in enumerate(unpacked):
            numerator += int(dot_rows[group][lane]) * significand * (1 << (exponent - common_exp))
        denominator = 1 << (15 - common_exp)
        require(SIGNED96_MIN <= numerator <= SIGNED96_MAX, "fusion numerator exceeds signed-96")
        require(1 <= denominator <= UNSIGNED64_MAX, "fusion denominator exceeds unsigned-64")
        fused.append((numerator, denominator))
        max_abs_numerator = max(max_abs_numerator, abs(numerator))
        max_denominator = max(max_denominator, denominator)
        if abs(numerator) * max_den > max_num * denominator:
            max_num, max_den = abs(numerator), denominator
    require(max_num > 0, "exact fused result is all zero")
    destination_record = ceil_scale32_from_ratio(max_num, max_den * 127)
    destination_num, destination_den = scale32_ratio(destination_record)

    outputs: list[int] = []
    references: list[int] = []
    exact_values: list[float] = []
    for numerator, denominator in fused:
        impl = round_divide_even_signed(numerator * destination_den, denominator * destination_num)
        reference_fraction = Fraction(numerator, denominator) / Fraction(destination_num, destination_den)
        ref = scalar_rne(reference_fraction.numerator, reference_fraction.denominator)
        outputs.append(max(-128, min(127, impl)))
        references.append(max(-128, min(127, ref)))
        exact_values.append(numerator / denominator)
    require(outputs == references, "fusion differs from independent scalar integer reference")
    output = torch.tensor(outputs, dtype=torch.int8)
    scale = destination_num / destination_den
    exact = torch.tensor(exact_values, dtype=torch.float64)
    dequantized = output.to(torch.float64) * scale
    exact_norm = torch.linalg.vector_norm(exact)
    relative_error = float(torch.linalg.vector_norm(dequantized - exact) / exact_norm)
    error_bound = float(math.sqrt(HIDDEN_SIZE) * scale / 2.0 / exact_norm)
    require(relative_error <= error_bound + 1e-15, "fusion error exceeds half-LSB bound")
    return output, destination_record, {
        "group_dot_count": int(dots.numel()),
        "maximum_absolute_fusion_numerator": max_abs_numerator,
        "maximum_fusion_denominator": max_denominator,
        "destination_scale32_record": destination_record,
        "destination_scale": scale,
        "nonzero_fraction": float(torch.ne(output, 0).to(torch.float64).mean()),
        "positive_saturations": sum(value > 127 for value in outputs),
        "negative_saturations": sum(value < -128 for value in outputs),
        "relative_l2_error": relative_error,
        "relative_l2_half_lsb_bound": error_bound,
        "exact_scalar_reference_match": True,
    }


def rmsnorm_consume(hidden_s8: Tensor, scale32_record: int, gains: Tensor) -> tuple[Tensor, dict[str, Any]]:
    unpack_scale32(scale32_record)
    values = hidden_s8.to(torch.int64)
    sumsq = int((values * values).sum())
    require(0 <= sumsq < (1 << 48), "RMSNorm sumsq exceeds unsigned-48")
    mean_square = (sumsq + HIDDEN_SIZE // 2) // HIDDEN_SIZE
    root = max(1, math.isqrt(mean_square))
    if root * root < mean_square:
        root += 1
    inv_rms_q30 = (1 << 30) // root
    require(0 <= inv_rms_q30 < (1 << 31), "RMSNorm inverse exceeds unsigned-31")
    products = values * gains.to(torch.int64) * inv_rms_q30
    require(bool(torch.all(products >= -(1 << 63)) and torch.all(products <= (1 << 63) - 1)), "RMSNorm product exceeds signed-64")
    impl = fixed_point.round_shift_even(products, 38).clamp(-128, 127).to(torch.int8)
    reference = reference_rmsnorm(hidden_s8.tolist(), gains.tolist())
    require(impl.tolist() == reference.outputs, "RMSNorm differs from independent scalar integer reference")
    return impl, {
        "input_scale32_record": scale32_record,
        "sumsq_u48": sumsq,
        "mean_square": mean_square,
        "ceil_root": root,
        "inv_rms_q30": inv_rms_q30,
        "nonzero_fraction": float(torch.ne(impl, 0).to(torch.float64).mean()),
        "exact_scalar_reference_match": True,
        "saturation_seen": reference.saturation_seen,
    }


def run(contract_path: Path, output_path: Path) -> int:
    contract, contract_digest = load_bound_contract(contract_path)
    env = environment()
    source_contract = verify_source_contract()
    source_identity = verify_source_snapshot()
    scales_path = CALIBRATION_DIR / "derived_scales.json"
    scales = load_json(scales_path)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    model = reconstruct_fixed_model(scales).eval()
    layer2 = model.model.layers[2]
    layer3_gains = model.model.layers[3].input_layernorm.scaled_gains_q8.detach().cpu()
    residual_record = ceil_scale32_from_float(float(layer2.post_attention_scale))

    cases: list[dict[str, Any]] = []
    for case in PROMPTS:
        input_ids = chat_input(tokenizer, case.prompt, DEFAULT_SYSTEM)
        boundaries, _ = capture_first_layers(model, input_ids)
        hidden_raw = boundaries["model.layers.2.post_attention_layernorm.raw"]
        residual_real = boundaries["model.layers.2.post_attention_residual"][0, -1].to(torch.float64)
        residual_s8 = fixed_point.quantize_int8(residual_real, layer2.post_attention_scale).to(torch.int8)
        with torch.inference_mode():
            gate = layer2.mlp.gate_proj.forward_hardware_input(hidden_raw)
            up = layer2.mlp.up_proj.forward_hardware_input(hidden_raw)
        product = silu_product_q9_21(gate, up, layer2.mlp.gate_proj.output_scale, layer2.mlp.up_proj.output_scale)
        dynamic_raw, _dequantized, records, _scales = dynamic_group_requantize(product)
        fused, destination_record, fusion_trace = mixed_down_and_fuse(
            dynamic_raw[0, -1], records[-1], residual_s8, residual_record, layer2.mlp.down_proj
        )
        rms_output, rms_trace = rmsnorm_consume(fused, destination_record, layer3_gains)
        checks = {
            "prompt_hash_matches_predeclaration": sha256_bytes(case.prompt.encode()) == next(item["prompt_sha256"] for item in contract["prompt_binding"]["cases"] if item["case_id"] == case.case_id),
            "fusion_exact_scalar_reference_match": fusion_trace["exact_scalar_reference_match"],
            "fusion_nonzero": fusion_trace["nonzero_fraction"] > 0.0,
            "fusion_within_half_lsb_bound": fusion_trace["relative_l2_error"] <= fusion_trace["relative_l2_half_lsb_bound"] + 1e-15,
            "scale32_forwarded_unchanged": rms_trace["input_scale32_record"] == destination_record,
            "rmsnorm_exact_scalar_reference_match": rms_trace["exact_scalar_reference_match"],
            "rmsnorm_nonzero": rms_trace["nonzero_fraction"] > 0.0,
        }
        cases.append({
            "case_id": case.case_id,
            "prompt_sha256": sha256_bytes(case.prompt.encode()),
            "chat_template_token_count": int(input_ids.shape[1]),
            "fusion": fusion_trace,
            "layer3_input_rmsnorm": rms_trace,
            "output_sha256": sha256_bytes(bytes((int(value) & 0xFF) for value in rms_output.tolist())),
            "checks": checks,
            "passed": all(checks.values()),
        })
    passed = len(cases) == 9 and all(case["passed"] for case in cases)
    result = {
        "schema_version": 1,
        "classification": "offline_integer_reference_dynamic_scale32_layer2_fusion_to_layer3_rmsnorm",
        "status": "PASS_ALL_NINE_DYNAMIC_SCALE32_RECURRENT_BOUNDARY_INTEGER_REFERENCE" if passed else "FAIL_DYNAMIC_SCALE32_RECURRENT_BOUNDARY_INTEGER_REFERENCE",
        "predeclared_contract": {"artifact": file_record(contract_path), "contract_sha256": contract_digest},
        "environment": env,
        "model": source_identity,
        "source_contract_status": source_contract["status"],
        "cases": cases,
        "aggregate": {"case_count": len(cases), "passed_case_count": sum(case["passed"] for case in cases), "all_nine_passed": passed},
        "scope_guards": contract["containment"] | {"accelerator_executed": False, "network_access_performed": False},
        "artifacts": {"derived_scales": file_record(scales_path), "test_source": file_record(Path(__file__))},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_bytes(result))
    print(f"ACE2_LAYER2_SCALE32_RECURRENCE status={result['status']} passed={result['aggregate']['passed_case_count']}/9 output={output_path.relative_to(ROOT)}")
    return 0 if passed else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predeclare", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.predeclare is not None:
        require(args.contract is None and args.output is None, "predeclare mode is exclusive")
        write_contract(args.predeclare.resolve())
        print(f"ACE2_LAYER2_SCALE32_PREDECLARED output={args.predeclare.resolve().relative_to(ROOT)}")
        return 0
    require(args.contract is not None and args.output is not None, "run mode requires --contract and --output")
    return run(args.contract.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
