#!/usr/bin/env python3
"""Run the one predeclared layer-2 Dynamic Scale32 plus layer-23 per32 repair."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import ace2_full_model_fixed_point as fixed_point
from ace2_quality_contracts import (
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    scale32_ratio,
    unpack_scale32,
)
from diagnose_qwen_instruct_w4a8_divergence import (
    dynamic_group_requantize as scalar_dynamic_group_requantize,
    silu_product_q9_21,
)
from discriminate_qwen_instruct_w4_weight_policies import PROMPTS, chat_input
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    IMAGE_DIR,
    ROOT,
    SNAPSHOT,
    TERMINATION_TOKEN_IDS,
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
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM
from run_all_transformer_per32_stage1 import (
    GroupedScale32W4A8Linear,
    aggregate_prompts,
    decode_scale32,
    encode_scale32,
    generate_candidate,
    generate_reference,
    log_message,
    packed_w4,
    prompt_binding,
    require_project_python,
    scale32_bytes,
    scale32_self_test,
    sequence_disagreements,
    utc_now,
    write_json,
)
from test_qwen_instruct_layer2_scale32_recurrence import mixed_down_and_fuse


CANDIDATE_ID = "layer2_dyns32_layer23_down_per32_v1"
MISSION_ID = "stage1-layer2-dyns32-layer23-down-per32-v1-dispatch2"
MATRIX_SHA256 = "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
MATRIX_PATH = ROOT / "build/option-b-v2-weight-policy-20260806/full_sequence_discrimination.json"
RECURRENCE_CONTRACT_PATH = (
    ROOT / "build/option-b-layer2-scale32-recurrence-20260806/predeclared_contract.json"
)
RECURRENCE_RESULT_PATH = ROOT / "build/option-b-layer2-scale32-recurrence-20260806/results.json"
CONSUMED_GLOBAL_RESULT_PATH = (
    ROOT / "build/stage1-global-w4-per32-repair-v1/attempt-0001/results.json"
)
CONSUMED_CAUSAL_RESULT_PATH = (
    ROOT / "build/stage1-causal-four-w4-per32-repair-v1/attempt-0001/results.json"
)
OUTPUT_DIR = ROOT / "build/stage1-layer2-dyns32-layer23-down-per32-v1"
CONTRACT_PATH = OUTPUT_DIR / "predeclared_contract.json"
ATTEMPT_DIR = OUTPUT_DIR / "attempt-0001"
RUNNER_PATH = Path(__file__).resolve()
LAYER2_DOWN = "model.layers.2.mlp.down_proj"
LAYER23_DOWN = "model.layers.23.mlp.down_proj"
HIDDEN_SIZE = 896
INTERMEDIATE_SIZE = 4864
DYNAMIC_GROUP_LANES = 128
DYNAMIC_GROUP_COUNT = INTERMEDIATE_SIZE // DYNAMIC_GROUP_LANES
WEIGHT_GROUP_LANES = 32
WEIGHT_GROUP_COUNT = INTERMEDIATE_SIZE // WEIGHT_GROUP_LANES
MAX_NEW_TOKENS = 6
TORCH_THREADS = 16
INT64_MAX = (1 << 63) - 1


def layer23_target_spec() -> dict[str, Any]:
    return {
        "name": f"{LAYER23_DOWN}.weight",
        "module": LAYER23_DOWN,
        "shape": [HIDDEN_SIZE, INTERMEDIATE_SIZE],
        "input_group_lanes": WEIGHT_GROUP_LANES,
        "groups_per_output": WEIGHT_GROUP_COUNT,
        "scale32_record_count": HIDDEN_SIZE * WEIGHT_GROUP_COUNT,
    }


def candidate_cost() -> dict[str, Any]:
    target = layer23_target_spec()
    baseline_weight_scales = HIDDEN_SIZE
    candidate_weight_scales = target["scale32_record_count"]
    layer23_group_combines = candidate_weight_scales - baseline_weight_scales
    layer2_group_partials = HIDDEN_SIZE * DYNAMIC_GROUP_COUNT
    layer2_group_combines = layer2_group_partials - HIDDEN_SIZE
    weight_values = HIDDEN_SIZE * INTERMEDIATE_SIZE
    return {
        "changed_weight_tensor_count": 1,
        "changed_activation_boundary_count": 1,
        "unchanged_transformer_linear_tensor_count": 167,
        "candidate_weight_scale32_metadata_bytes": candidate_weight_scales * 4,
        "additional_weight_scale32_metadata_bytes": (
            candidate_weight_scales - baseline_weight_scales
        )
        * 4,
        "layer23_weight_group_partial_dot_products_per_model_token_position": (
            candidate_weight_scales
        ),
        "layer23_weight_group_combines_per_model_token_position": layer23_group_combines,
        "layer2_dynamic_input_scale32_records_per_model_token_position": DYNAMIC_GROUP_COUNT,
        "layer2_dynamic_destination_scale32_records_per_model_token_position": 1,
        "layer2_dynamic_scale32_metadata_bytes_per_model_token_position": (
            DYNAMIC_GROUP_COUNT + 1
        )
        * 4,
        "layer2_dynamic_group_partial_dot_products_per_model_token_position": (
            layer2_group_partials
        ),
        "layer2_dynamic_group_combines_per_model_token_position": layer2_group_combines,
        "group_combines_per_model_token_position": (
            layer23_group_combines + layer2_group_combines
        ),
        "layer23_changed_scope_w4_weight_values": weight_values,
        "layer23_changed_scope_packed_w4_payload_bytes": (weight_values + 1) // 2,
        "full_model_w4_payload_size_unchanged": True,
        "multiply_accumulate_count_ratio": 1.0,
    }


def contract_artifacts() -> dict[str, Any]:
    paths = {
        "matrix": MATRIX_PATH,
        "layer2_recurrence_contract": RECURRENCE_CONTRACT_PATH,
        "layer2_recurrence_result": RECURRENCE_RESULT_PATH,
        "consumed_global_attempt_result": CONSUMED_GLOBAL_RESULT_PATH,
        "consumed_causal_attempt_result": CONSUMED_CAUSAL_RESULT_PATH,
        "source_contract": ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json",
        "derived_scales": CALIBRATION_DIR / "derived_scales.json",
        "full_model_image": IMAGE_DIR / "full_model_image.bin",
        "image_contract": IMAGE_DIR / "image_contract_v2.json",
        "image_manifest": IMAGE_DIR / "manifest.json",
        "option_b_identity_manifest": IMAGE_DIR / "option_b_identity_manifest.json",
        "oracle_contract": IMAGE_DIR / "oracle_contract.json",
        "response_gate_contract": IMAGE_DIR / "response_gate_contract.json",
        "runner_source": RUNNER_PATH,
        "grouped_weight_arithmetic_source": ROOT / "tools/run_all_transformer_per32_stage1.py",
        "dynamic_recurrence_source": ROOT / "tools/test_qwen_instruct_layer2_scale32_recurrence.py",
        "dynamic_group_source": ROOT / "tools/diagnose_qwen_instruct_w4a8_divergence.py",
        "fixed_point_source": ROOT / "tools/ace2_full_model_fixed_point.py",
        "quality_contract_source": ROOT / "tools/ace2_quality_contracts.py",
        "prompt_and_matrix_evaluator_source": (
            ROOT / "tools/discriminate_qwen_instruct_w4_weight_policies.py"
        ),
        "response_gate_source": ROOT / "tools/qwen_instruct_response_gate.py",
        "oracle_source": ROOT / "tools/qwen_instruct_w4a8_oracle.py",
    }
    return {name: file_record(path) for name, path in paths.items()}


def scan_existing_candidate_results() -> list[str]:
    matches: list[str] = []
    for path in sorted((ROOT / "build").glob("**/attempt-*/results.json")):
        try:
            record = load_json(path)
        except Exception:
            continue
        if record.get("candidate_id") == CANDIDATE_ID:
            matches.append(path.relative_to(ROOT).as_posix())
    return matches


def verify_prerequisites() -> None:
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "immutable matrix hash differs")
    recurrence_wrapper = load_json(RECURRENCE_CONTRACT_PATH)
    recurrence_result = load_json(RECURRENCE_RESULT_PATH)
    require(
        recurrence_wrapper.get("contract", {}).get("contract_id")
        == "qwen-instruct-layer2-dynamic-scale32-to-layer3-rmsnorm-v1",
        "layer-2 recurrence contract differs",
    )
    require(
        recurrence_result.get("aggregate", {}).get("all_nine_passed") is True,
        "layer-2 recurrence result is not 9/9 PASS",
    )
    consumed_global = load_json(CONSUMED_GLOBAL_RESULT_PATH)
    consumed_causal = load_json(CONSUMED_CAUSAL_RESULT_PATH)
    require(
        consumed_global.get("candidate_id") == "all_transformer_per_32_v1"
        and consumed_global.get("eligibility", {}).get("eligible") is False,
        "consumed global candidate binding differs",
    )
    require(
        consumed_causal.get("candidate_id") == "causal_four_per_32_v1"
        and consumed_causal.get("eligibility", {}).get("eligible") is False,
        "consumed causal candidate binding differs",
    )


def prepare_contract() -> dict[str, Any]:
    require_project_python()
    require(not CONTRACT_PATH.exists(), "immutable predeclared contract already exists")
    require(not ATTEMPT_DIR.exists(), "attempt-0001 already exists")
    require(not scan_existing_candidate_results(), "a qualifying candidate result already exists")
    verify_prerequisites()
    verify_source_contract()
    source_identity = verify_source_snapshot()
    versions = verify_versions()
    matrix = load_json(MATRIX_PATH)
    prompts = prompt_binding(matrix)
    target = layer23_target_spec()
    cost = candidate_cost()
    contract = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "hypothesis": (
            "The independently checked layer-2 Dynamic Scale32 recurrence repairs the dominant "
            "SiLU/down-projection/residual activation boundary while isolated per-32 W4 only at "
            "layer-23 down_proj repairs a later weight-error source without replaying broader policies."
        ),
        "matrix": {
            "path": MATRIX_PATH.relative_to(ROOT).as_posix(),
            "sha256": MATRIX_SHA256,
            "case_count": 9,
            "full_generated_sequences_bound": True,
        },
        "exact_numerical_delta": {
            "layer2_dynamic_boundary": {
                "producer": "model.layers.2.mlp SiLU(gate_proj) multiplied by up_proj in signed Q9.21",
                "grouping": {
                    "axis": "intermediate_feature",
                    "group_lanes": DYNAMIC_GROUP_LANES,
                    "group_count": DYNAMIC_GROUP_COUNT,
                    "scope": "every token position in every official candidate model invocation",
                },
                "activation_quantization": (
                    "one normalized Scale32 per token and contiguous 128-lane group; select the "
                    "smallest legal Scale32 not below exact group absmax/127; round ties-to-even "
                    "to signed A8 symmetric [-127,127]"
                ),
                "down_projection_weights": (
                    "unchanged authenticated per-output-row signed-W4 qweight and scale for "
                    "model.layers.2.mlp.down_proj"
                ),
                "down_projection_accumulation": (
                    "38 signed-32 A8xW4 group dots per output lane; ceil-encode each dynamic "
                    "activation Scale32 times the unchanged row W4 scale; exact exponent "
                    "alignment with no intermediate rounding"
                ),
                "residual": (
                    "unchanged layer-2 post-attention residual quantized with its frozen static "
                    "Scale32 and aligned into the same exact fusion"
                ),
                "destination": (
                    "one fresh normalized Scale32 per token from exact fused absmax/127; one "
                    "final ties-to-even division and signed-A8 saturation"
                ),
                "consumer": (
                    "the exact fused signed-A8 tensor and destination Scale32 are forwarded "
                    "unchanged to model.layers.3.input_layernorm; the scalar scale cancels in "
                    "integer RMS normalization"
                ),
                "termination": (
                    "Dynamic Scale32 ends at the layer-3 input RMSNorm consumption; subsequent "
                    "activation policies remain the authenticated baseline"
                ),
                "authoritative_reference_contract": file_record(RECURRENCE_CONTRACT_PATH),
                "authoritative_reference_result": file_record(RECURRENCE_RESULT_PATH),
            },
            "layer23_weight_tensor": target,
            "layer23_weight_policy": {
                "weight_bits": 4,
                "signed_integer_range": [-8, 7],
                "group_axis": "input_feature",
                "input_group_lanes": WEIGHT_GROUP_LANES,
                "scale_scope": (
                    "one normalized Scale32 per output row and contiguous 32-input-lane group "
                    "only for model.layers.23.mlp.down_proj.weight"
                ),
                "scale_selection": (
                    "ceil-encode max(abs(BF16 group))/7; all-zero groups use the canonical "
                    "all-zero Scale32 record"
                ),
                "rounding": "torch.round round-to-nearest ties-to-even",
                "saturation": "signed W4 clamp [-8,7]; final signed A8 clamp [-128,127]",
                "packing": (
                    "row-major output row then input lane; signed-int4 two's complement; even "
                    "flat element in low nibble and odd flat element in high nibble"
                ),
            },
            "unchanged_scope": {
                "layer2_gate_up_down_weight_policy": "authenticated per-output-row W4 baseline",
                "other_167_transformer_weight_tensors": (
                    "authenticated per-output-row W4 baseline except the one named layer-23 tensor"
                ),
                "lm_head_weight_policy": "authenticated per-output-row W4 baseline",
                "rope_attention_and_other_activation_policies": "authenticated baseline unchanged",
                "prompt_gate_and_decode_policy": "frozen unchanged",
            },
        },
        "distinctness": {
            "consumed_candidates": [
                {
                    "candidate_id": "all_transformer_per_32_v1",
                    "scope": "168 transformer W4 tensors per-32; no layer-2 dynamic recurrence",
                    "result": file_record(CONSUMED_GLOBAL_RESULT_PATH),
                },
                {
                    "candidate_id": "causal_four_per_32_v1",
                    "scope": (
                        "four W4 tensors per-32 at layer-2/layer-23 gate/down; no layer-2 "
                        "dynamic recurrence"
                    ),
                    "result": file_record(CONSUMED_CAUSAL_RESULT_PATH),
                },
            ],
            "current_weight_scope_count": 1,
            "current_dynamic_boundary_count": 1,
            "replay": False,
            "scope_identical_to_consumed_candidate": False,
        },
        "generation_and_evaluator": {
            "system_prompt_utf8_bytes": len(DEFAULT_SYSTEM.encode()),
            "system_prompt_sha256": sha256_bytes(DEFAULT_SYSTEM.encode()),
            "chat_template_sha256": source_identity["chat_template_sha256"],
            "greedy_argmax": True,
            "sampling": False,
            "use_cache": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "termination_token_ids": list(TERMINATION_TOKEN_IDS),
            "greedy_tie_rule": "torch.argmax; exact ties choose the lowest token id",
            "response_gate": "frozen qwen-instruct-option-b-response-gate-v1",
            "aligned_logit_error": (
                "full-vocabulary candidate versus recomputed BF16 logits while the incoming "
                "generated prefix remains BF16-aligned, including the first mismatch step"
            ),
            "prompts": prompts,
        },
        "eligibility": {
            "rule": (
                "all nine complete generated token sequences exactly equal BF16 and all nine "
                "response-gate outcomes equal BF16"
            ),
            "score_policy": "binary only; 9/9 exact plus 9/9 gate-outcome agreement",
            "aggregate_improvement_is_insufficient": True,
            "local_arithmetic_agreement_is_insufficient": True,
            "selected_policy_id_before_execution": None,
            "fresh_reviewer_required": True,
        },
        "execution": {
            "exact_authorized_candidate_attempts": 1,
            "attempt_id": "attempt-0001",
            "working_directory": ROOT.as_posix(),
            "exact_command": [
                "./.venv/bin/python",
                "tools/run_layer2_dyns32_layer23_down_per32_stage1.py",
                "execute",
            ],
            "expected_outputs": [
                ATTEMPT_DIR.relative_to(ROOT).as_posix() + "/execution_started.json",
                ATTEMPT_DIR.relative_to(ROOT).as_posix() + "/run.log",
                ATTEMPT_DIR.relative_to(ROOT).as_posix() + "/results.json",
                ATTEMPT_DIR.relative_to(ROOT).as_posix() + "/fresh_reviewer_input.json",
                ATTEMPT_DIR.relative_to(ROOT).as_posix() + "/SHA256SUMS",
            ],
            "torch_num_threads": TORCH_THREADS,
            "network_access_permitted": False,
            "alternate_policy_permitted": False,
            "tuning_ranking_bisection_sweep_permitted": False,
            "selector_or_gate_relaxation_permitted": False,
            "rtl_demo_ppa_u280_stage2_mutation_permitted": False,
        },
        "model": source_identity,
        "image_binding": {
            "image_id": "build-qwen2.5-0.5b-instruct-w4a8-image-v1",
            "full_model_image": file_record(IMAGE_DIR / "full_model_image.bin"),
            "manifest": file_record(IMAGE_DIR / "manifest.json"),
            "option_b_identity_manifest": file_record(IMAGE_DIR / "option_b_identity_manifest.json"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {**versions, "numpy": importlib.metadata.version("numpy")},
        },
        "static_cost": cost,
        "artifacts": contract_artifacts(),
        "no_outcome_assumption": (
            "No token, response-gate, eligibility, demo, RTL, synthesis, timing, PPA, FPGA, "
            "U280, or stage-transition outcome is assumed before the single execution."
        ),
    }
    wrapper = {"contract": contract, "contract_sha256": sha256_bytes(canonical_bytes(contract))}
    write_json(CONTRACT_PATH, wrapper)
    return wrapper


def vector_dynamic_group_requantize(product_q9_21: Tensor) -> tuple[Tensor, Tensor]:
    require(product_q9_21.ndim == 3 and product_q9_21.shape[0] == 1, "dynamic input must be batch-1 rank-3")
    require(
        product_q9_21.shape[-1] == INTERMEDIATE_SIZE,
        "dynamic layer-2 intermediate width differs",
    )
    groups = product_q9_21.reshape(1, product_q9_21.shape[1], DYNAMIC_GROUP_COUNT, DYNAMIC_GROUP_LANES)
    real = groups.to(torch.float64) / float(1 << 21)
    absmax = real.abs().amax(dim=-1)
    zero = absmax == 0
    ideal = torch.where(zero, torch.ones_like(absmax), absmax / 127.0)
    records = encode_scale32(ideal, zero)
    scales = decode_scale32(records)
    quantized = torch.round(real / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return quantized.reshape_as(product_q9_21), records


def dynamic_group_self_test() -> None:
    source = ((torch.arange(2 * INTERMEDIATE_SIZE, dtype=torch.int64) * 7919) % 200001) - 100000
    source = source.reshape(1, 2, INTERMEDIATE_SIZE)
    source[:, 0, :DYNAMIC_GROUP_LANES] = 0
    observed_raw, observed_records = vector_dynamic_group_requantize(source)
    expected_raw, _dequantized, expected_records, _scales = scalar_dynamic_group_requantize(source)
    require(torch.equal(observed_raw, expected_raw), "vector dynamic quantization differs from frozen reference")
    require(
        observed_records[0].tolist() == expected_records,
        "vector dynamic Scale32 records differ from frozen reference",
    )


def unpack_record_tensor(records: Tensor) -> tuple[Tensor, Tensor]:
    record = records.to(torch.int64)
    require(bool(torch.all((record >> 24) == 0)), "Scale32 reserved byte is nonzero")
    significand = record & 0xFFFF
    exponent_u8 = (record >> 16) & 0xFF
    exponent = torch.where(exponent_u8 >= 128, exponent_u8 - 256, exponent_u8)
    require(
        bool(torch.all((significand >= 0x8000) & (significand <= 0xFFFF))),
        "Scale32 significand differs",
    )
    require(bool(torch.all((exponent >= -24) & (exponent <= 4))), "Scale32 exponent differs")
    return significand, exponent


def exact_destination_records(numerators: Tensor, denominators: Tensor) -> Tensor:
    records: list[int] = []
    for numerator_row, denominator_row in zip(
        numerators.detach().cpu().tolist(), denominators.detach().cpu().tolist(), strict=True
    ):
        max_num = 0
        max_den = 1
        for numerator, denominator in zip(numerator_row, denominator_row, strict=True):
            magnitude = abs(int(numerator))
            denominator_int = int(denominator)
            if magnitude * max_den > max_num * denominator_int:
                max_num = magnitude
                max_den = denominator_int
        require(max_num > 0, "dynamic fused token is all zero")
        records.append(ceil_scale32_from_ratio(max_num, max_den * 127))
    return torch.tensor(records, dtype=torch.int64, device=numerators.device)


def round_ratio_even_tensor(numerators: Tensor, denominators: Tensor) -> Tensor:
    require(bool(torch.all(denominators > 0)), "dynamic fusion denominator is non-positive")
    magnitudes = numerators.abs()
    quotient = torch.div(magnitudes, denominators, rounding_mode="floor")
    remainder = magnitudes - quotient * denominators
    doubled = remainder * 2
    rounded = quotient + ((doubled > denominators) | ((doubled == denominators) & ((quotient & 1) != 0)))
    return torch.where(numerators < 0, -rounded, rounded)


def dynamic_down_and_fuse(
    qinput: Tensor,
    input_records: Tensor,
    residual_s8: Tensor,
    residual_record: int,
    down: fixed_point.W4A8Linear,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    require(qinput.shape[:2] == residual_s8.shape[:2], "dynamic residual token geometry differs")
    require(qinput.shape[-1] == INTERMEDIATE_SIZE, "dynamic qinput width differs")
    require(residual_s8.shape[-1] == HIDDEN_SIZE, "dynamic residual width differs")
    require(input_records.shape == (*qinput.shape[:2], DYNAMIC_GROUP_COUNT), "dynamic Scale32 shape differs")
    require(down.in_features == INTERMEDIATE_SIZE and down.out_features == HIDDEN_SIZE, "layer-2 down geometry differs")
    require(down.bias_accumulator is None, "layer-2 down bias is outside the frozen recurrence")

    sequence = qinput.shape[1]
    qinput_groups = qinput.reshape(sequence, DYNAMIC_GROUP_COUNT, DYNAMIC_GROUP_LANES)
    qweight_groups = down.qweight.reshape(HIDDEN_SIZE, DYNAMIC_GROUP_COUNT, DYNAMIC_GROUP_LANES)
    dot_groups: list[Tensor] = []
    for group in range(DYNAMIC_GROUP_COUNT):
        dot = torch._int_mm(
            qinput_groups[:, group, :].contiguous(),
            qweight_groups[:, group, :].transpose(0, 1).contiguous(),
        ).to(torch.int64)
        dot_groups.append(dot)
    dots = torch.stack(dot_groups, dim=1)
    require(
        bool(torch.all(dots >= -(1 << 31)) and torch.all(dots <= (1 << 31) - 1)),
        "dynamic group dot exceeds signed-32",
    )

    input_scales = decode_scale32(input_records.reshape(sequence, DYNAMIC_GROUP_COUNT))
    product_records = encode_scale32(
        input_scales[:, :, None] * down.weight_scale.detach().to(torch.float64)[None, None, :]
    )
    product_sig, product_exp = unpack_record_tensor(product_records)
    residual_sig, residual_exp = unpack_scale32(residual_record)
    common_exp = torch.minimum(
        product_exp.amin(dim=1),
        torch.full((sequence, HIDDEN_SIZE), residual_exp, dtype=torch.int64),
    )
    shifts = product_exp - common_exp[:, None, :]
    residual_shifts = residual_exp - common_exp
    require(bool(torch.all((shifts >= 0) & (shifts <= 28))), "dynamic product alignment shift differs")
    require(
        bool(torch.all((residual_shifts >= 0) & (residual_shifts <= 28))),
        "dynamic residual alignment shift differs",
    )
    powers = torch.bitwise_left_shift(torch.ones_like(shifts), shifts)
    residual_powers = torch.bitwise_left_shift(torch.ones_like(residual_shifts), residual_shifts)
    safe_product = torch.div(INT64_MAX, product_sig * powers, rounding_mode="floor")
    require(bool(torch.all(dots.abs() <= safe_product)), "dynamic group term exceeds signed-64")
    terms = dots * product_sig * powers
    residual_flat = residual_s8.reshape(sequence, HIDDEN_SIZE).to(torch.int64)
    safe_residual = torch.div(
        INT64_MAX,
        torch.full_like(residual_powers, residual_sig) * residual_powers,
        rounding_mode="floor",
    )
    require(bool(torch.all(residual_flat.abs() <= safe_residual)), "dynamic residual term exceeds signed-64")
    residual_terms = residual_flat * residual_sig * residual_powers
    absolute_bound = terms.abs().to(torch.float64).sum(dim=1) + residual_terms.abs().to(torch.float64)
    require(bool(torch.all(absolute_bound <= float(INT64_MAX))), "dynamic fusion sum exceeds signed-64 implementation")
    numerators = terms.sum(dim=1) + residual_terms
    denominators = torch.bitwise_left_shift(
        torch.ones_like(common_exp),
        15 - common_exp,
    )
    require(bool(torch.all((denominators >= 1) & (denominators <= (1 << 39)))), "dynamic fusion denominator differs")

    destination_records = exact_destination_records(numerators, denominators)
    destination_sig, destination_exp = unpack_record_tensor(destination_records)
    destination_den = torch.bitwise_left_shift(
        torch.ones_like(destination_exp),
        15 - destination_exp,
    )
    scaled_numerator_bound = numerators.abs().to(torch.float64) * destination_den[:, None].to(torch.float64)
    scaled_denominator_bound = denominators.to(torch.float64) * destination_sig[:, None].to(torch.float64)
    require(bool(torch.all(scaled_numerator_bound <= float(INT64_MAX))), "dynamic final numerator exceeds signed-64")
    require(bool(torch.all(scaled_denominator_bound <= float(INT64_MAX))), "dynamic final denominator exceeds signed-64")
    scaled_numerators = numerators * destination_den[:, None]
    scaled_denominators = denominators * destination_sig[:, None]
    rounded = round_ratio_even_tensor(scaled_numerators, scaled_denominators)
    positive_saturations = int(torch.sum(rounded > 127))
    negative_saturations = int(torch.sum(rounded < -128))
    output = rounded.clamp(-128, 127).to(torch.int8).reshape_as(residual_s8)
    trace = {
        "token_count": sequence,
        "input_scale32_record_count": sequence * DYNAMIC_GROUP_COUNT,
        "destination_scale32_record_count": sequence,
        "group_dot_count": int(dots.numel()),
        "positive_saturations": positive_saturations,
        "negative_saturations": negative_saturations,
        "maximum_absolute_fusion_numerator": int(numerators.abs().max()),
        "maximum_fusion_denominator": int(denominators.max()),
        "minimum_input_scale32_record": int(input_records.min()),
        "maximum_input_scale32_record": int(input_records.max()),
        "minimum_destination_scale32_record": int(destination_records.min()),
        "maximum_destination_scale32_record": int(destination_records.max()),
    }
    return output, destination_records.reshape(1, sequence), trace


def layer2_boundary_self_test(down: fixed_point.W4A8Linear) -> dict[str, Any]:
    values = (((torch.arange(INTERMEDIATE_SIZE, dtype=torch.int64) * 37) % 255) - 127).to(torch.int8)
    residual = (((torch.arange(HIDDEN_SIZE, dtype=torch.int64) * 19) % 255) - 127).to(torch.int8)
    records = [ceil_scale32_from_float(0.0008 + group * 0.00001) for group in range(DYNAMIC_GROUP_COUNT)]
    observed, observed_destination, trace = dynamic_down_and_fuse(
        values.reshape(1, 1, -1),
        torch.tensor(records, dtype=torch.int64).reshape(1, 1, -1),
        residual.reshape(1, 1, -1),
        ceil_scale32_from_float(0.03),
        down,
    )
    expected, expected_destination, _expected_trace = mixed_down_and_fuse(
        values,
        records,
        residual,
        ceil_scale32_from_float(0.03),
        down,
    )
    require(torch.equal(observed[0, 0], expected), "vector layer-2 fusion differs from scalar recurrence")
    require(
        int(observed_destination[0, 0]) == expected_destination,
        "vector layer-2 destination Scale32 differs from scalar recurrence",
    )
    return {
        "status": "PASS",
        "output_sha256": hashlib.sha256(observed.detach().cpu().numpy().tobytes()).hexdigest(),
        "destination_scale32_record": expected_destination,
        "group_dot_count": trace["group_dot_count"],
    }


class DynamicLayer2BoundaryRuntime:
    def __init__(self) -> None:
        self.pending: tuple[Tensor, Tensor, Tensor] | None = None
        self.producer_count = 0
        self.consumer_count = 0
        self.model_token_positions = 0
        self.input_scale32_record_count = 0
        self.destination_scale32_record_count = 0
        self.group_dot_count = 0
        self.positive_saturations = 0
        self.negative_saturations = 0
        self.maximum_absolute_fusion_numerator = 0
        self.maximum_fusion_denominator = 0
        self._input_record_hash = hashlib.sha256()
        self._destination_record_hash = hashlib.sha256()
        self._fused_q8_hash = hashlib.sha256()

    def publish(self, fused_q8: Tensor, destination_records: Tensor, trace: dict[str, Any]) -> Tensor:
        require(self.pending is None, "dynamic layer-2 boundary has unconsumed output")
        require(
            destination_records.shape == fused_q8.shape[:-1],
            "dynamic destination Scale32 geometry differs",
        )
        scales = decode_scale32(destination_records).to(fused_q8.device)
        dequantized = fused_q8.to(torch.float64) * scales.unsqueeze(-1)
        published = dequantized.to(torch.bfloat16)
        self.pending = (fused_q8.detach().clone(), destination_records.detach().clone(), published.detach().clone())
        self.producer_count += 1
        self.model_token_positions += trace["token_count"]
        self.input_scale32_record_count += trace["input_scale32_record_count"]
        self.destination_scale32_record_count += trace["destination_scale32_record_count"]
        self.group_dot_count += trace["group_dot_count"]
        self.positive_saturations += trace["positive_saturations"]
        self.negative_saturations += trace["negative_saturations"]
        self.maximum_absolute_fusion_numerator = max(
            self.maximum_absolute_fusion_numerator,
            trace["maximum_absolute_fusion_numerator"],
        )
        self.maximum_fusion_denominator = max(
            self.maximum_fusion_denominator,
            trace["maximum_fusion_denominator"],
        )
        self._destination_record_hash.update(
            destination_records.detach().cpu().to(torch.int64).numpy().tobytes()
        )
        self._fused_q8_hash.update(fused_q8.detach().cpu().numpy().tobytes())
        return published

    def record_input_scales(self, input_records: Tensor) -> None:
        self._input_record_hash.update(
            input_records.detach().cpu().to(torch.int64).numpy().tobytes()
        )

    def consume(self, hidden_states: Tensor, gains_q8: Tensor) -> Tensor:
        require(self.pending is not None, "layer-3 RMSNorm lacks dynamic layer-2 output")
        fused_q8, _records, published = self.pending
        require(hidden_states.shape == fused_q8.shape, "dynamic layer-3 RMSNorm geometry differs")
        require(torch.equal(hidden_states, published), "dynamic layer-2 publication changed before RMSNorm")
        raw = fixed_point.fixed_rmsnorm_raw(fused_q8, gains_q8)
        self.pending = None
        self.consumer_count += 1
        return raw

    def summary(self) -> dict[str, Any]:
        return {
            "producer_count": self.producer_count,
            "consumer_count": self.consumer_count,
            "pending_output": self.pending is not None,
            "model_token_positions": self.model_token_positions,
            "input_scale32_record_count": self.input_scale32_record_count,
            "destination_scale32_record_count": self.destination_scale32_record_count,
            "group_dot_count": self.group_dot_count,
            "positive_saturations": self.positive_saturations,
            "negative_saturations": self.negative_saturations,
            "maximum_absolute_fusion_numerator": self.maximum_absolute_fusion_numerator,
            "maximum_fusion_denominator": self.maximum_fusion_denominator,
            "input_scale32_records_sha256": self._input_record_hash.hexdigest(),
            "destination_scale32_records_sha256": self._destination_record_hash.hexdigest(),
            "fused_q8_sha256": self._fused_q8_hash.hexdigest(),
        }


class DynamicLayer3InputRMSNorm(nn.Module):
    def __init__(self, source: fixed_point.FixedRMSNorm, runtime: DynamicLayer2BoundaryRuntime) -> None:
        super().__init__()
        require(isinstance(source, fixed_point.FixedRMSNorm), "layer-3 input norm is not fixed RMSNorm")
        self.input_scale = source.input_scale
        self.output_scale = source.output_scale
        self.runtime = runtime
        self.register_buffer(
            "scaled_gains_q8",
            source.scaled_gains_q8.detach().clone(),
            persistent=True,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.runtime.consume(hidden_states, self.scaled_gains_q8).to(hidden_states.dtype)


class DynamicLayer2MLP(nn.Module):
    def __init__(self, source: fixed_point.FixedMLP, runtime: DynamicLayer2BoundaryRuntime) -> None:
        super().__init__()
        require(isinstance(source, fixed_point.FixedMLP), "layer-2 MLP is not fixed")
        require(
            isinstance(source.down_proj, fixed_point.W4A8Linear)
            and not isinstance(source.down_proj, GroupedScale32W4A8Linear),
            "layer-2 down projection is not the unchanged per-row W4 baseline",
        )
        self.gate_proj = source.gate_proj
        self.up_proj = source.up_proj
        self.down_proj = source.down_proj
        self.runtime = runtime

    def forward_fused(self, hidden_states: Tensor, residual: Tensor, residual_scale: float) -> Tensor:
        gate = self.gate_proj.forward_hardware_input(hidden_states)
        up = self.up_proj.forward_hardware_input(hidden_states)
        product = silu_product_q9_21(
            gate,
            up,
            self.gate_proj.output_scale,
            self.up_proj.output_scale,
        )
        qinput, input_records = vector_dynamic_group_requantize(product)
        self.runtime.record_input_scales(input_records)
        residual_s8 = fixed_point.quantize_int8(residual, residual_scale).to(torch.int8)
        fused_q8, destination_records, trace = dynamic_down_and_fuse(
            qinput,
            input_records,
            residual_s8,
            ceil_scale32_from_float(residual_scale),
            self.down_proj,
        )
        return self.runtime.publish(fused_q8, destination_records, trace)

    def forward(self, hidden_states: Tensor) -> Tensor:
        raise RuntimeError("dynamic layer-2 MLP requires the residual-fusion boundary")


class DynamicLayer2Decoder(nn.Module):
    def __init__(self, source: fixed_point.FixedDecoderLayer, runtime: DynamicLayer2BoundaryRuntime) -> None:
        super().__init__()
        require(isinstance(source, fixed_point.FixedDecoderLayer), "layer-2 decoder is not fixed")
        require(source.all_projection_residual_fusion_runtime is None, "unexpected all-projection fusion")
        require(source.down_projection_fusion_runtime is None, "unexpected legacy down fusion")
        require(source.cross_layer_error_carry_runtime is None, "unexpected cross-layer carry")
        self.attention_type = source.attention_type
        self.input_scale = source.input_scale
        self.input_layernorm = source.input_layernorm
        self.post_attention_layernorm = source.post_attention_layernorm
        self.self_attn = source.self_attn
        self.mlp = DynamicLayer2MLP(source.mlp, runtime)
        self.post_attention_scale = source.post_attention_scale
        self.post_mlp_scale = source.post_mlp_scale

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: Any = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        require(position_embeddings is not None, "position embeddings are required")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = fixed_point.fixed_residual_add(
            residual,
            hidden_states,
            self.post_attention_scale,
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return self.mlp.forward_fused(hidden_states, residual, self.post_attention_scale)


def load_verified_contract(*, require_unconsumed: bool) -> dict[str, Any]:
    require_project_python()
    require(CONTRACT_PATH.is_file(), "predeclared contract is missing")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "contract wrapper differs")
    contract = wrapper["contract"]
    require(
        sha256_bytes(canonical_bytes(contract)) == wrapper["contract_sha256"],
        "contract digest differs",
    )
    require(contract.get("candidate_id") == CANDIDATE_ID, "candidate id differs")
    require(contract.get("matrix", {}).get("sha256") == MATRIX_SHA256, "matrix binding differs")
    require(
        contract.get("exact_numerical_delta", {}).get("layer23_weight_tensor")
        == layer23_target_spec(),
        "layer-23 tensor scope differs",
    )
    require(
        contract.get("execution", {}).get("exact_authorized_candidate_attempts") == 1,
        "attempt budget differs",
    )
    for record in contract.get("artifacts", {}).values():
        path = ROOT / record["path"]
        require(path.is_file(), f"contract artifact is missing: {record['path']}")
        require(path.stat().st_size == record["bytes"], f"artifact size differs: {record['path']}")
        require(sha256_file(path) == record["sha256"], f"artifact hash differs: {record['path']}")
    verify_prerequisites()
    verify_source_contract()
    verify_source_snapshot()
    verify_versions()
    matrix = load_json(MATRIX_PATH)
    require(
        prompt_binding(matrix) == contract["generation_and_evaluator"]["prompts"],
        "prompt binding differs",
    )
    attempts = sorted(OUTPUT_DIR.glob("attempt-*"))
    if require_unconsumed:
        require(not attempts, f"exactly-once authorization is consumed: {[path.name for path in attempts]}")
        require(not scan_existing_candidate_results(), "a qualifying candidate result already exists")
    probe = OUTPUT_DIR / ".preflight-write-probe"
    probe.write_bytes(b"preflight\n")
    require(probe.read_bytes() == b"preflight\n", "output path readback differs")
    probe.unlink()
    scale32_self_test()
    dynamic_group_self_test()
    return wrapper


def replace_candidate_linears(
    model: nn.Module,
    ranges: dict[str, fixed_point.CalibrationRange],
) -> list[str]:
    replacements = [
        (name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)
    ]
    require(len(replacements) == 169, f"expected 169 linears, found {len(replacements)}")
    grouped_scope: list[str] = []
    for name, module in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        use_per_head_output_scales, emit_scale32 = fixed_point.rope_linear_contract(
            name, fixed_point.ACTIVE_ROPE_MECHANISM
        )
        frozen_output_head_absmax = fixed_point.qk_residual_projection_output_absmax(
            name, fixed_point.ACTIVE_ROPE_MECHANISM
        )
        common = {
            "input_is_quantized": False,
            "output_head_absmax": (
                frozen_output_head_absmax
                if frozen_output_head_absmax is not None
                else ranges[name].output_head_absmax
                if use_per_head_output_scales
                else None
            ),
            "output_head_size": fixed_point.ROPE_HEAD_DIM if use_per_head_output_scales else None,
            "scale_cap": None,
            "use_percentile_scale": False,
            "scale32_output": emit_scale32,
        }
        if name == LAYER23_DOWN:
            replacement: nn.Module = GroupedScale32W4A8Linear(
                module,
                ranges[name],
                module_name=name,
                **common,
            )
            grouped_scope.append(name)
        else:
            replacement = fixed_point.W4A8Linear(module, ranges[name], **common)
        setattr(parent, child_name, replacement)
    require(grouped_scope == [LAYER23_DOWN], "live grouped W4 scope differs")
    return grouped_scope


def reconstruct_candidate_model(
    scales: dict[str, Any],
) -> tuple[nn.Module, DynamicLayer2BoundaryRuntime, dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    ranges: dict[str, fixed_point.CalibrationRange] = {}
    for name, metadata in scales["linears"].items():
        head_scales = metadata.get("output_head_scales") or []
        ranges[name] = fixed_point.CalibrationRange(
            input_absmax=float(metadata["input_absmax"]),
            output_absmax=float(metadata["output_absmax"]),
            output_head_absmax=[float(value) * 127.0 for value in head_scales] or None,
        )
    operators = {
        name: fixed_point.ObservedRange(absmax=float(metadata["absmax"]))
        for name, metadata in scales["operators"].items()
    }
    grouped_scope = replace_candidate_linears(model, ranges)
    fixed_point.replace_fixed_operators(
        model,
        operators,
        rope_diagnostic_mechanism=fixed_point.ACTIVE_ROPE_MECHANISM,
    )
    runtime = DynamicLayer2BoundaryRuntime()
    layer2 = model.model.layers[2]
    layer3 = model.model.layers[3]
    require(isinstance(layer2, fixed_point.FixedDecoderLayer), "fixed layer-2 reconstruction differs")
    require(isinstance(layer3, fixed_point.FixedDecoderLayer), "fixed layer-3 reconstruction differs")
    self_test = layer2_boundary_self_test(layer2.mlp.down_proj)
    model.model.layers[2] = DynamicLayer2Decoder(layer2, runtime)
    layer3.input_layernorm = DynamicLayer3InputRMSNorm(layer3.input_layernorm, runtime)
    grouped_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, GroupedScale32W4A8Linear)
    ]
    require(grouped_modules == grouped_scope == [LAYER23_DOWN], "post-replacement grouped scope differs")
    require(
        isinstance(model.model.layers[2].mlp.down_proj, fixed_point.W4A8Linear)
        and not isinstance(model.model.layers[2].mlp.down_proj, GroupedScale32W4A8Linear),
        "layer-2 down W4 baseline changed",
    )
    model.ace2_layer2_dynamic_scale32_runtime = runtime
    return model, runtime, self_test


def tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def candidate_manifest(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    modules = dict(model.named_modules())
    layer23 = modules.get(LAYER23_DOWN)
    layer2 = modules.get(LAYER2_DOWN)
    require(isinstance(layer23, GroupedScale32W4A8Linear), "layer-23 grouped target is missing")
    require(
        isinstance(layer2, fixed_point.W4A8Linear)
        and not isinstance(layer2, GroupedScale32W4A8Linear),
        "layer-2 baseline target changed",
    )
    spec = layer23_target_spec()
    packed = packed_w4(layer23)
    metadata = scale32_bytes(layer23)
    records = [
        {
            **spec,
            "qweight_s8_container_sha256": tensor_sha256(layer23.qweight),
            "packed_w4_bytes": len(packed),
            "packed_w4_sha256": hashlib.sha256(packed).hexdigest(),
            "weight_scale32_metadata_bytes": len(metadata),
            "weight_scale32_sha256": hashlib.sha256(metadata).hexdigest(),
            "weight_error": {
                "relative_l2_error": layer23.grouped_record.weight_error_relative_l2,
                "maximum_absolute_error": layer23.grouped_record.weight_error_maximum_absolute,
            },
        }
    ]
    baseline = {
        "module": LAYER2_DOWN,
        "policy": "unchanged per-output-row W4 baseline",
        "qweight_sha256": tensor_sha256(layer2.qweight),
        "weight_scale_sha256": tensor_sha256(layer2.weight_scale),
        "native_scale32_records_sha256": tensor_sha256(layer2.native_scale32_records),
    }
    hashes = {
        "layer23_packed_w4_sha256": hashlib.sha256(packed).hexdigest(),
        "layer23_weight_scale32_sha256": hashlib.sha256(metadata).hexdigest(),
        "layer23_tensor_manifest_sha256": sha256_bytes(canonical_bytes(records)),
        "layer2_baseline_manifest_sha256": sha256_bytes(canonical_bytes(baseline)),
    }
    return records, {"layer2_baseline": baseline, "hashes": hashes}


def write_attempt_checksums(paths: list[Path]) -> None:
    payload = "".join(
        f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in paths
    )
    (ATTEMPT_DIR / "SHA256SUMS").write_text(payload, encoding="utf-8")


def execute_once() -> int:
    wrapper = load_verified_contract(require_unconsumed=True)
    ATTEMPT_DIR.mkdir(parents=True, exist_ok=False)
    run_log_path = ATTEMPT_DIR / "run.log"
    started = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "candidate_id": CANDIDATE_ID,
        "attempt_id": "attempt-0001",
        "started_at_utc": utc_now(),
        "contract_sha256": wrapper["contract_sha256"],
        "matrix_sha256": MATRIX_SHA256,
        "exact_command": wrapper["contract"]["execution"]["exact_command"],
        "authorization_consumed": True,
    }
    write_json(ATTEMPT_DIR / "execution_started.json", started)
    started_monotonic = time.monotonic()
    with run_log_path.open("w", encoding="utf-8") as run_log:
        try:
            log_message(run_log, f"official_attempt_started candidate={CANDIDATE_ID}")
            torch.set_num_threads(TORCH_THREADS)
            torch.set_num_interop_threads(1)
            source_contract = verify_source_contract()
            source_identity = verify_source_snapshot()
            versions = verify_versions()
            matrix = load_json(MATRIX_PATH)
            bound_prompts = prompt_binding(matrix)
            prompt_specs = {record["case_id"]: record for record in bound_prompts}
            tokenizer = AutoTokenizer.from_pretrained(
                SNAPSHOT,
                local_files_only=True,
                trust_remote_code=False,
            )
            inputs = {
                prompt_case.case_id: chat_input(tokenizer, prompt_case.prompt, DEFAULT_SYSTEM)
                for prompt_case in PROMPTS
            }

            log_message(run_log, "bf16_reference_recomputation_started prompts=9")
            reference_model = AutoModelForCausalLM.from_pretrained(
                SNAPSHOT,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            ).eval()
            references: dict[str, dict[str, Any]] = {}
            for prompt_case in PROMPTS:
                record = generate_reference(
                    reference_model,
                    tokenizer,
                    inputs[prompt_case.case_id],
                    prompt_case.expected,
                )
                expected = prompt_specs[prompt_case.case_id]
                require(
                    record["generated_token_ids"] == expected["bf16_generated_token_ids"],
                    f"BF16 token binding differs: {prompt_case.case_id}",
                )
                require(
                    record["response_gate"]["status"] == expected["bf16_response_gate_status"],
                    f"BF16 response gate differs: {prompt_case.case_id}",
                )
                references[prompt_case.case_id] = record
                log_message(
                    run_log,
                    f"bf16_prompt_complete case={prompt_case.case_id} "
                    f"tokens={len(record['generated_token_ids'])} gate={record['response_gate']['status']}",
                )
            del reference_model
            gc.collect()

            log_message(
                run_log,
                "candidate_model_construction_started dynamic_layer=2 dynamic_group_lanes=128 "
                "grouped_weight_tensor=model.layers.23.mlp.down_proj group_lanes=32",
            )
            scales = load_json(CALIBRATION_DIR / "derived_scales.json")
            candidate_model, runtime, arithmetic_self_test = reconstruct_candidate_model(scales)
            tensor_manifest, candidate_hashes = candidate_manifest(candidate_model)
            static_cost = candidate_cost()
            require(
                tensor_manifest[0]["weight_scale32_metadata_bytes"]
                == static_cost["candidate_weight_scale32_metadata_bytes"],
                "layer-23 candidate metadata byte count differs",
            )
            require(
                tensor_manifest[0]["packed_w4_bytes"]
                == static_cost["layer23_changed_scope_packed_w4_payload_bytes"],
                "layer-23 packed W4 byte count differs",
            )
            log_message(
                run_log,
                "candidate_model_construction_complete "
                f"layer23_packed_w4_sha256={candidate_hashes['hashes']['layer23_packed_w4_sha256']} "
                f"layer23_scale32_sha256={candidate_hashes['hashes']['layer23_weight_scale32_sha256']} "
                f"layer2_arithmetic_self_test={arithmetic_self_test['status']}",
            )

            prompt_results: list[dict[str, Any]] = []
            for prompt_case in PROMPTS:
                reference = references[prompt_case.case_id]
                candidate = generate_candidate(
                    candidate_model,
                    tokenizer,
                    inputs[prompt_case.case_id],
                    prompt_case.expected,
                    reference,
                    static_cost,
                )
                disagreements = sequence_disagreements(
                    reference["generated_token_ids"], candidate["generated_token_ids"]
                )
                gate_match = (
                    candidate["response_gate"]["status"] == reference["response_gate"]["status"]
                )
                public_reference = dict(reference)
                public_reference.pop("_logits")
                prompt_results.append(
                    {
                        "case_id": prompt_case.case_id,
                        "prompt_sha256": prompt_specs[prompt_case.case_id]["prompt_sha256"],
                        "expected_sha256": prompt_specs[prompt_case.case_id]["expected_sha256"],
                        "bf16": public_reference,
                        "candidate": candidate,
                        "token_disagreements": disagreements,
                        "response_gate_outcome_match": gate_match,
                    }
                )
                log_message(
                    run_log,
                    f"candidate_prompt_complete case={prompt_case.case_id} "
                    f"tokens={len(candidate['generated_token_ids'])} "
                    f"exact={candidate['exact_bf16_sequence_match']} "
                    f"gate={candidate['response_gate']['status']} gate_match={gate_match}",
                )

            aggregate = aggregate_prompts(prompt_results)
            disagreement_set = [
                {
                    "case_id": prompt["case_id"],
                    "token_disagreements": prompt["token_disagreements"],
                    "bf16_response_gate_status": prompt["bf16"]["response_gate"]["status"],
                    "candidate_response_gate_status": prompt["candidate"]["response_gate"]["status"],
                    "response_gate_outcome_match": prompt["response_gate_outcome_match"],
                }
                for prompt in prompt_results
                if prompt["token_disagreements"] or not prompt["response_gate_outcome_match"]
            ]
            eligible = (
                aggregate["exact_bf16_sequence_match_count"] == 9
                and aggregate["response_gate_outcome_match_count"] == 9
            )
            status = (
                "PASS_ELIGIBLE_FOR_FRESH_REVIEW"
                if eligible
                else "BLOCKED_EXACT_BF16_DISAGREEMENT"
            )
            runtime_summary = runtime.summary()
            require(
                runtime_summary["producer_count"] == runtime_summary["consumer_count"]
                and runtime_summary["pending_output"] is False,
                "dynamic layer-2/layer-3 recurrence did not retire exactly",
            )
            result = {
                "schema_version": 1,
                "classification": "qwen_instruct_stage1_layer2_dyns32_layer23_down_per32_single_candidate",
                "status": status,
                "mission_id": MISSION_ID,
                "candidate_id": CANDIDATE_ID,
                "attempt_id": "attempt-0001",
                "contract": {
                    "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(CONTRACT_PATH),
                    "contract_sha256": wrapper["contract_sha256"],
                },
                "matrix": file_record(MATRIX_PATH),
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "packages": {**versions, "numpy": importlib.metadata.version("numpy")},
                    "torch_num_threads": torch.get_num_threads(),
                    "torch_num_interop_threads": torch.get_num_interop_threads(),
                },
                "model": source_identity,
                "source_contract_status": source_contract["status"],
                "exact_numerical_delta": wrapper["contract"]["exact_numerical_delta"],
                "static_cost": static_cost,
                "layer23_tensor_manifest": tensor_manifest,
                "candidate_hashes": candidate_hashes,
                "layer2_arithmetic_self_test": arithmetic_self_test,
                "layer2_dynamic_runtime": runtime_summary,
                "prompts": prompt_results,
                "aggregate": aggregate,
                "eligibility": {
                    "rule": wrapper["contract"]["eligibility"]["rule"],
                    "eligible": eligible,
                    "eligible_candidate_id": CANDIDATE_ID if eligible else None,
                    "selected_policy_id": None,
                    "fresh_reviewer_required": True,
                },
                "disagreement_set": disagreement_set,
                "disposition": "READY_FOR_FRESH_REVIEW",
                "scope_guards": {
                    "candidate_attempt_count": 1,
                    "consumed_global_candidate_replayed": False,
                    "consumed_causal_candidate_replayed": False,
                    "alternate_policy_executed": False,
                    "ranking_or_selection_executed": False,
                    "gate_or_matrix_changed": False,
                    "other_activation_policy_changed": False,
                    "rtl_mutated": False,
                    "demo_retargeted": False,
                    "ppa_executed": False,
                    "u280_or_stage2_entered": False,
                    "network_access_performed": False,
                },
                "timing": {
                    "completed_at_utc": utc_now(),
                    "elapsed_seconds": time.monotonic() - started_monotonic,
                },
            }
            results_path = ATTEMPT_DIR / "results.json"
            write_json(results_path, result)
            log_message(
                run_log,
                f"official_attempt_complete status={status} "
                f"exact_sequences={aggregate['exact_bf16_sequence_match_count']}/9 "
                f"gate_matches={aggregate['response_gate_outcome_match_count']}/9",
            )
            reviewer_input = {
                "schema_version": 1,
                "mission_id": MISSION_ID,
                "candidate_id": CANDIDATE_ID,
                "status": "READY_FOR_FRESH_INDEPENDENT_REVIEW",
                "contract": file_record(CONTRACT_PATH),
                "results": file_record(results_path),
                "truthful_eligibility_decision": eligible,
                "review_requirements": [
                    "verify contract predates attempt-0001",
                    "verify no replay of all_transformer_per_32_v1 or causal_four_per_32_v1",
                    "verify exact numerical scope and unchanged-policy guards",
                    "recompute all nine complete-token and response-gate outcomes",
                    "accept only 9/9 exact plus 9/9 gate agreement",
                ],
                "independent_acceptance_required": True,
            }
            reviewer_path = ATTEMPT_DIR / "fresh_reviewer_input.json"
            write_json(reviewer_path, reviewer_input)
            write_attempt_checksums(
                [
                    CONTRACT_PATH,
                    ATTEMPT_DIR / "execution_started.json",
                    run_log_path,
                    results_path,
                    reviewer_path,
                ]
            )
            return 0 if eligible else 4
        except Exception as exc:
            failure = {
                "schema_version": 1,
                "classification": "EVALUATOR_EXECUTION_FAILURE",
                "status": "BLOCKED_EVALUATOR_EXECUTION_FAILURE",
                "mission_id": MISSION_ID,
                "candidate_id": CANDIDATE_ID,
                "attempt_id": "attempt-0001",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "numerical_correctness_conclusion": None,
                "selected_policy_id": None,
                "authorization_consumed": True,
                "failed_at_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started_monotonic,
            }
            failure_path = ATTEMPT_DIR / "failure.json"
            write_json(failure_path, failure)
            log_message(
                run_log,
                "official_attempt_failed classification=EVALUATOR_EXECUTION_FAILURE "
                f"error_type={type(exc).__name__} error={exc}",
            )
            reviewer_input = {
                "schema_version": 1,
                "mission_id": MISSION_ID,
                "candidate_id": CANDIDATE_ID,
                "status": "READY_FOR_FRESH_INDEPENDENT_REVIEW_OF_NO_EXECUTION",
                "contract": file_record(CONTRACT_PATH),
                "failure": file_record(failure_path),
                "truthful_eligibility_decision": None,
                "numerical_correctness_conclusion": None,
                "independent_acceptance_required": True,
            }
            reviewer_path = ATTEMPT_DIR / "fresh_reviewer_input.json"
            write_json(reviewer_path, reviewer_input)
            write_attempt_checksums(
                [
                    CONTRACT_PATH,
                    ATTEMPT_DIR / "execution_started.json",
                    run_log_path,
                    failure_path,
                    reviewer_path,
                ]
            )
            return 5


def verify_checksum_file() -> None:
    sums_path = ATTEMPT_DIR / "SHA256SUMS"
    require(sums_path.is_file(), "attempt checksum list is missing")
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        require(path.is_file(), f"checksummed artifact is missing: {relative}")
        require(sha256_file(path) == digest, f"checksummed artifact differs: {relative}")


def verify_result() -> dict[str, Any]:
    wrapper = load_verified_contract(require_unconsumed=False)
    attempts = sorted(OUTPUT_DIR.glob("attempt-*"))
    require([path.name for path in attempts] == ["attempt-0001"], "attempt set differs")
    verify_checksum_file()
    failure_path = ATTEMPT_DIR / "failure.json"
    if failure_path.is_file():
        failure = load_json(failure_path)
        require(
            failure.get("classification") == "EVALUATOR_EXECUTION_FAILURE"
            and failure.get("numerical_correctness_conclusion") is None,
            "evaluator failure classification differs",
        )
        return {
            "candidate_id": CANDIDATE_ID,
            "attempt_count": 1,
            "execution_status": "EVALUATOR_EXECUTION_FAILURE",
            "eligible": None,
            "numerical_correctness_conclusion": None,
            "failure_sha256": sha256_file(failure_path),
        }

    results_path = ATTEMPT_DIR / "results.json"
    reviewer_path = ATTEMPT_DIR / "fresh_reviewer_input.json"
    require(results_path.is_file() and reviewer_path.is_file(), "attempt result is incomplete")
    result = load_json(results_path)
    require(result.get("candidate_id") == CANDIDATE_ID, "result candidate differs")
    require(
        result.get("contract", {}).get("contract_sha256") == wrapper["contract_sha256"],
        "result contract binding differs",
    )
    require(result.get("matrix", {}).get("sha256") == MATRIX_SHA256, "result matrix differs")
    require(len(result.get("prompts", [])) == 9, "result prompt count differs")
    recomputed = aggregate_prompts(result["prompts"])
    require(recomputed == result.get("aggregate"), "result aggregate differs")
    eligible = (
        recomputed["exact_bf16_sequence_match_count"] == 9
        and recomputed["response_gate_outcome_match_count"] == 9
    )
    require(result.get("eligibility", {}).get("eligible") is eligible, "eligibility differs")
    require(result.get("eligibility", {}).get("selected_policy_id") is None, "policy was selected")
    runtime = result.get("layer2_dynamic_runtime", {})
    require(
        runtime.get("producer_count") == runtime.get("consumer_count")
        and runtime.get("pending_output") is False,
        "dynamic recurrence retirement differs",
    )
    require(
        result.get("layer2_arithmetic_self_test", {}).get("status") == "PASS",
        "layer-2 arithmetic self-test differs",
    )
    require(
        result.get("scope_guards", {}).get("candidate_attempt_count") == 1
        and result.get("scope_guards", {}).get("consumed_global_candidate_replayed") is False
        and result.get("scope_guards", {}).get("consumed_causal_candidate_replayed") is False,
        "attempt/replay guards differ",
    )
    reviewer = load_json(reviewer_path)
    require(
        reviewer.get("truthful_eligibility_decision") is eligible
        and reviewer.get("independent_acceptance_required") is True,
        "fresh reviewer input differs",
    )
    return {
        "candidate_id": CANDIDATE_ID,
        "attempt_count": 1,
        "execution_status": result["status"],
        "eligible": eligible,
        "exact_sequence_matches": recomputed["exact_bf16_sequence_match_count"],
        "gate_outcome_matches": recomputed["response_gate_outcome_match_count"],
        "disagreement_case_ids": [entry["case_id"] for entry in result["disagreement_set"]],
        "results_sha256": sha256_file(results_path),
        "fresh_reviewer_input_sha256": sha256_file(reviewer_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare-contract", "verify-contract", "execute", "verify-result"),
    )
    args = parser.parse_args()
    if args.command == "prepare-contract":
        wrapper = prepare_contract()
        print(
            "ACE2_STAGE1_LAYER2_DYNS32_LAYER23_DOWN_PER32_CONTRACT_PREPARED "
            f"candidate={CANDIDATE_ID} matrix={MATRIX_SHA256} "
            f"contract_sha256={wrapper['contract_sha256']}"
        )
        return 0
    if args.command == "verify-contract":
        wrapper = load_verified_contract(require_unconsumed=True)
        print(
            "ACE2_STAGE1_LAYER2_DYNS32_LAYER23_DOWN_PER32_PREFLIGHT_PASS "
            f"candidate={CANDIDATE_ID} attempts=0 contract_sha256={wrapper['contract_sha256']}"
        )
        return 0
    if args.command == "execute":
        return execute_once()
    summary = verify_result()
    print(
        "ACE2_STAGE1_LAYER2_DYNS32_LAYER23_DOWN_PER32_RESULT_VERIFIED "
        + json.dumps(summary, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
