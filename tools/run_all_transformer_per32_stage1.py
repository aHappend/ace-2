#!/usr/bin/env python3
"""Run the single predeclared all-transformer W4 Scale32 Stage-1 repair."""

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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import ace2_full_model_fixed_point as fixed_point
from ace2_quality_contracts import (
    SCALE32_ALL_ZERO_RECORD,
    ceil_scale32_from_float,
)
from discriminate_qwen_instruct_w4_weight_policies import (
    PROMPTS,
    chat_input,
)
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
from qwen_instruct_response_gate import evaluate as evaluate_response_gate
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM


CANDIDATE_ID = "all_transformer_per_32_v1"
MISSION_ID = "stage1-global-w4-per32-repair-v1"
MATRIX_SHA256 = "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
MATRIX_PATH = ROOT / "build/option-b-v2-weight-policy-20260806/full_sequence_discrimination.json"
OUTPUT_DIR = ROOT / "build/stage1-global-w4-per32-repair-v1"
CONTRACT_PATH = OUTPUT_DIR / "predeclared_contract.json"
ATTEMPT_DIR = OUTPUT_DIR / "attempt-0001"
GROUP_LANES = 32
MAX_NEW_TOKENS = 6
TORCH_THREADS = 16

PROJECTION_GEOMETRY = (
    ("self_attn.q_proj", 896, 896),
    ("self_attn.k_proj", 896, 128),
    ("self_attn.v_proj", 896, 128),
    ("self_attn.o_proj", 896, 896),
    ("mlp.gate_proj", 896, 4864),
    ("mlp.up_proj", 896, 4864),
    ("mlp.down_proj", 4864, 896),
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def expected_transformer_tensors() -> list[dict[str, Any]]:
    tensors: list[dict[str, Any]] = []
    for layer_index in range(24):
        for suffix, input_features, output_features in PROJECTION_GEOMETRY:
            group_count = input_features // GROUP_LANES
            tensors.append(
                {
                    "name": f"model.layers.{layer_index}.{suffix}.weight",
                    "module": f"model.layers.{layer_index}.{suffix}",
                    "shape": [output_features, input_features],
                    "input_group_lanes": GROUP_LANES,
                    "groups_per_output": group_count,
                    "scale32_record_count": output_features * group_count,
                }
            )
    require(len(tensors) == 168, "predeclared transformer tensor count differs")
    return tensors


def candidate_cost(tensors: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_scale_count = sum(entry["shape"][0] for entry in tensors)
    candidate_scale_count = sum(entry["scale32_record_count"] for entry in tensors)
    w4_values = sum(math.prod(entry["shape"]) for entry in tensors)
    return {
        "transformer_linear_tensor_count": len(tensors),
        "input_group_lanes": GROUP_LANES,
        "baseline_weight_scale_count": baseline_scale_count,
        "candidate_weight_scale32_record_count": candidate_scale_count,
        "additional_weight_scale32_record_count": candidate_scale_count
        - baseline_scale_count,
        "baseline_weight_scale_metadata_bytes": baseline_scale_count * 4,
        "candidate_weight_scale32_metadata_bytes": candidate_scale_count * 4,
        "additional_weight_scale32_metadata_bytes": (candidate_scale_count - baseline_scale_count)
        * 4,
        "group_partial_dot_products_per_model_token_position": candidate_scale_count,
        "group_combines_per_model_token_position": candidate_scale_count
        - baseline_scale_count,
        "w4_weight_values": w4_values,
        "packed_w4_payload_bytes": (w4_values + 1) // 2,
        "w4_payload_size_unchanged": True,
        "multiply_accumulate_count_ratio": 1.0,
    }


def prompt_binding(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    reference_prompts = matrix.get("bf16_reference", {}).get("prompts", [])
    require(len(reference_prompts) == len(PROMPTS) == 9, "frozen prompt count differs")
    by_case = {record["case_id"]: record for record in reference_prompts}
    require(len(by_case) == 9, "frozen prompt case ids are not unique")
    records: list[dict[str, Any]] = []
    for prompt_case in PROMPTS:
        record = by_case.get(prompt_case.case_id)
        require(record is not None, f"frozen matrix lacks prompt: {prompt_case.case_id}")
        prompt_sha = sha256_bytes(prompt_case.prompt.encode())
        expected_sha = sha256_bytes(prompt_case.expected.encode())
        require(record["prompt_sha256"] == prompt_sha, f"prompt hash differs: {prompt_case.case_id}")
        require(record["expected_sha256"] == expected_sha, f"response target hash differs: {prompt_case.case_id}")
        records.append(
            {
                "case_id": prompt_case.case_id,
                "prompt_utf8_bytes": len(prompt_case.prompt.encode()),
                "prompt_sha256": prompt_sha,
                "expected_utf8_bytes": len(prompt_case.expected.encode()),
                "expected_sha256": expected_sha,
                "bf16_generated_token_ids": record["generated_token_ids"],
                "bf16_response_gate_status": record["response_gate"]["status"],
            }
        )
    return records


def contract_artifacts() -> dict[str, Any]:
    paths = {
        "matrix": MATRIX_PATH,
        "source_contract": ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json",
        "derived_scales": CALIBRATION_DIR / "derived_scales.json",
        "full_model_image": IMAGE_DIR / "full_model_image.bin",
        "image_contract": IMAGE_DIR / "image_contract_v2.json",
        "image_manifest": IMAGE_DIR / "manifest.json",
        "option_b_identity_manifest": IMAGE_DIR / "option_b_identity_manifest.json",
        "oracle_contract": IMAGE_DIR / "oracle_contract.json",
        "response_gate_contract": IMAGE_DIR / "response_gate_contract.json",
        "runner_source": Path(__file__),
        "fixed_point_source": ROOT / "tools/ace2_full_model_fixed_point.py",
        "quality_contract_source": ROOT / "tools/ace2_quality_contracts.py",
        "prompt_and_matrix_evaluator_source": ROOT
        / "tools/discriminate_qwen_instruct_w4_weight_policies.py",
        "response_gate_source": ROOT / "tools/qwen_instruct_response_gate.py",
        "oracle_source": ROOT / "tools/qwen_instruct_w4a8_oracle.py",
    }
    return {name: file_record(path) for name, path in paths.items()}


def prepare_contract() -> dict[str, Any]:
    require_project_python()
    verify_source_contract()
    source_identity = verify_source_snapshot()
    versions = verify_versions()
    require(MATRIX_PATH.is_file(), "immutable discrimination matrix is missing")
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "immutable matrix hash differs")
    matrix = load_json(MATRIX_PATH)
    prompts = prompt_binding(matrix)
    tensors = expected_transformer_tensors()
    cost = candidate_cost(tensors)
    contract = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "hypothesis": "Uniform per-output-row, per-32-input-lane Scale32 W4 across all transformer projections reduces distributed/interacting weight error.",
        "matrix": {
            "path": MATRIX_PATH.relative_to(ROOT).as_posix(),
            "sha256": MATRIX_SHA256,
            "case_count": 9,
            "full_generated_sequences_bound": True,
        },
        "scope": {
            "exact_tensor_count": 168,
            "tensors": tensors,
            "all_and_only_transformer_linears": True,
            "lm_head_policy": "unchanged authenticated per-output-row W4A8 oracle policy",
            "embedding_policy": "unchanged authenticated oracle policy",
            "nonlinear_and_residual_policy": "unchanged authenticated oracle policy",
        },
        "weight_policy": {
            "weight_bits": 4,
            "signed_integer_range": [-8, 7],
            "group_axis": "input_feature",
            "input_group_lanes": GROUP_LANES,
            "scale_scope": "one normalized Scale32 record per output row and contiguous 32-input-lane group",
            "scale_selection": "ceil-encode max(abs(BF16 group))/7 as the smallest representable Scale32 value not below the ratio; all-zero groups use the canonical all-zero Scale32 record",
            "rounding": "torch.round round-to-nearest ties-to-even",
            "saturation": "clamp signed W4 to [-8,7]; final projection output clamp signed A8 to [-128,127]",
            "packing": "row-major output row then input lane; two's-complement signed-int4; even flat element in low nibble and odd flat element in high nibble",
            "group_accumulation": "signed A8xW4 dot per 32-lane group; decode each Scale32 product scale, align and sum all group contributions plus the unchanged source bias in a wide domain, divide by the unchanged oracle output scale, round ties-to-even once, then saturate",
            "reference_realization": "torch 2.11 float32 GEMM over exactly decoded Scale32 group products followed by one torch.round and saturation; deterministic thread count is frozen",
            "hardware_realizable": True,
        },
        "activation_policy": {
            "contract": fixed_point.ACTIVE_ROPE_MECHANISM,
            "dynamic_scale32_semantics": "unchanged",
            "input_A8_quantization": "unchanged authenticated oracle scales and rounding",
            "projection_output_A8_scales": "unchanged authenticated oracle scales",
            "rope_and_attention_scale32": "unchanged authenticated oracle behavior",
            "activation_recalibration_permitted": False,
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
            "aligned_logit_error": "full-vocabulary candidate versus recomputed BF16 logits at every step whose incoming generated prefix exactly matches BF16; include the first mismatching token step",
            "prompts": prompts,
        },
        "eligibility": {
            "rule": "all nine complete generated token sequences exactly equal BF16 and all nine response-gate outcomes equal BF16",
            "aggregate_improvement_is_insufficient": True,
            "local_arithmetic_agreement_is_insufficient": True,
            "selected_policy_id_before_execution": None,
            "fresh_reviewer_required": True,
        },
        "execution": {
            "exact_authorized_candidate_attempts": 1,
            "attempt_id": "attempt-0001",
            "torch_num_threads": TORCH_THREADS,
            "network_access_permitted": False,
            "alternate_policy_permitted": False,
            "tuning_ranking_bisection_sweep_permitted": False,
            "rtl_demo_ppa_stage2_mutation_permitted": False,
        },
        "model": source_identity,
        "image_binding": {
            "image_id": "build-qwen2.5-0.5b-instruct-w4a8-image-v1",
            "dynamic_activation_semantics_source": "authenticated Option-B W4A8 oracle and derived scale table",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {**versions, "numpy": importlib.metadata.version("numpy")},
        },
        "static_cost": cost,
        "artifacts": contract_artifacts(),
        "no_outcome_assumption": "No token, gate, or eligibility outcome is assumed before the single execution.",
    }
    wrapper = {"contract": contract, "contract_sha256": sha256_bytes(canonical_bytes(contract))}
    write_json(CONTRACT_PATH, wrapper)
    return wrapper


def require_project_python() -> None:
    expected = ROOT / ".venv/bin/python"
    require(expected.is_file(), f"project Python is missing: {expected}")
    require(os.path.samefile(Path(sys.executable), expected), "wrong Python executable")


def load_verified_contract(*, require_unconsumed: bool) -> dict[str, Any]:
    require_project_python()
    require(CONTRACT_PATH.is_file(), "predeclared contract is missing")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "contract wrapper differs")
    contract = wrapper["contract"]
    require(sha256_bytes(canonical_bytes(contract)) == wrapper["contract_sha256"], "contract digest differs")
    require(contract.get("candidate_id") == CANDIDATE_ID, "candidate id differs")
    require(contract.get("matrix", {}).get("sha256") == MATRIX_SHA256, "matrix binding differs")
    require(contract.get("scope", {}).get("tensors") == expected_transformer_tensors(), "168-tensor scope differs")
    require(contract.get("weight_policy", {}).get("input_group_lanes") == GROUP_LANES, "group size differs")
    require(contract.get("generation_and_evaluator", {}).get("max_new_tokens") == MAX_NEW_TOKENS, "generation length differs")
    for record in contract.get("artifacts", {}).values():
        path = ROOT / record["path"]
        require(path.is_file(), f"contract artifact is missing: {record['path']}")
        require(path.stat().st_size == record["bytes"], f"contract artifact size differs: {record['path']}")
        require(sha256_file(path) == record["sha256"], f"contract artifact hash differs: {record['path']}")
    verify_source_contract()
    verify_source_snapshot()
    verify_versions()
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "immutable matrix hash differs")
    matrix = load_json(MATRIX_PATH)
    require(prompt_binding(matrix) == contract["generation_and_evaluator"]["prompts"], "prompt binding differs")
    attempts = sorted(OUTPUT_DIR.glob("attempt-*"))
    if require_unconsumed:
        require(not attempts, f"exactly-once authorization is already consumed: {[path.name for path in attempts]}")
    output_probe = OUTPUT_DIR / ".preflight-write-probe"
    output_probe.write_bytes(b"preflight\n")
    require(output_probe.read_bytes() == b"preflight\n", "output path readback differs")
    output_probe.unlink()
    scale32_self_test()
    return wrapper


def encode_scale32(values: Tensor, zero_mask: Tensor | None = None) -> Tensor:
    value = values.detach().to(torch.float64)
    if zero_mask is None:
        zero_mask = torch.zeros_like(value, dtype=torch.bool)
    safe = torch.where(zero_mask, torch.ones_like(value), value)
    require(bool(torch.all(torch.isfinite(safe))), "Scale32 source contains non-finite values")
    require(bool(torch.all(safe > 0)), "Scale32 source must be positive")
    mantissa, exponent = torch.frexp(safe)
    normalized_exponent = exponent.to(torch.int64) - 1
    significand = torch.ceil(mantissa * float(1 << 16)).to(torch.int64)
    carry = significand == (1 << 16)
    significand = torch.where(carry, torch.full_like(significand, 1 << 15), significand)
    normalized_exponent = normalized_exponent + carry.to(torch.int64)
    require(bool(torch.all((significand >= 0x8000) & (significand <= 0xFFFF))), "Scale32 significand escaped normalized range")
    require(bool(torch.all((normalized_exponent >= -24) & (normalized_exponent <= 4))), "Scale32 exponent escaped frozen range")
    record = significand | ((normalized_exponent & 0xFF) << 16)
    return torch.where(
        zero_mask,
        torch.full_like(record, SCALE32_ALL_ZERO_RECORD),
        record,
    )


def decode_scale32(records: Tensor) -> Tensor:
    record = records.detach().to(torch.int64)
    require(bool(torch.all((record >= 0) & (record <= 0xFFFFFFFF))), "Scale32 record escaped u32")
    require(bool(torch.all((record >> 24) == 0)), "Scale32 reserved byte is nonzero")
    significand = record & 0xFFFF
    exponent_u8 = (record >> 16) & 0xFF
    exponent = torch.where(exponent_u8 >= 128, exponent_u8 - 256, exponent_u8)
    require(bool(torch.all((significand >= 0x8000) & (significand <= 0xFFFF))), "Scale32 significand is malformed")
    require(bool(torch.all((exponent >= -24) & (exponent <= 4))), "Scale32 exponent is malformed")
    return torch.ldexp(significand.to(torch.float64), exponent.to(torch.int32) - 15)


def scale32_self_test() -> None:
    values = torch.tensor(
        [2.0**-24, 2.0**-20, 0.000123456789, 0.03125, 0.2, 1.0, 15.999],
        dtype=torch.float64,
    )
    records = encode_scale32(values)
    expected = torch.tensor(
        [ceil_scale32_from_float(float(value)) for value in values], dtype=torch.int64
    )
    require(torch.equal(records, expected), "vector Scale32 encoder differs from scalar contract")
    decoded = decode_scale32(records)
    require(bool(torch.all(decoded >= values)), "Scale32 ceil encoding undershot its source")
    zero_record = encode_scale32(torch.ones(1, dtype=torch.float64), torch.ones(1, dtype=torch.bool))
    require(int(zero_record.item()) == SCALE32_ALL_ZERO_RECORD, "all-zero Scale32 record differs")


@dataclass
class GroupedRecord:
    module_name: str
    weight_error_relative_l2: float
    weight_error_maximum_absolute: float


class GroupedScale32W4A8Linear(fixed_point.W4A8Linear):
    """W4A8 linear with one explicit Scale32 W4 scale per row and 32 lanes."""

    def __init__(
        self,
        source: nn.Linear,
        calibration: fixed_point.CalibrationRange,
        *,
        module_name: str,
        input_is_quantized: bool = False,
        output_head_absmax: list[float] | None = None,
        output_head_size: int | None = None,
        scale_cap: float | None = None,
        use_percentile_scale: bool = False,
        scale32_output: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        if calibration.input_absmax <= 0 or calibration.output_absmax <= 0:
            raise ValueError("linear calibration ranges must be positive")
        require(source.in_features % GROUP_LANES == 0, f"group size does not divide {module_name}")
        self.module_name = module_name
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.groups_per_output = self.in_features // GROUP_LANES
        input_absmax = (
            calibration.input_percentile_absmax
            if use_percentile_scale
            else calibration.input_absmax
        )
        output_absmax = (
            calibration.output_percentile_absmax
            if use_percentile_scale
            else calibration.output_absmax
        )
        if input_absmax is None or output_absmax is None:
            raise ValueError("requested percentile calibration ranges are missing")
        self.input_scale = fixed_point.positive_scale(input_absmax, cap=scale_cap)
        self.hardware_input_scale = self.input_scale
        if output_head_absmax is not None:
            if output_head_size is None or output_head_size <= 0:
                raise ValueError("per-head projection scales require a positive head size")
            if len(output_head_absmax) * output_head_size != self.out_features:
                raise ValueError("per-head projection scales do not cover all outputs")
            head_scales = torch.tensor(
                [fixed_point.positive_scale(value, cap=scale_cap) for value in output_head_absmax],
                dtype=torch.float64,
            )
            output_scale_per_channel = head_scales.repeat_interleave(output_head_size)
            self.output_scale = None
            self.output_scale32_record = None
        else:
            head_scales = torch.empty(0, dtype=torch.float64)
            self.output_scale = fixed_point.positive_scale(output_absmax, cap=scale_cap)
            self.output_scale32_record = (
                fixed_point.ceil_scale32_from_float(self.output_scale)
                if scale32_output
                else None
            )
            if self.output_scale32_record is not None:
                numerator, denominator = fixed_point.scale32_ratio(self.output_scale32_record)
                self.output_scale = numerator / denominator
            output_scale_per_channel = torch.full(
                (self.out_features,), self.output_scale, dtype=torch.float64
            )
        self.output_head_size = output_head_size
        self.register_buffer("output_head_scales", head_scales, persistent=True)
        self.register_buffer("output_scale_per_channel", output_scale_per_channel, persistent=True)
        self.input_is_quantized = input_is_quantized

        source_weight = source.weight.detach().to(torch.float64)
        grouped = source_weight.reshape(self.out_features, self.groups_per_output, GROUP_LANES)
        absmax = grouped.abs().amax(dim=2)
        zero_group = absmax == 0
        ideal_scale = torch.where(zero_group, torch.ones_like(absmax), absmax / 7.0)
        weight_scale32_records = encode_scale32(ideal_scale, zero_group)
        weight_scale = decode_scale32(weight_scale32_records)
        qgrouped = torch.round(grouped / weight_scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
        qweight = qgrouped.reshape_as(source.weight)
        reconstructed = (qgrouped.to(torch.float64) * weight_scale.unsqueeze(-1)).reshape_as(source_weight)
        difference = reconstructed - source_weight
        denominator = torch.linalg.vector_norm(source_weight)
        self.grouped_record = GroupedRecord(
            module_name=module_name,
            weight_error_relative_l2=float(torch.linalg.vector_norm(difference) / denominator),
            weight_error_maximum_absolute=float(difference.abs().amax()),
        )
        self.register_buffer("qweight", qweight.contiguous(), persistent=True)
        self.register_buffer("weight_scale32_records", weight_scale32_records, persistent=True)
        self.register_buffer("weight_scale", weight_scale, persistent=True)
        self.register_buffer("zero_group_mask", zero_group, persistent=False)
        self.register_buffer(
            "_source_bias",
            source.bias.detach().to(torch.float32) if source.bias is not None else None,
            persistent=False,
        )
        self.register_buffer(
            "native_scale32_records",
            self._native_scale32_for_input_scale(self.input_scale),
            persistent=True,
        )
        multiplier, right_shift = self._metadata_for_input_scale(self.input_scale)
        self.register_buffer("multiplier", multiplier, persistent=True)
        self.register_buffer("right_shift", right_shift, persistent=True)
        self.bias_accumulator = None
        self.register_buffer(
            "_dequantized_product_weight",
            self._product_weight(self.native_scale32_records),
            persistent=False,
        )
        self._pending_requantized: Tensor | None = None

    def _native_scale32_for_input_scale(self, input_scale: float) -> Tensor:
        require(math.isfinite(input_scale) and input_scale > 0, "projection input scale is invalid")
        return encode_scale32(self.weight_scale * input_scale, self.zero_group_mask)

    def _metadata_for_input_scale(self, input_scale: float) -> tuple[Tensor, Tensor]:
        native = self._native_scale32_for_input_scale(input_scale)
        real_multiplier = decode_scale32(native) / self.output_scale_per_channel[:, None]
        return fixed_point.derive_multiplier(real_multiplier)

    def _product_weight(self, native_records: Tensor) -> Tensor:
        product_scale = decode_scale32(native_records).to(torch.float32)
        grouped = self.qweight.reshape(
            self.out_features, self.groups_per_output, GROUP_LANES
        ).to(torch.float32)
        return (grouped * product_scale.unsqueeze(-1)).reshape(
            self.out_features, self.in_features
        ).contiguous()

    def bind_hardware_input_scale(self, input_scale: float) -> None:
        native = self._native_scale32_for_input_scale(input_scale)
        multiplier, right_shift = self._metadata_for_input_scale(input_scale)
        self.hardware_input_scale = input_scale
        self.native_scale32_records.copy_(native)
        self.multiplier.copy_(multiplier)
        self.right_shift.copy_(right_shift)
        self._dequantized_product_weight = self._product_weight(native)

    def accumulator_quantized(self, qinput: Tensor) -> Tensor:
        if qinput.dtype != torch.int8:
            raise TypeError("projection input must be signed int8")
        if qinput.shape[-1] != self.in_features:
            raise ValueError("projection input geometry differs")
        if self._pending_requantized is not None:
            raise RuntimeError("grouped projection has an unconsumed requantization")
        original_shape = qinput.shape[:-1]
        flat = qinput.reshape(-1, self.in_features).to(torch.float32)
        real = F.linear(flat, self._dequantized_product_weight, self._source_bias)
        scaled = real / self.output_scale_per_channel.to(torch.float32)
        proxy = torch.round(scaled).to(torch.int64)
        self._pending_requantized = proxy.clamp(-128, 127).to(torch.int8)
        return proxy.reshape(math.prod(original_shape), self.out_features)

    def requantize_accumulator(self, accumulator: Tensor, original_shape: tuple[int, ...]) -> Tensor:
        if accumulator.shape != (math.prod(original_shape), self.out_features):
            raise ValueError("grouped projection accumulator shape differs")
        if self._pending_requantized is None:
            raise RuntimeError("grouped projection lacks pending requantization")
        output = self._pending_requantized
        self._pending_requantized = None
        return output.reshape(*original_shape, self.out_features)

    def forward_native_hardware_input(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        raise RuntimeError("grouped Scale32 candidate does not retarget native-accumulator mode")

    def forward_shadow_hardware_input(self, inputs: Tensor) -> Tensor:
        raise RuntimeError("grouped Scale32 candidate does not retarget projection-shadow mode")

    def forward_qk_residual_hardware_input(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("grouped Scale32 candidate does not retarget Q/K residual mode")

    def forward_v_residual_hardware_input(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("grouped Scale32 candidate does not retarget V residual mode")


def replace_candidate_linears(
    model: nn.Module,
    ranges: dict[str, fixed_point.CalibrationRange],
) -> list[str]:
    replacements = [
        (name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)
    ]
    require(len(replacements) == 169, f"expected 169 linears, found {len(replacements)}")
    expected_modules = [entry["module"] for entry in expected_transformer_tensors()]
    observed_transformer = [name for name, _ in replacements if name != "lm_head"]
    require(observed_transformer == expected_modules, "live transformer linear order/scope differs")
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
        replacement: nn.Module
        if name == "lm_head":
            replacement = fixed_point.W4A8Linear(module, ranges[name], **common)
        else:
            replacement = GroupedScale32W4A8Linear(
                module, ranges[name], module_name=name, **common
            )
        setattr(parent, child_name, replacement)
    return observed_transformer


def reconstruct_candidate_model(scales: dict[str, Any]) -> tuple[nn.Module, list[str]]:
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
    scope = replace_candidate_linears(model, ranges)
    fixed_point.replace_fixed_operators(
        model,
        operators,
        rope_diagnostic_mechanism=fixed_point.ACTIVE_ROPE_MECHANISM,
    )
    grouped_modules = [
        name for name, module in model.named_modules() if isinstance(module, GroupedScale32W4A8Linear)
    ]
    require(grouped_modules == scope, "post-replacement grouped module scope differs")
    return model, scope


def hash_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def packed_w4(module: GroupedScale32W4A8Linear) -> bytes:
    flat = module.qweight.detach().cpu().contiguous().numpy().astype(np.int8, copy=False).reshape(-1)
    require(flat.size % 2 == 0, f"odd W4 payload geometry: {module.module_name}")
    nibble = flat.astype(np.uint8, copy=False) & np.uint8(0x0F)
    packed = nibble[0::2] | (nibble[1::2] << np.uint8(4))
    return packed.tobytes(order="C")


def scale32_bytes(module: GroupedScale32W4A8Linear) -> bytes:
    records = module.weight_scale32_records.detach().cpu().contiguous().numpy().astype("<u4", copy=False)
    return records.tobytes(order="C")


def candidate_tensor_manifest(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    modules = dict(model.named_modules())
    records: list[dict[str, Any]] = []
    aggregate_packed = hashlib.sha256()
    aggregate_scale32 = hashlib.sha256()
    for spec in expected_transformer_tensors():
        module = modules.get(spec["module"])
        require(isinstance(module, GroupedScale32W4A8Linear), f"grouped module missing: {spec['module']}")
        qweight_raw = module.qweight.detach().cpu().contiguous().numpy().astype(np.int8, copy=False).tobytes(order="C")
        packed_raw = packed_w4(module)
        metadata_raw = scale32_bytes(module)
        aggregate_packed.update(spec["name"].encode() + b"\0" + packed_raw)
        aggregate_scale32.update(spec["name"].encode() + b"\0" + metadata_raw)
        records.append(
            {
                **spec,
                "qweight_s8_container_sha256": hash_bytes(qweight_raw),
                "packed_w4_bytes": len(packed_raw),
                "packed_w4_sha256": hash_bytes(packed_raw),
                "weight_scale32_metadata_bytes": len(metadata_raw),
                "weight_scale32_sha256": hash_bytes(metadata_raw),
                "weight_error": {
                    "relative_l2_error": module.grouped_record.weight_error_relative_l2,
                    "maximum_absolute_error": module.grouped_record.weight_error_maximum_absolute,
                },
            }
        )
    return records, {
        "packed_w4_scope_sha256": aggregate_packed.hexdigest(),
        "weight_scale32_scope_sha256": aggregate_scale32.hexdigest(),
        "tensor_manifest_sha256": sha256_bytes(canonical_bytes(records)),
    }


def logit_comparison(reference: Tensor, candidate: Tensor) -> dict[str, Any]:
    left = reference.to(torch.float64)
    right = candidate.to(torch.float64)
    difference = right - left
    denominator = torch.linalg.vector_norm(left)
    top = torch.topk(right, k=2)
    return {
        "candidate_token_id": int(right.argmax()),
        "reference_token_id": int(left.argmax()),
        "candidate_top_one_margin": float(top.values[0] - top.values[1]),
        "candidate_top_two_token_ids": [int(value) for value in top.indices],
        "relative_l2_error": (
            None
            if float(denominator) == 0.0
            else float(torch.linalg.vector_norm(difference) / denominator)
        ),
        "maximum_absolute_error": float(difference.abs().amax()),
    }


def generate_reference(
    model: nn.Module,
    tokenizer: Any,
    input_ids: Tensor,
    expected: str,
) -> dict[str, Any]:
    prefix = input_ids
    generated: list[int] = []
    logits_by_step: list[Tensor] = []
    prefix_lengths: list[int] = []
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS):
            prefix_lengths.append(int(prefix.shape[1]))
            logits = model(input_ids=prefix, use_cache=False).logits[0, -1].detach().cpu()
            token = int(logits.argmax())
            generated.append(token)
            logits_by_step.append(logits)
            if token in TERMINATION_TOKEN_IDS:
                break
            prefix = torch.cat([prefix, torch.tensor([[token]], dtype=prefix.dtype)], dim=1)
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    gate = evaluate_response_gate(
        {"decoded_text": decoded, "generated_token_ids": generated}, expected
    )
    return {
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "response_gate": gate,
        "prefix_lengths": prefix_lengths,
        "_logits": logits_by_step,
    }


def generate_candidate(
    model: nn.Module,
    tokenizer: Any,
    input_ids: Tensor,
    expected: str,
    reference: dict[str, Any],
    static_cost: dict[str, Any],
) -> dict[str, Any]:
    prefix = input_ids
    generated: list[int] = []
    prefix_lengths: list[int] = []
    steps: list[dict[str, Any]] = []
    aligned = True
    with torch.inference_mode():
        for index in range(MAX_NEW_TOKENS):
            prefix_lengths.append(int(prefix.shape[1]))
            logits = model(input_ids=prefix, use_cache=False).logits[0, -1].detach().cpu()
            token = int(logits.argmax())
            comparison = None
            if aligned and index < len(reference["_logits"]):
                comparison = logit_comparison(reference["_logits"][index], logits)
            steps.append(
                {
                    "index": index,
                    "token_id": token,
                    "incoming_prefix_aligned_with_bf16": aligned,
                    "aligned_logit_comparison": comparison,
                }
            )
            generated.append(token)
            reference_ids = reference["generated_token_ids"]
            aligned = aligned and index < len(reference_ids) and token == reference_ids[index]
            if token in TERMINATION_TOKEN_IDS:
                break
            prefix = torch.cat([prefix, torch.tensor([[token]], dtype=prefix.dtype)], dim=1)
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    gate = evaluate_response_gate(
        {"decoded_text": decoded, "generated_token_ids": generated}, expected
    )
    reference_ids = reference["generated_token_ids"]
    common_prefix = 0
    for left, right in zip(generated, reference_ids):
        if left != right:
            break
        common_prefix += 1
    model_token_positions = sum(prefix_lengths)
    return {
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "response_gate": gate,
        "exact_bf16_sequence_match": generated == reference_ids,
        "common_bf16_prefix_tokens": common_prefix,
        "prefix_lengths": prefix_lengths,
        "model_token_positions_evaluated": model_token_positions,
        "candidate_weight_scale32_metadata_bytes": static_cost[
            "candidate_weight_scale32_metadata_bytes"
        ],
        "group_combines_per_model_token_position": static_cost[
            "group_combines_per_model_token_position"
        ],
        "group_combines_over_executed_prefixes": static_cost[
            "group_combines_per_model_token_position"
        ]
        * model_token_positions,
        "steps": steps,
    }


def sequence_disagreements(reference_ids: list[int], candidate_ids: list[int]) -> list[dict[str, Any]]:
    disagreements: list[dict[str, Any]] = []
    for index in range(max(len(reference_ids), len(candidate_ids))):
        reference_token = reference_ids[index] if index < len(reference_ids) else None
        candidate_token = candidate_ids[index] if index < len(candidate_ids) else None
        if reference_token != candidate_token:
            disagreements.append(
                {
                    "token_position": index,
                    "bf16_token_id": reference_token,
                    "candidate_token_id": candidate_token,
                }
            )
    return disagreements


def aggregate_prompts(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [
        step["aligned_logit_comparison"]["relative_l2_error"]
        for prompt in prompts
        for step in prompt["candidate"]["steps"]
        if step["aligned_logit_comparison"] is not None
        and step["aligned_logit_comparison"]["relative_l2_error"] is not None
    ]
    return {
        "prompt_count": len(prompts),
        "exact_bf16_sequence_match_count": sum(
            prompt["candidate"]["exact_bf16_sequence_match"] for prompt in prompts
        ),
        "response_gate_outcome_match_count": sum(
            prompt["candidate"]["response_gate"]["status"]
            == prompt["bf16"]["response_gate"]["status"]
            for prompt in prompts
        ),
        "aligned_logit_step_count": len(errors),
        "worst_aligned_relative_l2_error": max(errors) if errors else None,
        "mean_aligned_relative_l2_error": sum(errors) / len(errors) if errors else None,
        "total_common_bf16_prefix_tokens": sum(
            prompt["candidate"]["common_bf16_prefix_tokens"] for prompt in prompts
        ),
    }


def log_message(handle: Any, message: str) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


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
        "authorization_consumed": True,
    }
    write_json(ATTEMPT_DIR / "execution_started.json", started)
    started_monotonic = time.monotonic()
    with run_log_path.open("w", encoding="utf-8") as run_log:
        try:
            log_message(run_log, "official_attempt_started candidate=all_transformer_per_32_v1")
            torch.set_num_threads(TORCH_THREADS)
            torch.set_num_interop_threads(1)
            source_contract = verify_source_contract()
            source_identity = verify_source_snapshot()
            versions = verify_versions()
            matrix = load_json(MATRIX_PATH)
            bound_prompts = prompt_binding(matrix)
            prompt_specs = {record["case_id"]: record for record in bound_prompts}
            tokenizer = AutoTokenizer.from_pretrained(
                SNAPSHOT, local_files_only=True, trust_remote_code=False
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
            reference_records: dict[str, dict[str, Any]] = {}
            for prompt_case in PROMPTS:
                record = generate_reference(
                    reference_model,
                    tokenizer,
                    inputs[prompt_case.case_id],
                    prompt_case.expected,
                )
                expected_record = prompt_specs[prompt_case.case_id]
                require(
                    record["generated_token_ids"]
                    == expected_record["bf16_generated_token_ids"],
                    f"BF16 token binding differs: {prompt_case.case_id}",
                )
                require(
                    record["response_gate"]["status"]
                    == expected_record["bf16_response_gate_status"],
                    f"BF16 response gate differs: {prompt_case.case_id}",
                )
                reference_records[prompt_case.case_id] = record
                log_message(
                    run_log,
                    f"bf16_prompt_complete case={prompt_case.case_id} tokens={len(record['generated_token_ids'])} gate={record['response_gate']['status']}",
                )
            del reference_model
            gc.collect()

            log_message(run_log, "candidate_model_construction_started tensors=168 group_lanes=32")
            scales = load_json(CALIBRATION_DIR / "derived_scales.json")
            candidate_model, scope = reconstruct_candidate_model(scales)
            require(len(scope) == 168, "candidate scope count differs after construction")
            tensor_manifest, candidate_hashes = candidate_tensor_manifest(candidate_model)
            static_cost = candidate_cost(expected_transformer_tensors())
            require(
                sum(record["weight_scale32_metadata_bytes"] for record in tensor_manifest)
                == static_cost["candidate_weight_scale32_metadata_bytes"],
                "candidate metadata byte count differs",
            )
            require(
                sum(record["packed_w4_bytes"] for record in tensor_manifest)
                == static_cost["packed_w4_payload_bytes"],
                "candidate packed W4 byte count differs",
            )
            log_message(
                run_log,
                "candidate_model_construction_complete "
                f"packed_w4_sha256={candidate_hashes['packed_w4_scope_sha256']} "
                f"scale32_sha256={candidate_hashes['weight_scale32_scope_sha256']}",
            )

            prompt_results: list[dict[str, Any]] = []
            for prompt_case in PROMPTS:
                reference = reference_records[prompt_case.case_id]
                candidate = generate_candidate(
                    candidate_model,
                    tokenizer,
                    inputs[prompt_case.case_id],
                    prompt_case.expected,
                    reference,
                    static_cost,
                )
                token_disagreements = sequence_disagreements(
                    reference["generated_token_ids"], candidate["generated_token_ids"]
                )
                gate_match = (
                    candidate["response_gate"]["status"]
                    == reference["response_gate"]["status"]
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
                        "token_disagreements": token_disagreements,
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
            result = {
                "schema_version": 1,
                "classification": "qwen_instruct_stage1_all_transformer_per32_single_candidate",
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
                "policy": wrapper["contract"]["weight_policy"],
                "activation_policy": wrapper["contract"]["activation_policy"],
                "static_cost": static_cost,
                "tensor_manifest": tensor_manifest,
                "candidate_hashes": candidate_hashes,
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
                "disposition": "READY_FOR_FRESH_REVIEW" if eligible else "BLOCKED",
                "scope_guards": {
                    "candidate_attempt_count": 1,
                    "alternate_policy_executed": False,
                    "ranking_or_selection_executed": False,
                    "activation_policy_changed": False,
                    "rtl_mutated": False,
                    "demo_retargeted": False,
                    "ppa_executed": False,
                    "stage2_entered": False,
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
                f"official_attempt_complete status={status} exact_sequences={aggregate['exact_bf16_sequence_match_count']}/9 gate_matches={aggregate['response_gate_outcome_match_count']}/9",
            )
            sums_paths = [
                CONTRACT_PATH,
                ATTEMPT_DIR / "execution_started.json",
                run_log_path,
                results_path,
            ]
            sums = "".join(
                f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n"
                for path in sums_paths
            )
            (ATTEMPT_DIR / "SHA256SUMS").write_text(sums, encoding="utf-8")
            if eligible:
                reviewer_input = {
                    "schema_version": 1,
                    "mission_id": MISSION_ID,
                    "candidate_id": CANDIDATE_ID,
                    "status": "READY_FOR_FRESH_REVIEW",
                    "contract": file_record(CONTRACT_PATH),
                    "results": file_record(results_path),
                    "checksums": file_record(ATTEMPT_DIR / "SHA256SUMS"),
                    "independent_acceptance_required": True,
                }
                write_json(ATTEMPT_DIR / "fresh_reviewer_input.json", reviewer_input)
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
            write_json(ATTEMPT_DIR / "failure.json", failure)
            log_message(
                run_log,
                f"official_attempt_failed classification=EVALUATOR_EXECUTION_FAILURE error_type={type(exc).__name__} error={exc}",
            )
            return 5


def parse_sums(path: Path) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        records.append((digest, ROOT / relative))
    return records


def verify_result() -> dict[str, Any]:
    wrapper = load_verified_contract(require_unconsumed=False)
    require(ATTEMPT_DIR.is_dir(), "official attempt directory is missing")
    failure_path = ATTEMPT_DIR / "failure.json"
    if failure_path.is_file():
        failure = load_json(failure_path)
        require(failure.get("classification") == "EVALUATOR_EXECUTION_FAILURE", "failure classification differs")
        require(failure.get("numerical_correctness_conclusion") is None, "evaluator failure drew a numerical conclusion")
        return {
            "status": failure["status"],
            "candidate_id": CANDIDATE_ID,
            "attempt_count": 1,
            "numerical_correctness_conclusion": None,
        }
    results_path = ATTEMPT_DIR / "results.json"
    sums_path = ATTEMPT_DIR / "SHA256SUMS"
    require(results_path.is_file() and sums_path.is_file(), "official result artifacts are incomplete")
    for digest, path in parse_sums(sums_path):
        require(path.is_file(), f"checksummed artifact is missing: {path.relative_to(ROOT)}")
        require(sha256_file(path) == digest, f"checksummed artifact differs: {path.relative_to(ROOT)}")
    result = load_json(results_path)
    require(result.get("candidate_id") == CANDIDATE_ID, "result candidate id differs")
    require(result.get("matrix", {}).get("sha256") == MATRIX_SHA256, "result matrix hash differs")
    require(result.get("contract", {}).get("contract_sha256") == wrapper["contract_sha256"], "result contract binding differs")
    require(len(result.get("tensor_manifest", [])) == 168, "result tensor manifest count differs")
    require(len(result.get("prompts", [])) == 9, "result prompt count differs")
    recomputed_disagreements = []
    for prompt in result["prompts"]:
        token_disagreements = sequence_disagreements(
            prompt["bf16"]["generated_token_ids"],
            prompt["candidate"]["generated_token_ids"],
        )
        require(token_disagreements == prompt["token_disagreements"], f"token disagreement set differs: {prompt['case_id']}")
        gate_match = (
            prompt["bf16"]["response_gate"]["status"]
            == prompt["candidate"]["response_gate"]["status"]
        )
        require(gate_match == prompt["response_gate_outcome_match"], f"gate outcome match differs: {prompt['case_id']}")
        if token_disagreements or not gate_match:
            recomputed_disagreements.append(
                {
                    "case_id": prompt["case_id"],
                    "token_disagreements": token_disagreements,
                    "bf16_response_gate_status": prompt["bf16"]["response_gate"]["status"],
                    "candidate_response_gate_status": prompt["candidate"]["response_gate"]["status"],
                    "response_gate_outcome_match": gate_match,
                }
            )
    require(recomputed_disagreements == result["disagreement_set"], "aggregate disagreement set differs")
    eligible = not recomputed_disagreements
    require(result["eligibility"]["eligible"] == eligible, "eligibility disposition differs")
    require(result["eligibility"]["selected_policy_id"] is None, "result selected a policy without review")
    require(result["scope_guards"]["candidate_attempt_count"] == 1, "attempt count differs")
    return {
        "status": result["status"],
        "candidate_id": CANDIDATE_ID,
        "attempt_count": 1,
        "eligible": eligible,
        "exact_sequence_matches": result["aggregate"]["exact_bf16_sequence_match_count"],
        "gate_outcome_matches": result["aggregate"]["response_gate_outcome_match_count"],
        "disagreement_case_ids": [entry["case_id"] for entry in recomputed_disagreements],
        "results_sha256": sha256_file(results_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare-contract", "verify-contract", "execute", "verify-result"),
    )
    args = parser.parse_args()
    if args.command == "prepare-contract":
        require(not list(OUTPUT_DIR.glob("attempt-*")), "cannot rewrite contract after an attempt exists")
        wrapper = prepare_contract()
        print(
            "ACE2_STAGE1_PER32_CONTRACT_PREPARED "
            f"candidate={CANDIDATE_ID} tensors=168 matrix={MATRIX_SHA256} "
            f"contract_sha256={wrapper['contract_sha256']}"
        )
        return 0
    if args.command == "verify-contract":
        wrapper = load_verified_contract(require_unconsumed=True)
        print(
            "ACE2_STAGE1_PER32_PREFLIGHT_PASS "
            f"candidate={CANDIDATE_ID} tensors=168 attempts=0 "
            f"contract_sha256={wrapper['contract_sha256']}"
        )
        return 0
    if args.command == "execute":
        return execute_once()
    summary = verify_result()
    print("ACE2_STAGE1_PER32_RESULT_VERIFIED " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
