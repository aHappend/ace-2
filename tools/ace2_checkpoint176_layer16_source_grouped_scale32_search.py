#!/usr/bin/env python3
"""Score the frozen layer-16 five-group repair with exact Scale32 arithmetic."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_checkpoint176_hidden_state_localizer as localizer
from tools import ace2_checkpoint176_layer17_mechanism as mechanism
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner
from tools.ace2_quality_contracts import (
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    round_divide_even_signed,
    round_divide_even_unsigned,
    scale32_ratio,
    unpack_scale32,
)
from tools.ace2_rmsnorm_reference import derive_scaled_gains_q8, reference_rmsnorm


SOURCE_LAYER = 16
TARGET_LAYER = mechanism.LAYER
FROZEN_GROUP_SIZES = (32, 16, 8, 4, 1)
Q_PATTERN = re.compile(r"layer17_position(\d+)_q$")
PREDECESSOR = (
    ROOT / "evidence/candidates/w4a8-chat-hidden-state-localization-v1/candidate-0002"
)
FROZEN_BINARY32 = (
    ROOT
    / "evidence/candidates/w4a8-layer16-source-grouped-activation-repair-v1/"
    "candidate-0001"
)


def _scale32_float(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def _saturate_s8(value: int) -> tuple[int, bool]:
    return max(-128, min(127, value)), value < -128 or value > 127


def _requantize_scale32_s8(
    value: int, source_scale32: int, target_scale32: int
) -> tuple[int, bool]:
    localizer.require(-128 <= value <= 127, "source sample escaped signed A8")
    source_sig, source_exp = unpack_scale32(source_scale32)
    target_sig, target_exp = unpack_scale32(target_scale32)
    common_exp = min(source_exp, target_exp)
    numerator = value * source_sig * (1 << (source_exp - common_exp))
    denominator = target_sig * (1 << (target_exp - common_exp))
    return _saturate_s8(round_divide_even_signed(numerator, denominator))


def _derive_scale32_multiplier(
    input_scale32: int, weight_scale32: int, output_scale32: int
) -> tuple[int, int]:
    input_num, input_den = scale32_ratio(input_scale32)
    weight_num, weight_den = scale32_ratio(weight_scale32)
    output_num, output_den = scale32_ratio(output_scale32)
    numerator = input_num * weight_num * output_den
    denominator = input_den * weight_den * output_num
    for shift in range(63, -1, -1):
        multiplier = round_divide_even_unsigned(numerator << shift, denominator)
        if multiplier <= (1 << 31) - 1:
            return multiplier, shift
    raise OverflowError("Scale32 projection multiplier is not representable")


def _binary32_grouped_reconstruct(
    source_s8: torch.Tensor, base_scale: float, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the frozen binary32 calibration path without scoring it."""
    localizer.require(
        source_s8.numel() % group_size == 0, "source group does not divide hidden"
    )
    source = source_s8.to(torch.float64) * float(base_scale)
    reconstructed = torch.empty_like(source)
    quantized = torch.empty(source.numel(), dtype=torch.int8)
    scales = torch.empty(source.numel() // group_size, dtype=torch.float32)
    for group_id, start in enumerate(range(0, source.numel(), group_size)):
        stop = start + group_size
        group = source[start:stop]
        scale = torch.tensor(
            max(float(group.abs().max()) / 127.0, 1.0e-12), dtype=torch.float32
        )
        quantized[start:stop] = torch.round(group / float(scale)).clamp(-128, 127).to(
            torch.int8
        )
        reconstructed[start:stop] = quantized[start:stop].to(torch.float64) * float(
            scale
        )
        scales[group_id] = scale
    return reconstructed, quantized, scales


def _exact_group_scales(
    source_s8: torch.Tensor, base_scale32: int, group_size: int
) -> tuple[int, ...]:
    base_num, base_den = scale32_ratio(base_scale32)
    scales = []
    for start in range(0, source_s8.numel(), group_size):
        maximum = int(
            source_s8[start : start + group_size].to(torch.int64).abs().max().item()
        )
        scales.append(ceil_scale32_from_ratio(maximum * base_num, 127 * base_den))
    return tuple(scales)


def _exact_group_requantize(
    source_s8: torch.Tensor, base_scale32: int, group_scales: tuple[int, ...], group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(source_s8.numel(), dtype=torch.int8)
    saturation = torch.zeros(source_s8.numel(), dtype=torch.uint8)
    for index, value in enumerate(source_s8.to(torch.int64).tolist()):
        converted, saturated = _requantize_scale32_s8(
            int(value), base_scale32, group_scales[index // group_size]
        )
        output[index] = converted
        saturation[index] = int(saturated)
    return output, saturation


def _exact_sum_scale32(
    attention_s8: torch.Tensor,
    down_s8: torch.Tensor,
    attention_scales: tuple[int, ...],
    down_scales: tuple[int, ...],
    group_size: int,
) -> int:
    maximum = Fraction(0, 1)
    for index, (attention, down) in enumerate(
        zip(
            attention_s8.to(torch.int64).tolist(),
            down_s8.to(torch.int64).tolist(),
            strict=True,
        )
    ):
        attention_num, attention_den = scale32_ratio(
            attention_scales[index // group_size]
        )
        down_num, down_den = scale32_ratio(down_scales[index // group_size])
        value = Fraction(int(attention) * attention_num, attention_den) + Fraction(
            int(down) * down_num, down_den
        )
        maximum = max(maximum, abs(value))
    return ceil_scale32_from_ratio(maximum.numerator, maximum.denominator * 127)


def _exact_aligned_sum(
    attention_s8: torch.Tensor,
    down_s8: torch.Tensor,
    attention_scales: tuple[int, ...],
    down_scales: tuple[int, ...],
    sum_scale32: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(attention_s8.numel(), dtype=torch.int8)
    saturation = torch.zeros(attention_s8.numel(), dtype=torch.uint8)
    sum_sig, sum_exp = unpack_scale32(sum_scale32)
    for index, (attention, down) in enumerate(
        zip(
            attention_s8.to(torch.int64).tolist(),
            down_s8.to(torch.int64).tolist(),
            strict=True,
        )
    ):
        attention_sig, attention_exp = unpack_scale32(
            attention_scales[index // group_size]
        )
        down_sig, down_exp = unpack_scale32(down_scales[index // group_size])
        common_exp = min(attention_exp, down_exp, sum_exp)
        numerator = (
            int(attention) * attention_sig * (1 << (attention_exp - common_exp))
            + int(down) * down_sig * (1 << (down_exp - common_exp))
        )
        denominator = sum_sig * (1 << (sum_exp - common_exp))
        converted, saturated = _saturate_s8(
            round_divide_even_signed(numerator, denominator)
        )
        output[index] = converted
        saturation[index] = int(saturated)
    return output, saturation


def _dequantize_scale32(value: torch.Tensor, scale32: int) -> torch.Tensor:
    return value.to(torch.float64) * _scale32_float(scale32)


class ExactScale32QProjectionCache(localizer.FastProjectionCache):
    """Replace only layer-17 Q with the exact Scale32 RMSNorm/W4A8 payload."""

    def __init__(self, weights: Any, adapter: Any) -> None:
        super().__init__(weights, adapter)
        name = f"model.layers.{TARGET_LAYER}.self_attn.q_proj"
        self.q_merged = self.merged[name]
        self.q_metadata = self.metadata[self.q_merged.data_ptr()]
        self.q_weight_scale32 = tuple(
            ceil_scale32_from_float(float(value))
            for value in self.q_metadata["weight_scale"].tolist()
        )
        self.position_records: dict[int, dict[str, Any]] = {}

    def configure(self) -> None:
        self.position_records.clear()

    def derive_projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        match = Q_PATTERN.fullmatch(name)
        if match is None:
            return super().derive_projection(
                name, merged, input_q, input_scale, float_input, source_hashes
            )

        position = int(match.group(1))
        record = self.position_records[position]
        binary_input_norm = record["binary_input_norm"].to(torch.float32)
        float_output = torch.mv(merged, binary_input_norm).contiguous()
        binary_output_scale = backend.canonical.scale_for(float_output)
        output_scale32 = ceil_scale32_from_float(float(binary_output_scale))
        output_scale = _scale32_float(output_scale32)
        input_scale32 = int(record["q_input_scale32"])
        exact_input_q = record["rms_output_s8"]
        qweight = self.q_metadata["qweight"]
        accumulator = torch.mv(qweight.to(torch.int64), exact_input_q.to(torch.int64))
        localizer.require(
            bool(torch.all(accumulator >= -(1 << 31)))
            and bool(torch.all(accumulator < (1 << 31))),
            f"{name} accumulator overflow",
        )

        multiplier_values = []
        shift_values = []
        for weight_scale32 in self.q_weight_scale32:
            multiplier, shift = _derive_scale32_multiplier(
                input_scale32, weight_scale32, output_scale32
            )
            multiplier_values.append(multiplier)
            shift_values.append(shift)
        multiplier = torch.tensor(multiplier_values, dtype=torch.int64)
        right_shift = torch.tensor(shift_values, dtype=torch.int64)
        rounded = localizer.round_shift_even_tensor(accumulator * multiplier, right_shift)
        output_q = rounded.clamp(-128, 127).to(torch.int8)
        saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)

        record.update(
            {
                "q_weight_s4": qweight,
                "q_weight_scale32": torch.tensor(
                    self.q_weight_scale32, dtype=torch.int64
                ),
                "q_multiplier_s32": multiplier,
                "q_shift_u6": right_shift,
                "q_accumulator_s32": accumulator,
                "q_rounded_s64": rounded,
                "q_output_s8": output_q,
                "q_saturation": saturation,
                "q_output_scale32": output_scale32,
                "binary_q_output_scale": float(binary_output_scale),
                "binary_q_float_output": float_output,
            }
        )
        return {
            "name": name,
            "input_q": exact_input_q,
            "input_scale": _scale32_float(input_scale32),
            "qweight": qweight,
            "weight_scale": self.q_metadata["weight_scale"],
            "multiplier": multiplier,
            "right_shift": right_shift,
            "accumulator": accumulator,
            "rounded": rounded,
            "output_q": output_q,
            "output_scale": output_scale,
            "saturation": saturation,
            "float_output": float_output,
            "source_hashes": source_hashes,
            "exact_scale32": True,
        }


class ExactScale32SourceGuard:
    def __init__(
        self,
        cache: ExactScale32QProjectionCache,
        weights: Any,
        source_group_size: int,
    ) -> None:
        self.cache = cache
        self.weights = weights
        self.source_group_size = source_group_size
        self.original = backend.derive_layer_token
        self.records: dict[int, dict[str, Any]] = {}

    def install(self) -> None:
        def derive(
            layer_id: int,
            state: dict[str, Any],
            cache: dict[str, Any],
            template: dict[str, Any] | None,
            weights: Any,
            adapter: Any,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            if layer_id == TARGET_LAYER:
                position_id = int(state["position"])
                localizer.require(
                    position_id in self.records, "exact source record missing at layer 17"
                )
                self.cache.position_records[position_id] = self.records[position_id]
                if template is not None:
                    template["qkv"].pop("q", None)

            derived, next_state, next_template = self.original(
                layer_id, state, cache, template, weights, adapter
            )

            if layer_id == SOURCE_LAYER:
                position_id = int(state["position"])
                position = derived["positions"][0]
                attention_source_s8 = position["attention_residual"]["output"].contiguous()
                down_source_s8 = position["projections"]["down"]["output_q"].contiguous()
                attention_base_scale = float(position["attention_residual"]["scale"])
                down_base_scale = float(position["projections"]["down"]["output_scale"])

                binary_attention, binary_attention_q, binary_attention_scales = (
                    _binary32_grouped_reconstruct(
                        attention_source_s8,
                        attention_base_scale,
                        self.source_group_size,
                    )
                )
                binary_down, binary_down_q, binary_down_scales = (
                    _binary32_grouped_reconstruct(
                        down_source_s8,
                        down_base_scale,
                        self.source_group_size,
                    )
                )
                binary_sum = (binary_attention + binary_down).contiguous()
                gain = self.weights.get_tensor(
                    f"model.layers.{TARGET_LAYER}.input_layernorm.weight"
                ).contiguous()
                binary_input_norm = backend.canonical.float_rmsnorm(
                    binary_sum.to(torch.float32), gain
                ).contiguous()
                binary_q_input_scale = max(
                    float(binary_input_norm.to(torch.float64).abs().max()) / 127.0,
                    1.0e-12,
                )
                q_input_scale32 = ceil_scale32_from_float(binary_q_input_scale)

                attention_base_scale32 = ceil_scale32_from_float(attention_base_scale)
                down_base_scale32 = ceil_scale32_from_float(down_base_scale)
                attention_group_scale32 = _exact_group_scales(
                    attention_source_s8,
                    attention_base_scale32,
                    self.source_group_size,
                )
                down_group_scale32 = _exact_group_scales(
                    down_source_s8,
                    down_base_scale32,
                    self.source_group_size,
                )
                attention_group_s8, attention_saturation = _exact_group_requantize(
                    attention_source_s8,
                    attention_base_scale32,
                    attention_group_scale32,
                    self.source_group_size,
                )
                down_group_s8, down_saturation = _exact_group_requantize(
                    down_source_s8,
                    down_base_scale32,
                    down_group_scale32,
                    self.source_group_size,
                )
                sum_scale32 = _exact_sum_scale32(
                    attention_group_s8,
                    down_group_s8,
                    attention_group_scale32,
                    down_group_scale32,
                    self.source_group_size,
                )
                sum_s8, sum_saturation = _exact_aligned_sum(
                    attention_group_s8,
                    down_group_s8,
                    attention_group_scale32,
                    down_group_scale32,
                    sum_scale32,
                    self.source_group_size,
                )
                gains = derive_scaled_gains_q8(gain.to(torch.float64).tolist(), _scale32_float(q_input_scale32))
                rms = reference_rmsnorm(sum_s8.to(torch.int64).tolist(), gains)

                self.records[position_id] = {
                    "source_group_size": self.source_group_size,
                    "attention_source_s8": attention_source_s8,
                    "down_source_s8": down_source_s8,
                    "attention_base_scale32": attention_base_scale32,
                    "down_base_scale32": down_base_scale32,
                    "attention_group_scale32": torch.tensor(
                        attention_group_scale32, dtype=torch.int64
                    ),
                    "down_group_scale32": torch.tensor(
                        down_group_scale32, dtype=torch.int64
                    ),
                    "attention_group_s8": attention_group_s8,
                    "down_group_s8": down_group_s8,
                    "attention_saturation": attention_saturation,
                    "down_saturation": down_saturation,
                    "sum_scale32": sum_scale32,
                    "sum_s8": sum_s8,
                    "sum_saturation": sum_saturation,
                    "sum_dequantized": _dequantize_scale32(sum_s8, sum_scale32),
                    "rms_gain_s16_q8": torch.tensor(gains, dtype=torch.int16),
                    "rms_output_s8": torch.tensor(rms.outputs, dtype=torch.int8),
                    "rms_sumsq": rms.sumsq,
                    "rms_inv_q30": rms.inv_rms_q30,
                    "rms_saturation": rms.saturation_seen,
                    "q_input_scale32": q_input_scale32,
                    "binary_attention_q": binary_attention_q,
                    "binary_down_q": binary_down_q,
                    "binary_attention_scales": binary_attention_scales,
                    "binary_down_scales": binary_down_scales,
                    "binary_sum_reconstructed": binary_sum,
                    "binary_input_norm": binary_input_norm,
                    "binary_q_input_scale": binary_q_input_scale,
                }
            return derived, next_state, next_template

        backend.derive_layer_token = derive

    def restore(self) -> None:
        backend.derive_layer_token = self.original


def _tensor_artifacts(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    artifacts = {
        name: value for name, value in record.items() if isinstance(value, torch.Tensor)
    }
    for name in (
        "attention_base_scale32",
        "down_base_scale32",
        "sum_scale32",
        "q_input_scale32",
        "q_output_scale32",
        "rms_sumsq",
        "rms_inv_q30",
    ):
        artifacts[name] = torch.tensor([int(record[name])], dtype=torch.int64)
    artifacts["rms_saturation"] = torch.tensor(
        [int(record["rms_saturation"])], dtype=torch.uint8
    )
    return artifacts


def run_group(
    source_group_size: int,
    cache: ExactScale32QProjectionCache,
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cache.configure()
    guard = ExactScale32SourceGuard(cache, weights, source_group_size)
    cache.install()
    guard.install()
    try:
        result, _ = localizer.run_cut(
            None, [], weights, adapter, norm_gain, embedding, head, tokenizer
        )
    finally:
        guard.restore()
        cache.restore()

    position = len(localizer.TOKEN_IDS) - 1
    exact = guard.records[position]
    localizer.require("q_output_s8" in exact, "exact Q record missing")
    result.update(
        {
            "source_group_size": source_group_size,
            "source_group_count_per_stream": backend.HIDDEN // source_group_size,
            "attention_base_scale32": f"0x{int(exact['attention_base_scale32']):08x}",
            "down_base_scale32": f"0x{int(exact['down_base_scale32']):08x}",
            "attention_group_scale32_min": f"0x{int(exact['attention_group_scale32'].min()):08x}",
            "attention_group_scale32_max": f"0x{int(exact['attention_group_scale32'].max()):08x}",
            "down_group_scale32_min": f"0x{int(exact['down_group_scale32'].min()):08x}",
            "down_group_scale32_max": f"0x{int(exact['down_group_scale32'].max()):08x}",
            "sum_scale32": f"0x{int(exact['sum_scale32']):08x}",
            "q_input_scale32": f"0x{int(exact['q_input_scale32']):08x}",
            "q_output_scale32": f"0x{int(exact['q_output_scale32']):08x}",
            "source_saturation_count": int(
                exact["attention_saturation"].to(torch.int64).sum()
                + exact["down_saturation"].to(torch.int64).sum()
            ),
            "sum_saturation_count": int(exact["sum_saturation"].to(torch.int64).sum()),
            "rms_saturation": bool(exact["rms_saturation"]),
            "q_saturation_count": int(exact["q_saturation"].to(torch.int64).sum()),
            "sum_s8_sha256": localizer.tensor_sha256(exact["sum_s8"]),
            "rms_output_s8_sha256": localizer.tensor_sha256(exact["rms_output_s8"]),
            "q_output_s8_sha256": localizer.tensor_sha256(exact["q_output_s8"]),
            "binary32_calibration_q_input_scale": float(
                exact["binary_q_input_scale"]
            ),
            "binary32_calibration_q_output_scale": float(
                exact["binary_q_output_scale"]
            ),
        }
    )
    return result, _tensor_artifacts(exact)


def _verify_existing_group4_payload(artifacts: dict[str, torch.Tensor]) -> dict[str, Any]:
    from tools.ace2_layer16_group4_scale32_reference import derive_model_vectors

    expected = derive_model_vectors()
    comparisons = {
        "attention_source_s8": (
            artifacts["attention_source_s8"],
            torch.tensor(expected.attention_source_s8, dtype=torch.int8),
        ),
        "down_source_s8": (
            artifacts["down_source_s8"],
            torch.tensor(expected.down_source_s8, dtype=torch.int8),
        ),
        "attention_group_scale32": (
            artifacts["attention_group_scale32"],
            torch.tensor(expected.attention_group_scale32, dtype=torch.int64),
        ),
        "down_group_scale32": (
            artifacts["down_group_scale32"],
            torch.tensor(expected.down_group_scale32, dtype=torch.int64),
        ),
        "attention_group_s8": (
            artifacts["attention_group_s8"],
            torch.tensor(expected.attention_group_s8, dtype=torch.int8),
        ),
        "down_group_s8": (
            artifacts["down_group_s8"],
            torch.tensor(expected.down_group_s8, dtype=torch.int8),
        ),
        "sum_s8": (artifacts["sum_s8"], torch.tensor(expected.sum_s8, dtype=torch.int8)),
        "rms_gain_s16_q8": (
            artifacts["rms_gain_s16_q8"],
            torch.tensor(expected.rms_gain_s16_q8, dtype=torch.int16),
        ),
        "rms_output_s8": (
            artifacts["rms_output_s8"],
            torch.tensor(expected.rms_output_s8, dtype=torch.int8),
        ),
        "q_weight_s4": (
            artifacts["q_weight_s4"],
            torch.tensor(expected.q_weight_s4, dtype=torch.int8),
        ),
        "q_multiplier_s32": (
            artifacts["q_multiplier_s32"],
            torch.tensor(expected.q_multiplier_s32, dtype=torch.int64),
        ),
        "q_shift_u6": (
            artifacts["q_shift_u6"],
            torch.tensor(expected.q_shift_u6, dtype=torch.int64),
        ),
        "q_accumulator_s32": (
            artifacts["q_accumulator_s32"],
            torch.tensor(expected.q_accumulator_s32, dtype=torch.int64),
        ),
        "q_output_s8": (
            artifacts["q_output_s8"],
            torch.tensor(expected.q_output_s8, dtype=torch.int8),
        ),
        "q_saturation": (
            artifacts["q_saturation"],
            torch.tensor(expected.q_saturation, dtype=torch.uint8),
        ),
    }
    mismatches = {
        name: int(torch.count_nonzero(observed != reference).item())
        for name, (observed, reference) in comparisons.items()
    }
    scalar_mismatches = {
        "attention_base_scale32": int(artifacts["attention_base_scale32"][0])
        != expected.attention_base_scale32,
        "down_base_scale32": int(artifacts["down_base_scale32"][0])
        != expected.down_base_scale32,
        "sum_scale32": int(artifacts["sum_scale32"][0]) != expected.sum_scale32,
        "q_input_scale32": int(artifacts["q_input_scale32"][0])
        != expected.q_input_scale32,
        "q_output_scale32": int(artifacts["q_output_scale32"][0])
        != expected.q_output_scale32,
    }
    localizer.require(
        not any(mismatches.values()) and not any(scalar_mismatches.values()),
        f"group-4 search payload differs from existing RTL vectors: {mismatches} {scalar_mismatches}",
    )
    return {
        "status": "PASS_BIT_EXACT_EXISTING_GROUP4_RTL_PAYLOAD",
        "tensor_mismatches": mismatches,
        "scalar_mismatches": scalar_mismatches,
        "generated_manifest": localizer.file_record(
            ROOT / "verification/generated/ace2_layer16_group4_scale32/manifest.json"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    localizer.require(ROOT.resolve() in output.parents, "output must be repository-relative")
    localizer.require(not output.exists(), "candidate output already exists")

    frozen_freeze_path = FROZEN_BINARY32 / "candidate-freeze.json"
    frozen_result_path = FROZEN_BINARY32 / "result.json"
    frozen_freeze = json.loads(frozen_freeze_path.read_text(encoding="utf-8"))
    frozen_result = json.loads(frozen_result_path.read_text(encoding="utf-8"))
    source_group_sizes = tuple(int(value) for value in frozen_freeze["ordered_source_group_sizes"])
    localizer.require(source_group_sizes == FROZEN_GROUP_SIZES, "frozen group order differs")
    localizer.require(
        frozen_freeze["prompt"]["chat_token_ids"] == localizer.TOKEN_IDS,
        "frozen canonical prefix differs",
    )
    localizer.require(
        frozen_freeze["reference_step0_token_id"] == localizer.REFERENCE_TOKEN,
        "frozen reference token differs",
    )
    baseline_rank = int(frozen_freeze["baseline"]["reference_rank"])
    baseline_top = int(frozen_freeze["baseline"]["top_token_id"])

    output.mkdir(parents=True)
    freeze = {
        "schema_version": 1,
        "mission": "w4a8-layer16-source-grouped-activation-repair-v1",
        "classification": "candidate_only_no_official_attempt_created",
        "checkpoint": frozen_freeze["checkpoint"],
        "prompt": frozen_freeze["prompt"],
        "reference_step0_token_id": localizer.REFERENCE_TOKEN,
        "baseline": {"top_token_id": baseline_top, "reference_rank": baseline_rank},
        "ordered_source_group_sizes": list(source_group_sizes),
        "score_policy": frozen_freeze["score_policy"],
        "numeric_contract": {
            "carried_source_samples": "two signed-A8 streams",
            "source_base_scale": "ceil binary32 producer scale to smallest legal Scale32",
            "source_group_scale": "exact Scale32 max-abs/127 from source A8 and base Scale32",
            "sum": "exact source alignment and one signed ties-to-even A8 rounding into common Scale32",
            "rmsnorm": "accepted integer RMSNorm over exact common-scale sum A8",
            "q_projection": "unchanged signed W4 with exact Scale32-derived per-row multipliers",
            "q_input_and_output_calibration": "frozen binary32 exploratory calibration per causal position, transported upward to Scale32 exactly as the focused RTL",
            "token_or_logit_override": False,
            "bf16_runtime_sidecar": False,
        },
        "scope_guards": frozen_freeze["scope_guards"],
        "bindings": {
            "frozen_binary32_freeze": localizer.file_record(frozen_freeze_path),
            "frozen_binary32_result": localizer.file_record(frozen_result_path),
            "frozen_binary32_sha256s": localizer.file_record(
                FROZEN_BINARY32 / "SHA256SUMS"
            ),
            "predecessor_freeze": frozen_freeze["bindings"]["predecessor_freeze"],
            "predecessor_scan": frozen_freeze["bindings"]["predecessor_scan"],
            "runner": localizer.file_record(Path(__file__).resolve()),
        },
    }
    localizer.write_json(output / "candidate-freeze.json", freeze)

    started = time.monotonic()
    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )

    trace = []
    group_artifacts: dict[int, dict[str, torch.Tensor]] = {}
    with safe_open(backend.canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        cache = ExactScale32QProjectionCache(weights, adapter)
        for group_size in source_group_sizes:
            record, tensors = run_group(
                group_size,
                cache,
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
            )
            group_dir = output / f"groups/group-{group_size:03d}"
            artifacts = {
                name: localizer.write_tensor(group_dir / f"{name}.bin", value)
                for name, value in sorted(tensors.items())
            }
            localizer.write_json(
                group_dir / "result.json", {"metrics": record, "artifacts": artifacts}
            )
            group_artifacts[group_size] = tensors
            trace.append(record)
            print(
                "ACE2_LAYER16_SOURCE_GROUPED_SCALE32_CANDIDATE "
                f"group={group_size} top={record['top_token_id']} "
                f"rank={record['reference_rank']} q_sat={record['q_saturation_count']}",
                flush=True,
            )

    restored = [item for item in trace if item["top_token_id"] == localizer.REFERENCE_TOKEN]
    improved = [item for item in trace if int(item["reference_rank"]) < baseline_rank]
    if restored:
        selected = restored[0]
        selection_reason = "restored_step0_in_declared_execution_order"
    elif improved:
        order = {group: index for index, group in enumerate(source_group_sizes)}
        selected = min(
            improved,
            key=lambda item: (
                int(item["reference_rank"]),
                order[int(item["source_group_size"])],
            ),
        )
        selection_reason = "best_honest_reference_rank_improvement"
    else:
        selected = None
        selection_reason = "no_group_improved_baseline_rank"

    rtl_payload_binding = None
    if selected is not None and int(selected["source_group_size"]) == 4:
        rtl_payload_binding = _verify_existing_group4_payload(group_artifacts[4])

    restored_step0 = selected is not None and selected["top_token_id"] == localizer.REFERENCE_TOKEN
    result = {
        "schema_version": 1,
        "status": (
            "PASS_LAYER16_SOURCE_GROUPED_SCALE32_STEP0_RESTORED"
            if restored_step0
            else "PARTIAL_LAYER16_SOURCE_GROUPED_SCALE32_RANK_IMPROVED"
            if selected is not None
            else "PARTIAL_LAYER16_SOURCE_GROUPED_SCALE32_NO_IMPROVEMENT"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "baseline_reference_rank": baseline_rank,
        "baseline_top_token_id": baseline_top,
        "ordered_source_group_sizes": list(source_group_sizes),
        "trace": trace,
        "selected_source_group_size": (
            int(selected["source_group_size"]) if selected is not None else None
        ),
        "selected_reference_rank": (
            int(selected["reference_rank"]) if selected is not None else baseline_rank
        ),
        "selected_top_token_id": (
            int(selected["top_token_id"]) if selected is not None else baseline_top
        ),
        "selection_reason": selection_reason,
        "first_token_restored": bool(restored_step0),
        "rtl_payload_binding": rtl_payload_binding,
        "next_exact_source_boundary": (
            None
            if restored_step0
            else "model.layers.17.input_rmsnorm_to_self_attn.q_proj.output"
        ),
        "next_required_action": (
            "Bind the selected exact Scale32 payload to focused warning/X-clean RTL verification."
            if selected is not None
            else "Preserve all five negative results and advance to the exact layer17 Q-output activation boundary."
        ),
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
        "bindings": {
            "freeze": localizer.file_record(output / "candidate-freeze.json"),
            "frozen_binary32_sha256s": localizer.file_record(
                FROZEN_BINARY32 / "SHA256SUMS"
            ),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
        },
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    print(
        "ACE2_LAYER16_SOURCE_GROUPED_SCALE32_RESULT "
        f"status={result['status']} selected={result['selected_source_group_size']} "
        f"rank={result['selected_reference_rank']} restored={result['first_token_restored']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ACE2_LAYER16_SOURCE_GROUPED_SCALE32_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
