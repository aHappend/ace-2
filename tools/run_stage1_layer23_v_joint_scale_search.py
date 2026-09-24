#!/usr/bin/env python3
"""Run the bounded nonofficial layer-23 V joint W4/A8 Scale32 search."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import peft
import tokenizers
import torch
import transformers
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_dynamic_scale32_baseline as scale32
from tools import diagnose_stage1_layer23_qkv_families as diagnosis
from tools import run_stage1_layer23_v_precision_matrix as precision


MISSION_ID = "stage1vjointscale01"
LAYER = 23
TARGET_TOKEN = 39814
TOP_K = 8
OUTPUT = (
    ROOT
    / "evidence/candidates/stage1-layer23-v-joint-scale-search-v1/"
    "nonofficial-search-0001"
)
FREEZE = OUTPUT / "candidate-freeze.json"
RESULT = OUTPUT / "result.json"
REVIEW_REQUEST = OUTPUT / "fresh-l2-review-request.json"
SUMS = OUTPUT / "SHA256SUMS"
EXECUTION_RUNNER = OUTPUT / "joint-scale-execution-runner.py"
ACCEPTED_DIAGNOSIS = ROOT / "diagnosis/stage1layer23qkvfamilies01/result.json"
ACCEPTED_DIAGNOSIS_SUMS = (
    ROOT / "diagnosis/stage1layer23qkvfamilies01/SHA256SUMS"
)
PRECISION_MATRIX = (
    ROOT
    / "evidence/candidates/stage1-layer23-v-precision-matrix-v1/"
    "nonofficial-matrix-0001/result.json"
)
PRECISION_MATRIX_SUMS = PRECISION_MATRIX.parent / "SHA256SUMS"
BF16_LOGITS = (
    ROOT
    / "diagnosis/stage1qualitydiag01/logits/"
    "checkpoint176-bf16-greedy-step-00-f32le.bin"
)
V_PATTERN = precision.V_PATTERN
CALIBRATION_POSITIONS = tuple(range(17))
HELD_OUT_POSITIONS = tuple(range(17, 34))
BASELINE_INPUT_SCALE = 0.2578740157480315
RETAINED_OUTPUT_SCALE = 0.06586176203930472
EXPECTED_BASELINE = {
    "logits_sha256": "0b80dbb919618cb7f39a8a862cac652ba684176a520865020595bd048085ea17",
    "target_rank": 15,
    "top8_overlap_count": 0,
    "jensen_shannon_divergence_nats": 0.6701229933515445,
    "relative_l2": 0.9423817179633847,
    "top_token_id": 55725,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    return diagnosis.prior.localizer.file_record(path)


def write_tensor(path: Path, value: torch.Tensor, dtype: str) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(diagnosis.prior.localizer.tensor_bytes(tensor))
    record = diagnosis.prior.localizer.file_record(path)
    record.update({"dtype": dtype, "shape": list(tensor.shape)})
    return record


def verify_sum_tree(path: Path, expected_hash: str) -> None:
    localizer = diagnosis.prior.localizer
    require(localizer.sha256_file(path) == expected_hash, f"{path.name} hash changed")
    for line in path.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        artifact = path.parent / relative
        require(artifact.is_file(), f"protected artifact missing: {relative}")
        require(
            localizer.sha256_file(artifact) == expected,
            f"protected artifact changed: {relative}",
        )


def encode_scale32_scalar(value: float) -> int:
    return scale32.ceil_scale32_from_float(float(value))


def decode_scale32_scalar(record: int) -> float:
    numerator, denominator = scale32.scale32_ratio(int(record))
    return numerator / denominator


def encode_scale32_tensor(values: torch.Tensor) -> torch.Tensor:
    records = [
        encode_scale32_scalar(float(value))
        for value in values.detach().cpu().reshape(-1).tolist()
    ]
    return torch.tensor(records, dtype=torch.int64).reshape(values.shape)


def decode_scale32_tensor(records: torch.Tensor) -> torch.Tensor:
    values = [
        decode_scale32_scalar(int(record))
        for record in records.detach().cpu().reshape(-1).tolist()
    ]
    return torch.tensor(values, dtype=torch.float64).reshape(records.shape)


def scale32_detail(record: int, source_value: float) -> dict[str, Any]:
    significand, exponent = scale32.unpack_scale32(int(record))
    return {
        "record_u32": int(record),
        "record_hex": f"0x{int(record):08x}",
        "significand_u16": significand,
        "exponent_s8": exponent,
        "decoded_value": decode_scale32_scalar(record),
        "source_value": float(source_value),
    }


def scale32_records_detail(
    records: torch.Tensor, source_values: torch.Tensor
) -> dict[str, Any]:
    integer_records = [int(value) for value in records.reshape(-1).tolist()]
    decoded = [decode_scale32_scalar(value) for value in integer_records]
    return {
        "count": len(integer_records),
        "records_u32": integer_records,
        "records_hex": [f"0x{value:08x}" for value in integer_records],
        "decoded_values": decoded,
        "source_values": [
            float(value) for value in source_values.reshape(-1).tolist()
        ],
    }


def quantize_input(
    source: torch.Tensor, input_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    rounded = torch.round(source.to(torch.float64) / input_scale)
    return rounded.clamp(-128, 127).to(torch.int8), rounded


def quantize_weight(
    source: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rounded = torch.round(source.to(torch.float64) / weight_scale[:, None])
    return rounded.clamp(-8, 7).to(torch.int8), rounded


def fixed_projection_batch(
    input_q: torch.Tensor,
    qweight: torch.Tensor,
    multiplier: torch.Tensor,
    right_shift: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    accumulator = torch.matmul(
        input_q.to(torch.int64), qweight.to(torch.int64).transpose(0, 1)
    )
    require(
        bool(torch.all(accumulator >= -(1 << 31)))
        and bool(torch.all(accumulator < (1 << 31))),
        "candidate accumulator escaped signed 32 bits",
    )
    product = accumulator * multiplier[None, :]
    shifts = right_shift[None, :].expand_as(product)
    rounded = diagnosis.prior.localizer.round_shift_even_tensor(product, shifts)
    output_q = rounded.clamp(-128, 127).to(torch.int8)
    return output_q, rounded, accumulator


def error_stats(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate_f64 = candidate.to(torch.float64)
    reference_f64 = reference.to(torch.float64)
    difference = candidate_f64 - reference_f64
    denominator = torch.linalg.vector_norm(reference_f64)
    return {
        "count": difference.numel(),
        "mean_absolute_error": float(difference.abs().mean().item()),
        "maximum_absolute_error": float(difference.abs().max().item()),
        "mean_squared_error": float(torch.mean(difference * difference).item()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference).item() / denominator.item()
        ),
    }


def saturation_stats(rounded: torch.Tensor, lower: int, upper: int) -> dict[str, Any]:
    saturated = (rounded < lower) | (rounded > upper)
    count = int(saturated.sum().item())
    return {
        "count": count,
        "total": saturated.numel(),
        "rate": count / saturated.numel(),
        "below_count": int((rounded < lower).sum().item()),
        "above_count": int((rounded > upper).sum().item()),
    }


def subset_stats(
    input_source: torch.Tensor,
    input_q: torch.Tensor,
    input_scale: float,
    output_q: torch.Tensor,
    output_scale: float,
    oracle_v: torch.Tensor,
    positions: tuple[int, ...],
) -> dict[str, Any]:
    index = torch.tensor(positions, dtype=torch.int64)
    source = input_source.index_select(0, index)
    quantized_input = input_q.index_select(0, index)
    candidate_v = output_q.index_select(0, index).to(torch.float64) * output_scale
    reference_v = oracle_v.index_select(0, index)
    input_rounded = torch.round(source.to(torch.float64) / input_scale)
    return {
        "positions": list(positions),
        "input": {
            "saturation": saturation_stats(input_rounded, -128, 127),
            "error": error_stats(
                quantized_input.to(torch.float64) * input_scale, source
            ),
        },
        "v_projection_output": {
            "error": error_stats(candidate_v, reference_v),
        },
    }


def weight_stats(
    source: torch.Tensor, qweight: torch.Tensor, weight_scale: torch.Tensor
) -> dict[str, Any]:
    rounded = torch.round(source.to(torch.float64) / weight_scale[:, None])
    return {
        "saturation": saturation_stats(rounded, -8, 7),
        "error": error_stats(
            qweight.to(torch.float64) * weight_scale[:, None], source
        ),
    }


def candidate_configuration_value(
    candidate_id: str,
    input_factor: dict[str, Any],
    input_scale_record: int,
    weight_factor_order: list[dict[str, Any]],
    weight_selector: torch.Tensor,
    weight_scale_records: torch.Tensor,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "layer": LAYER,
        "component": "v",
        "input_factor": copy.deepcopy(input_factor),
        "input_scale32_record_u32": int(input_scale_record),
        "weight_factor_order": copy.deepcopy(weight_factor_order),
        "weight_factor_selector_by_output_row": [
            int(value) for value in weight_selector.tolist()
        ],
        "weight_scale32_records_u32": [
            int(value) for value in weight_scale_records.tolist()
        ],
        "weight_layout": (
            "unchanged signed-W4 output-channel-major row-major payload; "
            "one Scale32 scale per 896-element output row"
        ),
        "input_layout": "unchanged tensor-wide signed-A8 input and one Scale32 scale",
        "output_layout": "unchanged tensor-wide signed-A8 output and one Scale32 scale",
    }


def build_candidates(
    cache: "JointScaleCache",
    calibration_input: torch.Tensor,
    calibration_v: torch.Tensor,
    freeze: dict[str, Any],
) -> list[dict[str, Any]]:
    name = f"model.layers.{LAYER}.self_attn.v_proj"
    merged = cache.merged[name].to(torch.float64)
    metadata = cache.metadata[cache.merged[name].data_ptr()]
    baseline_weight_scale = metadata["weight_scale"].to(torch.float64)
    factor_order = freeze["search_policy"]["weight_factor_order"]
    candidates: list[dict[str, Any]] = []
    for input_factor in freeze["search_policy"]["input_factor_order"]:
        raw_input_scale = (
            BASELINE_INPUT_SCALE
            * int(input_factor["numerator"])
            / int(input_factor["denominator"])
        )
        input_record = encode_scale32_scalar(raw_input_scale)
        input_scale = decode_scale32_scalar(input_record)
        input_q, input_rounded = quantize_input(calibration_input, input_scale)
        scale_choices: list[torch.Tensor] = []
        record_choices: list[torch.Tensor] = []
        qweight_choices: list[torch.Tensor] = []
        sse_choices: list[torch.Tensor] = []
        for weight_factor in factor_order:
            raw_weight_scale = (
                baseline_weight_scale
                * int(weight_factor["numerator"])
                / int(weight_factor["denominator"])
            )
            weight_records = encode_scale32_tensor(raw_weight_scale)
            weight_scale = decode_scale32_tensor(weight_records)
            qweight, _ = quantize_weight(merged, weight_scale)
            multiplier, right_shift = (
                diagnosis.prior.backend.canonical.derive_multiplier(
                    input_scale * weight_scale / RETAINED_OUTPUT_SCALE
                )
            )
            output_q, _, _ = fixed_projection_batch(
                input_q, qweight, multiplier, right_shift
            )
            difference = (
                output_q.to(torch.float64) * RETAINED_OUTPUT_SCALE
                - calibration_v.to(torch.float64)
            )
            scale_choices.append(weight_scale)
            record_choices.append(weight_records)
            qweight_choices.append(qweight)
            sse_choices.append(torch.sum(difference * difference, dim=0))
        sse = torch.stack(sse_choices)
        selector = torch.argmin(sse, dim=0)
        rows = torch.arange(merged.shape[0], dtype=torch.int64)
        weight_scale = torch.stack(scale_choices)[selector, rows]
        weight_records = torch.stack(record_choices)[selector, rows]
        qweight = torch.stack(qweight_choices)[selector, rows, :]
        multiplier, right_shift = (
            diagnosis.prior.backend.canonical.derive_multiplier(
                input_scale * weight_scale / RETAINED_OUTPUT_SCALE
            )
        )
        output_q, output_rounded, accumulator = fixed_projection_batch(
            input_q, qweight, multiplier, right_shift
        )
        candidate_id = input_factor["candidate_id"]
        configuration = candidate_configuration_value(
            candidate_id,
            input_factor,
            input_record,
            factor_order,
            selector,
            weight_records,
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "input_factor": copy.deepcopy(input_factor),
                "input_scale_source": raw_input_scale,
                "input_scale_record": input_record,
                "input_scale": input_scale,
                "weight_scale_source": baseline_weight_scale
                * torch.tensor(
                    [
                        int(factor_order[int(index)]["numerator"])
                        / int(factor_order[int(index)]["denominator"])
                        for index in selector.tolist()
                    ],
                    dtype=torch.float64,
                ),
                "weight_scale_records": weight_records,
                "weight_scale": weight_scale,
                "weight_selector": selector,
                "qweight": qweight,
                "multiplier": multiplier.to(torch.int64),
                "right_shift": right_shift.to(torch.int64),
                "configuration": {
                    "value": configuration,
                    "sha256": diagnosis.prior.localizer.sha256_bytes(
                        canonical_bytes(configuration)
                    ),
                },
                "calibration": {
                    "input_saturation": saturation_stats(
                        input_rounded, -128, 127
                    ),
                    "output_saturation": saturation_stats(
                        output_rounded, -128, 127
                    ),
                    "projection_error": error_stats(
                        output_q.to(torch.float64) * RETAINED_OUTPUT_SCALE,
                        calibration_v,
                    ),
                    "accumulator_min": int(accumulator.min().item()),
                    "accumulator_max": int(accumulator.max().item()),
                    "weight_factor_selection_counts": {
                        factor_order[index]["factor_id"]: int(
                            (selector == index).sum().item()
                        )
                        for index in range(len(factor_order))
                    },
                },
            }
        )
    return candidates


class JointScaleCache(diagnosis.prior.localizer.FastProjectionCache):
    """Change only layer-23 V W4 row scales and its tensor-wide A8 input scale."""

    def __init__(
        self,
        weights: Any,
        adapter: Any,
        captures: dict[str, torch.Tensor],
    ) -> None:
        super().__init__(weights, adapter)
        self.captures = captures
        self.active: dict[str, Any] | None = None
        self.records: list[dict[str, torch.Tensor | int]] = []
        self.state: dict[str, Any] = {}

    def configure(self, candidate: dict[str, Any] | None) -> None:
        self.active = candidate
        self.records = []
        self.state = {}

    def _candidate_input(self, position: int) -> torch.Tensor:
        require(self.active is not None, "candidate input requested for baseline")
        input_q, _ = quantize_input(
            self.captures["input_norm"][position], self.active["input_scale"]
        )
        return input_q

    def _record(self, position: int, result: dict[str, Any]) -> None:
        if position == 0:
            if self.active is None:
                weight_scale = result["weight_scale"].to(torch.float64)
                weight_scale_records = encode_scale32_tensor(weight_scale)
                input_scale_record = encode_scale32_scalar(result["input_scale"])
            else:
                weight_scale = self.active["weight_scale"]
                weight_scale_records = self.active["weight_scale_records"]
                input_scale_record = self.active["input_scale_record"]
            self.state = {
                "qweight": result["qweight"].detach().cpu().contiguous(),
                "weight_scale": weight_scale.detach().cpu().contiguous(),
                "weight_scale_records": weight_scale_records.detach()
                .cpu()
                .contiguous(),
                "input_scale": float(result["input_scale"]),
                "input_scale_record": int(input_scale_record),
                "output_scale": float(result["output_scale"]),
                "output_scale_record": encode_scale32_scalar(
                    float(result["output_scale"])
                ),
                "multiplier": result["multiplier"].detach().cpu().contiguous(),
                "right_shift": result["right_shift"].detach().cpu().contiguous(),
            }
        self.records.append(
            {
                "position": position,
                "input_q": result["input_q"].detach().cpu().contiguous(),
                "output_q": result["output_q"].detach().cpu().contiguous(),
                "accumulator": result["accumulator"].detach().cpu().contiguous(),
                "rounded": result["rounded"].detach().cpu().contiguous(),
                "saturation": result["saturation"].detach().cpu().contiguous(),
            }
        )

    def derive_projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        match = V_PATTERN.fullmatch(name)
        if match is None:
            return super().derive_projection(
                name, merged, input_q, input_scale, float_input, source_hashes
            )
        position = int(match.group("position"))
        if self.active is None:
            result = super().derive_projection(
                name, merged, input_q, input_scale, float_input, source_hashes
            )
        else:
            output_scale = float(
                diagnosis.prior.backend.canonical.scale_for(
                    torch.mv(merged, float_input.to(torch.float32)).contiguous()
                )
            )
            require(
                output_scale == RETAINED_OUTPUT_SCALE,
                "candidate changed the retained A8 output scale",
            )
            result = self._fixed(
                name,
                self._candidate_input(position),
                self.active["input_scale"],
                self.active["qweight"],
                self.active["multiplier"],
                self.active["right_shift"],
                output_scale,
            )
            result.update(
                {
                    "weight_scale": self.active["weight_scale"],
                    "float_output": torch.mv(
                        merged, float_input.to(torch.float32)
                    ).contiguous(),
                    "source_hashes": source_hashes,
                }
            )
        self._record(position, result)
        return result

    def from_fixed_metadata(
        self,
        name: str,
        input_q: torch.Tensor,
        input_scale: float,
        qweight: torch.Tensor,
        multiplier: torch.Tensor,
        right_shift: torch.Tensor,
        output_scale: float,
    ) -> dict[str, Any]:
        match = V_PATTERN.fullmatch(name)
        if match is None:
            return super().from_fixed_metadata(
                name,
                input_q,
                input_scale,
                qweight,
                multiplier,
                right_shift,
                output_scale,
            )
        position = int(match.group("position"))
        if self.active is None:
            result = super().from_fixed_metadata(
                name,
                input_q,
                input_scale,
                qweight,
                multiplier,
                right_shift,
                output_scale,
            )
            result["weight_scale"] = self.state["weight_scale"]
        else:
            require(
                torch.equal(qweight, self.active["qweight"])
                and torch.equal(
                    multiplier.to(torch.int64), self.active["multiplier"]
                )
                and torch.equal(
                    right_shift.to(torch.int64), self.active["right_shift"]
                ),
                "candidate fixed metadata changed between positions",
            )
            result = self._fixed(
                name,
                self._candidate_input(position),
                self.active["input_scale"],
                self.active["qweight"],
                self.active["multiplier"],
                self.active["right_shift"],
                output_scale,
            )
            result["weight_scale"] = self.active["weight_scale"]
        self._record(position, result)
        return result

    def evidence(
        self,
        label: str,
        input_source: torch.Tensor,
        oracle_v: torch.Tensor,
        weight_source: torch.Tensor,
        candidate: dict[str, Any] | None,
    ) -> dict[str, Any]:
        require(len(self.records) == 34, f"{label} did not produce 34 V outputs")
        require(
            [int(record["position"]) for record in self.records] == list(range(34)),
            f"{label} V positions are not ordered",
        )
        input_q = torch.stack(
            [record["input_q"] for record in self.records]  # type: ignore[list-item]
        )
        output_q = torch.stack(
            [record["output_q"] for record in self.records]  # type: ignore[list-item]
        )
        accumulator = torch.stack(
            [record["accumulator"] for record in self.records]  # type: ignore[list-item]
        )
        rounded = torch.stack(
            [record["rounded"] for record in self.records]  # type: ignore[list-item]
        )
        directory = OUTPUT / "records" / label
        artifacts = {
            "input_q": write_tensor(directory / "v_input_s8.bin", input_q, "int8"),
            "qweight": write_tensor(
                directory / "v_qweight_s8.bin", self.state["qweight"], "int8"
            ),
            "weight_scale32": write_tensor(
                directory / "v_weight_scale32_u32le.bin",
                self.state["weight_scale_records"].to(torch.int32),
                "uint32_le",
            ),
            "weight_scale_decoded": write_tensor(
                directory / "v_weight_scale_f64le.bin",
                self.state["weight_scale"].to(torch.float64),
                "float64_le",
            ),
            "multiplier": write_tensor(
                directory / "v_multiplier_s32le.bin",
                self.state["multiplier"].to(torch.int32),
                "int32_le",
            ),
            "right_shift": write_tensor(
                directory / "v_right_shift_u8.bin",
                self.state["right_shift"].to(torch.uint8),
                "uint8",
            ),
            "output_q": write_tensor(
                directory / "v_output_s8.bin", output_q, "int8"
            ),
            "output_saturation": write_tensor(
                directory / "v_output_saturation_u8.bin",
                ((rounded < -128) | (rounded > 127)).to(torch.uint8),
                "uint8",
            ),
        }
        input_source_value = (
            BASELINE_INPUT_SCALE
            if candidate is None
            else float(candidate["input_scale_source"])
        )
        weight_source_values = (
            self.state["weight_scale"]
            if candidate is None
            else candidate["weight_scale_source"]
        )
        return {
            "artifacts": artifacts,
            "input_scale32": scale32_detail(
                self.state["input_scale_record"], input_source_value
            ),
            "weight_scale32": scale32_records_detail(
                self.state["weight_scale_records"], weight_source_values
            ),
            "output_scale32": scale32_detail(
                self.state["output_scale_record"], self.state["output_scale"]
            ),
            "multipliers_s32": [
                int(value) for value in self.state["multiplier"].tolist()
            ],
            "right_shifts_u6": [
                int(value) for value in self.state["right_shift"].tolist()
            ],
            "weight": weight_stats(
                weight_source, self.state["qweight"], self.state["weight_scale"]
            ),
            "calibration": subset_stats(
                input_source,
                input_q,
                self.state["input_scale"],
                output_q,
                self.state["output_scale"],
                oracle_v,
                CALIBRATION_POSITIONS,
            ),
            "held_out": subset_stats(
                input_source,
                input_q,
                self.state["input_scale"],
                output_q,
                self.state["output_scale"],
                oracle_v,
                HELD_OUT_POSITIONS,
            ),
            "full_34_positions": subset_stats(
                input_source,
                input_q,
                self.state["input_scale"],
                output_q,
                self.state["output_scale"],
                oracle_v,
                tuple(range(34)),
            ),
            "output_saturation": saturation_stats(rounded, -128, 127),
            "accumulator_observed_min": int(accumulator.min().item()),
            "accumulator_observed_max": int(accumulator.max().item()),
            "hardware_representable": (
                bool(torch.all(self.state["multiplier"] >= -(1 << 31)))
                and bool(torch.all(self.state["multiplier"] < (1 << 31)))
                and bool(torch.all(self.state["right_shift"] >= 0))
                and bool(torch.all(self.state["right_shift"] <= 63))
                and bool(torch.all(accumulator >= -(1 << 31)))
                and bool(torch.all(accumulator < (1 << 31)))
            ),
        }


def run_record(
    cache: JointScaleCache,
    candidate: dict[str, Any] | None,
    hidden_states: list[torch.Tensor],
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
    bf16_logits: torch.Tensor,
    captures: dict[str, torch.Tensor],
    weight_source: torch.Tensor,
) -> dict[str, Any]:
    cache.configure(candidate)
    label = "baseline-w4a8" if candidate is None else candidate["candidate_id"]
    record, _ = diagnosis.prior.run_cut_record(
        LAYER,
        f"layer23-v-joint-scale-{label}",
        hidden_states,
        weights,
        adapter,
        norm_gain,
        embedding,
        head,
        tokenizer,
        bf16_logits,
    )
    metrics = precision.summary_metrics(record)
    evidence = cache.evidence(
        label,
        captures["input_norm"],
        captures["v"],
        weight_source,
        candidate,
    )
    print(
        "ACE2_LAYER23_V_JOINT_SCALE "
        f"candidate={label} top={metrics['top_token_id']} "
        f"rank={metrics['target_token_rank']} "
        f"overlap={metrics['top8_overlap_count']} "
        f"jsd={metrics['jensen_shannon_divergence_nats']:.12f} "
        f"rel_l2={metrics['relative_l2']:.12f} "
        f"heldout_v_rel_l2={evidence['held_out']['v_projection_output']['error']['relative_l2']:.12f}",
        flush=True,
    )
    value = {
        "candidate_id": label,
        "metrics": metrics,
        "logits": record["accepted_w4a8_logits"],
        "scale_and_error_evidence": evidence,
    }
    if candidate is not None:
        value.update(
            {
                "configuration": candidate["configuration"],
                "calibration_search_summary": candidate["calibration"],
            }
        )
    return value


def decision_checks(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    baseline_metrics = baseline["metrics"]
    metrics = candidate["metrics"]
    rank_limit = max(TOP_K, int(baseline_metrics["target_token_rank"]) // 10)
    exact_recovery = (
        metrics["target_token_rank"] == 1
        and metrics["target_token_top1"]
        and metrics["top8_overlap_count"] >= baseline_metrics["top8_overlap_count"]
        and metrics["jensen_shannon_divergence_nats"]
        <= baseline_metrics["jensen_shannon_divergence_nats"]
    )
    distribution_recovery = (
        metrics["target_token_rank"] <= rank_limit
        and metrics["top8_overlap_count"] > baseline_metrics["top8_overlap_count"]
        and metrics["jensen_shannon_divergence_nats"]
        <= baseline_metrics["jensen_shannon_divergence_nats"] - 0.05
    )
    baseline_held_out = baseline["scale_and_error_evidence"]["held_out"][
        "v_projection_output"
    ]["error"]["relative_l2"]
    candidate_held_out = candidate["scale_and_error_evidence"]["held_out"][
        "v_projection_output"
    ]["error"]["relative_l2"]
    return {
        "rank_limit": rank_limit,
        "exact_rank1_without_top8_or_jsd_regression": exact_recovery,
        "material_rank_top8_jsd_recovery": distribution_recovery,
        "materially_improves_frozen_gate": exact_recovery
        or distribution_recovery,
        "held_out_projection_relative_l2": candidate_held_out,
        "baseline_held_out_projection_relative_l2": baseline_held_out,
        "held_out_no_regression": candidate_held_out <= baseline_held_out,
        "selectable": (exact_recovery or distribution_recovery)
        and candidate_held_out <= baseline_held_out
        and candidate["scale_and_error_evidence"]["hardware_representable"],
    }


def validate_freeze(freeze: dict[str, Any]) -> None:
    require(freeze["mission_id"] == MISSION_ID, "freeze mission changed")
    require(
        freeze["classification"]
        == "bounded_nonofficial_software_only_joint_scale_search",
        "freeze classification changed",
    )
    require(
        freeze["partition"]["calibration_positions"]
        == list(CALIBRATION_POSITIONS)
        and freeze["partition"]["held_out_positions"]
        == list(HELD_OUT_POSITIONS),
        "calibration/held-out partition changed",
    )
    require(
        [item["candidate_id"] for item in freeze["search_policy"]["input_factor_order"]]
        == [
            "joint-i1of1",
            "joint-i15of16",
            "joint-i17of16",
            "joint-i7of8",
            "joint-i9of8",
        ],
        "input candidate order changed",
    )
    require(
        [
            item["factor_id"]
            for item in freeze["search_policy"]["weight_factor_order"]
        ]
        == ["w1of1", "w15of16", "w17of16", "w7of8", "w9of8"],
        "weight factor order changed",
    )
    require(
        freeze["scope_guards"]
        == {
            "attempt_0003": False,
            "focused_or_full_rtl": False,
            "official_preflight_or_run": False,
            "precision_or_layout_change": False,
            "ppa_wiki_or_attestation": False,
            "proprietary_tools": False,
            "replay_attempt_0002": False,
            "stage2": False,
        },
        "scope guards changed",
    )


def protected_records() -> dict[str, Any]:
    localizer = diagnosis.prior.localizer
    return {
        "accepted_diagnosis": localizer.file_record(ACCEPTED_DIAGNOSIS),
        "accepted_diagnosis_sums": localizer.file_record(
            ACCEPTED_DIAGNOSIS_SUMS
        ),
        "accepted_precision_matrix": localizer.file_record(PRECISION_MATRIX),
        "accepted_precision_matrix_sums": localizer.file_record(
            PRECISION_MATRIX_SUMS
        ),
        "bf16_logits": localizer.file_record(BF16_LOGITS),
    }


def generate_sums() -> None:
    localizer = diagnosis.prior.localizer
    paths = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path != SUMS
    )
    lines = [
        f"{localizer.sha256_file(path)}  {path.relative_to(OUTPUT).as_posix()}"
        for path in paths
    ]
    SUMS.write_text("\n".join(lines) + "\n", encoding="ascii")


def generate() -> None:
    require(FREEZE.is_file(), "candidate freeze is missing")
    require(not RESULT.exists(), "joint scale result already exists")
    require(not SUMS.exists(), "joint scale search was already finalized")
    freeze = read_json(FREEZE)
    validate_freeze(freeze)
    localizer = diagnosis.prior.localizer
    require(
        localizer.sha256_file(Path(__file__).resolve())
        == freeze["runner"]["sha256"],
        "preregistered runner hash changed",
    )
    verify_sum_tree(
        ACCEPTED_DIAGNOSIS_SUMS,
        freeze["bindings"]["accepted_diagnosis_sums_sha256"],
    )
    verify_sum_tree(
        PRECISION_MATRIX_SUMS,
        freeze["bindings"]["accepted_precision_matrix_sums_sha256"],
    )
    require(
        localizer.sha256_file(ACCEPTED_DIAGNOSIS)
        == freeze["bindings"]["accepted_diagnosis_result_sha256"],
        "accepted joint-error diagnosis changed",
    )
    require(
        localizer.sha256_file(PRECISION_MATRIX)
        == freeze["bindings"]["accepted_precision_matrix_result_sha256"],
        "accepted negative precision matrix changed",
    )
    require(
        localizer.sha256_file(BF16_LOGITS)
        == freeze["bindings"]["bf16_logits_sha256"],
        "BF16 logit oracle changed",
    )
    protected_before = protected_records()
    started = time.monotonic()
    snapshot = diagnosis.prior.generation_runner.resolve_snapshot()
    diagnosis.prior.backend.configure_snapshot(snapshot)
    canonical = diagnosis.prior.backend.canonical
    require(
        localizer.sha256_file(canonical.MODEL) == canonical.MODEL_SHA256,
        "base model hash changed",
    )
    require(
        localizer.sha256_file(canonical.ADAPTER) == canonical.ADAPTER_SHA256,
        "checkpoint-176 adapter hash changed",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    token_ids = diagnosis.prior.generation_runner.canonical_chat_token_ids(
        tokenizer, diagnosis.prior.PROMPT
    )
    require(len(token_ids) == 34, "frozen prompt token count changed")
    require(
        localizer.sha256_bytes(canonical_bytes(token_ids))
        == freeze["prompt"]["token_ids_sha256"],
        "frozen token IDs changed",
    )
    require(
        localizer.sha256_bytes(diagnosis.prior.PROMPT.encode("utf-8"))
        == freeze["prompt"]["sha256"],
        "frozen prompt changed",
    )
    diagnosis.prior.localizer.PROMPT = diagnosis.prior.PROMPT
    diagnosis.prior.localizer.TOKEN_IDS = token_ids
    diagnosis.prior.localizer.REFERENCE_TOKEN = TARGET_TOKEN

    print("ACE2_LAYER23_V_JOINT_SCALE_BF16_CAPTURE", flush=True)
    layer_input, captures, bf16_logits = diagnosis.capture_layer_context(
        snapshot, token_ids
    )
    capture_bindings = {
        "layer_input": {
            "shape": list(layer_input.shape),
            "dtype": str(layer_input.dtype),
            "sha256": localizer.tensor_sha256(layer_input),
        },
        **{
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": localizer.tensor_sha256(value),
            }
            for name, value in sorted(captures.items())
        },
        "calibration_input_norm": {
            "positions": list(CALIBRATION_POSITIONS),
            "sha256": localizer.tensor_sha256(
                captures["input_norm"][list(CALIBRATION_POSITIONS)].contiguous()
            ),
        },
        "calibration_v_oracle": {
            "positions": list(CALIBRATION_POSITIONS),
            "sha256": localizer.tensor_sha256(
                captures["v"][list(CALIBRATION_POSITIONS)].contiguous()
            ),
        },
        "held_out_input_norm": {
            "positions": list(HELD_OUT_POSITIONS),
            "sha256": localizer.tensor_sha256(
                captures["input_norm"][list(HELD_OUT_POSITIONS)].contiguous()
            ),
        },
        "held_out_v_oracle": {
            "positions": list(HELD_OUT_POSITIONS),
            "sha256": localizer.tensor_sha256(
                captures["v"][list(HELD_OUT_POSITIONS)].contiguous()
            ),
        },
    }
    require(
        capture_bindings["input_norm"]["sha256"]
        == freeze["inputs"]["input_norm_sha256"]
        and capture_bindings["v"]["sha256"]
        == freeze["inputs"]["v_oracle_sha256"],
        "accepted local calibration tensors changed",
    )
    hidden_states = [torch.empty(0)] * LAYER + [layer_input]
    records: list[dict[str, Any]] = []
    suppressed_messages: list[str] = []
    prior_output = diagnosis.prior.OUTPUT
    diagnosis.prior.OUTPUT = OUTPUT
    try:
        with safe_open(
            canonical.MODEL, framework="pt", device="cpu"
        ) as weights, safe_open(
            canonical.ADAPTER, framework="pt", device="cpu"
        ) as adapter:
            embedding = weights.get_tensor(
                "model.embed_tokens.weight"
            ).contiguous()
            norm_gain = weights.get_tensor("model.norm.weight").contiguous()
            head = localizer.derive_lm_head_weights(embedding)
            cache = JointScaleCache(weights, adapter, captures)
            v_name = f"model.layers.{LAYER}.self_attn.v_proj"
            weight_source = cache.merged[v_name].to(torch.float64)
            candidates = build_candidates(
                cache,
                captures["input_norm"][
                    list(CALIBRATION_POSITIONS)
                ].contiguous(),
                captures["v"][list(CALIBRATION_POSITIONS)].contiguous(),
                freeze,
            )
            guard = diagnosis.prior.localizer.SubstitutionRequireGuard()
            cache.install()
            guard.install()
            try:
                records.append(
                    run_record(
                        cache,
                        None,
                        hidden_states,
                        weights,
                        adapter,
                        norm_gain,
                        embedding,
                        head,
                        tokenizer,
                        bf16_logits,
                        captures,
                        weight_source,
                    )
                )
                for candidate in candidates:
                    records.append(
                        run_record(
                            cache,
                            candidate,
                            hidden_states,
                            weights,
                            adapter,
                            norm_gain,
                            embedding,
                            head,
                            tokenizer,
                            bf16_logits,
                            captures,
                            weight_source,
                        )
                    )
            finally:
                suppressed_messages.extend(guard.suppressed)
                guard.restore()
                cache.restore()
    finally:
        diagnosis.prior.OUTPUT = prior_output

    baseline = records[0]
    metrics = baseline["metrics"]
    require(
        baseline["logits"]["sha256"] == EXPECTED_BASELINE["logits_sha256"]
        and metrics["target_token_rank"] == EXPECTED_BASELINE["target_rank"]
        and metrics["top8_overlap_count"]
        == EXPECTED_BASELINE["top8_overlap_count"]
        and math.isclose(
            metrics["jensen_shannon_divergence_nats"],
            EXPECTED_BASELINE["jensen_shannon_divergence_nats"],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            metrics["relative_l2"],
            EXPECTED_BASELINE["relative_l2"],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and metrics["top_token_id"] == EXPECTED_BASELINE["top_token_id"],
        "Fresh-L2-accepted W4A8 baseline did not reproduce",
    )
    for candidate in records[1:]:
        candidate["decision"] = decision_checks(baseline, candidate)
    passing = [
        candidate
        for candidate in records[1:]
        if candidate["decision"]["selectable"]
    ]
    selected = passing[0] if passing else None
    protected_after = protected_records()
    require(
        protected_before == protected_after,
        "accepted diagnosis or precision evidence changed",
    )
    shutil.copyfile(Path(__file__).resolve(), EXECUTION_RUNNER)
    result = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "status": (
            "JOINT_SCALE_CANDIDATE_SELECTED_PENDING_FRESH_L2"
            if selected is not None
            else "JOINT_SCALE_SEARCH_NEGATIVE_PENDING_FRESH_L2"
        ),
        "classification": "bounded_nonofficial_software_only_joint_scale_search",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "execution": {
            "candidate_count": len(records) - 1,
            "candidate_order": [
                record["candidate_id"] for record in records[1:]
            ],
            "prompt_token_count": len(token_ids),
            "generation_position": 0,
            "layer_cut": LAYER,
            "elapsed_wall_seconds": time.monotonic() - started,
        },
        "bindings": {
            "freeze": localizer.file_record(FREEZE),
            "runner": localizer.file_record(Path(__file__).resolve()),
            "archived_execution_runner": localizer.file_record(EXECUTION_RUNNER),
            "protected_before": protected_before,
            "protected_after": protected_after,
            "base_model": {
                "bytes": canonical.MODEL.stat().st_size,
                "sha256": canonical.MODEL_SHA256,
            },
            "checkpoint_176_adapter": {
                "bytes": canonical.ADAPTER.stat().st_size,
                "sha256": canonical.ADAPTER_SHA256,
            },
            "captured_tensors": capture_bindings,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "tokenizers": tokenizers.__version__,
            "numpy": np.__version__,
        },
        "partition": copy.deepcopy(freeze["partition"]),
        "baseline": baseline,
        "candidates": records[1:],
        "selection": {
            "policy": freeze["selection_policy"],
            "candidate_order": [
                record["candidate_id"] for record in records[1:]
            ],
            "selected_candidate_id": (
                selected["candidate_id"] if selected else None
            ),
            "selected_configuration_sha256": (
                selected["configuration"]["sha256"] if selected else None
            ),
            "negative_result": selected is None,
            "promotion_eligible": False,
            "promotion_gate": (
                "Any selected scale set requires a separate Fresh-L2-reviewed "
                "focused RTL/reference task."
            ),
        },
        "failure_taxonomy": (
            None
            if selected is not None
            else {
                "class": "bounded_joint_scale_search_non_improvement",
                "root_cause_hypothesis": (
                    "Small hardware-representable perturbations of the existing "
                    "per-row W4 scales and tensor-wide A8 input scale do not "
                    "recover the layer-23 V information lost by the joint "
                    "quantizers."
                ),
                "regression": (
                    "Retain the frozen 34-token gate and untouched positions "
                    "17-33 as rejection guards for later structurally different "
                    "V quantization repairs."
                ),
            }
        ),
        "suppressed_nonfunctional_cache_effect_assertions": {
            "count": len(suppressed_messages),
            "unique_messages": sorted(set(suppressed_messages)),
        },
        "independent_review_required": True,
        "scope_guards": copy.deepcopy(freeze["scope_guards"]),
    }
    write_json(RESULT, result)
    review = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "review_type": "Fresh-L2",
        "status": "PENDING_INDEPENDENT_REVIEW",
        "result": localizer.file_record(RESULT),
        "candidate_freeze": localizer.file_record(FREEZE),
        "execution_runner": localizer.file_record(EXECUTION_RUNNER),
        "reproduction_command": (
            ".venv/bin/python tools/run_stage1_layer23_v_joint_scale_search.py "
            "--verify-only"
        ),
        "required_checks": [
            "Reconstruct the five calibration-only row-scale selections from positions 0-16 and exact Scale32 records.",
            "Confirm positions 17-33 are absent from scale selection and independently recompute their projection errors.",
            "Recompute all six frozen-gate logit metrics from retained logits and BF16 token 39814.",
            "Confirm every multiplier, right shift, W4 payload, and Scale32 record preserves the existing layout and range.",
            "Confirm the frozen material-improvement plus held-out no-regression rule yields the recorded disposition.",
            "Confirm no RTL, official attempt, attempt-0003, PPA, wiki, attestation, or Stage-2 artifact was produced.",
        ],
    }
    write_json(REVIEW_REQUEST, review)
    generate_sums()
    print(
        "ACE2_LAYER23_V_JOINT_SCALE_RESULT "
        f"selected={result['selection']['selected_candidate_id']} "
        f"negative={result['selection']['negative_result']} "
        f"output={localizer.public_path(OUTPUT)}",
        flush=True,
    )


def verify() -> None:
    require(RESULT.is_file(), "joint scale result is missing")
    require(REVIEW_REQUEST.is_file(), "Fresh-L2 review request is missing")
    require(EXECUTION_RUNNER.is_file(), "archived execution runner is missing")
    require(SUMS.is_file(), "joint scale SHA256SUMS is missing")
    localizer = diagnosis.prior.localizer
    for line in SUMS.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = OUTPUT / relative
        require(path.is_file(), f"summed artifact missing: {relative}")
        require(
            localizer.sha256_file(path) == expected,
            f"summed artifact changed: {relative}",
        )
    freeze = read_json(FREEZE)
    validate_freeze(freeze)
    require(
        localizer.sha256_file(EXECUTION_RUNNER) == freeze["runner"]["sha256"],
        "archived execution runner differs from preregistration",
    )
    verify_sum_tree(
        ACCEPTED_DIAGNOSIS_SUMS,
        freeze["bindings"]["accepted_diagnosis_sums_sha256"],
    )
    verify_sum_tree(
        PRECISION_MATRIX_SUMS,
        freeze["bindings"]["accepted_precision_matrix_sums_sha256"],
    )
    result = read_json(RESULT)
    require(result["mission_id"] == MISSION_ID, "result mission changed")
    require(
        result["bindings"]["freeze"] == localizer.file_record(FREEZE)
        and result["bindings"]["archived_execution_runner"]
        == localizer.file_record(EXECUTION_RUNNER)
        and result["bindings"]["protected_before"]
        == result["bindings"]["protected_after"]
        == protected_records(),
        "result immutable bindings changed",
    )
    require(
        result["bindings"]["base_model"]["sha256"]
        == diagnosis.prior.backend.canonical.MODEL_SHA256
        and result["bindings"]["checkpoint_176_adapter"]["sha256"]
        == diagnosis.prior.backend.canonical.ADAPTER_SHA256,
        "model or adapter binding changed",
    )
    reference = np.fromfile(BF16_LOGITS, dtype="<f4")
    require(
        reference.shape == (diagnosis.prior.backend.MODEL_OUTPUT_DOMAIN,),
        "BF16 logits shape changed",
    )
    records = [result["baseline"], *result["candidates"]]
    require(
        [record["candidate_id"] for record in records]
        == [
            "baseline-w4a8",
            "joint-i1of1",
            "joint-i15of16",
            "joint-i17of16",
            "joint-i7of8",
            "joint-i9of8",
        ],
        "candidate record order changed",
    )
    oracle = np.fromfile(
        PRECISION_MATRIX.parent / "oracle/layer23-v-bf16-projection-f32le.bin",
        dtype="<f4",
    ).reshape(34, 128)
    for record in records:
        logits_record = record["logits"]
        logits_path = ROOT / logits_record["path"]
        require(
            localizer.sha256_file(logits_path) == logits_record["sha256"],
            f"{record['candidate_id']} logits changed",
        )
        scores = np.fromfile(logits_path, dtype="<f8")
        comparison = diagnosis.prior.quality.score_comparison(reference, scores)
        require(
            comparison
            == record["metrics"]["comparison_vs_checkpoint176_bf16"],
            f"{record['candidate_id']} logit metrics changed",
        )
        evidence = record["scale_and_error_evidence"]
        require(
            evidence["hardware_representable"]
            and len(evidence["weight_scale32"]["records_u32"]) == 128
            and len(evidence["multipliers_s32"]) == 128
            and len(evidence["right_shifts_u6"]) == 128,
            f"{record['candidate_id']} hardware metadata is malformed",
        )
        for scale_record in evidence["weight_scale32"]["records_u32"]:
            scale32.unpack_scale32(int(scale_record))
        scale32.unpack_scale32(
            int(evidence["input_scale32"]["record_u32"])
        )
        scale32.unpack_scale32(
            int(evidence["output_scale32"]["record_u32"])
        )
        output_record = evidence["artifacts"]["output_q"]
        output_path = ROOT / output_record["path"]
        require(
            localizer.sha256_file(output_path) == output_record["sha256"],
            f"{record['candidate_id']} output tensor changed",
        )
        output_q = np.fromfile(output_path, dtype=np.int8).reshape(34, 128)
        output_scale = float(evidence["output_scale32"]["source_value"])
        for subset in ("calibration", "held_out", "full_34_positions"):
            positions = evidence[subset]["positions"]
            candidate_v = (
                torch.from_numpy(output_q[positions].copy()).to(torch.float64)
                * output_scale
            )
            reference_v = torch.from_numpy(
                oracle[positions].copy()
            ).to(torch.float64)
            recomputed = error_stats(candidate_v, reference_v)
            require(
                recomputed
                == evidence[subset]["v_projection_output"]["error"],
                f"{record['candidate_id']} {subset} projection metrics changed",
            )
    baseline = result["baseline"]
    for candidate in result["candidates"]:
        require(
            candidate["decision"] == decision_checks(baseline, candidate),
            f"{candidate['candidate_id']} decision changed",
        )
    passing = [
        candidate
        for candidate in result["candidates"]
        if candidate["decision"]["selectable"]
    ]
    selected = passing[0] if passing else None
    require(
        result["selection"]["selected_candidate_id"]
        == (selected["candidate_id"] if selected else None)
        and result["selection"]["negative_result"] == (selected is None)
        and result["selection"]["promotion_eligible"] is False,
        "joint scale selection changed",
    )
    request = read_json(REVIEW_REQUEST)
    require(
        request["status"] == "PENDING_INDEPENDENT_REVIEW"
        and request["result"] == localizer.file_record(RESULT)
        and request["candidate_freeze"] == localizer.file_record(FREEZE)
        and request["execution_runner"] == localizer.file_record(EXECUTION_RUNNER),
        "Fresh-L2 review request binding changed",
    )
    require(
        not (
            ROOT
            / "evidence/verification/rtl-arbitrary-text-generation-v1/"
            "attempt-0003"
        ).exists(),
        "attempt-0003 exists",
    )
    print(
        "PASS stage1vjointscale01 "
        f"selected={result['selection']['selected_candidate_id']} "
        f"negative={result['selection']['negative_result']} "
        f"candidates={len(result['candidates'])}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
    else:
        generate()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"ACE2_LAYER23_V_JOINT_SCALE_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
