#!/usr/bin/env python3
"""Run the bounded nonofficial rank-1 integer V-residual correction candidate."""

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
from tools import diagnose_stage1_layer23_v_residual_structure as residual
from tools import run_stage1_layer23_v_precision_matrix as precision


MISSION_ID = "stage1vrank1int01"
LAYER = 23
TARGET_TOKEN = 39814
TOP_K = 8
CALIBRATION_POSITIONS = tuple(range(17))
HELD_OUT_POSITIONS = tuple(range(17, 34))
OUTPUT = (
    ROOT
    / "evidence/candidates/stage1-layer23-v-rank1-integer-correction-v1/"
    "nonofficial-candidate-0001"
)
FREEZE = OUTPUT / "candidate-freeze.json"
RESULT = OUTPUT / "result.json"
REVIEW_REQUEST = OUTPUT / "fresh-l2-review-request.json"
SUMS = OUTPUT / "SHA256SUMS"
EXECUTION_RUNNER = OUTPUT / "rank1-integer-candidate-execution-runner.py"
FLOAT_DIAGNOSIS_RESULT = (
    ROOT
    / "evidence/diagnostics/stage1-layer23-v-residual-structure-v1/"
    "nonofficial-diagnosis-0001/result.json"
)
FLOAT_DIAGNOSIS_SUMS = FLOAT_DIAGNOSIS_RESULT.parent / "SHA256SUMS"
FLOAT_DIAGNOSIS_RESULT_SHA256 = (
    "11dbd7dda4715827e3a16e4e530bdb6870fa51370fff86084a05c6d07ea3bc2d"
)
FLOAT_DIAGNOSIS_SUMS_SHA256 = (
    "b4c6c171d67c2692c0ba1558eb9c3fd967467d5ab38853fc58cf60d48746ba3e"
)
ACCEPTED_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/6a6e4fc4309f/round-0001.json"
)
ACCEPTED_REVIEW_SHA256 = (
    "5e1b1ddcc7e600d5dbb0818feeb670c8edb5c29fe6654f30b0481d7b89f985a4"
)
ACCEPTED_REVIEW_CHECKPOINT = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/6a6e4fc4309f/CHECKPOINT.md"
)
ACCEPTED_REVIEW_CHECKPOINT_SHA256 = (
    "c1d89c71b6cb994b6a04f19acbb1c4b1743ef8db66b1979857a41d670c7ecda9"
)
BF16_LOGITS = (
    ROOT
    / "diagnosis/stage1qualitydiag01/logits/"
    "checkpoint176-bf16-greedy-step-00-f32le.bin"
)
FLOAT_RANK1_LOGITS = (
    FLOAT_DIAGNOSIS_RESULT.parent
    / "logits/layer23-v-residual-rank-01-f64le.bin"
)
RETAINED_OUTPUT_SCALE = 0.06586176203930472
EXPECTED_BASELINE = {
    "logits_sha256": "0b80dbb919618cb7f39a8a862cac652ba684176a520865020595bd048085ea17",
    "target_rank": 15,
    "top8_overlap_count": 0,
    "jensen_shannon_divergence_nats": 0.6701229933515445,
    "relative_l2": 0.9423817179633847,
    "top_token_id": 55725,
    "held_out_projection_relative_l2": 1.01390282266012,
}
EXPECTED_FLOAT_RANK1 = {
    "logits_sha256": "09ea25901cc59b8d5a16f0b77d3ac9dd3adbd1f814e4721b984c831285257d78",
    "target_rank": 1,
    "top8_overlap_count": 8,
    "jensen_shannon_divergence_nats": 0.09849065207352257,
    "relative_l2": 0.3488828208172395,
    "held_out_projection_relative_l2": 0.3087007407356935,
}
SCOPE_GUARDS = {
    "attempt_0003": False,
    "candidate_changes_existing_w4a8_path": False,
    "focused_or_full_rtl": False,
    "icarus_or_other_rtl_simulation": False,
    "official_preflight_or_run": False,
    "ppa_or_stage2": False,
    "replay_attempt_0002": False,
    "specification_audit_refresh": False,
    "u280_or_xrt": False,
}
INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


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
    return residual.diagnosis.prior.localizer.file_record(path)


def write_tensor(path: Path, value: torch.Tensor, dtype: str) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(residual.diagnosis.prior.localizer.tensor_bytes(tensor))
    record = residual.diagnosis.prior.localizer.file_record(path)
    record.update({"dtype": dtype, "shape": list(tensor.shape)})
    return record


def load_tensor(record: dict[str, Any]) -> torch.Tensor:
    dtype_map = {
        "float64_le": np.dtype("<f8"),
        "float32_le": np.dtype("<f4"),
        "int8": np.dtype("i1"),
        "uint8": np.dtype("u1"),
        "int16_le": np.dtype("<i2"),
        "int32_le": np.dtype("<i4"),
        "uint32_le": np.dtype("<u4"),
        "int64_le": np.dtype("<i8"),
    }
    dtype = dtype_map[record["dtype"]]
    values = np.fromfile(ROOT / record["path"], dtype=dtype).copy()
    return torch.from_numpy(values).reshape(record["shape"])


def verify_sum_tree(path: Path, expected_hash: str) -> None:
    localizer = residual.diagnosis.prior.localizer
    require(localizer.sha256_file(path) == expected_hash, f"{path.name} hash changed")
    for line in path.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        artifact = path.parent / relative
        require(artifact.is_file(), f"protected artifact missing: {relative}")
        require(
            localizer.sha256_file(artifact) == expected,
            f"protected artifact changed: {relative}",
        )


def external_file_record(path: Path) -> dict[str, Any]:
    localizer = residual.diagnosis.prior.localizer
    require(path.is_file(), f"required external binding is missing: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": localizer.sha256_file(path),
    }


def scale32_record(value: float) -> int:
    require(math.isfinite(value) and value > 0.0, "invalid positive scale")
    return scale32.ceil_scale32_from_float(float(value))


def scale32_value(record: int) -> float:
    numerator, denominator = scale32.scale32_ratio(int(record))
    return numerator / denominator


def scale32_detail(record: int, source_value: float) -> dict[str, Any]:
    significand, exponent = scale32.unpack_scale32(int(record))
    return {
        "record_u32": int(record),
        "record_hex": f"0x{int(record):08x}",
        "significand_u16": significand,
        "exponent_s8": exponent,
        "decoded_value": scale32_value(record),
        "source_value": float(source_value),
    }


def scale32_tensor_detail(
    records: torch.Tensor, source_values: torch.Tensor
) -> dict[str, Any]:
    flat_records = [int(value) for value in records.reshape(-1).tolist()]
    flat_source = [float(value) for value in source_values.reshape(-1).tolist()]
    return {
        "count": len(flat_records),
        "records_u32": flat_records,
        "records_hex": [f"0x{value:08x}" for value in flat_records],
        "decoded_values": [scale32_value(value) for value in flat_records],
        "source_values": flat_source,
    }


def encode_scale32_tensor(values: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [
            scale32_record(float(value))
            for value in values.detach().cpu().reshape(-1).tolist()
        ],
        dtype=torch.int64,
    ).reshape(values.shape)


def decode_scale32_tensor(records: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [scale32_value(int(value)) for value in records.reshape(-1).tolist()],
        dtype=torch.float64,
    ).reshape(records.shape)


def quantize_s8(
    source: torch.Tensor, scale: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_tensor = torch.as_tensor(scale, dtype=torch.float64, device=source.device)
    rounded = torch.round(source.to(torch.float64) / scale_tensor)
    return rounded.clamp(INT8_MIN, INT8_MAX).to(torch.int8), rounded


def round_shift_even(value: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return residual.diagnosis.prior.localizer.round_shift_even_tensor(
        value.to(torch.int64), shift.to(torch.int64)
    )


def derive_multiplier(real: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.as_tensor(real, dtype=torch.float64)
    return residual.diagnosis.prior.backend.canonical.derive_multiplier(tensor)


def saturation_stats(rounded: torch.Tensor, lower: int, upper: int) -> dict[str, Any]:
    saturated = (rounded < lower) | (rounded > upper)
    return {
        "count": int(saturated.sum().item()),
        "total": int(saturated.numel()),
        "rate": float(saturated.sum().item() / saturated.numel()),
        "below_count": int((rounded < lower).sum().item()),
        "above_count": int((rounded > upper).sum().item()),
        "observed_min": int(rounded.min().item()),
        "observed_max": int(rounded.max().item()),
    }


def tensor_range(tensor: torch.Tensor) -> dict[str, int]:
    value = tensor.to(torch.int64)
    return {
        "min": int(value.min().item()),
        "max": int(value.max().item()),
    }


def error_summary(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate_f64 = candidate.to(torch.float64)
    reference_f64 = reference.to(torch.float64)
    difference = candidate_f64 - reference_f64
    denominator = torch.linalg.vector_norm(reference_f64)
    require(float(denominator.item()) > 0.0, "relative-L2 denominator is zero")
    return {
        "count": int(difference.numel()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference).item() / denominator.item()
        ),
        "rmse": float(torch.sqrt(torch.mean(difference * difference)).item()),
        "mean_absolute_error": float(difference.abs().mean().item()),
        "maximum_absolute_error": float(difference.abs().max().item()),
    }


def split_projection_summary(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, Any]:
    calibration = torch.tensor(CALIBRATION_POSITIONS, dtype=torch.int64)
    held_out = torch.tensor(HELD_OUT_POSITIONS, dtype=torch.int64)
    return {
        "calibration": error_summary(
            candidate.index_select(0, calibration),
            reference.index_select(0, calibration),
        ),
        "held_out": error_summary(
            candidate.index_select(0, held_out),
            reference.index_select(0, held_out),
        ),
        "full_34_positions": error_summary(candidate, reference),
    }


def generate_sums() -> None:
    localizer = residual.diagnosis.prior.localizer
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


def protected_records() -> dict[str, Any]:
    localizer = residual.diagnosis.prior.localizer
    return {
        "float_residual_diagnosis": localizer.file_record(FLOAT_DIAGNOSIS_RESULT),
        "float_residual_diagnosis_sums": localizer.file_record(FLOAT_DIAGNOSIS_SUMS),
        "accepted_fresh_l2_review": external_file_record(ACCEPTED_REVIEW),
        "accepted_fresh_l2_checkpoint": external_file_record(
            ACCEPTED_REVIEW_CHECKPOINT
        ),
        "bf16_logits": localizer.file_record(BF16_LOGITS),
        "float_rank1_logits": localizer.file_record(FLOAT_RANK1_LOGITS),
    }


def validate_preconditions() -> dict[str, Any]:
    localizer = residual.diagnosis.prior.localizer
    verify_sum_tree(FLOAT_DIAGNOSIS_SUMS, FLOAT_DIAGNOSIS_SUMS_SHA256)
    require(
        localizer.sha256_file(FLOAT_DIAGNOSIS_RESULT)
        == FLOAT_DIAGNOSIS_RESULT_SHA256,
        "accepted float residual diagnosis result changed",
    )
    review = external_file_record(ACCEPTED_REVIEW)
    checkpoint = external_file_record(ACCEPTED_REVIEW_CHECKPOINT)
    require(
        review["sha256"] == ACCEPTED_REVIEW_SHA256,
        "accepted Fresh-L2 handoff hash changed",
    )
    require(
        checkpoint["sha256"] == ACCEPTED_REVIEW_CHECKPOINT_SHA256,
        "accepted Fresh-L2 checkpoint hash changed",
    )
    review_json = read_json(ACCEPTED_REVIEW)
    require(
        review_json["producer_role"] == "reviewer"
        and review_json["review"]["status"] == "done"
        and "rank 1 explains 96.27%" in review_json["review"]["reason"],
        "Fresh-L2 rank-1 acceptance is not present",
    )
    diagnosis = read_json(FLOAT_DIAGNOSIS_RESULT)
    rank1 = next(
        item for item in diagnosis["candidates"] if int(item["rank"]) == 1
    )
    require(
        rank1["decision"]["diagnostic_pass"]
        and rank1["decision"]["held_out_no_regression"]
        and rank1["metrics"]["target_token_rank"] == EXPECTED_FLOAT_RANK1["target_rank"]
        and rank1["logits"]["sha256"] == EXPECTED_FLOAT_RANK1["logits_sha256"],
        "accepted rank-1 float oracle is not the supported diagnostic winner",
    )
    return diagnosis


def freeze_factor_quantization() -> None:
    require(not RESULT.exists(), "integer candidate result already exists")
    require(not SUMS.exists(), "integer candidate SHA256SUMS already exists")
    if FREEZE.exists():
        validate_freeze(read_json(FREEZE))
        print(
            "PASS stage1vrank1int01 freeze exists "
            f"sha256={residual.diagnosis.prior.localizer.sha256_file(FREEZE)}",
            flush=True,
        )
        return
    diagnosis = validate_preconditions()
    localizer = residual.diagnosis.prior.localizer
    snapshot = residual.diagnosis.prior.generation_runner.resolve_snapshot()
    residual.diagnosis.prior.backend.configure_snapshot(snapshot)
    rank1 = next(
        item for item in diagnosis["candidates"] if int(item["rank"]) == 1
    )
    factors = diagnosis["artifacts"]["rank_factors"]["1"]
    input_record = diagnosis["artifacts"]["tensors"]["input_dequantized"]
    input_dequantized = residual.load_tensor(input_record).to(torch.float64)
    left = residual.load_tensor(factors["left_factor"]).to(torch.float64)
    right = residual.load_tensor(factors["right_basis"]).to(torch.float64)
    require(left.shape == (896, 1), "rank-1 left factor shape changed")
    require(right.shape == (1, 128), "rank-1 right factor shape changed")
    calibration_input = input_dequantized[list(CALIBRATION_POSITIONS)]
    input_scale_source = float(calibration_input.abs().max().item() / 127.0)
    input_scale32 = scale32_record(input_scale_source)
    input_scale = scale32_value(input_scale32)
    q_input_calibration, input_rounded_calibration = quantize_s8(
        calibration_input, input_scale
    )
    left_scale_source = float(left.abs().max().item() / 127.0)
    left_scale32 = scale32_record(left_scale_source)
    left_scale = scale32_value(left_scale32)
    q_left, left_rounded = quantize_s8(left, left_scale)
    right_scale_source = right.abs().flatten() / 127.0
    require(
        bool(torch.all(right_scale_source > 0.0)),
        "rank-1 right basis unexpectedly contains exact zero channels",
    )
    right_scale32 = encode_scale32_tensor(right_scale_source)
    right_scale = decode_scale32_tensor(right_scale32)
    q_right, right_rounded = quantize_s8(right, right_scale.reshape(1, 128))
    first_acc_calibration = torch.matmul(
        q_input_calibration.to(torch.int64), q_left.to(torch.int64)
    ).flatten()
    require(
        bool(torch.all(first_acc_calibration >= INT32_MIN))
        and bool(torch.all(first_acc_calibration <= INT32_MAX)),
        "rank-1 calibration accumulator exceeds int32",
    )
    first_dequant_calibration = (
        first_acc_calibration.to(torch.float64) * input_scale * left_scale
    )
    rank_scale_source = float(first_dequant_calibration.abs().max().item() / 127.0)
    rank_scale32 = scale32_record(rank_scale_source)
    rank_scale = scale32_value(rank_scale32)
    first_multiplier, first_right_shift = derive_multiplier(
        torch.tensor(input_scale * left_scale / rank_scale, dtype=torch.float64)
    )
    second_multiplier, second_right_shift = derive_multiplier(
        rank_scale * right_scale.flatten() / RETAINED_OUTPUT_SCALE
    )
    freeze_dir = OUTPUT / "freeze"
    artifacts = {
        "input_to_rank_s8": write_tensor(
            freeze_dir / "input_to_rank_s8.bin", q_left, "int8"
        ),
        "rank_to_channel_s8": write_tensor(
            freeze_dir / "rank_to_channel_s8.bin", q_right, "int8"
        ),
        "right_scale32_u32le": write_tensor(
            freeze_dir / "rank_to_channel_scale32_u32le.bin",
            right_scale32.to(torch.int32),
            "uint32_le",
        ),
        "first_stage_multiplier_s32le": write_tensor(
            freeze_dir / "first_stage_multiplier_s32le.bin",
            first_multiplier.reshape(1).to(torch.int32),
            "int32_le",
        ),
        "first_stage_right_shift_u8": write_tensor(
            freeze_dir / "first_stage_right_shift_u8.bin",
            first_right_shift.reshape(1).to(torch.uint8),
            "uint8",
        ),
        "second_stage_multiplier_s32le": write_tensor(
            freeze_dir / "second_stage_multiplier_s32le.bin",
            second_multiplier.to(torch.int32),
            "int32_le",
        ),
        "second_stage_right_shift_u8": write_tensor(
            freeze_dir / "second_stage_right_shift_u8.bin",
            second_right_shift.to(torch.uint8),
            "uint8",
        ),
    }
    quantization = {
        "policy": (
            "Freeze all factor quantization and Scale32 scaling before downstream "
            "logit evaluation. Input scale and rank-intermediate scale use only "
            "positions 0..16; rank-1 float factors are the accepted calibration-only "
            "oracle factors from the Fresh-L2-reviewed diagnosis."
        ),
        "input_payload": {
            "dtype": "signed_int8",
            "scale32": scale32_detail(input_scale32, input_scale_source),
            "calibration_saturation": saturation_stats(
                input_rounded_calibration, INT8_MIN, INT8_MAX
            ),
            "held_out_positions_used_for_scale": False,
        },
        "input_to_rank_factor": {
            "dtype": "signed_int8",
            "shape": [896, 1],
            "scale32": scale32_detail(left_scale32, left_scale_source),
            "saturation": saturation_stats(left_rounded, INT8_MIN, INT8_MAX),
        },
        "rank_intermediate": {
            "dtype": "signed_int8",
            "scale32": scale32_detail(rank_scale32, rank_scale_source),
            "source": "maximum absolute first-stage dequantized coefficient on calibration positions 0..16",
            "held_out_positions_used_for_scale": False,
        },
        "rank_to_channel_factor": {
            "dtype": "signed_int8",
            "shape": [1, 128],
            "per_channel_scale32": scale32_tensor_detail(
                right_scale32, right_scale_source
            ),
            "saturation": saturation_stats(right_rounded, INT8_MIN, INT8_MAX),
        },
        "requantization": {
            "first_stage": {
                "real_multiplier": input_scale * left_scale / rank_scale,
                "multiplier_s32": int(first_multiplier.reshape(-1)[0].item()),
                "right_shift_u6": int(first_right_shift.reshape(-1)[0].item()),
            },
            "second_stage": {
                "real_multiplier": [
                    float(value)
                    for value in (
                        rank_scale * right_scale.flatten() / RETAINED_OUTPUT_SCALE
                    ).tolist()
                ],
                "multiplier_s32": [
                    int(value) for value in second_multiplier.reshape(-1).tolist()
                ],
                "right_shift_u6": [
                    int(value) for value in second_right_shift.reshape(-1).tolist()
                ],
            },
            "rounding": "signed round-to-nearest, ties-to-even, then signed-int8 saturation",
            "final_add": (
                "correction_q is in retained V output-scale quanta and is "
                "saturating-added to the unchanged baseline W4A8 V output_q"
            ),
        },
    }
    value = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "status": "FROZEN_PENDING_EVALUATION",
        "classification": (
            "bounded_nonofficial_software_only_rank1_integer_v_residual_candidate"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authority": {
            "basis": (
                "Supervisor conditional next step after Fresh-L2 acceptance of "
                "stage1vresidualstructure01 rank-1 residual diagnosis"
            ),
            "official_execution_authorized": False,
            "attempt_0003_authorized": False,
            "rtl_authorized": False,
            "ppa_stage2_u280_authorized": False,
        },
        "accepted_float_oracle": {
            "diagnosis_result": localizer.file_record(FLOAT_DIAGNOSIS_RESULT),
            "diagnosis_sums": localizer.file_record(FLOAT_DIAGNOSIS_SUMS),
            "fresh_l2_review": external_file_record(ACCEPTED_REVIEW),
            "fresh_l2_checkpoint": external_file_record(ACCEPTED_REVIEW_CHECKPOINT),
            "rank1_metrics": copy.deepcopy(rank1["metrics"]),
            "rank1_projection_output": copy.deepcopy(rank1["projection_output"]),
            "rank1_float_artifacts": copy.deepcopy(factors),
        },
        "partition": {
            "calibration_positions": list(CALIBRATION_POSITIONS),
            "held_out_positions": list(HELD_OUT_POSITIONS),
            "held_out_policy": (
                "No held-out position determines any factor, input scale, "
                "rank scale, or multiplier before evaluation."
            ),
        },
        "retained_v_output_scale": RETAINED_OUTPUT_SCALE,
        "quantization": quantization,
        "artifacts": artifacts,
        "runner": localizer.file_record(Path(__file__).resolve()),
        "decision_policy": {
            "logit_gate": (
                "rank 1 with no top-8/JSD regression, or rank <= 8 with "
                "strict top-8 improvement and JSD improvement >= 0.05 nats"
            ),
            "held_out_material_recovery_gate": (
                "integer corrected V held-out relative-L2 must be <= 0.75 times "
                "the accepted W4A8 baseline held-out relative-L2"
            ),
            "float_oracle_retention_reported_not_gating": True,
            "existing_w4a8_path_must_byte_match": True,
        },
        "scope_guards": copy.deepcopy(SCOPE_GUARDS),
    }
    write_json(FREEZE, value)
    print(
        "ACE2_LAYER23_V_RANK1_INTEGER_FREEZE "
        f"sha256={localizer.sha256_file(FREEZE)} output={localizer.public_path(OUTPUT)}",
        flush=True,
    )


def validate_freeze(freeze: dict[str, Any]) -> None:
    require(freeze["mission_id"] == MISSION_ID, "freeze mission changed")
    require(
        freeze["classification"]
        == "bounded_nonofficial_software_only_rank1_integer_v_residual_candidate",
        "freeze classification changed",
    )
    require(freeze["scope_guards"] == SCOPE_GUARDS, "freeze scope guards changed")
    require(
        freeze["partition"]["calibration_positions"] == list(CALIBRATION_POSITIONS)
        and freeze["partition"]["held_out_positions"] == list(HELD_OUT_POSITIONS),
        "freeze partition changed",
    )
    localizer = residual.diagnosis.prior.localizer
    require(
        localizer.sha256_file(Path(__file__).resolve())
        == freeze["runner"]["sha256"],
        "frozen runner hash changed",
    )
    validate_preconditions()
    require(
        freeze["accepted_float_oracle"]["diagnosis_result"]
        == localizer.file_record(FLOAT_DIAGNOSIS_RESULT),
        "float diagnosis result binding changed",
    )
    require(
        freeze["accepted_float_oracle"]["diagnosis_sums"]
        == localizer.file_record(FLOAT_DIAGNOSIS_SUMS),
        "float diagnosis sums binding changed",
    )
    for key in (
        "input_to_rank_s8",
        "rank_to_channel_s8",
        "right_scale32_u32le",
        "first_stage_multiplier_s32le",
        "first_stage_right_shift_u8",
        "second_stage_multiplier_s32le",
        "second_stage_right_shift_u8",
    ):
        record = freeze["artifacts"][key]
        path = ROOT / record["path"]
        require(path.is_file(), f"frozen artifact missing: {key}")
        require(
            localizer.sha256_file(path) == record["sha256"],
            f"frozen artifact hash changed: {key}",
        )


def integer_arithmetic(
    freeze: dict[str, Any],
    input_dequantized: torch.Tensor,
    baseline_output: torch.Tensor,
    *,
    write_artifacts: bool = True,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    input_scale = float(
        freeze["quantization"]["input_payload"]["scale32"]["decoded_value"]
    )
    rank_scale = float(
        freeze["quantization"]["rank_intermediate"]["scale32"]["decoded_value"]
    )
    q_left = load_tensor(freeze["artifacts"]["input_to_rank_s8"]).to(torch.int8)
    q_right = load_tensor(freeze["artifacts"]["rank_to_channel_s8"]).to(torch.int8)
    first_multiplier = load_tensor(
        freeze["artifacts"]["first_stage_multiplier_s32le"]
    ).to(torch.int64)
    first_shift = load_tensor(
        freeze["artifacts"]["first_stage_right_shift_u8"]
    ).to(torch.int64)
    second_multiplier = load_tensor(
        freeze["artifacts"]["second_stage_multiplier_s32le"]
    ).to(torch.int64)
    second_shift = load_tensor(
        freeze["artifacts"]["second_stage_right_shift_u8"]
    ).to(torch.int64)
    q_input, input_rounded = quantize_s8(input_dequantized, input_scale)
    first_acc = torch.matmul(q_input.to(torch.int64), q_left.to(torch.int64)).flatten()
    require(
        bool(torch.all(first_acc >= INT32_MIN))
        and bool(torch.all(first_acc <= INT32_MAX)),
        "first-stage accumulator escaped int32",
    )
    first_product = first_acc * int(first_multiplier.reshape(-1)[0].item())
    q_rank_rounded = round_shift_even(
        first_product, first_shift.reshape(1).expand_as(first_product)
    )
    q_rank = q_rank_rounded.clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    second_acc = q_rank.to(torch.int64)[:, None] * q_right.to(torch.int64)
    require(
        bool(torch.all(second_acc >= INT32_MIN))
        and bool(torch.all(second_acc <= INT32_MAX)),
        "second-stage accumulator escaped int32",
    )
    second_product = second_acc * second_multiplier.reshape(1, 128)
    q_correction_rounded = round_shift_even(
        second_product, second_shift.reshape(1, 128).expand_as(second_product)
    )
    q_correction = q_correction_rounded.clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    baseline_q_rounded = torch.round(
        baseline_output.to(torch.float64) / RETAINED_OUTPUT_SCALE
    )
    baseline_q = baseline_q_rounded.clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    require(
        torch.allclose(
            baseline_q.to(torch.float64) * RETAINED_OUTPUT_SCALE,
            baseline_output.to(torch.float64),
            rtol=0.0,
            atol=1.0e-12,
        ),
        "saved baseline V output is not retained A8 output-scale aligned",
    )
    output_sum = baseline_q.to(torch.int16) + q_correction.to(torch.int16)
    output_q = output_sum.clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    correction = q_correction.to(torch.float64) * RETAINED_OUTPUT_SCALE
    corrected = output_q.to(torch.float64) * RETAINED_OUTPUT_SCALE
    theoretical_first_abs = 896 * 128 * 128
    theoretical_second_abs = 128 * 128
    evidence = {
        "input_payload": {
            "scale": input_scale,
            "saturation": saturation_stats(input_rounded, INT8_MIN, INT8_MAX),
        },
        "rank_intermediate": {
            "scale": rank_scale,
            "accumulator_s32": tensor_range(first_acc),
            "accumulator_theoretical_abs_bound": theoretical_first_abs,
            "product_s64": tensor_range(first_product),
            "rounded_s32": tensor_range(q_rank_rounded),
            "saturation": saturation_stats(q_rank_rounded, INT8_MIN, INT8_MAX),
        },
        "correction_output": {
            "accumulator_s32": tensor_range(second_acc),
            "accumulator_theoretical_abs_bound": theoretical_second_abs,
            "product_s64": tensor_range(second_product),
            "rounded_s32": tensor_range(q_correction_rounded),
            "saturation": saturation_stats(
                q_correction_rounded, INT8_MIN, INT8_MAX
            ),
        },
        "final_saturating_add": {
            "pre_saturate_sum_s16": tensor_range(output_sum),
            "saturation": saturation_stats(output_sum, INT8_MIN, INT8_MAX),
        },
        "hardware_representable": (
            bool(torch.all(first_acc >= INT32_MIN))
            and bool(torch.all(first_acc <= INT32_MAX))
            and bool(torch.all(second_acc >= INT32_MIN))
            and bool(torch.all(second_acc <= INT32_MAX))
            and bool(torch.all(first_shift >= 0))
            and bool(torch.all(first_shift <= 63))
            and bool(torch.all(second_shift >= 0))
            and bool(torch.all(second_shift <= 63))
            and bool(torch.all(first_multiplier >= INT32_MIN))
            and bool(torch.all(first_multiplier <= INT32_MAX))
            and bool(torch.all(second_multiplier >= INT32_MIN))
            and bool(torch.all(second_multiplier <= INT32_MAX))
        ),
    }
    artifacts = {}
    if write_artifacts:
        payload_dir = OUTPUT / "payload"
        artifacts = {
            "input_payload_s8": write_tensor(
                payload_dir / "v_input_payload_s8.bin", q_input, "int8"
            ),
            "rank_intermediate_accumulator_s32le": write_tensor(
                payload_dir / "rank_intermediate_accumulator_s32le.bin",
                first_acc.to(torch.int32),
                "int32_le",
            ),
            "rank_intermediate_rounded_s32le": write_tensor(
                payload_dir / "rank_intermediate_rounded_s32le.bin",
                q_rank_rounded.to(torch.int32),
                "int32_le",
            ),
            "rank_intermediate_s8": write_tensor(
                payload_dir / "rank_intermediate_s8.bin", q_rank, "int8"
            ),
            "correction_accumulator_s32le": write_tensor(
                payload_dir / "correction_accumulator_s32le.bin",
                second_acc.to(torch.int32),
                "int32_le",
            ),
            "correction_rounded_s32le": write_tensor(
                payload_dir / "correction_rounded_s32le.bin",
                q_correction_rounded.to(torch.int32),
                "int32_le",
            ),
            "correction_s8": write_tensor(
                payload_dir / "correction_s8.bin", q_correction, "int8"
            ),
            "corrected_v_output_s8": write_tensor(
                payload_dir / "corrected_v_output_s8.bin", output_q, "int8"
            ),
            "predicted_residual_f64le": write_tensor(
                payload_dir / "predicted_residual_f64le.bin",
                correction,
                "float64_le",
            ),
            "corrected_v_output_f64le": write_tensor(
                payload_dir / "corrected_v_output_f64le.bin",
                corrected,
                "float64_le",
            ),
        }
    evidence["artifacts"] = artifacts
    return evidence, correction, corrected


def decision(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    baseline_projection: dict[str, Any],
    candidate_projection: dict[str, Any],
    hardware_representable: bool,
    baseline_byte_match: bool,
) -> dict[str, Any]:
    exact = (
        candidate_metrics["target_token_rank"] == 1
        and candidate_metrics["target_token_top1"]
        and candidate_metrics["top8_overlap_count"]
        >= baseline_metrics["top8_overlap_count"]
        and candidate_metrics["jensen_shannon_divergence_nats"]
        <= baseline_metrics["jensen_shannon_divergence_nats"]
    )
    material = (
        candidate_metrics["target_token_rank"] <= TOP_K
        and candidate_metrics["top8_overlap_count"] > baseline_metrics["top8_overlap_count"]
        and candidate_metrics["jensen_shannon_divergence_nats"]
        <= baseline_metrics["jensen_shannon_divergence_nats"] - 0.05
    )
    baseline_held_out = baseline_projection["held_out"]["relative_l2"]
    candidate_held_out = candidate_projection["held_out"]["relative_l2"]
    material_held_out = candidate_held_out <= 0.75 * baseline_held_out
    accepted = (
        (exact or material)
        and material_held_out
        and hardware_representable
        and baseline_byte_match
    )
    return {
        "exact_rank1_without_top8_or_jsd_regression": exact,
        "material_rank_top8_jsd_recovery": material,
        "baseline_held_out_projection_relative_l2": baseline_held_out,
        "held_out_projection_relative_l2": candidate_held_out,
        "held_out_material_recovery": material_held_out,
        "hardware_representable": hardware_representable,
        "existing_w4a8_path_byte_match": baseline_byte_match,
        "accepted_integer_candidate": accepted,
        "existing_w4a8_path_changed": False,
        "promotion_into_current_w4a8_path": False,
        "rtl_microarchitecture_task_justified": accepted,
    }


def estimates() -> dict[str, Any]:
    first_macs = 896
    second_macs = 128
    factor_bytes = first_macs + second_macs
    scale32_records = 1 + 1 + 1 + 128
    multiplier_shift_records = 1 + 128
    return {
        "estimate_only": True,
        "rank": 1,
        "input_width": 896,
        "output_width": 128,
        "factor_payload_bytes": factor_bytes,
        "scale32_records": scale32_records,
        "scale32_metadata_bytes": 4 * scale32_records,
        "requantization_multiplier_count": multiplier_shift_records,
        "requantization_metadata_bytes_estimate": 5 * multiplier_shift_records,
        "state_bytes_per_token": 1,
        "macs_per_token": first_macs + second_macs,
        "first_factor_macs": first_macs,
        "second_factor_macs": second_macs,
        "single_mac_cycle_estimate": first_macs + second_macs,
        "one_128_lane_second_stage_cycle_estimate": first_macs + 1,
        "accumulator_width_bits": 32,
        "datapath": (
            "signed-int8 input and factors; int32 accumulators; Scale32-derived "
            "signed-int32 multipliers with u6 right shifts; ties-to-even rounding; "
            "signed-int8 saturation; final correction in retained V output-scale quanta"
        ),
    }


def evaluate() -> None:
    require(FREEZE.is_file(), "integer candidate freeze is missing")
    require(not RESULT.exists(), "integer candidate result already exists")
    require(not SUMS.exists(), "integer candidate SHA256SUMS already exists")
    freeze = read_json(FREEZE)
    validate_freeze(freeze)
    localizer = residual.diagnosis.prior.localizer
    protected_before = protected_records()
    started = time.monotonic()
    diagnosis = read_json(FLOAT_DIAGNOSIS_RESULT)
    rank1 = next(
        item for item in diagnosis["candidates"] if int(item["rank"]) == 1
    )
    input_dequantized = residual.load_tensor(
        diagnosis["artifacts"]["tensors"]["input_dequantized"]
    ).to(torch.float64)
    oracle_v = residual.load_tensor(
        diagnosis["artifacts"]["tensors"]["bf16_v_output"]
    ).to(torch.float64)
    baseline_output = residual.load_tensor(
        diagnosis["artifacts"]["tensors"]["baseline_w4a8_v_output"]
    ).to(torch.float64)
    float_rank1_predicted = residual.load_tensor(
        rank1["artifacts"]["predicted_residual"]
    ).to(torch.float64)
    float_rank1_corrected = residual.load_tensor(
        rank1["artifacts"]["corrected_v_output"]
    ).to(torch.float64)
    integer_evidence, correction, corrected = integer_arithmetic(
        freeze, input_dequantized, baseline_output
    )
    projection = split_projection_summary(corrected, oracle_v)
    baseline_projection = split_projection_summary(baseline_output, oracle_v)
    float_oracle_projection = split_projection_summary(float_rank1_corrected, oracle_v)
    float_oracle_retention = {
        "predicted_residual_vs_float_rank1": split_projection_summary(
            correction, float_rank1_predicted
        ),
        "corrected_v_output_vs_float_rank1": split_projection_summary(
            corrected, float_rank1_corrected
        ),
    }

    snapshot = residual.diagnosis.prior.generation_runner.resolve_snapshot()
    residual.diagnosis.prior.backend.configure_snapshot(snapshot)
    canonical = residual.diagnosis.prior.backend.canonical
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
    token_ids = residual.diagnosis.prior.generation_runner.canonical_chat_token_ids(
        tokenizer, residual.diagnosis.prior.PROMPT
    )
    float_freeze = read_json(residual.FREEZE)
    require(
        localizer.sha256_bytes(residual.canonical_bytes(token_ids))
        == float_freeze["prompt"]["token_ids_sha256"]
        and len(token_ids) == 34,
        "frozen tokenization changed",
    )
    residual.diagnosis.prior.localizer.PROMPT = residual.diagnosis.prior.PROMPT
    residual.diagnosis.prior.localizer.TOKEN_IDS = token_ids
    residual.diagnosis.prior.localizer.REFERENCE_TOKEN = TARGET_TOKEN
    print("ACE2_LAYER23_V_RANK1_INTEGER_BF16_CAPTURE", flush=True)
    layer_input, captures, bf16_logits = residual.diagnosis.capture_layer_context(
        snapshot, token_ids
    )
    require(
        localizer.tensor_sha256(captures["input_norm"])
        == diagnosis["bindings"]["captured_tensors"]["input_norm"]["sha256"]
        and localizer.tensor_sha256(captures["v"])
        == diagnosis["bindings"]["captured_tensors"]["v"]["sha256"],
        "captured V tensors changed from accepted diagnosis",
    )
    hidden_states = [torch.empty(0)] * LAYER + [layer_input]
    suppressed_messages: list[str] = []
    prior_output = residual.diagnosis.prior.OUTPUT
    residual.diagnosis.prior.OUTPUT = OUTPUT
    try:
        with safe_open(canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
            canonical.ADAPTER, framework="pt", device="cpu"
        ) as adapter:
            embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
            norm_gain = weights.get_tensor("model.norm.weight").contiguous()
            head = localizer.derive_lm_head_weights(embedding)
            cache = residual.ResidualCorrectionCache(weights, adapter)
            guard = residual.diagnosis.prior.localizer.SubstitutionRequireGuard()
            cache.install()
            guard.install()
            try:
                baseline_record, baseline_state = residual.run_downstream(
                    cache,
                    "baseline-w4a8",
                    None,
                    hidden_states,
                    weights,
                    adapter,
                    norm_gain,
                    embedding,
                    head,
                    tokenizer,
                    bf16_logits,
                )
                integer_record, integer_state = residual.run_downstream(
                    cache,
                    "rank1-integer-s8",
                    correction,
                    hidden_states,
                    weights,
                    adapter,
                    norm_gain,
                    embedding,
                    head,
                    tokenizer,
                    bf16_logits,
                )
            finally:
                suppressed_messages.extend(guard.suppressed)
                guard.restore()
                cache.restore()
    finally:
        residual.diagnosis.prior.OUTPUT = prior_output
    baseline_metrics = precision.summary_metrics(baseline_record)
    candidate_metrics = precision.summary_metrics(integer_record)
    require(
        baseline_record["accepted_w4a8_logits"]["sha256"]
        == EXPECTED_BASELINE["logits_sha256"]
        and baseline_metrics["target_token_rank"] == EXPECTED_BASELINE["target_rank"]
        and baseline_metrics["top_token_id"] == EXPECTED_BASELINE["top_token_id"],
        "accepted cut-23 W4A8 baseline did not reproduce",
    )
    baseline_byte_match = torch.equal(
        baseline_state["corrected_output"].to(torch.float64), baseline_output
    )
    require(baseline_byte_match, "existing W4A8 V path changed during candidate run")
    require(
        torch.equal(integer_state["corrected_output"].to(torch.float64), corrected),
        "downstream replay did not use the frozen integer corrected V output",
    )
    baseline_projection = split_projection_summary(
        baseline_state["corrected_output"].to(torch.float64), oracle_v
    )
    projection = split_projection_summary(
        integer_state["corrected_output"].to(torch.float64), oracle_v
    )
    result_decision = decision(
        baseline_metrics,
        candidate_metrics,
        baseline_projection,
        projection,
        bool(integer_evidence["hardware_representable"]),
        baseline_byte_match,
    )
    bf16_reference = np.fromfile(BF16_LOGITS, dtype="<f4")
    float_rank1_logits = np.fromfile(FLOAT_RANK1_LOGITS, dtype="<f8")
    integer_logits = np.fromfile(
        ROOT / integer_record["accepted_w4a8_logits"]["path"], dtype="<f8"
    )
    logit_retention = {
        "integer_vs_float_rank1_raw_score_error": residual.diagnosis.prior.quality.vector_metrics(
            float_rank1_logits, integer_logits
        ),
        "integer_vs_bf16_recomputed_comparison": residual.diagnosis.prior.quality.score_comparison(
            bf16_reference, integer_logits
        ),
    }
    protected_after = protected_records()
    require(protected_before == protected_after, "accepted predecessor evidence changed")
    shutil.copyfile(Path(__file__).resolve(), EXECUTION_RUNNER)
    result = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "status": (
            "INTEGER_CANDIDATE_ACCEPTED_NONOFFICIAL"
            if result_decision["accepted_integer_candidate"]
            else "INTEGER_CANDIDATE_REJECTED_NONOFFICIAL"
        ),
        "classification": (
            "bounded_nonofficial_software_only_rank1_integer_v_residual_candidate"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "execution": {
            "layer_cut": LAYER,
            "prompt_token_count": len(token_ids),
            "generation_position": 0,
            "elapsed_wall_seconds": time.monotonic() - started,
            "freeze_existed_before_downstream_evaluation": True,
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
            "captured_tensors": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": localizer.tensor_sha256(value),
                }
                for name, value in sorted(captures.items())
            },
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
        "baseline": {
            "metrics": baseline_metrics,
            "logits": baseline_record["accepted_w4a8_logits"],
            "projection_output": baseline_projection,
        },
        "float_rank1_oracle": {
            "metrics": copy.deepcopy(rank1["metrics"]),
            "logits": copy.deepcopy(rank1["logits"]),
            "projection_output": float_oracle_projection,
        },
        "integer_candidate": {
            "metrics": candidate_metrics,
            "logits": integer_record["accepted_w4a8_logits"],
            "projection_output": projection,
            "float_oracle_retention": float_oracle_retention,
            "logit_retention": logit_retention,
            "integer_evidence": integer_evidence,
            "cost_cycle_storage_estimates": estimates(),
        },
        "decision": result_decision,
        "scope_guards": copy.deepcopy(SCOPE_GUARDS),
        "suppressed_nonfunctional_cache_effect_assertions": {
            "count": len(suppressed_messages),
            "unique_messages": sorted(set(suppressed_messages)),
        },
        "independent_review_required": True,
        "no_claims": {
            "official_attempt": False,
            "attempt_0003": False,
            "rtl": False,
            "icarus": False,
            "formal": False,
            "synthesis_or_ppa": False,
            "fpga_u280_xrt": False,
            "product_completion": False,
        },
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
            ".venv/bin/python "
            "tools/run_stage1_layer23_v_rank1_integer_candidate.py "
            "--verify-only"
        ),
        "required_checks": [
            "Verify the accepted 6a6e4fc4309f Fresh-L2 rank-1 diagnosis binding.",
            "Confirm candidate-freeze.json predates downstream logit evaluation and freezes all factor/input/rank scales.",
            "Recompute signed-int8 input/factor payloads, int32 accumulators, Scale32 multipliers/shifts, ties-to-even rounding, and saturations.",
            "Recompute calibration and held-out V projection relative-L2 values.",
            "Recompute frozen cut-23 downstream logits metrics and integer-vs-float rank-1 retention.",
            "Confirm the existing W4A8 baseline path byte-matches and no RTL/Icarus/official/PPA/Stage-2 artifact was produced.",
        ],
    }
    write_json(REVIEW_REQUEST, review)
    generate_sums()
    print(
        "ACE2_LAYER23_V_RANK1_INTEGER_RESULT "
        f"status={result['status']} "
        f"rank={candidate_metrics['target_token_rank']} "
        f"top8={candidate_metrics['top8_overlap_count']} "
        f"jsd={candidate_metrics['jensen_shannon_divergence_nats']:.12f} "
        f"heldout_v_rel_l2={projection['held_out']['relative_l2']:.12f} "
        f"accepted={result_decision['accepted_integer_candidate']} "
        f"output={localizer.public_path(OUTPUT)}",
        flush=True,
    )


def verify() -> None:
    require(FREEZE.is_file(), "integer candidate freeze is missing")
    require(RESULT.is_file(), "integer candidate result is missing")
    require(REVIEW_REQUEST.is_file(), "Fresh-L2 review request is missing")
    require(EXECUTION_RUNNER.is_file(), "archived execution runner is missing")
    require(SUMS.is_file(), "integer candidate SHA256SUMS is missing")
    localizer = residual.diagnosis.prior.localizer
    for line in SUMS.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        artifact = OUTPUT / relative
        require(artifact.is_file(), f"summed artifact missing: {relative}")
        require(
            localizer.sha256_file(artifact) == expected,
            f"summed artifact hash changed: {relative}",
        )
    freeze = read_json(FREEZE)
    validate_freeze(freeze)
    result = read_json(RESULT)
    require(result["scope_guards"] == SCOPE_GUARDS, "result scope guards changed")
    require(
        result["bindings"]["freeze"] == localizer.file_record(FREEZE),
        "result freeze binding changed",
    )
    require(
        result["bindings"]["runner"]["sha256"] == freeze["runner"]["sha256"]
        and result["bindings"]["archived_execution_runner"]["sha256"]
        == freeze["runner"]["sha256"],
        "runner binding changed",
    )
    diagnosis = read_json(FLOAT_DIAGNOSIS_RESULT)
    rank1 = next(
        item for item in diagnosis["candidates"] if int(item["rank"]) == 1
    )
    input_dequantized = residual.load_tensor(
        diagnosis["artifacts"]["tensors"]["input_dequantized"]
    ).to(torch.float64)
    oracle_v = residual.load_tensor(
        diagnosis["artifacts"]["tensors"]["bf16_v_output"]
    ).to(torch.float64)
    baseline_output = residual.load_tensor(
        diagnosis["artifacts"]["tensors"]["baseline_w4a8_v_output"]
    ).to(torch.float64)
    integer_evidence, correction, corrected = integer_arithmetic(
        freeze, input_dequantized, baseline_output, write_artifacts=False
    )
    stored = result["integer_candidate"]["integer_evidence"]["artifacts"]
    require(
        torch.equal(load_tensor(stored["predicted_residual_f64le"]).to(torch.float64), correction)
        and torch.equal(load_tensor(stored["corrected_v_output_f64le"]).to(torch.float64), corrected),
        "stored integer correction tensors changed",
    )
    projection = split_projection_summary(corrected, oracle_v)
    require(
        projection == result["integer_candidate"]["projection_output"],
        "integer projection metrics changed",
    )
    baseline_projection = split_projection_summary(baseline_output, oracle_v)
    require(
        baseline_projection == result["baseline"]["projection_output"],
        "baseline projection metrics changed",
    )
    float_rank1_corrected = residual.load_tensor(
        rank1["artifacts"]["corrected_v_output"]
    ).to(torch.float64)
    require(
        split_projection_summary(float_rank1_corrected, oracle_v)
        == result["float_rank1_oracle"]["projection_output"],
        "float rank-1 oracle projection metrics changed",
    )
    bf16_reference = np.fromfile(BF16_LOGITS, dtype="<f4")
    integer_logits = np.fromfile(
        ROOT / result["integer_candidate"]["logits"]["path"], dtype="<f8"
    )
    comparison = residual.diagnosis.prior.quality.score_comparison(
        bf16_reference, integer_logits
    )
    require(
        comparison
        == result["integer_candidate"]["metrics"][
            "comparison_vs_checkpoint176_bf16"
        ],
        "integer logit metrics changed",
    )
    candidate_metrics = {
        "target_token_id": TARGET_TOKEN,
        "target_token_rank": int(comparison["reference_top_rank_in_candidate"]),
        "target_token_top1": (
            int(comparison["reference_top_rank_in_candidate"]) == 1
            and int(np.argmax(integer_logits)) == TARGET_TOKEN
        ),
        "top_token_id": int(np.argmax(integer_logits)),
        "top8_overlap_count": int(comparison["top8_overlap_count"]),
        "jensen_shannon_divergence_nats": float(
            comparison["jensen_shannon_divergence_nats"]
        ),
        "relative_l2": float(comparison["raw_score_error"]["relative_l2"]),
        "comparison_vs_checkpoint176_bf16": comparison,
    }
    require(
        candidate_metrics == result["integer_candidate"]["metrics"],
        "integer summary metrics changed",
    )
    require(
        decision(
            result["baseline"]["metrics"],
            result["integer_candidate"]["metrics"],
            result["baseline"]["projection_output"],
            result["integer_candidate"]["projection_output"],
            bool(integer_evidence["hardware_representable"]),
            True,
        )
        == result["decision"],
        "integer candidate decision changed",
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
            / "evidence/verification/rtl-arbitrary-text-generation-v1/attempt-0003"
        ).exists(),
        "attempt-0003 exists",
    )
    print(
        "PASS stage1vrank1int01 "
        f"status={result['status']} "
        f"rank={result['integer_candidate']['metrics']['target_token_rank']} "
        f"heldout_v_rel_l2={result['integer_candidate']['projection_output']['held_out']['relative_l2']:.9f} "
        f"accepted={result['decision']['accepted_integer_candidate']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
    elif args.freeze_only:
        freeze_factor_quantization()
    elif args.evaluate:
        evaluate()
    else:
        freeze_factor_quantization()
        evaluate()


if __name__ == "__main__":
    main()
