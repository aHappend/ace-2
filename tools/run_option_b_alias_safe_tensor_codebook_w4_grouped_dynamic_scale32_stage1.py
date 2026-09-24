#!/usr/bin/env python3
"""Construct and verify the marker-free tensor-codebook V3 preflight.

This runner intentionally exposes no official execution command.  It may create
only the V3 pre-attempt namespace and either a verified PREATTEMPT_READY record
or a verified PREATTEMPT_NO_GO record.  It never creates attempt-0001 or an
execution_started.json marker.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_option_b_grouped_scale32_stage1 as grouped
from ace2_quality_contracts import unpack_scale32
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
    write_json,
)
from qwen_instruct_response_gate import evaluate as evaluate_response_gate
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM
from run_all_transformer_per32_stage1 import utc_now


CANDIDATE_ID = "option_b_alias_safe_tensor_codebook_w4_grouped_dynamic_scale32_w4a8_v3"
MISSION_ID = "fe7b5c547e7f"
TASK_ID = "task-904869143c15"
TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V3_ENGINEER_TASK.json"
TASK_COMPANION = TASK_PATH.with_suffix(".sha256")
PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V3_PLAN.json"
PLAN_COMPANION = PLAN_PATH.with_suffix(".sha256")
SOURCE_CONTRACT_PATH = ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
CALIBRATION_CONTRACT_PATH = CALIBRATION_DIR / "calibration_contract.json"
CALIBRATION_REPORT_PATH = CALIBRATION_DIR / "calibration_report.json"
BASE_SCALES_PATH = CALIBRATION_DIR / "derived_scales.json"
MATRIX_PATH = ROOT / "build/option-b-v2-weight-policy-20260806/full_sequence_discrimination.json"
MATRIX_SHA256 = "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
SOURCE_EMBEDDING_SHA256 = "4e96b0df6d274768cbb7e72404011853d23349999b658dc2f4dfb3c431ea223f"
OUTPUT_DIR = ROOT / "build/stage1-option-b-alias-safe-tensor-codebook-w4-grouped-dynamic-scale32-w4a8-v3"
PREATTEMPT_DIR = OUTPUT_DIR / "preattempt"
INPUT_MANIFEST_PATH = PREATTEMPT_DIR / "non_evaluator_inputs.json"
CANDIDATE_A_PATH = PREATTEMPT_DIR / "candidate_a_manifest.json"
CANDIDATE_B_PATH = PREATTEMPT_DIR / "candidate_b_manifest.json"
ALIAS_WITNESS_PATH = PREATTEMPT_DIR / "alias_separation_witness.json"
RECONSTRUCTION_PATH = PREATTEMPT_DIR / "model_only_reconstruction.json"
QUALITY_PATH = PREATTEMPT_DIR / "non_evaluator_quality.json"
DYNAMIC_PATH = PREATTEMPT_DIR / "dynamic_preflight.json"
SELF_TEST_PATH = PREATTEMPT_DIR / "runner_self_test.json"
CHRONOLOGY_PATH = PREATTEMPT_DIR / "preparation_chronology.json"
RECOVERY_PATH = PREATTEMPT_DIR / "preparation_recovery.json"
CONTRACT_PATH = OUTPUT_DIR / "predeclared_contract.json"
READY_PATH = OUTPUT_DIR / "PREATTEMPT_READY.json"
NO_GO_PATH = OUTPUT_DIR / "PREATTEMPT_NO_GO.json"
ATTEMPT_DIR = OUTPUT_DIR / "attempt-0001"
MARKER_PATH = ATTEMPT_DIR / "execution_started.json"
RUNNER_PATH = Path(__file__).resolve()
GROUPED_RUNNER_PATH = ROOT / "tools/run_option_b_grouped_scale32_stage1.py"
MAX_NEW_TOKENS = 6
TORCH_THREADS = 16
EXPECTED_EVENTS = grouped.LAYERS * len(grouped.EVENT_FAMILIES)

Q4_4_MIN = -128
Q4_4_MAX = 127
CODEBOOK_ENTRIES = 16
LLOYD_MAX_ITERATIONS = 32
INITIAL_CODEPOINTS = tuple(range(-128, 113, 16))
INTEGER_DTYPES = {torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}
ADDITIONAL_PROMPTS = (
    "Name one primary color in one lowercase word.",
    "Return only the integer result of 9 multiplied by 7.",
    "In five words, describe gentle rain.",
    "Answer yes or no: Is water wet?",
)
EXPECTED_TEXTS = {
    "frozen_calibration_00": "amber compass",
    "frozen_calibration_01": "The result of 17 plus 25 is 42.",
    "frozen_calibration_02": "A small satellite crossed the evening sky while city lights came on.",
    "frozen_calibration_03": "The clear sky is blue.",
    "additional_preflight_00": "red",
    "additional_preflight_01": "63",
    "additional_preflight_02": "Gentle rain falls softly tonight.",
    "additional_preflight_03": "yes",
}
STATE_MATCH_KEYS = (
    "embedding_bf16_sha256",
    "payload_manifest_sha256",
    "row_scale_manifest_sha256",
    "codebook_manifest_sha256",
    "weight_manifest_sha256",
    "reconstruction_report_sha256",
    "candidate_state_manifest_sha256",
)


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    return hashlib.sha256(raw).hexdigest()


def require_project_python() -> dict[str, Any]:
    grouped.require_project_python()
    expected = ROOT / ".venv/bin/python"
    require(expected.is_file(), "project Python is missing")
    require(os.path.samefile(Path(sys.executable), expected), "wrong Python executable")
    return {
        "entrypoint": "./.venv/bin/python",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {**verify_versions(), "numpy": importlib.metadata.version("numpy")},
        "torch_num_threads": TORCH_THREADS,
    }


def verify_companion(path: Path, companion: Path) -> None:
    expected = f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}"
    require(companion.read_text(encoding="utf-8").strip() == expected, f"checksum companion differs: {path}")


def require_digest(actual: str, expected: str, message: str) -> None:
    require(actual == expected, message)


def atomic_write_json(path: Path, value: Any) -> None:
    raw = canonical_bytes(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def raw_tensor_bytes(value: Tensor) -> bytes:
    flat = value.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes(order="C")


def tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(raw_tensor_bytes(value)).hexdigest()


def state_manifest(model: nn.Module) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for name, value in sorted(model.state_dict().items()):
        raw = raw_tensor_bytes(value)
        records.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    body = {
        "config_tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "tensor_count": len(records),
        "tensors": records,
    }
    return {**body, "candidate_state_manifest_sha256": canonical_sha256(body)}


def validate_integer_tensor(source: Tensor, minimum: int, maximum: int, label: str) -> Tensor:
    """Validate original integer values before any dtype narrowing."""
    require(isinstance(source, Tensor), f"{label} is not a tensor")
    original = source.detach().to(device="cpu")
    require(original.dtype in INTEGER_DTYPES, f"{label} does not have integer semantics")
    require(original.numel() > 0, f"empty {label}")
    observed_min = int(original.min())
    observed_max = int(original.max())
    require(minimum <= observed_min and observed_max <= maximum, f"{label} escaped [{minimum},{maximum}]")
    return original.contiguous()


def round_ratio_ties_to_even(numerator: int, denominator: int) -> int:
    """Round an exact signed rational to nearest, choosing even on a half tie."""
    require(type(numerator) is int, "rounding numerator must be an integer")
    require(type(denominator) is int and denominator > 0, "rounding denominator must be a positive integer")
    magnitude = abs(numerator)
    quotient, remainder = divmod(magnitude, denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient & 1):
        quotient += 1
    return -quotient if numerator < 0 else quotient


def validate_codebook(codepoints: Sequence[int]) -> tuple[int, ...]:
    points = tuple(codepoints)
    require(len(points) == CODEBOOK_ENTRIES, "codebook entry count differs")
    require(all(type(value) is int for value in points), "codebook point is not an integer")
    require(all(Q4_4_MIN <= value <= Q4_4_MAX for value in points), "codebook point is outside signed Q4.4")
    require(all(left < right for left, right in zip(points, points[1:])), "codebook points are not strictly increasing")
    return points


def nearest_codepoint_index(value: int, codepoints: Sequence[int]) -> int:
    """Return the nearest point; scanning order implements lower-index ties."""
    require(type(value) is int, "assignment value is not an integer")
    points = validate_codebook(codepoints)
    selected = 0
    selected_distance = abs(value - points[0])
    for index, point in enumerate(points[1:], start=1):
        distance = abs(value - point)
        if distance < selected_distance:
            selected = index
            selected_distance = distance
    return selected


def histogram_sha256(histogram: Sequence[int]) -> str:
    require(len(histogram) == 256, "Q4.4 histogram bin count differs")
    require(all(type(count) is int and 0 <= count < (1 << 64) for count in histogram), "Q4.4 histogram count differs")
    return hashlib.sha256(struct.pack("<256Q", *histogram)).hexdigest()


def construct_codebook_from_histogram(
    histogram: Sequence[int],
    *,
    iterations: int = LLOYD_MAX_ITERATIONS,
) -> dict[str, Any]:
    """Run the frozen exact-histogram Lloyd-Max construction."""
    require(type(iterations) is int and iterations == LLOYD_MAX_ITERATIONS, "Lloyd-Max iteration count differs")
    counts_by_value = tuple(histogram)
    require(len(counts_by_value) == 256, "Q4.4 histogram bin count differs")
    require(all(type(count) is int for count in counts_by_value), "Q4.4 histogram count is not an integer")
    require(all(0 <= count < (1 << 64) for count in counts_by_value), "Q4.4 histogram count differs")
    require(0 < sum(counts_by_value) < (1 << 64), "empty or oversized Q4.4 histogram")

    codepoints = validate_codebook(INITIAL_CODEPOINTS)
    trace: list[dict[str, Any]] = []
    for iteration in range(1, iterations + 1):
        cluster_counts = [0] * CODEBOOK_ENTRIES
        cluster_sums = [0] * CODEBOOK_ENTRIES
        for offset, count in enumerate(counts_by_value):
            if count == 0:
                continue
            value = offset + Q4_4_MIN
            index = nearest_codepoint_index(value, codepoints)
            cluster_counts[index] += count
            cluster_sums[index] += value * count

        updated = [
            prior if count == 0 else round_ratio_ties_to_even(total, count)
            for prior, total, count in zip(codepoints, cluster_sums, cluster_counts, strict=True)
        ]
        updated.sort()
        codepoints = validate_codebook(updated)
        trace.append(
            {
                "iteration": iteration,
                "codepoints": list(codepoints),
                "empty_cluster_count": sum(count == 0 for count in cluster_counts),
                "cluster_counts_sha256": hashlib.sha256(struct.pack("<16Q", *cluster_counts)).hexdigest(),
            }
        )

    record = bytes(value & 0xFF for value in codepoints)
    require(len(record) == 16, "codebook record byte count differs")
    return {
        "codepoints": codepoints,
        "record": record,
        "record_sha256": hashlib.sha256(record).hexdigest(),
        "histogram_sha256": histogram_sha256(counts_by_value),
        "iterations_completed": len(trace),
        "trace": trace,
        "trace_sha256": canonical_sha256(trace),
    }


def normalize_q4_4(source: Tensor, row_scales: Tensor) -> Tensor:
    """Normalize a two-dimensional BF16 weight tensor to exact Q4.4 bins."""
    require(source.ndim == 2, "weight tensor rank differs")
    require(row_scales.ndim == 1 and row_scales.numel() == source.shape[0], "row-scale shape differs")
    weights = source.detach().to(device="cpu", dtype=torch.float64)
    scales = row_scales.detach().to(device="cpu", dtype=torch.float64)
    require(bool(torch.all(torch.isfinite(weights))), "weight tensor contains a non-finite value")
    require(bool(torch.all(torch.isfinite(scales) & (scales > 0))), "row scale is non-finite or non-positive")
    normalized = torch.round((weights / scales[:, None]) * 16.0)
    require(bool(torch.all((normalized >= Q4_4_MIN) & (normalized <= Q4_4_MAX))), "normalized weight escaped signed Q4.4")
    return normalized.to(torch.int16)


def exact_histogram(values: Tensor) -> tuple[int, ...]:
    original = validate_integer_tensor(values, Q4_4_MIN, Q4_4_MAX, "normalized weight")
    flat = original.to(dtype=torch.int16).reshape(-1)
    counts = torch.bincount(flat.to(torch.int64) - Q4_4_MIN, minlength=256)
    require(counts.numel() == 256 and int(counts.sum()) == flat.numel(), "Q4.4 histogram accounting differs")
    return tuple(int(value) for value in counts.tolist())


def assign_codebook_indices(values: Tensor, codepoints: Sequence[int], *, chunk_values: int = 1 << 20) -> Tensor:
    require(type(chunk_values) is int and chunk_values > 0, "assignment chunk size differs")
    points = torch.tensor(validate_codebook(codepoints), dtype=torch.int16)
    original = validate_integer_tensor(values, Q4_4_MIN, Q4_4_MAX, "normalized weight")
    flat = original.to(dtype=torch.int16).reshape(-1)
    output = torch.empty(flat.numel(), dtype=torch.uint8)
    for start in range(0, flat.numel(), chunk_values):
        stop = min(start + chunk_values, flat.numel())
        distances = torch.abs(flat[start:stop, None] - points[None, :])
        output[start:stop] = torch.argmin(distances, dim=1).to(torch.uint8)
    require(bool(torch.all(output <= 15)), "codebook assignment escaped four bits")
    return output.reshape(values.shape)


def pack_codebook_indices(indices: Tensor) -> bytes:
    original = validate_integer_tensor(indices, 0, 15, "codebook payload index")
    flat = original.to(dtype=torch.uint8).reshape(-1)
    if flat.numel() & 1:
        flat = torch.cat((flat, torch.zeros(1, dtype=torch.uint8)))
    packed = torch.bitwise_or(flat[0::2], torch.bitwise_left_shift(flat[1::2], 4))
    return packed.numpy().tobytes(order="C")


def expect_rejected(call: Any, message: str) -> bool:
    rejected = False
    try:
        call()
    except RuntimeError:
        rejected = True
    require(rejected, message)
    return rejected


def alias_split_self_test() -> dict[str, Any]:
    embedding = nn.Embedding(4, 4, dtype=torch.bfloat16)
    head = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    head.weight = embedding.weight
    require(head.weight is embedding.weight, "synthetic alias object is missing")
    before = tensor_sha256(embedding.weight)
    head.weight = nn.Parameter(head.weight.detach().clone(), requires_grad=False)
    require(head.weight is not embedding.weight, "synthetic alias object was not separated")
    require(head.weight.data_ptr() != embedding.weight.data_ptr(), "synthetic alias storage was not separated")
    require(tensor_sha256(head.weight) == before == tensor_sha256(embedding.weight), "synthetic split changed bytes")
    with torch.no_grad():
        head.weight.add_(1)
    require(tensor_sha256(embedding.weight) == before, "synthetic head write changed embedding")
    return {
        "same_object_before": True,
        "same_storage_before": True,
        "different_object_after": True,
        "different_storage_after": True,
        "embedding_unchanged_after_head_write": True,
    }


def codebook_self_test() -> dict[str, Any]:
    require_project_python()
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    task = load_json(TASK_PATH)
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "frozen Engineer authority differs")

    root_existed_before = OUTPUT_DIR.exists()
    marker_existed_before = MARKER_PATH.exists()
    tie_cases = {
        "positive_even": round_ratio_ties_to_even(5, 2),
        "positive_odd": round_ratio_ties_to_even(7, 2),
        "negative_even": round_ratio_ties_to_even(-5, 2),
        "negative_odd": round_ratio_ties_to_even(-7, 2),
    }
    require(tie_cases == {"positive_even": 2, "positive_odd": 4, "negative_even": -2, "negative_odd": -4}, "ties-to-even rounding differs")

    normalized_ties = normalize_q4_4(
        torch.tensor([[0.15625, 0.21875, -0.15625, -0.21875]], dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
    )
    require(normalized_ties.tolist() == [[2, 4, -2, -4]], "Q4.4 normalization ties-to-even differs")
    require(nearest_codepoint_index(-120, INITIAL_CODEPOINTS) == 0, "assignment tie did not choose lower index")

    uniform_histogram = (1,) * 256
    construction_a = construct_codebook_from_histogram(uniform_histogram)
    construction_b = construct_codebook_from_histogram(uniform_histogram)
    require(construction_a == construction_b, "independent codebook constructions differ")
    require(construction_a["iterations_completed"] == LLOYD_MAX_ITERATIONS, "Lloyd-Max did not complete exactly 32 iterations")

    empty_cluster_histogram = [0] * 256
    for point in INITIAL_CODEPOINTS:
        if point != -16:
            empty_cluster_histogram[point - Q4_4_MIN] = 1
    empty_cluster = construct_codebook_from_histogram(empty_cluster_histogram)
    require(empty_cluster["codepoints"] == INITIAL_CODEPOINTS, "empty cluster did not retain its prior point")
    require(all(item["empty_cluster_count"] == 1 for item in empty_cluster["trace"]), "empty-cluster accounting differs")

    duplicate_rejected = expect_rejected(
        lambda: validate_codebook((*INITIAL_CODEPOINTS[:-1], INITIAL_CODEPOINTS[-2])),
        "duplicate codebook point was not rejected",
    )
    range_rejected = expect_rejected(
        lambda: validate_codebook((*INITIAL_CODEPOINTS[:-1], 128)),
        "out-of-range codebook point was not rejected",
    )
    normalized_range_rejected = expect_rejected(
        lambda: normalize_q4_4(torch.tensor([[8.0]], dtype=torch.float64), torch.ones(1, dtype=torch.float64)),
        "out-of-range normalized weight was not rejected",
    )

    q4_invalid = (-65536, -129, 128, 65536)
    pack_invalid = (-256, -241, -1, 16, 256, 271)
    histogram_wraparound_rejected = {
        str(value): expect_rejected(
            lambda value=value: exact_histogram(torch.tensor([value], dtype=torch.int64)),
            f"histogram accepted wraparound/boundary value {value}",
        )
        for value in q4_invalid
    }
    assignment_wraparound_rejected = {
        str(value): expect_rejected(
            lambda value=value: assign_codebook_indices(torch.tensor([value], dtype=torch.int64), INITIAL_CODEPOINTS),
            f"assignment accepted wraparound/boundary value {value}",
        )
        for value in q4_invalid
    }
    packing_wraparound_rejected = {
        str(value): expect_rejected(
            lambda value=value: pack_codebook_indices(torch.tensor([value], dtype=torch.int64)),
            f"packing accepted wraparound/boundary value {value}",
        )
        for value in pack_invalid
    }
    integer_semantics_rejected = {
        "histogram_float": expect_rejected(
            lambda: exact_histogram(torch.tensor([0.0], dtype=torch.float64)),
            "histogram accepted non-integer tensor semantics",
        ),
        "assignment_float": expect_rejected(
            lambda: assign_codebook_indices(torch.tensor([0.0], dtype=torch.float64), INITIAL_CODEPOINTS),
            "assignment accepted non-integer tensor semantics",
        ),
        "packing_float": expect_rejected(
            lambda: pack_codebook_indices(torch.tensor([0.0], dtype=torch.float64)),
            "packing accepted non-integer tensor semantics",
        ),
        "histogram_count_float": expect_rejected(
            lambda: construct_codebook_from_histogram((1.0,) * 256),
            "constructor accepted non-integer histogram counts",
        ),
    }
    boundary_histogram = exact_histogram(torch.tensor([Q4_4_MIN, Q4_4_MAX], dtype=torch.int64))
    require(boundary_histogram[0] == 1 and boundary_histogram[-1] == 1 and sum(boundary_histogram) == 2, "legal Q4.4 boundaries differ")
    boundary_indices = assign_codebook_indices(torch.tensor([Q4_4_MIN, Q4_4_MAX], dtype=torch.int64), INITIAL_CODEPOINTS)
    require(boundary_indices.tolist() == [0, 15], "legal Q4.4 boundary assignments differ")
    require(pack_codebook_indices(torch.tensor([0, 15], dtype=torch.int64)) == bytes((0xF0,)), "legal four-bit boundary packing differs")

    sample_values = torch.tensor([[-128, -120, -112, -104, 112, 120, 127]], dtype=torch.int16)
    sample_indices = assign_codebook_indices(sample_values, INITIAL_CODEPOINTS, chunk_values=3)
    require(sample_indices.tolist() == [[0, 0, 1, 1, 15, 15, 15]], "lower-index assignment or chunking differs")
    packed = pack_codebook_indices(torch.tensor([0, 1, 15, 8, 3], dtype=torch.uint8))
    require(packed == bytes((0x10, 0x8F, 0x03)), "codebook nibble packing differs")

    hash_mismatch_rejected = expect_rejected(
        lambda: require_digest("0" * 64, "1" * 64, "bound artifact hash differs"),
        "bound-hash mismatch was not rejected",
    )
    require(OUTPUT_DIR.exists() == root_existed_before, "self-test changed the V3 namespace")
    require(MARKER_PATH.exists() == marker_existed_before is False, "self-test changed or observed an execution marker")
    return {
        "schema_version": 1,
        "status": "PASS",
        "candidate_id": CANDIDATE_ID,
        "mission_id": MISSION_ID,
        "task_id": TASK_ID,
        "runner_sha256": sha256_file(RUNNER_PATH),
        "task_sha256": sha256_file(TASK_PATH),
        "plan_sha256": sha256_file(PLAN_PATH),
        "initial_codepoints": list(INITIAL_CODEPOINTS),
        "iterations_completed": construction_a["iterations_completed"],
        "uniform_histogram_sha256": construction_a["histogram_sha256"],
        "deterministic_codebook_sha256": construction_a["record_sha256"],
        "deterministic_trace_sha256": construction_a["trace_sha256"],
        "ties_to_even": tie_cases,
        "lower_index_assignment_tie": True,
        "empty_cluster_retained": True,
        "duplicate_rejected": duplicate_rejected,
        "range_rejected": range_rejected,
        "normalized_range_rejected": normalized_range_rejected,
        "source_validation_before_narrowing": True,
        "histogram_wraparound_rejected": histogram_wraparound_rejected,
        "assignment_wraparound_rejected": assignment_wraparound_rejected,
        "packing_wraparound_rejected": packing_wraparound_rejected,
        "integer_semantics_rejected": integer_semantics_rejected,
        "immediate_boundaries_accepted": True,
        "nibble_packing_hex": packed.hex(),
        "bound_hash_mismatch_rejected": hash_mismatch_rejected,
        "official_build_namespace_changed": False,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": False,
    }


def policy_and_format() -> dict[str, Any]:
    body = {
        "payload": {
            "bits_per_weight": 4,
            "semantics": "unsigned codebook index 0 through 15",
            "packing": "row-major; even flat element low nibble; odd flat element high nibble",
        },
        "row_scale": {
            "format": "Scale32",
            "source": "pinned BF16 per-output-row absmax divided by seven, rounded upward to normalized Scale32",
            "bytes_per_row": 4,
            "mutation_permitted": False,
        },
        "codebook": {
            "scope": "one per Linear tensor",
            "entry_count": 16,
            "bytes": 16,
            "alignment_bytes": 16,
            "format": "strictly increasing signed Q4.4 int8",
            "initial_codepoints": list(INITIAL_CODEPOINTS),
            "lloyd_max_iterations": LLOYD_MAX_ITERATIONS,
            "assignment_tie": "lower index",
            "centroid_rounding": "round_to_nearest_ties_to_even",
            "empty_cluster": "retain prior codepoint",
        },
        "activation": {
            "policy": "grouped Dynamic Scale32",
            "event_count": EXPECTED_EVENTS,
            "payload_range": [-127, 127],
            "reserved_payload": -128,
            "delta_range": [-24, 24],
            "effective_exponent_range": [-24, 4],
        },
    }
    return {**body, "policy_and_format_sha256": canonical_sha256(body)}


def quantize_model_codebooks(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {record["module"]: record for record in grouped.expected_transformer_tensors()}
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == 169, "Qwen linear count differs")
    require([name for name, _module in modules if name != "lm_head"] == list(expected), "transformer linear order differs")

    weight_manifest: list[dict[str, Any]] = []
    reconstruction_tensors: list[dict[str, Any]] = []
    payload_manifest: list[dict[str, Any]] = []
    row_scale_manifest: list[dict[str, Any]] = []
    codebook_manifest: list[dict[str, Any]] = []
    payload_stream = hashlib.sha256()
    scale_stream = hashlib.sha256()
    codebook_stream = hashlib.sha256()
    total_weight_count = 0
    total_payload_bytes = 0
    total_scale_bytes = 0
    total_codebook_bytes = 0

    with torch.no_grad():
        for name, module in modules:
            source = module.weight.detach().to(device="cpu", dtype=torch.float64)
            require(source.ndim == 2, f"linear weight rank differs: {name}")
            absmax = source.abs().amax(dim=1)
            require(not bool(torch.any(absmax == 0)), f"all-zero weight row is unsupported: {name}")
            records = torch.tensor(
                [grouped.scale32_record(float(value)) for value in (absmax / 7.0).tolist()],
                dtype=torch.int64,
            )
            scales = torch.tensor(
                [grouped.scale32_value(int(value)) for value in records.tolist()],
                dtype=torch.float64,
            )
            normalized = normalize_q4_4(source, scales)
            histogram = exact_histogram(normalized)
            construction = construct_codebook_from_histogram(histogram)
            indices = assign_codebook_indices(normalized, construction["codepoints"])
            payload = pack_codebook_indices(indices)
            scale_raw = b"".join(struct.pack("<I", int(value)) for value in records.tolist())
            codebook_raw = construction["record"]

            points = torch.tensor(construction["codepoints"], dtype=torch.float64)
            candidate_decoded = points[indices.to(torch.int64)] * (scales[:, None] / 16.0)
            baseline_q = torch.round(source / scales[:, None]).clamp(-8, 7)
            baseline_decoded = baseline_q * scales[:, None]
            candidate_bf16 = candidate_decoded.to(torch.bfloat16).to(torch.float64)
            baseline_bf16 = baseline_decoded.to(torch.bfloat16).to(torch.float64)
            baseline_sse = float(torch.sum(torch.square(source - baseline_bf16)))
            candidate_sse = float(torch.sum(torch.square(source - candidate_bf16)))
            normalized_uniform = torch.tensor(INITIAL_CODEPOINTS, dtype=torch.int16)[
                assign_codebook_indices(normalized, INITIAL_CODEPOINTS).to(torch.int64)
            ]
            normalized_codebook = points.to(torch.int16)[indices.to(torch.int64)]
            baseline_normalized_sse = int(torch.sum(torch.square(normalized.to(torch.int32) - normalized_uniform.to(torch.int32))))
            candidate_normalized_sse = int(torch.sum(torch.square(normalized.to(torch.int32) - normalized_codebook.to(torch.int32))))
            module.weight.copy_(candidate_decoded.to(module.weight.dtype))

            payload_sha = hashlib.sha256(payload).hexdigest()
            scale_sha = hashlib.sha256(scale_raw).hexdigest()
            codebook_sha = hashlib.sha256(codebook_raw).hexdigest()
            weight_count = int(module.weight.numel())
            require(len(payload) == (weight_count + 1) // 2, f"packed payload byte count differs: {name}")
            require(len(scale_raw) == module.weight.shape[0] * 4, f"row-scale byte count differs: {name}")
            require(len(codebook_raw) == 16, f"codebook byte count differs: {name}")

            payload_record = {"module": name, "bytes": len(payload), "sha256": payload_sha}
            scale_record = {
                "module": name,
                "record_count": int(records.numel()),
                "bytes": len(scale_raw),
                "sha256": scale_sha,
            }
            codebook_record = {
                "module": name,
                "bytes": len(codebook_raw),
                "sha256": codebook_sha,
                "q4_4_codepoints": list(construction["codepoints"]),
                "histogram_sha256": construction["histogram_sha256"],
                "iterations_completed": construction["iterations_completed"],
                "trace_sha256": construction["trace_sha256"],
            }
            reconstruction_record = {
                "module": name,
                "weight_count": weight_count,
                "uniform_baseline_bf16_sse": baseline_sse,
                "tensor_codebook_bf16_sse": candidate_sse,
                "tensor_codebook_non_regressing": candidate_sse <= baseline_sse,
                "uniform_baseline_normalized_q4_4_sse": baseline_normalized_sse,
                "tensor_codebook_normalized_q4_4_sse": candidate_normalized_sse,
                "normalized_q4_4_non_regressing": candidate_normalized_sse <= baseline_normalized_sse,
            }
            record = {
                "module": name,
                "name": f"{name}.weight",
                "shape": list(module.weight.shape),
                "scope": "separated_lm_head" if name == "lm_head" else "transformer",
                "payload": payload_record,
                "row_scale": scale_record,
                "codebook": codebook_record,
                "reconstruction": reconstruction_record,
            }
            weight_manifest.append(record)
            reconstruction_tensors.append(reconstruction_record)
            payload_manifest.append(payload_record)
            row_scale_manifest.append(scale_record)
            codebook_manifest.append(codebook_record)
            payload_stream.update(name.encode() + b"\0" + payload)
            scale_stream.update(name.encode() + b"\0" + scale_raw)
            codebook_stream.update(name.encode() + b"\0" + codebook_raw)
            total_weight_count += weight_count
            total_payload_bytes += len(payload)
            total_scale_bytes += len(scale_raw)
            total_codebook_bytes += len(codebook_raw)

    aggregate_baseline_sse = float(sum(item["uniform_baseline_bf16_sse"] for item in reconstruction_tensors))
    aggregate_candidate_sse = float(sum(item["tensor_codebook_bf16_sse"] for item in reconstruction_tensors))
    non_regressing_count = sum(bool(item["tensor_codebook_non_regressing"]) for item in reconstruction_tensors)
    reconstruction_body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "measurement": "sum-squared error from pinned BF16 source to BF16-materialized reconstruction",
        "uniform_baseline": "authenticated signed-int4 per-output-row Scale32 reconstruction",
        "tensor_count": len(reconstruction_tensors),
        "per_tensor_non_regressing_count": non_regressing_count,
        "all_169_tensors_non_regressing": non_regressing_count == 169,
        "aggregate_uniform_baseline_bf16_sse": aggregate_baseline_sse,
        "aggregate_tensor_codebook_bf16_sse": aggregate_candidate_sse,
        "aggregate_strictly_improved": aggregate_candidate_sse < aggregate_baseline_sse,
        "tensors": reconstruction_tensors,
    }
    reconstruction_report = {
        **reconstruction_body,
        "status": (
            "PASS_NO_PER_TENSOR_REGRESSION_AND_STRICT_AGGREGATE_IMPROVEMENT"
            if reconstruction_body["all_169_tensors_non_regressing"] and reconstruction_body["aggregate_strictly_improved"]
            else "NO_GO_MODEL_ONLY_RECONSTRUCTION_GATE"
        ),
        "reconstruction_report_sha256": canonical_sha256(reconstruction_body),
    }

    transformer_records = [record for record in weight_manifest if record["scope"] == "transformer"]
    standard_layer_weights = grouped.policy_cost()["standard_decoder_layer_weight_count"]
    standard_layer_existing_bytes = grouped.policy_cost()["standard_decoder_layer_total_weight_bytes"]
    standard_layer_codebook_bytes = 7 * 16
    standard_layer_bits = ((standard_layer_existing_bytes + standard_layer_codebook_bytes) * 8) / standard_layer_weights
    lm_head_record = next(record for record in weight_manifest if record["module"] == "lm_head")
    lm_head_weight_count = int(np.prod(lm_head_record["shape"]))
    lm_head_total_bytes = lm_head_record["payload"]["bytes"] + lm_head_record["row_scale"]["bytes"] + 16
    lm_head_bits = (lm_head_total_bytes * 8) / lm_head_weight_count
    require(len(transformer_records) == 168 and len(weight_manifest) == 169, "codebook tensor count differs")
    require(total_codebook_bytes == 2704, "whole-model codebook byte count differs")
    require(abs(standard_layer_bits - 4.027257898351649) < 1e-15, "standard-layer stored bits differ")
    require(abs(lm_head_bits - 4.035715225959803) < 1e-15, "lm_head stored bits differ")
    require(standard_layer_bits <= 4.036 and lm_head_bits <= 4.036, "stored-bit cap exceeded")
    manifests = {
        "payload": payload_manifest,
        "row_scale": row_scale_manifest,
        "codebook": codebook_manifest,
    }
    hashes = {
        "payload_manifest_sha256": canonical_sha256(payload_manifest),
        "row_scale_manifest_sha256": canonical_sha256(row_scale_manifest),
        "codebook_manifest_sha256": canonical_sha256(codebook_manifest),
        "weight_manifest_sha256": canonical_sha256(weight_manifest),
        "reconstruction_report_sha256": reconstruction_report["reconstruction_report_sha256"],
        "payload_stream_sha256": payload_stream.hexdigest(),
        "row_scale_stream_sha256": scale_stream.hexdigest(),
        "codebook_stream_sha256": codebook_stream.hexdigest(),
    }
    storage = {
        "total_weight_count": total_weight_count,
        "total_payload_bytes": total_payload_bytes,
        "total_row_scale_bytes": total_scale_bytes,
        "whole_model_codebook_bytes": total_codebook_bytes,
        "standard_decoder_layer_exact_bits_per_weight": standard_layer_bits,
        "lm_head_exact_bits_per_weight": lm_head_bits,
        "maximum_total_stored_bits_per_weight": 4.036,
    }
    return weight_manifest, {
        "manifests": manifests,
        "hashes": hashes,
        "storage": storage,
        "reconstruction": reconstruction_report,
    }


def load_candidate(label: str) -> tuple[nn.Module, dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    embedding = model.model.embed_tokens.weight
    lm_head = model.lm_head.weight
    require(model.config.tie_word_embeddings is True, "source tie_word_embeddings differs")
    require(lm_head is embedding, "source lm_head and embedding Parameter objects differ")
    require(lm_head.data_ptr() == embedding.data_ptr(), "source lm_head and embedding storage differs")
    source_shape = list(embedding.shape)
    source_dtype = str(embedding.dtype)
    source_object_id = id(embedding)
    source_pointer = embedding.data_ptr()
    source_hash = tensor_sha256(embedding)
    require(source_hash == SOURCE_EMBEDDING_SHA256, "source embedding BF16 hash differs")

    cloned = lm_head.detach().clone()
    require(tensor_sha256(cloned) == source_hash, "detached lm_head clone differs before split")
    model.lm_head.weight = nn.Parameter(cloned, requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight is not embedding, "lm_head Parameter object remained tied")
    require(model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head storage remained tied")
    require(tensor_sha256(model.lm_head.weight) == source_hash, "lm_head clone bytes differ after split")
    require(id(model.model.embed_tokens.weight) == source_object_id, "embedding Parameter object changed during split")
    require(model.model.embed_tokens.weight.data_ptr() == source_pointer, "embedding storage changed during split")

    weight_manifest, construction = quantize_model_codebooks(model)
    require(model.config.tie_word_embeddings is False, "candidate config was retied")
    require(model.model.embed_tokens.weight is embedding, "embedding Parameter object changed after codebook construction")
    require(model.model.embed_tokens.weight.data_ptr() == source_pointer, "embedding storage changed after codebook construction")
    require(list(embedding.shape) == source_shape and str(embedding.dtype) == source_dtype, "embedding shape or dtype changed")
    embedding_hash = tensor_sha256(embedding)
    require(embedding_hash == source_hash == SOURCE_EMBEDDING_SHA256, "embedding bytes changed after codebook construction")
    state = state_manifest(model)
    identity = {
        "embedding_bf16_sha256": embedding_hash,
        **construction["hashes"],
        "candidate_state_manifest_sha256": state["candidate_state_manifest_sha256"],
    }
    manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "construction_label": label,
        "source_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "config_tie_word_embeddings_after_split": False,
        "module_counts": {"transformer_linears": 168, "lm_head": 1, "total_codebook_linears": 169},
        "identity": identity,
        "storage": construction["storage"],
        "payload_manifest": construction["manifests"]["payload"],
        "row_scale_manifest": construction["manifests"]["row_scale"],
        "codebook_manifest": construction["manifests"]["codebook"],
        "weight_manifest": weight_manifest,
        "state_manifest": state,
    }
    witness = {
        "construction_label": label,
        "source": {
            "same_parameter_object": True,
            "same_storage_pointer": True,
            "embedding_object_id": source_object_id,
            "embedding_storage_pointer": source_pointer,
            "embedding_shape": source_shape,
            "embedding_dtype": source_dtype,
            "embedding_bf16_sha256": source_hash,
        },
        "post_split": {
            "different_parameter_objects": model.lm_head.weight is not embedding,
            "different_storage_pointers": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
            "embedding_object_preserved": id(embedding) == source_object_id,
            "embedding_storage_preserved": embedding.data_ptr() == source_pointer,
            "pre_codebook_byte_identical": True,
            "config_tie_word_embeddings": bool(model.config.tie_word_embeddings),
        },
        "post_codebook": {
            "embedding_bf16_sha256": embedding_hash,
            "embedding_hash_matches_source": embedding_hash == SOURCE_EMBEDDING_SHA256,
            "lm_head_storage_is_separate": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
            "transformer_codebook_count": 168,
            "lm_head_codebook_count": 1,
        },
    }
    return model, manifest, witness, construction["reconstruction"]


def compare_candidates(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> dict[str, Any]:
    comparisons = {key: candidate_a["identity"][key] == candidate_b["identity"][key] for key in STATE_MATCH_KEYS}
    require(all(comparisons.values()), f"candidate A/B identity differs: {comparisons}")
    require(candidate_a["payload_manifest"] == candidate_b["payload_manifest"], "candidate A/B payload manifests differ")
    require(candidate_a["row_scale_manifest"] == candidate_b["row_scale_manifest"], "candidate A/B row-scale manifests differ")
    require(candidate_a["codebook_manifest"] == candidate_b["codebook_manifest"], "candidate A/B codebook manifests differ")
    require(candidate_a["state_manifest"] == candidate_b["state_manifest"], "candidate A/B state manifests differ")
    return {
        "state_match_keys": list(STATE_MATCH_KEYS),
        "comparisons": comparisons,
        "payload_manifests_match": True,
        "row_scale_manifests_match": True,
        "codebook_manifests_match": True,
        "state_manifests_match": True,
        "all_match": True,
    }


def tokenize_messages(tokenizer: Any, messages: list[dict[str, str]]) -> Tensor:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    require(isinstance(ids, Tensor) and ids.ndim == 2 and ids.shape[0] == 1, "chat tokenization shape differs")
    return ids


def non_evaluator_inputs(tokenizer: Any) -> tuple[list[Tensor], dict[str, Any]]:
    calibration = load_json(CALIBRATION_CONTRACT_PATH)
    frozen = calibration["chat"]["prompts"]
    require(len(frozen) == 4, "frozen calibration prompt count differs")
    tensors: list[Tensor] = []
    records: list[dict[str, Any]] = []
    for item in frozen:
        ids = tokenize_messages(tokenizer, item["messages"])
        token_ids = [int(value) for value in ids[0].tolist()]
        require(token_ids == item["token_ids"], "frozen calibration token IDs differ")
        case_id = f"frozen_calibration_{int(item['ordinal']):02d}"
        tensors.append(ids)
        records.append(
            {
                "case_id": case_id,
                "source": "frozen_calibration_contract",
                "messages": item["messages"],
                "token_ids": token_ids,
                "token_ids_sha256": item["token_ids_sha256"],
                "response_gate_expected_text": EXPECTED_TEXTS[case_id],
                "response_gate_expected_text_sha256": sha256_bytes(EXPECTED_TEXTS[case_id].encode()),
            }
        )
    for index, prompt in enumerate(ADDITIONAL_PROMPTS):
        case_id = f"additional_preflight_{index:02d}"
        messages = [{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": prompt}]
        ids = tokenize_messages(tokenizer, messages)
        token_ids = [int(value) for value in ids[0].tolist()]
        tensors.append(ids)
        records.append(
            {
                "case_id": case_id,
                "source": "engineer_task_additional_prompt",
                "messages": messages,
                "prompt_sha256": sha256_bytes(prompt.encode()),
                "token_ids": token_ids,
                "token_ids_sha256": sha256_bytes(b"".join(value.to_bytes(4, "little") for value in token_ids)),
                "response_gate_expected_text": EXPECTED_TEXTS[case_id],
                "response_gate_expected_text_sha256": sha256_bytes(EXPECTED_TEXTS[case_id].encode()),
            }
        )
    require(len(tensors) == len(records) == 8, "non-evaluator input count differs")
    manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "input_count": 8,
        "official_nine_prompt_inputs_used": False,
        "calibration_contract": file_record(CALIBRATION_CONTRACT_PATH),
        "records": records,
    }
    return tensors, manifest


def generate_sequence(model: nn.Module, tokenizer: Any, input_ids: Tensor, expected_text: str) -> tuple[dict[str, Any], int]:
    prefix = input_ids
    generated: list[int] = []
    prefix_lengths: list[int] = []
    forwards = 0
    with torch.inference_mode():
        for _index in range(MAX_NEW_TOKENS):
            prefix_lengths.append(int(prefix.shape[1]))
            logits = model(input_ids=prefix, use_cache=False).logits[0, -1]
            forwards += 1
            token = int(logits.argmax())
            generated.append(token)
            if token in TERMINATION_TOKEN_IDS:
                break
            prefix = torch.cat([prefix, torch.tensor([[token]], dtype=prefix.dtype)], dim=1)
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    gate = evaluate_response_gate({"decoded_text": decoded, "generated_token_ids": generated}, expected_text)
    return {
        "generated_token_ids": generated,
        "generated_token_ids_sha256": sha256_bytes(b"".join(value.to_bytes(4, "little") for value in generated)),
        "decoded_text": decoded,
        "decoded_text_sha256": sha256_bytes(decoded.encode()),
        "response_gate": gate,
        "prefix_lengths": prefix_lengths,
    }, forwards


class AuditedDynamicPolicy(grouped.ActivationPolicy):
    """Grouped Dynamic Scale32 policy with per-boundary legality evidence."""

    def __init__(self, scales: dict[str, Any]) -> None:
        super().__init__(scales)
        self.events: dict[str, dict[str, Any]] = {}

    def _apply(self, name: str, value: Tensor, records: Tensor, lanes: int) -> Tensor:
        result = grouped.quantize_groups(value, records.to(value.device), lanes)
        groups = int(value.shape[-1] // lanes)
        base = records.reshape(-1)
        if base.numel() == 1:
            base = base.repeat(groups)
        require(base.numel() == groups, f"base group count differs: {name}")
        effective = result.records.reshape(-1, groups)
        for group_index, base_record in enumerate(base.detach().cpu().tolist()):
            base_significand, _base_exponent = unpack_scale32(int(base_record))
            for effective_record in torch.unique(effective[:, group_index]).detach().cpu().tolist():
                effective_significand, _effective_exponent = unpack_scale32(int(effective_record))
                require(effective_significand == base_significand, f"Scale32 significand changed: {name}")

        delta_min = int(result.deltas.min())
        delta_max = int(result.deltas.max())
        require(-24 <= delta_min <= delta_max <= 24, f"dynamic delta range differs: {name}")
        exponent_u8 = torch.bitwise_and(torch.bitwise_right_shift(result.records, 16), 0xFF)
        exponent = torch.where(exponent_u8 >= 128, exponent_u8 - 256, exponent_u8)
        exponent_min = int(exponent.min())
        exponent_max = int(exponent.max())
        require(-24 <= exponent_min <= exponent_max <= 4, f"effective exponent range differs: {name}")
        require(not bool(torch.any(result.payload == -128)), f"reserved -128 produced: {name}")

        self.calls[name] += 1
        self.group_count += result.records.numel()
        self.value_count += result.payload.numel()
        self.delta_min = delta_min if self.delta_min is None else min(self.delta_min, delta_min)
        self.delta_max = delta_max if self.delta_max is None else max(self.delta_max, delta_max)
        payload = result.payload.detach().cpu().contiguous().numpy().astype(np.int8, copy=False)
        record = result.records.detach().cpu().contiguous().numpy().astype("<u4", copy=False)
        self.payload_hash.update(name.encode() + b"\0" + payload.tobytes(order="C"))
        self.record_hash.update(name.encode() + b"\0" + record.tobytes(order="C"))
        event = self.events.setdefault(
            name,
            {
                "group_lanes": lanes,
                "call_count": 0,
                "group_count": 0,
                "value_count": 0,
                "minimum_delta": None,
                "maximum_delta": None,
                "minimum_effective_exponent": None,
                "maximum_effective_exponent": None,
                "saturation_count": 0,
                "clipping_count": 0,
                "reserved_minus_128_count": 0,
                "base_significand_preserved": True,
            },
        )
        require(event["group_lanes"] == lanes, f"event group width changed: {name}")
        event["call_count"] += 1
        event["group_count"] += int(result.records.numel())
        event["value_count"] += int(result.payload.numel())
        event["minimum_delta"] = delta_min if event["minimum_delta"] is None else min(event["minimum_delta"], delta_min)
        event["maximum_delta"] = delta_max if event["maximum_delta"] is None else max(event["maximum_delta"], delta_max)
        event["minimum_effective_exponent"] = exponent_min if event["minimum_effective_exponent"] is None else min(event["minimum_effective_exponent"], exponent_min)
        event["maximum_effective_exponent"] = exponent_max if event["maximum_effective_exponent"] is None else max(event["maximum_effective_exponent"], exponent_max)
        return result.dequantized.to(value.dtype)

    def audited_summary(self, model_forward_count: int) -> dict[str, Any]:
        summary = super().summary(model_forward_count)
        require(summary["named_boundary_count"] == EXPECTED_EVENTS == 312, "312-event closure differs")
        require(-24 <= int(summary["minimum_delta"]) <= int(summary["maximum_delta"]) <= 24, "summary delta range differs")
        require(summary["reserved_minus_128_produced"] is False, "summary reports reserved -128")
        require(set(self.events) == set(summary["calls"]), "audited event set differs")
        require(all(item["saturation_count"] == 0 and item["clipping_count"] == 0 for item in self.events.values()), "preflight clipping or saturation differs")
        summary.update(
            {
                "event_count": len(self.events),
                "all_312_layer_event_names_present": len(self.events) == 312,
                "legal_delta_range": [-24, 24],
                "legal_effective_exponent_range": [-24, 4],
                "saturation_count": 0,
                "clipping_count": 0,
                "transport_metadata_bytes_per_decoder_layer_token": 832,
                "tensor_sidecars_per_decoder_layer_token": 13,
                "events": dict(sorted(self.events.items())),
            }
        )
        return summary


def non_evaluator_quality_and_dynamic(
    candidate_model: nn.Module,
    tokenizer: Any,
    inputs: list[Tensor],
    input_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(len(inputs) == len(input_manifest["records"]) == 8, "quality input count differs")
    require(not MARKER_PATH.exists() and not ATTEMPT_DIR.exists(), "official attempt exists before quality preflight")
    reference_model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    references: list[dict[str, Any]] = []
    try:
        for record, input_ids in zip(input_manifest["records"], inputs, strict=True):
            result, forwards = generate_sequence(
                reference_model,
                tokenizer,
                input_ids,
                record["response_gate_expected_text"],
            )
            references.append({"case_id": record["case_id"], "model_forward_count": forwards, **result})
    finally:
        del reference_model
        gc.collect()

    policy = AuditedDynamicPolicy(load_json(BASE_SCALES_PATH))
    policy.install(candidate_model)
    candidate_results: list[dict[str, Any]] = []
    candidate_forwards = 0
    try:
        for record, input_ids in zip(input_manifest["records"], inputs, strict=True):
            result, forwards = generate_sequence(
                candidate_model,
                tokenizer,
                input_ids,
                record["response_gate_expected_text"],
            )
            candidate_forwards += forwards
            candidate_results.append({"case_id": record["case_id"], "model_forward_count": forwards, **result})
        dynamic_summary = policy.audited_summary(candidate_forwards)
    finally:
        policy.uninstall()

    cases: list[dict[str, Any]] = []
    exact_count = 0
    gate_count = 0
    for reference, candidate in zip(references, candidate_results, strict=True):
        require(reference["case_id"] == candidate["case_id"], "quality case order differs")
        exact = candidate["generated_token_ids"] == reference["generated_token_ids"]
        gate_match = candidate["response_gate"]["status"] == reference["response_gate"]["status"]
        exact_count += int(exact)
        gate_count += int(gate_match)
        cases.append(
            {
                "case_id": reference["case_id"],
                "bf16_generated_token_ids": reference["generated_token_ids"],
                "candidate_generated_token_ids": candidate["generated_token_ids"],
                "exact_bf16_sequence_match": exact,
                "bf16_response_gate_status": reference["response_gate"]["status"],
                "candidate_response_gate_status": candidate["response_gate"]["status"],
                "response_gate_outcome_match": gate_match,
                "bf16_decoded_text_sha256": reference["decoded_text_sha256"],
                "candidate_decoded_text_sha256": candidate["decoded_text_sha256"],
                "bf16_model_forward_count": reference["model_forward_count"],
                "candidate_model_forward_count": candidate["model_forward_count"],
            }
        )

    quality = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "status": "PASS_8_OF_8" if exact_count == gate_count == 8 else "NO_GO_NON_EVALUATOR_QUALITY_GATE",
        "official_nine_prompt_inputs_used": False,
        "bf16_reference_input_count": len(references),
        "candidate_input_count": len(candidate_results),
        "exact_bf16_sequence_match_count": exact_count,
        "response_gate_outcome_match_count": gate_count,
        "required_exact_bf16_sequence_match_count": 8,
        "required_response_gate_outcome_match_count": 8,
        "cases": cases,
    }
    dynamic = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "status": "PASS_ALL_312_DYNAMIC_EVENTS",
        "fresh_model_reconstruction": True,
        "official_nine_prompt_inputs_used": False,
        "official_execution_marker_exists": False,
        "input_count": len(inputs),
        "base_scales": file_record(BASE_SCALES_PATH),
        "selection": "minimum legal exponent delta preserving each frozen base Scale32 significand",
        "rounding": "round_to_nearest_ties_to_even",
        "activation_execution": dynamic_summary,
        "generated_sequences": [
            {
                "case_id": item["case_id"],
                "generated_token_ids_sha256": item["generated_token_ids_sha256"],
                "model_forward_count": item["model_forward_count"],
            }
            for item in candidate_results
        ],
    }
    require(not MARKER_PATH.exists() and not ATTEMPT_DIR.exists(), "official attempt appeared during marker-free preflight")
    return quality, dynamic


def validate_reconstruction(report: dict[str, Any]) -> bool:
    require(report["tensor_count"] == 169 and len(report["tensors"]) == 169, "reconstruction tensor count differs")
    require(report["per_tensor_non_regressing_count"] == sum(bool(item["tensor_codebook_non_regressing"]) for item in report["tensors"]), "reconstruction non-regressing count differs")
    passed = report["all_169_tensors_non_regressing"] is True and report["aggregate_strictly_improved"] is True
    require((report["status"].startswith("PASS_")) == passed, "reconstruction status differs")
    body = {key: value for key, value in report.items() if key not in {"status", "reconstruction_report_sha256"}}
    require(report["reconstruction_report_sha256"] == canonical_sha256(body), "reconstruction report hash differs")
    return passed


def validate_quality(report: dict[str, Any]) -> bool:
    require(report["official_nine_prompt_inputs_used"] is False, "official inputs were used for quality preflight")
    require(report["bf16_reference_input_count"] == report["candidate_input_count"] == len(report["cases"]) == 8, "quality input count differs")
    exact = sum(bool(item["exact_bf16_sequence_match"]) for item in report["cases"])
    gates = sum(bool(item["response_gate_outcome_match"]) for item in report["cases"])
    require(exact == report["exact_bf16_sequence_match_count"], "quality exact-sequence aggregate differs")
    require(gates == report["response_gate_outcome_match_count"], "quality gate aggregate differs")
    passed = exact == gates == 8
    require((report["status"] == "PASS_8_OF_8") == passed, "quality status differs")
    return passed


def validate_dynamic(preflight: dict[str, Any]) -> bool:
    summary = preflight["activation_execution"]
    require(preflight["status"] == "PASS_ALL_312_DYNAMIC_EVENTS", "dynamic preflight status differs")
    require(preflight["official_nine_prompt_inputs_used"] is False, "official inputs were used in dynamic preflight")
    require(preflight["official_execution_marker_exists"] is False, "dynamic preflight claims an execution marker")
    require(summary["event_count"] == summary["named_boundary_count"] == EXPECTED_EVENTS == 312, "dynamic event count differs")
    require(summary["all_312_layer_event_names_present"] is True, "dynamic event closure differs")
    require(-24 <= summary["minimum_delta"] <= summary["maximum_delta"] <= 24, "dynamic delta range differs")
    require(summary["saturation_count"] == 0 and summary["clipping_count"] == 0, "dynamic saturation or clipping differs")
    require(summary["reserved_minus_128_produced"] is False, "dynamic preflight produced reserved -128")
    require(summary["transport_metadata_bytes_per_decoder_layer_token"] <= 832, "transport metadata cap differs")
    require(summary["tensor_sidecars_per_decoder_layer_token"] <= 13, "sidecar count differs")
    return True


def audit_attempt_namespace(expect_root_absent: bool) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for marker in sorted((ROOT / "build").glob("**/execution_started.json")):
        try:
            record = load_json(marker)
        except Exception:
            continue
        if record.get("candidate_id") == CANDIDATE_ID:
            matches.append(file_record(marker))
    attempts = sorted(path.name for path in OUTPUT_DIR.glob("attempt-*") if path.is_dir()) if OUTPUT_DIR.exists() else []
    if expect_root_absent:
        require(not OUTPUT_DIR.exists(), "V3 namespace existed before preparation")
    require(not matches, "a prior V3 execution marker exists")
    require(not attempts, f"V3 attempt namespace is already consumed: {attempts}")
    return {
        "candidate_id": CANDIDATE_ID,
        "root_existed_at_audit": OUTPUT_DIR.exists(),
        "matching_execution_markers": matches,
        "attempt_directories": attempts,
        "authorization_consumed": False,
    }


def recover_nonqualifying_preparation() -> dict[str, Any]:
    if not OUTPUT_DIR.exists():
        return {"fresh_namespace": True, "recovered": False, "archived_artifacts": []}
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "cannot recover a consumed attempt namespace")
    complete = PREATTEMPT_DIR.is_dir() and CONTRACT_PATH.is_file() and (READY_PATH.is_file() or NO_GO_PATH.is_file())
    require(not complete, "pre-attempt closure already exists; use verify")
    candidates = [path for path in OUTPUT_DIR.iterdir() if path.name != "preflight-failures"]
    require(candidates, "existing V3 namespace has ambiguous provenance")
    archive_root = OUTPUT_DIR / "preflight-failures"
    archive_root.mkdir(exist_ok=True)
    ordinal = 1
    while (archive_root / f"failure-{ordinal:04d}").exists():
        ordinal += 1
    archive = archive_root / f"failure-{ordinal:04d}"
    archive.mkdir()
    archived: list[dict[str, Any]] = []
    for path in candidates:
        destination = archive / path.name
        os.replace(path, destination)
        if destination.is_file():
            destination.chmod(0o444)
            archived.append(file_record(destination))
        else:
            for item in sorted(entry for entry in destination.rglob("*") if entry.is_file()):
                item.chmod(0o444)
                archived.append(file_record(item))
    return {"fresh_namespace": False, "recovered": True, "archived_artifacts": archived}


def artifact_records() -> dict[str, Any]:
    paths = {
        "engineer_task": TASK_PATH,
        "engineer_task_companion": TASK_COMPANION,
        "planning_requirements": PLAN_PATH,
        "planning_requirements_companion": PLAN_COMPANION,
        "source_contract": SOURCE_CONTRACT_PATH,
        "calibration_contract": CALIBRATION_CONTRACT_PATH,
        "calibration_report": CALIBRATION_REPORT_PATH,
        "base_scales": BASE_SCALES_PATH,
        "image_manifest": IMAGE_DIR / "manifest.json",
        "official_matrix_hash_only": MATRIX_PATH,
        "runner_source": RUNNER_PATH,
        "reviewed_grouped_dynamic_runner": GROUPED_RUNNER_PATH,
        "response_gate_source": ROOT / "tools/qwen_instruct_response_gate.py",
        "identity_helper_source": ROOT / "tools/qwen_instruct_option_b.py",
        "quality_contracts": ROOT / "tools/ace2_quality_contracts.py",
        "non_evaluator_inputs": INPUT_MANIFEST_PATH,
        "candidate_a_manifest": CANDIDATE_A_PATH,
        "candidate_b_manifest": CANDIDATE_B_PATH,
        "alias_separation_witness": ALIAS_WITNESS_PATH,
        "model_only_reconstruction": RECONSTRUCTION_PATH,
        "non_evaluator_quality": QUALITY_PATH,
        "dynamic_preflight": DYNAMIC_PATH,
        "runner_self_test": SELF_TEST_PATH,
        "preparation_chronology": CHRONOLOGY_PATH,
    }
    if RECOVERY_PATH.is_file():
        paths["preparation_recovery"] = RECOVERY_PATH
    failure_root = OUTPUT_DIR / "preflight-failures"
    if failure_root.is_dir():
        for index, path in enumerate(sorted(item for item in failure_root.rglob("*") if item.is_file())):
            paths[f"nonqualifying_preflight_failure_{index:02d}"] = path
    return {name: file_record(path) for name, path in paths.items()}


def build_contract(
    environment: dict[str, Any],
    source_identity: dict[str, Any],
    namespace_audit: dict[str, Any],
    state_comparison: dict[str, Any],
    gate_passes: dict[str, bool],
) -> dict[str, Any]:
    task = load_json(TASK_PATH)
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "Engineer authority differs")
    require(task["fresh_attempt_namespace"]["attempt_budget"] == 1, "attempt budget differs")
    require(task["fresh_attempt_namespace"]["attempts_consumed_at_task_freeze"] == 0, "attempt was consumed at freeze")
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    artifacts = artifact_records()
    format_policy = policy_and_format()
    all_pass = all(gate_passes.values())
    contract = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "task_id": TASK_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "status": "PREATTEMPT_GATES_PASS" if all_pass else "PREATTEMPT_NO_GO",
        "authority": {
            "engineer_task": artifacts["engineer_task"],
            "planning_requirements": artifacts["planning_requirements"],
            "attempt_id": "attempt-0001",
            "exact_authorized_candidate_attempts": 1,
            "attempts_consumed": 0,
            "selected_policy_id": None,
        },
        "source_model": source_identity,
        "environment": environment,
        "format_and_policy": format_policy,
        "construction": {
            "source_tied_then_runtime_split_lm_head": True,
            "source_embedding_bf16_sha256": SOURCE_EMBEDDING_SHA256,
            "embedding_bytes_preserved": True,
            "transformer_codebook_linears": 168,
            "separated_lm_head_codebook_linears": 1,
            "candidate_a_and_b_manifests_match": state_comparison,
        },
        "pre_attempt_gates": {
            "model_only_reconstruction": {"passed": gate_passes["model_only_reconstruction"], "artifact": artifacts["model_only_reconstruction"]},
            "non_evaluator_quality_8_of_8": {"passed": gate_passes["non_evaluator_quality_8_of_8"], "artifact": artifacts["non_evaluator_quality"]},
            "dynamic_312_event_closure": {"passed": gate_passes["dynamic_312_event_closure"], "artifact": artifacts["dynamic_preflight"]},
            "two_build_and_alias_closure": {"passed": gate_passes["two_build_and_alias_closure"], "artifact": artifacts["alias_separation_witness"]},
            "all_pass": all_pass,
        },
        "official_matrix_binding": {
            "artifact": artifacts["official_matrix_hash_only"],
            "sha256": MATRIX_SHA256,
            "prompt_count": 9,
            "official_input_content_accessed_during_preflight": False,
        },
        "official_execution": {
            "mode_exposed_by_runner": False,
            "attempt_directory_exists": False,
            "execution_marker_exists": False,
            "later_increment_required_to_enable": True,
        },
        "namespace_audit": namespace_audit,
        "artifacts": artifacts,
        "claim_boundary": {
            "attempts_consumed": 0,
            "numerical_conclusion": None,
            "selected_policy_id": None,
            "rtl_or_ppa_claimed": False,
            "current_stage_remains": "specification",
        },
    }
    return {"contract": contract, "contract_sha256": canonical_sha256(contract)}


def verify_bound_record(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    require(path.is_file(), f"bound artifact is missing: {record['path']}")
    require(path.stat().st_size == record["bytes"], f"bound artifact size differs: {record['path']}")
    require_digest(sha256_file(path), record["sha256"], f"bound artifact hash differs: {record['path']}")


def load_verified_closure() -> dict[str, Any]:
    require_project_python()
    require(CONTRACT_PATH.is_file(), "predeclared contract is missing")
    require(READY_PATH.is_file() != NO_GO_PATH.is_file(), "exactly one pre-attempt terminal record is required")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "predeclared contract wrapper differs")
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "predeclared contract canonical hash differs")
    contract = wrapper["contract"]
    require(contract["candidate_id"] == CANDIDATE_ID and contract["task_id"] == TASK_ID, "predeclared authority differs")
    require(contract["artifacts"]["runner_source"]["sha256"] == sha256_file(RUNNER_PATH), "runner changed after freeze")
    for record in contract["artifacts"].values():
        verify_bound_record(record)
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(verify_source_snapshot() == contract["source_model"], "source snapshot identity differs")
    packages = {**verify_versions(), "numpy": importlib.metadata.version("numpy")}
    require(packages == contract["environment"]["packages"], "package versions differ")
    require(codebook_self_test() == load_json(SELF_TEST_PATH), "runner self-test result differs")
    candidate_a = load_json(CANDIDATE_A_PATH)
    candidate_b = load_json(CANDIDATE_B_PATH)
    comparison = compare_candidates(candidate_a, candidate_b)
    witness = load_json(ALIAS_WITNESS_PATH)
    require(witness["state_comparison"] == comparison, "alias witness state comparison differs")
    gate_passes = {
        "model_only_reconstruction": validate_reconstruction(load_json(RECONSTRUCTION_PATH)),
        "non_evaluator_quality_8_of_8": validate_quality(load_json(QUALITY_PATH)),
        "dynamic_312_event_closure": validate_dynamic(load_json(DYNAMIC_PATH)),
        "two_build_and_alias_closure": comparison["all_match"] and witness["embedding_preserved_in_both"] and witness["lm_head_separate_in_both"],
    }
    require(contract["pre_attempt_gates"]["all_pass"] == all(gate_passes.values()), "contract aggregate gate status differs")
    for name, passed in gate_passes.items():
        require(contract["pre_attempt_gates"][name]["passed"] == passed, f"contract gate differs: {name}")
    terminal_path = READY_PATH if READY_PATH.is_file() else NO_GO_PATH
    terminal = load_json(terminal_path)
    require(terminal["candidate_id"] == CANDIDATE_ID, "terminal pre-attempt candidate differs")
    require(terminal["contract"]["sha256"] == sha256_file(CONTRACT_PATH), "terminal contract file hash differs")
    require(terminal["contract_sha256"] == wrapper["contract_sha256"], "terminal contract canonical hash differs")
    require(terminal["bound_artifacts"] == contract["artifacts"], "terminal artifact snapshot differs")
    require(terminal["official_execution_marker_exists"] is False and terminal["attempts_consumed"] == 0, "terminal record claims an official attempt")
    if terminal_path == READY_PATH:
        require(terminal["status"] == "PREATTEMPT_READY" and all(gate_passes.values()), "ready record exists without all gates passing")
    else:
        require(terminal["status"] == "PREATTEMPT_NO_GO" and not all(gate_passes.values()), "no-go record does not preserve a failed gate")
    audit_attempt_namespace(expect_root_absent=False)
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "official attempt exists after marker-free closure")
    return {
        "status": terminal["status"],
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": wrapper["contract_sha256"],
        "gate_passes": gate_passes,
        "attempts_consumed": 0,
        "official_execution_mode_exposed": False,
    }


def prepare() -> dict[str, Any]:
    environment = require_project_python()
    recovery = recover_nonqualifying_preparation()
    namespace_audit = audit_attempt_namespace(expect_root_absent=recovery["fresh_namespace"])
    namespace_audit["preparation_recovery"] = recovery
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    verify_source_contract()
    source_identity = verify_source_snapshot()
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    probe = OUTPUT_DIR / ".preflight-write-probe"
    fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, b"preflight\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    require(probe.read_bytes() == b"preflight\n", "output-path readback differs")
    probe.unlink()

    chronology: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "preparation_started_at_utc": utc_now(),
        "namespace_audit": namespace_audit,
        "official_matrix_content_accessed": False,
        "output_path_preflight": {"exclusive_create": True, "readback": True, "collision_behavior": "O_EXCL", "execution_marker_created": False},
    }
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    inputs, input_manifest = non_evaluator_inputs(tokenizer)
    self_test = codebook_self_test()

    chronology["candidate_a_started_at_utc"] = utc_now()
    candidate_a, manifest_a, witness_a, reconstruction_a = load_candidate("candidate_a_independent_construction")
    chronology["candidate_a_complete_at_utc"] = utc_now()
    del candidate_a
    gc.collect()

    chronology["candidate_b_started_at_utc"] = utc_now()
    candidate_b, manifest_b, witness_b, reconstruction_b = load_candidate("candidate_b_independent_quality_and_dynamic_preflight")
    state_comparison = compare_candidates(manifest_a, manifest_b)
    require(reconstruction_a == reconstruction_b, "candidate A/B reconstruction reports differ")
    chronology["candidate_b_constructed_at_utc"] = utc_now()
    quality, dynamic = non_evaluator_quality_and_dynamic(candidate_b, tokenizer, inputs, input_manifest)
    chronology["candidate_b_quality_and_dynamic_complete_at_utc"] = utc_now()
    del candidate_b
    gc.collect()

    gate_passes = {
        "model_only_reconstruction": validate_reconstruction(reconstruction_b),
        "non_evaluator_quality_8_of_8": validate_quality(quality),
        "dynamic_312_event_closure": validate_dynamic(dynamic),
        "two_build_and_alias_closure": state_comparison["all_match"]
        and witness_a["post_codebook"]["embedding_hash_matches_source"]
        and witness_b["post_codebook"]["embedding_hash_matches_source"]
        and witness_a["post_codebook"]["lm_head_storage_is_separate"]
        and witness_b["post_codebook"]["lm_head_storage_is_separate"],
    }
    require(not MARKER_PATH.exists() and not ATTEMPT_DIR.exists(), "official attempt appeared before artifact freeze")

    PREATTEMPT_DIR.mkdir(parents=False, exist_ok=False)
    write_json(INPUT_MANIFEST_PATH, input_manifest)
    write_json(CANDIDATE_A_PATH, manifest_a)
    write_json(CANDIDATE_B_PATH, manifest_b)
    write_json(
        ALIAS_WITNESS_PATH,
        {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "candidate_a": witness_a,
            "candidate_b": witness_b,
            "state_comparison": state_comparison,
            "embedding_preserved_in_both": witness_a["post_codebook"]["embedding_hash_matches_source"] and witness_b["post_codebook"]["embedding_hash_matches_source"],
            "lm_head_separate_in_both": witness_a["post_codebook"]["lm_head_storage_is_separate"] and witness_b["post_codebook"]["lm_head_storage_is_separate"],
        },
    )
    write_json(RECONSTRUCTION_PATH, reconstruction_b)
    write_json(QUALITY_PATH, quality)
    write_json(DYNAMIC_PATH, dynamic)
    write_json(SELF_TEST_PATH, self_test)
    if recovery["recovered"]:
        write_json(RECOVERY_PATH, {"schema_version": 1, "candidate_id": CANDIDATE_ID, **recovery})
    chronology["pre_attempt_artifacts_frozen_at_utc"] = utc_now()
    chronology["gate_passes"] = gate_passes
    write_json(CHRONOLOGY_PATH, chronology)

    wrapper = build_contract(environment, source_identity, namespace_audit, state_comparison, gate_passes)
    write_json(CONTRACT_PATH, wrapper)
    all_pass = all(gate_passes.values())
    terminal_path = READY_PATH if all_pass else NO_GO_PATH
    terminal = {
        "schema_version": 1,
        "status": "PREATTEMPT_READY" if all_pass else "PREATTEMPT_NO_GO",
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "contract": file_record(CONTRACT_PATH),
        "contract_sha256": wrapper["contract_sha256"],
        "bound_artifacts": wrapper["contract"]["artifacts"],
        "gate_passes": gate_passes,
        "attempt_budget": 1,
        "attempts_consumed": 0,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": False,
        "disposition": (
            "all marker-free pre-attempt artifacts verified; a later reviewed increment is required before official execution can be enabled"
            if all_pass
            else "preserve no-go evidence, leave the sole official attempt unconsumed, keep selected_policy_id null, and submit for Fresh Review"
        ),
    }
    atomic_write_json(terminal_path, terminal)
    verified = load_verified_closure()
    for path in list(PREATTEMPT_DIR.iterdir()) + [CONTRACT_PATH, terminal_path]:
        if path.is_file():
            path.chmod(0o444)
    return {
        **verified,
        "candidate_state_manifest_sha256": manifest_b["identity"]["candidate_state_manifest_sha256"],
        "exact_sequence_matches": quality["exact_bf16_sequence_match_count"],
        "gate_outcome_matches": quality["response_gate_outcome_match_count"],
        "event_count": dynamic["activation_execution"]["event_count"],
        "minimum_delta": dynamic["activation_execution"]["minimum_delta"],
        "maximum_delta": dynamic["activation_execution"]["maximum_delta"],
    }


def record_preparation_failure(exc: Exception) -> None:
    if MARKER_PATH.exists():
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failure = OUTPUT_DIR / f"preparation_failure_{time.time_ns()}.json"
    write_json(
        failure,
        {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "classification": "PREATTEMPT_FAILURE",
            "failure_taxonomy": "PREATTEMPT_EXECUTION_FAILURE",
            "root_cause_hypothesis": f"runner raised {type(exc).__name__}: {exc}",
            "regression": "repair only the isolated marker-free runner and rerun pre-attempt preparation; official authorization remains unconsumed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "traceback": traceback.format_exc(),
            "attempt_authorization_consumed": False,
            "official_execution_marker_exists": False,
            "official_execution_mode_exposed": False,
            "failed_at_utc": utc_now(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "prepare", "verify"))
    args = parser.parse_args()
    if args.command == "self-test":
        print("ACE2_OPTION_B_CODEBOOK_V3_SELF_TEST_PASS " + json.dumps(codebook_self_test(), sort_keys=True))
        return 0
    if args.command == "prepare":
        try:
            result = prepare()
        except Exception as exc:
            record_preparation_failure(exc)
            raise
        print("ACE2_OPTION_B_CODEBOOK_V3_PREFLIGHT_COMPLETE " + json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PREATTEMPT_READY" else 2
    result = load_verified_closure()
    print("ACE2_OPTION_B_CODEBOOK_V3_PREFLIGHT_VERIFIED " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
