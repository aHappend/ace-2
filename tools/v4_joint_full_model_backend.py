#!/usr/bin/env python3
"""Scalable exact-dyadic full-model backend for the V4 joint W4 policy."""

from __future__ import annotations

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

import run_option_b_alias_safe_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_stage1 as reference
import run_option_b_alias_safe_tensor_codebook_w4_grouped_dynamic_scale32_stage1 as v3
import run_option_b_grouped_scale32_stage1 as grouped
from ace2_quality_contracts import (
    SCALE32_ALL_ZERO_RECORD,
    SCALE32_EXPONENT_MAX,
    SCALE32_EXPONENT_MIN,
    SCALE32_SIGNIFICAND_MAX,
    SCALE32_SIGNIFICAND_MIN,
    ceil_scale32_from_ratio,
    pack_scale32,
    scale32_ratio,
    unpack_scale32,
)
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    IMAGE_DIR,
    ROOT,
    SNAPSHOT,
    canonical_bytes,
    file_record,
    load_json,
    require,
    sha256_file,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
    write_json,
)
from run_all_transformer_per32_stage1 import utc_now


CANDIDATE_ID = reference.CANDIDATE_ID
MISSION_ID = reference.MISSION_ID
TASK_ID = reference.TASK_ID
TASK_PATH = reference.TASK_PATH
TASK_COMPANION = reference.TASK_COMPANION
PLAN_PATH = reference.PLAN_PATH
PLAN_COMPANION = reference.PLAN_COMPANION
RUNNER_PATH = reference.RUNNER_PATH
BACKEND_PATH = Path(__file__).resolve()
V3_RUNNER_PATH = Path(v3.__file__).resolve()
GROUPED_RUNNER_PATH = ROOT / "tools/run_option_b_grouped_scale32_stage1.py"
SOURCE_CONTRACT_PATH = ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
CALIBRATION_CONTRACT_PATH = CALIBRATION_DIR / "calibration_contract.json"
CALIBRATION_REPORT_PATH = CALIBRATION_DIR / "calibration_report.json"
BASE_SCALES_PATH = CALIBRATION_DIR / "derived_scales.json"
MATRIX_PATH = ROOT / "build/option-b-v2-weight-policy-20260806/full_sequence_discrimination.json"
MATRIX_SHA256 = "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
SOURCE_EMBEDDING_SHA256 = "4e96b0df6d274768cbb7e72404011853d23349999b658dc2f4dfb3c431ea223f"
V3_RECONSTRUCTION_PATH = ROOT / "build/stage1-option-b-alias-safe-tensor-codebook-w4-grouped-dynamic-scale32-w4a8-v3/preattempt/model_only_reconstruction.json"
V3_RECONSTRUCTION_SHA256 = "c39f0637e4433d924d4f03812fe924f762a748ec42e59c2fcaf44aa1570e9ba1"
V3_AGGREGATE_BF16_SSE = 2987.0298056936317

OUTPUT_DIR = reference.OFFICIAL_ROOT
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

Q4_4_MIN = reference.Q4_4_MIN
Q4_4_MAX = reference.Q4_4_MAX
CODEBOOK_ENTRIES = reference.CODEBOOK_ENTRIES
OUTER_ITERATIONS = reference.OUTER_ITERATIONS
TORCH_THREADS = 32
CHUNK_VALUES = 1 << 20
EXPECTED_EVENTS = grouped.LAYERS * len(grouped.EVENT_FAMILIES)
STATE_MATCH_KEYS = (
    "embedding_bf16_sha256",
    "payload_manifest_sha256",
    "row_scale_manifest_sha256",
    "codebook_manifest_sha256",
    "reconstruction_manifest_sha256",
    "iteration_trace_manifest_sha256",
    "weight_manifest_sha256",
    "candidate_state_manifest_sha256",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


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
        "backend": "chunked exact BF16 dyadic lattice with int64 reductions and Python-integer final decisions",
    }


def verify_companion(path: Path, companion: Path) -> None:
    expected = f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}"
    require(companion.is_file(), f"checksum companion is missing: {companion}")
    require(companion.read_text(encoding="utf-8").strip() == expected, f"checksum companion differs: {path}")


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


def _configure_v3_quality_helpers() -> None:
    """Reuse the reviewed marker-free quality harness under V4-bound globals."""
    v3.CANDIDATE_ID = CANDIDATE_ID
    v3.OUTPUT_DIR = OUTPUT_DIR
    v3.PREATTEMPT_DIR = PREATTEMPT_DIR
    v3.ATTEMPT_DIR = ATTEMPT_DIR
    v3.MARKER_PATH = MARKER_PATH
    v3.TORCH_THREADS = TORCH_THREADS


def _row_chunks(rows: int, width: int):
    rows_per_chunk = max(1, CHUNK_VALUES // width)
    for start in range(0, rows, rows_per_chunk):
        yield start, min(rows, start + rows_per_chunk)


def _decode_bf16(source: Tensor) -> tuple[Tensor, Tensor]:
    """Return exact signed significands and binary exponents for BF16 values."""
    require(source.dtype == torch.bfloat16, "source tensor is not BF16")
    bits = source.contiguous().view(torch.uint16).to(torch.int32)
    sign = torch.where((bits & 0x8000) != 0, -1, 1).to(torch.int64)
    exponent = (bits >> 7) & 0xFF
    fraction = bits & 0x7F
    nonzero = (exponent != 0) | (fraction != 0)
    require(not bool(torch.any(exponent == 0xFF)), "source BF16 tensor contains a non-finite value")
    significand = torch.where(exponent == 0, fraction, fraction + 128).to(torch.int64) * sign
    significand = torch.where(nonzero, significand, torch.zeros_like(significand))
    shift = torch.where(exponent == 0, torch.full_like(exponent, -133), exponent - 134).to(torch.int64)
    return significand, shift


def _tensor_shift_range(source: Tensor) -> tuple[int, int, int]:
    minimum: int | None = None
    maximum: int | None = None
    subnormals = 0
    rows, width = source.shape
    for start, stop in _row_chunks(rows, width):
        chunk = source[start:stop]
        bits = chunk.contiguous().view(torch.uint16).to(torch.int32)
        exponent = (bits >> 7) & 0xFF
        fraction = bits & 0x7F
        nonzero = (exponent != 0) | (fraction != 0)
        require(not bool(torch.any(exponent == 0xFF)), "source BF16 tensor contains a non-finite value")
        subnormals += int(torch.sum((exponent == 0) & (fraction != 0)))
        if bool(torch.any(nonzero)):
            shifts = torch.where(exponent == 0, torch.full_like(exponent, -133), exponent - 134)[nonzero]
            lo = int(shifts.min())
            hi = int(shifts.max())
            minimum = lo if minimum is None else min(minimum, lo)
            maximum = hi if maximum is None else max(maximum, hi)
    require(minimum is not None and maximum is not None, "entire BF16 tensor is zero")
    return minimum, maximum, subnormals


def _records_components(records: Tensor) -> tuple[Tensor, Tensor]:
    values = records.to(torch.int64)
    significands = values & 0xFFFF
    exponent_u8 = (values >> 16) & 0xFF
    exponents = torch.where(exponent_u8 >= 128, exponent_u8 - 256, exponent_u8)
    require(bool(torch.all((significands >= SCALE32_SIGNIFICAND_MIN) & (significands <= SCALE32_SIGNIFICAND_MAX))), "Scale32 significand range differs")
    require(bool(torch.all((exponents >= SCALE32_EXPONENT_MIN) & (exponents <= SCALE32_EXPONENT_MAX))), "Scale32 exponent range differs")
    return significands, exponents


def _compare_dyadic(
    left_numerator: int,
    left_denominator: int,
    left_shift: int,
    right_numerator: int,
    right_denominator: int,
    right_shift: int,
) -> int:
    require(left_denominator > 0 and right_denominator > 0, "dyadic denominator is not positive")
    shift = left_shift - right_shift
    left = left_numerator * right_denominator
    right = right_numerator * left_denominator
    if shift >= 0:
        left <<= shift
    else:
        right <<= -shift
    return (left > right) - (left < right)


def _compare_record_to_dyadic(record: int, numerator: int, denominator: int, shift: int) -> int:
    significand, exponent = unpack_scale32(record)
    return _compare_dyadic(significand, 1, exponent - 15, numerator, denominator, shift)


def _ceil_scale32_dyadic(numerator: int, denominator: int, shift: int) -> int:
    require(numerator >= 0 and denominator > 0, "Scale32 target ratio differs")
    if shift >= 0:
        numerator <<= shift
    else:
        denominator <<= -shift
    return int(ceil_scale32_from_ratio(numerator, denominator))


def _record_scale_key(record: int) -> tuple[int, int, int]:
    significand, exponent = unpack_scale32(record)
    return exponent, significand, record


def _initial_row_records(source: Tensor, base_shift: int) -> tuple[Tensor, int]:
    rows, width = source.shape
    records = torch.empty(rows, dtype=torch.int64)
    zero_rows = 0
    for start, stop in _row_chunks(rows, width):
        significands, shifts = _decode_bf16(source[start:stop])
        delta = torch.where(significands == 0, torch.zeros_like(shifts), shifts - base_shift)
        require(bool(torch.all((delta >= 0) & (delta <= 54))), "BF16 tensor lattice span exceeds int64 construction")
        exact = torch.bitwise_left_shift(significands, delta)
        maxima = torch.abs(exact).amax(dim=1).tolist()
        selected: list[int] = []
        for maximum in maxima:
            if int(maximum) == 0:
                selected.append(SCALE32_ALL_ZERO_RECORD)
                zero_rows += 1
            else:
                selected.append(_ceil_scale32_dyadic(int(maximum), 7, base_shift))
        records[start:stop] = torch.tensor(selected, dtype=torch.int64)
    return records, zero_rows


def _round_ratio_tensor_even(numerator: Tensor, denominator: Tensor) -> Tensor:
    require(numerator.dtype == denominator.dtype == torch.int64, "exact rounding tensor dtype differs")
    require(bool(torch.all(denominator > 0)), "exact rounding denominator is not positive")
    magnitude = torch.abs(numerator)
    quotient = torch.div(magnitude, denominator, rounding_mode="floor")
    remainder = torch.remainder(magnitude, denominator)
    require(int(denominator.max()) < (1 << 62), "exact rounding denominator exceeds safe doubled range")
    doubled = remainder * 2
    increment = (doubled > denominator) | ((doubled == denominator) & ((quotient & 1) != 0))
    rounded = quotient + increment.to(torch.int64)
    return torch.where(numerator < 0, -rounded, rounded)


def _initial_histogram(source: Tensor, records: Tensor, base_shift: int) -> tuple[int, ...]:
    rows, width = source.shape
    histogram = torch.zeros(256, dtype=torch.int64)
    for start, stop in _row_chunks(rows, width):
        significands, shifts = _decode_bf16(source[start:stop])
        delta = torch.where(significands == 0, torch.zeros_like(shifts), shifts - base_shift)
        require(bool(torch.all((delta >= 0) & (delta <= 54))), "BF16 tensor lattice span exceeds int64 normalization")
        exact = torch.bitwise_left_shift(significands, delta)
        scale_significands, scale_exponents = _records_components(records[start:stop])
        ratio_shift = base_shift + 19 - scale_exponents
        positive = torch.clamp(ratio_shift, min=0)
        negative = torch.clamp(-ratio_shift, min=0)
        require(int(positive.max()) <= 54 and int(negative.max()) <= 46, "Q4.4 normalization shift exceeds int64 range")
        numerators = torch.bitwise_left_shift(exact, positive[:, None])
        denominators = torch.bitwise_left_shift(scale_significands, negative)[:, None].expand_as(numerators)
        normalized = _round_ratio_tensor_even(numerators, denominators)
        require(bool(torch.all((normalized >= Q4_4_MIN) & (normalized <= Q4_4_MAX))), "normalized BF16 weight escaped signed Q4.4")
        histogram += torch.bincount((normalized.reshape(-1) - Q4_4_MIN), minlength=256)
    return tuple(int(value) for value in histogram.tolist())


def _exact_aggregates(
    source: Tensor,
    records: Tensor,
    assignments: Tensor,
    shift_min: int,
    shift_max: int,
) -> dict[str, Any]:
    rows, width = source.shape
    l_min = shift_min + SCALE32_EXPONENT_MIN - 19
    l_max = shift_max + SCALE32_EXPONENT_MAX - 19
    q_min = 2 * SCALE32_EXPONENT_MIN - 38
    q_max = 2 * SCALE32_EXPONENT_MAX - 38
    objective_exponent = min(l_min, q_min)
    l_range = l_max - l_min + 1
    q_range = q_max - q_min + 1
    l_bins = torch.zeros(CODEBOOK_ENTRIES * l_range, dtype=torch.int64)
    q_bins = torch.zeros(CODEBOOK_ENTRIES * q_range, dtype=torch.int64)
    counts = torch.zeros(CODEBOOK_ENTRIES, dtype=torch.int64)

    for start, stop in _row_chunks(rows, width):
        chunk = source[start:stop]
        indices = assignments[start:stop].to(torch.int64)
        weight_significands, weight_shifts = _decode_bf16(chunk)
        weight_shifts = torch.where(
            weight_significands == 0,
            torch.full_like(weight_shifts, shift_min),
            weight_shifts,
        )
        scale_significands, scale_exponents = _records_components(records[start:stop])
        l_exponents = weight_shifts + scale_exponents[:, None] - 19
        l_coefficients = weight_significands * scale_significands[:, None]
        l_keys = indices * l_range + (l_exponents - l_min)
        l_bins.scatter_add_(0, l_keys.reshape(-1), l_coefficients.reshape(-1))

        q_exponents = 2 * scale_exponents - 38
        q_coefficients = torch.square(scale_significands)
        q_keys = indices * q_range + (q_exponents[:, None] - q_min)
        q_values = q_coefficients[:, None].expand_as(indices)
        q_bins.scatter_add_(0, q_keys.reshape(-1), q_values.reshape(-1))
        counts += torch.bincount(indices.reshape(-1), minlength=CODEBOOK_ENTRIES)

    linear_scaled: list[int] = []
    quadratic_scaled: list[int] = []
    for cluster in range(CODEBOOK_ENTRIES):
        linear = 0
        for offset, coefficient in enumerate(l_bins[cluster * l_range : (cluster + 1) * l_range].tolist()):
            linear += int(coefficient) << (l_min + offset - objective_exponent)
        quadratic = 0
        for offset, coefficient in enumerate(q_bins[cluster * q_range : (cluster + 1) * q_range].tolist()):
            quadratic += int(coefficient) << (q_min + offset - objective_exponent)
        linear_scaled.append(linear)
        quadratic_scaled.append(quadratic)
    return {
        "objective_binary_exponent": objective_exponent,
        "linear_scaled": linear_scaled,
        "quadratic_scaled": quadratic_scaled,
        "cluster_counts": [int(value) for value in counts.tolist()],
    }


def _objective(aggregates: dict[str, Any], codepoints: Sequence[int]) -> int:
    points = reference.validate_codebook(codepoints)
    return sum(
        -2 * point * linear + point * point * quadratic
        for point, linear, quadratic in zip(
            points,
            aggregates["linear_scaled"],
            aggregates["quadratic_scaled"],
            strict=True,
        )
    )


def _assign_and_aggregate(
    source: Tensor,
    records: Tensor,
    codepoints: Sequence[int],
    shift_min: int,
    shift_max: int,
) -> tuple[Tensor, dict[str, Any]]:
    points = reference.validate_codebook(codepoints)
    rows, width = source.shape
    assignments = torch.empty(source.shape, dtype=torch.uint8)
    midpoint_sums = torch.tensor(
        [left + right for left, right in zip(points, points[1:])],
        dtype=torch.float64,
    )
    for start, stop in _row_chunks(rows, width):
        scale_significands, scale_exponents = _records_components(records[start:stop])
        scales = torch.ldexp(scale_significands.to(torch.float64), scale_exponents.to(torch.int32) - 15)
        boundaries = scales[:, None] * (midpoint_sums[None, :] / 32.0)
        values = source[start:stop].to(torch.float64).contiguous()
        selected = torch.searchsorted(boundaries.contiguous(), values, right=False)
        require(bool(torch.all((selected >= 0) & (selected < CODEBOOK_ENTRIES))), "codebook assignment escaped four bits")
        assignments[start:stop] = selected.to(torch.uint8)
    aggregates = _exact_aggregates(source, records, assignments, shift_min, shift_max)
    return assignments, aggregates


def _row_statistics(
    source: Tensor,
    assignments: Tensor,
    codepoints: Sequence[int],
    base_shift: int,
) -> tuple[list[int], list[int], list[int]]:
    points = torch.tensor(reference.validate_codebook(codepoints), dtype=torch.int64)
    rows, width = source.shape
    weight_codepoint: list[int] = []
    codepoint_squared: list[int] = []
    absolute_maximum: list[int] = []
    for start, stop in _row_chunks(rows, width):
        significands, shifts = _decode_bf16(source[start:stop])
        delta = torch.where(significands == 0, torch.zeros_like(shifts), shifts - base_shift)
        require(bool(torch.all((delta >= 0) & (delta <= 54))), "BF16 tensor lattice span exceeds int64 row update")
        exact = torch.bitwise_left_shift(significands, delta)
        selected = points[assignments[start:stop].to(torch.int64)]
        maximum = int(torch.abs(exact).max())
        require(maximum * 127 * width < (1 << 63), "exact row-update reduction may overflow int64")
        weight_codepoint.extend(int(value) for value in torch.sum(exact * selected, dim=1).tolist())
        codepoint_squared.extend(int(value) for value in torch.sum(selected * selected, dim=1).tolist())
        absolute_maximum.extend(int(value) for value in torch.abs(exact).amax(dim=1).tolist())
    return weight_codepoint, codepoint_squared, absolute_maximum


def _row_objective(
    record: int,
    weight_codepoint: int,
    codepoint_squared: int,
    base_shift: int,
    objective_exponent: int,
) -> int:
    significand, exponent = unpack_scale32(record)
    linear_shift = base_shift + exponent - 19 - objective_exponent
    quadratic_shift = 2 * exponent - 38 - objective_exponent
    require(linear_shift >= 0 and quadratic_shift >= 0, "row objective exponent differs")
    return (
        (-2 * significand * weight_codepoint << linear_shift)
        + (significand * significand * codepoint_squared << quadratic_shift)
    )


def _candidate_row_records(
    weight_codepoint: int,
    codepoint_squared: int,
    absolute_maximum: int,
    base_shift: int,
    prior_record: int,
) -> tuple[list[int], dict[str, Any]]:
    if absolute_maximum == 0:
        return [SCALE32_ALL_ZERO_RECORD], {"all_zero": True, "candidate_count": 1}
    require(codepoint_squared > 0, "nonzero row has zero assigned-codepoint energy")
    lower_numerator = absolute_maximum * 16
    lower_denominator = 127
    maximum_record = pack_scale32(SCALE32_SIGNIFICAND_MAX, SCALE32_EXPONENT_MAX)
    require(_compare_record_to_dyadic(maximum_record, lower_numerator, lower_denominator, base_shift) >= 0, "no legal Scale32 record satisfies the no-overflow lower bound")

    ideal_numerator = weight_codepoint * 16
    ideal_denominator = codepoint_squared
    if ideal_numerator <= 0 or _compare_dyadic(
        ideal_numerator,
        ideal_denominator,
        base_shift,
        lower_numerator,
        lower_denominator,
        base_shift,
    ) < 0:
        target_numerator, target_denominator = lower_numerator, lower_denominator
    else:
        target_numerator, target_denominator = ideal_numerator, ideal_denominator

    minimum_record = pack_scale32(SCALE32_SIGNIFICAND_MIN, SCALE32_EXPONENT_MIN)
    if _compare_record_to_dyadic(minimum_record, target_numerator, target_denominator, base_shift) > 0:
        floor_record = ceil_record = minimum_record
    elif _compare_record_to_dyadic(maximum_record, target_numerator, target_denominator, base_shift) < 0:
        floor_record = ceil_record = maximum_record
    else:
        ceil_record = _ceil_scale32_dyadic(target_numerator, target_denominator, base_shift)
        floor_record = (
            ceil_record
            if _compare_record_to_dyadic(ceil_record, target_numerator, target_denominator, base_shift) == 0
            else reference.previous_scale32_record(ceil_record)
        )
        require(floor_record is not None, "Scale32 floor record is missing")

    records = {int(floor_record), int(ceil_record), int(prior_record)}
    for record in (int(floor_record), int(ceil_record)):
        previous = reference.previous_scale32_record(record)
        following = reference.next_scale32_record(record)
        if previous is not None:
            records.add(previous)
        if following is not None:
            records.add(following)
    filtered = sorted(
        (
            record
            for record in records
            if _compare_record_to_dyadic(record, lower_numerator, lower_denominator, base_shift) >= 0
        ),
        key=_record_scale_key,
    )
    require(filtered, "row-scale candidate set is empty")
    return filtered, {
        "all_zero": False,
        "candidate_count": len(filtered),
        "floor_record": int(floor_record),
        "ceil_record": int(ceil_record),
    }


def _update_row_records(
    source: Tensor,
    assignments: Tensor,
    codepoints: Sequence[int],
    prior_records: Tensor,
    base_shift: int,
    objective_exponent: int,
) -> tuple[Tensor, dict[str, Any]]:
    weight_codepoint, codepoint_squared, absolute_maximum = _row_statistics(
        source, assignments, codepoints, base_shift
    )
    prior = [int(value) for value in prior_records.tolist()]
    selected: list[int] = []
    candidate_counts: list[int] = []
    changed = 0
    all_zero = 0
    for wp, p2, maximum, previous in zip(
        weight_codepoint,
        codepoint_squared,
        absolute_maximum,
        prior,
        strict=True,
    ):
        candidates, summary = _candidate_row_records(wp, p2, maximum, base_shift, previous)
        if summary["all_zero"]:
            choice = SCALE32_ALL_ZERO_RECORD
            all_zero += 1
        else:
            choice = min(
                candidates,
                key=lambda record: (
                    _row_objective(record, wp, p2, base_shift, objective_exponent),
                    *_record_scale_key(record),
                ),
            )
        selected.append(choice)
        candidate_counts.append(summary["candidate_count"])
        changed += int(choice != previous)
    records = torch.tensor(selected, dtype=torch.int64)
    return records, {
        "row_count": len(selected),
        "changed_row_count": changed,
        "all_zero_row_count": all_zero,
        "minimum_candidate_count": min(candidate_counts),
        "maximum_candidate_count": max(candidate_counts),
        "mean_candidate_count": sum(candidate_counts) / len(candidate_counts),
    }


def _strict_integer_codebook_dp(costs: Sequence[dict[int, int]]) -> tuple[tuple[int, ...], int]:
    require(bool(costs), "empty strict-codebook DP")
    states: dict[int, tuple[int, tuple[int, ...]]] = {
        point: (cost, (point,)) for point, cost in sorted(costs[0].items())
    }
    require(bool(states), "first strict-codebook cluster has no legal value")
    for cluster in costs[1:]:
        previous = sorted(states.items())
        current: dict[int, tuple[int, tuple[int, ...]]] = {}
        best: tuple[int, tuple[int, ...]] | None = None
        prior_index = 0
        for point, point_cost in sorted(cluster.items()):
            while prior_index < len(previous) and previous[prior_index][0] < point:
                candidate = previous[prior_index][1]
                if best is None or candidate < best:
                    best = candidate
                prior_index += 1
            if best is not None:
                current[point] = (best[0] + point_cost, best[1] + (point,))
        require(bool(current), "strict-codebook DP has no feasible increasing path")
        states = current
    cost, points = min(states.values())
    return points, cost


def _update_codebook(
    aggregates: dict[str, Any],
    prior_codebook: Sequence[int],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    prior = reference.validate_codebook(prior_codebook)
    counts = aggregates["cluster_counts"]
    costs: list[dict[int, int]] = []
    for index in range(CODEBOOK_ENTRIES):
        allowed = (prior[index],) if counts[index] == 0 else range(Q4_4_MIN, Q4_4_MAX + 1)
        linear = aggregates["linear_scaled"][index]
        quadratic = aggregates["quadratic_scaled"][index]
        costs.append({point: -2 * point * linear + point * point * quadratic for point in allowed})
    selected, optimum = _strict_integer_codebook_dp(costs)
    selected = reference.validate_codebook(selected)
    require(
        all(count != 0 or selected[index] == prior[index] for index, count in enumerate(counts)),
        "empty cluster did not retain its prior codepoint",
    )
    return selected, {
        "cluster_counts": counts,
        "empty_cluster_count": sum(count == 0 for count in counts),
        "fixed_assignment_optimum": {
            "scaled_integer": str(optimum),
            "binary_exponent": aggregates["objective_binary_exponent"],
        },
    }


def _objective_record(value: int, exponent: int) -> dict[str, Any]:
    return {"scaled_integer": str(value), "binary_exponent": exponent}


def _construct_joint_state(source: Tensor, module_name: str) -> dict[str, Any]:
    require(source.ndim == 2 and source.dtype == torch.bfloat16, f"source weight format differs: {module_name}")
    shift_min, shift_max, subnormals = _tensor_shift_range(source)
    require(subnormals == 0, f"pinned tensor contains BF16 subnormals: {module_name}")
    require(shift_max - shift_min <= 54, f"pinned tensor exponent span exceeds exact int64 backend: {module_name}")
    records, zero_rows = _initial_row_records(source, shift_min)
    histogram = _initial_histogram(source, records, shift_min)
    initial = reference.construct_initial_codebook(histogram)
    codebook = initial["codepoints"]
    assignments, aggregates = _assign_and_aggregate(source, records, codebook, shift_min, shift_max)
    objective_exponent = aggregates["objective_binary_exponent"]
    previous_objective = _objective(aggregates, codebook)
    trace: list[dict[str, Any]] = []

    for iteration in range(1, OUTER_ITERATIONS + 1):
        prior_records = records
        records, row_summary = _update_row_records(
            source,
            assignments,
            codebook,
            prior_records,
            shift_min,
            objective_exponent,
        )
        assignments_after_scale, scale_aggregates = _assign_and_aggregate(
            source, records, codebook, shift_min, shift_max
        )
        after_scale_objective = _objective(scale_aggregates, codebook)
        require(after_scale_objective <= previous_objective, f"row-scale update increased exact source SSE: {module_name}")

        codebook, codebook_summary = _update_codebook(scale_aggregates, codebook)
        assignments, final_aggregates = _assign_and_aggregate(
            source, records, codebook, shift_min, shift_max
        )
        final_objective = _objective(final_aggregates, codebook)
        require(final_objective <= after_scale_objective, f"codebook update increased exact source SSE: {module_name}")
        require(final_objective <= previous_objective, f"outer iteration increased exact source SSE: {module_name}")
        payload = v3.pack_codebook_indices(assignments)
        scale_raw = b"".join(struct.pack("<I", int(value)) for value in records.tolist())
        codebook_raw = bytes(point & 0xFF for point in codebook)
        trace.append(
            {
                "iteration": iteration,
                "prior_objective": _objective_record(previous_objective, objective_exponent),
                "after_row_scale_objective": _objective_record(after_scale_objective, objective_exponent),
                "final_objective": _objective_record(final_objective, objective_exponent),
                "row_scale_update": row_summary,
                "row_scale_stream_sha256": hashlib.sha256(scale_raw).hexdigest(),
                "codepoints": list(codebook),
                "codebook_record_sha256": hashlib.sha256(codebook_raw).hexdigest(),
                "codebook_update": codebook_summary,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        previous_objective = final_objective

    require(len(trace) == OUTER_ITERATIONS, f"joint constructor iteration count differs: {module_name}")
    payload = v3.pack_codebook_indices(assignments)
    scale_raw = b"".join(struct.pack("<I", int(value)) for value in records.tolist())
    codebook_raw = bytes(point & 0xFF for point in codebook)
    return {
        "shift_min": shift_min,
        "shift_max": shift_max,
        "shift_span": shift_max - shift_min,
        "subnormal_count": subnormals,
        "initial_all_zero_row_count": zero_rows,
        "initial_row_scale_stream_sha256": hashlib.sha256(
            b"".join(
                struct.pack("<I", int(value))
                for value in _initial_row_records(source, shift_min)[0].tolist()
            )
        ).hexdigest(),
        "initial_histogram_sha256": initial["histogram_sha256"],
        "initial_codepoints": list(initial["codepoints"]),
        "initial_lloyd_max_iterations": initial["iterations_completed"],
        "initial_lloyd_max_trace_sha256": initial["trace_sha256"],
        "outer_iterations_completed": len(trace),
        "final_records": records,
        "final_codepoints": codebook,
        "final_assignments": assignments,
        "payload": payload,
        "scale_raw": scale_raw,
        "codebook_raw": codebook_raw,
        "final_objective": _objective_record(previous_objective, objective_exponent),
        "iteration_trace": trace,
        "iteration_trace_sha256": canonical_sha256(trace),
    }


def _materialize_and_measure(
    module: nn.Linear,
    state: dict[str, Any],
    v3_record: dict[str, Any],
) -> dict[str, Any]:
    source = module.weight.detach()
    records = state["final_records"]
    assignments = state["final_assignments"]
    points = torch.tensor(state["final_codepoints"], dtype=torch.float64)
    rows, width = source.shape
    sse = 0.0
    with torch.no_grad():
        for start, stop in _row_chunks(rows, width):
            scale_significands, scale_exponents = _records_components(records[start:stop])
            scales = torch.ldexp(scale_significands.to(torch.float64), scale_exponents.to(torch.int32) - 15)
            decoded = points[assignments[start:stop].to(torch.int64)] * (scales[:, None] / 16.0)
            decoded_bf16 = decoded.to(torch.bfloat16)
            difference = source[start:stop].to(torch.float64) - decoded_bf16.to(torch.float64)
            sse += float(torch.sum(difference * difference))
            module.weight[start:stop].copy_(decoded_bf16)
    reconstruction_sha = v3.tensor_sha256(module.weight)
    v3_sse = float(v3_record["tensor_codebook_bf16_sse"])
    return {
        "module": v3_record["module"],
        "weight_count": int(module.weight.numel()),
        "v3_tensor_codebook_bf16_sse": v3_sse,
        "v4_joint_bf16_sse": sse,
        "v4_non_regressing_against_v3": sse <= v3_sse,
        "v4_strictly_improved_against_v3": sse < v3_sse,
        "reconstruction_bf16_sha256": reconstruction_sha,
    }


def _policy_and_format() -> dict[str, Any]:
    body = {
        "payload": {
            "bits_per_weight": 4,
            "semantics": "unsigned codebook index 0 through 15",
            "packing": "row-major; even flat element low nibble; odd flat element high nibble",
        },
        "row_scale": {
            "format": "Scale32",
            "source": "eight-iteration exact joint least-squares update from pinned BF16 weights",
            "bytes_per_row": 4,
            "additional_records": 0,
            "selection_tie_order": ["source_space_sse", "smaller_positive_scale", "smaller_packed_record"],
        },
        "codebook": {
            "scope": "one per Linear tensor",
            "entry_count": 16,
            "bytes": 16,
            "alignment_bytes": 16,
            "format": "strictly increasing signed Q4.4 int8",
            "initial_lloyd_max_iterations": reference.LLOYD_MAX_ITERATIONS,
            "outer_iterations": OUTER_ITERATIONS,
            "empty_cluster": "retain prior codepoint",
            "global_update": "exact strictly-increasing dynamic program with lexicographic ties",
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


def _quantize_model_joint(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {record["module"]: record for record in grouped.expected_transformer_tensors()}
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == 169, "Qwen linear count differs")
    require([name for name, _module in modules if name != "lm_head"] == list(expected), "transformer linear order differs")
    require(sha256_file(V3_RECONSTRUCTION_PATH) == V3_RECONSTRUCTION_SHA256, "V3 reconstruction reference hash differs")
    v3_report = load_json(V3_RECONSTRUCTION_PATH)
    v3_records = {item["module"]: item for item in v3_report["tensors"]}
    require(list(v3_records) == [name for name, _module in modules], "V3 reconstruction tensor order differs")

    weight_manifest: list[dict[str, Any]] = []
    payload_manifest: list[dict[str, Any]] = []
    row_scale_manifest: list[dict[str, Any]] = []
    codebook_manifest: list[dict[str, Any]] = []
    reconstruction_manifest: list[dict[str, Any]] = []
    iteration_trace_manifest: list[dict[str, Any]] = []
    payload_stream = hashlib.sha256()
    scale_stream = hashlib.sha256()
    codebook_stream = hashlib.sha256()
    reconstruction_stream = hashlib.sha256()
    total_weight_count = 0
    total_payload_bytes = 0
    total_scale_bytes = 0
    total_codebook_bytes = 0

    for ordinal, (name, module) in enumerate(modules, start=1):
        started = time.monotonic()
        print(f"V4_CONSTRUCT_START {ordinal}/169 {name}", flush=True)
        source_sha = v3.tensor_sha256(module.weight)
        state = _construct_joint_state(module.weight.detach(), name)
        reconstruction = _materialize_and_measure(module, state, v3_records[name])
        payload = state["payload"]
        scale_raw = state["scale_raw"]
        codebook_raw = state["codebook_raw"]
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
            "record_count": int(module.weight.shape[0]),
            "bytes": len(scale_raw),
            "sha256": scale_sha,
        }
        codebook_record = {
            "module": name,
            "bytes": 16,
            "sha256": codebook_sha,
            "q4_4_codepoints": list(state["final_codepoints"]),
        }
        trace_record = {
            "module": name,
            "source_bf16_sha256": source_sha,
            "shift_min": state["shift_min"],
            "shift_max": state["shift_max"],
            "shift_span": state["shift_span"],
            "subnormal_count": state["subnormal_count"],
            "initial_all_zero_row_count": state["initial_all_zero_row_count"],
            "initial_row_scale_stream_sha256": state["initial_row_scale_stream_sha256"],
            "initial_histogram_sha256": state["initial_histogram_sha256"],
            "initial_codepoints": state["initial_codepoints"],
            "initial_lloyd_max_iterations": state["initial_lloyd_max_iterations"],
            "initial_lloyd_max_trace_sha256": state["initial_lloyd_max_trace_sha256"],
            "outer_iterations_completed": state["outer_iterations_completed"],
            "final_objective": state["final_objective"],
            "iteration_trace": state["iteration_trace"],
            "iteration_trace_sha256": state["iteration_trace_sha256"],
        }
        record = {
            "module": name,
            "name": f"{name}.weight",
            "shape": list(module.weight.shape),
            "scope": "separated_lm_head" if name == "lm_head" else "transformer",
            "payload": payload_record,
            "row_scale": scale_record,
            "codebook": codebook_record,
            "reconstruction": reconstruction,
            "iteration_trace_sha256": state["iteration_trace_sha256"],
        }
        weight_manifest.append(record)
        payload_manifest.append(payload_record)
        row_scale_manifest.append(scale_record)
        codebook_manifest.append(codebook_record)
        reconstruction_manifest.append(reconstruction)
        iteration_trace_manifest.append(trace_record)
        payload_stream.update(name.encode() + b"\0" + payload)
        scale_stream.update(name.encode() + b"\0" + scale_raw)
        codebook_stream.update(name.encode() + b"\0" + codebook_raw)
        reconstruction_stream.update(name.encode() + b"\0" + bytes.fromhex(reconstruction["reconstruction_bf16_sha256"]))
        total_weight_count += weight_count
        total_payload_bytes += len(payload)
        total_scale_bytes += len(scale_raw)
        total_codebook_bytes += len(codebook_raw)
        elapsed = time.monotonic() - started
        print(
            f"V4_CONSTRUCT_DONE {ordinal}/169 {name} seconds={elapsed:.3f} "
            f"v3_sse={reconstruction['v3_tensor_codebook_bf16_sse']:.12g} "
            f"v4_sse={reconstruction['v4_joint_bf16_sse']:.12g}",
            flush=True,
        )
        del state
        gc.collect()

    aggregate_v4_sse = float(sum(item["v4_joint_bf16_sse"] for item in reconstruction_manifest))
    non_regressing_count = sum(bool(item["v4_non_regressing_against_v3"]) for item in reconstruction_manifest)
    strictly_improved_count = sum(bool(item["v4_strictly_improved_against_v3"]) for item in reconstruction_manifest)
    reconstruction_body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "measurement": "sum-squared error from pinned BF16 source to BF16-materialized reconstruction",
        "reference": {
            "candidate_id": v3_report["candidate_id"],
            "artifact": file_record(V3_RECONSTRUCTION_PATH),
            "aggregate_bf16_sse": V3_AGGREGATE_BF16_SSE,
        },
        "tensor_count": len(reconstruction_manifest),
        "per_tensor_non_regressing_count": non_regressing_count,
        "per_tensor_strictly_improved_count": strictly_improved_count,
        "all_169_tensors_non_regressing_against_v3": non_regressing_count == 169,
        "aggregate_v3_bf16_sse": V3_AGGREGATE_BF16_SSE,
        "aggregate_v4_joint_bf16_sse": aggregate_v4_sse,
        "aggregate_strictly_improved_against_v3": aggregate_v4_sse < V3_AGGREGATE_BF16_SSE,
        "tensors": reconstruction_manifest,
    }
    reconstruction_report = {
        **reconstruction_body,
        "status": (
            "PASS_169_OF_169_AND_STRICT_AGGREGATE_V3_IMPROVEMENT"
            if reconstruction_body["all_169_tensors_non_regressing_against_v3"]
            and reconstruction_body["aggregate_strictly_improved_against_v3"]
            else "NO_GO_V3_RELATIVE_MODEL_ONLY_RECONSTRUCTION_GATE"
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
    require(len(transformer_records) == 168 and len(weight_manifest) == 169, "joint tensor count differs")
    require(total_codebook_bytes == 2704, "whole-model codebook byte count differs")
    require(abs(standard_layer_bits - 4.027257898351649) < 1e-15, "standard-layer stored bits differ")
    require(abs(lm_head_bits - 4.035715225959803) < 1e-15, "lm_head stored bits differ")
    require(standard_layer_bits <= 4.036 and lm_head_bits <= 4.036, "stored-bit cap exceeded")
    manifests = {
        "payload": payload_manifest,
        "row_scale": row_scale_manifest,
        "codebook": codebook_manifest,
        "reconstruction": reconstruction_manifest,
        "iteration_trace": iteration_trace_manifest,
    }
    hashes = {
        "payload_manifest_sha256": canonical_sha256(payload_manifest),
        "row_scale_manifest_sha256": canonical_sha256(row_scale_manifest),
        "codebook_manifest_sha256": canonical_sha256(codebook_manifest),
        "reconstruction_manifest_sha256": canonical_sha256(reconstruction_manifest),
        "iteration_trace_manifest_sha256": canonical_sha256(iteration_trace_manifest),
        "weight_manifest_sha256": canonical_sha256(weight_manifest),
        "reconstruction_report_sha256": reconstruction_report["reconstruction_report_sha256"],
        "payload_stream_sha256": payload_stream.hexdigest(),
        "row_scale_stream_sha256": scale_stream.hexdigest(),
        "codebook_stream_sha256": codebook_stream.hexdigest(),
        "reconstruction_stream_sha256": reconstruction_stream.hexdigest(),
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


def _load_candidate(label: str) -> tuple[nn.Module, dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    source_hash = v3.tensor_sha256(embedding)
    require(source_hash == SOURCE_EMBEDDING_SHA256, "source embedding BF16 hash differs")

    cloned = lm_head.detach().clone()
    require(v3.tensor_sha256(cloned) == source_hash, "detached lm_head clone differs before split")
    model.lm_head.weight = nn.Parameter(cloned, requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight is not embedding, "lm_head Parameter object remained tied")
    require(model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head storage remained tied")
    require(v3.tensor_sha256(model.lm_head.weight) == source_hash, "lm_head clone bytes differ after split")
    require(id(model.model.embed_tokens.weight) == source_object_id, "embedding Parameter object changed during split")
    require(model.model.embed_tokens.weight.data_ptr() == source_pointer, "embedding storage changed during split")

    weight_manifest, construction = _quantize_model_joint(model)
    require(model.config.tie_word_embeddings is False, "candidate config was retied")
    require(model.model.embed_tokens.weight is embedding, "embedding Parameter object changed after joint construction")
    require(model.model.embed_tokens.weight.data_ptr() == source_pointer, "embedding storage changed after joint construction")
    require(list(embedding.shape) == source_shape and str(embedding.dtype) == source_dtype, "embedding shape or dtype changed")
    embedding_hash = v3.tensor_sha256(embedding)
    require(embedding_hash == source_hash == SOURCE_EMBEDDING_SHA256, "embedding bytes changed after joint construction")
    state = v3.state_manifest(model)
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
        "module_counts": {"transformer_linears": 168, "lm_head": 1, "total_joint_linears": 169},
        "identity": identity,
        "storage": construction["storage"],
        "payload_manifest": construction["manifests"]["payload"],
        "row_scale_manifest": construction["manifests"]["row_scale"],
        "codebook_manifest": construction["manifests"]["codebook"],
        "reconstruction_manifest": construction["manifests"]["reconstruction"],
        "iteration_trace_manifest": construction["manifests"]["iteration_trace"],
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
            "pre_joint_byte_identical": True,
            "config_tie_word_embeddings": bool(model.config.tie_word_embeddings),
        },
        "post_joint": {
            "embedding_bf16_sha256": embedding_hash,
            "embedding_hash_matches_source": embedding_hash == SOURCE_EMBEDDING_SHA256,
            "lm_head_storage_is_separate": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
            "transformer_joint_count": 168,
            "lm_head_joint_count": 1,
        },
    }
    return model, manifest, witness, construction["reconstruction"]


def _compare_candidates(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> dict[str, Any]:
    comparisons = {key: candidate_a["identity"][key] == candidate_b["identity"][key] for key in STATE_MATCH_KEYS}
    require(all(comparisons.values()), f"candidate A/B identity differs: {comparisons}")
    manifest_fields = (
        "payload_manifest",
        "row_scale_manifest",
        "codebook_manifest",
        "reconstruction_manifest",
        "iteration_trace_manifest",
        "weight_manifest",
        "state_manifest",
    )
    field_matches = {field: candidate_a[field] == candidate_b[field] for field in manifest_fields}
    require(all(field_matches.values()), f"candidate A/B whole-model manifests differ: {field_matches}")
    return {
        "state_match_keys": list(STATE_MATCH_KEYS),
        "identity_comparisons": comparisons,
        "manifest_byte_equivalent_fields": list(manifest_fields),
        "manifest_comparisons": field_matches,
        "all_match": True,
    }


def combined_self_test() -> dict[str, Any]:
    environment = require_project_python()
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    task = load_json(TASK_PATH)
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "frozen V4 Engineer authority differs")
    root_existed_before = OUTPUT_DIR.exists()
    marker_existed_before = MARKER_PATH.exists()
    require(not marker_existed_before, "V4 execution marker already exists")

    exact_reference = reference.self_test()
    fixture = reference.faithful_fixture()
    source = torch.tensor(
        [[float(value) for value in row] for row in fixture],
        dtype=torch.bfloat16,
    )
    scalable_a = _construct_joint_state(source, "scalable_fixture_a")
    scalable_b = _construct_joint_state(source, "scalable_fixture_b")
    expected = exact_reference["construction"]
    require(scalable_a["final_records"].tolist() == expected["final_row_scale_records"], "scalable fixture row scales differ from exact reference")
    require(list(scalable_a["final_codepoints"]) == expected["final_codepoints"], "scalable fixture codebook differs from exact reference")
    require(scalable_a["payload"].hex() == expected["payload_hex"], "scalable fixture payload differs from exact reference")
    require(scalable_a["scale_raw"] == scalable_b["scale_raw"], "independent scalable fixture row scales differ")
    require(scalable_a["codebook_raw"] == scalable_b["codebook_raw"], "independent scalable fixture codebooks differ")
    require(scalable_a["payload"] == scalable_b["payload"], "independent scalable fixture payloads differ")
    require(scalable_a["iteration_trace"] == scalable_b["iteration_trace"], "independent scalable fixture traces differ")
    for scalable, exact in zip(scalable_a["iteration_trace"], expected["iteration_trace"], strict=True):
        exact_scale_raw = b"".join(struct.pack("<I", int(value)) for value in exact["row_scale_records"])
        require(scalable["row_scale_stream_sha256"] == hashlib.sha256(exact_scale_raw).hexdigest(), "scalable fixture iteration row scales differ")
        require(scalable["codepoints"] == exact["codepoints"], "scalable fixture iteration codebook differs")
        require(scalable["payload_sha256"] == exact["payload_sha256"], "scalable fixture iteration payload differs")

    tie_costs_integer = [{point: 0 for point in range(-2, 3)} for _ in range(3)]
    integer_points, integer_cost = _strict_integer_codebook_dp(tie_costs_integer)
    require(integer_points == (-2, -1, 0) and integer_cost == 0, "scalable strict-DP tie order differs")
    require(OUTPUT_DIR.exists() == root_existed_before, "combined self-test changed the V4 namespace")
    require(MARKER_PATH.exists() == marker_existed_before is False, "combined self-test changed the V4 marker")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "candidate_id": CANDIDATE_ID,
        "mission_id": MISSION_ID,
        "task_id": TASK_ID,
        "environment": environment,
        "runner_sha256": sha256_file(RUNNER_PATH),
        "backend_sha256": sha256_file(BACKEND_PATH),
        "task_sha256": sha256_file(TASK_PATH),
        "plan_sha256": sha256_file(PLAN_PATH),
        "reference_self_test_sha256": canonical_sha256(exact_reference),
        "checks": {
            "exact_reference_self_test_passed": exact_reference["status"] == "PASS",
            "scalable_exact_dyadic_lattice": True,
            "scalable_matches_fraction_reference_final_row_scales": True,
            "scalable_matches_fraction_reference_final_codebook": True,
            "scalable_matches_fraction_reference_final_payload": True,
            "scalable_matches_fraction_reference_all_eight_iteration_manifests": True,
            "independent_scalable_fixture_a_b_match": True,
            "strict_integer_dp_lexicographic_tie": True,
            "chunk_values": CHUNK_VALUES,
            "official_execution_marker_exists": False,
            "official_execution_mode_exposed": False,
        },
        "scalable_fixture": {
            "shape": list(source.shape),
            "final_row_scale_stream_sha256": hashlib.sha256(scalable_a["scale_raw"]).hexdigest(),
            "final_codebook_record_sha256": hashlib.sha256(scalable_a["codebook_raw"]).hexdigest(),
            "final_payload_sha256": hashlib.sha256(scalable_a["payload"]).hexdigest(),
            "iteration_trace_sha256": scalable_a["iteration_trace_sha256"],
        },
        "official_build_namespace_changed": False,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": False,
        "full_model_preflight_exposed": True,
    }
    return result


def _validate_reconstruction(report: dict[str, Any]) -> bool:
    require(report["tensor_count"] == len(report["tensors"]) == 169, "V4 reconstruction tensor count differs")
    require(report["reference"]["artifact"]["sha256"] == V3_RECONSTRUCTION_SHA256, "V4 reconstruction reference hash differs")
    non_regressing = sum(bool(item["v4_non_regressing_against_v3"]) for item in report["tensors"])
    strict = sum(bool(item["v4_strictly_improved_against_v3"]) for item in report["tensors"])
    require(non_regressing == report["per_tensor_non_regressing_count"], "V4 reconstruction non-regression aggregate differs")
    require(strict == report["per_tensor_strictly_improved_count"], "V4 reconstruction strict-improvement aggregate differs")
    passed = (
        report["all_169_tensors_non_regressing_against_v3"] is True
        and non_regressing == 169
        and report["aggregate_v4_joint_bf16_sse"] < report["aggregate_v3_bf16_sse"] == V3_AGGREGATE_BF16_SSE
        and report["aggregate_strictly_improved_against_v3"] is True
    )
    require((report["status"].startswith("PASS_")) == passed, "V4 reconstruction status differs")
    body = {key: value for key, value in report.items() if key not in {"status", "reconstruction_report_sha256"}}
    require(report["reconstruction_report_sha256"] == canonical_sha256(body), "V4 reconstruction report hash differs")
    return passed


def _validate_quality(report: dict[str, Any]) -> bool:
    _configure_v3_quality_helpers()
    return v3.validate_quality(report)


def _validate_dynamic(report: dict[str, Any]) -> bool:
    _configure_v3_quality_helpers()
    return v3.validate_dynamic(report)


def _audit_attempt_namespace(expect_root_absent: bool) -> dict[str, Any]:
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
        require(not OUTPUT_DIR.exists(), "V4 namespace existed before preparation")
    require(not matches, "a prior V4 execution marker exists")
    require(not attempts, f"V4 attempt namespace is already consumed: {attempts}")
    return {
        "candidate_id": CANDIDATE_ID,
        "root_existed_at_audit": OUTPUT_DIR.exists(),
        "matching_execution_markers": matches,
        "attempt_directories": attempts,
        "authorization_consumed": False,
    }


def _recover_nonqualifying_preparation() -> dict[str, Any]:
    if not OUTPUT_DIR.exists():
        return {"fresh_namespace": True, "recovered": False, "archived_artifacts": []}
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "cannot recover a consumed V4 attempt namespace")
    complete = PREATTEMPT_DIR.is_dir() and CONTRACT_PATH.is_file() and (READY_PATH.is_file() or NO_GO_PATH.is_file())
    require(not complete, "V4 pre-attempt closure already exists; use verify")
    candidates = [path for path in OUTPUT_DIR.iterdir() if path.name != "preflight-failures"]
    require(candidates, "existing V4 namespace has ambiguous provenance")
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


def _artifact_records() -> dict[str, Any]:
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
        "v3_reconstruction_reference": V3_RECONSTRUCTION_PATH,
        "official_matrix_hash_only": MATRIX_PATH,
        "runner_source": RUNNER_PATH,
        "scalable_backend_source": BACKEND_PATH,
        "reviewed_v3_preflight_helper": V3_RUNNER_PATH,
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


def _build_contract(
    environment: dict[str, Any],
    source_identity: dict[str, Any],
    namespace_audit: dict[str, Any],
    state_comparison: dict[str, Any],
    gate_passes: dict[str, bool],
) -> dict[str, Any]:
    task = load_json(TASK_PATH)
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "V4 Engineer authority differs")
    require(task["fresh_attempt_namespace"]["attempt_budget"] == 1, "V4 attempt budget differs")
    require(task["fresh_attempt_namespace"]["attempts_consumed_at_task_freeze"] == 0, "V4 attempt was consumed at freeze")
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(sha256_file(V3_RECONSTRUCTION_PATH) == V3_RECONSTRUCTION_SHA256, "V3 reconstruction reference hash differs")
    artifacts = _artifact_records()
    format_policy = _policy_and_format()
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
            "backend": "exact chunked BF16 dyadic lattice",
            "source_tied_then_runtime_split_lm_head": True,
            "source_embedding_bf16_sha256": SOURCE_EMBEDDING_SHA256,
            "embedding_bytes_preserved": True,
            "transformer_joint_linears": 168,
            "separated_lm_head_joint_linears": 1,
            "candidate_a_and_b_whole_model_manifests_match": state_comparison,
        },
        "pre_attempt_gates": {
            "v3_relative_model_only_reconstruction": {"passed": gate_passes["v3_relative_model_only_reconstruction"], "artifact": artifacts["model_only_reconstruction"]},
            "non_evaluator_quality_8_of_8": {"passed": gate_passes["non_evaluator_quality_8_of_8"], "artifact": artifacts["non_evaluator_quality"]},
            "dynamic_312_event_closure": {"passed": gate_passes["dynamic_312_event_closure"], "artifact": artifacts["dynamic_preflight"]},
            "two_build_alias_and_manifest_closure": {"passed": gate_passes["two_build_alias_and_manifest_closure"], "artifact": artifacts["alias_separation_witness"]},
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
            "conditional_execution_rule": "consume attempt-0001 only after PREATTEMPT_READY and a separately verified official executor",
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


def _verify_bound_record(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    require(path.is_file(), f"bound artifact is missing: {record['path']}")
    require(path.stat().st_size == record["bytes"], f"bound artifact size differs: {record['path']}")
    require(sha256_file(path) == record["sha256"], f"bound artifact hash differs: {record['path']}")


def load_verified_closure() -> dict[str, Any]:
    _configure_v3_quality_helpers()
    require_project_python()
    require(CONTRACT_PATH.is_file(), "V4 predeclared contract is missing")
    require(READY_PATH.is_file() != NO_GO_PATH.is_file(), "exactly one V4 pre-attempt terminal record is required")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "V4 predeclared contract wrapper differs")
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "V4 predeclared contract canonical hash differs")
    contract = wrapper["contract"]
    require(contract["candidate_id"] == CANDIDATE_ID and contract["task_id"] == TASK_ID, "V4 predeclared authority differs")
    require(contract["artifacts"]["runner_source"]["sha256"] == sha256_file(RUNNER_PATH), "V4 runner changed after freeze")
    require(contract["artifacts"]["scalable_backend_source"]["sha256"] == sha256_file(BACKEND_PATH), "V4 backend changed after freeze")
    for record in contract["artifacts"].values():
        _verify_bound_record(record)
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(sha256_file(V3_RECONSTRUCTION_PATH) == V3_RECONSTRUCTION_SHA256, "V3 reconstruction reference hash differs")
    require(verify_source_snapshot() == contract["source_model"], "source snapshot identity differs")
    packages = {**verify_versions(), "numpy": importlib.metadata.version("numpy")}
    require(packages == contract["environment"]["packages"], "package versions differ")
    require(combined_self_test() == load_json(SELF_TEST_PATH), "V4 runner self-test result differs")
    candidate_a = load_json(CANDIDATE_A_PATH)
    candidate_b = load_json(CANDIDATE_B_PATH)
    comparison = _compare_candidates(candidate_a, candidate_b)
    witness = load_json(ALIAS_WITNESS_PATH)
    require(witness["state_comparison"] == comparison, "V4 alias witness state comparison differs")
    gate_passes = {
        "v3_relative_model_only_reconstruction": _validate_reconstruction(load_json(RECONSTRUCTION_PATH)),
        "non_evaluator_quality_8_of_8": _validate_quality(load_json(QUALITY_PATH)),
        "dynamic_312_event_closure": _validate_dynamic(load_json(DYNAMIC_PATH)),
        "two_build_alias_and_manifest_closure": comparison["all_match"]
        and witness["embedding_preserved_in_both"]
        and witness["lm_head_separate_in_both"],
    }
    require(contract["pre_attempt_gates"]["all_pass"] == all(gate_passes.values()), "V4 contract aggregate gate status differs")
    for name, passed in gate_passes.items():
        require(contract["pre_attempt_gates"][name]["passed"] == passed, f"V4 contract gate differs: {name}")
    terminal_path = READY_PATH if READY_PATH.is_file() else NO_GO_PATH
    terminal = load_json(terminal_path)
    require(terminal["candidate_id"] == CANDIDATE_ID, "V4 terminal pre-attempt candidate differs")
    require(terminal["contract"]["sha256"] == sha256_file(CONTRACT_PATH), "V4 terminal contract file hash differs")
    require(terminal["contract_sha256"] == wrapper["contract_sha256"], "V4 terminal contract canonical hash differs")
    require(terminal["bound_artifacts"] == contract["artifacts"], "V4 terminal artifact snapshot differs")
    require(terminal["official_execution_marker_exists"] is False and terminal["attempts_consumed"] == 0, "V4 terminal record claims an official attempt")
    if terminal_path == READY_PATH:
        require(terminal["status"] == "PREATTEMPT_READY" and all(gate_passes.values()), "V4 ready record exists without all gates passing")
    else:
        require(terminal["status"] == "PREATTEMPT_NO_GO" and not all(gate_passes.values()), "V4 no-go record does not preserve a failed gate")
    _audit_attempt_namespace(expect_root_absent=False)
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "V4 official attempt exists after marker-free closure")
    quality = load_json(QUALITY_PATH)
    dynamic = load_json(DYNAMIC_PATH)
    reconstruction = load_json(RECONSTRUCTION_PATH)
    return {
        "status": terminal["status"],
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": wrapper["contract_sha256"],
        "gate_passes": gate_passes,
        "aggregate_v4_joint_bf16_sse": reconstruction["aggregate_v4_joint_bf16_sse"],
        "per_tensor_non_regressing_count": reconstruction["per_tensor_non_regressing_count"],
        "exact_sequence_matches": quality["exact_bf16_sequence_match_count"],
        "gate_outcome_matches": quality["response_gate_outcome_match_count"],
        "event_count": dynamic["activation_execution"]["event_count"],
        "attempts_consumed": 0,
        "official_execution_mode_exposed": False,
    }


def prepare() -> dict[str, Any]:
    _configure_v3_quality_helpers()
    environment = require_project_python()
    recovery = _recover_nonqualifying_preparation()
    namespace_audit = _audit_attempt_namespace(expect_root_absent=recovery["fresh_namespace"])
    namespace_audit["preparation_recovery"] = recovery
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    verify_source_contract()
    source_identity = verify_source_snapshot()
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(sha256_file(V3_RECONSTRUCTION_PATH) == V3_RECONSTRUCTION_SHA256, "V3 reconstruction reference hash differs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    probe = OUTPUT_DIR / ".preflight-write-probe"
    fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, b"preflight\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    require(probe.read_bytes() == b"preflight\n", "V4 output-path readback differs")
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
    inputs, input_manifest = v3.non_evaluator_inputs(tokenizer)
    self_test = combined_self_test()

    chronology["candidate_a_started_at_utc"] = utc_now()
    print("V4_CANDIDATE_A_START", flush=True)
    candidate_a, manifest_a, witness_a, reconstruction_a = _load_candidate("candidate_a_independent_construction")
    chronology["candidate_a_complete_at_utc"] = utc_now()
    print("V4_CANDIDATE_A_DONE", flush=True)
    del candidate_a
    gc.collect()

    chronology["candidate_b_started_at_utc"] = utc_now()
    print("V4_CANDIDATE_B_START", flush=True)
    candidate_b, manifest_b, witness_b, reconstruction_b = _load_candidate("candidate_b_independent_quality_and_dynamic_preflight")
    state_comparison = _compare_candidates(manifest_a, manifest_b)
    require(reconstruction_a == reconstruction_b, "V4 candidate A/B reconstruction reports differ")
    chronology["candidate_b_constructed_at_utc"] = utc_now()
    print("V4_CANDIDATE_B_CONSTRUCTED quality_dynamic_start", flush=True)
    quality, dynamic = v3.non_evaluator_quality_and_dynamic(candidate_b, tokenizer, inputs, input_manifest)
    chronology["candidate_b_quality_and_dynamic_complete_at_utc"] = utc_now()
    print("V4_QUALITY_DYNAMIC_DONE", flush=True)
    del candidate_b
    gc.collect()

    gate_passes = {
        "v3_relative_model_only_reconstruction": _validate_reconstruction(reconstruction_b),
        "non_evaluator_quality_8_of_8": _validate_quality(quality),
        "dynamic_312_event_closure": _validate_dynamic(dynamic),
        "two_build_alias_and_manifest_closure": state_comparison["all_match"]
        and witness_a["post_joint"]["embedding_hash_matches_source"]
        and witness_b["post_joint"]["embedding_hash_matches_source"]
        and witness_a["post_joint"]["lm_head_storage_is_separate"]
        and witness_b["post_joint"]["lm_head_storage_is_separate"],
    }
    require(not MARKER_PATH.exists() and not ATTEMPT_DIR.exists(), "V4 official attempt appeared before artifact freeze")

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
            "embedding_preserved_in_both": witness_a["post_joint"]["embedding_hash_matches_source"] and witness_b["post_joint"]["embedding_hash_matches_source"],
            "lm_head_separate_in_both": witness_a["post_joint"]["lm_head_storage_is_separate"] and witness_b["post_joint"]["lm_head_storage_is_separate"],
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

    wrapper = _build_contract(environment, source_identity, namespace_audit, state_comparison, gate_passes)
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
            "all marker-free V4 gates pass; PREATTEMPT_READY is sealed and attempt-0001 may be consumed only by the separately verified conditional official executor"
            if all_pass
            else "preserve V4 no-go evidence, leave attempt-0001 absent and unconsumed, keep selected_policy_id null, and submit for Fresh Review"
        ),
    }
    atomic_write_json(terminal_path, terminal)
    verified = load_verified_closure()
    for path in list(PREATTEMPT_DIR.iterdir()) + [CONTRACT_PATH, terminal_path]:
        if path.is_file():
            path.chmod(0o444)
    return verified


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
            "regression": "repair only the isolated marker-free V4 runner and rerun pre-attempt preparation; official authorization remains unconsumed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "backend_sha256": sha256_file(BACKEND_PATH),
            "traceback": traceback.format_exc(),
            "attempt_authorization_consumed": False,
            "official_execution_marker_exists": False,
            "official_execution_mode_exposed": False,
            "failed_at_utc": utc_now(),
        },
    )
