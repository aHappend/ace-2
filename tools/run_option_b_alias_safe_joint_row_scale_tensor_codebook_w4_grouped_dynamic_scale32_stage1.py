#!/usr/bin/env python3
"""Construct and verify the marker-free V4 joint W4 policy.

The in-file Fraction implementation remains the exact small-fixture oracle.
The prepare/verify commands delegate to a chunked exact-dyadic pinned-model
backend.  No command creates attempt-0001 or an execution marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

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


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "option_b_alias_safe_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_w4a8_v4"
MISSION_ID = "8982b28c1746"
TASK_ID = "task-5c2e87fd2509"
TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_JOINT_ROW_SCALE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V4_ENGINEER_TASK.json"
TASK_COMPANION = TASK_PATH.with_suffix(".sha256")
PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_JOINT_ROW_SCALE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V4_PLAN.json"
PLAN_COMPANION = PLAN_PATH.with_suffix(".sha256")
OFFICIAL_ROOT = ROOT / "build/stage1-option-b-alias-safe-joint-row-scale-tensor-codebook-w4-grouped-dynamic-scale32-w4a8-v4"
OFFICIAL_MARKER = OFFICIAL_ROOT / "attempt-0001/execution_started.json"
RUNNER_PATH = Path(__file__).resolve()

Q4_4_MIN = -128
Q4_4_MAX = 127
CODEBOOK_ENTRIES = 16
LLOYD_MAX_ITERATIONS = 32
OUTER_ITERATIONS = 8
INITIAL_CODEPOINTS = tuple(range(-128, 113, 16))
ARTIFACT_FILENAMES = (
    "manifest.json",
    "status.json",
    "raw_events.jsonl",
    "runner.log",
    "report.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fraction_record(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def verify_companion(path: Path, companion: Path) -> None:
    expected = f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}"
    require(companion.is_file(), f"checksum companion is missing: {companion}")
    require(companion.read_text(encoding="utf-8").strip() == expected, f"checksum companion differs: {path}")


def require_project_python() -> dict[str, str]:
    expected = ROOT / ".venv/bin/python"
    require(expected.is_file(), "project Python is missing")
    require(os.path.samefile(Path(sys.executable), expected), "wrong Python executable")
    return {"entrypoint": "./.venv/bin/python", "python": platform.python_version()}


def expect_rejected(call: Callable[[], Any], message: str) -> bool:
    try:
        call()
    except (RuntimeError, ValueError, OverflowError):
        return True
    raise RuntimeError(message)


def require_digest(actual: str, expected: str, message: str) -> None:
    require(actual == expected, message)


def round_fraction_ties_to_even(value: Fraction) -> int:
    magnitude = abs(value.numerator)
    quotient, remainder = divmod(magnitude, value.denominator)
    doubled = remainder * 2
    if doubled > value.denominator or (doubled == value.denominator and quotient & 1):
        quotient += 1
    return -quotient if value.numerator < 0 else quotient


def bf16_fraction(bits: int) -> Fraction:
    require(type(bits) is int and 0 <= bits <= 0xFFFF, "BF16 bit pattern differs")
    sign = -1 if bits & 0x8000 else 1
    exponent = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    require(exponent != 0xFF, "non-finite BF16 fixture value")
    if exponent == 0:
        value = Fraction(mantissa, 1 << 133)
    else:
        significand = 128 + mantissa
        shift = exponent - 134
        value = Fraction(significand << shift, 1) if shift >= 0 else Fraction(significand, 1 << -shift)
    return sign * value


def bf16_bits_for_exact_fraction(value: Fraction) -> int:
    sign = 0x8000 if value < 0 else 0
    magnitude = abs(value)
    if magnitude == 0:
        return sign
    subnormal = magnitude * (1 << 133)
    if subnormal.denominator == 1 and 1 <= subnormal.numerator <= 127:
        return sign | subnormal.numerator
    for exponent in range(1, 255):
        shift = exponent - 134
        scaled = magnitude / (1 << shift) if shift >= 0 else magnitude * (1 << -shift)
        if scaled.denominator == 1 and 128 <= scaled.numerator <= 255:
            return sign | (exponent << 7) | (scaled.numerator - 128)
    raise RuntimeError(f"fixture value is not exactly representable as BF16: {value}")


def validate_bf16_rows(rows: Sequence[Sequence[Fraction]]) -> list[list[int]]:
    require(bool(rows), "empty BF16 matrix")
    width = len(rows[0])
    require(width > 0 and all(len(row) == width for row in rows), "BF16 matrix is ragged")
    encoded: list[list[int]] = []
    for row in rows:
        bits = [bf16_bits_for_exact_fraction(value) for value in row]
        require([bf16_fraction(value) for value in bits] == list(row), "BF16 fixture round-trip differs")
        encoded.append(bits)
    return encoded


def scale32_value(record: int) -> Fraction:
    numerator, denominator = scale32_ratio(record)
    return Fraction(numerator, denominator)


def validate_codebook(codepoints: Sequence[int]) -> tuple[int, ...]:
    points = tuple(codepoints)
    require(len(points) == CODEBOOK_ENTRIES, "codebook entry count differs")
    require(all(type(value) is int for value in points), "codebook point is not an integer")
    require(all(Q4_4_MIN <= value <= Q4_4_MAX for value in points), "codebook point escaped signed Q4.4")
    require(all(left < right for left, right in zip(points, points[1:])), "codebook is not strictly increasing")
    return points


def nearest_codepoint_index(value: Fraction, codepoints: Sequence[int]) -> int:
    points = validate_codebook(codepoints)
    selected = 0
    distance = abs(value - points[0])
    for index, point in enumerate(points[1:], start=1):
        candidate = abs(value - point)
        if candidate < distance:
            selected = index
            distance = candidate
    return selected


def initial_row_scale(row: Sequence[Fraction]) -> int:
    maximum = max(abs(value) for value in row)
    if maximum == 0:
        return SCALE32_ALL_ZERO_RECORD
    return ceil_scale32_from_ratio(maximum.numerator, maximum.denominator * 7)


def normalized_histogram(rows: Sequence[Sequence[Fraction]], records: Sequence[int]) -> tuple[int, ...]:
    require(len(rows) == len(records), "row-scale count differs")
    histogram = [0] * 256
    for row, record in zip(rows, records, strict=True):
        scale = scale32_value(record)
        for weight in row:
            normalized = round_fraction_ties_to_even(weight * 16 / scale)
            require(Q4_4_MIN <= normalized <= Q4_4_MAX, "normalized BF16 weight escaped signed Q4.4")
            histogram[normalized - Q4_4_MIN] += 1
    return tuple(histogram)


def construct_initial_codebook(histogram: Sequence[int]) -> dict[str, Any]:
    require(len(histogram) == 256, "Q4.4 histogram bin count differs")
    require(all(type(count) is int and count >= 0 for count in histogram), "Q4.4 histogram count differs")
    require(sum(histogram) > 0, "empty Q4.4 histogram")
    points = INITIAL_CODEPOINTS
    trace: list[dict[str, Any]] = []
    for iteration in range(1, LLOYD_MAX_ITERATIONS + 1):
        counts = [0] * CODEBOOK_ENTRIES
        sums = [0] * CODEBOOK_ENTRIES
        for offset, count in enumerate(histogram):
            if count == 0:
                continue
            value = offset + Q4_4_MIN
            index = nearest_codepoint_index(Fraction(value), points)
            counts[index] += count
            sums[index] += value * count
        updated = [
            prior if count == 0 else round_fraction_ties_to_even(Fraction(total, count))
            for prior, total, count in zip(points, sums, counts, strict=True)
        ]
        updated.sort()
        points = validate_codebook(updated)
        trace.append(
            {
                "iteration": iteration,
                "codepoints": list(points),
                "empty_cluster_count": sum(count == 0 for count in counts),
            }
        )
    return {
        "codepoints": points,
        "iterations_completed": len(trace),
        "trace_sha256": canonical_sha256(trace),
        "histogram_sha256": hashlib.sha256(struct.pack("<256Q", *histogram)).hexdigest(),
    }


def assign_rows(
    rows: Sequence[Sequence[Fraction]],
    records: Sequence[int],
    codepoints: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    points = validate_codebook(codepoints)
    require(len(rows) == len(records), "assignment row-scale count differs")
    assigned: list[tuple[int, ...]] = []
    for row, record in zip(rows, records, strict=True):
        scale = scale32_value(record)
        assigned.append(
            tuple(nearest_codepoint_index(weight * 16 / scale, points) for weight in row)
        )
    return tuple(assigned)


def pack_indices(assignments: Sequence[Sequence[int]]) -> bytes:
    flat = [value for row in assignments for value in row]
    require(all(type(value) is int and 0 <= value <= 15 for value in flat), "payload index escaped four bits")
    if len(flat) & 1:
        flat.append(0)
    return bytes(flat[index] | (flat[index + 1] << 4) for index in range(0, len(flat), 2))


def source_sse(
    rows: Sequence[Sequence[Fraction]],
    records: Sequence[int],
    codepoints: Sequence[int],
    assignments: Sequence[Sequence[int]],
) -> Fraction:
    points = validate_codebook(codepoints)
    total = Fraction(0)
    for row, record, indices in zip(rows, records, assignments, strict=True):
        require(len(row) == len(indices), "SSE assignment width differs")
        scale = scale32_value(record)
        for weight, index in zip(row, indices, strict=True):
            error = weight - scale * points[index] / 16
            total += error * error
    return total


def previous_scale32_record(record: int) -> int | None:
    significand, exponent = unpack_scale32(record)
    if significand > SCALE32_SIGNIFICAND_MIN:
        return pack_scale32(significand - 1, exponent)
    if exponent > SCALE32_EXPONENT_MIN:
        return pack_scale32(SCALE32_SIGNIFICAND_MAX, exponent - 1)
    return None


def next_scale32_record(record: int) -> int | None:
    significand, exponent = unpack_scale32(record)
    if significand < SCALE32_SIGNIFICAND_MAX:
        return pack_scale32(significand + 1, exponent)
    if exponent < SCALE32_EXPONENT_MAX:
        return pack_scale32(SCALE32_SIGNIFICAND_MIN, exponent + 1)
    return None


def scaled_significand(value: Fraction, exponent: int) -> Fraction:
    shift = 15 - exponent
    return value * (1 << shift) if shift >= 0 else value / (1 << -shift)


def bracket_scale32(value: Fraction) -> tuple[int, int]:
    minimum_record = pack_scale32(SCALE32_SIGNIFICAND_MIN, SCALE32_EXPONENT_MIN)
    maximum_record = pack_scale32(SCALE32_SIGNIFICAND_MAX, SCALE32_EXPONENT_MAX)
    minimum = scale32_value(minimum_record)
    maximum = scale32_value(maximum_record)
    require(minimum <= value <= maximum, "Scale32 bracket target escaped legal range")
    candidates: set[int] = set()
    for exponent in range(SCALE32_EXPONENT_MIN, SCALE32_EXPONENT_MAX + 1):
        scaled = scaled_significand(value, exponent)
        floor_value = scaled.numerator // scaled.denominator
        ceil_value = -(-scaled.numerator // scaled.denominator)
        for significand in (floor_value, ceil_value):
            if SCALE32_SIGNIFICAND_MIN <= significand <= SCALE32_SIGNIFICAND_MAX:
                candidates.add(pack_scale32(significand, exponent))
    below = [record for record in candidates if scale32_value(record) <= value]
    above = [record for record in candidates if scale32_value(record) >= value]
    require(below and above, "Scale32 bracketing records are missing")
    floor_record = max(below, key=lambda record: (scale32_value(record), -record))
    ceil_record = min(above, key=lambda record: (scale32_value(record), record))
    return floor_record, ceil_record


def candidate_row_scale_records(ideal: Fraction, lower_bound: Fraction, prior_record: int) -> dict[str, Any]:
    minimum_record = pack_scale32(SCALE32_SIGNIFICAND_MIN, SCALE32_EXPONENT_MIN)
    maximum_record = pack_scale32(SCALE32_SIGNIFICAND_MAX, SCALE32_EXPONENT_MAX)
    minimum = scale32_value(minimum_record)
    maximum = scale32_value(maximum_record)
    require(lower_bound <= maximum, "no legal Scale32 record satisfies the no-overflow lower bound")
    constrained = min(max(ideal, lower_bound, minimum), maximum)
    floor_record, ceil_record = bracket_scale32(constrained)
    records = {floor_record, ceil_record, prior_record}
    for record in (floor_record, ceil_record):
        previous = previous_scale32_record(record)
        following = next_scale32_record(record)
        if previous is not None:
            records.add(previous)
        if following is not None:
            records.add(following)
    filtered = sorted(
        (record for record in records if scale32_value(record) >= lower_bound),
        key=lambda record: (scale32_value(record), record),
    )
    require(filtered, "row-scale candidate set is empty")
    return {
        "ideal": ideal,
        "constrained_ideal": constrained,
        "lower_bound": lower_bound,
        "floor_record": floor_record,
        "ceil_record": ceil_record,
        "candidate_records": tuple(filtered),
    }


def choose_row_scale(records: Iterable[int], scorer: Callable[[int], Fraction]) -> int:
    candidates = tuple(records)
    require(bool(candidates), "empty row-scale choice")
    return min(candidates, key=lambda record: (scorer(record), scale32_value(record), record))


def update_row_scale(
    row: Sequence[Fraction],
    assignments: Sequence[int],
    codepoints: Sequence[int],
    prior_record: int,
) -> tuple[int, dict[str, Any]]:
    require(len(row) == len(assignments), "row-scale assignment width differs")
    if all(weight == 0 for weight in row):
        zero_index = nearest_codepoint_index(Fraction(0), codepoints)
        require(all(index == zero_index for index in assignments), "all-zero row did not use nearest-zero codepoint")
        return SCALE32_ALL_ZERO_RECORD, {
            "all_zero": True,
            "selected_record": SCALE32_ALL_ZERO_RECORD,
            "nearest_zero_index": zero_index,
        }
    points = validate_codebook(codepoints)
    selected_points = [points[index] for index in assignments]
    sum_codepoint_squared = sum(point * point for point in selected_points)
    require(sum_codepoint_squared > 0, "nonzero row has zero assigned-codepoint energy")
    sum_weight_codepoint = sum(
        (weight * point for weight, point in zip(row, selected_points, strict=True)),
        Fraction(0),
    )
    ideal = Fraction(16) * sum_weight_codepoint / sum_codepoint_squared
    maximum = max(abs(weight) for weight in row)
    lower_bound = maximum * 16 / 127
    candidates = candidate_row_scale_records(ideal, lower_bound, prior_record)

    def fixed_assignment_sse(record: int) -> Fraction:
        scale = scale32_value(record)
        return sum(
            (
                weight - scale * point / 16
            ) ** 2
            for weight, point in zip(row, selected_points, strict=True)
        )

    selected = choose_row_scale(candidates["candidate_records"], fixed_assignment_sse)
    return selected, {
        "all_zero": False,
        "ideal": fraction_record(ideal),
        "constrained_ideal": fraction_record(candidates["constrained_ideal"]),
        "lower_bound": fraction_record(lower_bound),
        "floor_record": candidates["floor_record"],
        "ceil_record": candidates["ceil_record"],
        "candidate_records": list(candidates["candidate_records"]),
        "selected_record": selected,
        "selected_sse": fraction_record(fixed_assignment_sse(selected)),
    }


def strict_codebook_dp(costs: Sequence[dict[int, Fraction]]) -> tuple[tuple[int, ...], Fraction]:
    require(bool(costs), "empty strict-codebook DP")
    states: dict[int, tuple[Fraction, tuple[int, ...]]] = {
        point: (cost, (point,)) for point, cost in sorted(costs[0].items())
    }
    require(bool(states), "first strict-codebook cluster has no legal value")
    for cluster in costs[1:]:
        previous = sorted(states.items())
        current: dict[int, tuple[Fraction, tuple[int, ...]]] = {}
        best: tuple[Fraction, tuple[int, ...]] | None = None
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


def update_codebook(
    rows: Sequence[Sequence[Fraction]],
    records: Sequence[int],
    assignments: Sequence[Sequence[int]],
    prior_codebook: Sequence[int],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    prior = validate_codebook(prior_codebook)
    counts = [0] * CODEBOOK_ENTRIES
    constants = [Fraction(0) for _ in range(CODEBOOK_ENTRIES)]
    linears = [Fraction(0) for _ in range(CODEBOOK_ENTRIES)]
    quadratics = [Fraction(0) for _ in range(CODEBOOK_ENTRIES)]
    for row, record, indices in zip(rows, records, assignments, strict=True):
        scale_over_16 = scale32_value(record) / 16
        for weight, index in zip(row, indices, strict=True):
            counts[index] += 1
            constants[index] += weight * weight
            linears[index] += weight * scale_over_16
            quadratics[index] += scale_over_16 * scale_over_16
    costs: list[dict[int, Fraction]] = []
    for index in range(CODEBOOK_ENTRIES):
        allowed = (prior[index],) if counts[index] == 0 else range(Q4_4_MIN, Q4_4_MAX + 1)
        costs.append(
            {
                point: constants[index] - 2 * point * linears[index] + point * point * quadratics[index]
                for point in allowed
            }
        )
    selected, cost = strict_codebook_dp(costs)
    selected = validate_codebook(selected)
    require(
        all(count != 0 or selected[index] == prior[index] for index, count in enumerate(counts)),
        "empty cluster did not retain its prior codepoint",
    )
    return selected, {
        "cluster_counts": counts,
        "empty_cluster_count": sum(count == 0 for count in counts),
        "fixed_assignment_optimum_sse": fraction_record(cost),
    }


def reconstruction_values(
    records: Sequence[int],
    codepoints: Sequence[int],
    assignments: Sequence[Sequence[int]],
) -> list[list[dict[str, int]]]:
    points = validate_codebook(codepoints)
    return [
        [fraction_record(scale32_value(record) * points[index] / 16) for index in row]
        for record, row in zip(records, assignments, strict=True)
    ]


def construct_joint_candidate(rows: Sequence[Sequence[Fraction]]) -> dict[str, Any]:
    bf16_bits = validate_bf16_rows(rows)
    records = tuple(initial_row_scale(row) for row in rows)
    histogram = normalized_histogram(rows, records)
    initial = construct_initial_codebook(histogram)
    codebook = initial["codepoints"]
    assignments = assign_rows(rows, records, codebook)
    previous_sse = source_sse(rows, records, codebook, assignments)
    trace: list[dict[str, Any]] = []

    for iteration in range(1, OUTER_ITERATIONS + 1):
        updated_records: list[int] = []
        row_updates: list[dict[str, Any]] = []
        for row, row_assignments, prior_record in zip(rows, assignments, records, strict=True):
            selected, update = update_row_scale(row, row_assignments, codebook, prior_record)
            updated_records.append(selected)
            row_updates.append(update)
        records = tuple(updated_records)
        assignments_after_scale = assign_rows(rows, records, codebook)
        after_scale_sse = source_sse(rows, records, codebook, assignments_after_scale)
        require(after_scale_sse <= previous_sse, "row-scale update increased source-space SSE")

        codebook, codebook_update = update_codebook(rows, records, assignments_after_scale, codebook)
        assignments = assign_rows(rows, records, codebook)
        final_sse = source_sse(rows, records, codebook, assignments)
        require(final_sse <= after_scale_sse, "codebook update increased source-space SSE")
        require(final_sse <= previous_sse, "outer iteration increased source-space SSE")
        trace.append(
            {
                "iteration": iteration,
                "prior_sse": fraction_record(previous_sse),
                "after_row_scale_sse": fraction_record(after_scale_sse),
                "final_sse": fraction_record(final_sse),
                "row_scale_records": list(records),
                "row_scale_updates_sha256": canonical_sha256(row_updates),
                "codepoints": list(codebook),
                "codebook_update": codebook_update,
                "payload_sha256": hashlib.sha256(pack_indices(assignments)).hexdigest(),
            }
        )
        previous_sse = final_sse

    require(len(trace) == OUTER_ITERATIONS, "joint constructor did not execute exactly eight iterations")
    payload = pack_indices(assignments)
    scale_bytes = b"".join(struct.pack("<I", record) for record in records)
    codebook_bytes = bytes(point & 0xFF for point in codebook)
    reconstructed = reconstruction_values(records, codebook, assignments)
    body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "fixture_shape": [len(rows), len(rows[0])],
        "source_bf16_bits": bf16_bits,
        "source_bf16_sha256": hashlib.sha256(
            b"".join(struct.pack("<H", bits) for row in bf16_bits for bits in row)
        ).hexdigest(),
        "initialization": {
            "row_scale_records": [initial_row_scale(row) for row in rows],
            "histogram_sha256": initial["histogram_sha256"],
            "lloyd_max_iterations": initial["iterations_completed"],
            "lloyd_max_trace_sha256": initial["trace_sha256"],
        },
        "outer_iterations_completed": len(trace),
        "final_row_scale_records": list(records),
        "final_codepoints": list(codebook),
        "payload_hex": payload.hex(),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "row_scale_stream_sha256": hashlib.sha256(scale_bytes).hexdigest(),
        "codebook_record_sha256": hashlib.sha256(codebook_bytes).hexdigest(),
        "reconstruction_sha256": canonical_sha256(reconstructed),
        "final_source_space_sse": fraction_record(previous_sse),
        "iteration_trace": trace,
        "iteration_trace_sha256": canonical_sha256(trace),
    }
    return {**body, "construction_sha256": canonical_sha256(body)}


def faithful_fixture() -> tuple[tuple[Fraction, ...], ...]:
    points = tuple(range(-112, 113, 16))
    rows: list[tuple[Fraction, ...]] = []
    scales = (Fraction(1, 8), Fraction(1, 16), Fraction(1, 32))
    for row_index, scale in enumerate(scales):
        values: list[Fraction] = []
        for point_index, point in enumerate(points):
            if abs(point) == 112:
                offset = 0
            elif row_index == 0:
                offset = 1 if point >= 0 else -1
            elif row_index == 1:
                offset = -1 if point_index & 1 else 1
            else:
                offset = (1, 0, -1)[point_index % 3]
            values.append(scale * Fraction(2 * point + offset, 32))
        rows.append(tuple(values))
    rows.append(tuple(Fraction(0) for _ in points))
    return tuple(rows)


def self_test() -> dict[str, Any]:
    environment = require_project_python()
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "frozen V4 Engineer authority differs")
    root_existed_before = OFFICIAL_ROOT.exists()
    marker_existed_before = OFFICIAL_MARKER.exists()
    require(not marker_existed_before, "V4 execution marker already exists")

    rounding = {
        "positive_even": round_fraction_ties_to_even(Fraction(5, 2)),
        "positive_odd": round_fraction_ties_to_even(Fraction(7, 2)),
        "negative_even": round_fraction_ties_to_even(Fraction(-5, 2)),
        "negative_odd": round_fraction_ties_to_even(Fraction(-7, 2)),
    }
    require(rounding == {"positive_even": 2, "positive_odd": 4, "negative_even": -2, "negative_odd": -4}, "ties-to-even differs")
    require(nearest_codepoint_index(Fraction(-120), INITIAL_CODEPOINTS) == 0, "assignment tie did not choose lower index")

    prior = ceil_scale32_from_ratio(1, 8)
    candidates = candidate_row_scale_records(Fraction(3, 20), Fraction(1, 10), prior)
    require(scale32_value(candidates["floor_record"]) <= Fraction(3, 20), "Scale32 floor is above target")
    require(scale32_value(candidates["ceil_record"]) >= Fraction(3, 20), "Scale32 ceil is below target")
    require(prior in candidates["candidate_records"], "prior Scale32 record was omitted")
    first, second = candidates["candidate_records"][:2]
    tie_choice = choose_row_scale((second, first), lambda _record: Fraction(1))
    require(tie_choice == min((first, second), key=lambda record: (scale32_value(record), record)), "row-scale tie-break differs")
    overflow_rejected = expect_rejected(
        lambda: candidate_row_scale_records(Fraction(1), scale32_value(pack_scale32(SCALE32_SIGNIFICAND_MAX, SCALE32_EXPONENT_MAX)) + 1, prior),
        "illegal no-overflow lower bound was accepted",
    )

    zero_row = (Fraction(0),) * 5
    zero_assignments = assign_rows((zero_row,), (SCALE32_ALL_ZERO_RECORD,), INITIAL_CODEPOINTS)[0]
    zero_record, zero_update = update_row_scale(zero_row, zero_assignments, INITIAL_CODEPOINTS, SCALE32_ALL_ZERO_RECORD)
    require(zero_record == SCALE32_ALL_ZERO_RECORD and zero_update["all_zero"] is True, "all-zero row rule differs")

    tie_costs = [{point: Fraction(0) for point in range(-2, 3)} for _ in range(3)]
    tie_points, tie_cost = strict_codebook_dp(tie_costs)
    require(tie_points == (-2, -1, 0) and tie_cost == 0, "lexicographic strict-codebook tie-break differs")
    empty_costs = [
        {-2: Fraction(0), -1: Fraction(1)},
        {0: Fraction(0)},
        {1: Fraction(1), 2: Fraction(0)},
    ]
    empty_points, _empty_cost = strict_codebook_dp(empty_costs)
    require(empty_points == (-2, 0, 2), "empty-cluster hard constraint differs")
    duplicate_rejected = expect_rejected(
        lambda: validate_codebook((*INITIAL_CODEPOINTS[:-1], INITIAL_CODEPOINTS[-2])),
        "duplicate codebook point was accepted",
    )
    range_rejected = expect_rejected(
        lambda: validate_codebook((*INITIAL_CODEPOINTS[:-1], 128)),
        "out-of-range codebook point was accepted",
    )
    packed = pack_indices(((0, 1, 15, 8, 3),))
    require(packed == bytes((0x10, 0x8F, 0x03)), "nibble packing differs")
    hash_mismatch_rejected = expect_rejected(
        lambda: require_digest("0" * 64, "1" * 64, "bound artifact hash differs"),
        "bound hash mismatch was accepted",
    )

    fixture = faithful_fixture()
    candidate_a = construct_joint_candidate(fixture)
    candidate_b = construct_joint_candidate(fixture)
    require(candidate_a == candidate_b, "independent V4 constructions differ")
    require(candidate_a["outer_iterations_completed"] == OUTER_ITERATIONS, "V4 iteration count differs")
    require(
        all(
            Fraction(item["final_sse"]["numerator"], item["final_sse"]["denominator"])
            <= Fraction(item["prior_sse"]["numerator"], item["prior_sse"]["denominator"])
            for item in candidate_a["iteration_trace"]
        ),
        "V4 monotonic iteration trace differs",
    )
    require(OFFICIAL_ROOT.exists() == root_existed_before, "self-test changed the V4 namespace")
    require(OFFICIAL_MARKER.exists() == marker_existed_before is False, "self-test changed the V4 execution marker")

    return {
        "schema_version": 1,
        "status": "PASS",
        "candidate_id": CANDIDATE_ID,
        "mission_id": MISSION_ID,
        "task_id": TASK_ID,
        "environment": environment,
        "runner_sha256": sha256_file(RUNNER_PATH),
        "task_sha256": sha256_file(TASK_PATH),
        "plan_sha256": sha256_file(PLAN_PATH),
        "checks": {
            "exact_bf16_fixture": True,
            "ties_to_even": rounding,
            "lower_index_assignment_tie": True,
            "scale32_bracketing": True,
            "scale32_immediate_neighbors": True,
            "scale32_tie_order": ["source_space_sse", "smaller_positive_scale", "smaller_packed_record"],
            "scale32_overflow_rejected": overflow_rejected,
            "all_zero_row_rule": True,
            "strict_codebook_dp_optimum": True,
            "strict_codebook_empty_cluster": True,
            "strict_codebook_lexicographic_tie": True,
            "duplicate_codepoint_rejected": duplicate_rejected,
            "range_rejected": range_rejected,
            "nibble_packing_hex": packed.hex(),
            "bound_hash_mismatch_rejected": hash_mismatch_rejected,
            "independent_candidate_a_b_match": True,
            "outer_iterations_completed": OUTER_ITERATIONS,
            "monotonic_source_space_sse": True,
        },
        "construction": candidate_a,
        "official_build_namespace_changed": False,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": False,
        "full_model_preflight_exposed": False,
    }


def write_exclusive(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def write_smoke_artifacts(output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), "smoke output path already exists")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = self_test()
    events = b"".join(canonical_bytes(event) for event in report["construction"]["iteration_trace"])
    manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "mode": "exact_reference_constructor_smoke",
        "runner": {"path": RUNNER_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(RUNNER_PATH)},
        "task": {"path": TASK_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(TASK_PATH)},
        "plan": {"path": PLAN_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(PLAN_PATH)},
        "artifact_files": list(ARTIFACT_FILENAMES),
        "official_inputs_accessed": False,
        "official_execution_marker_created": False,
        "full_model_preflight_claimed": False,
    }
    status = {
        "schema_version": 1,
        "status": "PASS",
        "candidate_id": CANDIDATE_ID,
        "construction_sha256": report["construction"]["construction_sha256"],
        "iteration_trace_sha256": report["construction"]["iteration_trace_sha256"],
        "outer_iterations_completed": report["construction"]["outer_iterations_completed"],
        "attempts_consumed": 0,
    }
    log = (
        "V4 exact reference constructor smoke PASS\n"
        f"construction_sha256={status['construction_sha256']}\n"
        f"iteration_trace_sha256={status['iteration_trace_sha256']}\n"
        "official_attempts_consumed=0\n"
    ).encode()
    payloads = {
        "manifest.json": canonical_bytes(manifest),
        "status.json": canonical_bytes(status),
        "raw_events.jsonl": events,
        "runner.log": log,
        "report.json": canonical_bytes(report),
    }
    for name in ARTIFACT_FILENAMES:
        write_exclusive(output_dir / name, payloads[name])
    sums = "".join(f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n" for name in ARTIFACT_FILENAMES)
    write_exclusive(output_dir / "SHA256SUMS", sums.encode())
    return verify_smoke_artifacts(output_dir)


def verify_smoke_artifacts(output_dir: Path) -> dict[str, Any]:
    require(output_dir.is_dir(), "smoke artifact directory is missing")
    expected_names = set(ARTIFACT_FILENAMES) | {"SHA256SUMS"}
    observed_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    require(observed_names == expected_names, "smoke artifact file set differs")
    expected_sums = "".join(
        f"{sha256_file(output_dir / name)}  {name}\n" for name in ARTIFACT_FILENAMES
    )
    require((output_dir / "SHA256SUMS").read_text(encoding="utf-8") == expected_sums, "smoke artifact hashes differ")
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    require(report == self_test(), "smoke report differs from a fresh deterministic rebuild")
    status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    require(status["status"] == "PASS" and status["attempts_consumed"] == 0, "smoke status differs")
    require(not OFFICIAL_MARKER.exists(), "official marker exists after smoke verification")
    return {
        "status": "PASS",
        "candidate_id": CANDIDATE_ID,
        "artifact_count": len(expected_names),
        "construction_sha256": status["construction_sha256"],
        "iteration_trace_sha256": status["iteration_trace_sha256"],
        "attempts_consumed": 0,
        "official_execution_mode_exposed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "smoke", "prepare", "verify"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.command == "self-test":
        require(args.output_dir is None, "self-test does not accept --output-dir")
        from v4_joint_full_model_backend import combined_self_test

        result = combined_self_test()
        print("ACE2_OPTION_B_JOINT_CODEBOOK_V4_SELF_TEST_PASS " + json.dumps(result, sort_keys=True))
        return 0
    if args.command == "prepare":
        require(args.output_dir is None, "prepare does not accept --output-dir")
        from v4_joint_full_model_backend import prepare, record_preparation_failure

        try:
            result = prepare()
        except Exception as exc:
            record_preparation_failure(exc)
            raise
        print("ACE2_OPTION_B_JOINT_CODEBOOK_V4_PREFLIGHT_COMPLETE " + json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PREATTEMPT_READY" else 2
    if args.command == "verify" and args.output_dir is None:
        from v4_joint_full_model_backend import load_verified_closure

        result = load_verified_closure()
        print("ACE2_OPTION_B_JOINT_CODEBOOK_V4_PREFLIGHT_VERIFIED " + json.dumps(result, sort_keys=True))
        return 0
    require(args.output_dir is not None, f"{args.command} requires --output-dir")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    result = write_smoke_artifacts(output_dir) if args.command == "smoke" else verify_smoke_artifacts(output_dir)
    print(f"ACE2_OPTION_B_JOINT_CODEBOOK_V4_{args.command.upper()}_PASS " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
