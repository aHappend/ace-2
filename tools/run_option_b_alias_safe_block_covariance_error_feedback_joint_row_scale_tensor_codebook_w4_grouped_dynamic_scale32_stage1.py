#!/usr/bin/env python3
"""Exact marker-free arithmetic core for the block-covariance V6 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Sequence

import run_option_b_alias_safe_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_stage1 as v4
from ace2_quality_contracts import (
    SCALE32_ALL_ZERO_RECORD,
    SCALE32_EXPONENT_MAX,
    SCALE32_EXPONENT_MIN,
    SCALE32_SIGNIFICAND_MAX,
    SCALE32_SIGNIFICAND_MIN,
    pack_scale32,
    unpack_scale32,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "option_b_alias_safe_block_covariance_error_feedback_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_w4a8_v6"
MISSION_ID = "5551a459e416"
TASK_ID = "task-0394d005ed34"
TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_BLOCK_COVARIANCE_ERROR_FEEDBACK_JOINT_ROW_SCALE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V6_ENGINEER_TASK.json"
TASK_COMPANION = TASK_PATH.with_suffix(".sha256")
PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_BLOCK_COVARIANCE_ERROR_FEEDBACK_JOINT_ROW_SCALE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V6_PLAN.json"
PLAN_COMPANION = PLAN_PATH.with_suffix(".sha256")
OFFICIAL_ROOT = ROOT / "build/stage1-option-b-alias-safe-block-covariance-error-feedback-joint-row-scale-tensor-codebook-w4-grouped-dynamic-scale32-w4a8-v6"
OFFICIAL_MARKER = OFFICIAL_ROOT / "attempt-0001/execution_started.json"
RUNNER_PATH = Path(__file__).resolve()

BLOCK_SIZE = 16
OUTER_SWEEPS = 4
Q4_4_MIN = v4.Q4_4_MIN
Q4_4_MAX = v4.Q4_4_MAX
CODEBOOK_ENTRIES = v4.CODEBOOK_ENTRIES


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


def verify_companion(path: Path, companion: Path) -> None:
    expected = f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}"
    require(companion.is_file(), f"checksum companion is missing: {companion}")
    require(
        companion.read_text(encoding="utf-8").strip() == expected,
        f"checksum companion differs: {path}",
    )


def require_project_python() -> dict[str, str]:
    expected = ROOT / ".venv/bin/python"
    require(expected.is_file(), "project Python is missing")
    require(os.path.samefile(Path(sys.executable), expected), "wrong Python executable")
    return {"entrypoint": "./.venv/bin/python", "python": platform.python_version()}


def expect_rejected(call: Callable[[], Any], message: str) -> bool:
    try:
        call()
    except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
        return True
    raise RuntimeError(message)


def require_digest(actual: str, expected: str, message: str) -> None:
    require(actual == expected, message)


def _dyadic_denominator_exponent(value: Fraction) -> int:
    denominator = value.denominator
    require(denominator > 0 and denominator & (denominator - 1) == 0, "non-dyadic exact value")
    return denominator.bit_length() - 1


def normalize_gram(matrix: Sequence[Sequence[int]]) -> tuple[tuple[tuple[int, ...], ...], int]:
    rows = tuple(tuple(row) for row in matrix)
    require(bool(rows) and len(rows) <= BLOCK_SIZE, "Gram width differs")
    require(all(len(row) == len(rows) for row in rows), "Gram matrix is not square")
    require(all(type(value) is int for row in rows for value in row), "Gram entry is not an integer")
    require(
        all(rows[i][j] == rows[j][i] for i in range(len(rows)) for j in range(len(rows))),
        "Gram matrix is not symmetric",
    )
    require(all(rows[index][index] >= 0 for index in range(len(rows))), "Gram diagonal is negative")
    divisor = math.gcd(*(abs(value) for row in rows for value in row)) or 1
    normalized = tuple(tuple(value // divisor for value in row) for row in rows)
    require(
        math.gcd(*(abs(value) for row in normalized for value in row)) in (0, 1),
        "Gram block was not completely GCD-reduced",
    )
    return normalized, divisor


def capture_exact_gram_blocks(
    positions: Sequence[Sequence[Fraction]], block_size: int = BLOCK_SIZE
) -> dict[str, Any]:
    require(block_size == BLOCK_SIZE, "V6 covariance block size differs")
    bf16_bits = v4.validate_bf16_rows(positions)
    width = len(positions[0])
    blocks: list[dict[str, Any]] = []
    for start in range(0, width, block_size):
        stop = min(start + block_size, width)
        live = [tuple(row[start:stop]) for row in positions]
        denominator_exponent = max(
            _dyadic_denominator_exponent(value) for row in live for value in row
        )
        integer_rows: list[tuple[int, ...]] = []
        for row in live:
            scaled = tuple(value * (1 << denominator_exponent) for value in row)
            require(all(value.denominator == 1 for value in scaled), "BF16 lattice conversion differs")
            integer_rows.append(tuple(value.numerator for value in scaled))
        gram = tuple(
            tuple(
                sum(row[left] * row[right] for row in integer_rows)
                for right in range(stop - start)
            )
            for left in range(stop - start)
        )
        normalized, divisor = normalize_gram(gram)
        gram_strings = [[str(value) for value in row] for row in normalized]
        blocks.append(
            {
                "start_channel": start,
                "live_channels": stop - start,
                "activation_lattice_binary_exponent": -denominator_exponent,
                "gram_lattice_binary_exponent": -2 * denominator_exponent,
                "whole_block_gcd": str(divisor),
                "normalized_gram": gram_strings,
                "normalized_gram_sha256": canonical_sha256(gram_strings),
            }
        )
    body = {
        "schema_version": 1,
        "block_size": block_size,
        "input_channels": width,
        "captured_position_count": len(positions),
        "source_bf16_sha256": hashlib.sha256(
            b"".join(struct.pack("<H", bits) for row in bf16_bits for bits in row)
        ).hexdigest(),
        "blocks": blocks,
        "floating_point_accumulation_used": False,
    }
    return {**body, "covariance_manifest_sha256": canonical_sha256(body)}


def gram_from_record(record: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    width = int(record["live_channels"])
    gram = tuple(tuple(int(value) for value in row) for row in record["normalized_gram"])
    normalized, divisor = normalize_gram(gram)
    require(len(normalized) == width and divisor == 1, "stored Gram block normalization differs")
    return normalized


def damping_for_gram(gram: Sequence[Sequence[int]]) -> int:
    matrix, _divisor = normalize_gram(gram)
    trace = sum(matrix[index][index] for index in range(len(matrix)))
    require(trace >= 0, "Gram trace is negative")
    denominator = 1000 * len(matrix)
    return max(1, (trace + denominator - 1) // denominator)


def damp_gram(gram: Sequence[Sequence[int]]) -> tuple[tuple[Fraction, ...], ...]:
    matrix, _divisor = normalize_gram(gram)
    damping = damping_for_gram(matrix)
    return tuple(
        tuple(Fraction(value + (damping if row == column else 0)) for column, value in enumerate(values))
        for row, values in enumerate(matrix)
    )


def exact_ldlt(
    matrix: Sequence[Sequence[int | Fraction]],
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    values = tuple(tuple(Fraction(value) for value in row) for row in matrix)
    size = len(values)
    require(size > 0 and all(len(row) == size for row in values), "LDLT matrix shape differs")
    require(
        all(values[i][j] == values[j][i] for i in range(size) for j in range(size)),
        "LDLT matrix is not symmetric",
    )
    lower = [[Fraction(int(row == column)) for column in range(size)] for row in range(size)]
    diagonal = [Fraction(0) for _ in range(size)]
    for column in range(size):
        pivot = values[column][column] - sum(
            lower[column][prior] * lower[column][prior] * diagonal[prior]
            for prior in range(column)
        )
        require(pivot > 0, f"LDLT nonpositive pivot at channel {column}")
        diagonal[column] = pivot
        for row in range(column + 1, size):
            residual = values[row][column] - sum(
                lower[row][prior] * lower[column][prior] * diagonal[prior]
                for prior in range(column)
            )
            lower[row][column] = residual / pivot
    return tuple(tuple(row) for row in lower), tuple(diagonal)


def inverse_from_ldlt(
    lower: Sequence[Sequence[Fraction]], diagonal: Sequence[Fraction]
) -> tuple[tuple[Fraction, ...], ...]:
    size = len(diagonal)
    require(size > 0 and len(lower) == size, "inverse LDLT shape differs")
    inverse_columns: list[list[Fraction]] = []
    for target in range(size):
        forward = [Fraction(0) for _ in range(size)]
        for row in range(size):
            forward[row] = Fraction(int(row == target)) - sum(
                lower[row][column] * forward[column] for column in range(row)
            )
        scaled = [forward[index] / diagonal[index] for index in range(size)]
        solution = [Fraction(0) for _ in range(size)]
        for row in range(size - 1, -1, -1):
            solution[row] = scaled[row] - sum(
                lower[column][row] * solution[column] for column in range(row + 1, size)
            )
        inverse_columns.append(solution)
    inverse = tuple(
        tuple(inverse_columns[column][row] for column in range(size)) for row in range(size)
    )
    require(
        all(inverse[i][j] == inverse[j][i] for i in range(size) for j in range(size)),
        "exact inverse lost symmetry",
    )
    return inverse


def inverse_damped_gram(gram: Sequence[Sequence[int]]) -> tuple[tuple[Fraction, ...], ...]:
    lower, diagonal = exact_ldlt(damp_gram(gram))
    inverse = inverse_from_ldlt(lower, diagonal)
    require(all(inverse[index][index] > 0 for index in range(len(inverse))), "inverse diagonal is nonpositive")
    return inverse


def _block_ranges(width: int, grams: Sequence[Sequence[Sequence[int]]]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for gram in grams:
        live = len(gram)
        require(1 <= live <= BLOCK_SIZE and all(len(row) == live for row in gram), "Gram partition differs")
        ranges.append((start, start + live))
        start += live
    require(start == width, "Gram blocks do not cover the complete input width")
    return tuple(ranges)


def block_objective_row(
    row: Sequence[Fraction],
    record: int,
    codepoints: Sequence[int],
    assignments: Sequence[int],
    grams: Sequence[Sequence[Sequence[int]]],
) -> Fraction:
    points = v4.validate_codebook(codepoints)
    require(len(row) == len(assignments), "block-objective assignment width differs")
    scale = v4.scale32_value(record)
    total = Fraction(0)
    for gram, (start, stop) in zip(grams, _block_ranges(len(row), grams), strict=True):
        errors = [row[index] - scale * points[assignments[index]] / 16 for index in range(start, stop)]
        total += sum(
            errors[left] * int(gram[left][right]) * errors[right]
            for left in range(stop - start)
            for right in range(stop - start)
        )
    return total


def block_objective_rows(
    rows: Sequence[Sequence[Fraction]],
    records: Sequence[int],
    codepoints: Sequence[int],
    assignments: Sequence[Sequence[int]],
    grams: Sequence[Sequence[Sequence[int]]],
) -> Fraction:
    require(len(rows) == len(records) == len(assignments), "block-objective row count differs")
    return sum(
        (
            block_objective_row(row, record, codepoints, indices, grams)
            for row, record, indices in zip(rows, records, assignments, strict=True)
        ),
        Fraction(0),
    )


def sequential_error_feedback_assign(
    row: Sequence[Fraction],
    record: int,
    codepoints: Sequence[int],
    grams: Sequence[Sequence[Sequence[int]]],
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    points = v4.validate_codebook(codepoints)
    scale = v4.scale32_value(record)
    reconstructions = tuple(scale * point / 16 for point in points)
    assigned: list[int] = []
    trace: list[dict[str, Any]] = []
    for gram, (start, stop) in zip(grams, _block_ranges(len(row), grams), strict=True):
        inverse = inverse_damped_gram(gram)
        compensated = list(row[start:stop])
        block_indices: list[int] = []
        block_trace: list[dict[str, Any]] = []
        for channel in range(stop - start):
            index = min(
                range(CODEBOOK_ENTRIES),
                key=lambda candidate: (abs(compensated[channel] - reconstructions[candidate]), candidate),
            )
            quantized = reconstructions[index]
            error = compensated[channel] - quantized
            denominator = inverse[channel][channel]
            require(denominator > 0, "inverse-Hessian diagonal is nonpositive")
            updates: list[dict[str, Any]] = []
            for later in range(channel + 1, stop - start):
                correction = error * inverse[channel][later] / denominator
                compensated[later] -= correction
                updates.append(
                    {
                        "channel": start + later,
                        "correction": v4.fraction_record(correction),
                    }
                )
            block_indices.append(index)
            block_trace.append(
                {
                    "channel": start + channel,
                    "selected_index": index,
                    "compensated_source": v4.fraction_record(compensated[channel]),
                    "quantized": v4.fraction_record(quantized),
                    "quantization_error": v4.fraction_record(error),
                    "later_updates": updates,
                }
            )
        assigned.extend(block_indices)
        trace.append(
            {
                "start_channel": start,
                "live_channels": stop - start,
                "damping": damping_for_gram(gram),
                "assignment_trace": block_trace,
            }
        )
    return tuple(assigned), trace


def assign_rows_error_feedback(
    rows: Sequence[Sequence[Fraction]],
    records: Sequence[int],
    codepoints: Sequence[int],
    grams: Sequence[Sequence[Sequence[int]]],
) -> tuple[tuple[tuple[int, ...], ...], list[list[dict[str, Any]]]]:
    require(len(rows) == len(records), "error-feedback row-scale count differs")
    assignments: list[tuple[int, ...]] = []
    traces: list[list[dict[str, Any]]] = []
    for row, record in zip(rows, records, strict=True):
        indices, trace = sequential_error_feedback_assign(row, record, codepoints, grams)
        assignments.append(indices)
        traces.append(trace)
    return tuple(assignments), traces


def update_row_scale_block(
    row: Sequence[Fraction],
    assignments: Sequence[int],
    codepoints: Sequence[int],
    grams: Sequence[Sequence[Sequence[int]]],
    prior_record: int,
) -> tuple[int, dict[str, Any]]:
    require(len(row) == len(assignments), "block row-scale assignment width differs")
    if all(weight == 0 for weight in row):
        return SCALE32_ALL_ZERO_RECORD, {
            "all_zero": True,
            "selected_record": SCALE32_ALL_ZERO_RECORD,
        }
    points = v4.validate_codebook(codepoints)
    linear = Fraction(0)
    quadratic = Fraction(0)
    for gram, (start, stop) in zip(grams, _block_ranges(len(row), grams), strict=True):
        selected = [Fraction(points[assignments[index]], 16) for index in range(start, stop)]
        source = row[start:stop]
        linear += sum(
            source[left] * int(gram[left][right]) * selected[right]
            for left in range(stop - start)
            for right in range(stop - start)
        )
        quadratic += sum(
            selected[left] * int(gram[left][right]) * selected[right]
            for left in range(stop - start)
            for right in range(stop - start)
        )
    require(quadratic > 0, "nonzero block row has zero assigned-codepoint quadratic")
    ideal = linear / quadratic
    lower_bound = max(abs(weight) for weight in row) * 16 / 127
    candidates = v4.candidate_row_scale_records(ideal, lower_bound, prior_record)

    def score(record: int) -> Fraction:
        return block_objective_row(row, record, points, assignments, grams)

    selected_record = v4.choose_row_scale(candidates["candidate_records"], score)
    return selected_record, {
        "all_zero": False,
        "linear": v4.fraction_record(linear),
        "quadratic": v4.fraction_record(quadratic),
        "ideal": v4.fraction_record(ideal),
        "constrained_ideal": v4.fraction_record(candidates["constrained_ideal"]),
        "lower_bound": v4.fraction_record(lower_bound),
        "candidate_records": list(candidates["candidate_records"]),
        "selected_record": selected_record,
        "selected_block_objective": v4.fraction_record(score(selected_record)),
    }


def update_codebook_coordinate_sweep(
    rows: Sequence[Sequence[Fraction]],
    records: Sequence[int],
    assignments: Sequence[Sequence[int]],
    grams: Sequence[Sequence[Sequence[int]]],
    prior_codebook: Sequence[int],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    codebook = list(v4.validate_codebook(prior_codebook))
    counts = [0] * CODEBOOK_ENTRIES
    for row in assignments:
        for index in row:
            require(type(index) is int and 0 <= index < CODEBOOK_ENTRIES, "payload index escaped four bits")
            counts[index] += 1
    trace: list[dict[str, Any]] = []
    for index in range(CODEBOOK_ENTRIES):
        prior = codebook[index]
        if counts[index] == 0:
            trace.append(
                {
                    "index": index,
                    "assignment_count": 0,
                    "prior_value": prior,
                    "selected_value": prior,
                    "empty_assignment_retained": True,
                }
            )
            continue
        lower = Q4_4_MIN if index == 0 else codebook[index - 1] + 1
        upper = Q4_4_MAX if index == CODEBOOK_ENTRIES - 1 else codebook[index + 1] - 1
        require(lower <= prior <= upper, "prior codebook value escaped coordinate range")
        scored: list[tuple[Fraction, int]] = []
        for value in range(lower, upper + 1):
            candidate = tuple(codebook[:index] + [value] + codebook[index + 1 :])
            scored.append(
                (
                    block_objective_rows(rows, records, candidate, assignments, grams),
                    value,
                )
            )
        objective, selected = min(scored, key=lambda item: (item[0], item[1]))
        codebook[index] = selected
        trace.append(
            {
                "index": index,
                "assignment_count": counts[index],
                "prior_value": prior,
                "minimum_legal_value": lower,
                "maximum_legal_value": upper,
                "candidate_count": upper - lower + 1,
                "selected_value": selected,
                "selected_block_objective": v4.fraction_record(objective),
                "empty_assignment_retained": False,
            }
        )
    selected_codebook = v4.validate_codebook(codebook)
    return selected_codebook, {
        "cluster_counts": counts,
        "empty_cluster_count": sum(count == 0 for count in counts),
        "coordinate_trace": trace,
        "coordinate_trace_sha256": canonical_sha256(trace),
    }


def construct_block_candidate(
    rows: Sequence[Sequence[Fraction]], grams: Sequence[Sequence[Sequence[int]]]
) -> dict[str, Any]:
    bf16_bits = v4.validate_bf16_rows(rows)
    width = len(rows[0])
    _block_ranges(width, grams)
    normalized_grams = tuple(normalize_gram(gram)[0] for gram in grams)
    v4_initial = v4.construct_joint_candidate(rows)
    records = tuple(v4_initial["final_row_scale_records"])
    codebook = tuple(v4_initial["final_codepoints"])
    v4_assignments = v4.assign_rows(rows, records, codebook)
    v4_baseline = block_objective_rows(rows, records, codebook, v4_assignments, normalized_grams)
    assignments, assignment_trace = assign_rows_error_feedback(
        rows, records, codebook, normalized_grams
    )
    previous = block_objective_rows(rows, records, codebook, assignments, normalized_grams)
    trace: list[dict[str, Any]] = []

    for sweep in range(1, OUTER_SWEEPS + 1):
        updated_records: list[int] = []
        row_updates: list[dict[str, Any]] = []
        for row, indices, prior_record in zip(rows, assignments, records, strict=True):
            selected, update = update_row_scale_block(
                row, indices, codebook, normalized_grams, prior_record
            )
            updated_records.append(selected)
            row_updates.append(update)
        records = tuple(updated_records)
        assignments_after_scale, scale_assignment_trace = assign_rows_error_feedback(
            rows, records, codebook, normalized_grams
        )
        after_scale = block_objective_rows(
            rows, records, codebook, assignments_after_scale, normalized_grams
        )
        codebook, codebook_update = update_codebook_coordinate_sweep(
            rows, records, assignments_after_scale, normalized_grams, codebook
        )
        assignments, final_assignment_trace = assign_rows_error_feedback(
            rows, records, codebook, normalized_grams
        )
        final = block_objective_rows(rows, records, codebook, assignments, normalized_grams)
        require(final <= previous, "V6 outer sweep increased the exact undamped block objective")
        trace.append(
            {
                "sweep": sweep,
                "prior_block_objective": v4.fraction_record(previous),
                "after_row_scale_block_objective": v4.fraction_record(after_scale),
                "final_block_objective": v4.fraction_record(final),
                "row_scale_records": list(records),
                "row_scale_updates_sha256": canonical_sha256(row_updates),
                "scale_assignment_trace_sha256": canonical_sha256(scale_assignment_trace),
                "codepoints": list(codebook),
                "codebook_update": codebook_update,
                "final_assignment_trace_sha256": canonical_sha256(final_assignment_trace),
                "payload_sha256": hashlib.sha256(v4.pack_indices(assignments)).hexdigest(),
            }
        )
        previous = final

    payload = v4.pack_indices(assignments)
    scale_raw = b"".join(struct.pack("<I", record) for record in records)
    codebook_raw = bytes(point & 0xFF for point in codebook)
    body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "fixture_shape": [len(rows), width],
        "source_bf16_sha256": hashlib.sha256(
            b"".join(struct.pack("<H", bits) for row in bf16_bits for bits in row)
        ).hexdigest(),
        "normalized_gram_sha256": canonical_sha256(normalized_grams),
        "fresh_v4_initialization_sha256": v4_initial["construction_sha256"],
        "fresh_v4_block_objective": v4.fraction_record(v4_baseline),
        "initial_error_feedback_block_objective": v4.fraction_record(
            block_objective_rows(
                rows,
                tuple(v4_initial["final_row_scale_records"]),
                tuple(v4_initial["final_codepoints"]),
                assign_rows_error_feedback(
                    rows,
                    tuple(v4_initial["final_row_scale_records"]),
                    tuple(v4_initial["final_codepoints"]),
                    normalized_grams,
                )[0],
                normalized_grams,
            )
        ),
        "initial_assignment_trace_sha256": canonical_sha256(assignment_trace),
        "outer_sweeps_completed": len(trace),
        "final_row_scale_records": list(records),
        "final_codepoints": list(codebook),
        "payload_hex": payload.hex(),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "row_scale_stream_sha256": hashlib.sha256(scale_raw).hexdigest(),
        "codebook_record_sha256": hashlib.sha256(codebook_raw).hexdigest(),
        "final_block_objective": v4.fraction_record(previous),
        "non_regressing_against_fresh_v4": previous <= v4_baseline,
        "strictly_improved_against_fresh_v4": previous < v4_baseline,
        "iteration_trace": trace,
        "iteration_trace_sha256": canonical_sha256(trace),
    }
    return {**body, "construction_sha256": canonical_sha256(body)}


def _fixture_grams(width: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    blocks: list[tuple[tuple[int, ...], ...]] = []
    for start in range(0, width, BLOCK_SIZE):
        live = min(BLOCK_SIZE, width - start)
        blocks.append(
            tuple(
                tuple(4 if row == column else (1 if abs(row - column) == 1 else 0) for column in range(live))
                for row in range(live)
            )
        )
    return tuple(blocks)


def self_test() -> dict[str, Any]:
    environment = require_project_python()
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    require(
        task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID,
        "frozen V6 Engineer authority differs",
    )
    root_existed_before = OFFICIAL_ROOT.exists()
    marker_existed_before = OFFICIAL_MARKER.exists()
    require(not marker_existed_before, "V6 execution marker already exists")

    capture_positions = (
        (Fraction(1, 2), Fraction(1, 4), Fraction(-1, 2), Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(-1, 4), Fraction(1, 2), Fraction(1, 8), Fraction(-1)),
        (Fraction(-1, 2), Fraction(1, 2), Fraction(0), Fraction(-1, 8), Fraction(1, 2)),
    )
    padded_positions = tuple(row + tuple(Fraction(0) for _ in range(12)) for row in capture_positions)
    capture_a = capture_exact_gram_blocks(padded_positions)
    capture_b = capture_exact_gram_blocks(padded_positions)
    require(capture_a == capture_b, "duplicate exact covariance fixture captures differ")
    require(
        [block["live_channels"] for block in capture_a["blocks"]] == [16, 1],
        "final short covariance block differs",
    )
    captured_grams = tuple(gram_from_record(record) for record in capture_a["blocks"])
    normalized_probe, normalized_probe_gcd = normalize_gram(((6, 3), (3, 9)))
    require(
        normalized_probe == ((2, 1), (1, 3)) and normalized_probe_gcd == 3,
        "whole-block Gram GCD reduction differs",
    )
    require(damping_for_gram(((0, 0), (0, 0))) == 1, "zero-block damping differs")
    singular_damped = damp_gram(((1, 1), (1, 1)))
    singular_lower, singular_diagonal = exact_ldlt(singular_damped)
    singular_inverse = inverse_from_ldlt(singular_lower, singular_diagonal)
    require(all(singular_inverse[index][index] > 0 for index in range(2)), "damped singular inverse differs")
    require(
        expect_rejected(lambda: exact_ldlt(((1, 1), (1, 1))), "singular undamped LDLT was accepted"),
        "singular LDLT rejection differs",
    )
    require(
        expect_rejected(lambda: exact_ldlt(((1, 2), (2, 1))), "negative-pivot LDLT was accepted"),
        "nonpositive-pivot rejection differs",
    )

    identity_gram = ((1, 0), (0, 1))
    codebook = tuple(range(-120, 121, 16))
    tie_scale = v4.scale32_value(v4.initial_row_scale((Fraction(1), Fraction(1))))
    midpoint = tie_scale * Fraction(codebook[7] + codebook[8], 32)
    tie_indices, _tie_trace = sequential_error_feedback_assign(
        (midpoint, midpoint),
        v4.initial_row_scale((Fraction(1), Fraction(1))),
        codebook,
        (identity_gram,),
    )
    require(tie_indices == (7, 7), "lower-index assignment tie differs")

    propagation_gram = ((2, 1), (1, 2))
    propagation_record = pack_scale32(SCALE32_SIGNIFICAND_MIN, 0)
    propagation_row = (Fraction(9, 32), Fraction(9, 32))
    _propagation_indices, propagation_trace = sequential_error_feedback_assign(
        propagation_row,
        propagation_record,
        codebook,
        (propagation_gram,),
    )
    first_update = propagation_trace[0]["assignment_trace"][0]["later_updates"][0]
    require(
        Fraction(
            first_update["correction"]["numerator"],
            first_update["correction"]["denominator"],
        )
        != 0,
        "sequential inverse-Hessian error propagation was not exercised",
    )

    row_scale_row = (Fraction(3, 8), Fraction(-5, 16))
    row_scale_assignments = (8, 7)
    row_scale_gram = (((3, 1), (1, 2)),)
    row_scale_prior = pack_scale32(SCALE32_SIGNIFICAND_MIN, 0)
    row_scale_selected, row_scale_trace = update_row_scale_block(
        row_scale_row,
        row_scale_assignments,
        codebook,
        row_scale_gram,
        row_scale_prior,
    )
    row_scale_candidates = tuple(row_scale_trace["candidate_records"])
    require(
        row_scale_selected
        == min(
            row_scale_candidates,
            key=lambda record: (
                block_objective_row(
                    row_scale_row,
                    record,
                    codebook,
                    row_scale_assignments,
                    row_scale_gram,
                ),
                v4.scale32_value(record),
                record,
            ),
        ),
        "full-block row-scale endpoint or tie selection differs",
    )
    minimum_record = pack_scale32(SCALE32_SIGNIFICAND_MIN, SCALE32_EXPONENT_MIN)
    maximum_record = pack_scale32(SCALE32_SIGNIFICAND_MAX, SCALE32_EXPONENT_MAX)
    require(
        v4.candidate_row_scale_records(Fraction(-1), Fraction(0), row_scale_prior)[
            "constrained_ideal"
        ]
        == v4.scale32_value(minimum_record),
        "row-scale lower endpoint clamp differs",
    )
    require(
        v4.candidate_row_scale_records(
            v4.scale32_value(maximum_record) * 2,
            Fraction(0),
            row_scale_prior,
        )["constrained_ideal"]
        == v4.scale32_value(maximum_record),
        "row-scale upper endpoint clamp differs",
    )

    endpoint_record = pack_scale32(SCALE32_SIGNIFICAND_MIN, 0)
    lower_endpoint_codebook, lower_endpoint_trace = update_codebook_coordinate_sweep(
        ((Fraction(-100),),),
        (endpoint_record,),
        ((0,),),
        (((1,),),),
        codebook,
    )
    require(lower_endpoint_codebook[0] == Q4_4_MIN, "codebook lower endpoint optimum differs")
    require(
        all(
            item["selected_value"] == item["prior_value"]
            for item in lower_endpoint_trace["coordinate_trace"][1:]
        ),
        "empty codebook assignments did not retain prior values",
    )
    upper_endpoint_codebook, _upper_endpoint_trace = update_codebook_coordinate_sweep(
        ((Fraction(100),),),
        (endpoint_record,),
        ((15,),),
        (((1,),),),
        codebook,
    )
    require(upper_endpoint_codebook[-1] == Q4_4_MAX, "codebook upper endpoint optimum differs")
    tie_codebook, tie_codebook_trace = update_codebook_coordinate_sweep(
        ((Fraction(-15, 32),),),
        (endpoint_record,),
        ((7,),),
        (((1,),),),
        codebook,
    )
    require(tie_codebook[7] == -8, "codebook lower-signed-value tie differs")
    require(
        tie_codebook_trace["coordinate_trace"][7]["minimum_legal_value"]
        < tie_codebook[7]
        < tie_codebook_trace["coordinate_trace"][7]["maximum_legal_value"],
        "codebook tie fixture did not exercise an interior coordinate",
    )
    require(v4.validate_codebook(tie_codebook) == tie_codebook, "codebook strict order differs")

    fixture = v4.faithful_fixture()
    grams = _fixture_grams(len(fixture[0]))
    candidate_a = construct_block_candidate(fixture, grams)
    candidate_b = construct_block_candidate(fixture, grams)
    require(candidate_a == candidate_b, "independent V6 fixture constructions differ")
    require(candidate_a["outer_sweeps_completed"] == OUTER_SWEEPS, "V6 sweep count differs")
    require(
        candidate_a["non_regressing_against_fresh_v4"] is True,
        "V6 fixture regressed against its fresh V4 baseline",
    )
    require(
        candidate_a["strictly_improved_against_fresh_v4"] is True,
        "V6 fixture did not strictly improve against its fresh V4 baseline",
    )
    require(
        all(
            Fraction(item["final_block_objective"]["numerator"], item["final_block_objective"]["denominator"])
            <= Fraction(item["prior_block_objective"]["numerator"], item["prior_block_objective"]["denominator"])
            for item in candidate_a["iteration_trace"]
        ),
        "V6 fixture monotonicity differs",
    )
    require(
        expect_rejected(
            lambda: normalize_gram(((1, 2), (3, 1))),
            "asymmetric Gram fixture was accepted",
        ),
        "asymmetric Gram rejection differs",
    )
    require(
        expect_rejected(
            lambda: require_digest("00", "11", "synthetic hash mismatch"),
            "hash mismatch fixture was accepted",
        ),
        "hash mismatch rejection differs",
    )
    packed = v4.pack_indices(((0, 15, 1),))
    require(packed == bytes((0xF0, 0x01)), "nibble packing differs")
    for record in candidate_a["final_row_scale_records"]:
        significand, exponent = unpack_scale32(record)
        require(
            SCALE32_SIGNIFICAND_MIN <= significand <= SCALE32_SIGNIFICAND_MAX
            and SCALE32_EXPONENT_MIN <= exponent <= SCALE32_EXPONENT_MAX,
            "fixture Scale32 record escaped the legal range",
        )
    require(OFFICIAL_ROOT.exists() == root_existed_before, "self-test changed the V6 namespace")
    require(OFFICIAL_MARKER.exists() == marker_existed_before is False, "self-test changed the V6 marker")

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
            "duplicate_exact_covariance_capture": True,
            "whole_block_gcd_reduction": True,
            "final_short_block": True,
            "integer_damping_rule": True,
            "exact_fraction_ldlt_and_inverse": True,
            "singular_and_nonpositive_pivot_fail_closed": True,
            "sequential_inverse_hessian_error_feedback": True,
            "lower_index_assignment_ties": True,
            "full_block_row_scale_quadratic": True,
            "row_scale_endpoint_and_tie_rules": True,
            "strict_coordinate_codebook_sweep": True,
            "empty_assignment_retention": True,
            "codebook_order_endpoint_and_tie_rules": True,
            "nibble_packing": True,
            "scale32_legality": True,
            "hash_mismatch_fail_closed": True,
            "independent_candidate_a_b_match": True,
            "outer_sweeps_completed": OUTER_SWEEPS,
            "monotonic_undamped_block_objective": True,
            "official_execution_marker_exists": False,
        },
        "covariance_fixture": {
            "manifest_sha256": capture_a["covariance_manifest_sha256"],
            "block_widths": [block["live_channels"] for block in capture_a["blocks"]],
            "captured_gram_sha256": canonical_sha256(captured_grams),
        },
        "candidate_fixture": {
            "construction_sha256": candidate_a["construction_sha256"],
            "payload_sha256": candidate_a["payload_sha256"],
            "row_scale_stream_sha256": candidate_a["row_scale_stream_sha256"],
            "codebook_record_sha256": candidate_a["codebook_record_sha256"],
            "iteration_trace_sha256": candidate_a["iteration_trace_sha256"],
            "non_regressing_against_fresh_v4": candidate_a["non_regressing_against_fresh_v4"],
            "strictly_improved_against_fresh_v4": candidate_a["strictly_improved_against_fresh_v4"],
        },
        "official_build_namespace_changed": False,
        "official_execution_marker_exists": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "prepare", "verify"))
    args = parser.parse_args()
    if args.command == "self-test":
        from v6_block_covariance_full_model_backend import combined_self_test

        result = combined_self_test()
        print(
            "ACE2_OPTION_B_BLOCK_COVARIANCE_CODEBOOK_V6_SELF_TEST_PASS "
            + json.dumps(result, sort_keys=True)
        )
        return 0
    from v6_block_covariance_full_model_backend import (
        load_verified_closure,
        prepare,
        record_preparation_failure,
    )

    if args.command == "prepare":
        try:
            result = prepare()
        except Exception as exc:
            record_preparation_failure(exc)
            raise
        print(
            "ACE2_OPTION_B_BLOCK_COVARIANCE_CODEBOOK_V6_PREFLIGHT_COMPLETE "
            + json.dumps(result, sort_keys=True)
        )
        return 0 if result["status"] == "PREATTEMPT_READY" else 2
    result = load_verified_closure()
    print(
        "ACE2_OPTION_B_BLOCK_COVARIANCE_CODEBOOK_V6_PREFLIGHT_VERIFIED "
        + json.dumps(result, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
