#!/usr/bin/env python3
"""Construct, preflight, and conditionally execute the exact weighted V5 policy."""

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
from typing import Any, Sequence

import run_option_b_alias_safe_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_stage1 as v4
from ace2_quality_contracts import SCALE32_ALL_ZERO_RECORD


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "option_b_alias_safe_calibration_weighted_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_w4a8_v5"
MISSION_ID = "4c4398b2acd0"
TASK_ID = "task-0b08da0c4088"
TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_CALIBRATION_WEIGHTED_JOINT_ROW_SCALE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V5_ENGINEER_TASK.json"
TASK_COMPANION = TASK_PATH.with_suffix(".sha256")
PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_CALIBRATION_WEIGHTED_JOINT_ROW_SCALE_TENSOR_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V5_PLAN.json"
PLAN_COMPANION = PLAN_PATH.with_suffix(".sha256")
OFFICIAL_ROOT = ROOT / "build/stage1-option-b-alias-safe-calibration-weighted-joint-row-scale-tensor-codebook-w4-grouped-dynamic-scale32-w4a8-v5"
OFFICIAL_MARKER = OFFICIAL_ROOT / "attempt-0001/execution_started.json"
RUNNER_PATH = Path(__file__).resolve()

Q4_4_MIN = v4.Q4_4_MIN
Q4_4_MAX = v4.Q4_4_MAX
CODEBOOK_ENTRIES = v4.CODEBOOK_ENTRIES
OUTER_ITERATIONS = 8


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
    require(companion.read_text(encoding="utf-8").strip() == expected, f"checksum companion differs: {path}")


def require_project_python() -> dict[str, str]:
    expected = ROOT / ".venv/bin/python"
    require(expected.is_file(), "project Python is missing")
    require(os.path.samefile(Path(sys.executable), expected), "wrong Python executable")
    return {"entrypoint": "./.venv/bin/python", "python": platform.python_version()}


def normalize_activation_energies(values: Sequence[int]) -> tuple[int, ...]:
    require(bool(values), "empty activation-energy vector")
    require(all(type(value) is int and value >= 0 for value in values), "activation energy differs")
    positive = tuple(max(value, 1) for value in values)
    divisor = math.gcd(*positive)
    require(divisor > 0, "activation-energy GCD differs")
    normalized = tuple(value // divisor for value in positive)
    require(all(value > 0 for value in normalized), "normalized activation energy is not positive")
    require(math.gcd(*normalized) == 1, "activation-energy vector was not completely reduced")
    return normalized


def validate_channel_weights(weights: Sequence[int], width: int) -> tuple[int, ...]:
    values = tuple(weights)
    require(len(values) == width, "activation-energy width differs")
    require(all(type(value) is int and value > 0 for value in values), "activation energy is not a positive integer")
    require(math.gcd(*values) == 1, "activation-energy vector is not GCD-reduced")
    return values


def weighted_source_sse(
    rows: Sequence[Sequence[Fraction]],
    channel_weights: Sequence[int],
    records: Sequence[int],
    codepoints: Sequence[int],
    assignments: Sequence[Sequence[int]],
) -> Fraction:
    width = len(rows[0])
    weights = validate_channel_weights(channel_weights, width)
    points = v4.validate_codebook(codepoints)
    total = Fraction(0)
    for row, record, indices in zip(rows, records, assignments, strict=True):
        require(len(row) == len(indices) == width, "weighted SSE row width differs")
        scale = v4.scale32_value(record)
        for weight, energy, index in zip(row, weights, indices, strict=True):
            error = weight - scale * points[index] / 16
            total += energy * error * error
    return total


def weighted_update_row_scale(
    row: Sequence[Fraction],
    channel_weights: Sequence[int],
    assignments: Sequence[int],
    codepoints: Sequence[int],
    prior_record: int,
) -> tuple[int, dict[str, Any]]:
    weights = validate_channel_weights(channel_weights, len(row))
    require(len(row) == len(assignments), "weighted row-scale assignment width differs")
    if all(weight == 0 for weight in row):
        zero_index = v4.nearest_codepoint_index(Fraction(0), codepoints)
        require(all(index == zero_index for index in assignments), "all-zero row did not use nearest-zero codepoint")
        return SCALE32_ALL_ZERO_RECORD, {
            "all_zero": True,
            "selected_record": SCALE32_ALL_ZERO_RECORD,
            "nearest_zero_index": zero_index,
        }
    points = v4.validate_codebook(codepoints)
    selected_points = [points[index] for index in assignments]
    denominator = sum(energy * point * point for energy, point in zip(weights, selected_points, strict=True))
    require(denominator > 0, "nonzero weighted row has zero assigned-codepoint energy")
    numerator = sum(
        (energy * weight * point for weight, energy, point in zip(row, weights, selected_points, strict=True)),
        Fraction(0),
    )
    ideal = Fraction(16) * numerator / denominator
    maximum = max(abs(weight) for weight in row)
    lower_bound = maximum * 16 / 127
    candidates = v4.candidate_row_scale_records(ideal, lower_bound, prior_record)

    def fixed_assignment_sse(record: int) -> Fraction:
        scale = v4.scale32_value(record)
        return sum(
            energy * (weight - scale * point / 16) ** 2
            for weight, energy, point in zip(row, weights, selected_points, strict=True)
        )

    selected = v4.choose_row_scale(candidates["candidate_records"], fixed_assignment_sse)
    return selected, {
        "all_zero": False,
        "ideal": v4.fraction_record(ideal),
        "constrained_ideal": v4.fraction_record(candidates["constrained_ideal"]),
        "lower_bound": v4.fraction_record(lower_bound),
        "floor_record": candidates["floor_record"],
        "ceil_record": candidates["ceil_record"],
        "candidate_records": list(candidates["candidate_records"]),
        "selected_record": selected,
        "selected_weighted_sse": v4.fraction_record(fixed_assignment_sse(selected)),
    }


def weighted_update_codebook(
    rows: Sequence[Sequence[Fraction]],
    channel_weights: Sequence[int],
    records: Sequence[int],
    assignments: Sequence[Sequence[int]],
    prior_codebook: Sequence[int],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    width = len(rows[0])
    weights = validate_channel_weights(channel_weights, width)
    prior = v4.validate_codebook(prior_codebook)
    counts = [0] * CODEBOOK_ENTRIES
    constants = [Fraction(0) for _ in range(CODEBOOK_ENTRIES)]
    linears = [Fraction(0) for _ in range(CODEBOOK_ENTRIES)]
    quadratics = [Fraction(0) for _ in range(CODEBOOK_ENTRIES)]
    for row, record, indices in zip(rows, records, assignments, strict=True):
        scale_over_16 = v4.scale32_value(record) / 16
        for weight, energy, index in zip(row, weights, indices, strict=True):
            counts[index] += 1
            constants[index] += energy * weight * weight
            linears[index] += energy * weight * scale_over_16
            quadratics[index] += energy * scale_over_16 * scale_over_16
    costs: list[dict[int, Fraction]] = []
    for index in range(CODEBOOK_ENTRIES):
        allowed = (prior[index],) if counts[index] == 0 else range(Q4_4_MIN, Q4_4_MAX + 1)
        costs.append(
            {
                point: constants[index] - 2 * point * linears[index] + point * point * quadratics[index]
                for point in allowed
            }
        )
    selected, cost = v4.strict_codebook_dp(costs)
    selected = v4.validate_codebook(selected)
    require(
        all(count != 0 or selected[index] == prior[index] for index, count in enumerate(counts)),
        "weighted empty cluster did not retain its prior codepoint",
    )
    return selected, {
        "cluster_counts": counts,
        "empty_cluster_count": sum(count == 0 for count in counts),
        "fixed_assignment_optimum_weighted_sse": v4.fraction_record(cost),
    }


def construct_weighted_candidate(
    rows: Sequence[Sequence[Fraction]], channel_weights: Sequence[int]
) -> dict[str, Any]:
    bf16_bits = v4.validate_bf16_rows(rows)
    weights = validate_channel_weights(channel_weights, len(rows[0]))
    v4_initial = v4.construct_joint_candidate(rows)
    records = tuple(v4_initial["final_row_scale_records"])
    codebook = tuple(v4_initial["final_codepoints"])
    assignments = v4.assign_rows(rows, records, codebook)
    baseline = weighted_source_sse(rows, weights, records, codebook, assignments)
    previous = baseline
    trace: list[dict[str, Any]] = []

    for iteration in range(1, OUTER_ITERATIONS + 1):
        updated_records: list[int] = []
        row_updates: list[dict[str, Any]] = []
        for row, row_assignments, prior_record in zip(rows, assignments, records, strict=True):
            selected, update = weighted_update_row_scale(
                row, weights, row_assignments, codebook, prior_record
            )
            updated_records.append(selected)
            row_updates.append(update)
        records = tuple(updated_records)
        assignments_after_scale = v4.assign_rows(rows, records, codebook)
        after_scale = weighted_source_sse(rows, weights, records, codebook, assignments_after_scale)
        require(after_scale <= previous, "weighted row-scale update increased the exact objective")

        codebook, codebook_update = weighted_update_codebook(
            rows, weights, records, assignments_after_scale, codebook
        )
        assignments = v4.assign_rows(rows, records, codebook)
        final = weighted_source_sse(rows, weights, records, codebook, assignments)
        require(final <= after_scale, "weighted codebook update increased the exact objective")
        require(final <= previous, "weighted outer iteration increased the exact objective")
        trace.append(
            {
                "iteration": iteration,
                "prior_weighted_objective": v4.fraction_record(previous),
                "after_row_scale_weighted_objective": v4.fraction_record(after_scale),
                "final_weighted_objective": v4.fraction_record(final),
                "row_scale_records": list(records),
                "row_scale_updates_sha256": canonical_sha256(row_updates),
                "codepoints": list(codebook),
                "codebook_update": codebook_update,
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
        "fixture_shape": [len(rows), len(rows[0])],
        "source_bf16_sha256": hashlib.sha256(
            b"".join(struct.pack("<H", bits) for row in bf16_bits for bits in row)
        ).hexdigest(),
        "activation_energy_sha256": canonical_sha256(list(weights)),
        "fresh_v4_initialization_sha256": v4_initial["construction_sha256"],
        "v4_baseline_weighted_objective": v4.fraction_record(baseline),
        "outer_iterations_completed": len(trace),
        "final_row_scale_records": list(records),
        "final_codepoints": list(codebook),
        "payload_hex": payload.hex(),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "row_scale_stream_sha256": hashlib.sha256(scale_raw).hexdigest(),
        "codebook_record_sha256": hashlib.sha256(codebook_raw).hexdigest(),
        "final_weighted_objective": v4.fraction_record(previous),
        "iteration_trace": trace,
        "iteration_trace_sha256": canonical_sha256(trace),
    }
    return {**body, "construction_sha256": canonical_sha256(body)}


def self_test() -> dict[str, Any]:
    environment = require_project_python()
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "frozen V5 Engineer authority differs")
    root_existed_before = OFFICIAL_ROOT.exists()
    marker_existed_before = OFFICIAL_MARKER.exists()
    require(not marker_existed_before, "V5 execution marker already exists")

    require(normalize_activation_energies((2, 4, 6)) == (1, 2, 3), "activation-energy GCD reduction differs")
    require(normalize_activation_energies((0, 4, 8)) == (1, 4, 8), "activation-energy zero floor differs")
    fixture = v4.faithful_fixture()
    uniform = (1,) * len(fixture[0])
    weighted = normalize_activation_energies(tuple((index + 1) ** 2 for index in range(len(fixture[0]))))
    uniform_candidate = construct_weighted_candidate(fixture, uniform)
    weighted_a = construct_weighted_candidate(fixture, weighted)
    weighted_b = construct_weighted_candidate(fixture, weighted)
    require(weighted_a == weighted_b, "independent weighted fixture constructions differ")
    v4_candidate = v4.construct_joint_candidate(fixture)
    require(uniform_candidate["payload_hex"] == v4_candidate["payload_hex"], "uniform weighted payload differs from V4 fixed point")
    require(uniform_candidate["final_row_scale_records"] == v4_candidate["final_row_scale_records"], "uniform weighted row scales differ from V4 fixed point")
    require(uniform_candidate["final_codepoints"] == v4_candidate["final_codepoints"], "uniform weighted codebook differs from V4 fixed point")
    require(
        all(
            Fraction(item["final_weighted_objective"]["numerator"], item["final_weighted_objective"]["denominator"])
            <= Fraction(item["prior_weighted_objective"]["numerator"], item["prior_weighted_objective"]["denominator"])
            for item in weighted_a["iteration_trace"]
        ),
        "weighted fixture monotonicity differs",
    )
    require(OFFICIAL_ROOT.exists() == root_existed_before, "self-test changed the V5 namespace")
    require(OFFICIAL_MARKER.exists() == marker_existed_before is False, "self-test changed the V5 execution marker")

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
            "activation_energy_positive_floor": True,
            "activation_energy_whole_vector_gcd_reduction": True,
            "weighted_scale32_bracketing_neighbors_ties_range_and_zero_row": True,
            "weighted_strict_codebook_dp_optimum_empty_clusters_ties_range_and_packing": True,
            "fresh_v4_initialization": True,
            "uniform_weighted_fixed_point_matches_v4": True,
            "independent_weighted_candidate_a_b_match": True,
            "outer_iterations_completed": OUTER_ITERATIONS,
            "monotonic_weighted_objective": True,
            "official_execution_marker_exists": False,
        },
        "weighted_fixture": weighted_a,
        "official_build_namespace_changed": False,
        "official_execution_marker_exists": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "prepare", "verify", "execute", "verify-result"))
    args = parser.parse_args()
    from v5_calibration_weighted_full_model_backend import (
        combined_self_test,
        execute_once,
        load_verified_closure,
        prepare,
        record_preparation_failure,
        verify_result,
    )

    if args.command == "self-test":
        result = combined_self_test()
        print("ACE2_OPTION_B_CALIBRATION_WEIGHTED_CODEBOOK_V5_SELF_TEST_PASS " + json.dumps(result, sort_keys=True))
        return 0
    if args.command == "prepare":
        try:
            result = prepare()
        except Exception as exc:
            record_preparation_failure(exc)
            raise
        print("ACE2_OPTION_B_CALIBRATION_WEIGHTED_CODEBOOK_V5_PREFLIGHT_COMPLETE " + json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PREATTEMPT_READY" else 2
    if args.command == "verify":
        result = load_verified_closure()
        print("ACE2_OPTION_B_CALIBRATION_WEIGHTED_CODEBOOK_V5_PREFLIGHT_VERIFIED " + json.dumps(result, sort_keys=True))
        return 0
    if args.command == "execute":
        return execute_once()
    result = verify_result()
    print("ACE2_OPTION_B_CALIBRATION_WEIGHTED_CODEBOOK_V5_RESULT_VERIFIED " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
