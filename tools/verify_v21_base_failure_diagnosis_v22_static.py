#!/usr/bin/env python3
"""Read-only verifier for the V21 Base diagnosis and static V22 successor plan."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS = ROOT / "diagnosis/QK_GBFP8_HEAD64_V21_BASE_FAILURE_DIAGNOSIS_AND_V22_STATIC_SUCCESSOR.json"
RESULT = ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_289140ba/primary/result/base/result.json"
EXPECTED_RESULT_FILE_SHA256 = "4901c835dd9b8700cb3a3dd5ecac54f2d1112f7f2f6e909a7ad37ee35ab8941f"
EXPECTED_RESULT_BYTE_COUNT = 8767
EXPECTED_LABELS = ["G8", "G4", "G2", "G1"]
SOURCE_HASHES = {
    "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py": "8ab74c7397006c9f419059f295613ce4a743a3e0b540176ba7cf6dbb6efd7f63",
    "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root/tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v21.py": "751c28ae2744b695e0bffe364680ba4451a21332624411b9a94d1a0aee42e9b2",
    "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root/tools/qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v21.py": "201e84ba8c5ae1460698a4595c00d51ca698646eab4dd250708a1c4d25ce39e0",
    "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root/tools/qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v21.py": "64ac5f95ba9876ad84244b9c66d09115303dfcaf8c111216b2b44dc821487e04",
    "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root/tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v21.py": "c16be825cef50b92bd70e171fca0cc1eedf2a73fc39327cc84ceae23b1451367",
    "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root/tools/qk_gbfp8_head64_granularity_sweep_v21_binding_resolution.py": "02f3d808b036e706f981253b9bcbf713dfe999034d3704400e4b9975534054ea",
    "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root/bindings/C02_EXACT_BINDINGS_25.json": "87ab3deb25f77d89b0797d72004b4436f9541987da13a42f44a2bff7f9fa9655",
    "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root/accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json": "3d36df763e775bc8f3fb5d106eaf842f7b74482c236b9697906bb93647c44832",
    "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root/accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json": "07f818190e7f97032a6bc3724914f00f36070f429f1cd549c67e4e7b026ae720",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def compact_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict, f"JSON object: {path}")
    return value, raw


def fraction_at_least(actual: dict[str, int], limit: dict[str, int]) -> bool:
    return actual["numerator"] * limit["denominator"] >= limit["numerator"] * actual["denominator"]


def derive_thresholds(metrics: dict[str, Any], hard_gates: dict[str, Any]) -> dict[str, Any]:
    invalid = metrics["invalid_accounting"]
    rank = metrics["rank_margin"]
    top = metrics["top_key"]
    actuals = {
        "cross_lane_record_count_maximum": invalid["cross_lane_record_count"],
        "invalid_or_non_finite_value_count_maximum": invalid["invalid_or_non_finite_value_count"],
        "normalization_rejection_count_maximum": invalid["normalization_rejection_count"],
        "positive_centered_realized_score_count_maximum": invalid["positive_centered_realized_score_count"],
        "rank_margin_violation_count_maximum": rank["violation_count"],
        "saturation_event_count_maximum": invalid["saturation_event_count"],
        "top_key_matching_fraction_minimum": top["matching_fraction"],
        "top_key_mismatch_count_maximum": top["mismatch_count"],
        "unique_oracle_positive_margin_preserved_fraction_minimum": rank["preserved_positive_margin_fraction"],
    }
    checks: dict[str, Any] = {}
    for name, limit in hard_gates.items():
        actual = actuals[name]
        passed = fraction_at_least(actual, limit) if type(limit) is dict else actual <= limit
        checks[name] = {"actual": actual, "limit": limit, "pass": passed}
    checks["all_hard_gates_pass"] = all(check["pass"] for check in checks.values())
    return checks


def formatted(value: float) -> str:
    return f"{value:.12f}"


def verify_result_seal(result: dict[str, Any]) -> None:
    unsealed = deepcopy(result)
    embedded = unsealed.pop("result_sha256")
    require(sha256_bytes(compact_bytes(unsealed)) == embedded, "embedded result_sha256")


def verify_candidate_projection(source: dict[str, Any], projected: dict[str, Any]) -> None:
    metrics = source["metrics"]
    top = metrics["top_key"]
    rank = metrics["rank_margin"]
    score = metrics["score_error"]
    require(projected["label"] == source["label"], f"label projection {source['label']}")
    require(projected["group_size"] == source["group_size"], f"group size projection {source['label']}")

    projected_top = projected["top_key"]
    for key in ("matching_row_count", "mismatch_count", "row_count"):
        require(projected_top[key] == top[key], f"top projection {source['label']} {key}")
    require(
        projected_top["matching_percentage"] == formatted(100.0 * top["matching_row_count"] / top["row_count"]),
        f"top percentage {source['label']}",
    )

    projected_rank = projected["rank_margin"]
    for key in (
        "preserved_positive_margin_row_count",
        "unique_oracle_top_row_count",
        "violation_count",
        "minimum_realized_margin_q12_20_lsb",
    ):
        require(projected_rank[key] == rank[key], f"rank projection {source['label']} {key}")
    require(
        projected_rank["preserved_percentage"]
        == formatted(100.0 * rank["preserved_positive_margin_row_count"] / rank["unique_oracle_top_row_count"]),
        f"rank percentage {source['label']}",
    )
    require(
        projected_rank["minimum_realized_margin_q12_20"]
        == formatted(rank["minimum_realized_margin_q12_20_lsb"] / (1 << 20)),
        f"minimum margin {source['label']}",
    )

    projected_score = projected["score_error"]
    integer_score_keys = (
        "valid_value_count",
        "maximum_absolute_error_q12_20_lsb",
        "sum_absolute_error_q12_20_lsb",
        "sum_signed_error_q12_20_lsb",
        "sum_squared_error_q40_40_lsb2",
    )
    for key in integer_score_keys:
        require(projected_score[key] == score[key], f"score projection {source['label']} {key}")
    count = score["valid_value_count"]
    require(
        projected_score["mean_absolute_error_q12_20"]
        == formatted(score["sum_absolute_error_q12_20_lsb"] / count / (1 << 20)),
        f"mean absolute {source['label']}",
    )
    require(
        projected_score["mean_signed_error_q12_20"]
        == formatted(score["sum_signed_error_q12_20_lsb"] / count / (1 << 20)),
        f"mean signed {source['label']}",
    )
    require(
        projected_score["rmse_q12_20"]
        == formatted(math.sqrt(score["sum_squared_error_q40_40_lsb2"] / count) / (1 << 20)),
        f"rmse {source['label']}",
    )
    require(
        projected_score["signed_to_absolute_sum_ratio"]
        == formatted(score["sum_signed_error_q12_20_lsb"] / score["sum_absolute_error_q12_20_lsb"]),
        f"signed/absolute ratio {source['label']}",
    )


def candidate_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int | str]:
    before_metrics = before["metrics"]
    after_metrics = after["metrics"]
    before_rank = before_metrics["rank_margin"]
    after_rank = after_metrics["rank_margin"]
    before_score = before_metrics["score_error"]
    after_score = after_metrics["score_error"]
    return {
        "from": before["label"],
        "to": after["label"],
        "top_key_matching_row_count": after_metrics["top_key"]["matching_row_count"] - before_metrics["top_key"]["matching_row_count"],
        "rank_margin_violation_count": after_rank["violation_count"] - before_rank["violation_count"],
        "minimum_realized_margin_q12_20_lsb": after_rank["minimum_realized_margin_q12_20_lsb"] - before_rank["minimum_realized_margin_q12_20_lsb"],
        "sum_absolute_error_q12_20_lsb": after_score["sum_absolute_error_q12_20_lsb"] - before_score["sum_absolute_error_q12_20_lsb"],
        "sum_signed_error_q12_20_lsb": after_score["sum_signed_error_q12_20_lsb"] - before_score["sum_signed_error_q12_20_lsb"],
        "sum_squared_error_q40_40_lsb2": after_score["sum_squared_error_q40_40_lsb2"] - before_score["sum_squared_error_q40_40_lsb2"],
        "maximum_absolute_error_q12_20_lsb": after_score["maximum_absolute_error_q12_20_lsb"] - before_score["maximum_absolute_error_q12_20_lsb"],
    }


def main() -> int:
    diagnosis, diagnosis_raw = load_json(DIAGNOSIS)
    result, result_raw = load_json(RESULT)
    require(len(result_raw) == EXPECTED_RESULT_BYTE_COUNT, "immutable result byte count")
    require(sha256_bytes(result_raw) == EXPECTED_RESULT_FILE_SHA256, "immutable result file SHA-256")
    verify_result_seal(result)

    immutable = diagnosis["immutable_v21_evidence"]
    require(immutable["result_file_sha256"] == EXPECTED_RESULT_FILE_SHA256, "diagnosis result hash binding")
    require(immutable["result_byte_count"] == EXPECTED_RESULT_BYTE_COUNT, "diagnosis result size binding")
    require(immutable["embedded_result_sha256"] == result["result_sha256"], "diagnosis embedded result hash")
    require(immutable["irreversible_action_id"] == result["irreversible_action_id"], "diagnosis action binding")
    require(immutable["candidate_order"] == EXPECTED_LABELS, "diagnosis candidate order")
    require(immutable["selected_candidate"] is None and result["selected_candidate"] is None, "no selected candidate")
    require(result["terminal"]["status"] == "FAILED_TERMINAL", "terminal status")
    require(result["terminal"]["reason_code"] == "HARD_THRESHOLD_FAILED", "terminal reason")
    require(result["terminal"]["retry_replay_resume_repair_permitted"] is False, "terminal no replay")

    gate_section = diagnosis["hard_gate_recomputation"]
    hard_gates = gate_section["hard_limits"]
    candidates = result["candidate_results"]
    require([candidate["label"] for candidate in candidates] == EXPECTED_LABELS, "result candidate order")
    require(len(gate_section["candidates"]) == len(candidates), "candidate projection count")
    expected_passing = set(gate_section["passing_gate_names_for_every_candidate"])
    expected_failing = set(gate_section["failing_gate_names_for_every_candidate"])
    require(len(expected_passing) == 5 and len(expected_failing) == 4, "gate partition cardinality")

    for source, projected in zip(candidates, gate_section["candidates"]):
        derived = derive_thresholds(source["metrics"], hard_gates)
        require(derived == source["threshold_evaluation"], f"threshold derivation {source['label']}")
        passing = {name for name in hard_gates if derived[name]["pass"]}
        failing = set(hard_gates) - passing
        require(passing == expected_passing, f"passing gates {source['label']}")
        require(failing == expected_failing, f"failing gates {source['label']}")
        require(projected["failed_hard_gate_count"] == len(failing), f"failed gate count {source['label']}")
        require(projected["all_hard_gates_pass"] is False, f"candidate failure {source['label']}")
        verify_candidate_projection(source, projected)

    expected_deltas = [candidate_delta(candidates[index], candidates[index + 1]) for index in range(3)]
    require(diagnosis["cross_candidate_trends"]["successive_deltas"] == expected_deltas, "successive deltas")
    g2, g1 = candidates[2], candidates[3]
    g2_top = g2["metrics"]["top_key"]
    g1_top = g1["metrics"]["top_key"]
    comparison = diagnosis["cross_candidate_trends"]["g1_relative_to_g2"]
    require(comparison["top_key_matching_rows"] == -10, "G1 top match regression")
    require(comparison["top_key_mismatches"] == 10, "G1 mismatch regression")
    require(
        comparison["top_key_matching_fraction_delta"] == {"numerator": -10, "denominator": 574},
        "G1 exact top fraction delta",
    )
    require(
        comparison["top_key_matching_percentage_point_delta"]
        == formatted(100.0 * (g1_top["matching_row_count"] - g2_top["matching_row_count"]) / g1_top["row_count"]),
        "G1 percentage-point regression",
    )
    require(
        comparison["mismatch_count_relative_increase_percentage"]
        == formatted(100.0 * (g1_top["mismatch_count"] - g2_top["mismatch_count"]) / g2_top["mismatch_count"]),
        "G1 mismatch relative increase",
    )

    for relative_path, expected_hash in SOURCE_HASHES.items():
        require(sha256_bytes((ROOT / relative_path).read_bytes()) == expected_hash, f"source hash {relative_path}")
    trace_hashes = {item["path"]: item["sha256"] for item in diagnosis["production_path_trace"]}
    for relative_path, expected_hash in SOURCE_HASHES.items():
        if relative_path in trace_hashes:
            require(trace_hashes[relative_path] == expected_hash, f"trace hash binding {relative_path}")

    successor = diagnosis["v22_static_successor"]
    frozen = successor["frozen_bindings"]
    require(successor["identity"]["action_id"] != result["irreversible_action_id"], "distinct V22 identity")
    require(successor["identity"]["runtime_namespace"] is None, "no V22 runtime namespace")
    require(successor["identity"]["execution_authority_granted"] is False, "no V22 execution authority")
    require(frozen["official_evaluator_sha256"] == result["evaluator_sha256"], "frozen evaluator")
    require(frozen["benchmark_package_sha256"] == result["package_sha256"], "frozen benchmark package")
    require(frozen["tensor_bundle_sha256"] == result["input_bindings"]["tensor_bundle_sha256"], "frozen tensor bundle")
    require(frozen["tensor_record_count"] == result["input_bindings"]["tensor_record_count"], "frozen tensor count")
    require(frozen["sealed_set_id"] == result["input_bindings"]["sealed_set_id"], "frozen sealed set")
    require(frozen["candidate_order"] == EXPECTED_LABELS, "frozen successor candidate order")
    require(frozen["hard_gates"] == hard_gates, "frozen successor hard gates")
    require(frozen["score_error_affects_selection"] is False, "score error remains tracking only")
    require(successor["future_separately_authorized_experiment_design"]["authority_status"] == "NOT_GRANTED_BY_THIS_ARTIFACT", "future authority absent")
    require(diagnosis["review"]["status"] == "PENDING", "Fresh-L2 review pending")
    require(diagnosis["review"]["engineer_self_review_can_close"] is False, "independent review required")
    require(diagnosis["claim_boundary"]["official_payload_access_performed"] is False, "no payload claim")
    require(diagnosis["claim_boundary"]["official_evaluator_invocation_performed"] is False, "no evaluator claim")

    report = {
        "artifact_sha256": sha256_bytes(diagnosis_raw),
        "candidate_count": len(candidates),
        "failed_hard_gates_per_candidate": 4,
        "g1_top_match_delta_vs_g2": -10,
        "official_evaluator_invocations": 0,
        "official_payload_opens": 0,
        "result_file_sha256": EXPECTED_RESULT_FILE_SHA256,
        "status": "PASS_STATIC_READ_ONLY",
        "v22_execution_authority_granted": False,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
