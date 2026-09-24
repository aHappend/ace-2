#!/usr/bin/env python3
"""Pure V7 package/result validation and terminal classification.

This module is import-safe and static-only.  It contains no controller launch,
payload access, namespace creation, or publication entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


PACKAGE_ID = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_PACKAGE"
PACKAGE_SCHEMA_ID = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_PACKAGE_V1"
RESULT_SCHEMA_ID = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_RESULT_V1"
RESULT_SCHEMA_DOCUMENT_ID = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_RESULT_V1_SCHEMA"
MISSION_ID = "2a5b510c82b4"
GENERATION_ID = "qk-gbfp8-head64-granularity-sweep-v7-base"
SEALED_SET_ID = "w4a8-c02-attention-substage-trace-v2"
ACTION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,255}$"
PACKAGE_PATH = "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_PACKAGE.json"
CONTROLLER_PATH = "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v7.py"
EVALUATOR_PATH = "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v7.py"
INTERPRETER_BINDING = {
    "path": "/home/argustest/miniconda3/bin/python3.13",
    "sha256": "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad",
    "version": "3.13.5",
}
STATIC_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}
FUTURE_CONTROLLER_ARGV = [
    INTERPRETER_BINDING["path"],
    CONTROLLER_PATH,
    "--package",
    PACKAGE_PATH,
    "--authority",
    "build/qk-gbfp8-head64-granularity-sweep-v7-authority/base/authority.json",
    "--credential",
    "build/qk-gbfp8-head64-granularity-sweep-v7-authority/base/credential.json",
    "--ledger",
    "build/qk-gbfp8-head64-granularity-sweep-v7-authority/base/authority-ledger.json",
    "--result",
    "build/qk-gbfp8-head64-granularity-sweep-v7/base/result.json",
    "--first-terminal",
    "build/qk-gbfp8-head64-granularity-sweep-v7-authority/base/first-terminal.json",
    "--irreversible-action-id",
    "<NEW_IRREVERSIBLE_ACTION_ID>",
]
FUTURE_EVALUATOR_ARGV = [
    INTERPRETER_BINDING["path"],
    EVALUATOR_PATH,
    "--package",
    PACKAGE_PATH,
    "--consumed-ledger",
    "build/qk-gbfp8-head64-granularity-sweep-v7-authority/base/authority-ledger.json",
    "--result",
    "build/qk-gbfp8-head64-granularity-sweep-v7/base/result.json",
    "--irreversible-action-id",
    "<NEW_IRREVERSIBLE_ACTION_ID>",
]
REVIEWER_BINDING = {
    "acceptance_artifact": "review/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_REVIEW.json",
    "role": "Fresh-L2",
    "status": "PENDING_INDEPENDENT_REVIEW",
}

CANDIDATES = (
    {
        "bytes_per_head": 80,
        "exponent_bytes_per_head": 16,
        "group_count": 8,
        "group_size": 8,
        "label": "G8",
        "mantissa_bytes_per_head": 64,
    },
    {
        "bytes_per_head": 96,
        "exponent_bytes_per_head": 32,
        "group_count": 16,
        "group_size": 4,
        "label": "G4",
        "mantissa_bytes_per_head": 64,
    },
    {
        "bytes_per_head": 128,
        "exponent_bytes_per_head": 64,
        "group_count": 32,
        "group_size": 2,
        "label": "G2",
        "mantissa_bytes_per_head": 64,
    },
    {
        "bytes_per_head": 192,
        "exponent_bytes_per_head": 128,
        "group_count": 64,
        "group_size": 1,
        "label": "G1",
        "mantissa_bytes_per_head": 64,
    },
)
CANDIDATE_LABELS = tuple(item["label"] for item in CANDIDATES)

HARD_GATES = {
    "cross_lane_record_count_maximum": 0,
    "invalid_or_non_finite_value_count_maximum": 0,
    "normalization_rejection_count_maximum": 0,
    "positive_centered_realized_score_count_maximum": 0,
    "rank_margin_violation_count_maximum": 0,
    "saturation_event_count_maximum": 0,
    "top_key_matching_fraction_minimum": {"denominator": 1, "numerator": 1},
    "top_key_mismatch_count_maximum": 0,
    "unique_oracle_positive_margin_preserved_fraction_minimum": {
        "denominator": 1,
        "numerator": 1,
    },
}

NUMERICAL_CONTRACT = {
    "candidate_evaluation": "ALL_FOUR_IN_ONE_INVOCATION",
    "c02_accepted_reader_cross_parser": "INDEPENDENT_REQUIRED",
    "c02_producer_magic": "ACE2-C02-TENSORS-V1\n",
    "full_record_nonfinite_scan_before_transformation": True,
    "head_lanes": 64,
    "kv_head_mapping": "floor(query_head/7)",
    "mantissa": {
        "forbidden_value": -128,
        "maximum": 127,
        "minimum": -127,
        "saturation_or_clamping_permitted": False,
        "signed_bits": 8,
    },
    "oracle_and_hard_gates": "SAME_AS_IMMUTABLE_G16_V6",
    "rounding": {
        "mantissa": "round_to_nearest_ties_to_even(value / 2**exponent)",
        "ranking_tie": "lowest_key_index",
        "zero_group_exponent": 0,
    },
    "score_error_policy": "TRACKING_ONLY_NOT_A_HARD_GATE",
    "score_path": "Q12.20_INTEGER_LSB",
    "shared_exponent": {
        "bytes_per_group": 2,
        "canonical_rule": "smallest_signed_int16_exponent_with_all_rounded_mantissas_in_-127_to_127",
        "signed_bits": 16,
    },
    "tensor_open_count": 1,
}

RESULT_KEYS = {
    "authority_sha256",
    "candidate_results",
    "consumed_ledger_sha256",
    "evaluator_sha256",
    "fresh_l2_acceptance_sha256",
    "generation_id",
    "input_bindings",
    "invocation_sha256",
    "irreversible_action_id",
    "lane_label",
    "model_identity_sha256",
    "namespace_label",
    "package_id",
    "package_sha256",
    "result_sha256",
    "schema_id",
    "selected_candidate",
    "selection",
    "terminal",
}
CANDIDATE_RESULT_KEYS = {
    "bytes_per_head",
    "exponent_bytes_per_head",
    "group_count",
    "group_size",
    "label",
    "mantissa_bytes_per_head",
    "metrics",
    "threshold_evaluation",
}
METRICS_KEYS = {"invalid_accounting", "rank_margin", "score_error", "top_key"}
INVALID_KEYS = {
    "cross_lane_record_count",
    "invalid_or_non_finite_value_count",
    "normalization_rejection_count",
    "positive_centered_realized_score_count",
    "saturation_event_count",
}
RANK_KEYS = {
    "minimum_realized_margin_q12_20_lsb",
    "preserved_positive_margin_fraction",
    "preserved_positive_margin_row_count",
    "unique_oracle_top_row_count",
    "violation_count",
}
SCORE_ERROR_KEYS = {
    "maximum_absolute_error_q12_20_lsb",
    "sum_absolute_error_q12_20_lsb",
    "sum_signed_error_q12_20_lsb",
    "sum_squared_error_q40_40_lsb2",
    "valid_value_count",
}
TOP_KEY_KEYS = {
    "matching_fraction",
    "matching_row_count",
    "mismatch_count",
    "row_count",
}
THRESHOLD_KEYS = set(HARD_GATES) | {"all_hard_gates_pass"}
CHECK_KEYS = {"actual", "limit", "pass"}
SELECTION_KEYS = {"passing_candidates", "policy", "selected_candidate"}
TERMINAL_KEYS = {
    "first_record_immutable",
    "invocation_count_performed",
    "metrics_published",
    "reason_code",
    "retry_replay_resume_repair_permitted",
    "status",
    "tensor_open_count",
    "thresholds_evaluated",
}


class ControllerError(RuntimeError):
    """Fail-closed validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ControllerError(message)


def exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    require(type(value) is dict, f"{context} object")
    require(set(value) == expected, f"{context} exact keys")
    return value


def compact_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def valid_sha256(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def reject_nonfinite(value: Any, context: str = "record") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"{context} nonfinite")
    elif type(value) is dict:
        for key, nested in value.items():
            require(type(key) is str, f"{context} key")
            reject_nonfinite(nested, f"{context}.{key}")
    elif type(value) is list:
        for index, nested in enumerate(value):
            reject_nonfinite(nested, f"{context}[{index}]")


def _nonnegative_integer(value: Any, context: str) -> int:
    require(type(value) is int and value >= 0, f"{context} nonnegative integer")
    return value


def _fraction(value: Any, context: str) -> dict[str, int]:
    item = exact_keys(value, {"denominator", "numerator"}, context)
    numerator = _nonnegative_integer(item["numerator"], f"{context}.numerator")
    denominator = item["denominator"]
    require(type(denominator) is int and denominator > 0, f"{context}.denominator")
    require(numerator <= denominator, f"{context} range")
    return item


def _fraction_at_least(actual: dict[str, int], limit: dict[str, int]) -> bool:
    return actual["numerator"] * limit["denominator"] >= limit["numerator"] * actual["denominator"]


def _validate_candidates(candidates: Any) -> None:
    require(type(candidates) is list and len(candidates) == 4, "candidate cardinality")
    for index, expected in enumerate(CANDIDATES):
        candidate = exact_keys(candidates[index], set(expected), f"candidate[{index}]")
        require(candidate == expected, f"candidate[{index}] definition")
        require(candidate["group_count"] * candidate["group_size"] == 64, f"candidate[{index}] lanes")
        require(candidate["exponent_bytes_per_head"] == 2 * candidate["group_count"], f"candidate[{index}] exponent bytes")
        require(
            candidate["bytes_per_head"]
            == candidate["mantissa_bytes_per_head"] + candidate["exponent_bytes_per_head"],
            f"candidate[{index}] total bytes",
        )


def validate_package(package: dict[str, Any]) -> None:
    reject_nonfinite(package, "package")
    exact_keys(
        package,
        {
            "artifact_kind",
            "candidates",
            "claim_boundary",
            "future_namespaces",
            "future_protocol",
            "hard_gates",
            "mission_id",
            "numerical_contract",
            "package_id",
            "package_schema_id",
            "result_contract",
            "schema_version",
            "selection_policy",
            "static_bindings",
            "v6_immutable_evidence",
        },
        "package",
    )
    require(package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_static_v7_package", "package artifact kind")
    require(package["package_id"] == PACKAGE_ID, "package identity")
    require(package["package_schema_id"] == PACKAGE_SCHEMA_ID, "package schema identity")
    require(package["schema_version"] == 7, "package schema version")
    require(package["mission_id"] == MISSION_ID, "mission identity")
    _validate_candidates(package["candidates"])
    require(package["numerical_contract"] == NUMERICAL_CONTRACT, "numerical contract")
    require(package["hard_gates"] == HARD_GATES, "hard gates")
    require(
        package["selection_policy"]
        == {
            "failure_if_none_pass": "HARD_THRESHOLD_FAILED",
            "order": list(CANDIDATE_LABELS),
            "policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER",
            "require_every_hard_gate": True,
            "score_error_affects_selection": False,
        },
        "selection policy",
    )
    require(
        package["claim_boundary"]
        == {
            "authority_materialized": False,
            "checkpoint_176_activity": False,
            "controller_or_evaluator_invoked": False,
            "credential_materialized": False,
            "fresh_l2_accepted": False,
            "hardware_or_rtl_activity": False,
            "ledger_result_or_terminal_materialized": False,
            "sealed_tensor_accessed_or_projected": False,
            "status": "STATIC_ONLY_NOT_AUTHORIZED_FOR_EXECUTION",
        },
        "claim boundary",
    )
    result_contract = exact_keys(
        package["result_contract"],
        {
            "candidate_result_exact_keys",
            "failure_terminal",
            "package_id",
            "result_schema",
            "schema_id",
            "success_terminal",
            "top_level_exact_keys",
        },
        "result contract",
    )
    require(result_contract["package_id"] == PACKAGE_ID, "result contract package identity")
    require(result_contract["schema_id"] == RESULT_SCHEMA_ID, "result contract schema identity")
    require(result_contract["top_level_exact_keys"] == sorted(RESULT_KEYS), "result top-level keys")
    require(result_contract["candidate_result_exact_keys"] == sorted(CANDIDATE_RESULT_KEYS), "candidate result keys")
    require(
        result_contract["failure_terminal"]
        == {"reason_code": "HARD_THRESHOLD_FAILED", "status": "FAILED_TERMINAL"},
        "failure terminal contract",
    )
    require(
        result_contract["success_terminal"]
        == {"reason_code": "HARD_THRESHOLDS_PASSED", "status": "SUCCEEDED_TERMINAL"},
        "success terminal contract",
    )
    result_schema = exact_keys(result_contract["result_schema"], {"path", "schema_id", "sha256"}, "result schema binding")
    require(result_schema["schema_id"] == RESULT_SCHEMA_DOCUMENT_ID, "result schema document identity")
    require(valid_sha256(result_schema["sha256"]), "result schema checksum syntax")
    require(
        package["future_protocol"]
        == {
            "consume_before_payload": True,
            "consumed_orphan_handling": "CONSERVATIVE_NO_REPLAY",
            "create_only_publication": True,
            "durable_ledger_before_tensor_open": True,
            "execution_requires_new_irreversible_action_id": True,
            "one_evaluator_invocation": True,
            "one_tensor_open": True,
            "retry_replay_resume_repair_permitted": False,
        },
        "future protocol",
    )
    namespaces = exact_keys(
        package["future_namespaces"],
        {"authority", "credential", "first_terminal", "ledger", "result"},
        "future namespaces",
    )
    require(all(type(value) is str and "v7" in value for value in namespaces.values()), "distinct V7 namespaces")
    bindings = exact_keys(
        package["static_bindings"],
        {
            "controller",
            "controller_argv",
            "environment",
            "evaluator",
            "evaluator_argv",
            "interpreter",
            "result_schema",
            "reviewer",
            "verifier",
        },
        "static bindings",
    )
    for name in ("controller", "evaluator", "result_schema", "verifier"):
        binding = exact_keys(bindings[name], {"path", "sha256"}, f"binding.{name}")
        require(type(binding["path"]) is str and binding["path"], f"binding.{name}.path")
        require(valid_sha256(binding["sha256"]), f"binding.{name}.sha256")
    require(bindings["controller"]["path"] == CONTROLLER_PATH, "controller path binding")
    require(bindings["evaluator"]["path"] == EVALUATOR_PATH, "evaluator path binding")
    require(bindings["controller_argv"] == FUTURE_CONTROLLER_ARGV, "controller argv binding")
    require(bindings["evaluator_argv"] == FUTURE_EVALUATOR_ARGV, "evaluator argv binding")
    require(bindings["environment"] == STATIC_ENVIRONMENT, "environment binding")
    interpreter = exact_keys(bindings["interpreter"], {"path", "sha256", "version"}, "interpreter binding")
    require(interpreter == INTERPRETER_BINDING, "interpreter binding")
    require(bindings["reviewer"] == REVIEWER_BINDING, "reviewer binding")
    evidence = exact_keys(
        package["v6_immutable_evidence"],
        {
            "aftermath_recomputation_sha256",
            "base_failed_all_four_ranking_hard_gates",
            "first_terminal_file_sha256",
            "fresh_l2_terminal_handoff_sha256",
            "no_replay",
            "result_file_sha256",
            "result_self_checksum",
            "task_id",
        },
        "V6 evidence",
    )
    require(evidence["task_id"] == "c9e0569b083e", "V6 task")
    require(evidence["fresh_l2_terminal_handoff_sha256"] == "0d20e696e77e47e425a761d726fa10ab57c09733d7d3af066ea6f874b7debd49", "V6 handoff")
    require(evidence["aftermath_recomputation_sha256"] == "7f17929b5b087764e3d321b8354e750ac42542aad772fe18cb4f4942e01b53fb", "V6 aftermath")
    require(evidence["result_file_sha256"] == "74b5d8fded1e344848023144336d0f8de8e0409c6f589fb8c9a7b642fade695c", "V6 result")
    require(evidence["result_self_checksum"] == "12769e09c6d0ebbcc52d6487b716d5f6bae921d1dd324832f77ed0f7d127f7f5", "V6 result self-checksum")
    require(evidence["first_terminal_file_sha256"] == "83a875fab228f552e6b26cfeacf021296ad65b122981873570d65ea9c4f367e6", "V6 first terminal")
    require(evidence["base_failed_all_four_ranking_hard_gates"] is True and evidence["no_replay"] is True, "V6 terminal state")


def _validate_metrics(metrics: Any, context: str) -> dict[str, Any]:
    value = exact_keys(metrics, METRICS_KEYS, context)
    invalid = exact_keys(value["invalid_accounting"], INVALID_KEYS, f"{context}.invalid_accounting")
    for key in INVALID_KEYS:
        _nonnegative_integer(invalid[key], f"{context}.invalid_accounting.{key}")
    rank = exact_keys(value["rank_margin"], RANK_KEYS, f"{context}.rank_margin")
    require(type(rank["minimum_realized_margin_q12_20_lsb"]) is int, f"{context}.rank minimum")
    _fraction(rank["preserved_positive_margin_fraction"], f"{context}.rank preserved fraction")
    _nonnegative_integer(rank["preserved_positive_margin_row_count"], f"{context}.rank preserved rows")
    _nonnegative_integer(rank["unique_oracle_top_row_count"], f"{context}.rank unique rows")
    _nonnegative_integer(rank["violation_count"], f"{context}.rank violations")
    require(
        rank["preserved_positive_margin_row_count"] == rank["preserved_positive_margin_fraction"]["numerator"]
        and rank["unique_oracle_top_row_count"] == rank["preserved_positive_margin_fraction"]["denominator"],
        f"{context}.rank fraction accounting",
    )
    score = exact_keys(value["score_error"], SCORE_ERROR_KEYS, f"{context}.score_error")
    for key in SCORE_ERROR_KEYS - {"sum_signed_error_q12_20_lsb"}:
        _nonnegative_integer(score[key], f"{context}.score_error.{key}")
    require(type(score["sum_signed_error_q12_20_lsb"]) is int, f"{context}.score signed error")
    top = exact_keys(value["top_key"], TOP_KEY_KEYS, f"{context}.top_key")
    _fraction(top["matching_fraction"], f"{context}.top matching fraction")
    for key in TOP_KEY_KEYS - {"matching_fraction"}:
        _nonnegative_integer(top[key], f"{context}.top_key.{key}")
    require(top["matching_row_count"] + top["mismatch_count"] == top["row_count"], f"{context}.top accounting")
    require(
        top["matching_fraction"]
        == {"numerator": top["matching_row_count"], "denominator": top["row_count"]},
        f"{context}.top fraction accounting",
    )
    return value


def derive_threshold_evaluation(metrics: dict[str, Any]) -> dict[str, Any]:
    invalid = metrics["invalid_accounting"]
    rank = metrics["rank_margin"]
    top = metrics["top_key"]
    actuals: dict[str, Any] = {
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
    for name, limit in HARD_GATES.items():
        actual = actuals[name]
        passed = _fraction_at_least(actual, limit) if type(limit) is dict else actual <= limit
        checks[name] = {"actual": actual, "limit": limit, "pass": passed}
    checks["all_hard_gates_pass"] = all(check["pass"] for check in checks.values())
    return checks


def _validate_thresholds(value: Any, metrics: dict[str, Any], context: str) -> dict[str, Any]:
    thresholds = exact_keys(value, THRESHOLD_KEYS, context)
    for name in HARD_GATES:
        exact_keys(thresholds[name], CHECK_KEYS, f"{context}.{name}")
    require(thresholds == derive_threshold_evaluation(metrics), f"{context} derivation")
    return thresholds


def validate_result_record(
    package: dict[str, Any],
    result: dict[str, Any],
    *,
    package_sha256: str,
    evaluator_sha256: str,
) -> None:
    validate_package(package)
    reject_nonfinite(result, "result")
    exact_keys(result, RESULT_KEYS, "result")
    require(result["package_id"] == PACKAGE_ID, "result package_id")
    require(result["schema_id"] == RESULT_SCHEMA_ID, "result schema_id")
    require(valid_sha256(package_sha256) and result["package_sha256"] == package_sha256, "result package checksum")
    require(valid_sha256(evaluator_sha256) and result["evaluator_sha256"] == evaluator_sha256, "result evaluator checksum")
    for key in (
        "authority_sha256",
        "consumed_ledger_sha256",
        "fresh_l2_acceptance_sha256",
        "invocation_sha256",
        "model_identity_sha256",
    ):
        require(valid_sha256(result[key]), f"result {key}")
    require(result["generation_id"] == GENERATION_ID, "result generation")
    require(result["lane_label"] == "Base" and result["namespace_label"] == "base", "result Base lane")
    require(type(result["irreversible_action_id"]) is str and re.fullmatch(ACTION_ID_PATTERN, result["irreversible_action_id"]), "result action id")
    inputs = exact_keys(
        result["input_bindings"],
        {"sealed_set_id", "tensor_bundle_sha256", "tensor_record_count"},
        "result input bindings",
    )
    require(inputs["sealed_set_id"] == SEALED_SET_ID, "result sealed set")
    require(valid_sha256(inputs["tensor_bundle_sha256"]), "result tensor checksum")
    require(inputs["tensor_record_count"] == 25, "result tensor record count")
    require(type(result["candidate_results"]) is list and len(result["candidate_results"]) == 4, "result candidate cardinality")
    passing: list[str] = []
    for index, expected in enumerate(CANDIDATES):
        candidate = exact_keys(result["candidate_results"][index], CANDIDATE_RESULT_KEYS, f"result candidate[{index}]")
        for key, expected_value in expected.items():
            require(candidate[key] == expected_value, f"result candidate[{index}] {key}")
        metrics = _validate_metrics(candidate["metrics"], f"result candidate[{index}].metrics")
        thresholds = _validate_thresholds(candidate["threshold_evaluation"], metrics, f"result candidate[{index}].thresholds")
        if thresholds["all_hard_gates_pass"]:
            passing.append(expected["label"])
    selected = passing[0] if passing else None
    require(result["selected_candidate"] == selected, "selected candidate")
    selection = exact_keys(result["selection"], SELECTION_KEYS, "result selection")
    require(selection["policy"] == "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER", "selection policy")
    require(selection["passing_candidates"] == passing, "selection passing candidates")
    require(selection["selected_candidate"] == selected, "selection selected candidate")
    terminal = exact_keys(result["terminal"], TERMINAL_KEYS, "result terminal")
    require(terminal["first_record_immutable"] is True, "terminal immutability")
    require(terminal["invocation_count_performed"] == 1, "terminal invocation count")
    require(terminal["tensor_open_count"] == 1, "terminal tensor open count")
    require(terminal["metrics_published"] is True and terminal["thresholds_evaluated"] is True, "terminal publication")
    require(terminal["retry_replay_resume_repair_permitted"] is False, "terminal replay policy")
    if selected is None:
        require(terminal["status"] == "FAILED_TERMINAL", "failure terminal status")
        require(terminal["reason_code"] == "HARD_THRESHOLD_FAILED", "failure terminal reason")
    else:
        require(terminal["status"] == "SUCCEEDED_TERMINAL", "success terminal status")
        require(terminal["reason_code"] == "HARD_THRESHOLDS_PASSED", "success terminal reason")
    require(valid_sha256(result["result_sha256"]), "result self-checksum syntax")
    payload = dict(result)
    observed = payload.pop("result_sha256")
    require(hashlib.sha256(compact_bytes(payload)).hexdigest() == observed, "result self-checksum")


def classify_terminal(
    evaluator_returncode: int,
    consumed_ledger_sha256: str | None,
    validated_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if validated_result is not None:
        require(valid_sha256(consumed_ledger_sha256), "result without durable consumed ledger")
        status = validated_result["terminal"]["status"]
        reason = validated_result["terminal"]["reason_code"]
        require(evaluator_returncode == (0 if status == "SUCCEEDED_TERMINAL" else 1), "return code/result mismatch")
        return {
            "authority_consumed": True,
            "consumed_ledger_sha256": consumed_ledger_sha256,
            "execution_started": True,
            "orphaned_after_consumption": False,
            "reason_code": reason,
            "status": status,
        }
    return {
        "authority_consumed": True,
        "consumed_ledger_sha256": consumed_ledger_sha256,
        "execution_started": False,
        "orphaned_after_consumption": True,
        "reason_code": (
            "EVALUATOR_EXIT_WITHOUT_VALID_RESULT"
            if valid_sha256(consumed_ledger_sha256)
            else "NO_VISIBLE_LEDGER_AFTER_EVALUATOR_TREATED_AS_CONSUMED_ORPHAN"
        ),
        "status": "CONSUMED_ORPHAN",
    }
