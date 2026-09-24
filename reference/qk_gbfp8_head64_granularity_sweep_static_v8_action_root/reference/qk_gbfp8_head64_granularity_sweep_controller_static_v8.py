#!/usr/bin/env python3
"""Static-only V8 validation, durability, lifecycle, and classification.

There is no execution CLI.  Filesystem and evaluator effects are injected so
the verifier can exercise the future lifecycle without touching live paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any


PACKAGE_ID = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE"
PACKAGE_SCHEMA_ID = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE_V1"
RESULT_SCHEMA_ID = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_RESULT_V1"
RESULT_SCHEMA_DOCUMENT_ID = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_RESULT_V1_SCHEMA"
MISSION_ID = "f0fa5c269681"
GENERATION_ID = "qk-gbfp8-head64-granularity-sweep-v8-base"
SEALED_SET_ID = "w4a8-c02-attention-substage-trace-v2"
OFFICIAL_MODEL_IDENTITY_SHA256 = "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7"
OFFICIAL_INPUT_BINDINGS = {
    "input_token_ids_sha256": "1b8c972381a2c3d7c754d1d2879b4389a13aca3f1d485f703510702c1cf2eb86",
    "lane_metadata": {
        "byte_count": 20057,
        "path": "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json",
        "sha256": "4d5904378b9ccbabd434119b6a57f8d4266abf3d8e772c4c1bca98df785e7f3a",
    },
    "sealed_set_id": SEALED_SET_ID,
    "tensor_bundle": {
        "byte_count": 1305797,
        "path": "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin",
        "sha256": "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175",
    },
    "tensor_records": {
        "bf16_oracle_scores": {
            "dtype": "torch.bfloat16",
            "sha256": "49627e8364e534c61f4db8208d82798e53409c3c10d5d5e28c3c1462ef617765",
            "shape": [1, 14, 41, 41],
            "tensor_name": "bf16.qk_scaled_scores",
        },
        "realized_key_source": {
            "dtype": "torch.bfloat16",
            "sha256": "401cdb0dc4a8def3190ac424f96df272c2bcf11241874759977d692845a19c0a",
            "shape": [1, 2, 41, 64],
            "tensor_name": "bf16.k_rope",
        },
        "realized_query_source": {
            "dtype": "torch.bfloat16",
            "sha256": "285e064ffea9571b7e3ed192a7083bf831dc5d444e0139558d3d51f084995429",
            "shape": [1, 14, 41, 64],
            "tensor_name": "bf16.q_rope",
        },
    },
}
OFFICIAL_BENCHMARK = {
    "input_bindings": OFFICIAL_INPUT_BINDINGS,
    "model_identity_sha256": OFFICIAL_MODEL_IDENTITY_SHA256,
}
ACTION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,255}$"
PACKAGE_PATH = "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.json"
CONTROLLER_PATH = "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v8.py"
EVALUATOR_PATH = "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"
C02_PARSER_PATH = "reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"
RESULT_SCHEMA_PATH = "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.json"
VERIFIER_PATH = "tools/verify_qk_gbfp8_head64_granularity_sweep_static_v8.py"
ACTION_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
STATIC_ARTIFACT_PATHS = {
    "c02_parser": C02_PARSER_PATH,
    "controller": CONTROLLER_PATH,
    "evaluator": EVALUATOR_PATH,
    "result_schema": RESULT_SCHEMA_PATH,
    "verifier": VERIFIER_PATH,
}
INTERPRETER_BINDING = {
    "path": "/home/argustest/miniconda3/bin/python3.13",
    "sha256": "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad",
    "version": "3.13.5",
}
SCHEMA_VALIDATION_STACK = {
    "attrs": "26.1.0",
    "jsonschema": "4.23.0",
    "jsonschema-specifications": "2025.9.1",
    "referencing": "0.37.0",
    "rpds-py": "0.30.0",
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
    "build/qk-gbfp8-head64-granularity-sweep-v8-authority/base/authority.json",
    "--credential",
    "build/qk-gbfp8-head64-granularity-sweep-v8-authority/base/credential.json",
    "--ledger",
    "build/qk-gbfp8-head64-granularity-sweep-v8-authority/base/authority-ledger.json",
    "--result",
    "build/qk-gbfp8-head64-granularity-sweep-v8/base/result.json",
    "--first-terminal",
    "build/qk-gbfp8-head64-granularity-sweep-v8-authority/base/first-terminal.json",
    "--irreversible-action-id",
    "<NEW_IRREVERSIBLE_ACTION_ID>",
]
FUTURE_EVALUATOR_ARGV = [
    INTERPRETER_BINDING["path"],
    EVALUATOR_PATH,
    "--package",
    PACKAGE_PATH,
    "--consumed-ledger",
    "build/qk-gbfp8-head64-granularity-sweep-v8-authority/base/authority-ledger.json",
    "--result",
    "build/qk-gbfp8-head64-granularity-sweep-v8/base/result.json",
    "--irreversible-action-id",
    "<NEW_IRREVERSIBLE_ACTION_ID>",
]
REVIEWER_BINDING = {
    "acceptance_artifact": "review/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_REVIEW.json",
    "role": "Fresh-L2",
    "status": "PENDING_INDEPENDENT_REVIEW",
}
FUTURE_NAMESPACES = {
    "authority": "build/qk-gbfp8-head64-granularity-sweep-v8-authority/base/authority.json",
    "credential": "build/qk-gbfp8-head64-granularity-sweep-v8-authority/base/credential.json",
    "first_terminal": "build/qk-gbfp8-head64-granularity-sweep-v8-authority/base/first-terminal.json",
    "ledger": "build/qk-gbfp8-head64-granularity-sweep-v8-authority/base/authority-ledger.json",
    "result": "build/qk-gbfp8-head64-granularity-sweep-v8/base/result.json",
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
FIXED_METRIC_POPULATIONS = {
    "rank_margin_unique_oracle_top_row_count": 346,
    "score_error_valid_value_count": 12054,
    "top_key_row_count": 574,
}

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
    "fixed_metric_populations": FIXED_METRIC_POPULATIONS,
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


def _static_artifact_sha256(relative_path: str) -> str:
    digest = hashlib.sha256()
    with open(os.path.join(ACTION_ROOT, relative_path), "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


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
            "official_benchmark",
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
    require(package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_static_v8_package", "package artifact kind")
    require(package["package_id"] == PACKAGE_ID, "package identity")
    require(package["package_schema_id"] == PACKAGE_SCHEMA_ID, "package schema identity")
    require(package["schema_version"] == 8, "package schema version")
    require(package["mission_id"] == MISSION_ID, "mission identity")
    _validate_candidates(package["candidates"])
    require(package["numerical_contract"] == NUMERICAL_CONTRACT, "numerical contract")
    require(package["hard_gates"] == HARD_GATES, "hard gates")
    require(package["official_benchmark"] == OFFICIAL_BENCHMARK, "official V6 benchmark identity")
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
    require(result_schema["path"] == RESULT_SCHEMA_PATH, "result contract schema path binding")
    require(
        result_schema["sha256"] == _static_artifact_sha256(RESULT_SCHEMA_PATH),
        "result contract schema checksum binding",
    )
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
    require(namespaces == FUTURE_NAMESPACES, "exact distinct V8 namespaces")
    bindings = exact_keys(
        package["static_bindings"],
        {
            "c02_parser",
            "controller",
            "controller_argv",
            "environment",
            "evaluator",
            "evaluator_argv",
            "interpreter",
            "result_schema",
            "reviewer",
            "schema_validation_stack",
            "verifier",
        },
        "static bindings",
    )
    for name, expected_path in STATIC_ARTIFACT_PATHS.items():
        binding = exact_keys(bindings[name], {"path", "sha256"}, f"binding.{name}")
        require(binding["path"] == expected_path, f"binding.{name}.path")
        require(binding["sha256"] == _static_artifact_sha256(expected_path), f"binding.{name}.sha256")
    require(
        result_schema["path"] == bindings["result_schema"]["path"]
        and result_schema["sha256"] == bindings["result_schema"]["sha256"],
        "duplicate result schema binding",
    )
    require(bindings["controller_argv"] == FUTURE_CONTROLLER_ARGV, "controller argv binding")
    require(bindings["evaluator_argv"] == FUTURE_EVALUATOR_ARGV, "evaluator argv binding")
    require(bindings["environment"] == STATIC_ENVIRONMENT, "environment binding")
    interpreter = exact_keys(bindings["interpreter"], {"path", "sha256", "version"}, "interpreter binding")
    require(interpreter == INTERPRETER_BINDING, "interpreter binding")
    require(bindings["reviewer"] == REVIEWER_BINDING, "reviewer binding")
    require(bindings["schema_validation_stack"] == SCHEMA_VALIDATION_STACK, "schema validation stack binding")
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
    minimum_margin = rank["minimum_realized_margin_q12_20_lsb"]
    require(type(minimum_margin) is int, f"{context}.rank minimum")
    fraction = _fraction(rank["preserved_positive_margin_fraction"], f"{context}.rank preserved fraction")
    preserved = _nonnegative_integer(rank["preserved_positive_margin_row_count"], f"{context}.rank preserved rows")
    unique = _nonnegative_integer(rank["unique_oracle_top_row_count"], f"{context}.rank unique rows")
    violations = _nonnegative_integer(rank["violation_count"], f"{context}.rank violations")
    require(
        unique == FIXED_METRIC_POPULATIONS["rank_margin_unique_oracle_top_row_count"],
        f"{context}.rank fixed population",
    )
    require(preserved + violations == unique, f"{context}.rank accounting")
    require(
        fraction == {"numerator": preserved, "denominator": unique},
        f"{context}.rank fraction accounting",
    )
    require((violations == 0) == (minimum_margin > 0), f"{context}.rank minimum accounting")
    score = exact_keys(value["score_error"], SCORE_ERROR_KEYS, f"{context}.score_error")
    maximum = _nonnegative_integer(score["maximum_absolute_error_q12_20_lsb"], f"{context}.score maximum")
    sum_absolute = _nonnegative_integer(score["sum_absolute_error_q12_20_lsb"], f"{context}.score absolute sum")
    sum_signed = score["sum_signed_error_q12_20_lsb"]
    require(type(sum_signed) is int, f"{context}.score signed error")
    sum_squared = _nonnegative_integer(score["sum_squared_error_q40_40_lsb2"], f"{context}.score squared sum")
    valid_values = _nonnegative_integer(score["valid_value_count"], f"{context}.score valid values")
    require(
        valid_values == FIXED_METRIC_POPULATIONS["score_error_valid_value_count"],
        f"{context}.score fixed population",
    )
    require(abs(sum_signed) <= sum_absolute, f"{context}.score signed/absolute accounting")
    require(maximum <= sum_absolute <= valid_values * maximum, f"{context}.score maximum/absolute accounting")
    require(maximum * maximum <= sum_squared <= maximum * sum_absolute, f"{context}.score squared accounting")
    require(sum_absolute * sum_absolute <= valid_values * sum_squared, f"{context}.score Cauchy accounting")
    top = exact_keys(value["top_key"], TOP_KEY_KEYS, f"{context}.top_key")
    top_fraction = _fraction(top["matching_fraction"], f"{context}.top matching fraction")
    matching = _nonnegative_integer(top["matching_row_count"], f"{context}.top matching rows")
    mismatches = _nonnegative_integer(top["mismatch_count"], f"{context}.top mismatches")
    rows = _nonnegative_integer(top["row_count"], f"{context}.top rows")
    require(rows == FIXED_METRIC_POPULATIONS["top_key_row_count"], f"{context}.top fixed population")
    require(matching + mismatches == rows, f"{context}.top accounting")
    require(
        top_fraction == {"numerator": matching, "denominator": rows},
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
    expected_authority_sha256: str | None = None,
    expected_consumed_ledger_sha256: str | None = None,
    expected_fresh_l2_acceptance_sha256: str | None = None,
    expected_invocation_sha256: str | None = None,
    expected_irreversible_action_id: str | None = None,
    expected_model_identity_sha256: str | None = None,
    expected_tensor_bundle_sha256: str | None = None,
) -> None:
    validate_package(package)
    reject_nonfinite(result, "result")
    exact_keys(result, RESULT_KEYS, "result")
    require(result["package_id"] == PACKAGE_ID, "result package_id")
    require(result["schema_id"] == RESULT_SCHEMA_ID, "result schema_id")
    require(valid_sha256(package_sha256) and result["package_sha256"] == package_sha256, "result package checksum")
    require(valid_sha256(evaluator_sha256) and result["evaluator_sha256"] == evaluator_sha256, "result evaluator checksum")
    if expected_authority_sha256 is not None:
        require(result["authority_sha256"] == expected_authority_sha256, "result authority checksum")
    if expected_consumed_ledger_sha256 is not None:
        require(result["consumed_ledger_sha256"] == expected_consumed_ledger_sha256, "result ledger checksum")
    if expected_fresh_l2_acceptance_sha256 is not None:
        require(result["fresh_l2_acceptance_sha256"] == expected_fresh_l2_acceptance_sha256, "result Fresh-L2 acceptance checksum")
    if expected_invocation_sha256 is not None:
        require(result["invocation_sha256"] == expected_invocation_sha256, "result invocation checksum")
    if expected_irreversible_action_id is not None:
        require(result["irreversible_action_id"] == expected_irreversible_action_id, "result action identity")
    if expected_model_identity_sha256 is not None:
        require(result["model_identity_sha256"] == expected_model_identity_sha256, "result model identity checksum")
    require(result["model_identity_sha256"] == OFFICIAL_MODEL_IDENTITY_SHA256, "result official model identity")
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
    require(
        inputs["tensor_bundle_sha256"] == OFFICIAL_INPUT_BINDINGS["tensor_bundle"]["sha256"],
        "result official tensor bundle identity",
    )
    if expected_tensor_bundle_sha256 is not None:
        require(inputs["tensor_bundle_sha256"] == expected_tensor_bundle_sha256, "result tensor authority checksum")
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


AUTHORITY_KEYS = {
    "artifact_kind",
    "authority_sha256",
    "evaluator_sha256",
    "fresh_l2_acceptance_sha256",
    "input_bindings",
    "invocation_sha256",
    "irreversible_action_id",
    "model_identity_sha256",
    "package_sha256",
}
CREDENTIAL_KEYS = {
    "artifact_kind",
    "authority_sha256",
    "credential_nonce_sha256",
    "credential_sha256",
    "irreversible_action_id",
}
LEDGER_KEYS = {
    "artifact_kind",
    "authority_sha256",
    "consumed_ledger_sha256",
    "credential_sha256",
    "invocation_count_permitted",
    "irreversible_action_id",
    "package_sha256",
    "replay_permitted",
    "state",
}


def seal_record(value: dict[str, Any], checksum_key: str) -> dict[str, Any]:
    sealed = dict(value)
    sealed.pop(checksum_key, None)
    sealed[checksum_key] = hashlib.sha256(compact_bytes(sealed)).hexdigest()
    return sealed


def load_canonical_json_bytes(data: bytes, context: str) -> dict[str, Any]:
    require(type(data) is bytes, f"{context} bytes")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"{context} duplicate key")
            result[key] = value
        return result

    try:
        text = data.decode("ascii", "strict")
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ControllerError(f"{context} nonfinite {token}")),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ControllerError(f"{context} JSON") from error
    require(type(value) is dict, f"{context} object")
    reject_nonfinite(value, context)
    require(compact_bytes(value) == data, f"{context} canonical bytes")
    return value


def _validate_self_checksum(value: dict[str, Any], checksum_key: str, context: str) -> None:
    require(valid_sha256(value[checksum_key]), f"{context} checksum syntax")
    payload = dict(value)
    observed = payload.pop(checksum_key)
    require(hashlib.sha256(compact_bytes(payload)).hexdigest() == observed, f"{context} self checksum")


def validate_authority_record(authority: dict[str, Any], *, package_sha256: str, evaluator_sha256: str) -> None:
    value = exact_keys(authority, AUTHORITY_KEYS, "authority")
    require(value["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_v8_authority", "authority kind")
    require(value["package_sha256"] == package_sha256 and valid_sha256(package_sha256), "authority package")
    require(value["evaluator_sha256"] == evaluator_sha256 and valid_sha256(evaluator_sha256), "authority evaluator")
    for key in ("fresh_l2_acceptance_sha256", "invocation_sha256"):
        require(valid_sha256(value[key]), f"authority {key}")
    require(value["model_identity_sha256"] == OFFICIAL_MODEL_IDENTITY_SHA256, "authority official model identity")
    require(value["input_bindings"] == OFFICIAL_INPUT_BINDINGS, "authority official input bindings")
    require(type(value["irreversible_action_id"]) is str and re.fullmatch(ACTION_ID_PATTERN, value["irreversible_action_id"]), "authority action id")
    _validate_self_checksum(value, "authority_sha256", "authority")


def validate_credential_record(credential: dict[str, Any], authority: dict[str, Any]) -> None:
    value = exact_keys(credential, CREDENTIAL_KEYS, "credential")
    require(value["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_v8_credential", "credential kind")
    require(value["authority_sha256"] == authority["authority_sha256"], "credential authority")
    require(value["irreversible_action_id"] == authority["irreversible_action_id"], "credential action id")
    require(valid_sha256(value["credential_nonce_sha256"]), "credential nonce")
    _validate_self_checksum(value, "credential_sha256", "credential")


def build_consumed_ledger(authority: dict[str, Any], credential: dict[str, Any]) -> dict[str, Any]:
    return seal_record(
        {
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_v8_consumed_ledger",
            "authority_sha256": authority["authority_sha256"],
            "credential_sha256": credential["credential_sha256"],
            "invocation_count_permitted": 1,
            "irreversible_action_id": authority["irreversible_action_id"],
            "package_sha256": authority["package_sha256"],
            "replay_permitted": False,
            "state": "CONSUMED_BEFORE_PAYLOAD",
        },
        "consumed_ledger_sha256",
    )


def validate_consumed_ledger(ledger: dict[str, Any], authority: dict[str, Any], credential: dict[str, Any]) -> None:
    value = exact_keys(ledger, LEDGER_KEYS, "consumed ledger")
    require(value["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_v8_consumed_ledger", "ledger kind")
    require(value["authority_sha256"] == authority["authority_sha256"], "ledger authority")
    require(value["credential_sha256"] == credential["credential_sha256"], "ledger credential")
    require(value["irreversible_action_id"] == authority["irreversible_action_id"], "ledger action id")
    require(value["package_sha256"] == authority["package_sha256"], "ledger package")
    require(value["invocation_count_permitted"] == 1, "ledger invocation count")
    require(value["replay_permitted"] is False and value["state"] == "CONSUMED_BEFORE_PAYLOAD", "ledger no replay")
    _validate_self_checksum(value, "consumed_ledger_sha256", "consumed ledger")


def durable_create_bytes(path: str, payload: bytes, ops: Any) -> None:
    require(type(path) is str and path and type(payload) is bytes, "durable create request")
    descriptor: int | None = None
    try:
        descriptor = ops.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        offset = 0
        while offset < len(payload):
            written = ops.write(descriptor, payload[offset:])
            require(type(written) is int and 0 < written <= len(payload) - offset, "short write made no progress")
            offset += written
        ops.fsync(descriptor)
    except Exception as error:
        raise ControllerError(f"durable create failed: {path}") from error
    finally:
        if descriptor is not None:
            try:
                ops.close(descriptor)
            except Exception as error:
                raise ControllerError(f"durable close failed: {path}") from error
    directory = os.path.dirname(path) or "."
    directory_descriptor: int | None = None
    try:
        directory_descriptor = ops.open(directory, os.O_RDONLY | os.O_DIRECTORY, 0)
        ops.fsync(directory_descriptor)
    except Exception as error:
        raise ControllerError(f"durable directory sync failed: {path}") from error
    finally:
        if directory_descriptor is not None:
            try:
                ops.close(directory_descriptor)
            except Exception as error:
                raise ControllerError(f"directory close failed: {path}") from error


def durable_unlink(path: str, ops: Any) -> None:
    try:
        ops.unlink(path)
        descriptor = ops.open(os.path.dirname(path) or ".", os.O_RDONLY | os.O_DIRECTORY, 0)
        try:
            ops.fsync(descriptor)
        finally:
            ops.close(descriptor)
    except Exception as error:
        raise ControllerError(f"durable unlink failed: {path}") from error


def classify_terminal(
    evaluator_returncode: int,
    consumed_ledger_sha256: str | None,
    validated_result: dict[str, Any] | None,
    *,
    invocation_started: bool = True,
    invalid_result_observed: bool = False,
) -> dict[str, Any]:
    if validated_result is not None:
        require(valid_sha256(consumed_ledger_sha256), "result without durable consumed ledger")
        status = validated_result["terminal"]["status"]
        reason = validated_result["terminal"]["reason_code"]
        require(evaluator_returncode == (0 if status == "SUCCEEDED_TERMINAL" else 1), "return code/result mismatch")
        return {
            "authority_consumed": True,
            "consumed_ledger_sha256": consumed_ledger_sha256,
            "execution_started": invocation_started,
            "invocation_count_performed": 1,
            "orphaned_after_consumption": False,
            "reason_code": reason,
            "status": status,
        }
    return {
        "authority_consumed": True,
        "consumed_ledger_sha256": consumed_ledger_sha256,
        "execution_started": invocation_started,
        "invocation_count_performed": 1 if invocation_started else 0,
        "orphaned_after_consumption": True,
        "reason_code": (
            "INVALID_EVALUATOR_RESULT"
            if invalid_result_observed
            else "EVALUATOR_EXIT_WITHOUT_VALID_RESULT"
            if invocation_started and valid_sha256(consumed_ledger_sha256)
            else "CONSUMED_BEFORE_EVALUATOR_INVOCATION"
            if valid_sha256(consumed_ledger_sha256)
            else "NO_VISIBLE_LEDGER_AFTER_EVALUATOR_TREATED_AS_CONSUMED_ORPHAN"
        ),
        "status": "CONSUMED_ORPHAN",
    }


def _first_terminal_record(classification: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    value = dict(classification)
    value.update(
        {
            "authority_sha256": authority["authority_sha256"],
            "first_record_immutable": True,
            "irreversible_action_id": authority["irreversible_action_id"],
            "retry_replay_resume_repair_permitted": False,
        }
    )
    return seal_record(value, "first_terminal_sha256")


def run_inert_lifecycle(
    package: dict[str, Any],
    authority_bytes: bytes,
    credential_bytes: bytes,
    *,
    package_sha256: str,
    evaluator_sha256: str,
    ops: Any,
    evaluator_invoke: Any,
) -> dict[str, Any]:
    """Exercise one fail-closed lifecycle using only injected effects."""

    validate_package(package)
    namespaces = package["future_namespaces"]
    for name in ("ledger", "result", "first_terminal"):
        require(not ops.exists(namespaces[name]), f"permanent no-replay: {name} already exists")
    require(ops.exists(namespaces["authority"]), "authority path absent")
    require(ops.exists(namespaces["credential"]), "credential path absent")
    authority = load_canonical_json_bytes(authority_bytes, "authority")
    credential = load_canonical_json_bytes(credential_bytes, "credential")
    validate_authority_record(authority, package_sha256=package_sha256, evaluator_sha256=evaluator_sha256)
    validate_credential_record(credential, authority)
    ledger = build_consumed_ledger(authority, credential)
    validate_consumed_ledger(ledger, authority, credential)
    durable_create_bytes(namespaces["ledger"], compact_bytes(ledger), ops)
    durable_unlink(namespaces["credential"], ops)

    returncode = 255
    result_bytes: bytes | None = None
    invalid_result = False
    try:
        response = evaluator_invoke(ledger)
        require(type(response) is tuple and len(response) == 2, "evaluator response")
        returncode, result_bytes = response
        require(type(returncode) is int, "evaluator return code")
        require(result_bytes is None or type(result_bytes) is bytes, "evaluator result bytes")
    except Exception:
        result_bytes = None

    validated_result: dict[str, Any] | None = None
    if result_bytes is not None:
        try:
            candidate = load_canonical_json_bytes(result_bytes, "evaluator result")
            validate_result_record(
                package,
                candidate,
                package_sha256=package_sha256,
                evaluator_sha256=evaluator_sha256,
                expected_authority_sha256=authority["authority_sha256"],
                expected_consumed_ledger_sha256=ledger["consumed_ledger_sha256"],
                expected_fresh_l2_acceptance_sha256=authority["fresh_l2_acceptance_sha256"],
                expected_invocation_sha256=authority["invocation_sha256"],
                expected_irreversible_action_id=authority["irreversible_action_id"],
                expected_model_identity_sha256=authority["model_identity_sha256"],
                expected_tensor_bundle_sha256=authority["input_bindings"]["tensor_bundle"]["sha256"],
            )
            classify_terminal(returncode, ledger["consumed_ledger_sha256"], candidate)
            durable_create_bytes(namespaces["result"], result_bytes, ops)
            validated_result = candidate
        except ControllerError:
            invalid_result = True
            validated_result = None
    classification = classify_terminal(
        returncode,
        ledger["consumed_ledger_sha256"],
        validated_result,
        invocation_started=True,
        invalid_result_observed=invalid_result,
    )
    first_terminal = _first_terminal_record(classification, authority)
    durable_create_bytes(namespaces["first_terminal"], compact_bytes(first_terminal), ops)
    return first_terminal


def main() -> int:
    raise RuntimeError("STATIC_ONLY: controller execution requires a separate irreversible action ID")


if __name__ == "__main__":
    raise SystemExit(main())
