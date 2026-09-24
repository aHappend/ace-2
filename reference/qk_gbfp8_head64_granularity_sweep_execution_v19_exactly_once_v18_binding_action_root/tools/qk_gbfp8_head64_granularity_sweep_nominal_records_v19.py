#!/usr/bin/env python3
"""Marker-free synthetic fixture using the exact V16 production result validators."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Callable

from qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v19 import (
    EvaluatorReturnPathError,
    call_evaluator_once,
    compact_bytes,
    complete_result_path,
    process_evaluator_return,
)
from qk_gbfp8_head64_granularity_sweep_result_validation_v19 import (
    RESULT_SCHEMA_SHA256,
    ResultBindings,
    bind_result_validators,
    construct_result_schema_validator,
    frozen_result_schema,
)


ACTION_ID = "ace2:qk-gbfp8-base-v19:execute-once:fdfc33a1:additive-0001"
LABELS = ("G8", "G4", "G2", "G1")
CANDIDATE_SPECS = {
    "G8": (8, 8, 80, 16),
    "G4": (4, 16, 96, 32),
    "G2": (2, 32, 128, 64),
    "G1": (1, 64, 192, 128),
}
SYNTHETIC_BINDINGS = ResultBindings(
    acceptance_sha256="a" * 64,
    authority_sha256="b" * 64,
    evaluator_sha256="c" * 64,
    invocation_sha256="d" * 64,
    ledger_sha256="e" * 64,
    model_identity_sha256="f" * 64,
    package_sha256="0" * 64,
    tensor_bundle_sha256="1" * 64,
)
RESULT_VALIDATOR = construct_result_schema_validator(frozen_result_schema())
RESULT_VALIDATION = bind_result_validators(RESULT_VALIDATOR, SYNTHETIC_BINDINGS, ACTION_ID)


def count_gate(actual: int) -> dict[str, Any]:
    return {"actual": actual, "limit": 0, "pass": actual == 0}


def make_metrics(passing: bool) -> dict[str, Any]:
    saturation_events = 0 if passing else 1
    return {
        "invalid_accounting": {
            "cross_lane_record_count": 0,
            "invalid_or_non_finite_value_count": 0,
            "normalization_rejection_count": 0,
            "positive_centered_realized_score_count": 0,
            "saturation_event_count": saturation_events,
        },
        "rank_margin": {
            "minimum_realized_margin_q12_20_lsb": 1,
            "preserved_positive_margin_fraction": {"denominator": 346, "numerator": 346},
            "preserved_positive_margin_row_count": 346,
            "unique_oracle_top_row_count": 346,
            "violation_count": 0,
        },
        "score_error": {
            "maximum_absolute_error_q12_20_lsb": 0,
            "sum_absolute_error_q12_20_lsb": 0,
            "sum_signed_error_q12_20_lsb": 0,
            "sum_squared_error_q40_40_lsb2": 0,
            "valid_value_count": 12054,
        },
        "top_key": {
            "matching_fraction": {"denominator": 574, "numerator": 574},
            "matching_row_count": 574,
            "mismatch_count": 0,
            "row_count": 574,
        },
    }


def make_thresholds(passing: bool) -> dict[str, Any]:
    saturation_events = 0 if passing else 1
    return {
        "all_hard_gates_pass": passing,
        "cross_lane_record_count_maximum": count_gate(0),
        "invalid_or_non_finite_value_count_maximum": count_gate(0),
        "normalization_rejection_count_maximum": count_gate(0),
        "positive_centered_realized_score_count_maximum": count_gate(0),
        "rank_margin_violation_count_maximum": count_gate(0),
        "saturation_event_count_maximum": count_gate(saturation_events),
        "top_key_matching_fraction_minimum": {"actual": {"denominator": 574, "numerator": 574}, "limit": {"denominator": 1, "numerator": 1}, "pass": True},
        "top_key_mismatch_count_maximum": count_gate(0),
        "unique_oracle_positive_margin_preserved_fraction_minimum": {"actual": {"denominator": 346, "numerator": 346}, "limit": {"denominator": 1, "numerator": 1}, "pass": True},
    }


def make_candidate(label: str, passing: bool) -> dict[str, Any]:
    group_size, group_count, bytes_per_head, exponent_bytes_per_head = CANDIDATE_SPECS[label]
    return {
        "bytes_per_head": bytes_per_head,
        "exponent_bytes_per_head": exponent_bytes_per_head,
        "group_count": group_count,
        "group_size": group_size,
        "label": label,
        "mantissa_bytes_per_head": 64,
        "metrics": make_metrics(passing),
        "threshold_evaluation": make_thresholds(passing),
    }


def make_result(first_passing: str | None) -> dict[str, Any]:
    passing_index = LABELS.index(first_passing) if first_passing is not None else len(LABELS)
    candidates = [make_candidate(label, index >= passing_index) for index, label in enumerate(LABELS)]
    passing = [item["label"] for item in candidates if item["threshold_evaluation"]["all_hard_gates_pass"]]
    selected = passing[0] if passing else None
    result = {
        "authority_sha256": SYNTHETIC_BINDINGS.authority_sha256,
        "candidate_results": candidates,
        "consumed_ledger_sha256": SYNTHETIC_BINDINGS.ledger_sha256,
        "evaluator_sha256": SYNTHETIC_BINDINGS.evaluator_sha256,
        "fresh_l2_acceptance_sha256": SYNTHETIC_BINDINGS.acceptance_sha256,
        "generation_id": "qk-gbfp8-head64-granularity-sweep-v8-base",
        "input_bindings": {"sealed_set_id": "w4a8-c02-attention-substage-trace-v2", "tensor_bundle_sha256": SYNTHETIC_BINDINGS.tensor_bundle_sha256, "tensor_record_count": 25},
        "invocation_sha256": SYNTHETIC_BINDINGS.invocation_sha256,
        "irreversible_action_id": ACTION_ID,
        "lane_label": "Base",
        "model_identity_sha256": SYNTHETIC_BINDINGS.model_identity_sha256,
        "namespace_label": "base",
        "package_id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE",
        "package_sha256": SYNTHETIC_BINDINGS.package_sha256,
        "schema_id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_RESULT_V1",
        "selected_candidate": selected,
        "selection": {"passing_candidates": passing, "policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER", "selected_candidate": selected},
        "terminal": {
            "first_record_immutable": True,
            "invocation_count_performed": 1,
            "metrics_published": True,
            "reason_code": "HARD_THRESHOLDS_PASSED" if selected is not None else "HARD_THRESHOLD_FAILED",
            "retry_replay_resume_repair_permitted": False,
            "status": "SUCCEEDED_TERMINAL" if selected is not None else "FAILED_TERMINAL",
            "tensor_open_count": 1,
            "thresholds_evaluated": True,
        },
    }
    result["result_sha256"] = hashlib.sha256(compact_bytes(result)).hexdigest()
    return result


def pipeline(raw: bytes, *, result_exists: bool = False, terminal_build_error: bool = False, terminal_schema_error: bool = False, terminal_publication_error: bool = False) -> dict[str, Any]:
    processed = process_evaluator_return(raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record)
    store: dict[str, Any] = {"result": b"occupied"} if result_exists else {}

    def publish_result(value: bytes) -> None:
        if "result" in store:
            raise FileExistsError("create-once result collision")
        store["result"] = value

    def build_terminal(record: dict[str, Any], durable: bytes) -> dict[str, Any]:
        if terminal_build_error:
            raise ValueError("synthetic terminal build failure")
        return {"failure_stage": None, "result_sha256": hashlib.sha256(durable).hexdigest(), "reason_code": record["terminal"]["reason_code"], "status": record["terminal"]["status"]}

    def validate_terminal(record: dict[str, Any]) -> None:
        if terminal_schema_error:
            raise ValueError("synthetic terminal schema failure")
        if set(record) != {"failure_stage", "reason_code", "result_sha256", "status"} or record["failure_stage"] is not None:
            raise ValueError("terminal record")

    def publish_terminal(record: dict[str, Any]) -> None:
        if terminal_publication_error:
            raise OSError("synthetic terminal publication failure")
        if "terminal" in store:
            raise FileExistsError("create-once terminal collision")
        store["terminal"] = copy.deepcopy(record)

    terminal = complete_result_path(processed, publish_result, build_terminal, validate_terminal, publish_terminal)
    return {"result": store["result"], "terminal": terminal}


def observe(name: str, expected_stage: str | None, operation: Callable[[], Any]) -> dict[str, Any]:
    observed_stage: str | None = None
    detail = ""
    try:
        operation()
    except EvaluatorReturnPathError as error:
        observed_stage = error.failure_stage
        detail = error.error_type
    except BaseException as error:
        detail = f"UNSTRUCTURED:{type(error).__name__}"
    return {"detail_class": detail, "expected_stage": expected_stage, "name": name, "observed_stage": observed_stage, "passed": observed_stage == expected_stage}


def callable_identity(value: Any) -> str:
    target = getattr(value, "func", value)
    return f"{target.__module__}.{target.__qualname__}"


def run_fixture() -> dict[str, Any]:
    all_fail = make_result(None)
    first_pass = make_result("G4")
    for nominal in (all_fail, first_pass):
        raw = compact_bytes(nominal)
        RESULT_VALIDATION.validate_result_schema(nominal)
        RESULT_VALIDATION.validate_result_record(nominal, raw)
    schema_bad = copy.deepcopy(all_fail)
    schema_bad["candidate_results"][0]["metrics"]["top_key"]["row_count"] = 573
    exact_key_bad = copy.deepcopy(all_fail)
    exact_key_bad["unexpected"] = True
    binding_bad = copy.deepcopy(all_fail)
    binding_bad["package_sha256"] = "2" * 64
    checksum_bad = copy.deepcopy(all_fail)
    checksum_bad["result_sha256"] = "3" * 64
    cases = [
        observe("all_hard_gates_fail", None, lambda: pipeline(compact_bytes(all_fail))),
        observe("first_passing_candidate", None, lambda: pipeline(compact_bytes(first_pass))),
        observe("evaluator_call_failure", "EVALUATOR_CALL", lambda: call_evaluator_once(lambda: (_ for _ in ()).throw(RuntimeError("synthetic evaluator call failure")))),
        observe("noncanonical_json", "CANONICAL_DECODE", lambda: process_evaluator_return((json.dumps(all_fail, indent=2, sort_keys=True) + "\n").encode("ascii"), RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record)),
        observe("schema_mismatch", "RESULT_SCHEMA", lambda: process_evaluator_return(compact_bytes(schema_bad), RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record)),
        observe("exact_key_mismatch", "RESULT_SCHEMA", lambda: process_evaluator_return(compact_bytes(exact_key_bad), RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record)),
        observe("binding_mismatch", "RESULT_RECORD", lambda: process_evaluator_return(compact_bytes(binding_bad), RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record)),
        observe("self_checksum_mismatch", "RESULT_RECORD", lambda: process_evaluator_return(compact_bytes(checksum_bad), RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record)),
        observe("create_once_result_collision", "RESULT_PUBLICATION", lambda: pipeline(compact_bytes(all_fail), result_exists=True)),
        observe("terminal_build_failure", "TERMINAL_BUILD", lambda: pipeline(compact_bytes(all_fail), terminal_build_error=True)),
        observe("terminal_validation_failure", "TERMINAL_SCHEMA", lambda: pipeline(compact_bytes(all_fail), terminal_schema_error=True)),
        observe("terminal_publication_failure", "TERMINAL_PUBLICATION", lambda: pipeline(compact_bytes(all_fail), terminal_publication_error=True)),
    ]
    return {
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v19_nominal_records_fixture_report",
        "case_count": len(cases),
        "cases": cases,
        "fixture_reads_result_schema_file": False,
        "fixture_uses_official_evaluator": False,
        "fixture_uses_official_payload": False,
        "nominal_record_valid_case_count": 2,
        "nominal_schema_conforming_case_count": 2,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "result_schema_sha256": RESULT_SCHEMA_SHA256,
        "schema_validator_class": f"{type(RESULT_VALIDATOR).__module__}.{type(RESULT_VALIDATOR).__qualname__}",
        "schema_validator_callable": callable_identity(RESULT_VALIDATION.validate_result_schema),
        "record_validator_callable": callable_identity(RESULT_VALIDATION.validate_result_record),
        "validator_factory": "qk_gbfp8_head64_granularity_sweep_result_validation_v19.bind_result_validators",
        "status": "PASS_MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE" if all(case["passed"] for case in cases) else "FAIL_MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE",
    }


def main() -> int:
    report = run_fixture()
    print(json.dumps(report, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
