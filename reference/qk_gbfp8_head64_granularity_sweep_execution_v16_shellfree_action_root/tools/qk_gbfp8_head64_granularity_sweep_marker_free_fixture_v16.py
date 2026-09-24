#!/usr/bin/env python3
"""Marker-free synthetic fixture for the shared V16 evaluator-return path."""

from __future__ import annotations

import copy
import json
from typing import Any, Callable

from qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v16 import (
    EvaluatorReturnPathError,
    call_evaluator_once,
    compact_bytes,
    complete_result_path,
    process_evaluator_return,
)


LABELS = ("G8", "G4", "G2", "G1")
EXPECTED_KEYS = {"binding_sha256", "candidate_results", "selected_candidate", "selection", "terminal"}
SYNTHETIC_BINDING = "a" * 64


def make_result(first_passing: str | None) -> dict[str, Any]:
    candidates = [
        {"label": label, "all_hard_gates_pass": first_passing is not None and LABELS.index(label) >= LABELS.index(first_passing)}
        for label in LABELS
    ]
    passing = [item["label"] for item in candidates if item["all_hard_gates_pass"]]
    selected = passing[0] if passing else None
    return {
        "binding_sha256": SYNTHETIC_BINDING,
        "candidate_results": candidates,
        "selected_candidate": selected,
        "selection": {"passing_candidates": passing, "policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER", "selected_candidate": selected},
        "terminal": {
            "reason_code": "HARD_THRESHOLDS_PASSED" if selected is not None else "HARD_THRESHOLD_FAILED",
            "status": "SUCCEEDED_TERMINAL" if selected is not None else "FAILED_TERMINAL",
        },
    }


def schema_validate(record: dict[str, Any]) -> None:
    if not isinstance(record.get("binding_sha256"), str):
        raise ValueError("binding type")
    if not isinstance(record.get("candidate_results"), list) or len(record["candidate_results"]) != 4:
        raise ValueError("candidate schema")
    if record.get("selected_candidate") is not None and record.get("selected_candidate") not in LABELS:
        raise ValueError("selected candidate schema")
    if not isinstance(record.get("selection"), dict) or not isinstance(record.get("terminal"), dict):
        raise ValueError("nested schema")


def record_validate(record: dict[str, Any], raw: bytes) -> None:
    if set(record) != EXPECTED_KEYS:
        raise ValueError("exact keys")
    if record["binding_sha256"] != SYNTHETIC_BINDING:
        raise ValueError("binding mismatch")
    if compact_bytes(record) != raw:
        raise ValueError("canonical record bytes")
    labels = [item.get("label") for item in record["candidate_results"]]
    if labels != list(LABELS):
        raise ValueError("candidate order")
    passing = [item["label"] for item in record["candidate_results"] if item.get("all_hard_gates_pass") is True]
    selected = passing[0] if passing else None
    if record["selected_candidate"] != selected:
        raise ValueError("selection mismatch")
    if record["selection"] != {"passing_candidates": passing, "policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER", "selected_candidate": selected}:
        raise ValueError("selection record mismatch")


def pipeline(raw: bytes, *, result_exists: bool = False, terminal_build_error: bool = False, terminal_schema_error: bool = False, terminal_publication_error: bool = False) -> dict[str, Any]:
    processed = process_evaluator_return(raw, schema_validate, record_validate)
    store: dict[str, Any] = {"result": b"occupied"} if result_exists else {}

    def publish_result(value: bytes) -> None:
        if "result" in store:
            raise FileExistsError("create-once result collision")
        store["result"] = value

    def build_terminal(record: dict[str, Any], durable: bytes) -> dict[str, Any]:
        if terminal_build_error:
            raise ValueError("synthetic terminal build failure")
        return {"failure_stage": None, "result_sha256": __import__("hashlib").sha256(durable).hexdigest(), **record["terminal"]}

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
    return {
        "detail_class": detail,
        "expected_stage": expected_stage,
        "name": name,
        "observed_stage": observed_stage,
        "passed": observed_stage == expected_stage,
    }


def run_fixture() -> dict[str, Any]:
    all_fail = make_result(None)
    first_pass = make_result("G4")
    schema_bad = copy.deepcopy(all_fail)
    schema_bad["selected_candidate"] = 7
    exact_key_bad = copy.deepcopy(all_fail)
    exact_key_bad["unexpected"] = True
    binding_bad = copy.deepcopy(all_fail)
    binding_bad["binding_sha256"] = "b" * 64
    cases = [
        observe("all_hard_gates_fail", None, lambda: pipeline(compact_bytes(all_fail))),
        observe("first_passing_candidate", None, lambda: pipeline(compact_bytes(first_pass))),
        observe("evaluator_call_failure", "EVALUATOR_CALL", lambda: call_evaluator_once(lambda: (_ for _ in ()).throw(RuntimeError("synthetic evaluator call failure")))),
        observe("noncanonical_json", "CANONICAL_DECODE", lambda: process_evaluator_return((json.dumps(all_fail, indent=2, sort_keys=True) + "\n").encode("ascii"), schema_validate, record_validate)),
        observe("schema_mismatch", "RESULT_SCHEMA", lambda: process_evaluator_return(compact_bytes(schema_bad), schema_validate, record_validate)),
        observe("exact_key_mismatch", "RESULT_RECORD", lambda: process_evaluator_return(compact_bytes(exact_key_bad), schema_validate, record_validate)),
        observe("binding_mismatch", "RESULT_RECORD", lambda: process_evaluator_return(compact_bytes(binding_bad), schema_validate, record_validate)),
        observe("create_once_result_collision", "RESULT_PUBLICATION", lambda: pipeline(compact_bytes(all_fail), result_exists=True)),
        observe("terminal_build_failure", "TERMINAL_BUILD", lambda: pipeline(compact_bytes(all_fail), terminal_build_error=True)),
        observe("terminal_validation_failure", "TERMINAL_SCHEMA", lambda: pipeline(compact_bytes(all_fail), terminal_schema_error=True)),
        observe("terminal_publication_failure", "TERMINAL_PUBLICATION", lambda: pipeline(compact_bytes(all_fail), terminal_publication_error=True)),
    ]
    return {
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v16_marker_free_fixture_report",
        "case_count": len(cases),
        "cases": cases,
        "fixture_uses_official_evaluator": False,
        "fixture_uses_official_payload": False,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "status": "PASS_MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE" if all(case["passed"] for case in cases) else "FAIL_MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE",
    }


def main() -> int:
    report = run_fixture()
    print(json.dumps(report, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
