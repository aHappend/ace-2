#!/usr/bin/env python3
"""Synthetic-only regression fixture for the shared V17 adapter."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import jsonschema

from qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17 import (
    EvaluatorInvocationError,
    InvocationSpec,
    compact_bytes,
    invoke_evaluator,
)
from qk_gbfp8_head64_granularity_sweep_nominal_records_v17 import RESULT_VALIDATION, make_result


ACTION_ID = 'ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z'
ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v17.py"
TERMINAL_SCHEMA = ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_FIRST_TERMINAL_SCHEMA.json"
INTERPRETER = '/home/argustest/miniconda3/bin/python3.13'
EXACT_ENVIRONMENT = (('LANG', 'C'), ('LC_ALL', 'C'), ('PYTHONDONTWRITEBYTECODE', '1'), ('PYTHONHASHSEED', '0'), ('TZ', 'UTC'))


def spec(scenario: str) -> InvocationSpec:
    return InvocationSpec(
        argv=(INTERPRETER, str(WORKER), "--mode", "synthetic", "--scenario", scenario),
        cwd=str(ROOT),
        environment=EXACT_ENVIRONMENT,
        max_capture_bytes=16 * 1024 * 1024,
    )


def terminal(diagnostic: dict[str, Any], failure_stage: str) -> dict[str, Any]:
    return {
        "action_id": ACTION_ID,
        "action_retired": True,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_first_terminal",
        "evaluator_diagnostic": diagnostic,
        "failure_detail_sha256": hashlib.sha256(compact_bytes(diagnostic)).hexdigest(),
        "failure_stage": failure_stage,
        "first_record_immutable": True,
        "invocation_count_performed": 1,
        "official_target_process_starts": 1 if diagnostic["process"]["started"] else 0,
        "payload_open_count": 1,
        "reason_code": "POST_CONSUMPTION_AMBIGUITY",
        "result_file_sha256": None,
        "retry_replay_resume_repair_replacement_permitted": False,
        "status": "CONSUMED_ORPHAN",
    }


def observe(
    name: str,
    expected: str,
    operation: Callable[[], Any],
    terminal_validator: Any,
) -> dict[str, Any]:
    diagnostic: dict[str, Any] | None = None
    try:
        operation()
    except EvaluatorInvocationError as error:
        diagnostic = error.diagnostic
    passed = diagnostic is not None and diagnostic["failure_class"] == expected
    terminal_valid = False
    if diagnostic is not None:
        stage = "RESULT_PUBLICATION" if expected == "PUBLICATION" else "EVALUATOR_CALL"
        terminal_validator.validate(terminal(diagnostic, stage))
        terminal_valid = True
    return {
        "diagnostic": diagnostic,
        "expected_failure_class": expected,
        "name": name,
        "passed": passed and terminal_valid,
        "terminal_schema_valid": terminal_valid,
    }


def run_fixture() -> dict[str, Any]:
    terminal_schema = json.loads(TERMINAL_SCHEMA.read_text(encoding="ascii"))
    jsonschema.Draft202012Validator.check_schema(terminal_schema)
    terminal_validator = jsonschema.Draft202012Validator(terminal_schema)
    nominal_results = (make_result(None), make_result("G4"))
    nominal_valid = 0
    success_diagnostics: list[dict[str, Any]] = []
    for result in nominal_results:
        raw = compact_bytes(result)
        RESULT_VALIDATION.validate_result_schema(result)
        RESULT_VALIDATION.validate_result_record(result, raw)
        published: list[bytes] = []
        observed = invoke_evaluator(
            spec("nominal"),
            raw,
            RESULT_VALIDATION.validate_result_schema,
            RESULT_VALIDATION.validate_result_record,
            published.append,
        )
        if observed.raw == raw and published == [raw] and observed.diagnostic["outcome"] == "SUCCEEDED":
            nominal_valid += 1
            success_diagnostics.append(observed.diagnostic)

    all_fail_raw = compact_bytes(nominal_results[0])
    cases = [
        observe("evaluator_nonzero_exit", "RETURN_CODE", lambda: invoke_evaluator(spec("nonzero"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None), terminal_validator),
        observe("evaluator_no_output", "STDOUT", lambda: invoke_evaluator(spec("no-output"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None), terminal_validator),
        observe("evaluator_malformed_output", "DECODE", lambda: invoke_evaluator(spec("malformed"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None), terminal_validator),
        observe("evaluator_stderr", "STDERR", lambda: invoke_evaluator(spec("stderr"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None), terminal_validator),
        observe("evaluator_spawn_failure", "SPAWN", lambda: invoke_evaluator(InvocationSpec(argv=("/definitely/absent/v17-evaluator",), cwd=str(ROOT), environment=EXACT_ENVIRONMENT, max_capture_bytes=1024), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None), terminal_validator),
        observe("evaluator_abi_failure", "ABI", lambda: invoke_evaluator(spec("nominal"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None, runner=lambda *args, **kwargs: object()), terminal_validator),
        observe("evaluator_exception", "EXCEPTION", lambda: invoke_evaluator(spec("nominal"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: None, runner=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic runner exception"))), terminal_validator),
        observe("result_publication_failure", "PUBLICATION", lambda: invoke_evaluator(spec("nominal"), all_fail_raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record, lambda raw: (_ for _ in ()).throw(OSError("synthetic publication failure"))), terminal_validator),
    ]
    required = {"evaluator_nonzero_exit", "evaluator_no_output", "evaluator_malformed_output", "evaluator_exception", "result_publication_failure"}
    names = {case["name"] for case in cases if case["passed"]}
    status = "PASS_V17_EVALUATOR_ADAPTER_SYNTHETIC_FIXTURE" if nominal_valid == 2 and required <= names and all(case["passed"] for case in cases) else "FAIL_V17_EVALUATOR_ADAPTER_SYNTHETIC_FIXTURE"
    return {
        "adapter_case_count": len(cases),
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_adapter_fixture_report",
        "cases": cases,
        "exact_result_record_validator": "qk_gbfp8_head64_granularity_sweep_result_validation_v17.validate_result_record_exact",
        "exact_result_schema_validator": "qk_gbfp8_head64_granularity_sweep_result_validation_v17.validate_result_schema_exact",
        "nominal_frozen_schema_record_count": nominal_valid,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "production_mode_exercised": False,
        "required_fail_closed_cases": sorted(required),
        "status": status,
        "synthetic_bytes_only": True,
    }


def main() -> int:
    report = run_fixture()
    print(json.dumps(report, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
