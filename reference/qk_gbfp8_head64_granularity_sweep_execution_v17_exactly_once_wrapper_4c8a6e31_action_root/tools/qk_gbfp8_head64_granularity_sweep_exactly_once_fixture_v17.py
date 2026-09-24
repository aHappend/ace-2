
#!/usr/bin/env python3
"""Marker-free synthetic lifecycle regression for the V17 wrapper."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
CORE_TOOLS = Path('/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root/tools')
ACTION_ID = 'ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z'
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(CORE_TOOLS))

from qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v17 import RuntimePaths, compact_bytes, run_execution, sha256_bytes
import qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17 as adapter
import qk_gbfp8_head64_granularity_sweep_nominal_records_v17 as nominal
import qk_gbfp8_head64_granularity_sweep_result_validation_v17 as result_validation


def load_schema(name: str) -> dict[str, Any]:
    raw = (ROOT / "reference" / name).read_bytes()
    value = json.loads(raw.decode("ascii"))
    if compact_bytes(value) != raw:
        raise ValueError("fixture schema canonical")
    return value


def validators() -> dict[str, Any]:
    from jsonschema import Draft202012Validator
    names = {
        "authority": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_AUTHORITY_SCHEMA.json",
        "credential": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_CREDENTIAL_SCHEMA.json",
        "ledger": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_LEDGER_SCHEMA.json",
        "terminal": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_FIRST_TERMINAL_SCHEMA.json",
    }
    return {key: Draft202012Validator(load_schema(name)).validate for key, name in names.items()}


def make_paths(root: Path) -> RuntimePaths:
    directories = [root, root / "primary", root / "primary/authority", root / "primary/authority/base", root / "primary/result", root / "primary/result/base", root / "fallback"]
    for directory in directories:
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
    return RuntimePaths(root / "primary/authority/base/authority.json", root / "primary/authority/base/credential.json",
        root / "primary/authority/base/authority-ledger.json", root / "primary/result/base/result.json",
        root / "primary/authority/base/first-terminal.json", root / "fallback/first-terminal.json")


def bindings() -> dict[str, Any]:
    attestation = {
        "api": "os.posix_spawn", "immediate_parent_argv_sha256": "1" * 64,
        "immediate_parent_environment_sha256": "2" * 64,
        "immediate_parent_executable_sha256": 'fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad',
        "immediate_parent_pid": 1, "immediate_parent_transport_sha256": "3" * 64,
        "launcher_argv_sha256": "4" * 64, "launcher_environment_sha256": "5" * 64,
        "shell": False,
    }
    attestation["transport_attestation_sha256"] = sha256_bytes(compact_bytes(attestation))
    return {
        "action_id": ACTION_ID, "core_acceptance_file_sha256": '9e8eb7a1a7c4340ab8d7b70a2ee97a132169711659f0ca7608d6701ad3225761',
        "core_manifest_file_sha256": '9e120e0e831ec736df45bd1b9f825217ce3368c5e22dbf3e4fc67ac537ea61f1',
        "execution_package_file_sha256": "6" * 64, "fresh_l2_acceptance_sha256": "7" * 64,
        "interpreter_sha256": 'fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad', "invocation_sha256": "8" * 64,
        "launcher_invocation_sha256": "9" * 64,
        "official_identities": {"input_tokens_sha256": "a" * 64, "lane_metadata_sha256": "b" * 64,
            "model_sha256": "c" * 64, "tensor_bundle_sha256": "d" * 64},
        "transport_attestation": attestation,
    }


def result_bytes(authority_sha256: str, ledger_sha256: str, *, invalid_schema: bool = False) -> tuple[bytes, Any]:
    value = nominal.make_result("G4")
    value["authority_sha256"] = authority_sha256
    value["consumed_ledger_sha256"] = ledger_sha256
    value["evaluator_sha256"] = "e" * 64
    value["fresh_l2_acceptance_sha256"] = "7" * 64
    value["invocation_sha256"] = "8" * 64
    value["model_identity_sha256"] = "c" * 64
    value["package_sha256"] = "f" * 64
    value["input_bindings"]["tensor_bundle_sha256"] = "d" * 64
    if invalid_schema:
        value["candidate_results"][0]["metrics"]["top_key"]["row_count"] = 573
    value.pop("result_sha256", None)
    value["result_sha256"] = sha256_bytes(compact_bytes(value))
    schema_validator = result_validation.construct_result_schema_validator(result_validation.frozen_result_schema())
    bound = result_validation.bind_result_validators(schema_validator, result_validation.ResultBindings(
        acceptance_sha256="7" * 64, authority_sha256=authority_sha256, evaluator_sha256="e" * 64,
        invocation_sha256="8" * 64, ledger_sha256=ledger_sha256, model_identity_sha256="c" * 64,
        package_sha256="f" * 64, tensor_bundle_sha256="d" * 64), ACTION_ID)
    return compact_bytes(value), bound


def invoker(scenario: str) -> Callable[..., Any]:
    def invoke(payload: bytes, authority_sha256: str, ledger_sha256: str, publish: Any) -> Any:
        invalid = scenario == "schema"
        raw, bound = result_bytes(authority_sha256, ledger_sha256, invalid_schema=invalid)
        if scenario == "decode":
            raw = b"{malformed\n"
        return_code = 7 if scenario == "nonzero" else 0
        stderr = b"synthetic nonzero\n" if scenario == "nonzero" else b""
        completed = subprocess.CompletedProcess(["synthetic"], return_code, stdout=raw, stderr=stderr)
        spec = adapter.InvocationSpec(argv=("synthetic-v17-adapter", scenario), cwd="/synthetic", environment=(("LANG", "C"),), max_capture_bytes=1 << 20)
        return adapter.invoke_evaluator(spec, b"synthetic-envelope", bound.validate_result_schema, bound.validate_result_record, publish, runner=lambda *args, **kwargs: completed)
    return invoke


def read_terminal(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def run_case(name: str, operation: Callable[[RuntimePaths], dict[str, Any]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v17-wrapper-fixture-") as temporary:
        paths = make_paths(Path(temporary) / "runtime")
        try:
            observed = operation(paths)
            passed = bool(observed.pop("passed"))
            return {"name": name, "passed": passed, **observed}
        except BaseException as error:
            return {"name": name, "passed": False, "error_type": type(error).__name__}


def execute(paths: RuntimePaths, scenario: str = "nominal", *, faults: frozenset[str] = frozenset(), preflight_error: bool = False) -> Any:
    def preflight() -> None:
        if preflight_error:
            raise ValueError("synthetic preflight")
    return run_execution(paths, bindings(), validators(), preflight, lambda: b"synthetic-payload", invoker(scenario), faults=faults)


def run_fixture() -> dict[str, Any]:
    cases = []
    cases.append(run_case("nominal", lambda p: (lambda o: {"passed": o.status == "SUCCEEDED_TERMINAL" and o.counts.invocation_count == 1 and o.counts.payload_open_count == 1 and p.result.exists() and p.ledger.exists() and not p.credential.exists(), "status": o.status})(execute(p))))
    cases.append(run_case("preflight_failure", lambda p: (lambda o: {"passed": o.status == "PREFLIGHT_FAILED_TERMINAL" and o.counts.invocation_count == 0 and not p.authority.exists(), "status": o.status})(execute(p, preflight_error=True))))
    cases.append(run_case("authority_failure", lambda p: (lambda o: {"passed": o.status == "FAILED_TERMINAL" and o.counts.invocation_count == 0 and not p.authority.exists(), "status": o.status})(execute(p, faults=frozenset({"AUTHORITY_WRITE"})))))
    cases.append(run_case("credential_failure", lambda p: (lambda o: {"passed": o.status == "FAILED_TERMINAL" and o.counts.invocation_count == 0 and p.authority.exists() and not p.credential.exists() and not p.ledger.exists(), "status": o.status})(execute(p, faults=frozenset({"CREDENTIAL_WRITE"})))))
    cases.append(run_case("ledger_failure", lambda p: (lambda o: {"passed": o.status == "FAILED_TERMINAL" and o.counts.invocation_count == 0 and not p.credential.exists() and not p.ledger.exists(), "status": o.status})(execute(p, faults=frozenset({"LEDGER_WRITE"})))))

    def duplicate(p: RuntimePaths) -> dict[str, Any]:
        first = execute(p)
        before = p.first_terminal.read_bytes()
        second = execute(p)
        return {"passed": first.status == "SUCCEEDED_TERMINAL" and second.duplicate_rejected and second.counts.invocation_count == 0 and p.first_terminal.read_bytes() == before, "status": second.status}
    cases.append(run_case("duplicate_invocation", duplicate))
    for name, scenario, failure_class in (("nonzero_evaluator", "nonzero", "RETURN_CODE"), ("decode_failure", "decode", "DECODE"), ("schema_failure", "schema", "RESULT_SCHEMA")):
        def operation(p: RuntimePaths, scenario: str = scenario, failure_class: str = failure_class) -> dict[str, Any]:
            outcome = execute(p, scenario)
            terminal = read_terminal(Path(outcome.terminal_path))
            return {"passed": outcome.status == "CONSUMED_ORPHAN" and outcome.counts.invocation_count == 1 and terminal["evaluator_diagnostic"]["failure_class"] == failure_class, "failure_class": terminal["evaluator_diagnostic"]["failure_class"], "status": outcome.status}
        cases.append(run_case(name, operation))
    cases.append(run_case("result_publication_failure", lambda p: (lambda o: {"passed": o.status == "CONSUMED_ORPHAN" and o.counts.invocation_count == 1 and read_terminal(Path(o.terminal_path))["evaluator_diagnostic"]["failure_class"] == "PUBLICATION", "status": o.status})(execute(p, faults=frozenset({"RESULT_WRITE"})))))
    cases.append(run_case("terminal_primary_fallback", lambda p: (lambda o: {"passed": o.status == "CONSUMED_ORPHAN" and o.terminal_path == str(p.fallback_terminal) and p.result.exists(), "status": o.status})(execute(p, faults=frozenset({"TERMINAL_PRIMARY"})))))
    cases.append(run_case("terminal_total_failure", lambda p: (lambda o: {"passed": o.status == "CONSUMED_ORPHAN_UNPUBLISHED" and o.terminal_publication_failed and o.counts.invocation_count == 1 and not p.first_terminal.exists() and not p.fallback_terminal.exists(), "status": o.status})(execute(p, faults=frozenset({"TERMINAL_PRIMARY", "TERMINAL_FALLBACK"})))))
    passed = all(case["passed"] for case in cases)
    return {
        "action_id": ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_synthetic_fixture_report",
        "case_count": len(cases),
        "cases": cases,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "production_mode_exercised": False,
        "shared_adapter_sha256": '14b236ae1fdf10028088520c0816df0d65783764e6f5a6a385aa8bd879e84de3',
        "status": "PASS_V17_EXACTLY_ONCE_SYNTHETIC_FIXTURE" if passed else "FAIL_V17_EXACTLY_ONCE_SYNTHETIC_FIXTURE",
        "synthetic_bytes_only": True,
    }


def main() -> int:
    report = run_fixture()
    sys.stdout.buffer.write(compact_bytes(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
