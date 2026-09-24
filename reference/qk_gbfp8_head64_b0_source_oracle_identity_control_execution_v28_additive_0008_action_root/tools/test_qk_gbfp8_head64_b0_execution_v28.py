#!/home/argustest/miniconda3/bin/python3.13
"""Protected-data-free, non-consuming regression for the inert V28 action."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator, ValidationError

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import execute_qk_gbfp8_head64_b0_source_oracle_identity_control_once_v28 as wrapper
import qk_gbfp8_head64_b0_evaluator_adapter_v28 as adapter
import qk_gbfp8_head64_b0_source_oracle_worker_v28 as worker
from qk_gbfp8_head64_b0_execution_common_v28 import (
    ACTION_ID, ACTION_ROOT, EXACT_ENVIRONMENT, EXECUTION_BINDING, RUNTIME_ROOT, SOURCE_STATIC_ACCEPTANCE,
    compact_bytes, sealed, sha256_bytes, validate_external_records,
)


AUDIT = {"official_payload_open_count": 0, "process_start_count": 0}


class RegressionError(RuntimeError):
    pass


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise RegressionError(detail)


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            candidate = Path(args[0]).resolve()
        except (TypeError, OSError, RuntimeError):
            candidate = None
        if candidate == wrapper.OFFICIAL_PAYLOAD.resolve():
            AUDIT["official_payload_open_count"] += 1
            raise RegressionError("official payload open prohibited")
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.posix_spawnp"}:
        AUDIT["process_start_count"] += 1
        raise RegressionError("process start prohibited")


def synthetic_rows() -> list[list[int]]:
    rows: list[list[int]] = []
    tied_remaining = 214
    for head in range(14):
        for query in range(41):
            if query == 0:
                rows.append([0])
            elif tied_remaining:
                rows.append([0] * (query + 1))
                tied_remaining -= 1
            else:
                rows.append([0] + [-(key + 1) for key in range(query)])
    require(tied_remaining == 0 and len(rows) == 574, "synthetic partition construction")
    return rows


def sidecar_cases() -> dict[str, Any]:
    rows = synthetic_rows()
    match = worker.build_sidecar(rows, copy.deepcopy(rows), "a" * 64)
    require(match["outcome"] == "SOURCE_ORACLE_MATCH", "match outcome")
    mismatch_rows = copy.deepcopy(rows)
    mismatch_rows[-1][-1] -= 1
    mismatch = worker.build_sidecar(mismatch_rows, rows, "b" * 64)
    require(mismatch["outcome"] == "SOURCE_ORACLE_MISMATCH", "mismatch outcome")
    require(match["partition_accounting"]["partition_sum"] == 574, "partition closure")
    forbidden_keys = {"payload", "payload_base64", "preview_ascii", "input_token_ids", "token_ids", "tensor_values"}

    def keys(value: Any) -> set[str]:
        if type(value) is dict:
            return set(value) | {item for child in value.values() for item in keys(child)}
        if type(value) is list:
            return {item for child in value for item in keys(child)}
        return set()

    require(not (keys(match) & forbidden_keys), "aggregate leakage boundary")
    return {"classification_cases": 2, "row_count": 574, "safe_aggregate_only": True}


def completed(return_code: int, stdout: bytes = b"", stderr: bytes = b"") -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(["synthetic"], return_code, stdout=stdout, stderr=stderr)


def expect_adapter_failure(name: str, expected: str, operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        operation()
    except adapter.EvaluatorInvocationError as error:
        require(error.diagnostic["failure_class"] == expected, f"{name} failure class")
        require("preview_ascii" not in json.dumps(error.diagnostic, sort_keys=True), f"{name} diagnostic leakage")
        return {"failure_class": expected, "name": name, "passed": True}
    raise RegressionError(f"{name} was not rejected")


def adapter_cases() -> list[dict[str, Any]]:
    rows = synthetic_rows()
    sidecar = worker.build_sidecar(rows, copy.deepcopy(rows), "c" * 64)
    raw = compact_bytes(sidecar)
    spec = adapter.InvocationSpec(("/synthetic/interpreter", "/synthetic/worker"), "/synthetic", tuple(EXACT_ENVIRONMENT.items()), 1 << 20)
    validate = lambda record: None
    record_validate = lambda record, observed: require(compact_bytes(record) == observed, "record canonical")
    published: list[bytes] = []
    nominal = adapter.invoke_evaluator(spec, b"synthetic", validate, record_validate, published.append, runner=lambda *args, **kwargs: completed(0, raw))
    require(nominal.record["outcome"] == "SOURCE_ORACLE_MATCH" and published == [raw], "nominal adapter")
    invalid_abi = object()
    cases = [
        expect_adapter_failure("spawn", "SPAWN", lambda: adapter.invoke_evaluator(spec, b"x", validate, record_validate, published.append, runner=lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic")))),
        expect_adapter_failure("abi", "ABI", lambda: adapter.invoke_evaluator(spec, b"x", validate, record_validate, published.append, runner=lambda *a, **k: invalid_abi)),
        expect_adapter_failure("nonzero", "RETURN_CODE", lambda: adapter.invoke_evaluator(spec, b"x", validate, record_validate, published.append, runner=lambda *a, **k: completed(7))),
        expect_adapter_failure("missing_stdout", "STDOUT", lambda: adapter.invoke_evaluator(spec, b"x", validate, record_validate, published.append, runner=lambda *a, **k: completed(0))),
        expect_adapter_failure("stderr", "STDERR", lambda: adapter.invoke_evaluator(spec, b"x", validate, record_validate, published.append, runner=lambda *a, **k: completed(0, raw, b"synthetic"))),
        expect_adapter_failure("decode", "DECODE", lambda: adapter.invoke_evaluator(spec, b"x", validate, record_validate, published.append, runner=lambda *a, **k: completed(0, b"{bad\n"))),
        expect_adapter_failure("schema", "RESULT_SCHEMA", lambda: adapter.invoke_evaluator(spec, b"x", lambda record: (_ for _ in ()).throw(ValueError("schema")), record_validate, published.append, runner=lambda *a, **k: completed(0, raw))),
        expect_adapter_failure("record", "RESULT_RECORD", lambda: adapter.invoke_evaluator(spec, b"x", validate, lambda *args: (_ for _ in ()).throw(ValueError("record")), published.append, runner=lambda *a, **k: completed(0, raw))),
        expect_adapter_failure("publication", "PUBLICATION", lambda: adapter.invoke_evaluator(spec, b"x", validate, record_validate, lambda raw: (_ for _ in ()).throw(OSError("publication")), runner=lambda *a, **k: completed(0, raw))),
    ]
    cases.append({"failure_class": None, "name": "nominal", "passed": True})
    return cases


def synthetic_package() -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    package = sealed({
        "action_identity": {"action_id": ACTION_ID},
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "execution_binding": EXECUTION_BINDING,
    }, "package_content_sha256")
    package_raw = compact_bytes(package)
    acceptance = sealed({
        "accepted": True, "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "decision": "ACCEPT_STATIC_PACKAGE", "execution_package_content_sha256": package["package_content_sha256"],
        "execution_package_file_sha256": sha256_bytes(package_raw),
        "static_acceptance_grants_execution_authority": False,
    }, "acceptance_sha256")
    return package, package_raw, acceptance, compact_bytes(acceptance)


def authority_records(package_raw: bytes, package: dict[str, Any], acceptance_raw: bytes, acceptance: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    common = {
        "execution_binding": EXECUTION_BINDING,
        "execution_package_acceptance_file_sha256": sha256_bytes(acceptance_raw),
        "execution_package_acceptance_self_sha256": acceptance["acceptance_sha256"],
        "execution_package_content_sha256": package["package_content_sha256"],
        "execution_package_file_sha256": sha256_bytes(package_raw),
        "fresh": True,
        "future_execution_action_id": ACTION_ID,
        "issued_at_utc": (now - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "not_after_utc": (now + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reservation_nonce_sha256": "1" * 64,
        "schema_version": 1,
        "source_static_acceptance": SOURCE_STATIC_ACCEPTANCE,
    }
    manager = dict(common, actor="manager", artifact_kind="qk_gbfp8_head64_b0_v28_external_manager_admission",
                   blocked=False, decision="ADMIT_B0_ONCE", event_sha256="2" * 64)
    manager_raw = compact_bytes(manager)
    operator = dict(common, actor="operator", affirmative=True,
                    artifact_kind="qk_gbfp8_head64_b0_v28_external_operator_authority",
                    decision="AUTHORIZE_B0_ONCE", manager_admission_file_sha256=sha256_bytes(manager_raw),
                    provenance_event_sha256="3" * 64)
    return manager, operator


def write_record(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compact_bytes(value))
    path.chmod(0o400)


def schema_and_mutation_cases() -> dict[str, Any]:
    package, package_raw, acceptance, acceptance_raw = synthetic_package()
    manager, operator = authority_records(package_raw, package, acceptance_raw, acceptance)
    manager_schema = Draft202012Validator(json.loads(wrapper.MANAGER_SCHEMA.read_text(encoding="ascii")))
    operator_schema = Draft202012Validator(json.loads(wrapper.OPERATOR_SCHEMA.read_text(encoding="ascii")))
    manager_schema.validate(manager)
    operator_schema.validate(operator)
    schema_negatives = 0
    for schema, baseline in ((manager_schema, manager), (operator_schema, operator)):
        for key in list(baseline):
            mutation = dict(baseline)
            del mutation[key]
            try:
                schema.validate(mutation)
            except ValidationError:
                schema_negatives += 1
            else:
                raise RegressionError(f"schema missing-key accepted: {key}")
        mutation = dict(baseline, unexpected=True)
        try:
            schema.validate(mutation)
        except ValidationError:
            schema_negatives += 1
        else:
            raise RegressionError("schema additional property accepted")
    binding_negatives = 0
    mutation_paths = [
        ("future_execution_action_id", "wrong"),
        ("execution_package_file_sha256", "0" * 64),
        ("execution_package_content_sha256", "0" * 64),
        ("execution_package_acceptance_file_sha256", "0" * 64),
        ("execution_package_acceptance_self_sha256", "0" * 64),
        ("source_static_acceptance", {}),
        ("execution_binding", {}),
        ("reservation_nonce_sha256", "0" * 64),
    ]
    for key, value in mutation_paths:
        altered = copy.deepcopy(operator)
        altered[key] = value
        try:
            validate_external_records(manager, compact_bytes(manager), altered, sha256_bytes(package_raw),
                                      package["package_content_sha256"], sha256_bytes(acceptance_raw),
                                      acceptance["acceptance_sha256"])
        except Exception:
            binding_negatives += 1
        else:
            raise RegressionError(f"binding mutation accepted: {key}")
    return {"binding_mutation_negative_count": binding_negatives, "schema_negative_count": schema_negatives}


def wrapper_cases() -> dict[str, Any]:
    package, package_raw, acceptance, acceptance_raw = synthetic_package()
    manager, operator = authority_records(package_raw, package, acceptance_raw, acceptance)
    rows = synthetic_rows()

    def preflight(_cwd: Path, _environment: dict[str, str], _argv: list[str]) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
        return package, package_raw, acceptance, acceptance_raw

    with tempfile.TemporaryDirectory(prefix="b0-v28-inert-") as temporary:
        root = Path(temporary)
        manager_path = root / "external/manager.json"
        operator_path = root / "external/operator.json"
        write_record(manager_path, manager)
        write_record(operator_path, operator)

        def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            envelope_sha256 = argv[-1]
            sidecar = worker.build_sidecar(rows, copy.deepcopy(rows), envelope_sha256)
            return completed(0, compact_bytes(sidecar))

        runtime = root / "runtime-success"
        outcome = wrapper.run_attempt(
            root=runtime, argv=list(wrapper.EXPECTED_ARGS), observed_cwd=ACTION_ROOT,
            observed_environment=EXACT_ENVIRONMENT, manager_path=manager_path, operator_path=operator_path,
            payload_reader=lambda: b"protected-data-free-synthetic-payload", adapter_runner=runner,
            static_preflight=preflight,
        )
        require(outcome.status == "SOURCE_ORACLE_MATCH", "synthetic wrapper success")
        paths = wrapper.runtime_paths(runtime)
        require(all(paths[name].is_file() for name in paths), "complete create-once lifecycle")
        before = sorted(path.relative_to(runtime).as_posix() for path in runtime.rglob("*") if path.is_file())
        duplicate = wrapper.run_attempt(root=runtime, static_preflight=preflight)
        after = sorted(path.relative_to(runtime).as_posix() for path in runtime.rglob("*") if path.is_file())
        require(duplicate.duplicate_rejected and before == after, "duplicate no-write rejection")
        terminal = json.loads(paths["terminal"].read_text(encoding="ascii"))
        require(terminal["counts"] == {
            "authority_consumption": 1, "credential_consumption": 1, "official_evaluator_invocations": 1,
            "official_payload_opens": 1, "official_target_process_starts": 1,
        }, "exactly-once budgets")

        failure_runtime = root / "runtime-failure"
        failure = wrapper.run_attempt(
            root=failure_runtime, argv=["--invalid"], observed_cwd=ACTION_ROOT,
            observed_environment=EXACT_ENVIRONMENT,
            static_preflight=lambda *args: (_ for _ in ()).throw(wrapper.ExecutionError("synthetic argv", "PRESTART", "ARGV")),
        )
        require(failure.status == "FAILED_TERMINAL", "prestart terminal")
        failure_paths = wrapper.runtime_paths(failure_runtime)
        require(failure_paths["terminal"].is_file(), "prestart terminal publication")
        require(not any(failure_paths[name].exists() for name in ("authority_envelope", "authority_consumption", "credential_consumption", "ledger", "sidecar")), "prestart zero live authority")
        return {"duplicate_rejected": True, "prestart_terminal": True, "synthetic_authorized_dry_path": True}


def run() -> dict[str, Any]:
    report = {
        "adapter_boundary_cases": adapter_cases(),
        "artifact_kind": "qk_gbfp8_head64_b0_source_oracle_identity_control_execution_v28_synthetic_regression",
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "schema_and_mutation": schema_and_mutation_cases(),
        "schema_version": 1,
        "sidecar": sidecar_cases(),
        "status": "PASS_V28_INERT_PROTECTED_DATA_FREE_REGRESSION",
        "wrapper": wrapper_cases(),
    }
    require(AUDIT == {"official_payload_open_count": 0, "process_start_count": 0}, "inert audit counts")
    return sealed(report, "report_sha256")


def main() -> int:
    sys.addaudithook(audit_hook)
    try:
        report = run()
    except Exception as error:
        sys.stderr.write(f"B0_V28_SYNTHETIC_REGRESSION_FAIL:{type(error).__name__}:{error}\n")
        return 1
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
