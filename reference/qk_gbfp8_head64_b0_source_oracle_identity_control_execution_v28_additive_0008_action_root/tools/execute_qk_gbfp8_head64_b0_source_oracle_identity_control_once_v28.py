#!/home/argustest/miniconda3/bin/python3.13
"""Create-once V28 B0 launcher. Static acceptance alone never authorizes it."""

from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from qk_gbfp8_head64_b0_execution_common_v28 import (
    ACTION_ID, ACTION_ROOT, EXACT_ARGV, EXACT_ENVIRONMENT, EXECUTION_BINDING, EXTERNAL_MANAGER_ADMISSION,
    EXTERNAL_OPERATOR_AUTHORITY, INTERPRETER, INTERPRETER_SHA256, OFFICIAL_PAYLOAD, OUTPUT_PATHS, PACKAGE,
    RUNTIME_ROOT, SOURCE_STATIC_ACCEPTANCE, STATIC_ACCEPTANCE, STATIC_ACCEPTANCE_FILE_SHA256,
    STATIC_ACCEPTANCE_SELF_SHA256, STATIC_ENVELOPE_SCHEMA, STATIC_PACKAGE, STATIC_PACKAGE_CONTENT_SHA256,
    STATIC_PACKAGE_FILE_SHA256, STATIC_SIDECAR_SCHEMA, ExecutionError, build_runtime_envelope,
    canonical_object, compact_bytes, durable_create, require, require_regular_mode, safe_detail, sealed,
    sha256_bytes, sha256_file, validate_execution_acceptance, validate_external_records, verify_self_hash,
)
from qk_gbfp8_head64_b0_evaluator_adapter_v28 import (
    EvaluatorInvocationError, InvocationSpec, invoke_evaluator,
)


EXECUTION_ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
MANAGER_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_B0_V28_EXTERNAL_MANAGER_ADMISSION_SCHEMA.json"
OPERATOR_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_B0_V28_EXTERNAL_OPERATOR_AUTHORITY_SCHEMA.json"
TERMINAL_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_B0_V28_FIRST_TERMINAL_SCHEMA.json"
WORKER = ACTION_ROOT / "tools/qk_gbfp8_head64_b0_source_oracle_worker_v28.py"
EXPECTED_ARGS = EXACT_ARGV[2:]
PAYLOAD_SHA256 = "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175"
PAYLOAD_SIZE = 1305797


@dataclass
class Counts:
    authority_consumption: int = 0
    credential_consumption: int = 0
    evaluator_invocations: int = 0
    payload_opens: int = 0
    target_starts: int = 0


@dataclass
class AttemptResult:
    status: str
    duplicate_rejected: bool
    terminal_path: str | None


def runtime_paths(root: Path) -> dict[str, Path]:
    return {
        "authority_envelope": root / "authority/exactly-once-envelope.json",
        "authority_consumption": root / "authority/authority-consumption.json",
        "credential_consumption": root / "authority/credential-consumption.json",
        "ledger": root / "ledger/exactly-once-ledger.json",
        "sidecar": root / "primary/result/b0/source-oracle-aggregate-sidecar.json",
        "terminal": root / "primary/result/b0/first-terminal.json",
    }


def claim_runtime(root: Path) -> bool:
    try:
        os.mkdir(root, 0o700)
    except FileExistsError:
        return False
    return True


def terminal_record(status: str, stage: str, failure_class: str | None, error: BaseException | None,
                    counts: Counts, *, authority_envelope_sha256: str | None = None,
                    sidecar_sha256: str | None = None, evaluator_diagnostic: dict[str, Any] | None = None) -> dict[str, Any]:
    return sealed({
        "action_retired": True,
        "artifact_kind": "qk_gbfp8_head64_b0_source_oracle_identity_control_v28_first_terminal",
        "authority_envelope_sha256": authority_envelope_sha256,
        "counts": {
            "authority_consumption": counts.authority_consumption,
            "credential_consumption": counts.credential_consumption,
            "official_evaluator_invocations": counts.evaluator_invocations,
            "official_payload_opens": counts.payload_opens,
            "official_target_process_starts": counts.target_starts,
        },
        "detail": None if error is None else safe_detail(error, stage, failure_class or "NONE"),
        "evaluator_diagnostic": evaluator_diagnostic,
        "failure_class": failure_class,
        "first_record_immutable": True,
        "future_execution_action_id": ACTION_ID,
        "retry_replay_resume_repair_permitted": False,
        "schema_version": 1,
        "sidecar_sha256": sidecar_sha256,
        "stage": stage,
        "status": status,
    }, "terminal_sha256")


def publish_terminal(paths: dict[str, Path], record: dict[str, Any], validator: Callable[[dict[str, Any]], None] | None) -> str | None:
    verify_self_hash(record, "terminal_sha256")
    if validator is not None:
        validator(record)
    try:
        durable_create(paths["terminal"], record)
    except (OSError, FileExistsError):
        return None
    return str(paths["terminal"])


def load_schema(path: Path) -> Draft202012Validator:
    schema, _ = canonical_object(path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_static_and_package(observed_cwd: Path, observed_environment: dict[str, str], argv: list[str]) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    require(argv == EXPECTED_ARGS, "exact frozen argv", "PRESTART", "ARGV")
    require(observed_cwd == ACTION_ROOT, "exact frozen cwd", "PRESTART", "CWD")
    require(observed_environment == EXACT_ENVIRONMENT, "exact replacement environment", "PRESTART", "ENVIRONMENT")
    require(Path(sys.executable).resolve() == INTERPRETER, "interpreter path", "PRESTART", "INTERPRETER")
    require(sha256_file(INTERPRETER) == INTERPRETER_SHA256, "interpreter hash", "PRESTART", "INTERPRETER")
    require_regular_mode(STATIC_PACKAGE, 0o444, "static package")
    require_regular_mode(STATIC_ACCEPTANCE, 0o444, "static acceptance")
    require(sha256_file(STATIC_PACKAGE) == STATIC_PACKAGE_FILE_SHA256, "static package file hash", "STATIC_BINDING")
    require(sha256_file(STATIC_ACCEPTANCE) == STATIC_ACCEPTANCE_FILE_SHA256, "static acceptance file hash", "STATIC_BINDING")
    static_package, _ = canonical_object(STATIC_PACKAGE)
    static_acceptance, _ = canonical_object(STATIC_ACCEPTANCE)
    require(static_package.get("package_content_sha256") == STATIC_PACKAGE_CONTENT_SHA256, "static package content", "STATIC_BINDING")
    require(static_acceptance.get("acceptance_sha256") == STATIC_ACCEPTANCE_SELF_SHA256, "static acceptance self", "STATIC_BINDING")
    require(static_acceptance.get("accepted") is True and static_acceptance.get("decision") == "ACCEPT_STATIC_PACKAGE", "static acceptance decision", "STATIC_BINDING")
    require(static_acceptance.get("static_acceptance_grants_execution_authority") is False, "static authority boundary", "STATIC_BINDING")
    package, package_raw = canonical_object(PACKAGE)
    verify_self_hash(package, "package_content_sha256")
    require(package["action_identity"]["action_id"] == ACTION_ID, "execution package action", "PACKAGE_BINDING")
    require(package["execution_binding"] == EXECUTION_BINDING, "execution package binding", "PACKAGE_BINDING")
    require(package["claim_boundary"] == "STATIC_ONLY_NO_EXECUTION_AUTHORITY", "execution package claim", "PACKAGE_BINDING")
    require(EXECUTION_ACCEPTANCE.is_file(), "execution package Fresh-L2 acceptance", "PACKAGE_ACCEPTANCE")
    acceptance, acceptance_raw = canonical_object(EXECUTION_ACCEPTANCE)
    verify_self_hash(acceptance, "acceptance_sha256")
    require(acceptance.get("accepted") is True and acceptance.get("decision") == "ACCEPT_STATIC_PACKAGE", "execution acceptance decision", "PACKAGE_ACCEPTANCE")
    validate_execution_acceptance(acceptance, package_raw, package["package_content_sha256"])
    return package, package_raw, acceptance, acceptance_raw


def read_external(path: Path, schema: Draft202012Validator) -> tuple[dict[str, Any], bytes]:
    require(path.is_absolute() and RUNTIME_ROOT not in path.parents, "external authority namespace", "AUTHORITY", "NAMESPACE")
    info = os.lstat(path)
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o400,
            "external authority mode", "AUTHORITY", "NAMESPACE")
    value, raw = canonical_object(path)
    schema.validate(value)
    require(compact_bytes(value) == raw, "canonical external capsule", "AUTHORITY", "SCHEMA")
    return value, raw


def read_payload() -> bytes:
    descriptor = os.open(OFFICIAL_PAYLOAD, os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0))
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode) and info.st_size == PAYLOAD_SIZE, "payload metadata", "PAYLOAD_OPEN", "PAYLOAD")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    require(sha256_bytes(raw) == PAYLOAD_SHA256, "payload hash", "PAYLOAD_OPEN", "PAYLOAD")
    return raw


def run_attempt(
    *,
    root: Path = RUNTIME_ROOT,
    argv: list[str] | None = None,
    observed_cwd: Path | None = None,
    observed_environment: dict[str, str] | None = None,
    manager_path: Path = EXTERNAL_MANAGER_ADMISSION,
    operator_path: Path = EXTERNAL_OPERATOR_AUTHORITY,
    payload_reader: Callable[[], bytes] = read_payload,
    adapter_runner: Callable[..., Any] | None = None,
    static_preflight: Callable[[Path, dict[str, str], list[str]], tuple[dict[str, Any], bytes, dict[str, Any], bytes]] = validate_static_and_package,
    external_validator: Callable[..., None] = validate_external_records,
    record_creator: Callable[[Path, dict[str, Any] | bytes], str] = durable_create,
    lifecycle_hook: Callable[[str], None] = lambda _event: None,
) -> AttemptResult:
    paths = runtime_paths(root)
    if not claim_runtime(root):
        terminal = paths["terminal"]
        return AttemptResult("DUPLICATE_REJECTED", True, str(terminal) if terminal.is_file() else None)
    lifecycle_hook("runtime_namespace_claimed")
    counts = Counts()
    terminal_validator: Callable[[dict[str, Any]], None] | None = None
    envelope_sha256: str | None = None
    stage = "PRESTART"
    try:
        terminal_validator = load_schema(TERMINAL_SCHEMA).validate
        root_info = os.lstat(root)
        require(
            stat.S_ISDIR(root_info.st_mode)
            and not stat.S_ISLNK(root_info.st_mode)
            and stat.S_IMODE(root_info.st_mode) == 0o700,
            "runtime root mode",
            "PRESTART",
            "RUNTIME",
        )
        package, package_raw, acceptance, acceptance_raw = static_preflight(
            ACTION_ROOT if observed_cwd is None else observed_cwd,
            dict(os.environ) if observed_environment is None else observed_environment,
            list(sys.argv[1:] if argv is None else argv),
        )
        stage = "AUTHORITY"
        manager_schema = load_schema(MANAGER_SCHEMA)
        operator_schema = load_schema(OPERATOR_SCHEMA)
        manager, manager_raw = read_external(manager_path, manager_schema)
        operator, operator_raw = read_external(operator_path, operator_schema)
        external_validator(
            manager, manager_raw, operator, operator_raw, sha256_bytes(package_raw), package["package_content_sha256"],
            sha256_bytes(acceptance_raw), acceptance["acceptance_sha256"],
        )
        lifecycle_hook("external_capsules_verified")
        envelope = build_runtime_envelope(manager, operator)
        envelope_validator = load_schema(STATIC_ENVELOPE_SCHEMA)
        envelope_validator.validate(envelope)
        verify_self_hash(envelope, "authority_envelope_sha256")
        manager_file_sha256 = sha256_bytes(manager_raw)
        operator_file_sha256 = sha256_bytes(operator_raw)
        authority_consumption = sealed({
            "artifact_kind": "qk_gbfp8_head64_b0_v28_authority_consumption",
            "budget": 1, "consumed": 1, "future_execution_action_id": ACTION_ID,
            "manager_admission_file_sha256": manager_file_sha256,
            "operator_authority_file_sha256": operator_file_sha256,
            "retry_replay_resume_repair_permitted": False, "schema_version": 1,
        }, "authority_consumption_sha256")
        credential_consumption = sealed({
            "artifact_kind": "qk_gbfp8_head64_b0_v28_credential_consumption",
            "budget": 1, "consumed": 1, "future_execution_action_id": ACTION_ID,
            "operator_authority_file_sha256": operator_file_sha256,
            "reservation_nonce_sha256": manager["reservation_nonce_sha256"],
            "retry_replay_resume_repair_permitted": False, "schema_version": 1,
        }, "credential_consumption_sha256")
        record_creator(paths["authority_consumption"], authority_consumption)
        counts.authority_consumption = 1
        lifecycle_hook("manager_capsule_consumed")
        record_creator(paths["credential_consumption"], credential_consumption)
        counts.credential_consumption = 1
        lifecycle_hook("operator_capsule_consumed")
        record_creator(paths["authority_envelope"], envelope)
        envelope_sha256 = envelope["authority_envelope_sha256"]
        lifecycle_hook("internal_authority_envelope_created")
        ledger = sealed({
            "artifact_kind": "qk_gbfp8_head64_b0_v28_exactly_once_ledger",
            "authority_consumption_sha256": authority_consumption["authority_consumption_sha256"],
            "credential_consumption_sha256": credential_consumption["credential_consumption_sha256"],
            "future_execution_action_id": ACTION_ID,
            "official_evaluator_invocation_budget": 1, "official_payload_open_budget": 1,
            "official_target_process_start_budget": 1, "retry_replay_resume_repair_permitted": False,
            "schema_version": 1, "state": "CONSUMED_BEFORE_PAYLOAD",
        }, "ledger_sha256")
        record_creator(paths["ledger"], ledger)
        stage = "PAYLOAD_OPEN"
        lifecycle_hook("before_protected_payload_open")
        counts.payload_opens = 1
        payload = payload_reader()
        require(type(payload) is bytes, "payload ABI", stage, "ABI")
        stage = "EVALUATOR"
        lifecycle_hook("before_official_evaluator_invocation")
        counts.evaluator_invocations = 1
        sidecar_schema = load_schema(STATIC_SIDECAR_SCHEMA)
        spec = InvocationSpec(
            argv=(str(INTERPRETER), str(WORKER), "--mode", "production", "--static-package", str(STATIC_PACKAGE),
                  "--static-acceptance", str(STATIC_ACCEPTANCE), "--authority-envelope-sha256", envelope_sha256),
            cwd=str(ACTION_ROOT), environment=tuple(EXACT_ENVIRONMENT.items()), max_capture_bytes=1 << 20,
        )

        def validate_sidecar(record: dict[str, Any], raw: bytes) -> None:
            verify_self_hash(record, "sidecar_sha256")
            require(record["authority_envelope_sha256"] == envelope_sha256, "sidecar authority binding", "EVALUATOR", "RESULT_RECORD")
            require(record["outcome"] in {"SOURCE_ORACLE_MATCH", "SOURCE_ORACLE_MISMATCH"}, "exactly one outcome", "EVALUATOR", "RESULT_RECORD")
            require(compact_bytes(record) == raw, "canonical sidecar", "EVALUATOR", "RESULT_RECORD")

        def publish(raw: bytes) -> None:
            record_creator(paths["sidecar"], raw)

        keywords = {} if adapter_runner is None else {"runner": adapter_runner}
        adapter_result = invoke_evaluator(spec, payload, sidecar_schema.validate, validate_sidecar, publish, **keywords)
        counts.target_starts = int(adapter_result.diagnostic["process"]["started"])
        outcome = adapter_result.record["outcome"]
        terminal = terminal_record(
            outcome, "COMPLETE", None, None, counts, authority_envelope_sha256=envelope_sha256,
            sidecar_sha256=adapter_result.record["sidecar_sha256"], evaluator_diagnostic=adapter_result.diagnostic,
        )
        terminal_path = publish_terminal(paths, terminal, terminal_validator)
        return AttemptResult(outcome if terminal_path else "TERMINAL_PUBLICATION_FAILED", False, terminal_path)
    except BaseException as error:
        diagnostic = error.diagnostic if isinstance(error, EvaluatorInvocationError) else None
        if diagnostic is not None:
            counts.target_starts = int(diagnostic["process"]["started"])
            failure_class = diagnostic["failure_class"]
        else:
            failure_class = getattr(error, "failure_class", type(error).__name__)
        failure_stage = getattr(error, "stage", stage)
        terminal = terminal_record(
            "FAILED_TERMINAL", failure_stage, failure_class, error, counts,
            authority_envelope_sha256=envelope_sha256, evaluator_diagnostic=diagnostic,
        )
        terminal_path = publish_terminal(paths, terminal, terminal_validator)
        return AttemptResult("FAILED_TERMINAL" if terminal_path else "TERMINAL_PUBLICATION_FAILED", False, terminal_path)


def main() -> int:
    outcome = run_attempt()
    return 0 if outcome.status in {"SOURCE_ORACLE_MATCH", "SOURCE_ORACLE_MISMATCH"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
