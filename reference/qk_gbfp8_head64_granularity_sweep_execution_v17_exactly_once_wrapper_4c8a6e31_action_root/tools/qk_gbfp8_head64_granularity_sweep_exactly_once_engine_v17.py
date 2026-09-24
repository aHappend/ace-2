
#!/usr/bin/env python3
"""Atomic exactly-once lifecycle engine; evaluator behavior is injected."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, NamedTuple


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class LifecycleError(RuntimeError):
    pass


class DuplicateInvocationError(LifecycleError):
    pass


class RuntimePaths(NamedTuple):
    authority: Path
    credential: Path
    ledger: Path
    result: Path
    first_terminal: Path
    fallback_terminal: Path


class Counts:
    def __init__(self) -> None:
        self.invocation_count = 0
        self.payload_open_count = 0
        self.official_target_process_starts = 0


class Outcome(NamedTuple):
    counts: Counts
    duplicate_rejected: bool
    result_sha256: str | None
    status: str
    terminal_path: str | None
    terminal_publication_failed: bool


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = sha256_bytes(compact_bytes(result))
    return result


def verify_self_hash(value: dict[str, Any], field: str) -> None:
    observed = value.get(field)
    candidate = dict(value)
    candidate.pop(field, None)
    if observed != sha256_bytes(compact_bytes(candidate)):
        raise LifecycleError(f"self hash: {field}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_create(path: Path, raw: bytes, *, fault: bool = False) -> None:
    if fault:
        raise OSError("injected create failure")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def durable_unlink(path: Path) -> None:
    os.unlink(path)
    _fsync_directory(path.parent)


def _diagnostic(error: BaseException, stage: str) -> dict[str, Any]:
    raw = f"{type(error).__name__}:{stage}".encode("ascii", "replace")
    return {"detail_sha256": sha256_bytes(raw), "exception_type": type(error).__name__, "failure_stage": stage}


def _authority(bindings: dict[str, Any]) -> dict[str, Any]:
    return sealed({
        "action_id": bindings["action_id"],
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_authority",
        "core_acceptance_file_sha256": bindings["core_acceptance_file_sha256"],
        "core_manifest_file_sha256": bindings["core_manifest_file_sha256"],
        "execution_package_file_sha256": bindings["execution_package_file_sha256"],
        "fresh_l2_acceptance_sha256": bindings["fresh_l2_acceptance_sha256"],
        "interpreter_sha256": bindings["interpreter_sha256"],
        "invocation_sha256": bindings["invocation_sha256"],
        "launcher_invocation_sha256": bindings["launcher_invocation_sha256"],
        "official_identities": bindings["official_identities"],
        "single_use": True,
        "state": "READY_UNCONSUMED",
        "transport_attestation": bindings["transport_attestation"],
    }, "authority_sha256")


def _credential(action_id: str, authority: dict[str, Any]) -> dict[str, Any]:
    nonce = sha256_bytes(("credential\0" + authority["authority_sha256"]).encode("ascii"))
    return sealed({
        "action_id": action_id,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_credential",
        "authority_sha256": authority["authority_sha256"],
        "credential_nonce_sha256": nonce,
        "single_use": True,
        "state": "READY_UNCONSUMED",
    }, "credential_sha256")


def _ledger(bindings: dict[str, Any], authority: dict[str, Any], credential: dict[str, Any]) -> dict[str, Any]:
    return sealed({
        "action_id": bindings["action_id"],
        "action_retired": True,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_consumed_ledger",
        "authority_sha256": authority["authority_sha256"],
        "credential_consumption": {"credential_sha256": credential["credential_sha256"], "state": "CONSUMED_BEFORE_PAYLOAD"},
        "execution_package_file_sha256": bindings["execution_package_file_sha256"],
        "invocation_cardinality": {"invocation_count_performed_at_publish": 0, "invocation_count_permitted": 1},
        "payload_cardinality": {"payload_open_count_performed_at_publish": 0, "payload_open_count_permitted": 1},
        "post_consumption_ambiguity_default": "CONSUMED_ORPHAN",
        "replay_permitted": False,
        "retry_replay_resume_repair_replacement_permitted": False,
        "transport_attestation_sha256": authority["transport_attestation"]["transport_attestation_sha256"],
    }, "consumed_ledger_sha256")


def _terminal(
    action_id: str,
    *,
    status: str,
    reason_code: str,
    authority_sha256: str | None,
    ledger_sha256: str | None,
    credential_consumed: bool,
    counts: Counts,
    result_sha256: str | None,
    failure_stage: str | None,
    evaluator_diagnostic: dict[str, Any] | None,
    wrapper_diagnostic: dict[str, Any] | None,
) -> dict[str, Any]:
    detail = evaluator_diagnostic or wrapper_diagnostic
    return sealed({
        "action_id": action_id,
        "action_retired": True,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_first_terminal",
        "authority_sha256": authority_sha256,
        "consumed_ledger_sha256": ledger_sha256,
        "credential_consumed": credential_consumed,
        "evaluator_diagnostic": evaluator_diagnostic,
        "failure_detail_sha256": EMPTY_SHA256 if detail is None else sha256_bytes(compact_bytes(detail)),
        "failure_stage": failure_stage,
        "first_record_immutable": True,
        "invocation_count_performed": counts.invocation_count,
        "official_target_process_starts": counts.official_target_process_starts,
        "orphaned_after_consumption": status == "CONSUMED_ORPHAN",
        "payload_open_count": counts.payload_open_count,
        "reason_code": reason_code,
        "result_file_sha256": result_sha256,
        "retry_replay_resume_repair_replacement_permitted": False,
        "status": status,
        "wrapper_diagnostic": wrapper_diagnostic,
    }, "first_terminal_sha256")


def _publish_preconsumption_terminal(paths: RuntimePaths, validate: Callable[[dict[str, Any]], None], record: dict[str, Any], faults: frozenset[str]) -> tuple[str | None, bool]:
    validate(record)
    try:
        durable_create(paths.first_terminal, compact_bytes(record), fault="TERMINAL_PRIMARY" in faults)
        return str(paths.first_terminal), False
    except FileExistsError:
        return None, True
    except OSError:
        try:
            durable_create(paths.fallback_terminal, compact_bytes(record), fault="TERMINAL_FALLBACK" in faults)
            return str(paths.fallback_terminal), False
        except (OSError, FileExistsError):
            return None, True


def _publish_postconsumption_terminal(
    paths: RuntimePaths,
    validate: Callable[[dict[str, Any]], None],
    record: dict[str, Any],
    orphan_record: dict[str, Any],
    faults: frozenset[str],
) -> tuple[str | None, bool, str]:
    validate(record)
    try:
        durable_create(paths.first_terminal, compact_bytes(record), fault="TERMINAL_PRIMARY" in faults)
        return str(paths.first_terminal), False, record["status"]
    except FileExistsError:
        return None, True, "DUPLICATE_REJECTED"
    except OSError:
        validate(orphan_record)
        try:
            durable_create(paths.fallback_terminal, compact_bytes(orphan_record), fault="TERMINAL_FALLBACK" in faults)
            return str(paths.fallback_terminal), False, "CONSUMED_ORPHAN"
        except (OSError, FileExistsError):
            return None, True, "CONSUMED_ORPHAN_UNPUBLISHED"


def _lifecycle_exists(paths: RuntimePaths) -> bool:
    return any(os.path.lexists(path) for path in paths)


def run_execution(
    paths: RuntimePaths,
    bindings: dict[str, Any],
    validators: dict[str, Callable[[dict[str, Any]], None]],
    preflight: Callable[[], Any],
    payload_reader: Callable[[], bytes],
    invoke_once: Callable[[bytes, str, str, Callable[[bytes], None]], Any],
    *,
    faults: frozenset[str] = frozenset(),
) -> Outcome:
    counts = Counts()
    if _lifecycle_exists(paths):
        return Outcome(counts, True, None, "DUPLICATE_REJECTED", None, False)
    try:
        preflight()
    except BaseException as error:
        record = _terminal(bindings["action_id"], status="PREFLIGHT_FAILED_TERMINAL", reason_code="READ_ONLY_PREFLIGHT_FAILED",
            authority_sha256=None, ledger_sha256=None, credential_consumed=False, counts=counts, result_sha256=None,
            failure_stage="PREFLIGHT", evaluator_diagnostic=None, wrapper_diagnostic=_diagnostic(error, "PREFLIGHT"))
        terminal_path, failed = _publish_preconsumption_terminal(paths, validators["terminal"], record, faults)
        return Outcome(counts, False, None, record["status"], terminal_path, failed)

    authority = _authority(bindings)
    credential = _credential(bindings["action_id"], authority)
    ledger = _ledger(bindings, authority, credential)
    validators["authority"](authority)
    validators["credential"](credential)
    validators["ledger"](ledger)
    stage = "AUTHORITY_MATERIALIZATION"
    try:
        durable_create(paths.authority, compact_bytes(authority), fault="AUTHORITY_WRITE" in faults)
        stage = "CREDENTIAL_MATERIALIZATION"
        durable_create(paths.credential, compact_bytes(credential), fault="CREDENTIAL_WRITE" in faults)
        stage = "LEDGER_MATERIALIZATION"
        durable_create(paths.ledger, compact_bytes(ledger), fault="LEDGER_WRITE" in faults)
    except BaseException as error:
        if os.path.lexists(paths.credential) and not os.path.lexists(paths.ledger):
            try:
                durable_unlink(paths.credential)
            except OSError:
                pass
        record = _terminal(bindings["action_id"], status="FAILED_TERMINAL", reason_code="LIVE_MATERIALIZATION_FAILED",
            authority_sha256=authority["authority_sha256"] if os.path.lexists(paths.authority) else None,
            ledger_sha256=None, credential_consumed=False, counts=counts, result_sha256=None,
            failure_stage=stage, evaluator_diagnostic=None, wrapper_diagnostic=_diagnostic(error, stage))
        terminal_path, failed = _publish_preconsumption_terminal(paths, validators["terminal"], record, faults)
        return Outcome(counts, False, None, record["status"], terminal_path, failed)

    result_sha256: str | None = None
    try:
        stage = "CREDENTIAL_CONSUMPTION"
        if "CREDENTIAL_UNLINK" in faults:
            raise OSError("injected credential unlink failure")
        durable_unlink(paths.credential)
        stage = "PAYLOAD_OPEN"
        counts.payload_open_count = 1
        payload = payload_reader()
        if type(payload) is not bytes:
            raise TypeError("payload ABI")
        stage = "EVALUATOR_CALL"
        counts.invocation_count = 1

        def publish_result(raw: bytes) -> None:
            nonlocal result_sha256, stage
            stage = "RESULT_PUBLICATION"
            durable_create(paths.result, raw, fault="RESULT_WRITE" in faults)
            result_sha256 = sha256_bytes(raw)

        adapter_result = invoke_once(payload, authority["authority_sha256"], ledger["consumed_ledger_sha256"], publish_result)
        diagnostic = adapter_result.diagnostic
        counts.official_target_process_starts = 1 if diagnostic["process"]["started"] else 0
        record_value = adapter_result.record
        terminal_value = record_value["terminal"]
        record = _terminal(bindings["action_id"], status=terminal_value["status"], reason_code=terminal_value["reason_code"],
            authority_sha256=authority["authority_sha256"], ledger_sha256=ledger["consumed_ledger_sha256"],
            credential_consumed=True, counts=counts, result_sha256=result_sha256, failure_stage=None,
            evaluator_diagnostic=diagnostic, wrapper_diagnostic=None)
        orphan = _terminal(bindings["action_id"], status="CONSUMED_ORPHAN", reason_code="POST_CONSUMPTION_AMBIGUITY",
            authority_sha256=authority["authority_sha256"], ledger_sha256=ledger["consumed_ledger_sha256"],
            credential_consumed=True, counts=counts, result_sha256=result_sha256, failure_stage="TERMINAL_PUBLICATION",
            evaluator_diagnostic=diagnostic, wrapper_diagnostic=_diagnostic(OSError("terminal publication"), "TERMINAL_PUBLICATION"))
        terminal_path, failed, status_value = _publish_postconsumption_terminal(paths, validators["terminal"], record, orphan, faults)
        return Outcome(counts, False, result_sha256, status_value, terminal_path, failed)
    except BaseException as error:
        diagnostic = getattr(error, "diagnostic", None)
        if type(diagnostic) is dict:
            counts.official_target_process_starts = 1 if diagnostic.get("process", {}).get("started") else 0
            failure_stage = "RESULT_PUBLICATION" if diagnostic.get("failure_class") == "PUBLICATION" else "EVALUATOR_CALL"
            wrapper = None
        else:
            failure_stage = stage
            wrapper = _diagnostic(error, failure_stage)
        orphan = _terminal(bindings["action_id"], status="CONSUMED_ORPHAN", reason_code="POST_CONSUMPTION_AMBIGUITY",
            authority_sha256=authority["authority_sha256"], ledger_sha256=ledger["consumed_ledger_sha256"],
            credential_consumed=True, counts=counts, result_sha256=result_sha256, failure_stage=failure_stage,
            evaluator_diagnostic=diagnostic if type(diagnostic) is dict else None, wrapper_diagnostic=wrapper)
        terminal_path, failed = _publish_preconsumption_terminal(paths, validators["terminal"], orphan, faults)
        return Outcome(counts, False, result_sha256, "CONSUMED_ORPHAN", terminal_path, failed)
