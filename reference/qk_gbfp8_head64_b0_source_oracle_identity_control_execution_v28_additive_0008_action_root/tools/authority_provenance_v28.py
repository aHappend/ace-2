#!/home/argustest/miniconda3/bin/python3.13
"""Fail-closed event-log provenance checks for V28 authority and review.

The external capsules are data inputs, not trust anchors.  Their provenance
witnesses must resolve to exact records in the append-only Argus project event
log.  Tests may supply a protected-data-free synthetic log explicitly; the
production wrapper always uses the fixed authoritative log below.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path("/home/argustest/ace-2")
EVENT_LOG = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/events.jsonl")
ACTION_ID = "ace2:qk-gbfp8-base-v28:b0-source-oracle-identity-control-execute-once:38571d38:additive-0008"
AUTHORITATIVE_EVENT_SOURCE = "ARGUS_PROJECT_EVENTS_JSONL"


class ProvenanceError(RuntimeError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.stage = "AUTHORITY"
        self.failure_class = "PROVENANCE"


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ProvenanceError(detail)


def compact_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_identity_hash(value: dict[str, Any]) -> None:
    observed = value.get("provenance_identity_sha256")
    candidate = dict(value)
    candidate.pop("provenance_identity_sha256", None)
    require(observed == sha256_bytes(compact_bytes(candidate)), "provenance identity self hash")


def _read_snapshot(witness: dict[str, Any], event_log: Path) -> tuple[list[tuple[dict[str, Any], bytes]], bytes]:
    require(witness.get("authoritative_event_source") == AUTHORITATIVE_EVENT_SOURCE, "event source")
    require(event_log.is_absolute(), "event log absolute path")
    path_info = os.lstat(event_log)
    require(stat.S_ISREG(path_info.st_mode) and not stat.S_ISLNK(path_info.st_mode), "event log regular file")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(event_log, flags)
    try:
        before = os.fstat(descriptor)
        count = witness.get("event_log_snapshot_byte_count")
        require(type(count) is int and count > 0 and count <= before.st_size, "event log snapshot size")
        chunks: list[bytes] = []
        remaining = count
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), "event log snapshot truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "event log changed while read",
    )
    require((path_info.st_dev, path_info.st_ino) == (before.st_dev, before.st_ino), "event log replaced")
    require(witness.get("event_log_device") == before.st_dev, "event log device")
    require(witness.get("event_log_inode") == before.st_ino, "event log inode")
    raw = b"".join(chunks)
    require(raw.endswith(b"\n"), "event log snapshot final newline")
    require(witness.get("event_log_snapshot_sha256") == sha256_bytes(raw), "event log snapshot hash")
    parsed: list[tuple[dict[str, Any], bytes]] = []
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        require(line not in {b"", b"\n"}, f"blank event line {number}")
        try:
            value = json.loads(line.decode("utf-8", "strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProvenanceError(f"invalid event line {number}") from error
        require(type(value) is dict, f"non-object event line {number}")
        parsed.append((value, line))
    return parsed, raw


def _event(
    events: list[tuple[dict[str, Any], bytes]],
    line_number: Any,
    line_sha256: Any,
) -> tuple[dict[str, Any], bytes]:
    require(type(line_number) is int and 1 <= line_number <= len(events), "event line number")
    event, raw = events[line_number - 1]
    require(type(line_sha256) is str and line_sha256 == sha256_bytes(raw), "event line hash")
    return event, raw


def _contains_tokens(text: Any, tokens: Iterable[str], detail: str) -> None:
    require(type(text) is str, f"{detail} text")
    for token in tokens:
        require(token in text, f"{detail} missing binding token")


def validate_external_capsule_provenance(
    manager: dict[str, Any],
    operator: dict[str, Any],
    required_binding_tokens: list[str],
    *,
    event_log: Path = EVENT_LOG,
) -> dict[str, str]:
    """Bind both capsules to one explicit user intent and Manager decision."""

    manager_witness = manager.get("issuer_provenance")
    operator_witness = operator.get("issuer_provenance")
    require(type(manager_witness) is dict and type(operator_witness) is dict, "capsule issuer provenance")
    verify_identity_hash(manager_witness)
    verify_identity_hash(operator_witness)

    operator_events, _ = _read_snapshot(operator_witness, event_log)
    operator_event, operator_raw = _event(
        operator_events,
        operator_witness.get("event_line_number"),
        operator_witness.get("event_line_sha256"),
    )
    require(operator_event.get("type") == "life.manager.intent.started", "operator event type")
    require(operator_event.get("source") == "user", "operator event source")
    require(operator_event.get("agent_layer") == "manager", "operator event agent layer")
    require(operator_event.get("intent_id") == operator_witness.get("intent_id"), "operator intent id")
    operator_text = operator_event.get("objective")
    require(
        operator_witness.get("authorization_text_sha256")
        == sha256_bytes((operator_text + "\n").encode("utf-8")) if type(operator_text) is str else False,
        "operator authorization text hash",
    )
    _contains_tokens(operator_text, ["AUTHORIZE_B0_ONCE", *required_binding_tokens], "operator event")

    manager_events, _ = _read_snapshot(manager_witness, event_log)
    manager_event, manager_raw = _event(
        manager_events,
        manager_witness.get("event_line_number"),
        manager_witness.get("event_line_sha256"),
    )
    require(manager_event.get("type") == "life.manager.intent.completed", "manager event type")
    require(manager_event.get("source") == "user", "manager event source")
    require(manager_event.get("agent_layer") == "manager", "manager event agent layer")
    require(manager_event.get("intent_id") == manager_witness.get("intent_id"), "manager intent id")
    require(manager_witness.get("intent_id") == operator_witness.get("intent_id"), "shared authority intent")
    manager_text = manager_event.get("execution_task")
    require(
        manager_witness.get("authorization_text_sha256")
        == sha256_bytes((manager_text + "\n").encode("utf-8")) if type(manager_text) is str else False,
        "manager authorization text hash",
    )
    _contains_tokens(manager_text, ["ADMIT_B0_ONCE", *required_binding_tokens], "manager event")
    require(
        operator_witness.get("event_line_number") < manager_witness.get("event_line_number"),
        "operator authority must precede Manager admission",
    )
    return {
        "manager_event_sha256": sha256_bytes(manager_raw),
        "operator_event_sha256": sha256_bytes(operator_raw),
    }


def validate_reviewer_provenance(
    witness: dict[str, Any],
    required_binding_tokens: list[str],
    *,
    event_log: Path = EVENT_LOG,
) -> dict[str, str]:
    """Verify that acceptance was created by an active Reviewer invocation."""

    require(type(witness) is dict, "reviewer provenance object")
    verify_identity_hash(witness)
    events, _ = _read_snapshot(witness, event_log)
    provider, provider_raw = _event(
        events,
        witness.get("provider_start_event_line_number"),
        witness.get("provider_start_event_sha256"),
    )
    agent, agent_raw = _event(
        events,
        witness.get("agent_start_event_line_number"),
        witness.get("agent_start_event_sha256"),
    )
    command, command_raw = _event(
        events,
        witness.get("reviewer_command_event_line_number"),
        witness.get("reviewer_command_event_sha256"),
    )
    call_id = witness.get("active_call_id")
    require(provider.get("type") == "provider.request.started", "reviewer provider event type")
    require(agent.get("type") == "agent.io.start", "reviewer agent event type")
    require(provider.get("call_id") == agent.get("call_id") == call_id, "reviewer call binding")
    require(provider.get("run_label") == agent.get("run_label") == "reviewer", "reviewer run label")
    require(agent.get("working_dir") == str(PROJECT_ROOT), "reviewer working directory")
    _contains_tokens(agent.get("prompt"), [ACTION_ID, *required_binding_tokens], "reviewer prompt")
    require(command.get("type") == "engineer.progress", "reviewer command event type")
    require(command.get("kind") == "command_execution", "reviewer command kind")
    require(command.get("actor") == "reviewer" and command.get("agent_layer") == "reviewer", "reviewer role")
    require(command.get("status") == "completed" and command.get("exit_code") == 0, "reviewer command completion")
    require(
        witness.get("provider_start_event_line_number")
        < witness.get("agent_start_event_line_number")
        < witness.get("reviewer_command_event_line_number"),
        "reviewer event order",
    )
    return {
        "agent_start_event_sha256": sha256_bytes(agent_raw),
        "provider_start_event_sha256": sha256_bytes(provider_raw),
        "reviewer_command_event_sha256": sha256_bytes(command_raw),
    }
