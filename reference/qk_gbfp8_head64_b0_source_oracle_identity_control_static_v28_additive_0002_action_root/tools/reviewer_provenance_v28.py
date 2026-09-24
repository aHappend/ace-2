#!/usr/bin/env python3
"""Authoritative Reviewer invocation attestation for additive static V28.

This module is deliberately write-free.  The future create-once acceptance
tool must call ``attest_current_reviewer`` before creating any audit or
acceptance bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
SESSION_ROOT = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b")
EVENT_LOG = SESSION_ROOT / "events.jsonl"
MISSION_ID = "b0identitycontrolpackagev28"
ACTION_ID = "ace2:qk-gbfp8-base-v28:b0-source-oracle-identity-control:38571d38:additive-0002"
REQUIRED_RUN_LABEL = "reviewer"
REQUIRED_ACTOR = "reviewer"
REQUIRED_AGENT_LAYER = "reviewer"
CONCURRENT_ACTIVE_WINDOW_SECONDS = 3600.0
CALL_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class ProvenanceError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise ProvenanceError(code, detail)


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


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parse_events(raw: bytes) -> list[tuple[int, dict[str, Any], bytes]]:
    require(raw.endswith(b"\n"), "ROLE_PROVENANCE_TRUNCATED_LOG", "event log lacks final newline")
    parsed: list[tuple[int, dict[str, Any], bytes]] = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        require(line not in {b"", b"\n"}, "ROLE_PROVENANCE_BLANK_EVENT", f"line {line_number}")
        try:
            value = json.loads(line.decode("utf-8", "strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProvenanceError("ROLE_PROVENANCE_INVALID_EVENT", f"line {line_number}: {error}") from error
        require(type(value) is dict, "ROLE_PROVENANCE_NON_OBJECT_EVENT", f"line {line_number}")
        parsed.append((line_number, value, line))
    return parsed


def derive_reviewer_provenance(raw: bytes, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Derive one active Reviewer identity from an authoritative event snapshot."""

    events = _parse_events(raw)
    provider_starts: dict[str, tuple[int, dict[str, Any], bytes]] = {}
    agent_starts: dict[str, tuple[int, dict[str, Any], bytes]] = {}
    completed: set[str] = set()

    for line_number, event, line in events:
        event_type = event.get("type")
        call_id = event.get("call_id")
        if event_type == "provider.request.started" and type(call_id) is str:
            require(call_id not in provider_starts, "ROLE_PROVENANCE_DUPLICATE_PROVIDER_START", call_id)
            provider_starts[call_id] = (line_number, event, line)
        elif event_type == "agent.io.start" and type(call_id) is str:
            require(call_id not in agent_starts, "ROLE_PROVENANCE_DUPLICATE_AGENT_START", call_id)
            agent_starts[call_id] = (line_number, event, line)
        elif event_type in {"provider.request.completed", "agent.io.complete"} and type(call_id) is str:
            completed.add(call_id)

    project_root_text = str(project_root)
    project_starts = [
        item for item in agent_starts.values() if item[1].get("working_dir") == project_root_text
    ]
    require(project_starts, "ROLE_PROVENANCE_PROJECT_START_ABSENT", project_root_text)
    agent_line_number, agent_start, agent_line = max(project_starts, key=lambda item: item[0])
    call_id = agent_start.get("call_id")
    require(type(call_id) is str and CALL_ID_PATTERN.fullmatch(call_id) is not None, "ROLE_PROVENANCE_CALL_ID", str(call_id))
    require(
        call_id not in completed,
        "ROLE_PROVENANCE_FRONTIER_CALL_COMPLETED",
        "latest project invocation is not active",
    )
    frontier_ts = agent_start.get("ts")
    require(type(frontier_ts) in {int, float}, "ROLE_PROVENANCE_START_TIMESTAMP", call_id)
    concurrent = [
        item
        for item in project_starts
        if item[1].get("call_id") != call_id
        and item[1].get("call_id") not in completed
        and type(item[1].get("ts")) in {int, float}
        and item[1]["ts"] >= frontier_ts - CONCURRENT_ACTIVE_WINDOW_SECONDS
    ]
    require(
        not concurrent,
        "ROLE_PROVENANCE_CONCURRENT_ACTIVE_CALLS",
        f"observed {len(concurrent)} other recent active project call(s)",
    )
    require(call_id in provider_starts, "ROLE_PROVENANCE_PROVIDER_START_ABSENT", call_id)
    provider_line_number, provider_start, provider_line = provider_starts[call_id]

    require(
        agent_start.get("run_label") == REQUIRED_RUN_LABEL,
        "ROLE_PROVENANCE_ACTIVE_RUN_LABEL",
        f"observed {agent_start.get('run_label')!r}",
    )
    require(
        provider_start.get("run_label") == REQUIRED_RUN_LABEL,
        "ROLE_PROVENANCE_PROVIDER_RUN_LABEL",
        f"observed {provider_start.get('run_label')!r}",
    )
    require(agent_start.get("io_kind") == "start", "ROLE_PROVENANCE_AGENT_EVENT_KIND", str(agent_start.get("io_kind")))
    require(agent_start.get("working_dir") == project_root_text, "ROLE_PROVENANCE_WORKING_DIR", str(agent_start.get("working_dir")))

    prompt = agent_start.get("prompt")
    require(type(prompt) is str, "ROLE_PROVENANCE_PROMPT_ABSENT", call_id)
    require(MISSION_ID in prompt, "ROLE_PROVENANCE_MISSION_BINDING", MISSION_ID)
    require(ACTION_ID in prompt, "ROLE_PROVENANCE_ACTION_BINDING", ACTION_ID)

    start_ts = frontier_ts
    command_events = [
        (line_number, event, line)
        for line_number, event, line in events
        if event.get("type") == "engineer.progress"
        and event.get("kind") == "command_execution"
        and type(event.get("ts")) in {int, float}
        and event["ts"] >= start_ts
    ]
    require(command_events, "ROLE_PROVENANCE_REVIEWER_COMMAND_ABSENT", call_id)
    command_line_number, command_event, command_line = command_events[-1]
    require(
        command_event.get("actor") == REQUIRED_ACTOR,
        "ROLE_PROVENANCE_COMMAND_ACTOR",
        f"observed {command_event.get('actor')!r}",
    )
    require(
        command_event.get("agent_layer") == REQUIRED_AGENT_LAYER,
        "ROLE_PROVENANCE_COMMAND_AGENT_LAYER",
        f"observed {command_event.get('agent_layer')!r}",
    )
    require(command_event.get("status") == "completed", "ROLE_PROVENANCE_COMMAND_STATUS", str(command_event.get("status")))
    require(command_event.get("exit_code") == 0, "ROLE_PROVENANCE_COMMAND_EXIT", str(command_event.get("exit_code")))

    identity: dict[str, Any] = {
        "action_id": ACTION_ID,
        "active_call_id": call_id,
        "actor": REQUIRED_ACTOR,
        "agent_layer": REQUIRED_AGENT_LAYER,
        "agent_start_event_line_number": agent_line_number,
        "agent_start_event_sha256": sha256_bytes(agent_line),
        "authoritative_event_source": "ARGUS_PROJECT_EVENTS_JSONL",
        "mission_id": MISSION_ID,
        "provider_start_event_line_number": provider_line_number,
        "provider_start_event_sha256": sha256_bytes(provider_line),
        "reviewer_command_event_line_number": command_line_number,
        "reviewer_command_event_sha256": sha256_bytes(command_line),
        "run_label": REQUIRED_RUN_LABEL,
        "working_dir_sha256": sha256_bytes((project_root_text + "\n").encode("utf-8")),
    }
    identity["provenance_identity_sha256"] = sha256_bytes(compact_bytes(identity))
    return identity


def _read_fixed_event_log() -> tuple[bytes, os.stat_result]:
    configured = os.environ.get("ARGUS_SKILL_AGENT_IO_LOG")
    require(configured == str(EVENT_LOG), "ROLE_PROVENANCE_EVENT_LOG_ENV", repr(configured))
    path_info = os.lstat(EVENT_LOG)
    require(stat.S_ISREG(path_info.st_mode), "ROLE_PROVENANCE_EVENT_LOG_TYPE", "not a regular file")
    require(not stat.S_ISLNK(path_info.st_mode), "ROLE_PROVENANCE_EVENT_LOG_SYMLINK", str(EVENT_LOG))

    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(EVENT_LOG, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "ROLE_PROVENANCE_EVENT_LOG_DRIFT",
        "authoritative event log changed while read",
    )
    require(
        (path_info.st_dev, path_info.st_ino) == (before.st_dev, before.st_ino),
        "ROLE_PROVENANCE_EVENT_LOG_REPLACED",
        "path and opened file identities differ",
    )
    return b"".join(chunks), after


def attest_current_reviewer() -> dict[str, Any]:
    raw, info = _read_fixed_event_log()
    identity = derive_reviewer_provenance(raw)
    identity.update(
        {
            "event_log_device": info.st_dev,
            "event_log_inode": info.st_ino,
            "event_log_snapshot_byte_count": len(raw),
            "event_log_snapshot_sha256": sha256_bytes(raw),
        }
    )
    unhashed = dict(identity)
    unhashed.pop("provenance_identity_sha256", None)
    identity["provenance_identity_sha256"] = sha256_bytes(compact_bytes(unhashed))
    return identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", required=True)
    parser.parse_args()
    try:
        result = attest_current_reviewer()
    except ProvenanceError as error:
        sys.stdout.buffer.write(
            compact_bytes(
                {
                    "accepted_reviewer_provenance": False,
                    "error_code": error.code,
                    "error_detail": error.detail,
                }
            )
        )
        return 2
    sys.stdout.buffer.write(compact_bytes({"accepted_reviewer_provenance": True, **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
