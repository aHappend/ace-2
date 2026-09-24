#!/usr/bin/env python3
"""Create and seal layer-16 authority package 0016 exclusively."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path("/home/argustest/ace-2")
SOURCE = ROOT / "reports/ace2-layer16-runtime-pass-authority-0015"
SOURCE_REVIEW = ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0015"
TARGET = ROOT / "reports/ace2-layer16-runtime-pass-authority-0016"
SOURCE_ROOT = "a513b7c7003169c39abe39d0c56ec861f3ba59407fcf1a653e6a79ae5e5441fc"
ENGINEER_MISSION_ID = "860efb205a06"
DIRECTIVE = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "active_manager_directive.json"
)
AGENT_IO = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/agent_io.jsonl"
)
SOURCE_MEMBERS = {
    "execute_layer16.py",
    "hostile_controls.py",
    "review_package.py",
    "seal_package.py",
    "validate_nonexecuting.py",
    "validate_review.py",
}
SOURCE_REVIEW_HASHES = {
    "review-log.json": (
        "0cb1a79ea9c77916b08a58ea0e92b94b1d5bb33fc5e6d65094e066cb65c50eff"
    ),
    "reviewer-adjudication.json": (
        "87cd89561c141310f9b87f1626cdb3448d89a8f31fe565726f670ccbb1e72a17"
    ),
    "reviewer-receipt.json": (
        "2ecbd0d2440ef22c16d6008d50d220f291179af70e7497fa489965d9bd37d216"
    ),
}


class PreparationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, value: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(value):
            offset += os.write(descriptor, value[offset:])
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def replace_once(text: str, old: str, new: str) -> str:
    require(text.count(old) == 1, f"source fragment count changed: {old[:80]!r}")
    return text.replace(old, new)


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    require(text.count(start) == 1, f"source start marker changed: {start!r}")
    require(text.count(end) == 1, f"source end marker changed: {end!r}")
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement + text[finish:]


def validate_source_review() -> None:
    require(
        SOURCE_REVIEW.is_dir()
        and not SOURCE_REVIEW.is_symlink()
        and stat.S_IMODE(SOURCE_REVIEW.stat().st_mode) == 0o555,
        "package0015 review namespace changed",
    )
    require(
        {item.name for item in SOURCE_REVIEW.iterdir()} == set(SOURCE_REVIEW_HASHES),
        "package0015 genuine PASS artifact set changed",
    )
    for name, digest in SOURCE_REVIEW_HASHES.items():
        path = SOURCE_REVIEW / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"package0015 genuine PASS artifact changed: {path}",
        )


def validate_source() -> dict[str, str]:
    require(
        SOURCE.is_dir()
        and not SOURCE.is_symlink()
        and stat.S_IMODE(SOURCE.stat().st_mode) == 0o555,
        "package0015 directory mode changed",
    )
    records: dict[str, str] = {}
    for line in (SOURCE / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name not in records,
            "package0015 checksum manifest changed",
        )
        records[name] = digest
    observed = {
        item.relative_to(SOURCE).as_posix()
        for item in SOURCE.rglob("*")
        if item.is_file() and item.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    }
    require(observed == set(records), "package0015 sealed member set changed")
    for name, digest in records.items():
        path = SOURCE / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"package0015 member changed: {path}",
        )
    require(
        sha256_file(SOURCE / "SHA256SUMS") == SOURCE_ROOT
        and (SOURCE / "TREE_ROOT.sha256").read_text(encoding="ascii").split()
        == [SOURCE_ROOT, "SHA256SUMS"],
        "package0015 root changed",
    )
    validate_source_review()
    require(DIRECTIVE.is_file() and not DIRECTIVE.is_symlink(), "Manager V5 absent")
    directive = json.loads(DIRECTIVE.read_text(encoding="utf-8"))
    require(
        isinstance(directive, dict)
        and isinstance(directive.get("revision"), str)
        and isinstance(directive.get("objective_sha256"), str)
        and len(directive["objective_sha256"]) == 64
        and isinstance(directive.get("text"), str)
        and "MANAGER GRANT LAYER16 V5" in directive["text"]
        and "package0016" in directive["text"]
        and "assistant.turn_end.parentId" in directive["text"]
        and "Do not demand parentToolCallId" in directive["text"],
        "active Manager V5 directive changed",
    )
    require(AGENT_IO.is_file() and not AGENT_IO.is_symlink(), "Host agent_io absent")
    for path in (
        TARGET,
        ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0016",
        ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0016",
        ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0016",
    ):
        require(not os.path.lexists(path), f"package0016 target preexists: {path}")
    return {
        "revision": directive["revision"],
        "sha256": sha256_file(DIRECTIVE),
        "objective_sha256": directive["objective_sha256"],
    }


HOST_STREAM = r'''def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field rejected: {key}")
        result[key] = value
    return result


def _parse_json_no_duplicates(raw: str) -> Any:
    return json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)


def _host_stream_events() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in HOST_AGENT_IO.read_text(encoding="utf-8").splitlines():
        try:
            outer = _parse_json_no_duplicates(raw)
            if not isinstance(outer, dict):
                continue
            inner = _parse_json_no_duplicates(outer.get("line", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if (
            outer.get("type") == "agent.io.stream"
            and outer.get("io_kind") == "stream"
            and isinstance(inner, dict)
        ):
            events.append(
                {
                    "host_call_id": outer.get("call_id"),
                    "host_timestamp": outer.get("ts"),
                    "inner": inner,
                }
            )
    return events


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event["inner"].get("data")
    return data if isinstance(data, dict) else {}


def _event_id(event: dict[str, Any]) -> str:
    value = event["inner"].get("id")
    require(isinstance(value, str) and bool(value), "Host event id required")
    return value


def _event_parent_id(event: dict[str, Any]) -> str:
    value = event["inner"].get("parentId")
    require(isinstance(value, str) and bool(value), "Host event parentId required")
    return value


def _event_turn_id(event: dict[str, Any]) -> str:
    value = _event_data(event).get("turnId", event["inner"].get("turnId"))
    require(isinstance(value, str) and bool(value), "Host event turnId required")
    return value


def _event_timestamp(event: dict[str, Any]) -> int | float | str:
    value = event["inner"].get("timestamp")
    require(
        (
            isinstance(value, str)
            and bool(value)
        )
        or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ),
        "Host event timestamp required",
    )
    return value


def _require_chronology(
    events: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> None:
    indexes = [events.index(event) for event in selected]
    require(
        indexes == sorted(indexes) and len(set(indexes)) == len(indexes),
        "Host events are missing, duplicated, or reordered",
    )
    timestamps = [_event_timestamp(event) for event in selected]
    require(
        all(type(value) is type(timestamps[0]) for value in timestamps)
        and all(left < right for left, right in zip(timestamps, timestamps[1:])),
        "Host event timestamps are missing, duplicated, or reordered",
    )
    host_timestamps = [event.get("host_timestamp") for event in selected]
    require(
        all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in host_timestamps
        )
        and all(
            left < right
            for left, right in zip(host_timestamps, host_timestamps[1:])
        ),
        "Host stream timestamps are missing, duplicated, or reordered",
    )


'''


HOST_LAUNCH = r'''def validate_supported_task_launch(
    expected_agent_id: str,
) -> dict[str, str]:
    require(
        _valid_agent_id(expected_agent_id)
        and expected_agent_id != ENGINEER_MISSION_ID
        and expected_agent_id not in STALE_REVIEWER_IDENTITIES,
        "fresh source-disjoint supported-task Reviewer identity required",
    )
    events = _host_stream_events()
    launch_index, binding = _task_launch(events)
    assigned = [
        event
        for event in events[launch_index + 1 :]
        if event.get("host_call_id") == binding["host_call_id"]
        and event["inner"].get("agentId") == expected_agent_id
        and event["inner"].get("type")
        in {"assistant.message", "assistant.turn_end"}
    ]
    require(
        bool(assigned)
        and not any(
            event.get("host_call_id") == binding["host_call_id"]
            and event["inner"].get("agentId") == ENGINEER_MISSION_ID
            for event in events[launch_index + 1 :]
        ),
        "Host task stream does not assign the claimed Reviewer identity",
    )
    return binding


'''


HOST_COMPLETION = r'''def _structured_result(
    event: dict[str, Any],
    label: str,
) -> dict[str, str]:
    data = _event_data(event)
    require("result" in data, f"{label} result field required")
    result = data["result"]
    require(
        isinstance(result, dict)
        and set(result) == {"content", "detailedContent"}
        and isinstance(result["content"], str)
        and isinstance(result["detailedContent"], str),
        f"{label} result must use the exact structured Host schema",
    )
    return result


def _require_exact_final(
    observed: str,
    expected: str,
    expected_sha256: str,
    label: str,
) -> None:
    require(observed == expected, f"{label} final-response content mismatch")
    require(
        hashlib.sha256(observed.encode("utf-8")).hexdigest() == expected_sha256,
        f"{label} final-response hash mismatch",
    )


def _read_agent_final_response(
    result: dict[str, str],
    expected_agent_id: str,
    supported_task_state: str,
) -> str:
    match = re.fullmatch(
        r"Agent .*? agent_id: (?P<agent_id>[^,\n]+), "
        r"agent_type: (?P<agent_type>[^,\n]+), "
        r"status: (?P<status>[^,\n]+), "
        r"description: (?P<description>.*), elapsed: (?P<elapsed>[0-9]+s), "
        r"total_turns: (?P<turns>[0-9]+), model: (?P<model>[^\n]+)"
        r"\n\n\[Turn 0\]\n(?P<response>.*)",
        result["content"],
        flags=re.DOTALL,
    )
    require(match is not None, "read_agent result framing is malformed")
    require(
        match.group("agent_id") == expected_agent_id
        and match.group("agent_type") == "general-purpose"
        and match.group("status") == supported_task_state
        and match.group("turns") == "1"
        and bool(match.group("model")),
        "read_agent result identity, state, or turn count mismatch",
    )
    require(
        result["detailedContent"]
        == f"<agent {expected_agent_id} {supported_task_state}>",
        "read_agent detailed result identity or state mismatch",
    )
    return match.group("response")


def validate_host_completion_events(
    events: list[dict[str, Any]],
    expected_agent_id: str,
    supported_task_state: str,
) -> dict[str, str]:
    require(
        supported_task_state in {"completed", "idle"},
        "running, no-response, cancelled, and failed task states are rejected",
    )
    require(
        _valid_agent_id(expected_agent_id)
        and expected_agent_id != ENGINEER_MISSION_ID
        and expected_agent_id not in STALE_REVIEWER_IDENTITIES,
        "fresh source-disjoint supported-task Reviewer identity required",
    )
    launch_index, binding = _task_launch(events)
    tool_call_id = binding["task_tool_call_id"]
    launch = events[launch_index]
    later = events[launch_index + 1 :]
    finals = [
        event
        for event in later
        if event["inner"].get("type") == "assistant.message"
        and event["inner"].get("agentId") == expected_agent_id
        and _event_data(event).get("phase") == "final_answer"
        and isinstance(_event_data(event).get("content"), str)
        and bool(_event_data(event)["content"])
    ]
    require(len(finals) == 1, "exactly one durable completed Reviewer final response required")
    final = finals[0]
    final_index = events.index(final)
    final_response = _event_data(final)["content"]
    final_response_sha256 = hashlib.sha256(
        final_response.encode("utf-8")
    ).hexdigest()
    final_turn_id = _event_turn_id(final)
    final_turn_ends = [
        event
        for event in events[final_index + 1 :]
        if event["inner"].get("type") == "assistant.turn_end"
        and event["inner"].get("agentId") == expected_agent_id
        and _event_turn_id(event) == final_turn_id
        and event.get("host_call_id") == final.get("host_call_id")
    ]
    require(
        len(final_turn_ends) == 1,
        "idle is terminal only after exactly one durable completed final turn",
    )
    final_turn_end = final_turn_ends[0]
    _require_chronology(events, [final, final_turn_end])
    task_completions = [
        event
        for event in later
        if event["inner"].get("type") == "tool.execution_complete"
        and _event_data(event).get("toolCallId") == tool_call_id
        and _event_data(event).get("success") is True
    ]
    require(len(task_completions) == 1, "successful supported task tool return required")
    task_completion = task_completions[0]
    assigned = {
        event["inner"].get("agentId")
        for event in events[launch_index + 1 : events.index(task_completion) + 1]
        if isinstance(event["inner"].get("agentId"), str)
    }
    require(
        expected_agent_id in assigned
        and ENGINEER_MISSION_ID not in assigned
        and assigned <= {expected_agent_id},
        "forged, joint, stale, or Engineer-reused task identity rejected",
    )
    require(
        events.index(task_completion) > events.index(final_turn_end)
        and task_completion.get("host_call_id") == binding["host_call_id"]
        and _event_turn_id(task_completion) == _event_turn_id(launch),
        "supported task completion call, turn, or order mismatch",
    )
    _require_chronology(events, [launch, final, final_turn_end, task_completion])
    task_result = _structured_result(task_completion, "task")
    require(
        task_result["content"] == task_result["detailedContent"],
        "task result content fields disagree",
    )
    _require_exact_final(
        task_result["content"],
        final_response,
        final_response_sha256,
        "task",
    )
    read_starts = [
        event
        for event in events[events.index(task_completion) + 1 :]
        if event["inner"].get("type") == "tool.execution_start"
        and _event_data(event).get("toolName")
        in {"read_agent", "functions.read_agent"}
        and isinstance(_event_data(event).get("arguments"), dict)
        and _event_data(event)["arguments"].get("agent_id") == expected_agent_id
    ]
    require(len(read_starts) == 1, "exactly one matching read_agent request required")
    read_start = read_starts[0]
    read_call_id = _event_data(read_start).get("toolCallId")
    require(
        isinstance(read_call_id, str) and bool(read_call_id),
        "read_agent toolCallId required",
    )
    request_messages = [
        event
        for event in events[events.index(task_completion) + 1 : events.index(read_start)]
        if event["inner"].get("type") == "assistant.message"
        and isinstance(_event_data(event).get("toolRequests"), list)
        and any(
            isinstance(request, dict)
            and request.get("toolCallId") == read_call_id
            and request.get("name")
            in {"read_agent", "functions.read_agent"}
            and isinstance(request.get("arguments"), dict)
            and request["arguments"].get("agent_id") == expected_agent_id
            for request in _event_data(event)["toolRequests"]
        )
    ]
    require(
        len(request_messages) == 1,
        "exactly one preceding assistant.message read_agent request required",
    )
    request_message = request_messages[0]
    matching_requests = [
        request
        for request in _event_data(request_message)["toolRequests"]
        if isinstance(request, dict)
        and request.get("toolCallId") == read_call_id
    ]
    require(
        len(matching_requests) == 1
        and matching_requests[0].get("name")
        in {"read_agent", "functions.read_agent"}
        and isinstance(matching_requests[0].get("arguments"), dict)
        and matching_requests[0]["arguments"].get("agent_id")
        == expected_agent_id,
        "read_agent assistant tool request identity mismatch",
    )
    read_completions = [
        event
        for event in events[events.index(read_start) + 1 :]
        if event["inner"].get("type") == "tool.execution_complete"
        and _event_data(event).get("toolCallId") == read_call_id
        and _event_data(event).get("success") is True
    ]
    require(len(read_completions) == 1, "successful matching read_agent result required")
    read_completion = read_completions[0]
    parent_turn_ends = [
        event
        for event in events[events.index(read_completion) + 1 :]
        if event["inner"].get("type") == "assistant.turn_end"
        and _event_parent_id(event) == _event_id(read_completion)
    ]
    require(
        len(parent_turn_ends) == 1,
        "exactly one parent-bound read_agent assistant.turn_end required",
    )
    parent_turn_end = parent_turn_ends[0]
    chain = [request_message, read_start, read_completion, parent_turn_end]
    turn_ids = {_event_turn_id(event) for event in chain}
    host_call_ids = {event.get("host_call_id") for event in chain}
    require(
        len(turn_ids) == 1
        and len(host_call_ids) == 1
        and next(iter(host_call_ids)) == binding["host_call_id"],
        "read_agent Host call_id or turnId mismatch",
    )
    require(
        _event_parent_id(read_completion) == _event_id(read_start)
        and _event_parent_id(parent_turn_end) == _event_id(read_completion)
        and _event_data(read_start).get("toolCallId") == read_call_id
        and _event_data(read_completion).get("toolCallId") == read_call_id,
        "read_agent parentId or toolCallId ancestry mismatch",
    )
    _require_chronology(events, [task_completion, *chain])
    read_result = _structured_result(read_completion, "read_agent")
    read_final_response = _read_agent_final_response(
        read_result,
        expected_agent_id,
        supported_task_state,
    )
    _require_exact_final(
        read_final_response,
        final_response,
        final_response_sha256,
        "read_agent",
    )
    return {
        **binding,
        "agent_id": expected_agent_id,
        "final_response_sha256": final_response_sha256,
        "read_agent_completion_event_id": _event_id(read_completion),
        "read_agent_start_event_id": _event_id(read_start),
        "read_agent_tool_call_id": str(read_call_id),
        "read_agent_turn_end_event_id": _event_id(parent_turn_end),
        "turn_id": next(iter(turn_ids)),
        "state": supported_task_state,
    }


'''


HOSTILE_EVENTS = r'''    host_agent = "task-agent-reviewer-0016"
    task_call = "call_layer16_package0016"
    read_call = "call_read_agent_package0016"
    final_response = "PACKAGE0016 REVIEW PASS\nstructured Host result"

    def event(
        event_type: str,
        data: dict[str, object],
        *,
        event_id: str,
        parent_id: str | None = None,
        agent_id: str | None = None,
        timestamp: int,
    ) -> dict[str, object]:
        inner: dict[str, object] = {
            "id": event_id,
            "type": event_type,
            "data": data,
            "timestamp": timestamp,
        }
        if parent_id is not None:
            inner["parentId"] = parent_id
        if agent_id is not None:
            inner["agentId"] = agent_id
        return {
            "host_call_id": "host-call-package0016",
            "host_timestamp": float(timestamp),
            "inner": inner,
        }

    task_prompt = (
        layer16.PACKAGE.relative_to(layer16.ROOT).as_posix()
        + " "
        + str((layer16.PACKAGE / "review_package.py").resolve())
    )
    read_content = (
        "Agent is idle (waiting for messages). "
        f"agent_id: {host_agent}, agent_type: general-purpose, status: idle, "
        "description: Adjudicate sealed package0016, elapsed: 1s, "
        "total_turns: 1, model: reviewer-model\n\n[Turn 0]\n"
        + final_response
    )
    host_events = [
        event(
            "tool.execution_start",
            {
                "toolCallId": task_call,
                "toolName": "functions.task",
                "turnId": "parent-turn",
                "arguments": {
                    "name": layer16.REVIEW_TASK_NAME,
                    "agent_type": "general-purpose",
                    "mode": "sync",
                    "prompt": task_prompt,
                },
            },
            event_id="task-start",
            timestamp=1,
        ),
        event(
            "assistant.message",
            {
                "phase": "final_answer",
                "content": final_response,
                "turnId": "reviewer-turn",
            },
            event_id="reviewer-final",
            agent_id=host_agent,
            timestamp=2,
        ),
        event(
            "assistant.turn_end",
            {"turnId": "reviewer-turn"},
            event_id="reviewer-turn-end",
            agent_id=host_agent,
            timestamp=3,
        ),
        event(
            "tool.execution_complete",
            {
                "toolCallId": task_call,
                "turnId": "parent-turn",
                "success": True,
                "result": {
                    "content": final_response,
                    "detailedContent": final_response,
                },
            },
            event_id="task-complete",
            parent_id="task-start",
            agent_id=host_agent,
            timestamp=4,
        ),
        event(
            "assistant.message",
            {
                "turnId": "parent-turn",
                "toolRequests": [
                    {
                        "toolCallId": read_call,
                        "name": "read_agent",
                        "arguments": {"agent_id": host_agent},
                    }
                ],
            },
            event_id="read-request-message",
            timestamp=5,
        ),
        event(
            "tool.execution_start",
            {
                "toolCallId": read_call,
                "toolName": "read_agent",
                "turnId": "parent-turn",
                "arguments": {"agent_id": host_agent},
            },
            event_id="read-start",
            parent_id="read-request-message",
            timestamp=6,
        ),
        event(
            "tool.execution_complete",
            {
                "toolCallId": read_call,
                "turnId": "parent-turn",
                "success": True,
                "result": {
                    "content": read_content,
                    "detailedContent": f"<agent {host_agent} idle>",
                },
            },
            event_id="read-complete",
            parent_id="read-start",
            timestamp=7,
        ),
        event(
            "assistant.turn_end",
            {"turnId": "parent-turn"},
            event_id="parent-turn-end",
            parent_id="read-complete",
            timestamp=8,
        ),
    ]
    layer16.validate_host_completion_events(host_events, host_agent, "idle")

    def validate_events(
        value: list[dict[str, object]],
        agent_id: str = host_agent,
    ) -> None:
        layer16.validate_host_completion_events(value, agent_id, "idle")

    escaped_response = copy.deepcopy(host_events)
    escaped = layer16.json.dumps(final_response)
    escaped_response[3]["inner"]["data"]["result"] = {
        "content": escaped,
        "detailedContent": escaped,
    }
    cases.append(
        expect_rejection(
            "escaped-versus-raw-final-content",
            lambda: validate_events(escaped_response),
        )
    )
    wrong_field = copy.deepcopy(host_events)
    wrong_field[3]["inner"]["data"]["result"] = {
        "response": final_response,
        "detailedContent": final_response,
    }
    cases.append(
        expect_rejection("wrong-result-field", lambda: validate_events(wrong_field))
    )
    missing_result = copy.deepcopy(host_events)
    missing_result[3]["inner"]["data"].pop("result")
    cases.append(
        expect_rejection("missing-result-field", lambda: validate_events(missing_result))
    )
    cases.append(
        expect_rejection(
            "duplicate-result-field",
            lambda: layer16._parse_json_no_duplicates(
                '{"data":{"result":{},"result":{}}}'
            ),
        )
    )
    mismatched_response = copy.deepcopy(host_events)
    mismatched_response[3]["inner"]["data"]["result"]["content"] += " altered"
    cases.append(
        expect_rejection(
            "mismatched-final-response",
            lambda: validate_events(mismatched_response),
        )
    )
    cases.append(
        expect_rejection(
            "mismatched-final-response-hash",
            lambda: layer16._require_exact_final(
                final_response,
                final_response,
                "0" * 64,
                "hostile",
            ),
        )
    )
    mismatched_agent = copy.deepcopy(host_events)
    mismatched_agent[6]["inner"]["data"]["result"]["content"] = read_content.replace(
        host_agent,
        "task-agent-forged-0016",
    )
    cases.append(
        expect_rejection(
            "mismatched-result-agent",
            lambda: validate_events(mismatched_agent),
        )
    )
    missing_final = copy.deepcopy(host_events)
    missing_final.pop(1)
    cases.append(
        expect_rejection("missing-final-response", lambda: validate_events(missing_final))
    )
    idle_without_final_turn = copy.deepcopy(host_events)
    idle_without_final_turn.pop(2)
    cases.append(
        expect_rejection(
            "idle-without-final-turn",
            lambda: validate_events(idle_without_final_turn),
        )
    )
    cases.append(
        expect_rejection(
            "engineer-identity-reuse",
            lambda: validate_events(host_events, layer16.ENGINEER_MISSION_ID),
        )
    )
    altered_root = copy.deepcopy(host_events)
    altered_root[0]["inner"]["data"]["arguments"]["prompt"] = task_prompt.replace(
        layer16.PACKAGE.relative_to(layer16.ROOT).as_posix(),
        "reports/altered-package-root",
    )
    cases.append(
        expect_rejection("altered-task-root", lambda: validate_events(altered_root))
    )
    altered_argv = copy.deepcopy(host_events)
    altered_argv[0]["inner"]["data"]["arguments"]["prompt"] = task_prompt.replace(
        str((layer16.PACKAGE / "review_package.py").resolve()),
        "/tmp/altered-review-package.py",
    )
    cases.append(
        expect_rejection("altered-task-argv", lambda: validate_events(altered_argv))
    )
    joint = copy.deepcopy(host_events)
    joint.insert(
        2,
        event(
            "assistant.message",
            {
                "phase": "commentary",
                "content": "joint record",
                "turnId": "reviewer-turn",
            },
            event_id="joint-record",
            agent_id=layer16.ENGINEER_MISSION_ID,
            timestamp=2.5,
        ),
    )
    cases.append(expect_rejection("joint-records", lambda: validate_events(joint)))
    for state in ("running", "cancelled", "failed"):
        cases.append(
            expect_rejection(
                f"nonterminal-{state}",
                lambda state=state: layer16.validate_host_completion_events(
                    host_events,
                    host_agent,
                    state,
                ),
            )
        )

    relative_argv_log = copy.deepcopy(malformed_log)
    relative_argv_log["review_argv"] = layer16._review_argv(root, agent)
    relative_argv_log["review_argv"][2] = (
        layer16.PACKAGE / "review_package.py"
    ).relative_to(layer16.ROOT).as_posix()
    cases.append(
        expect_rejection(
            "relative-review-argv",
            lambda: layer16.validate_review_log_record(
                relative_argv_log,
                root,
                agent,
            ),
        )
    )

    missing = copy.deepcopy(host_events)
    missing[7]["inner"].pop("parentId")
    cases.append(
        expect_rejection("missing-parent", lambda: validate_events(missing))
    )
    duplicate = copy.deepcopy(host_events)
    duplicate.insert(7, copy.deepcopy(duplicate[6]))
    cases.append(
        expect_rejection("duplicate-completion", lambda: validate_events(duplicate))
    )
    reordered = copy.deepcopy(host_events)
    reordered[5], reordered[6] = reordered[6], reordered[5]
    cases.append(
        expect_rejection("reordered-events", lambda: validate_events(reordered))
    )
    cross_call = copy.deepcopy(host_events)
    cross_call[6]["host_call_id"] = "host-call-crossed"
    cases.append(
        expect_rejection("cross-call", lambda: validate_events(cross_call))
    )
    cross_turn = copy.deepcopy(host_events)
    cross_turn[6]["inner"]["data"]["turnId"] = "crossed-turn"
    cases.append(
        expect_rejection("cross-turn", lambda: validate_events(cross_turn))
    )
    wrong_parent = copy.deepcopy(host_events)
    wrong_parent[6]["inner"]["parentId"] = "wrong-start"
    cases.append(
        expect_rejection("wrong-parent", lambda: validate_events(wrong_parent))
    )
    wrong_tool_call = copy.deepcopy(host_events)
    wrong_tool_call[6]["inner"]["data"]["toolCallId"] = "wrong-tool-call"
    cases.append(
        expect_rejection(
            "wrong-toolCallId",
            lambda: validate_events(wrong_tool_call),
        )
    )
    wrong_agent = copy.deepcopy(host_events)
    wrong_agent[4]["inner"]["data"]["toolRequests"][0]["arguments"]["agent_id"] = (
        "task-agent-wrong-0016"
    )
    cases.append(
        expect_rejection("wrong-agent", lambda: validate_events(wrong_agent))
    )
    forged_content = copy.deepcopy(host_events)
    forged_content[6]["inner"]["data"]["result"]["content"] += " forged"
    cases.append(
        expect_rejection("forged-content", lambda: validate_events(forged_content))
    )

'''


def common_transform(text: str) -> str:
    text = text.replace("0015", "0016")
    text = text.replace("package 0015", "package 0016")
    text = text.replace("42e453e6fff4", ENGINEER_MISSION_ID)
    text = text.replace("_v8", "_v9")
    text = text.replace("-v8", "-v9")
    return text


def transform_executor(text: str, directive: dict[str, str]) -> str:
    text = common_transform(text)
    text = replace_once(
        text,
        'ACTIVE_DIRECTIVE_REVISION = "30d5970ccd9b45ce9caa6dbda424093f"',
        f'ACTIVE_DIRECTIVE_REVISION = "{directive["revision"]}"',
    )
    text = replace_once(
        text,
        '"8bc6368686cc0f779ff77ca08abf0cd6e478dc7429013e6d34960bb2d95dadd4"',
        f'"{directive["sha256"]}"',
    )
    text = replace_once(
        text,
        '"43fd081a2ec0196f5100f18980c9c58017888abf17b7cfcc77444e31ebd8ad39"',
        f'"{directive["objective_sha256"]}"',
    )
    text = replace_once(
        text,
        '''    "0014": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0014",
        "d8c0f7ebf53acdbfe46a94678502d8be3ac95d3019be7afb47231ce80a36b67b",
        "HOST_TASK_RESULT_RAW_SERIALIZATION_SUBSTRING",
    ),
}''',
        '''    "0014": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0014",
        "d8c0f7ebf53acdbfe46a94678502d8be3ac95d3019be7afb47231ce80a36b67b",
        "HOST_TASK_RESULT_RAW_SERIALIZATION_SUBSTRING",
    ),
    "0015": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0015",
        "a513b7c7003169c39abe39d0c56ec861f3ba59407fcf1a653e6a79ae5e5441fc",
        "HOST_ASSISTANT_TURN_END_PARENT_TOOL_CALL_ID_SCHEMA",
    ),
    "0014": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0014",
        "d8c0f7ebf53acdbfe46a94678502d8be3ac95d3019be7afb47231ce80a36b67b",
        "HOST_TASK_RESULT_RAW_SERIALIZATION_SUBSTRING",
    ),
}''',
    )
    text = replace_once(
        text,
        '''    "ce1b86cb-7a9c-4add-a1e7-3a76e0f341e2",
}''',
        '''    "ce1b86cb-7a9c-4add-a1e7-3a76e0f341e2",
    "5bdef423-848d-4f36-93e2-a625cc4dd444",
}''',
    )
    text = replace_once(
        text,
        '''            "STRUCTURED_HOST_TASK_RESULT_EXACT_FINAL_CONTENT_AND_HASH"''',
        '''            "HOST_PARENT_ID_ANCESTRY_AND_EXACT_FINAL_TURN"''',
    )
    text = replace_section(
        text,
        "def _host_stream_events()",
        "def _expected_task_arguments(",
        HOST_STREAM,
    )
    text = replace_section(
        text,
        "def validate_supported_task_launch(",
        "def _structured_result(",
        HOST_LAUNCH,
    )
    text = replace_section(
        text,
        "def validate_host_completion_events(",
        "def validate_supported_task_completion(",
        HOST_COMPLETION,
    )
    text = text.replace(
        '"failed_lifecycle_packages": "PASS_0009_0011_0012_0013_0014_PRESERVED",',
        '"failed_lifecycle_packages": "PASS_0009_0011_0012_0013_0014_0015_PRESERVED",',
    )
    require(
        "parentToolCallId" not in text
        and "final_response in task_result" not in text
        and "final_response in read_result" not in text
        and "json.dumps(\n        _event_data(task_completions" not in text,
        "obsolete Host ancestry or raw serialized result validator remained",
    )
    return text


def transform_hostile(text: str) -> str:
    text = common_transform(text)
    text = replace_section(
        text,
        '    host_agent = "task-agent-reviewer-0016"',
        "    after = layer16.live_snapshot()",
        HOSTILE_EVENTS,
    )
    return text


def transformed_sources(directive: dict[str, str]) -> dict[str, str]:
    transforms = {
        "execute_layer16.py": lambda value: transform_executor(value, directive),
        "hostile_controls.py": transform_hostile,
        "review_package.py": common_transform,
        "seal_package.py": common_transform,
        "validate_nonexecuting.py": common_transform,
        "validate_review.py": common_transform,
    }
    result = {
        name: transforms[name]((SOURCE / name).read_text(encoding="utf-8"))
        for name in sorted(SOURCE_MEMBERS)
    }
    for name, text in result.items():
        compile(text, str(TARGET / name), "exec")
        require("0015-review" not in text, f"stale package review path in {name}")
    return result


def main() -> int:
    directive = validate_source()
    sources = transformed_sources(directive)
    os.mkdir(TARGET, mode=0o700)
    for name, text in sources.items():
        write_exclusive(TARGET / name, text.encode("utf-8"), mode=0o444)
    result = subprocess.run(
        [
            "/home/argustest/miniconda3/bin/python3.13",
            "-B",
            str(TARGET / "seal_package.py"),
        ],
        cwd=ROOT,
        env={
            "LC_ALL": "C",
            "PATH": "/home/argustest/miniconda3/bin:/usr/bin:/bin",
            "PYTHONCOERCECLOCALE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONSAFEPATH": "1",
            "PYTHONUTF8": "1",
        },
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(
        result.returncode == 0,
        "package0016 seal failed: " + result.stderr.decode("utf-8", "replace"),
    )
    validate_source_review()
    require(sha256_file(SOURCE / "SHA256SUMS") == SOURCE_ROOT, "package0015 changed")
    sys.stdout.buffer.write(result.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
