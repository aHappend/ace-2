#!/usr/bin/env python3
"""Create and seal layer-16 authority package 0015 exclusively."""

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
SOURCE = ROOT / "reports/ace2-layer16-runtime-pass-authority-0014"
SOURCE_REVIEW = ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0014"
TARGET = ROOT / "reports/ace2-layer16-runtime-pass-authority-0015"
SOURCE_ROOT = "d8c0f7ebf53acdbfe46a94678502d8be3ac95d3019be7afb47231ce80a36b67b"
ENGINEER_MISSION_ID = "42e453e6fff4"
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
        "225119b92f46c17f4318aec23f659969d849e2754ff5f342ea14fcb136d22da9"
    ),
    "reviewer-adjudication.json": (
        "c673d0d84e920cd724d0e9857e39a0514fbde57eab99c7d4fa60809b8fba94f2"
    ),
    "reviewer-receipt.json": (
        "63f6bd0b67b5de5d9c39f5e3063e90ec9f06803f4e5524ae7ec8088109b332ca"
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
        "package0014 review namespace changed",
    )
    require(
        {item.name for item in SOURCE_REVIEW.iterdir()} == set(SOURCE_REVIEW_HASHES),
        "package0014 genuine PASS artifact set changed",
    )
    for name, digest in SOURCE_REVIEW_HASHES.items():
        path = SOURCE_REVIEW / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"package0014 genuine PASS artifact changed: {path}",
        )


def validate_source() -> dict[str, str]:
    require(
        SOURCE.is_dir()
        and not SOURCE.is_symlink()
        and stat.S_IMODE(SOURCE.stat().st_mode) == 0o555,
        "package0014 directory mode changed",
    )
    records: dict[str, str] = {}
    for line in (SOURCE / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name not in records,
            "package0014 checksum manifest changed",
        )
        records[name] = digest
    observed = {
        item.relative_to(SOURCE).as_posix()
        for item in SOURCE.rglob("*")
        if item.is_file() and item.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    }
    require(observed == set(records), "package0014 sealed member set changed")
    for name, digest in records.items():
        path = SOURCE / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"package0014 member changed: {path}",
        )
    require(
        sha256_file(SOURCE / "SHA256SUMS") == SOURCE_ROOT
        and (SOURCE / "TREE_ROOT.sha256").read_text(encoding="ascii").split()
        == [SOURCE_ROOT, "SHA256SUMS"],
        "package0014 root changed",
    )
    validate_source_review()
    require(DIRECTIVE.is_file() and not DIRECTIVE.is_symlink(), "Manager V4 absent")
    directive = json.loads(DIRECTIVE.read_text(encoding="utf-8"))
    require(
        isinstance(directive, dict)
        and isinstance(directive.get("revision"), str)
        and isinstance(directive.get("objective_sha256"), str)
        and len(directive["objective_sha256"]) == 64
        and isinstance(directive.get("text"), str)
        and "MANAGER GRANT LAYER16 V4" in directive["text"]
        and "package0015" in directive["text"]
        and "parse the Host task-result JSON/event structure" in directive["text"],
        "active Manager V4 directive changed",
    )
    require(AGENT_IO.is_file() and not AGENT_IO.is_symlink(), "Host agent_io absent")
    for path in (
        TARGET,
        ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0015",
        ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0015",
        ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0015",
    ):
        require(not os.path.lexists(path), f"package0015 target preexists: {path}")
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
            events.append({"host_call_id": outer.get("call_id"), "inner": inner})
    return events


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
    later = events[launch_index + 1 :]
    assigned = {
        event["inner"].get("agentId")
        for event in later
        if _event_parent_tool(event) == tool_call_id
        and isinstance(event["inner"].get("agentId"), str)
    }
    require(
        expected_agent_id in assigned
        and ENGINEER_MISSION_ID not in assigned
        and assigned <= {expected_agent_id},
        "forged, joint, stale, or Engineer-reused task identity rejected",
    )
    finals = [
        event
        for event in later
        if event["inner"].get("type") == "assistant.message"
        and event["inner"].get("agentId") == expected_agent_id
        and _event_parent_tool(event) == tool_call_id
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
    require(
        any(
            event["inner"].get("type") == "assistant.turn_end"
            and event["inner"].get("agentId") == expected_agent_id
            and _event_parent_tool(event) == tool_call_id
            for event in events[final_index + 1 :]
        ),
        "idle is terminal only after a durable completed final turn",
    )
    task_completions = [
        event
        for event in later
        if event["inner"].get("type") == "tool.execution_complete"
        and _event_data(event).get("toolCallId") == tool_call_id
        and _event_data(event).get("success") is True
    ]
    require(len(task_completions) == 1, "successful supported task tool return required")
    task_result = _structured_result(task_completions[0], "task")
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
        for event in events[events.index(task_completions[0]) + 1 :]
        if event["inner"].get("type") == "tool.execution_start"
        and _event_data(event).get("toolName") == "read_agent"
        and isinstance(_event_data(event).get("arguments"), dict)
        and _event_data(event)["arguments"].get("agent_id") == expected_agent_id
    ]
    require(len(read_starts) == 1, "exactly one matching read_agent request required")
    read_call_id = _event_data(read_starts[0]).get("toolCallId")
    read_completions = [
        event
        for event in events[events.index(read_starts[0]) + 1 :]
        if event["inner"].get("type") == "tool.execution_complete"
        and _event_data(event).get("toolCallId") == read_call_id
        and _event_data(event).get("success") is True
    ]
    require(len(read_completions) == 1, "successful matching read_agent result required")
    read_result = _structured_result(read_completions[0], "read_agent")
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
        "read_agent_tool_call_id": str(read_call_id),
        "state": supported_task_state,
    }


'''


HOSTILE_EVENTS = r'''    host_agent = "task-agent-reviewer-0015"
    task_call = "call_layer16_package0015"
    read_call = "call_read_agent_package0015"
    final_response = "PACKAGE0015 REVIEW PASS\nstructured Host result"

    def event(
        event_type: str,
        data: dict[str, object],
        *,
        agent_id: str | None = None,
        parent_tool: str | None = None,
    ) -> dict[str, object]:
        inner: dict[str, object] = {"type": event_type, "data": data}
        if agent_id is not None:
            inner["agentId"] = agent_id
        if parent_tool is not None:
            inner["parentToolCallId"] = parent_tool
        return {"host_call_id": "host-call-package0015", "inner": inner}

    task_prompt = (
        layer16.PACKAGE.relative_to(layer16.ROOT).as_posix()
        + " "
        + str((layer16.PACKAGE / "review_package.py").resolve())
    )
    read_content = (
        "Agent is idle (waiting for messages). "
        f"agent_id: {host_agent}, agent_type: general-purpose, status: idle, "
        "description: Adjudicate sealed package0015, elapsed: 1s, "
        "total_turns: 1, model: reviewer-model\n\n[Turn 0]\n"
        + final_response
    )
    host_events = [
        event(
            "tool.execution_start",
            {
                "toolCallId": task_call,
                "toolName": "functions.task",
                "arguments": {
                    "name": layer16.REVIEW_TASK_NAME,
                    "agent_type": "general-purpose",
                    "mode": "sync",
                    "prompt": task_prompt,
                },
            },
        ),
        event(
            "assistant.message",
            {"phase": "final_answer", "content": final_response},
            agent_id=host_agent,
            parent_tool=task_call,
        ),
        event(
            "assistant.turn_end",
            {},
            agent_id=host_agent,
            parent_tool=task_call,
        ),
        event(
            "tool.execution_complete",
            {
                "toolCallId": task_call,
                "success": True,
                "result": {
                    "content": final_response,
                    "detailedContent": final_response,
                },
            },
            agent_id=host_agent,
        ),
        event(
            "tool.execution_start",
            {
                "toolCallId": read_call,
                "toolName": "read_agent",
                "arguments": {"agent_id": host_agent},
            },
        ),
        event(
            "tool.execution_complete",
            {
                "toolCallId": read_call,
                "success": True,
                "result": {
                    "content": read_content,
                    "detailedContent": f"<agent {host_agent} idle>",
                },
            },
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
    mismatched_agent[5]["inner"]["data"]["result"]["content"] = read_content.replace(
        host_agent,
        "task-agent-forged-0015",
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
            {"phase": "commentary", "content": "joint record"},
            agent_id=layer16.ENGINEER_MISSION_ID,
            parent_tool=task_call,
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

'''


def common_transform(text: str) -> str:
    text = text.replace("0014", "0015")
    text = text.replace("package 0014", "package 0015")
    text = text.replace("610fe5211f7e", ENGINEER_MISSION_ID)
    text = text.replace("_v7", "_v8")
    text = text.replace("-v7", "-v8")
    return text


def transform_executor(text: str, directive: dict[str, str]) -> str:
    text = common_transform(text)
    text = replace_once(
        text,
        'ACTIVE_DIRECTIVE_REVISION = "8c62818c6a4d469f897673d76cf1cd0a"',
        f'ACTIVE_DIRECTIVE_REVISION = "{directive["revision"]}"',
    )
    text = replace_once(
        text,
        '"3e5b6daea29ac50bb18c6acc25e59ba8cb348437ea0955c986285f233d6dd491"',
        f'"{directive["sha256"]}"',
    )
    text = replace_once(
        text,
        '"43fd081a2ec0196f5100f18980c9c58017888abf17b7cfcc77444e31ebd8ad39"',
        f'"{directive["objective_sha256"]}"',
    )
    text = replace_once(
        text,
        '''    "0013": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0013",
        "d766ea01b33e9b4139cd302696a4fada33634e2b4fd25b4db2702a5f4139ab1e",
        "PRE_COMPUTATION_REVIEW_ARGV_CONTRACT",
    ),
}''',
        '''    "0013": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0013",
        "d766ea01b33e9b4139cd302696a4fada33634e2b4fd25b4db2702a5f4139ab1e",
        "PRE_COMPUTATION_REVIEW_ARGV_CONTRACT",
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
        '''    "2f27533f74db",
}''',
        '''    "2f27533f74db",
    "ce1b86cb-7a9c-4add-a1e7-3a76e0f341e2",
}''',
    )
    text = replace_once(
        text,
        '''            "SUPPORTED_TASK_TOOL_AGENT_ID_AND_COMPLETION_REPLACE_FRAMEWORK_"
            "REVIEWER_LIFECYCLE_PREDICATES"''',
        '''            "STRUCTURED_HOST_TASK_RESULT_EXACT_FINAL_CONTENT_AND_HASH"''',
    )
    text = replace_section(
        text,
        "def _host_stream_events()",
        "def _event_data(",
        HOST_STREAM,
    )
    text = replace_section(
        text,
        "def validate_host_completion_events(",
        "def validate_supported_task_completion(",
        HOST_COMPLETION,
    )
    text = text.replace(
        '"failed_lifecycle_packages": "PASS_0009_0011_0012_0013_PRESERVED",',
        '"failed_lifecycle_packages": "PASS_0009_0011_0012_0013_0014_PRESERVED",',
    )
    require(
        "final_response in task_result" not in text
        and "final_response in read_result" not in text
        and "json.dumps(\n        _event_data(task_completions" not in text,
        "raw serialized result substring validator remained",
    )
    return text


def transform_hostile(text: str) -> str:
    text = common_transform(text)
    text = replace_section(
        text,
        '    host_agent = "task-agent-reviewer-0015"',
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
        require("0014-review" not in text, f"stale package review path in {name}")
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
        "package0015 seal failed: " + result.stderr.decode("utf-8", "replace"),
    )
    validate_source_review()
    require(sha256_file(SOURCE / "SHA256SUMS") == SOURCE_ROOT, "package0014 changed")
    sys.stdout.buffer.write(result.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
