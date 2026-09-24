#!/usr/bin/env python3
"""Create and seal layer-16 authority package 0014 exclusively."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path("/home/argustest/ace-2")
SOURCE = ROOT / "reports/ace2-layer16-runtime-pass-authority-0013"
TARGET = ROOT / "reports/ace2-layer16-runtime-pass-authority-0014"
SOURCE_ROOT = "d766ea01b33e9b4139cd302696a4fada33634e2b4fd25b4db2702a5f4139ab1e"
ENGINEER_MISSION_ID = "610fe5211f7e"
DIRECTIVE = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "active_manager_directive.json"
)
DIRECTIVE_REVISION = "8c62818c6a4d469f897673d76cf1cd0a"
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


def validate_source() -> str:
    require(
        SOURCE.is_dir()
        and not SOURCE.is_symlink()
        and stat.S_IMODE(SOURCE.stat().st_mode) == 0o555,
        "package0013 directory mode changed",
    )
    records: dict[str, str] = {}
    for line in (SOURCE / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name not in records,
            "package0013 checksum manifest changed",
        )
        records[name] = digest
    observed = {
        item.relative_to(SOURCE).as_posix()
        for item in SOURCE.rglob("*")
        if item.is_file() and item.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    }
    require(observed == set(records), "package0013 sealed member set changed")
    for name, digest in records.items():
        path = SOURCE / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"package0013 member changed: {path}",
        )
    require(
        sha256_file(SOURCE / "SHA256SUMS") == SOURCE_ROOT
        and (SOURCE / "TREE_ROOT.sha256").read_text(encoding="ascii").split()
        == [SOURCE_ROOT, "SHA256SUMS"],
        "package0013 root changed",
    )
    require(
        DIRECTIVE.is_file()
        and not DIRECTIVE.is_symlink()
        and DIRECTIVE_REVISION
        in DIRECTIVE.read_text(encoding="utf-8")
        and "Authorize fresh create-exclusive package 0014 only"
        in DIRECTIVE.read_text(encoding="utf-8"),
        "active Manager V3 directive changed",
    )
    require(AGENT_IO.is_file() and not AGENT_IO.is_symlink(), "Host agent_io absent")
    for path in (
        TARGET,
        ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0014",
        ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0014",
        ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0014",
    ):
        require(not os.path.lexists(path), f"package0014 target preexists: {path}")
    return sha256_file(DIRECTIVE)


PRESERVED = '''PRESERVED_FAILED_PACKAGES = {
    "0009": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0009",
        "0b344534479b7db2a229760c70b0c72319497014dc4324f4f716e93cf5d19a8f",
        "PRE_COMPUTATION_PREDECESSOR_MISSION",
    ),
    "0011": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0011",
        "226b25f230096f38b41467c67e4f83e71328013c8b3f1174606a87ff90e35219",
        "PRE_COMPUTATION_REVIEWER_LIFECYCLE_DEPENDENCY",
    ),
    "0012": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0012",
        "ba56f7c5b97e2bac32c1626af1dbf3dd06131067b8473fb153c602a4070351a3",
        "PRE_COMPUTATION_REVIEWER_LIFECYCLE_DEPENDENCY",
    ),
    "0013": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0013",
        "d766ea01b33e9b4139cd302696a4fada33634e2b4fd25b4db2702a5f4139ab1e",
        "PRE_COMPUTATION_REVIEW_ARGV_CONTRACT",
    ),
}
'''


REVIEW_CONSTANTS = f'''ENGINEER_MISSION_ID = "{ENGINEER_MISSION_ID}"
REVIEW_TASK_NAME = "layer16-package0014-reviewer"
REVIEW_LOG = REVIEW_NAMESPACE / "review-log.json"
REVIEW_RECEIPT = REVIEW_NAMESPACE / "reviewer-receipt.json"
HOST_AGENT_IO = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/agent_io.jsonl"
)
'''


HOST_PROVENANCE = r'''def _host_stream_events() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in HOST_AGENT_IO.read_text(encoding="utf-8").splitlines():
        try:
            outer = json.loads(raw)
            inner = json.loads(outer.get("line", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if (
            outer.get("type") == "agent.io.stream"
            and outer.get("io_kind") == "stream"
            and isinstance(inner, dict)
        ):
            events.append({"host_call_id": outer.get("call_id"), "inner": inner})
    return events


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event["inner"].get("data")
    return data if isinstance(data, dict) else {}


def _event_parent_tool(event: dict[str, Any]) -> object:
    inner = event["inner"]
    return inner.get("parentToolCallId", _event_data(event).get("parentToolCallId"))


def _expected_task_arguments(arguments: object) -> bool:
    if not isinstance(arguments, dict):
        return False
    prompt = arguments.get("prompt")
    return (
        arguments.get("name") == REVIEW_TASK_NAME
        and arguments.get("agent_type") == "general-purpose"
        and arguments.get("mode") == "sync"
        and isinstance(prompt, str)
        and PACKAGE.relative_to(ROOT).as_posix() in prompt
        and str((PACKAGE / "review_package.py").resolve()) in prompt
    )


def _task_launch(events: list[dict[str, Any]]) -> tuple[int, dict[str, str]]:
    matches: list[tuple[int, dict[str, str]]] = []
    for index, event in enumerate(events):
        inner = event["inner"]
        data = _event_data(event)
        if (
            inner.get("type") == "tool.execution_start"
            and data.get("toolName") in {"task", "functions.task"}
            and _expected_task_arguments(data.get("arguments"))
            and isinstance(data.get("toolCallId"), str)
            and isinstance(event.get("host_call_id"), str)
        ):
            matches.append(
                (
                    index,
                    {
                        "agent_io_path": str(HOST_AGENT_IO),
                        "host_call_id": event["host_call_id"],
                        "task_tool_call_id": data["toolCallId"],
                    },
                )
            )
    require(len(matches) == 1, "exactly one supported functions.task launch required")
    return matches[0]


def validate_supported_task_launch(expected_agent_id: str) -> dict[str, str]:
    require(
        _valid_agent_id(expected_agent_id)
        and expected_agent_id != ENGINEER_MISSION_ID
        and expected_agent_id not in STALE_REVIEWER_IDENTITIES,
        "fresh source-disjoint supported-task Reviewer identity required",
    )
    events = _host_stream_events()
    launch_index, binding = _task_launch(events)
    assigned = {
        event["inner"].get("agentId")
        for event in events[launch_index + 1 :]
        if _event_parent_tool(event) == binding["task_tool_call_id"]
        and isinstance(event["inner"].get("agentId"), str)
    }
    require(
        expected_agent_id in assigned and ENGINEER_MISSION_ID not in assigned,
        "Host task stream does not assign the claimed Reviewer identity",
    )
    return binding


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
    require(
        any(
            event["inner"].get("type") == "assistant.turn_end"
            and event["inner"].get("agentId") == expected_agent_id
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
    task_result = json.dumps(
        _event_data(task_completions[0]).get("result"),
        sort_keys=True,
        ensure_ascii=True,
    )
    require(
        expected_agent_id in task_result and final_response in task_result,
        "task return must bind the assigned agent_id and final response",
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
    read_result = json.dumps(
        _event_data(read_completions[0]).get("result"),
        sort_keys=True,
        ensure_ascii=True,
    )
    require(
        expected_agent_id in read_result and final_response in read_result,
        "read_agent evidence must bind the assigned agent_id and final response",
    )
    return {
        **binding,
        "agent_id": expected_agent_id,
        "final_response_sha256": hashlib.sha256(
            final_response.encode("utf-8")
        ).hexdigest(),
        "read_agent_tool_call_id": str(read_call_id),
        "state": supported_task_state,
    }


def validate_supported_task_completion(
    expected_agent_id: str,
    supported_task_state: str,
) -> dict[str, str]:
    package_root = validate_sealed_tree(PACKAGE, None, require_read_only=True)
    agent_id = validate_review_bundle(package_root, expected_agent_id)
    host = validate_host_completion_events(
        _host_stream_events(),
        expected_agent_id,
        supported_task_state,
    )
    return {
        **host,
        "agent_id": agent_id,
        "package_tree_root_sha256": package_root,
        "status": "PASS_FRAMEWORK_L2_SUPPORTED_TASK_AUTHENTICATED",
    }


'''


HOSTILE_ADDITIONS = r'''
    host_agent = "task-agent-reviewer-0014"
    task_call = "call_layer16_package0014"
    final_response = "PACKAGE0014 REVIEW PASS"

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
        return {"host_call_id": "host-call-package0014", "inner": inner}

    host_events = [
        event(
            "tool.execution_start",
            {
                "toolCallId": task_call,
                "toolName": "task",
                "arguments": {
                    "name": layer16.REVIEW_TASK_NAME,
                    "agent_type": "general-purpose",
                    "mode": "sync",
                    "prompt": (
                        layer16.PACKAGE.relative_to(layer16.ROOT).as_posix()
                        + " "
                        + str((layer16.PACKAGE / "review_package.py").resolve())
                    ),
                },
            },
        ),
        event(
            "assistant.message",
            {"phase": "final_answer", "content": final_response},
            agent_id=host_agent,
            parent_tool=task_call,
        ),
        event("assistant.turn_end", {}, agent_id=host_agent, parent_tool=task_call),
        event(
            "tool.execution_complete",
            {
                "toolCallId": task_call,
                "success": True,
                "result": {"agent_id": host_agent, "response": final_response},
            },
            agent_id=host_agent,
        ),
        event(
            "tool.execution_start",
            {
                "toolCallId": "call_read_agent_package0014",
                "toolName": "read_agent",
                "arguments": {"agent_id": host_agent},
            },
        ),
        event(
            "tool.execution_complete",
            {
                "toolCallId": "call_read_agent_package0014",
                "success": True,
                "result": {"agent_id": host_agent, "response": final_response},
            },
        ),
    ]
    layer16.validate_host_completion_events(host_events, host_agent, "idle")

    def validate_events(value: list[dict[str, object]], agent_id: str = host_agent) -> None:
        layer16.validate_host_completion_events(value, agent_id, "idle")

    missing_final = copy.deepcopy(host_events)
    missing_final.pop(1)
    cases.append(
        expect_rejection("missing-final-response", lambda: validate_events(missing_final))
    )
    idle_without_result = copy.deepcopy(host_events)
    idle_without_result.pop()
    cases.append(
        expect_rejection("idle-without-result", lambda: validate_events(idle_without_result))
    )
    cases.append(
        expect_rejection(
            "forged-agent-id",
            lambda: validate_events(host_events, "task-agent-forged-0014"),
        )
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
                    host_events, host_agent, state
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
                relative_argv_log, root, agent
            ),
        )
    )
'''


def common_transform(name: str, text: str) -> str:
    text = text.replace("0013", "0014")
    text = text.replace("package 0013", "package 0014")
    text = text.replace("2f27533f74db", ENGINEER_MISSION_ID)
    text = text.replace("_v6", "_v7")
    text = text.replace("-v6", "-v7")
    return text


def transform_executor(text: str, directive_sha256: str) -> str:
    text = common_transform("execute_layer16.py", text)
    text = replace_section(
        text,
        f'ENGINEER_MISSION_ID = "{ENGINEER_MISSION_ID}"',
        "PRESERVED_FAILED_PACKAGES = {",
        REVIEW_CONSTANTS,
    )
    text = replace_section(
        text,
        "PRESERVED_FAILED_PACKAGES = {",
        "STALE_REVIEWER_IDENTITIES = {",
        PRESERVED,
    )
    text = replace_once(
        text,
        '''STALE_REVIEWER_IDENTITIES = {
    "ef520aec-144c-4d42-931a-4a3d6fd7e0ce",
    "398f2f5776d8",
}''',
        '''STALE_REVIEWER_IDENTITIES = {
    "ef520aec-144c-4d42-931a-4a3d6fd7e0ce",
    "398f2f5776d8",
    "191b6063-1350-4469-a165-8bd83c9e7a24",
    "2f27533f74db",
}''',
    )
    text = replace_once(
        text,
        'ACTIVE_DIRECTIVE_REVISION = "93588fc3cf524ca2abc6681a2ec461eb"',
        f'ACTIVE_DIRECTIVE_REVISION = "{DIRECTIVE_REVISION}"',
    )
    text = replace_once(
        text,
        '''ACTIVE_DIRECTIVE_SHA256 = (
    "edd0cea2f036404866878b8d4268e3391d73c76c7f11e892bf7ad1fbb066eb5e"
)''',
        f'''ACTIVE_DIRECTIVE_SHA256 = (
    "{directive_sha256}"
)''',
    )
    text = replace_once(
        text,
        '''        (PACKAGE / "review_package.py").relative_to(ROOT).as_posix(),''',
        '''        str((PACKAGE / "review_package.py").resolve()),''',
    )
    text = replace_once(
        text,
        '''            (PACKAGE / "validate_nonexecuting.py").relative_to(ROOT).as_posix(),''',
        '''            str((PACKAGE / "validate_nonexecuting.py").resolve()),''',
    )
    text = replace_once(
        text,
        '''            (PACKAGE / "hostile_controls.py").relative_to(ROOT).as_posix(),''',
        '''            str((PACKAGE / "hostile_controls.py").resolve()),''',
    )
    text = replace_once(
        text,
        '''                "completed_state": "completed",''',
        '''                "terminal_states": [
                    "completed",
                    "idle-with-durable-completed-final-turn",
                ],''',
    )
    text = replace_once(
        text,
        '''        "expected_completed_state": "completed",''',
        '''        "terminal_semantics": "COMPLETED_OR_IDLE_WITH_DURABLE_FINAL_TURN",''',
    )
    text = replace_once(
        text,
        '''            "supported-task-tool-agent-identity-and-completed-return",''',
        '''            "supported-task-tool-agent-identity-final-return-and-read-agent",''',
    )
    text = replace_once(
        text,
        '''            "reject-hand-written-registry-missing-launch-or-completion",''',
        '''            "reject-missing-launch-final-response-or-read-agent-result",''',
    )
    text = replace_once(
        text,
        '''        and {item.name for item in REVIEW_NAMESPACE.iterdir()} == {REVIEW.name},''',
        '''        and {item.name for item in REVIEW_NAMESPACE.iterdir()}
        == {REVIEW.name, REVIEW_LOG.name, REVIEW_RECEIPT.name},''',
    )
    text = replace_section(
        text,
        "def validate_supported_task_completion(",
        "def _directive_binding(",
        HOST_PROVENANCE,
    )
    text = replace_once(
        text,
        '"failed_lifecycle_packages": "PASS_0009_0011_0012_PRESERVED",',
        '"failed_lifecycle_packages": "PASS_0009_0011_0012_0013_PRESERVED",',
    )
    require(".argus_subagents" not in text, "package0014 retained registry dependency")
    return text


def transform_review(text: str) -> str:
    text = common_transform("review_package.py", text)
    text = replace_once(
        text,
        '''    layer16.require(
        layer16._valid_agent_id(args.agent_id)''',
        '''    layer16.validate_supported_task_launch(args.agent_id)
    layer16.require(
        layer16._valid_agent_id(args.agent_id)''',
    )
    text = replace_once(
        text,
        '''            (layer16.PACKAGE / "validate_nonexecuting.py")
            .relative_to(layer16.ROOT)
            .as_posix(),''',
        '''            str((layer16.PACKAGE / "validate_nonexecuting.py").resolve()),''',
    )
    text = replace_once(
        text,
        '''            (layer16.PACKAGE / "hostile_controls.py")
            .relative_to(layer16.ROOT)
            .as_posix(),''',
        '''            str((layer16.PACKAGE / "hostile_controls.py").resolve()),''',
    )
    require(".argus_subagents" not in text, "review writer retained registry dependency")
    return text


def transform_hostile(text: str) -> str:
    text = common_transform("hostile_controls.py", text)
    text = replace_once(
        text,
        '''    after = layer16.live_snapshot()''',
        HOSTILE_ADDITIONS + '''\n    after = layer16.live_snapshot()''',
    )
    require(".argus_subagents" not in text, "hostile controls retained registry dependency")
    return text


def transform_validator(text: str) -> str:
    text = common_transform("validate_review.py", text)
    return text


def transform_seal(text: str) -> str:
    return common_transform("seal_package.py", text)


def transform_nonexecuting(text: str) -> str:
    return common_transform("validate_nonexecuting.py", text)


def transformed_sources(directive_sha256: str) -> dict[str, str]:
    transforms = {
        "execute_layer16.py": lambda value: transform_executor(
            value, directive_sha256
        ),
        "hostile_controls.py": transform_hostile,
        "review_package.py": transform_review,
        "seal_package.py": transform_seal,
        "validate_nonexecuting.py": transform_nonexecuting,
        "validate_review.py": transform_validator,
    }
    result = {
        name: transforms[name]((SOURCE / name).read_text(encoding="utf-8"))
        for name in sorted(SOURCE_MEMBERS)
    }
    for name, text in result.items():
        compile(text, str(TARGET / name), "exec")
        require("0013-review" not in text, f"stale package review path in {name}")
    return result


def main() -> int:
    directive_sha256 = validate_source()
    sources = transformed_sources(directive_sha256)
    os.mkdir(TARGET, mode=0o700)
    for name, text in sources.items():
        write_exclusive(TARGET / name, text.encode("utf-8"), mode=0o444)
    result = subprocess.run(
        ["/home/argustest/miniconda3/bin/python3.13", "-B", str(TARGET / "seal_package.py")],
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
        "package0014 seal failed: " + result.stderr.decode("utf-8", "replace"),
    )
    sys.stdout.buffer.write(result.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
