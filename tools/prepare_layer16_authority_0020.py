#!/usr/bin/env python3
"""Create and seal layer-16 authority package 0020 exclusively."""

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
SOURCE = ROOT / "reports/ace2-layer16-runtime-pass-authority-0019"
SOURCE_REVIEW = ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0019"
TARGET = ROOT / "reports/ace2-layer16-runtime-pass-authority-0020"
DIRECTIVE = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "active_manager_directive.json"
)
SOURCE_ROOT = "f0865f8bb05f3a62740a4a791ca35bcf8d8e53ce892615cc2588fa3222566a61"
SOURCE_REVIEW_HASHES = {
    "review-log.json": "d6bc5ce863deaf85f0ed74315f8a4e0e1bc70116e088a7bd89dad6f5b2984437",
    "reviewer-adjudication.json": "0b3c7b5dace04e4d1c13711e01b9a8025f28a6633617410dc30302045ece0626",
    "reviewer-receipt.json": "986b04f9714fbf56e74f6d66d1d9eb8b39cc490332fee88028d014db0497498d",
}
ENGINEER_MISSION_ID = "272d6b89076b"
DIRECTIVE_REVISION = "17bab73cafb1456e815825534269d32a"
DIRECTIVE_SHA256 = "b9074a0d3145a0148c2f8651941d8bf134812e229fcc675ba67e9e9621e37b2b"
DIRECTIVE_OBJECTIVE_SHA256 = (
    "43fd081a2ec0196f5100f18980c9c58017888abf17b7cfcc77444e31ebd8ad39"
)
SOURCE_MEMBERS = {
    "execute_layer16.py",
    "hostile_controls.py",
    "review_package.py",
    "seal_package.py",
    "validate_nonexecuting.py",
    "validate_review.py",
}
SEALED_MEMBERS = SOURCE_MEMBERS | {
    "acceptance-contract.json",
    "authority.json",
    "failed-predecessor-provenance.json",
    "hostile-controls.json",
    "public-task-aggregate-fixtures.json",
    "review-request.json",
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


def validate_sealed_source() -> None:
    require(
        SOURCE.is_dir()
        and not SOURCE.is_symlink()
        and stat.S_IMODE(SOURCE.stat().st_mode) == 0o555,
        "sealed package0019 directory changed",
    )
    records: dict[str, str] = {}
    for line in (SOURCE / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name not in records,
            "sealed package0019 checksum manifest changed",
        )
        records[name] = digest
    observed = {
        item.relative_to(SOURCE).as_posix()
        for item in SOURCE.rglob("*")
        if item.is_file() and item.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    }
    require(
        observed == set(records)
        and sha256_file(SOURCE / "SHA256SUMS") == SOURCE_ROOT,
        "sealed package0019 root or members changed",
    )
    for name, digest in records.items():
        path = SOURCE / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"sealed package0019 member changed: {path}",
        )
    require(
        (SOURCE / "TREE_ROOT.sha256").read_text(encoding="ascii").split()
        == [SOURCE_ROOT, "SHA256SUMS"],
        "sealed package0019 root sidecar changed",
    )
    require(
        SOURCE_REVIEW.is_dir()
        and not SOURCE_REVIEW.is_symlink()
        and stat.S_IMODE(SOURCE_REVIEW.stat().st_mode) == 0o555
        and {item.name for item in SOURCE_REVIEW.iterdir()}
        == set(SOURCE_REVIEW_HASHES),
        "package0019 PASS namespace changed",
    )
    for name, digest in SOURCE_REVIEW_HASHES.items():
        path = SOURCE_REVIEW / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"package0019 PASS artifact changed: {path}",
        )


def validate_inputs() -> None:
    validate_sealed_source()
    require(DIRECTIVE.is_file() and not DIRECTIVE.is_symlink(), "Manager V9 absent")
    directive = json.loads(DIRECTIVE.read_text(encoding="utf-8"))
    text = directive.get("text")
    require(
        isinstance(directive, dict)
        and directive.get("revision") == DIRECTIVE_REVISION
        and directive.get("objective_sha256") == DIRECTIVE_OBJECTIVE_SHA256
        and isinstance(text, str)
        and "MANAGER GRANT LAYER16 V9" in text
        and "package0020" in text
        and "layer16-package0020-reviewer" in text
        and "agent_type=`general-purpose`" in text
        and "mode=`sync`" in text
        and "exactly once" in text
        and "Zero workload" in text,
        "active Manager V9 directive changed",
    )
    require(sha256_file(DIRECTIVE) == DIRECTIVE_SHA256, "Manager V9 bytes changed")
    for path in (
        TARGET,
        ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0020",
        ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0020",
        ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0020",
    ):
        require(not os.path.lexists(path), f"package0020 target preexists: {path}")


PUBLIC_TASK_BLOCK = r'''def _reject_duplicate_json_keys(
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
            events.append({"inner": inner})
    return events


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("inner", {}).get("data")
    return data if isinstance(data, dict) else {}


REVIEW_TASK_DESCRIPTION = "Adjudicate sealed package0020"


def expected_review_prompt(package_root: str) -> str:
    return (
        "Act as the sole source-disjoint external Reviewer for the sealed package "
        f"{PACKAGE}. Work in {ROOT}. Inspect the sealed source and review-request.json, "
        "preserve package0019 and its PASS artifacts and all earlier packages byte-for-byte, "
        "and create no Manager grant, consumption state, authority, RTL execution, reference "
        "execution, model execution, tokens, PPA, U280, or Stage2 work. Use your exact "
        "Host-assigned task agent_id as <agent-id>; do not use the task name, description, "
        "creator identity, or a fabricated identifier. Run exactly: "
        "/home/argustest/miniconda3/bin/python3.13 -B "
        f"{(PACKAGE / 'review_package.py').resolve()} --package-root {package_root} "
        f"--agent-id <agent-id> --task-name {REVIEW_TASK_NAME} "
        f"--engineer-mission-id {ENGINEER_MISSION_ID}. The package-native script "
        "exclusively creates the adjudication, immutable review log, and receipt and "
        "validates the complete sealed public-contract fixture suite. Do not hand-create, "
        "edit, replace, retry, repair, or reseal any package or review artifact. If the "
        "exact command succeeds, return only its exact canonical one-line JSON stdout as "
        "your final response, with no Markdown or additional text. If it fails, return a "
        "concise failure and do not weaken or retry the checks."
    )


def expected_task_arguments(package_root: str) -> dict[str, str]:
    return {
        "agent_type": "general-purpose",
        "description": REVIEW_TASK_DESCRIPTION,
        "mode": "sync",
        "name": REVIEW_TASK_NAME,
        "prompt": expected_review_prompt(package_root),
    }


def expected_final_payload(
    package_root: str,
    agent_id: str,
    adjudication_sha256: str,
    review_log_sha256: str,
    receipt_sha256: str,
) -> dict[str, str]:
    return {
        "adjudication_sha256": adjudication_sha256,
        "agent_id": agent_id,
        "package_tree_root_sha256": package_root,
        "receipt_sha256": receipt_sha256,
        "review_log_sha256": review_log_sha256,
        "status": "PASS_ARTIFACTS_CREATED_AWAITING_SUPPORTED_TASK_RETURN",
    }


def _task_tool_name(value: object) -> bool:
    return value in {"task", "functions.task"}


def _task_requests(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for event in events:
        if event.get("inner", {}).get("type") != "assistant.message":
            continue
        requests = _event_data(event).get("toolRequests")
        if not isinstance(requests, list):
            continue
        for request in requests:
            arguments = request.get("arguments") if isinstance(request, dict) else None
            if (
                isinstance(arguments, dict)
                and arguments.get("name") == REVIEW_TASK_NAME
                and _task_tool_name(request.get("name"))
            ):
                matches.append(request)
    return matches


def _task_starts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _event_data(event)
        for event in events
        if event.get("inner", {}).get("type") == "tool.execution_start"
        and _task_tool_name(_event_data(event).get("toolName"))
        and isinstance(_event_data(event).get("arguments"), dict)
        and _event_data(event)["arguments"].get("name") == REVIEW_TASK_NAME
    ]


def _launch_records(
    events: list[dict[str, Any]],
    package_root: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    requests = _task_requests(events)
    starts = _task_starts(events)
    require(len(requests) == 1, "exactly one public functions.task request required")
    require(len(starts) == 1, "exactly one public functions.task start required")
    request = requests[0]
    start = starts[0]
    expected_arguments = expected_task_arguments(package_root)
    require(
        set(request) >= {"arguments", "name", "toolCallId"}
        and _task_tool_name(request.get("name"))
        and request.get("arguments") == expected_arguments
        and set(start) >= {"arguments", "toolCallId", "toolName"}
        and _task_tool_name(start.get("toolName"))
        and start.get("arguments") == expected_arguments
        and isinstance(request.get("toolCallId"), str)
        and bool(request["toolCallId"])
        and request.get("toolCallId") == start.get("toolCallId"),
        "public task request/start arguments or toolCallId mismatch",
    )
    return request, start


def validate_supported_task_launch(expected_agent_id: str) -> dict[str, str]:
    require(
        _valid_agent_id(expected_agent_id)
        and expected_agent_id != ENGINEER_MISSION_ID
        and expected_agent_id not in STALE_REVIEWER_IDENTITIES,
        "fresh source-disjoint supported-task Reviewer identity required",
    )
    package_root = validate_sealed_tree(PACKAGE, None, require_read_only=True)
    request, _start = _launch_records(_host_stream_events(), package_root)
    return {"task_tool_call_id": request["toolCallId"]}


def _normalized_task_aggregate(
    events: list[dict[str, Any]],
    package_root: str,
    public_telemetry: dict[str, Any],
) -> dict[str, Any]:
    request, start = _launch_records(events, package_root)
    tool_call_id = request["toolCallId"]
    completions = [
        _event_data(event)
        for event in events
        if event.get("inner", {}).get("type") == "tool.execution_complete"
        and _event_data(event).get("toolCallId") == tool_call_id
    ]
    require(len(completions) == 1, "exactly one matching public task completion required")
    complete = completions[0]
    tool_telemetry = complete.get("toolTelemetry")
    properties = (
        tool_telemetry.get("properties") if isinstance(tool_telemetry, dict) else None
    )
    restricted = (
        tool_telemetry.get("restrictedProperties")
        if isinstance(tool_telemetry, dict)
        else None
    )
    require(
        isinstance(properties, dict)
        and isinstance(restricted, dict)
        and restricted.get("agent_id") == public_telemetry.get("agent_id")
        and properties.get("agent_type") == public_telemetry.get("agent_type"),
        "public completion telemetry identity or type mismatch",
    )
    return {
        "request": [
            {
                "arguments": request["arguments"],
                "name": "functions.task",
                "toolCallId": tool_call_id,
            }
        ],
        "start": [
            {
                "arguments": start["arguments"],
                "toolCallId": start["toolCallId"],
                "toolName": "functions.task",
            }
        ],
        "complete": [
            {
                "result": complete.get("result"),
                "success": complete.get("success"),
                "telemetry": public_telemetry,
                "toolCallId": complete.get("toolCallId"),
            }
        ],
    }


def validate_public_task_aggregate(
    aggregate: object,
    package_root: str,
    expected_payload: dict[str, str],
) -> dict[str, Any]:
    require(
        isinstance(aggregate, dict)
        and set(aggregate) == {"request", "start", "complete"},
        "exact public task aggregate required",
    )
    requests = aggregate["request"]
    starts = aggregate["start"]
    completions = aggregate["complete"]
    require(
        isinstance(requests, list)
        and isinstance(starts, list)
        and isinstance(completions, list)
        and len(requests) == len(starts) == len(completions) == 1,
        "one request, start, and completion required",
    )
    request, start, complete = requests[0], starts[0], completions[0]
    require(
        isinstance(request, dict)
        and set(request) == {"arguments", "name", "toolCallId"}
        and request.get("name") == "functions.task"
        and request.get("arguments") == expected_task_arguments(package_root)
        and isinstance(start, dict)
        and set(start) == {"arguments", "toolCallId", "toolName"}
        and start.get("toolName") == "functions.task"
        and start.get("arguments") == expected_task_arguments(package_root),
        "exact public task request and start arguments required",
    )
    tool_call_id = request.get("toolCallId")
    require(
        isinstance(tool_call_id, str)
        and bool(tool_call_id)
        and start.get("toolCallId") == tool_call_id
        and isinstance(complete, dict)
        and set(complete) == {"result", "success", "telemetry", "toolCallId"}
        and complete.get("toolCallId") == tool_call_id,
        "one unique matching public task toolCallId required",
    )
    telemetry = complete.get("telemetry")
    require(
        isinstance(telemetry, dict)
        and set(telemetry) == {"agent_id", "agent_type", "status", "total_turns"}
        and _valid_agent_id(telemetry.get("agent_id"))
        and telemetry.get("agent_id") != ENGINEER_MISSION_ID
        and telemetry.get("agent_id") not in STALE_REVIEWER_IDENTITIES
        and telemetry.get("agent_type") == "general-purpose"
        and telemetry.get("status") == "completed"
        and isinstance(telemetry.get("total_turns"), int)
        and not isinstance(telemetry["total_turns"], bool)
        and telemetry["total_turns"] >= 1,
        "fresh completed public task telemetry required",
    )
    require(
        complete.get("success") is True,
        "failed, cancelled, or incomplete public task completion rejected",
    )
    require(
        expected_payload
        == expected_final_payload(
            package_root,
            telemetry["agent_id"],
            expected_payload.get("adjudication_sha256", ""),
            expected_payload.get("review_log_sha256", ""),
            expected_payload.get("receipt_sha256", ""),
        ),
        "expected artifact payload is malformed",
    )
    result = complete.get("result")
    expected_final = canonical_bytes(expected_payload).decode("ascii").rstrip("\n")
    require(
        isinstance(result, dict)
        and set(result) == {"content", "detailedContent"}
        and result.get("content") == expected_final
        and result.get("detailedContent") == expected_final,
        "exact public task final PASS payload required",
    )
    return {
        "agent_id": telemetry["agent_id"],
        "final_response_sha256": hashlib.sha256(
            expected_final.encode("ascii")
        ).hexdigest(),
        "state": telemetry["status"],
        "task_tool_call_id": tool_call_id,
        "total_turns": telemetry["total_turns"],
    }


def _fixture_rejection(
    name: str,
    operation: Callable[[], None],
) -> dict[str, str]:
    try:
        operation()
    except Layer16Error:
        return {"name": name, "result": "PASS_REJECTED_WITHOUT_CONSUMPTION"}
    raise Layer16Error(f"public task fixture unexpectedly accepted: {name}")


def run_public_task_aggregate_fixtures() -> dict[str, Any]:
    import copy
    from typing import Callable

    root = "a" * 64
    agent_id = "00000000-0000-4000-8000-000000002020"
    payload = expected_final_payload(root, agent_id, "b" * 64, "c" * 64, "d" * 64)
    final = canonical_bytes(payload).decode("ascii").rstrip("\n")
    tool_call_id = "call_layer16_package0020_public_contract"
    arguments = expected_task_arguments(root)
    aggregate = {
        "request": [
            {
                "arguments": arguments,
                "name": "functions.task",
                "toolCallId": tool_call_id,
            }
        ],
        "start": [
            {
                "arguments": arguments,
                "toolCallId": tool_call_id,
                "toolName": "functions.task",
            }
        ],
        "complete": [
            {
                "result": {"content": final, "detailedContent": final},
                "success": True,
                "telemetry": {
                    "agent_id": agent_id,
                    "agent_type": "general-purpose",
                    "status": "completed",
                    "total_turns": 1,
                },
                "toolCallId": tool_call_id,
            }
        ],
    }
    validate_public_task_aggregate(aggregate, root, payload)
    cases = [{"name": "positive-exact-public-contract", "result": "PASS_ACCEPTED"}]

    def reject(name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        candidate = copy.deepcopy(aggregate)
        mutate(candidate)
        cases.append(
            _fixture_rejection(
                name,
                lambda: validate_public_task_aggregate(candidate, root, payload),
            )
        )

    reject("duplicate-request", lambda value: value["request"].append(value["request"][0]))
    reject("duplicate-start", lambda value: value["start"].append(value["start"][0]))
    reject("duplicate-completion", lambda value: value["complete"].append(value["complete"][0]))
    reject("missing-request", lambda value: value.__setitem__("request", []))
    reject("missing-start", lambda value: value.__setitem__("start", []))
    reject("missing-completion", lambda value: value.__setitem__("complete", []))
    reject("mismatched-start-toolCallId", lambda value: value["start"][0].__setitem__("toolCallId", "call_mismatch"))
    reject("mismatched-complete-toolCallId", lambda value: value["complete"][0].__setitem__("toolCallId", "call_mismatch"))
    reject("mismatched-name", lambda value: value["request"][0]["arguments"].__setitem__("name", "wrong-reviewer"))
    reject("mismatched-agent-type", lambda value: value["start"][0]["arguments"].__setitem__("agent_type", "explore"))
    reject("mismatched-mode", lambda value: value["request"][0]["arguments"].__setitem__("mode", "background"))
    reject("mismatched-description", lambda value: value["start"][0]["arguments"].__setitem__("description", "altered"))
    reject("mismatched-prompt", lambda value: value["request"][0]["arguments"].__setitem__("prompt", "altered"))
    reject("failed-completion", lambda value: value["complete"][0].__setitem__("success", False))
    reject("missing-result", lambda value: value["complete"][0].__setitem__("result", None))
    reject("missing-telemetry", lambda value: value["complete"][0].__setitem__("telemetry", None))
    reject("failed-telemetry-status", lambda value: value["complete"][0]["telemetry"].__setitem__("status", "failed"))
    reject("zero-total-turns", lambda value: value["complete"][0]["telemetry"].__setitem__("total_turns", 0))
    reject("engineer-identity-reuse", lambda value: value["complete"][0]["telemetry"].__setitem__("agent_id", ENGINEER_MISSION_ID))
    reject("mismatched-telemetry-type", lambda value: value["complete"][0]["telemetry"].__setitem__("agent_type", "task"))
    reject("payload-root-forgery", lambda value: value["complete"][0]["result"].__setitem__("content", final.replace("a" * 64, "e" * 64)))
    reject("payload-adjudication-hash-forgery", lambda value: value["complete"][0]["result"].__setitem__("content", final.replace("b" * 64, "e" * 64)))
    reject("payload-review-log-hash-forgery", lambda value: value["complete"][0]["result"].__setitem__("content", final.replace("c" * 64, "e" * 64)))
    reject("payload-receipt-hash-forgery", lambda value: value["complete"][0]["result"].__setitem__("content", final.replace("d" * 64, "e" * 64)))
    reject("payload-status-forgery", lambda value: value["complete"][0]["result"].__setitem__("content", final.replace("PASS_ARTIFACTS_CREATED_AWAITING_SUPPORTED_TASK_RETURN", "PASS")))
    reject("payload-agent-forgery", lambda value: value["complete"][0]["result"].__setitem__("content", final.replace(agent_id, "00000000-0000-4000-8000-000000002021")))
    reject("detailed-payload-forgery", lambda value: value["complete"][0]["result"].__setitem__("detailedContent", final + " forged"))
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **ZERO_WORKLOAD,
        },
        "cases": cases,
        "execution_performed": False,
        "positive_aggregate": aggregate,
        "schema": "ace2-supported-task-public-aggregate-fixtures-v1",
        "status": "PASS_PUBLIC_TASK_AGGREGATE_FIXTURES",
    }


def validate_supported_task_completion(
    expected_agent_id: str,
    supported_task_status: str,
    total_turns: int,
) -> dict[str, Any]:
    package_root = validate_sealed_tree(PACKAGE, None, require_read_only=True)
    agent_id = validate_review_bundle(package_root, expected_agent_id)
    payload = expected_final_payload(
        package_root,
        expected_agent_id,
        sha256_file(REVIEW),
        sha256_file(REVIEW_LOG),
        sha256_file(REVIEW_RECEIPT),
    )
    aggregate = _normalized_task_aggregate(
        _host_stream_events(),
        package_root,
        {
            "agent_id": expected_agent_id,
            "agent_type": "general-purpose",
            "status": supported_task_status,
            "total_turns": total_turns,
        },
    )
    host = validate_public_task_aggregate(aggregate, package_root, payload)
    return {
        **host,
        "agent_id": agent_id,
        "package_tree_root_sha256": package_root,
        "status": "PASS_FRAMEWORK_L2_PUBLIC_TASK_CONTRACT_AUTHENTICATED",
    }


'''


HOSTILE_TAIL = r'''    public_fixtures = layer16.run_public_task_aggregate_fixtures()
    cases.extend(public_fixtures["cases"])
    after = layer16.live_snapshot()
    layer16.require(before == after, "hostile controls changed the live frontier")
    layer16.require(
        before["consumption_exists"] is False
        and before["grant_namespace_exists"] is False
        and before["layer16_pending"] is False
        and before["layer16_published"] is False
        and before["layer16_work"] is False
        and before["review_log_exists"] is False
        and before["review_namespace_exists"] is False
        and before["reviewer_receipt_exists"] is False,
        "hostile controls observed layer-16 execution or review residue",
    )
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **layer16.ZERO_WORKLOAD,
        },
        "cases": cases,
        "execution_performed": False,
        "live_snapshot_sha256_after": layer16.hashlib.sha256(
            layer16.canonical_bytes(after)
        ).hexdigest(),
        "live_snapshot_sha256_before": layer16.hashlib.sha256(
            layer16.canonical_bytes(before)
        ).hexdigest(),
        "public_task_aggregate_fixtures": public_fixtures,
        "schema": "ace2-stage1-layer16-hostile-controls-v12",
        "status": "PASS_ZERO_WORKLOAD_ZERO_CONSUMPTION",
    }


def main() -> int:
    sys.stdout.buffer.write(layer16.canonical_bytes(run_controls()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def transform_executor(text: str) -> str:
    text = text.replace("0019", "0020").replace("v11", "v12")
    text = replace_once(
        text,
        'ENGINEER_MISSION_ID = "4341527f33f4"',
        f'ENGINEER_MISSION_ID = "{ENGINEER_MISSION_ID}"',
    )
    text = replace_once(
        text,
        'ACTIVE_DIRECTIVE_REVISION = "999497e7567e4d1a9a751775239f1cd0"',
        f'ACTIVE_DIRECTIVE_REVISION = "{DIRECTIVE_REVISION}"',
    )
    text = replace_once(
        text,
        '"059f18166aaddaff10703fa59fa1766d33b5cd18c24cbdacc00f2e96b55d7290"',
        f'"{DIRECTIVE_SHA256}"',
    )
    predecessor = '''    "0018": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0018",
        "f2419fb98c58255fd2d86a1142f497e42d82e295efda2851b633efc9778402c8",
        "HOST_SUPPORTED_TASK_NAME_ARGUMENT_MISMATCH",
    ),
'''
    text = replace_once(
        text,
        predecessor,
        predecessor
        + '''    "0019": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0019",
        "f0865f8bb05f3a62740a4a791ca35bcf8d8e53ce892615cc2588fa3222566a61",
        "FRAMEWORK_INTERNAL_EVENT_TOPOLOGY_DEPENDENCY",
    ),
''',
    )
    text = replace_once(
        text,
        '    "5bdef423-848d-4f36-93e2-a625cc4dd444",\n}',
        '    "5bdef423-848d-4f36-93e2-a625cc4dd444",\n'
        '    "d0df6331-ca6d-469d-bbb1-aab9e9a06f69",\n}',
    )
    text = replace_once(
        text,
        "FAILED_CONSTRUCTION_0016 = (\n",
        '''PRESERVED_REVIEW_0019 = ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0019"
PRESERVED_REVIEW_0019_HASHES = {
    "review-log.json": "d6bc5ce863deaf85f0ed74315f8a4e0e1bc70116e088a7bd89dad6f5b2984437",
    "reviewer-adjudication.json": "0b3c7b5dace04e4d1c13711e01b9a8025f28a6633617410dc30302045ece0626",
    "reviewer-receipt.json": "986b04f9714fbf56e74f6d66d1d9eb8b39cc490332fee88028d014db0497498d",
}

FAILED_CONSTRUCTION_0016 = (
''',
    )
    text = replace_once(
        text,
        '\n\ndef _failed_construction_binding() -> dict[str, Any]:\n',
        '''

def validate_preserved_review_0019() -> None:
    require(
        PRESERVED_REVIEW_0019.is_dir()
        and not PRESERVED_REVIEW_0019.is_symlink()
        and stat.S_IMODE(PRESERVED_REVIEW_0019.stat().st_mode) == 0o555
        and {item.name for item in PRESERVED_REVIEW_0019.iterdir()}
        == set(PRESERVED_REVIEW_0019_HASHES),
        "package0019 PASS namespace changed",
    )
    for name, digest in PRESERVED_REVIEW_0019_HASHES.items():
        path = PRESERVED_REVIEW_0019 / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"package0019 PASS artifact changed: {path}",
        )


def _failed_construction_binding() -> dict[str, Any]:
''',
    )
    text = replace_once(
        text,
        '        "regression": (\n'
        '            "HOST_PARENT_ID_JOINT_RECORD_AND_OPTIONAL_MODEL_FRAMING"\n'
        '        ),',
        '''        "package0019_review": {
            "failure_taxonomy_class": "FRAMEWORK_INTERNAL_EVENT_TOPOLOGY_DEPENDENCY",
            "member_sha256": PRESERVED_REVIEW_0019_HASHES,
            "namespace_mode": "0555",
            "path": PRESERVED_REVIEW_0019.relative_to(ROOT).as_posix(),
            "preserved_unchanged": True,
        },
        "regression": "SUPPORTED_PUBLIC_TASK_AGGREGATE_ONLY",''',
    )
    text = replace_once(
        text,
        '            "supported-task-tool-agent-identity-final-return-and-read-agent",\n'
        '            "reviewer-agent-differs-from-engineer-package-creator",\n'
        '            "exact-review-root-argv-source-hashes-and-immutable-command-logs",\n'
        '            "reject-missing-launch-final-response-or-read-agent-result",\n'
        '            "reject-stale-mismatched-engineer-or-predecessor-identities",\n'
        '            "reject-altered-root-argv-and-joint-adjudication-receipt-forgery",',
        '            "supported-task-public-request-start-complete-result-and-telemetry",\n'
        '            "reviewer-agent-differs-from-engineer-package-creator",\n'
        '            "exact-review-root-argv-source-hashes-and-immutable-command-logs",\n'
        '            "reject-duplicate-missing-mismatched-or-failed-public-task-records",\n'
        '            "reject-engineer-reuse-and-identity-hash-or-payload-forgery",\n'
        '            "ignore-internal-turn-end-subagent-topology-and-optional-fields",',
    )
    text = replace_once(
        text,
        '        "completion_authentication": "SUPPORTED_TASK_TOOL_RETURN_REQUIRED",\n'
        '        "terminal_semantics": "COMPLETED_OR_IDLE_WITH_DURABLE_FINAL_TURN",\n'
        '        "identity_source": "SUPPORTED_TASK_TOOL_ASSIGNED_AGENT_ID",',
        '        "completion_authentication": "SUPPORTED_PUBLIC_TASK_AGGREGATE_REQUIRED",\n'
        '        "terminal_semantics": "PUBLIC_COMPLETION_SUCCESS_AND_COMPLETED_TELEMETRY",\n'
        '        "identity_source": "SUPPORTED_TASK_PUBLIC_TELEMETRY_AGENT_ID",',
    )
    old_completion = '''            "completion_authentication": {
                "accepted_identity_source": "SUPPORTED_TASK_TOOL_RETURN_ONLY",
                "terminal_states": [
                    "completed",
                    "idle-with-durable-completed-final-turn",
                ],
                "hand_written_registry_accepted": False,
                "required_agent_type": "general-purpose",
                "required_task_name": REVIEW_TASK_NAME,
                "tool": "functions.task",
            },'''
    new_completion = '''            "completion_authentication": {
                "authoritative_fields": [
                    "request.arguments",
                    "request.toolCallId",
                    "start.arguments",
                    "start.toolCallId",
                    "complete.success",
                    "complete.toolCallId",
                    "result.content",
                    "result.detailedContent",
                    "telemetry.agent_id",
                    "telemetry.agent_type",
                    "telemetry.status",
                    "telemetry.total_turns",
                ],
                "hand_written_registry_accepted": False,
                "ignored_internal_fields": [
                    "assistant.turn_end",
                    "subagent.completed",
                    "parentId",
                    "optional_internal_fields",
                ],
                "required_agent_type": "general-purpose",
                "required_task_name": REVIEW_TASK_NAME,
                "required_task_status": "completed",
                "tool": "functions.task",
            },'''
    text = replace_once(text, old_completion, new_completion)
    start = text.index("def _reject_duplicate_json_keys(")
    end = text.index("def _directive_binding(", start)
    text = text[:start] + PUBLIC_TASK_BLOCK + text[end:]
    text = replace_once(
        text,
        "def _validate_bound_evidence(*, current_directive_required: bool) -> None:\n"
        "    validate_failed_construction_0016()\n",
        "def _validate_bound_evidence(*, current_directive_required: bool) -> None:\n"
        "    validate_failed_construction_0016()\n"
        "    validate_preserved_review_0019()\n",
    )
    live_marker = '''            *(
                item
                for package, _tree_root, _failure_class
                in PRESERVED_FAILED_PACKAGES.values()
                for item in (package / "SHA256SUMS", package / "TREE_ROOT.sha256")
            ),
'''
    text = replace_once(
        text,
        live_marker,
        live_marker
        + '''            *(
                PRESERVED_REVIEW_0019 / name
                for name in sorted(PRESERVED_REVIEW_0019_HASHES)
            ),
''',
    )
    require(
        "assistant.turn_end" not in text[start : start + len(PUBLIC_TASK_BLOCK)],
        "public task validator retained internal turn-end authority",
    )
    return text


def transform_hostile(text: str) -> str:
    text = text.replace("0019", "0020").replace("v11", "v12")
    marker = "    host_agent ="
    require(text.count(marker) == 1, "hostile Host fixture marker changed")
    prefix = text.split(marker, 1)[0]
    prefix = replace_once(
        prefix,
        'lambda: layer16.validate_supported_task_completion(agent, "running"),',
        'lambda: layer16.validate_supported_task_completion(agent, "running", 0),',
    )
    return prefix + HOSTILE_TAIL


def transform_seal(text: str) -> str:
    text = text.replace("0019", "0020").replace("v11", "v12")
    text = replace_once(text, "import os\nimport sys\n", "import os\nimport stat\nimport sys\n")
    text = replace_once(
        text,
        "def write_json(path: Path, value: Any) -> None:\n"
        "    write_exclusive(path, layer16.canonical_bytes(value))\n",
        '''def write_json(path: Path, value: Any) -> None:
    data = layer16.canonical_bytes(value)
    if os.path.lexists(path):
        layer16.require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and path.read_bytes() == data,
            f"pre-seal create-exclusive record changed: {path}",
        )
        return
    write_exclusive(path, data)
''',
    )
    start = text.index("    controls = _load(")
    end = text.index("    members = sorted(", start)
    replacement = '''    controls = _load(
        "hostile_controls_layer16_seal",
        PACKAGE / "hostile_controls.py",
    ).run_controls()
    fixtures = controls.pop("public_task_aggregate_fixtures")
    layer16.require(
        fixtures.get("status") == "PASS_PUBLIC_TASK_AGGREGATE_FIXTURES"
        and all(
            case.get("result")
            in {"PASS_ACCEPTED", "PASS_REJECTED_WITHOUT_CONSUMPTION"}
            for case in fixtures.get("cases", [])
        ),
        "public task aggregate fixtures failed",
    )
    write_json(PACKAGE / "public-task-aggregate-fixtures.json", fixtures)
    controls["public_task_aggregate_fixtures"] = {
        "path": "public-task-aggregate-fixtures.json",
        "sha256": sha256_file(PACKAGE / "public-task-aggregate-fixtures.json"),
    }
    write_json(PACKAGE / "hostile-controls.json", controls)
'''
    text = text[:start] + replacement + text[end:]
    text = replace_once(
        text,
        '        "hostile-controls.json",\n        "review-request.json",',
        '        "hostile-controls.json",\n'
        '        "public-task-aggregate-fixtures.json",\n'
        '        "review-request.json",',
    )
    return text


def transform_validate_review(text: str) -> str:
    text = text.replace("0019", "0020").replace("v11", "v12")
    text = replace_once(
        text,
        '    parser.add_argument("--supported-task-state", required=True)\n'
        "    args = parser.parse_args()\n",
        '    parser.add_argument("--supported-task-status", required=True)\n'
        '    parser.add_argument("--total-turns", required=True, type=int)\n'
        "    args = parser.parse_args()\n",
    )
    text = replace_once(
        text,
        "        args.supported_task_state,\n"
        "    )",
        "        args.supported_task_status,\n"
        "        args.total_turns,\n"
        "    )",
    )
    return text


def transformed_sources() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in sorted(SOURCE_MEMBERS):
        source = (SOURCE / name).read_text(encoding="utf-8")
        if name == "execute_layer16.py":
            transformed = transform_executor(source)
        elif name == "hostile_controls.py":
            transformed = transform_hostile(source)
        elif name == "seal_package.py":
            transformed = transform_seal(source)
        elif name == "validate_review.py":
            transformed = transform_validate_review(source)
        else:
            transformed = source.replace("0019", "0020").replace("v11", "v12")
        compile(transformed, str(TARGET / name), "exec")
        require("layer16-package0019-reviewer" not in transformed, f"stale task token in {name}")
        result[name] = transformed
    require(
        'REVIEW_TASK_NAME = "layer16-package0020-reviewer"'
        in result["execute_layer16.py"],
        "exact package0020 review task token absent",
    )
    return result


def main() -> int:
    validate_inputs()
    sources = transformed_sources()
    os.mkdir(TARGET, mode=0o700)
    for name, text in sources.items():
        write_exclusive(TARGET / name, text.encode("utf-8"))
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
        "package0020 seal failed: " + result.stderr.decode("utf-8", "replace"),
    )
    require(
        TARGET.is_dir()
        and stat.S_IMODE(TARGET.stat().st_mode) == 0o555
        and {
            item.relative_to(TARGET).as_posix()
            for item in TARGET.rglob("*")
            if item.is_file() and item.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
        }
        == SEALED_MEMBERS,
        "sealed package0020 member set changed",
    )
    validate_sealed_source()
    sys.stdout.buffer.write(result.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
