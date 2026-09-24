#!/usr/bin/env python3
"""Create and seal layer-16 authority package 0021 exclusively."""

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
SOURCE = ROOT / "reports/ace2-layer16-runtime-pass-authority-0020"
SOURCE_REVIEW = ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0020"
TARGET = ROOT / "reports/ace2-layer16-runtime-pass-authority-0021"
DIRECTIVE = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "active_manager_directive.json"
)
SOURCE_ROOT = "9bc41be32b52e04e067a00230bb0fe472e70e81854d281a5d0920a1357157f3e"
SOURCE_REVIEW_HASHES = {
    "review-log.json": "d125c71053c56adc01c748d4a174d4cee59e66393d6b38397591365a2cc00502",
    "reviewer-adjudication.json": "2cb12a868d3c600a93e62ee4ebe064cbb3f1171ed18857646fbfd82865d899cd",
    "reviewer-receipt.json": "6f94b6e117d369663062b324d086ef0196e0c31a3422f31c3f3bce36b5197d6b",
}
ENGINEER_MISSION_ID = "dbf49e79c99c"
DIRECTIVE_REVISION = "8385764f119445349ca498ba2dff4822"
DIRECTIVE_SHA256 = "2671b8c99a428e348f1a05e201c6490f9c19967a2735ca3233a125043e7554bd"
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
    "write_host_completion_binding.py",
}
SEALED_MEMBERS = SOURCE_MEMBERS | {
    "acceptance-contract.json",
    "authority.json",
    "failed-predecessor-provenance.json",
    "hostile-controls.json",
    "review-request.json",
    "two-namespace-aggregate-fixtures.json",
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
    require(text.count(old) == 1, f"source fragment count changed: {old[:100]!r}")
    return text.replace(old, new)


def validate_sealed_source() -> None:
    require(
        SOURCE.is_dir()
        and not SOURCE.is_symlink()
        and stat.S_IMODE(SOURCE.stat().st_mode) == 0o555,
        "sealed package0020 directory changed",
    )
    records: dict[str, str] = {}
    for line in (SOURCE / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name not in records,
            "sealed package0020 checksum manifest changed",
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
        "sealed package0020 root or members changed",
    )
    for name, digest in records.items():
        path = SOURCE / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"sealed package0020 member changed: {path}",
        )
    require(
        (SOURCE / "TREE_ROOT.sha256").read_text(encoding="ascii").split()
        == [SOURCE_ROOT, "SHA256SUMS"],
        "sealed package0020 root sidecar changed",
    )
    require(
        SOURCE_REVIEW.is_dir()
        and not SOURCE_REVIEW.is_symlink()
        and stat.S_IMODE(SOURCE_REVIEW.stat().st_mode) == 0o555
        and {item.name for item in SOURCE_REVIEW.iterdir()}
        == set(SOURCE_REVIEW_HASHES),
        "package0020 PASS namespace changed",
    )
    for name, digest in SOURCE_REVIEW_HASHES.items():
        path = SOURCE_REVIEW / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"package0020 PASS artifact changed: {path}",
        )


def validate_inputs() -> None:
    validate_sealed_source()
    require(DIRECTIVE.is_file() and not DIRECTIVE.is_symlink(), "Manager V10 absent")
    directive = json.loads(DIRECTIVE.read_text(encoding="utf-8"))
    text = directive.get("text")
    require(
        isinstance(directive, dict)
        and directive.get("revision") == DIRECTIVE_REVISION
        and directive.get("objective_sha256") == DIRECTIVE_OBJECTIVE_SHA256
        and isinstance(text, str)
        and "MANAGER GRANT LAYER16 V10" in text
        and "package0021" in text
        and "reviewer_session_id" in text
        and "host_agent_id" in text
        and "host-completion-binding.json" in text
        and "Never require equality or preknowledge" in text
        and "Zero workload" in text,
        "active Manager V10 directive changed",
    )
    require(sha256_file(DIRECTIVE) == DIRECTIVE_SHA256, "Manager V10 bytes changed")
    for path in (
        TARGET,
        ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0021",
        ROOT / "reports/ace2-layer16-runtime-pass-authority-host-binding-0021",
        ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0021",
        ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0021",
    ):
        require(not os.path.lexists(path), f"package0021 target preexists: {path}")


TWO_NAMESPACE_BLOCK = r'''def _valid_typed_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", value) is not None
    )


def _review_argv(package_root: str, reviewer_session_id: str) -> list[str]:
    return [
        "/home/argustest/miniconda3/bin/python3.13",
        "-B",
        str((PACKAGE / "review_package.py").resolve()),
        "--package-root",
        package_root,
        "--reviewer-session-id",
        reviewer_session_id,
        "--task-name",
        REVIEW_TASK_NAME,
        "--engineer-mission-id",
        ENGINEER_MISSION_ID,
    ]


def _external_subject(package_root: str, mission_id: str) -> dict[str, Any]:
    return {
        "authority_identity": IDENTITY,
        "corrected_checkpoint": _frontier_binding()["corrected_checkpoint"],
        "engineer_mission_id": mission_id,
        "package_path": PACKAGE.relative_to(ROOT).as_posix(),
        "package_tree_root_sha256": package_root,
        "unit_key": UNIT,
    }


def _reviewer_identity(reviewer_session_id: str) -> dict[str, Any]:
    return {
        "agent_type": "general-purpose",
        "identity_field": "reviewer_session_id",
        "identity_source": "REVIEWER_RUNTIME_SESSION",
        "reviewer_session_id": reviewer_session_id,
        "task_name": REVIEW_TASK_NAME,
        "tool": "functions.task",
    }


def expected_review_record(
    package_root: str,
    hostile_controls_sha256: str,
    reviewer_session_id: str,
    review_log_sha256: str,
) -> dict[str, Any]:
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **ZERO_WORKLOAD,
        },
        "control_receipts": {
            "hostile_controls": {
                "path": (PACKAGE / "hostile-controls.json").relative_to(ROOT).as_posix(),
                "sha256": hostile_controls_sha256,
            },
            "immutable_review_log": {
                "path": REVIEW_LOG.relative_to(ROOT).as_posix(),
                "sha256": review_log_sha256,
            },
        },
        "creation": {
            "path": REVIEW.relative_to(ROOT).as_posix(),
            "semantics": "O_EXCL_NOFOLLOW",
            "target_preexisting": False,
        },
        "decision": "PASS",
        "execution_performed": False,
        "level": "L2",
        "producer_role": "reviewer",
        "reviewer_identity": _reviewer_identity(reviewer_session_id),
        "schema": "ace2-stage1-layer16-runtime-pass-review-adjudication-v13",
        "source_disjoint_from_package_creator": True,
        "subject": _external_subject(package_root, ENGINEER_MISSION_ID),
    }


def validate_review_record(
    record: dict[str, Any],
    package_root: str,
    hostile_controls_sha256: str,
    reviewer_session_id: str,
    review_log_sha256: str,
) -> None:
    subject = record.get("subject")
    require(isinstance(subject, dict), "review subject object required")
    root_token = subject.get("package_tree_root_sha256")
    require(
        isinstance(root_token, str)
        and re.fullmatch(r"[0-9a-f]{64}", root_token) is not None
        and root_token == package_root,
        "review subject must bind the exact bare canonical package root",
    )
    require(
        _valid_typed_id(reviewer_session_id)
        and reviewer_session_id != ENGINEER_MISSION_ID
        and reviewer_session_id not in STALE_REVIEWER_IDENTITIES,
        "fresh source-disjoint Reviewer session identity required",
    )
    require(
        "host_agent_id" not in json.dumps(record, sort_keys=True),
        "Stage-A adjudication must not claim a Host identity",
    )
    require(
        record
        == expected_review_record(
            package_root,
            hostile_controls_sha256,
            reviewer_session_id,
            review_log_sha256,
        ),
        "exact Reviewer-session Stage-A PASS required",
    )


def expected_reviewer_receipt_record(
    review_sha256: str,
    package_root: str,
    reviewer_session_id: str,
    review_log_sha256: str,
) -> dict[str, Any]:
    return {
        "adjudication": {
            "path": REVIEW.relative_to(ROOT).as_posix(),
            "sha256": review_sha256,
        },
        "creation": {
            "path": REVIEW_RECEIPT.relative_to(ROOT).as_posix(),
            "semantics": "O_EXCL_NOFOLLOW",
            "target_preexisting": False,
        },
        "engineer_creator": {
            "identity": ENGINEER_MISSION_ID,
            "role": "engineer",
        },
        "evidence": {
            "immutable_review_log": {
                "path": REVIEW_LOG.relative_to(ROOT).as_posix(),
                "sha256": review_log_sha256,
            },
            "package": {
                "path": PACKAGE.relative_to(ROOT).as_posix(),
                "tree_root_sha256": package_root,
            },
        },
        "producer_role": "reviewer",
        "reviewer_session": _reviewer_identity(reviewer_session_id),
        "schema": "ace2-supported-task-reviewer-receipt-v2",
        "source_disjoint_from_package_creator": True,
    }


def validate_reviewer_receipt_record(
    record: object,
    review_sha256: str,
    package_root: str,
    reviewer_session_id: str,
    review_log_sha256: str,
) -> None:
    require(isinstance(record, dict), "durable external Reviewer receipt required")
    require(
        "host_agent_id" not in json.dumps(record, sort_keys=True),
        "Stage-A receipt must not claim a Host identity",
    )
    require(
        record
        == expected_reviewer_receipt_record(
            review_sha256,
            package_root,
            reviewer_session_id,
            review_log_sha256,
        ),
        "exact Reviewer-session receipt required",
    )


def _decode_logged_stream(record: object, name: str) -> bytes:
    import base64

    require(isinstance(record, dict), f"{name} transcript stream required")
    encoded = record.get(f"{name}_base64")
    digest = record.get(f"{name}_sha256")
    require(
        isinstance(encoded, str)
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
        f"valid {name} transcript binding required",
    )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise Layer16Error(f"valid {name} transcript encoding required") from error
    require(hashlib.sha256(raw).hexdigest() == digest, f"{name} transcript changed")
    return raw


def validate_review_log_record(
    record: object,
    package_root: str,
    reviewer_session_id: str,
) -> None:
    require(isinstance(record, dict), "immutable supported-task review log required")
    log = record
    require(
        set(log)
        == {
            "activity",
            "engineer_creator",
            "execution_performed",
            "package",
            "review_argv",
            "reviewer_session",
            "schema",
            "source_inspection",
            "transcripts",
        },
        "review log surface changed",
    )
    require(
        log.get("schema") == "ace2-supported-task-review-log-v2"
        and log.get("activity")
        == {
            "authority_consumed": 0,
            "authority_created": 0,
            **ZERO_WORKLOAD,
        }
        and log.get("execution_performed") is False
        and log.get("engineer_creator")
        == {"identity": ENGINEER_MISSION_ID, "role": "engineer"}
        and log.get("reviewer_session") == _reviewer_identity(reviewer_session_id)
        and log.get("review_argv")
        == _review_argv(package_root, reviewer_session_id)
        and log.get("package")
        == {
            "path": PACKAGE.relative_to(ROOT).as_posix(),
            "tree_root_sha256": package_root,
        },
        "review log identity, root, argv, or activity changed",
    )
    require(
        "host_agent_id" not in json.dumps(log, sort_keys=True),
        "Stage-A log must not claim a Host identity",
    )
    sums = parse_sums(PACKAGE / "SHA256SUMS")
    expected_sources = {
        name: digest for name, digest in sums.items() if name.endswith(".py")
    }
    require(
        log.get("source_inspection")
        == {
            "inspected_by_reviewer_session_id": reviewer_session_id,
            "member_sha256": expected_sources,
            "result": "PASS_SOURCE_READ_AND_HASHED",
        },
        "Reviewer source inspection binding changed",
    )
    transcripts = log.get("transcripts")
    expected_commands = [
        [
            "/home/argustest/miniconda3/bin/python3.13",
            "-B",
            str((PACKAGE / "validate_nonexecuting.py").resolve()),
        ],
        [
            "/home/argustest/miniconda3/bin/python3.13",
            "-B",
            str((PACKAGE / "hostile_controls.py").resolve()),
        ],
    ]
    require(
        isinstance(transcripts, list) and len(transcripts) == 2,
        "exact preflight and hostile-control transcripts required",
    )
    decoded = []
    for transcript_record, expected_argv in zip(transcripts, expected_commands):
        require(
            isinstance(transcript_record, dict)
            and set(transcript_record)
            == {
                "argv",
                "cwd",
                "exit_code",
                "stderr_base64",
                "stderr_sha256",
                "stdout_base64",
                "stdout_sha256",
            }
            and transcript_record.get("argv") == expected_argv
            and transcript_record.get("cwd") == str(ROOT)
            and transcript_record.get("exit_code") == 0,
            "review transcript command, root, or completion changed",
        )
        stdout = _decode_logged_stream(transcript_record, "stdout")
        stderr = _decode_logged_stream(transcript_record, "stderr")
        require(stderr == b"", "review command emitted stderr")
        try:
            decoded.append(json.loads(stdout))
        except json.JSONDecodeError as error:
            raise Layer16Error("review command did not emit JSON") from error
    require(
        decoded[0].get("status") == "PASS_READY_WITHOUT_WORKLOAD_EXECUTION"
        and decoded[0].get("package_tree_root_sha256") == package_root
        and decoded[1].get("status") == "PASS_ZERO_WORKLOAD_ZERO_CONSUMPTION"
        and decoded[0].get("activity")
        == {"authority_consumed": 0, "authority_created": 0, **ZERO_WORKLOAD}
        and decoded[1].get("activity")
        == {"authority_consumed": 0, "authority_created": 0, **ZERO_WORKLOAD},
        "review transcript results are not zero-workload PASS records",
    )


def validate_review_bundle(package_root: str) -> str:
    for path in (REVIEW_LOG, REVIEW, REVIEW_RECEIPT):
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444,
            f"immutable supported-task review artifact absent: {path}",
        )
    require(
        REVIEW_NAMESPACE.is_dir()
        and not REVIEW_NAMESPACE.is_symlink()
        and stat.S_IMODE(REVIEW_NAMESPACE.stat().st_mode) == 0o555
        and {item.name for item in REVIEW_NAMESPACE.iterdir()}
        == {REVIEW.name, REVIEW_LOG.name, REVIEW_RECEIPT.name},
        "exact read-only Stage-A review namespace required",
    )
    receipt = load_json(REVIEW_RECEIPT)
    reviewer = receipt.get("reviewer_session")
    reviewer_session_id = (
        reviewer.get("reviewer_session_id") if isinstance(reviewer, dict) else None
    )
    require(
        _valid_typed_id(reviewer_session_id),
        "Reviewer session identity absent from Stage-A receipt",
    )
    log_sha256 = sha256_file(REVIEW_LOG)
    review_sha256 = sha256_file(REVIEW)
    validate_review_log_record(
        load_json(REVIEW_LOG), package_root, reviewer_session_id
    )
    controls = parse_sums(PACKAGE / "SHA256SUMS")
    validate_review_record(
        load_json(REVIEW),
        package_root,
        controls["hostile-controls.json"],
        reviewer_session_id,
        log_sha256,
    )
    validate_reviewer_receipt_record(
        receipt,
        review_sha256,
        package_root,
        reviewer_session_id,
        log_sha256,
    )
    return reviewer_session_id


def _reject_duplicate_json_keys(
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


REVIEW_TASK_DESCRIPTION = "Adjudicate sealed package0021"


def expected_review_prompt(package_root: str) -> str:
    return (
        "Act as the sole source-disjoint external Reviewer for the sealed package "
        f"{PACKAGE}. Work in {ROOT}. Inspect the sealed source and review-request.json, "
        "preserve package0020 and its PASS artifacts and all earlier packages byte-for-byte, "
        "and create no Host completion binding, Manager grant, consumption state, authority, "
        "RTL execution, reference execution, model execution, tokens, PPA, U280, or Stage2 "
        "work. Use your exact Reviewer backend/session identity visible inside your runtime "
        "as <reviewer-session-id>; this is reviewer_session_id, not the Host public task "
        "agent_id, which is unavailable until completion. Run exactly: "
        "/home/argustest/miniconda3/bin/python3.13 -B "
        f"{(PACKAGE / 'review_package.py').resolve()} --package-root {package_root} "
        f"--reviewer-session-id <reviewer-session-id> --task-name {REVIEW_TASK_NAME} "
        f"--engineer-mission-id {ENGINEER_MISSION_ID}. The package-native script "
        "exclusively creates the Stage-A adjudication, immutable review log, and reviewer "
        "receipt; none may contain host_agent_id. Do not hand-create, edit, replace, retry, "
        "repair, or reseal any package or review artifact. If the exact command succeeds, "
        "return only its exact canonical one-line JSON stdout as your final response, with "
        "no Markdown or additional text. If it fails, return a concise failure and do not "
        "weaken or retry the checks."
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
    reviewer_session_id: str,
    adjudication_sha256: str,
    review_log_sha256: str,
    receipt_sha256: str,
) -> dict[str, str]:
    return {
        "adjudication_sha256": adjudication_sha256,
        "package_tree_root_sha256": package_root,
        "receipt_sha256": receipt_sha256,
        "review_log_sha256": review_log_sha256,
        "reviewer_session_id": reviewer_session_id,
        "status": "PASS_STAGE_A_REVIEW_ARTIFACTS_CREATED",
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


def validate_supported_task_launch() -> dict[str, str]:
    package_root = validate_sealed_tree(PACKAGE, None, require_read_only=True)
    request, _start = _launch_records(_host_stream_events(), package_root)
    return {"task_tool_call_id": request["toolCallId"]}


def _normalized_host_aggregate(
    events: list[dict[str, Any]],
    package_root: str,
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
        and isinstance(restricted, dict),
        "public completion telemetry absent",
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
                "telemetry": {
                    "agent_type": properties.get("agent_type"),
                    "execution_mode": properties.get("execution_mode"),
                    "host_agent_id": restricted.get("agent_id"),
                },
                "toolCallId": complete.get("toolCallId"),
            }
        ],
    }


def _stage_a_hashes() -> dict[str, str]:
    return {
        "adjudication_sha256": sha256_file(REVIEW),
        "receipt_sha256": sha256_file(REVIEW_RECEIPT),
        "review_log_sha256": sha256_file(REVIEW_LOG),
    }


def _host_binding_from_aggregate(
    aggregate: object,
    package_root: str,
    reviewer_session_id: str,
    stage_a_hashes: dict[str, str],
) -> dict[str, Any]:
    require(
        isinstance(aggregate, dict)
        and set(aggregate) == {"request", "start", "complete"},
        "exact two-namespace task aggregate required",
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
    arguments = expected_task_arguments(package_root)
    require(
        isinstance(request, dict)
        and set(request) == {"arguments", "name", "toolCallId"}
        and request.get("name") == "functions.task"
        and request.get("arguments") == arguments
        and isinstance(start, dict)
        and set(start) == {"arguments", "toolCallId", "toolName"}
        and start.get("toolName") == "functions.task"
        and start.get("arguments") == arguments,
        "exact public task request and start arguments required",
    )
    tool_call_id = request.get("toolCallId")
    require(
        isinstance(tool_call_id, str)
        and bool(tool_call_id)
        and start.get("toolCallId") == tool_call_id
        and isinstance(complete, dict)
        and set(complete) == {"result", "success", "telemetry", "toolCallId"}
        and complete.get("toolCallId") == tool_call_id
        and complete.get("success") is True,
        "one successful matching public task completion required",
    )
    require(
        _valid_typed_id(reviewer_session_id)
        and reviewer_session_id != ENGINEER_MISSION_ID,
        "fresh Reviewer session identity required",
    )
    require(
        set(stage_a_hashes)
        == {"adjudication_sha256", "receipt_sha256", "review_log_sha256"}
        and all(
            isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            for value in stage_a_hashes.values()
        ),
        "exact Stage-A artifact hashes required",
    )
    telemetry = complete.get("telemetry")
    require(
        isinstance(telemetry, dict)
        and set(telemetry) == {"agent_type", "execution_mode", "host_agent_id"}
        and _valid_typed_id(telemetry.get("host_agent_id"))
        and telemetry.get("agent_type") == "general-purpose"
        and telemetry.get("execution_mode") == "sync",
        "typed Host completion telemetry required",
    )
    payload = expected_final_payload(
        package_root,
        reviewer_session_id,
        stage_a_hashes["adjudication_sha256"],
        stage_a_hashes["review_log_sha256"],
        stage_a_hashes["receipt_sha256"],
    )
    final = canonical_bytes(payload).decode("ascii").rstrip("\n")
    result = complete.get("result")
    require(
        isinstance(result, dict)
        and set(result) == {"content", "detailedContent"}
        and result.get("content") == final
        and result.get("detailedContent") == final,
        "exact Stage-A final payload required",
    )
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **ZERO_WORKLOAD,
        },
        "completion": {
            "final_payload": payload,
            "final_payload_sha256": hashlib.sha256(final.encode("ascii")).hexdigest(),
            "success": True,
        },
        "creation": {
            "path": HOST_BINDING.relative_to(ROOT).as_posix(),
            "semantics": "O_EXCL_NOFOLLOW",
            "target_preexisting": False,
        },
        "execution_performed": False,
        "identity_namespaces": {
            "comparison_policy": "TYPED_NAMESPACES_NOT_COMPARED_FOR_EQUALITY",
            "host": {
                "host_agent_id": telemetry["host_agent_id"],
                "identity_field": "host_agent_id",
                "identity_source": "HOST_PUBLIC_TOOL_COMPLETION_TELEMETRY",
            },
            "reviewer": {
                "identity_field": "reviewer_session_id",
                "identity_source": "STAGE_A_REVIEWER_RUNTIME_SESSION",
                "reviewer_session_id": reviewer_session_id,
            },
        },
        "package": {
            "path": PACKAGE.relative_to(ROOT).as_posix(),
            "tree_root_sha256": package_root,
        },
        "producer": {
            "mechanism": "SEALED_DETERMINISTIC_PACKAGE_SCRIPT",
            "role": "host_binding_writer",
            "source": HOST_AGENT_IO.as_posix(),
        },
        "schema": "ace2-stage1-layer16-host-completion-binding-v1",
        "stage_a_artifacts": {
            "adjudication": {
                "path": REVIEW.relative_to(ROOT).as_posix(),
                "sha256": stage_a_hashes["adjudication_sha256"],
            },
            "review_log": {
                "path": REVIEW_LOG.relative_to(ROOT).as_posix(),
                "sha256": stage_a_hashes["review_log_sha256"],
            },
            "reviewer_receipt": {
                "path": REVIEW_RECEIPT.relative_to(ROOT).as_posix(),
                "sha256": stage_a_hashes["receipt_sha256"],
            },
        },
        "task": {
            "arguments": arguments,
            "tool": "functions.task",
            "toolCallId": tool_call_id,
        },
    }


def validate_host_binding_record(
    record: object,
    aggregate: object,
    package_root: str,
    reviewer_session_id: str,
    stage_a_hashes: dict[str, str],
) -> dict[str, Any]:
    expected = _host_binding_from_aggregate(
        aggregate, package_root, reviewer_session_id, stage_a_hashes
    )
    require(record == expected, "Host completion binding changed or was forged")
    return expected


def _fixture_rejection(name: str, operation: Any) -> dict[str, str]:
    try:
        operation()
    except Layer16Error:
        return {"name": name, "result": "PASS_REJECTED_WITHOUT_CONSUMPTION"}
    raise Layer16Error(f"two-namespace fixture unexpectedly accepted: {name}")


def run_two_namespace_aggregate_fixtures() -> dict[str, Any]:
    import copy

    root = "a" * 64
    reviewer_session_id = "reviewer-session-package0021"
    host_agent_id = "00000000-0000-4000-8000-000000002021"
    hashes = {
        "adjudication_sha256": "b" * 64,
        "review_log_sha256": "c" * 64,
        "receipt_sha256": "d" * 64,
    }
    payload = expected_final_payload(
        root,
        reviewer_session_id,
        hashes["adjudication_sha256"],
        hashes["review_log_sha256"],
        hashes["receipt_sha256"],
    )
    final = canonical_bytes(payload).decode("ascii").rstrip("\n")
    tool_call_id = "call_layer16_package0021_two_namespace"
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
                    "agent_type": "general-purpose",
                    "execution_mode": "sync",
                    "host_agent_id": host_agent_id,
                },
                "toolCallId": tool_call_id,
            }
        ],
    }
    binding = _host_binding_from_aggregate(
        aggregate, root, reviewer_session_id, hashes
    )
    validate_host_binding_record(
        binding, aggregate, root, reviewer_session_id, hashes
    )
    cases = [{"name": "positive-distinct-typed-namespaces", "result": "PASS_ACCEPTED"}]
    equal = copy.deepcopy(aggregate)
    equal["complete"][0]["telemetry"]["host_agent_id"] = reviewer_session_id
    _host_binding_from_aggregate(equal, root, reviewer_session_id, hashes)
    cases.append(
        {
            "name": "positive-equal-values-no-equality-assumption",
            "result": "PASS_ACCEPTED",
        }
    )

    def reject_aggregate(name: str, mutate: Any) -> None:
        candidate = copy.deepcopy(aggregate)
        mutate(candidate)
        cases.append(
            _fixture_rejection(
                name,
                lambda: _host_binding_from_aggregate(
                    candidate, root, reviewer_session_id, hashes
                ),
            )
        )

    def reject_binding(name: str, mutate: Any) -> None:
        candidate = copy.deepcopy(binding)
        mutate(candidate)
        cases.append(
            _fixture_rejection(
                name,
                lambda: validate_host_binding_record(
                    candidate, aggregate, root, reviewer_session_id, hashes
                ),
            )
        )

    reject_aggregate("duplicate-request", lambda value: value["request"].append(value["request"][0]))
    reject_aggregate("duplicate-start", lambda value: value["start"].append(value["start"][0]))
    reject_aggregate("duplicate-completion", lambda value: value["complete"].append(value["complete"][0]))
    reject_aggregate("missing-completion", lambda value: value.__setitem__("complete", []))
    reject_aggregate("failed-completion", lambda value: value["complete"][0].__setitem__("success", False))
    reject_aggregate("missing-result", lambda value: value["complete"][0].__setitem__("result", None))
    reject_aggregate("missing-host-agent-id", lambda value: value["complete"][0]["telemetry"].pop("host_agent_id"))
    reject_aggregate("mismatched-task-name", lambda value: value["request"][0]["arguments"].__setitem__("name", "forged"))
    reject_aggregate("mismatched-task-type", lambda value: value["start"][0]["arguments"].__setitem__("agent_type", "explore"))
    reject_aggregate("mismatched-task-mode", lambda value: value["request"][0]["arguments"].__setitem__("mode", "background"))
    reject_aggregate("mismatched-toolCallId", lambda value: value["complete"][0].__setitem__("toolCallId", "call_forged"))
    reject_aggregate("forged-final-payload", lambda value: value["complete"][0]["result"].__setitem__("content", final + " forged"))
    reject_binding("missing-reviewer-session-id", lambda value: value["identity_namespaces"]["reviewer"].pop("reviewer_session_id"))
    reject_binding("swapped-identity-fields", lambda value: (
        value["identity_namespaces"]["reviewer"].__setitem__("reviewer_session_id", host_agent_id),
        value["identity_namespaces"]["host"].__setitem__("host_agent_id", reviewer_session_id),
    ))
    reject_binding("false-equality-requirement", lambda value: value["identity_namespaces"].__setitem__("comparison_policy", "REQUIRE_EQUAL"))
    reject_binding("mismatched-reviewer-session", lambda value: value["identity_namespaces"]["reviewer"].__setitem__("reviewer_session_id", "reviewer-session-forged"))
    reject_binding("mismatched-stage-a-hash", lambda value: value["stage_a_artifacts"]["adjudication"].__setitem__("sha256", "e" * 64))
    reject_binding("forged-binding-payload-hash", lambda value: value["completion"].__setitem__("final_payload_sha256", "e" * 64))
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **ZERO_WORKLOAD,
        },
        "cases": cases,
        "execution_performed": False,
        "positive_aggregate": aggregate,
        "positive_binding": binding,
        "schema": "ace2-two-namespace-task-aggregate-fixtures-v1",
        "status": "PASS_TWO_NAMESPACE_AGGREGATE_FIXTURES",
    }


def _write_exclusive_artifact(path: Path, value: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        offset = 0
        while offset < len(value):
            offset += os.write(descriptor, value[offset:])
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_host_completion_binding() -> dict[str, Any]:
    package_root = validate_sealed_tree(PACKAGE, None, require_read_only=True)
    reviewer_session_id = validate_review_bundle(package_root)
    hashes = _stage_a_hashes()
    aggregate = _normalized_host_aggregate(_host_stream_events(), package_root)
    binding = _host_binding_from_aggregate(
        aggregate, package_root, reviewer_session_id, hashes
    )
    require(
        not os.path.lexists(HOST_BINDING_NAMESPACE),
        "Host completion binding namespace preexisted",
    )
    os.mkdir(HOST_BINDING_NAMESPACE, mode=0o755)
    _write_exclusive_artifact(HOST_BINDING, canonical_bytes(binding))
    os.chmod(HOST_BINDING_NAMESPACE, 0o555)
    for directory in (HOST_BINDING_NAMESPACE, HOST_BINDING_NAMESPACE.parent):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    validate_host_binding_record(
        _parse_json_no_duplicates(HOST_BINDING.read_text(encoding="ascii")),
        aggregate,
        package_root,
        reviewer_session_id,
        hashes,
    )
    return {
        "host_agent_id": binding["identity_namespaces"]["host"]["host_agent_id"],
        "host_completion_binding_sha256": sha256_file(HOST_BINDING),
        "package_tree_root_sha256": package_root,
        "reviewer_session_id": reviewer_session_id,
        "status": "PASS_HOST_COMPLETION_BINDING_CREATED",
        "task_tool_call_id": binding["task"]["toolCallId"],
    }


def validate_framework_l2() -> dict[str, Any]:
    package_root = validate_sealed_tree(PACKAGE, None, require_read_only=True)
    reviewer_session_id = validate_review_bundle(package_root)
    hashes = _stage_a_hashes()
    require(
        HOST_BINDING_NAMESPACE.is_dir()
        and not HOST_BINDING_NAMESPACE.is_symlink()
        and stat.S_IMODE(HOST_BINDING_NAMESPACE.stat().st_mode) == 0o555
        and {item.name for item in HOST_BINDING_NAMESPACE.iterdir()}
        == {HOST_BINDING.name}
        and HOST_BINDING.is_file()
        and not HOST_BINDING.is_symlink()
        and stat.S_IMODE(HOST_BINDING.stat().st_mode) == 0o444,
        "exact immutable Host completion binding namespace required",
    )
    aggregate = _normalized_host_aggregate(_host_stream_events(), package_root)
    record = _parse_json_no_duplicates(HOST_BINDING.read_text(encoding="ascii"))
    binding = validate_host_binding_record(
        record, aggregate, package_root, reviewer_session_id, hashes
    )
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **ZERO_WORKLOAD,
        },
        "execution_performed": False,
        "host_agent_id": binding["identity_namespaces"]["host"]["host_agent_id"],
        "host_completion_binding_sha256": sha256_file(HOST_BINDING),
        "package_tree_root_sha256": package_root,
        "reviewer_session_id": reviewer_session_id,
        "stage_a_hashes": hashes,
        "status": "PASS_FRAMEWORK_L2_TWO_NAMESPACE_REVIEW_AUTHENTICATED",
        "task_tool_call_id": binding["task"]["toolCallId"],
    }


'''


HOSTILE_SOURCE = r'''#!/usr/bin/env python3
"""Zero-workload hostile controls for layer-16 authority package 0021."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Callable


def _load_layer16() -> object:
    path = Path(__file__).with_name("execute_layer16.py")
    spec = importlib.util.spec_from_file_location("execute_layer16_hostile_v13", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load layer-16 executor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


layer16 = _load_layer16()


def expect_rejection(name: str, operation: Callable[[], None]) -> dict[str, str]:
    try:
        operation()
    except (layer16.Layer16Error, FileNotFoundError):
        return {"name": name, "result": "PASS_REJECTED_WITHOUT_CONSUMPTION"}
    raise RuntimeError(f"hostile control unexpectedly accepted: {name}")


def run_controls() -> dict[str, object]:
    authority = layer16.authority_record()
    before = layer16.live_snapshot()
    cases: list[dict[str, str]] = []
    for name, mutate in (
        ("unit-scope", lambda value: value.__setitem__("unit_key", "position-00/layer-17")),
        ("cardinality", lambda value: value.__setitem__("cardinality", 2)),
        (
            "corrected-checkpoint",
            lambda value: value["bindings"]["corrected_checkpoint"].__setitem__(
                "sha256", "0" * 64
            ),
        ),
        (
            "manager-corrected-directive",
            lambda value: value["bindings"]["manager_corrected_directive"].__setitem__(
                "sha256", "0" * 64
            ),
        ),
        (
            "sealed-invocation",
            lambda value: value["exact_invocation"]["argv"].append("--alternate-route"),
        ),
    ):
        candidate = copy.deepcopy(authority)
        mutate(candidate)
        cases.append(
            expect_rejection(
                name,
                lambda candidate=candidate: layer16.require(
                    candidate == layer16.authority_record(),
                    "authority mutation rejected",
                ),
            )
        )

    root = "a" * 64
    controls_sha = "b" * 64
    review_sha = "c" * 64
    log_sha = "d" * 64
    reviewer_session_id = "reviewer-session-package0021"
    review = layer16.expected_review_record(
        root, controls_sha, reviewer_session_id, log_sha
    )
    receipt = layer16.expected_reviewer_receipt_record(
        review_sha, root, reviewer_session_id, log_sha
    )
    for name, mutate in (
        (
            "review-root-forgery",
            lambda value: value["subject"].__setitem__(
                "package_tree_root_sha256", "0" * 64
            ),
        ),
        (
            "review-session-mismatch",
            lambda value: value["reviewer_identity"].__setitem__(
                "reviewer_session_id", "reviewer-session-forged"
            ),
        ),
        (
            "review-host-identity-injection",
            lambda value: value.__setitem__("host_agent_id", "host-agent-forged"),
        ),
    ):
        candidate = copy.deepcopy(review)
        mutate(candidate)
        cases.append(
            expect_rejection(
                name,
                lambda candidate=candidate: layer16.validate_review_record(
                    candidate,
                    root,
                    controls_sha,
                    reviewer_session_id,
                    log_sha,
                ),
            )
        )
    for name, mutate in (
        (
            "receipt-log-hash-forgery",
            lambda value: value["evidence"]["immutable_review_log"].__setitem__(
                "sha256", "0" * 64
            ),
        ),
        (
            "receipt-session-mismatch",
            lambda value: value["reviewer_session"].__setitem__(
                "reviewer_session_id", "reviewer-session-forged"
            ),
        ),
        (
            "receipt-host-identity-injection",
            lambda value: value.__setitem__("host_agent_id", "host-agent-forged"),
        ),
    ):
        candidate = copy.deepcopy(receipt)
        mutate(candidate)
        cases.append(
            expect_rejection(
                name,
                lambda candidate=candidate: layer16.validate_reviewer_receipt_record(
                    candidate,
                    review_sha,
                    root,
                    reviewer_session_id,
                    log_sha,
                ),
            )
        )

    fixtures = layer16.run_two_namespace_aggregate_fixtures()
    cases.extend(fixtures["cases"])
    after = layer16.live_snapshot()
    layer16.require(before == after, "hostile controls changed the live frontier")
    layer16.require(
        before["consumption_exists"] is False
        and before["grant_namespace_exists"] is False
        and before["host_binding_namespace_exists"] is False
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
        "schema": "ace2-stage1-layer16-hostile-controls-v13",
        "status": "PASS_ZERO_WORKLOAD_ZERO_CONSUMPTION",
    }


def main() -> int:
    sys.stdout.buffer.write(layer16.canonical_bytes(run_controls()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


REVIEW_SOURCE = r'''#!/usr/bin/env python3
"""Create package-native Reviewer Stage-A PASS, immutable log, and receipt."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_layer16() -> object:
    path = Path(__file__).with_name("execute_layer16.py")
    spec = importlib.util.spec_from_file_location("execute_layer16_review_v13", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load layer-16 executor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


layer16 = _load_layer16()
PYTHON = "/home/argustest/miniconda3/bin/python3.13"


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


def transcript(argv: list[str]) -> dict[str, Any]:
    environment = {
        "LC_ALL": "C",
        "PATH": "/home/argustest/miniconda3/bin:/usr/bin:/bin",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONSAFEPATH": "1",
        "PYTHONUTF8": "1",
    }
    result = subprocess.run(
        argv,
        cwd=layer16.ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    layer16.require(result.returncode == 0, f"review command failed: {argv}")
    return {
        "argv": argv,
        "cwd": str(layer16.ROOT),
        "exit_code": result.returncode,
        "stderr_base64": base64.b64encode(result.stderr).decode("ascii"),
        "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
        "stdout_base64": base64.b64encode(result.stdout).decode("ascii"),
        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--reviewer-session-id", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--engineer-mission-id", required=True)
    args = parser.parse_args()
    package_root = layer16.validate_sealed_tree(
        layer16.PACKAGE, None, require_read_only=True
    )
    layer16.require(
        args.package_root == package_root
        and args.task_name == layer16.REVIEW_TASK_NAME
        and args.engineer_mission_id == layer16.ENGINEER_MISSION_ID
        and sys.argv
        == [
            str(Path(__file__).resolve()),
            *layer16._review_argv(
                package_root, args.reviewer_session_id
            )[3:],
        ],
        "exact sealed review argv required",
    )
    layer16.validate_supported_task_launch()
    layer16.require(
        layer16._valid_typed_id(args.reviewer_session_id)
        and args.reviewer_session_id != layer16.ENGINEER_MISSION_ID
        and args.reviewer_session_id not in layer16.STALE_REVIEWER_IDENTITIES,
        "fresh Reviewer backend/session identity required",
    )
    before = layer16.live_snapshot()
    for path in (
        layer16.REVIEW_NAMESPACE,
        layer16.HOST_BINDING_NAMESPACE,
        layer16.GRANT_NAMESPACE,
        layer16.CONSUMPTION,
    ):
        layer16.require(not os.path.lexists(path), f"review target preexisted: {path}")
    source_hashes = {
        name: digest
        for name, digest in layer16.parse_sums(
            layer16.PACKAGE / "SHA256SUMS"
        ).items()
        if name.endswith(".py")
    }
    commands = [
        [
            PYTHON,
            "-B",
            str((layer16.PACKAGE / "validate_nonexecuting.py").resolve()),
        ],
        [
            PYTHON,
            "-B",
            str((layer16.PACKAGE / "hostile_controls.py").resolve()),
        ],
    ]
    transcripts = [transcript(command) for command in commands]
    after_controls = layer16.live_snapshot()
    layer16.require(before == after_controls, "Reviewer controls changed frontier")
    log = {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **layer16.ZERO_WORKLOAD,
        },
        "engineer_creator": {
            "identity": layer16.ENGINEER_MISSION_ID,
            "role": "engineer",
        },
        "execution_performed": False,
        "package": {
            "path": layer16.PACKAGE.relative_to(layer16.ROOT).as_posix(),
            "tree_root_sha256": package_root,
        },
        "review_argv": layer16._review_argv(
            package_root, args.reviewer_session_id
        ),
        "reviewer_session": layer16._reviewer_identity(
            args.reviewer_session_id
        ),
        "schema": "ace2-supported-task-review-log-v2",
        "source_inspection": {
            "inspected_by_reviewer_session_id": args.reviewer_session_id,
            "member_sha256": source_hashes,
            "result": "PASS_SOURCE_READ_AND_HASHED",
        },
        "transcripts": transcripts,
    }
    log_bytes = layer16.canonical_bytes(log)
    log_sha256 = hashlib.sha256(log_bytes).hexdigest()
    controls_sha256 = layer16.parse_sums(
        layer16.PACKAGE / "SHA256SUMS"
    )["hostile-controls.json"]
    review = layer16.expected_review_record(
        package_root,
        controls_sha256,
        args.reviewer_session_id,
        log_sha256,
    )
    review_bytes = layer16.canonical_bytes(review)
    review_sha256 = hashlib.sha256(review_bytes).hexdigest()
    receipt = layer16.expected_reviewer_receipt_record(
        review_sha256,
        package_root,
        args.reviewer_session_id,
        log_sha256,
    )
    os.mkdir(layer16.REVIEW_NAMESPACE, mode=0o755)
    write_exclusive(layer16.REVIEW_LOG, log_bytes)
    write_exclusive(layer16.REVIEW, review_bytes)
    write_exclusive(layer16.REVIEW_RECEIPT, layer16.canonical_bytes(receipt))
    os.chmod(layer16.REVIEW_NAMESPACE, 0o555)
    for directory in (
        layer16.REVIEW_NAMESPACE,
        layer16.REVIEW_NAMESPACE.parent,
    ):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    layer16.validate_review_bundle(package_root)
    result = layer16.expected_final_payload(
        package_root,
        args.reviewer_session_id,
        review_sha256,
        log_sha256,
        layer16.sha256_file(layer16.REVIEW_RECEIPT),
    )
    sys.stdout.buffer.write(layer16.canonical_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


SEAL_SOURCE = r'''#!/usr/bin/env python3
"""Create-exclusive construction and sealing for layer-16 package 0021."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent
EXPECTED_SOURCE_MEMBERS = {
    "execute_layer16.py",
    "hostile_controls.py",
    "review_package.py",
    "seal_package.py",
    "validate_nonexecuting.py",
    "validate_review.py",
    "write_host_completion_binding.py",
}


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load package member: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


layer16 = _load("execute_layer16_seal", PACKAGE / "execute_layer16.py")


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


def write_json(path: Path, value: Any) -> None:
    write_exclusive(path, layer16.canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    observed_sources = {
        item.name
        for item in PACKAGE.iterdir()
        if item.is_file() and item.suffix == ".py"
    }
    layer16.require(
        observed_sources == EXPECTED_SOURCE_MEMBERS,
        "source member set changed before sealing",
    )
    layer16._validate_bound_evidence(current_directive_required=True)
    for path in (
        layer16.REVIEW_NAMESPACE,
        layer16.HOST_BINDING_NAMESPACE,
        layer16.GRANT_NAMESPACE,
        layer16.CONSUMPTION,
    ):
        layer16.require(
            not os.path.lexists(path),
            f"fresh package downstream namespace must remain absent: {path}",
        )
    write_json(
        PACKAGE / "failed-predecessor-provenance.json",
        {
            **layer16._failed_lifecycle_binding(),
            "activity": {
                "authority_consumed": 0,
                "authority_created": 0,
                **layer16.ZERO_WORKLOAD,
            },
            "schema": "ace2-stage1-layer16-failed-predecessor-provenance-v13",
        },
    )
    write_json(PACKAGE / "authority.json", layer16.authority_record())
    write_json(
        PACKAGE / "acceptance-contract.json",
        layer16.acceptance_contract_record(),
    )
    write_json(PACKAGE / "review-request.json", layer16.review_request_record())
    fixtures = layer16.run_two_namespace_aggregate_fixtures()
    layer16.require(
        fixtures.get("status") == "PASS_TWO_NAMESPACE_AGGREGATE_FIXTURES"
        and all(
            case.get("result")
            in {"PASS_ACCEPTED", "PASS_REJECTED_WITHOUT_CONSUMPTION"}
            for case in fixtures.get("cases", [])
        ),
        "two-namespace aggregate fixtures failed",
    )
    write_json(PACKAGE / "two-namespace-aggregate-fixtures.json", fixtures)
    controls = _load(
        "hostile_controls_layer16_seal",
        PACKAGE / "hostile_controls.py",
    ).run_controls()
    write_json(PACKAGE / "hostile-controls.json", controls)
    members = sorted(
        item.relative_to(PACKAGE).as_posix()
        for item in PACKAGE.rglob("*")
        if item.is_file()
        and item.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
        and item.parent.name != "__pycache__"
    )
    expected = EXPECTED_SOURCE_MEMBERS | {
        "acceptance-contract.json",
        "authority.json",
        "failed-predecessor-provenance.json",
        "hostile-controls.json",
        "review-request.json",
        "two-namespace-aggregate-fixtures.json",
    }
    layer16.require(set(members) == expected, "sealed member set changed")
    sums = "".join(
        f"{sha256_file(PACKAGE / member)}  {member}\n" for member in members
    ).encode("ascii")
    write_exclusive(PACKAGE / "SHA256SUMS", sums)
    root = hashlib.sha256(sums).hexdigest()
    write_exclusive(
        PACKAGE / "TREE_ROOT.sha256",
        f"{root}  SHA256SUMS\n".encode("ascii"),
    )
    for item in PACKAGE.iterdir():
        if item.is_file():
            os.chmod(item, 0o444)
    os.chmod(PACKAGE, 0o555)
    descriptor = os.open(PACKAGE.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


VALIDATE_REVIEW_SOURCE = r'''#!/usr/bin/env python3
"""Validate Reviewer Stage A and Host completion Stage B at Framework L2."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_layer16() -> object:
    path = Path(__file__).with_name("execute_layer16.py")
    spec = importlib.util.spec_from_file_location(
        "execute_layer16_validate_review_v13", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load layer-16 executor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


layer16 = _load_layer16()


def main() -> int:
    before = layer16.live_snapshot()
    result = layer16.validate_framework_l2()
    authority = layer16.load_json(layer16.PACKAGE / "authority.json")
    result["preflight"] = layer16.validate_preflight(authority)
    after = layer16.live_snapshot()
    layer16.require(before == after, "Framework L2 validation changed the frontier")
    sys.stdout.buffer.write(layer16.canonical_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


HOST_WRITER_SOURCE = r'''#!/usr/bin/env python3
"""Create the Host completion binding from public Host agent_io."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_layer16() -> object:
    path = Path(__file__).with_name("execute_layer16.py")
    spec = importlib.util.spec_from_file_location(
        "execute_layer16_host_binding_v13", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load layer-16 executor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


layer16 = _load_layer16()


def main() -> int:
    result = layer16.create_host_completion_binding()
    sys.stdout.buffer.write(layer16.canonical_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def transform_executor(source: str) -> str:
    text = source.replace("0020", "0021").replace("v12", "v13")
    text = replace_once(
        text,
        'ENGINEER_MISSION_ID = "272d6b89076b"',
        f'ENGINEER_MISSION_ID = "{ENGINEER_MISSION_ID}"',
    )
    text = replace_once(
        text,
        'ACTIVE_DIRECTIVE_REVISION = "17bab73cafb1456e815825534269d32a"',
        f'ACTIVE_DIRECTIVE_REVISION = "{DIRECTIVE_REVISION}"',
    )
    text = replace_once(
        text,
        '"b9074a0d3145a0148c2f8651941d8bf134812e229fcc675ba67e9e9621e37b2b"',
        f'"{DIRECTIVE_SHA256}"',
    )
    text = replace_once(
        text,
        'REVIEW_RECEIPT = REVIEW_NAMESPACE / "reviewer-receipt.json"\n',
        'REVIEW_RECEIPT = REVIEW_NAMESPACE / "reviewer-receipt.json"\n'
        'HOST_BINDING_NAMESPACE = ROOT / '
        '"reports/ace2-layer16-runtime-pass-authority-host-binding-0021"\n'
        'HOST_BINDING = HOST_BINDING_NAMESPACE / "host-completion-binding.json"\n',
    )
    predecessor = '''    "0019": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0019",
        "f0865f8bb05f3a62740a4a791ca35bcf8d8e53ce892615cc2588fa3222566a61",
        "FRAMEWORK_INTERNAL_EVENT_TOPOLOGY_DEPENDENCY",
    ),
'''
    text = replace_once(
        text,
        predecessor,
        predecessor
        + '''    "0020": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0020",
        "9bc41be32b52e04e067a00230bb0fe472e70e81854d281a5d0920a1357157f3e",
        "REVIEWER_SESSION_AND_HOST_AGENT_NAMESPACE_CONFLATION",
    ),
''',
    )
    text = replace_once(
        text,
        'PRESERVED_REVIEW_0019_HASHES = {\n'
        '    "review-log.json": "d6bc5ce863deaf85f0ed74315f8a4e0e1bc70116e088a7bd89dad6f5b2984437",\n'
        '    "reviewer-adjudication.json": "0b3c7b5dace04e4d1c13711e01b9a8025f28a6633617410dc30302045ece0626",\n'
        '    "reviewer-receipt.json": "986b04f9714fbf56e74f6d66d1d9eb8b39cc490332fee88028d014db0497498d",\n'
        '}\n',
        'PRESERVED_REVIEW_0019_HASHES = {\n'
        '    "review-log.json": "d6bc5ce863deaf85f0ed74315f8a4e0e1bc70116e088a7bd89dad6f5b2984437",\n'
        '    "reviewer-adjudication.json": "0b3c7b5dace04e4d1c13711e01b9a8025f28a6633617410dc30302045ece0626",\n'
        '    "reviewer-receipt.json": "986b04f9714fbf56e74f6d66d1d9eb8b39cc490332fee88028d014db0497498d",\n'
        '}\n'
        'PRESERVED_REVIEW_0020 = ROOT / '
        '"reports/ace2-layer16-runtime-pass-authority-review-0020"\n'
        'PRESERVED_REVIEW_0020_HASHES = {\n'
        '    "review-log.json": "d125c71053c56adc01c748d4a174d4cee59e66393d6b38397591365a2cc00502",\n'
        '    "reviewer-adjudication.json": "2cb12a868d3c600a93e62ee4ebe064cbb3f1171ed18857646fbfd82865d899cd",\n'
        '    "reviewer-receipt.json": "6f94b6e117d369663062b324d086ef0196e0c31a3422f31c3f3bce36b5197d6b",\n'
        '}\n',
    )
    marker = "\n\ndef _failed_construction_binding() -> dict[str, Any]:\n"
    preserved_validator = '''

def validate_preserved_review_0020() -> None:
    require(
        PRESERVED_REVIEW_0020.is_dir()
        and not PRESERVED_REVIEW_0020.is_symlink()
        and stat.S_IMODE(PRESERVED_REVIEW_0020.stat().st_mode) == 0o555
        and {item.name for item in PRESERVED_REVIEW_0020.iterdir()}
        == set(PRESERVED_REVIEW_0020_HASHES),
        "package0020 PASS namespace changed",
    )
    for name, digest in PRESERVED_REVIEW_0020_HASHES.items():
        path = PRESERVED_REVIEW_0020 / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"package0020 PASS artifact changed: {path}",
        )
'''
    text = replace_once(text, marker, preserved_validator + marker)
    text = replace_once(
        text,
        '        "regression": "SUPPORTED_PUBLIC_TASK_AGGREGATE_ONLY",',
        '''        "package0020_review": {
            "failure_taxonomy_class": (
                "REVIEWER_SESSION_AND_HOST_AGENT_NAMESPACE_CONFLATION"
            ),
            "member_sha256": PRESERVED_REVIEW_0020_HASHES,
            "namespace_mode": "0555",
            "path": PRESERVED_REVIEW_0020.relative_to(ROOT).as_posix(),
            "preserved_unchanged": True,
        },
        "regression": "TWO_NAMESPACE_REVIEW_AND_HOST_COMPLETION_BINDING",''',
    )
    start = text.index("def _valid_agent_id(")
    end = text.index("def _directive_binding(", start)
    text = text[:start] + TWO_NAMESPACE_BLOCK + text[end:]
    text = replace_once(
        text,
        "def _validate_bound_evidence(*, current_directive_required: bool) -> None:\n"
        "    validate_failed_construction_0016()\n"
        "    validate_preserved_review_0019()\n",
        "def _validate_bound_evidence(*, current_directive_required: bool) -> None:\n"
        "    validate_failed_construction_0016()\n"
        "    validate_preserved_review_0019()\n"
        "    validate_preserved_review_0020()\n",
    )
    review_snapshot = '''            *(
                PRESERVED_REVIEW_0019 / name
                for name in sorted(PRESERVED_REVIEW_0019_HASHES)
            ),
'''
    text = replace_once(
        text,
        review_snapshot,
        review_snapshot
        + '''            *(
                PRESERVED_REVIEW_0020 / name
                for name in sorted(PRESERVED_REVIEW_0020_HASHES)
            ),
''',
    )
    text = replace_once(
        text,
        '        "grant_namespace_exists": os.path.lexists(GRANT_NAMESPACE),\n',
        '        "grant_namespace_exists": os.path.lexists(GRANT_NAMESPACE),\n'
        '        "host_binding_namespace_exists": '
        'os.path.lexists(HOST_BINDING_NAMESPACE),\n',
    )
    text = replace_once(
        text,
        '            REVIEW_RECEIPT.relative_to(ROOT).as_posix(),\n'
        '            GRANT.relative_to(ROOT).as_posix(),',
        '            REVIEW_RECEIPT.relative_to(ROOT).as_posix(),\n'
        '            HOST_BINDING.relative_to(ROOT).as_posix(),\n'
        '            GRANT.relative_to(ROOT).as_posix(),',
    )
    text = replace_once(
        text,
        '            "manager_grant": True,\n'
        '            "package": True,',
        '            "host_completion_binding": True,\n'
        '            "manager_grant": True,\n'
        '            "package": True,',
    )
    completion_start = text.index(
        '            "completion_authentication": {',
        text.index("def acceptance_contract_record"),
    )
    completion_end = text.index(
        '            "durable_external_process_receipt": {',
        completion_start,
    )
    completion_contract = '''            "completion_authentication": {
                "host_binding_path": HOST_BINDING.relative_to(ROOT).as_posix(),
                "host_identity_field": "host_agent_id",
                "host_identity_source": "HOST_PUBLIC_TOOL_COMPLETION_TELEMETRY",
                "identity_equality_required": False,
                "reviewer_identity_field": "reviewer_session_id",
                "reviewer_identity_source": "REVIEWER_RUNTIME_SESSION",
                "required_agent_type": "general-purpose",
                "required_task_mode": "sync",
                "required_task_name": REVIEW_TASK_NAME,
                "tool": "functions.task",
            },
'''
    text = text[:completion_start] + completion_contract + text[completion_end:]
    text = replace_once(
        text,
        '                "schema": "ace2-supported-task-reviewer-receipt-v1",',
        '                "schema": "ace2-supported-task-reviewer-receipt-v2",',
    )
    text = replace_once(
        text,
        '            "fresh_review_sha256_binding": "REQUIRED",\n',
        '            "fresh_host_completion_binding_sha256": "REQUIRED",\n'
        '            "fresh_review_sha256_binding": "REQUIRED",\n',
    )
    old_ordering = '''        "ordering": [
            "CREATE_EXCLUSIVE_AND_SEAL_LAYER16_PACKAGE",
            "SUPPORTED_TASK_TOOL_EXTERNAL_REVIEWER_CREATES_BOUND_PASS_LOG_AND_RECEIPT",
            "SUPPORTED_TASK_TOOL_RETURN_CROSS_AUTHENTICATES_AGENT_ID_AND_COMPLETION",
            "MANAGER_API_CREATE_EXCLUSIVE_GRANT_BINDS_REVIEW_RECEIPT_AND_DIRECTIVE",
            "CONSUME_ONCE_BEFORE_LAYER16_EXECUTION",
        ],'''
    new_ordering = '''        "ordering": [
            "CREATE_EXCLUSIVE_AND_SEAL_LAYER16_PACKAGE",
            "REVIEWER_CREATES_STAGE_A_WITH_REVIEWER_SESSION_ID_ONLY",
            "HOST_WRITER_BINDS_PUBLIC_COMPLETION_HOST_AGENT_ID_AND_STAGE_A_HASHES",
            "FRAMEWORK_L2_VALIDATES_BOTH_TYPED_IDENTITY_NAMESPACES",
            "MANAGER_API_CREATE_EXCLUSIVE_GRANT_BINDS_ACCEPTED_TWO_STAGE_REVIEW",
            "CONSUME_ONCE_BEFORE_LAYER16_EXECUTION",
        ],'''
    text = replace_once(text, old_ordering, new_ordering)
    text = text.replace(
        '"supported-task-public-request-start-complete-result-and-telemetry"',
        '"two-stage-reviewer-session-and-host-completion-authentication"',
    )
    text = text.replace(
        '"reviewer-agent-differs-from-engineer-package-creator"',
        '"reviewer-session-differs-from-engineer-creator-when-comparable"',
    )
    text = text.replace(
        '"reject-duplicate-missing-mismatched-or-failed-public-task-records"',
        '"reject-swapped-missing-mismatched-forged-duplicate-or-failed-bindings"',
    )
    text = text.replace(
        '"reject-engineer-reuse-and-identity-hash-or-payload-forgery"',
        '"accept-equal-id-values-without-cross-namespace-equality-requirement"',
    )
    text = text.replace(
        '"ignore-internal-turn-end-subagent-topology-and-optional-fields"',
        '"host-binding-derived-from-public-agent-io-completion-result-and-telemetry"',
    )
    text = text.replace(
        '"SUPPORTED_TASK_TOOL_SOURCE_DISJOINT_LAYER16_PACKAGE_REVIEW"',
        '"TWO_STAGE_TWO_NAMESPACE_SOURCE_DISJOINT_LAYER16_PACKAGE_REVIEW"',
    )
    require("def _valid_agent_id(" not in text, "legacy untyped identity validator retained")
    require(
        'REVIEW_TASK_NAME = "layer16-package0021-reviewer"' in text,
        "exact package0021 review task token absent",
    )
    return text


def transformed_sources() -> dict[str, str]:
    result = {
        "execute_layer16.py": transform_executor(
            (SOURCE / "execute_layer16.py").read_text(encoding="utf-8")
        ),
        "hostile_controls.py": HOSTILE_SOURCE,
        "review_package.py": REVIEW_SOURCE,
        "seal_package.py": SEAL_SOURCE,
        "validate_nonexecuting.py": (
            (SOURCE / "validate_nonexecuting.py")
            .read_text(encoding="utf-8")
            .replace("0020", "0021")
            .replace("v12", "v13")
        ),
        "validate_review.py": VALIDATE_REVIEW_SOURCE,
        "write_host_completion_binding.py": HOST_WRITER_SOURCE,
    }
    for name, text in result.items():
        compile(text, str(TARGET / name), "exec")
        require(
            "layer16-package0020-reviewer" not in text,
            f"stale task token in {name}",
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
        "package0021 seal failed: " + result.stderr.decode("utf-8", "replace"),
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
        "sealed package0021 member set changed",
    )
    validate_sealed_source()
    sys.stdout.buffer.write(result.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
