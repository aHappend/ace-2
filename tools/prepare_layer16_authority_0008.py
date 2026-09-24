#!/usr/bin/env python3
"""Create and seal layer-16 authority package 0008 exclusively."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/argustest/ace-2")
SOURCE = ROOT / "reports/ace2-layer16-runtime-pass-authority-0007"
TARGET = ROOT / "reports/ace2-layer16-runtime-pass-authority-0008"
SOURCE_ROOT = "27e1cf06c93f451880fa2de42a2d3ce9f88981a82fa3a6819437fe7ec5186e06"
SOURCE_EXECUTOR_SHA256 = (
    "fa85645ecad417e0b7e4e898c1c57e72c4214a5ac6ee8c28398be46e78ace4c5"
)
STALE_REVIEWER_SESSION_ID = "ef520aec-144c-4d42-931a-4a3d6fd7e0ce"
HISTORICAL_CREATOR_SESSION_ID = "11e34d00-6350-4965-81db-31d915ef77d7"
MISSION_ID = "398f2f5776d8"
OBJECTIVE_REVISION = "e0a6beb1d680a4c9"


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


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


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


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    require(text.count(start) == 1, f"source marker changed: {start!r}")
    require(text.count(end) == 1, f"source marker changed: {end!r}")
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement + text[finish:]


def validate_source() -> None:
    require(
        SOURCE.is_dir()
        and not SOURCE.is_symlink()
        and stat.S_IMODE(SOURCE.stat().st_mode) == 0o555,
        "package 0007 mode changed",
    )
    records: dict[str, str] = {}
    for line in (SOURCE / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name not in records,
            "package 0007 checksum manifest changed",
        )
        records[name] = digest
    observed = {
        item.relative_to(SOURCE).as_posix()
        for item in SOURCE.rglob("*")
        if item.is_file() and item.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    }
    require(observed == set(records), "package 0007 sealed member set changed")
    for name, digest in records.items():
        path = SOURCE / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"package 0007 member changed: {path}",
        )
    require(
        sha256_file(SOURCE / "SHA256SUMS") == SOURCE_ROOT
        and (SOURCE / "TREE_ROOT.sha256").read_text(encoding="ascii").split()
        == [SOURCE_ROOT, "SHA256SUMS"]
        and records.get("execute_layer16.py") == SOURCE_EXECUTOR_SHA256,
        "package 0007 root or executor changed",
    )
    executor = (SOURCE / "execute_layer16.py").read_text(encoding="utf-8")
    require(
        f'FRAMEWORK_REVIEWER_SESSION_ID = "{STALE_REVIEWER_SESSION_ID}"' in executor,
        "package 0007 no longer proves the stale Reviewer identity binding",
    )
    for path in (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0007",
        ROOT
        / ".argus_subagents"
        / "ace2-layer16-runtime-pass-authority-review-0007-reviewer-receipt.json",
        ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0007",
        ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0007",
    ):
        require(
            not os.path.lexists(path),
            f"failed package 0007 downstream namespace exists: {path}",
        )


CONSTANTS = f'''FAILED_PREDECESSOR_PACKAGE = (
    ROOT / "reports/ace2-layer16-runtime-pass-authority-0007"
)
FAILED_PREDECESSOR_TREE_ROOT = "{SOURCE_ROOT}"
FAILED_PREDECESSOR_EXECUTOR_SHA256 = (
    "{SOURCE_EXECUTOR_SHA256}"
)
FAILED_PREDECESSOR_REVIEW_NAMESPACE = (
    ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0007"
)
FAILED_PREDECESSOR_RECEIPT = (
    ROOT
    / ".argus_subagents"
    / "ace2-layer16-runtime-pass-authority-review-0007-reviewer-receipt.json"
)
FAILED_PREDECESSOR_GRANT_NAMESPACE = (
    ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0007"
)
FAILED_PREDECESSOR_CONSUMPTION = (
    ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0007"
)

REVIEW_RECEIPT = (
    ROOT
    / ".argus_subagents"
    / "ace2-layer16-runtime-pass-authority-review-0008-reviewer-receipt.json"
)
MISSION_ID = "{MISSION_ID}"
HOST_OBJECTIVE_REVISION = "{OBJECTIVE_REVISION}"
HOST_MISSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/398f2f5776d8/mission.json"
)
HOST_CHECKPOINT = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/398f2f5776d8/CHECKPOINT.md"
)
HOST_REVIEWER_ROLE_SESSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/398f2f5776d8/role-sessions/reviewer.json"
)
HOST_ENGINEER_ROLE_SESSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/398f2f5776d8/role-sessions/engineer.json"
)
HISTORICAL_STALE_REVIEWER_SESSION_ID = "{STALE_REVIEWER_SESSION_ID}"
HISTORICAL_PACKAGE_CREATOR_SESSION_ID = "{HISTORICAL_CREATOR_SESSION_ID}"

'''


FAILED_BINDING = '''def _failed_predecessor_binding() -> dict[str, Any]:
    return {
        "consumption_namespace_absent": True,
        "disposition": (
            "PRESERVED_FAILED_PRECOMPUTATION_REVIEWER_IDENTITY_BINDING"
        ),
        "executor_sha256": FAILED_PREDECESSOR_EXECUTOR_SHA256,
        "failure_taxonomy_class": "PRE_COMPUTATION_REVIEWER_IDENTITY_BINDING",
        "grant_namespace_absent": True,
        "hard_bound_future_reviewer_session_id": (
            HISTORICAL_STALE_REVIEWER_SESSION_ID
        ),
        "live_authorization_predicate": False,
        "package_path": FAILED_PREDECESSOR_PACKAGE.relative_to(ROOT).as_posix(),
        "package_mode": "0555",
        "package_member_mode": "0444",
        "package_tree_root_sha256": FAILED_PREDECESSOR_TREE_ROOT,
        "regression": (
            "REVIEWER_SESSION_AUTHENTICATED_RELATIONALLY_FROM_HOST_ROLE_SESSION"
        ),
        "review_namespace_absent": True,
        "reviewer_receipt_absent": True,
        "reviewer_receipt_path": (
            FAILED_PREDECESSOR_RECEIPT.relative_to(ROOT).as_posix()
        ),
        "root_cause_hypothesis": (
            "SEALED_PACKAGE_PREBOUND_A_FUTURE_REVIEWER_UUID_BEFORE_HOST_"
            "CREATED_THE_REVIEWER_ROLE_SESSION"
        ),
    }


'''


HOST_RECEIPT_FUNCTIONS = '''def expected_host_role_session_record(
    role: str,
    thread_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "role": role,
        "policy": "rolling",
        "objective_revision": HOST_OBJECTIVE_REVISION,
        "workdir": str(ROOT),
        "checkpoint_path": str(HOST_CHECKPOINT),
        "mission_context_path": str(HOST_MISSION),
        "thread_id": thread_id,
    }


def validate_host_role_session(record: object, required_role: str) -> str:
    require(isinstance(record, dict), f"Host {required_role} role-session required")
    role_session = record
    thread_id = role_session.get("thread_id")
    require(
        role_session.get("schema_version") == 2
        and role_session.get("role") == required_role
        and role_session.get("policy") == "rolling"
        and role_session.get("objective_revision") == HOST_OBJECTIVE_REVISION
        and role_session.get("workdir") == str(ROOT)
        and role_session.get("checkpoint_path") == str(HOST_CHECKPOINT)
        and role_session.get("mission_context_path") == str(HOST_MISSION)
        and isinstance(thread_id, str)
        and re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            thread_id,
        )
        is not None,
        f"authenticated Host {required_role} role-session required",
    )
    mission = load_json(HOST_MISSION)
    require(
        mission.get("mission_id") == MISSION_ID
        and mission.get("node_key") == "layer16-0008-dynamic-reviewer-package",
        "Host role-session is not bound to this mission",
    )
    return thread_id


def _load_host_role_sessions() -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (HOST_REVIEWER_ROLE_SESSION, HOST_ENGINEER_ROLE_SESSION, HOST_MISSION):
        require(
            path.is_file() and not path.is_symlink(),
            f"Host identity registry record absent: {path}",
        )
    return (
        load_json(HOST_REVIEWER_ROLE_SESSION),
        load_json(HOST_ENGINEER_ROLE_SESSION),
    )


def _host_reviewer_identity_binding(session_id: str) -> dict[str, Any]:
    return {
        "mission_id": MISSION_ID,
        "objective_revision": HOST_OBJECTIVE_REVISION,
        "path": str(HOST_REVIEWER_ROLE_SESSION),
        "registry_kind": "HOST_ROLE_SESSION_V2",
        "role": "reviewer",
        "session_id": session_id,
    }


def expected_reviewer_receipt_record(
    review_sha256: str,
    package_root: str,
    reviewer_role_session: object,
) -> dict[str, Any]:
    reviewer_identity = validate_host_role_session(
        reviewer_role_session,
        "reviewer",
    )
    return {
        "adjudication": {
            "path": REVIEW.relative_to(ROOT).as_posix(),
            "sha256": review_sha256,
        },
        "command_transcripts": [
            {
                "cwd": str(ROOT),
                "inner_exit_code": 0,
            }
        ],
        "completed_at": "EXTERNAL_COMPLETION_TIMESTAMP",
        "creation": {
            "path": REVIEW_RECEIPT.relative_to(ROOT).as_posix(),
            "semantics": "O_EXCL_NOFOLLOW",
            "target_preexisting": False,
        },
        "evidence": {
            "package": {
                "path": PACKAGE.relative_to(ROOT).as_posix(),
                "tree_root_sha256": package_root,
            }
        },
        "exit_code": 0,
        "host_identity": _host_reviewer_identity_binding(reviewer_identity),
        "mode": "direct-reviewer-nonexecuting",
        "producer_role": "reviewer",
        "provenance": {
            "session_id": reviewer_identity,
            "source_disjoint_from_package_creator": True,
        },
        "run_id": (
            "ace2-layer16-runtime-pass-authority-review-0008-"
            + reviewer_identity
        ),
        "schema": "argus-subagent-reviewer-receipt-v1",
        "session_id": reviewer_identity,
        "state": "done",
        "task_id": "ace2-layer16-runtime-pass-authority-review-0008-reviewer",
    }


def validate_reviewer_receipt_record(
    record: object,
    review_sha256: str,
    package_root: str,
    reviewer_role_session: object,
    engineer_role_session: object,
) -> None:
    require(isinstance(record, dict), "durable external Reviewer receipt required")
    receipt = record
    reviewer_identity = validate_host_role_session(
        reviewer_role_session,
        "reviewer",
    )
    engineer_identity = validate_host_role_session(
        engineer_role_session,
        "engineer",
    )
    require(
        receipt.get("schema") == "argus-subagent-reviewer-receipt-v1"
        and receipt.get("state") == "done"
        and receipt.get("exit_code") == 0
        and receipt.get("mode") == "direct-reviewer-nonexecuting",
        "completed external Reviewer process receipt required",
    )
    require(
        receipt.get("adjudication")
        == {
            "path": REVIEW.relative_to(ROOT).as_posix(),
            "sha256": review_sha256,
        },
        "Reviewer receipt must bind the exact adjudication",
    )
    evidence = receipt.get("evidence")
    require(isinstance(evidence, dict), "Reviewer receipt evidence object required")
    package = evidence.get("package")
    root_token = package.get("tree_root_sha256") if isinstance(package, dict) else None
    require(
        isinstance(package, dict)
        and package.get("path") == PACKAGE.relative_to(ROOT).as_posix()
        and isinstance(root_token, str)
        and re.fullmatch(r"[0-9a-f]{64}", root_token) is not None
        and root_token == package_root,
        "Reviewer receipt must bind the exact bare canonical package root",
    )
    provenance = receipt.get("provenance")
    require(isinstance(provenance, dict), "Reviewer process provenance required")
    excluded = {
        engineer_identity,
        HISTORICAL_PACKAGE_CREATOR_SESSION_ID,
        HISTORICAL_STALE_REVIEWER_SESSION_ID,
    }
    require(
        provenance.get("session_id") == reviewer_identity
        and receipt.get("session_id") == reviewer_identity
        and reviewer_identity not in excluded,
        "Reviewer identity must equal the Host Reviewer role-session and differ "
        "from creator, Engineer, and stale identities",
    )
    require(
        receipt.get("host_identity")
        == _host_reviewer_identity_binding(reviewer_identity),
        "Reviewer receipt lacks the exact Host mission role-session binding",
    )
    require(
        receipt.get("run_id")
        == (
            "ace2-layer16-runtime-pass-authority-review-0008-"
            + reviewer_identity
        )
        and receipt.get("task_id")
        == "ace2-layer16-runtime-pass-authority-review-0008-reviewer",
        "durable Reviewer task and run identity mismatch",
    )
    require(
        receipt.get("creation")
        == {
            "path": REVIEW_RECEIPT.relative_to(ROOT).as_posix(),
            "semantics": "O_EXCL_NOFOLLOW",
            "target_preexisting": False,
        },
        "Reviewer receipt must be create-exclusive and no-follow",
    )
    transcripts = receipt.get("command_transcripts")
    require(
        isinstance(receipt.get("completed_at"), str)
        and bool(receipt["completed_at"])
        and isinstance(transcripts, list)
        and bool(transcripts)
        and all(
            isinstance(item, dict)
            and item.get("cwd") == str(ROOT)
            and item.get("inner_exit_code") == 0
            for item in transcripts
        ),
        "completed successful external Reviewer process transcript required",
    )
    require(
        receipt.get("producer_role") == "reviewer"
        and provenance.get("source_disjoint_from_package_creator") is True,
        "Reviewer role and source-disjoint declaration required",
    )


'''


HOSTILE_IDENTITY_CONTROLS = '''    reviewer_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    engineer_id = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
    reviewer_host = layer16.expected_host_role_session_record(
        "reviewer",
        reviewer_id,
    )
    engineer_host = layer16.expected_host_role_session_record(
        "engineer",
        engineer_id,
    )
    valid_receipt = layer16.expected_reviewer_receipt_record(
        "c" * 64,
        canonical_root,
        reviewer_host,
    )
    layer16.validate_reviewer_receipt_record(
        valid_receipt,
        "c" * 64,
        canonical_root,
        reviewer_host,
        engineer_host,
    )

    def validate_receipt(
        receipt: object,
        reviewer_session: object = reviewer_host,
        engineer_session: object = engineer_host,
    ) -> None:
        layer16.validate_reviewer_receipt_record(
            receipt,
            "c" * 64,
            canonical_root,
            reviewer_session,
            engineer_session,
        )

    cases.append(
        expect_rejection("receipt-missing", lambda: validate_receipt(None))
    )
    cases.append(
        expect_rejection(
            "host-receipt-missing",
            lambda: validate_receipt(valid_receipt, None),
        )
    )
    cases.append(
        expect_rejection(
            "receipt-self-declared-only",
            lambda: layer16.validate_reviewer_receipt_record(
                {
                    "producer_role": "reviewer",
                    "provenance": {"session_id": reviewer_id},
                },
                "c" * 64,
                canonical_root,
                None,
                engineer_host,
            ),
        )
    )
    for name, mutate in (
        (
            "receipt-forged-role",
            lambda value: value.__setitem__("producer_role", "engineer"),
        ),
        (
            "receipt-unbound-adjudication",
            lambda value: value["adjudication"].__setitem__(
                "path",
                "reports/unbound/reviewer-adjudication.json",
            ),
        ),
        (
            "receipt-tampered-adjudication",
            lambda value: value["adjudication"].__setitem__("sha256", "0" * 64),
        ),
        (
            "receipt-altered-root",
            lambda value: value["evidence"]["package"].__setitem__(
                "tree_root_sha256",
                "0" * 64,
            ),
        ),
        (
            "receipt-mismatched-session",
            lambda value: value["provenance"].__setitem__(
                "session_id",
                "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            ),
        ),
        (
            "receipt-arbitrary-uuid",
            lambda value: (
                value.__setitem__(
                    "session_id",
                    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                ),
                value["provenance"].__setitem__(
                    "session_id",
                    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                ),
                value["host_identity"].__setitem__(
                    "session_id",
                    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                ),
            ),
        ),
        (
            "receipt-mismatched-run",
            lambda value: value.__setitem__("run_id", "arbitrary-run"),
        ),
        (
            "receipt-not-create-exclusive",
            lambda value: value["creation"].__setitem__(
                "semantics",
                "REPLACE",
            ),
        ),
    ):
        receipt = copy.deepcopy(valid_receipt)
        mutate(receipt)
        cases.append(
            expect_rejection(
                name,
                lambda receipt=receipt: validate_receipt(receipt),
            )
        )

    forged_host = copy.deepcopy(reviewer_host)
    forged_host["role"] = "engineer"
    cases.append(
        expect_rejection(
            "host-forged-role",
            lambda: validate_receipt(valid_receipt, forged_host),
        )
    )

    reused_host = layer16.expected_host_role_session_record(
        "reviewer",
        engineer_id,
    )
    reused_receipt = layer16.expected_reviewer_receipt_record(
        "c" * 64,
        canonical_root,
        reused_host,
    )
    cases.append(
        expect_rejection(
            "receipt-engineer-identity-reuse",
            lambda: validate_receipt(reused_receipt, reused_host),
        )
    )

    stale_host = layer16.expected_host_role_session_record(
        "reviewer",
        layer16.HISTORICAL_STALE_REVIEWER_SESSION_ID,
    )
    stale_receipt = layer16.expected_reviewer_receipt_record(
        "c" * 64,
        canonical_root,
        stale_host,
    )
    cases.append(
        expect_rejection(
            "receipt-stale-0007-identity",
            lambda: validate_receipt(stale_receipt, stale_host),
        )
    )

'''


def transform_executor() -> str:
    text = (SOURCE / "execute_layer16.py").read_text(encoding="utf-8")
    text = text.replace("0007", "0008")
    text = replace_section(
        text,
        "FAILED_PREDECESSOR_PACKAGE = (",
        "LAYER15_ROOT =",
        CONSTANTS,
    )
    text = replace_section(
        text,
        "def _failed_predecessor_binding() -> dict[str, Any]:",
        "def _checkpoint_resume_import_binding() -> dict[str, str]:",
        FAILED_BINDING,
    )
    text = text.replace(
        '''            "creator_process_session_id": PACKAGE_CREATOR_PROCESS_SESSION_ID,
            "creator_role": "engineer",
''',
        '''            "creator_identity": {
                "mission_id": MISSION_ID,
                "path": str(HOST_ENGINEER_ROLE_SESSION),
                "registry_kind": "HOST_ROLE_SESSION_V2",
                "role": "engineer",
            },
''',
    )
    text = text.replace(
        '''                "producer_role_claim_sufficient": False,
                "required_process_session_identity": FRAMEWORK_REVIEWER_SESSION_ID,
                "schema": "argus-subagent-reviewer-receipt-v1",
''',
        '''                "host_identity_relation": (
                    "RECEIPT_SESSION_ID_EQUALS_HOST_REVIEWER_ROLE_SESSION_THREAD_ID"
                ),
                "host_role_session_path": str(HOST_REVIEWER_ROLE_SESSION),
                "mission_id": MISSION_ID,
                "producer_role_claim_sufficient": False,
                "schema": "argus-subagent-reviewer-receipt-v1",
                "write_policy": "O_EXCL_NOFOLLOW",
''',
    )
    text = text.replace(
        '''            "excluded_process_session_identities": sorted(
                {
                    PACKAGE_CREATOR_PROCESS_SESSION_ID,
                    *TASK_ENGINEER_PROCESS_SESSION_IDENTITIES,
                }
            ),
''',
        '''            "excluded_identity_sources": [
                {
                    "path": str(HOST_ENGINEER_ROLE_SESSION),
                    "role": "engineer",
                },
                {
                    "session_id": HISTORICAL_PACKAGE_CREATOR_SESSION_ID,
                    "source": "PRESERVED_PACKAGE_0007_CREATOR",
                },
            ],
            "historical_stale_reviewer_session_id": (
                HISTORICAL_STALE_REVIEWER_SESSION_ID
            ),
''',
    )
    text = replace_section(
        text,
        "def expected_reviewer_receipt_record(",
        "def _directive_binding(record: dict[str, Any]) -> dict[str, Any]:",
        HOST_RECEIPT_FUNCTIONS,
    )
    old_evidence = '''    failed_sums = parse_sums(FAILED_PREDECESSOR_PACKAGE / "SHA256SUMS")
    require(
        failed_sums.get("execute_layer16.py") == FAILED_PREDECESSOR_EXECUTOR_SHA256,
        "failed 0006 executor binding evidence changed",
    )
    require(
        not os.path.lexists(FAILED_PREDECESSOR_REVIEW_NAMESPACE)
        and not os.path.lexists(FAILED_PREDECESSOR_RECEIPT)
        and not os.path.lexists(FAILED_PREDECESSOR_GRANT_NAMESPACE)
        and not os.path.lexists(FAILED_PREDECESSOR_CONSUMPTION),
        "failed 0006 downstream namespace unexpectedly exists",
    )
'''
    new_evidence = '''    failed_sums = parse_sums(FAILED_PREDECESSOR_PACKAGE / "SHA256SUMS")
    require(
        failed_sums.get("execute_layer16.py") == FAILED_PREDECESSOR_EXECUTOR_SHA256,
        "failed 0007 executor evidence changed",
    )
    failed_executor = (
        FAILED_PREDECESSOR_PACKAGE / "execute_layer16.py"
    ).read_text(encoding="utf-8")
    require(
        (
            'FRAMEWORK_REVIEWER_SESSION_ID = "'
            + HISTORICAL_STALE_REVIEWER_SESSION_ID
            + '"'
        )
        in failed_executor,
        "failed 0007 stale Reviewer identity evidence changed",
    )
    require(
        not os.path.lexists(FAILED_PREDECESSOR_REVIEW_NAMESPACE)
        and not os.path.lexists(FAILED_PREDECESSOR_RECEIPT)
        and not os.path.lexists(FAILED_PREDECESSOR_GRANT_NAMESPACE)
        and not os.path.lexists(FAILED_PREDECESSOR_CONSUMPTION),
        "failed 0007 downstream namespace unexpectedly exists",
    )
'''
    require(old_evidence in text, "bound evidence source changed")
    text = text.replace(old_evidence, new_evidence)
    old_receipt_preflight = '''    if os.path.lexists(REVIEW_RECEIPT):
        require(
            REVIEW_RECEIPT.is_file() and not REVIEW_RECEIPT.is_symlink(),
            "durable external Reviewer receipt is invalid",
        )
'''
    new_receipt_preflight = '''    if os.path.lexists(REVIEW_RECEIPT):
        require(
            REVIEW_RECEIPT.is_file()
            and not REVIEW_RECEIPT.is_symlink()
            and stat.S_IMODE(REVIEW_RECEIPT.stat().st_mode) == 0o444,
            "durable external Reviewer receipt is invalid",
        )
        require(REVIEW.is_file() and not REVIEW.is_symlink(), "bound review absent")
        package_root = validate_sealed_tree(
            PACKAGE,
            None,
            require_read_only=True,
        )
        controls = parse_sums(PACKAGE / "SHA256SUMS")
        review_sha256 = sha256_file(REVIEW)
        validate_review_record(
            load_json(REVIEW),
            package_root,
            controls["hostile-controls.json"],
        )
        reviewer_role_session, engineer_role_session = _load_host_role_sessions()
        validate_reviewer_receipt_record(
            load_json(REVIEW_RECEIPT),
            review_sha256,
            package_root,
            reviewer_role_session,
            engineer_role_session,
        )
'''
    require(old_receipt_preflight in text, "receipt preflight source changed")
    text = text.replace(old_receipt_preflight, new_receipt_preflight)
    old_receipt_call = '''    validate_reviewer_receipt_record(
        load_json(receipt_path),
        supplied,
        package_root,
    )
'''
    new_receipt_call = '''    reviewer_role_session, engineer_role_session = (
        _load_host_role_sessions()
    )
    validate_reviewer_receipt_record(
        load_json(receipt_path),
        supplied,
        package_root,
        reviewer_role_session,
        engineer_role_session,
    )
'''
    require(old_receipt_call in text, "receipt validation call source changed")
    text = text.replace(old_receipt_call, new_receipt_call)
    text = text.replace("failed-0006", "failed-0007")
    text = text.replace("failed_0006", "failed_0007")
    text = text.replace("failed 0006", "failed 0007")
    text = text.replace("package 0006", "package 0007")
    text = text.replace("PRECOMPUTATION_EXECUTOR_IMPORT", "PRECOMPUTATION_REVIEWER_IDENTITY")
    text = text.replace("-v3", "-v4")
    for forbidden in (
        "\nFRAMEWORK_REVIEWER_SESSION_ID =",
        "\nPACKAGE_CREATOR_PROCESS_SESSION_ID =",
        "\nTASK_ENGINEER_PROCESS_SESSION_IDENTITIES =",
    ):
        require(forbidden not in text, f"future identity constant survived: {forbidden}")
    compile(text, str(TARGET / "execute_layer16.py"), "exec")
    return text


def transform_hostile() -> str:
    text = (SOURCE / "hostile_controls.py").read_text(encoding="utf-8")
    text = text.replace("0007", "0008")
    text = replace_section(
        text,
        "    valid_receipt =",
        "    after =",
        HOSTILE_IDENTITY_CONTROLS,
    )
    text = text.replace("-v3", "-v4")
    compile(text, str(TARGET / "hostile_controls.py"), "exec")
    return text


def transform_simple(name: str) -> str:
    text = (SOURCE / name).read_text(encoding="utf-8")
    text = text.replace("0007", "0008").replace("-v3", "-v4")
    compile(text, str(TARGET / name), "exec")
    return text


def provenance_record() -> dict[str, Any]:
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            "generated_tokens": 0,
            "model_execution": 0,
            "ppa_execution": 0,
            "reference_execution": 0,
            "rtl_execution": 0,
            "u280_execution": 0,
            "unit_execution": 0,
        },
        "consumption_namespace_absent": True,
        "disposition": "PRESERVED_FAILED_PRECOMPUTATION_REVIEWER_IDENTITY_BINDING",
        "executor_sha256": SOURCE_EXECUTOR_SHA256,
        "failure_taxonomy_class": "PRE_COMPUTATION_REVIEWER_IDENTITY_BINDING",
        "grant_namespace_absent": True,
        "hard_bound_future_reviewer_session_id": STALE_REVIEWER_SESSION_ID,
        "live_authorization_predicate": False,
        "package_member_mode": "0444",
        "package_mode": "0555",
        "package_path": SOURCE.relative_to(ROOT).as_posix(),
        "package_tree_root_sha256": SOURCE_ROOT,
        "regression": (
            "REVIEWER_SESSION_AUTHENTICATED_RELATIONALLY_FROM_HOST_ROLE_SESSION"
        ),
        "review_namespace_absent": True,
        "reviewer_receipt_absent": True,
        "reviewer_receipt_path": (
            ".argus_subagents/"
            "ace2-layer16-runtime-pass-authority-review-0007-reviewer-receipt.json"
        ),
        "root_cause_hypothesis": (
            "SEALED_PACKAGE_PREBOUND_A_FUTURE_REVIEWER_UUID_BEFORE_HOST_"
            "CREATED_THE_REVIEWER_ROLE_SESSION"
        ),
        "schema": "ace2-stage1-layer16-failed-predecessor-provenance-v4",
    }


def main() -> int:
    validate_source()
    for path in (
        TARGET,
        ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0008",
        ROOT
        / ".argus_subagents"
        / "ace2-layer16-runtime-pass-authority-review-0008-reviewer-receipt.json",
        ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0008",
        ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0008",
    ):
        require(not os.path.lexists(path), f"fresh package namespace exists: {path}")

    sources = {
        "execute_layer16.py": transform_executor(),
        "hostile_controls.py": transform_hostile(),
        "seal_package.py": transform_simple("seal_package.py"),
        "validate_nonexecuting.py": transform_simple("validate_nonexecuting.py"),
    }
    TARGET.mkdir(mode=0o700)
    for name, text in sources.items():
        write_exclusive(TARGET / name, text.encode("utf-8"))
    write_exclusive(
        TARGET / "failed-predecessor-provenance.json",
        canonical_bytes(provenance_record()),
    )
    completed = subprocess.run(
        [sys.executable, "-B", str(TARGET / "seal_package.py")],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        sys.stdout.buffer.write(completed.stdout)
        sys.stderr.buffer.write(completed.stderr)
        raise PreparationError(
            f"package 0008 sealing failed with status {completed.returncode}"
        )
    sys.stdout.buffer.write(completed.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
