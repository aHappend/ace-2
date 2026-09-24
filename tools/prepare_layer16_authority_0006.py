#!/usr/bin/env python3
"""Create and seal layer-16 authority package 0006 exclusively."""

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
SOURCE = ROOT / "reports/ace2-layer16-runtime-pass-authority-0005"
SOURCE_REVIEW_NAMESPACE = (
    ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0005"
)
SOURCE_REVIEW = SOURCE_REVIEW_NAMESPACE / "reviewer-adjudication.json"
SOURCE_RECEIPT = (
    ROOT
    / ".argus_subagents"
    / "ace2-layer16-runtime-pass-authority-review-0005-reviewer-receipt.json"
)
SOURCE_GRANT = ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0005"
SOURCE_CONSUMPTION = (
    ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0005"
)

TARGET = ROOT / "reports/ace2-layer16-runtime-pass-authority-0006"
TARGET_REVIEW = ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0006"
TARGET_RECEIPT = (
    ROOT
    / ".argus_subagents"
    / "ace2-layer16-runtime-pass-authority-review-0006-reviewer-receipt.json"
)
TARGET_GRANT = ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0006"
TARGET_CONSUMPTION = (
    ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0006"
)

FRAMEWORK_REVIEWER_SESSION_ID = "ef520aec-144c-4d42-931a-4a3d6fd7e0ce"
ENGINEER_SESSION_ID = "11e34d00-6350-4965-81db-31d915ef77d7"


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


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


def validate_source() -> dict[str, Any]:
    require(
        stat.S_IMODE(SOURCE.stat().st_mode) == 0o555,
        "package 0005 mode changed",
    )
    records: dict[str, str] = {}
    for line in (SOURCE / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name not in records,
            "package 0005 checksum manifest changed",
        )
        records[name] = digest
    observed = {
        item.relative_to(SOURCE).as_posix()
        for item in SOURCE.rglob("*")
        if item.is_file() and item.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    }
    require(observed == set(records), "package 0005 sealed member set changed")
    for name, digest in records.items():
        path = SOURCE / name
        require(
            not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"package 0005 member changed: {path}",
        )
    package_root = sha256_file(SOURCE / "SHA256SUMS")
    require(
        (SOURCE / "TREE_ROOT.sha256").read_text(encoding="ascii").split()
        == [package_root, "SHA256SUMS"],
        "package 0005 tree root sidecar changed",
    )
    require(
        package_root
        == "8b6f80799134ad053c1a857f38f7e6413ef89ce8ee175e75330a8e2e67b2e5b5",
        "package 0005 canonical tree root changed",
    )

    require(
        SOURCE_REVIEW_NAMESPACE.is_dir()
        and not SOURCE_REVIEW_NAMESPACE.is_symlink()
        and stat.S_IMODE(SOURCE_REVIEW_NAMESPACE.stat().st_mode) == 0o755
        and {item.name for item in SOURCE_REVIEW_NAMESPACE.iterdir()}
        == {SOURCE_REVIEW.name},
        "package 0005 review namespace changed",
    )
    require(
        SOURCE_REVIEW.is_file()
        and not SOURCE_REVIEW.is_symlink()
        and stat.S_IMODE(SOURCE_REVIEW.stat().st_mode) == 0o444,
        "package 0005 adjudication mode changed",
    )
    review_sha256 = sha256_file(SOURCE_REVIEW)
    require(
        review_sha256
        == "5f0a41d2889e1aa77bf18b9aecfeb262ffd727ef7e117c65a14b2a72f574e20a",
        "package 0005 adjudication changed",
    )

    require(
        SOURCE_RECEIPT.is_file()
        and not SOURCE_RECEIPT.is_symlink()
        and stat.S_IMODE(SOURCE_RECEIPT.stat().st_mode) == 0o444,
        "package 0005 external receipt mode changed",
    )
    receipt = load_json(SOURCE_RECEIPT)
    receipt_sha256 = sha256_file(SOURCE_RECEIPT)
    require(
        receipt.get("schema") == "argus-subagent-reviewer-receipt-v1"
        and receipt.get("state") == "done"
        and receipt.get("exit_code") == 0
        and receipt.get("producer_role") == "reviewer"
        and receipt.get("provenance", {}).get("session_id") == ENGINEER_SESSION_ID
        and receipt.get("adjudication")
        == {
            "path": SOURCE_REVIEW.relative_to(ROOT).as_posix(),
            "sha256": review_sha256,
        }
        and receipt.get("evidence", {}).get("package", {}).get(
            "tree_root_sha256"
        )
        == package_root,
        "package 0005 receipt no longer proves the frozen identity mismatch",
    )
    require(
        not os.path.lexists(SOURCE_GRANT)
        and not os.path.lexists(SOURCE_CONSUMPTION),
        "package 0005 grant or consumption unexpectedly exists",
    )
    return {
        "package_root": package_root,
        "receipt_sha256": receipt_sha256,
        "review_sha256": review_sha256,
    }


def failed_predecessor_binding(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "consumption_namespace_absent": True,
        "declared_producer_role": "reviewer",
        "disposition": (
            "PRESERVED_FAILED_PRECOMPUTATION_REVIEWER_IDENTITY_PROVENANCE"
        ),
        "failure_class": "REVIEWER_RECEIPT_ENGINEER_IDENTITY_REUSE",
        "framework_reviewer_session_id": FRAMEWORK_REVIEWER_SESSION_ID,
        "grant_namespace_absent": True,
        "live_authorization_predicate": False,
        "observed_receipt_session_id": ENGINEER_SESSION_ID,
        "package_member_mode": "0444",
        "package_mode": "0555",
        "package_path": SOURCE.relative_to(ROOT).as_posix(),
        "package_tree_root_sha256": evidence["package_root"],
        "producer_role_claim_sufficient": False,
        "review_mode": "0444",
        "review_namespace_mode": "0755",
        "review_path": SOURCE_REVIEW.relative_to(ROOT).as_posix(),
        "review_sha256": evidence["review_sha256"],
        "reviewer_receipt_mode": "0444",
        "reviewer_receipt_path": SOURCE_RECEIPT.relative_to(ROOT).as_posix(),
        "reviewer_receipt_schema": "argus-subagent-reviewer-receipt-v1",
        "reviewer_receipt_sha256": evidence["receipt_sha256"],
        "reviewer_receipt_state": "done",
        "schema": "ace2-stage1-layer16-failed-predecessor-provenance-v3",
        "violated_requirement": (
            "DURABLE_EXTERNAL_REVIEWER_PROCESS_IDENTITY_MUST_EQUAL_FRAMEWORK_"
            "REVIEWER_AND_DIFFER_FROM_ENGINEER_IDENTITIES"
        ),
    }


def transform_executor(evidence: dict[str, Any]) -> str:
    text = (SOURCE / "execute_layer16.py").read_text(encoding="utf-8")
    text = text.replace(
        "ace2-layer16-runtime-pass-authority-0005",
        "ace2-layer16-runtime-pass-authority-0006",
    )
    text = text.replace(
        "ace2-layer16-runtime-pass-authority-review-0005",
        "ace2-layer16-runtime-pass-authority-review-0006",
    )
    text = text.replace(
        "ace2-layer16-runtime-pass-manager-grant-0005",
        "ace2-layer16-runtime-pass-manager-grant-0006",
    )
    text = text.replace(
        "ace2-layer16-continuation-0010-consumption-state-0005",
        "ace2-layer16-continuation-0010-consumption-state-0006",
    )
    text = text.replace(
        "runtime-pass-authority-0005",
        "runtime-pass-authority-0006",
    )

    constants = f'''FAILED_PREDECESSOR_PACKAGE = (
    ROOT / "reports/ace2-layer16-runtime-pass-authority-0005"
)
FAILED_PREDECESSOR_TREE_ROOT = "{evidence["package_root"]}"
FAILED_PREDECESSOR_REVIEW_NAMESPACE = (
    ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0005"
)
FAILED_PREDECESSOR_REVIEW = (
    FAILED_PREDECESSOR_REVIEW_NAMESPACE / "reviewer-adjudication.json"
)
FAILED_PREDECESSOR_REVIEW_SHA256 = "{evidence["review_sha256"]}"
FAILED_PREDECESSOR_RECEIPT = (
    ROOT
    / ".argus_subagents"
    / "ace2-layer16-runtime-pass-authority-review-0005-reviewer-receipt.json"
)
FAILED_PREDECESSOR_RECEIPT_SHA256 = "{evidence["receipt_sha256"]}"
FAILED_PREDECESSOR_GRANT_NAMESPACE = (
    ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0005"
)
FAILED_PREDECESSOR_CONSUMPTION = (
    ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0005"
)

REVIEW_RECEIPT = (
    ROOT
    / ".argus_subagents"
    / "ace2-layer16-runtime-pass-authority-review-0006-reviewer-receipt.json"
)
FRAMEWORK_REVIEWER_SESSION_ID = "{FRAMEWORK_REVIEWER_SESSION_ID}"
PACKAGE_CREATOR_PROCESS_SESSION_ID = "{ENGINEER_SESSION_ID}"
TASK_ENGINEER_PROCESS_SESSION_IDENTITIES = (
    "{ENGINEER_SESSION_ID}",
)

'''
    text = replace_section(
        text,
        "FAILED_PREDECESSOR_PACKAGE = (",
        "LAYER15_ROOT =",
        constants,
    )

    failed_binding = '''def _failed_predecessor_binding() -> dict[str, Any]:
    return {
        "consumption_namespace_absent": True,
        "declared_producer_role": "reviewer",
        "disposition": (
            "PRESERVED_FAILED_PRECOMPUTATION_REVIEWER_IDENTITY_PROVENANCE"
        ),
        "failure_class": "REVIEWER_RECEIPT_ENGINEER_IDENTITY_REUSE",
        "framework_reviewer_session_id": FRAMEWORK_REVIEWER_SESSION_ID,
        "grant_namespace_absent": True,
        "live_authorization_predicate": False,
        "observed_receipt_session_id": PACKAGE_CREATOR_PROCESS_SESSION_ID,
        "package_path": FAILED_PREDECESSOR_PACKAGE.relative_to(ROOT).as_posix(),
        "package_mode": "0555",
        "package_member_mode": "0444",
        "package_tree_root_sha256": FAILED_PREDECESSOR_TREE_ROOT,
        "producer_role_claim_sufficient": False,
        "review_path": FAILED_PREDECESSOR_REVIEW.relative_to(ROOT).as_posix(),
        "review_mode": "0444",
        "review_namespace_mode": "0755",
        "review_sha256": FAILED_PREDECESSOR_REVIEW_SHA256,
        "reviewer_receipt_mode": "0444",
        "reviewer_receipt_path": (
            FAILED_PREDECESSOR_RECEIPT.relative_to(ROOT).as_posix()
        ),
        "reviewer_receipt_schema": "argus-subagent-reviewer-receipt-v1",
        "reviewer_receipt_sha256": FAILED_PREDECESSOR_RECEIPT_SHA256,
        "reviewer_receipt_state": "done",
        "violated_requirement": (
            "DURABLE_EXTERNAL_REVIEWER_PROCESS_IDENTITY_MUST_EQUAL_FRAMEWORK_"
            "REVIEWER_AND_DIFFER_FROM_ENGINEER_IDENTITIES"
        ),
    }


'''
    text = replace_section(
        text,
        "def _failed_predecessor_binding() -> dict[str, Any]:",
        "def _exact_invocation() -> dict[str, Any]:",
        failed_binding,
    )

    invocation = '''def _exact_invocation() -> dict[str, Any]:
    return {
        "argv": [
            "/usr/bin/env",
            "-i",
            "LC_ALL=C",
            "PATH=/home/argustest/miniconda3/bin:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONCOERCECLOCALE=0",
            "PYTHONHASHSEED=0",
            "PYTHONSAFEPATH=1",
            "PYTHONUTF8=1",
            "/home/argustest/miniconda3/bin/python3.13",
            "-B",
            "reports/ace2-layer16-runtime-pass-authority-0006/execute_layer16.py",
            "--consume-authority",
            "--unit-key",
            UNIT,
            "--review-artifact",
            REVIEW.relative_to(ROOT).as_posix(),
            "--review-artifact-sha256",
            "<FRESH_REVIEW_ARTIFACT_SHA256>",
            "--reviewer-receipt",
            REVIEW_RECEIPT.relative_to(ROOT).as_posix(),
            "--reviewer-receipt-sha256",
            "<DURABLE_FRAMEWORK_REVIEWER_RECEIPT_SHA256>",
            "--manager-grant",
            GRANT.relative_to(ROOT).as_posix(),
            "--manager-grant-sha256",
            "<FRAMEWORK_MANAGER_GRANT_SHA256>",
        ],
        "cwd": str(ROOT),
        "placeholder_policy": (
            "ONLY_THE_THREE_EXACT_ARTIFACT_SHA256_VALUES_MAY_REPLACE_"
            "THEIR_PLACEHOLDERS"
        ),
    }


'''
    text = replace_section(
        text,
        "def _exact_invocation() -> dict[str, Any]:",
        "def authority_record() -> dict[str, Any]:",
        invocation,
    )
    text = text.replace(
        '''        "acceptance_artifact_targets": [
            REVIEW.relative_to(ROOT).as_posix(),
            GRANT.relative_to(ROOT).as_posix(),
        ],''',
        '''        "acceptance_artifact_targets": [
            REVIEW.relative_to(ROOT).as_posix(),
            REVIEW_RECEIPT.relative_to(ROOT).as_posix(),
            GRANT.relative_to(ROOT).as_posix(),
        ],''',
    )
    text = text.replace(
        '''            "creator_role": "engineer",
            "fresh_review_effect": "PACKAGE_ACCEPTANCE_ONLY",''',
        '''            "creator_process_session_id": PACKAGE_CREATOR_PROCESS_SESSION_ID,
            "creator_role": "engineer",
            "fresh_review_effect": "PACKAGE_ACCEPTANCE_ONLY",''',
    )
    text = text.replace(
        "SEALED_CANDIDATE_AWAITING_SOURCE_DISJOINT_REVIEW_AND_MANAGER_GRANT",
        "SEALED_CANDIDATE_AWAITING_RECEIPT_BACKED_REVIEW_AND_MANAGER_GRANT",
    )
    text = text.replace(
        '''            "ONE_POSITION00_LAYER16_PRODUCTION_UNIT_ONLY_AFTER_SOURCE_DISJOINT_"
            "REVIEW_PASS_AND_MANAGER_API_CARDINALITY_ONE_GRANT"
''',
        '''            "ONE_POSITION00_LAYER16_PRODUCTION_UNIT_ONLY_AFTER_DURABLE_EXTERNAL_"
            "REVIEWER_RECEIPT_PASS_AND_MANAGER_API_CARDINALITY_ONE_GRANT"
''',
    )

    acceptance = '''def acceptance_contract_record() -> dict[str, Any]:
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **ZERO_WORKLOAD,
        },
        "authority_identity": IDENTITY,
        "create_exclusive": {
            "consumption": True,
            "manager_grant": True,
            "package": True,
            "review": True,
            "reviewer_receipt": True,
        },
        "fresh_reviewer_acceptance": {
            "adjudication": {
                "effect": "PACKAGE_ACCEPTANCE_ONLY",
                "path": REVIEW.relative_to(ROOT).as_posix(),
                "required_decision": "PASS",
                "required_level": "L2",
                "schema": "ace2-stage1-layer16-runtime-pass-review-adjudication-v3",
            },
            "durable_external_process_receipt": {
                "adjudication_sha256_binding": "REQUIRED",
                "completed_state": "done",
                "package_root_binding": "REQUIRED_BARE_TOKEN_EXACT_EQUALITY",
                "path": REVIEW_RECEIPT.relative_to(ROOT).as_posix(),
                "producer_role_claim_sufficient": False,
                "required_process_session_identity": FRAMEWORK_REVIEWER_SESSION_ID,
                "schema": "argus-subagent-reviewer-receipt-v1",
            },
            "excluded_process_session_identities": sorted(
                {
                    PACKAGE_CREATOR_PROCESS_SESSION_ID,
                    *TASK_ENGINEER_PROCESS_SESSION_IDENTITIES,
                }
            ),
            "required_source_disjoint_from_creator": True,
        },
        "failed_predecessor_evidence": _failed_predecessor_binding(),
        "live_predicate_policy": LIVE_PREDICATE_POLICY,
        "manager_acceptance": {
            "fresh_review_sha256_binding": "REQUIRED",
            "fresh_reviewer_receipt_sha256_binding": "REQUIRED",
            "origin": "IMMUTABLE_FRAMEWORK_MANAGER_HANDOFF",
            "path": GRANT.relative_to(ROOT).as_posix(),
            "required_authority_cardinality": 1,
            "required_decision": "AUTHORIZE_EXACTLY_ONE_LAYER16_UNIT",
            "required_execution_limit": 1,
            "required_producer_role": "manager",
            "requirements": _manager_requirement(),
            "schema": "ace2-stage1-layer16-framework-manager-grant-v3",
        },
        "ordering": [
            "CREATE_EXCLUSIVE_AND_SEAL_LAYER16_PACKAGE",
            "EXTERNAL_FRAMEWORK_REVIEWER_PROCESS_CREATES_BOUND_PASS_AND_RECEIPT",
            "MANAGER_API_CREATE_EXCLUSIVE_GRANT_BINDS_REVIEW_RECEIPT_AND_DIRECTIVE",
            "CONSUME_ONCE_BEFORE_LAYER16_EXECUTION",
        ],
        "package_cardinality": 1,
        "package_tree_binding": {
            "algorithm": "SHA256",
            "external_records_must_equal_tree_root": True,
            "package_path": PACKAGE.relative_to(ROOT).as_posix(),
            "review_subject_contract": {
                "field": "subject.package_tree_root_sha256",
                "pattern": "^[0-9a-f]{64}$",
                "rejected_representations": [
                    "ALGORITHM_OR_FIELD_LABELS",
                    "FILENAMES",
                    "LEADING_OR_TRAILING_WHITESPACE",
                    "NEWLINE_SUFFIXES",
                    "SHA256SUM_FORMATTED_LINES",
                ],
                "required_format": "LOWERCASE_BARE_64_HEX_ONLY",
                "required_relation": (
                    "EXACT_EQUALITY_TO_CANONICAL_PACKAGE_TREE_ROOT_TOKEN"
                ),
            },
            "tree_root_sidecar": (
                PACKAGE / "TREE_ROOT.sha256"
            ).relative_to(ROOT).as_posix(),
        },
        "position00_layer16_frontier": _frontier_binding(),
        "schema": "ace2-stage1-layer16-acceptance-contract-v3",
        "unit_key": UNIT,
    }


'''
    text = replace_section(
        text,
        "def acceptance_contract_record() -> dict[str, Any]:",
        "def review_request_record() -> dict[str, Any]:",
        acceptance,
    )

    review_request = '''def review_request_record() -> dict[str, Any]:
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **ZERO_WORKLOAD,
        },
        "decision_requested": "PASS",
        "package_path": PACKAGE.relative_to(ROOT).as_posix(),
        "required_checks": [
            "cardinality-one-layer16-specific-create-exclusive-package",
            "sealed-package-member-set-digests-modes-and-root",
            "exact-live-predicate-set-manager-directive-checkpoint-and-unit",
            "no-historical-attempt-or-pre-correction-adjudication-live-predicates",
            "corrected-checkpoint-position00-layer16-frontier",
            "external-framework-reviewer-process-receipt-is-complete-and-bound",
            "reviewer-process-identity-differs-from-creator-and-engineer-identities",
            "self-declared-producer-role-is-insufficient",
            "exact-layer16-only-sealed-invocation",
            "package-native-hostile-controls-pass-with-zero-workload",
            "review-subject-package-root-is-exact-bare-lowercase-64-hex-token",
            "reject-root-labels-filenames-whitespace-newlines-and-sha256sum-lines",
            "failed-0005-package-review-and-receipt-preserved-as-nonauthorizing",
            "zero-authority-consumption-and-zero-model-reference-rtl-unit-token-ppa-u280",
        ],
        "preserved_failed_predecessor": _failed_predecessor_binding(),
        "review_effect": (
            "ACCEPT_EXACT_LAYER16_PACKAGE_WITHOUT_CREATING_OR_CONSUMING_"
            "EXECUTION_AUTHORITY"
        ),
        "review_level": "L2",
        "review_type": (
            "FRESH_RECEIPT_BACKED_SOURCE_DISJOINT_LAYER16_RUNTIME_PASS_PACKAGE_REVIEW"
        ),
        "schema": "ace2-stage1-layer16-runtime-pass-review-request-v3",
        "subject": {
            "authority_identity": IDENTITY,
            "unit_key": UNIT,
        },
    }


'''
    text = replace_section(
        text,
        "def review_request_record() -> dict[str, Any]:",
        "def _external_subject(package_root: str) -> dict[str, Any]:",
        review_request,
    )
    text = text.replace(
        '"schema": "ace2-stage1-layer16-runtime-pass-review-adjudication-v2",',
        '"schema": "ace2-stage1-layer16-runtime-pass-review-adjudication-v3",',
    )

    review_validation = '''def validate_review_record(
    record: dict[str, Any],
    package_root: str,
    hostile_controls_sha256: str,
) -> None:
    subject = record.get("subject")
    require(isinstance(subject, dict), "review subject object required")
    root_token = subject.get("package_tree_root_sha256")
    require(
        isinstance(root_token, str)
        and re.fullmatch(r"[0-9a-f]{64}", root_token) is not None,
        "review subject package root must be one bare lowercase 64-hex token",
    )
    require(
        root_token == package_root,
        "review subject package root must exactly equal the canonical package root",
    )
    require(
        record == expected_review_record(package_root, hostile_controls_sha256),
        "exact Fresh Reviewer PASS required",
    )


def expected_reviewer_receipt_record(
    review_sha256: str,
    package_root: str,
) -> dict[str, Any]:
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
        "evidence": {
            "package": {
                "path": PACKAGE.relative_to(ROOT).as_posix(),
                "tree_root_sha256": package_root,
            }
        },
        "exit_code": 0,
        "mode": "direct-reviewer-nonexecuting",
        "producer_role": "reviewer",
        "provenance": {
            "session_id": FRAMEWORK_REVIEWER_SESSION_ID,
            "source_disjoint_from_package_creator": True,
        },
        "run_id": (
            "ace2-layer16-runtime-pass-authority-review-0006-"
            + FRAMEWORK_REVIEWER_SESSION_ID
        ),
        "schema": "argus-subagent-reviewer-receipt-v1",
        "state": "done",
        "task_id": "ace2-layer16-runtime-pass-authority-review-0006-reviewer",
    }


def validate_reviewer_receipt_record(
    record: object,
    review_sha256: str,
    package_root: str,
) -> None:
    require(isinstance(record, dict), "durable external Reviewer receipt required")
    receipt = record
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
    require(
        isinstance(package, dict)
        and package.get("path") == PACKAGE.relative_to(ROOT).as_posix()
        and package.get("tree_root_sha256") == package_root,
        "Reviewer receipt must bind the exact canonical package root",
    )
    provenance = receipt.get("provenance")
    require(isinstance(provenance, dict), "Reviewer process provenance required")
    reviewer_identity = provenance.get("session_id")
    excluded = {
        PACKAGE_CREATOR_PROCESS_SESSION_ID,
        *TASK_ENGINEER_PROCESS_SESSION_IDENTITIES,
    }
    require(
        reviewer_identity == FRAMEWORK_REVIEWER_SESSION_ID
        and reviewer_identity not in excluded,
        "Reviewer process identity must be the disjoint framework Reviewer",
    )
    require(
        receipt.get("run_id")
        == (
            "ace2-layer16-runtime-pass-authority-review-0006-"
            + FRAMEWORK_REVIEWER_SESSION_ID
        )
        and receipt.get("task_id")
        == "ace2-layer16-runtime-pass-authority-review-0006-reviewer",
        "durable Reviewer task and run identity mismatch",
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
    text = replace_section(
        text,
        "def validate_review_record(",
        "def _directive_binding(record: dict[str, Any]) -> dict[str, Any]:",
        review_validation,
    )

    expected_grant = '''def expected_grant_record(
    package_root: str,
    review_sha256: str,
    reviewer_receipt_sha256: str,
    directive: dict[str, Any],
) -> dict[str, Any]:
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **ZERO_WORKLOAD,
        },
        "authority_cardinality": 1,
        "consumption_state": "UNCONSUMED",
        "decision": "AUTHORIZE_EXACTLY_ONE_LAYER16_UNIT",
        "exact_invocation": _exact_invocation(),
        "execution_limit": 1,
        "execution_performed": False,
        "fresh_review": {
            "path": REVIEW.relative_to(ROOT).as_posix(),
            "sha256": review_sha256,
        },
        "fresh_reviewer_receipt": {
            "path": REVIEW_RECEIPT.relative_to(ROOT).as_posix(),
            "sha256": reviewer_receipt_sha256,
        },
        "manager_directive": directive,
        "origin": "IMMUTABLE_FRAMEWORK_MANAGER_HANDOFF",
        "producer_role": "manager",
        "schema": "ace2-stage1-layer16-framework-manager-grant-v3",
        "subject": _external_subject(package_root),
    }


'''
    text = replace_section(
        text,
        "def expected_grant_record(",
        "def _validate_namespace(",
        expected_grant,
    )

    old_bound = '''    _validate_file(
        FAILED_PREDECESSOR_REVIEW,
        FAILED_PREDECESSOR_REVIEW_SHA256,
    )
    validate_sealed_tree(SOURCE_PACKAGE, SOURCE_PACKAGE_ROOT)'''
    new_bound = '''    _validate_file(
        FAILED_PREDECESSOR_REVIEW,
        FAILED_PREDECESSOR_REVIEW_SHA256,
    )
    _validate_file(
        FAILED_PREDECESSOR_RECEIPT,
        FAILED_PREDECESSOR_RECEIPT_SHA256,
    )
    failed_receipt = load_json(FAILED_PREDECESSOR_RECEIPT)
    require(
        failed_receipt.get("schema") == "argus-subagent-reviewer-receipt-v1"
        and failed_receipt.get("state") == "done"
        and failed_receipt.get("producer_role") == "reviewer"
        and failed_receipt.get("provenance", {}).get("session_id")
        == PACKAGE_CREATOR_PROCESS_SESSION_ID
        and failed_receipt.get("adjudication")
        == {
            "path": FAILED_PREDECESSOR_REVIEW.relative_to(ROOT).as_posix(),
            "sha256": FAILED_PREDECESSOR_REVIEW_SHA256,
        }
        and failed_receipt.get("evidence", {}).get("package", {}).get(
            "tree_root_sha256"
        )
        == FAILED_PREDECESSOR_TREE_ROOT,
        "failed 0005 Reviewer identity provenance changed",
    )
    require(
        not os.path.lexists(FAILED_PREDECESSOR_GRANT_NAMESPACE)
        and not os.path.lexists(FAILED_PREDECESSOR_CONSUMPTION),
        "failed 0005 grant or consumption unexpectedly exists",
    )
    validate_sealed_tree(SOURCE_PACKAGE, SOURCE_PACKAGE_ROOT)'''
    require(old_bound in text, "bound evidence insertion point changed")
    text = text.replace(old_bound, new_bound)

    live_snapshot = '''def live_snapshot() -> dict[str, Any]:
    records = {
        str(path): sha256_file(path)
        for path in (
            MANAGER_DIRECTIVE,
            CHECKPOINT,
            FAILED_PREDECESSOR_PACKAGE / "SHA256SUMS",
            FAILED_PREDECESSOR_PACKAGE / "TREE_ROOT.sha256",
            FAILED_PREDECESSOR_REVIEW,
            FAILED_PREDECESSOR_RECEIPT,
        )
    }
    return {
        "bound_records": records,
        "consumption_exists": os.path.lexists(CONSUMPTION),
        "grant_namespace_exists": os.path.lexists(GRANT_NAMESPACE),
        "layer16_pending": os.path.lexists(LIVE_RUNTIME / ".pending" / UNIT),
        "layer16_published": os.path.lexists(LIVE_RUNTIME / "published" / UNIT),
        "layer16_work": os.path.lexists(LIVE_RUNTIME / "work" / UNIT),
        "review_namespace_exists": os.path.lexists(REVIEW_NAMESPACE),
        "reviewer_receipt_exists": os.path.lexists(REVIEW_RECEIPT),
    }


'''
    text = replace_section(
        text,
        "def live_snapshot() -> dict[str, Any]:",
        "def validate_preflight(",
        live_snapshot,
    )

    preflight = '''def validate_preflight(
    authority: dict[str, Any],
    *,
    current_directive_required: bool = True,
) -> dict[str, str]:
    require(authority == authority_record(), "authority contract changed")
    require(
        set(authority["bindings"])
        == {
            "acceptance_contract",
            "corrected_checkpoint",
            "failed_predecessor_evidence",
            "manager_corrected_directive",
        },
        "live authorization predicate surface changed",
    )
    require(
        authority["live_predicate_policy"] == LIVE_PREDICATE_POLICY,
        "live predicate policy changed",
    )
    require(
        load_json(ACCEPTANCE) == acceptance_contract_record(),
        "acceptance contract changed",
    )
    require(
        load_json(PACKAGE / "review-request.json") == review_request_record(),
        "review request changed",
    )
    require(
        load_json(PACKAGE / "failed-predecessor-provenance.json")
        == {
            **_failed_predecessor_binding(),
            "activity": {
                "authority_consumed": 0,
                "authority_created": 0,
                **ZERO_WORKLOAD,
            },
            "schema": "ace2-stage1-layer16-failed-predecessor-provenance-v3",
        },
        "failed 0005 provenance record changed",
    )
    review_namespace_absent = not os.path.lexists(REVIEW_NAMESPACE)
    reviewer_receipt_absent = not os.path.lexists(REVIEW_RECEIPT)
    _validate_namespace(REVIEW_NAMESPACE, REVIEW, allow_absent=True)
    if os.path.lexists(REVIEW_RECEIPT):
        require(
            REVIEW_RECEIPT.is_file() and not REVIEW_RECEIPT.is_symlink(),
            "durable external Reviewer receipt is invalid",
        )
    if os.path.lexists(GRANT_NAMESPACE):
        _validate_namespace(GRANT_NAMESPACE, GRANT)
    _validate_bound_evidence(
        current_directive_required=current_directive_required,
    )
    require(not os.path.lexists(CONSUMPTION), "layer-16 authority already consumed")
    for candidate in (
        LIVE_RUNTIME / ".pending" / UNIT,
        LIVE_RUNTIME / "published" / UNIT,
        LIVE_RUNTIME / "work" / UNIT,
    ):
        require(not os.path.lexists(candidate), f"stale layer-16 path exists: {candidate}")
    return {
        "cardinality_one": "PASS",
        "corrected_checkpoint": "PASS",
        "create_exclusive": "PASS",
        "exact_invocation": "PASS",
        "failed_0005_provenance": "PASS_PRESERVED_NONAUTHORIZING",
        "historical_attempt_live_predicates": "PASS_ABSENT",
        "manager_corrected_directive": "PASS",
        "pre_correction_adjudication_live_predicates": "PASS_ABSENT",
        "review_namespace": (
            "PASS_ABSENT_PRISTINE"
            if review_namespace_absent
            else "PASS_PRESENT_EXACT_SET"
        ),
        "reviewer_receipt": (
            "PASS_ABSENT_PRISTINE"
            if reviewer_receipt_absent
            else "PASS_PRESENT_REGULAR_FILE"
        ),
        "source_execution_package": "PASS",
        "unit_scope": "PASS_POSITION00_LAYER16_ONLY",
    }


'''
    text = replace_section(
        text,
        "def validate_preflight(",
        "def _validate_review(",
        preflight,
    )

    review_helpers = '''def _validate_review(
    path: Path,
    supplied: str,
    receipt_path: Path,
    supplied_receipt: str,
    package_root: str,
) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{64}", supplied) is not None, "invalid review digest")
    require(
        re.fullmatch(r"[0-9a-f]{64}", supplied_receipt) is not None,
        "invalid Reviewer receipt digest",
    )
    _validate_file(path, supplied)
    _validate_file(receipt_path, supplied_receipt)
    require(
        stat.S_IMODE(receipt_path.stat().st_mode) == 0o444,
        "durable Reviewer receipt must be read-only",
    )
    controls = parse_sums(PACKAGE / "SHA256SUMS")
    record = load_json(path)
    validate_review_record(
        record,
        package_root,
        controls["hostile-controls.json"],
    )
    validate_reviewer_receipt_record(
        load_json(receipt_path),
        supplied,
        package_root,
    )
    return record


'''
    text = replace_section(
        text,
        "def _validate_review(",
        "def _validate_grant(",
        review_helpers,
    )

    grant_helper = '''def _validate_grant(
    path: Path,
    supplied: str,
    package_root: str,
    review_sha256: str,
    reviewer_receipt_sha256: str,
) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{64}", supplied) is not None, "invalid grant digest")
    _validate_file(path, supplied)
    record = load_json(path)
    directive = _directive_binding(record)
    require(
        record
        == expected_grant_record(
            package_root,
            review_sha256,
            reviewer_receipt_sha256,
            directive,
        ),
        "exact framework-origin layer-16 Manager grant required",
    )
    return record


'''
    text = replace_section(
        text,
        "def _validate_grant(",
        "def _write_create_only(",
        grant_helper,
    )

    old_args = '''        len(sys.argv) == 12
        and sys.argv[1:4] == ["--consume-authority", "--unit-key", UNIT]
        and sys.argv[4] == "--review-artifact"
        and Path(sys.argv[5]) == REVIEW.relative_to(ROOT)
        and sys.argv[6] == "--review-artifact-sha256"
        and sys.argv[8] == "--manager-grant"
        and Path(sys.argv[9]) == GRANT.relative_to(ROOT)
        and sys.argv[10] == "--manager-grant-sha256",'''
    new_args = '''        len(sys.argv) == 16
        and sys.argv[1:4] == ["--consume-authority", "--unit-key", UNIT]
        and sys.argv[4] == "--review-artifact"
        and Path(sys.argv[5]) == REVIEW.relative_to(ROOT)
        and sys.argv[6] == "--review-artifact-sha256"
        and sys.argv[8] == "--reviewer-receipt"
        and Path(sys.argv[9]) == REVIEW_RECEIPT.relative_to(ROOT)
        and sys.argv[10] == "--reviewer-receipt-sha256"
        and sys.argv[12] == "--manager-grant"
        and Path(sys.argv[13]) == GRANT.relative_to(ROOT)
        and sys.argv[14] == "--manager-grant-sha256",'''
    require(old_args in text, "execution argument contract changed")
    text = text.replace(old_args, new_args)
    text = text.replace(
        '''    _validate_review(REVIEW, sys.argv[7], package_root)
    _validate_grant(GRANT, sys.argv[11], package_root, sys.argv[7])''',
        '''    _validate_review(
        REVIEW,
        sys.argv[7],
        REVIEW_RECEIPT,
        sys.argv[11],
        package_root,
    )
    _validate_grant(
        GRANT,
        sys.argv[15],
        package_root,
        sys.argv[7],
        sys.argv[11],
    )''',
    )
    text = text.replace('"grant_sha256": sys.argv[11],', '"grant_sha256": sys.argv[15],')
    text = text.replace(
        '"review_sha256": sys.argv[7],',
        '"review_sha256": sys.argv[7],\n            "reviewer_receipt_sha256": sys.argv[11],',
    )
    text = text.replace(
        '"schema": "ace2-stage1-layer16-framework-manager-grant-v2",',
        '"schema": "ace2-stage1-layer16-framework-manager-grant-v3",',
    )
    text = text.replace(
        '"schema": "ace2-stage1-layer16-runtime-pass-authority-v2",',
        '"schema": "ace2-stage1-layer16-runtime-pass-authority-v3",',
    )
    require(
        "def validate_reviewer_receipt_record(" in text
        and "DURABLE_FRAMEWORK_REVIEWER_RECEIPT_SHA256" in text
        and "sys.argv[15]" in text
        and "failed-0004" not in text
        and "SOURCE_DISJOINT_REVIEW_AND_MANAGER_GRANT" not in text,
        "generated executor is missing the 0006 receipt-backed contract",
    )
    compile(text, "execute_layer16.py", "exec")
    return text


def transform_hostile_controls() -> str:
    text = (SOURCE / "hostile_controls.py").read_text(encoding="utf-8")
    text = text.replace("package 0005", "package 0006")
    insertion = '''    valid_receipt = layer16.expected_reviewer_receipt_record(
        "c" * 64,
        canonical_root,
    )
    layer16.validate_reviewer_receipt_record(
        valid_receipt,
        "c" * 64,
        canonical_root,
    )
    cases.append(
        expect_rejection(
            "receipt-missing",
            lambda: layer16.validate_reviewer_receipt_record(
                None,
                "c" * 64,
                canonical_root,
            ),
        )
    )
    cases.append(
        expect_rejection(
            "receipt-malformed",
            lambda: layer16.validate_reviewer_receipt_record(
                [],
                "c" * 64,
                canonical_root,
            ),
        )
    )
    for name, mutate in (
        (
            "receipt-unbound",
            lambda value: value["adjudication"].__setitem__(
                "path",
                "reports/unbound/reviewer-adjudication.json",
            ),
        ),
        (
            "receipt-tampered",
            lambda value: value["adjudication"].__setitem__("sha256", "0" * 64),
        ),
        (
            "receipt-engineer-authored",
            lambda value: (
                value.__setitem__("producer_role", "engineer"),
                value["provenance"].__setitem__(
                    "session_id",
                    layer16.TASK_ENGINEER_PROCESS_SESSION_IDENTITIES[0],
                ),
            ),
        ),
        (
            "receipt-creator-authored",
            lambda value: value["provenance"].__setitem__(
                "session_id",
                layer16.PACKAGE_CREATOR_PROCESS_SESSION_ID,
            ),
        ),
        (
            "receipt-identity-reused",
            lambda value: (
                value["provenance"].__setitem__(
                    "session_id",
                    layer16.TASK_ENGINEER_PROCESS_SESSION_IDENTITIES[0],
                ),
                value.__setitem__(
                    "run_id",
                    "ace2-layer16-runtime-pass-authority-review-0006-"
                    + layer16.TASK_ENGINEER_PROCESS_SESSION_IDENTITIES[0],
                ),
            ),
        ),
    ):
        receipt = copy.deepcopy(valid_receipt)
        mutate(receipt)
        cases.append(
            expect_rejection(
                name,
                lambda receipt=receipt: layer16.validate_reviewer_receipt_record(
                    receipt,
                    "c" * 64,
                    canonical_root,
                ),
            )
        )
    cases.append(
        expect_rejection(
            "receipt-self-declared-only",
            lambda: layer16.validate_reviewer_receipt_record(
                {
                    "producer_role": "reviewer",
                    "source_disjoint_from_package_creator": True,
                },
                "c" * 64,
                canonical_root,
            ),
        )
    )

'''
    marker = "    after = layer16.live_snapshot()\n"
    require(text.count(marker) == 1, "hostile receipt insertion point changed")
    text = text.replace(marker, insertion + marker)
    text = text.replace(
        '''        and before["review_namespace_exists"] is False,''',
        '''        and before["review_namespace_exists"] is False
        and before["reviewer_receipt_exists"] is False,''',
    )
    text = text.replace(
        '"schema": "ace2-stage1-layer16-hostile-controls-v2",',
        '"schema": "ace2-stage1-layer16-hostile-controls-v3",',
    )
    compile(text, "hostile_controls.py", "exec")
    return text


def transform_sealer() -> str:
    text = (SOURCE / "seal_package.py").read_text(encoding="utf-8")
    text = text.replace("package 0005", "package 0006")
    first_loop = '''    for namespace in (
        layer16.REVIEW_NAMESPACE,
        layer16.GRANT_NAMESPACE,
        layer16.CONSUMPTION,
    ):
        layer16.require(
            not os.path.lexists(namespace),
            f"fresh authority namespace must remain absent: {namespace}",
        )'''
    first_replacement = first_loop + '''
    layer16.require(
        not os.path.lexists(layer16.REVIEW_RECEIPT),
        f"fresh Reviewer receipt must remain absent: {layer16.REVIEW_RECEIPT}",
    )'''
    require(text.count(first_loop) == 1, "sealer initial namespace check changed")
    text = text.replace(first_loop, first_replacement)
    second_loop = '''    for namespace in (
        layer16.REVIEW_NAMESPACE,
        layer16.GRANT_NAMESPACE,
        layer16.CONSUMPTION,
    ):
        layer16.require(
            not os.path.lexists(namespace),
            f"sealing created a forbidden authority namespace: {namespace}",
        )'''
    second_replacement = second_loop + '''
    layer16.require(
        not os.path.lexists(layer16.REVIEW_RECEIPT),
        f"sealing created forbidden Reviewer receipt: {layer16.REVIEW_RECEIPT}",
    )'''
    require(text.count(second_loop) == 1, "sealer final namespace check changed")
    text = text.replace(second_loop, second_replacement)
    compile(text, "seal_package.py", "exec")
    return text


def transform_validator() -> str:
    text = (SOURCE / "validate_nonexecuting.py").read_text(encoding="utf-8")
    text = text.replace(
        '"review_state": (',
        '''"reviewer_receipt_state": (
            "AWAITING_DURABLE_EXTERNAL_REVIEWER_RECEIPT"
            if not before["reviewer_receipt_exists"]
            else "DURABLE_EXTERNAL_REVIEWER_RECEIPT_PRESENT"
        ),
        "review_state": (''',
    )
    text = text.replace(
        '"schema": "ace2-stage1-layer16-runtime-pass-preflight-v2",',
        '"schema": "ace2-stage1-layer16-runtime-pass-preflight-v3",',
    )
    compile(text, "validate_nonexecuting.py", "exec")
    return text


def main() -> int:
    require(os.getcwd() == str(ROOT), "exact repository cwd required")
    evidence = validate_source()
    for path in (
        TARGET,
        TARGET_REVIEW,
        TARGET_RECEIPT,
        TARGET_GRANT,
        TARGET_CONSUMPTION,
    ):
        require(not os.path.lexists(path), f"fresh 0006 target already exists: {path}")

    generated = {
        "execute_layer16.py": transform_executor(evidence),
        "hostile_controls.py": transform_hostile_controls(),
        "seal_package.py": transform_sealer(),
        "validate_nonexecuting.py": transform_validator(),
    }
    provenance = {
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
        **failed_predecessor_binding(evidence),
    }

    os.mkdir(TARGET, mode=0o700)
    for name, text in generated.items():
        write_exclusive(TARGET / name, text.encode("utf-8"))
    write_exclusive(
        TARGET / "failed-predecessor-provenance.json",
        canonical_bytes(provenance),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(TARGET / "seal_package.py")],
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
    if result.returncode != 0:
        sys.stdout.buffer.write(result.stdout)
        sys.stderr.buffer.write(result.stderr)
        raise PreparationError(
            f"package 0006 sealing failed with status {result.returncode}"
        )
    root = result.stdout.decode("ascii").strip()
    require(
        len(root) == 64
        and all(character in "0123456789abcdef" for character in root),
        "sealer returned a noncanonical package root",
    )
    sys.stdout.write(root + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
