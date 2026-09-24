#!/usr/bin/env python3
"""Verify retired V21 containment and optionally publish one create-only report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
HANDOFF_ROOT = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/0d56568407e7"
)
LATEST = HANDOFF_ROOT / "latest.json"
MISSION = HANDOFF_ROOT / "mission.json"
SEALED_REVIEW_HANDOFF = HANDOFF_ROOT / "round-0001.json"
REPORT = PROJECT_ROOT / "build/v21-retirement-containment-report-0001.json"
RECONCILIATION = PROJECT_ROOT / "build/v21-mission-reconciliation-report-0001.json"
ATTEMPT_ROOT = PROJECT_ROOT / (
    "build/v21-order-independent-binding-terminal-coverage-attempt-0001"
)
ADMISSION = ATTEMPT_ROOT / "manager-v21-execution-admission-node-0d56568407e7.json"
REVIEW_REQUEST = ATTEMPT_ROOT / "fresh-reviewer-v21-terminal-classification-request.json"
REVIEW = ATTEMPT_ROOT / "fresh-reviewer-v21-terminal-classification.json"
ACTION_ROOT = PROJECT_ROOT / (
    "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_"
    "order_independent_binding_terminal_coverage_action_root"
)
PACKAGE = ACTION_ROOT / (
    "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_"
    "ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_PACKAGE.json"
)
STATIC_ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
RUNTIME_ROOT = PROJECT_ROOT / "runtime" / (
    "qk_gbfp8_head64_granularity_sweep_execution_v21_"
    "order_independent_binding_terminal_coverage_289140ba"
)
OWNER = RUNTIME_ROOT / "primary/authority/base/owner-claim.json"
AUTHORITY = RUNTIME_ROOT / "primary/authority/base/authority.json"
CREDENTIAL = RUNTIME_ROOT / "primary/authority/base/credential.json"
LEDGER = RUNTIME_ROOT / "primary/authority/base/authority-ledger.json"
TERMINAL = RUNTIME_ROOT / "primary/authority/base/first-terminal.json"
RESULT = RUNTIME_ROOT / "primary/result/base/result.json"
FALLBACK = RUNTIME_ROOT / "fallback/first-terminal.json"
ACTION_ID = "ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001"
MISSION_ID = "0d56568407e7"
PUBLISHED_REPORT_SHA256 = "450d231f3bad56ccf115be6e7b018745c24dd928dc507594db3f98023c6aa6c5"
PUBLISHED_VERIFIER_SHA256 = "2953e1136d74f1ea1d29a6c63bbf13a6b68c0b04c0161cb561fde60a0aa5fc8c"

EXPECTED_HASHES = {
    "reconciliation": "2dd6d024b93cfd47ee55f66bf859083727b95a7e86cf4c7ebf6ed68b8ea887d8",
    "admission": "6d0a74c4ef353886104330ce335fa99e8fc58a85822ddc9c5011bd103e9c5d42",
    "review_request": "e64e7eb7482753c002a5994f3b38bee03aa4992ebb84a92a125283aff80c89ef",
    "review": "2d7627d745a44e57df5488c3009235602898fed6c5390382ec760fb3d467ea38",
    "package": "45caa0015e55979dd6176a319792a3ae3a4da9f25a19baf9bec69b98de828d69",
    "static_acceptance": "19663777b84352691030325189ffb397b6386b5443200010cda0e6cd4a286c48",
    "owner": "8ff9d78d93689d095d2d64f5fe56f0e1a15632118705376c0baa874874b26ef5",
    "authority": "f6db7fb33ff3d05abb43dce9c1799a39d4321c420a99bccdfea422d419dc999d",
    "ledger": "f6bbe0b614c2f9bcc8fee2509067ca39917302409ade6dbabec694fd3611351e",
    "terminal": "77695ecbca2332e0bd3f22f979d2d0ccc640b3f0af5b934c1954c62b733236b0",
    "result": "4901c835dd9b8700cb3a3dd5ecac54f2d1112f7f2f6e909a7ad37ee35ab8941f",
}

V20_RECORDS = {
    "execution_package": (
        PROJECT_ROOT
        / "reference/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_action_root/"
        "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_BOUND_PATH_REPAIR_PACKAGE.json",
        "e1ab2b55dd07b6cfc9f8540d42eb3fd9b387cb57aadf0704933a29fe6c05c314",
    ),
    "acceptance": (
        PROJECT_ROOT
        / "reference/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_action_root/"
        "review/FRESH_L2_STATIC_ACCEPTANCE.json",
        "516f3007e0bedc00b98748d835b5240fb09b08b189ccc87286aef9ea1bfbe963",
    ),
    "terminal_observation": (
        PROJECT_ROOT
        / "build/v20-bound-path-repair-attempt-0001/sole-v20-transport-terminal-observation.json",
        "289140ba983b6808ef0a2bb56381579857ecfc50c35349e96c266ac2b0c75d33",
    ),
    "stderr": (
        PROJECT_ROOT / "build/v20-bound-path-repair-attempt-0001/sole-v20-transport-stderr.log",
        "6470257db40ccc24d46277d34818419f11cc1a4752dba8263d6c742d321a9e4d",
    ),
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


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


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_regular(path: Path) -> tuple[dict[str, Any], bytes]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"file changed while read: {path}",
        )
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    require(len(raw) == before.st_size, f"short read: {path}")
    return {
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "sha256": sha256(raw),
        "size": len(raw),
    }, raw


def read_json(path: Path, *, canonical: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    identity, raw = read_regular(path)
    value = json.loads(raw.decode("utf-8", "strict"))
    require(type(value) is dict, f"JSON object required: {path}")
    if canonical:
        require(compact_bytes(value) == raw, f"noncanonical JSON: {path}")
    return value, identity


def pinned_json(
    label: str, path: Path, *, canonical: bool = True, mode: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    value, identity = read_json(path, canonical=canonical)
    require(identity["sha256"] == EXPECTED_HASHES[label], f"{label} hash drift")
    if mode is not None:
        require(identity["mode"] == mode, f"{label} mode drift")
    identity["logical_name"] = label
    return value, identity


def has_key_fragment(value: Any, fragment: str) -> bool:
    if isinstance(value, dict):
        return any(
            fragment in str(key).lower() or has_key_fragment(item, fragment)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(has_key_fragment(item, fragment) for item in value)
    return False


def runtime_inventory() -> set[Path]:
    return {path for path in RUNTIME_ROOT.rglob("*") if path.is_file()}


def build_record() -> dict[str, Any]:
    require(os.geteuid() != 0, "containment audit must not run as root")
    require(PROJECT_ROOT.resolve(strict=True) == PROJECT_ROOT, "project namespace drift")
    require(HANDOFF_ROOT.resolve(strict=True) == HANDOFF_ROOT, "handoff namespace drift")

    latest, latest_identity = read_json(LATEST, canonical=False)
    mission_path = Path(latest["mission"]["path"])
    active_handoff_path = Path(latest["handoff"]["path"])
    require(mission_path == MISSION, "latest mission substitution")
    require(active_handoff_path.parent == HANDOFF_ROOT, "latest handoff namespace substitution")
    require(
        active_handoff_path.name.startswith("round-")
        and active_handoff_path.suffix == ".json",
        "latest handoff name",
    )

    mission, mission_identity = read_json(mission_path, canonical=False)
    active_handoff, _ = read_json(active_handoff_path, canonical=False)
    require(active_handoff["mission_id"] == MISSION_ID, "latest handoff mission identity")
    require(
        Path(active_handoff["mission_context"]) == MISSION,
        "latest handoff mission substitution",
    )

    # The Reviewer disposition is a sealed historical decision.  The latest
    # pointer is mutable workflow state and may legitimately advance to an
    # Engineer or Manager handoff after that decision.
    handoff_path = SEALED_REVIEW_HANDOFF
    handoff, handoff_identity = read_json(handoff_path, canonical=False)
    require(mission["mission_id"] == MISSION_ID, "mission identity")
    require(mission["stage"] == "rtl", "mission stage")
    require(mission["plan_id"] == "plan-2bbc1edc3cbc", "mission plan identity")
    require(mission["plan_version"] == 1, "mission plan version")
    require(mission["scope"] == "bounded", "mission scope")
    require(
        mission["objective"]
        == "Manager reconciles the stage, obtains fresh operator exactly-once authority, then an Engineer executes the accepted V21 wrapper once and records immutable terminal/result evidence without modifying V20.",
        "mission objective drift",
    )
    require(
        mission["acceptance_check"]
        == "Exactly one V21 invocation has schema-valid terminal evidence, zero V20 mutation, and Fresh Reviewer classification.",
        "mission acceptance contract drift",
    )
    require(handoff["producer_role"] == "reviewer", "handoff role")
    require(handoff["review"]["status"] == "replan_requested", "handoff disposition")
    next_action = handoff["review"]["next_action"].lower()
    require(
        "never replay v21" in next_action or "do not replay v21" in next_action,
        "handoff replay prohibition",
    )

    reconciliation, reconciliation_identity = pinned_json(
        "reconciliation", RECONCILIATION, mode="0444"
    )
    admission, admission_identity = pinned_json("admission", ADMISSION, mode="0444")
    review_request, request_identity = pinned_json("review_request", REVIEW_REQUEST, mode="0444")
    review, review_identity = pinned_json("review", REVIEW, mode="0444")
    package, package_identity = pinned_json("package", PACKAGE, mode="0444")
    static_acceptance, acceptance_identity = pinned_json(
        "static_acceptance", STATIC_ACCEPTANCE, mode="0444"
    )

    require(reconciliation["action_id"] == ACTION_ID, "reconciliation action")
    require(reconciliation["acceptance_criterion_met"] is False, "reconciliation acceptance")
    require(
        reconciliation["disposition"]
        == "HOLD_REPLAN_SEPARATE_AUTHORITY_CONTAINMENT_AND_SUCCESSOR",
        "reconciliation disposition",
    )
    require(reconciliation["execution_cardinality_this_reconciliation"] == 0, "reconciliation cardinality")
    require(reconciliation["v21_replay_permitted"] is False, "reconciliation replay policy")
    require(
        reconciliation["external_operator_authority"]["status"] == "ABSENT_AT_CONSUMPTION",
        "reconciliation authority status",
    )
    require(admission["action_id"] == ACTION_ID, "admission action")
    require(admission["status"] == "BLOCKED_PRECONSUMPTION", "admission status")
    require(admission["authority_state"]["authority_consumed"] is False, "admission consumption")
    require(admission["authority_state"]["v21_invocation_count"] == 0, "admission cardinality")
    require(not has_key_fragment(admission, "external_operator_event"), "admission gained event provenance")

    checks = review_request["required_checks"]
    require(type(checks) is list and checks, "review request checks")
    normalized_checks = " ".join(str(item).lower() for item in checks)
    require("external-operator" not in normalized_checks, "review request scope changed")
    require("operator grant" not in normalized_checks, "review request scope changed")
    require(review["decision"] == "REPLAN", "review decision")
    require(review["terminal_status"] == "FAILED_TERMINAL", "review terminal")
    require(review["reason_code"] == "HARD_THRESHOLD_FAILED", "review reason")

    require(package["action_identity"]["action_id"] == ACTION_ID, "package action")
    require(
        package["action_identity"]["retry_replay_resume_repair_replacement_permitted"] is False,
        "package replay policy",
    )
    require(package["claim_boundary"]["execution_authorized"] is False, "package authority")
    require(static_acceptance["action_id"] == ACTION_ID, "static acceptance action")
    require(static_acceptance["claim_boundary"] == "STATIC_ONLY_NO_EXECUTION_AUTHORITY", "static claim")
    require(
        static_acceptance["static_acceptance_grants_execution_authority"] is False,
        "static acceptance authority",
    )

    expected_runtime = {OWNER, AUTHORITY, LEDGER, TERMINAL, RESULT}
    actual_runtime = runtime_inventory()
    require(actual_runtime == expected_runtime, "V21 runtime inventory drift")
    require(not os.path.lexists(CREDENTIAL), "V21 credential reappeared")
    require(not os.path.lexists(FALLBACK), "V21 fallback terminal appeared")

    runtime_values: dict[str, dict[str, Any]] = {}
    runtime_identities: dict[str, dict[str, Any]] = {}
    for label, path in (
        ("owner", OWNER),
        ("authority", AUTHORITY),
        ("ledger", LEDGER),
        ("terminal", TERMINAL),
        ("result", RESULT),
    ):
        value, identity = pinned_json(label, path, mode="0400")
        action_field = "irreversible_action_id" if label == "result" else "action_id"
        require(value[action_field] == ACTION_ID, f"{label} action")
        runtime_values[label] = value
        runtime_identities[label] = identity

    terminal = runtime_values["terminal"]
    result = runtime_values["result"]
    ledger = runtime_values["ledger"]
    require(ledger["credential_consumption"]["state"] == "CONSUMED_BEFORE_PAYLOAD", "ledger state")
    require(ledger["replay_permitted"] is False, "ledger replay policy")
    require(terminal["status"] == "FAILED_TERMINAL", "terminal status")
    require(terminal["reason_code"] == "HARD_THRESHOLD_FAILED", "terminal reason")
    require(terminal["invocation_count_performed"] == 1, "terminal invocation count")
    require(terminal["payload_open_count"] == 1, "terminal payload count")
    require(terminal["official_target_process_starts"] == 1, "terminal evaluator count")
    require(result["selected_candidate"] is None, "result selection")
    require(result["selection"]["passing_candidates"] == [], "result passing set")

    all_identities = [
        reconciliation_identity,
        admission_identity,
        request_identity,
        review_identity,
        package_identity,
        acceptance_identity,
        *runtime_identities.values(),
    ]
    inode_keys = {(item["device"], item["inode"]) for item in all_identities}
    require(len(inode_keys) == len(all_identities), "aliased lifecycle/evidence files")

    v20_identities = {}
    for label, (path, expected_hash) in V20_RECORDS.items():
        identity, _ = read_regular(path)
        require(identity["sha256"] == expected_hash, f"V20 drift: {label}")
        identity["expected_sha256"] = expected_hash
        identity["logical_name"] = label
        identity["unchanged"] = True
        v20_identities[label] = identity

    verifier_identity, _ = read_regular(Path(__file__).resolve(strict=True))
    parent_identity = os.stat(REPORT.parent, follow_symlinks=False)
    require(stat.S_ISDIR(parent_identity.st_mode), "report parent is not a directory")
    require(not REPORT.parent.is_symlink(), "report parent symlink")

    record: dict[str, Any] = {
        "action_id": ACTION_ID,
        "artifact_kind": "ace2_v21_retirement_authority_containment",
        "audit_consuming_actions": 0,
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "claim_boundary": (
            "READ_ONLY_CONTAINMENT_MEASUREMENT_LOCAL_CREATE_ONLY_REPORT_"
            "NOT_INDEPENDENTLY_SEALED_NOT_FRESH_REVIEWER_ACCEPTANCE"
        ),
        "current_mission": {
            "acceptance_reachable": False,
            "handoff_file": handoff_path.name,
            "handoff_identity": handoff_identity,
            "handoff_status": handoff["review"]["status"],
            "latest_identity": latest_identity,
            "mission_id": MISSION_ID,
            "mission_identity": mission_identity,
        },
        "evidence": {
            "admission": admission_identity,
            "package": package_identity,
            "reconciliation": reconciliation_identity,
            "review": review_identity,
            "review_request": request_identity,
            "runtime": runtime_identities,
            "static_acceptance": acceptance_identity,
        },
        "external_operator_authority": {
            "fresh_grant_present_before_consumption": False,
            "retroactive_grant_permitted": False,
            "status": "ABSENT_AT_CONSUMPTION",
        },
        "fresh_reviewer_status": "PENDING",
        "privilege_context": {"effective_gid": os.getegid(), "effective_uid": os.geteuid()},
        "publication_parent": {
            "device": parent_identity.st_dev,
            "inode": parent_identity.st_ino,
            "mode": f"{stat.S_IMODE(parent_identity.st_mode):04o}",
        },
        "required_next_state": (
            "SEPARATE_SCOPED_AUTHORITY_CONTAINMENT_AND_SUCCESSOR_MISSION_"
            "WITH_PROVENANCE_VALID_OPERATOR_GRANT_BEFORE_CONSUMPTION"
        ),
        "schema_version": 1,
        "status": "PASS_V21_RETIRED_BLOCKED_PENDING_NEW_MISSION_AND_FRESH_GRANT",
        "v20_immutability": {
            "all_critical_records_unchanged": True,
            "records": v20_identities,
        },
        "v21": {
            "credential_consumed_and_absent": True,
            "official_payload_open_count": 1,
            "official_target_process_starts": 1,
            "replay_permitted": False,
            "runtime_file_count": len(actual_runtime),
            "sole_invocation_count": 1,
            "terminal_reason_code": terminal["reason_code"],
            "terminal_status": terminal["status"],
        },
        "verifier": verifier_identity,
    }
    record["record_sha256"] = sha256(compact_bytes(record))
    return record


def verify_record(record: dict[str, Any], raw: bytes) -> None:
    require(compact_bytes(record) == raw, "published report is not canonical")
    claimed = record.get("record_sha256")
    require(type(claimed) is str, "published report self-hash missing")
    unsigned = dict(record)
    del unsigned["record_sha256"]
    require(claimed == sha256(compact_bytes(unsigned)), "published report self-hash")
    require(record["status"] == "PASS_V21_RETIRED_BLOCKED_PENDING_NEW_MISSION_AND_FRESH_GRANT", "report status")
    current = build_record()
    for key in (
        "action_id",
        "artifact_kind",
        "audit_consuming_actions",
        "claim_boundary",
        "evidence",
        "external_operator_authority",
        "fresh_reviewer_status",
        "required_next_state",
        "schema_version",
        "status",
        "v20_immutability",
        "v21",
    ):
        require(record[key] == current[key], f"published report stale: {key}")

    published_mission = dict(record["current_mission"])
    current_mission = dict(current["current_mission"])
    for identity_key in ("handoff_identity", "latest_identity", "mission_identity"):
        snapshot_identity = published_mission.pop(identity_key)
        current_mission.pop(identity_key)
        require(type(snapshot_identity) is dict, f"published {identity_key} object")
        require(
            type(snapshot_identity.get("sha256")) is str
            and len(snapshot_identity["sha256"]) == 64,
            f"published {identity_key} hash",
        )
    require(
        published_mission == current_mission,
        "published report stale: reviewer mission semantics",
    )
    require(
        record["verifier"]["sha256"] == PUBLISHED_VERIFIER_SHA256,
        "published report verifier provenance",
    )


def publish(record: dict[str, Any]) -> tuple[str, int]:
    raw = compact_bytes(record)
    dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        dir_flags |= os.O_NOFOLLOW
    dir_fd = os.open(REPORT.parent, dir_flags)
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(REPORT.name, flags, 0o444, dir_fd=dir_fd)
        try:
            written = 0
            while written < len(raw):
                written += os.write(fd, raw[written:])
            os.fsync(fd)
            info = os.fstat(fd)
            require(stat.S_IMODE(info.st_mode) == 0o444, "published report mode")
            require(info.st_nlink == 1, "published report link count")
        finally:
            os.close(fd)
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return sha256(raw), len(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--publish", action="store_true")
    group.add_argument("--verify-published", action="store_true")
    args = parser.parse_args()

    if args.verify_published:
        identity, raw = read_regular(REPORT)
        require(identity["mode"] == "0444", "published report mode drift")
        require(
            identity["sha256"] == PUBLISHED_REPORT_SHA256,
            "published report hash drift",
        )
        record = json.loads(raw.decode("ascii", "strict"))
        require(type(record) is dict, "published report object")
        verify_record(record, raw)
        print(f"PASS_PUBLISHED_V21_RETIREMENT_CONTAINMENT sha256={identity['sha256']}")
        return 0

    record = build_record()
    if args.publish:
        digest, size = publish(record)
        print(f"PASS_PUBLISHED_V21_RETIREMENT_CONTAINMENT sha256={digest} size={size}")
    else:
        print(record["status"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL_V21_RETIREMENT_CONTAINMENT: {exc}", file=sys.stderr)
        raise SystemExit(1)
