#!/usr/bin/env python3
"""Shared static V29 execution-binding contract. No protected payload access."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
ACTION_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_execution_binding_repair_static_v29_additive_0001_action_root"
PACKAGE_PATH = ACTION_ROOT / "QK_GBFP8_HEAD64_B0_EXECUTION_BINDING_REPAIR_STATIC_V29_PACKAGE.json"
ACCEPTANCE_PATH = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
EVENT_LOG = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/events.jsonl")
MISSION_ID = "b0executionbindingrepairv29"
STATIC_ACTION_ID = "ace2:qk-gbfp8-base-v29:b0-execution-binding-repair:7f3c1a62:additive-0001"
EXECUTION_ACTION_ID = "ace2:qk-gbfp8-base-v29:b0-source-oracle-identity-control-execute-once:7f3c1a62:additive-0001"
CLAIM_BOUNDARY = "STATIC_ONLY_NO_EXECUTION_AUTHORITY"
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
RUNTIME_NAMESPACE = PROJECT_ROOT / "runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v29_7f3c1a62_additive_0001_once"
FAILURE_SEAL_NAMESPACE = Path(str(RUNTIME_NAMESPACE) + ".failed-start")
AUTHORITY_NAMESPACE = PROJECT_ROOT / "external/qk_gbfp8_head64_b0_source_oracle_identity_control_v29_7f3c1a62_additive_0001_authority"
MANAGER_CAPSULE = AUTHORITY_NAMESPACE / "manager-admission.json"
OPERATOR_CAPSULE = AUTHORITY_NAMESPACE / "operator-authority.json"
V28_STATIC_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_source_oracle_identity_control_static_v28_additive_0008_action_root"
V28_STATIC_PACKAGE = V28_STATIC_ROOT / "QK_GBFP8_HEAD64_SOURCE_ORACLE_IDENTITY_CONTROL_STATIC_V28_PACKAGE.json"
V28_STATIC_ACCEPTANCE = V28_STATIC_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
V28_REJECTED_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_source_oracle_identity_control_execution_v28_additive_0008_action_root"
V28_REJECTED_ARCHIVE = PROJECT_ROOT / "research/archive/rtl/v28-b0-execution-package-rejected-provenance-0001"
V28_STATIC_PACKAGE_SHA256 = "c3052c9d72733a36fa83993aa8fb22e037e00f1f0b52c26eebd6c9664fe78237"
V28_STATIC_CONTENT_SHA256 = "21b763f0f64c0a832781860729d906f8af594afff7c107008b41aebae65377f9"
V28_STATIC_ACCEPTANCE_SHA256 = "4e47a94ee3f6faed3ebba23d5cea499171c51303b4cfbe1d18884ddcf19e46c2"
V28_STATIC_ACCEPTANCE_SELF_SHA256 = "9077821d69bfcb6e63d6c602ee99be14fcae1c9f3ef0a688ccd8f8ea3fe4059a"
V28_REVIEWER_PROVENANCE_SHA256 = "3358f7e154dff28f2403f365e0860d2c37bda08b7157d68fe05b20907c21b667"
V28_REJECTED_PACKAGE_ORIGINAL_SHA256 = "ad4735802b3b5a61bd158f43c28ff7c217f6155aafa544f363864a41c618e15b"
V28_REJECTED_CONTENT_ORIGINAL_SHA256 = "5b01e4e2bd3dc5355817dc527e171b8eda19f82860535d30d98b2f11a36baafc"
V28_REJECTED_TREE_ORIGINAL_SHA256 = "f515aea8f140cb3ba12cfe47954e46e40ab372e0d226fb1a6feaf25d495edb19"
EXACT_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}
PACKAGE_NAME = "QK_GBFP8_HEAD64_B0_EXECUTION_BINDING_REPAIR_STATIC_V29_PACKAGE.json"
ACCEPTANCE_RELATIVE = "review/FRESH_L2_STATIC_ACCEPTANCE.json"
AUTHORITY_MARKER = "ACE2_B0_V29_AUTHORITY_V1 "
MAX_AUTHORITY_LIFETIME_SECONDS = 900.0
CONCURRENT_ACTIVE_WINDOW_SECONDS = 3600.0
CALL_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class ContractError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise ContractError(code, detail)


def compact_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_sha256(path: Path) -> str:
    return sha256_bytes((str(path) + "\n").encode("utf-8"))


def load_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, "NONCANONICAL_JSON", str(path))
    return value, raw


def add_self_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(value)
    result[field] = sha256_bytes(compact_bytes(result))
    return result


def verify_self_hash(value: dict[str, Any], field: str) -> None:
    observed = value.get(field)
    require(type(observed) is str, "SELF_HASH_ABSENT", field)
    unhashed = dict(value)
    unhashed.pop(field)
    require(observed == sha256_bytes(compact_bytes(unhashed)), "SELF_HASH_MISMATCH", field)


def execution_binding() -> dict[str, Any]:
    wrapper = ACTION_ROOT / "tools/future_execute_once_wrapper_v29.py"
    argv = [
        str(INTERPRETER),
        "-B",
        str(wrapper),
        "--mode",
        "production",
        "--package",
        str(PACKAGE_PATH),
        "--acceptance",
        str(ACCEPTANCE_PATH),
        "--manager-capsule",
        str(MANAGER_CAPSULE),
        "--operator-capsule",
        str(OPERATOR_CAPSULE),
        "--runtime",
        str(RUNTIME_NAMESPACE),
        "--irreversible-action-id",
        EXECUTION_ACTION_ID,
    ]
    value = {
        "argv": argv,
        "cwd": str(ACTION_ROOT),
        "environment": EXACT_ENVIRONMENT,
        "external_authority_namespace": str(AUTHORITY_NAMESPACE),
        "failure_seal_namespace": str(FAILURE_SEAL_NAMESPACE),
        "future_execution_action_id": EXECUTION_ACTION_ID,
        "interpreter": {
            "path": str(INTERPRETER),
            "sha256": INTERPRETER_SHA256,
            "version": "3.13.5",
        },
        "manager_capsule_path": str(MANAGER_CAPSULE),
        "operator_capsule_path": str(OPERATOR_CAPSULE),
        "runtime_namespace": str(RUNTIME_NAMESPACE),
        "shell": False,
    }
    value["argv_sha256"] = sha256_bytes(compact_bytes(argv))
    value["cwd_sha256"] = sha256_bytes((value["cwd"] + "\n").encode("utf-8"))
    value["environment_sha256"] = sha256_bytes(compact_bytes(EXACT_ENVIRONMENT))
    value["runtime_namespace_sha256"] = path_sha256(RUNTIME_NAMESPACE)
    value["authority_namespace_sha256"] = path_sha256(AUTHORITY_NAMESPACE)
    value["execution_binding_sha256"] = sha256_bytes(compact_bytes(value))
    return value


def expected_claim(package: dict[str, Any], role: str, nonce: str, issued_at_ts: float, fresh_until_ts: float) -> dict[str, Any]:
    binding = package["future_execution_binding"]
    decision = "ADMIT_B0_ONCE" if role == "MANAGER" else "AUTHORIZE_B0_ONCE"
    value = {
        "argv_sha256": binding["argv_sha256"],
        "authority_namespace_sha256": binding["authority_namespace_sha256"],
        "cwd_sha256": binding["cwd_sha256"],
        "decision": decision,
        "environment_sha256": binding["environment_sha256"],
        "execution_action_id": EXECUTION_ACTION_ID,
        "execution_binding_sha256": binding["execution_binding_sha256"],
        "fresh_until_ts": fresh_until_ts,
        "interpreter_path": str(INTERPRETER),
        "interpreter_sha256": INTERPRETER_SHA256,
        "issued_at_ts": issued_at_ts,
        "mission_id": MISSION_ID,
        "nonce": nonce,
        "package_content_sha256": package["package_content_sha256"],
        "package_file_sha256": sha256_file(PACKAGE_PATH),
        "runtime_namespace_sha256": binding["runtime_namespace_sha256"],
        "static_action_id": STATIC_ACTION_ID,
    }
    value["claim_sha256"] = sha256_bytes(compact_bytes(value))
    return value


def authority_event_text(claim: dict[str, Any]) -> str:
    return AUTHORITY_MARKER + compact_bytes(claim).decode("ascii").rstrip("\n")


def _event_records(raw: bytes) -> list[dict[str, Any]]:
    require(raw.endswith(b"\n"), "EVENT_LOG_TRUNCATED", "missing final newline")
    records: list[dict[str, Any]] = []
    offset = 0
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        require(line not in {b"", b"\n"}, "EVENT_LOG_BLANK_LINE", str(line_number))
        try:
            event = json.loads(line.decode("utf-8", "strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError("EVENT_LOG_INVALID_JSON", f"line {line_number}: {error}") from error
        require(type(event) is dict, "EVENT_LOG_NON_OBJECT", str(line_number))
        records.append(
            {
                "end": offset + len(line),
                "event": event,
                "line": line,
                "line_number": line_number,
                "start": offset,
            }
        )
        offset += len(line)
    return records


def _read_event_log_stable(path: Path = EVENT_LOG) -> tuple[bytes, os.stat_result]:
    path_info = os.lstat(path)
    require(stat.S_ISREG(path_info.st_mode), "EVENT_LOG_NOT_REGULAR", str(path))
    require(not stat.S_ISLNK(path_info.st_mode), "EVENT_LOG_SYMLINK", str(path))
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
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
        "EVENT_LOG_CONCURRENT_FRONTIER_DRIFT",
        str(path),
    )
    require(
        (path_info.st_dev, path_info.st_ino) == (before.st_dev, before.st_ino),
        "EVENT_LOG_REPLACED",
        str(path),
    )
    return b"".join(chunks), after


def _verify_bound_source(source: dict[str, Any], raw_override: bytes | None = None, info_override: Any | None = None) -> tuple[bytes, list[dict[str, Any]]]:
    require(source["source_id"] == "ARGUS_PROJECT_EVENTS_JSONL", "WRONG_EVENT_SOURCE_ID", str(source.get("source_id")))
    require(source["source_path"] == str(EVENT_LOG), "WRONG_EVENT_SOURCE_PATH", str(source.get("source_path")))
    if raw_override is None:
        raw, info = _read_event_log_stable(EVENT_LOG)
    else:
        raw = raw_override
        info = info_override
    require(info is not None, "EVENT_SOURCE_INFO_ABSENT", "source stat")
    require(source["source_device"] == info.st_dev and source["source_inode"] == info.st_ino, "EVENT_LOG_IDENTITY_MISMATCH", "device/inode")
    frontier = source["frontier_byte_count"]
    require(0 < frontier <= len(raw), "EVENT_LOG_TRUNCATED_OR_FRONTIER", str(frontier))
    prefix = raw[:frontier]
    require(prefix.endswith(b"\n"), "EVENT_FRONTIER_PARTIAL_LINE", str(frontier))
    require(sha256_bytes(prefix) == source["frontier_sha256"], "EVENT_FRONTIER_HASH_MISMATCH", str(frontier))
    prefix_records = _event_records(prefix)
    require(len(prefix_records) == source["frontier_line_count"], "EVENT_FRONTIER_LINE_MISMATCH", str(frontier))
    all_records = _event_records(raw)
    return raw, all_records


def validate_authority_capsule(
    capsule: dict[str, Any],
    role: str,
    package: dict[str, Any],
    now_ts: float,
    consumed_nonces: set[str] | None = None,
    raw_override: bytes | None = None,
    info_override: Any | None = None,
) -> dict[str, Any]:
    require(role in {"MANAGER", "OPERATOR"}, "AUTHORITY_ROLE", role)
    verify_self_hash(capsule, "capsule_sha256")
    require(capsule.get("role") == role, "AUTHORITY_ROLE_MISMATCH", str(capsule.get("role")))
    claim = capsule.get("claim")
    require(type(claim) is dict, "AUTHORITY_CLAIM_ABSENT", role)
    verify_self_hash(claim, "claim_sha256")
    nonce = claim.get("nonce")
    require(type(nonce) is str and re.fullmatch(r"[0-9a-f]{64}", nonce) is not None, "AUTHORITY_NONCE", str(nonce))
    expected = expected_claim(package, role, nonce, claim["issued_at_ts"], claim["fresh_until_ts"])
    require(claim == expected, "AUTHORITY_BINDING_MISMATCH", role)
    lifetime = claim["fresh_until_ts"] - claim["issued_at_ts"]
    require(0.0 < lifetime <= MAX_AUTHORITY_LIFETIME_SECONDS, "AUTHORITY_FRESHNESS_WINDOW", str(lifetime))
    require(claim["issued_at_ts"] <= now_ts <= claim["fresh_until_ts"], "AUTHORITY_STALE_OR_NOT_YET_VALID", str(now_ts))
    require(consumed_nonces is None or nonce not in consumed_nonces, "AUTHORITY_REPLAY", nonce)
    source = capsule.get("source")
    require(type(source) is dict, "AUTHORITY_SOURCE_ABSENT", role)
    raw, records = _verify_bound_source(source, raw_override, info_override)
    index = source["event_line_number"] - 1
    require(0 <= index < len(records), "AUTHORITY_EVENT_LINE", str(source["event_line_number"]))
    record = records[index]
    require(record["start"] == source["event_start_byte"] and record["end"] == source["event_end_byte"], "AUTHORITY_EVENT_FRONTIER", role)
    require(record["end"] <= source["frontier_byte_count"], "AUTHORITY_EVENT_AFTER_FRONTIER", role)
    require(sha256_bytes(record["line"]) == source["event_hash"], "AUTHORITY_EVENT_HASH_MISMATCH", role)
    event = record["event"]
    expected_type = "life.manager.intent.started" if role == "MANAGER" else "ui.operator"
    require(event.get("type") == source["event_type"] == expected_type, "AUTHORITY_EVENT_TYPE", str(event.get("type")))
    event_ts = event.get("ts")
    require(type(event_ts) in {int, float} and float(event_ts) == float(claim["issued_at_ts"]), "AUTHORITY_EVENT_TIMESTAMP", role)
    expected_text = authority_event_text(claim)
    if role == "MANAGER":
        require(event.get("agent_layer") == source["origin_agent_layer"] == "manager", "MANAGER_ORIGIN_LAYER", str(event.get("agent_layer")))
        require(event.get("source") == source["origin_source"] == "user", "MANAGER_ORIGIN_SOURCE", str(event.get("source")))
        require(event.get("intent_id") == source["intent_id"], "MANAGER_INTENT_ID", str(event.get("intent_id")))
        require(event.get("text") == expected_text and event.get("objective") == expected_text, "MANAGER_EVENT_CLAIM", "text/objective")
    else:
        require(event.get("agent_layer") == source["origin_agent_layer"] == "operator", "OPERATOR_ORIGIN_LAYER", str(event.get("agent_layer")))
        require(event.get("message_id") == source["message_id"], "OPERATOR_MESSAGE_ID", str(event.get("message_id")))
        require(event.get("text") == expected_text, "OPERATOR_EVENT_CLAIM", "text")
    occurrences = 0
    for later in records:
        later_event = later["event"]
        serialized = json.dumps(later_event, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        if nonce in serialized and later_event.get("type") == expected_type:
            occurrences += 1
            if later["line_number"] > source["event_line_number"]:
                raise ContractError("AUTHORITY_COMPLETED_REPLAY_OR_FRONTIER_DRIFT", f"nonce at line {later['line_number']}")
        if (
            nonce in serialized
            and later["line_number"] > source["event_line_number"]
            and later_event.get("type") in {"life.operator.execution.completed", "life.manager.execution.completed", "life.operator.execution.revoked"}
        ):
            raise ContractError("AUTHORITY_COMPLETED_REPLAY_OR_FRONTIER_DRIFT", f"terminal nonce at line {later['line_number']}")
    require(occurrences == 1, "AUTHORITY_NONCE_OCCURRENCE", str(occurrences))
    return {
        "event_hash": source["event_hash"],
        "event_line_number": source["event_line_number"],
        "frontier_sha256": source["frontier_sha256"],
        "nonce": nonce,
        "role": role,
        "source_byte_count": len(raw),
    }


def derive_reviewer_provenance(raw: bytes, info: Any) -> dict[str, Any]:
    records = _event_records(raw)
    provider_starts: dict[str, dict[str, Any]] = {}
    agent_starts: dict[str, dict[str, Any]] = {}
    completed: set[str] = set()
    for record in records:
        event = record["event"]
        call_id = event.get("call_id")
        if event.get("type") == "provider.request.started" and type(call_id) is str:
            require(call_id not in provider_starts, "REVIEWER_DUPLICATE_PROVIDER_START", call_id)
            provider_starts[call_id] = record
        elif event.get("type") == "agent.io.start" and type(call_id) is str:
            require(call_id not in agent_starts, "REVIEWER_DUPLICATE_AGENT_START", call_id)
            agent_starts[call_id] = record
        elif event.get("type") in {"provider.request.completed", "agent.io.complete"} and type(call_id) is str:
            completed.add(call_id)
    project_starts = [record for record in agent_starts.values() if record["event"].get("working_dir") == str(PROJECT_ROOT)]
    require(project_starts, "REVIEWER_PROJECT_START_ABSENT", str(PROJECT_ROOT))
    agent_record = max(project_starts, key=lambda item: item["line_number"])
    agent_event = agent_record["event"]
    call_id = agent_event.get("call_id")
    require(type(call_id) is str and CALL_ID_PATTERN.fullmatch(call_id) is not None, "REVIEWER_CALL_ID", str(call_id))
    require(call_id not in completed, "REVIEWER_COMPLETED_CALL_REPLAY", call_id)
    require(call_id in provider_starts, "REVIEWER_PROVIDER_START_ABSENT", call_id)
    provider_record = provider_starts[call_id]
    provider_event = provider_record["event"]
    require(agent_event.get("run_label") == "reviewer" and provider_event.get("run_label") == "reviewer", "REVIEWER_RUN_LABEL", call_id)
    prompt = agent_event.get("prompt")
    require(type(prompt) is str and MISSION_ID in prompt and STATIC_ACTION_ID in prompt, "REVIEWER_MISSION_ACTION_BINDING", call_id)
    start_ts = agent_event.get("ts")
    require(type(start_ts) in {int, float}, "REVIEWER_START_TIMESTAMP", call_id)
    concurrent = [
        record for record in project_starts
        if record["event"].get("call_id") != call_id
        and record["event"].get("call_id") not in completed
        and type(record["event"].get("ts")) in {int, float}
        and record["event"]["ts"] >= start_ts - CONCURRENT_ACTIVE_WINDOW_SECONDS
    ]
    require(not concurrent, "REVIEWER_CONCURRENT_ACTIVE_CALLS", str(len(concurrent)))
    commands = [
        record for record in records
        if record["event"].get("type") == "engineer.progress"
        and record["event"].get("kind") == "command_execution"
        and type(record["event"].get("ts")) in {int, float}
        and record["event"]["ts"] >= start_ts
        and record["event"].get("actor") == "reviewer"
        and record["event"].get("agent_layer") == "reviewer"
        and record["event"].get("status") == "completed"
        and record["event"].get("exit_code") == 0
    ]
    require(commands, "REVIEWER_COMMAND_EVENT_ABSENT", call_id)
    command_record = commands[-1]

    def event_fields(prefix: str, record: dict[str, Any]) -> dict[str, Any]:
        return {
            f"{prefix}_event_end_byte": record["end"],
            f"{prefix}_event_line_number": record["line_number"],
            f"{prefix}_event_sha256": sha256_bytes(record["line"]),
            f"{prefix}_event_start_byte": record["start"],
        }

    identity: dict[str, Any] = {
        "action_id": STATIC_ACTION_ID,
        "active_call_id": call_id,
        "actor": "reviewer",
        "agent_layer": "reviewer",
        "authoritative_event_source": "ARGUS_PROJECT_EVENTS_JSONL",
        "event_log_device": info.st_dev,
        "event_log_frontier_byte_count": len(raw),
        "event_log_frontier_line_count": len(records),
        "event_log_frontier_sha256": sha256_bytes(raw),
        "event_log_inode": info.st_ino,
        "event_log_path": str(EVENT_LOG),
        "mission_id": MISSION_ID,
        "run_label": "reviewer",
        "working_dir_sha256": path_sha256(PROJECT_ROOT),
        **event_fields("provider_start", provider_record),
        **event_fields("agent_start", agent_record),
        **event_fields("reviewer_command", command_record),
    }
    identity["provenance_identity_sha256"] = sha256_bytes(compact_bytes(identity))
    return identity


def attest_current_reviewer() -> dict[str, Any]:
    configured = os.environ.get("ARGUS_SKILL_AGENT_IO_LOG")
    require(configured == str(EVENT_LOG), "REVIEWER_EVENT_LOG_ENV", repr(configured))
    raw, info = _read_event_log_stable(EVENT_LOG)
    return derive_reviewer_provenance(raw, info)


def verify_reviewer_provenance(identity: dict[str, Any]) -> None:
    verify_self_hash(identity, "provenance_identity_sha256")
    require(identity.get("event_log_path") == str(EVENT_LOG), "REVIEWER_EVENT_LOG_PATH", str(identity.get("event_log_path")))
    source = {
        "source_id": identity["authoritative_event_source"],
        "source_path": identity["event_log_path"],
        "source_device": identity["event_log_device"],
        "source_inode": identity["event_log_inode"],
        "frontier_byte_count": identity["event_log_frontier_byte_count"],
        "frontier_line_count": identity["event_log_frontier_line_count"],
        "frontier_sha256": identity["event_log_frontier_sha256"],
    }
    raw, _ = _verify_bound_source(source)
    prefix = raw[: identity["event_log_frontier_byte_count"]]
    class Info:
        st_dev = identity["event_log_device"]
        st_ino = identity["event_log_inode"]
    derived = derive_reviewer_provenance(prefix, Info())
    require(derived == identity, "REVIEWER_PROVENANCE_RECOMPUTE_MISMATCH", identity.get("active_call_id", ""))


def inventory(action_root: Path, include_acceptance: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(action_root.rglob("*"), key=lambda item: item.relative_to(action_root).as_posix()):
        relative = path.relative_to(action_root).as_posix()
        if relative == ACCEPTANCE_RELATIVE and not include_acceptance:
            continue
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), "INVENTORY_SYMLINK", relative)
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "dir", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append(
                {
                    "kind": "file",
                    "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                    "path": relative,
                    "sha256": sha256_file(path),
                    "size": info.st_size,
                }
            )
        else:
            raise ContractError("INVENTORY_SPECIAL_FILE", relative)
    return records


def normalized_preacceptance_inventory(action_root: Path) -> list[dict[str, Any]]:
    records = inventory(action_root, include_acceptance=False)
    for record in records:
        if record["kind"] == "dir" and record["path"] == "review":
            record["mode"] = "0755"
    return records


def preacceptance_tree_sha256(action_root: Path = ACTION_ROOT) -> str:
    return sha256_bytes(compact_bytes(normalized_preacceptance_inventory(action_root)))


def verify_package(package: dict[str, Any], package_raw: bytes) -> None:
    verify_self_hash(package, "package_content_sha256")
    require(package.get("action_identity", {}).get("action_id") == STATIC_ACTION_ID, "PACKAGE_ACTION_ID", "static")
    require(package.get("claim_boundary", {}).get("claim") == CLAIM_BOUNDARY, "PACKAGE_CLAIM_BOUNDARY", "static")
    require(package.get("future_execution_binding") == execution_binding(), "PACKAGE_EXECUTION_BINDING", "exact binding")
    require(package["predecessor_preservation"]["accepted_v28_static"]["package_file_sha256"] == V28_STATIC_PACKAGE_SHA256, "PACKAGE_V28_STATIC_HASH", "package")
    require(package["predecessor_preservation"]["rejected_v28_execution"]["accepted_lineage"] is False, "PACKAGE_REJECTED_LINEAGE", "V28 execution")
    require(package["predecessor_preservation"]["rejected_v28_execution"]["binding_status"] == "EXPLICITLY_RETIRED_PERMANENTLY_REJECTED_PLANNER_MUTATED_FAILURE_EVIDENCE_ONLY", "PACKAGE_REJECTED_RETIREMENT", "V28 execution")
    require(package["b0_control"]["permanently_nonselecting"] is True and package["b0_control"]["quantization_applied"] is False, "PACKAGE_B0_CONTROL", "nonselecting/nonquantizing")
    require(package["fresh_l2_review"]["status"] == "PENDING_INDEPENDENT_REVIEW", "PACKAGE_REVIEW_STATUS", str(package["fresh_l2_review"].get("status")))
    require(package["fresh_l2_review"]["acceptance_materialized_by_engineer"] is False, "PACKAGE_ENGINEER_ACCEPTANCE", "forbidden")
    require(package["package_file_policy"]["package_relative_path"] == PACKAGE_NAME, "PACKAGE_RELATIVE_PATH", PACKAGE_NAME)
    require(package_raw == compact_bytes(package), "PACKAGE_CANONICAL_BYTES", PACKAGE_NAME)


def verify_predecessor_preservation() -> dict[str, Any]:
    require(sha256_file(V28_STATIC_PACKAGE) == V28_STATIC_PACKAGE_SHA256, "V28_STATIC_PACKAGE_DRIFT", str(V28_STATIC_PACKAGE))
    accepted, _ = load_canonical(V28_STATIC_PACKAGE)
    require(accepted.get("package_content_sha256") == V28_STATIC_CONTENT_SHA256, "V28_STATIC_CONTENT_DRIFT", str(V28_STATIC_PACKAGE))
    require(sha256_file(V28_STATIC_ACCEPTANCE) == V28_STATIC_ACCEPTANCE_SHA256, "V28_STATIC_ACCEPTANCE_DRIFT", str(V28_STATIC_ACCEPTANCE))
    acceptance, _ = load_canonical(V28_STATIC_ACCEPTANCE)
    require(acceptance.get("acceptance_sha256") == V28_STATIC_ACCEPTANCE_SELF_SHA256, "V28_STATIC_ACCEPTANCE_SELF_DRIFT", str(V28_STATIC_ACCEPTANCE))
    require(acceptance.get("provenance_identity_sha256") == V28_REVIEWER_PROVENANCE_SHA256, "V28_STATIC_REVIEWER_PROVENANCE_DRIFT", str(V28_STATIC_ACCEPTANCE))
    require(V28_REJECTED_ROOT.is_dir(), "V28_REJECTED_ROOT_ABSENT", str(V28_REJECTED_ROOT))
    require(V28_REJECTED_ARCHIVE.is_dir(), "V28_REJECTED_ARCHIVE_ABSENT", str(V28_REJECTED_ARCHIVE))
    return {
        "accepted_v28_static_package_sha256": V28_STATIC_PACKAGE_SHA256,
        "accepted_v28_static_acceptance_sha256": V28_STATIC_ACCEPTANCE_SHA256,
        "rejected_v28_archive_present": True,
        "rejected_v28_mutated_root_present": True,
    }


def verify_exact_inventory(package: dict[str, Any], allow_acceptance: bool) -> dict[str, Any]:
    expected_files = set(package["package_file_policy"]["allowed_relative_files"])
    preacceptance_file_count = len(expected_files) - 1
    if not allow_acceptance:
        expected_files.remove(ACCEPTANCE_RELATIVE)
        require(not os.path.lexists(ACCEPTANCE_PATH), "ENGINEER_ACCEPTANCE_MATERIALIZED", str(ACCEPTANCE_PATH))
    else:
        require(ACCEPTANCE_PATH.is_file(), "ACCEPTANCE_ABSENT", str(ACCEPTANCE_PATH))
    records = inventory(ACTION_ROOT, include_acceptance=allow_acceptance)
    observed_files = {record["path"] for record in records if record["kind"] == "file"}
    observed_dirs = {record["path"] for record in records if record["kind"] == "dir"}
    require(observed_files == expected_files, "INVENTORY_FILES", str(sorted(observed_files ^ expected_files)))
    require(observed_dirs == set(package["package_file_policy"]["allowed_relative_directories"]), "INVENTORY_DIRECTORIES", str(sorted(observed_dirs)))
    for record in records:
        path = record["path"]
        require("__pycache__" not in path and not path.endswith((".pyc", ".pyo", ".tmp", ".swp", "~")), "INVENTORY_FORBIDDEN", path)
        if record["kind"] == "file":
            require(record["mode"] == "0444", "INVENTORY_FILE_MODE", path)
        elif path == "review" and not allow_acceptance:
            require(record["mode"] == "0755", "INVENTORY_REVIEW_MODE", path)
        else:
            require(record["mode"] == "0555", "INVENTORY_DIRECTORY_MODE", path)
    return {
        "file_count": preacceptance_file_count,
        "directory_count": len(observed_dirs),
        "preacceptance_tree_sha256": preacceptance_tree_sha256(),
    }


def verify_generated_files(package: dict[str, Any]) -> dict[str, Any]:
    for relative, expected in package["generated_files"].items():
        path = ACTION_ROOT / relative
        require(path.is_file(), "GENERATED_FILE_ABSENT", relative)
        require(path.stat().st_size == expected["size"] and sha256_file(path) == expected["sha256"], "GENERATED_FILE_DRIFT", relative)
    return {"generated_file_count": len(package["generated_files"])}


def synthetic_rows() -> list[tuple[int, int, int]]:
    return [(index, (index * 17) - 4096, (index % 13) - 6) for index in range(574)]


def classify_rows(source: list[tuple[int, int, int]], oracle: list[tuple[int, int, int]]) -> dict[str, Any]:
    require(len(source) == len(oracle) == 574, "ROW_COUNT", f"{len(source)}/{len(oracle)}")
    mismatches = [index for index, (left, right) in enumerate(zip(source, oracle, strict=True)) if left != right]
    result = {
        "classification": "SOURCE_ORACLE_MATCH" if not mismatches else "SOURCE_ORACLE_MISMATCH",
        "mismatch_count": len(mismatches),
        "mismatch_membership_sha256": sha256_bytes(compact_bytes(mismatches)),
        "row_count": 574,
        "safe_output_boundary": "AGGREGATE_ONLY_NO_RAW_OR_RECONSTRUCTABLE_DATA",
    }
    require(set(result) == {"classification", "mismatch_count", "mismatch_membership_sha256", "row_count", "safe_output_boundary"}, "SAFE_AGGREGATE_OUTPUT", "unexpected fields")
    return result


def transition_model(runtime_preexists: bool, failure_seal_preexists: bool, starts: int, capsules_valid: bool) -> str:
    if failure_seal_preexists:
        return "REJECT_TERMINALLY_SEALED_PRIOR_FAILURE"
    if runtime_preexists:
        return "REJECT_RUNTIME_PREEXISTENCE_AND_SEAL_SIBLING"
    if starts != 1:
        return "REJECT_DUPLICATE_START"
    if not capsules_valid:
        return "CLAIM_RUNTIME_THEN_FAILED_TERMINAL_NO_RETRY"
    return "CLAIM_CONSUME_FREEZE_BEFORE_PROTECTED_BOUNDARY"


def adapter_boundary_model(case: str) -> str:
    mapping = {
        "spawn": "SPAWN",
        "abi": "ABI",
        "return_code": "RETURN_CODE",
        "stdout": "STDOUT",
        "stderr": "STDERR",
        "decode": "DECODE",
        "result_schema": "RESULT_SCHEMA",
        "result_record": "RESULT_RECORD",
        "exception": "EXCEPTION",
        "publication": "PUBLICATION",
    }
    require(case in mapping, "ADAPTER_CASE", case)
    return mapping[case]


def acceptance_unhashed(package: dict[str, Any], report: dict[str, Any], report_raw: bytes, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": True,
        "action_id": STATIC_ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_b0_execution_binding_repair_static_v29_fresh_l2_acceptance",
        "candidate_action_tree_sha256": report["preacceptance_tree_sha256"],
        "candidate_report_file_sha256": sha256_bytes(report_raw),
        "candidate_report_sha256": report["report_sha256"],
        "claim_boundary": CLAIM_BOUNDARY,
        "creator_provenance": provenance,
        "decision": "ACCEPT_STATIC_PACKAGE",
        "execution_binding_sha256": package["future_execution_binding"]["execution_binding_sha256"],
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "package_content_sha256": package["package_content_sha256"],
        "package_file_sha256": sha256_file(PACKAGE_PATH),
        "regression_case_count": report["regression_case_count"],
        "schema_version": 1,
        "static_acceptance_grants_execution_authority": False,
    }


def verify_acceptance(package: dict[str, Any], acceptance: dict[str, Any], acceptance_raw: bytes, report: dict[str, Any]) -> None:
    verify_self_hash(acceptance, "acceptance_sha256")
    expected = add_self_hash(acceptance_unhashed(package, report, compact_bytes(report), acceptance["creator_provenance"]), "acceptance_sha256")
    require(acceptance == expected, "ACCEPTANCE_BINDING_MISMATCH", "package/tree/report/provenance")
    require(acceptance_raw == compact_bytes(acceptance), "ACCEPTANCE_CANONICAL_BYTES", str(ACCEPTANCE_PATH))
    verify_reviewer_provenance(acceptance["creator_provenance"])


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"}:
        raise ContractError("STATIC_VERIFIER_PROCESS_START", event)
    if event == "open" and args:
        path = args[0]
        if isinstance(path, (str, bytes)):
            text = path.decode("utf-8", "replace") if isinstance(path, bytes) else path
            lowered = text.lower()
            if lowered.endswith((".safetensors", ".npy", ".npz", ".pt", ".pth", ".bin")) or "/protected-payload/" in lowered:
                raise ContractError("STATIC_VERIFIER_PROTECTED_OPEN", text)


def install_audit_hook() -> None:
    sys.addaudithook(audit_hook)
