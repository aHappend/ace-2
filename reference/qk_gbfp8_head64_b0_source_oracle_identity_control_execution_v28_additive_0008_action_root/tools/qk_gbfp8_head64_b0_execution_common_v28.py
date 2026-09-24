#!/home/argustest/miniconda3/bin/python3.13
"""Shared immutable identities and create-once helpers for the inert V28 action."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from authority_provenance_v28 import (
    EVENT_LOG,
    validate_external_capsule_provenance,
    validate_reviewer_provenance,
)


PROJECT_ROOT = Path("/home/argustest/ace-2")
ACTION_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_source_oracle_identity_control_execution_v28_additive_0008_action_root"
STATIC_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_source_oracle_identity_control_static_v28_additive_0008_action_root"
STATIC_PACKAGE = STATIC_ROOT / "QK_GBFP8_HEAD64_SOURCE_ORACLE_IDENTITY_CONTROL_STATIC_V28_PACKAGE.json"
STATIC_ACCEPTANCE = STATIC_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
STATIC_ENVELOPE_SCHEMA = STATIC_ROOT / "reference/QK_GBFP8_HEAD64_SOURCE_ORACLE_IDENTITY_CONTROL_STATIC_V28_FUTURE_EXACTLY_ONCE_AUTHORITY_ENVELOPE_SCHEMA.json"
STATIC_SIDECAR_SCHEMA = STATIC_ROOT / "reference/QK_GBFP8_HEAD64_SOURCE_ORACLE_IDENTITY_CONTROL_STATIC_V28_B0_SIDECAR_SCHEMA.json"
PACKAGE = ACTION_ROOT / "QK_GBFP8_HEAD64_B0_SOURCE_ORACLE_IDENTITY_CONTROL_EXECUTION_V28_PACKAGE.json"
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0008_once"
EXTERNAL_AUTHORITY_ROOT = PROJECT_ROOT / "authority/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0008_once"
EXTERNAL_MANAGER_ADMISSION = EXTERNAL_AUTHORITY_ROOT / "manager-admission.json"
EXTERNAL_OPERATOR_AUTHORITY = EXTERNAL_AUTHORITY_ROOT / "operator-authority.json"
OFFICIAL_PAYLOAD = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
V21_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root"
V21_BINDING_TABLE = V21_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"
OFFICIAL_PARSER = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"
OFFICIAL_EVALUATOR = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"

ACTION_ID = "ace2:qk-gbfp8-base-v28:b0-source-oracle-identity-control-execute-once:38571d38:additive-0008"
STATIC_ACTION_ID = "ace2:qk-gbfp8-base-v28:b0-source-oracle-identity-control:38571d38:additive-0008"
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
INTERPRETER_VERSION = "3.13.5"
STATIC_PACKAGE_FILE_SHA256 = "c3052c9d72733a36fa83993aa8fb22e037e00f1f0b52c26eebd6c9664fe78237"
STATIC_PACKAGE_CONTENT_SHA256 = "21b763f0f64c0a832781860729d906f8af594afff7c107008b41aebae65377f9"
STATIC_ACCEPTANCE_FILE_SHA256 = "4e47a94ee3f6faed3ebba23d5cea499171c51303b4cfbe1d18884ddcf19e46c2"
STATIC_ACCEPTANCE_SELF_SHA256 = "9077821d69bfcb6e63d6c602ee99be14fcae1c9f3ef0a688ccd8f8ea3fe4059a"
STATIC_ACCEPTED_TREE_SHA256 = "cfb9a89108e1906c1ac82898d4bbea90a449d1ba5bae304b7e7ad4438540ebf8"
EXECUTION_BINDING_SHA256 = "cf61c411b167203607dbe6e69e52cb147f4c7a33bf9d4b22503f59bfdb1d216e"
IDENTITY_BUNDLE_SHA256 = "9e5061229d0c4bc7b63778707d432c27b59d7c7b30bc33f65661c3c757ae9574"
EXACT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "TZ": "UTC"}

OUTPUT_PATHS = {
    "authority_envelope": str(RUNTIME_ROOT / "authority/exactly-once-envelope.json"),
    "authority_consumption": str(RUNTIME_ROOT / "authority/authority-consumption.json"),
    "credential_consumption": str(RUNTIME_ROOT / "authority/credential-consumption.json"),
    "ledger": str(RUNTIME_ROOT / "ledger/exactly-once-ledger.json"),
    "sidecar": str(RUNTIME_ROOT / "primary/result/b0/source-oracle-aggregate-sidecar.json"),
    "terminal": str(RUNTIME_ROOT / "primary/result/b0/first-terminal.json"),
}
EXACT_ARGV = [
    str(INTERPRETER),
    str(ACTION_ROOT / "tools/execute_qk_gbfp8_head64_b0_source_oracle_identity_control_once_v28.py"),
    "--mode", "production",
    "--static-package", str(STATIC_PACKAGE),
    "--static-acceptance", str(STATIC_ACCEPTANCE),
    "--authority-envelope", OUTPUT_PATHS["authority_envelope"],
    "--output-sidecar", OUTPUT_PATHS["sidecar"],
    "--terminal", OUTPUT_PATHS["terminal"],
    "--irreversible-action-id", ACTION_ID,
]
EXECUTION_BINDING = {
    "argv": EXACT_ARGV,
    "cwd": str(ACTION_ROOT),
    "environment": EXACT_ENVIRONMENT,
    "execution_binding_sha256": EXECUTION_BINDING_SHA256,
    "future_execution_action_id": ACTION_ID,
    "interpreter": {"path": str(INTERPRETER), "sha256": INTERPRETER_SHA256, "version": INTERPRETER_VERSION},
    "output_paths": OUTPUT_PATHS,
    "runtime_namespace": str(RUNTIME_ROOT),
    "shell": False,
}
SOURCE_STATIC_ACCEPTANCE = {
    "accepted_action_tree_sha256": STATIC_ACCEPTED_TREE_SHA256,
    "fresh_l2_acceptance_file_sha256": STATIC_ACCEPTANCE_FILE_SHA256,
    "fresh_l2_acceptance_self_sha256": STATIC_ACCEPTANCE_SELF_SHA256,
    "package_content_sha256": STATIC_PACKAGE_CONTENT_SHA256,
    "package_file_sha256": STATIC_PACKAGE_FILE_SHA256,
}
EXACTLY_ONCE = {
    "attempt_ordinal": 1,
    "authority_consumption_budget": 1,
    "credential_consumption_budget": 1,
    "official_evaluator_invocation_budget": 1,
    "official_payload_open_budget": 1,
    "official_target_process_start_budget": 1,
    "retry_replay_resume_repair_permitted": False,
    "runtime_namespace_must_not_preexist": True,
    "terminal_sealing_required_on_every_outcome": True,
    "terminal_states": ["SOURCE_ORACLE_MATCH", "SOURCE_ORACLE_MISMATCH", "FAILED_TERMINAL"],
}


class ExecutionError(RuntimeError):
    def __init__(self, detail: str, stage: str = "PREFLIGHT", failure_class: str = "CONTRACT") -> None:
        super().__init__(detail)
        self.stage = stage
        self.failure_class = failure_class


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0))
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    result[field] = sha256_bytes(compact_bytes(result))
    return result


def verify_self_hash(value: dict[str, Any], field: str) -> None:
    observed = value.get(field)
    candidate = dict(value)
    candidate.pop(field, None)
    if observed != sha256_bytes(compact_bytes(candidate)):
        raise ExecutionError(f"self hash {field}", "SCHEMA", "SELF_HASH")


def candidate_inventory(root: Path = ACTION_ROOT) -> list[dict[str, Any]]:
    """Mode/hash inventory excluding the future Reviewer-created acceptance."""

    acceptance = root / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
    records: list[dict[str, Any]] = []
    root_info = os.lstat(root)
    records.append({"kind": "directory", "mode": f"{stat.S_IMODE(root_info.st_mode):04o}", "path": "."})
    for path in sorted(root.rglob("*")):
        if path == acceptance:
            continue
        info = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        require(not stat.S_ISLNK(info.st_mode), f"candidate symlink {relative}", "PACKAGE_ACCEPTANCE", "TREE")
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append({
                "kind": "file",
                "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                "path": relative,
                "sha256": sha256_file(path),
                "size": info.st_size,
            })
        else:
            raise ExecutionError(f"candidate entry {relative}", "PACKAGE_ACCEPTANCE", "TREE")
    return records


def candidate_tree_sha256(root: Path = ACTION_ROOT) -> str:
    return sha256_bytes(compact_bytes(candidate_inventory(root)))


def canonical_object(path: Path) -> tuple[dict[str, Any], bytes]:
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            duplicate |= key in result
            result[key] = value
        return result

    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"), object_pairs_hook=pairs, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if duplicate or type(value) is not dict:
        raise ExecutionError(f"canonical object {path}", "SCHEMA", "DECODE")
    return value, raw


def require(condition: bool, detail: str, stage: str = "PREFLIGHT", failure_class: str = "CONTRACT") -> None:
    if not condition:
        raise ExecutionError(detail, stage, failure_class)


def require_regular_mode(path: Path, mode: int, detail: str) -> None:
    info = os.lstat(path)
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), detail, "STATIC_BINDING")
    require(stat.S_IMODE(info.st_mode) == mode, f"{detail} mode", "STATIC_BINDING")


def durable_create(path: Path, value: dict[str, Any] | bytes) -> str:
    raw = value if type(value) is bytes else compact_bytes(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0), 0o400)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short create-only write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_bytes(raw)


def safe_detail(error: BaseException, stage: str, failure_class: str) -> dict[str, Any]:
    identity = f"{type(error).__name__}:{stage}:{failure_class}".encode("ascii", "replace")
    return {"detail_sha256": sha256_bytes(identity), "exception_type": type(error).__name__}


def parse_time(value: Any) -> datetime:
    require(type(value) is str and value.endswith("Z"), "authority time syntax", "AUTHORITY", "FRESHNESS")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ExecutionError("authority time decode", "AUTHORITY", "FRESHNESS") from error
    require(parsed.tzinfo is not None, "authority time zone", "AUTHORITY", "FRESHNESS")
    return parsed.astimezone(timezone.utc)


def validate_fresh_window(record: dict[str, Any], now: datetime | None = None) -> None:
    observed = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
    issued = parse_time(record.get("issued_at_utc"))
    expires = parse_time(record.get("not_after_utc"))
    require(issued <= observed <= expires, "authority freshness window", "AUTHORITY", "FRESHNESS")
    require((expires - issued).total_seconds() <= 86400, "authority freshness duration", "AUTHORITY", "FRESHNESS")


def expected_external_binding(
    package_file_sha256: str,
    package_content_sha256: str,
    package_acceptance_file_sha256: str,
    package_acceptance_self_sha256: str,
) -> dict[str, Any]:
    return {
        "execution_binding": EXECUTION_BINDING,
        "execution_package_content_sha256": package_content_sha256,
        "execution_package_file_sha256": package_file_sha256,
        "execution_package_acceptance_file_sha256": package_acceptance_file_sha256,
        "execution_package_acceptance_self_sha256": package_acceptance_self_sha256,
        "future_execution_action_id": ACTION_ID,
        "source_static_acceptance": SOURCE_STATIC_ACCEPTANCE,
    }


def external_binding_tokens(
    package_file_sha256: str,
    package_content_sha256: str,
    package_acceptance_file_sha256: str,
    package_acceptance_self_sha256: str,
) -> list[str]:
    return [
        ACTION_ID,
        package_file_sha256,
        package_content_sha256,
        package_acceptance_file_sha256,
        package_acceptance_self_sha256,
        EXECUTION_BINDING_SHA256,
        str(INTERPRETER),
        INTERPRETER_SHA256,
        str(ACTION_ROOT),
        str(RUNTIME_ROOT),
        json.dumps(EXACT_ARGV, ensure_ascii=True, separators=(",", ":")),
        json.dumps(EXACT_ENVIRONMENT, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
    ]


def validate_execution_acceptance(
    acceptance: dict[str, Any],
    package_raw: bytes,
    package_content_sha256: str,
    *,
    event_log: Path = EVENT_LOG,
    reviewer_validator: Callable[..., dict[str, str]] = validate_reviewer_provenance,
) -> str:
    verify_self_hash(acceptance, "acceptance_sha256")
    package_file_sha256 = sha256_bytes(package_raw)
    expected_tree = candidate_tree_sha256()
    require(acceptance.get("accepted") is True, "execution acceptance accepted", "PACKAGE_ACCEPTANCE")
    require(acceptance.get("decision") == "ACCEPT_STATIC_PACKAGE", "execution acceptance decision", "PACKAGE_ACCEPTANCE")
    require(acceptance.get("claim_boundary") == "STATIC_ONLY_NO_EXECUTION_AUTHORITY", "execution acceptance claim", "PACKAGE_ACCEPTANCE")
    require(acceptance.get("static_acceptance_grants_execution_authority") is False, "execution acceptance authority", "PACKAGE_ACCEPTANCE")
    require(acceptance.get("execution_package_file_sha256") == package_file_sha256, "execution acceptance package", "PACKAGE_ACCEPTANCE")
    require(acceptance.get("execution_package_content_sha256") == package_content_sha256, "execution acceptance content", "PACKAGE_ACCEPTANCE")
    require(acceptance.get("candidate_tree_sha256") == expected_tree, "execution acceptance candidate tree", "PACKAGE_ACCEPTANCE", "TREE")
    witness = acceptance.get("creator_provenance")
    required_tokens = [ACTION_ID, package_file_sha256, package_content_sha256, expected_tree]
    reviewer_validator(witness, required_tokens, event_log=event_log)
    require(
        acceptance.get("provenance_identity_sha256") == witness.get("provenance_identity_sha256"),
        "execution acceptance provenance identity",
        "PACKAGE_ACCEPTANCE",
        "PROVENANCE",
    )
    return expected_tree


def validate_external_records(
    manager: dict[str, Any],
    manager_raw: bytes,
    operator: dict[str, Any],
    operator_raw: bytes,
    package_file_sha256: str,
    package_content_sha256: str,
    package_acceptance_file_sha256: str,
    package_acceptance_self_sha256: str,
    *,
    now: datetime | None = None,
    event_log: Path = EVENT_LOG,
    provenance_validator: Callable[..., dict[str, str]] = validate_external_capsule_provenance,
) -> None:
    verify_self_hash(manager, "capsule_sha256")
    verify_self_hash(operator, "capsule_sha256")
    require(compact_bytes(manager) == manager_raw, "manager capsule canonical bytes", "AUTHORITY", "MANAGER_ADMISSION")
    require(compact_bytes(operator) == operator_raw, "operator capsule canonical bytes", "AUTHORITY", "OPERATOR_AUTHORITY")
    binding = expected_external_binding(
        package_file_sha256, package_content_sha256,
        package_acceptance_file_sha256, package_acceptance_self_sha256,
    )
    manager_expected = {
        "actor": "manager", "artifact_kind": "qk_gbfp8_head64_b0_v28_external_manager_admission",
        "blocked": False, "decision": "ADMIT_B0_ONCE", "fresh": True, "schema_version": 1,
    }
    operator_expected = {
        "actor": "operator", "affirmative": True,
        "artifact_kind": "qk_gbfp8_head64_b0_v28_external_operator_authority",
        "decision": "AUTHORIZE_B0_ONCE", "fresh": True, "schema_version": 1,
    }
    for key, value in manager_expected.items():
        require(manager.get(key) == value, f"manager {key}", "AUTHORITY", "MANAGER_ADMISSION")
    for key, value in operator_expected.items():
        require(operator.get(key) == value, f"operator {key}", "AUTHORITY", "OPERATOR_AUTHORITY")
    for key, value in binding.items():
        require(manager.get(key) == value, f"manager binding {key}", "AUTHORITY", "MANAGER_ADMISSION")
        require(operator.get(key) == value, f"operator binding {key}", "AUTHORITY", "OPERATOR_AUTHORITY")
    require(type(manager.get("event_sha256")) is str and len(manager["event_sha256"]) == 64, "manager event", "AUTHORITY", "MANAGER_ADMISSION")
    require(type(operator.get("provenance_event_sha256")) is str and len(operator["provenance_event_sha256"]) == 64, "operator provenance", "AUTHORITY", "OPERATOR_AUTHORITY")
    require(type(manager.get("reservation_nonce_sha256")) is str and len(manager["reservation_nonce_sha256"]) == 64, "manager nonce", "AUTHORITY", "MANAGER_ADMISSION")
    require(operator.get("reservation_nonce_sha256") == manager["reservation_nonce_sha256"], "shared reservation nonce", "AUTHORITY", "PROVENANCE")
    require(operator.get("manager_admission_file_sha256") == sha256_bytes(manager_raw), "operator manager provenance", "AUTHORITY", "PROVENANCE")
    validate_fresh_window(manager, now)
    validate_fresh_window(operator, now)
    tokens = external_binding_tokens(
        package_file_sha256,
        package_content_sha256,
        package_acceptance_file_sha256,
        package_acceptance_self_sha256,
    )
    observed = provenance_validator(manager, operator, tokens, event_log=event_log)
    require(manager.get("event_sha256") == observed["manager_event_sha256"], "manager authoritative event", "AUTHORITY", "PROVENANCE")
    require(operator.get("provenance_event_sha256") == observed["operator_event_sha256"], "operator authoritative event", "AUTHORITY", "PROVENANCE")


def build_runtime_envelope(manager: dict[str, Any], operator: dict[str, Any]) -> dict[str, Any]:
    envelope = {
        "artifact_kind": "qk_gbfp8_head64_source_oracle_identity_control_v28_future_exactly_once_authority_envelope",
        "exactly_once": EXACTLY_ONCE,
        "execution_binding": EXECUTION_BINDING,
        "future_execution_action_id": ACTION_ID,
        "identity_bundle_sha256": IDENTITY_BUNDLE_SHA256,
        "manager_admission": {
            "actor": "manager", "blocked": False, "decision": "ADMIT_B0_ONCE",
            "event_sha256": manager["event_sha256"], "source_static_acceptance": SOURCE_STATIC_ACCEPTANCE,
        },
        "operator_authority": {
            "actor": "operator", "affirmative": True, "decision": "AUTHORIZE_B0_ONCE",
            "execution_binding_sha256": EXECUTION_BINDING_SHA256,
            "provenance_event_sha256": operator["provenance_event_sha256"],
            "source_static_acceptance": SOURCE_STATIC_ACCEPTANCE,
        },
        "reservation_nonce_sha256": manager["reservation_nonce_sha256"],
        "schema_version": 1,
        "source_static_acceptance": SOURCE_STATIC_ACCEPTANCE,
        "source_static_action_id": STATIC_ACTION_ID,
    }
    return sealed(envelope, "authority_envelope_sha256")
