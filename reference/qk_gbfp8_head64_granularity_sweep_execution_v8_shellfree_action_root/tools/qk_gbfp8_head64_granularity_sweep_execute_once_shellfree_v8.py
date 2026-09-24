#!/usr/bin/env python3
"""Future Base V8 execute-once launcher behind the frozen shell-free transport.

Static verification parses this file but never imports it.  If eventually
invoked, transport attestation completes before acceptance is opened or any
authority, credential, or ledger record is constructed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_action_root")
OLD_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v8_action_root")
ACCEPTED_V8_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root")
V6_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
PACKAGE = ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_PACKAGE.json"
ACCEPTANCE = ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
TRANSPORT = ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v8.py"
LAUNCHER = ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v8.py"
ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1724Z"
OLD_ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1614Z"
RETIRED_PREPACKAGE_ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1610Z"
INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
EXACT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}
EXACT_TRANSPORT_ARGV = [
    str(INTERPRETER),
    str(TRANSPORT),
    "--package",
    str(PACKAGE),
    "--acceptance",
    str(ACCEPTANCE),
    "--irreversible-action-id",
    ACTION_ID,
]
EXACT_LAUNCHER_ARGV = [
    str(INTERPRETER),
    str(LAUNCHER),
    "--package",
    str(PACKAGE),
    "--acceptance",
    str(ACCEPTANCE),
    "--irreversible-action-id",
    ACTION_ID,
]
LIVE_ROOT = ROOT / "live"
LIVE_PATHS = {
    "authority": LIVE_ROOT / "authority/base/authority.json",
    "credential": LIVE_ROOT / "authority/base/credential.json",
    "ledger": LIVE_ROOT / "authority/base/authority-ledger.json",
    "result": LIVE_ROOT / "result/base/result.json",
    "first_terminal": LIVE_ROOT / "authority/base/first-terminal.json",
}
SEALED_TENSOR = V6_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
LANE_METADATA = V6_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json"
OLD_PATHS = {
    "prior_execution_package": OLD_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_PACKAGE.json",
    "prior_fresh_l2_acceptance": OLD_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json",
    "prior_authority": OLD_ROOT / "live/authority/base/authority.json",
    "prior_consumed_ledger": OLD_ROOT / "live/authority/base/authority-ledger.json",
    "prior_first_terminal": OLD_ROOT / "live/authority/base/first-terminal.json",
}
OLD_RESULT = OLD_ROOT / "live/result/base/result.json"
OLD_CREDENTIAL = OLD_ROOT / "live/authority/base/credential.json"
ACCEPTED_PATHS = {
    "accepted_v8_handoff": Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/f0fa5c269681/round-0007.json"),
    "accepted_r2_handoff": Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/b8v8execpkg02/round-0003.json"),
    "accepted_v8_manifest": ACCEPTED_V8_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_MANIFEST.json",
    "runtime_package": ACCEPTED_V8_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RUNTIME_PACKAGE.json",
    "package": ACCEPTED_V8_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.json",
    "result_schema": ACCEPTED_V8_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.json",
    "controller": ACCEPTED_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v8.py",
    "evaluator": ACCEPTED_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py",
    "c02_parser": ACCEPTED_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py",
    "decisive_verifier": ACCEPTED_V8_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_static_v8.py",
    **OLD_PATHS,
}
EXPECTED_BINDING_HASHES = {
    "accepted_v8_handoff": "2254dd910cf2aae776d73c190402b42f2db66c252a80a1a0d958618f577798e3",
    "accepted_r2_handoff": "df40a23647066bab6210b661f842258223dbbf88bbb69fc05ad3ec4e625e72b2",
    "accepted_v8_manifest": "52324ad6ebd1b831774b9fae6923015f2b4cb24f5cde40e3f7d83435651efd64",
    "runtime_package": "ef2ba57e278be4fb24f6fa56faadfe84a91123781fadab5b6110163b3b5518be",
    "package": "b7c758c7141d7ac8037502da4a00fe9e0780d30501e2af483e9b8c571451493a",
    "result_schema": "fc5cf29d7ec7486b106edac592787b5888557c4b1b7453cb0a668b02fca113ec",
    "controller": "faab3064733be0d0fc68c565f00dd2c77022b7f4974d39b798eda1f150d55aa4",
    "evaluator": "8ab74c7397006c9f419059f295613ce4a743a3e0b540176ba7cf6dbb6efd7f63",
    "c02_parser": "ff9216d58bcd1584361e17d8d1854f3bcb96923c17927b1f309861546807bdb1",
    "decisive_verifier": "67da980b637b93af6784dd8ab11ceb727b37ba4f5357e41c758c53015d0ff6eb",
    "prior_execution_package": "dc03586e17db25a199d9564189b34616d314372c5373fc4f2901c57182363ea6",
    "prior_fresh_l2_acceptance": "4daece97b7e0ec8b9b936d369cae943e61aa7e9b0cf08a4baa6e0043c600a1a0",
    "prior_authority": "6944069d566823e1ae12353f7434eed6eae057f31540f36e55df82f58d67300f",
    "prior_consumed_ledger": "0b4485abb861a8aeda82a283ecab7814045599e49a7d7e5256eea7a7cc6d0708",
    "prior_first_terminal": "5d6243a9fbd830334829bd96a63a0be7da965a22a55cab93a6500f2a0463c0c8",
}
EXPECTED_BINDING_ORDER = list(EXPECTED_BINDING_HASHES)
LOCAL_PATHS = {
    "authority_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_AUTHORITY_SCHEMA.json",
    "credential_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_CREDENTIAL_SCHEMA.json",
    "ledger_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_LEDGER_SCHEMA.json",
    "first_terminal_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_FIRST_TERMINAL_SCHEMA.json",
    "fresh_l2_acceptance_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_FRESH_L2_ACCEPTANCE_SCHEMA.json",
    "transport": TRANSPORT,
    "launcher": LAUNCHER,
    "static_verifier": ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree.py",
}
LOCAL_BINDING_ORDER = list(LOCAL_PATHS)
SCHEMA_STACK = {
    "jsonschema": "4.23.0",
    "attrs": "26.1.0",
    "jsonschema-specifications": "2025.9.1",
    "referencing": "0.37.0",
    "rpds-py": "0.30.0",
}
OFFICIAL_IDENTITIES = {
    "model_sha256": "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7",
    "tensor_bundle_sha256": "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175",
    "input_tokens_sha256": "1b8c972381a2c3d7c754d1d2879b4389a13aca3f1d485f703510702c1cf2eb86",
    "lane_metadata_sha256": "4d5904378b9ccbabd434119b6a57f8d4266abf3d8e772c4c1bca98df785e7f3a",
}
FIXED_POPULATIONS = {"query_rows": 574, "valid_causal_scores": 12054, "oracle_unique_ranking_rows": 346}
CANDIDATE_ORDER = ["G8", "G4", "G2", "G1"]


class ExecutionError(RuntimeError):
    pass


class TransportAttestationError(ExecutionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutionError(message)


def require_transport(condition: bool, message: str) -> None:
    if not condition:
        raise TransportAttestationError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def read_canonical_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    data = read_bytes(path)
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("ascii", "strict"), object_pairs_hook=pairs, parse_constant=lambda token: (_ for _ in ()).throw(ExecutionError(token)))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExecutionError(f"{context}: invalid JSON") from error
    require(not duplicate and type(value) is dict, f"{context}: duplicate key or non-object")
    require(compact_bytes(value) == data, f"{context}: noncanonical bytes")
    return value, data


def verify_self_checksum(value: dict[str, Any], key: str, context: str) -> None:
    observed = value.get(key)
    require(type(observed) is str and len(observed) == 64, f"{context}: checksum syntax")
    payload = dict(value)
    payload.pop(key)
    require(sha256_bytes(compact_bytes(payload)) == observed, f"{context}: checksum mismatch")


def sealed(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(key, None)
    result[key] = sha256_bytes(compact_bytes(result))
    return result


def invocation_record(argv: list[str]) -> dict[str, Any]:
    return {
        "argv": argv,
        "command_representation": "ARGV_VECTOR_ONLY",
        "cwd": str(ROOT),
        "environment": EXACT_ENVIRONMENT,
        "shell": False,
    }


def _read_proc_vector(path: Path) -> list[str]:
    data = read_bytes(path)
    require_transport(data.endswith(b"\0"), f"unterminated proc vector: {path}")
    return [item.decode("utf-8", "strict") for item in data[:-1].split(b"\0")]


def _read_proc_environment(path: Path) -> dict[str, str]:
    values = _read_proc_vector(path)
    environment: dict[str, str] = {}
    for item in values:
        require_transport("=" in item, "parent environment entry")
        key, value = item.split("=", 1)
        require_transport(key not in environment, "duplicate parent environment key")
        environment[key] = value
    return environment


def _attest_transport() -> tuple[dict[str, Any], dict[str, Any], bytes]:
    require_transport(Path.cwd() == ROOT, "launcher cwd mismatch")
    require_transport([sys.executable, *sys.argv] == EXACT_LAUNCHER_ARGV, "launcher argv mismatch")
    require_transport(dict(os.environ) == EXACT_ENVIRONMENT, "launcher environment mismatch")
    package, package_bytes = read_canonical_json(PACKAGE, "execution package")
    verify_self_checksum(package, "package_content_sha256", "execution package")
    local = {item["id"]: item for item in package["local_artifact_bindings"]}
    require_transport(local["transport"]["path"] == str(TRANSPORT), "parent transport path binding")
    require_transport(local["transport"]["sha256"] == sha256_bytes(read_bytes(TRANSPORT)), "parent transport bytes")
    require_transport(local["launcher"]["path"] == str(LAUNCHER), "launcher path binding")
    require_transport(local["launcher"]["sha256"] == sha256_bytes(read_bytes(LAUNCHER)), "launcher bytes")
    parent_pid = os.getppid()
    proc_root = Path("/proc") / str(parent_pid)
    parent_argv = _read_proc_vector(proc_root / "cmdline")
    parent_environment = _read_proc_environment(proc_root / "environ")
    parent_executable_path = os.readlink(proc_root / "exe")
    parent_executable_sha256 = sha256_bytes(read_bytes(Path(parent_executable_path)))
    require_transport(parent_argv == EXACT_TRANSPORT_ARGV, "immediate parent argv mismatch")
    require_transport(parent_environment == EXACT_ENVIRONMENT, "immediate parent environment mismatch")
    require_transport(parent_executable_path == str(INTERPRETER), "immediate parent executable mismatch")
    require_transport(parent_executable_sha256 == INTERPRETER_SHA256, "immediate parent executable bytes")
    static_attestation = {
        "api": "os.posix_spawn",
        "immediate_parent_argv": parent_argv,
        "immediate_parent_environment": parent_environment,
        "immediate_parent_executable_path": parent_executable_path,
        "immediate_parent_executable_sha256": parent_executable_sha256,
        "immediate_parent_transport_path": str(TRANSPORT),
        "immediate_parent_transport_sha256": local["transport"]["sha256"],
        "launcher_argv": EXACT_LAUNCHER_ARGV,
        "launcher_environment": EXACT_ENVIRONMENT,
        "launcher_sha256": local["launcher"]["sha256"],
        "shell": False,
    }
    observed_attestation_sha256 = sha256_bytes(compact_bytes(static_attestation))
    require_transport(observed_attestation_sha256 == package["transport_contract"]["attestation_sha256"], "transport attestation digest mismatch")
    return (
        static_attestation
        | {
            "immediate_parent_pid": parent_pid,
            "transport_attestation_sha256": observed_attestation_sha256,
        },
        package,
        package_bytes,
    )


def _validate_package(package: dict[str, Any]) -> None:
    require(package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_package", "package kind")
    require(package["action_identity"] == {
        "future_action_id": ACTION_ID,
        "prior_terminal_action_id": OLD_ACTION_ID,
        "retired_action_ids": [RETIRED_PREPACKAGE_ACTION_ID, OLD_ACTION_ID],
        "reuse_permitted": False,
    }, "action identity")
    transport = invocation_record(EXACT_TRANSPORT_ARGV)
    launcher = invocation_record(EXACT_LAUNCHER_ARGV)
    require(package["future_invocation"] == transport | {"invocation_sha256": sha256_bytes(compact_bytes(transport))}, "future transport invocation")
    require(package["launcher_invocation"] == launcher | {"invocation_sha256": sha256_bytes(compact_bytes(launcher))}, "launcher invocation")
    require([item["id"] for item in package["canonical_bindings"]] == EXPECTED_BINDING_ORDER, "canonical binding order")
    require([item["id"] for item in package["local_artifact_bindings"]] == LOCAL_BINDING_ORDER, "local binding order")
    require(package["official_identities"] == OFFICIAL_IDENTITIES, "official identities")
    require(package["fixed_populations"] == FIXED_POPULATIONS, "fixed populations")
    require(package["selection"]["candidate_order"] == CANDIDATE_ORDER, "candidate order")
    require(package["transport_contract"]["direct_exec_api"] == "os.posix_spawn", "transport API")
    require(package["transport_contract"]["shell"] is False, "transport shell")
    require(package["transport_contract"]["environment_inheritance_permitted"] is False, "transport environment")
    require(package["prior_terminal_state"] == {
        "action_id": OLD_ACTION_ID,
        "invocation_count_performed": 0,
        "no_replay_retry_resume_repair_replacement": True,
        "payload_open_count": 0,
        "result_file_sha256": None,
        "status": "CONSUMED_ORPHAN",
    }, "prior terminal declaration")


def _verify_prior_terminal() -> None:
    authority, _ = read_canonical_json(OLD_PATHS["prior_authority"], "prior authority")
    ledger, _ = read_canonical_json(OLD_PATHS["prior_consumed_ledger"], "prior consumed ledger")
    terminal, _ = read_canonical_json(OLD_PATHS["prior_first_terminal"], "prior first terminal")
    require(authority["action_id"] == OLD_ACTION_ID and authority["state"] == "READY_UNCONSUMED", "prior authority state")
    require(ledger["action_id"] == OLD_ACTION_ID and ledger["action_retired"] is True, "prior ledger action")
    require(ledger["invocation_cardinality"]["invocation_count_performed_at_publish"] == 0, "prior ledger invocation")
    require(ledger["payload_cardinality"]["payload_open_count_performed_at_publish"] == 0, "prior ledger payload")
    require(ledger["replay_permitted"] is False and ledger["retry_replay_resume_repair_replacement_permitted"] is False, "prior ledger no replay")
    require(terminal["action_id"] == OLD_ACTION_ID and terminal["status"] == "CONSUMED_ORPHAN", "prior terminal status")
    require(terminal["invocation_count_performed"] == 0 and terminal["payload_open_count"] == 0, "prior terminal cardinality")
    require(terminal["result_file_sha256"] is None and terminal["credential_consumed"] is True, "prior terminal result")
    require(terminal["retry_replay_resume_repair_replacement_permitted"] is False, "prior terminal no replay")
    require(not os.path.lexists(OLD_RESULT) and not os.path.lexists(OLD_CREDENTIAL), "prior terminal absent result or credential")


def _preflight(transport_attestation: dict[str, Any], package: dict[str, Any], package_bytes: bytes) -> dict[str, Any]:
    _validate_package(package)
    for binding in package["canonical_bindings"]:
        binding_id = binding["id"]
        require(binding_id in ACCEPTED_PATHS, f"unknown binding: {binding_id}")
        require(binding["path"] == str(ACCEPTED_PATHS[binding_id]), f"binding path: {binding_id}")
        require(binding["sha256"] == EXPECTED_BINDING_HASHES[binding_id], f"binding hash declaration: {binding_id}")
        require(sha256_bytes(read_bytes(ACCEPTED_PATHS[binding_id])) == EXPECTED_BINDING_HASHES[binding_id], f"binding bytes: {binding_id}")
    for binding in package["local_artifact_bindings"]:
        binding_id = binding["id"]
        require(binding["path"] == str(LOCAL_PATHS[binding_id]), f"local path: {binding_id}")
        require(sha256_bytes(read_bytes(LOCAL_PATHS[binding_id])) == binding["sha256"], f"local bytes: {binding_id}")
    _verify_prior_terminal()
    require(sha256_bytes(read_bytes(INTERPRETER)) == INTERPRETER_SHA256, "interpreter hash")
    require(sys.version.split()[0] == "3.13.5", "interpreter version")
    for distribution, version in SCHEMA_STACK.items():
        require(importlib.metadata.version(distribution) == version, f"schema stack: {distribution}")
    import jsonschema

    schemas: dict[str, dict[str, Any]] = {}
    for binding_id in ("authority_schema", "credential_schema", "ledger_schema", "first_terminal_schema", "fresh_l2_acceptance_schema"):
        schema, _ = read_canonical_json(LOCAL_PATHS[binding_id], binding_id)
        jsonschema.Draft202012Validator.check_schema(schema)
        schemas[binding_id] = schema
    require(sha256_bytes(read_bytes(LANE_METADATA)) == OFFICIAL_IDENTITIES["lane_metadata_sha256"], "lane metadata")
    tensor_stat = os.lstat(SEALED_TENSOR)
    require(stat.S_ISREG(tensor_stat.st_mode), "sealed tensor file type")
    require(stat.S_IMODE(tensor_stat.st_mode) == 0o400 and tensor_stat.st_size == 1305797, "sealed tensor lstat")
    require(not os.path.lexists(LIVE_ROOT), "new live namespace already exists")
    acceptance, acceptance_bytes = read_canonical_json(ACCEPTANCE, "Fresh-L2 acceptance")
    verify_self_checksum(acceptance, "acceptance_sha256", "Fresh-L2 acceptance")
    require(acceptance["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_fresh_l2_static_acceptance", "acceptance kind")
    require(acceptance["action_id"] == ACTION_ID, "acceptance action")
    require(acceptance["decision"] == "ACCEPT_STATIC_PACKAGE" and acceptance["reviewer_role"] == "Fresh-L2", "acceptance decision")
    require(acceptance["static_acceptance_grants_execution_authority"] is False, "acceptance authority boundary")
    require(acceptance["execution_package_file_sha256"] == sha256_bytes(package_bytes), "acceptance package binding")
    jsonschema.Draft202012Validator(schemas["fresh_l2_acceptance_schema"]).validate(acceptance)
    return {
        "acceptance_sha256": sha256_bytes(acceptance_bytes),
        "invocation_sha256": package["future_invocation"]["invocation_sha256"],
        "launcher_invocation_sha256": package["launcher_invocation"]["invocation_sha256"],
        "package": package,
        "package_file_sha256": sha256_bytes(package_bytes),
        "schemas": schemas,
        "transport_attestation": transport_attestation,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    if path == ROOT:
        return
    if path.exists():
        require(path.is_dir() and not path.is_symlink(), f"unsafe live directory: {path}")
        return
    _ensure_directory(path.parent)
    os.mkdir(path, 0o700)
    _fsync_directory(path.parent)


def _durable_create(path: Path, data: bytes, mode: int = 0o400) -> None:
    _ensure_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            require(written > 0, f"short write: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    observed = os.lstat(path)
    require(stat.S_ISREG(observed.st_mode) and stat.S_IMODE(observed.st_mode) == mode, f"published mode: {path}")
    _fsync_directory(path.parent)


def _durable_unlink(path: Path) -> None:
    os.unlink(path)
    _fsync_directory(path.parent)


def _terminal_record(
    *,
    status: str,
    reason_code: str,
    authority_sha256: str | None,
    consumed_ledger_sha256: str | None,
    result_file_sha256: str | None,
    credential_consumed: bool,
    invocation_count: int,
    payload_open_count: int,
    orphaned: bool,
    failure_detail: str,
) -> dict[str, Any]:
    return sealed(
        {
            "action_id": ACTION_ID,
            "action_retired": True,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_first_terminal",
            "authority_sha256": authority_sha256,
            "consumed_ledger_sha256": consumed_ledger_sha256,
            "credential_consumed": credential_consumed,
            "failure_detail_sha256": sha256_bytes(failure_detail.encode("utf-8")),
            "first_record_immutable": True,
            "invocation_count_performed": invocation_count,
            "orphaned_after_consumption": orphaned,
            "payload_open_count": payload_open_count,
            "reason_code": reason_code,
            "result_file_sha256": result_file_sha256,
            "retry_replay_resume_repair_replacement_permitted": False,
            "status": status,
        },
        "first_terminal_sha256",
    )


def _publish_terminal(record: dict[str, Any]) -> None:
    _durable_create(LIVE_PATHS["first_terminal"], compact_bytes(record))


def _retire_preflight_failure(error: BaseException, reason_code: str) -> int:
    record = _terminal_record(
        status="PREFLIGHT_FAILED_TERMINAL",
        reason_code=reason_code,
        authority_sha256=None,
        consumed_ledger_sha256=None,
        result_file_sha256=None,
        credential_consumed=False,
        invocation_count=0,
        payload_open_count=0,
        orphaned=False,
        failure_detail=f"{type(error).__name__}:{error}",
    )
    try:
        _publish_terminal(record)
    except Exception:
        return 4
    return 3


def _validate_generated(context: dict[str, Any], schema_id: str, record: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.Draft202012Validator(context["schemas"][schema_id]).validate(record)


def _authority(context: dict[str, Any]) -> dict[str, Any]:
    return sealed(
        {
            "action_id": ACTION_ID,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_authority",
            "execution_package_file_sha256": context["package_file_sha256"],
            "fresh_l2_acceptance_sha256": context["acceptance_sha256"],
            "invocation_sha256": context["invocation_sha256"],
            "launcher_invocation_sha256": context["launcher_invocation_sha256"],
            "official_identities": OFFICIAL_IDENTITIES,
            "single_use": True,
            "state": "READY_UNCONSUMED",
            "transport_attestation": context["transport_attestation"],
        },
        "authority_sha256",
    )


def _credential(authority: dict[str, Any]) -> dict[str, Any]:
    return sealed(
        {
            "action_id": ACTION_ID,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_credential",
            "authority_sha256": authority["authority_sha256"],
            "credential_nonce_sha256": sha256_bytes(os.urandom(32)),
            "single_use": True,
            "state": "READY_UNCONSUMED",
        },
        "credential_sha256",
    )


def _ledger(authority: dict[str, Any], credential: dict[str, Any], package_file_sha256: str) -> dict[str, Any]:
    return sealed(
        {
            "action_id": ACTION_ID,
            "action_retired": True,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_consumed_ledger",
            "authority_sha256": authority["authority_sha256"],
            "credential_consumption": {
                "credential_sha256": credential["credential_sha256"],
                "state": "CONSUMED_BEFORE_PAYLOAD",
            },
            "execution_package_file_sha256": package_file_sha256,
            "invocation_cardinality": {"invocation_count_performed_at_publish": 0, "invocation_count_permitted": 1},
            "payload_cardinality": {"payload_open_count_performed_at_publish": 0, "payload_open_count_permitted": 1},
            "post_consumption_ambiguity_default": "CONSUMED_ORPHAN",
            "replay_permitted": False,
            "retry_replay_resume_repair_replacement_permitted": False,
            "transport_attestation_sha256": authority["transport_attestation"]["transport_attestation_sha256"],
        },
        "consumed_ledger_sha256",
    )


def _load_module(name: str, path: Path) -> Any:
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module spec: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_tensor_once(counter: dict[str, int]) -> bytes:
    require(counter["payload_open_count"] == 0, "payload already opened")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(SEALED_TENSOR, flags)
    counter["payload_open_count"] = 1
    try:
        observed = os.fstat(descriptor)
        require(stat.S_ISREG(observed.st_mode) and observed.st_size == 1305797, "payload fstat")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _run_consumed(context: dict[str, Any], authority: dict[str, Any], ledger: dict[str, Any]) -> int:
    counter = {"invocation_count": 0, "payload_open_count": 0}
    result_file_sha256: str | None = None
    try:
        controller = _load_module("accepted_v8_controller", ACCEPTED_PATHS["controller"])
        evaluator = _load_module("accepted_v8_evaluator", ACCEPTED_PATHS["evaluator"])
        parser = _load_module("accepted_v8_c02_parser", ACCEPTED_PATHS["c02_parser"])
        accepted_package, _ = read_canonical_json(ACCEPTED_PATHS["package"], "accepted V8 package")
        result_schema, _ = read_canonical_json(ACCEPTED_PATHS["result_schema"], "accepted V8 result schema")
        controller.validate_package(accepted_package)
        tensor_records = accepted_package["official_benchmark"]["input_bindings"]["tensor_records"]
        bindings = {
            record["tensor_name"]: {"dtype": record["dtype"], "sha256": record["sha256"], "shape": record["shape"]}
            for record in tensor_records.values()
        }
        result_context = {
            "authority_sha256": authority["authority_sha256"],
            "consumed_ledger_sha256": ledger["consumed_ledger_sha256"],
            "evaluator_sha256": EXPECTED_BINDING_HASHES["evaluator"],
            "fresh_l2_acceptance_sha256": context["acceptance_sha256"],
            "input_bindings": accepted_package["official_benchmark"]["input_bindings"],
            "invocation_sha256": context["invocation_sha256"],
            "irreversible_action_id": ACTION_ID,
            "model_identity_sha256": OFFICIAL_IDENTITIES["model_sha256"],
            "package_sha256": EXPECTED_BINDING_HASHES["package"],
        }
        counter["invocation_count"] = 1
        result_bytes = evaluator.evaluate_bundle_once(lambda: _read_tensor_once(counter), bindings, parser, result_context)
        result = json.loads(result_bytes.decode("ascii"))
        controller.validate_result_record(
            accepted_package,
            result,
            package_sha256=EXPECTED_BINDING_HASHES["package"],
            evaluator_sha256=EXPECTED_BINDING_HASHES["evaluator"],
            expected_authority_sha256=authority["authority_sha256"],
            expected_consumed_ledger_sha256=ledger["consumed_ledger_sha256"],
            expected_fresh_l2_acceptance_sha256=context["acceptance_sha256"],
            expected_invocation_sha256=context["invocation_sha256"],
            expected_irreversible_action_id=ACTION_ID,
            expected_model_identity_sha256=OFFICIAL_IDENTITIES["model_sha256"],
            expected_tensor_bundle_sha256=OFFICIAL_IDENTITIES["tensor_bundle_sha256"],
        )
        import jsonschema

        jsonschema.Draft202012Validator(result_schema).validate(result)
        _durable_create(LIVE_PATHS["result"], result_bytes)
        result_file_sha256 = sha256_bytes(result_bytes)
        terminal = result["terminal"]
        terminal_record = _terminal_record(
            status=terminal["status"],
            reason_code=terminal["reason_code"],
            authority_sha256=authority["authority_sha256"],
            consumed_ledger_sha256=ledger["consumed_ledger_sha256"],
            result_file_sha256=result_file_sha256,
            credential_consumed=True,
            invocation_count=1,
            payload_open_count=1,
            orphaned=False,
            failure_detail="",
        )
        _validate_generated(context, "first_terminal_schema", terminal_record)
        _publish_terminal(terminal_record)
        return 0 if terminal["status"] == "SUCCEEDED_TERMINAL" else 1
    except Exception as error:
        try:
            terminal_record = _terminal_record(
                status="CONSUMED_ORPHAN",
                reason_code="POST_CONSUMPTION_AMBIGUITY",
                authority_sha256=authority["authority_sha256"],
                consumed_ledger_sha256=ledger["consumed_ledger_sha256"],
                result_file_sha256=result_file_sha256,
                credential_consumed=True,
                invocation_count=counter["invocation_count"],
                payload_open_count=counter["payload_open_count"],
                orphaned=True,
                failure_detail=f"{type(error).__name__}:{error}",
            )
            _validate_generated(context, "first_terminal_schema", terminal_record)
            _publish_terminal(terminal_record)
        except Exception:
            pass
        return 2


def main() -> int:
    try:
        transport_attestation, package, package_bytes = _attest_transport()
    except Exception as error:
        return _retire_preflight_failure(error, "TRANSPORT_ATTESTATION_FAILED")
    try:
        context = _preflight(transport_attestation, package, package_bytes)
    except Exception as error:
        return _retire_preflight_failure(error, "READ_ONLY_PREFLIGHT_FAILED")
    authority = _authority(context)
    credential = _credential(authority)
    ledger = _ledger(authority, credential, context["package_file_sha256"])
    _validate_generated(context, "authority_schema", authority)
    _validate_generated(context, "credential_schema", credential)
    _validate_generated(context, "ledger_schema", ledger)
    authority_published = False
    ledger_published = False
    try:
        _durable_create(LIVE_PATHS["authority"], compact_bytes(authority))
        authority_published = True
        _durable_create(LIVE_PATHS["credential"], compact_bytes(credential))
        _durable_create(LIVE_PATHS["ledger"], compact_bytes(ledger))
        ledger_published = True
        _durable_unlink(LIVE_PATHS["credential"])
    except Exception as error:
        post_ledger = ledger_published
        try:
            terminal_record = _terminal_record(
                status="CONSUMED_ORPHAN" if post_ledger else "FAILED_TERMINAL",
                reason_code="POST_CONSUMPTION_AMBIGUITY" if post_ledger else "LIVE_MATERIALIZATION_FAILED",
                authority_sha256=authority["authority_sha256"] if authority_published else None,
                consumed_ledger_sha256=ledger["consumed_ledger_sha256"] if post_ledger else None,
                result_file_sha256=None,
                credential_consumed=post_ledger,
                invocation_count=0,
                payload_open_count=0,
                orphaned=post_ledger,
                failure_detail=f"{type(error).__name__}:{error}",
            )
            _validate_generated(context, "first_terminal_schema", terminal_record)
            _publish_terminal(terminal_record)
        except Exception:
            pass
        return 2
    return _run_consumed(context, authority, ledger)


if __name__ == "__main__":
    raise SystemExit(main())
