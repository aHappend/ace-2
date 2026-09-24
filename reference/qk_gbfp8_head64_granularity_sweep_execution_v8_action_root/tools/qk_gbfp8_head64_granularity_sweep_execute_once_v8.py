#!/usr/bin/env python3
"""Future Base V8 execute-once launcher.

This file is inert unless invoked with the exact package-bound argv,
environment, and working directory.  Static package verification imports
nothing from this module and never calls main().
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


ACTION_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v8_action_root")
ACCEPTED_V8_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root")
V6_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
EXECUTION_PACKAGE = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_PACKAGE.json"
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
EXECUTOR = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_v8.py"
ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1614Z"
RETIRED_ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1610Z"
EXACT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}
EXACT_ARGV = [
    str(INTERPRETER),
    str(EXECUTOR),
    "--package",
    str(EXECUTION_PACKAGE),
    "--acceptance",
    str(ACCEPTANCE),
    "--irreversible-action-id",
    ACTION_ID,
]
LIVE_ROOT = ACTION_ROOT / "live"
LIVE_PATHS = {
    "authority": LIVE_ROOT / "authority/base/authority.json",
    "credential": LIVE_ROOT / "authority/base/credential.json",
    "ledger": LIVE_ROOT / "authority/base/authority-ledger.json",
    "result": LIVE_ROOT / "result/base/result.json",
    "first_terminal": LIVE_ROOT / "authority/base/first-terminal.json",
}
SEALED_TENSOR = V6_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
LANE_METADATA = V6_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json"
ACCEPTED_PATHS = {
    "accepted_v8_manifest": ACCEPTED_V8_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_MANIFEST.json",
    "runtime_package": ACCEPTED_V8_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RUNTIME_PACKAGE.json",
    "package": ACCEPTED_V8_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.json",
    "result_schema": ACCEPTED_V8_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.json",
    "controller": ACCEPTED_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v8.py",
    "evaluator": ACCEPTED_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py",
    "c02_parser": ACCEPTED_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py",
    "decisive_verifier": ACCEPTED_V8_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_static_v8.py",
    "fresh_l2_handoff": Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/f0fa5c269681/round-0007.json"),
}
EXPECTED_BINDING_HASHES = {
    "accepted_v8_manifest": "52324ad6ebd1b831774b9fae6923015f2b4cb24f5cde40e3f7d83435651efd64",
    "runtime_package": "ef2ba57e278be4fb24f6fa56faadfe84a91123781fadab5b6110163b3b5518be",
    "package": "b7c758c7141d7ac8037502da4a00fe9e0780d30501e2af483e9b8c571451493a",
    "result_schema": "fc5cf29d7ec7486b106edac592787b5888557c4b1b7453cb0a668b02fca113ec",
    "controller": "faab3064733be0d0fc68c565f00dd2c77022b7f4974d39b798eda1f150d55aa4",
    "evaluator": "8ab74c7397006c9f419059f295613ce4a743a3e0b540176ba7cf6dbb6efd7f63",
    "c02_parser": "ff9216d58bcd1584361e17d8d1854f3bcb96923c17927b1f309861546807bdb1",
    "decisive_verifier": "67da980b637b93af6784dd8ab11ceb727b37ba4f5357e41c758c53015d0ff6eb",
    "fresh_l2_handoff": "2254dd910cf2aae776d73c190402b42f2db66c252a80a1a0d958618f577798e3",
}
EXPECTED_BINDING_ORDER = [
    "fresh_l2_handoff",
    "accepted_v8_manifest",
    "runtime_package",
    "package",
    "result_schema",
    "controller",
    "evaluator",
    "c02_parser",
    "decisive_verifier",
]
LOCAL_PATHS = {
    "authority_schema": ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_AUTHORITY_SCHEMA.json",
    "credential_schema": ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_CREDENTIAL_SCHEMA.json",
    "ledger_schema": ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_LEDGER_SCHEMA.json",
    "first_terminal_schema": ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_FIRST_TERMINAL_SCHEMA.json",
    "fresh_l2_acceptance_schema": ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_FRESH_L2_ACCEPTANCE_SCHEMA.json",
    "executor": EXECUTOR,
    "static_verifier": ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v8.py",
}
LOCAL_BINDING_ORDER = ["authority_schema", "credential_schema", "ledger_schema", "first_terminal_schema", "fresh_l2_acceptance_schema", "executor", "static_verifier"]
INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutionError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def read_canonical_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    data = read_bytes(path)
    seen_duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal seen_duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                seen_duplicate = True
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("ascii", "strict"), object_pairs_hook=pairs, parse_constant=lambda token: (_ for _ in ()).throw(ExecutionError(token)))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExecutionError(f"{context}: invalid JSON") from error
    require(not seen_duplicate and type(value) is dict, f"{context}: duplicate key or non-object")
    require(compact_bytes(value) == data, f"{context}: noncanonical bytes")
    return value, data


def sealed(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(key, None)
    result[key] = sha256_bytes(compact_bytes(result))
    return result


def verify_self_checksum(value: dict[str, Any], key: str, context: str) -> None:
    observed = value.get(key)
    require(type(observed) is str and len(observed) == 64, f"{context}: checksum syntax")
    payload = dict(value)
    payload.pop(key)
    require(sha256_bytes(compact_bytes(payload)) == observed, f"{context}: checksum mismatch")


def _load_module(name: str, path: Path) -> Any:
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module spec: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_package(package: dict[str, Any]) -> None:
    verify_self_checksum(package, "package_content_sha256", "execution package")
    require(package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v8_package", "package kind")
    require(package["action_identity"]["future_action_id"] == ACTION_ID, "action id")
    require(package["action_identity"]["retired_action_ids"] == [RETIRED_ACTION_ID], "retired action id")
    require(package["future_invocation"]["argv"] == EXACT_ARGV, "future argv")
    require(package["future_invocation"]["environment"] == EXACT_ENVIRONMENT, "future environment")
    require(package["future_invocation"]["cwd"] == str(ACTION_ROOT), "future cwd")
    require(
        [binding.get("id") for binding in package["canonical_bindings"]] == EXPECTED_BINDING_ORDER,
        "canonical binding completeness and order",
    )
    require(
        len({binding["id"] for binding in package["canonical_bindings"]}) == len(EXPECTED_BINDING_ORDER)
        and len({binding["path"] for binding in package["canonical_bindings"]}) == len(EXPECTED_BINDING_ORDER),
        "canonical binding uniqueness",
    )
    require(
        [binding.get("id") for binding in package["local_artifact_bindings"]] == LOCAL_BINDING_ORDER,
        "local binding completeness and order",
    )
    require(
        len({binding["id"] for binding in package["local_artifact_bindings"]}) == len(LOCAL_BINDING_ORDER)
        and len({binding["path"] for binding in package["local_artifact_bindings"]}) == len(LOCAL_BINDING_ORDER),
        "local binding uniqueness",
    )
    require(package["official_identities"] == OFFICIAL_IDENTITIES, "official identities")
    require(package["fixed_populations"] == FIXED_POPULATIONS, "fixed populations")
    require(
        package["population_arithmetic"]
        == {
            "oracle_unique_ranking_rows": "FROZEN_ACCEPTED_COUNT=346",
            "query_rows": "14*41=574",
            "valid_causal_scores": "14*(41*42/2)=12054",
        },
        "population arithmetic",
    )
    require(package["selection"]["candidate_order"] == CANDIDATE_ORDER, "candidate order")
    require(package["selection"]["policy"] == "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER", "selection policy")
    require(package["selection"]["none_pass"] == {"reason_code": "HARD_THRESHOLD_FAILED", "selected_candidate": None, "status": "FAILED_TERMINAL"}, "none-pass terminal")
    require(package["schema_stack"] == SCHEMA_STACK, "schema stack")
    require(package["lifecycle"]["bound_import_bytecode_writes_permitted"] is False, "bound import bytecode policy")
    require(package["lifecycle"]["post_ledger_exception_terminal"] == "CONSUMED_ORPHAN", "post-ledger exception terminal")
    require(package["lifecycle"]["terminal_schema_conditional_invariants"] is True, "terminal conditional invariants")


def _preflight() -> dict[str, Any]:
    require(Path.cwd() == ACTION_ROOT, "working directory mismatch")
    require([sys.executable, *sys.argv] == EXACT_ARGV, "argv mismatch")
    require(dict(os.environ) == EXACT_ENVIRONMENT, "environment mismatch")
    package, package_bytes = read_canonical_json(EXECUTION_PACKAGE, "execution package")
    _validate_package(package)
    for binding in package["canonical_bindings"]:
        binding_id = binding["id"]
        require(binding_id in ACCEPTED_PATHS, f"unknown binding: {binding_id}")
        require(set(binding) == {"id", "path", "sha256"}, f"binding keys: {binding_id}")
        require(binding["path"] == str(ACCEPTED_PATHS[binding_id]), f"binding path: {binding_id}")
        require(binding["sha256"] == EXPECTED_BINDING_HASHES[binding_id], f"binding declared hash: {binding_id}")
        require(sha256_bytes(read_bytes(ACCEPTED_PATHS[binding_id])) == EXPECTED_BINDING_HASHES[binding_id], f"binding bytes: {binding_id}")
    for binding in package["local_artifact_bindings"]:
        binding_id = binding["id"]
        require(set(binding) == {"id", "path", "sha256"}, f"local binding keys: {binding_id}")
        require(binding_id in LOCAL_PATHS, f"unknown local binding: {binding_id}")
        require(binding["path"] == str(LOCAL_PATHS[binding_id]), f"local binding path: {binding_id}")
        require(sha256_bytes(read_bytes(LOCAL_PATHS[binding_id])) == binding["sha256"], f"local binding bytes: {binding_id}")
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
    require(not os.path.lexists(LIVE_ROOT), "live namespace already exists")
    acceptance, acceptance_bytes = read_canonical_json(ACCEPTANCE, "Fresh-L2 acceptance")
    verify_self_checksum(acceptance, "acceptance_sha256", "Fresh-L2 acceptance")
    require(acceptance["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v8_fresh_l2_static_acceptance", "acceptance kind")
    require(acceptance["action_id"] == ACTION_ID, "acceptance action id")
    require(acceptance["decision"] == "ACCEPT_STATIC_PACKAGE", "acceptance decision")
    require(acceptance["reviewer_role"] == "Fresh-L2", "acceptance reviewer")
    require(acceptance["static_acceptance_grants_execution_authority"] is False, "acceptance authority boundary")
    require(acceptance["execution_package_file_sha256"] == sha256_bytes(package_bytes), "acceptance package binding")
    jsonschema.Draft202012Validator(schemas["fresh_l2_acceptance_schema"]).validate(acceptance)
    invocation_sha256 = sha256_bytes(compact_bytes({"argv": EXACT_ARGV, "cwd": str(ACTION_ROOT), "environment": EXACT_ENVIRONMENT}))
    require(package["future_invocation"]["invocation_sha256"] == invocation_sha256, "invocation checksum")
    return {
        "acceptance_sha256": sha256_bytes(acceptance_bytes),
        "invocation_sha256": invocation_sha256,
        "package": package,
        "package_file_sha256": sha256_bytes(package_bytes),
        "schemas": schemas,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    if path == ACTION_ROOT:
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
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v8_first_terminal",
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


def _validate_generated(context: dict[str, Any], schema_id: str, record: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.Draft202012Validator(context["schemas"][schema_id]).validate(record)


def _retire_preflight_failure(error: BaseException) -> int:
    record = _terminal_record(
        status="PREFLIGHT_FAILED_TERMINAL",
        reason_code="READ_ONLY_PREFLIGHT_FAILED",
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


def _authority(context: dict[str, Any]) -> dict[str, Any]:
    return sealed(
        {
            "action_id": ACTION_ID,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v8_authority",
            "execution_package_file_sha256": context["package_file_sha256"],
            "fresh_l2_acceptance_sha256": context["acceptance_sha256"],
            "invocation_sha256": context["invocation_sha256"],
            "official_identities": OFFICIAL_IDENTITIES,
            "single_use": True,
            "state": "READY_UNCONSUMED",
        },
        "authority_sha256",
    )


def _credential(authority: dict[str, Any]) -> dict[str, Any]:
    return sealed(
        {
            "action_id": ACTION_ID,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v8_credential",
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
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v8_consumed_ledger",
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
        },
        "consumed_ledger_sha256",
    )


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
        bindings = {record["tensor_name"]: {"dtype": record["dtype"], "sha256": record["sha256"], "shape": record["shape"]} for record in tensor_records.values()}
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
        context = _preflight()
    except Exception as error:
        return _retire_preflight_failure(error)
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
