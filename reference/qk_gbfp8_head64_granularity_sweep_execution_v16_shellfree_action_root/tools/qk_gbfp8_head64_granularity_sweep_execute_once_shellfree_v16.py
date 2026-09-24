#!/usr/bin/env python3
"""Future Base V16 execute-once launcher behind the frozen shell-free transport.

Static verification parses this file but never imports it. If a later mission
materializes Fresh-L2 acceptance and invokes the action, every avoidable load,
dependency check, package/schema validation, and runtime binding is completed
before any authority, credential, or consumed ledger is published.
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
from types import FunctionType, ModuleType
from typing import Any, Callable, NamedTuple

from qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v16 import (
    EvaluatorReturnPathError,
    call_evaluator_once,
    complete_result_path,
    process_evaluator_return,
)


ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_action_root")
RUNTIME_CONTAINER = Path("/home/argustest/ace-2/runtime")
RUNTIME_ROOT = Path("/home/argustest/ace-2/runtime/qk_gbfp8_head64_granularity_sweep_execution_v16_7f7e672a")
EXPECTED_RUNTIME_UID = 1000
EXPECTED_RUNTIME_GID = 1000
V9_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_action_root")
V8_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_action_root")
V7_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v8_action_root")
ACCEPTED_V8_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root")
V6_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
PACKAGE = ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V16_SHELLFREE_PACKAGE.json"
ACCEPTANCE = ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
TRANSPORT = ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v16.py"
LAUNCHER = ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v16.py"
ACTION_ID = "ace2:qk-gbfp8-base-v16:execute-once:7f7e672a:20260814T100000Z"
V15_ACTION_ID = "ace2:qk-gbfp8-base-v15:execute-once:c2dfe170:20260814T084500Z"
V9_ACTION_ID = "ace2:qk-gbfp8-base-v9:execute-once:2254dd91:20260813T2013Z"
V8_ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1724Z"
V7_ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1614Z"
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
LIVE_ROOT = RUNTIME_ROOT / "primary"
FALLBACK_ROOT = RUNTIME_ROOT / "fallback"
FALLBACK_TERMINAL = FALLBACK_ROOT / "first-terminal.json"
RUNTIME_DIRECTORIES = (
    RUNTIME_ROOT,
    LIVE_ROOT,
    LIVE_ROOT / "authority",
    LIVE_ROOT / "authority/base",
    LIVE_ROOT / "result",
    LIVE_ROOT / "result/base",
    FALLBACK_ROOT,
)
LIVE_PATHS = {
    "authority": LIVE_ROOT / "authority/base/authority.json",
    "credential": LIVE_ROOT / "authority/base/credential.json",
    "ledger": LIVE_ROOT / "authority/base/authority-ledger.json",
    "result": LIVE_ROOT / "result/base/result.json",
    "first_terminal": LIVE_ROOT / "authority/base/first-terminal.json",
}
SEALED_TENSOR = V6_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
LANE_METADATA = V6_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json"
V8_PATHS = {
    "v8_authority": V8_ROOT / "live/authority/base/authority.json",
    "v8_consumed_ledger": V8_ROOT / "live/authority/base/authority-ledger.json",
    "v8_first_terminal": V8_ROOT / "live/authority/base/first-terminal.json",
}
V8_RESULT = V8_ROOT / "live/result/base/result.json"
V8_CREDENTIAL = V8_ROOT / "live/authority/base/credential.json"
V7_PATHS = {
    "v7_consumed_ledger": V7_ROOT / "live/authority/base/authority-ledger.json",
    "v7_first_terminal": V7_ROOT / "live/authority/base/first-terminal.json",
}
ACCEPTED_PATHS = {
    "preauthority_contract": Path("/home/argustest/ace-2/design/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_V16_PREAUTHORITY_ENGINEER_TASK.json"),
    "v15_package": Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v15_shellfree_action_root/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_PACKAGE.json"),
    "v15_static_acceptance": Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v15_shellfree_action_root/review/FRESH_L2_STATIC_ACCEPTANCE.json"),
    "package": Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_action_root/accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"),
    "result_schema": Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_action_root/accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json"),
    "controller": Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_controller_static_v8.py"),
    "evaluator": Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"),
    "c02_parser": Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"),
}
EXPECTED_BINDING_HASHES = {
    "preauthority_contract": "7f7e672a264924c103eda7568edac54be0a8b9a72643e75c8ac91f136e63152b",
    "v15_package": "e53777f220db67702c5886112a262bde84e262b6b2e0bb5b817ae61a1f611dd3",
    "v15_static_acceptance": "779273404f2d2a66dc3823b20d917ed0f1f1011cff19577a559998cf27b08416",
    "package": "3d36df763e775bc8f3fb5d106eaf842f7b74482c236b9697906bb93647c44832",
    "result_schema": "07f818190e7f97032a6bc3724914f00f36070f429f1cd549c67e4e7b026ae720",
    "controller": "faab3064733be0d0fc68c565f00dd2c77022b7f4974d39b798eda1f150d55aa4",
    "evaluator": "8ab74c7397006c9f419059f295613ce4a743a3e0b540176ba7cf6dbb6efd7f63",
    "c02_parser": "ff9216d58bcd1584361e17d8d1854f3bcb96923c17927b1f309861546807bdb1",
}
EXPECTED_BINDING_ORDER = list(EXPECTED_BINDING_HASHES)
LOCAL_PATHS = {
    "authority_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V16_SHELLFREE_AUTHORITY_SCHEMA.json",
    "credential_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V16_SHELLFREE_CREDENTIAL_SCHEMA.json",
    "ledger_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V16_SHELLFREE_LEDGER_SCHEMA.json",
    "first_terminal_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V16_SHELLFREE_FIRST_TERMINAL_SCHEMA.json",
    "fresh_l2_acceptance_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V16_SHELLFREE_FRESH_L2_ACCEPTANCE_SCHEMA.json",
    "shared_result_path": ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v16.py",
    "marker_free_fixture": ROOT / "tools/qk_gbfp8_head64_granularity_sweep_marker_free_fixture_v16.py",
    "fixture_report": ROOT / "evidence/MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE_REPORT.json",
    "transport": ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v16.py",
    "launcher": ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v16.py",
    "static_verifier": ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree.py",
}
LOCAL_BINDING_ORDER = list(LOCAL_PATHS)
SCHEMA_STACK = {
    "attrs": "26.1.0",
    "jsonschema": "4.23.0",
    "jsonschema-specifications": "2025.9.1",
    "referencing": "0.37.0",
    "rpds-py": "0.30.0",
}
OFFICIAL_IDENTITIES = {
    "input_tokens_sha256": "1b8c972381a2c3d7c754d1d2879b4389a13aca3f1d485f703510702c1cf2eb86",
    "lane_metadata_sha256": "4d5904378b9ccbabd434119b6a57f8d4266abf3d8e772c4c1bca98df785e7f3a",
    "model_sha256": "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7",
    "tensor_bundle_sha256": "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175",
}
FIXED_POPULATIONS = {"oracle_unique_ranking_rows": 346, "query_rows": 574, "valid_causal_scores": 12054}
CANDIDATE_ORDER = ("G8", "G4", "G2", "G1")
CANDIDATE_SPECS = (
    ("G8", 8, 8, 80, 16),
    ("G4", 4, 16, 96, 32),
    ("G2", 2, 32, 128, 64),
    ("G1", 1, 64, 192, 128),
)
RESULT_KEYS = frozenset({
    "authority_sha256", "candidate_results", "consumed_ledger_sha256", "evaluator_sha256",
    "fresh_l2_acceptance_sha256", "generation_id", "input_bindings", "invocation_sha256",
    "irreversible_action_id", "lane_label", "model_identity_sha256", "namespace_label", "package_id",
    "package_sha256", "result_sha256", "schema_id", "selected_candidate", "selection", "terminal",
})
CANDIDATE_RESULT_KEYS = frozenset({
    "bytes_per_head", "exponent_bytes_per_head", "group_count", "group_size", "label",
    "mantissa_bytes_per_head", "metrics", "threshold_evaluation",
})
SELECTION_KEYS = frozenset({"passing_candidates", "policy", "selected_candidate"})
TERMINAL_KEYS = frozenset({
    "first_record_immutable", "invocation_count_performed", "metrics_published", "reason_code",
    "retry_replay_resume_repair_permitted", "status", "tensor_open_count", "thresholds_evaluated",
})


class ExecutionError(RuntimeError):
    pass


class TransportAttestationError(ExecutionError):
    pass


class FrozenParserBindings(NamedTuple):
    parse_c02_producer_bytes: Callable[..., Any]
    parse_c02_accepted_reader: Callable[..., Any]


class RuntimeBindings(NamedTuple):
    evaluate_bundle_once: Callable[[Callable[[], bytes]], bytes]
    validate_result_record: Callable[[dict[str, Any], bytes], None]
    validate_result_schema: Callable[[dict[str, Any]], None]
    validate_terminal_record: Callable[[dict[str, Any]], None]


class ResultBindings(NamedTuple):
    acceptance_sha256: str
    authority_sha256: str
    evaluator_sha256: str
    invocation_sha256: str
    ledger_sha256: str
    model_identity_sha256: str
    package_sha256: str
    tensor_bundle_sha256: str


class PreflightContext(NamedTuple):
    acceptance_sha256: str
    evaluate_bundle_once: Callable[..., bytes]
    input_bindings: dict[str, Any]
    invocation_sha256: str
    launcher_invocation_sha256: str
    package_file_sha256: str
    parser: FrozenParserBindings
    result_validator: Any
    schema_validators: dict[str, Any]
    tensor_bindings: dict[str, dict[str, Any]]
    transport_attestation: dict[str, Any]


class PreparedRecords(NamedTuple):
    authority_sha256: str
    authority_bytes: bytes
    credential_bytes: bytes
    ledger_sha256: str
    ledger_bytes: bytes


class ExecutionPlan(NamedTuple):
    authority_sha256: str
    authority_bytes: bytes
    credential_bytes: bytes
    ledger_sha256: str
    ledger_bytes: bytes
    runtime: RuntimeBindings


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


def decode_canonical_json(data: bytes, context: str) -> dict[str, Any]:
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
    return value


def read_canonical_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    data = read_bytes(path)
    value = decode_canonical_json(data, context)
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
    require(package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_package", "package kind")
    require(package["action_identity"] == {
        "future_action_id": ACTION_ID,
        "prior_action_id": V15_ACTION_ID,
        "replacement_is_distinct": True,
        "retry_or_resume": False,
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
    require(package["selection"]["candidate_order"] == list(CANDIDATE_ORDER), "candidate order")
    require(package["selection"]["policy"] == "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER_WITH_EVERY_HARD_GATE_REQUIRED", "selection policy")
    require(package["required_disposition"] == "V16_EVALUATOR_RETURN_PATH_SUCCESSOR_STATIC_PACKAGE_READY_NO_EXECUTION_AUTHORITY", "disposition")
    require(package["failure_stage_contract"]["stages"] == ["EVALUATOR_CALL", "CANONICAL_DECODE", "RESULT_SCHEMA", "RESULT_RECORD", "RESULT_PUBLICATION", "TERMINAL_BUILD", "TERMINAL_SCHEMA", "TERMINAL_PUBLICATION"], "failure stages")
    require(package["claim_boundary"]["execution_authorized"] is False, "execution authority")
    require(package["claim_boundary"]["official_payload_open_count"] == 0, "payload opens")
    require(package["claim_boundary"]["official_target_process_starts"] == 0, "target starts")
    runtime = package["runtime_namespace"]
    require(runtime["runtime_root"] == str(RUNTIME_ROOT), "runtime root")
    require(runtime["precreated_directories"] == [str(path) for path in RUNTIME_DIRECTORIES], "runtime directories")
    require(runtime["paths"] == {key: str(path) for key, path in LIVE_PATHS.items()}, "runtime paths")
    require(runtime["fallback_terminal"] == str(FALLBACK_TERMINAL), "fallback terminal")
    require(runtime["file_count"] == 0, "runtime file count")


def _verify_v15_retirement() -> None:
    prior_package, _ = read_canonical_json(ACCEPTED_PATHS["v15_package"], "retired V15 package")
    require(prior_package["action_identity"]["future_action_id"] == V15_ACTION_ID, "retired V15 action")
    require(prior_package["claim_boundary"]["execution_authorized"] is False, "retired V15 package authority declaration")
    prior_acceptance, _ = read_canonical_json(ACCEPTED_PATHS["v15_static_acceptance"], "retired V15 static acceptance")
    require(prior_acceptance["static_acceptance_grants_execution_authority"] is False, "retired V15 acceptance authority")


def _load_frozen_module(name: str, path: Path) -> ModuleType:
    require(name not in sys.modules, f"module namespace already occupied: {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None and spec.origin == str(path), f"module spec: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(name not in sys.modules and module.__name__ == name, f"isolated module namespace: {name}")
    return module


def _verify_execution_package_and_bindings(package: dict[str, Any]) -> None:
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


def _verify_interpreter_and_schema_stack() -> None:
    require(sha256_bytes(read_bytes(INTERPRETER)) == INTERPRETER_SHA256, "interpreter hash")
    require(sys.version.split()[0] == "3.13.5", "interpreter version")
    for distribution, version in SCHEMA_STACK.items():
        require(importlib.metadata.version(distribution) == version, f"schema stack: {distribution}")


def _load_lifecycle_schema_validators() -> dict[str, Any]:
    import jsonschema

    validator_class = jsonschema.Draft202012Validator
    schema_validators: dict[str, Any] = {}
    for binding_id in ("authority_schema", "credential_schema", "ledger_schema", "first_terminal_schema", "fresh_l2_acceptance_schema"):
        schema, _ = read_canonical_json(LOCAL_PATHS[binding_id], binding_id)
        validator_class.check_schema(schema)
        schema_validators[binding_id] = validator_class(schema)
    return schema_validators


def _verify_lane_metadata_and_tensor_lstat() -> None:
    require(sha256_bytes(read_bytes(LANE_METADATA)) == OFFICIAL_IDENTITIES["lane_metadata_sha256"], "lane metadata")
    tensor_stat = os.lstat(SEALED_TENSOR)
    require(stat.S_ISREG(tensor_stat.st_mode), "sealed tensor file type")
    require(stat.S_IMODE(tensor_stat.st_mode) == 0o400 and tensor_stat.st_size == 1305797, "sealed tensor lstat")




def _validate_fresh_l2_acceptance(package_bytes: bytes, acceptance_validator: Any) -> str:
    acceptance, acceptance_bytes = read_canonical_json(ACCEPTANCE, "Fresh-L2 acceptance")
    verify_self_checksum(acceptance, "acceptance_sha256", "Fresh-L2 acceptance")
    require(acceptance["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_fresh_l2_static_acceptance", "acceptance kind")
    require(acceptance["official_payload_open_count"] == 0 and acceptance["official_target_process_starts"] == 0, "acceptance no-execution counters")
    require(acceptance["runtime_namespace_file_count"] == 0, "acceptance runtime file count")
    require(acceptance["required_disposition"] == "V16_EVALUATOR_RETURN_PATH_SUCCESSOR_STATIC_PACKAGE_READY_NO_EXECUTION_AUTHORITY", "acceptance disposition")
    require(acceptance["action_id"] == ACTION_ID, "acceptance action")
    require(acceptance["decision"] == "ACCEPT_STATIC_PACKAGE" and acceptance["reviewer_role"] == "Fresh-L2", "acceptance decision")
    require(acceptance["static_acceptance_grants_execution_authority"] is False, "acceptance authority boundary")
    require(acceptance["execution_package_file_sha256"] == sha256_bytes(package_bytes), "acceptance package binding")
    acceptance_validator.validate(acceptance)
    return sha256_bytes(acceptance_bytes)


def _load_frozen_runtime_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    sys.dont_write_bytecode = True
    controller = _load_frozen_module("v16_frozen_accepted_v8_controller", ACCEPTED_PATHS["controller"])
    evaluator = _load_frozen_module("v16_frozen_accepted_v8_evaluator", ACCEPTED_PATHS["evaluator"])
    parser_module = _load_frozen_module("v16_frozen_accepted_v8_c02_parser", ACCEPTED_PATHS["c02_parser"])
    return controller, evaluator, parser_module


def _bind_runtime_callables(
    controller: ModuleType,
    evaluator: ModuleType,
    parser_module: ModuleType,
) -> tuple[Callable[..., None], Callable[..., bytes], Callable[..., Any], Callable[..., Any]]:
    controller_validate_package = controller.validate_package
    evaluate_bundle_once = evaluator.evaluate_bundle_once
    parse_c02_producer_bytes = parser_module.parse_c02_producer_bytes
    parse_c02_accepted_reader = parser_module.parse_c02_accepted_reader
    require(type(controller_validate_package) is FunctionType, "controller validate_package callable")
    require(type(evaluate_bundle_once) is FunctionType, "evaluator evaluate_bundle_once callable")
    require(type(parse_c02_producer_bytes) is FunctionType, "C02 parse_c02_producer_bytes callable")
    require(type(parse_c02_accepted_reader) is FunctionType, "C02 parse_c02_accepted_reader callable")
    return controller_validate_package, evaluate_bundle_once, parse_c02_producer_bytes, parse_c02_accepted_reader


def _verify_runtime_dependencies(controller: ModuleType, evaluator: ModuleType, parser_module: ModuleType) -> None:
    require(controller.hashlib.__name__ == "hashlib" and controller.json.__name__ == "json" and controller.math.__name__ == "math" and controller.os.__name__ == "os" and controller.re.__name__ == "re", "controller runtime dependencies")
    require(evaluator.hashlib.__name__ == "hashlib" and evaluator.json.__name__ == "json" and evaluator.math.__name__ == "math" and evaluator.re.__name__ == "re" and evaluator.struct.__name__ == "struct", "evaluator runtime dependencies")
    require(parser_module.hashlib.__name__ == "hashlib" and parser_module.io.__name__ == "io" and parser_module.math.__name__ == "math" and parser_module.struct.__name__ == "struct", "C02 runtime dependencies")


def _read_accepted_package_and_result_schema() -> tuple[dict[str, Any], dict[str, Any]]:
    accepted_package, _ = read_canonical_json(ACCEPTED_PATHS["package"], "accepted V8 package")
    result_schema, _ = read_canonical_json(ACCEPTED_PATHS["result_schema"], "accepted V8 result schema")
    return accepted_package, result_schema


def _validate_accepted_package(controller_validate_package: Callable[..., None], accepted_package: dict[str, Any]) -> None:
    controller_validate_package(accepted_package)


def _construct_result_validator(result_schema: dict[str, Any]) -> Any:
    import jsonschema

    validator_class = jsonschema.Draft202012Validator
    validator_class.check_schema(result_schema)
    return validator_class(result_schema)


def _construct_tensor_bindings(accepted_package: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tensor_records = accepted_package["official_benchmark"]["input_bindings"]["tensor_records"]
    return {
        record["tensor_name"]: {"dtype": record["dtype"], "sha256": record["sha256"], "shape": list(record["shape"])}
        for record in tensor_records.values()
    }


def _preflight(transport_attestation: dict[str, Any], package: dict[str, Any], package_bytes: bytes) -> PreflightContext:
    _verify_execution_package_and_bindings(package)
    _verify_v15_retirement()
    _verify_interpreter_and_schema_stack()
    schema_validators = _load_lifecycle_schema_validators()
    _verify_lane_metadata_and_tensor_lstat()
    _verify_runtime_namespace_pre_authority()
    acceptance_sha256 = _validate_fresh_l2_acceptance(package_bytes, schema_validators["fresh_l2_acceptance_schema"])
    controller, evaluator, parser_module = _load_frozen_runtime_modules()
    controller_validate_package, evaluate_bundle_once, parse_c02_producer_bytes, parse_c02_accepted_reader = _bind_runtime_callables(controller, evaluator, parser_module)
    _verify_runtime_dependencies(controller, evaluator, parser_module)
    accepted_package, result_schema = _read_accepted_package_and_result_schema()
    _validate_accepted_package(controller_validate_package, accepted_package)
    result_validator = _construct_result_validator(result_schema)
    tensor_bindings = _construct_tensor_bindings(accepted_package)
    return PreflightContext(
        acceptance_sha256=acceptance_sha256,
        evaluate_bundle_once=evaluate_bundle_once,
        input_bindings=accepted_package["official_benchmark"]["input_bindings"],
        invocation_sha256=package["future_invocation"]["invocation_sha256"],
        launcher_invocation_sha256=package["launcher_invocation"]["invocation_sha256"],
        package_file_sha256=sha256_bytes(package_bytes),
        parser=FrozenParserBindings(parse_c02_producer_bytes=parse_c02_producer_bytes, parse_c02_accepted_reader=parse_c02_accepted_reader),
        result_validator=result_validator,
        schema_validators=schema_validators,
        tensor_bindings=tensor_bindings,
        transport_attestation=transport_attestation,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_runtime_directory(path: Path) -> None:
    observed = os.lstat(path)
    require(stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode), f"unsafe runtime directory: {path}")
    require(stat.S_IMODE(observed.st_mode) == 0o700, f"runtime directory mode: {path}")
    require(observed.st_uid == EXPECTED_RUNTIME_UID and observed.st_gid == EXPECTED_RUNTIME_GID, f"runtime directory owner: {path}")


def _ensure_directory(path: Path) -> None:
    require(path in RUNTIME_DIRECTORIES, f"unbound runtime directory: {path}")
    _require_runtime_directory(path)


def _probe_create_once(parent: Path, label: str) -> None:
    probe = parent / f".v16-production-{label}-create-once-probe"
    require(not os.path.lexists(probe), f"stale runtime probe: {probe}")
    _durable_create(probe, f"{ACTION_ID}:{label}\n".encode("ascii"))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    duplicate_rejected = False
    try:
        duplicate = os.open(probe, flags, 0o400)
    except FileExistsError:
        duplicate_rejected = True
    else:
        os.close(duplicate)
    require(duplicate_rejected, f"create-once probe accepted duplicate: {label}")
    _durable_unlink(probe)


def _verify_runtime_namespace_pre_authority() -> None:
    for directory in RUNTIME_DIRECTORIES:
        _require_runtime_directory(directory)
    for path in (*LIVE_PATHS.values(), FALLBACK_TERMINAL):
        require(not os.path.lexists(path), f"pre-authority runtime file exists: {path}")
    _probe_create_once(LIVE_PATHS["first_terminal"].parent, "primary-terminal")
    _probe_create_once(FALLBACK_TERMINAL.parent, "fallback-terminal")
    for path in (*LIVE_PATHS.values(), FALLBACK_TERMINAL):
        require(not os.path.lexists(path), f"runtime probe left lifecycle state: {path}")


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
    failure_stage: str | None,
) -> dict[str, Any]:
    return sealed(
        {
            "action_id": ACTION_ID,
            "action_retired": True,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_first_terminal",
            "authority_sha256": authority_sha256,
            "consumed_ledger_sha256": consumed_ledger_sha256,
            "credential_consumed": credential_consumed,
            "failure_detail_sha256": sha256_bytes(failure_detail.encode("utf-8")),
            "failure_stage": failure_stage,
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
    raw = compact_bytes(record)
    try:
        _durable_create(LIVE_PATHS["first_terminal"], raw)
        return
    except Exception as primary_error:
        try:
            _durable_create(FALLBACK_TERMINAL, raw)
            return
        except Exception as fallback_error:
            raise ExecutionError(
                f"terminal publication failed: primary={type(primary_error).__name__}:{primary_error}; "
                f"fallback={type(fallback_error).__name__}:{fallback_error}"
            ) from fallback_error


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
        failure_stage="TRANSPORT" if reason_code == "TRANSPORT_ATTESTATION_FAILED" else "PREFLIGHT",
    )
    try:
        _publish_terminal(record)
    except Exception:
        return 4
    return 3


def _validate_generated(context: PreflightContext, schema_id: str, record: dict[str, Any]) -> None:
    context.schema_validators[schema_id].validate(record)


def _authority(context: PreflightContext) -> dict[str, Any]:
    return sealed(
        {
            "action_id": ACTION_ID,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_authority",
            "execution_package_file_sha256": context.package_file_sha256,
            "fresh_l2_acceptance_sha256": context.acceptance_sha256,
            "invocation_sha256": context.invocation_sha256,
            "launcher_invocation_sha256": context.launcher_invocation_sha256,
            "official_identities": OFFICIAL_IDENTITIES,
            "single_use": True,
            "state": "READY_UNCONSUMED",
            "transport_attestation": context.transport_attestation,
        },
        "authority_sha256",
    )


def _credential(authority: dict[str, Any]) -> dict[str, Any]:
    return sealed(
        {
            "action_id": ACTION_ID,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_credential",
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
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_consumed_ledger",
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


def _prepare_records(context: PreflightContext) -> PreparedRecords:
    authority = _authority(context)
    credential = _credential(authority)
    ledger = _ledger(authority, credential, context.package_file_sha256)
    _validate_generated(context, "authority_schema", authority)
    _validate_generated(context, "credential_schema", credential)
    _validate_generated(context, "ledger_schema", ledger)
    return PreparedRecords(
        authority_sha256=authority["authority_sha256"],
        authority_bytes=compact_bytes(authority),
        credential_bytes=compact_bytes(credential),
        ledger_sha256=ledger["consumed_ledger_sha256"],
        ledger_bytes=compact_bytes(ledger),
    )


def _valid_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_result_record(result: dict[str, Any], result_bytes: bytes, bindings: ResultBindings) -> None:
    require(compact_bytes(result) == result_bytes, "result canonical bytes")
    require(set(result) == RESULT_KEYS, "result exact keys")
    require(result["package_id"] == "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE", "result package id")
    require(result["schema_id"] == "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_RESULT_V1", "result schema id")
    require(result["generation_id"] == "qk-gbfp8-head64-granularity-sweep-v8-base", "result generation")
    require(result["lane_label"] == "Base" and result["namespace_label"] == "base", "result Base lane")
    require(result["package_sha256"] == bindings.package_sha256, "result package checksum")
    require(result["evaluator_sha256"] == bindings.evaluator_sha256, "result evaluator checksum")
    require(result["authority_sha256"] == bindings.authority_sha256, "result authority checksum")
    require(result["consumed_ledger_sha256"] == bindings.ledger_sha256, "result ledger checksum")
    require(result["fresh_l2_acceptance_sha256"] == bindings.acceptance_sha256, "result acceptance checksum")
    require(result["invocation_sha256"] == bindings.invocation_sha256, "result invocation checksum")
    require(result["irreversible_action_id"] == ACTION_ID, "result action identity")
    require(result["model_identity_sha256"] == bindings.model_identity_sha256, "result model checksum")
    for key in (
        "authority_sha256", "consumed_ledger_sha256", "evaluator_sha256", "fresh_l2_acceptance_sha256",
        "invocation_sha256", "model_identity_sha256", "package_sha256", "result_sha256",
    ):
        require(_valid_sha256(result[key]), f"result checksum syntax: {key}")
    require(result["input_bindings"] == {
        "sealed_set_id": "w4a8-c02-attention-substage-trace-v2",
        "tensor_bundle_sha256": bindings.tensor_bundle_sha256,
        "tensor_record_count": 25,
    }, "result input bindings")
    candidates = result["candidate_results"]
    require(type(candidates) is list and len(candidates) == len(CANDIDATE_SPECS), "result candidate cardinality")
    passing: list[str] = []
    for candidate, (label, group_size, group_count, bytes_per_head, exponent_bytes_per_head) in zip(candidates, CANDIDATE_SPECS):
        require(type(candidate) is dict and set(candidate) == CANDIDATE_RESULT_KEYS, f"result candidate keys: {label}")
        require(candidate["label"] == label, f"result candidate label: {label}")
        require(candidate["group_size"] == group_size and candidate["group_count"] == group_count, f"result candidate groups: {label}")
        require(candidate["bytes_per_head"] == bytes_per_head, f"result candidate bytes: {label}")
        require(candidate["exponent_bytes_per_head"] == exponent_bytes_per_head, f"result candidate exponent bytes: {label}")
        require(candidate["mantissa_bytes_per_head"] == 64, f"result candidate mantissa bytes: {label}")
        if candidate["threshold_evaluation"]["all_hard_gates_pass"]:
            passing.append(label)
    selected = passing[0] if passing else None
    require(result["selected_candidate"] == selected, "result selected candidate")
    selection = result["selection"]
    require(type(selection) is dict and set(selection) == SELECTION_KEYS, "result selection keys")
    require(selection == {
        "passing_candidates": passing,
        "policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER",
        "selected_candidate": selected,
    }, "result selection")
    terminal = result["terminal"]
    require(type(terminal) is dict and set(terminal) == TERMINAL_KEYS, "result terminal keys")
    require(terminal["first_record_immutable"] is True, "result terminal immutability")
    require(terminal["invocation_count_performed"] == 1 and terminal["tensor_open_count"] == 1, "result terminal counters")
    require(terminal["metrics_published"] is True and terminal["thresholds_evaluated"] is True, "result terminal publication")
    require(terminal["retry_replay_resume_repair_permitted"] is False, "result terminal replay")
    require(terminal["status"] == ("SUCCEEDED_TERMINAL" if selected is not None else "FAILED_TERMINAL"), "result terminal status")
    require(terminal["reason_code"] == ("HARD_THRESHOLDS_PASSED" if selected is not None else "HARD_THRESHOLD_FAILED"), "result terminal reason")
    payload = dict(result)
    observed = payload.pop("result_sha256")
    require(sha256_bytes(compact_bytes(payload)) == observed, "result self-checksum")


def _freeze_execution_plan(context: PreflightContext, prepared: PreparedRecords) -> ExecutionPlan:
    acceptance_sha256 = context.acceptance_sha256
    authority_sha256 = prepared.authority_sha256
    evaluate_bundle_once = context.evaluate_bundle_once
    invocation_sha256 = context.invocation_sha256
    ledger_sha256 = prepared.ledger_sha256
    parser = context.parser
    result_validator = context.result_validator
    terminal_validator = context.schema_validators["first_terminal_schema"]
    tensor_bindings = context.tensor_bindings
    result_bindings = ResultBindings(
        acceptance_sha256=acceptance_sha256,
        authority_sha256=authority_sha256,
        evaluator_sha256=EXPECTED_BINDING_HASHES["evaluator"],
        invocation_sha256=invocation_sha256,
        ledger_sha256=ledger_sha256,
        model_identity_sha256=OFFICIAL_IDENTITIES["model_sha256"],
        package_sha256=EXPECTED_BINDING_HASHES["package"],
        tensor_bundle_sha256=OFFICIAL_IDENTITIES["tensor_bundle_sha256"],
    )
    result_context = {
        "authority_sha256": authority_sha256,
        "consumed_ledger_sha256": ledger_sha256,
        "evaluator_sha256": EXPECTED_BINDING_HASHES["evaluator"],
        "fresh_l2_acceptance_sha256": acceptance_sha256,
        "input_bindings": context.input_bindings,
        "invocation_sha256": invocation_sha256,
        "irreversible_action_id": ACTION_ID,
        "model_identity_sha256": OFFICIAL_IDENTITIES["model_sha256"],
        "package_sha256": EXPECTED_BINDING_HASHES["package"],
    }

    def evaluate_once(open_bundle: Callable[[], bytes]) -> bytes:
        return evaluate_bundle_once(open_bundle, tensor_bindings, parser, result_context)

    def validate_result_once(result: dict[str, Any], result_bytes: bytes) -> None:
        _validate_result_record(result, result_bytes, result_bindings)

    def validate_result_schema_once(result: dict[str, Any]) -> None:
        result_validator.validate(result)

    def validate_terminal_once(record: dict[str, Any]) -> None:
        terminal_validator.validate(record)

    runtime = RuntimeBindings(
        evaluate_bundle_once=evaluate_once,
        validate_result_record=validate_result_once,
        validate_result_schema=validate_result_schema_once,
        validate_terminal_record=validate_terminal_once,
    )
    return ExecutionPlan(
        authority_sha256=authority_sha256,
        authority_bytes=prepared.authority_bytes,
        credential_bytes=prepared.credential_bytes,
        ledger_sha256=ledger_sha256,
        ledger_bytes=prepared.ledger_bytes,
        runtime=runtime,
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


def _run_consumed(plan: ExecutionPlan) -> int:
    counter = {"invocation_count": 0, "payload_open_count": 0}
    result_file_sha256: str | None = None
    try:
        counter["invocation_count"] = 1
        result_bytes = call_evaluator_once(lambda: plan.runtime.evaluate_bundle_once(lambda: _read_tensor_once(counter)))
        processed = process_evaluator_return(result_bytes, plan.runtime.validate_result_schema, plan.runtime.validate_result_record)

        def publish_result(raw: bytes) -> None:
            nonlocal result_file_sha256
            _durable_create(LIVE_PATHS["result"], raw)
            result_file_sha256 = sha256_bytes(raw)

        def build_terminal(result: dict[str, Any], raw: bytes) -> dict[str, Any]:
            terminal = result["terminal"]
            return _terminal_record(
                status=terminal["status"],
                reason_code=terminal["reason_code"],
                authority_sha256=plan.authority_sha256,
                consumed_ledger_sha256=plan.ledger_sha256,
                result_file_sha256=sha256_bytes(raw),
                credential_consumed=True,
                invocation_count=1,
                payload_open_count=1,
                orphaned=False,
                failure_detail="",
                failure_stage=None,
            )

        terminal_record = complete_result_path(
            processed,
            publish_result,
            build_terminal,
            plan.runtime.validate_terminal_record,
            _publish_terminal,
        )
        return 0 if terminal_record["status"] == "SUCCEEDED_TERMINAL" else 1
    except Exception as error:
        failure_stage = error.failure_stage if isinstance(error, EvaluatorReturnPathError) else "TERMINAL_BUILD"
        try:
            terminal_record = _terminal_record(
                status="CONSUMED_ORPHAN",
                reason_code="POST_CONSUMPTION_AMBIGUITY",
                authority_sha256=plan.authority_sha256,
                consumed_ledger_sha256=plan.ledger_sha256,
                result_file_sha256=result_file_sha256,
                credential_consumed=True,
                invocation_count=counter["invocation_count"],
                payload_open_count=counter["payload_open_count"],
                orphaned=True,
                failure_detail=f"{type(error).__name__}:{error}",
                failure_stage=failure_stage,
            )
            plan.runtime.validate_terminal_record(terminal_record)
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
        prepared = _prepare_records(context)
        plan = _freeze_execution_plan(context, prepared)
        del context, prepared
    except Exception as error:
        return _retire_preflight_failure(error, "READ_ONLY_PREFLIGHT_FAILED")
    authority_published = False
    ledger_published = False
    try:
        _durable_create(LIVE_PATHS["authority"], plan.authority_bytes)
        authority_published = True
        _durable_create(LIVE_PATHS["credential"], plan.credential_bytes)
        _durable_create(LIVE_PATHS["ledger"], plan.ledger_bytes)
        ledger_published = True
        _durable_unlink(LIVE_PATHS["credential"])
    except Exception as error:
        post_ledger = ledger_published
        try:
            terminal_record = _terminal_record(
                status="CONSUMED_ORPHAN" if post_ledger else "FAILED_TERMINAL",
                reason_code="POST_CONSUMPTION_AMBIGUITY" if post_ledger else "LIVE_MATERIALIZATION_FAILED",
                authority_sha256=plan.authority_sha256 if authority_published else None,
                consumed_ledger_sha256=plan.ledger_sha256 if post_ledger else None,
                result_file_sha256=None,
                credential_consumed=post_ledger,
                invocation_count=0,
                payload_open_count=0,
                orphaned=post_ledger,
                failure_detail=f"{type(error).__name__}:{error}",
                failure_stage="LIVE_MATERIALIZATION",
            )
            plan.runtime.validate_terminal_record(terminal_record)
            _publish_terminal(terminal_record)
        except Exception:
            pass
        return 2
    return _run_consumed(plan)


if __name__ == "__main__":
    raise SystemExit(main())
