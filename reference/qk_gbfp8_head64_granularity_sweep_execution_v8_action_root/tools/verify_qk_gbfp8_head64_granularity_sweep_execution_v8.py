#!/usr/bin/env python3
"""Decisive static verifier for the Base V8 execute-once package.

The verifier hashes and parses static files, performs lstat on the sealed
tensor, and checks absence of future live paths.  It never imports or invokes
the packaged launcher, controller, evaluator, or C02 parser.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable

import jsonschema


ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v8_action_root")
PACKAGE_PATH = ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_PACKAGE.json"
EXECUTOR_PATH = ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_v8.py"
VERIFIER_PATH = ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v8.py"
V8_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root")
V6_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1614Z"
RETIRED_ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1610Z"
ACCEPTANCE_PATH = ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
EXACT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}
EXACT_ARGV = [
    str(INTERPRETER),
    str(EXECUTOR_PATH),
    "--package",
    str(PACKAGE_PATH),
    "--acceptance",
    str(ACCEPTANCE_PATH),
    "--irreversible-action-id",
    ACTION_ID,
]
EXACT_INVOCATION_SHA256 = hashlib.sha256(
    (json.dumps({"argv": EXACT_ARGV, "cwd": str(ROOT), "environment": EXACT_ENVIRONMENT}, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")
).hexdigest()
EXPECTED_BINDINGS = {
    "fresh_l2_handoff": (
        "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/f0fa5c269681/round-0007.json",
        "2254dd910cf2aae776d73c190402b42f2db66c252a80a1a0d958618f577798e3",
    ),
    "accepted_v8_manifest": (
        str(V8_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_MANIFEST.json"),
        "52324ad6ebd1b831774b9fae6923015f2b4cb24f5cde40e3f7d83435651efd64",
    ),
    "runtime_package": (
        str(V8_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RUNTIME_PACKAGE.json"),
        "ef2ba57e278be4fb24f6fa56faadfe84a91123781fadab5b6110163b3b5518be",
    ),
    "package": (
        str(V8_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.json"),
        "b7c758c7141d7ac8037502da4a00fe9e0780d30501e2af483e9b8c571451493a",
    ),
    "result_schema": (
        str(V8_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.json"),
        "fc5cf29d7ec7486b106edac592787b5888557c4b1b7453cb0a668b02fca113ec",
    ),
    "controller": (
        str(V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v8.py"),
        "faab3064733be0d0fc68c565f00dd2c77022b7f4974d39b798eda1f150d55aa4",
    ),
    "evaluator": (
        str(V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"),
        "8ab74c7397006c9f419059f295613ce4a743a3e0b540176ba7cf6dbb6efd7f63",
    ),
    "c02_parser": (
        str(V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"),
        "ff9216d58bcd1584361e17d8d1854f3bcb96923c17927b1f309861546807bdb1",
    ),
    "decisive_verifier": (
        str(V8_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_static_v8.py"),
        "67da980b637b93af6784dd8ab11ceb727b37ba4f5357e41c758c53015d0ff6eb",
    ),
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
INTERPRETER_BINDING = {"path": str(INTERPRETER), "sha256": "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad", "version": "3.13.5"}
SCHEMA_STACK = {"attrs": "26.1.0", "jsonschema": "4.23.0", "jsonschema-specifications": "2025.9.1", "referencing": "0.37.0", "rpds-py": "0.30.0"}
OFFICIAL_IDENTITIES = {
    "input_tokens_sha256": "1b8c972381a2c3d7c754d1d2879b4389a13aca3f1d485f703510702c1cf2eb86",
    "lane_metadata_sha256": "4d5904378b9ccbabd434119b6a57f8d4266abf3d8e772c4c1bca98df785e7f3a",
    "model_sha256": "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7",
    "tensor_bundle_sha256": "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175",
}
FIXED_POPULATIONS = {"oracle_unique_ranking_rows": 346, "query_rows": 574, "valid_causal_scores": 12054}
POPULATION_ARITHMETIC = {
    "oracle_unique_ranking_rows": "FROZEN_ACCEPTED_COUNT=346",
    "query_rows": "14*41=574",
    "valid_causal_scores": "14*(41*42/2)=12054",
}
LIVE_PATHS = {
    "authority": str(ROOT / "live/authority/base/authority.json"),
    "credential": str(ROOT / "live/authority/base/credential.json"),
    "first_terminal": str(ROOT / "live/authority/base/first-terminal.json"),
    "ledger": str(ROOT / "live/authority/base/authority-ledger.json"),
    "result": str(ROOT / "live/result/base/result.json"),
}
PREFLIGHT_CHECKS = [
    "exact cwd argv and four-key environment",
    "canonical execution package and self-checksum",
    "authoritative round-0007 handoff exact path and bytes",
    "all accepted V8 path and SHA-256 bindings without substitution or disagreement",
    "frozen interpreter hash CPython version and exact schema stack",
    "official model tensor token and lane-metadata identities",
    "fixed populations arithmetic candidate order and hard gates",
    "sealed tensor lstat regular mode-0400 byte-count only",
    "all future live paths and live root absent",
    "independent Fresh-L2 static acceptance exact package binding",
]
LIFECYCLE_EVENTS = [
    "READ_ONLY_PREFLIGHT_COMPLETE",
    "AUTHORITY_CREATE_ONLY_0400_FILE_AND_DIRECTORY_FSYNC",
    "CREDENTIAL_CREATE_ONLY_0400_FILE_AND_DIRECTORY_FSYNC",
    "CONSUMED_LEDGER_CREATE_ONLY_0400_FILE_AND_DIRECTORY_FSYNC",
    "CREDENTIAL_DURABLE_UNLINK",
    "DISABLE_BYTECODE_WRITES_BEFORE_BOUND_IMPORTS",
    "LOAD_BOUND_CONTROLLER_EVALUATOR_AND_C02",
    "OPEN_SEALED_TENSOR_EXACTLY_ONCE",
    "CALL_EVALUATOR_EXACTLY_ONCE_G8_G4_G2_G1",
    "RESULT_CREATE_ONLY_0400_FILE_AND_DIRECTORY_FSYNC_IF_VALID",
    "FIRST_TERMINAL_CREATE_ONLY_0400_FILE_AND_DIRECTORY_FSYNC",
]
NO_REPLAY_ACTIONS = ["retry", "replay", "resume", "repair", "replacement", "authority reuse", "credential reuse", "result replacement", "terminal replacement"]
PROHIBITED_DURING_PREPARATION = [
    "open read or hash sealed tensor",
    "create authority credential ledger projected payload result or first-terminal",
    "invoke execute-once launcher controller or evaluator",
    "create build artifact",
    "touch checkpoint-176 RTL CSR U280 XRT HBM2 wiki paper or stage-closing",
]
LOCAL_PATHS = {
    "authority_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_AUTHORITY_SCHEMA.json",
    "credential_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_CREDENTIAL_SCHEMA.json",
    "ledger_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_LEDGER_SCHEMA.json",
    "first_terminal_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_FIRST_TERMINAL_SCHEMA.json",
    "fresh_l2_acceptance_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_FRESH_L2_ACCEPTANCE_SCHEMA.json",
    "executor": EXECUTOR_PATH,
    "static_verifier": VERIFIER_PATH,
}
LOCAL_ORDER = ["authority_schema", "credential_schema", "ledger_schema", "first_terminal_schema", "fresh_l2_acceptance_schema", "executor", "static_verifier"]
SCHEMA_RECORDS = {
    "authority": ("authority_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_AUTHORITY_SCHEMA"),
    "credential": ("credential_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_CREDENTIAL_SCHEMA"),
    "consumed_ledger": ("ledger_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_LEDGER_SCHEMA"),
    "first_terminal": ("first_terminal_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_FIRST_TERMINAL_SCHEMA"),
    "fresh_l2_acceptance": ("fresh_l2_acceptance_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_FRESH_L2_ACCEPTANCE_SCHEMA"),
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def file_sha256(path: Path) -> str:
    return sha256_bytes(file_bytes(path))


def canonical_json(path: Path) -> dict[str, Any]:
    data = file_bytes(path)
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
        value = json.loads(data.decode("ascii", "strict"), object_pairs_hook=pairs, parse_constant=lambda token: (_ for _ in ()).throw(VerificationError(token)))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON: {path}") from error
    require(not duplicate and type(value) is dict, f"duplicate key or non-object: {path}")
    require(compact_bytes(value) == data, f"noncanonical JSON bytes: {path}")
    return value


def reseal(package: dict[str, Any]) -> None:
    package.pop("package_content_sha256", None)
    package["package_content_sha256"] = sha256_bytes(compact_bytes(package))


def check_self_checksum(package: dict[str, Any]) -> None:
    observed = package.get("package_content_sha256")
    require(type(observed) is str and len(observed) == 64, "package self checksum syntax")
    candidate = copy.deepcopy(package)
    candidate.pop("package_content_sha256")
    require(sha256_bytes(compact_bytes(candidate)) == observed, "package self checksum")


def observed_static_hashes() -> tuple[dict[str, str], dict[str, str]]:
    accepted = {binding_id: file_sha256(Path(path)) for binding_id, (path, _) in EXPECTED_BINDINGS.items()}
    local = {artifact_id: file_sha256(path) for artifact_id, path in LOCAL_PATHS.items()}
    return accepted, local


def validate_package(package: dict[str, Any], accepted_hashes: dict[str, str], local_hashes: dict[str, str]) -> None:
    check_self_checksum(package)
    require(set(package) == {
        "action_identity", "artifact_kind", "canonical_bindings", "canonicalization", "claim_boundary",
        "fixed_populations", "future_invocation", "interpreter", "lifecycle", "live_namespace",
        "local_artifact_bindings", "official_identities", "package_content_sha256", "population_arithmetic",
        "preflight", "prohibited_during_preparation", "record_formats", "root_id", "schema_stack",
        "schema_version", "selection"
    }, "package exact keys")
    require(package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v8_package", "package kind")
    require(package["schema_version"] == 1 and package["root_id"] == ROOT.name, "package identity")
    require(package["action_identity"] == {"future_action_id": ACTION_ID, "retired_action_ids": [RETIRED_ACTION_ID], "reuse_permitted": False}, "action identity")
    require(package["canonicalization"] == {"json": "UTF-8_ASCII_SUBSET_SORTED_KEYS_COMPACT_NEWLINE", "sha256": "RAW_FILE_BYTES", "duplicate_keys_permitted": False}, "canonicalization")
    require(package["claim_boundary"] == {
        "authority_materialized": False, "build_artifact_created": False, "controller_or_evaluator_invoked": False,
        "credential_materialized": False, "execution_authorized": False, "first_terminal_materialized": False,
        "ledger_materialized": False, "projected_payload_materialized": False, "result_materialized": False,
        "sealed_tensor_access": "LSTAT_ONLY", "static_acceptance_grants_execution_authority": False,
    }, "claim boundary")
    bindings = package["canonical_bindings"]
    require(type(bindings) is list and [item.get("id") for item in bindings] == EXPECTED_BINDING_ORDER, "canonical binding order and ids")
    require(len({item["id"] for item in bindings}) == len(bindings), "duplicate canonical binding id")
    require(len({item["path"] for item in bindings}) == len(bindings), "duplicate canonical binding path")
    for item in bindings:
        require(set(item) == {"id", "path", "sha256"}, f"binding exact keys: {item.get('id')}")
        expected_path, expected_hash = EXPECTED_BINDINGS[item["id"]]
        require(item["path"] == expected_path and item["sha256"] == expected_hash, f"binding declaration: {item['id']}")
        require(accepted_hashes[item["id"]] == expected_hash, f"binding bytes: {item['id']}")
    require(package["interpreter"] == INTERPRETER_BINDING, "interpreter binding")
    require(package["schema_stack"] == SCHEMA_STACK, "schema stack")
    require(package["official_identities"] == OFFICIAL_IDENTITIES, "official identities")
    require(package["fixed_populations"] == FIXED_POPULATIONS, "fixed populations")
    require(package["population_arithmetic"] == POPULATION_ARITHMETIC, "population arithmetic")
    future = package["future_invocation"]
    require(future == {
        "argv": EXACT_ARGV, "candidate_order_in_single_evaluator_call": ["G8", "G4", "G2", "G1"],
        "controller_evaluator_process_count": 1, "cwd": str(ROOT), "environment": EXACT_ENVIRONMENT,
        "evaluator_call_count": 1, "invocation_sha256": EXACT_INVOCATION_SHA256, "shell": False,
    }, "future invocation")
    preflight = package["preflight"]
    require(preflight == {
        "checks_in_order": PREFLIGHT_CHECKS, "creates_live_artifacts": False, "payload_access": "LSTAT_ONLY",
        "failure_terminal": {"action_retired": True, "authority_created": False, "credential_created": False, "invocation_count": 0, "payload_open_count": 0, "status": "PREFLIGHT_FAILED_TERMINAL"},
        "read_only": True,
    }, "preflight")
    require(package["live_namespace"] == {
        "directory_mode_octal": "0700", "file_mode_octal": "0400", "initial_live_root_required_absent": str(ROOT / "live"),
        "paths": LIVE_PATHS, "publication": "O_CREAT|O_EXCL then file fsync then parent-directory fsync",
    }, "live namespace")
    local = package["local_artifact_bindings"]
    require(type(local) is list and [item.get("id") for item in local] == LOCAL_ORDER, "local artifact order and ids")
    require(len({item["id"] for item in local}) == len(local), "duplicate local binding id")
    for item in local:
        require(set(item) == {"id", "path", "sha256"}, f"local binding exact keys: {item.get('id')}")
        artifact_id = item["id"]
        require(item["path"] == str(LOCAL_PATHS[artifact_id]), f"local binding path: {artifact_id}")
        require(item["sha256"] == local_hashes[artifact_id], f"local binding bytes: {artifact_id}")
    formats = package["record_formats"]
    require(set(formats) == set(SCHEMA_RECORDS), "record format ids")
    local_by_id = {item["id"]: item for item in local}
    for record_id, (artifact_id, schema_id) in SCHEMA_RECORDS.items():
        require(formats[record_id] == {
            "canonical": True, "create_only": record_id != "fresh_l2_acceptance", "file_mode_octal": "0400",
            "schema_id": schema_id, "schema_path": str(LOCAL_PATHS[artifact_id]), "schema_sha256": local_by_id[artifact_id]["sha256"],
        }, f"record format: {record_id}")
    lifecycle = package["lifecycle"]
    require(lifecycle == {
        "ambiguity_policy": "EVERY_POST_CONSUMPTION_AMBIGUITY_IS_CONSUMED_ORPHAN_AND_TERMINAL",
        "bound_import_bytecode_writes_permitted": False,
        "credential_consumption_record": "consumed-ledger.credential_consumption",
        "invocation_cardinality_record": "consumed-ledger.invocation_cardinality plus first-terminal.invocation_count_performed",
        "ordered_events": LIFECYCLE_EVENTS,
        "payload_open_count_maximum": 1,
        "post_ledger_exception_terminal": "CONSUMED_ORPHAN",
        "prohibited_after_every_terminal": NO_REPLAY_ACTIONS,
        "result_and_first_terminal_create_only": True,
        "terminal_outcomes": ["SUCCEEDED_TERMINAL", "FAILED_TERMINAL", "CONSUMED_ORPHAN", "PREFLIGHT_FAILED_TERMINAL"],
        "terminal_schema_conditional_invariants": True,
    }, "lifecycle")
    require(package["selection"] == {
        "candidate_order": ["G8", "G4", "G2", "G1"], "every_hard_gate_required": True,
        "none_pass": {"reason_code": "HARD_THRESHOLD_FAILED", "selected_candidate": None, "status": "FAILED_TERMINAL"},
        "policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER",
    }, "selection")
    require(package["prohibited_during_preparation"] == PROHIBITED_DURING_PREPARATION, "preparation prohibitions")


def validate_terminal_schema(terminal_schema: dict[str, Any]) -> None:
    jsonschema.Draft202012Validator.check_schema(terminal_schema)
    require(terminal_schema["properties"]["retry_replay_resume_repair_replacement_permitted"]["const"] is False, "terminal no replay")
    validator = jsonschema.Draft202012Validator(terminal_schema)
    digest = "a" * 64
    empty_digest = sha256_bytes(b"")

    def record(**updates: Any) -> dict[str, Any]:
        value = {
            "action_id": ACTION_ID,
            "action_retired": True,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v8_first_terminal",
            "authority_sha256": None,
            "consumed_ledger_sha256": None,
            "credential_consumed": False,
            "failure_detail_sha256": digest,
            "first_record_immutable": True,
            "first_terminal_sha256": digest,
            "invocation_count_performed": 0,
            "orphaned_after_consumption": False,
            "payload_open_count": 0,
            "reason_code": "READ_ONLY_PREFLIGHT_FAILED",
            "result_file_sha256": None,
            "retry_replay_resume_repair_replacement_permitted": False,
            "status": "PREFLIGHT_FAILED_TERMINAL",
        }
        value.update(updates)
        return value

    valid = {
        "preflight": record(),
        "live_materialization": record(authority_sha256=digest, reason_code="LIVE_MATERIALIZATION_FAILED", status="FAILED_TERMINAL"),
        "consumed_orphan_before_invocation": record(
            authority_sha256=digest,
            consumed_ledger_sha256=digest,
            credential_consumed=True,
            orphaned_after_consumption=True,
            reason_code="POST_CONSUMPTION_AMBIGUITY",
            status="CONSUMED_ORPHAN",
        ),
        "consumed_orphan_after_result": record(
            authority_sha256=digest,
            consumed_ledger_sha256=digest,
            credential_consumed=True,
            invocation_count_performed=1,
            orphaned_after_consumption=True,
            payload_open_count=1,
            reason_code="POST_CONSUMPTION_AMBIGUITY",
            result_file_sha256=digest,
            status="CONSUMED_ORPHAN",
        ),
        "succeeded": record(
            authority_sha256=digest,
            consumed_ledger_sha256=digest,
            credential_consumed=True,
            failure_detail_sha256=empty_digest,
            invocation_count_performed=1,
            payload_open_count=1,
            reason_code="HARD_THRESHOLDS_PASSED",
            result_file_sha256=digest,
            status="SUCCEEDED_TERMINAL",
        ),
        "sweep_failed": record(
            authority_sha256=digest,
            consumed_ledger_sha256=digest,
            credential_consumed=True,
            failure_detail_sha256=empty_digest,
            invocation_count_performed=1,
            payload_open_count=1,
            reason_code="HARD_THRESHOLD_FAILED",
            result_file_sha256=digest,
            status="FAILED_TERMINAL",
        ),
    }
    for name, value in valid.items():
        require(validator.is_valid(value), f"terminal valid outcome rejected: {name}")

    invalid = {
        "contradictory_success": record(status="SUCCEEDED_TERMINAL"),
        "preflight_consumed": record(consumed_ledger_sha256=digest, credential_consumed=True),
        "preflight_wrong_reason": {**valid["preflight"], "reason_code": "LIVE_MATERIALIZATION_FAILED"},
        "orphan_not_marked": {**valid["consumed_orphan_before_invocation"], "orphaned_after_consumption": False},
        "orphan_payload_without_invocation": {**valid["consumed_orphan_before_invocation"], "payload_open_count": 1},
        "orphan_result_without_payload": {**valid["consumed_orphan_before_invocation"], "result_file_sha256": digest},
        "live_failure_with_consumed_ledger": {**valid["live_materialization"], "consumed_ledger_sha256": digest},
        "sweep_failure_without_result": {**valid["sweep_failed"], "result_file_sha256": None},
        "failed_with_orphan_reason": {**valid["sweep_failed"], "reason_code": "POST_CONSUMPTION_AMBIGUITY"},
    }
    for name, value in invalid.items():
        require(not validator.is_valid(value), f"terminal contradiction accepted: {name}")


def validate_schemas() -> None:
    for artifact_id in ("authority_schema", "credential_schema", "ledger_schema", "first_terminal_schema", "fresh_l2_acceptance_schema"):
        schema = canonical_json(LOCAL_PATHS[artifact_id])
        jsonschema.Draft202012Validator.check_schema(schema)
    authority_schema = canonical_json(LOCAL_PATHS["authority_schema"])
    credential_schema = canonical_json(LOCAL_PATHS["credential_schema"])
    ledger_schema = canonical_json(LOCAL_PATHS["ledger_schema"])
    terminal_schema = canonical_json(LOCAL_PATHS["first_terminal_schema"])
    require(authority_schema["properties"]["action_id"]["const"] == ACTION_ID, "authority schema action")
    require(credential_schema["properties"]["single_use"]["const"] is True, "credential schema single use")
    require(ledger_schema["properties"]["invocation_cardinality"]["properties"]["invocation_count_permitted"]["const"] == 1, "ledger invocation cardinality")
    require(ledger_schema["properties"]["credential_consumption"]["properties"]["state"]["const"] == "CONSUMED_BEFORE_PAYLOAD", "ledger credential consumption")
    require(ledger_schema["properties"]["post_consumption_ambiguity_default"]["const"] == "CONSUMED_ORPHAN", "ledger ambiguity default")
    require(ledger_schema["properties"]["retry_replay_resume_repair_replacement_permitted"]["const"] is False, "ledger no replay")
    validate_terminal_schema(terminal_schema)


def validate_executor_source(source: str | None = None) -> None:
    if source is None:
        source = file_bytes(EXECUTOR_PATH).decode("utf-8", "strict")
    tree = ast.parse(source, filename=str(EXECUTOR_PATH))
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    require("subprocess" not in imports, "executor subprocess prohibited")
    require("eval(" not in source and "exec(" not in source, "executor dynamic execution prohibited")
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in (
        "_preflight",
        "_fsync_directory",
        "_durable_create",
        "_durable_unlink",
        "_publish_terminal",
        "_ledger",
        "_read_tensor_once",
        "_run_consumed",
        "main",
    ):
        require(name in functions, f"executor function: {name}")

    def is_attribute_call(node: ast.AST, owner: str, attribute: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == owner
            and node.func.attr == attribute
        )

    def is_name_call(node: ast.AST, name: str) -> bool:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name

    def is_name(node: ast.AST, name: str) -> bool:
        return isinstance(node, ast.Name) and node.id == name

    def is_path_parent(node: ast.AST) -> bool:
        return isinstance(node, ast.Attribute) and is_name(node.value, "path") and node.attr == "parent"

    def live_path_key(node: ast.AST) -> str | None:
        if not isinstance(node, ast.Subscript) or not is_name(node.value, "LIVE_PATHS"):
            return None
        return node.slice.value if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) else None

    directory_fsync_calls = [
        node
        for node in ast.walk(functions["_fsync_directory"])
        if is_attribute_call(node, "os", "fsync") and len(node.args) == 1 and is_name(node.args[0], "descriptor")
    ]
    require(len(directory_fsync_calls) == 1, "directory fsync implementation")

    file_fsync_calls = [
        node
        for node in ast.walk(functions["_durable_create"])
        if is_attribute_call(node, "os", "fsync") and len(node.args) == 1 and is_name(node.args[0], "descriptor")
    ]
    create_parent_fsync_calls = [
        node
        for node in ast.walk(functions["_durable_create"])
        if is_name_call(node, "_fsync_directory") and len(node.args) == 1 and is_path_parent(node.args[0])
    ]
    require(len(file_fsync_calls) == 1, "durable create file fsync")
    require(
        len(create_parent_fsync_calls) == 1 and create_parent_fsync_calls[0].lineno > file_fsync_calls[0].lineno,
        "durable create parent-directory fsync after file fsync",
    )

    unlink_calls = [
        node
        for node in ast.walk(functions["_durable_unlink"])
        if is_attribute_call(node, "os", "unlink") and len(node.args) == 1 and is_name(node.args[0], "path")
    ]
    unlink_parent_fsync_calls = [
        node
        for node in ast.walk(functions["_durable_unlink"])
        if is_name_call(node, "_fsync_directory") and len(node.args) == 1 and is_path_parent(node.args[0])
    ]
    require(
        len(unlink_calls) == 1
        and len(unlink_parent_fsync_calls) == 1
        and unlink_parent_fsync_calls[0].lineno > unlink_calls[0].lineno,
        "durable unlink parent-directory fsync",
    )

    publication_functions = {"credential": "main", "result": "_run_consumed", "first_terminal": "_publish_terminal"}
    for record_id, function_name in publication_functions.items():
        publication_calls = [
            node
            for node in ast.walk(functions[function_name])
            if is_name_call(node, "_durable_create") and node.args and live_path_key(node.args[0]) == record_id
        ]
        require(len(publication_calls) == 1, f"create-only publication site: {record_id}")

    credential_unlink_calls = [
        node
        for node in ast.walk(functions["main"])
        if is_name_call(node, "_durable_unlink") and node.args and live_path_key(node.args[0]) == "credential"
    ]
    require(len(credential_unlink_calls) == 1, "durable credential unlink site")

    ledger_seals = [
        node
        for node in ast.walk(functions["_ledger"])
        if is_name_call(node, "sealed") and node.args and isinstance(node.args[0], ast.Dict)
    ]
    require(len(ledger_seals) == 1, "ledger sealed record construction")
    ledger_fields = {
        key.value: value
        for key, value in zip(ledger_seals[0].args[0].keys, ledger_seals[0].args[0].values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    for field in ("replay_permitted", "retry_replay_resume_repair_replacement_permitted"):
        value = ledger_fields.get(field)
        require(isinstance(value, ast.Constant) and value.value is False, f"ledger no-replay field: {field}")

    load_module = functions.get("_load_module")
    require(load_module is not None and load_module.body, "executor load module")
    bytecode_guard = load_module.body[0]
    require(
        isinstance(bytecode_guard, ast.Assign)
        and len(bytecode_guard.targets) == 1
        and isinstance(bytecode_guard.targets[0], ast.Attribute)
        and isinstance(bytecode_guard.targets[0].value, ast.Name)
        and bytecode_guard.targets[0].value.id == "sys"
        and bytecode_guard.targets[0].attr == "dont_write_bytecode"
        and isinstance(bytecode_guard.value, ast.Constant)
        and bytecode_guard.value.value is True,
        "disable bytecode before bound imports",
    )
    bound_load_lines = [
        node.lineno
        for node in ast.walk(load_module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"spec_from_file_location", "exec_module"}
    ]
    require(bound_load_lines and bytecode_guard.lineno < min(bound_load_lines), "bytecode guard ordering")
    tensor_open_calls = [node for node in ast.walk(functions["_read_tensor_once"]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "open"]
    require(len(tensor_open_calls) == 1, "one payload os.open site")
    evaluator_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "evaluate_bundle_once"]
    require(len(evaluator_calls) == 1, "one evaluator call site")
    require("os.O_EXCL" in source and "mode: int = 0o400" in source, "create-only mode-0400 implementation")
    require(source.index('_durable_create(LIVE_PATHS["ledger"]') < source.index('_run_consumed(context, authority, ledger)'), "ledger before evaluator path")
    require(source.index('_durable_create(LIVE_PATHS["ledger"]') < source.index('_durable_unlink(LIVE_PATHS["credential"]'), "ledger before credential unlink")
    ledger_state_positions = [
        source.find("ledger_published = False"),
        source.find('_durable_create(LIVE_PATHS["ledger"]'),
        source.find("ledger_published = True"),
        source.find('_durable_unlink(LIVE_PATHS["credential"]'),
    ]
    require(all(position >= 0 for position in ledger_state_positions) and ledger_state_positions == sorted(ledger_state_positions), "durable ledger publication state ordering")
    require('LIVE_PATHS["ledger"].exists()' not in source, "post-ledger state must not use path existence")
    main_terminal_calls = [
        node
        for node in ast.walk(functions["main"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_terminal_record"
    ]
    conditional_calls = [
        node
        for node in main_terminal_calls
        if any(keyword.arg == "status" and isinstance(keyword.value, ast.IfExp) for keyword in node.keywords)
    ]
    require(len(conditional_calls) == 1, "one ledger-conditional materialization terminal")
    materialization_call = conditional_calls[0]
    keywords = {keyword.arg: keyword.value for keyword in materialization_call.keywords}

    def require_if_expression(name: str, test_name: str, true_value: Any, false_value: Any) -> None:
        node = keywords.get(name)
        require(
            isinstance(node, ast.IfExp)
            and isinstance(node.test, ast.Name)
            and node.test.id == test_name
            and isinstance(node.body, ast.Constant)
            and node.body.value == true_value
            and isinstance(node.orelse, ast.Constant)
            and node.orelse.value == false_value,
            f"materialization conditional: {name}",
        )

    require_if_expression("status", "post_ledger", "CONSUMED_ORPHAN", "FAILED_TERMINAL")
    require_if_expression("reason_code", "post_ledger", "POST_CONSUMPTION_AMBIGUITY", "LIVE_MATERIALIZATION_FAILED")
    consumed_ledger = keywords.get("consumed_ledger_sha256")
    require(
        isinstance(consumed_ledger, ast.IfExp)
        and isinstance(consumed_ledger.test, ast.Name)
        and consumed_ledger.test.id == "post_ledger"
        and isinstance(consumed_ledger.orelse, ast.Constant)
        and consumed_ledger.orelse.value is None,
        "materialization consumed ledger binding",
    )
    for name in ("credential_consumed", "orphaned"):
        node = keywords.get(name)
        require(isinstance(node, ast.Name) and node.id == "post_ledger", f"materialization post-ledger flag: {name}")
    post_ledger_assignments = [
        node
        for node in ast.walk(functions["main"])
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "post_ledger" for target in node.targets)
    ]
    require(
        len(post_ledger_assignments) == 1
        and isinstance(post_ledger_assignments[0].value, ast.Name)
        and post_ledger_assignments[0].value.id == "ledger_published",
        "post-ledger classification source",
    )
    require(source.index("context = _preflight()") < source.index("authority = _authority(context)"), "preflight before live records")
    require(source.count("if __name__ == \"__main__\":") == 1, "executor main guard")


def check_runtime_bindings() -> None:
    require(file_sha256(INTERPRETER) == INTERPRETER_BINDING["sha256"], "interpreter bytes")
    require(sys.version.split()[0] == "3.13.5", "verification interpreter version")
    for distribution, version in SCHEMA_STACK.items():
        require(importlib.metadata.version(distribution) == version, f"installed schema stack: {distribution}")


def check_lstat_and_zero_live() -> os.stat_result:
    tensor = V6_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
    observed = os.lstat(tensor)
    require(stat.S_ISREG(observed.st_mode), "sealed tensor regular")
    require(stat.S_IMODE(observed.st_mode) == 0o400 and observed.st_size == 1305797, "sealed tensor lstat identity")
    require(not os.path.lexists(ROOT / "live"), "zero live root")
    require(not os.path.lexists(ROOT / "build"), "zero build root")
    require(not os.path.lexists(ACCEPTANCE_PATH), "no Fresh-L2 acceptance pre-created")
    forbidden = [path for path in ROOT.rglob("*") if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}]
    require(not forbidden, "zero generated Python artifacts")
    return observed


def mutation_suite(package: dict[str, Any], accepted_hashes: dict[str, str], local_hashes: dict[str, str]) -> tuple[int, int]:
    cases: list[tuple[str, Callable[[dict[str, Any]], None]]] = []

    def add(name: str, mutator: Callable[[dict[str, Any]], None]) -> None:
        cases.append((name, mutator))

    for index, binding_id in enumerate(EXPECTED_BINDING_ORDER):
        add(f"binding_path_{binding_id}", lambda value, i=index: value["canonical_bindings"][i].__setitem__("path", "/tmp/substituted"))
        add(f"binding_hash_{binding_id}", lambda value, i=index: value["canonical_bindings"][i].__setitem__("sha256", "0" * 64))
        add(f"binding_coordinated_{binding_id}", lambda value, i=index: value["canonical_bindings"][i].update({"path": "/tmp/substituted", "sha256": "1" * 64}))
    add("duplicate_binding_disagreement", lambda value: value["canonical_bindings"].append({"id": "package", "path": "/tmp/duplicate", "sha256": "2" * 64}))
    for index, artifact_id in enumerate(LOCAL_ORDER):
        add(f"local_path_{artifact_id}", lambda value, i=index: value["local_artifact_bindings"][i].__setitem__("path", "/tmp/local-substitute"))
        add(f"local_hash_{artifact_id}", lambda value, i=index: value["local_artifact_bindings"][i].__setitem__("sha256", "3" * 64))
        add(f"local_coordinated_{artifact_id}", lambda value, i=index: value["local_artifact_bindings"][i].update({"path": "/tmp/local-substitute", "sha256": "4" * 64}))
    add("action_id", lambda value: value["action_identity"].__setitem__("future_action_id", RETIRED_ACTION_ID))
    add("retired_action_id", lambda value: value["action_identity"].__setitem__("retired_action_ids", []))
    add("action_reuse", lambda value: value["action_identity"].__setitem__("reuse_permitted", True))
    for index in range(len(EXACT_ARGV)):
        add(f"argv_{index}", lambda value, i=index: value["future_invocation"]["argv"].__setitem__(i, "MUTATED"))
    for key in EXACT_ENVIRONMENT:
        add(f"environment_{key}", lambda value, k=key: value["future_invocation"]["environment"].__setitem__(k, "MUTATED"))
    add("environment_extra", lambda value: value["future_invocation"]["environment"].__setitem__("PATH", "/bin"))
    add("invocation_sha", lambda value: value["future_invocation"].__setitem__("invocation_sha256", "5" * 64))
    add("cwd", lambda value: value["future_invocation"].__setitem__("cwd", "/tmp"))
    add("shell", lambda value: value["future_invocation"].__setitem__("shell", True))
    for key in OFFICIAL_IDENTITIES:
        add(f"identity_{key}", lambda value, k=key: value["official_identities"].__setitem__(k, "6" * 64))
    for key in SCHEMA_STACK:
        add(f"schema_stack_{key}", lambda value, k=key: value["schema_stack"].__setitem__(k, "0.0.0"))
    for key in FIXED_POPULATIONS:
        add(f"population_{key}", lambda value, k=key: value["fixed_populations"].__setitem__(k, value["fixed_populations"][k] + 1))
        add(f"arithmetic_{key}", lambda value, k=key: value["population_arithmetic"].__setitem__(k, "MUTATED"))
    add("preflight_order", lambda value: value["preflight"]["checks_in_order"].reverse())
    add("preflight_writes", lambda value: value["preflight"].__setitem__("creates_live_artifacts", True))
    add("preflight_payload", lambda value: value["preflight"].__setitem__("payload_access", "READ"))
    add("preflight_zero_invocation", lambda value: value["preflight"]["failure_terminal"].__setitem__("invocation_count", 1))
    add("preflight_retirement", lambda value: value["preflight"]["failure_terminal"].__setitem__("action_retired", False))
    add("authority_mode", lambda value: value["live_namespace"].__setitem__("file_mode_octal", "0600"))
    add("create_only_publication", lambda value: value["live_namespace"].__setitem__("publication", "truncate-and-replace"))
    add("live_path", lambda value: value["live_namespace"]["paths"].__setitem__("ledger", "/tmp/ledger"))
    add("lifecycle_order", lambda value: value["lifecycle"]["ordered_events"].reverse())
    add("credential_consumption_record", lambda value: value["lifecycle"].__setitem__("credential_consumption_record", "missing"))
    add("invocation_cardinality_record", lambda value: value["lifecycle"].__setitem__("invocation_cardinality_record", "missing"))
    add("payload_count", lambda value: value["lifecycle"].__setitem__("payload_open_count_maximum", 2))
    add("ambiguity", lambda value: value["lifecycle"].__setitem__("ambiguity_policy", "RETRY"))
    add("bound_import_bytecode", lambda value: value["lifecycle"].__setitem__("bound_import_bytecode_writes_permitted", True))
    add("post_ledger_terminal", lambda value: value["lifecycle"].__setitem__("post_ledger_exception_terminal", "FAILED_TERMINAL"))
    add("result_terminal_create_only", lambda value: value["lifecycle"].__setitem__("result_and_first_terminal_create_only", False))
    add("no_replay", lambda value: value["lifecycle"]["prohibited_after_every_terminal"].remove("replay"))
    add("terminal_outcome", lambda value: value["lifecycle"]["terminal_outcomes"].remove("CONSUMED_ORPHAN"))
    add("terminal_conditional_invariants", lambda value: value["lifecycle"].__setitem__("terminal_schema_conditional_invariants", False))
    add("candidate_order", lambda value: value["selection"]["candidate_order"].reverse())
    add("hard_gates", lambda value: value["selection"].__setitem__("every_hard_gate_required", False))
    add("selection_policy", lambda value: value["selection"].__setitem__("policy", "BEST_SCORE"))
    add("none_pass", lambda value: value["selection"]["none_pass"].__setitem__("selected_candidate", "G1"))
    add("claim_invocation", lambda value: value["claim_boundary"].__setitem__("controller_or_evaluator_invoked", True))
    add("claim_tensor", lambda value: value["claim_boundary"].__setitem__("sealed_tensor_access", "READ"))
    add("static_acceptance_authority", lambda value: value["claim_boundary"].__setitem__("static_acceptance_grants_execution_authority", True))
    add("preparation_prohibition", lambda value: value["prohibited_during_preparation"].remove("open read or hash sealed tensor"))
    for record_id in SCHEMA_RECORDS:
        add(f"record_mode_{record_id}", lambda value, r=record_id: value["record_formats"][r].__setitem__("file_mode_octal", "0600"))
        add(f"record_schema_hash_{record_id}", lambda value, r=record_id: value["record_formats"][r].__setitem__("schema_sha256", "7" * 64))

    rejected = 0
    for name, mutate in cases:
        candidate = copy.deepcopy(package)
        mutate(candidate)
        reseal(candidate)
        try:
            validate_package(candidate, accepted_hashes, local_hashes)
        except VerificationError:
            rejected += 1
        else:
            raise VerificationError(f"resealed mutation accepted: {name}")
    return rejected, len(cases)


def coordinated_local_candidate(
    package: dict[str, Any],
    accepted_hashes: dict[str, str],
    local_hashes: dict[str, str],
    artifact_id: str,
    mutated_bytes: bytes,
) -> tuple[dict[str, Any], dict[str, str]]:
    candidate = copy.deepcopy(package)
    mutated_hashes = dict(local_hashes)
    mutated_hashes[artifact_id] = sha256_bytes(mutated_bytes)
    for binding in candidate["local_artifact_bindings"]:
        if binding["id"] == artifact_id:
            binding["sha256"] = mutated_hashes[artifact_id]
    for record_id, (schema_artifact_id, _) in SCHEMA_RECORDS.items():
        if schema_artifact_id == artifact_id:
            candidate["record_formats"][record_id]["schema_sha256"] = mutated_hashes[artifact_id]
    reseal(candidate)
    validate_package(candidate, accepted_hashes, mutated_hashes)
    return candidate, mutated_hashes


def semantic_mutation_suite(package: dict[str, Any], accepted_hashes: dict[str, str], local_hashes: dict[str, str]) -> tuple[int, int]:
    rejected = 0
    total = 0
    source = file_bytes(EXECUTOR_PATH).decode("utf-8", "strict")
    executor_mutations = [
        ("executor_bytecode_guard_false", source.replace("    sys.dont_write_bytecode = True\n", "    sys.dont_write_bytecode = False\n", 1)),
        (
            "executor_bytecode_guard_after_spec",
            source.replace(
                "    sys.dont_write_bytecode = True\n    spec = importlib.util.spec_from_file_location(name, path)\n",
                "    spec = importlib.util.spec_from_file_location(name, path)\n    sys.dont_write_bytecode = True\n",
                1,
            ),
        ),
        (
            "executor_durable_create_file_fsync_removed",
            source.replace(
                "            offset += written\n        os.fsync(descriptor)\n    finally:\n",
                "            offset += written\n    finally:\n",
                1,
            ),
        ),
        (
            "executor_durable_create_parent_fsync_removed",
            source.replace(
                "    _fsync_directory(path.parent)\n\n\ndef _durable_unlink",
                "\n\ndef _durable_unlink",
                1,
            ),
        ),
        (
            "executor_credential_create_only_removed",
            source.replace(
                '        _durable_create(LIVE_PATHS["credential"], compact_bytes(credential))\n',
                '        LIVE_PATHS["credential"].write_bytes(compact_bytes(credential))\n',
                1,
            ),
        ),
        (
            "executor_result_create_only_removed",
            source.replace(
                '        _durable_create(LIVE_PATHS["result"], result_bytes)\n',
                '        LIVE_PATHS["result"].write_bytes(result_bytes)\n',
                1,
            ),
        ),
        (
            "executor_terminal_create_only_removed",
            source.replace(
                '    _durable_create(LIVE_PATHS["first_terminal"], compact_bytes(record))\n',
                '    LIVE_PATHS["first_terminal"].write_bytes(compact_bytes(record))\n',
                1,
            ),
        ),
        (
            "executor_credential_unlink_parent_fsync_removed",
            source.replace(
                "def _durable_unlink(path: Path) -> None:\n    os.unlink(path)\n    _fsync_directory(path.parent)\n",
                "def _durable_unlink(path: Path) -> None:\n    os.unlink(path)\n",
                1,
            ),
        ),
        (
            "executor_ledger_no_replay_fields_true",
            source.replace(
                '            "replay_permitted": False,\n            "retry_replay_resume_repair_replacement_permitted": False,\n',
                '            "replay_permitted": True,\n            "retry_replay_resume_repair_replacement_permitted": True,\n',
                1,
            ),
        ),
        ("executor_ledger_publish_state_false", source.replace("        ledger_published = True\n", "        ledger_published = False\n", 1)),
        ("executor_post_ledger_status_failed", source.replace('                    status="CONSUMED_ORPHAN" if post_ledger else "FAILED_TERMINAL",\n', '                    status="FAILED_TERMINAL",\n', 1)),
        ("executor_post_ledger_reason_live", source.replace('                    reason_code="POST_CONSUMPTION_AMBIGUITY" if post_ledger else "LIVE_MATERIALIZATION_FAILED",\n', '                    reason_code="LIVE_MATERIALIZATION_FAILED",\n', 1)),
        ("executor_post_ledger_credential_false", source.replace("                    credential_consumed=post_ledger,\n", "                    credential_consumed=False,\n", 1)),
        ("executor_post_ledger_orphan_false", source.replace("                    orphaned=post_ledger,\n", "                    orphaned=False,\n", 1)),
    ]
    for name, mutated_source in executor_mutations:
        total += 1
        require(mutated_source != source, f"semantic mutation did not alter executor: {name}")
        coordinated_local_candidate(package, accepted_hashes, local_hashes, "executor", mutated_source.encode("utf-8"))
        try:
            validate_executor_source(mutated_source)
        except VerificationError:
            rejected += 1
        else:
            raise VerificationError(f"coordinated semantic mutation accepted: {name}")

    terminal_schema = canonical_json(LOCAL_PATHS["first_terminal_schema"])

    def status_then(schema: dict[str, Any], status: str) -> dict[str, Any]:
        for clause in schema["allOf"]:
            if clause.get("if", {}).get("properties", {}).get("status", {}).get("const") == status:
                return clause["then"]
        raise VerificationError(f"terminal status condition missing: {status}")

    def failed_branch(schema: dict[str, Any], reason: str) -> dict[str, Any]:
        for branch in status_then(schema, "FAILED_TERMINAL")["oneOf"]:
            if branch["properties"]["reason_code"].get("const") == reason:
                return branch
        raise VerificationError(f"failed terminal reason branch missing: {reason}")

    def remove_all_conditions(schema: dict[str, Any]) -> None:
        schema.pop("allOf")

    def remove_preflight_reason(schema: dict[str, Any]) -> None:
        status_then(schema, "PREFLIGHT_FAILED_TERMINAL")["properties"].pop("reason_code")

    def remove_orphan_cardinality(schema: dict[str, Any]) -> None:
        status_then(schema, "CONSUMED_ORPHAN").pop("oneOf")

    def allow_unmarked_orphan(schema: dict[str, Any]) -> None:
        status_then(schema, "CONSUMED_ORPHAN")["properties"]["orphaned_after_consumption"]["const"] = False

    def allow_zero_count_success(schema: dict[str, Any]) -> None:
        status_then(schema, "SUCCEEDED_TERMINAL")["properties"]["invocation_count_performed"]["const"] = 0

    def allow_live_failure_ledger(schema: dict[str, Any]) -> None:
        failed_branch(schema, "LIVE_MATERIALIZATION_FAILED")["properties"].pop("consumed_ledger_sha256")

    def allow_resultless_sweep_failure(schema: dict[str, Any]) -> None:
        failed_branch(schema, "HARD_THRESHOLD_FAILED")["properties"].pop("result_file_sha256")

    schema_mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("terminal_conditions_removed", remove_all_conditions),
        ("terminal_preflight_reason_uncoupled", remove_preflight_reason),
        ("terminal_orphan_cardinality_uncoupled", remove_orphan_cardinality),
        ("terminal_orphan_flag_false", allow_unmarked_orphan),
        ("terminal_success_zero_invocation", allow_zero_count_success),
        ("terminal_live_failure_ledger_allowed", allow_live_failure_ledger),
        ("terminal_sweep_failure_result_optional", allow_resultless_sweep_failure),
    ]
    for name, mutate in schema_mutations:
        total += 1
        mutated_schema = copy.deepcopy(terminal_schema)
        mutate(mutated_schema)
        mutated_bytes = compact_bytes(mutated_schema)
        coordinated_local_candidate(package, accepted_hashes, local_hashes, "first_terminal_schema", mutated_bytes)
        try:
            validate_terminal_schema(mutated_schema)
        except VerificationError:
            rejected += 1
        else:
            raise VerificationError(f"coordinated semantic mutation accepted: {name}")
    return rejected, total


def main() -> int:
    package = canonical_json(PACKAGE_PATH)
    accepted_hashes, local_hashes = observed_static_hashes()
    validate_package(package, accepted_hashes, local_hashes)
    validate_schemas()
    validate_executor_source()
    check_runtime_bindings()
    tensor_stat = check_lstat_and_zero_live()
    package_rejected, package_total = mutation_suite(package, accepted_hashes, local_hashes)
    semantic_rejected, semantic_total = semantic_mutation_suite(package, accepted_hashes, local_hashes)
    for path in sorted([PACKAGE_PATH, *LOCAL_PATHS.values()], key=str):
        print(f"FILE_SHA256 {file_sha256(path)} {path}")
    print(f"MUTATIONS_REJECTED {package_rejected + semantic_rejected}/{package_total + semantic_total}")
    print(f"COORDINATED_SEMANTIC_MUTATIONS_REJECTED {semantic_rejected}/{semantic_total}")
    print(f"SEALED_TENSOR_LSTAT_ONLY regular size={tensor_stat.st_size} mode={stat.S_IMODE(tensor_stat.st_mode):04o}")
    print("ZERO_LIVE_ARTIFACTS authority=0 credential=0 ledger=0 projected_payload=0 result=0 first_terminal=0 build=0")
    print("CONTROLLER_EVALUATOR_INVOCATIONS 0")
    print("PASS qk_gbfp8_head64_granularity_sweep_execution_v8_static")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(1)
