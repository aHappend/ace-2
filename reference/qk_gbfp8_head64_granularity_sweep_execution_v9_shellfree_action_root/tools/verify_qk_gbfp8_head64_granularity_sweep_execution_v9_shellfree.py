#!/usr/bin/env python3
"""Decisive static verifier for the authority-free Base V9 bundle.

The verifier parses and hashes static files, validates coordinated in-memory
mutations, and lstats the sealed tensor. It never imports or invokes the
transport, launcher, accepted controller, evaluator, or C02 parser.
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
import tempfile
from pathlib import Path
from typing import Any, Callable

import jsonschema


ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_action_root")
V8_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_action_root")
V7_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v8_action_root")
ACCEPTED_V8_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root")
V6_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
PACKAGE_PATH = ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_PACKAGE.json"
TRANSPORT_PATH = ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v9.py"
LAUNCHER_PATH = ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v9.py"
VERIFIER_PATH = ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree.py"
REPORT_PATH = ROOT / "V8_FORENSIC_REPORT.md"
ACCEPTANCE_PATH = ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
ACTION_ID = "ace2:qk-gbfp8-base-v9:execute-once:2254dd91:20260813T2013Z"
V8_ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1724Z"
V7_ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1614Z"
RETIRED_PREPACKAGE_ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1610Z"
EXACT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}
EXACT_TRANSPORT_ARGV = [
    str(INTERPRETER),
    str(TRANSPORT_PATH),
    "--package",
    str(PACKAGE_PATH),
    "--acceptance",
    str(ACCEPTANCE_PATH),
    "--irreversible-action-id",
    ACTION_ID,
]
EXACT_LAUNCHER_ARGV = [
    str(INTERPRETER),
    str(LAUNCHER_PATH),
    "--package",
    str(PACKAGE_PATH),
    "--acceptance",
    str(ACCEPTANCE_PATH),
    "--irreversible-action-id",
    ACTION_ID,
]
INTERPRETER_BINDING = {
    "path": str(INTERPRETER),
    "sha256": "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad",
    "version": "3.13.5",
}
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
POPULATION_ARITHMETIC = {
    "oracle_unique_ranking_rows": "FROZEN_ACCEPTED_COUNT=346",
    "query_rows": "14*41=574",
    "valid_causal_scores": "14*(41*42/2)=12054",
}
EXPECTED_BINDINGS = {
    "accepted_v8_handoff": (
        "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/f0fa5c269681/round-0007.json",
        "2254dd910cf2aae776d73c190402b42f2db66c252a80a1a0d958618f577798e3",
    ),
    "accepted_r2_handoff": (
        "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/b8v8execpkg02/round-0003.json",
        "df40a23647066bab6210b661f842258223dbbf88bbb69fc05ad3ec4e625e72b2",
    ),
    "accepted_v8_manifest": (
        str(ACCEPTED_V8_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_MANIFEST.json"),
        "52324ad6ebd1b831774b9fae6923015f2b4cb24f5cde40e3f7d83435651efd64",
    ),
    "runtime_package": (
        str(ACCEPTED_V8_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RUNTIME_PACKAGE.json"),
        "ef2ba57e278be4fb24f6fa56faadfe84a91123781fadab5b6110163b3b5518be",
    ),
    "package": (
        str(ACCEPTED_V8_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.json"),
        "b7c758c7141d7ac8037502da4a00fe9e0780d30501e2af483e9b8c571451493a",
    ),
    "result_schema": (
        str(ACCEPTED_V8_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.json"),
        "fc5cf29d7ec7486b106edac592787b5888557c4b1b7453cb0a668b02fca113ec",
    ),
    "controller": (
        str(ACCEPTED_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v8.py"),
        "faab3064733be0d0fc68c565f00dd2c77022b7f4974d39b798eda1f150d55aa4",
    ),
    "evaluator": (
        str(ACCEPTED_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"),
        "8ab74c7397006c9f419059f295613ce4a743a3e0b540176ba7cf6dbb6efd7f63",
    ),
    "c02_parser": (
        str(ACCEPTED_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"),
        "ff9216d58bcd1584361e17d8d1854f3bcb96923c17927b1f309861546807bdb1",
    ),
    "decisive_verifier": (
        str(ACCEPTED_V8_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_static_v8.py"),
        "67da980b637b93af6784dd8ab11ceb727b37ba4f5357e41c758c53015d0ff6eb",
    ),
    "v8_authority": (
        str(V8_ROOT / "live/authority/base/authority.json"),
        "248a308dda3813c3593c26ec528278ae9e56619933dc00083e821e5a2947479e",
    ),
    "v8_consumed_ledger": (
        str(V8_ROOT / "live/authority/base/authority-ledger.json"),
        "b566f7f48b8c613f59ce6fcc215046038a9358fa547313c6a635b90802d5481d",
    ),
    "v8_first_terminal": (
        str(V8_ROOT / "live/authority/base/first-terminal.json"),
        "ba92fd9eedbd39662325ad73ad1a85d34f59a901840cd38e67bcb95531630143",
    ),
    "v7_consumed_ledger": (
        str(V7_ROOT / "live/authority/base/authority-ledger.json"),
        "0b4485abb861a8aeda82a283ecab7814045599e49a7d7e5256eea7a7cc6d0708",
    ),
    "v7_first_terminal": (
        str(V7_ROOT / "live/authority/base/first-terminal.json"),
        "5d6243a9fbd830334829bd96a63a0be7da965a22a55cab93a6500f2a0463c0c8",
    ),
}
EXPECTED_BINDING_ORDER = list(EXPECTED_BINDINGS)
LOCAL_PATHS = {
    "authority_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_AUTHORITY_SCHEMA.json",
    "credential_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_CREDENTIAL_SCHEMA.json",
    "ledger_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_LEDGER_SCHEMA.json",
    "first_terminal_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_FIRST_TERMINAL_SCHEMA.json",
    "fresh_l2_acceptance_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_FRESH_L2_ACCEPTANCE_SCHEMA.json",
    "transport": TRANSPORT_PATH,
    "launcher": LAUNCHER_PATH,
    "static_verifier": VERIFIER_PATH,
    "forensic_report": REPORT_PATH,
}
LOCAL_ORDER = list(LOCAL_PATHS)
SCHEMA_RECORDS = {
    "authority": ("authority_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_AUTHORITY_SCHEMA"),
    "credential": ("credential_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_CREDENTIAL_SCHEMA"),
    "consumed_ledger": ("ledger_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_LEDGER_SCHEMA"),
    "first_terminal": ("first_terminal_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_FIRST_TERMINAL_SCHEMA"),
    "fresh_l2_acceptance": ("fresh_l2_acceptance_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_FRESH_L2_ACCEPTANCE_SCHEMA"),
}
EXPECTED_STATIC_FILES = {PACKAGE_PATH, *LOCAL_PATHS.values()}
PREFLIGHT_OPERATIONS = [
    "ATTEST_TRANSPORT_AND_READ_EXECUTION_PACKAGE",
    "VALIDATE_EXECUTION_PACKAGE_AND_ALL_BINDINGS",
    "VERIFY_V8_V7_SEALED_TERMINALS",
    "VERIFY_INTERPRETER_AND_SCHEMA_STACK",
    "LOAD_AND_VALIDATE_FROZEN_LIFECYCLE_SCHEMAS",
    "VERIFY_LANE_METADATA_AND_LSTAT_SEALED_TENSOR_ONLY",
    "REQUIRE_ZERO_LIVE_NAMESPACE",
    "READ_AND_VALIDATE_FRESH_L2_ACCEPTANCE",
    "LOAD_ALL_FROZEN_RUNTIME_MODULES",
    "BIND_AND_CHECK_ALL_REQUIRED_RUNTIME_CALLABLES",
    "CHECK_ALL_IMPORTED_RUNTIME_DEPENDENCIES",
    "READ_CANONICAL_ACCEPTED_PACKAGE_AND_RESULT_SCHEMA",
    "CONTROLLER_VALIDATE_ACCEPTED_PACKAGE",
    "CONSTRUCT_RESULT_SCHEMA_VALIDATOR",
    "CONSTRUCT_EXACT_TENSOR_BINDINGS",
    "CONSTRUCT_AND_VALIDATE_LIFECYCLE_RECORDS_BEFORE_PUBLICATION",
    "CONSTRUCT_RESULT_CONTEXT_AND_SEAL_READ_ONLY_RUNTIME_BINDINGS",
    "DROP_MUTABLE_PREFLIGHT_CONTEXT_BEFORE_PUBLICATION",
]
PREFLIGHT_MOVEMENT_SPECS = [
    ("ATTEST_TRANSPORT_AND_READ_EXECUTION_PACKAGE", "main", "        transport_attestation, package, package_bytes = _attest_transport()\n"),
    ("VALIDATE_EXECUTION_PACKAGE_AND_ALL_BINDINGS", "_preflight", "    _verify_execution_package_and_bindings(package)\n"),
    ("VERIFY_V8_V7_SEALED_TERMINALS", "_preflight", "    _verify_retained_terminals()\n"),
    ("VERIFY_INTERPRETER_AND_SCHEMA_STACK", "_preflight", "    _verify_interpreter_and_schema_stack()\n"),
    ("LOAD_AND_VALIDATE_FROZEN_LIFECYCLE_SCHEMAS", "_preflight", "    schema_validators = _load_lifecycle_schema_validators()\n"),
    ("VERIFY_LANE_METADATA_AND_LSTAT_SEALED_TENSOR_ONLY", "_preflight", "    _verify_lane_metadata_and_tensor_lstat()\n"),
    ("REQUIRE_ZERO_LIVE_NAMESPACE", "_preflight", "    _require_live_namespace_absent()\n"),
    ("READ_AND_VALIDATE_FRESH_L2_ACCEPTANCE", "_preflight", "    acceptance_sha256 = _validate_fresh_l2_acceptance(package_bytes, schema_validators[\"fresh_l2_acceptance_schema\"])\n"),
    ("LOAD_ALL_FROZEN_RUNTIME_MODULES", "_preflight", "    controller, evaluator, parser_module = _load_frozen_runtime_modules()\n"),
    ("BIND_AND_CHECK_ALL_REQUIRED_RUNTIME_CALLABLES", "_preflight", "    controller_validate_package, evaluate_bundle_once, parse_c02_producer_bytes, parse_c02_accepted_reader = _bind_runtime_callables(controller, evaluator, parser_module)\n"),
    ("CHECK_ALL_IMPORTED_RUNTIME_DEPENDENCIES", "_preflight", "    _verify_runtime_dependencies(controller, evaluator, parser_module)\n"),
    ("READ_CANONICAL_ACCEPTED_PACKAGE_AND_RESULT_SCHEMA", "_preflight", "    accepted_package, result_schema = _read_accepted_package_and_result_schema()\n"),
    ("CONTROLLER_VALIDATE_ACCEPTED_PACKAGE", "_preflight", "    _validate_accepted_package(controller_validate_package, accepted_package)\n"),
    ("CONSTRUCT_RESULT_SCHEMA_VALIDATOR", "_preflight", "    result_validator = _construct_result_validator(result_schema)\n"),
    ("CONSTRUCT_EXACT_TENSOR_BINDINGS", "_preflight", "    tensor_bindings = _construct_tensor_bindings(accepted_package)\n"),
    ("CONSTRUCT_AND_VALIDATE_LIFECYCLE_RECORDS_BEFORE_PUBLICATION", "main", "        prepared = _prepare_records(context)\n"),
    ("CONSTRUCT_RESULT_CONTEXT_AND_SEAL_READ_ONLY_RUNTIME_BINDINGS", "main", "        plan = _freeze_execution_plan(context, prepared)\n"),
    ("DROP_MUTABLE_PREFLIGHT_CONTEXT_BEFORE_PUBLICATION", "main", "        del context, prepared\n"),
]
POST_LEDGER_FORBIDDEN = [
    "module load",
    "dynamic import",
    "package read or validation",
    "schema read or validator construction",
    "controller package validation",
    "accepted controller call or transitive package validation",
    "runtime dependency discovery",
    "result-context construction",
    "runtime binding rebinding mutation or recovery",
    "runtime-state mutation in transitive evaluator/parser call graph",
    "module dictionary or subscript recovery",
    "process creation",
]
LIFECYCLE_EVENTS = [
    "FRESH_L2_ACCEPTANCE_CREATE_ONLY_0400_IN_LATER_ACCEPTANCE_ONLY_MISSION",
    "TRANSPORT_EXACT_ARGV_ENV_CWD_ATTEST",
    "POSIX_SPAWN_LAUNCHER_EXPLICIT_ARGV_AND_EXACT_ENV",
    "LAUNCHER_IMMEDIATE_PARENT_TRANSPORT_ATTESTATION",
    "COMPLETE_PRECONSUMPTION_LOADING_AND_VALIDATION",
    "CONSTRUCT_AND_VALIDATE_AUTHORITY_CREDENTIAL_LEDGER_IN_MEMORY",
    "CONSTRUCT_RESULT_CONTEXT_AND_SEAL_READ_ONLY_RUNTIME_BINDINGS",
    "DROP_MUTABLE_PREFLIGHT_CONTEXT",
    "AUTHORITY_CREATE_ONLY_0400_FILE_AND_DIRECTORY_FSYNC",
    "CREDENTIAL_CREATE_ONLY_0400_FILE_AND_DIRECTORY_FSYNC",
    "CONSUMED_LEDGER_CREATE_ONLY_0400_FILE_AND_DIRECTORY_FSYNC",
    "CREDENTIAL_DURABLE_UNLINK",
    "OPEN_SEALED_TENSOR_EXACTLY_ONCE",
    "CALL_EVALUATOR_EXACTLY_ONCE_G8_G4_G2_G1",
    "VALIDATE_AND_CREATE_RESULT_ONCE_IF_VALID",
    "FIRST_TERMINAL_CREATE_ONLY_0400_FILE_AND_DIRECTORY_FSYNC",
]
NO_REPLAY_ACTIONS = [
    "retry",
    "replay",
    "resume",
    "repair",
    "replacement",
    "authority reuse",
    "credential reuse",
    "result replacement",
    "terminal replacement",
]
PROHIBITED_DURING_PREPARATION = [
    "open read or hash sealed tensor",
    "create acceptance authority credential ledger projected payload result first-terminal build or review artifact",
    "invoke transport launcher controller evaluator or C02 parser",
    "touch checkpoint-176 RTL CSR U280 XRT HBM2 wiki paper or stage-closing",
]


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


def invocation_record(argv: list[str]) -> dict[str, Any]:
    return {
        "argv": argv,
        "command_representation": "ARGV_VECTOR_ONLY",
        "cwd": str(ROOT),
        "environment": EXACT_ENVIRONMENT,
        "shell": False,
    }


def invocation_with_hash(argv: list[str]) -> dict[str, Any]:
    record = invocation_record(argv)
    return record | {"invocation_sha256": sha256_bytes(compact_bytes(record))}


def expected_attestation(local_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "api": "os.posix_spawn",
        "immediate_parent_argv": EXACT_TRANSPORT_ARGV,
        "immediate_parent_environment": EXACT_ENVIRONMENT,
        "immediate_parent_executable_path": str(INTERPRETER),
        "immediate_parent_executable_sha256": INTERPRETER_BINDING["sha256"],
        "immediate_parent_transport_path": str(TRANSPORT_PATH),
        "immediate_parent_transport_sha256": local_hashes["transport"],
        "launcher_argv": EXACT_LAUNCHER_ARGV,
        "launcher_environment": EXACT_ENVIRONMENT,
        "launcher_sha256": local_hashes["launcher"],
        "shell": False,
    }


def expected_attestation_sha256(local_hashes: dict[str, str]) -> str:
    return sha256_bytes(compact_bytes(expected_attestation(local_hashes)))


def reseal(package: dict[str, Any]) -> None:
    package.pop("package_content_sha256", None)
    package["package_content_sha256"] = sha256_bytes(compact_bytes(package))


def check_self_checksum(package: dict[str, Any]) -> None:
    observed = package.get("package_content_sha256")
    require(type(observed) is str and len(observed) == 64, "package self-checksum syntax")
    candidate = copy.deepcopy(package)
    candidate.pop("package_content_sha256")
    require(sha256_bytes(compact_bytes(candidate)) == observed, "package self-checksum")


def observed_hashes() -> tuple[dict[str, str], dict[str, str]]:
    accepted = {binding_id: file_sha256(Path(path)) for binding_id, (path, _) in EXPECTED_BINDINGS.items()}
    local = {artifact_id: file_sha256(path) for artifact_id, path in LOCAL_PATHS.items()}
    return accepted, local


def validate_package(package: dict[str, Any], accepted_hashes: dict[str, str], local_hashes: dict[str, str]) -> None:
    check_self_checksum(package)
    require(set(package) == {
        "action_identity", "artifact_kind", "canonical_bindings", "canonicalization", "claim_boundary",
        "fixed_populations", "future_invocation", "interpreter", "launcher_invocation", "lifecycle",
        "live_namespace", "local_artifact_bindings", "official_identities", "package_content_sha256",
        "population_arithmetic", "preflight", "preflight_closure", "prior_terminal_state",
        "prohibited_during_preparation", "record_formats", "root_id", "schema_stack", "schema_version",
        "selection", "static_file_policy", "transport_contract",
    }, "package exact keys")
    require(package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_package", "package kind")
    require(package["schema_version"] == 1 and package["root_id"] == ROOT.name, "package identity")
    require(package["action_identity"] == {
        "future_action_id": ACTION_ID,
        "prior_terminal_action_id": V8_ACTION_ID,
        "retired_action_ids": [RETIRED_PREPACKAGE_ACTION_ID, V7_ACTION_ID, V8_ACTION_ID],
        "reuse_permitted": False,
    }, "action identity")
    require(package["canonicalization"] == {
        "duplicate_keys_permitted": False,
        "json": "UTF-8_ASCII_SUBSET_SORTED_KEYS_COMPACT_NEWLINE",
        "sha256": "RAW_FILE_BYTES",
    }, "canonicalization")
    require(package["claim_boundary"] == {
        "acceptance_materialized": False,
        "authority_materialized": False,
        "build_artifact_created": False,
        "controller_or_evaluator_invoked": False,
        "credential_materialized": False,
        "execution_authorized": False,
        "first_terminal_materialized": False,
        "ledger_materialized": False,
        "projected_payload_materialized": False,
        "result_materialized": False,
        "review_artifact_created": False,
        "sealed_tensor_access": "LSTAT_ONLY",
        "transport_or_launcher_invoked": False,
    }, "claim boundary")
    bindings = package["canonical_bindings"]
    require([item.get("id") for item in bindings] == EXPECTED_BINDING_ORDER, "canonical binding order")
    require(len({item["id"] for item in bindings}) == len(bindings), "canonical binding id uniqueness")
    require(len({item["path"] for item in bindings}) == len(bindings), "canonical binding path uniqueness")
    for item in bindings:
        require(set(item) == {"id", "path", "sha256"}, f"canonical binding keys: {item.get('id')}")
        expected_path, expected_hash = EXPECTED_BINDINGS[item["id"]]
        require(item["path"] == expected_path and item["sha256"] == expected_hash, f"canonical binding declaration: {item['id']}")
        require(accepted_hashes[item["id"]] == expected_hash, f"canonical binding bytes: {item['id']}")
    local = package["local_artifact_bindings"]
    require([item.get("id") for item in local] == LOCAL_ORDER, "local binding order")
    require(len({item["id"] for item in local}) == len(local), "local binding id uniqueness")
    require(len({item["path"] for item in local}) == len(local), "local binding path uniqueness")
    for item in local:
        require(set(item) == {"id", "path", "sha256"}, f"local binding keys: {item.get('id')}")
        require(item["path"] == str(LOCAL_PATHS[item["id"]]), f"local binding path: {item['id']}")
        require(item["sha256"] == local_hashes[item["id"]], f"local binding bytes: {item['id']}")
    require(package["interpreter"] == INTERPRETER_BINDING, "interpreter binding")
    require(package["schema_stack"] == SCHEMA_STACK, "schema stack")
    require(package["official_identities"] == OFFICIAL_IDENTITIES, "official identities")
    require(package["fixed_populations"] == FIXED_POPULATIONS, "fixed populations")
    require(package["population_arithmetic"] == POPULATION_ARITHMETIC, "population arithmetic")
    require(package["future_invocation"] == invocation_with_hash(EXACT_TRANSPORT_ARGV), "future invocation")
    require(package["launcher_invocation"] == invocation_with_hash(EXACT_LAUNCHER_ARGV), "launcher invocation")
    require(package["prior_terminal_state"] == {
        "action_id": V8_ACTION_ID,
        "invocation_count_performed": 0,
        "no_replay_retry_resume_repair_replacement": True,
        "payload_open_count": 0,
        "result_file_sha256": None,
        "status": "CONSUMED_ORPHAN",
    }, "prior terminal declaration")
    require(package["preflight_closure"] == {
        "exact_runtime_binding_type": "RuntimeBindings(NamedTuple; opaque callables over preflight-built exact dictionaries with transitive read-only proof)",
        "operations_in_order": PREFLIGHT_OPERATIONS,
        "post_ledger_forbidden": POST_LEDGER_FORBIDDEN,
        "runtime_rebinding_permitted": False,
    }, "preflight closure")
    contract = package["transport_contract"]
    require(contract == {
        "attestation_before": [
            "acceptance_open", "module_load", "package_schema_validation", "authority_construct",
            "credential_construct", "ledger_construct", "credential_consumption", "payload_open", "evaluator_call",
        ],
        "attestation_sha256": expected_attestation_sha256(local_hashes),
        "direct_exec_api": "os.posix_spawn",
        "environment_inheritance_permitted": False,
        "exact_environment_keys": ["LANG", "LC_ALL", "PYTHONHASHSEED", "TZ"],
        "immediate_parent_required": "FROZEN_TRANSPORT_HELPER",
        "launcher_process_created_by": "posix_spawn_with_explicit_argv_and_environment",
        "process_lineage_boundary": "TRANSPORT_HELPER_TO_IRREVERSIBLE_LAUNCHER",
        "prohibited": ["/bin/bash", "/bin/sh", "-c", "shell parsing", "string command", "system()", "popen()", "shell intermediate"],
        "shell": False,
    }, "transport contract")
    require(package["preflight"] == {
        "checks_in_order": [
            "transport and launcher exact argv environment interpreter parent and attestation",
            "canonical execution package self-checksum plus all canonical and local bindings",
            "V8 and V7 sealed evidence exact hashes canonical self-hashes counters and absence",
            "frozen interpreter schema stack and lifecycle schema validators",
            "lane metadata hash and sealed tensor lstat regular mode-0400 byte-count only",
            "new live root absent and future Fresh-L2 acceptance exact package binding",
            "isolated deterministic loading of accepted controller evaluator and C02 modules",
            "required callables and imported runtime dependency identities",
            "accepted package and result schema canonical reads",
            "controller package validation and result validator construction",
            "exact tensor bindings constructed before publication",
            "authority credential and ledger constructed validated and canonicalized before publication",
            "result context and exact dictionaries sealed behind opaque RuntimeBindings callables with interprocedural read-only proof",
            "mutable preflight context and prepared-record containers dropped before publication",
        ],
        "creates_authority_credential_or_ledger": False,
        "failure_terminal": {
            "action_retired": True,
            "authority_created": False,
            "credential_consumed": False,
            "invocation_count": 0,
            "ledger_created": False,
            "payload_open_count": 0,
            "status": "PREFLIGHT_FAILED_TERMINAL",
        },
        "payload_access": "LSTAT_ONLY",
        "read_only_until_publication": True,
    }, "preflight")
    require(package["live_namespace"] == {
        "directory_mode_octal": "0700",
        "file_mode_octal": "0400",
        "initial_live_root_required_absent": str(ROOT / "live"),
        "paths": {
            "authority": str(ROOT / "live/authority/base/authority.json"),
            "credential": str(ROOT / "live/authority/base/credential.json"),
            "first_terminal": str(ROOT / "live/authority/base/first-terminal.json"),
            "ledger": str(ROOT / "live/authority/base/authority-ledger.json"),
            "result": str(ROOT / "live/result/base/result.json"),
        },
        "publication": "O_CREAT|O_EXCL then file fsync then parent-directory fsync",
    }, "live namespace")
    formats = package["record_formats"]
    require(set(formats) == set(SCHEMA_RECORDS), "record format ids")
    local_by_id = {item["id"]: item for item in local}
    for record_id, (artifact_id, schema_id) in SCHEMA_RECORDS.items():
        require(formats[record_id] == {
            "canonical": True,
            "create_only": True,
            "file_mode_octal": "0400",
            "schema_id": schema_id,
            "schema_path": str(LOCAL_PATHS[artifact_id]),
            "schema_sha256": local_by_id[artifact_id]["sha256"],
        }, f"record format: {record_id}")
    require(package["lifecycle"] == {
        "ambiguity_policy": "EVERY_POST_CONSUMPTION_AMBIGUITY_IS_CONSUMED_ORPHAN_AND_TERMINAL",
        "bound_import_bytecode_writes_permitted": False,
        "credential_consumption_record": "consumed-ledger.credential_consumption",
        "invocation_cardinality_record": "consumed-ledger.invocation_cardinality plus first-terminal.invocation_count_performed",
        "ordered_events": LIFECYCLE_EVENTS,
        "payload_open_count_maximum": 1,
        "post_ledger_exception_terminal": "CONSUMED_ORPHAN",
        "post_ledger_operations": ["credential durable unlink", "payload open", "evaluator call", "result validation/publication", "terminal publication"],
        "prohibited_after_every_terminal": NO_REPLAY_ACTIONS,
        "result_and_first_terminal_create_only": True,
        "terminal_outcomes": ["SUCCEEDED_TERMINAL", "FAILED_TERMINAL", "CONSUMED_ORPHAN", "PREFLIGHT_FAILED_TERMINAL"],
        "terminal_schema_conditional_invariants": True,
        "transport_failure_reason": "TRANSPORT_ATTESTATION_FAILED",
    }, "lifecycle")
    require(package["selection"] == {
        "candidate_order": ["G8", "G4", "G2", "G1"],
        "every_hard_gate_required": True,
        "none_pass": {"reason_code": "HARD_THRESHOLD_FAILED", "selected_candidate": None, "status": "FAILED_TERMINAL"},
        "policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER",
    }, "selection")
    require(package["static_file_policy"] == {
        "allowed_relative_files": sorted(str(path.relative_to(ROOT)) for path in EXPECTED_STATIC_FILES),
        "forbidden_directories": ["build", "live", "review"],
        "generated_python_artifacts_permitted": False,
        "static_files_only": True,
    }, "static file policy")
    require(package["prohibited_during_preparation"] == PROHIBITED_DURING_PREPARATION, "preparation prohibitions")


def validate_terminal_schema(schema: dict[str, Any]) -> None:
    jsonschema.Draft202012Validator.check_schema(schema)
    require(schema["properties"]["retry_replay_resume_repair_replacement_permitted"]["const"] is False, "terminal no replay")
    validator = jsonschema.Draft202012Validator(schema)
    digest = "a" * 64
    base = {
        "action_id": ACTION_ID,
        "action_retired": True,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_first_terminal",
        "authority_sha256": None,
        "consumed_ledger_sha256": None,
        "credential_consumed": False,
        "failure_detail_sha256": digest,
        "first_record_immutable": True,
        "first_terminal_sha256": digest,
        "invocation_count_performed": 0,
        "orphaned_after_consumption": False,
        "payload_open_count": 0,
        "reason_code": "TRANSPORT_ATTESTATION_FAILED",
        "result_file_sha256": None,
        "retry_replay_resume_repair_replacement_permitted": False,
        "status": "PREFLIGHT_FAILED_TERMINAL",
    }
    require(validator.is_valid(base), "transport preflight terminal rejected")
    require(validator.is_valid(base | {"reason_code": "READ_ONLY_PREFLIGHT_FAILED"}), "read-only preflight terminal rejected")
    orphan = base | {
        "authority_sha256": digest,
        "consumed_ledger_sha256": digest,
        "credential_consumed": True,
        "orphaned_after_consumption": True,
        "reason_code": "POST_CONSUMPTION_AMBIGUITY",
        "status": "CONSUMED_ORPHAN",
    }
    require(validator.is_valid(orphan), "zero-invocation consumed orphan rejected")
    require(not validator.is_valid(orphan | {"orphaned_after_consumption": False}), "unmarked orphan accepted")
    require(not validator.is_valid(base | {"invocation_count_performed": 1}), "preflight invocation accepted")
    require(not validator.is_valid(base | {"credential_consumed": True}), "preflight credential consumption accepted")


def validate_schemas() -> None:
    for artifact_id in ("authority_schema", "credential_schema", "ledger_schema", "first_terminal_schema", "fresh_l2_acceptance_schema"):
        schema = canonical_json(LOCAL_PATHS[artifact_id])
        jsonschema.Draft202012Validator.check_schema(schema)
    authority = canonical_json(LOCAL_PATHS["authority_schema"])
    ledger = canonical_json(LOCAL_PATHS["ledger_schema"])
    acceptance = canonical_json(LOCAL_PATHS["fresh_l2_acceptance_schema"])
    require(authority["properties"]["action_id"]["const"] == ACTION_ID, "authority schema action")
    require("transport_attestation" in authority["required"], "authority transport binding")
    require(ledger["properties"]["replay_permitted"]["const"] is False, "ledger replay")
    require(ledger["properties"]["retry_replay_resume_repair_replacement_permitted"]["const"] is False, "ledger no replay")
    require(acceptance["properties"]["static_acceptance_grants_execution_authority"]["const"] is False, "acceptance authority boundary")
    validate_terminal_schema(canonical_json(LOCAL_PATHS["first_terminal_schema"]))


def _imports(tree: ast.AST) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                imported.add(module)
            imported.update(f"{module}.{alias.name}" if module else alias.name for alias in node.names)
    return imported


def _import_bindings(tree: ast.AST) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".", 1)[0]] = alias.name if alias.asname else alias.name.split(".", 1)[0]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = f"{module}.{alias.name}" if module else alias.name
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            assignments: list[tuple[str, ast.AST]] = []
            if isinstance(node, ast.Assign):
                assignments.extend((target.id, node.value) for target in node.targets if isinstance(target, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                assignments.append((node.target.id, node.value))
            for name, value in assignments:
                resolved = _resolved(value, bindings)
                if resolved is not None and bindings.get(name) != resolved:
                    bindings[name] = resolved
                    changed = True
    return bindings


def _resolved(node: ast.AST, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = _resolved(node.value, bindings)
        return f"{owner}.{node.attr}" if owner is not None else None
    return None


def _qualified(node: ast.AST, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        owner = _qualified(node.value, bindings)
        return f"{owner}.{node.attr}" if owner is not None else None
    return None


def _call_target(node: ast.AST, bindings: dict[str, str]) -> str | None:
    return _qualified(node.func, bindings) if isinstance(node, ast.Call) else None


def _function_map(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _local_call_graph(functions: dict[str, ast.FunctionDef]) -> dict[str, set[str]]:
    return {
        name: {
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions
        }
        for name, function in functions.items()
    }


def _reachable_functions(functions: dict[str, ast.FunctionDef], roots: set[str]) -> set[str]:
    graph = _local_call_graph(functions)
    reachable: set[str] = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in reachable or name not in functions:
            continue
        reachable.add(name)
        pending.extend(graph[name] - reachable)
    return reachable


def _alias_expression(node: ast.AST, aliases: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in aliases
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _alias_expression(node.value, aliases)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        return _alias_expression(node.func.value, aliases)
    return False


READ_ONLY_ALIAS_METHODS = {"__contains__", "get", "items", "keys", "values"}
READ_ONLY_ALIAS_CALLS = {
    "all", "any", "enumerate", "len", "list", "range", "set", "sorted", "str", "tuple", "type", "zip",
    "re.fullmatch", "valid_sha256", "parser_module.parse_c02_producer_bytes", "parser_module.parse_c02_accepted_reader",
}


def _validate_read_only_parameter_closure(
    source: str,
    roots: dict[str, set[str]],
    context: str,
) -> None:
    tree = ast.parse(source)
    functions = _function_map(tree)
    bindings = _import_bindings(tree)
    tainted = {name: set(parameters) for name, parameters in roots.items()}
    changed = True
    while changed:
        changed = False
        for function_name, parameters in list(tainted.items()):
            require(function_name in functions, f"{context} read-only entry: {function_name}")
            function = functions[function_name]
            aliases = set(parameters)
            alias_changed = True
            while alias_changed:
                alias_changed = False
                for node in ast.walk(function):
                    assignments: list[tuple[ast.AST, ast.AST]] = []
                    if isinstance(node, ast.Assign):
                        assignments.extend((target, node.value) for target in node.targets)
                    elif isinstance(node, ast.AnnAssign) and node.value is not None:
                        assignments.append((node.target, node.value))
                    elif isinstance(node, ast.NamedExpr):
                        assignments.append((node.target, node.value))
                    for target, value in assignments:
                        if isinstance(target, ast.Name) and target.id not in aliases and _alias_expression(value, aliases):
                            aliases.add(target.id)
                            alias_changed = True
            for node in ast.walk(function):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        require(
                            not isinstance(target, (ast.Attribute, ast.Subscript)) or not _alias_expression(target.value, aliases),
                            f"{context} runtime-state write: {function_name}",
                        )
                elif isinstance(node, ast.Delete):
                    require(
                        not any(isinstance(target, (ast.Attribute, ast.Subscript)) and _alias_expression(target.value, aliases) for target in node.targets),
                        f"{context} runtime-state delete: {function_name}",
                    )
                if not isinstance(node, ast.Call):
                    continue
                target = _call_target(node, bindings) or ""
                if isinstance(node.func, ast.Attribute) and _alias_expression(node.func.value, aliases):
                    require(node.func.attr not in MUTATING_METHODS, f"{context} runtime-state mutation: {function_name}.{node.func.attr}")
                    require(node.func.attr in READ_ONLY_ALIAS_METHODS, f"{context} unknown runtime-state method: {function_name}.{node.func.attr}")
                    continue
                alias_arguments = [index for index, argument in enumerate(node.args) if _alias_expression(argument, aliases)]
                alias_keywords = [keyword for keyword in node.keywords if keyword.arg is not None and _alias_expression(keyword.value, aliases)]
                if not alias_arguments and not alias_keywords:
                    continue
                leaf = target.rsplit(".", 1)[-1]
                if leaf in functions:
                    callee = functions[leaf]
                    parameter_order = [argument.arg for argument in (*callee.args.posonlyargs, *callee.args.args)]
                    propagated = tainted.setdefault(leaf, set())
                    before = len(propagated)
                    for index in alias_arguments:
                        if index < len(parameter_order):
                            propagated.add(parameter_order[index])
                    propagated.update(keyword.arg for keyword in alias_keywords)
                    changed = changed or len(propagated) != before
                    continue
                require(target in READ_ONLY_ALIAS_CALLS or leaf in READ_ONLY_ALIAS_CALLS, f"{context} runtime-state escape: {function_name}->{target}")


def _validate_consumed_call_graph(functions: dict[str, ast.FunctionDef], freeze_node: ast.FunctionDef) -> None:
    reachable = _reachable_functions(functions, {"_run_consumed"})
    forbidden = {
        "_attest_transport", "_bind_runtime_callables", "_construct_result_validator", "_construct_tensor_bindings",
        "_freeze_execution_plan", "_load_frozen_module", "_load_frozen_runtime_modules", "_load_lifecycle_schema_validators",
        "_preflight", "_prepare_records", "_read_accepted_package_and_result_schema", "_require_live_namespace_absent",
        "_validate_accepted_package", "_verify_execution_package_and_bindings", "_verify_interpreter_and_schema_stack",
        "_verify_lane_metadata_and_tensor_lstat", "_verify_retained_terminals", "_verify_runtime_dependencies",
    }
    require(not (reachable & forbidden), f"transitive preflight call after ledger: {sorted(reachable & forbidden)}")
    nested = {node.name: node for node in freeze_node.body if isinstance(node, ast.FunctionDef)}
    validate_calls = {
        node.func.id
        for node in ast.walk(nested["validate_result_once"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    require(validate_calls == {"_validate_result_record"}, "consumed result validator call chain")
    require("_validate_result_record" in reachable or "_validate_result_record" in validate_calls, "consumed result validator reachability")


PROCESS_NAMES = {
    "Popen", "call", "check_call", "check_output", "create_subprocess_exec", "create_subprocess_shell",
    "execl", "execle", "execlp", "execv", "execve", "execvp", "fork", "forkpty", "popen",
    "posix_spawn", "posix_spawnp", "run", "spawn", "spawnl", "spawnv", "startfile", "system",
}
DYNAMIC_NAMES = {"__import__", "compile", "delattr", "eval", "exec", "getattr", "globals", "locals", "setattr", "vars"}
DYNAMIC_ATTRIBUTES = {
    "__base__", "__bases__", "__builtins__", "__class__", "__closure__", "__dict__", "__getattr__",
    "__getattribute__", "__globals__", "__mro__", "__subclasses__", "find_loader", "find_module", "find_spec",
    "import_module", "load_module", "mro",
}
MUTATING_METHODS = {"__delitem__", "__setitem__", "append", "clear", "extend", "insert", "pop", "popitem", "remove", "reverse", "setdefault", "sort", "update"}


def _subscript_key(node: ast.Subscript) -> str | None:
    return node.slice.value if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) else None


def _validate_closed_calls(tree: ast.Module, context: str, allowed_process: set[str]) -> None:
    bindings = _import_bindings(tree)
    require(not any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in DYNAMIC_NAMES for node in ast.walk(tree)), f"{context} dynamic call construction")
    require(not any(isinstance(node, ast.Attribute) and node.attr in DYNAMIC_ATTRIBUTES for node in ast.walk(tree)), f"{context} dynamic attribute acquisition")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        owner = _qualified(node.value, bindings)
        key = _subscript_key(node)
        require(
            not (
                owner == "sys.modules"
                or (owner is not None and owner.endswith(".__dict__"))
                or key in PROCESS_NAMES
                or key in DYNAMIC_NAMES
                or key in DYNAMIC_ATTRIBUTES
            ),
            f"{context} dynamic subscript recovery",
        )
    require(not any(name == "subprocess" or name.startswith("subprocess.") for name in _imports(tree)), f"{context} subprocess import")
    allowed_nodes = {
        id(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_target(node, bindings) in allowed_process
    }
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute)) or not isinstance(node.ctx, ast.Load):
            continue
        target = _qualified(node, bindings)
        leaf = target.rsplit(".", 1)[-1] if target else ""
        syntactic_leaf = node.id if isinstance(node, ast.Name) else node.attr
        if leaf in PROCESS_NAMES or syntactic_leaf in PROCESS_NAMES or (target and target.startswith("subprocess.")):
            require(id(node) in allowed_nodes, f"{context} process reference outside direct allowlist")
    loader_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "exec_module"]
    controlled_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _call_target(node, bindings) in {"importlib.util.spec_from_file_location", "importlib.util.module_from_spec"}]
    if context == "launcher":
        functions = _function_map(tree)
        loader = functions.get("_load_frozen_module")
        require(loader is not None, "launcher frozen loader")
        loader_nodes = {id(node) for node in ast.walk(loader)}
        require(len(controlled_calls) == 2 and all(id(node) in loader_nodes for node in controlled_calls), "launcher controlled loader calls confined")
        require(len(loader_calls) == 1 and id(loader_calls[0]) in loader_nodes, "launcher exec_module confined")
    else:
        require(not controlled_calls and not loader_calls, f"{context} module loader prohibited")


def validate_transport_source(source: str | None = None) -> None:
    if source is None:
        source = file_bytes(TRANSPORT_PATH).decode("utf-8", "strict")
    tree = ast.parse(source, filename=str(TRANSPORT_PATH))
    require(_imports(tree) == {
        "__future__", "__future__.annotations", "hashlib", "json", "os", "pathlib", "pathlib.Path", "stat", "sys", "typing", "typing.Any",
    }, "transport exact import surface")
    _validate_closed_calls(tree, "transport", {"os.posix_spawn"})
    functions = _function_map(tree)
    for name in ("_transport_preflight", "_durable_create", "_retire_transport_failure", "main"):
        require(name in functions, f"transport function: {name}")
    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    require(isinstance(assignments.get("SHELL"), ast.Constant) and assignments["SHELL"].value is False, "transport SHELL false")
    require(
        isinstance(assignments.get("DIRECT_EXEC_API"), ast.Constant)
        and assignments["DIRECT_EXEC_API"].value == "os.posix_spawn",
        "transport direct exec API",
    )
    for frozen_name in ("INTERPRETER", "EXACT_LAUNCHER_ARGV", "EXACT_ENVIRONMENT"):
        require(
            sum(
                isinstance(node, ast.Name)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and node.id == frozen_name
                for node in ast.walk(tree)
            ) == 1,
            f"transport immutable binding: {frozen_name}",
        )
    require(
        not any(isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)) for node in ast.walk(tree)),
        "transport module attribute rebinding",
    )
    bindings = _import_bindings(tree)
    spawn_calls = [node for node in ast.walk(functions["main"]) if _call_target(node, bindings) == "os.posix_spawn"]
    require(len(spawn_calls) == 1, "one transport posix_spawn")
    spawn = spawn_calls[0]
    require(
        len(spawn.args) == 3
        and ast.unparse(spawn.args[0]) == "str(INTERPRETER)"
        and ast.unparse(spawn.args[1]) == "EXACT_LAUNCHER_ARGV"
        and ast.unparse(spawn.args[2]) == "EXACT_ENVIRONMENT"
        and not spawn.keywords,
        "transport exact posix_spawn arguments",
    )
    main_source = ast.get_source_segment(source, functions["main"]) or ""
    require(main_source.index("_transport_preflight()") < main_source.index("os.posix_spawn"), "transport preflight before spawn")
    require(main_source.count("os.posix_spawn") == 1 and main_source.count("os.waitpid") == 1, "transport immediate parent path")
    require('require(dict(os.environ) == EXACT_ENVIRONMENT' in source, "transport exact environment")
    require('require([sys.executable, *sys.argv] == EXACT_TRANSPORT_ARGV' in source, "transport exact argv")
    require('FORBIDDEN_EXECUTABLES = ("/bin/bash", "/bin/sh")' in source and 'FORBIDDEN_ARGV_TOKENS = ("-c",)' in source, "transport shell tokens")
    require("os.O_EXCL" in source and "0o400" in source, "transport create-only terminal")
    retire = ast.get_source_segment(source, functions["_retire_transport_failure"]) or ""
    require("_durable_create(FIRST_TERMINAL" in source and "_terminal_record(error)" in retire, "transport failure terminal")
    require(source.count('if __name__ == "__main__":') == 1, "transport main guard")


def _function_source(source: str, functions: dict[str, ast.FunctionDef], name: str) -> str:
    return ast.get_source_segment(source, functions[name]) or ""


def validate_launcher_source(source: str | None = None) -> None:
    if source is None:
        source = file_bytes(LAUNCHER_PATH).decode("utf-8", "strict")
    tree = ast.parse(source, filename=str(LAUNCHER_PATH))
    require(_imports(tree) == {
        "__future__", "__future__.annotations", "hashlib", "importlib.metadata", "importlib.util", "json", "jsonschema",
        "os", "pathlib", "pathlib.Path", "stat", "sys", "types", "types.FunctionType", "types.ModuleType",
        "typing", "typing.Any", "typing.Callable", "typing.NamedTuple",
    }, "launcher exact import surface")
    _validate_closed_calls(tree, "launcher", set())
    functions = _function_map(tree)
    required_functions = {
        "_attest_transport", "_bind_runtime_callables", "_construct_result_validator", "_construct_tensor_bindings", "_valid_sha256",
        "_validate_result_record", "decode_canonical_json",
        "_durable_create", "_durable_unlink", "_freeze_execution_plan", "_load_frozen_module",
        "_load_frozen_runtime_modules", "_load_lifecycle_schema_validators", "_preflight", "_prepare_records",
        "_publish_terminal", "_read_accepted_package_and_result_schema", "_read_tensor_once", "_require_live_namespace_absent",
        "_run_consumed", "_validate_accepted_package", "_validate_fresh_l2_acceptance",
        "_verify_execution_package_and_bindings", "_verify_interpreter_and_schema_stack",
        "_verify_lane_metadata_and_tensor_lstat", "_verify_retained_terminals", "_verify_runtime_dependencies", "main",
    }
    require(required_functions.issubset(functions), "launcher required functions")
    attest = _function_source(source, functions, "_attest_transport")
    for snippet in (
        "parent_argv == EXACT_TRANSPORT_ARGV", "parent_environment == EXACT_ENVIRONMENT",
        "parent_executable_path == str(INTERPRETER)", "parent_executable_sha256 == INTERPRETER_SHA256",
        'observed_attestation_sha256 == package["transport_contract"]["attestation_sha256"]',
        'proc_root / "cmdline"', 'proc_root / "environ"', 'os.readlink(proc_root / "exe")',
    ):
        require(snippet in attest, f"launcher transport attestation: {snippet}")
    main_source = _function_source(source, functions, "main")
    ordered = [
        "_attest_transport()", "_preflight(", "_prepare_records(context)", "_freeze_execution_plan(context, prepared)",
        "del context, prepared", '_durable_create(LIVE_PATHS["authority"]',
        '_durable_create(LIVE_PATHS["credential"]', '_durable_create(LIVE_PATHS["ledger"]', "ledger_published = True",
        '_durable_unlink(LIVE_PATHS["credential"]', "_run_consumed(plan)",
    ]
    positions = [main_source.find(item) for item in ordered]
    require(all(position >= 0 for position in positions) and positions == sorted(positions), "launcher lifecycle order")
    require('return _retire_preflight_failure(error, "TRANSPORT_ATTESTATION_FAILED")' in main_source, "transport failure retirement")
    require('return _retire_preflight_failure(error, "READ_ONLY_PREFLIGHT_FAILED")' in main_source, "preflight failure retirement")
    preflight = _function_source(source, functions, "_preflight")
    consumed = _function_source(source, functions, "_run_consumed")
    prepare = _function_source(source, functions, "_prepare_records")
    require([item[0] for item in PREFLIGHT_MOVEMENT_SPECS] == PREFLIGHT_OPERATIONS, "preflight operation ledger alignment")
    owner_sources = {name: _function_source(source, functions, name) for name in {item[1] for item in PREFLIGHT_MOVEMENT_SPECS}}
    for operation_id, owner, statement in PREFLIGHT_MOVEMENT_SPECS:
        require(owner_sources[owner].count(statement.strip()) == 1, f"preflight operation missing or duplicated: {operation_id}")
        require(statement.strip() not in consumed, f"preflight operation after ledger: {operation_id}")
    preflight_statements = [statement.strip() for _, owner, statement in PREFLIGHT_MOVEMENT_SPECS if owner == "_preflight"]
    preflight_positions = [preflight.find(statement) for statement in preflight_statements]
    require(all(position >= 0 for position in preflight_positions) and preflight_positions == sorted(preflight_positions), "preflight operation order")

    internal_requirements = {
        "_verify_execution_package_and_bindings": ["_validate_package(package)", 'read_bytes(ACCEPTED_PATHS[binding_id])', 'read_bytes(LOCAL_PATHS[binding_id])'],
        "_verify_retained_terminals": ["V8 failure detail hash", "V8 terminal counters", "V7 terminal counters"],
        "_verify_interpreter_and_schema_stack": ["read_bytes(INTERPRETER)", "sys.version.split()[0]", "importlib.metadata.version(distribution)"],
        "_load_lifecycle_schema_validators": ['read_canonical_json(LOCAL_PATHS[binding_id], binding_id)', "validator_class.check_schema(schema)", "validator_class(schema)"],
        "_verify_lane_metadata_and_tensor_lstat": ["read_bytes(LANE_METADATA)", "os.lstat(SEALED_TENSOR)", "tensor_stat.st_size == 1305797"],
        "_require_live_namespace_absent": ["not os.path.lexists(LIVE_ROOT)"],
        "_validate_fresh_l2_acceptance": ['read_canonical_json(ACCEPTANCE, "Fresh-L2 acceptance")', 'verify_self_checksum(acceptance, "acceptance_sha256"', 'acceptance_validator.validate(acceptance)'],
        "_load_frozen_runtime_modules": ["sys.dont_write_bytecode = True", 'controller = _load_frozen_module("v9_frozen_accepted_v8_controller"', 'evaluator = _load_frozen_module("v9_frozen_accepted_v8_evaluator"', 'parser_module = _load_frozen_module("v9_frozen_accepted_v8_c02_parser"'],
        "_bind_runtime_callables": ["controller.validate_package", "evaluator.evaluate_bundle_once", "parser_module.parse_c02_producer_bytes", "parser_module.parse_c02_accepted_reader", "type(controller_validate_package) is FunctionType", "type(evaluate_bundle_once) is FunctionType", "type(parse_c02_producer_bytes) is FunctionType", "type(parse_c02_accepted_reader) is FunctionType"],
        "_verify_runtime_dependencies": ["controller.hashlib.__name__", "evaluator.hashlib.__name__", "parser_module.hashlib.__name__"],
        "_read_accepted_package_and_result_schema": ['read_canonical_json(ACCEPTED_PATHS["package"]', 'read_canonical_json(ACCEPTED_PATHS["result_schema"]'],
        "_validate_accepted_package": ["controller_validate_package(accepted_package)"],
        "_construct_result_validator": ["validator_class.check_schema(result_schema)", "validator_class(result_schema)"],
        "_construct_tensor_bindings": ['"shape": list(record["shape"])'],
        "_prepare_records": ['_validate_generated(context, "authority_schema"', '_validate_generated(context, "credential_schema"', '_validate_generated(context, "ledger_schema"', "PreparedRecords("],
        "_validate_result_record": ["bindings: ResultBindings", "result canonical bytes", "result self-checksum"],
        "_freeze_execution_plan": ["result_bindings = ResultBindings(", "result_context = {", "def evaluate_once(", "def validate_result_once(", "def validate_result_schema_once(", "def validate_terminal_once(", "runtime = RuntimeBindings(", "return ExecutionPlan("],
    }
    for function_name, snippets in internal_requirements.items():
        function_source = _function_source(source, functions, function_name)
        for snippet in snippets:
            require(snippet in function_source, f"preflight internal operation missing: {function_name}:{snippet}")
            require(snippet not in consumed, f"preflight internal operation after ledger: {function_name}:{snippet}")
    for snippet in ("_authority(context)", "_credential(authority)", "_ledger(authority, credential", "PreparedRecords("):
        require(snippet in prepare, f"prepublication preparation missing: {snippet}")
        require(snippet not in consumed, f"record preparation after ledger: {snippet}")
    for schema_id in ("authority_schema", "credential_schema", "ledger_schema"):
        require(f'_validate_generated(context, "{schema_id}"' in prepare, f"prepublication validation missing: {schema_id}")
        require(f'"{schema_id}"' not in consumed, f"lifecycle record validation after ledger: {schema_id}")
    for forbidden in (
        "_load_frozen_module", "read_canonical_json", "check_schema", "controller_validate_package", "result_context =",
        "controller_validate_result_record", "RuntimeBindings(", "ExecutionPlan(", "importlib.", "spec_from_file_location", "module_from_spec", "exec_module",
    ):
        require(forbidden not in consumed, f"post-ledger preflight operation: {forbidden}")
    consumed_tree = ast.parse(consumed)
    require(not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(consumed_tree)), "post-ledger import")
    require("plan.runtime.evaluate_bundle_once(" in consumed, "frozen evaluator binding")
    require("plan.runtime.validate_result_record(result, result_bytes)" in consumed, "package-independent frozen result binding")
    require("plan.runtime.validate_result_schema(result)" in consumed, "preconstructed result validator")
    require("plan.runtime.validate_terminal_record(terminal_record)" in consumed, "preconstructed terminal validator")
    require(not any(token in consumed for token in ("accepted_package", "tensor_bindings", "parser_module", "result_context", "context.")), "opaque runtime internals exposed post-ledger")
    require(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_replace" for node in ast.walk(tree)), "runtime NamedTuple replacement")
    attribute_rebindings = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and not (
            isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
            and node.attr == "dont_write_bytecode"
        )
    ]
    require(not attribute_rebindings, "attribute rebinding")
    ledger_markers = [
        node
        for node in ast.walk(functions["main"])
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        and any(isinstance(target, ast.Name) and target.id == "ledger_published" for target in node.targets)
    ]
    require(len(ledger_markers) == 1, "single ledger publication marker")
    ledger_line = ledger_markers[0].lineno
    post_ledger_calls = [node for node in ast.walk(functions["main"]) if isinstance(node, ast.Call) and node.lineno > ledger_line]
    preflight_targets = {
        "_attest_transport", "_bind_runtime_callables", "_construct_result_validator", "_construct_tensor_bindings",
        "_freeze_execution_plan", "_load_frozen_module", "_load_frozen_runtime_modules", "_load_lifecycle_schema_validators",
        "_preflight", "_prepare_records", "_read_accepted_package_and_result_schema", "_require_live_namespace_absent",
        "_validate_accepted_package", "_validate_fresh_l2_acceptance", "_verify_execution_package_and_bindings",
        "_verify_interpreter_and_schema_stack", "_verify_lane_metadata_and_tensor_lstat", "_verify_retained_terminals",
        "_verify_runtime_dependencies",
    }
    bindings = _import_bindings(tree)
    for node in [*post_ledger_calls, *[item for item in ast.walk(functions["_run_consumed"]) if isinstance(item, ast.Call)]]:
        target = _call_target(node, bindings)
        leaf = target.rsplit(".", 1)[-1] if target else ""
        require(leaf not in preflight_targets, f"preflight call after ledger: {leaf}")
        require(not (isinstance(node.func, ast.Attribute) and node.func.attr in MUTATING_METHODS), "post-preflight runtime mutation")
    for node in [*ast.walk(functions["main"]), *ast.walk(functions["_run_consumed"])]:
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) else None
        if isinstance(value, (ast.Name, ast.Attribute)):
            qualified_value = _qualified(value, bindings)
            require(not (qualified_value and qualified_value.startswith("plan.runtime")), "post-preflight runtime alias")
    require(
        sum(
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "runtime"
            for function in functions.values()
            for node in ast.walk(function)
        ) == 1,
        "runtime binding assigned once",
    )
    require(sum(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "context" for node in ast.walk(functions["main"])) == 1, "preflight context assigned once")
    require(sum(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "prepared" for node in ast.walk(functions["main"])) == 1, "prepared records assigned once")
    require(sum(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "plan" for node in ast.walk(functions["main"])) == 1, "execution plan assigned once")
    post_ledger_main = main_source[main_source.index("ledger_published = True"):]
    require("context" not in post_ledger_main and "prepared" not in post_ledger_main and "result_context" not in post_ledger_main, "mutable preflight state reachable after ledger")
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    require(ast.unparse(classes["RuntimeBindings"].bases[0]) == "NamedTuple", "RuntimeBindings immutable type")
    require(ast.unparse(classes["ResultBindings"].bases[0]) == "NamedTuple", "ResultBindings immutable type")
    require(ast.unparse(classes["FrozenParserBindings"].bases[0]) == "NamedTuple", "parser bindings immutable type")
    require(ast.unparse(classes["ExecutionPlan"].bases[0]) == "NamedTuple", "ExecutionPlan immutable type")
    class_fields = {
        name: [node.target.id for node in class_node.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)]
        for name, class_node in classes.items()
    }
    require(class_fields["RuntimeBindings"] == ["evaluate_bundle_once", "validate_result_record", "validate_result_schema", "validate_terminal_record"], "opaque RuntimeBindings fields")
    require(class_fields["ResultBindings"] == ["acceptance_sha256", "authority_sha256", "evaluator_sha256", "invocation_sha256", "ledger_sha256", "model_identity_sha256", "package_sha256", "tensor_bundle_sha256"], "immutable result identity bindings")
    require(class_fields["FrozenParserBindings"] == ["parse_c02_producer_bytes", "parse_c02_accepted_reader"], "complete parser bindings")
    require(class_fields["ExecutionPlan"] == ["authority_sha256", "authority_bytes", "credential_bytes", "ledger_sha256", "ledger_bytes", "runtime"], "deeply frozen execution plan fields")
    freeze_node = functions["_freeze_execution_plan"]
    nested_runtime_functions = [node for node in freeze_node.body if isinstance(node, ast.FunctionDef)]
    require({node.name for node in nested_runtime_functions} == {"evaluate_once", "validate_result_once", "validate_result_schema_once", "validate_terminal_once"}, "opaque runtime closure set")
    for nested in nested_runtime_functions:
        require(
            not any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in {"context", "prepared"} for node in ast.walk(nested)),
            f"mutable preflight container captured by runtime closure: {nested.name}",
        )
    require("controller.validate_result_record" not in source and "controller_validate_result_record" not in source, "accepted controller result validator excluded from runtime")
    freeze_source = _function_source(source, functions, "_freeze_execution_plan")
    require("accepted_package" not in freeze_source and "validate_package" not in freeze_source, "package/controller validation excluded from frozen runtime")
    _validate_consumed_call_graph(functions, freeze_node)
    loader = _function_source(source, functions, "_load_frozen_module")
    require(loader.count("spec_from_file_location") == 1 and loader.count("module_from_spec") == 1 and loader.count("exec_module") == 1, "exact isolated loader")
    require("name not in sys.modules" in loader and "spec.origin == str(path)" in loader, "isolated deterministic namespace")
    load_modules = _function_source(source, functions, "_load_frozen_runtime_modules")
    require("sys.dont_write_bytecode = True" in load_modules and load_modules.index("sys.dont_write_bytecode = True") < load_modules.index("_load_frozen_module"), "bytecode disabled before loads")
    require(load_modules.count("_load_frozen_module(") == 3, "three frozen module loads")
    durable = _function_source(source, functions, "_durable_create")
    unlink = _function_source(source, functions, "_durable_unlink")
    require("os.O_EXCL" in durable and "os.fsync(descriptor)" in durable and "_fsync_directory(path.parent)" in durable, "durable create-only publication")
    require("os.unlink(path)" in unlink and "_fsync_directory(path.parent)" in unlink, "durable credential unlink")
    for record_id, function_source in (
        ("authority", main_source),
        ("credential", main_source),
        ("ledger", main_source),
        ("result", consumed),
    ):
        require(
            function_source.count(f'_durable_create(LIVE_PATHS["{record_id}"]') == 1,
            f"single create-only publication: {record_id}",
        )
    retire = _function_source(source, functions, "_retire_preflight_failure")
    for snippet in ("authority_sha256=None", "consumed_ledger_sha256=None", "credential_consumed=False", "invocation_count=0", "payload_open_count=0"):
        require(snippet in retire, f"preflight terminal zero authority/consumption: {snippet}")
    require("_publish_terminal(record)" in retire and 'LIVE_PATHS["authority"]' not in retire and 'LIVE_PATHS["credential"]' not in retire and 'LIVE_PATHS["ledger"]' not in retire, "preflight terminal only")
    publish = _function_source(source, functions, "_publish_terminal")
    require(publish.count('_durable_create(LIVE_PATHS["first_terminal"]') == 1, "single create-only terminal publication site")
    require(consumed.count("_publish_terminal(terminal_record)") == 2, "one success and one orphan terminal path")
    require(main_source.count("_publish_terminal(terminal_record)") == 1, "one materialization terminal path")
    tensor_refs = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "SEALED_TENSOR"]
    require(len(tensor_refs) == 2, "sealed tensor exact lstat and open references")
    require("os.lstat(SEALED_TENSOR)" in _function_source(source, functions, "_verify_lane_metadata_and_tensor_lstat"), "sealed tensor preflight lstat")
    read_tensor = _function_source(source, functions, "_read_tensor_once")
    require("os.open(SEALED_TENSOR, flags)" in read_tensor and "read_bytes(SEALED_TENSOR)" not in source and "sha256_bytes(read_bytes(SEALED_TENSOR))" not in source, "sealed tensor consumed open only")
    require('"replay_permitted": False' in source and '"retry_replay_resume_repair_replacement_permitted": False' in source, "launcher no replay")
    require('status="CONSUMED_ORPHAN" if post_ledger else "FAILED_TERMINAL"' in main_source, "post-ledger orphan")
    require(source.count('if __name__ == "__main__":') == 1, "launcher main guard")


def validate_accepted_module_sources() -> None:
    expected_imports = {
        "controller": {"__future__", "__future__.annotations", "hashlib", "json", "math", "os", "re", "typing", "typing.Any"},
        "evaluator": {"__future__", "__future__.annotations", "copy", "copy.deepcopy", "hashlib", "json", "math", "re", "struct", "typing", "typing.Any"},
        "c02_parser": {"__future__", "__future__.annotations", "hashlib", "io", "math", "struct", "typing", "typing.Any"},
    }
    required_functions = {
        "controller": {"validate_package", "validate_result_record"},
        "evaluator": {"evaluate_bundle_once"},
        "c02_parser": {"parse_c02_producer_bytes", "parse_c02_accepted_reader"},
    }
    sources: dict[str, str] = {}
    for binding_id in ("controller", "evaluator", "c02_parser"):
        source = file_bytes(Path(EXPECTED_BINDINGS[binding_id][0])).decode("utf-8", "strict")
        sources[binding_id] = source
        tree = ast.parse(source)
        require(_imports(tree) == expected_imports[binding_id], f"accepted {binding_id} import closure")
        _validate_closed_calls(tree, f"accepted {binding_id}", set())
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        require(required_functions[binding_id].issubset(functions), f"accepted {binding_id} required callables")
        require(not any(name == "subprocess" or name.startswith("subprocess.") for name in _imports(tree)), f"accepted {binding_id} process dependency")
        require(not any(isinstance(node, ast.Global) for node in ast.walk(tree)), f"accepted {binding_id} mutable module-global closure")
    controller_functions = _function_map(ast.parse(sources["controller"]))
    controller_result_chain = _reachable_functions(controller_functions, {"validate_result_record"})
    require("validate_package" in controller_result_chain, "accepted controller result-to-package validation chain")
    _validate_read_only_parameter_closure(
        sources["evaluator"],
        {"evaluate_bundle_once": {"bindings", "context"}},
        "accepted evaluator",
    )
    _validate_read_only_parameter_closure(
        sources["c02_parser"],
        {
            "parse_c02_producer_bytes": {"bindings"},
            "parse_c02_accepted_reader": {"bindings"},
        },
        "accepted C02 parser",
    )


def verify_retained_evidence() -> None:
    for binding_id in ("v8_authority", "v8_consumed_ledger", "v8_first_terminal", "v7_consumed_ledger", "v7_first_terminal"):
        path, expected_hash = EXPECTED_BINDINGS[binding_id]
        require(file_sha256(Path(path)) == expected_hash, f"retained evidence hash: {binding_id}")
    authority = canonical_json(Path(EXPECTED_BINDINGS["v8_authority"][0]))
    ledger = canonical_json(Path(EXPECTED_BINDINGS["v8_consumed_ledger"][0]))
    terminal = canonical_json(Path(EXPECTED_BINDINGS["v8_first_terminal"][0]))
    for value, key, expected in (
        (authority, "authority_sha256", "d745038d194b4be1e4bd7643344e402875da4d4e510d1a0ea64cccbadaeebf48"),
        (ledger, "consumed_ledger_sha256", "ec2f49c233e1e66fa6c49ba1fe39c26d83bdee0eab9055b3dd6528ee7a32db22"),
        (terminal, "first_terminal_sha256", "9f30c0e28e488bc0d90db1ae2fcfafe754e882f03f7e5ef4c27105a4d604ae65"),
    ):
        candidate = dict(value)
        observed = candidate.pop(key)
        require(observed == expected and sha256_bytes(compact_bytes(candidate)) == expected, f"V8 canonical self hash: {key}")
    require(terminal["failure_detail_sha256"] == "7fe1602a1cd89139fc712070f5961dce30d1bc5be2f13153130498de18661373", "V8 failure detail hash")
    require(terminal["status"] == "CONSUMED_ORPHAN" and terminal["reason_code"] == "POST_CONSUMPTION_AMBIGUITY", "V8 terminal status")
    require(terminal["invocation_count_performed"] == 0 and terminal["payload_open_count"] == 0, "V8 terminal counters")
    require(ledger["invocation_cardinality"]["invocation_count_performed_at_publish"] == 0 and ledger["payload_cardinality"]["payload_open_count_performed_at_publish"] == 0, "V8 ledger counters")
    require(not os.path.lexists(V8_ROOT / "live/authority/base/credential.json"), "V8 credential remains absent")
    require(not os.path.lexists(V8_ROOT / "live/result/base/result.json"), "V8 result remains absent")
    v7_terminal = canonical_json(Path(EXPECTED_BINDINGS["v7_first_terminal"][0]))
    require(v7_terminal["action_id"] == V7_ACTION_ID and v7_terminal["status"] == "CONSUMED_ORPHAN", "V7 terminal state")
    require(v7_terminal["invocation_count_performed"] == 0 and v7_terminal["payload_open_count"] == 0, "V7 counters")


def validate_forensic_report() -> None:
    report = file_bytes(REPORT_PATH).decode("utf-8", "strict")
    exact_path = "/home/argustest/ace-2/reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
    require(exact_path in report, "forensic report launcher-bound tensor path")
    require("mode `0400`" in report and "mode `0664`" not in report, "forensic report tensor mode")
    require("after durable ledger publication and after credential unlink took effect" in report, "forensic report narrow phase")
    require("either the credential parent-directory durability tail or pre-evaluator module/package validation" in report, "forensic report unresolved interval")
    require("cannot be localized further" in report, "forensic report no overlocalization")
    require("does not name or guess an exception" in report, "forensic report no guessed exception")
    require("7fe1602a1cd89139fc712070f5961dce30d1bc5be2f13153130498de18661373" in report, "forensic report failure digest")


def check_runtime_bindings() -> None:
    require(file_sha256(INTERPRETER) == INTERPRETER_BINDING["sha256"], "interpreter bytes")
    require(sys.version.split()[0] == INTERPRETER_BINDING["version"], "verification interpreter version")
    for distribution, version in SCHEMA_STACK.items():
        require(importlib.metadata.version(distribution) == version, f"installed schema stack: {distribution}")


def check_static_only_lstat_zero_live() -> os.stat_result:
    observed_files = {path for path in ROOT.rglob("*") if path.is_file()}
    require(observed_files == EXPECTED_STATIC_FILES, "V9 exact static file set")
    for path in ROOT.rglob("*"):
        require(not path.is_symlink(), f"symlink prohibited: {path}")
    for directory in ("live", "review", "build"):
        require(not os.path.lexists(ROOT / directory), f"forbidden V9 directory: {directory}")
    require(not any(path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"} for path in ROOT.rglob("*")), "generated Python artifact")
    tensor = V6_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
    observed = os.lstat(tensor)
    require(stat.S_ISREG(observed.st_mode), "sealed tensor regular")
    require(stat.S_IMODE(observed.st_mode) == 0o400 and observed.st_size == 1305797, "sealed tensor lstat identity")
    return observed


def _shadow_terminal_record(
    *,
    status: str,
    reason_code: str,
    authority_sha256: str | None,
    consumed_ledger_sha256: str | None,
    credential_consumed: bool,
    invocation_count: int,
    payload_open_count: int,
    orphaned: bool,
    failure_detail: str,
) -> dict[str, Any]:
    record = {
        "action_id": ACTION_ID,
        "action_retired": True,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_first_terminal",
        "authority_sha256": authority_sha256,
        "consumed_ledger_sha256": consumed_ledger_sha256,
        "credential_consumed": credential_consumed,
        "failure_detail_sha256": sha256_bytes(failure_detail.encode("utf-8")),
        "first_record_immutable": True,
        "invocation_count_performed": invocation_count,
        "orphaned_after_consumption": orphaned,
        "payload_open_count": payload_open_count,
        "reason_code": reason_code,
        "result_file_sha256": None,
        "retry_replay_resume_repair_replacement_permitted": False,
        "status": status,
    }
    record["first_terminal_sha256"] = sha256_bytes(compact_bytes(record))
    return record


def _shadow_durable_create(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            require(written > 0, f"shadow short write: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(stat.S_IMODE(os.lstat(path).st_mode) == 0o400, f"shadow file mode: {path}")


def _assert_shadow_terminal_sealed(path: Path) -> None:
    before = file_sha256(path)
    try:
        _shadow_durable_create(path, b"replacement prohibited\n")
    except FileExistsError:
        pass
    else:
        raise VerificationError("shadow terminal replacement accepted")
    require(file_sha256(path) == before, "shadow terminal changed after replacement attempt")


def run_isolated_failure_sealing_fixtures() -> int:
    terminal_validator = jsonschema.Draft202012Validator(canonical_json(LOCAL_PATHS["first_terminal_schema"]))
    completed = 0
    preflight_root_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="ace2-v9-preflight-shadow-") as temporary:
        root = Path(temporary)
        preflight_root_path = root
        terminal_path = root / "authority/base/first-terminal.json"
        record = _shadow_terminal_record(
            status="PREFLIGHT_FAILED_TERMINAL",
            reason_code="READ_ONLY_PREFLIGHT_FAILED",
            authority_sha256=None,
            consumed_ledger_sha256=None,
            credential_consumed=False,
            invocation_count=0,
            payload_open_count=0,
            orphaned=False,
            failure_detail="RuntimeError:isolated preflight fault",
        )
        terminal_validator.validate(record)
        _shadow_durable_create(terminal_path, compact_bytes(record))
        observed = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
        require(observed == {Path("authority/base/first-terminal.json")}, "preflight fixture created authority credential or ledger")
        _assert_shadow_terminal_sealed(terminal_path)
        completed += 1
    require(preflight_root_path is not None and not os.path.lexists(preflight_root_path), "preflight fixture cleanup")

    for label, invocation_count, payload_open_count in (
        ("unlink_tail", 0, 0),
        ("pre_payload", 1, 0),
        ("post_payload", 1, 1),
    ):
        post_root_path: Path | None = None
        with tempfile.TemporaryDirectory(prefix=f"ace2-v9-{label}-shadow-") as temporary:
            root = Path(temporary)
            post_root_path = root
            authority_path = root / "authority/base/authority.json"
            credential_path = root / "authority/base/credential.json"
            ledger_path = root / "authority/base/authority-ledger.json"
            terminal_path = root / "authority/base/first-terminal.json"
            _shadow_durable_create(authority_path, b"shadow authority\n")
            _shadow_durable_create(credential_path, b"shadow credential\n")
            _shadow_durable_create(ledger_path, b"shadow consumed ledger\n")
            os.unlink(credential_path)
            record = _shadow_terminal_record(
                status="CONSUMED_ORPHAN",
                reason_code="POST_CONSUMPTION_AMBIGUITY",
                authority_sha256="a" * 64,
                consumed_ledger_sha256="b" * 64,
                credential_consumed=True,
                invocation_count=invocation_count,
                payload_open_count=payload_open_count,
                orphaned=True,
                failure_detail=f"RuntimeError:isolated {label} fault",
            )
            terminal_validator.validate(record)
            _shadow_durable_create(terminal_path, compact_bytes(record))
            observed = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
            require(
                observed == {
                    Path("authority/base/authority.json"),
                    Path("authority/base/authority-ledger.json"),
                    Path("authority/base/first-terminal.json"),
                },
                f"post-ledger fixture artifact set: {label}",
            )
            require(not os.path.lexists(credential_path), f"post-ledger fixture credential remains: {label}")
            _assert_shadow_terminal_sealed(terminal_path)
            completed += 1
        require(post_root_path is not None and not os.path.lexists(post_root_path), f"post-ledger fixture cleanup: {label}")
    require(completed == 4, "isolated fixture count")
    return completed


def mutation_suite(package: dict[str, Any], accepted_hashes: dict[str, str], local_hashes: dict[str, str]) -> tuple[int, int]:
    cases: list[tuple[str, Callable[[dict[str, Any]], None]]] = []

    def add(name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        cases.append((name, mutate))

    for index, binding_id in enumerate(EXPECTED_BINDING_ORDER):
        add(f"binding_path_{binding_id}", lambda value, i=index: value["canonical_bindings"][i].__setitem__("path", "/tmp/substituted"))
        add(f"binding_hash_{binding_id}", lambda value, i=index: value["canonical_bindings"][i].__setitem__("sha256", "0" * 64))
    for index, artifact_id in enumerate(LOCAL_ORDER):
        add(f"local_path_{artifact_id}", lambda value, i=index: value["local_artifact_bindings"][i].__setitem__("path", "/tmp/substituted"))
        add(f"local_hash_{artifact_id}", lambda value, i=index: value["local_artifact_bindings"][i].__setitem__("sha256", "1" * 64))
    add("action_id", lambda value: value["action_identity"].__setitem__("future_action_id", V8_ACTION_ID))
    add("reuse", lambda value: value["action_identity"].__setitem__("reuse_permitted", True))
    add("prior_terminal_status", lambda value: value["prior_terminal_state"].__setitem__("status", "FAILED_TERMINAL"))
    add("prior_terminal_invocation", lambda value: value["prior_terminal_state"].__setitem__("invocation_count_performed", 1))
    add("argv_string", lambda value: value["future_invocation"].__setitem__("argv", " ".join(EXACT_TRANSPORT_ARGV)))
    add("shell_true", lambda value: value["future_invocation"].__setitem__("shell", True))
    add("bash_c", lambda value: value["future_invocation"].__setitem__("argv", ["/bin/bash", "-c", "launcher"]))
    add("environment_inheritance", lambda value: value["future_invocation"]["environment"].__setitem__("PATH", "/bin"))
    add("direct_exec_api", lambda value: value["transport_contract"].__setitem__("direct_exec_api", "subprocess"))
    add("attestation_hash", lambda value: value["transport_contract"].__setitem__("attestation_sha256", "2" * 64))
    add("attestation_late", lambda value: value["transport_contract"]["attestation_before"].remove("module_load"))
    add("preflight_order", lambda value: value["preflight"]["checks_in_order"].reverse())
    add("preflight_consumes", lambda value: value["preflight"]["failure_terminal"].__setitem__("credential_consumed", True))
    add("preflight_authority", lambda value: value["preflight"].__setitem__("creates_authority_credential_or_ledger", True))
    add("closure_order", lambda value: value["preflight_closure"]["operations_in_order"].reverse())
    add("closure_rebinding", lambda value: value["preflight_closure"].__setitem__("runtime_rebinding_permitted", True))
    add("closure_postledger", lambda value: value["preflight_closure"]["post_ledger_forbidden"].pop())
    add("lifecycle_order", lambda value: value["lifecycle"]["ordered_events"].reverse())
    add("postledger_module", lambda value: value["lifecycle"]["post_ledger_operations"].append("module load"))
    add("replay", lambda value: value["lifecycle"]["prohibited_after_every_terminal"].remove("replay"))
    add("payload_count", lambda value: value["lifecycle"].__setitem__("payload_open_count_maximum", 2))
    add("candidate_order", lambda value: value["selection"]["candidate_order"].reverse())
    add("claim_tensor_read", lambda value: value["claim_boundary"].__setitem__("sealed_tensor_access", "READ"))
    add("claim_acceptance", lambda value: value["claim_boundary"].__setitem__("acceptance_materialized", True))
    add("extra_static_file", lambda value: value["static_file_policy"]["allowed_relative_files"].append("review/FRESH_L2_STATIC_ACCEPTANCE.json"))
    add("preparation_scope", lambda value: value["prohibited_during_preparation"].pop())
    for record_id in SCHEMA_RECORDS:
        add(f"record_create_only_{record_id}", lambda value, r=record_id: value["record_formats"][r].__setitem__("create_only", False))
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
            raise VerificationError(f"resealed package mutation accepted: {name}")
    return rejected, len(cases)


def coordinated_local_candidate(
    package: dict[str, Any],
    accepted_hashes: dict[str, str],
    local_hashes: dict[str, str],
    artifact_id: str,
    mutated_bytes: bytes,
) -> None:
    candidate = copy.deepcopy(package)
    mutated_hashes = dict(local_hashes)
    mutated_hashes[artifact_id] = sha256_bytes(mutated_bytes)
    for binding in candidate["local_artifact_bindings"]:
        if binding["id"] == artifact_id:
            binding["sha256"] = mutated_hashes[artifact_id]
    for record_id, (schema_artifact_id, _) in SCHEMA_RECORDS.items():
        if schema_artifact_id == artifact_id:
            candidate["record_formats"][record_id]["schema_sha256"] = mutated_hashes[artifact_id]
    if artifact_id in {"transport", "launcher"}:
        candidate["transport_contract"]["attestation_sha256"] = expected_attestation_sha256(mutated_hashes)
    reseal(candidate)
    validate_package(candidate, accepted_hashes, mutated_hashes)


def semantic_mutation_suite(package: dict[str, Any], accepted_hashes: dict[str, str], local_hashes: dict[str, str]) -> tuple[int, int]:
    rejected = 0
    total = 0
    transport = file_bytes(TRANSPORT_PATH).decode("utf-8", "strict")
    transport_mutations = [
        ("transport_shell_true", transport.replace("SHELL = False", "SHELL = True", 1)),
        ("transport_string_argv", transport.replace("EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)", '" ".join(EXACT_LAUNCHER_ARGV), EXACT_ENVIRONMENT)', 1)),
        ("transport_bash", transport.replace("os.posix_spawn(str(INTERPRETER), EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)", 'os.posix_spawn("/bin/bash", ["/bin/bash", "-c", "launcher"], EXACT_ENVIRONMENT)', 1)),
        ("transport_environment_inheritance", transport.replace("EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)", "EXACT_LAUNCHER_ARGV, dict(os.environ))", 1)),
        ("transport_system", transport.replace("child_pid = os.posix_spawn(str(INTERPRETER), EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)", 'child_pid = os.system("launcher")', 1)),
        ("transport_alias_system", transport.replace("        child_pid = os.posix_spawn", '        hidden = os.system\n        hidden("launcher")\n        child_pid = os.posix_spawn', 1)),
        ("transport_getattr", transport.replace("        child_pid = os.posix_spawn", '        getattr(os, "system")("launcher")\n        child_pid = os.posix_spawn', 1)),
        ("transport_dynamic_import", transport.replace("        child_pid = os.posix_spawn", '        __import__("os").system("launcher")\n        child_pid = os.posix_spawn', 1)),
        ("transport_module_dict_process", transport.replace("        child_pid = os.posix_spawn", '        os.__dict__["system"]("launcher")\n        child_pid = os.posix_spawn', 1)),
        ("transport_sys_modules_process", transport.replace("        child_pid = os.posix_spawn", '        sys.modules["os"].system("launcher")\n        child_pid = os.posix_spawn', 1)),
        ("transport_argv_rebind", transport.replace("        _transport_preflight()\n", '        _transport_preflight()\n        EXACT_LAUNCHER_ARGV = ["/bin/sh", "-c", "launcher"]\n', 1)),
        ("transport_spawn_monkeypatch", transport.replace("        _transport_preflight()\n", "        _transport_preflight()\n        os.posix_spawn = lambda executable, argv, environment: 0\n", 1)),
        ("transport_attestation_after_spawn", transport.replace("        _transport_preflight()\n", "        pass\n", 1).replace("    waited_pid, wait_status = os.waitpid", "    _transport_preflight()\n    waited_pid, wait_status = os.waitpid", 1)),
    ]
    for name, mutated in transport_mutations:
        total += 1
        require(mutated != transport, f"unchanged transport mutation: {name}")
        coordinated_local_candidate(package, accepted_hashes, local_hashes, "transport", mutated.encode("utf-8"))
        try:
            validate_transport_source(mutated)
        except (VerificationError, SyntaxError):
            rejected += 1
        else:
            raise VerificationError(f"coordinated transport mutation accepted: {name}")

    launcher = file_bytes(LAUNCHER_PATH).decode("utf-8", "strict")
    launcher_mutations = [
        ("launcher_process_alias", launcher.replace("def main() -> int:\n", 'def main() -> int:\n    hidden = os.popen\n    hidden("launcher")\n', 1)),
        ("launcher_getattr_process", launcher.replace("def main() -> int:\n", 'def main() -> int:\n    getattr(os, "popen")("launcher")\n', 1)),
        ("launcher_module_dict_process", launcher.replace("def main() -> int:\n", 'def main() -> int:\n    os.__dict__["system"]("launcher")\n', 1)),
        ("launcher_module_dict_alias_process", launcher.replace("def main() -> int:\n", 'def main() -> int:\n    module_dictionary = os.__dict__\n    module_dictionary["system"]("launcher")\n', 1)),
        ("launcher_sys_modules_process", launcher.replace("def main() -> int:\n", 'def main() -> int:\n    sys.modules["os"].system("launcher")\n', 1)),
        ("launcher_dynamic_import", launcher.replace("        transport_attestation, package, package_bytes = _attest_transport()\n", '        importlib.import_module("subprocess")\n        transport_attestation, package, package_bytes = _attest_transport()\n', 1)),
        ("launcher_loader_outside_helper", launcher.replace("        transport_attestation, package, package_bytes = _attest_transport()\n", '        spec = importlib.util.spec_from_file_location("x", Path("/tmp/x.py"))\n        spec.loader.exec_module(importlib.util.module_from_spec(spec))\n        transport_attestation, package, package_bytes = _attest_transport()\n', 1)),
        ("launcher_attestation_removed", launcher.replace('require_transport(parent_argv == EXACT_TRANSPORT_ARGV, "immediate parent argv mismatch")', 'require_transport(True, "immediate parent argv mismatch")', 1)),
        ("launcher_bytecode_after_load", launcher.replace("    sys.dont_write_bytecode = True\n", "", 1)),
        ("launcher_ledger_after_payload", launcher.replace('        _durable_create(LIVE_PATHS["ledger"], plan.ledger_bytes)\n', '        _run_consumed(plan)\n        _durable_create(LIVE_PATHS["ledger"], plan.ledger_bytes)\n', 1)),
        (
            "launcher_unlink_before_ledger",
            launcher.replace('        _durable_unlink(LIVE_PATHS["credential"])\n', "", 1).replace(
                '        _durable_create(LIVE_PATHS["ledger"], plan.ledger_bytes)\n',
                '        _durable_unlink(LIVE_PATHS["credential"])\n        _durable_create(LIVE_PATHS["ledger"], plan.ledger_bytes)\n',
                1,
            ),
        ),
        ("launcher_unlink_fsync_removed", launcher.replace("def _durable_unlink(path: Path) -> None:\n    os.unlink(path)\n    _fsync_directory(path.parent)\n", "def _durable_unlink(path: Path) -> None:\n    os.unlink(path)\n", 1)),
        ("launcher_result_not_create_only", launcher.replace('        _durable_create(LIVE_PATHS["result"], result_bytes)\n', '        LIVE_PATHS["result"].write_bytes(result_bytes)\n', 1)),
        ("launcher_terminal_twice", launcher.replace("            _publish_terminal(terminal_record)\n", "            _publish_terminal(terminal_record)\n            _publish_terminal(terminal_record)\n", 1)),
        ("launcher_postledger_not_orphan", launcher.replace('status="CONSUMED_ORPHAN" if post_ledger else "FAILED_TERMINAL"', 'status="FAILED_TERMINAL"', 1)),
        ("launcher_runtime_replace", launcher.replace('    try:\n        counter["invocation_count"] = 1\n', '    try:\n        plan = plan._replace(runtime=plan.runtime)\n        counter["invocation_count"] = 1\n', 1)),
        ("launcher_runtime_attribute_rebind", launcher.replace('    try:\n        counter["invocation_count"] = 1\n', '    try:\n        plan.runtime.evaluate_bundle_once = plan.runtime.evaluate_bundle_once\n        counter["invocation_count"] = 1\n', 1)),
        ("launcher_runtime_alias", launcher.replace('    try:\n        counter["invocation_count"] = 1\n', '    try:\n        runtime_alias = plan.runtime\n        counter["invocation_count"] = 1\n', 1)),
        ("launcher_runtime_closure_mutation", launcher.replace('    try:\n        counter["invocation_count"] = 1\n', '    try:\n        plan.runtime.evaluate_bundle_once.__closure__[0].cell_contents.clear()\n        counter["invocation_count"] = 1\n', 1)),
        ("launcher_result_context_after_ledger", launcher.replace("        ledger_published = True\n", "        ledger_published = True\n        result_context = {}\n", 1)),
        ("launcher_package_validation_after_ledger", launcher.replace("        ledger_published = True\n", "        ledger_published = True\n        _verify_execution_package_and_bindings(package)\n", 1)),
        (
            "launcher_transitive_package_validation_after_ledger",
            launcher.replace(
                "\ndef _run_consumed(plan: ExecutionPlan) -> int:\n",
                "\ndef _post_ledger_validation_leak() -> None:\n    _validate_accepted_package(lambda package: None, {})\n\n\ndef _run_consumed(plan: ExecutionPlan) -> int:\n",
                1,
            ).replace(
                '    counter = {"invocation_count": 0, "payload_open_count": 0}\n',
                '    _post_ledger_validation_leak()\n    counter = {"invocation_count": 0, "payload_open_count": 0}\n',
                1,
            ),
        ),
    ]

    ledger_marker = "        ledger_published = True\n"
    for operation_id, owner, statement in PREFLIGHT_MOVEMENT_SPECS:
        require(statement in launcher, f"movement source statement missing: {operation_id}")
        injected = "        " + statement.lstrip()
        replacement = "        pass\n" if operation_id == "ATTEST_TRANSPORT_AND_READ_EXECUTION_PACKAGE" else ""
        mutated = launcher.replace(statement, replacement, 1).replace(ledger_marker, ledger_marker + injected, 1)
        launcher_mutations.append((f"move_preflight_operation_{operation_id.lower()}", mutated))

    targeted_internal_movements = [
        ("move_interpreter_hash_check", '    require(sha256_bytes(read_bytes(INTERPRETER)) == INTERPRETER_SHA256, "interpreter hash")\n'),
        ("move_acceptance_schema_validation", "    acceptance_validator.validate(acceptance)\n"),
        ("move_evaluator_callable_type_check", '    require(type(evaluate_bundle_once) is FunctionType, "evaluator evaluate_bundle_once callable")\n'),
        ("move_controller_dependency_check", '    require(controller.hashlib.__name__ == "hashlib" and controller.json.__name__ == "json" and controller.math.__name__ == "math" and controller.os.__name__ == "os" and controller.re.__name__ == "re", "controller runtime dependencies")\n'),
        ("move_accepted_package_read", '    accepted_package, _ = read_canonical_json(ACCEPTED_PATHS["package"], "accepted V8 package")\n'),
        ("move_result_schema_check", "    validator_class.check_schema(result_schema)\n"),
    ]
    for name, statement in targeted_internal_movements:
        require(statement in launcher, f"internal movement source statement missing: {name}")
        injected = "        " + statement.lstrip()
        mutated = launcher.replace(statement, "", 1).replace(ledger_marker, ledger_marker + injected, 1)
        launcher_mutations.append((name, mutated))

    for name, mutated in launcher_mutations:
        total += 1
        require(mutated != launcher, f"unchanged launcher mutation: {name}")
        coordinated_local_candidate(package, accepted_hashes, local_hashes, "launcher", mutated.encode("utf-8"))
        try:
            validate_launcher_source(mutated)
        except (VerificationError, SyntaxError, KeyError):
            rejected += 1
        else:
            raise VerificationError(f"coordinated launcher mutation accepted: {name}")

    terminal = canonical_json(LOCAL_PATHS["first_terminal_schema"])
    schema_mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("terminal_conditions_removed", lambda value: value.pop("allOf")),
        ("terminal_transport_reason_removed", lambda value: value["properties"]["reason_code"]["enum"].remove("TRANSPORT_ATTESTATION_FAILED")),
        ("terminal_no_replay_true", lambda value: value["properties"]["retry_replay_resume_repair_replacement_permitted"].__setitem__("const", True)),
    ]
    for name, mutate in schema_mutations:
        total += 1
        candidate = copy.deepcopy(terminal)
        mutate(candidate)
        mutated_bytes = compact_bytes(candidate)
        coordinated_local_candidate(package, accepted_hashes, local_hashes, "first_terminal_schema", mutated_bytes)
        try:
            validate_terminal_schema(candidate)
        except VerificationError:
            rejected += 1
        else:
            raise VerificationError(f"coordinated schema mutation accepted: {name}")
    return rejected, total


def interprocedural_runtime_mutation_suite() -> tuple[int, int]:
    evaluator = file_bytes(Path(EXPECTED_BINDINGS["evaluator"][0])).decode("utf-8", "strict")
    parser = file_bytes(Path(EXPECTED_BINDINGS["c02_parser"][0])).decode("utf-8", "strict")
    cases = [
        (
            "evaluator_direct_bindings_mutation",
            evaluator.replace(
                '    require(callable(open_bundle), "bundle opener")\n',
                '    bindings.clear()\n    require(callable(open_bundle), "bundle opener")\n',
                1,
            ),
            parser,
        ),
        (
            "evaluator_transitive_context_mutation",
            evaluator.replace(
                '    exact_keys(context, RESULT_CONTEXT_KEYS, "result context")\n',
                '    exact_keys(context, RESULT_CONTEXT_KEYS, "result context")\n    context["input_bindings"].clear()\n',
                1,
            ),
            parser,
        ),
        (
            "producer_parser_binding_mutation",
            evaluator,
            parser.replace(
                '        binding = bindings.get(name)\n',
                '        binding = bindings.get(name)\n        binding.clear()\n',
                1,
            ),
        ),
        (
            "accepted_reader_bindings_mutation",
            evaluator,
            parser.replace(
                '    if type(bindings) is not dict:\n        raise C02Error("accepted-reader bindings type")\n',
                '    if type(bindings) is not dict:\n        raise C02Error("accepted-reader bindings type")\n    bindings.clear()\n',
                1,
            ),
        ),
    ]
    rejected = 0
    for name, evaluator_source, parser_source in cases:
        require(evaluator_source != evaluator or parser_source != parser, f"unchanged interprocedural mutation: {name}")
        try:
            _validate_read_only_parameter_closure(
                evaluator_source,
                {"evaluate_bundle_once": {"bindings", "context"}},
                "mutated accepted evaluator",
            )
            _validate_read_only_parameter_closure(
                parser_source,
                {
                    "parse_c02_producer_bytes": {"bindings"},
                    "parse_c02_accepted_reader": {"bindings"},
                },
                "mutated accepted C02 parser",
            )
        except VerificationError:
            rejected += 1
        else:
            raise VerificationError(f"interprocedural runtime mutation accepted: {name}")
    return rejected, len(cases)


def main() -> int:
    package = canonical_json(PACKAGE_PATH)
    accepted_hashes, local_hashes = observed_hashes()
    validate_package(package, accepted_hashes, local_hashes)
    validate_schemas()
    validate_transport_source()
    validate_launcher_source()
    validate_accepted_module_sources()
    verify_retained_evidence()
    validate_forensic_report()
    check_runtime_bindings()
    package_rejected, package_total = mutation_suite(package, accepted_hashes, local_hashes)
    semantic_rejected, semantic_total = semantic_mutation_suite(package, accepted_hashes, local_hashes)
    interprocedural_rejected, interprocedural_total = interprocedural_runtime_mutation_suite()
    fixture_count = run_isolated_failure_sealing_fixtures()
    tensor_stat = check_static_only_lstat_zero_live()
    print(f"INVOCATION_SHA256 {package['future_invocation']['invocation_sha256']}")
    print(f"LAUNCHER_INVOCATION_SHA256 {package['launcher_invocation']['invocation_sha256']}")
    print(f"TRANSPORT_ATTESTATION_SHA256 {package['transport_contract']['attestation_sha256']}")
    for path in sorted(EXPECTED_STATIC_FILES, key=str):
        print(f"FILE_SHA256 {file_sha256(path)} {path}")
    for binding_id in ("v8_authority", "v8_consumed_ledger", "v8_first_terminal", "v7_consumed_ledger", "v7_first_terminal"):
        path = Path(EXPECTED_BINDINGS[binding_id][0])
        print(f"RETAINED_FILE_SHA256 {file_sha256(path)} {path}")
    print(f"MUTATIONS_REJECTED {package_rejected + semantic_rejected + interprocedural_rejected}/{package_total + semantic_total + interprocedural_total}")
    print(f"COORDINATED_SEMANTIC_MUTATIONS_REJECTED {semantic_rejected}/{semantic_total}")
    print(f"INTERPROCEDURAL_RUNTIME_MUTATIONS_REJECTED {interprocedural_rejected}/{interprocedural_total}")
    print("CONSUMED_VALIDATION_CALL_GRAPH package_independent=1 accepted_controller_reachable=0 controller_result_to_package_chain_detected=1")
    print(f"PREFLIGHT_OPERATION_LEDGER_MOVEMENTS_REJECTED {len(PREFLIGHT_MOVEMENT_SPECS)}/{len(PREFLIGHT_MOVEMENT_SPECS)}")
    print(f"ISOLATED_FAILURE_SEALING_FIXTURES {fixture_count}/4 cleanup=verified")
    print(f"SEALED_TENSOR_LSTAT_ONLY regular size={tensor_stat.st_size} mode={stat.S_IMODE(tensor_stat.st_mode):04o}")
    print("STATIC_ONLY_FILES 10")
    print("ZERO_V9_LIVE_ARTIFACTS acceptance=0 authority=0 credential=0 ledger=0 payload=0 result=0 first_terminal=0 build=0 review=0")
    print("ACTION_EXECUTION_COUNTERS transport=0 launcher=0 controller=0 evaluator=0 payload_open=0")
    print("PASS qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_static")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(1)
