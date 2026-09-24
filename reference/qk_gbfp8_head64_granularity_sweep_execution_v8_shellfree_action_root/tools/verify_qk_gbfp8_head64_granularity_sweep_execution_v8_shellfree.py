#!/usr/bin/env python3
"""Decisive static verifier for the Base V8 shell-free transport package.

The verifier parses and hashes static files, validates in-memory resealed
mutations, lstats the sealed tensor, and checks absence of new live state.
It never imports or invokes the transport, launcher, controller, evaluator,
or C02 parser.
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


ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_action_root")
OLD_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v8_action_root")
V8_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root")
V6_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
PACKAGE_PATH = ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_PACKAGE.json"
TRANSPORT_PATH = ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v8.py"
LAUNCHER_PATH = ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v8.py"
VERIFIER_PATH = ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree.py"
ACCEPTANCE_PATH = ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1724Z"
OLD_ACTION_ID = "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1614Z"
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
    "prior_execution_package": (
        str(OLD_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_PACKAGE.json"),
        "dc03586e17db25a199d9564189b34616d314372c5373fc4f2901c57182363ea6",
    ),
    "prior_fresh_l2_acceptance": (
        str(OLD_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"),
        "4daece97b7e0ec8b9b936d369cae943e61aa7e9b0cf08a4baa6e0043c600a1a0",
    ),
    "prior_authority": (
        str(OLD_ROOT / "live/authority/base/authority.json"),
        "6944069d566823e1ae12353f7434eed6eae057f31540f36e55df82f58d67300f",
    ),
    "prior_consumed_ledger": (
        str(OLD_ROOT / "live/authority/base/authority-ledger.json"),
        "0b4485abb861a8aeda82a283ecab7814045599e49a7d7e5256eea7a7cc6d0708",
    ),
    "prior_first_terminal": (
        str(OLD_ROOT / "live/authority/base/first-terminal.json"),
        "5d6243a9fbd830334829bd96a63a0be7da965a22a55cab93a6500f2a0463c0c8",
    ),
}
EXPECTED_BINDING_ORDER = list(EXPECTED_BINDINGS)
LOCAL_PATHS = {
    "authority_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_AUTHORITY_SCHEMA.json",
    "credential_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_CREDENTIAL_SCHEMA.json",
    "ledger_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_LEDGER_SCHEMA.json",
    "first_terminal_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_FIRST_TERMINAL_SCHEMA.json",
    "fresh_l2_acceptance_schema": ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_FRESH_L2_ACCEPTANCE_SCHEMA.json",
    "transport": TRANSPORT_PATH,
    "launcher": LAUNCHER_PATH,
    "static_verifier": VERIFIER_PATH,
}
LOCAL_ORDER = list(LOCAL_PATHS)
SCHEMA_RECORDS = {
    "authority": ("authority_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_AUTHORITY_SCHEMA"),
    "credential": ("credential_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_CREDENTIAL_SCHEMA"),
    "consumed_ledger": ("ledger_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_LEDGER_SCHEMA"),
    "first_terminal": ("first_terminal_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_FIRST_TERMINAL_SCHEMA"),
    "fresh_l2_acceptance": ("fresh_l2_acceptance_schema", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V8_SHELLFREE_FRESH_L2_ACCEPTANCE_SCHEMA"),
}
EXPECTED_STATIC_FILES = {
    PACKAGE_PATH,
    TRANSPORT_PATH,
    LAUNCHER_PATH,
    VERIFIER_PATH,
    *[path for artifact_id, path in LOCAL_PATHS.items() if artifact_id.endswith("_schema")],
}
LIFECYCLE_EVENTS = [
    "FRESH_L2_ACCEPTANCE_CREATE_ONLY_0400_BEFORE_FUTURE_INVOCATION",
    "TRANSPORT_EXACT_ARGV_ENV_CWD_ATTEST",
    "POSIX_SPAWN_LAUNCHER_EXPLICIT_ARGV_AND_EXACT_ENV",
    "LAUNCHER_IMMEDIATE_PARENT_TRANSPORT_ATTESTATION",
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
        "population_arithmetic", "preflight", "prior_terminal_state", "prohibited_during_preparation",
        "record_formats", "root_id", "schema_stack", "schema_version", "selection", "static_file_policy",
        "transport_contract",
    }, "package exact keys")
    require(package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_package", "package kind")
    require(package["schema_version"] == 1 and package["root_id"] == ROOT.name, "package identity")
    require(package["action_identity"] == {
        "future_action_id": ACTION_ID,
        "prior_terminal_action_id": OLD_ACTION_ID,
        "retired_action_ids": [RETIRED_PREPACKAGE_ACTION_ID, OLD_ACTION_ID],
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
        "action_id": OLD_ACTION_ID,
        "invocation_count_performed": 0,
        "no_replay_retry_resume_repair_replacement": True,
        "payload_open_count": 0,
        "result_file_sha256": None,
        "status": "CONSUMED_ORPHAN",
    }, "prior terminal declaration")
    contract = package["transport_contract"]
    require(contract == {
        "attestation_before": [
            "acceptance_open",
            "authority_construct",
            "credential_construct",
            "ledger_construct",
            "credential_consumption",
            "payload_open",
            "evaluator_call",
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
            "transport exact argv list cwd exact four-key environment and no shell tokens",
            "launcher immediate parent executable argv environment helper hash launcher hash and attestation digest",
            "canonical execution package and self-checksum",
            "accepted V8 round-7 and R2 round-3 handoffs exact paths and bytes",
            "all accepted V8 artifact and official identity bindings",
            "old package acceptance authority ledger and first-terminal exact bytes",
            "old terminal exact CONSUMED_ORPHAN zero invocation zero payload result absent no replay",
            "frozen interpreter version hash and schema stack",
            "fixed populations arithmetic candidate order and hard gates",
            "sealed tensor lstat regular mode-0400 byte-count only",
            "new live root absent and Fresh-L2 acceptance exact package binding",
        ],
        "creates_authority_credential_or_ledger": False,
        "failure_terminal": {
            "action_retired": True,
            "authority_created": False,
            "credential_created": False,
            "invocation_count": 0,
            "ledger_created": False,
            "payload_open_count": 0,
            "status": "PREFLIGHT_FAILED_TERMINAL",
        },
        "payload_access": "LSTAT_ONLY",
        "read_only": True,
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
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_first_terminal",
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


def validate_schemas() -> None:
    for artifact_id in ("authority_schema", "credential_schema", "ledger_schema", "first_terminal_schema", "fresh_l2_acceptance_schema"):
        schema = canonical_json(LOCAL_PATHS[artifact_id])
        jsonschema.Draft202012Validator.check_schema(schema)
    authority = canonical_json(LOCAL_PATHS["authority_schema"])
    ledger = canonical_json(LOCAL_PATHS["ledger_schema"])
    require(authority["properties"]["action_id"]["const"] == ACTION_ID, "authority schema action")
    require("transport_attestation" in authority["required"], "authority transport binding")
    require(ledger["properties"]["replay_permitted"]["const"] is False, "ledger replay")
    require(ledger["properties"]["retry_replay_resume_repair_replacement_permitted"]["const"] is False, "ledger no replay")
    require("transport_attestation_sha256" in ledger["required"], "ledger transport binding")
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
                bound_name = alias.asname or alias.name.split(".", 1)[0]
                bindings[bound_name] = alias.name if alias.asname else bound_name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound_name = alias.asname or alias.name
                bindings[bound_name] = f"{module}.{alias.name}" if module else alias.name
    pending: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            pending.extend((target.id, node.value) for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            pending.append((node.target.id, node.value))
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            pending.append((node.target.id, node.value))
    while pending:
        unresolved: list[tuple[str, ast.AST]] = []
        progress = False
        for bound_name, value in pending:
            qualified = _resolved_binding(value, bindings)
            if qualified is None:
                unresolved.append((bound_name, value))
            elif bound_name not in bindings:
                bindings[bound_name] = qualified
                progress = True
        if not progress:
            break
        pending = unresolved
    return bindings


def _resolved_binding(node: ast.AST, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = _resolved_binding(node.value, bindings)
        return f"{owner}.{node.attr}" if owner is not None else None
    return None


def _qualified_name(node: ast.AST, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        owner = _qualified_name(node.value, bindings)
        return f"{owner}.{node.attr}" if owner is not None else None
    if isinstance(node, ast.Call):
        callable_name = _qualified_name(node.func, bindings)
        return f"{callable_name}()" if callable_name is not None else None
    return None


def _call_target(node: ast.AST, bindings: dict[str, str]) -> str | None:
    return _qualified_name(node.func, bindings) if isinstance(node, ast.Call) else None


def _imports_module(imported: set[str], module: str) -> bool:
    return any(name == module or name.startswith(f"{module}.") for name in imported)


def _attribute_call(node: ast.AST, owner: str, attribute: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
        and node.func.attr == attribute
    )


def _name_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


TRANSPORT_IMPORT_SURFACE = {
    "__future__",
    "__future__.annotations",
    "hashlib",
    "json",
    "os",
    "pathlib",
    "pathlib.Path",
    "stat",
    "sys",
    "typing",
    "typing.Any",
}
LAUNCHER_IMPORT_SURFACE = TRANSPORT_IMPORT_SURFACE | {
    "importlib.metadata",
    "importlib.util",
    "jsonschema",
}
DYNAMIC_CALL_CONSTRUCTION_NAMES = {
    "__builtins__",
    "__import__",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "locals",
    "setattr",
    "vars",
}
DYNAMIC_MODULE_ACQUISITION_CALLS = {
    "importlib.import_module",
}
DYNAMIC_MODULE_LOADER_ATTRIBUTES = {
    "__loader__",
    "__spec__",
    "create_module",
    "find_loader",
    "find_module",
    "find_spec",
    "load_module",
}
DYNAMIC_INTROSPECTION_ATTRIBUTES = {
    "__base__",
    "__bases__",
    "__class__",
    "__closure__",
    "__code__",
    "__func__",
    "__globals__",
    "__module__",
    "__mro__",
    "__reduce__",
    "__reduce_ex__",
    "__self__",
    "__subclasses__",
    "ag_frame",
    "cr_frame",
    "f_back",
    "f_builtins",
    "f_globals",
    "gi_frame",
    "mro",
    "tb_frame",
}
CONTROLLED_MODULE_LOADER_CALLS = {
    "importlib.util.module_from_spec",
    "importlib.util.spec_from_file_location",
}
ALLOWED_IMPORTLIB_CALLS = CONTROLLED_MODULE_LOADER_CALLS | {
    "importlib.metadata.version",
}
SAFE_BARE_CALL_NAMES = {
    "SystemExit",
    "any",
    "dict",
    "hasattr",
    "len",
    "list",
    "str",
    "type",
}
PROCESS_CREATION_CALL_NAMES = {
    "Popen",
    "call",
    "check_call",
    "check_output",
    "create_subprocess_exec",
    "create_subprocess_shell",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "fork",
    "forkpty",
    "getoutput",
    "getstatusoutput",
    "popen",
    "posix_spawn",
    "posix_spawnp",
    "run",
    "spawn",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "startfile",
    "system",
}


def _is_process_reference(target: str | None) -> bool:
    return target is not None and (
        target.rsplit(".", 1)[-1] in PROCESS_CREATION_CALL_NAMES
        or target.startswith("subprocess.")
    )


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.List, ast.Tuple)):
        return set().union(*(_target_names(item) for item in node.elts)) if node.elts else set()
    return set()


def _runtime_bound_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(_target_names(node.target))
        elif isinstance(node, ast.NamedExpr):
            names.update(_target_names(node.target))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            names.update(_target_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    names.update(_target_names(item.optional_vars))
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            names.add(node.name)
    return names


def _assigned_call(function: ast.FunctionDef, target_name: str, call: ast.Call) -> bool:
    return any(
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == target_name
        and node.value is call
        for node in ast.walk(function)
    )


def _accepted_path_subscript(node: ast.AST, key: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "ACCEPTED_PATHS"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == key
    )


def _validate_module_loader_calls(tree: ast.AST, bindings: dict[str, str], context: str) -> None:
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    importlib_calls = [(node, _call_target(node, bindings)) for node in calls]
    require(
        all(target in ALLOWED_IMPORTLIB_CALLS for _, target in importlib_calls if target is not None and target.startswith("importlib.")),
        f"{context} importlib call outside closed allowlist",
    )
    controlled = [node for node, target in importlib_calls if target in CONTROLLED_MODULE_LOADER_CALLS]
    exec_module_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "exec_module"
    ]
    if context != "launcher":
        require(not controlled and not exec_module_calls, f"{context} module loader call prohibited")
        return

    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    loader = functions.get("_load_module")
    require(loader is not None, "launcher module loader function")
    accepted_root_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ACCEPTED_V8_ROOT" for target in node.targets)
    ]
    accepted_path_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ACCEPTED_PATHS" for target in node.targets)
    ]
    require(
        len(accepted_root_assignments) == 1
        and accepted_root_assignments[0] in tree.body
        and len(accepted_root_assignments[0].targets) == 1
        and isinstance(accepted_root_assignments[0].targets[0], ast.Name)
        and isinstance(accepted_root_assignments[0].value, ast.Call)
        and isinstance(accepted_root_assignments[0].value.func, ast.Name)
        and accepted_root_assignments[0].value.func.id == "Path"
        and len(accepted_root_assignments[0].value.args) == 1
        and isinstance(accepted_root_assignments[0].value.args[0], ast.Constant)
        and accepted_root_assignments[0].value.args[0].value == str(V8_ROOT)
        and not accepted_root_assignments[0].value.keywords
        and sum(
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "ACCEPTED_V8_ROOT"
            for node in ast.walk(tree)
        ) == 1,
        "launcher frozen accepted V8 root",
    )
    require(
        len(accepted_path_assignments) == 1
        and accepted_path_assignments[0] in tree.body
        and len(accepted_path_assignments[0].targets) == 1
        and isinstance(accepted_path_assignments[0].targets[0], ast.Name)
        and isinstance(accepted_path_assignments[0].value, ast.Dict)
        and sum(
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "ACCEPTED_PATHS"
            for node in ast.walk(tree)
        ) == 1
        and not any(
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and isinstance(node.value, ast.Name)
            and node.value.id == "ACCEPTED_PATHS"
            for node in ast.walk(tree)
        )
        and not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ACCEPTED_PATHS"
            for node in ast.walk(tree)
        ),
        "launcher immutable accepted path table",
    )
    accepted_path_entries = {
        key.value: value
        for key, value in zip(accepted_path_assignments[0].value.keys, accepted_path_assignments[0].value.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    expected_loader_paths = {
        "controller": "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v8.py",
        "evaluator": "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py",
        "c02_parser": "reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py",
    }
    for binding_id, relative_path in expected_loader_paths.items():
        value = accepted_path_entries.get(binding_id)
        require(
            isinstance(value, ast.BinOp)
            and isinstance(value.op, ast.Div)
            and isinstance(value.left, ast.Name)
            and value.left.id == "ACCEPTED_V8_ROOT"
            and isinstance(value.right, ast.Constant)
            and value.right.value == relative_path,
            f"launcher frozen accepted loader path: {binding_id}",
        )
    require(
        [argument.arg for argument in loader.args.args] == ["name", "path"]
        and not loader.args.posonlyargs
        and not loader.args.kwonlyargs
        and loader.args.vararg is None
        and loader.args.kwarg is None,
        "launcher module loader signature",
    )
    spec_calls = [node for node in controlled if _call_target(node, bindings) == "importlib.util.spec_from_file_location"]
    module_calls = [node for node in controlled if _call_target(node, bindings) == "importlib.util.module_from_spec"]
    require(len(spec_calls) == 1 and len(module_calls) == 1 and len(exec_module_calls) == 1, "launcher exact module loader calls")
    spec_call = spec_calls[0]
    module_call = module_calls[0]
    exec_call = exec_module_calls[0]
    require(
        _assigned_call(loader, "spec", spec_call)
        and len(spec_call.args) == 2
        and isinstance(spec_call.args[0], ast.Name)
        and spec_call.args[0].id == "name"
        and isinstance(spec_call.args[1], ast.Name)
        and spec_call.args[1].id == "path"
        and not spec_call.keywords,
        "launcher file-bound module spec",
    )
    require(
        _assigned_call(loader, "module", module_call)
        and len(module_call.args) == 1
        and isinstance(module_call.args[0], ast.Name)
        and module_call.args[0].id == "spec"
        and not module_call.keywords,
        "launcher module creation from bound spec",
    )
    require(
        isinstance(exec_call.func, ast.Attribute)
        and isinstance(exec_call.func.value, ast.Attribute)
        and isinstance(exec_call.func.value.value, ast.Name)
        and exec_call.func.value.value.id == "spec"
        and exec_call.func.value.attr == "loader"
        and len(exec_call.args) == 1
        and isinstance(exec_call.args[0], ast.Name)
        and exec_call.args[0].id == "module"
        and not exec_call.keywords,
        "launcher exact bound exec_module",
    )
    require(spec_call.lineno < module_call.lineno < exec_call.lineno, "launcher module loader order")
    load_calls = sorted(
        [node for node in calls if _call_target(node, bindings) == "_load_module"],
        key=lambda node: node.lineno,
    )
    run_consumed = functions.get("_run_consumed")
    require(run_consumed is not None, "launcher consumed execution function")
    run_consumed_call_ids = {
        id(node)
        for node in ast.walk(run_consumed)
        if isinstance(node, ast.Call) and _call_target(node, bindings) == "_load_module"
    }
    require(
        {id(node) for node in load_calls} == run_consumed_call_ids,
        "launcher bound module loads confined to consumed execution",
    )
    expected_loads = [
        ("accepted_v8_controller", "controller"),
        ("accepted_v8_evaluator", "evaluator"),
        ("accepted_v8_c02_parser", "c02_parser"),
    ]
    require(len(load_calls) == len(expected_loads), "launcher exact bound module load count")
    for call, (module_name, binding_id) in zip(load_calls, expected_loads, strict=True):
        require(
            len(call.args) == 2
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == module_name
            and _accepted_path_subscript(call.args[1], binding_id)
            and not call.keywords,
            f"launcher bound module load: {binding_id}",
        )


def _validate_closed_process_calls(
    tree: ast.AST,
    bindings: dict[str, str],
    context: str,
    allowed_process_calls: set[str],
) -> None:
    dynamic_names = [
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in DYNAMIC_CALL_CONSTRUCTION_NAMES
    ]
    require(not dynamic_names, f"{context} dynamic call construction prohibited")
    require(
        not any(isinstance(node, ast.Attribute) and node.attr in {"__dict__", "__getattr__", "__getattribute__"} for node in ast.walk(tree)),
        f"{context} dynamic attribute construction prohibited",
    )
    require(
        not any(isinstance(node, ast.Attribute) and node.attr in DYNAMIC_INTROSPECTION_ATTRIBUTES for node in ast.walk(tree)),
        f"{context} callable introspection prohibited",
    )
    require(
        not any(
            isinstance(node, ast.Attribute) and node.attr in DYNAMIC_MODULE_LOADER_ATTRIBUTES
            for node in ast.walk(tree)
        ),
        f"{context} dynamic module loader acquisition prohibited",
    )
    require(
        not any(
            isinstance(node, ast.Call)
            and _call_target(node, bindings) in DYNAMIC_MODULE_ACQUISITION_CALLS
            for node in ast.walk(tree)
        ),
        f"{context} dynamic module acquisition prohibited",
    )
    _validate_module_loader_calls(tree, bindings, context)
    allowed_reference_nodes = {
        id(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
        and _call_target(node, bindings) in allowed_process_calls
    }
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute)) or not isinstance(node.ctx, ast.Load):
            continue
        target = _qualified_name(node, bindings)
        direct_process_syntax = (
            isinstance(node, ast.Name) and node.id in PROCESS_CREATION_CALL_NAMES
        ) or (
            isinstance(node, ast.Attribute) and node.attr in PROCESS_CREATION_CALL_NAMES
        )
        if direct_process_syntax or _is_process_reference(target):
            require(
                id(node) in allowed_reference_nodes,
                f"{context} process callable reference outside exact direct allowlist",
            )
    declared_callables = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    runtime_bound_names = _runtime_bound_names(tree)
    parameter_names = {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        require(isinstance(node.func, (ast.Name, ast.Attribute)), f"{context} dynamic call target prohibited")
        target = _call_target(node, bindings)
        if isinstance(node.func, ast.Name):
            require(
                node.func.id not in runtime_bound_names and node.func.id not in parameter_names,
                f"{context} runtime-bound bare callable prohibited",
            )
            require(
                node.func.id in declared_callables
                or node.func.id in bindings
                or node.func.id in SAFE_BARE_CALL_NAMES,
                f"{context} unresolved bare callable prohibited",
            )
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
            require(target is not None, f"{context} unresolved call-valued receiver prohibited")
        direct_process_syntax = isinstance(node.func, ast.Attribute) and node.func.attr in PROCESS_CREATION_CALL_NAMES
        if direct_process_syntax or _is_process_reference(target):
            require(target in allowed_process_calls, f"{context} process call outside closed allowlist")


def _binding_occurrences(tree: ast.AST, name: str) -> list[ast.AST]:
    occurrences: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == name:
            occurrences.append(node)
        elif isinstance(node, ast.arg) and node.arg == name:
            occurrences.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            occurrences.append(node)
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            occurrences.append(node)
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            occurrences.append(node)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == name:
            occurrences.append(node)
        elif isinstance(node, ast.MatchMapping) and node.rest == name:
            occurrences.append(node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.asname or alias.name.split(".", 1)[0]) == name:
                    occurrences.append(alias)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    occurrences.append(alias)
    return occurrences


def _contains_loaded_name(node: ast.AST, names: set[str]) -> bool:
    return any(
        isinstance(candidate, ast.Name)
        and isinstance(candidate.ctx, ast.Load)
        and candidate.id in names
        for candidate in ast.walk(node)
    )


def _require_exact_function(function: ast.FunctionDef, expected_source: str, message: str) -> None:
    expected_tree = ast.parse(expected_source)
    expected_functions = [node for node in expected_tree.body if isinstance(node, ast.FunctionDef)]
    require(
        len(expected_functions) == 1
        and ast.dump(function, include_attributes=False) == ast.dump(expected_functions[0], include_attributes=False),
        message,
    )


def _validate_frozen_transport_spawn_bindings(
    tree: ast.Module,
    functions: dict[str, ast.FunctionDef],
    bindings: dict[str, str],
) -> None:
    expected_expressions = {
        "INTERPRETER": 'Path("/home/argustest/miniconda3/bin/python3.13")',
        "EXACT_LAUNCHER_ARGV": """[
            str(INTERPRETER),
            str(LAUNCHER),
            "--package",
            str(PACKAGE),
            "--acceptance",
            str(ACCEPTANCE),
            "--irreversible-action-id",
            ACTION_ID,
        ]""",
        "EXACT_ENVIRONMENT": '{"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}',
    }
    for name, expression in expected_expressions.items():
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ]
        require(len(assignments) == 1, f"transport single module definition: {name}")
        assignment = assignments[0]
        expected_assignment = ast.parse(f"_EXPECTED = {expression}").body[0]
        require(
            isinstance(expected_assignment, ast.Assign)
            and ast.dump(assignment.value, include_attributes=False)
            == ast.dump(expected_assignment.value, include_attributes=False),
            f"transport exact module definition: {name}",
        )
        require(
            _binding_occurrences(tree, name) == [assignment.targets[0]],
            f"transport immutable unshadowed binding: {name}",
        )

    os_imports = [
        alias
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "os" and alias.asname is None
    ]
    require(
        len(os_imports) == 1 and _binding_occurrences(tree, "os") == os_imports,
        "transport immutable unshadowed os module binding",
    )
    require(
        not any(
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.attr == "posix_spawn"
            for node in ast.walk(tree)
        ),
        "transport os.posix_spawn replacement prohibited",
    )

    frozen_mutable_names = {"EXACT_LAUNCHER_ARGV", "EXACT_ENVIRONMENT"}
    assignment_values: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if not (isinstance(node.value, ast.Call) and _call_target(node.value, bindings) == "os.posix_spawn"):
                assignment_values.append(node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            assignment_values.append(node.value)
        elif isinstance(node, ast.NamedExpr):
            assignment_values.append(node.value)
    require(
        not any(_contains_loaded_name(value, frozen_mutable_names) for value in assignment_values),
        "transport frozen argv/environment alias assignment prohibited",
    )
    require(
        not any(
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and _contains_loaded_name(node.value, frozen_mutable_names)
            for node in ast.walk(tree)
        ),
        "transport frozen argv/environment subscript mutation prohibited",
    )
    mutating_methods = {
        "__delitem__",
        "__iadd__",
        "__imul__",
        "__ior__",
        "__setitem__",
        "append",
        "clear",
        "extend",
        "insert",
        "pop",
        "popitem",
        "remove",
        "reverse",
        "setdefault",
        "sort",
        "update",
    }
    require(
        not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in mutating_methods
            and _contains_loaded_name(node.func.value, frozen_mutable_names)
            for node in ast.walk(tree)
        ),
        "transport frozen argv/environment mutating method prohibited",
    )

    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def nearest_call_target(node: ast.AST) -> str | None:
        current = parents.get(id(node))
        while current is not None and not isinstance(current, ast.Call):
            current = parents.get(id(current))
        return _call_target(current, bindings) if isinstance(current, ast.Call) else None

    expected_use_counts: dict[str, dict[str | None, int]] = {
        "INTERPRETER": {"str": 3},
        "EXACT_LAUNCHER_ARGV": {"any": 1, "invocation_record": 2, "os.posix_spawn": 1, "type": 1},
        "EXACT_ENVIRONMENT": {None: 1, "os.posix_spawn": 1, "require": 1},
    }
    for name, expected_counts in expected_use_counts.items():
        observed_targets = [
            nearest_call_target(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name
        ]
        require(
            len(observed_targets) == sum(expected_counts.values())
            and all(observed_targets.count(target) == count for target, count in expected_counts.items()),
            f"transport exact immutable uses: {name}",
        )

    _require_exact_function(
        functions["invocation_record"],
        '''def invocation_record(argv: list[str]) -> dict[str, Any]:
    return {
        "argv": argv,
        "command_representation": COMMAND_REPRESENTATION,
        "cwd": str(ROOT),
        "environment": EXACT_ENVIRONMENT,
        "shell": SHELL,
    }
''',
        "transport immutable invocation record helper",
    )
    _require_exact_function(
        functions["main"],
        '''def main() -> int:
    try:
        _transport_preflight()
    except Exception as error:
        return _retire_transport_failure(error)
    try:
        child_pid = os.posix_spawn(str(INTERPRETER), EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)
    except Exception as error:
        return _retire_transport_failure(error)
    waited_pid, wait_status = os.waitpid(child_pid, 0)
    if waited_pid != child_pid:
        return 5
    return os.waitstatus_to_exitcode(wait_status)
''',
        "transport exact attested spawn path",
    )


def validate_transport_source(source: str | None = None) -> None:
    if source is None:
        source = file_bytes(TRANSPORT_PATH).decode("utf-8", "strict")
    tree = ast.parse(source, filename=str(TRANSPORT_PATH))
    imported = _imports(tree)
    bindings = _import_bindings(tree)
    require(imported == TRANSPORT_IMPORT_SURFACE, "transport exact import surface")
    _validate_closed_process_calls(tree, bindings, "transport", {"os.posix_spawn"})
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("_transport_preflight", "_durable_create", "_retire_transport_failure", "main"):
        require(name in functions, f"transport function: {name}")
    require("invocation_record" in functions, "transport function: invocation_record")
    _validate_frozen_transport_spawn_bindings(tree, functions, bindings)
    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    require(isinstance(assignments.get("SHELL"), ast.Constant) and assignments["SHELL"].value is False, "transport shell false")
    require(isinstance(assignments.get("DIRECT_EXEC_API"), ast.Constant) and assignments["DIRECT_EXEC_API"].value == "os.posix_spawn", "transport API constant")
    spawn_calls = [node for node in ast.walk(functions["main"]) if _call_target(node, bindings) == "os.posix_spawn"]
    require(len(spawn_calls) == 1, "one transport posix_spawn")
    spawn = spawn_calls[0]
    require(
        len(spawn.args) == 3
        and isinstance(spawn.args[0], ast.Call)
        and isinstance(spawn.args[0].func, ast.Name)
        and spawn.args[0].func.id == "str"
        and len(spawn.args[0].args) == 1
        and isinstance(spawn.args[0].args[0], ast.Name)
        and spawn.args[0].args[0].id == "INTERPRETER"
        and not spawn.args[0].keywords
        and isinstance(spawn.args[1], ast.Name)
        and spawn.args[1].id == "EXACT_LAUNCHER_ARGV"
        and isinstance(spawn.args[2], ast.Name)
        and spawn.args[2].id == "EXACT_ENVIRONMENT"
        and not spawn.keywords,
        "posix_spawn exact interpreter executable, argv, and environment",
    )
    wait_calls = [node for node in ast.walk(functions["main"]) if _call_target(node, bindings) == "os.waitpid"]
    require(len(wait_calls) == 1 and wait_calls[0].lineno > spawn.lineno, "transport remains immediate parent")
    preflight_calls = [node for node in ast.walk(functions["main"]) if _name_call(node, "_transport_preflight")]
    require(len(preflight_calls) == 1 and preflight_calls[0].lineno < spawn.lineno, "transport preflight before spawn")
    require('require(dict(os.environ) == EXACT_ENVIRONMENT' in source, "transport exact environment check")
    require('require([sys.executable, *sys.argv] == EXACT_TRANSPORT_ARGV' in source, "transport exact argv check")
    require('"TRANSPORT_ATTESTATION_FAILED"' in source, "transport mismatch terminal")
    require('"ARGV_VECTOR_ONLY"' in source and 'FORBIDDEN_EXECUTABLES = ("/bin/bash", "/bin/sh")' in source, "transport shell-free declarations")
    require(source.count('if __name__ == "__main__":') == 1, "transport main guard")


def validate_launcher_source(source: str | None = None) -> None:
    if source is None:
        source = file_bytes(LAUNCHER_PATH).decode("utf-8", "strict")
    tree = ast.parse(source, filename=str(LAUNCHER_PATH))
    imported = _imports(tree)
    bindings = _import_bindings(tree)
    require(imported == LAUNCHER_IMPORT_SURFACE, "launcher exact import surface")
    _validate_closed_process_calls(tree, bindings, "launcher", set())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in (
        "_attest_transport",
        "_preflight",
        "_durable_create",
        "_durable_unlink",
        "_publish_terminal",
        "_ledger",
        "_load_module",
        "_read_tensor_once",
        "_run_consumed",
        "main",
    ):
        require(name in functions, f"launcher function: {name}")
    attest_source_requirements = [
        'parent_argv == EXACT_TRANSPORT_ARGV',
        'parent_environment == EXACT_ENVIRONMENT',
        'parent_executable_path == str(INTERPRETER)',
        'parent_executable_sha256 == INTERPRETER_SHA256',
        'observed_attestation_sha256 == package["transport_contract"]["attestation_sha256"]',
        'proc_root / "cmdline"',
        'proc_root / "environ"',
        'os.readlink(proc_root / "exe")',
    ]
    for requirement in attest_source_requirements:
        require(requirement in source, f"launcher transport attestation: {requirement}")
    require(not any(_name_call(node, "_durable_create") for node in ast.walk(functions["_attest_transport"])), "attestation creates live state")
    main_source = ast.get_source_segment(source, functions["main"]) or ""
    positions = [
        main_source.find("_attest_transport()"),
        main_source.find("_preflight("),
        main_source.find("authority = _authority(context)"),
        main_source.find("credential = _credential(authority)"),
        main_source.find("ledger = _ledger("),
        main_source.find('_durable_create(LIVE_PATHS["authority"]'),
    ]
    require(all(position >= 0 for position in positions) and positions == sorted(positions), "transport attestation before consumption")
    require('return _retire_preflight_failure(error, "TRANSPORT_ATTESTATION_FAILED")' in main_source, "transport mismatch zero-invocation terminal")
    durable = functions["_durable_create"]
    require(sum(_attribute_call(node, "os", "fsync") for node in ast.walk(durable)) == 1, "durable create file fsync")
    require(sum(_name_call(node, "_fsync_directory") for node in ast.walk(durable)) == 1, "durable create parent fsync")
    unlink = functions["_durable_unlink"]
    require(sum(_attribute_call(node, "os", "unlink") for node in ast.walk(unlink)) == 1, "durable unlink")
    require(sum(_name_call(node, "_fsync_directory") for node in ast.walk(unlink)) == 1, "durable unlink parent fsync")
    require("os.O_EXCL" in source and "mode: int = 0o400" in source, "create-only mode-0400")

    def live_path_key(node: ast.AST) -> str | None:
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name) or node.value.id != "LIVE_PATHS":
            return None
        return node.slice.value if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) else None

    publication_functions = {
        "authority": "main",
        "credential": "main",
        "ledger": "main",
        "result": "_run_consumed",
        "first_terminal": "_publish_terminal",
    }
    for record_id, function_name in publication_functions.items():
        calls = [
            node
            for node in ast.walk(functions[function_name])
            if _name_call(node, "_durable_create") and node.args and live_path_key(node.args[0]) == record_id
        ]
        require(len(calls) == 1, f"create-only publication site: {record_id}")
    load_module = functions["_load_module"]
    first = load_module.body[0]
    require(
        isinstance(first, ast.Assign)
        and isinstance(first.targets[0], ast.Attribute)
        and isinstance(first.targets[0].value, ast.Name)
        and first.targets[0].value.id == "sys"
        and first.targets[0].attr == "dont_write_bytecode"
        and isinstance(first.value, ast.Constant)
        and first.value.value is True,
        "bytecode disabled before bound imports",
    )
    tensor_opens = [node for node in ast.walk(functions["_read_tensor_once"]) if _attribute_call(node, "os", "open")]
    require(len(tensor_opens) == 1, "one tensor open site")
    evaluator_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "evaluate_bundle_once"
    ]
    require(len(evaluator_calls) == 1, "one evaluator call site")
    require(source.index('_durable_create(LIVE_PATHS["ledger"]') < source.index('_run_consumed(context, authority, ledger)'), "ledger before payload path")
    require(source.index('_durable_create(LIVE_PATHS["ledger"]') < source.index('_durable_unlink(LIVE_PATHS["credential"]'), "ledger before credential unlink")
    require('"replay_permitted": False' in source and '"retry_replay_resume_repair_replacement_permitted": False' in source, "launcher no replay")
    require('status="CONSUMED_ORPHAN" if post_ledger else "FAILED_TERMINAL"' in source, "post-ledger orphan")
    require(source.count('if __name__ == "__main__":') == 1, "launcher main guard")


def validate_prior_terminal() -> None:
    authority = canonical_json(Path(EXPECTED_BINDINGS["prior_authority"][0]))
    ledger = canonical_json(Path(EXPECTED_BINDINGS["prior_consumed_ledger"][0]))
    terminal = canonical_json(Path(EXPECTED_BINDINGS["prior_first_terminal"][0]))
    require(authority["action_id"] == OLD_ACTION_ID and authority["state"] == "READY_UNCONSUMED", "old authority")
    require(ledger["action_id"] == OLD_ACTION_ID and ledger["action_retired"] is True, "old ledger")
    require(ledger["invocation_cardinality"]["invocation_count_performed_at_publish"] == 0, "old ledger invocation")
    require(ledger["payload_cardinality"]["payload_open_count_performed_at_publish"] == 0, "old ledger payload")
    require(ledger["replay_permitted"] is False and ledger["retry_replay_resume_repair_replacement_permitted"] is False, "old ledger no replay")
    require(terminal["status"] == "CONSUMED_ORPHAN" and terminal["action_id"] == OLD_ACTION_ID, "old terminal status")
    require(terminal["invocation_count_performed"] == 0 and terminal["payload_open_count"] == 0, "old terminal cardinality")
    require(terminal["result_file_sha256"] is None and terminal["retry_replay_resume_repair_replacement_permitted"] is False, "old terminal result and no replay")
    require(not os.path.lexists(OLD_ROOT / "live/result/base/result.json"), "old result remains absent")
    require(not os.path.lexists(OLD_ROOT / "live/authority/base/credential.json"), "old credential remains absent")


def check_runtime_bindings() -> None:
    require(file_sha256(INTERPRETER) == INTERPRETER_BINDING["sha256"], "interpreter bytes")
    require(sys.version.split()[0] == INTERPRETER_BINDING["version"], "verification interpreter version")
    for distribution, version in SCHEMA_STACK.items():
        require(importlib.metadata.version(distribution) == version, f"installed schema stack: {distribution}")


def check_static_only_lstat_zero_live() -> os.stat_result:
    observed_files = {path for path in ROOT.rglob("*") if path.is_file()}
    require(observed_files == EXPECTED_STATIC_FILES, "new root exact static file set")
    for path in ROOT.rglob("*"):
        require(not path.is_symlink(), f"symlink prohibited: {path}")
    for directory in ("live", "review", "build"):
        require(not os.path.lexists(ROOT / directory), f"forbidden new directory: {directory}")
    require(not any(path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"} for path in ROOT.rglob("*")), "generated Python artifact")
    tensor = V6_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
    observed = os.lstat(tensor)
    require(stat.S_ISREG(observed.st_mode), "sealed tensor regular")
    require(stat.S_IMODE(observed.st_mode) == 0o400 and observed.st_size == 1305797, "sealed tensor lstat identity")
    return observed


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
    add("action_id", lambda value: value["action_identity"].__setitem__("future_action_id", OLD_ACTION_ID))
    add("reuse", lambda value: value["action_identity"].__setitem__("reuse_permitted", True))
    add("prior_terminal_status", lambda value: value["prior_terminal_state"].__setitem__("status", "FAILED_TERMINAL"))
    add("prior_terminal_invocation", lambda value: value["prior_terminal_state"].__setitem__("invocation_count_performed", 1))
    add("prior_terminal_payload", lambda value: value["prior_terminal_state"].__setitem__("payload_open_count", 1))
    add("prior_terminal_result", lambda value: value["prior_terminal_state"].__setitem__("result_file_sha256", "2" * 64))
    add("prior_terminal_replay", lambda value: value["prior_terminal_state"].__setitem__("no_replay_retry_resume_repair_replacement", False))
    add("argv_list_to_string", lambda value: value["future_invocation"].__setitem__("argv", " ".join(EXACT_TRANSPORT_ARGV)))
    add("shell_true", lambda value: value["future_invocation"].__setitem__("shell", True))
    add("bash_sh_c_insertion", lambda value: value["future_invocation"].__setitem__("argv", ["/bin/bash", "-c", " ".join(EXACT_TRANSPORT_ARGV)]))
    add("transport_env_inheritance", lambda value: value["future_invocation"]["environment"].__setitem__("PATH", "/bin"))
    add("transport_invocation_hash", lambda value: value["future_invocation"].__setitem__("invocation_sha256", "3" * 64))
    add("launcher_argv", lambda value: value["launcher_invocation"]["argv"].reverse())
    add("launcher_shell", lambda value: value["launcher_invocation"].__setitem__("shell", True))
    add("direct_exec_api", lambda value: value["transport_contract"].__setitem__("direct_exec_api", "system"))
    add("attestation_hash", lambda value: value["transport_contract"].__setitem__("attestation_sha256", "4" * 64))
    add("env_inheritance", lambda value: value["transport_contract"].__setitem__("environment_inheritance_permitted", True))
    add("parent_identity", lambda value: value["transport_contract"].__setitem__("immediate_parent_required", "ANY"))
    add("attestation_late", lambda value: value["transport_contract"]["attestation_before"].remove("credential_consumption"))
    add("prohibited_shell_removed", lambda value: value["transport_contract"]["prohibited"].remove("/bin/bash"))
    add("preflight_order", lambda value: value["preflight"]["checks_in_order"].reverse())
    add("preflight_live", lambda value: value["preflight"].__setitem__("creates_authority_credential_or_ledger", True))
    add("preflight_invocation", lambda value: value["preflight"]["failure_terminal"].__setitem__("invocation_count", 1))
    add("preflight_ledger", lambda value: value["preflight"]["failure_terminal"].__setitem__("ledger_created", True))
    add("mode", lambda value: value["live_namespace"].__setitem__("file_mode_octal", "0600"))
    add("create_only", lambda value: value["live_namespace"].__setitem__("publication", "truncate"))
    add("lifecycle_order", lambda value: value["lifecycle"]["ordered_events"].reverse())
    add("payload_count", lambda value: value["lifecycle"].__setitem__("payload_open_count_maximum", 2))
    add("ambiguity", lambda value: value["lifecycle"].__setitem__("ambiguity_policy", "RETRY"))
    add("bytecode", lambda value: value["lifecycle"].__setitem__("bound_import_bytecode_writes_permitted", True))
    add("no_replay", lambda value: value["lifecycle"]["prohibited_after_every_terminal"].remove("replay"))
    add("result_terminal_create_only", lambda value: value["lifecycle"].__setitem__("result_and_first_terminal_create_only", False))
    add("candidate_order", lambda value: value["selection"]["candidate_order"].reverse())
    add("hard_gate", lambda value: value["selection"].__setitem__("every_hard_gate_required", False))
    add("selection_policy", lambda value: value["selection"].__setitem__("policy", "BEST_SCORE"))
    add("none_pass", lambda value: value["selection"]["none_pass"].__setitem__("selected_candidate", "G1"))
    add("claim_transport", lambda value: value["claim_boundary"].__setitem__("transport_or_launcher_invoked", True))
    add("claim_tensor", lambda value: value["claim_boundary"].__setitem__("sealed_tensor_access", "READ"))
    add("static_file_extra", lambda value: value["static_file_policy"]["allowed_relative_files"].append("README"))
    add("preparation_prohibition", lambda value: value["prohibited_during_preparation"].pop())
    for record_id in SCHEMA_RECORDS:
        add(f"record_create_only_{record_id}", lambda value, r=record_id: value["record_formats"][r].__setitem__("create_only", False))
        add(f"record_mode_{record_id}", lambda value, r=record_id: value["record_formats"][r].__setitem__("file_mode_octal", "0600"))
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
    if artifact_id in {"transport", "launcher"}:
        candidate["transport_contract"]["attestation_sha256"] = expected_attestation_sha256(mutated_hashes)
    reseal(candidate)
    validate_package(candidate, accepted_hashes, mutated_hashes)
    return candidate, mutated_hashes


def semantic_mutation_suite(package: dict[str, Any], accepted_hashes: dict[str, str], local_hashes: dict[str, str]) -> tuple[int, int]:
    rejected = 0
    total = 0
    transport = file_bytes(TRANSPORT_PATH).decode("utf-8", "strict")
    transport_mutations = [
        ("transport_argv_vector_to_string", transport.replace("EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)", '" ".join(EXACT_LAUNCHER_ARGV), EXACT_ENVIRONMENT)', 1)),
        ("transport_shell_true", transport.replace("SHELL = False", "SHELL = True", 1)),
        ("transport_executable_only_bash", transport.replace("os.posix_spawn(str(INTERPRETER), EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)", 'os.posix_spawn("/bin/bash", EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)', 1)),
        ("transport_executable_only_sh", transport.replace("os.posix_spawn(str(INTERPRETER), EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)", 'os.posix_spawn("/bin/sh", EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)', 1)),
        ("transport_bash_sh_c", transport.replace("child_pid = os.posix_spawn(str(INTERPRETER), EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)", 'child_pid = os.posix_spawn("/bin/bash", ["/bin/bash", "-c", "launcher"], EXACT_ENVIRONMENT)', 1)),
        ("transport_env_inheritance", transport.replace("EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)", "EXACT_LAUNCHER_ARGV, dict(os.environ))", 1)),
        ("transport_missing_posix_spawn", transport.replace("child_pid = os.posix_spawn(str(INTERPRETER), EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)", "child_pid = 0", 1)),
        ("transport_system_call", transport.replace("child_pid = os.posix_spawn(str(INTERPRETER), EXACT_LAUNCHER_ARGV, EXACT_ENVIRONMENT)", 'child_pid = os.system("launcher")', 1)),
        ("transport_imported_system_alias", transport.replace("import os\n", "import os\nfrom os import system as forbidden_system\n", 1).replace("        child_pid = os.posix_spawn", '        forbidden_system("launcher")\n        child_pid = os.posix_spawn', 1)),
        ("transport_imported_subprocess_shell_alias", transport.replace("import os\n", "import os\nfrom subprocess import run as hidden_run\n", 1).replace("        child_pid = os.posix_spawn", '        hidden_run("/bin/sh -c launcher", shell=True)\n        child_pid = os.posix_spawn', 1)),
        ("transport_assigned_system_alias_chain", transport.replace("        child_pid = os.posix_spawn", '        hidden_system = os.system\n        chained_system = hidden_system\n        chained_system("launcher")\n        child_pid = os.posix_spawn', 1)),
        ("transport_assigned_module_alias_chain", transport.replace("        child_pid = os.posix_spawn", '        process_api = os\n        chained_process_api = process_api\n        chained_process_api.system("launcher")\n        child_pid = os.posix_spawn', 1)),
        ("transport_tuple_unpacked_system_alias", transport.replace("        child_pid = os.posix_spawn", '        hidden_system, = (os.system,)\n        hidden_system("launcher")\n        child_pid = os.posix_spawn', 1)),
        ("transport_default_system_alias", transport.replace("        child_pid = os.posix_spawn", '        def hidden_system(command: str, process_call: Any = os.system) -> Any:\n            return process_call(command)\n        hidden_system("launcher")\n        child_pid = os.posix_spawn', 1)),
        ("transport_loop_system_alias", transport.replace("        child_pid = os.posix_spawn", '        for hidden_system in (os.system,):\n            hidden_system("launcher")\n        child_pid = os.posix_spawn', 1)),
        ("transport_getattr_system", transport.replace("        child_pid = os.posix_spawn", '        getattr(os, "system")("launcher")\n        child_pid = os.posix_spawn', 1)),
        ("transport_dunder_import_system", transport.replace("        child_pid = os.posix_spawn", '        __import__("os").system("launcher")\n        child_pid = os.posix_spawn', 1)),
        ("transport_loader_acquired_subprocess_run", transport.replace("        _transport_preflight()\n", '        json.__loader__.load_module("subprocess").run(["/bin/sh", "-c", "launcher"], shell=False)\n        _transport_preflight()\n', 1)),
        (
            "coordinated_resealed_runtime_spawn_binding_reassignment",
            transport.replace(
                "        _transport_preflight()\n",
                '        _transport_preflight()\n'
                '        INTERPRETER = Path("/bin/sh")\n'
                '        EXACT_LAUNCHER_ARGV = ["/bin/sh", "-c", "exit 0"]\n'
                '        EXACT_ENVIRONMENT = dict(os.environ)\n',
                1,
            ),
        ),
        (
            "transport_runtime_spawn_argv_slice_mutation",
            transport.replace(
                "        _transport_preflight()\n",
                '        _transport_preflight()\n'
                '        EXACT_LAUNCHER_ARGV[:] = ["/bin/sh", "-c", "exit 0"]\n',
                1,
            ),
        ),
        (
            "transport_runtime_spawn_argv_delete",
            transport.replace(
                "        _transport_preflight()\n",
                "        _transport_preflight()\n        del EXACT_LAUNCHER_ARGV[:]\n",
                1,
            ),
        ),
        (
            "transport_runtime_spawn_environment_mutating_methods",
            transport.replace(
                "        _transport_preflight()\n",
                "        _transport_preflight()\n"
                "        EXACT_ENVIRONMENT.clear()\n"
                "        EXACT_ENVIRONMENT.update(dict(os.environ))\n",
                1,
            ),
        ),
        (
            "transport_runtime_os_posix_spawn_monkey_patch",
            transport.replace(
                "        _transport_preflight()\n",
                "        _transport_preflight()\n"
                "        os.posix_spawn = lambda executable, argv, environment: 0\n",
                1,
            ),
        ),
        ("transport_attestation_after_spawn", transport.replace("        _transport_preflight()\n", "        pass\n", 1).replace("    waited_pid, wait_status = os.waitpid(child_pid, 0)\n", "    _transport_preflight()\n    waited_pid, wait_status = os.waitpid(child_pid, 0)\n", 1)),
        ("transport_helper_replacement", "#!/usr/bin/env python3\nimport os\nos.posix_spawn('/bin/sh',['/bin/sh','-c','launcher'],dict(os.environ))\n"),
    ]
    for name, mutated in transport_mutations:
        total += 1
        require(mutated != transport, f"transport mutation unchanged: {name}")
        coordinated_local_candidate(package, accepted_hashes, local_hashes, "transport", mutated.encode("utf-8"))
        try:
            validate_transport_source(mutated)
        except (VerificationError, SyntaxError):
            rejected += 1
        else:
            raise VerificationError(f"coordinated transport mutation accepted: {name}")
    launcher = file_bytes(LAUNCHER_PATH).decode("utf-8", "strict")
    launcher_mutations = [
        ("launcher_imported_popen_alias", launcher.replace("import os\n", "import os\nfrom os import popen as forbidden_popen\n", 1).replace("def main() -> int:\n", 'def main() -> int:\n    forbidden_popen("launcher")\n', 1)),
        ("launcher_assigned_popen_alias_chain", launcher.replace("def main() -> int:\n", 'def main() -> int:\n    hidden_popen = os.popen\n    chained_popen = hidden_popen\n    chained_popen("launcher")\n', 1)),
        ("launcher_tuple_unpacked_popen_alias", launcher.replace("def main() -> int:\n", 'def main() -> int:\n    hidden_popen, = (os.popen,)\n    hidden_popen("launcher")\n', 1)),
        ("launcher_default_popen_alias", launcher.replace("def main() -> int:\n", 'def main() -> int:\n    def hidden_popen(command: str, process_call: Any = os.popen) -> Any:\n        return process_call(command)\n    hidden_popen("launcher")\n', 1)),
        ("launcher_loop_popen_alias", launcher.replace("def main() -> int:\n", 'def main() -> int:\n    for hidden_popen in (os.popen,):\n        hidden_popen("launcher")\n', 1)),
        ("launcher_getattr_popen", launcher.replace("def main() -> int:\n", 'def main() -> int:\n    getattr(os, "popen")("launcher")\n', 1)),
        ("launcher_dynamic_import_subprocess_run", launcher.replace("        transport_attestation, package, package_bytes = _attest_transport()\n", '        importlib.import_module("subprocess").run(["/bin/sh", "-c", "launcher"], shell=False)\n        transport_attestation, package, package_bytes = _attest_transport()\n', 1)),
        (
            "coordinated_resealed_dynamic_subprocess_popen_before_attestation",
            launcher.replace(
                "        transport_attestation, package, package_bytes = _attest_transport()\n",
                '        dynamic_spec = importlib.util.find_spec("subprocess")\n'
                '        dynamic_module = importlib.util.module_from_spec(dynamic_spec)\n'
                '        dynamic_spec.loader.exec_module(dynamic_module)\n'
                '        discovered_classes = ().__class__.__base__.__subclasses__()\n'
                '        hidden_process = [candidate for candidate in discovered_classes if candidate.__module__ == "subprocess" and candidate.__name__ == "Popen"][0]\n'
                '        hidden_process(["/bin/sh", "-c", "launcher"])\n'
                "        transport_attestation, package, package_bytes = _attest_transport()\n",
                1,
            ),
        ),
        (
            "launcher_loader_sequence_outside_bound_helper",
            launcher.replace(
                "        transport_attestation, package, package_bytes = _attest_transport()\n",
                '        dynamic_spec = importlib.util.spec_from_file_location("subprocess", Path("/tmp/subprocess.py"))\n'
                '        dynamic_module = importlib.util.module_from_spec(dynamic_spec)\n'
                '        dynamic_spec.loader.exec_module(dynamic_module)\n'
                "        transport_attestation, package, package_bytes = _attest_transport()\n",
                1,
            ),
        ),
        (
            "launcher_bound_loader_call_before_attestation",
            launcher.replace(
                "        transport_attestation, package, package_bytes = _attest_transport()\n",
                '        _load_module("accepted_v8_controller", ACCEPTED_PATHS["controller"])\n'
                "        transport_attestation, package, package_bytes = _attest_transport()\n",
                1,
            ).replace(
                '        controller = _load_module("accepted_v8_controller", ACCEPTED_PATHS["controller"])\n',
                "        controller = None\n",
                1,
            ),
        ),
        (
            "launcher_accepted_loader_path_rebound",
            launcher.replace(
                '"controller": ACCEPTED_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v8.py"',
                '"controller": ACCEPTED_V8_ROOT / "/usr/local/lib/python3.13/subprocess.py"',
                1,
            ),
        ),
        (
            "launcher_object_subclasses_introspection",
            launcher.replace(
                "        transport_attestation, package, package_bytes = _attest_transport()\n",
                "        discovered_classes = object.__subclasses__()\n"
                "        transport_attestation, package, package_bytes = _attest_transport()\n",
                1,
            ),
        ),
        (
            "launcher_runtime_bound_bare_callable",
            launcher.replace(
                "        transport_attestation, package, package_bytes = _attest_transport()\n",
                "        benign_callable = require\n"
                '        benign_callable(True, "launcher")\n'
                "        transport_attestation, package, package_bytes = _attest_transport()\n",
                1,
            ),
        ),
        ("launcher_parent_argv_mismatch", launcher.replace('require_transport(parent_argv == EXACT_TRANSPORT_ARGV, "immediate parent argv mismatch")', 'require_transport(True, "immediate parent argv mismatch")', 1)),
        ("launcher_parent_env_mismatch", launcher.replace('require_transport(parent_environment == EXACT_ENVIRONMENT, "immediate parent environment mismatch")', 'require_transport(True, "immediate parent environment mismatch")', 1)),
        ("launcher_parent_exe_mismatch", launcher.replace('require_transport(parent_executable_path == str(INTERPRETER), "immediate parent executable mismatch")', 'require_transport(True, "immediate parent executable mismatch")', 1)),
        ("launcher_parent_hash_mismatch", launcher.replace('require_transport(parent_executable_sha256 == INTERPRETER_SHA256, "immediate parent executable bytes")', 'require_transport(True, "immediate parent executable bytes")', 1)),
        ("launcher_attestation_digest_removed", launcher.replace('require_transport(observed_attestation_sha256 == package["transport_contract"]["attestation_sha256"], "transport attestation digest mismatch")', 'require_transport(True, "transport attestation digest mismatch")', 1)),
        ("launcher_attestation_after_consumption", launcher.replace("        transport_attestation, package, package_bytes = _attest_transport()\n", "        transport_attestation = {}\n        package, package_bytes = read_canonical_json(PACKAGE, 'late')\n", 1).replace("    authority = _authority(context)\n", "    authority = _authority(context)\n    _attest_transport()\n", 1)),
        ("launcher_bytecode_false", launcher.replace("    sys.dont_write_bytecode = True\n", "    sys.dont_write_bytecode = False\n", 1)),
        ("launcher_file_fsync_removed", launcher.replace("        os.fsync(descriptor)\n    finally:\n", "    finally:\n", 1)),
        ("launcher_parent_fsync_removed", launcher.replace("    _fsync_directory(path.parent)\n\n\ndef _durable_unlink", "\n\ndef _durable_unlink", 1)),
        ("launcher_unlink_fsync_removed", launcher.replace("def _durable_unlink(path: Path) -> None:\n    os.unlink(path)\n    _fsync_directory(path.parent)\n", "def _durable_unlink(path: Path) -> None:\n    os.unlink(path)\n", 1)),
        ("launcher_credential_create_only_removed", launcher.replace('        _durable_create(LIVE_PATHS["credential"], compact_bytes(credential))\n', '        LIVE_PATHS["credential"].write_bytes(compact_bytes(credential))\n', 1)),
        ("launcher_result_create_only_removed", launcher.replace('        _durable_create(LIVE_PATHS["result"], result_bytes)\n', '        LIVE_PATHS["result"].write_bytes(result_bytes)\n', 1)),
        ("launcher_terminal_create_only_removed", launcher.replace('    _durable_create(LIVE_PATHS["first_terminal"], compact_bytes(record))\n', '    LIVE_PATHS["first_terminal"].write_bytes(compact_bytes(record))\n', 1)),
        ("launcher_ledger_after_payload", launcher.replace('        _durable_create(LIVE_PATHS["ledger"], compact_bytes(ledger))\n', '        _run_consumed(context, authority, ledger)\n        _durable_create(LIVE_PATHS["ledger"], compact_bytes(ledger))\n', 1)),
        ("launcher_no_replay_true", launcher.replace('"replay_permitted": False,\n            "retry_replay_resume_repair_replacement_permitted": False,', '"replay_permitted": True,\n            "retry_replay_resume_repair_replacement_permitted": True,', 1)),
        ("launcher_post_ledger_failed", launcher.replace('status="CONSUMED_ORPHAN" if post_ledger else "FAILED_TERMINAL"', 'status="FAILED_TERMINAL"', 1)),
    ]
    for name, mutated in launcher_mutations:
        total += 1
        require(mutated != launcher, f"launcher mutation unchanged: {name}")
        coordinated_local_candidate(package, accepted_hashes, local_hashes, "launcher", mutated.encode("utf-8"))
        try:
            validate_launcher_source(mutated)
        except (VerificationError, SyntaxError):
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


def main() -> int:
    package = canonical_json(PACKAGE_PATH)
    accepted_hashes, local_hashes = observed_hashes()
    validate_package(package, accepted_hashes, local_hashes)
    validate_schemas()
    validate_transport_source()
    validate_launcher_source()
    validate_prior_terminal()
    check_runtime_bindings()
    tensor_stat = check_static_only_lstat_zero_live()
    package_rejected, package_total = mutation_suite(package, accepted_hashes, local_hashes)
    semantic_rejected, semantic_total = semantic_mutation_suite(package, accepted_hashes, local_hashes)
    print(f"INVOCATION_SHA256 {package['future_invocation']['invocation_sha256']}")
    print(f"LAUNCHER_INVOCATION_SHA256 {package['launcher_invocation']['invocation_sha256']}")
    print(f"TRANSPORT_ATTESTATION_SHA256 {package['transport_contract']['attestation_sha256']}")
    for path in sorted(EXPECTED_STATIC_FILES, key=str):
        print(f"FILE_SHA256 {file_sha256(path)} {path}")
    for binding_id in ("prior_execution_package", "prior_fresh_l2_acceptance", "prior_authority", "prior_consumed_ledger", "prior_first_terminal"):
        path = Path(EXPECTED_BINDINGS[binding_id][0])
        print(f"OLD_TERMINAL_FILE_SHA256 {file_sha256(path)} {path}")
    print(f"MUTATIONS_REJECTED {package_rejected + semantic_rejected}/{package_total + semantic_total}")
    print(f"COORDINATED_SEMANTIC_MUTATIONS_REJECTED {semantic_rejected}/{semantic_total}")
    print(f"SEALED_TENSOR_LSTAT_ONLY regular size={tensor_stat.st_size} mode={stat.S_IMODE(tensor_stat.st_mode):04o}")
    print("STATIC_ONLY_FILES 9")
    print("ZERO_NEW_LIVE_ARTIFACTS acceptance=0 authority=0 credential=0 ledger=0 payload=0 result=0 first_terminal=0 build=0 review=0")
    print("TRANSPORT_LAUNCHER_CONTROLLER_EVALUATOR_INVOCATIONS 0")
    print("PASS qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_static")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(1)
