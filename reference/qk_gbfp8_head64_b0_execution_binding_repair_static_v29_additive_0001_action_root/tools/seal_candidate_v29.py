#!/usr/bin/env python3
"""Prepare and irreversibly mode-seal the Engineer V29 candidate tree."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


ACTION_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load_module(ACTION_ROOT / "tools/v29_contract.py", "v29_seal_contract")

SOURCE_FILES = [
    "reference/QK_GBFP8_HEAD64_B0_V29_FRESH_L2_ACCEPTANCE_SCHEMA.json",
    "reference/QK_GBFP8_HEAD64_B0_V29_MANAGER_ADMISSION_CAPSULE_SCHEMA.json",
    "reference/QK_GBFP8_HEAD64_B0_V29_OPERATOR_AUTHORITY_CAPSULE_SCHEMA.json",
    "tools/b0_source_oracle_protected_boundary_v29.py",
    "tools/create_fresh_l2_acceptance_v29.py",
    "tools/future_execute_once_wrapper_v29.py",
    "tools/seal_candidate_v29.py",
    "tools/v29_contract.py",
    "tools/verify_candidate_v29.py",
    "tools/verify_post_acceptance_v29.py",
]


def canonicalize_json_sources() -> None:
    for relative in SOURCE_FILES:
        if not relative.endswith(".json"):
            continue
        path = ACTION_ROOT / relative
        value = json.loads(path.read_text(encoding="ascii"))
        path.write_bytes(C.compact_bytes(value))


def generated_files() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        path = ACTION_ROOT / relative
        result[relative] = {"sha256": C.sha256_file(path), "size": path.stat().st_size}
    return result


def build_package() -> dict[str, Any]:
    files = generated_files()
    binding = C.execution_binding()
    package: dict[str, Any] = {
        "action_identity": {
            "action_id": C.STATIC_ACTION_ID,
            "additive_successor": True,
            "future_execution_action_id": C.EXECUTION_ACTION_ID,
            "predecessor_accepted_immutable": True,
            "predecessor_action_id": "ace2:qk-gbfp8-base-v28:b0-source-oracle-identity-control:38571d38:additive-0008",
            "rejected_v28_execution_is_lineage": False,
            "retry_replay_resume_repair_reinterpret_permitted": False,
            "root_id": "qk_gbfp8_head64_b0_execution_binding_repair_static_v29_additive_0001_action_root",
            "static_runtime_namespace": None,
        },
        "artifact_kind": "qk_gbfp8_head64_b0_execution_binding_repair_static_v29_package",
        "authority_contract": {
            "capsules_live_strictly_outside_runtime": True,
            "event_source": {
                "id": "ARGUS_PROJECT_EVENTS_JSONL",
                "path": str(C.EVENT_LOG),
                "replacement_symlink_truncation_or_prefix_drift_rejected": True,
            },
            "manager_capsule_path": str(C.MANAGER_CAPSULE),
            "manager_capsule_schema_path": "reference/QK_GBFP8_HEAD64_B0_V29_MANAGER_ADMISSION_CAPSULE_SCHEMA.json",
            "manager_event_contract": {
                "event_type": "life.manager.intent.started",
                "origin_agent_layer": "manager",
                "origin_source": "user",
                "required_decision": "ADMIT_B0_ONCE",
            },
            "operator_capsule_path": str(C.OPERATOR_CAPSULE),
            "operator_capsule_schema_path": "reference/QK_GBFP8_HEAD64_B0_V29_OPERATOR_AUTHORITY_CAPSULE_SCHEMA.json",
            "operator_event_contract": {
                "event_type": "ui.operator",
                "origin_agent_layer": "operator",
                "required_decision": "AUTHORIZE_B0_ONCE",
            },
            "required_bindings": [
                "event source/device/inode",
                "line and byte frontier",
                "event hash and origin identity",
                "mission/static action/execution action/package",
                "interpreter/argv/cwd/replacement environment",
                "runtime/external authority namespaces",
                "freshness and one-time nonce",
            ],
            "same_nonce_required_for_manager_and_operator": True,
            "synthetic_or_self_asserted_digest_is_authority": False,
        },
        "b0_control": {
            "base_selection_permitted": False,
            "causal_row_count": 574,
            "classification_labels": ["SOURCE_ORACLE_MATCH", "SOURCE_ORACLE_MISMATCH"],
            "may_count_as_any_quantization_group": False,
            "permanently_nonselecting": True,
            "quantization_applied": False,
            "safe_output_boundary": "AGGREGATE_ONLY_NO_RAW_OR_RECONSTRUCTABLE_DATA",
            "threshold_change_permitted": False,
        },
        "claim_boundary": {
            "admission_materialized": False,
            "authority_capsules_materialized": False,
            "claim": C.CLAIM_BOUNDARY,
            "execution_authorized": False,
            "official_evaluator_invocations": 0,
            "official_payload_open_count": 0,
            "official_target_process_starts": 0,
            "result_ledger_or_terminal_materialized": False,
            "runtime_namespace_materialized": False,
            "static_acceptance_grants_execution_authority": False,
        },
        "forbidden_live_paths": [
            str(C.RUNTIME_NAMESPACE),
            str(C.FAILURE_SEAL_NAMESPACE),
            str(C.AUTHORITY_NAMESPACE),
            str(C.ACCEPTANCE_PATH),
        ],
        "fresh_l2_review": {
            "acceptance_materialized_by_engineer": False,
            "acceptance_path": C.ACCEPTANCE_RELATIVE,
            "acceptance_schema_path": "reference/QK_GBFP8_HEAD64_B0_V29_FRESH_L2_ACCEPTANCE_SCHEMA.json",
            "atomic_create_exclusive": True,
            "candidate_tree_binding": "EXACT_PRE_ACCEPTANCE_TREE",
            "creator_event_frontier_recomputed_by_post_and_launcher": True,
            "decision_requested": "ACCEPT_STATIC_PACKAGE",
            "engineer_self_review_can_close": False,
            "required_actor": "reviewer",
            "required_agent_layer": "reviewer",
            "required_mission_id": C.MISSION_ID,
            "required_role": "Fresh-L2",
            "required_run_label": "reviewer",
            "status": "PENDING_INDEPENDENT_REVIEW",
        },
        "future_execution_binding": binding,
        "future_execution_target": {
            "entrypoint": "tools/b0_source_oracle_protected_boundary_v29.py",
            "protected_access_before_frozen_envelope_permitted": False,
            "static_acceptance_authorizes_invocation": False,
            "status": "STATIC_PROTECTED_BOUNDARY_ONLY_FAIL_CLOSED_UNTIL_SEPARATE_EXECUTION_AUTHORITY",
        },
        "generated_files": files,
        "package_file_policy": {
            "acceptance_relative_path": C.ACCEPTANCE_RELATIVE,
            "allowed_relative_directories": ["reference", "review", "tools"],
            "allowed_relative_files": [C.PACKAGE_NAME, *SOURCE_FILES, C.ACCEPTANCE_RELATIVE],
            "inventory_policy": "EXACT_STATIC_FILES_PLUS_ONE_AUTHORITATIVE_REVIEWER_CREATED_ACCEPTANCE",
            "package_relative_path": C.PACKAGE_NAME,
        },
        "predecessor_preservation": {
            "accepted_v28_static": {
                "acceptance_file_sha256": C.V28_STATIC_ACCEPTANCE_SHA256,
                "acceptance_self_sha256": C.V28_STATIC_ACCEPTANCE_SELF_SHA256,
                "accepted_lineage": True,
                "content_sha256": C.V28_STATIC_CONTENT_SHA256,
                "package_file_sha256": C.V28_STATIC_PACKAGE_SHA256,
                "reviewer_provenance_sha256": C.V28_REVIEWER_PROVENANCE_SHA256,
                "root": str(C.V28_STATIC_ROOT),
                "status": "IMMUTABLE_ACCEPTED_STATIC_PREDECESSOR",
            },
            "rejected_v28_execution": {
                "accepted_lineage": False,
                "archive_root": str(C.V28_REJECTED_ARCHIVE),
                "binding_status": "EXPLICITLY_RETIRED_PERMANENTLY_REJECTED_PLANNER_MUTATED_FAILURE_EVIDENCE_ONLY",
                "current_mutated_root": str(C.V28_REJECTED_ROOT),
                "original_content_sha256": C.V28_REJECTED_CONTENT_ORIGINAL_SHA256,
                "original_package_file_sha256": C.V28_REJECTED_PACKAGE_ORIGINAL_SHA256,
                "original_tree_sha256": C.V28_REJECTED_TREE_ORIGINAL_SHA256,
                "restore_chmod_patch_execute_accept_or_reinterpret_permitted": False,
            },
        },
        "regression_contract": {
            "adapter_failure_publication_boundaries": ["SPAWN", "ABI", "RETURN_CODE", "STDOUT", "STDERR", "DECODE", "RESULT_SCHEMA", "RESULT_RECORD", "EXCEPTION", "PUBLICATION"],
            "protected_data_free": True,
            "required_cases": [
                "arbitrary synthetic operator digest",
                "wrong candidate tree without Reviewer provenance",
                "wrong event source/hash/frontier/type",
                "stale/completed/replayed authority",
                "capsule substitution",
                "runtime preexistence and duplicate starts",
                "SOURCE_ORACLE_MATCH and each of 574 single-row mismatches",
                "safe aggregate-only output and zero official effects",
            ],
        },
        "runtime_ordering": {
            "failure_is_terminal_without_retry_replay_resume_or_repair": True,
            "ordered_steps": [
                "validate exact invocation and atomically claim wholly absent runtime",
                "verify and atomically consume external Manager and operator capsules once",
                "write and freeze internal authority envelope",
                "only then cross protected payload/evaluator/target boundary",
            ],
            "runtime_namespace_must_not_preexist": True,
        },
        "schema_version": 1,
        "static_review_output_paths": {
            "candidate_report": str(C.PROJECT_ROOT / "build/v29-b0-execution-binding-repair-static-0001/fresh-l2-sealed-0001/V29_CANDIDATE_REPORT.json"),
            "post_report": str(C.PROJECT_ROOT / "build/v29-b0-execution-binding-repair-static-0001/fresh-l2-post-0001/V29_POST_ACCEPTANCE_REPORT.json"),
        },
        "verification_toolchain": {
            "interpreter_path": str(C.INTERPRETER),
            "interpreter_sha256": C.INTERPRETER_SHA256,
            "interpreter_version": "3.13.5",
            "jsonschema_version": "4.23.0",
            "required_environment": C.EXACT_ENVIRONMENT,
            "required_interpreter_flags": ["-B"],
        },
    }
    package["package_content_sha256"] = C.sha256_bytes(C.compact_bytes(package))
    return package


def prepare() -> None:
    if os.path.lexists(C.ACCEPTANCE_PATH):
        raise RuntimeError("Engineer acceptance already exists")
    canonicalize_json_sources()
    package = build_package()
    C.PACKAGE_PATH.write_bytes(C.compact_bytes(package))


def seal() -> None:
    package, package_raw = C.load_canonical(C.PACKAGE_PATH)
    C.verify_package(package, package_raw)
    C.verify_generated_files(package)
    for relative in package["package_file_policy"]["allowed_relative_files"]:
        if relative == C.ACCEPTANCE_RELATIVE:
            continue
        os.chmod(ACTION_ROOT / relative, 0o444)
    os.chmod(ACTION_ROOT / "reference", 0o555)
    os.chmod(ACTION_ROOT / "tools", 0o555)
    os.chmod(ACTION_ROOT / "review", 0o755)
    os.chmod(ACTION_ROOT, 0o555)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "seal"))
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare()
    else:
        seal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
