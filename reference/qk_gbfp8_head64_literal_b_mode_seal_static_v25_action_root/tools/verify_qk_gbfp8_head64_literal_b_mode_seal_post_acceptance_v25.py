#!/usr/bin/env python3
"""Final inert verifier for one independently accepted, mode-sealed V25."""

from __future__ import annotations

import sys

EARLY_DONT_WRITE_BYTECODE = sys.dont_write_bytecode is True
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import jsonschema

from literal_b_launch_guard_v25 import prove_effective_suppression, validate_current_process


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACTION_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = PROJECT_ROOT / "build/v25-literal-b-mode-seal-static-0002"
LAUNCH_PROOF = {**validate_current_process(Path(__file__), EARLY_DONT_WRITE_BYTECODE), **prove_effective_suppression()}


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory_records(root: Path) -> list[dict[str, Any]]:
    info = os.lstat(root)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError("V25 action root absent, linked, or non-directory")
    records: list[dict[str, Any]] = [{"kind": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": "."}]
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"V25 action-tree symlink: {relative}")
        mode = f"{stat.S_IMODE(info.st_mode):04o}"
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "directory", "mode": mode, "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append({"kind": "file", "mode": mode, "path": relative, "sha256": sha256_file(path), "size": info.st_size})
        else:
            raise RuntimeError(f"V25 unsupported action-tree entry: {relative}")
    return records


PRE_IMPORT_TREE = inventory_records(ACTION_ROOT)

import verify_qk_gbfp8_head64_literal_b_mode_seal_candidate_v25 as candidate


def canonical_report(relative: str) -> tuple[Path, dict[str, Any], bytes]:
    path = (BUILD_ROOT / relative).resolve(strict=True)
    try:
        path.relative_to(BUILD_ROOT.resolve(strict=True))
    except ValueError as error:
        raise candidate.VerificationError(f"Fresh-L2 report escapes declared build root: {relative}") from error
    candidate.require(path.is_file() and not path.is_symlink(), f"Fresh-L2 report absent or linked: {relative}")
    value, raw = candidate.canonical(path)
    return path, value, raw


def verify_acceptance(package: dict[str, Any]) -> dict[str, Any]:
    acceptance, raw = candidate.canonical(candidate.ACCEPTANCE)
    candidate.verify_self_hash(acceptance, "acceptance_sha256")
    schema = candidate.json_object(candidate.ACCEPTANCE_SCHEMA)
    jsonschema.Draft202012Validator(schema).validate(acceptance)

    _, candidate_report, candidate_raw = canonical_report(acceptance["candidate_report_path"])
    _, negative_report, negative_raw = canonical_report(acceptance["negative_fixture_report_path"])
    _, matrix_report, matrix_raw = canonical_report(acceptance["launch_matrix_report_path"])

    candidate.require(sha256_bytes(candidate_raw) == acceptance["candidate_report_file_sha256"], "accepted candidate report file hash")
    candidate.require(candidate_report["report_sha256"] == acceptance["candidate_report_sha256"], "accepted candidate report self hash")
    candidate.require(candidate_report["status"] == "PASS_V25_STATIC_CANDIDATE_PENDING_FRESH_L2", "accepted candidate status")
    candidate.require(candidate_report["package_file_sha256"] == sha256_file(candidate.PACKAGE), "accepted candidate package hash")
    candidate.require(candidate_report["inventory"]["mode_seal_exact"] is True, "accepted candidate mode seal")
    candidate.require(candidate_report["inventory_guard"]["candidate_action_tree_before_sha256"] == acceptance["candidate_action_tree_sha256"], "accepted candidate tree hash")

    candidate.require(sha256_bytes(negative_raw) == acceptance["negative_fixture_report_file_sha256"], "accepted negative report file hash")
    candidate.require(negative_report["report_sha256"] == acceptance["negative_fixture_report_sha256"], "accepted negative report self hash")
    candidate.require(negative_report["status"] == "PASS_V25_LITERAL_B_MODE_SEAL_AND_SYNTHETIC_MUTATION_REJECTION", "accepted negative status")
    candidate.require(negative_report["package_contract_mutations"]["rejected_mutation_count"] == 187, "accepted carried mutation count")
    candidate.require(negative_report["b0_document_mutations"]["rejected_mutation_count"] == 19, "accepted B0 mutation count")

    candidate.require(sha256_bytes(matrix_raw) == acceptance["launch_matrix_report_file_sha256"], "accepted launch matrix file hash")
    candidate.require(matrix_report["report_sha256"] == acceptance["launch_matrix_report_sha256"], "accepted launch matrix self hash")
    candidate.require(matrix_report["status"] == "PASS_V25_LITERAL_B_LAUNCH_MATRIX_PENDING_FRESH_L2", "accepted launch matrix status")
    candidate.require(matrix_report["negative_launch_count"] == acceptance["negative_launch_count"] == 18, "accepted launch matrix count")
    candidate.require(matrix_report["candidate_positive"]["candidate_report_file_sha256"] == acceptance["candidate_report_file_sha256"], "matrix candidate report binding")
    candidate.require(matrix_report["candidate_positive"]["negative_fixture_report_file_sha256"] == acceptance["negative_fixture_report_file_sha256"], "matrix negative report binding")
    candidate.require(matrix_report["sealed_action_tree_before_sha256"] == matrix_report["sealed_action_tree_after_sha256"], "matrix sealed tree identity")

    expected_dynamic = {
        "action_id": candidate.EXPECTED_ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_literal_b_mode_seal_static_v25_fresh_l2_acceptance",
        "candidate_action_tree_sha256": candidate_report["inventory_guard"]["candidate_action_tree_before_sha256"],
        "claim_boundary": candidate.EXPECTED_CLAIM,
        "decision": "ACCEPT_STATIC_PACKAGE",
        "package_content_sha256": package["package_content_sha256"],
        "package_file_sha256": sha256_file(candidate.PACKAGE),
        "post_acceptance_verifier_file_sha256": package["generated_files"][package["bytecode_free_verifier_contract"]["post_acceptance_entrypoint"]]["sha256"],
        "reviewer_role": "Fresh-L2",
        "v22_action_tree_sha256": package["predecessor_preservation"]["v22_action_root"]["tree_sha256"],
        "v23_action_tree_sha256": package["v23_preservation"]["v23_action_root"]["tree_sha256"],
        "v24_action_tree_sha256": package["v24_preservation"]["v24_action_root"]["tree_sha256"],
        "v24_package_file_sha256": package["v24_preservation"]["package"]["sha256"],
    }
    for key, expected in expected_dynamic.items():
        candidate.require(acceptance[key] == expected, f"acceptance binding: {key}")
    candidate.require(sha256_file(candidate.ACCEPTANCE_SCHEMA) == package["fresh_l2_review"]["acceptance_schema_file_sha256"], "acceptance schema binding")
    return {
        "acceptance_file_sha256": sha256_bytes(raw),
        "acceptance_sha256": acceptance["acceptance_sha256"],
        "candidate_report_file_sha256": acceptance["candidate_report_file_sha256"],
        "decision": acceptance["decision"],
        "launch_matrix_report_file_sha256": acceptance["launch_matrix_report_file_sha256"],
    }


def verify() -> dict[str, Any]:
    candidate.require(candidate.ACCEPTANCE.is_file() and not candidate.ACCEPTANCE.is_symlink(), "Fresh-L2 acceptance absent or linked")
    candidate.require(f"{stat.S_IMODE(os.lstat(candidate.ACCEPTANCE).st_mode):04o}" == "0444", "Fresh-L2 acceptance mode")
    candidate.require(f"{stat.S_IMODE(os.lstat(candidate.ACCEPTANCE.parent).st_mode):04o}" == "0555", "Fresh-L2 review directory mode")
    package, _ = candidate.verify_manifest()
    inventory = candidate.verify_inventory(package, True)
    candidate.verify_generated_files(package)
    candidate.verify_mission_and_toolchain()
    candidate.verify_v24_preservation()
    candidate.verify_v23_preservation()
    candidate.verify_v22_and_predecessors()
    candidate.verify_regression_report()
    candidate.verify_no_live_effects(package)
    acceptance = verify_acceptance(package)
    post_tree = inventory_records(ACTION_ROOT)
    candidate.require(post_tree == PRE_IMPORT_TREE, "V25 post-acceptance verifier mutated exact final tree")
    tree_sha256 = sha256_bytes(compact_bytes(post_tree))
    report = {
        "acceptance": acceptance,
        "artifact_kind": "qk_gbfp8_head64_literal_b_mode_seal_static_v25_post_acceptance_inert_report",
        "claim_boundary": candidate.EXPECTED_CLAIM,
        "final_mode_seal": inventory,
        "launch_proof": LAUNCH_PROOF,
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "post_acceptance_inventory": {
            "action_tree_after_sha256": tree_sha256,
            "action_tree_before_sha256": tree_sha256,
            "exact_tree_and_mode_match": True,
            "forbidden_artifact_count": 0
        },
        "status": "PASS_V25_STATIC_POST_ACCEPTANCE_LITERAL_B_MODE_SEALED",
        "v25_execution_authority_granted": False
    }
    report["report_sha256"] = candidate.sha256_bytes(candidate.compact_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.addaudithook(candidate.audit_hook)
    report = verify()
    raw = candidate.compact_bytes(report)
    output = args.output.resolve(strict=False)
    try:
        output.relative_to(BUILD_ROOT.resolve(strict=True))
    except ValueError as error:
        raise candidate.VerificationError("V25 post report must stay inside declared build root") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
