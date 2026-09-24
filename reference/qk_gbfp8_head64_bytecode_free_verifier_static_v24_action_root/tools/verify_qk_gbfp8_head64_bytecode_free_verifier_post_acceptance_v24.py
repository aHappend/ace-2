#!/usr/bin/env python3
"""Bytecode-free post-acceptance verifier for independently accepted static V24."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACTION_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = PROJECT_ROOT / "build/v24-bytecode-free-verifier-static-0001"
FORBIDDEN_COMPONENTS = {"__pycache__"}
FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".swp", ".temp", ".tmp", "~")


def _compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_records(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"V24 action root absent or linked: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"V24 action-tree symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if FORBIDDEN_COMPONENTS.intersection(Path(relative).parts) or relative.endswith(FORBIDDEN_SUFFIXES):
            raise RuntimeError(f"V24 forbidden pre-import artifact: {relative}")
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append({"kind": "file", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative, "sha256": _sha256_file(path), "size": info.st_size})
        else:
            raise RuntimeError(f"V24 unsupported action-tree entry: {path}")
    return records


if sys.flags.dont_write_bytecode != 1:
    raise RuntimeError("V24 post-acceptance verifier requires interpreter -B")
if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
    raise RuntimeError("V24 post-acceptance verifier requires PYTHONDONTWRITEBYTECODE=1")
if sys.dont_write_bytecode is not True:
    raise RuntimeError("V24 post-acceptance verifier requires early sys.dont_write_bytecode=True")
if sys.pycache_prefix is not None:
    raise RuntimeError("V24 post-acceptance verifier rejects PYTHONPYCACHEPREFIX")

PRE_IMPORT_TREE = _inventory_records(ACTION_ROOT)

import verify_qk_gbfp8_head64_bytecode_free_verifier_candidate_v24 as candidate


def verify_acceptance(package: dict[str, Any], pre_report: dict[str, Any], negative: dict[str, Any]) -> dict[str, Any]:
    acceptance, raw = candidate.canonical(candidate.ACCEPTANCE)
    candidate.verify_self_hash(acceptance, "acceptance_sha256")
    expected = {
        "acceptance_sha256": acceptance["acceptance_sha256"],
        "accepted": True,
        "action_id": candidate.EXPECTED_ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_bytecode_free_verifier_static_v24_fresh_l2_acceptance",
        "b0_sidecar_schema_file_sha256": package["b0_identity_control"]["schema_file_sha256"],
        "b0_validator_file_sha256": package["b0_identity_control"]["validator_file_sha256"],
        "bytecode_free_launch_required": True,
        "candidate_action_tree_sha256": pre_report["inventory_guard"]["candidate_action_tree_before_sha256"],
        "candidate_report_file_sha256": candidate.sha256_bytes(candidate.compact_bytes(pre_report)),
        "candidate_report_sha256": pre_report["report_sha256"],
        "claim_boundary": candidate.EXPECTED_CLAIM,
        "decision": "ACCEPT_STATIC_PACKAGE",
        "negative_fixture_report_file_sha256": candidate.sha256_bytes(candidate.compact_bytes(negative)),
        "negative_fixture_report_sha256": negative["report_sha256"],
        "normal_import_inventory_mutation_detected": True,
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "package_file_sha256": candidate.sha256_file(candidate.PACKAGE),
        "post_acceptance_verifier_file_sha256": package["generated_files"][package["bytecode_free_verifier_contract"]["post_acceptance_entrypoint"]]["sha256"],
        "regression_report_file_sha256": package["arithmetic_regressions"]["report_file_sha256"],
        "reviewer_role": "Fresh-L2",
        "runtime_namespace_file_count": 0,
        "static_acceptance_grants_execution_authority": False,
        "v21_or_predecessor_mutated": False,
        "v22_action_tree_sha256": package["predecessor_preservation"]["v22_action_root"]["tree_sha256"],
        "v23_acceptance_file_sha256": package["v23_preservation"]["acceptance"]["sha256"],
        "v23_action_tree_sha256": package["v23_preservation"]["v23_action_root"]["tree_sha256"],
        "v23_mutated_resealed_or_accepted_in_place": False,
        "v23_preserved_byte_identical": True,
        "v24_executed": False,
    }
    candidate.require(acceptance == expected, "Fresh-L2 acceptance content")
    candidate.require(candidate.sha256_file(candidate.ACCEPTANCE_SCHEMA) == package["fresh_l2_review"]["acceptance_schema_file_sha256"], "acceptance schema binding")
    return {"acceptance_file_sha256": candidate.sha256_bytes(raw), "acceptance_sha256": acceptance["acceptance_sha256"], "decision": acceptance["decision"]}


def verify() -> dict[str, Any]:
    candidate.require(candidate.ACCEPTANCE.is_file(), "Fresh-L2 acceptance absent")
    package, _ = candidate.verify_manifest()
    pre_report, negative = candidate.verify(True)
    acceptance = verify_acceptance(package, pre_report, negative)
    post_tree = _inventory_records(ACTION_ROOT)
    candidate.require(post_tree == PRE_IMPORT_TREE, "V24 post-acceptance verifier mutated exact action tree")
    tree_sha256 = candidate.sha256_bytes(_compact_bytes(post_tree))
    report = {
        "acceptance": acceptance,
        "artifact_kind": "qk_gbfp8_head64_bytecode_free_verifier_static_v24_post_acceptance_inert_report",
        "candidate_report_sha256": pre_report["report_sha256"],
        "claim_boundary": candidate.EXPECTED_CLAIM,
        "negative_fixture_report_sha256": negative["report_sha256"],
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "post_acceptance_inventory": {
            "action_tree_after_sha256": tree_sha256,
            "action_tree_before_sha256": tree_sha256,
            "exact_tree_match": True,
            "forbidden_artifact_count": 0,
        },
        "status": "PASS_V24_STATIC_POST_ACCEPTANCE_BYTECODE_FREE",
        "v24_execution_authority_granted": False,
    }
    report["report_sha256"] = candidate.sha256_bytes(candidate.compact_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.addaudithook(candidate.audit_hook)
    report = verify()
    raw = candidate.compact_bytes(report)
    if args.output is not None:
        try:
            args.output.resolve(strict=False).relative_to(BUILD_ROOT.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise candidate.VerificationError(f"V24 post report output must stay inside declared build root: {args.output}") from error
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
    sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
