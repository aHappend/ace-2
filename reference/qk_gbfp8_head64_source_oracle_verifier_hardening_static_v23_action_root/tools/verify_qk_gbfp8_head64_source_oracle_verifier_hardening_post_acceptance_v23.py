#!/usr/bin/env python3
"""Post-acceptance inert verifier for independently accepted static V23."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import verify_qk_gbfp8_head64_source_oracle_verifier_hardening_candidate_v23 as candidate


def verify_acceptance(package: dict[str, Any], pre_report: dict[str, Any], negative: dict[str, Any]) -> dict[str, Any]:
    acceptance, raw = candidate.canonical(candidate.ACCEPTANCE)
    candidate.verify_self_hash(acceptance, "acceptance_sha256")
    expected = {
        "acceptance_sha256": acceptance["acceptance_sha256"],
        "accepted": True,
        "action_id": candidate.EXPECTED_ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_source_oracle_verifier_hardening_static_v23_fresh_l2_acceptance",
        "b0_sidecar_schema_file_sha256": package["b0_identity_control"]["schema_file_sha256"],
        "b0_validator_file_sha256": package["b0_identity_control"]["validator_file_sha256"],
        "candidate_report_file_sha256": candidate.sha256_bytes(candidate.compact_bytes(pre_report)),
        "candidate_report_sha256": pre_report["report_sha256"],
        "claim_boundary": candidate.EXPECTED_CLAIM,
        "decision": "ACCEPT_STATIC_PACKAGE",
        "negative_fixture_report_file_sha256": candidate.sha256_bytes(candidate.compact_bytes(negative)),
        "negative_fixture_report_sha256": negative["report_sha256"],
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "package_file_sha256": candidate.sha256_file(candidate.PACKAGE),
        "regression_report_file_sha256": package["arithmetic_regressions"]["report_file_sha256"],
        "reviewer_role": "Fresh-L2",
        "runtime_namespace_file_count": 0,
        "static_acceptance_grants_execution_authority": False,
        "v21_or_predecessor_mutated": False,
        "v22_action_tree_sha256": package["predecessor_preservation"]["v22_action_root"]["tree_sha256"],
        "v22_mutated_or_accepted": False,
        "v23_executed": False,
    }
    candidate.require(acceptance == expected, "Fresh-L2 acceptance content")
    candidate.require(candidate.sha256_file(candidate.ACCEPTANCE_SCHEMA) == package["fresh_l2_review"]["acceptance_schema_file_sha256"], "acceptance schema binding")
    return {"acceptance_file_sha256": candidate.sha256_bytes(raw), "acceptance_sha256": acceptance["acceptance_sha256"], "decision": acceptance["decision"]}


def verify() -> dict[str, Any]:
    candidate.require(candidate.ACCEPTANCE.is_file(), "Fresh-L2 acceptance absent")
    package, _ = candidate.verify_manifest()
    pre_report, negative = candidate.verify(True)
    acceptance = verify_acceptance(package, pre_report, negative)
    report = {
        "acceptance": acceptance,
        "artifact_kind": "qk_gbfp8_head64_source_oracle_verifier_hardening_static_v23_post_acceptance_inert_report",
        "candidate_report_sha256": pre_report["report_sha256"],
        "claim_boundary": candidate.EXPECTED_CLAIM,
        "negative_fixture_report_sha256": negative["report_sha256"],
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "status": "PASS_V23_STATIC_POST_ACCEPTANCE_INERT",
        "v23_execution_authority_granted": False,
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
    sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
