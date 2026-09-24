#!/usr/bin/env python3
"""Reverify the accepted V17 static package without mutating acceptance state."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import Any

import v17_evaluator_diagnostic_verify_independent as inert


PRE_REVIEW_REPORT = inert.BUILD_ROOT / "independent-inert-verifier-report.json"
FRESH_L2_RERUN = inert.BUILD_ROOT / "fresh-l2-independent-rerun.json"


def verify_acceptance(pre_review_report: dict[str, Any]) -> dict[str, Any]:
    acceptance, acceptance_raw = inert.canonical_object(inert.ACCEPTANCE)
    inert.verify_self_checksum(acceptance, "acceptance_sha256")

    info = os.lstat(inert.ACCEPTANCE)
    inert.require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), "acceptance file type")
    inert.require(stat.S_IMODE(info.st_mode) == 0o444, "acceptance file mode")

    stored_report, stored_raw = inert.canonical_object(PRE_REVIEW_REPORT)
    rerun_report, rerun_raw = inert.canonical_object(FRESH_L2_RERUN)
    expected_raw = inert.compact_bytes(pre_review_report)
    inert.require(stored_report == pre_review_report and stored_raw == expected_raw, "stored verifier report drift")
    inert.require(rerun_report == pre_review_report and rerun_raw == expected_raw, "Fresh-L2 rerun report drift")
    inert.verify_self_checksum(stored_report, "report_sha256")
    inert.verify_self_checksum(rerun_report, "report_sha256")

    inert.require(acceptance == {
        "acceptance_sha256": acceptance["acceptance_sha256"],
        "accepted": True,
        "action_id": inert.ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_fresh_l2_static_acceptance",
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "decision": "ACCEPT_STATIC_PACKAGE",
        "diagnosis_adapter_cause_class": "EVALUATOR_CALL_OBSERVABILITY_COLLAPSE",
        "diagnosis_underlying_cause_class": "UNKNOWN_IN_PROCESS_EVALUATOR_EXCEPTION",
        "fixture_report_sha256": pre_review_report["fixture"]["report_sha256"],
        "independent_verifier_report_file_sha256": inert.sha256_bytes(stored_raw),
        "independent_verifier_report_sha256": pre_review_report["report_sha256"],
        "independent_verifier_rerun_file_sha256": inert.sha256_bytes(rerun_raw),
        "manifest_sha256": pre_review_report["manifest_sha256"],
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "required_disposition": "V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_STATIC_PACKAGE_READY_FOR_FRESH_L2_NO_EXECUTION_AUTHORITY",
        "reviewer_role": "Fresh-L2",
        "runtime_namespace_file_count": 0,
        "static_acceptance_grants_execution_authority": False,
        "v16_replayed": False,
        "v17_executed": False,
    }, "Fresh-L2 acceptance content")

    return {
        "acceptance_file_sha256": inert.sha256_bytes(acceptance_raw),
        "acceptance_self_sha256": acceptance["acceptance_sha256"],
        "decision": acceptance["decision"],
        "reviewer_role": acceptance["reviewer_role"],
        "static_acceptance_grants_execution_authority": False,
    }


def verify() -> dict[str, Any]:
    inert.require(inert.ACCEPTANCE.is_file(), "Fresh-L2 acceptance absent")
    real_acceptance = inert.ACCEPTANCE
    sentinel = inert.BUILD_ROOT / ".post-acceptance-audit-acceptance-must-remain-absent"
    inert.require(not sentinel.exists(), "post-acceptance sentinel exists")
    inert.AUDIT.update({"official_payload_opens": 0, "official_target_starts": 0})

    try:
        inert.ACCEPTANCE = sentinel
        pre_review_report = inert.verify()
    finally:
        inert.ACCEPTANCE = real_acceptance

    acceptance = verify_acceptance(pre_review_report)
    inert.require(inert.AUDIT == {"official_payload_opens": 0, "official_target_starts": 0}, "inert audit boundary")

    report = {
        "acceptance": acceptance,
        "action_id": inert.ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v17_post_acceptance_inert_audit",
        "manifest_sha256": pre_review_report["manifest_sha256"],
        "official_payload_open_count": inert.AUDIT["official_payload_opens"],
        "official_target_process_starts": inert.AUDIT["official_target_starts"],
        "pre_review_report_file_sha256": inert.sha256_bytes(inert.compact_bytes(pre_review_report)),
        "pre_review_report_sha256": pre_review_report["report_sha256"],
        "preserved_root_count": len(pre_review_report["preservation"]),
        "runtime_namespace_file_count": pre_review_report["runtime"]["file_count"],
        "status": "PASS_V17_EVALUATOR_DIAGNOSTIC_POST_ACCEPTANCE_INERT_AUDIT",
        "v16_replayed": False,
        "v17_executed": False,
    }
    report["report_sha256"] = inert.sha256_bytes(inert.compact_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.addaudithook(inert.audit_hook)
    report = verify()
    raw = inert.compact_bytes(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
    sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
