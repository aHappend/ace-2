#!/usr/bin/env python3
"""Post-acceptance inert verifier for the static-only V18 package."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import Any

import verify_qk_gbfp8_head64_granularity_sweep_c02_binding_repair_candidate_v18 as candidate


def verify_acceptance(package: dict[str, Any], pre_report: dict[str, Any]) -> dict[str, Any]:
    acceptance, raw = candidate.canonical(candidate.ACCEPTANCE)
    candidate.verify_self_hash(acceptance, "acceptance_sha256")
    info = os.lstat(candidate.ACCEPTANCE)
    candidate.require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), "acceptance file type")
    candidate.require(stat.S_IMODE(info.st_mode) == 0o444, "acceptance file mode")
    expected = {
        "acceptance_sha256": acceptance["acceptance_sha256"],
        "accepted": True,
        "action_id": candidate.EXPECTED_ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_fresh_l2_static_acceptance",
        "binding_table_file_sha256": package["complete_binding_table"]["file_sha256"],
        "candidate_report_sha256": pre_report["report_sha256"],
        "claim_boundary": candidate.EXPECTED_CLAIM_BOUNDARY,
        "decision": "ACCEPT_STATIC_PACKAGE",
        "fixture_report_file_sha256": package["synthetic_fixture"]["report_file_sha256"],
        "manifest_file_sha256": candidate.sha256_file(candidate.MANIFEST),
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "reviewer_role": "Fresh-L2",
        "static_acceptance_grants_execution_authority": False,
        "v17_retried": False,
        "v18_executed": False,
    }
    candidate.require(acceptance == expected, "Fresh-L2 acceptance content")
    return {
        "acceptance_file_sha256": candidate.sha256_bytes(raw),
        "acceptance_sha256": acceptance["acceptance_sha256"],
        "decision": acceptance["decision"],
        "reviewer_role": acceptance["reviewer_role"],
    }


def verify() -> dict[str, Any]:
    candidate.require(candidate.ACCEPTANCE.is_file(), "Fresh-L2 acceptance absent")
    package, _ = candidate.verify_manifest()
    pre_report = candidate.verify(True)
    acceptance = verify_acceptance(package, pre_report)
    candidate.require(candidate.AUDIT == {"official_payload_opens": 0, "official_target_starts": 0}, "post-acceptance inert audit boundary")
    report = {
        "acceptance": acceptance,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_post_acceptance_inert_report",
        "candidate_report_sha256": pre_report["report_sha256"],
        "inventory_policy": "EXACT_STATIC_FILES_PLUS_ONE_BOUND_FRESH_L2_ACCEPTANCE",
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "status": "PASS_V18_C02_BINDING_REPAIR_POST_ACCEPTANCE_INERT",
        "v17_retried": False,
        "v18_executed": False,
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
