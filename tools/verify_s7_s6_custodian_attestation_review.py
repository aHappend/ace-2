#!/usr/bin/env python3
"""Read-only verification of the accepted opaque S7/S6 custodian attestation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "research/execution/qwen25_bf16_successor_s7_20260809"
sys.path.insert(0, str(PACKAGE))

from custodian_attestation import (  # noqa: E402
    EXPECTED_ACCEPTED_PACKAGE_MANIFEST_SHA256,
    EXPECTED_ATTESTATION_SHA256,
    EXPECTED_DATA_MANIFEST_SHA256,
    validate_accepted_opaque_attestation_files,
)


ATTESTATION = (
    ROOT
    / "research/raw/specification/"
    "qwen25-bf16-successor-s7-s6-disjointness-custodian-attestation.json"
)
DATA_MANIFEST = PACKAGE / "materialized/data-manifest.json"
ACCEPTED_PACKAGE_MANIFEST = (
    PACKAGE
    / "history"
    / "accepted-d1c6be0d70c446d86574292aabd136ad4109699d626f15fcea3f69406ad5f053"
    / "package-manifest.json"
)
CONSUMING_PATHS = (
    ROOT / "build/bf16-successor-s7/ATTEMPT_CONSUMPTION_MARKER.json",
    ROOT / "build/bf16-successor-s7/ATTEMPT_RUN_START_SEAL.json",
    ROOT / "build/bf16-successor-s7/backend-submission-intent.json",
    ROOT / "build/bf16-successor-s7/candidate",
    ROOT / "build/bf16-successor-s7/probe-results.json",
    ROOT / "build/bf16-successor-s7/.phase-a-merged.tmp",
    ROOT / "build/bf16-successor-s7/.candidate.tmp",
)


def active_s7_execution_processes() -> list[int]:
    active: list[int] = []
    own_pids = {os.getpid(), os.getppid()}
    proc = Path("/proc")
    if not proc.is_dir():
        return active
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in own_pids:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (
            "run_future.py" in command
            and "qwen25_bf16_successor_s7_20260809" in command
            and "--execute" in command
        ):
            active.append(int(entry.name))
    return sorted(active)


def main() -> int:
    try:
        attestation_report = validate_accepted_opaque_attestation_files(
            ATTESTATION,
            DATA_MANIFEST,
            ACCEPTED_PACKAGE_MANIFEST,
        )
        validation_error = None
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        attestation_report = {"checks": {}, "failures": ["load_or_validation_error"], "result": "FAIL"}
        validation_error = str(error)

    active_pids = active_s7_execution_processes()
    state_checks = {
        "state.known_consuming_paths_absent": not any(
            path.exists() for path in CONSUMING_PATHS
        ),
        "state.no_active_run_future_execute_process": not active_pids,
    }
    checks = {**attestation_report.get("checks", {}), **state_checks}
    failures = [name for name, passed in checks.items() if not passed]
    if validation_error is not None:
        failures.append("load_or_validation_error")
    result = "PASS" if not failures else "FAIL"
    report = {
        "active_run_future_execute_pids": active_pids,
        "bindings": {
            "accepted_package_manifest_sha256": EXPECTED_ACCEPTED_PACKAGE_MANIFEST_SHA256,
            "attestation_sha256": EXPECTED_ATTESTATION_SHA256,
            "data_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256,
        },
        "check_count": len(checks),
        "checks": checks,
        "failed_check_count": len(failures),
        "failures": failures,
        "measurements": attestation_report.get("measurements", {}),
        "result": result,
        "scope": "READ_ONLY_OPAQUE_ATTESTATION_VERIFICATION_NO_ACCEPTANCE_OR_AUTHORITY",
        "validation_error": validation_error,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
