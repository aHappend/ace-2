#!/usr/bin/env python3
"""Create or validate the Package0015 fail-closed pre-execution guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import prepare_position01_package0015_admission as admission


ROOT = Path("/home/argustest/ace-2")
ADMISSION = (
    ROOT
    / ".argus/live/package0015-no-execution-admission-v1"
    / "admission-preparation.json"
)
TARGET_DIR = ROOT / ".argus/live/package0015-runtime-preexecution-guard-v1"
TARGET = TARGET_DIR / "runtime-guard.json"

PACKAGE0014_NAMESPACES = {
    "package": (
        ROOT / "reports/ace2-position01-integrated-runtime-pass-package-0014"
    ),
    "grant": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-manager-grant-0014"
    ),
    "gate": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-live-launch-gate-0014"
    ),
    "execution_authorization": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-execution-authorization-0014"
    ),
    "consumption": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-consumption-state-0014"
    ),
    "output": ROOT / "reports/ace2-position01-runtime-pass-0007",
    "scratch": ROOT / "reports/ace2-position01-runtime-pass-scratch-0007",
}

ZERO_EXECUTION_ACTIVITY = {
    "authority_consumed": 0,
    "consumption_created": 0,
    "continuation_created": 0,
    "execution_authorization_created": 0,
    "executor_invocation": 0,
    "generated_tokens": 0,
    "grant_consumed": 0,
    "launch_gate_created": 0,
    "manager_grant_created": 0,
    "model_execution": 0,
    "output_created": 0,
    "reference_execution": 0,
    "replay_created": 0,
    "rtl_execution": 0,
    "scratch_created": 0,
    "stage1_closure": 0,
    "stage2_execution": 0,
    "workload_execution": 0,
}


class GuardError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GuardError(message)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
    except ValueError:
        return False
    return True


def validate_namespace_plan(
    record: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    package = record.get("package0015")
    require(isinstance(package, dict), "Package0015 admission section absent")
    observed = package.get("fresh_namespaces")
    require(isinstance(observed, dict), "Package0015 namespace plan absent")
    expected_labels = set(admission.PROSPECTIVE_PATHS)
    require(set(observed) == expected_labels, "Package0015 namespace labels changed")

    paths = list(admission.PROSPECTIVE_PATHS.items())
    for label, path in paths:
        require(
            observed[label]
            == {"exists": False, "path": relative(path)}
            and not os.path.lexists(path),
            f"Package0015 {label} namespace is not fresh",
        )
    for index, (left_label, left) in enumerate(paths):
        for right_label, right in paths[index + 1 :]:
            require(
                not nested(left, right) and not nested(right, left),
                f"Package0015 namespaces overlap: {left_label}, {right_label}",
            )
        for old_label, old in PACKAGE0014_NAMESPACES.items():
            require(
                not nested(left, old) and not nested(old, left),
                (
                    "Package0015 namespace overlaps Package0014: "
                    f"{left_label}, {old_label}"
                ),
            )
    return observed


def validated_admission() -> dict[str, Any]:
    require(
        ADMISSION.is_file() and not ADMISSION.is_symlink(),
        "validated Package0015 admission artifact absent",
    )
    record = admission.load_json(ADMISSION)
    admission.validate_record(record)
    require(
        record.get("status")
        == "PASS_PACKAGE0015_NO_EXECUTION_ADMISSION_PREPARATION",
        "Package0015 admission preparation did not pass",
    )
    return record


def build_record() -> dict[str, Any]:
    admission_record = validated_admission()
    fresh_namespaces = validate_namespace_plan(admission_record)
    capacity = admission.available_bytes()
    require(
        capacity >= admission.REQUIRED_AVAILABLE_BYTES,
        "live capacity below the operator threshold",
    )
    require(
        admission.package0015_executor_count() == 0,
        "Package0015 executor process exists",
    )
    directive = admission_record["active_directive"]
    compatibility = admission_record["directive_compatibility"]
    require(
        directive.get("authorizes_package0015") is False
        and compatibility.get("status") == "ambiguous_objective",
        "active directive unexpectedly authorizes Package0015",
    )

    return {
        "activity": ZERO_EXECUTION_ACTIVITY,
        "admission_preparation": {
            "path": relative(ADMISSION),
            "sha256": sha256_file(ADMISSION),
            "status": admission_record["status"],
            "validation": "PASS",
        },
        "dispatch_guard": {
            "decision": "BLOCK_BEFORE_AUTHORITY_CONSUMPTION",
            "dispatch_allowed": False,
            "reason": (
                "Active Manager V49 authorizes terminal Package0014 only; "
                "Package0015 has no package, authenticated simulator binding, "
                "grant, gate, or execution authorization."
            ),
            "status": "PASS_FAIL_CLOSED",
        },
        "live_prerequisites": {
            "authentication": {
                "active_directive_authorizes_package0015": False,
                "authenticated_simulator_binding_verified": False,
                "package0015_execution_authority_exists": False,
                "requirement": (
                    "A fresh Package0015 directive and sealed successor must "
                    "authenticate and admit the exact pinned simulator "
                    "executable identity before authority consumption."
                ),
                "satisfied": False,
                "status": "BLOCKED_MISSING_PACKAGE0015_AUTHORITY",
            },
            "capacity": {
                "available_bytes": capacity,
                "method": "os.statvfs(ROOT).f_bavail_times_f_frsize",
                "required_available_bytes": admission.REQUIRED_AVAILABLE_BYTES,
                "satisfied": True,
            },
            "fresh_namespaces": {
                "all_absent": True,
                "pairwise_disjoint": True,
                "package0014_disjoint": True,
                "plan": fresh_namespaces,
            },
            "package0015_executor_process_count": 0,
        },
        "package0014_terminal_binding": (
            admission_record["package0014_retirement"]
        ),
        "same_level_conflict": {
            "classification": "ambiguous_objective",
            "current_bounded_objective": "PACKAGE0015_PREEXECUTION_GUARD",
            "live_manager_directive": "PACKAGE0014_EXECUTION_AUTHORIZATION_PREPARATION",
            "resolution": (
                "No Package0015 dispatch state may be created until a fresh "
                "Manager or operator directive resolves the conflict."
            ),
        },
        "schema": "ace2-position01-package0015-runtime-preexecution-guard-v1",
        "status": "PASS_PACKAGE0015_PREEXECUTION_GUARD_BLOCKED_AUTHORIZATION_REQUIRED",
    }


def validate_record(record: dict[str, Any]) -> None:
    expected = build_record()
    observed_prerequisites = record.get("live_prerequisites")
    require(isinstance(observed_prerequisites, dict), "live prerequisites absent")
    observed_capacity = observed_prerequisites.get("capacity")
    require(isinstance(observed_capacity, dict), "capacity prerequisite absent")
    recorded_bytes = observed_capacity.get("available_bytes")
    require(
        isinstance(recorded_bytes, int)
        and recorded_bytes >= admission.REQUIRED_AVAILABLE_BYTES,
        "recorded capacity did not satisfy the threshold",
    )
    expected["live_prerequisites"]["capacity"]["available_bytes"] = recorded_bytes
    require(record == expected, "Package0015 runtime guard changed")


def write_exclusive(record: dict[str, Any]) -> None:
    os.mkdir(TARGET_DIR, mode=0o700)
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    descriptor = os.open(
        TARGET,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(TARGET_DIR, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_record(admission.load_json(TARGET))
    else:
        write_exclusive(build_record())
        validate_record(admission.load_json(TARGET))
    print(
        json.dumps(
            {
                "artifact": relative(TARGET),
                "dispatch_allowed": False,
                "status": (
                    "PASS_PACKAGE0015_PREEXECUTION_GUARD_"
                    "BLOCKED_AUTHORIZATION_REQUIRED"
                ),
                "validation": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
