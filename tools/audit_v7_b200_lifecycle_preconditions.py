#!/usr/bin/env python3
"""Audit local preconditions for the exactly-once V7 B200 lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "design/V7_B200_FULL_LIFECYCLE_ENGINEER_TASK.json"
PIPELINE_PATH = ROOT / "research/PIPELINE_STATE.json"
OUTPUT_PATH = ROOT / "build/v7-b200-full-lifecycle/precondition-audit.json"
FALSE_TRANSITION_MISSION_ID = "6416068ef27b"
RUNNER = ROOT / "pilot/qwen25_05b_bf16_full_finetune_product_v7_runner"

sys.path.insert(0, str(RUNNER))

from runtime import (  # noqa: E402
    ATTEMPT_AUTHORITY_PATH,
    LAUNCH_INTENT,
    LAUNCH_PROCESS,
    LAUNCH_TERMINAL_STATUS,
    RUN_ROOT,
    WORKER_TERMINAL_STATUS,
    execution_package_manifest,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def companion_matches(path: Path, companion: Path | None = None) -> bool:
    companion = companion or path.with_suffix(path.suffix + ".sha256")
    return (
        path.is_file()
        and companion.is_file()
        and companion.read_text(encoding="ascii").split() == [sha256(path), path.name]
    )


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the audit and checksum")
    args = parser.parse_args()

    task = load(TASK_PATH)
    pipeline = load(PIPELINE_PATH)
    observed_package = execution_package_manifest()
    expected_tree = task["accepted_inputs"]["execution_tree_sha256"]
    qualification_binding = task["accepted_inputs"]["one_b200_qualification"]
    qualification_path = ROOT / qualification_binding["path"]
    qualification = load(qualification_path)

    accepted_binding_checks: dict[str, bool] = {}
    accepted_binding_details: dict[str, dict[str, Any]] = {}
    for name, binding in task["accepted_inputs"].items():
        if not isinstance(binding, dict) or "path" not in binding or "sha256" not in binding:
            continue
        path = ROOT / binding["path"]
        observed = sha256(path) if path.is_file() else None
        accepted_binding_checks[f"accepted_input.{name}"] = observed == binding["sha256"]
        accepted_binding_details[name] = {
            "expected_sha256": binding["sha256"],
            "observed_sha256": observed,
            "path": binding["path"],
        }

    stale_transitions = [
        entry
        for entry in pipeline.get("stage_history", [])
        if isinstance(entry, dict) and entry.get("mission_id") == FALSE_TRANSITION_MISSION_ID
    ]
    authority_companion = ATTEMPT_AUTHORITY_PATH.with_suffix(
        ATTEMPT_AUTHORITY_PATH.suffix + ".sha256"
    )
    launch_records = (LAUNCH_INTENT, LAUNCH_PROCESS, LAUNCH_TERMINAL_STATUS, WORKER_TERMINAL_STATUS)
    attempt_markers = sorted(RUN_ROOT.glob("attempt-*/ATTEMPT_CONSUMPTION_MARKER.json"))

    checks: dict[str, bool] = {
        "task_sidecar_exact": companion_matches(TASK_PATH),
        "pipeline_sidecar_exact": companion_matches(
            PIPELINE_PATH, PIPELINE_PATH.with_suffix(".sha256")
        ),
        "pipeline_stage_specification": pipeline.get("current_stage") == "specification",
        "false_rtl_transition_projection_absent": not stale_transitions,
        **accepted_binding_checks,
        "execution_tree_exact": observed_package["tree_sha256"] == expected_tree,
        "qualification_status_exact": qualification.get("status") == qualification_binding["status"],
        "qualification_check_count_exact": len(qualification.get("checks", {})) == qualification_binding["check_count"],
        "qualification_checks_all_pass": bool(qualification.get("checks")) and all(qualification["checks"].values()),
        "attempt_authority_absent": not ATTEMPT_AUTHORITY_PATH.exists(),
        "attempt_authority_companion_absent": not authority_companion.exists(),
        "official_namespace_absent": not RUN_ROOT.exists(),
        "attempt_marker_absent": not attempt_markers,
        "detached_lifecycle_records_absent": not any(path.exists() for path in launch_records),
    }
    ordered_checks = [
        "task_sidecar_exact",
        "pipeline_sidecar_exact",
        "pipeline_stage_specification",
        "false_rtl_transition_projection_absent",
        *accepted_binding_checks,
        "execution_tree_exact",
        "qualification_status_exact",
        "qualification_check_count_exact",
        "qualification_checks_all_pass",
        "attempt_authority_absent",
        "attempt_authority_companion_absent",
        "official_namespace_absent",
        "attempt_marker_absent",
        "detached_lifecycle_records_absent",
    ]
    first_failed = next((name for name in ordered_checks if not checks[name]), None)
    result = {
        "accepted_binding_details": accepted_binding_details,
        "backend_inventory": {
            "claim_boundary": "Not performed by this local audit; a fresh matching-job inventory remains mandatory immediately before authority creation and submission.",
            "status": "REQUIRED_BEFORE_SUBMISSION",
        },
        "check_count": len(checks),
        "checks": checks,
        "claim_boundary": "Non-consuming local V7 lifecycle precondition audit. It creates no authority, launch intent, official namespace, attempt marker, backend job, training, or evaluator evidence.",
        "first_failed_gate": first_failed,
        "observed": {
            "attempt_authority_exists": ATTEMPT_AUTHORITY_PATH.exists(),
            "attempt_markers": [path.relative_to(ROOT).as_posix() for path in attempt_markers],
            "execution_tree_sha256": observed_package["tree_sha256"],
            "false_transition_mission_id": FALSE_TRANSITION_MISSION_ID,
            "false_transition_projection_count": len(stale_transitions),
            "launch_records": {
                path.relative_to(ROOT).as_posix(): path.exists() for path in launch_records
            },
            "official_namespace_exists": RUN_ROOT.exists(),
            "pipeline_sha256": sha256(PIPELINE_PATH),
            "pipeline_stage": pipeline.get("current_stage"),
        },
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "schema_version": 1,
        "status": "LOCAL_READY_FOR_BACKEND_INVENTORY" if first_failed is None else "PRECONDITION_NO_GO",
    }
    if args.write:
        atomic_write_json(OUTPUT_PATH, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if first_failed is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
