#!/usr/bin/env python3
"""Read-only audit for rejected S6 v1 containment and clean-room rerouting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
REJECTED_DESIGN = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_CATEGORY_BALANCED_SUCCESSOR_S6_DESIGN_FREEZE.json"
REJECTED_AUDIT = ROOT / "build/bf16-category-balanced-successor-s6-design-freeze/package-audit.json"
DISPOSITION = ROOT / "research/raw/specification/qwen25-bf16-category-balanced-successor-s6-design-freeze-v1-review-replan-20260808.json"
TASK = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_SUCCESSOR_S6_CLEAN_ROOM_REPLACEMENT_TASK.json"
SPEC = ROOT / "design/SPEC.md"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
CHECKPOINT = ROOT / "CHECKPOINT.md"
LATEST = ROOT / "latest.json"

S5_SOURCES = {
    ROOT / "build/bf16-full-finetune-successor-s5/backend-terminal-audit-20260808T150208Z.json":
        "bd7bd5c0f55bbb63b9bd66473eb95dc0306380bdd96d53c5c71600b555c192b8",
    ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_SUCCESSOR_S5_EXECUTION_PACKAGE_CONTRACT.json":
        "6b9425f027bcc50f399fdded1ac91147fbdde91c9a041e10d1e00e6b3c9c0a54",
    ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-successor-s5/training_recipe.json":
        "24a4b8acf314c0bd90a026b0a2438c424ab281a0f29c9ab9e183e9b272b42df2",
}

EXPECTED = {
    "pipeline": "1fdc818abc8458749eeccb2fa3dc6a09d2dc2ab7fd6ea2e446ee222f777999bc",
    "rejected_design": "0fc1c2dcb10c3b477045f0767bb60f03e5f06a7d7c9b349db82cea34b37355cb",
    "rejected_audit": "a0db50afb9081227e262fcbad1ec31a3ae7f055fe565c8edfc210dabefaeced2",
    "disposition": "018a1a8283e8705ee64bfb6742aa3bb00db3a9400cad101f8447432b7603b05f",
    "task": "6243dd3bbacdf35b572b0a7c402f00318b2d2ad572e76740616fd9df4d80148a",
    "review_round": "08590f7de91381b9463ecb87c7193508fa8201a7dba4ef374ac736cae6dfdce7",
    "review_checkpoint": "0baa28d11cdb7df47660a0165d66f7d157d1450fe2010aaa7a1f9c65cb7c23d4",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def companion_matches(path: Path) -> bool:
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        return False
    fields = sidecar.read_text(encoding="ascii").strip().split()
    return len(fields) >= 2 and fields[0] == sha256(path) and fields[-1] == path.name


def audit() -> dict:
    pipeline = load(PIPELINE)
    rejected_audit = load(REJECTED_AUDIT)
    disposition = load(DISPOSITION)
    task = load(TASK)
    latest = load(LATEST)
    spec = SPEC.read_text(encoding="utf-8")
    ground_truth = GROUND_TRUTH.read_text(encoding="utf-8")
    checkpoint = CHECKPOINT.read_text(encoding="utf-8")

    checks: dict[str, bool] = {}
    for path, expected in S5_SOURCES.items():
        checks[f"source.{path.name}.sha256_exact"] = sha256(path) == expected

    checks["state.pipeline_hash_unchanged"] = sha256(PIPELINE) == EXPECTED["pipeline"]
    checks["state.current_stage_specification"] = pipeline.get("current_stage") == "specification"
    checks["rejected.design_hash_unchanged"] = sha256(REJECTED_DESIGN) == EXPECTED["rejected_design"]
    checks["rejected.audit_hash_unchanged"] = sha256(REJECTED_AUDIT) == EXPECTED["rejected_audit"]
    checks["rejected.content_audit_still_passes"] = (
        rejected_audit.get("status") == "PASS_MARKER_FREE_S6_DESIGN_FREEZE_ZERO_CONSUMING_STATE"
        and rejected_audit.get("check_count") == 39
        and rejected_audit.get("failed_checks") == []
    )
    checks["rejected.disposition_hash_exact"] = sha256(DISPOSITION) == EXPECTED["disposition"]
    checks["rejected.disposition_status_exact"] = (
        disposition.get("artifact_disposition", {}).get("status") == "PRESERVED_NON_ACCEPTED_PROVENANCE"
        and disposition.get("artifact_disposition", {}).get("process_provenance_accepted") is False
        and disposition.get("preserved_artifacts", {}).get("review_round", {}).get("status") == "replan_requested"
        and disposition.get("preserved_artifacts", {}).get("review_round", {}).get("sha256") == EXPECTED["review_round"]
        and disposition.get("preserved_artifacts", {}).get("review_checkpoint", {}).get("sha256") == EXPECTED["review_checkpoint"]
    )

    task_sources = {entry.get("path"): entry.get("sha256") for entry in task.get("permitted_evidence_sources", [])}
    expected_sources = {str(path.relative_to(ROOT)): digest for path, digest in S5_SOURCES.items()}
    checks["task.hash_exact"] = sha256(TASK) == EXPECTED["task"]
    checks["task.identity_and_status_exact"] = (
        task.get("task_id") == "qwen2.5-0.5b-instruct-ace2-bf16-successor-s6-clean-room-replacement-task-v1"
        and task.get("status") == "READY_FOR_FRESH_CLEAN_ROOM_CONSTRUCTION_NO_SUCCESSOR_FROZEN"
        and task.get("current_stage") == "specification"
    )
    checks["task.permitted_sources_exact"] = task_sources == expected_sources
    checks["task.rejected_s6_forbidden"] = str(REJECTED_DESIGN.relative_to(ROOT)) in task.get("forbidden_construction_inputs", [])
    checks["task.fresh_reviewer_required"] = task.get("acceptance", {}).get("fresh_reviewer_required") is True
    checks["task.zero_consuming_boundary_explicit"] = len(task.get("acceptance", {}).get("zero_state_requirements", [])) == 4

    checks["projection.spec_current"] = "PRESERVED_NON_ACCEPTED_PROVENANCE" in spec and EXPECTED["task"] in spec
    checks["projection.ground_truth_current"] = "PRESERVED_NON_ACCEPTED_PROVENANCE" in ground_truth and EXPECTED["task"] in ground_truth
    checks["projection.checkpoint_current"] = "replan_requested" in checkpoint and EXPECTED["task"] in checkpoint
    checks["projection.latest_current"] = (
        latest.get("status") == "CURRENT_S6_V1_REPLAN_CLEAN_ROOM_REPLACEMENT_TASK_READY"
        and latest.get("current_stage") == "specification"
        and latest.get("planner_execution_permitted_in_current_context") is False
        and latest.get("replacement_task", {}).get("sha256") == EXPECTED["task"]
        and latest.get("s6_v1", {}).get("disposition") == "PRESERVED_NON_ACCEPTED_PROVENANCE"
    )

    for path in (REJECTED_DESIGN, REJECTED_AUDIT, DISPOSITION, TASK, LATEST):
        checks[f"sidecar.{path.name}"] = companion_matches(path)

    consuming_paths = [
        ROOT / "research/raw/specification/qwen25-bf16-category-balanced-successor-s6-attempt-operator-authority.json",
        ROOT / "build/bf16-category-balanced-successor-s6/backend-submission-intent.json",
        ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-category-balanced-successor-s6",
        ROOT / "pilot/qwen25_05b_bf16_category_balanced_successor_s6",
        ROOT / "pilot/qwen25_05b_bf16_category_balanced_successor_s6_runner",
        ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-category-balanced-successor-s6",
    ]
    checks["state.known_consuming_paths_absent"] = not any(path.exists() for path in consuming_paths)
    checks["state.no_s6_attempt_marker"] = not any(
        ("successor-s6" in str(path).lower() or "category_balanced" in str(path).lower())
        for path in ROOT.rglob("ATTEMPT_CONSUMPTION_MARKER.json")
    )
    checks["state.no_s6_authority"] = not any(
        ("successor-s6" in str(path).lower() or "category-balanced" in str(path).lower())
        for path in ROOT.rglob("*operator-authority*.json")
    )
    checks["state.no_s6_submission_intent"] = not any(
        ("successor-s6" in str(path).lower() or "category-balanced" in str(path).lower())
        for path in ROOT.rglob("backend-submission-intent.json")
    )

    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "check_count": len(checks),
        "checks": dict(sorted(checks.items())),
        "claim_boundary": "Read-only specification-stage containment audit. PASS preserves rejected S6 v1 as non-accepted provenance and establishes only that the clean-room replacement task is ready with zero local consuming state; it does not freeze a replacement successor or authorize execution/downstream work.",
        "failed_checks": failed,
        "failure_taxonomy": None if not failed else "S6_REPLAN_CONTAINMENT_AUDIT_FAILURE",
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (PIPELINE, REJECTED_DESIGN, REJECTED_AUDIT, DISPOSITION, TASK, SPEC, GROUND_TRUTH, CHECKPOINT, LATEST)
        },
        "legacy_audit_boundary": {
            "repository_wide_audit_status": "FAIL_STALE_PRE_S5_S4_STATUS_EXPECTATION",
            "s5_audit_status": "FAIL_STALE_MARKER_FREE_S5_EXPECTATION",
            "these_legacy_failures_are_not_used_as_s6_acceptance": True,
        },
        "schema_version": 1,
        "status": "PASS_S6_REPLAN_CONTAINMENT_CLEAN_ROOM_TASK_READY_ZERO_CONSUMING_STATE" if not failed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path, help="compare deterministic output to an existing audit JSON")
    args = parser.parse_args()
    result = audit()
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if args.check:
        target = args.check if args.check.is_absolute() else ROOT / args.check
        if not target.is_file() or target.read_bytes() != encoded:
            raise SystemExit("audit artifact is missing or stale")
        if not companion_matches(target):
            raise SystemExit("audit sidecar is missing or stale")
    else:
        print(encoded.decode("utf-8"), end="")
    return 0 if not result["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
