#!/usr/bin/env python3
"""Resolve only the Fresh Review checkpoint blocker for terminal S3 evidence."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import argus_skill
from argus_skill.adapters.agent_cli_backend import AgentCliBackend
from argus_skill.core.knobs import (
    resolve_role_backend,
    resolve_role_model,
    resolve_role_reasoning_effort,
    resolve_runner_bin_setting,
)
from argus_skill.core.paths import global_root
from argus_skill.reviewer import Reviewer, ReviewerConfig


ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = ROOT / "build/bf16-full-finetune-successor-s3"
AUDIT = BUILD_ROOT / "backend-terminal-audit-20260808T122720Z.json"
ORIGINAL_REQUEST = BUILD_ROOT / "backend-terminal-fresh-review-request-20260808T122720Z.json"
BLOCKED_DECISION = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s3-terminal-fresh-review-20260808.json"
LOCAL_CHECKPOINT = ROOT / "CHECKPOINT.md"
SHARED_CHECKPOINT = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/a92f9b84348c/CHECKPOINT.md")
REQUEST = BUILD_ROOT / "backend-terminal-fresh-review-followup-request-20260808T123646Z.json"
DECISION = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s3-terminal-fresh-review-final-20260808.json"
MISSION_ID = "qwen25-bf16-full-finetune-successor-s3-terminal-review-final-v1"
RULINGS = (
    "s3.cardinality",
    "s3.backend-terminal",
    "s3.download-binding",
    "s3.preattempt-interlock",
    "s3.lifecycle-order",
    "s3.evaluator-no-execution",
    "s3.hygiene-complete",
    "s3.no-retry-downstream",
    "s3.checkpoint-current",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    return companion.is_file() and companion.read_text(encoding="ascii").split() == [sha256(path), path.name]


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    try:
        display = path.relative_to(ROOT).as_posix()
    except ValueError:
        display = path.as_posix()
    return {"path": display, "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n", encoding="ascii"
    )


def supports(reason: str, ruling: str) -> bool:
    return f"{ruling}: supported" in reason.lower().replace("`", "")


def verified_inputs() -> dict[str, Any]:
    for path in (AUDIT, ORIGINAL_REQUEST, BLOCKED_DECISION):
        require(companion_matches(path), f"checksum companion differs: {path}")
    audit = load(AUDIT)
    blocked = load(BLOCKED_DECISION)
    require(audit.get("status") == "TERMINAL_S3_PREATTEMPT_NO_GO_EVIDENCE_COMPLETE", "audit status differs")
    require(audit.get("check_count") == 64 and audit.get("failed_check_count") == 0, "audit checks differ")
    review = blocked.get("review", {})
    require(review.get("status") == "blocked", "prior review is not the preserved blocked disposition")
    prior_reason = str(review.get("reason", ""))
    for ruling in RULINGS[:-1]:
        require(supports(prior_reason, ruling), f"prior review lacks supported ruling: {ruling}")
    require("checkpoint.md" in prior_reason.lower() and "read-only" in prior_reason.lower(), "prior blocker differs")
    local_text = LOCAL_CHECKPOINT.read_text(encoding="utf-8")
    shared_text = SHARED_CHECKPOINT.read_text(encoding="utf-8")
    required_fragments = (
        "TERMINAL_S3_PREATTEMPT_NO_GO_EVIDENCE_COMPLETE",
        "source_snapshot_exact",
        "372,533",
        "zero retries",
        "No backend action is permitted",
    )
    require(all(fragment in local_text for fragment in required_fragments), "local checkpoint is not current")
    require(all(fragment in shared_text for fragment in required_fragments), "shared checkpoint is not current")
    require("zero authority" not in local_text and "still reports that sole job as `queued`" not in shared_text, "stale state remains")
    return {
        "claim_boundary": "Final Fresh Review of the already-supported terminal S3 evidence after resolving only the stale-checkpoint blocker. No backend retry, resubmit, relaunch, resume, cancellation, repair, model/evaluator execution, BF16 quality, downstream work, stage advance, or project completion is requested or authorized.",
        "terminal_audit": artifact(AUDIT),
        "original_review_request": artifact(ORIGINAL_REQUEST),
        "blocked_review_decision": artifact(BLOCKED_DECISION),
        "local_checkpoint": artifact(LOCAL_CHECKPOINT),
        "shared_checkpoint": artifact(SHARED_CHECKPOINT),
        "prior_supported_rulings": list(RULINGS[:-1]),
        "audit_summary": {
            "status": audit["status"],
            "failure_taxonomy": audit["failure_taxonomy"],
            "check_count": audit["check_count"],
            "failed_check_count": audit["failed_check_count"],
            "cardinality": audit["cardinality"],
            "backend": audit["backend"],
            "failure": audit["failure"],
            "lifecycle_order": audit["lifecycle_order"],
            "execution_boundaries": audit["execution_boundaries"],
            "hygiene": {
                key: audit["hygiene"][key]
                for key in (
                    "status",
                    "file_count",
                    "bytes_scanned",
                    "complete_byte_coverage",
                    "size_limit_bytes",
                    "skipped_file_count",
                    "findings",
                )
            },
        },
    }


def check_existing() -> None:
    verified_inputs()
    require(companion_matches(REQUEST), "follow-up request checksum differs")
    require(companion_matches(DECISION), "final decision checksum differs")
    decision = load(DECISION)
    review = decision.get("review", {})
    require(review.get("status") == "done", "final Fresh Review status is not done")
    reason = str(review.get("reason", ""))
    require(all(supports(reason, ruling) for ruling in RULINGS), "final review reason lacks exact rulings")
    require(not review.get("next_action"), "final review unexpectedly has a next action")


def main() -> int:
    if "--check" in sys.argv[1:]:
        check_existing()
        print("ACE2_S3_TERMINAL_REVIEW_FINAL_CHECK_PASS")
        return 0

    require(not REQUEST.exists(), "follow-up request already exists")
    require(not DECISION.exists(), "final decision already exists")
    verified = verified_inputs()
    request = {
        "schema_version": 1,
        "status": "PENDING_FRESH_REVIEW",
        "mission_id": MISSION_ID,
        "requested_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "requested_rulings": list(RULINGS),
        "evidence": verified,
    }
    write_json(REQUEST, request)

    backend_name = resolve_role_backend("reviewer")
    runner = AgentCliBackend(backend=backend_name, runner_bin=resolve_runner_bin_setting("reviewer") or None)
    sid = os.environ.get("ARGUS_SKILL_SESSION_ID", ROOT.name)
    runner.set_usage_context(
        project_root=global_root() / "projects" / sid,
        global_root=global_root(),
        mission_id=MISSION_ID,
    )
    package_root = Path(argus_skill.__file__).resolve().parent
    reviewer_skill = (
        package_root / "verticals/chip_design/skills/reviewer/chip-design-signoff-review.md"
    ).read_text(encoding="utf-8")
    objective = (
        "Issue the terminal Fresh Reviewer disposition for the sole S3 backend job after independently "
        "confirming that the previously stale repository and shared checkpoints now match the sealed evidence."
    )
    instruction = (
        "The prior Fresh Reviewer independently supported all eight substantive S3 rulings and blocked only because "
        "CHECKPOINT.md was stale in its read-only sandbox. The orchestrating Engineer has now updated both checkpoint "
        "projections. Do not edit either checkpoint. Independently run short read-only checks of the audit, blocked decision, "
        "and both checkpoint files. Return done only if the terminal evidence still reproduces and the checkpoint blocker is "
        "fully resolved. In the reason include exactly: 's3.cardinality: supported', 's3.backend-terminal: supported', "
        "'s3.download-binding: supported', 's3.preattempt-interlock: supported', 's3.lifecycle-order: supported', "
        "'s3.evaluator-no-execution: supported', 's3.hygiene-complete: supported', 's3.no-retry-downstream: supported', "
        "and 's3.checkpoint-current: supported'. A done verdict accepts only the evidence-backed terminal NO-GO without retry "
        "and must return no next action or downstream authority."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=sid,
        main_summary=(
            "Prior Fresh Review thread 019fe15c-a666-75e1-bd8d-def35ed13fb6 supported all substantive rulings. "
            "Both checkpoint projections now record the terminal failed job, pre-attempt source-audit staging omission, "
            "64/64 audit, complete hygiene, evaluator no-execution, and permanent no-retry boundary."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="s3_terminal_checkpoint_blocker_resolution_only",
        checkpoint_path=str(LOCAL_CHECKPOINT),
        escalate_hint="The checkpoint blocker is resolved; no mutation or further backend action is in scope.",
        preselected_skill_block=reviewer_skill,
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort(
                "ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"
            ),
            skip_git_repo_check=True,
            full_auto=True,
            dangerous_yolo=False,
            sandbox_mode="read-only",
            isolate_workdir=False,
            working_dir=str(ROOT),
        ),
    )
    raw = asdict(review)
    reason = str(raw.get("reason", ""))
    if raw.get("status") == "done":
        require(all(supports(reason, ruling) for ruling in RULINGS), "final Reviewer reason lacks exact rulings")
        require(not raw.get("next_action"), "final Reviewer returned a next action")
        require(bool(raw.get("thread_id")), "final Reviewer identity is missing")
    decision = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "reviewed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "s3_terminal_checkpoint_blocker_resolution_only",
        "claim_boundary": verified["claim_boundary"],
        "review_request": artifact(REQUEST),
        "verified_evidence": verified,
        "review": raw,
    }
    write_json(DECISION, decision)
    if raw.get("status") != "done":
        print(f"ACE2_S3_TERMINAL_REVIEW_FINAL_REPLAN status={raw.get('status')}")
        return 2
    print(f"ACE2_S3_TERMINAL_REVIEW_FINAL_DONE thread={raw.get('thread_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
