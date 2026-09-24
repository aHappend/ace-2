#!/usr/bin/env python3
"""Resolve only the S4 terminal review's checkpoint-index blocker."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import argus_skill
from argus_skill.adapters.agent_cli_backend import AgentCliBackend
from argus_skill.core.knobs import resolve_role_backend, resolve_role_model, resolve_role_reasoning_effort, resolve_runner_bin_setting
from argus_skill.core.paths import global_root
from argus_skill.reviewer import Reviewer, ReviewerConfig


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/bf16-full-finetune-successor-s4"
AUDIT = BUILD / "backend-terminal-audit-v2-20260808T132245Z.json"
ORIGINAL_REQUEST = BUILD / "backend-terminal-fresh-review-request-20260808T132245Z.json"
BLOCKED_DECISION = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s4-terminal-fresh-review-20260808.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
INDEX = ROOT / "latest.json"
REQUEST = BUILD / "backend-terminal-fresh-review-followup-request-20260808T134000Z.json"
DECISION = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s4-terminal-fresh-review-final-20260808.json"
MISSION_ID = "qwen25-bf16-full-finetune-successor-s4-terminal-review-final-v1"
RULINGS = (
    "s4.cardinality", "s4.backend-terminal", "s4.download-binding",
    "s4.preattempt-interlock", "s4.binding-root-cause",
    "s4.evaluator-no-execution", "s4.hygiene-complete",
    "s4.no-retry-downstream", "s4.checkpoint-current",
)


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(detail)


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
    return {"bytes": path.stat().st_size, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def write_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists() and not path.with_suffix(path.suffix + ".sha256").exists(), f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="ascii")


def supports(reason: str, ruling: str) -> bool:
    return f"{ruling}: supported" in reason.lower().replace("`", "")


def verified() -> dict[str, Any]:
    for path in (AUDIT, ORIGINAL_REQUEST, BLOCKED_DECISION, INDEX):
        require(companion_matches(path), f"checksum companion differs: {path}")
    audit = load(AUDIT)
    blocked = load(BLOCKED_DECISION).get("review", {})
    require(blocked.get("status") == "blocked", "preserved first review is not blocked")
    prior_reason = str(blocked.get("reason", ""))
    require(all(supports(prior_reason, ruling) for ruling in RULINGS[:-1]), "first review did not support substantive rulings")
    require("canonical index unavailable" in json.dumps(blocked).lower() or "latest.json" in prior_reason.lower(), "first blocker differs")
    index = load(INDEX)
    require(index.get("status") == "CURRENT_S4_TERMINAL_PREATTEMPT_NO_GO", "local index status differs")
    require(index.get("checkpoint", {}).get("sha256") == sha256(CHECKPOINT), "local index checkpoint binding differs")
    require(index.get("terminal_audit", {}).get("sha256") == sha256(AUDIT), "local index audit binding differs")
    text = CHECKPOINT.read_text(encoding="utf-8")
    for fragment in ("a70f167d", "frozen_bindings_exact", "62/62", "No backend action is", "three runtime-required"):
        require(fragment in text, f"checkpoint lacks current fragment: {fragment}")
    require(audit.get("status") == "TERMINAL_S4_PREATTEMPT_NO_GO_EVIDENCE_COMPLETE", "terminal audit status differs")
    return {
        "claim_boundary": "Final read-only S4 terminal review after resolving only the project-local context-index/checkpoint blocker. No backend action, S4 repair, successor construction, stage transition, or downstream authority is requested.",
        "terminal_audit": artifact(AUDIT),
        "original_request": artifact(ORIGINAL_REQUEST),
        "blocked_decision": artifact(BLOCKED_DECISION),
        "current_checkpoint": artifact(CHECKPOINT),
        "current_context_index": artifact(INDEX),
        "prior_supported_rulings": list(RULINGS[:-1]),
        "terminal_summary": {
            "status": audit["status"],
            "failure_taxonomy": audit["failure_taxonomy"],
            "check_count": audit["check_count"],
            "failed_check_count": audit["failed_check_count"],
            "cardinality": audit["cardinality"],
            "backend": audit["backend"],
            "failure": audit["failure"],
            "execution_boundaries": audit["execution_boundaries"],
        },
    }


def check_existing() -> None:
    # The final review is immutable historical evidence. Its request captured
    # the then-current checkpoint and context-index hashes; those live routing
    # files legitimately move on after S4. Revalidate the sealed review chain
    # and immutable S4 artifacts without requiring today's latest.json or
    # CHECKPOINT.md to reproduce the historical S4 routing snapshot.
    for path in (AUDIT, ORIGINAL_REQUEST, BLOCKED_DECISION, REQUEST, DECISION):
        require(companion_matches(path), f"checksum companion differs: {path}")
    request = load(REQUEST)
    decision = load(DECISION)
    evidence = request.get("evidence", {})
    require(decision.get("verified_evidence") == evidence, "review evidence differs from sealed request")
    require(
        decision.get("review_request", {}).get("sha256") == sha256(REQUEST),
        "final decision does not bind the sealed follow-up request",
    )
    for key, path in (
        ("terminal_audit", AUDIT),
        ("original_request", ORIGINAL_REQUEST),
        ("blocked_decision", BLOCKED_DECISION),
    ):
        record = evidence.get(key, {})
        require(record.get("path") == path.relative_to(ROOT).as_posix(), f"{key} path differs")
        require(record.get("sha256") == sha256(path), f"{key} hash differs")
    for key, expected_path in (
        ("current_checkpoint", "CHECKPOINT.md"),
        ("current_context_index", "latest.json"),
    ):
        record = evidence.get(key, {})
        require(record.get("path") == expected_path, f"historical {key} path differs")
        require(
            isinstance(record.get("sha256"), str)
            and len(record["sha256"]) == 64
            and all(char in "0123456789abcdef" for char in record["sha256"]),
            f"historical {key} digest is malformed",
        )
    audit = load(AUDIT)
    summary = evidence.get("terminal_summary", {})
    for key in (
        "status",
        "failure_taxonomy",
        "check_count",
        "failed_check_count",
        "cardinality",
        "backend",
        "failure",
        "execution_boundaries",
    ):
        require(summary.get(key) == audit.get(key), f"sealed terminal summary differs: {key}")
    review = decision.get("review", {})
    require(review.get("status") == "done", "final review status is not done")
    reason = str(review.get("reason", ""))
    require(all(supports(reason, ruling) for ruling in RULINGS), "final review lacks required rulings")


def main() -> int:
    if "--check" in sys.argv[1:]:
        check_existing()
        print("ACE2_S4_TERMINAL_REVIEW_FINAL_CHECK_PASS")
        return 0
    require(not REQUEST.exists() and not DECISION.exists(), "follow-up review already exists")
    evidence = verified()
    write_json(REQUEST, {
        "schema_version": 1, "status": "PENDING_FRESH_REVIEW", "mission_id": MISSION_ID,
        "requested_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "requested_rulings": list(RULINGS), "evidence": evidence,
    })
    backend_name = resolve_role_backend("reviewer")
    runner = AgentCliBackend(backend=backend_name, runner_bin=resolve_runner_bin_setting("reviewer") or None)
    sid = os.environ.get("ARGUS_SKILL_SESSION_ID", ROOT.name)
    runner.set_usage_context(project_root=global_root() / "projects" / sid, global_root=global_root(), mission_id=MISSION_ID)
    skill = (Path(argus_skill.__file__).resolve().parent / "verticals/chip_design/skills/reviewer/chip-design-signoff-review.md").read_text(encoding="utf-8")
    objective = "Issue the final independent terminal disposition for the sole consumed ACE-2 S4 lifecycle."
    instruction = (
        "The first read-only review already supported all eight substantive S4 rulings and blocked only because root latest.json was absent "
        "and it could not write CHECKPOINT.md. Both are now present, writable by the Planner, hash-bound, and current; no Reviewer write is "
        "needed or permitted. Re-open the audit, index, checkpoint, and prior blocked decision read-only. Do not execute or repair S4, create a "
        "successor, or open downstream work. Return done only if the index binds the current checkpoint and 62/62 audit and terminal containment "
        "still holds. Include exactly: " + ", ".join(f"'{r}: supported'" for r in RULINGS) + ". "
        "A done verdict is TERMINAL_NO_GO with no retry, successor authority, or downstream authority."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective, original_objective=objective,
        operator_messages=[instruction], planner_review_instruction=instruction,
        round_index=1, round_max=1, session_id=sid,
        main_summary="Prior review supported all terminal evidence and withheld only checkpoint-current. The project-local latest.json now binds current CHECKPOINT.md and the immutable 62/62 audit.",
        main_error=None, raw_evidence=json.dumps(evidence, indent=2, sort_keys=True),
        scope="s4_terminal_preattempt_no_go_final_checkpoint_resolution_only",
        checkpoint_path="", escalate_hint="No writes or backend actions are needed.",
        preselected_skill_block=skill,
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort("ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"),
            skip_git_repo_check=True, full_auto=True, dangerous_yolo=False,
            sandbox_mode="read-only", isolate_workdir=False, working_dir=str(ROOT),
        ),
    )
    raw = asdict(review)
    reason = str(raw.get("reason", ""))
    if raw.get("status") == "done":
        require(all(supports(reason, ruling) for ruling in RULINGS), "Reviewer reason lacks final rulings")
        require(not raw.get("next_action"), "Reviewer returned done with next action")
    write_json(DECISION, {
        "schema_version": 1, "mission_id": MISSION_ID,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "s4_terminal_preattempt_no_go_final_checkpoint_resolution_only",
        "claim_boundary": evidence["claim_boundary"], "review_request": artifact(REQUEST),
        "verified_evidence": evidence, "review": raw,
    })
    if raw.get("status") != "done":
        print(f"ACE2_S4_TERMINAL_REVIEW_FINAL_REPLAN status={raw.get('status')}")
        return 2
    print(f"ACE2_S4_TERMINAL_REVIEW_FINAL_DONE thread={raw.get('thread_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
