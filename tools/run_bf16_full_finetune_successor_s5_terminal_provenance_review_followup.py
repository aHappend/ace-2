#!/usr/bin/env python3
"""Resolve only the S5 terminal provenance review checkpoint-write blocker."""

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
from argus_skill.core.knobs import resolve_role_backend, resolve_role_model, resolve_role_reasoning_effort, resolve_runner_bin_setting
from argus_skill.core.paths import global_root
from argus_skill.reviewer import Reviewer, ReviewerConfig


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/bf16-full-finetune-successor-s5"
TERMINAL_AUDIT = BUILD / "backend-terminal-audit-20260808T150208Z.json"
PROVENANCE_AUDIT = BUILD / "submission-client-provenance-audit-20260808T151648Z.json"
ORIGINAL_REQUEST = BUILD / "backend-terminal-provenance-fresh-review-request-20260808T151900Z.json"
BLOCKED_DECISION = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-terminal-provenance-fresh-review-20260808.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
INDEX = ROOT / "latest.json"
REQUEST = BUILD / "backend-terminal-provenance-fresh-review-followup-request-20260808T152700Z.json"
DECISION = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-terminal-provenance-fresh-review-final-20260808.json"
MISSION_ID = "qwen25-bf16-full-finetune-successor-s5-terminal-provenance-review-final-v1"
ARGUS_PROJECT = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b")
RULINGS = (
    "s5.terminal-cardinality",
    "s5.dev-quality-no-go",
    "s5.client-provenance-deviation",
    "s5.transport-equivalence-not-established",
    "s5.no-retry-downstream",
    "s5.stage-remains-specification",
    "s5.checkpoint-current",
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
    require(not path.exists() and not path.with_suffix(path.suffix + ".sha256").exists(), f"refusing to overwrite immutable review artifact: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="ascii")


def supports(reason: str, ruling: str) -> bool:
    return f"{ruling}: supported" in reason.lower().replace("`", "")


def verified() -> dict[str, Any]:
    for path in (TERMINAL_AUDIT, PROVENANCE_AUDIT, ORIGINAL_REQUEST, BLOCKED_DECISION, INDEX):
        require(companion_matches(path), f"checksum companion differs: {path}")
    blocked = load(BLOCKED_DECISION).get("review", {})
    require(blocked.get("status") == "blocked", "first review is not preserved as blocked")
    prior_reason = str(blocked.get("reason", ""))
    require(all(supports(prior_reason, ruling) for ruling in RULINGS[:-1]), "first review did not support all substantive rulings")
    require("checkpoint.md edit was rejected" in prior_reason.lower(), "first review blocker differs")
    index = load(INDEX)
    require(index.get("status") == "CURRENT_S5_TERMINAL_DEV_QUALITY_NO_GO_PROVENANCE_RECORDED", "local index status differs")
    require(index.get("checkpoint", {}).get("sha256") == sha256(CHECKPOINT), "local index checkpoint binding differs")
    require(index.get("s5_terminal_audit", {}).get("sha256") == sha256(TERMINAL_AUDIT), "local index terminal-audit binding differs")
    require(index.get("s5_client_provenance_audit", {}).get("sha256") == sha256(PROVENANCE_AUDIT), "local index provenance-audit binding differs")
    text = CHECKPOINT.read_text(encoding="utf-8")
    for fragment in ("bced4736", "27/56", "48", "DEV_QUALITY_GATE_FAILURE", "91791a9e", "29c4b23b", "equivalence is not established", "60/60", "22/22", "019fe1f7-6723-7c70-bef9-bf31b7e8df87"):
        require(fragment.lower() in text.lower(), f"checkpoint lacks current fragment: {fragment}")
    return {
        "claim_boundary": "Final read-only S5 terminal/provenance review after resolving only the project-local CHECKPOINT.md recording blocker. No backend action, evidence mutation, S5 replay or repair, successor construction, score change, stage transition, or downstream authority is requested.",
        "terminal_audit": artifact(TERMINAL_AUDIT),
        "provenance_audit": artifact(PROVENANCE_AUDIT),
        "original_request": artifact(ORIGINAL_REQUEST),
        "blocked_decision": artifact(BLOCKED_DECISION),
        "current_checkpoint": artifact(CHECKPOINT),
        "current_context_index": artifact(INDEX),
        "prior_supported_rulings": list(RULINGS[:-1]),
    }


def check_existing() -> None:
    verified()
    require(companion_matches(REQUEST) and companion_matches(DECISION), "follow-up review binding differs")
    review = load(DECISION).get("review", {})
    require(review.get("status") == "done", "final review status is not done")
    ruling_text = str(review.get("ruling_text", review.get("reason", "")))
    require(all(supports(ruling_text, ruling) for ruling in RULINGS), "final review lacks required rulings")


def recover_completed_review() -> None:
    require(REQUEST.exists() and companion_matches(REQUEST), "follow-up request is absent or differs")
    require(not DECISION.exists() and not DECISION.with_suffix(DECISION.suffix + ".sha256").exists(), "final review decision already exists")
    usage_rows = [json.loads(line) for line in (ARGUS_PROJECT / "usage.jsonl").read_text(encoding="utf-8").splitlines()]
    matches = [row for row in usage_rows if row.get("mission_id") == MISSION_ID and row.get("run_label") == "reviewer"]
    require(len(matches) == 1, f"expected one durable reviewer usage row, observed {len(matches)}")
    usage = matches[0]
    require(usage.get("status") == "completed" and not usage.get("error"), "durable reviewer run did not complete cleanly")
    call_id = usage.get("call_id")
    stream_rows = [json.loads(line) for line in (ARGUS_PROJECT / "agent_io.jsonl").read_text(encoding="utf-8").splitlines()]
    final_messages: list[str] = []
    for row in stream_rows:
        if row.get("call_id") != call_id or row.get("io_kind") != "stream" or row.get("stream") != "stdout":
            continue
        payload = json.loads(row.get("line", "{}"))
        item = payload.get("item", {})
        if payload.get("type") == "item.completed" and item.get("type") == "agent_message":
            final_messages.append(str(item.get("text", "")))
    completed_messages = [
        message for message in final_messages
        if "STATUS=done" in message and all(supports(message, ruling) for ruling in RULINGS)
    ]
    require(len(completed_messages) == 1, f"expected one durable completed reviewer message, observed {len(completed_messages)}")
    message = completed_messages[0]
    require("STATUS=done" in message, "durable reviewer final status is not done")
    require(all(supports(message, ruling) for ruling in RULINGS), "durable reviewer final message lacks required rulings")
    reason = next((line.removeprefix("REASON=") for line in message.splitlines() if line.startswith("REASON=")), "")
    require(reason, "durable reviewer reason is absent")
    evidence = verified()
    write_json(DECISION, {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "reviewed_at_utc": datetime.fromtimestamp(float(usage["completed_at"]), UTC).isoformat().replace("+00:00", "Z"),
        "scope": "s5_terminal_quality_no_go_provenance_final_checkpoint_resolution_only",
        "claim_boundary": evidence["claim_boundary"],
        "review_request": artifact(REQUEST),
        "verified_evidence": evidence,
        "review": {
            "status": "done",
            "reason": reason,
            "ruling_text": message,
            "thread_id": usage.get("thread_id"),
            "call_id": call_id,
            "review_source": "reviewer_recovered_from_durable_argus_agent_io",
            "backend": usage.get("provider"),
            "model": usage.get("model"),
            "input_tokens": usage.get("input_tokens"),
            "cached_input_tokens": usage.get("cached_input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
            "next_action": "",
            "operator_question": "none",
            "checkpoint_recommended": False,
            "recovery_note": "The Reviewer completed with STATUS=done and all seven rulings in its durable final agent message. The original wrapper rejected the parsed REASON field before writing a decision; no Reviewer rerun was performed.",
        },
    })
    print(f"ACE2_S5_TERMINAL_PROVENANCE_REVIEW_FINAL_RECOVERED thread={usage.get('thread_id')}")


def main() -> int:
    if "--recover" in sys.argv[1:]:
        recover_completed_review()
        return 0
    if "--check" in sys.argv[1:]:
        check_existing()
        print("ACE2_S5_TERMINAL_PROVENANCE_REVIEW_FINAL_CHECK_PASS")
        return 0
    require(not REQUEST.exists() and not DECISION.exists(), "S5 terminal provenance follow-up review already exists")
    evidence = verified()
    write_json(REQUEST, {
        "schema_version": 1,
        "status": "PENDING_FRESH_REVIEW",
        "mission_id": MISSION_ID,
        "requested_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "requested_rulings": list(RULINGS),
        "evidence": evidence,
    })
    backend_name = resolve_role_backend("reviewer")
    runner = AgentCliBackend(backend=backend_name, runner_bin=resolve_runner_bin_setting("reviewer") or None)
    sid = os.environ.get("ARGUS_SKILL_SESSION_ID", ROOT.name)
    runner.set_usage_context(project_root=global_root() / "projects" / sid, global_root=global_root(), mission_id=MISSION_ID)
    reviewer_skill = (Path(argus_skill.__file__).resolve().parent / "verticals/chip_design/skills/reviewer/chip-design-signoff-review.md").read_text(encoding="utf-8")
    objective = "Issue the final independent terminal disposition for the sole consumed ACE-2 S5 lifecycle and AMLT client provenance deviation."
    instruction = (
        "The first read-only review explicitly supported all six substantive S5 rulings and blocked only because its sandbox could not write CHECKPOINT.md. The Planner has now recorded the exact bounded finding and created a hash-bound local index; no Reviewer write is needed or permitted. Re-open the two audits, first decision, checkpoint, and index read-only. Do not submit, retry, rescore, inspect hidden/golden case content, create a successor, run W4A8/RTL/synthesis/PPA/FPGA/U280, or advance the stage. Return done only if the index binds the current checkpoint and both audits and the first review's six substantive rulings still hold. Include exactly: "
        + ", ".join(f"'{r}: supported'" for r in RULINGS)
        + ". A done verdict is TERMINAL_NO_GO with a bounded submission-client provenance deviation, transport equivalence not established, and no retry or downstream authority."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=sid,
        main_summary="The first Fresh Reviewer supported all substantive terminal and provenance rulings. CHECKPOINT.md and latest.json now hash-bind the 60/60 terminal audit and 22/22 provenance audit.",
        main_error=None,
        raw_evidence=json.dumps(evidence, indent=2, sort_keys=True),
        scope="s5_terminal_quality_no_go_provenance_final_checkpoint_resolution_only",
        checkpoint_path="",
        escalate_hint="No writes or backend actions are needed.",
        preselected_skill_block=reviewer_skill,
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort("ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"),
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
        require(all(supports(reason, ruling) for ruling in RULINGS), "Reviewer reason lacks final rulings")
        require(not raw.get("next_action"), "Reviewer returned done with next action")
    write_json(DECISION, {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "reviewed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "s5_terminal_quality_no_go_provenance_final_checkpoint_resolution_only",
        "claim_boundary": evidence["claim_boundary"],
        "review_request": artifact(REQUEST),
        "verified_evidence": evidence,
        "review": raw,
    })
    if raw.get("status") != "done":
        print(f"ACE2_S5_TERMINAL_PROVENANCE_REVIEW_FINAL_REPLAN status={raw.get('status')}")
        return 2
    print(f"ACE2_S5_TERMINAL_PROVENANCE_REVIEW_FINAL_DONE thread={raw.get('thread_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
