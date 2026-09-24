#!/usr/bin/env python3
"""Request/check one independent read-only review of terminal S4 evidence."""

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
BUILD = ROOT / "build/bf16-full-finetune-successor-s4"
AUDIT = BUILD / "backend-terminal-audit-v2-20260808T132245Z.json"
REQUEST = BUILD / "backend-terminal-fresh-review-request-20260808T132245Z.json"
DECISION = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s4-terminal-fresh-review-20260808.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
MISSION_ID = "qwen25-bf16-full-finetune-successor-s4-terminal-review-v1"
RULINGS = (
    "s4.cardinality",
    "s4.backend-terminal",
    "s4.download-binding",
    "s4.preattempt-interlock",
    "s4.binding-root-cause",
    "s4.evaluator-no-execution",
    "s4.hygiene-complete",
    "s4.no-retry-downstream",
    "s4.checkpoint-current",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="ascii")


def supports(reason: str, ruling: str) -> bool:
    return f"{ruling}: supported" in reason.lower().replace("`", "")


def verified_inputs() -> dict[str, Any]:
    require(load(PIPELINE).get("current_stage") == "specification", "pipeline stage differs")
    require(companion_matches(AUDIT), "terminal audit checksum differs")
    audit = load(AUDIT)
    require(audit.get("status") == "TERMINAL_S4_PREATTEMPT_NO_GO_EVIDENCE_COMPLETE", "audit status differs")
    require(audit.get("failure_taxonomy") == "PREATTEMPT_RUNTIME_CHECKSUM_SIDECARS_OMITTED_FROM_STAGE", "failure taxonomy differs")
    require(audit.get("check_count") == 62 and audit.get("failed_check_count") == 0, "audit check totals differ")
    require(all(audit.get("checks", {}).values()), "audit has a false check")
    require(audit.get("cardinality") == {
        "authority_count": 1, "intent_count": 1, "submit_invocation_count": 1,
        "backend_experiment_count": 1, "backend_job_count": 1, "retry_count": 0,
    }, "cardinality differs")
    require(audit.get("backend", {}).get("job_id") == "ace2-bf16-full-finetune-successor-s4-20260808-a70f167d", "job identity differs")
    require(audit.get("backend", {}).get("status") == "failed", "backend status differs")
    failure = audit.get("failure", {})
    require(failure.get("first_failed_gate") == "frozen_bindings_exact", "first failed gate differs")
    require(len(failure.get("runtime_required_missing_sidecars", [])) == 3, "causal sidecar count differs")
    require(failure.get("noncausal_reason") == "independent_execution_package_l2 passed and runtime.py has no L2-sidecar companion check", "L2 sidecar classification differs")
    boundaries = audit.get("execution_boundaries", {})
    for key in (
        "detached_launcher_intent_published", "attempt_marker_created", "worker_executed",
        "training_executed", "selector_executed", "dev_executed",
        "selected_bf16_materialized", "retention_executed", "holdout_executed",
        "model_executed", "evaluator_executed", "checkpoint_created",
        "retry_resume_relaunch_cancel_permitted", "downstream_authority_opened",
    ):
        require(boundaries.get(key) is False, f"execution boundary unexpectedly true: {key}")
    require(boundaries.get("quality_conclusion") is None, "quality conclusion exists")
    hygiene = audit.get("hygiene", {})
    require(hygiene.get("status") == "PASS_COMPLETE_NO_SIZE_CAP", "hygiene status differs")
    require(hygiene.get("complete_byte_coverage") is True, "hygiene coverage differs")
    require(hygiene.get("size_limit_bytes") is None, "hygiene used a size limit")
    require(hygiene.get("skipped_file_count") == 0 and not hygiene.get("findings"), "hygiene skips/findings exist")
    checkpoint = CHECKPOINT.read_text(encoding="utf-8")
    for fragment in ("a70f167d", "frozen_bindings_exact", "62/62", "No backend action is", "L2 acceptance companion is also"):
        require(fragment in checkpoint, f"checkpoint lacks current fragment: {fragment}")
    return {
        "claim_boundary": "Independent terminal disposition of the sole consumed S4 backend lifecycle. Acceptance is limited to evidence completeness, exact pre-attempt binding failure, no model/evaluator execution, and permanent no-retry containment; it grants no successor authority or downstream work.",
        "pipeline_state": artifact(PIPELINE),
        "checkpoint": artifact(CHECKPOINT),
        "terminal_audit": artifact(AUDIT),
        "cardinality": audit["cardinality"],
        "backend": audit["backend"],
        "failure": failure,
        "execution_boundaries": boundaries,
        "hygiene": {key: hygiene[key] for key in ("status", "file_count", "bytes_scanned", "complete_byte_coverage", "size_limit_bytes", "skipped_file_count", "findings")},
        "terminal_evidence": audit["terminal_evidence"],
    }


def check_existing() -> None:
    verified_inputs()
    require(companion_matches(REQUEST), "review request checksum differs")
    require(companion_matches(DECISION), "review decision checksum differs")
    review = load(DECISION).get("review", {})
    require(review.get("status") == "done", "review status is not done")
    reason = str(review.get("reason", ""))
    require(all(supports(reason, ruling) for ruling in RULINGS), "review reason lacks required rulings")


def main() -> int:
    if "--check" in sys.argv[1:]:
        check_existing()
        print("ACE2_S4_TERMINAL_REVIEW_CHECK_PASS")
        return 0

    require(not REQUEST.exists() and not DECISION.exists(), "terminal S4 review already exists")
    verified = verified_inputs()
    write_json(REQUEST, {
        "schema_version": 1,
        "status": "PENDING_FRESH_REVIEW",
        "mission_id": MISSION_ID,
        "requested_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "requested_rulings": list(RULINGS),
        "evidence": verified,
    })

    backend_name = resolve_role_backend("reviewer")
    runner = AgentCliBackend(backend=backend_name, runner_bin=resolve_runner_bin_setting("reviewer") or None)
    sid = os.environ.get("ARGUS_SKILL_SESSION_ID", ROOT.name)
    runner.set_usage_context(project_root=global_root() / "projects" / sid, global_root=global_root(), mission_id=MISSION_ID)
    reviewer_skill = (Path(argus_skill.__file__).resolve().parent / "verticals/chip_design/skills/reviewer/chip-design-signoff-review.md").read_text(encoding="utf-8")
    objective = "Independently verify and terminally contain the sole authorized ACE-2 S4 BF16 backend lifecycle."
    instruction = (
        "Inspect the listed project files directly and run only short read-only hash/content checks. "
        "Do not submit, retry, resubmit, relaunch, resume, cancel, repair S4, create a successor, train, access an evaluator, "
        "inspect hidden/golden outputs, modify evidence, run W4A8/RTL/simulation/synthesis/PPA, or advance the pipeline. "
        "Verify one authority/intent/invocation/experiment/job and zero retries; terminal Failed job a70f167d; downloaded hashes; "
        "PREATTEMPT_NO_GO at frozen_bindings_exact; the exact three causal runtime sidecars (contract, manifest, self-test); "
        "the absent L2 companion is non-causal because L2 passed and runtime has no L2-sidecar check; no detached intent, marker, "
        "worker, training, selector, dev, retention, holdout, model/evaluator execution, checkpoint, or quality conclusion; complete "
        "no-size-cap hygiene; and current checkpoint containment. Return done only if all hold. Include exactly these rulings: "
        + ", ".join(f"'{r}: supported'" for r in RULINGS)
        + ". A done verdict is TERMINAL_NO_GO only and grants no retry, successor authority, or downstream action."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=sid,
        main_summary="The immutable v2 audit passes 62/62 checks and reports one consumed S4 job terminal failed before attempt/worker/model/evaluator execution due three omitted runtime-required checksum companions.",
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="s4_terminal_preattempt_no_go_evidence_only",
        checkpoint_path=str(CHECKPOINT),
        escalate_hint="No backend execution, repair, successor construction, or stage transition is in scope.",
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
        require(all(supports(reason, ruling) for ruling in RULINGS), "Reviewer reason lacks exact rulings")
        require(not raw.get("next_action"), "Reviewer returned done with next action")
        require(bool(raw.get("thread_id")), "Reviewer execution identity is missing")
    write_json(DECISION, {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "reviewed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "s4_terminal_preattempt_no_go_evidence_only",
        "claim_boundary": verified["claim_boundary"],
        "review_request": artifact(REQUEST),
        "verified_evidence": verified,
        "review": raw,
    })
    if raw.get("status") != "done":
        print(f"ACE2_S4_TERMINAL_REVIEW_REPLAN status={raw.get('status')}")
        return 2
    print(f"ACE2_S4_TERMINAL_REVIEW_DONE thread={raw.get('thread_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
