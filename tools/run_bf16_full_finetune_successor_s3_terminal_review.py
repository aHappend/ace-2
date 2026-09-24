#!/usr/bin/env python3
"""Request one independent read-only review of the terminal S3 packet."""

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
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s3-attempt-operator-authority.json"
INTENT = BUILD_ROOT / "backend-submission-intent.json"
CLIENT_RESULT = BUILD_ROOT / "backend-submission-result.json"
CLIENT_RAW = BUILD_ROOT / "backend-submission.raw.txt"
TERMINAL_AUDIT = BUILD_ROOT / "backend-terminal-audit-20260808T122720Z.json"
TERMINAL_ROOT = BUILD_ROOT / "backend-terminal-20260808T122720Z"
REQUEST = BUILD_ROOT / "backend-terminal-fresh-review-request-20260808T122720Z.json"
DECISION = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s3-terminal-fresh-review-20260808.json"
MISSION_ID = "qwen25-bf16-full-finetune-successor-s3-terminal-review-v1"
RULINGS = (
    "s3.cardinality",
    "s3.backend-terminal",
    "s3.download-binding",
    "s3.preattempt-interlock",
    "s3.lifecycle-order",
    "s3.evaluator-no-execution",
    "s3.hygiene-complete",
    "s3.no-retry-downstream",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    return companion.is_file() and companion.read_text(encoding="ascii").split() == [sha256(path), path.name]


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n", encoding="ascii"
    )


def verified_inputs() -> dict[str, Any]:
    require(load(PIPELINE).get("current_stage") == "specification", "current stage is not specification")
    for path in (AUTHORITY, INTENT, CLIENT_RESULT, TERMINAL_AUDIT):
        require(companion_matches(path), f"checksum companion differs: {path}")
    audit = load(TERMINAL_AUDIT)
    require(audit.get("status") == "TERMINAL_S3_PREATTEMPT_NO_GO_EVIDENCE_COMPLETE", "terminal audit status differs")
    require(audit.get("failure_taxonomy") == "PREATTEMPT_SOURCE_SNAPSHOT_AUDIT_OMITTED_FROM_STAGE", "terminal taxonomy differs")
    require(audit.get("check_count") == 64, "terminal audit check count differs")
    require(audit.get("failed_check_count") == 0, "terminal audit has failed checks")
    require(all(audit.get("checks", {}).values()), "terminal audit contains a false check")
    require(audit.get("cardinality") == {
        "authority_count": 1,
        "backend_experiment_count": 1,
        "backend_job_count": 1,
        "intent_count": 1,
        "retry_count": 0,
        "submit_invocation_count": 1,
    }, "terminal cardinality differs")
    require(audit.get("backend", {}).get("status") == "failed", "terminal backend status differs")
    require(audit.get("failure", {}).get("first_failed_gate") == "source_snapshot_exact", "first failed gate differs")
    require(audit.get("failure", {}).get("staged_source_audit_present") is False, "staged source audit unexpectedly exists")
    require(audit.get("lifecycle_order", {}).get("observed") == [
        "backend_setup",
        "non_consuming_launch_readiness",
        "terminal_preattempt_no_go",
    ], "observed lifecycle differs")
    boundaries = audit.get("execution_boundaries", {})
    for key in (
        "detached_launcher_intent_published",
        "attempt_marker_created",
        "training_executed",
        "selector_executed",
        "dev_executed",
        "selected_bf16_materialized",
        "retention_executed",
        "holdout_executed",
        "model_executed",
        "evaluator_executed",
        "checkpoint_created",
        "retry_resume_relaunch_cancel_permitted",
        "downstream_authority_opened",
    ):
        require(boundaries.get(key) is False, f"execution boundary unexpectedly true: {key}")
    require(boundaries.get("quality_conclusion") is None, "quality conclusion exists")
    hygiene = audit.get("hygiene", {})
    require(hygiene.get("status") == "PASS_COMPLETE_NO_SIZE_CAP", "hygiene status differs")
    require(hygiene.get("file_count") == 49, "hygiene file count differs")
    require(hygiene.get("bytes_scanned") == 372533, "hygiene byte count differs")
    require(hygiene.get("complete_byte_coverage") is True, "hygiene coverage is incomplete")
    require(hygiene.get("size_limit_bytes") is None, "hygiene scan used a size limit")
    require(hygiene.get("skipped_file_count") == 0 and not hygiene.get("findings"), "hygiene scan has skips or findings")
    user_log = ROOT / audit["download"]["user_log"]["path"]
    readiness = ROOT / audit["download"]["readiness"]["path"]
    require(sha256(user_log) == audit["download"]["user_log"]["sha256"], "downloaded user log differs")
    require(sha256(readiness) == audit["download"]["readiness"]["sha256"], "downloaded readiness differs")
    return {
        "claim_boundary": "Independent terminal disposition of the sole exactly-once S3 backend job. The requested ruling may accept only evidence completeness and the pre-attempt terminal NO-GO. It must not authorize retry, resubmit, relaunch, resume, repair-in-place, training, evaluator access, BF16 quality, W4A8, RTL, synthesis/PPA, FPGA, U280, stage advance, or project completion.",
        "pipeline_state": artifact(PIPELINE),
        "authority": artifact(AUTHORITY),
        "intent": artifact(INTENT),
        "client_result": artifact(CLIENT_RESULT),
        "client_raw": artifact(CLIENT_RAW),
        "terminal_audit": artifact(TERMINAL_AUDIT),
        "terminal_root": {
            "path": TERMINAL_ROOT.relative_to(ROOT).as_posix(),
            "file_count": audit["download"]["terminal_file_count"],
            "bytes": audit["download"]["terminal_bytes"],
        },
        "downloaded_user_log": artifact(user_log),
        "downloaded_readiness": artifact(readiness),
        "cardinality": audit["cardinality"],
        "backend": audit["backend"],
        "failure": audit["failure"],
        "lifecycle_order": audit["lifecycle_order"],
        "execution_boundaries": boundaries,
        "hygiene": {
            key: hygiene[key]
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
    }


def supports(reason: str, ruling: str) -> bool:
    return f"{ruling}: supported" in reason.lower().replace("`", "")


def check_existing() -> None:
    verified_inputs()
    require(companion_matches(REQUEST), "review request checksum differs")
    require(companion_matches(DECISION), "review decision checksum differs")
    decision = load(DECISION)
    require(decision.get("review", {}).get("status") == "done", "review status is not done")
    reason = str(decision.get("review", {}).get("reason", ""))
    require(all(supports(reason, ruling) for ruling in RULINGS), "review reason lacks exact rulings")


def main() -> int:
    if "--check" in sys.argv[1:]:
        check_existing()
        print("ACE2_S3_TERMINAL_REVIEW_CHECK_PASS")
        return 0

    require(not REQUEST.exists(), "terminal S3 review request already exists")
    require(not DECISION.exists(), "terminal S3 review decision already exists")
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
    runner = AgentCliBackend(
        backend=backend_name,
        runner_bin=resolve_runner_bin_setting("reviewer") or None,
    )
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
        "Independently verify the exact terminal evidence for the sole authorized ACE-2 S3 "
        "BF16 full-finetune backend job and issue only an evidence-completeness terminal disposition."
    )
    instruction = (
        "Inspect the listed project files directly and run only short read-only hash/content checks. "
        "Do not submit, retry, resubmit, relaunch, resume, cancel, repair the stage, train, access an evaluator, "
        "inspect hidden/golden outputs, modify evidence, run W4A8/RTL/simulation/synthesis/PPA, or advance the pipeline. "
        "Verify exact-one cardinality; terminal failed backend status; downloaded log/result hashes; the source_snapshot_exact "
        "pre-attempt interlock caused by the missing staged local-audit file; configured and observed lifecycle order; no detached "
        "intent, attempt marker, training, selector, dev, selected BF16, retention, holdout, model/evaluator execution, checkpoint, "
        "or quality conclusion; and complete no-size-cap hygiene coverage. Return done only if all bindings hold. In the reason "
        "include exactly these rulings: 's3.cardinality: supported', 's3.backend-terminal: supported', "
        "'s3.download-binding: supported', 's3.preattempt-interlock: supported', 's3.lifecycle-order: supported', "
        "'s3.evaluator-no-execution: supported', 's3.hygiene-complete: supported', and 's3.no-retry-downstream: supported'. "
        "A done verdict accepts an evidence-backed terminal NO-GO without retry and authorizes no downstream action."
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
            "Planner's immutable terminal audit reports 64/64 checks, one authority, one intent, one invocation, "
            "one experiment, one terminal failed job, zero retries, a pre-attempt source-audit staging omission, "
            "no model/evaluator execution, and 49 files/372533 bytes of complete no-size-cap hygiene coverage."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="s3_terminal_preattempt_no_go_evidence_only",
        checkpoint_path="",
        escalate_hint="No stage transition, replay, or further backend execution is in scope.",
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
        require(all(supports(reason, ruling) for ruling in RULINGS), "Reviewer reason lacks exact rulings")
        require(not raw.get("next_action"), "Reviewer returned done with a next action")
        require(bool(raw.get("thread_id")), "Reviewer execution identity is missing")
    decision = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "reviewed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "s3_terminal_preattempt_no_go_evidence_only",
        "claim_boundary": verified["claim_boundary"],
        "review_request": artifact(REQUEST),
        "verified_evidence": verified,
        "review": raw,
    }
    write_json(DECISION, decision)
    if raw.get("status") != "done":
        print(f"ACE2_S3_TERMINAL_REVIEW_REPLAN status={raw.get('status')}")
        return 2
    print(f"ACE2_S3_TERMINAL_REVIEW_DONE thread={raw.get('thread_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
