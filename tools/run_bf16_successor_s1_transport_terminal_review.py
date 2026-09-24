#!/usr/bin/env python3
"""Run one independent read-only review of the terminal S1 transport packet."""

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
from argus_skill.core.knobs import (
    resolve_role_backend,
    resolve_role_model,
    resolve_role_reasoning_effort,
    resolve_runner_bin_setting,
)
from argus_skill.core.paths import global_root
from argus_skill.reviewer import Reviewer, ReviewerConfig


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-successor-s1-transport-attempt-operator-authority.json"
INTENT = ROOT / "build/bf16-successor-s1-transport/backend-submission-intent.json"
CLIENT_RESULT = ROOT / "build/bf16-successor-s1-transport/backend-submission-result.json"
CLIENT_RAW = ROOT / "build/bf16-successor-s1-transport/backend-submission.raw.txt"
TERMINAL_AUDIT = ROOT / "build/bf16-successor-s1-transport/backend-terminal-audit-20260808T111251Z.json"
DOWNLOADED_STDOUT = ROOT / "build/bf16-successor-s1-transport/backend-terminal-download-20260808T111251Z/logs/ace2-bf16-successor-s1-transport-20260808-preauthority/ace2-bf16-successor-s1-transport-gate/stdout.txt"
REQUEST = ROOT / "build/bf16-successor-s1-transport/backend-terminal-fresh-review-request-20260808T111251Z.json"
DECISION = ROOT / "research/raw/specification/bf16-successor-s1-transport-terminal-backend-fresh-review-20260808.json"
MISSION_ID = "s1-transport-terminal-backend-evidence-review-v1"
RULINGS = (
    "s1.cardinality",
    "s1.client-misclassification",
    "s1.backend-terminal",
    "s1.download-binding",
    "s1.no-model-evaluator",
    "s1.no-retry-downstream",
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
    require(audit.get("status") == "TERMINAL_S1_TRANSPORT_EVIDENCE_COMPLETE", "terminal audit status differs")
    require(audit.get("check_count") == 34, "terminal audit check count differs")
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
    require(audit.get("backend", {}).get("job_id") == "ace2-bf16-successor-s1-transport-20260808-pre-0e51f05b", "terminal job id differs")
    require(audit.get("backend", {}).get("status") == "failed", "terminal backend status differs")
    require(audit.get("execution_boundaries", {}).get("transport_payload_executed") is True, "transport payload did not execute")
    require(audit.get("execution_boundaries", {}).get("training_executed") is False, "training was reported executed")
    require(audit.get("execution_boundaries", {}).get("model_executed") is False, "model was reported executed")
    require(audit.get("execution_boundaries", {}).get("evaluator_accessed") is False, "evaluator was reported accessed")
    require(audit.get("execution_boundaries", {}).get("quality_conclusion") is None, "quality conclusion exists")
    require(audit.get("download", {}).get("stdout", {}).get("sha256") == sha256(DOWNLOADED_STDOUT), "downloaded stdout binding differs")
    return {
        "claim_boundary": "Independent disposition of the sole terminal S1 transport-only backend job. No replay, retry, resume, cancel, training, model/evaluator access, quality, W4A8, RTL, synthesis/PPA, FPGA, U280, stage advance, or project-completion authority is requested.",
        "pipeline_state": artifact(PIPELINE),
        "authority": artifact(AUTHORITY),
        "intent": artifact(INTENT),
        "client_result": artifact(CLIENT_RESULT),
        "client_raw": artifact(CLIENT_RAW),
        "terminal_audit": artifact(TERMINAL_AUDIT),
        "downloaded_stdout": artifact(DOWNLOADED_STDOUT),
        "cardinality": audit["cardinality"],
        "backend": audit["backend"],
        "client_result_disposition": audit["client_result"],
        "execution_boundaries": audit["execution_boundaries"],
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
        print("ACE2_S1_TRANSPORT_TERMINAL_REVIEW_CHECK_PASS")
        return 0

    require(not REQUEST.exists(), "terminal S1 review request already exists")
    require(not DECISION.exists(), "terminal S1 review decision already exists")
    verified = verified_inputs()
    request = {
        "schema_version": 1,
        "status": "PENDING_FRESH_REVIEW",
        "mission_id": MISSION_ID,
        "requested_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
        "Independently verify the exact terminal evidence for the sole authorized ACE-2 S1 "
        "transport-only AMLT job. Decide only whether one authority, one intent, one invocation, "
        "one experiment, one job, zero retries, the preserved client misclassification, terminal "
        "backend status, and downloaded output hash are supported."
    )
    instruction = (
        "Inspect the listed project files directly and run only short read-only hash/content checks. "
        "Do not submit, retry, resume, cancel, train, access an evaluator, inspect hidden/golden outputs, "
        "modify evidence, run RTL/simulation/synthesis/PPA, or advance the pipeline. The backend job's "
        "failed state is expected transport evidence because the frozen inert payload returns 97; it is "
        "not a model-quality failure. Return done only if all bindings and claim boundaries hold. In the "
        "reason include exactly these rulings: 's1.cardinality: supported', "
        "'s1.client-misclassification: supported', 's1.backend-terminal: supported', "
        "'s1.download-binding: supported', 's1.no-model-evaluator: supported', and "
        "'s1.no-retry-downstream: supported'. A done verdict authorizes no further backend or downstream action."
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
            "Planner's immutable terminal audit reports 34/34 checks, one authority, one intent, "
            "one invocation, one experiment, one terminal failed job, zero retries, and a downloaded "
            "stdout SHA-256. The Reviewer must independently verify these claims."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="s1_transport_terminal_backend_evidence_only",
        checkpoint_path="",
        escalate_hint="No stage transition or further backend execution is in scope.",
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
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "s1_transport_terminal_backend_evidence_only",
        "claim_boundary": verified["claim_boundary"],
        "review_request": artifact(REQUEST),
        "verified_evidence": verified,
        "review": raw,
    }
    write_json(DECISION, decision)
    if raw.get("status") != "done":
        print(f"ACE2_S1_TRANSPORT_TERMINAL_REVIEW_REPLAN status={raw.get('status')}")
        return 2
    print(f"ACE2_S1_TRANSPORT_TERMINAL_REVIEW_DONE thread={raw.get('thread_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
