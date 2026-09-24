#!/usr/bin/env python3
"""Request/check one read-only Fresh Reviewer ruling on S5 terminal provenance."""

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
BUILD = ROOT / "build/bf16-full-finetune-successor-s5"
TERMINAL_AUDIT = BUILD / "backend-terminal-audit-20260808T150208Z.json"
PROVENANCE_AUDIT = BUILD / "submission-client-provenance-audit-20260808T151648Z.json"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-attempt-operator-authority.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
REQUEST = BUILD / "backend-terminal-provenance-fresh-review-request-20260808T151900Z.json"
DECISION = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-terminal-provenance-fresh-review-20260808.json"
MISSION_ID = "qwen25-bf16-full-finetune-successor-s5-terminal-provenance-review-v1"
RULINGS = (
    "s5.terminal-cardinality",
    "s5.dev-quality-no-go",
    "s5.client-provenance-deviation",
    "s5.transport-equivalence-not-established",
    "s5.no-retry-downstream",
    "s5.stage-remains-specification",
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
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists() and not path.with_suffix(path.suffix + ".sha256").exists(), f"refusing to overwrite immutable review artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="ascii")


def supports(reason: str, ruling: str) -> bool:
    return f"{ruling}: supported" in reason.lower().replace("`", "")


def verified_inputs() -> dict[str, Any]:
    for path in (TERMINAL_AUDIT, PROVENANCE_AUDIT, AUTHORITY):
        require(companion_matches(path), f"checksum companion differs: {path}")
    terminal = load(TERMINAL_AUDIT)
    provenance = load(PROVENANCE_AUDIT)
    pipeline = load(PIPELINE)
    require(terminal.get("status") == "TERMINAL_S5_DEV_QUALITY_NO_GO_EVIDENCE_COMPLETE", "terminal audit status differs")
    require(terminal.get("failure_taxonomy") == "DEV_QUALITY_GATE_FAILURE", "terminal failure taxonomy differs")
    require(terminal.get("check_count") == 60 and terminal.get("failed_check_count") == 0 and all(terminal.get("checks", {}).values()), "terminal audit checks differ")
    require(terminal.get("cardinality") == {
        "authority_count": 1,
        "backend_experiment_count": 1,
        "backend_job_count": 1,
        "intent_count": 1,
        "retry_count": 0,
        "submit_invocation_count": 1,
    }, "terminal cardinality differs")
    require(terminal.get("quality", {}).get("dev", {}).get("hard_pass_count") == 27, "dev hard-pass count differs")
    require(terminal.get("failure", {}).get("dev_minimum_hard_passes") == 48, "dev threshold differs")
    require(terminal.get("execution_boundaries", {}).get("retention_executed") is False, "retention unexpectedly executed")
    require(terminal.get("execution_boundaries", {}).get("holdout_executed") is False, "holdout unexpectedly executed")
    require(provenance.get("status") == "PASS_BOUNDED_SUBMISSION_CLIENT_PROVENANCE_DEVIATION", "provenance audit status differs")
    require(provenance.get("classification") == "BOUNDED_SUBMISSION_CLIENT_PROVENANCE_DEVIATION", "provenance classification differs")
    require(provenance.get("check_count") == 22 and provenance.get("failed_check_count") == 0 and all(provenance.get("checks", {}).values()), "provenance audit checks differ")
    require(provenance.get("classification_limits", {}).get("canonical_executable_used_for_submission") is False, "canonical executable use is misstated")
    require(provenance.get("classification_limits", {}).get("transport_equivalence_claimed") is False, "transport equivalence was claimed")
    require(pipeline.get("current_stage") == "specification", "pipeline stage differs")
    return {
        "claim_boundary": "Fresh independent disposition of the sole consumed S5 terminal NO-GO plus the AMLT executable provenance deviation. The review may accept only evidence authenticity, the negative 27/56 quality disposition, and bounded provenance classification. It grants no transport-equivalence claim, retry, rescoring, threshold change, W4A8, RTL, synthesis/PPA, FPGA/U280, or stage advance.",
        "terminal_audit": artifact(TERMINAL_AUDIT),
        "provenance_audit": artifact(PROVENANCE_AUDIT),
        "authority": artifact(AUTHORITY),
        "pipeline_state": artifact(PIPELINE),
        "terminal_summary": {
            "status": terminal["status"],
            "failure_taxonomy": terminal["failure_taxonomy"],
            "cardinality": terminal["cardinality"],
            "backend": terminal["backend"],
            "dev_hard_pass_count": terminal["quality"]["dev"]["hard_pass_count"],
            "dev_response_count": terminal["quality"]["dev"]["response_count"],
            "dev_minimum_hard_passes": terminal["failure"]["dev_minimum_hard_passes"],
            "selected_epoch": terminal["quality"]["selected_epoch"],
            "retention_executed": terminal["execution_boundaries"]["retention_executed"],
            "holdout_executed": terminal["execution_boundaries"]["holdout_executed"],
        },
        "provenance_summary": {
            "classification": provenance["classification"],
            "submitted_launcher_sha256": provenance["submitted_client"]["sha256"],
            "canonical_launcher_sha256": provenance["canonical_client"]["sha256"],
            "same_reported_version": provenance["classification_limits"]["same_reported_version"],
            "same_entry_module_bytes": provenance["classification_limits"]["same_entry_module_bytes"],
            "same_distribution_metadata_bytes": provenance["classification_limits"]["same_distribution_metadata_bytes"],
            "same_installation_record": provenance["classification_limits"]["same_installation_record"],
            "canonical_executable_used_for_submission": provenance["classification_limits"]["canonical_executable_used_for_submission"],
            "transport_equivalence_claimed": provenance["classification_limits"]["transport_equivalence_claimed"],
        },
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
        print("ACE2_S5_TERMINAL_PROVENANCE_REVIEW_CHECK_PASS")
        return 0

    require(not REQUEST.exists() and not DECISION.exists(), "S5 terminal provenance review already exists")
    evidence = verified_inputs()
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
    objective = "Independently classify the sole consumed ACE-2 S5 terminal quality result and its AMLT client provenance deviation."
    instruction = (
        "Inspect only the listed hash-bound project artifacts and run short read-only checks. Do not submit, retry, resubmit, resume, relaunch, cancel, rescore, alter thresholds, inspect or expose case prompts/responses/golden content, train, run retention or holdout, create a successor, run W4A8/RTL/simulation/synthesis/PPA/FPGA/U280 work, or advance the pipeline. "
        "Verify exactly one authority, intent, experiment, and backend job with zero retries; training completed 132 steps; epoch 3 was probe-locked before one official dev; dev is terminal NO_GO at 27/56 versus 48; retention and holdout did not run. Independently inspect the archived submitted and canonical launchers and the provenance audit. The submitted launcher hash is 91791a9e..., the directed canonical launcher hash is 29c4b23b..., and the canonical executable was not used for submission. Both report AMLT 11.17.0 and have identical amlt.amlt and METADATA bytes, but launcher/shebang/install provenance and RECORD differ. Classify this only as a bounded submission-client provenance deviation; do not claim transport equivalence. "
        "Return done only if all evidence and boundaries hold. Include exactly these rulings: "
        + ", ".join(f"'{r}: supported'" for r in RULINGS)
        + ". A done verdict is TERMINAL_NO_GO and opens no retry or downstream authority."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=sid,
        main_summary="The immutable S5 terminal audit passes 60/60 at 27/56 NO_GO. A separate 22/22 audit preserves the noncanonical AMLT launcher and classifies it as a bounded provenance deviation without claiming transport equivalence.",
        main_error=None,
        raw_evidence=json.dumps(evidence, indent=2, sort_keys=True),
        scope="s5_terminal_quality_no_go_and_client_provenance_only",
        checkpoint_path="",
        escalate_hint="No backend execution, evidence mutation, successor construction, or stage transition is in scope.",
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
        "scope": "s5_terminal_quality_no_go_and_client_provenance_only",
        "claim_boundary": evidence["claim_boundary"],
        "review_request": artifact(REQUEST),
        "verified_evidence": evidence,
        "review": raw,
    })
    if raw.get("status") != "done":
        print(f"ACE2_S5_TERMINAL_PROVENANCE_REVIEW_REPLAN status={raw.get('status')}")
        return 2
    print(f"ACE2_S5_TERMINAL_PROVENANCE_REVIEW_DONE thread={raw.get('thread_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
