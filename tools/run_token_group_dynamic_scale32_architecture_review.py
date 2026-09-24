#!/usr/bin/env python3
"""Run one fresh independent architecture review for the dynamic Scale32 proposal."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
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
CONTRACT = "shared_token_group_dynamic_scale32_v1"
MISSION_ID = "architecture-review-shared-token-group-dynamic-scale32-v1"
PROPOSAL = ROOT / f"evidence/{CONTRACT}/architecture/PROPOSAL.json"
PROPOSAL_SUM = ROOT / f"evidence/{CONTRACT}/architecture/PROPOSAL.sha256"
PUBLIC_CERTIFICATION = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1/PUBLIC_CERTIFICATION.json"
HARNESS_REPORT = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1/HARNESS_TEST_REPORT.json"
HARNESS_MANIFEST = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1/MANIFEST.json"
HARNESS_L2_REVIEW = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1/L2_HARNESS_REVIEW.json"
MANAGER_RECOVERY_SEAL = ROOT / "evidence/cross_layer_quantization_error_carry_final_output_v1/recovery/manager_architecture_rollback_v1/SEAL.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
LIVE = ROOT / ".argus/live-view.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
OUT = ROOT / f"evidence/review/architecture_{CONTRACT}/decision.json"
BASELINE_CERTIFICATION_BINDER = ROOT / "tools/bind_synthetic_baseline_harness_certification.py"

STAGE_CHECKLIST = {
    "architecture.compute-memory-model": True,
    "architecture.interface-control": True,
    "architecture.leverage-risk": True,
    "architecture.area-reuse-plan": True,
}

HARNESS_PROPERTIES = [
    "stage_authorization_before_execution",
    "durable_exactly_one_reservation_and_run_ledger",
    "atomic_artifact_and_companion_hash_commit",
    "exact_provenance_binding",
    "candidate_entrypoint_interlock",
    "crash_recovery",
    "missing_or_mismatched_artifact_fail_closed",
]

REVIEW_CHECKLIST = {
    "architecture_stage_and_manager_authority_preserved": True,
    "proposal_and_companion_hash_are_exact": True,
    "public_certification_and_report_manifest_l2_seal_hashes_are_exact": True,
    "all_seven_harness_properties_are_enumerated": True,
    "no_execution_claim_and_implementation_lock_are_preserved": True,
    "rtl_stage_is_limited_to_reference_vector_lint_elaboration_bit_exact_interface_minimal_formal": True,
    "verification_is_two_separately_reviewed_exactly_one_tasks": True,
    "candidate_entry_requires_independent_disabled_mode_baseline_pass": True,
    "full_shell_and_sky130_ppa_require_candidate_go_and_separate_manager_transitions": True,
    "compute_memory_interface_leverage_and_reuse_contracts_are_complete": True,
    "operator_targets_prefix_mode_and_historical_ppa_frontier_are_preserved": True,
    "proposal_is_structurally_distinct_and_contains_no_fabricated_result": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    candidate = copy.deepcopy(value)
    candidate.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(encoded.encode()).hexdigest()


def session_id() -> str:
    configured = os.environ.get("ARGUS_SKILL_SESSION_ID")
    if configured:
        return configured
    session_file = ROOT / ".ace2-session.json"
    if session_file.is_file():
        value = load(session_file).get("sid")
        if isinstance(value, str) and value:
            return value
    return ROOT.name


def run_check(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    require(completed.returncode == 0, f"check failed: {' '.join(command)}\n{completed.stdout}")
    return completed.stdout.strip()


def preflight() -> dict[str, Any]:
    pipeline = load(PIPELINE)
    proposal = load(PROPOSAL)
    certification = load(PUBLIC_CERTIFICATION)
    public = load(PUBLIC)

    require(pipeline.get("current_stage") == "architecture", "Manager-owned stage is not architecture")
    require(proposal.get("contract_id") == CONTRACT, "proposal contract differs")
    require(proposal.get("schema_version") == 2, "proposal schema is stale")
    require(proposal.get("architecture_checklist") == STAGE_CHECKLIST, "architecture checklist is incomplete")
    require(proposal.get("authority", {}).get("successor_frozen") is False, "proposal self-freezes")
    require(proposal.get("authority", {}).get("implementation_authorized") is False, "proposal authorizes implementation")
    require(proposal.get("authority", {}).get("stage_closing") is False, "proposal closes the Manager-owned stage")
    require(proposal.get("execution_gates", {}).get("no_execution_claim") is True, "proposal makes an execution claim")
    require(
        PROPOSAL_SUM.read_text(encoding="utf-8").strip().split() == [sha256(PROPOSAL), PROPOSAL.name],
        "proposal companion hash differs",
    )

    bound = proposal.get("baseline_harness_certification", {})
    require(bound.get("public_certification") == artifact(PUBLIC_CERTIFICATION), "public certification binding differs")
    require(bound.get("required_properties_in_execution_order") == HARNESS_PROPERTIES, "harness property order differs")
    require(bound.get("candidate_or_model_execution_count") == 0, "proposal claims harness execution")
    exact = bound.get("exact_evidence", {})
    require(exact.get("report") == artifact(HARNESS_REPORT), "harness report binding differs")
    require(exact.get("manifest") == artifact(HARNESS_MANIFEST), "harness manifest binding differs")
    require(exact.get("independent_l2_review") == artifact(HARNESS_L2_REVIEW), "harness L2 binding differs")
    require(exact.get("manager_recovery_seal") == artifact(MANAGER_RECOVERY_SEAL), "Manager seal binding differs")
    require(certification.get("decision") == "PASS", "public harness certification is not PASS")

    rtl = proposal["execution_gates"]["rtl_stage"]
    require(
        rtl.get("allowed_checks")
        == [
            "reference_and_vector_generation",
            "lint",
            "elaboration",
            "bit_exact_simulation",
            "interface_checks",
            "minimal_formal",
        ],
        "RTL-stage check allowlist differs",
    )
    require(
        set(rtl.get("prohibited", []))
        == {
            "disabled_mode_baseline_execution",
            "candidate_quality_execution",
            "full_shell_regression",
            "canonical_sky130_ppa",
        },
        "RTL-stage prohibition set differs",
    )
    tasks = proposal["execution_gates"]["verification_stage"]["tasks_in_strict_order"]
    require([item.get("task") for item in tasks] == [1, 2], "verification tasks are not strictly ordered")
    require(all(item.get("execution_limit") == 1 for item in tasks), "verification task is not exactly-one")
    require(all(item.get("independent_review_required") is True for item in tasks), "verification review split differs")
    require(tasks[1].get("entry_condition") == "independent_task_1_baseline_PASS", "candidate interlock differs")

    require(public.get("ordered_supported_layer_operator_prefix") == proposal["frontier"]["ordered_supported_layer_operator_prefix"], "public prefix differs")
    require(public.get("first_unsupported_layer_operator") == "layer_0.rope_q", "public first unsupported differs")
    require(public.get("current_mode") == "ADVANCE", "public mode differs")
    require(public.get("architecture_successor_proposal", {}).get("evidence") == artifact(PROPOSAL), "public proposal hash differs")

    architecture_check = run_check([sys.executable, "tools/bind_token_group_dynamic_scale32_proposal.py", "--check"])
    harness_check = run_check([sys.executable, "tools/bind_synthetic_baseline_harness_certification.py", "--check"])
    require("ACE2_TOKEN_GROUP_DYNAMIC_SCALE32_PROPOSAL_CHECK_PASS" in architecture_check, "proposal check marker missing")
    require("ACE2_SYNTHETIC_BASELINE_HARNESS_CERTIFICATION_CHECK_PASS" in harness_check, "harness check marker missing")

    return {
        "stage": pipeline["current_stage"],
        "stage_checklist": STAGE_CHECKLIST,
        "review_checklist": REVIEW_CHECKLIST,
        "proposal": artifact(PROPOSAL),
        "proposal_companion": artifact(PROPOSAL_SUM),
        "certification_chain": {
            "public_certification": artifact(PUBLIC_CERTIFICATION),
            "report": artifact(HARNESS_REPORT),
            "manifest": artifact(HARNESS_MANIFEST),
            "independent_l2_review": artifact(HARNESS_L2_REVIEW),
            "manager_recovery_seal": artifact(MANAGER_RECOVERY_SEAL),
        },
        "artifacts": {
            "architecture": artifact(ROOT / "design/ARCHITECTURE.md"),
            "memory_model": artifact(ROOT / "design/MEMORY_MODEL.json"),
            "spec": artifact(ROOT / "design/SPEC.md"),
            "target": artifact(ROOT / "design/TARGET.json"),
            "fast_loop_policy": artifact(ROOT / "design/FAST_LOOP_POLICY.json"),
            "public_status": artifact(PUBLIC),
            "binder": artifact(ROOT / "tools/bind_token_group_dynamic_scale32_proposal.py"),
        },
        "decisive_checks": {
            "architecture_proposal": architecture_check,
            "baseline_harness_certification": harness_check,
        },
        "prohibited_runs": [
            "baseline_or_candidate_model_execution",
            "rtl",
            "shell",
            "verification",
            "synthesis",
            "sta",
            "ppa",
            "prototype",
            "benchmark",
            "signoff",
        ],
    }


def archive_prior_decision() -> str | None:
    if not OUT.is_file():
        return None
    digest = sha256(OUT)
    archive = OUT.parent / "archive" / f"decision.{digest}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        shutil.copy2(OUT, archive)
    return archive.relative_to(ROOT).as_posix()


def publish_review() -> None:
    decision = load(OUT)
    accepted = decision.get("reviewer_status") == "done" and decision.get("architecture_accepted") is True
    public = load(PUBLIC)
    review_projection = {
        "contract_id": CONTRACT,
        "status": "independently_accepted_pending_manager_freeze" if accepted else "changes_required",
        "reviewer_status": decision.get("reviewer_status"),
        "architecture_accepted": accepted,
        "proposal": artifact(PROPOSAL),
        "decision": artifact(OUT),
        "stage_closing": False,
        "successor_frozen": False,
        "implementation_authorized": False,
        "required_manager_action": (
            "freeze_this_structurally_distinct_successor_or_record_architecture_no_go"
            if accepted
            else "hold_freeze_and_repair_the_architecture_contract"
        ),
        "claim_boundary": "independent architecture review only; no execution or downstream acceptance",
    }
    public["architecture_successor_review"] = copy.deepcopy(review_projection)
    dispatch = public.setdefault("architecture_successor_dispatch", {})
    dispatch.update({
        "independent_architecture_review": artifact(OUT),
        "reviewer_status": decision.get("reviewer_status"),
        "architecture_accepted": accepted,
        "successor_frozen": False,
        "implementation_authorized": False,
        "stage_closing": False,
        "required_manager_action": review_projection["required_manager_action"],
    })
    for name in ("dashboard_fields", "implementation_frontier"):
        section = public.setdefault(name, {})
        section["architecture_successor_review"] = copy.deepcopy(review_projection)
        section["latest_decision"] = (
            "shared_token_group_dynamic_scale32_architecture_independently_accepted_pending_manager_freeze"
            if accepted
            else "shared_token_group_dynamic_scale32_architecture_changes_required"
        )
    public["latest_decision"] = (
        "shared_token_group_dynamic_scale32_architecture_independently_accepted_pending_manager_freeze"
        if accepted
        else "shared_token_group_dynamic_scale32_architecture_changes_required"
    )
    records = {
        item.get("path"): item
        for item in public.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path")
    }
    for path in (
        CHECKPOINT,
        PROPOSAL,
        PROPOSAL_SUM,
        OUT,
        BASELINE_CERTIFICATION_BINDER,
        Path(__file__).resolve(),
    ):
        records[path.relative_to(ROOT).as_posix()] = artifact(path)
    public["artifact_hashes"] = [records[key] for key in sorted(records)]
    public.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    public["integrity"]["canonical_sha256"] = None
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)

    live = load(LIVE)
    live.update({
        "proposal_id": CONTRACT,
        "architecture_review_status": "accepted" if accepted else "changes_required",
        "architecture_review_evidence": OUT.relative_to(ROOT).as_posix(),
        "successor_frozen": False,
        "implementation_authorized": False,
        "reason": (
            "fresh independent architecture review accepted; Manager freeze or architecture no-go pending"
            if accepted
            else "fresh independent architecture review requested repairs; Manager freeze remains prohibited"
        ),
    })
    dump(LIVE, live)


def check_existing() -> None:
    decision = load(OUT)
    require(decision.get("reviewer_status") == "done", "fresh architecture review is not done")
    require(decision.get("architecture_accepted") is True, "architecture review did not accept the proposal")
    require(decision.get("contract_id") == CONTRACT, "review contract differs")
    require(decision.get("proposal") == artifact(PROPOSAL), "review proposal binding is stale")
    require(decision.get("checklist") == REVIEW_CHECKLIST, "review checklist differs")
    require(decision.get("stage_checklist") == STAGE_CHECKLIST, "stage checklist differs")
    require(decision.get("implementation_authorized") is False, "review authorizes implementation")
    require(decision.get("stage_closing") is False, "review closes the Manager-owned stage")
    require(load(PIPELINE).get("current_stage") == "architecture", "Manager-owned stage changed")
    public = load(PUBLIC)
    review = public.get("architecture_successor_review", {})
    require(review.get("architecture_accepted") is True, "public review acceptance missing")
    require(review.get("decision") == artifact(OUT), "public review decision binding differs")
    require(review.get("implementation_authorized") is False, "public review authorizes implementation")
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public), "public canonical hash differs")
    print(f"ACE2_TOKEN_GROUP_DYNAMIC_SCALE32_ARCHITECTURE_REVIEW_CHECK_PASS contract={CONTRACT}")


def main() -> None:
    if "--refresh-public" in sys.argv[1:]:
        publish_review()
        check_existing()
        return
    if "--check" in sys.argv[1:]:
        check_existing()
        return

    verified = preflight()
    archived = archive_prior_decision()
    backend_name = resolve_role_backend("reviewer")
    runner = AgentCliBackend(
        backend=backend_name,
        runner_bin=resolve_runner_bin_setting("reviewer") or None,
    )
    sid = session_id()
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
        "Independently decide whether shared_token_group_dynamic_scale32_v1 is a complete and internally "
        "consistent architecture proposal eligible for a later Manager freeze, while preserving "
        "current_stage=architecture, implementation_authorized=false, and the no-execution claim."
    )
    instruction = (
        "Read design/CHIP_SCOPE.json and CHECKPOINT.md first. Then inspect the exact proposal and companion "
        "hash, design/ARCHITECTURE.md, design/MEMORY_MODEL.json, design/SPEC.md, design/TARGET.json, "
        "design/FAST_LOOP_POLICY.json, research/PUBLIC_STATUS.json, PUBLIC_CERTIFICATION.json, its exact "
        "report/manifest/L2/seal evidence, and tools/bind_token_group_dynamic_scale32_proposal.py. Verify all "
        "four architecture checklist items and every REVIEW_CHECKLIST item from the supplied evidence. In "
        "particular, reject any plan that runs baseline, candidate, full shell, or PPA in RTL; verification "
        "must be two separately reviewed exactly-one tasks, candidate entry must follow independent baseline "
        "PASS, and shell/PPA must wait for candidate quality GO plus separate Manager authorization/transitions. "
        "You may run only `make architecture-proposal-check`, `make baseline-harness-certification-check`, and "
        "short read-only hash/content checks. Edit only CHECKPOINT.md for the reviewer handoff. Do not execute "
        "models, datasets, RTL, shell, verification, synthesis, STA, PPA, prototype, benchmark, or signoff. "
        "Return done only if the exact hash-bound proposal is acceptable for a later Manager freeze. A done "
        "verdict is architecture-only and must keep stage_closing=false and implementation_authorized=false."
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
            "The Planner removed full shell and canonical PPA from the RTL task, split verification into two "
            "separately reviewed exactly-one executions, bound the proposal to the exact certified harness "
            "chain and all seven properties, regenerated the proposal/hash/public projections, and passed "
            "architecture-only binder checks. Independent inspection is still required."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="architecture_acceptance_only_no_execution_stage_closing_false",
        checkpoint_path=str(CHECKPOINT),
        background_context=(
            "The predecessor QECR contract is sealed terminally. The candidate-independent synthetic harness "
            "is independently certified and has zero candidate/model executions. The proposed successor is "
            "not Manager-frozen and no RTL or downstream action is authorized."
        ),
        escalate_hint=(
            "If accepted, Manager alone may freeze the successor and advance stages in order. This review "
            "does not authorize implementation or any execution."
        ),
        preselected_skill_block=reviewer_skill,
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort(
                "ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"
            ),
            skip_git_repo_check=True,
            full_auto=True,
            dangerous_yolo=False,
            sandbox_mode="workspace-write",
            isolate_workdir=False,
            working_dir=str(ROOT),
        ),
    )

    raw = asdict(review)
    status = raw.get("status", "continue")
    accepted = status == "done"
    reviewed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload = {
        "schema_version": 1,
        "reviewed_at_utc": reviewed_at,
        "reviewer_role": "independent_architecture",
        "reviewer_status": status,
        "contract_id": CONTRACT,
        "scope": "architecture_acceptance_for_manager_freeze_decision_only",
        "status": "accepted" if accepted else "changes_required",
        "architecture_accepted": accepted,
        "stage_closing": False,
        "implementation_authorized": False,
        "no_execution_claim": True,
        "proposal": artifact(PROPOSAL),
        "proposal_companion": artifact(PROPOSAL_SUM),
        "certification_chain": verified["certification_chain"],
        "checklist": REVIEW_CHECKLIST if accepted else {key: False for key in REVIEW_CHECKLIST},
        "stage_checklist": STAGE_CHECKLIST if accepted else {key: False for key in STAGE_CHECKLIST},
        "decisive_checks": verified["decisive_checks"],
        "reviewed_public_status_sha256": sha256(PUBLIC),
        "binder": artifact(ROOT / "tools/bind_token_group_dynamic_scale32_proposal.py"),
        "archived_prior_decision": archived,
        "reason": raw.get("reason", ""),
        "required_remediation": "none" if accepted else (raw.get("next_action") or raw.get("reason") or "review did not pass"),
        "next_required_gate": (
            "Manager freeze or architecture no-go; implementation remains unauthorized"
            if accepted
            else "repair and rerun a fresh independent architecture review before Manager freeze"
        ),
        "reviewer_thread_id": raw.get("thread_id", ""),
        "raw_decision": raw,
    }
    dump(OUT, payload)
    publish_review()
    print(
        "ACE2_TOKEN_GROUP_DYNAMIC_SCALE32_ARCHITECTURE_REVIEW_RESULT "
        f"status={status} contract={CONTRACT} decision={OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
