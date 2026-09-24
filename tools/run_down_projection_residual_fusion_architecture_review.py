#!/usr/bin/env python3
"""Run one fresh independent architecture review for the active successor."""

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
CONTRACT = "shared_down_projection_residual_fusion_v1"
PREDECESSOR = "shared_v_residual_value_correction_attention_v1"
MISSION_ID = "architecture-refreeze-shared-down-projection-residual-fusion-v1"
PACKET = ROOT / f"evidence/{CONTRACT}/latest/ARCHITECTURE_REFREEZE_REVIEW_PACKET.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
MEMORY = ROOT / "design/MEMORY_MODEL.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
OUT = ROOT / f"evidence/review/architecture_refreeze_{CONTRACT}/decision.json"
QK_NO_GO = ROOT / "evidence/shared_qk_residual_cross_term_attention_v1/latest/VERIFICATION_NO_GO.json"
V_NO_GO = ROOT / "evidence/shared_v_residual_value_correction_attention_v1/latest/VERIFICATION_DECISION.json"

STAGE_CHECKLIST = {
    "architecture.compute-memory-model": True,
    "architecture.interface-control": True,
    "architecture.leverage-risk": True,
    "architecture.area-reuse-plan": True,
}

REVIEW_CHECKLIST = {
    "structurally_distinct_from_both_sealed_families": True,
    "sealed_family_hash_bindings_are_current": True,
    "exact_arithmetic_scaling_rounding_and_saturation_are_frozen": True,
    "interface_memory_reset_backpressure_completion_and_error_semantics_are_frozen": True,
    "accepted_prefix_frontier_and_operator_owned_targets_are_preserved": True,
    "integer_cycle_sram_and_bandwidth_budgets_reconcile": True,
    "resource_sharing_area_and_timing_risks_are_bounded_without_fabricated_ppa": True,
    "first_party_provenance_and_full_model_hook_are_hash_bound": True,
    "minimal_two_dataset_discriminator_exposes_c4_cross_layer_and_final_output_regression_before_rtl": True,
    "public_probe_projection_matches_the_frozen_claim_boundary": True,
    "active_public_compute_model_matches_the_successor": True,
    "maintained_check_only_path_covers_active_public_compute_model": True,
    "negative_stale_predecessor_rejection_is_exercised": True,
    "review_order_and_implementation_lock_are_preserved": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def dump(path: Path, value: dict[str, Any]) -> None:
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


def projection_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    pipeline = load(PIPELINE)
    packet = load(PACKET)
    public = load(PUBLIC)
    memory = load(MEMORY)

    require(pipeline.get("current_stage") == "architecture", "Manager-owned stage is not architecture")
    require(packet.get("contract_id") == CONTRACT, "review packet contract mismatch")
    require(
        packet.get("status") == "ready_for_exactly_one_fresh_independent_architecture_review",
        "packet is not review-ready",
    )
    require(packet.get("implementation_authorized") is False, "packet authorizes implementation")
    require(
        packet.get("integrity", {}).get("canonical_sha256") == canonical_sha256(packet),
        "packet canonical digest mismatch",
    )
    require(memory.get("architecture_checklist") == STAGE_CHECKLIST, "architecture checklist is incomplete")

    for location, model in (
        ("dashboard_fields", public["dashboard_fields"]["current_architecture_performance_model"]),
        ("implementation_frontier", public["implementation_frontier"]["current_architecture_performance_model"]),
    ):
        require(model.get("contract_id") == CONTRACT, f"{location} current model is stale")
        require(model.get("contract_id") != PREDECESSOR, f"{location} exposes the sealed predecessor")
        require(model.get("planned_sram_peak_bytes") == 464320, f"{location} SRAM peak mismatch")
        require(model.get("remaining_sram_margin_bytes") == 59968, f"{location} SRAM margin mismatch")
        require(model.get("combined_cycles_per_layer_token") == 1100288, f"{location} cycle mismatch")
        require(model.get("dma_payload_floor_cycles_at_16_bytes_per_cycle") == 136608, f"{location} DMA mismatch")
        require(model.get("area_status") == "unmeasured_no_candidate_ppa", f"{location} area claim mismatch")
        require(
            model.get("frequency_status") == "unmeasured_100mhz_is_operator_floor_not_evidence",
            f"{location} frequency claim mismatch",
        )

    binder_output = run_check([sys.executable, "tools/bind_down_projection_residual_fusion_architecture.py", "--check"])
    hook_output = run_check([sys.executable, "tools/ace2_down_projection_residual_fusion_hook.py", "--self-test"])
    require("ACE2_DOWN_PROJECTION_RESIDUAL_FUSION_BINDING_PASS" in binder_output, "binder pass marker missing")
    require("ACE2_EXACT_SCALE32_FUSION_HOOK_SELF_TEST PASS" in hook_output, "hook self-test marker missing")

    verified = {
        "stage": pipeline["current_stage"],
        "stage_checklist": STAGE_CHECKLIST,
        "review_checklist": REVIEW_CHECKLIST,
        "artifacts": {
            "packet": artifact(PACKET),
            "public_status": artifact(PUBLIC),
            "memory_model": artifact(MEMORY),
            "architecture": artifact(ROOT / "design/ARCHITECTURE.md"),
            "spec": artifact(ROOT / "design/SPEC.md"),
            "target": artifact(ROOT / "design/TARGET.json"),
            "fast_loop_policy": artifact(ROOT / "design/FAST_LOOP_POLICY.json"),
            "binder": artifact(ROOT / "tools/bind_down_projection_residual_fusion_architecture.py"),
            "exact_hook": artifact(ROOT / "tools/ace2_down_projection_residual_fusion_hook.py"),
            "qk_no_go": artifact(QK_NO_GO),
            "v_no_go": artifact(V_NO_GO),
            "prior_review": artifact(OUT) if OUT.is_file() else None,
        },
        "decisive_checks": {
            "architecture_binding": binder_output,
            "exact_scale32_hook": hook_output,
        },
        "active_public_models": {
            "dashboard_fields": public["dashboard_fields"]["current_architecture_performance_model"],
            "implementation_frontier": public["implementation_frontier"]["current_architecture_performance_model"],
        },
        "prohibited_runs": [
            "datasets",
            "rtl",
            "shell",
            "verification",
            "ppa",
            "prototype",
            "benchmark",
            "signoff",
        ],
    }
    return packet, verified


def archive_prior_decision() -> str | None:
    if not OUT.is_file():
        return None
    digest = sha256(OUT)
    archive = OUT.parent / "archive" / f"decision.{digest}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        shutil.copy2(OUT, archive)
    return archive.relative_to(ROOT).as_posix()


def check_existing() -> None:
    decision = load(OUT)
    require(decision.get("reviewer_status") == "done", "fresh architecture review is not done")
    require(decision.get("architecture_accepted") is True, "architecture is not accepted")
    require(decision.get("contract_id") == CONTRACT, "review contract mismatch")
    require(decision.get("packet", {}).get("sha256") == sha256(PACKET), "review packet binding stale")
    require(decision.get("checklist") == REVIEW_CHECKLIST, "review checklist incomplete")
    require(decision.get("stage_checklist") == STAGE_CHECKLIST, "stage checklist incomplete")
    require(decision.get("implementation_authorized") is False, "review authorizes implementation")
    require(decision.get("stage_closing") is False, "review changes the Manager-owned stage")
    require("public_status" not in decision, "review decision contains a circular live public-status binding")
    binding = decision.get("review_binding", {})
    require(isinstance(binding, dict), "review binding missing")
    require(binding.get("packet_sha256") == sha256(PACKET), "review binding packet digest stale")
    require(
        binding.get("public_projection_hash_algorithm") == "sha256-json-sort-keys-compact-utf8-v1",
        "review binding projection hash algorithm differs",
    )
    expected_projection_paths = {
        "dashboard_fields.current_architecture_performance_model",
        "implementation_frontier.current_architecture_performance_model",
    }
    recorded_projection_hashes = binding.get("public_projection_hashes", {})
    require(
        isinstance(recorded_projection_hashes, dict)
        and set(recorded_projection_hashes) == expected_projection_paths,
        "review binding projection hash set differs",
    )
    require(binding.get("public_projections_equal") is True, "reviewed public projections were not equal")
    require(
        isinstance(binding.get("public_status_pre_verdict_sha256"), str)
        and len(binding["public_status_pre_verdict_sha256"]) == 64,
        "reviewed pre-verdict public-status digest missing",
    )
    public = load(PUBLIC)
    pipeline = load(PIPELINE)
    dashboard_model = public["dashboard_fields"]["current_architecture_performance_model"]
    frontier_model = public["implementation_frontier"]["current_architecture_performance_model"]
    require(dashboard_model == frontier_model, "live public architecture projections differ")
    require(
        projection_sha256(dashboard_model)
        == recorded_projection_hashes["dashboard_fields.current_architecture_performance_model"],
        "dashboard architecture projection changed after review",
    )
    require(
        projection_sha256(frontier_model)
        == recorded_projection_hashes["implementation_frontier.current_architecture_performance_model"],
        "implementation-frontier architecture projection changed after review",
    )
    require(dashboard_model.get("contract_id") == CONTRACT, "live public architecture contract differs")
    require(
        public["architecture_certification_mission"].get("reviewer_status") == "done",
        "public architecture certification is not accepted",
    )
    require(
        public["architecture_certification_mission"].get("contract_id") == CONTRACT,
        "public architecture certification contract differs",
    )
    require(
        public.get("stage", {}).get("current_stage") == pipeline.get("current_stage"),
        "public stage differs from Manager-owned stage",
    )
    require(
        pipeline.get("current_stage") in {"architecture", "environment", "rtl", "verification", "ppa", "prototype", "benchmark", "signoff"},
        "Manager-owned stage predates the accepted architecture review",
    )
    if pipeline.get("current_stage") != "architecture":
        mission = public["architecture_certification_mission"]
        require(
            str(mission.get("projection_scope", "")).startswith("historical_"),
            "advanced-stage architecture certification is not historical-scoped",
        )
        require(bool(mission.get("historical_reason")), "historical architecture reason missing")
    review_records = [
        record
        for record in public.get("artifact_hashes", [])
        if isinstance(record, dict) and record.get("path") == OUT.relative_to(ROOT).as_posix()
    ]
    require(len(review_records) == 1, "public review artifact binding missing or duplicated")
    require(review_records[0] == artifact(OUT), "public review artifact binding stale")
    require(
        public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public),
        "public status canonical digest stale",
    )
    print(f"ACE2_DOWN_PROJECTION_RESIDUAL_FUSION_ARCHITECTURE_REVIEW_CHECK_PASS contract={CONTRACT}")


def publish_accepted_review() -> None:
    decision = load(OUT)
    require(decision.get("reviewer_status") == "done", "cannot publish a non-done review")
    require(decision.get("architecture_accepted") is True, "cannot publish a rejected architecture")
    prior_public_record = decision.pop("public_status", None)
    if prior_public_record is not None:
        require(isinstance(prior_public_record, dict), "invalid reviewed public-status record")
        reviewed_sha = prior_public_record.get("sha256")
        require(isinstance(reviewed_sha, str) and len(reviewed_sha) == 64, "reviewed public hash missing")
        decision["reviewed_public_status_sha256"] = reviewed_sha
    require(
        isinstance(decision.get("reviewed_public_status_sha256"), str),
        "reviewed public-status digest unavailable",
    )
    dump(OUT, decision)

    public = load(PUBLIC)
    reviewed_at = decision["reviewed_at_utc"]
    accepted_status = "architecture_independently_accepted_manager_transition_pending"
    latest_decision = "down_projection_residual_fusion_architecture_independently_accepted"
    routing_status = "architecture_review_accepted_manager_transition_pending"
    manager_action = "manager_may_advance_architecture_to_environment_in_order"

    mission = public["architecture_certification_mission"]
    mission["status"] = "independent_architecture_review_accepted_manager_transition_pending"
    mission["reviewer_status"] = "done"
    mission["reviewed_at_utc"] = reviewed_at

    proposal = public["architecture_proposal_gate"]
    proposal["status"] = accepted_status
    proposal["required_manager_action"] = manager_action

    selected = public["selected_replacement_contract"]
    selected["status"] = accepted_status
    selected["required_manager_action"] = manager_action

    for container_name in ("dashboard_fields", "implementation_frontier"):
        container = public[container_name]
        container["candidate_mechanism"]["status"] = accepted_status
        container["latest_decision"] = latest_decision
        container["routing_status"] = routing_status

    public["latest_decision"] = latest_decision
    public["routing_authorized"] = "manager_may_advance_architecture_to_environment_only"
    public["routing_status"] = routing_status
    public["generated_at_utc"] = reviewed_at
    public["last_updated_utc"] = reviewed_at

    review_record = artifact(OUT)
    review_path = review_record["path"]
    public["artifact_hashes"] = [
        record
        for record in public.get("artifact_hashes", [])
        if not (isinstance(record, dict) and record.get("path") == review_path)
    ]
    public["artifact_hashes"].append(review_record)
    public.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)


def main() -> None:
    if "--publish-accepted" in sys.argv[1:]:
        publish_accepted_review()
        check_existing()
        return
    if "--check" in sys.argv[1:]:
        check_existing()
        return

    packet, verified = preflight()
    resume_thread_id = None
    if OUT.is_file():
        prior = load(OUT)
        prior_raw = prior.get("raw_decision", {})
        if (
            prior.get("reviewer_status") == "blocked"
            and isinstance(prior_raw, dict)
            and isinstance(prior_raw.get("thread_id"), str)
            and prior_raw["thread_id"]
        ):
            resume_thread_id = prior_raw["thread_id"]
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
        "Independently decide whether the active shared_down_projection_residual_fusion_v1 "
        "architecture refreeze is accepted for Manager-owned architecture-stage exit only. "
        "Acceptance requires all four architecture checklist items, structural distinction from "
        "both sealed predecessor families, internally reconciled integer compute/memory budgets, "
        "frozen interface and numerical semantics, bounded reuse/timing risks, a sufficient exact "
        "two-dataset pre-RTL discriminator, and repaired public current-model projections."
    )
    instruction = (
        "Read design/CHIP_SCOPE.json first, then inspect the exact refreeze packet, "
        "design/ARCHITECTURE.md, design/MEMORY_MODEL.json, design/SPEC.md, design/TARGET.json, "
        "design/FAST_LOOP_POLICY.json, research/PUBLIC_STATUS.json, both sealed no-go artifacts, "
        "tools/bind_down_projection_residual_fusion_architecture.py, and the exact Scale32 hook. "
        "The prior review withheld acceptance only because both public current-performance-model "
        "projections exposed the sealed V-residual model and the binder missed that contradiction. "
        "Verify those two repairs independently, including the negative stale-predecessor tests. "
        "This is a standalone project-native review runner: no project-root latest.json is required; "
        "the authoritative context is the supplied raw evidence, current CHECKPOINT.md, and current "
        "packet/design files. You may run only `make architecture-contract-check`, the exact hook "
        "`--self-test`, and short read-only hash/content checks. Edit only CHECKPOINT.md for the "
        "mandatory reviewer handoff. Do not run datasets, RTL, shell, "
        "verification, synthesis, STA, PPA, prototype, benchmark, or signoff. Return done only if "
        "every REVIEW_CHECKLIST item in the raw evidence is supported. A done verdict accepts only "
        "the architecture refreeze; it must keep implementation_authorized=false, stage_closing=false, "
        "and leave current_stage unchanged for the Manager. Otherwise return continue with exact repairs."
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
            "Planner repaired both active public compute-model projections from the successor "
            "MEMORY_MODEL, added exact successor schema/value checks plus two negative stale-predecessor "
            "tests to the maintained binder, and passed the architecture binder and exact Scale32 hook "
            "self-test. The Reviewer must inspect current files and decisive checks independently."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="architecture_refreeze_acceptance_only_stage_closing_false",
        checkpoint_path=str(CHECKPOINT),
        background_context=(
            "A prior attempt already completed the independent evidence inspection and found all "
            "architecture items passing, but its final verdict was procedurally blocked because a "
            "read-only sandbox rejected the mandatory CHECKPOINT.md edit. The current retry permits "
            "only that checkpoint handoff; absence of a project-root latest.json is expected for this "
            "standalone runner and is not an architecture provenance requirement."
        ),
        escalate_hint=(
            "If accepted, Manager alone may advance architecture to environment in order. "
            "No environment or implementation authorization is granted by this review."
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
        resume_thread_id=resume_thread_id,
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
        "scope": "bounded_architecture_refreeze_acceptance_for_environment_review_entry_only",
        "status": "accepted" if accepted else "changes_required",
        "architecture_accepted": accepted,
        "stage_closing": False,
        "implementation_authorized": False,
        "packet": artifact(PACKET),
        "sealed_family_bindings": {
            "shared_qk_residual_cross_term_attention_v1": artifact(QK_NO_GO),
            PREDECESSOR: artifact(V_NO_GO),
        },
        "checklist": REVIEW_CHECKLIST if accepted else {key: False for key in REVIEW_CHECKLIST},
        "stage_checklist": STAGE_CHECKLIST if accepted else {key: False for key in STAGE_CHECKLIST},
        "decisive_checks": verified["decisive_checks"],
        "reviewed_public_status_sha256": sha256(PUBLIC),
        "binder": artifact(ROOT / "tools/bind_down_projection_residual_fusion_architecture.py"),
        "archived_prior_decision": archived,
        "reason": raw.get("reason", ""),
        "required_remediation": (
            "none"
            if accepted
            else (raw.get("next_action") or raw.get("reason") or "review did not pass")
        ),
        "next_required_gate": (
            "Manager-owned architecture-to-environment transition; implementation remains unauthorized"
            if accepted
            else "fresh independent architecture acceptance remains pending"
        ),
        "reviewer_thread_id": raw.get("thread_id", ""),
        "raw_decision": raw,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, OUT)
    if accepted:
        publish_accepted_review()
    print(
        "ACE2_DOWN_PROJECTION_RESIDUAL_FUSION_ARCHITECTURE_REVIEW_RESULT "
        f"status={status} contract={CONTRACT} decision={OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
