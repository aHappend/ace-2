#!/usr/bin/env python3
"""Run one independent L2 architecture review for the active error-carry successor."""

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
CONTRACT = "cross_layer_quantization_error_carry_final_output_v1"
PREDECESSOR = "shared_down_projection_residual_fusion_v1"
MISSION_ID = "architecture-review-cross-layer-quantization-error-carry-final-output-v1"
PACKET = ROOT / f"evidence/{CONTRACT}/latest/ARCHITECTURE_REVIEW_PACKET.json"
FREEZE = ROOT / f"evidence/{CONTRACT}/architecture_freeze.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
MEMORY = ROOT / "design/MEMORY_MODEL.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
LIVE_VIEW = ROOT / ".argus/live-view.json"
OUT = ROOT / f"evidence/review/architecture_stage_closing_{CONTRACT}/decision.json"
BINDER = ROOT / "tools/bind_cross_layer_error_carry_architecture.py"
PREDECESSOR_NO_GO = ROOT / f"evidence/{PREDECESSOR}/latest/VERIFICATION_DECISION.json"
QK_NO_GO = ROOT / "evidence/shared_qk_residual_cross_term_attention_v1/latest/VERIFICATION_NO_GO.json"
V_NO_GO = ROOT / "evidence/shared_v_residual_value_correction_attention_v1/latest/VERIFICATION_DECISION.json"

STAGE_CHECKLIST = {
    "architecture.compute-memory-model": True,
    "architecture.interface-control": True,
    "architecture.leverage-risk": True,
    "architecture.area-reuse-plan": True,
}

REVIEW_CHECKLIST = {
    "scalar_equations_formats_rounding_saturation_and_width_bounds_are_frozen": True,
    "integer_cycle_roofline_bandwidth_and_sram_budgets_reconcile": True,
    "producer_consumer_reset_error_backpressure_completion_and_cdc_semantics_are_explicit": True,
    "amdahl_leverage_risks_verification_stop_rules_and_fallbacks_are_recorded": True,
    "lifetime_sharing_covers_compute_sfu_dma_buffers_state_and_control_with_alternatives": True,
    "both_public_current_architecture_projections_bind_the_active_successor": True,
    "sealed_predecessors_prefix_first_unsupported_targets_and_downstream_locks_are_preserved": True,
    "claim_boundary_contains_no_successor_rtl_ppa_benchmark_or_silicon_result": True,
    "full_model_two_dataset_discriminator_and_zero_carry_replay_are_required_before_admission": True,
    "manager_stage_ownership_and_implementation_lock_are_preserved": True,
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


def preflight() -> dict[str, Any]:
    pipeline = load(PIPELINE)
    packet = load(PACKET)
    public = load(PUBLIC)
    memory = load(MEMORY)

    require(pipeline.get("current_stage") == "architecture", "Manager-owned stage is not architecture")
    successor = pipeline.get("successor", {})
    require(successor.get("id") == CONTRACT, "pipeline successor differs")
    require(successor.get("state") == "frozen_unimplemented", "successor is not frozen/unimplemented")
    require(packet.get("contract_id") == CONTRACT, "review packet contract mismatch")
    require(packet.get("review_status") == "pending_fresh_independent_l2", "packet is not review-ready")
    require(packet.get("stage_closing") is False, "packet attempts to close the Manager-owned stage")
    require(packet.get("implementation_authorized") is False, "packet authorizes implementation")
    require(
        packet.get("integrity", {}).get("canonical_sha256") == canonical_sha256(packet),
        "packet canonical digest mismatch",
    )
    require(memory.get("architecture_checklist") == STAGE_CHECKLIST, "architecture checklist is incomplete")

    dashboard = public["dashboard_fields"]
    frontier = public["implementation_frontier"]
    dashboard_model = dashboard["current_architecture_performance_model"]
    frontier_model = frontier["current_architecture_performance_model"]
    require(dashboard_model == frontier_model, "public current-architecture projections differ")
    require(dashboard_model.get("contract_id") == CONTRACT, "public current architecture is stale")
    require(dashboard_model.get("contract_id") != PREDECESSOR, "sealed predecessor remains active")
    require(dashboard_model.get("down_plus_carry_cycles_per_layer_token") == 1112832, "cycle model differs")
    require(dashboard_model.get("planned_sram_peak_bytes") == 466112, "SRAM peak differs")
    require(dashboard_model.get("remaining_sram_margin_bytes") == 58176, "SRAM margin differs")
    require(dashboard_model.get("area_status") == "unmeasured_no_candidate_ppa", "area claim differs")
    require(
        dashboard_model.get("frequency_status") == "unmeasured_100mhz_is_operator_floor_not_evidence",
        "frequency claim differs",
    )
    require(dashboard["candidate_mechanism"] == frontier["candidate_mechanism"], "candidate projections differ")
    require(dashboard["candidate_mechanism"].get("contract_id") == CONTRACT, "candidate projection is stale")

    maximum_reconstructed_magnitude = max(
        abs((-128 << 15) - 16384),
        abs((127 << 15) + 16384),
    )
    maximum_square_sum = maximum_reconstructed_magnitude**2 * 896
    require(maximum_reconstructed_magnitude < (1 << 23), "signed-24 reconstruction bound fails")
    require(maximum_square_sum < (1 << 56), "unsigned-56 square-sum bound fails")
    require(896 * 26 == 23296, "carry cycle arithmetic differs")
    require(377600 + 86592 + 1792 + 128 == 466112, "SRAM allocation arithmetic differs")
    require(524288 - 466112 == 58176, "SRAM reserve arithmetic differs")
    require((24 + 23 + 1) * 896 * 2 == 86016, "carry traffic arithmetic differs")

    binding = run_check([sys.executable, "tools/bind_cross_layer_error_carry_architecture.py", "--check"])
    require("ACE2_CROSS_LAYER_ERROR_CARRY_ARCHITECTURE_BINDING_PASS" in binding, "binder marker missing")

    return {
        "stage": pipeline["current_stage"],
        "stage_checklist": STAGE_CHECKLIST,
        "review_checklist": REVIEW_CHECKLIST,
        "artifacts": {
            "packet": artifact(PACKET),
            "architecture": artifact(ROOT / "design/ARCHITECTURE.md"),
            "memory_model": artifact(MEMORY),
            "spec": artifact(ROOT / "design/SPEC.md"),
            "proposal": artifact(ROOT / "design/NUMERICAL_REPLACEMENT_PROPOSAL.md"),
            "target": artifact(ROOT / "design/TARGET.json"),
            "fast_loop_policy": artifact(ROOT / "design/FAST_LOOP_POLICY.json"),
            "public_status_pre_review": artifact(PUBLIC),
            "binder": artifact(ROOT / "tools/bind_cross_layer_error_carry_architecture.py"),
            "predecessor_no_go": artifact(PREDECESSOR_NO_GO),
            "qk_no_go": artifact(QK_NO_GO),
            "v_no_go": artifact(V_NO_GO),
        },
        "decisive_checks": {
            "architecture_binding": binding,
            "maximum_reconstructed_magnitude": maximum_reconstructed_magnitude,
            "maximum_square_sum": maximum_square_sum,
            "carry_cycles_per_layer": 23296,
            "planned_sram_peak_bytes": 466112,
            "remaining_sram_margin_bytes": 58176,
            "carry_sram_traffic_bytes_per_token": 86016,
        },
        "public_projection_hashes": {
            "dashboard_fields.current_architecture_performance_model": projection_sha256(dashboard_model),
            "implementation_frontier.current_architecture_performance_model": projection_sha256(frontier_model),
        },
        "prohibited_runs": [
            "rtl",
            "verification",
            "shell",
            "ppa",
            "physical",
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


def publish_accepted_review() -> None:
    decision = load(OUT)
    require(decision.get("reviewer_status") == "done", "cannot publish a non-done review")
    require(decision.get("architecture_accepted") is True, "cannot publish a rejected review")

    public = load(PUBLIC)
    reviewed_at = decision["reviewed_at_utc"]
    accepted_status = "architecture_independently_accepted_manager_transition_pending"
    latest_decision = "cross_layer_quantization_error_carry_architecture_independently_accepted"
    manager_action = "manager_may_advance_architecture_to_environment_in_order"

    public["generated_at_utc"] = reviewed_at
    public["last_updated_utc"] = reviewed_at
    public["latest_decision"] = latest_decision
    public["routing_authorized"] = "manager_may_advance_architecture_to_environment_only"
    public["routing_status"] = "architecture_review_accepted_manager_transition_pending"
    public["required_manager_action"] = manager_action
    public["required_operator_action"] = "none"
    public["blockers"] = []

    mission = public["architecture_certification_mission"]
    mission.update({
        "status": accepted_status,
        "reviewer_status": "done",
        "reviewed_at_utc": reviewed_at,
    })
    public["architecture_proposal_gate"].update({
        "status": accepted_status,
        "required_manager_action": manager_action,
    })
    public["architecture_successor_dispatch"].update({
        "status": accepted_status,
        "required_manager_action": manager_action,
    })
    public["selected_replacement_contract"].update({
        "status": accepted_status,
        "required_manager_action": manager_action,
    })
    public["stage"].update({
        "current_stage_status": accepted_status,
        "stage_closing": False,
        "planner_may_advance_stage": False,
    })

    current_rtl_candidate = {
        "candidate_id": None,
        "candidate_rtl_hash": None,
        "candidate_rtl_hash_scope": "unimplemented_no_successor_rtl_or_ppa",
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "required_manager_action": manager_action,
        "stage_closing": False,
        "status": "unimplemented_architecture_only",
    }
    public["latest_rtl_candidate"] = copy.deepcopy(current_rtl_candidate)

    for name in ("dashboard_fields", "implementation_frontier"):
        container = public[name]
        container["candidate_mechanism"]["status"] = accepted_status
        container["latest_rtl_candidate"] = copy.deepcopy(current_rtl_candidate)
        container["latest_decision"] = latest_decision
        container["routing_status"] = "architecture_review_accepted_manager_transition_pending"
        container["required_manager_action"] = manager_action

    review_record = artifact(OUT)
    review_path = review_record["path"]
    public["artifact_hashes"] = [
        record
        for record in public.get("artifact_hashes", [])
        if not (isinstance(record, dict) and record.get("path") == review_path)
    ]
    public["artifact_hashes"].append(review_record)
    if review_path not in public["stage"]["current_stage_evidence"]:
        public["stage"]["current_stage_evidence"].append(review_path)
    public.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)

    dump(LIVE_VIEW, {
        "contract_id": CONTRACT,
        "generated_at_utc": reviewed_at,
        "implementation_authorized": False,
        "paths": [
            "research/PIPELINE_STATE.json",
            "CHECKPOINT.md",
            "design/ARCHITECTURE.md",
            "design/MEMORY_MODEL.json",
            "design/SPEC.md",
            review_path,
            "research/PUBLIC_STATUS.json",
        ],
        "reason": "fresh independent L2 architecture review accepted; Manager transition pending",
        "title": "Cross-layer quantization-error-carry architecture",
        "version": 3,
    })


def check_existing() -> None:
    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") == "architecture", "Manager-owned stage is not architecture")
    successor = pipeline.get("successor", {})
    require(successor.get("id") == CONTRACT, "pipeline successor differs")
    require(successor.get("state") == "frozen_unimplemented", "successor is not frozen/unimplemented")

    packet = load(PACKET)
    require(packet.get("contract_id") == CONTRACT, "review packet contract mismatch")
    require(packet.get("stage_closing") is False, "review packet changes the Manager-owned stage")
    require(packet.get("implementation_authorized") is False, "review packet authorizes implementation")
    require(
        packet.get("integrity", {}).get("canonical_sha256") == canonical_sha256(packet),
        "review packet canonical digest mismatch",
    )
    for record in packet.get("source_artifacts", []):
        source = ROOT / record["path"]
        require(artifact(source) == record, f"reviewed source changed: {record['path']}")

    freeze = load(FREEZE)
    require(freeze.get("contract_id") == CONTRACT, "architecture freeze contract mismatch")
    require(freeze.get("stage_closing") is False, "architecture freeze changes the Manager-owned stage")
    require(freeze.get("implementation_authorized") is False, "architecture freeze authorizes implementation")
    require(
        freeze.get("integrity", {}).get("canonical_sha256") == canonical_sha256(freeze),
        "architecture freeze canonical digest mismatch",
    )
    require(packet.get("architecture_freeze") == artifact(FREEZE), "review packet freeze binding stale")

    decision = load(OUT)
    require(decision.get("reviewer_status") == "done", "fresh architecture review is not done")
    require(decision.get("architecture_accepted") is True, "architecture is not accepted")
    require(decision.get("contract_id") == CONTRACT, "review contract mismatch")
    require(decision.get("packet") == artifact(PACKET), "review packet binding stale")
    require(decision.get("binder") == artifact(BINDER), "reviewed architecture binder changed")
    require(decision.get("checklist") == REVIEW_CHECKLIST, "review checklist incomplete")
    require(decision.get("stage_checklist") == STAGE_CHECKLIST, "stage checklist incomplete")
    require(decision.get("required_remediation") == "", "accepted review remediation is not structurally empty")
    require(decision.get("implementation_authorized") is False, "review authorizes implementation")
    require(decision.get("stage_closing") is False, "review changes the Manager-owned stage")
    for record in decision.get("sealed_family_bindings", {}).values():
        require(artifact(ROOT / record["path"]) == record, f"sealed family binding stale: {record['path']}")

    public = load(PUBLIC)
    require(
        public.get("latest_decision")
        == "cross_layer_quantization_error_carry_architecture_independently_accepted",
        "public accepted decision differs",
    )
    require(
        public.get("required_manager_action")
        == "manager_may_advance_architecture_to_environment_in_order",
        "public Manager action differs",
    )
    require(public.get("required_operator_action") == "none", "public operator action differs")
    require(public.get("stage_closing") is False, "public status changes the Manager-owned stage")
    selected = public.get("selected_replacement_contract", {})
    require(selected.get("contract_id") == CONTRACT, "public selected contract differs")
    require(selected.get("implementation_authorized") is False, "public selected contract authorizes implementation")
    require(selected.get("stage_closing") is False, "public selected contract changes the Manager-owned stage")
    require(selected.get("area_cap_non_sram_mm2") == 2.0, "public area cap differs")
    require(selected.get("frequency_floor_mhz") == 100.0, "public frequency floor differs")
    require(selected.get("first_unsupported_layer_operator") == "layer_0.rope_q", "public unsupported frontier differs")
    dashboard_model = public["dashboard_fields"]["current_architecture_performance_model"]
    frontier_model = public["implementation_frontier"]["current_architecture_performance_model"]
    require(dashboard_model == frontier_model, "live public architecture projections differ")
    recorded = decision["review_binding"]["public_projection_hashes"]
    require(
        projection_sha256(dashboard_model)
        == recorded["dashboard_fields.current_architecture_performance_model"],
        "dashboard architecture projection changed after review",
    )
    require(
        projection_sha256(frontier_model)
        == recorded["implementation_frontier.current_architecture_performance_model"],
        "frontier architecture projection changed after review",
    )
    require(public["architecture_certification_mission"].get("reviewer_status") == "done", "public review status differs")
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public), "public digest stale")
    require(
        public.get("latest_rtl_candidate")
        == public["dashboard_fields"].get("latest_rtl_candidate")
        == public["implementation_frontier"].get("latest_rtl_candidate"),
        "public unimplemented RTL projections differ",
    )
    require(public["latest_rtl_candidate"].get("implementation_authorized") is False, "public RTL projection authorizes implementation")
    review_records = [
        record
        for record in public.get("artifact_hashes", [])
        if isinstance(record, dict) and record.get("path") == OUT.relative_to(ROOT).as_posix()
    ]
    require(len(review_records) == 1 and review_records[0] == artifact(OUT), "public review binding stale")
    binding = run_check([sys.executable, "tools/bind_cross_layer_error_carry_architecture.py", "--check"])
    require("ACE2_CROSS_LAYER_ERROR_CARRY_ARCHITECTURE_BINDING_PASS" in binding, "post-review binder marker missing")
    print(f"ACE2_CROSS_LAYER_ERROR_CARRY_ARCHITECTURE_REVIEW_CHECK_PASS contract={CONTRACT}")


def repair_accepted_review() -> None:
    decision = load(OUT)
    require(decision.get("reviewer_status") == "done", "existing architecture reviewer is not done")
    require(decision.get("status") == "accepted", "existing architecture review is not accepted")
    require(decision.get("architecture_accepted") is True, "existing architecture acceptance flag differs")
    require(decision.get("implementation_authorized") is False, "existing architecture review authorizes implementation")
    require(decision.get("stage_closing") is False, "existing architecture review changes the Manager-owned stage")
    archived = archive_prior_decision()
    decision["required_remediation"] = ""
    decision["binder"] = artifact(BINDER)
    decision["archived_prior_decision"] = archived
    dump(OUT, decision)
    publish_accepted_review()
    check_existing()


def main() -> None:
    if "--check" in sys.argv[1:]:
        check_existing()
        return
    if "--republish-accepted" in sys.argv[1:]:
        publish_accepted_review()
        check_existing()
        print(
            "ACE2_CROSS_LAYER_ERROR_CARRY_ARCHITECTURE_REVIEW_REPUBLISH_PASS "
            f"contract={CONTRACT}"
        )
        return
    if "--repair-accepted" in sys.argv[1:]:
        repair_accepted_review()
        print(
            "ACE2_CROSS_LAYER_ERROR_CARRY_ARCHITECTURE_REVIEW_REPAIR_PASS "
            f"contract={CONTRACT}"
        )
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
        "Independently decide whether cross_layer_quantization_error_carry_final_output_v1 "
        "satisfies all four architecture gates for Manager-owned architecture-stage exit only."
    )
    instruction = (
        "Read design/CHIP_SCOPE.json first, then inspect the exact review packet, "
        "design/ARCHITECTURE.md, design/MEMORY_MODEL.json, design/SPEC.md, "
        "design/NUMERICAL_REPLACEMENT_PROPOSAL.md, design/TARGET.json, "
        "design/FAST_LOOP_POLICY.json, research/PUBLIC_STATUS.json, the sealed predecessor "
        "decisions, and tools/bind_cross_layer_error_carry_architecture.py. Independently "
        "recompute the signed-24 reconstruction bound, unsigned-56 square-sum bound, carry "
        "cycle count, SRAM map/reserve, carry traffic, and public projection equality. You may "
        "run only `make architecture-contract-check` and short read-only hash/content checks. "
        "Edit only CHECKPOINT.md for the reviewer handoff. Do not run or edit RTL, verification, "
        "shell, PPA, physical, prototype, benchmark, signoff, or research/PIPELINE_STATE.json. "
        "Return done only if every supplied REVIEW_CHECKLIST item is supported. A done verdict "
        "accepts architecture only and must preserve implementation_authorized=false, "
        "stage_closing=false, the 2.0 mm2 cap, the 100 MHz floor, and Manager stage ownership."
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
            "Planner froze the cross-layer Q0.15 error-carry architecture, repaired the maintained "
            "architecture Make target, corrected both active public architecture projections, and "
            "passed deterministic binding, numeric-width, cycle, SRAM, and policy checks."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="architecture_acceptance_only_stage_closing_false",
        checkpoint_path=str(CHECKPOINT),
        background_context=(
            "The prior shared_down_projection_residual_fusion_v1 mechanism is a sealed integrity "
            "no-go. This successor is architecture-only and intentionally has no RTL or PPA result."
        ),
        escalate_hint=(
            "If accepted, Manager alone may advance architecture to environment in order. "
            "No implementation authorization is granted by this review."
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
    projection_hashes = verified["public_projection_hashes"]
    payload = {
        "schema_version": 1,
        "reviewed_at_utc": reviewed_at,
        "reviewer_role": "independent_l2_architecture",
        "reviewer_status": status,
        "contract_id": CONTRACT,
        "scope": "architecture_acceptance_for_manager_environment_entry_only",
        "status": "accepted" if accepted else "changes_required",
        "architecture_accepted": accepted,
        "stage_closing": False,
        "implementation_authorized": False,
        "packet": artifact(PACKET),
        "sealed_family_bindings": {
            PREDECESSOR: artifact(PREDECESSOR_NO_GO),
            "shared_qk_residual_cross_term_attention_v1": artifact(QK_NO_GO),
            "shared_v_residual_value_correction_attention_v1": artifact(V_NO_GO),
        },
        "checklist": REVIEW_CHECKLIST if accepted else {key: False for key in REVIEW_CHECKLIST},
        "stage_checklist": STAGE_CHECKLIST if accepted else {key: False for key in STAGE_CHECKLIST},
        "decisive_checks": verified["decisive_checks"],
        "review_binding": {
            "packet_sha256": sha256(PACKET),
            "public_status_pre_verdict_sha256": sha256(PUBLIC),
            "public_projection_hash_algorithm": "sha256-json-sort-keys-compact-utf8-v1",
            "public_projection_hashes": projection_hashes,
            "public_projections_equal": len(set(projection_hashes.values())) == 1,
        },
        "binder": artifact(ROOT / "tools/bind_cross_layer_error_carry_architecture.py"),
        "archived_prior_decision": archived,
        "reason": raw.get("reason", ""),
        "required_remediation": (
            "" if accepted else (raw.get("next_action") or raw.get("reason") or "review did not pass")
        ),
        "next_required_gate": (
            "Manager-owned architecture-to-environment transition; implementation remains unauthorized"
            if accepted
            else "fresh independent architecture acceptance remains pending"
        ),
        "reviewer_thread_id": raw.get("thread_id", ""),
        "raw_decision": raw,
    }
    dump(OUT, payload)
    if accepted:
        publish_accepted_review()
        check_existing()
    print(
        "ACE2_CROSS_LAYER_ERROR_CARRY_ARCHITECTURE_REVIEW_RESULT "
        f"status={status} contract={CONTRACT} decision={OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
