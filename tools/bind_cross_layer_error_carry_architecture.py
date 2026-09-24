#!/usr/bin/env python3
"""Bind and validate the active ACE-2 cross-layer error-carry architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "cross_layer_quantization_error_carry_final_output_v1"
PREDECESSOR = "shared_down_projection_residual_fusion_v1"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
NO_GO = ROOT / "evidence/shared_down_projection_residual_fusion_v1/latest/VERIFICATION_DECISION.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
CHIP_SCOPE = ROOT / "design/CHIP_SCOPE.json"
FREEZE = ROOT / f"evidence/{CONTRACT}/architecture_freeze.json"
PACKET = ROOT / f"evidence/{CONTRACT}/latest/ARCHITECTURE_REVIEW_PACKET.json"
REVIEW_DECISION = (
    ROOT
    / f"evidence/review/architecture_stage_closing_{CONTRACT}/decision.json"
)
SOURCE_PATHS = [
    "MISSION.md",
    "design/ARCHITECTURE.md",
    "design/MEMORY_MODEL.json",
    "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
    "design/SPEC.md",
    "design/WORKLOAD.md",
    "design/FAST_LOOP_POLICY.json",
    "design/TARGET.json",
    "research/PIPELINE_STATE.json",
    "evidence/shared_down_projection_residual_fusion_v1/latest/VERIFICATION_DECISION.json",
]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise ValueError(f"missing architecture source: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(value))
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(clone, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") == "architecture", "Manager-owned stage is not architecture")
    successor = pipeline.get("successor")
    require(isinstance(successor, dict), "pipeline successor object is missing")
    require(successor.get("id") == CONTRACT, "pipeline successor differs")
    require(successor.get("state") == "frozen_unimplemented", "pipeline successor state differs")
    require(successor.get("predecessor") == PREDECESSOR, "pipeline predecessor differs")
    require(successor.get("downstream_runs") == {
        "benchmark": "unrun",
        "paired": "unrun",
        "physical": "unrun",
        "ppa": "unrun",
        "prototype": "unrun",
        "shell": "unrun",
        "signoff": "unrun",
    }, "successor downstream-run locks differ")
    require(successor.get("excluded_residual_families") == [
        "q_k_residual",
        "v_residual",
        "down_projection_residual_fusion_parameter_variant",
    ], "excluded residual families differ")

    no_go = load(NO_GO)
    require(sha256(NO_GO) == "9655ca8a724fc04c6e32c26f7883f07d3ff117b82d2aa04ebbc68b15de26d8c9", "predecessor no-go hash differs")
    require(no_go.get("contract_id") == PREDECESSOR, "predecessor no-go contract differs")

    architecture = (ROOT / "design/ARCHITECTURE.md").read_text(encoding="utf-8")
    spec = (ROOT / "design/SPEC.md").read_text(encoding="utf-8")
    proposal = (ROOT / "design/NUMERICAL_REPLACEMENT_PROPOSAL.md").read_text(encoding="utf-8")
    memory = load(ROOT / "design/MEMORY_MODEL.json")
    policy = load(ROOT / "design/FAST_LOOP_POLICY.json")
    target = load(ROOT / "design/TARGET.json")

    for token in (
        CONTRACT,
        "signed-16 Q0.15",
        "signed-24",
        "SSq30",
        "23,296 cycles",
        "88,512 bytes",
        "58,176 bytes",
        "exactly once",
    ):
        require(token in architecture, f"architecture contract lacks {token!r}")
    for token in (CONTRACT, "flag bit 7", "CHAIN_END", "numeric_overflow", "layer_id=1..24"):
        require(token in spec, f"interface contract lacks {token!r}")
    require(CONTRACT in proposal and "zero-carry" in proposal, "numerical proposal is incomplete")
    require(memory.get("contract_id") == CONTRACT, "memory-model contract differs")
    require(memory.get("architecture_checklist") == {
        "architecture.compute-memory-model": True,
        "architecture.interface-control": True,
        "architecture.leverage-risk": True,
        "architecture.area-reuse-plan": True,
    }, "architecture checklist differs")
    require(memory["memory_hierarchy"]["planned_peak_sram_bytes"] == 466112, "planned SRAM peak differs")
    require(memory["compute_model"]["carry_cycles_per_layer_token"] == 23296, "carry cycle hook differs")
    maximum_reconstructed_magnitude = max(
        abs((-128 << 15) - 16384),
        abs((127 << 15) + 16384),
    )
    require(maximum_reconstructed_magnitude < (1 << 23), "signed-24 reconstruction bound fails")
    maximum_square_sum = maximum_reconstructed_magnitude**2 * 896
    require(maximum_square_sum < (1 << 56), "unsigned-56 square-sum bound fails")
    require(896 * 26 == memory["compute_model"]["carry_cycles_per_layer_token"], "carry cycle arithmetic differs")
    require(
        377600 + 86592 + 1792 + 128
        == memory["memory_hierarchy"]["planned_peak_sram_bytes"],
        "SRAM mapping arithmetic differs",
    )
    require(
        524288 - memory["memory_hierarchy"]["planned_peak_sram_bytes"]
        == memory["memory_hierarchy"]["remaining_sram_margin_bytes"],
        "SRAM margin arithmetic differs",
    )
    require(
        (24 + 23 + 1) * 896 * 2
        == memory["memory_hierarchy"]["carry_sram_traffic_bytes_per_token"],
        "carry SRAM traffic arithmetic differs",
    )
    require(policy["active_architecture_contract"]["contract_id"] == CONTRACT, "fast-loop active contract differs")
    require(target["current_architecture_contract"]["contract_id"] == CONTRACT, "target active contract differs")
    require(policy["area_cap_non_sram_mm2"] == 2.0 and policy["frequency_floor_mhz"] == 100.0, "operator targets changed")
    return pipeline, no_go


def active_contract(now: str) -> dict[str, Any]:
    return {
        "contract_id": CONTRACT,
        "predecessor_contract_id": PREDECESSOR,
        "predecessor_status": "sealed_integrity_no_go",
        "frozen_at_utc": now,
        "mechanism": "signed_q0_15_post_mlp_error_carry_consumed_once_by_next_or_final_rmsnorm",
        "implementation_authorized": False,
        "operator_approval_consumed": False,
        "rtl_started": False,
        "stage_closing": False,
        "mode": "ADVANCE",
        "ordered_supported_layer_operator_prefix": PREFIX,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "area_cap_non_sram_mm2": 2.0,
        "frequency_floor_mhz": 100.0,
        "memory_boundary_bits": 128,
        "required_manager_action": "independent_architecture_review_then_advance_to_environment_in_order",
        "status": "architecture_frozen_fresh_independent_review_pending",
        "forbidden": [
            "rtl_before_manager_advance",
            "double_carry_consumption",
            "carry_added_to_residual_bypass",
            "silent_carry_saturation",
            "prompt_or_dataset_control",
            "candidate_ppa_before_full_verification",
            "operator_target_relaxation",
        ],
    }


def candidate_mechanism() -> dict[str, Any]:
    return {
        "contract_id": CONTRACT,
        "mechanism": "signed_q0_15_post_mlp_error_carry_consumed_once_by_next_or_final_rmsnorm",
        "status": "architecture_frozen_fresh_independent_review_pending",
        "implementation_authorized": False,
        "planned_carry_cycles_per_layer": 23296,
        "planned_sram_peak_bytes": 466112,
        "remaining_sram_bytes": 58176,
        "incremental_external_bytes_per_token": 0,
        "quality_status": "unrun_for_successor",
    }


def current_performance_model() -> dict[str, Any]:
    return {
        "contract_id": CONTRACT,
        "source": "design/MEMORY_MODEL.json",
        "status": "architecture_estimate_reconciled_not_measured_no_candidate_ppa",
        "claim_status": "architecture estimates only; serial carry and RMSNorm schedules are planning hooks, not measured successor RTL or benchmark results",
        "down_projection_cycles_per_layer_token": 1089536,
        "carry_cycles_per_layer_token": 23296,
        "down_plus_carry_cycles_per_layer_token": 1112832,
        "all_24_layer_carry_cycles_per_token": 559104,
        "carry_fraction_of_down_dot_percent": 2.138157895,
        "maximum_slice_speedup_if_carry_were_free": 1.021381579,
        "rmsnorm_square_cycles_selected": 896,
        "rmsnorm_added_cycles_vs_16_lane": 840,
        "macs_per_layer_token": 4358144,
        "external_bytes_per_layer_token": 2185728,
        "arithmetic_intensity_macs_per_external_byte": 1.9939095807,
        "compute_memory_balance_bytes_per_cycle": 1.964113182,
        "dma_payload_floor_cycles_at_16_bytes_per_cycle": 136608,
        "dma_planning_cycles_at_16_bytes_per_cycle": 136672,
        "nonoverlapped_total_cycles_at_16_bytes_per_cycle": 1249504,
        "candidate_allocation_bytes": 88512,
        "planned_sram_peak_bytes": 466112,
        "remaining_sram_margin_bytes": 58176,
        "carry_sram_traffic_bytes_per_token": 86016,
        "incremental_external_or_kv_bytes_per_token": 0,
        "area_status": "unmeasured_no_candidate_ppa",
        "frequency_status": "unmeasured_100mhz_is_operator_floor_not_evidence",
        "quality_evidence_status": "unrun_for_successor",
    }


def update_chip_scope(now: str) -> None:
    scope = load(CHIP_SCOPE)
    contract = active_contract(now)
    scope["authority_override"] = {
        "active_replacement_contract": CONTRACT,
        "implementation_authorized": False,
        "operator_approval_consumed": False,
        "stage_closing": False,
        "status": "architecture_frozen_fresh_independent_review_pending",
        "required_next_action": "fresh_independent_architecture_review",
        "operator_implementation_approval": contract,
        "predecessor_no_go": artifact("evidence/shared_down_projection_residual_fusion_v1/latest/VERIFICATION_DECISION.json"),
    }
    scope["stage"] = {
        "current_stage": "architecture",
        "current_stage_status": "architecture_frozen_fresh_independent_review_pending",
        "current_stage_checklist": {
            "architecture.compute-memory-model": True,
            "architecture.interface-control": True,
            "architecture.leverage-risk": True,
            "architecture.area-reuse-plan": True,
        },
        "current_stage_evidence": [
            "design/ARCHITECTURE.md",
            "design/MEMORY_MODEL.json",
            "design/SPEC.md",
            "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        ],
        "stage_closing": False,
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
        "downstream_stages_locked_until_manager_advance": [
            "environment", "rtl", "verification", "ppa", "prototype", "benchmark", "signoff"
        ],
    }
    frontier = scope.setdefault("implementation_frontier", {})
    frontier.update({
        "current_mode": "ADVANCE",
        "latest_decision": "cross_layer_quantization_error_carry_architecture_frozen_review_pending",
        "candidate_rtl_hash": None,
        "candidate_rtl_hash_scope": "unimplemented_no_successor_rtl_or_ppa",
        "ordered_supported_layer_operator_prefix": PREFIX,
        "supported_layer_operator_prefix": PREFIX,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
    })
    policy = scope.setdefault("operator_owned_execution_policy", {})
    policy["active_successor_contract"] = contract
    scope["last_updated_utc"] = now
    dump(CHIP_SCOPE, scope)


def build_freeze(now: str) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "generated_at_utc": now,
        "contract_id": CONTRACT,
        "stage": "architecture",
        "state": "frozen_unimplemented",
        "stage_closing": False,
        "implementation_authorized": False,
        "architecture_checklist": {
            "architecture.compute-memory-model": True,
            "architecture.interface-control": True,
            "architecture.leverage-risk": True,
            "architecture.area-reuse-plan": True,
        },
        "numeric_contract": {
            "carry_format": "signed16_q0_15",
            "reconstructed_rmsnorm_input": "signed24_q8_15",
            "rmsnorm_square_sum": "unsigned56",
            "carry_consumption": "exactly_once_next_input_rmsnorm_or_final_rmsnorm",
            "final_hidden_saturation": "numeric_overflow_no_success",
        },
        "planning": {
            "carry_cycles_per_layer": 23296,
            "down_plus_carry_cycles_per_layer": 1112832,
            "candidate_allocation_bytes": 88512,
            "planned_peak_sram_bytes": 466112,
            "remaining_sram_margin_bytes": 58176,
            "incremental_external_bytes_per_token": 0,
        },
        "frontier": {
            "mode": "ADVANCE",
            "ordered_supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "area_cap_non_sram_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "historical_cells": 62199,
            "historical_non_sram_area_mm2": 0.6108746272,
            "historical_setup_slack_ns_at_100mhz": 0.1502,
        },
        "predecessor_no_go": artifact("evidence/shared_down_projection_residual_fusion_v1/latest/VERIFICATION_DECISION.json"),
        "downstream_runs": {
            "rtl": "unrun",
            "verification": "unrun",
            "shell": "unrun",
            "ppa": "unrun",
            "physical": "unrun",
            "prototype": "unrun",
            "benchmark": "unrun",
            "signoff": "unrun",
        },
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    value["integrity"]["canonical_sha256"] = canonical_sha256(value)
    return value


def build_packet(now: str) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "generated_at_utc": now,
        "contract_id": CONTRACT,
        "stage": "architecture",
        "review_status": "pending_fresh_independent_l2",
        "reviewer_status": "pending",
        "stage_closing": False,
        "implementation_authorized": False,
        "checklist": {
            "architecture.compute-memory-model": True,
            "architecture.interface-control": True,
            "architecture.leverage-risk": True,
            "architecture.area-reuse-plan": True,
        },
        "review_questions": [
            "Are the signed-Q0.15 carry, signed-Q8.15 reconstruction, and unsigned-56 RMSNorm bounds internally complete and unambiguous?",
            "Do producer/consumer descriptors, completion gating, reset, errors, backpressure, and exactly-once semantics close interface-control?",
            "Do roofline/Amdahl values and the one-engine reuse choice quantify leverage and area risk without claiming PPA?",
            "Are the sealed predecessor, immutable targets, supported prefix, first unsupported operator, and downstream locks preserved?",
        ],
        "source_artifacts": [artifact(relative) for relative in SOURCE_PATHS],
        "architecture_freeze": artifact(FREEZE.relative_to(ROOT).as_posix()),
        "claim_boundary": [
            "Architecture contract and planning estimates only.",
            "No successor RTL, verification, PPA, power, physical, prototype, benchmark, signoff, tapeout, or silicon result.",
            "Manager alone changes current_stage; implementation remains unauthorized.",
        ],
        "required_manager_action": "hold_architecture_for_fresh_independent_review",
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    value["integrity"]["canonical_sha256"] = canonical_sha256(value)
    return value


def update_public(now: str) -> None:
    public = load(PUBLIC)
    checklist = {
        "architecture.compute-memory-model": True,
        "architecture.interface-control": True,
        "architecture.leverage-risk": True,
        "architecture.area-reuse-plan": True,
    }
    selected = active_contract(now)
    selected["architecture_review_packet"] = artifact(PACKET.relative_to(ROOT).as_posix())
    public.update({
        "generated_at_utc": now,
        "last_updated_utc": now,
        "current_mode": "ADVANCE",
        "latest_decision": "cross_layer_quantization_error_carry_architecture_frozen_review_pending",
        "ordered_supported_layer_operator_prefix": PREFIX,
        "supported_layer_operator_prefix": PREFIX,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "selected_replacement_contract": selected,
        "routing_authorized": False,
        "routing_status": "architecture_frozen_fresh_independent_review_pending",
        "stage_closing": False,
        "required_operator_action": "none",
        "required_manager_action": "hold_architecture_for_fresh_independent_review",
    })
    public["architecture_proposal_gate"] = {
        "contract_id": CONTRACT,
        "status": "architecture_frozen_fresh_independent_review_pending",
        "implementation_authorized": False,
        "stage_closing": False,
        "checklist": checklist,
        "review_packet": selected["architecture_review_packet"],
    }
    public["architecture_certification_mission"] = {
        "contract_id": CONTRACT,
        "status": "architecture_frozen_fresh_independent_review_pending",
        "reviewer_status": "pending",
        "reviewed_at_utc": None,
        "implementation_authorized": False,
        "projection_scope": "active_architecture_stage_projection",
        "checklist": checklist,
        "review_packet": PACKET.relative_to(ROOT).as_posix(),
    }
    public["architecture_successor_dispatch"] = {
        "contract_id": CONTRACT,
        "current_stage": "architecture",
        "status": "architecture_frozen_fresh_independent_review_pending",
        "stage_closing": False,
        "implementation_authorized": False,
        "required_manager_action": "hold_architecture_for_fresh_independent_review",
        "result_binding": PACKET.relative_to(ROOT).as_posix(),
    }
    public["stage"] = {
        "current_stage": "architecture",
        "completed_prior_stages": ["definition"],
        "current_stage_status": "architecture_frozen_fresh_independent_review_pending",
        "current_stage_checklist": checklist,
        "current_stage_evidence": [
            "design/ARCHITECTURE.md",
            "design/MEMORY_MODEL.json",
            "design/SPEC.md",
            "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
            PACKET.relative_to(ROOT).as_posix(),
        ],
        "stage_closing": False,
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    public["blockers"] = [{
        "id": "fresh_independent_architecture_review",
        "stage": "architecture",
        "status": "active",
        "reason": "The successor architecture is frozen and machine-bound but has not received a fresh independent L2 verdict.",
        "required_resolution": "Independent reviewer accepts all four architecture gates; Manager may then advance to environment in order.",
        "evidence": PACKET.relative_to(ROOT).as_posix(),
    }]
    public["latest_rtl_candidate"] = {
        "contract_id": CONTRACT,
        "candidate_id": None,
        "candidate_rtl_hash": None,
        "status": "unimplemented_architecture_only",
        "stage_closing": False,
    }
    model = current_performance_model()
    mechanism = candidate_mechanism()
    public["implementation_frontier"].update({
        "current_stage": "architecture",
        "current_mode": "ADVANCE",
        "latest_decision": "cross_layer_quantization_error_carry_architecture_frozen_review_pending",
        "ordered_supported_layer_operator_prefix": PREFIX,
        "supported_layer_operator_prefix": PREFIX,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "candidate_rtl_hash": None,
        "candidate_rtl_hash_scope": "unimplemented_no_successor_rtl_or_ppa",
        "candidate_mechanism": mechanism,
        "current_architecture_performance_model": model,
    })
    dashboard = public.setdefault("dashboard_fields", {})
    dashboard.update({
        "current_stage": "architecture",
        "current_mode": "ADVANCE",
        "mode": "ADVANCE",
        "latest_decision": "cross_layer_quantization_error_carry_architecture_frozen_review_pending",
        "ordered_supported_layer_operator_prefix": PREFIX,
        "supported_layer_operator_prefix": PREFIX,
        "supported_layers": PREFIX,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "implementation_authorized": False,
        "candidate_meets_numeric_acceptance": False,
        "candidate_rtl_hash": None,
        "candidate_rtl_hash_scope": "unimplemented_no_successor_rtl_or_ppa",
        "latest_rtl_candidate": {
            "contract_id": CONTRACT,
            "candidate_id": None,
            "candidate_rtl_hash": None,
            "candidate_rtl_hash_scope": "unimplemented_no_successor_rtl_or_ppa",
            "implementation_authorized": False,
            "stage_closing": False,
            "status": "unimplemented_architecture_only",
            "required_manager_action": "hold_architecture_for_fresh_independent_review",
        },
        "operator_policy": load(ROOT / "design/FAST_LOOP_POLICY.json"),
        "required_operator_action": "none",
        "required_manager_action": "hold_architecture_for_fresh_independent_review",
        "routing_status": "architecture_frozen_fresh_independent_review_pending",
        "candidate_mechanism": mechanism,
        "current_architecture_performance_model": model,
    })
    public["artifact_hashes"] = [
        artifact("design/ARCHITECTURE.md"),
        artifact("design/MEMORY_MODEL.json"),
        artifact("design/NUMERICAL_REPLACEMENT_PROPOSAL.md"),
        artifact("design/SPEC.md"),
        artifact("design/FAST_LOOP_POLICY.json"),
        artifact("design/TARGET.json"),
        artifact("design/CHIP_SCOPE.json"),
        artifact(FREEZE.relative_to(ROOT).as_posix()),
        artifact(PACKET.relative_to(ROOT).as_posix()),
        artifact("evidence/shared_down_projection_residual_fusion_v1/latest/VERIFICATION_DECISION.json"),
    ]
    public.setdefault("integrity", {})["canonical_sha256"] = None
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)


def verify_generated() -> None:
    freeze = load(FREEZE)
    packet = load(PACKET)
    require(freeze.get("contract_id") == CONTRACT, "architecture freeze contract differs")
    require(freeze.get("integrity", {}).get("canonical_sha256") == canonical_sha256(freeze), "architecture freeze hash differs")
    require(packet.get("contract_id") == CONTRACT, "review packet contract differs")
    require(packet.get("review_status") == "pending_fresh_independent_l2", "review status differs")
    require(packet.get("integrity", {}).get("canonical_sha256") == canonical_sha256(packet), "review packet hash differs")
    for record in packet["source_artifacts"]:
        require(artifact(record["path"]) == record, f"packet source differs: {record['path']}")
    public = load(PUBLIC)
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public), "public status hash differs")
    latest_decision = public.get("latest_decision")
    require(latest_decision in {
        "cross_layer_quantization_error_carry_architecture_frozen_review_pending",
        "cross_layer_quantization_error_carry_architecture_independently_accepted",
    }, "public decision differs")
    dashboard = public.get("dashboard_fields", {})
    frontier = public.get("implementation_frontier", {})
    expected_mechanism = candidate_mechanism()
    if latest_decision == "cross_layer_quantization_error_carry_architecture_frozen_review_pending":
        require(dashboard.get("candidate_mechanism") == expected_mechanism, "dashboard candidate mechanism differs")
        require(frontier.get("candidate_mechanism") == expected_mechanism, "frontier candidate mechanism differs")
    else:
        require(REVIEW_DECISION.is_file(), "accepted architecture review decision is missing")
        decision = load(REVIEW_DECISION)
        require(decision.get("reviewer_status") == "done", "accepted architecture reviewer is not done")
        require(decision.get("status") == "accepted", "architecture review is not accepted")
        require(decision.get("architecture_accepted") is True, "architecture acceptance flag differs")
        require(decision.get("required_remediation") == "", "accepted architecture remediation is not empty")
        require(
            decision.get("binder")
            == artifact("tools/bind_cross_layer_error_carry_architecture.py"),
            "accepted architecture review binder binding is stale",
        )
        for label, mechanism in (
            ("dashboard", dashboard.get("candidate_mechanism", {})),
            ("frontier", frontier.get("candidate_mechanism", {})),
        ):
            for key, value in expected_mechanism.items():
                if key != "status":
                    require(mechanism.get(key) == value, f"{label} candidate mechanism differs: {key}")
            require(
                mechanism.get("status")
                == "architecture_independently_accepted_manager_transition_pending",
                f"{label} accepted candidate status differs",
            )
    require(
        dashboard.get("current_architecture_performance_model") == current_performance_model(),
        "dashboard architecture performance model differs",
    )
    require(
        frontier.get("current_architecture_performance_model") == current_performance_model(),
        "frontier architecture performance model differs",
    )
    require(
        dashboard.get("current_architecture_performance_model")
        == frontier.get("current_architecture_performance_model"),
        "public architecture projections differ",
    )
    require(
        frontier["current_architecture_performance_model"].get("contract_id") != PREDECESSOR,
        "sealed predecessor remains in the active public architecture projection",
    )
    scope = load(CHIP_SCOPE)
    require(scope["authority_override"]["active_replacement_contract"] == CONTRACT, "chip scope active contract differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    validate_sources()
    if args.check:
        verify_generated()
        print("ACE2_CROSS_LAYER_ERROR_CARRY_ARCHITECTURE_BINDING_PASS")
        return 0
    now = utc_now()
    update_chip_scope(now)
    dump(FREEZE, build_freeze(now))
    dump(PACKET, build_packet(now))
    update_public(now)
    verify_generated()
    print("ACE2_CROSS_LAYER_ERROR_CARRY_ARCHITECTURE_BINDING_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
