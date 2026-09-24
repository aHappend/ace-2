#!/usr/bin/env python3
"""Bind the QECR RTL preflight into current project state without changing stage."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "cross_layer_quantization_error_carry_final_output_v1"
PRECHECK = ROOT / f"evidence/{CONTRACT}/latest/PRECHECK.json"
PREFIX = ["layer_0.input_rmsnorm", "layer_0.q_proj", "layer_0.k_proj", "layer_0.v_proj"]
FIRST_UNSUPPORTED = "layer_0.rope_q"
CHECKLIST = {
    "rtl.contract-traceability": True,
    "rtl.hardware-discipline": True,
    "rtl.ip-provenance": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: str | Path) -> dict[str, Any]:
    resolved = path if isinstance(path, Path) else ROOT / path
    value = json.loads(resolved.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {resolved}")
    return value


def dump(path: str | Path, value: dict[str, Any]) -> None:
    resolved = path if isinstance(path, Path) else ROOT / path
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: str | Path) -> dict[str, Any]:
    resolved = path if isinstance(path, Path) else ROOT / path
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def generated_record(path: str, generator: str, kind: str, reference: str | None = None) -> dict[str, Any]:
    value = artifact(path)
    value.update({
        "kind": kind,
        "generator": generator,
        "regeneration_command": f".venv/bin/python {generator}",
        "third_party": False,
    })
    if reference:
        value["reference"] = reference
    return value


def replacement_contract(packet: dict[str, Any], now: str) -> dict[str, Any]:
    return {
        "authority": "operator_fast_loop_policy_plus_manager_environment_to_rtl_transition",
        "contract_id": CONTRACT,
        "predecessor_contract_id": "shared_down_projection_residual_fusion_v1",
        "predecessor_status": "sealed_integrity_no_go",
        "mechanism": "signed_q0_15_post_mlp_error_carry_consumed_once_by_next_or_final_rmsnorm",
        "candidate_id": packet["candidate_id"],
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "operator_approval_consumed": True,
        "rtl_started": True,
        "stage_closing": False,
        "status": "bounded_rtl_preflight_pass_independent_review_pending",
        "required_manager_action": "hold_rtl_for_fresh_independent_rtl_checklist_review",
        "ordered_supported_layer_operator_prefix": PREFIX,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "mode": "ADVANCE",
        "area_cap_non_sram_mm2": 2.0,
        "frequency_floor_mhz": 100.0,
        "memory_boundary_bits": 128,
        "updated_at_utc": now,
        "manager_transition": packet["manager_transition"],
        "forbidden": [
            "double_carry_consumption",
            "carry_added_to_residual_bypass",
            "silent_carry_saturation",
            "prompt_or_dataset_control",
            "quality_or_shell_or_ppa_before_stage_authority",
            "operator_target_relaxation",
        ],
    }


def main() -> int:
    packet = load(PRECHECK)
    pipeline = load("research/PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(packet.get("contract_id") == CONTRACT, "preflight contract mismatch")
    require(packet.get("status") == "pass_ready_for_independent_rtl_review", "preflight is not ready")
    require(packet.get("stage_checklist") == CHECKLIST, "preflight checklist differs")
    implementation_provenance = packet.get("implementation_provenance", {})
    producer_role = implementation_provenance.get("role")
    require(
        producer_role in {"engineer", "planner_direct_executor"},
        "preflight producer role is unsupported",
    )
    planner_archive = implementation_provenance.get("planner_draft_archive")
    require(isinstance(planner_archive, dict), "Planner-draft archive provenance is missing")
    require(artifact(planner_archive["path"]) == planner_archive, "Planner-draft archive provenance changed")
    require(not any(packet.get("prohibited_runs", {}).values()), "preflight records prohibited work")
    require(canonical_sha256(packet) == packet.get("integrity", {}).get("canonical_sha256"), "preflight hash mismatch")
    for relative, expected in packet["source_hashes"].items():
        require(sha256_file(ROOT / relative) == expected, f"bound source changed: {relative}")

    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    contract = replacement_contract(packet, now)
    rtl_record = artifact("rtl/ace2_cross_layer_error_carry_core.sv")
    rtl_record.update({
        "kind": "first_party_bounded_candidate_rtl",
        "modules": [module["name"] for module in packet["modules"]],
        "third_party": False,
    })
    generated = [
        generated_record(
            "verification/generated/cross_layer_error_carry_vectors.json",
            "tools/gen_cross_layer_error_carry_vectors.py",
            "generated_verification_source",
            "tools/ace2_cross_layer_error_carry_reference.py",
        ),
        generated_record(
            "verification/generated/cross_layer_error_carry_vectors.svh",
            "tools/gen_cross_layer_error_carry_vectors.py",
            "generated_verification_source",
            "tools/ace2_cross_layer_error_carry_reference.py",
        ),
        generated_record(
            "reference/generated/cross_layer_error_carry_metadata.json",
            "tools/gen_cross_layer_error_carry_metadata.py",
            "generated_model_image_metadata_manifest",
        ),
        generated_record(
            "reference/generated/cross_layer_error_carry_metadata.bin",
            "tools/gen_cross_layer_error_carry_metadata.py",
            "generated_model_image_metadata_binary",
        ),
    ]

    manifest = load("design/RTL_MANIFEST.json")
    manifest.update({
        "architecture_contract_status": "manager_advanced_to_rtl_successor_implemented",
        "candidate_id": packet["candidate_id"],
        "candidate_layer_operator": CONTRACT,
        "candidate_status": "bounded_rtl_preflight_pass_independent_review_pending",
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "candidate_rtl_hash_scope": packet["candidate_rtl_hash_scope"],
        "candidate_capability_accepted": False,
        "candidate_implementation_provenance": copy.deepcopy(implementation_provenance),
        "candidate_rtl_sources": [rtl_record],
        "candidate_generated_sources": generated,
        "candidate_generated_hashes": {record["path"]: record["sha256"] for record in generated},
        "candidate_source_hashes": packet["source_hashes"],
        "candidate_interface": {
            "descriptor_or_csr_change": False,
            "planned_shell_flag_bit": 7,
            "modules": packet["modules"],
            "interfaces": packet["interfaces"],
            "status": "standalone_modules_implemented_linted_elaborated_not_shell_admitted",
        },
        "candidate_model_metadata": {
            "status": "generated_and_hash_bound_before_quality",
            "schema": "QECR-1",
            "allocation_bytes": 88512,
            "immutable_image_bytes": 86592,
            "carry_buffer_bytes": 1792,
            "runtime_state_bytes": 128,
            "manifest": artifact("reference/generated/cross_layer_error_carry_metadata.json"),
            "binary": artifact("reference/generated/cross_layer_error_carry_metadata.bin"),
            "quality_tuning_permitted": False,
        },
        "candidate_schedule": {
            "producer_valid_cycles_per_output_lane": 26,
            "producer_cycles_per_layer": 23296,
            "rmsnorm_square_cycles": 896,
            "status": "producer_cycle_exact_in_rtl_simulation_rmsnorm_serial_schedule_implemented_no_end_to_end_throughput_claim",
        },
        "candidate_geometry": {
            "incremental_external_bytes_per_token": 0,
            "candidate_allocation_bytes": 88512,
            "planned_sram_peak_bytes": 466112,
            "remaining_sram_bytes": 58176,
        },
        "candidate_meets_numeric_acceptance": False,
        "candidate_verification_complete": False,
        "candidate_verification_binding": None,
        "candidate_supported_layer_operator_prefix_after_review": PREFIX,
        "candidate_first_unsupported_layer_operator_after_review": FIRST_UNSUPPORTED,
        "candidate_review_binding": {
            "candidate_capability_accepted": False,
            "reviewer_status": "pending",
            "scope": "standalone_candidate_rtl_checklist_only",
            "stage_closing": False,
            "precheck": artifact(PRECHECK),
            "prior_planner_draft_review": "archived_historical_not_live_acceptance",
        },
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "proposed_replacement_contract": contract,
        "interfaces_contract_status": "qecr_standalone_interfaces_implemented_review_pending",
        "claim_boundaries": packet["claim_boundary"],
        "current_stage": "rtl",
        "stage": "rtl",
        "stage_closing": False,
    })
    manifest["traceability"] = {
        **CHECKLIST,
        "stage_checklist": copy.deepcopy(CHECKLIST),
        "selected_mechanism": CONTRACT,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "architecture_contract_gap": {
            "status": "closed_by_hash_bound_successor_rtl_preflight_review_pending",
            "resolution_owner": "fresh_independent_l2_reviewer",
        },
    }
    manifest["independent_reviewer_acceptance"] = {
        "architecture_review": "accepted",
        "environment_review": "accepted_manager_advanced_to_rtl",
        "rtl_review": "pending_fresh_independent_review",
        "status": "rtl_preflight_pass_review_pending",
    }
    provenance_entries = manifest.setdefault("ip_provenance", [])
    provenance_names = {entry.get("name") for entry in provenance_entries}
    for module in packet["modules"]:
        matching = [entry for entry in provenance_entries if entry.get("name") == module["name"]]
        if matching:
            for entry in matching:
                entry["kind"] = "new_project_rtl"
                entry["source_revision"] = rtl_record["sha256"]
                entry["license"] = "repository project license not separately declared in this manifest"
                entry["third_party"] = False
                entry["producer_role"] = producer_role
                entry.pop("planner_draft_provenance", None)
                entry["superseded_planner_draft_provenance"] = copy.deepcopy(planner_archive)
        else:
            provenance_entries.append({
                "name": module["name"],
                "kind": "new_project_rtl",
                "source_revision": rtl_record["sha256"],
                "license": "repository project license not separately declared in this manifest",
                "third_party": False,
                "producer_role": producer_role,
                "superseded_planner_draft_provenance": copy.deepcopy(planner_archive),
            })
    for record in generated:
        matching = [entry for entry in provenance_entries if entry.get("name") == record["path"]]
        if matching:
            for entry in matching:
                entry.update({
                    "kind": record["kind"],
                    "generator": record["generator"],
                    "regeneration_command": record["regeneration_command"],
                    "source_revision": record["sha256"],
                    "third_party": False,
                    "producer_role": producer_role,
                    "superseded_planner_draft_provenance": copy.deepcopy(planner_archive),
                })
                entry.pop("planner_draft_provenance", None)
        else:
            provenance_entries.append({
                "name": record["path"],
                "kind": record["kind"],
                "generator": record["generator"],
                "regeneration_command": record["regeneration_command"],
                "source_revision": record["sha256"],
                "third_party": False,
                "producer_role": producer_role,
                "superseded_planner_draft_provenance": copy.deepcopy(planner_archive),
            })
    dump("design/RTL_MANIFEST.json", manifest)

    policy = load("design/FAST_LOOP_POLICY.json")
    policy["active_architecture_contract"] = copy.deepcopy(contract)
    policy["selected_replacement_contract"] = copy.deepcopy(contract)
    policy["manager_recommendation"] = contract["required_manager_action"]
    dump("design/FAST_LOOP_POLICY.json", policy)

    target = load("design/TARGET.json")
    target["generated_at_utc"] = now
    target["delivery_contract"]["architecture_review_status"] = "accepted"
    target["current_architecture_contract"] = copy.deepcopy(contract)
    target["fast_loop_contract"]["environment_readiness"] = "accepted_manager_advanced_to_rtl"
    target["fast_loop_contract"]["rtl_status"] = "bounded_preflight_pass_independent_review_pending"
    target["non_claims"] = [
        "no successor quality or full verification result",
        "no successor shell admission or full shell regression",
        "no successor synthesis PPA or power result",
        "no FPGA prototype or benchmark comparison",
        "no routed GDS DRC LVS or signoff result",
        "no tapeout readiness or silicon claim",
    ]
    dump("design/TARGET.json", target)

    scope = load("design/CHIP_SCOPE.json")
    authority = scope.setdefault("authority_override", {})
    authority.update({
        "active_replacement_contract": CONTRACT,
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "operator_approval_consumed": True,
        "required_next_action": "fresh_independent_rtl_checklist_review",
        "stage_closing": False,
        "status": "bounded_rtl_preflight_pass_independent_review_pending",
    })
    approval = authority.setdefault("operator_implementation_approval", {})
    frozen_at = approval.get("frozen_at_utc")
    approval.update(copy.deepcopy(contract))
    if frozen_at:
        approval["frozen_at_utc"] = frozen_at
    execution_policy = scope.setdefault("operator_owned_execution_policy", {})
    active_successor = execution_policy.setdefault("active_successor_contract", {})
    successor_frozen_at = active_successor.get("frozen_at_utc")
    active_successor.update(copy.deepcopy(contract))
    if successor_frozen_at:
        active_successor["frozen_at_utc"] = successor_frozen_at
    if isinstance(scope.get("implementation_frontier"), dict):
        scope["implementation_frontier"]["latest_decision"] = "cross_layer_error_carry_rtl_preflight_pass"
        scope["implementation_frontier"]["candidate_rtl_hash"] = packet["candidate_rtl_hash"]
        scope["implementation_frontier"]["candidate_rtl_hash_scope"] = packet["candidate_rtl_hash_scope"]
        scope["implementation_frontier"]["current_mode"] = "ADVANCE"
        scope["implementation_frontier"]["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
    scope["stage"] = {
        "current_stage": "rtl",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_evidence": [
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
            PRECHECK.relative_to(ROOT).as_posix(),
        ],
        "current_stage_status": "rtl_preflight_pass_independent_review_pending",
        "downstream_stages_locked_until_manager_advance": [
            "verification", "ppa", "prototype", "benchmark", "signoff",
        ],
        "planner_may_advance_stage": False,
        "stage_closing": False,
        "stage_transition_owner": "Manager",
    }
    scope["last_updated_utc"] = now
    dump("design/CHIP_SCOPE.json", scope)

    trace = f"""# ACE-2 RTL traceability notes

## Current standalone QECR successor

- Contract: `{CONTRACT}`.
- Candidate: `{packet['candidate_id']}`; ordered aggregate hash
  `{packet['candidate_rtl_hash']}`.
- `ace2_quantization_error_carry_lane_core` implements the frozen Scale32
  common-exponent producer, signed-int8 no-saturation rule, exact residual
  error, signed-Q0.15 carry, ties-to-even rounding, and 26-cycle valid latency.
- `ace2_error_carry_state_core` implements one 896-entry signed-16 carry
  buffer, atomic validity, producer layer/token/model/tag identity, stale-order
  rejection, accepted-consumer completion-tag capture and terminal-backpressure
  stability, and clear-on-terminal-consumer semantics. Carry memory contents
  are not reset; reset clears validity, which preserves RAM inference and makes
  stale contents architecturally unreachable.
- `ace2_carry_aware_rmsnorm_core` implements signed-24 Q8.15 reconstruction,
  a one-lane 24x24 square pass, unsigned-56 accumulation, ties-even division by
  hidden width, ceil square root, floor reciprocal, serial scale multiply,
  ties-even output rounding, and fail-closed saturation status. Its streamed
  signed-Q7.8 gain is the output-scale-folded metadata
  `round(weight / output_scale * 2^8)`, not a raw model gain. The deterministic
  vectors derive that metadata from explicit weights/output scale and the
  scalar regression includes a 896-lane non-collapse case.
- Verilator lint is warning-free for all three tops. Icarus elaboration and the
  integrated testbench pass producer boundary/tie/overflow vectors,
  backpressure stability, atomic state commit/read/clear, and RMSNorm aggregate
  values. A depth-3 Yosys SAT check covers reset, error exclusivity, range,
  latency bounds, and stall stability; the 26-cycle arithmetic schedule is
  checked cycle-exactly in simulation.
- All RTL and generated collateral are first-party, hash-bound, and have
  deterministic regeneration commands in `design/RTL_MANIFEST.json`.
- Fresh independent RTL checklist review is pending. The shell is not modified
  or admitted, the accepted prefix remains through `layer_0.v_proj`, and first
  unsupported remains `layer_0.rope_q`.

No quality discriminator, shell regression, candidate PPA/power, physical
design, prototype, benchmark, signoff, tapeout-readiness, or silicon run is
claimed.
"""
    (ROOT / "design/RTL_TRACEABILITY.md").write_text(trace, encoding="utf-8")

    checkpoint = f"""# Goal

Implement the Manager-authorized bounded RTL successor
`{CONTRACT}` without entering a downstream stage.

# Current state

The Manager-owned stage is `rtl`. Candidate `{packet['candidate_id']}` is
implemented at aggregate hash `{packet['candidate_rtl_hash']}`. The bounded
preflight supports all three RTL checklist items; fresh independent RTL review
is still pending. Implementation authority is consumed and fail-closed.

# Verified RTL evidence

- Seven exact scalar checks and deterministic QECR schema-1 metadata/vectors
  reproduce, including output-scale-folded RMSNorm gains and a 896-lane
  non-collapse case.
- All three standalone synthesizable modules lint without warnings and elaborate.
- Integrated Icarus simulation passes producer arithmetic, 26-cycle latency,
  output backpressure, atomic carry state/identity, accepted consumer-tag
  stability, exactly-once terminal clear, and carry-aware RMSNorm
  aggregate/output checks.
- Bounded Yosys SAT invariants pass at depth 3.

# Preserved frontier and locks

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
`layer_0.rope_q`; mode remains `ADVANCE`. The 2.0 mm2 non-SRAM cap, 100 MHz
floor, 128-bit memory boundary, and historical 62,199-cell / 0.6108746272-mm2 /
+0.1502-ns frontier are unchanged. Historical PPA is not successor PPA.

Quality/full verification, shell admission/regression, PPA/power, physical
design, prototype, benchmark, and signoff were not run because they are locked
downstream of the current RTL stage.

# Next gate

Run one fresh independent RTL-checklist review against
`evidence/{CONTRACT}/latest/PRECHECK.json`. Planner and Engineer must not edit
`research/PIPELINE_STATE.json`.
"""
    (ROOT / "CHECKPOINT.md").write_text(checkpoint, encoding="utf-8")

    status = load("research/PUBLIC_STATUS.json")
    architecture_status = "architecture_accepted_rtl_implemented_review_pending"
    status.setdefault("architecture_certification_mission", {}).update({
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "projection_scope": "accepted_architecture_bound_to_active_rtl_candidate",
        "status": architecture_status,
    })
    status.setdefault("architecture_proposal_gate", {}).update({
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "required_manager_action": "hold_rtl_for_fresh_independent_rtl_checklist_review",
        "stage_closing": False,
        "status": architecture_status,
    })
    status["architecture_successor_dispatch"] = {
        "contract_id": CONTRACT,
        "current_stage": "rtl",
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "operator_approval_consumed": True,
        "required_manager_action": contract["required_manager_action"],
        "stage_closing": False,
        "status": contract["status"],
    }
    if isinstance(status.get("latest_environment_stage"), dict):
        status["latest_environment_stage"].update({
            "implementation_authorized": False,
            "implementation_authority_consumed": True,
            "status": "manager_advanced_to_rtl",
        })
    candidate = {
        "contract_id": CONTRACT,
        "mechanism": contract["mechanism"],
        "candidate_id": packet["candidate_id"],
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "operator_approval_consumed": True,
        "quality_status": "unrun_for_successor",
        "shell_status": "not_admitted",
        "ppa_status": "unrun",
        "status": "bounded_rtl_preflight_pass_independent_review_pending",
        "planned_carry_cycles_per_layer": 23296,
        "planned_sram_peak_bytes": 466112,
        "remaining_sram_bytes": 58176,
        "incremental_external_bytes_per_token": 0,
    }
    latest_rtl_candidate = {
        "candidate_capability_accepted": False,
        "candidate_id": packet["candidate_id"],
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "candidate_rtl_hash_scope": packet["candidate_rtl_hash_scope"],
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "required_manager_action": contract["required_manager_action"],
        "shell_admitted": False,
        "stage_closing": False,
        "status": contract["status"],
    }
    status["latest_rtl_candidate"] = copy.deepcopy(latest_rtl_candidate)
    status["required_manager_action"] = contract["required_manager_action"]
    for key in ("dashboard_fields", "implementation_frontier"):
        section = status.setdefault(key, {})
        section["candidate_mechanism"] = copy.deepcopy(candidate)
        section["candidate_rtl_hash"] = packet["candidate_rtl_hash"]
        section["candidate_rtl_hash_scope"] = packet["candidate_rtl_hash_scope"]
        section["candidate_meets_numeric_acceptance"] = False
        section["current_stage"] = "rtl"
        section["current_mode"] = "ADVANCE"
        section["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
        section["ordered_supported_layer_operator_prefix"] = PREFIX
        section["implementation_authorized"] = False
        section["implementation_authority_consumed"] = True
        section["implementation_completed"] = True
        section["latest_decision"] = "cross_layer_error_carry_rtl_preflight_pass"
        section["required_manager_action"] = contract["required_manager_action"]
        section["routing_status"] = "fresh_independent_rtl_review_pending"
        section["rtl_contract_traceability"] = True
        section["latest_rtl_candidate"] = copy.deepcopy(latest_rtl_candidate)
        if isinstance(section.get("latest_environment_stage"), dict):
            section["latest_environment_stage"].update({
                "implementation_authorized": False,
                "implementation_authority_consumed": True,
                "status": "manager_advanced_to_rtl",
            })
        operator_policy = section.get("operator_policy")
        if isinstance(operator_policy, dict):
            active_contract = operator_policy.setdefault("active_architecture_contract", {})
            active_contract.update(copy.deepcopy(contract))
    status["selected_replacement_contract"] = copy.deepcopy(contract)
    status["latest_decision"] = "cross_layer_error_carry_rtl_preflight_pass"
    status["routing_authorized"] = "fresh_independent_rtl_review_only"
    status["routing_status"] = "fresh_independent_rtl_review_pending"
    status["rtl_contract_traceability"] = True
    status["stage_closing"] = False
    status["generated_at_utc"] = now
    status["last_updated_utc"] = now
    status["blockers"] = [{
        "id": "fresh_independent_rtl_review_pending",
        "owner": "reviewer",
        "severity": "gate",
        "status": "open",
        "detail": "Bounded RTL preflight passes; one fresh independent review must rule on the exact candidate hash.",
    }]
    status["stage"] = {
        "completed_prior_stages": ["definition", "architecture", "environment"],
        "current_stage": "rtl",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_evidence": [
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
            PRECHECK.relative_to(ROOT).as_posix(),
        ],
        "current_stage_status": "rtl_preflight_pass_independent_review_pending",
        "planner_may_advance_stage": False,
        "stage_closing": False,
        "stage_transition_owner": "Manager",
    }
    status.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    status["integrity"]["canonical_sha256"] = None
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump("research/PUBLIC_STATUS.json", status)

    print(
        "ACE2_QECR_RTL_BINDING_PASS "
        f"candidate={packet['candidate_id']} aggregate={packet['candidate_rtl_hash']} review=pending"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
