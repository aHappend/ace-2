#!/usr/bin/env python3
"""Bind and validate the Planner-authored dynamic Scale32 architecture proposal.

This script never edits research/PIPELINE_STATE.json and never authorizes RTL or
downstream execution.  It publishes an architecture-only proposal for a Manager
freeze decision while preserving the sealed predecessor and the certified
candidate-independent baseline harness.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_token_group_dynamic_scale32_v1"
SEALED_PREDECESSOR = "cross_layer_quantization_error_carry_final_output_v1"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
CHECKLIST = {
    "architecture.compute-memory-model": True,
    "architecture.interface-control": True,
    "architecture.leverage-risk": True,
    "architecture.area-reuse-plan": True,
}

POLICY = ROOT / "design/FAST_LOOP_POLICY.json"
MEMORY = ROOT / "design/MEMORY_MODEL.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
LIVE = ROOT / ".argus/live-view.json"
PPA_LEDGER = ROOT / "design/PPA_FRONTIER_LEDGER.json"
PROPOSAL_DIR = ROOT / "evidence" / CONTRACT / "architecture"
PROPOSAL = PROPOSAL_DIR / "PROPOSAL.json"
PROPOSAL_SUM = PROPOSAL_DIR / "PROPOSAL.sha256"
HARNESS_DIR = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1"
PUBLIC_CERTIFICATION = HARNESS_DIR / "PUBLIC_CERTIFICATION.json"
HARNESS_REPORT = HARNESS_DIR / "HARNESS_TEST_REPORT.json"
HARNESS_MANIFEST = HARNESS_DIR / "MANIFEST.json"
HARNESS_L2_REVIEW = HARNESS_DIR / "L2_HARNESS_REVIEW.json"
MANAGER_RECOVERY_SEAL = (
    ROOT
    / "evidence/cross_layer_quantization_error_carry_final_output_v1"
    / "recovery/manager_architecture_rollback_v1/SEAL.json"
)
HARNESS_PROPERTIES = [
    "stage_authorization_before_execution",
    "durable_exactly_one_reservation_and_run_ledger",
    "atomic_artifact_and_companion_hash_commit",
    "exact_provenance_binding",
    "candidate_entrypoint_interlock",
    "crash_recovery",
    "missing_or_mismatched_artifact_fail_closed",
]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def bandwidth_sweep(weight_bytes: int, compute_cycles: int) -> list[dict[str, Any]]:
    rows = []
    for bandwidth in (1, 2, 4, 8, 16):
        payload = math.ceil(weight_bytes / bandwidth)
        dma = payload + 64
        if bandwidth < 2:
            bottleneck = "memory"
        elif bandwidth == 2:
            bottleneck = "compute_memory_balance"
        else:
            bottleneck = "compute"
        rows.append({
            "bytes_per_cycle": bandwidth,
            "weight_payload_cycles": payload,
            "dma_cycles_with_64_cycle_setup_hook": dma,
            "nonoverlapped_layer_cycles": dma + compute_cycles,
            "roofline_bottleneck": bottleneck,
        })
    return rows


def baseline_harness_binding() -> dict[str, Any]:
    certification = load(PUBLIC_CERTIFICATION)
    evidence_hashes = certification.get("evidence_hashes", {})
    require(certification.get("decision") == "PASS", "baseline harness certification is not PASS")
    require(
        certification.get("status") == "independently_certified_pass",
        "baseline harness certification status differs",
    )
    require(
        certification.get("candidate_or_model_execution_count") == 0,
        "baseline harness certification contains an execution claim",
    )
    require(
        certification.get("implementation_authorized") is False,
        "baseline harness certification authorizes implementation",
    )
    require(
        set(certification.get("required_properties", [])) == set(HARNESS_PROPERTIES),
        "baseline harness property set differs",
    )
    exact_evidence = {
        "report": artifact(HARNESS_REPORT),
        "manifest": artifact(HARNESS_MANIFEST),
        "independent_l2_review": artifact(HARNESS_L2_REVIEW),
        "manager_recovery_seal": artifact(MANAGER_RECOVERY_SEAL),
    }
    require(
        exact_evidence["report"]["sha256"] == evidence_hashes.get("report_sha256"),
        "baseline harness report hash differs",
    )
    require(
        exact_evidence["manifest"]["sha256"] == evidence_hashes.get("manifest_sha256"),
        "baseline harness manifest hash differs",
    )
    require(
        exact_evidence["independent_l2_review"]["sha256"]
        == evidence_hashes.get("independent_l2_review_sha256"),
        "baseline harness L2 hash differs",
    )
    require(
        exact_evidence["manager_recovery_seal"]["sha256"]
        == evidence_hashes.get("manager_recovery_seal_sha256"),
        "Manager recovery seal hash differs",
    )
    return {
        "public_certification": artifact(PUBLIC_CERTIFICATION),
        "decision": "PASS",
        "status": "independently_certified_pass",
        "candidate_or_model_execution_count": 0,
        "implementation_authorized": False,
        "required_properties_in_execution_order": HARNESS_PROPERTIES,
        "exact_evidence": exact_evidence,
        "claim_boundary": certification["claim_boundary"],
    }


def build_proposal(now: str) -> dict[str, Any]:
    hidden = 896
    intermediate = 4864
    q_heads = 14
    kv_heads = 2
    head_width = 64
    layers = 24
    lanes = 4

    projection_macs = {
        "q_proj": hidden * hidden,
        "k_proj": hidden * (kv_heads * head_width),
        "v_proj": hidden * (kv_heads * head_width),
        "o_proj": hidden * hidden,
        "mlp_gate_proj": hidden * intermediate,
        "mlp_up_proj": hidden * intermediate,
        "mlp_down_proj": intermediate * hidden,
    }
    macs = sum(projection_macs.values())
    weight_bytes = sum(value // 2 for value in projection_macs.values())
    compute_cycles = math.ceil(macs / lanes)

    partial_tag_events = (
        hidden * 7
        + (kv_heads * head_width) * 7
        + (kv_heads * head_width) * 7
        + hidden * 7
        + intermediate * 7
        + intermediate * 7
        + hidden * 38
    )
    group_counts = {
        "input_rmsnorm_output": 7,
        "q_projection_output_by_head": 14,
        "k_projection_output_by_head": 2,
        "v_projection_output_by_head": 2,
        "attention_value_output": 7,
        "o_projection_output": 7,
        "attention_residual_output": 7,
        "post_attention_rmsnorm_output": 7,
        "mlp_gate_output": 38,
        "mlp_up_output": 38,
        "silu_gate_output": 38,
        "mlp_down_output": 7,
        "mlp_residual_output": 7,
    }
    groups_per_layer = sum(group_counts.values())
    vector_cycles = (
        (q_heads + 2 * kv_heads) * 4
        + (groups_per_layer - q_heads - 2 * kv_heads) * 8
        + groups_per_layer
    )

    return {
        "schema_version": 2,
        "contract_id": CONTRACT,
        "status": "planner_architecture_corrected_ready_for_fresh_independent_review",
        "generated_at_utc": now,
        "authority": {
            "author": "L4_Planner",
            "successor_frozen": False,
            "implementation_authorized": False,
            "stage_closing": False,
            "required_manager_action": "hold_freeze_until_fresh_independent_architecture_review_then_freeze_this_successor_or_record_architecture_no_go",
            "manager_owned_stage_unchanged": "architecture",
            "operator_targets_unchanged": {
                "non_sram_area_cap_mm2": 2.0,
                "frequency_floor_mhz": 100.0,
                "memory_boundary_bits": 128,
                "quality_limit_perplexity_ratio": 1.05,
            },
        },
        "frontier": {
            "mode": "ADVANCE",
            "ordered_supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "accepted_prefix_advanced": False,
            "new_ppa_run": False,
            "historical_ppa_ledger": artifact(PPA_LEDGER),
        },
        "baseline_harness_certification": baseline_harness_binding(),
        "execution_gates": {
            "no_execution_claim": True,
            "rtl_stage": {
                "stage_closing": False,
                "allowed_only_after_manager_freeze_and_stage_advance": True,
                "allowed_checks": [
                    "reference_and_vector_generation",
                    "lint",
                    "elaboration",
                    "bit_exact_simulation",
                    "interface_checks",
                    "minimal_formal",
                ],
                "prohibited": [
                    "disabled_mode_baseline_execution",
                    "candidate_quality_execution",
                    "full_shell_regression",
                    "canonical_sky130_ppa",
                ],
            },
            "verification_stage": {
                "allowed_only_after_separate_manager_transition": True,
                "tasks_in_strict_order": [
                    {
                        "task": 1,
                        "name": "exactly_one_disabled_mode_baseline_via_certified_harness",
                        "execution_limit": 1,
                        "independent_review_required": True,
                        "candidate_entrypoint_interlocked": True,
                    },
                    {
                        "task": 2,
                        "name": "exactly_one_candidate_quality_run",
                        "execution_limit": 1,
                        "entry_condition": "independent_task_1_baseline_PASS",
                        "independent_review_required": True,
                    },
                ],
                "full_shell_regression": (
                    "prohibited_until_independent_candidate_quality_GO_and_fresh_separate_Manager_authorization"
                ),
            },
            "ppa_stage": {
                "fresh_canonical_sky130_ppa": (
                    "prohibited_until_independent_candidate_quality_GO_full_shell_acceptance_and_separate_Manager_transition_to_ppa"
                ),
            },
            "downstream_release_stages": (
                "prototype_full_benchmark_and_signoff_reserved_for_complete_workload_or_model_system_release_milestone"
            ),
        },
        "selection": {
            "mechanism": "token_and_tensor_group_dynamic_power_of_two_Scale32_activation_scaling",
            "scope": "all_24_layers_plus_final_rmsnorm_and_lm_head_activation_boundaries",
            "structurally_distinct_from": [
                "attention_score_storage_or_softmax_representations",
                "per_head_qk_static_or_dynamic_scale_only",
                "qk_or_v_residual_cross_terms",
                "down_projection_residual_fusion",
                "cross_layer_quantization_error_carry",
            ],
            "evidence_basis": [
                "attention-local mechanisms repeatedly improved layer-0 score/value metrics without preserving C4 final-output improvement",
                "the sealed QECR direction never produced an admissible candidate run and cannot be reused",
                "a recurrent activation-boundary mechanism has higher Amdahl leverage than another attention-local correction",
            ],
        },
        "numeric_contract": {
            "stored_mantissa": "signed_int8_symmetric_range_minus127_to_plus127",
            "reserved_value": -128,
            "base_scale": "existing_immutable_normalized_Scale32_for_the_tensor_group",
            "effective_scale": "base_Scale32_significand_with_exponent_plus_dynamic_delta",
            "dynamic_delta_range": [-24, 24],
            "effective_scale32_exponent_range": [-24, 4],
            "grouping": {
                "hidden_and_intermediate_tensors": 128,
                "q_k_v_rope_sensitive_tensors": 64,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_width": head_width,
            },
            "selection_rule": (
                "choose the smallest legal integer delta for which every group lane rounded-to-nearest-ties-to-even "
                "into effective_scale is in [-127,127]; all-zero group uses delta zero"
            ),
            "projection_rule": (
                "form signed32 W4A8 partial sums per 128-input group, attach the exact effective activation and "
                "weight Scale32 product, align in one shared tagged wide accumulator, and round only at final requantization"
            ),
            "residual_rule": "align both input group Scale32 values before add, then select one fresh output delta",
            "rmsnorm_rule": "square group mantissas, align squared group sums by twice the effective exponent, then apply the frozen integer RMSNorm rounding and saturation",
            "silu_rule": "multiply gate/up mantissas, add effective exponents, use the existing integer SiLU approximation, then select one fresh output delta",
            "rope_rule": "each 64-lane head shares one effective Scale32 so every RoPE pair has identical scale and rotation remains integer Q1.15",
            "kv_rule": "write dynamic normalized Scale32 per layer, token, K/V head in the already-required four-byte scale records",
            "failure": "invalid header, illegal exponent, reserved mantissa, overflow, or identity mismatch is numeric_overflow_or_descriptor_error_with_no_success",
            "rounding": "round_to_nearest_ties_to_even",
            "floating_point_rtl": False,
        },
        "compute_memory_model": {
            "representative_shape": "one_batch1_decoder_layer_token_excluding_attention_context_length_dependent_dot_products",
            "projection_macs": projection_macs,
            "projection_macs_per_layer_token": macs,
            "projection_lanes": lanes,
            "projection_compute_cycles_per_layer_token": compute_cycles,
            "packed_w4_weight_bytes_per_layer": weight_bytes,
            "weight_only_arithmetic_intensity_macs_per_byte": macs / weight_bytes,
            "compute_memory_balance_bytes_per_cycle": weight_bytes / compute_cycles,
            "tagged_partial_events_per_layer": partial_tag_events,
            "shared_aligner_utilization_percent": 100.0 * partial_tag_events / compute_cycles,
            "dynamic_groups_per_layer": groups_per_layer,
            "dynamic_group_finalize_cycles_planning_hook": vector_cycles,
            "dynamic_finalize_fraction_of_projection_cycles_percent": 100.0 * vector_cycles / compute_cycles,
            "bandwidth_sweep": bandwidth_sweep(weight_bytes, compute_cycles),
            "claim_boundary": "architecture equations and zero-backpressure planning hooks only; no candidate RTL latency, tokens_per_second, PPA, or power measurement",
        },
        "memory_model": {
            "external_stream_width_bits": 128,
            "internal_noc": "none",
            "sram_banks": 8,
            "sram_capacity_bytes": 524288,
            "preserved_noncandidate_live_set_bytes": 377600,
            "wide_group_ping_pong_bytes": 1280,
            "maximum_simultaneous_64_byte_tensor_sidecars": 13,
            "tensor_sidecar_live_bytes": 832,
            "runtime_state_bytes": 256,
            "proposal_incremental_sram_bytes": 2368,
            "planned_peak_sram_bytes": 379968,
            "remaining_sram_margin_bytes": 144320,
            "tensor_sidecar_bytes": 64,
            "maximum_sidecar_exponents": 38,
            "worst_case_external_sidecar_bytes_per_layer_if_every_intermediate_spills": 832,
            "incremental_kv_scale_bytes": 0,
            "kv_reason": "the frozen workload already requires one four-byte Scale32 record per layer, token, and K/V head; proposal makes those records dynamic rather than adding records",
            "maximum_cached_tokens": 32768,
            "lifetime": "one ping-pong wide group buffer, sidecar header, aligner, and control state are reused across every mutually exclusive operator and all 24 layers",
        },
        "interface_contract": {
            "descriptor_bytes": 64,
            "proposal_flag": "flags[6]=1",
            "sealed_qecr_flag": "flags[7] must remain zero",
            "tensor_payload_alignment_bytes": 64,
            "tensor_sidecar_location": "the 64 bytes immediately preceding each candidate-mode tensor payload in the same address space",
            "sidecar_layout": {
                "bytes_0_3": "ASCII BFP1",
                "byte_4": "schema_version_1",
                "byte_5": "group_lanes_64_or_128",
                "byte_6": "group_count_1_to_38",
                "byte_7": "flags_reserved_zero",
                "bytes_8_9": "producer_completion_tag_u16",
                "byte_10": "layer_id",
                "byte_11": "producer_opcode",
                "bytes_12_15": "tensor_elements_u32",
                "bytes_16_23": "model_identity64",
                "bytes_24_61": "signed_dynamic_exponent_deltas_then_zero_fill",
                "bytes_62_63": "reserved_zero",
            },
            "publication": "payload_and_sidecar_become_valid_atomically_with_successful_completion",
            "consumer_validation": "magic, schema, shape, group count, producer tag, layer/opcode, model identity, exponent range, reserved bytes, address range, and dependency must pass before payload issue",
            "backpressure": "all input, output, sidecar, DMA, and SRAM channels may stall without lane reorder, exponent change, duplicate publication, or lost sticky bits",
            "reset_clock_cdc": "single clock; asynchronous reset assertion and synchronous deassertion clear all sidecar-valid, tagged-accumulator, DMA, completion, and error state",
            "interrupt_error_completion": "existing interrupt and result codes are retained; candidate validation errors publish no successful tag or usable destination",
        },
        "roofline_amdahl_risk": {
            "leverage": "corrects every recurrent activation quantization boundary in all layers and final output rather than one attention-local tensor",
            "maximum_candidate_overhead_is_not_a_measured_result": True,
            "highest_risks": [
                "dynamic scale selection may improve local tensors but still fail C4 final-output fidelity",
                "mixed Scale32 projection accumulation must avoid repeated rounding",
                "RMSNorm square-sum exponent alignment requires proved width bounds",
                "sidecar atomicity and identity must survive backpressure, reset, and tensor reuse",
                "the shared aligner and leading-bit detector must preserve at least 100 MHz",
            ],
            "verification_strategy": [
                "independent scalar oracle for every rounding, exponent, reserved-value, overflow, and sidecar rule",
                "RTL stage limited to reference/vector generation, lint, elaboration, bit-exact simulation, interface checks, and minimal formal",
                "verification task 1 is exactly one disabled-mode baseline via the certified harness followed by independent review",
                "verification task 2 is exactly one all-24-layer two-dataset candidate run only after independent baseline PASS, followed by independent review",
                "full shell regression remains prohibited until candidate quality GO and fresh separate Manager authorization",
                "fresh canonical SKY130 PPA remains prohibited until shell acceptance and a separate Manager transition to ppa",
            ],
            "stop_rules": [
                "seal the direction if either frozen dataset fails the candidate admission gate",
                "do not retune group sizes or exponent rules under the same approval",
                "if verified area increment reaches 3 percent, perform global sharing review before capability advancement",
                "if total non-SRAM area exceeds 2.0 mm2, enter GLOBAL_COMPRESSION without relaxing the cap",
                "do not advance if Fmax is below 100 MHz",
            ],
            "fallbacks_requiring_fresh_architecture_review": [
                "uniform_128_lane_groups_including_qkv",
                "external_sidecar_stream_instead_of_prepayload_header",
                "wider_tagged_accumulator_or_two_pass_exact_alignment",
            ],
        },
        "resource_sharing": [
            {
                "resource": "W4A8_MACs",
                "selected": "reuse_existing_four_projection_lanes",
                "dedicated_alternative": "separate_dynamic_scale_projection_array",
                "expected_cycle_cost": 0,
                "area_reason": "avoids_duplicate_MAC_array",
            },
            {
                "resource": "tagged_align_accumulate",
                "selected": "one_pipelined_shared_wide_aligner_accumulator",
                "dedicated_alternative": "one_per_projection_or_operator",
                "expected_cycle_cost": "one_event_per_input_group_hidden_under_32_or_more_MAC_cycles_per_group",
                "expected_utilization_percent": 100.0 * partial_tag_events / compute_cycles,
            },
            {
                "resource": "max_abs_and_exponent_select",
                "selected": "reuse_one_16_lane_vector_max_and_leading_bit_unit",
                "dedicated_alternative": "one_detector_per_tensor_producer",
                "expected_cycle_cost": vector_cycles,
                "area_reason": "all_producers_are_lifetime_mutually_exclusive",
            },
            {
                "resource": "wide_group_buffer",
                "selected": "one_128_lane_ping_pong_buffer_reused_globally",
                "dedicated_alternative": "private_buffer_per_operator",
                "expected_cycle_cost": "second_pass_quantization_in_dynamic_group_finalize_hook",
                "selected_bytes": 1280,
            },
            {
                "resource": "round_saturate_SFU",
                "selected": "reuse_existing_vector_round_saturate_RMSNorm_and_SiLU_resources",
                "dedicated_alternative": "candidate_local_rounders_and_SFUs",
                "expected_cycle_cost": "included_in_dynamic_group_finalize_hook",
            },
            {
                "resource": "DMA_SRAM_control",
                "selected": "reuse_existing_DMA_eight_banks_descriptor_engine_interrupts_and_errors",
                "dedicated_alternative": "private_sidecar_DMA_and_control_path",
                "expected_cycle_cost": "64_byte_sidecar_transfers_only_when_tensor_is_spilled",
            },
        ],
        "architecture_checklist": CHECKLIST,
        "claim_boundary": (
            "Planner-authored architecture proposal only. It is not a Manager freeze, implementation authorization, "
            "independent review, RTL, verification, PPA, prototype, benchmark, signoff, tapeout, or silicon evidence."
        ),
    }


def public_projection(proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_id": CONTRACT,
        "status": proposal["status"],
        "successor_frozen": False,
        "implementation_authorized": False,
        "stage_closing": False,
        "mode": "ADVANCE",
        "ordered_supported_layer_operator_prefix": PREFIX,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "architecture_checklist_ready_for_manager_review": CHECKLIST,
        "required_manager_action": proposal["authority"]["required_manager_action"],
        "mechanism": proposal["selection"]["mechanism"],
        "area_cap_non_sram_mm2": 2.0,
        "frequency_floor_mhz": 100.0,
        "new_ppa_run": False,
        "no_execution_claim": proposal["execution_gates"]["no_execution_claim"],
        "baseline_harness_certification": copy.deepcopy(proposal["baseline_harness_certification"]),
        "execution_gates": copy.deepcopy(proposal["execution_gates"]),
        "evidence": artifact(PROPOSAL),
        "claim_boundary": proposal["claim_boundary"],
    }


def update_files(proposal: dict[str, Any]) -> None:
    dump(PROPOSAL, proposal)
    PROPOSAL_SUM.write_text(f"{sha256(PROPOSAL)}  {PROPOSAL.name}\n", encoding="utf-8")

    memory = load(MEMORY)
    memory["planner_successor_proposal"] = copy.deepcopy(proposal)
    dump(MEMORY, memory)

    policy = load(POLICY)
    next_successor = policy.setdefault("next_successor", {})
    next_successor["status"] = "not_frozen"
    next_successor["architecture_checklist_closed"] = False
    next_successor["planner_proposal"] = public_projection(proposal)
    policy["planner_architecture_proposal"] = copy.deepcopy(proposal)
    dump(POLICY, policy)

    public = load(PUBLIC)
    projection = public_projection(proposal)
    public["architecture_successor_proposal"] = copy.deepcopy(projection)
    dispatch = public.setdefault("architecture_successor_dispatch", {})
    dispatch.update({
        "proposal_id": CONTRACT,
        "proposal_status": proposal["status"],
        "proposal_evidence": artifact(PROPOSAL),
        "successor_frozen": False,
        "implementation_authorized": False,
        "stage_closing": False,
    })
    for name in ("dashboard_fields", "implementation_frontier"):
        section = public.setdefault(name, {})
        section["architecture_successor_proposal"] = copy.deepcopy(projection)
        section["proposed_successor_id"] = CONTRACT
        section["proposal_status"] = proposal["status"]
    for blocker in public.get("blockers", []):
        if blocker.get("id") == "manager_structurally_distinct_successor_freeze_required":
            blocker["detail"] = (
                "The candidate-independent harness is certified and a complete Planner architecture proposal for "
                f"{CONTRACT} is available. The failed QECR contract remains sealed; Manager freeze or architecture no-go is still required."
            )
            blocker.setdefault("evidence", []).append(artifact(PROPOSAL))

    records = {
        item.get("path"): item
        for item in public.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path")
    }
    for path in (
        ROOT / "MISSION.md",
        ROOT / "CHECKPOINT.md",
        ROOT / "Makefile",
        ROOT / "design/ARCHITECTURE.md",
        ROOT / "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        ROOT / "design/SPEC.md",
        PUBLIC_CERTIFICATION,
        HARNESS_REPORT,
        HARNESS_MANIFEST,
        HARNESS_L2_REVIEW,
        MANAGER_RECOVERY_SEAL,
        MEMORY,
        POLICY,
        PROPOSAL,
        PROPOSAL_SUM,
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
        "proposal_status": proposal["status"],
        "proposal_evidence": PROPOSAL.relative_to(ROOT).as_posix(),
        "successor_frozen": False,
        "implementation_authorized": False,
        "reason": "complete architecture proposal available; Manager freeze or architecture no-go pending",
    })
    dump(LIVE, live)


def validate() -> None:
    require(PROPOSAL.is_file() and PROPOSAL_SUM.is_file(), "proposal artifacts are missing")
    fields = PROPOSAL_SUM.read_text(encoding="utf-8").strip().split()
    require(fields == [sha256(PROPOSAL), PROPOSAL.name], "proposal companion hash differs")
    proposal = load(PROPOSAL)
    require(proposal.get("contract_id") == CONTRACT, "proposal contract differs")
    require(proposal.get("architecture_checklist") == CHECKLIST, "proposal checklist differs")
    require(proposal.get("authority", {}).get("successor_frozen") is False, "proposal self-freezes")
    require(proposal.get("authority", {}).get("implementation_authorized") is False, "proposal authorizes implementation")
    require(proposal.get("execution_gates", {}).get("no_execution_claim") is True, "proposal makes an execution claim")
    require(
        proposal.get("baseline_harness_certification") == baseline_harness_binding(),
        "proposal baseline harness binding differs",
    )
    rtl_gate = proposal.get("execution_gates", {}).get("rtl_stage", {})
    require(
        rtl_gate.get("allowed_checks")
        == [
            "reference_and_vector_generation",
            "lint",
            "elaboration",
            "bit_exact_simulation",
            "interface_checks",
            "minimal_formal",
        ],
        "RTL-stage allowed check set differs",
    )
    require(
        {item.get("name") for item in proposal.get("execution_gates", {}).get("verification_stage", {}).get("tasks_in_strict_order", [])}
        == {
            "exactly_one_disabled_mode_baseline_via_certified_harness",
            "exactly_one_candidate_quality_run",
        },
        "verification task split differs",
    )
    require(
        "full shell regression before one fresh canonical SKY130 PPA inside the bounded RTL task"
        not in PROPOSAL.read_text(encoding="utf-8"),
        "stale collapsed RTL/shell/PPA plan remains",
    )
    require(load(PIPELINE).get("current_stage") == "architecture", "Manager-owned stage changed")
    pipeline_successor = load(PIPELINE).get("successor", {})
    require(pipeline_successor.get("id") == SEALED_PREDECESSOR, "sealed predecessor identity changed")
    require(pipeline_successor.get("successor_freeze_authorized") is False, "pipeline unexpectedly authorizes freeze")

    memory = load(MEMORY)
    require(memory.get("planner_successor_proposal", {}).get("contract_id") == CONTRACT, "memory proposal differs")
    policy = load(POLICY)
    require(policy.get("next_successor", {}).get("status") == "not_frozen", "policy incorrectly freezes successor")
    require(policy.get("next_successor", {}).get("planner_proposal", {}).get("contract_id") == CONTRACT, "policy proposal differs")
    require(policy.get("mode") == "ADVANCE", "operator mode changed")
    require(policy.get("area_cap_non_sram_mm2") == 2.0, "operator area cap changed")
    require(policy.get("frequency_floor_mhz") == 100.0, "operator frequency floor changed")
    require(policy.get("advance_continue_delta_area_threshold_percent") == 3.0, "ADVANCE threshold changed")
    require(policy.get("compression_continue_threshold_percent") == 3.0, "compression threshold changed")
    public = load(PUBLIC)
    require(public.get("architecture_successor_proposal", {}).get("contract_id") == CONTRACT, "public proposal differs")
    require(public.get("stage", {}).get("current_stage_checklist") == {key: False for key in CHECKLIST}, "public stage checklist was closed without Manager freeze")
    require(public.get("ordered_supported_layer_operator_prefix") == PREFIX, "supported prefix changed")
    require(public.get("first_unsupported_layer_operator") == FIRST_UNSUPPORTED, "first unsupported operator changed")
    require(public.get("current_mode") == "ADVANCE", "mode changed")
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public), "public canonical hash differs")
    records = {
        item.get("path"): item
        for item in public.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path")
    }
    for path in (
        ROOT / "CHECKPOINT.md",
        ROOT / "Makefile",
        ROOT / "design/ARCHITECTURE.md",
        ROOT / "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        ROOT / "design/SPEC.md",
        PUBLIC_CERTIFICATION,
        HARNESS_REPORT,
        HARNESS_MANIFEST,
        HARNESS_L2_REVIEW,
        MANAGER_RECOVERY_SEAL,
        MEMORY,
        POLICY,
        PROPOSAL,
        PROPOSAL_SUM,
        Path(__file__).resolve(),
    ):
        relative = path.relative_to(ROOT).as_posix()
        require(records.get(relative) == artifact(path), f"public artifact binding differs: {relative}")
    live = load(LIVE)
    require(live.get("proposal_id") == CONTRACT, "live proposal differs")
    require(live.get("successor_frozen") is False, "live view incorrectly freezes successor")
    require("shared_token_group_dynamic_scale32_v1" in (ROOT / "design/ARCHITECTURE.md").read_text(encoding="utf-8"), "architecture proposal text missing")
    require("flags[6]" in (ROOT / "design/SPEC.md").read_text(encoding="utf-8"), "interface proposal text missing")
    architecture_text = (ROOT / "design/ARCHITECTURE.md").read_text(encoding="utf-8")
    require(
        "Task 1 is exactly" in architecture_text and "disabled-mode baseline" in architecture_text,
        "architecture verification split text missing",
    )
    require(
        "PUBLIC_CERTIFICATION.json" in (ROOT / "design/ARCHITECTURE.md").read_text(encoding="utf-8"),
        "architecture certification binding text missing",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        update_files(build_proposal(now))
    validate()
    print(
        "ACE2_TOKEN_GROUP_DYNAMIC_SCALE32_PROPOSAL_CHECK_PASS "
        f"contract={CONTRACT} successor_frozen=false implementation_authorized=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
