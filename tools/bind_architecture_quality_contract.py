#!/usr/bin/env python3
"""Bind and validate the active native-accumulator attention architecture."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_native_accumulator_tagged_attention_v1"
PROPOSAL = ROOT / "design" / "NUMERICAL_REPLACEMENT_PROPOSAL.md"
PREDECESSOR = ROOT / "evidence/layer0_tile_bfp_score_attention_v1/latest/BOUNDED_NO_GO.json"
PREDECESSOR_SHA256 = "c4a9084dcfe137451a8e7b226df7990fe8a4fe6dc8f922f5b88879a251ce875c"
REVIEW = ROOT / "evidence/review/architecture_stage_closing_shared_native_accumulator_tagged_attention_v1/decision.json"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
THRESHOLDS = {
    "wikitext2_ratio_strictly_below": 13549.939049967887,
    "c4_en_512_ratio_strictly_below": 4477.990517308544,
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def artifact(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: dict[str, Any]) -> str:
    candidate = copy.deepcopy(value)
    candidate.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def review_accepted() -> bool:
    if not REVIEW.is_file():
        return False
    review = load(REVIEW)
    return (
        review.get("reviewer_status") == "done"
        and review.get("contract_id") == CONTRACT
        and all(review.get("checklist", {}).values())
    )


def contract_record(proposal_sha: str) -> dict[str, Any]:
    accepted = review_accepted()
    return {
        "authority": "operator",
        "operator_authority_source": "standing_authorization_after_tile_bfp_no_go",
        "approved_on_utc_date": "2026-08-01",
        "contract_id": CONTRACT,
        "proposal": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        "proposal_sha256": proposal_sha,
        "implementation_authorized": True,
        "operator_approval_consumed": False,
        "stage_closing": False,
        "status": (
            "architecture_independently_accepted_manager_advance_pending"
            if accepted
            else "architecture_frozen_independent_review_pending"
        ),
        "scope": "one_shared_all_24_layer_native_accumulator_tagged_attention_attempt",
        "stop_rule": (
            "all_layer_focused_layer0_score_value_and_final_lm_head_strict_"
            "improvement_then_both_unique_paired_smoke_ratios_strictly_improve"
        ),
        "thresholds": THRESHOLDS,
        "required_manager_action": (
            "advance_architecture_to_environment"
            if accepted
            else "hold_architecture_until_independent_review"
        ),
        "forbidden": [
            "planner_pipeline_stage_edit",
            "layer0_only_score_storage_retune",
            "floating_point_primary_datapath",
            "full_shell_before_focused_and_paired_smoke_pass",
            "ppa_before_focused_and_paired_smoke_pass",
            "operator_target_relaxation",
        ],
        "predecessor_bounded_no_go": {
            "contract_id": "layer0_tile_bfp_score_attention_v1",
            "evidence": PREDECESSOR.relative_to(ROOT).as_posix(),
            "evidence_sha256": PREDECESSOR_SHA256,
        },
    }


def update_memory_model(proposal_sha: str) -> None:
    path = ROOT / "design/MEMORY_MODEL.json"
    memory = load(path)
    memory["authority_override"] = {
        "active_replacement_contract": CONTRACT,
        "proposal_sha256": proposal_sha,
        "status": (
            "independently_accepted_manager_advance_pending"
            if review_accepted()
            else "independent_review_pending"
        ),
        "predecessor_bounded_no_go": PREDECESSOR.relative_to(ROOT).as_posix(),
        "implementation_authorized": True,
        "stage_closing": False,
    }
    memory["stage"] = "architecture"
    memory["claim_status"] = "architecture_estimates_not_measured_rtl_or_ppa"
    memory["source_contracts"] = [
        "MISSION.md",
        "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        "design/ARCHITECTURE.md",
        "design/SPEC.md",
        PREDECESSOR.relative_to(ROOT).as_posix(),
    ]
    memory["numeric_contract"] = {
        "contract_id": CONTRACT,
        "layer_scope": "all_24_transformer_layers_one_shared_engine",
        "projection_boundary": "signed32_accumulator_plus_per_output_scale32",
        "rope_boundary": "signed32_mantissa_plus_signed8_exponent",
        "score_boundary": "signed_q20_44",
        "probability_boundary": "unsigned_q0_15",
        "floating_point_rtl": False,
    }
    arch = memory.setdefault("architecture_a0", {})
    arch["pending_candidate_resources"] = {
        "shared_projection_lanes": 4,
        "shared_tagged_attention_paths": 1,
        "shared_rope_normalization_lanes": 2,
        "candidate_control_overhead_proxy": {
            "align_add_width_bits": 68,
            "score_accumulator_width_bits": 128,
            "score_converter_width_bits": 64,
            "leading_sign_counters": 2,
            "one_bit_two_to_one_operand_mux_positions": 612,
            "candidate_specific_state_bits": 736,
            "mux_bit_derivation": {
                "multiplier_operand_a_four_way": 96,
                "multiplier_operand_b_four_way": 96,
                "aligner_source_three_way": 136,
                "accumulator_input_two_way": 128,
                "converter_input_two_way": 64,
                "round_destination_two_way": 32,
                "control_and_tag_selects": 60,
            },
            "state_bit_derivation": {
                "native_operands": 128,
                "tagged_operands": 80,
                "score_accumulator": 128,
                "exponents": 24,
                "score_register": 64,
                "descriptor_and_addresses": 192,
                "layer_head_lane_tile_counters": 64,
                "phase_error_backpressure": 56,
            },
            "acceptance_area_cap_non_sram_mm2": 2.0,
            "area_claim": "none_before_fresh_synthesis",
        },
        "planning_area_treatment": (
            "resource proxies are advisory only and not an acceptance gate; "
            "fresh total non-SRAM area must remain within 2.0 mm2"
        ),
    }
    memory["execution_schedule"] = {
        "order": "native_qk_projection_then_tagged_rope_then_q20_44_score_then_staged_softmax_then_attention_value",
        "layer_scope": "all_24_layers_serially_reusing_one_engine",
        "key_tile_max": 64,
        "score_external_tile_bytes": 7392,
        "probability_external_tile_bytes": 1792,
        "kv_record_bytes_per_token": 800,
        "workspace_layout_bytes": {
            "coefficient_tile": 8320,
            "tagged_query_row": 4480,
            "tagged_k_tile": 40960,
            "v_tile": 8192,
            "score_head_buffer": 512,
            "probability_weight_tile": 1792,
            "output_slab": 896,
            "total": 65152,
        },
    }
    memory["cycle_model"] = {
        "estimate_boundary": "zero_stall_nonoverlapped_architecture_schedule_not_measured_rtl",
        "definitions": {"L": "valid_key_count", "T": "ceil_div_L_by_64", "C": "ceil_div_L_by_16"},
        "formulas": {
            "score_cycles": "4480_plus_4480_times_L_plus_560_times_T",
            "softmax_cycles": "14_times_open_T_plus_5_times_C_plus_32_close",
            "attention_value_cycles": "28_times_L_plus_56",
            "external_bytes_complete_row": "1320_times_L_plus_1088_plus_448_times_T",
            "dma_cycles_at_bandwidth_B": "ceil_div_external_bytes_by_B_plus_128_times_T_plus_600",
        },
    }
    memory["bandwidth_accounting"] = {
        "per_valid_key_bytes": {
            "tagged_kv_record_score_read": 800,
            "score_q20_44_write": 112,
            "score_q20_44_read": 112,
            "q1_31_weight_write": 56,
            "q1_31_weight_read": 56,
            "q0_15_probability_write": 28,
            "q0_15_probability_read": 28,
            "value_payload_reread": 128,
            "total": 1320,
        },
        "per_tile_bytes": {"score_headers_write": 224, "score_headers_read": 224, "total": 448},
        "per_row_fixed_bytes": 1088,
        "total_formula": "1320_times_L_plus_1088_plus_448_times_T",
        "all_24_layer_kv_cache": {
            "bytes_per_token": 19200,
            "bytes_at_32768_tokens": 629145600,
            "increase_over_272_byte_baseline_at_32768_tokens": 415236096,
        },
    }
    memory["decode_context_estimates"] = [
        {"context_tokens": 1, "tiles": 1, "external_bytes": 2856, "arithmetic_cycles": 10136, "dma_cycles_at_16_bytes_per_cycle": 907, "nonoverlapped_total_cycles_at_16_bytes_per_cycle": 11043, "all_24_layer_arithmetic_cycles": 243264, "all_24_layer_external_bytes": 68544},
        {"context_tokens": 1024, "tiles": 16, "external_bytes": 1359936, "arithmetic_cycles": 4634840, "dma_cycles_at_16_bytes_per_cycle": 87644, "nonoverlapped_total_cycles_at_16_bytes_per_cycle": 4722484, "all_24_layer_arithmetic_cycles": 111236160, "all_24_layer_external_bytes": 32638464},
        {"context_tokens": 32768, "tiles": 512, "external_bytes": 43484224, "arithmetic_cycles": 148160376, "dma_cycles_at_16_bytes_per_cycle": 2783900, "nonoverlapped_total_cycles_at_16_bytes_per_cycle": 150944276, "all_24_layer_arithmetic_cycles": 3555849024, "all_24_layer_external_bytes": 1043621376},
    ]
    memory["prefill_estimates"] = [
        {"prompt_tokens": 128, "sum_row_lengths": 8256, "sum_tiles": 192, "external_bytes": 11123200, "arithmetic_cycles": 38006528, "dma_cycles_at_16_bytes_per_cycle": 796576, "nonoverlapped_total_cycles_at_16_bytes_per_cycle": 38803104}
    ]
    memory["arithmetic_intensity_and_utilization"] = {
        "definition": "zero-stall architecture estimates, not measured RTL activity",
        "representative_rows": [
            {"valid_keys": 1, "dominant_macs_per_external_byte": 0.627451, "tagged_score_multiplier_busy_fraction": 0.353591, "tagged_rope_fraction": 0.239937, "softmax_fraction": 0.052486},
            {"valid_keys": 1024, "dominant_macs_per_external_byte": 1.349334, "tagged_score_multiplier_busy_fraction": 0.791832, "tagged_rope_fraction": 0.141785, "softmax_fraction": 0.001112},
            {"valid_keys": 32768, "dominant_macs_per_external_byte": 1.350381, "tagged_score_multiplier_busy_fraction": 0.792658, "tagged_rope_fraction": 0.141558, "softmax_fraction": 0.001019},
        ],
        "external_link_utilization": [
            {"valid_keys": 1, "bytes_per_cycle": 1, "during_dma_fraction": 0.796875, "end_to_end_fraction": 0.208163},
            {"valid_keys": 1, "bytes_per_cycle": 4, "during_dma_fraction": 0.495146, "end_to_end_fraction": 0.061669},
            {"valid_keys": 1, "bytes_per_cycle": 16, "during_dma_fraction": 0.196803, "end_to_end_fraction": 0.016164},
            {"valid_keys": 1024, "bytes_per_cycle": 1, "during_dma_fraction": 0.998057, "end_to_end_fraction": 0.226753},
            {"valid_keys": 1024, "bytes_per_cycle": 4, "during_dma_fraction": 0.992272, "end_to_end_fraction": 0.068305},
            {"valid_keys": 1024, "bytes_per_cycle": 16, "during_dma_fraction": 0.969787, "end_to_end_fraction": 0.017998},
            {"valid_keys": 32768, "bytes_per_cycle": 1, "during_dma_fraction": 0.998481, "end_to_end_fraction": 0.226822},
            {"valid_keys": 32768, "bytes_per_cycle": 4, "during_dma_fraction": 0.993953, "end_to_end_fraction": 0.068329},
            {"valid_keys": 32768, "bytes_per_cycle": 16, "during_dma_fraction": 0.976243, "end_to_end_fraction": 0.018005},
        ],
    }
    memory["resource_sharing_tradeoffs"] = [
        {"resource": "projection_macs", "selected": "existing_four_lanes", "dedicated_alternative": "separate_qk_array", "cost": "none beyond native-record write mode"},
        {"resource": "tagged_multiplier", "selected": "one_shared_32x32_limb_path", "dedicated_alternative": "fourteen_head_paths", "cost": "612 mux-bit positions and serialization"},
        {"resource": "score_accumulator", "selected": "one_signed128_folded_accumulator", "dedicated_alternative": "per_head_accumulators", "cost": "serialized score cycles"},
        {"resource": "round_saturate", "selected": "existing_shared_unit", "dedicated_alternative": "candidate_local_normalizers", "cost": "mode and control only"},
        {"resource": "softmax_sfu_divider", "selected": "existing_shared_exp_and_divide", "dedicated_alternative": "candidate_local_softmax", "cost": "no duplicate SFU"},
        {"resource": "dma_sram_control", "selected": "existing_dma_and_banks", "dedicated_alternative": "private_candidate_dma", "cost": "65152-byte workspace and larger external records"},
    ]
    memory["sram_live_set_budget_bytes"] = {
        "preserved_noncandidate_live_set": 377600,
        "candidate_workspace": 65152,
        "total_planned_peak": 442752,
        "planned_capacity": 524288,
        "remaining_planning_margin": 81536,
    }
    memory["risk_notes"] = [
        "leading-sign normalization and exponent alignment are timing risks",
        "all three signed ties-to-even boundaries require independent vectors",
        "800-byte K/V records increase external bandwidth",
        "all-layer runtime selection must be proven before paired smoke",
        "no numeric area or frequency claim exists before fresh PPA",
    ]
    dump(path, memory)


def update_machine_contracts() -> None:
    proposal_sha = sha256(PROPOSAL)
    record = contract_record(proposal_sha)
    update_memory_model(proposal_sha)

    policy_path = ROOT / "design/FAST_LOOP_POLICY.json"
    policy = load(policy_path)
    for key in ("active_repair_authorization", "architecture_proposal_authorization", "selected_replacement_contract"):
        policy[key] = copy.deepcopy(record)
    dump(policy_path, policy)

    target_path = ROOT / "design/TARGET.json"
    target = load(target_path)
    target["current_stage"] = "architecture"
    target["current_architecture_contract"] = copy.deepcopy(record)
    target["fast_loop_contract"]["active_repair_authorization"] = copy.deepcopy(record)
    target["fast_loop_contract"]["architecture_proposal_authorization"] = copy.deepcopy(record)
    target["fast_loop_contract"]["environment_readiness"] = "requires_hash_only_compatibility_rebind_for_native_accumulator_tagged_attention"
    target["fast_loop_contract"]["manager_recommendation"] = record["required_manager_action"]
    dump(target_path, target)

    scope_path = ROOT / "design/CHIP_SCOPE.json"
    scope = load(scope_path)
    scope["authority_override"] = {
        "active_replacement_contract": CONTRACT,
        "proposal_sha256": proposal_sha,
        "implementation_authorized": True,
        "stage_closing": False,
        "required_next_action": record["required_manager_action"],
        "predecessor_bounded_no_go": PREDECESSOR.relative_to(ROOT).as_posix(),
        "operator_implementation_approval": copy.deepcopy(record),
    }
    frontier = scope["implementation_frontier"]
    frontier["current_mode"] = "ADVANCE"
    frontier["ordered_supported_layer_operator_prefix"] = PREFIX
    frontier["first_unsupported_layer_operator"] = {
        "identifier": FIRST_UNSUPPORTED,
        "layer": 0,
        "operator": "rope_q",
        "reason": "shared all-layer candidate is architecture-frozen and not RTL-admitted",
    }
    frontier["latest_decision"] = "shared_native_accumulator_tagged_attention_architecture_frozen"
    scope["stage"] = {
        "current_stage": "architecture",
        "current_stage_status": record["status"],
        "downstream_stages_locked_until_manager_advance": ["environment", "rtl", "verification", "ppa", "prototype", "benchmark", "signoff"],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    scope["operator_owned_execution_policy"]["active_successor_contract"] = copy.deepcopy(record)
    scope["numerical_behavior"]["active_native_accumulator_tagged_attention_contract"] = {
        "contract_id": CONTRACT,
        "layer_scope": "all_24_layers_one_shared_engine",
        "native_projection_record": "signed32_accumulator_plus_scale32",
        "tagged_rope_record": "signed32_mantissa_plus_signed8_exponent",
        "score": "signed_q20_44",
        "probability": "unsigned_q0_15",
        "floating_point_rtl": False,
    }
    dump(scope_path, scope)

    manifest_path = ROOT / "design/RTL_MANIFEST.json"
    manifest = load(manifest_path)
    manifest["architecture_contract_status"] = record["status"]
    manifest["architecture_selection_evidence"] = {
        "contract_id": CONTRACT,
        "proposal": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        "proposal_sha256": proposal_sha,
        "status": record["status"],
    }
    manifest["candidate_evidence_hashes"] = {"proposal_sha256": proposal_sha}
    manifest["candidate_geometry"] = {
        "layer_scope": "all_24_layers",
        "query_heads": 14,
        "kv_heads": 2,
        "head_dim": 64,
        "tile_keys": 64,
        "native_lane_record_bits": 64,
        "tagged_lane_record_bits": 40,
        "score_width_bits": 64,
        "score_fraction_bits": 44,
        "score_tile_bytes": 7392,
        "kv_record_bytes": 800,
        "planned_sram_peak_bytes": 442752,
        "planned_sram_margin_bytes": 81536,
    }
    manifest["stage"] = "architecture"
    manifest["current_stage"] = "architecture"
    manifest["candidate_layer_operator"] = CONTRACT
    manifest["candidate_rtl_hash"] = None
    manifest["candidate_rtl_hash_scope"] = "successor_rtl_not_started"
    manifest["candidate_source_hashes"] = {}
    manifest["candidate_generated_hashes"] = {}
    manifest["candidate_rtl_sources"] = []
    manifest["candidate_generated_sources"] = []
    manifest["candidate_interface"] = {
        "planned_modules": ["ace2_native_accumulator_rope_core", "ace2_tagged_attention_score_core"],
        "flags_select": 5,
        "status": "architecture_bound_rtl_not_started",
    }
    manifest["candidate_review_binding"] = {
        "candidate_capability_accepted": False,
        "decision": "architecture_review_pending",
        "evidence": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        "scope": "architecture_stage_close_for_environment_entry_only",
        "stage_closing": False,
    }
    manifest["candidate_status"] = record["status"]
    manifest["candidate_verification_binding"] = {"status": "pending_no_native_tagged_rtl_or_vectors", "evidence": []}
    manifest["candidate_verification_complete"] = False
    manifest["candidate_first_unsupported_layer_operator_after_review"] = FIRST_UNSUPPORTED
    manifest["candidate_meets_numeric_acceptance"] = False
    manifest["candidate_requires_fresh_sky130_ppa"] = True
    manifest["proposed_replacement_contract"] = {
        **copy.deepcopy(record),
        "planned_modules": ["ace2_native_accumulator_rope_core", "ace2_tagged_attention_score_core"],
        "planned_generated_sources": [
            "verification/generated/native_accumulator_tagged_attention_vectors.json",
            "verification/generated/native_accumulator_tagged_attention_vectors.svh",
        ],
        "rtl_started": False,
        "accepted_frontier_changed": False,
        "rtl_tree_sha256_at_architecture_freeze": tree_sha256(ROOT / "rtl"),
    }
    manifest["traceability"]["architecture_contract_gap"] = {
        "resolution_owner": "independent Reviewer then Manager",
        "status": "architecture_review_pending_no_successor_rtl_started",
    }
    manifest["traceability"]["selected_mechanism"] = CONTRACT
    manifest["traceability"]["rtl.contract-traceability"] = False
    manifest["interfaces_contract_status"] = "accepted_shell_prefix_through_v_proj_native_tagged_contract_bound_rtl_not_started"
    dump(manifest_path, manifest)

    oracle_path = ROOT / "reference/ORACLE_MANIFEST.json"
    oracle = load(oracle_path)
    oracle["proposed_architecture_contract"] = {
        **copy.deepcopy(record),
        "architecture_review_status": "independently_accepted_manager_advance_pending" if review_accepted() else "independent_review_pending",
        "existing_vectors_status": "historical_vectors_only_new_contract_vectors_pending",
        "required_new_focused_vectors": [
            "native_signed32_accumulator_and_per_output_scale32",
            "tagged_absolute_rope_normalization_and_exponent_boundaries",
            "signed128_tagged_product_accumulation_and_q20_44_conversion",
            "q1_31_exp_u48_sum_q0_15_normalization_and_attention_value",
            "one_two_63_64_65_key_rows_14_to_2_mapping_and_all_24_layers",
            "reset_backpressure_generation_stride_reserved_memory_numeric_and_watchdog_errors",
        ],
    }
    dump(oracle_path, oracle)

    status_path = ROOT / "research/PUBLIC_STATUS.json"
    status = load(status_path)
    public_candidate = {
        "contract_id": CONTRACT,
        "proposal": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        "proposal_sha256": proposal_sha,
        "implementation_authorized": True,
        "stage_closing": False,
        "status": record["status"],
        "mechanism": "all_layer_native_accumulator_tagged_integer_attention",
        "predecessor_no_go": PREDECESSOR.relative_to(ROOT).as_posix(),
    }
    status["current_mode"] = "ADVANCE"
    status["supported_layer_operator_prefix"] = PREFIX
    status["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
    status["latest_decision"] = "native_accumulator_tagged_architecture_accepted_manager_advance_pending" if review_accepted() else "native_accumulator_tagged_architecture_review_pending"
    status["latest_ppa_frontier_status"] = "historical_frontier_preserved_no_candidate_ppa"
    status["selected_replacement_contract"] = public_candidate
    status["architecture_proposal_gate"] = {
        "contract_id": CONTRACT,
        "implementation_authorized": True,
        "status": "independently_accepted_manager_advance_pending" if review_accepted() else "independent_review_pending",
        "required_operator_action": "none_standing_authorization_is_current",
        "required_manager_action": record["required_manager_action"],
        "source": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
    }
    status["stage"].update({
        "current_stage": "architecture",
        "current_stage_status": record["status"],
        "current_stage_checklist": {
            "architecture.compute-memory-model": True,
            "architecture.interface-control": True,
            "architecture.leverage-risk": True,
            "architecture.area-reuse-plan": True,
        },
        "current_stage_evidence": [
            "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
            "design/ARCHITECTURE.md",
            "design/MEMORY_MODEL.json",
            "design/SPEC.md",
            PREDECESSOR.relative_to(ROOT).as_posix(),
        ],
    })
    status["blockers"] = [{
        "id": "manager_stage_advance_pending" if review_accepted() else "independent_architecture_review_pending",
        "stage": "architecture",
        "status": "active",
        "reason": "Independent architecture review accepted all four checklist items." if review_accepted() else "The sole successor is frozen and awaits independent architecture review.",
        "required_resolution": "Manager advances architecture to environment." if review_accepted() else "Independent Reviewer accepts all four architecture checklist items.",
        "evidence": REVIEW.relative_to(ROOT).as_posix() if review_accepted() else "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
    }]
    model = {
        "contract_id": CONTRACT,
        "decode_context_32768_total_cycles_at_16_bytes_per_cycle": 150944276,
        "prefill_128_total_cycles_at_16_bytes_per_cycle": 38803104,
        "planned_sram_peak_bytes": 442752,
        "remaining_sram_margin_bytes": 81536,
        "status": "architecture_estimate_not_measured",
        "source": "design/MEMORY_MODEL.json",
    }
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status[name]
        container["candidate_mechanism"] = public_candidate
        container["current_architecture_performance_model"] = model
        container["current_mode"] = "ADVANCE"
        container["current_stage"] = "architecture"
        container["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
        container["latest_decision"] = status["latest_decision"]
        container["latest_ppa_frontier_status"] = status["latest_ppa_frontier_status"]
        container["ordered_supported_layer_operator_prefix"] = PREFIX
        container["supported_layer_operator_prefix"] = PREFIX
        container["routing_status"] = "manager_stage_advance_pending" if review_accepted() else "independent_architecture_review_pending"
        container["rtl_contract_traceability"] = False
        container["required_operator_action"] = "none_standing_authorization_is_current"
        container["required_manager_action"] = record["required_manager_action"]
    status["dashboard_fields"]["latest_quality_diagnostic"] = {
        "evidence": PREDECESSOR.relative_to(ROOT).as_posix(),
        "finding": "layer0 focused arithmetic improved but layer0-only paired smoke remained end-to-end insensitive",
        "required_scope_change": "all_24_layers_one_shared_engine",
    }
    status["implementation_frontier"]["latest_quality_diagnostic"] = copy.deepcopy(status["dashboard_fields"]["latest_quality_diagnostic"])
    status["public_claims"] = [
        {"claim": "tile-BFP is a sealed bounded no-go", "evidence": [PREDECESSOR.relative_to(ROOT).as_posix()]},
        {"claim": "one all-layer tagged-integer successor is architecture-frozen", "evidence": ["design/NUMERICAL_REPLACEMENT_PROPOSAL.md", "design/ARCHITECTURE.md"]},
        {"claim": "the supported prefix and immutable targets are unchanged", "evidence": ["design/CHIP_SCOPE.json", "design/TARGET.json"]},
        {"claim": "no successor RTL, verification, PPA, prototype, benchmark, signoff, tapeout, or silicon result exists", "evidence": ["design/RTL_MANIFEST.json"]},
    ]
    paths = {
        "MISSION.md", "CHECKPOINT.md", "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        "design/ARCHITECTURE.md", "design/MEMORY_MODEL.json", "design/SPEC.md",
        "design/FAST_LOOP_POLICY.json", "design/CHIP_SCOPE.json", "design/TARGET.json",
        "design/RTL_MANIFEST.json", "reference/ORACLE_MANIFEST.json",
        "research/PIPELINE_STATE.json", "tools/bind_architecture_quality_contract.py",
        PREDECESSOR.relative_to(ROOT).as_posix(),
    }
    if review_accepted():
        paths.add(REVIEW.relative_to(ROOT).as_posix())
    status["artifact_hashes"] = [artifact(path) for path in sorted(paths)]
    timestamp = utc_now()
    status["generated_at_utc"] = timestamp
    status["last_updated_utc"] = timestamp
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": "UTF-8 sorted keys two-space indentation trailing newline canonical hash null during hash",
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump(status_path, status)

    dump(ROOT / ".argus/live-view.json", {
        "version": 1,
        "title": "Native-accumulator tagged attention architecture review",
        "reason": "One all-layer integer successor is frozen; independent architecture review is the next gate." if not review_accepted() else "Architecture review is done; Manager architecture-to-environment transition is pending.",
        "paths": ["research/PIPELINE_STATE.json", "design/NUMERICAL_REPLACEMENT_PROPOSAL.md", "design/MEMORY_MODEL.json", "research/PUBLIC_STATUS.json"],
    })


def validate() -> None:
    pipeline = load(ROOT / "research/PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "architecture", "Manager stage is not architecture")
    require(sha256(PREDECESSOR) == PREDECESSOR_SHA256, "predecessor no-go hash changed")
    proposal_sha = sha256(PROPOSAL)
    proposal = PROPOSAL.read_text(encoding="utf-8")
    for phrase in (
        f"`contract_id: {CONTRACT}`", "all 24 transformer layers",
        "signed Q20.44", "peak SRAM is 442,752", "2.0 mm2", "100 MHz",
    ):
        require(phrase in proposal, f"proposal phrase missing: {phrase}")
    memory = load(ROOT / "design/MEMORY_MODEL.json")
    require(memory["authority_override"]["active_replacement_contract"] == CONTRACT, "memory contract stale")
    require(memory["sram_live_set_budget_bytes"]["total_planned_peak"] == 442752, "SRAM model stale")
    require(memory["decode_context_estimates"][-1]["nonoverlapped_total_cycles_at_16_bytes_per_cycle"] == 150944276, "cycle model stale")
    require(memory["bandwidth_accounting"]["per_valid_key_bytes"]["total"] == 1320, "traffic derivation stale")
    require(sum(memory["architecture_a0"]["pending_candidate_resources"]["candidate_control_overhead_proxy"]["mux_bit_derivation"].values()) == 612, "mux derivation stale")
    require(sum(memory["architecture_a0"]["pending_candidate_resources"]["candidate_control_overhead_proxy"]["state_bit_derivation"].values()) == 736, "state derivation stale")
    require(len(memory["resource_sharing_tradeoffs"]) >= 6, "sharing tradeoffs incomplete")
    require({x["bytes_per_cycle"] for x in memory["arithmetic_intensity_and_utilization"]["external_link_utilization"]} == {1, 4, 16}, "bandwidth sweep incomplete")
    target = load(ROOT / "design/TARGET.json")
    scope = load(ROOT / "design/CHIP_SCOPE.json")
    manifest = load(ROOT / "design/RTL_MANIFEST.json")
    policy = load(ROOT / "design/FAST_LOOP_POLICY.json")
    oracle = load(ROOT / "reference/ORACLE_MANIFEST.json")
    for candidate in (
        policy["active_repair_authorization"], target["current_architecture_contract"],
        scope["authority_override"]["operator_implementation_approval"],
        manifest["proposed_replacement_contract"], oracle["proposed_architecture_contract"],
    ):
        require(candidate["contract_id"] == CONTRACT, "machine contract stale")
        require(candidate["proposal_sha256"] == proposal_sha, "proposal hash stale")
        require(candidate["implementation_authorized"] is True, "authorization missing")
        require(candidate["stage_closing"] is False, "implementation became stage closing")
    require(target["frozen_numeric_targets"]["area_cap_non_sram_mm2"] == 2.0, "area cap changed")
    require(target["frozen_numeric_targets"]["frequency_floor_mhz"] == 100.0, "frequency floor changed")
    require(scope["implementation_frontier"]["ordered_supported_layer_operator_prefix"] == PREFIX, "prefix changed")
    require(scope["implementation_frontier"]["first_unsupported_layer_operator"]["identifier"] == FIRST_UNSUPPORTED, "first unsupported changed")
    require(manifest["candidate_rtl_hash"] is None and not manifest["candidate_rtl_sources"], "RTL started during architecture")
    require(manifest["candidate_geometry"]["layer_scope"] == "all_24_layers", "candidate scope stale")
    require(manifest["candidate_verification_binding"] == {"status": "pending_no_native_tagged_rtl_or_vectors", "evidence": []}, "verification binding stale")
    architecture = (ROOT / "design/ARCHITECTURE.md").read_text(encoding="utf-8")
    spec = (ROOT / "design/SPEC.md").read_text(encoding="utf-8")
    for phrase in ("24x workload coverage", "612 mux-bit positions", "81,536 bytes", "flags[5]"):
        require(phrase in architecture + spec, f"architecture/interface phrase missing: {phrase}")
    for phrase in (
        "448 Q or 64 K 128-bit beats",
        "C_dma = ceil_div(1320*L + 1088 + 448*T,B)",
        "629,145,600 bytes",
        "advances `DESC_TAIL`",
    ):
        require(phrase in spec, f"SPEC repair missing: {phrase}")
    for stale in ("`3136*T`", "92,928-byte SRAM workspace", "560 Q or 80 K"):
        require(stale not in spec, f"stale candidate layout remains: {stale}")
    require("pending staged layer-0 mode" not in spec, "stale layer-0 candidate wording remains")
    require("tail advance before terminal result/error" in architecture, "tail retirement wording stale")
    require("completion tag, consumed ring slot" in architecture, "result CSR wording stale")
    require("descriptor sequence, opcode, layer, and failing" not in architecture, "result CSR overclaim remains")
    require(memory["source_contracts"][-1] == PREDECESSOR.relative_to(ROOT).as_posix(), "memory predecessor binding stale")
    status = load(ROOT / "research/PUBLIC_STATUS.json")
    require(status["stage"]["current_stage"] == "architecture", "public stage stale")
    require(status["selected_replacement_contract"]["contract_id"] == CONTRACT, "public contract stale")
    require(status["supported_layer_operator_prefix"] == PREFIX, "public prefix stale")
    require(status["first_unsupported_layer_operator"] == FIRST_UNSUPPORTED, "public first unsupported stale")
    require(status["privacy_policy"]["public_safe"] is True, "privacy policy false")
    require("/home/" not in json.dumps(status, sort_keys=True), "public status leaks private path")
    require(status["integrity"]["canonical_sha256"] == canonical_sha256(status), "public status hash stale")
    for item in status["artifact_hashes"]:
        require(sha256(ROOT / item["path"]) == item["sha256"], f"artifact hash stale: {item['path']}")
    if REVIEW.is_file():
        review = load(REVIEW)
        require(review.get("reviewer_status") == "done", "architecture review not done")
        require(review.get("contract_id") == CONTRACT, "review contract mismatch")
        require(review.get("proposal_sha256") == proposal_sha, "review proposal hash mismatch")
        require(all(review.get("checklist", {}).values()), "review checklist incomplete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        update_machine_contracts()
    validate()
    print(
        f"ACE2_ARCHITECTURE_CONTRACT_PASS stage=architecture contract={CONTRACT} "
        f"prefix={len(PREFIX)} first_unsupported={FIRST_UNSUPPORTED}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
