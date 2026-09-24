#!/usr/bin/env python3
"""Bind and validate the V-residual architecture-stage successor packet."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_qk_residual_cross_term_attention_v1"
PROPOSAL = ROOT / "design/NUMERICAL_REPLACEMENT_PROPOSAL.md"
FREEZE = ROOT / f"evidence/{CONTRACT}/architecture_freeze.json"
BLOCKER = ROOT / f"evidence/{CONTRACT}/latest/POST_REFREEZE_CONTRACT_BLOCKER.json"
REVIEW_PACKET = ROOT / f"evidence/{CONTRACT}/latest/ARCHITECTURE_REFREEZE_REVIEW_PACKET.json"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected object: {path}")
    return value


def dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def artifact(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def canonical_sha256(value: dict[str, Any]) -> str:
    candidate = copy.deepcopy(value)
    candidate.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def rounded(value: float) -> float:
    return round(value, 6)


def v_residual_projection_integer_contract() -> dict[str, Any]:
    return {
        "architectural_inputs": [
            "signed32_v_projection_accumulator",
            "ordinary_positive_signed32_multiplier",
            "ordinary_unsigned6_right_shift",
            "ordinary_zero_point_fixed_zero",
            "per_kv_head_baseline_v_output_scale32",
            "per_kv_head_residual_v_scale32",
        ],
        "forbidden_inputs": [
            "floating_point_input_scale",
            "floating_point_weight_scale",
            "pre_rounded_residual_coefficient",
            "q_or_k_projection_state",
        ],
        "scale32": {
            "byte_order": "little_endian",
            "layout": "u16_significand_at_0_s8_exponent_at_2_u8_reserved_zero_at_3",
            "significand_range": [32768, 65535],
            "exponent_range": [-24, 4],
            "exact_value": "significand_times_2_power_exponent_minus_15",
        },
        "operation_order": [
            "P_s64_equals_acc_s32_times_multiplier_positive_s32",
            "baseline_unclamped_equals_signed_rne_divide_P_by_2_power_shift",
            "v8_equals_signed8_saturation_of_baseline_unclamped",
            "E_s72_equals_P_minus_v8_times_2_power_shift",
            "delta_equals_baseline_v_exp_minus_residual_v_exp_minus_shift",
            "N_equals_E_times_baseline_v_significand_then_left_shift_by_nonnegative_delta",
            "D_equals_residual_v_significand_then_left_shift_by_negative_delta_magnitude",
            "residual_unclamped_equals_one_signed_rne_divide_N_by_D",
            "residual_v_s4_equals_symmetric_clamp_minus7_to_plus7",
        ],
        "rounding": {
            "residual_rounding_points": 1,
            "rule": "divide_absolute_numerator_increment_if_twice_remainder_gt_denominator_or_equal_and_quotient_odd_then_restore_sign",
        },
        "bounds": {
            "delta_inclusive": [-91, 28],
            "product_signed_bits": 64,
            "error_signed_bits": 72,
            "maximum_shifted_numerator_signed_bits": 116,
            "maximum_shifted_denominator_unsigned_bits": 107,
            "division_workspace_bits": 128,
        },
        "overflow_and_clamp": {
            "declared_width_failure": "numeric_overflow_no_successful_destination",
            "ordinary_int8_saturation": "preserved_baseline_behavior",
            "residual_clamp": "specified_behavior_with_separate_positive_and_negative_counters",
            "reserved_memory_code": -8,
        },
    }


SUCCESSOR = "shared_v_residual_value_correction_attention_v1"
VERIFICATION_NO_GO = ROOT / f"evidence/{CONTRACT}/latest/VERIFICATION_NO_GO.json"
SEALED_RTL_HASH = "16229d15cc23b8d78ab53aa3b247061e0965d2642f20c909e5e2675f28beec14"
PACKET_STATUS = "ready_for_fresh_independent_architecture_review_environment_review_pending"
HISTORICAL_PPA = {
    "candidate_ppa_run": False,
    "cells": 62199,
    "non_sram_area_mm2": 0.6108746272,
    "setup_slack_ns_at_100mhz": 0.1502,
}
PRESERVED = {
    "abstract_streaming_memory_boundary_bits": 128,
    "area_cap_non_sram_mm2": 2.0,
    "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
    "frequency_floor_mhz": 100.0,
    "historical_ppa_frontier": HISTORICAL_PPA,
    "mode": "ADVANCE",
    "ordered_supported_layer_operator_prefix": PREFIX,
}
SUCCESSOR_MEMORY_SOURCE_CONTRACTS = [
    "MISSION.md",
    "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
    "design/ARCHITECTURE.md",
    "design/SPEC.md",
    "design/RTL_MANIFEST.json",
    "research/PIPELINE_STATE.json",
    "research/PUBLIC_STATUS.json",
    "evidence/shared_qk_residual_cross_term_attention_v1/latest/VERIFICATION_NO_GO.json",
    "evidence/shared_qk_residual_cross_term_attention_v1/latest/ARCHITECTURE_REFREEZE_REVIEW_PACKET.json",
]
SUCCESSOR_BANDWIDTH_SWEEP = (1, 2, 4, 8, 16)
SUCCESSOR_REPRESENTATIVE_ROWS = (1, 1024, 32768)
SUCCESSOR_PREFILL_TOKENS = 128
SUCCESSOR_KV_RECORD_BYTES = 400
SUCCESSOR_WORKSPACE = {
    "residual_v_tile": 8192,
    "correction_accumulators_s64": 7168,
    "active_header_and_scale32": 128,
    "packed_current_token_projection_staging": 64,
    "visible_clamp_and_error_counters": 64,
    "alignment_and_guard_reserve": 768,
    "total": 16384,
}
SUCCESSOR_SRAM_BUDGET = {
    "preserved_noncandidate_live_set": 377600,
    "candidate_workspace": 16384,
    "total_planned_peak": 393984,
    "planned_capacity": 524288,
    "remaining_planning_margin": 130304,
}


def successor_record_contract() -> dict[str, Any]:
    return {
        "bytes": SUCCESSOR_KV_RECORD_BYTES,
        "layout": [
            {"offset": 0, "bytes": 128, "field": "baseline_k_s8"},
            {"offset": 128, "bytes": 128, "field": "baseline_v_s8"},
            {"offset": 256, "bytes": 128, "field": "residual_v_canonical_s4_in_s8_byte"},
            {"offset": 384, "bytes": 16, "field": "existing_checked_header"},
        ],
        "residual_lane_encoding": (
            "one_byte_per_lane_high_nibble_sign_extends_low_signed4_nibble_"
            "canonical_f8_reserved_minus8"
        ),
        "baseline_header_extended": False,
    }


def successor_interface_contract() -> dict[str, Any]:
    return {
        "descriptor_bytes": 64,
        "descriptor_or_csr_change": False,
        "external_stream_width_bits": 128,
        "candidate_flag_bit": 6,
        "candidate_flag_legal_opcodes": ["V_W4A8_PROJ", "KV_READ", "KV_WRITE", "ATTN_VALUE"],
        "candidate_flag_forbidden_opcodes": ["Q_W4A8_PROJ", "K_W4A8_PROJ", "ROPE", "ATTN_SCORE", "SOFTMAX"],
        "correction_core_inputs": [
            "authoritative_probability_q0_15",
            "canonical_residual_v_s4",
            "baseline_v_scale32",
            "residual_v_scale32",
            "lane_count",
            "authoritative_baseline_attention_value_accumulator",
        ],
        "correction_core_forbidden_inputs": [
            "query_residual", "key_residual", "rope_residual", "score_correction",
            "prompt_id", "dataset_id", "runtime_learned_gate",
        ],
        "scheduler_identity_fields": [
            "layer_id", "query_position", "query_head", "kv_head", "key_index", "output_lane",
        ],
        "scheduler_validates_identity_before_start": True,
        "correction_core_identity_ports": [],
        "identity_mismatch_behavior": "suppress_start_valid_terminate_descriptor_error_no_correction_transaction",
        "kv_record": successor_record_contract(),
    }


def successor_row_estimate(valid_keys: int) -> dict[str, Any]:
    tiles = ceil_div(valid_keys, 64)
    chunks = ceil_div(valid_keys, 16)
    v_residual_projection_cycles = 128
    base_score_mac_cycles = 896 * valid_keys
    base_score_conversion_cycles = 112 * valid_keys
    softmax_cycles = 14 * (tiles + 5 * chunks + 32)
    baseline_attention_value_cycles = 28 * valid_keys + 56
    residual_value_mac_cycles = 896 * valid_keys
    residual_value_conversion_cycles = 896 * 8
    candidate_core_cycles = (
        v_residual_projection_cycles
        + residual_value_mac_cycles
        + residual_value_conversion_cycles
    )
    arithmetic_cycles = (
        v_residual_projection_cycles
        + base_score_mac_cycles
        + base_score_conversion_cycles
        + softmax_cycles
        + baseline_attention_value_cycles
        + residual_value_mac_cycles
        + residual_value_conversion_cycles
    )
    external_bytes = 1048 * valid_keys + 1088 + 448 * tiles
    attention_macs = 2688 * valid_keys
    dedicated_projection_cycles = 64
    dedicated_residual_value_mac_cycles = 64 * valid_keys
    dedicated_residual_value_conversion_cycles = 512
    dedicated_candidate_core_cycles = (
        dedicated_projection_cycles
        + dedicated_residual_value_mac_cycles
        + dedicated_residual_value_conversion_cycles
    )
    link = []
    for bandwidth in SUCCESSOR_BANDWIDTH_SWEEP:
        dma_cycles = ceil_div(external_bytes, bandwidth) + 96 * tiles + 400
        total_cycles = arithmetic_cycles + dma_cycles
        link.append({
            "bytes_per_cycle": bandwidth,
            "dma_cycles": dma_cycles,
            "during_dma_fraction": rounded(external_bytes / (bandwidth * dma_cycles)),
            "end_to_end_fraction": rounded(external_bytes / (bandwidth * total_cycles)),
        })
    dma_cycles_16 = next(item["dma_cycles"] for item in link if item["bytes_per_cycle"] == 16)
    total_cycles_16 = arithmetic_cycles + dma_cycles_16
    return {
        "valid_keys": valid_keys,
        "tiles": tiles,
        "chunks": chunks,
        "external_bytes": external_bytes,
        "v_residual_projection_cycles": v_residual_projection_cycles,
        "base_score_mac_cycles": base_score_mac_cycles,
        "base_score_conversion_cycles": base_score_conversion_cycles,
        "softmax_cycles": softmax_cycles,
        "baseline_attention_value_cycles": baseline_attention_value_cycles,
        "residual_value_mac_cycles": residual_value_mac_cycles,
        "residual_value_conversion_cycles": residual_value_conversion_cycles,
        "candidate_core_cycles": candidate_core_cycles,
        "arithmetic_cycles": arithmetic_cycles,
        "attention_macs": attention_macs,
        "attention_macs_per_external_byte": rounded(attention_macs / external_bytes),
        "candidate_residual_macs_per_external_byte": rounded((896 * valid_keys) / external_bytes),
        "base_score_fraction": rounded((base_score_mac_cycles + base_score_conversion_cycles) / arithmetic_cycles),
        "softmax_fraction": rounded(softmax_cycles / arithmetic_cycles),
        "baseline_attention_value_fraction": rounded(baseline_attention_value_cycles / arithmetic_cycles),
        "residual_value_mac_fraction": rounded(residual_value_mac_cycles / arithmetic_cycles),
        "residual_value_conversion_fraction": rounded(residual_value_conversion_cycles / arithmetic_cycles),
        "candidate_core_fraction": rounded(candidate_core_cycles / arithmetic_cycles),
        "compute_memory_balance_bytes_per_cycle": rounded(external_bytes / arithmetic_cycles),
        "link": link,
        "dma_cycles_at_16_bytes_per_cycle": dma_cycles_16,
        "nonoverlapped_total_cycles_at_16_bytes_per_cycle": total_cycles_16,
        "dma_fraction_at_16_bytes_per_cycle": rounded(dma_cycles_16 / total_cycles_16),
        "max_speedup_from_eliminating_dma_at_16_bytes_per_cycle": rounded(total_cycles_16 / arithmetic_cycles),
        "max_speedup_from_infinite_candidate_core_at_16_bytes_per_cycle": rounded(
            total_cycles_16 / (total_cycles_16 - candidate_core_cycles)
        ),
        "dedicated_fourteen_path_candidate_core_cycles": dedicated_candidate_core_cycles,
        "speedup_from_fourteen_value_paths_and_two_projection_finalizers_at_16_bytes_per_cycle": rounded(
            total_cycles_16
            / (total_cycles_16 - candidate_core_cycles + dedicated_candidate_core_cycles)
        ),
    }


def successor_prefill_estimate(prompt_tokens: int) -> dict[str, Any]:
    rows = [successor_row_estimate(valid_keys) for valid_keys in range(1, prompt_tokens + 1)]
    return {
        "prompt_tokens": prompt_tokens,
        "sum_row_lengths": sum(row["valid_keys"] for row in rows),
        "sum_tiles": sum(row["tiles"] for row in rows),
        "sum_chunks": sum(row["chunks"] for row in rows),
        "external_bytes": sum(row["external_bytes"] for row in rows),
        "arithmetic_cycles": sum(row["arithmetic_cycles"] for row in rows),
        "dma_cycles_at_16_bytes_per_cycle": sum(row["dma_cycles_at_16_bytes_per_cycle"] for row in rows),
        "nonoverlapped_total_cycles_at_16_bytes_per_cycle": sum(
            row["nonoverlapped_total_cycles_at_16_bytes_per_cycle"] for row in rows
        ),
        "row_aggregation": "sum_of_independent_complete_row_schedules",
        "cross_row_payload_beat_packing": False,
        "final_partial_beat_rule": "pad_and_close_each_row_before_the_next_row",
    }


def successor_resource_sharing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "resource": "v_residual_projection_finalizer",
            "cycle_scope": "one_new_128_lane_v_token",
            "selected": "reuse_one_shared_residual_finalizer_across_two_kv_heads_and_24_layers",
            "selected_instances": 1,
            "selected_cycles": 128,
            "dedicated_alternative": "two_kv_head_finalizers",
            "dedicated_instances": 2,
            "dedicated_cycles": 64,
            "dedicated_cycle_savings": 64,
            "dedicated_added_area_proxy_units_over_selected": 1072,
        },
        {
            "resource": "residual_v_probability_mac_accumulator",
            "cycle_scope": "complete_attention_row",
            "selected": "one_signed_q0_15_times_s4_path_serialized_across_14_heads_and_896_lanes",
            "selected_instances": 1,
            "selected_cycles": "896_times_L",
            "dedicated_alternative": "fourteen_head_local_paths",
            "dedicated_instances": 14,
            "dedicated_cycles": "64_times_L",
            "dedicated_cycle_savings": "832_times_L",
            "representative_cycle_savings": [
                {"valid_keys": row["valid_keys"], "cycles": 832 * row["valid_keys"]}
                for row in rows
            ],
            "dedicated_added_area_proxy_units_over_selected": 1274,
        },
        {
            "resource": "residual_v_scale32_convert_and_checked_add",
            "cycle_scope": "896_completed_lane_sums_per_row",
            "selected": "reuse_one_eight_cycle_scale32_converter",
            "selected_instances": 1,
            "selected_cycles": 7168,
            "dedicated_alternative": "fourteen_head_local_converters",
            "dedicated_instances": 14,
            "dedicated_cycles": 512,
            "dedicated_cycle_savings": 6656,
            "dedicated_added_area_proxy_units_over_selected": 14144,
        },
        {
            "resource": "softmax_probability_backend",
            "cycle_scope": "score_then_softmax_then_value_dependency_chain",
            "selected": "reuse_authoritative_baseline_softmax_and_probability_store",
            "dedicated_alternative": "duplicate_candidate_softmax_backend",
            "dedicated_cycle_savings": 0,
            "dedicated_area_cost": "one_full_exp_lut_divider_and_probability_store",
        },
        {
            "resource": "dma_sram",
            "cycle_scope": "single_frozen_128_bit_external_stream",
            "selected": "reuse_existing_dma_and_banks_with_incremental_workspace",
            "selected_workspace_bytes": 16384,
            "dedicated_alternative": "private_candidate_dma_and_workspace_on_same_external_link",
            "dedicated_cycle_savings": 0,
            "dedicated_added_storage_bytes": 16384,
        },
        {
            "resource": "all_layer_engine_lifetime",
            "cycle_scope": "24_transformer_layers_execute_sequentially",
            "selected": "one_candidate_engine_reused_by_all_layers",
            "selected_instances": 1,
            "dedicated_alternative": "one_candidate_engine_per_layer",
            "dedicated_instances": 24,
            "dedicated_cycle_savings": 0,
            "dedicated_added_area_proxy_units_over_selected": 37007,
        },
    ]


def successor_contract(timestamp: str) -> dict[str, Any]:
    return {
        "approved_on_utc_date": "2026-08-01",
        "authority": "manager_exercising_live_operator_verification_no_go_directive",
        "authorization_effective_after": [
            "fresh_independent_architecture_acceptance",
            "fresh_independent_environment_acceptance",
            "manager_advance_to_rtl",
        ],
        "contract_id": SUCCESSOR,
        "frozen_at_utc": timestamp,
        "implementation_authorized": False,
        "mechanism": "baseline_attention_distribution_plus_signed4_v_residual_attention_value_correction",
        "operator_approval_consumed": False,
        "predecessor_contract_id": CONTRACT,
        "predecessor_no_go": artifact(VERIFICATION_NO_GO.relative_to(ROOT).as_posix()),
        "proposal": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        "proposal_sha256": sha256(PROPOSAL),
        "required_manager_action": "hold_architecture_for_fresh_independent_architecture_then_environment_acceptance",
        "rtl_started": False,
        "stage_closing": False,
        "status": PACKET_STATUS,
        "structurally_distinct_from_predecessor": True,
    }


def update_successor_memory(timestamp: str, current: dict[str, Any]) -> None:
    path = ROOT / "design/MEMORY_MODEL.json"
    previous = load(path)
    rows = [successor_row_estimate(valid_keys) for valid_keys in SUCCESSOR_REPRESENTATIVE_ROWS]
    memory = {
        "schema_version": 2,
        "generated_at_utc": timestamp,
        "stage": "architecture",
        "claim_status": "architecture_estimates_reconciled_not_measured_rtl_or_ppa",
        "source_contracts": SUCCESSOR_MEMORY_SOURCE_CONTRACTS,
        "authority_override": copy.deepcopy(current),
        "model": copy.deepcopy(previous["model"]),
        "preserved_frontier": copy.deepcopy(previous["preserved_frontier"]),
        "projection_macs_preserved_baseline": copy.deepcopy(previous["projection_macs_preserved_baseline"]),
        "architecture_a0": {
            key: copy.deepcopy(value)
            for key, value in previous["architecture_a0"].items()
            if key != "pending_candidate_resources"
        },
        "numeric_contract": {
            "contract_id": SUCCESSOR,
            "layer_scope": "all_24_transformer_layers_one_shared_engine",
            "baseline_equal_boundaries": ["q_proj", "k_proj", "v_proj", "rope", "score", "softmax_probability"],
            "residual_projection": "v_only_signed4_symmetric_minus7_to_plus7_reserved_minus8",
            "residual_v_scale32_rule": "smallest_legal_normalized_scale32_ceiling_of_exact_baseline_v_scale32_divided_by_14",
            "baseline_v_scale32_preservation": "bit_identical",
            "residual_projection_integer_transform": v_residual_projection_integer_contract(),
            "probability_boundary": "authoritative_unsigned_q0_15",
            "correction_equation": "sum_probability_times_residual_v_then_one_exact_residual_v_over_baseline_v_scale32_ratio_conversion_per_output_lane",
            "correction_insertion_point": "before_unchanged_attention_value_requantizer",
            "correction_accumulator_signed_bits_minimum": 34,
            "correction_accumulator_storage_bits": 64,
            "floating_point_rtl": False,
            "forbidden": [
                "q_residual", "k_residual", "residual_rope", "score_cross_term",
                "score_offset", "dataset_selector", "prompt_selector", "runtime_learned_gate",
            ],
        },
        "candidate_interface_contract": successor_interface_contract(),
        "execution_schedule": {
            "order": "baseline_v_plus_residual_generation_then_unchanged_qk_rope_score_softmax_then_baseline_value_plus_residual_v_correction",
            "layer_scope": "all_24_layers_serially_reusing_one_engine",
            "key_tile_max": 64,
            "kv_record_bytes_per_token": SUCCESSOR_KV_RECORD_BYTES,
            "score_external_tile_bytes": 7392,
            "probability_external_tile_bytes": 1792,
            "workspace_layout_bytes": copy.deepcopy(SUCCESSOR_WORKSPACE),
            "workspace_lifetime": (
                "residual_v_tile_live_only_during_candidate_value_traversal_correction_accumulators_"
                "live_until_lane_conversion_and_checked_add_all_candidate_storage_invalidated_on_error_or_reset"
            ),
        },
        "cycle_model": {
            "estimate_boundary": "zero_stall_nonoverlapped_architecture_schedule_not_measured_rtl",
            "definitions": {
                "L": "valid_key_count",
                "T": "ceil_div_L_by_64",
                "C": "ceil_div_L_by_16",
                "B": "effective_external_bytes_per_cycle",
                "prefill_P": "independent_rows_with_L_from_1_through_P",
            },
            "formulas": {
                "v_residual_projection_cycles": "128",
                "score_cycles": "1008_times_L",
                "softmax_cycles": "14_times_open_T_plus_5_times_C_plus_32_close",
                "baseline_attention_value_cycles": "28_times_L_plus_56",
                "residual_value_mac_cycles": "896_times_L",
                "residual_value_conversion_cycles": "896_times_8_equals_7168",
                "arithmetic_cycles_complete_row": "128_plus_1008_times_L_plus_softmax_plus_28_times_L_plus_56_plus_896_times_L_plus_7168",
                "external_bytes_complete_row": "1048_times_L_plus_1088_plus_448_times_T",
                "dma_cycles_at_bandwidth_B": "ceil_div_external_bytes_by_B_plus_96_times_T_plus_400",
                "prefill_dma_cycles_at_bandwidth_B": "sum_L_1_through_P_of_complete_row_dma_cycles",
            },
            "conversion_ownership": "accumulate_complete_lane_sum_then_one_eight_cycle_scale32_conversion_and_checked_add_per_896_output_lanes",
            "forbidden_stale_formula": "112_times_L_residual_value_conversions",
            "prefill_row_boundary_rule": "no_cross_row_payload_beat_packing_each_row_pads_and_closes_its_final_B_byte_beat",
        },
        "bandwidth_accounting": {
            "per_valid_key_bytes": {
                "baseline_k_v_residual_v_record": 400,
                "score_write": 112,
                "score_read": 112,
                "q1_31_write": 56,
                "q1_31_read": 56,
                "q0_15_write": 28,
                "q0_15_read": 28,
                "baseline_and_residual_v_reread": 256,
                "total": 1048,
            },
            "per_tile_bytes": {"score_headers_write": 224, "score_headers_read": 224, "total": 448},
            "per_row_fixed_bytes": 1088,
            "all_24_layer_kv_cache": {
                "bytes_per_token": 9600,
                "bytes_at_32768_tokens": 314572800,
                "increase_over_272_byte_baseline_at_32768_tokens": 100663296,
            },
        },
        "decode_context_estimates": [
            {
                "context_tokens": row["valid_keys"],
                "tiles": row["tiles"],
                "external_bytes": row["external_bytes"],
                "arithmetic_cycles": row["arithmetic_cycles"],
                "dma_cycles_at_16_bytes_per_cycle": row["dma_cycles_at_16_bytes_per_cycle"],
                "nonoverlapped_total_cycles_at_16_bytes_per_cycle": row["nonoverlapped_total_cycles_at_16_bytes_per_cycle"],
            }
            for row in rows
        ],
        "prefill_estimates": [successor_prefill_estimate(SUCCESSOR_PREFILL_TOKENS)],
        "sram_live_set_budget_bytes": copy.deepcopy(SUCCESSOR_SRAM_BUDGET),
        "arithmetic_intensity_and_utilization": {
            "definition": "v_residual_successor_zero_stall_architecture_estimates_not_measured_rtl_activity",
            "cycle_decomposition": {
                "v_residual_projection_cycles": "128",
                "base_score_mac_cycles": "896_times_L",
                "base_score_conversion_cycles": "112_times_L",
                "softmax_cycles": "14_times_open_T_plus_5_times_C_plus_32_close",
                "baseline_attention_value_cycles": "28_times_L_plus_56",
                "residual_value_mac_cycles": "896_times_L",
                "residual_value_conversion_cycles": "7168",
            },
            "representative_rows": [
                {
                    key: row[key]
                    for key in (
                        "valid_keys", "external_bytes", "arithmetic_cycles", "attention_macs",
                        "attention_macs_per_external_byte", "candidate_residual_macs_per_external_byte",
                        "base_score_fraction", "softmax_fraction", "baseline_attention_value_fraction",
                        "residual_value_mac_fraction", "residual_value_conversion_fraction",
                        "candidate_core_fraction",
                    )
                }
                for row in rows
            ],
            "external_link_utilization": [
                {"valid_keys": row["valid_keys"], **link}
                for row in rows
                for link in row["link"]
            ],
        },
        "roofline_amdahl_bounds": {
            "assumptions": (
                "nonoverlapped_zero_stall_schedule_one_shared_residual_mac_one_reused_eight_cycle_"
                "scale32_converter_one_128_bit_link_dedicated_case_uses_fourteen_head_paths_and_two_projection_finalizers"
            ),
            "representative_rows": [
                {
                    key: row[key]
                    for key in (
                        "valid_keys", "compute_memory_balance_bytes_per_cycle",
                        "dma_fraction_at_16_bytes_per_cycle",
                        "max_speedup_from_eliminating_dma_at_16_bytes_per_cycle",
                        "max_speedup_from_infinite_candidate_core_at_16_bytes_per_cycle",
                        "speedup_from_fourteen_value_paths_and_two_projection_finalizers_at_16_bytes_per_cycle",
                        "candidate_core_cycles", "dedicated_fourteen_path_candidate_core_cycles",
                    )
                }
                for row in rows
            ],
            "conclusions": [
                "all_representative_rows_are_compute_bound_at_one_byte_per_cycle",
                "the_128_bit_link_is_not_the_primary_limit_in_the_nonoverlapped_model",
                "fourteen_head_paths_trade_large_area_for_bounded_row_speedup",
                "one_physical_engine_covers_all_24_sequential_layers_without_cross_layer_throughput_loss",
            ],
        },
        "resource_sharing_tradeoffs": successor_resource_sharing(rows),
        "risk_notes": [
            "residual_v_scale32_calibration_must_remain_frozen_and_quality_independent",
            "canonical_sign_extension_and_reserved_minus8_must_be_checked_on_every_residual_v_lane",
            "400_byte_records_increase_external_kv_capacity_by_100663296_bytes_at_max_context",
            "the_shared_mac_and_converter_dominate_candidate_cycles_at_long_context",
            "local_attention_value_improvement_may_still_regress_final_lm_head",
            "no_numeric_area_frequency_power_or_throughput_claim_exists_before_fresh_implementation_and_measurement",
        ],
    }
    memory["architecture_a0"]["pending_candidate_resources"] = {
        "shared_v_residual_projection_finalizers": 1,
        "shared_v_residual_value_mac_paths": 1,
        "shared_v_residual_scale32_converters": 1,
        "candidate_control_overhead_proxy": {
            "acceptance_area_cap_non_sram_mm2": 2.0,
            "area_claim": "none_before_fresh_synthesis",
            "area_proxy_method": "normalized_logic_units_for_relative_sharing_only",
            "area_proxy_units_selected_incremental": 521,
            "area_proxy_breakdown_units": {
                "q0_15_times_s4_multiplier_partial_products": 64,
                "minimum_correction_accumulator_bits": 34,
                "checked_value_adder_bits": 39,
                "operand_mux_bits": 128,
                "candidate_control_state_bits": 256,
            },
            "reused_scale32_converter_equivalent_proxy_units": 1088,
            "selected_engine_equivalent_proxy_units_including_reused_converter": 1609,
        },
    }
    dump(path, memory)


def rollback_pipeline(timestamp: str) -> None:
    path = ROOT / "research/PIPELINE_STATE.json"
    pipeline = load(path)
    require(pipeline.get("current_stage") in {"verification", "architecture"}, "unexpected Manager stage")
    reason = (
        "The hash-bound shared_qk_residual_cross_term_attention_v1 candidate passed "
        "standalone verification and improved five mandatory focused metrics, but C4-en "
        "final lm_head relative-L2 regressed from 1.1934673932274116 to "
        "1.1984861902850406. Its authority is consumed and the mechanism is sealed; paired "
        "smoke, shell, PPA, prototype, benchmark, and signoff remain prohibited. Roll "
        "verification back to architecture and freeze structurally distinct successor "
        f"{SUCCESSOR}, preserving ADVANCE mode, the prefix through layer_0.v_proj, first "
        "unsupported layer_0.rope_q, the 2.0 mm2 cap, 100 MHz floor, abstract streaming "
        "memory boundary, and historical PPA frontier."
    )
    already = any(
        item.get("from_stage") == "verification"
        and item.get("to_stage") == "architecture"
        and SUCCESSOR in item.get("reason", "")
        for item in pipeline.get("stage_history", [])
    )
    if not already:
        require(pipeline.get("current_stage") == "verification", "rollback source stage changed")
        event = {
            "at": timestamp,
            "from_stage": "verification",
            "reason": reason,
            "rolled_back_by": "manager",
            "to_stage": "architecture",
        }
        pipeline.setdefault("rollback_history", []).append(event)
        pipeline.setdefault("stage_history", []).append({
            "at": timestamp,
            "by": "manager",
            "direction": "rollback",
            "from_stage": "verification",
            "reason": reason,
            "to_stage": "architecture",
        })
    pipeline["current_stage"] = "architecture"
    stages = pipeline.setdefault("stages", {})
    stages.setdefault("architecture", {})["status"] = "pending_fresh_independent_review"
    for name in ("environment", "rtl", "verification", "ppa", "prototype", "benchmark", "signoff"):
        stages.setdefault(name, {})["status"] = "locked_pending_predecessor_stage_acceptance"
    dump(path, pipeline)


def update_scope(timestamp: str, current: dict[str, Any]) -> None:
    path = ROOT / "design/CHIP_SCOPE.json"
    scope = load(path)
    frontier = scope["implementation_frontier"]
    frontier["current_mode"] = "ADVANCE"
    frontier["ordered_supported_layer_operator_prefix"] = PREFIX
    frontier["supported_layer_operator_prefix"] = PREFIX
    frontier["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
    frontier["latest_decision"] = "v_residual_value_correction_architecture_frozen_review_pending"
    scope["stage"] = {
        "current_stage": "architecture",
        "current_stage_status": PACKET_STATUS,
        "downstream_stages_locked_until_manager_advance": [
            "environment", "rtl", "verification", "ppa", "prototype", "benchmark", "signoff"
        ],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    scope["authority_override"] = {
        "active_replacement_contract": SUCCESSOR,
        "implementation_authorized": False,
        "operator_approval_consumed": False,
        "stage_closing": False,
        "required_next_action": current["required_manager_action"],
        "operator_implementation_approval": copy.deepcopy(current),
        "sealed_predecessor_verification_no_go": VERIFICATION_NO_GO.relative_to(ROOT).as_posix(),
    }
    scope["operator_owned_execution_policy"]["active_successor_contract"] = copy.deepcopy(current)
    numerical = scope["numerical_behavior"]
    predecessor = numerical.pop("active_qk_residual_cross_term_contract", None)
    if predecessor is not None:
        predecessor["status"] = "sealed_verification_no_go_historical_only"
        numerical["sealed_qk_residual_cross_term_predecessor"] = predecessor
    numerical["active_v_residual_value_correction_contract"] = {
        "contract_id": SUCCESSOR,
        "baseline_equal_boundaries": ["q_proj", "k_proj", "v_proj", "rope", "score", "softmax_probability"],
        "correction": "signed4_v_projection_residual_weighted_by_authoritative_q0_15_probability",
        "correction_insertion_point": "before_unchanged_attention_value_requantizer",
        "floating_point_rtl": False,
        "forbidden": [
            "q_or_k_residual_sidecar", "score_cross_terms", "score_offsets",
            "dataset_or_prompt_selector", "runtime_learned_gate",
        ],
        "layer_scope": "all_24_layers_one_shared_engine",
        "metadata_rows": 48,
        "residual_format": "signed4_symmetric_minus7_to_plus7_reserved_minus8",
        "kv_record": successor_record_contract(),
        "interface": successor_interface_contract(),
        "workspace_bytes": SUCCESSOR_WORKSPACE["total"],
        "planned_sram_peak_bytes": SUCCESSOR_SRAM_BUDGET["total_planned_peak"],
        "residual_value_mac_cycles": "896_times_L",
        "residual_value_conversion_cycles": 7168,
        "status": "architecture_frozen_independent_review_pending",
    }
    scope["last_updated_utc"] = timestamp
    dump(path, scope)


def write_blocker(timestamp: str) -> None:
    blocker = {
        "schema_version": 2,
        "bound_at_utc": timestamp,
        "claim_boundary": "Architecture rollback and successor freeze only; no new RTL or downstream result.",
        "contract_id": SUCCESSOR,
        "current_stage": "architecture",
        "decision": "verification_no_go_consumed_mechanism_successor_review_required",
        "implementation_authorization_consumed": False,
        "measured_failure": {
            "dataset": "c4_en_512",
            "metric": "lm_head.relative_l2",
            "baseline": 1.1934673932274116,
            "candidate": 1.1984861902850406,
            "delta": 0.005018797057628976,
            "local_metrics_improved": 5,
            "interpretation": "local score/value improvement did not propagate to C4 final output",
        },
        "preserved_contracts": PRESERVED,
        "prohibited_runs_until_both_acceptances": [
            "new_rtl", "focused_quality", "paired_smoke", "full_shell",
            "canonical_sky130_ppa", "prototype", "benchmark", "signoff",
        ],
        "required_resolution": (
            "Fresh independent architecture acceptance and then fresh independent environment "
            "acceptance must bind the refreeze packet before the Manager may return to RTL."
        ),
        "resolution": {
            "paper_status": "structurally_distinct_successor_frozen_review_pending",
            "predecessor_contract_id": CONTRACT,
            "predecessor_rtl_hash": SEALED_RTL_HASH,
            "selected_solution": SUCCESSOR,
            "structural_difference": "value_path_residual_correction_with_bit_identical_attention_distribution",
        },
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    blocker["integrity"]["canonical_sha256"] = canonical_sha256(blocker)
    dump(BLOCKER, blocker)


def write_packet(timestamp: str, current: dict[str, Any]) -> None:
    rows = [successor_row_estimate(valid_keys) for valid_keys in SUCCESSOR_REPRESENTATIVE_ROWS]
    prefill = successor_prefill_estimate(SUCCESSOR_PREFILL_TOKENS)
    packet = {
        "schema_version": 2,
        "contract_id": SUCCESSOR,
        "generated_at_utc": timestamp,
        "current_stage": "architecture",
        "status": PACKET_STATUS,
        "sealed_predecessor": {
            "contract_id": CONTRACT,
            "candidate_rtl_hash": SEALED_RTL_HASH,
            "decision": "bounded_no_go_focused_discriminator_c4_lm_head_failed",
            "evidence": artifact(VERIFICATION_NO_GO.relative_to(ROOT).as_posix()),
            "mechanism_reuse_prohibited": True,
        },
        "measured_failure_basis": {
            "c4_en_512_lm_head_baseline_relative_l2": 1.1934673932274116,
            "c4_en_512_lm_head_candidate_relative_l2": 1.1984861902850406,
            "delta_relative_l2": 0.005018797057628976,
            "other_mandatory_metrics_strictly_improved": 5,
            "architecture_response": "preserve_attention_distribution_and_move_correction_to_value_path",
        },
        "selected_solution": {
            "name": SUCCESSOR,
            "baseline_equal_boundaries": ["q_proj", "k_proj", "v_proj", "rope", "score", "softmax_probability"],
            "correction_core": "shared_signed4_v_residual_attention_value_engine",
            "correction_insertion_point": "before_unchanged_attention_value_requantizer",
            "metadata": "48_static_layer_kv_head_scale32_records",
            "metadata_derivation": "smallest_legal_normalized_scale32_ceiling_of_exact_baseline_v_scale32_divided_by_14",
            "residual_format": "signed4_symmetric_minus7_to_plus7_reserved_minus8",
            "score_behavior": "bit_identical_authoritative_baseline",
            "structurally_distinct": True,
        },
        "numerical_contract": {
            "rounding": "round_to_nearest_ties_to_even_once_per_conversion",
            "value_domain_conversion": "exact_residual_v_scale32_over_baseline_v_scale32_ratio_after_complete_lane_accumulation",
            "overflow": "checked_visible_descriptor_error_no_silent_wrap",
            "runtime_dataset_or_prompt_selection": False,
            "floating_point_rtl": False,
        },
        "interface_contract": {
            **successor_interface_contract(),
        },
        "compute_memory_budget": {
            "kv_record_bytes": SUCCESSOR_KV_RECORD_BYTES,
            "kv_record_layout": successor_record_contract(),
            "candidate_workspace_layout_bytes": copy.deepcopy(SUCCESSOR_WORKSPACE),
            "candidate_workspace_bytes": SUCCESSOR_SRAM_BUDGET["candidate_workspace"],
            "preserved_noncandidate_live_set_bytes": SUCCESSOR_SRAM_BUDGET["preserved_noncandidate_live_set"],
            "planned_peak_sram_bytes": SUCCESSOR_SRAM_BUDGET["total_planned_peak"],
            "planned_sram_capacity_bytes": SUCCESSOR_SRAM_BUDGET["planned_capacity"],
            "remaining_sram_margin_bytes": SUCCESSOR_SRAM_BUDGET["remaining_planning_margin"],
            "added_v_residual_projection_cycles_per_token": 128,
            "added_value_mac_cycles_per_row": "896_times_L",
            "added_conversion_cycles_per_row": 7168,
            "conversion_derivation": "896_completed_output_lane_sums_times_one_eight_cycle_scale32_conversion_and_checked_add",
            "external_bytes_complete_row": "1048_times_L_plus_1088_plus_448_times_T",
            "all_24_layer_kv_cache_bytes_per_token": 9600,
            "all_24_layer_kv_cache_bytes_at_32768_tokens": 314572800,
            "increase_over_272_byte_baseline_at_32768_tokens": 100663296,
            "status": "architecture_estimate_reconciled_pending_independent_acceptance",
        },
        "compute_memory_derivation": {
            "definitions": {"L": "valid_key_count", "T": "ceil_div_L_by_64", "C": "ceil_div_L_by_16", "B": "effective_external_bytes_per_cycle"},
            "formulas": {
                "v_residual_projection_cycles": "128",
                "score_cycles": "1008_times_L",
                "softmax_cycles": "14_times_open_T_plus_5_times_C_plus_32_close",
                "baseline_attention_value_cycles": "28_times_L_plus_56",
                "residual_value_mac_cycles": "896_times_L",
                "residual_value_conversion_cycles": "7168",
                "dma_cycles": "ceil_div_open_1048_times_L_plus_1088_plus_448_times_T_close_by_B_plus_96_times_T_plus_400",
            },
            "representative_rows": rows,
            "prefill_128": prefill,
            "roofline_amdahl_conclusions": [
                "all_representative_rows_are_compute_bound_at_one_byte_per_cycle",
                "eliminating_dma_has_only_the_published_bounded_speedup_at_16_bytes_per_cycle",
                "fourteen_head_local_value_paths_trade_area_for_the_published_bounded_speedup",
            ],
        },
        "resource_sharing_tradeoffs": successor_resource_sharing(rows),
        "focused_discriminator_contract": {
            "datasets": ["wikitext2", "c4_en_512"],
            "token_limit": 128,
            "all_24_layers": True,
            "layer_0_score": "bit_identical_required",
            "layer_0_attention_value": "strict_relative_l2_improvement_required_both_datasets",
            "lm_head": "strict_relative_l2_improvement_required_both_datasets",
            "failure_action": "seal_before_paired_smoke",
        },
        "preserved_contracts": PRESERVED,
        "acceptance": {
            "architecture": {
                "status": "pending_fresh_independent_reviewer",
                "required_checks": [
                    "structural_difference_from_sealed_qk_cross_term_mechanism",
                    "exact_integer_value_residual_and_accumulation_contract",
                    "interface_memory_cycle_and_error_semantics_reconciled",
                    "successor_specific_roofline_amdahl_and_resource_sharing_derivations",
                    "preserved_prefix_targets_stream_boundary_and_historical_ppa_frontier",
                ],
            },
            "environment": {
                "status": "pending_after_architecture_acceptance",
                "required_checks": [
                    "maintained_sim_lint_formal_synthesis_sta_support",
                    "no_tool_pdk_board_or_license_assumption_changed",
                ],
            },
        },
        "prohibited_runs_until_both_acceptances": [
            "new_rtl", "focused_quality", "paired_smoke", "full_shell",
            "canonical_sky130_ppa", "prototype", "benchmark", "signoff",
        ],
        "bindings": {
            "architecture": artifact("design/ARCHITECTURE.md"),
            "architecture_contract": artifact("design/NUMERICAL_REPLACEMENT_PROPOSAL.md"),
            "spec": artifact("design/SPEC.md"),
            "memory_model": artifact("design/MEMORY_MODEL.json"),
            "chip_scope": artifact("design/CHIP_SCOPE.json"),
            "pipeline_state": artifact("research/PIPELINE_STATE.json"),
            "active_blocker": artifact(BLOCKER.relative_to(ROOT).as_posix()),
            "verification_no_go": artifact(VERIFICATION_NO_GO.relative_to(ROOT).as_posix()),
        },
        "projection_targets": ["design/RTL_MANIFEST.json", "research/PUBLIC_STATUS.json"],
        "implementation_authorized": False,
        "claim_boundary": "Architecture packet only; no RTL, verification, PPA, prototype, benchmark, signoff, tapeout, or silicon claim.",
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    packet["integrity"]["canonical_sha256"] = canonical_sha256(packet)
    dump(REVIEW_PACKET, packet)


def update_manifest(timestamp: str, current: dict[str, Any]) -> None:
    path = ROOT / "design/RTL_MANIFEST.json"
    manifest = load(path)
    require(manifest.get("candidate_rtl_hash") == SEALED_RTL_HASH, "sealed RTL hash changed")
    historical = copy.deepcopy(manifest.get("historical_standalone_rtl", {}))
    require(historical, "sealed predecessor historical RTL record missing")
    historical["contract_id"] = CONTRACT
    historical["candidate_rtl_hash"] = SEALED_RTL_HASH
    historical["status"] = "sealed_verification_no_go_historical_only"
    historical["verification_no_go"] = artifact(VERIFICATION_NO_GO.relative_to(ROOT).as_posix())
    manifest["historical_standalone_rtl"] = historical
    manifest.pop("independent_reviewer_verdict", None)
    manifest["stage"] = "architecture"
    manifest["current_stage"] = "architecture"
    manifest["architecture_contract_status"] = PACKET_STATUS
    manifest["architecture_selection_evidence"] = {
        "contract_id": SUCCESSOR,
        "proposal": current["proposal"],
        "proposal_sha256": current["proposal_sha256"],
        "status": PACKET_STATUS,
    }
    manifest["architecture_refreeze_review_packet"] = artifact(REVIEW_PACKET.relative_to(ROOT).as_posix())
    manifest["proposed_replacement_contract"] = copy.deepcopy(current)
    manifest["candidate_id"] = "v_residual_value_correction_architecture_only"
    manifest["candidate_layer_operator"] = SUCCESSOR
    manifest["candidate_rtl_hash_scope"] = "historical_sealed_predecessor_rtl_no_current_execution_authority"
    manifest["candidate_status"] = PACKET_STATUS
    manifest["candidate_evidence_hashes"] = {
        "scope": "current_v_residual_successor_architecture_evidence",
        "contract_id": SUCCESSOR,
        "proposal_sha256": current["proposal_sha256"],
        "architecture_description_sha256": sha256(ROOT / "design/ARCHITECTURE.md"),
        "interface_spec_sha256": sha256(ROOT / "design/SPEC.md"),
        "memory_model_sha256": sha256(ROOT / "design/MEMORY_MODEL.json"),
        "chip_scope_sha256": sha256(ROOT / "design/CHIP_SCOPE.json"),
        "architecture_refreeze_review_packet_sha256": sha256(REVIEW_PACKET),
        "sealed_predecessor_verification_no_go_sha256": sha256(VERIFICATION_NO_GO),
        "architecture_review_sha256": None,
        "architecture_review_scope": "fresh_independent_architecture_review_pending",
    }
    manifest["candidate_generated_hashes"] = {}
    manifest["candidate_generated_sources"] = []
    manifest["candidate_source_hashes"] = {}
    manifest["candidate_rtl_sources"] = []
    manifest["candidate_interface"] = {
        "status": "architecture_only_no_successor_rtl_implemented",
        "implemented_modules": [],
        "planned_module": "ace2_v_residual_value_correction_core",
        "descriptor_or_csr_change": False,
        "external_stream_width_bits": 128,
        "scheduler_identity_validation": successor_interface_contract()["scheduler_identity_fields"],
        "correction_core_identity_ports": [],
        "inputs": successor_interface_contract()["correction_core_inputs"],
        "forbidden_inputs": successor_interface_contract()["correction_core_forbidden_inputs"],
    }
    manifest["candidate_geometry"] = {
        "layer_scope": "all_24_layers",
        "query_heads": 14,
        "kv_heads": 2,
        "head_dim": 64,
        "tile_keys": 64,
        "residual_v_width_bits": 4,
        "residual_v_storage_bytes_per_token": 128,
        "kv_record_bytes": SUCCESSOR_KV_RECORD_BYTES,
        "score_width_bits": 64,
        "score_fraction_bits": 44,
        "score_tile_bytes": 7392,
        "probability_width_bits": 16,
        "probability_fraction_bits": 15,
        "metadata_records": 48,
        "candidate_workspace_bytes": SUCCESSOR_WORKSPACE["total"],
        "planned_sram_peak_bytes": SUCCESSOR_SRAM_BUDGET["total_planned_peak"],
        "planned_sram_margin_bytes": SUCCESSOR_SRAM_BUDGET["remaining_planning_margin"],
    }
    manifest["candidate_model_metadata"] = {
        "record_count": 48,
        "record_order": "layer_0_to_23_then_kv_head_0_to_1",
        "record_format": "little_endian_u16_significand_s8_exponent_u8_reserved_zero",
        "derivation_owner": "baseline_v_calibration_provenance_frozen_before_quality",
        "derivation_rule": "smallest_legal_normalized_scale32_ceiling_of_exact_baseline_v_scale32_divided_by_14",
        "baseline_v_scale32_preservation": "bit_identical",
        "quality_tuning_permitted": False,
        "generated_table": None,
        "status": "architecture_contract_frozen_generation_deferred_until_legal_rtl_entry",
    }
    manifest["candidate_schedule"] = {
        "estimate_boundary": "architecture_only_zero_stall_nonoverlapped",
        "v_residual_projection_cycles_per_token": 128,
        "score_cycles": "1008_times_L_unchanged_baseline",
        "softmax_cycles": "14_times_open_T_plus_5_times_C_plus_32_close_unchanged_baseline",
        "baseline_attention_value_cycles": "28_times_L_plus_56",
        "residual_value_mac_cycles": "896_times_L",
        "residual_value_conversion_cycles": 7168,
        "external_bytes_complete_row": "1048_times_L_plus_1088_plus_448_times_T",
        "kv_record_bytes": SUCCESSOR_KV_RECORD_BYTES,
    }
    manifest["candidate_review_binding"] = {
        "candidate_capability_accepted": False,
        "decision": "fresh_independent_architecture_review_pending",
        "evidence": REVIEW_PACKET.relative_to(ROOT).as_posix(),
        "scope": "architecture_stage_packet_only",
        "stage_closing": False,
    }
    manifest["candidate_verification_binding"] = {
        "status": "no_successor_verification_authority_predecessor_sealed_historical_only",
        "successor_evidence": [],
        "sealed_predecessor_no_go": artifact(VERIFICATION_NO_GO.relative_to(ROOT).as_posix()),
    }
    manifest["candidate_verification_complete"] = False
    manifest["candidate_meets_numeric_acceptance"] = False
    manifest["candidate_requires_fresh_sky130_ppa"] = True
    manifest["candidate_first_unsupported_layer_operator_after_review"] = FIRST_UNSUPPORTED
    manifest["candidate_supported_layer_operator_prefix_after_review"] = PREFIX
    manifest["independent_reviewer_acceptance"] = {
        "status": "pending_fresh_independent_architecture_review",
        "architecture_packet": artifact(REVIEW_PACKET.relative_to(ROOT).as_posix()),
        "environment_review": "locked_until_architecture_acceptance",
        "rtl_review": "not_applicable_no_successor_rtl",
        "ppa_run": False,
        "quality_discriminator_complete": False,
    }
    manifest["claim_boundaries"] = [
        "The residual cross-term predecessor is sealed by its C4-en final-lm-head verification no-go.",
        "The V-residual value-path successor is architecture-frozen with no RTL implementation authority.",
        "The accepted prefix, 2.0 mm2 cap, 100 MHz floor, streaming boundary, and historical PPA frontier are unchanged.",
        "No paired smoke, shell regression, candidate PPA, prototype, benchmark, signoff, tapeout, or silicon result followed.",
    ]
    manifest["interfaces_contract_status"] = "v_residual_value_path_interface_reconciled_fresh_architecture_review_pending_no_successor_rtl"
    traceability = manifest.setdefault("traceability", {})
    traceability["architecture_contract_gap"] = {
        "resolution_owner": "fresh_independent_architecture_reviewer_then_environment_reviewer_then_manager",
        "status": "v_residual_contract_reconciled_on_paper_no_successor_rtl",
    }
    traceability["selected_mechanism"] = SUCCESSOR
    traceability["rtl.contract-traceability"] = False
    traceability["stage_checklist"] = {
        "rtl.contract-traceability": False,
        "rtl.hardware-discipline": False,
        "rtl.ip-provenance": False,
    }
    manifest["generated_at_utc"] = timestamp
    dump(path, manifest)


def write_freeze(timestamp: str, current: dict[str, Any]) -> None:
    existing = load(FREEZE)
    freeze = {
        "schema_version": 3,
        "frozen_at_utc": timestamp,
        "prior_freeze_lineage_at_utc": existing.get("frozen_at_utc"),
        "authority": current["authority"],
        "contract_id": SUCCESSOR,
        "sealed_predecessor_contract_id": CONTRACT,
        "sealed_predecessor_verification_no_go": artifact(VERIFICATION_NO_GO.relative_to(ROOT).as_posix()),
        "proposal": artifact("design/NUMERICAL_REPLACEMENT_PROPOSAL.md"),
        "architecture_description": artifact("design/ARCHITECTURE.md"),
        "interface_spec": artifact("design/SPEC.md"),
        "memory_model": artifact("design/MEMORY_MODEL.json"),
        "chip_scope": artifact("design/CHIP_SCOPE.json"),
        "rtl_manifest": artifact("design/RTL_MANIFEST.json"),
        "binding_script": artifact("tools/bind_residual_cross_term_architecture_contract.py"),
        "architecture_refreeze_review_packet": artifact(REVIEW_PACKET.relative_to(ROOT).as_posix()),
        "post_refreeze_contract_blocker": artifact(BLOCKER.relative_to(ROOT).as_posix()),
        "current_stage": "architecture",
        "ordered_supported_layer_operator_prefix": PREFIX,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "targets": {"area_cap_non_sram_mm2": 2.0, "frequency_floor_mhz": 100.0},
        "historical_ppa_frontier": HISTORICAL_PPA,
        "abstract_streaming_memory_boundary_bits": 128,
        "downstream_execution_locked": True,
        "locked_stages": ["environment", "rtl", "verification", "ppa", "prototype", "benchmark", "signoff"],
        "selected_mechanism": SUCCESSOR,
        "claim_boundary": "Architecture successor freeze only; predecessor RTL remains sealed and no downstream claim is made.",
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    freeze["integrity"]["canonical_sha256"] = canonical_sha256(freeze)
    dump(FREEZE, freeze)


def update_public(timestamp: str, current: dict[str, Any]) -> None:
    path = ROOT / "research/PUBLIC_STATUS.json"
    status = load(path)
    rows = [successor_row_estimate(valid_keys) for valid_keys in SUCCESSOR_REPRESENTATIVE_ROWS]
    row_32768 = rows[-1]
    prefill = successor_prefill_estimate(SUCCESSOR_PREFILL_TOKENS)
    status["architecture_proposal_gate"] = {
        "contract_id": SUCCESSOR,
        "implementation_authorized": False,
        "operator_approval_consumed": False,
        "stage_closing": False,
        "status": "fresh_independent_architecture_review_pending",
        "source": current["proposal"],
        "proposal_sha256": current["proposal_sha256"],
        "required_manager_action": current["required_manager_action"],
        "sealed_predecessor_rtl_hash": SEALED_RTL_HASH,
        "sealed_predecessor_rtl_hash_scope": "historical_no_current_execution_authority",
    }
    status["stage"] = {
        "completed_prior_stages": ["definition"],
        "current_stage": "architecture",
        "current_stage_status": PACKET_STATUS,
        "current_stage_checklist": {
            "architecture.compute-memory-model": True,
            "architecture.interface-control": True,
            "architecture.leverage-risk": True,
            "architecture.area-reuse-plan": True,
        },
        "current_stage_evidence": [
            "design/ARCHITECTURE.md", "design/NUMERICAL_REPLACEMENT_PROPOSAL.md", "design/SPEC.md",
            "design/MEMORY_MODEL.json", "design/RTL_MANIFEST.json",
            REVIEW_PACKET.relative_to(ROOT).as_posix(), VERIFICATION_NO_GO.relative_to(ROOT).as_posix(),
        ],
    }
    status["latest_decision"] = "verification_rollback_v_residual_value_correction_architecture_frozen"
    status["routing_status"] = "fresh_independent_architecture_then_environment_review_required"
    status["selected_replacement_contract"] = copy.deepcopy(current)
    status["supported_layer_operator_prefix"] = PREFIX
    status["ordered_supported_layer_operator_prefix"] = PREFIX
    status["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
    status["latest_environment_stage"] = {
        "contract_binding": SUCCESSOR,
        "status": "pending_after_fresh_independent_architecture_acceptance",
        "prior_environment_evidence_reusable_only_as_historical_tool_capability_evidence": True,
        "implementation_authorized": False,
    }
    status["blockers"] = [{
        "id": "fresh_independent_v_residual_architecture_review_pending",
        "stage": "architecture",
        "status": "active",
        "reason": "The reconciled V-residual architecture packet requires independent acceptance before environment review.",
        "required_resolution": "Bind the architecture review packet, then perform fresh independent environment acceptance before any Manager advance to RTL.",
        "evidence": REVIEW_PACKET.relative_to(ROOT).as_posix(),
    }]
    public_candidate = {
        "contract_id": SUCCESSOR,
        "implementation_authorized": False,
        "mechanism": current["mechanism"],
        "predecessor_contract_id": CONTRACT,
        "proposal": current["proposal"],
        "proposal_sha256": current["proposal_sha256"],
        "status": PACKET_STATUS,
        "structurally_distinct": True,
    }
    performance = {
        "contract_id": SUCCESSOR,
        "kv_record_bytes": SUCCESSOR_KV_RECORD_BYTES,
        "candidate_workspace_bytes": SUCCESSOR_WORKSPACE["total"],
        "planned_sram_peak_bytes": SUCCESSOR_SRAM_BUDGET["total_planned_peak"],
        "remaining_sram_margin_bytes": SUCCESSOR_SRAM_BUDGET["remaining_planning_margin"],
        "decode_context_32768_total_cycles_at_16_bytes_per_cycle": row_32768["nonoverlapped_total_cycles_at_16_bytes_per_cycle"],
        "decode_context_32768_external_bytes": row_32768["external_bytes"],
        "prefill_128_dma_cycles_at_16_bytes_per_cycle": prefill["dma_cycles_at_16_bytes_per_cycle"],
        "prefill_128_total_cycles_at_16_bytes_per_cycle": prefill["nonoverlapped_total_cycles_at_16_bytes_per_cycle"],
        "prefill_row_aggregation": prefill["row_aggregation"],
        "prefill_cross_row_payload_beat_packing": prefill["cross_row_payload_beat_packing"],
        "candidate_core_fraction_at_32768_keys": row_32768["candidate_core_fraction"],
        "dma_fraction_at_16_bytes_per_cycle_at_32768_keys": row_32768["dma_fraction_at_16_bytes_per_cycle"],
        "fourteen_path_speedup_bound_at_32768_keys": row_32768["speedup_from_fourteen_value_paths_and_two_projection_finalizers_at_16_bytes_per_cycle"],
        "selected_incremental_logic_area_proxy_units": 521,
        "status": "architecture_estimate_reconciled_not_measured",
        "source": "design/MEMORY_MODEL.json",
    }
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status[name]
        container["candidate_mechanism"] = copy.deepcopy(public_candidate)
        container["current_architecture_performance_model"] = copy.deepcopy(performance)
        container["current_stage"] = "architecture"
        container["latest_decision"] = status["latest_decision"]
        container["latest_environment_stage"] = copy.deepcopy(status["latest_environment_stage"])
        container["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
        container["ordered_supported_layer_operator_prefix"] = PREFIX
        container["supported_layer_operator_prefix"] = PREFIX
        container["candidate_rtl_hash"] = SEALED_RTL_HASH
        container["candidate_rtl_hash_scope"] = "historical_sealed_predecessor_rtl_no_current_execution_authority"
        container["routing_status"] = status["routing_status"]
        container["rtl_contract_traceability"] = False
        container["required_operator_action"] = "none_fresh_independent_reviews_and_manager_advance_required"
        container["required_manager_action"] = current["required_manager_action"]
        operator_policy = container.get("operator_policy")
        if isinstance(operator_policy, dict):
            operator_policy["active_repair_authorization"] = copy.deepcopy(current)
            operator_policy["architecture_proposal_authorization"] = copy.deepcopy(current)
            operator_policy["selected_replacement_contract"] = copy.deepcopy(current)
            operator_policy["manager_recommendation"] = current["required_manager_action"]
            operator_policy["steady_state_stage"] = "architecture"
    status["latest_rtl_candidate"] = {
        "contract_id": CONTRACT,
        "candidate_rtl_hash": SEALED_RTL_HASH,
        "rtl_hash_scope": "historical_sealed_predecessor_rtl_no_current_execution_authority",
        "result_binding": VERIFICATION_NO_GO.relative_to(ROOT).as_posix(),
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "status": "sealed_verification_no_go_historical_only",
    }
    status["routing_authorized"] = "manager_rollback_complete_fresh_independent_architecture_review_required"
    status["rtl_contract_traceability"] = False
    status["stage_closing"] = False
    status["public_claims"] = [
        {"claim": "the residual cross-term candidate is a sealed verification no-go", "evidence": [VERIFICATION_NO_GO.relative_to(ROOT).as_posix()]},
        {"claim": "one structurally distinct V-residual value-path successor is frozen for fresh independent architecture review", "evidence": ["design/ARCHITECTURE.md", REVIEW_PACKET.relative_to(ROOT).as_posix()]},
        {"claim": "the accepted prefix, immutable targets, streaming boundary, and historical PPA frontier are unchanged", "evidence": ["design/CHIP_SCOPE.json", FREEZE.relative_to(ROOT).as_posix()]},
        {"claim": "no new RTL or downstream run is authorized", "evidence": [BLOCKER.relative_to(ROOT).as_posix()]},
    ]
    status["artifact_hashes"] = [
        artifact(relative)
        for relative in (
            "design/ARCHITECTURE.md",
            "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
            "design/SPEC.md",
            "design/MEMORY_MODEL.json",
            "design/CHIP_SCOPE.json",
            "design/RTL_MANIFEST.json",
            "research/PIPELINE_STATE.json",
            REVIEW_PACKET.relative_to(ROOT).as_posix(),
            BLOCKER.relative_to(ROOT).as_posix(),
            FREEZE.relative_to(ROOT).as_posix(),
            VERIFICATION_NO_GO.relative_to(ROOT).as_posix(),
            "tools/bind_residual_cross_term_architecture_contract.py",
        )
    ]
    status["generated_at_utc"] = timestamp
    status["last_updated_utc"] = timestamp
    status.setdefault("integrity", {})["canonical_sha256"] = None
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump(path, status)


def bind_successor(timestamp: str) -> None:
    rollback_pipeline(timestamp)
    current = successor_contract(timestamp)
    update_scope(timestamp, current)
    update_successor_memory(timestamp, current)
    write_blocker(timestamp)
    write_packet(timestamp, current)
    update_manifest(timestamp, current)
    write_freeze(timestamp, current)
    update_public(timestamp, current)


def validate_successor() -> None:
    no_go = load(VERIFICATION_NO_GO)
    require(no_go["decision"] == "bounded_no_go_focused_discriminator_c4_lm_head_failed", "verification no-go decision changed")
    require(no_go["candidate_rtl_hash"] == SEALED_RTL_HASH, "sealed predecessor RTL hash changed")
    c4 = no_go["focused_discriminator"]["comparisons"]["c4_en_512"]["lm_head"]
    require(c4["baseline_relative_l2"] == 1.1934673932274116, "C4 baseline metric changed")
    require(c4["candidate_relative_l2"] == 1.1984861902850406, "C4 candidate metric changed")
    require(no_go["integrity"]["canonical_sha256"] == canonical_sha256(no_go), "verification no-go integrity changed")
    pipeline = load(ROOT / "research/PIPELINE_STATE.json")
    require(pipeline["current_stage"] == "architecture", "Manager rollback missing")
    require(any(SUCCESSOR in item.get("reason", "") for item in pipeline["stage_history"]), "successor rollback history missing")
    architecture = (ROOT / "design/ARCHITECTURE.md").read_text(encoding="utf-8")
    proposal = PROPOSAL.read_text(encoding="utf-8")
    spec = (ROOT / "design/SPEC.md").read_text(encoding="utf-8")
    for phrase in (SUCCESSOR, "1.1934673932274116", "1.1984861902850406", "2.0 mm2", "100 MHz"):
        require(phrase in architecture + proposal + spec, f"architecture contract phrase missing: {phrase}")
    require("three score cross terms" in proposal, "sealed mechanism prohibition missing")
    require("preserves the complete attention-distribution path" in architecture, "C4 propagation response missing")
    for stale in (
        "416-byte", "37,888-byte", "415,488", "residual-Q", "residual-K",
        "adds `896*L` probability/residual-V MAC cycles and `112*L` Scale32 conversion",
    ):
        require(stale not in spec + architecture + proposal, f"stale predecessor construct remains in architecture text: {stale}")
    for required in (
        "400-byte", "16,384", "393,984", "130,304", "7,168",
        "1048*L + 1088 + 448*T", "100,663,296", "baseline-V Scale32", "divided by 14",
    ):
        require(required in spec + architecture + proposal, f"reconciled V-residual derivation missing: {required}")
    memory = load(ROOT / "design/MEMORY_MODEL.json")
    expected_rows = [successor_row_estimate(valid_keys) for valid_keys in SUCCESSOR_REPRESENTATIVE_ROWS]
    expected_prefill = successor_prefill_estimate(SUCCESSOR_PREFILL_TOKENS)
    require(memory["stage"] == "architecture", "memory model stage stale")
    require(memory["authority_override"]["contract_id"] == SUCCESSOR, "memory authority contract stale")
    require(memory["authority_override"]["implementation_authorized"] is False, "memory model self-authorized RTL")
    require(memory["numeric_contract"]["contract_id"] == SUCCESSOR, "memory numeric contract stale")
    require(memory["candidate_interface_contract"] == successor_interface_contract(), "memory interface contract stale")
    require(memory["execution_schedule"]["kv_record_bytes_per_token"] == SUCCESSOR_KV_RECORD_BYTES, "memory K/V record stale")
    require(memory["execution_schedule"]["workspace_layout_bytes"] == SUCCESSOR_WORKSPACE, "memory workspace derivation stale")
    require(memory["sram_live_set_budget_bytes"] == SUCCESSOR_SRAM_BUDGET, "memory SRAM budget stale")
    require(memory["bandwidth_accounting"]["per_valid_key_bytes"]["total"] == 1048, "memory traffic total stale")
    require(memory["decode_context_estimates"] == [
        {
            "context_tokens": row["valid_keys"],
            "tiles": row["tiles"],
            "external_bytes": row["external_bytes"],
            "arithmetic_cycles": row["arithmetic_cycles"],
            "dma_cycles_at_16_bytes_per_cycle": row["dma_cycles_at_16_bytes_per_cycle"],
            "nonoverlapped_total_cycles_at_16_bytes_per_cycle": row["nonoverlapped_total_cycles_at_16_bytes_per_cycle"],
        }
        for row in expected_rows
    ], "memory decode estimates stale")
    require(memory["prefill_estimates"] == [expected_prefill], "memory prefill estimate stale")
    require(memory["resource_sharing_tradeoffs"] == successor_resource_sharing(expected_rows), "memory sharing derivation stale")
    require(len(memory["arithmetic_intensity_and_utilization"]["external_link_utilization"]) == 15, "memory bandwidth sweep incomplete")
    memory_encoded = json.dumps(memory, sort_keys=True)
    for stale in (
        "ATTN_SCORE_CORRECTION_INTERNAL", "residual_query_rope",
        "residual_cross_term_engine_cycles", "residual_k_tile",
        '"kv_record_bytes_per_token": 416', '"candidate_workspace": 37888',
    ):
        require(stale not in memory_encoded, f"stale predecessor field remains in memory model: {stale}")
    packet = load(REVIEW_PACKET)
    require(packet["current_stage"] == "architecture", "review packet stage stale")
    require(packet["status"] == PACKET_STATUS, "review packet status stale")
    require(packet["contract_id"] == SUCCESSOR, "review packet successor stale")
    require(packet["selected_solution"]["structurally_distinct"] is True, "successor is not structurally distinct")
    require(packet["implementation_authorized"] is False, "packet self-authorized RTL")
    require(packet["preserved_contracts"] == PRESERVED, "preserved packet contracts changed")
    require(packet["acceptance"]["architecture"]["status"] == "pending_fresh_independent_reviewer", "architecture acceptance self-claimed")
    require(packet["acceptance"]["environment"]["status"] == "pending_after_architecture_acceptance", "environment acceptance self-claimed")
    require(packet["compute_memory_budget"]["kv_record_bytes"] == SUCCESSOR_KV_RECORD_BYTES, "packet K/V record stale")
    require(packet["compute_memory_budget"]["candidate_workspace_layout_bytes"] == SUCCESSOR_WORKSPACE, "packet workspace stale")
    require(packet["compute_memory_budget"]["added_conversion_cycles_per_row"] == 7168, "packet conversion schedule stale")
    require(packet["compute_memory_derivation"]["representative_rows"] == expected_rows, "packet row derivation stale")
    require(packet["compute_memory_derivation"]["prefill_128"] == expected_prefill, "packet prefill derivation stale")
    require(packet["resource_sharing_tradeoffs"] == successor_resource_sharing(expected_rows), "packet sharing derivation stale")
    require(packet["bindings"]["memory_model"] == artifact("design/MEMORY_MODEL.json"), "packet memory binding stale")
    for binding in packet["bindings"].values():
        require(sha256(ROOT / binding["path"]) == binding["sha256"], f"packet binding stale: {binding['path']}")
    require(packet["integrity"]["canonical_sha256"] == canonical_sha256(packet), "review packet integrity stale")
    blocker = load(BLOCKER)
    require(blocker["current_stage"] == "architecture", "blocker stage stale")
    require(blocker["preserved_contracts"] == PRESERVED, "blocker preserved contracts changed")
    require(blocker["integrity"]["canonical_sha256"] == canonical_sha256(blocker), "blocker integrity stale")
    freeze = load(FREEZE)
    require(freeze["contract_id"] == SUCCESSOR, "freeze successor stale")
    require(freeze["ordered_supported_layer_operator_prefix"] == PREFIX, "freeze prefix changed")
    require(freeze["first_unsupported_layer_operator"] == FIRST_UNSUPPORTED, "freeze first unsupported changed")
    require(freeze["targets"] == {"area_cap_non_sram_mm2": 2.0, "frequency_floor_mhz": 100.0}, "freeze targets changed")
    require(freeze["historical_ppa_frontier"] == HISTORICAL_PPA, "historical PPA changed")
    require(freeze["memory_model"] == artifact("design/MEMORY_MODEL.json"), "freeze memory binding stale")
    require(freeze["architecture_refreeze_review_packet"] == artifact(REVIEW_PACKET.relative_to(ROOT).as_posix()), "freeze packet binding stale")
    require(freeze["integrity"]["canonical_sha256"] == canonical_sha256(freeze), "freeze integrity stale")
    scope = load(ROOT / "design/CHIP_SCOPE.json")
    require(scope["stage"]["current_stage"] == "architecture", "chip scope stage stale")
    require(scope["authority_override"]["active_replacement_contract"] == SUCCESSOR, "chip scope active contract stale")
    require(scope["authority_override"]["implementation_authorized"] is False, "chip scope self-authorized RTL")
    require(scope["implementation_frontier"]["ordered_supported_layer_operator_prefix"] == PREFIX, "chip scope prefix changed")
    require(scope["implementation_frontier"]["first_unsupported_layer_operator"] == FIRST_UNSUPPORTED, "chip scope first unsupported changed")
    require("active_qk_residual_cross_term_contract" not in scope["numerical_behavior"], "chip scope retains active predecessor key")
    require(scope["numerical_behavior"]["active_v_residual_value_correction_contract"]["kv_record"] == successor_record_contract(), "chip scope V record stale")
    manifest = load(ROOT / "design/RTL_MANIFEST.json")
    require(manifest["current_stage"] == "architecture", "RTL manifest stage stale")
    require(manifest["candidate_rtl_hash"] == SEALED_RTL_HASH, "historical RTL hash changed")
    require(manifest["proposed_replacement_contract"]["contract_id"] == SUCCESSOR, "manifest successor stale")
    require(manifest["architecture_refreeze_review_packet"] == artifact(REVIEW_PACKET.relative_to(ROOT).as_posix()), "manifest packet binding stale")
    require(manifest["candidate_id"] == "v_residual_value_correction_architecture_only", "manifest candidate id stale")
    require(manifest["candidate_status"] == PACKET_STATUS, "manifest candidate status stale")
    require(manifest["candidate_evidence_hashes"]["contract_id"] == SUCCESSOR, "manifest evidence contract stale")
    require(manifest["candidate_evidence_hashes"]["memory_model_sha256"] == sha256(ROOT / "design/MEMORY_MODEL.json"), "manifest memory hash stale")
    require(manifest["candidate_generated_hashes"] == {}, "manifest exposes predecessor generated hashes as current")
    require(manifest["candidate_generated_sources"] == [], "manifest exposes predecessor generated sources as current")
    require(manifest["candidate_source_hashes"] == {}, "manifest exposes predecessor source hashes as current")
    require(manifest["candidate_rtl_sources"] == [], "manifest exposes predecessor RTL sources as current")
    require(manifest["candidate_interface"]["implemented_modules"] == [], "manifest claims successor RTL modules")
    require(manifest["candidate_interface"]["planned_module"] == "ace2_v_residual_value_correction_core", "manifest planned interface stale")
    require(manifest["candidate_geometry"]["kv_record_bytes"] == SUCCESSOR_KV_RECORD_BYTES, "manifest K/V record stale")
    require(manifest["candidate_geometry"]["candidate_workspace_bytes"] == SUCCESSOR_WORKSPACE["total"], "manifest workspace stale")
    require(manifest["candidate_geometry"]["planned_sram_peak_bytes"] == SUCCESSOR_SRAM_BUDGET["total_planned_peak"], "manifest SRAM peak stale")
    require(manifest["candidate_model_metadata"]["record_count"] == 48, "manifest metadata count stale")
    require(manifest["candidate_model_metadata"]["generated_table"] is None, "manifest claims generated successor metadata")
    require(manifest["candidate_schedule"]["residual_value_conversion_cycles"] == 7168, "manifest conversion schedule stale")
    reviewer_acceptance = manifest["independent_reviewer_acceptance"]
    require(
        reviewer_acceptance["status"] == "pending_fresh_independent_architecture_review",
        "manifest reviewer acceptance status stale",
    )
    require(
        "independent_reviewer_verdict" not in manifest,
        "manifest exposes an unscoped reviewer verdict while current architecture acceptance is pending",
    )
    require(
        manifest["candidate_review_binding"]["decision"] == "fresh_independent_architecture_review_pending"
        and manifest["candidate_review_binding"]["candidate_capability_accepted"] is False
        and manifest["candidate_review_binding"]["stage_closing"] is False,
        "manifest reviewer verdict and acceptance state are inconsistent",
    )
    require(manifest["traceability"]["selected_mechanism"] == SUCCESSOR, "manifest traceability mechanism stale")
    require(manifest["traceability"]["rtl.contract-traceability"] is False, "manifest self-claims successor RTL traceability")
    require(manifest["historical_standalone_rtl"]["candidate_rtl_hash"] == SEALED_RTL_HASH, "manifest historical predecessor hash missing")
    public = load(ROOT / "research/PUBLIC_STATUS.json")
    require(public["stage"]["current_stage"] == "architecture", "public stage stale")
    require(public["selected_replacement_contract"]["contract_id"] == SUCCESSOR, "public successor stale")
    require(public["architecture_proposal_gate"]["contract_id"] == SUCCESSOR, "public architecture gate stale")
    require(public["supported_layer_operator_prefix"] == PREFIX, "public prefix changed")
    require(public["first_unsupported_layer_operator"] == FIRST_UNSUPPORTED, "public first unsupported changed")
    for name in ("dashboard_fields", "implementation_frontier"):
        container = public[name]
        require(container["candidate_mechanism"]["contract_id"] == SUCCESSOR, f"public {name} candidate stale")
        require(container["current_architecture_performance_model"]["contract_id"] == SUCCESSOR, f"public {name} performance contract stale")
        require(container["current_architecture_performance_model"]["kv_record_bytes"] == SUCCESSOR_KV_RECORD_BYTES, f"public {name} K/V record stale")
        require(container["current_architecture_performance_model"]["planned_sram_peak_bytes"] == SUCCESSOR_SRAM_BUDGET["total_planned_peak"], f"public {name} SRAM peak stale")
        require(container["current_architecture_performance_model"]["decode_context_32768_total_cycles_at_16_bytes_per_cycle"] == expected_rows[-1]["nonoverlapped_total_cycles_at_16_bytes_per_cycle"], f"public {name} decode estimate stale")
        require(container["current_architecture_performance_model"]["prefill_128_total_cycles_at_16_bytes_per_cycle"] == expected_prefill["nonoverlapped_total_cycles_at_16_bytes_per_cycle"], f"public {name} prefill estimate stale")
        require(container["candidate_rtl_hash_scope"] == "historical_sealed_predecessor_rtl_no_current_execution_authority", f"public {name} predecessor RTL scope stale")
        require(container["rtl_contract_traceability"] is False, f"public {name} self-claims successor RTL traceability")
        operator_policy = container.get("operator_policy")
        if isinstance(operator_policy, dict):
            for key in ("active_repair_authorization", "architecture_proposal_authorization", "selected_replacement_contract"):
                require(operator_policy[key]["contract_id"] == SUCCESSOR, f"public {name} operator policy stale: {key}")
                require(operator_policy[key]["implementation_authorized"] is False, f"public {name} operator policy self-authorized: {key}")
    for item in public["artifact_hashes"]:
        require(sha256(ROOT / item["path"]) == item["sha256"], f"public artifact hash stale: {item['path']}")
    require("/home/" not in json.dumps(public, sort_keys=True), "public status leaks private path")
    require(public["integrity"]["canonical_sha256"] == canonical_sha256(public), "public integrity stale")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        bind_successor(utc_now())
    validate_successor()
    print(
        f"ACE2_ARCHITECTURE_REFREEZE_PASS stage=architecture contract={SUCCESSOR} "
        f"prefix={len(PREFIX)} first_unsupported={FIRST_UNSUPPORTED}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
