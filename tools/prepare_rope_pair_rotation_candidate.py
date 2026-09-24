#!/usr/bin/env python3
"""Bind the selected RoPE-commuting Q/K basis candidate to RTL-stage evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "rope_pair_rotation" / "latest"
ARCHIVE = ROOT / "evidence" / "rope_pair_rotation" / "archive"
RTL_MANIFEST = ROOT / "design" / "RTL_MANIFEST.json"
FAST_LOOP_POLICY = ROOT / "design" / "FAST_LOOP_POLICY.json"
PUBLIC_STATUS = ROOT / "research" / "PUBLIC_STATUS.json"
NUMERICAL_RTL = [
    "rtl/ace2_pkg.sv",
    "rtl/ace2_rmsnorm_core.sv",
    "rtl/ace2_w4a8_proj_core.sv",
    "rtl/ace2_rope_core.sv",
    "rtl/ace2_attention_score_core.sv",
    "rtl/ace2_softmax_core.sv",
    "rtl/ace2_attention_compose_core.sv",
    "rtl/ace2_silu_gate_core.sv",
    "rtl/generated/ace2_silu_lut.svh",
    "rtl/ace2_shell.sv",
]
ACCEPTED_PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
ANGLE_DEGREES = 22.5
ANGLE_HALF_DEGREES = 45


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def artifact(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {
        "bytes": path.stat().st_size,
        "path": relative,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_manifest(now: str, source_artifacts: list[dict[str, Any]], rtl_hash: str) -> None:
    manifest = json.loads(RTL_MANIFEST.read_text(encoding="utf-8"))
    manifest.update(
        {
            "generated_at_utc": now,
            "stage": "rtl",
            "stage_closing": False,
            "supported_layer_operator_prefix": ACCEPTED_PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "candidate_layer_operator": "rope_commuting_qk_pair_rotation_22_5deg",
            "candidate_status": "focused_rtl_pass_pending_frozen_paired_smoke",
            "candidate_meets_numeric_acceptance": False,
            "candidate_requires_fresh_sky130_ppa": False,
            "candidate_verification_complete": False,
            "candidate_first_unsupported_layer_operator_after_review": FIRST_UNSUPPORTED,
            "candidate_supported_layer_operator_prefix_after_review": ACCEPTED_PREFIX,
            "candidate_reusable_operator_capabilities_after_review": [
                "input_rmsnorm",
                "w4a8_projection_q_proj",
                "w4a8_projection_k_proj",
                "w4a8_projection_v_proj",
            ],
            "candidate_rtl_hash": rtl_hash,
            "candidate_source_hashes": source_artifacts,
            "source_hashes": source_artifacts,
            "candidate_geometry": {
                "qk_basis_rotation_angle_degrees": ANGLE_DEGREES,
                "qk_basis_rotation_half_degree_command_tag": ANGLE_HALF_DEGREES,
                "runtime_arithmetic_added": False,
                "rope_lanes": 2,
                "projection_mac_lanes": 4,
                "attention_mac_lanes": 1,
            },
            "interfaces": {
                "clock": "clk_i single accelerator clock",
                "command_dispatch": "existing direct descriptor ingress; RoPE commands require cmd_flags_i=45 half-degree units to bind the fused +22.5 degree Q/K basis",
                "completion": "cmd_done_valid/cmd_done_ready plus tag, error, sumsq, inv_rms_q30, saturation_seen",
                "csr": "32-bit address, 64-bit data, valid/ready request with valid response",
                "external_memory_dma": "abstract 128-bit request/write/read/response ready-valid channels with 8-bit tags and 16-bit burst length",
                "qk_weight_contract": "the same +22.5 degree orthogonal rotation is fused into each Q/K projection RoPE pair; no per-head Q/K metadata table is used",
                "reset": "rst_ni active-low asynchronous assertion at the module boundary",
                "scratchpad": "8 logical 128-bit SRAM-bank ports are exposed as blackbox macro boundary signals and held inactive by the current slice",
            },
            "claim_boundaries": [
                "The rejected per-head Q/K result remains preserved in evidence/per_head_qk_repair/latest/RESULTS.json.",
                "This candidate is a non-stage-closing RTL task and has only focused lint/RoPE evidence before the paired-smoke gate.",
                "No candidate full-shell, synthesis, STA, power, PPA, prototype, benchmark, signoff, tapeout, or silicon claim is made.",
                "The historical accepted PPA frontier is not republished for this numerical contract.",
            ],
            "candidate_verification_binding": {
                "classification": "focused_rtl_checks_not_independent_verification_stage",
                "rtl_lint": artifact("evidence/frontier/latest/rtl_lint.log"),
                "rtl_rope_core": artifact("evidence/frontier/latest/rtl_rope_sim.log"),
                "rtl_rope_shell": artifact("evidence/frontier/latest/rtl_rope_shell_sim.log"),
            },
            "traceability": {
                "rtl.contract-traceability": True,
                "rtl.hardware-discipline": True,
                "rtl.ip-provenance": True,
                "selected_mechanism": "shared_rope_commuting_qk_pair_rotation_22_5deg_fused_into_weights",
                "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            },
        }
    )
    parameters = manifest.setdefault("parameters", {})
    for key in (
        "ATTENTION_HEAD_METADATA_BYTES",
        "ATTENTION_HEAD_METADATA_RECORDS",
        "ATTENTION_KV_HEADS",
        "ATTENTION_QUERY_HEADS",
    ):
        parameters.pop(key, None)
    parameters.update(
        {
            "QK_BASIS_ROTATION_ANGLE_DEGREES": ANGLE_DEGREES,
            "QK_BASIS_ROTATION_COMMAND_FLAG_HALF_DEGREES": ANGLE_HALF_DEGREES,
            "QK_BASIS_RUNTIME_ARITHMETIC_ADDED": False,
            "ROPE_SCALE_GRANULARITY": "static_per_tensor_repeated_across_q_or_k_elements",
            "ATTN_SCORE_METADATA": "one signed-int32 multiplier and unsigned-6-bit right shift per descriptor",
        }
    )
    write_json(RTL_MANIFEST, manifest)


def update_policy(now: str, rtl_hash: str) -> None:
    policy = json.loads(FAST_LOOP_POLICY.read_text(encoding="utf-8"))
    policy["active_repair_authorization"] = {
        "authority": "operator",
        "authorization": "approved_structurally_different_replan",
        "scope": "shared_rope_commuting_qk_pair_rotation_22_5deg_fused_into_projection_weights",
        "excluded_direction": "no_additional_per_head_qk_variant",
        "baseline_quality_evidence": "benchmark/raw/quality/operator-paired-smoke-20260731T081721Z/results.json",
        "rejected_direction_evidence": "evidence/per_head_qk_repair/latest/RESULTS.json",
        "selection_probe_evidence": [
            "evidence/diagnostics/rope-structural-probe-control-r3-20260731/results.json",
            "evidence/diagnostics/rope-pair-angle-22p5-probe-r2-20260731/results.json",
        ],
        "selection_probe_improvement_percent": {
            "c4_en_512": 44.570833747207786,
            "wikitext2": 1.880967268273229,
        },
        "implementation_scope": [
            "fuse one shared plus-22.5-degree orthogonal rotation into every Q/K RoPE pair",
            "retain scalar per-tensor signed-int8 Q/K scales",
            "retain range-safe Q6.9 RoPE conversion",
            "bind RoPE command flags to 45 half-degree units",
            "add no runtime arithmetic datapath",
        ],
        "observed_candidate_rtl_hash": rtl_hash,
        "execution_order": [
            "small_operator_probes",
            "focused_rtl_implementation_and_tests",
            "short_paired_smoke_wikitext2",
            "short_paired_smoke_c4_en_512",
            "conditional_single_full_shell_regression",
            "conditional_single_canonical_sky130_ppa",
            "evidence_binding",
            "independent_reviewer_gate",
        ],
        "execution_status": "focused_rtl_pass_pending_short_paired_smoke",
        "short_smoke_gate": {
            "comparison": "same_frozen_128_token_inputs_as_baseline",
            "metric": "w4a8_to_bf16_perplexity_ratio",
            "pass_condition": "candidate_ratio_strictly_lower_than_corresponding_baseline_ratio_on_both_datasets",
            "required_datasets": ["wikitext2", "c4_en_512"],
            "missing_or_equal_result": "fail_closed_stop_task",
        },
        "smoke_gate_passed": False,
        "full_shell_regression_run": False,
        "canonical_sky130_ppa_run": False,
        "stage_closing": False,
        "task_count": 1,
        "immutable_targets": {
            "area_cap_non_sram_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "quality_target_ratio_max": 1.05,
            "targets_relaxed": False,
        },
        "updated_at_utc": now,
    }
    write_json(FAST_LOOP_POLICY, policy)


def update_public_status(now: str, rtl_hash: str) -> None:
    status = json.loads(PUBLIC_STATUS.read_text(encoding="utf-8"))
    status["generated_at_utc"] = now
    status["last_updated_utc"] = now
    stage = status.setdefault("stage", {})
    stage["current_stage"] = "rtl"
    stage["current_stage_status"] = "bounded_structural_candidate_focused_rtl_pass_pending_smoke"
    stage["current_stage_evidence"] = [
        "design/RTL_MANIFEST.json",
        "design/RTL_TRACEABILITY.md",
        "evidence/frontier/latest/rtl_lint.log",
        "evidence/frontier/latest/rtl_rope_sim.log",
        "evidence/frontier/latest/rtl_rope_shell_sim.log",
    ]
    status["blockers"] = [
        {
            "id": "rope_pair_rotation_paired_smoke_pending",
            "stage": "rtl",
            "status": "active",
            "reason": "The selected +22.5 degree fused Q/K basis has focused RTL evidence but has not yet passed both comparable 128-token smokes.",
            "required_resolution": "Run the frozen WikiText-2 and C4-en short smokes; stop unless both ratios strictly improve.",
            "evidence": "design/RTL_MANIFEST.json",
        }
    ]
    frontier = status.setdefault("implementation_frontier", {})
    frontier.update(
        {
            "current_stage": "rtl",
            "current_mode": "ADVANCE",
            "supported_layer_operator_prefix": ACCEPTED_PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "latest_decision": "selected_rope_commuting_qk_pair_rotation_22_5deg_after_joint_probe",
            "candidate_rtl_hash": rtl_hash,
            "candidate_mechanism": {
                "angle_degrees": ANGLE_DEGREES,
                "command_flag_half_degree_units": ANGLE_HALF_DEGREES,
                "per_head_qk_variant": False,
                "runtime_arithmetic_added": False,
                "status": "focused_rtl_pass_pending_paired_smoke",
            },
            "latest_ppa_frontier_status": "historical_pre_selected_basis_contract",
        }
    )
    dashboard = status.setdefault("dashboard_fields", {})
    dashboard.update(
        {
            "supported_layer_operator_prefix": ACCEPTED_PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "current_mode": "ADVANCE",
            "latest_decision": "selected_rope_commuting_qk_pair_rotation_22_5deg_after_joint_probe",
            "candidate_rtl_hash": rtl_hash,
            "candidate_mechanism": frontier["candidate_mechanism"],
            "latest_ppa_frontier_status": "historical_pre_selected_basis_contract",
        }
    )
    claims = [
        claim
        for claim in status.get("public_claims", [])
        if "per-head" not in claim.get("claim", "").lower()
    ]
    claims.extend(
        [
            {
                "claim": "the rejected per-head Q/K direction is preserved as a bounded negative result and is not the active candidate",
                "evidence": ["evidence/per_head_qk_repair/latest/RESULTS.json"],
            },
            {
                "claim": "the active bounded RTL candidate uses a shared +22.5 degree RoPE-commuting Q/K basis rotation fused into projection weights with no runtime arithmetic",
                "evidence": [
                    "benchmark/quality/QUALITY_CONFIG.json",
                    "design/RTL_MANIFEST.json",
                    "evidence/frontier/latest/rtl_rope_shell_sim.log",
                ],
            },
            {
                "claim": "the accepted prefix remains through layer_0.v_proj and layer_0.rope_q remains first unsupported until both frozen short smokes improve",
                "evidence": ["design/RTL_MANIFEST.json", "research/PUBLIC_STATUS.json"],
            },
        ]
    )
    status["public_claims"] = claims
    write_json(PUBLIC_STATUS, status)


def main() -> None:
    now = utc_now()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    source_artifacts = [artifact(relative) for relative in NUMERICAL_RTL]
    source_lines = "".join(
        f"{item['sha256']}  {item['path']}\n" for item in source_artifacts
    )
    source_list = EVIDENCE / "source_hashes.before"
    source_list.write_text(source_lines, encoding="utf-8")
    rtl_hash = sha256_file(source_list)
    update_manifest(now, source_artifacts, rtl_hash)
    update_policy(now, rtl_hash)
    update_public_status(now, rtl_hash)
    packet = {
        "schema_version": 1,
        "generated_at_utc": now,
        "status": "rope_pair_rotation_candidate_ready_for_frozen_paired_smoke",
        "candidate_id": f"rope_pair_rotation_{rtl_hash[:16]}",
        "mechanism": {
            "angle_degrees": ANGLE_DEGREES,
            "command_flag_half_degree_units": ANGLE_HALF_DEGREES,
            "per_head_qk_variant": False,
            "runtime_arithmetic_added": False,
        },
        "source_binding": {
            "source_hash_list": source_list.relative_to(ROOT).as_posix(),
            "ordered_source_hash_list_sha256": rtl_hash,
        },
        "accepted_frontier_preservation": {
            "rtl_manifest_sha256": sha256_file(RTL_MANIFEST),
            "pipeline_stage_unchanged": "rtl",
            "stage_closing": False,
            "supported_prefix": ACCEPTED_PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        },
        "focused_checks": {
            "rtl_lint": artifact("evidence/frontier/latest/rtl_lint.log"),
            "rtl_rope_core": artifact("evidence/frontier/latest/rtl_rope_sim.log"),
            "rtl_rope_shell": artifact("evidence/frontier/latest/rtl_rope_shell_sim.log"),
        },
        "claim_boundaries": [
            "Candidate packet only; it is not verification, PPA, benchmark, or signoff evidence.",
            "The frozen two-dataset improvement gate must pass before a full-shell or PPA rerun.",
            "The per-head Q/K direction remains rejected and is not reused by this candidate.",
        ],
    }
    payload = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    packet_sha256 = sha256_bytes(payload.encode("utf-8"))
    output = EVIDENCE / "candidate_evidence.json"
    archive = ARCHIVE / f"candidate_evidence.{packet_sha256}.json"
    archive.write_text(payload, encoding="utf-8")
    output.write_text(payload, encoding="utf-8")
    print(
        "ACE2_ROPE_PAIR_ROTATION_CANDIDATE_PREP_PASS "
        f"candidate_id={packet['candidate_id']} rtl_hash={rtl_hash} "
        f"packet_sha256={packet_sha256}"
    )


if __name__ == "__main__":
    main()
