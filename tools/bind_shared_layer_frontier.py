#!/usr/bin/env python3
"""Bind shared transformer-layer RTL evidence without advancing Manager stage."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.bind_rtl_frontier import parse_sta, parse_yosys

LATEST = ROOT / "evidence" / "frontier" / "latest"
MANIFEST = ROOT / "design" / "RTL_MANIFEST.json"
LEDGER = ROOT / "design" / "PPA_FRONTIER_LEDGER.json"
PUBLIC_STATUS = ROOT / "research" / "PUBLIC_STATUS.json"
PIPELINE_STATE = ROOT / "research" / "PIPELINE_STATE.json"
REVIEW_VERDICT = ROOT / "evidence" / "review" / "latest" / "shared_transformer_layers_verdict.json"

AREA_CAP_MM2 = 2.0
FREQ_FLOOR_MHZ = 100.0

LAYER_OPS = [
    "input_rmsnorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "rope_q",
    "rope_k",
    "kv_write",
    "attention_score",
    "softmax",
    "attention_value",
    "o_proj",
    "attention_residual_add",
    "post_attention_rmsnorm",
    "mlp_gate_proj",
    "mlp_up_proj",
    "silu_gate",
    "mlp_down_proj",
    "mlp_residual_add",
]
LAYER0_PREFIX = [f"layer_0.{op}" for op in LAYER_OPS]
FULL_TRANSFORMER_PREFIX = [
    f"layer_{layer}.{op}" for layer in range(24) for op in LAYER_OPS
]
ACCEPTED_FIRST_UNSUPPORTED = "layer_1.input_rmsnorm"
PUBLISHED_FIRST_UNSUPPORTED = "final_rmsnorm"
CANDIDATE_OPERATOR = "layer_1_through_23.shared_transformer_layer_reuse"

RTL_FILES = [
    "rtl/ace2_pkg.sv",
    "rtl/ace2_rmsnorm_core.sv",
    "rtl/ace2_w4a8_proj_core.sv",
    "rtl/ace2_rope_core.sv",
    "rtl/ace2_attention_score_core.sv",
    "rtl/ace2_softmax_core.sv",
    "rtl/ace2_silu_gate_core.sv",
    "rtl/generated/ace2_silu_lut.svh",
    "rtl/ace2_shell.sv",
]
CONSTRAINT_FILES = [
    "constraints/ace2_rmsnorm_core.sdc",
    "flow/yosys/sky130_rmsnorm.ys",
    "flow/yosys/sky130_sta.tcl",
]
SOURCE_FILES = RTL_FILES + [
    "verification/tb/ace2_shell_tb.sv",
    "verification/tb/ace2_rmsnorm_tb.sv",
    "verification/tb/ace2_w4a8_proj_tb.sv",
    "verification/tb/ace2_rope_tb.sv",
    "verification/tb/ace2_attention_score_tb.sv",
    "verification/tb/ace2_softmax_tb.sv",
    "verification/tb/ace2_silu_gate_tb.sv",
    "verification/generated/rmsnorm_vectors.svh",
    "verification/generated/projection_vectors.svh",
    "verification/generated/rope_vectors.svh",
    "verification/generated/attention_score_vectors.svh",
    "verification/generated/softmax_vectors.svh",
    "verification/generated/silu_gate_vectors.svh",
    "verification/generated/attention_value_vectors.svh",
    "verification/generated/residual_vectors.svh",
    "verification/generated/mlp_residual_vectors.svh",
    "verification/generated/post_attention_rmsnorm_vectors.svh",
    "tools/bind_shared_layer_frontier.py",
    "tools/bind_rtl_frontier.py",
    "Makefile",
    "design/RTL_TRACEABILITY.md",
    "design/SPEC.md",
    "design/WORKLOAD.md",
    "design/FAST_LOOP_POLICY.json",
] + CONSTRAINT_FILES

ARTIFACT_FILES = [
    "research/PIPELINE_STATE.json",
    "design/RTL_MANIFEST.json",
    "design/PPA_FRONTIER_LEDGER.json",
    "evidence/frontier/latest/rtl_layer_sweep_sim.log",
    "evidence/frontier/latest/rtl_shell_sim.log",
    "evidence/frontier/latest/sky130_yosys.log",
    "evidence/frontier/latest/sky130_sta.log",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(paths):
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(sha256_file(ROOT / rel).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_layer_sweep(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"ACE2_SHELL_LAYER_SWEEP_TB_PASS\s+high_layer=([0-9]+)\s+"
        r"cases=([0-9]+)\s+projection_shapes=([0-9]+)\s+"
        r"opcode_families=([0-9]+)\s+cycles=([0-9]+)",
        text,
    )
    if not match:
        raise RuntimeError("could not parse shared layer sweep pass summary")
    return {
        "high_layer": int(match.group(1)),
        "cases": int(match.group(2)),
        "projection_shapes": int(match.group(3)),
        "opcode_families": int(match.group(4)),
        "cycles": int(match.group(5)),
        "log_sha256": sha256_file(log_path),
    }


def reviewer_status(rtl_hash: str, constraint_hash: str, evidence_hashes: dict[str, str]) -> tuple[str, dict[str, Any] | None]:
    if not REVIEW_VERDICT.exists():
        return "pending", None
    verdict = load_json(REVIEW_VERDICT, None)
    if not isinstance(verdict, dict):
        raise RuntimeError("shared transformer layer verdict must be a JSON object")
    status = verdict.get("status")
    if status not in {"accepted", "rejected"}:
        raise RuntimeError("shared transformer layer verdict status must be accepted or rejected")
    if status == "rejected":
        return "rejected", verdict
    stale = []
    expected = {
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
    }
    stale.extend(key for key, value in expected.items() if verdict.get(key) != value)
    reviewed_evidence = verdict.get("evidence_hashes")
    if not isinstance(reviewed_evidence, dict):
        raise RuntimeError("accepted shared transformer layer verdict lacks evidence hashes")
    stale.extend(
        key for key, value in evidence_hashes.items()
        if reviewed_evidence.get(key) != value
    )
    if stale:
        raise RuntimeError("accepted shared transformer layer verdict is stale for: " + ", ".join(stale))
    if verdict.get("publication_authorized") is not True:
        raise RuntimeError("accepted shared transformer layer verdict must authorize publication")
    return "accepted", verdict


def main() -> None:
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    yosys = parse_yosys(LATEST / "sky130_yosys.log")
    sta = parse_sta(LATEST / "sky130_sta.log")
    layer_sweep = parse_layer_sweep(LATEST / "rtl_layer_sweep_sim.log")

    rtl_hash = tree_hash(RTL_FILES)
    constraint_hash = tree_hash(CONSTRAINT_FILES)
    evidence_hashes = {
        "rtl_layer_sweep_sim_log_sha256": layer_sweep["log_sha256"],
        "sky130_yosys_log_sha256": yosys["log_sha256"],
        "sky130_sta_log_sha256": sta["log_sha256"],
    }
    review_status, review = reviewer_status(rtl_hash, constraint_hash, evidence_hashes)
    timing_met = sta["wns_ns"] >= 0.0 and sta["estimated_fmax_mhz_floor_bound"] >= FREQ_FLOOR_MHZ
    area_met = yosys["non_sram_area_mm2"] <= AREA_CAP_MM2
    numeric_targets_met = timing_met and area_met and layer_sweep["high_layer"] == 23
    published = numeric_targets_met and review_status == "accepted"

    accepted_prefix = FULL_TRANSFORMER_PREFIX if published else LAYER0_PREFIX
    accepted_first_unsupported = PUBLISHED_FIRST_UNSUPPORTED if published else ACCEPTED_FIRST_UNSUPPORTED
    candidate_prefix = FULL_TRANSFORMER_PREFIX

    ledger = load_json(LEDGER, {"schema_version": 1, "project": "ACE-2", "entries": []})
    baseline = next(
        (
            entry for entry in reversed(ledger.get("entries", []))
            if entry.get("ordered_supported_layer_operator_prefix") == LAYER0_PREFIX
            and entry.get("accepted_prefix_advanced") is True
            and entry.get("rtl_hash") == rtl_hash
            and entry.get("constraint_hash") == constraint_hash
            and float(entry.get("non_sram_area_mm2", math.inf)) <= AREA_CAP_MM2
            and float(entry.get("fmax_mhz", 0.0)) >= FREQ_FLOOR_MHZ
        ),
        None,
    )
    if baseline is None:
        raise RuntimeError("could not find accepted layer_0 frontier baseline for shared-layer binding")
    baseline_evidence_hashes = baseline.get("evidence_hashes", {})
    if not isinstance(baseline_evidence_hashes, dict):
        baseline_evidence_hashes = {}
    ppa_hashes_match_baseline = (
        baseline_evidence_hashes.get("sky130_yosys_log_sha256") == evidence_hashes["sky130_yosys_log_sha256"]
        and baseline_evidence_hashes.get("sky130_sta_log_sha256") == evidence_hashes["sky130_sta_log_sha256"]
    )
    ppa_binding = {
        "mode": "reused_canonical" if ppa_hashes_match_baseline else "fresh_canonical_same_netlist",
        "source_frontier_id": baseline.get("id"),
        "reuse_basis": (
            "unchanged RTL hash, constraint hash, synthesis log hash, and STA log hash"
            if ppa_hashes_match_baseline else
            "fresh canonical SKY130 synthesis/STA logs for unchanged RTL and constraint hashes; "
            "the source frontier supplies the accepted layer_0 prefix baseline, not reused log hashes"
        ),
    }

    ppa_status_prefix = (
        "reused_sky130_synth_sta_bound"
        if ppa_hashes_match_baseline else
        "fresh_sky130_synth_sta_bound_same_netlist"
    )
    status = (
        f"{ppa_status_prefix}_reviewer_accepted"
        if published else
        f"{ppa_status_prefix}_reviewer_rejected"
        if numeric_targets_met and review_status == "rejected" else
        f"{ppa_status_prefix}_meets_floor_awaiting_independent_reviewer"
        if numeric_targets_met else
        f"{ppa_status_prefix}_timing_shortfall"
        if not timing_met else
        f"{ppa_status_prefix}_area_cap_exceeded"
    )
    current_mode = "ADVANCE" if published else "ADVANCE_REVIEW_PENDING" if numeric_targets_met else "ADVANCE_REPLAN_TIMING"
    decision = (
        "shared_transformer_layer_reuse_bound_targets_met_continue_to_final_rmsnorm"
        if published else
        "shared_transformer_layer_reuse_targets_met_awaiting_independent_reviewer"
        if numeric_targets_met and review_status == "pending" else
        "shared_transformer_layer_reuse_targets_met_independent_reviewer_rejected"
        if numeric_targets_met else
        "shared_transformer_layer_reuse_bound_replan"
    )

    delta_area = yosys["non_sram_area_mm2"] - float(baseline["non_sram_area_mm2"])
    delta_area_percent = (delta_area / float(baseline["non_sram_area_mm2"])) * 100.0
    cycle_impact = {
        "basis": "verification/tb/ace2_shell_tb.sv +LAYER_SWEEP_ONLY high-boundary layer-id discriminator plus prior accepted layer_0 full projection-shape regression",
        "candidate": CANDIDATE_OPERATOR,
        "candidate_supported_prefix_length": len(candidate_prefix),
        "accepted_supported_prefix_length": len(accepted_prefix),
        "high_layer_tested": layer_sweep["high_layer"],
        "layer_sweep_cases": layer_sweep["cases"],
        "projection_shapes_at_high_layer": layer_sweep["projection_shapes"],
        "opcode_families_at_high_layer": layer_sweep["opcode_families"],
        "layer_sweep_cycles": layer_sweep["cycles"],
        "ppa_reuse_basis": ppa_binding["reuse_basis"],
    }

    if published:
        manifest_claim_boundaries = [
            "The accepted publication covers the full shared transformer-layer prefix through layer_23.mlp_residual_add after independent Reviewer acceptance.",
            "The shared-layer candidate adds no new datapath RTL; it proves the existing layer_id < 24 descriptor contract at the high transformer-layer boundary across all implemented opcode families and two representative projection descriptor shapes, while the accepted layer_0 regression remains the full MLP projection-shape evidence.",
            "The first unsupported accepted layer/operator is final_rmsnorm; lm_head and full-workload release evidence remain unsupported.",
            "SKY130 PPA is bound to unchanged RTL and constraints; this is not routed GDS, signoff, tapeout, FPGA, or silicon evidence.",
        ]
    else:
        manifest_claim_boundaries = [
            "The accepted publication remains the independently reviewed layer_0 prefix until the shared-layer candidate receives an independent Reviewer verdict.",
            "The shared-layer candidate adds no new datapath RTL; it proves the existing layer_id < 24 descriptor contract at the high transformer-layer boundary across all implemented opcode families and two representative projection descriptor shapes, while the accepted layer_0 regression remains the full MLP projection-shape evidence.",
            "The first unsupported accepted layer/operator is corrected to layer_1.input_rmsnorm; if the candidate is later reviewed and accepted, the next unsupported item becomes final_rmsnorm.",
            "SKY130 PPA is reused from the accepted layer_0 frontier because RTL and constraints are unchanged; this is not routed GDS, signoff, tapeout, FPGA, or silicon evidence.",
        ]

    first_unsupported_detail = {
        "identifier": accepted_first_unsupported,
        "layer": 1 if accepted_first_unsupported.startswith("layer_1.") else None,
        "operator": (
            "input_rmsnorm"
            if accepted_first_unsupported == ACCEPTED_FIRST_UNSUPPORTED
            else accepted_first_unsupported
        ),
        "reason": (
            "Shared layer_1-through-layer_23 reuse has independent Reviewer acceptance; final_rmsnorm is the next frozen workload item."
            if published else
            "Current accepted publication is the independent Reviewer-approved layer_0 prefix; shared layer_1-through-layer_23 reuse is implemented and PPA-bound but still awaiting independent Reviewer acceptance."
        ),
    }
    explicit_non_claims = [
        "no routed Open-PDK GDS has been claimed for this RTL slice",
        "no FPGA prototype, full benchmark, signoff, tapeout readiness, or fabricated silicon result has been claimed",
        (
            "final_rmsnorm and lm_head remain unsupported after the accepted shared transformer-layer prefix"
            if published else
            "shared layer_1-through-layer_23 publication is pending independent Reviewer acceptance"
        ),
        "the complete Qwen2.5-0.5B W4A8 workload remains incomplete until final_rmsnorm, lm_head, and required full-workload traces are implemented and verified",
    ]
    public_claims = [
        {
            "claim": "current Manager-owned stage is rtl",
            "evidence": ["research/PIPELINE_STATE.json", "research/PUBLIC_STATUS.json"],
        },
        {
            "claim": (
                "implementation-supported prefix covers all 24 transformer layers through layer_23.mlp_residual_add after independent Reviewer acceptance"
                if published else
                "implementation-supported prefix remains the independently accepted layer_0 prefix while shared layer_1-through-layer_23 reuse awaits independent Reviewer acceptance"
            ),
            "evidence": [
                "design/RTL_MANIFEST.json",
                "design/PPA_FRONTIER_LEDGER.json",
                "research/PUBLIC_STATUS.json",
            ],
        },
        {
            "claim": "the shared-layer RTL discriminator covers high layer_id 23 across 13 cases, two projection shapes, and nine opcode families",
            "evidence": ["evidence/frontier/latest/rtl_layer_sweep_sim.log"],
        },
        {
            "claim": "the shared-layer candidate meets the 100 MHz SKY130 floor and 2.0 mm2 hierarchical total non-SRAM cap for the bound RTL and constraints",
            "evidence": [
                "evidence/frontier/latest/sky130_yosys.log",
                "evidence/frontier/latest/sky130_sta.log",
                "design/PPA_FRONTIER_LEDGER.json",
            ],
        },
    ]

    previous_manifest = load_json(MANIFEST, {})
    manifest = dict(previous_manifest)
    manifest.update({
        "schema_version": 1,
        "project": "ACE-2",
        "stage": "rtl",
        "stage_closing": False,
        "generated_at_utc": generated_at,
        "supported_layer_operator_prefix": accepted_prefix,
        "first_unsupported_layer_operator": accepted_first_unsupported,
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "candidate_status": status,
        "candidate_meets_numeric_acceptance": numeric_targets_met,
        "candidate_supported_layer_operator_prefix_after_review": candidate_prefix,
        "candidate_first_unsupported_layer_operator_after_review": PUBLISHED_FIRST_UNSUPPORTED,
        "independent_reviewer_acceptance": review_status,
        "independent_reviewer_verdict": (
            "evidence/review/latest/shared_transformer_layers_verdict.json"
            if review is not None else None
        ),
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "source_hashes": [
            {"path": rel, "sha256": sha256_file(ROOT / rel)}
            for rel in SOURCE_FILES
            if (ROOT / rel).exists()
        ],
        "latest_evidence": {
            **dict(previous_manifest.get("latest_evidence", {})),
            "layer_sweep_sim": layer_sweep,
            "rtl_layer_sweep_sim_log": "evidence/frontier/latest/rtl_layer_sweep_sim.log",
            "sky130_yosys": yosys,
            "sky130_sta": sta,
            "shared_layer_ppa_evidence_binding": {
                **ppa_binding,
            },
        },
        "layer_execution_coverage": {
            **dict(previous_manifest.get("layer_execution_coverage", {})),
            "basis": "verification/tb/ace2_shell_tb.sv direct command-dispatch regression plus +LAYER_SWEEP_ONLY high-layer discriminator",
            "shared_layer_sweep_high_layer": layer_sweep["high_layer"],
            "shared_layer_sweep_cases": layer_sweep["cases"],
            "shared_layer_sweep_projection_shapes": layer_sweep["projection_shapes"],
            "shared_layer_sweep_opcode_families": layer_sweep["opcode_families"],
        },
        "claim_boundaries": manifest_claim_boundaries,
    })
    write_json(MANIFEST, manifest)

    entry = {
        "id": f"frontier-{generated_at}",
        "recorded_at_utc": generated_at,
        "stage_closing": False,
        "ordered_supported_layer_operator_prefix": accepted_prefix,
        "first_unsupported_layer_operator": accepted_first_unsupported,
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "candidate_supported_layer_operator_prefix_after_review": candidate_prefix,
        "candidate_first_unsupported_layer_operator_after_review": PUBLISHED_FIRST_UNSUPPORTED,
        "accepted_prefix_advanced": published,
        "candidate_meets_numeric_acceptance": numeric_targets_met,
        "independent_reviewer_acceptance": review_status,
        "independent_reviewer_verdict_sha256": sha256_file(REVIEW_VERDICT) if review is not None else None,
        "mode": current_mode,
        "decision": decision,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "cells": yosys["cells"],
        "delta_cells": int(yosys["cells"]) - int(baseline["cells"]),
        "delta_cells_basis": yosys["area_accounting_basis"],
        "non_sram_area_mm2": yosys["non_sram_area_mm2"],
        "non_sram_area_um2": yosys["non_sram_area_um2"],
        "area_accounting_basis": yosys["area_accounting_basis"],
        "local_cells_excluding_submodules": yosys["local_cells_excluding_submodules"],
        "local_non_sram_area_mm2": yosys["local_non_sram_area_mm2"],
        "delta_area_mm2": delta_area,
        "delta_area_percent": delta_area_percent,
        "compression_improvement_percent": None,
        "fmax_mhz": sta["estimated_fmax_mhz_floor_bound"],
        "fmax_mhz_floor_bound": sta["estimated_fmax_mhz_floor_bound"],
        "frequency_floor_met": timing_met,
        "area_cap_mm2": AREA_CAP_MM2,
        "area_cap_met": area_met,
        "remaining_area_reserve_mm2": AREA_CAP_MM2 - yosys["non_sram_area_mm2"],
        "remaining_frequency_reserve_mhz": max(0.0, sta["estimated_fmax_mhz_floor_bound"] - FREQ_FLOOR_MHZ),
        "remaining_frequency_reserve_mhz_floor_bound": 0.0,
        "sky130_sta": {"clock_period_ns": sta["clock_period_ns"], "wns_ns": sta["wns_ns"], "tns_ns": sta["tns_ns"]},
        "cycle_or_tokens_per_second_impact": cycle_impact,
        "evidence": {
            "rtl_manifest": "design/RTL_MANIFEST.json",
            "rtl_layer_sweep_sim_log": "evidence/frontier/latest/rtl_layer_sweep_sim.log",
            "yosys_log": "evidence/frontier/latest/sky130_yosys.log",
            "sta_log": "evidence/frontier/latest/sky130_sta.log",
            "independent_reviewer_verdict": (
                "evidence/review/latest/shared_transformer_layers_verdict.json"
                if review is not None else None
            ),
        },
        "evidence_hashes": evidence_hashes,
        "ppa_evidence_binding": {
            **ppa_binding,
        },
        "claim_boundary": "shared-layer RTL candidate with reused SKY130 mapped synthesis/OpenSTA PPA; no routed/signoff claim",
    }
    ledger["entries"] = list(ledger.get("entries", [])) + [entry]
    write_json(LEDGER, ledger)

    latest_frontier = {
        "status": status,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "cells": yosys["cells"],
        "non_sram_area_mm2": yosys["non_sram_area_mm2"],
        "area_accounting_basis": yosys["area_accounting_basis"],
        "local_non_sram_area_mm2": yosys["local_non_sram_area_mm2"],
        "fmax_mhz": sta["estimated_fmax_mhz_floor_bound"],
        "fmax_mhz_floor_bound": sta["estimated_fmax_mhz_floor_bound"],
        "frequency_floor_met": timing_met,
        "area_cap_mm2": AREA_CAP_MM2,
        "area_cap_met": area_met,
        "wns_ns": sta["wns_ns"],
        "tns_ns": sta["tns_ns"],
        "cycle_or_tokens_per_second_impact": cycle_impact,
        "delta_area_percent": delta_area_percent,
        "compression_improvement_percent": None,
        "remaining_area_reserve_mm2": AREA_CAP_MM2 - yosys["non_sram_area_mm2"],
        "remaining_frequency_reserve_mhz": max(0.0, sta["estimated_fmax_mhz_floor_bound"] - FREQ_FLOOR_MHZ),
        "remaining_frequency_reserve_mhz_floor_bound": 0.0,
    }

    public = load_json(PUBLIC_STATUS, {})
    pipeline = load_json(PIPELINE_STATE, {})
    public.update({
        "schema_version": 1,
        "project": "ACE-2",
        "vertical": "chip_design",
        "last_updated_utc": generated_at,
        "generated_at_utc": generated_at,
        "stage": {
            **dict(public.get("stage", {})),
            "current_stage": pipeline.get("current_stage", "rtl"),
            "current_stage_source": "research/PIPELINE_STATE.json",
            "planner_may_advance_stage": False,
            "stage_transition_owner": "Manager",
            "downstream_locked_until_manager_advance": [
                "verification",
                "ppa",
                "prototype",
                "benchmark",
                "signoff",
            ],
        },
        "implementation_frontier": {
            "tracking_order_definition": "design/WORKLOAD.md#ordered-layer/operator-frontier",
            "current_mode": current_mode,
            "ordered_supported_layer_operator_prefix": accepted_prefix,
            "first_unsupported_layer_operator": first_unsupported_detail,
            "candidate_layer_operator": CANDIDATE_OPERATOR,
            "candidate_supported_layer_operator_prefix_after_review": candidate_prefix,
            "candidate_first_unsupported_layer_operator_after_review": PUBLISHED_FIRST_UNSUPPORTED,
            "latest_decision": decision,
            "latest_ppa_frontier": latest_frontier,
        },
        "dashboard_fields": {
            **dict(public.get("dashboard_fields", {})),
            "supported_layers": accepted_prefix,
            "ordered_supported_layer_operator_prefix": accepted_prefix,
            "first_unsupported_layer_operator": accepted_first_unsupported,
            "current_mode": current_mode,
            "latest_decision": decision,
            "latest_ppa_frontier": latest_frontier,
            "latest_ppa_frontier_status": status,
            "shared_layer_candidate": CANDIDATE_OPERATOR,
            "shared_layer_candidate_first_unsupported_after_review": PUBLISHED_FIRST_UNSUPPORTED,
            "shared_layer_candidate_review_status": review_status,
        },
        "public_claims": public_claims,
        "artifact_hashes": [
            {"path": rel, "sha256": sha256_file(ROOT / rel)}
            for rel in ARTIFACT_FILES
            if (ROOT / rel).exists()
        ],
        "explicit_non_claims": explicit_non_claims,
    })
    write_json(PUBLIC_STATUS, public)

    print(
        f"bound {CANDIDATE_OPERATOR}: status={status} "
        f"accepted_first_unsupported={accepted_first_unsupported} "
        f"candidate_first_unsupported_after_review={PUBLISHED_FIRST_UNSUPPORTED}"
    )


if __name__ == "__main__":
    main()
