#!/usr/bin/env python3
"""Atomically publish the accepted RMSNorm-repaired SKY130 100 MHz NO_GO.

This tool is metadata-only.  It validates sealed evidence and updates the
authoritative publication surfaces; it never invokes simulation, synthesis,
STA, PPA, model, runtime, benchmark, or image commands.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

LEDGER = ROOT / "design/PPA_FRONTIER_LEDGER.json"
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
MANIFEST_COMPANION = ROOT / "design/RTL_MANIFEST.sha256"
TRACEABILITY = ROOT / "design/RTL_TRACEABILITY.md"
TRACEABILITY_COMPANION = ROOT / "design/RTL_TRACEABILITY.sha256"
PUBLIC_STATUS = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"
PIPELINE_COMPANION = ROOT / "research/PIPELINE_STATE.sha256"

PPA_DIR_REL = "evidence/canonical_sky130/rtl-rmsnorm-repaired-tree-canonical-sky130-ppa-v1"
PPA_DIR = ROOT / PPA_DIR_REL
REPAIR_DIR_REL = "evidence/verification/rtl-rmsnorm-critical-cone-repair-7483-to-7667-v1"
REPAIR_DIR = ROOT / REPAIR_DIR_REL

HANDOFF_ROOT = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs")
REPAIR_REVIEW = HANDOFF_ROOT / "rtl-rmsnorm-critical-cone-repair-7483-to-7667-v1/round-0002.json"
PPA_REVIEW = HANDOFF_ROOT / "rtl-rmsnorm-repaired-tree-canonical-sky130-ppa-v1/round-0001.json"

PUBLICATION_AT = "2026-08-04T06:00:28Z"
PUBLICATION_ID = f"frontier-publication-{PUBLICATION_AT}"
PUBLICATION_MISSION = "manager-publish-a42-rmsnorm-repaired-ppa-no-go-v1"

PREVIOUS_PUBLICATION_ID = "frontier-publication-2026-08-04T04:43:19Z"
PREVIOUS_TREE = "e623db3663ef3854616ce365c51aaef36e8915e0a60a7bb936c4d6c34c0b8fa0"
PREVIOUS_RMSNORM = "7f1ee70b194fe560cc654a6836615d65344925a13bb6f1308f9dfd59879c9afc"
TREE = "a42bc8469b929a1cc193fefb45ff630457db9d298b0f611571992a79bacb360a"
RMSNORM = "18177d9fc5eefaf550e129fd13aa908d16f2b2d24a8925ed97e81cf4cf38d5b0"
SHELL = "8bf63ef4bb98700e2f3b86ae4faf1883da76d3875e8632f086453cc51d15f091"
CONSTRAINT = "b21f8da39a0bdc5bf4db70b82013efb3d06a686347749c0070f37f37eca9e9ea"

REPAIR_MISSION = "rtl-rmsnorm-critical-cone-repair-7483-to-7667-v1"
PPA_MISSION = "rtl-rmsnorm-repaired-tree-canonical-sky130-ppa-v1"
SUCCESSOR = "rtl-rmsnorm-final-sumsq-carry-cone-repair-v1"

CELLS = 62330
AREA = 0.61393256
AREA_RESERVE = 1.38606744
SLACK = -0.1741
WNS = -0.17
TNS = -0.29

CLAIM = (
    "Functional Layer-0 18/18 and full-Qwen 13,914/13,914 two-token evidence "
    "remains accepted on Fresh-Reviewer-accepted RTL tree a42bc846...b360a, "
    "with generated tokens [0, 0]. Its sole canonical SKY130 gate is an "
    "explicit 100 MHz NO_GO with -0.1741 ns detailed setup slack, -0.17 ns "
    "WNS, and -0.29 ns TNS; final product timing is not certified."
)
PPA_CLAIM = (
    "Accepted canonical SKY130 mapped synthesis/OpenSTA evidence only. The "
    "0.61393256 mm2 non-SRAM area passes the 2.0 mm2 cap, but timing is an "
    "explicit 100 MHz NO_GO and final timing is not certified; no routed, "
    "power, DRC/LVS, signoff, GDS, pre-tapeout-readiness, tapeout, or silicon claim."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    try:
        display = path.relative_to(ROOT)
    except ValueError:
        display = path
    require(isinstance(value, dict), f"{display} is not a JSON object")
    return value


def encode_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, logical_path: str | None = None) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": logical_path or path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def handoff_artifact(path: Path, logical_path: str, status: str = "done") -> dict[str, Any]:
    record = artifact(path, logical_path)
    record.update({"producer_role": "reviewer", "status": status})
    return record


def canonical_public_hash(value: dict[str, Any]) -> str:
    candidate = copy.deepcopy(value)
    candidate.setdefault("integrity", {})["canonical_sha256"] = None
    return sha256_bytes(encode_json(candidate))


def verify_sha256_manifest(directory: Path, manifest_name: str = "SHA256SUMS") -> None:
    manifest = directory / manifest_name
    require(manifest.is_file(), f"missing aggregate manifest {manifest}")
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        digest, relative = raw.split(None, 1)
        relative = relative.strip()
        if relative.startswith("*"):
            relative = relative[1:]
        if relative.startswith("./"):
            relative = relative[2:]
        target = ROOT / relative if relative.startswith("evidence/") else directory / relative
        require(target.is_file(), f"{manifest}:{number} missing {relative}")
        require(sha256_file(target) == digest, f"{manifest}:{number} hash mismatch for {relative}")


def verify_live_tree() -> None:
    tree_manifest = REPAIR_DIR / "rtl_tree_manifest.sha256"
    require(sha256_file(tree_manifest) == TREE, "sealed repaired RTL-tree manifest digest differs")
    for number, raw in enumerate(tree_manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        digest, relative = raw.split(None, 1)
        target = ROOT / relative.strip()
        require(target.is_file(), f"live RTL path missing at tree line {number}: {relative}")
        require(sha256_file(target) == digest, f"live RTL hash mismatch at tree line {number}: {relative}")
    require(sha256_file(ROOT / "rtl/ace2_rmsnorm_core.sv") == RMSNORM, "live RMSNorm source hash differs")
    require(sha256_file(ROOT / "rtl/ace2_shell.sv") == SHELL, "live shell source hash differs")
    require(sha256_file(ROOT / "constraints/ace2_rmsnorm_core.sdc") == CONSTRAINT, "RMSNorm constraint hash differs")


def verify_sealed_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    verify_sha256_manifest(REPAIR_DIR)
    verify_sha256_manifest(PPA_DIR)
    verify_live_tree()

    aggregate_companion = (PPA_DIR / "SHA256SUMS.sha256").read_text(encoding="utf-8").split()
    require(aggregate_companion, "canonical PPA aggregate companion is empty")
    require(aggregate_companion[0] == sha256_file(PPA_DIR / "SHA256SUMS"), "canonical PPA aggregate companion differs")

    result = load_json(PPA_DIR / "RESULT.json")
    metrics = load_json(PPA_DIR / "METRICS.json")
    independent = load_json(PPA_DIR / "INDEPENDENT_METRICS.json")
    marker = load_json(PPA_DIR / "authorization_consumed/CONSUMPTION_MARKER.json")
    repair_witness = load_json(REPAIR_DIR / "repair_witness.json")
    repair_review = load_json(REPAIR_REVIEW)
    ppa_review = load_json(PPA_REVIEW)

    require(result.get("make_exit_status") == 0, "canonical PPA make did not exit zero")
    require(result.get("protected_state") == "MATCH", "canonical PPA protected-state result differs")
    require(result.get("explicit_100mhz_gate") == "NO_GO", "canonical PPA gate is not NO_GO")
    require(marker.get("exactly_once_consumed") is True, "canonical PPA authorization was not consumed exactly once")
    require(marker.get("locked_rtl_tree_sha256") == TREE, "canonical PPA marker tree differs")
    require(marker.get("locked_ace2_rmsnorm_core_sha256") == RMSNORM, "canonical PPA marker RMSNorm hash differs")
    require(repair_witness.get("new_tree", {}).get("rtl_tree_sha256") == TREE, "repair witness tree differs")
    require(repair_witness.get("repair", {}).get("file_sha256") == RMSNORM, "repair witness source differs")
    require(repair_review.get("producer_role") == "reviewer", "repair handoff producer is not Reviewer")
    require(repair_review.get("review", {}).get("status") == "done", "repair Reviewer did not accept")
    require(ppa_review.get("producer_role") == "reviewer", "PPA handoff producer is not Reviewer")
    require(ppa_review.get("review", {}).get("status") == "done", "PPA Reviewer did not accept")

    expected = {
        "top_level_cell_count": CELLS,
        "non_sram_area_mm2": "0.61393256",
        "worst_setup_slack_ns": "-0.1741",
        "wns_ns": "-0.17",
        "tns_ns": "-0.29",
        "timing_100mhz_status": "NO_GO",
        "area_cap_status": "PASS",
    }
    for key, value in expected.items():
        require(metrics.get(key) == value, f"METRICS.json {key} differs")
        require(independent.get(key) == value, f"INDEPENDENT_METRICS.json {key} differs")
    require(independent.get("reconciliation", {}).get("all_required_metrics_match") is True,
            "independent PPA reconciliation is not complete")
    require(metrics.get("cell_delta_vs_baseline") == 3, "cell delta differs")
    require(metrics.get("area_delta_vs_baseline_mm2") == "0.0000963424", "area delta differs")
    require(metrics.get("slack_delta_vs_baseline_ns") == "0.3534", "slack delta differs")
    require(metrics.get("wns_delta_vs_baseline_ns") == "0.36", "WNS delta differs")
    require(metrics.get("tns_delta_vs_baseline_ns") == "67.39", "TNS delta differs")
    critical = metrics.get("critical_path", {})
    require(critical.get("startpoint") == "u_rmsnorm_core/_7721_", "critical startpoint differs")
    require(critical.get("endpoint") == "u_rmsnorm_core/_7482_", "critical endpoint differs")
    require((PPA_DIR / "postflight/protected_state_result.txt").read_text(encoding="utf-8").strip() == "MATCH",
            "protected-state marker is not MATCH")
    require((PPA_DIR / "preflight/protected_state_pre.sha256").read_bytes()
            == (PPA_DIR / "postflight/protected_state_post.sha256").read_bytes(),
            "protected publication state changed during canonical PPA")
    return metrics, repair_witness


def reviewer_binding(path: Path, logical: str) -> dict[str, Any]:
    record = handoff_artifact(path, logical)
    return record


def build_repair_binding(repair_witness: dict[str, Any]) -> dict[str, Any]:
    return {
        "aggregate_manifest": artifact(REPAIR_DIR / "SHA256SUMS"),
        "fresh_reviewer_binding": reviewer_binding(
            REPAIR_REVIEW, "handoff:rtl-rmsnorm-critical-cone-repair-7483-to-7667-v1/round-0002.json"
        ),
        "mission_id": REPAIR_MISSION,
        "repair_witness": artifact(REPAIR_DIR / "repair_witness.json"),
        "rtl_tree_sha256": TREE,
        "rmsnorm_rtl_sha256": RMSNORM,
        "scope": "remove_only_the_redundant_state_predicate_from_the_rmsnorm_collect_shift_enable",
        "source_to_netlist_witness": artifact(REPAIR_DIR / "source_to_netlist_witness.md"),
        "verification": repair_witness["verification"],
    }


def build_canonical_ppa(repair: dict[str, Any]) -> dict[str, Any]:
    return {
        "aggregate_manifest": artifact(PPA_DIR / "SHA256SUMS"),
        "area_cap_status": "PASS",
        "canonical_gate_verdict": "NO_GO",
        "claim_boundary": PPA_CLAIM,
        "clock_period_ns": 10.0,
        "constraint_sha256": CONSTRAINT,
        "corner": "SKY130 HD TT 25C / 1.80V",
        "critical_path": {
            "causal_final": "next_sumsq_w + HIDDEN_HALF_ACC carry cone",
            "data_arrival_ns": 9.9426,
            "data_required_ns": 9.7685,
            "endpoint": "u_rmsnorm_core/_7482_",
            "path_group": "clk_i",
            "path_type": "max",
            "reported_slack_ns": SLACK,
            "source_mapping": "collect_square_q[0] -> div_dividend_q[47]",
            "startpoint": "u_rmsnorm_core/_7721_",
        },
        "delta_vs_prior_tree": {
            "area_delta_mm2": 0.0000963424,
            "baseline_mission_id": "rtl-critical-cone-repaired-tree-canonical-sky130-ppa-v1",
            "baseline_rtl_tree_sha256": PREVIOUS_TREE,
            "cell_delta": 3,
            "tns_delta_ns": 67.39,
            "wns_delta_ns": 0.36,
            "worst_setup_slack_delta_ns": 0.3534,
        },
        "evidence_directory": f"{PPA_DIR_REL}/",
        "final_metrics": artifact(PPA_DIR / "METRICS.json"),
        "frequency_target_mhz": 100.0,
        "fresh_reviewer_binding": reviewer_binding(
            PPA_REVIEW, "handoff:rtl-rmsnorm-repaired-tree-canonical-sky130-ppa-v1/round-0001.json"
        ),
        "independent_metrics": artifact(PPA_DIR / "INDEPENDENT_METRICS.json"),
        "mission_id": PPA_MISSION,
        "non_sram_area_cap_mm2": 2.0,
        "non_sram_area_mm2": AREA,
        "product_timing_certified": False,
        "remaining_area_reserve_mm2": AREA_RESERVE,
        "result": artifact(PPA_DIR / "RESULT.json"),
        "rmsnorm_rtl_sha256": RMSNORM,
        "rtl_tree_sha256": TREE,
        "source_repair": repair,
        "timing_100mhz_status": "NO_GO",
        "timing_interpretation": (
            "The accepted enable-cone repair substantially improved timing but did not pass 100 MHz; "
            "the newly reported local cause is the final next_sumsq_w + HIDDEN_HALF_ACC carry cone."
        ),
        "tns_ns": TNS,
        "top_level_cell_count": CELLS,
        "wns_ns": WNS,
        "worst_setup_slack_ns": SLACK,
    }


def historical_terminal_predecessor(public: dict[str, Any]) -> dict[str, Any]:
    historical = (
        public.get("historical_dashboard_fields", {})
        .get("fields", {})
        .get("terminal_baseline_no_go")
    )
    require(isinstance(historical, dict), "PUBLIC_STATUS historical terminal predecessor is missing")
    require(
        historical.get("contract_id") == "cross_layer_quantization_error_carry_final_output_v1",
        "PUBLIC_STATUS historical terminal predecessor contract differs",
    )
    require(
        historical.get("status") == "sealed_terminal_baseline_protocol_no_go",
        "PUBLIC_STATUS historical terminal predecessor status differs",
    )
    require(
        historical.get("updated_at_utc") == "2026-08-02T05:58:17Z",
        "PUBLIC_STATUS historical terminal predecessor timestamp differs",
    )
    return historical


def build_terminal_baseline_no_go(
    ppa: dict[str, Any], full: dict[str, Any], next_action: dict[str, Any]
) -> dict[str, Any]:
    terminal = copy.deepcopy(ppa)
    terminal.update({
        "final_product_timing_certified": False,
        "functional_integration": full,
        "next_action_queue": [next_action],
        "publication_id": PUBLICATION_ID,
        "publication_mission_id": PUBLICATION_MISSION,
        "publication_review_status": "PENDING_FRESH_REVIEWER",
        "required_manager_action": (
            "obtain_fresh_publication_review_then_release_only_the_single_final_sumsq_carry_cone_repair"
        ),
        "status": "ACCEPTED_CANONICAL_100MHZ_NO_GO_PENDING_PUBLICATION_REVIEW",
    })
    return terminal


def update_current_public_projections(
    public: dict[str, Any],
    ppa: dict[str, Any],
    full: dict[str, Any],
    next_action: dict[str, Any],
    current_publication: dict[str, Any],
) -> None:
    # The predecessor remains immutable in the historical dashboard snapshot.
    historical_terminal_predecessor(public)
    dashboard = copy.deepcopy(public["dashboard_fields"])
    dashboard.update({
        "canonical_ppa": ppa,
        "canonical_sky130_ppa": ppa,
        "current_publication": current_publication,
        "final_product_timing_certified": False,
        "functional_integration": full,
        "next_action_queue": [next_action],
        "publication_id": PUBLICATION_ID,
    })
    public["dashboard_fields"] = dashboard
    public["terminal_baseline_no_go"] = build_terminal_baseline_no_go(ppa, full, next_action)


def build_successor(ppa: dict[str, Any], full: dict[str, Any]) -> dict[str, Any]:
    task = {
        "accepted_baseline": {
            "causal_final": "next_sumsq_w + HIDDEN_HALF_ACC carry cone",
            "critical_path_endpoint": "u_rmsnorm_core/_7482_",
            "critical_path_source_mapping": "collect_square_q[0] -> div_dividend_q[47]",
            "critical_path_startpoint": "u_rmsnorm_core/_7721_",
            "frequency_target_mhz": 100.0,
            "rtl_tree_sha256": TREE,
            "tns_ns": TNS,
            "wns_ns": WNS,
            "worst_setup_slack_ns": SLACK,
        },
        "allowed_work": [
            "rtl_repair_limited_to_the_final_next_sumsq_plus_hidden_half_acc_carry_cone",
            "sequential_equivalence_against_the_accepted_a42_tree",
            "directed_rmsnorm_and_final_shell_simulation_for_the_repaired_cone",
        ],
        "canonical_ppa_authorized": False,
        "execution_authorized": False,
        "id": SUCCESSOR,
        "materialization_count": 1,
        "preserved_functional_contract": {
            "full_runtime_commands_passed": 13914,
            "full_runtime_commands_total": 13914,
            "generated_tokens": [0, 0],
            "image_sha256": full["runtime"]["image_sha256"],
            "layer0_commands_passed": 18,
            "layer0_commands_total": 18,
            "raw_model_sha256": full["runtime"]["raw_model_sha256"],
            "schedule_sha256": full["runtime"]["schedule_sha256"],
        },
        "prohibited_work": [
            "ppa_yosys_opensta_openroad_or_any_synthesis_sta_inside_this_repair_task",
            "broad_functional_runtime_model_or_image_replay",
            "target_or_constraint_relaxation",
            "schedule_image_model_or_runtime_contract_change",
            "broad_unrelated_rtl_refactor",
            "duplicate_successor_creation",
        ],
        "release_condition": f"fresh_reviewer_accepts_{PUBLICATION_MISSION}",
        "scope": "repair_only_the_final_rmsnorm_sumsq_rounding_carry_cone_feeding_div_dividend_q_47",
        "stage": "rtl",
        "status": "materialized_once_dependency_gated_pending_fresh_reviewer_acceptance_of_publication",
    }
    return {
        "accepted_ppa_evidence": ppa,
        "accepted_rtl_tree_sha256": TREE,
        "candidate_capability_accepted": False,
        "canonical_ppa_authorized": False,
        "contract_id": SUCCESSOR,
        "current_stage": "rtl",
        "final_product_timing_certified": False,
        "full_qwen_functional_integration_accepted": True,
        "id": SUCCESSOR,
        "implementation_authorized": False,
        "materialization_count": 1,
        "mode": "TIMING_REPAIR",
        "next_action_queue": [task],
        "status": task["status"],
        "targets": {
            "abstract_streaming_memory_boundary_bits": 128,
            "frequency_floor_mhz": 100.0,
            "non_sram_area_cap_mm2": 2.0,
        },
    }


def update_manifest_source(manifest: dict[str, Any]) -> None:
    found = False
    live = ROOT / "rtl/ace2_rmsnorm_core.sv"
    for record in manifest.get("source_hashes", []):
        if record.get("path") == "rtl/ace2_rmsnorm_core.sv":
            record.update({"bytes": live.stat().st_size, "sha256": RMSNORM})
            found = True
    require(found, "manifest RMSNorm source record is missing")
    historical = manifest.setdefault("historical_source_hashes", [])
    if not any(item.get("path") == "rtl/ace2_rmsnorm_core.sv" and item.get("sha256") == PREVIOUS_RMSNORM
               for item in historical):
        historical.append({
            "bytes": 23291,
            "path": "rtl/ace2_rmsnorm_core.sv",
            "sha256": PREVIOUS_RMSNORM,
            "superseded_by_rtl_tree_sha256": TREE,
        })


def build_payloads() -> dict[Path, bytes]:
    metrics, repair_witness = verify_sealed_inputs()
    ledger = load_json(LEDGER)
    manifest = load_json(MANIFEST)
    public = load_json(PUBLIC_STATUS)
    pipeline = load_json(PIPELINE_STATE)
    trace = TRACEABILITY.read_text(encoding="utf-8")

    require(ledger.get("entries", [])[-1].get("rtl_hash") == PREVIOUS_TREE, "ledger predecessor differs")
    require(manifest.get("rtl_hash") == PREVIOUS_TREE, "manifest predecessor differs")
    require(public.get("candidate_rtl_hash") == PREVIOUS_TREE, "PUBLIC_STATUS predecessor differs")
    require(pipeline.get("current_publication", {}).get("rtl_tree_sha256") == PREVIOUS_TREE,
            "PIPELINE_STATE predecessor publication differs")
    require(not any(item.get("id") == PUBLICATION_ID for item in ledger["entries"]), "publication already exists")

    repair = build_repair_binding(repair_witness)
    ppa = build_canonical_ppa(repair)

    full = copy.deepcopy(manifest["full_qwen_integration"])
    full["rtl_tree_sha256"] = TREE
    full["canonical_ppa"] = ppa
    full["claim_boundary"] = CLAIM
    full["final_product_timing_certified"] = False
    full["publication_review_status"] = "PENDING_FRESH_REVIEWER"
    full["repair_binding"] = repair
    full.setdefault("accepted_post_runtime_rtl_repairs", []).append({
        "changed_rtl_paths": ["rtl/ace2_rmsnorm_core.sv"],
        "fresh_reviewer_binding": repair["fresh_reviewer_binding"],
        "from_rtl_tree_sha256": PREVIOUS_TREE,
        "scope": "semantics_preserving_rmsnorm_collect_enable_cone_repair",
        "to_rtl_tree_sha256": TREE,
    })
    full["layer0"]["publication_rtl_tree_sha256"] = TREE
    full["layer0"]["report_review_status_resolution"] = (
        "sealed_fresh_reviewer_binding_plus_accepted_shell_only_rtl_delta_chain_"
        "plus_accepted_rmsnorm_semantics_preserving_repair"
    )

    successor = build_successor(ppa, full)
    next_action = successor["next_action_queue"][0]

    entry = copy.deepcopy(ledger["entries"][-1])
    entry.update({
        "area_cap_status": "PASS",
        "canonical_gate_verdict": "NO_GO",
        "cells": CELLS,
        "claim_boundary": f"{CLAIM} {PPA_CLAIM}",
        "comparison_limitation": (
            "This publication performed no engineering execution and did not mutate RTL, constraints, "
            "runtime, model, image, benchmark, repair, or PPA evidence."
        ),
        "decision": "RMSNORM_REPAIRED_TREE_100MHZ_NO_GO_ENQUEUE_SINGLE_FINAL_SUMSQ_CARRY_CONE_REPAIR",
        "delta_vs_prior_tree": ppa["delta_vs_prior_tree"],
        "evidence": copy.deepcopy(entry["evidence"]),
        "functional_authority": {
            "final_product_timing_certified": False,
            "full_runtime_commands_passed": 13914,
            "full_runtime_commands_total": 13914,
            "generated_tokens": [0, 0],
            "image_sha256": full["runtime"]["image_sha256"],
            "layer0_commands_passed": 18,
            "layer0_commands_total": 18,
            "raw_model_sha256": full["runtime"]["raw_model_sha256"],
            "schedule_sha256": full["runtime"]["schedule_sha256"],
            "status": "FUNCTIONAL_FULL_QWEN_TWO_TOKEN_INTEGRATION_ACCEPTED",
        },
        "id": PUBLICATION_ID,
        "mode": "TIMING_REPAIR",
        "next_action": next_action,
        "non_sram_area_mm2": AREA,
        "non_sram_area_um2": AREA * 1_000_000,
        "publication_review_status": "PENDING_FRESH_REVIEWER",
        "recorded_at_utc": PUBLICATION_AT,
        "remaining_area_reserve_mm2": AREA_RESERVE,
        "repair_binding": repair,
        "repaired_rmsnorm_sha256": RMSNORM,
        "repaired_shell_sha256": SHELL,
        "rtl_file_count": 23,
        "rtl_hash": TREE,
        "rtl_hash_method": "sha256 of the UTF-8 path-sorted per-file SHA256 manifest for rtl/**/*.sv",
        "sky130_sta": {
            "clock_period_ns": 10.0,
            "critical_path": ppa["critical_path"],
            "tns_ns": TNS,
            "wns_ns": WNS,
            "worst_setup_slack_ns": SLACK,
        },
        "timing_100mhz_status": "NO_GO",
    })
    entry["evidence"].update({
        "canonical_ppa_aggregate_manifest": ppa["aggregate_manifest"],
        "canonical_ppa_evidence_directory": ppa["evidence_directory"],
        "canonical_ppa_final_metrics": ppa["final_metrics"],
        "canonical_ppa_fresh_review": ppa["fresh_reviewer_binding"],
        "canonical_ppa_independent_metrics": ppa["independent_metrics"],
        "canonical_ppa_result": ppa["result"],
        "source_repair": repair,
    })
    ledger["entries"].append(entry)

    manifest.update({
        "candidate_capability_accepted": True,
        "candidate_id": "rmsnorm_repaired_tree_a42bc846",
        "candidate_meets_numeric_acceptance": True,
        "candidate_requires_fresh_sky130_ppa": False,
        "candidate_rtl_hash": TREE,
        "candidate_status": "FUNCTIONAL_ACCEPTED_CANONICAL_100MHZ_NO_GO_PENDING_PUBLICATION_REVIEW",
        "candidate_verification_complete": True,
        "canonical_ppa": ppa,
        "claim_boundaries": [
            CLAIM,
            "The 0.61393256 mm2 non-SRAM area result passes the 2.0 mm2 cap; the 100 MHz gate is NO_GO.",
            "The critical path is u_rmsnorm_core/_7721_ -> _7482_, mapped from collect_square_q[0] to div_dividend_q[47].",
            "The causal final cone is next_sumsq_w + HIDDEN_HALF_ACC; no timing-certified claim is made.",
            "No RTL, evidence, PPA, simulation, runtime, model, image, or benchmark flow ran in this publication.",
            "No routed, power, DRC/LVS, signoff, GDS, pre-tapeout-readiness, tapeout, or silicon claim is made.",
        ],
        "full_qwen_integration": full,
        "generated_at_utc": PUBLICATION_AT,
        "next_action_queue": [next_action],
        "next_product_boundary": "repair_only_the_final_next_sumsq_w_plus_HIDDEN_HALF_ACC_carry_cone",
        "ppa_evidence_binding": {
            "canonical_ppa": ppa,
            "claim_boundary": CLAIM,
            "functional_full_qwen_integration": full,
            "publication_id": PUBLICATION_ID,
            "publication_mission_id": PUBLICATION_MISSION,
        },
        "publication_review_status": "PENDING_FRESH_REVIEWER",
        "required_manager_action": "obtain_fresh_publication_review_then_release_only_the_single_final_sumsq_carry_cone_repair",
        "rtl_hash": TREE,
        "status": "FUNCTIONAL_FULL_QWEN_ACCEPTED_RMSNORM_REPAIRED_TREE_100MHZ_NO_GO_PENDING_PUBLICATION_REVIEW",
    })
    update_manifest_source(manifest)
    manifest["traceability"].update({
        "final_product_timing_certified": False,
        "next_product_boundary": "repair_only_the_final_next_sumsq_w_plus_HIDDEN_HALF_ACC_carry_cone",
        "selected_mechanism": "rmsnorm_repaired_tree_a42bc846",
        "timing_contract_gap": "100mhz_worst_setup_slack_minus_0p1741ns",
    })
    manifest["latest_evidence"].update({
        "repaired_tree_canonical_ppa_aggregate_manifest": ppa["aggregate_manifest"],
        "repaired_tree_canonical_ppa_final_metrics": ppa["final_metrics"],
        "repaired_tree_canonical_ppa_fresh_review": ppa["fresh_reviewer_binding"],
        "repaired_tree_canonical_ppa_independent_metrics": ppa["independent_metrics"],
        "repaired_tree_canonical_ppa_result": ppa["result"],
        "rmsnorm_repair_aggregate_manifest": repair["aggregate_manifest"],
        "rmsnorm_repair_fresh_review": repair["fresh_reviewer_binding"],
        "rmsnorm_repair_witness": repair["repair_witness"],
    })
    manifest["frontier_publication_history"].append({
        "canonical_ppa_mission_id": PPA_MISSION,
        "functional_status": "FUNCTIONAL_FULL_QWEN_TWO_TOKEN_INTEGRATION_ACCEPTED",
        "non_sram_area_mm2": AREA,
        "publication_id": PUBLICATION_ID,
        "rtl_tree_sha256": TREE,
        "successor_task_id": SUCCESSOR,
        "timing_100mhz_status": "NO_GO",
        "tns_ns": TNS,
        "top_level_cell_count": CELLS,
        "wns_ns": WNS,
        "worst_setup_slack_ns": SLACK,
    })

    trace_section = f"""

## RMSNorm-repaired tree canonical timing NO_GO publication

This append-only publication supersedes current pointers for publication
`{PREVIOUS_PUBLICATION_ID}` without deleting or rewriting it. The accepted
23-file RTL tree is SHA-256 `{TREE}` and `rtl/ace2_rmsnorm_core.sv` is SHA-256
`{RMSNORM}`. The accepted bounded repair is `{REPAIR_MISSION}`, bound by its
repair witness (SHA-256 `{repair['repair_witness']['sha256']}`), aggregate
`SHA256SUMS` (SHA-256 `{repair['aggregate_manifest']['sha256']}`), and Fresh
Reviewer decision (SHA-256 `{repair['fresh_reviewer_binding']['sha256']}`).

Functional authority is preserved: Layer 0 passes 18 of 18 ordered operators,
and the full-Qwen two-token runtime passes 13,914 of 13,914 commands with tokens
`[0, 0]`. The accepted RMSNorm repair is semantics-preserving and changed only
`rtl/ace2_rmsnorm_core.sv`; this publication performed no functional replay.

The sole canonical PPA packet is `{PPA_DIR_REL}/`. It reports {CELLS:,} cells,
{AREA:.8f} mm2 non-SRAM area PASS, {SLACK:.4f} ns detailed setup slack, WNS
{WNS:.2f} ns, and TNS {TNS:.2f} ns. The explicit 100 MHz verdict is `NO_GO`,
so final product timing is not certified. Relative to the e623 tree, the deltas
are +3 cells, +0.0000963424 mm2, +0.3534 ns detailed slack, +0.36 ns WNS, and
+67.39 ns TNS.

The critical path is `u_rmsnorm_core/_7721_ -> _7482_`, source-mapped as
`collect_square_q[0] -> div_dividend_q[47]`. The causal final logic is the
`next_sumsq_w + HIDDEN_HALF_ACC` carry cone. The canonical `RESULT.json`,
`METRICS.json`, `INDEPENDENT_METRICS.json`, and aggregate `SHA256SUMS` hash to
`{ppa['result']['sha256']}`, `{ppa['final_metrics']['sha256']}`,
`{ppa['independent_metrics']['sha256']}`, and `{ppa['aggregate_manifest']['sha256']}`.
Fresh Reviewer acceptance of the exactly-once PPA packet hashes to
`{ppa['fresh_reviewer_binding']['sha256']}`.

This publication ran no RTL, simulation, formal, synthesis, STA, PPA, runtime,
model, image, benchmark, prototype, or signoff flow and changed no engineering
evidence. Its own cross-file publication review remains pending. Exactly one
dependency-gated successor, `{SUCCESSOR}`, is materialized; it may be released
only after Fresh Reviewer acceptance of this publication. PPA/Yosys/OpenSTA/
OpenROAD and broad functional replay are forbidden inside that repair.
"""
    trace = trace.rstrip() + trace_section

    old_publication = pipeline["current_publication"]
    current_publication = {
        "canonical_ppa": ppa,
        "claim_boundary": CLAIM,
        "final_product_timing_certified": False,
        "full_qwen_integration": full,
        "id": PUBLICATION_ID,
        "next_action_queue": [next_action],
        "performed_without_engineering_execution": True,
        "publication_mission_id": PUBLICATION_MISSION,
        "publication_review_status": "PENDING_FRESH_REVIEWER",
        "repair_binding": repair,
        "rtl_tree_sha256": TREE,
        "status": "pending_fresh_reviewer",
        "supersedes_publication_id": PREVIOUS_PUBLICATION_ID,
    }
    old_successor = copy.deepcopy(pipeline["successor"])
    old_successor.update({
        "completed_at_utc": PUBLICATION_AT,
        "completed_rtl_tree_sha256": TREE,
        "status": "completed_and_superseded_by_final_sumsq_carry_cone_repair",
        "superseded_by": SUCCESSOR,
    })
    pipeline.setdefault("historical_successors", []).append(old_successor)
    pipeline["current_publication"] = current_publication
    pipeline["current_stage"] = "rtl"
    pipeline["stages"]["ppa"]["status"] = "done_no_go_rmsnorm_repaired_tree_100mhz"
    pipeline["stages"]["rtl"]["status"] = "in_progress"
    pipeline["successor"] = successor
    pipeline["workflow_mode"] = "TIMING_REPAIR"
    pipeline["stage_history"].append({
        "at": PUBLICATION_AT,
        "by": "manager",
        "direction": "publication",
        "from_stage": "rtl",
        "publication_id": PUBLICATION_ID,
        "reason": (
            "Publish the Fresh-Reviewer-accepted RMSNorm-repaired RTL tree and its sole canonical "
            "SKY130 100 MHz NO_GO without engineering execution; preserve accepted Layer-0/full-Qwen "
            "functional evidence and dependency-gate exactly one final sumsq carry-cone repair."
        ),
        "to_stage": "rtl",
    })

    previous_public_history = {
        "canonical_sky130_ppa": {
            "aggregate_sha256": old_publication["canonical_ppa"]["aggregate_manifest"]["sha256"],
            "mission_id": old_publication["canonical_ppa"]["mission_id"],
            "non_sram_area_mm2": old_publication["canonical_ppa"]["non_sram_area_mm2"],
            "timing_100mhz_status": old_publication["canonical_ppa"]["timing_100mhz_status"],
            "tns_ns": old_publication["canonical_ppa"]["tns_ns"],
            "top_level_cell_count": old_publication["canonical_ppa"]["top_level_cell_count"],
            "wns_ns": old_publication["canonical_ppa"]["wns_ns"],
            "worst_setup_slack_ns": old_publication["canonical_ppa"]["worst_setup_slack_ns"],
        },
        "functional_status": "FUNCTIONAL_FULL_QWEN_TWO_TOKEN_INTEGRATION_ACCEPTED",
        "publication_id": PREVIOUS_PUBLICATION_ID,
        "rtl_tree_sha256": PREVIOUS_TREE,
        "status": "SUPERSEDED_BY_FRONTIER_PUBLICATION_2026_08_04T06_00_28Z",
        "successor_task_id": REPAIR_MISSION,
    }
    public.setdefault("historical_frontier_publications", []).append(previous_public_history)
    public.update({
        "blockers": [
            {
                "detail": (
                    "The accepted repaired tree misses 100 MHz with -0.1741 ns detailed slack on "
                    "u_rmsnorm_core/_7721_ -> _7482_; source mapping is collect_square_q[0] -> "
                    "div_dividend_q[47], caused by the final next_sumsq_w + HIDDEN_HALF_ACC carry cone."
                ),
                "id": "final_rmsnorm_sumsq_carry_cone_timing_no_go",
                "status": "OPEN",
            },
            {
                "detail": "Fresh cross-file Reviewer acceptance is required before releasing the single materialized repair.",
                "id": "publication_review_pending",
                "status": "OPEN",
            },
        ],
        "candidate_rtl_hash": TREE,
        "current_mode": "TIMING_REPAIR",
        "current_stage": "rtl",
        "generated_at_utc": PUBLICATION_AT,
        "last_updated_utc": PUBLICATION_AT,
        "latest_decision": "FUNCTIONAL_FULL_QWEN_ACCEPTED_RMSNORM_REPAIRED_TREE_100MHZ_NO_GO",
        "latest_ppa_frontier": ppa,
        "latest_ppa_frontier_status": "accepted_canonical_rmsnorm_repaired_tree_no_go_timing_area_pass",
        "latest_rtl_candidate": {
            "candidate_id": "rmsnorm_repaired_tree_a42bc846",
            "rmsnorm_rtl_sha256": RMSNORM,
            "rtl_tree_sha256": TREE,
            "status": "accepted_functional_and_canonical_ppa_no_go",
            "verification_complete": True,
        },
        "next_action_queue": [next_action],
        "public_claims": [
            {"claim": "Layer-0 passes 18/18 ordered operators.", "evidence": ["design/RTL_MANIFEST.json"]},
            {"claim": "Full-Qwen passes 13,914/13,914 commands with token IDs [0,0].", "evidence": ["design/RTL_MANIFEST.json"]},
            {"claim": "The sole canonical repaired-tree SKY130 PPA is area PASS and explicit 100 MHz NO_GO; timing is not certified.",
             "evidence": [f"{PPA_DIR_REL}/METRICS.json", f"{PPA_DIR_REL}/INDEPENDENT_METRICS.json"]},
        ],
        "required_manager_action": "obtain_fresh_publication_review_then_release_only_the_single_final_sumsq_carry_cone_repair",
        "required_operator_action": "none",
        "stage": "rtl",
        "stage_closing": False,
    })
    update_current_public_projections(public, ppa, full, next_action, current_publication)
    implementation = copy.deepcopy(public["implementation_frontier"])
    implementation.update({
        "canonical_ppa": ppa,
        "claim_boundary": CLAIM,
        "final_product_timing_certified": False,
        "full_qwen_integration": full,
        "next_action_queue": [next_action],
        "rtl_tree_sha256": TREE,
    })
    public["implementation_frontier"] = implementation
    public["reviewer_certified_metrics"] = {
        "canonical_sky130": ppa,
        "functional_integration": {
            "full_runtime_commands_passed": 13914,
            "full_runtime_commands_total": 13914,
            "generated_tokens": [0, 0],
            "layer0_commands_passed": 18,
            "layer0_commands_total": 18,
        },
    }

    ledger_bytes = encode_json(ledger)
    manifest_bytes = encode_json(manifest)
    trace_bytes = trace.encode("utf-8")
    pipeline_bytes = encode_json(pipeline)
    manifest_companion = f"{sha256_bytes(manifest_bytes)}  RTL_MANIFEST.json\n".encode("utf-8")
    trace_companion = f"{sha256_bytes(trace_bytes)}  RTL_TRACEABILITY.md\n".encode("utf-8")
    pipeline_companion = f"{sha256_bytes(pipeline_bytes)}  PIPELINE_STATE.json\n".encode("utf-8")

    records = list(public["artifact_hashes"])
    additions = [
        artifact(REPAIR_DIR / "repair_witness.json"),
        artifact(REPAIR_DIR / "SHA256SUMS"),
        artifact(PPA_DIR / "RESULT.json"),
        artifact(PPA_DIR / "METRICS.json"),
        artifact(PPA_DIR / "INDEPENDENT_METRICS.json"),
        artifact(PPA_DIR / "SHA256SUMS"),
        {"bytes": len(ledger_bytes), "path": "design/PPA_FRONTIER_LEDGER.json", "sha256": sha256_bytes(ledger_bytes)},
        {"bytes": len(manifest_bytes), "path": "design/RTL_MANIFEST.json", "sha256": sha256_bytes(manifest_bytes)},
        {"bytes": len(manifest_companion), "path": "design/RTL_MANIFEST.sha256", "sha256": sha256_bytes(manifest_companion)},
        {"bytes": len(trace_bytes), "path": "design/RTL_TRACEABILITY.md", "sha256": sha256_bytes(trace_bytes)},
        {"bytes": len(trace_companion), "path": "design/RTL_TRACEABILITY.sha256", "sha256": sha256_bytes(trace_companion)},
        {"bytes": len(pipeline_bytes), "path": "research/PIPELINE_STATE.json", "sha256": sha256_bytes(pipeline_bytes)},
        {"bytes": len(pipeline_companion), "path": "research/PIPELINE_STATE.sha256", "sha256": sha256_bytes(pipeline_companion)},
    ]
    by_path = {item["path"]: item for item in records}
    for item in additions:
        by_path[item["path"]] = item
    public["artifact_hashes"] = [by_path[path] for path in sorted(by_path)]
    public["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonical_sha256": None,
    }
    public["integrity"]["canonical_sha256"] = canonical_public_hash(public)
    public_bytes = encode_json(public)

    return {
        LEDGER: ledger_bytes,
        MANIFEST: manifest_bytes,
        MANIFEST_COMPANION: manifest_companion,
        TRACEABILITY: trace_bytes,
        TRACEABILITY_COMPANION: trace_companion,
        PIPELINE_STATE: pipeline_bytes,
        PIPELINE_COMPANION: pipeline_companion,
        PUBLIC_STATUS: public_bytes,
    }


def build_public_projection_repair() -> dict[Path, bytes]:
    _, repair_witness = verify_sealed_inputs()
    public = load_json(PUBLIC_STATUS)
    pipeline = load_json(PIPELINE_STATE)

    require(public.get("candidate_rtl_hash") == TREE, "PUBLIC_STATUS current tree differs")
    current_publication = copy.deepcopy(pipeline.get("current_publication", {}))
    require(current_publication.get("id") == PUBLICATION_ID, "PIPELINE_STATE current publication differs")
    full = copy.deepcopy(current_publication.get("full_qwen_integration"))
    require(isinstance(full, dict), "PIPELINE_STATE current functional integration is missing")
    next_actions = current_publication.get("next_action_queue", [])
    require(len(next_actions) == 1 and next_actions[0].get("id") == SUCCESSOR,
            "PIPELINE_STATE current successor differs")

    ppa = build_canonical_ppa(build_repair_binding(repair_witness))
    require(current_publication.get("canonical_ppa") == ppa, "PIPELINE_STATE canonical PPA differs")
    update_current_public_projections(public, ppa, full, next_actions[0], current_publication)
    public["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonical_sha256": None,
    }
    public["integrity"]["canonical_sha256"] = canonical_public_hash(public)
    return {PUBLIC_STATUS: encode_json(public)}


def atomic_commit(payloads: dict[Path, bytes]) -> None:
    staged: dict[Path, Path] = {}
    try:
        for path, content in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            staged[path] = Path(name)
        for path in payloads:
            if path == PUBLIC_STATUS:
                continue
            os.replace(staged[path], path)
        os.replace(staged[PUBLIC_STATUS], PUBLIC_STATUS)
        for directory in {path.parent for path in payloads}:
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        for path in staged.values():
            if path.exists():
                path.unlink()


def verify_artifact_record(record: dict[str, Any]) -> None:
    path = record.get("path")
    if not isinstance(path, str) or path.startswith("handoff:"):
        return
    target = ROOT / path
    require(target.is_file(), f"PUBLIC_STATUS artifact path missing: {path}")
    require(target.stat().st_size == record.get("bytes"), f"PUBLIC_STATUS artifact bytes differ: {path}")
    require(sha256_file(target) == record.get("sha256"), f"PUBLIC_STATUS artifact hash differs: {path}")


def check_published() -> None:
    _, repair_witness = verify_sealed_inputs()
    ledger = load_json(LEDGER)
    manifest = load_json(MANIFEST)
    public = load_json(PUBLIC_STATUS)
    pipeline = load_json(PIPELINE_STATE)
    trace = TRACEABILITY.read_text(encoding="utf-8")

    entries = ledger.get("entries", [])
    require(entries[-1].get("id") == PUBLICATION_ID, "ledger current publication differs")
    require(sum(item.get("id") == PUBLICATION_ID for item in entries) == 1, "ledger publication is not unique")
    require(entries[-1].get("rtl_hash") == TREE, "ledger tree differs")
    require(entries[-1].get("cells") == CELLS and entries[-1].get("non_sram_area_mm2") == AREA,
            "ledger PPA metrics differ")
    require(entries[-1].get("timing_100mhz_status") == "NO_GO", "ledger timing verdict differs")
    require(entries[-2].get("rtl_hash") == PREVIOUS_TREE, "ledger append-only predecessor was not preserved")

    require(manifest.get("rtl_hash") == TREE and manifest.get("candidate_rtl_hash") == TREE,
            "manifest current tree differs")
    require(manifest.get("accepted_frontier_item_count") == 18, "manifest accepted count differs")
    require(len(manifest.get("supported_layer_operator_prefix", [])) == 18, "manifest supported prefix length differs")
    require(manifest.get("canonical_ppa", {}).get("mission_id") == PPA_MISSION, "manifest canonical PPA differs")
    require(manifest.get("canonical_ppa", {}).get("timing_100mhz_status") == "NO_GO",
            "manifest timing verdict differs")
    require(manifest.get("full_qwen_integration", {}).get("runtime", {}).get("commands_passed") == 13914,
            "manifest full runtime pass count differs")
    require(manifest.get("full_qwen_integration", {}).get("runtime", {}).get("generated_tokens") == [0, 0],
            "manifest tokens differ")
    require(manifest.get("full_qwen_integration", {}).get("layer0", {}).get("accepted_operator_count") == 18,
            "manifest Layer-0 count differs")
    require(manifest.get("frontier_publication_history", [])[-1].get("publication_id") == PUBLICATION_ID,
            "manifest publication history differs")
    require(sum(item.get("publication_id") == PUBLICATION_ID for item in manifest["frontier_publication_history"]) == 1,
            "manifest publication history is not unique")
    rmsnorm_records = [item for item in manifest.get("source_hashes", [])
                       if item.get("path") == "rtl/ace2_rmsnorm_core.sv"]
    require(len(rmsnorm_records) == 1 and rmsnorm_records[0].get("sha256") == RMSNORM,
            "manifest RMSNorm source binding differs")

    require("## RMSNorm-repaired tree canonical timing NO_GO publication" in trace,
            "traceability publication section is missing")
    require(trace.count("## RMSNorm-repaired tree canonical timing NO_GO publication") == 1,
            "traceability publication section is duplicated")
    for text in [TREE, RMSNORM, "62,330", "-0.1741", "collect_square_q[0] -> div_dividend_q[47]",
                 "next_sumsq_w + HIDDEN_HALF_ACC", "final product timing is not certified"]:
        require(text in trace, f"traceability is missing {text}")

    require(public.get("candidate_rtl_hash") == TREE, "PUBLIC_STATUS tree differs")
    require(public.get("latest_ppa_frontier", {}).get("mission_id") == PPA_MISSION,
            "PUBLIC_STATUS canonical PPA differs")
    require(public.get("latest_ppa_frontier", {}).get("product_timing_certified") is False,
            "PUBLIC_STATUS overclaims timing")
    require(public.get("next_action_queue", [{}])[0].get("id") == SUCCESSOR,
            "PUBLIC_STATUS successor differs")
    require(public.get("next_action_queue", [{}])[0].get("materialization_count") == 1,
            "PUBLIC_STATUS successor materialization differs")
    expected_ppa = build_canonical_ppa(build_repair_binding(repair_witness))
    current_publication = pipeline.get("current_publication", {})
    full = current_publication.get("full_qwen_integration")
    next_actions = current_publication.get("next_action_queue", [])
    require(isinstance(full, dict), "PIPELINE_STATE current functional integration is missing")
    require(len(next_actions) == 1 and next_actions[0].get("id") == SUCCESSOR,
            "PIPELINE_STATE current successor projection differs")
    dashboard = public.get("dashboard_fields", {})
    require(dashboard.get("publication_id") == PUBLICATION_ID,
            "PUBLIC_STATUS dashboard publication differs")
    require(dashboard.get("canonical_ppa") == expected_ppa,
            "PUBLIC_STATUS dashboard canonical PPA differs")
    require(dashboard.get("canonical_sky130_ppa") == expected_ppa,
            "PUBLIC_STATUS dashboard canonical SKY130 PPA alias differs")
    require(dashboard.get("current_publication") == current_publication,
            "PUBLIC_STATUS dashboard current publication projection differs")
    require(dashboard.get("functional_integration") == full,
            "PUBLIC_STATUS dashboard functional projection differs")
    require(dashboard.get("next_action_queue") == next_actions,
            "PUBLIC_STATUS dashboard successor projection differs")
    historical_terminal_predecessor(public)
    require(
        public.get("terminal_baseline_no_go")
        == build_terminal_baseline_no_go(expected_ppa, full, next_actions[0]),
        "PUBLIC_STATUS current terminal NO_GO projection differs",
    )
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_public_hash(public),
            "PUBLIC_STATUS canonical integrity differs")
    paths = [item.get("path") for item in public.get("artifact_hashes", [])]
    require(len(paths) == len(set(paths)), "PUBLIC_STATUS artifact paths are duplicated")
    for record in public.get("artifact_hashes", []):
        verify_artifact_record(record)

    require(pipeline.get("current_stage") == "rtl", "PIPELINE_STATE stage differs")
    require(pipeline.get("current_publication", {}).get("id") == PUBLICATION_ID,
            "PIPELINE_STATE current publication differs")
    require(pipeline.get("current_publication", {}).get("canonical_ppa", {}).get("mission_id") == PPA_MISSION,
            "PIPELINE_STATE canonical PPA differs")
    require(pipeline.get("successor", {}).get("id") == SUCCESSOR, "PIPELINE_STATE successor differs")
    require(pipeline.get("successor", {}).get("materialization_count") == 1,
            "PIPELINE_STATE successor materialization differs")
    require(pipeline.get("successor", {}).get("canonical_ppa_authorized") is False,
            "PIPELINE_STATE successor improperly authorizes PPA")
    require(sum(item.get("publication_id") == PUBLICATION_ID for item in pipeline.get("stage_history", [])) == 1,
            "PIPELINE_STATE publication history is not unique")

    companions = [
        (MANIFEST, MANIFEST_COMPANION, "RTL_MANIFEST.json"),
        (TRACEABILITY, TRACEABILITY_COMPANION, "RTL_TRACEABILITY.md"),
        (PIPELINE_STATE, PIPELINE_COMPANION, "PIPELINE_STATE.json"),
    ]
    for target, companion, name in companions:
        expected = f"{sha256_file(target)}  {name}\n"
        require(companion.read_text(encoding="utf-8") == expected, f"stale companion: {companion.relative_to(ROOT)}")

    print(json.dumps({
        "area_mm2": AREA,
        "cells": CELLS,
        "publication_id": PUBLICATION_ID,
        "status": "PASS",
        "timing_100mhz": "NO_GO",
        "tns_ns": TNS,
        "tree": TREE,
        "wns_ns": WNS,
        "worst_setup_slack_ns": SLACK,
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.apply:
        public = load_json(PUBLIC_STATUS)
        payloads = build_public_projection_repair() if public.get("candidate_rtl_hash") == TREE else build_payloads()
        atomic_commit(payloads)
    check_published()


if __name__ == "__main__":
    main()
