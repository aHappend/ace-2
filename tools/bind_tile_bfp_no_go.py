#!/usr/bin/env python3
"""Seal the bounded no-go for layer0_tile_bfp_score_attention_v1."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "layer0_tile_bfp_score_attention_v1"
PROPOSAL_SHA = "a437a57a7d2dc35254222d818d7c2c2300fbc42e49bd60fa3001af5de4569c13"
LATEST = ROOT / "evidence" / CONTRACT / "latest"
CANDIDATE = LATEST / "candidate_evidence.json"
SMOKE = ROOT / f"evidence/{CONTRACT}/paired-smoke-20260801-v1/results.json"
SMOKE_LOG = LATEST / "paired_smoke_stdout_stderr.log"
NO_GO = LATEST / "BOUNDED_NO_GO.json"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
THRESHOLDS = {
    "wikitext2": 13549.939049967887,
    "c4_en_512": 4477.990517308544,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected object: {path}")
    return value


def dump(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: dict[str, Any]) -> str:
    candidate = copy.deepcopy(value)
    candidate.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    require(load(ROOT / "research/PIPELINE_STATE.json").get("current_stage") == "rtl",
            "Manager stage is not rtl")
    require(sha256(ROOT / "design/NUMERICAL_REPLACEMENT_PROPOSAL.md") == PROPOSAL_SHA,
            "proposal hash changed")
    candidate = load(CANDIDATE)
    require(candidate.get("contract_id") == CONTRACT, "candidate contract differs")
    require(candidate["focused_verification"].get("passed") is True,
            "focused discriminator did not pass")
    source_list = ROOT / candidate["source_binding"]["source_hash_list"]
    require(sha256(source_list) == candidate["source_binding"]["ordered_source_hash_list_sha256"],
            "candidate source list differs")
    for line in source_list.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        require(bool(separator) and len(expected) == 64, "candidate source list malformed")
        require(sha256(ROOT / relative) == expected, f"candidate source differs: {relative}")

    smoke = load(SMOKE)
    ratios = {dataset: smoke["metrics"][dataset]["ratio"] for dataset in THRESHOLDS}
    require(all(ratios[d] >= THRESHOLDS[d] for d in THRESHOLDS),
            "paired smoke unexpectedly passed one or more frozen thresholds")
    require("gate_passed=false" in SMOKE_LOG.read_text(encoding="utf-8"),
            "paired smoke log does not report failed gate")

    if not NO_GO.exists():
        payload = {
            "schema_version": 1,
            "contract_id": CONTRACT,
            "candidate_id": candidate["candidate_id"],
            "sealed_at_utc": utc_now(),
            "decision": "bounded_no_go_unique_paired_smoke_failed_both_datasets",
            "accepted_capability": False,
            "accepted_frontier_changed": False,
            "operator_approval_consumed": True,
            "stage_closing": False,
            "manager_owned_stage": "rtl",
            "proposal_sha256": PROPOSAL_SHA,
            "source_binding": {
                "candidate_evidence": artifact(CANDIDATE),
                "ordered_source_hash_list": artifact(source_list),
                "candidate_source_hashes": candidate["source_binding"]["source_hashes"],
                "no_go_binder": artifact(ROOT / "tools/bind_tile_bfp_no_go.py"),
            },
            "implementation_adequacy": {
                "focused_verification": candidate["focused_verification"],
                "construct_fidelity": "independent scalar/tensor equality, generated RTL vectors, standalone Icarus, full-model self-test, and warning-free Verilator lint passed",
            },
            "focused_discriminator": {
                "passed": True,
                "comparisons": candidate["focused_verification"]["comparisons"],
                "finding": "Block-floating storage preserved the complete centered-score range and strictly improved score and attention-value relative-L2 on both frozen datasets.",
            },
            "paired_smoke": {
                "run": True,
                "results": artifact(SMOKE),
                "stdout_stderr": artifact(SMOKE_LOG),
                "comparisons": {
                    dataset: {
                        "candidate_ratio": ratios[dataset],
                        "required_strictly_below": THRESHOLDS[dataset],
                        "strictly_improved": ratios[dataset] < THRESHOLDS[dataset],
                    }
                    for dataset in THRESHOLDS
                },
                "passed": False,
                "finding": "Both frozen 128-token ratios regressed relative to the immediate predecessor threshold, so the unique paired-smoke gate failed.",
            },
            "downstream_runs": {
                "full_shell_regression": False,
                "canonical_sky130_ppa": False,
                "official_validation": False,
                "prototype": False,
                "full_benchmark": False,
                "signoff": False,
            },
            "preserved_frontier": {
                "mode": "ADVANCE",
                "supported_layer_operator_prefix": PREFIX,
                "first_unsupported_layer_operator": "layer_0.rope_q",
                "cells": 62199,
                "non_sram_area_mm2": 0.6108746272,
                "setup_slack_ns_at_100mhz": 0.1502,
                "area_cap_mm2": 2.0,
                "frequency_floor_mhz": 100.0,
                "remaining_area_reserve_mm2": 1.3891253728,
                "candidate_ppa_run": False,
            },
            "required_next_action": "Manager reroutes rtl to architecture for a structurally distinct successor; no tile-BFP retune, full-shell regression, or PPA is authorized.",
            "claim_boundary": "Bounded negative paired-smoke result only; not accepted RTL capability, PPA, benchmark acceptance, signoff, tapeout, or silicon evidence.",
        }
        dump(NO_GO, payload)
    no_go_sha = sha256(NO_GO)

    policy_path = ROOT / "design/FAST_LOOP_POLICY.json"
    policy = load(policy_path)
    for key in ("active_repair_authorization", "architecture_proposal_authorization", "selected_replacement_contract"):
        policy[key].update({
            "implementation_authorized": False,
            "status": "authorization_consumed_rejected_unique_paired_smoke",
            "operator_approval_consumed": True,
            "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
            "result_binding": NO_GO.relative_to(ROOT).as_posix(),
            "result_binding_sha256": no_go_sha,
        })
    dump(policy_path, policy)

    target_path = ROOT / "design/TARGET.json"
    target = load(target_path)
    target["current_stage"] = "rtl"
    target["fast_loop_contract"]["manager_recommendation"] = "reroute_rtl_to_architecture_after_tile_bfp_paired_smoke_no_go"
    for key in ("active_repair_authorization", "architecture_proposal_authorization"):
        target["fast_loop_contract"][key] = copy.deepcopy(policy[key])
    target["current_architecture_contract"] = copy.deepcopy(policy["selected_replacement_contract"])
    dump(target_path, target)

    scope_path = ROOT / "design/CHIP_SCOPE.json"
    scope = load(scope_path)
    scope["authority_override"].update({
        "implementation_authorized": False,
        "required_next_action": "manager_reroute_rtl_to_architecture",
        "operator_approval_consumed": True,
        "result_binding": NO_GO.relative_to(ROOT).as_posix(),
        "result_binding_sha256": no_go_sha,
    })
    scope["stage"] = {
        "current_stage": "rtl",
        "current_stage_status": "candidate_rejected_manager_reroute_required",
        "downstream_stages_locked_until_manager_advance": ["verification", "ppa", "prototype", "benchmark", "signoff"],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    scope["implementation_frontier"]["latest_decision"] = "tile_bfp_unique_paired_smoke_bounded_no_go"
    scope["implementation_frontier"]["first_unsupported_layer_operator"]["reason"] = "The accepted prefix remains through layer_0.v_proj; the tile-BFP successor passed focused checks but failed both unique paired-smoke thresholds."
    scope["operator_owned_execution_policy"]["active_successor_contract"] = copy.deepcopy(policy["selected_replacement_contract"])
    dump(scope_path, scope)

    manifest_path = ROOT / "design/RTL_MANIFEST.json"
    manifest = load(manifest_path)
    manifest.update({
        "stage": "rtl",
        "current_stage": "rtl",
        "architecture_contract_status": f"{CONTRACT}_rejected_paired_smoke_manager_reroute_required",
        "candidate_status": "authorization_consumed_rejected_unique_paired_smoke",
        "candidate_meets_numeric_acceptance": False,
        "candidate_requires_fresh_sky130_ppa": False,
        "candidate_verification_complete": False,
        "candidate_review_binding": {
            "candidate_capability_accepted": False,
            "decision": "bounded_no_go",
            "evidence": NO_GO.relative_to(ROOT).as_posix(),
            "evidence_sha256": no_go_sha,
            "focused_discriminator_passed": True,
            "paired_smoke_passed": False,
            "stage_closing": False,
        },
    })
    manifest["candidate_verification_binding"] = {
        "status": "focused_pass_paired_smoke_failed_candidate_rejected",
        "evidence": [artifact(NO_GO), artifact(SMOKE), artifact(SMOKE_LOG)],
    }
    manifest["traceability"]["architecture_contract_gap"] = {
        "resolution_owner": "Manager",
        "status": "candidate_rejected_manager_architecture_reroute_required",
    }
    manifest["traceability"]["rtl.contract-traceability"] = False
    manifest["traceability"]["stage_checklist"]["rtl.contract-traceability"] = False
    manifest["latest_evidence"]["tile_bfp_bounded_no_go"] = artifact(NO_GO)
    dump(manifest_path, manifest)

    oracle_path = ROOT / "reference/ORACLE_MANIFEST.json"
    oracle = load(oracle_path)
    oracle["proposed_architecture_contract"] = copy.deepcopy(policy["selected_replacement_contract"])
    dump(oracle_path, oracle)

    status_path = ROOT / "research/PUBLIC_STATUS.json"
    status = load(status_path)
    status["stage"].update({
        "current_stage": "rtl",
        "current_stage_status": "candidate_rejected_manager_reroute_required",
        "current_stage_checklist": {
            "rtl.contract-traceability": False,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
        },
        "current_stage_evidence": ["design/RTL_MANIFEST.json", "design/RTL_TRACEABILITY.md", NO_GO.relative_to(ROOT).as_posix()],
    })
    status["latest_decision"] = "tile_bfp_unique_paired_smoke_bounded_no_go"
    status["selected_replacement_contract"] = {
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "status": "rejected_unique_paired_smoke",
        "result_binding": NO_GO.relative_to(ROOT).as_posix(),
        "result_binding_sha256": no_go_sha,
    }
    status["blockers"] = [{
        "id": "manager_architecture_reroute_required",
        "stage": "rtl",
        "status": "active",
        "reason": "The tile-BFP candidate passed focused arithmetic and quality checks but failed both frozen unique paired-smoke improvement thresholds.",
        "required_resolution": "Manager rolls rtl back to architecture for one structurally distinct successor.",
        "evidence": NO_GO.relative_to(ROOT).as_posix(),
    }]
    latest_quality = {
        "evidence": NO_GO.relative_to(ROOT).as_posix(),
        "focused_discriminator_passed": True,
        "paired_smoke_passed": False,
        "wikitext2_ratio": ratios["wikitext2"],
        "c4_en_512_ratio": ratios["c4_en_512"],
        "wikitext2_required_strictly_below": THRESHOLDS["wikitext2"],
        "c4_en_512_required_strictly_below": THRESHOLDS["c4_en_512"],
    }
    latest_rtl = {
        "candidate_id": candidate["candidate_id"],
        "evidence": NO_GO.relative_to(ROOT).as_posix(),
        "rtl_hash": candidate["source_binding"]["ordered_source_hash_list_sha256"],
        "rtl_hash_scope": "tile_bfp_reference_vectors_standalone_rtl_and_focused_quality_not_shell_admitted",
        "stage_closing": False,
        "status": "rejected_unique_paired_smoke_failed_both_datasets",
    }
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status[name]
        container.update({
            "current_stage": "rtl",
            "current_mode": "ADVANCE",
            "mode": "ADVANCE",
            "latest_decision": status["latest_decision"],
            "routing_status": "manager_architecture_reroute_required",
            "required_manager_action": "reroute_rtl_to_architecture",
            "rtl_contract_traceability": False,
            "ordered_supported_layer_operator_prefix": PREFIX,
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "latest_quality_diagnostic": latest_quality,
            "latest_rtl_candidate": latest_rtl,
            "latest_ppa_frontier_status": "historical_frontier_preserved_no_candidate_ppa",
        })
        if isinstance(container.get("latest_ppa_frontier"), dict):
            container["latest_ppa_frontier"].update({
                "decision": "preserve_historical_hash_bound_ppa_after_tile_bfp_failed_pre_ppa_smoke_gate",
                "status": "historical_preserved_after_tile_bfp_failed_pre_ppa_gate",
                "first_unsupported_layer_operator": "layer_0.rope_q",
                "ordered_supported_layer_operator_prefix": PREFIX,
                "mode": "ADVANCE",
                "candidate_ppa_run": False,
            })
    paths = {
        item["path"] for item in status.get("artifact_hashes", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str) and (ROOT / item["path"]).is_file()
    }
    paths.update({
        CANDIDATE.relative_to(ROOT).as_posix(),
        source_list.relative_to(ROOT).as_posix(),
        SMOKE.relative_to(ROOT).as_posix(),
        SMOKE_LOG.relative_to(ROOT).as_posix(),
        NO_GO.relative_to(ROOT).as_posix(),
        "tools/bind_tile_bfp_no_go.py",
        "tools/bind_tile_bfp_candidate.py",
        "tools/bind_tile_bfp_rtl_state.py",
    })
    status["artifact_hashes"] = [artifact(ROOT / path) for path in sorted(paths)]
    status["generated_at_utc"] = utc_now()
    status["last_updated_utc"] = status["generated_at_utc"]
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": "UTF-8 sorted keys two-space indentation trailing newline integrity hash null during hash",
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump(status_path, status)
    print(f"ACE2_TILE_BFP_BOUNDED_NO_GO no_go_sha256={no_go_sha}")


if __name__ == "__main__":
    main()
