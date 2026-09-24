#!/usr/bin/env python3
"""Seal the focused no-go for layer0_tile_max_delta_attention_v1."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "layer0_tile_max_delta_attention_v1"
PROPOSAL_SHA = "c9163bef51e0505f1b5157c8a847ef29c0524a8f1ca328abb35ad6ec84455e61"
BASE = ROOT / "evidence" / CONTRACT / "focused-baseline-full-20260801-v1" / "results.json"
CAND = ROOT / "evidence" / CONTRACT / "focused-candidate-full-20260801-v1" / "results.json"
LATEST = ROOT / "evidence" / CONTRACT / "latest"
NO_GO = LATEST / "BOUNDED_NO_GO.json"
SOURCE_LIST = LATEST / "source_hashes.txt"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
SOURCES = [
    "Makefile",
    "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
    "rtl/ace2_tile_max_delta_attention_core.sv",
    "tools/ace2_full_model_fixed_point.py",
    "tools/ace2_tile_max_delta_reference.py",
    "tools/gen_tile_max_delta_attention_vectors.py",
    "tools/localize_score_to_lm_head.py",
    "verification/generated/tile_max_delta_attention_vectors.json",
    "verification/generated/tile_max_delta_attention_vectors.svh",
    "verification/tb/ace2_tile_max_delta_attention_tb.sv",
    "verification/test_tile_max_delta_attention.py",
]
LOGS = [
    "evidence/layer0_tile_max_delta_attention_v1/latest/software_reference_unittest.log",
    "evidence/layer0_tile_max_delta_attention_v1/latest/full_model_self_test.log",
    "evidence/layer0_tile_max_delta_attention_v1/latest/rtl_tile_max_delta.log",
    "evidence/layer0_tile_max_delta_attention_v1/latest/rtl_score_lint.log",
    "evidence/layer0_tile_max_delta_attention_v1/latest/rtl_softmax_lint.log",
]


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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {
        "bytes": path.stat().st_size,
        "path": relative,
        "sha256": sha256(path),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    candidate = copy.deepcopy(value)
    candidate.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    require(load(ROOT / "research" / "PIPELINE_STATE.json").get("current_stage") == "rtl", "Manager stage is not rtl")
    require(sha256(ROOT / "design" / "NUMERICAL_REPLACEMENT_PROPOSAL.md") == PROPOSAL_SHA, "proposal hash changed")
    for relative in SOURCES + LOGS:
        require((ROOT / relative).is_file(), f"candidate artifact missing: {relative}")
    require("OK" in (ROOT / LOGS[0]).read_text(encoding="utf-8"), "software tests did not pass")
    require("SELF_TEST status=pass" in (ROOT / LOGS[1]).read_text(encoding="utf-8"), "full-model self-test did not pass")
    require("TB_PASS" in (ROOT / LOGS[2]).read_text(encoding="utf-8"), "RTL simulation did not pass")
    require((ROOT / LOGS[3]).stat().st_size == 0, "score lint is not warning-free")
    require((ROOT / LOGS[4]).stat().st_size == 0, "softmax lint is not warning-free")

    baseline = load(BASE)
    candidate = load(CAND)
    require(baseline["input_observations"] == candidate["input_observations"], "focused inputs differ")
    comparisons: dict[str, Any] = {}
    gate = True
    for dataset in ("wikitext2", "c4_en_512"):
        score_base = baseline["comparisons"][dataset]["model.layers.0.score"]
        score_cand = candidate["comparisons"][dataset]["model.layers.0.score"]
        value_base = baseline["comparisons"][dataset]["model.layers.0.attention_value"]
        value_cand = candidate["comparisons"][dataset]["model.layers.0.attention_value"]
        comparisons[dataset] = {
            "score_relative_l2": {
                "baseline": score_base["relative_l2_error"],
                "candidate": score_cand["relative_l2_error"],
                "strictly_improved": score_cand["relative_l2_error"] < score_base["relative_l2_error"],
            },
            "score_centered_zero_fraction": score_cand["candidate_zero_fraction"],
            "softmax_relative_l2": {
                "baseline": baseline["comparisons"][dataset]["model.layers.0.softmax"]["relative_l2_error"],
                "candidate": candidate["comparisons"][dataset]["model.layers.0.softmax"]["relative_l2_error"],
            },
            "attention_value_relative_l2": {
                "baseline": value_base["relative_l2_error"],
                "candidate": value_cand["relative_l2_error"],
                "strictly_improved": value_cand["relative_l2_error"] < value_base["relative_l2_error"],
            },
        }
        gate &= comparisons[dataset]["score_relative_l2"]["strictly_improved"]
        gate &= comparisons[dataset]["attention_value_relative_l2"]["strictly_improved"]
        gate &= comparisons[dataset]["score_centered_zero_fraction"] < 0.10
    require(not gate, "focused gate unexpectedly passed")
    require(all(comparisons[d]["score_centered_zero_fraction"] < 0.10 for d in comparisons), "zero collapse not removed")
    require(all(comparisons[d]["attention_value_relative_l2"]["strictly_improved"] for d in comparisons), "attention value did not improve")
    require(all(not comparisons[d]["score_relative_l2"]["strictly_improved"] for d in comparisons), "score unexpectedly improved")

    source_hashes = {relative: sha256(ROOT / relative) for relative in sorted(SOURCES)}
    SOURCE_LIST.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in source_hashes.items()),
        encoding="utf-8",
    )
    source_list_sha = sha256(SOURCE_LIST)
    payload = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "candidate_id": f"tile_max_delta_attention_{source_list_sha[:16]}",
        "sealed_at_utc": utc_now(),
        "decision": "bounded_no_go_focused_score_relative_l2_failed_both_datasets",
        "accepted_capability": False,
        "accepted_frontier_changed": False,
        "operator_approval_consumed": True,
        "stage_closing": False,
        "manager_owned_stage": "rtl",
        "proposal_sha256": PROPOSAL_SHA,
        "source_binding": {
            "ordered_source_hash_list": SOURCE_LIST.relative_to(ROOT).as_posix(),
            "ordered_source_hash_list_sha256": source_list_sha,
            "source_hashes": source_hashes,
        },
        "implementation_adequacy": {
            "software_reference_tests": artifact(LOGS[0]),
            "full_model_self_test": artifact(LOGS[1]),
            "standalone_rtl_simulation": artifact(LOGS[2]),
            "score_lint": artifact(LOGS[3]),
            "softmax_lint": artifact(LOGS[4]),
            "construct_fidelity": "independent scalar and tensor paths match; standalone RTL passes generated vectors; both lint logs are empty",
        },
        "focused_discriminator": {
            "baseline": artifact(BASE.relative_to(ROOT).as_posix()),
            "candidate": artifact(CAND.relative_to(ROOT).as_posix()),
            "comparisons": comparisons,
            "passed": False,
            "finding": "The mechanism removes the greater-than-90-percent zero collapse and improves softmax plus attention value, but the frozen score relative-L2 gate regresses on both datasets because the underflow representation clips far-negative centered scores.",
        },
        "paired_smoke": {
            "run": False,
            "reason": "forbidden because the focused discriminator failed the frozen strict score-improvement condition",
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
        },
        "required_next_action": "Manager reroutes rtl to architecture for a structurally distinct successor; no tile-max Q6.17 retune or paired smoke is authorized.",
        "claim_boundary": "Bounded negative focused result only; not RTL stage closure, PPA, benchmark acceptance, signoff, tapeout, or silicon evidence.",
    }
    dump(NO_GO, payload)
    no_go_sha = sha256(NO_GO)

    policy_path = ROOT / "design" / "FAST_LOOP_POLICY.json"
    policy = load(policy_path)
    for key in ("active_repair_authorization", "architecture_proposal_authorization", "selected_replacement_contract"):
        policy[key].update({
            "implementation_authorized": False,
            "status": "authorization_consumed_rejected_focused_discriminator",
            "operator_approval_consumed": True,
            "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
            "result_binding": NO_GO.relative_to(ROOT).as_posix(),
            "result_binding_sha256": no_go_sha,
        })
    dump(policy_path, policy)

    target_path = ROOT / "design" / "TARGET.json"
    target = load(target_path)
    target["current_stage"] = "rtl"
    target["fast_loop_contract"]["manager_recommendation"] = "reroute_rtl_to_architecture_after_tile_max_delta_focused_no_go"
    for key in ("active_repair_authorization", "architecture_proposal_authorization"):
        target["fast_loop_contract"][key] = copy.deepcopy(policy[key])
    target["current_architecture_contract"] = copy.deepcopy(policy["selected_replacement_contract"])
    dump(target_path, target)

    scope_path = ROOT / "design" / "CHIP_SCOPE.json"
    scope = load(scope_path)
    scope["authority_override"].update({
        "implementation_authorized": False,
        "required_next_action": "manager_reroute_rtl_to_architecture",
        "operator_approval_consumed": True,
        "result_binding": NO_GO.relative_to(ROOT).as_posix(),
    })
    scope["stage"] = {
        "current_stage": "rtl",
        "current_stage_status": "candidate_rejected_manager_reroute_required",
        "downstream_stages_locked_until_manager_advance": ["verification", "ppa", "prototype", "benchmark", "signoff"],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    scope["implementation_frontier"]["latest_decision"] = "tile_max_delta_focused_bounded_no_go"
    scope["operator_owned_execution_policy"]["active_successor_contract"] = copy.deepcopy(policy["selected_replacement_contract"])
    dump(scope_path, scope)

    manifest_path = ROOT / "design" / "RTL_MANIFEST.json"
    manifest = load(manifest_path)
    manifest["stage"] = "rtl"
    manifest["current_stage"] = "rtl"
    manifest["architecture_contract_status"] = "layer0_tile_max_delta_attention_v1_rejected_focused_gate_manager_reroute_required"
    manifest["candidate_rtl_hash"] = source_list_sha
    manifest["candidate_rtl_hash_scope"] = "tile_max_delta_reference_generated_vectors_and_standalone_rtl_not_shell_admitted"
    manifest["candidate_source_hashes"] = source_hashes
    manifest["candidate_meets_numeric_acceptance"] = False
    manifest["candidate_requires_fresh_sky130_ppa"] = False
    manifest["candidate_review_binding"] = {
        "candidate_capability_accepted": False,
        "decision": "bounded_no_go",
        "evidence": NO_GO.relative_to(ROOT).as_posix(),
        "evidence_sha256": no_go_sha,
        "scope": "focused_discriminator_stop_rule",
        "stage_closing": False,
    }
    manifest["proposed_replacement_contract"] = copy.deepcopy(policy["selected_replacement_contract"])
    manifest["traceability"]["architecture_contract_gap"] = {
        "resolution_owner": "Manager",
        "status": "candidate_rejected_manager_architecture_reroute_required",
    }
    manifest["traceability"]["rtl.contract-traceability"] = False
    dump(manifest_path, manifest)

    oracle_path = ROOT / "reference" / "ORACLE_MANIFEST.json"
    oracle = load(oracle_path)
    oracle["proposed_architecture_contract"] = copy.deepcopy(policy["selected_replacement_contract"])
    dump(oracle_path, oracle)

    status_path = ROOT / "research" / "PUBLIC_STATUS.json"
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
    status["latest_decision"] = "tile_max_delta_focused_bounded_no_go"
    status["selected_replacement_contract"] = {
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "status": "rejected_focused_discriminator",
        "result_binding": NO_GO.relative_to(ROOT).as_posix(),
        "result_binding_sha256": no_go_sha,
    }
    status["blockers"] = [{
        "id": "manager_architecture_reroute_required",
        "stage": "rtl",
        "status": "active",
        "reason": "The candidate removed zero collapse and improved attention value but failed the frozen score relative-L2 condition on both datasets.",
        "required_resolution": "Manager rolls rtl back to architecture for one structurally distinct successor.",
        "evidence": NO_GO.relative_to(ROOT).as_posix(),
    }]
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status[name]
        container["current_stage"] = "rtl"
        container["latest_decision"] = status["latest_decision"]
        container["routing_status"] = "manager_architecture_reroute_required"
        container["required_manager_action"] = "reroute_rtl_to_architecture"
        container["rtl_contract_traceability"] = False
        container["latest_quality_diagnostic"] = {
            "evidence": NO_GO.relative_to(ROOT).as_posix(),
            "score_zero_collapse_removed": True,
            "score_relative_l2_gate_passed": False,
            "attention_value_strictly_improved_both_datasets": True,
        }
    paths = {
        item["path"] for item in status.get("artifact_hashes", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str) and (ROOT / item["path"]).is_file()
    }
    paths.update(SOURCES + LOGS + [
        BASE.relative_to(ROOT).as_posix(), CAND.relative_to(ROOT).as_posix(),
        SOURCE_LIST.relative_to(ROOT).as_posix(), NO_GO.relative_to(ROOT).as_posix(),
        "tools/bind_tile_max_delta_no_go.py",
    ])
    status["artifact_hashes"] = [artifact(path) for path in sorted(paths)]
    status["generated_at_utc"] = utc_now()
    status["last_updated_utc"] = status["generated_at_utc"]
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": "UTF-8 sorted keys two-space indentation trailing newline integrity hash null during hash",
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump(status_path, status)
    print(
        "ACE2_TILE_MAX_DELTA_BOUNDED_NO_GO "
        f"source_list_sha256={source_list_sha} no_go_sha256={no_go_sha}"
    )


if __name__ == "__main__":
    main()
