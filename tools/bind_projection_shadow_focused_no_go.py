#!/usr/bin/env python3
"""Seal the projection-shadow candidate after its unique paired smoke."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "layer0_projection_shadow_staged_attention_v1"
PROPOSAL_SHA256 = "bb686fdb1a8e886ba80e8578db25a5db39eb58a757fc053986fc4538d6c36d92"
EVIDENCE = ROOT / "evidence" / CONTRACT / "latest"
APPROVAL = ROOT / "evidence" / "authorization" / CONTRACT / "operator_approval.json"
BASELINE = ROOT / "evidence" / CONTRACT / "focused-baseline-full-20260801" / "results.json"
CANDIDATE = ROOT / "evidence" / CONTRACT / "focused-candidate-full-20260801" / "results.json"
CANDIDATE_BINDING = EVIDENCE / "candidate_evidence.json"
SMOKE_DIR = (
    ROOT
    / "benchmark/raw/quality"
    / "layer0-projection-shadow-staged-attention-v1-smoke-128-20260801"
)
SMOKE = SMOKE_DIR / "results.json"
SMOKE_LOG = EVIDENCE / "paired_smoke_stdout_stderr.log"
NO_GO = EVIDENCE / "BOUNDED_NO_GO.json"
SOURCES = [
    "Makefile",
    "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
    f"evidence/authorization/{CONTRACT}/operator_approval.json",
    "rtl/ace2_projection_shadow_staged_attention_core.sv",
    "tools/ace2_full_model_fixed_point.py",
    "tools/ace2_projection_shadow_reference.py",
    "tools/bind_projection_shadow_candidate.py",
    "tools/gen_projection_shadow_vectors.py",
    "tools/localize_score_to_lm_head.py",
    "verification/generated/projection_shadow_vectors.json",
    "verification/generated/projection_shadow_vectors.svh",
    "verification/tb/ace2_projection_shadow_staged_attention_tb.sv",
    "verification/test_projection_shadow_staged_attention.py",
]
LOGS = [
    "evidence/layer0_projection_shadow_staged_attention_v1/latest/software_reference_unittest.log",
    "evidence/layer0_projection_shadow_staged_attention_v1/latest/full_model_self_test.log",
    "evidence/layer0_projection_shadow_staged_attention_v1/latest/runtime_dependency.log",
    "evidence/layer0_projection_shadow_staged_attention_v1/latest/rtl_projection_shadow.log",
    "evidence/layer0_projection_shadow_staged_attention_v1/latest/rtl_projection_lint.log",
    "evidence/layer0_projection_shadow_staged_attention_v1/latest/rtl_score_lint.log",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: str | Path) -> dict[str, Any]:
    target = path if isinstance(path, Path) and path.is_absolute() else ROOT / path
    value = json.loads(target.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {target}")
    return value


def write(path: str | Path, value: dict[str, Any]) -> None:
    target = path if isinstance(path, Path) and path.is_absolute() else ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def artifact(path: str | Path) -> dict[str, Any]:
    target = path if isinstance(path, Path) and path.is_absolute() else ROOT / path
    require(target.is_file(), f"missing artifact: {target}")
    return {
        "bytes": target.stat().st_size,
        "path": target.relative_to(ROOT).as_posix(),
        "sha256": sha256(target),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    canonical = copy.deepcopy(value)
    canonical.setdefault("integrity", {})["canonical_sha256"] = None
    raw = (json.dumps(canonical, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def metric(result: dict[str, Any], dataset: str, boundary: str) -> dict[str, Any]:
    return result["comparisons"][dataset][boundary]


def refresh_public_integrity() -> None:
    status = load("research/PUBLIC_STATUS.json")
    paths = {
        str(item.get("path"))
        for item in status.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path") and (ROOT / str(item["path"])).is_file()
    }
    paths.update({
        "CHECKPOINT.md",
        "MISSION.md",
        "design/RTL_TRACEABILITY.md",
        "tools/bind_projection_shadow_candidate.py",
        "tools/bind_projection_shadow_focused_no_go.py",
    })
    status["artifact_hashes"] = [artifact(path) for path in sorted(paths)]
    status.setdefault("integrity", {})["canonical_sha256"] = None
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    write("research/PUBLIC_STATUS.json", status)


def main() -> None:
    require(load("research/PIPELINE_STATE.json").get("current_stage") == "rtl",
            "Manager-owned stage is not rtl")
    require(sha256(ROOT / "design/NUMERICAL_REPLACEMENT_PROPOSAL.md") == PROPOSAL_SHA256,
            "proposal hash changed")
    approval = load(APPROVAL)
    require(approval.get("authority") == "operator" and approval.get("contract_id") == CONTRACT,
            "operator approval binding is missing")
    for source in SOURCES:
        require((ROOT / source).is_file(), f"candidate source missing: {source}")
    for log in LOGS:
        require((ROOT / log).is_file(), f"focused log missing: {log}")
    require("OK" in (ROOT / LOGS[0]).read_text(encoding="utf-8"),
            "software unit tests did not pass")
    require("SELF_TEST status=pass" in (ROOT / LOGS[1]).read_text(encoding="utf-8"),
            "full-model self-test did not pass")
    require("gmpy2 2.3.1" in (ROOT / LOGS[2]).read_text(encoding="utf-8"),
            "exact runtime repair is not bound")
    require("TB_PASS" in (ROOT / LOGS[3]).read_text(encoding="utf-8"),
            "focused RTL simulation did not pass")
    require((ROOT / LOGS[4]).stat().st_size == 0 and (ROOT / LOGS[5]).stat().st_size == 0,
            "focused RTL lint is not warning-free")

    baseline = load(BASELINE)
    candidate = load(CANDIDATE)
    require(baseline["input_observations"] == candidate["input_observations"],
            "baseline and candidate focused inputs differ")
    require(baseline.get("capture_token_limit") == candidate.get("capture_token_limit") == 128,
            "focused run did not use the full frozen captures")
    require(candidate.get("diagnostic_rope_mechanism") == CONTRACT,
            "candidate mechanism mismatch")

    comparisons: dict[str, Any] = {}
    focused_gate_passed = True
    for dataset in ("wikitext2", "c4_en_512"):
        comparisons[dataset] = {}
        for boundary in (
            "model.layers.0.score",
            "model.layers.0.softmax",
            "model.layers.0.attention_value",
        ):
            before = metric(baseline, dataset, boundary)
            after = metric(candidate, dataset, boundary)
            improved = after["relative_l2_error"] < before["relative_l2_error"]
            comparisons[dataset][boundary] = {
                "baseline_relative_l2": before["relative_l2_error"],
                "candidate_relative_l2": after["relative_l2_error"],
                "strictly_improved": improved,
                "baseline_candidate_zero_fraction": before["candidate_zero_fraction"],
                "candidate_candidate_zero_fraction": after["candidate_zero_fraction"],
                "baseline_candidate_sha256": before["candidate_sha256"],
                "candidate_candidate_sha256": after["candidate_sha256"],
            }
            if boundary in ("model.layers.0.score", "model.layers.0.attention_value"):
                focused_gate_passed &= improved
    require(not focused_gate_passed, "no-go binder refuses a passing focused gate")

    candidate_binding = load(CANDIDATE_BINDING)
    require(candidate_binding.get("contract_id") == CONTRACT,
            "candidate binding contract mismatch")
    source_list = ROOT / candidate_binding["source_binding"]["source_hash_list"]
    require(source_list.is_file(), "candidate source hash list is missing")
    rtl_hash = sha256(source_list)
    require(
        rtl_hash
        == candidate_binding["source_binding"]["ordered_source_hash_list_sha256"],
        "candidate source hash list changed after binding",
    )
    source_hashes: dict[str, str] = {}
    for line in source_list.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        require(separator == "  " and len(digest) == 64,
                f"malformed candidate source hash line: {line!r}")
        require((ROOT / relative).is_file(), f"candidate source missing: {relative}")
        require(sha256(ROOT / relative) == digest,
                f"candidate source changed after smoke: {relative}")
        source_hashes[relative] = digest

    smoke = load(SMOKE)
    require(smoke.get("gate_passed") is False, "no-go binder refuses a passing paired smoke")
    require(
        smoke.get("diagnostic_numerical_repair", {}).get("mechanism") == CONTRACT,
        "paired smoke mechanism mismatch",
    )
    smoke_rtl = smoke["artifacts"]["accepted_rtl"]
    require(smoke_rtl.get("candidate_id") == candidate_binding["candidate_id"],
            "paired smoke candidate id mismatch")
    require(smoke_rtl.get("candidate_rtl_hash") == rtl_hash,
            "paired smoke candidate hash mismatch")
    require(smoke_rtl.get("binding", {}).get("sha256") == sha256(CANDIDATE_BINDING),
            "paired smoke candidate evidence binding mismatch")
    approval_thresholds = approval["thresholds"]
    smoke_ratios: dict[str, dict[str, float | bool]] = {}
    for dataset, threshold_key in (
        ("wikitext2", "wikitext2_ratio_strictly_below"),
        ("c4_en_512", "c4_en_512_ratio_strictly_below"),
    ):
        ratio = float(smoke["metrics"][dataset]["ratio"])
        threshold = float(approval_thresholds[threshold_key])
        smoke_ratios[dataset] = {
            "required_strictly_below": threshold,
            "candidate": ratio,
            "strictly_improved": ratio < threshold,
            "delta": ratio - threshold,
        }
    require(not any(item["strictly_improved"] for item in smoke_ratios.values()),
            "expected both paired-smoke datasets to fail")

    generated_at = utc_now()
    no_go = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "proposal_sha256": PROPOSAL_SHA256,
        "generated_at_utc": generated_at,
        "decision": "bounded_no_go_focused_discriminator_and_paired_smoke_failed_both_datasets",
        "accepted_capability": False,
        "accepted_frontier_changed": False,
        "operator_approval_consumed": True,
        "stage_closing": False,
        "manager_owned_stage": "rtl",
        "implementation_adequacy": {
            "software_reference_unit_tests": {"status": "pass", **artifact(LOGS[0])},
            "full_model_self_test": {"status": "pass", **artifact(LOGS[1])},
            "exact_runtime": {"status": "pass", **artifact(LOGS[2])},
            "standalone_rtl_simulation": {"status": "pass", **artifact(LOGS[3])},
            "projection_lint": {"status": "pass_warning_free", **artifact(LOGS[4])},
            "score_lint": {"status": "pass_warning_free", **artifact(LOGS[5])},
            "candidate_binding": artifact(CANDIDATE_BINDING),
            "paired_smoke_runtime": artifact(SMOKE_LOG),
            "construct_fidelity": (
                "candidate tensor path matches the independent scalar oracle and standalone RTL "
                "on generated vectors; baseline and candidate full-capture input hashes match"
            ),
        },
        "focused_discriminator": {
            "baseline": artifact(BASELINE),
            "candidate": artifact(CANDIDATE),
            "capture_token_limit": 128,
            "input_observations": candidate["input_observations"],
            "comparisons": comparisons,
            "passed": False,
            "strongest_finding": (
                "The frozen Q6.9 materialization collapses centered score diversity: candidate "
                "zero fractions exceed 90% on both datasets and score plus attention-value "
                "relative-L2 are worse than the same-flow accepted baseline."
            ),
        },
        "paired_smoke": {
            "run": True,
            "passed": False,
            "result": artifact(SMOKE),
            "run_contract": artifact(SMOKE_DIR / "run_contract.json"),
            "ratios": smoke_ratios,
            "reason": "Both mandatory paired-smoke ratios regressed against the frozen thresholds.",
        },
        "downstream_runs": {
            "full_shell_regression": False,
            "canonical_sky130_ppa": False,
            "official_14_item_evaluation": False,
            "prototype": False,
            "full_benchmark": False,
            "signoff": False,
        },
        "source_binding": {
            "candidate_id": candidate_binding["candidate_id"],
            "candidate_evidence": artifact(CANDIDATE_BINDING),
            "ordered_source_hash_list": source_list.relative_to(ROOT).as_posix(),
            "ordered_source_hash_list_sha256": rtl_hash,
            "source_hashes": source_hashes,
        },
        "preserved_frontier": {
            "mode": "ADVANCE",
            "supported_layer_operator_prefix": [
                "layer_0.input_rmsnorm",
                "layer_0.q_proj",
                "layer_0.k_proj",
                "layer_0.v_proj",
            ],
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "cells": 62199,
            "non_sram_area_mm2": 0.6108746272,
            "setup_slack_ns_at_100mhz": 0.1502,
            "area_cap_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
        },
        "required_next_action": (
            "Manager reroutes rtl to architecture for a structurally distinct successor; "
            "no continuation of projection-shadow Q6.9 staged attention is authorized."
        ),
    }
    write(NO_GO, no_go)
    no_go_hash = sha256(NO_GO)

    manifest = load("design/RTL_MANIFEST.json")
    manifest.update({
        "generated_at_utc": generated_at,
        "candidate_layer_operator": CONTRACT,
        "candidate_status": "rejected_focused_discriminator_and_paired_smoke_failed",
        "candidate_rtl_hash": rtl_hash,
        "candidate_rtl_hash_scope": (
            "projection_shadow_finalizer_wide_absolute_rope_score_reference_vectors_and_"
            "focused_standalone_rtl_not_shell_admitted"
        ),
        "candidate_source_hashes": source_hashes,
        "candidate_verification_complete": True,
        "candidate_verification_binding": {
            "classification": "focused_current_stage_arithmetic_not_independent_verification_stage",
            "logs": [artifact(path) for path in LOGS],
            "baseline_full_capture": artifact(BASELINE),
            "candidate_full_capture": artifact(CANDIDATE),
            "candidate_binding": artifact(CANDIDATE_BINDING),
            "paired_smoke": artifact(SMOKE),
            "result": artifact(NO_GO),
        },
        "candidate_meets_numeric_acceptance": False,
        "candidate_requires_fresh_sky130_ppa": False,
        "candidate_supported_layer_operator_prefix_after_review": manifest[
            "supported_layer_operator_prefix"
        ],
        "candidate_first_unsupported_layer_operator_after_review": "layer_0.rope_q",
        "architecture_contract_status": (
            "layer0_projection_shadow_staged_attention_v1_implementation_completed_"
            "rejected_paired_smoke_manager_reroute_required"
        ),
        "interfaces_contract_status": (
            "accepted_shell_prefix_through_v_proj_projection_shadow_candidate_rejected_"
            "before_shell_admission"
        ),
    })
    manifest["proposed_replacement_contract"].update({
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "status": "rejected_focused_discriminator_and_paired_smoke_failed",
        "required_operator_action": "fresh_authority_only_after_new_architecture_contract",
        "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
        "result_binding": NO_GO.relative_to(ROOT).as_posix(),
        "result_binding_sha256": no_go_hash,
        "implementation_result": {
            "rtl_change_started": True,
            "focused_software_rtl_tests_run": True,
            "focused_software_rtl_tests_passed": True,
            "focused_full_capture_gate_run": True,
            "focused_full_capture_gate_passed": False,
            "paired_smoke_run": True,
            "paired_smoke_passed": False,
            "accepted_rtl_change": False,
            "accepted_verification_evidence": False,
            "accepted_ppa_evidence": False,
            "result_binding": NO_GO.relative_to(ROOT).as_posix(),
        },
    })
    manifest["traceability"].update({
        "selected_mechanism": CONTRACT,
        "architecture_contract_gap": {
            "status": "candidate_rejected_paired_smoke_manager_reroute_required",
            "resolution_owner": "Manager",
        },
        "rtl.contract-traceability": False,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
    })
    manifest["traceability"]["stage_checklist"] = {
        "rtl.contract-traceability": False,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
    }
    for entry in (
        {
            "kind": "first_party_rejected_diagnostic_rtl",
            "license": "repository project license not separately declared in this manifest",
            "name": "ace2_projection_shadow_core_and_score_core",
            "path": "rtl/ace2_projection_shadow_staged_attention_core.sv",
            "sha256": source_hashes["rtl/ace2_projection_shadow_staged_attention_core.sv"],
            "source_revision": source_hashes["rtl/ace2_projection_shadow_staged_attention_core.sv"],
            "third_party": False,
        },
        {
            "kind": "generated_rejected_diagnostic_verification_source",
            "name": "projection_shadow_vectors",
            "generator": "tools/gen_projection_shadow_vectors.py",
            "regeneration_command": "./.venv/bin/python tools/gen_projection_shadow_vectors.py",
            "json_path": "verification/generated/projection_shadow_vectors.json",
            "json_sha256": source_hashes["verification/generated/projection_shadow_vectors.json"],
            "svh_path": "verification/generated/projection_shadow_vectors.svh",
            "svh_sha256": source_hashes["verification/generated/projection_shadow_vectors.svh"],
            "third_party": False,
        },
    ):
        manifest["ip_provenance"] = [
            item for item in manifest["ip_provenance"] if item.get("name") != entry["name"]
        ]
        manifest["ip_provenance"].append(entry)
    write("design/RTL_MANIFEST.json", manifest)

    policy = load("design/FAST_LOOP_POLICY.json")
    for key in ("active_repair_authorization", "architecture_proposal_authorization", "selected_replacement_contract"):
        policy[key].update({
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "status": "authorization_consumed_rejected_paired_smoke",
            "required_operator_action": "fresh_authority_only_after_new_architecture_contract",
            "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
            "result_binding": NO_GO.relative_to(ROOT).as_posix(),
            "result_binding_sha256": no_go_hash,
        })
    policy["active_repair_authorization"]["execution_status"] = (
        "completed_bounded_no_go_paired_smoke_failed"
    )
    policy["selected_replacement_contract"]["execution_result"] = manifest[
        "proposed_replacement_contract"
    ]["implementation_result"]
    write("design/FAST_LOOP_POLICY.json", policy)

    target = load("design/TARGET.json")
    target["generated_at_utc"] = generated_at
    target["fast_loop_contract"]["manager_recommendation"] = (
        "reroute_rtl_to_architecture_after_projection_shadow_paired_smoke_no_go"
    )
    for key in ("active_repair_authorization", "architecture_proposal_authorization"):
        target["fast_loop_contract"][key].update({
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "status": "authorization_consumed_rejected_paired_smoke",
            "required_operator_action": "fresh_authority_only_after_new_architecture_contract",
            "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
            "result_binding": NO_GO.relative_to(ROOT).as_posix(),
        })
    write("design/TARGET.json", target)

    chip = load("design/CHIP_SCOPE.json")
    chip["authority_override"].update({
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "required_next_action": "manager_reroute_to_architecture_for_structurally_distinct_successor",
        "result_binding": NO_GO.relative_to(ROOT).as_posix(),
    })
    chip["numerical_behavior"]["rope"]["projection_shadow_staged_attention_contract"].update({
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "status": "rejected_focused_discriminator_and_paired_smoke_failed",
        "result_binding": NO_GO.relative_to(ROOT).as_posix(),
    })
    chip["operator_owned_execution_policy"]["active_successor_contract"].update({
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "status": "rejected_focused_discriminator_and_paired_smoke_failed",
        "observed_result": "focused_quality_worsened_and_both_mandatory_paired_smoke_ratios_regressed",
        "result_binding": NO_GO.relative_to(ROOT).as_posix(),
    })
    chip["stage"]["current_stage_status"] = "candidate_rejected_manager_reroute_required"
    write("design/CHIP_SCOPE.json", chip)

    memory = load("design/MEMORY_MODEL.json")
    memory["authority_override"].update({
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "required_next_action": "manager_reroute_to_architecture_for_structurally_distinct_successor",
        "result_binding": NO_GO.relative_to(ROOT).as_posix(),
    })
    memory["claim_status"] = "candidate_rejected_before_shell_admission_or_ppa"
    write("design/MEMORY_MODEL.json", memory)

    oracle = load("reference/ORACLE_MANIFEST.json")
    oracle["proposed_architecture_contract"].update({
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "status": "focused_vectors_passed_but_focused_discriminator_and_paired_smoke_failed",
        "result_binding": NO_GO.relative_to(ROOT).as_posix(),
    })
    oracle["generated_at_utc"] = generated_at
    oracle["claim_boundary"] = (
        "The projection-shadow oracle and standalone RTL are retained as rejected diagnostic "
        "evidence. The paired smoke ran and failed both mandatory ratios; no shell capability, "
        "PPA, official evaluation, prototype, full benchmark, or signoff claim exists."
    )
    write("reference/ORACLE_MANIFEST.json", oracle)

    status = load("research/PUBLIC_STATUS.json")
    decision = "projection_shadow_paired_smoke_bounded_no_go"
    for key in ("architecture_proposal_gate", "selected_replacement_contract"):
        status[key].update({
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "status": "rejected_focused_discriminator_and_paired_smoke_failed",
            "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
            "required_operator_action": "fresh_authority_only_after_new_architecture_contract",
            "result_binding": NO_GO.relative_to(ROOT).as_posix(),
            "result_binding_sha256": no_go_hash,
        })
    status["dashboard_fields"]["candidate_mechanism"].update({
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "status": "rejected_focused_discriminator_and_paired_smoke_failed",
        "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
        "required_operator_action": "fresh_authority_only_after_new_architecture_contract",
        "result_binding": NO_GO.relative_to(ROOT).as_posix(),
    })
    status["dashboard_fields"]["candidate_rtl_hash"] = rtl_hash
    status["dashboard_fields"]["operator_policy"] = policy
    status["dashboard_fields"]["latest_rtl_candidate"] = {
        "candidate_id": candidate_binding["candidate_id"],
        "evidence": NO_GO.relative_to(ROOT).as_posix(),
        "rtl_hash": rtl_hash,
        "rtl_hash_scope": manifest["candidate_rtl_hash_scope"],
        "stage_closing": False,
        "status": "rejected_focused_discriminator_and_paired_smoke_failed",
    }
    status["dashboard_fields"]["latest_quality_diagnostic"] = {
        "status": decision,
        "evidence": NO_GO.relative_to(ROOT).as_posix(),
        "paired_smoke_run": True,
        "paired_smoke_passed": False,
        "wikitext2_ratio": smoke_ratios["wikitext2"]["candidate"],
        "c4_en_512_ratio": smoke_ratios["c4_en_512"]["candidate"],
    }
    status["dashboard_fields"]["required_manager_action"] = (
        "reroute_rtl_to_architecture_for_structurally_distinct_contract"
    )
    status["dashboard_fields"]["required_operator_action"] = (
        "fresh_authority_only_after_new_architecture_contract"
    )
    status["dashboard_fields"]["current_architecture_performance_model"]["status"] = (
        "candidate_rejected_paired_smoke_no_shell_admission"
    )
    status["dashboard_fields"]["latest_ppa_frontier"]["decision"] = (
        "preserve_historical_hash_bound_ppa_after_projection_shadow_failed_pre_ppa_smoke_gate"
    )
    status["dashboard_fields"]["latest_ppa_frontier"]["status"] = (
        "historical_preserved_after_projection_shadow_failed_pre_ppa_gate"
    )
    status["dashboard_fields"]["latest_ppa_frontier"][
        "cycle_or_tokens_per_second_impact"
    ]["status"] = "not_remeasured_for_rejected_projection_shadow_candidate"
    status["dashboard_fields"]["latest_decision"] = decision
    status["dashboard_fields"]["routing_status"] = "rtl_candidate_rejected_manager_reroute_required"
    status["implementation_frontier"].update({
        "latest_decision": decision,
        "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
        "required_operator_action": "fresh_authority_only_after_new_architecture_contract",
        "routing_status": "rtl_candidate_rejected_manager_reroute_required",
        "rtl_contract_traceability": False,
        "latest_rtl_candidate": {
            "candidate_id": no_go["source_binding"]["candidate_id"],
            "evidence": NO_GO.relative_to(ROOT).as_posix(),
            "rtl_hash": rtl_hash,
            "rtl_hash_scope": manifest["candidate_rtl_hash_scope"],
            "stage_closing": False,
            "status": "rejected_focused_discriminator_and_paired_smoke_failed",
        },
        "latest_quality_diagnostic": {
            "status": decision,
            "evidence": NO_GO.relative_to(ROOT).as_posix(),
            "paired_smoke_run": True,
            "paired_smoke_passed": False,
            "wikitext2_paired_smoke_ratio": smoke_ratios["wikitext2"]["candidate"],
            "c4_en_512_paired_smoke_ratio": smoke_ratios["c4_en_512"]["candidate"],
            "wikitext2_score_relative_l2": comparisons["wikitext2"]["model.layers.0.score"]
            ["candidate_relative_l2"],
            "c4_en_512_score_relative_l2": comparisons["c4_en_512"]["model.layers.0.score"]
            ["candidate_relative_l2"],
        },
    })
    status["latest_decision"] = decision
    status["stage"]["current_stage_checklist"] = manifest["traceability"]["stage_checklist"]
    status["stage"]["current_stage_status"] = "candidate_rejected_manager_reroute_required"
    status["generated_at_utc"] = generated_at
    status["last_updated_utc"] = generated_at
    paths = {
        str(item.get("path"))
        for item in status.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path") and (ROOT / str(item["path"])).is_file()
    }
    paths.update(SOURCES + LOGS + [
        BASELINE.relative_to(ROOT).as_posix(),
        CANDIDATE.relative_to(ROOT).as_posix(),
        CANDIDATE_BINDING.relative_to(ROOT).as_posix(),
        NO_GO.relative_to(ROOT).as_posix(),
        SMOKE.relative_to(ROOT).as_posix(),
        (SMOKE_DIR / "run_contract.json").relative_to(ROOT).as_posix(),
        SMOKE_LOG.relative_to(ROOT).as_posix(),
        source_list.relative_to(ROOT).as_posix(),
        "tools/bind_projection_shadow_candidate.py",
        "tools/bind_projection_shadow_focused_no_go.py",
    ])
    status["artifact_hashes"] = [artifact(path) for path in sorted(paths)]
    status["integrity"] = {
        "canonicalization": (
            "UTF-8, sorted keys, two-space indentation, trailing newline, "
            "with integrity.canonical_sha256 set to null"
        ),
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    write("research/PUBLIC_STATUS.json", status)

    mission_path = ROOT / "MISSION.md"
    mission = mission_path.read_text(encoding="utf-8")
    old = (
        "softmax, and attention-value stages. Independent architecture and environment\n"
        "review completed and the operator authorized one bounded attempt. The focused\n"
        "full-capture discriminator then failed on both datasets: score relative-L2\n"
        "worsened from 0.2572662459 to 0.9687165038 on WikiText-2 and from\n"
        "0.2685861396 to 0.9831480397 on C4-en; attention-value also worsened on both.\n"
        "The direction is sealed before paired smoke, full shell, or PPA; its approval is\n"
        "consumed and `implementation_authorized=false`.\n"
    )
    new = (
        "softmax, and attention-value stages. Independent architecture and environment\n"
        "review completed and the operator authorized one bounded attempt. Focused score\n"
        "and attention-value quality worsened on both datasets, and the unique paired smoke\n"
        "also regressed: WikiText-2 `13549.939049967887 -> 13685.620430101628` and\n"
        "C4-en `4477.990517308544 -> 4622.743279550318`. The direction is sealed before\n"
        "full shell or PPA; its approval is consumed and `implementation_authorized=false`.\n"
    )
    if old in mission:
        mission_path.write_text(mission.replace(old, new), encoding="utf-8")
    else:
        require(new in mission, "MISSION authorization result paragraph changed")

    trace_path = ROOT / "design" / "RTL_TRACEABILITY.md"
    trace = trace_path.read_text(encoding="utf-8")
    old_trace = """

## Projection-shadow focused no-go

- Contract: `layer0_projection_shadow_staged_attention_v1`; source-list SHA-256 `683c9475f3cb3ac1cd5a7916203206fe2cfdea8a658431779942765e80e7e18f`.
- Independent scalar/tensor checks, standalone Icarus RTL, and warning-free
  Verilator lint pass.
- Full frozen-capture inputs match between baseline and candidate.
- WikiText-2 score relative-L2: `0.2572662459149663 -> 0.9687165037552808`;
  attention-value: `0.6339256877012315 -> 0.7882014449624917`.
- C4-en score relative-L2: `0.2685861395787881 -> 0.9831480397084055`;
  attention-value: `0.6143822670723456 -> 0.7550544485468513`.
- Candidate centered-score zero fractions exceed 90% on both datasets. The
  frozen focused prerequisite failed, so no paired smoke, shell admission, or
  PPA ran. `rtl.contract-traceability` remains false and Manager rerouting is
  required.
"""
    new_trace = f"""

## Projection-shadow paired-smoke no-go

- Contract: `{CONTRACT}`; candidate `{candidate_binding['candidate_id']}`;
  source-list SHA-256 `{rtl_hash}`.
- Independent scalar/tensor checks, full-model self-test, standalone Icarus RTL,
  and warning-free Verilator lint pass.
- Full frozen-capture score and attention-value relative-L2 worsen on both
  WikiText-2 and C4-en; centered-score zero fractions exceed 90% on both.
- The unique 128-token paired smoke regresses WikiText-2
  `{smoke_ratios['wikitext2']['required_strictly_below']} -> {smoke_ratios['wikitext2']['candidate']}`
  and C4-en `{smoke_ratios['c4_en_512']['required_strictly_below']} -> {smoke_ratios['c4_en_512']['candidate']}`.
- The candidate is not admitted to the shell. No full-shell regression, PPA,
  official evaluation, prototype, full benchmark, or signoff ran.
  `rtl.contract-traceability` remains false and Manager rerouting is required.
"""
    if old_trace in trace:
        trace_path.write_text(trace.replace(old_trace, new_trace), encoding="utf-8")
    else:
        require(new_trace in trace, "existing projection-shadow trace section changed")

    checkpoint = f"""# Goal

Execute the one operator-authorized bounded RTL attempt for `{CONTRACT}`.

# Result

The bounded implementation is complete and is a hash-bound no-go.
Independent scalar/tensor tests, full-model self-test, standalone RTL simulation,
and warning-free lint pass. Exact full-capture comparison worsens score and
attention-value quality on both datasets. The unique 128-token paired smoke also
regresses both mandatory ratios.

# Decisive Evidence

- No-go: `{NO_GO.relative_to(ROOT).as_posix()}`, SHA-256 `{no_go_hash}`.
- Candidate source-list SHA-256: `{rtl_hash}`.
- WikiText-2 score relative-L2: `0.2572662459149663 -> 0.9687165037552808`;
  attention-value: `0.6339256877012315 -> 0.7882014449624917`.
- C4-en score relative-L2: `0.2685861395787881 -> 0.9831480397084055`;
  attention-value: `0.6143822670723456 -> 0.7550544485468513`.
- Candidate centered-score zero fractions: `0.9040784192` WikiText-2 and
  `0.9179821551` C4-en.
- Paired smoke ratios: WikiText-2
  `{smoke_ratios['wikitext2']['required_strictly_below']} -> {smoke_ratios['wikitext2']['candidate']}`;
  C4-en `{smoke_ratios['c4_en_512']['required_strictly_below']} -> {smoke_ratios['c4_en_512']['candidate']}`.

# Current State

- Manager-owned stage remains `rtl`; Planner did not edit `PIPELINE_STATE.json`.
- Accepted prefix remains through `layer_0.v_proj`; first unsupported remains
  `layer_0.rope_q`; mode remains `ADVANCE`.
- Historical frontier remains 62,199 cells, 0.6108746272 mm2, and +0.1502 ns at
  100 MHz. No full-shell regression, PPA, official evaluation, prototype, full
  benchmark, or signoff ran.
- The exact operator approval is consumed. Manager must reroute to architecture
  for a structurally distinct successor and any implementation needs fresh
  operator authority.
"""
    (ROOT / "CHECKPOINT.md").write_text(checkpoint, encoding="utf-8")

    refresh_public_integrity()

    require(load("research/PIPELINE_STATE.json").get("current_stage") == "rtl",
            "pipeline stage changed unexpectedly")
    print(
        "ACE2_PROJECTION_SHADOW_PAIRED_NO_GO_BOUND "
        f"rtl_hash={rtl_hash} no_go_sha256={no_go_hash} paired_smoke_run=true"
    )


if __name__ == "__main__":
    main()
