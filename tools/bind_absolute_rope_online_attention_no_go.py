#!/usr/bin/env python3
"""Seal the approved absolute-RoPE attempt after its mandatory smoke failure."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "layer0_absolute_rope_online_attention_v1"
BASELINE = ROOT / "benchmark/raw/quality/operator-paired-smoke-20260731T081721Z/results.json"
CANDIDATE_DIR = ROOT / "benchmark/raw/quality/layer0-absolute-rope-online-attention-v1-smoke-128-20260801"
CANDIDATE = CANDIDATE_DIR / "results.json"
EVIDENCE_DIR = ROOT / "evidence" / CONTRACT / "latest"
CANDIDATE_EVIDENCE = EVIDENCE_DIR / "candidate_evidence.json"
NO_GO = EVIDENCE_DIR / "BOUNDED_NO_GO.json"
PUBLIC_STATUS = ROOT / "research/PUBLIC_STATUS.json"
LIVE_VIEW = ROOT / ".argus/live-view.json"
RTL_LINT_LOG = ROOT / "evidence/frontier/latest/rtl_lint.log"
RTL_ELABORATION_LOG = ROOT / "evidence/frontier/latest/rtl_elaboration.log"
NO_GO_RELATIVE = f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json"
DECISION = "absolute_rope_online_attention_bounded_no_go_both_smokes_regressed"
MANAGER_ACTION = "reroute_rtl_to_architecture_for_structurally_distinct_contract"
OPERATOR_ACTION = "fresh_authority_only_after_new_architecture_contract"
MANIFEST_ARCHITECTURE_STATUS = (
    "layer0_absolute_rope_online_attention_v1_implementation_completed_"
    "rejected_bounded_no_go_manager_reroute_required"
)
ABSOLUTE_ROPE_MODULES = (
    "ace2_absolute_rope_score_core",
    "ace2_absolute_rope_online_attention_core",
    "ace2_absolute_rope_online_attention_finalize_core",
)
ABSOLUTE_ROPE_PROVENANCE_NAMES = (
    "ace2_absolute_rope_score_core",
    "ace2_absolute_rope_online_attention_core_and_finalizer",
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_path(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load(relative: str) -> dict[str, Any]:
    return load_path(ROOT / relative)


def write(relative: str, value: dict[str, Any]) -> None:
    (ROOT / relative).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    canonical = json.loads(json.dumps(value))
    canonical.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(canonical, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def upsert_public_artifact(status: dict[str, Any], path: Path) -> None:
    relative = path.relative_to(ROOT).as_posix()
    items = [
        item
        for item in status.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path") != relative
    ]
    item = artifact(path)
    item["exists"] = True
    items.append(item)
    status["artifact_hashes"] = sorted(items, key=lambda entry: str(entry.get("path", "")))


def refresh_public_artifacts(status: dict[str, Any]) -> None:
    refreshed: list[dict[str, Any]] = []
    public_relative = PUBLIC_STATUS.relative_to(ROOT).as_posix()
    for prior in status.get("artifact_hashes", []):
        if not isinstance(prior, dict) or not isinstance(prior.get("path"), str):
            continue
        relative = prior["path"]
        if relative == public_relative:
            continue
        path = ROOT / relative
        item = dict(prior)
        item["exists"] = path.is_file()
        if path.is_file():
            item["bytes"] = path.stat().st_size
            item["sha256"] = sha256(path)
        refreshed.append(item)
    status["artifact_hashes"] = sorted(
        refreshed, key=lambda entry: str(entry.get("path", ""))
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def repair_contract_routing_metadata() -> None:
    """Normalize post-no-go routing without changing the Manager-owned stage."""

    no_go = load(NO_GO_RELATIVE)
    no_go_hash = sha256(NO_GO)
    sealed_at = str(no_go["sealed_at_utc"])

    policy = load("design/FAST_LOOP_POLICY.json")
    active = policy["active_repair_authorization"]
    active.update({
        "implementation_authorized": False,
        "execution_status": "completed_bounded_no_go_both_smokes_regressed",
        "status": "authorization_consumed_rejected_bounded_no_go",
        "operator_approval_consumed": True,
        "smoke_gate_passed": False,
        "result_binding": NO_GO_RELATIVE,
        "result_binding_sha256": no_go_hash,
        "required_manager_action": MANAGER_ACTION,
        "required_operator_action": "fresh_authority_required_for_any_new_contract",
    })
    proposal = policy["architecture_proposal_authorization"]
    proposal.update({
        "implementation_authorized": False,
        "status": "implementation_exhausted_bounded_no_go",
        "operator_approval_consumed": True,
        "result_binding": NO_GO_RELATIVE,
        "required_manager_action": MANAGER_ACTION,
        "required_operator_action": OPERATOR_ACTION,
    })
    selected = policy["selected_replacement_contract"]
    selected.update({
        "implementation_authorized": False,
        "status": "rejected_bounded_no_go_both_smokes_regressed",
        "required_manager_action": MANAGER_ACTION,
        "required_operator_action": OPERATOR_ACTION,
        "updated_at_utc": sealed_at,
    })
    write("design/FAST_LOOP_POLICY.json", policy)

    target = load("design/TARGET.json")
    target["generated_at_utc"] = sealed_at
    target["fast_loop_contract"]["manager_recommendation"] = (
        "reroute_rtl_to_architecture_after_absolute_rope_bounded_no_go"
    )
    target_active = target["fast_loop_contract"]["active_repair_authorization"]
    target_active.update({
        "implementation_authorized": False,
        "status": "authorization_consumed_rejected_bounded_no_go",
        "result_binding": NO_GO_RELATIVE,
        "required_manager_action": MANAGER_ACTION,
        "required_operator_action": "fresh_authority_required_for_any_new_contract",
    })
    target_proposal = target["fast_loop_contract"]["architecture_proposal_authorization"]
    target_proposal.update({
        "implementation_authorized": False,
        "status": "authorization_consumed_rejected_bounded_no_go",
        "result_binding": NO_GO_RELATIVE,
        "required_manager_action": MANAGER_ACTION,
        "required_operator_action": OPERATOR_ACTION,
    })
    write("design/TARGET.json", target)

    manifest = load("design/RTL_MANIFEST.json")
    review = selected.get("architecture_review", active.get("architecture_acceptance", {}))
    approval = selected.get("implementation_approval", active.get("implementation_approval", {}))
    manifest.update({
        "architecture_contract_status": MANIFEST_ARCHITECTURE_STATUS,
        "architecture_selection_evidence": {
            "contract_id": CONTRACT,
            "implementation_authorized": False,
            "implementation_approved_at_utc": approval.get("approved_at_utc"),
            "proposal": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
            "proposal_sha256": selected.get("proposal_sha256"),
            "review": review.get("evidence"),
            "review_sha256": review.get("evidence_sha256"),
            "reviewed_at_utc": review.get("reviewed_at_utc"),
            "status": "implementation_completed_rejected_by_both_dataset_paired_smoke_gate",
            "result_binding": NO_GO_RELATIVE,
            "result_binding_sha256": no_go_hash,
        },
        "candidate_status": "rejected_bounded_no_go_both_smokes_regressed",
        "generated_at_utc": sealed_at,
        "interfaces_contract_status": (
            "accepted_shell_prefix_through_v_proj_absolute_rope_candidate_"
            "rejected_before_shell_admission"
        ),
    })
    manifest["interfaces"].update({
        "command_dispatch": (
            "existing direct descriptor ingress for the accepted prefix through v_proj; "
            "no layer_0 RoPE command is admitted"
        ),
        "qk_weight_contract": (
            "no accepted layer_0 RoPE Q/K transport exists beyond v_proj; all recorded "
            "RoPE successors are rejected standalone diagnostics"
        ),
    })
    manifest["proposed_replacement_contract"].update({
        "implementation_authorized": False,
        "status": "rejected_bounded_no_go_both_smokes_regressed",
        "required_manager_action": MANAGER_ACTION,
        "required_operator_action": OPERATOR_ACTION,
    })
    traced = manifest["traceability"].setdefault("rejected_diagnostic_sources_traced", [])
    manifest["traceability"]["rejected_diagnostic_sources_traced"] = list(
        dict.fromkeys([*traced, *ABSOLUTE_ROPE_MODULES])
    )
    manifest["traceability"].update({
        "architecture_contract_gap": {
            "status": "candidate_rejected_quality_gate_manager_reroute_required",
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
    if RTL_LINT_LOG.is_file() and RTL_ELABORATION_LOG.is_file():
        lint_text = RTL_LINT_LOG.read_text(encoding="utf-8")
        elaboration_text = RTL_ELABORATION_LOG.read_text(encoding="utf-8")
        require("ACE2_RTL_LINT_PASS" in lint_text, "accepted-shell lint pass token missing")
        require(
            "ACE2_RTL_ELABORATION_PASS" in elaboration_text,
            "accepted-shell elaboration pass token missing",
        )
        rechecked_at = datetime.fromtimestamp(
            max(RTL_LINT_LOG.stat().st_mtime, RTL_ELABORATION_LOG.stat().st_mtime), UTC
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        manifest["accepted_shell_rtl_recheck"] = {
            "classification": "current_rtl_stage_lint_and_elaboration_only",
            "rechecked_at_utc": rechecked_at,
            "rtl_lint": artifact(RTL_LINT_LOG),
            "rtl_elaboration": artifact(RTL_ELABORATION_LOG),
            "verilator_warning_count": sum(
                1 for line in lint_text.splitlines() if line.startswith("%Warning-")
            ),
            "warning_explanation": "design/RTL_TRACEABILITY.md",
            "candidate_rerun": False,
            "full_shell_regression_run": False,
            "ppa_run": False,
            "status": "pass",
        }
    for item in manifest.get("ip_provenance", []):
        if item.get("name") in ABSOLUTE_ROPE_PROVENANCE_NAMES:
            item["kind"] = "first_party_rejected_diagnostic_rtl"
            item["source_revision"] = item.get("sha256", item.get("source_revision"))
        elif item.get("name") == "ace2_exp_q31_lut":
            item["kind"] = "generated_rejected_diagnostic_rtl_include"
    for item in manifest.get("file_hashes", []):
        relative = item.get("path")
        if isinstance(relative, str) and (ROOT / relative).is_file():
            item["sha256"] = sha256(ROOT / relative)
    write("design/RTL_MANIFEST.json", manifest)


def repair_public_dashboard() -> None:
    status = load("research/PUBLIC_STATUS.json")
    no_go = load(f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json")
    frontier_record = no_go["frontier_preserved"]
    prefix = frontier_record["ordered_supported_layer_operator_prefix"]
    first_unsupported = frontier_record["first_unsupported_layer_operator"]
    mode = frontier_record["mode"]
    decision = DECISION

    status["current_mode"] = mode
    status["supported_layer_operator_prefix"] = prefix
    status["first_unsupported_layer_operator"] = first_unsupported
    status["latest_decision"] = decision

    dashboard = status["dashboard_fields"]
    dashboard["current_mode"] = mode
    dashboard["supported_layer_operator_prefix"] = prefix
    dashboard["ordered_supported_layer_operator_prefix"] = prefix
    dashboard["first_unsupported_layer_operator"] = first_unsupported
    dashboard["latest_decision"] = decision
    dashboard["routing_status"] = "rtl_bounded_no_go_manager_reroute_required"
    dashboard["required_manager_action"] = MANAGER_ACTION
    dashboard["required_operator_action"] = OPERATOR_ACTION

    policy = load("design/FAST_LOOP_POLICY.json")
    dashboard["operator_policy"]["active_repair_authorization"] = policy[
        "active_repair_authorization"
    ]
    dashboard["operator_policy"]["selected_replacement_contract"] = policy[
        "selected_replacement_contract"
    ]

    selected_status = status["selected_replacement_contract"]
    selected_status.update({
        "implementation_authorized": False,
        "status": "bounded_no_go_both_smokes_regressed",
        "required_manager_action": MANAGER_ACTION,
        "required_operator_action": OPERATOR_ACTION,
        "result_binding": NO_GO_RELATIVE,
    })
    for key in ("architecture_proposal_gate",):
        if isinstance(status.get(key), dict):
            status[key].update({
                "implementation_authorized": False,
                "status": "bounded_no_go_both_smokes_regressed",
                "required_manager_action": MANAGER_ACTION,
                "required_operator_action": OPERATOR_ACTION,
                "result_binding": NO_GO_RELATIVE,
            })

    implementation = status["implementation_frontier"]
    implementation["latest_decision"] = decision
    implementation["ordered_supported_layer_operator_prefix"] = prefix
    implementation["first_unsupported_layer_operator"] = first_unsupported
    implementation["mode"] = mode
    implementation["routing_status"] = "rtl_bounded_no_go_manager_reroute_required"
    implementation["required_manager_action"] = MANAGER_ACTION
    implementation["required_operator_action"] = OPERATOR_ACTION
    implementation["rtl_contract_traceability"] = False
    latest_quality = {
        "status": decision,
        "evidence": f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json",
        "wikitext2_ratio": no_go["ratios"]["wikitext2"]["candidate"],
        "c4_en_512_ratio": no_go["ratios"]["c4_en_512"]["candidate"],
    }
    implementation["latest_quality_diagnostic"] = latest_quality
    implementation["latest_rtl_candidate"].update({
        "evidence": f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json",
        "status": "rejected_bounded_no_go",
    })
    dashboard["latest_quality_diagnostic"] = dict(latest_quality)
    dashboard["latest_rtl_candidate"] = dict(implementation["latest_rtl_candidate"])
    for relative in (
        "design/RTL_MANIFEST.json",
        "design/RTL_TRACEABILITY.md",
        "evidence/frontier/latest/rtl_lint.log",
        "evidence/frontier/latest/rtl_elaboration.log",
    ):
        if (ROOT / relative).is_file() and relative not in status["stage"]["current_stage_evidence"]:
            status["stage"]["current_stage_evidence"].append(relative)

    stamp = utc_now()
    status["generated_at_utc"] = stamp
    status["last_updated_utc"] = stamp
    write(
        ".argus/live-view.json",
        {
            "version": 1,
            "title": "RTL bounded no-go: architecture reroute required",
            "paths": [
                "research/PIPELINE_STATE.json",
                f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json",
                "design/RTL_MANIFEST.json",
                "design/RTL_TRACEABILITY.md",
                "design/FAST_LOOP_POLICY.json",
                "research/PUBLIC_STATUS.json",
                "CHECKPOINT.md",
            ],
            "reason": (
                "Shows the hash-bound absolute-RoPE online-attention failure on both "
                "mandatory datasets, consumed operator approval, preserved accepted "
                "prefix and PPA frontier, false RTL traceability gate, and required "
                "Manager reroute to architecture before any structurally distinct RTL."
            ),
        },
    )
    upsert_public_artifact(status, LIVE_VIEW)
    upsert_public_artifact(status, NO_GO)
    upsert_public_artifact(status, Path(__file__).resolve())
    refresh_public_artifacts(status)
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonical_sha256": None,
        "canonicalization": (
            "UTF-8, sorted keys, two-space indentation, trailing newline, with "
            "integrity.canonical_sha256 set to null"
        ),
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    write("research/PUBLIC_STATUS.json", status)


def validate_post_no_go_routing() -> None:
    no_go_hash = sha256(NO_GO)
    pipeline = load("research/PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage changed from rtl")

    policy = load("design/FAST_LOOP_POLICY.json")
    manifest = load("design/RTL_MANIFEST.json")
    target = load("design/TARGET.json")
    status = load("research/PUBLIC_STATUS.json")

    require(
        manifest.get("architecture_contract_status") == MANIFEST_ARCHITECTURE_STATUS,
        "RTL manifest architecture status is stale",
    )
    selection = manifest.get("architecture_selection_evidence", {})
    require(selection.get("contract_id") == CONTRACT, "RTL manifest selects stale contract")
    require(selection.get("result_binding_sha256") == no_go_hash, "manifest no-go hash mismatch")
    require(manifest.get("candidate_status") == "rejected_bounded_no_go_both_smokes_regressed", "candidate status mismatch")
    checklist = manifest.get("traceability", {}).get("stage_checklist", {})
    require(checklist == {
        "rtl.contract-traceability": False,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
    }, "RTL checklist no longer matches the bounded no-go")
    traced = set(manifest.get("traceability", {}).get("rejected_diagnostic_sources_traced", []))
    require(set(ABSOLUTE_ROPE_MODULES).issubset(traced), "absolute-RoPE modules are not traced as rejected diagnostics")
    provenance = {item.get("name"): item for item in manifest.get("ip_provenance", [])}
    for name in ABSOLUTE_ROPE_PROVENANCE_NAMES:
        require(provenance.get(name, {}).get("kind") == "first_party_rejected_diagnostic_rtl", f"{name} provenance is not rejected-diagnostic RTL")
    require(provenance.get("ace2_exp_q31_lut", {}).get("kind") == "generated_rejected_diagnostic_rtl_include", "absolute-RoPE LUT provenance is stale")
    shell_recheck = manifest.get("accepted_shell_rtl_recheck", {})
    require(shell_recheck.get("status") == "pass", "accepted-shell RTL recheck is missing")
    require(shell_recheck.get("verilator_warning_count") == 4, "accepted-shell lint warning count changed")
    require(shell_recheck.get("candidate_rerun") is False, "rejected candidate was marked rerun")
    for key in ("rtl_lint", "rtl_elaboration"):
        binding = shell_recheck.get(key, {})
        relative = binding.get("path")
        require(isinstance(relative, str) and (ROOT / relative).is_file(), f"missing {key} evidence")
        require(sha256(ROOT / relative) == binding.get("sha256"), f"{key} evidence hash mismatch")

    for node in (
        policy["active_repair_authorization"],
        policy["architecture_proposal_authorization"],
        policy["selected_replacement_contract"],
        target["fast_loop_contract"]["active_repair_authorization"],
        target["fast_loop_contract"]["architecture_proposal_authorization"],
        status["dashboard_fields"]["operator_policy"]["active_repair_authorization"],
        status["dashboard_fields"]["operator_policy"]["selected_replacement_contract"],
        status["selected_replacement_contract"],
        status["implementation_frontier"],
    ):
        require(node.get("implementation_authorized") is not True, "post-no-go node still authorizes implementation")
        require(node.get("required_manager_action") == MANAGER_ACTION, "post-no-go Manager routing is inconsistent")
    require(
        status["dashboard_fields"].get("required_operator_action") == OPERATOR_ACTION,
        "dashboard operator routing is stale",
    )
    require(
        status["stage"]["current_stage_checklist"] == checklist,
        "public RTL checklist differs from manifest",
    )
    require(
        status.get("supported_layer_operator_prefix")
        == manifest.get("supported_layer_operator_prefix"),
        "public supported prefix differs from manifest",
    )
    require(
        status.get("first_unsupported_layer_operator")
        == manifest.get("first_unsupported_layer_operator"),
        "public first unsupported operator differs from manifest",
    )

    for relative, expected in manifest.get("candidate_source_hashes", {}).items():
        path = ROOT / relative
        require(path.is_file(), f"missing candidate source: {relative}")
        require(sha256(path) == expected, f"candidate source hash drift: {relative}")


def main() -> None:
    if sys.argv[1:] == ["--repair-status-only"]:
        repair_contract_routing_metadata()
        repair_public_dashboard()
        validate_post_no_go_routing()
        print(
            "ACE2_ABSOLUTE_ROPE_STATUS_REPAIR_PASS "
            f"public_sha256={sha256(PUBLIC_STATUS)} live_view_sha256={sha256(LIVE_VIEW)}"
        )
        return
    if sys.argv[1:] == ["--check-routing"]:
        validate_post_no_go_routing()
        print(
            "ACE2_ABSOLUTE_ROPE_ROUTING_CHECK_PASS "
            f"manifest_sha256={sha256(ROOT / 'design/RTL_MANIFEST.json')} "
            f"public_sha256={sha256(PUBLIC_STATUS)} no_go_sha256={sha256(NO_GO)}"
        )
        return
    if sys.argv[1:]:
        raise RuntimeError(f"unsupported arguments: {sys.argv[1:]}")
    if load("research/PIPELINE_STATE.json").get("current_stage") != "rtl":
        raise RuntimeError("no-go binder requires Manager-owned rtl stage")
    baseline = load_path(BASELINE)
    candidate = load_path(CANDIDATE)
    packet = load_path(CANDIDATE_EVIDENCE)
    source_list = ROOT / packet["source_binding"]["source_hash_list"]
    if sha256(source_list) != packet["source_binding"]["ordered_source_hash_list_sha256"]:
        raise RuntimeError("candidate source-list hash changed")
    for line in source_list.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or sha256(ROOT / relative) != expected:
            raise RuntimeError(f"candidate source changed: {relative}")
    if sha256(ROOT / "design/RTL_MANIFEST.json") != packet[
        "accepted_frontier_preservation"
    ]["rtl_manifest_sha256"]:
        raise RuntimeError("candidate manifest changed before no-go binding")

    ratios: dict[str, dict[str, float | bool]] = {}
    for dataset in ("wikitext2", "c4_en_512"):
        old = float(baseline["metrics"][dataset]["ratio"])
        new = float(candidate["metrics"][dataset]["ratio"])
        ratios[dataset] = {
            "baseline": old,
            "candidate": new,
            "strictly_improved": new < old,
            "delta": new - old,
        }
    if any(item["strictly_improved"] for item in ratios.values()):
        raise RuntimeError("expected both mandatory paired-smoke datasets to fail")

    smoke_artifacts = {
        path.name: artifact(path) for path in sorted(CANDIDATE_DIR.iterdir()) if path.is_file()
    }
    manifest_pre_no_go = artifact(ROOT / "design/RTL_MANIFEST.json")
    no_go = {
        "schema_version": 1,
        "project": "ACE-2",
        "contract_id": CONTRACT,
        "status": "bounded_no_go_both_mandatory_paired_smokes_regressed",
        "sealed_at_utc": utc_now(),
        "stage_closing": False,
        "operator_approval_consumed": True,
        "decision": {
            "candidate_capability_accepted": False,
            "reason": "both mandatory comparable paired-smoke ratios regressed",
            "required_action": "stop_direction_immediately_and_request_manager_reroute_to_architecture_for_a_structurally_distinct_contract",
        },
        "ratios": ratios,
        "baseline": artifact(BASELINE),
        "candidate_smoke": artifact(CANDIDATE),
        "candidate_run_contract": artifact(CANDIDATE_DIR / "run_contract.json"),
        "candidate_packet": artifact(CANDIDATE_EVIDENCE),
        "candidate_source_hash_list": artifact(source_list),
        "candidate_manifest_before_no_go": manifest_pre_no_go,
        "focused_verification": packet["focused_verification"],
        "smoke_artifacts": smoke_artifacts,
        "prohibited_runs": {
            "full_shell_regression_run": False,
            "canonical_or_noncanonical_ppa_run": False,
            "official_14_item_evaluation_run": False,
            "prototype_or_signoff_run": False,
        },
        "frontier_preserved": {
            "ordered_supported_layer_operator_prefix": [
                "layer_0.input_rmsnorm",
                "layer_0.q_proj",
                "layer_0.k_proj",
                "layer_0.v_proj",
            ],
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "mode": "ADVANCE",
            "cells": 62199,
            "non_sram_area_mm2": 0.6108746272,
            "setup_slack_ns_at_100mhz": 0.1502,
            "frequency_mhz": 100.0,
            "area_cap_non_sram_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "frontier_replaced": False,
        },
        "claim_boundary": "This is a bounded negative quality result, not RTL stage closure, PPA, benchmark acceptance, signoff, tapeout, or silicon evidence.",
    }
    write(f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json", no_go)
    no_go_hash = sha256(NO_GO)

    policy = load("design/FAST_LOOP_POLICY.json")
    active = policy["active_repair_authorization"]
    active.update({
        "implementation_authorized": False,
        "execution_status": "completed_bounded_no_go_both_smokes_regressed",
        "status": "authorization_consumed_rejected_bounded_no_go",
        "operator_approval_consumed": True,
        "smoke_gate_passed": False,
        "observed_smoke_ratios": ratios,
        "result_binding": f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json",
        "result_binding_sha256": no_go_hash,
        "full_shell_regression_run": False,
        "canonical_sky130_ppa_run": False,
        "official_14_item_evaluation_run": False,
        "required_operator_action": "fresh_authority_required_for_any_new_contract",
        "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
    })
    proposal = policy["architecture_proposal_authorization"]
    proposal.update({
        "implementation_authorized": False,
        "status": "implementation_exhausted_bounded_no_go",
        "operator_approval_consumed": True,
        "result_binding": f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json",
        "required_operator_action": "none_for_rejected_contract",
        "required_manager_action": "reroute_to_architecture",
    })
    selected = policy["selected_replacement_contract"]
    selected["implementation_authorized"] = False
    selected["status"] = "rejected_bounded_no_go_both_smokes_regressed"
    selected["execution_result"] = {
        "rtl_change_started": True,
        "focused_software_rtl_tests_run": True,
        "focused_software_rtl_tests_passed": True,
        "paired_smoke_run": True,
        "paired_smoke_passed": False,
        "full_shell_regression_run": False,
        "canonical_sky130_ppa_run": False,
        "official_14_item_evaluation_run": False,
        "result_binding": f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json",
    }
    write("design/FAST_LOOP_POLICY.json", policy)

    target = load("design/TARGET.json")
    target["generated_at_utc"] = utc_now()
    target["fast_loop_contract"]["manager_recommendation"] = (
        "reroute_rtl_to_architecture_after_absolute_rope_bounded_no_go"
    )
    for name in ("active_repair_authorization", "architecture_proposal_authorization"):
        target["fast_loop_contract"][name].update({
            "implementation_authorized": False,
            "status": "authorization_consumed_rejected_bounded_no_go",
            "result_binding": f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json",
            "required_operator_action": "none_for_rejected_contract",
            "required_manager_action": "reroute_to_architecture",
        })
    write("design/TARGET.json", target)

    chip = load("design/CHIP_SCOPE.json")
    chip["authority_override"].update({
        "implementation_authorized": False,
        "required_next_action": "manager_reroute_to_architecture_for_structurally_distinct_contract",
        "bounded_no_go": f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json",
    })
    chip["numerical_behavior"]["rope"]["absolute_rope_online_attention_contract"].update({
        "implementation_authorized": False,
        "status": "bounded_no_go_both_smokes_regressed",
    })
    chip["operator_owned_execution_policy"]["sixth_mechanism_architecture_proposal"].update({
        "implementation_authorized": False,
        "status": "implementation_exhausted_bounded_no_go",
        "observed_result": "both_mandatory_paired_smokes_regressed",
        "required_manager_action": "reroute_to_architecture",
    })
    chip["stage"]["current_stage_status"] = "bounded_no_go_manager_reroute_required"
    write("design/CHIP_SCOPE.json", chip)

    memory = load("design/MEMORY_MODEL.json")
    memory["authority_override"].update({
        "implementation_authorized": False,
        "required_next_action": "manager_reroute_to_architecture_for_structurally_distinct_contract",
        "bounded_no_go": f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json",
    })
    memory["claim_status"] = "candidate_rejected_before_shell_admission_or_ppa"
    write("design/MEMORY_MODEL.json", memory)

    oracle = load("reference/ORACLE_MANIFEST.json")
    oracle["proposed_architecture_contract"].update({
        "implementation_authorized": False,
        "status": "focused_vectors_passed_but_paired_smoke_bounded_no_go",
        "result_binding": f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json",
    })
    oracle["claim_boundary"] = (
        "The exact oracle and focused vectors passed, but both mandatory paired smokes regressed; "
        "the contract is rejected and authorizes no further RTL, shell, PPA, or official evaluation."
    )
    write("reference/ORACLE_MANIFEST.json", oracle)

    manifest = load("design/RTL_MANIFEST.json")
    manifest.update({
        "candidate_status": "rejected_bounded_no_go_both_smokes_regressed",
        "candidate_meets_numeric_acceptance": False,
        "candidate_requires_fresh_sky130_ppa": False,
        "candidate_review_binding": {
            "candidate_capability_accepted": False,
            "decision": "bounded_no_go",
            "evidence": f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json",
            "evidence_sha256": no_go_hash,
            "scope": "mandatory_paired_smoke_quality_stop_rule",
            "stage_closing": False,
        },
        "candidate_evidence_hashes": {
            "baseline_smoke_sha256": sha256(BASELINE),
            "candidate_smoke_sha256": sha256(CANDIDATE),
            "candidate_evidence_sha256": sha256(CANDIDATE_EVIDENCE),
            "candidate_source_hash_list_sha256": sha256(source_list),
            "bounded_no_go_sha256": no_go_hash,
        },
        "generated_at_utc": utc_now(),
    })
    manifest["proposed_replacement_contract"].update({
        "implementation_authorized": False,
        "status": "rejected_bounded_no_go_both_smokes_regressed",
        "implementation_result": {
            "rtl_change_started": True,
            "focused_software_rtl_tests_run": True,
            "focused_software_rtl_tests_passed": True,
            "paired_smoke_run": True,
            "paired_smoke_passed": False,
            "accepted_rtl_change": False,
            "accepted_verification_evidence": False,
            "accepted_ppa_evidence": False,
            "result_binding": f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json",
        },
    })
    manifest["traceability"].update({
        "architecture_contract_gap": {
            "status": "candidate_rejected_quality_gate_manager_reroute_required",
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
    manifest["claim_boundaries"] = list(dict.fromkeys(
        manifest["claim_boundaries"] + [
            "The absolute-RoPE candidate is rejected; its standalone RTL is not an accepted shell capability.",
            "No full-shell, PPA, official-14, prototype, benchmark-release, or signoff run followed the failed smoke gate.",
        ]
    ))
    write("design/RTL_MANIFEST.json", manifest)

    status = load("research/PUBLIC_STATUS.json")
    for path in (
        status["architecture_proposal_gate"],
        status["selected_replacement_contract"],
        status["dashboard_fields"]["candidate_mechanism"],
        status["dashboard_fields"]["current_architecture_performance_model"],
    ):
        path["implementation_authorized"] = False
        path["status"] = "bounded_no_go_both_smokes_regressed"
    status["dashboard_fields"]["operator_policy"]["active_repair_authorization"] = active
    status["dashboard_fields"]["operator_policy"]["selected_replacement_contract"] = selected
    status["dashboard_fields"]["latest_decision"] = (
        "absolute_rope_online_attention_bounded_no_go_both_smokes_regressed"
    )
    status["dashboard_fields"]["routing_status"] = "rtl_bounded_no_go_manager_reroute_required"
    frontier = status["implementation_frontier"]
    frontier.update({
        "latest_decision": "absolute_rope_online_attention_bounded_no_go_both_smokes_regressed",
        "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
        "required_operator_action": "fresh_approval_only_after_new_architecture_contract",
        "routing_status": "rtl_bounded_no_go_manager_reroute_required",
        "rtl_contract_traceability": False,
        "latest_quality_diagnostic": {
            "status": "absolute_rope_online_attention_bounded_no_go_both_smokes_regressed",
            "evidence": f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json",
            "wikitext2_ratio": ratios["wikitext2"]["candidate"],
            "c4_en_512_ratio": ratios["c4_en_512"]["candidate"],
        },
    })
    frontier["latest_rtl_candidate"]["status"] = "rejected_bounded_no_go"
    frontier["latest_rtl_candidate"]["evidence"] = f"evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json"
    status["latest_decision"] = "absolute_rope_online_attention_bounded_no_go_both_smokes_regressed"
    status["stage"]["current_stage_checklist"] = manifest["traceability"]["stage_checklist"]
    status["stage"]["current_stage_status"] = "bounded_no_go_manager_reroute_required"
    write("research/PUBLIC_STATUS.json", status)

    trace = (ROOT / "design/RTL_TRACEABILITY.md").read_text(encoding="utf-8")
    trace += f"""

## Bounded no-go

The unique paired smoke rejected `{CONTRACT}`. WikiText-2 regressed
`{ratios['wikitext2']['baseline']} -> {ratios['wikitext2']['candidate']}` and
C4-en regressed `{ratios['c4_en_512']['baseline']} -> {ratios['c4_en_512']['candidate']}`.
The operator approval is consumed. The standalone arithmetic remains recorded
as rejected diagnostic RTL; it is not admitted to `ace2_shell`, the supported
prefix remains through `layer_0.v_proj`, and no full-shell or PPA run occurred.
"""
    (ROOT / "design/RTL_TRACEABILITY.md").write_text(trace, encoding="utf-8")

    checkpoint = f"""# Goal

Execute the single operator-approved `{CONTRACT}` attempt and enforce its
mandatory two-dataset stop rule.

# Result

The bounded attempt is complete and rejected. Focused software and standalone
RTL checks passed, but the unique paired smoke regressed both datasets:

- WikiText-2: `{ratios['wikitext2']['baseline']} -> {ratios['wikitext2']['candidate']}`.
- C4-en: `{ratios['c4_en_512']['baseline']} -> {ratios['c4_en_512']['candidate']}`.

The hash-bound no-go is `evidence/{CONTRACT}/latest/BOUNDED_NO_GO.json`
(SHA-256 `{no_go_hash}`). The exact operator approval is consumed. No full-shell
regression, PPA, official-14 evaluation, prototype, or signoff run followed.

# Current State

The Manager-owned stage remains `rtl`; Planner did not edit
`research/PIPELINE_STATE.json`. The accepted prefix remains through
`layer_0.v_proj`, first unsupported remains `layer_0.rope_q`, mode remains
`ADVANCE`, and the historical accepted 62,199-cell / 0.6108746272 mm2 /
+0.1502 ns at 100 MHz frontier is unchanged. Manager rerouting to architecture
is required before a structurally distinct seventh contract can be proposed.
"""
    (ROOT / "CHECKPOINT.md").write_text(checkpoint, encoding="utf-8")
    repair_contract_routing_metadata()
    repair_public_dashboard()
    validate_post_no_go_routing()
    print(
        "ACE2_ABSOLUTE_ROPE_BOUNDED_NO_GO_BOUND "
        f"wikitext2={ratios['wikitext2']['candidate']} "
        f"c4={ratios['c4_en_512']['candidate']} no_go_sha256={no_go_hash}"
    )


if __name__ == "__main__":
    main()
