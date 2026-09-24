#!/usr/bin/env python3
"""Bind the architecture-level metadata blocker found before focused quality."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_qk_residual_cross_term_attention_v1"
PACKET = "evidence/shared_qk_residual_cross_term_attention_v1/latest/PRECHECK.json"
REVIEW = "evidence/review/rtl_checklist_shared_qk_residual_cross_term_attention_v1/decision.json"
BLOCKER = "evidence/shared_qk_residual_cross_term_attention_v1/latest/QUALITY_GATE_BLOCKER.json"
BINDER = "tools/bind_qk_residual_cross_term_quality_blocker.py"
PREFIX = ["layer_0.input_rmsnorm", "layer_0.q_proj", "layer_0.k_proj", "layer_0.v_proj"]
FIRST_UNSUPPORTED = "layer_0.rope_q"
STATUS = "quality_gate_blocked_architecture_metadata_derivation_missing"
DECISION = "residual_cross_term_quality_gate_blocked_architecture_contract_gap"
MANAGER_ACTION = "rollback_rtl_to_architecture_refreeze_residual_scale_metadata_derivation"
CHECKLIST = {
    "rtl.contract-traceability": True,
    "rtl.hardware-discipline": True,
    "rtl.ip-provenance": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: str) -> dict[str, Any]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def dump(path: str, value: dict[str, Any]) -> None:
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: str) -> dict[str, Any]:
    item = ROOT / path
    return {"path": path, "bytes": item.stat().st_size, "sha256": sha256(path)}


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    return hashlib.sha256((json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def line_of(path: str, phrase: str) -> int:
    for index, line in enumerate((ROOT / path).read_text(encoding="utf-8").splitlines(), 1):
        if phrase in line:
            return index
    raise RuntimeError(f"missing expected phrase in {path}: {phrase}")


def candidate_auth(value: dict[str, Any], candidate_hash: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    require(result.get("contract_id") == CONTRACT, "live contract mismatch")
    result.update({
        "candidate_rtl_hash": candidate_hash,
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "rtl_started": True,
        "status": STATUS,
        "required_manager_action": MANAGER_ACTION,
        "blocker_binding": BLOCKER,
    })
    return result


def audit() -> tuple[dict[str, Any], dict[str, Any], str, list[dict[str, Any]]]:
    require(load("research/PIPELINE_STATE.json").get("current_stage") == "rtl",
            "Manager-owned current stage is not rtl")
    packet = load(PACKET)
    review = load(REVIEW)
    candidate_hash = str(packet.get("candidate_rtl_hash") or "")
    require(packet.get("contract_id") == CONTRACT, "candidate preflight contract mismatch")
    require(review.get("reviewer_status") == "done", "independent RTL review is not done")
    require(review.get("candidate_rtl_hash") == candidate_hash, "review candidate hash mismatch")
    require(review.get("checklist") == CHECKLIST, "review checklist is incomplete")
    for path, digest in packet.get("source_hashes", {}).items():
        require(sha256(path) == digest, f"candidate source changed: {path}")

    proposal = (ROOT / "design/NUMERICAL_REPLACEMENT_PROPOSAL.md").read_text(encoding="utf-8")
    spec = (ROOT / "design/SPEC.md").read_text(encoding="utf-8")
    full_model = (ROOT / "tools/ace2_full_model_fixed_point.py").read_text(encoding="utf-8")
    quality = load("benchmark/quality/QUALITY_CONFIG.json")
    manifest = load("design/RTL_MANIFEST.json")

    require("calibrated signed" in proposal and "residual sidecar scale" in proposal,
            "proposal no longer contains the calibrated residual metadata contract")
    require("quantized to a calibrated signed-4 residual" in spec,
            "SPEC no longer contains the calibrated signed-4 contract")
    require(CONTRACT not in full_model,
            "full-model implementation now contains the successor and must be reviewed instead of blocked")
    require(
        quality.get("full_model_scope", {}).get("residual_scale")
        == "static_per_residual_tensor_maximum_absolute_observed_bf16_value_divided_by_127",
        "quality config residual-scale rule changed and requires a fresh audit",
    )
    require(not manifest.get("candidate_model_metadata"),
            "candidate model metadata now exists and requires a fresh audit")

    findings = [
        {
            "id": "residual_scale32_derivation_not_frozen",
            "materiality": "blocks_construct_faithful_all_layer_quality_discriminator",
            "evidence": [
                {
                    **artifact("design/NUMERICAL_REPLACEMENT_PROPOSAL.md"),
                    "line": line_of("design/NUMERICAL_REPLACEMENT_PROPOSAL.md", "Each layer/head has frozen Scale32 metadata"),
                    "observation": "The contract freezes Scale32 format and use but gives no deterministic calibration equation, prompt scope, clamp objective, or tie rule for selecting one residual Scale32 per layer/head.",
                },
                {
                    **artifact("design/SPEC.md"),
                    "line": line_of("design/SPEC.md", "quantized to a calibrated signed-4 residual"),
                    "observation": "The specification says calibrated signed-4 but does not define the calibration algorithm or bind a generated metadata table.",
                },
            ],
        },
        {
            "id": "baseline_scale32_preservation_rule_not_frozen",
            "materiality": "blocks_bit_for_bit_base_path_and_cross_term_scale_consistency",
            "evidence": [
                {
                    **artifact("design/NUMERICAL_REPLACEMENT_PROPOSAL.md"),
                    "line": line_of("design/NUMERICAL_REPLACEMENT_PROPOSAL.md", "baseline Q/K output scale"),
                    "observation": "No exact rule maps the existing calibrated per-head baseline scales into Scale32 records while proving the ordinary int8 Q/K and base-score path remains bit-for-bit unchanged.",
                }
            ],
        },
        {
            "id": "quality_config_and_reference_not_candidate_bound",
            "materiality": "running_now_would_measure_an_authored_unfrozen_construct",
            "evidence": [
                {
                    **artifact("benchmark/quality/QUALITY_CONFIG.json"),
                    "json_path": "$.full_model_scope.residual_scale",
                    "observed": quality["full_model_scope"]["residual_scale"],
                    "observation": "The only residual-scale rule is tensor-wide signed-int8 /127 behavior, not candidate per-layer/head signed-4 /7 metadata.",
                },
                {
                    **artifact("tools/ace2_full_model_fixed_point.py"),
                    "observation": "The accepted quality runner has no shared_qk_residual_cross_term_attention_v1 mechanism, metadata generator, or tensor/scalar equality hook.",
                },
            ],
        },
        {
            "id": "candidate_model_image_metadata_absent",
            "materiality": "no_hash_bound_24_layer_scale_table_exists",
            "evidence": [
                {
                    **artifact("design/RTL_MANIFEST.json"),
                    "json_path": "$.candidate_model_metadata",
                    "observed": None,
                    "observation": "No model-image baseline/residual Scale32 table or regeneration command is bound for the candidate.",
                }
            ],
        },
    ]
    return packet, review, candidate_hash, findings


def bind() -> None:
    packet, review, candidate_hash, findings = audit()
    now = utc_now()
    blocker = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "candidate_id": packet["candidate_id"],
        "candidate_rtl_hash": candidate_hash,
        "bound_at_utc": now,
        "current_stage": "rtl",
        "stage_closing": False,
        "decision": DECISION,
        "accepted_capability": False,
        "rtl_checklist_review": artifact(REVIEW),
        "findings": findings,
        "prohibited_runs": {
            "focused_all_layer_quality_discriminator": False,
            "paired_smoke": False,
            "full_shell_regression": False,
            "canonical_sky130_ppa": False,
            "prototype": False,
            "benchmark": False,
            "signoff": False,
        },
        "preserved_contracts": {
            "mode": "ADVANCE",
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "area_cap_non_sram_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "quality_limit_ratio": 1.05,
            "historical_ppa_frontier": {
                "cells": 62199,
                "non_sram_area_mm2": 0.6108746272,
                "setup_slack_ns_at_100mhz": 0.1502,
                "candidate_ppa_run": False,
            },
        },
        "required_resolution": (
            "Manager rolls rtl back to architecture; freeze and independently review the exact residual Scale32 derivation, "
            "baseline Scale32 preservation rule, model-image metadata table/regeneration command, and construct-faithful full-model hook before returning to rtl."
        ),
        "claim_boundary": "Architecture-contract blocker only; no quality, PPA, benchmark, signoff, tapeout, or silicon claim.",
    }
    blocker["integrity"] = {"canonical_sha256": None}
    blocker["integrity"]["canonical_sha256"] = canonical_sha256(blocker)
    dump(BLOCKER, blocker)

    policy = load("design/FAST_LOOP_POLICY.json")
    auth = candidate_auth(policy["active_repair_authorization"], candidate_hash)
    auth["blocker_binding_sha256"] = sha256(BLOCKER)
    for key in ("active_repair_authorization", "architecture_proposal_authorization", "selected_replacement_contract"):
        policy[key] = copy.deepcopy(auth)
    policy["manager_recommendation"] = MANAGER_ACTION
    dump("design/FAST_LOOP_POLICY.json", policy)

    manifest = load("design/RTL_MANIFEST.json")
    manifest.update({
        "architecture_contract_status": STATUS,
        "candidate_status": STATUS,
        "candidate_meets_numeric_acceptance": False,
        "candidate_verification_complete": False,
        "candidate_requires_fresh_sky130_ppa": False,
        "generated_at_utc": now,
    })
    manifest["proposed_replacement_contract"] = copy.deepcopy(auth)
    manifest["candidate_review_binding"]["quality_gate_status"] = STATUS
    manifest["candidate_review_binding"]["quality_gate_evidence"] = artifact(BLOCKER)
    manifest["traceability"]["architecture_contract_gap"] = {
        "status": STATUS,
        "resolution_owner": "Manager rollback to architecture and independent architecture re-review",
        "evidence": BLOCKER,
    }
    manifest["traceability"]["stage_checklist"] = copy.deepcopy(CHECKLIST)
    manifest["claim_boundaries"] = [
        "The exact standalone residual cross-term RTL checklist remains independently reviewed.",
        "No construct-faithful all-layer quality run is legal because candidate Scale32 metadata derivation is not frozen.",
        "The accepted prefix remains through layer_0.v_proj and layer_0.rope_q remains first unsupported.",
        "No paired smoke, shell regression, candidate PPA, prototype, benchmark, or signoff result is claimed.",
    ]
    manifest.setdefault("latest_evidence", {})["qk_residual_cross_term_quality_blocker"] = artifact(BLOCKER)
    dump("design/RTL_MANIFEST.json", manifest)

    oracle = load("reference/ORACLE_MANIFEST.json")
    oracle["architecture_contract_status"] = STATUS
    oracle["proposed_architecture_contract"] = copy.deepcopy(auth)
    for item in oracle.get("oracles", []):
        if item.get("contract_id") == CONTRACT:
            item["status"] = STATUS
            item["quality_gate_blocker"] = artifact(BLOCKER)
    oracle["generated_at_utc"] = now
    dump("reference/ORACLE_MANIFEST.json", oracle)

    target = load("design/TARGET.json")
    target["current_stage"] = "rtl"
    target["current_architecture_contract"] = copy.deepcopy(auth)
    target["fast_loop_contract"]["manager_recommendation"] = MANAGER_ACTION
    for key in ("active_repair_authorization", "architecture_proposal_authorization"):
        target["fast_loop_contract"][key] = copy.deepcopy(auth)
    dump("design/TARGET.json", target)

    scope = load("design/CHIP_SCOPE.json")
    scope["stage"] = {
        "current_stage": "rtl",
        "current_stage_status": STATUS,
        "downstream_stages_locked_until_manager_advance": ["verification", "ppa", "prototype", "benchmark", "signoff"],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    scope["authority_override"]["required_next_action"] = MANAGER_ACTION
    scope["authority_override"]["operator_implementation_approval"] = copy.deepcopy(auth)
    scope["operator_owned_execution_policy"]["active_successor_contract"] = copy.deepcopy(auth)
    scope["implementation_frontier"]["latest_decision"] = DECISION
    dump("design/CHIP_SCOPE.json", scope)

    (ROOT / "design/RTL_TRACEABILITY.md").write_text(f"""# ACE-2 RTL traceability notes

## Accepted shell frontier

The accepted shell remains hash-bound through `layer_0.v_proj`; first unsupported
remains `layer_0.rope_q`. Historical accepted PPA remains 62,199 cells,
0.6108746272 mm2, and +0.1502 ns setup slack at 100 MHz.

## Current residual cross-term standalone RTL

- Candidate `{packet['candidate_id']}` remains hash-bound at `{candidate_hash}`.
- Independent review `{REVIEW}` certifies all three standalone RTL checklist
  items for the exact interfaces, parameters, hardware discipline, and provenance.

## Quality-admission architecture blocker

`{BLOCKER}` records four material gaps discovered before the focused all-layer
quality discriminator: no exact residual Scale32 calibration rule, no exact
baseline-to-Scale32 preservation rule, no candidate-bound quality-runner hook,
and no hash-bound 24-layer model-image metadata table. Choosing any of those in
RTL would author new numerical behavior after architecture freeze. Therefore no
quality, paired smoke, shell regression, or PPA run was started.

The required action is Manager rollback to architecture for a refreeze and
independent review. Planner and Engineer do not edit `research/PIPELINE_STATE.json`.
""", encoding="utf-8")

    (ROOT / "CHECKPOINT.md").write_text(f"""# Goal

Execute the bounded `shared_qk_residual_cross_term_attention_v1` capability loop
without inventing numerical behavior or entering a downstream stage.

# Current State

Standalone RTL candidate `{packet['candidate_id']}` at `{candidate_hash}` passed
independent RTL checklist review. The next focused quality gate is blocked by a
material architecture-contract gap recorded in `{BLOCKER}`.

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
`layer_0.rope_q`; mode remains `ADVANCE`. The 2.0 mm2 non-SRAM cap, 100 MHz floor,
1.05x quality limit, and historical PPA frontier are unchanged.

# Verified Done

- Reference/vector generation, lint, elaboration, standalone RTL simulation, and
  exact interface binding pass for the candidate hash.
- Independent review certifies `rtl.contract-traceability`,
  `rtl.hardware-discipline`, and `rtl.ip-provenance` for the standalone modules.
- A pre-quality audit confirms that no deterministic per-layer/head residual
  Scale32 derivation, baseline Scale32 preservation rule, model-image metadata
  table, or construct-faithful full-model hook is frozen.

# Not Run

The focused all-layer quality discriminator, paired smoke, shell regression,
canonical SKY130 PPA, prototype, benchmark, and signoff were not run. Running
them now would measure an authored, unfrozen numerical construct.

# Blocker / Required Action

Manager must roll `rtl` back to `architecture`. Freeze and independently review:

1. the exact residual Scale32 calibration equation and frozen prompt scope;
2. the exact baseline per-head scale-to-Scale32 rule preserving the base path;
3. the complete 24-layer Q/K baseline/residual metadata table and regeneration command;
4. the construct-faithful full-model reference hook and tensor/scalar equality test.

Planner and Engineer must not edit `research/PIPELINE_STATE.json`.
""", encoding="utf-8")

    public = load("research/PUBLIC_STATUS.json")
    public.update({
        "current_mode": "ADVANCE",
        "supported_layer_operator_prefix": PREFIX,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "latest_decision": DECISION,
        "latest_ppa_frontier_status": "historical_frontier_preserved_no_candidate_ppa",
        "selected_replacement_contract": copy.deepcopy(auth),
        "generated_at_utc": now,
        "last_updated_utc": now,
    })
    public["architecture_proposal_gate"].update(copy.deepcopy(auth))
    public["architecture_proposal_gate"]["required_operator_action"] = "none_targets_unchanged_manager_architecture_reroute_required"
    public["stage"] = {
        "completed_prior_stages": ["definition", "architecture", "environment"],
        "current_stage": "rtl",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_evidence": [
            "research/PIPELINE_STATE.json", "design/RTL_MANIFEST.json", PACKET, REVIEW, BLOCKER
        ],
        "current_stage_status": STATUS,
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    for name in ("dashboard_fields", "implementation_frontier"):
        container = public[name]
        container.update({
            "current_stage": "rtl",
            "current_mode": "ADVANCE",
            "latest_decision": DECISION,
            "required_manager_action": MANAGER_ACTION,
            "required_operator_action": "none_targets_unchanged_manager_architecture_reroute_required",
            "candidate_mechanism": copy.deepcopy(auth),
            "candidate_rtl_hash": candidate_hash,
            "candidate_rtl_hash_scope": "ordered_standalone_candidate_sources_not_shell_admitted",
            "rtl_contract_traceability": True,
            "routing_status": "manager_architecture_rollback_required",
            "operator_policy": copy.deepcopy(policy),
        })
    public["blockers"] = [{
        "id": "qk_residual_scale32_architecture_contract_gap",
        "stage": "rtl",
        "status": "active",
        "evidence": BLOCKER,
        "reason": "Candidate model-image Scale32 derivation and construct-faithful quality hook are not frozen.",
        "required_resolution": MANAGER_ACTION,
    }]
    public["public_claims"] = [
        {"claim": "current Manager-owned stage remains rtl", "evidence": ["research/PIPELINE_STATE.json", "research/PUBLIC_STATUS.json"]},
        {"claim": "the exact standalone residual cross-term RTL checklist is independently reviewed", "evidence": [PACKET, REVIEW, "design/RTL_MANIFEST.json"]},
        {"claim": "focused quality is blocked before execution by an architecture metadata-derivation gap", "evidence": [BLOCKER]},
        {"claim": "the accepted prefix remains through layer_0.v_proj and layer_0.rope_q remains first unsupported", "evidence": ["design/RTL_MANIFEST.json", "research/PUBLIC_STATUS.json"]},
        {"claim": "the historical PPA frontier, 2.0 mm2 cap, 100 MHz floor, and 1.05x quality limit are unchanged", "evidence": ["design/PPA_FRONTIER_LEDGER.json", "design/TARGET.json"]},
    ]
    refresh_paths = {record.get("path") for record in public.get("artifact_hashes", [])
                     if isinstance(record, dict) and record.get("path")}
    refresh_paths.update({
        "CHECKPOINT.md", "design/CHIP_SCOPE.json", "design/FAST_LOOP_POLICY.json",
        "design/RTL_MANIFEST.json", "design/RTL_TRACEABILITY.md", "design/TARGET.json",
        "reference/ORACLE_MANIFEST.json", PACKET, REVIEW, BLOCKER, BINDER,
    })
    public["artifact_hashes"] = [artifact(path) for path in sorted(refresh_paths)
                                 if (ROOT / path).is_file()]
    public.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    public["integrity"]["canonicalization"] = (
        "UTF-8 sorted keys two-space indentation trailing newline canonical hash null during hash"
    )
    public["integrity"]["canonical_sha256"] = None
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump("research/PUBLIC_STATUS.json", public)


def validate() -> None:
    packet, review, candidate_hash, findings = audit()
    blocker = load(BLOCKER)
    manifest = load("design/RTL_MANIFEST.json")
    public = load("research/PUBLIC_STATUS.json")
    require(blocker.get("decision") == DECISION, "blocker decision stale")
    require(blocker.get("candidate_rtl_hash") == candidate_hash, "blocker candidate hash stale")
    require(len(findings) == len(blocker.get("findings", [])) == 4, "blocker findings incomplete")
    require(not any(blocker.get("prohibited_runs", {}).values()), "blocker falsely records a downstream run")
    require(blocker.get("integrity", {}).get("canonical_sha256") == canonical_sha256(blocker),
            "blocker canonical integrity mismatch")
    require(manifest.get("candidate_status") == STATUS, "manifest blocker status stale")
    require(manifest.get("traceability", {}).get("stage_checklist") == CHECKLIST,
            "standalone RTL checklist changed")
    require(public.get("latest_decision") == DECISION, "public latest decision stale")
    require(public.get("stage", {}).get("current_stage_status") == STATUS, "public stage status stale")
    require(public.get("supported_layer_operator_prefix") == PREFIX, "supported prefix changed")
    require(public.get("first_unsupported_layer_operator") == FIRST_UNSUPPORTED,
            "first unsupported operator changed")
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public),
            "public canonical integrity mismatch")
    require("/home/" not in json.dumps(public, sort_keys=True), "public status leaks private paths")
    require(review.get("reviewer_status") == "done", "RTL review is no longer done")
    print(f"QK_RESIDUAL_QUALITY_BLOCKER_CHECK_PASS candidate={packet['candidate_id']}")


if __name__ == "__main__":
    if "--check" not in sys.argv[1:]:
        bind()
    validate()
