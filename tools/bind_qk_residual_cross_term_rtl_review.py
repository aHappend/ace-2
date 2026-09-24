#!/usr/bin/env python3
"""Bind the independent residual cross-term RTL checklist verdict to live status."""

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
RTL = "rtl/ace2_qk_residual_cross_term_core.sv"
PACKET = "evidence/shared_qk_residual_cross_term_attention_v1/latest/PRECHECK.json"
REVIEW = "evidence/review/rtl_checklist_shared_qk_residual_cross_term_attention_v1/decision.json"
RUNNER = "tools/run_qk_residual_cross_term_rtl_review.py"
BINDER = "tools/bind_qk_residual_cross_term_rtl_review.py"
PREFIX = ["layer_0.input_rmsnorm", "layer_0.q_proj", "layer_0.k_proj", "layer_0.v_proj"]
FIRST_UNSUPPORTED = "layer_0.rope_q"
CHECKLIST = {
    "rtl.contract-traceability": True,
    "rtl.hardware-discipline": True,
    "rtl.ip-provenance": True,
}
STATUS = "standalone_rtl_checklist_done_quality_discriminator_pending"
DECISION = "residual_cross_term_rtl_checklist_done_quality_pending"
MANAGER_ACTION = "hold_rtl_for_focused_quality_discriminator"
BLOCKED_STATUS = "quality_gate_blocked_architecture_metadata_derivation_missing"
BLOCKED_DECISION = "residual_cross_term_quality_gate_blocked_architecture_contract_gap"
BLOCKED_MANAGER_ACTION = "rollback_rtl_to_architecture_refreeze_residual_scale_metadata_derivation"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: str) -> dict[str, Any]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def dump(path: str, value: dict[str, Any]) -> None:
    destination = ROOT / path
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


def reviewed_authorization(policy: dict[str, Any], candidate_hash: str) -> dict[str, Any]:
    result = copy.deepcopy(policy["active_repair_authorization"])
    require(result.get("contract_id") == CONTRACT, "active authorization contract mismatch")
    require(result.get("operator_approval_consumed") is True, "implementation approval is not consumed")
    result.update({
        "candidate_rtl_hash": candidate_hash,
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "rtl_started": True,
        "status": STATUS,
        "required_manager_action": MANAGER_ACTION,
    })
    return result


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], str]:
    require(load("research/PIPELINE_STATE.json").get("current_stage") == "rtl",
            "Manager-owned stage is not rtl")
    packet = load(PACKET)
    review = load(REVIEW)
    candidate_hash = str(packet.get("candidate_rtl_hash") or "")
    require(packet.get("contract_id") == CONTRACT, "preflight contract mismatch")
    require(packet.get("status") == "pass_ready_for_independent_rtl_review", "preflight is not passing")
    require(review.get("reviewer_status") == "done", "independent RTL review is not done")
    require(review.get("contract_id") == CONTRACT, "review contract mismatch")
    require(review.get("candidate_id") == packet.get("candidate_id"), "review candidate ID mismatch")
    require(review.get("candidate_rtl_hash") == candidate_hash, "review candidate hash mismatch")
    require(review.get("live_rtl_source_sha256") == sha256(RTL), "review live RTL hash mismatch")
    require(review.get("checklist") == CHECKLIST, "review checklist is incomplete")
    require(review.get("stage_closing") is False, "review unexpectedly closes the stage")
    manifest = load("design/RTL_MANIFEST.json")
    require(manifest.get("candidate_rtl_hash") == candidate_hash, "manifest candidate hash mismatch")
    for path, digest in packet.get("source_hashes", {}).items():
        require(sha256(path) == digest, f"candidate source changed after review: {path}")
    return packet, review, candidate_hash


def bind() -> None:
    packet, review, candidate_hash = validate_inputs()
    now = utc_now()

    policy = load("design/FAST_LOOP_POLICY.json")
    auth = reviewed_authorization(policy, candidate_hash)
    for key in ("active_repair_authorization", "architecture_proposal_authorization", "selected_replacement_contract"):
        policy[key] = copy.deepcopy(auth)
    policy["manager_recommendation"] = MANAGER_ACTION
    dump("design/FAST_LOOP_POLICY.json", policy)

    manifest = load("design/RTL_MANIFEST.json")
    manifest.update({
        "architecture_contract_status": STATUS,
        "candidate_status": STATUS,
        "candidate_review_binding": {
            "status": "independent_rtl_checklist_done",
            "candidate_capability_accepted": False,
            "evidence": REVIEW,
            "evidence_sha256": sha256(REVIEW),
            "stage_closing": False,
            "quality_discriminator_complete": False,
        },
        "independent_reviewer_verdict": "done",
        "independent_reviewer_acceptance": {
            **artifact(REVIEW),
            "scope": "standalone_candidate_rtl_checklist_only",
            "quality_discriminator_complete": False,
            "shell_admitted": False,
            "ppa_run": False,
        },
        "interfaces_contract_status": "candidate_standalone_interfaces_reviewed_quality_pending_not_shell_admitted",
        "generated_at_utc": now,
    })
    manifest["proposed_replacement_contract"] = copy.deepcopy(auth)
    manifest["traceability"]["architecture_contract_gap"] = {
        "status": "standalone_rtl_checklist_done_quality_admission_pending",
        "resolution_owner": "focused all-layer quality discriminator",
    }
    manifest["traceability"]["stage_checklist"] = copy.deepcopy(CHECKLIST)
    manifest["claim_boundaries"] = [
        "The residual cross-term standalone RTL checklist is independently reviewed for the exact candidate hash.",
        "The accepted prefix remains through layer_0.v_proj and layer_0.rope_q remains first unsupported.",
        "The focused all-layer quality discriminator, paired smoke, shell admission/regression, and candidate PPA have not run.",
        "The historical 0.6108746272 mm2 and +0.1502 ns at 100 MHz frontier remains unchanged.",
    ]
    manifest.setdefault("latest_evidence", {})["qk_residual_cross_term_rtl_review"] = artifact(REVIEW)
    dump("design/RTL_MANIFEST.json", manifest)

    oracle = load("reference/ORACLE_MANIFEST.json")
    oracle["architecture_contract_status"] = STATUS
    oracle["proposed_architecture_contract"] = copy.deepcopy(auth)
    for item in oracle.get("oracles", []):
        if item.get("contract_id") == CONTRACT:
            item["status"] = "rtl_checklist_done_quality_discriminator_pending"
            item["independent_rtl_review"] = artifact(REVIEW)
    oracle["generated_at_utc"] = now
    dump("reference/ORACLE_MANIFEST.json", oracle)

    target = load("design/TARGET.json")
    target["current_stage"] = "rtl"
    target["current_architecture_contract"] = copy.deepcopy(auth)
    target.setdefault("fast_loop_contract", {})["manager_recommendation"] = MANAGER_ACTION
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
    scope["authority_override"]["required_next_action"] = "run_focused_all_layer_quality_discriminator"
    scope["authority_override"]["operator_implementation_approval"] = copy.deepcopy(auth)
    scope["operator_owned_execution_policy"]["active_successor_contract"] = copy.deepcopy(auth)
    scope["implementation_frontier"]["latest_decision"] = DECISION
    dump("design/CHIP_SCOPE.json", scope)

    (ROOT / "design/RTL_TRACEABILITY.md").write_text(f"""# ACE-2 RTL traceability notes

## Accepted shell frontier

The accepted shell remains hash-bound through `layer_0.v_proj`; first unsupported
remains `layer_0.rope_q`. Historical accepted PPA remains 62,199 cells,
0.6108746272 mm2, and +0.1502 ns setup slack at 100 MHz.

## Current residual cross-term candidate

- Contract: `{CONTRACT}`.
- Candidate: `{packet['candidate_id']}`; ordered source hash `{candidate_hash}`.
- `ace2_qk_residual_sidecar_core` traces projection residual generation, signed-4
  clamping and reserved-code handling, residual RoPE, one multiplier, and the
  existing ready/valid unsigned-divider service.
- `ace2_residual_cross_term_score_core` serializes three correction dots through
  one multiplier, converts all four Scale32 terms to signed Q20.44, and performs
  checked signed-67 addition.
- Exact interfaces and parameters, warning-free lint, clean elaboration,
  bit-exact standalone simulation, reset/clear/backpressure coverage, and
  first-party/generated-source provenance are bound in `{PACKET}`.
- Independent Reviewer decision `{REVIEW}` certifies all three RTL checklist
  items for this exact hash. This is not capability admission: focused all-layer
  quality discrimination is next, and paired smoke, shell regression, and PPA
  remain gated.

## Historical rejected candidates

All earlier RoPE/score successors remain historical bounded no-go evidence and
are not current RTL capability.
""", encoding="utf-8")

    (ROOT / "CHECKPOINT.md").write_text(f"""# Goal

Execute the one bounded `shared_qk_residual_cross_term_attention_v1` RTL capability
loop without entering a downstream stage.

# Current State

The exact standalone candidate hash `{candidate_hash}` has passed independent RTL
checklist review. `current_stage` remains Manager-owned `rtl`; the candidate is not
shell-admitted or quality-admitted.

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
`layer_0.rope_q`; mode remains `ADVANCE`. The 2.0 mm2 non-SRAM cap, 100 MHz floor,
and historical PPA frontier are unchanged.

# Verified Done

- Integer reference and deterministic JSON/SVH generation pass.
- Both synthesizable modules elaborate and lint without warnings.
- Standalone RTL simulation is bit-exact and exercises reset, clear, errors, and
  backpressure.
- Independent review `{REVIEW}` certifies `rtl.contract-traceability`,
  `rtl.hardware-discipline`, and `rtl.ip-provenance` for the exact candidate hash.

# Locked / Not Run

The focused all-layer quality discriminator has not run. Paired smoke remains
locked until that discriminator passes; shell admission/regression and canonical
SKY130 PPA remain locked until paired smoke passes. Prototype, benchmark, and
signoff remain out of scope.

# Next Required Action

Run the frozen focused all-layer quality discriminator. Preserve the candidate
hash and stop immediately on any mandatory metric that does not strictly improve.
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
    public["architecture_proposal_gate"]["required_operator_action"] = "none_implementation_consumed_quality_gate_pending"
    public["stage"] = {
        "completed_prior_stages": ["definition", "architecture", "environment"],
        "current_stage": "rtl",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_evidence": [
            "research/PIPELINE_STATE.json",
            "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
            "design/RTL_MANIFEST.json",
            PACKET,
            REVIEW,
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
            "required_operator_action": "none_implementation_consumed_quality_gate_pending",
            "candidate_mechanism": copy.deepcopy(auth),
            "candidate_rtl_hash": candidate_hash,
            "candidate_rtl_hash_scope": "ordered_standalone_candidate_sources_not_shell_admitted",
            "rtl_contract_traceability": True,
            "routing_status": "focused_quality_discriminator_pending",
            "operator_policy": copy.deepcopy(policy),
        })
    public["blockers"] = [{
        "id": "focused_quality_discriminator_pending",
        "stage": "rtl",
        "status": "active",
        "evidence": REVIEW,
        "reason": "The exact standalone RTL checklist is reviewed, but model-level quality admission has not run.",
        "required_resolution": "Run the frozen all-layer focused discriminator and stop on any non-improving mandatory metric.",
    }]
    public["public_claims"] = [
        {"claim": "current Manager-owned stage remains rtl", "evidence": ["research/PIPELINE_STATE.json", "research/PUBLIC_STATUS.json"]},
        {"claim": "the exact residual cross-term standalone RTL checklist is independently reviewed", "evidence": [PACKET, REVIEW, "design/RTL_MANIFEST.json"]},
        {"claim": "the accepted prefix remains through layer_0.v_proj and layer_0.rope_q remains first unsupported", "evidence": ["design/RTL_MANIFEST.json", "research/PUBLIC_STATUS.json"]},
        {"claim": "focused quality, paired smoke, shell regression, and candidate PPA have not run", "evidence": [PACKET, REVIEW]},
        {"claim": "the historical PPA frontier, 2.0 mm2 cap, and 100 MHz floor are unchanged", "evidence": ["design/PPA_FRONTIER_LEDGER.json", "design/TARGET.json"]},
    ]
    metrics = [item for item in public.get("reviewer_certified_metrics", [])
               if item.get("name") != "rtl_checklist_shared_qk_residual_cross_term_attention_v1"]
    metrics.append({
        "name": "rtl_checklist_shared_qk_residual_cross_term_attention_v1",
        "status": "standalone_rtl_checklist_done_quality_pending",
        "certified_at_utc": review["review_bound_at_utc"],
        "contract_binding_status": CONTRACT,
        "candidate_rtl_hash": candidate_hash,
        "checklist": copy.deepcopy(CHECKLIST),
        "evidence": [REVIEW],
    })
    public["reviewer_certified_metrics"] = metrics

    refresh_paths = {record.get("path") for record in public.get("artifact_hashes", [])
                     if isinstance(record, dict) and record.get("path")}
    refresh_paths.update({
        "CHECKPOINT.md", "design/CHIP_SCOPE.json", "design/FAST_LOOP_POLICY.json",
        "design/RTL_MANIFEST.json", "design/RTL_TRACEABILITY.md", "design/TARGET.json",
        "reference/ORACLE_MANIFEST.json", PACKET, REVIEW, RUNNER, BINDER,
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
    packet, review, candidate_hash = validate_inputs()
    manifest = load("design/RTL_MANIFEST.json")
    policy = load("design/FAST_LOOP_POLICY.json")
    public = load("research/PUBLIC_STATUS.json")
    require(manifest.get("independent_reviewer_verdict") == "done", "manifest review verdict stale")
    require(manifest.get("candidate_status") in {STATUS, BLOCKED_STATUS},
            "manifest candidate status stale")
    require(manifest.get("traceability", {}).get("stage_checklist") == CHECKLIST,
            "manifest checklist stale")
    require(policy.get("manager_recommendation") in {MANAGER_ACTION, BLOCKED_MANAGER_ACTION},
            "policy next action stale")
    require(public.get("latest_decision") in {DECISION, BLOCKED_DECISION},
            "public latest decision stale")
    require(public.get("stage", {}).get("current_stage_status") in {STATUS, BLOCKED_STATUS},
            "public stage status stale")
    require(public.get("supported_layer_operator_prefix") == PREFIX, "public supported prefix changed")
    require(public.get("first_unsupported_layer_operator") == FIRST_UNSUPPORTED,
            "public first unsupported operator changed")
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public),
            "public canonical integrity mismatch")
    require("/home/" not in json.dumps(public, sort_keys=True), "public status leaks private paths")
    require(packet.get("candidate_rtl_hash") == candidate_hash, "candidate hash changed")
    require(review.get("reviewer_status") == "done", "review no longer done")
    print(f"QK_RESIDUAL_RTL_REVIEW_BIND_CHECK_PASS candidate={packet['candidate_id']}")


if __name__ == "__main__":
    if "--check" not in sys.argv[1:]:
        bind()
    validate()
