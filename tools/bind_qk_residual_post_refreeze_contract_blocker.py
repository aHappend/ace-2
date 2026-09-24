#!/usr/bin/env python3
"""Bind the post-refreeze base-score preservation contract blocker."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_qk_residual_cross_term_attention_v1"
RTL = "rtl/ace2_qk_residual_cross_term_core.sv"
PROPOSAL = "design/NUMERICAL_REPLACEMENT_PROPOSAL.md"
SPEC = "design/SPEC.md"
HOOK = "reference/qk_residual_cross_term_full_model_hook.json"
METADATA = "reference/generated/qk_residual_scale32_metadata.json"
BLOCKER = "evidence/shared_qk_residual_cross_term_attention_v1/latest/POST_REFREEZE_CONTRACT_BLOCKER.json"
BINDER = "tools/bind_qk_residual_post_refreeze_contract_blocker.py"
STATUS = "post_refreeze_base_score_preservation_contract_gap"
DECISION = "residual_cross_term_post_refreeze_rtl_contract_gap"
MANAGER_ACTION = "rollback_rtl_to_architecture_refreeze_base_score_preservation_interface"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
CHECKLIST = {
    "rtl.contract-traceability": False,
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
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def artifact(path: str) -> dict[str, Any]:
    item = ROOT / path
    return {"path": path, "bytes": item.stat().st_size, "sha256": sha256(path)}


def canonical_sha256(value: dict[str, Any]) -> str:
    candidate = copy.deepcopy(value)
    candidate.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(candidate, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def line_of(path: str, phrase: str) -> int:
    for index, line in enumerate(
        (ROOT / path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if phrase in line:
            return index
    raise RuntimeError(f"missing expected phrase in {path}: {phrase}")


def blocked_authorization(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    require(result.get("contract_id") == CONTRACT, "active contract changed")
    require(result.get("operator_approval_consumed") is False,
            "post-refreeze implementation authorization was unexpectedly consumed")
    result.update(
        {
            "implementation_authorized": False,
            "operator_approval_consumed": False,
            "required_manager_action": MANAGER_ACTION,
            "rtl_started": False,
            "stage_closing": False,
            "status": STATUS,
        }
    )
    return result


def audit(*, require_executable_authorization: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pipeline = load("research/PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    latest = pipeline.get("stage_history", [])[-1]
    require(
        latest.get("by") == "manager"
        and latest.get("from_stage") == "environment"
        and latest.get("to_stage") == "rtl",
        "latest Manager stage transition is not environment to rtl",
    )

    proposal = (ROOT / PROPOSAL).read_text(encoding="utf-8")
    spec = (ROOT / SPEC).read_text(encoding="utf-8")
    rtl = (ROOT / RTL).read_text(encoding="utf-8")
    hook = load(HOOK)
    metadata = load(METADATA)
    manifest = load("design/RTL_MANIFEST.json")
    policy = load("design/FAST_LOOP_POLICY.json")

    require("The base term remains the accepted baseline computation" in proposal,
            "proposal no longer freezes accepted baseline computation")
    require("Baseline Scale32 is sideband metadata only." in proposal,
            "proposal no longer freezes the sideband-only Scale32 rule")
    require("produce bit-identical Q/K" in spec and "base-score tensors" in spec,
            "SPEC no longer requires baseline tensor identity")
    require("base_score_before_residual_correction" in hook.get("baseline_equality_boundaries", []),
            "hook no longer requires base-score equality")
    require(
        hook.get("tensor_hook_order", []).index("assert_candidate_base_score_equal")
        < hook.get("tensor_hook_order", []).index(
            "add_base_times_residual_residual_times_base_and_residual_times_residual_terms"
        ),
        "hook no longer checks base score before correction",
    )
    require("input  logic signed [31:0]        base_dot_s32_i" in rtl,
            "RTL score interface no longer accepts the frozen base-dot input")
    require("base_score_q20_44" not in rtl,
            "RTL now has a base-score preservation interface and requires a fresh audit")
    require("selected_dot_w = base_dot_q;" in rtl,
            "RTL no longer selects the base dot for Scale32 conversion")
    require("query_sig_q" in rtl and "key_sig_q" in rtl,
            "RTL no longer applies Q/K Scale32 metadata to the selected base dot")
    require(len(metadata.get("records", [])) == 384, "metadata table row count changed")
    require(manifest.get("candidate_rtl_hash") is None,
            "a post-refreeze current candidate unexpectedly started")
    require(
        manifest.get("candidate_rtl_hash_scope") == "successor_rtl_not_started",
        "post-refreeze candidate start boundary changed",
    )
    active = policy.get("active_repair_authorization", {})
    if require_executable_authorization:
        require(active.get("implementation_authorized") is True,
                "reconciled standing authorization is not active")
    else:
        require(active.get("implementation_authorized") is False,
                "blocked standing authorization is unexpectedly executable")
    require(active.get("operator_approval_consumed") is False,
            "reconciled standing authorization is consumed")

    findings = [
        {
            "id": "frozen_base_score_must_be_preserved_before_correction",
            "materiality": "mandatory_tensor_equality_gate",
            "evidence": [
                {
                    **artifact(PROPOSAL),
                    "line": line_of(PROPOSAL, "The base term remains the accepted baseline computation"),
                    "observation": "The selected mechanism may add residual corrections only after preserving the accepted base computation.",
                },
                {
                    **artifact(PROPOSAL),
                    "line": line_of(PROPOSAL, "Baseline Scale32 is sideband metadata only."),
                    "observation": "Baseline Scale32 is explicitly sideband-only and may not rederive the base-score result.",
                },
                {
                    **artifact(SPEC),
                    "line": line_of(SPEC, "produce bit-identical Q/K"),
                    "observation": "The frozen numerical contract requires bit-identical base-score tensors.",
                },
                {
                    **artifact(HOOK),
                    "json_path": "$.baseline_equality_boundaries[4]",
                    "observation": "The candidate hook must assert authoritative base-score equality before adding correction terms.",
                },
            ],
        },
        {
            "id": "rtl_rederives_base_term_from_dot_and_scale32",
            "materiality": "rtl_contract_traceability_false",
            "evidence": [
                {
                    **artifact(RTL),
                    "line": line_of(RTL, "input  logic signed [31:0]        base_dot_s32_i"),
                    "observation": "The score core accepts a base dot, not the already-authoritative base-score result.",
                },
                {
                    **artifact(RTL),
                    "line": line_of(RTL, "selected_dot_w = base_dot_q;"),
                    "observation": "Term zero is reconstructed by multiplying the base dot with Q/K Scale32 records, which is the operation the sideband-only rule forbids.",
                },
            ],
        },
        {
            "id": "frozen_interface_has_no_legal_bit_exact_base_score_path",
            "materiality": "cannot_choose_numerical_interpretation_in_rtl",
            "evidence": [
                {
                    **artifact(HOOK),
                    "json_path": "$.tensor_hook_order",
                    "observation": "The required sequence runs and checks the authoritative base score before residual correction.",
                },
                {
                    **artifact(RTL),
                    "observation": "No base-score input exists, so current RTL cannot implement that sequence without an interface/numerical refreeze.",
                },
            ],
        },
    ]
    return active, findings


def bind() -> None:
    active, findings = audit(require_executable_authorization=True)
    now = utc_now()
    blocker = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "current_stage": "rtl",
        "decision": DECISION,
        "accepted_capability": False,
        "stage_closing": False,
        "implementation_authorization_consumed": False,
        "findings": findings,
        "preserved_contracts": {
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "mode": "ADVANCE",
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
        "prohibited_runs": {
            "post_refreeze_rtl_preflight": False,
            "focused_all_layer_quality_discriminator": False,
            "paired_smoke": False,
            "full_shell_regression": False,
            "canonical_sky130_ppa": False,
            "prototype": False,
            "benchmark": False,
            "signoff": False,
        },
        "required_resolution": (
            "Manager rolls rtl back to architecture and refreezes one unambiguous score interface: "
            "either carry the authoritative base-score result into the correction core, or explicitly "
            "replace the frozen bit-identical base-score rule. Independent architecture and environment "
            "review must pass before returning to rtl."
        ),
        "claim_boundary": (
            "Post-refreeze RTL contract-traceability blocker only; no new RTL candidate, quality, "
            "verification, PPA, prototype, benchmark, signoff, tapeout, or silicon claim."
        ),
        "bound_at_utc": now,
        "integrity": {"canonical_sha256": None},
    }
    blocker["integrity"]["canonical_sha256"] = canonical_sha256(blocker)
    dump(BLOCKER, blocker)

    policy = load("design/FAST_LOOP_POLICY.json")
    auth = blocked_authorization(active)
    auth["blocker_binding"] = BLOCKER
    auth["blocker_binding_sha256"] = sha256(BLOCKER)
    for key in (
        "active_repair_authorization",
        "architecture_proposal_authorization",
        "selected_replacement_contract",
    ):
        policy[key] = copy.deepcopy(auth)
    policy["manager_recommendation"] = MANAGER_ACTION
    dump("design/FAST_LOOP_POLICY.json", policy)

    manifest = load("design/RTL_MANIFEST.json")
    manifest.update(
        {
            "stage": "rtl",
            "current_stage": "rtl",
            "architecture_contract_status": STATUS,
            "candidate_status": STATUS,
            "candidate_meets_numeric_acceptance": False,
            "candidate_verification_complete": False,
            "candidate_requires_fresh_sky130_ppa": False,
            "generated_at_utc": now,
        }
    )
    manifest["proposed_replacement_contract"] = copy.deepcopy(auth)
    manifest["candidate_review_binding"] = {
        "status": STATUS,
        "candidate_capability_accepted": False,
        "evidence": BLOCKER,
        "stage_closing": False,
        "historical_standalone_rtl": copy.deepcopy(
            manifest.get("historical_standalone_rtl")
        ),
    }
    manifest["traceability"]["architecture_contract_gap"] = {
        "status": STATUS,
        "resolution_owner": "Manager architecture rollback and independent re-review",
        "evidence": BLOCKER,
    }
    manifest["traceability"]["rtl.contract-traceability"] = False
    manifest["traceability"]["stage_checklist"] = copy.deepcopy(CHECKLIST)
    manifest["claim_boundaries"] = [
        "The post-refreeze implementation authorization remains unconsumed.",
        "The historical standalone RTL remains hardware-discipline and provenance evidence only.",
        "The frozen base-score equality hook and current base-dot/Scale32 RTL interface are inconsistent.",
        "The accepted prefix remains through layer_0.v_proj and layer_0.rope_q remains first unsupported.",
        "No post-refreeze RTL preflight, quality, shell regression, candidate PPA, prototype, benchmark, or signoff run is claimed.",
    ]
    manifest.setdefault("latest_evidence", {})[
        "qk_residual_post_refreeze_contract_blocker"
    ] = artifact(BLOCKER)
    dump("design/RTL_MANIFEST.json", manifest)

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
        "downstream_stages_locked_until_manager_advance": [
            "verification",
            "ppa",
            "prototype",
            "benchmark",
            "signoff",
        ],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    scope["authority_override"]["required_next_action"] = MANAGER_ACTION
    scope["authority_override"]["operator_implementation_approval"] = copy.deepcopy(auth)
    scope["operator_owned_execution_policy"]["active_successor_contract"] = copy.deepcopy(auth)
    scope["implementation_frontier"]["latest_decision"] = DECISION
    dump("design/CHIP_SCOPE.json", scope)

    oracle = load("reference/ORACLE_MANIFEST.json")
    oracle["stage"] = "rtl"
    oracle["current_stage"] = "rtl"
    oracle["architecture_contract_status"] = STATUS
    oracle["proposed_architecture_contract"] = copy.deepcopy(auth)
    oracle["claim_boundary"] = blocker["claim_boundary"]
    dump("reference/ORACLE_MANIFEST.json", oracle)

    (ROOT / "design/RTL_TRACEABILITY.md").write_text(
        f"""# ACE-2 RTL traceability notes

## Accepted shell frontier

The accepted shell remains hash-bound through `layer_0.v_proj`; first unsupported
remains `layer_0.rope_q`. Historical accepted PPA remains 62,199 cells,
0.6108746272 mm2, and +0.1502 ns setup slack at 100 MHz.

## Post-refreeze blocker

The frozen hook requires the authoritative base-score tensor to remain bit-identical
and to be checked before residual correction. The current score core accepts only
`base_dot_s32_i` and routes that dot through Q/K Scale32 conversion as term zero.
It has no authoritative base-score input. Therefore `rtl.contract-traceability` is
false for the refrozen contract even though the historical standalone source remains
disciplined and first-party.

Evidence: `{BLOCKER}`.

No post-refreeze candidate was started, and the one-time implementation authorization
remains unconsumed. Manager must roll back to architecture and refreeze one exact
base-score interface before RTL can resume.
""",
        encoding="utf-8",
    )

    (ROOT / "CHECKPOINT.md").write_text(
        f"""# Goal

Execute the bounded post-refreeze RTL refit for `{CONTRACT}` without changing
the Manager-owned stage or operator-owned targets.

# Current State

The Manager transition to `rtl` is reconciled, but the first construct-faithful
traceability audit found an architecture/interface contradiction before a new RTL
candidate started. The frozen hook requires bit-identical authoritative base score
before correction; the current score core accepts only a base dot and rederives term
zero with Q/K Scale32. The implementation authorization remains unconsumed.

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
`layer_0.rope_q`; mode remains `ADVANCE`. The 2.0 mm2 non-SRAM cap, 100 MHz floor,
1.05x quality limit, and historical PPA frontier are unchanged.

# Verified Evidence

- Manager-owned stage is `rtl`, with a recorded `environment -> rtl` transition.
- RTL-entry state reconciliation passed without changing `research/PIPELINE_STATE.json`.
- `{BLOCKER}` binds the proposal, specification, full-model hook, metadata table,
  and current RTL interface/source hashes.
- `rtl.contract-traceability=false`; historical hardware-discipline and first-party
  provenance evidence remain intact.

# Not Run

No post-refreeze RTL preflight, focused quality, paired smoke, shell regression,
canonical SKY130 PPA, prototype, benchmark, or signoff run was legal or executed.

# Required Action

Manager must roll `rtl` back to `architecture` and refreeze one unambiguous score
interface, then obtain independent architecture and environment acceptance before
returning to RTL. Planner and Engineer must not edit `research/PIPELINE_STATE.json`.
""",
        encoding="utf-8",
    )

    public = load("research/PUBLIC_STATUS.json")
    public.update(
        {
            "current_mode": "ADVANCE",
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "latest_decision": DECISION,
            "latest_ppa_frontier_status": "historical_frontier_preserved_no_candidate_ppa",
            "selected_replacement_contract": copy.deepcopy(auth),
            "generated_at_utc": now,
            "last_updated_utc": now,
        }
    )
    public["architecture_proposal_gate"].update(copy.deepcopy(auth))
    public["architecture_proposal_gate"]["required_operator_action"] = (
        "none_targets_unchanged_manager_architecture_reroute_required"
    )
    public["stage"] = {
        "completed_prior_stages": ["definition", "architecture", "environment"],
        "current_stage": "rtl",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_evidence": [
            "research/PIPELINE_STATE.json",
            PROPOSAL,
            SPEC,
            HOOK,
            RTL,
            BLOCKER,
        ],
        "current_stage_status": STATUS,
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    for name in ("dashboard_fields", "implementation_frontier"):
        container = public[name]
        container.update(
            {
                "current_stage": "rtl",
                "current_mode": "ADVANCE",
                "latest_decision": DECISION,
                "required_manager_action": MANAGER_ACTION,
                "required_operator_action": "none_targets_unchanged_manager_architecture_reroute_required",
                "candidate_mechanism": copy.deepcopy(auth),
                "candidate_rtl_hash": None,
                "candidate_rtl_hash_scope": "successor_rtl_not_started",
                "rtl_contract_traceability": False,
                "routing_status": "manager_architecture_rollback_required",
                "operator_policy": copy.deepcopy(policy),
            }
        )
    public["blockers"] = [
        {
            "id": "qk_residual_base_score_preservation_contract_gap",
            "stage": "rtl",
            "status": "active",
            "evidence": BLOCKER,
            "reason": "The frozen hook requires authoritative base-score equality, but the current RTL interface rederives the base term from a dot and Scale32 metadata.",
            "required_resolution": MANAGER_ACTION,
        }
    ]
    public["public_claims"] = [
        {
            "claim": "current Manager-owned stage remains rtl",
            "evidence": ["research/PIPELINE_STATE.json", "research/PUBLIC_STATUS.json"],
        },
        {
            "claim": "post-refreeze rtl.contract-traceability is false before a new candidate started",
            "evidence": [BLOCKER, "design/RTL_MANIFEST.json"],
        },
        {
            "claim": "implementation authorization remains unconsumed",
            "evidence": ["design/FAST_LOOP_POLICY.json", BLOCKER],
        },
        {
            "claim": "accepted prefix, immutable targets, and historical PPA frontier are unchanged",
            "evidence": ["design/PPA_FRONTIER_LEDGER.json", "design/TARGET.json"],
        },
    ]
    public_paths = {
        str(item.get("path"))
        for item in public.get("artifact_hashes", [])
        if isinstance(item, dict)
        and item.get("path")
        and item.get("path") != "research/PUBLIC_STATUS.json"
    }
    public_paths.update(
        {
            "CHECKPOINT.md",
            "design/CHIP_SCOPE.json",
            "design/FAST_LOOP_POLICY.json",
            "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
            "design/SPEC.md",
            "design/TARGET.json",
            BLOCKER,
            BINDER,
            HOOK,
            METADATA,
            "reference/ORACLE_MANIFEST.json",
            "research/PIPELINE_STATE.json",
            RTL,
        }
    )
    public["artifact_hashes"] = [
        artifact(path) for path in sorted(public_paths) if (ROOT / path).is_file()
    ]
    public["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": (
            "UTF-8 sorted keys two-space indentation trailing newline canonical hash null during hash"
        ),
        "canonical_sha256": None,
    }
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump("research/PUBLIC_STATUS.json", public)


def validate() -> None:
    active, findings = audit(require_executable_authorization=False)
    blocker = load(BLOCKER)
    require(blocker.get("decision") == DECISION, "blocker decision is stale")
    require(blocker.get("implementation_authorization_consumed") is False,
            "blocker consumed implementation authorization")
    require(len(blocker.get("findings", [])) == len(findings) == 3,
            "blocker findings are incomplete")
    require(blocker.get("integrity", {}).get("canonical_sha256") == canonical_sha256(blocker),
            "blocker canonical integrity mismatch")
    require(not any(blocker.get("prohibited_runs", {}).values()),
            "blocker falsely records a prohibited run")
    manifest = load("design/RTL_MANIFEST.json")
    require(manifest.get("candidate_status") == STATUS, "manifest blocker status is stale")
    require(manifest.get("candidate_rtl_hash") is None, "manifest started a current candidate")
    require(manifest.get("traceability", {}).get("stage_checklist") == CHECKLIST,
            "manifest RTL checklist differs")
    policy = load("design/FAST_LOOP_POLICY.json")
    for key in (
        "active_repair_authorization",
        "architecture_proposal_authorization",
        "selected_replacement_contract",
    ):
        record = policy[key]
        require(record.get("implementation_authorized") is False,
                f"blocked authorization still executable: {key}")
        require(record.get("operator_approval_consumed") is False,
                f"blocked authorization was consumed: {key}")
        require(record.get("required_manager_action") == MANAGER_ACTION,
                f"blocked routing is stale: {key}")
    public = load("research/PUBLIC_STATUS.json")
    require(public.get("latest_decision") == DECISION, "public decision is stale")
    require(public.get("stage", {}).get("current_stage") == "rtl", "public stage is stale")
    require(public.get("stage", {}).get("current_stage_checklist") == CHECKLIST,
            "public checklist differs")
    require(public.get("supported_layer_operator_prefix") == PREFIX,
            "public supported prefix changed")
    require(public.get("first_unsupported_layer_operator") == FIRST_UNSUPPORTED,
            "public first unsupported operator changed")
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public),
            "public canonical integrity mismatch")
    require("/home/" not in json.dumps(public, sort_keys=True),
            "public status leaks a private path")
    require(active.get("operator_approval_consumed") is False,
            "audit unexpectedly consumed authorization")
    print(
        "QK_RESIDUAL_POST_REFREEZE_CONTRACT_BLOCKER_CHECK_PASS "
        f"contract={CONTRACT} traceability=false authorization_consumed=false"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        bind()
    validate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
