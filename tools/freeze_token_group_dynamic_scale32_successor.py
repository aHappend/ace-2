#!/usr/bin/env python3
"""Freeze the reviewed dynamic-Scale32 successor without advancing stages.

This is a Manager-state binder.  It preserves the sealed QECR predecessor and
the exact certified synthetic-harness chain, keeps implementation unauthorized,
and queues only an environment compatibility rebind/review.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_token_group_dynamic_scale32_v1"
PREDECESSOR = "cross_layer_quantization_error_carry_final_output_v1"
PROPOSAL_SHA256 = "b55ab977574dc2bdeb760e8859ec2fa49f9f835846e3da24bd0f887d84a74f41"
REVIEW_SHA256 = "386a2951ea1f39e3ef51978ced42be72dea8e10c53be188286abbe1507124297"
RECOVERY_SEAL_SHA256 = "5dfd2fd991f3e0237a9ffdc60660bd2df04cd19ffd8f083eb39d306c6a182d3f"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
CHECKLIST = {
    "architecture.compute-memory-model": True,
    "architecture.interface-control": True,
    "architecture.leverage-risk": True,
    "architecture.area-reuse-plan": True,
}
TARGETS = {
    "non_sram_area_cap_mm2": 2.0,
    "frequency_floor_mhz": 100.0,
    "abstract_streaming_memory_boundary_bits": 128,
}
HISTORICAL_PPA = {
    "cells": 62199,
    "non_sram_area_mm2": 0.6108746272,
    "setup_slack_ns_at_100mhz": 0.1502,
    "remaining_area_reserve_mm2": 1.3891253728,
    "candidate_ppa_run": False,
    "status": "accepted_historical_frontier_not_successor_ppa",
}
DOWNSTREAM_RUNS = {
    "paired": "unrun",
    "shell": "unrun",
    "ppa": "unrun",
    "physical": "unrun",
    "prototype": "unrun",
    "benchmark": "unrun",
    "signoff": "unrun",
}

PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
POLICY = ROOT / "design/FAST_LOOP_POLICY.json"
SCOPE = ROOT / "design/CHIP_SCOPE.json"
TARGET = ROOT / "design/TARGET.json"
LIVE = ROOT / ".argus/live-view.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
MAKEFILE = ROOT / "Makefile"
PROPOSAL = ROOT / f"evidence/{CONTRACT}/architecture/PROPOSAL.json"
PROPOSAL_SUM = ROOT / f"evidence/{CONTRACT}/architecture/PROPOSAL.sha256"
REVIEW = ROOT / f"evidence/review/architecture_{CONTRACT}/decision.json"
FREEZE = ROOT / f"evidence/{CONTRACT}/architecture/MANAGER_FREEZE.json"
ARCHIVE = ROOT / f"evidence/{CONTRACT}/architecture/pre_freeze"
PUBLIC_CERT = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1/PUBLIC_CERTIFICATION.json"
HARNESS_REPORT = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1/HARNESS_TEST_REPORT.json"
HARNESS_MANIFEST = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1/MANIFEST.json"
HARNESS_REVIEW = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1/L2_HARNESS_REVIEW.json"
RECOVERY_SEAL = ROOT / (
    "evidence/cross_layer_quantization_error_carry_final_output_v1/"
    "recovery/manager_architecture_rollback_v1/SEAL.json"
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path.relative_to(ROOT)}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_record(record: dict[str, Any], label: str) -> None:
    path = ROOT / str(record.get("path", ""))
    require(path.is_file(), f"{label} missing: {record.get('path')}")
    require(path.stat().st_size == record.get("bytes"), f"{label} byte mismatch")
    require(sha256(path) == record.get("sha256"), f"{label} hash mismatch")


def archive_once(path: Path) -> dict[str, Any]:
    digest = sha256(path)
    destination = ARCHIVE / f"{path.stem}.{digest}{path.suffix}"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    return artifact(destination)


def certified_harness_chain() -> dict[str, Any]:
    cert = load(PUBLIC_CERT)
    require(cert.get("decision") == "PASS", "synthetic harness certification is not PASS")
    require(cert.get("status") == "independently_certified_pass", "synthetic harness certification status differs")
    require(cert.get("candidate_or_model_execution_count") == 0, "harness certification claims execution")
    require(cert.get("implementation_authorized") is False, "harness certification authorizes implementation")
    chain = {
        "public_certification": artifact(PUBLIC_CERT),
        "report": artifact(HARNESS_REPORT),
        "manifest": artifact(HARNESS_MANIFEST),
        "independent_l2_review": artifact(HARNESS_REVIEW),
        "manager_recovery_seal": artifact(RECOVERY_SEAL),
        "protected_artifact_count": cert.get("protected_artifact_count"),
        "protected_manifest_sha256": cert.get("protected_manifest_sha256"),
        "candidate_or_model_execution_count": 0,
        "decision": "PASS",
        "status": "independently_certified_pass",
    }
    expected = {
        "public_certification": "49c8cb699ee559a09f0298ec16a1c4e6168cd812aff11357b7c49c68f9d1d315",
        "report": "88e08ad61aa2f7969ea7e08c74ac4875dc62b915646a618717428513087636b7",
        "manifest": "224dd28455c5cdda567e2b9f499acdb9da8be3ef5c54e228f65661c29f3a3e1e",
        "independent_l2_review": "2fca2c7c8759d82294cc9ec52cdb9d95fb22a163faf29f302d4231ef718e491f",
        "manager_recovery_seal": RECOVERY_SEAL_SHA256,
    }
    for key, digest in expected.items():
        require(chain[key]["sha256"] == digest, f"certified harness chain changed: {key}")
    require(chain["protected_artifact_count"] == 264, "protected harness artifact count changed")
    require(
        chain["protected_manifest_sha256"] == "0845b1a920933986dc1ce08eb014c5d83e15f92e2a88d2412d85927671638209",
        "protected harness manifest hash changed",
    )
    return chain


def next_task() -> dict[str, Any]:
    return {
        "id": f"environment_compatibility_rebind_review_{CONTRACT}",
        "stage": "environment",
        "status": "queued_pending_manager_stage_advance",
        "scope": "environment_compatibility_rebind_and_independent_review_only",
        "stage_closing": True,
        "implementation_authorized": False,
        "candidate_or_model_execution_authorized": False,
        "allowed_work": [
            "rebind_existing_eda_pdk_ip_license_capability_evidence_to_frozen_successor",
            "run_contract_specific_synthetic_environment_compatibility_probe_if_needed",
            "independent_environment_review",
        ],
        "prohibited_work": [
            "rtl_implementation",
            "rtl_simulation_or_formal_for_successor",
            "baseline_or_candidate_quality_execution",
            "shell_regression",
            "canonical_ppa",
            "prototype_benchmark_or_signoff",
        ],
    }


def active_contract(freeze_record: dict[str, Any], predecessor: dict[str, Any]) -> dict[str, Any]:
    proposal = load(PROPOSAL)
    review = load(REVIEW)
    return {
        "id": CONTRACT,
        "contract_id": CONTRACT,
        "mechanism": proposal.get("selection", {}).get("mechanism"),
        "state": "architecture_frozen_environment_compatibility_rebind_review_queued",
        "status": "architecture_frozen_environment_compatibility_rebind_review_queued",
        "authority": "manager_exercising_live_operator_guidance",
        "frozen_at_utc": freeze_record["frozen_at_utc"],
        "manager_freeze": artifact(FREEZE),
        "proposal": artifact(PROPOSAL),
        "proposal_companion": artifact(PROPOSAL_SUM),
        "independent_architecture_review": artifact(REVIEW),
        "architecture_checklist": copy.deepcopy(CHECKLIST),
        "architecture_accepted": review.get("architecture_accepted") is True,
        "successor_frozen": True,
        "implementation_authorized": False,
        "baseline_or_candidate_execution_authorized": False,
        "candidate_capability_accepted": False,
        "candidate_or_model_execution_count": 0,
        "no_execution_claim": True,
        "stage_closing": False,
        "current_stage": "architecture",
        "mode": "ADVANCE",
        "ordered_supported_layer_operator_prefix": copy.deepcopy(PREFIX),
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "targets": copy.deepcopy(TARGETS),
        "historical_ppa_frontier": copy.deepcopy(HISTORICAL_PPA),
        "downstream_runs": copy.deepcopy(DOWNSTREAM_RUNS),
        "certified_harness_chain": certified_harness_chain(),
        "predecessor_contract_id": PREDECESSOR,
        "predecessor": PREDECESSOR,
        "sealed_predecessor": copy.deepcopy(predecessor),
        "same_contract_rtl_authorized": False,
        "successor_freeze_authorized": False,
        "required_next_gate": "environment_compatibility_rebind_and_independent_review",
        "next_action_queue": [next_task()],
        "required_manager_action": "advance_to_environment_for_compatibility_rebind_review_only",
    }


def build_freeze(
    now: str,
    predecessor: dict[str, Any],
    pre_freeze_snapshots: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proposal = load(PROPOSAL)
    review = load(REVIEW)
    freeze = {
        "schema_version": 1,
        "kind": "manager_architecture_successor_freeze",
        "frozen_at_utc": now,
        "authority": "manager_exercising_live_operator_guidance",
        "contract_id": CONTRACT,
        "mechanism": proposal.get("selection", {}).get("mechanism"),
        "current_stage": "architecture",
        "stage_changed": False,
        "status": "architecture_frozen_environment_compatibility_rebind_review_queued",
        "proposal": artifact(PROPOSAL),
        "proposal_companion": artifact(PROPOSAL_SUM),
        "independent_architecture_decision": artifact(REVIEW),
        "architecture_checklist": copy.deepcopy(CHECKLIST),
        "architecture_accepted": review.get("architecture_accepted") is True,
        "implementation_authorized": False,
        "baseline_or_candidate_execution_authorized": False,
        "candidate_or_model_execution_count": 0,
        "no_execution_claim": True,
        "ordered_supported_layer_operator_prefix": copy.deepcopy(PREFIX),
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "mode": "ADVANCE",
        "targets": copy.deepcopy(TARGETS),
        "historical_ppa_frontier": copy.deepcopy(HISTORICAL_PPA),
        "downstream_runs": copy.deepcopy(DOWNSTREAM_RUNS),
        "certified_harness_chain": certified_harness_chain(),
        "sealed_predecessor": copy.deepcopy(predecessor),
        "pre_freeze_snapshots": pre_freeze_snapshots or {
            "pipeline_state": archive_once(PIPELINE),
            "public_status": archive_once(PUBLIC),
        },
        "binder": artifact(Path(__file__).resolve()),
        "project_entrypoint": artifact(MAKEFILE),
        "next_action_queue": [next_task()],
        "claim_boundary": (
            "Manager architecture freeze and environment-compatibility queue only; no RTL, baseline, candidate, "
            "quality, shell, PPA, physical, prototype, benchmark, signoff, tapeout, or silicon execution claim."
        ),
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    require(proposal.get("execution_gates", {}).get("no_execution_claim") is True, "proposal no-execution gate changed")
    freeze["integrity"]["canonical_sha256"] = canonical_sha256(freeze)
    return freeze


def update_pipeline(active: dict[str, Any]) -> None:
    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") == "architecture", "Manager-owned current stage changed")
    pipeline["successor"] = copy.deepcopy(active)
    stages = pipeline.setdefault("stages", {})
    stages.setdefault("definition", {})["status"] = "done"
    stages.setdefault("architecture", {})["status"] = "in_progress_successor_frozen_pending_manager_advance"
    stages.setdefault("environment", {})["status"] = "queued_pending_manager_advance_for_compatibility_rebind_review"
    for name in ("rtl", "verification", "ppa", "prototype", "benchmark", "signoff"):
        stages.setdefault(name, {})["status"] = "locked_pending_predecessor_stage_acceptance"
    dump(PIPELINE, pipeline)


def update_policy(active: dict[str, Any]) -> None:
    policy = load(POLICY)
    require(policy.get("mode") == "ADVANCE", "operator mode changed")
    require(policy.get("area_cap_non_sram_mm2") == 2.0, "operator area cap changed")
    require(policy.get("frequency_floor_mhz") == 100.0, "operator frequency floor changed")
    policy["active_architecture_contract"] = copy.deepcopy(active)
    policy["selected_replacement_contract"] = copy.deepcopy(active)
    next_successor = policy.setdefault("next_successor", {})
    next_successor.update({
        "status": "architecture_frozen",
        "contract_id": CONTRACT,
        "successor_frozen": True,
        "architecture_checklist_closed": True,
        "implementation_authorized": False,
        "successor_selection_authorized": False,
        "manager_freeze": artifact(FREEZE),
        "next_action_queue": [next_task()],
        "required_manager_action": "advance_to_environment_for_compatibility_rebind_review_only",
    })
    policy["manager_recommendation"] = "advance_to_environment_for_compatibility_rebind_review_only"
    policy["environment_compatibility_queue"] = [next_task()]
    gate = policy.setdefault("baseline_harness_hardening_gate", {})
    gate["successor_frozen"] = True
    gate["successor_freeze_permitted"] = False
    dump(POLICY, policy)


def update_scope(active: dict[str, Any], now: str) -> None:
    scope = load(SCOPE)
    require(scope.get("current_stage") == "architecture", "CHIP_SCOPE stage changed")
    policy = scope.setdefault("operator_owned_execution_policy", {})
    policy["active_successor_contract"] = copy.deepcopy(active)
    policy["selected_replacement_contract"] = copy.deepcopy(active)
    scope["implementation_frontier"] = copy.deepcopy(active)
    scope["latest_decision"] = "shared_token_group_dynamic_scale32_v1_frozen_environment_rebind_review_queued"
    scope["required_manager_action"] = "advance_to_environment_for_compatibility_rebind_review_only"
    scope["routing_status"] = "architecture_successor_frozen_environment_rebind_review_queued"
    scope["last_updated_utc"] = now
    stage = scope.setdefault("stage", {})
    stage.update({
        "current_stage": "architecture",
        "current_stage_status": "successor_frozen_pending_manager_advance_to_environment",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_evidence": [
            PROPOSAL.relative_to(ROOT).as_posix(),
            REVIEW.relative_to(ROOT).as_posix(),
            FREEZE.relative_to(ROOT).as_posix(),
        ],
        "stage_closing": False,
        "stage_transition_owner": "Manager",
    })
    scope["stage_closing"] = False
    dump(SCOPE, scope)


def update_target(active: dict[str, Any], now: str) -> None:
    target = load(TARGET)
    target["current_architecture_contract"] = copy.deepcopy(active)
    fast = target.setdefault("fast_loop_contract", {})
    fast["selected_replacement_contract"] = copy.deepcopy(active)
    fast["active_repair_authorization"] = copy.deepcopy(active)
    fast["next_action_queue"] = [next_task()]
    target["generated_at_utc"] = now
    dump(TARGET, target)


def update_checkpoint(active: dict[str, Any]) -> None:
    text = f"""# Goal

Hold the exact independently accepted `shared_token_group_dynamic_scale32_v1`
architecture as the active Manager-frozen successor while preserving the sealed
QECR predecessor and queuing only environment compatibility rebind/review.

# Current state

The Manager froze `shared_token_group_dynamic_scale32_v1` at
`{active['frozen_at_utc']}` using proposal SHA-256
`{PROPOSAL_SHA256}` and the accepted independent architecture decision
`evidence/review/architecture_shared_token_group_dynamic_scale32_v1/decision.json`.
The Manager-owned stage remains `architecture`; `stage_closing=false`,
`implementation_authorized=false`, and the freeze makes no execution claim.

The predecessor `cross_layer_quantization_error_carry_final_output_v1` remains
sealed by `evidence/cross_layer_quantization_error_carry_final_output_v1/recovery/manager_architecture_rollback_v1/SEAL.json`
with SHA-256 `{RECOVERY_SEAL_SHA256}`. The independently certified synthetic
baseline harness chain is preserved exactly, including all 264 protected
artifacts and zero candidate/model executions.

# Preserved contracts

- Ordered supported prefix: through `layer_0.v_proj`.
- First unsupported operator: `layer_0.rope_q`.
- Mode: `ADVANCE`.
- Immutable targets: 2.0 mm2 non-SRAM cap, 100 MHz floor, 128-bit boundary.
- Historical frontier only: 62,199 cells, 0.6108746272 mm2 non-SRAM,
  +0.1502 ns setup slack at 100 MHz; no successor PPA exists.
- Paired/shell/PPA/physical/prototype/benchmark/signoff remain unrun.

# Next queued action

Exactly one task is queued: environment compatibility rebind plus fresh
independent environment review. It is not executable until the Manager advances
the stage to `environment`. RTL implementation, RTL/quality execution, shell,
canonical PPA, prototype, benchmark, and signoff remain locked.
"""
    CHECKPOINT.write_text(text, encoding="utf-8")


def refresh_public_container(container: dict[str, Any], active: dict[str, Any]) -> None:
    container.update({
        "current_stage": "architecture",
        "current_mode": "ADVANCE",
        "ordered_supported_layer_operator_prefix": copy.deepcopy(PREFIX),
        "supported_layer_operator_prefix": copy.deepcopy(PREFIX),
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "latest_decision": "shared_token_group_dynamic_scale32_v1_frozen_environment_rebind_review_queued",
        "required_manager_action": "advance_to_environment_for_compatibility_rebind_review_only",
        "required_operator_action": "none",
        "routing_status": "architecture_successor_frozen_environment_rebind_review_queued",
        "candidate_mechanism": copy.deepcopy(active),
        "selected_replacement_contract": copy.deepcopy(active),
        "next_action_queue": [next_task()],
    })
    operator_policy = container.setdefault("operator_policy", {})
    operator_policy["active_successor_contract"] = copy.deepcopy(active)
    operator_policy["selected_replacement_contract"] = copy.deepcopy(active)


def update_public(active: dict[str, Any], now: str) -> None:
    public = load(PUBLIC)
    public.update({
        "current_stage": "architecture",
        "current_mode": "ADVANCE",
        "ordered_supported_layer_operator_prefix": copy.deepcopy(PREFIX),
        "supported_layer_operator_prefix": copy.deepcopy(PREFIX),
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "selected_replacement_contract": copy.deepcopy(active),
        "latest_decision": "shared_token_group_dynamic_scale32_v1_frozen_environment_rebind_review_queued",
        "required_manager_action": "advance_to_environment_for_compatibility_rebind_review_only",
        "required_operator_action": "none",
        "routing_status": "architecture_successor_frozen_environment_rebind_review_queued",
        "routing_authorized": "manager_advance_to_environment_only",
        "stage_closing": False,
        "next_action_queue": [next_task()],
        "generated_at_utc": now,
        "last_updated_utc": now,
    })
    dispatch = public.setdefault("architecture_successor_dispatch", {})
    dispatch.update({
        "proposal_id": CONTRACT,
        "proposal_status": "independently_accepted_and_manager_frozen",
        "status": "architecture_frozen_environment_compatibility_rebind_review_queued",
        "current_stage": "architecture",
        "architecture_accepted": True,
        "successor_frozen": True,
        "implementation_authorized": False,
        "stage_closing": False,
        "proposal_evidence": artifact(PROPOSAL),
        "independent_architecture_review": artifact(REVIEW),
        "manager_freeze": artifact(FREEZE),
        "sealed_contract_id": PREDECESSOR,
        "required_manager_action": "advance_to_environment_for_compatibility_rebind_review_only",
        "next_action_queue": [next_task()],
    })
    proposal_projection = public.setdefault("architecture_successor_proposal", {})
    proposal_projection.update({
        "status": "planner_proposal_independently_accepted_and_manager_frozen",
        "successor_frozen": True,
        "implementation_authorized": False,
        "stage_closing": False,
        "manager_disposition": artifact(FREEZE),
        "required_manager_action": "advance_to_environment_for_compatibility_rebind_review_only",
    })
    review_projection = public.setdefault("architecture_successor_review", {})
    review_projection.update({
        "status": "independently_accepted_then_manager_frozen",
        "successor_frozen": True,
        "implementation_authorized": False,
        "stage_closing": False,
        "manager_disposition": artifact(FREEZE),
        "required_manager_action": "advance_to_environment_for_compatibility_rebind_review_only",
    })
    for name in ("architecture_certification_mission", "architecture_proposal_gate"):
        gate = public.setdefault(name, {})
        gate.update({
            "status": "architecture_frozen_environment_compatibility_rebind_review_queued",
            "current_successor": True,
            "successor_frozen": True,
            "implementation_authorized": False,
            "stage_closing": False,
            "checklist": copy.deepcopy(CHECKLIST),
            "required_manager_action": "advance_to_environment_for_compatibility_rebind_review_only",
        })
    stage = public.setdefault("stage", {})
    stage.update({
        "checklist_scope": CONTRACT,
        "completed_prior_stages": ["definition"],
        "current_stage": "architecture",
        "current_stage_status": "successor_frozen_pending_manager_advance_to_environment",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_evidence": [
            PROPOSAL.relative_to(ROOT).as_posix(),
            REVIEW.relative_to(ROOT).as_posix(),
            FREEZE.relative_to(ROOT).as_posix(),
        ],
        "downstream_stages_locked_until_manager_advance": [
            "environment", "rtl", "verification", "ppa", "prototype", "benchmark", "signoff"
        ],
        "planner_may_advance_stage": False,
        "stage_closing": False,
        "stage_transition_owner": "Manager",
    })
    blockers = [
        item for item in public.get("blockers", [])
        if item.get("id") not in {
            "manager_structurally_distinct_successor_freeze_required",
            "manager_successor_freeze_required",
            "architecture_successor_freeze_pending",
            "manager_stage_advance_pending",
        }
    ]
    blockers.append({
        "id": "manager_stage_advance_pending",
        "severity": "gate",
        "status": "open",
        "owner": "Manager",
        "detail": "Successor is frozen; advance only to environment for compatibility rebind and independent review.",
    })
    public["blockers"] = blockers
    for name in ("dashboard_fields", "implementation_frontier"):
        container = public.setdefault(name, {})
        refresh_public_container(container, active)
        container["architecture_successor_proposal"] = copy.deepcopy(proposal_projection)
        container["architecture_successor_review"] = copy.deepcopy(review_projection)
        performance = container.setdefault("current_architecture_performance_model", {})
        performance.update({
            "contract_id": CONTRACT,
            "successor_frozen": True,
            "implementation_authorized": False,
            "stage_closing": False,
            "required_manager_action": "advance_to_environment_for_compatibility_rebind_review_only",
            "manager_freeze": artifact(FREEZE),
        })
        container.setdefault("operator_policy", {})["manager_recommendation"] = (
            "advance_to_environment_for_compatibility_rebind_review_only"
        )
    records = {
        item.get("path"): item
        for item in public.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path") and (ROOT / item["path"]).is_file()
    }
    for path in (
        PIPELINE, POLICY, SCOPE, TARGET, CHECKPOINT, MAKEFILE, PROPOSAL, PROPOSAL_SUM,
        REVIEW, FREEZE, PUBLIC_CERT, HARNESS_REPORT, HARNESS_MANIFEST,
        HARNESS_REVIEW, RECOVERY_SEAL, Path(__file__).resolve(),
    ):
        records[path.relative_to(ROOT).as_posix()] = artifact(path)
    public["artifact_hashes"] = [records[key] for key in sorted(records)]
    public.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    public["integrity"]["canonical_sha256"] = None
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)


def update_live(active: dict[str, Any], now: str) -> None:
    live = load(LIVE)
    live.update({
        "title": "ACE-2 frozen architecture successor",
        "current_stage": "architecture",
        "current_mode": "ADVANCE",
        "proposal_id": CONTRACT,
        "proposal_status": "independently_accepted_and_manager_frozen",
        "proposal_evidence": PROPOSAL.relative_to(ROOT).as_posix(),
        "architecture_review_evidence": REVIEW.relative_to(ROOT).as_posix(),
        "architecture_review_status": "accepted",
        "manager_freeze_evidence": FREEZE.relative_to(ROOT).as_posix(),
        "successor_frozen": True,
        "implementation_authorized": False,
        "ordered_supported_layer_operator_prefix": copy.deepcopy(PREFIX),
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "latest_decision": "shared_token_group_dynamic_scale32_v1_frozen_environment_rebind_review_queued",
        "reason": "Manager freeze complete; only environment compatibility rebind/review is queued",
        "sealed_contract_id": PREDECESSOR,
        "next_action_queue": [next_task()],
        "generated_at_utc": now,
    })
    paths = set(live.get("paths", []))
    paths.update({
        PIPELINE.relative_to(ROOT).as_posix(),
        PUBLIC.relative_to(ROOT).as_posix(),
        POLICY.relative_to(ROOT).as_posix(),
        SCOPE.relative_to(ROOT).as_posix(),
        TARGET.relative_to(ROOT).as_posix(),
        CHECKPOINT.relative_to(ROOT).as_posix(),
        FREEZE.relative_to(ROOT).as_posix(),
    })
    live["paths"] = sorted(paths)
    dump(LIVE, live)


def bind() -> None:
    require(sha256(PROPOSAL) == PROPOSAL_SHA256, "exact proposal SHA-256 changed")
    require(PROPOSAL_SUM.read_text(encoding="utf-8").strip().split() == [PROPOSAL_SHA256, PROPOSAL.name], "proposal companion differs")
    require(sha256(REVIEW) == REVIEW_SHA256, "accepted architecture decision changed")
    require(sha256(RECOVERY_SEAL) == RECOVERY_SEAL_SHA256, "Manager recovery seal changed")
    review = load(REVIEW)
    require(review.get("architecture_accepted") is True and review.get("status") == "accepted", "architecture review is not accepted")
    require(review.get("proposal", {}).get("sha256") == PROPOSAL_SHA256, "architecture review proposal binding differs")
    require(review.get("stage_checklist") == CHECKLIST, "architecture review checklist differs")
    require(review.get("implementation_authorized") is False, "architecture review authorizes implementation")
    require(review.get("no_execution_claim") is True, "architecture review makes an execution claim")

    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") == "architecture", "Manager-owned stage is not architecture")
    existing = pipeline.get("successor", {})
    if (existing.get("id") == CONTRACT or existing.get("contract_id") == CONTRACT) and FREEZE.is_file():
        predecessor = copy.deepcopy(existing.get("sealed_predecessor", {}))
        prior_freeze = load(FREEZE)
        now = prior_freeze.get("frozen_at_utc")
        pre_freeze_snapshots = copy.deepcopy(prior_freeze.get("pre_freeze_snapshots", {}))
    else:
        require(existing.get("id") == PREDECESSOR, "active predecessor is not the sealed QECR contract")
        require(existing.get("state") == "sealed_terminal_baseline_protocol_no_go", "QECR predecessor is not sealed")
        require(existing.get("seal_evidence", {}).get("sha256") == RECOVERY_SEAL_SHA256, "QECR predecessor seal differs")
        predecessor = copy.deepcopy(existing)
        now = utc_now()
        pre_freeze_snapshots = None
    require(isinstance(now, str) and now.endswith("Z"), "freeze timestamp is invalid")

    freeze = build_freeze(now, predecessor, pre_freeze_snapshots)
    dump(FREEZE, freeze)
    active = active_contract(freeze, predecessor)
    update_pipeline(active)
    update_policy(active)
    update_scope(active, now)
    update_target(active, now)
    update_checkpoint(active)
    update_public(active, now)
    update_live(active, now)


def validate() -> None:
    require(sha256(PROPOSAL) == PROPOSAL_SHA256, "exact proposal SHA-256 changed")
    require(sha256(REVIEW) == REVIEW_SHA256, "accepted architecture decision changed")
    require(sha256(RECOVERY_SEAL) == RECOVERY_SEAL_SHA256, "predecessor recovery seal changed")
    freeze = load(FREEZE)
    require(freeze.get("contract_id") == CONTRACT, "freeze contract differs")
    require(
        freeze.get("mechanism") == "token_and_tensor_group_dynamic_power_of_two_Scale32_activation_scaling",
        "freeze mechanism differs",
    )
    require(freeze.get("current_stage") == "architecture" and freeze.get("stage_changed") is False, "freeze changes stage")
    require(freeze.get("implementation_authorized") is False, "freeze authorizes implementation")
    require(freeze.get("no_execution_claim") is True, "freeze makes execution claim")
    require(freeze.get("architecture_checklist") == CHECKLIST, "freeze checklist differs")
    require(freeze.get("integrity", {}).get("canonical_sha256") == canonical_sha256(freeze), "freeze canonical hash differs")
    require(freeze.get("next_action_queue") == [next_task()], "freeze queue is not the single environment task")
    for name, record in freeze.get("certified_harness_chain", {}).items():
        if isinstance(record, dict) and "path" in record:
            verify_record(record, f"freeze harness chain {name}")
    for name, record in freeze.get("pre_freeze_snapshots", {}).items():
        verify_record(record, f"pre-freeze snapshot {name}")

    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") == "architecture", "PIPELINE_STATE stage changed")
    active = pipeline.get("successor", {})
    require(active.get("id") == CONTRACT, "PIPELINE_STATE successor id differs")
    require(active.get("contract_id") == CONTRACT, "PIPELINE_STATE active successor differs")
    require(
        active.get("mechanism") == "token_and_tensor_group_dynamic_power_of_two_Scale32_activation_scaling",
        "PIPELINE_STATE successor mechanism differs",
    )
    require(active.get("successor_frozen") is True, "PIPELINE_STATE successor is not frozen")
    require(active.get("implementation_authorized") is False, "PIPELINE_STATE authorizes implementation")
    require(active.get("no_execution_claim") is True, "PIPELINE_STATE makes execution claim")
    require(active.get("next_action_queue") == [next_task()], "PIPELINE_STATE queue differs")
    require(active.get("sealed_predecessor", {}).get("id") == PREDECESSOR, "sealed predecessor identity lost")
    require(active.get("sealed_predecessor", {}).get("seal_evidence", {}).get("sha256") == RECOVERY_SEAL_SHA256, "sealed predecessor evidence lost")
    require(active.get("ordered_supported_layer_operator_prefix") == PREFIX, "supported prefix changed")
    require(active.get("first_unsupported_layer_operator") == FIRST_UNSUPPORTED, "first unsupported operator changed")
    require(active.get("targets") == TARGETS, "operator targets changed")
    require(active.get("historical_ppa_frontier") == HISTORICAL_PPA, "historical PPA frontier changed")
    require(active.get("downstream_runs") == DOWNSTREAM_RUNS, "downstream unrun locks changed")
    require(pipeline.get("stages", {}).get("environment", {}).get("status") == "queued_pending_manager_advance_for_compatibility_rebind_review", "environment task not queued")
    for name in ("rtl", "verification", "ppa", "prototype", "benchmark", "signoff"):
        require(str(pipeline.get("stages", {}).get(name, {}).get("status", "")).startswith("locked_"), f"{name} is not locked")

    policy = load(POLICY)
    require(policy.get("active_architecture_contract", {}).get("contract_id") == CONTRACT, "FAST_LOOP_POLICY active contract differs")
    require(policy.get("next_successor", {}).get("successor_frozen") is True, "FAST_LOOP_POLICY freeze projection differs")
    require(policy.get("environment_compatibility_queue") == [next_task()], "FAST_LOOP_POLICY queue differs")
    require(policy.get("mode") == "ADVANCE" and policy.get("area_cap_non_sram_mm2") == 2.0 and policy.get("frequency_floor_mhz") == 100.0, "operator policy targets changed")

    public = load(PUBLIC)
    require(public.get("current_stage") == "architecture", "PUBLIC_STATUS stage changed")
    require(public.get("selected_replacement_contract", {}).get("contract_id") == CONTRACT, "PUBLIC_STATUS successor differs")
    require(public.get("selected_replacement_contract", {}).get("implementation_authorized") is False, "PUBLIC_STATUS authorizes implementation")
    require(public.get("stage", {}).get("current_stage_checklist") == CHECKLIST, "PUBLIC_STATUS architecture checklist differs")
    require(public.get("architecture_successor_review", {}).get("successor_frozen") is True, "PUBLIC_STATUS review disposition is stale")
    require(public.get("next_action_queue") == [next_task()], "PUBLIC_STATUS queue differs")
    require(public.get("latest_ppa_frontier", {}).get("candidate_ppa_run") is False, "PUBLIC_STATUS claims successor PPA")
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public), "PUBLIC_STATUS canonical hash differs")
    for index, record in enumerate(public.get("artifact_hashes", [])):
        verify_record(record, f"PUBLIC_STATUS artifact_hashes[{index}]")
    for name in ("dashboard_fields", "implementation_frontier"):
        container = public.get(name, {})
        require(container.get("candidate_mechanism", {}).get("contract_id") == CONTRACT, f"{name} successor differs")
        require(container.get("first_unsupported_layer_operator") == FIRST_UNSUPPORTED, f"{name} boundary differs")
        require(container.get("current_mode") == "ADVANCE", f"{name} mode differs")

    scope = load(SCOPE)
    target = load(TARGET)
    live = load(LIVE)
    require(scope.get("operator_owned_execution_policy", {}).get("active_successor_contract", {}).get("contract_id") == CONTRACT, "CHIP_SCOPE successor differs")
    require(scope.get("stage_closing") is False, "CHIP_SCOPE closes architecture stage")
    require(target.get("current_architecture_contract", {}).get("contract_id") == CONTRACT, "TARGET successor differs")
    require(live.get("successor_frozen") is True and live.get("implementation_authorized") is False, "live view freeze differs")
    require(live.get("next_action_queue") == [next_task()], "live view queue differs")
    checkpoint = CHECKPOINT.read_text(encoding="utf-8")
    require(PROPOSAL_SHA256 in checkpoint and "environment compatibility rebind" in checkpoint, "CHECKPOINT freeze summary differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        bind()
    validate()
    print(
        "ACE2_TOKEN_GROUP_DYNAMIC_SCALE32_MANAGER_FREEZE_CHECK_PASS "
        f"contract={CONTRACT} proposal_sha256={PROPOSAL_SHA256} "
        "current_stage=architecture implementation_authorized=false "
        "next=environment_compatibility_rebind_review_only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
