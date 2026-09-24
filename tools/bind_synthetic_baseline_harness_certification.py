#!/usr/bin/env python3
"""Bind the independently certified synthetic baseline harness without changing stage."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "cross_layer_quantization_error_carry_final_output_v1"
EVIDENCE_ID = "candidate_independent_synthetic_baseline_harness_v1"
EVIDENCE_DIR = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1"
REPORT = EVIDENCE_DIR / "HARNESS_TEST_REPORT.json"
REPORT_SUM = EVIDENCE_DIR / "HARNESS_TEST_REPORT.sha256"
MANIFEST = EVIDENCE_DIR / "MANIFEST.json"
MANIFEST_SUM = EVIDENCE_DIR / "MANIFEST.sha256"
REVIEW = EVIDENCE_DIR / "L2_HARNESS_REVIEW.json"
REVIEW_SUM = EVIDENCE_DIR / "L2_HARNESS_REVIEW.sha256"
PROTECTED_BEFORE = EVIDENCE_DIR / "PROTECTED_BEFORE.json"
PROTECTED_BEFORE_SUM = EVIDENCE_DIR / "PROTECTED_BEFORE.sha256"
PROTECTED_AFTER = EVIDENCE_DIR / "PROTECTED_AFTER.json"
PROTECTED_AFTER_SUM = EVIDENCE_DIR / "PROTECTED_AFTER.sha256"
PUBLIC_CERTIFICATION = EVIDENCE_DIR / "PUBLIC_CERTIFICATION.json"
PUBLIC_CERTIFICATION_SUM = EVIDENCE_DIR / "PUBLIC_CERTIFICATION.sha256"
RECOVERY_SEAL = (
    ROOT
    / "evidence/cross_layer_quantization_error_carry_final_output_v1/recovery/"
    "manager_architecture_rollback_v1/SEAL.json"
)
RECOVERY_SEAL_SUM = RECOVERY_SEAL.with_suffix(".sha256")
SEAL_CHECKER = ROOT / "tools/seal_cross_layer_baseline_recovery.py"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
POLICY = ROOT / "design/FAST_LOOP_POLICY.json"
SCOPE = ROOT / "design/CHIP_SCOPE.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
LIVE_VIEW = ROOT / ".argus/live-view.json"
BINDER = Path(__file__).resolve()

PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
ARCHITECTURE_CHECKLIST = {
    "architecture.compute-memory-model": False,
    "architecture.interface-control": False,
    "architecture.leverage-risk": False,
    "architecture.area-reuse-plan": False,
}
EXPECTED_PROPERTIES = [
    "atomic_artifact_and_companion_hash_commit",
    "exact_provenance_binding",
    "durable_exactly_one_reservation_and_run_ledger",
    "stage_authorization_before_execution",
    "candidate_entrypoint_interlock",
    "crash_recovery",
    "missing_or_mismatched_artifact_fail_closed",
]
DECISION = "synthetic_baseline_harness_independently_certified"
ROUTING_STATUS = "architecture_waiting_manager_structurally_distinct_successor_freeze"
MANAGER_ACTION = "freeze_structurally_distinct_successor_or_record_architecture_no_go"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def dump(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_companion(payload: Path, companion: Path) -> None:
    require(payload.is_file(), f"missing payload: {payload}")
    require(companion.is_file(), f"missing companion hash: {companion}")
    fields = companion.read_text(encoding="utf-8").strip().split()
    require(len(fields) == 2, f"invalid companion hash format: {companion}")
    require(fields[1] == payload.name, f"companion names wrong payload: {companion}")
    require(fields[0] == sha256(payload), f"companion hash mismatch: {companion}")


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def write_public_certification(value: dict[str, Any]) -> None:
    dump(PUBLIC_CERTIFICATION, value)
    temporary = PUBLIC_CERTIFICATION_SUM.with_suffix(PUBLIC_CERTIFICATION_SUM.suffix + ".tmp")
    temporary.write_text(
        f"{sha256(PUBLIC_CERTIFICATION)}  {PUBLIC_CERTIFICATION.name}\n",
        encoding="utf-8",
    )
    os.replace(temporary, PUBLIC_CERTIFICATION_SUM)


def verify_manifest() -> None:
    manifest = load(MANIFEST)
    require(manifest.get("evidence_id") == EVIDENCE_ID, "manifest evidence ID differs")
    entries = manifest.get("artifacts")
    require(isinstance(entries, list) and entries, "manifest artifact list is empty")
    seen: set[str] = set()
    for entry in entries:
        require(isinstance(entry, dict), "manifest entry is not an object")
        name = entry.get("path")
        require(isinstance(name, str) and name not in seen, "manifest path is missing or duplicated")
        seen.add(name)
        path = EVIDENCE_DIR / name
        require(path.is_file(), f"manifest artifact is missing: {name}")
        require(entry.get("bytes") == path.stat().st_size, f"manifest byte count differs: {name}")
        require(entry.get("sha256") == sha256(path), f"manifest hash differs: {name}")


def verify_protected_manifest_current(value: dict[str, Any]) -> None:
    entries = value.get("artifacts")
    require(isinstance(entries, list), "protected artifact list is missing")
    require(value.get("artifact_count") == len(entries), "protected artifact count is inconsistent")
    for entry in entries:
        require(isinstance(entry, dict), "protected manifest entry is not an object")
        relative = entry.get("path")
        require(isinstance(relative, str), "protected manifest path is missing")
        path = ROOT / relative
        require(path.is_file(), f"protected artifact is missing: {relative}")
        require(entry.get("bytes") == path.stat().st_size, f"protected byte count differs: {relative}")
        require(entry.get("sha256") == sha256(path), f"protected hash differs: {relative}")


def run_seal_check() -> None:
    completed = subprocess.run(
        [sys.executable, str(SEAL_CHECKER), "--check"],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    require(completed.returncode == 0, "recovery seal check failed: " + completed.stdout + completed.stderr)
    require("ACE2_BASELINE_RECOVERY_SEAL_CHECK_PASS" in completed.stdout, "seal PASS marker missing")


def validate_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for payload, companion in (
        (REPORT, REPORT_SUM),
        (MANIFEST, MANIFEST_SUM),
        (REVIEW, REVIEW_SUM),
        (PROTECTED_BEFORE, PROTECTED_BEFORE_SUM),
        (PROTECTED_AFTER, PROTECTED_AFTER_SUM),
        (RECOVERY_SEAL, RECOVERY_SEAL_SUM),
    ):
        verify_companion(payload, companion)
    verify_manifest()
    run_seal_check()

    pipeline = load(PIPELINE)
    report = load(REPORT)
    review = load(REVIEW)
    before = load(PROTECTED_BEFORE)
    after = load(PROTECTED_AFTER)

    require(pipeline.get("current_stage") == "architecture", "Manager-owned current_stage is not architecture")
    require(report.get("evidence_id") == EVIDENCE_ID, "report evidence ID differs")
    require(report.get("decision") == "PASS", "harness report is not PASS")
    require(report.get("stage") == "architecture", "harness report stage differs")
    require(report.get("scope") == "candidate-independent synthetic/dry-fixture harness only", "harness scope differs")
    require(report.get("contract_disposition") == "failed mechanism remains sealed; successor freeze unauthorized", "contract disposition differs")
    properties = report.get("properties", {})
    require(sorted(properties) == sorted(EXPECTED_PROPERTIES), "harness property set differs")
    for name in EXPECTED_PROPERTIES:
        require(properties[name].get("status") == "PASS", f"harness property did not pass: {name}")
    boundary = report.get("execution_boundary", {})
    require(boundary.get("arbitrary_subprocess_surface_in_harness") is False, "arbitrary subprocess surface exists")
    require(boundary.get("candidate_entrypoint_available") is False, "candidate entrypoint is available")
    require(boundary.get("candidate_or_model_execution_count") == 0, "candidate/model execution occurred")
    require(boundary.get("rtl_reference_vector_precheck_execution_count") == 0, "protected precheck execution occurred")
    require(boundary.get("successor_frozen") is False, "report claims a successor was frozen")
    protected = report.get("protected_state", {})
    require(before == after, "protected before/after manifests differ")
    verify_protected_manifest_current(after)
    require(protected.get("unchanged") is True, "report does not mark protected state unchanged")
    require(protected.get("artifact_count") == before.get("artifact_count"), "protected artifact count differs")
    require(protected.get("before_manifest_sha256") == before.get("manifest_sha256"), "before digest differs")
    require(protected.get("after_manifest_sha256") == after.get("manifest_sha256"), "after digest differs")
    require(report.get("manager_recovery_seal", {}).get("sha256") == sha256(RECOVERY_SEAL), "report seal hash differs")

    require(review.get("decision") == "PASS", "independent review is not PASS")
    require(review.get("reviewer_role") == "fresh_independent_l2_reviewer", "reviewer role differs")
    require(review.get("reviewed_report", {}).get("sha256") == sha256(REPORT), "review report hash differs")
    require(review.get("reviewed_manifest", {}).get("sha256") == sha256(MANIFEST), "review manifest hash differs")
    checks = review.get("acceptance_checks", {})
    require(checks and all(value is True for value in checks.values()), "not all independent acceptance checks passed")
    require(review.get("findings") == [], "independent review has findings")
    require(review.get("remediation") == [], "independent review has remediation")
    return report, review, pipeline


def certification(report: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "status": "independently_certified_pass",
        "decision": "PASS",
        "certified_at_utc": review["reviewed_at_utc"],
        "scope": report["scope"],
        "required_properties": EXPECTED_PROPERTIES,
        "independent_acceptance_check_count": len(review["acceptance_checks"]),
        "protected_artifact_count": report["protected_state"]["artifact_count"],
        "protected_manifest_sha256": report["protected_state"]["after_manifest_sha256"],
        "candidate_or_model_execution_count": 0,
        "failed_contract_id": CONTRACT,
        "failed_contract_status": "sealed_terminal_baseline_protocol_no_go",
        "same_contract_repair_or_execution_permitted": False,
        "successor_freeze_precondition_satisfied": True,
        "successor_frozen": False,
        "successor_freeze_authority": "Manager",
        "implementation_authorized": False,
        "required_manager_action": MANAGER_ACTION,
        "report": artifact(REPORT),
        "manifest": artifact(MANIFEST),
        "independent_l2_review": artifact(REVIEW),
        "manager_recovery_seal": artifact(RECOVERY_SEAL),
        "binder": artifact(BINDER),
    }


def public_projection(cert: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "project": "ACE-2",
        "evidence_id": cert["evidence_id"],
        "status": cert["status"],
        "decision": cert["decision"],
        "certified_at_utc": cert["certified_at_utc"],
        "scope": cert["scope"],
        "required_properties": cert["required_properties"],
        "independent_acceptance_check_count": cert["independent_acceptance_check_count"],
        "protected_artifact_count": cert["protected_artifact_count"],
        "protected_manifest_sha256": cert["protected_manifest_sha256"],
        "candidate_or_model_execution_count": 0,
        "failed_contract_id": CONTRACT,
        "failed_contract_status": cert["failed_contract_status"],
        "successor_freeze_precondition_satisfied": True,
        "successor_frozen": False,
        "successor_freeze_authority": "Manager",
        "implementation_authorized": False,
        "required_manager_action": MANAGER_ACTION,
        "evidence_hashes": {
            "report_sha256": cert["report"]["sha256"],
            "manifest_sha256": cert["manifest"]["sha256"],
            "independent_l2_review_sha256": cert["independent_l2_review"]["sha256"],
            "manager_recovery_seal_sha256": cert["manager_recovery_seal"]["sha256"],
        },
        "claim_boundary": (
            "Synthetic harness certification only; no baseline, candidate, model, RTL, verification, "
            "PPA, prototype, benchmark, signoff, tapeout, or silicon execution claim."
        ),
    }


def update_failed_contract(value: Any, cert: dict[str, Any], now: str) -> None:
    if not isinstance(value, dict) or value.get("contract_id") != CONTRACT:
        return
    value["status"] = "sealed_terminal_baseline_protocol_no_go"
    value["candidate_capability_accepted"] = False
    value["implementation_authorized"] = False
    value["successor_selection_authorized"] = False
    value["required_manager_action"] = MANAGER_ACTION
    value["baseline_harness_certification"] = copy.deepcopy(cert)
    value["updated_at_utc"] = now


def update_all_failed_contract_copies(value: Any, cert: dict[str, Any], now: str) -> None:
    if isinstance(value, dict):
        if value.get("contract_id") == CONTRACT:
            update_failed_contract(value, cert, now)
        for child in value.values():
            update_all_failed_contract_copies(child, cert, now)
    elif isinstance(value, list):
        for child in value:
            update_all_failed_contract_copies(child, cert, now)


def update_policy(cert: dict[str, Any]) -> None:
    policy = load(POLICY)
    gate = policy.setdefault("baseline_harness_hardening_gate", {})
    gate.update({
        "status": "independently_certified_pass",
        "candidate_independent": True,
        "dry_fixture_only": True,
        "candidate_or_model_execution_permitted": False,
        "candidate_entrypoint_permitted": False,
        "independent_review_required": True,
        "independent_review_satisfied": True,
        "successor_freeze_precondition_satisfied": True,
        "successor_freeze_authority": "Manager",
        "successor_frozen": False,
        "required_properties": EXPECTED_PROPERTIES,
        "certification": copy.deepcopy(cert),
    })
    policy["manager_recommendation"] = MANAGER_ACTION
    policy["next_successor"] = {
        "status": "not_frozen",
        "required_distinction": "structurally_distinct_from_cross_layer_quantization_error_carry_final_output_v1",
        "selection_authority": "Manager",
        "implementation_authorized": False,
        "architecture_checklist_closed": False,
        "baseline_harness_precondition_satisfied": True,
        "required_manager_action": MANAGER_ACTION,
    }
    update_all_failed_contract_copies(policy, cert, cert["certified_at_utc"])
    dump(POLICY, policy)


def update_scope(cert: dict[str, Any], now: str) -> None:
    scope = load(SCOPE)
    scope["current_stage"] = "architecture"
    scope["last_updated_utc"] = now
    scope["latest_decision"] = DECISION
    scope["required_manager_action"] = MANAGER_ACTION
    scope["routing_status"] = ROUTING_STATUS
    scope["stage_closing"] = False
    scope["baseline_harness_certification"] = copy.deepcopy(cert)
    stage = scope.setdefault("stage", {})
    stage.update({
        "current_stage": "architecture",
        "current_stage_status": ROUTING_STATUS,
        "current_stage_checklist": copy.deepcopy(ARCHITECTURE_CHECKLIST),
        "checklist_scope": "next_structurally_distinct_successor_not_yet_frozen",
        "current_stage_evidence": [
            cert["report"]["path"],
            cert["manifest"]["path"],
            cert["independent_l2_review"]["path"],
            cert["manager_recovery_seal"]["path"],
        ],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
        "stage_closing": False,
    })
    frontier = scope.setdefault("implementation_frontier", {})
    frontier["latest_decision"] = DECISION
    frontier["required_manager_action"] = MANAGER_ACTION
    frontier["candidate_meets_numeric_acceptance"] = False
    frontier["baseline_harness_certification"] = copy.deepcopy(cert)
    update_all_failed_contract_copies(scope, cert, now)
    dump(SCOPE, scope)


def update_public(cert: dict[str, Any], now: str) -> None:
    public = load(PUBLIC)
    public_cert = artifact(PUBLIC_CERTIFICATION)
    public["current_stage"] = "architecture"
    public["generated_at_utc"] = now
    public["last_updated_utc"] = now
    public["latest_decision"] = DECISION
    public["required_manager_action"] = MANAGER_ACTION
    public["required_operator_action"] = "none"
    public["routing_authorized"] = "manager_successor_freeze_only"
    public["routing_status"] = ROUTING_STATUS
    public["stage_closing"] = False
    public["current_mode"] = "ADVANCE"
    public["ordered_supported_layer_operator_prefix"] = PREFIX
    public["supported_layer_operator_prefix"] = PREFIX
    public["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
    public["baseline_harness_certification"] = {
        "status": cert["status"],
        "decision": "PASS",
        "certified_at_utc": cert["certified_at_utc"],
        "successor_freeze_precondition_satisfied": True,
        "successor_frozen": False,
        "candidate_or_model_execution_count": 0,
        "public_evidence": public_cert,
        "required_manager_action": MANAGER_ACTION,
    }
    public["blockers"] = [{
        "id": "manager_structurally_distinct_successor_freeze_required",
        "status": "open",
        "severity": "gate",
        "owner": "manager",
        "detail": (
            "The candidate-independent synthetic harness is independently certified. The failed QECR "
            "contract remains sealed, and no structurally distinct successor is frozen."
        ),
        "evidence": [public_cert],
        "resolution": MANAGER_ACTION,
    }]

    for key in ("selected_replacement_contract", "latest_rtl_candidate"):
        update_failed_contract(public.get(key), cert, now)

    for key in ("dashboard_fields", "implementation_frontier"):
        section = public.get(key)
        if not isinstance(section, dict):
            continue
        section.update({
            "current_stage": "architecture",
            "latest_decision": DECISION,
            "required_manager_action": MANAGER_ACTION,
            "routing_status": ROUTING_STATUS,
            "stage_closing": False,
            "current_mode": "ADVANCE",
            "mode": "ADVANCE",
            "ordered_supported_layer_operator_prefix": PREFIX,
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "candidate_meets_numeric_acceptance": False,
            "baseline_harness_certification": copy.deepcopy(public["baseline_harness_certification"]),
            "next_successor_status": "not_frozen",
        })
        update_failed_contract(section.get("candidate_mechanism"), cert, now)
        update_failed_contract(section.get("latest_rtl_candidate"), cert, now)

    public["architecture_successor_dispatch"] = {
        "current_stage": "architecture",
        "sealed_contract_id": CONTRACT,
        "status": "awaiting_manager_structurally_distinct_successor_freeze",
        "successor_frozen": False,
        "implementation_authorized": False,
        "stage_closing": False,
        "required_manager_action": MANAGER_ACTION,
        "baseline_harness_certification": copy.deepcopy(public["baseline_harness_certification"]),
    }
    for key in ("architecture_certification_mission", "architecture_proposal_gate"):
        section = public.setdefault(key, {})
        section.update({
            "contract_id": CONTRACT,
            "status": "historical_architecture_sealed_terminal_no_go",
            "current_successor": False,
            "implementation_authorized": False,
            "stage_closing": False,
            "required_manager_action": MANAGER_ACTION,
        })

    stage = public.setdefault("stage", {})
    stage.update({
        "completed_prior_stages": ["definition"],
        "current_stage": "architecture",
        "current_stage_status": ROUTING_STATUS,
        "current_stage_checklist": copy.deepcopy(ARCHITECTURE_CHECKLIST),
        "checklist_scope": "next_structurally_distinct_successor_not_yet_frozen",
        "current_stage_evidence": [public_cert["path"]],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
        "stage_closing": False,
        "downstream_stages_locked_until_manager_advance": [
            "environment", "rtl", "verification", "ppa", "prototype", "benchmark", "signoff"
        ],
    })
    public["public_claims"] = [
        {"claim": "the Manager-owned current stage is architecture", "evidence": ["research/PIPELINE_STATE.json"]},
        {"claim": "the candidate-independent synthetic baseline harness is independently certified", "evidence": [public_cert["path"]]},
        {"claim": "the failed QECR contract and accidental post-terminal artifacts remain sealed", "evidence": ["evidence/cross_layer_quantization_error_carry_final_output_v1/recovery/manager_architecture_rollback_v1/SEAL.json"]},
        {"claim": "no successor is frozen and no candidate or model execution is authorized", "evidence": [public_cert["path"]]},
        {"claim": "the supported prefix, first unsupported operator, ADVANCE mode, immutable targets, and historical PPA frontier are unchanged", "evidence": ["design/CHIP_SCOPE.json", "design/TARGET.json", "design/FAST_LOOP_POLICY.json"]},
    ]
    update_all_failed_contract_copies(public, cert, now)

    records = {
        item.get("path"): item
        for item in public.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path")
    }
    for path in (
        PUBLIC_CERTIFICATION,
        PUBLIC_CERTIFICATION_SUM,
        BINDER,
        POLICY,
        SCOPE,
        ROOT / "MISSION.md",
        ROOT / "design/ARCHITECTURE.md",
    ):
        records[path.relative_to(ROOT).as_posix()] = artifact(path)
    public["artifact_hashes"] = [records[key] for key in sorted(records)]
    public.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    public["integrity"]["canonical_sha256"] = None
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)


def update_checkpoint(cert: dict[str, Any]) -> None:
    CHECKPOINT.write_text(
        f"""# Goal

Preserve the terminal `{CONTRACT}` protocol/provenance NO_GO, bind the
independently certified candidate-independent baseline harness, and wait for the
Manager to freeze a structurally distinct successor before new architecture or execution work.

# Current state

`research/PIPELINE_STATE.json` records Manager-owned `current_stage=architecture`.
The failed QECR contract and accidental post-terminal artifacts remain sealed by
`{cert['manager_recovery_seal']['path']}`.

The candidate-independent synthetic/dry-fixture harness is independently certified
`PASS`. All seven required properties and twelve independent acceptance checks pass;
the protected {cert['protected_artifact_count']}-artifact manifest is unchanged, and
candidate/model execution count is zero. The privacy-filtered projection is
`{PUBLIC_CERTIFICATION.relative_to(ROOT).as_posix()}`.

# Preserved contracts

- The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
  `layer_0.rope_q`; mode remains `ADVANCE`.
- The 2.0 mm2 non-SRAM cap, 100 MHz floor, 128-bit memory boundary, and historical
  62,199-cell / 0.6108746272-mm2 / +0.1502-ns frontier are unchanged.
- Do not rerun or repair the failed QECR baseline/candidate contract. Do not execute
  candidate/model, RTL, verification, PPA, prototype, benchmark, signoff, tapeout,
  or silicon work for a successor that has not been frozen.
- Planner and Engineer must not edit `research/PIPELINE_STATE.json`.

# Remaining authority gate

The harness prerequisite is satisfied, but no successor is frozen. Only the Manager
may now freeze a structurally distinct successor or record an architecture no-go.
Until that happens, the four architecture checklist items remain open for the next
exact successor and downstream stages remain locked.
""",
        encoding="utf-8",
    )


def update_live_view(cert: dict[str, Any], now: str) -> None:
    dump(LIVE_VIEW, {
        "version": 4,
        "title": "ACE-2 architecture successor gate",
        "generated_at_utc": now,
        "current_stage": "architecture",
        "sealed_contract_id": CONTRACT,
        "successor_frozen": False,
        "implementation_authorized": False,
        "reason": "synthetic baseline harness independently certified; Manager successor freeze pending",
        "paths": [
            "research/PIPELINE_STATE.json",
            "CHECKPOINT.md",
            "design/FAST_LOOP_POLICY.json",
            "design/CHIP_SCOPE.json",
            "research/PUBLIC_STATUS.json",
            PUBLIC_CERTIFICATION.relative_to(ROOT).as_posix(),
        ],
        "certification_sha256": cert["independent_l2_review"]["sha256"],
    })


def check_bound(cert: dict[str, Any]) -> None:
    verify_companion(PUBLIC_CERTIFICATION, PUBLIC_CERTIFICATION_SUM)
    policy = load(POLICY)
    scope = load(SCOPE)
    public = load(PUBLIC)
    require(policy.get("baseline_harness_hardening_gate", {}).get("status") == "independently_certified_pass", "policy certification is stale")
    require(policy.get("manager_recommendation") == MANAGER_ACTION, "policy Manager action is stale")
    require(policy.get("next_successor", {}).get("status") == "not_frozen", "policy incorrectly freezes successor")
    require(scope.get("stage", {}).get("current_stage_checklist") == ARCHITECTURE_CHECKLIST, "scope architecture checklist differs")
    require(scope.get("routing_status") == ROUTING_STATUS, "scope routing is stale")
    latest_decision = public.get("latest_decision")
    if latest_decision != DECISION:
        later_review = public.get("architecture_successor_review", {})
        require(
            later_review.get("architecture_accepted") is True
            and later_review.get("implementation_authorized") is False
            and later_review.get("successor_frozen") is False,
            "public decision supersedes certification without an accepted locked architecture review",
        )
    require(public.get("routing_status") == ROUTING_STATUS, "public routing is stale")
    require(public.get("stage", {}).get("current_stage_checklist") == ARCHITECTURE_CHECKLIST, "public architecture checklist differs")
    require(public.get("baseline_harness_certification", {}).get("decision") == "PASS", "public certification is stale")
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public), "public canonical hash differs")
    records = {
        item.get("path"): item
        for item in public.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path")
    }
    for path in (
        PUBLIC_CERTIFICATION,
        PUBLIC_CERTIFICATION_SUM,
        BINDER,
        POLICY,
        SCOPE,
        ROOT / "MISSION.md",
        ROOT / "design/ARCHITECTURE.md",
    ):
        relative = path.relative_to(ROOT).as_posix()
        require(records.get(relative) == artifact(path), f"public artifact binding differs: {relative}")
    require("Only the Manager" in CHECKPOINT.read_text(encoding="utf-8"), "checkpoint authority boundary missing")
    live = load(LIVE_VIEW)
    require(live.get("successor_frozen") is False, "live view incorrectly freezes successor")
    require(load(PIPELINE).get("current_stage") == "architecture", "pipeline stage changed")
    require(cert["candidate_or_model_execution_count"] == 0, "certification execution boundary differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report, review, _ = validate_evidence()
    cert = certification(report, review)
    if args.check:
        check_bound(cert)
        print(
            "ACE2_SYNTHETIC_BASELINE_HARNESS_CERTIFICATION_CHECK_PASS "
            f"review_sha256={cert['independent_l2_review']['sha256']} "
            f"protected_artifacts={cert['protected_artifact_count']} successor_frozen=false"
        )
        return 0

    write_public_certification(public_projection(cert))
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    update_policy(cert)
    update_scope(cert, now)
    update_checkpoint(cert)
    update_live_view(cert, now)
    update_public(cert, now)
    check_bound(cert)
    print(
        "ACE2_SYNTHETIC_BASELINE_HARNESS_CERTIFICATION_BIND_PASS "
        f"review_sha256={cert['independent_l2_review']['sha256']} "
        f"protected_artifacts={cert['protected_artifact_count']} successor_frozen=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
