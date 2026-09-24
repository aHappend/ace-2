#!/usr/bin/env python3
"""Run and bind one independent RTL-checklist review for the QECR successor."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import argus_skill
from argus_skill.adapters.agent_cli_backend import AgentCliBackend
from argus_skill.core.knobs import (
    resolve_role_backend,
    resolve_role_model,
    resolve_role_reasoning_effort,
    resolve_runner_bin_setting,
)
from argus_skill.core.paths import global_root
from argus_skill.reviewer import Reviewer, ReviewerConfig


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "cross_layer_quantization_error_carry_final_output_v1"
RTL = ROOT / "rtl/ace2_cross_layer_error_carry_core.sv"
PRECHECK = ROOT / f"evidence/{CONTRACT}/latest/PRECHECK.json"
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
OUT = ROOT / f"evidence/review/rtl_checklist_{CONTRACT}/decision.json"
REVIEW_INPUTS = OUT.parent / "inputs"
MISSION_ID = "rtl-checklist-cross-layer-error-carry-final-output-v1"
FINAL_RTL_STATUS = "independent_rtl_checklist_accepted_stage_closing_false"
FINAL_MANAGER_ACTION = "hold_rtl_downstream_stages_remain_locked"
CHECKLIST = {
    "rtl.contract-traceability": True,
    "rtl.hardware-discipline": True,
    "rtl.ip-provenance": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path | str) -> dict[str, Any]:
    resolved = path if isinstance(path, Path) else ROOT / path
    value = json.loads(resolved.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {resolved}")
    return value


def dump(path: Path | str, value: dict[str, Any]) -> None:
    resolved = path if isinstance(path, Path) else ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path | str) -> dict[str, Any]:
    resolved = path if isinstance(path, Path) else ROOT / path
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def require_artifact(record: dict[str, Any]) -> None:
    require(isinstance(record, dict), "artifact record is not an object")
    path = ROOT / record["path"]
    require(path.is_file(), f"artifact is missing: {record['path']}")
    require(path.stat().st_size == record["bytes"], f"artifact size changed: {record['path']}")
    require(sha256_file(path) == record["sha256"], f"artifact hash changed: {record['path']}")


def require_engineer_ledger_proof(provenance: dict[str, Any]) -> None:
    proof = provenance.get("execution_ledger", {})
    require(isinstance(proof, dict), "Engineer execution-ledger proof is missing")
    thread_id = provenance.get("thread_id", "")
    session = provenance.get("session_id", "")
    call_id = proof.get("call_id", "")
    require(proof.get("thread_id") == thread_id, "Engineer ledger proof thread differs")
    require(str(proof.get("run_label", "")).startswith("engineer"), "producer ledger role is not Engineer")
    require(proof.get("status") == "completed", "Engineer ledger execution is not complete")
    ledger_name = os.environ.get("ARGUS_SKILL_AGENT_IO_LOG", "")
    ledger = Path(ledger_name) if ledger_name else Path()
    require(ledger.is_file(), "execution ledger is unavailable")
    start: dict[str, Any] | None = None
    completion: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("call_id") != call_id:
            continue
        if event.get("type") == "agent.io.start":
            start = event
        elif event.get("type") == "agent.io.complete" and event.get("thread_id") == thread_id:
            completion = event
        elif event.get("type") == "usage.recorded" and event.get("thread_id") == thread_id:
            usage = event
    require(start is not None and str(start.get("run_label", "")).startswith("engineer"), "Engineer ledger start is missing")
    require(Path(start.get("working_dir", "")).resolve() == ROOT, "Engineer ledger working directory differs")
    require(
        completion is not None and completion.get("exit_code") == 0 and completion.get("turn_completed") is True,
        "Engineer ledger completion is missing or unsuccessful",
    )
    if usage is not None:
        require(usage.get("project_id") == session and usage.get("status") == "completed", "Engineer usage ledger differs")


def require_producer_provenance(provenance: dict[str, Any]) -> None:
    role = provenance.get("role")
    if role == "engineer":
        require_engineer_ledger_proof(provenance)
        return
    require(role == "planner_direct_executor", "preflight producer role is unsupported")
    require(
        provenance.get("execution_contract")
        == "operator_planner_direct_execution_contract",
        "Planner direct-execution contract is missing",
    )
    require(
        provenance.get("status")
        == "fresh_direct_implementation_from_frozen_contract",
        "Planner direct-execution provenance is incomplete",
    )
    require(bool(provenance.get("implementation_source_aggregate_sha256")),
            "Planner direct-execution source aggregate is missing")


def reviewer_accepts(reason: str, item: str) -> bool:
    normalized = reason.lower().replace("`", "")
    return any(
        token in normalized
        for token in (f"{item}: supported", f"{item}: true", f"{item}=true")
    )


def reviewer_preserves_non_closing(reason: str) -> bool:
    normalized = reason.lower().replace("`", "").replace(" ", "")
    return "stage_closing=false" in normalized


def prior_planner_draft_decision(provenance: dict[str, Any]) -> dict[str, Any] | None:
    archive_record = provenance.get("planner_draft_archive")
    if not isinstance(archive_record, dict):
        return None
    require_artifact(archive_record)
    archive = load(ROOT / archive_record["path"])
    for record in archive.get("artifacts", []):
        if record.get("original_path") == OUT.relative_to(ROOT).as_posix():
            return load(ROOT / record["archived_path"])
    return None


def session_id() -> str:
    configured = os.environ.get("ARGUS_SKILL_SESSION_ID")
    if configured:
        return configured
    session_file = ROOT / ".ace2-session.json"
    if session_file.is_file():
        value = load(session_file).get("sid")
        if isinstance(value, str) and value:
            return value
    return ROOT.name


def preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    if OUT.exists():
        prior = load(OUT)
        require(prior.get("reviewer_status") != "done", "accepted review already exists")
        digest = sha256_file(OUT)
        archive = OUT.parent / "archive" / f"decision.{digest}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            require(archive.read_bytes() == OUT.read_bytes(), "review archive collision")
        else:
            shutil.copy2(OUT, archive)
        OUT.unlink()
    packet = load(PRECHECK)
    manifest = load(MANIFEST)
    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(packet.get("contract_id") == CONTRACT, "preflight contract mismatch")
    require(packet.get("status") == "pass_ready_for_independent_rtl_review", "preflight is not ready")
    require(packet.get("stage_closing") is False, "preflight unexpectedly closes RTL")
    require(packet.get("stage_checklist") == CHECKLIST, "preflight checklist differs")
    implementation_provenance = packet.get("implementation_provenance", {})
    require_producer_provenance(implementation_provenance)
    if implementation_provenance.get("role") == "engineer":
        require(prior_planner_draft_decision(implementation_provenance) is not None,
                "Planner-draft review provenance is missing")
    require(not any(packet.get("prohibited_runs", {}).values()), "a prohibited run is recorded")
    require(canonical_sha256(packet) == packet.get("integrity", {}).get("canonical_sha256"), "preflight canonical hash mismatch")
    for relative, expected in packet.get("source_hashes", {}).items():
        require(sha256_file(ROOT / relative) == expected, f"bound source changed: {relative}")
    require(manifest.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"), "manifest candidate hash mismatch")
    require(manifest.get("traceability", {}).get("stage_checklist") == CHECKLIST, "manifest checklist differs")
    manifest_digest = sha256_file(MANIFEST)
    manifest_snapshot = REVIEW_INPUTS / f"RTL_MANIFEST.pending.{manifest_digest}.json"
    manifest_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if manifest_snapshot.exists():
        require(manifest_snapshot.read_bytes() == MANIFEST.read_bytes(), "review-input manifest collision")
    else:
        shutil.copy2(MANIFEST, manifest_snapshot)
    verified = {
        "precheck": artifact(PRECHECK),
        "manifest": artifact(manifest_snapshot),
        "pipeline_state": artifact(PIPELINE),
        "rtl_source": artifact(RTL),
        "candidate_id": packet["candidate_id"],
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "stage_checklist": CHECKLIST,
        "logs": packet["logs"],
        "interfaces": packet["interfaces"],
        "modules": packet["modules"],
        "workspace_proof": packet["workspace_proof"],
        "prohibited_runs": packet["prohibited_runs"],
        "implementation_provenance": implementation_provenance,
    }
    return packet, verified


def check_existing() -> None:
    decision = load(OUT)
    packet = load(PRECHECK)
    require(decision.get("reviewer_status") == "done", "independent RTL review is not done")
    require(decision.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"), "review hash mismatch")
    require(decision.get("checklist") == CHECKLIST, "review checklist differs")
    rulings = decision.get("checklist_rulings", {})
    require(set(rulings) == set(CHECKLIST), "review does not rule separately on all RTL checklist items")
    require(all(rulings[item].get("accepted") is True for item in CHECKLIST), "an RTL checklist ruling is not accepted")
    provenance = decision.get("producer_provenance", {})
    require_producer_provenance(provenance)
    if provenance.get("role") == "engineer":
        require(prior_planner_draft_decision(provenance) is not None,
                "review does not bind Planner-draft provenance")
    require(decision.get("candidate_capability_accepted") is False, "review improperly accepts candidate capability")
    require(decision.get("stage_closing") is False, "standalone RTL review changed immutable stage_closing=false")
    require(reviewer_preserves_non_closing(decision.get("reason", "")), "review reason does not preserve stage_closing=false")
    require(decision.get("required_remediation") == "", "accepted RTL review retains remediation")
    require(decision.get("live_rtl_source_sha256") == sha256_file(RTL), "review RTL source hash is stale")
    verified = decision.get("verified_evidence", {})
    for key in ("precheck", "manifest", "pipeline_state", "rtl_source"):
        require_artifact(verified.get(key, {}))
    for collection in (verified.get("logs", {}), verified.get("interfaces", {})):
        for record in collection.values():
            require_artifact(record)
    prior = prior_planner_draft_decision(provenance)
    require(bool(decision.get("reviewer_thread_id")), "reviewer execution identity is missing")
    if prior is not None:
        require(
            decision.get("reviewer_thread_id") != prior.get("reviewer_thread_id"),
            "reviewer execution is not fresh relative to the Planner-draft verdict",
        )
    manifest = load(MANIFEST)
    require(manifest.get("candidate_rtl_hash") == decision.get("candidate_rtl_hash"), "live manifest candidate changed")
    require(
        manifest.get("candidate_review_binding", {}).get("evidence") == artifact(OUT),
        "live manifest does not bind the current Reviewer decision",
    )
    require(manifest.get("stage_closing") is False, "manifest changes immutable stage_closing=false")
    require(
        manifest.get("candidate_review_binding", {}).get("stage_closing") is False,
        "manifest review binding changes immutable stage_closing=false",
    )
    policy = load("design/FAST_LOOP_POLICY.json")
    for key in ("active_architecture_contract", "selected_replacement_contract"):
        require(policy.get(key, {}).get("stage_closing") is False, f"policy {key} changes immutable stage_closing=false")
    target = load("design/TARGET.json")
    require(
        target.get("current_architecture_contract", {}).get("stage_closing") is False,
        "target changes immutable stage_closing=false",
    )
    trace = (ROOT / "design/RTL_TRACEABILITY.md").read_text(encoding="utf-8")
    require("`stage_closing=false`" in trace, "traceability does not preserve stage_closing=false")
    public = load("research/PUBLIC_STATUS.json")
    require(public.get("stage_closing") is False, "public status changes immutable stage_closing=false")
    require(public.get("stage", {}).get("stage_closing") is False, "public stage changes immutable stage_closing=false")
    require(
        public.get("selected_replacement_contract", {}).get("stage_closing") is False,
        "public selected contract changes immutable stage_closing=false",
    )
    require(
        public.get("architecture_successor_dispatch", {}).get("stage_closing") is False,
        "public successor dispatch changes immutable stage_closing=false",
    )
    require(manifest.get("candidate_status") == FINAL_RTL_STATUS, "manifest candidate status is not final")
    require(manifest.get("implementation_completed") is True, "manifest does not record completed RTL implementation")
    require(
        manifest.get("implementation_authority_consumed") is True,
        "manifest does not record consumed RTL implementation authority",
    )
    require(
        manifest.get("interfaces_contract_status") == "qecr_standalone_interfaces_independently_reviewed",
        "manifest interface status is still review-pending",
    )
    manifest_contract = manifest.get("proposed_replacement_contract", {})
    require(manifest_contract.get("status") == FINAL_RTL_STATUS, "manifest replacement contract is not final")
    require(
        manifest_contract.get("required_manager_action") == FINAL_MANAGER_ACTION,
        "manifest replacement contract retains stale routing",
    )
    require(manifest_contract.get("rtl_review") == artifact(OUT), "manifest replacement contract lacks review binding")

    scope = load("design/CHIP_SCOPE.json")
    scope_authority = scope.get("authority_override", {})
    scope_approval = scope_authority.get("operator_implementation_approval", {})
    require(scope_authority.get("status") == FINAL_RTL_STATUS, "CHIP_SCOPE authority remains pre-RTL")
    require(scope_authority.get("implementation_completed") is True, "CHIP_SCOPE omits completed implementation")
    require(
        scope_authority.get("implementation_authority_consumed") is True,
        "CHIP_SCOPE omits consumed implementation authority",
    )
    require(scope_approval.get("status") == FINAL_RTL_STATUS, "CHIP_SCOPE approval remains pending")
    require(scope_approval.get("rtl_review") == artifact(OUT), "CHIP_SCOPE approval lacks review binding")
    active_successor = scope.get("operator_owned_execution_policy", {}).get("active_successor_contract", {})
    require(active_successor.get("status") == FINAL_RTL_STATUS, "CHIP_SCOPE execution policy remains pending")
    require(
        active_successor.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"),
        "CHIP_SCOPE execution policy candidate hash is stale",
    )
    require(scope.get("stage", {}).get("current_stage") == "rtl", "CHIP_SCOPE stage mirror is stale")
    require(scope.get("stage", {}).get("current_stage_checklist") == CHECKLIST, "CHIP_SCOPE checklist differs")
    require(
        scope.get("implementation_frontier", {}).get("candidate_rtl_hash_scope")
        == packet.get("candidate_rtl_hash_scope"),
        "CHIP_SCOPE candidate hash scope is stale",
    )

    for key in ("architecture_certification_mission", "architecture_proposal_gate"):
        section = public.get(key, {})
        require(section.get("implementation_completed") is True, f"public {key} omits completed implementation")
        require(
            section.get("implementation_authority_consumed") is True,
            f"public {key} omits consumed implementation authority",
        )
        require(section.get("status") == FINAL_RTL_STATUS, f"public {key} remains architecture-pending")
    require(public.get("stage", {}).get("current_stage") == "rtl", "public stage mirror is stale")
    require(public.get("stage", {}).get("current_stage_checklist") == CHECKLIST, "public checklist differs")
    require(
        public.get("selected_replacement_contract", {}).get("status") == FINAL_RTL_STATUS,
        "public selected contract is not final",
    )
    require(
        public.get("architecture_successor_dispatch", {}).get("status") == FINAL_RTL_STATUS,
        "public successor dispatch is not final",
    )
    require(public.get("required_manager_action") == FINAL_MANAGER_ACTION, "public top-level routing is stale")
    public_latest = public.get("latest_rtl_candidate", {})
    require(public_latest.get("status") == FINAL_RTL_STATUS, "public latest RTL candidate is not final")
    require(
        public_latest.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"),
        "public latest RTL candidate hash is stale",
    )
    require(public_latest.get("rtl_review") == artifact(OUT), "public latest RTL candidate lacks review binding")
    for key in ("dashboard_fields", "implementation_frontier"):
        section = public.get(key, {})
        nested_latest = section.get("latest_rtl_candidate", {})
        require(nested_latest.get("status") == FINAL_RTL_STATUS, f"public {key} latest RTL candidate is stale")
        require(
            nested_latest.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"),
            f"public {key} latest RTL candidate hash is stale",
        )
        environment = section.get("latest_environment_stage", {})
        require(environment.get("status") == "manager_advanced_to_rtl", f"public {key} environment route is stale")
        operator_policy = section.get("operator_policy")
        if isinstance(operator_policy, dict):
            active_contract = operator_policy.get("active_architecture_contract", {})
            require(active_contract.get("status") == FINAL_RTL_STATUS, f"public {key} active contract is stale")
            require(
                active_contract.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"),
                f"public {key} active contract hash is stale",
            )

    archive_dir = OUT.parent / "archive"
    for archived in archive_dir.glob("superseded_planner_draft.*.json") if archive_dir.is_dir() else []:
        archived_decision = load(archived)
        archived_provenance = archived_decision.get("producer_provenance", {})
        require(
            not isinstance(archived_provenance, dict) or archived_provenance.get("role") != "engineer",
            f"Engineer/Reviewer decision is mislabeled as Planner draft: {archived.relative_to(ROOT)}",
        )
    print(f"ACE2_QECR_RTL_REVIEW_CHECK_PASS candidate={decision['candidate_id']}")


def bind_done(payload: dict[str, Any]) -> None:
    packet = load(PRECHECK)
    decision_record = artifact(OUT)
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    require(payload.get("stage_closing") is False, "review payload changes immutable stage_closing=false")
    stage_closing = False
    closing_word = "false"
    review_status = FINAL_RTL_STATUS
    manager_action = FINAL_MANAGER_ACTION
    manifest = load(MANIFEST)
    manifest["stage_closing"] = False
    manifest["candidate_status"] = review_status
    manifest["implementation_authorized"] = False
    manifest["implementation_authority_consumed"] = True
    manifest["implementation_completed"] = True
    manifest["interfaces_contract_status"] = "qecr_standalone_interfaces_independently_reviewed"
    manifest["candidate_implementation_provenance"] = copy.deepcopy(payload["producer_provenance"])
    manifest["candidate_review_binding"] = {
        "candidate_capability_accepted": False,
        "decision": "accepted_rtl_checklist_only",
        "reviewer_status": "done",
        "evidence": decision_record,
        "scope": "standalone_candidate_rtl_checklist_only",
        "stage_closing": stage_closing,
        "quality_discriminator_complete": False,
        "shell_admitted": False,
        "ppa_run": False,
        "producer_provenance": copy.deepcopy(payload["producer_provenance"]),
        "checklist_rulings": copy.deepcopy(payload["checklist_rulings"]),
    }
    manifest["independent_reviewer_acceptance"] = {
        "architecture_review": "accepted",
        "environment_review": "accepted_manager_advanced_to_rtl",
        "rtl_review": "accepted_exact_candidate_hash",
        "status": "rtl_checklist_accepted_stage_closing_false",
    }
    manifest["traceability"]["architecture_contract_gap"] = {
        "status": "closed_for_exact_standalone_rtl",
        "resolution_owner": "independent_reviewer_done",
    }
    manifest_contract = manifest.setdefault("proposed_replacement_contract", {})
    manifest_contract.update({
        "candidate_id": packet["candidate_id"],
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "operator_approval_consumed": True,
        "required_manager_action": manager_action,
        "rtl_review": decision_record,
        "rtl_started": True,
        "stage_closing": False,
        "status": review_status,
        "updated_at_utc": now,
    })
    dump(MANIFEST, manifest)

    policy = load("design/FAST_LOOP_POLICY.json")
    for key in ("active_architecture_contract", "selected_replacement_contract"):
        policy[key]["status"] = review_status
        policy[key]["stage_closing"] = stage_closing
        policy[key]["required_manager_action"] = manager_action
        policy[key]["rtl_review"] = decision_record
        policy[key]["implementation_authorized"] = False
        policy[key]["implementation_authority_consumed"] = True
        policy[key]["implementation_completed"] = True
        policy[key]["updated_at_utc"] = now
    policy["manager_recommendation"] = manager_action
    dump("design/FAST_LOOP_POLICY.json", policy)

    target = load("design/TARGET.json")
    target["generated_at_utc"] = now
    target["current_architecture_contract"] = copy.deepcopy(policy["selected_replacement_contract"])
    target["fast_loop_contract"]["rtl_status"] = "independent_checklist_accepted_stage_closing_false"
    dump("design/TARGET.json", target)

    scope = load("design/CHIP_SCOPE.json")
    authority = scope.setdefault("authority_override", {})
    authority.update({
        "active_replacement_contract": CONTRACT,
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "operator_approval_consumed": True,
        "required_next_action": manager_action,
        "stage_closing": False,
        "status": review_status,
    })
    approval = authority.setdefault("operator_implementation_approval", {})
    frozen_at = approval.get("frozen_at_utc")
    approval.update(copy.deepcopy(policy["selected_replacement_contract"]))
    if frozen_at:
        approval["frozen_at_utc"] = frozen_at
    execution_policy = scope.setdefault("operator_owned_execution_policy", {})
    active_successor = execution_policy.setdefault("active_successor_contract", {})
    successor_frozen_at = active_successor.get("frozen_at_utc")
    active_successor.update(copy.deepcopy(policy["selected_replacement_contract"]))
    if successor_frozen_at:
        active_successor["frozen_at_utc"] = successor_frozen_at
    frontier = scope.setdefault("implementation_frontier", {})
    frontier.update({
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "candidate_rtl_hash_scope": packet["candidate_rtl_hash_scope"],
        "current_mode": "ADVANCE",
        "first_unsupported_layer_operator": "layer_0.rope_q",
        "latest_decision": "cross_layer_error_carry_rtl_checklist_independently_accepted",
    })
    scope["stage"] = {
        "current_stage": "rtl",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_evidence": [
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
            PRECHECK.relative_to(ROOT).as_posix(),
            decision_record["path"],
        ],
        "current_stage_status": review_status,
        "downstream_stages_locked_until_manager_advance": [
            "verification", "ppa", "prototype", "benchmark", "signoff",
        ],
        "planner_may_advance_stage": False,
        "stage_closing": False,
        "stage_transition_owner": "Manager",
    }
    scope["last_updated_utc"] = now
    dump("design/CHIP_SCOPE.json", scope)

    trace_path = ROOT / "design/RTL_TRACEABILITY.md"
    trace = trace_path.read_text(encoding="utf-8")
    trace = trace.replace(
        "- Fresh independent RTL checklist review is pending. The shell is not modified\n"
        "  or admitted, the accepted prefix remains through `layer_0.v_proj`, and first\n"
        "  unsupported remains `layer_0.rope_q`.\n",
        "- Fresh independent review accepted contract traceability, hardware discipline,\n"
        "  and IP provenance for the exact candidate aggregate. The verdict is\n"
        f"  `stage_closing={closing_word}`; the shell is not modified or admitted, the accepted\n"
        "  prefix remains through `layer_0.v_proj`, and first unsupported remains\n"
        "  `layer_0.rope_q`.\n",
    )
    trace_path.write_text(trace, encoding="utf-8")

    status = load("research/PUBLIC_STATUS.json")
    for key in ("architecture_certification_mission", "architecture_proposal_gate"):
        section = status.setdefault(key, {})
        section.update({
            "contract_id": CONTRACT,
            "implementation_authorized": False,
            "implementation_authority_consumed": True,
            "implementation_completed": True,
            "required_manager_action": manager_action,
            "stage_closing": False,
            "status": review_status,
        })
    if isinstance(status.get("latest_environment_stage"), dict):
        status["latest_environment_stage"].update({
            "implementation_authorized": False,
            "implementation_authority_consumed": True,
            "status": "manager_advanced_to_rtl",
        })
    latest_rtl_candidate = {
        "candidate_capability_accepted": False,
        "candidate_id": packet["candidate_id"],
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "candidate_rtl_hash_scope": packet["candidate_rtl_hash_scope"],
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "required_manager_action": manager_action,
        "rtl_review": decision_record,
        "shell_admitted": False,
        "stage_closing": False,
        "status": review_status,
    }
    status["latest_rtl_candidate"] = copy.deepcopy(latest_rtl_candidate)
    status["required_manager_action"] = manager_action
    for key in ("dashboard_fields", "implementation_frontier"):
        section = status.setdefault(key, {})
        section["latest_decision"] = "cross_layer_error_carry_rtl_checklist_independently_accepted"
        section["routing_status"] = "rtl_checklist_complete_stage_remains_rtl_downstream_locked"
        section["required_manager_action"] = manager_action
        section["rtl_review"] = decision_record
        section["implementation_authorized"] = False
        section["implementation_authority_consumed"] = True
        section["implementation_completed"] = True
        section["latest_rtl_candidate"] = copy.deepcopy(latest_rtl_candidate)
        if isinstance(section.get("latest_environment_stage"), dict):
            section["latest_environment_stage"].update({
                "implementation_authorized": False,
                "implementation_authority_consumed": True,
                "status": "manager_advanced_to_rtl",
            })
        operator_policy = section.get("operator_policy")
        if isinstance(operator_policy, dict):
            active_contract = operator_policy.setdefault("active_architecture_contract", {})
            active_contract.update(copy.deepcopy(policy["selected_replacement_contract"]))
        if isinstance(section.get("candidate_mechanism"), dict):
            section["candidate_mechanism"]["status"] = review_status
            section["candidate_mechanism"]["implementation_authorized"] = False
            section["candidate_mechanism"]["implementation_authority_consumed"] = True
            section["candidate_mechanism"]["implementation_completed"] = True
    status["latest_decision"] = "cross_layer_error_carry_rtl_checklist_independently_accepted"
    status["routing_authorized"] = "none_standalone_rtl_review_non_stage_closing"
    status["routing_status"] = "rtl_checklist_complete_stage_remains_rtl_downstream_locked"
    status["stage_closing"] = False
    status["selected_replacement_contract"] = copy.deepcopy(policy["selected_replacement_contract"])
    status["architecture_successor_dispatch"] = {
        "contract_id": CONTRACT,
        "current_stage": "rtl",
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "operator_approval_consumed": True,
        "required_manager_action": manager_action,
        "result_binding": decision_record,
        "stage_closing": stage_closing,
        "status": review_status,
    }
    status["blockers"] = [{
        "id": "downstream_locked_after_non_stage_closing_rtl_review",
        "owner": "manager",
        "severity": "gate",
        "status": "open",
        "detail": f"RTL checklist is independently accepted with stage_closing={closing_word}; only Manager may transition, and downstream work remains forbidden while current_stage is rtl.",
    }]
    status["stage"]["current_stage_status"] = review_status
    status["stage"]["current_stage"] = "rtl"
    status["stage"]["current_stage_checklist"] = copy.deepcopy(CHECKLIST)
    status["stage"]["stage_closing"] = stage_closing
    if decision_record["path"] not in status["stage"]["current_stage_evidence"]:
        status["stage"]["current_stage_evidence"].append(decision_record["path"])
    status["generated_at_utc"] = now
    status["last_updated_utc"] = now
    status["integrity"]["canonical_sha256"] = None
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump("research/PUBLIC_STATUS.json", status)

    checkpoint = f"""# Goal

Implement and independently review the bounded RTL successor
`{CONTRACT}` without entering downstream stages.

# Current state

The Manager-owned stage remains `rtl`. Independent review accepted all three
RTL checklist items for candidate `{packet['candidate_id']}` at exact aggregate
hash `{packet['candidate_rtl_hash']}` with `stage_closing={closing_word}`.

# Accepted RTL evidence

- Contract traceability, hardware discipline, and first-party/generated-source
  provenance are independently accepted for the three standalone modules.
- Exact reference/vector/metadata checks, warning-free lint, elaboration,
  integrated simulation, and bounded formal invariants pass.
- The shell is not modified or admitted. The accepted prefix remains through
  `layer_0.v_proj`; first unsupported remains `layer_0.rope_q`; mode is `ADVANCE`.

# Locked work

Quality/full verification, shell regression, candidate PPA/power, physical
design, prototype, benchmark, and signoff remain unrun and forbidden by the
current-stage gate. The 2.0 mm2 non-SRAM cap and 100 MHz floor remain unmet for
this successor because no candidate PPA is legal at `current_stage=rtl`.

# Routing

No Planner or Engineer stage transition is authorized. Manager owns any later
transition; `research/PIPELINE_STATE.json` was not edited.
"""
    CHECKPOINT.write_text(checkpoint, encoding="utf-8")


def main() -> int:
    if "--check" in sys.argv[1:]:
        check_existing()
        return 0
    if "--bind-existing" in sys.argv[1:]:
        check_existing()
        bind_done(load(OUT))
        print("ACE2_QECR_RTL_REVIEW_REBIND_PASS stage_closing=false")
        return 0
    packet, verified = preflight()
    backend_name = resolve_role_backend("reviewer")
    runner = AgentCliBackend(backend=backend_name, runner_bin=resolve_runner_bin_setting("reviewer") or None)
    sid = session_id()
    runner.set_usage_context(
        project_root=global_root() / "projects" / sid,
        global_root=global_root(),
        mission_id=MISSION_ID,
    )
    package_root = Path(argus_skill.__file__).resolve().parent
    reviewer_skill = (package_root / "verticals/chip_design/skills/reviewer/chip-design-signoff-review.md").read_text(encoding="utf-8")
    objective = (
        "Independently inspect the hash-bound standalone RTL preflight for "
        f"{CONTRACT}. Decide only whether the three RTL checklist items are supported "
        "for the exact candidate hash. Do not accept model capability, shell admission, "
        "quality, PPA, prototype, benchmark, signoff, tapeout, or silicon claims. "
        "This is a standalone evidence review and the immutable operator contract requires "
        "stage_closing=false regardless of whether all three checklist items are supported. "
        "The current operator also explicitly requires Planner direct project execution and "
        "forbids delegating implementation to another role. Therefore a hash-bound "
        "planner_direct_executor provenance record is an authorized producer identity; it "
        "must not be rejected merely because it is not an Engineer ledger record. "
        "The verdict reason must state a separate ruling for rtl.contract-traceability, "
        "rtl.hardware-discipline, and rtl.ip-provenance using those exact identifiers."
    )
    instruction = (
        "Read design/ARCHITECTURE.md, design/NUMERICAL_REPLACEMENT_PROPOSAL.md, "
        "design/RTL_MANIFEST.json, design/RTL_TRACEABILITY.md, "
        "rtl/ace2_cross_layer_error_carry_core.sv, and the PRECHECK directly. "
        "Use only short deterministic hash/content checks and recorded logs/XML; do not rerun "
        "generation, simulation, quality, synthesis, STA, or PPA. Return done only if exact "
        "traceability, hardware discipline, and provenance are supported. Distinguish the "
        "operator-classified unaccepted Planner draft archive from the fresh direct-execution "
        "source and require a new Reviewer execution identity. Judge rtl.ip-provenance by the "
        "checklist contract: pinned first-party/generated source revisions, license status, "
        "wrappers/configuration, and deterministic regeneration commands. Do not add an "
        "Engineer-only producer requirement that the operator contract supersedes. State each "
        "ruling exactly as "
        "'<identifier>: supported' when supported. Keep candidate_capability_accepted=false. "
        "If all three rulings are supported for the final hashes with no remediation, return "
        "done with empty remediation and explicitly state stage_closing=false; otherwise return "
        "continue with concrete remediation and stage_closing=false. This review never closes "
        "or advances the Manager-owned stage."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=sid,
        main_summary="Operator-authorized Planner direct execution binds fresh source hashes, the archived unaccepted Planner draft, warning-free lint, elaboration, integrated simulation, bounded formal invariants, and no prohibited downstream run.",
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="standalone_rtl_checklist_only_stage_closing_false",
        checkpoint_path=str(CHECKPOINT),
        escalate_hint="If done, stage_closing remains false, current_stage remains rtl, and all downstream work stays locked.",
        preselected_skill_block=reviewer_skill,
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort("ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"),
            skip_git_repo_check=True,
            full_auto=True,
            dangerous_yolo=False,
            sandbox_mode="workspace-write",
            isolate_workdir=False,
            working_dir=str(ROOT),
        ),
    )
    raw = asdict(review)
    status = raw.get("status", "continue")
    reason = raw.get("reason", "")
    remediation = raw.get("required_remediation", "")
    require(isinstance(remediation, str), "Reviewer remediation is not text")
    prior_decision = prior_planner_draft_decision(packet["implementation_provenance"])
    if status == "done":
        require(all(reviewer_accepts(reason, item) for item in CHECKLIST), "Reviewer reason lacks supported checklist rulings")
        require(reviewer_preserves_non_closing(reason), "Reviewer reason does not preserve stage_closing=false")
        require(remediation == "", "Reviewer returned done with non-empty remediation")
        require(bool(raw.get("thread_id")), "Reviewer execution identity is missing")
        if prior_decision is not None:
            require(
                raw.get("thread_id") != prior_decision.get("reviewer_thread_id"),
                "Reviewer execution is not fresh relative to the Planner-draft verdict",
            )
            prior_fingerprint = prior_decision.get("raw_decision", {}).get("static_fingerprint")
            require(
                not prior_fingerprint or raw.get("static_fingerprint") != prior_fingerprint,
                "Reviewer static fingerprint is not fresh relative to the Planner-draft verdict",
            )
    checklist_rulings = {
        item: {
            "accepted": status == "done" and reviewer_accepts(reason, item),
            "reviewer_reason_uses_exact_identifier": item in reason,
        }
        for item in CHECKLIST
    }
    stage_closing = False
    payload = {
        "schema_version": 1,
        "review_bound_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "reviewer_status": status,
        "contract_id": CONTRACT,
        "candidate_id": packet["candidate_id"],
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "live_rtl_source_sha256": sha256_file(RTL),
        "checklist": CHECKLIST if status == "done" else raw.get("checklist", {}),
        "checklist_rulings": checklist_rulings,
        "stage_closing": stage_closing,
        "scope": "standalone_candidate_rtl_checklist_only",
        "candidate_capability_accepted": False,
        "verified_evidence": verified,
        "producer_provenance": copy.deepcopy(packet["implementation_provenance"]),
        "reason": reason,
        "required_remediation": remediation,
        "reviewer_thread_id": raw.get("thread_id", ""),
        "raw_decision": raw,
    }
    dump(OUT, payload)
    if status == "done":
        bind_done(payload)
        print(f"ACE2_QECR_RTL_REVIEW_PASS candidate={packet['candidate_id']} stage_closing={str(stage_closing).lower()}")
        return 0
    print(f"ACE2_QECR_RTL_REVIEW_REPLAN candidate={packet['candidate_id']} remediation={payload['required_remediation']}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
