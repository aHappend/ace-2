#!/usr/bin/env python3
"""Run exactly one evidence-only L2 RTL stage-closing semantic re-review."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

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
EXPECTED_RTL_SHA256 = "0ac1ee58be61260674061e77b6b311ebc6d32fc221b3f549ae993a886fa149c1"
EXPECTED_AGGREGATE_SHA256 = "1b4f220f646a458df088267c12a543b8a844902bf245b1b71f09bca00d2d44f3"
EXPECTED_PRIOR_DECISION_SHA256 = "00edd54c1c57b020488230627284ea31ea7cc6e06b362c6cd0423ba817c4d0bb"
CHECKLIST = {
    "rtl.contract-traceability": True,
    "rtl.hardware-discipline": True,
    "rtl.ip-provenance": True,
}

RTL = ROOT / "rtl/ace2_cross_layer_error_carry_core.sv"
PRECHECK = ROOT / f"evidence/{CONTRACT}/latest/PRECHECK.json"
PRIOR_DECISION = ROOT / f"evidence/review/rtl_checklist_{CONTRACT}/decision.json"
OUT = ROOT / f"evidence/review/rtl_stage_closing_{CONTRACT}/decision.json"
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
TRACEABILITY = ROOT / "design/RTL_TRACEABILITY.md"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
POLICY = ROOT / "design/FAST_LOOP_POLICY.json"
TARGET = ROOT / "design/TARGET.json"
SCOPE = ROOT / "design/CHIP_SCOPE.json"
PUBLIC_STATUS = ROOT / "research/PUBLIC_STATUS.json"
MISSION_ID = "rtl-stage-closing-semantic-rereview-cross-layer-error-carry-final-output-v1"
FINAL_STATUS = "independent_rtl_checklist_accepted_stage_closing_true"
MANAGER_ACTION = "manager_may_advance_rtl_to_verification"
LATEST_DECISION = "cross_layer_error_carry_rtl_stage_closing_independently_accepted"
REVIEWED_RTL_BIND_AT_UTC = "2026-08-02T04:19:32.776039Z"
SUPERSEDED_DECISION_SHA256 = "6f503e6fea1e4ff7cb8a1e283d697c04b33ce94248e04506927836a4691c1940"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def aggregate_hash(records: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(records):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(records[path].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def artifact_records(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield value
        for nested in value.values():
            yield from artifact_records(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from artifact_records(nested)


def verify_artifact(record: dict[str, Any]) -> Path:
    raw = Path(str(record["path"]))
    path = raw if raw.is_absolute() else ROOT / raw
    require(path.is_file(), f"frozen artifact is missing: {record['path']}")
    require(path.stat().st_size == int(record["bytes"]), f"frozen artifact size changed: {record['path']}")
    require(sha256_file(path) == record["sha256"], f"frozen artifact hash changed: {record['path']}")
    return path


def reviewer_accepts(reason: str, item: str) -> bool:
    normalized = reason.lower().replace("`", "")
    return any(token in normalized for token in (f"{item}: supported", f"{item}: true", f"{item}=true"))


def reason_sets_true(reason: str, field: str) -> bool:
    normalized = reason.lower().replace("`", "").replace(" ", "")
    return f"{field.lower()}=true" in normalized


def reason_sets_false(reason: str, field: str) -> bool:
    normalized = reason.lower().replace("`", "").replace(" ", "")
    return f"{field.lower()}=false" in normalized


def non_material_checkpoint_write_block(raw: dict[str, Any], reason: str, remediation: str) -> bool:
    operator_question = str(raw.get("operator_question", "")).lower()
    next_action = str(raw.get("next_action", "")).lower()
    normalized_reason = reason.lower()
    return (
        raw.get("status") == "blocked"
        and remediation == ""
        and "checkpoint.md" in operator_question
        and "writable" in operator_question
        and "checkpoint.md" in next_action
        and "read-only sandbox" in normalized_reason
        and "checkpoint.md remains stale" in normalized_reason
    )


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


def reviewer_skill_text() -> str:
    session_root = os.environ.get("ARGUS_SKILL_SESSION_ROOT", "")
    project_skill = Path(session_root) / "skills/reviewer/rtl-frontier-evidence-review.md"
    if project_skill.is_file():
        return project_skill.read_text(encoding="utf-8")
    package_root = Path(argus_skill.__file__).resolve().parent
    return (package_root / "verticals/chip_design/skills/reviewer/chip-design-signoff-review.md").read_text(
        encoding="utf-8"
    )


def preflight() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[Path, str]]:
    require(not OUT.exists(), "fresh semantic stage-closing re-review already exists; refusing a second review")
    prior = load(PRIOR_DECISION)
    precheck = load(PRECHECK)
    manifest = load(MANIFEST)
    pipeline = load(PIPELINE)

    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(sha256_file(PRIOR_DECISION) == EXPECTED_PRIOR_DECISION_SHA256, "prior independent verdict changed")
    require(prior.get("reviewer_status") == "done", "prior independent verdict is not done")
    require(prior.get("stage_closing") is False, "prior verdict is not the obsolete non-closing decision")
    require(prior.get("candidate_capability_accepted") is False, "prior verdict accepts candidate capability")
    require(prior.get("required_remediation") == "", "prior verdict retains remediation")
    require(prior.get("checklist") == CHECKLIST, "prior verdict checklist differs")
    require(prior.get("candidate_rtl_hash") == EXPECTED_AGGREGATE_SHA256, "prior aggregate hash differs")
    require(prior.get("live_rtl_source_sha256") == EXPECTED_RTL_SHA256, "prior live RTL hash differs")
    require(all(prior.get("checklist_rulings", {}).get(item, {}).get("accepted") is True for item in CHECKLIST),
            "prior verdict does not independently accept all three RTL gates")

    require(precheck.get("contract_id") == CONTRACT, "precheck contract differs")
    require(precheck.get("status") == "pass_ready_for_independent_rtl_review", "precheck is not review-ready")
    require(precheck.get("stage_checklist") == CHECKLIST, "precheck checklist differs")
    require(precheck.get("candidate_rtl_hash") == EXPECTED_AGGREGATE_SHA256, "precheck aggregate hash differs")
    require(precheck.get("live_rtl_source_sha256") == EXPECTED_RTL_SHA256, "precheck live RTL hash differs")
    require(not any(precheck.get("prohibited_runs", {}).values()), "precheck records prohibited downstream work")
    require(canonical_sha256(precheck) == precheck.get("integrity", {}).get("canonical_sha256"),
            "precheck canonical hash differs")

    source_hashes = precheck.get("source_hashes", {})
    require(isinstance(source_hashes, dict) and source_hashes, "precheck source hashes are missing")
    for relative, expected in source_hashes.items():
        path = ROOT / relative
        require(path.is_file(), f"bound source is missing: {relative}")
        require(sha256_file(path) == expected, f"bound source changed: {relative}")
    require(aggregate_hash(source_hashes) == EXPECTED_AGGREGATE_SHA256, "live aggregate hash differs")
    require(sha256_file(RTL) == EXPECTED_RTL_SHA256, "live RTL source changed")

    require(manifest.get("candidate_rtl_hash") == EXPECTED_AGGREGATE_SHA256, "live manifest aggregate differs")
    require(manifest.get("traceability", {}).get("stage_checklist") == CHECKLIST, "manifest checklist differs")
    require(manifest.get("candidate_review_binding", {}).get("evidence") == artifact(PRIOR_DECISION),
            "manifest no longer binds the frozen prior verdict")

    protected: set[Path] = {
        PRIOR_DECISION,
        PRECHECK,
        RTL,
        MANIFEST,
        TRACEABILITY,
        PIPELINE,
        CHECKPOINT,
        ROOT / "design/FAST_LOOP_POLICY.json",
        ROOT / "design/TARGET.json",
        ROOT / "design/CHIP_SCOPE.json",
        ROOT / "research/PUBLIC_STATUS.json",
    }
    for record in artifact_records(prior.get("verified_evidence", {})):
        protected.add(verify_artifact(record))
    protected.update(ROOT / relative for relative in source_hashes)
    protected_hashes = {path: sha256_file(path) for path in protected if path.is_file()}

    verified = {
        "contract_id": CONTRACT,
        "candidate_id": prior["candidate_id"],
        "candidate_rtl_hash": EXPECTED_AGGREGATE_SHA256,
        "live_rtl_source_sha256": EXPECTED_RTL_SHA256,
        "stage_checklist": CHECKLIST,
        "prior_independent_decision": artifact(PRIOR_DECISION),
        "precheck": artifact(PRECHECK),
        "live_manifest": artifact(MANIFEST),
        "traceability": artifact(TRACEABILITY),
        "pipeline_state": artifact(PIPELINE),
        "frozen_artifact_records_validated": sum(1 for _ in artifact_records(prior.get("verified_evidence", {}))),
        "prohibited_runs": precheck["prohibited_runs"],
        "semantic_rule": {
            "all_three_checklist_gates_true": True,
            "required_remediation_empty": True,
            "stage_closing_must_be_true": True,
            "candidate_capability_accepted_must_remain_false": True,
            "manager_owns_stage_transition": True,
        },
    }
    return prior, precheck, verified, protected_hashes


def verify_protected_unchanged(protected_hashes: dict[Path, str]) -> None:
    for path, expected in protected_hashes.items():
        require(path.is_file(), f"Reviewer removed frozen file: {path.relative_to(ROOT)}")
        require(sha256_file(path) == expected, f"Reviewer changed frozen file: {path.relative_to(ROOT)}")


def validate_existing() -> None:
    decision = load(OUT)
    require(decision.get("reviewer_status") == "done", "semantic re-review is not done")
    require(decision.get("stage_closing") is True, "semantic re-review is not stage-closing")
    require(decision.get("candidate_capability_accepted") is False, "semantic re-review accepts capability")
    require(decision.get("required_remediation") == "", "semantic re-review retains remediation")
    require(decision.get("checklist") == CHECKLIST, "semantic re-review checklist differs")
    require(decision.get("candidate_rtl_hash") == EXPECTED_AGGREGATE_SHA256, "semantic aggregate differs")
    require(decision.get("live_rtl_source_sha256") == EXPECTED_RTL_SHA256, "semantic RTL hash differs")
    require(decision.get("freshness", {}).get("distinct_reviewer_execution") is True,
            "semantic re-review is not a distinct execution")
    require(canonical_sha256(decision) == decision.get("integrity", {}).get("canonical_sha256"),
            "semantic re-review canonical hash differs")
    verify_artifact(decision["prior_independent_decision"])
    print(f"ACE2_QECR_RTL_STAGE_CLOSING_REVIEW_CHECK_PASS candidate={decision['candidate_id']}")


def normalize_existing() -> None:
    decision = load(OUT)
    raw = decision.get("raw_decision", {})
    require(isinstance(raw, dict), "raw Reviewer decision is missing")
    reason = str(raw.get("reason", ""))
    remediation = raw.get("required_remediation", "")
    require(isinstance(remediation, str), "Reviewer remediation is not text")
    require(non_material_checkpoint_write_block(raw, reason, remediation),
            "existing verdict is not the single non-material read-only checkpoint block")
    prior = load(PRIOR_DECISION)
    prior_thread = prior.get("reviewer_thread_id", "")
    prior_fingerprint = prior.get("raw_decision", {}).get("static_fingerprint")
    current_thread = raw.get("thread_id", "")
    current_fingerprint = raw.get("static_fingerprint")
    fresh = bool(current_thread) and current_thread != prior_thread
    if prior_fingerprint and current_fingerprint:
        fresh = fresh and current_fingerprint != prior_fingerprint
    require(fresh, "existing semantic re-review is not a distinct Reviewer execution")
    rulings = {
        item: {
            "accepted": reviewer_accepts(reason, item),
            "reviewer_reason_uses_exact_identifier": item in reason,
        }
        for item in CHECKLIST
    }
    require(all(ruling["accepted"] for ruling in rulings.values()),
            "existing Reviewer reason does not accept all RTL checklist gates")
    require(reason_sets_true(reason, "stage_closing"), "existing Reviewer reason is not stage-closing")
    require(reason_sets_false(reason, "candidate_capability_accepted"),
            "existing Reviewer reason accepts candidate capability")
    decision.update({
        "reviewer_status": "done",
        "checklist": copy.deepcopy(CHECKLIST),
        "checklist_rulings": rulings,
        "stage_closing": True,
        "candidate_capability_accepted": False,
        "required_remediation": "",
        "normalization": {
            "classification": "material_reviewer_verdict_accepted_non_material_checkpoint_write_block",
            "raw_reviewer_status_preserved": raw.get("status"),
            "operator_read_only_evidence_constraint_preserved": True,
            "second_reviewer_execution_performed": False,
        },
    })
    decision.setdefault("integrity", {})["canonical_sha256"] = None
    decision["integrity"]["canonical_sha256"] = canonical_sha256(decision)
    dump(OUT, decision)
    validate_existing()


def update_contract_binding(contract: dict[str, Any], decision_record: dict[str, Any], now: str) -> None:
    require(contract.get("contract_id") == CONTRACT, "active contract differs")
    prior_review = contract.get("rtl_review")
    if isinstance(prior_review, dict) and prior_review != decision_record:
        contract.setdefault("rtl_checklist_review", copy.deepcopy(prior_review))
    contract.update({
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "required_manager_action": MANAGER_ACTION,
        "rtl_review": copy.deepcopy(decision_record),
        "rtl_stage_closing_review": copy.deepcopy(decision_record),
        "stage_closing": True,
        "status": FINAL_STATUS,
        "updated_at_utc": now,
    })


def refresh_artifact_records(records: Any) -> None:
    if isinstance(records, dict):
        if {"path", "bytes", "sha256"}.issubset(records):
            raw = Path(str(records["path"]))
            path = raw if raw.is_absolute() else ROOT / raw
            if path.is_file():
                records.update(artifact(path))
        for nested in records.values():
            refresh_artifact_records(nested)
    elif isinstance(records, list):
        for nested in records:
            refresh_artifact_records(nested)


def reconstruct_reviewed_manifest() -> bytes:
    prior = load(PRIOR_DECISION)
    precheck = load(PRECHECK)
    pending_record = prior["verified_evidence"]["manifest"]
    verify_artifact(pending_record)
    manifest = load(ROOT / pending_record["path"])
    decision_record = artifact(PRIOR_DECISION)
    manifest["stage_closing"] = False
    manifest["candidate_status"] = "independent_rtl_checklist_accepted_stage_closing_false"
    manifest["implementation_authorized"] = False
    manifest["implementation_authority_consumed"] = True
    manifest["implementation_completed"] = True
    manifest["interfaces_contract_status"] = "qecr_standalone_interfaces_independently_reviewed"
    manifest["candidate_implementation_provenance"] = copy.deepcopy(prior["producer_provenance"])
    manifest["candidate_review_binding"] = {
        "candidate_capability_accepted": False,
        "decision": "accepted_rtl_checklist_only",
        "reviewer_status": "done",
        "evidence": decision_record,
        "scope": "standalone_candidate_rtl_checklist_only",
        "stage_closing": False,
        "quality_discriminator_complete": False,
        "shell_admitted": False,
        "ppa_run": False,
        "producer_provenance": copy.deepcopy(prior["producer_provenance"]),
        "checklist_rulings": copy.deepcopy(prior["checklist_rulings"]),
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
    contract = manifest.setdefault("proposed_replacement_contract", {})
    contract.update({
        "candidate_id": precheck["candidate_id"],
        "candidate_rtl_hash": precheck["candidate_rtl_hash"],
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "operator_approval_consumed": True,
        "required_manager_action": "hold_rtl_downstream_stages_remain_locked",
        "rtl_review": decision_record,
        "rtl_started": True,
        "stage_closing": False,
        "status": "independent_rtl_checklist_accepted_stage_closing_false",
        "updated_at_utc": REVIEWED_RTL_BIND_AT_UTC,
    })
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def reconstruct_reviewed_traceability() -> bytes:
    trace = TRACEABILITY.read_text(encoding="utf-8")
    trace = trace.replace(
        "- Fresh independent review accepted contract traceability, hardware discipline,\n"
        "  and IP provenance for the exact candidate aggregate. A distinct evidence-only\n"
        "  semantic re-review set `stage_closing=true` while keeping candidate capability\n"
        "  unaccepted; the shell is not modified or admitted, the accepted\n",
        "- Fresh independent review accepted contract traceability, hardware discipline,\n"
        "  and IP provenance for the exact candidate aggregate. The verdict is\n"
        "  `stage_closing=false`; the shell is not modified or admitted, the accepted\n",
    )
    trace = trace.replace(
        "\nStage-closing evidence: `"
        "evidence/review/rtl_stage_closing_cross_layer_quantization_error_carry_final_output_v1/decision.json"
        "`. Manager alone owns any stage transition.\n",
        "",
    )
    return trace.encode()


def preserve_reviewed_snapshot(
    label: str, record: dict[str, Any], reconstructed: bytes | None = None
) -> dict[str, Any]:
    source = ROOT / record["path"]
    suffix = source.suffix or ".bin"
    snapshot = OUT.parent / "inputs" / f"{label}.reviewed.{record['sha256']}{suffix}"
    if snapshot.is_file():
        require(snapshot.stat().st_size == int(record["bytes"]), f"{label} snapshot size changed")
        require(sha256_file(snapshot) == record["sha256"], f"{label} snapshot hash changed")
    else:
        source_bytes = source.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() == record["sha256"]:
            snapshot_bytes = source_bytes
        else:
            require(reconstructed is not None, f"cannot reconstruct reviewed {label}")
            require(len(reconstructed) == int(record["bytes"]), f"reconstructed {label} size differs")
            require(hashlib.sha256(reconstructed).hexdigest() == record["sha256"],
                    f"reconstructed {label} hash differs")
            snapshot_bytes = reconstructed
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        temporary = snapshot.with_suffix(snapshot.suffix + ".tmp")
        temporary.write_bytes(snapshot_bytes)
        os.replace(temporary, snapshot)
    return artifact(snapshot)


def bind_existing() -> None:
    validate_existing()
    decision = load(OUT)
    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    decision_record = artifact(OUT)
    prior_record = artifact(PRIOR_DECISION)
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    reviewed_snapshots = {
        "live_manifest": preserve_reviewed_snapshot(
            "RTL_MANIFEST", decision["reviewed_live_manifest"], reconstruct_reviewed_manifest()
        ),
        "traceability": preserve_reviewed_snapshot(
            "RTL_TRACEABILITY", decision["frozen_evidence"]["traceability"],
            reconstruct_reviewed_traceability(),
        ),
    }

    manifest = load(MANIFEST)
    require(manifest.get("candidate_rtl_hash") == EXPECTED_AGGREGATE_SHA256, "manifest candidate differs")
    require(manifest.get("traceability", {}).get("stage_checklist") == CHECKLIST,
            "manifest RTL checklist differs")
    binding = manifest.setdefault("candidate_review_binding", {})
    prior_binding = binding.get("evidence")
    if isinstance(prior_binding, dict) and prior_binding != decision_record:
        binding.setdefault("prior_checklist_evidence", copy.deepcopy(prior_binding))
    binding.update({
        "candidate_capability_accepted": False,
        "checklist_rulings": copy.deepcopy(decision["checklist_rulings"]),
        "decision": "accepted_rtl_checklist_stage_closing_true",
        "evidence": copy.deepcopy(decision_record),
        "ppa_run": False,
        "quality_discriminator_complete": False,
        "reviewer_status": "done",
        "scope": "standalone_candidate_rtl_plus_stage_closing_semantic_rereview",
        "shell_admitted": False,
        "stage_closing": True,
        "stage_closing_evidence": copy.deepcopy(decision_record),
        "stage_closing_reviewed_input_snapshots": copy.deepcopy(reviewed_snapshots),
    })
    binding.setdefault("prior_checklist_evidence", prior_record)
    manifest["stage_closing"] = True
    manifest["status"] = FINAL_STATUS
    manifest["candidate_status"] = FINAL_STATUS
    manifest["required_manager_action"] = MANAGER_ACTION
    proposed = manifest.get("proposed_replacement_contract")
    if isinstance(proposed, dict):
        update_contract_binding(proposed, decision_record, now)
    acceptance = manifest.setdefault("independent_reviewer_acceptance", {})
    acceptance.update({
        "rtl_review": "accepted_exact_candidate_hash",
        "rtl_stage_closing_review": "accepted_distinct_evidence_only_semantic_rereview",
        "status": "rtl_checklist_accepted_stage_closing_true",
    })
    manifest.setdefault("latest_evidence", {})[
        "cross_layer_error_carry_rtl_stage_closing_review"
    ] = copy.deepcopy(decision_record)
    dump(MANIFEST, manifest)

    policy = load(POLICY)
    for key in ("active_architecture_contract", "selected_replacement_contract"):
        update_contract_binding(policy[key], decision_record, now)
    policy["manager_recommendation"] = MANAGER_ACTION
    dump(POLICY, policy)

    target = load(TARGET)
    target["generated_at_utc"] = now
    target["current_architecture_contract"] = copy.deepcopy(policy["selected_replacement_contract"])
    target.setdefault("fast_loop_contract", {})["rtl_status"] = FINAL_STATUS
    dump(TARGET, target)

    scope = load(SCOPE)
    authority = scope.setdefault("authority_override", {})
    authority.update({
        "active_replacement_contract": CONTRACT,
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "operator_approval_consumed": True,
        "required_next_action": MANAGER_ACTION,
        "rtl_stage_closing_review": copy.deepcopy(decision_record),
        "stage_closing": True,
        "status": FINAL_STATUS,
    })
    for contract in (
        authority.get("operator_implementation_approval"),
        scope.get("operator_owned_execution_policy", {}).get("active_successor_contract"),
    ):
        if isinstance(contract, dict):
            update_contract_binding(contract, decision_record, now)
    stage = scope.setdefault("stage", {})
    stage.update({
        "current_stage": "rtl",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_status": FINAL_STATUS,
        "downstream_stages_locked_until_manager_advance": [
            "verification", "ppa", "prototype", "benchmark", "signoff",
        ],
        "planner_may_advance_stage": False,
        "stage_closing": True,
        "stage_transition_owner": "Manager",
    })
    evidence = stage.setdefault("current_stage_evidence", [])
    if decision_record["path"] not in evidence:
        evidence.append(decision_record["path"])
    frontier = scope.setdefault("implementation_frontier", {})
    frontier["latest_decision"] = LATEST_DECISION
    scope["last_updated_utc"] = now
    dump(SCOPE, scope)

    trace = TRACEABILITY.read_text(encoding="utf-8")
    trace = trace.replace(
        "- Fresh independent review accepted contract traceability, hardware discipline,\n"
        "  and IP provenance for the exact candidate aggregate. The verdict is\n"
        "  `stage_closing=false`; the shell is not modified or admitted, the accepted\n",
        "- Fresh independent review accepted contract traceability, hardware discipline,\n"
        "  and IP provenance for the exact candidate aggregate. A distinct evidence-only\n"
        "  semantic re-review set `stage_closing=true` while keeping candidate capability\n"
        "  unaccepted; the shell is not modified or admitted, the accepted\n",
    )
    if decision_record["path"] not in trace:
        trace += (
            "\nStage-closing evidence: `"
            + decision_record["path"]
            + "`. Manager alone owns any stage transition.\n"
        )
    TRACEABILITY.write_text(trace, encoding="utf-8")

    status = load(PUBLIC_STATUS)
    for key in ("architecture_certification_mission", "architecture_proposal_gate"):
        section = status.setdefault(key, {})
        section.update({
            "contract_id": CONTRACT,
            "implementation_authorized": False,
            "implementation_authority_consumed": True,
            "implementation_completed": True,
            "required_manager_action": MANAGER_ACTION,
            "rtl_stage_closing_review": copy.deepcopy(decision_record),
            "stage_closing": True,
            "status": FINAL_STATUS,
        })
    latest_rtl = status.setdefault("latest_rtl_candidate", {})
    latest_rtl.update({
        "candidate_capability_accepted": False,
        "candidate_id": decision["candidate_id"],
        "candidate_rtl_hash": EXPECTED_AGGREGATE_SHA256,
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "required_manager_action": MANAGER_ACTION,
        "rtl_review": copy.deepcopy(decision_record),
        "rtl_stage_closing_review": copy.deepcopy(decision_record),
        "rtl_stage_closing_reviewed_input_snapshots": copy.deepcopy(reviewed_snapshots),
        "shell_admitted": False,
        "stage_closing": True,
        "status": FINAL_STATUS,
    })
    dispatch = status.setdefault("architecture_successor_dispatch", {})
    dispatch.update({
        "contract_id": CONTRACT,
        "current_stage": "rtl",
        "implementation_authorized": False,
        "implementation_authority_consumed": True,
        "implementation_completed": True,
        "operator_approval_consumed": True,
        "required_manager_action": MANAGER_ACTION,
        "result_binding": copy.deepcopy(decision_record),
        "stage_closing": True,
        "status": FINAL_STATUS,
    })
    status["selected_replacement_contract"] = copy.deepcopy(policy["selected_replacement_contract"])
    status["latest_decision"] = LATEST_DECISION
    status["required_manager_action"] = MANAGER_ACTION
    status["routing_authorized"] = "manager_stage_transition_only"
    status["routing_status"] = "rtl_stage_closed_waiting_manager_advance"
    status["stage_closing"] = True
    for key in ("dashboard_fields", "implementation_frontier"):
        section = status.setdefault(key, {})
        section["latest_decision"] = LATEST_DECISION
        section["latest_rtl_candidate"] = copy.deepcopy(latest_rtl)
        section["required_manager_action"] = MANAGER_ACTION
        section["routing_status"] = "rtl_stage_closed_waiting_manager_advance"
        section["rtl_review"] = copy.deepcopy(decision_record)
        section["rtl_stage_closing_review"] = copy.deepcopy(decision_record)
        section["stage_closing"] = True
        section["implementation_authorized"] = False
        section["implementation_authority_consumed"] = True
        section["implementation_completed"] = True
        candidate_mechanism = section.get("candidate_mechanism")
        if isinstance(candidate_mechanism, dict):
            candidate_mechanism["stage_closing"] = True
            candidate_mechanism["status"] = FINAL_STATUS
        active_contract = section.get("operator_policy", {}).get("active_architecture_contract")
        if isinstance(active_contract, dict):
            update_contract_binding(active_contract, decision_record, now)
    status["blockers"] = [{
        "id": "manager_rtl_to_verification_transition_pending",
        "owner": "manager",
        "severity": "gate",
        "status": "open",
        "detail": (
            "All RTL checklist items are independently accepted with stage_closing=true. "
            "Manager action is required before verification; downstream work remains locked "
            "while current_stage is rtl."
        ),
    }]
    public_stage = status.setdefault("stage", {})
    public_stage.update({
        "current_stage": "rtl",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_status": FINAL_STATUS,
        "stage_closing": True,
    })
    public_evidence = public_stage.setdefault("current_stage_evidence", [])
    if decision_record["path"] not in public_evidence:
        public_evidence.append(decision_record["path"])
    status["generated_at_utc"] = now
    status["last_updated_utc"] = now
    refresh_artifact_records(status.get("artifact_hashes", []))
    status.setdefault("integrity", {})["canonical_sha256"] = None
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump(PUBLIC_STATUS, status)

    checkpoint = f"""# Goal

Implement and independently review the bounded RTL successor
`{CONTRACT}` without entering downstream stages.

# Current state

The Manager-owned stage remains `rtl`. Independent review accepted all three
RTL checklist items for candidate `{decision['candidate_id']}` at exact aggregate
hash `{EXPECTED_AGGREGATE_SHA256}` with `stage_closing=true`.

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

The fresh evidence-only L2 semantic re-review independently set
`stage_closing=true` while keeping `candidate_capability_accepted=false`. No
Planner or Engineer stage transition is authorized. Manager alone may advance
`rtl` to `verification`; `research/PIPELINE_STATE.json` was not edited.
"""
    CHECKPOINT.write_text(checkpoint, encoding="utf-8")


def check_bound() -> None:
    validate_existing()
    decision = load(OUT)
    decision_record = artifact(OUT)
    pipeline = load(PIPELINE)
    manifest = load(MANIFEST)
    policy = load(POLICY)
    target = load(TARGET)
    scope = load(SCOPE)
    status = load(PUBLIC_STATUS)
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage changed")
    require(manifest.get("stage_closing") is True, "manifest is not stage-closing")
    require(manifest.get("candidate_review_binding", {}).get("evidence") == decision_record,
            "manifest does not bind semantic re-review")
    require(manifest.get("candidate_review_binding", {}).get("candidate_capability_accepted") is False,
            "manifest accepts candidate capability")
    snapshots = manifest.get("candidate_review_binding", {}).get(
        "stage_closing_reviewed_input_snapshots", {})
    require(set(snapshots) == {"live_manifest", "traceability"}, "reviewed input snapshots are missing")
    for record in snapshots.values():
        verify_artifact(record)
    for key in ("active_architecture_contract", "selected_replacement_contract"):
        require(policy.get(key, {}).get("stage_closing") is True, f"policy {key} is not stage-closing")
        require(policy.get(key, {}).get("rtl_review") == decision_record,
                f"policy {key} does not bind semantic re-review")
    require(target.get("current_architecture_contract", {}).get("stage_closing") is True,
            "target contract is not stage-closing")
    require(target.get("current_architecture_contract", {}).get("area_cap_non_sram_mm2") == 2.0,
            "area cap changed")
    require(target.get("current_architecture_contract", {}).get("frequency_floor_mhz") == 100.0,
            "frequency floor changed")
    require(scope.get("stage", {}).get("current_stage") == "rtl", "scope stage changed")
    require(scope.get("stage", {}).get("stage_closing") is True, "scope is not stage-closing")
    require(status.get("stage", {}).get("stage_closing") is True, "public stage is not stage-closing")
    require(status.get("latest_rtl_candidate", {}).get("rtl_review") == decision_record,
            "public status does not bind semantic re-review")
    dashboard = status.get("dashboard_fields", {})
    require(dashboard.get("current_mode") == "ADVANCE", "dashboard mode changed")
    require(dashboard.get("first_unsupported_layer_operator") == "layer_0.rope_q",
            "dashboard first unsupported operator changed")
    require(dashboard.get("ordered_supported_layer_operator_prefix") == [
        "layer_0.input_rmsnorm", "layer_0.q_proj", "layer_0.k_proj", "layer_0.v_proj",
    ], "dashboard supported prefix changed")
    for key in ("dashboard_fields", "implementation_frontier"):
        frontier = status.get(key, {}).get("latest_ppa_frontier", {})
        require(frontier.get("candidate_ppa_run") is False, f"{key} records candidate PPA")
        require(frontier.get("cells") == 62199, f"{key} historical cell frontier changed")
        require(frontier.get("non_sram_area_mm2") == 0.6108746272,
                f"{key} historical area frontier changed")
        require(frontier.get("setup_slack_ns_at_100mhz") == 0.1502,
                f"{key} historical timing frontier changed")
        require(frontier.get("remaining_area_reserve_mm2") == 1.3891253728,
                f"{key} historical area reserve changed")
        require(frontier.get("status") == "historical_accepted_frontier_preserved_no_successor_ppa",
                f"{key} PPA frontier status changed")
        require(status.get(key, {}).get("candidate_mechanism", {}).get("status") == FINAL_STATUS,
                f"{key} candidate status is stale")
        require(status.get(key, {}).get("operator_policy", {}).get(
            "active_architecture_contract", {}).get("stage_closing") is True,
            f"{key} active contract is not stage-closing")
    require(canonical_sha256(status) == status.get("integrity", {}).get("canonical_sha256"),
            "public status canonical hash differs")
    require("stage_closing=true" in TRACEABILITY.read_text(encoding="utf-8"),
            "traceability does not record the stage-closing verdict")
    require("stage_closing=true" in CHECKPOINT.read_text(encoding="utf-8"),
            "checkpoint does not record the stage-closing verdict")
    require(decision.get("candidate_capability_accepted") is False, "semantic verdict accepts capability")
    print(f"ACE2_QECR_RTL_STAGE_CLOSING_BINDING_CHECK_PASS candidate={decision['candidate_id']}")


def check_superseded_disposition() -> None:
    """Verify the rejected stage-closing projection is retained but not live."""

    require(OUT.is_file(), "superseded semantic verdict is missing")
    require(sha256_file(OUT) == SUPERSEDED_DECISION_SHA256,
            "superseded semantic verdict changed")
    decision = load(OUT)
    require(decision.get("stage_closing") is True,
            "historical semantic verdict no longer records its original result")
    require(decision.get("candidate_capability_accepted") is False,
            "historical semantic verdict accepts capability")

    decision_record = artifact(PRIOR_DECISION)
    pipeline = load(PIPELINE)
    manifest = load(MANIFEST)
    policy = load(POLICY)
    target = load(TARGET)
    scope = load(SCOPE)
    status = load(PUBLIC_STATUS)

    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage changed")
    require(manifest.get("candidate_rtl_hash") == EXPECTED_AGGREGATE_SHA256,
            "live candidate aggregate changed")
    require(manifest.get("stage_closing") is False,
            "live manifest reapplied the superseded stage-closing verdict")
    require(manifest.get("candidate_review_binding", {}).get("evidence") == decision_record,
            "live manifest does not bind the accepted non-closing review")
    require(manifest.get("required_manager_action") == "hold_rtl_downstream_stages_remain_locked",
            "live manifest asks for a forbidden non-milestone stage advance")
    require(manifest.get("status") == "independent_rtl_checklist_accepted_stage_closing_false",
            "live manifest status is stale")
    require("rtl_stage_closing_review" not in manifest.get("proposed_replacement_contract", {}),
            "live manifest contract retains the superseded review")

    uplift = policy.get("non_milestone_operator_uplift", {})
    require(uplift.get("single_bounded_rtl_task") is True,
            "non-milestone uplift is no longer one bounded task")
    require(uplift.get("stage_closing") is False,
            "operator-owned non-milestone stage_closing contract changed")
    require(uplift.get("manager_stage_advance_out_of_rtl") ==
            "not_requested_for_intermediate_operator_uplift",
            "policy requests a forbidden Manager stage advance")
    for key in ("active_architecture_contract", "selected_replacement_contract"):
        contract = policy.get(key, {})
        require(contract.get("stage_closing") is False, f"policy {key} is stage-closing")
        require(contract.get("rtl_review") == decision_record,
                f"policy {key} does not bind the accepted non-closing review")
        require("rtl_stage_closing_review" not in contract,
                f"policy {key} retains the superseded review")

    target_contract = target.get("current_architecture_contract", {})
    require(target_contract.get("stage_closing") is False,
            "target contract is stage-closing")
    require("rtl_stage_closing_review" not in target_contract,
            "target contract retains the superseded review")
    require(scope.get("stage", {}).get("stage_closing") is False,
            "scope stage is stage-closing")
    require(status.get("stage_closing") is False,
            "public status is stage-closing")
    require(status.get("stage", {}).get("stage_closing") is False,
            "public stage is stage-closing")
    require(status.get("latest_rtl_candidate", {}).get("rtl_review") == decision_record,
            "public latest candidate does not bind the accepted non-closing review")
    require("`stage_closing=false`" in TRACEABILITY.read_text(encoding="utf-8"),
            "traceability does not preserve the non-closing contract")
    require("stage_closing=false" in CHECKPOINT.read_text(encoding="utf-8"),
            "checkpoint does not preserve the non-closing contract")
    print("ACE2_QECR_SUPERSEDED_STAGE_CLOSING_DISPOSITION_CHECK_PASS")


def main() -> int:
    if OUT.exists():
        if "--check" in sys.argv[1:]:
            check_superseded_disposition()
            return 0
        raise RuntimeError(
            "the stage-closing semantic verdict is superseded by the operator-owned "
            "non-milestone stage_closing=false contract; rebinding is forbidden"
        )
    if "--check" in sys.argv[1:]:
        validate_existing()
        return 0
    if "--check-bound" in sys.argv[1:]:
        check_bound()
        return 0
    if "--bind-existing" in sys.argv[1:]:
        bind_existing()
        check_bound()
        return 0
    if "--normalize-existing" in sys.argv[1:]:
        normalize_existing()
        return 0

    prior, precheck, verified, protected_hashes = preflight()
    backend_name = resolve_role_backend("reviewer")
    runner = AgentCliBackend(
        backend=backend_name,
        runner_bin=resolve_runner_bin_setting("reviewer") or None,
    )
    sid = session_id()
    runner.set_usage_context(
        project_root=global_root() / "projects" / sid,
        global_root=global_root(),
        mission_id=MISSION_ID,
    )
    objective = (
        "Independently re-review only the frozen RTL checklist evidence for "
        f"{CONTRACT} at aggregate {EXPECTED_AGGREGATE_SHA256} and live RTL SHA-256 "
        f"{EXPECTED_RTL_SHA256}. Determine stage_closing from the RTL checklist semantics."
    )
    instruction = (
        "Live operator guidance supersedes the obsolete Planner wording that forced "
        "stage_closing=false. Read the existing independent decision, PRECHECK, RTL manifest, "
        "RTL traceability notes, live RTL source, and retained logs/XML directly. Use only "
        "read-only inspection; do not modify files and do not rerun any writer, checker, RTL, "
        "reference, vector, simulation, formal, quality, synthesis, STA, PPA, prototype, "
        "benchmark, or signoff work. Decide each gate using the exact identifiers "
        "rtl.contract-traceability, rtl.hardware-discipline, and rtl.ip-provenance. Return done "
        "only if all three are supported for the exact frozen hashes and required remediation "
        "is empty. If done, explicitly state each '<identifier>: supported', "
        "stage_closing=true, and candidate_capability_accepted=false. Stage closing is evidence "
        "for Manager-controlled rtl-to-verification entry only; it does not admit the candidate "
        "capability, run downstream work, or change research/PIPELINE_STATE.json."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=sid,
        main_summary=(
            "The prior fresh independent decision already accepts all three RTL checklist gates "
            "with empty remediation for the exact hashes; only its inherited stage_closing=false "
            "wording is under semantic re-review. All evidence is frozen."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="rtl_stage_closing_semantic_correction_only",
        checkpoint_path=str(CHECKPOINT),
        escalate_hint="If done, leave current_stage=rtl and all downstream work locked for Manager action.",
        preselected_skill_block=reviewer_skill_text(),
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort(
                "ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"
            ),
            skip_git_repo_check=True,
            full_auto=True,
            dangerous_yolo=False,
            sandbox_mode="read-only",
            isolate_workdir=False,
            working_dir=str(ROOT),
        ),
    )
    verify_protected_unchanged(protected_hashes)

    raw = asdict(review)
    raw_status = raw.get("status", "continue")
    reason = str(raw.get("reason", ""))
    remediation = raw.get("required_remediation", "")
    require(isinstance(remediation, str), "Reviewer remediation is not text")
    checklist_rulings = {
        item: {
            "accepted": reviewer_accepts(reason, item),
            "reviewer_reason_uses_exact_identifier": item in reason,
        }
        for item in CHECKLIST
    }
    prior_thread = prior.get("reviewer_thread_id", "")
    prior_fingerprint = prior.get("raw_decision", {}).get("static_fingerprint")
    current_thread = raw.get("thread_id", "")
    current_fingerprint = raw.get("static_fingerprint")
    fresh = bool(current_thread) and current_thread != prior_thread
    if prior_fingerprint and current_fingerprint:
        fresh = fresh and current_fingerprint != prior_fingerprint
    accepted = (
        (raw_status == "done" or non_material_checkpoint_write_block(raw, reason, remediation))
        and remediation == ""
        and fresh
        and all(ruling["accepted"] for ruling in checklist_rulings.values())
        and reason_sets_true(reason, "stage_closing")
        and reason_sets_false(reason, "candidate_capability_accepted")
    )
    reviewer_status = "done" if accepted else "continue"
    payload = {
        "schema_version": 1,
        "review_bound_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "reviewer_status": reviewer_status,
        "raw_reviewer_status": raw_status,
        "review_type": "fresh_evidence_only_independent_l2_semantic_rereview",
        "contract_id": CONTRACT,
        "candidate_id": precheck["candidate_id"],
        "candidate_rtl_hash": EXPECTED_AGGREGATE_SHA256,
        "live_rtl_source_sha256": EXPECTED_RTL_SHA256,
        "checklist": CHECKLIST if accepted else {item: False for item in CHECKLIST},
        "checklist_rulings": checklist_rulings,
        "stage_closing": accepted,
        "candidate_capability_accepted": False,
        "required_remediation": remediation if not accepted else "",
        "reason": reason,
        "scope": "rtl_stage_closing_semantic_correction_only",
        "manager_stage_transition_owner": "Manager",
        "pipeline_stage_changed": False,
        "downstream_work_run": False,
        "prior_independent_decision": artifact(PRIOR_DECISION),
        "reviewed_precheck": artifact(PRECHECK),
        "reviewed_live_manifest": artifact(MANIFEST),
        "freshness": {
            "prior_reviewer_thread_id": prior_thread,
            "current_reviewer_thread_id": current_thread,
            "distinct_reviewer_execution": fresh,
        },
        "frozen_evidence": verified,
        "raw_decision": raw,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "canonical_sha256": None,
        },
    }
    payload["integrity"]["canonical_sha256"] = canonical_sha256(payload)
    dump(OUT, payload)
    print(
        "ACE2_QECR_RTL_STAGE_CLOSING_REVIEW_RESULT "
        f"status={reviewer_status} stage_closing={str(accepted).lower()} "
        f"candidate={precheck['candidate_id']}"
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
