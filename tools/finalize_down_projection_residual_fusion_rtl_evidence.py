#!/usr/bin/env python3
"""Finalize and check DPRF RTL evidence after the fresh stage-closing review."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from validate_down_projection_residual_fusion_evidence import (
    ValidationSummary,
    validate_artifact_tree,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_down_projection_residual_fusion_v1"
FINAL_STATUS = "rtl_stage_closing_review_accepted_waiting_manager_advance"
FINAL_DECISION = "down_projection_residual_fusion_rtl_checklist_independently_accepted"
FINAL_MANAGER_ACTION = "manager_may_advance_rtl_to_verification"
FINAL_ROUTING_STATUS = "manager_rtl_to_verification_transition_pending"
FINAL_ROUTING_AUTHORIZATION = "manager_advance_rtl_to_verification_pending"
PRECHECK = ROOT / "evidence/shared_down_projection_residual_fusion_v1/latest/PRECHECK.json"
DECISION = ROOT / "evidence/review/rtl_checklist_shared_down_projection_residual_fusion_v1/decision.json"
TARGET = ROOT / "design/TARGET.json"
FAST_LOOP = ROOT / "design/FAST_LOOP_POLICY.json"
CHIP_SCOPE = ROOT / "design/CHIP_SCOPE.json"
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
TRACEABILITY = ROOT / "design/RTL_TRACEABILITY.md"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
BINDER = ROOT / "tools/bind_down_projection_residual_fusion_architecture.py"
REVIEW_TOOL = ROOT / "tools/run_down_projection_residual_fusion_rtl_review.py"
SNAPSHOTS = ROOT / "evidence/shared_down_projection_residual_fusion_v1/transitions/snapshots"
CHECKLIST = {
    "rtl.contract-traceability": True,
    "rtl.hardware-discipline": True,
    "rtl.ip-provenance": True,
}
PENDING_VALUES = {
    "bounded_rtl_preflight_pass_review_pending",
    "candidate_preflight_pass_independent_review_pending",
    "environment_independently_accepted_manager_rtl_entry_pending",
    "fresh_independent_rtl_stage_closing_review_pending",
    "independent_review_done_manager_stage_transition_pending",
    "rtl_preflight_pass_independent_review_pending",
}


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path, *, scope: str, recursive: bool = False) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "binding_scope": scope,
        "content_validation": "recursive_json" if recursive else "whole_file_only",
    }


def verify_record(record: dict[str, Any], label: str) -> Path:
    relative = str(record.get("path") or "")
    path = ROOT / relative
    require(relative and not Path(relative).is_absolute(), f"{label}: invalid path")
    require(path.is_file(), f"{label}: missing {relative}")
    require(path.stat().st_size == int(record.get("bytes", -1)), f"{label}: bytes differ")
    require(sha256_file(path) == record.get("sha256"), f"{label}: hash differs")
    return path


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def nested(value: dict[str, Any], path: str) -> dict[str, Any]:
    node: Any = value
    for key in path.split("."):
        require(isinstance(node, dict) and key in node, f"projection missing: {path}")
        node = node[key]
    require(isinstance(node, dict), f"projection is not an object: {path}")
    return node


def archive_reviewed_precheck() -> Path:
    data = PRECHECK.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    path = SNAPSHOTS / f"PRECHECK.reviewed.{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, "reviewed PRECHECK snapshot collision")
    else:
        path.write_bytes(data)
    return path


def resolve_reviewed_precheck(decision: dict[str, Any], now: str) -> tuple[Path, dict[str, Any]]:
    record = decision.get("reviewed_precheck", {})
    require(isinstance(record, dict), "fresh RTL review has no reviewed PRECHECK binding")
    relative = str(record.get("path") or "")
    if relative == PRECHECK.relative_to(ROOT).as_posix():
        require(record.get("sha256") == sha256_file(PRECHECK), "fresh RTL review PRECHECK hash differs")
        reviewed_snapshot = archive_reviewed_precheck()
        decision["reviewed_precheck"] = artifact(reviewed_snapshot, scope="sealed_whole_file")
        decision["reviewed_precheck_role"] = "immutable_pre_review_input"
        decision["binding_finalized_at_utc"] = now
        decision.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
        decision["integrity"]["canonical_sha256"] = canonical_sha256(decision)
        dump(DECISION, decision)
        return reviewed_snapshot, load(reviewed_snapshot)

    reviewed_snapshot = verify_record(record, "fresh RTL review reviewed_precheck")
    require(
        decision.get("reviewed_precheck_role") == "immutable_pre_review_input",
        "fresh RTL review PRECHECK is not immutable pre-review input",
    )
    return reviewed_snapshot, load(reviewed_snapshot)


def bind_active_projection(
    projection: dict[str, Any],
    *,
    candidate_id: str,
    candidate_hash: str,
    decision_record: dict[str, Any],
) -> None:
    projection.update(
        {
            "candidate_id": candidate_id,
            "candidate_rtl_hash": candidate_hash,
            "contract_id": CONTRACT,
            # The one-time RTL implementation authority was consumed when this
            # candidate was built.  A completed RTL review authorizes only the
            # Manager-owned transition to verification, not another RTL edit.
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "projection_scope": "active_rtl_stage_projection",
            "required_manager_action": FINAL_MANAGER_ACTION,
            "rtl_review": copy.deepcopy(decision_record),
            "rtl_review_stage_closing": True,
            "stage_closing": False,
            "status": FINAL_STATUS,
        }
    )


def mark_historical(projection: dict[str, Any], scope: str, reason: str) -> None:
    projection["projection_scope"] = scope
    projection["historical_reason"] = reason


def reconcile_live_projections(reviewed_packet: dict[str, Any], now: str) -> None:
    candidate_id = reviewed_packet["candidate_id"]
    candidate_hash = reviewed_packet["candidate_rtl_hash"]
    decision_record = artifact(DECISION, scope="sealed_whole_file")

    target = load(TARGET)
    fast_loop = load(FAST_LOOP)
    chip_scope = load(CHIP_SCOPE)
    manifest = load(MANIFEST)
    public = load(PUBLIC)

    active_projections: Iterable[dict[str, Any]] = (
        nested(target, "current_architecture_contract"),
        nested(target, "fast_loop_contract.active_repair_authorization"),
        nested(target, "fast_loop_contract.architecture_proposal_authorization"),
        nested(target, "fast_loop_contract.selected_replacement_contract"),
        nested(fast_loop, "active_repair_authorization"),
        nested(fast_loop, "architecture_proposal_authorization"),
        nested(fast_loop, "selected_replacement_contract"),
        nested(chip_scope, "authority_override.operator_implementation_approval"),
        nested(chip_scope, "numerical_behavior.active_down_projection_residual_fusion_contract"),
        nested(manifest, "proposed_replacement_contract"),
        nested(public, "architecture_proposal_gate"),
        nested(public, "selected_replacement_contract"),
        nested(public, "latest_rtl_candidate"),
        nested(public, "dashboard_fields.latest_rtl_candidate"),
        nested(public, "implementation_frontier.latest_rtl_candidate"),
    )
    for projection in active_projections:
        bind_active_projection(
            projection,
            candidate_id=candidate_id,
            candidate_hash=candidate_hash,
            decision_record=decision_record,
        )

    target["fast_loop_contract"]["manager_recommendation"] = FINAL_MANAGER_ACTION
    fast_loop["manager_recommendation"] = FINAL_MANAGER_ACTION
    fast_loop["steady_state_stage"] = "rtl"

    authority = chip_scope["authority_override"]
    authority.update(
        {
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "required_next_action": FINAL_MANAGER_ACTION,
            "review": copy.deepcopy(decision_record),
            "rtl_review_stage_closing": True,
            "stage_closing": False,
            "status": FINAL_STATUS,
        }
    )
    chip_scope["implementation_frontier"]["latest_decision"] = FINAL_DECISION
    chip_scope["stage"].update(
        {
            "current_stage_status": FINAL_STATUS,
            "stage_closing": False,
        }
    )
    chip_scope["stage"]["current_stage_evidence"] = [
        "design/RTL_MANIFEST.json",
        "design/RTL_TRACEABILITY.md",
        PRECHECK.relative_to(ROOT).as_posix(),
        DECISION.relative_to(ROOT).as_posix(),
    ]

    manifest["implementation_authorized"] = False
    manifest["candidate_status"] = FINAL_STATUS
    manifest["candidate_review_binding"].update(
        {
            "decision": "independent_rtl_checklist_accepted",
            "reviewer_status": "done",
            "stage_closing": True,
        }
    )
    manifest["independent_reviewer_acceptance"].update(
        {
            "rtl_review": "accepted_exact_candidate_hash_stage_closing",
            "status": FINAL_STATUS,
        }
    )

    historical_architecture_reason = (
        "Architecture-stage review and planning values are retained only as historical context; "
        "the Manager subsequently advanced environment to RTL."
    )
    historical_environment_reason = (
        "Environment-stage completion is retained as a historical predecessor state; the Manager "
        "transition to RTL has already occurred."
    )
    mark_historical(
        public["architecture_certification_mission"],
        "historical_architecture_stage_projection",
        historical_architecture_reason,
    )
    for path in (
        "dashboard_fields.candidate_mechanism",
        "implementation_frontier.candidate_mechanism",
    ):
        mark_historical(
            nested(public, path),
            "historical_architecture_planning_projection",
            historical_architecture_reason,
        )
    for path in (
        "latest_environment_stage",
        "dashboard_fields.latest_environment_stage",
        "implementation_frontier.latest_environment_stage",
    ):
        mark_historical(
            nested(public, path),
            "historical_environment_stage_projection",
            historical_environment_reason,
        )

    for path in (
        "latest_rtl_candidate",
        "dashboard_fields.latest_rtl_candidate",
        "implementation_frontier.latest_rtl_candidate",
    ):
        projection = nested(public, path)
        projection["review"] = copy.deepcopy(decision_record)
        projection["derived_latency_cycles_per_lane"] = reviewed_packet["schedule"][
            "valid_cycles_per_output_lane"
        ]

    public["latest_decision"] = FINAL_DECISION
    public["routing_status"] = FINAL_ROUTING_STATUS
    public["routing_authorized"] = FINAL_ROUTING_AUTHORIZATION
    public["stage"].update(
        {
            "current_stage_status": FINAL_STATUS,
            "stage_closing": False,
        }
    )
    public["stage"]["current_stage_evidence"] = [
        "design/RTL_MANIFEST.json",
        "design/RTL_TRACEABILITY.md",
        PRECHECK.relative_to(ROOT).as_posix(),
        DECISION.relative_to(ROOT).as_posix(),
    ]
    for name in ("dashboard_fields", "implementation_frontier"):
        public[name]["latest_decision"] = FINAL_DECISION
        public[name]["routing_status"] = FINAL_ROUTING_STATUS
        public[name]["required_manager_action"] = FINAL_MANAGER_ACTION
        public[name]["implementation_authorized"] = False
    public["generated_at_utc"] = now
    public.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)

    dump(TARGET, target)
    dump(FAST_LOOP, fast_loop)
    dump(CHIP_SCOPE, chip_scope)
    dump(MANIFEST, manifest)
    dump(PUBLIC, public)


def require_active_projection(
    projection: dict[str, Any],
    *,
    label: str,
    candidate_id: str,
    candidate_hash: str,
    decision_sha: str,
) -> None:
    require(projection.get("projection_scope") == "active_rtl_stage_projection", f"{label}: scope differs")
    require(projection.get("contract_id") == CONTRACT, f"{label}: contract differs")
    require(projection.get("candidate_id") == candidate_id, f"{label}: candidate differs")
    require(projection.get("candidate_rtl_hash") == candidate_hash, f"{label}: candidate hash differs")
    require(projection.get("implementation_authorized") is False, f"{label}: consumed implementation authority is active")
    require(projection.get("operator_approval_consumed") is True, f"{label}: operator approval differs")
    require(projection.get("status") == FINAL_STATUS, f"{label}: status differs")
    require(projection.get("required_manager_action") == FINAL_MANAGER_ACTION, f"{label}: Manager action differs")
    require(projection.get("rtl_review_stage_closing") is True, f"{label}: RTL review is not stage-closing")
    require(projection.get("stage_closing") is False, f"{label}: live project stage was closed")
    review = projection.get("rtl_review", {})
    require(isinstance(review, dict), f"{label}: RTL review binding missing")
    require(review.get("path") == DECISION.relative_to(ROOT).as_posix(), f"{label}: review path differs")
    require(review.get("sha256") == decision_sha, f"{label}: review hash differs")


def validate_post_review_projection_consistency() -> None:
    decision = load(DECISION)
    require(decision.get("reviewer_status") == "done", "fresh RTL review is not done")
    require(decision.get("stage_closing") is True, "fresh RTL review is not stage-closing")
    require(decision.get("checklist") == CHECKLIST, "fresh RTL review checklist is incomplete")
    require(
        decision.get("integrity", {}).get("canonical_sha256") == canonical_sha256(decision),
        "fresh RTL review canonical hash differs",
    )
    verify_record(decision.get("reviewed_precheck", {}), "fresh RTL review reviewed_precheck")
    candidate_id = str(decision.get("candidate_id") or "")
    candidate_hash = str(decision.get("candidate_rtl_hash") or "")
    decision_sha = sha256_file(DECISION)

    target = load(TARGET)
    fast_loop = load(FAST_LOOP)
    chip_scope = load(CHIP_SCOPE)
    manifest = load(MANIFEST)
    public = load(PUBLIC)
    pipeline = load(PIPELINE)

    active = (
        ("TARGET.current_architecture_contract", nested(target, "current_architecture_contract")),
        ("TARGET.fast_loop_contract.active_repair_authorization", nested(target, "fast_loop_contract.active_repair_authorization")),
        ("TARGET.fast_loop_contract.architecture_proposal_authorization", nested(target, "fast_loop_contract.architecture_proposal_authorization")),
        ("TARGET.fast_loop_contract.selected_replacement_contract", nested(target, "fast_loop_contract.selected_replacement_contract")),
        ("FAST_LOOP_POLICY.active_repair_authorization", nested(fast_loop, "active_repair_authorization")),
        ("FAST_LOOP_POLICY.architecture_proposal_authorization", nested(fast_loop, "architecture_proposal_authorization")),
        ("FAST_LOOP_POLICY.selected_replacement_contract", nested(fast_loop, "selected_replacement_contract")),
        ("CHIP_SCOPE.authority_override.operator_implementation_approval", nested(chip_scope, "authority_override.operator_implementation_approval")),
        ("CHIP_SCOPE.numerical_behavior.active_down_projection_residual_fusion_contract", nested(chip_scope, "numerical_behavior.active_down_projection_residual_fusion_contract")),
        ("RTL_MANIFEST.proposed_replacement_contract", nested(manifest, "proposed_replacement_contract")),
        ("PUBLIC_STATUS.architecture_proposal_gate", nested(public, "architecture_proposal_gate")),
        ("PUBLIC_STATUS.selected_replacement_contract", nested(public, "selected_replacement_contract")),
        ("PUBLIC_STATUS.latest_rtl_candidate", nested(public, "latest_rtl_candidate")),
        ("PUBLIC_STATUS.dashboard_fields.latest_rtl_candidate", nested(public, "dashboard_fields.latest_rtl_candidate")),
        ("PUBLIC_STATUS.implementation_frontier.latest_rtl_candidate", nested(public, "implementation_frontier.latest_rtl_candidate")),
    )
    for label, projection in active:
        require_active_projection(
            projection,
            label=label,
            candidate_id=candidate_id,
            candidate_hash=candidate_hash,
            decision_sha=decision_sha,
        )

    require(target["fast_loop_contract"].get("manager_recommendation") == FINAL_MANAGER_ACTION, "TARGET Manager recommendation differs")
    require(target.get("current_stage") == "rtl", "TARGET current stage is not rtl")
    require(target["fast_loop_contract"].get("steady_state_stage") == "rtl", "TARGET steady-state stage is not rtl")
    require(fast_loop.get("manager_recommendation") == FINAL_MANAGER_ACTION, "FAST_LOOP_POLICY Manager recommendation differs")
    require(fast_loop.get("steady_state_stage") == "rtl", "FAST_LOOP_POLICY steady-state stage is not rtl")
    authority = chip_scope["authority_override"]
    require(authority.get("implementation_authorized") is False, "CHIP_SCOPE consumed implementation authority is active")
    require(authority.get("operator_approval_consumed") is True, "CHIP_SCOPE approval is not recorded as consumed")
    require(authority.get("required_next_action") == FINAL_MANAGER_ACTION, "CHIP_SCOPE next action differs")
    require(authority.get("status") == FINAL_STATUS, "CHIP_SCOPE authority status differs")
    require(chip_scope["implementation_frontier"].get("latest_decision") == FINAL_DECISION, "CHIP_SCOPE latest decision differs")
    require(chip_scope["stage"].get("current_stage") == "rtl", "CHIP_SCOPE stage is not rtl")
    require(chip_scope["stage"].get("current_stage_status") == FINAL_STATUS, "CHIP_SCOPE stage status differs")
    require(DECISION.relative_to(ROOT).as_posix() in chip_scope["stage"].get("current_stage_evidence", []), "CHIP_SCOPE omits RTL review evidence")

    require(manifest.get("implementation_authorized") is False, "RTL_MANIFEST consumed implementation authority is active")
    require(manifest.get("candidate_status") == FINAL_STATUS, "RTL_MANIFEST candidate status differs")
    require(manifest.get("candidate_id") == candidate_id, "RTL_MANIFEST candidate differs")
    require(manifest.get("candidate_rtl_hash") == candidate_hash, "RTL_MANIFEST candidate hash differs")
    require(manifest.get("candidate_review_binding", {}).get("reviewer_status") == "done", "RTL_MANIFEST review is not done")
    require(manifest.get("candidate_review_binding", {}).get("stage_closing") is True, "RTL_MANIFEST review is not stage-closing")
    require(manifest.get("independent_reviewer_acceptance", {}).get("status") == FINAL_STATUS, "RTL_MANIFEST reviewer status differs")

    for label, projection in (
        ("PUBLIC_STATUS.architecture_certification_mission", public["architecture_certification_mission"]),
        ("PUBLIC_STATUS.dashboard_fields.candidate_mechanism", nested(public, "dashboard_fields.candidate_mechanism")),
        ("PUBLIC_STATUS.implementation_frontier.candidate_mechanism", nested(public, "implementation_frontier.candidate_mechanism")),
        ("PUBLIC_STATUS.latest_environment_stage", public["latest_environment_stage"]),
        ("PUBLIC_STATUS.dashboard_fields.latest_environment_stage", nested(public, "dashboard_fields.latest_environment_stage")),
        ("PUBLIC_STATUS.implementation_frontier.latest_environment_stage", nested(public, "implementation_frontier.latest_environment_stage")),
    ):
        require(str(projection.get("projection_scope", "")).startswith("historical_"), f"{label}: not historical-scoped")
        require(bool(projection.get("historical_reason")), f"{label}: historical reason missing")

    def reject_unscoped_stale(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            identifies_contract = (
                node.get("contract_id") == CONTRACT
                or node.get("contract_binding") == CONTRACT
                or node.get("candidate_id") == candidate_id
            )
            stale = (
                node.get("status") in PENDING_VALUES
                or node.get("required_manager_action") == "hold_rtl_for_independent_checklist_review"
            )
            if identifies_contract and stale:
                require(
                    str(node.get("projection_scope", "")).startswith("historical_"),
                    f"{trail}: stale authorization/review projection is still active",
                )
                require(bool(node.get("historical_reason")), f"{trail}: historical reason missing")
            consumed_but_active = (
                identifies_contract
                and node.get("operator_approval_consumed") is True
                and node.get("implementation_authorized") is True
            )
            if consumed_but_active:
                require(
                    str(node.get("projection_scope", "")).startswith("historical_"),
                    f"{trail}: consumed implementation authority remains active",
                )
                require(bool(node.get("historical_reason")), f"{trail}: historical reason missing")
            for key, child in node.items():
                reject_unscoped_stale(child, f"{trail}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                reject_unscoped_stale(child, f"{trail}[{index}]")

    for label, value in (
        ("TARGET", target),
        ("FAST_LOOP_POLICY", fast_loop),
        ("CHIP_SCOPE", chip_scope),
        ("RTL_MANIFEST", manifest),
        ("PUBLIC_STATUS", public),
    ):
        reject_unscoped_stale(value, label)

    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(public.get("stage", {}).get("current_stage") == "rtl", "PUBLIC_STATUS stage is not rtl")
    require(public.get("stage", {}).get("current_stage_status") == FINAL_STATUS, "PUBLIC_STATUS stage status differs")
    require(public.get("routing_status") == FINAL_ROUTING_STATUS, "PUBLIC_STATUS routing differs")
    require(public.get("routing_authorized") == FINAL_ROUTING_AUTHORIZATION, "PUBLIC_STATUS routing authorization differs")
    require(public.get("latest_decision") == FINAL_DECISION, "PUBLIC_STATUS latest decision differs")
    for name in ("dashboard_fields", "implementation_frontier"):
        require(public[name].get("current_stage") == "rtl", f"PUBLIC_STATUS {name} stage differs")
        require(public[name].get("routing_status") == FINAL_ROUTING_STATUS, f"PUBLIC_STATUS {name} routing differs")
        require(public[name].get("required_manager_action") == FINAL_MANAGER_ACTION, f"PUBLIC_STATUS {name} Manager action differs")
        require(public[name].get("implementation_authorized") is False, f"PUBLIC_STATUS {name} implementation authority is active")
    require(
        public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public),
        "PUBLIC_STATUS canonical hash differs",
    )


def build_final_precheck(
    reviewed_packet: dict[str, Any],
    reviewed_snapshot: Path,
    decision: dict[str, Any],
    now: str,
) -> dict[str, Any]:
    final = copy.deepcopy(reviewed_packet)
    reviewed_controls = final.pop("control_source_hashes", {})
    final["reviewed_control_source_hashes"] = {
        "hashes": reviewed_controls,
        "scope": "immutable_pre_review_input",
    }
    final["current_check_source_hashes"] = {
        BINDER.relative_to(ROOT).as_posix(): sha256_file(BINDER),
        Path(__file__).resolve().relative_to(ROOT).as_posix(): sha256_file(Path(__file__).resolve()),
        REVIEW_TOOL.relative_to(ROOT).as_posix(): sha256_file(REVIEW_TOOL),
    }
    final["generated_at_utc"] = now
    final["status"] = "pass_rtl_stage_closing_review_accepted_waiting_manager_advance"
    final["stage_closing"] = False
    final["rtl_review_stage_closing"] = True
    final["checks"]["fresh_independent_l2_rtl_stage_closing_review_pass"] = True
    final["checks"]["active_authorization_review_projections_consistent"] = True
    final["projection_consistency"] = {
        "active_status": FINAL_STATUS,
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "manager_owned_stage": "rtl",
        "review_pending_values_rejected_unless_historical": sorted(PENDING_VALUES),
        "rtl_review_stage_closing": True,
    }
    final["review"] = {
        "reviewer_status": "done",
        "stage_closing": True,
        "checklist": copy.deepcopy(CHECKLIST),
        "distinct_reviewer_execution": decision.get("freshness", {}).get("distinct_reviewer_execution"),
        "decision": artifact(DECISION, scope="sealed_whole_file", recursive=True),
        "reviewed_precheck": artifact(reviewed_snapshot, scope="sealed_whole_file"),
    }
    evidence_paths = {
        "architecture_binder": (BINDER, False),
        "chip_scope": (CHIP_SCOPE, True),
        "fast_loop_policy": (FAST_LOOP, True),
        "finalization_tool": (Path(__file__).resolve(), False),
        "manifest": (MANIFEST, True),
        "pipeline_state": (PIPELINE, True),
        "public_status": (PUBLIC, True),
        "rtl_review_checker": (REVIEW_TOOL, False),
        "target": (TARGET, True),
        "traceability": (TRACEABILITY, False),
    }
    for name, (path, recursive) in evidence_paths.items():
        final["evidence"][name] = artifact(path, scope="current_live_binding", recursive=recursive)
    final.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    final["integrity"]["canonical_sha256"] = canonical_sha256(final)
    return final


def validate_final_precheck() -> ValidationSummary:
    packet = load(PRECHECK)
    require(packet.get("status") == "pass_rtl_stage_closing_review_accepted_waiting_manager_advance", "final PRECHECK status differs")
    require(packet.get("stage_closing") is False, "final PRECHECK closes the Manager-owned stage")
    require(packet.get("rtl_review_stage_closing") is True, "final PRECHECK omits stage-closing RTL review")
    require(packet.get("checklist") == CHECKLIST, "final PRECHECK checklist differs")
    require(not any(packet.get("prohibited_runs", {}).values()), "final PRECHECK records a prohibited downstream run")
    require(
        packet.get("reviewed_control_source_hashes", {}).get("scope") == "immutable_pre_review_input",
        "reviewed controls are not immutable-pre-review scoped",
    )
    current_checks = packet.get("current_check_source_hashes", {})
    require(isinstance(current_checks, dict) and current_checks, "current checker hashes missing")
    for relative, expected in current_checks.items():
        path = ROOT / relative
        require(path.is_file(), f"current checker missing: {relative}")
        require(sha256_file(path) == expected, f"current checker hash differs: {relative}")
    require(
        packet.get("checks", {}).get("active_authorization_review_projections_consistent") is True,
        "final PRECHECK projection check is not true",
    )
    validate_post_review_projection_consistency()
    return validate_artifact_tree(packet, label=PRECHECK.relative_to(ROOT).as_posix())


def finalize() -> ValidationSummary:
    now = utc_now()
    decision = load(DECISION)
    require(decision.get("reviewer_status") == "done", "fresh RTL review is not done")
    require(decision.get("stage_closing") is True, "fresh RTL review is not stage-closing")
    require(decision.get("checklist") == CHECKLIST, "fresh RTL review checklist is incomplete")
    require(
        decision.get("integrity", {}).get("canonical_sha256") == canonical_sha256(decision),
        "fresh RTL review canonical hash differs",
    )
    reviewed_snapshot, reviewed_packet = resolve_reviewed_precheck(decision, now)
    decision = load(DECISION)
    require(
        decision.get("reviewed_precheck", {}).get("sha256") == sha256_file(reviewed_snapshot),
        "fresh RTL review does not bind the immutable reviewed PRECHECK",
    )

    reconcile_live_projections(reviewed_packet, now)
    validate_post_review_projection_consistency()
    final = build_final_precheck(reviewed_packet, reviewed_snapshot, decision, now)
    dump(PRECHECK, final)
    return validate_final_precheck()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate finalized state without modifying files")
    args = parser.parse_args()
    if args.check:
        summary = validate_final_precheck()
        prefix = "ACE2_DPRF_RTL_EVIDENCE_FINAL_CHECK_PASS"
    else:
        summary = finalize()
        prefix = "ACE2_DPRF_RTL_EVIDENCE_FINALIZED"
    packet = load(PRECHECK)
    print(
        f"{prefix} candidate={packet['candidate_id']} current_records={summary.current_records} "
        f"historical_records={summary.historical_records}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
