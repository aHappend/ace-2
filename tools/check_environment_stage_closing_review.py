#!/usr/bin/env python3
"""Validate the frozen successor's hash-only environment stage-closing review."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_token_group_dynamic_scale32_v1"
PROPOSAL_SHA256 = "b55ab977574dc2bdeb760e8859ec2fa49f9f835846e3da24bd0f887d84a74f41"
MANAGER_FREEZE_SHA256 = "175d1dcc7016df0c94fb6ec0d086d78fe6b0e4b80f26d1ef42f746825b53c2d8"
ARCH_REVIEW_SHA256 = "386a2951ea1f39e3ef51978ced42be72dea8e10c53be188286abbe1507124297"
PREFIX = ["layer_0.input_rmsnorm", "layer_0.q_proj", "layer_0.k_proj", "layer_0.v_proj"]
CHECKLIST = {"environment.eda-capabilities": True, "environment.tool-ip-selection": True}
ENVIRONMENT_REBIND_GATE = "environment_compatibility_rebind_and_independent_review"
MANAGER_RTL_GATE = "manager_environment_to_rtl_decision"
MANAGER_RTL_TASK = "manager_environment_to_rtl_decision_shared_token_group_dynamic_scale32_v1"
AUDIT = ROOT / "research/ENVIRONMENT_AUDIT.json"
VERDICT = ROOT / "research/ENVIRONMENT_REVIEWER_VERDICT.json"
CERT = ROOT / "research/ENVIRONMENT_L2_CERTIFICATION.md"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
LIVE = ROOT / ".argus/live-view.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
REVIEW_DIR = ROOT / f"evidence/review/environment_stage_closing_{CONTRACT}"
REVIEW_AUDIT = REVIEW_DIR / "audit.json"
DECISION = REVIEW_DIR / "decision.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_record(record: dict[str, Any], label: str) -> None:
    path = ROOT / str(record.get("path", ""))
    require(path.is_file(), f"{label} missing")
    require(path.stat().st_size == int(record.get("bytes", -1)), f"{label} byte mismatch")
    require(sha256(path) == record.get("sha256"), f"{label} hash mismatch")


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    return hashlib.sha256((json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def all_zero(value: dict[str, Any]) -> bool:
    counts = value.get("prohibited_execution_counts", {})
    return bool(counts) and all(item == 0 for item in counts.values())


def manager_decision_queue() -> list[dict[str, Any]]:
    return [{
        "id": MANAGER_RTL_TASK,
        "stage": "environment",
        "status": "pending_manager_decision",
        "scope": "manager_environment_to_rtl_decision_only",
        "implementation_authorized": False,
        "candidate_or_model_execution_authorized": False,
        "prohibited_work": [
            "rtl_implementation",
            "rtl_simulation_or_formal_for_successor",
            "baseline_or_candidate_quality_execution",
            "shell_regression",
            "canonical_ppa",
            "prototype_benchmark_or_signoff",
        ],
    }]


def iter_dicts(value: Any, path: str = "root") -> Any:
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from iter_dicts(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_dicts(child, f"{path}[{index}]")


def validate_manager_decision_queue(queue: Any, label: str) -> None:
    require(queue == manager_decision_queue(), f"{label} Manager decision queue stale")


def validate_no_stale_successor_state(value: Any, label: str) -> None:
    for path, node in iter_dicts(value, label):
        successor_identity = any(node.get(key) == CONTRACT for key in ("contract_id", "id", "proposal_id"))
        if not successor_identity:
            continue
        require(node.get("implementation_authorized") is not True, f"{path} authorizes implementation")
        require(node.get("implementation_completed") is not True, f"{path} claims successor implementation")
        require(node.get("implementation_authority_consumed") is not True,
                f"{path} claims successor implementation authority consumed")
        require(not node.get("candidate_rtl_hash"), f"{path} publishes stale successor RTL hash")
        require(node.get("required_next_gate") != ENVIRONMENT_REBIND_GATE, f"{path} retains completed environment gate")
        if "next_action_queue" in node:
            validate_manager_decision_queue(node.get("next_action_queue"), path)


def validate_current_public_projection(container: dict[str, Any], label: str) -> None:
    require(container.get("current_stage") == "environment", f"{label} stage stale")
    require(container.get("implementation_authorized") is False, f"{label} authorizes implementation")
    require(container.get("implementation_completed") is False, f"{label} claims implementation complete")
    require(container.get("implementation_authority_consumed") is False,
            f"{label} claims implementation authority consumed")
    require(container.get("candidate_rtl_hash") is None, f"{label} publishes candidate RTL hash")
    require(container.get("input_rmsnorm_layer_execution_coverage") == "none_no_successor_rtl",
            f"{label} publishes predecessor RTL execution coverage")
    validate_manager_decision_queue(container.get("next_action_queue"), label)
    selected = container.get("selected_replacement_contract", {})
    require(selected.get("contract_id") == CONTRACT, f"{label} selected contract stale")
    require(selected.get("required_next_gate") == MANAGER_RTL_GATE, f"{label} selected gate stale")
    rtl = container.get("latest_rtl_candidate", {})
    require(rtl.get("contract_id") == CONTRACT, f"{label} RTL projection contract stale")
    require(rtl.get("status") == "not_started_locked_pending_manager_environment_to_rtl_decision",
            f"{label} RTL projection status stale")
    require(rtl.get("candidate_rtl_hash") is None, f"{label} RTL projection hash stale")
    performance = container.get("current_architecture_performance_model", {})
    require(performance.get("contract_id") == CONTRACT, f"{label} performance contract stale")
    require(performance.get("mechanism") == "token_and_tensor_group_dynamic_power_of_two_Scale32_activation_scaling",
            f"{label} performance mechanism stale")
    require(performance.get("planned_sram_peak_bytes") == 379968, f"{label} SRAM projection stale")
    require(performance.get("remaining_sram_margin_bytes") == 144320, f"{label} SRAM margin stale")
    require(performance.get("dynamic_group_finalize_cycles_planning_hook") == 1557,
            f"{label} dynamic finalize projection stale")
    require(not any("carry" in key.lower() for key in performance), f"{label} retains QECR performance fields")
    policy = container.get("operator_policy", {})
    for name in ("active_architecture_contract", "active_successor_contract", "selected_replacement_contract"):
        current = policy.get(name, {})
        require(current.get("contract_id") == CONTRACT, f"{label}.operator_policy.{name} stale")
        require(current.get("required_next_gate") == MANAGER_RTL_GATE,
                f"{label}.operator_policy.{name} gate stale")


def validate_public_artifact_ledger(public: dict[str, Any]) -> None:
    records = public.get("artifact_hashes", [])
    require(isinstance(records, list) and records, "PUBLIC_STATUS artifact ledger missing")
    paths = [record.get("path") for record in records if isinstance(record, dict)]
    require(len(paths) == len(records), "PUBLIC_STATUS artifact ledger malformed")
    require(len(paths) == len(set(paths)), "PUBLIC_STATUS artifact ledger contains duplicate paths")
    require(CHECKPOINT.relative_to(ROOT).as_posix() in paths, "PUBLIC_STATUS checkpoint binding missing")
    require(LIVE.relative_to(ROOT).as_posix() in paths, "PUBLIC_STATUS live-view binding missing")
    for index, record in enumerate(records):
        verify_record(record, f"PUBLIC_STATUS.artifact_hashes[{index}]")


def check_existing() -> None:
    audit = load(AUDIT)
    review_audit = load(REVIEW_AUDIT)
    decision = load(DECISION)
    verdict = load(VERDICT)
    pipeline = load(PIPELINE)
    public = load(PUBLIC)

    require(audit.get("contract_binding", {}).get("active_contract_id") == CONTRACT, "audit contract mismatch")
    require(audit.get("stage_gate_items") == CHECKLIST, "audit checklist mismatch")
    require(audit.get("independent_review", {}).get("required_remediation") == [], "audit remediation is not empty")
    require(audit.get("execution_accounting", {}).get("total_prohibited_executions") == 0, "audit execution total changed")
    require(all_zero(audit.get("execution_accounting", {})), "audit execution count changed")
    require(canonical_sha256(audit) == audit.get("integrity", {}).get("canonical_sha256"), "audit canonical hash mismatch")

    require(review_audit.get("contract_id") == CONTRACT, "review audit contract mismatch")
    require(review_audit.get("environment_checklist") == CHECKLIST, "review audit checklist mismatch")
    require(review_audit.get("required_remediation") == [], "review audit remediation is not empty")
    require(review_audit.get("implementation_authorized") is False, "review audit authorizes implementation")
    require(review_audit.get("execution_accounting", {}).get("total_prohibited_executions") == 0,
            "review audit execution total changed")
    require(all_zero(review_audit.get("execution_accounting", {})), "review audit execution count changed")
    require(canonical_sha256(review_audit) == review_audit.get("integrity", {}).get("canonical_sha256"),
            "review audit canonical hash mismatch")
    for record in review_audit.get("bindings", []):
        verify_record(record, "review_audit_binding")

    require(decision.get("contract_id") == CONTRACT, "decision contract mismatch")
    require(decision.get("status") == "accepted" and decision.get("verdict") == "done", "decision not accepted")
    require(decision.get("decision", {}).get("status") == "done", "decision reviewer status incomplete")
    require(decision.get("required_remediation") == [], "decision remediation is not empty")
    require(decision.get("implementation_authorized") is False, "decision authorizes implementation")
    require(decision.get("preserved_contracts", {}).get("ordered_supported_layer_operator_prefix") == PREFIX,
            "decision prefix changed")
    require(decision.get("preserved_contracts", {}).get("first_unsupported_layer_operator") == "layer_0.rope_q",
            "decision frontier changed")
    require(decision.get("preserved_contracts", {}).get("area_cap_non_sram_mm2") == 2.0, "area cap changed")
    require(decision.get("preserved_contracts", {}).get("frequency_floor_mhz") == 100.0, "frequency floor changed")
    require(decision.get("preserved_contracts", {}).get("abstract_streaming_memory_boundary_bits") == 128,
            "memory boundary changed")
    require(decision.get("execution_accounting", {}).get("total_prohibited_executions") == 0,
            "decision execution total changed")
    require(all_zero(decision.get("execution_accounting", {})), "decision execution count changed")
    for record in decision.get("bindings", {}).values():
        if isinstance(record, dict) and "path" in record:
            verify_record(record, "decision_binding")
        elif isinstance(record, list):
            for child in record:
                verify_record(child, "decision_binding")
    require(decision.get("bindings", {}).get("proposal", {}).get("sha256") == PROPOSAL_SHA256,
            "proposal hash binding changed")
    require(decision.get("bindings", {}).get("manager_freeze", {}).get("sha256") == MANAGER_FREEZE_SHA256,
            "Manager freeze hash binding changed")
    require(decision.get("bindings", {}).get("architecture_decision", {}).get("sha256") == ARCH_REVIEW_SHA256,
            "architecture decision hash binding changed")

    require(verdict.get("contract_id") == CONTRACT, "verdict contract mismatch")
    require(verdict.get("status") == "accepted" and verdict.get("verdict") == "done", "verdict not accepted")
    require(verdict.get("checklist") == CHECKLIST, "verdict checklist mismatch")
    require(verdict.get("required_remediation") == [], "verdict remediation is not empty")
    require(verdict.get("implementation_authorized") is False, "verdict authorizes implementation")
    require(verdict.get("total_prohibited_executions") == 0 and all_zero(verdict), "verdict execution count changed")
    for name in ("audit", "decision", "current_audit", "manager_freeze", "proposal", "architecture_decision"):
        verify_record(verdict.get(name, {}), f"verdict.{name}")

    require(pipeline.get("current_stage") == "environment", "pipeline stage mismatch")
    successor = pipeline.get("successor", {})
    require(successor.get("contract_id") == CONTRACT, "pipeline successor contract mismatch")
    require(successor.get("implementation_authorized") is False, "pipeline authorizes implementation")
    require(successor.get("required_next_gate") == MANAGER_RTL_GATE, "pipeline successor gate stale")
    validate_manager_decision_queue(successor.get("next_action_queue"), "pipeline.successor")
    require(successor.get("candidate_or_model_execution_count") == 0, "pipeline successor reports execution")
    require(all_zero(successor), "pipeline successor execution count changed")
    validate_no_stale_successor_state(pipeline, "pipeline")
    require(public.get("current_stage") == "environment", "public stage mismatch")
    require(public.get("routing_status") == "manager_stage_advance_pending", "public routing mismatch")
    require(public.get("implementation_authorized") is False, "public authorizes implementation")
    require(public.get("implementation_completed") is False, "public claims successor implementation complete")
    require(public.get("implementation_authority_consumed") is False,
            "public claims successor implementation authority consumed")
    require(public.get("candidate_rtl_hash") is None, "public publishes successor RTL hash")
    require(public.get("rtl_contract_traceability") is False, "public claims successor RTL traceability")
    validate_manager_decision_queue(public.get("next_action_queue"), "public")
    require(public.get("supported_layer_operator_prefix") == PREFIX, "public prefix changed")
    require(public.get("first_unsupported_layer_operator") == "layer_0.rope_q", "public frontier changed")
    rtl = public.get("latest_rtl_candidate", {})
    require(rtl.get("contract_id") == CONTRACT, "public RTL projection contract stale")
    require(rtl.get("status") == "not_started_locked_pending_manager_environment_to_rtl_decision",
            "public RTL projection status stale")
    require(rtl.get("candidate_rtl_hash") is None, "public RTL projection hash stale")
    for name in ("dashboard_fields", "implementation_frontier"):
        validate_current_public_projection(public.get(name, {}), f"public.{name}")
    validate_no_stale_successor_state(public, "public")
    validate_public_artifact_ledger(public)
    require(canonical_sha256(public) == public.get("integrity", {}).get("canonical_sha256"),
            "PUBLIC_STATUS canonical hash mismatch")
    require("Required remediation: `empty`" in CERT.read_text(encoding="utf-8"), "L2 certification remediation is not empty")
    require("Prohibited execution count: `0`" in CERT.read_text(encoding="utf-8"), "L2 certification execution count changed")

    print(
        "ACE2_TOKEN_GROUP_DYNAMIC_SCALE32_ENVIRONMENT_REVIEW_PASS "
        f"contract={CONTRACT} proposal_sha256={PROPOSAL_SHA256} "
        f"manager_freeze_sha256={MANAGER_FREEZE_SHA256} architecture_decision_sha256={ARCH_REVIEW_SHA256} "
        "checklist=2/2 remediation=empty implementation_authorized=false prohibited_execution_count=0"
    )


if __name__ == "__main__":
    check_existing()
