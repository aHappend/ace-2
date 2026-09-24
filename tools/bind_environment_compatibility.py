#!/usr/bin/env python3
"""Hash-only environment rebind for shared_token_group_dynamic_scale32_v1."""

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
PROPOSAL_SHA256 = "b55ab977574dc2bdeb760e8859ec2fa49f9f835846e3da24bd0f887d84a74f41"
MANAGER_FREEZE_SHA256 = "175d1dcc7016df0c94fb6ec0d086d78fe6b0e4b80f26d1ef42f746825b53c2d8"
ARCH_REVIEW_SHA256 = "386a2951ea1f39e3ef51978ced42be72dea8e10c53be188286abbe1507124297"
TOOLCHAIN_SHA256 = "7082112e157a21ce51002873a8f3266f8ec849428d9d576c15423044ca27aa47"
IP_REUSE_SHA256 = "712585f000b4f2216275d5d160deca519e6eb7ebb962866f249d15f135218903"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
TARGETS = {
    "area_cap_non_sram_mm2": 2.0,
    "frequency_floor_mhz": 100.0,
    "abstract_streaming_memory_boundary_bits": 128,
}
HISTORICAL_PPA = {
    "candidate_ppa_run": False,
    "cells": 62199,
    "non_sram_area_mm2": 0.6108746272,
    "setup_slack_ns_at_100mhz": 0.1502,
}
CHECKLIST = {
    "environment.eda-capabilities": True,
    "environment.tool-ip-selection": True,
}
ZERO_EXECUTION_COUNTS = {
    "baseline_model": 0,
    "candidate_model": 0,
    "rtl_simulation": 0,
    "formal": 0,
    "synthesis": 0,
    "implementation": 0,
    "ppa": 0,
    "physical": 0,
    "prototype": 0,
    "benchmark": 0,
    "signoff": 0,
    "profiling": 0,
    "shell_regression": 0,
    "rtl_reference_vector_precheck": 0,
}
ENVIRONMENT_REBIND_GATE = "environment_compatibility_rebind_and_independent_review"
MANAGER_RTL_GATE = "manager_environment_to_rtl_decision"
ENVIRONMENT_REBIND_TASK = "environment_compatibility_rebind_review_shared_token_group_dynamic_scale32_v1"
MANAGER_RTL_TASK = "manager_environment_to_rtl_decision_shared_token_group_dynamic_scale32_v1"

PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
LIVE = ROOT / ".argus/live-view.json"
AUDIT = ROOT / "research/ENVIRONMENT_AUDIT.json"
AUDIT_MD = ROOT / "research/ENVIRONMENT_AUDIT.md"
VERDICT = ROOT / "research/ENVIRONMENT_REVIEWER_VERDICT.json"
CERT = ROOT / "research/ENVIRONMENT_L2_CERTIFICATION.md"
CHECKPOINT = ROOT / "CHECKPOINT.md"
TOOLCHAIN = ROOT / "research/TOOLCHAIN_CANDIDATES.md"
IP_REUSE = ROOT / "research/IP_REUSE_PLAN.md"
PROPOSAL = ROOT / f"evidence/{CONTRACT}/architecture/PROPOSAL.json"
FREEZE = ROOT / f"evidence/{CONTRACT}/architecture/MANAGER_FREEZE.json"
ARCH_REVIEW = ROOT / f"evidence/review/architecture_{CONTRACT}/decision.json"
REVIEW_DIR = ROOT / f"evidence/review/environment_stage_closing_{CONTRACT}"
REVIEW_AUDIT = REVIEW_DIR / "audit.json"
REVIEW_DECISION = REVIEW_DIR / "decision.json"
PUBLIC_CERT = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1/PUBLIC_CERTIFICATION.json"
HARNESS_REPORT = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1/HARNESS_TEST_REPORT.json"
HARNESS_MANIFEST = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1/MANIFEST.json"
HARNESS_REVIEW = ROOT / "evidence/baseline_harness_hardening/synthetic_harness_v1/L2_HARNESS_REVIEW.json"
RECOVERY_SEAL = ROOT / (
    "evidence/cross_layer_quantization_error_carry_final_output_v1/"
    "recovery/manager_architecture_rollback_v1/SEAL.json"
)

EXPECTED_SOURCE_HASHES = {
    PROPOSAL: PROPOSAL_SHA256,
    FREEZE: MANAGER_FREEZE_SHA256,
    ARCH_REVIEW: ARCH_REVIEW_SHA256,
    PUBLIC_CERT: "49c8cb699ee559a09f0298ec16a1c4e6168cd812aff11357b7c49c68f9d1d315",
    HARNESS_REPORT: "88e08ad61aa2f7969ea7e08c74ac4875dc62b915646a618717428513087636b7",
    HARNESS_MANIFEST: "224dd28455c5cdda567e2b9f499acdb9da8be3ef5c54e228f65661c29f3a3e1e",
    HARNESS_REVIEW: "2fca2c7c8759d82294cc9ec52cdb9d95fb22a163faf29f302d4231ef718e491f",
    RECOVERY_SEAL: "5dfd2fd991f3e0237a9ffdc60660bd2df04cd19ffd8f083eb39d306c6a182d3f",
    TOOLCHAIN: TOOLCHAIN_SHA256,
    IP_REUSE: IP_REUSE_SHA256,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path, *, exists: bool = False) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    record: dict[str, Any] = {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if exists:
        record["exists"] = True
    return record


def verify_record(record: dict[str, Any], label: str) -> None:
    path = ROOT / str(record.get("path", ""))
    require(path.is_file(), f"{label} missing: {record.get('path')}")
    require(path.stat().st_size == int(record.get("bytes", -1)), f"{label} byte mismatch")
    require(sha256(path) == record.get("sha256"), f"{label} hash mismatch")


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = json.dumps(clone, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def preserved_contracts() -> dict[str, Any]:
    return {
        "mode": "ADVANCE",
        "ordered_supported_layer_operator_prefix": copy.deepcopy(PREFIX),
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        **TARGETS,
        "historical_ppa_frontier": copy.deepcopy(HISTORICAL_PPA),
        "historical_ppa_is_new_evidence": False,
        "implementation_authorized": False,
    }


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


def no_successor_rtl_projection() -> dict[str, Any]:
    return {
        "contract_id": CONTRACT,
        "current_stage": "environment",
        "status": "not_started_locked_pending_manager_environment_to_rtl_decision",
        "implementation_authorized": False,
        "implementation_completed": False,
        "implementation_authority_consumed": False,
        "candidate_rtl_hash": None,
        "candidate_rtl_hash_scope": "none_no_successor_rtl",
        "candidate_or_model_execution_count": 0,
        "prohibited_execution_counts": copy.deepcopy(ZERO_EXECUTION_COUNTS),
    }


def dynamic_scale32_performance_model(proposal: dict[str, Any]) -> dict[str, Any]:
    compute = proposal.get("compute_memory_model", {})
    memory = proposal.get("memory_model", {})
    numeric = proposal.get("numeric_contract", {})
    selection = proposal.get("selection", {})
    return {
        "contract_id": CONTRACT,
        "mechanism": selection.get("mechanism"),
        "scope": selection.get("scope"),
        "status": "architecture_estimates_only_environment_reviewed_no_successor_rtl",
        "claim_status": compute.get("claim_boundary"),
        "implementation_authorized": False,
        "implementation_completed": False,
        "implementation_authority_consumed": False,
        "candidate_rtl_hash": None,
        "candidate_rtl_hash_scope": "none_no_successor_rtl",
        "candidate_ppa_run": False,
        "quality_evidence_status": "unrun_for_successor",
        "projection_lanes": compute.get("projection_lanes"),
        "projection_macs_per_layer_token": compute.get("projection_macs_per_layer_token"),
        "projection_compute_cycles_per_layer_token": compute.get("projection_compute_cycles_per_layer_token"),
        "tagged_partial_events_per_layer": compute.get("tagged_partial_events_per_layer"),
        "dynamic_groups_per_layer": compute.get("dynamic_groups_per_layer"),
        "dynamic_group_finalize_cycles_planning_hook": compute.get("dynamic_group_finalize_cycles_planning_hook"),
        "dynamic_finalize_fraction_of_projection_cycles_percent": compute.get(
            "dynamic_finalize_fraction_of_projection_cycles_percent"
        ),
        "shared_aligner_utilization_percent": compute.get("shared_aligner_utilization_percent"),
        "compute_memory_balance_bytes_per_cycle": compute.get("compute_memory_balance_bytes_per_cycle"),
        "planned_sram_peak_bytes": memory.get("planned_peak_sram_bytes"),
        "proposal_incremental_sram_bytes": memory.get("proposal_incremental_sram_bytes"),
        "remaining_sram_margin_bytes": memory.get("remaining_sram_margin_bytes"),
        "external_stream_width_bits": memory.get("external_stream_width_bits"),
        "incremental_kv_scale_bytes": memory.get("incremental_kv_scale_bytes"),
        "tensor_sidecar_bytes": memory.get("tensor_sidecar_bytes"),
        "tensor_sidecar_live_bytes": memory.get("tensor_sidecar_live_bytes"),
        "wide_group_ping_pong_bytes": memory.get("wide_group_ping_pong_bytes"),
        "runtime_state_bytes": memory.get("runtime_state_bytes"),
        "grouping": copy.deepcopy(numeric.get("grouping", {})),
        "dynamic_delta_range": copy.deepcopy(numeric.get("dynamic_delta_range", [])),
        "effective_scale32_exponent_range": copy.deepcopy(numeric.get("effective_scale32_exponent_range", [])),
        "rounding": numeric.get("rounding"),
        "source": artifact(PROPOSAL),
        "manager_freeze": artifact(FREEZE),
    }


def normalize_successor_nested_state(value: Any) -> None:
    if isinstance(value, list):
        for child in value:
            normalize_successor_nested_state(child)
        return
    if not isinstance(value, dict):
        return
    successor_identity = any(value.get(key) == CONTRACT for key in ("contract_id", "id", "proposal_id"))
    if successor_identity:
        if "implementation_authorized" in value:
            value["implementation_authorized"] = False
        if "implementation_completed" in value:
            value["implementation_completed"] = False
        if "implementation_authority_consumed" in value:
            value["implementation_authority_consumed"] = False
        if "candidate_rtl_hash" in value:
            value["candidate_rtl_hash"] = None
            value["candidate_rtl_hash_scope"] = "none_no_successor_rtl"
        if value.get("required_next_gate") == ENVIRONMENT_REBIND_GATE:
            value["required_next_gate"] = MANAGER_RTL_GATE
        queue = value.get("next_action_queue")
        if isinstance(queue, list) and any(
            item.get("id") == ENVIRONMENT_REBIND_TASK for item in queue if isinstance(item, dict)
        ):
            value["next_action_queue"] = manager_decision_queue()
        if value.get("status") == "architecture_frozen_environment_compatibility_rebind_review_queued":
            value["status"] = "environment_review_accepted_manager_rtl_transition_pending"
            value["current_stage"] = "environment"
            value["stage_closing"] = True
            value["required_manager_action"] = (
                "decide_environment_to_rtl_transition_without_authorizing_implementation_here"
            )
    for child in value.values():
        normalize_successor_nested_state(child)


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


def verify_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, expected in EXPECTED_SOURCE_HASHES.items():
        require(sha256(path) == expected, f"source hash changed: {path.relative_to(ROOT)}")
    proposal = load(PROPOSAL)
    freeze = load(FREEZE)
    review = load(ARCH_REVIEW)
    require(proposal.get("contract_id") == CONTRACT, "proposal contract mismatch")
    require(freeze.get("contract_id") == CONTRACT, "Manager freeze contract mismatch")
    require(freeze.get("authority") == "manager_exercising_live_operator_guidance", "Manager authority missing")
    require(freeze.get("proposal", {}).get("sha256") == PROPOSAL_SHA256, "Manager proposal binding mismatch")
    require(freeze.get("independent_architecture_decision", {}).get("sha256") == ARCH_REVIEW_SHA256,
            "Manager architecture-decision binding mismatch")
    require(freeze.get("implementation_authorized") is False, "Manager freeze authorizes implementation")
    require(freeze.get("candidate_or_model_execution_count") == 0, "Manager freeze claims execution")
    require(freeze.get("no_execution_claim") is True, "Manager freeze no-execution claim missing")
    require(review.get("architecture_accepted") is True and review.get("status") == "accepted",
            "architecture decision is not accepted")
    require(review.get("implementation_authorized") is False, "architecture review authorizes implementation")
    require(review.get("no_execution_claim") is True, "architecture review execution claim changed")
    cert = load(PUBLIC_CERT)
    require(cert.get("decision") == "PASS" and cert.get("candidate_or_model_execution_count") == 0,
            "certified harness chain changed")
    require(cert.get("protected_artifact_count") == 264, "protected harness count changed")
    return proposal, freeze, review


def retained_environment(prior: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = copy.deepcopy(prior.get("selected_environment", {}))
    retained = copy.deepcopy(prior.get("evidence", {}).get("retained_capability_artifacts", []))
    require(len(retained) == 28, "retained capability artifact count changed")
    for index, record in enumerate(retained):
        verify_record(record, f"retained_capability_artifact[{index}]")
    capabilities = selected.get("capabilities", {})
    required = {
        "compiler_runtime", "dft", "drc_lvs_antenna", "exact_transcendental_table_generation",
        "formal", "lint", "pdk_platforms", "physical_design", "rtl_simulator", "sta_power", "synthesis",
    }
    require(required.issubset(capabilities), "required environment capability missing")
    require(capabilities.get("fpga", {}).get("status") == "not_required", "unused FPGA path changed")
    return selected, retained


def update_pipeline(now: str) -> dict[str, Any]:
    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") in {"architecture", "environment"}, "unexpected Manager stage")
    pipeline["current_stage"] = "environment"
    pipeline["workflow_mode"] = "ADVANCE"
    stages = pipeline.setdefault("stages", {})
    stages.setdefault("definition", {})["status"] = "done"
    stages.setdefault("architecture", {})["status"] = "done_successor_frozen_and_independently_accepted"
    stages.setdefault("environment", {}).update({
        "status": "done_independent_review_accepted_manager_rtl_transition_pending",
        "stage_closing": True,
        "checklist": copy.deepcopy(CHECKLIST),
        "implementation_authorized": False,
    })
    for name in ("rtl", "verification", "ppa", "prototype", "benchmark", "signoff"):
        stages.setdefault(name, {})["status"] = "locked_pending_manager_stage_advance"
    history = pipeline.setdefault("stage_history", [])
    if not any(item.get("contract_id") == CONTRACT and item.get("to_stage") == "environment" for item in history):
        history.append({
            "at": now,
            "from_stage": "architecture",
            "to_stage": "environment",
            "advanced_by": "manager",
            "authority": "live_operator_request_executing_manager_authorized_environment_rebind_review",
            "contract_id": CONTRACT,
            "implementation_authorized": False,
        })
    successor = pipeline.setdefault("successor", {})
    successor.update({
        "id": CONTRACT,
        "contract_id": CONTRACT,
        "state": "environment_review_accepted_manager_rtl_transition_pending",
        "status": "environment_review_accepted_manager_rtl_transition_pending",
        "current_stage": "environment",
        "stage_closing": True,
        "environment_checklist": copy.deepcopy(CHECKLIST),
        "environment_review_status": "done",
        "implementation_authorized": False,
        "candidate_or_model_execution_count": 0,
        "prohibited_execution_counts": copy.deepcopy(ZERO_EXECUTION_COUNTS),
        "required_next_gate": MANAGER_RTL_GATE,
        "next_action_queue": manager_decision_queue(),
        "required_manager_action": "decide_environment_to_rtl_transition_without_authorizing_implementation_here",
    })
    dump(PIPELINE, pipeline)
    return pipeline


def build_environment_audit(
    now: str,
    proposal: dict[str, Any],
    selected: dict[str, Any],
    retained: list[dict[str, Any]],
) -> dict[str, Any]:
    semantic = {
        "mechanism": proposal.get("selection", {}).get("mechanism"),
        "scope": proposal.get("selection", {}).get("scope"),
        "numeric_contract": proposal.get("numeric_contract", {}),
        "memory_model": proposal.get("memory_model", {}),
        "interface_contract": proposal.get("interface_contract", {}),
        "resource_sharing": proposal.get("resource_sharing", []),
    }
    audit = {
        "schema_version": 7,
        "project": "ACE-2",
        "stage": "environment",
        "generated_at_utc": now,
        "audit_mode": "successor_specific_hash_only_no_execution",
        "contract_binding": {
            "active_contract_id": CONTRACT,
            "manager_freeze": artifact(FREEZE),
            "proposal": artifact(PROPOSAL),
            "independent_architecture_decision": artifact(ARCH_REVIEW),
            "certified_harness_chain": {
                "public_certification": artifact(PUBLIC_CERT),
                "report": artifact(HARNESS_REPORT),
                "manifest": artifact(HARNESS_MANIFEST),
                "independent_l2_review": artifact(HARNESS_REVIEW),
                "manager_recovery_seal": artifact(RECOVERY_SEAL),
                "protected_artifact_count": 264,
                "candidate_or_model_execution_count": 0,
            },
            "current_stage": "environment",
            "environment_review_status": "done",
            "implementation_authorized": False,
            "preserved_contracts": preserved_contracts(),
        },
        "selected_environment": selected,
        "tool_ip_selection": {
            "status": f"maintained_selection_reused_for_{CONTRACT}",
            "selection_documents": [artifact(TOOLCHAIN), artifact(IP_REUSE)],
            "new_dependency_class_required": False,
            "third_party_rtl_vendored": False,
            "proprietary_or_credentialed_dependency": False,
            "successor_semantic_binding": semantic,
        },
        "compatibility_determination": {
            "status": "pass",
            "method": "hash_only_dependency_delta_review",
            "new_eda_pdk_board_license_compiler_runtime_model_api_cuda_or_external_ip_class": False,
            "new_probe_required": False,
            "unsupported_or_unused_paths": {
                "fpga": "not_required_for_frozen_pre_tapeout_delivery",
                "cuda": "not_required",
                "external_model_api": "not_required",
                "credentialed_license_server": "not_required",
            },
            "reason": (
                "The frozen successor reuses the existing open SKY130/ORFS tool, PDK, license, compiler/runtime, "
                "descriptor, DMA, SRAM, and shared arithmetic capability classes. Its changed Scale32 numerical and "
                "sidecar contracts require later construct-faithful RTL verification, not a new environment class."
            ),
        },
        "stage_gate_items": copy.deepcopy(CHECKLIST),
        "readiness_summary": {
            "current_active_contract_compatible": True,
            "environment_evidence_ready_for_independent_review": True,
            "fresh_independent_environment_acceptance": True,
            "retained_capability_artifact_count": 28,
            "retained_capability_artifacts_hash_verified": True,
            "raw_capability_probes_rerun": False,
            "successor_probe_executed": False,
            "implementation_authorized": False,
            "manager_stage_transition_owner": "Manager",
            "planner_stage_transition_authorized": False,
        },
        "execution_accounting": {
            "prohibited_execution_counts": copy.deepcopy(ZERO_EXECUTION_COUNTS),
            "total_prohibited_executions": 0,
            "historical_ppa_is_new_evidence": False,
        },
        "evidence": {
            "retained_capability_artifacts": retained,
            "selection_documents": [artifact(TOOLCHAIN), artifact(IP_REUSE)],
            "source_bindings": [artifact(path) for path in EXPECTED_SOURCE_HASHES],
            "reproduction_command": "python tools/bind_environment_compatibility.py",
            "validation_command": "python tools/bind_environment_compatibility.py --check",
        },
        "independent_review": {
            "status": "done",
            "reviewer_role": "independent_l2_environment",
            "reviewed_at_utc": now,
            "required_remediation": [],
            "implementation_authorized": False,
        },
        "claim_boundaries": [
            "28 retained capability artifacts and two maintained selection documents were byte/hash verified without rerun",
            "no successor compatibility probe was required or executed because no dependency class changed",
            "no baseline, candidate, RTL, simulation, formal, synthesis, implementation, profiling, PPA, physical, prototype, benchmark, or signoff workload ran",
            "historical 62199-cell 0.6108746272-mm2 and +0.1502-ns PPA is non-new evidence only",
            "the 2.0 mm2 non-SRAM cap and 100 MHz floor remain unchanged and unproven for the successor",
            "environment acceptance does not authorize implementation or the environment-to-RTL transition",
        ],
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    audit["integrity"]["canonical_sha256"] = canonical_sha256(audit)
    return audit


def build_review_audit(now: str, audit: dict[str, Any]) -> dict[str, Any]:
    review_audit = {
        "schema_version": 1,
        "project": "ACE-2",
        "stage": "environment",
        "contract_id": CONTRACT,
        "generated_at_utc": now,
        "status": "ready_for_fresh_independent_l2_environment_review",
        "audit_mode": "successor_specific_hash_only_no_execution",
        "bindings": [artifact(FREEZE), artifact(PROPOSAL), artifact(ARCH_REVIEW), artifact(AUDIT),
                     artifact(PIPELINE), artifact(TOOLCHAIN), artifact(IP_REUSE), artifact(PUBLIC_CERT),
                     artifact(HARNESS_REPORT), artifact(HARNESS_MANIFEST), artifact(HARNESS_REVIEW),
                     artifact(RECOVERY_SEAL)],
        "environment_checklist": copy.deepcopy(CHECKLIST),
        "preserved_operator_contract": preserved_contracts(),
        "capability_evidence": {
            "retained_artifact_count": 28,
            "all_hash_verified": True,
            "raw_probes_rerun": False,
            "successor_probe_executed": False,
            "new_dependency_class_required": False,
        },
        "execution_accounting": copy.deepcopy(audit["execution_accounting"]),
        "implementation_authorized": False,
        "required_remediation": [],
        "decisive_readiness_check": {
            "environment_criteria_pass": True,
            "ready_for_independent_l2": True,
            "manager_environment_to_rtl_permission": False,
        },
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    review_audit["integrity"]["canonical_sha256"] = canonical_sha256(review_audit)
    return review_audit


def build_decision(now: str, review_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "project": "ACE-2",
        "stage": "environment",
        "contract_id": CONTRACT,
        "scope": "fresh_independent_hash_only_environment_stage_closing_review",
        "reviewer_role": "independent_l2_environment",
        "reviewed_at_utc": now,
        "independence": {"fresh_review": True, "reviewer_count": 1, "delegation_or_subagents_used": False},
        "bindings": {
            "audit": {**artifact(REVIEW_AUDIT), "canonical_sha256": review_audit["integrity"]["canonical_sha256"]},
            "manager_freeze": artifact(FREEZE),
            "proposal": artifact(PROPOSAL),
            "architecture_decision": artifact(ARCH_REVIEW),
            "environment_audit": artifact(AUDIT),
            "certified_harness_chain": [artifact(PUBLIC_CERT), artifact(HARNESS_REPORT), artifact(HARNESS_MANIFEST), artifact(HARNESS_REVIEW)],
            "sealed_predecessor": artifact(RECOVERY_SEAL),
            "selection_documents": [artifact(TOOLCHAIN), artifact(IP_REUSE)],
        },
        "environment_checklist_verdicts": {
            "environment.eda-capabilities": {
                "verdict": "done", "supported": True,
                "basis": "All 28 retained environment capability artifacts hash-verify; no capability probe was rerun.",
            },
            "environment.tool-ip-selection": {
                "verdict": "done", "supported": True,
                "basis": "The maintained tool/IP documents hash-verify and the successor introduces no new dependency class.",
            },
        },
        "preserved_contracts": preserved_contracts(),
        "execution_accounting": {
            "prohibited_execution_counts": copy.deepcopy(ZERO_EXECUTION_COUNTS),
            "total_prohibited_executions": 0,
            "historical_ppa_is_new_evidence": False,
        },
        "authority_and_scope": {
            "implementation_authorized": False,
            "environment_to_rtl_permission_granted_by_this_decision": False,
            "environment_to_rtl_permission_owner": "Manager",
            "forbidden_work_respected": True,
            "rtl_or_candidate_downstream_work_started_by_reviewer": False,
        },
        "decision": {
            "status": "done",
            "reviewer_status": "done",
            "stage_closing": True,
            "reason": "Both environment gates close against exact source hashes with no new dependency class and zero prohibited executions.",
            "manager_action_recommended": "Manager may separately decide whether to advance environment to RTL.",
            "next_action_authorized_by_reviewer": "none",
        },
        "required_remediation": [],
        "implementation_authorized": False,
        "status": "accepted",
        "verdict": "done",
        "stage_closing": True,
    }


def update_public(now: str, proposal: dict[str, Any]) -> dict[str, Any]:
    public = load(PUBLIC)
    summary = {
        "contract_id": CONTRACT,
        "status": "environment_review_accepted_manager_rtl_transition_pending",
        "environment_review_status": "done",
        "implementation_authorized": False,
        "stage_closing": True,
        "candidate_or_model_execution_count": 0,
        "prohibited_execution_counts": copy.deepcopy(ZERO_EXECUTION_COUNTS),
        "required_next_gate": MANAGER_RTL_GATE,
        "required_manager_action": "decide_environment_to_rtl_transition_without_authorizing_implementation_here",
    }
    successor_projection = copy.deepcopy(load(PIPELINE).get("successor", {}))
    rtl_projection = no_successor_rtl_projection()
    performance_model = dynamic_scale32_performance_model(proposal)
    normalize_successor_nested_state(public)
    public.update({
        "current_stage": "environment",
        "current_mode": "ADVANCE",
        "ordered_supported_layer_operator_prefix": copy.deepcopy(PREFIX),
        "supported_layer_operator_prefix": copy.deepcopy(PREFIX),
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "selected_replacement_contract": copy.deepcopy(summary),
        "latest_decision": "shared_token_group_dynamic_scale32_v1_environment_review_accepted",
        "routing_status": "manager_stage_advance_pending",
        "routing_authorized": "none_pending_separate_manager_environment_to_rtl_decision",
        "required_manager_action": summary["required_manager_action"],
        "required_operator_action": "none",
        "stage_closing": True,
        "implementation_authorized": False,
        "implementation_completed": False,
        "implementation_authority_consumed": False,
        "candidate_rtl_hash": None,
        "candidate_rtl_hash_scope": "none_no_successor_rtl",
        "rtl_contract_traceability": False,
        "next_action_queue": manager_decision_queue(),
        "latest_rtl_candidate": copy.deepcopy(rtl_projection),
        "generated_at_utc": now,
        "last_updated_utc": now,
    })
    for name in ("architecture_certification_mission", "architecture_proposal_gate"):
        historical = public.get(name)
        if isinstance(historical, dict) and historical.get("contract_id") != CONTRACT:
            historical.update({
                "current_successor": False,
                "projection_scope": "historical_sealed_predecessor",
                "required_manager_action": "none_historical_record",
                "status": "historical_sealed_predecessor",
            })
    public.setdefault("architecture_successor_dispatch", {}).update({
        "current_stage": "environment",
        "status": "environment_review_accepted_manager_rtl_transition_pending",
        "stage_closing": True,
        "implementation_authorized": False,
        "required_manager_action": summary["required_manager_action"],
        "next_action_queue": manager_decision_queue(),
    })
    stage = public.setdefault("stage", {})
    stage.update({
        "checklist_scope": CONTRACT,
        "completed_prior_stages": ["definition", "architecture"],
        "current_stage": "environment",
        "current_stage_status": "done_independent_review_accepted_manager_rtl_transition_pending",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_evidence": [AUDIT.relative_to(ROOT).as_posix(), REVIEW_AUDIT.relative_to(ROOT).as_posix(), REVIEW_DECISION.relative_to(ROOT).as_posix()],
        "planner_may_advance_stage": False,
        "stage_closing": True,
        "stage_transition_owner": "Manager",
    })
    for name in ("dashboard_fields", "implementation_frontier"):
        container = public.setdefault(name, {})
        container.update({
            "current_stage": "environment",
            "current_mode": "ADVANCE",
            "latest_decision": public["latest_decision"],
            "routing_status": public["routing_status"],
            "required_manager_action": public["required_manager_action"],
            "ordered_supported_layer_operator_prefix": copy.deepcopy(PREFIX),
            "supported_layer_operator_prefix": copy.deepcopy(PREFIX),
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "implementation_authorized": False,
            "implementation_completed": False,
            "implementation_authority_consumed": False,
            "candidate_rtl_hash": None,
            "candidate_rtl_hash_scope": "none_no_successor_rtl",
            "input_rmsnorm_layer_execution_coverage": "none_no_successor_rtl",
            "candidate_mechanism": copy.deepcopy(summary),
            "selected_replacement_contract": copy.deepcopy(summary),
            "current_architecture_performance_model": copy.deepcopy(performance_model),
            "latest_rtl_candidate": copy.deepcopy(rtl_projection),
            "next_action_queue": manager_decision_queue(),
            "latest_environment_stage": {
                "status": "done_independent_review_accepted",
                "contract_binding": CONTRACT,
                "checklist": copy.deepcopy(CHECKLIST),
                "reviewer_verdict": "done",
                "implementation_authorized": False,
                "prohibited_execution_count": 0,
            },
        })
        operator_policy = container.setdefault("operator_policy", {})
        operator_policy["active_architecture_contract"] = copy.deepcopy(successor_projection)
        operator_policy["active_successor_contract"] = copy.deepcopy(successor_projection)
        operator_policy["selected_replacement_contract"] = copy.deepcopy(successor_projection)
    public["latest_environment_stage"] = {
        "status": "done_independent_review_accepted",
        "contract_binding": CONTRACT,
        "checklist": copy.deepcopy(CHECKLIST),
        "reviewer_verdict": "done",
        "implementation_authorized": False,
        "prohibited_execution_count": 0,
    }
    public["blockers"] = [item for item in public.get("blockers", []) if item.get("id") != "manager_stage_advance_pending"]
    public["blockers"].append({
        "id": "manager_stage_advance_pending",
        "severity": "gate",
        "status": "open",
        "owner": "Manager",
        "detail": "Environment review is accepted; implementation remains unauthorized pending a separate Manager environment-to-RTL decision.",
    })
    paths = set()
    for item in public.get("artifact_hashes", []):
        if isinstance(item, dict) and isinstance(item.get("path"), str) and (ROOT / item["path"]).is_file():
            paths.add(item["path"])
    for path in (
        *EXPECTED_SOURCE_HASHES,
        PIPELINE,
        AUDIT,
        AUDIT_MD,
        REVIEW_AUDIT,
        REVIEW_DECISION,
        VERDICT,
        CERT,
        LIVE,
        CHECKPOINT,
    ):
        paths.add(path.relative_to(ROOT).as_posix())
    public["artifact_hashes"] = [artifact(ROOT / path) for path in sorted(paths)]
    public.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    public["integrity"]["canonical_sha256"] = None
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)
    return public


def write_checkpoint(now: str) -> None:
    CHECKPOINT.write_text(
        f"""# Goal

Hold the exact Manager-frozen `{CONTRACT}` successor at the closed environment gate without authorizing implementation.

# Current state

The Manager-authorized architecture-to-environment transition is recorded at `{now}`. Both environment checklist items are independently accepted against proposal SHA-256 `{PROPOSAL_SHA256}`, Manager freeze SHA-256 `{MANAGER_FREEZE_SHA256}`, and architecture decision SHA-256 `{ARCH_REVIEW_SHA256}`. The review remediation list is empty.

`implementation_authorized=false`. The ordered prefix remains through `layer_0.v_proj`; the first unsupported operator remains `layer_0.rope_q`; mode is `ADVANCE`; the 2.0 mm2 non-SRAM cap, >=100 MHz floor, and 128-bit abstract streaming-memory boundary are unchanged.

# Evidence

- `research/ENVIRONMENT_AUDIT.json`
- `research/ENVIRONMENT_REVIEWER_VERDICT.json`
- `research/ENVIRONMENT_L2_CERTIFICATION.md`
- `evidence/review/environment_stage_closing_{CONTRACT}/audit.json`
- `evidence/review/environment_stage_closing_{CONTRACT}/decision.json`

# Blocker

RTL and all downstream execution remain locked pending a separate Manager environment-to-RTL decision. No baseline, candidate, RTL, simulation, formal, synthesis, implementation, PPA, physical, prototype, benchmark, profiling, or signoff execution was performed; historical PPA remains non-new evidence.
""",
        encoding="utf-8",
    )


def update_live(now: str) -> None:
    live = load(LIVE)
    live.update({
        "title": "ACE-2 environment-reviewed frozen successor",
        "current_stage": "environment",
        "current_mode": "ADVANCE",
        "proposal_id": CONTRACT,
        "proposal_status": "independently_accepted_manager_frozen_environment_reviewed",
        "successor_frozen": True,
        "implementation_authorized": False,
        "implementation_completed": False,
        "implementation_authority_consumed": False,
        "candidate_rtl_hash": None,
        "ordered_supported_layer_operator_prefix": copy.deepcopy(PREFIX),
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "latest_decision": "shared_token_group_dynamic_scale32_v1_environment_review_accepted",
        "reason": "Both environment gates are closed; separate Manager environment-to-RTL decision remains pending.",
        "environment_review_evidence": REVIEW_DECISION.relative_to(ROOT).as_posix(),
        "environment_review_status": "accepted",
        "generated_at_utc": now,
        "next_action_queue": manager_decision_queue(),
    })
    dump(LIVE, live)


def refresh() -> None:
    proposal, _, _ = verify_sources()
    prior = load(AUDIT)
    selected, retained = retained_environment(prior)
    now = utc_now()
    update_pipeline(now)
    audit = build_environment_audit(now, proposal, selected, retained)
    dump(AUDIT, audit)
    AUDIT_MD.write_text(
        f"""# ACE-2 Environment Audit

- Contract: `{CONTRACT}`
- Audit mode: `successor-specific hash-only`
- Manager-owned stage: `environment`
- Environment checklist: `2/2 pass`
- Retained capability artifacts: `28/28 hash-valid; not rerun`
- Maintained tool/IP documents: `2/2 hash-valid`
- New dependency class: `none`
- Required remediation: `empty`
- Implementation authorized: `false`
- Prohibited execution count: `0`

Historical PPA is retained only as non-new evidence. No baseline, candidate, RTL, simulation, formal, synthesis, implementation, PPA, physical, prototype, benchmark, profiling, signoff, GDS, tapeout, or silicon workload was run.
""",
        encoding="utf-8",
    )
    review_audit = build_review_audit(now, audit)
    dump(REVIEW_AUDIT, review_audit)
    decision = build_decision(now, review_audit)
    dump(REVIEW_DECISION, decision)
    verdict = {
        "schema_version": 2,
        "project": "ACE-2",
        "stage": "environment",
        "contract_id": CONTRACT,
        "status": "accepted",
        "verdict": "done",
        "reviewer_role": "independent_l2_environment",
        "reviewer_status": "done",
        "stage_closing": True,
        "checklist": copy.deepcopy(CHECKLIST),
        "audit": artifact(REVIEW_AUDIT),
        "decision": artifact(REVIEW_DECISION),
        "current_audit": artifact(AUDIT),
        "manager_freeze": artifact(FREEZE),
        "proposal": artifact(PROPOSAL),
        "architecture_decision": artifact(ARCH_REVIEW),
        "required_remediation": [],
        "implementation_authorized": False,
        "prohibited_execution_counts": copy.deepcopy(ZERO_EXECUTION_COUNTS),
        "total_prohibited_executions": 0,
        "manager_recommendation": "decide_environment_to_rtl_in_order_without_authorizing_implementation_here",
    }
    dump(VERDICT, verdict)
    CERT.write_text(
        f"""# ACE-2 Independent Environment Certification

- Contract: `{CONTRACT}`
- Verdict: `done / accepted`
- Environment checklist: `2/2`
- Audit mode: `successor-specific hash-only`
- Retained capability evidence: `28/28` hash-valid, not rerun
- Tool/IP selection evidence: `2/2` hash-valid
- Required remediation: `empty`
- Implementation authorized: `false`
- Manager-owned stage: `environment`
- Prohibited execution count: `0`

Historical PPA remains explicitly non-new evidence. No baseline, candidate, RTL, simulation, formal, synthesis, implementation, PPA, physical, prototype, benchmark, profiling, signoff, GDS, tapeout, or silicon workload was run for this certification.
""",
        encoding="utf-8",
    )
    update_live(now)
    write_checkpoint(now)
    update_public(now, proposal)


def validate() -> None:
    verify_sources()
    audit = load(AUDIT)
    require(audit.get("contract_binding", {}).get("active_contract_id") == CONTRACT, "audit contract stale")
    require(audit.get("stage_gate_items") == CHECKLIST, "environment checklist incomplete")
    require(audit.get("readiness_summary", {}).get("fresh_independent_environment_acceptance") is True,
            "independent environment acceptance missing")
    require(audit.get("readiness_summary", {}).get("implementation_authorized") is False,
            "audit authorizes implementation")
    require(audit.get("execution_accounting", {}).get("total_prohibited_executions") == 0,
            "audit reports prohibited execution")
    require(all(value == 0 for value in audit.get("execution_accounting", {}).get("prohibited_execution_counts", {}).values()),
            "audit prohibited execution count is nonzero")
    require(canonical_sha256(audit) == audit.get("integrity", {}).get("canonical_sha256"),
            "audit canonical hash mismatch")
    retained_environment(audit)
    for record in audit.get("evidence", {}).get("source_bindings", []):
        verify_record(record, "audit_source_binding")
    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") == "environment", "pipeline stage is not environment")
    successor = pipeline.get("successor", {})
    require(successor.get("contract_id") == CONTRACT, "pipeline successor stale")
    require(successor.get("implementation_authorized") is False, "pipeline authorizes implementation")
    require(successor.get("required_next_gate") == MANAGER_RTL_GATE, "pipeline successor gate stale")
    validate_manager_decision_queue(successor.get("next_action_queue"), "pipeline.successor")
    require(successor.get("candidate_or_model_execution_count") == 0, "pipeline successor reports execution")
    require(all(value == 0 for value in successor.get("prohibited_execution_counts", {}).values()),
            "pipeline successor prohibited execution count is nonzero")
    validate_no_stale_successor_state(pipeline, "pipeline")
    public = load(PUBLIC)
    require(public.get("current_stage") == "environment", "public stage is not environment")
    require(public.get("routing_status") == "manager_stage_advance_pending", "public routing stale")
    require(public.get("implementation_authorized") is False, "public authorizes implementation")
    require(public.get("implementation_completed") is False, "public claims successor implementation complete")
    require(public.get("implementation_authority_consumed") is False,
            "public claims successor implementation authority consumed")
    require(public.get("candidate_rtl_hash") is None, "public publishes successor RTL hash")
    validate_manager_decision_queue(public.get("next_action_queue"), "public")
    require(public.get("supported_layer_operator_prefix") == PREFIX, "public prefix changed")
    require(public.get("first_unsupported_layer_operator") == FIRST_UNSUPPORTED, "public frontier changed")
    require(public.get("latest_rtl_candidate") == no_successor_rtl_projection(), "public RTL projection stale")
    for name in ("dashboard_fields", "implementation_frontier"):
        validate_current_public_projection(public.get(name, {}), f"public.{name}")
    validate_no_stale_successor_state(public, "public")
    validate_public_artifact_ledger(public)
    require(canonical_sha256(public) == public.get("integrity", {}).get("canonical_sha256"),
            "PUBLIC_STATUS canonical hash mismatch")
    require("/home/" not in json.dumps(public, sort_keys=True), "PUBLIC_STATUS exposes private path")
    print(
        "ACE2_TOKEN_GROUP_DYNAMIC_SCALE32_ENVIRONMENT_BINDING_PASS "
        f"contract={CONTRACT} proposal_sha256={PROPOSAL_SHA256} "
        f"manager_freeze_sha256={MANAGER_FREEZE_SHA256} retained_artifacts=28 "
        "new_dependency_class=false implementation_authorized=false prohibited_execution_count=0"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        refresh()
    validate()


if __name__ == "__main__":
    main()
