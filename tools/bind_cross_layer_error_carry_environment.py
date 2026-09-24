#!/usr/bin/env python3
"""Advance and bind the error-carry successor to a hash-only environment audit."""

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
CONTRACT = "cross_layer_quantization_error_carry_final_output_v1"
PREDECESSOR = "shared_down_projection_residual_fusion_v1"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
AUDIT = ROOT / "research/ENVIRONMENT_AUDIT.json"
AUDIT_MD = ROOT / "research/ENVIRONMENT_AUDIT.md"
TOOLCHAIN = ROOT / "research/TOOLCHAIN_CANDIDATES.md"
IP_REUSE = ROOT / "research/IP_REUSE_PLAN.md"
MEMORY_MODEL = ROOT / "design/MEMORY_MODEL.json"
PACKET = ROOT / f"evidence/{CONTRACT}/latest/ARCHITECTURE_REVIEW_PACKET.json"
FREEZE = ROOT / f"evidence/{CONTRACT}/architecture_freeze.json"
ARCH_REVIEW = (
    ROOT / f"evidence/review/architecture_stage_closing_{CONTRACT}/decision.json"
)
ENV_REVIEW = ROOT / f"evidence/review/environment_stage_closing_{CONTRACT}/decision.json"
ARCHIVE = ROOT / f"research/archive/environment/historical_before_{CONTRACT}"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
EXPECTED_DOWNSTREAM_RUNS = {
    "benchmark": "unrun",
    "paired": "unrun",
    "physical": "unrun",
    "ppa": "unrun",
    "prototype": "unrun",
    "shell": "unrun",
    "signoff": "unrun",
}
EXPECTED_FREEZE_DOWNSTREAM_RUNS = {
    "benchmark": "unrun",
    "physical": "unrun",
    "ppa": "unrun",
    "prototype": "unrun",
    "rtl": "unrun",
    "shell": "unrun",
    "signoff": "unrun",
    "verification": "unrun",
}
ENVIRONMENT_CHECKLIST = {
    "environment.eda-capabilities": True,
    "environment.tool-ip-selection": True,
}
HISTORICAL_PPA = {
    "candidate_ppa_run": False,
    "cells": 62199,
    "non_sram_area_mm2": 0.6108746272,
    "setup_slack_ns_at_100mhz": 0.1502,
}
SUCCESSOR_SEMANTIC_CONTRACT = {
    "contract_id": CONTRACT,
    "numeric_contract": {
        "carry_format": "signed16_q0_15",
        "carry_consumption": "exactly_once_next_input_rmsnorm_or_final_rmsnorm",
        "reconstructed_rmsnorm_input": "signed24_q8_15",
        "rmsnorm_square_sum": "unsigned56",
    },
    "resource_contract": {
        "immutable_metadata_bytes": 86592,
        "carry_buffer_bytes": 1792,
        "runtime_state_bytes": 128,
        "candidate_allocation_bytes": 88512,
        "carry_cycles_per_layer": 23296,
        "rmsnorm_square_cycles": 896,
        "planned_peak_sram_bytes": 466112,
        "remaining_sram_margin_bytes": 58176,
        "external_stream_width_bits": 128,
        "incremental_external_bytes_per_token": 0,
    },
    "reuse_contract": {
        "projection_lanes": 4,
        "down_projection_accumulator": "signed32_including_bias",
        "align_divide_remainder": "one_serial_engine_all_lanes_and_layers",
        "rmsnorm_square": "reuse_carry_engine_signed24x24_multiplier_one_lane_per_cycle",
        "dma_sram_command": "reuse_existing_dma_eight_banks_command_engine",
    },
    "dependency_contract": {
        "new_eda_pdk_board_license_compiler_runtime_or_external_ip_class": False,
        "successor_probe_executed": False,
        "rtl_or_downstream_workload_executed": False,
        "implementation_authorized": False,
    },
}
SELECTION_REQUIRED_MARKERS = (
    f"`{CONTRACT}`",
    "`signed16_q0_15`",
    "`exactly_once_next_input_rmsnorm_or_final_rmsnorm`",
    "`signed24_q8_15`",
    "`unsigned56`",
    "86,592",
    "1,792-byte carry buffer",
    "128 mutable state bytes",
    "88,512",
    "23,296 cycles per layer",
    "466,112",
    "58,176",
    "128-bit",
    "four W4A8 projection lanes",
    "signed-32 accumulator",
    "`implementation_authorized=false`",
)
SELECTION_FORBIDDEN_NORMALIZED_MARKERS = (
    f"compatibility rebind for `{PREDECESSOR}`",
    f"architecture acceptance of `{PREDECESSOR}`",
    f"compatible with `{PREDECESSOR}`",
    f"active architecture contract is `{PREDECESSOR}`",
    "signed-96 numerator",
    "unsigned-64 denominator",
    "86,720-byte",
)


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


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_record(record: dict[str, Any], label: str) -> None:
    path = ROOT / str(record.get("path", ""))
    require(path.is_file(), f"{label} missing: {path}")
    require(path.stat().st_size == int(record.get("bytes", -1)), f"{label} byte mismatch")
    require(sha256(path) == record.get("sha256"), f"{label} hash mismatch")


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = json.dumps(clone, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compact_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def public_projection_hashes(public: dict[str, Any]) -> dict[str, str]:
    return {
        f"{name}.current_architecture_performance_model": compact_sha256(
            public[name]["current_architecture_performance_model"]
        )
        for name in ("dashboard_fields", "implementation_frontier")
    }


def archive_file(path: Path) -> dict[str, Any]:
    digest = sha256(path)
    destination = ARCHIVE / f"{path.stem}.{digest[:16]}{path.suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(path.read_bytes())
    record = artifact(destination)
    require(record["bytes"] == path.stat().st_size, f"archive byte mismatch: {path}")
    require(record["sha256"] == digest, f"archive hash mismatch: {path}")
    return record


def preserved_contracts(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "ADVANCE",
        "ordered_supported_layer_operator_prefix": PREFIX,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "area_cap_non_sram_mm2": 2.0,
        "frequency_floor_mhz": 100.0,
        "abstract_streaming_memory_boundary_bits": 128,
        "historical_ppa_frontier": HISTORICAL_PPA,
        "sealed_predecessor_hashes": {
            name: record["sha256"]
            for name, record in decision["sealed_family_bindings"].items()
        },
    }


def verify_architecture_chain(*, allow_environment: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    pipeline = load(PIPELINE)
    expected_stages = {"architecture", "environment"} if allow_environment else {"architecture"}
    require(pipeline.get("current_stage") in expected_stages, "Manager-owned stage is not eligible")
    successor = pipeline.get("successor", {})
    require(successor.get("id") == CONTRACT, "pipeline successor differs")
    require(successor.get("state") == "frozen_unimplemented", "successor is not frozen/unimplemented")
    require(successor.get("predecessor") == PREDECESSOR, "pipeline predecessor differs")
    require(successor.get("downstream_runs") == EXPECTED_DOWNSTREAM_RUNS, "successor downstream locks differ")

    packet = load(PACKET)
    freeze = load(FREEZE)
    decision = load(ARCH_REVIEW)
    require(packet.get("contract_id") == CONTRACT, "architecture packet contract mismatch")
    require(packet.get("implementation_authorized") is False, "architecture packet authorizes implementation")
    require(packet.get("integrity", {}).get("canonical_sha256") == canonical_sha256(packet), "packet digest stale")
    require(freeze.get("contract_id") == CONTRACT, "architecture freeze contract mismatch")
    require(freeze.get("implementation_authorized") is False, "architecture freeze authorizes implementation")
    require(freeze.get("integrity", {}).get("canonical_sha256") == canonical_sha256(freeze), "freeze digest stale")
    require(packet.get("architecture_freeze") == artifact(FREEZE), "packet freeze binding stale")
    require(decision.get("contract_id") == CONTRACT, "architecture review contract mismatch")
    require(decision.get("reviewer_status") == "done", "architecture reviewer is not done")
    require(decision.get("status") == "accepted", "architecture review is not accepted")
    require(decision.get("architecture_accepted") is True, "architecture acceptance flag differs")
    require(decision.get("required_remediation") == "", "architecture remediation is not empty")
    require(decision.get("packet") == artifact(PACKET), "architecture review packet binding stale")
    require(decision.get("implementation_authorized") is False, "architecture review authorizes implementation")
    require(len(decision.get("checklist", {})) >= 10 and all(decision["checklist"].values()), "architecture checklist incomplete")
    require(all(decision.get("stage_checklist", {}).values()), "architecture stage checklist incomplete")
    for name, record in decision.get("sealed_family_bindings", {}).items():
        verify_record(record, f"sealed_family[{name}]")

    frontier = freeze.get("frontier", {})
    require(frontier.get("mode") == "ADVANCE", "mode changed")
    require(frontier.get("ordered_supported_layer_operator_prefix") == PREFIX, "prefix changed")
    require(frontier.get("first_unsupported_layer_operator") == FIRST_UNSUPPORTED, "unsupported frontier changed")
    require(frontier.get("area_cap_non_sram_mm2") == 2.0, "area cap changed")
    require(frontier.get("frequency_floor_mhz") == 100.0, "frequency floor changed")
    require(freeze.get("downstream_runs") == EXPECTED_FREEZE_DOWNSTREAM_RUNS, "freeze downstream locks changed")

    public = load(PUBLIC)
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public), "public digest stale")
    require(public.get("selected_replacement_contract", {}).get("contract_id") == CONTRACT, "public contract differs")
    require(public.get("selected_replacement_contract", {}).get("implementation_authorized") is False, "public authorizes implementation")
    hashes = public_projection_hashes(public)
    require(len(set(hashes.values())) == 1, "public architecture projections differ")
    require(hashes == decision.get("review_binding", {}).get("public_projection_hashes"), "accepted architecture projection hashes changed")
    return pipeline, decision


def get_bootstrap_source() -> tuple[dict[str, Any], dict[str, Any]]:
    current = load(AUDIT)
    if current.get("contract_binding", {}).get("active_contract_id") != CONTRACT:
        source_record = archive_file(AUDIT)
        if AUDIT_MD.is_file():
            archive_file(AUDIT_MD)
        return current, source_record
    source_record = current.get("evidence", {}).get("bootstrap_source_audit", {})
    verify_record(source_record, "bootstrap_source_audit")
    return load(ROOT / source_record["path"]), source_record


def retained_capability_records(source: dict[str, Any]) -> list[dict[str, Any]]:
    raw = copy.deepcopy(source.get("evidence", {}).get("artifacts", []))
    require(len(raw) == 27, "bootstrap capability artifact count changed")
    historical_probe = source.get("evidence", {}).get("compatibility_probe")
    require(isinstance(historical_probe, dict), "historical compatibility probe record missing")
    records = raw + [copy.deepcopy(historical_probe)]
    require(len({item.get("path") for item in records}) == 28, "retained capability records are not unique")
    for index, record in enumerate(records):
        verify_record(record, f"retained_capability[{index}]")
    return records


def selection_document_semantic_contract() -> dict[str, Any]:
    freeze = load(FREEZE)
    memory = load(MEMORY_MODEL)
    freeze_numeric = freeze.get("numeric_contract", {})
    for key, value in SUCCESSOR_SEMANTIC_CONTRACT["numeric_contract"].items():
        require(freeze_numeric.get(key) == value, f"architecture freeze numeric contract differs: {key}")
    planning = freeze.get("planning", {})
    resources = SUCCESSOR_SEMANTIC_CONTRACT["resource_contract"]
    require(planning.get("candidate_allocation_bytes") == resources["candidate_allocation_bytes"], "freeze candidate allocation differs")
    require(planning.get("carry_cycles_per_layer") == resources["carry_cycles_per_layer"], "freeze carry cycles differ")
    require(planning.get("planned_peak_sram_bytes") == resources["planned_peak_sram_bytes"], "freeze planned SRAM peak differs")
    require(planning.get("remaining_sram_margin_bytes") == resources["remaining_sram_margin_bytes"], "freeze SRAM margin differs")

    hierarchy = memory.get("memory_hierarchy", {})
    compute = memory.get("compute_model", {})
    numeric = memory.get("numeric_contract", {})
    require(memory.get("contract_id") == CONTRACT, "memory model contract differs")
    require(hierarchy.get("immutable_metadata_bytes") == resources["immutable_metadata_bytes"], "memory immutable metadata differs")
    require(hierarchy.get("carry_buffer_bytes") == resources["carry_buffer_bytes"], "memory carry buffer differs")
    require(hierarchy.get("runtime_state_bytes") == resources["runtime_state_bytes"], "memory runtime state differs")
    require(hierarchy.get("candidate_allocation_bytes") == resources["candidate_allocation_bytes"], "memory candidate allocation differs")
    require(hierarchy.get("planned_peak_sram_bytes") == resources["planned_peak_sram_bytes"], "memory planned SRAM peak differs")
    require(hierarchy.get("remaining_sram_margin_bytes") == resources["remaining_sram_margin_bytes"], "memory SRAM margin differs")
    require(hierarchy.get("external_stream_width_bits") == resources["external_stream_width_bits"], "memory boundary differs")
    require(hierarchy.get("incremental_external_bytes_per_token") == resources["incremental_external_bytes_per_token"], "memory external traffic differs")
    require(compute.get("carry_cycles_per_layer_token") == resources["carry_cycles_per_layer"], "memory carry schedule differs")
    require(compute.get("rmsnorm_square_cycles_selected") == resources["rmsnorm_square_cycles"], "memory RMSNorm square schedule differs")
    require(numeric.get("reconstructed_rmsnorm_input") == "signed24_q8_15", "memory reconstruction format differs")
    require(numeric.get("rmsnorm_square_sum") == "unsigned56_across_896_lanes", "memory square-sum format differs")

    for path in (TOOLCHAIN, IP_REUSE):
        content = path.read_text(encoding="utf-8")
        normalized = " ".join(content.split())
        for marker in SELECTION_REQUIRED_MARKERS:
            require(" ".join(marker.split()) in normalized, f"selection document missing successor semantic marker {marker!r}: {path.relative_to(ROOT)}")
        for marker in SELECTION_FORBIDDEN_NORMALIZED_MARKERS:
            require(marker not in normalized, f"selection document retains predecessor semantic marker {marker!r}: {path.relative_to(ROOT)}")
    return copy.deepcopy(SUCCESSOR_SEMANTIC_CONTRACT)


def selection_records() -> list[dict[str, Any]]:
    selection_document_semantic_contract()
    records = [artifact(TOOLCHAIN), artifact(IP_REUSE)]
    for index, record in enumerate(records):
        verify_record(record, f"selection_document[{index}]")
    return records


def existing_pre_transition_snapshot() -> dict[str, Any]:
    current = load(AUDIT)
    require(current.get("contract_binding", {}).get("active_contract_id") == CONTRACT, "live audit is not successor-bound")
    record = copy.deepcopy(current.get("contract_binding", {}).get("pre_transition_pipeline_snapshot", {}))
    verify_record(record, "pre_transition_pipeline_snapshot")
    packet_pipeline = next(
        item for item in load(PACKET)["source_artifacts"]
        if item.get("path") == "research/PIPELINE_STATE.json"
    )
    require(record.get("bytes") == packet_pipeline.get("bytes"), "pre-transition pipeline byte binding differs")
    require(record.get("sha256") == packet_pipeline.get("sha256"), "pre-transition pipeline hash binding differs")
    return record


def advance_pipeline(pipeline: dict[str, Any], now: str) -> dict[str, Any]:
    require(pipeline.get("current_stage") == "architecture", "pipeline already left architecture")
    pre_transition = archive_file(PIPELINE)
    packet_pipeline = next(
        record for record in load(PACKET)["source_artifacts"]
        if record.get("path") == "research/PIPELINE_STATE.json"
    )
    require(pre_transition["bytes"] == packet_pipeline["bytes"], "pre-transition pipeline byte binding differs")
    require(pre_transition["sha256"] == packet_pipeline["sha256"], "pre-transition pipeline hash binding differs")

    pipeline["current_stage"] = "environment"
    pipeline["stages"]["architecture"]["status"] = "done"
    pipeline["stages"]["environment"]["status"] = "in_progress"
    pipeline["stages"]["rtl"]["status"] = "locked_pending_predecessor_stage_acceptance"
    reason = (
        "Fresh independent architecture acceptance now satisfies the literal gate with structurally "
        "empty remediation and post-review binding checks. Advance only to environment for one "
        "successor-specific hash-only compatibility audit and independent review; implementation, "
        "RTL, verification, PPA, physical, prototype, benchmark, and signoff remain locked."
    )
    pipeline.setdefault("stage_history", []).append({
        "at": now,
        "by": "manager",
        "direction": "advance",
        "from_stage": "architecture",
        "reason": reason,
        "to_stage": "environment",
    })
    dump(PIPELINE, pipeline)
    return pre_transition


def public_artifacts(public: dict[str, Any], additions: list[Path]) -> list[dict[str, Any]]:
    paths = {
        str(record.get("path"))
        for record in public.get("artifact_hashes", [])
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    paths.update(path.relative_to(ROOT).as_posix() for path in additions)
    return [artifact(ROOT / path) for path in sorted(paths) if (ROOT / path).is_file()]


def refresh_public_pending(now: str, retained_count: int) -> None:
    public = load(PUBLIC)
    latest_environment = {
        "status": "hash_only_evidence_ready_independent_review_pending",
        "contract_binding": CONTRACT,
        "generated_at_utc": now,
        "checklist": ENVIRONMENT_CHECKLIST,
        "retained_capability_artifact_count": retained_count,
        "tool_ip_selection_document_count": 2,
        "audit_mode": "successor_specific_hash_only_no_probe_or_workload_rerun",
        "raw_capability_probes_rerun": False,
        "reviewer_verdict": "pending",
        "implementation_authorized": False,
        "stage_transition_owner": "Manager",
    }
    public["latest_environment_stage"] = copy.deepcopy(latest_environment)
    public["latest_decision"] = "cross_layer_error_carry_environment_hash_only_evidence_ready_review_pending"
    public["routing_status"] = "independent_environment_review_pending"
    public["routing_authorized"] = "none_pending_independent_environment_review"
    public["required_manager_action"] = "hold_environment_until_independent_review"
    public["required_operator_action"] = "none"
    public["generated_at_utc"] = now
    public["last_updated_utc"] = now
    public["stage_closing"] = False

    selected = public["selected_replacement_contract"]
    require(selected.get("contract_id") == CONTRACT, "public selected contract differs")
    selected.update({
        "status": "environment_hash_only_evidence_ready_review_pending",
        "environment_review_status": "pending_fresh_independent_reviewer",
        "implementation_authorized": False,
        "required_manager_action": "hold_environment_until_independent_review",
        "stage_closing": False,
    })
    public["architecture_successor_dispatch"].update({
        "current_stage": "environment",
        "status": "environment_hash_only_evidence_ready_review_pending",
        "required_manager_action": "hold_environment_until_independent_review",
        "implementation_authorized": False,
    })

    stage = public["stage"]
    stage.update({
        "current_stage": "environment",
        "current_stage_status": "hash_only_evidence_ready_independent_review_pending",
        "completed_prior_stages": ["definition", "architecture"],
        "current_stage_checklist": ENVIRONMENT_CHECKLIST,
        "current_stage_evidence": [
            "research/ENVIRONMENT_AUDIT.json",
            "research/TOOLCHAIN_CANDIDATES.md",
            "research/IP_REUSE_PLAN.md",
            ARCH_REVIEW.relative_to(ROOT).as_posix(),
        ],
        "planner_may_advance_stage": False,
        "stage_closing": True,
        "stage_transition_owner": "Manager",
    })

    rtl = public["latest_rtl_candidate"]
    rtl.update({
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "required_manager_action": "hold_environment_until_independent_review",
        "status": "unimplemented_environment_review_pending",
        "stage_closing": False,
    })
    for name in ("dashboard_fields", "implementation_frontier"):
        container = public[name]
        container.update({
            "current_stage": "environment",
            "current_mode": "ADVANCE",
            "latest_decision": public["latest_decision"],
            "routing_status": public["routing_status"],
            "required_manager_action": "hold_environment_until_independent_review",
            "required_operator_action": "none",
            "latest_environment_stage": copy.deepcopy(latest_environment),
            "latest_rtl_candidate": copy.deepcopy(rtl),
        })
        mechanism = container["candidate_mechanism"]
        require(mechanism.get("contract_id") == CONTRACT, f"{name} candidate contract differs")
        mechanism.update({
            "status": "environment_hash_only_evidence_ready_review_pending",
            "environment_review_status": "pending_fresh_independent_reviewer",
            "implementation_authorized": False,
        })

    blockers = [
        item for item in public.get("blockers", [])
        if item.get("id") not in {"manager_stage_advance_pending", "independent_environment_review_pending"}
    ]
    blockers.append({
        "id": "independent_environment_review_pending",
        "severity": "gate",
        "status": "open",
        "owner": "independent_reviewer",
        "detail": "The exact-successor hash-only environment audit requires one independent review.",
    })
    public["blockers"] = blockers
    claims = [
        item for item in public.get("public_claims", [])
        if "current Manager-owned stage is" not in str(item.get("claim", ""))
        and "environment evidence" not in str(item.get("claim", ""))
    ]
    claims.insert(0, {"claim": "current Manager-owned stage is environment", "evidence": ["research/PIPELINE_STATE.json"]})
    claims.insert(1, {
        "claim": f"successor-specific environment evidence for {CONTRACT} is hash-only and review-pending; no RTL or downstream workload ran",
        "evidence": ["research/ENVIRONMENT_AUDIT.json"],
    })
    public["public_claims"] = claims
    public["artifact_hashes"] = public_artifacts(public, [AUDIT, TOOLCHAIN, IP_REUSE, PACKET, FREEZE, ARCH_REVIEW])
    public.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)


def refresh() -> None:
    pipeline, decision = verify_architecture_chain(allow_environment=True)
    source, source_record = get_bootstrap_source()
    retained = retained_capability_records(source)
    selections = selection_records()
    semantic_contract = selection_document_semantic_contract()
    now = utc_now()
    if pipeline.get("current_stage") == "architecture":
        pre_transition = advance_pipeline(pipeline, now)
    else:
        pre_transition = existing_pre_transition_snapshot()
    live_pipeline = load(PIPELINE)
    preserved = preserved_contracts(decision)

    audit = {
        "schema_version": 7,
        "project": "ACE-2",
        "stage": "environment",
        "generated_at_utc": now,
        "audit_scope": {
            "mode": "successor_specific_hash_only",
            "delivery_level": "pre_tapeout",
            "primary_fast_loop_platform": "sky130hd",
            "comparative_platforms": ["nangate45", "asap7"],
            "forbidden_downstream_work_respected": True,
            "new_probe_executed": False,
            "rtl_or_downstream_workload_executed": False,
        },
        "contract_binding": {
            "active_contract_id": CONTRACT,
            "active_repair_scope": "all_24_layer_signed_q0_15_quantization_error_carry_with_final_output_reinjection",
            "architecture_packet": artifact(PACKET),
            "architecture_freeze": artifact(FREEZE),
            "architecture_review": artifact(ARCH_REVIEW),
            "pre_transition_pipeline_snapshot": pre_transition,
            "pipeline_state": artifact(PIPELINE),
            "current_stage": "environment",
            "environment_review_status": "pending_fresh_independent_reviewer",
            "implementation_authorized": False,
            "preserved_contracts": preserved,
        },
        "compatibility_determination": {
            "status": "pass_ready_for_independent_environment_review",
            "method": "hash_only_dependency_delta_review",
            "new_eda_pdk_board_license_compiler_runtime_or_external_ip_class": False,
            "successor_numeric_requirements": {
                "carry_format": "signed16_q0_15",
                "carry_consumption": "exactly_once_next_input_rmsnorm_or_final_rmsnorm",
                "reconstructed_rmsnorm_input": "signed24_q8_15",
                "rmsnorm_square_sum": "unsigned56",
            },
            "successor_semantic_contract": semantic_contract,
            "environment_coverage_basis": [
                "retained simulator lint formal synthesis STA physical and signoff-tool capability hashes",
                "retained Python gmpy2 MPFR compiler and runtime capability hashes",
                "maintained public ORFS SKY130 tool and IP selection documents",
                "accepted architecture-only width control memory and interface contract",
            ],
            "historical_predecessor_probe_treatment": "hash_retained_for_environment_provenance_only_not_used_as_successor_execution_evidence",
            "new_probe_required": False,
            "reason": "The successor changes the frozen numerical/dataflow architecture but introduces no new environment dependency class; compatibility is therefore established by exact hash rebind and independent review, not by executing RTL or a downstream workload.",
        },
        "selected_environment": copy.deepcopy(source["selected_environment"]),
        "tool_ip_selection": {
            **copy.deepcopy(source.get("tool_ip_selection", {})),
            "status": f"selected_before_{CONTRACT}_rtl",
            "selection_documents": selections,
            "selection_document_semantic_contract": semantic_contract,
            "third_party_rtl_vendored": False,
            "proprietary_or_credentialed_dependency": False,
        },
        "stage_gate_items": ENVIRONMENT_CHECKLIST,
        "readiness_summary": {
            "current_active_contract_compatible": True,
            "environment_evidence_ready_for_independent_review": True,
            "fresh_independent_environment_acceptance": False,
            "implementation_authorized": False,
            "manager_stage_transition_owner": "Manager",
            "planner_stage_transition_authorized": False,
            "retained_capability_artifact_count": len(retained),
            "tool_ip_selection_document_count": len(selections),
            "raw_capability_probes_rerun": False,
            "successor_probe_executed": False,
        },
        "evidence": {
            "bootstrap_source_audit": source_record,
            "retained_capability_artifacts": retained,
            "selection_documents": selections,
            "reproduction_command": "python3 tools/bind_cross_layer_error_carry_environment.py",
            "validation_command": "python3 tools/bind_cross_layer_error_carry_environment.py --check",
        },
        "non_gating_future_limitations": copy.deepcopy(source.get("non_gating_future_limitations", [])),
        "claim_boundaries": [
            "28 retained capability artifacts and two maintained selection documents were byte/hash verified without rerun",
            "the predecessor compatibility probe is retained only as historical environment provenance and is not successor execution evidence",
            "no successor RTL simulation lint formal synthesis PPA physical prototype benchmark signoff GDS tapeout or silicon workload ran",
            "the 2.0 mm2 non-SRAM cap and 100 MHz floor remain unchanged and unproven for the successor",
            "historical 62199-cell 0.6108746272-mm2 and +0.1502-ns PPA remains historical only",
            "environment compatibility does not authorize implementation or an environment-to-RTL transition",
        ],
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    require(live_pipeline.get("current_stage") == "environment", "pipeline transition did not persist")
    audit["integrity"]["canonical_sha256"] = canonical_sha256(audit)
    dump(AUDIT, audit)
    AUDIT_MD.write_text(
        "# ACE-2 Environment Audit\n\n"
        f"- Contract: `{CONTRACT}`\n"
        "- Manager-owned stage: `environment`\n"
        "- Audit mode: `successor-specific hash-only`\n"
        "- Retained capability artifacts: `28/28` byte/hash verified; no rerun\n"
        "- Maintained tool/IP selection documents: `2/2` byte/hash verified\n"
        "- Fresh independent environment acceptance: `pending`\n"
        "- Implementation authorized: `false`\n\n"
        "No successor RTL, verification, synthesis/PPA, physical, prototype, benchmark, "
        "signoff, GDS, tapeout, or silicon workload was run. The predecessor synthetic "
        "probe is retained as historical environment provenance only.\n",
        encoding="utf-8",
    )
    refresh_public_pending(now, len(retained))


def validate() -> None:
    pipeline, decision = verify_architecture_chain(allow_environment=True)
    require(pipeline.get("current_stage") == "environment", "Manager-owned stage is not environment")
    require(pipeline.get("stages", {}).get("rtl", {}).get("status") == "locked_pending_predecessor_stage_acceptance", "RTL stage is not locked")
    audit = load(AUDIT)
    require(audit.get("integrity", {}).get("canonical_sha256") == canonical_sha256(audit), "audit digest stale")
    binding = audit.get("contract_binding", {})
    require(binding.get("active_contract_id") == CONTRACT, "audit contract differs")
    require(binding.get("current_stage") == "environment", "audit stage differs")
    require(binding.get("implementation_authorized") is False, "audit authorizes implementation")
    require(binding.get("architecture_packet") == artifact(PACKET), "audit packet binding stale")
    require(binding.get("architecture_freeze") == artifact(FREEZE), "audit freeze binding stale")
    require(binding.get("architecture_review") == artifact(ARCH_REVIEW), "audit architecture review binding stale")
    require(binding.get("pipeline_state") == artifact(PIPELINE), "audit pipeline binding stale")
    require(binding.get("preserved_contracts") == preserved_contracts(decision), "audit preserved contracts differ")
    pre_transition = binding.get("pre_transition_pipeline_snapshot", {})
    verify_record(pre_transition, "pre_transition_pipeline_snapshot")
    packet_pipeline = next(
        record for record in load(PACKET)["source_artifacts"]
        if record.get("path") == "research/PIPELINE_STATE.json"
    )
    require(pre_transition.get("bytes") == packet_pipeline.get("bytes"), "pipeline snapshot byte binding differs")
    require(pre_transition.get("sha256") == packet_pipeline.get("sha256"), "pipeline snapshot hash binding differs")
    retained = audit.get("evidence", {}).get("retained_capability_artifacts", [])
    require(len(retained) == 28, "retained capability artifact count differs")
    for index, record in enumerate(retained):
        verify_record(record, f"retained_capability[{index}]")
    selections = audit.get("evidence", {}).get("selection_documents", [])
    require(selections == selection_records(), "selection document bindings differ")
    semantic_contract = selection_document_semantic_contract()
    require(
        audit.get("compatibility_determination", {}).get("successor_semantic_contract") == semantic_contract,
        "audit successor semantic contract differs",
    )
    require(
        audit.get("tool_ip_selection", {}).get("selection_document_semantic_contract") == semantic_contract,
        "audit selection document semantic contract differs",
    )
    require(audit.get("stage_gate_items") == ENVIRONMENT_CHECKLIST, "environment checklist differs")
    require(audit.get("audit_scope", {}).get("new_probe_executed") is False, "audit claims a new probe")
    require(audit.get("audit_scope", {}).get("rtl_or_downstream_workload_executed") is False, "audit claims downstream execution")
    review_status = binding.get("environment_review_status")
    require(review_status in {"pending_fresh_independent_reviewer", "done"}, "environment review status differs")

    public = load(PUBLIC)
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public), "public digest stale")
    require(public.get("stage", {}).get("current_stage") == "environment", "public stage differs")
    require(public.get("selected_replacement_contract", {}).get("contract_id") == CONTRACT, "public contract differs")
    require(public.get("selected_replacement_contract", {}).get("implementation_authorized") is False, "public authorizes implementation")
    require(public_projection_hashes(public) == decision.get("review_binding", {}).get("public_projection_hashes"), "public architecture projections changed")
    if review_status == "pending_fresh_independent_reviewer":
        require(public.get("routing_status") == "independent_environment_review_pending", "public review-pending routing differs")
    else:
        require(ENV_REVIEW.is_file(), "accepted environment review decision is missing")
        environment_review = load(ENV_REVIEW)
        require(environment_review.get("status") == "accepted", "environment review is not accepted")
        require(environment_review.get("required_remediation") == "", "environment remediation is not empty")
        require(public.get("routing_status") == "manager_stage_advance_pending", "public accepted routing differs")
    for index, record in enumerate(public.get("artifact_hashes", [])):
        verify_record(record, f"public_artifact[{index}]")
    print(
        "ACE2_CROSS_LAYER_ERROR_CARRY_ENVIRONMENT_BINDING_PASS "
        f"contract={CONTRACT} retained=28 selections=2 review={review_status} "
        "implementation_authorized=false downstream_runs=unrun"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        validate()
    else:
        refresh()
        validate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
