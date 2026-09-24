#!/usr/bin/env python3
"""Run and seal one independent hash-only environment review for the active successor."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
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
PREDECESSOR = "shared_down_projection_residual_fusion_v1"
MISSION_ID = "environment-review-cross-layer-quantization-error-carry-final-output-v1"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
AUDIT = ROOT / "research/ENVIRONMENT_AUDIT.json"
AUDIT_MD = ROOT / "research/ENVIRONMENT_AUDIT.md"
TOOLCHAIN = ROOT / "research/TOOLCHAIN_CANDIDATES.md"
IP_REUSE = ROOT / "research/IP_REUSE_PLAN.md"
MEMORY_MODEL = ROOT / "design/MEMORY_MODEL.json"
PACKET = ROOT / f"evidence/{CONTRACT}/latest/ARCHITECTURE_REVIEW_PACKET.json"
FREEZE = ROOT / f"evidence/{CONTRACT}/architecture_freeze.json"
ARCH_REVIEW = ROOT / f"evidence/review/architecture_stage_closing_{CONTRACT}/decision.json"
OUT = ROOT / f"evidence/review/environment_stage_closing_{CONTRACT}/decision.json"
VERDICT = ROOT / "research/ENVIRONMENT_REVIEWER_VERDICT.json"
CERT = ROOT / "research/ENVIRONMENT_L2_CERTIFICATION.md"
CHECKPOINT = ROOT / "CHECKPOINT.md"
BINDER = ROOT / "tools/bind_cross_layer_error_carry_environment.py"
ARCHIVE = ROOT / f"research/archive/environment/historical_before_{CONTRACT}/review_inputs"
DECISION_ARCHIVE = OUT.parent / "archive"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
ENVIRONMENT_CHECKLIST = {
    "environment.eda-capabilities": True,
    "environment.tool-ip-selection": True,
}
REVIEW_CHECKLIST = {
    "accepted_architecture_decision_is_exactly_bound": True,
    "architecture_remediation_is_structurally_empty": True,
    "all_28_retained_capability_artifacts_byte_and_hash_verify": True,
    "both_maintained_tool_ip_selection_documents_byte_and_hash_verify": True,
    "both_maintained_tool_ip_selection_documents_semantically_bind_the_active_successor": True,
    "successor_numeric_resource_and_reuse_contract_matches_the_accepted_architecture": True,
    "successor_introduces_no_new_environment_dependency_class": True,
    "audit_is_hash_only_and_no_probe_or_workload_was_rerun": True,
    "advance_mode_prefix_and_first_unsupported_operator_are_preserved": True,
    "area_frequency_and_128_bit_boundary_locks_are_preserved": True,
    "historical_ppa_is_not_presented_as_successor_ppa": True,
    "all_three_sealed_predecessor_hashes_are_preserved": True,
    "implementation_and_rtl_downstream_work_remain_locked": True,
    "manager_stage_transition_ownership_is_preserved": True,
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
    require(path.is_file(), f"{label} missing")
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


def projection_hashes(public: dict[str, Any]) -> dict[str, str]:
    return {
        f"{name}.current_architecture_performance_model": compact_sha256(
            public[name]["current_architecture_performance_model"]
        )
        for name in ("dashboard_fields", "implementation_frontier")
    }


def run_check(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    require(completed.returncode == 0, f"check failed: {' '.join(command)}\n{completed.stdout}")
    return completed.stdout.strip()


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


def archive_reviewed_audit() -> dict[str, Any]:
    digest = sha256(AUDIT)
    destination = ARCHIVE / f"ENVIRONMENT_AUDIT.{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(AUDIT, destination)
    record = artifact(destination)
    require(record["bytes"] == AUDIT.stat().st_size and record["sha256"] == digest, "reviewed audit archive mismatch")
    return record


def archive_prior_decision() -> dict[str, Any] | None:
    if not OUT.is_file():
        return None
    digest = sha256(OUT)
    destination = DECISION_ARCHIVE / f"decision.{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(OUT, destination)
    record = artifact(destination)
    require(record["bytes"] == OUT.stat().st_size and record["sha256"] == digest, "prior decision archive mismatch")
    return record


def verify_successor_semantics(audit: dict[str, Any]) -> dict[str, Any]:
    freeze = load(FREEZE)
    memory = load(MEMORY_MODEL)
    freeze_numeric = freeze.get("numeric_contract", {})
    for key, value in SUCCESSOR_SEMANTIC_CONTRACT["numeric_contract"].items():
        require(freeze_numeric.get(key) == value, f"architecture freeze numeric contract differs: {key}")
    resources = SUCCESSOR_SEMANTIC_CONTRACT["resource_contract"]
    planning = freeze.get("planning", {})
    require(planning.get("candidate_allocation_bytes") == resources["candidate_allocation_bytes"], "freeze candidate allocation differs")
    require(planning.get("carry_cycles_per_layer") == resources["carry_cycles_per_layer"], "freeze carry cycles differ")
    require(planning.get("planned_peak_sram_bytes") == resources["planned_peak_sram_bytes"], "freeze planned SRAM peak differs")
    require(planning.get("remaining_sram_margin_bytes") == resources["remaining_sram_margin_bytes"], "freeze SRAM margin differs")

    hierarchy = memory.get("memory_hierarchy", {})
    compute = memory.get("compute_model", {})
    require(memory.get("contract_id") == CONTRACT, "memory model contract differs")
    for key in (
        "immutable_metadata_bytes",
        "carry_buffer_bytes",
        "runtime_state_bytes",
        "candidate_allocation_bytes",
        "planned_peak_sram_bytes",
        "remaining_sram_margin_bytes",
        "external_stream_width_bits",
        "incremental_external_bytes_per_token",
    ):
        require(hierarchy.get(key) == resources[key], f"memory resource differs: {key}")
    require(compute.get("carry_cycles_per_layer_token") == resources["carry_cycles_per_layer"], "memory carry schedule differs")
    require(compute.get("rmsnorm_square_cycles_selected") == resources["rmsnorm_square_cycles"], "memory RMSNorm schedule differs")

    for path in (TOOLCHAIN, IP_REUSE):
        content = path.read_text(encoding="utf-8")
        normalized = " ".join(content.split())
        for marker in SELECTION_REQUIRED_MARKERS:
            require(" ".join(marker.split()) in normalized, f"selection document missing successor semantic marker {marker!r}: {path.relative_to(ROOT)}")
        for marker in SELECTION_FORBIDDEN_NORMALIZED_MARKERS:
            require(marker not in normalized, f"selection document retains predecessor semantic marker {marker!r}: {path.relative_to(ROOT)}")

    expected = copy.deepcopy(SUCCESSOR_SEMANTIC_CONTRACT)
    require(
        audit.get("compatibility_determination", {}).get("successor_semantic_contract") == expected,
        "audit successor semantic contract differs",
    )
    require(
        audit.get("tool_ip_selection", {}).get("selection_document_semantic_contract") == expected,
        "audit selection semantic contract differs",
    )
    return expected


def preflight() -> dict[str, Any]:
    marker = run_check([sys.executable, "tools/bind_cross_layer_error_carry_environment.py", "--check"])
    require("ACE2_CROSS_LAYER_ERROR_CARRY_ENVIRONMENT_BINDING_PASS" in marker, "environment binder marker missing")
    pipeline = load(PIPELINE)
    audit = load(AUDIT)
    architecture = load(ARCH_REVIEW)
    public = load(PUBLIC)
    require(pipeline.get("current_stage") == "environment", "Manager-owned stage is not environment")
    require(audit.get("contract_binding", {}).get("active_contract_id") == CONTRACT, "audit contract differs")
    require(audit.get("contract_binding", {}).get("environment_review_status") == "pending_fresh_independent_reviewer", "audit is not review-pending")
    require(audit.get("readiness_summary", {}).get("environment_evidence_ready_for_independent_review") is True, "audit is not review-ready")
    require(audit.get("readiness_summary", {}).get("fresh_independent_environment_acceptance") is False, "audit already claims review acceptance")
    require(audit.get("stage_gate_items") == ENVIRONMENT_CHECKLIST, "environment checklist differs")
    require(audit.get("audit_scope", {}).get("new_probe_executed") is False, "audit claims a new probe")
    require(audit.get("audit_scope", {}).get("rtl_or_downstream_workload_executed") is False, "audit claims a downstream workload")
    retained = audit.get("evidence", {}).get("retained_capability_artifacts", [])
    require(len(retained) == 28, "retained capability count differs")
    for index, record in enumerate(retained):
        verify_record(record, f"retained_capability[{index}]")
    selections = audit.get("evidence", {}).get("selection_documents", [])
    require(len(selections) == 2, "selection document count differs")
    for index, record in enumerate(selections):
        verify_record(record, f"selection_document[{index}]")
    semantic_contract = verify_successor_semantics(audit)
    require(architecture.get("required_remediation") == "", "architecture remediation is not empty")
    require(architecture.get("status") == "accepted", "architecture review is not accepted")
    require(architecture.get("implementation_authorized") is False, "architecture review authorizes implementation")
    require(projection_hashes(public) == architecture.get("review_binding", {}).get("public_projection_hashes"), "public architecture projection hashes changed")
    return {
        "decisive_check": marker,
        "artifacts": {
            "audit": artifact(AUDIT),
            "architecture_packet": artifact(PACKET),
            "architecture_freeze": artifact(FREEZE),
            "architecture_review": artifact(ARCH_REVIEW),
            "toolchain": artifact(TOOLCHAIN),
            "ip_reuse": artifact(IP_REUSE),
            "pipeline": artifact(PIPELINE),
            "public_status": artifact(PUBLIC),
            "binder": artifact(BINDER),
        },
        "retained_capability_artifact_count": len(retained),
        "selection_document_count": len(selections),
        "selection_document_semantic_contract": semantic_contract,
        "environment_checklist": ENVIRONMENT_CHECKLIST,
        "review_checklist": REVIEW_CHECKLIST,
        "preserved_contracts": audit["contract_binding"]["preserved_contracts"],
        "public_projection_hashes": projection_hashes(public),
        "prohibited_runs": [
            "rtl",
            "verification",
            "simulation",
            "formal",
            "synthesis",
            "ppa",
            "physical",
            "prototype",
            "benchmark",
            "signoff",
        ],
    }


def update_public(reviewed_at: str) -> None:
    public = load(PUBLIC)
    latest_environment = {
        "status": "independent_review_done_manager_stage_transition_pending",
        "contract_binding": CONTRACT,
        "generated_at_utc": reviewed_at,
        "checklist": ENVIRONMENT_CHECKLIST,
        "retained_capability_artifact_count": 28,
        "tool_ip_selection_document_count": 2,
        "audit_mode": "successor_specific_hash_only_no_probe_or_workload_rerun",
        "raw_capability_probes_rerun": False,
        "reviewer_verdict": "done",
        "review_decision": artifact(OUT),
        "implementation_authorized": False,
        "stage_transition_owner": "Manager",
    }
    public["latest_environment_stage"] = copy.deepcopy(latest_environment)
    public["latest_decision"] = "cross_layer_error_carry_environment_independently_accepted"
    public["routing_status"] = "manager_stage_advance_pending"
    public["routing_authorized"] = "manager_may_advance_environment_to_rtl_only"
    public["required_manager_action"] = "manager_may_advance_environment_to_rtl_in_order"
    public["required_operator_action"] = "none"
    public["generated_at_utc"] = reviewed_at
    public["last_updated_utc"] = reviewed_at
    public["stage_closing"] = True

    selected = public["selected_replacement_contract"]
    selected.update({
        "status": "environment_independently_accepted_manager_rtl_entry_pending",
        "environment_review_status": "done",
        "implementation_authorized": False,
        "required_manager_action": "manager_may_advance_environment_to_rtl_in_order",
        "stage_closing": True,
    })
    public["architecture_successor_dispatch"].update({
        "current_stage": "environment",
        "status": "environment_independently_accepted_manager_rtl_entry_pending",
        "required_manager_action": "manager_may_advance_environment_to_rtl_in_order",
        "implementation_authorized": False,
    })
    public["stage"].update({
        "current_stage": "environment",
        "current_stage_status": "environment_independently_accepted_manager_rtl_entry_pending",
        "completed_prior_stages": ["definition", "architecture"],
        "current_stage_checklist": ENVIRONMENT_CHECKLIST,
        "current_stage_evidence": [
            "research/ENVIRONMENT_AUDIT.json",
            "research/TOOLCHAIN_CANDIDATES.md",
            "research/IP_REUSE_PLAN.md",
            OUT.relative_to(ROOT).as_posix(),
        ],
        "planner_may_advance_stage": False,
        "stage_closing": True,
        "stage_transition_owner": "Manager",
    })
    rtl = public["latest_rtl_candidate"]
    rtl.update({
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "required_manager_action": "manager_may_advance_environment_to_rtl_in_order",
        "status": "unimplemented_environment_accepted_manager_rtl_transition_pending",
        "stage_closing": False,
    })
    for name in ("dashboard_fields", "implementation_frontier"):
        container = public[name]
        container.update({
            "current_stage": "environment",
            "latest_decision": public["latest_decision"],
            "routing_status": public["routing_status"],
            "required_manager_action": "manager_may_advance_environment_to_rtl_in_order",
            "required_operator_action": "none",
            "latest_environment_stage": copy.deepcopy(latest_environment),
            "latest_rtl_candidate": copy.deepcopy(rtl),
        })
        container["candidate_mechanism"].update({
            "status": "environment_independently_accepted_manager_rtl_entry_pending",
            "environment_review_status": "done",
            "implementation_authorized": False,
        })

    blockers = [
        item for item in public.get("blockers", [])
        if item.get("id") not in {"independent_environment_review_pending", "manager_stage_advance_pending"}
    ]
    blockers.append({
        "id": "manager_stage_advance_pending",
        "severity": "gate",
        "status": "open",
        "owner": "manager",
        "detail": "Independent environment review is done; no RTL transition or implementation authorization is part of this mission.",
    })
    public["blockers"] = blockers
    claims = [
        item for item in public.get("public_claims", [])
        if "environment evidence" not in str(item.get("claim", ""))
        and "independent environment reviewer" not in str(item.get("claim", ""))
    ]
    claims.insert(1, {
        "claim": f"independent environment reviewer accepted the exact-successor hash-only audit for {CONTRACT}; implementation remains unauthorized",
        "evidence": [OUT.relative_to(ROOT).as_posix(), "research/ENVIRONMENT_AUDIT.json"],
    })
    public["public_claims"] = claims
    paths = {
        str(record.get("path"))
        for record in public.get("artifact_hashes", [])
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    paths.update({
        AUDIT.relative_to(ROOT).as_posix(),
        OUT.relative_to(ROOT).as_posix(),
        PACKET.relative_to(ROOT).as_posix(),
        FREEZE.relative_to(ROOT).as_posix(),
        ARCH_REVIEW.relative_to(ROOT).as_posix(),
        TOOLCHAIN.relative_to(ROOT).as_posix(),
        IP_REUSE.relative_to(ROOT).as_posix(),
    })
    public["artifact_hashes"] = [artifact(ROOT / path) for path in sorted(paths) if (ROOT / path).is_file()]
    public.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)


def publish(payload: dict[str, Any], reviewed_audit: dict[str, Any]) -> None:
    dump(OUT, payload)
    audit = load(AUDIT)
    audit["contract_binding"]["environment_review_status"] = "done"
    audit["readiness_summary"]["fresh_independent_environment_acceptance"] = True
    audit["independent_review"] = {
        "status": "done",
        "reviewed_at_utc": payload["reviewed_at_utc"],
        "reviewed_audit": reviewed_audit,
        "decision": artifact(OUT),
        "implementation_authorized": False,
        "stage_transition_owner": "Manager",
    }
    audit["generated_at_utc"] = payload["reviewed_at_utc"]
    audit["integrity"] = {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None}
    audit["integrity"]["canonical_sha256"] = canonical_sha256(audit)
    dump(AUDIT, audit)
    AUDIT_MD.write_text(
        "# ACE-2 Environment Audit\n\n"
        f"- Contract: `{CONTRACT}`\n"
        "- Manager-owned stage: `environment`\n"
        "- Audit mode: `successor-specific hash-only`\n"
        "- Retained capability artifacts: `28/28` byte/hash verified; no rerun\n"
        "- Maintained tool/IP selection documents: `2/2` byte/hash verified\n"
        "- Fresh independent environment acceptance: `done`\n"
        "- Implementation authorized: `false`\n\n"
        "The independent review accepted both environment checklist items while preserving all "
        "frozen locks. No RTL or downstream workload was run.\n",
        encoding="utf-8",
    )
    update_public(payload["reviewed_at_utc"])
    verdict = {
        "schema_version": 1,
        "project": "ACE-2",
        "stage": "environment",
        "contract_id": CONTRACT,
        "reviewer_role": "independent_l2_environment",
        "reviewer_status": "done",
        "status": "accepted",
        "verdict": "done",
        "stage_closing": True,
        "implementation_authorized": False,
        "required_remediation": "",
        "checklist": ENVIRONMENT_CHECKLIST,
        "decision": artifact(OUT),
        "current_audit": artifact(AUDIT),
        "reviewed_audit": reviewed_audit,
        "architecture_decision": artifact(ARCH_REVIEW),
        "manager_recommendation": "advance_environment_to_rtl_in_order",
    }
    dump(VERDICT, verdict)
    CERT.write_text(
        "# ACE-2 Independent Environment Certification\n\n"
        f"- Contract: `{CONTRACT}`\n"
        "- Verdict: `done / accepted`\n"
        "- Environment checklist: `2/2`\n"
        "- Audit mode: `successor-specific hash-only`\n"
        "- Retained capability evidence: `28/28` hash-valid, not rerun\n"
        "- Tool/IP selection evidence: `2/2` hash-valid\n"
        "- Implementation authorized: `false`\n"
        "- Manager-owned stage remains: `environment`\n\n"
        "No RTL, verification, synthesis/PPA, physical, prototype, benchmark, signoff, GDS, "
        "tapeout, or silicon workload was run for this certification.\n",
        encoding="utf-8",
    )


def check_existing() -> None:
    marker = run_check([sys.executable, "tools/bind_cross_layer_error_carry_environment.py", "--check"])
    require("review=done" in marker, "post-review environment binder status is not done")
    pipeline = load(PIPELINE)
    audit = load(AUDIT)
    decision = load(OUT)
    public = load(PUBLIC)
    architecture = load(ARCH_REVIEW)
    require(pipeline.get("current_stage") == "environment", "Manager-owned stage changed")
    successor = pipeline.get("successor", {})
    require(successor.get("state") == "frozen_unimplemented", "successor implementation state changed")
    require(all(value == "unrun" for value in successor.get("downstream_runs", {}).values()), "a successor downstream run is no longer unrun")
    require(pipeline.get("stages", {}).get("rtl", {}).get("status") == "locked_pending_predecessor_stage_acceptance", "RTL stage is unlocked")
    require(decision.get("contract_id") == CONTRACT, "environment decision contract differs")
    require(decision.get("reviewer_status") == "done", "environment reviewer is not done")
    require(decision.get("status") == "accepted", "environment decision is not accepted")
    require(decision.get("environment_accepted") is True, "environment acceptance flag differs")
    require(decision.get("required_remediation") == "", "environment remediation is not empty")
    require(decision.get("implementation_authorized") is False, "environment decision authorizes implementation")
    require(decision.get("stage_closing") is True, "environment decision is not stage-closing")
    require(decision.get("environment_checklist") == ENVIRONMENT_CHECKLIST, "environment checklist differs")
    require(decision.get("checklist") == REVIEW_CHECKLIST, "review checklist differs")
    require(decision.get("architecture_decision") == artifact(ARCH_REVIEW), "architecture decision binding stale")
    require(decision.get("architecture_packet") == artifact(PACKET), "architecture packet binding stale")
    require(decision.get("architecture_freeze") == artifact(FREEZE), "architecture freeze binding stale")
    reviewed = decision.get("reviewed_audit", {})
    verify_record(reviewed, "reviewed_audit")
    reviewed_payload = load(ROOT / reviewed["path"])
    require(reviewed_payload.get("integrity", {}).get("canonical_sha256") == canonical_sha256(reviewed_payload), "reviewed audit canonical digest stale")
    require(reviewed_payload.get("contract_binding", {}).get("environment_review_status") == "pending_fresh_independent_reviewer", "reviewed audit was not review-pending")
    semantic_contract = verify_successor_semantics(reviewed_payload)
    require(decision.get("selection_document_semantic_contract") == semantic_contract, "review decision semantic contract differs")
    require(audit.get("integrity", {}).get("canonical_sha256") == canonical_sha256(audit), "live audit digest stale")
    require(audit.get("contract_binding", {}).get("environment_review_status") == "done", "live audit review status differs")
    require(audit.get("independent_review", {}).get("decision") == artifact(OUT), "live audit decision binding stale")
    require(public.get("routing_status") == "manager_stage_advance_pending", "public routing differs")
    require(public.get("selected_replacement_contract", {}).get("environment_review_status") == "done", "public environment review status differs")
    require(public.get("selected_replacement_contract", {}).get("implementation_authorized") is False, "public authorizes implementation")
    require(projection_hashes(public) == architecture.get("review_binding", {}).get("public_projection_hashes"), "public architecture projections changed")
    require(load(VERDICT).get("decision") == artifact(OUT), "canonical verdict decision binding stale")
    require(CONTRACT in CERT.read_text(encoding="utf-8"), "environment certification contract missing")
    print(
        "ACE2_CROSS_LAYER_ERROR_CARRY_ENVIRONMENT_REVIEW_CHECK_PASS "
        f"contract={CONTRACT} reviewer=done checklist=2/2 retained=28 selections=2 "
        "implementation_authorized=false downstream_runs=unrun"
    )


def main() -> None:
    if "--check" in sys.argv[1:]:
        check_existing()
        return
    if "--publish-existing" in sys.argv[1:]:
        payload = load(OUT)
        require(payload.get("status") == "accepted", "existing environment decision is not accepted")
        publish(payload, payload["reviewed_audit"])
        check_existing()
        return

    verified = preflight()
    reviewed_audit = archive_reviewed_audit()
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
    package_root = Path(argus_skill.__file__).resolve().parent
    reviewer_skill = (
        package_root / "verticals/chip_design/skills/reviewer/chip-design-signoff-review.md"
    ).read_text(encoding="utf-8")
    objective = (
        "Independently decide whether the exact-successor hash-only environment audit for "
        f"{CONTRACT} satisfies both environment gates without authorizing implementation."
    )
    instruction = (
        "Inspect design/CHIP_SCOPE.json first, then only research/ENVIRONMENT_AUDIT.json, "
        "research/TOOLCHAIN_CANDIDATES.md, research/IP_REUSE_PLAN.md, the accepted architecture "
        "packet/freeze/decision, research/PIPELINE_STATE.json, and research/PUBLIC_STATUS.json. "
        "You may run only `python3 tools/bind_cross_layer_error_carry_environment.py --check` "
        "and short read-only Python hash/content checks. Verify that both maintained selection "
        "documents bind signed16_q0_15 carry, signed24_q8_15 reconstruction, unsigned56 square "
        "sum, the 88,512-byte candidate allocation, 466,112-byte peak, 58,176-byte margin, and "
        "the shared serial-engine/reused-IP model. Do not edit any file. Do not run RTL, "
        "simulation, formal, synthesis, PPA, physical, prototype, benchmark, signoff, datasets, "
        "or any downstream workload. Return done only if 28 retained capability artifacts and "
        "both maintained selection documents byte/hash verify, the successor adds no new EDA/PDK/"
        "license/compiler/runtime/external-IP dependency class, the audit is genuinely hash-only, "
        "all frozen ADVANCE/prefix/frontier/area/frequency/128-bit/historical-PPA/sealed-family "
        "locks hold, and implementation remains unauthorized. This review may close environment "
        "evidence only; Manager alone controls any environment-to-RTL transition."
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
            "The Manager advanced only to environment after repaired architecture acceptance. "
            "A successor-specific audit now hash-verifies 28 retained capability artifacts and "
            "two maintained tool/IP documents without rerunning a probe or downstream workload."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="environment_hash_only_acceptance_stage_closing_only",
        checkpoint_path=str(CHECKPOINT),
        background_context=(
            "The prior environment audit was predecessor-bound. Its raw capability evidence is "
            "retained only by hash; the new audit binds the accepted error-carry successor."
        ),
        escalate_hint=(
            "If accepted, recommend only that Manager may later advance environment to RTL in "
            "order. Do not authorize implementation in this review."
        ),
        preselected_skill_block=reviewer_skill,
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort(
                "ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"
            ),
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
    accepted = status == "done"
    reviewed_at = utc_now()
    architecture = load(ARCH_REVIEW)
    archived_prior_decision = archive_prior_decision()
    payload = {
        "schema_version": 1,
        "reviewed_at_utc": reviewed_at,
        "reviewer_role": "independent_l2_environment",
        "reviewer_status": status,
        "contract_id": CONTRACT,
        "stage": "environment",
        "scope": "successor_specific_hash_only_environment_acceptance",
        "status": "accepted" if accepted else "changes_required",
        "environment_accepted": accepted,
        "stage_closing": accepted,
        "implementation_authorized": False,
        "required_remediation": "" if accepted else (raw.get("next_action") or raw.get("reason") or "review did not pass"),
        "environment_checklist": ENVIRONMENT_CHECKLIST if accepted else {key: False for key in ENVIRONMENT_CHECKLIST},
        "checklist": REVIEW_CHECKLIST if accepted else {key: False for key in REVIEW_CHECKLIST},
        "reviewed_audit": reviewed_audit,
        "reviewed_audit_canonical_sha256": load(ROOT / reviewed_audit["path"])["integrity"]["canonical_sha256"],
        "architecture_packet": artifact(PACKET),
        "architecture_freeze": artifact(FREEZE),
        "architecture_decision": artifact(ARCH_REVIEW),
        "sealed_family_bindings": architecture["sealed_family_bindings"],
        "preserved_contracts": verified["preserved_contracts"],
        "public_projection_hashes": verified["public_projection_hashes"],
        "retained_capability_artifact_count": 28,
        "tool_ip_selection_document_count": 2,
        "selection_document_semantic_contract": verified["selection_document_semantic_contract"],
        "audit_mode": "successor_specific_hash_only_no_probe_or_workload_rerun",
        "forbidden_downstream_work_respected": True,
        "binder": artifact(BINDER),
        "archived_prior_decision": archived_prior_decision,
        "reason": raw.get("reason", ""),
        "next_required_gate": "Manager-owned environment-to-RTL transition; implementation remains unauthorized",
        "reviewer_thread_id": raw.get("thread_id", ""),
        "raw_decision": raw,
    }
    if not accepted:
        dump(OUT, payload)
        raise RuntimeError(payload["required_remediation"])
    publish(payload, reviewed_audit)
    check_existing()


if __name__ == "__main__":
    main()
