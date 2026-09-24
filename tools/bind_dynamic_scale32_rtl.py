#!/usr/bin/env python3
"""Bind the bounded dynamic Scale32 RTL preflight without advancing stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_token_group_dynamic_scale32_v1"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
EXPECTED_HASHES = {
    "evidence/shared_token_group_dynamic_scale32_v1/architecture/PROPOSAL.json":
        "b55ab977574dc2bdeb760e8859ec2fa49f9f835846e3da24bd0f887d84a74f41",
    "evidence/shared_token_group_dynamic_scale32_v1/architecture/MANAGER_FREEZE.json":
        "175d1dcc7016df0c94fb6ec0d086d78fe6b0e4b80f26d1ef42f746825b53c2d8",
    "evidence/review/architecture_shared_token_group_dynamic_scale32_v1/decision.json":
        "386a2951ea1f39e3ef51978ced42be72dea8e10c53be188286abbe1507124297",
    "evidence/review/environment_stage_closing_shared_token_group_dynamic_scale32_v1/decision.json":
        "910efb2fd11e53f4404905b7f6e8e0a528986b873bad15dfe8da5ec522907eff",
}
SOURCE_PATHS = [
    "rtl/ace2_dynamic_scale32_core.sv",
    "tools/ace2_dynamic_scale32_reference.py",
    "tools/gen_dynamic_scale32_vectors.py",
    "verification/generated/dynamic_scale32_vectors.json",
    "verification/generated/dynamic_scale32_vectors.svh",
    "verification/test_dynamic_scale32.py",
    "verification/tb/ace2_dynamic_scale32_tb.sv",
    "formal/ace2_dynamic_scale32_formal.sv",
    "formal/ace2_dynamic_scale32_formal.ys",
    "tools/run_dynamic_scale32_formal.py",
    "Makefile",
    "design/DYNAMIC_SCALE32_IP_PROVENANCE.json",
]
LOG_MARKERS = {
    "evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/vector_generation.log":
        "ACE2_DYNAMIC_SCALE32_VECTOR_GENERATION_PASS cases=5",
    "evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/reference_unittest.log":
        "Ran 5 tests",
    "evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/rtl_simulation.log":
        "ACE2_DYNAMIC_SCALE32_RTL_PASS groups=4 sidecar=pass accumulator=pass counter_bound=pass stalls=pass errors=pass",
    "evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/minimal_formal.log":
        "ACE2_DYNAMIC_SCALE32_MINIMAL_FORMAL_PASS",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def aggregate_hash(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode())
        digest.update(b"\0")
        digest.update(entry["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_inputs() -> tuple[list[dict[str, Any]], str]:
    state = json.loads((ROOT / "research/PIPELINE_STATE.json").read_text(encoding="utf-8"))
    require(state.get("current_stage") == "rtl", "Manager-owned current_stage is not rtl")
    last = state.get("stage_history", [])[-1]
    require(last.get("from_stage") == "environment" and last.get("to_stage") == "rtl",
            "latest Manager transition is not environment->rtl")
    require(last.get("by") == "manager", "latest rtl transition is not Manager-owned")
    require(last.get("at") == "2026-08-02T07:08:41.858846Z",
            "unexpected Manager rtl-entry transition timestamp")
    require("RTL implementation may now begin" in last.get("reason", ""),
            "Manager transition does not authorize bounded RTL implementation")

    for relative, expected in EXPECTED_HASHES.items():
        require((ROOT / relative).is_file(), f"missing frozen dependency: {relative}")
        require(sha256(ROOT / relative) == expected, f"frozen dependency changed: {relative}")

    for relative, marker in LOG_MARKERS.items():
        path = ROOT / relative
        require(path.is_file(), f"missing RTL preflight log: {relative}")
        require(marker in path.read_text(encoding="utf-8"), f"RTL preflight marker missing: {relative}")
    lint = ROOT / "evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/verilator_lint.log"
    require(lint.is_file(), "missing Verilator lint log")
    require(lint.read_text(encoding="utf-8") == "", "Verilator lint log is not warning-free")
    iverilog = ROOT / "evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/iverilog.log"
    require(iverilog.is_file(), "missing Icarus elaboration log")
    require("error" not in iverilog.read_text(encoding="utf-8").lower(),
            "Icarus elaboration log contains an error")

    vectors = json.loads((ROOT / "verification/generated/dynamic_scale32_vectors.json").read_text(encoding="utf-8"))
    require(vectors.get("contract_id") == CONTRACT, "generated vectors bind the wrong contract")
    require(vectors.get("numeric_acceptance") == "bit_exact_fixed_point_vector_match",
            "generated vectors do not declare bit-exact acceptance")
    require(len(vectors.get("cases", [])) == 5, "unexpected dynamic Scale32 vector count")

    provenance = json.loads((ROOT / "design/DYNAMIC_SCALE32_IP_PROVENANCE.json").read_text(encoding="utf-8"))
    require(provenance.get("contract_id") == CONTRACT, "IP provenance binds the wrong contract")
    require(provenance.get("third_party_ip") == [], "dynamic Scale32 packet unexpectedly uses third-party IP")
    policy = provenance.get("license_policy", {})
    require(policy.get("status") == "first_party_operator_directed_project_internal_work_product",
            "first-party license provenance status is missing")
    require(policy.get("external_redistribution_grant_claimed") is False,
            "IP provenance overclaims external redistribution rights")
    declared_paths = {entry.get("path") for entry in provenance.get("sources", [])}
    require(set(SOURCE_PATHS) - {"design/DYNAMIC_SCALE32_IP_PROVENANCE.json"} <= declared_paths,
            "one or more ordered sources lack explicit IP provenance")

    rtl = (ROOT / "rtl/ace2_dynamic_scale32_core.sv").read_text(encoding="utf-8")
    for module in (
        "ace2_dynamic_scale32_group_core",
        "ace2_dynamic_scale32_sidecar_builder_core",
        "ace2_dynamic_scale32_sidecar_validator_core",
        "ace2_scale32_tagged_accumulator_core",
    ):
        require(rtl.count(f"module {module}") == 1, f"missing or duplicate module {module}")
    require("value_mem_q [0:1][0:MAX_GROUP_LANES-1]" in rtl,
            "frozen ping-pong group storage is missing")
    require("parameter integer VALUE_WIDTH = 40" in rtl, "wide group storage width changed")
    require("parameter integer MAX_EVENTS = 38" in rtl, "tagged event bound changed")
    require("canonical_exponent_s8_o = -8'sd78" in rtl, "canonical tagged exponent changed")
    require("$real" not in rtl and "shortreal" not in rtl and "always #" not in rtl,
            "synthesizable source contains a prohibited simulation/floating construct")

    sources = []
    generated_paths = {
        "verification/generated/dynamic_scale32_vectors.json",
        "verification/generated/dynamic_scale32_vectors.svh",
    }
    for path in SOURCE_PATHS:
        record = artifact(path)
        record.update({
            "origin": "generated_from_first_party_reference_and_generator" if path in generated_paths
                      else "first_party_operator_directed_active_worktree",
            "license_provenance": "design/DYNAMIC_SCALE32_IP_PROVENANCE.json",
            "license_status": "project_internal_first_party_no_external_redistribution_claim",
            "third_party": False,
        })
        sources.append(record)
    return sources, aggregate_hash(sources)


def update_manifest(
    sources: list[dict[str, Any]], candidate_hash: str, precheck: dict[str, Any], timestamp: str
) -> None:
    path = ROOT / "design/RTL_MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    candidate_id = f"dynamic_scale32_{candidate_hash[:16]}"
    source_map = {entry["path"]: entry["sha256"] for entry in sources}
    modules = [
        {
            "name": "ace2_dynamic_scale32_group_core",
            "parameters": {"VALUE_WIDTH": 40, "LANES_PER_BEAT": 16, "MAX_GROUP_LANES": 128},
            "requirement": "64/128-lane smallest-legal-delta RNE finalizer with exact 1,280-byte ping-pong storage",
        },
        {
            "name": "ace2_dynamic_scale32_sidecar_builder_core",
            "parameters": {},
            "requirement": "exact 64-byte BFP1 sidecar construction and pre-publication exponent/address validation",
        },
        {
            "name": "ace2_dynamic_scale32_sidecar_validator_core",
            "parameters": {"MAX_GROUPS": 38},
            "requirement": "consumer-side magic/schema/shape/identity/exponent/reserved-byte validation before payload issue",
        },
        {
            "name": "ace2_scale32_tagged_accumulator_core",
            "parameters": {"ACC_WIDTH": 160, "MAX_EVENTS": 38},
            "requirement": "one shared exact Scale32-product aligner/accumulator at canonical exponent -78 with no intermediate rounding",
        },
    ]
    generated = [
        {
            **artifact("verification/generated/dynamic_scale32_vectors.json"),
            "kind": "generated_verification_source",
            "generator": "tools/gen_dynamic_scale32_vectors.py",
            "reference": "tools/ace2_dynamic_scale32_reference.py",
            "regeneration_command": "python tools/gen_dynamic_scale32_vectors.py",
            "third_party": False,
        },
        {
            **artifact("verification/generated/dynamic_scale32_vectors.svh"),
            "kind": "generated_verification_include",
            "generator": "tools/gen_dynamic_scale32_vectors.py",
            "reference": "tools/ace2_dynamic_scale32_reference.py",
            "regeneration_command": "python tools/gen_dynamic_scale32_vectors.py",
            "third_party": False,
        },
    ]
    precheck_artifact = artifact("evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/PRECHECK.json")

    manifest.update({
        "architecture_contract_status": "manager_frozen_environment_certified_and_manager_advanced_to_rtl",
        "candidate_capability_accepted": False,
        "candidate_first_unsupported_layer_operator_after_review": FIRST_UNSUPPORTED,
        "candidate_generated_hashes": {
            entry["path"]: entry["sha256"] for entry in generated
        },
        "candidate_generated_sources": generated,
        "candidate_geometry": {
            "wide_group_ping_pong_bytes": 1280,
            "tensor_sidecar_live_bytes": 832,
            "runtime_state_bytes": 256,
            "proposal_incremental_sram_bytes": 2368,
            "planned_sram_peak_bytes": 379968,
            "remaining_sram_bytes": 144320,
        },
        "candidate_id": candidate_id,
        "candidate_implementation_provenance": {
            "role": "planner_direct_executor",
            "agent_layer": "planner",
            "execution_contract": "operator_planner_direct_execution_contract",
            "implementation_source_aggregate_sha256": candidate_hash,
            "implementation_source_hash_scope": "ordered_standalone_rtl_reference_generated_test_formal_and_makefile_not_shell_admitted",
            "status": "fresh_direct_implementation_from_manager_frozen_contract",
        },
        "candidate_interface": {
            "status": "standalone_interfaces_implemented_not_shell_admitted",
            "descriptor_or_csr_change": False,
            "planned_shell_flag_bit": 6,
            "sealed_qecr_flag_bit": 7,
            "modules": modules,
            "publication": "payload beats remain provisional until a stable commit record; shell admission remains unperformed",
        },
        "candidate_layer_operator": CONTRACT,
        "candidate_meets_numeric_acceptance": False,
        "candidate_requires_fresh_sky130_ppa": True,
        "candidate_rtl_hash": candidate_hash,
        "candidate_rtl_hash_scope": "ordered_standalone_rtl_reference_generated_test_formal_and_makefile_not_shell_admitted",
        "candidate_rtl_sources": [artifact("rtl/ace2_dynamic_scale32_core.sv")],
        "candidate_schedule": {
            "group_capture": "overlaps producer compute into one ping-pong bank",
            "selection_cycles_per_group": 1,
            "payload_emit_cycles_64_lane_group": 4,
            "payload_emit_cycles_128_lane_group": 8,
            "architecture_dynamic_finalize_cycles_per_layer": 1557,
            "status": "construct_matches_frozen_planning_hook_not_a_throughput_measurement",
        },
        "candidate_source_hashes": source_map,
        "candidate_source_provenance": sources,
        "candidate_status": "bounded_rtl_preflight_pass_independent_review_pending",
        "candidate_supported_layer_operator_prefix_after_review": PREFIX,
        "candidate_verification_binding": None,
        "candidate_verification_complete": False,
        "current_stage": "rtl",
        "generated_at_utc": timestamp,
        "implementation_authority_consumed": True,
        "implementation_authorized": False,
        "implementation_completed": True,
        "independent_reviewer_acceptance": {
            "architecture_review": "accepted",
            "environment_review": "accepted",
            "rtl_review": "pending_fresh_independent_checklist_review",
            "status": "pending",
        },
        "interfaces_contract_status": "dynamic_scale32_standalone_interfaces_implemented_review_pending",
        "latest_dynamic_scale32_rtl_evidence": {
            "precheck": precheck_artifact,
            "logs": {relative: artifact(relative) for relative in LOG_MARKERS},
            "verilator_lint": artifact("evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/verilator_lint.log"),
            "iverilog_elaboration": artifact("evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/iverilog.log"),
        },
        "proposed_replacement_contract": {
            "contract_id": CONTRACT,
            "candidate_id": candidate_id,
            "candidate_rtl_hash": candidate_hash,
            "mechanism": "token_and_tensor_group_dynamic_power_of_two_Scale32_activation_scaling",
            "ordered_supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "mode": "ADVANCE",
            "area_cap_non_sram_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "memory_boundary_bits": 128,
            "manager_transition_at": "2026-08-02T07:08:41.858846Z",
            "implementation_authority_consumed": True,
            "implementation_authorized": False,
            "implementation_completed": True,
            "rtl_started": True,
            "stage_closing": False,
            "status": "bounded_rtl_preflight_pass_independent_review_pending",
            "forbidden": [
                "baseline_or_candidate_model_execution_in_rtl",
                "full_shell_regression_before_separate_authority",
                "canonical_ppa_before_ppa_stage",
                "operator_target_relaxation",
            ],
        },
        "required_manager_action": "none_current_rtl_task_stage_closing_false",
        "stage": "rtl",
        "stage_closing": False,
        "status": "planner_rtl_preflight_pass_independent_review_pending_stage_closing_false",
        "traceability": {
            "selected_mechanism": CONTRACT,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "architecture_contract_gap": None,
            "rtl.contract-traceability": True,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
            "stage_checklist": {
                "rtl.contract-traceability": True,
                "rtl.hardware-discipline": True,
                "rtl.ip-provenance": True,
            },
        },
    })

    old_names = {module["name"] for module in modules}
    provenance = [
        entry for entry in manifest.get("ip_provenance", [])
        if entry.get("name") not in old_names and
           entry.get("name") not in {"dynamic_scale32_vectors_json", "dynamic_scale32_vectors_svh"}
    ]
    provenance.extend(
        {
            "name": module["name"],
            "kind": "first_party_candidate_rtl",
            "path": "rtl/ace2_dynamic_scale32_core.sv",
            "sha256": source_map["rtl/ace2_dynamic_scale32_core.sv"],
            "source_revision": source_map["rtl/ace2_dynamic_scale32_core.sv"],
            "license": "first-party operator-directed project-internal work product; no external redistribution grant claimed",
            "license_provenance": "design/DYNAMIC_SCALE32_IP_PROVENANCE.json",
            "license_status": "not_a_third_party_dependency",
            "third_party": False,
        }
        for module in modules
    )
    provenance.extend(
        {
            "name": f"dynamic_scale32_vectors_{kind}",
            "kind": "generated_verification_source",
            "path": f"verification/generated/dynamic_scale32_vectors.{kind}",
            "sha256": source_map[f"verification/generated/dynamic_scale32_vectors.{kind}"],
            "generator": "tools/gen_dynamic_scale32_vectors.py",
            "reference": "tools/ace2_dynamic_scale32_reference.py",
            "regeneration_command": "python tools/gen_dynamic_scale32_vectors.py",
            "license": "generated from first-party project-internal reference and generator; no external redistribution grant claimed",
            "license_provenance": "design/DYNAMIC_SCALE32_IP_PROVENANCE.json",
            "license_status": "not_a_third_party_dependency",
            "third_party": False,
        }
        for kind in ("json", "svh")
    )
    manifest["ip_provenance"] = provenance
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_public_status(candidate_hash: str, timestamp: str) -> None:
    path = ROOT / "research/PUBLIC_STATUS.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    latest_decision = "shared_token_group_dynamic_scale32_v1_rtl_preflight_pass_independent_review_pending"
    precheck_artifact = artifact("evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/PRECHECK.json")
    manifest_artifact = artifact("design/RTL_MANIFEST.json")
    trace_artifact = artifact("design/RTL_TRACEABILITY.md")
    status.update({
        "current_stage": "rtl",
        "stage_closing": False,
        "current_mode": "ADVANCE",
        "latest_decision": latest_decision,
        "implementation_authority_consumed": True,
        "implementation_authorized": False,
        "implementation_completed": True,
        "candidate_rtl_hash": candidate_hash,
        "candidate_rtl_hash_scope": "ordered_standalone_rtl_reference_generated_test_formal_and_makefile_not_shell_admitted",
        "rtl_contract_traceability": True,
        "last_updated_utc": timestamp,
        "generated_at_utc": timestamp,
        "required_manager_action": "none_current_rtl_task_stage_closing_false",
        "routing_authorized": "rtl_stage_only_no_manager_transition_requested",
        "routing_status": "independent_rtl_checklist_review_pending",
        "blockers": [{
            "id": "fresh_independent_rtl_checklist_review_pending",
            "owner": "Reviewer",
            "severity": "gate",
            "status": "open",
            "detail": "Planner preflight passes all three RTL checklist items; independent review is still required before any later verification authorization.",
        }],
        "next_action_queue": [{
            "id": "independent_rtl_checklist_review_shared_token_group_dynamic_scale32_v1",
            "stage": "rtl",
            "scope": "evidence_only_independent_review_of_exact_rtl_manifest_traceability_and_preflight",
            "status": "pending_independent_review",
            "stage_closing": False,
            "implementation_authorized": False,
            "candidate_or_model_execution_authorized": False,
            "prohibited_work": [
                "disabled_mode_baseline_execution",
                "candidate_quality_execution",
                "full_shell_regression",
                "canonical_sky130_ppa",
                "prototype_benchmark_or_signoff",
            ],
        }],
        "latest_rtl_candidate": {
            "contract_id": CONTRACT,
            "candidate_rtl_hash": candidate_hash,
            "candidate_rtl_hash_scope": "ordered_standalone_rtl_reference_generated_test_formal_and_makefile_not_shell_admitted",
            "current_stage": "rtl",
            "implementation_authority_consumed": True,
            "implementation_authorized": False,
            "implementation_completed": True,
            "candidate_capability_accepted": False,
            "candidate_or_model_execution_count": 0,
            "status": "bounded_rtl_preflight_pass_independent_review_pending",
            "modules": [
                "ace2_dynamic_scale32_group_core",
                "ace2_dynamic_scale32_sidecar_builder_core",
                "ace2_dynamic_scale32_sidecar_validator_core",
                "ace2_scale32_tagged_accumulator_core",
            ],
            "rtl_allowed_execution_counts": {
                "reference_vector_generation": 1,
                "reference_unit_test_suite": 1,
                "rtl_bit_exact_simulation": 1,
                "verilator_lint_tops": 4,
                "minimal_formal": 1,
            },
            "downstream_execution_counts": {
                "baseline_model": 0,
                "candidate_model": 0,
                "shell_regression": 0,
                "synthesis": 0,
                "ppa": 0,
                "physical": 0,
                "prototype": 0,
                "benchmark": 0,
                "signoff": 0,
            },
            "evidence": {
                "precheck": precheck_artifact,
                "manifest": manifest_artifact,
                "traceability": trace_artifact,
            },
        },
    })
    status["stage"] = {
        "checklist_scope": CONTRACT,
        "completed_prior_stages": ["definition", "architecture", "environment"],
        "current_stage": "rtl",
        "current_stage_checklist": {
            "rtl.contract-traceability": True,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
        },
        "current_stage_status": "planner_preflight_pass_independent_review_pending",
        "current_stage_evidence": [
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
            "evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/PRECHECK.json",
        ],
        "downstream_stages_locked_until_manager_advance": [
            "verification", "ppa", "prototype", "benchmark", "signoff"
        ],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
        "stage_closing": False,
    }
    dashboard = status.setdefault("dashboard_fields", {})
    dashboard.update({
        "supported_layers": PREFIX,
        "ordered_supported_layer_operator_prefix": PREFIX,
        "first_unsupported_layer": FIRST_UNSUPPORTED,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "current_mode": "ADVANCE",
        "latest_decision": latest_decision,
        "ppa_frontier": status.get("latest_ppa_frontier"),
        "rtl_candidate": {
            "contract_id": CONTRACT,
            "rtl_hash": candidate_hash,
            "status": "bounded_rtl_preflight_pass_independent_review_pending",
            "candidate_capability_accepted": False,
        },
    })
    status["public_claims"] = [
        {
            "claim": "the Manager-owned current stage is rtl and bounded RTL implementation was authorized on August 2, 2026",
            "evidence": ["research/PIPELINE_STATE.json"],
        },
        {
            "claim": "the standalone dynamic Scale32 primitives pass deterministic reference, lint, elaboration, bit-exact simulation, and minimal formal preflight",
            "evidence": [precheck_artifact["path"], manifest_artifact["path"], trace_artifact["path"]],
        },
        {
            "claim": "the supported prefix, first unsupported operator, ADVANCE mode, immutable targets, and historical PPA frontier are unchanged",
            "evidence": ["design/FAST_LOOP_POLICY.json", "research/PUBLIC_STATUS.json"],
        },
    ]
    nonclaims = set(status.get("explicit_non_claims", []))
    nonclaims.update({
        "no disabled-mode baseline or candidate model execution is claimed for the dynamic Scale32 successor",
        "no shell admission, full shell regression, candidate synthesis/PPA, physical design, prototype, benchmark, signoff, tapeout-readiness, or silicon result is claimed",
    })
    status["explicit_non_claims"] = sorted(nonclaims)
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_checkpoint(candidate_hash: str, timestamp: str) -> None:
    content = f"""# Goal

Implement and bind the exact Manager-frozen `{CONTRACT}` standalone RTL primitives while holding the project in `rtl` with `stage_closing=false`.

# Current state

The Manager advanced `environment -> rtl` at `2026-08-02T07:08:41.858846Z`, authorizing bounded RTL implementation only. The Planner implemented the 64/128-lane dynamic group finalizer with the exact 1,280-byte ping-pong allocation, the 64-byte `BFP1` sidecar builder and validator, and the signed-160 exact Scale32-tagged accumulator. The ordered implementation hash is `{candidate_hash}`.

All project-native RTL preflight checks pass: five scalar reference tests, five deterministic vector families, Icarus elaboration and bit-exact simulation, warning-free Verilator lint for four tops, and a depth-4 Yosys SAT proof. The three RTL checklist items are Planner-evidenced true; fresh independent RTL checklist review remains pending.

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains `layer_0.rope_q`; mode remains `ADVANCE`. The 2.0 mm2 non-SRAM cap, >=100 MHz floor, 128-bit abstract streaming-memory boundary, and historical PPA frontier are unchanged.

# Evidence

- `design/RTL_MANIFEST.json`
- `design/RTL_TRACEABILITY.md`
- `evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/PRECHECK.json`
- `evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/rtl_simulation.log`
- `evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/verilator_lint.log`
- `evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/minimal_formal.log`

# Boundary

Independent RTL checklist review is the remaining current-stage gate. No Manager stage transition is requested because this bounded task is `stage_closing=false`. No baseline/model run, shell admission/full regression, canonical PPA, physical design, prototype, benchmark, signoff, tapeout-readiness, or silicon work was performed. Updated at `{timestamp}`.
"""
    (ROOT / "CHECKPOINT.md").write_text(content, encoding="utf-8")


def bind() -> tuple[str, dict[str, Any]]:
    sources, candidate_hash = validate_inputs()
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    precheck = {
        "schema_version": 1,
        "project": "ACE-2",
        "contract_id": CONTRACT,
        "candidate_rtl_hash": candidate_hash,
        "candidate_rtl_hash_scope": "ordered_standalone_rtl_reference_generated_test_formal_and_makefile_not_shell_admitted",
        "bound_at_utc": timestamp,
        "current_stage": "rtl",
        "stage_closing": False,
        "implementation_authority": {
            "manager_transition_at": "2026-08-02T07:08:41.858846Z",
            "bounded_rtl_implementation_authorized": True,
            "implementation_completed": True,
            "implementation_authority_consumed": True,
            "downstream_authorized": False,
        },
        "frozen_dependencies": {relative: artifact(relative) for relative in EXPECTED_HASHES},
        "source_artifacts": sources,
        "modules": [
            "ace2_dynamic_scale32_group_core",
            "ace2_dynamic_scale32_sidecar_builder_core",
            "ace2_dynamic_scale32_sidecar_validator_core",
            "ace2_scale32_tagged_accumulator_core",
        ],
        "checks": {
            "reference_vector_generation": "PASS_5_cases",
            "reference_unit_tests": "PASS_5_tests",
            "iverilog_elaboration": "PASS",
            "bit_exact_simulation": "PASS_groups_sidecar_accumulator_stalls_errors",
            "verilator_lint": "PASS_4_tops_warning_free",
            "minimal_formal": "PASS_depth_4",
        },
        "stage_checklist": {
            "rtl.contract-traceability": True,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
        },
        "frontier": {
            "ordered_supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "mode": "ADVANCE",
            "candidate_capability_accepted": False,
            "new_ppa_run": False,
            "historical_ppa_preserved": True,
        },
        "independent_review": "pending",
        "claim_boundary": "Standalone RTL-stage preflight only; no model execution, shell admission, full regression, synthesis/PPA, physical, prototype, benchmark, signoff, tapeout, or silicon claim.",
        "status": "bounded_rtl_preflight_pass_independent_review_pending",
    }
    latest = ROOT / "evidence/shared_token_group_dynamic_scale32_v1/rtl/latest"
    latest.mkdir(parents=True, exist_ok=True)
    precheck_path = latest / "PRECHECK.json"
    precheck_path.write_text(json.dumps(precheck, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (latest / "PRECHECK.sha256").write_text(f"{sha256(precheck_path)}  PRECHECK.json\n", encoding="utf-8")
    update_manifest(sources, candidate_hash, precheck, timestamp)
    update_public_status(candidate_hash, timestamp)
    update_checkpoint(candidate_hash, timestamp)
    return candidate_hash, precheck


def check() -> str:
    _, candidate_hash = validate_inputs()
    manifest = json.loads((ROOT / "design/RTL_MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest.get("candidate_rtl_hash") == candidate_hash, "RTL manifest candidate hash is stale")
    require(manifest.get("candidate_layer_operator") == CONTRACT, "RTL manifest contract is stale")
    require(manifest.get("traceability", {}).get("stage_checklist") == {
        "rtl.contract-traceability": True,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
    }, "RTL manifest checklist is not closed")
    require(manifest.get("stage_closing") is False, "RTL task must remain stage_closing=false")
    status = json.loads((ROOT / "research/PUBLIC_STATUS.json").read_text(encoding="utf-8"))
    require(status.get("current_stage") == "rtl", "public current stage is stale")
    require(status.get("candidate_rtl_hash") == candidate_hash, "public candidate hash is stale")
    require(status.get("first_unsupported_layer_operator") == FIRST_UNSUPPORTED,
            "public first unsupported operator changed")
    require(status.get("ordered_supported_layer_operator_prefix") == PREFIX,
            "public supported prefix changed")
    return candidate_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        candidate_hash = check()
        print(f"ACE2_DYNAMIC_SCALE32_RTL_BINDING_CHECK_PASS candidate_rtl_hash={candidate_hash}")
    else:
        candidate_hash, _ = bind()
        print(f"ACE2_DYNAMIC_SCALE32_RTL_BINDING_PASS candidate_rtl_hash={candidate_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
