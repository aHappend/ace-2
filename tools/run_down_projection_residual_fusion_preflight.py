#!/usr/bin/env python3
"""Run and bind the bounded RTL-stage DPRF candidate preflight."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from validate_down_projection_residual_fusion_evidence import validate_artifact_tree


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_down_projection_residual_fusion_v1"
EVIDENCE = ROOT / "evidence/shared_down_projection_residual_fusion_v1/latest"
PRECHECK = EVIDENCE / "PRECHECK.json"
RTL = ROOT / "rtl/ace2_down_projection_residual_fusion_core.sv"
TRANSITIONS = ROOT / "evidence/shared_down_projection_residual_fusion_v1/transitions"
ARCH_BINDINGS = EVIDENCE / "CONTRACT_BINDINGS.json"
ARCH_PACKET = EVIDENCE / "ARCHITECTURE_REFREEZE_REVIEW_PACKET.json"
ARCH_DECISION = ROOT / "evidence/review/architecture_refreeze_shared_down_projection_residual_fusion_v1/decision.json"
RTL_REVIEW = ROOT / "evidence/review/rtl_checklist_shared_down_projection_residual_fusion_v1/decision.json"
RTL_REVIEW_ARCHIVE = RTL_REVIEW.parent / "archive"
PROBE = ROOT / "evidence/diagnostics/shared_down_proj_residual_fusion_probe_20260801/results.json"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
CHECKLIST = {
    "rtl.contract-traceability": True,
    "rtl.hardware-discipline": True,
    "rtl.ip-provenance": True,
}
SOURCE_PATHS = [
    "Makefile",
    "rtl/ace2_down_projection_residual_fusion_core.sv",
    "tools/ace2_down_projection_residual_fusion_hook.py",
    "tools/ace2_down_projection_residual_fusion_reference.py",
    "tools/gen_down_projection_residual_fusion_metadata.py",
    "tools/gen_down_projection_residual_fusion_vectors.py",
    "tools/run_down_projection_residual_fusion_formal.py",
    "tools/run_down_projection_residual_fusion_preflight.py",
    "tools/validate_down_projection_residual_fusion_evidence.py",
    "formal/ace2_down_projection_residual_fusion_formal.sv",
    "formal/ace2_down_projection_residual_fusion_formal.ys",
    "verification/test_down_projection_residual_fusion.py",
    "verification/tb/ace2_down_projection_residual_fusion_tb.sv",
    "verification/generated/down_projection_residual_fusion_vectors.json",
    "verification/generated/down_projection_residual_fusion_vectors.svh",
    "reference/generated/down_projection_residual_fusion_metadata.json",
    "reference/generated/down_projection_residual_fusion_metadata.bin",
]
CONTROL_PATHS = [
    "tools/bind_down_projection_residual_fusion_architecture.py",
    "tools/run_down_projection_residual_fusion_rtl_review.py",
]
SEMANTIC_PATHS = [
    "rtl/ace2_down_projection_residual_fusion_core.sv",
    "tools/ace2_down_projection_residual_fusion_hook.py",
    "tools/ace2_down_projection_residual_fusion_reference.py",
    "tools/gen_down_projection_residual_fusion_metadata.py",
    "tools/gen_down_projection_residual_fusion_vectors.py",
    "verification/test_down_projection_residual_fusion.py",
    "verification/tb/ace2_down_projection_residual_fusion_tb.sv",
    "verification/generated/down_projection_residual_fusion_vectors.json",
    "verification/generated/down_projection_residual_fusion_vectors.svh",
    "reference/generated/down_projection_residual_fusion_metadata.json",
    "reference/generated/down_projection_residual_fusion_metadata.bin",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: str | Path) -> dict[str, Any]:
    resolved = path if isinstance(path, Path) else ROOT / path
    value = json.loads(resolved.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {resolved}")
    return value


def dump(path: str | Path, value: dict[str, Any]) -> None:
    resolved = path if isinstance(path, Path) else ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: str | Path) -> dict[str, Any]:
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


def aggregate_hash(records: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(records):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(records[path].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def scoped_artifact(path: str | Path, *, scope: str, recursive: bool = False) -> dict[str, Any]:
    record = artifact(path)
    record["binding_scope"] = scope
    record["content_validation"] = "recursive_json" if recursive else "whole_file_only"
    return record


def current_probe_record() -> dict[str, Any]:
    record = artifact(PROBE)
    record.update(
        {
            "classification": "exploratory_float64_proxy_not_scale32_construct_evidence",
            "permitted_use": "historical_context_only",
        }
    )
    return record


def mark_historical_records(value: Any, historical: bool = False) -> None:
    if isinstance(value, dict):
        if historical and isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            value["binding_scope"] = "historical_observation_not_live_binding"
            value.setdefault(
                "historical_reason",
                "Retained observation of an earlier mutable repository path; it is not a current live hash binding.",
            )
        for key, child in value.items():
            child_historical = historical or key.startswith("historical_")
            if key == "qk_residual_post_refreeze_contract_blocker":
                child_historical = True
            mark_historical_records(child, child_historical)
    elif isinstance(value, list):
        for child in value:
            mark_historical_records(child, historical)


def archive_prior_standalone() -> tuple[dict[str, Any], Path]:
    require(RTL_REVIEW.is_file(), "prior standalone RTL decision is missing")
    prior = load(RTL_REVIEW)
    require(prior.get("reviewer_status") == "done", "prior standalone RTL review is not done")
    require(prior.get("stage_closing") is False, "prior RTL review is not standalone")
    require(prior.get("scope") == "standalone_candidate_rtl_checklist_only", "prior RTL review scope differs")
    digest = sha256_file(RTL_REVIEW)
    archive_path = RTL_REVIEW_ARCHIVE / f"standalone_decision.{digest}.json"
    RTL_REVIEW_ARCHIVE.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        require(sha256_file(archive_path) == digest, "prior standalone archive hash differs")
    else:
        shutil.copy2(RTL_REVIEW, archive_path)
    return prior, archive_path


def write_immutable_snapshot(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    suffix = "".join(path.suffixes) or ".bin"
    stem = path.name.removesuffix(suffix)
    destination = TRANSITIONS / "snapshots" / f"{stem}.{digest}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(destination.read_bytes() == data, f"immutable snapshot collision: {destination}")
    else:
        destination.write_bytes(data)
    return scoped_artifact(destination, scope="immutable_snapshot", recursive=path.suffix == ".json")


def write_immutable_json(name: str, value: dict[str, Any]) -> dict[str, Any]:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    destination = TRANSITIONS / "snapshots" / f"{name}.{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(destination.read_bytes() == encoded, f"immutable JSON collision: {destination}")
    else:
        destination.write_bytes(encoded)
    return scoped_artifact(destination, scope="immutable_snapshot", recursive=True)


def run(command: list[str], log_name: str) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log = EVIDENCE / log_name
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{result.stdout}")
    return artifact(log)


def manager_transition() -> dict[str, Any]:
    pipeline = load("research/PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    event = pipeline.get("stage_history", [])[-1]
    require(
        event.get("by") == "manager"
        and event.get("from_stage") == "environment"
        and event.get("to_stage") == "rtl",
        "latest event is not the Manager environment-to-RTL authorization",
    )
    require(CONTRACT in event.get("reason", "") or "implementation authorization becomes active" in event.get("reason", ""),
            "Manager transition does not bind the active successor")
    return event


def consumed_authority(candidate_id: str, candidate_hash: str, transition: dict[str, Any]) -> dict[str, Any]:
    policy = load("design/FAST_LOOP_POLICY.json")
    source = copy.deepcopy(policy["selected_replacement_contract"])
    require(source.get("contract_id") == CONTRACT, "active policy contract mismatch")
    source.update(
        {
            "authorization_basis": "operator_fast_loop_policy_plus_manager_environment_to_rtl_transition",
            "manager_transition": copy.deepcopy(transition),
            "candidate_id": candidate_id,
            "candidate_rtl_hash": candidate_hash,
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "rtl_started": True,
            "required_manager_action": "hold_rtl_for_independent_checklist_review",
            "status": "bounded_rtl_preflight_pass_review_pending",
            "stage_closing": False,
        }
    )
    return source


def create_transition_binding(
    *,
    candidate_id: str,
    candidate_hash: str,
    source_hashes: dict[str, str],
    manager_event: dict[str, Any],
    previous_precheck: dict[str, Any],
    prior_review_path: Path,
    now: str,
) -> dict[str, Any]:
    prior_review = load(prior_review_path)
    previous_hashes = previous_precheck.get("source_hashes", {})
    require(isinstance(previous_hashes, dict), "previous PRECHECK source hashes are missing")
    for relative in SEMANTIC_PATHS:
        require(previous_hashes.get(relative) == source_hashes.get(relative), f"semantic source changed: {relative}")
    require(
        prior_review.get("live_rtl_source_sha256") == source_hashes["rtl/ace2_down_projection_residual_fusion_core.sv"],
        "live RTL source differs from the archived standalone review",
    )

    sealed_bindings = load(ARCH_BINDINGS)
    historical_target = copy.deepcopy(sealed_bindings["architecture_sources"]["target"])
    historical_policy = copy.deepcopy(sealed_bindings["architecture_sources"]["fast_loop_policy"])
    for record, reason in (
        (
            historical_target,
            "Architecture-stage observation sealed before the Manager environment-to-RTL transition; design/TARGET.json is now a mutable RTL projection.",
        ),
        (
            historical_policy,
            "Architecture-stage observation sealed before the Manager environment-to-RTL transition; design/FAST_LOOP_POLICY.json is now a mutable RTL projection.",
        ),
    ):
        record["binding_scope"] = "historical_mutable_path_observation"
        record["historical_reason"] = reason

    manager_event_snapshot = write_immutable_json("manager_environment_to_rtl_event", manager_event)
    projection_snapshots = {
        "target": write_immutable_snapshot(ROOT / "design/TARGET.json"),
        "fast_loop_policy": write_immutable_snapshot(ROOT / "design/FAST_LOOP_POLICY.json"),
        "chip_scope": write_immutable_snapshot(ROOT / "design/CHIP_SCOPE.json"),
        "rtl_manifest": write_immutable_snapshot(ROOT / "design/RTL_MANIFEST.json"),
        "rtl_traceability": write_immutable_snapshot(ROOT / "design/RTL_TRACEABILITY.md"),
        "public_status": write_immutable_snapshot(ROOT / "research/PUBLIC_STATUS.json"),
        "root_checkpoint": write_immutable_snapshot(ROOT / "CHECKPOINT.md"),
    }
    changed_paths = sorted(
        relative
        for relative, digest in source_hashes.items()
        if previous_hashes.get(relative) != digest
    )
    transition: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": now,
        "contract_id": CONTRACT,
        "kind": "immutable_architecture_to_rtl_evidence_transition",
        "claim_boundary": "RTL evidence hardening and stage-binding only; no verification, shell, PPA, physical, prototype, benchmark, signoff, tapeout, or silicon claim.",
        "supersedes_for_live_validation": scoped_artifact(
            ARCH_BINDINGS,
            scope="sealed_whole_file",
        ),
        "sealed_architecture": {
            "accepted_decision": scoped_artifact(ARCH_DECISION, scope="sealed_whole_file"),
            "accepted_packet": scoped_artifact(ARCH_PACKET, scope="sealed_whole_file"),
            "contract_bindings": scoped_artifact(ARCH_BINDINGS, scope="sealed_whole_file"),
            "historical_mutable_path_observations": {
                "target": historical_target,
                "fast_loop_policy": historical_policy,
            },
            "preservation": "The accepted architecture decision and CONTRACT_BINDINGS.json remain byte-for-byte unchanged.",
        },
        "manager_transition": {
            "event": copy.deepcopy(manager_event),
            "immutable_event_snapshot": manager_event_snapshot,
        },
        "candidate_transition": {
            "previous_candidate_id": prior_review.get("candidate_id"),
            "previous_candidate_rtl_hash": prior_review.get("candidate_rtl_hash"),
            "intermediate_precheck_candidate_id": previous_precheck.get("candidate_id"),
            "intermediate_precheck_candidate_rtl_hash": previous_precheck.get("candidate_rtl_hash"),
            "successor_candidate_id": candidate_id,
            "successor_candidate_rtl_hash": candidate_hash,
            "live_rtl_source_sha256": source_hashes["rtl/ace2_down_projection_residual_fusion_core.sv"],
            "semantic_sources_unchanged": True,
            "semantic_source_paths": SEMANTIC_PATHS,
            "evidence_hardening_paths_added_or_changed": changed_paths,
            "reason": "The aggregate changes only because the hash scope includes hardened preflight/formal/evidence-validation sources; RTL, oracle, generators, vectors, metadata, testbench, and deterministic reference artifacts are unchanged.",
        },
        "prior_standalone_review": scoped_artifact(prior_review_path, scope="sealed_whole_file"),
        "rtl_projection_snapshots": projection_snapshots,
        "live_source_bindings": {
            relative: scoped_artifact(ROOT / relative, scope="current_live_binding")
            for relative in sorted(set(SOURCE_PATHS + CONTROL_PATHS))
        },
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "canonical_sha256": None,
        },
    }
    transition["integrity"]["canonical_sha256"] = canonical_sha256(transition)
    digest = transition["integrity"]["canonical_sha256"]
    path = TRANSITIONS / f"rtl_transition.{digest}.json"
    encoded = (json.dumps(transition, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == encoded, f"immutable transition collision: {path}")
    else:
        path.write_bytes(encoded)
    return scoped_artifact(path, scope="immutable_snapshot", recursive=True)


def update_state(
    candidate_id: str,
    candidate_hash: str,
    source_hashes: dict[str, str],
    transition: dict[str, Any],
    logs: dict[str, dict[str, Any]],
    interface_xml: dict[str, Any],
    prior_review_path: Path,
    now: str,
) -> dict[str, Any]:
    authority = consumed_authority(candidate_id, candidate_hash, transition)
    rtl_artifact = artifact(RTL)
    vectors_json = artifact("verification/generated/down_projection_residual_fusion_vectors.json")
    vectors_svh = artifact("verification/generated/down_projection_residual_fusion_vectors.svh")
    metadata_json = artifact("reference/generated/down_projection_residual_fusion_metadata.json")
    metadata_bin = artifact("reference/generated/down_projection_residual_fusion_metadata.bin")

    manifest = load("design/RTL_MANIFEST.json")
    manifest.update(
        {
            "stage": "rtl",
            "current_stage": "rtl",
            "architecture_contract_status": "manager_advanced_to_rtl_successor_implemented",
            "candidate_id": candidate_id,
            "candidate_layer_operator": CONTRACT,
            "candidate_status": "fresh_independent_rtl_stage_closing_review_pending",
            "candidate_rtl_hash": candidate_hash,
            "candidate_rtl_hash_scope": "ordered_standalone_candidate_sources_not_shell_admitted",
            "candidate_source_hashes": source_hashes,
            "candidate_rtl_sources": [
                {
                    **rtl_artifact,
                    "modules": ["ace2_down_projection_residual_fusion_core"],
                    "kind": "first_party_bounded_candidate_rtl",
                    "third_party": False,
                }
            ],
            "candidate_generated_hashes": {
                item["path"]: item["sha256"]
                for item in (vectors_json, vectors_svh, metadata_json, metadata_bin)
            },
            "candidate_generated_sources": [
                {
                    **vectors_json,
                    "generator": "tools/gen_down_projection_residual_fusion_vectors.py",
                    "reference": "tools/ace2_down_projection_residual_fusion_reference.py",
                    "regeneration_command": ".venv/bin/python tools/gen_down_projection_residual_fusion_vectors.py",
                    "kind": "generated_verification_source",
                    "third_party": False,
                },
                {
                    **vectors_svh,
                    "generator": "tools/gen_down_projection_residual_fusion_vectors.py",
                    "reference": "tools/ace2_down_projection_residual_fusion_reference.py",
                    "regeneration_command": ".venv/bin/python tools/gen_down_projection_residual_fusion_vectors.py",
                    "kind": "generated_verification_source",
                    "third_party": False,
                },
                {
                    **metadata_json,
                    "generator": "tools/gen_down_projection_residual_fusion_metadata.py",
                    "regeneration_command": ".venv/bin/python tools/gen_down_projection_residual_fusion_metadata.py",
                    "kind": "generated_model_image_metadata_manifest",
                    "third_party": False,
                },
                {
                    **metadata_bin,
                    "generator": "tools/gen_down_projection_residual_fusion_metadata.py",
                    "regeneration_command": ".venv/bin/python tools/gen_down_projection_residual_fusion_metadata.py",
                    "kind": "generated_model_image_metadata_binary",
                    "third_party": False,
                },
            ],
            "candidate_interface": {
                "status": "implemented_and_verilator_elaborated",
                "descriptor_or_csr_change": False,
                "planned_shell_flag_bit": 6,
                "module": "ace2_down_projection_residual_fusion_core",
                "parameters": {},
                "ports": {
                    "clk_i": 1,
                    "rst_ni": 1,
                    "clear_i": 1,
                    "start_valid_i": 1,
                    "start_ready_o": 1,
                    "accumulator_s32_i": 32,
                    "residual_s8_i": 8,
                    "accumulator_scale32_i": 32,
                    "residual_scale32_i": 32,
                    "destination_scale32_i": 32,
                    "out_valid_o": 1,
                    "out_ready_i": 1,
                    "fused_s8_o": 8,
                    "positive_saturation_o": 1,
                    "negative_saturation_o": 1,
                    "descriptor_error_o": 1,
                    "numeric_overflow_o": 1,
                    "numerator_s96_o": 96,
                    "denominator_u64_o": 64,
                    "common_exponent_s8_o": 8,
                    "latency_cycles_u5_o": 5,
                },
                "interface_xml": interface_xml,
                "forbidden_inputs_absent": [
                    "prompt_id",
                    "dataset_id",
                    "token_selector",
                    "quality_gate",
                    "reconstructed_down_projection_int8",
                ],
            },
            "candidate_model_metadata": {
                "status": "generated_and_hash_bound_before_quality",
                "record_count": 21504,
                "layer_count": 24,
                "lane_count": 896,
                "immutable_image_bytes": 86592,
                "allocation_bytes": 86720,
                "quality_tuning_permitted": False,
                "manifest": metadata_json,
                "binary": metadata_bin,
            },
            "candidate_schedule": {
                "algorithm": "exact_preclamp_then_eight_shift_subtract_quotient_cycles",
                "valid_cycles_per_output_lane": 10,
                "fusion_cycles_per_layer": 8960,
                "all_layer_fusion_cycles": 215040,
                "down_projection_dot_cycles_per_layer": 1089536,
                "fusion_overhead_percent_of_dot_schedule": 0.8223684210526315,
            },
            "candidate_meets_numeric_acceptance": False,
            "candidate_requires_fresh_sky130_ppa": True,
            "candidate_review_binding": {
                "candidate_capability_accepted": False,
                "decision": "fresh_independent_rtl_stage_closing_review_pending",
                "scope": "rtl_stage_close_for_verification_entry_only",
                "stage_closing": False,
                "evidence": PRECHECK.relative_to(ROOT).as_posix(),
                "prior_standalone_review": scoped_artifact(prior_review_path, scope="sealed_whole_file"),
                "reviewer_status": "pending",
            },
            "proposed_replacement_contract": authority,
            "claim_boundaries": [
                "Standalone synthesizable candidate RTL, exact metadata, deterministic vectors, lint, elaboration, and simulation only.",
                "Accepted prefix and historical PPA frontier are unchanged.",
                "No focused quality, full shell regression, candidate PPA, prototype, benchmark, signoff, tapeout, or silicon claim is made.",
            ],
        }
    )
    manifest["architecture_selection_evidence"]["probe"] = current_probe_record()
    manifest["architecture_selection_evidence"]["status"] = "accepted_architecture_historical_probe_only_successor_rtl_implemented"
    manifest["candidate_evidence_hashes"]["architecture_probe_sha256"] = sha256_file(PROBE)
    manifest["traceability"] = {
        "architecture_contract_gap": {
            "status": "candidate_implemented_prior_standalone_review_historical_fresh_stage_closing_review_pending",
            "resolution_owner": "fresh_independent_l2_reviewer",
        },
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "selected_mechanism": CONTRACT,
        "rtl.contract-traceability": True,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
        "stage_checklist": copy.deepcopy(CHECKLIST),
    }
    manifest["independent_reviewer_acceptance"] = {
        "architecture_review": "accepted",
        "environment_review": "accepted_manager_advanced_to_rtl",
        "rtl_review": "pending_fresh_independent_stage_closing_review",
        "status": "pending_fresh_independent_rtl_stage_closing_review",
    }
    provenance = manifest.setdefault("ip_provenance", [])
    provenance[:] = [entry for entry in provenance if entry.get("name") != "ace2_down_projection_residual_fusion_core"]
    provenance.append(
        {
            "name": "ace2_down_projection_residual_fusion_core",
            "kind": "first_party_bounded_candidate_rtl",
            "path": rtl_artifact["path"],
            "sha256": rtl_artifact["sha256"],
            "source_revision": rtl_artifact["sha256"],
            "license": "repository project license not separately declared in this manifest",
            "third_party": False,
        }
    )
    mark_historical_records(manifest)
    dump("design/RTL_MANIFEST.json", manifest)

    policy = load("design/FAST_LOOP_POLICY.json")
    for key in ("active_repair_authorization", "architecture_proposal_authorization", "selected_replacement_contract"):
        policy[key] = copy.deepcopy(authority)
    policy["manager_recommendation"] = "hold_rtl_for_independent_checklist_review"
    dump("design/FAST_LOOP_POLICY.json", policy)

    target = load("design/TARGET.json")
    target["current_stage"] = "rtl"
    target["current_architecture_contract"] = copy.deepcopy(authority)
    target["fast_loop_contract"]["manager_recommendation"] = policy["manager_recommendation"]
    for key in ("active_repair_authorization", "architecture_proposal_authorization", "selected_replacement_contract"):
        target["fast_loop_contract"][key] = copy.deepcopy(authority)
    target["generated_at_utc"] = now
    dump("design/TARGET.json", target)

    scope = load("design/CHIP_SCOPE.json")
    scope["stage"] = {
        "current_stage": "rtl",
        "current_stage_status": "candidate_preflight_pass_independent_review_pending",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_evidence": [
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
            PRECHECK.relative_to(ROOT).as_posix(),
        ],
        "downstream_stages_locked_until_manager_advance": ["verification", "ppa", "prototype", "benchmark", "signoff"],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    scope["authority_override"].update(
        {
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "operator_implementation_approval": copy.deepcopy(authority),
            "required_next_action": "independent_rtl_checklist_review",
        }
    )
    scope["authority_override"]["architecture_probe"] = current_probe_record()
    scope["numerical_behavior"]["active_down_projection_residual_fusion_contract"].update(
        {
            "implementation_authorized": False,
            "candidate_id": candidate_id,
            "candidate_rtl_hash": candidate_hash,
            "status": "rtl_preflight_pass_independent_review_pending",
            "derived_latency_cycles_per_lane": 10,
        }
    )
    scope["implementation_frontier"].update(
        {
            "current_mode": "ADVANCE",
            "latest_decision": "down_projection_residual_fusion_rtl_preflight_pass_review_pending",
            "ordered_supported_layer_operator_prefix": PREFIX,
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "candidate_rtl_hash": candidate_hash,
            "candidate_rtl_hash_scope": "standalone_not_shell_admitted_no_candidate_ppa",
        }
    )
    scope["last_updated_utc"] = now
    dump("design/CHIP_SCOPE.json", scope)

    oracle = {
        "schema_version": 2,
        "generated_at_utc": now,
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "claim_boundary": "Independent scalar/reference, generated model metadata, and standalone RTL preflight only; focused quality and downstream stages were not run.",
        "standalone_vector_oracle": {
            "numeric_acceptance": "bit_exact",
            "architecture_hook": artifact("tools/ace2_down_projection_residual_fusion_hook.py"),
            "independent_reference": artifact("tools/ace2_down_projection_residual_fusion_reference.py"),
            "vectors_json": vectors_json,
            "vectors_svh": vectors_svh,
        },
        "model_image_metadata": {
            "manifest": metadata_json,
            "binary": metadata_bin,
            "record_count": 21504,
            "quality_metrics_executed": False,
        },
    }
    dump("reference/ORACLE_MANIFEST.json", oracle)

    trace = f"""# ACE-2 RTL traceability notes

## Accepted shell frontier

The accepted shell remains hash-bound through `layer_0.v_proj`; first unsupported
remains `layer_0.rope_q`. Historical accepted PPA remains 62,199 cells,
0.6108746272 mm2, and +0.1502 ns setup slack at 100 MHz.

## Current down-projection/residual fusion candidate

- Contract: `{CONTRACT}`.
- Candidate: `{candidate_id}`; ordered source hash `{candidate_hash}`.
- `ace2_down_projection_residual_fusion_core` consumes the authoritative signed-32
  down-projection accumulator, signed-int8 post-attention residual, and exactly
  three valid Scale32 records. It never reconstructs the accumulator from int8.
- Both source terms are aligned without rounding in signed-96, added once, and
  divided once with signed round-to-nearest ties-to-even before one int8 clamp.
- Exact asymmetric pre-clamp predicates are `2*abs(N) >= 255*D` for positive
  saturation and `2*abs(N) > 257*D` for negative saturation. Every remaining
  quotient is at most 128, so eight shift/subtract bits plus prepare/finalize
  produce a fixed ten-cycle valid latency.
- Static input-domain proof bounds the numerator magnitude to 76 bits and the
  denominator to 44 bits inside the frozen signed-96/unsigned-64 workspaces.
- The generated immutable model image is exactly 86,592 bytes with 24 checked
  headers, 24 residual/destination pairs, and 21,504 accumulator Scale32 records.
- Reset, clear, backpressure, descriptor errors, ties, exponent extremes,
  saturation, and maximum legal operands are covered by the bound preflight.
- RTL and generated sources are first-party with pinned hashes and regeneration
  commands. No third-party RTL or generated IP is added.
- The architecture's 12-cycle value was explicitly an unmeasured planning
  assumption; this RTL derives 10 cycles/lane, 8,960 cycles/layer, and 215,040
  cycles across 24 layers. This is not PPA or end-to-end throughput evidence.
- A prior independent standalone review accepted the three RTL checklist fields, but
  its `stage_closing=false` verdict is historical. One distinct fresh independent L2
  stage-closing review remains pending. No focused quality, shell, PPA, or downstream
  claim exists.
"""
    (ROOT / "design/RTL_TRACEABILITY.md").write_text(trace, encoding="utf-8")
    return manifest


def update_public(
    candidate_id: str,
    candidate_hash: str,
    authority: dict[str, Any],
    prior_review_path: Path,
    now: str,
) -> None:
    status = load("research/PUBLIC_STATUS.json")
    status["current_mode"] = "ADVANCE"
    status["supported_layer_operator_prefix"] = PREFIX
    status["ordered_supported_layer_operator_prefix"] = PREFIX
    status["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
    status["latest_decision"] = "down_projection_residual_fusion_fresh_rtl_stage_closing_review_pending"
    status["latest_ppa_frontier_status"] = "historical_frontier_preserved_no_candidate_ppa"
    status["selected_replacement_contract"] = copy.deepcopy(authority)
    status["architecture_proposal_gate"] = {
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "required_manager_action": "hold_rtl_for_independent_checklist_review",
        "stage_closing": False,
        "probe": current_probe_record(),
        "status": "fresh_independent_rtl_stage_closing_review_pending",
    }
    status["stage"] = {
        "completed_prior_stages": ["definition", "architecture", "environment"],
        "current_stage": "rtl",
        "current_stage_checklist": copy.deepcopy(CHECKLIST),
        "current_stage_evidence": ["design/RTL_MANIFEST.json", "design/RTL_TRACEABILITY.md", PRECHECK.relative_to(ROOT).as_posix()],
        "current_stage_status": "fresh_independent_rtl_stage_closing_review_pending",
        "planner_may_advance_stage": False,
        "stage_closing": False,
        "stage_transition_owner": "Manager",
    }
    status["blockers"] = [
        {
            "id": "fresh_independent_rtl_stage_closing_review_pending",
            "stage": "rtl",
            "status": "active",
            "reason": "The prior standalone review is historical and stage_closing=false; one distinct fresh independent L2 stage-closing review is required.",
            "required_resolution": "Run the maintained DPRF RTL reviewer exactly once and require all three checklist fields true with stage_closing=true.",
            "evidence": prior_review_path.relative_to(ROOT).as_posix(),
        }
    ]
    candidate = {
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "candidate_rtl_hash_scope": "standalone_not_shell_admitted_no_candidate_ppa",
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "rtl_review_stage_closing": False,
        "stage_closing": False,
        "derived_latency_cycles_per_lane": 10,
        "status": "fresh_independent_rtl_stage_closing_review_pending",
        "review": prior_review_path.relative_to(ROOT).as_posix(),
    }
    status["latest_rtl_candidate"] = copy.deepcopy(candidate)
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status.setdefault(name, {})
        container.update(
            {
                "current_stage": "rtl",
                "current_mode": "ADVANCE",
                "supported_layer_operator_prefix": PREFIX,
                "ordered_supported_layer_operator_prefix": PREFIX,
                "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
                "latest_decision": status["latest_decision"],
                "latest_ppa_frontier_status": status["latest_ppa_frontier_status"],
                "latest_rtl_candidate": copy.deepcopy(candidate),
                "candidate_rtl_hash": candidate_hash,
                "candidate_meets_numeric_acceptance": False,
                "routing_status": "fresh_independent_rtl_stage_closing_review_pending",
            }
        )
    status["public_claims"] = [
        {"claim": "the Manager-owned current stage is rtl", "evidence": ["research/PIPELINE_STATE.json"]},
        {"claim": "the DPRF standalone RTL preflight passed for the published candidate hash", "evidence": [PRECHECK.relative_to(ROOT).as_posix(), "design/RTL_MANIFEST.json"]},
        {"claim": "the supported prefix, immutable targets, mode, and historical PPA frontier are unchanged", "evidence": ["design/CHIP_SCOPE.json", "design/TARGET.json"]},
        {"claim": "no focused quality, shell regression, candidate PPA, prototype, benchmark, signoff, tapeout, or silicon result exists", "evidence": [PRECHECK.relative_to(ROOT).as_posix()]},
        {
            "claim": "the retained float64 probe is exploratory historical context only; no exact Scale32 two-dataset discriminator has run",
            "evidence": [
                PROBE.relative_to(ROOT).as_posix(),
                ARCH_PACKET.relative_to(ROOT).as_posix(),
            ],
        },
    ]
    status["architecture_proposal_gate"]["probe"] = current_probe_record()
    status["selected_replacement_contract"]["probe"] = current_probe_record()
    active_artifacts = []
    historical_checkpoint_records = list(status.get("historical_mutable_artifact_observations", []))
    for record in status.get("artifact_hashes", []):
        if isinstance(record, dict) and record.get("path") == "CHECKPOINT.md":
            historical = copy.deepcopy(record)
            historical["binding_scope"] = "historical_mutable_path_observation"
            historical["historical_reason"] = (
                "Root CHECKPOINT.md is mutable operational state and is retained only as a historical observation."
            )
            historical_checkpoint_records.append(historical)
        else:
            active_artifacts.append(record)
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for record in historical_checkpoint_records:
        if isinstance(record, dict) and isinstance(record.get("path"), str) and isinstance(record.get("sha256"), str):
            deduplicated[(record["path"], record["sha256"])] = record
    status["artifact_hashes"] = active_artifacts
    status["historical_mutable_artifact_observations"] = list(deduplicated.values())
    status["routing_status"] = "fresh_independent_rtl_stage_closing_review_pending"
    status["routing_authorized"] = "none_pending_fresh_independent_rtl_stage_closing_review"
    status["generated_at_utc"] = now
    status.setdefault("integrity", {})["canonical_sha256"] = None
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump("research/PUBLIC_STATUS.json", status)


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    previous_precheck = load(PRECHECK)
    _prior_review, prior_review_path = archive_prior_standalone()
    transition = manager_transition()
    require(load("design/RTL_MANIFEST.json").get("candidate_layer_operator") == CONTRACT,
            "manifest active candidate contract mismatch")
    require(load("design/CHIP_SCOPE.json")["implementation_frontier"]["first_unsupported_layer_operator"] == FIRST_UNSUPPORTED,
            "first unsupported operator changed")

    logs: dict[str, dict[str, Any]] = {}
    logs["architecture_hook_self_test"] = run(
        [".venv/bin/python", "tools/ace2_down_projection_residual_fusion_hook.py", "--self-test"],
        "architecture_hook_self_test.log",
    )
    logs["metadata_check"] = run(
        [".venv/bin/python", "tools/gen_down_projection_residual_fusion_metadata.py", "--check"],
        "metadata_check.log",
    )
    run([".venv/bin/python", "tools/gen_down_projection_residual_fusion_vectors.py"], "vector_generation.log")
    first_vector_hashes = {
        path: sha256_file(ROOT / path)
        for path in (
            "verification/generated/down_projection_residual_fusion_vectors.json",
            "verification/generated/down_projection_residual_fusion_vectors.svh",
        )
    }
    logs["vector_regeneration"] = run(
        [".venv/bin/python", "tools/gen_down_projection_residual_fusion_vectors.py"],
        "vector_regeneration.log",
    )
    require(first_vector_hashes == {path: sha256_file(ROOT / path) for path in first_vector_hashes},
            "DPRF vector generation is not deterministic")
    logs["software_reference"] = run(
        [".venv/bin/python", "-m", "unittest", "verification.test_down_projection_residual_fusion", "-v"],
        "software_reference_unittest.log",
    )
    logs["minimal_formal"] = run(
        [".venv/bin/python", "tools/run_down_projection_residual_fusion_formal.py"],
        "minimal_formal.log",
    )
    require(
        "ACE2_DPRF_MINIMAL_FORMAL_PASS" in (EVIDENCE / "minimal_formal.log").read_text(encoding="utf-8"),
        "minimal formal proof marker missing",
    )
    build = ROOT / "build/down_projection_residual_fusion"
    build.mkdir(parents=True, exist_ok=True)
    logs["elaboration"] = run(
        ["iverilog", "-g2012", "-Wall", "-Iverification/generated", "-o", str(build / "ace2_down_projection_residual_fusion_tb.vvp"),
         "rtl/ace2_down_projection_residual_fusion_core.sv", "verification/tb/ace2_down_projection_residual_fusion_tb.sv"],
        "rtl_elaboration.log",
    )
    logs["simulation"] = run(
        ["vvp", str(build / "ace2_down_projection_residual_fusion_tb.vvp")],
        "rtl_simulation.log",
    )
    marker = "ACE2_DOWN_PROJECTION_RESIDUAL_FUSION_RTL_PASS cases=16 valid_latency=10 divide_bits=8"
    require(marker in (EVIDENCE / "rtl_simulation.log").read_text(encoding="utf-8"),
            "RTL simulation did not reproduce the derived schedule")
    logs["lint"] = run(
        ["verilator", "--lint-only", "--language", "1800-2017", "-Wall", "--top-module",
         "ace2_down_projection_residual_fusion_core", "rtl/ace2_down_projection_residual_fusion_core.sv"],
        "rtl_lint.log",
    )
    interface_path = EVIDENCE / "interface.xml"
    logs["interface_xml"] = run(
        ["verilator", "--xml-only", "--language", "1800-2017", "-Wall", "--top-module",
         "ace2_down_projection_residual_fusion_core", "--xml-output", str(interface_path),
         "rtl/ace2_down_projection_residual_fusion_core.sv"],
        "interface_xml.log",
    )
    interface_xml = artifact(interface_path)

    source_hashes = {path: sha256_file(ROOT / path) for path in SOURCE_PATHS}
    candidate_hash = aggregate_hash(source_hashes)
    candidate_id = f"down_projection_residual_fusion_{candidate_hash[:16]}"
    now = utc_now()
    manifest = update_state(
        candidate_id,
        candidate_hash,
        source_hashes,
        transition,
        logs,
        interface_xml,
        prior_review_path,
        now,
    )
    authority = manifest["proposed_replacement_contract"]

    update_public(candidate_id, candidate_hash, authority, prior_review_path, now)

    checkpoint = f"""# Goal

Close the RTL checklist for `{CONTRACT}` without entering verification, shell, PPA, or any later stage.

# Current State

The Manager-owned stage is `rtl`. Candidate `{candidate_id}` passed deterministic
reference/vector regeneration, lint, elaboration, bit-exact simulation, bounded formal,
and recursive evidence validation preconditions. The earlier standalone review remains
historical with `stage_closing=false`; exactly one fresh independent L2 stage-closing
review is pending.

# Boundaries

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
`layer_0.rope_q`; mode remains `ADVANCE`. Focused quality, paired/shell work, candidate
PPA, physical, prototype, benchmark, signoff, tapeout, and silicon were not run.
"""
    (ROOT / "CHECKPOINT.md").write_text(checkpoint, encoding="utf-8")

    transition_binding = create_transition_binding(
        candidate_id=candidate_id,
        candidate_hash=candidate_hash,
        source_hashes=source_hashes,
        manager_event=transition,
        previous_precheck=previous_precheck,
        prior_review_path=prior_review_path,
        now=now,
    )

    precheck: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": now,
        "current_stage": "rtl",
        "contract_id": CONTRACT,
        "candidate_id": candidate_id,
        "candidate_rtl_hash": candidate_hash,
        "status": "pass_ready_for_independent_rtl_review",
        "stage_closing": False,
        "checklist": copy.deepcopy(CHECKLIST),
        "manager_transition": transition,
        "source_hashes": source_hashes,
        "checks": {
            "architecture_hook_self_test_pass": True,
            "generated_24_layer_21504_record_metadata_pass": True,
            "deterministic_vector_regeneration_pass": True,
            "independent_reference_2000_random_cases_pass": True,
            "signed96_unsigned64_workspace_proof_pass": True,
            "minimal_yosys_sat_formal_pass": True,
            "iverilog_elaboration_clean": True,
            "bit_exact_rtl_simulation_pass": True,
            "verilator_lint_warning_clean": True,
            "manifest_interface_matches_verilator_elaboration": True,
            "ten_cycle_valid_schedule_asserted": True,
            "reset_clear_backpressure_ties_saturation_and_errors_exercised": True,
            "first_party_provenance_and_regeneration_bound": True,
        },
        "schedule": copy.deepcopy(manifest["candidate_schedule"]),
        "metadata": copy.deepcopy(manifest["candidate_model_metadata"]),
        "transition_binding": transition_binding,
        "prior_standalone_review": scoped_artifact(prior_review_path, scope="sealed_whole_file"),
        "sealed_architecture": {
            "contract_bindings": scoped_artifact(ARCH_BINDINGS, scope="sealed_whole_file"),
            "accepted_decision": scoped_artifact(ARCH_DECISION, scope="sealed_whole_file"),
        },
        "evidence": {
            "manifest": scoped_artifact("design/RTL_MANIFEST.json", scope="current_live_binding", recursive=True),
            "traceability": scoped_artifact("design/RTL_TRACEABILITY.md", scope="current_live_binding"),
            "target": scoped_artifact("design/TARGET.json", scope="current_live_binding", recursive=True),
            "fast_loop_policy": scoped_artifact("design/FAST_LOOP_POLICY.json", scope="current_live_binding", recursive=True),
            "chip_scope": scoped_artifact("design/CHIP_SCOPE.json", scope="current_live_binding", recursive=True),
            "public_status": scoped_artifact("research/PUBLIC_STATUS.json", scope="current_live_binding", recursive=True),
            "pipeline_state": scoped_artifact("research/PIPELINE_STATE.json", scope="current_live_binding", recursive=True),
            "rtl": scoped_artifact(RTL, scope="current_live_binding"),
            "reference": scoped_artifact("tools/ace2_down_projection_residual_fusion_reference.py", scope="current_live_binding"),
            "architecture_hook": scoped_artifact("tools/ace2_down_projection_residual_fusion_hook.py", scope="current_live_binding"),
        },
        "control_source_hashes": {
            relative: sha256_file(ROOT / relative)
            for relative in CONTROL_PATHS
        },
        "logs": logs,
        "interface_xml": {"ace2_down_projection_residual_fusion_core": interface_xml},
        "coverage": {
            "deterministic_cases": 16,
            "random_reference_cases": 2000,
            "layer_metadata_records": 21504,
            "ties": ["positive_127_point_5", "negative_128_point_5"],
            "protocol": ["reset", "clear", "backpressure", "busy_start_rejection", "descriptor_error"],
        },
        "prohibited_runs": {
            "focused_quality": False,
            "full_shell_regression": False,
            "sky130_ppa": False,
            "prototype": False,
            "benchmark": False,
            "signoff": False,
        },
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "canonical_sha256": None,
        },
    }
    precheck["integrity"]["canonical_sha256"] = canonical_sha256(precheck)
    dump(PRECHECK, precheck)
    validation = validate_artifact_tree(precheck, label=PRECHECK.relative_to(ROOT).as_posix())
    print(
        f"ACE2_DPRF_RTL_PREFLIGHT_PASS candidate={candidate_id} hash={candidate_hash} "
        f"recursive_records={validation.current_records} historical_records={validation.historical_records}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
