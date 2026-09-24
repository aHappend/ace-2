#!/usr/bin/env python3
"""Manager-only recovery seal for the terminal QECR baseline protocol NO_GO.

This tool never executes model, baseline, candidate, RTL, or reference entrypoints.
It snapshots the immutable negative evidence, atomically publishes a seal plus its
companion hash, and rolls the Manager-owned pipeline state back to architecture.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "cross_layer_quantization_error_carry_final_output_v1"
LATEST = ROOT / "evidence" / CONTRACT / "latest"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
POLICY = ROOT / "design/FAST_LOOP_POLICY.json"
SCOPE = ROOT / "design/CHIP_SCOPE.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
SEAL_DIR = ROOT / "evidence" / CONTRACT / "recovery" / "manager_architecture_rollback_v1"
SEAL = SEAL_DIR / "SEAL.json"
SEAL_SUM = SEAL_DIR / "SEAL.sha256"
NO_GO = LATEST / "BASELINE_NO_GO.json"
NO_GO_SUM = LATEST / "BASELINE_NO_GO.sha256"
REVIEW = LATEST / "L2_BASELINE_REVIEW.json"
REVIEW_SUM = LATEST / "L2_BASELINE_REVIEW.sha256"
LIVE_RTL = ROOT / "rtl/ace2_cross_layer_error_carry_core.sv"

ACCIDENTAL_0530_NAMES = (
    "vector_check.log",
    "metadata_check.log",
    "software_reference_unittest.log",
    "rtl_elaboration.log",
    "rtl_simulation.log",
    "lint_ace2_quantization_error_carry_lane_core.log",
    "elaborate_ace2_quantization_error_carry_lane_core.log",
    "interface_ace2_quantization_error_carry_lane_core.xml",
    "interface_ace2_quantization_error_carry_lane_core.log",
    "lint_ace2_error_carry_state_core.log",
    "elaborate_ace2_error_carry_state_core.log",
    "interface_ace2_error_carry_state_core.xml",
    "interface_ace2_error_carry_state_core.log",
    "lint_ace2_carry_aware_rmsnorm_core.log",
    "elaborate_ace2_carry_aware_rmsnorm_core.log",
    "interface_ace2_carry_aware_rmsnorm_core.xml",
    "interface_ace2_carry_aware_rmsnorm_core.log",
    "minimal_formal.log",
    "PRECHECK.json",
)

ROLLBACK_REASON = (
    "Terminal BASELINE_NO_GO and independent L2 review establish an unusable "
    "baseline protocol: zero baseline executions, missing companion hash, "
    "misbound provenance, and a prohibited candidate entrypoint invocation. "
    "Roll rtl back to architecture, seal cross_layer_quantization_error_carry_"
    "final_output_v1 and all accidental post-terminal 05:30 artifacts, and "
    "authorize only candidate-independent synthetic harness hardening."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing immutable artifact: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ),
    }


def verify_companion(payload: Path, companion: Path) -> None:
    fields = companion.read_text(encoding="utf-8").strip().split()
    require(len(fields) == 2, f"invalid companion hash: {companion}")
    require(fields[1] == payload.name, f"companion names wrong payload: {companion}")
    require(fields[0] == sha256(payload), f"companion mismatch: {companion}")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_seal() -> dict[str, Any]:
    verify_companion(NO_GO, NO_GO_SUM)
    verify_companion(REVIEW, REVIEW_SUM)
    no_go = load(NO_GO)
    review = load(REVIEW)
    require(no_go.get("contract_id") == CONTRACT, "NO_GO contract mismatch")
    require(no_go.get("decision") == "NO_GO", "NO_GO decision mismatch")
    require(review.get("contract_id") == CONTRACT, "review contract mismatch")
    require(review.get("decision") == "NO_GO", "review decision mismatch")
    require(review.get("reviewed_artifact", {}).get("sha256") == sha256(NO_GO), "review binding mismatch")
    require(review.get("acceptance_checks", {}).get("candidate_mechanism_execution_count") == 0, "candidate mechanism execution was recorded")

    accidental = [artifact(LATEST / name) for name in ACCIDENTAL_0530_NAMES]
    for item in accidental:
        require(
            "2026-08-02T05:30:00" <= item["mtime_utc"] < "2026-08-02T05:31:00",
            f"post-terminal artifact escaped 05:30 window: {item['path']}",
        )

    return {
        "schema_version": 1,
        "project": "ACE-2",
        "contract_id": CONTRACT,
        "status": "sealed_terminal_baseline_protocol_no_go",
        "decision": "NO_GO",
        "sealed_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sealed_by": "manager",
        "manager_transition": {
            "from_stage": "rtl",
            "to_stage": "architecture",
            "reason": ROLLBACK_REASON,
        },
        "immutable_terminal_evidence": [
            artifact(NO_GO),
            artifact(NO_GO_SUM),
            artifact(REVIEW),
            artifact(REVIEW_SUM),
        ],
        "immutable_accidental_post_terminal_0530_evidence": accidental,
        "observed_failed_candidate_source": artifact(LIVE_RTL),
        "execution_boundary": {
            "baseline_executed_by_this_tool": False,
            "candidate_entrypoint_invoked_by_this_tool": False,
            "candidate_mechanism_executed_by_this_tool": False,
            "rtl_or_reference_or_vector_or_precheck_modified_by_this_tool": False,
            "paired_shell_ppa_physical_prototype_benchmark_signoff_run": False,
        },
        "authorization_after_seal": {
            "candidate_independent_harness_code_tests_and_synthetic_fixtures_only": True,
            "candidate_or_model_execution": False,
            "candidate_rtl_reference_vectors_precheck_mutation": False,
            "successor_freeze": False,
            "fresh_independent_l2_harness_review_required": True,
        },
        "preserved_operator_contract": {
            "mode": "ADVANCE",
            "area_cap_non_sram_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "memory_boundary_bits": 128,
            "ordered_supported_layer_operator_prefix": [
                "layer_0.input_rmsnorm",
                "layer_0.q_proj",
                "layer_0.k_proj",
                "layer_0.v_proj",
            ],
            "first_unsupported_layer_operator": "layer_0.rope_q",
        },
        "pipeline_state_before_sha256": sha256(PIPELINE),
    }


def publish_seal_atomically(value: dict[str, Any]) -> None:
    if SEAL_DIR.exists():
        verify_companion(SEAL, SEAL_SUM)
        require(load(SEAL) == value or load(SEAL).get("status") == value.get("status"), "existing recovery seal differs")
        return

    staging = SEAL_DIR.with_name(f".{SEAL_DIR.name}.tmp.{os.getpid()}")
    require(not staging.exists(), f"stale staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        payload = staging / SEAL.name
        companion = staging / SEAL_SUM.name
        with payload.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        with companion.open("w", encoding="utf-8") as stream:
            stream.write(f"{sha256(payload)}  {payload.name}\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, SEAL_DIR)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def transition_record(seal: dict[str, Any]) -> dict[str, Any]:
    return {
        "at": seal["sealed_at_utc"],
        "by": "manager",
        "direction": "rollback",
        "from_stage": "rtl",
        "to_stage": "architecture",
        "reason": ROLLBACK_REASON,
        "evidence": {
            "path": SEAL.relative_to(ROOT).as_posix(),
            "sha256": sha256(SEAL),
        },
    }


def update_pipeline(seal: dict[str, Any]) -> None:
    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") in {"rtl", "architecture"}, "unexpected Manager stage")
    successor = pipeline.get("successor", {})
    require(successor.get("id") == CONTRACT, "active mechanism changed before recovery seal")
    record = transition_record(seal)
    already_recorded = any(
        isinstance(item, dict)
        and item.get("evidence", {}).get("sha256") == record["evidence"]["sha256"]
        for item in pipeline.get("stage_history", [])
    )
    if not already_recorded:
        require(pipeline.get("current_stage") == "rtl", "architecture state lacks recovery transition")
        pipeline.setdefault("rollback_history", []).append(copy.deepcopy(record))
        pipeline.setdefault("stage_history", []).append(copy.deepcopy(record))

    pipeline["current_stage"] = "architecture"
    stages = pipeline.setdefault("stages", {})
    stages.setdefault("architecture", {})["status"] = "in_progress"
    stages.setdefault("rtl", {})["status"] = "locked_pending_harness_certification_and_successor_freeze"
    stages.setdefault("verification", {})["status"] = "locked_pending_predecessor_stage_acceptance"
    for name in ("ppa", "prototype", "benchmark", "signoff"):
        stages.setdefault(name, {})["status"] = "locked_pending_predecessor_stage_acceptance"

    successor.update({
        "state": "sealed_terminal_baseline_protocol_no_go",
        "sealed": True,
        "candidate_capability_accepted": False,
        "baseline_or_candidate_execution_authorized": False,
        "same_contract_rtl_authorized": False,
        "successor_freeze_authorized": False,
        "required_next_gate": "fresh_independent_l2_review_of_candidate_independent_synthetic_baseline_harness",
        "seal_evidence": record["evidence"],
    })
    atomic_json(PIPELINE, pipeline)


def update_projection(path: Path, seal: dict[str, Any]) -> None:
    value = load(path)
    seal_ref = {
        "path": SEAL.relative_to(ROOT).as_posix(),
        "sha256": sha256(SEAL),
    }
    if path == POLICY:
        value["manager_recommendation"] = "certify_candidate_independent_synthetic_baseline_harness"
        gate = value.setdefault("baseline_harness_hardening_gate", {})
        gate.update({
            "status": "authorized_at_architecture_pending_engineer_evidence_and_independent_l2",
            "candidate_independent": True,
            "dry_fixture_only": True,
            "candidate_entrypoint_permitted": False,
            "candidate_or_model_execution_permitted": False,
            "successor_freeze_permitted": False,
            "independent_review_required": True,
            "manager_recovery_seal": seal_ref,
            "required_properties": [
                "atomic_artifact_and_companion_hash_commit",
                "exact_provenance_binding",
                "durable_exactly_one_reservation_and_run_ledger",
                "stage_authorization_before_execution",
                "candidate_entrypoint_interlock",
                "crash_recovery",
                "missing_or_mismatched_artifact_fail_closed",
            ],
        })
        for key in ("active_architecture_contract", "selected_replacement_contract"):
            contract = value.get(key)
            if isinstance(contract, dict) and contract.get("contract_id") == CONTRACT:
                contract.update({
                    "status": "sealed_terminal_baseline_protocol_no_go",
                    "implementation_authorized": False,
                    "successor_selection_authorized": False,
                    "required_manager_action": "certify_candidate_independent_synthetic_baseline_harness",
                    "manager_recovery_seal": seal_ref,
                })
                disposition = contract.get("baseline_protocol_disposition")
                if isinstance(disposition, dict):
                    disposition["status"] = "sealed_terminal_baseline_protocol_no_go"
                    disposition["manager_recovery_seal"] = seal_ref
    else:
        stage = value.setdefault("stage", {})
        stage.update({
            "current_stage": "architecture",
            "current_stage_status": "failed_mechanism_sealed_harness_hardening_authorized",
            "planner_may_advance_stage": False,
            "stage_closing": False,
            "stage_transition_owner": "Manager",
        })
        value["current_stage"] = "architecture" if "current_stage" in value else value.get("current_stage")
        value["latest_decision"] = "terminal_baseline_no_go_manager_architecture_rollback_complete"
        value["required_manager_action"] = "certify_candidate_independent_synthetic_baseline_harness"
        value["manager_recovery_seal"] = seal_ref
        value["routing_status"] = "architecture_harness_hardening_only"
        value["stage_closing"] = False
        for key in ("selected_replacement_contract", "active_architecture_contract"):
            contract = value.get(key)
            if isinstance(contract, dict) and contract.get("contract_id") == CONTRACT:
                contract.update({
                    "status": "sealed_terminal_baseline_protocol_no_go",
                    "implementation_authorized": False,
                    "successor_selection_authorized": False,
                    "manager_recovery_seal": seal_ref,
                })
    atomic_json(path, value)


def verify_seal() -> dict[str, Any]:
    verify_companion(SEAL, SEAL_SUM)
    seal = load(SEAL)
    require(seal.get("contract_id") == CONTRACT, "seal contract mismatch")
    for section in ("immutable_terminal_evidence", "immutable_accidental_post_terminal_0530_evidence"):
        for item in seal.get(section, []):
            path = ROOT / item["path"]
            require(path.is_file(), f"sealed artifact missing: {path}")
            require(path.stat().st_size == item["bytes"], f"sealed artifact size changed: {path}")
            require(sha256(path) == item["sha256"], f"sealed artifact hash changed: {path}")
    failed_source = seal["observed_failed_candidate_source"]
    require(sha256(ROOT / failed_source["path"]) == failed_source["sha256"], "failed candidate RTL changed")
    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") == "architecture", "pipeline is not at architecture")
    successor = pipeline.get("successor", {})
    require(successor.get("id") == CONTRACT, "failed mechanism identity changed")
    require(successor.get("state") == "sealed_terminal_baseline_protocol_no_go", "failed mechanism is not sealed")
    require(successor.get("successor_freeze_authorized") is False, "successor freeze became authorized")
    gate = load(POLICY).get("baseline_harness_hardening_gate", {})
    require(gate.get("candidate_independent") is True, "harness scope is not candidate-independent")
    require(gate.get("candidate_or_model_execution_permitted") is False, "candidate/model execution is permitted")
    require(gate.get("manager_recovery_seal", {}).get("sha256") == sha256(SEAL), "policy seal binding mismatch")
    return seal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        seal = verify_seal()
        print(
            "ACE2_BASELINE_RECOVERY_SEAL_CHECK_PASS "
            f"seal_sha256={sha256(SEAL)} stage=architecture "
            f"accidental_artifacts={len(seal['immutable_accidental_post_terminal_0530_evidence'])}"
        )
        return 0

    if SEAL_DIR.exists():
        seal = load(SEAL)
    else:
        seal = build_seal()
        publish_seal_atomically(seal)
    update_pipeline(seal)
    update_projection(POLICY, seal)
    update_projection(SCOPE, seal)
    update_projection(PUBLIC, seal)
    verify_seal()
    print(
        "ACE2_BASELINE_RECOVERY_SEAL_BIND_PASS "
        f"seal_sha256={sha256(SEAL)} stage=architecture "
        f"accidental_artifacts={len(seal['immutable_accidental_post_terminal_0530_evidence'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
