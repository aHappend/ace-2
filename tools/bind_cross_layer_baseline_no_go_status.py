#!/usr/bin/env python3
"""Bind the independently reviewed QECR baseline NO_GO without changing stage."""

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
LATEST = ROOT / "evidence" / CONTRACT / "latest"
NO_GO = LATEST / "BASELINE_NO_GO.json"
NO_GO_SUM = LATEST / "BASELINE_NO_GO.sha256"
REVIEW = LATEST / "L2_BASELINE_REVIEW.json"
REVIEW_SUM = LATEST / "L2_BASELINE_REVIEW.sha256"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
POLICY = ROOT / "design/FAST_LOOP_POLICY.json"
SCOPE = ROOT / "design/CHIP_SCOPE.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
LIVE_PRECHECK = LATEST / "PRECHECK.json"
LIVE_RTL = ROOT / "rtl/ace2_cross_layer_error_carry_core.sv"
BINDER = Path(__file__).resolve()

DECISION = "terminal_baseline_no_go_manager_rollback_required"
ROUTING_STATUS = "terminal_baseline_no_go_waiting_manager_rollback"
MANAGER_ACTION = (
    "rollback_rtl_to_architecture_seal_cross_layer_protocol_no_go_then_"
    "certify_candidate_independent_baseline_harness"
)
STATUS = "terminal_baseline_no_go_pending_manager_seal"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def verify_companion(payload: Path, companion: Path) -> None:
    fields = companion.read_text(encoding="utf-8").strip().split()
    require(len(fields) == 2, f"invalid companion hash format: {companion}")
    require(fields[1] == payload.name, f"companion names wrong payload: {companion}")
    require(fields[0] == sha256(payload), f"companion hash mismatch: {companion}")


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    verify_companion(NO_GO, NO_GO_SUM)
    verify_companion(REVIEW, REVIEW_SUM)
    no_go = load(NO_GO)
    review = load(REVIEW)
    pipeline = load(PIPELINE)

    require(pipeline.get("current_stage") == "rtl", "Manager-owned current_stage is not rtl")
    require(no_go.get("contract_id") == CONTRACT, "NO_GO contract differs")
    require(no_go.get("status") == "terminal_no_go", "baseline disposition is not terminal NO_GO")
    require(no_go.get("decision") == "NO_GO", "baseline decision differs")
    execution = no_go.get("execution_in_this_mission", {})
    require(execution.get("baseline_run_count") == 0, "producing mission baseline count differs")
    require(execution.get("candidate_mechanism_execution_count") == 0, "candidate mechanism executed")
    require(review.get("contract_id") == CONTRACT, "review contract differs")
    require(review.get("decision") == "NO_GO", "independent review decision differs")
    require(review.get("status") == "terminal_no_go", "independent review status differs")
    require(review.get("artifact_supports_pass") is False, "independent review incorrectly supports PASS")
    reviewed = review.get("reviewed_artifact", {})
    require(reviewed.get("sha256") == sha256(NO_GO), "review is not bound to live NO_GO")
    require(reviewed.get("companion_hash_matches") is True, "review did not accept companion hash")
    checks = review.get("acceptance_checks", {})
    require(checks.get("run_count_exactly_one") is False, "review incorrectly accepts exact-one")
    require(checks.get("candidate_entrypoint_never_invoked") is False, "review misses candidate invocation")
    require(checks.get("candidate_mechanism_execution_count") == 0, "review records candidate execution")
    return no_go, review, pipeline


def disposition(no_go: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    live_precheck = load(LIVE_PRECHECK) if LIVE_PRECHECK.is_file() else {}
    post_review_rtl = (
        LIVE_PRECHECK.is_file()
        and LIVE_RTL.is_file()
        and LIVE_PRECHECK.stat().st_mtime > REVIEW.stat().st_mtime
        and LIVE_RTL.stat().st_mtime > REVIEW.stat().st_mtime
    )
    return {
        "contract_id": CONTRACT,
        "status": STATUS,
        "decision": "NO_GO",
        "failure_class": "baseline_protocol_and_provenance",
        "baseline_run_count_in_producing_mission": no_go["execution_in_this_mission"]["baseline_run_count"],
        "candidate_entrypoint_invoked": review["acceptance_checks"]["candidate_entrypoint_never_invoked"] is False,
        "candidate_mechanism_execution_count": review["acceptance_checks"]["candidate_mechanism_execution_count"],
        "baseline_rerun_permitted": False,
        "candidate_run_permitted": False,
        "same_contract_rtl_permitted": False,
        "successor_freeze_permitted": False,
        "required_manager_action": MANAGER_ACTION,
        "required_pre_successor_gate": (
            "fresh independent certification of a candidate-independent dry-fixture baseline harness"
        ),
        "required_harness_properties": [
            "one_run_enforcement",
            "artifact_and_companion_hash_atomicity",
            "exact_provenance_binding",
            "stage_gating",
            "candidate_entrypoint_prevention",
        ],
        "reviewed_no_go": artifact(NO_GO),
        "independent_l2_review": artifact(REVIEW),
        "status_binder": artifact(BINDER),
        "post_no_go_same_contract_rtl_observation": {
            "accepted": False,
            "observed_after_independent_review": post_review_rtl,
            "independent_review_mtime_utc": mtime_utc(REVIEW),
            "precheck": artifact(LIVE_PRECHECK) if LIVE_PRECHECK.is_file() else None,
            "precheck_mtime_utc": mtime_utc(LIVE_PRECHECK) if LIVE_PRECHECK.is_file() else None,
            "rtl": artifact(LIVE_RTL) if LIVE_RTL.is_file() else None,
            "rtl_mtime_utc": mtime_utc(LIVE_RTL) if LIVE_RTL.is_file() else None,
            "candidate_id": live_precheck.get("candidate_id"),
            "candidate_rtl_hash": live_precheck.get("candidate_rtl_hash"),
            "reason": "same-contract RTL was written after the terminal review and has no live independent decision",
        },
    }


def update_contract(container: dict[str, Any], key: str, terminal: dict[str, Any], now: str) -> None:
    value = container.get(key)
    if not isinstance(value, dict) or value.get("contract_id") != CONTRACT:
        return
    value["status"] = STATUS
    value["required_manager_action"] = MANAGER_ACTION
    value["implementation_authorized"] = False
    value["candidate_capability_accepted"] = False
    value["stage_closing"] = False
    value["successor_selection_authorized"] = False
    value["baseline_protocol_disposition"] = copy.deepcopy(terminal)
    value["updated_at_utc"] = now


def update_public(terminal: dict[str, Any], now: str) -> None:
    public = load(PUBLIC)
    public["generated_at_utc"] = now
    public["last_updated_utc"] = now
    public["latest_decision"] = DECISION
    public["required_manager_action"] = MANAGER_ACTION
    public["required_operator_action"] = "none"
    public["routing_authorized"] = "manager_rollback_only"
    public["routing_status"] = ROUTING_STATUS
    public["stage_closing"] = False
    public["ordered_supported_layer_operator_prefix"] = PREFIX
    public["supported_layer_operator_prefix"] = PREFIX
    public["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
    public["current_mode"] = "ADVANCE"
    public["terminal_baseline_no_go"] = copy.deepcopy(terminal)
    public["blockers"] = [{
        "id": "terminal_baseline_no_go_manager_rollback_required",
        "status": "open",
        "severity": "gate",
        "owner": "manager",
        "detail": (
            "Independent L2 review confirms terminal baseline protocol/provenance NO_GO. "
            "Baseline and candidate reruns, same-contract RTL, and successor freeze remain prohibited."
        ),
        "evidence": [terminal["reviewed_no_go"], terminal["independent_l2_review"]],
        "resolution": MANAGER_ACTION,
    }]

    update_contract(public, "selected_replacement_contract", terminal, now)
    latest = public.get("latest_rtl_candidate")
    if isinstance(latest, dict):
        latest["status"] = STATUS
        latest["candidate_capability_accepted"] = False
        latest["implementation_authorized"] = False
        latest["required_manager_action"] = MANAGER_ACTION
        latest["stage_closing"] = False
        latest["baseline_protocol_disposition"] = copy.deepcopy(terminal)

    verification = {
        "contract_id": CONTRACT,
        "status": "terminal_baseline_no_go_independently_confirmed",
        "decision": terminal["reviewed_no_go"],
        "independent_l2_review": terminal["independent_l2_review"],
        "baseline_run_count_in_producing_mission": terminal["baseline_run_count_in_producing_mission"],
        "candidate_entrypoint_invoked": terminal["candidate_entrypoint_invoked"],
        "candidate_mechanism_execution_count": terminal["candidate_mechanism_execution_count"],
        "paired_smoke_run": False,
        "shell_regression_run": False,
        "ppa_run": False,
        "stage_closing": False,
        "required_manager_action": MANAGER_ACTION,
    }
    public["latest_verification_stage"] = verification

    for key in ("dashboard_fields", "implementation_frontier"):
        section = public.get(key)
        if not isinstance(section, dict):
            continue
        section["current_stage"] = "rtl"
        section["latest_decision"] = DECISION
        section["required_manager_action"] = MANAGER_ACTION
        section["routing_status"] = ROUTING_STATUS
        section["stage_closing"] = False
        section["current_mode"] = "ADVANCE"
        section["mode"] = "ADVANCE"
        section["ordered_supported_layer_operator_prefix"] = PREFIX
        section["supported_layer_operator_prefix"] = PREFIX
        section["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
        section["candidate_meets_numeric_acceptance"] = False
        section["latest_verification_stage"] = copy.deepcopy(verification)
        section["terminal_baseline_no_go"] = copy.deepcopy(terminal)
        update_contract(section, "candidate_mechanism", terminal, now)
        section_latest = section.get("latest_rtl_candidate")
        if isinstance(section_latest, dict):
            section_latest["status"] = STATUS
            section_latest["candidate_capability_accepted"] = False
            section_latest["implementation_authorized"] = False
            section_latest["required_manager_action"] = MANAGER_ACTION

    stage = public.setdefault("stage", {})
    stage["current_stage"] = "rtl"
    stage["current_stage_status"] = ROUTING_STATUS
    stage["current_stage_checklist"] = {
        "rtl.contract-traceability": True,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
    }
    stage["current_stage_evidence"] = [
        "design/RTL_MANIFEST.json",
        "design/RTL_TRACEABILITY.md",
        terminal["reviewed_no_go"]["path"],
        "evidence/cross_layer_quantization_error_carry_final_output_v1/latest/BASELINE_NO_GO.sha256",
        terminal["independent_l2_review"]["path"],
        "evidence/cross_layer_quantization_error_carry_final_output_v1/latest/L2_BASELINE_REVIEW.sha256",
    ]
    stage["planner_may_advance_stage"] = False
    stage["stage_transition_owner"] = "Manager"
    stage["stage_closing"] = False
    stage["downstream_stages_locked_until_manager_advance"] = [
        "verification", "ppa", "prototype", "benchmark", "signoff"
    ]

    public["public_claims"] = [
        {"claim": "the Manager-owned current stage remains rtl", "evidence": ["research/PIPELINE_STATE.json"]},
        {"claim": "independent L2 review confirms terminal baseline protocol/provenance NO_GO", "evidence": [terminal["reviewed_no_go"]["path"], terminal["independent_l2_review"]["path"]]},
        {"claim": "baseline rerun, candidate execution, same-contract RTL, and downstream work are prohibited pending Manager rollback and harness certification", "evidence": [terminal["independent_l2_review"]["path"]]},
        {"claim": "the accepted prefix, first unsupported operator, ADVANCE mode, immutable targets, and historical PPA frontier are unchanged", "evidence": ["design/CHIP_SCOPE.json", "design/TARGET.json", "design/FAST_LOOP_POLICY.json"]},
    ]

    records = {item.get("path"): item for item in public.get("artifact_hashes", []) if isinstance(item, dict) and item.get("path")}
    for path in (NO_GO, NO_GO_SUM, REVIEW, REVIEW_SUM, LIVE_PRECHECK, LIVE_RTL, BINDER):
        if path.is_file():
            records[path.relative_to(ROOT).as_posix()] = artifact(path)
    public["artifact_hashes"] = [records[key] for key in sorted(records)]
    public.setdefault("integrity", {})["algorithm"] = "sha256-canonical-json-v1"
    public["integrity"]["canonical_sha256"] = None
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)


def update_policy_and_scope(terminal: dict[str, Any], now: str) -> None:
    policy = load(POLICY)
    policy["manager_recommendation"] = MANAGER_ACTION
    policy["baseline_harness_hardening_gate"] = {
        "status": "required_before_successor_freeze",
        "candidate_independent": True,
        "dry_fixture_only": True,
        "independent_review_required": True,
        "required_properties": terminal["required_harness_properties"],
        "candidate_entrypoint_permitted": False,
    }
    update_contract(policy, "active_architecture_contract", terminal, now)
    update_contract(policy, "selected_replacement_contract", terminal, now)
    dump(POLICY, policy)

    scope = load(SCOPE)
    scope["last_updated_utc"] = now
    stage = scope.setdefault("stage", {})
    stage["current_stage"] = "rtl"
    stage["current_stage_status"] = ROUTING_STATUS
    stage["current_stage_checklist"] = {
        "rtl.contract-traceability": True,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
    }
    stage["current_stage_evidence"] = [
        "design/RTL_MANIFEST.json",
        "design/RTL_TRACEABILITY.md",
        terminal["reviewed_no_go"]["path"],
        terminal["independent_l2_review"]["path"],
    ]
    stage["planner_may_advance_stage"] = False
    stage["stage_closing"] = False
    stage["stage_transition_owner"] = "Manager"
    frontier = scope.setdefault("implementation_frontier", {})
    frontier["latest_decision"] = DECISION
    frontier["required_manager_action"] = MANAGER_ACTION
    frontier["candidate_meets_numeric_acceptance"] = False
    frontier["terminal_baseline_no_go"] = copy.deepcopy(terminal)
    dump(SCOPE, scope)


def update_checkpoint(terminal: dict[str, Any]) -> None:
    text = f"""# Goal

Preserve the terminal `{CONTRACT}` protocol/provenance NO_GO and wait for the
Manager-owned rollback to `architecture` before any validation-harness or successor work.

# Current state

`research/PIPELINE_STATE.json` still records Manager-owned `current_stage=rtl`.
The independent L2 review at `{terminal['independent_l2_review']['path']}` confirms
terminal `BASELINE_NO_GO`: the producing mission ran the baseline zero times, the
required top-level companion hash was absent from the purported baseline payload,
frozen provenance was misbound, and a prohibited candidate entrypoint was invoked.
Candidate mechanism execution count remained zero.

A same-contract RTL/preflight was written after the terminal review. It has no live
independent decision, is not accepted, and must not be reviewed, executed, or treated
as a successor. Its observed records are bound in `research/PUBLIC_STATUS.json`.

# Preserved evidence and contracts

- `BASELINE_NO_GO.json` and `L2_BASELINE_REVIEW.json` both match their companion
  SHA-256 files.
- Do not rerun the baseline, run the candidate, repair this contract into PASS, or
  execute paired/shell/PPA/physical/prototype/benchmark/signoff work.
- The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
  `layer_0.rope_q`; mode remains `ADVANCE`.
- The 2.0 mm2 non-SRAM cap, 100 MHz floor, 128-bit memory boundary, and historical
  62,199-cell / 0.6108746272-mm2 / +0.1502-ns frontier are unchanged.

# Next legal sequence

1. Manager rolls `rtl` back to `architecture` and seals this contract as a
   protocol/provenance NO_GO while preserving all negative evidence.
2. Create one candidate-independent baseline-harness hardening task using only a
   non-candidate dry fixture. It must prove one-run enforcement, atomic artifact plus
   companion-hash publication, exact provenance binding, stage gating, and prevention
   of candidate entrypoint invocation.
3. Obtain fresh independent review of that infrastructure.
4. Only then may the Manager freeze a structurally distinct cross-layer/final-output
   successor. Planner and Engineer must not edit `research/PIPELINE_STATE.json`.
"""
    CHECKPOINT.write_text(text, encoding="utf-8")


def check_bound(terminal: dict[str, Any]) -> None:
    public = load(PUBLIC)
    policy = load(POLICY)
    scope = load(SCOPE)
    require(public.get("latest_decision") == DECISION, "public decision is stale")
    require(public.get("required_manager_action") == MANAGER_ACTION, "public Manager action is stale")
    require(public.get("routing_status") == ROUTING_STATUS, "public routing is stale")
    require(public.get("terminal_baseline_no_go") == terminal, "public NO_GO binding differs")
    require(public.get("integrity", {}).get("canonical_sha256") == canonical_sha256(public), "public canonical hash differs")
    require(policy.get("manager_recommendation") == MANAGER_ACTION, "policy Manager action is stale")
    gate = policy.get("baseline_harness_hardening_gate", {})
    require(gate.get("candidate_independent") is True, "harness gate is not candidate-independent")
    require(gate.get("candidate_entrypoint_permitted") is False, "harness gate permits candidate entrypoint")
    require(scope.get("stage", {}).get("current_stage_status") == ROUTING_STATUS, "scope routing is stale")
    require(scope.get("implementation_frontier", {}).get("terminal_baseline_no_go") == terminal, "scope NO_GO binding differs")
    checkpoint = CHECKPOINT.read_text(encoding="utf-8")
    require("Do not rerun the baseline" in checkpoint, "checkpoint rerun prohibition missing")
    require("Manager rolls `rtl` back to `architecture`" in checkpoint, "checkpoint Manager action missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    no_go, review, _ = validate_evidence()
    terminal = disposition(no_go, review)
    if args.check:
        check_bound(terminal)
        print(
            "ACE2_QECR_BASELINE_NO_GO_STATUS_CHECK_PASS "
            f"no_go_sha256={sha256(NO_GO)} review_sha256={sha256(REVIEW)} current_stage=rtl"
        )
        return 0
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    update_public(terminal, now)
    update_policy_and_scope(terminal, now)
    update_checkpoint(terminal)
    check_bound(terminal)
    print(
        "ACE2_QECR_BASELINE_NO_GO_STATUS_BIND_PASS "
        f"no_go_sha256={sha256(NO_GO)} review_sha256={sha256(REVIEW)} current_stage=rtl"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
