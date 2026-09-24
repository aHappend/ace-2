#!/usr/bin/env python3
"""Run and bind independent RTL-checklist review for the V-residual candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_v_residual_value_correction_attention_v1"
RTL = ROOT / "rtl/ace2_v_residual_value_correction_core.sv"
PACKET = ROOT / "evidence/shared_v_residual_value_correction_attention_v1/latest/PRECHECK.json"
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
OUT = ROOT / "evidence/review/rtl_checklist_shared_v_residual_value_correction_attention_v1/decision.json"
MISSION_ID = "rtl-checklist-shared-v-residual-value-correction-attention-v1"
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path | str) -> dict[str, Any]:
    resolved = path if isinstance(path, Path) else ROOT / path
    value = json.loads(resolved.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {resolved}")
    return value


def dump(path: Path | str, value: dict[str, Any]) -> None:
    resolved = path if isinstance(path, Path) else ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path | str) -> dict[str, Any]:
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


def verify_preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    require(not OUT.exists(), f"review already exists: {OUT.relative_to(ROOT)}")
    packet = load(PACKET)
    manifest = load(MANIFEST)
    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(packet.get("contract_id") == CONTRACT, "preflight contract mismatch")
    require(packet.get("status") == "pass_ready_for_independent_rtl_review", "preflight not ready")
    require(packet.get("stage_closing") is False, "preflight unexpectedly closes RTL")
    require(all(packet.get("checklist", {}).values()), "preflight checklist incomplete")
    require(packet.get("stage_checklist") == CHECKLIST, "preflight RTL checklist differs")
    require(not any(packet.get("prohibited_runs", {}).values()), "a downstream run is recorded")
    require(canonical_sha256(packet) == packet.get("integrity", {}).get("canonical_sha256"),
            "preflight canonical hash mismatch")
    source_hashes = packet.get("source_hashes", {})
    require(isinstance(source_hashes, dict) and source_hashes, "preflight source hashes missing")
    for relative, expected in source_hashes.items():
        path = ROOT / relative
        require(path.is_file(), f"missing bound source: {relative}")
        require(sha256_file(path) == expected, f"bound source changed: {relative}")
    require(aggregate_hash(source_hashes) == packet.get("candidate_rtl_hash"),
            "candidate aggregate hash mismatch")
    require(manifest.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"),
            "manifest candidate hash mismatch")
    require(manifest.get("candidate_interface", {}).get("modules"), "manifest interfaces missing")
    require(manifest.get("traceability", {}).get("stage_checklist") == CHECKLIST,
            "manifest checklist differs")
    verified = {
        "precheck": artifact(PACKET),
        "manifest": artifact(MANIFEST),
        "pipeline_state": artifact(PIPELINE),
        "rtl_source": artifact(RTL),
        "candidate_id": packet["candidate_id"],
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "source_hashes": source_hashes,
        "stage_checklist": CHECKLIST,
        "logs": packet["logs"],
        "interface_xml": packet["interface_xml"],
        "coverage": packet["coverage"],
        "schedule_contract": packet["schedule_contract"],
        "prohibited_runs": packet["prohibited_runs"],
    }
    return packet, verified


def bind_done(payload: dict[str, Any]) -> None:
    decision = artifact(OUT)
    packet = load(PACKET)
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    policy = load("design/FAST_LOOP_POLICY.json")
    for key in (
        "active_repair_authorization",
        "architecture_proposal_authorization",
        "selected_replacement_contract",
    ):
        policy[key]["required_manager_action"] = "advance_rtl_to_verification_for_focused_discriminator"
        policy[key]["status"] = "authorization_consumed_rtl_checklist_accepted"
    policy["manager_recommendation"] = "advance_rtl_to_verification_for_focused_discriminator"

    manifest = load(MANIFEST)
    require(payload.get("candidate_rtl_hash") == manifest.get("candidate_rtl_hash"),
            "review and manifest candidate hashes differ")
    manifest["architecture_selection_evidence"]["status"] = (
        "accepted_architecture_and_environment_bound_candidate_rtl_implemented"
    )
    manifest["candidate_evidence_hashes"]["architecture_review_scope"] = (
        "manager_advanced_after_fresh_independent_architecture_acceptance"
    )
    manifest["candidate_status"] = "independent_rtl_checklist_accepted"
    manifest["candidate_review_binding"] = {
        "candidate_capability_accepted": False,
        "decision": "accepted_rtl_checklist_only",
        "reviewer_status": "done",
        "evidence": decision,
        "scope": "standalone_candidate_rtl_checklist_only",
        "stage_closing": False,
        "quality_discriminator_complete": False,
        "shell_admitted": False,
        "ppa_run": False,
    }
    manifest["traceability"]["stage_checklist"] = copy.deepcopy(CHECKLIST)
    manifest["traceability"].update(CHECKLIST)
    manifest["traceability"]["architecture_contract_gap"] = {
        "status": "closed_for_rtl_checklist",
        "resolution_owner": "independent_reviewer_done",
    }
    manifest["independent_reviewer_acceptance"]["environment_review"] = (
        "manager_advanced_after_fresh_independent_environment_acceptance"
    )
    manifest["independent_reviewer_acceptance"]["rtl_review"] = (
        "accepted_exact_candidate_hash"
    )
    manifest["independent_reviewer_acceptance"]["status"] = (
        "rtl_checklist_accepted_waiting_manager_advance"
    )
    manifest["interfaces_contract_status"] = (
        "v_residual_value_path_interface_implemented_and_rtl_checklist_accepted"
    )
    manifest["proposed_replacement_contract"] = copy.deepcopy(
        policy["selected_replacement_contract"]
    )
    dump(MANIFEST, manifest)

    dump("design/FAST_LOOP_POLICY.json", policy)

    target = load("design/TARGET.json")
    target["current_stage"] = "rtl"
    target["fast_loop_contract"]["manager_recommendation"] = policy["manager_recommendation"]
    target["fast_loop_contract"]["active_repair_authorization"] = copy.deepcopy(
        policy["active_repair_authorization"]
    )
    target["fast_loop_contract"]["architecture_proposal_authorization"] = copy.deepcopy(
        policy["architecture_proposal_authorization"]
    )
    target["fast_loop_contract"]["environment_readiness"] = (
        "accepted_for_v_residual_contract_no_dependency_delta"
    )
    target["current_architecture_contract"] = copy.deepcopy(policy["selected_replacement_contract"])
    target["generated_at_utc"] = now
    dump("design/TARGET.json", target)

    scope = load("design/CHIP_SCOPE.json")
    scope["stage"]["current_stage_status"] = "rtl_checklist_complete_waiting_manager_advance"
    scope["authority_override"]["required_next_action"] = policy["manager_recommendation"]
    scope["authority_override"]["operator_implementation_approval"] = copy.deepcopy(
        policy["selected_replacement_contract"]
    )
    scope["numerical_behavior"]["active_v_residual_value_correction_contract"]["status"] = (
        "rtl_checklist_accepted_waiting_manager_advance"
    )
    scope["numerical_behavior"]["attention_projection_scales"]["quality_status"] = (
        "architecture_accepted_candidate_quality_not_run"
    )
    scope["operator_owned_execution_policy"]["active_successor_contract"] = copy.deepcopy(
        policy["selected_replacement_contract"]
    )
    scope["implementation_frontier"]["latest_decision"] = "v_residual_rtl_checklist_independently_accepted"
    dump("design/CHIP_SCOPE.json", scope)

    trace_path = ROOT / "design/RTL_TRACEABILITY.md"
    trace = trace_path.read_text(encoding="utf-8")
    pending_review = (
        "  Independent review remains pending; the candidate is not shell-admitted and\n"
        "  has no focused quality, PPA, prototype, benchmark, or signoff claim."
    )
    accepted_review = (
        "  Independent review accepted all three RTL checklist items for the exact\n"
        "  candidate hash; the candidate is not shell-admitted and has no focused\n"
        "  quality, PPA, prototype, benchmark, or signoff claim."
    )
    if pending_review in trace:
        trace = trace.replace(pending_review, accepted_review)
    require(accepted_review in trace, "RTL traceability review status is not bindable")
    trace_path.write_text(trace, encoding="utf-8")

    checkpoint = f"""# Goal

Close the RTL checklist for `{CONTRACT}` without entering verification or PPA.

# Current State

The Manager-owned stage remains `rtl`. Independent review accepted all three RTL
checklist items for candidate `{packet['candidate_id']}` at exact ordered hash
`{packet['candidate_rtl_hash']}`. The accepted prefix remains through
`layer_0.v_proj`; first unsupported remains `layer_0.rope_q`; mode remains
`ADVANCE`.

# Verified Done

- Frozen 24-layer calibration source and 48-row Scale32 metadata are hash-bound.
- Exact integer reference, deterministic vectors, Icarus elaboration/simulation,
  warning-free Verilator lint, interface XML, reset/clear/backpressure/error
  coverage, and the eight-cycle conversion schedule pass.
- Independent Reviewer accepted contract traceability, hardware discipline, and
  first-party/generated-source provenance for the exact candidate hash.

# Locked / Not Run

Focused quality, paired smoke, shell admission/regression, canonical SKY130 PPA,
prototype, benchmark, signoff, tapeout, and silicon were not run.

# Next Required Action

Manager may advance `rtl -> verification` for the frozen all-24-layer focused
discriminator. Planner and Engineer must not edit `research/PIPELINE_STATE.json`.
"""
    CHECKPOINT.write_text(checkpoint, encoding="utf-8")

    status = load("research/PUBLIC_STATUS.json")
    status["latest_decision"] = "v_residual_rtl_checklist_independently_accepted"
    status["architecture_proposal_gate"] = {
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "required_manager_action": "advance_rtl_to_verification_for_focused_discriminator",
        "stage_closing": False,
        "status": "independent_rtl_checklist_accepted_waiting_manager_advance",
    }
    status["selected_replacement_contract"] = copy.deepcopy(
        policy["selected_replacement_contract"]
    )
    environment_status = {
        "contract_binding": CONTRACT,
        "implementation_authorized": False,
        "manager_advanced_to_rtl": True,
        "status": "accepted_manager_advanced_to_rtl_no_dependency_delta",
    }
    status["latest_environment_stage"] = copy.deepcopy(environment_status)
    status["stage"]["current_stage_status"] = "rtl_checklist_complete_waiting_manager_advance"
    status["stage"]["current_stage_checklist"] = copy.deepcopy(CHECKLIST)
    status["stage"]["current_stage_evidence"] = [
        "design/RTL_MANIFEST.json",
        "design/RTL_TRACEABILITY.md",
        PACKET.relative_to(ROOT).as_posix(),
        OUT.relative_to(ROOT).as_posix(),
    ]
    status["blockers"] = [
        {
            "id": "manager_rtl_to_verification_transition_pending",
            "stage": "rtl",
            "status": "active",
            "reason": "Independent RTL checklist review is complete; Manager owns the ordered stage transition.",
            "required_resolution": "Manager advances rtl to verification for the frozen focused discriminator.",
            "evidence": OUT.relative_to(ROOT).as_posix(),
        }
    ]
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status[name]
        container["current_stage"] = "rtl"
        container["latest_decision"] = status["latest_decision"]
        container["routing_status"] = "manager_rtl_to_verification_transition_pending"
        container["required_manager_action"] = (
            "advance_rtl_to_verification_for_focused_discriminator"
        )
        container["required_operator_action"] = "none"
        container["rtl_contract_traceability"] = True
        container["latest_environment_stage"] = copy.deepcopy(environment_status)
        container["candidate_rtl_hash_scope"] = "standalone_not_shell_admitted_no_candidate_ppa"
        container["operator_policy"] = copy.deepcopy(policy)
        if isinstance(container.get("candidate_mechanism"), dict):
            container["candidate_mechanism"]["status"] = (
                "rtl_checklist_accepted_waiting_manager_advance"
            )
        candidate = container.get("latest_rtl_candidate", {})
        candidate.update(
            {
                "candidate_id": packet["candidate_id"],
                "candidate_rtl_hash": packet["candidate_rtl_hash"],
                "contract_id": CONTRACT,
                "status": "independent_rtl_checklist_accepted",
                "review": decision,
                "stage_closing": False,
            }
        )
        container["latest_rtl_candidate"] = candidate
    status["latest_rtl_candidate"] = copy.deepcopy(
        status["dashboard_fields"]["latest_rtl_candidate"]
    )
    status["routing_authorized"] = "manager_advance_rtl_to_verification_pending"
    status["routing_status"] = "manager_rtl_to_verification_transition_pending"
    status["rtl_contract_traceability"] = True
    status["public_claims"] = [
        {
            "claim": "the Manager-owned current stage remains rtl",
            "evidence": ["research/PIPELINE_STATE.json"],
        },
        {
            "claim": "independent review accepted all three RTL checklist items for the exact V-residual candidate hash",
            "evidence": [OUT.relative_to(ROOT).as_posix(), PACKET.relative_to(ROOT).as_posix()],
        },
        {
            "claim": "the supported prefix, immutable targets, mode, and historical PPA frontier are unchanged",
            "evidence": ["design/CHIP_SCOPE.json", "design/TARGET.json"],
        },
        {
            "claim": "no focused quality, shell, candidate PPA, prototype, benchmark, signoff, tapeout, or silicon result exists",
            "evidence": [PACKET.relative_to(ROOT).as_posix()],
        },
    ]
    paths = {
        item.get("path")
        for item in status.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path") != "research/PUBLIC_STATUS.json"
    }
    paths.update(
        {
            "CHECKPOINT.md",
            "design/CHIP_SCOPE.json",
            "design/FAST_LOOP_POLICY.json",
            "design/RTL_MANIFEST.json",
            "design/TARGET.json",
            PACKET.relative_to(ROOT).as_posix(),
            OUT.relative_to(ROOT).as_posix(),
            "research/PIPELINE_STATE.json",
        }
    )
    status["artifact_hashes"] = [
        artifact(path) for path in sorted(paths) if path and (ROOT / path).is_file()
    ]
    status["generated_at_utc"] = now
    status["last_updated_utc"] = now
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump("research/PUBLIC_STATUS.json", status)


def main() -> int:
    if "--check" in sys.argv[1:]:
        decision = load(OUT)
        packet = load(PACKET)
        status = load("research/PUBLIC_STATUS.json")
        require(decision.get("reviewer_status") == "done", "independent review is not done")
        require(decision.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"),
                "review candidate hash mismatch")
        require(decision.get("checklist") == CHECKLIST, "review checklist incomplete")
        require(status.get("rtl_contract_traceability") is True,
                "public RTL traceability alias is stale")
        require(status.get("current_mode") == "ADVANCE", "public mode is stale")
        require(status.get("supported_layer_operator_prefix") == PREFIX,
                "public supported prefix is stale")
        require(status.get("first_unsupported_layer_operator") == FIRST_UNSUPPORTED,
                "public first unsupported operator is stale")
        require(status.get("stage", {}).get("current_stage_checklist") == CHECKLIST,
                "public stage checklist is stale")
        for name in ("dashboard_fields", "implementation_frontier"):
            container = status.get(name, {})
            require(container.get("rtl_contract_traceability") is True,
                    f"{name} RTL traceability alias is stale")
            require(container.get("required_manager_action") ==
                    "advance_rtl_to_verification_for_focused_discriminator",
                    f"{name} manager action is stale")
            require(container.get("supported_layer_operator_prefix") == PREFIX,
                    f"{name} supported prefix is stale")
            require(container.get("first_unsupported_layer_operator") == FIRST_UNSUPPORTED,
                    f"{name} first unsupported operator is stale")
            require(container.get("latest_ppa_frontier", {}).get("candidate_ppa_run") is False,
                    f"{name} candidate PPA boundary is stale")
        print(f"ACE2_V_RESIDUAL_RTL_REVIEW_CHECK_PASS candidate={decision['candidate_id']}")
        return 0

    if "--rebind" in sys.argv[1:]:
        decision = load(OUT)
        packet = load(PACKET)
        require(decision.get("reviewer_status") == "done", "independent review is not done")
        require(decision.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"),
                "review candidate hash mismatch")
        require(decision.get("checklist") == CHECKLIST, "review checklist incomplete")
        bind_done(decision)
        print(f"ACE2_V_RESIDUAL_RTL_REVIEW_REBIND_PASS candidate={decision['candidate_id']}")
        return 0

    packet, verified = verify_preflight()
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
        "Independently inspect the hash-bound standalone RTL preflight for "
        f"{CONTRACT}. Decide only whether the three current RTL checklist items "
        "are supported for the exact candidate hash. Do not accept model capability, "
        "close the RTL stage, or authorize quality, shell, PPA, prototype, benchmark, "
        "signoff, tapeout, or silicon claims."
    )
    instruction = (
        "Read design/NUMERICAL_REPLACEMENT_PROPOSAL.md, design/RTL_MANIFEST.json, "
        "design/RTL_TRACEABILITY.md, rtl/ace2_v_residual_value_correction_core.sv, "
        "tools/ace2_v_residual_value_correction_reference.py, the generated 48-row "
        "metadata, and evidence/shared_v_residual_value_correction_attention_v1/latest/PRECHECK.json. "
        "Inspect exact interfaces, widths, signedness, reset/clear/backpressure, bounded counters, "
        "reserved-code and overflow handling, eight-cycle conversion schedule, and provenance. "
        "Use only short deterministic checks and already-recorded logs/XML. Do not rerun calibration, "
        "simulation, quality, synthesis, STA, or PPA. Do not modify files except CHECKPOINT.md. "
        f"Return done only if all three RTL checklist items hold for hash {packet['candidate_rtl_hash']}; "
        "otherwise return continue with precise repairs."
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
            "Planner preflight verified the packet canonical hash, every bound source, aggregate "
            "candidate hash, interfaces, logs, Manager-owned RTL stage, and absence of downstream runs."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="standalone_rtl_checklist_only_stage_closing_false",
        checkpoint_path=str(CHECKPOINT),
        escalate_hint=(
            "If done, Manager may advance to verification for the frozen all-layer focused discriminator."
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
    payload = {
        "schema_version": 1,
        "review_bound_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "reviewer_status": status,
        "contract_id": CONTRACT,
        "candidate_id": packet["candidate_id"],
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "live_rtl_source_sha256": sha256_file(RTL),
        "scope": "standalone_candidate_rtl_checklist_only",
        "stage_closing": False,
        "checklist": CHECKLIST if status == "done" else {name: False for name in CHECKLIST},
        "reason": raw.get("reason", ""),
        "required_repairs": [] if status == "done" else [raw.get("reason", "review did not pass")],
        "quality_discriminator_complete": False,
        "shell_admitted": False,
        "ppa_run": False,
        "reviewer_thread_id": raw.get("thread_id", ""),
        "raw_decision": raw,
    }
    dump(OUT, payload)
    if status == "done":
        bind_done(payload)
    print(
        "ACE2_V_RESIDUAL_RTL_REVIEW_RESULT "
        f"status={status} candidate={packet['candidate_id']} decision={OUT.relative_to(ROOT)}"
    )
    return 0 if status == "done" else 2


if __name__ == "__main__":
    raise SystemExit(main())
