#!/usr/bin/env python3
"""Run and bind one independent RTL-checklist review for dynamic Scale32."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
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
CONTRACT = "shared_token_group_dynamic_scale32_v1"
RTL = ROOT / "rtl/ace2_dynamic_scale32_core.sv"
PRECHECK = ROOT / f"evidence/{CONTRACT}/rtl/latest/PRECHECK.json"
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
TRACE = ROOT / "design/RTL_TRACEABILITY.md"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
OUT = ROOT / f"evidence/review/rtl_checklist_{CONTRACT}/decision.json"
INPUTS = OUT.parent / "inputs"
MISSION_ID = "rtl-checklist-shared-token-group-dynamic-scale32-v1"
FINAL_STATUS = "independent_rtl_checklist_accepted_stage_closing_false"
CHECKLIST = {
    "rtl.contract-traceability": True,
    "rtl.hardware-discipline": True,
    "rtl.ip-provenance": True,
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def require_artifact(record: dict[str, Any]) -> None:
    path = ROOT / record["path"]
    require(path.is_file(), f"review artifact missing: {record['path']}")
    require(path.stat().st_size == record["bytes"], f"review artifact size changed: {record['path']}")
    require(sha256(path) == record["sha256"], f"review artifact hash changed: {record['path']}")


def session_id() -> str:
    configured = os.environ.get("ARGUS_SKILL_SESSION_ID")
    if configured:
        return configured
    session = ROOT / ".ace2-session.json"
    if session.is_file():
        value = load(session).get("sid")
        if isinstance(value, str) and value:
            return value
    return ROOT.name


def reviewer_accepts(reason: str, identifier: str) -> bool:
    normalized = reason.lower().replace("`", "")
    return any(
        token in normalized
        for token in (
            f"{identifier}: supported",
            f"{identifier}: true",
            f"{identifier}=true",
        )
    )


def review_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    if OUT.exists():
        prior = load(OUT)
        require(prior.get("reviewer_status") != "done", "accepted independent decision already exists")
        archive = OUT.parent / "archive" / f"decision.{sha256(OUT)}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            require(archive.read_bytes() == OUT.read_bytes(), "review archive collision")
        else:
            shutil.copy2(OUT, archive)
        OUT.unlink()
    packet = load(PRECHECK)
    manifest = load(MANIFEST)
    pipeline = load(PIPELINE)
    require(pipeline.get("current_stage") == "rtl", "Manager-owned current_stage is not rtl")
    require(packet.get("contract_id") == CONTRACT, "precheck contract mismatch")
    require(packet.get("status") == "bounded_rtl_preflight_pass_independent_review_pending",
            "precheck is not ready for independent review")
    require(packet.get("stage_checklist") == CHECKLIST, "precheck checklist differs")
    require(packet.get("stage_closing") is False, "precheck changed stage_closing=false")
    require(manifest.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"),
            "manifest/precheck candidate hash mismatch")
    require(manifest.get("traceability", {}).get("stage_checklist") == CHECKLIST,
            "manifest checklist differs")
    for source in packet.get("source_artifacts", []):
        require_artifact(source)
    for record in packet.get("frozen_dependencies", {}).values():
        require_artifact(record)

    INPUTS.mkdir(parents=True, exist_ok=True)
    manifest_snapshot = INPUTS / f"RTL_MANIFEST.pending.{sha256(MANIFEST)}.json"
    trace_snapshot = INPUTS / f"RTL_TRACEABILITY.pending.{sha256(TRACE)}.md"
    if not manifest_snapshot.exists():
        shutil.copy2(MANIFEST, manifest_snapshot)
    else:
        require(manifest_snapshot.read_bytes() == MANIFEST.read_bytes(), "manifest snapshot collision")
    if not trace_snapshot.exists():
        shutil.copy2(TRACE, trace_snapshot)
    else:
        require(trace_snapshot.read_bytes() == TRACE.read_bytes(), "traceability snapshot collision")
    verified = {
        "precheck": artifact(PRECHECK),
        "manifest": artifact(manifest_snapshot),
        "traceability": artifact(trace_snapshot),
        "pipeline_state": artifact(PIPELINE),
        "rtl_source": artifact(RTL),
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "source_artifacts": packet["source_artifacts"],
        "frozen_dependencies": packet["frozen_dependencies"],
        "checks": packet["checks"],
        "stage_checklist": CHECKLIST,
        "claim_boundary": packet["claim_boundary"],
    }
    return packet, verified


def bind_done(payload: dict[str, Any]) -> None:
    packet = load(PRECHECK)
    decision = artifact(OUT)
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    manifest = load(MANIFEST)
    manifest["candidate_status"] = FINAL_STATUS
    manifest["status"] = FINAL_STATUS
    manifest["stage_closing"] = False
    manifest["interfaces_contract_status"] = "dynamic_scale32_standalone_interfaces_independently_reviewed"
    manifest["candidate_review_binding"] = {
        "candidate_capability_accepted": False,
        "decision": "accepted_rtl_checklist_only",
        "reviewer_status": "done",
        "evidence": decision,
        "scope": "standalone_dynamic_scale32_rtl_checklist_only",
        "stage_closing": False,
        "quality_discriminator_complete": False,
        "shell_admitted": False,
        "ppa_run": False,
        "checklist_rulings": copy.deepcopy(payload["checklist_rulings"]),
    }
    manifest["independent_reviewer_acceptance"] = {
        "architecture_review": "accepted",
        "environment_review": "accepted_manager_advanced_to_rtl",
        "rtl_review": "accepted_exact_candidate_hash",
        "status": FINAL_STATUS,
    }
    manifest["proposed_replacement_contract"].update({
        "status": FINAL_STATUS,
        "rtl_review": decision,
        "required_manager_action": "none_current_rtl_task_stage_closing_false",
        "updated_at_utc": now,
    })
    manifest["required_manager_action"] = "none_current_rtl_task_stage_closing_false"
    dump(MANIFEST, manifest)

    trace = TRACE.read_text(encoding="utf-8")
    trace = trace.replace(
        "Candidate capability acceptance is\nfalse pending independent RTL checklist review and later gated verification.\n",
        "Independent review accepts all three RTL checklist items for the exact\n"
        "candidate hash with `stage_closing=false`. Candidate capability acceptance\n"
        "remains false pending later gated verification.\n",
    )
    TRACE.write_text(trace, encoding="utf-8")

    status = load(PUBLIC)
    latest_decision = "shared_token_group_dynamic_scale32_v1_rtl_checklist_independently_accepted"
    status["latest_decision"] = latest_decision
    status["stage_closing"] = False
    status["required_manager_action"] = "none_current_rtl_task_stage_closing_false"
    status["routing_authorized"] = "rtl_stage_only_non_stage_closing_review_complete"
    status["routing_status"] = "rtl_checklist_independently_accepted_downstream_still_locked"
    status["blockers"] = []
    status["next_action_queue"] = []
    status["last_updated_utc"] = now
    status["generated_at_utc"] = now
    status["stage"]["current_stage"] = "rtl"
    status["stage"]["current_stage_status"] = FINAL_STATUS
    status["stage"]["current_stage_checklist"] = copy.deepcopy(CHECKLIST)
    status["stage"]["stage_closing"] = False
    if decision["path"] not in status["stage"]["current_stage_evidence"]:
        status["stage"]["current_stage_evidence"].append(decision["path"])
    status["latest_rtl_candidate"].update({
        "status": FINAL_STATUS,
        "rtl_review": decision,
        "candidate_capability_accepted": False,
        "stage_closing": False,
    })
    status["dashboard_fields"].update({
        "latest_decision": latest_decision,
        "rtl_review": decision,
        "rtl_candidate": {
            "contract_id": CONTRACT,
            "rtl_hash": packet["candidate_rtl_hash"],
            "status": FINAL_STATUS,
            "candidate_capability_accepted": False,
            "stage_closing": False,
        },
    })
    dump(PUBLIC, status)

    CHECKPOINT.write_text(
        f"""# Goal

Implement and independently review the exact Manager-frozen `{CONTRACT}` standalone RTL primitives while holding `current_stage=rtl` and `stage_closing=false`.

# Current state

Independent review accepted `rtl.contract-traceability`, `rtl.hardware-discipline`, and `rtl.ip-provenance` for candidate hash `{packet['candidate_rtl_hash']}`. The reviewed implementation contains the 64/128-lane dynamic group finalizer with exact 1,280-byte ping-pong storage, the exact 64-byte `BFP1` sidecar builder/validator, and the signed-160 exact Scale32-tagged accumulator.

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains `layer_0.rope_q`; mode remains `ADVANCE`. The 2.0 mm2 non-SRAM cap, >=100 MHz floor, 128-bit abstract streaming-memory boundary, and historical PPA frontier are unchanged.

# Evidence

- `design/RTL_MANIFEST.json`
- `design/RTL_TRACEABILITY.md`
- `evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/PRECHECK.json`
- `{decision['path']}`

# Boundary

The independent ruling is RTL-checklist-only and `stage_closing=false`. Candidate capability acceptance remains false. No Manager stage transition is requested. Verification, baseline/model execution, shell admission/full regression, canonical PPA, physical design, prototype, benchmark, signoff, tapeout-readiness, and silicon remain unrun and locked. Updated at `{now}`.
""",
        encoding="utf-8",
    )


def check_existing() -> None:
    decision = load(OUT)
    packet = load(PRECHECK)
    require(decision.get("reviewer_status") == "done", "independent RTL review is not done")
    require(decision.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"), "review hash mismatch")
    require(decision.get("checklist") == CHECKLIST, "review checklist differs")
    require(decision.get("stage_closing") is False, "review changed stage_closing=false")
    require(decision.get("required_remediation") == "", "accepted review retains remediation")
    require(all(decision.get("checklist_rulings", {}).get(item, {}).get("accepted") for item in CHECKLIST),
            "one or more RTL checklist rulings are not accepted")
    for key in ("precheck", "manifest", "traceability", "pipeline_state", "rtl_source"):
        require_artifact(decision["verified_evidence"][key])
    manifest = load(MANIFEST)
    require(manifest.get("candidate_status") == FINAL_STATUS, "manifest review status is stale")
    require(manifest.get("candidate_review_binding", {}).get("evidence") == artifact(OUT),
            "manifest does not bind the current review")
    public = load(PUBLIC)
    require(public.get("latest_rtl_candidate", {}).get("status") == FINAL_STATUS,
            "public RTL review status is stale")


def main() -> int:
    if "--check" in sys.argv[1:]:
        check_existing()
        print("ACE2_DYNAMIC_SCALE32_RTL_REVIEW_CHECK_PASS stage_closing=false")
        return 0

    packet, verified = review_inputs()
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
        f"Independently inspect the hash-bound standalone RTL preflight for {CONTRACT}. "
        "Decide only whether rtl.contract-traceability, rtl.hardware-discipline, and "
        "rtl.ip-provenance are supported for the exact candidate hash. Keep "
        "candidate_capability_accepted=false and stage_closing=false. Do not authorize "
        "verification, shell admission, PPA, prototype, benchmark, signoff, tapeout, or silicon."
    )
    instruction = (
        "Read the PRECHECK, pending RTL_MANIFEST snapshot, pending RTL_TRACEABILITY snapshot, "
        "live RTL source, explicit DYNAMIC_SCALE32_IP_PROVENANCE declaration, frozen "
        "proposal/freeze/environment bindings, and recorded logs directly. "
        "Use only deterministic content/hash inspection; do not rerun generation, simulation, "
        "formal, synthesis, PPA, quality, or model execution. The Planner direct-execution producer "
        "identity is authorized by the operator and must not be rejected for lacking an Engineer "
        "ledger. Return done only if every synthesizable module exactly traces to the frozen contract, "
        "widths/signedness/reset/backpressure/arrays/counters are disciplined (including the saturated "
        "38-event bound), and every first-party or generated source has pinned hashes plus explicit "
        "project-internal provenance and regeneration metadata. The project honestly claims no external "
        "redistribution grant and introduces no third-party IP; do not require the reviewer to choose a "
        "public license on the operator's behalf. In the reason, "
        "state each exact ruling as '<identifier>: supported'. If accepted, remediation must be empty "
        "and the reason must explicitly state stage_closing=false. Otherwise return continue with "
        "concrete remediation; this review never advances the Manager-owned stage."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=sid,
        main_summary="Planner directly implemented and hash-bound the frozen dynamic Scale32 standalone primitives; focused preflight passes and no downstream run occurred.",
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="standalone_dynamic_scale32_rtl_checklist_only_stage_closing_false",
        checkpoint_path=str(CHECKPOINT),
        escalate_hint="If accepted, current_stage remains rtl, stage_closing remains false, candidate capability remains unaccepted, and all downstream work stays locked.",
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
    reason = raw.get("reason", "")
    remediation = raw.get("required_remediation", "")
    if status == "done":
        require(all(reviewer_accepts(reason, item) for item in CHECKLIST),
                "Reviewer reason lacks one or more exact supported rulings")
        require("stage_closing=false" in reason.lower().replace("`", "").replace(" ", ""),
                "Reviewer reason does not preserve stage_closing=false")
        require(remediation == "", "Reviewer returned done with remediation")
    rulings = {
        item: {
            "accepted": status == "done" and reviewer_accepts(reason, item),
            "reviewer_reason_uses_exact_identifier": item in reason,
        }
        for item in CHECKLIST
    }
    payload = {
        "schema_version": 1,
        "review_bound_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "reviewer_status": status,
        "contract_id": CONTRACT,
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "live_rtl_source_sha256": sha256(RTL),
        "checklist": CHECKLIST if status == "done" else raw.get("checklist", {}),
        "checklist_rulings": rulings,
        "stage_closing": False,
        "scope": "standalone_dynamic_scale32_rtl_checklist_only",
        "candidate_capability_accepted": False,
        "verified_evidence": verified,
        "producer_provenance": copy.deepcopy(load(MANIFEST).get("candidate_implementation_provenance", {})),
        "reason": reason,
        "required_remediation": remediation,
        "reviewer_thread_id": raw.get("thread_id", ""),
        "raw_decision": raw,
    }
    dump(OUT, payload)
    if status == "done":
        bind_done(payload)
        print("ACE2_DYNAMIC_SCALE32_RTL_REVIEW_PASS stage_closing=false")
        return 0
    print(f"ACE2_DYNAMIC_SCALE32_RTL_REVIEW_REPLAN remediation={remediation}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
