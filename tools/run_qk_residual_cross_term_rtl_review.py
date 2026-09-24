#!/usr/bin/env python3
"""Run one independent RTL-checklist review for the residual cross-term candidate."""

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
CONTRACT = "shared_qk_residual_cross_term_attention_v1"
RTL = "rtl/ace2_qk_residual_cross_term_core.sv"
PACKET = ROOT / "evidence/shared_qk_residual_cross_term_attention_v1/latest/PRECHECK.json"
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
OUT = ROOT / "evidence/review/rtl_checklist_shared_qk_residual_cross_term_attention_v1/decision.json"
MISSION_ID = "rtl-checklist-shared-qk-residual-cross-term-attention-v1"
CHECKLIST = {
    "rtl.contract-traceability": True,
    "rtl.hardware-discipline": True,
    "rtl.ip-provenance": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
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


def verify_record(record: dict[str, Any], label: str) -> dict[str, Any]:
    relative = str(record.get("path") or "")
    path = ROOT / relative
    require(relative != "" and path.is_file(), f"{label}: missing artifact {relative}")
    require(path.stat().st_size == int(record.get("bytes", -1)), f"{label}: byte-size mismatch")
    require(sha256_file(path) == str(record.get("sha256") or ""), f"{label}: SHA-256 mismatch")
    return artifact(path)


def session_id() -> str:
    configured = os.environ.get("ARGUS_SKILL_SESSION_ID")
    if configured:
        return configured
    session_file = ROOT / ".ace2-session.json"
    if session_file.is_file():
        value = load_json(session_file).get("sid")
        if isinstance(value, str) and value:
            return value
    return ROOT.name


def preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    require(not OUT.exists(), f"review already exists: {OUT.relative_to(ROOT)}")
    packet = load_json(PACKET)
    manifest = load_json(MANIFEST)
    pipeline = load_json(PIPELINE)

    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(packet.get("contract_id") == CONTRACT, "preflight contract mismatch")
    require(packet.get("status") == "pass_ready_for_independent_rtl_review", "preflight is not ready")
    require(packet.get("stage_closing") is False, "preflight unexpectedly closes the stage")
    require(all(packet.get("checklist", {}).values()), "preflight checklist is incomplete")
    require(not any(packet.get("prohibited_runs", {}).values()), "a prohibited downstream run is recorded")
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
            "manifest RTL checklist differs from the stage contract")

    verified = {
        "precheck": artifact(PACKET),
        "manifest": artifact(MANIFEST),
        "pipeline_state": artifact(PIPELINE),
        "rtl_source": artifact(ROOT / RTL),
        "logs": {
            name: verify_record(record, f"logs.{name}")
            for name, record in sorted(packet.get("logs", {}).items())
        },
        "interface_xml": {
            name: verify_record(record, f"interface_xml.{name}")
            for name, record in sorted(packet.get("interface_xml", {}).items())
        },
        "candidate_rtl_hash": packet["candidate_rtl_hash"],
        "candidate_id": packet["candidate_id"],
        "source_hashes": source_hashes,
        "stage_checklist": CHECKLIST,
        "prohibited_runs": packet["prohibited_runs"],
    }
    return packet, verified


def check_existing() -> None:
    decision = load_json(OUT)
    packet = load_json(PACKET)
    require(decision.get("reviewer_status") == "done", "independent RTL review is not done")
    require(decision.get("contract_id") == CONTRACT, "review contract mismatch")
    require(decision.get("candidate_rtl_hash") == packet.get("candidate_rtl_hash"),
            "review candidate hash mismatch")
    require(decision.get("live_rtl_source_sha256") == sha256_file(ROOT / RTL),
            "review live RTL hash mismatch")
    require(decision.get("checklist") == CHECKLIST, "review checklist is incomplete")
    require(decision.get("stage_closing") is False, "review unexpectedly closes the stage")
    print(f"QK_RESIDUAL_RTL_REVIEW_CHECK_PASS candidate={decision['candidate_id']}")


def main() -> None:
    if "--check" in sys.argv[1:]:
        check_existing()
        return

    packet, verified = preflight()
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
        "shared_qk_residual_cross_term_attention_v1. Decide only whether the three "
        "current RTL checklist items are supported for the exact candidate hash. "
        "Do not accept the candidate as model capability, close the RTL stage, or "
        "authorize quality, shell, PPA, prototype, benchmark, signoff, tapeout, or silicon claims."
    )
    instruction = (
        "Read design/NUMERICAL_REPLACEMENT_PROPOSAL.md, design/RTL_MANIFEST.json, "
        "design/RTL_TRACEABILITY.md, rtl/ace2_qk_residual_cross_term_core.sv, and "
        "evidence/shared_qk_residual_cross_term_attention_v1/latest/PRECHECK.json directly. "
        "Use only short deterministic hash/content checks and the already-recorded lint, "
        "elaboration, simulation, and interface XML. Do not modify any file except CHECKPOINT.md, "
        "which must receive a concise verdict handoff. Do not rerun generation, "
        "simulation, quality evaluation, synthesis, STA, or PPA. Return done only if exact "
        "interface/parameter traceability, hardware discipline, and first-party/generated-IP "
        "provenance are all supported for candidate hash "
        f"{packet['candidate_rtl_hash']}; otherwise return continue with precise repairs."
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
            "Planner integrity preflight verified the packet canonical hash, every bound source, "
            "the aggregate candidate hash, all recorded logs/XML, the Manager-owned RTL stage, "
            "and that no downstream run is recorded. Reviewer must inspect the design evidence independently."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="standalone_rtl_checklist_only_stage_closing_false",
        checkpoint_path=str(CHECKPOINT),
        escalate_hint=(
            "If done, the next in-stage gate is the frozen all-layer focused quality discriminator. "
            "The accepted prefix and historical PPA frontier remain unchanged."
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
        "live_rtl_source_sha256": sha256_file(ROOT / RTL),
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, OUT)
    print(
        f"QK_RESIDUAL_RTL_REVIEW_RESULT status={status} "
        f"candidate={packet['candidate_id']} decision={OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
