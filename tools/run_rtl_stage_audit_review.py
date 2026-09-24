#!/usr/bin/env python3
"""Run one independent review of a frozen current-tree RTL-stage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
CHECKLIST = (
    "rtl.spec-traceability",
    "rtl.synthesizable-discipline",
    "rtl.width-reset-parameters",
)
MISSION_ID = "current-tree-rtl-stage-contract-review-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


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


def verify_manifest(audit_dir: Path) -> None:
    manifest = audit_dir / "SHA256SUMS"
    companion = audit_dir / "SHA256SUMS.sha256"
    require(manifest.is_file() and companion.is_file(), "audit SHA-256 manifests are missing")
    expected_companion = f"{sha256(manifest)}  SHA256SUMS\n"
    require(companion.read_text(encoding="utf-8") == expected_companion,
            "audit aggregate manifest companion differs")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = audit_dir / name
        require(path.is_file(), f"audit artifact missing: {name}")
        require(sha256(path) == digest, f"audit artifact changed: {name}")


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


def verify_live_bindings(results: dict[str, Any]) -> None:
    bindings = results.get("audit_input_bindings", {})
    require(isinstance(bindings, dict) and bindings, "audit input bindings are missing")
    for relative, expected in bindings.items():
        path = ROOT / relative
        require(path.is_file(), f"bound live input is missing: {relative}")
        require(sha256(path) == expected, f"bound live input changed: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--checkpoint-free", action="store_true")
    parser.add_argument("--mission-id", default=MISSION_ID)
    args = parser.parse_args()

    audit_dir = args.audit_dir if args.audit_dir.is_absolute() else ROOT / args.audit_dir
    output = args.output if args.output.is_absolute() else ROOT / args.output
    results_path = audit_dir / "results.json"

    if args.check:
        decision = load_json(output)
        require(decision.get("reviewer_status") == "done", "RTL-stage review is not done")
        require(all(decision.get("checklist_rulings", {}).get(item) is True for item in CHECKLIST),
                "RTL-stage review does not accept every checklist item")
        require(decision.get("manager_stage_edited") is False,
                "review incorrectly reports a Manager-owned stage edit")
        verify_manifest(audit_dir)
        verify_live_bindings(load_json(results_path))
        print("ACE2_RTL_STAGE_REVIEW_CHECK_PASS")
        return 0

    require(not output.exists(), f"review output already exists: {output.relative_to(ROOT)}")
    verify_manifest(audit_dir)
    results = load_json(results_path)
    interface = load_json(audit_dir / "interface_audit.json")
    pipeline = load_json(PIPELINE)
    require(results.get("status") == "PASS", "RTL-stage audit did not pass")
    require(interface.get("status") == "PASS", "interface audit did not pass")
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(results.get("checks", {}).get("stage_gate", {}).get("status") == "PASS",
            "audit stage gate did not pass")
    verify_live_bindings(results)

    verified = {
        "audit_results": artifact(results_path),
        "interface_audit": artifact(audit_dir / "interface_audit.json"),
        "commands": artifact(audit_dir / "commands.jsonl"),
        "iverilog_elaboration_log": artifact(audit_dir / "iverilog_elaboration.log"),
        "verilator_shell_lint_log": artifact(audit_dir / "verilator_shell_lint.log"),
        "verilator_shell_hierarchy_log": artifact(
            audit_dir / "verilator_shell_hierarchy.log"
        ),
        "verilator_shell_hierarchy_xml": artifact(
            audit_dir / "ace2_shell_hierarchy.xml"
        ),
        "verilator_dynamic_scale32_lint_log": artifact(
            audit_dir / "verilator_dynamic_scale32_lint.log"
        ),
        "verilator_repository_lint_log": artifact(
            audit_dir / "verilator_repository_rtl_lint.log"
        ),
        "aggregate_manifest": artifact(audit_dir / "SHA256SUMS"),
        "aggregate_manifest_companion": artifact(audit_dir / "SHA256SUMS.sha256"),
        "pipeline_state": artifact(PIPELINE),
        "active_rtl_input_closure_manifest_sha256": results[
            "active_rtl_input_closure_manifest_sha256"
        ],
        "rtl_sv_compatibility_manifest_sha256": results[
            "rtl_sv_compatibility_manifest_sha256"
        ],
    }

    backend_name = resolve_role_backend("reviewer")
    runner = AgentCliBackend(
        backend=backend_name,
        runner_bin=resolve_runner_bin_setting("reviewer") or None,
    )
    sid = session_id()
    runner.set_usage_context(
        project_root=global_root() / "projects" / sid,
        global_root=global_root(),
        mission_id=args.mission_id,
    )
    package_root = Path(argus_skill.__file__).resolve().parent
    reviewer_skill = (
        package_root / "verticals/chip_design/skills/reviewer/chip-design-signoff-review.md"
    ).read_text(encoding="utf-8")
    closure = results["active_rtl_input_closure_manifest_sha256"]
    objective = (
        "Independently review the frozen live-tree RTL-stage audit for ace2_shell. "
        "Rule only on the three current RTL checklist items for exact active input "
        f"closure {closure}. Do not run simulation, formal, synthesis, timing, PPA, "
        "runtime, benchmark, FPGA, U280, or model execution, and do not edit the "
        "Manager-owned pipeline state."
    )
    instruction = (
        "Read research/RTL_STAGE_AUDIT.md, research/GROUND_TRUTH.md, design/SPEC.md, "
        "design/BENCHMARK_INTERFACE.json, design/RTL_TRACEABILITY.md, the live RTL sources, "
        f"and {audit_dir.relative_to(ROOT).as_posix()}/results.json plus its recorded lint "
        "and elaboration logs. Use deterministic read-only hash/content checks. Return done "
        "only if every live state/datapath/interface response is traceable to the frozen spec, "
        "the exact 12-parameter/64-port public contract is preserved, synthesizable discipline "
        "is supported without active latch/multiple-driver/CDC/width findings, and width, signedness, "
        "reset, counters, arrays, and frozen parameter edge conditions are intentional. Historical "
        "or rejected diagnostic RTL outside the active shell closure must not be mistaken for active "
        "product RTL. In the reason, state each accepted ruling exactly as "
        "'<identifier>: supported'. If any item is unsupported, return continue with concrete, "
        "evidence-backed remediation. This review may recommend the RTL checklist as stage-ready, "
        "but it must not edit research/PIPELINE_STATE.json or claim product, verification, synthesis, "
        "timing, demo, or U280 completion. Do not modify any workspace file."
    )
    if args.checkpoint_free:
        instruction += (
            " This is a repair review after an otherwise-supportive read-only review "
            "was blocked only because a checkpoint path was supplied. No checkpoint "
            "write is requested or permitted in this review; return the checklist "
            "verdict directly from the frozen evidence."
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
            "The Planner directly audited the exact current RTL closure after the Manager restored "
            "current_stage=rtl; interface, lint, and elaboration checks pass, and no downstream run occurred."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="current_tree_rtl_stage_checklist_only",
        checkpoint_path="" if args.checkpoint_free else str(CHECKPOINT),
        escalate_hint=(
            "If accepted, report the RTL checklist as stage-ready while preserving the independent "
            "Option-B numerical-policy blocker and all downstream non-claims."
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
            sandbox_mode="read-only",
            isolate_workdir=False,
            working_dir=str(ROOT),
        ),
    )

    raw = asdict(review)
    status = raw.get("status", "continue")
    reason = raw.get("reason", "")
    rulings = {
        item: status == "done" and reviewer_accepts(reason, item) for item in CHECKLIST
    }
    if status == "done":
        require(all(rulings.values()), "Reviewer reason lacks one or more exact supported rulings")
        require(raw.get("required_remediation", "") == "",
                "Reviewer returned done with remediation")
    payload = {
        "schema_version": 1,
        "review_bound_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "reviewer_status": status,
        "scope": "current_tree_rtl_stage_checklist_only",
        "active_rtl_input_closure_manifest_sha256": closure,
        "rtl_sv_compatibility_manifest_sha256": results[
            "rtl_sv_compatibility_manifest_sha256"
        ],
        "checklist_rulings": rulings,
        "stage_ready_recommendation": status == "done" and all(rulings.values()),
        "manager_stage_edited": False,
        "product_complete": False,
        "downstream_execution_performed": False,
        "verified_evidence": verified,
        "reason": reason,
        "required_remediation": raw.get("required_remediation", ""),
        "reviewer_thread_id": raw.get("thread_id", ""),
        "raw_decision": raw,
        "review_invocation": {
            "checkpoint_free": args.checkpoint_free,
            "mission_id": args.mission_id,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(
        f"ACE2_RTL_STAGE_REVIEW_RESULT status={status} "
        f"stage_ready={payload['stage_ready_recommendation']} output={output.relative_to(ROOT)}"
    )
    return 0 if payload["stage_ready_recommendation"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
