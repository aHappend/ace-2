#!/usr/bin/env python3
"""Run exactly one independent, read-only L2 review of the per-head Q/K no-go."""

from __future__ import annotations

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
RESULTS = ROOT / "evidence/per_head_qk_repair/latest/RESULTS.json"
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"
OUT_DIR = ROOT / "evidence/review/per_head_qk_no_go_l2"
OUT = OUT_DIR / "decision.json"
MISSION_ID = "per-head-qk-no-go-l2"
RECOVERED_BY_HASH = {
    "8d853ad49d9d869c4a5cc27ec282304a9517058fb36590ae2a97b6beb1ce78fd": (
        ROOT
        / "evidence/per_head_qk_repair/archive/"
        "rtl_lint.8d853ad49d9d869c4a5cc27ec282304a9517058fb36590ae2a97b6beb1ce78fd.log"
    )
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_artifact(record: dict[str, Any], label: str) -> dict[str, Any]:
    relative = str(record.get("path") or "")
    expected_hash = str(record.get("sha256") or "")
    expected_bytes = int(record.get("bytes") or -1)
    require(relative != "", f"{label}: missing path")
    require(expected_hash != "", f"{label}: missing sha256")
    path = ROOT / relative
    require(path.is_file(), f"{label}: missing artifact {relative}")
    actual_hash = sha256_file(path)
    if path.stat().st_size == expected_bytes and actual_hash == expected_hash:
        return {"path": relative, "bytes": expected_bytes, "sha256": actual_hash}

    recovered = RECOVERED_BY_HASH.get(expected_hash)
    require(recovered is not None, f"{label}: SHA-256 mismatch without recovery")
    require(recovered.is_file(), f"{label}: missing recovered artifact")
    require(recovered.stat().st_size == expected_bytes, f"{label}: recovered byte-size mismatch")
    recovered_hash = sha256_file(recovered)
    require(recovered_hash == expected_hash, f"{label}: recovered SHA-256 mismatch")
    return {
        "path": relative,
        "path_status": "mutable_path_drifted",
        "observed_sha256_at_recorded_path": actual_hash,
        "resolved_path": recovered.relative_to(ROOT).as_posix(),
        "bytes": expected_bytes,
        "sha256": recovered_hash,
        "recovery_binding": (
            "evidence/per_head_qk_repair/archive/FOCUSED_EVIDENCE_RECOVERY.json"
        ),
    }


def preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    require(not OUT.exists(), f"review already exists: {OUT.relative_to(ROOT)}")
    result = load_json(RESULTS)
    pipeline = load_json(PIPELINE_STATE)
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(
        result.get("status") == "bounded_no_go_per_head_qk_smoke_gate_failed",
        "per-head Q/K result is not the frozen bounded no-go",
    )
    require(result.get("smoke_gate", {}).get("passed") is False, "smoke gate is not failed")
    expensive = result.get("expensive_runs", {})
    require(expensive.get("full_shell_regression_run") is False, "full shell was reported run")
    require(expensive.get("canonical_sky130_ppa_run") is False, "PPA was reported run")
    require(expensive.get("full_shell_attempt_completed") is False, "aborted shell attempt is marked complete")

    verified: dict[str, Any] = {
        "results": {
            "path": RESULTS.relative_to(ROOT).as_posix(),
            "bytes": RESULTS.stat().st_size,
            "sha256": sha256_file(RESULTS),
        },
        "focused_checks": {},
        "smoke_gate": {},
        "expensive_runs": {},
    }
    for name, record in sorted(result.get("focused_checks", {}).items()):
        verified["focused_checks"][name] = verify_artifact(record, f"focused_checks.{name}")
    for name in ("baseline", "candidate"):
        verified["smoke_gate"][name] = verify_artifact(
            result.get("smoke_gate", {}).get(name, {}), f"smoke_gate.{name}"
        )
    verified["expensive_runs"]["aborted_full_shell_attempt"] = verify_artifact(
        expensive.get("aborted_full_shell_attempt", {}),
        "expensive_runs.aborted_full_shell_attempt",
    )
    return result, verified


def main() -> None:
    result, verified = preflight()
    backend_name = resolve_role_backend("reviewer")
    runner_bin = resolve_runner_bin_setting("reviewer") or None
    runner = AgentCliBackend(backend=backend_name, runner_bin=runner_bin)
    runner.set_usage_context(
        project_root=global_root() / "projects" / "s-c8ae985b",
        global_root=global_root(),
        mission_id=MISSION_ID,
    )

    package_root = Path(argus_skill.__file__).resolve().parent
    reviewer_skill = (
        package_root
        / "verticals/chip_design/skills/reviewer/chip-design-signoff-review.md"
    ).read_text(encoding="utf-8")
    objective = (
        "Independently review the frozen per-head Q/K bounded no-go at "
        "evidence/per_head_qk_repair/latest/RESULTS.json. Certify only whether "
        "the negative result is hash-bound and whether the failed operator-owned "
        "smoke gate correctly blocked completed full-shell regression and canonical "
        "SKY130 PPA. Do not certify the candidate RTL as accepted capability, do not "
        "close the RTL stage, and do not authorize benchmark, PPA, prototype, "
        "signoff, tapeout, or silicon claims."
    )
    instruction = (
        "This is exactly one bounded L2 evidence review. Inspect the files directly "
        "and use only short deterministic hash/content checks. Do not rerun either "
        "smoke, the full shell, synthesis, STA, PPA, or official evaluation. Return "
        "done only if the bounded negative result and expensive-run blocking are "
        "supported; otherwise return continue with the precise evidence defect."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id="s-c8ae985b",
        main_summary=(
            "Planner preflight verified the frozen RESULTS packet and every focused, "
            "baseline-smoke, candidate-smoke, and aborted-attempt artifact against "
            "its recorded byte count and SHA-256. The Reviewer must independently "
            "inspect those claims rather than trust this summary."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="bounded_rtl_no_go_review_stage_closing_false",
        escalate_hint=(
            "If accepted, the next legal action is Manager routing of one separately "
            "scoped, operator-authorized, structurally different layer_0.rope_q / "
            "earliest-divergence diagnosis."
        ),
        preselected_skill_block=reviewer_skill,
        config=ReviewerConfig(
            model=resolve_role_model(
                "reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"
            ),
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

    payload = {
        "schema_version": 1,
        "reviewer_role": "independent_l2",
        "reviewed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "stage": "rtl",
        "stage_closing": False,
        "scope": "per_head_qk_bounded_no_go_only",
        "backend": backend_name,
        "model": resolve_role_model(
            "reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"
        ),
        "evidence": verified,
        "smoke_gate_passed": result["smoke_gate"]["passed"],
        "full_shell_regression_run": result["expensive_runs"]["full_shell_regression_run"],
        "canonical_sky130_ppa_run": result["expensive_runs"]["canonical_sky130_ppa_run"],
        "decision": asdict(review),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, OUT)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
