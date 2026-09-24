#!/usr/bin/env python3
"""Run one independent L2 review of the frozen Q/K rotation no-go."""

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
RESULTS = ROOT / "evidence/rope_pair_rotation/latest/RESULTS.json"
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"
OUT_DIR = ROOT / "evidence/review/rope_pair_rotation_no_go_l2"
OUT = OUT_DIR / "decision.json"
MISSION_ID = "rope-pair-rotation-no-go-l2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    require(path.stat().st_size == expected_bytes, f"{label}: byte-size mismatch")
    actual_hash = sha256_file(path)
    require(actual_hash == expected_hash, f"{label}: SHA-256 mismatch")
    return {"path": relative, "bytes": expected_bytes, "sha256": actual_hash}


def preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    require(not OUT.exists(), f"review already exists: {OUT.relative_to(ROOT)}")
    result = load_json(RESULTS)
    pipeline = load_json(PIPELINE_STATE)
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(
        result.get("status") == "bounded_no_go_rope_pair_rotation_c4_regressed",
        "rotation result is not the frozen bounded no-go",
    )
    require(result.get("stage_closing") is False, "result is unexpectedly stage-closing")
    require(result.get("smoke_gate", {}).get("passed") is False, "smoke gate is not failed")
    expensive = result.get("expensive_runs", {})
    require(expensive.get("full_shell_regression_run") is False, "full shell was reported run")
    require(expensive.get("canonical_sky130_ppa_run") is False, "PPA was reported run")
    require(
        result.get("rtl_checklist_evidence", {}).get("rtl.contract-traceability") is False,
        "result does not preserve the RTL contract mismatch",
    )

    verified: dict[str, Any] = {
        "results": {
            "path": RESULTS.relative_to(ROOT).as_posix(),
            "bytes": RESULTS.stat().st_size,
            "sha256": sha256_file(RESULTS),
        },
        "focused_checks": {},
        "selection_probe": {},
        "smoke_gate": {},
        "smoke_provenance": {},
        "unbound_claims": [
            "focused_checks.fixed_point_self_test is a status string without a standalone artifact record; this bounded review must not rely on it"
        ],
    }
    for name in ("rtl_lint", "rtl_rope_core", "rtl_rope_shell"):
        verified["focused_checks"][name] = verify_artifact(
            result.get("focused_checks", {}).get(name, {}),
            f"focused_checks.{name}",
        )
    for name in ("control", "selected"):
        verified["selection_probe"][name] = verify_artifact(
            result.get("selection_probe", {}).get(name, {}),
            f"selection_probe.{name}",
        )
    for name in ("baseline", "candidate"):
        verified["smoke_gate"][name] = verify_artifact(
            result.get("smoke_gate", {}).get(name, {}),
            f"smoke_gate.{name}",
        )
    for name in ("candidate_packet", "source_hash_list"):
        verified["smoke_provenance"][name] = verify_artifact(
            result.get("smoke_provenance", {}).get(name, {}),
            f"smoke_provenance.{name}",
        )
    require(
        verified["smoke_provenance"]["source_hash_list"]["sha256"]
        == result.get("candidate_rtl_hash"),
        "candidate RTL hash is not the source-list hash",
    )
    return result, verified


def main() -> None:
    result, verified = preflight()
    backend_name = resolve_role_backend("reviewer")
    runner_bin = resolve_runner_bin_setting("reviewer") or None
    runner = AgentCliBackend(backend=backend_name, runner_bin=runner_bin)
    session_id = os.environ.get("ARGUS_SKILL_SESSION_ID", "s-c8ae985b")
    runner.set_usage_context(
        project_root=global_root() / "projects" / session_id,
        global_root=global_root(),
        mission_id=MISSION_ID,
    )

    package_root = Path(argus_skill.__file__).resolve().parent
    reviewer_skill = (
        package_root
        / "verticals/chip_design/skills/reviewer/chip-design-signoff-review.md"
    ).read_text(encoding="utf-8")
    objective = (
        "Independently review the frozen shared +22.5 degree Q/K basis-rotation "
        "bounded no-go at evidence/rope_pair_rotation/latest/RESULTS.json. Certify "
        "only whether the two-dataset negative result is hash-bound, whether the "
        "failed operator-owned gate correctly blocked completed full-shell and "
        "canonical SKY130 PPA runs, and whether the packet honestly preserves the "
        "supported prefix and RTL contract-traceability failure. Do not certify the "
        "candidate as accepted capability, close the RTL stage, authorize a new "
        "numerical contract, or authorize downstream stages."
    )
    instruction = (
        "This is exactly one bounded, stage_closing=false L2 evidence review. Read "
        "the files directly and use only short deterministic hash/content checks. "
        "Do not rerun either smoke, the full shell, synthesis, STA, PPA, official "
        "evaluation, or any replacement candidate. Update CHECKPOINT.md before the "
        "verdict. Return done only if the bounded no-go, expensive-run blocking, "
        "unsupported frontier, and required Manager rollback are supported; otherwise "
        "return continue with the precise evidence defect."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=session_id,
        main_summary=(
            "Planner preflight verified the frozen result packet and all recorded "
            "selection-probe, focused RTL, paired-smoke, candidate-packet, and source-"
            "list artifacts against byte counts and SHA-256. The standalone fixed-"
            "point self-test status is explicitly unbound and is not needed to accept "
            "the bounded negative result."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="bounded_rtl_no_go_review_stage_closing_false",
        checkpoint_path=str(ROOT / "CHECKPOINT.md"),
        escalate_hint=(
            "If accepted, the next legal action is Manager rollback to architecture. "
            "Any replacement numerical contract still requires explicit operator approval."
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
            sandbox_mode="workspace-write",
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
        "scope": "rope_pair_rotation_bounded_no_go_only",
        "backend": backend_name,
        "model": resolve_role_model(
            "reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"
        ),
        "evidence": verified,
        "smoke_gate_passed": result["smoke_gate"]["passed"],
        "rtl_contract_traceability": result["rtl_checklist_evidence"][
            "rtl.contract-traceability"
        ],
        "full_shell_regression_run": result["expensive_runs"][
            "full_shell_regression_run"
        ],
        "canonical_sky130_ppa_run": result["expensive_runs"][
            "canonical_sky130_ppa_run"
        ],
        "decision": asdict(review),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, OUT)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
