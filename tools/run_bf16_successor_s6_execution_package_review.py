#!/usr/bin/env python3
"""Request/check one read-only Fresh Review of the marker-free S6 package."""

from __future__ import annotations

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
OFFLINE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s6"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_SUCCESSOR_S6_EXECUTION_PACKAGE_CONTRACT.json"
MANIFEST = OFFLINE / "s6-execution-package-manifest.json"
SELF_TEST = OFFLINE / "s6-execution-package-self-test.json"
PACKAGE_AUDIT = ROOT / "build/bf16-full-finetune-successor-s6/package-audit.json"
STAGE_AUDIT = ROOT / "build/bf16-full-finetune-successor-s6/stage-closure-audit.json"
DESIGN = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_CATEGORY_BALANCED_CONFLICT_PROJECTED_SUCCESSOR_S6_CLEAN_ROOM_V2_DESIGN_FREEZE.json"
DATASET = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-successor-s6/dataset_manifest.json"
RECIPE = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-successor-s6/training_recipe.json"
SCHEDULE = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-successor-s6/optimizer_schedule.json"
SIDECAR = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-successor-s6/category_sidecar.jsonl"
REQUEST = OFFLINE / "execution-package-l2-review-request.json"
DECISION = OFFLINE / "execution-package-l2-acceptance.json"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s6-attempt-operator-authority.json"
INTENT = ROOT / "build/bf16-full-finetune-successor-s6/backend-submission-intent.json"
RESULT = ROOT / "build/bf16-full-finetune-successor-s6/backend-submission-result.json"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s6"
MISSION_ID = "qwen25-bf16-full-finetune-successor-s6-execution-package-fresh-review-v2"
REQUIRED_STATUS = "ACCEPTED_BF16_FULL_FINETUNE_SUCCESSOR_S6_PACKAGE_NO_ATTEMPT"
RULINGS = (
    "s6.package-identity-and-hashes",
    "s6.category-balance-and-schedule",
    "s6.conflict-projection-implementation",
    "s6.selector-no-fallback",
    "s6.evaluator-data-boundaries",
    "s6.relocated-runtime-closure",
    "s6.zero-consuming-state",
    "s6.no-model-or-evaluator-execution",
    "s6.no-downstream-authority",
    "s6.false-rtl-goal-gate-contained",
)


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    return companion.is_file() and companion.read_text(encoding="ascii").split() == [sha256(path), path.name]


def artifact(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def write_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists() and not path.with_suffix(path.suffix + ".sha256").exists(), f"refusing to overwrite review artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="ascii")


def supports(reason: str, ruling: str) -> bool:
    return f"{ruling}: supported" in reason.lower().replace("`", "")


def state() -> dict[str, bool]:
    return {
        "authority": AUTHORITY.exists(),
        "intent": INTENT.exists(),
        "result": RESULT.exists(),
        "official_namespace": RUN_ROOT.exists(),
    }


def verified_inputs() -> dict[str, Any]:
    for path in (CONTRACT, MANIFEST, SELF_TEST, PACKAGE_AUDIT, STAGE_AUDIT, DESIGN, DATASET, RECIPE, SCHEDULE):
        require(path.is_file(), f"review input is absent: {path}")
    for path in (CONTRACT, MANIFEST, SELF_TEST):
        require(companion_matches(path), f"checksum companion differs: {path}")
    require(state() == {"authority": False, "intent": False, "result": False, "official_namespace": False}, "consuming S6 state exists")
    contract = load(CONTRACT)
    manifest = load(MANIFEST)
    self_test = load(SELF_TEST)
    package_audit = load(PACKAGE_AUDIT)
    stage_audit = load(STAGE_AUDIT)
    design = load(DESIGN)
    dataset = load(DATASET)
    recipe = load(RECIPE)
    schedule = load(SCHEDULE)
    require(contract["status"] == "FROZEN_MARKER_FREE_PENDING_FRESH_INDEPENDENT_REVIEW", "S6 contract status differs")
    require(contract["attempt_authority"]["granted"] is False, "S6 contract grants authority")
    require(contract["mechanism"]["name"] == design["successor_mechanism"]["name"], "S6 mechanism differs from accepted design")
    require(manifest["tree_sha256"] == self_test["execution_tree_sha256"], "S6 execution tree bindings differ")
    require(self_test["status"] == "PASS" and self_test["check_count"] == 45 and all(self_test["checks"].values()), "S6 self-test differs")
    require(package_audit["status"] == "PASS_MARKER_FREE_FULL_TRAINING_SUCCESSOR_S6_PACKAGE", "S6 package audit status differs")
    require(package_audit["check_count"] == 100 and not package_audit["failed_checks"] and all(package_audit["checks"].values()), "S6 package audit checks differ")
    require(stage_audit["status"] == "PASS_PREREVIEW_MARKER_FREE_S6_TRANSITIVE_STAGE_CLOSURE", "S6 stage audit status differs")
    require(stage_audit["closure_file_count"] == stage_audit["omission_regression_count"] == 83, "S6 stage closure count differs")
    require(len(stage_audit["runtime_sidecar_omission_rejections"]) == 3, "S6 runtime sidecar regression count differs")
    require(dataset["counts"] == {"additive_patch": 72, "backbone": 280, "bridge_patch": 2, "category_patch": 70, "synthetic_probes": 28, "train": 352}, "S6 dataset counts differ")
    require(dataset["construction"]["dev_or_holdout_records_parsed"] is False, "S6 package parsed evaluation rows")
    require(dataset["construction"]["external_model_outputs_used"] is False, "S6 package used model outputs")
    require(recipe["early_stop"]["no_passing_epoch_disposition"] == "NO_CHECKPOINT_AND_NO_OFFICIAL_DEV", "S6 selector fallback differs")
    require(schedule["epoch_count"] == 3 and all(len(epoch["steps"]) == 44 for epoch in schedule["epochs"]), "S6 schedule shape differs")
    require(sum(1 for _ in SIDECAR.open("r", encoding="utf-8")) == 352, "S6 category sidecar count differs")
    return {
        "claim_boundary": "Fresh read-only review of the marker-free S6 full execution package only. Acceptance may certify package fidelity and relocated closure but grants no authority, intent, namespace, attempt, model/evaluator execution, quality result, stage transition, W4A8, RTL, synthesis/PPA, FPGA, or U280 work.",
        "contract": artifact(CONTRACT),
        "execution_manifest": artifact(MANIFEST),
        "execution_self_test": artifact(SELF_TEST),
        "package_audit": artifact(PACKAGE_AUDIT),
        "stage_audit": artifact(STAGE_AUDIT),
        "design_freeze": artifact(DESIGN),
        "dataset_manifest": artifact(DATASET),
        "training_recipe": artifact(RECIPE),
        "optimizer_schedule": artifact(SCHEDULE),
        "category_sidecar": artifact(SIDECAR),
        "execution_tree_sha256": manifest["tree_sha256"],
        "state_before_review": state(),
    }


def check_existing() -> None:
    evidence = verified_inputs()
    require(companion_matches(REQUEST), "S6 review request checksum differs")
    require(companion_matches(DECISION), "S6 review decision checksum differs")
    decision = load(DECISION)
    require(decision.get("status") == REQUIRED_STATUS and decision.get("accepted") is True, "S6 acceptance status differs")
    require(decision.get("independent_from_constructor") is True, "S6 acceptance is not independent")
    require(decision.get("attempt_authority_granted") is False and decision.get("detached_launch_authorized") is False, "S6 acceptance grants execution")
    require(decision.get("execution_tree_sha256") == evidence["execution_tree_sha256"], "S6 acceptance tree differs")
    reason = str(decision.get("review", {}).get("reason", ""))
    require(all(supports(reason, ruling) for ruling in RULINGS), "S6 acceptance lacks required rulings")


def main() -> int:
    if "--check" in sys.argv[1:]:
        check_existing()
        print("ACE2_S6_EXECUTION_PACKAGE_FRESH_REVIEW_CHECK_PASS")
        return 0

    require(not REQUEST.exists() and not DECISION.exists(), "S6 package Fresh Review already exists")
    evidence = verified_inputs()
    write_json(
        REQUEST,
        {
            "evidence": evidence,
            "mission_id": MISSION_ID,
            "requested_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "requested_rulings": list(RULINGS),
            "schema_version": 1,
            "status": "PENDING_FRESH_REVIEW",
        },
    )

    backend_name = resolve_role_backend("reviewer")
    runner = AgentCliBackend(backend=backend_name, runner_bin=resolve_runner_bin_setting("reviewer") or None)
    sid = os.environ.get("ARGUS_SKILL_SESSION_ID", ROOT.name)
    runner.set_usage_context(project_root=global_root() / "projects" / sid, global_root=global_root(), mission_id=MISSION_ID)
    reviewer_skill = (
        Path(argus_skill.__file__).resolve().parent
        / "verticals/digital_circuit/skills/reviewer/digital-circuit-signoff-review.md"
    ).read_text(encoding="utf-8")
    objective = "Independently accept or reject the exact marker-free ACE-2 S6 full execution package and relocated runtime closure."
    instruction = (
        "Work read-only. Inspect only the listed hash-bound package, data metadata, runner, audit, and stage artifacts. "
        "Do not inspect or expose dev/holdout row content, scored rows, golden outputs, hidden harnesses, or historical response text. "
        "Do not create or modify CHECKPOINT.md, PIPELINE_STATE.json, authority, intent, namespace, marker, process, job, model/evaluator output, or downstream artifact. "
        "Do not train, generate, score, run dev/retention/holdout, submit AMLT, run W4A8/RTL/simulation/formal/synthesis/PPA/FPGA/U280, or advance stages. "
        "Independently verify that the accepted S6 design is implemented by the category sidecar, 44-step explicit schedule, seven-category FP32 raw-gradient projection using immutable references, post-combine clipping, and strict no-fallback selector. "
        "Verify package audit 100/100, transitive closure 83/83, three runtime sidecar omissions rejected, canonical AMLT 11.17.0 read-only dump, zero matching experiments/jobs, and absence of authority/intent/official namespace. "
        "The live pipeline current_stage=rtl is an inapplicable false goal-gate advance; package acceptance must explicitly keep this work at specification and authorize no RTL. "
        "Return done only if every boundary holds. Include exactly these rulings: "
        + ", ".join(f"'{r}: supported'" for r in RULINGS)
        + ". A done verdict accepts only the marker-free package and relocated closure, with no attempt or downstream authority."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=sid,
        main_summary="S6 package audit passes 100/100; runner self-test passes 45 checks; relocated stage closure passes 83/83 with three mandatory runtime-sidecar negative cases; consuming state remains absent.",
        main_error=None,
        raw_evidence=json.dumps(evidence, indent=2, sort_keys=True),
        scope="s6_marker_free_full_execution_package_and_relocated_closure_only",
        checkpoint_path="",
        escalate_hint="No execution, authority, pipeline mutation, or downstream hardware work is in scope.",
        preselected_skill_block=reviewer_skill,
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort("ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"),
            skip_git_repo_check=True,
            full_auto=True,
            dangerous_yolo=False,
            sandbox_mode="read-only",
            isolate_workdir=False,
            working_dir=str(ROOT),
        ),
    )
    raw = asdict(review)
    reason = str(raw.get("reason", ""))
    if raw.get("status") == "done":
        require(all(supports(reason, ruling) for ruling in RULINGS), "Fresh Reviewer reason lacks exact required rulings")
        require(not raw.get("next_action"), "Fresh Reviewer returned done with a next action")
        require(bool(raw.get("thread_id")), "Fresh Reviewer execution identity is missing")
    require(state() == evidence["state_before_review"], "Fresh Review changed consuming S6 state")
    write_json(
        DECISION,
        {
            "accepted": raw.get("status") == "done",
            "attempt_authority_granted": False,
            "claim_boundary": evidence["claim_boundary"],
            "contract_sha256": sha256(CONTRACT),
            "detached_launch_authorized": False,
            "execution_manifest_sha256": sha256(MANIFEST),
            "execution_tree_sha256": evidence["execution_tree_sha256"],
            "independent_from_constructor": True,
            "mission_id": MISSION_ID,
            "review": raw,
            "review_request": artifact(REQUEST),
            "reviewed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "schema_version": 1,
            "status": REQUIRED_STATUS if raw.get("status") == "done" else "S6_EXECUTION_PACKAGE_REPLAN_REQUIRED",
            "verified_evidence": evidence,
        },
    )
    if raw.get("status") != "done":
        print(f"ACE2_S6_EXECUTION_PACKAGE_FRESH_REVIEW_REPLAN status={raw.get('status')}")
        return 2
    print(f"ACE2_S6_EXECUTION_PACKAGE_FRESH_REVIEW_DONE thread={raw.get('thread_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
