#!/usr/bin/env python3
"""Run one marker-free independent L2 review of the S4 package."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
OFFLINE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s4"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s4"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s4-attempt-operator-authority.json"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_SUCCESSOR_S4_EXECUTION_PACKAGE_CONTRACT.json"
MANIFEST = OFFLINE / "s4-execution-package-manifest.json"
SELF_TEST = OFFLINE / "s4-execution-package-self-test.json"
REQUEST = OFFLINE / "execution-package-l2-review-request.json"
DECISION = OFFLINE / "execution-package-l2-decision.json"
ACCEPTANCE = OFFLINE / "execution-package-l2-acceptance.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
MISSION_ID = "qwen25-full-finetune-s4-execution-package-l2-v3"
PACKAGE_AUDIT = ROOT / "build/bf16-full-finetune-successor-s4/package-audit.json"
SUBMISSION_ROOT = ROOT / "build/bf16-full-finetune-successor-s4"
RULINGS = (
    "s4.identity-distinct",
    "s4.s3-terminal-boundary",
    "s4.source-audit-staged",
    "s4.all-required-omissions-rejected",
    "s4.marker-free",
    "s4.lifecycle-frozen",
)
SOURCE_FILES = tuple(
    ROOT / "pilot/qwen25_05b_bf16_full_finetune_successor_s4_runner" / name
    for name in (
        "runtime.py",
        "train.py",
        "evaluate_dev.py",
        "select_model.py",
        "retention.py",
        "evaluate_holdout.py",
        "launch_detached.py",
        "run_worker.py",
        "status.py",
        "self_test.py",
    )
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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"artifact is absent: {path}")
    return {"bytes": path.stat().st_size, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    if not companion.is_file():
        return False
    return companion.read_text(encoding="ascii").split() == [sha256(path), path.name]


def write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n", encoding="ascii"
    )


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


def reviewer_supports(reason: str, identifier: str) -> bool:
    return f"{identifier}: supported" in reason.lower().replace("`", "")


def prepare_request() -> None:
    if REQUEST.exists():
        require(companion_matches(REQUEST), "S4 review request checksum differs")
        return
    audit = load(PACKAGE_AUDIT)
    manifest = load(MANIFEST)
    write_json(
        REQUEST,
        {
            "acceptance_requested": "ACCEPTED_BF16_FULL_FINETUNE_SUCCESSOR_S4_PACKAGE_NO_ATTEMPT",
            "claim_boundary": (
                "Independent Fresh Review of the exact marker-free S4 package and its complete staged-repo "
                "closure only. No authority, official stage, intent, backend job, namespace, marker, model/evaluator "
                "execution, checkpoint, quality, W4A8, RTL, FPGA, U280, or stage-advance claim is requested."
            ),
            "contract_sha256": sha256(CONTRACT),
            "execution_tree_sha256": manifest["tree_sha256"],
            "official_namespace_exists": RUN_ROOT.exists(),
            "official_v7_checkpoint_or_output_dependency": False,
            "package_audit": artifact(PACKAGE_AUDIT),
            "package_identity_sha256": audit["package_identity_sha256"],
            "requested_checks": list(RULINGS),
            "s3_terminal_boundary": {
                "failure_taxonomy": "PREATTEMPT_SOURCE_SNAPSHOT_AUDIT_OMITTED_FROM_STAGE",
                "package_identity_sha256": "5da5fd12747430377beaff78e19f2ed9aa85f11e3862ee248f8764a8e84a163b",
                "retry_or_repair_allowed": False,
                "terminal_audit_sha256": "bd2b492ae9b4172f8799dc5e96bbe153b72c39f1f7451511db2bb83c181aee16",
            },
            "schema_version": 1,
            "stage_closure": audit["stage_closure"],
            "status": "PENDING_INDEPENDENT_L2",
        },
    )


def verified_inputs() -> dict[str, Any]:
    require(load(PIPELINE).get("current_stage") == "specification", "current stage is not specification")
    require(not RUN_ROOT.exists(), "official S4 namespace exists")
    require(not AUTHORITY.exists(), "S4 attempt authority exists")
    for name in ("backend-submission-intent.json", "backend-submission-result.json", "backend-submission.raw.txt"):
        require(not (SUBMISSION_ROOT / name).exists(), f"S4 submission state exists: {name}")
    for name in ("s4-detached-launch-intent.json", "s4-detached-process.json", "s4-launch-terminal-status.json", "s4-worker-terminal-status.json"):
        require(not (OFFLINE / name).exists(), f"S4 execution record exists: {name}")
    for path in (CONTRACT, MANIFEST, SELF_TEST, REQUEST):
        require(companion_matches(path), f"checksum companion differs: {path}")
    manifest = load(MANIFEST)
    self_test = load(SELF_TEST)
    request = load(REQUEST)
    audit = load(PACKAGE_AUDIT)
    require(self_test.get("status") == "PASS", "S4 self-test is not PASS")
    require(all(self_test.get("checks", {}).values()), "one or more S4 self-test checks failed")
    required_checks = {
        "dev_policy_exactly_matches_source_contract",
        "dev_policy_schema_validates",
        "dev_gate_positive_path",
        "dev_gate_all_isolated_negative_paths",
        "legacy_v7_policy_shape_rejected_before_scoring",
    }
    require(required_checks <= set(self_test.get("checks", {})), "S4 scorer/policy schema checks are incomplete")
    require(self_test.get("official_namespace_exists") is False, "S4 self-test reports a namespace")
    require(request.get("status") == "PENDING_INDEPENDENT_L2", "S4 review request status differs")
    require(request.get("official_namespace_exists") is False, "S4 review request reports a namespace")
    require(request.get("official_v7_checkpoint_or_output_dependency") is False, "S4 request depends on V7 execution evidence")
    require(request.get("execution_tree_sha256") == manifest.get("tree_sha256"), "request tree differs")
    require(audit.get("status") == "PASS_MARKER_FREE_FULL_TRAINING_SUCCESSOR_S4_PACKAGE", "S4 package audit is not PASS")
    require(audit.get("check_count") == 84 and not audit.get("failed_checks"), "S4 package audit checks differ")
    require(audit.get("stage_closure", {}).get("required_file_count") == 63, "S4 stage closure count differs")
    require(audit.get("checks", {}).get("stage.every_required_file_omission_rejected") is True, "S4 omission regression failed")
    return {
        "claim_boundary": "Independent exact fresh-successor package review only; no authority, launch, namespace, marker, training, evaluator, checkpoint, quantization, RTL, PPA, FPGA, or product claim.",
        "contract": artifact(CONTRACT),
        "execution_manifest": artifact(MANIFEST),
        "execution_self_test": artifact(SELF_TEST),
        "package_audit": artifact(PACKAGE_AUDIT),
        "package_identity_sha256": audit["package_identity_sha256"],
        "required_stage_file_count": audit["stage_closure"]["required_file_count"],
        "required_source_audit": audit["stage_closure"]["required_source_audit"],
        "self_test_check_count": self_test["check_count"],
        "pipeline_state": artifact(PIPELINE),
        "review_request": artifact(REQUEST),
        "source_files": [artifact(path) for path in SOURCE_FILES],
        "tree_sha256": manifest["tree_sha256"],
    }


def check_existing() -> None:
    verified = verified_inputs()
    require(companion_matches(DECISION), "S4 L2 decision checksum differs")
    require(companion_matches(ACCEPTANCE), "S4 L2 acceptance checksum differs")
    decision = load(DECISION)
    acceptance = load(ACCEPTANCE)
    require(decision.get("reviewer_status") == "done", "S4 L2 decision is not done")
    require(acceptance.get("status") == "ACCEPTED_BF16_FULL_FINETUNE_SUCCESSOR_S4_PACKAGE_NO_ATTEMPT", "S4 L2 status differs")
    require(acceptance.get("accepted") is True, "S4 L2 acceptance is false")
    require(acceptance.get("attempt_authority_granted") is False, "S4 L2 grants attempt authority")
    require(acceptance.get("detached_launch_authorized") is False, "S4 L2 grants launch authority")
    require(acceptance.get("independent_from_constructor") is True, "S4 L2 is not independent")
    require(acceptance.get("execution_tree_sha256") == verified["tree_sha256"], "S4 L2 tree differs")
    require(acceptance.get("execution_manifest_sha256") == sha256(MANIFEST), "S4 L2 manifest differs")
    require(acceptance.get("contract_sha256") == sha256(CONTRACT), "S4 L2 contract differs")
    require(acceptance.get("reviewer_decision_sha256") == sha256(DECISION), "S4 L2 decision binding differs")


def main() -> int:
    if "--check" in sys.argv[1:]:
        prepare_request()
        check_existing()
        print("ACE2_FULL_FINETUNE_S4_EXECUTION_PACKAGE_L2_CHECK_PASS")
        return 0

    require(not DECISION.exists(), "S4 L2 decision already exists")
    require(not ACCEPTANCE.exists(), "S4 L2 acceptance already exists")
    prepare_request()
    verified = verified_inputs()
    runner = AgentCliBackend(
        backend=resolve_role_backend("reviewer"),
        runner_bin=resolve_runner_bin_setting("reviewer") or None,
    )
    sid = session_id()
    runner.set_usage_context(
        project_root=global_root() / "projects" / sid,
        global_root=global_root(),
        mission_id=MISSION_ID,
    )
    objective = (
        "Independently review the exact marker-free ACE-2 Qwen2.5-0.5B S4 full-parameter BF16 package, "
        "with special attention to the S3 terminal boundary and complete staged-file closure."
    )
    instruction = (
        "Inspect the exact request, contract, package audit, freeze/data/environment records, manifest, "
        "self-test, stage builder, and listed runner sources directly. You may run read-only hashes, "
        "py_compile, package self-tests in --check mode, and source scans. Do not create "
        "authority, launch records, a namespace or marker; do not run preflight, launch_detached.py, "
        "train, official dev/holdout, retention, selection, quantization, RTL, synthesis, or U280 work. "
        "Return done only if S4 is byte-distinct from immutable consumed S3; consumes no S3 output or state; "
        "explicitly stages the required source-audit file at SHA-256 5ea75b61105ff5ce30803c5c7b33e2bde36acb0511ffddc2da55303909cf657d; "
        "the 84-check package audit and 39-check runner self-test reproduce; all 63 required staged-repo files "
        "are covered and omission of each is rejected locally; no authority, official stage, intent, namespace, "
        "marker, experiment, job, model/evaluator execution, or quality result exists; and the frozen full-training "
        "lifecycle remains ordered and fail-closed. In the reason state exactly: "
        "'s4.identity-distinct: supported', 's4.s3-terminal-boundary: supported', "
        "'s4.source-audit-staged: supported', 's4.all-required-omissions-rejected: supported', "
        "'s4.marker-free: supported', and 's4.lifecycle-frozen: supported'. "
        "A done verdict grants no attempt or launch authority."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=sid,
        main_summary=f"Planner froze marker-free S4 tree {verified['tree_sha256']} with package identity {verified['package_identity_sha256']}; 84 package checks, {verified['self_test_check_count']} runner checks, and all {verified['required_stage_file_count']} staged-file omission regressions pass.",
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="s4_full_finetune_exact_package_and_stage_closure_l2_only",
        checkpoint_path="",
        escalate_hint="This review never grants attempt authority and never advances the Manager-owned specification stage.",
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort("ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"),
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
    if status == "done":
        require(all(reviewer_supports(reason, item) for item in RULINGS), "Reviewer reason lacks exact rulings")
        require(not raw.get("next_action"), "Reviewer returned done with a next action")
        require(bool(raw.get("thread_id")), "Reviewer execution identity is missing")
    decision = {
        "claim_boundary": verified["claim_boundary"],
        "execution_tree_sha256": verified["tree_sha256"],
        "raw_decision": raw,
        "reason": reason,
        "reviewed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "reviewer_status": status,
        "reviewer_thread_id": raw.get("thread_id", ""),
        "schema_version": 1,
        "verified_evidence": verified,
    }
    write_json(DECISION, decision)
    if status != "done":
        print(f"ACE2_FULL_FINETUNE_S4_EXECUTION_PACKAGE_L2_REPLAN status={status}")
        return 2
    acceptance = {
        "accepted": True,
        "attempt_authority_granted": False,
        "claim_boundary": verified["claim_boundary"],
        "contract_sha256": sha256(CONTRACT),
        "detached_launch_authorized": False,
        "execution_manifest_sha256": sha256(MANIFEST),
        "execution_tree_sha256": verified["tree_sha256"],
        "package_audit_sha256": sha256(PACKAGE_AUDIT),
        "package_identity_sha256": verified["package_identity_sha256"],
        "required_stage_file_count": verified["required_stage_file_count"],
        "s3_retry_or_repair_allowed": False,
        "independent_from_constructor": True,
        "score_policy_schema": "ace2_score_gates_v1",
        "reviewed_at_utc": decision["reviewed_at_utc"],
        "reviewer_decision_sha256": sha256(DECISION),
        "reviewer_status": "done",
        "reviewer_thread_id": raw["thread_id"],
        "schema_version": 1,
        "status": "ACCEPTED_BF16_FULL_FINETUNE_SUCCESSOR_S4_PACKAGE_NO_ATTEMPT",
    }
    write_json(ACCEPTANCE, acceptance)
    check_existing()
    print(f"ACE2_FULL_FINETUNE_S4_EXECUTION_PACKAGE_L2_PASS tree={verified['tree_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
