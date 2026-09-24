#!/usr/bin/env python3
"""Reproduce the terminal S5 exactly-once lifecycle evidence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/bf16-full-finetune-successor-s5"
TERMINAL = BUILD / "backend-terminal-20260808T150208Z"
OUTPUT = BUILD / "backend-terminal-audit-20260808T150208Z.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-attempt-operator-authority.json"
STAGED_AUTHORITY = (
    ROOT
    / "research/execution/qwen25_bf16_full_finetune_successor_s5_20260808/staged/repo"
    / AUTHORITY.relative_to(ROOT)
)
INTENT = BUILD / "backend-submission-intent.json"
RESULT = BUILD / "backend-submission-result.json"
RAW = BUILD / "backend-submission.raw.txt"
MONITOR = BUILD / "backend-monitor.log"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_SUCCESSOR_S5_EXECUTION_PACKAGE_CONTRACT.json"
RECIPE = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-successor-s5/training_recipe.json"
OFFLINE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s5"
MANIFEST = OFFLINE / "s5-execution-package-manifest.json"
SELF_TEST = OFFLINE / "s5-execution-package-self-test.json"
L2 = OFFLINE / "execution-package-l2-acceptance.json"
PACKAGE_AUDIT = BUILD / "package-audit.json"
STAGE_AUDIT = BUILD / "stage-closure-audit.json"
EXPERIMENT = "ace2-bf16-full-finetune-successor-s5-20260808-attempt-0001"
JOB = "ace2-bf16-full-finetune-successor-s5-attempt-0001"
PACKAGE_IDENTITY = "0487a96540d733eeee88bbef1342b4af5f99204ec334d56c50092be6a5c28a02"
EXECUTION_TREE = "54602351773e85fc28e0347aacbee988cc787157306f01913bd74ff190f79a0a"
RUN_REL = Path("build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s5")
OFFLINE_REL = Path("build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s5")


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    return companion.is_file() and companion.read_text(encoding="ascii").split() == [
        sha256(path),
        path.name,
    ]


def record(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    require(len(matches) == 1, f"expected one {pattern} under {root}, observed {len(matches)}")
    return matches[0]


def verify_sums(path: Path) -> tuple[bool, int]:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        candidate = path.parent / relative.removeprefix("*").removeprefix("./")
        if not candidate.is_file() or sha256(candidate) != digest:
            return False, count
        count += 1
    return True, count


def summarize_rows(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    categories = Counter(str(row["category"]) for row in rows)
    passes = Counter(
        str(row["category"])
        for row in rows
        if bool(row["check"]["passed"])
    )
    return {
        "category_passes": {
            category: passes.get(category, 0)
            for category in sorted(categories)
        },
        "category_totals": dict(sorted(categories.items())),
        "critical_safety_failures": sum(
            int(bool(row["check"]["critical_failure"])) for row in rows
        ),
        "hard_pass_count": sum(int(bool(row["check"]["passed"])) for row in rows),
        "response_count": len(rows),
    }


def score_gates(summary: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "response_count": summary["response_count"] == policy["response_count"],
        "minimum_hard_passes": summary["hard_pass_count"] >= policy["minimum_hard_passes"],
    }
    for category, gate in policy["category_gates"].items():
        checks[f"{category}.total"] = summary["category_totals"].get(category) == gate["total"]
        checks[f"{category}.minimum_hard_passes"] = (
            summary["category_passes"].get(category, 0) >= gate["minimum_hard_passes"]
        )
    checks["critical_safety_failures"] = (
        summary["critical_safety_failures"] <= policy["critical_safety_failures_maximum"]
    )
    first_failed = next((name for name, passed in checks.items() if not passed), None)
    return {
        "checks": checks,
        "first_failed_gate": first_failed,
        "status": "PASS" if first_failed is None else "NO_GO",
    }


def probe_policy(recipe: dict[str, Any]) -> dict[str, Any]:
    early = recipe["early_stop"]
    return {
        "category_gates": {
            category: {"minimum_hard_passes": minimum, "total": 4}
            for category, minimum in early["probe_category_minimum_hard_passes"].items()
        },
        "critical_safety_failures_maximum": early["probe_critical_safety_failures_maximum"],
        "minimum_hard_passes": early["probe_hard_pass_minimum"],
        "response_count": recipe["data"]["synthetic_probe_count"],
    }


def result_root() -> tuple[Path, Path]:
    sums = one_match(TERMINAL / "results", "SHA256SUMS")
    return sums.parent, sums


def build_audit() -> dict[str, Any]:
    required_local = (
        TERMINAL,
        PIPELINE,
        AUTHORITY,
        STAGED_AUTHORITY,
        INTENT,
        RESULT,
        RAW,
        MONITOR,
        CONTRACT,
        RECIPE,
        MANIFEST,
        SELF_TEST,
        L2,
        PACKAGE_AUDIT,
        STAGE_AUDIT,
    )
    for path in required_local:
        require(path.exists(), f"required terminal input is absent: {path}")
    for path in (AUTHORITY, STAGED_AUTHORITY, INTENT, RESULT, CONTRACT, MANIFEST, SELF_TEST, L2):
        require(companion_matches(path), f"checksum companion differs: {path}")

    remote, sums = result_root()
    run_root = remote / RUN_REL
    offline = remote / OFFLINE_REL
    authority_remote = remote / AUTHORITY.relative_to(ROOT)
    readiness_path = offline / "s5-launch-readiness.json"
    launch_intent_path = offline / "s5-detached-launch-intent.json"
    launch_process_path = offline / "s5-detached-process.json"
    worker_path = offline / "s5-worker-terminal-status.json"
    terminal_no_go_path = run_root / "TERMINAL_NO_GO.json"
    terminal_go_path = run_root / "TERMINAL_GO.json"
    attempt = run_root / "attempt-0001"
    marker_path = attempt / "ATTEMPT_CONSUMPTION_MARKER.json"
    training_path = attempt / "training_result.json"
    lock_path = attempt / "checkpoint_lock.json"
    dev_dir = run_root / "qualification/dev"
    dev_path = dev_dir / "RESULT.json"
    dev_rows_path = dev_dir / "scored_rows.jsonl"
    stdout_path = one_match(TERMINAL / "logs", "stdout.txt")
    worker_console = one_match(TERMINAL / "results", "s5-detached-worker-console-*.log")
    wrapper_paths = sorted((TERMINAL / "results").rglob("s5-lifecycle-wrapper-console.log"))

    for path in (
        authority_remote,
        readiness_path,
        launch_intent_path,
        launch_process_path,
        worker_path,
        terminal_no_go_path,
        marker_path,
        training_path,
        lock_path,
        dev_path,
        dev_rows_path,
        stdout_path,
        worker_console,
    ):
        require(path.is_file(), f"required remote evidence is absent: {path}")
    for path in (authority_remote, readiness_path, launch_intent_path, launch_process_path, worker_path):
        require(companion_matches(path), f"remote checksum companion differs: {path}")
    require(len(wrapper_paths) == 2, "expected two wrapper-console copies")

    pipeline = load(PIPELINE)
    authority = load(AUTHORITY)
    intent = load(INTENT)
    submission = load(RESULT)
    inventory = load(TERMINAL / "amlt-inventory-terminal.json")
    status_rows = load(TERMINAL / "status.raw.json")
    matching = [row for row in inventory if row.get("experiment_name") == EXPERIMENT]
    require(len(matching) == 1, f"expected one matching terminal experiment, observed {len(matching)}")
    backend = matching[0]
    require(len(status_rows) == 1, f"expected one terminal status row, observed {len(status_rows)}")
    status_row = status_rows[0]
    readiness = load(readiness_path)
    launch_intent = load(launch_intent_path)
    launch_process = load(launch_process_path)
    worker = load(worker_path)
    terminal_no_go = load(terminal_no_go_path)
    marker = load(marker_path)
    training = load(training_path)
    lock = load(lock_path)
    dev = load(dev_path)
    recipe = load(RECIPE)
    contract = load(CONTRACT)
    package_audit = load(PACKAGE_AUDIT)
    stage_audit = load(STAGE_AUDIT)
    raw_text = RAW.read_text(encoding="utf-8")
    monitor_text = MONITOR.read_text(encoding="utf-8")
    stdout = stdout_path.read_text(encoding="utf-8")
    console = worker_console.read_text(encoding="utf-8")
    sums_valid, sums_count = verify_sums(sums)

    probe_summaries: dict[str, dict[str, Any]] = {}
    probe_candidates: list[dict[str, Any]] = []
    ppolicy = probe_policy(recipe)
    for epoch in (1, 2, 3):
        directory = attempt / f"probe-selection/epoch-{epoch}"
        summary_path = directory / "summary.json"
        rows_path = directory / "scored_rows.jsonl"
        identity_path = directory / "checkpoint_identity.json"
        for path in (summary_path, rows_path, identity_path):
            require(path.is_file(), f"probe evidence is absent: {path}")
        observed = load(summary_path)
        recomputed = summarize_rows(rows_path)
        gates = score_gates(recomputed, ppolicy)
        probe_summaries[str(epoch)] = {
            **recomputed,
            "category_minimum_gates_satisfied": sum(
                int(passed)
                for name, passed in gates["checks"].items()
                if name.endswith(".minimum_hard_passes")
            ),
            "gates": gates,
        }
        probe_candidates.append(
            {
                "checkpoint_epoch": epoch,
                "checkpoint_tree_sha256": observed["checkpoint_tree_sha256"],
                **probe_summaries[str(epoch)],
            }
        )
        require(
            all(observed.get(key) == value for key, value in recomputed.items()),
            f"probe aggregate differs for epoch {epoch}",
        )
        require(observed.get("gates") == gates, f"probe gates differ for epoch {epoch}")

    passing = [item for item in probe_candidates if item["gates"]["status"] == "PASS"]
    selected = (
        min(passing, key=lambda item: item["checkpoint_epoch"])
        if passing
        else min(
            probe_candidates,
            key=lambda item: (
                -item["hard_pass_count"],
                -item["category_minimum_gates_satisfied"],
                item["checkpoint_epoch"],
            ),
        )
    )

    dev_recomputed = summarize_rows(dev_rows_path)
    dev_policy = contract["evaluator_contract"]["dev"]
    dev_gates = score_gates(dev_recomputed, dev_policy)
    for key, value in dev_recomputed.items():
        require(dev.get(key) == value, f"dev aggregate differs: {key}")
    require(dev.get("gates") == dev_gates, "dev gates differ from recomputation")

    local_runtime_bindings = {
        MANIFEST.relative_to(ROOT): MANIFEST,
        SELF_TEST.relative_to(ROOT): SELF_TEST,
        L2.relative_to(ROOT): L2,
    }
    remote_runtime_bindings_match = all(
        (remote / relative).is_file() and sha256(remote / relative) == sha256(local)
        for relative, local in local_runtime_bindings.items()
    )
    checkpoints = sorted((attempt / "checkpoints").glob("checkpoint-*"))
    forbidden = (
        run_root / "selected",
        run_root / "qualification/retention",
        run_root / "qualification/holdout-exactly-once",
    )

    checks = {
        "pipeline.current_stage_specification": pipeline.get("current_stage") == "specification",
        "pipeline.hash_matches_authority": sha256(PIPELINE) == authority.get("pipeline_state_sha256"),
        "preexecution.package_92_of_92": package_audit.get("check_count") == 92 and not package_audit.get("failed_checks"),
        "preexecution.runner_40_of_40": load(SELF_TEST).get("check_count") == 40,
        "preexecution.stage_69_of_69": stage_audit.get("closure_file_count") == 69 and stage_audit.get("omission_regression_count") == 69,
        "preexecution.relocated_postreview_pass": stage_audit.get("relocated_readiness", {}).get("status") == "PASS_POSTREVIEW_RELOCATED_AUTHORITY_READY_NO_AUTHORITY",
        "authority.companion_matches": companion_matches(AUTHORITY),
        "authority.local_and_staged_identical": sha256(AUTHORITY) == sha256(STAGED_AUTHORITY),
        "authority.remote_identical": sha256(AUTHORITY) == sha256(authority_remote),
        "authority.status_exact": authority.get("status") == "AUTHORIZE_SINGLE_BF16_FULL_FINETUNE_SUCCESSOR_S5_ATTEMPT",
        "authority.maximum_backend_jobs_one": authority.get("backend", {}).get("maximum_backend_jobs") == 1,
        "authority.no_retry_resume_relaunch": all(authority.get(key) is True for key in ("no_relaunch", "no_resubmit", "no_resume")),
        "authority.package_identity_exact": authority.get("package_identity_sha256") == PACKAGE_IDENTITY,
        "authority.execution_tree_exact": authority.get("execution_tree_sha256") == EXECUTION_TREE,
        "intent.companion_matches": companion_matches(INTENT),
        "intent.binds_authority": intent.get("authority_sha256") == sha256(AUTHORITY),
        "intent.target_exact": intent.get("experiment_name") == EXPERIMENT and intent.get("job_name") == JOB,
        "submission.companion_matches": companion_matches(RESULT),
        "submission.binds_intent": submission.get("submission_intent_sha256") == sha256(INTENT),
        "submission.binds_raw": submission.get("raw_output_sha256") == sha256(RAW),
        "submission.client_exit_zero_no_prompt": submission.get("client_exit_code") == 0 and submission.get("prompt_detected") is False and submission.get("timed_out") is False,
        "submission.created_one_experiment": raw_text.count("Created new experiment") == 1,
        "submission.reported_one_job": "with 1 job" in raw_text,
        "backend.one_experiment_one_job": len(matching) == 1 and backend.get("n_jobs") == 1,
        "backend.terminal_failed": str(backend.get("job_status", "")).lower() == "failed" and backend.get("_all_terminated") is True,
        "backend.identity_exact": backend.get("_first_job_name") == JOB and status_row.get("job_name") == f":{JOB}",
        "backend.status_row_failed": status_row.get("status") == "failed" and status_row.get("experiment") == EXPERIMENT,
        "backend.monitor_terminal_failed": monitor_text.rstrip().endswith("status=failed") and monitor_text.count("status=failed") == 1,
        "download.logs_succeeded": (TERMINAL / "logs-download.exit-code.txt").read_text(encoding="ascii").strip() == "0",
        "download.results_succeeded": (TERMINAL / "results-download.exit-code.txt").read_text(encoding="ascii").strip() == "0",
        "download.remote_sums_complete": sums_valid and sums_count == len([path for path in remote.rglob("*") if path.is_file()]) - 1,
        "download.wrapper_copies_match": sha256(wrapper_paths[0]) == sha256(wrapper_paths[1]),
        "download.runtime_bindings_match_local": remote_runtime_bindings_match,
        "launch.readiness_passed": readiness.get("status") == "PASS_READY_TO_CREATE_ATTEMPT" and readiness.get("check_count") == 15 and all(readiness.get("checks", {}).values()),
        "launch.detached_intent_bound": launch_intent.get("attempt_authority_sha256") == sha256(AUTHORITY) and launch_intent.get("package_tree_sha256") == EXECUTION_TREE,
        "launch.detached_process_started": launch_process.get("status") == "DETACHED_WORKER_STARTED",
        "launch.worker_terminal_no_go": worker.get("status") == "TERMINAL_NO_GO" and worker.get("exit_code") == 2,
        "launch.stage_order_exact": [stage.get("name") for stage in worker.get("stages", [])] == ["train.py", "evaluate_dev.py"],
        "attempt.marker_created_once": marker.get("attempt") == 1 and marker.get("attempt_authority_sha256") == sha256(AUTHORITY) and marker.get("execution_tree_sha256") == EXECUTION_TREE and marker.get("resume_allowed") is False,
        "training.completed": training.get("status") == "PASS_CHECKPOINT_LOCKED_BEFORE_DEV",
        "training.full_parameter_count_exact": training.get("trainable_parameter_count") == 494032768,
        "training.optimizer_steps_exact": training.get("optimizer_steps") == 132,
        "training.three_checkpoint_epochs": training.get("probe_checkpoint_epochs") == [1, 2, 3] and len(checkpoints) == 3,
        "training.identity_exact": training.get("contract_sha256") == sha256(CONTRACT) and training.get("execution_tree_sha256") == EXECUTION_TREE,
        "selector.probe_aggregates_recomputed": all(probe_summaries[str(epoch)]["response_count"] == 28 for epoch in (1, 2, 3)),
        "selector.locked_epoch_exact": selected["checkpoint_epoch"] == lock.get("checkpoint_epoch") == training.get("selected_epoch") == 3,
        "selector.lock_before_dev": lock.get("dev_accessed") is False and lock.get("training_loss_selected_checkpoint") is False,
        "dev.executed_once": dev.get("official_dev_checkpoint_count") == 1 and dev.get("official_dev_reselection_allowed") is False,
        "dev.rows_recomputed": dev_recomputed.get("response_count") == 56 and dev_recomputed.get("hard_pass_count") == 27,
        "dev.gates_recomputed_no_go": dev_gates.get("status") == "NO_GO" and dev_gates.get("first_failed_gate") == "minimum_hard_passes",
        "terminal.no_go_exact": terminal_no_go.get("status") == "NO_GO" and terminal_no_go.get("failure_taxonomy") == "DEV_QUALITY_GATE_FAILURE" and terminal_no_go.get("first_failed_gate") == "minimum_hard_passes",
        "terminal.binds_dev_result": terminal_no_go.get("evidence_sha256") == sha256(dev_path),
        "terminal.go_absent": not terminal_go_path.exists(),
        "terminal.worker_console_exact": console.count("ACE2_FULL_FINETUNE_S5_TRAINING_PASS steps=132 epochs=[1, 2, 3] locked_epoch=3") == 1 and console.count("ACE2_FULL_FINETUNE_S5_DEV_NO_GO:minimum_hard_passes") == 1,
        "terminal.backend_stdout_detached_once": stdout.count("ACE2_FULL_FINETUNE_S5_DETACHED_STARTED") == 1,
        "lifecycle.retention_holdout_selected_absent": all(not path.exists() for path in forbidden),
        "cardinality.one_authority": len(list(AUTHORITY.parent.glob(AUTHORITY.name))) == 1,
        "cardinality.one_intent": len(list(BUILD.glob("backend-submission-intent.json"))) == 1,
        "cardinality.one_backend_job": backend.get("n_jobs") == 1,
        "cardinality.zero_retries": not any(
            "successor-s5" in str(row.get("experiment_name", ""))
            and "attempt-0002" in str(row.get("experiment_name", ""))
            for row in inventory
        ) and not (ROOT / RUN_REL / "attempt-0002").exists(),
    }
    failed = sorted(name for name, value in checks.items() if value is not True)
    require(not failed, f"terminal S5 audit failed checks: {failed}")

    return {
        "backend": {
            "all_terminated": True,
            "experiment": EXPERIMENT,
            "job_id": status_row["job_id"],
            "job_name": JOB,
            "result_size": backend.get("size"),
            "status": "failed",
        },
        "cardinality": {
            "authority_count": 1,
            "backend_experiment_count": 1,
            "backend_job_count": 1,
            "intent_count": 1,
            "retry_count": 0,
            "submit_invocation_count": 1,
        },
        "check_count": len(checks),
        "checks": dict(sorted(checks.items())),
        "claim_boundary": "Exactly one authorized S5 backend job completed full training and single locked-checkpoint dev evaluation, then terminated NO-GO at the unchanged dev quality gate. Selection materialization, retention, and holdout did not execute. No retry, resubmit, resume, relaunch, replay, repair-in-place, W4A8, RTL, FPGA/U280, or stage advance is authorized.",
        "execution_boundaries": {
            "attempt_marker_created": True,
            "backend_container_executed": True,
            "checkpoint_created": True,
            "dev_executed": True,
            "downstream_authority_opened": False,
            "evaluator_executed": True,
            "evaluator_no_execution_classification": None,
            "holdout_executed": False,
            "model_executed": True,
            "quality_conclusion": "NO_GO",
            "retention_executed": False,
            "retry_resume_relaunch_cancel_permitted": False,
            "selected_bf16_materialized": False,
            "selector_executed": True,
            "training_executed": True,
            "worker_executed": True,
        },
        "failed_check_count": 0,
        "failure": {
            "classification": "DEV_QUALITY_GATE_FAILURE",
            "dev_category_passes": dev_recomputed["category_passes"],
            "dev_hard_pass_count": dev_recomputed["hard_pass_count"],
            "dev_minimum_hard_passes": dev_policy["minimum_hard_passes"],
            "dev_response_count": dev_recomputed["response_count"],
            "first_failed_gate": dev_gates["first_failed_gate"],
            "regression": terminal_no_go["regression"],
            "root_cause_hypothesis": terminal_no_go["root_cause_hypothesis"],
        },
        "failure_taxonomy": "DEV_QUALITY_GATE_FAILURE",
        "frozen_identity": {
            "contract_sha256": sha256(CONTRACT),
            "execution_tree_sha256": EXECUTION_TREE,
            "package_identity_sha256": PACKAGE_IDENTITY,
        },
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "quality": {
            "dev": {**dev_recomputed, "gates": dev_gates},
            "probe_selection": probe_summaries,
            "selected_epoch": selected["checkpoint_epoch"],
        },
        "regeneration_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_bf16_full_finetune_successor_s5_terminal.py --check",
        "schema_version": 1,
        "status": "TERMINAL_S5_DEV_QUALITY_NO_GO_EVIDENCE_COMPLETE",
        "terminal_evidence": {
            "authority": record(AUTHORITY),
            "backend_inventory": record(TERMINAL / "amlt-inventory-terminal.json"),
            "backend_status": record(TERMINAL / "status.raw.json"),
            "backend_stdout": record(stdout_path),
            "checkpoint_lock": record(lock_path),
            "dev_result": record(dev_path),
            "intent": record(INTENT),
            "monitor": record(MONITOR),
            "remote_sha256sums": record(sums),
            "submission_raw": record(RAW),
            "submission_result": record(RESULT),
            "terminal_no_go": record(terminal_no_go_path),
            "training_result": record(training_path),
            "worker_terminal": record(worker_path),
        },
    }


def write(path: Path, value: dict[str, Any]) -> None:
    require(
        not path.exists() and not path.with_suffix(path.suffix + ".sha256").exists(),
        f"refusing to overwrite immutable audit: {path}",
    )
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n",
        encoding="ascii",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    observed = build_audit()
    if args.check:
        require(companion_matches(OUTPUT), "terminal S5 audit companion differs")
        existing = load(OUTPUT)
        for key in (
            "status",
            "failure_taxonomy",
            "check_count",
            "failed_check_count",
            "checks",
            "cardinality",
            "backend",
            "failure",
            "execution_boundaries",
            "frozen_identity",
            "quality",
        ):
            require(existing.get(key) == observed.get(key), f"terminal S5 audit field differs: {key}")
        print(f"ACE2_S5_TERMINAL_AUDIT_CHECK_PASS checks={observed['check_count']}")
        return 0
    write(OUTPUT, observed)
    print(f"ACE2_S5_TERMINAL_AUDIT_WRITTEN checks={observed['check_count']} sha256={sha256(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
