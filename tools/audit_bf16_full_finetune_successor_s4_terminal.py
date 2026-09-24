#!/usr/bin/env python3
"""Reproduce the terminal, non-consuming S4 backend evidence audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/bf16-full-finetune-successor-s4"
TERMINAL = BUILD / "backend-terminal-20260808T132133Z"
OUTPUT = BUILD / "backend-terminal-audit-v2-20260808T132245Z.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s4-attempt-operator-authority.json"
INTENT = BUILD / "backend-submission-intent.json"
RESULT = BUILD / "backend-submission-result.json"
RAW = BUILD / "backend-submission.raw.txt"
MONITOR = BUILD / "backend-monitor.log"
STAGED_REPO = ROOT / "research/execution/qwen25_bf16_full_finetune_successor_s4_20260808/staged/repo"
AUTHORITY_PIPELINE_SNAPSHOT = STAGED_REPO / "research/PIPELINE_STATE.json"
STAGE_BUILDER = ROOT / "research/execution/qwen25_bf16_full_finetune_successor_s4_20260808/prepare_stage.py"
RUNTIME = ROOT / "pilot/qwen25_05b_bf16_full_finetune_successor_s4_runner/runtime.py"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_SUCCESSOR_S4_EXECUTION_PACKAGE_CONTRACT.json"
OFFLINE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s4"
MANIFEST = OFFLINE / "s4-execution-package-manifest.json"
SELF_TEST = OFFLINE / "s4-execution-package-self-test.json"
L2 = OFFLINE / "execution-package-l2-acceptance.json"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s4"
EXPERIMENT = "ace2-bf16-full-finetune-successor-s4-20260808-attempt-0001"
JOB = "ace2-bf16-full-finetune-successor-s4-attempt-0001"
JOB_ID = "ace2-bf16-full-finetune-successor-s4-20260808-a70f167d"
PACKAGE_IDENTITY = "8e2d4fad25dd4a102911b2e85e69147578a60727d10bf81d3d472b67c357c016"
EXECUTION_TREE = "4d67d8d444681a11d3e1060b53da65ae67d2557aa9f780fe8e87106c7473de43"
RUNTIME_REQUIRED_SIDECARS = (CONTRACT, MANIFEST, SELF_TEST)
NON_CAUSAL_ABSENT_SIDECAR = L2


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
    return companion.is_file() and companion.read_text(encoding="ascii").split() == [sha256(path), path.name]


def record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


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


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("ace2_s4_stage_builder_terminal_audit", path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan_hygiene(paths: list[Path]) -> dict[str, Any]:
    signatures = {
        "private_key_header": re.compile(("-----" + "BEGIN ") + r"(?:RSA |EC |OPENSSH |DSA )?" + ("PRIVATE " + "KEY-----")),
        "aws_access_key": re.compile(("AK" + "IA") + r"[0-9A-Z]{16}"),
        "github_token": re.compile(("gh" + r"[pousr]_") + r"[A-Za-z0-9]{20,}"),
        "slack_token": re.compile(("xo" + r"[xbaprs]-") + r"[A-Za-z0-9-]{20,}"),
    }
    records: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(set(paths)):
        raw = path.read_bytes()
        relative = path.relative_to(ROOT).as_posix()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            skipped.append({"path": relative, "reason": str(exc)})
            continue
        records.append({"bytes": len(raw), "path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
        for name, pattern in signatures.items():
            if pattern.search(text):
                findings.append({"path": relative, "signature": name})
    return {
        "bytes_scanned": sum(item["bytes"] for item in records),
        "complete_byte_coverage": len(records) == len(set(paths)) and not skipped,
        "file_count": len(records),
        "findings": findings,
        "size_limit_bytes": None,
        "skipped_file_count": len(skipped),
        "skipped_files": skipped,
        "status": "PASS_COMPLETE_NO_SIZE_CAP" if not findings and not skipped else "HYGIENE_NO_GO",
    }


def build_audit() -> dict[str, Any]:
    for path in (
        TERMINAL,
        PIPELINE,
        AUTHORITY_PIPELINE_SNAPSHOT,
        AUTHORITY,
        INTENT,
        RESULT,
        RAW,
        MONITOR,
        STAGED_REPO,
        STAGE_BUILDER,
        RUNTIME,
    ):
        require(path.exists(), f"required terminal input is absent: {path}")
    for path in (AUTHORITY, INTENT, RESULT):
        require(companion_matches(path), f"checksum companion differs: {path}")

    pipeline = load(PIPELINE)
    authority = load(AUTHORITY)
    intent = load(INTENT)
    result = load(RESULT)
    inventory = load(TERMINAL / "amlt-inventory-terminal.json")
    matching = [row for row in inventory if row.get("experiment_name") == EXPERIMENT]
    require(len(matching) == 1, f"expected one matching terminal experiment, observed {len(matching)}")
    row = matching[0]

    results_root = TERMINAL / "results"
    logs_root = TERMINAL / "logs"
    sums = one_match(results_root, "SHA256SUMS")
    remote_root = sums.parent
    readiness_path = one_match(results_root, "s4-launch-readiness.json")
    stdout_path = one_match(logs_root, "stdout.txt")
    wrapper_paths = sorted(results_root.rglob("s4-lifecycle-wrapper-console.log"))
    require(len(wrapper_paths) == 2, "expected two downloaded wrapper-log copies")
    readiness = load(readiness_path)
    stdout = stdout_path.read_text(encoding="utf-8")
    wrapper = wrapper_paths[0].read_text(encoding="utf-8")
    raw_text = RAW.read_text(encoding="utf-8")
    runtime_text = RUNTIME.read_text(encoding="utf-8")
    stage_builder = load_module(STAGE_BUILDER)
    contract = load(CONTRACT)
    freeze = load(ROOT / contract["frozen_inputs"]["freeze_manifest"]["path"])
    manifest = load(MANIFEST)
    bindings = stage_builder.required_repo_bindings(contract, freeze, manifest, include_review=True)

    required_sidecars = [path.with_suffix(path.suffix + ".sha256") for path in RUNTIME_REQUIRED_SIDECARS]
    l2_sidecar = NON_CAUSAL_ABSENT_SIDECAR.with_suffix(NON_CAUSAL_ABSENT_SIDECAR.suffix + ".sha256")
    staged_required_sidecars = [STAGED_REPO / path.relative_to(ROOT) for path in required_sidecars]
    staged_l2_sidecar = STAGED_REPO / l2_sidecar.relative_to(ROOT)
    remote_relatives = [path.relative_to(remote_root).as_posix() for path in remote_root.rglob("*") if path.is_file()]
    forbidden_names = {
        "ATTEMPT_CONSUMPTION_MARKER.json", "s4-detached-launch-intent.json", "s4-detached-process.json",
        "s4-launch-terminal-status.json", "s4-worker-terminal-status.json", "training_result.json",
        "checkpoint_lock.json", "TERMINAL_GO.json", "TERMINAL_NO_GO.json", "selection.json",
    }
    observed_forbidden = sorted({Path(name).name for name in remote_relatives} & forbidden_names)
    sums_valid, sums_count = verify_sums(sums)
    nvidia = one_match(results_root, "nvidia-smi-q.raw.txt").read_text(encoding="utf-8")

    checks = {
        "pipeline.current_stage_specification": pipeline.get("current_stage") == "specification",
        # The authority binds the pipeline bytes staged for the consumed S4
        # lifecycle. The live Manager-owned pipeline legitimately advances or
        # rolls back afterward and is checked independently for the current
        # specification stage above.
        "pipeline.hash_matches_authority": sha256(AUTHORITY_PIPELINE_SNAPSHOT)
        == authority.get("pipeline_state_sha256"),
        "authority.companion_matches": companion_matches(AUTHORITY),
        "authority.status_exact": authority.get("status") == "AUTHORIZE_SINGLE_BF16_FULL_FINETUNE_SUCCESSOR_S4_ATTEMPT",
        "authority.maximum_backend_jobs_one": authority.get("backend", {}).get("maximum_backend_jobs") == 1,
        "authority.no_retry_resume_relaunch": all(authority.get(key) is True for key in ("no_relaunch", "no_resubmit", "no_resume")),
        "authority.package_identity_exact": authority.get("package_identity_sha256") == PACKAGE_IDENTITY,
        "authority.execution_tree_exact": authority.get("execution_tree_sha256") == EXECUTION_TREE,
        "intent.companion_matches": companion_matches(INTENT),
        "intent.binds_authority": intent.get("authority_sha256") == sha256(AUTHORITY),
        "intent.target_exact": intent.get("experiment_name") == EXPERIMENT and intent.get("job_name") == JOB,
        "result.companion_matches": companion_matches(RESULT),
        "result.binds_intent": result.get("submission_intent_sha256") == sha256(INTENT),
        "result.binds_raw": result.get("raw_output_sha256") == sha256(RAW),
        "result.client_exit_zero_no_prompt": result.get("client_exit_code") == 0 and result.get("prompt_detected") is False,
        "submission.created_one_experiment": raw_text.count("Created new experiment") == 1,
        "submission.reported_one_job": "with 1 job" in raw_text,
        "backend.one_experiment_one_job": len(matching) == 1 and row.get("n_jobs") == 1,
        "backend.terminal_failed": str(row.get("job_status", "")).lower() == "failed" and row.get("_all_terminated") is True,
        "backend.identity_exact": row.get("_first_job_name") == JOB and JOB_ID in str(row.get("job_url", "")),
        "backend.status_message_preserved": "[nfs-sidecar] Error; [master] Error" in str(row.get("description", "")),
        "download.logs_succeeded": (TERMINAL / "logs-download.exit-code.txt").read_text(encoding="ascii").strip() == "0",
        "download.results_succeeded": (TERMINAL / "results-download.exit-code.txt").read_text(encoding="ascii").strip() == "0",
        "download.remote_sums_complete": sums_valid and sums_count == len([p for p in remote_root.rglob("*") if p.is_file()]) - 1,
        "download.wrapper_copies_match": sha256(wrapper_paths[0]) == sha256(wrapper_paths[1]),
        "failure.readiness_status_exact": readiness.get("status") == "PREATTEMPT_NO_GO",
        "failure.taxonomy_exact": readiness.get("failure_taxonomy") == "EXECUTION_INTERLOCK_NOT_SATISFIED",
        "failure.first_gate_exact": readiness.get("first_failed_gate") == "frozen_bindings_exact",
        "failure.contract_sidecar_error_exact": readiness.get("errors", {}).get("frozen_bindings_exact") == "S4 execution contract sidecar differs",
        "failure.manifest_sidecar_error_exact": readiness.get("errors", {}).get("execution_self_test_passed") == "S4 execution manifest sidecar differs",
        "failure.wrapper_exact": wrapper.count("RuntimeError: launch preflight failed: frozen_bindings_exact") == 1,
        "failure.stdout_exact": stdout.count("RuntimeError: launch preflight failed: frozen_bindings_exact") == 1,
        "failure.dependencies_installed_before_gate": "Successfully installed" in stdout and stdout.index("Successfully installed") < stdout.index("RuntimeError: launch preflight failed"),
        "failure.runtime_requires_contract_sidecar": "companion_matches(CONTRACT_PATH, CONTRACT_SUM)" in runtime_text,
        "failure.runtime_requires_manifest_sidecar": "companion_matches(EXECUTION_MANIFEST, EXECUTION_MANIFEST_SUM)" in runtime_text,
        "failure.runtime_requires_self_test_sidecar": "companion_matches(SELF_TEST_RESULT, SELF_TEST_RESULT_SUM)" in runtime_text,
        "failure.runtime_does_not_require_l2_sidecar": "EXECUTION_L2_SUM" not in runtime_text and "companion_matches(EXECUTION_L2_PATH" not in runtime_text,
        "failure.local_runtime_sidecars_valid": all(companion_matches(path) for path in RUNTIME_REQUIRED_SIDECARS),
        "failure.staged_runtime_sidecars_absent": all(not path.exists() for path in staged_required_sidecars),
        "failure.staged_l2_sidecar_absent_noncausal": not staged_l2_sidecar.exists() and readiness.get("checks", {}).get("independent_execution_package_l2") is True,
        "failure.stage_binding_set_omits_runtime_sidecars": all(path.relative_to(ROOT).as_posix() not in bindings for path in required_sidecars),
        "failure.stage_binding_set_omits_l2_sidecar": l2_sidecar.relative_to(ROOT).as_posix() not in bindings,
        "lifecycle.launch_readiness_executed": readiness.get("check_count") == 15,
        "lifecycle.fresh_authority_verified": readiness.get("checks", {}).get("fresh_attempt_authority") is True,
        "lifecycle.l2_verified": readiness.get("checks", {}).get("independent_execution_package_l2") is True,
        "lifecycle.namespace_absent": readiness.get("checks", {}).get("official_namespace_absent") is True and not RUN_ROOT.exists(),
        "lifecycle.attempt_marker_absent": readiness.get("checks", {}).get("attempt_marker_absent") is True,
        "lifecycle.detached_intent_absent": readiness.get("checks", {}).get("detached_launch_intent_absent") is True,
        "lifecycle.detached_process_absent": readiness.get("checks", {}).get("detached_process_record_absent") is True,
        "lifecycle.worker_status_absent": readiness.get("checks", {}).get("worker_terminal_status_absent") is True,
        "lifecycle.no_forbidden_remote_files": not observed_forbidden,
        "lifecycle.no_training_or_evaluator_paths": not any(fragment in "/" + relative for relative in remote_relatives for fragment in ("/qualification/dev/", "/qualification/retention/", "/qualification/holdout-exactly-once/", "/selected/bf16/", "/attempt-0001/")),
        "lifecycle.gpu_idle_no_model_process": "Processes                             : None" in nvidia and "Used                              : 0 MiB" in nvidia,
        "cardinality.one_authority": len(list(AUTHORITY.parent.glob(AUTHORITY.name))) == 1,
        "cardinality.one_intent": len(list(BUILD.glob("backend-submission-intent.json"))) == 1,
        "cardinality.one_submit_invocation": raw_text.count("Created new experiment") == 1,
        "cardinality.one_backend_job": row.get("n_jobs") == 1,
        "cardinality.zero_retries": not any("attempt-0002" in path.as_posix() for path in BUILD.rglob("*")),
    }

    hygiene_paths = [path for path in TERMINAL.rglob("*") if path.is_file()] + [
        AUTHORITY, AUTHORITY.with_suffix(AUTHORITY.suffix + ".sha256"), INTENT,
        INTENT.with_suffix(INTENT.suffix + ".sha256"), RESULT,
        RESULT.with_suffix(RESULT.suffix + ".sha256"), RAW, MONITOR, CONTRACT,
        CONTRACT.with_suffix(CONTRACT.suffix + ".sha256"), MANIFEST,
        MANIFEST.with_suffix(MANIFEST.suffix + ".sha256"), SELF_TEST,
        SELF_TEST.with_suffix(SELF_TEST.suffix + ".sha256"), L2,
        L2.with_suffix(L2.suffix + ".sha256"), STAGE_BUILDER, RUNTIME,
    ]
    hygiene = scan_hygiene(hygiene_paths)
    checks.update({
        "hygiene.complete_byte_coverage": hygiene["complete_byte_coverage"] is True,
        "hygiene.no_size_cap": hygiene["size_limit_bytes"] is None,
        "hygiene.no_skips": hygiene["skipped_file_count"] == 0,
        "hygiene.no_credential_signatures": not hygiene["findings"],
    })
    failed = sorted(name for name, value in checks.items() if value is not True)
    require(not failed, f"terminal S4 audit failed checks: {failed}")

    return {
        "schema_version": 2,
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "TERMINAL_S4_PREATTEMPT_NO_GO_EVIDENCE_COMPLETE",
        "failure_taxonomy": "PREATTEMPT_RUNTIME_CHECKSUM_SIDECARS_OMITTED_FROM_STAGE",
        "claim_boundary": "Exactly one authorized S4 backend job reached terminal failed after non-consuming launch readiness rejected missing runtime checksum companions. No detached launch intent, attempt marker, worker, training, selector, dev, selected BF16 materialization, retention, holdout, model/evaluator execution, checkpoint, or quality result exists. No retry, resume, relaunch, repair-in-place, W4A8, RTL, FPGA/U280, or stage advance is authorized.",
        "check_count": len(checks),
        "failed_check_count": 0,
        "checks": dict(sorted(checks.items())),
        "cardinality": {"authority_count": 1, "intent_count": 1, "submit_invocation_count": 1, "backend_experiment_count": 1, "backend_job_count": 1, "retry_count": 0},
        "backend": {"experiment": EXPERIMENT, "job_name": JOB, "job_id": JOB_ID, "status": "failed", "all_terminated": True, "result_size": row.get("size")},
        "failure": {
            "classification": "EXECUTION_INTERLOCK_PREVENTED_ATTEMPT_START",
            "readiness_status": readiness["status"],
            "readiness_failure_taxonomy": readiness["failure_taxonomy"],
            "first_failed_gate": readiness["first_failed_gate"],
            "readiness_observed_at_utc": readiness["observed_at_utc"],
            "runtime_required_missing_sidecars": [path.relative_to(ROOT).as_posix() for path in required_sidecars],
            "noncausal_absent_sidecar": l2_sidecar.relative_to(ROOT).as_posix(),
            "noncausal_reason": "independent_execution_package_l2 passed and runtime.py has no L2-sidecar companion check",
            "root_cause": "prepare_stage.required_repo_bindings copied the bound JSON artifacts but omitted their checksum companions; runtime.py requires the contract, execution-manifest, and self-test companions.",
        },
        "execution_boundaries": {
            "backend_container_executed": True, "launch_readiness_executed": True,
            "detached_launcher_intent_published": False, "attempt_marker_created": False,
            "worker_executed": False, "training_executed": False, "selector_executed": False,
            "dev_executed": False, "selected_bf16_materialized": False,
            "retention_executed": False, "holdout_executed": False,
            "model_executed": False, "evaluator_executed": False,
            "checkpoint_created": False, "quality_conclusion": None,
            "evaluator_no_execution_classification": "PREATTEMPT_INTERLOCK_NO_EVALUATOR_EXECUTION",
            "retry_resume_relaunch_cancel_permitted": False, "downstream_authority_opened": False,
        },
        "frozen_identity": {"package_identity_sha256": PACKAGE_IDENTITY, "contract_sha256": sha256(CONTRACT), "execution_tree_sha256": EXECUTION_TREE},
        "terminal_evidence": {"authority": record(AUTHORITY), "intent": record(INTENT), "submission_result": record(RESULT), "submission_raw": record(RAW), "backend_inventory": record(TERMINAL / "amlt-inventory-terminal.json"), "backend_stdout": record(stdout_path), "readiness": record(readiness_path), "remote_sha256sums": record(sums), "wrapper_console": record(wrapper_paths[0])},
        "hygiene": hygiene,
        "regression": "Only a distinct successor may repair this class: derive staged closure from runtime companion checks and add positive/negative coverage for all three required checksum companions before any new authority. S4 itself remains immutable and may not be retried or repaired in place.",
    }


def write(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists() and not path.with_suffix(path.suffix + ".sha256").exists(), f"refusing to overwrite immutable audit: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    observed = build_audit()
    if args.check:
        require(companion_matches(OUTPUT), "terminal S4 v2 audit companion differs")
        existing = load(OUTPUT)
        for key in ("status", "failure_taxonomy", "check_count", "failed_check_count", "checks", "cardinality", "backend", "failure", "execution_boundaries", "frozen_identity"):
            require(existing.get(key) == observed.get(key), f"terminal S4 v2 audit field differs: {key}")
        print(f"ACE2_S4_TERMINAL_AUDIT_CHECK_PASS checks={observed['check_count']}")
        return 0
    write(OUTPUT, observed)
    print(f"ACE2_S4_TERMINAL_AUDIT_WRITTEN checks={observed['check_count']} sha256={sha256(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
