#!/usr/bin/env python3
"""Audit the sole terminal S3 backend job without retrying or resuming it."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = ROOT / "build/bf16-full-finetune-successor-s3"
TERMINAL_ROOT = BUILD_ROOT / "backend-terminal-20260808T122720Z"
TERMINAL_AUDIT = BUILD_ROOT / "backend-terminal-audit-20260808T122720Z.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s3-attempt-operator-authority.json"
INTENT = BUILD_ROOT / "backend-submission-intent.json"
CLIENT_RESULT = BUILD_ROOT / "backend-submission-result.json"
CLIENT_RAW = BUILD_ROOT / "backend-submission.raw.txt"
POST_SUBMIT_INVENTORY = BUILD_ROOT / "backend-inventory-after-submit.json"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_SUCCESSOR_S3_EXECUTION_PACKAGE_CONTRACT.json"
SOURCE_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V2_CONTRACT.json"
SOURCE_AUDIT = ROOT / "build/ace2_chat_diagnostics/qwen2.5-0.5b-instruct-local-audit-20260806.json"
STAGED_ROOT = ROOT / "research/execution/qwen25_bf16_full_finetune_successor_s3_20260808/staged/repo"
STAGED_SOURCE_CONTRACT = STAGED_ROOT / SOURCE_CONTRACT.relative_to(ROOT)
STAGED_SOURCE_AUDIT = STAGED_ROOT / SOURCE_AUDIT.relative_to(ROOT)
RUN_ONCE = ROOT / "research/execution/qwen25_bf16_full_finetune_successor_s3_20260808/run_once.sh"
LAUNCHER = ROOT / "pilot/qwen25_05b_bf16_full_finetune_successor_s3_runner/launch_detached.py"
COMMON = ROOT / "pilot/qwen25_05b_bf16_full_finetune_successor_s3_runner/common.py"
RUNTIME = ROOT / "pilot/qwen25_05b_bf16_full_finetune_successor_s3_runner/runtime.py"
WORKER = ROOT / "pilot/qwen25_05b_bf16_full_finetune_successor_s3_runner/run_worker.py"
EXPERIMENT = "ace2-bf16-full-finetune-successor-s3-20260808-attempt-0001"
JOB = "ace2-bf16-full-finetune-successor-s3-attempt-0001"
JOB_ID = "ace2-bf16-full-finetune-successor-s3-20260808-0fa95e5e"
OFFICIAL_RUN_PREFIX = "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s3"
EXPECTED_LIFECYCLE = [
    "full_training_from_pinned_source_model",
    "ordered_synthetic_probe_selector",
    "single_locked_checkpoint_dev_qualification",
    "selected_bf16_materialization",
    "retention",
    "exactly_once_holdout",
]
EXPECTED_WORKER_SEQUENCE = [
    "train.py",
    "evaluate_dev.py",
    "select_model.py",
    "retention.py",
    "evaluate_holdout.py",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    return companion.is_file() and companion.read_text(encoding="ascii").split() == [sha256(path), path.name]


def record(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    require(len(matches) == 1, f"expected one {pattern} below {root}, observed {len(matches)}")
    return matches[0]


def worker_sequence(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name) or node.target.id != "name":
            continue
        if not isinstance(node.iter, (ast.Tuple, ast.List)):
            continue
        values = [item.value for item in node.iter.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)]
        if values == EXPECTED_WORKER_SEQUENCE:
            return values
    return []


def verify_remote_sums(path: Path) -> tuple[bool, int]:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        candidate = path.parent / relative.removeprefix("*").removeprefix("./")
        if not candidate.is_file() or sha256(candidate) != digest:
            return False, count
        count += 1
    return True, count


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
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            skipped.append({"path": relative, "reason": str(exc)})
            continue
        records.append({"bytes": len(raw), "path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "utf8": True})
        for label, pattern in signatures.items():
            if pattern.search(decoded):
                findings.append({"path": relative, "signature": label})
    return {
        "bytes_scanned": sum(item["bytes"] for item in records),
        "complete_byte_coverage": len(records) == len(set(paths)) and not skipped,
        "file_count": len(records),
        "files": records,
        "findings": findings,
        "size_limit_bytes": None,
        "skipped_file_count": len(skipped),
        "skipped_files": skipped,
        "status": "PASS_COMPLETE_NO_SIZE_CAP" if not findings and not skipped else "HYGIENE_NO_GO",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    require(args.check != (args.output is not None), "select exactly one of --output or --check")
    output = TERMINAL_AUDIT if args.check else (args.output if args.output.is_absolute() else ROOT / args.output)
    if not args.check:
        require(not output.exists(), f"refusing to overwrite immutable audit: {output}")

    required = [
        TERMINAL_ROOT,
        PIPELINE,
        AUTHORITY,
        INTENT,
        CLIENT_RESULT,
        CLIENT_RAW,
        POST_SUBMIT_INVENTORY,
        CONTRACT,
        SOURCE_CONTRACT,
        SOURCE_AUDIT,
        STAGED_ROOT,
        STAGED_SOURCE_CONTRACT,
        RUN_ONCE,
        LAUNCHER,
        COMMON,
        RUNTIME,
        WORKER,
    ]
    for path in required:
        require(path.exists(), f"required terminal evidence is absent: {path}")

    status_rows = load_json(TERMINAL_ROOT / "status.json")
    require(isinstance(status_rows, list) and len(status_rows) == 1, "expected exactly one terminal status row")
    status_row = status_rows[0]
    authority = load_json(AUTHORITY)
    intent = load_json(INTENT)
    client_result = load_json(CLIENT_RESULT)
    inventory = load_json(POST_SUBMIT_INVENTORY)
    contract = load_json(CONTRACT)
    source_contract = load_json(SOURCE_CONTRACT)
    staged_source_contract = load_json(STAGED_SOURCE_CONTRACT)
    pipeline = load_json(PIPELINE)

    results_root = TERMINAL_ROOT / "results"
    logs_root = TERMINAL_ROOT / "logs"
    remote_sums = one_match(results_root, "SHA256SUMS")
    remote_evidence_root = remote_sums.parent
    readiness_path = one_match(results_root, "s3-launch-readiness.json")
    user_log = one_match(logs_root, f"{JOB_ID}-master-0.log")
    infrastructure_log = one_match(logs_root, "master-0")
    remote_wrapper = remote_evidence_root / "s3-lifecycle-wrapper-console.log"
    wrapper_matches = sorted(results_root.rglob("s3-lifecycle-wrapper-console.log"))
    require(len(wrapper_matches) == 2 and remote_wrapper in wrapper_matches, "expected the two AMLT wrapper-log copies")
    duplicate_wrapper = next(path for path in wrapper_matches if path != remote_wrapper)
    remote_authority = remote_evidence_root / AUTHORITY.relative_to(ROOT)
    remote_pipeline = remote_evidence_root / PIPELINE.relative_to(ROOT)
    remote_l2 = remote_evidence_root / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s3/execution-package-l2-acceptance.json"
    remote_manifest = remote_evidence_root / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s3/s3-execution-package-manifest.json"
    remote_self_test = remote_evidence_root / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s3/s3-execution-package-self-test.json"
    remote_run_once = remote_evidence_root / "run_once.sh"
    remote_amlt = remote_evidence_root / "amlt.yaml"
    backend_started = one_match(results_root, "backend-started-at-utc.txt")
    nvidia_smi = one_match(results_root, "nvidia-smi-q.raw.txt")

    readiness = load_json(readiness_path)
    client_raw_text = CLIENT_RAW.read_text(encoding="utf-8")
    user_log_text = user_log.read_text(encoding="utf-8")
    infrastructure_text = infrastructure_log.read_text(encoding="utf-8")
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    remote_files = sorted(path for path in remote_evidence_root.rglob("*") if path.is_file())
    remote_relatives = [path.relative_to(remote_evidence_root).as_posix() for path in remote_files]
    terminal_files = sorted(path for path in TERMINAL_ROOT.rglob("*") if path.is_file())
    hygiene_paths = terminal_files + [
        AUTHORITY,
        AUTHORITY.with_suffix(AUTHORITY.suffix + ".sha256"),
        INTENT,
        INTENT.with_suffix(INTENT.suffix + ".sha256"),
        CLIENT_RESULT,
        CLIENT_RESULT.with_suffix(CLIENT_RESULT.suffix + ".sha256"),
        CLIENT_RAW,
        POST_SUBMIT_INVENTORY,
        POST_SUBMIT_INVENTORY.with_suffix(POST_SUBMIT_INVENTORY.suffix + ".sha256"),
        CONTRACT,
        SOURCE_CONTRACT,
        SOURCE_AUDIT,
        RUN_ONCE,
        LAUNCHER,
        COMMON,
        RUNTIME,
        WORKER,
    ]
    hygiene = scan_hygiene(hygiene_paths)
    sums_valid, sums_count = verify_remote_sums(remote_sums)
    sequence = worker_sequence(WORKER)

    source = source_contract["source_model"]
    readiness_checks = readiness.get("checks", {})
    all_other_readiness_checks_pass = all(
        value is True for name, value in readiness_checks.items() if name != "source_snapshot_exact"
    )
    forbidden_remote_names = {
        "ATTEMPT_CONSUMPTION_MARKER.json",
        "s3-detached-launch-intent.json",
        "s3-detached-process.json",
        "s3-launch-terminal-status.json",
        "s3-worker-terminal-status.json",
        "training_result.json",
        "checkpoint_lock.json",
        "TERMINAL_GO.json",
        "TERMINAL_NO_GO.json",
        "s3-status-after-terminal.raw.txt",
    }
    observed_forbidden_names = sorted({Path(relative).name for relative in remote_relatives} & forbidden_remote_names)
    official_namespace_files = sorted(
        relative for relative in remote_relatives if relative == OFFICIAL_RUN_PREFIX or relative.startswith(OFFICIAL_RUN_PREFIX + "/")
    )
    authority_count = len(list(AUTHORITY.parent.glob(AUTHORITY.name)))
    intent_count = len(list(INTENT.parent.glob("backend-submission-intent.json")))

    checks = {
        "pipeline.current_stage_specification": pipeline.get("current_stage") == "specification",
        "pipeline.hash_matches_authority": sha256(PIPELINE) == authority.get("pipeline_state_sha256"),
        "authority.count_one": authority_count == 1,
        "authority.companion_matches": companion_matches(AUTHORITY),
        "authority.status_exact": authority.get("status") == "AUTHORIZE_SINGLE_BF16_FULL_FINETUNE_SUCCESSOR_S3_ATTEMPT",
        "authority.no_retry_resume_relaunch": all(authority.get(key) is True for key in ("no_relaunch", "no_resubmit", "no_resume")),
        "authority.maximum_backend_jobs_one": authority.get("backend", {}).get("maximum_backend_jobs") == 1,
        "authority.lifecycle_exact": authority.get("authorized_lifecycle") == EXPECTED_LIFECYCLE,
        "contract.lifecycle_exact": contract.get("future_authorized_lifecycle_order") == EXPECTED_LIFECYCLE,
        "intent.count_one": intent_count == 1,
        "intent.companion_matches": companion_matches(INTENT),
        "intent.binds_authority": intent.get("authority_sha256") == sha256(AUTHORITY),
        "intent.target_exact": intent.get("experiment_name") == EXPERIMENT and intent.get("job_name") == JOB,
        "intent.logical_command_exact": intent.get("command") == authority.get("authorized_logical_argv"),
        "client_result.companion_matches": companion_matches(CLIENT_RESULT),
        "client_result.exit_zero": client_result.get("client_exit_code") == 0,
        "client_result.no_prompt": client_result.get("prompt_detected") is False,
        "client_result.binds_intent": client_result.get("submission_intent_sha256") == sha256(INTENT),
        "client_result.binds_raw": client_result.get("raw_output_sha256") == sha256(CLIENT_RAW),
        "client_raw.created_one_experiment": client_raw_text.count("Created new experiment") == 1 and client_raw_text.count(EXPERIMENT) >= 2,
        "client_raw.reported_one_job": "with 1 job" in client_raw_text,
        "inventory.companion_matches": companion_matches(POST_SUBMIT_INVENTORY),
        "inventory.one_experiment_one_job": inventory.get("matching_experiment_count") == 1 and inventory.get("matching_job_count") == 1,
        "inventory.target_exact": inventory.get("experiment", {}).get("experiment_name") == EXPERIMENT and inventory.get("experiment", {}).get("_first_job_name") == JOB,
        "backend.status_one_row": len(status_rows) == 1,
        "backend.identity_exact": status_row.get("experiment") == EXPERIMENT and status_row.get("job_name") == f":{JOB}" and status_row.get("job_id") == JOB_ID,
        "backend.terminal_failed": str(status_row.get("status", "")).lower() == "failed",
        "backend.status_message_preserved": status_row.get("status_message") == "[nfs-sidecar] Error; [master] Error",
        "backend.command_exact": status_row.get("command") == ["umask 077", "timeout --signal=TERM --kill-after=10m 47h bash run_once.sh \"$AMLT_OUTPUT_DIR\""],
        "backend.target_exact": status_row.get("service") == "volcano" and status_row.get("cluster") == "msr02" and status_row.get("vc") == "bonete01" and status_row.get("sku") == "G1",
        "download.list_commands_succeeded": (TERMINAL_ROOT / "list-command-exits.txt").read_text(encoding="utf-8").split() == ["status=0", "logs_list=0", "results_list=0"],
        "download.user_log_exists": user_log.is_file(),
        "download.infrastructure_log_exists": infrastructure_log.is_file(),
        "download.wrapper_copies_match_user_log": sha256(user_log) == sha256(remote_wrapper) == sha256(duplicate_wrapper),
        "download.remote_sums_complete": sums_valid and sums_count == len(remote_files) - 1,
        "download.remote_authority_matches": sha256(remote_authority) == sha256(AUTHORITY),
        "download.remote_pipeline_matches": sha256(remote_pipeline) == sha256(PIPELINE),
        "download.remote_l2_matches_authority": sha256(remote_l2) == authority.get("execution_l2_sha256"),
        "download.remote_manifest_matches_authority": sha256(remote_manifest) == authority.get("execution_manifest_sha256"),
        "download.remote_self_test_matches_authority": sha256(remote_self_test) == authority.get("execution_self_test_sha256"),
        "download.remote_run_once_matches": sha256(remote_run_once) == sha256(RUN_ONCE),
        "download.remote_amlt_matches_contract": sha256(remote_amlt) == contract.get("package_files", {}).get("artifact_017", {}).get("sha256"),
        "failure.user_log_exact_gate": user_log_text.count("RuntimeError: launch preflight failed: source_snapshot_exact") == 1,
        "failure.readiness_status_preattempt_no_go": readiness.get("status") == "PREATTEMPT_NO_GO",
        "failure.readiness_taxonomy_exact": readiness.get("failure_taxonomy") == "EXECUTION_INTERLOCK_NOT_SATISFIED",
        "failure.first_failed_gate_exact": readiness.get("first_failed_gate") == "source_snapshot_exact",
        "failure.only_readiness_check_failed": readiness_checks.get("source_snapshot_exact") is False and all_other_readiness_checks_pass,
        "failure.missing_path_exact": SOURCE_AUDIT.as_posix().replace(ROOT.as_posix(), "/root/code/staged/repo") in readiness.get("errors", {}).get("source_snapshot_exact", ""),
        "failure.local_source_audit_hash_exact": SOURCE_AUDIT.is_file() and sha256(SOURCE_AUDIT) == source.get("local_audit_sha256"),
        "failure.source_contract_path_exact": source.get("local_audit_path") == SOURCE_AUDIT.relative_to(ROOT).as_posix(),
        "failure.staged_source_contract_exact": staged_source_contract == source_contract,
        "failure.staged_source_audit_absent": not STAGED_SOURCE_AUDIT.exists(),
        "lifecycle.worker_sequence_exact": sequence == EXPECTED_WORKER_SEQUENCE,
        "lifecycle.readiness_before_detached_intent": launcher_text.index("readiness = audit_launch_readiness()") < launcher_text.index("exclusive_write_json(LAUNCH_INTENT, intent)"),
        "lifecycle.backend_started_before_readiness": backend_started.read_text(encoding="utf-8").strip() < readiness.get("observed_at_utc", ""),
        "lifecycle.no_forbidden_state_files": not observed_forbidden_names,
        "lifecycle.official_namespace_absent": not official_namespace_files and readiness.get("official_namespace_exists") is False,
        "lifecycle.no_training_or_evaluator_artifacts": not any(fragment in "/" + relative for relative in remote_relatives for fragment in ("/qualification/dev/", "/qualification/retention/", "/qualification/holdout-exactly-once/", "/selected/bf16/", "/training_result.json")),
        "lifecycle.gpu_idle_no_model_process": "Processes                             : None" in nvidia_smi.read_text(encoding="utf-8") and "Used                              : 0 MiB" in nvidia_smi.read_text(encoding="utf-8"),
        "lifecycle.infrastructure_reached_user_command": "NFS sidecar is ready!" in infrastructure_text and "RuntimeError: launch preflight failed: source_snapshot_exact" in infrastructure_text,
        "hygiene.complete_byte_coverage": hygiene["complete_byte_coverage"] is True,
        "hygiene.no_size_cap": hygiene["size_limit_bytes"] is None,
        "hygiene.no_skips": hygiene["skipped_file_count"] == 0,
        "hygiene.no_credential_signatures": not hygiene["findings"],
    }
    failed = sorted(name for name, value in checks.items() if value is not True)
    require(not failed, f"terminal S3 audit failed checks: {failed}")

    audit = {
        "schema_version": 1,
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "TERMINAL_S3_PREATTEMPT_NO_GO_EVIDENCE_COMPLETE",
        "failure_taxonomy": "PREATTEMPT_SOURCE_SNAPSHOT_AUDIT_OMITTED_FROM_STAGE",
        "claim_boundary": "Exactly one authorized S3 backend job reached terminal failed after the non-consuming launch-readiness audit rejected an omitted source-audit file. No detached launch intent, attempt marker, training, selector, dev, selected BF16 materialization, retention, holdout, model/evaluator execution, checkpoint, or quality result exists. Retry, resubmit, relaunch, resume, repair-in-place, W4A8, RTL, synthesis/PPA, FPGA, U280, and stage advance remain forbidden.",
        "cardinality": {
            "authority_count": authority_count,
            "intent_count": intent_count,
            "submit_invocation_count": client_raw_text.count("Created new experiment"),
            "backend_experiment_count": inventory.get("matching_experiment_count"),
            "backend_job_count": inventory.get("matching_job_count"),
            "retry_count": 0,
        },
        "backend": {
            "experiment": EXPERIMENT,
            "job_name": JOB,
            "job_id": JOB_ID,
            "status": status_row.get("status"),
            "status_message": status_row.get("status_message"),
            "result_size": status_row.get("size"),
            "service": status_row.get("service"),
            "cluster": status_row.get("cluster"),
            "vc": status_row.get("vc"),
            "sku": status_row.get("sku"),
            "command": status_row.get("command"),
            "all_terminated": True,
        },
        "failure": {
            "first_failed_gate": readiness.get("first_failed_gate"),
            "readiness_status": readiness.get("status"),
            "readiness_failure_taxonomy": readiness.get("failure_taxonomy"),
            "readiness_observed_at_utc": readiness.get("observed_at_utc"),
            "error": readiness.get("errors", {}).get("source_snapshot_exact"),
            "required_source_audit_path": source.get("local_audit_path"),
            "required_source_audit_sha256": source.get("local_audit_sha256"),
            "local_source_audit_present": SOURCE_AUDIT.is_file(),
            "staged_source_audit_present": STAGED_SOURCE_AUDIT.is_file(),
            "classification": "EXECUTION_INTERLOCK_PREVENTED_ATTEMPT_CONSUMPTION",
        },
        "lifecycle_order": {
            "authorized": EXPECTED_LIFECYCLE,
            "configured_worker_sequence": sequence,
            "observed": ["backend_setup", "non_consuming_launch_readiness", "terminal_preattempt_no_go"],
            "stopped_before": "detached_launch_intent_and_attempt_marker",
        },
        "execution_boundaries": {
            "backend_container_executed": True,
            "launch_readiness_executed": True,
            "detached_launcher_intent_published": False,
            "attempt_marker_created": False,
            "training_executed": False,
            "selector_executed": False,
            "dev_executed": False,
            "selected_bf16_materialized": False,
            "retention_executed": False,
            "holdout_executed": False,
            "model_executed": False,
            "evaluator_executed": False,
            "evaluator_no_execution_classification": "PREATTEMPT_INTERLOCK_NO_EVALUATOR_EXECUTION",
            "checkpoint_created": False,
            "quality_conclusion": None,
            "terminal_disposition_requested": "EVIDENCE_BACKED_TERMINAL_NO_GO_WITHOUT_RETRY",
            "retry_resume_relaunch_cancel_permitted": False,
            "downstream_authority_opened": False,
        },
        "download": {
            "root": TERMINAL_ROOT.relative_to(ROOT).as_posix(),
            "terminal_file_count": len(terminal_files),
            "terminal_bytes": sum(path.stat().st_size for path in terminal_files),
            "remote_evidence_file_count": len(remote_files),
            "remote_sha256sum_entry_count": sums_count,
            "user_log": record(user_log),
            "infrastructure_log": record(infrastructure_log),
            "readiness": record(readiness_path),
            "remote_sha256sums": record(remote_sums),
        },
        "hygiene": hygiene,
        "evidence": {
            "authority": record(AUTHORITY),
            "intent": record(INTENT),
            "client_result": record(CLIENT_RESULT),
            "client_raw": record(CLIENT_RAW),
            "post_submit_inventory": record(POST_SUBMIT_INVENTORY),
            "pipeline": record(PIPELINE),
            "contract": record(CONTRACT),
            "source_contract": record(SOURCE_CONTRACT),
            "local_source_audit": record(SOURCE_AUDIT),
            "launcher": record(LAUNCHER),
            "worker": record(WORKER),
        },
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "reproduction_command": f"PYTHONDONTWRITEBYTECODE=1 python3 tools/{Path(__file__).name} --check",
    }

    if args.check:
        require(companion_matches(output), "terminal audit checksum companion differs")
        existing = load_json(output)
        for key in (
            "status",
            "failure_taxonomy",
            "claim_boundary",
            "cardinality",
            "backend",
            "failure",
            "lifecycle_order",
            "execution_boundaries",
            "download",
            "hygiene",
            "evidence",
            "check_count",
            "failed_check_count",
            "checks",
        ):
            require(existing.get(key) == audit.get(key), f"terminal audit field differs: {key}")
        print(f"ACE2_S3_TERMINAL_AUDIT_CHECK_PASS checks={len(checks)} job_id={JOB_ID}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    companion = output.with_suffix(output.suffix + ".sha256")
    companion.write_text(f"{sha256(output)}  {output.name}\n", encoding="ascii")
    print(f"ACE2_S3_TERMINAL_AUDIT_PASS checks={len(checks)} job_id={JOB_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
