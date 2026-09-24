#!/usr/bin/env python3
"""Audit the sole S1 transport job after terminal backend completion.

This tool is deliberately read-only with respect to AMLT.  It may query status,
logs, and result listings, but it never submits, resumes, cancels, trains, or
accesses an evaluator.  The downloaded log tree must already exist.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-successor-s1-transport-attempt-operator-authority.json"
INTENT = ROOT / "build/bf16-successor-s1-transport/backend-submission-intent.json"
CLIENT_RESULT = ROOT / "build/bf16-successor-s1-transport/backend-submission-result.json"
CLIENT_RAW = ROOT / "build/bf16-successor-s1-transport/backend-submission.raw.txt"
PRETERMINAL_AUDIT = ROOT / "build/bf16-successor-s1-transport/backend-audit-20260808T110503Z.json"
PRETERMINAL_REVIEW = ROOT / "research/raw/specification/bf16-successor-s1-transport-terminal-fresh-review-f0548a712bb0-20260808.json"
PAYLOAD = ROOT / "research/execution/qwen25_bf16_successor_s1_transport_20260808/preauthority_only.py"
DOWNLOAD_ROOT = ROOT / "build/bf16-successor-s1-transport/backend-terminal-download-20260808T111251Z"
DOWNLOADED_STDOUT = DOWNLOAD_ROOT / "logs/ace2-bf16-successor-s1-transport-20260808-preauthority/ace2-bf16-successor-s1-transport-gate/stdout.txt"
RESULTS_ROOT = DOWNLOAD_ROOT / "results"
TERMINAL_AUDIT = ROOT / "build/bf16-successor-s1-transport/backend-terminal-audit-20260808T111251Z.json"
EXPERIMENT = "ace2-bf16-successor-s1-transport-20260808-preauthority"
JOB = "ace2-bf16-successor-s1-transport-gate"
JOB_ID = "ace2-bf16-successor-s1-transport-20260808-pre-0e51f05b"
PAYLOAD_MARKER = "ACE2_BF16_SUCCESSOR_S1_PREAUTHORITY_ONLY no_training_no_evaluator_no_checkpoint"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    if not companion.is_file():
        return False
    return companion.read_text(encoding="ascii").split() == [sha256(path), path.name]


def record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def run_text(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    require(args.check != (args.output is not None), "select exactly one of --output or --check")
    if args.check:
        output = TERMINAL_AUDIT
    else:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        require(not output.exists(), f"refusing to overwrite immutable audit: {output}")

    pipeline = load_json(PIPELINE)
    authority = load_json(AUTHORITY)
    intent = load_json(INTENT)
    client_result = load_json(CLIENT_RESULT)
    preterminal_audit = load_json(PRETERMINAL_AUDIT)
    preterminal_review = load_json(PRETERMINAL_REVIEW)

    amlt = Path(str(authority.get("runtime_binding", {}).get("amlt_executable", "")))
    project = Path(str(authority.get("runtime_binding", {}).get("project", "")))
    require(amlt.is_file(), "authority-bound AMLT executable is missing")
    require(project.is_dir(), "authority-bound AMLT project is missing")

    status_proc = run_text(
        [str(amlt), "--json-tables", "status", "--no-update-db", EXPERIMENT, f":{JOB}"],
        project,
    )
    require(status_proc.returncode == 0, f"AMLT status failed: {status_proc.stderr[-500:]}")
    status_rows = json.loads(status_proc.stdout)
    require(isinstance(status_rows, list) and len(status_rows) == 1, "expected exactly one status row")
    status_row = status_rows[0]

    logs_proc = run_text([str(amlt), "logs", "list", EXPERIMENT, f":{JOB}"], project)
    results_proc = run_text([str(amlt), "results", "list", EXPERIMENT, f":{JOB}"], project)
    result_files = sorted(path for path in RESULTS_ROOT.rglob("*") if path.is_file())
    payload_text = PAYLOAD.read_text(encoding="utf-8")
    downloaded_text = DOWNLOADED_STDOUT.read_text(encoding="utf-8")
    client_raw_text = CLIENT_RAW.read_text(encoding="utf-8")

    authority_count = len(list(AUTHORITY.parent.glob(AUTHORITY.name)))
    intent_count = len(list(INTENT.parent.glob("backend-submission-intent.json")))
    checks = {
        "pipeline.current_stage_specification": pipeline.get("current_stage") == "specification",
        "authority.count_one": authority_count == 1,
        "authority.companion_matches": companion_matches(AUTHORITY),
        "authority.maximum_backend_jobs_one": authority.get("maximum_backend_jobs") == 1,
        "authority.no_retry_resume_replay": all(
            authority.get(key) is True for key in ("no_relaunch", "no_replay", "no_resume")
        ),
        "intent.count_one": intent_count == 1,
        "intent.companion_matches": companion_matches(INTENT),
        "intent.binds_authority": intent.get("authority_sha256") == sha256(AUTHORITY),
        "intent.target_exact": intent.get("experiment_name") == EXPERIMENT and intent.get("job_name") == JOB,
        "client_result.companion_matches": companion_matches(CLIENT_RESULT),
        "client_result.preserved_terminal_no_go": client_result.get("status") == "TERMINAL_SUBMISSION_NO_GO",
        "client_result.exit_zero": client_result.get("client_exit_code") == 0,
        "client_result.prompt_false_positive_preserved": client_result.get("prompt_detected") is True,
        "client_result.binds_intent": client_result.get("submission_intent_sha256") == sha256(INTENT),
        "client_result.binds_raw": client_result.get("raw_output_sha256") == sha256(CLIENT_RAW),
        "client_raw.created_one_experiment": client_raw_text.count(f"Created new experiment {EXPERIMENT}") == 1,
        "client_raw.reported_one_job": "with 1 job" in client_raw_text,
        "client_raw.normal_description_summary_present": "Description:" in client_raw_text,
        "preterminal_audit.companion_matches": companion_matches(PRETERMINAL_AUDIT),
        "preterminal_audit.observed_one_queued_job": preterminal_audit.get("backend_inventory", {}).get("match_count") == 1 and preterminal_audit.get("backend_inventory", {}).get("matches", [{}])[0].get("n_jobs") == 1 and preterminal_audit.get("backend_inventory", {}).get("matches", [{}])[0].get("job_status") == "Queued",
        "preterminal_review.companion_matches": companion_matches(PRETERMINAL_REVIEW),
        "preterminal_review.counted_one_authority_intent_job": preterminal_review.get("acceptance", {}).get("authority_count") == 1 and preterminal_review.get("acceptance", {}).get("intent_count") == 1 and preterminal_review.get("acceptance", {}).get("backend_target_job_count") == 1,
        "backend.status_exact_job": status_row.get("job_id") == JOB_ID and status_row.get("job_name") == f":{JOB}" and status_row.get("experiment") == EXPERIMENT,
        "backend.status_terminal_failed": str(status_row.get("status", "")).lower() == "failed",
        "backend.status_command_inert_payload": status_row.get("command") == ["python preauthority_only.py"],
        "backend.status_no_retry": status_row.get("status") == "failed" and preterminal_review.get("acceptance", {}).get("retry_count") == 0,
        "payload.configured_exit_97": "return 97" in payload_text,
        "payload.no_training_evaluator_checkpoint_marker": all(
            fragment in payload_text
            for fragment in (
                "ACE2_BF16_SUCCESSOR_S1_PREAUTHORITY_ONLY ",
                "no_training_no_evaluator_no_checkpoint",
            )
        ),
        "download.stdout_exists": DOWNLOADED_STDOUT.is_file(),
        "download.stdout_contains_exact_marker_once": downloaded_text.count(PAYLOAD_MARKER) == 1,
        "download.log_listing_succeeded": logs_proc.returncode == 0 and "blob://" in logs_proc.stdout,
        "download.results_listing_succeeded": results_proc.returncode == 0,
        "download.result_file_count_zero": len(result_files) == 0,
        "s1.model_evaluator_checkpoint_absent": not (ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-successor-s1").exists(),
    }
    failed = sorted(name for name, value in checks.items() if value is not True)
    require(not failed, f"terminal S1 audit failed checks: {failed}")

    audit = {
        "schema_version": 1,
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "TERMINAL_S1_TRANSPORT_EVIDENCE_COMPLETE",
        "failure_taxonomy": "INERT_TRANSPORT_GATE_EXECUTED_TERMINAL_FAILED",
        "claim_boundary": "Exactly one authorized S1 transport-only AMLT invocation created exactly one backend experiment and one job. The job executed only the inert preauthority payload, printed the bound no-training/no-evaluator/no-checkpoint marker, and terminated failed. This is transport evidence, not model, evaluator, quality, W4A8, RTL, synthesis/PPA, FPGA, U280, or project-completion evidence.",
        "cardinality": {
            "authority_count": authority_count,
            "intent_count": intent_count,
            "submit_invocation_count": preterminal_review.get("acceptance", {}).get("submit_invocation_count"),
            "backend_experiment_count": 1,
            "backend_job_count": 1,
            "retry_count": preterminal_review.get("acceptance", {}).get("retry_count"),
        },
        "client_result": {
            "status": client_result.get("status"),
            "client_exit_code": client_result.get("client_exit_code"),
            "prompt_detected": client_result.get("prompt_detected"),
            "classification": "RAW_CLIENT_RESULT_MISCLASSIFICATION_PRESERVED",
            "prompt_marker_context": "normal AMLT terminal Description summary after experiment/job creation",
        },
        "backend": {
            "experiment": EXPERIMENT,
            "job_name": JOB,
            "job_id": JOB_ID,
            "status": str(status_row.get("status", "")).lower(),
            "status_message": status_row.get("status_message"),
            "service": status_row.get("service"),
            "cluster": status_row.get("cluster"),
            "vc": status_row.get("vc"),
            "command": status_row.get("command"),
            "configured_payload_exit_code": 97,
            "backend_reported_exit_code": None,
            "all_terminated": True,
        },
        "download": {
            "root": DOWNLOAD_ROOT.relative_to(ROOT).as_posix(),
            "stdout": record(DOWNLOADED_STDOUT),
            "stdout_marker": PAYLOAD_MARKER,
            "result_file_count": len(result_files),
            "result_files": [record(path) for path in result_files],
        },
        "evidence": {
            "authority": record(AUTHORITY),
            "intent": record(INTENT),
            "client_result": record(CLIENT_RESULT),
            "client_raw": record(CLIENT_RAW),
            "preterminal_audit": record(PRETERMINAL_AUDIT),
            "preterminal_review": record(PRETERMINAL_REVIEW),
            "payload": record(PAYLOAD),
        },
        "execution_boundaries": {
            "transport_payload_executed": True,
            "training_executed": False,
            "model_executed": False,
            "evaluator_accessed": False,
            "checkpoint_created": False,
            "quality_conclusion": None,
            "retry_resume_relaunch_cancel_permitted": False,
            "downstream_authority_opened": False,
        },
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "reproduction_command": f"PYTHONDONTWRITEBYTECODE=1 python3 tools/{Path(__file__).name} --output {output.relative_to(ROOT).as_posix()}",
    }

    if args.check:
        require(companion_matches(output), "terminal audit checksum companion differs")
        existing = load_json(output)
        for key in (
            "status",
            "failure_taxonomy",
            "claim_boundary",
            "cardinality",
            "client_result",
            "backend",
            "download",
            "evidence",
            "execution_boundaries",
            "check_count",
            "failed_check_count",
            "checks",
        ):
            require(existing.get(key) == audit.get(key), f"terminal audit field differs: {key}")
        print(f"ACE2_S1_TERMINAL_AUDIT_CHECK_PASS checks={len(checks)} job_id={JOB_ID}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    companion = output.with_suffix(output.suffix + ".sha256")
    companion.write_text(f"{sha256(output)}  {output.name}\n", encoding="ascii")
    print(f"ACE2_S1_TERMINAL_AUDIT_PASS checks={len(checks)} job_id={JOB_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
