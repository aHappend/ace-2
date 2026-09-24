#!/usr/bin/env python3
"""Run and immutably bind the official paired ACE-2 quality measurement."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ace2_quality_contracts import validate_oracle_manifest, validate_rtl_binding
from run_quality_gate import run_gate


ROOT = Path(__file__).resolve().parents[1]
QUALITY_RAW = ROOT / "benchmark" / "raw" / "quality"
LATEST = ROOT / "benchmark" / "raw" / "latest"
POINTER = LATEST / "quality_official_pointer.json"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
VENV_PIP = ROOT / ".venv" / "bin" / "pip"
PROMPT_MANIFEST = ROOT / "benchmark" / "quality" / "PROMPT_MANIFEST.json"
QUALITY_CONFIG = ROOT / "benchmark" / "quality" / "QUALITY_CONFIG.json"
QUALITY_REQUIREMENTS = ROOT / "benchmark" / "quality" / "requirements.txt"
LM_EVAL_TASKS = ROOT / "benchmark" / "quality" / "lm_eval_tasks"
FIXED_POINT_MODEL = ROOT / "tools" / "ace2_full_model_fixed_point.py"
QUALITY_GATE = ROOT / "tools" / "run_quality_gate.py"
QUALITY_CONTRACTS = ROOT / "tools" / "ace2_quality_contracts.py"
ORACLE_MANIFEST = ROOT / "reference" / "ORACLE_MANIFEST.json"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def source_artifacts(
    accepted_rtl: dict[str, Any],
    oracle_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    paths = [
        PROMPT_MANIFEST,
        QUALITY_CONFIG,
        QUALITY_REQUIREMENTS,
        FIXED_POINT_MODEL,
        QUALITY_GATE,
        QUALITY_CONTRACTS,
        ORACLE_MANIFEST,
        Path(__file__),
        ROOT / accepted_rtl["binding"]["path"],
        ROOT / accepted_rtl["manifest"]["path"],
        *sorted(path for path in LM_EVAL_TASKS.iterdir() if path.is_file()),
        *(ROOT / item["path"] for item in accepted_rtl["numerical_rtl"]),
        *(ROOT / item["path"] for item in oracle_contract["artifacts"]),
    ]
    unique_paths = sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())
    return [artifact(path) for path in unique_paths]


def build_run_manifest(
    *,
    accepted_rtl: dict[str, Any],
    command: list[str],
    generated_at_utc: str,
    runtime_packages: dict[str, str],
    sources: list[dict[str, Any]],
    oracle_contract: dict[str, Any],
) -> dict[str, Any]:
    prompt_manifest = json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8"))
    quality_config = json.loads(QUALITY_CONFIG.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc,
        "claim_boundary": "paired_full_model_quality_measurement_only",
        "command": command,
        "model": prompt_manifest["model"],
        "public_inputs": {
            "datasets": prompt_manifest["datasets"],
            "lm_eval": prompt_manifest["lm_eval"],
        },
        "numerical_contract": {
            "activation_quantization": quality_config["activation_quantization"],
            "arithmetic": quality_config["arithmetic"],
            "full_model_scope": quality_config["full_model_scope"],
            "weight_quantization": quality_config["weight_quantization"],
        },
        "determinism": quality_config["determinism"],
        "acceptance_thresholds": quality_config["acceptance_thresholds"],
        "evaluation": quality_config["evaluation"],
        "runtime": {
            "executable": ".venv/bin/python",
            "packages": runtime_packages,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "accepted_rtl": accepted_rtl,
        "independent_reference_oracles": oracle_contract,
        "source_artifacts": sources,
    }


def write_manifest(run_dir: Path) -> Path:
    files = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    manifest = run_dir / "SHA256SUMS"
    manifest.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(run_dir).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    return manifest


def make_read_only(run_dir: Path) -> None:
    for path in sorted(run_dir.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    run_dir.chmod(0o555)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default=datetime.now(UTC).strftime("official-%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    if "/" in args.run_id or args.run_id in {".", ".."}:
        raise SystemExit("--run-id must be one path component")
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise SystemExit("official measurement must run under the project .venv")
    readiness = run_gate()
    if readiness.get("status") != "ready_for_explicit_official_measurement":
        raise SystemExit(
            "official quality measurement is not allowed while preflight status is "
            f"{readiness.get('classification', readiness.get('status', 'unknown'))}"
        )

    accepted_rtl = validate_rtl_binding(ROOT)
    oracle_contract = validate_oracle_manifest(ROOT)
    runtime_packages = {
        name: importlib.metadata.version(distribution)
        for name, distribution in {
            "datasets": "datasets",
            "lm_eval": "lm-eval",
            "torch": "torch",
            "transformers": "transformers",
        }.items()
    }

    run_dir = QUALITY_RAW / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = run_dir / "artifacts"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command_path = run_dir / "command.json"
    command = [
        str(VENV_PYTHON),
        str(ROOT / "tools" / "ace2_full_model_fixed_point.py"),
        "--mode",
        "official",
        "--output-dir",
        str(artifacts),
    ]
    started_at = utc_now()
    recorded_command = [
        "./.venv/bin/python",
        "tools/ace2_full_model_fixed_point.py",
        "--mode",
        "official",
        "--output-dir",
        artifacts.relative_to(ROOT).as_posix(),
    ]
    run_manifest = build_run_manifest(
        accepted_rtl=accepted_rtl,
        command=recorded_command,
        generated_at_utc=started_at,
        runtime_packages=runtime_packages,
        sources=source_artifacts(accepted_rtl, oracle_contract),
        oracle_contract=oracle_contract,
    )
    (run_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    command_record = {
        "command": recorded_command,
        "cwd": ".",
        "exit_code": completed.returncode,
        "finished_at_utc": utc_now(),
        "started_at_utc": started_at,
    }
    command_path.write_text(
        json.dumps(command_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze = subprocess.run(
        [str(VENV_PIP), "freeze", "--all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (run_dir / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    manifest = write_manifest(run_dir)
    if completed.returncode != 0:
        make_read_only(run_dir)
        raise SystemExit(
            f"official quality measurement failed with exit {completed.returncode}; "
            f"see {stderr_path}"
        )

    result_path = artifacts / "results.json"
    if not result_path.is_file():
        raise SystemExit("official runner succeeded without artifacts/results.json")
    LATEST.mkdir(parents=True, exist_ok=True)
    pointer = {
        "generated_at_utc": utc_now(),
        "results_sha256": sha256_file(result_path),
        "run_path": run_dir.relative_to(ROOT).as_posix(),
        "schema_version": 1,
        "sha256s_sha256": sha256_file(manifest),
    }
    POINTER.write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = run_gate()
    make_read_only(run_dir)
    print(
        "ACE2_OFFICIAL_QUALITY "
        f"classification={result['classification']} "
        f"gate_passed={str(result['gate_passed']).lower()} "
        f"run={run_dir.relative_to(ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
