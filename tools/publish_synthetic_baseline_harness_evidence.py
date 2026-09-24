#!/usr/bin/env python3
"""Run synthetic harness tests and atomically publish hash-bound evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tools/synthetic_baseline_harness.py"
TESTS = ROOT / "tools/test_synthetic_baseline_harness.py"
PUBLISHER = Path(__file__).resolve()
SEAL_TOOL = ROOT / "tools/seal_cross_layer_baseline_recovery.py"
POLICY = ROOT / "design/FAST_LOOP_POLICY.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
CONTRACT = "cross_layer_quantization_error_carry_final_output_v1"
SEAL_DIR = ROOT / "evidence" / CONTRACT / "recovery/manager_architecture_rollback_v1"
SEAL = SEAL_DIR / "SEAL.json"
SEAL_SUM = SEAL_DIR / "SEAL.sha256"
EVIDENCE_PARENT = ROOT / "evidence/baseline_harness_hardening"
EVIDENCE_DIR = EVIDENCE_PARENT / "synthetic_harness_v1"
PROPERTY_TESTS = {
    "atomic_artifact_and_companion_hash_commit": (
        "test_01_atomic_artifact_and_companion_hash_commit"
    ),
    "exact_provenance_binding": "test_02_exact_provenance_binding",
    "durable_exactly_one_reservation_and_run_ledger": (
        "test_03_durable_exactly_one_reservation_and_run_ledger"
    ),
    "stage_authorization_before_execution": "test_04_stage_authorization_before_execution",
    "candidate_entrypoint_interlock": "test_05_candidate_entrypoint_interlock",
    "crash_recovery": "test_06_crash_recovery",
    "missing_or_mismatched_artifact_fail_closed": (
        "test_07_missing_or_mismatched_artifact_fail_closed"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_durable(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def write_json_artifact(directory: Path, name: str, value: dict[str, Any]) -> None:
    content = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    write_durable(directory / name, content)
    write_durable(
        directory / f"{name.removesuffix('.json')}.sha256",
        f"{sha256_bytes(content)}  {name}\n".encode("ascii"),
    )


def write_log_artifact(directory: Path, name: str, content: str) -> None:
    encoded = content.encode("utf-8")
    write_durable(directory / name, encoded)
    write_durable(
        directory / f"{name}.sha256",
        f"{sha256_bytes(encoded)}  {name}\n".encode("ascii"),
    )


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def source_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def sealed_paths() -> list[Path]:
    seal = load_json(SEAL)
    paths: list[Path] = []
    for section in (
        "immutable_terminal_evidence",
        "immutable_accidental_post_terminal_0530_evidence",
    ):
        for item in seal.get(section, []):
            paths.append(ROOT / item["path"])
    paths.append(ROOT / seal["observed_failed_candidate_source"]["path"])
    return paths


def protected_files() -> list[Path]:
    paths: set[Path] = set(sealed_paths())
    for directory_name in (
        "rtl",
        "reference",
        "verification",
        "vector",
        "vectors",
        "model",
        "models",
    ):
        directory = ROOT / directory_name
        if directory.is_dir():
            paths.update(path for path in directory.rglob("*") if path.is_file())
    paths.add(ROOT / f"evidence/{CONTRACT}/latest/PRECHECK.json")
    return sorted(paths, key=lambda path: relative(path))


def protected_manifest() -> dict[str, Any]:
    artifacts = []
    for path in protected_files():
        require(path.is_file(), f"protected artifact missing: {path}")
        artifacts.append(
            {
                "path": relative(path),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256_file(path),
            }
        )
    digest = sha256_bytes(canonical_json_bytes(artifacts))
    return {
        "schema_version": 1,
        "scope": [
            "rtl/**",
            "reference/**",
            "verification/**",
            "vector(s)/**",
            "model(s)/**",
            f"all artifacts bound by {relative(SEAL)}",
            f"evidence/{CONTRACT}/latest/PRECHECK.json",
        ],
        "artifact_count": len(artifacts),
        "manifest_sha256": digest,
        "artifacts": artifacts,
    }


def run_command(command: list[str]) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    rendered = "$ " + " ".join(command) + "\n" + completed.stdout + completed.stderr
    return completed.returncode, rendered


def artifact_manifest(directory: Path, excluded: Iterable[str] = ()) -> list[dict[str, Any]]:
    exclusions = set(excluded)
    return [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name not in exclusions
    ]


def main() -> int:
    require(not EVIDENCE_DIR.exists(), f"evidence already published: {EVIDENCE_DIR}")
    pipeline = load_json(PIPELINE)
    policy = load_json(POLICY)
    gate = policy.get("baseline_harness_hardening_gate", {})
    require(pipeline.get("current_stage") == "architecture", "pipeline is not at architecture")
    require(gate.get("candidate_independent") is True, "policy is not candidate-independent")
    require(gate.get("dry_fixture_only") is True, "policy is not dry-fixture-only")
    require(
        gate.get("candidate_or_model_execution_permitted") is False,
        "policy permits candidate/model execution",
    )
    require(gate.get("required_properties") == list(PROPERTY_TESTS), "policy properties changed")

    before = protected_manifest()
    seal_command = [sys.executable, str(SEAL_TOOL), "--check"]
    seal_before_status, seal_before_log = run_command(seal_command)
    require(seal_before_status == 0, "pre-test recovery seal check failed")

    test_command = [sys.executable, str(TESTS)]
    test_status, test_log = run_command(test_command)
    require(test_status == 0, "synthetic harness tests failed")
    require("Ran 7 tests" in test_log and "\nOK\n" in test_log, "seven-test proof missing")
    for test_name in PROPERTY_TESTS.values():
        require(test_name in test_log, f"property test missing from log: {test_name}")

    seal_after_status, seal_after_log = run_command(seal_command)
    require(seal_after_status == 0, "post-test recovery seal check failed")
    after = protected_manifest()
    require(before == after, "protected candidate/model/RTL/reference/vector/sealed assets changed")

    sources = [
        source_artifact(path)
        for path in (HARNESS, TESTS, PUBLISHER, POLICY, PIPELINE, SEAL, SEAL_SUM)
    ]
    report = {
        "schema_version": 1,
        "project": "ACE-2",
        "evidence_id": "candidate_independent_synthetic_baseline_harness_v1",
        "decision": "PASS",
        "published_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "candidate-independent synthetic/dry-fixture harness only",
        "stage": pipeline["current_stage"],
        "contract_disposition": "failed mechanism remains sealed; successor freeze unauthorized",
        "properties": {
            name: {"status": "PASS", "test": test_name}
            for name, test_name in PROPERTY_TESTS.items()
        },
        "commands": [
            {
                "argv": test_command,
                "class": "synthetic_only_tests",
                "exit_status": test_status,
                "log": "HARNESS_TEST.log",
            },
            {
                "argv": seal_command,
                "class": "read_only_recovery_seal_check_before_and_after",
                "exit_status_before": seal_before_status,
                "exit_status_after": seal_after_status,
                "logs": ["RECOVERY_SEAL_CHECK_BEFORE.log", "RECOVERY_SEAL_CHECK_AFTER.log"],
            },
        ],
        "execution_boundary": {
            "arbitrary_subprocess_surface_in_harness": False,
            "candidate_entrypoint_available": False,
            "candidate_or_model_execution_count": 0,
            "rtl_reference_vector_precheck_execution_count": 0,
            "successor_frozen": False,
        },
        "protected_state": {
            "before_manifest_sha256": before["manifest_sha256"],
            "after_manifest_sha256": after["manifest_sha256"],
            "artifact_count": before["artifact_count"],
            "unchanged": True,
        },
        "manager_recovery_seal": source_artifact(SEAL),
        "source_and_authority_artifacts": sources,
    }

    EVIDENCE_PARENT.mkdir(parents=True, exist_ok=True)
    fsync_directory(EVIDENCE_PARENT.parent)
    staging = EVIDENCE_PARENT / f".synthetic_harness_v1.staging.{os.getpid()}"
    require(not staging.exists(), f"staging path exists: {staging}")
    staging.mkdir()
    try:
        write_json_artifact(staging, "HARNESS_TEST_REPORT.json", report)
        write_json_artifact(staging, "PROTECTED_BEFORE.json", before)
        write_json_artifact(staging, "PROTECTED_AFTER.json", after)
        write_log_artifact(staging, "HARNESS_TEST.log", test_log)
        write_log_artifact(staging, "RECOVERY_SEAL_CHECK_BEFORE.log", seal_before_log)
        write_log_artifact(staging, "RECOVERY_SEAL_CHECK_AFTER.log", seal_after_log)
        manifest = {
            "schema_version": 1,
            "evidence_id": report["evidence_id"],
            "artifacts": artifact_manifest(staging),
        }
        write_json_artifact(staging, "MANIFEST.json", manifest)
        fsync_directory(staging)
        os.replace(staging, EVIDENCE_DIR)
        fsync_directory(EVIDENCE_PARENT)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    report_sha256 = sha256_file(EVIDENCE_DIR / "HARNESS_TEST_REPORT.json")
    manifest_sha256 = sha256_file(EVIDENCE_DIR / "MANIFEST.json")
    print(
        "ACE2_SYNTHETIC_BASELINE_HARNESS_EVIDENCE_PASS "
        f"report_sha256={report_sha256} manifest_sha256={manifest_sha256} "
        f"properties={len(PROPERTY_TESTS)} protected_artifacts={before['artifact_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
