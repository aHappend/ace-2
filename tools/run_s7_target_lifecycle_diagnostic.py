#!/usr/bin/python3
"""One-shot sanitized observer for the exact S7 target-list command."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from s7_bounded_process_lifecycle import run_bounded_process, sha256_bytes, sha256_file


ROOT = Path("/home/argustest/ace-2")
OUTPUT = ROOT / "diagnosis/s7-target-lifecycle-diagnostic-20260811T122714Z-3f0b10b629c4/evidence.sanitized.json"
DIAGNOSTIC_ID = "s7-target-lifecycle-diagnostic-20260811T122714Z-3f0b10b629c4"
TARGET_ARGV = (
    "amlt",
    "--json-tables",
    "--quiet",
    "target",
    "list",
    "singularity",
    "--no-update",
    "--target-name",
    "msrresrchbasicvc",
    "--verbose",
)
EXPECTED_TARGET_STDOUT_BYTES = 201
EXPECTED_TARGET_STDOUT_SHA256 = "cdc580d647f9ecae10bc60ca1e1d1a701a1d437f7db973c5f3a875080b98c426"
EXPECTED_AMLT_INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
TARGET_FIELDS = {
    "cuda",
    "location",
    "name",
    "resource_group",
    "service",
    "sku",
    "subscription",
    "vc",
    "workspace_name",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def evaluate_target_output(
    stdout: bytes,
    stderr: bytes,
    return_code: int,
    *,
    expected_stdout_bytes: int = EXPECTED_TARGET_STDOUT_BYTES,
    expected_stdout_sha256: str = EXPECTED_TARGET_STDOUT_SHA256,
) -> dict[str, Any]:
    payload = json.loads(
        stdout.decode("utf-8", errors="strict"), object_pairs_hook=reject_duplicate_keys
    )
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError("target payload must be exactly one row")
    row = payload[0]
    if set(row) != TARGET_FIELDS or not all(isinstance(row[key], str) for key in TARGET_FIELDS):
        raise ValueError("target row field contract mismatch")
    equalities = {
        "location_empty": row["location"] == "",
        "name_equal": row["name"] == "msrresrchbasicvc",
        "resource_group_equal": row["resource_group"] == "gcr-singularity",
        "service_equal": row["service"] == "sing",
        "subscription_equal": row["subscription"] == "Singularity Shared",
        "vc_empty": row["vc"] == "",
        "workspace_name_empty": row["workspace_name"] == "",
    }
    exact_stream = {
        "return_code_zero": return_code == 0,
        "stderr_empty": stderr == b"",
        "stdout_byte_count_equal": len(stdout) == expected_stdout_bytes,
        "stdout_sha256_equal": sha256_bytes(stdout) == expected_stdout_sha256,
    }
    return {
        "exact_stream_equalities": exact_stream,
        "passed": all(equalities.values()) and all(exact_stream.values()),
        "selected_equalities": equalities,
    }


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o400)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> int:
    if OUTPUT.exists() or OUTPUT.parent.exists():
        raise RuntimeError("diagnostic identity is already consumed")
    library_path = ROOT / "tools/s7_bounded_process_lifecycle.py"
    environment = os.environ.copy()
    environment.update({"AMLT_TELEMETRY_LOGGING": "false", "AZURE_CORE_OUTPUT": "none"})
    observation = run_bounded_process(
        TARGET_ARGV,
        cwd="/home/argustest/singularity",
        environment=environment,
        evaluator=evaluate_target_output,
        expected_executable_sha256s={EXPECTED_AMLT_INTERPRETER_SHA256},
        identity_salt=DIAGNOSTIC_ID,
        timeout_seconds=30.0,
        natural_quiescence_seconds=2.0,
        term_grace_seconds=2.0,
        kill_grace_seconds=3.0,
    )
    evidence = {
        "artifact_kind": "s7_target_binding_descendant_lifecycle_diagnostic",
        "authority_created": False,
        "diagnostic_id": DIAGNOSTIC_ID,
        "diagnostic_process_invocations": 1,
        "exact_target_argv": list(TARGET_ARGV),
        "library_sha256": sha256_file(library_path),
        "observation": observation,
        "read_only": True,
        "schema_version": 3,
        "submission_process_invocations": 0,
    }
    write_once(OUTPUT, canonical_json_bytes(evidence))
    print(
        json.dumps(
            {
                "cleanup_uncertain": observation["process_lifecycle"]["cleanup_uncertain"],
                "evidence_path": str(OUTPUT.relative_to(ROOT)),
                "evidence_sha256": sha256_file(OUTPUT),
                "process_exception_type": observation["process_exception_type"],
                "stdout_byte_count": observation["sanitized_streams"]["stdout_byte_count"],
                "stdout_sha256": observation["sanitized_streams"]["stdout_sha256"],
            },
            sort_keys=True,
        )
    )
    return 2 if observation["process_lifecycle"]["cleanup_uncertain"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
