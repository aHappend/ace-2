#!/usr/bin/env python3
"""Construct and verify the marker-free V8 hierarchical row/group-scale policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "option_b_alias_safe_hierarchical_row_group_scale_w4_grouped_dynamic_scale32_w4a8_v8"
MISSION_ID = "d16e65301b57"
TASK_ID = "task-c4731ac4e0b8"
TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_HIERARCHICAL_ROW_GROUP_SCALE_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V8_ENGINEER_TASK.json"
TASK_COMPANION = TASK_PATH.with_suffix(".sha256")
PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_HIERARCHICAL_ROW_GROUP_SCALE_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V8_PLAN.json"
PLAN_COMPANION = PLAN_PATH.with_suffix(".sha256")
OFFICIAL_ROOT = ROOT / "build/stage1-option-b-alias-safe-hierarchical-row-group-scale-w4-grouped-dynamic-scale32-w4a8-v8"
OFFICIAL_MARKER = OFFICIAL_ROOT / "attempt-0001/execution_started.json"
RUNNER_PATH = Path(__file__).resolve()

WEIGHT_GROUP_LANES = 128
LEGAL_INPUT_FEATURE_COUNTS = (896, 4864)
TABLE_ENTRIES = 16
SIGNED_INT4_MIN = -8
SIGNED_INT4_MAX = 7
GROUP_DELTA_MIN = -8
GROUP_DELTA_MAX = 7
OUTER_ITERATIONS = 8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_companion(path: Path, companion: Path) -> None:
    expected = f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}"
    require(companion.is_file(), f"checksum companion is missing: {companion}")
    require(companion.read_text(encoding="utf-8").strip() == expected, f"checksum companion differs: {path}")


def require_project_python() -> dict[str, str]:
    expected = ROOT / ".venv/bin/python"
    require(expected.is_file(), "project Python is missing")
    require(os.path.samefile(Path(sys.executable), expected), "wrong Python executable")
    return {"entrypoint": "./.venv/bin/python", "python": platform.python_version()}


def expect_rejected(call: Callable[[], Any], message: str) -> bool:
    try:
        call()
    except (RuntimeError, ValueError, OverflowError):
        return True
    raise RuntimeError(message)


def weight_group_ranges(width: int) -> tuple[tuple[int, int], ...]:
    require(width in LEGAL_INPUT_FEATURE_COUNTS, f"illegal V8 input feature count: {width}")
    require(width % WEIGHT_GROUP_LANES == 0, "input feature count is not group aligned")
    return tuple((start, start + WEIGHT_GROUP_LANES) for start in range(0, width, WEIGHT_GROUP_LANES))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "prepare", "verify"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import v8_hierarchical_row_group_full_model_backend as backend

    if args.command == "self-test":
        result = backend.combined_self_test()
        if args.output is not None:
            output = args.output if args.output.is_absolute() else ROOT / args.output
            output.parent.mkdir(parents=True, exist_ok=True)
            raw = json.dumps(result, sort_keys=True, indent=2).encode() + b"\n"
            descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            try:
                os.write(descriptor, raw)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        print("ACE2_OPTION_B_HIERARCHICAL_ROW_GROUP_V8_SELF_TEST_PASS " + json.dumps(result, sort_keys=True))
        return 0
    if args.command == "prepare":
        try:
            result = backend.prepare()
        except Exception as exc:
            backend.record_preparation_failure(exc)
            raise
        print("ACE2_OPTION_B_HIERARCHICAL_ROW_GROUP_V8_PREFLIGHT_COMPLETE " + json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PREATTEMPT_READY" else 2
    result = backend.load_verified_closure()
    print("ACE2_OPTION_B_HIERARCHICAL_ROW_GROUP_V8_PREFLIGHT_VERIFIED " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
