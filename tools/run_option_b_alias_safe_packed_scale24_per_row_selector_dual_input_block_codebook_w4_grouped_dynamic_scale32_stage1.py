#!/usr/bin/env python3
"""Construct, verify, and conditionally execute the frozen V10 candidate."""

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
CANDIDATE_ID = "option_b_alias_safe_packed_scale24_per_row_selector_dual_input_block_codebook_w4_grouped_dynamic_scale32_w4a8_v10"
MISSION_ID = "b1ff12309055"
TASK_ID = "task-b87978f9a5a8"
TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_PACKED_SCALE24_PER_ROW_SELECTOR_DUAL_INPUT_BLOCK_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V10_ENGINEER_TASK.json"
TASK_COMPANION = TASK_PATH.with_suffix(".sha256")
PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_PACKED_SCALE24_PER_ROW_SELECTOR_DUAL_INPUT_BLOCK_CODEBOOK_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V10_PLAN.json"
PLAN_COMPANION = PLAN_PATH.with_suffix(".sha256")
OFFICIAL_ROOT = ROOT / "build/stage1-option-b-alias-safe-packed-scale24-per-row-selector-dual-input-block-codebook-w4-grouped-dynamic-scale32-w4a8-v10"
OFFICIAL_MARKER = OFFICIAL_ROOT / "attempt-0001/execution_started.json"
RUNNER_PATH = Path(__file__).resolve()

INPUT_BLOCK_LANES = 128
LEGAL_INPUT_FEATURE_COUNTS = (896, 4864)
CODEBOOK_ENTRIES = 16
OUTER_ITERATIONS = 8
ALIGNMENT_BYTES = 16
ROW_GROUP_SIZE_BY_FAMILY = {
    "q_proj": 1,
    "k_proj": 1,
    "v_proj": 1,
    "o_proj": 1,
    "gate_proj": 1,
    "up_proj": 1,
    "down_proj": 1,
    "lm_head": 1,
}


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


def input_block_ranges(width: int) -> tuple[tuple[int, int], ...]:
    require(width in LEGAL_INPUT_FEATURE_COUNTS, f"illegal V10 input feature count: {width}")
    require(width % INPUT_BLOCK_LANES == 0, "input feature count is not block aligned")
    return tuple((start, start + INPUT_BLOCK_LANES) for start in range(0, width, INPUT_BLOCK_LANES))


def row_group_size(module_name: str) -> int:
    family = "lm_head" if module_name == "lm_head" or module_name.endswith(".lm_head") else module_name.rsplit(".", 1)[-1]
    require(family in ROW_GROUP_SIZE_BY_FAMILY, f"unsupported V10 Linear family: {module_name}")
    return ROW_GROUP_SIZE_BY_FAMILY[family]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "prepare", "verify", "execute", "verify-result"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import v10_packed_scale24_per_row_selector_backend as backend

    if args.command == "self-test":
        result = backend.combined_self_test()
        if args.output is not None:
            output = args.output.resolve()
            require(output.is_relative_to(ROOT), "V10 self-test output must remain inside the repository")
            require(not output.is_relative_to(OFFICIAL_ROOT), "V10 self-test output may not enter the official namespace")
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".tmp")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(output)
        print("ACE2_OPTION_B_PACKED_SCALE24_PER_ROW_SELECTOR_V10_SELF_TEST_PASS " + json.dumps(result, sort_keys=True))
        return 0
    if args.command == "prepare":
        try:
            result = backend.prepare()
        except Exception as exc:
            backend.record_preparation_failure(exc)
            raise
        print("ACE2_OPTION_B_PACKED_SCALE24_PER_ROW_SELECTOR_V10_PREFLIGHT_COMPLETE " + json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PREATTEMPT_READY" else 2
    if args.command == "verify":
        result = backend.load_verified_closure(require_unconsumed=True)
        print("ACE2_OPTION_B_PACKED_SCALE24_PER_ROW_SELECTOR_V10_PREFLIGHT_VERIFIED " + json.dumps(result, sort_keys=True))
        return 0
    if args.command == "execute":
        return backend.execute_once()
    result = backend.verify_result()
    print("ACE2_OPTION_B_PACKED_SCALE24_PER_ROW_SELECTOR_V10_RESULT_VERIFIED " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
