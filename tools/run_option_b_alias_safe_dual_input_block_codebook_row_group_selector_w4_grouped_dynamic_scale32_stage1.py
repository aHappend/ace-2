#!/usr/bin/env python3
"""Construct and verify the marker-free V9 dual-codebook policy.

Preparation constructs two independent full models, runs the unchanged eight
non-evaluator cases and 312 Dynamic Scale32 events, and seals PREATTEMPT_READY
or PREATTEMPT_NO_GO without creating an official execution marker or reading
the frozen nine-prompt matrix content.
"""

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
CANDIDATE_ID = "option_b_alias_safe_dual_input_block_codebook_row_group_selector_w4_grouped_dynamic_scale32_w4a8_v9"
MISSION_ID = "3b229e217703"
TASK_ID = "task-9d075ab327ee"
TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_DUAL_INPUT_BLOCK_CODEBOOK_ROW_GROUP_SELECTOR_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V9_ENGINEER_TASK.json"
TASK_COMPANION = TASK_PATH.with_suffix(".sha256")
PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_DUAL_INPUT_BLOCK_CODEBOOK_ROW_GROUP_SELECTOR_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V9_PLAN.json"
PLAN_COMPANION = PLAN_PATH.with_suffix(".sha256")
OFFICIAL_ROOT = ROOT / "build/stage1-option-b-alias-safe-dual-input-block-codebook-row-group-selector-w4-grouped-dynamic-scale32-w4a8-v9"
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
    "down_proj": 2,
    "lm_head": 32,
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
    require(width in LEGAL_INPUT_FEATURE_COUNTS, f"illegal V9 input feature count: {width}")
    require(width % INPUT_BLOCK_LANES == 0, "input feature count is not block aligned")
    return tuple((start, start + INPUT_BLOCK_LANES) for start in range(0, width, INPUT_BLOCK_LANES))


def row_group_size(module_name: str) -> int:
    if module_name == "lm_head" or module_name.endswith(".lm_head"):
        return ROW_GROUP_SIZE_BY_FAMILY["lm_head"]
    family = module_name.rsplit(".", 1)[-1]
    require(family in ROW_GROUP_SIZE_BY_FAMILY and family != "lm_head", f"unsupported V9 Linear family: {module_name}")
    return ROW_GROUP_SIZE_BY_FAMILY[family]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "prepare", "verify"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import v9_dual_input_block_codebook_row_group_selector_backend as backend

    if args.command == "self-test":
        result = backend.combined_self_test()
        if args.output is not None:
            output = args.output.resolve()
            require(output.is_relative_to(ROOT), "V9 self-test output must remain inside the repository")
            require(not output.is_relative_to(OFFICIAL_ROOT), "V9 self-test output may not enter the official namespace")
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".tmp")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(output)
        print("ACE2_OPTION_B_DUAL_INPUT_BLOCK_CODEBOOK_ROW_GROUP_SELECTOR_V9_SELF_TEST_PASS " + json.dumps(result, sort_keys=True))
        return 0
    if args.command == "prepare":
        try:
            result = backend.prepare()
        except Exception as exc:
            backend.record_preparation_failure(exc)
            raise
        print("ACE2_OPTION_B_DUAL_INPUT_BLOCK_CODEBOOK_ROW_GROUP_SELECTOR_V9_PREFLIGHT_COMPLETE " + json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PREATTEMPT_READY" else 2
    result = backend.load_verified_closure()
    print("ACE2_OPTION_B_DUAL_INPUT_BLOCK_CODEBOOK_ROW_GROUP_SELECTOR_V9_PREFLIGHT_VERIFIED " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
