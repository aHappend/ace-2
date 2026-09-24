#!/usr/bin/env python3
"""Recorded bit-exact verifier for an existing ACE2RT2 chat runtime journal."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ace2_chat_demo import (
    canonical_bytes,
    read_ace2rt2_package_metadata,
    prefill_skip_intermediate_lm_head_commands,
    reconstruct_ace2rt2_schedule,
    sha256_file,
    verify_positions_through_argmax,
    write_atomic,
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def exact_process_argv() -> list[str]:
    proc_cmdline = Path("/proc/self/cmdline")
    if proc_cmdline.is_file():
        fields = proc_cmdline.read_bytes().split(b"\0")
        return [field.decode("utf-8", errors="surrogateescape") for field in fields if field]
    return [sys.executable, *sys.argv]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify an existing ACE2RT2 journal through a prompt or generated LM-head boundary"
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--through-position", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-skip-intermediate-lm-head", action="store_true")
    args = parser.parse_args()

    package_path = args.package.resolve()
    runtime_output = args.runtime_output.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.through_position < 0:
        raise RuntimeError("--through-position must be nonnegative")
    journal_path = runtime_output / "progress.journal"
    if not journal_path.is_file():
        raise RuntimeError("runtime progress journal is missing")

    invocation = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "cwd": str(Path.cwd().resolve()),
        "argv": exact_process_argv(),
    }
    provenance_path = output / "provenance.json"
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": invocation["generated_at_utc"],
        "classification": "independent_ace2rt2_command_by_command_reference_replay",
        "invocation": invocation,
        "requested_through_position": args.through_position,
        "status": "RUNNING",
    }
    write_atomic(provenance_path, canonical_bytes(provenance))

    package = read_ace2rt2_package_metadata(package_path)
    maximum_position = (
        int(package["prompt_token_count"]) + int(package["max_new_tokens"]) - 2
    )
    if args.through_position > maximum_position:
        raise RuntimeError("requested verifier position is outside the package schedule")
    reconstruction = reconstruct_ace2rt2_schedule(package_path, package)
    commands = reconstruction["commands"]
    if args.prefill_skip_intermediate_lm_head:
        commands = prefill_skip_intermediate_lm_head_commands(
            commands,
            prompt_token_count=int(package["prompt_token_count"]),
        )
    reconstructed_schedule_sha256 = reconstruction["schedule_sha256"]
    target_commands = [
        command
        for command in commands
        if int(command["token_step"]) == args.through_position
    ]
    if not target_commands:
        raise RuntimeError("requested verifier position has no selected commands")

    start = time.perf_counter()
    result = verify_positions_through_argmax(
        commands,
        prompt_tokens=[int(token) for token in package["prompt_tokens"]],
        through_position=args.through_position,
        runtime_output=runtime_output,
        max_new_tokens=int(package["max_new_tokens"]),
        termination_token_ids=[int(token) for token in package["termination_token_ids"]],
        required_final_operator=str(target_commands[-1]["operator"]),
    )
    wall_seconds = time.perf_counter() - start
    result_path = output / f"position{args.through_position}_reference.json"
    write_atomic(result_path, canonical_bytes(result))

    source_paths = [Path(__file__).resolve(), ROOT / "tools/ace2_chat_demo.py"]
    provenance.update(
        {
            "completed_at_utc": utc_now(),
            "package": package,
            "runtime": {
                "journal": file_record(journal_path),
                "progress": file_record(runtime_output / "progress.json")
                if (runtime_output / "progress.json").is_file()
                else None,
                "summary": file_record(runtime_output / "summary.json")
                if (runtime_output / "summary.json").is_file()
                else None,
            },
            "reconstructed_schedule_sha256": reconstructed_schedule_sha256,
            "execution_policy": (
                "prefill_skip_intermediate_lm_head"
                if args.prefill_skip_intermediate_lm_head
                else "full_package_schedule"
            ),
            "sources": {str(path.relative_to(ROOT)): file_record(path) for path in source_paths},
            "verification": {
                "wall_seconds": wall_seconds,
                "scope": "independent Python reference replay only; excludes RTL simulation",
                "result": file_record(result_path),
                "status": result["status"],
                "commands_compared": int(result.get("commands_compared", 0)),
                "bytes_compared": int(result.get("bytes_compared", 0)),
                "first_mismatch_ordinal": result.get("first_mismatch_ordinal"),
                "generated_token_ids": result.get("generated_token_ids", []),
                "generated_token_feedback": result.get("generated_token_feedback", []),
                "journal_token_history": result.get("journal_token_history"),
                "generated_position_kv": result.get("generated_position_kv"),
                "argmax": result.get("argmax"),
            },
            "status": result["status"],
        }
    )
    write_atomic(provenance_path, canonical_bytes(provenance))
    print(
        "ACE2_CHAT_VERIFY_RESULT "
        f"status={result['status']} through_position={args.through_position} "
        f"commands={result.get('commands_compared', 0)} "
        f"bytes={result.get('bytes_compared', 0)} "
        f"first_mismatch={result.get('first_mismatch_ordinal')} "
        f"wall_seconds={wall_seconds:.6f}"
    )
    if result["status"] == "PASS_BIT_EXACT":
        return 0
    if result["status"] == "FAIL":
        return 2
    return 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_CHAT_VERIFY_SETUP_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
