#!/usr/bin/env python3
"""Run and independently verify a complete layer prefix at real user positions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_full_qwen_command_schedule_runtime as accepted_runtime
from tools.ace2_chat_demo import (
    canonical_bytes,
    certified_rtl_binding,
    focused_complete_layer_prefix_commands,
    read_ace2rt2_package_metadata,
    reconstruct_ace2rt2_schedule,
    sha256_file,
    utc_now,
    verify_complete_layer_prefix_prompt_prefix,
    write_atomic,
)


USER_PREFIX = [151644, 872, 198]
IM_END_TOKEN = 151645


def user_content_slice(
    prompt_tokens: list[int],
    user_token_count: int,
) -> tuple[int, list[int]]:
    starts = [
        index + len(USER_PREFIX)
        for index in range(len(prompt_tokens) - len(USER_PREFIX) + 1)
        if prompt_tokens[index : index + len(USER_PREFIX)] == USER_PREFIX
    ]
    if len(starts) != 1:
        raise RuntimeError("authenticated package lacks one canonical user-content prefix")
    start = starts[0]
    stop = start + user_token_count
    if stop >= len(prompt_tokens) or prompt_tokens[stop] != IM_END_TOKEN:
        raise RuntimeError("authenticated package user-content boundary differs")
    return start, prompt_tokens[start:stop]


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        public_path = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        public_path = resolved.name
    return {
        "path": public_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def runtime_rtl_binding_record(
    source_binding: dict[str, Any],
    live_binding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_package_rtl_tree_sha256": source_binding["rtl_tree_sha256"],
        "source_package_constraint_tree_sha256": source_binding[
            "constraint_tree_sha256"
        ],
        "rtl_tree_matches_source_package": (
            live_binding["rtl_tree_sha256"] == source_binding["rtl_tree_sha256"]
        ),
        "constraint_tree_matches_source_package": (
            live_binding["constraint_tree_sha256"]
            == source_binding["constraint_tree_sha256"]
        ),
        "live": live_binding,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-position-count", type=int, default=2)
    parser.add_argument("--layer-count", type=int, default=1)
    parser.add_argument("--timeout-cycles", type=int, default=100_000_000)
    args = parser.parse_args()
    if args.user_position_count < 1:
        raise RuntimeError("layer-prefix scale-out requires at least one real user position")
    if not 1 <= args.layer_count <= 24:
        raise RuntimeError("--layer-count must be between 1 and 24")

    package_path = args.source_package.resolve()
    source_provenance_path = args.source_provenance.resolve()
    output = args.output.resolve()
    runtime_output = output / "rtl"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("layer-prefix scale-out output is not empty")
    output.mkdir(parents=True, exist_ok=True)

    source_provenance = json.loads(source_provenance_path.read_text(encoding="utf-8"))
    package = read_ace2rt2_package_metadata(package_path)
    if package["sha256"] != source_provenance["runtime_package"]["sha256"]:
        raise RuntimeError("source package and provenance hashes differ")
    prompt_tokens = [int(token) for token in package["prompt_tokens"]]
    user_start, user_tokens = user_content_slice(
        prompt_tokens,
        int(source_provenance["prompt"]["user_token_count"]),
    )
    selected_tokens = user_tokens[: args.user_position_count]
    if len(selected_tokens) < args.user_position_count:
        raise RuntimeError("source prompt has fewer user positions than requested")
    selected_positions = [user_start + index for index in range(len(selected_tokens))]

    reconstruction = reconstruct_ace2rt2_schedule(package_path, package)
    commands = reconstruction["commands"]
    binary = accepted_runtime.DEFAULT_BINARY.resolve()
    if not binary.is_file():
        raise RuntimeError("missing Verilator runtime binary; run `make full-qwen-runtime-build`")
    source_rtl_binding = source_provenance["runtime_package"]["rtl_binding"]
    runtime_rtl_binding = runtime_rtl_binding_record(
        source_rtl_binding,
        certified_rtl_binding(),
    )

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "bounded_real_user_position_complete_layer_prefix_rtl_scaleout",
        "source": {
            "package": file_record(package_path),
            "provenance": file_record(source_provenance_path),
            "schedule_sha256": reconstruction["schedule_sha256"],
            "prompt_sha256": source_provenance["prompt"]["prompt_sha256"],
            "prompt_text_persisted": False,
            "rtl_binding": source_rtl_binding,
        },
        "runtime": {
            "binary": file_record(binary),
            "image": file_record(accepted_runtime.IMAGE.resolve()),
            "model": file_record(accepted_runtime.MODEL.resolve()),
            "host_source": file_record(Path(__file__).resolve()),
            "reference_source": file_record(
                ROOT / "tools/ace2_chat_demo.py"
            ),
            "verilator_runtime_source": file_record(
                ROOT / "verification/verilator/ace2_shell_runtime_main.cpp"
            ),
            "rtl_binding": runtime_rtl_binding,
            "execution_mode": (
                "focused_complete_layer0_prompt_prefix"
                if args.layer_count == 1
                else "focused_complete_layer_prefix_prompt_prefix"
            ),
            "layer_count": args.layer_count,
            "timeout_cycles_per_command": args.timeout_cycles,
        },
        "scope": {
            "user_position_count": len(selected_positions),
            "user_positions": selected_positions,
            "user_token_ids": selected_tokens,
            "layer_count": args.layer_count,
            "layers": list(range(args.layer_count)),
            "prior_positions_executed_from_zero": True,
            "prior_position_kv_preserved_by_journal_replay": True,
            "upper_layers_executed": args.layer_count > 1,
            "lm_head_executed": False,
            "ppa_executed": False,
        },
        "batches": [],
        "status": "RUNNING",
    }
    provenance_path = output / "provenance.json"
    write_atomic(provenance_path, canonical_bytes(provenance))

    for batch_index, position in enumerate(selected_positions):
        focused = focused_complete_layer_prefix_commands(
            commands,
            through_position=position,
            layer_count=args.layer_count,
        )
        command = [
            str(binary),
            "--package", str(package_path),
            "--image", str(accepted_runtime.IMAGE.resolve()),
            "--model", str(accepted_runtime.MODEL.resolve()),
            "--output", str(runtime_output),
            "--timeout-cycles", str(args.timeout_cycles),
        ]
        if args.layer_count == 1:
            command.extend(["--focus-layer0-through-token-step", str(position)])
        else:
            command.extend(
                [
                    "--focus-layer-prefix-through-token-step", str(position),
                    "--focus-layer-prefix-count", str(args.layer_count),
                ]
            )
        if batch_index:
            command.append("--resume")
        started = time.perf_counter()
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        wall_seconds = time.perf_counter() - started
        log_path = output / f"position{position:04d}_runtime.log"
        write_atomic(
            log_path,
            (completed.stdout + completed.stderr).encode("utf-8"),
        )
        runtime_summary_path = runtime_output / "summary.json"
        runtime_summary = (
            json.loads(runtime_summary_path.read_text(encoding="utf-8"))
            if runtime_summary_path.is_file()
            else {}
        )
        summary_snapshot_path = output / f"position{position:04d}_runtime_summary.json"
        write_atomic(summary_snapshot_path, canonical_bytes(runtime_summary))
        reference = (
            verify_complete_layer_prefix_prompt_prefix(
                commands,
                prompt_tokens=prompt_tokens,
                through_position=position,
                layer_count=args.layer_count,
                runtime_output=runtime_output,
            )
            if (runtime_output / "progress.journal").is_file()
            else {"status": "NOT_REACHED"}
        )
        reference_path = output / f"position{position:04d}_reference.json"
        write_atomic(reference_path, canonical_bytes(reference))
        batch = {
            "batch_index": batch_index,
            "position": position,
            "token_id": selected_tokens[batch_index],
            "resume": bool(batch_index),
            "commands_expected": len(focused),
            "source_ordinal_last": focused[-1]["source_ordinal"],
            "runtime_exit_code": completed.returncode,
            "runtime_wall_seconds": wall_seconds,
            "runtime_log": file_record(log_path),
            "runtime_summary": file_record(summary_snapshot_path),
            "reference": file_record(reference_path),
            "reference_status": reference.get("status"),
            "first_mismatch_ordinal": reference.get("first_mismatch_ordinal"),
            "target_position_kv": reference.get("target_position_kv"),
        }
        provenance["batches"].append(batch)
        if completed.returncode != 0 or reference.get("status") != "PASS_BIT_EXACT":
            provenance["status"] = "STOPPED_AT_FIRST_MISMATCH"
            write_atomic(provenance_path, canonical_bytes(provenance))
            print(
                "ACE2_CHAT_LAYER_PREFIX_SCALEOUT "
                f"status={provenance['status']} position={position} "
                f"runtime_exit={completed.returncode} "
                f"reference={reference.get('status')} "
                f"first_mismatch={reference.get('first_mismatch_ordinal')}"
            )
            return 2
        write_atomic(provenance_path, canonical_bytes(provenance))

    provenance["completed_at_utc"] = utc_now()
    provenance["status"] = "PASS_BIT_EXACT"
    write_atomic(provenance_path, canonical_bytes(provenance))
    print(
        "ACE2_CHAT_LAYER_PREFIX_SCALEOUT "
        f"status=PASS_BIT_EXACT positions={selected_positions} layers={args.layer_count} "
        f"commands={provenance['batches'][-1]['commands_expected']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_CHAT_LAYER_PREFIX_SCALEOUT_SETUP_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
