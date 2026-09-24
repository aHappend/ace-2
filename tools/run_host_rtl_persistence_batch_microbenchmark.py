#!/usr/bin/env python3
"""Measure ACE-2 runtime persistence batching on a fresh bounded RTL prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_full_qwen_command_schedule_runtime as accepted_runtime
from tools.ace2_chat_demo import (
    PINNED_EMBEDDING_OFFSET,
    build_ace2rt2_package,
    build_full_prompt_commands,
    parse_journal_v2,
    verify_first_attention_triplet,
    verify_first_kv_publication,
)

DEFAULT_OUTPUT = (
    ROOT / "build/host-rtl-persistence-batch-microbenchmark-v1/attempt-0002-sanitized"
)
PROMPT_TOKEN_ID = 42
COMMAND_COUNT = 10
BASELINE_BATCH_COMMANDS = 1
CANDIDATE_BATCH_COMMANDS = 10
MEASURED_TRIALS = 3


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_atomic(path: Path, raw: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def display_argv(argv: list[str]) -> list[str]:
    displayed: list[str] = []
    model = accepted_runtime.MODEL.resolve()
    for value in argv:
        path = Path(value)
        if not path.is_absolute():
            displayed.append(value)
            continue
        resolved = path.resolve()
        if resolved == model:
            displayed.append(
                f"hf-cache://Qwen/Qwen2.5-0.5B@{accepted_runtime.REVISION}/model.safetensors"
            )
            continue
        try:
            displayed.append(resolved.relative_to(ROOT).as_posix())
        except ValueError:
            displayed.append(f"external://{resolved.name}")
    return displayed


def run_trial(
    *,
    label: str,
    trial: int,
    batch_commands: int,
    root: Path,
    package: Path,
    commands: list[dict[str, Any]],
    measured: bool,
) -> dict[str, Any]:
    output = root / ("measured" if measured else "warmup") / f"{label}-{trial:02d}"
    output.mkdir(parents=True)
    argv = [
        str(accepted_runtime.DEFAULT_BINARY.resolve()),
        "--package",
        str(package),
        "--image",
        str(accepted_runtime.IMAGE.resolve()),
        "--model",
        str(accepted_runtime.MODEL.resolve()),
        "--output",
        str(output),
        "--stop-after",
        str(COMMAND_COUNT),
        "--timeout-cycles",
        "100000000",
        "--read-response-latency-cycles",
        "1",
        "--persistence-batch-commands",
        str(batch_commands),
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    wall_seconds = time.perf_counter() - start
    (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} trial {trial} failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    records = parse_journal_v2(output / "progress.journal")
    if summary["status"] != "PREFIX_COMPLETE" or len(records) != COMMAND_COUNT:
        raise RuntimeError(f"{label} trial {trial} did not complete the frozen prefix")
    kv_reference = verify_first_kv_publication(
        commands,
        first_prompt_token=PROMPT_TOKEN_ID,
        runtime_output=output,
        journal_records=records,
    )
    attention_reference = verify_first_attention_triplet(
        commands,
        first_prompt_token=PROMPT_TOKEN_ID,
        runtime_output=output,
        journal_records=records,
    )
    if (
        kv_reference["status"] != "PASS_BIT_EXACT"
        or attention_reference["status"] != "PASS_BIT_EXACT"
    ):
        raise RuntimeError(f"{label} trial {trial} failed RTL/reference equality")
    return {
        "label": label,
        "trial": trial,
        "measured": measured,
        "batch_commands": batch_commands,
        "command_count": COMMAND_COUNT,
        "wall_seconds": wall_seconds,
        "commands_per_second": COMMAND_COUNT / wall_seconds,
        "segment_simulator_cycles": summary["segment_simulator_cycles"],
        "journal": file_record(output / "progress.journal"),
        "commands_jsonl": file_record(output / "commands.jsonl"),
        "summary": file_record(output / "summary.json"),
        "rtl_reference": {
            "status": "PASS_BIT_EXACT",
            "commands_compared": 4,
            "bytes_compared": int(kv_reference["bytes_compared"])
            + int(attention_reference["bytes_compared"]),
            "kv": kv_reference,
            "attention": attention_reference,
        },
        "argv": display_argv(argv),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("--output must be a fresh empty directory")
    output.mkdir(parents=True, exist_ok=True)

    binary = accepted_runtime.DEFAULT_BINARY.resolve()
    if not binary.is_file():
        raise RuntimeError("missing runtime binary; run `make full-qwen-runtime-build`")
    schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
    commands = build_full_prompt_commands(
        schedule,
        prompt_token_count=1,
        max_new_tokens=1,
    )
    embedding_offset, embedding_shape = accepted_runtime.embedding_tensor_offset(
        accepted_runtime.MODEL
    )
    if embedding_offset != PINNED_EMBEDDING_OFFSET:
        raise RuntimeError("embedding offset changed")
    package_path = output / "microbenchmark_package.bin"
    package = build_ace2rt2_package(
        package_path,
        prompt_token_ids=[PROMPT_TOKEN_ID],
        max_new_tokens=1,
        commands=commands,
        embedding_offset=embedding_offset,
        embedding_shape=embedding_shape,
    )

    source_paths = [
        ROOT / "rtl/ace2_shell.sv",
        ROOT / "verification/verilator/ace2_shell_runtime_harness.sv",
        ROOT / "verification/verilator/ace2_shell_runtime_main.cpp",
        ROOT / "constraints/ace2_rmsnorm_core.sdc",
        Path(__file__).resolve(),
    ]
    contract = {
        "schema_version": 1,
        "name": "host-rtl-persistence-batch-microbenchmark-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sealed_evidence_used": False,
        "official_full_chat_invoked": False,
        "public_rtl_contract": {
            "top_module": "ace2_shell_runtime_harness",
            "parameters": [],
            "port_contract_frozen_by": file_record(
                ROOT / "verification/verilator/ace2_shell_runtime_harness.sv"
            ),
            "clock": "clk_i; one rising edge per Simulator::step",
            "reset": "rst_ni active-low; five reset cycles before shell enable",
            "protocol": "one ordered descriptor at a time; completion consumed before next dispatch",
        },
        "workload": {
            "prompt_token_ids": [PROMPT_TOKEN_ID],
            "max_new_tokens": 1,
            "package_commands": package["commands"],
            "executed_prefix_commands": COMMAND_COUNT,
            "read_response_latency_cycles": 1,
            "timeout_cycles_per_command": 100_000_000,
        },
        "comparison": {
            "baseline_persistence_batch_commands": BASELINE_BATCH_COMMANDS,
            "candidate_persistence_batch_commands": CANDIDATE_BATCH_COMMANDS,
            "warmups_per_mode": 1,
            "measured_trials_per_mode": MEASURED_TRIALS,
            "trial_order": ["baseline", "candidate"] * MEASURED_TRIALS,
            "metric": "executed RTL commands / subprocess wall seconds",
            "measurement_boundary": "immediately before runtime spawn through child exit",
            "score_policy": "median measured commands/s; candidate must be faster with exact artifacts and reference checks",
        },
        "tools": {
            "python": sys.version.split()[0],
            "verilator": run_text(["verilator", "--version"]),
        },
        "source_revision": run_text(["git", "rev-parse", "HEAD"]),
        "sources_and_constraint_hashes": {
            str(path.relative_to(ROOT)): file_record(path) for path in source_paths
        },
        "runtime_binary": file_record(binary),
        "runtime_package": file_record(package_path),
    }
    write_atomic(output / "contract.json", canonical_bytes(contract))

    trials = [
        run_trial(
            label=label,
            trial=0,
            batch_commands=batch,
            root=output,
            package=package_path,
            commands=commands,
            measured=False,
        )
        for label, batch in (
            ("baseline", BASELINE_BATCH_COMMANDS),
            ("candidate", CANDIDATE_BATCH_COMMANDS),
        )
    ]
    measured_counts = {"baseline": 0, "candidate": 0}
    for label, batch in [
        ("baseline", BASELINE_BATCH_COMMANDS),
        ("candidate", CANDIDATE_BATCH_COMMANDS),
    ] * MEASURED_TRIALS:
        measured_counts[label] += 1
        trials.append(
            run_trial(
                label=label,
                trial=measured_counts[label],
                batch_commands=batch,
                root=output,
                package=package_path,
                commands=commands,
                measured=True,
            )
        )

    measured = [trial for trial in trials if trial["measured"]]
    baseline = [trial for trial in measured if trial["label"] == "baseline"]
    candidate = [trial for trial in measured if trial["label"] == "candidate"]
    baseline_throughput = statistics.median(
        trial["commands_per_second"] for trial in baseline
    )
    candidate_throughput = statistics.median(
        trial["commands_per_second"] for trial in candidate
    )
    exact_artifacts = all(
        candidate_trial[name]["sha256"] == baseline_trial[name]["sha256"]
        for baseline_trial, candidate_trial in zip(baseline, candidate, strict=True)
        for name in ("journal", "commands_jsonl", "summary")
    )
    exact_cycles = all(
        trial["segment_simulator_cycles"] == baseline[0]["segment_simulator_cycles"]
        for trial in measured
    )
    reference_pass = all(
        trial["rtl_reference"]["status"] == "PASS_BIT_EXACT" for trial in measured
    )
    result = {
        "schema_version": 1,
        "status": (
            "PASS"
            if exact_artifacts
            and exact_cycles
            and reference_pass
            and candidate_throughput > baseline_throughput
            else "FAIL"
        ),
        "contract": file_record(output / "contract.json"),
        "command_coverage": {
            "executed_per_trial": COMMAND_COUNT,
            "measured_per_mode": COMMAND_COUNT * MEASURED_TRIALS,
            "operator_first": commands[0]["operator"],
            "operator_last": commands[COMMAND_COUNT - 1]["operator"],
            "rtl_reference_commands_per_trial": 4,
            "rtl_reference_bytes_per_trial": 368,
        },
        "baseline": {
            "batch_commands": BASELINE_BATCH_COMMANDS,
            "median_wall_seconds": statistics.median(
                trial["wall_seconds"] for trial in baseline
            ),
            "median_commands_per_second": baseline_throughput,
        },
        "candidate": {
            "batch_commands": CANDIDATE_BATCH_COMMANDS,
            "median_wall_seconds": statistics.median(
                trial["wall_seconds"] for trial in candidate
            ),
            "median_commands_per_second": candidate_throughput,
        },
        "improvement": {
            "commands_per_second": candidate_throughput - baseline_throughput,
            "percent": (candidate_throughput / baseline_throughput - 1.0) * 100.0,
            "over_sealed_1_24_commands_per_second": candidate_throughput > 1.24,
        },
        "equivalence": {
            "journal_jsonl_summary_sha256_exact": exact_artifacts,
            "simulator_cycles_exact": exact_cycles,
            "rtl_reference": "PASS_BIT_EXACT" if reference_pass else "FAIL",
        },
        "limitations": [
            "The workload is a fresh ten-command prefix, not the official full-chat workload.",
            "Independent RTL/reference checks cover four of ten commands (368 output bytes); all ten journal and JSONL records are compared exactly between modes.",
            "Batching preserves clean termination and detected-error durability, but an abrupt host or power loss can lose up to batch_commands-1 recently completed records.",
            "No synthesizable RTL or timing constraint changed; this is host-simulator throughput evidence, not PPA evidence.",
        ],
        "trials": trials,
    }
    write_atomic(output / "results.json", canonical_bytes(result))
    print(
        "ACE2_HOST_RTL_PERSISTENCE_BATCH_"
        f"{result['status']} commands={COMMAND_COUNT} "
        f"baseline_commands_per_second={baseline_throughput:.6f} "
        f"candidate_commands_per_second={candidate_throughput:.6f} "
        f"improvement_percent={result['improvement']['percent']:.3f} "
        f"rtl_reference={result['equivalence']['rtl_reference']} "
        f"artifacts_exact={str(exact_artifacts).lower()}"
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
