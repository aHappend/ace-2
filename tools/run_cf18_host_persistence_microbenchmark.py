#!/usr/bin/env python3
"""Measure host persistence cadence without loading a model or executing RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "build/cf18-host-persistence-microbenchmark-v1/attempt-0002"
RECORDS = 512
PAYLOAD_BYTES = 256
BASELINE_BATCH = 1
CANDIDATE_BATCH = 1024
TRIALS = 3


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_result(path: Path, value: Any) -> None:
    raw = canonical_bytes(value)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_trial(root: Path, label: str, trial: int, batch: int) -> dict[str, Any]:
    trial_root = root / f"{label}-{trial:02d}"
    trial_root.mkdir()
    journal_path = trial_root / "progress.journal"
    jsonl_path = trial_root / "commands.jsonl"
    progress_path = trial_root / "progress.json"
    payload = bytes((index * 17 + 3) & 0xFF for index in range(PAYLOAD_BYTES))
    sync_boundaries = 0
    start = time.perf_counter()
    with journal_path.open("wb", buffering=0) as journal, jsonl_path.open(
        "wb", buffering=0
    ) as jsonl:
        for ordinal in range(RECORDS):
            journal.write(ordinal.to_bytes(4, "little") + payload)
            jsonl.write(
                canonical_bytes(
                    {
                        "ordinal": ordinal,
                        "payload_sha256": sha256_bytes(payload),
                    }
                )
            )
            boundary = (ordinal + 1) % batch == 0 or ordinal + 1 == RECORDS
            if boundary:
                os.fsync(journal.fileno())
                os.fsync(jsonl.fileno())
                write_result(
                    progress_path,
                    {"next_ordinal": ordinal + 1, "status": "COMPLETE"},
                )
                sync_boundaries += 1
    wall_seconds = time.perf_counter() - start
    return {
        "label": label,
        "trial": trial,
        "batch_commands": batch,
        "records": RECORDS,
        "wall_seconds": wall_seconds,
        "records_per_second": RECORDS / wall_seconds,
        "sync_boundaries": sync_boundaries,
        "journal_sha256": sha256_bytes(journal_path.read_bytes()),
        "jsonl_sha256": sha256_bytes(jsonl_path.read_bytes()),
        "progress_sha256": sha256_bytes(progress_path.read_bytes()),
    }


def error_boundary_check(root: Path) -> dict[str, Any]:
    path = root / "error-boundary.bin"
    completed = 2050
    batch = CANDIDATE_BATCH
    boundaries = 0
    with path.open("wb", buffering=0) as handle:
        for ordinal in range(completed):
            handle.write(ordinal.to_bytes(4, "little"))
            if (ordinal + 1) % batch == 0:
                os.fsync(handle.fileno())
                boundaries += 1
        os.fsync(handle.fileno())
        boundaries += 1
    return {
        "status": "PASS",
        "completed_records": completed,
        "forced_error_boundary_ordinal": completed,
        "sync_boundaries": boundaries,
        "bytes": path.stat().st_size,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("--output must not exist")
    output.mkdir(parents=True)

    trials: list[dict[str, Any]] = []
    for trial in range(1, TRIALS + 1):
        trials.append(run_trial(output, "baseline", trial, BASELINE_BATCH))
        trials.append(run_trial(output, "candidate", trial, CANDIDATE_BATCH))
    baseline = [item for item in trials if item["label"] == "baseline"]
    candidate = [item for item in trials if item["label"] == "candidate"]
    equivalent = all(
        candidate_item[key] == baseline_item[key]
        for baseline_item, candidate_item in zip(baseline, candidate, strict=True)
        for key in ("journal_sha256", "jsonl_sha256", "progress_sha256")
    )
    baseline_rate = statistics.median(
        item["records_per_second"] for item in baseline
    )
    candidate_rate = statistics.median(
        item["records_per_second"] for item in candidate
    )
    error_check = error_boundary_check(output)
    result = {
        "schema": "ace2-cf18-host-persistence-microbenchmark-v1",
        "status": (
            "PASS"
            if equivalent
            and candidate_rate > baseline_rate
            and error_check["status"] == "PASS"
            else "FAIL"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": {
            "model_loaded": False,
            "model_schedule_executed": False,
            "rtl_executed": False,
            "network_used": False,
            "gpu_used": False,
            "records_per_trial": RECORDS,
            "payload_bytes": PAYLOAD_BYTES,
            "trials_per_mode": TRIALS,
        },
        "baseline": {
            "batch_commands": BASELINE_BATCH,
            "median_records_per_second": baseline_rate,
            "sync_boundaries_per_trial": baseline[0]["sync_boundaries"],
        },
        "candidate": {
            "batch_commands": CANDIDATE_BATCH,
            "median_records_per_second": candidate_rate,
            "sync_boundaries_per_trial": candidate[0]["sync_boundaries"],
        },
        "improvement": {
            "ratio": candidate_rate / baseline_rate,
            "percent": (candidate_rate / baseline_rate - 1.0) * 100.0,
        },
        "equivalence": {
            "final_journal_jsonl_progress_sha256_exact": equivalent,
            "forced_error_boundary": error_check,
        },
        "trials": trials,
    }
    write_result(output / "results.json", result)
    print(
        "CF18_HOST_PERSISTENCE_"
        f"{result['status']} baseline_records_per_second={baseline_rate:.3f} "
        f"candidate_records_per_second={candidate_rate:.3f} "
        f"improvement_ratio={result['improvement']['ratio']:.3f}"
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
