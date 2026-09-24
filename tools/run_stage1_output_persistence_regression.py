#!/usr/bin/env python3
"""Measure and bound the Stage-1 host output-persistence repair without model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import rtl_arbitrary_text_generation_backend as backend


SOURCE = ROOT / (
    "build/ace2_chat_demo/"
    "stage1-official-rtl-backed-chat-demo-20260829T175516Z-"
    "runtime-output-0012-6f2c8d4e"
)
TERMINAL = ROOT / (
    "build/ace2_chat_demo/"
    "stage1-official-rtl-backed-chat-demo-20260829T175516Z-"
    "attempt-0012-6f2c8d4e-authority-state/terminal-status.json"
)
DEFAULT_OUTPUT = ROOT / "build/stage1-output-persistence-successor-v1/attempt-0001"
MAX_POSITIONS = backend.MAX_CONTEXT_TOKENS - 1
LAYERS = backend.LAYERS
MAX_HEADS = 4
HEAD_RETAINED_ALLOWANCE_BYTES = 64 << 20
TOP_LEVEL_RETAINED_ALLOWANCE_BYTES = 32 << 20
FILESYSTEM_METADATA_ALLOWANCE_BYTES = 256 << 20
TEMPORARY_WRITE_ALLOWANCE_BYTES = 8 << 30
PACKAGE_ALLOWANCE_BYTES = 32 << 20
SAFETY_MARGIN_PERCENT = 25


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    backend.write_json(path, value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def usage(path: Path, *, exclude_transient: bool = False) -> dict[str, int]:
    logical = 0
    allocated = 0
    files = 0
    directories = 0
    for item in [path, *path.rglob("*")]:
        relative = item.relative_to(path)
        if exclude_transient and any(
            part in backend.TRANSIENT_EXECUTION_DIRECTORIES
            for part in relative.parts
        ):
            continue
        stat = item.stat(follow_symlinks=False)
        allocated += stat.st_blocks * 512
        if item.is_dir():
            directories += 1
        elif item.is_file():
            logical += stat.st_size
            files += 1
    return {
        "logical_bytes": logical,
        "allocated_bytes": allocated,
        "files": files,
        "directories": directories,
    }


def classify(relative: Path) -> str:
    parts = relative.parts
    if any(part in backend.TRANSIENT_EXECUTION_DIRECTORIES for part in parts):
        return "regenerable_execution_working_set"
    if parts and parts[0] == "shared":
        return "shared_live_support"
    if "logs" in parts:
        return "claim_bearing_simulator_logs"
    if "rtl_observed" in parts:
        return "claim_bearing_rtl_observations"
    if relative.suffix == ".json":
        return "claim_bearing_results_and_accountability"
    return "other_retained"


def classified_inventory(root: Path) -> dict[str, Any]:
    classes: dict[str, dict[str, int]] = {}
    total = {"logical_bytes": 0, "allocated_bytes": 0, "files": 0}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        record = classes.setdefault(
            classify(path.relative_to(root)),
            {"logical_bytes": 0, "allocated_bytes": 0, "files": 0},
        )
        record["logical_bytes"] += stat.st_size
        record["allocated_bytes"] += stat.st_blocks * 512
        record["files"] += 1
        total["logical_bytes"] += stat.st_size
        total["allocated_bytes"] += stat.st_blocks * 512
        total["files"] += 1
    return {"total": total, "classes": classes}


def complete_units(pattern: str, result_name: str) -> list[Path]:
    return sorted(
        path.parent
        for path in SOURCE.glob(f"{pattern}/{result_name}")
        if path.is_file()
    )


def retained_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not any(
            part in backend.TRANSIENT_EXECUTION_DIRECTORIES
            for part in path.relative_to(root).parts
        )
    }


def source_guard() -> dict[str, Any]:
    inventory = classified_inventory(SOURCE)
    stat = SOURCE.stat()
    return {
        "device": stat.st_dev,
        "mtime_ns": stat.st_mtime_ns,
        "total": inventory["total"],
        "run_started_sha256": sha256_file(SOURCE / "run.started.json"),
        "terminal_status_sha256": sha256_file(TERMINAL),
    }


def main() -> int:
    global SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("--output must not exist")
    if not SOURCE.is_dir() or not TERMINAL.is_file():
        raise RuntimeError("attempt-0012 immutable output or terminal seal is absent")
    terminal = json.loads(TERMINAL.read_text(encoding="utf-8"))
    if (
        terminal.get("status") != "SEALED_TERMINAL_NO_RETRY"
        or terminal.get("classification") != "MODEL_DISK_GROWTH_LIMIT"
    ):
        raise RuntimeError("attempt-0012 terminal classification differs")

    output.mkdir(parents=True)
    before = source_guard()
    source_inventory = classified_inventory(SOURCE)
    layer_units = complete_units(
        "tokens/position-*/layer-*", "rtl_execution.json"
    )
    sidecar_units = sorted(
        path
        for path in SOURCE.glob(
            "rank1-sidecar-steps/position-*/layer-*"
        )
        if path.is_dir()
    )
    if not layer_units:
        raise RuntimeError("attempt-0012 has no complete layer unit")

    layer_measurements = [
        {
            "path": path.relative_to(SOURCE).as_posix(),
            "unrepaired": usage(path),
            "repaired": usage(path, exclude_transient=True),
        }
        for path in layer_units
    ]
    sidecar_measurements = [
        {
            "path": path.relative_to(SOURCE).as_posix(),
            "unrepaired": usage(path),
            "repaired": usage(path, exclude_transient=True),
        }
        for path in sidecar_units
    ]
    selected_layer = max(
        layer_measurements,
        key=lambda item: item["unrepaired"]["allocated_bytes"],
    )
    selected_path = SOURCE / selected_layer["path"]
    baseline = output / "controlled-ab/baseline"
    candidate = output / "controlled-ab/candidate"
    shutil.copytree(selected_path, baseline)
    shutil.copytree(selected_path, candidate)
    expected_retained = retained_hashes(baseline)
    prune_result = backend.prune_transient_execution_artifacts(candidate)
    observed_retained = retained_hashes(candidate)
    baseline_usage = usage(baseline)
    candidate_usage = usage(candidate)
    comparison_pass = (
        expected_retained == observed_retained
        and candidate_usage["allocated_bytes"] < baseline_usage["allocated_bytes"]
        and all(
            not (candidate / name).exists()
            for name in backend.TRANSIENT_EXECUTION_DIRECTORIES
        )
    )

    max_layer_retained = max(
        item["repaired"]["allocated_bytes"] for item in layer_measurements
    )
    max_sidecar_retained = max(
        (
            item["repaired"]["allocated_bytes"]
            for item in sidecar_measurements
        ),
        default=0,
    )
    shared_live = usage(SOURCE / "shared")["allocated_bytes"]
    runtime_output_bound = (
        MAX_POSITIONS * LAYERS * max_layer_retained
        + MAX_POSITIONS * max_sidecar_retained
        + MAX_HEADS * HEAD_RETAINED_ALLOWANCE_BYTES
        + shared_live
        + TOP_LEVEL_RETAINED_ALLOWANCE_BYTES
        + FILESYSTEM_METADATA_ALLOWANCE_BYTES
    )
    subtotal = (
        runtime_output_bound
        + TEMPORARY_WRITE_ALLOWANCE_BYTES
        + PACKAGE_ALLOWANCE_BYTES
    )
    safety_margin_bytes = math.ceil(
        subtotal * SAFETY_MARGIN_PERCENT / 100
    )
    required_available_bytes = subtotal + safety_margin_bytes
    destination_parent = output.parent
    filesystem = os.statvfs(destination_parent)
    available_bytes = filesystem.f_bavail * filesystem.f_frsize
    same_device = SOURCE.stat().st_dev == destination_parent.stat().st_dev
    capacity_pass = same_device and available_bytes >= required_available_bytes
    after = source_guard()
    source_unchanged = before == after

    contract = {
        "schema": "ace2-stage1-output-persistence-successor-contract-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": {
            "official_model_path_executed": False,
            "rtl_executed": False,
            "sealed_evidence_mutated": False,
            "controlled_input": selected_layer["path"],
        },
        "complete_run_geometry": {
            "maximum_context_tokens": backend.MAX_CONTEXT_TOKENS,
            "maximum_executed_positions": MAX_POSITIONS,
            "layers_per_position": LAYERS,
            "maximum_heads": MAX_HEADS,
        },
        "retention_policy": {
            "retained": [
                "result JSON including reference checks and latency",
                "simulator stdout/stderr logs",
                "RTL-observed K/V and corrected-V bytes",
            ],
            "regenerable_after_durable_result": list(
                backend.TRANSIENT_EXECUTION_DIRECTORIES
            ),
        },
        "writer_code_paths": {
            "layer_vectors": "persist_layer_vectors",
            "layer_tensors": "persist_layer_tensors",
            "generated_testbenches_and_simulators": "compile_and_run",
            "rank1_vectors": "Rank1SidecarSession.apply",
            "shared_lm_head_weight": "derive_lm_head_weights",
            "layer_result": "run_layer_rtl",
            "head_result": "run_final_head_rtl",
            "session_result": "run_generation",
        },
        "source_bindings": {
            "backend": {
                "path": "tools/rtl_arbitrary_text_generation_backend.py",
                "sha256": sha256_file(
                    ROOT / "tools/rtl_arbitrary_text_generation_backend.py"
                ),
            },
            "regression": {
                "path": "tools/run_stage1_output_persistence_regression.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }
    diagnosis = {
        "schema": "ace2-attempt-0012-output-growth-diagnosis-v1",
        "terminal_classification": terminal["classification"],
        "source_read_only_guard_before": before,
        "source_read_only_guard_after": after,
        "source_unchanged": source_unchanged,
        "artifact_inventory": source_inventory,
        "complete_layer_units": len(layer_units),
        "complete_rank1_sidecar_units": len(sidecar_units),
        "layer_worst_cases": {
            "unrepaired_allocated_bytes": max(
                item["unrepaired"]["allocated_bytes"]
                for item in layer_measurements
            ),
            "repaired_allocated_bytes": max_layer_retained,
        },
        "rank1_sidecar_worst_cases": {
            "repaired_allocated_bytes": max_sidecar_retained,
        },
        "root_cause_hypothesis": (
            "Per-layer regenerated vectors, tensors, generated testbenches, "
            "and simulator binaries were retained for every layer; their "
            "measured allocated bytes materially dominate the sealed output."
        ),
        "failure_taxonomy": "host_output_capacity",
        "regression": (
            "copy one complete immutable layer, apply the production pruning "
            "helper, and require byte-exact retained evidence with lower allocation"
        ),
    }
    controlled = {
        "schema": "ace2-stage1-output-persistence-controlled-ab-v1",
        "status": "PASS" if comparison_pass else "FAIL",
        "official": False,
        "model_executed": False,
        "rtl_executed": False,
        "selected_unit": selected_layer["path"],
        "baseline": baseline_usage,
        "candidate": candidate_usage,
        "prune_result": prune_result,
        "retained_file_count": len(expected_retained),
        "retained_sha256_exact": expected_retained == observed_retained,
    }
    capacity = {
        "schema": "ace2-stage1-output-capacity-proof-v1",
        "status": "PASS" if capacity_pass else "FAIL",
        "same_device": same_device,
        "filesystem_device": destination_parent.stat().st_dev,
        "available_bytes": available_bytes,
        "runtime_output_bound_bytes": runtime_output_bound,
        "bound_terms": {
            "layer_units": MAX_POSITIONS * LAYERS,
            "per_layer_retained_allocated_bytes": max_layer_retained,
            "rank1_sidecar_units": MAX_POSITIONS,
            "per_rank1_sidecar_retained_allocated_bytes": max_sidecar_retained,
            "head_units": MAX_HEADS,
            "per_head_retained_allowance_bytes": HEAD_RETAINED_ALLOWANCE_BYTES,
            "shared_live_support_bytes": shared_live,
            "top_level_retained_allowance_bytes": TOP_LEVEL_RETAINED_ALLOWANCE_BYTES,
            "filesystem_metadata_allowance_bytes": FILESYSTEM_METADATA_ALLOWANCE_BYTES,
        },
        "temporary_write_allowance_bytes": TEMPORARY_WRITE_ALLOWANCE_BYTES,
        "package_allowance_bytes": PACKAGE_ALLOWANCE_BYTES,
        "safety_margin_percent": SAFETY_MARGIN_PERCENT,
        "safety_margin_bytes": safety_margin_bytes,
        "required_available_bytes": required_available_bytes,
        "headroom_bytes": available_bytes - required_available_bytes,
    }
    overall_pass = comparison_pass and capacity_pass and source_unchanged
    review_request = {
        "schema": "ace2-stage1-output-persistence-review-request-v1",
        "status": "PENDING_FRESH_REVIEW",
        "engineer_result": "PASS" if overall_pass else "FAIL",
        "review_checks": [
            "attempt-0012 diagnosis is read-only and terminal-bound",
            "controlled A/B retains claim-bearing files byte-exactly",
            "complete-run arithmetic includes temporary and package allowances",
            "same-device available capacity exceeds the bound plus 25 percent",
            "no official model path, RTL execution, authority, or U280 work occurred",
        ],
    }
    write_json(output / "contract.json", contract)
    write_json(output / "diagnosis.json", diagnosis)
    write_json(output / "controlled-regression.json", controlled)
    write_json(output / "capacity.json", capacity)
    write_json(output / "review-request.json", review_request)
    manifest = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.glob("*.json"))
    }
    write_json(output / "manifest.json", manifest)
    print(
        "ACE2_STAGE1_OUTPUT_PERSISTENCE_"
        f"{'PASS' if overall_pass else 'FAIL'} "
        f"source_bytes={source_inventory['total']['allocated_bytes']} "
        f"runtime_bound_bytes={runtime_output_bound} "
        f"required_available_bytes={required_available_bytes} "
        f"available_bytes={available_bytes} "
        f"retained_exact={str(expected_retained == observed_retained).lower()}"
    )
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
