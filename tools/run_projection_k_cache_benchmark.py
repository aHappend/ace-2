#!/usr/bin/env python3
"""Freeze and measure the synthetic ACE-2 K-projection beat-cache benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "benchmark/projection_k_cache_v3/contract.json"
PUBLIC_RESULTS = ROOT / "benchmark/projection_k_cache_v3/results.json"
BASELINE_SHELL = (
    ROOT / "build/projection-k-cache-v3/frozen-baseline/rtl/ace2_shell.sv"
)
TB = ROOT / "verification/tb/ace2_projection_k_cache_benchmark_tb.sv"
LIVE_SHELL = ROOT / "rtl/ace2_shell.sv"
OUTPUTS = 128
K_SIZE = 896
MEASURED_TRIALS = 3
PASS_RE = re.compile(
    r"ACE2_PROJECTION_K_CACHE_BENCHMARK_PASS "
    r"host_commands=(?P<host_commands>\d+) "
    r"outputs_checked=(?P<outputs_checked>\d+) "
    r"simulator_cycles=(?P<simulator_cycles>\d+) "
    r"memory_read_requests=(?P<memory_read_requests>\d+) "
    r"memory_write_requests=(?P<memory_write_requests>\d+) "
    r"saturation=(?P<saturation>\d+)"
)
RTL_SOURCES = [
    "rtl/ace2_pkg.sv",
    "rtl/ace2_rmsnorm_core.sv",
    "rtl/ace2_dynamic_scale32_core.sv",
    "rtl/ace2_w4a8_proj_core.sv",
    "rtl/ace2_rope_core.sv",
    "rtl/ace2_dynamic_rope_head_core.sv",
    "rtl/ace2_fixed_q7_rope_score_core.sv",
    "rtl/ace2_relative_rope_score_fusion_core.sv",
    "rtl/ace2_attention_score_core.sv",
    "rtl/ace2_softmax_core.sv",
    "rtl/ace2_attention_compose_core.sv",
    "rtl/ace2_silu_gate_core.sv",
]


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def tool_version(argv: list[str]) -> str:
    completed = subprocess.run(
        argv, cwd=ROOT, check=False, text=True, capture_output=True
    )
    raw = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0 or not raw:
        raise RuntimeError(f"tool version failed: {' '.join(argv)}")
    return raw.splitlines()[0]


def to_sint(value: int, width: int) -> int:
    value &= (1 << width) - 1
    return value - (1 << width) if value & (1 << (width - 1)) else value


def round_shift_even(value: int, shift: int) -> int:
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    base = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (base & 1)):
        base += 1
    return sign * base


def activation(element: int) -> int:
    return ((element * 13 + 7) % 255) - 127


def weight(output: int, element: int) -> int:
    return ((output * 5 + element * 3 + 1) % 15) - 7


def reference_outputs() -> bytes:
    outputs = bytearray()
    activations = [activation(element) for element in range(K_SIZE)]
    for output in range(OUTPUTS):
        accumulator = sum(
            act * weight(output, element)
            for element, act in enumerate(activations)
        )
        accumulator += output * 31 - 1984
        multiplier = to_sint(1_000_003 + output * 97, 32)
        scaled = round_shift_even(accumulator * multiplier, 25)
        outputs.append(max(-128, min(127, scaled)) & 0xFF)
    return bytes(outputs)


def source_records(shell: Path) -> dict[str, dict[str, Any]]:
    records = {
        path: file_record(ROOT / path)
        for path in RTL_SOURCES
    }
    records["rtl/ace2_shell.sv"] = file_record(shell)
    records[str(TB.relative_to(ROOT))] = file_record(TB)
    records[str(Path(__file__).resolve().relative_to(ROOT))] = file_record(
        Path(__file__).resolve()
    )
    records["constraints/ace2_rmsnorm_core.sdc"] = file_record(
        ROOT / "constraints/ace2_rmsnorm_core.sdc"
    )
    return records


def freeze() -> None:
    if CONTRACT.exists() or BASELINE_SHELL.exists():
        raise RuntimeError("projection K-cache benchmark is already frozen")
    BASELINE_SHELL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LIVE_SHELL, BASELINE_SHELL)
    expected = reference_outputs()
    contract = {
        "schema_version": 1,
        "name": "projection-k-cache-v3",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "fixed_external_benchmark": False,
        "protected_or_sealed_inputs_used": False,
        "official_full_chat_invoked": False,
        "public_rtl_contract": {
            "top_module": "ace2_shell",
            "parameters": "unchanged defaults",
            "ports_frozen_by": file_record(ROOT / "design/BENCHMARK_INTERFACE.json"),
            "clock": "clk_i, 10 ns period",
            "reset": "rst_ni active-low for five rising edges",
            "command_protocol": "one ready/valid descriptor and one tagged completion",
            "memory_protocol": "one-beat abstract streaming requests with deterministic ready stalls and one-cycle read response",
        },
        "workload": {
            "operator": "layer-0 K projection",
            "opcode": 1,
            "flags": 0,
            "m": 1,
            "n": OUTPUTS,
            "k": K_SIZE,
            "host_descriptor_commands": 1,
            "outputs_checked": OUTPUTS,
            "activation_formula": "((element*13+7)%255)-127",
            "weight_formula": "((output*5+element*3+1)%15)-7",
            "bias_formula": "output*31-1984",
            "multiplier_formula": "1000003+output*97",
            "right_shift": 25,
            "output_zero_point": 0,
            "expected_output": {
                "bytes": len(expected),
                "sha256": sha256_bytes(expected),
            },
        },
        "score_policy": {
            "primary": "candidate simulator cycles <= 75% of frozen baseline",
            "secondary": "report median simulator wall time over three measured trials",
            "correctness": "all 128 RTL output bytes exactly equal the Python fixed-point reference",
            "command_accounting": "report host descriptor, external memory read, and external memory write request counts",
        },
        "tools": {
            "python": sys.version.split()[0],
            "iverilog": tool_version(["iverilog", "-V"]),
            "vvp": tool_version(["vvp", "-V"]),
        },
        "frozen_baseline": {
            "path": str(BASELINE_SHELL.relative_to(ROOT)),
            **file_record(BASELINE_SHELL),
        },
        "sources_and_constraint_hashes": source_records(BASELINE_SHELL),
        "limitations": [
            "This is a deterministic synthetic layer-0 K projection, not a full model or chat attempt.",
            "The benchmark measures one shell descriptor and the abstract streaming-memory boundary; it does not model a physical DRAM controller.",
            "The score is cycle reduction with exact output agreement; wall time is reported but not a pass gate.",
        ],
    }
    write_atomic(CONTRACT, canonical_bytes(contract))
    print(
        "ACE2_PROJECTION_K_CACHE_FREEZE_PASS "
        f"baseline_sha256={contract['frozen_baseline']['sha256']} "
        f"expected_sha256={contract['workload']['expected_output']['sha256']}"
    )


def read_contract() -> dict[str, Any]:
    if not CONTRACT.is_file():
        raise RuntimeError("missing frozen contract; run the freeze action first")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    baseline = contract["frozen_baseline"]
    if file_record(BASELINE_SHELL) != {
        "bytes": baseline["bytes"],
        "sha256": baseline["sha256"],
    }:
        raise RuntimeError("frozen baseline shell changed")
    expected = reference_outputs()
    if {
        "bytes": len(expected),
        "sha256": sha256_bytes(expected),
    } != contract["workload"]["expected_output"]:
        raise RuntimeError("reference workload changed after freeze")
    return contract


def compile_benchmark(shell: Path, output: Path) -> Path:
    binary = output / "ace2_projection_k_cache_benchmark.vvp"
    sources = [str(ROOT / path) for path in RTL_SOURCES]
    sources.extend([str(shell), str(TB)])
    completed = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-Irtl",
            "-s",
            "ace2_projection_k_cache_benchmark_tb",
            "-o",
            str(binary),
            *sources,
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    (output / "compile.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "compile.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark compile failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return binary


def measure(label: str, shell: Path, output: Path) -> None:
    contract = read_contract()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("measurement output must be a fresh empty directory")
    output.mkdir(parents=True, exist_ok=True)
    if label == "baseline" and file_record(shell) != {
        "bytes": contract["frozen_baseline"]["bytes"],
        "sha256": contract["frozen_baseline"]["sha256"],
    }:
        raise RuntimeError("baseline measurement did not use the frozen shell")

    expected_path = output / "expected.hex"
    expected_path.write_text(
        "".join(f"{value:02x}\n" for value in reference_outputs()),
        encoding="ascii",
    )
    binary = compile_benchmark(shell, output)
    trials: list[dict[str, Any]] = []
    for trial in range(MEASURED_TRIALS + 1):
        start = time.perf_counter()
        completed = subprocess.run(
            ["vvp", str(binary), f"+EXPECTED={expected_path}"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        wall_seconds = time.perf_counter() - start
        trial_name = "warmup" if trial == 0 else f"measured-{trial:02d}"
        (output / f"{trial_name}.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (output / f"{trial_name}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        match = PASS_RE.search(completed.stdout)
        if completed.returncode != 0 or match is None:
            raise RuntimeError(
                f"{label} {trial_name} failed:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        metrics = {key: int(value) for key, value in match.groupdict().items()}
        if metrics["outputs_checked"] != OUTPUTS or metrics["host_commands"] != 1:
            raise RuntimeError("benchmark coverage changed")
        trials.append(
            {
                "trial": trial_name,
                "measured": trial != 0,
                "wall_seconds": wall_seconds,
                **metrics,
            }
        )

    measured = [trial for trial in trials if trial["measured"]]
    deterministic = all(
        {
            key: trial[key]
            for key in (
                "host_commands",
                "outputs_checked",
                "simulator_cycles",
                "memory_read_requests",
                "memory_write_requests",
                "saturation",
            )
        }
        == {
            key: measured[0][key]
            for key in (
                "host_commands",
                "outputs_checked",
                "simulator_cycles",
                "memory_read_requests",
                "memory_write_requests",
                "saturation",
            )
        }
        for trial in measured
    )
    if not deterministic:
        raise RuntimeError("cycle or command metrics were not deterministic")
    result = {
        "schema_version": 1,
        "label": label,
        "status": "PASS_BIT_EXACT",
        "contract": file_record(CONTRACT),
        "rtl_shell": {
            "path": str(shell.relative_to(ROOT)),
            **file_record(shell),
        },
        "expected": file_record(expected_path),
        "binary": file_record(binary),
        "metrics": {
            **{
                key: measured[0][key]
                for key in (
                    "host_commands",
                    "outputs_checked",
                    "simulator_cycles",
                    "memory_read_requests",
                    "memory_write_requests",
                    "saturation",
                )
            },
            "median_wall_seconds": statistics.median(
                trial["wall_seconds"] for trial in measured
            ),
        },
        "trials": trials,
    }
    write_atomic(output / "result.json", canonical_bytes(result))
    print(
        f"ACE2_PROJECTION_K_CACHE_{label.upper()}_PASS "
        f"cycles={result['metrics']['simulator_cycles']} "
        f"reads={result['metrics']['memory_read_requests']} "
        f"wall_seconds={result['metrics']['median_wall_seconds']:.6f}"
    )


def compare(baseline_path: Path, candidate_path: Path, output: Path) -> None:
    contract = read_contract()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if baseline["status"] != "PASS_BIT_EXACT" or candidate["status"] != "PASS_BIT_EXACT":
        raise RuntimeError("baseline or candidate lacks exact reference agreement")
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    cycles_ratio = (
        candidate_metrics["simulator_cycles"]
        / baseline_metrics["simulator_cycles"]
    )
    reads_ratio = (
        candidate_metrics["memory_read_requests"]
        / baseline_metrics["memory_read_requests"]
    )
    status = (
        "PASS"
        if candidate_metrics["host_commands"] == baseline_metrics["host_commands"] == 1
        and candidate_metrics["memory_write_requests"]
        == baseline_metrics["memory_write_requests"]
        and cycles_ratio <= 0.75
        and candidate_metrics["memory_read_requests"]
        < baseline_metrics["memory_read_requests"]
        else "FAIL"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "contract": file_record(CONTRACT),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "improvement": {
            "simulator_cycles_percent": (1.0 - cycles_ratio) * 100.0,
            "memory_read_requests_percent": (1.0 - reads_ratio) * 100.0,
            "median_wall_time_percent": (
                1.0
                - candidate_metrics["median_wall_seconds"]
                / baseline_metrics["median_wall_seconds"]
            )
            * 100.0,
        },
        "equivalence": {
            "rtl_reference": "PASS_BIT_EXACT",
            "bytes_compared_per_trial": OUTPUTS,
            "host_descriptor_commands_unchanged": True,
            "memory_write_requests_unchanged": (
                candidate_metrics["memory_write_requests"]
                == baseline_metrics["memory_write_requests"]
            ),
        },
        "sources_and_constraint_hashes": source_records(LIVE_SHELL),
        "rtl_scope": (
            "One 128-bit activation beat register and one 128-bit W4 weight beat "
            "register in ace2_shell; reuse is internal to a projection output and "
            "does not change ports, descriptors, W4A8 arithmetic, or memory layout."
        ),
        "limitations": contract["limitations"]
        + [
            "Activation-beat reuse is disabled for dynamic-sidecar projections; weight-beat reuse remains enabled.",
            "Canonical SKY130 synthesis/OpenSTA is reported separately and independent Reviewer acceptance remains pending.",
        ],
    }
    write_atomic(output, canonical_bytes(result))
    print(
        f"ACE2_PROJECTION_K_CACHE_COMPARE_{status} "
        f"cycle_improvement_percent={result['improvement']['simulator_cycles_percent']:.3f} "
        f"read_improvement_percent={result['improvement']['memory_read_requests_percent']:.3f} "
        f"wall_improvement_percent={result['improvement']['median_wall_time_percent']:.3f}"
    )
    if status != "PASS":
        raise RuntimeError("candidate did not meet the frozen score policy")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("freeze")
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--label", choices=("baseline", "candidate"), required=True)
    measure_parser.add_argument("--rtl-shell", type=Path, required=True)
    measure_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, default=PUBLIC_RESULTS)
    args = parser.parse_args()

    if args.action == "freeze":
        freeze()
    elif args.action == "measure":
        measure(
            args.label,
            (ROOT / args.rtl_shell).resolve()
            if not args.rtl_shell.is_absolute()
            else args.rtl_shell.resolve(),
            (ROOT / args.output).resolve()
            if not args.output.is_absolute()
            else args.output.resolve(),
        )
    else:
        compare(
            (ROOT / args.baseline).resolve()
            if not args.baseline.is_absolute()
            else args.baseline.resolve(),
            (ROOT / args.candidate).resolve()
            if not args.candidate.is_absolute()
            else args.candidate.resolve(),
            (ROOT / args.output).resolve()
            if not args.output.is_absolute()
            else args.output.resolve(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
