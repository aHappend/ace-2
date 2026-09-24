#!/usr/bin/env python3
"""Run and bind focused RTL verification for the selected group-1 repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "evidence/candidates/w4a8-layer16-source-grouped-activation-repair-v1/"
    "candidate-0002"
)
SELECTED_GROUP = CANDIDATE / "groups/group-001"
GENERATED = ROOT / "verification/generated/ace2_layer16_group1_scale32"
RTL = ROOT / "rtl/ace2_layer16_group1_scale32_core.sv"
RMS_RTL = ROOT / "rtl/ace2_rmsnorm_core.sv"
PROJ_RTL = ROOT / "rtl/ace2_w4a8_proj_core.sv"
TB = ROOT / "verification/tb/ace2_layer16_group1_scale32_tb.sv"
SEARCH = ROOT / "tools/ace2_checkpoint176_layer16_source_grouped_scale32_search.py"
GENERATOR = ROOT / "tools/gen_layer16_group1_scale32_vectors.py"
UNIT_TEST = ROOT / "verification/test_layer16_group1_scale32_reference.py"
RUNNER = Path(__file__).resolve()


PASS_PATTERN = re.compile(
    r"ACE2_LAYER16_GROUP1_SCALE32_RTL_PASS "
    r"directed=(?P<directed>\d+) groups=(?P<groups>\d+) "
    r"rms_beats=(?P<rms_beats>\d+) q_outputs=(?P<q_outputs>\d+) "
    r"q_saturations=(?P<q_saturations>\d+) "
    r"source_stalls=(?P<source_stalls>\d+) gain_stalls=(?P<gain_stalls>\d+) "
    r"weight_stalls=(?P<weight_stalls>\d+) output_stalls=(?P<output_stalls>\d+) "
    r"cycles=(?P<cycles>\d+) xz_clean=1 reset_recovery=1 clear_recovery=1"
)
EXPECTED_METRICS = {
    "directed": 4,
    "groups": 896,
    "rms_beats": 56,
    "q_outputs": 896,
    "q_saturations": 1,
    "source_stalls": 449,
    "gain_stalls": 8725,
    "weight_stalls": 114583,
    "output_stalls": 1791,
    "cycles": 1940159,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def run(
    command: list[str],
    log: Path,
    *,
    cwd: Path = ROOT,
) -> str:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(completed.stdout, encoding="utf-8")
    require(
        completed.returncode == 0,
        f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}",
    )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")
    parser.add_argument("--verilator", default="verilator")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(ROOT.resolve() in output.parents, "output must be repository-relative")
    require(not output.exists(), "output already exists")
    output.mkdir(parents=True)

    candidate_result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
    require(candidate_result["selected_source_group_size"] == 1, "selected group differs")
    require(
        candidate_result["selected_reference_rank"]
        < candidate_result["baseline_reference_rank"],
        "selected candidate does not improve the frozen baseline",
    )

    candidate_log = output / "candidate_sha256_check.log"
    unit_log = output / "reference_unittest.log"
    vector_log = output / "vector_regeneration_check.log"
    compile_log = output / "iverilog_compile.log"
    simulation_log = output / "simulation.log"
    lint_log = output / "verilator_lint.log"
    tool_log = output / "tool_versions.log"

    run(["sha256sum", "-c", "SHA256SUMS"], candidate_log, cwd=CANDIDATE)
    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "verification.test_layer16_group1_scale32_reference",
            "-v",
        ],
        unit_log,
    )
    run([sys.executable, str(GENERATOR), "--check"], vector_log)

    with tempfile.TemporaryDirectory(
        prefix="ace2-layer16-group1-scale32-", dir=ROOT / "build"
    ) as temporary:
        executable = Path(temporary) / "tb.vvp"
        compile_stdout = run(
            [
                args.iverilog,
                "-g2012",
                "-Wall",
                "-Wno-timescale",
                "-Iverification/generated",
                "-s",
                "ace2_layer16_group1_scale32_tb",
                "-o",
                str(executable),
                str(RMS_RTL),
                str(PROJ_RTL),
                str(RTL),
                str(TB),
            ],
            compile_log,
        )
        require(not compile_stdout.strip(), "Icarus emitted warning or diagnostic output")
        simulation_stdout = run([args.vvp, str(executable)], simulation_log)

    match = PASS_PATTERN.search(simulation_stdout)
    require(match is not None, "focused RTL PASS marker is missing")
    metrics = {name: int(value) for name, value in match.groupdict().items()}
    require(metrics == EXPECTED_METRICS, f"focused RTL metrics differ: {metrics}")

    lint_stdout = run(
        [
            args.verilator,
            "--lint-only",
            "-Wall",
            "-Wno-fatal",
            "--top-module",
            "ace2_layer16_group1_rmsnorm_q_core",
            str(RMS_RTL),
            str(PROJ_RTL),
            str(RTL),
        ],
        lint_log,
    )
    require(not lint_stdout.strip(), "Verilator emitted warning or diagnostic output")

    versions = []
    for command in (
        [args.iverilog, "-V"],
        [args.verilator, "--version"],
        [sys.executable, "--version"],
    ):
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        require(completed.returncode == 0, f"tool version failed: {' '.join(command)}")
        versions.append(completed.stdout.strip())
    tool_log.write_text("\n".join(versions) + "\n", encoding="utf-8")

    generated_members = sorted(path for path in GENERATED.iterdir() if path.is_file())
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "PASS_FOCUSED_SELECTED_GROUP1_SCALE32_RTL_REFERENCE_AGREEMENT",
        "classification": "candidate_only_no_official_attempt_created",
        "boundary": "layer16_attention_and_down_A8_group1_quantize_aligned_sum_layer17_rmsnorm_to_q",
        "selected_candidate": {
            "source_group_size": 1,
            "reference_rank": candidate_result["selected_reference_rank"],
            "baseline_reference_rank": candidate_result["baseline_reference_rank"],
            "rank_improvement": candidate_result["baseline_reference_rank"]
            - candidate_result["selected_reference_rank"],
            "top_token_id": candidate_result["selected_top_token_id"],
            "first_token_restored": candidate_result["first_token_restored"],
            "selection_reason": candidate_result["selection_reason"],
        },
        "claim_boundary": (
            "Exact five-group CPU rank selection plus focused synthesizable RTL/reference "
            "agreement; no official generation, shell admission, U280/Stage 2, benchmark, "
            "synthesis, timing, area, power, or product-quality claim."
        ),
        "numeric_contract": {
            "carried_sources": "two signed-A8 streams",
            "source_groups": "one lane with one model-derived Scale32 per source sample",
            "scale32": "normalized unsigned Q1.15 significand and signed exponent -24..4",
            "rounding": "signed round-to-nearest ties-to-even",
            "sum": "exact source alignment followed by one common-scale signed-A8 rounding",
            "rmsnorm": "accepted ace2_rmsnorm_core fixed-point semantics",
            "q_projection": "accepted ace2_w4a8_proj_core W4A8 semantics with exact Scale32 metadata",
        },
        "checks": {
            "candidate_sha256_unchanged": True,
            "all_five_search_groups_preserved": True,
            "selection_strictly_improves_frozen_baseline": True,
            "independent_reference_unittest": True,
            "model_vector_regeneration_byte_exact": True,
            "iverilog_warning_clean": True,
            "verilator_warning_clean": True,
            "cycle_accurate_model_vector_match": True,
            "signed_ties_to_even": True,
            "positive_and_negative_saturation": True,
            "source_backpressure": True,
            "gain_backpressure": True,
            "weight_backpressure": True,
            "output_backpressure_and_stability": True,
            "asynchronous_reset_recovery": True,
            "synchronous_clear_recovery": True,
            "xz_clean_every_checked_cycle": True,
        },
        "metrics": metrics,
        "bindings": {
            "candidate_result": artifact(CANDIDATE / "result.json"),
            "candidate_sha256s": artifact(CANDIDATE / "SHA256SUMS"),
            "selected_group_result": artifact(SELECTED_GROUP / "result.json"),
            "search_runner": artifact(SEARCH),
            "rtl": artifact(RTL),
            "rmsnorm_rtl": artifact(RMS_RTL),
            "projection_rtl": artifact(PROJ_RTL),
            "testbench": artifact(TB),
            "generator": artifact(GENERATOR),
            "runner": artifact(RUNNER),
            "reference_test": artifact(UNIT_TEST),
            "generated_vectors": [artifact(path) for path in generated_members],
            "logs": {
                "candidate_sha256_check": artifact(candidate_log),
                "reference_unittest": artifact(unit_log),
                "vector_regeneration_check": artifact(vector_log),
                "iverilog_compile": artifact(compile_log),
                "simulation": artifact(simulation_log),
                "verilator_lint": artifact(lint_log),
                "tool_versions": artifact(tool_log),
            },
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "u280_or_stage2_entered": False,
        },
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    members = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in members
            if path.name != "SHA256SUMS"
        ),
        encoding="ascii",
    )
    print(
        "ACE2_LAYER16_GROUP1_SCALE32_RTL_RESULT "
        f"status={result['status']} rank={result['selected_candidate']['reference_rank']} "
        f"cycles={metrics['cycles']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
