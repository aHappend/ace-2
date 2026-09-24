#!/usr/bin/env python3
"""Run focused RTL verification for grouped layer-17 RMSNorm output."""

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
    / "evidence/candidates/w4a8-layer17-rmsnorm-output-source-grouped-repair-v1/"
    "candidate-0001"
)
GENERATED = ROOT / "verification/generated/ace2_layer17_rmsnorm_output_grouped_scale32"
RTL_DEP = ROOT / "rtl/ace2_rmsnorm_core.sv"
RTL = ROOT / "rtl/ace2_layer17_rmsnorm_output_grouped_scale32_core.sv"
TB = ROOT / "verification/tb/ace2_layer17_rmsnorm_output_grouped_scale32_tb.sv"
SEARCH = ROOT / "tools/ace2_checkpoint176_layer17_rmsnorm_output_grouped_scale32_search.py"
GENERATOR = ROOT / "tools/gen_layer17_rmsnorm_output_grouped_scale32_vectors.py"
UNIT_TEST = ROOT / "verification/test_layer17_rmsnorm_output_grouped_scale32_reference.py"
RUNNER = Path(__file__).resolve()
RTL_TOP = "ace2_layer17_rmsnorm_output_grouped_scale32_core"


PASS_PATTERN = re.compile(
    r"ACE2_LAYER17_RMSNORM_OUTPUT_GROUPED_SCALE32_RTL_PASS "
    r"samples=(?P<samples>\d+) beats=(?P<beats>\d+) groups=(?P<groups>\d+) "
    r"input_stalls=(?P<input_stalls>\d+) output_stalls=(?P<output_stalls>\d+) "
    r"cycles=(?P<cycles>\d+) xz_clean=1 reset_recovery=1 clear_recovery=1 "
    r"backpressure_stable=1"
)


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


def run(command: list[str], log: Path, *, cwd: Path = ROOT) -> str:
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
    group_size = int(candidate_result["rtl_validation_rms_output_group_size"])
    selected_group = CANDIDATE / f"groups/group-{group_size:03d}"

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
            "verification.test_layer17_rmsnorm_output_grouped_scale32_reference",
            "-v",
        ],
        unit_log,
    )
    run([sys.executable, str(GENERATOR), "--check"], vector_log)

    with tempfile.TemporaryDirectory(
        prefix="ace2-layer17-rmsnorm-output-grouped-", dir=ROOT / "build"
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
                "ace2_layer17_rmsnorm_output_grouped_scale32_tb",
                "-o",
                str(executable),
                str(RTL_DEP),
                str(RTL),
                str(TB),
            ],
            compile_log,
        )
        require(not compile_stdout.strip(), "Icarus emitted warning or diagnostic output")
        simulation_stdout = run([args.vvp, str(executable)], simulation_log)

    manifest = json.loads((GENERATED / "manifest.json").read_text(encoding="utf-8"))
    match = PASS_PATTERN.search(simulation_stdout)
    require(match is not None, "focused RTL PASS marker is missing")
    metrics = {name: int(value) for name, value in match.groupdict().items()}
    require(metrics["samples"] == 896, "model sample count differs")
    require(metrics["beats"] == 56, "model beat count differs")
    require(metrics["groups"] == manifest["shape"]["groups"], "group count differs")

    lint_stdout = run(
        [
            args.verilator,
            "--lint-only",
            "-Wall",
            "-Wno-fatal",
            "--top-module",
            RTL_TOP,
            str(RTL_DEP),
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
        "status": "PASS_FOCUSED_LAYER17_RMSNORM_OUTPUT_GROUPED_SCALE32_RTL_REFERENCE_AGREEMENT",
        "classification": "candidate_only_no_official_attempt_created",
        "boundary": "model.layers.17.input_rmsnorm.output_to_self_attn.q_proj.input",
        "validation_candidate": {
            "rms_output_group_size": group_size,
            "selected_as_repair": candidate_result["selected_rms_output_group_size"] == group_size,
            "reference_rank": candidate_result["best_observed_reference_rank"],
            "baseline_reference_rank": candidate_result["baseline_reference_rank"],
            "top_token_id": candidate_result["best_observed_top_token_id"],
            "first_token_restored": candidate_result["first_token_restored"],
            "selection_reason": candidate_result["selection_reason"],
        },
        "claim_boundary": (
            "Frozen candidate outcome plus focused synthesizable RTL/reference agreement "
            "only; no independent acceptance, official generation, U280/Stage 2, "
            "benchmark, synthesis, timing, area, power, or product-quality claim."
        ),
        "numeric_contract": {
            "weights": "unchanged signed W4",
            "carried_activation_samples": "signed A8",
            "group_scales": "explicit model-derived Scale32",
            "gain_metadata": "signed Q7.8 per channel",
            "rounding": "signed round-to-nearest ties-to-even",
            "saturation": "signed int8 clamp",
        },
        "checks": {
            "candidate_sha256_unchanged": True,
            "all_frozen_search_groups_preserved": True,
            "independent_reference_unittest": True,
            "model_vector_regeneration_byte_exact": True,
            "iverilog_warning_clean": True,
            "verilator_warning_clean": True,
            "cycle_accurate_model_vector_match": True,
            "grouped_data_scale_correspondence": True,
            "input_backpressure": True,
            "output_backpressure_and_stability": True,
            "asynchronous_reset_recovery": True,
            "synchronous_clear_recovery": True,
            "xz_clean_every_checked_cycle": True,
        },
        "metrics": metrics,
        "bindings": {
            "candidate_result": artifact(CANDIDATE / "result.json"),
            "candidate_sha256s": artifact(CANDIDATE / "SHA256SUMS"),
            "selected_group_result": artifact(selected_group / "result.json"),
            "search_runner": artifact(SEARCH),
            "rtl_dependency": artifact(RTL_DEP),
            "rtl": artifact(RTL),
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
            "independent_review_still_required": True,
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
        "ACE2_LAYER17_RMSNORM_OUTPUT_GROUPED_SCALE32_RTL_RESULT "
        f"status={result['status']} group={group_size} "
        f"rank={result['validation_candidate']['reference_rank']} cycles={metrics['cycles']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
