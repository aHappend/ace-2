#!/usr/bin/env python3
"""Run focused cycle-accurate verification for the layer-16 dual-scale residual sum."""

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
    / "evidence/candidates/w4a8-layer16-residual-sum-dual-scale-repair-v1/"
    "candidate-0006"
)
GENERATED = ROOT / "verification/generated/ace2_layer16_residual_sum_dual_scale"
RTL = ROOT / "rtl/ace2_layer16_residual_sum_dual_scale_core.sv"
TB = ROOT / "verification/tb/ace2_layer16_residual_sum_dual_scale_tb.sv"
SEARCH = ROOT / "tools/ace2_checkpoint176_layer16_residual_sum_dual_scale_search.py"
GENERATOR = ROOT / "tools/gen_layer16_residual_sum_dual_scale_vectors.py"
UNIT_TEST = ROOT / "verification/test_layer16_residual_sum_dual_scale_reference.py"
RUNNER = Path(__file__).resolve()
RTL_TOP = "ace2_layer16_residual_sum_dual_scale_core"


PASS_PATTERN = re.compile(
    r"ACE2_LAYER16_RESIDUAL_SUM_DUAL_SCALE_RTL_PASS "
    r"directed=(?P<directed>\d+) outputs=(?P<outputs>\d+) "
    r"groups=(?P<groups>\d+) saturations=(?P<saturations>\d+) "
    r"input_backpressure=(?P<input_backpressure>\d+) "
    r"output_stalls=(?P<output_stalls>\d+) cycles=(?P<cycles>\d+) "
    r"xz_clean=1 reset_recovery=1 clear_recovery=1 stable_backpressure=1"
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
    group_size = int(candidate_result["rtl_validation_output_group_size"])
    selected_group = CANDIDATE / f"contracts/group-{group_size:03d}"
    require(candidate_result["selected_output_group_size"] is None, "candidate is not a null result")
    require(group_size == 32, "best-null RTL validation contract differs")
    require(candidate_result["control"]["reference_rank"] == 31879, "control rank differs")
    require(candidate_result["control"]["top_token_id"] == 37865, "control top token differs")
    require(candidate_result["best_non_control_reference_rank"] == 34409, "best-null rank differs")
    require(candidate_result["next_exact_upstream_scale_alignment_boundary"] is not None, "negative result did not identify one next boundary")

    candidate_log = output / "candidate_sha256_check.log"
    unit_log = output / "reference_unittest.log"
    vector_generation_log = output / "vector_generation.log"
    vector_check_log = output / "vector_regeneration_check.log"
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
            "verification.test_layer16_residual_sum_dual_scale_reference",
            "-v",
        ],
        unit_log,
    )
    if not GENERATED.exists():
        run([sys.executable, str(GENERATOR)], vector_generation_log)
    else:
        vector_generation_log.write_text("existing generated vector directory retained\n", encoding="utf-8")
    run([sys.executable, str(GENERATOR), "--check"], vector_check_log)

    with tempfile.TemporaryDirectory(prefix="ace2-layer16-residual-sum-", dir=ROOT / "build") as temporary:
        executable = Path(temporary) / "tb.vvp"
        compile_stdout = run(
            [
                args.iverilog,
                "-g2012",
                "-Wall",
                "-Wno-timescale",
                "-Iverification/generated",
                "-s",
                "ace2_layer16_residual_sum_dual_scale_tb",
                "-o",
                str(executable),
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
    require(metrics["directed"] == 9, "directed case count differs")
    require(metrics["outputs"] == 896, "model output count differs")
    require(metrics["groups"] == manifest["shape"]["output_groups"], "group count differs")
    require(metrics["saturations"] == manifest["expected"]["saturation_count"], "model saturation count differs")
    require(metrics["input_backpressure"] >= 3, "held-valid input backpressure was not exercised")
    require(metrics["output_stalls"] > 0, "output backpressure was not exercised")

    lint_stdout = run(
        [
            args.verilator,
            "--lint-only",
            "-Wall",
            "-Wno-fatal",
            "--top-module",
            RTL_TOP,
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
        "status": "PASS_FOCUSED_LAYER16_RESIDUAL_SUM_DUAL_SCALE_BEST_NULL_RTL_REFERENCE_AGREEMENT",
        "classification": "candidate_only_no_official_attempt_created",
        "boundary": "model.layers.16.mlp.down_proj.output_to_residual_sum",
        "validation_candidate": {
            "output_group_size": group_size,
            "reference_rank": candidate_result["best_non_control_reference_rank"],
            "baseline_reference_rank": candidate_result["baseline_reference_rank"],
            "top_token_id": candidate_result["best_non_control_top_token_id"],
            "first_token_restored": candidate_result["first_token_restored"],
            "selected_as_repair": False,
            "selection_reason": candidate_result["selection_reason"],
        },
        "next_exact_upstream_scale_alignment_boundary": candidate_result[
            "next_exact_upstream_scale_alignment_boundary"
        ],
        "claim_boundary": (
            "Frozen negative candidate plus focused synthesizable boundary RTL/reference agreement only; "
            "no official generation, independent acceptance, synthesis/OpenSTA, PPA, U280/Stage 2, "
            "benchmark, or product-quality claim."
        ),
        "numeric_contract": {
            "source_samples": "two independently scaled signed-A8 streams",
            "source_scales": "explicit Scale32 residual metadata per channel and down metadata per four channels",
            "output_samples": "signed A8",
            "output_scales": "explicit model-derived Scale32 per 32 outputs",
            "internal_alignment": "two 8x16 signed products, bounded exponent shifts, one signed-65-bit add",
            "rounding": "single signed round-to-nearest ties-to-even after the add",
            "saturation": "single signed-int8 clamp",
            "accepted_w4_bytes_and_scales": "unchanged",
        },
        "checks": {
            "candidate_sha256_unchanged": True,
            "all_frozen_outcomes_preserved": True,
            "independent_reference_unittest": True,
            "model_vector_regeneration_byte_exact": True,
            "cycle_accurate_model_vector_match": True,
            "signed_ties_to_even": True,
            "positive_and_negative_saturation": True,
            "descriptor_error": True,
            "held_valid_input_backpressure": True,
            "same_edge_consume_and_replace": True,
            "output_backpressure_and_stability": True,
            "asynchronous_reset_recovery": True,
            "synchronous_clear_recovery": True,
            "idle_xz_injection": True,
            "iverilog_warning_clean": True,
            "verilator_warning_clean": True,
            "xz_clean_every_checked_cycle": True,
        },
        "metrics": metrics,
        "source_and_constraint_binding": {
            "rtl_sha256": sha256(RTL),
            "testbench_sha256": sha256(TB),
            "timing_constraints": None,
            "reason": "Focused functional RTL mission explicitly excludes synthesis/PPA; canonical SKY130 synthesis/OpenSTA remains an acceptance prerequisite outside this bounded increment.",
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "u280_or_stage2_entered": False,
            "independent_review_still_required": True,
        },
        "bindings": {
            "candidate_result": artifact(CANDIDATE / "result.json"),
            "candidate_sha256s": artifact(CANDIDATE / "SHA256SUMS"),
            "validation_contract_result": artifact(selected_group / "result.json"),
            "search_runner": artifact(SEARCH),
            "generator": artifact(GENERATOR),
            "reference_test": artifact(UNIT_TEST),
            "rtl": artifact(RTL),
            "testbench": artifact(TB),
            "runner": artifact(RUNNER),
            "generated_vectors": [artifact(path) for path in generated_members],
            "logs": {
                "candidate_sha256_check": artifact(candidate_log),
                "reference_unittest": artifact(unit_log),
                "vector_generation": artifact(vector_generation_log),
                "vector_regeneration_check": artifact(vector_check_log),
                "iverilog_compile": artifact(compile_log),
                "simulation": artifact(simulation_log),
                "verilator_lint": artifact(lint_log),
                "tool_versions": artifact(tool_log),
            },
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    members = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in members),
        encoding="ascii",
    )
    print(
        "ACE2_LAYER16_RESIDUAL_SUM_DUAL_SCALE_VERIFY_PASS "
        f"group={group_size} rank={candidate_result['best_non_control_reference_rank']} "
        f"outputs={metrics['outputs']} xz_clean=1 warning_clean=1"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ACE2_LAYER16_RESIDUAL_SUM_DUAL_SCALE_VERIFY_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
