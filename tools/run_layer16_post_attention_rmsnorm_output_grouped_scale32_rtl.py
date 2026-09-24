#!/usr/bin/env python3
"""Run focused RTL checks for the layer-16 post-attention RMSNorm output repair."""

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
SOURCE_CANDIDATE = ROOT / (
    "evidence/candidates/w4a8-layer16-gate-up-output-grouped-a8-repair-v1/"
    "candidate-0004"
)
GENERATED = ROOT / (
    "verification/generated/"
    "ace2_layer16_post_attention_rmsnorm_output_grouped_scale32"
)
RTL = ROOT / (
    "rtl/ace2_layer16_post_attention_rmsnorm_output_grouped_scale32_core.sv"
)
TB = ROOT / (
    "verification/tb/"
    "ace2_layer16_post_attention_rmsnorm_output_grouped_scale32_tb.sv"
)
GENERATOR = ROOT / (
    "tools/gen_layer16_post_attention_rmsnorm_output_grouped_scale32_vectors.py"
)
REFERENCE = ROOT / (
    "tools/ace2_layer16_post_attention_rmsnorm_output_reference.py"
)
UNIT_TEST = ROOT / (
    "verification/"
    "test_layer16_post_attention_rmsnorm_output_grouped_scale32_reference.py"
)
RUNNER = Path(__file__).resolve()
RTL_TOP = "ace2_layer16_post_attention_rmsnorm_output_grouped_scale32_core"


PASS_PATTERN = re.compile(
    r"ACE2_LAYER16_POST_ATTENTION_RMSNORM_OUTPUT_GROUPED_SCALE32_RTL_PASS "
    r"samples=(?P<samples>\d+) positions=(?P<positions>\d+) groups=(?P<groups>\d+) "
    r"nonzero=(?P<nonzero>\d+) saturations=(?P<saturations>\d+) "
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

    candidate_log = output / "candidate_sha256_check.log"
    unit_log = output / "reference_unittest.log"
    vector_log = output / "vector_regeneration_check.log"
    compile_log = output / "iverilog_compile.log"
    simulation_log = output / "simulation.log"
    lint_log = output / "verilator_lint.log"
    tool_log = output / "tool_versions.log"

    run(["sha256sum", "-c", "SHA256SUMS"], candidate_log, cwd=SOURCE_CANDIDATE)
    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "verification.test_layer16_post_attention_rmsnorm_output_grouped_scale32_reference",
            "-v",
        ],
        unit_log,
    )
    run([sys.executable, str(GENERATOR), "--check"], vector_log)

    with tempfile.TemporaryDirectory(
        prefix="ace2-layer16-post-rmsnorm-output-", dir=ROOT / "build"
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
                "ace2_layer16_post_attention_rmsnorm_output_grouped_scale32_tb",
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
    require(metrics["samples"] == 30 * 896, "model sample count differs")
    require(metrics["positions"] == 30, "model position count differs")
    require(metrics["groups"] == 30 * 7, "Scale32 group count differs")
    require(
        metrics["nonzero"] == manifest["observed"]["nonzero_outputs"],
        "nonzero output count differs",
    )

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
        "status": "PASS_FOCUSED_LAYER16_POST_ATTENTION_RMSNORM_OUTPUT_GROUPED_SCALE32_RTL_REFERENCE_AGREEMENT_INCREMENT",
        "classification": "candidate_arithmetic_increment_no_official_attempt",
        "boundary": "model.layers.16.post_attention_layernorm.output_to_signed_a8",
        "executed_contract": manifest["executed_contract"],
        "claim_boundary": (
            "Bounded RMSNorm-output arithmetic and streaming RTL/reference agreement "
            "only; rank selection, gate/up integration, independent acceptance, official "
            "generation, synthesis, timing, PPA, U280, and Stage 2 remain unclaimed."
        ),
        "numeric_contract": manifest["numeric_contract"],
        "checks": {
            "source_candidate_sha256_unchanged": True,
            "control_rank_and_top_token_frozen": True,
            "all_26880_model_outputs_match": True,
            "nonzero_causal_payload_produced": metrics["nonzero"] > 0,
            "q7_8_gain_floor_representable": True,
            "bounded_widened_internal_product": True,
            "ties_to_even_directed": True,
            "signed_a8_saturation_directed": True,
            "model_vector_regeneration_byte_exact": True,
            "iverilog_warning_clean": True,
            "verilator_warning_clean": True,
            "input_backpressure": True,
            "output_backpressure_and_stability": True,
            "asynchronous_reset_recovery": True,
            "synchronous_clear_recovery": True,
            "xz_clean_every_checked_cycle": True,
        },
        "metrics": metrics,
        "bindings": {
            "source_candidate_result": artifact(SOURCE_CANDIDATE / "result.json"),
            "source_candidate_sha256s": artifact(SOURCE_CANDIDATE / "SHA256SUMS"),
            "rtl": artifact(RTL),
            "testbench": artifact(TB),
            "generator": artifact(GENERATOR),
            "reference": artifact(REFERENCE),
            "reference_test": artifact(UNIT_TEST),
            "runner": artifact(RUNNER),
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
            "gate_up_w4_mutated": False,
            "carried_activation_wider_than_a8": False,
            "u280_or_stage2_entered": False,
            "rank_selection_completed": False,
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
        "ACE2_LAYER16_POST_ATTENTION_RMSNORM_OUTPUT_GROUPED_SCALE32_RTL_RESULT "
        f"status={result['status']} samples={metrics['samples']} "
        f"nonzero={metrics['nonzero']} cycles={metrics['cycles']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
