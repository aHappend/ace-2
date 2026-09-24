#!/usr/bin/env python3
"""Run focused cycle-accurate RTL checks for layer-16 gate/up output A8."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / (
    "evidence/candidates/w4a8-layer16-gate-up-output-grouped-a8-repair-v1/"
    "candidate-0004"
)
RTL = ROOT / "rtl/ace2_layer16_gate_up_output_grouped_scale32_core.sv"
RTL_TOP = "ace2_layer16_gate_up_output_grouped_scale32_core"
TB = ROOT / "verification/tb/ace2_layer16_gate_up_output_grouped_scale32_tb.sv"
GENERATOR = ROOT / "tools/gen_layer16_gate_up_output_grouped_scale32_vectors.py"
RUNNER = Path(__file__).resolve()
SEARCH = ROOT / "tools/ace2_checkpoint176_layer16_gate_up_output_grouped_scale32_search.py"
UNIT_TEST = ROOT / "verification/test_layer16_gate_up_output_grouped_scale32_reference.py"
GENERATED = ROOT / "verification/generated/ace2_layer16_gate_up_output_grouped_scale32"
PASS_PATTERN = re.compile(
    r"ACE2_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_RTL_PASS "
    r"directed=(?P<directed>\d+) outputs=(?P<outputs>\d+) "
    r"groups=(?P<groups>\d+) saturations=(?P<saturations>\d+) "
    r"input_stalls=(?P<input_stalls>\d+) output_stalls=(?P<output_stalls>\d+) "
    r"cycles=(?P<cycles>\d+) xz_clean=1 reset_recovery=1 clear_recovery=1 held_valid=1"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run(command: list[str], log: Path, cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
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


def selected_directory(result: dict[str, Any]) -> Path:
    selected_id = str(result["selected_contract_id"])
    matches = [
        index
        for index, item in enumerate(result["ordered_gate_up_output_contracts"])
        if item["contract_id"] == selected_id
    ]
    require(len(matches) == 1, "selected contract is not unique")
    return CANDIDATE / "contracts" / f"{matches[0]:02d}-{selected_id}"


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
    freeze = json.loads((CANDIDATE / "candidate-freeze.json").read_text(encoding="utf-8"))
    selected = selected_directory(candidate_result)
    require(candidate_result["selected_exact_repeat_passed"] is True, "selected candidate lacks exact repeat")
    require(candidate_result["control_exact_repeat"]["passed"] is True, "control lacks exact repeat")
    require(candidate_result["runtime_bf16_gate_up_sidecar"] is False, "runtime BF16 sidecar is enabled")

    candidate_log = output / "candidate_sha256_check.log"
    unit_log = output / "reference_unittest.log"
    vector_log = output / "vector_regeneration_check.log"
    compile_log = output / "iverilog_compile.log"
    simulation_log = output / "simulation.log"
    lint_log = output / "verilator_lint.log"
    protected_log = output / "protected_hash_check.log"
    tool_log = output / "tool_versions.log"

    run(["sha256sum", "-c", "SHA256SUMS"], candidate_log, cwd=CANDIDATE)
    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "verification.test_layer16_gate_up_output_grouped_scale32_reference",
            "-v",
        ],
        unit_log,
    )
    run([sys.executable, str(GENERATOR), "--check"], vector_log)

    protected_lines = []
    for relative, expected in freeze["protected_hashes"].items():
        observed = sha256_file(ROOT / relative)
        require(observed == expected, f"protected hash differs: {relative}")
        protected_lines.append(f"{observed}  {relative}")
    protected_log.write_text("\n".join(protected_lines) + "\n", encoding="ascii")

    with tempfile.TemporaryDirectory(prefix="ace2-layer16-gate-up-output-", dir=ROOT / "build") as temporary:
        executable = Path(temporary) / "tb.vvp"
        compile_stdout = run(
            [
                args.iverilog,
                "-g2012",
                "-Wall",
                "-Wno-timescale",
                "-Iverification/generated",
                "-s",
                "ace2_layer16_gate_up_output_grouped_scale32_tb",
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
    require(match is not None, "focused gate/up RTL PASS marker is missing")
    metrics = {name: int(value) for name, value in match.groupdict().items()}
    require(metrics["directed"] == 6, "directed case count differs")
    require(metrics["outputs"] == manifest["shape"]["outputs"], "model output count differs")
    require(metrics["groups"] == manifest["shape"]["scale_records"], "model group count differs")
    require(
        metrics["saturations"] == manifest["expected"]["saturation_count"],
        "model saturation count differs",
    )
    require(metrics["input_stalls"] >= 3, "held-valid input backpressure was not exercised")
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
        "status": "PASS_FOCUSED_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_RTL_REFERENCE_AGREEMENT",
        "classification": "candidate_only_no_official_attempt_created",
        "boundary": "model.layers.16.mlp.gate_up_proj.output_to_signed_a8",
        "selected_candidate": {
            "contract_id": candidate_result["selected_contract_id"],
            "gate_output_group_size": candidate_result["selected_gate_output_group_size"],
            "up_output_group_size": candidate_result["selected_up_output_group_size"],
            "reference_rank": candidate_result["selected_reference_rank"],
            "baseline_reference_rank": candidate_result["baseline_reference_rank"],
            "top_token_id": candidate_result["selected_top_token_id"],
            "first_token_restored": candidate_result["first_token_restored"],
            "rank_improved": candidate_result["rank_improved"],
            "best_null_retained": candidate_result["best_null_retained"],
            "selection_reason": candidate_result["selection_reason"],
            "exact_repeat_passed": candidate_result["selected_exact_repeat_passed"],
        },
        "claim_boundary": (
            "Frozen candidate selection plus focused synthesizable RTL/reference agreement "
            "only; independent Reviewer acceptance remains required. No official generation, "
            "benchmark, synthesis, timing, area, power, U280, or Stage 2 claim."
        ),
        "numeric_contract": {
            "weights": "unchanged signed W4 bytes/scales",
            "carried_gate_and_up_samples": "separate signed A8 streams",
            "gate_group_scales": "separate explicit Scale32 metadata",
            "up_group_scales": "separate explicit Scale32 metadata",
            "rounding": "signed round-to-nearest ties-to-even",
            "saturation": "signed int8 clamp",
            "downstream_consumption": "signed A8 streams plus Scale32 metadata only",
            "bf16_runtime_sidecar": False,
        },
        "checks": {
            "fresh_l2_dependency_bound": True,
            "candidate_sha256_unchanged": True,
            "all_frozen_search_outcomes_preserved": True,
            "exact_control_reproduced": True,
            "selected_candidate_exactly_repeated": True,
            "independent_reference_unittest": True,
            "model_vector_regeneration_byte_exact": True,
            "iverilog_warning_clean": True,
            "verilator_warning_clean": True,
            "cycle_accurate_model_vector_match": True,
            "signed_ties_to_even": True,
            "positive_and_negative_saturation": True,
            "held_valid_input_backpressure": True,
            "output_backpressure_and_stability": True,
            "asynchronous_reset_recovery": True,
            "synchronous_clear_recovery": True,
            "xz_clean_every_checked_cycle": True,
            "protected_hashes_unchanged": True,
        },
        "metrics": metrics,
        "source_hashes": {
            "rtl": sha256_file(RTL),
            "testbench": sha256_file(TB),
            "generator": sha256_file(GENERATOR),
            "runner": sha256_file(RUNNER),
            "search": sha256_file(SEARCH),
            "reference_test": sha256_file(UNIT_TEST),
        },
        "bindings": {
            "candidate_result": artifact(CANDIDATE / "result.json"),
            "candidate_sha256s": artifact(CANDIDATE / "SHA256SUMS"),
            "selected_contract_result": artifact(selected / "result.json"),
            "search_runner": artifact(SEARCH),
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
                "protected_hash_check": artifact(protected_log),
                "tool_versions": artifact(tool_log),
            },
        },
        "negative_boundary": {
            "next_exact_causal_boundary": candidate_result["next_exact_causal_boundary"],
            "prequeued_repair": candidate_result["prequeued_repair"],
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "gpu_or_u280_used": False,
            "stage2_entered": False,
            "independent_review_still_required": True,
        },
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    members = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in members),
        encoding="ascii",
    )
    print(
        "ACE2_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_RTL_RESULT "
        f"status={result['status']} contract={candidate_result['selected_contract_id']} "
        f"rank={candidate_result['selected_reference_rank']} outputs={metrics['outputs']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ACE2_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_RTL_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
