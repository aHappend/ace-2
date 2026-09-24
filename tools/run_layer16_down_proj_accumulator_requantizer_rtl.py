#!/usr/bin/env python3
"""Run focused RTL verification for layer-16 down_proj accumulator requantization."""

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
    / "evidence/candidates/w4a8-layer16-down-proj-accumulator-requantization-repair-v1/"
    "candidate-0002"
)
GENERATED = ROOT / "verification/generated/ace2_layer16_down_proj_accumulator_requantizer"
RTL = ROOT / "rtl/ace2_layer16_down_proj_accumulator_requantizer_core.sv"
TB = ROOT / "verification/tb/ace2_layer16_down_proj_accumulator_requantizer_tb.sv"
SEARCH = ROOT / "tools/ace2_checkpoint176_layer16_down_proj_accumulator_grouped_scale32_search.py"
GENERATOR = ROOT / "tools/gen_layer16_down_proj_accumulator_requantizer_vectors.py"
UNIT_TEST = ROOT / "verification/test_layer16_down_proj_accumulator_requantizer_reference.py"
RUNNER = Path(__file__).resolve()
RTL_TOP = "ace2_layer16_down_proj_accumulator_requantizer_core"
BOUNDARY = "model.layers.16.mlp.down_proj.accumulator_to_output_requantization"


PASS_PATTERN = re.compile(
    r"ACE2_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANT_RTL_PASS "
    r"directed=(?P<directed>\d+) outputs=(?P<outputs>\d+) "
    r"groups=(?P<groups>\d+) saturations=(?P<saturations>\d+) "
    r"input_stalls=(?P<input_stalls>\d+) output_stalls=(?P<output_stalls>\d+) "
    r"cycles=(?P<cycles>\d+) xz_clean=1 reset_recovery=1 clear_recovery=1 "
    r"idle_xz_injection=1"
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
    freeze = json.loads((CANDIDATE / "candidate-freeze.json").read_text(encoding="utf-8"))
    group_size = int(candidate_result["rtl_validation_down_output_group_size"])
    selected_group = CANDIDATE / f"groups/group-{group_size:03d}"
    selected_as_repair = candidate_result["selected_down_output_group_size"] is not None
    reference_rank = (
        int(candidate_result["selected_reference_rank"])
        if selected_as_repair
        else int(candidate_result["best_observed_reference_rank"])
    )
    top_token_id = (
        int(candidate_result["selected_top_token_id"])
        if selected_as_repair
        else int(candidate_result["best_observed_top_token_id"])
    )
    require(
        len(candidate_result["trace"]) + len(candidate_result["failures"])
        == len(candidate_result["ordered_down_output_group_sizes"]),
        "frozen candidate group set is incomplete",
    )
    require(not candidate_result["failures"], "a frozen candidate group did not execute")
    if selected_as_repair:
        require(
            reference_rank < int(candidate_result["baseline_reference_rank"])
            or top_token_id == int(freeze["reference_step0_token_id"]),
            "selected candidate does not meet the frozen score policy",
        )
    else:
        require(
            candidate_result["selected_reference_rank"]
            == candidate_result["baseline_reference_rank"],
            "negative candidate changed the accepted baseline selection",
        )

    candidate_log = output / "candidate_sha256_check.log"
    protected_log = output / "protected_hash_check.log"
    unit_log = output / "reference_unittest.log"
    vector_generation_log = output / "vector_generation.log"
    vector_log = output / "vector_regeneration_check.log"
    compile_log = output / "iverilog_compile.log"
    simulation_log = output / "simulation.log"
    lint_log = output / "verilator_lint.log"
    tool_log = output / "tool_versions.log"

    run(["sha256sum", "-c", "SHA256SUMS"], candidate_log, cwd=CANDIDATE)
    protected_lines = []
    for relative, expected in freeze["protected_hashes"].items():
        actual = sha256(ROOT / relative)
        require(actual == expected, f"protected hash differs: {relative}")
        protected_lines.append(f"{actual}  {relative}")
    protected_log.write_text("\n".join(protected_lines) + "\n", encoding="ascii")
    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "verification.test_layer16_down_proj_accumulator_requantizer_reference",
            "-v",
        ],
        unit_log,
    )
    run([sys.executable, str(GENERATOR)], vector_generation_log)
    run([sys.executable, str(GENERATOR), "--check"], vector_log)

    with tempfile.TemporaryDirectory(
        prefix="ace2-layer16-down-acc-requant-", dir=ROOT / "build"
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
                "ace2_layer16_down_proj_accumulator_requantizer_tb",
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
    require(metrics["directed"] == 7, "directed case count differs")
    require(metrics["outputs"] == 896, "model output count differs")
    require(
        metrics["groups"] == manifest["shape"]["down_output_groups"],
        "model group count differs",
    )
    require(
        metrics["saturations"] == manifest["expected"]["down_saturation_count"],
        "model saturation count differs",
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
        "status": "PASS_FOCUSED_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANTIZATION_RTL_REFERENCE_AGREEMENT",
        "classification": "candidate_only_no_official_attempt_created",
        "boundary": BOUNDARY,
        "validation_candidate": {
            "down_output_group_size": group_size,
            "selected_as_repair": selected_as_repair,
            "reference_rank": reference_rank,
            "baseline_reference_rank": candidate_result["baseline_reference_rank"],
            "top_token_id": top_token_id,
            "first_token_restored": candidate_result["first_token_restored"],
            "selection_reason": candidate_result["selection_reason"],
        },
        "next_exact_upstream_boundary": candidate_result["next_exact_upstream_boundary"],
        "claim_boundary": (
            "Frozen candidate plus focused synthesizable accumulator/requantizer RTL/reference "
            "agreement only; independent Fresh-L2 acceptance remains required."
        ),
        "numeric_contract": {
            "weights": "unchanged signed W4",
            "input_samples": "signed A8 plus explicit producer Scale32",
            "accumulator": "internal signed 32-bit MAC result only",
            "output_samples": "signed A8",
            "output_scales": "explicit model-derived Scale32 per frozen output group",
            "rounding": "signed round-to-nearest ties-to-even",
            "saturation": "signed int8 clamp",
        },
        "checks": {
            "candidate_sha256_unchanged": True,
            "all_frozen_search_groups_preserved": True,
            "score_policy_honored": True,
            "protected_hashes_unchanged": True,
            "independent_reference_unittest": True,
            "model_vector_regeneration_byte_exact": True,
            "iverilog_warning_clean": True,
            "verilator_warning_clean": True,
            "cycle_accurate_model_vector_match": True,
            "signed_ties_to_even": True,
            "positive_and_negative_saturation": True,
            "invalid_scale_record_rejected": True,
            "input_backpressure": True,
            "output_backpressure_and_stability": True,
            "asynchronous_reset_recovery": True,
            "synchronous_clear_recovery": True,
            "idle_xz_injection_no_output": True,
            "xz_clean_every_checked_cycle": True,
        },
        "metrics": metrics,
        "bindings": {
            "candidate_result": artifact(CANDIDATE / "result.json"),
            "candidate_sha256s": artifact(CANDIDATE / "SHA256SUMS"),
            "selected_group_result": artifact(selected_group / "result.json"),
            "search_runner": artifact(SEARCH),
            "rtl": artifact(RTL),
            "testbench": artifact(TB),
            "generator": artifact(GENERATOR),
            "runner": artifact(RUNNER),
            "reference_test": artifact(UNIT_TEST),
            "generated_vectors": [artifact(path) for path in generated_members],
            "logs": {
                "candidate_sha256_check": artifact(candidate_log),
                "protected_hash_check": artifact(protected_log),
                "reference_unittest": artifact(unit_log),
                "vector_generation": artifact(vector_generation_log),
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
        "ACE2_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANT_RTL_RESULT "
        f"status={result['status']} group={group_size} rank={reference_rank} "
        f"selected={selected_as_repair} cycles={metrics['cycles']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
