#!/usr/bin/env python3
"""Fresh-L2 comparison and focused RTL for RMSNorm-to-gate/up integration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARITHMETIC_RTL = ROOT / (
    "rtl/ace2_layer16_post_attention_rmsnorm_output_grouped_scale32_core.sv"
)
RUNTIME_RTL = ROOT / (
    "rtl/ace2_layer16_post_attention_rmsnorm_runtime_metadata_core.sv"
)
TB = ROOT / (
    "verification/tb/ace2_layer16_post_attention_rmsnorm_runtime_metadata_tb.sv"
)
GATE_W4 = ROOT / (
    "evidence/verification/lora-v4-two-token-24layer-rtl-v1/attempt-0003/"
    "layers/layer-16/tensors/shared_gate_qweight_s4_in_s8.bin"
)
UP_W4 = ROOT / (
    "evidence/verification/lora-v4-two-token-24layer-rtl-v1/attempt-0003/"
    "layers/layer-16/tensors/shared_up_qweight_s4_in_s8.bin"
)
SOURCE_CANDIDATE = ROOT / (
    "evidence/candidates/w4a8-layer16-gate-up-output-grouped-a8-repair-v1/"
    "candidate-0004"
)
RUNTIME_R3_RESULT = ROOT / (
    "build/layer16-post-attention-rmsnorm-runtime-metadata-focused-r3/result.json"
)
FRESH_RUNTIME_R3_RESULT = ROOT / (
    "build/l2-review-layer16-post-attention-rmsnorm-runtime-metadata-focused-r1/"
    "result.json"
)
RUNNER = Path(__file__).resolve()
PASS_PATTERN = re.compile(
    r"ACE2_LAYER16_POST_ATTENTION_RMSNORM_RUNTIME_METADATA_RTL_PASS "
    r"samples=(?P<samples>\d+) positions=(?P<positions>\d+) "
    r"groups=(?P<groups>\d+) nonzero=(?P<nonzero>\d+) "
    r"saturations=(?P<saturations>\d+) input_stalls=(?P<input_stalls>\d+) "
    r"output_stalls=(?P<output_stalls>\d+) cycles=(?P<cycles>\d+) "
    r"xz_clean=1 reset_recovery=1 clear_recovery=1 invalid_scale_warning=1 "
    r"backpressure_stable=1 metadata_live_derived=1"
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


def _contract_dir(base: Path, index: int, contract_id: str) -> Path:
    return base / "contracts" / f"{index:02d}-{contract_id}"


def _compare_fresh_l2(engineer: dict[str, Any], fresh: dict[str, Any]) -> None:
    fields = (
        "status",
        "selected_contract_id",
        "selected_reference_rank",
        "selected_top_token_id",
        "focused_rtl_contract_id",
        "focused_rtl_output_group_size",
        "group_128_reference_rank",
        "group_128_top_token_id",
    )
    for field in fields:
        require(engineer[field] == fresh[field], f"Fresh-L2 result differs: {field}")
    require(len(engineer["trace"]) == len(fresh["trace"]) == 4, "trace length differs")
    for left, right in zip(engineer["trace"], fresh["trace"], strict=True):
        for field in (
            "contract_id",
            "output_group_size",
            "reference_rank",
            "top_token_id",
            "gate_input_nonzero_count",
            "up_input_nonzero_count",
            "gate_accumulator_nonzero_count",
            "up_accumulator_nonzero_count",
            "gate_output_nonzero_count",
            "up_output_nonzero_count",
            "accepted_chain_observed_hashes",
            "reproducibility_signature",
        ):
            require(left[field] == right[field], f"Fresh-L2 trace differs: {field}")


def _verify_all_positions_and_accumulators(
    engineer_dir: Path, fresh_dir: Path
) -> dict[str, Any]:
    weights = {
        "gate": np.memmap(GATE_W4, dtype=np.int8, mode="r", shape=(4864, 896)),
        "up": np.memmap(UP_W4, dtype=np.int8, mode="r", shape=(4864, 896)),
    }
    summaries: dict[str, Any] = {}
    for index, contract_id in enumerate(("group-896", "group-128", "group-064"), 1):
        contract = _contract_dir(engineer_dir, index, contract_id)
        fresh_contract = _contract_dir(fresh_dir, index, contract_id)
        item: dict[str, Any] = {}
        for stream in ("gate", "up"):
            input_s8 = np.fromfile(contract / f"{stream}_input_s8.bin", dtype=np.int8).reshape(30, 896)
            aligned = np.fromfile(
                contract / f"{stream}_aligned_input_s16.bin", dtype="<i2"
            ).reshape(30, 896)
            accumulator = np.fromfile(
                contract / f"{stream}_accumulator_s32.bin", dtype="<i8"
            ).reshape(30, 4864)
            require(
                (contract / f"{stream}_input_s8.bin").read_bytes()
                == (fresh_contract / f"{stream}_input_s8.bin").read_bytes(),
                f"Fresh-L2 {contract_id} {stream} input bytes differ",
            )
            require(
                (contract / f"{stream}_accumulator_s32.bin").read_bytes()
                == (fresh_contract / f"{stream}_accumulator_s32.bin").read_bytes(),
                f"Fresh-L2 {contract_id} {stream} accumulator bytes differ",
            )
            input_nz = np.count_nonzero(input_s8, axis=1)
            accumulator_nz = np.count_nonzero(accumulator, axis=1)
            require(bool(np.all(input_nz > 0)), f"{contract_id} {stream} has a zero input frame")
            require(
                bool(np.all(accumulator_nz > 0)),
                f"{contract_id} {stream} has a zero accumulator frame",
            )
            for position in range(30):
                recomputed = np.asarray(weights[stream], dtype=np.int64) @ aligned[position].astype(np.int64)
                require(
                    np.array_equal(recomputed, accumulator[position]),
                    f"{contract_id} {stream} accepted-W4 accumulator differs at position {position}",
                )
            item[stream] = {
                "minimum_input_nonzero_per_position": int(input_nz.min()),
                "minimum_accumulator_nonzero_per_position": int(accumulator_nz.min()),
                "accumulator_min": int(accumulator.min()),
                "accumulator_max": int(accumulator.max()),
                "input_sha256": sha256(contract / f"{stream}_input_s8.bin"),
                "accumulator_sha256": sha256(contract / f"{stream}_accumulator_s32.bin"),
            }
        summaries[contract_id] = item
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineer-dir", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")
    parser.add_argument("--verilator", default="verilator")
    args = parser.parse_args()
    engineer_dir = args.engineer_dir.resolve()
    fresh_dir = args.fresh_dir.resolve()
    output = args.output_dir.resolve()
    require(ROOT.resolve() in engineer_dir.parents, "engineer directory must be repository-relative")
    require(ROOT.resolve() in fresh_dir.parents, "fresh directory must be repository-relative")
    require(ROOT.resolve() in output.parents, "output directory must be repository-relative")
    require(not output.exists(), "output directory already exists")
    output.mkdir(parents=True)

    engineer = json.loads((engineer_dir / "result.json").read_text())
    fresh = json.loads((fresh_dir / "result.json").read_text())
    _compare_fresh_l2(engineer, fresh)
    require(engineer["selected_contract_id"] == "control", "negative selection differs")
    require(engineer["selected_reference_rank"] == 3252, "selected control rank differs")
    require(engineer["selected_top_token_id"] == 12, "selected control top token differs")
    require(engineer["group_128_reference_rank"] == 3771, "group-128 rank differs")
    require(engineer["group_128_top_token_id"] == 12, "group-128 top token differs")

    accumulator_summary = _verify_all_positions_and_accumulators(engineer_dir, fresh_dir)

    contract_id = str(engineer["focused_rtl_contract_id"])
    group_size = int(engineer["focused_rtl_output_group_size"])
    require(contract_id == "group-896" and group_size == 896, "focused best-null differs")
    vectors = engineer_dir / "rtl-vectors" / contract_id
    fresh_vectors = fresh_dir / "rtl-vectors" / contract_id
    require(
        (vectors / "SHA256SUMS").read_bytes() == (fresh_vectors / "SHA256SUMS").read_bytes(),
        "Fresh-L2 focused RTL vectors differ",
    )
    vector_manifest = json.loads((vectors / "manifest.json").read_text())

    compile_log = output / "iverilog_compile.log"
    simulation_log = output / "simulation.log"
    lint_log = output / "verilator_lint.log"
    versions_log = output / "tool_versions.log"
    with tempfile.TemporaryDirectory(prefix="ace2-rmsnorm-gate-up-focused-", dir=ROOT / "build") as temporary:
        stage = Path(temporary)
        staged_vectors = stage / (
            "verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata"
        )
        shutil.copytree(vectors, staged_vectors)
        tb_text = TB.read_text(encoding="utf-8")
        needle = "ace2_layer16_post_attention_rmsnorm_runtime_metadata_core dut ("
        replacement = (
            "ace2_layer16_post_attention_rmsnorm_runtime_metadata_core "
            f"#(.OUTPUT_GROUP_SIZE({group_size})) dut ("
        )
        require(tb_text.count(needle) == 1, "focused TB DUT instantiation differs")
        staged_tb = stage / "focused_tb.sv"
        staged_tb.write_text(tb_text.replace(needle, replacement), encoding="utf-8")
        executable = stage / "tb.vvp"
        compile_stdout = run(
            [
                args.iverilog,
                "-g2012",
                "-Wall",
                "-Wno-timescale",
                f"-I{stage / 'verification/generated'}",
                "-s",
                "ace2_layer16_post_attention_rmsnorm_runtime_metadata_tb",
                "-o",
                str(executable),
                str(ARITHMETIC_RTL),
                str(RUNTIME_RTL),
                str(staged_tb),
            ],
            compile_log,
        )
        require(not compile_stdout.strip(), "Icarus emitted a warning or diagnostic")
        simulation_stdout = run([args.vvp, str(executable)], simulation_log, cwd=stage)

    match = PASS_PATTERN.search(simulation_stdout)
    require(match is not None, "focused RTL PASS marker is missing")
    metrics = {name: int(value) for name, value in match.groupdict().items()}
    require(metrics["samples"] == 30 * 896, "focused sample count differs")
    require(metrics["positions"] == 30, "focused position count differs")
    require(metrics["groups"] == 30, "focused group-896 count differs")
    require(metrics["nonzero"] == vector_manifest["nonzero_outputs"], "focused nonzero count differs")
    require(metrics["saturations"] == 0, "focused model vectors saturated")

    lint_stdout = run(
        [
            args.verilator,
            "--lint-only",
            "-Wall",
            "-Wno-fatal",
            "--top-module",
            "ace2_layer16_post_attention_rmsnorm_runtime_metadata_core",
            f"-GOUTPUT_GROUP_SIZE={group_size}",
            str(ARITHMETIC_RTL),
            str(RUNTIME_RTL),
        ],
        lint_log,
    )
    require(not lint_stdout.strip(), "Verilator emitted a warning or diagnostic")

    versions: list[str] = []
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
    versions_log.write_text("\n".join(versions) + "\n", encoding="utf-8")

    candidate_check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=SOURCE_CANDIDATE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(candidate_check.returncode == 0, candidate_check.stdout)
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "PASS_FRESH_L2_NEGATIVE_RANK_REPRODUCTION_AND_GROUP896_FOCUSED_RTL",
        "classification": "bounded_fresh_l2_and_focused_rtl_no_official_attempt",
        "boundary": "model.layers.16.post_attention_layernorm.output_to_signed_a8",
        "selected_contract_id": "control",
        "selected_reference_rank": 3252,
        "selected_top_token_id": 12,
        "focused_rtl_contract_id": contract_id,
        "focused_rtl_output_group_size": group_size,
        "focused_rtl_reference_rank": engineer["focused_rtl_reference_rank"],
        "focused_rtl_top_token_id": engineer["focused_rtl_top_token_id"],
        "group_128_reference_rank": 3771,
        "group_128_top_token_id": 12,
        "rank_reproduction": [
            {
                "contract_id": item["contract_id"],
                "reference_rank": item["reference_rank"],
                "top_token_id": item["top_token_id"],
            }
            for item in engineer["trace"]
        ],
        "accumulator_summary": accumulator_summary,
        "metrics": metrics,
        "checks": {
            "every_rank_and_top_token_fresh_l2_exact": True,
            "every_reproducibility_signature_fresh_l2_exact": True,
            "all_30_positions_have_nonzero_runtime_inputs": True,
            "all_30_positions_have_nonzero_gate_accumulators": True,
            "all_30_positions_have_nonzero_up_accumulators": True,
            "all_gate_accumulators_independently_recomputed_from_accepted_w4": True,
            "all_up_accumulators_independently_recomputed_from_accepted_w4": True,
            "focused_runtime_outputs_match": True,
            "focused_reset_clear_backpressure_output_stability_xz_clean": True,
            "focused_ties_to_even_and_saturation_paths_checked": True,
            "iverilog_warning_clean": True,
            "verilator_warning_clean": True,
            "accepted_candidate_manifest_unchanged": True,
            "focused_runtime_r3_preserved": (
                sha256(RUNTIME_R3_RESULT)
                == "90e7f61ff6d69d273a9c1b0194e16e063ba4e9e358d4e62d146947d80ca27068"
            ),
            "fresh_runtime_r3_preserved": (
                sha256(FRESH_RUNTIME_R3_RESULT)
                == "ecf7141f62bebd808891b012c8f443fb4977ebd863f22baa474bf40171426dde"
            ),
        },
        "bindings": {
            "engineer_result": artifact(engineer_dir / "result.json"),
            "fresh_l2_result": artifact(fresh_dir / "result.json"),
            "focused_vectors": artifact(vectors / "SHA256SUMS"),
            "gate_w4": artifact(GATE_W4),
            "up_w4": artifact(UP_W4),
            "runtime_rtl": artifact(RUNTIME_RTL),
            "arithmetic_rtl": artifact(ARITHMETIC_RTL),
            "testbench": artifact(TB),
            "runner": artifact(RUNNER),
            "logs": {
                "iverilog_compile": artifact(compile_log),
                "simulation": artifact(simulation_log),
                "verilator_lint": artifact(lint_log),
                "tool_versions": artifact(versions_log),
            },
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "gate_up_w4_mutated": False,
            "runtime_sidecar_used": False,
            "silu_contract_changed": False,
            "u280_or_stage2_entered": False,
        },
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        "ACE2_LAYER16_POST_ATTENTION_RMSNORM_GATE_UP_FOCUSED_RESULT "
        f"status={result['status']} contract={contract_id} samples={metrics['samples']} "
        f"nonzero={metrics['nonzero']} cycles={metrics['cycles']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
