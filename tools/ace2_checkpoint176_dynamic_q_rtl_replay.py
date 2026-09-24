#!/usr/bin/env python3
"""Build hash-bound vectors and replay the four-position dynamic-Q candidate in RTL."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_checkpoint176_hidden_state_localizer as localizer


STEPS = 4
CHANNELS = 896
HIDDEN = 896
GROUPS = HIDDEN // 4
VOCAB = 151936
PROJ_CORE = ROOT / "rtl/ace2_w4a8_proj_core.sv"
PROJ_TB = ROOT / "verification/tb/ace2_candidate_layer17_dynamic_q_tb.sv"
RANK_CORE = ROOT / "rtl/ace2_lm_head_rank_guard_core.sv"
RANK_TB = ROOT / "verification/tb/ace2_candidate_dynamic_q_rank_guard_tb.sv"


def write_hex(path: Path, values: Iterable[int], width: int) -> dict[str, Any]:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{int(value) & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )
    return localizer.file_record(path)


def pack_lanes(values: np.ndarray, width: int) -> list[int]:
    localizer.require(values.ndim == 1 and values.size % 4 == 0, "lane packing shape differs")
    mask = (1 << width) - 1
    packed = []
    for start in range(0, values.size, 4):
        word = 0
        for lane in range(4):
            word |= (int(values[start + lane]) & mask) << (lane * width)
        packed.append(word)
    return packed


def run_process(command: list[str], log_out: Path, log_err: Path) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_out.parent.mkdir(parents=True, exist_ok=True)
    log_out.write_text(result.stdout, encoding="utf-8")
    log_err.write_text(result.stderr, encoding="utf-8")
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": localizer.file_record(log_out),
        "stderr": localizer.file_record(log_err),
        "elapsed_wall_seconds": time.monotonic() - started,
        "stdout_text": result.stdout,
        "stderr_text": result.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate_dir.resolve()
    output = candidate / "rtl-dynamic-q-replay"
    localizer.require(candidate.is_dir(), "candidate absent")
    localizer.require(not output.exists(), "RTL dynamic-Q replay exists")
    output.mkdir(parents=True)
    dynamic_path = candidate / "dynamic-q-repair.json"
    dynamic = json.loads(dynamic_path.read_text(encoding="utf-8"))
    localizer.require(len(dynamic["steps"]) == STEPS, "dynamic-Q step count differs")
    started = time.monotonic()
    vectors = output / "vectors"
    source_records = []
    activation_words = []
    multiplier_values = []
    shift_values = []
    expected_values = []
    accumulator_values = []
    saturation_values = []
    rank_scores = []
    qweight_reference = None

    for step in range(STEPS):
        base = candidate / f"cpu/dynamic-q/step-{step:02d}"
        paths = {
            "activation": base / "layer17-q-input_q.bin",
            "weight": base / "layer17-q-qweight.bin",
            "multiplier": base / "layer17-q-multiplier.bin",
            "shift": base / "layer17-q-right_shift.bin",
            "expected": base / "layer17-q-output_q.bin",
            "accumulator": base / "layer17-q-accumulator.bin",
            "saturation": base / "layer17-q-saturation.bin",
            "rank": base / "lm-head-output_q.bin",
        }
        source_records.append(
            {"step": step, **{name: localizer.file_record(path) for name, path in paths.items()}}
        )
        activation = np.fromfile(paths["activation"], dtype=np.int8)
        qweight = np.fromfile(paths["weight"], dtype=np.int8).reshape(CHANNELS, HIDDEN)
        multiplier = np.fromfile(paths["multiplier"], dtype="<i8")
        shift = np.fromfile(paths["shift"], dtype="<i8")
        expected = np.fromfile(paths["expected"], dtype=np.int8)
        accumulator = np.fromfile(paths["accumulator"], dtype="<i8")
        saturation = np.fromfile(paths["saturation"], dtype=np.uint8)
        rank = np.fromfile(paths["rank"], dtype=np.int8)
        localizer.require(activation.size == HIDDEN, "activation size differs")
        localizer.require(qweight.shape == (CHANNELS, HIDDEN), "weight shape differs")
        localizer.require(multiplier.size == CHANNELS, "multiplier size differs")
        localizer.require(shift.size == CHANNELS, "shift size differs")
        localizer.require(expected.size == CHANNELS, "expected size differs")
        localizer.require(accumulator.size == CHANNELS, "accumulator size differs")
        localizer.require(saturation.size == CHANNELS, "saturation size differs")
        localizer.require(rank.size == VOCAB, "rank size differs")
        localizer.require(np.all((accumulator >= -(1 << 31)) & (accumulator < (1 << 31))), "accumulator exceeds RTL width")
        localizer.require(np.all((multiplier >= -(1 << 31)) & (multiplier < (1 << 31))), "multiplier exceeds RTL width")
        localizer.require(np.all((shift >= 0) & (shift <= 63)), "shift exceeds RTL width")
        if qweight_reference is None:
            qweight_reference = qweight.copy()
        else:
            localizer.require(np.array_equal(qweight_reference, qweight), "Q weights changed by position")
        activation_words.extend(pack_lanes(activation, 8))
        multiplier_values.extend(int(value) for value in multiplier)
        shift_values.extend(int(value) for value in shift)
        expected_values.extend(int(value) for value in expected)
        accumulator_values.extend(int(value) for value in accumulator)
        saturation_values.extend(int(value) for value in saturation)
        rank_scores.extend(int(value) for value in rank)

    localizer.require(qweight_reference is not None, "Q weight reference absent")
    weight_words = []
    for channel in range(CHANNELS):
        weight_words.extend(pack_lanes(qweight_reference[channel], 4))
    vector_records = {
        "q_activation": write_hex(vectors / "q_activation.hex", activation_words, 32),
        "q_weight": write_hex(vectors / "q_weight.hex", weight_words, 16),
        "q_multiplier": write_hex(vectors / "q_multiplier.hex", multiplier_values, 32),
        "q_shift": write_hex(vectors / "q_shift.hex", shift_values, 8),
        "q_expected": write_hex(vectors / "q_expected.hex", expected_values, 8),
        "q_accumulator": write_hex(vectors / "q_accumulator.hex", accumulator_values, 32),
        "q_saturation": write_hex(vectors / "q_saturation.hex", saturation_values, 8),
        "rank_scores": write_hex(vectors / "rank_scores_s16.hex", rank_scores, 16),
        "rank_expected": write_hex(
            vectors / "rank_expected_u18.hex", dynamic["generated_token_ids"], 18
        ),
    }
    vector_manifest = {
        "schema_version": 1,
        "classification": "candidate_only_hash_bound_vectors",
        "geometry": {
            "steps": STEPS,
            "channels": CHANNELS,
            "hidden": HIDDEN,
            "groups": GROUPS,
            "vocab": VOCAB,
        },
        "source_binary_tensors": source_records,
        "generated_vectors": vector_records,
        "cpu_generated_token_ids": dynamic["generated_token_ids"],
        "bf16_reference_token_ids": dynamic["reference_token_ids"],
    }
    localizer.write_json(output / "vector-manifest.json", vector_manifest)

    sim = output / "sim"
    logs = output / "logs"
    sim.mkdir()
    logs.mkdir()
    proj_image = sim / "ace2_candidate_layer17_dynamic_q_tb.vvp"
    rank_image = sim / "ace2_candidate_dynamic_q_rank_guard_tb.vvp"
    proj_compile = run_process(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-Wno-timescale",
            "-s",
            "ace2_candidate_layer17_dynamic_q_tb",
            "-o",
            localizer.public_path(proj_image),
            localizer.public_path(PROJ_CORE),
            localizer.public_path(PROJ_TB),
        ],
        logs / "q-iverilog.stdout.log",
        logs / "q-iverilog.stderr.log",
    )
    localizer.require(proj_compile["returncode"] == 0, "Q RTL compilation failed")
    localizer.require(not proj_compile["stdout_text"] and not proj_compile["stderr_text"], "Q RTL compilation emitted warnings")
    proj_run = run_process(
        [
            "vvp",
            localizer.public_path(proj_image),
            f"+VECTOR_DIR={vectors.relative_to(ROOT).as_posix()}",
        ],
        logs / "q-vvp.stdout.log",
        logs / "q-vvp.stderr.log",
    )
    localizer.require(proj_run["returncode"] == 0, "Q RTL simulation failed")
    localizer.require(not proj_run["stderr_text"], "Q RTL simulation stderr is nonempty")
    q_marker = "ACE2_CANDIDATE_LAYER17_DYNAMIC_Q_PASS steps=4 channels=896 reset=pass clear=pass stalls=pass rne=pass saturation=pass xz=0"
    localizer.require(q_marker in proj_run["stdout_text"], "Q RTL PASS marker absent")

    rank_compile = run_process(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "ace2_candidate_dynamic_q_rank_guard_tb",
            "-o",
            localizer.public_path(rank_image),
            localizer.public_path(RANK_CORE),
            localizer.public_path(RANK_TB),
        ],
        logs / "rank-iverilog.stdout.log",
        logs / "rank-iverilog.stderr.log",
    )
    localizer.require(rank_compile["returncode"] == 0, "rank RTL compilation failed")
    localizer.require(not rank_compile["stdout_text"] and not rank_compile["stderr_text"], "rank RTL compilation emitted warnings")
    rank_run = run_process(
        [
            "vvp",
            localizer.public_path(rank_image),
            f"+VECTOR_DIR={vectors.relative_to(ROOT).as_posix()}",
        ],
        logs / "rank-vvp.stdout.log",
        logs / "rank-vvp.stderr.log",
    )
    localizer.require(rank_run["returncode"] == 0, "rank RTL simulation failed")
    localizer.require(not rank_run["stderr_text"], "rank RTL simulation stderr is nonempty")
    rank_marker = "ACE2_CANDIDATE_DYNAMIC_Q_RANK_PASS steps=4 reset=pass clear=pass stalls=pass xz=0"
    localizer.require(rank_marker in rank_run["stdout_text"], "rank RTL PASS marker absent")

    for record in (proj_compile, proj_run, rank_compile, rank_run):
        record.pop("stdout_text", None)
        record.pop("stderr_text", None)
    agreement = [
        int(a) == int(b)
        for a, b in zip(
            dynamic["generated_token_ids"], dynamic["reference_token_ids"], strict=True
        )
    ]
    result = {
        "schema_version": 1,
        "status": "PASS_RTL_EXACT_FOR_NEGATIVE_DYNAMIC_Q_CANDIDATE",
        "classification": "candidate_only_no_official_attempt_created",
        "cpu": {
            "dynamic_q_token_ids": dynamic["generated_token_ids"],
            "bf16_reference_token_ids": dynamic["reference_token_ids"],
            "per_step_bf16_agreement": agreement,
            "matching_bf16_greedy_prefix_tokens": 0,
        },
        "rtl": {
            "q_projection": {
                "steps": STEPS,
                "channels_per_step": CHANNELS,
                "scaling_metadata_checked": True,
                "round_to_nearest_even_checked": True,
                "saturation_checked": True,
                "asynchronous_reset_recovery_checked": True,
                "clear_recovery_checked": True,
                "input_and_output_stalls_checked": True,
                "xz_count": 0,
                "compile": proj_compile,
                "simulation": proj_run,
            },
            "rank_selection": {
                "steps": STEPS,
                "selected_token_ids": dynamic["generated_token_ids"],
                "reset_clear_stalls_xz_checked": True,
                "compile": rank_compile,
                "simulation": rank_run,
            },
        },
        "vectors": localizer.file_record(output / "vector-manifest.json"),
        "bindings": {
            "runner": localizer.file_record(Path(__file__).resolve()),
            "dynamic_q_cpu": localizer.file_record(dynamic_path),
            "q_core": localizer.file_record(PROJ_CORE),
            "q_testbench": localizer.file_record(PROJ_TB),
            "rank_core": localizer.file_record(RANK_CORE),
            "rank_testbench": localizer.file_record(RANK_TB),
        },
        "verdict": (
            "RTL exactly reproduces the four-token dynamic-Q CPU candidate and all focused "
            "protocol/numerical checks pass, but the candidate remains 0/4 versus BF16 and "
            "is not a repair."
        ),
        "official_run_launched": False,
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    localizer.write_sums(candidate)
    print(
        "ACE2_DYNAMIC_Q_RTL_RESULT status=PASS cpu_rtl=4/4 bf16=0/4 "
        f"output={localizer.public_path(output)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_DYNAMIC_Q_RTL_FAIL detail={error}", file=sys.stderr, flush=True)
        raise
