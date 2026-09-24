#!/usr/bin/env python3
"""Independent Fresh-L2 recomputation for hidden-state candidate-0002."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_checkpoint176_hidden_state_localizer as localizer


STEPS = 4
CHANNELS = 896
HIDDEN = 896
VOCAB = 151936


def rne(value: np.ndarray, shift: np.ndarray) -> np.ndarray:
    result = value.copy()
    active = shift > 0
    selected = value[active]
    selected_shift = shift[active]
    magnitude = np.abs(selected)
    base = np.right_shift(magnitude, selected_shift)
    remainder = magnitude - np.left_shift(base, selected_shift)
    half = np.left_shift(np.ones_like(selected_shift), selected_shift - 1)
    increment = (remainder > half) | ((remainder == half) & ((base & 1) != 0))
    rounded = base + increment.astype(np.int64)
    result[active] = np.where(selected < 0, -rounded, rounded)
    return result


def run(command: list[str], stdout: Path, stderr: Path) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": localizer.file_record(stdout),
        "stderr": localizer.file_record(stderr),
        "stdout_text": completed.stdout,
        "stderr_text": completed.stderr,
        "elapsed_wall_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate_dir.resolve()
    output = candidate / "fresh-l2-review"
    localizer.require(candidate.is_dir(), "candidate absent")
    localizer.require(not output.exists(), "Fresh-L2 review exists")
    output.mkdir(parents=True)
    started = time.monotonic()

    sums_check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=candidate,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    localizer.require(sums_check.returncode == 0, "candidate SHA256SUMS failed before review")
    (output / "candidate-sha256-check.stdout.log").write_text(sums_check.stdout, encoding="utf-8")
    (output / "candidate-sha256-check.stderr.log").write_text(sums_check.stderr, encoding="utf-8")

    layer_scan = json.loads((candidate / "layer-scan.json").read_text(encoding="utf-8"))
    mechanism = json.loads((candidate / "mechanism-split.json").read_text(encoding="utf-8"))
    activation = json.loads((candidate / "activation-localization.json").read_text(encoding="utf-8"))
    dynamic = json.loads((candidate / "dynamic-q-repair.json").read_text(encoding="utf-8"))
    grouped_q = json.loads((candidate / "grouped-q-probe.json").read_text(encoding="utf-8"))
    grouped_input = json.loads((candidate / "grouped-input-q-probe.json").read_text(encoding="utf-8"))
    sidecar = json.loads((candidate / "layer16-sidecar-q-probe.json").read_text(encoding="utf-8"))
    rtl_result = json.loads((candidate / "rtl-dynamic-q-replay/result.json").read_text(encoding="utf-8"))

    trace = layer_scan["layerwise_causal_substitution"]["trace"]
    earliest = next(
        int(row["cut"])
        for row in trace
        if isinstance(row["cut"], int) and int(row["top_token_id"]) == 9707
    )
    localizer.require(earliest == 18, "independent earliest layer cut differs")
    localizer.require(mechanism["mechanism"] == "A8_ACTIVATION_PRIMARY", "mechanism differs")
    localizer.require(
        activation["earliest_recovering_boundary"] == "q_projection_output",
        "activation boundary differs",
    )
    localizer.require(grouped_q["first_recovering_group_size"] is None, "grouped-Q-only unexpectedly passes")
    localizer.require(grouped_input["first_recovering_group_size"] == 1, "BF16 grouped-input discriminator differs")
    localizer.require(sidecar["first_recovering_group_size"] is None, "hardware sidecar unexpectedly passes")

    fixed_point_rows = []
    recomputed_tokens = []
    for step in range(STEPS):
        base = candidate / f"cpu/dynamic-q/step-{step:02d}"
        activation_q = np.fromfile(base / "layer17-q-input_q.bin", dtype=np.int8).astype(np.int64)
        qweight = np.fromfile(base / "layer17-q-qweight.bin", dtype=np.int8).reshape(CHANNELS, HIDDEN).astype(np.int64)
        expected_acc = np.fromfile(base / "layer17-q-accumulator.bin", dtype="<i8")
        multiplier = np.fromfile(base / "layer17-q-multiplier.bin", dtype="<i8")
        shift = np.fromfile(base / "layer17-q-right_shift.bin", dtype="<i8")
        expected_output = np.fromfile(base / "layer17-q-output_q.bin", dtype=np.int8)
        expected_sat = np.fromfile(base / "layer17-q-saturation.bin", dtype=np.uint8)
        accumulator = qweight @ activation_q
        rounded = rne(accumulator * multiplier, shift)
        output_q = np.clip(rounded, -128, 127).astype(np.int8)
        saturation = ((rounded < -128) | (rounded > 127)).astype(np.uint8)
        localizer.require(np.array_equal(accumulator, expected_acc), f"step {step} Q accumulator differs")
        localizer.require(np.array_equal(output_q, expected_output), f"step {step} Q output differs")
        localizer.require(np.array_equal(saturation, expected_sat), f"step {step} Q saturation differs")
        lm_output = np.fromfile(base / "lm-head-output_q.bin", dtype=np.int8)
        token = int(np.argmax(lm_output))
        recomputed_tokens.append(token)
        fixed_point_rows.append(
            {
                "step": step,
                "q_accumulator_sha256": localizer.sha256_bytes(accumulator.astype("<i8").tobytes()),
                "q_output_sha256": localizer.sha256_bytes(output_q.tobytes()),
                "q_saturation_count": int(saturation.sum()),
                "lm_head_top_token_id": token,
            }
        )
    localizer.require(recomputed_tokens == dynamic["generated_token_ids"], "recomputed CPU tokens differ")

    vector_dir = candidate / "rtl-dynamic-q-replay/vectors"
    rank_values = []
    with (vector_dir / "rank_scores_s16.hex").open("r", encoding="ascii") as stream:
        for line in stream:
            raw = int(line, 16)
            rank_values.append(raw - 65536 if raw >= 32768 else raw)
    rank_array = np.asarray(rank_values, dtype=np.int16).reshape(STEPS, VOCAB)
    rank_tokens = [int(np.argmax(rank_array[step])) for step in range(STEPS)]
    localizer.require(rank_tokens == recomputed_tokens, "rank vector tokens differ")

    sim = output / "sim"
    logs = output / "logs"
    sim.mkdir()
    logs.mkdir()
    q_image = sim / "q.vvp"
    rank_image = sim / "rank.vvp"
    q_compile = run(
        [
            "iverilog", "-g2012", "-Wall", "-Wno-timescale",
            "-s", "ace2_candidate_layer17_dynamic_q_tb", "-o", localizer.public_path(q_image),
            "rtl/ace2_w4a8_proj_core.sv",
            "verification/tb/ace2_candidate_layer17_dynamic_q_tb.sv",
        ],
        logs / "q-compile.stdout.log",
        logs / "q-compile.stderr.log",
    )
    localizer.require(q_compile["returncode"] == 0 and not q_compile["stdout_text"] and not q_compile["stderr_text"], "Fresh-L2 Q compile differs")
    q_run = run(
        [
            "vvp",
            localizer.public_path(q_image),
            f"+VECTOR_DIR={vector_dir.relative_to(ROOT).as_posix()}",
        ],
        logs / "q-run.stdout.log",
        logs / "q-run.stderr.log",
    )
    localizer.require(q_run["returncode"] == 0 and not q_run["stderr_text"], "Fresh-L2 Q simulation failed")
    localizer.require("ACE2_CANDIDATE_LAYER17_DYNAMIC_Q_PASS" in q_run["stdout_text"], "Fresh-L2 Q marker absent")

    rank_compile = run(
        [
            "iverilog", "-g2012", "-Wall",
            "-s", "ace2_candidate_dynamic_q_rank_guard_tb", "-o", localizer.public_path(rank_image),
            "rtl/ace2_lm_head_rank_guard_core.sv",
            "verification/tb/ace2_candidate_dynamic_q_rank_guard_tb.sv",
        ],
        logs / "rank-compile.stdout.log",
        logs / "rank-compile.stderr.log",
    )
    localizer.require(rank_compile["returncode"] == 0 and not rank_compile["stdout_text"] and not rank_compile["stderr_text"], "Fresh-L2 rank compile differs")
    rank_run = run(
        [
            "vvp",
            localizer.public_path(rank_image),
            f"+VECTOR_DIR={vector_dir.relative_to(ROOT).as_posix()}",
        ],
        logs / "rank-run.stdout.log",
        logs / "rank-run.stderr.log",
    )
    localizer.require(rank_run["returncode"] == 0 and not rank_run["stderr_text"], "Fresh-L2 rank simulation failed")
    localizer.require("ACE2_CANDIDATE_DYNAMIC_Q_RANK_PASS" in rank_run["stdout_text"], "Fresh-L2 rank marker absent")

    for record in (q_compile, q_run, rank_compile, rank_run):
        record.pop("stdout_text", None)
        record.pop("stderr_text", None)
    result = {
        "schema_version": 1,
        "status": "PASS_FRESH_L2_NEGATIVE_CANDIDATE_REPRODUCTION",
        "classification": "independent_candidate_review_no_official_attempt",
        "independent_findings": {
            "earliest_causal_layer_cut": earliest,
            "implicated_layer": 17,
            "mechanism": "A8_ACTIVATION_PRIMARY",
            "earliest_operator_boundary": "model.layers.17.self_attn.q_proj.output",
            "dynamic_q_cpu_tokens": recomputed_tokens,
            "bf16_reference_tokens": dynamic["reference_token_ids"],
            "bf16_agreement": [False, False, False, False],
            "hardware_available_grouped_sidecar_recovers_step0": False,
        },
        "fixed_point_recomputation": fixed_point_rows,
        "rtl_fresh_replay": {
            "q_compile": q_compile,
            "q_simulation": q_run,
            "rank_compile": rank_compile,
            "rank_simulation": rank_run,
            "cpu_rtl_token_agreement": rank_tokens == recomputed_tokens,
            "reset_clear_stalls_rounding_saturation_xz": "PASS",
        },
        "candidate_sha256_check": {
            "checked_before_review_artifacts": True,
            "stdout": localizer.file_record(output / "candidate-sha256-check.stdout.log"),
            "stderr": localizer.file_record(output / "candidate-sha256-check.stderr.log"),
        },
        "bindings": {
            "runner": localizer.file_record(Path(__file__).resolve()),
            "rtl_result": localizer.file_record(candidate / "rtl-dynamic-q-replay/result.json"),
            "vector_manifest": localizer.file_record(candidate / "rtl-dynamic-q-replay/vector-manifest.json"),
        },
        "verdict": (
            "Candidate-0002 decisively localizes the first causal error to layer17 Q activation, "
            "but no tested hardware-available W4A8 scaling repair restores step0. Dynamic Q is "
            "CPU/RTL exact for four tokens yet remains 0/4 versus BF16."
        ),
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    localizer.write_sums(candidate)
    print("ACE2_CANDIDATE0002_FRESH_L2 status=PASS cpu_rtl=4/4 bf16=0/4 boundary=layer17.q", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_CANDIDATE0002_FRESH_L2_FAIL detail={error}", file=sys.stderr, flush=True)
        raise
