#!/usr/bin/env python3
"""Measure the bounded layer-23 QKV grouped-output-scale candidate."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import peft
import tokenizers
import torch
import transformers
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_checkpoint176_hidden_state_localizer as localizer
from tools import diagnose_stage1_same_prompt_causal as causal
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner
from tools.ace2_quality_contracts import ceil_scale32_from_float, scale32_ratio


MISSION_ID = "stage1layer23qkvscale01"
PROMPT = "Tell me a joke."
TARGET_LAYER = 23
TARGET_TOKEN = 39814
GROUP_SIZE = 64
TOP_K = 8
DIAGNOSIS = ROOT / "diagnosis/stage1samepromptcausal01/result.json"
BF16_LOGITS = (
    ROOT
    / "diagnosis/stage1qualitydiag01/logits/"
    "checkpoint176-bf16-greedy-step-00-f32le.bin"
)
RTL = ROOT / "rtl/ace2_layer17_q_output_grouped_scale32_core.sv"
RTL_TOP = "ace2_layer17_q_output_grouped_scale32_core"
QKV_PATTERN = re.compile(r"layer23_position(?P<position>\d+)_(?P<key>q|k|v)$")
O_PATTERN = re.compile(r"layer23_position(?P<position>\d+)_o$")
SCORE_PATTERN = re.compile(
    r"layer23_position(?P<position>\d+)_head(?P<head>\d+)(?:_zero_cached_k)?$"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(localizer.canonical_bytes(value))
    return localizer.file_record(path)


def scale32_float(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def group_scale32(
    accumulator: torch.Tensor,
    input_scale: float,
    weight_scale: torch.Tensor,
) -> tuple[int, ...]:
    require(accumulator.numel() % GROUP_SIZE == 0, "QKV output group is partial")
    real = accumulator.to(torch.float64).abs() * input_scale * weight_scale
    records = []
    for start in range(0, accumulator.numel(), GROUP_SIZE):
        maximum = float(real[start : start + GROUP_SIZE].max().item())
        records.append(ceil_scale32_from_float(max(maximum / 127.0, 1.0e-12)))
    return tuple(records)


class Layer23QKVGroupedScaleCache(localizer.FastProjectionCache):
    """Replace only layer-23 QKV tensor-wide output scales and their consumers."""

    def __init__(self, weights: Any, adapter: Any) -> None:
        super().__init__(weights, adapter)
        self.enabled = False
        self.scales: dict[str, tuple[int, ...]] = {}
        self.records: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
        self.score_replacements = 0
        self.o_replacements = 0

    def enable(self) -> None:
        self.enabled = True
        self.scales.clear()
        self.records.clear()
        self.score_replacements = 0
        self.o_replacements = 0

    def _record(
        self,
        position: int,
        key: str,
        result: dict[str, Any],
        scales: tuple[int, ...],
    ) -> None:
        self.records.setdefault(position, {})[key] = {
            "accumulator_s32": result["accumulator"].detach().cpu().contiguous(),
            "multiplier_s32": result["multiplier"].detach().cpu().contiguous(),
            "right_shift_u6": result["right_shift"].detach().cpu().contiguous(),
            "output_s8": result["output_q"].detach().cpu().contiguous(),
            "group_scale32": torch.tensor(scales, dtype=torch.int64),
        }

    def _derive_grouped_qkv(
        self,
        name: str,
        position: int,
        key: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        metadata = self.metadata.get(merged.data_ptr())
        require(metadata is not None, f"projection metadata absent for {name}")
        accumulator = torch.mv(
            metadata["qweight"].to(torch.int64), input_q.to(torch.int64)
        )
        require(
            bool(torch.all(accumulator >= -(1 << 31)))
            and bool(torch.all(accumulator < (1 << 31))),
            f"{name} accumulator overflow",
        )
        if key not in self.scales:
            self.scales[key] = group_scale32(
                accumulator, input_scale, metadata["weight_scale"]
            )
        scales = self.scales[key]
        require(
            len(scales) == accumulator.numel() // GROUP_SIZE,
            f"{name} grouped scale count differs",
        )
        expanded_scale = torch.tensor(
            [
                scale32_float(scales[index // GROUP_SIZE])
                for index in range(accumulator.numel())
            ],
            dtype=torch.float64,
        )
        multiplier, right_shift = backend.canonical.derive_multiplier(
            input_scale * metadata["weight_scale"] / expanded_scale
        )
        result = self._fixed(
            name,
            input_q,
            input_scale,
            metadata["qweight"],
            multiplier,
            right_shift,
            scale32_float(scales[0]),
        )
        float_output = torch.mv(merged, float_input.to(torch.float32)).contiguous()
        result.update(
            {
                "weight_scale": metadata["weight_scale"],
                "float_output": float_output,
                "source_hashes": source_hashes,
                "output_group_scale32": torch.tensor(scales, dtype=torch.int64),
                "output_group_size": GROUP_SIZE,
                "tensorwide_output_scale": float(
                    backend.canonical.scale_for(float_output)
                ),
                "layer23_qkv_grouped_scale": True,
            }
        )
        self._record(position, key, result, scales)
        return result

    def _derive_grouped_o_consumer(
        self,
        name: str,
        position: int,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        result = super().derive_projection(
            name, merged, input_q, input_scale, float_input, source_hashes
        )
        require("v" in self.scales, "layer-23 V grouped scales are unavailable")
        v_scales = self.scales["v"]
        require(len(v_scales) == backend.KV_HEADS, "V group count differs from KV heads")
        attention_scales = torch.tensor(
            [
                scale32_float(v_scales[head // (backend.Q_HEADS // backend.KV_HEADS)])
                for head in range(backend.Q_HEADS)
            ],
            dtype=torch.float64,
        )
        metadata = self.metadata.get(merged.data_ptr())
        require(metadata is not None, f"projection metadata absent for {name}")
        grouped_accumulator = (
            metadata["qweight"]
            .to(torch.int64)
            .reshape(backend.HIDDEN, backend.Q_HEADS, GROUP_SIZE)
            * input_q.to(torch.int64).reshape(1, backend.Q_HEADS, GROUP_SIZE)
        ).sum(dim=2, dtype=torch.int64)
        real_output = (
            grouped_accumulator.to(torch.float64) * attention_scales.reshape(1, -1)
        ).sum(dim=1) * metadata["weight_scale"]
        rounded = torch.round(real_output / float(result["output_scale"]))
        result["output_q"] = rounded.clamp(-128, 127).to(torch.int8)
        result["saturation"] = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
        result["grouped_input_accumulator_s32"] = grouped_accumulator
        result["grouped_input_scale32"] = torch.tensor(
            [
                v_scales[head // (backend.Q_HEADS // backend.KV_HEADS)]
                for head in range(backend.Q_HEADS)
            ],
            dtype=torch.int64,
        )
        result["layer23_grouped_v_scale_consumer"] = True
        self.o_replacements += 1
        return result

    def derive_projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        if not self.enabled:
            return super().derive_projection(
                name, merged, input_q, input_scale, float_input, source_hashes
            )
        qkv_match = QKV_PATTERN.fullmatch(name)
        if qkv_match is not None:
            return self._derive_grouped_qkv(
                name,
                int(qkv_match.group("position")),
                qkv_match.group("key"),
                merged,
                input_q,
                input_scale,
                float_input,
                source_hashes,
            )
        o_match = O_PATTERN.fullmatch(name)
        if o_match is not None:
            return self._derive_grouped_o_consumer(
                name,
                int(o_match.group("position")),
                merged,
                input_q,
                input_scale,
                float_input,
                source_hashes,
            )
        return super().derive_projection(
            name, merged, input_q, input_scale, float_input, source_hashes
        )

    def from_fixed_metadata(
        self,
        name: str,
        input_q: torch.Tensor,
        input_scale: float,
        qweight: torch.Tensor,
        multiplier: torch.Tensor,
        right_shift: torch.Tensor,
        output_scale: float,
    ) -> dict[str, Any]:
        result = super().from_fixed_metadata(
            name,
            input_q,
            input_scale,
            qweight,
            multiplier,
            right_shift,
            output_scale,
        )
        if self.enabled:
            match = QKV_PATTERN.fullmatch(name)
            if match is not None:
                key = match.group("key")
                require(key in self.scales, f"{key} grouped scales are unavailable")
                self._record(
                    int(match.group("position")), key, result, self.scales[key]
                )
                result["output_group_scale32"] = torch.tensor(
                    self.scales[key], dtype=torch.int64
                )
                result["output_group_size"] = GROUP_SIZE
                result["layer23_qkv_grouped_scale"] = True
        return result


class Layer23GroupedScoreGuard:
    def __init__(self, cache: Layer23QKVGroupedScaleCache) -> None:
        self.cache = cache
        self.original = backend.canonical.reference_attention_score

    def install(self) -> None:
        def score(case: Any) -> Any:
            match = SCORE_PATTERN.fullmatch(case.name)
            if match is None:
                return self.original(case)
            head = int(match.group("head"))
            kv_head = head // (backend.Q_HEADS // backend.KV_HEADS)
            require(
                "q" in self.cache.scales and "k" in self.cache.scales,
                "layer-23 Q/K grouped scales are unavailable",
            )
            replaced = backend.canonical.AttentionScoreCase(
                case.name,
                case.q_values,
                case.k_values,
                scale32_float(self.cache.scales["q"][head]),
                scale32_float(self.cache.scales["k"][kv_head]),
            )
            self.cache.score_replacements += 1
            return self.original(replaced)

        backend.canonical.reference_attention_score = score

    def restore(self) -> None:
        backend.canonical.reference_attention_score = self.original


def metric_summary(record: dict[str, Any]) -> dict[str, Any]:
    comparison = record["comparison_vs_checkpoint176_bf16"]
    return {
        "target_token_id": TARGET_TOKEN,
        "target_token_rank": int(comparison["reference_top_rank_in_candidate"]),
        "top_token_id": int(record["top_token_id"]),
        "top1_matches_bf16": bool(comparison["top1_match"]),
        "top8_overlap_count": int(comparison["top8_overlap_count"]),
        "jensen_shannon_divergence_nats": float(
            comparison["jensen_shannon_divergence_nats"]
        ),
        "relative_l2": float(comparison["raw_score_error"]["relative_l2"]),
        "logits": record["accepted_w4a8_logits"],
    }


def material_decision(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    rank_limit = max(TOP_K, int(baseline["target_token_rank"]) // 10)
    checks = {
        "exact_target_top1_restoration": candidate["top_token_id"] == TARGET_TOKEN,
        "target_rank_at_least_10x_better": candidate["target_token_rank"] <= rank_limit,
        "top8_overlap_strictly_increases": (
            candidate["top8_overlap_count"] > baseline["top8_overlap_count"]
        ),
        "js_divergence_decreases_by_at_least_0_05_nats": (
            candidate["jensen_shannon_divergence_nats"]
            <= baseline["jensen_shannon_divergence_nats"] - 0.05
        ),
        "relative_l2_strictly_decreases": (
            candidate["relative_l2"] < baseline["relative_l2"]
        ),
    }
    distribution_improvement = all(
        checks[name]
        for name in (
            "target_rank_at_least_10x_better",
            "top8_overlap_strictly_increases",
            "js_divergence_decreases_by_at_least_0_05_nats",
            "relative_l2_strictly_decreases",
        )
    )
    return {
        "rank_limit": rank_limit,
        "checks": checks,
        "distribution_improvement": distribution_improvement,
        "materially_improves": bool(
            checks["exact_target_top1_restoration"] or distribution_improvement
        ),
    }


def write_boundary_artifacts(
    output: Path, cache: Layer23QKVGroupedScaleCache
) -> dict[str, Any]:
    require(
        sorted(cache.records) == list(range(34)),
        "candidate did not retain all 34 layer-23 QKV positions",
    )
    artifacts: dict[str, Any] = {}
    for key, width in (("q", 896), ("k", 128), ("v", 128)):
        require(
            all(key in cache.records[position] for position in range(34)),
            f"candidate did not retain all {key.upper()} outputs",
        )
        key_dir = output / "boundary" / key
        for field in (
            "accumulator_s32",
            "multiplier_s32",
            "right_shift_u6",
            "output_s8",
        ):
            tensor = torch.stack(
                [cache.records[position][key][field] for position in range(34)]
            )
            require(tuple(tensor.shape) == (34, width), f"{key} {field} shape differs")
            artifacts[f"{key}_{field}"] = localizer.write_tensor(
                key_dir / f"{field}.bin", tensor
            )
        artifacts[f"{key}_group_scale32"] = localizer.write_tensor(
            key_dir / "group_scale32.bin",
            cache.records[0][key]["group_scale32"],
        )
    return artifacts


def write_hex(path: Path, values: torch.Tensor, bits: int) -> dict[str, Any]:
    mask = (1 << bits) - 1
    digits = (bits + 3) // 4
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{int(value) & mask:0{digits}x}\n" for value in values.tolist()),
        encoding="ascii",
    )
    return localizer.file_record(path)


def run_focused_rtl(
    output: Path, cache: Layer23QKVGroupedScaleCache
) -> dict[str, Any]:
    rtl_dir = output / "focused-rtl"
    rtl_dir.mkdir(parents=True)
    final_records = [cache.records[33][key] for key in ("q", "k", "v")]
    accumulators = torch.cat([record["accumulator_s32"] for record in final_records])
    multipliers = torch.cat([record["multiplier_s32"] for record in final_records])
    shifts = torch.cat([record["right_shift_u6"] for record in final_records])
    expected = torch.cat([record["output_s8"] for record in final_records])
    scales = torch.cat(
        [
            record["group_scale32"].repeat_interleave(GROUP_SIZE)
            for record in final_records
        ]
    )
    channels = torch.cat(
        [
            torch.arange(record["output_s8"].numel(), dtype=torch.int64)
            for record in final_records
        ]
    )
    lasts = torch.cat(
        [
            torch.nn.functional.pad(
                torch.ones(1, dtype=torch.int64),
                (record["output_s8"].numel() - 1, 0),
            )
            for record in final_records
        ]
    )
    count = int(expected.numel())
    require(count == 1152, "focused QKV RTL vector count differs")
    vectors = {
        "accumulator": write_hex(rtl_dir / "accumulator.hex", accumulators, 32),
        "multiplier": write_hex(rtl_dir / "multiplier.hex", multipliers, 32),
        "shift": write_hex(rtl_dir / "shift.hex", shifts, 6),
        "scale32": write_hex(rtl_dir / "scale32.hex", scales, 32),
        "channel": write_hex(rtl_dir / "channel.hex", channels, 10),
        "last": write_hex(rtl_dir / "last.hex", lasts, 1),
        "expected": write_hex(rtl_dir / "expected.hex", expected, 8),
    }
    testbench = rtl_dir / "ace2_layer23_qkv_grouped_scale32_tb.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
`default_nettype none
module ace2_layer23_qkv_grouped_scale32_tb;
    localparam integer CASES = {count};
    reg clk_i = 1'b0;
    reg rst_ni = 1'b0;
    reg clear_i = 1'b0;
    reg in_valid_i = 1'b0;
    wire in_ready_o;
    reg signed [31:0] accumulator_i;
    reg signed [31:0] multiplier_i;
    reg [5:0] right_shift_i;
    reg [31:0] group_scale32_i;
    reg [9:0] channel_i;
    reg last_i;
    wire out_valid_o;
    reg out_ready_i = 1'b1;
    wire signed [7:0] q_o;
    wire [31:0] group_scale32_o;
    wire [9:0] channel_o;
    wire last_o;
    wire saturation_o;
    reg signed [31:0] accumulator_mem [0:CASES-1];
    reg signed [31:0] multiplier_mem [0:CASES-1];
    reg [5:0] shift_mem [0:CASES-1];
    reg [31:0] scale_mem [0:CASES-1];
    reg [9:0] channel_mem [0:CASES-1];
    reg last_mem [0:CASES-1];
    reg signed [7:0] expected_mem [0:CASES-1];
    integer index;
    integer saturations;
    always #5 clk_i = ~clk_i;
    {RTL_TOP} dut (.*);
    initial begin
        $readmemh("{(rtl_dir / 'accumulator.hex').as_posix()}", accumulator_mem);
        $readmemh("{(rtl_dir / 'multiplier.hex').as_posix()}", multiplier_mem);
        $readmemh("{(rtl_dir / 'shift.hex').as_posix()}", shift_mem);
        $readmemh("{(rtl_dir / 'scale32.hex').as_posix()}", scale_mem);
        $readmemh("{(rtl_dir / 'channel.hex').as_posix()}", channel_mem);
        $readmemh("{(rtl_dir / 'last.hex').as_posix()}", last_mem);
        $readmemh("{(rtl_dir / 'expected.hex').as_posix()}", expected_mem);
        repeat (2) @(posedge clk_i);
        @(negedge clk_i); rst_ni = 1'b1; clear_i = 1'b1;
        @(posedge clk_i); #1;
        if (out_valid_o !== 1'b0) $fatal(1, "clear did not empty output");
        @(negedge clk_i); clear_i = 1'b0;
        saturations = 0;
        for (index = 0; index < CASES; index = index + 1) begin
            in_valid_i = 1'b1;
            accumulator_i = accumulator_mem[index];
            multiplier_i = multiplier_mem[index];
            right_shift_i = shift_mem[index];
            group_scale32_i = scale_mem[index];
            channel_i = channel_mem[index];
            last_i = last_mem[index];
            @(posedge clk_i); #1;
            if (out_valid_o !== 1'b1 || q_o !== expected_mem[index] ||
                group_scale32_o !== scale_mem[index] ||
                channel_o !== channel_mem[index] || last_o !== last_mem[index])
                $fatal(1, "mismatch at case %0d", index);
            saturations = saturations + saturation_o;
            @(negedge clk_i);
        end
        in_valid_i = 1'b0;
        $display("ACE2_LAYER23_QKV_GROUPED_SCALE32_RTL_PASS cases=%0d saturations=%0d", CASES, saturations);
        $finish;
    end
endmodule
`default_nettype wire
""",
        encoding="ascii",
    )
    compile_log = rtl_dir / "iverilog-compile.log"
    simulation_log = rtl_dir / "simulation.log"
    version_log = rtl_dir / "tool-versions.log"
    with tempfile.TemporaryDirectory(prefix="ace2-layer23-qkv-", dir=ROOT / "build") as tmp:
        executable = Path(tmp) / "tb.vvp"
        compile_command = [
            "iverilog",
            "-g2012",
            "-Wall",
            "-Wno-timescale",
            "-s",
            "ace2_layer23_qkv_grouped_scale32_tb",
            "-o",
            str(executable),
            str(RTL),
            str(testbench),
        ]
        completed = subprocess.run(
            compile_command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        compile_log.write_text(completed.stdout, encoding="utf-8")
        require(completed.returncode == 0, "focused RTL compile failed")
        require(not completed.stdout.strip(), "focused RTL compile emitted diagnostics")
        simulated = subprocess.run(
            ["vvp", str(executable)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    simulation_log.write_text(simulated.stdout, encoding="utf-8")
    marker = (
        f"ACE2_LAYER23_QKV_GROUPED_SCALE32_RTL_PASS cases={count} "
        f"saturations={int(sum(int(value) in (-128, 127) for value in expected.tolist()))}"
    )
    require(simulated.returncode == 0, "focused RTL simulation failed")
    require(
        f"ACE2_LAYER23_QKV_GROUPED_SCALE32_RTL_PASS cases={count}" in simulated.stdout,
        "focused RTL PASS marker is absent",
    )
    versions = []
    for command in (["iverilog", "-V"], ["vvp", "-V"]):
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
    version_log.write_text("\n".join(versions) + "\n", encoding="utf-8")
    return {
        "status": "PASS_FOCUSED_LAYER23_QKV_GROUPED_SCALE32_RTL_REFERENCE_AGREEMENT",
        "cases": count,
        "generation_position": 0,
        "prefix_position": 33,
        "marker_prefix": marker.split(" saturations=", 1)[0],
        "public_contract": {
            "top": RTL_TOP,
            "latency": "one registered ready/valid stage",
            "reset": "asynchronous active-low reset; synchronous clear",
            "rtl": localizer.file_record(RTL),
        },
        "vectors": vectors,
        "testbench": localizer.file_record(testbench),
        "compile_log": localizer.file_record(compile_log),
        "simulation_log": localizer.file_record(simulation_log),
        "tool_versions": localizer.file_record(version_log),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(ROOT.resolve() in output.parents, "output must be repository-relative")
    require(not output.exists(), "candidate output already exists")

    diagnosis = read_json(DIAGNOSIS)
    require(
        diagnosis["causal_conclusion"]["narrowest_supported_operator_boundary"]
        == "qkv_projection_outputs",
        "accepted diagnosis boundary changed",
    )
    require(
        diagnosis["layer_cut_scan"]["implicated_layer"] == TARGET_LAYER,
        "accepted diagnosis layer changed",
    )
    require(
        diagnosis["causal_conclusion"]["recovery_kind"] == "exact_token",
        "accepted diagnosis no longer has exact recovery",
    )
    sealed_before = causal.sealed_records()
    diagnosis_before = localizer.file_record(DIAGNOSIS)
    output.mkdir(parents=True)
    causal.OUTPUT = output

    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    token_ids = generation_runner.canonical_chat_token_ids(tokenizer, PROMPT)
    require(len(token_ids) == 34, "frozen prefix length changed")
    require(
        localizer.sha256_bytes(localizer.canonical_bytes(token_ids))
        == causal.TOKEN_IDS_SHA256,
        "frozen prefix token IDs changed",
    )
    localizer.PROMPT = PROMPT
    localizer.TOKEN_IDS = token_ids
    localizer.REFERENCE_TOKEN = TARGET_TOKEN
    bf16_array = np.fromfile(BF16_LOGITS, dtype="<f4")
    require(
        bf16_array.shape == (backend.MODEL_OUTPUT_DOMAIN,),
        "BF16 reference logits shape changed",
    )
    require(
        localizer.sha256_file(BF16_LOGITS) == causal.BF16_LOGITS_SHA256,
        "BF16 reference logits hash changed",
    )
    bf16_logits = torch.from_numpy(bf16_array.copy())

    freeze = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "classification": "bounded_nonofficial_software_candidate",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate_count": 1,
        "candidate_configuration": {
            "layer": TARGET_LAYER,
            "boundary": "qkv_projection_outputs",
            "group_size": GROUP_SIZE,
            "group_semantics": "one Scale32 activation-output scale per complete 64-channel attention head",
            "calibration": "position-0 model W4 accumulator range, held for all 34 frozen prefix positions",
            "weights": "unchanged signed W4",
            "rounding": "signed round-to-nearest ties-to-even",
            "saturation": "signed int8 clamp",
            "consumer_scope": "layer-23 attention score scales and grouped V-scale o-projection input only",
        },
        "prompt": {
            "sha256": causal.PROMPT_SHA256,
            "token_ids_sha256": causal.TOKEN_IDS_SHA256,
            "token_count": len(token_ids),
            "text_persisted": False,
        },
        "generation": {
            "position": 0,
            "sampling": False,
            "target_bf16_token_id": TARGET_TOKEN,
            "selection": "greedy over 151936 outputs; lower token ID wins exact ties",
        },
        "comparison_policy": {
            "reference": "checkpoint-176 BF16 logits",
            "top_k": TOP_K,
            "target_rank": "one-based stable descending rank with lower token ID tie-break",
            "material_improvement": (
                "BF16 token 39814 becomes top-1, OR rank improves by at least 10x to "
                "rank <= max(8, baseline_rank//10), top-8 overlap strictly increases, "
                "JSD decreases by at least 0.05 nats, and relative L2 strictly decreases"
            ),
            "rtl_gate": "run the focused grouped-scale RTL/reference check iff material_improvement is true",
        },
        "rtl_public_contract": {
            "top": RTL_TOP,
            "latency": "one registered ready/valid stage",
            "reset": "asynchronous active-low reset; synchronous clear",
            "ports": [
                "clk_i",
                "rst_ni",
                "clear_i",
                "in_valid_i",
                "in_ready_o",
                "accumulator_i:s32",
                "multiplier_i:s32",
                "right_shift_i:u6",
                "group_scale32_i:u32",
                "channel_i:u10",
                "last_i",
                "out_valid_o",
                "out_ready_i",
                "q_o:s8",
                "group_scale32_o:u32",
                "channel_o:u10",
                "last_o",
                "saturation_o",
            ],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "bindings": {
            "accepted_diagnosis": diagnosis_before,
            "bf16_logits": localizer.file_record(BF16_LOGITS),
            "runner": localizer.file_record(Path(__file__).resolve()),
            "w4a8_backend": localizer.file_record(
                ROOT / "tools/rtl_arbitrary_text_generation_backend.py"
            ),
            "metric_evaluator": localizer.file_record(
                ROOT / "tools/diagnose_stage1_quality_attempt0002.py"
            ),
            "rtl": localizer.file_record(RTL),
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_preflight_launched": False,
            "attempt_0003_created": False,
            "full_rtl_generation_run": False,
            "w4_weights_changed": False,
            "stage2_entered": False,
        },
    }
    freeze_record = write_json(output / "candidate-freeze.json", freeze)

    started = time.monotonic()
    with safe_open(
        backend.canonical.MODEL, framework="pt", device="cpu"
    ) as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        cache = Layer23QKVGroupedScaleCache(weights, adapter)
        cache.install()
        try:
            baseline_record, _ = causal.run_cut_record(
                None,
                "baseline-w4a8",
                [],
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
                bf16_logits,
            )
            expected_baseline = diagnosis["layer_cut_scan"]["baseline"]
            require(
                baseline_record["accepted_w4a8_logits"]["sha256"]
                == expected_baseline["accepted_w4a8_logits"]["sha256"],
                "fresh W4A8 baseline logits differ from accepted diagnosis",
            )
            require(
                baseline_record["top_token_id"] == causal.BASELINE_TOKEN,
                "fresh W4A8 baseline top token changed",
            )

            cache.enable()
            score_guard = Layer23GroupedScoreGuard(cache)
            score_guard.install()
            try:
                candidate_record, _ = causal.run_cut_record(
                    None,
                    "candidate-layer23-qkv-group64",
                    [],
                    weights,
                    adapter,
                    norm_gain,
                    embedding,
                    head,
                    tokenizer,
                    bf16_logits,
                )
            finally:
                score_guard.restore()
        finally:
            cache.restore()

        expected_score_replacements = backend.Q_HEADS * (2 * len(token_ids) - 1)
        require(
            cache.score_replacements == expected_score_replacements,
            "grouped Q/K scales did not reach every layer-23 attention score",
        )
        require(
            cache.o_replacements == len(token_ids),
            "grouped V scales did not reach every layer-23 o-projection input",
        )
        boundary_artifacts = write_boundary_artifacts(output, cache)

    baseline = metric_summary(baseline_record)
    candidate = metric_summary(candidate_record)
    decision = material_decision(baseline, candidate)
    focused_rtl = (
        run_focused_rtl(output, cache)
        if decision["materially_improves"]
        else {
            "status": "NOT_RUN_NONIMPROVING_SOFTWARE_CANDIDATE",
            "authorized": False,
        }
    )
    sealed_after = causal.sealed_records()
    require(sealed_after == sealed_before, "sealed attempt-0002 changed")
    require(
        localizer.file_record(DIAGNOSIS) == diagnosis_before,
        "accepted diagnosis changed",
    )

    status = (
        "CANDIDATE_MATERIALLY_IMPROVES_PENDING_FRESH_L2"
        if decision["materially_improves"]
        else "CANDIDATE_NONIMPROVING_REJECT_PENDING_FRESH_L2_CONFIRMATION"
    )
    result = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "status": status,
        "classification": "candidate_only_no_official_execution",
        "baseline": baseline,
        "candidate": candidate,
        "material_improvement_decision": decision,
        "focused_rtl": focused_rtl,
        "boundary_artifacts": boundary_artifacts,
        "execution": {
            "prompt_token_count": len(token_ids),
            "generation_position": 0,
            "candidate_count": 1,
            "layer23_score_scale_replacements": cache.score_replacements,
            "layer23_o_input_scale_replacements": cache.o_replacements,
            "elapsed_wall_seconds": time.monotonic() - started,
        },
        "failure_taxonomy": (
            None
            if decision["materially_improves"]
            else {
                "class": "candidate_repair_non_improvement",
                "root_cause_hypothesis": (
                    "Per-head QKV output scaling alone does not recover enough of the "
                    "layer-23 projection error identified by exact BF16 substitution."
                ),
                "regression": (
                    "Retain the frozen same-prompt baseline/candidate logit comparison "
                    "as a rejection guard for any future layer-23 repair."
                ),
            }
        ),
        "bindings": {
            "freeze": freeze_record,
            "accepted_diagnosis": diagnosis_before,
            "sealed_attempt_before": sealed_before,
            "sealed_attempt_after": sealed_after,
        },
        "scope_guards": freeze["scope_guards"],
        "independent_review_required": True,
    }
    result_record = write_json(output / "result.json", result)
    review_request = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "review_type": "Fresh-L2",
        "status": "PENDING_INDEPENDENT_REVIEW",
        "result": result_record,
        "requested_checks": [
            "Recompute both conditions from the retained float64 logits.",
            "Confirm the single group-64 candidate and predeclared acceptance rule.",
            "Confirm Q, K, and V W4 payloads are unchanged and only grouped output scales propagate.",
            "Confirm RTL ran iff the software material-improvement gate passed.",
            "Confirm sealed attempt-0002 and accepted diagnosis hashes are unchanged.",
        ],
    }
    write_json(output / "fresh-l2-review-request.json", review_request)
    localizer.write_sums(output)
    print(
        "ACE2_STAGE1_LAYER23_QKV_GROUPED_SCALE_RESULT "
        f"status={status} baseline_rank={baseline['target_token_rank']} "
        f"candidate_rank={candidate['target_token_rank']} "
        f"candidate_top={candidate['top_token_id']} "
        f"material={int(decision['materially_improves'])} "
        f"rtl={focused_rtl['status']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ACE2_STAGE1_LAYER23_QKV_GROUPED_SCALE_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
