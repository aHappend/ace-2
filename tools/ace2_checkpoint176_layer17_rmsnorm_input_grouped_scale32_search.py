#!/usr/bin/env python3
"""Search grouped signed-A8 Scale32 at the layer-16 output/layer-17 RMSNorm input."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_checkpoint176_hidden_state_localizer as localizer
from tools import ace2_checkpoint176_layer16_source_grouped_scale32_search as layer16
from tools import ace2_checkpoint176_layer17_q_output_grouped_scale32_search as qoutput
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner
from tools.ace2_quality_contracts import (
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    round_divide_even_signed,
    scale32_ratio,
    unpack_scale32,
)
from tools.ace2_rmsnorm_reference import derive_scaled_gains_q8


MISSION = "w4a8-layer16-output-to-layer17-rmsnorm-input-repair-v1"
SOURCE_GROUP_SIZE = 1
Q_OUTPUT_GROUP_SIZE = 128
RMS_INPUT_GROUP_SIZES = (32, 16, 8, 4, 1)
PREDECESSOR = (
    ROOT
    / "evidence/candidates/w4a8-layer17-q-output-source-grouped-repair-v1/"
    "candidate-0001"
)
PREDECESSOR_SEARCH_REVIEW = ROOT / "build/l2-review-layer17-q-output-search-r1/result.json"
PREDECESSOR_RTL_REVIEW = ROOT / "build/l2-review-layer17-q-output-rtl-r1/result.json"
ROUTING_RESULT = (
    ROOT
    / "evidence/candidates/w4a8-layer17-rmsnorm-output-source-grouped-repair-v1/"
    "candidate-0001/result.json"
)
ROUTING_SEARCH_REVIEW = ROOT / "build/l2-review-layer17-rmsnorm-output-search-r1/result.json"
RTL = ROOT / "rtl/ace2_layer17_rmsnorm_input_grouped_scale32_core.sv"
RTL_TOP = "ace2_layer17_rmsnorm_input_grouped_scale32_core"


def _scale32_float(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def _sum_reals(
    attention_s8: torch.Tensor,
    down_s8: torch.Tensor,
    attention_scales: tuple[int, ...],
    down_scales: tuple[int, ...],
) -> list[Fraction]:
    values = []
    for index, (attention, down) in enumerate(
        zip(
            attention_s8.to(torch.int64).tolist(),
            down_s8.to(torch.int64).tolist(),
            strict=True,
        )
    ):
        attention_num, attention_den = scale32_ratio(attention_scales[index])
        down_num, down_den = scale32_ratio(down_scales[index])
        values.append(
            Fraction(int(attention) * attention_num, attention_den)
            + Fraction(int(down) * down_num, down_den)
        )
    return values


def _group_scales(values: list[Fraction], group_size: int) -> tuple[int, ...]:
    scales = []
    for start in range(0, len(values), group_size):
        maximum = max(abs(value) for value in values[start : start + group_size])
        scales.append(
            ceil_scale32_from_ratio(maximum.numerator, maximum.denominator * 127)
        )
    return tuple(scales)


def _quantize_fraction(value: Fraction, scale32: int) -> int:
    scale_num, scale_den = scale32_ratio(scale32)
    return round_divide_even_signed(
        value.numerator * scale_den,
        value.denominator * scale_num,
    )


def _quantize_groups(
    values: list[Fraction], scales: tuple[int, ...], group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(len(values), dtype=torch.int8)
    saturation = torch.zeros(len(values), dtype=torch.uint8)
    for index, value in enumerate(values):
        rounded = _quantize_fraction(value, scales[index // group_size])
        clipped = max(-128, min(127, rounded))
        output[index] = clipped
        saturation[index] = int(clipped != rounded)
    return output, saturation


def _grouped_rmsnorm(
    activations: torch.Tensor,
    scales: tuple[int, ...],
    group_size: int,
    gains_q8: list[int],
) -> dict[str, Any]:
    exponents = [unpack_scale32(record)[1] for record in scales]
    common_exponent = min(exponents)
    aligned = []
    for index, activation in enumerate(activations.to(torch.int64).tolist()):
        significand, exponent = unpack_scale32(scales[index // group_size])
        aligned.append(int(activation) * significand * (1 << (exponent - common_exponent)))

    sumsq = sum(value * value for value in aligned)
    mean_square = (sumsq + (backend.HIDDEN // 2)) // backend.HIDDEN
    rms_ceil = max(math.isqrt(mean_square), 1)
    if rms_ceil * rms_ceil < mean_square:
        rms_ceil += 1
    inv_rms_q30 = (1 << 30) // rms_ceil

    outputs = []
    saturation = []
    for value, gain in zip(aligned, gains_q8, strict=True):
        rounded = round_divide_even_signed(value * int(gain) * inv_rms_q30, 1 << 38)
        clipped = max(-128, min(127, rounded))
        outputs.append(clipped)
        saturation.append(int(clipped != rounded))
    return {
        "aligned": torch.tensor(aligned, dtype=torch.int64),
        "outputs": torch.tensor(outputs, dtype=torch.int8),
        "saturation": torch.tensor(saturation, dtype=torch.uint8),
        "sumsq": sumsq,
        "mean_square": mean_square,
        "rms_ceil": rms_ceil,
        "inv_rms_q30": inv_rms_q30,
        "common_exponent": common_exponent,
    }


class GroupedRmsInputSourceGuard:
    """Replace only the layer-16 output representation consumed by layer-17 RMSNorm."""

    def __init__(
        self,
        cache: qoutput.GroupedQOutputScale32Cache,
        weights: Any,
        rms_input_group_size: int,
    ) -> None:
        self.cache = cache
        self.weights = weights
        self.rms_input_group_size = rms_input_group_size
        self.original = backend.derive_layer_token
        self.records: dict[int, dict[str, Any]] = {}

    def install(self) -> None:
        def derive(
            layer_id: int,
            state: dict[str, Any],
            cache: dict[str, Any],
            template: dict[str, Any] | None,
            weights: Any,
            adapter: Any,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            if layer_id == layer16.TARGET_LAYER:
                position_id = int(state["position"])
                localizer.require(
                    position_id in self.records,
                    "grouped RMSNorm-input record missing at layer 17",
                )
                self.cache.position_records[position_id] = self.records[position_id]
                if template is not None:
                    template["qkv"].pop("q", None)

            derived, next_state, next_template = self.original(
                layer_id, state, cache, template, weights, adapter
            )
            if layer_id != layer16.SOURCE_LAYER:
                return derived, next_state, next_template

            position_id = int(state["position"])
            position = derived["positions"][0]
            attention_source_s8 = position["attention_residual"]["output"].contiguous()
            down_source_s8 = position["projections"]["down"]["output_q"].contiguous()
            attention_base_scale = float(position["attention_residual"]["scale"])
            down_base_scale = float(position["projections"]["down"]["output_scale"])

            binary_attention, binary_attention_q, binary_attention_scales = (
                layer16._binary32_grouped_reconstruct(
                    attention_source_s8, attention_base_scale, SOURCE_GROUP_SIZE
                )
            )
            binary_down, binary_down_q, binary_down_scales = (
                layer16._binary32_grouped_reconstruct(
                    down_source_s8, down_base_scale, SOURCE_GROUP_SIZE
                )
            )
            binary_sum = (binary_attention + binary_down).contiguous()
            gain = self.weights.get_tensor(
                f"model.layers.{layer16.TARGET_LAYER}.input_layernorm.weight"
            ).contiguous()
            binary_input_norm = backend.canonical.float_rmsnorm(
                binary_sum.to(torch.float32), gain
            ).contiguous()
            binary_q_input_scale = max(
                float(binary_input_norm.to(torch.float64).abs().max()) / 127.0,
                1.0e-12,
            )
            q_input_scale32 = ceil_scale32_from_float(binary_q_input_scale)

            attention_base_scale32 = ceil_scale32_from_float(attention_base_scale)
            down_base_scale32 = ceil_scale32_from_float(down_base_scale)
            attention_group_scale32 = layer16._exact_group_scales(
                attention_source_s8, attention_base_scale32, SOURCE_GROUP_SIZE
            )
            down_group_scale32 = layer16._exact_group_scales(
                down_source_s8, down_base_scale32, SOURCE_GROUP_SIZE
            )
            attention_group_s8, attention_saturation = layer16._exact_group_requantize(
                attention_source_s8,
                attention_base_scale32,
                attention_group_scale32,
                SOURCE_GROUP_SIZE,
            )
            down_group_s8, down_saturation = layer16._exact_group_requantize(
                down_source_s8,
                down_base_scale32,
                down_group_scale32,
                SOURCE_GROUP_SIZE,
            )

            sum_values = _sum_reals(
                attention_group_s8,
                down_group_s8,
                attention_group_scale32,
                down_group_scale32,
            )
            rms_input_group_scale32 = _group_scales(
                sum_values, self.rms_input_group_size
            )
            rms_input_s8, rms_input_saturation = _quantize_groups(
                sum_values, rms_input_group_scale32, self.rms_input_group_size
            )
            gains = derive_scaled_gains_q8(
                gain.to(torch.float64).tolist(), _scale32_float(q_input_scale32)
            )
            rms = _grouped_rmsnorm(
                rms_input_s8,
                rms_input_group_scale32,
                self.rms_input_group_size,
                gains,
            )

            tensor_sum_scale32 = layer16._exact_sum_scale32(
                attention_group_s8,
                down_group_s8,
                attention_group_scale32,
                down_group_scale32,
                SOURCE_GROUP_SIZE,
            )
            tensor_sum_s8, tensor_sum_saturation = layer16._exact_aligned_sum(
                attention_group_s8,
                down_group_s8,
                attention_group_scale32,
                down_group_scale32,
                tensor_sum_scale32,
                SOURCE_GROUP_SIZE,
            )

            self.records[position_id] = {
                "source_group_size": SOURCE_GROUP_SIZE,
                "rms_input_group_size": self.rms_input_group_size,
                "attention_source_s8": attention_source_s8,
                "down_source_s8": down_source_s8,
                "attention_base_scale32": attention_base_scale32,
                "down_base_scale32": down_base_scale32,
                "attention_group_scale32": torch.tensor(attention_group_scale32, dtype=torch.int64),
                "down_group_scale32": torch.tensor(down_group_scale32, dtype=torch.int64),
                "attention_group_s8": attention_group_s8,
                "down_group_s8": down_group_s8,
                "attention_saturation": attention_saturation,
                "down_saturation": down_saturation,
                "sum_scale32": tensor_sum_scale32,
                "sum_s8": tensor_sum_s8,
                "sum_saturation": tensor_sum_saturation,
                "rms_input_group_scale32": torch.tensor(rms_input_group_scale32, dtype=torch.int64),
                "rms_input_s8": rms_input_s8,
                "rms_input_saturation": rms_input_saturation,
                "rms_input_aligned_s64": rms["aligned"],
                "rms_input_common_exponent": rms["common_exponent"],
                "rms_gain_s16_q8": torch.tensor(gains, dtype=torch.int16),
                "rms_output_s8": rms["outputs"],
                "rms_output_saturation": rms["saturation"],
                "rms_sumsq": rms["sumsq"],
                "rms_mean_square": rms["mean_square"],
                "rms_ceil": rms["rms_ceil"],
                "rms_inv_q30": rms["inv_rms_q30"],
                "rms_saturation": bool(rms["saturation"].any()),
                "q_input_scale32": q_input_scale32,
                "binary_attention_q": binary_attention_q,
                "binary_down_q": binary_down_q,
                "binary_attention_scales": binary_attention_scales,
                "binary_down_scales": binary_down_scales,
                "binary_sum_reconstructed": binary_sum,
                "binary_input_norm": binary_input_norm,
                "binary_q_input_scale": binary_q_input_scale,
            }
            return derived, next_state, next_template

        backend.derive_layer_token = derive

    def restore(self) -> None:
        backend.derive_layer_token = self.original


def _artifacts(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    artifacts = {
        name: value for name, value in record.items() if isinstance(value, torch.Tensor)
    }
    for name in (
        "attention_base_scale32",
        "down_base_scale32",
        "sum_scale32",
        "q_input_scale32",
        "q_tensor_output_scale32",
        "rms_inv_q30",
        "rms_input_common_exponent",
    ):
        artifacts[name] = torch.tensor([int(record[name])], dtype=torch.int64)
    artifacts["rms_saturation"] = torch.tensor(
        [int(record["rms_saturation"])], dtype=torch.uint8
    )
    artifacts["rms_sumsq_u128_le"] = torch.tensor(
        list(int(record["rms_sumsq"]).to_bytes(16, "little")), dtype=torch.uint8
    )
    return artifacts


def run_group(
    group_size: int,
    cache: qoutput.GroupedQOutputScale32Cache,
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cache.configure(Q_OUTPUT_GROUP_SIZE)
    source_guard = GroupedRmsInputSourceGuard(cache, weights, group_size)
    score_guard = qoutput.QHeadScaleScoreGuard(cache)
    cache.install()
    score_guard.install()
    source_guard.install()
    try:
        result, _ = localizer.run_cut(
            None, [], weights, adapter, norm_gain, embedding, head, tokenizer
        )
    finally:
        source_guard.restore()
        score_guard.restore()
        cache.restore()

    expected_scores = backend.Q_HEADS * (2 * len(localizer.TOKEN_IDS) - 1)
    localizer.require(
        score_guard.replacements == expected_scores,
        "accepted grouped Q-output scale did not reach every layer-17 score",
    )
    exact = source_guard.records[len(localizer.TOKEN_IDS) - 1]
    rms_scales = exact["rms_input_group_scale32"]
    q_scales = exact["q_output_group_scale32"]
    result.update(
        {
            "source_group_size": SOURCE_GROUP_SIZE,
            "rms_input_group_size": group_size,
            "rms_input_group_count": backend.HIDDEN // group_size,
            "rms_input_group_scale32_min": f"0x{int(rms_scales.min()):08x}",
            "rms_input_group_scale32_max": f"0x{int(rms_scales.max()):08x}",
            "rms_input_s8_sha256": localizer.tensor_sha256(exact["rms_input_s8"]),
            "rms_input_saturation_count": int(exact["rms_input_saturation"].sum()),
            "rms_input_common_exponent": int(exact["rms_input_common_exponent"]),
            "rms_sumsq": str(exact["rms_sumsq"]),
            "rms_output_s8_sha256": localizer.tensor_sha256(exact["rms_output_s8"]),
            "rms_output_saturation_count": int(exact["rms_output_saturation"].sum()),
            "q_output_group_size": Q_OUTPUT_GROUP_SIZE,
            "q_output_group_count": backend.HIDDEN // Q_OUTPUT_GROUP_SIZE,
            "q_output_group_scale32_min": f"0x{int(q_scales.min()):08x}",
            "q_output_group_scale32_max": f"0x{int(q_scales.max()):08x}",
            "q_output_s8_sha256": localizer.tensor_sha256(exact["q_output_s8"]),
            "q_saturation_count": int(exact["q_saturation"].sum()),
            "layer17_score_scale_replacements": score_guard.replacements,
        }
    )
    return result, _artifacts(exact)


def _rtl_preflight(output: Path) -> dict[str, Any]:
    results = {}
    with tempfile.TemporaryDirectory(prefix="ace2-rmsnorm-input-", dir=ROOT / "build") as temporary:
        executable = Path(temporary) / "preflight.vvp"
        commands = {
            "iverilog": [
                "iverilog", "-g2012", "-Wall", "-Wno-timescale", "-s", RTL_TOP,
                "-o", str(executable), str(RTL),
            ],
            "verilator": [
                "verilator", "--lint-only", "-Wall", "-Wno-fatal",
                "--top-module", RTL_TOP, str(RTL),
            ],
        }
        for name, command in commands.items():
            completed = subprocess.run(
                command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False,
            )
            localizer.require(completed.returncode == 0, f"{name} RTL preflight failed")
            localizer.require(not completed.stdout.strip(), f"{name} RTL preflight warned")
            results[name] = {"command": command, "returncode": completed.returncode}
    return {
        "schema_version": 1,
        "top": RTL_TOP,
        "rtl": localizer.file_record(RTL),
        "checks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    localizer.require(ROOT.resolve() in output.parents, "output must be repository-relative")
    localizer.require(not output.exists(), "candidate output already exists")

    predecessor_freeze_path = PREDECESSOR / "candidate-freeze.json"
    predecessor_result_path = PREDECESSOR / "result.json"
    predecessor_freeze = json.loads(predecessor_freeze_path.read_text(encoding="utf-8"))
    predecessor_result = json.loads(predecessor_result_path.read_text(encoding="utf-8"))
    search_review = json.loads(PREDECESSOR_SEARCH_REVIEW.read_text(encoding="utf-8"))
    rtl_review = json.loads(PREDECESSOR_RTL_REVIEW.read_text(encoding="utf-8"))
    routing_result = json.loads(ROUTING_RESULT.read_text(encoding="utf-8"))
    routing_review = json.loads(ROUTING_SEARCH_REVIEW.read_text(encoding="utf-8"))
    localizer.require(
        predecessor_result["selected_q_output_group_size"] == Q_OUTPUT_GROUP_SIZE,
        "predecessor no longer selects Q-output group 128",
    )
    localizer.require(
        search_review["selected_q_output_group_size"] == Q_OUTPUT_GROUP_SIZE
        and search_review["selected_reference_rank"]
        == predecessor_result["selected_reference_rank"],
        "independent Q-output search review differs",
    )
    localizer.require(
        rtl_review["status"]
        == "PASS_FOCUSED_LAYER17_Q_OUTPUT_GROUPED_SCALE32_RTL_REFERENCE_AGREEMENT",
        "independent Q-output RTL review is absent",
    )
    localizer.require(
        routing_result["next_exact_upstream_boundary"]
        == "model.layers.16.output_to_model.layers.17.input_rmsnorm.input"
        and routing_review["next_exact_upstream_boundary"]
        == routing_result["next_exact_upstream_boundary"],
        "Fresh-L2 routing evidence no longer names the RMSNorm-input boundary",
    )
    localizer.require(
        predecessor_freeze["prompt"]["chat_token_ids"] == localizer.TOKEN_IDS,
        "canonical 30-token prefix differs",
    )
    baseline_rank = int(predecessor_result["selected_reference_rank"])
    baseline_top = int(predecessor_result["selected_top_token_id"])

    output.mkdir(parents=True)
    freeze = {
        "schema_version": 1,
        "mission": MISSION,
        "classification": "candidate_only_no_official_attempt_created",
        "checkpoint": predecessor_freeze["checkpoint"],
        "prompt": predecessor_freeze["prompt"],
        "reference_step0_token_id": localizer.REFERENCE_TOKEN,
        "baseline": {
            "layer16_source_group_size": SOURCE_GROUP_SIZE,
            "layer17_q_output_group_size": Q_OUTPUT_GROUP_SIZE,
            "top_token_id": baseline_top,
            "reference_rank": baseline_rank,
        },
        "ordered_rms_input_group_sizes": list(RMS_INPUT_GROUP_SIZES),
        "score_policy": (
            "first restored group in declared execution order; otherwise lowest "
            "reference rank strictly better than the independently accepted layer17 "
            "Q-output group-128 baseline, ties by declared execution order"
        ),
        "numeric_contract": {
            "boundary": "model.layers.16.output_to_model.layers.17.input_rmsnorm.input",
            "source_inputs": "accepted layer16 attention/down group-1 signed A8 plus Scale32",
            "sum_output": "one signed-A8 sample with explicit Scale32 per RMS input group",
            "group_scale": "smallest legal Scale32 not below exact group max-abs/127",
            "rmsnorm": "tensor-wide normalization of exact Scale32-aligned grouped A8 samples",
            "rms_output": "signed A8 with the accepted tensor q-input Scale32",
            "q_output_dependency": "independently accepted group-128 Scale32",
            "weights": "unchanged signed W4",
            "rounding": "signed round-to-nearest ties-to-even",
            "bf16_runtime_sidecar": False,
            "wider_carried_activation_payload": False,
            "token_or_logit_override": False,
        },
        "rtl_public_contract": {
            "top": RTL_TOP,
            "latency": "one registered ready/valid stage",
            "reset": "asynchronous active-low reset; synchronous clear",
            "ports": [
                "clk_i", "rst_ni", "clear_i", "in_valid_i", "in_ready_o",
                "attention_i:s8", "attention_scale32_i:u32", "down_i:s8",
                "down_scale32_i:u32", "group_scale32_i:u32", "channel_i:u10",
                "last_i", "out_valid_o", "out_ready_i", "sum_o:s8",
                "group_scale32_o:u32", "channel_o:u10", "last_o", "saturation_o",
            ],
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "u280_or_stage2_entered": False,
        },
        "bindings": {
            "predecessor_freeze": localizer.file_record(predecessor_freeze_path),
            "predecessor_result": localizer.file_record(predecessor_result_path),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
            "independent_search_review": localizer.file_record(PREDECESSOR_SEARCH_REVIEW),
            "independent_rtl_review": localizer.file_record(PREDECESSOR_RTL_REVIEW),
            "fresh_l2_routing_result": localizer.file_record(ROUTING_RESULT),
            "fresh_l2_routing_review": localizer.file_record(ROUTING_SEARCH_REVIEW),
            "runner": localizer.file_record(Path(__file__).resolve()),
            "rtl": localizer.file_record(RTL),
        },
    }
    localizer.write_json(output / "candidate-freeze.json", freeze)
    localizer.write_json(output / "rtl-interface-preflight.json", _rtl_preflight(output))

    started = time.monotonic()
    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    trace = []
    failures = []
    with safe_open(backend.canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        cache = qoutput.GroupedQOutputScale32Cache(weights, adapter)
        for group_size in RMS_INPUT_GROUP_SIZES:
            group_dir = output / f"groups/group-{group_size:03d}"
            try:
                record, tensors = run_group(
                    group_size,
                    cache,
                    weights,
                    adapter,
                    norm_gain,
                    embedding,
                    head,
                    tokenizer,
                )
            except Exception as error:
                detail = str(error)
                classification = (
                    "Q_OUTPUT_ZERO_GROUP_UNSUPPORTED"
                    if detail == "zero Q-output group is unsupported"
                    else "CANDIDATE_EXECUTION_FAILURE"
                )
                failure = {
                    "rms_input_group_size": group_size,
                    "classification": classification,
                    "detail": detail,
                    "root_cause_hypothesis": (
                        "The RMSNorm-input candidate produced an all-zero 128-channel "
                        "Q-output group, outside the independently accepted Q-output "
                        "group-128 evaluator contract."
                        if classification == "Q_OUTPUT_ZERO_GROUP_UNSUPPORTED"
                        else "Candidate execution failed outside the classified zero-group edge case."
                    ),
                    "regression": (
                        "Preserve the failure without extending the accepted Q-output contract; "
                        "continue the remaining frozen RMSNorm-input groups."
                    ),
                }
                localizer.write_json(group_dir / "failure.json", failure)
                failures.append(failure)
                print(
                    "ACE2_LAYER17_RMSNORM_INPUT_GROUPED_SCALE32_CANDIDATE_FAIL "
                    f"group={group_size} classification={classification} detail={detail}",
                    flush=True,
                )
                continue
            artifacts = {
                name: localizer.write_tensor(group_dir / f"{name}.bin", value)
                for name, value in sorted(tensors.items())
            }
            localizer.write_json(
                group_dir / "result.json", {"metrics": record, "artifacts": artifacts}
            )
            trace.append(record)
            print(
                "ACE2_LAYER17_RMSNORM_INPUT_GROUPED_SCALE32_CANDIDATE "
                f"group={group_size} top={record['top_token_id']} "
                f"rank={record['reference_rank']} sat={record['rms_input_saturation_count']}",
                flush=True,
            )

    localizer.require(trace, "no RMSNorm-input candidate completed execution")
    order = {group: index for index, group in enumerate(RMS_INPUT_GROUP_SIZES)}
    restored = [item for item in trace if item["top_token_id"] == localizer.REFERENCE_TOKEN]
    improved = [item for item in trace if int(item["reference_rank"]) < baseline_rank]
    best_observed = min(
        trace,
        key=lambda item: (
            int(item["reference_rank"]), order[int(item["rms_input_group_size"])]
        ),
    )
    if restored:
        selected = restored[0]
        selection_reason = "restored_step0_in_declared_execution_order"
    elif improved:
        selected = min(
            improved,
            key=lambda item: (
                int(item["reference_rank"]), order[int(item["rms_input_group_size"])]
            ),
        )
        selection_reason = "best_honest_reference_rank_improvement"
    else:
        selected = None
        selection_reason = "no_group_improved_independently_accepted_q_output_baseline"

    validation = selected if selected is not None else best_observed
    restored_step0 = selected is not None and selected["top_token_id"] == localizer.REFERENCE_TOKEN
    result = {
        "schema_version": 1,
        "status": (
            "PASS_LAYER17_RMSNORM_INPUT_GROUPED_SCALE32_STEP0_RESTORED"
            if restored_step0
            else "PARTIAL_LAYER17_RMSNORM_INPUT_GROUPED_SCALE32_RANK_IMPROVED"
            if selected is not None
            else "PARTIAL_LAYER17_RMSNORM_INPUT_GROUPED_SCALE32_NO_IMPROVEMENT"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "baseline_reference_rank": baseline_rank,
        "baseline_top_token_id": baseline_top,
        "ordered_rms_input_group_sizes": list(RMS_INPUT_GROUP_SIZES),
        "trace": trace,
        "failures": failures,
        "selected_rms_input_group_size": (
            int(selected["rms_input_group_size"]) if selected is not None else None
        ),
        "selected_reference_rank": (
            int(selected["reference_rank"]) if selected is not None else baseline_rank
        ),
        "selected_top_token_id": (
            int(selected["top_token_id"]) if selected is not None else baseline_top
        ),
        "selection_reason": selection_reason,
        "best_observed_rms_input_group_size": int(best_observed["rms_input_group_size"]),
        "best_observed_reference_rank": int(best_observed["reference_rank"]),
        "best_observed_top_token_id": int(best_observed["top_token_id"]),
        "rtl_validation_rms_input_group_size": int(validation["rms_input_group_size"]),
        "first_token_restored": bool(restored_step0),
        "next_exact_upstream_boundary": (
            None
            if restored_step0
            else "model.layers.16.mlp.down_proj.output_to_model.layers.16.output"
        ),
        "next_required_action": (
            "Bind the selected grouped RMSNorm-input Scale32 payload to focused RTL."
            if selected is not None
            else "Validate the best null boundary and investigate the exact layer-16 down-projection/residual source boundary."
        ),
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
        "bindings": {
            "freeze": localizer.file_record(output / "candidate-freeze.json"),
            "rtl_preflight": localizer.file_record(output / "rtl-interface-preflight.json"),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
        },
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    print(
        "ACE2_LAYER17_RMSNORM_INPUT_GROUPED_SCALE32_RESULT "
        f"status={result['status']} selected={result['selected_rms_input_group_size']} "
        f"best={result['best_observed_rms_input_group_size']} "
        f"rank={result['best_observed_reference_rank']} restored={result['first_token_restored']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ACE2_LAYER17_RMSNORM_INPUT_GROUPED_SCALE32_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
