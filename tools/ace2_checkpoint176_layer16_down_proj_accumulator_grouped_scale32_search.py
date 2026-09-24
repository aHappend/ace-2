#!/usr/bin/env python3
"""Search layer-16 down-projection accumulator-to-A8 Scale32 contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
from tools import ace2_checkpoint176_layer16_down_proj_output_grouped_scale32_search as downoutput
from tools import ace2_checkpoint176_layer16_source_grouped_scale32_search as layer16
from tools import ace2_checkpoint176_layer17_q_output_grouped_scale32_search as qoutput
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner
from tools.ace2_quality_contracts import (
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    scale32_ratio,
)
from tools.ace2_rmsnorm_reference import derive_scaled_gains_q8, reference_rmsnorm


MISSION = "w4a8-layer16-down-proj-accumulator-requantization-repair-v1"
BOUNDARY = "model.layers.16.mlp.down_proj.accumulator_to_output_requantization"
NEXT_BOUNDARY = "model.layers.16.mlp.down_proj.input_scale_to_accumulator_formation"
ATTENTION_SOURCE_GROUP_SIZE = 1
Q_OUTPUT_GROUP_SIZE = 128
DOWN_OUTPUT_GROUP_SIZES = (32, 16, 8, 4, 1)
DOWN_PATTERN = re.compile(r"layer16_position(\d+)_down$")
ACCEPTED_LAYER16 = (
    ROOT
    / "evidence/candidates/w4a8-layer16-source-grouped-activation-repair-v1/"
    "candidate-0002"
)
ACCEPTED_Q_OUTPUT = (
    ROOT
    / "evidence/candidates/w4a8-layer17-q-output-source-grouped-repair-v1/"
    "candidate-0001"
)
ACCEPTED_Q_SEARCH_REVIEW = ROOT / "build/l2-review-layer17-q-output-search-r1/result.json"
ACCEPTED_Q_RTL_REVIEW = ROOT / "build/l2-review-layer17-q-output-rtl-r1/result.json"
PREDECESSOR = (
    ROOT
    / "evidence/candidates/w4a8-layer16-down-proj-output-residual-source-repair-v1/"
    "candidate-0001"
)
PREDECESSOR_SEARCH_REVIEW = ROOT / "build/l2-review-layer16-down-proj-output-search-r1/result.json"
PREDECESSOR_RTL_REVIEW = ROOT / "build/l2-review-layer16-down-proj-output-rtl-r1/result.json"
PREDECESSOR_REVIEW_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "w4a8-layer16-down-proj-output-residual-source-repair-v1/round-0001.json"
)
RTL = ROOT / "rtl/ace2_layer16_down_proj_accumulator_requantizer_core.sv"
RTL_TOP = "ace2_layer16_down_proj_accumulator_requantizer_core"
PROTECTED_HASHES = dict(downoutput.PROTECTED_HASHES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _scale32_float(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def _group_dequantize(
    values: torch.Tensor, scales: tuple[int, ...], group_size: int
) -> torch.Tensor:
    output = torch.empty(values.numel(), dtype=torch.float64)
    for index, value in enumerate(values.to(torch.int64).tolist()):
        output[index] = int(value) * _scale32_float(scales[index // group_size])
    return output


def _down_group_scales_from_accumulator(
    accumulator: torch.Tensor,
    input_scale32: int,
    weight_scale32: tuple[int, ...],
    group_size: int,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Derive output Scale32 records, including the canonical all-zero record."""
    input_num, input_den = scale32_ratio(input_scale32)
    scales = []
    zero_groups = []
    for start in range(0, accumulator.numel(), group_size):
        maximum = Fraction(0, 1)
        for row in range(start, start + group_size):
            weight_num, weight_den = scale32_ratio(weight_scale32[row])
            value = Fraction(
                abs(int(accumulator[row])) * input_num * weight_num,
                input_den * weight_den,
            )
            maximum = max(maximum, value)
        zero_groups.append(maximum == 0)
        scales.append(
            ceil_scale32_from_ratio(
                maximum.numerator,
                maximum.denominator * 127,
            )
        )
    return tuple(scales), tuple(zero_groups)


class GroupedDownProjAccumulatorScale32Cache(qoutput.GroupedQOutputScale32Cache):
    """Replace only layer-16 down_proj accumulator requantization and accepted layer-17 Q."""

    def __init__(self, weights: Any, adapter: Any) -> None:
        super().__init__(weights, adapter)
        name = f"model.layers.{layer16.SOURCE_LAYER}.mlp.down_proj"
        self.down_merged = self.merged[name]
        self.down_metadata = self.metadata[self.down_merged.data_ptr()]
        self.down_weight_scale32 = tuple(
            ceil_scale32_from_float(float(value))
            for value in self.down_metadata["weight_scale"].tolist()
        )
        self.down_output_group_size = DOWN_OUTPUT_GROUP_SIZES[0]
        self.down_records: dict[int, dict[str, Any]] = {}

    def configure_down(self, output_group_size: int) -> None:
        localizer.require(
            backend.HIDDEN % output_group_size == 0,
            "down-projection output group does not divide 896 channels",
        )
        self.down_output_group_size = output_group_size
        self.down_records.clear()
        super().configure(Q_OUTPUT_GROUP_SIZE)

    def derive_projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        match = DOWN_PATTERN.fullmatch(name)
        if match is None:
            return super().derive_projection(
                name, merged, input_q, input_scale, float_input, source_hashes
            )

        position = int(match.group(1))
        metadata = self.metadata.get(merged.data_ptr())
        localizer.require(metadata is self.down_metadata, "layer-16 down metadata differs")
        input_scale32 = ceil_scale32_from_float(float(input_scale))
        qweight = metadata["qweight"]
        accumulator = torch.mv(qweight.to(torch.int64), input_q.to(torch.int64))
        localizer.require(
            bool(torch.all(accumulator >= -(1 << 31)))
            and bool(torch.all(accumulator < (1 << 31))),
            f"{name} accumulator overflow",
        )
        group_scales, zero_groups = _down_group_scales_from_accumulator(
            accumulator,
            input_scale32,
            self.down_weight_scale32,
            self.down_output_group_size,
        )
        multiplier_values = []
        shift_values = []
        for row, weight_scale32 in enumerate(self.down_weight_scale32):
            group_id = row // self.down_output_group_size
            if zero_groups[group_id]:
                localizer.require(
                    int(accumulator[row]) == 0,
                    "zero down-projection group contains a nonzero accumulator",
                )
                multiplier, shift = 0, 0
            else:
                multiplier, shift = layer16._derive_scale32_multiplier(
                    input_scale32,
                    weight_scale32,
                    group_scales[group_id],
                )
            multiplier_values.append(multiplier)
            shift_values.append(shift)
        multiplier = torch.tensor(multiplier_values, dtype=torch.int64)
        right_shift = torch.tensor(shift_values, dtype=torch.int64)
        rounded = localizer.round_shift_even_tensor(accumulator * multiplier, right_shift)
        output_q = rounded.clamp(-128, 127).to(torch.int8)
        saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
        group_scale_tensor = torch.tensor(group_scales, dtype=torch.int64)
        float_output = torch.mv(merged, float_input.to(torch.float32)).contiguous()
        tensor_output_scale32 = ceil_scale32_from_float(
            float(backend.canonical.scale_for(float_output))
        )
        record = {
            "down_input_s8": input_q.contiguous(),
            "down_input_scale32": input_scale32,
            "down_weight_s4": qweight,
            "down_weight_scale32": torch.tensor(
                self.down_weight_scale32, dtype=torch.int64
            ),
            "down_accumulator_s32": accumulator,
            "down_output_group_scale32": group_scale_tensor,
            "down_multiplier_s32": multiplier,
            "down_shift_u6": right_shift,
            "down_rounded_s64": rounded,
            "down_output_s8": output_q,
            "down_saturation": saturation,
            "down_tensor_output_scale32": tensor_output_scale32,
            "down_output_group_size": self.down_output_group_size,
            "binary_down_float_output": float_output,
        }
        self.down_records[position] = record
        return {
            "name": name,
            "input_q": input_q,
            "input_scale": _scale32_float(input_scale32),
            "qweight": qweight,
            "weight_scale": metadata["weight_scale"],
            "multiplier": multiplier,
            "right_shift": right_shift,
            "accumulator": accumulator,
            "rounded": rounded,
            "output_q": output_q,
            "output_scale": _scale32_float(group_scales[0]),
            "saturation": saturation,
            "float_output": float_output,
            "source_hashes": source_hashes,
            "exact_grouped_down_accumulator_requantization": True,
        }


class DownProjAccumulatorSourceGuard:
    """Feed the direct grouped down-projection A8 payload into the accepted source path."""

    def __init__(
        self,
        cache: GroupedDownProjAccumulatorScale32Cache,
        weights: Any,
        down_group_size: int,
    ) -> None:
        self.cache = cache
        self.weights = weights
        self.down_group_size = down_group_size
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
                    "down accumulator record missing at layer 17",
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
            localizer.require(
                position_id in self.cache.down_records,
                "layer-16 down accumulator requantization was not captured",
            )
            down = self.cache.down_records[position_id]
            down_group_scales = tuple(
                int(value) for value in down["down_output_group_scale32"].tolist()
            )
            down_group_s8 = down["down_output_s8"]
            localizer.require(
                torch.equal(position["projections"]["down"]["output_q"], down_group_s8),
                "backend did not consume direct down accumulator output",
            )

            attention_source_s8 = position["attention_residual"]["output"].contiguous()
            attention_base_scale = float(position["attention_residual"]["scale"])
            binary_attention, binary_attention_q, binary_attention_scales = (
                layer16._binary32_grouped_reconstruct(
                    attention_source_s8,
                    attention_base_scale,
                    ATTENTION_SOURCE_GROUP_SIZE,
                )
            )
            binary_down = _group_dequantize(
                down_group_s8, down_group_scales, self.down_group_size
            )
            binary_down_scales = torch.tensor(
                [_scale32_float(value) for value in down_group_scales],
                dtype=torch.float32,
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
            attention_group_scales = layer16._exact_group_scales(
                attention_source_s8,
                attention_base_scale32,
                ATTENTION_SOURCE_GROUP_SIZE,
            )
            attention_group_s8, attention_saturation = layer16._exact_group_requantize(
                attention_source_s8,
                attention_base_scale32,
                attention_group_scales,
                ATTENTION_SOURCE_GROUP_SIZE,
            )
            sum_values = downoutput._sum_reals(
                attention_group_s8,
                down_group_s8,
                attention_group_scales,
                down_group_scales,
                self.down_group_size,
            )
            sum_scale32 = downoutput._tensor_scale(sum_values)
            sum_s8, sum_saturation = downoutput._quantize_values(
                sum_values, sum_scale32
            )
            gains = derive_scaled_gains_q8(
                gain.to(torch.float64).tolist(), _scale32_float(q_input_scale32)
            )
            rms = reference_rmsnorm(sum_s8.to(torch.int64).tolist(), gains)

            record = dict(down)
            record.update(
                {
                    "attention_source_group_size": ATTENTION_SOURCE_GROUP_SIZE,
                    "attention_source_s8": attention_source_s8,
                    "attention_base_scale32": attention_base_scale32,
                    "attention_group_scale32": torch.tensor(
                        attention_group_scales, dtype=torch.int64
                    ),
                    "attention_group_s8": attention_group_s8,
                    "attention_saturation": attention_saturation,
                    "down_group_s8": down_group_s8,
                    "down_group_scale32": down["down_output_group_scale32"],
                    "sum_scale32": sum_scale32,
                    "sum_s8": sum_s8,
                    "sum_saturation": sum_saturation,
                    "rms_gain_s16_q8": torch.tensor(gains, dtype=torch.int16),
                    "rms_output_s8": torch.tensor(rms.outputs, dtype=torch.int8),
                    "rms_sumsq": rms.sumsq,
                    "rms_inv_q30": rms.inv_rms_q30,
                    "rms_saturation": rms.saturation_seen,
                    "q_input_scale32": q_input_scale32,
                    "binary_attention_q": binary_attention_q,
                    "binary_down_q": down_group_s8,
                    "binary_attention_scales": binary_attention_scales,
                    "binary_down_scales": binary_down_scales,
                    "binary_sum_reconstructed": binary_sum,
                    "binary_input_norm": binary_input_norm,
                    "binary_q_input_scale": binary_q_input_scale,
                }
            )
            self.records[position_id] = record
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
        "down_input_scale32",
        "down_tensor_output_scale32",
        "sum_scale32",
        "q_input_scale32",
        "q_tensor_output_scale32",
        "rms_sumsq",
        "rms_inv_q30",
    ):
        artifacts[name] = torch.tensor([int(record[name])], dtype=torch.int64)
    artifacts["rms_saturation"] = torch.tensor(
        [int(record["rms_saturation"])], dtype=torch.uint8
    )
    return artifacts


def run_group(
    down_group_size: int,
    cache: GroupedDownProjAccumulatorScale32Cache,
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cache.configure_down(down_group_size)
    source_guard = DownProjAccumulatorSourceGuard(cache, weights, down_group_size)
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
    down_scales = exact["down_output_group_scale32"]
    q_scales = exact["q_output_group_scale32"]
    result.update(
        {
            "attention_source_group_size": ATTENTION_SOURCE_GROUP_SIZE,
            "down_output_group_size": down_group_size,
            "down_output_group_count": backend.HIDDEN // down_group_size,
            "down_output_group_scale32_min": f"0x{int(down_scales.min()):08x}",
            "down_output_group_scale32_max": f"0x{int(down_scales.max()):08x}",
            "down_output_s8_sha256": localizer.tensor_sha256(exact["down_output_s8"]),
            "down_accumulator_s32_sha256": localizer.tensor_sha256(
                exact["down_accumulator_s32"]
            ),
            "down_saturation_count": int(exact["down_saturation"].sum()),
            "sum_scale32": f"0x{int(exact['sum_scale32']):08x}",
            "sum_s8_sha256": localizer.tensor_sha256(exact["sum_s8"]),
            "sum_saturation_count": int(exact["sum_saturation"].sum()),
            "rms_output_s8_sha256": localizer.tensor_sha256(exact["rms_output_s8"]),
            "rms_saturation": bool(exact["rms_saturation"]),
            "q_output_group_size": Q_OUTPUT_GROUP_SIZE,
            "q_output_group_scale32_min": f"0x{int(q_scales.min()):08x}",
            "q_output_group_scale32_max": f"0x{int(q_scales.max()):08x}",
            "q_output_s8_sha256": localizer.tensor_sha256(exact["q_output_s8"]),
            "q_saturation_count": int(exact["q_saturation"].sum()),
            "layer17_score_scale_replacements": score_guard.replacements,
        }
    )
    return result, _artifacts(exact)


def _rtl_preflight() -> dict[str, Any]:
    results = {}
    with tempfile.TemporaryDirectory(
        prefix="ace2-down-acc-requant-", dir=ROOT / "build"
    ) as temporary:
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
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
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


def _tool_versions() -> dict[str, str]:
    commands = {
        "python": [sys.executable, "--version"],
        "iverilog": ["iverilog", "-V"],
        "verilator": ["verilator", "--version"],
    }
    versions = {"torch": torch.__version__}
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        localizer.require(completed.returncode == 0, f"tool version failed: {name}")
        versions[name] = completed.stdout.strip().splitlines()[0]
    return versions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    localizer.require(ROOT.resolve() in output.parents, "output must be repository-relative")
    localizer.require(not output.exists(), "candidate output already exists")

    for relative, expected in PROTECTED_HASHES.items():
        localizer.require(
            _sha256(ROOT / relative) == expected,
            f"protected hash differs: {relative}",
        )

    predecessor_result_path = PREDECESSOR / "result.json"
    predecessor_result = json.loads(predecessor_result_path.read_text(encoding="utf-8"))
    predecessor_search_review = json.loads(
        PREDECESSOR_SEARCH_REVIEW.read_text(encoding="utf-8")
    )
    predecessor_rtl_review = json.loads(
        PREDECESSOR_RTL_REVIEW.read_text(encoding="utf-8")
    )
    predecessor_handoff = json.loads(
        PREDECESSOR_REVIEW_HANDOFF.read_text(encoding="utf-8")
    )
    layer16_result_path = ACCEPTED_LAYER16 / "result.json"
    layer16_result = json.loads(layer16_result_path.read_text(encoding="utf-8"))
    q_freeze_path = ACCEPTED_Q_OUTPUT / "candidate-freeze.json"
    q_result_path = ACCEPTED_Q_OUTPUT / "result.json"
    q_freeze = json.loads(q_freeze_path.read_text(encoding="utf-8"))
    q_result = json.loads(q_result_path.read_text(encoding="utf-8"))
    q_search_review = json.loads(ACCEPTED_Q_SEARCH_REVIEW.read_text(encoding="utf-8"))
    q_rtl_review = json.loads(ACCEPTED_Q_RTL_REVIEW.read_text(encoding="utf-8"))

    localizer.require(
        predecessor_result["selected_down_output_group_size"] is None
        and predecessor_result["next_exact_upstream_boundary"] == BOUNDARY,
        "predecessor result is not the required negative boundary",
    )
    localizer.require(
        predecessor_search_review["selected_down_output_group_size"] is None
        and predecessor_search_review["next_exact_upstream_boundary"] == BOUNDARY,
        "Fresh-L2 predecessor search review differs",
    )
    localizer.require(
        predecessor_rtl_review["status"]
        == "PASS_FOCUSED_LAYER16_DOWN_PROJ_OUTPUT_GROUPED_SCALE32_RTL_REFERENCE_AGREEMENT",
        "Fresh-L2 predecessor RTL review is absent",
    )
    localizer.require(
        predecessor_handoff["producer_role"] == "reviewer"
        and predecessor_handoff["review"]["status"] == "done"
        and BOUNDARY in predecessor_handoff["review"]["reason"],
        "sealed Fresh-L2 reviewer did not accept the predecessor negative boundary",
    )
    localizer.require(
        layer16_result["selected_source_group_size"] == ATTENTION_SOURCE_GROUP_SIZE,
        "accepted layer-16 attention source is no longer group 1",
    )
    localizer.require(
        q_result["selected_q_output_group_size"] == Q_OUTPUT_GROUP_SIZE
        and q_search_review["selected_q_output_group_size"] == Q_OUTPUT_GROUP_SIZE,
        "accepted layer-17 Q-output path is no longer group 128",
    )
    localizer.require(
        q_rtl_review["status"]
        == "PASS_FOCUSED_LAYER17_Q_OUTPUT_GROUPED_SCALE32_RTL_REFERENCE_AGREEMENT",
        "accepted layer-17 Q-output RTL review is absent",
    )
    localizer.require(
        q_freeze["prompt"]["chat_token_ids"] == localizer.TOKEN_IDS,
        "canonical 30-token prefix differs",
    )
    baseline_rank = int(q_result["selected_reference_rank"])
    baseline_top = int(q_result["selected_top_token_id"])
    localizer.require(baseline_rank == 33375, "frozen rank baseline differs")

    output.mkdir(parents=True)
    freeze = {
        "schema_version": 1,
        "mission": MISSION,
        "classification": "candidate_only_no_official_attempt_created",
        "checkpoint": q_freeze["checkpoint"],
        "prompt": q_freeze["prompt"],
        "reference_step0_token_id": localizer.REFERENCE_TOKEN,
        "baseline": {
            "layer16_attention_source_group_size": ATTENTION_SOURCE_GROUP_SIZE,
            "layer17_q_output_group_size": Q_OUTPUT_GROUP_SIZE,
            "top_token_id": baseline_top,
            "reference_rank": baseline_rank,
        },
        "ordered_down_output_group_sizes": list(DOWN_OUTPUT_GROUP_SIZES),
        "score_policy": (
            "first restored group in declared execution order; otherwise lowest reference "
            "rank strictly better than 33375, ties by declared execution order"
        ),
        "numeric_contract": {
            "boundary": BOUNDARY,
            "down_projection_input": "unchanged model-derived signed A8 plus producer Scale32",
            "down_projection_weights": "unchanged signed W4 plus one Scale32 per output row",
            "accumulator": "internal signed 32-bit MAC result; not a carried activation sidecar",
            "down_projection_output": "signed A8 with one explicit Scale32 per frozen output group",
            "attention_source": "independently accepted signed-A8 group-1 Scale32 payload",
            "q_output_dependency": "independently accepted group-128 signed-A8 Scale32",
            "group_scale": (
                "smallest legal Scale32 not below exact max(abs(accumulator * input "
                "Scale32 * row-weight Scale32))/127"
            ),
            "rounding": "signed round-to-nearest ties-to-even",
            "saturation": "signed int8 clamp",
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
                "accumulator_i:s32", "multiplier_i:s32", "right_shift_i:u6",
                "group_scale32_i:u32", "channel_i:u10", "last_i",
                "out_valid_o", "out_ready_i", "down_o:s8",
                "group_scale32_o:u32", "channel_o:u10", "last_o",
                "saturation_o",
            ],
        },
        "tool_versions": _tool_versions(),
        "protected_hashes": PROTECTED_HASHES,
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "u280_or_stage2_entered": False,
        },
        "bindings": {
            "accepted_layer16_result": localizer.file_record(layer16_result_path),
            "accepted_layer16_sha256s": localizer.file_record(ACCEPTED_LAYER16 / "SHA256SUMS"),
            "accepted_q_freeze": localizer.file_record(q_freeze_path),
            "accepted_q_result": localizer.file_record(q_result_path),
            "accepted_q_sha256s": localizer.file_record(ACCEPTED_Q_OUTPUT / "SHA256SUMS"),
            "accepted_q_search_review": localizer.file_record(ACCEPTED_Q_SEARCH_REVIEW),
            "accepted_q_rtl_review": localizer.file_record(ACCEPTED_Q_RTL_REVIEW),
            "predecessor_result": localizer.file_record(predecessor_result_path),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
            "predecessor_search_review": localizer.file_record(PREDECESSOR_SEARCH_REVIEW),
            "predecessor_rtl_review": localizer.file_record(PREDECESSOR_RTL_REVIEW),
            "predecessor_reviewer_handoff": _external_record(PREDECESSOR_REVIEW_HANDOFF),
            "runner": localizer.file_record(Path(__file__).resolve()),
            "rtl": localizer.file_record(RTL),
        },
    }
    localizer.write_json(output / "candidate-freeze.json", freeze)
    localizer.write_json(output / "rtl-interface-preflight.json", _rtl_preflight())

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
        cache = GroupedDownProjAccumulatorScale32Cache(weights, adapter)
        for group_size in DOWN_OUTPUT_GROUP_SIZES:
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
                failure = {
                    "down_output_group_size": group_size,
                    "classification": "CANDIDATE_EXECUTION_FAILURE",
                    "detail": str(error),
                    "root_cause_hypothesis": (
                        "The bounded accumulator-derived A8 payload could not be consumed "
                        "by the frozen downstream evaluator."
                    ),
                    "regression": (
                        "Preserve this failed group and continue the remaining frozen groups "
                        "without changing the evaluator or contract order."
                    ),
                }
                localizer.write_json(group_dir / "failure.json", failure)
                failures.append(failure)
                print(
                    "ACE2_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANT_CANDIDATE_FAIL "
                    f"group={group_size} detail={error}",
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
                "ACE2_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANT_CANDIDATE "
                f"group={group_size} top={record['top_token_id']} "
                f"rank={record['reference_rank']} sat={record['down_saturation_count']}",
                flush=True,
            )

    localizer.require(trace, "no accumulator requantization candidate completed execution")
    order = {group: index for index, group in enumerate(DOWN_OUTPUT_GROUP_SIZES)}
    restored = [item for item in trace if item["top_token_id"] == localizer.REFERENCE_TOKEN]
    improved = [item for item in trace if int(item["reference_rank"]) < baseline_rank]
    best_observed = min(
        trace,
        key=lambda item: (
            int(item["reference_rank"]), order[int(item["down_output_group_size"])]
        ),
    )
    if restored:
        selected = restored[0]
        selection_reason = "restored_step0_in_declared_execution_order"
    elif improved:
        selected = min(
            improved,
            key=lambda item: (
                int(item["reference_rank"]), order[int(item["down_output_group_size"])]
            ),
        )
        selection_reason = "best_honest_reference_rank_improvement"
    else:
        selected = None
        selection_reason = "no_group_improved_independently_accepted_rank_33375_baseline"

    validation = selected if selected is not None else best_observed
    restored_step0 = selected is not None and selected["top_token_id"] == localizer.REFERENCE_TOKEN
    result = {
        "schema_version": 1,
        "status": (
            "PASS_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANT_STEP0_RESTORED"
            if restored_step0
            else "PARTIAL_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANT_RANK_IMPROVED"
            if selected is not None
            else "PARTIAL_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANT_NO_IMPROVEMENT"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "boundary": BOUNDARY,
        "baseline_reference_rank": baseline_rank,
        "baseline_top_token_id": baseline_top,
        "ordered_down_output_group_sizes": list(DOWN_OUTPUT_GROUP_SIZES),
        "trace": trace,
        "failures": failures,
        "selected_down_output_group_size": (
            int(selected["down_output_group_size"]) if selected is not None else None
        ),
        "selected_reference_rank": (
            int(selected["reference_rank"]) if selected is not None else baseline_rank
        ),
        "selected_top_token_id": (
            int(selected["top_token_id"]) if selected is not None else baseline_top
        ),
        "selection_reason": selection_reason,
        "best_observed_down_output_group_size": int(best_observed["down_output_group_size"]),
        "best_observed_reference_rank": int(best_observed["reference_rank"]),
        "best_observed_top_token_id": int(best_observed["top_token_id"]),
        "rtl_validation_down_output_group_size": int(validation["down_output_group_size"]),
        "first_token_restored": bool(restored_step0),
        "next_exact_upstream_boundary": None if selected is not None else NEXT_BOUNDARY,
        "next_required_action": (
            "Bind the selected accumulator requantization payload to focused RTL."
            if selected is not None
            else "Validate the best clean null and preserve the input-scale-to-accumulator boundary."
        ),
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
        "bindings": {
            "freeze": localizer.file_record(output / "candidate-freeze.json"),
            "rtl_preflight": localizer.file_record(output / "rtl-interface-preflight.json"),
            "accepted_layer16_sha256s": localizer.file_record(ACCEPTED_LAYER16 / "SHA256SUMS"),
            "accepted_q_sha256s": localizer.file_record(ACCEPTED_Q_OUTPUT / "SHA256SUMS"),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
        },
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    print(
        "ACE2_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANT_RESULT "
        f"status={result['status']} selected={result['selected_down_output_group_size']} "
        f"best={result['best_observed_down_output_group_size']} "
        f"rank={result['best_observed_reference_rank']} restored={result['first_token_restored']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ACE2_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANT_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
