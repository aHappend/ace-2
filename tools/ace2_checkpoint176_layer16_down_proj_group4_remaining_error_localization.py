#!/usr/bin/env python3
"""Localize the remaining checkpoint-176 error inside layer-16 down_proj."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
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

import peft
import tokenizers
import torch
import transformers
from peft import PeftModel
from safetensors import safe_open
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_checkpoint176_hidden_state_localizer as localizer
from tools import ace2_checkpoint176_layer16_down_proj_accumulator_grouped_scale32_search as accumulator_search
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


MISSION = "w4a8-layer16-down-proj-group4-remaining-error-localization-v1"
BOUNDARY_PREFIX = "model.layers.16.mlp.down_proj"
DOWN_OUTPUT_GROUP_SIZE = 4
ATTENTION_SOURCE_GROUP_SIZE = 1
Q_OUTPUT_GROUP_SIZE = 128
MATERIAL_RANK_RECOVERY = 1496
ORDERED_CUTS = (
    "input_activation",
    "w4_row_dequantized_product_accumulator_formation",
    "accumulator",
    "requantized_output",
    "residual_sum",
)
EXACT_BOUNDARIES = {
    "input_activation": f"{BOUNDARY_PREFIX}.input_activation",
    "w4_row_dequantized_product_accumulator_formation": (
        f"{BOUNDARY_PREFIX}.w4_row_dequantized_product_accumulator_formation"
    ),
    "accumulator": f"{BOUNDARY_PREFIX}.accumulator",
    "requantized_output": f"{BOUNDARY_PREFIX}.requantized_output",
    "residual_sum": f"{BOUNDARY_PREFIX}.output_to_residual_sum",
}

PREDECESSOR = (
    ROOT
    / "evidence/candidates/w4a8-layer16-down-proj-accumulator-requantization-repair-v1/"
    "candidate-0002"
)
PREDECESSOR_SEARCH_REVIEW = (
    ROOT / "build/l2-review-layer16-down-proj-accumulator-search-r1/result.json"
)
PREDECESSOR_RTL_REVIEW = (
    ROOT / "build/l2-review-layer16-down-proj-accumulator-rtl-r1/result.json"
)
PREDECESSOR_REVIEW_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "w4a8-layer16-down-proj-accumulator-requantization-repair-v1/round-0001.json"
)
PROTECTED_HASHES = dict(accumulator_search.PROTECTED_HASHES)
EXPECTED_CONTROL_HASHES = {
    "down_input_s8": "567012f9a4028e12ee7cf2223daaa016ef578f334ccd46388f50e6b3b9901cd0",
    "down_weight_s4": "58abe01a09f61573707b9af98b105da07ad0056eb05a8cac5343452ab15e40bb",
    "down_weight_scale32": "6bef8f087de7842aa664168e6c3beefe172778500c658fb5ecedf4174f0d8f2f",
    "down_accumulator_s32": "ce7dcb06952e5e98ec432c7f53641b044388e625f9e374b4843e03a3cafd7da0",
    "down_output_group_scale32": "6a5131529256fac41e859bf8c924afd88186d4f691229e59a15637148fc1d8a6",
    "down_output_s8": "d37971e185d79d5d34dc1135793363a9aac92303d99b4840cf02503fec6f4fd3",
    "sum_s8": "2dd4036d0584650b847b89eb904c7f44587f35f1ec73d2cd81f02ac39c1a71fb",
    "q_output_s8": "a3490fe52dbe08a6c6b289ab9168ba249d1b6e36de637d2d0d38e917f5f02f26",
}


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


def _verify_manifest(directory: Path) -> int:
    manifest = directory / "SHA256SUMS"
    count = 0
    for line in manifest.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        localizer.require(
            _sha256(directory / relative) == expected,
            f"sealed predecessor manifest mismatch: {relative}",
        )
        count += 1
    localizer.require(count > 0, "sealed predecessor manifest is empty")
    return count


def _tokenizer_records(snapshot: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
    ):
        path = snapshot / name
        if path.is_file():
            records[name] = localizer.file_record(path)
    localizer.require(
        "tokenizer.json" in records and "tokenizer_config.json" in records,
        "tokenizer binding files are absent",
    )
    return records


def _event(output: Path, kind: str, **fields: Any) -> None:
    record = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "kind": kind,
        **fields,
    }
    with (output / "execution-events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _group_scales_from_real(values: torch.Tensor, group_size: int) -> tuple[int, ...]:
    localizer.require(values.dtype == torch.float64, "real group-scale dtype differs")
    scales = []
    for start in range(0, values.numel(), group_size):
        maximum = float(values[start : start + group_size].abs().max())
        numerator, denominator = maximum.as_integer_ratio()
        scales.append(ceil_scale32_from_ratio(numerator, denominator * 127))
    return tuple(scales)


def _quantize_grouped_real(
    values: torch.Tensor, scales: tuple[int, ...], group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rounded = torch.empty(values.numel(), dtype=torch.int64)
    output = torch.empty(values.numel(), dtype=torch.int8)
    saturation = torch.zeros(values.numel(), dtype=torch.uint8)
    for index, value in enumerate(values.tolist()):
        record = scales[index // group_size]
        if record == 0:
            localizer.require(float(value) == 0.0, "zero Scale32 group contains nonzero real")
            converted = 0
        else:
            converted = int(
                torch.round(
                    torch.tensor(value / _scale32_float(record), dtype=torch.float64)
                )
            )
        clipped = max(-128, min(127, converted))
        rounded[index] = converted
        output[index] = clipped
        saturation[index] = int(clipped != converted)
    return output, rounded, saturation


def _capture_bf16_boundaries(
    snapshot: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        attn_implementation="eager",
        device_map=None,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    peft_model = PeftModel.from_pretrained(
        model,
        backend.canonical.ADAPTER_DIR,
        is_trainable=False,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=True,
    )
    model = peft_model.merge_and_unload().eval()
    layer = model.model.layers[layer16.SOURCE_LAYER]
    captures: dict[str, torch.Tensor] = {}

    def capture_output(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            localizer.require(isinstance(value, torch.Tensor), f"BF16 capture differs: {name}")
            captures[name] = value[0].float().cpu().contiguous()

        return hook

    handles = [
        layer.mlp.down_proj.register_forward_pre_hook(
            lambda _module, inputs: captures.__setitem__(
                "down_input", inputs[0][0].float().cpu().contiguous()
            )
        ),
        layer.mlp.down_proj.register_forward_hook(capture_output("full_bf16_down_output")),
        layer.register_forward_hook(capture_output("full_bf16_layer_output")),
    ]
    input_ids = torch.tensor([localizer.TOKEN_IDS], dtype=torch.long)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
    for handle in handles:
        handle.remove()
    logits = output.logits[0, -1].float().cpu().contiguous()
    captures["full_bf16_logits"] = logits
    localizer.require(
        int(torch.argmax(logits)) == localizer.REFERENCE_TOKEN,
        "full BF16 reference first token changed",
    )
    localizer.require(
        list(captures["down_input"].shape)
        == [len(localizer.TOKEN_IDS), backend.INTERMEDIATE],
        "BF16 down input shape differs",
    )
    localizer.require(
        list(captures["full_bf16_down_output"].shape)
        == [len(localizer.TOKEN_IDS), backend.HIDDEN],
        "BF16 down output shape differs",
    )
    binding = {
        "top_token_id": int(torch.argmax(logits)),
        "reference_rank": localizer.rank_of(logits, localizer.REFERENCE_TOKEN),
        "captures": {
            name: {"shape": list(value.shape), "sha256": localizer.tensor_sha256(value)}
            for name, value in sorted(captures.items())
        },
    }
    del model, peft_model, output, input_ids
    gc.collect()
    return captures, binding


def _w4_dequant_reference(
    bf16_input: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale32: tuple[int, ...],
) -> torch.Tensor:
    row_products = torch.matmul(
        bf16_input.to(torch.float64), qweight.to(torch.float64).transpose(0, 1)
    )
    scales = torch.tensor(
        [_scale32_float(value) for value in weight_scale32], dtype=torch.float64
    )
    return (row_products * scales[None, :]).contiguous()


def _rtl_preflight() -> dict[str, Any]:
    rtl = ROOT / "rtl/ace2_layer16_down_proj_accumulator_requantizer_core.sv"
    top = "ace2_layer16_down_proj_accumulator_requantizer_core"
    checks = {}
    with tempfile.TemporaryDirectory(
        prefix="ace2-layer16-group4-localization-", dir=ROOT / "build"
    ) as temporary:
        executable = Path(temporary) / "preflight.vvp"
        commands = {
            "iverilog": [
                "iverilog",
                "-g2012",
                "-Wall",
                "-Wno-timescale",
                "-s",
                top,
                "-o",
                str(executable),
                str(rtl),
            ],
            "verilator": [
                "verilator",
                "--lint-only",
                "-Wall",
                "-Wno-fatal",
                "--top-module",
                top,
                str(rtl),
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
            checks[name] = {"command": command, "returncode": completed.returncode}
    return {
        "schema_version": 1,
        "status": "PASS_FROZEN_ACCEPTED_RTL_INTERFACE_COMPILES_WARNING_CLEAN",
        "top": top,
        "rtl": localizer.file_record(rtl),
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
            "down_o:s8",
            "group_scale32_o:u32",
            "channel_o:u10",
            "last_o",
            "saturation_o",
        ],
        "checks": checks,
    }


class LocalizationCache(accumulator_search.GroupedDownProjAccumulatorScale32Cache):
    """Inject one diagnostic boundary while retaining accepted W4 and group-4 logic."""

    def __init__(
        self,
        weights: Any,
        adapter: Any,
        references: dict[str, torch.Tensor],
    ) -> None:
        super().__init__(weights, adapter)
        self.references = references
        self.cut = "control"

    def configure_cut(self, cut: str) -> None:
        localizer.require(cut == "control" or cut in ORDERED_CUTS, "localization cut differs")
        self.cut = cut
        self.configure_down(DOWN_OUTPUT_GROUP_SIZE)

    def _requantize_accumulator(
        self,
        accumulator: torch.Tensor,
        input_scale32: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        group_scales, zero_groups = accumulator_search._down_group_scales_from_accumulator(
            accumulator,
            input_scale32,
            self.down_weight_scale32,
            DOWN_OUTPUT_GROUP_SIZE,
        )
        multipliers = []
        shifts = []
        for row, weight_scale32 in enumerate(self.down_weight_scale32):
            group = row // DOWN_OUTPUT_GROUP_SIZE
            if zero_groups[group]:
                localizer.require(int(accumulator[row]) == 0, "zero group has nonzero accumulator")
                multiplier, shift = 0, 0
            else:
                multiplier, shift = layer16._derive_scale32_multiplier(
                    input_scale32, weight_scale32, group_scales[group]
                )
            multipliers.append(multiplier)
            shifts.append(shift)
        multiplier = torch.tensor(multipliers, dtype=torch.int64)
        right_shift = torch.tensor(shifts, dtype=torch.int64)
        rounded = localizer.round_shift_even_tensor(accumulator * multiplier, right_shift)
        output_q = rounded.clamp(-128, 127).to(torch.int8)
        saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
        return (
            torch.tensor(group_scales, dtype=torch.int64),
            multiplier,
            right_shift,
            rounded,
            output_q,
            saturation,
        )

    def derive_projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        match = accumulator_search.DOWN_PATTERN.fullmatch(name)
        if match is None or self.cut == "control" or self.cut == "residual_sum":
            return super().derive_projection(
                name, merged, input_q, input_scale, float_input, source_hashes
            )

        position = int(match.group(1))
        metadata = self.metadata.get(merged.data_ptr())
        localizer.require(metadata is self.down_metadata, "layer-16 down metadata differs")
        qweight = metadata["qweight"]
        input_scale32 = ceil_scale32_from_float(float(input_scale))
        exact_input_scale = _scale32_float(input_scale32)
        reference_input = self.references["down_input"][position].to(torch.float64)
        reference_down = self.references["w4_dequant_down_output"][position].to(torch.float64)
        diagnostic_input_q = input_q.contiguous()
        control_accumulator = torch.mv(qweight.to(torch.int64), input_q.to(torch.int64))

        if self.cut == "input_activation":
            diagnostic_input_q = backend.canonical.quantize_int8(
                reference_input, exact_input_scale
            )
            accumulator = torch.mv(
                qweight.to(torch.int64), diagnostic_input_q.to(torch.int64)
            )
            (
                group_scales,
                multiplier,
                right_shift,
                rounded,
                output_q,
                saturation,
            ) = self._requantize_accumulator(accumulator, input_scale32)
        elif self.cut == "w4_row_dequantized_product_accumulator_formation":
            normalized = reference_input / exact_input_scale
            accumulator = torch.round(
                qweight.to(torch.float64) * normalized[None, :]
            ).sum(dim=1, dtype=torch.float64).to(torch.int64)
            (
                group_scales,
                multiplier,
                right_shift,
                rounded,
                output_q,
                saturation,
            ) = self._requantize_accumulator(accumulator, input_scale32)
        elif self.cut == "accumulator":
            normalized = reference_input / exact_input_scale
            accumulator = torch.round(
                torch.mv(qweight.to(torch.float64), normalized)
            ).to(torch.int64)
            (
                group_scales,
                multiplier,
                right_shift,
                rounded,
                output_q,
                saturation,
            ) = self._requantize_accumulator(accumulator, input_scale32)
        else:
            localizer.require(self.cut == "requantized_output", "unexpected localization cut")
            accumulator = control_accumulator
            scale_records = _group_scales_from_real(
                reference_down, DOWN_OUTPUT_GROUP_SIZE
            )
            group_scales = torch.tensor(scale_records, dtype=torch.int64)
            output_q, rounded, saturation = _quantize_grouped_real(
                reference_down, scale_records, DOWN_OUTPUT_GROUP_SIZE
            )
            multipliers = []
            shifts = []
            for row, weight_scale32 in enumerate(self.down_weight_scale32):
                multiplier, shift = layer16._derive_scale32_multiplier(
                    input_scale32,
                    weight_scale32,
                    scale_records[row // DOWN_OUTPUT_GROUP_SIZE],
                )
                multipliers.append(multiplier)
                shifts.append(shift)
            multiplier = torch.tensor(multipliers, dtype=torch.int64)
            right_shift = torch.tensor(shifts, dtype=torch.int64)

        localizer.require(
            bool(torch.all(accumulator >= -(1 << 31)))
            and bool(torch.all(accumulator < (1 << 31))),
            f"{name} diagnostic accumulator overflow",
        )
        tensor_output_scale32 = ceil_scale32_from_float(
            float(backend.canonical.scale_for(reference_down))
        )
        record = {
            "down_input_s8": diagnostic_input_q,
            "down_input_scale32": input_scale32,
            "down_weight_s4": qweight,
            "down_weight_scale32": torch.tensor(
                self.down_weight_scale32, dtype=torch.int64
            ),
            "down_accumulator_s32": accumulator,
            "down_output_group_scale32": group_scales,
            "down_multiplier_s32": multiplier,
            "down_shift_u6": right_shift,
            "down_rounded_s64": rounded,
            "down_output_s8": output_q,
            "down_saturation": saturation,
            "down_tensor_output_scale32": tensor_output_scale32,
            "down_output_group_size": DOWN_OUTPUT_GROUP_SIZE,
            "binary_down_float_output": reference_down,
            "diagnostic_reference_input_f32": reference_input.to(torch.float32),
            "diagnostic_reference_down_f64": reference_down,
            "diagnostic_cut": self.cut,
        }
        self.down_records[position] = record
        return {
            "name": name,
            "input_q": diagnostic_input_q,
            "input_scale": exact_input_scale,
            "qweight": qweight,
            "weight_scale": metadata["weight_scale"],
            "multiplier": multiplier,
            "right_shift": right_shift,
            "accumulator": accumulator,
            "rounded": rounded,
            "output_q": output_q,
            "output_scale": _scale32_float(int(group_scales[0])),
            "saturation": saturation,
            "float_output": reference_down.to(torch.float32),
            "source_hashes": source_hashes,
            "exact_grouped_down_accumulator_requantization": True,
            "diagnostic_only_substitution": self.cut,
        }


class LocalizationSourceGuard(accumulator_search.DownProjAccumulatorSourceGuard):
    """Optionally inject the W4-dequant/BF16 reference at the residual-sum boundary."""

    def __init__(
        self,
        cache: LocalizationCache,
        weights: Any,
        references: dict[str, torch.Tensor],
        cut: str,
    ) -> None:
        super().__init__(cache, weights, DOWN_OUTPUT_GROUP_SIZE)
        self.references = references
        self.cut = cut

    def _replace_residual_sum(self, position: int) -> None:
        record = self.records[position]
        attention_scales = tuple(
            int(value) for value in record["attention_group_scale32"].tolist()
        )
        binary_attention = accumulator_search._group_dequantize(
            record["attention_group_s8"],
            attention_scales,
            ATTENTION_SOURCE_GROUP_SIZE,
        )
        reference_down = self.references["w4_dequant_down_output"][position].to(
            torch.float64
        )
        binary_sum = (binary_attention + reference_down).contiguous()
        sum_scale32 = _group_scales_from_real(binary_sum, binary_sum.numel())[0]
        sum_s8, _rounded, sum_saturation = _quantize_grouped_real(
            binary_sum, (sum_scale32,), binary_sum.numel()
        )
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
        gains = derive_scaled_gains_q8(
            gain.to(torch.float64).tolist(), _scale32_float(q_input_scale32)
        )
        rms = reference_rmsnorm(sum_s8.to(torch.int64).tolist(), gains)
        record.update(
            {
                "sum_scale32": sum_scale32,
                "sum_s8": sum_s8,
                "sum_saturation": sum_saturation,
                "rms_gain_s16_q8": torch.tensor(gains, dtype=torch.int16),
                "rms_output_s8": torch.tensor(rms.outputs, dtype=torch.int8),
                "rms_sumsq": rms.sumsq,
                "rms_inv_q30": rms.inv_rms_q30,
                "rms_saturation": rms.saturation_seen,
                "q_input_scale32": q_input_scale32,
                "binary_down_scales": torch.tensor(
                    [float("nan")], dtype=torch.float32
                ),
                "binary_sum_reconstructed": binary_sum,
                "binary_input_norm": binary_input_norm,
                "binary_q_input_scale": binary_q_input_scale,
                "diagnostic_reference_down_f64": reference_down,
                "diagnostic_cut": self.cut,
            }
        )

    def install(self) -> None:
        super().install()
        derived_with_group4 = backend.derive_layer_token

        def derive(
            layer_id: int,
            state: dict[str, Any],
            cache: dict[str, Any],
            template: dict[str, Any] | None,
            weights: Any,
            adapter: Any,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            result = derived_with_group4(
                layer_id, state, cache, template, weights, adapter
            )
            if layer_id == layer16.SOURCE_LAYER and self.cut == "residual_sum":
                self._replace_residual_sum(int(state["position"]))
            return result

        backend.derive_layer_token = derive


STACKED_KEYS = (
    "down_input_s8",
    "down_accumulator_s32",
    "down_output_group_scale32",
    "down_multiplier_s32",
    "down_shift_u6",
    "down_output_s8",
    "down_saturation",
    "attention_group_s8",
    "sum_s8",
    "sum_saturation",
    "rms_output_s8",
    "q_output_s8",
    "q_saturation",
    "binary_sum_reconstructed",
)


def _bundle_records(records: dict[int, dict[str, Any]]) -> dict[str, torch.Tensor]:
    localizer.require(
        sorted(records) == list(range(len(localizer.TOKEN_IDS))),
        "localization position record set differs",
    )
    tensors = {}
    for name in STACKED_KEYS:
        tensors[name] = torch.stack(
            [records[position][name] for position in range(len(localizer.TOKEN_IDS))]
        ).contiguous()
    for name in ("down_input_scale32", "sum_scale32", "q_input_scale32"):
        tensors[name] = torch.tensor(
            [int(records[position][name]) for position in range(len(localizer.TOKEN_IDS))],
            dtype=torch.int64,
        )
    final = records[len(localizer.TOKEN_IDS) - 1]
    tensors["down_weight_s4"] = final["down_weight_s4"]
    tensors["down_weight_scale32"] = final["down_weight_scale32"]
    return tensors


def _run_variant(
    cut: str,
    cache: LocalizationCache,
    references: dict[str, torch.Tensor],
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cache.configure_cut(cut)
    source_guard = LocalizationSourceGuard(cache, weights, references, cut)
    score_guard = qoutput.QHeadScaleScoreGuard(cache)
    cache.install()
    score_guard.install()
    source_guard.install()
    try:
        result, _final_q = localizer.run_cut(
            None, [], weights, adapter, norm_gain, embedding, head, tokenizer
        )
    finally:
        source_guard.restore()
        score_guard.restore()
        cache.restore()
    expected_scores = backend.Q_HEADS * (2 * len(localizer.TOKEN_IDS) - 1)
    localizer.require(
        score_guard.replacements == expected_scores,
        "accepted layer-17 grouped Q scale did not reach every score",
    )
    bundle = _bundle_records(source_guard.records)
    final = source_guard.records[len(localizer.TOKEN_IDS) - 1]
    signature = {
        name: localizer.tensor_sha256(value) for name, value in sorted(bundle.items())
    }
    result.update(
        {
            "diagnostic_cut": cut,
            "exact_boundary": None if cut == "control" else EXACT_BOUNDARIES[cut],
            "diagnostic_only_bf16_reference": cut != "control",
            "down_output_group_size": DOWN_OUTPUT_GROUP_SIZE,
            "attention_source_group_size": ATTENTION_SOURCE_GROUP_SIZE,
            "q_output_group_size": Q_OUTPUT_GROUP_SIZE,
            "layer17_score_scale_replacements": score_guard.replacements,
            "down_input_s8_sha256": localizer.tensor_sha256(final["down_input_s8"]),
            "down_weight_s4_sha256": localizer.tensor_sha256(final["down_weight_s4"]),
            "down_weight_scale32_sha256": localizer.tensor_sha256(
                final["down_weight_scale32"]
            ),
            "down_accumulator_s32_sha256": localizer.tensor_sha256(
                final["down_accumulator_s32"]
            ),
            "down_output_group_scale32_sha256": localizer.tensor_sha256(
                final["down_output_group_scale32"]
            ),
            "down_output_s8_sha256": localizer.tensor_sha256(final["down_output_s8"]),
            "sum_s8_sha256": localizer.tensor_sha256(final["sum_s8"]),
            "q_output_s8_sha256": localizer.tensor_sha256(final["q_output_s8"]),
            "down_saturation_count": int(final["down_saturation"].sum()),
            "sum_saturation_count": int(final["sum_saturation"].sum()),
            "q_saturation_count": int(final["q_saturation"].sum()),
            "reproducibility_signature": signature,
        }
    )
    return result, bundle


def _write_variant(
    directory: Path,
    metrics: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> None:
    artifacts = {
        name: localizer.write_tensor(directory / f"{name}.bin", value)
        for name, value in sorted(tensors.items())
    }
    localizer.write_json(directory / "result.json", {"metrics": metrics, "artifacts": artifacts})


def _qualifies(record: dict[str, Any], control_rank: int) -> bool:
    return bool(
        int(record["top_token_id"]) == localizer.REFERENCE_TOKEN
        or control_rank - int(record["reference_rank"]) >= MATERIAL_RANK_RECOVERY
    )


def _failure(cut: str, error: Exception) -> dict[str, Any]:
    return {
        "diagnostic_cut": cut,
        "classification": "DIAGNOSTIC_CUT_EXECUTION_FAILURE",
        "detail": str(error),
        "root_cause_hypothesis": (
            "The diagnostic reference could not be consumed at the frozen layer-16 "
            "down-projection boundary without violating the accepted downstream contract."
        ),
        "regression": (
            "Preserve this failed candidate and require the same cut to execute before "
            "drawing any causal conclusion from later boundaries."
        ),
    }


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
            f"protected hash differs before localization: {relative}",
        )

    predecessor_result_path = PREDECESSOR / "result.json"
    predecessor_freeze_path = PREDECESSOR / "candidate-freeze.json"
    predecessor_result = json.loads(predecessor_result_path.read_text(encoding="utf-8"))
    predecessor_freeze = json.loads(predecessor_freeze_path.read_text(encoding="utf-8"))
    search_review = json.loads(PREDECESSOR_SEARCH_REVIEW.read_text(encoding="utf-8"))
    rtl_review = json.loads(PREDECESSOR_RTL_REVIEW.read_text(encoding="utf-8"))
    reviewer_handoff = json.loads(PREDECESSOR_REVIEW_HANDOFF.read_text(encoding="utf-8"))
    manifest_members = _verify_manifest(PREDECESSOR)
    localizer.require(
        predecessor_result["selected_down_output_group_size"] == DOWN_OUTPUT_GROUP_SIZE
        and predecessor_result["selected_reference_rank"] == 31879
        and predecessor_result["selected_top_token_id"] == 37865,
        "accepted predecessor selection differs",
    )
    localizer.require(
        search_review["selected_down_output_group_size"] == DOWN_OUTPUT_GROUP_SIZE
        and search_review["selected_reference_rank"] == 31879
        and search_review["selected_top_token_id"] == 37865,
        "Fresh-L2 search acceptance differs",
    )
    localizer.require(
        rtl_review["status"]
        == "PASS_FOCUSED_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANTIZATION_RTL_REFERENCE_AGREEMENT",
        "Fresh-L2 focused RTL acceptance is absent",
    )
    localizer.require(
        reviewer_handoff["producer_role"] == "reviewer"
        and reviewer_handoff["review"]["status"] == "done"
        and "31,879" in reviewer_handoff["review"]["reason"],
        "sealed Fresh-L2 Reviewer did not accept rank 31,879",
    )
    localizer.require(
        predecessor_freeze["prompt"]["chat_token_ids"] == localizer.TOKEN_IDS
        and predecessor_freeze["prompt"]["chat_token_count"] == 30,
        "canonical 30-token prefix differs",
    )
    localizer.require(
        predecessor_freeze["baseline"]["layer16_attention_source_group_size"]
        == ATTENTION_SOURCE_GROUP_SIZE
        and predecessor_freeze["baseline"]["layer17_q_output_group_size"]
        == Q_OUTPUT_GROUP_SIZE,
        "fixed attention/Q dependencies differ",
    )

    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    localizer.require(
        generation_runner.canonical_chat_token_ids(tokenizer, "Hello")
        == localizer.TOKEN_IDS,
        "canonical tokenizer prefix differs",
    )

    output.mkdir(parents=True)
    freeze = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "mission": MISSION,
        "classification": "candidate_only_diagnostic_no_official_attempt_created",
        "checkpoint": predecessor_freeze["checkpoint"],
        "base_model": {
            "repository": generation_runner.MODEL_REPOSITORY,
            "revision": generation_runner.REVISION,
            "model": localizer.file_record(snapshot / "model.safetensors"),
            "config": localizer.file_record(snapshot / "config.json"),
        },
        "adapter": {
            "model": localizer.file_record(backend.canonical.ADAPTER),
            "config": localizer.file_record(backend.canonical.ADAPTER_CONFIG),
        },
        "tokenizer": _tokenizer_records(snapshot),
        "prompt": predecessor_freeze["prompt"],
        "reference_step0_token_id": localizer.REFERENCE_TOKEN,
        "accepted_control": {
            "top_token_id": 37865,
            "reference_rank": 31879,
            "down_output_group_size": DOWN_OUTPUT_GROUP_SIZE,
            "layer16_attention_source_group_size": ATTENTION_SOURCE_GROUP_SIZE,
            "layer17_q_output_group_size": Q_OUTPUT_GROUP_SIZE,
            "expected_final_position_hashes": EXPECTED_CONTROL_HASHES,
        },
        "ordered_diagnostic_cuts": list(ORDERED_CUTS),
        "exact_boundaries": EXACT_BOUNDARIES,
        "material_rank_recovery": {
            "minimum_places": MATERIAL_RANK_RECOVERY,
            "basis": (
                "the independently accepted predecessor improvement magnitude "
                "33375 -> 31879"
            ),
        },
        "method": {
            "definition": (
                "Capture the full-BF16 layer-16 SiLU/down-projection input for the exact "
                "30-token prefix. Retain the accepted signed-W4 bytes and per-row Scale32. "
                "At each ordered cut, defer rounding through exactly one later boundary: "
                "input A8, per-W4-product accumulator formation, row accumulator, group-4 "
                "requantized output, then accepted-attention-plus-reference-down residual sum."
            ),
            "cuts_are_isolated": True,
            "diagnostic_only": True,
            "bf16_runtime_payload": False,
            "token_or_logit_override": False,
            "hard_coded_output": False,
        },
        "rtl_public_contract": predecessor_freeze["rtl_public_contract"],
        "tool_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "protected_hashes": PROTECTED_HASHES,
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "rtl_repair_implemented": False,
            "u280_or_stage2_entered": False,
        },
        "bindings": {
            "predecessor_freeze": localizer.file_record(predecessor_freeze_path),
            "predecessor_result": localizer.file_record(predecessor_result_path),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
            "predecessor_manifest_members_verified": manifest_members,
            "fresh_l2_search": localizer.file_record(PREDECESSOR_SEARCH_REVIEW),
            "fresh_l2_rtl": localizer.file_record(PREDECESSOR_RTL_REVIEW),
            "fresh_l2_reviewer_handoff": _external_record(PREDECESSOR_REVIEW_HANDOFF),
            "accepted_layer16_sha256s": predecessor_result["bindings"]["accepted_layer16_sha256s"],
            "accepted_q_sha256s": predecessor_result["bindings"]["accepted_q_sha256s"],
            "runner": localizer.file_record(Path(__file__).resolve()),
            "accepted_rtl": localizer.file_record(
                ROOT / "rtl/ace2_layer16_down_proj_accumulator_requantizer_core.sv"
            ),
        },
    }
    localizer.write_json(output / "candidate-freeze.json", freeze)
    _event(output, "freeze_written", freeze_sha256=_sha256(output / "candidate-freeze.json"))
    localizer.write_json(output / "rtl-interface-preflight.json", _rtl_preflight())
    _event(
        output,
        "rtl_interface_preflight_complete",
        result_sha256=_sha256(output / "rtl-interface-preflight.json"),
    )

    started = time.monotonic()
    captures, capture_binding = _capture_bf16_boundaries(snapshot)
    with safe_open(
        backend.canonical.MODEL, framework="pt", device="cpu"
    ) as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        cache = LocalizationCache(weights, adapter, {})
        qweight = cache.down_metadata["qweight"]
        localizer.require(
            localizer.tensor_sha256(qweight) == EXPECTED_CONTROL_HASHES["down_weight_s4"],
            "accepted down W4 bytes differ before reference construction",
        )
        localizer.require(
            localizer.tensor_sha256(
                torch.tensor(cache.down_weight_scale32, dtype=torch.int64)
            )
            == EXPECTED_CONTROL_HASHES["down_weight_scale32"],
            "accepted down Scale32 rows differ before reference construction",
        )
        captures["w4_dequant_down_output"] = _w4_dequant_reference(
            captures["down_input"], qweight, cache.down_weight_scale32
        )
        cache.references = captures

        reference_artifacts = {
            name: localizer.write_tensor(output / f"reference/{name}.bin", value)
            for name, value in sorted(captures.items())
        }
        reference_record = {
            "schema_version": 1,
            "definition": (
                "Full-BF16 layer-16 SiLU input plus an offline reference down projection "
                "formed only from the accepted signed-W4 bytes and per-row Scale32 records."
            ),
            "binding": capture_binding,
            "accepted_w4_sha256": localizer.tensor_sha256(qweight),
            "accepted_weight_scale32_sha256": localizer.tensor_sha256(
                torch.tensor(cache.down_weight_scale32, dtype=torch.int64)
            ),
            "artifacts": reference_artifacts,
        }
        localizer.write_json(output / "reference/result.json", reference_record)
        _event(
            output,
            "reference_captured",
            down_input_sha256=localizer.tensor_sha256(captures["down_input"]),
            w4_dequant_down_sha256=localizer.tensor_sha256(
                captures["w4_dequant_down_output"]
            ),
        )

        control, control_tensors = _run_variant(
            "control", cache, captures, weights, adapter, norm_gain, embedding, head, tokenizer
        )
        localizer.require(
            control["top_token_id"] == 37865 and control["reference_rank"] == 31879,
            "checkpoint-176 group-4 control outcome differs",
        )
        for name, expected in EXPECTED_CONTROL_HASHES.items():
            localizer.require(
                control[f"{name}_sha256"] == expected,
                f"checkpoint-176 group-4 control hash differs: {name}",
            )
        _write_variant(output / "runs/control", control, control_tensors)
        _event(
            output,
            "control_complete",
            top_token_id=control["top_token_id"],
            reference_rank=control["reference_rank"],
        )
        print(
            "ACE2_LAYER16_DOWN_PROJ_LOCALIZATION "
            f"cut=control top={control['top_token_id']} rank={control['reference_rank']}",
            flush=True,
        )

        trace = []
        failures = []
        selected = None
        selected_index = None
        variant_by_cut = {"control": control}
        for index, cut in enumerate(ORDERED_CUTS):
            try:
                record, tensors = _run_variant(
                    cut,
                    cache,
                    captures,
                    weights,
                    adapter,
                    norm_gain,
                    embedding,
                    head,
                    tokenizer,
                )
            except Exception as error:
                failure = _failure(cut, error)
                localizer.write_json(output / f"runs/{index + 1:02d}-{cut}/failure.json", failure)
                failures.append(failure)
                _event(output, "cut_failed", cut=cut, detail=str(error))
                break
            record["reference_rank_change"] = int(control["reference_rank"]) - int(
                record["reference_rank"]
            )
            record["restored_token_9707"] = (
                int(record["top_token_id"]) == localizer.REFERENCE_TOKEN
            )
            record["meets_material_rank_recovery"] = (
                int(record["reference_rank_change"]) >= MATERIAL_RANK_RECOVERY
            )
            _write_variant(output / f"runs/{index + 1:02d}-{cut}", record, tensors)
            trace.append(record)
            variant_by_cut[cut] = record
            _event(
                output,
                "cut_complete",
                cut=cut,
                top_token_id=record["top_token_id"],
                reference_rank=record["reference_rank"],
                reference_rank_change=record["reference_rank_change"],
            )
            print(
                "ACE2_LAYER16_DOWN_PROJ_LOCALIZATION "
                f"cut={cut} top={record['top_token_id']} "
                f"rank={record['reference_rank']} delta={record['reference_rank_change']}",
                flush=True,
            )
            if _qualifies(record, int(control["reference_rank"])):
                selected = record
                selected_index = index
                break

        confirmation = None
        if selected is not None and selected_index is not None:
            selected_cut = ORDERED_CUTS[selected_index]
            adjacent_cut = "control" if selected_index == 0 else ORDERED_CUTS[selected_index - 1]
            repeat, repeat_tensors = _run_variant(
                selected_cut,
                cache,
                captures,
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
            )
            adjacent, adjacent_tensors = _run_variant(
                adjacent_cut,
                cache,
                captures,
                weights,
                adapter,
                norm_gain,
                embedding,
                head,
                tokenizer,
            )
            _write_variant(output / "confirmation/selected-repeat", repeat, repeat_tensors)
            _write_variant(output / "confirmation/adjacent-repeat", adjacent, adjacent_tensors)
            localizer.require(
                repeat["top_token_id"] == selected["top_token_id"]
                and repeat["reference_rank"] == selected["reference_rank"]
                and repeat["reproducibility_signature"]
                == selected["reproducibility_signature"],
                "selected localization cut did not reproduce exactly",
            )
            adjacent_expected = variant_by_cut[adjacent_cut]
            localizer.require(
                adjacent["top_token_id"] == adjacent_expected["top_token_id"]
                and adjacent["reference_rank"] == adjacent_expected["reference_rank"]
                and adjacent["reproducibility_signature"]
                == adjacent_expected["reproducibility_signature"],
                "adjacent localization control did not reproduce exactly",
            )
            localizer.require(
                not _qualifies(adjacent, int(control["reference_rank"])),
                "adjacent earlier boundary also qualifies; earliest boundary is ambiguous",
            )
            confirmation = {
                "selected_cut": selected_cut,
                "adjacent_cut": adjacent_cut,
                "selected_repeat": {
                    "top_token_id": repeat["top_token_id"],
                    "reference_rank": repeat["reference_rank"],
                    "reproducibility_signature": repeat["reproducibility_signature"],
                },
                "adjacent_repeat": {
                    "top_token_id": adjacent["top_token_id"],
                    "reference_rank": adjacent["reference_rank"],
                    "reproducibility_signature": adjacent["reproducibility_signature"],
                },
            }
            _event(
                output,
                "boundary_confirmed",
                selected_cut=selected_cut,
                adjacent_cut=adjacent_cut,
            )

    for relative, expected in PROTECTED_HASHES.items():
        localizer.require(
            _sha256(ROOT / relative) == expected,
            f"protected hash differs after localization: {relative}",
        )

    result = {
        "schema_version": 1,
        "status": (
            "PASS_EARLIEST_LAYER16_DOWN_PROJ_BOUNDARY_LOCALIZED"
            if selected is not None and confirmation is not None
            else "BLOCKED_DIAGNOSTIC_CUT_EXECUTION_FAILURE"
            if failures
            else "PARTIAL_NO_MATERIAL_BOUNDARY_WITHIN_BOUNDED_SEQUENCE"
        ),
        "classification": "candidate_only_diagnostic_no_official_attempt_created",
        "control": control,
        "ordered_cuts": list(ORDERED_CUTS),
        "trace": trace,
        "failures": failures,
        "material_rank_recovery_minimum": MATERIAL_RANK_RECOVERY,
        "earliest_qualifying_cut": (
            selected["diagnostic_cut"] if selected is not None else None
        ),
        "next_exact_repair_boundary": (
            selected["exact_boundary"] if selected is not None else None
        ),
        "selection_reason": (
            "restored_token_9707"
            if selected is not None and selected["restored_token_9707"]
            else "material_rank_recovery"
            if selected is not None
            else "none"
        ),
        "confirmation": confirmation,
        "diagnostic_bf16_is_not_hardware_repair": True,
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "rtl_repair_implemented": False,
        "u280_or_stage2_entered": False,
        "elapsed_wall_seconds": time.monotonic() - started,
        "bindings": {
            "freeze": localizer.file_record(output / "candidate-freeze.json"),
            "rtl_interface_preflight": localizer.file_record(
                output / "rtl-interface-preflight.json"
            ),
            "reference": localizer.file_record(output / "reference/result.json"),
            "events": localizer.file_record(output / "execution-events.jsonl"),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
            "fresh_l2_search": localizer.file_record(PREDECESSOR_SEARCH_REVIEW),
            "fresh_l2_rtl": localizer.file_record(PREDECESSOR_RTL_REVIEW),
        },
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    print(
        "ACE2_LAYER16_DOWN_PROJ_LOCALIZATION_RESULT "
        f"status={result['status']} boundary={result['next_exact_repair_boundary']}",
        flush=True,
    )
    return 0 if selected is not None and confirmation is not None else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ACE2_LAYER16_DOWN_PROJ_LOCALIZATION_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
