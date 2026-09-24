#!/usr/bin/env python3
"""Localize the first remaining checkpoint-176 error after layer-16 group-1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import traceback
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
from tools import ace2_checkpoint176_layer16_down_proj_group4_remaining_error_localization as downlocal
from tools import ace2_checkpoint176_layer16_post_attention_residual_sum_scale_alignment_search as postattn
from tools import ace2_checkpoint176_layer16_source_grouped_scale32_search as layer16
from tools import ace2_checkpoint176_layer17_q_output_grouped_scale32_search as qoutput
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner
from tools.ace2_quality_contracts import ceil_scale32_from_float, scale32_ratio
from tools.ace2_rmsnorm_reference import derive_scaled_gains_q8, reference_rmsnorm


MISSION = "w4a8-layer16-post-attention-group1-remaining-error-localization-v1"
ATTENTION_OUTPUT_GROUP_SIZE = 1
ATTENTION_SOURCE_GROUP_SIZE = 1
DOWN_OUTPUT_GROUP_SIZE = 4
Q_OUTPUT_GROUP_SIZE = 128
BASELINE_RANK = 3252
BASELINE_TOP_TOKEN = 12
ORDERED_CUTS = (
    "post_attention_rmsnorm",
    "gate_up_projections",
    "silu_product",
    "down_proj_accumulator",
    "down_proj_output",
    "layer16_final_residual",
    "layer17_input_rmsnorm",
    "layer17_q_output",
)
EXACT_BOUNDARIES = {
    "post_attention_rmsnorm": "model.layers.16.post_attention_layernorm.output_to_signed_a8",
    "gate_up_projections": "model.layers.16.mlp.gate_up_proj.output_to_signed_a8",
    "silu_product": "model.layers.16.mlp.silu_product.output_to_signed_a8",
    "down_proj_accumulator": "model.layers.16.mlp.down_proj.accumulator",
    "down_proj_output": "model.layers.16.mlp.down_proj.output_to_signed_a8",
    "layer16_final_residual": "model.layers.16.mlp.output_to_final_residual_sum",
    "layer17_input_rmsnorm": "model.layers.17.input_layernorm.output_to_signed_a8",
    "layer17_q_output": "model.layers.17.self_attn.q_proj.output_to_group128_signed_a8",
}
PREDECESSOR = ROOT / (
    "evidence/candidates/w4a8-layer16-post-attention-residual-sum-"
    "scale-alignment-repair-v1/candidate-0002"
)
FRESH_SEARCH = ROOT / (
    "build/l2-review-layer16-post-attention-residual-sum-scale-alignment-r1/result.json"
)
FRESH_RTL = ROOT / (
    "build/l2-review-layer16-post-attention-residual-sum-scale-alignment-focused-r1/result.json"
)
FRESH_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "w4a8-layer16-post-attention-residual-sum-scale-alignment-repair-v1/round-0001.json"
)
PROTECTED_HASHES = dict(postattn.PROTECTED_HASHES)
PROJECTION_PATTERN = re.compile(r"layer(?P<layer>16|17)_position(?P<position>\d+)_(?P<key>q|k|v|gate|up|down)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _verify_manifest(directory: Path) -> int:
    count = 0
    for line in (directory / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        localizer.require(
            _sha256(directory / relative) == expected,
            f"sealed predecessor manifest mismatch: {relative}",
        )
        count += 1
    localizer.require(count > 0, "sealed predecessor manifest is empty")
    return count


def _scale32_float(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def _representable_rms_scale32(gain: torch.Tensor, output: torch.Tensor) -> int:
    scale = backend.canonical.derive_rmsnorm_output_scale(
        gain.to(torch.float64).tolist(),
        float(output.to(torch.float64).abs().max()),
    )
    scale32 = ceil_scale32_from_float(math.nextafter(scale, math.inf))
    try:
        derive_scaled_gains_q8(
            gain.to(torch.float64).tolist(), _scale32_float(scale32)
        )
    except OverflowError:
        scale32 = ceil_scale32_from_float(scale * (1.0 + 2.0**-20))
        derive_scaled_gains_q8(
            gain.to(torch.float64).tolist(), _scale32_float(scale32)
        )
    return scale32


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
    record = {"created_at_utc": datetime.now(UTC).isoformat(), "kind": kind, **fields}
    with (output / "execution-events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _selected_trace(result: dict[str, Any]) -> dict[str, Any]:
    selected = [
        record
        for record in result["trace"]
        if int(record["output_group_size"]) == ATTENTION_OUTPUT_GROUP_SIZE
    ]
    localizer.require(len(selected) == 1, "selected group-1 trace is not unique")
    return selected[0]


def _gate_predecessor() -> tuple[dict[str, Any], dict[str, Any], int]:
    predecessor = json.loads((PREDECESSOR / "result.json").read_text(encoding="utf-8"))
    predecessor_freeze = json.loads(
        (PREDECESSOR / "candidate-freeze.json").read_text(encoding="utf-8")
    )
    fresh_search = json.loads(FRESH_SEARCH.read_text(encoding="utf-8"))
    fresh_rtl = json.loads(FRESH_RTL.read_text(encoding="utf-8"))
    handoff = json.loads(FRESH_HANDOFF.read_text(encoding="utf-8"))
    members = _verify_manifest(PREDECESSOR)
    for record in (predecessor, fresh_search):
        localizer.require(
            int(record["baseline_reference_rank"]) == 31879
            and int(record["baseline_top_token_id"]) == 37865
            and int(record["selected_output_group_size"]) == ATTENTION_OUTPUT_GROUP_SIZE
            and int(record["selected_reference_rank"]) == BASELINE_RANK
            and int(record["selected_top_token_id"]) == BASELINE_TOP_TOKEN,
            "Fresh-L2 control or selected group-1 outcome differs",
        )
        selected = _selected_trace(record)
        localizer.require(
            int(selected["post_attention_sum_saturation_count"]) == 0,
            "selected group-1 model-vector saturation differs",
        )
    localizer.require(
        fresh_rtl["status"]
        == "PASS_FOCUSED_LAYER16_POST_ATTENTION_RESIDUAL_SUM_SCALE_ALIGNMENT_SELECTED_OR_BEST_NULL_RTL_REFERENCE_AGREEMENT",
        "Fresh-L2 focused RTL acceptance is absent",
    )
    metrics = fresh_rtl["metrics"]
    checks = fresh_rtl["checks"]
    localizer.require(
        int(metrics["groups"]) == 896
        and int(metrics["outputs"]) == 896
        and int(metrics["saturations"]) == 0
        and int(metrics["directed"]) == 9,
        "Fresh-L2 focused RTL vector/arithmetic counts differ",
    )
    for name in (
        "cycle_accurate_model_vector_match",
        "signed_ties_to_even",
        "positive_and_negative_saturation",
        "asynchronous_reset_recovery",
        "synchronous_clear_recovery",
        "held_valid_input_backpressure",
        "output_backpressure_and_stability",
        "same_edge_consume_and_replace",
        "iverilog_warning_clean",
        "verilator_warning_clean",
        "xz_clean_every_checked_cycle",
    ):
        localizer.require(bool(checks[name]), f"Fresh-L2 focused RTL check differs: {name}")
    localizer.require(
        handoff["producer_role"] == "reviewer"
        and handoff["review"]["status"] == "done"
        and "31,879" in handoff["review"]["reason"]
        and "3,252" in handoff["review"]["reason"]
        and "896" in handoff["review"]["reason"],
        "sealed Fresh-L2 Reviewer acceptance differs",
    )
    localizer.require(
        predecessor_freeze["prompt"]["chat_token_ids"] == localizer.TOKEN_IDS
        and int(predecessor_freeze["prompt"]["chat_token_count"]) == 30,
        "canonical 30-token prefix differs",
    )
    return predecessor, predecessor_freeze, members


def _capture_reference_boundaries(
    snapshot: Path,
    cache: downlocal.LocalizationCache,
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
    captures: dict[str, torch.Tensor] = {}
    layer_16 = model.model.layers[layer16.SOURCE_LAYER]
    layer_17 = model.model.layers[layer16.TARGET_LAYER]

    def capture_output(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            localizer.require(isinstance(value, torch.Tensor), f"BF16 capture differs: {name}")
            captures[name] = value[0].float().cpu().contiguous()

        return hook

    handles = [
        layer_16.post_attention_layernorm.register_forward_hook(
            capture_output("post_attention_rmsnorm")
        ),
        layer_16.register_forward_hook(capture_output("layer16_final_residual")),
        layer_17.input_layernorm.register_forward_hook(
            capture_output("layer17_input_rmsnorm")
        ),
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
    localizer.require(
        int(torch.argmax(logits)) == localizer.REFERENCE_TOKEN,
        "full BF16 reference first token changed",
    )

    def dequant_projection(name: str, activation: torch.Tensor) -> torch.Tensor:
        merged = cache.merged[name]
        metadata = cache.metadata[merged.data_ptr()]
        qweight = metadata["qweight"].to(torch.float64)
        if name.endswith("mlp.down_proj"):
            scales = torch.tensor(
                [_scale32_float(value) for value in cache.down_weight_scale32],
                dtype=torch.float64,
            )
        elif name.endswith("self_attn.q_proj") and name.startswith("model.layers.17"):
            scales = torch.tensor(
                [_scale32_float(value) for value in cache.q_weight_scale32],
                dtype=torch.float64,
            )
        else:
            scales = metadata["weight_scale"].to(torch.float64)
        return (
            torch.matmul(activation.to(torch.float64), qweight.transpose(0, 1))
            * scales[None, :]
        ).contiguous()

    captures["gate_projection"] = dequant_projection(
        "model.layers.16.mlp.gate_proj", captures["post_attention_rmsnorm"]
    )
    captures["up_projection"] = dequant_projection(
        "model.layers.16.mlp.up_proj", captures["post_attention_rmsnorm"]
    )
    captures["silu_product"] = (
        torch.nn.functional.silu(captures["gate_projection"])
        * captures["up_projection"]
    ).contiguous()
    captures["down_input"] = captures["silu_product"]
    captures["w4_dequant_down_output"] = dequant_projection(
        "model.layers.16.mlp.down_proj", captures["silu_product"]
    )
    captures["layer17_q_output"] = dequant_projection(
        "model.layers.17.self_attn.q_proj", captures["layer17_input_rmsnorm"]
    )
    expected_shapes = {
        "post_attention_rmsnorm": [30, backend.HIDDEN],
        "gate_projection": [30, backend.INTERMEDIATE],
        "up_projection": [30, backend.INTERMEDIATE],
        "silu_product": [30, backend.INTERMEDIATE],
        "w4_dequant_down_output": [30, backend.HIDDEN],
        "layer16_final_residual": [30, backend.HIDDEN],
        "layer17_input_rmsnorm": [30, backend.HIDDEN],
        "layer17_q_output": [30, backend.HIDDEN],
    }
    for name, shape in expected_shapes.items():
        localizer.require(list(captures[name].shape) == shape, f"reference shape differs: {name}")
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


class BoundaryCache(downlocal.LocalizationCache):
    """Inject one ordered reference boundary while preserving group-1/4/128 logic."""

    def __init__(self, weights: Any, adapter: Any, references: dict[str, torch.Tensor]) -> None:
        super().__init__(weights, adapter, references)
        self.logical_cut = "control"

    def configure_boundary(self, cut: str) -> None:
        localizer.require(cut == "control" or cut in ORDERED_CUTS, "boundary cut differs")
        self.logical_cut = cut
        mapped = {
            "silu_product": "input_activation",
            "down_proj_accumulator": "accumulator",
            "down_proj_output": "requantized_output",
        }.get(cut, "control")
        super().configure_cut(mapped)

    def _replace_input(
        self, name: str, input_q: torch.Tensor, input_scale: float
    ) -> torch.Tensor:
        match = PROJECTION_PATTERN.fullmatch(name)
        if match is None:
            return input_q
        layer_id = int(match.group("layer"))
        position = int(match.group("position"))
        key = match.group("key")
        if (
            self.logical_cut == "post_attention_rmsnorm"
            and layer_id == 16
            and key in {"gate", "up"}
        ):
            return backend.canonical.quantize_int8(
                self.references["post_attention_rmsnorm"][position], input_scale
            )
        if (
            self.logical_cut == "layer17_input_rmsnorm"
            and layer_id == 17
            and key in {"k", "v"}
        ):
            return backend.canonical.quantize_int8(
                self.references["layer17_input_rmsnorm"][position], input_scale
            )
        return input_q

    def _replace_output(self, name: str, result: dict[str, Any]) -> dict[str, Any]:
        match = PROJECTION_PATTERN.fullmatch(name)
        if match is None:
            return result
        layer_id = int(match.group("layer"))
        position = int(match.group("position"))
        key = match.group("key")
        if (
            self.logical_cut == "gate_up_projections"
            and layer_id == 16
            and key in {"gate", "up"}
        ):
            reference = self.references[f"{key}_projection"][position]
            scale = float(result["output_scale"])
            rounded = torch.round(reference.to(torch.float64) / scale).to(torch.int64)
            output_q = rounded.clamp(-128, 127).to(torch.int8)
            result.update(
                {
                    "rounded": rounded,
                    "output_q": output_q,
                    "saturation": ((rounded < -128) | (rounded > 127)).to(torch.uint8),
                    "diagnostic_only_substitution": self.logical_cut,
                }
            )
        if self.logical_cut == "layer17_q_output" and layer_id == 17 and key == "q":
            record = self.position_records[position]
            scales = tuple(int(value) for value in record["q_output_group_scale32"].tolist())
            output_q, rounded, saturation = downlocal._quantize_grouped_real(
                self.references["layer17_q_output"][position].to(torch.float64),
                scales,
                Q_OUTPUT_GROUP_SIZE,
            )
            record.update(
                {
                    "q_output_s8": output_q,
                    "q_rounded_s64": rounded,
                    "q_saturation": saturation,
                    "diagnostic_cut": self.logical_cut,
                }
            )
            result.update(
                {
                    "output_q": output_q,
                    "rounded": rounded,
                    "saturation": saturation,
                    "diagnostic_only_substitution": self.logical_cut,
                }
            )
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
        input_q = self._replace_input(name, input_q, input_scale)
        result = super().derive_projection(
            name, merged, input_q, input_scale, float_input, source_hashes
        )
        return self._replace_output(name, result)

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
        input_q = self._replace_input(name, input_q, input_scale)
        result = super().from_fixed_metadata(
            name,
            input_q,
            input_scale,
            qweight,
            multiplier,
            right_shift,
            output_scale,
        )
        return self._replace_output(name, result)


class BoundarySourceGuard(postattn.PostAttentionSourceGuard):
    def __init__(
        self,
        cache: BoundaryCache,
        weights: Any,
        alignment: postattn.PostAttentionAlignmentHook,
        references: dict[str, torch.Tensor],
        cut: str,
    ) -> None:
        super().__init__(cache, weights, alignment)
        self.references = references
        self.cut = cut

    def _replace_record(self, position: int) -> None:
        record = self.records[position]
        boundary = self.alignment.records[position]
        boundary_scales = tuple(
            int(value)
            for value in boundary["post_attention_output_group_scale32"].tolist()
        )
        boundary_values = postattn._real_values(
            boundary["post_attention_sum_s8"],
            boundary_scales,
            self.alignment.output_group_size,
        )
        attention_scales = postattn.rmsinput._group_scales(
            boundary_values, ATTENTION_SOURCE_GROUP_SIZE
        )
        attention_group_s8, attention_saturation = postattn.rmsinput._quantize_groups(
            boundary_values, attention_scales, ATTENTION_SOURCE_GROUP_SIZE
        )
        down_group_s8 = record["down_output_s8"]
        down_scales = tuple(
            int(value) for value in record["down_output_group_scale32"].tolist()
        )
        binary_attention = postattn.accumulator._group_dequantize(
            attention_group_s8, attention_scales, ATTENTION_SOURCE_GROUP_SIZE
        )
        binary_down = postattn.accumulator._group_dequantize(
            down_group_s8, down_scales, DOWN_OUTPUT_GROUP_SIZE
        )
        binary_sum = (binary_attention + binary_down).contiguous()
        gain = self.weights.get_tensor(
            f"model.layers.{layer16.TARGET_LAYER}.input_layernorm.weight"
        ).contiguous()
        binary_input_norm = backend.canonical.float_rmsnorm(
            binary_sum.to(torch.float32), gain
        ).contiguous()
        q_input_scale32 = _representable_rms_scale32(gain, binary_input_norm)
        binary_q_input_scale = _scale32_float(q_input_scale32)
        final_values = postattn.downoutput._sum_reals(
            attention_group_s8,
            down_group_s8,
            attention_scales,
            down_scales,
            DOWN_OUTPUT_GROUP_SIZE,
        )
        sum_scale32 = postattn.downoutput._tensor_scale(final_values)
        sum_s8, sum_saturation = postattn.downoutput._quantize_values(
            final_values, sum_scale32
        )
        gains = derive_scaled_gains_q8(
            gain.to(torch.float64).tolist(), _scale32_float(q_input_scale32)
        )
        rms = backend.canonical.reference_rmsnorm(
            sum_s8.to(torch.int64).tolist(), gains
        )
        record.update(boundary)
        record.update(
            {
                "attention_source_group_size": ATTENTION_SOURCE_GROUP_SIZE,
                "attention_source_s8": boundary["post_attention_sum_s8"],
                "attention_base_scale32": int(boundary_scales[0]),
                "attention_group_scale32": torch.tensor(
                    attention_scales, dtype=torch.int64
                ),
                "attention_group_s8": attention_group_s8,
                "attention_saturation": attention_saturation,
                "down_group_s8": down_group_s8,
                "down_group_scale32": record["down_output_group_scale32"],
                "sum_scale32": sum_scale32,
                "sum_s8": sum_s8,
                "sum_saturation": sum_saturation,
                "rms_gain_s16_q8": torch.tensor(gains, dtype=torch.int16),
                "rms_output_s8": torch.tensor(rms.outputs, dtype=torch.int8),
                "rms_sumsq": rms.sumsq,
                "rms_inv_q30": rms.inv_rms_q30,
                "rms_saturation": rms.saturation_seen,
                "q_input_scale32": q_input_scale32,
                "binary_attention_q": attention_group_s8,
                "binary_down_q": down_group_s8,
                "binary_attention_scales": torch.tensor(
                    [_scale32_float(value) for value in attention_scales],
                    dtype=torch.float32,
                ),
                "binary_down_scales": torch.tensor(
                    [_scale32_float(value) for value in down_scales],
                    dtype=torch.float32,
                ),
                "binary_sum_reconstructed": binary_sum,
                "binary_input_norm": binary_input_norm,
                "binary_q_input_scale": binary_q_input_scale,
            }
        )

    def _replace_final_residual(
        self, position: int, next_state: dict[str, Any]
    ) -> None:
        reference = self.references["layer16_final_residual"][position].to(torch.float32)
        scale = float(next_state["fixed_scale"])
        scale32 = ceil_scale32_from_float(scale)
        output_q = backend.canonical.quantize_int8(reference, _scale32_float(scale32))
        gain = self.weights.get_tensor(
            f"model.layers.{layer16.TARGET_LAYER}.input_layernorm.weight"
        ).contiguous()
        binary_input_norm = backend.canonical.float_rmsnorm(reference, gain).contiguous()
        q_input_scale32 = _representable_rms_scale32(gain, binary_input_norm)
        gains = derive_scaled_gains_q8(
            gain.to(torch.float64).tolist(), _scale32_float(q_input_scale32)
        )
        rms = reference_rmsnorm(output_q.to(torch.int64).tolist(), gains)
        record = self.records[position]
        record.update(
            {
                "sum_scale32": scale32,
                "sum_s8": output_q,
                "sum_saturation": torch.zeros(output_q.numel(), dtype=torch.uint8),
                "binary_sum_reconstructed": reference.to(torch.float64),
                "binary_input_norm": binary_input_norm,
                "binary_q_input_scale": _scale32_float(q_input_scale32),
                "q_input_scale32": q_input_scale32,
                "rms_gain_s16_q8": torch.tensor(gains, dtype=torch.int16),
                "rms_output_s8": torch.tensor(rms.outputs, dtype=torch.int8),
                "rms_sumsq": rms.sumsq,
                "rms_inv_q30": rms.inv_rms_q30,
                "rms_saturation": rms.saturation_seen,
                "diagnostic_cut": self.cut,
            }
        )
        next_state.update(
            {
                "fixed_q": output_q,
                "fixed_scale": _scale32_float(scale32),
                "float_hidden": reference,
            }
        )

    def _replace_layer17_input_rmsnorm(self, position: int) -> None:
        record = self.records[position]
        scale32 = int(record["q_input_scale32"])
        reference = self.references["layer17_input_rmsnorm"][position].to(torch.float32)
        output_q = backend.canonical.quantize_int8(reference, _scale32_float(scale32))
        record.update(
            {
                "rms_output_s8": output_q,
                "binary_input_norm": reference,
                "diagnostic_cut": self.cut,
            }
        )

    def install(self) -> None:
        def derive(
            layer_id: int,
            state: dict[str, Any],
            cache: dict[str, Any],
            template: dict[str, Any] | None,
            weights: Any,
            adapter: Any,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            position = int(state["position"])
            if layer_id == layer16.TARGET_LAYER:
                localizer.require(
                    position in self.records,
                    "group-1/group-4 source record missing at layer 17",
                )
                if self.cut == "layer17_input_rmsnorm":
                    self._replace_layer17_input_rmsnorm(position)
                self.cache.position_records[position] = self.records[position]
                if template is not None:
                    template["qkv"].pop("q", None)
            derived, next_state, next_template = self.original(
                layer_id, state, cache, template, weights, adapter
            )
            if layer_id == layer16.SOURCE_LAYER:
                layer_position = derived["positions"][0]
                localizer.require(
                    position in self.cache.down_records,
                    "layer-16 group-4 down record was not captured",
                )
                down = self.cache.down_records[position]
                localizer.require(
                    torch.equal(
                        layer_position["projections"]["down"]["output_q"],
                        down["down_output_s8"],
                    ),
                    "backend did not consume frozen group-4 down output",
                )
                self.records[position] = dict(down)
                self._replace_record(position)
                if self.cut == "layer16_final_residual":
                    self._replace_final_residual(position, next_state)
            return derived, next_state, next_template

        backend.derive_layer_token = derive


def _bundle_records(records: dict[int, dict[str, Any]]) -> dict[str, torch.Tensor]:
    tensors = downlocal._bundle_records(records)
    for name in (
        "post_attention_sum_s8",
        "post_attention_output_group_scale32",
        "post_attention_sum_saturation",
        "q_output_group_scale32",
    ):
        tensors[name] = torch.stack(
            [records[position][name] for position in range(len(localizer.TOKEN_IDS))]
        ).contiguous()
    return tensors


def _run_variant(
    cut: str,
    cache: BoundaryCache,
    references: dict[str, torch.Tensor],
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cache.configure_boundary(cut)
    cache.install()
    alignment = postattn.PostAttentionAlignmentHook(ATTENTION_OUTPUT_GROUP_SIZE)
    alignment.install()
    source_guard = BoundarySourceGuard(cache, weights, alignment, references, cut)
    score_guard = qoutput.QHeadScaleScoreGuard(cache)
    score_guard.install()
    source_guard.install()
    try:
        result, _ = localizer.run_cut(
            None, [], weights, adapter, norm_gain, embedding, head, tokenizer
        )
    finally:
        source_guard.restore()
        score_guard.restore()
        alignment.restore()
        cache.restore()
    localizer.require(
        score_guard.replacements == backend.Q_HEADS * (2 * len(localizer.TOKEN_IDS) - 1),
        "accepted layer-17 Q group scales did not reach every score",
    )
    bundle = _bundle_records(source_guard.records)
    final = source_guard.records[len(localizer.TOKEN_IDS) - 1]
    result.update(
        {
            "diagnostic_cut": cut,
            "exact_boundary": None if cut == "control" else EXACT_BOUNDARIES[cut],
            "diagnostic_only_bf16_or_reference": cut != "control",
            "attention_output_group_size": ATTENTION_OUTPUT_GROUP_SIZE,
            "attention_source_group_size": ATTENTION_SOURCE_GROUP_SIZE,
            "down_output_group_size": DOWN_OUTPUT_GROUP_SIZE,
            "q_output_group_size": Q_OUTPUT_GROUP_SIZE,
            "layer17_score_scale_replacements": score_guard.replacements,
            "post_attention_sum_saturation_count": int(
                final["post_attention_sum_saturation"].sum()
            ),
            "post_attention_sum_s8_sha256": localizer.tensor_sha256(
                final["post_attention_sum_s8"]
            ),
            "down_output_s8_sha256": localizer.tensor_sha256(
                final["down_output_s8"]
            ),
            "down_output_group_scale32_sha256": localizer.tensor_sha256(
                final["down_output_group_scale32"]
            ),
            "q_output_s8_sha256": localizer.tensor_sha256(final["q_output_s8"]),
            "down_saturation_count": int(final["down_saturation"].sum()),
            "sum_saturation_count": int(final["sum_saturation"].sum()),
            "q_saturation_count": int(final["q_saturation"].sum()),
            "reproducibility_signature": {
                name: localizer.tensor_sha256(value) for name, value in sorted(bundle.items())
            },
            "vector_shape": {
                "positions": len(localizer.TOKEN_IDS),
                "hidden_channels": backend.HIDDEN,
                "intermediate_channels": backend.INTERMEDIATE,
            },
        }
    )
    return result, bundle


def _write_variant(
    directory: Path, metrics: dict[str, Any], tensors: dict[str, torch.Tensor]
) -> None:
    artifacts = {
        name: localizer.write_tensor(directory / f"{name}.bin", value)
        for name, value in sorted(tensors.items())
    }
    localizer.write_json(directory / "result.json", {"metrics": metrics, "artifacts": artifacts})


def _exact_repeat(first: dict[str, Any], repeat: dict[str, Any]) -> None:
    localizer.require(
        first["top_token_id"] == repeat["top_token_id"]
        and first["reference_rank"] == repeat["reference_rank"]
        and first["reproducibility_signature"] == repeat["reproducibility_signature"],
        f"diagnostic cut did not reproduce exactly: {first['diagnostic_cut']}",
    )


def _qualifies(record: dict[str, Any]) -> bool:
    return bool(
        int(record["top_token_id"]) == localizer.REFERENCE_TOKEN
        or int(record["reference_rank"]) < BASELINE_RANK
    )


def _failure(cut: str, error: Exception) -> dict[str, Any]:
    return {
        "diagnostic_cut": cut,
        "classification": "DIAGNOSTIC_CUT_EXECUTION_FAILURE",
        "detail": str(error),
        "traceback": traceback.format_exc(),
        "root_cause_hypothesis": (
            "The BF16/reference boundary could not be consumed while preserving the frozen "
            "group-1/group-4/group-128 signed-A8 and W4 contracts."
        ),
        "regression": (
            "Preserve this failed cut and require exact execution at the same boundary before "
            "using any later cut as causal evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    localizer.require(ROOT.resolve() in output.parents, "output must be repository-relative")
    localizer.require(not output.exists(), "qualifying localization output already exists")
    for relative, expected in PROTECTED_HASHES.items():
        localizer.require(
            _sha256(ROOT / relative) == expected,
            f"protected hash differs before localization: {relative}",
        )
    predecessor, predecessor_freeze, manifest_members = _gate_predecessor()
    selected_predecessor = _selected_trace(predecessor)

    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    localizer.require(
        generation_runner.canonical_chat_token_ids(tokenizer, "Hello") == localizer.TOKEN_IDS,
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
            "top_token_id": BASELINE_TOP_TOKEN,
            "reference_rank": BASELINE_RANK,
            "post_attention_output_group_size": ATTENTION_OUTPUT_GROUP_SIZE,
            "attention_source_group_size": ATTENTION_SOURCE_GROUP_SIZE,
            "down_output_group_size": DOWN_OUTPUT_GROUP_SIZE,
            "q_output_group_size": Q_OUTPUT_GROUP_SIZE,
            "expected_selected_trace": selected_predecessor,
        },
        "ordered_diagnostic_cuts": list(ORDERED_CUTS),
        "exact_boundaries": EXACT_BOUNDARIES,
        "score_policy": {
            "qualifies": "token 9707 becomes top-1 or its rank is strictly below 3252",
            "reason": "No larger materiality threshold is authorized by the live objective.",
            "stop": "first qualifying cut in execution order after an exact repeat",
        },
        "method": {
            "cuts_are_isolated": True,
            "each_executed_cut_repeated_exactly": True,
            "bf16_or_reference_substitution_is_diagnostic_only": True,
            "accepted_w4_bytes_and_scales_unchanged": True,
            "signed_a8_carried_payloads_unchanged": True,
            "runtime_bf16_sidecar": False,
            "token_or_logit_override": False,
        },
        "rtl_public_contract": predecessor_freeze["rtl_public_contract"],
        "rtl_interface_preflight": postattn._rtl_preflight(),
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
            "predecessor_result": localizer.file_record(PREDECESSOR / "result.json"),
            "predecessor_freeze": localizer.file_record(PREDECESSOR / "candidate-freeze.json"),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
            "predecessor_manifest_members_verified": manifest_members,
            "fresh_l2_search": localizer.file_record(FRESH_SEARCH),
            "fresh_l2_rtl": localizer.file_record(FRESH_RTL),
            "fresh_l2_reviewer_handoff": _external_record(FRESH_HANDOFF),
            "runner": localizer.file_record(Path(__file__).resolve()),
        },
    }
    localizer.write_json(output / "candidate-freeze.json", freeze)
    _event(output, "freeze_written", freeze_sha256=_sha256(output / "candidate-freeze.json"))

    started = time.monotonic()
    with safe_open(backend.canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        cache = BoundaryCache(weights, adapter, {})
        references, reference_binding = _capture_reference_boundaries(snapshot, cache)
        cache.references = references
        reference_artifacts = {
            name: localizer.write_tensor(output / f"reference/{name}.bin", value)
            for name, value in sorted(references.items())
        }
        localizer.write_json(
            output / "reference/result.json",
            {
                "schema_version": 1,
                "definition": (
                    "BF16 hidden-state boundaries plus accepted-W4 dequantized gate/up/down/Q "
                    "references for diagnostic substitution only."
                ),
                "binding": reference_binding,
                "artifacts": reference_artifacts,
            },
        )
        _event(output, "reference_captured", binding=reference_binding)

        control, control_tensors = _run_variant(
            "control", cache, references, weights, adapter, norm_gain, embedding, head, tokenizer
        )
        localizer.require(
            int(control["top_token_id"]) == BASELINE_TOP_TOKEN
            and int(control["reference_rank"]) == BASELINE_RANK
            and int(control["post_attention_sum_saturation_count"]) == 0,
            "accepted group-1 control outcome differs",
        )
        for key in (
            "post_attention_sum_s8_sha256",
            "down_s8_sha256",
            "down_scale32_sha256",
            "q_output_s8_sha256",
        ):
            if key in selected_predecessor:
                observed_key = key
                if key == "down_s8_sha256":
                    observed_key = "down_output_s8_sha256"
                elif key == "down_scale32_sha256":
                    observed_key = "down_output_group_scale32_sha256"
                localizer.require(
                    control[observed_key] == selected_predecessor[key],
                    f"accepted group-1 control hash differs: {key}",
                )
        _write_variant(output / "runs/control", control, control_tensors)
        control_repeat, control_repeat_tensors = _run_variant(
            "control", cache, references, weights, adapter, norm_gain, embedding, head, tokenizer
        )
        _exact_repeat(control, control_repeat)
        _write_variant(output / "repeats/control", control_repeat, control_repeat_tensors)
        _event(
            output,
            "control_repeated",
            top_token_id=control["top_token_id"],
            reference_rank=control["reference_rank"],
        )
        print(
            f"ACE2_POSTATTN_GROUP1_LOCALIZATION cut=control top={control['top_token_id']} rank={control['reference_rank']}",
            flush=True,
        )

        trace = []
        failures = []
        selected = None
        for index, cut in enumerate(ORDERED_CUTS):
            try:
                record, tensors = _run_variant(
                    cut,
                    cache,
                    references,
                    weights,
                    adapter,
                    norm_gain,
                    embedding,
                    head,
                    tokenizer,
                )
                repeat, repeat_tensors = _run_variant(
                    cut,
                    cache,
                    references,
                    weights,
                    adapter,
                    norm_gain,
                    embedding,
                    head,
                    tokenizer,
                )
                _exact_repeat(record, repeat)
            except Exception as error:
                failure = _failure(cut, error)
                localizer.write_json(output / f"runs/{index + 1:02d}-{cut}/failure.json", failure)
                failures.append(failure)
                _event(output, "cut_failed", cut=cut, detail=str(error))
                break
            record.update(
                {
                    "reference_rank_change": BASELINE_RANK - int(record["reference_rank"]),
                    "restored_token_9707": int(record["top_token_id"]) == localizer.REFERENCE_TOKEN,
                    "strict_rank_improvement": int(record["reference_rank"]) < BASELINE_RANK,
                    "exact_repeat_passed": True,
                }
            )
            repeat.update(record | {"reproducibility_signature": repeat["reproducibility_signature"]})
            _write_variant(output / f"runs/{index + 1:02d}-{cut}", record, tensors)
            _write_variant(output / f"repeats/{index + 1:02d}-{cut}", repeat, repeat_tensors)
            trace.append(record)
            _event(
                output,
                "cut_repeated",
                cut=cut,
                top_token_id=record["top_token_id"],
                reference_rank=record["reference_rank"],
                reference_rank_change=record["reference_rank_change"],
            )
            print(
                "ACE2_POSTATTN_GROUP1_LOCALIZATION "
                f"cut={cut} top={record['top_token_id']} rank={record['reference_rank']} "
                f"delta={record['reference_rank_change']} repeat=exact",
                flush=True,
            )
            if _qualifies(record):
                selected = record
                break

    for relative, expected in PROTECTED_HASHES.items():
        localizer.require(
            _sha256(ROOT / relative) == expected,
            f"protected hash differs after localization: {relative}",
        )
    next_boundary = selected["exact_boundary"] if selected is not None else None
    result = {
        "schema_version": 1,
        "status": (
            "PASS_EARLIEST_POST_ATTENTION_GROUP1_REMAINING_BOUNDARY_LOCALIZED"
            if selected is not None
            else "BLOCKED_DIAGNOSTIC_CUT_EXECUTION_FAILURE"
            if failures
            else "PARTIAL_NO_MATERIAL_BOUNDARY_WITHIN_BOUNDED_SEQUENCE"
        ),
        "classification": "candidate_only_diagnostic_no_official_attempt_created",
        "control": control,
        "control_exact_repeat": {
            "passed": True,
            "top_token_id": control_repeat["top_token_id"],
            "reference_rank": control_repeat["reference_rank"],
            "reproducibility_signature": control_repeat["reproducibility_signature"],
        },
        "ordered_cuts": list(ORDERED_CUTS),
        "trace": trace,
        "failures": failures,
        "earliest_qualifying_cut": selected["diagnostic_cut"] if selected else None,
        "next_exact_repair_boundary": next_boundary,
        "selection_reason": (
            "restored_token_9707"
            if selected is not None and selected["restored_token_9707"]
            else "strict_rank_improvement_beyond_3252"
            if selected is not None
            else "none"
        ),
        "prequeued_stage1_repair": (
            {
                "status": "dependency_gated_prequeued",
                "dependency": "fresh_independent_reviewer_acceptance_of_this_localization",
                "boundary": next_boundary,
                "implementation_class": (
                    "hardware-realizable signed-A8/Scale32 arithmetic or quantizer repair at "
                    "the localized boundary; accepted W4 bytes/scales and carried width fixed"
                ),
                "implemented": False,
            }
            if selected is not None
            else None
        ),
        "diagnostic_bf16_is_not_hardware_repair": True,
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "rtl_repair_implemented": False,
        "u280_or_stage2_entered": False,
        "elapsed_wall_seconds": time.monotonic() - started,
        "bindings": {
            "freeze": localizer.file_record(output / "candidate-freeze.json"),
            "reference": localizer.file_record(output / "reference/result.json"),
            "events": localizer.file_record(output / "execution-events.jsonl"),
            "predecessor_sha256s": localizer.file_record(PREDECESSOR / "SHA256SUMS"),
            "fresh_l2_search": localizer.file_record(FRESH_SEARCH),
            "fresh_l2_rtl": localizer.file_record(FRESH_RTL),
        },
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    print(
        "ACE2_POSTATTN_GROUP1_LOCALIZATION_RESULT "
        f"status={result['status']} boundary={result['next_exact_repair_boundary']}",
        flush=True,
    )
    return 0 if selected is not None else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ACE2_POSTATTN_GROUP1_LOCALIZATION_FAIL detail={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
