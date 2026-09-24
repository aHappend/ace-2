#!/usr/bin/env python3
"""Test one packable joint Layer-21 activation/residual-domain repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_full_model_fixed_point as fixed_point
from tools.ace2_chat_all_residual_reference import (
    FROZEN_SCALES,
    MODEL_REPOSITORY,
    ROPE_MECHANISM,
    exact_process_argv,
    frozen_ranges,
    prompt_coherence,
    sha256_bytes,
    summarize_execution,
    token_piece,
    top_k,
    utc_now,
    validate_frozen_scale_binding,
)
from tools.ace2_chat_demo import (
    REVISION,
    TOKENIZER_SNAPSHOT,
    canonical_bytes,
    decode_generated,
    load_tokenizer,
    sha256_file,
    tokenize_prompt,
    write_atomic,
)
from tools.ace2_chat_layer21_groupwise_w4_repair import (
    append_section,
    pack_signed_int4,
    realized_scale32,
    tensor_sha256,
)
from tools.ace2_chat_layer_prefix_substitution import PrefixSubstitution, relative_l2
from tools.ace2_chat_reference_repairs import readability_gate


LAYER_INDEX = 21
GROUP_SIZE = 32
HIDDEN_SIZE = 896
HIDDEN_GROUPS = HIDDEN_SIZE // GROUP_SIZE
PREFIX21 = (
    ROOT
    / "build/ace2_chat_diagnostics/layer-prefix-substitution-p21-two-prompts-20260805.json"
)
PREFIX22 = (
    ROOT
    / "build/ace2_chat_diagnostics/layer-prefix-substitution-p22-eight-token-two-prompts-20260805.json"
)


def expand_group_values(values: Tensor, width: int) -> Tensor:
    if width % GROUP_SIZE:
        raise ValueError("per-group activation width is not divisible by 32")
    if values.shape != (width // GROUP_SIZE,):
        raise ValueError("per-group activation metadata has the wrong geometry")
    return values.repeat_interleave(GROUP_SIZE)


def scale32_records_from_absmax(absmax: Tensor) -> tuple[Tensor, Tensor]:
    if absmax.ndim != 1 or torch.any(~torch.isfinite(absmax)) or torch.any(absmax <= 0):
        raise ValueError("activation group maxima must be finite and positive")
    records = torch.tensor(
        [
            fixed_point.ceil_scale32_from_float(float(value) / 127.0)
            for value in absmax.detach().cpu().tolist()
        ],
        dtype=torch.int64,
    )
    scales = torch.tensor(
        [realized_scale32(int(record)) for record in records.tolist()],
        dtype=torch.float64,
    )
    return records, scales


def quantize_grouped(value: Tensor, scales: Tensor) -> tuple[Tensor, dict[str, int]]:
    expanded = expand_group_values(scales.to(value.device), value.shape[-1])
    rounded = torch.round(value.to(torch.float64) / expanded)
    negative = int((rounded < -128).sum())
    positive = int((rounded > 127).sum())
    output = rounded.clamp(-128, 127).to(torch.int8)
    return output, {
        "elements": output.numel(),
        "negative_saturation_events": negative,
        "positive_saturation_events": positive,
        "negative_extrema": int((output == -128).sum()),
        "positive_extrema": int((output == 127).sum()),
    }


def dequantize_grouped(value: Tensor, scales: Tensor, dtype: torch.dtype) -> Tensor:
    expanded = expand_group_values(scales.to(value.device), value.shape[-1])
    return (value.to(torch.float64) * expanded).to(dtype)


def merge_saturation(records: list[dict[str, int]]) -> dict[str, int]:
    return {
        key: sum(int(record[key]) for record in records)
        for key in (
            "elements",
            "negative_saturation_events",
            "positive_saturation_events",
            "negative_extrema",
            "positive_extrema",
        )
    }


class PerGroupW4A8Linear(nn.Module):
    """Signed-W4 groups with fixed per-32-input signed-int8 activation scales."""

    def __init__(
        self,
        source: nn.Linear,
        template: fixed_point.W4A8Linear,
        input_group_scales: Tensor,
    ) -> None:
        super().__init__()
        if source.in_features % GROUP_SIZE:
            raise ValueError("Layer-21 projection input width is not divisible by 32")
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.group_size = GROUP_SIZE
        self.groups = source.in_features // GROUP_SIZE
        if input_group_scales.shape != (self.groups,):
            raise ValueError("projection activation scales do not cover all input groups")
        self.input_scale = float(template.input_scale)
        self.hardware_input_scale = float(template.hardware_input_scale)
        self.output_scale = template.output_scale
        self.output_head_size = template.output_head_size
        self.output_scale32_record = template.output_scale32_record
        self.input_is_quantized = template.input_is_quantized
        self.register_buffer(
            "input_group_scales",
            input_group_scales.detach().to(torch.float64).clone(),
            persistent=True,
        )
        self.register_buffer(
            "input_group_scale32_records",
            torch.tensor(
                [
                    fixed_point.ceil_scale32_from_float(float(value))
                    for value in input_group_scales.detach().cpu().tolist()
                ],
                dtype=torch.int64,
            ),
            persistent=True,
        )
        self.register_buffer(
            "output_head_scales",
            template.output_head_scales.detach().clone(),
            persistent=True,
        )
        self.register_buffer(
            "output_scale_per_channel",
            template.output_scale_per_channel.detach().clone(),
            persistent=True,
        )
        weight = source.weight.detach().to(torch.float64).reshape(
            self.out_features, self.groups, GROUP_SIZE
        )
        weight_scale = weight.abs().amax(dim=2) / 7.0
        weight_scale = torch.where(
            weight_scale > 0, weight_scale, torch.ones_like(weight_scale)
        )
        qweight = torch.round(weight / weight_scale[:, :, None]).clamp(-8, 7).to(
            torch.int8
        )
        self.register_buffer("qweight", qweight, persistent=True)
        self.register_buffer("weight_scale", weight_scale, persistent=True)
        self.register_buffer(
            "_source_bias",
            source.bias.detach().to(torch.float64) if source.bias is not None else None,
            persistent=False,
        )

        native = weight_scale * self.input_group_scales[None, :]
        common_records = torch.tensor(
            [
                fixed_point.ceil_scale32_from_float(float(value))
                for value in native.amin(dim=1).detach().cpu().tolist()
            ],
            dtype=torch.int64,
        )
        common = torch.tensor(
            [realized_scale32(int(value)) for value in common_records.tolist()],
            dtype=torch.float64,
        )
        group_multiplier, group_right_shift = fixed_point.derive_multiplier(
            native / common[:, None]
        )
        multiplier, right_shift = fixed_point.derive_multiplier(
            common / self.output_scale_per_channel
        )
        self.register_buffer("native_scale32_records", common_records, persistent=True)
        self.register_buffer("group_multiplier", group_multiplier, persistent=True)
        self.register_buffer("group_right_shift", group_right_shift, persistent=True)
        self.register_buffer("multiplier", multiplier, persistent=True)
        self.register_buffer("right_shift", right_shift, persistent=True)
        self.register_buffer(
            "bias_accumulator",
            (
                torch.round(self._source_bias / common).to(torch.int64)
                if self._source_bias is not None
                else None
            ),
            persistent=True,
        )

    def bind_hardware_input_scale(self, input_scale: float) -> None:
        if not math.isfinite(input_scale) or input_scale <= 0:
            raise ValueError("projection scalar compatibility scale must be positive")
        self.hardware_input_scale = float(input_scale)

    def accumulator_quantized(self, qinput: Tensor) -> Tensor:
        if qinput.dtype != torch.int8 or qinput.shape[-1] != self.in_features:
            raise ValueError("per-group projection requires signed-int8 input geometry")
        flat = qinput.reshape(-1, self.groups, GROUP_SIZE).to(torch.int64)
        group_accumulator = torch.einsum(
            "rgk,ogk->rog", flat, self.qweight.to(torch.int64)
        )
        converted = fixed_point.round_shift_even(
            group_accumulator * self.group_multiplier[None, :, :],
            self.group_right_shift[None, :, :],
        )
        accumulator = converted.sum(dim=2)
        if self.bias_accumulator is not None:
            accumulator = accumulator + self.bias_accumulator
        if torch.any(accumulator < -(1 << 31)) or torch.any(accumulator >= (1 << 31)):
            raise OverflowError("per-group projection common accumulator exceeds signed-32")
        return accumulator

    def requantize_accumulator(
        self, accumulator: Tensor, original_shape: tuple[int, ...]
    ) -> Tensor:
        product = accumulator.to(torch.int64) * self.multiplier
        output = fixed_point.round_shift_even(product, self.right_shift)
        return output.clamp(-128, 127).to(torch.int8).reshape(
            *original_shape, self.out_features
        )

    def forward_quantized(self, qinput: Tensor) -> Tensor:
        return self.requantize_accumulator(
            self.accumulator_quantized(qinput), qinput.shape[:-1]
        )

    def forward_hardware_input(self, inputs: Tensor) -> Tensor:
        if inputs.dtype == torch.int8:
            qinput = inputs
        elif inputs.is_floating_point():
            if torch.any(inputs < -128) or torch.any(inputs > 127) or not torch.equal(
                inputs, torch.round(inputs)
            ):
                raise ValueError("per-group hardware input is not signed-int8 valued")
            qinput = inputs.to(torch.int8)
        else:
            raise TypeError("per-group hardware input type differs")
        return self.forward_quantized(qinput)

    def forward_raw(self, inputs: Tensor) -> Tensor:
        qinput, _ = quantize_grouped(inputs, self.input_group_scales)
        return self.forward_quantized(qinput)

    def forward(self, inputs: Tensor) -> Tensor:
        raw = (
            self.forward_hardware_input(inputs)
            if self.input_is_quantized
            else self.forward_raw(inputs)
        )
        return raw.to(inputs.dtype) * self.output_scale_per_channel.to(inputs.dtype)


class PerGroupRMSNorm(nn.Module):
    """RMSNorm after exact Scale32 conversion into one signed-32 common domain."""

    def __init__(
        self,
        source: nn.Module,
        input_scales: Tensor,
        output_scales: Tensor,
    ) -> None:
        super().__init__()
        if input_scales.shape != (HIDDEN_GROUPS,) or output_scales.shape != (
            HIDDEN_GROUPS,
        ):
            raise ValueError("RMSNorm group scales must cover 28 hidden groups")
        common_record = fixed_point.ceil_scale32_from_float(
            float(input_scales.amin())
        )
        common_scale = realized_scale32(common_record)
        input_multiplier, input_right_shift = fixed_point.derive_multiplier(
            input_scales / common_scale
        )
        output_per_channel = expand_group_values(output_scales, HIDDEN_SIZE)
        gain = torch.round(
            source.weight.detach().to(torch.float64)
            / output_per_channel
            * (1 << fixed_point.RMS_GAIN_FRAC)
        )
        if torch.any(gain < -32768) or torch.any(gain > 32767):
            raise OverflowError("per-group RMSNorm gain is not signed-int16")
        self.input_scale = float(input_scales.max())
        self.output_scale = float(output_scales.max())
        self.common_scale32_record = common_record
        self.register_buffer("input_scales", input_scales.clone(), persistent=True)
        self.register_buffer("output_scales", output_scales.clone(), persistent=True)
        self.register_buffer("input_multiplier", input_multiplier, persistent=True)
        self.register_buffer("input_right_shift", input_right_shift, persistent=True)
        self.register_buffer("scaled_gains_q8", gain.to(torch.int16), persistent=True)

    def forward_quantized(self, qinput: Tensor) -> tuple[Tensor, dict[str, int]]:
        if qinput.dtype != torch.int8 or qinput.shape[-1] != HIDDEN_SIZE:
            raise ValueError("per-group RMSNorm requires 896 signed-int8 lanes")
        grouped = qinput.reshape(*qinput.shape[:-1], HIDDEN_GROUPS, GROUP_SIZE).to(
            torch.int64
        )
        converted = fixed_point.round_shift_even(
            grouped * self.input_multiplier.reshape(
                *((1,) * (grouped.ndim - 2)), HIDDEN_GROUPS, 1
            ),
            self.input_right_shift.reshape(
                *((1,) * (grouped.ndim - 2)), HIDDEN_GROUPS, 1
            ),
        ).reshape(*qinput.shape[:-1], HIDDEN_SIZE)
        if torch.any(converted < -(1 << 31)) or torch.any(converted >= (1 << 31)):
            raise OverflowError("RMSNorm common-domain carry exceeds signed-32")
        sumsq = (converted * converted).sum(dim=-1, keepdim=True)
        mean_square = (sumsq + HIDDEN_SIZE // 2) // HIDDEN_SIZE
        root = torch.floor(torch.sqrt(mean_square.to(torch.float64))).to(torch.int64)
        root = (root + (root * root < mean_square).to(torch.int64)).clamp(min=1)
        inv_rms_q30 = (1 << fixed_point.RMS_INV_FRAC) // root
        product = converted * self.scaled_gains_q8.to(torch.int64) * inv_rms_q30
        unbounded = fixed_point.round_shift_even(
            product,
            fixed_point.RMS_INV_FRAC + fixed_point.RMS_GAIN_FRAC,
        )
        trace = {
            "elements": unbounded.numel(),
            "negative_saturation_events": int((unbounded < -128).sum()),
            "positive_saturation_events": int((unbounded > 127).sum()),
            "negative_extrema": int((unbounded.clamp(-128, 127) == -128).sum()),
            "positive_extrema": int((unbounded.clamp(-128, 127) == 127).sum()),
        }
        return unbounded.clamp(-128, 127).to(torch.int8), trace

    def forward(self, hidden_states: Tensor) -> Tensor:
        qinput, _ = quantize_grouped(hidden_states, self.input_scales)
        output, _ = self.forward_quantized(qinput)
        return output.to(hidden_states.dtype)


def grouped_silu_gate_raw(
    gate: Tensor,
    up: Tensor,
    gate_scale: float,
    up_scale: float,
    output_scales: Tensor,
) -> tuple[Tensor, dict[str, int], Tensor, Tensor]:
    if gate.dtype != torch.int8 or up.dtype != torch.int8 or gate.shape != up.shape:
        raise ValueError("grouped SiLU requires matching signed-int8 gate/up tensors")
    gate_q6_9 = torch.round(gate.to(torch.float64) * gate_scale * (1 << 9))
    up_q6_9 = torch.round(up.to(torch.float64) * up_scale * (1 << 9))
    gate_q6_9 = gate_q6_9.clamp(-32768, 32767).to(torch.int64)
    up_q6_9 = up_q6_9.clamp(-32768, 32767).to(torch.int64)
    table_index = torch.bitwise_right_shift(gate_q6_9, 6).clamp(-64, 64) + 64
    table = torch.tensor(fixed_point.SILU_LUT, dtype=torch.int64, device=gate.device)
    product_q9_21 = table[table_index] * up_q6_9
    multiplier, right_shift = fixed_point.derive_multiplier(
        1.0 / ((1 << 21) * output_scales)
    )
    expanded_multiplier = expand_group_values(multiplier, gate.shape[-1]).to(gate.device)
    expanded_shift = expand_group_values(right_shift, gate.shape[-1]).to(gate.device)
    unbounded = fixed_point.round_shift_even(
        product_q9_21 * expanded_multiplier,
        expanded_shift,
    )
    output = unbounded.clamp(-128, 127).to(torch.int8)
    trace = {
        "elements": output.numel(),
        "negative_saturation_events": int((unbounded < -128).sum()),
        "positive_saturation_events": int((unbounded > 127).sum()),
        "negative_extrema": int((output == -128).sum()),
        "positive_extrema": int((output == 127).sum()),
    }
    return output, trace, multiplier, right_shift


def fuse_grouped_residual(
    accumulator: Tensor,
    residual: Tensor,
    accumulator_scale32: Tensor,
    residual_scale32: Tensor,
    destination_scale32: Tensor,
) -> tuple[Tensor, dict[str, int]]:
    if accumulator.dtype != torch.int32 or residual.dtype != torch.int8:
        raise TypeError("grouped residual fusion requires signed-int32/signed-int8")
    if accumulator.shape != residual.shape or accumulator.shape[-1] != HIDDEN_SIZE:
        raise ValueError("grouped residual fusion geometry differs")
    if accumulator_scale32.shape != (HIDDEN_SIZE,):
        raise ValueError("grouped residual accumulator scales do not cover 896 lanes")
    if residual_scale32.shape != (HIDDEN_GROUPS,) or destination_scale32.shape != (
        HIDDEN_GROUPS,
    ):
        raise ValueError("grouped residual scales do not cover 28 groups")
    accumulator_rows = accumulator.detach().cpu().reshape(-1, HIDDEN_SIZE).tolist()
    residual_rows = residual.detach().cpu().reshape(-1, HIDDEN_SIZE).tolist()
    accumulator_records = accumulator_scale32.detach().cpu().tolist()
    residual_records = expand_group_values(residual_scale32, HIDDEN_SIZE).tolist()
    destination_records = expand_group_values(destination_scale32, HIDDEN_SIZE).tolist()
    output_rows: list[list[int]] = []
    positive = 0
    negative = 0
    common_min = 127
    common_max = -128
    for accumulator_row, residual_row in zip(
        accumulator_rows, residual_rows, strict=True
    ):
        output_row = []
        for lane, (accumulator_value, residual_value) in enumerate(
            zip(accumulator_row, residual_row, strict=True)
        ):
            fused = fixed_point.fuse_lane(
                accumulator_value,
                residual_value,
                int(accumulator_records[lane]),
                int(residual_records[lane]),
                int(destination_records[lane]),
            )
            output_row.append(fused.output_s8)
            positive += int(fused.positive_saturation)
            negative += int(fused.negative_saturation)
            common_min = min(common_min, fused.common_exponent)
            common_max = max(common_max, fused.common_exponent)
        output_rows.append(output_row)
    output = torch.tensor(output_rows, dtype=torch.int8).reshape(residual.shape)
    output = output.to(residual.device)
    return output, {
        "rows": len(output_rows),
        "elements": output.numel(),
        "positive_saturation_events": positive,
        "negative_saturation_events": negative,
        "positive_extrema": int((output == 127).sum()),
        "negative_extrema": int((output == -128).sum()),
        "common_exponent_minimum": common_min,
        "common_exponent_maximum": common_max,
    }


def update_group_maximum(target: Tensor, value: Tensor) -> None:
    groups = value.shape[-1] // GROUP_SIZE
    if target.shape != (groups,):
        raise ValueError("calibration target geometry differs")
    maximum = value.detach().to(torch.float64).reshape(-1, groups, GROUP_SIZE).abs().amax(
        dim=(0, 2)
    )
    target.copy_(torch.maximum(target, maximum.cpu()))


def collect_fixed_group_scales(
    source_model: Any,
    tokenizations: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    prefix22 = json.loads(PREFIX22.read_text(encoding="utf-8"))
    source_layer = source_model.model.layers[LAYER_INDEX]
    maxima = {
        "layer_input": torch.zeros(HIDDEN_GROUPS, dtype=torch.float64),
        "input_norm_output": torch.zeros(HIDDEN_GROUPS, dtype=torch.float64),
        "post_attention_residual": torch.zeros(HIDDEN_GROUPS, dtype=torch.float64),
        "post_attention_norm_output": torch.zeros(HIDDEN_GROUPS, dtype=torch.float64),
        "gated_down_input": torch.zeros(
            source_layer.mlp.down_proj.in_features // GROUP_SIZE,
            dtype=torch.float64,
        ),
        "post_mlp_output": torch.zeros(HIDDEN_GROUPS, dtype=torch.float64),
    }
    handles = []

    def layer_pre(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = kwargs.get("hidden_states") if kwargs else None
        if hidden is None:
            hidden = args[0]
        update_group_maximum(maxima["layer_input"], hidden)

    def capture_output(name: str):
        def hook(_module: Any, _args: tuple[Any, ...], output: Tensor) -> None:
            update_group_maximum(maxima[name], output)

        return hook

    def capture_input(name: str):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            update_group_maximum(maxima[name], args[0])

        return hook

    handles.extend(
        [
            source_layer.register_forward_pre_hook(layer_pre, with_kwargs=True),
            source_layer.input_layernorm.register_forward_hook(
                capture_output("input_norm_output")
            ),
            source_layer.post_attention_layernorm.register_forward_pre_hook(
                capture_input("post_attention_residual")
            ),
            source_layer.post_attention_layernorm.register_forward_hook(
                capture_output("post_attention_norm_output")
            ),
            source_layer.mlp.down_proj.register_forward_pre_hook(
                capture_input("gated_down_input")
            ),
            source_layer.register_forward_hook(capture_output("post_mlp_output")),
        ]
    )
    trajectories = []
    try:
        with torch.inference_mode():
            for prompt_name, tokenization in tokenizations.items():
                generated = prefix22["generations"][prompt_name]["generated_token_ids"]
                tokens = [*tokenization["chat_template_token_ids"], *generated]
                source_model(
                    input_ids=torch.tensor([tokens], dtype=torch.long),
                    use_cache=False,
                )
                trajectories.append(
                    {
                        "prompt": prompt_name,
                        "prompt_tokens": len(tokenization["chat_template_token_ids"]),
                        "calibration_continuation_tokens": generated,
                        "total_tokens": len(tokens),
                    }
                )
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    for name, maximum in maxima.items():
        if torch.any(maximum <= 0):
            raise RuntimeError(f"calibration group remained empty: {name}")
        records, scales = scale32_records_from_absmax(maximum)
        result[name] = {
            "absmax": maximum,
            "scale32_records": records,
            "scales": scales,
        }
    return result, trajectories


def projection_manifest(
    name: str,
    source: nn.Linear,
    baseline: fixed_point.W4A8Linear,
    candidate: PerGroupW4A8Linear,
    payload: bytearray,
) -> dict[str, Any]:
    weight = source.weight.detach().to(torch.float64)
    baseline_weight = baseline.qweight.to(torch.float64) * baseline.weight_scale[:, None]
    candidate_weight = (
        candidate.qweight.to(torch.float64) * candidate.weight_scale[:, :, None]
    ).reshape_as(weight)
    sections = [
        append_section(payload, f"{name}.qweight_s4", pack_signed_int4(candidate.qweight))
    ]
    arrays = (
        (
            "input_group_scale32_u32",
            candidate.input_group_scale32_records.cpu().numpy().astype("<u4").tobytes(),
        ),
        (
            "group_multiplier_s32",
            candidate.group_multiplier.cpu().numpy().astype("<i4").tobytes(),
        ),
        (
            "group_right_shift_u8",
            candidate.group_right_shift.cpu().numpy().astype("u1").tobytes(),
        ),
        (
            "common_scale32_u32",
            candidate.native_scale32_records.cpu().numpy().astype("<u4").tobytes(),
        ),
        (
            "output_multiplier_s32",
            candidate.multiplier.cpu().numpy().astype("<i4").tobytes(),
        ),
        (
            "output_right_shift_u8",
            candidate.right_shift.cpu().numpy().astype("u1").tobytes(),
        ),
    )
    for suffix, data in arrays:
        sections.append(append_section(payload, f"{name}.{suffix}", data))
    if candidate.bias_accumulator is not None:
        sections.append(
            append_section(
                payload,
                f"{name}.bias_common_s32",
                candidate.bias_accumulator.cpu().numpy().astype("<i4").tobytes(),
            )
        )
    return {
        "name": name,
        "in_features": candidate.in_features,
        "out_features": candidate.out_features,
        "group_size": GROUP_SIZE,
        "groups_per_output": candidate.groups,
        "activation_scale_groups": candidate.groups,
        "weight_format": "signed_int4_two_values_per_byte",
        "activation_format": "signed_int8_fixed_per_32_input_channels",
        "group_accumulator": "signed_int32",
        "rounding": "round_to_nearest_ties_to_even",
        "baseline_per_output_weight_relative_l2": relative_l2(weight, baseline_weight),
        "candidate_groupwise_weight_relative_l2": relative_l2(weight, candidate_weight),
        "qweight_sha256": tensor_sha256(candidate.qweight),
        "input_group_scale32_sha256": tensor_sha256(
            candidate.input_group_scale32_records
        ),
        "sections": sections,
    }


def install_projection_repair(
    fixed_layer: Any,
    source_layer: Any,
    calibration: dict[str, dict[str, Any]],
    payload: bytearray,
) -> list[dict[str, Any]]:
    pairs = (
        (
            "self_attn.q_proj",
            fixed_layer.self_attn,
            source_layer.self_attn,
            "q_proj",
            calibration["input_norm_output"]["scales"],
        ),
        (
            "self_attn.k_proj",
            fixed_layer.self_attn,
            source_layer.self_attn,
            "k_proj",
            calibration["input_norm_output"]["scales"],
        ),
        (
            "self_attn.v_proj",
            fixed_layer.self_attn,
            source_layer.self_attn,
            "v_proj",
            calibration["input_norm_output"]["scales"],
        ),
        (
            "self_attn.o_proj",
            fixed_layer.self_attn,
            source_layer.self_attn,
            "o_proj",
            torch.full(
                (HIDDEN_GROUPS,),
                float(fixed_layer.self_attn.v_proj.output_scale),
                dtype=torch.float64,
            ),
        ),
        (
            "mlp.gate_proj",
            fixed_layer.mlp,
            source_layer.mlp,
            "gate_proj",
            calibration["post_attention_norm_output"]["scales"],
        ),
        (
            "mlp.up_proj",
            fixed_layer.mlp,
            source_layer.mlp,
            "up_proj",
            calibration["post_attention_norm_output"]["scales"],
        ),
        (
            "mlp.down_proj",
            fixed_layer.mlp,
            source_layer.mlp,
            "down_proj",
            calibration["gated_down_input"]["scales"],
        ),
    )
    manifest = []
    for suffix, fixed_parent, source_parent, child, input_scales in pairs:
        baseline = getattr(fixed_parent, child)
        source = getattr(source_parent, child)
        if not isinstance(baseline, fixed_point.W4A8Linear):
            raise TypeError(f"Layer-21 {suffix} is not the frozen W4A8 module")
        candidate = PerGroupW4A8Linear(source, baseline, input_scales)
        manifest.append(
            projection_manifest(
                f"model.layers.{LAYER_INDEX}.{suffix}",
                source,
                baseline,
                candidate,
                payload,
            )
        )
        setattr(fixed_parent, child, candidate)
    return manifest


def append_rms_manifest(
    name: str,
    module: PerGroupRMSNorm,
    payload: bytearray,
) -> dict[str, Any]:
    sections = []
    arrays = (
        (
            "common_scale32_u32",
            int(module.common_scale32_record).to_bytes(4, "little", signed=False),
        ),
        (
            "input_multiplier_s32",
            module.input_multiplier.cpu().numpy().astype("<i4").tobytes(),
        ),
        (
            "input_right_shift_u8",
            module.input_right_shift.cpu().numpy().astype("u1").tobytes(),
        ),
        (
            "scaled_gains_q8_s16",
            module.scaled_gains_q8.cpu().numpy().astype("<i2").tobytes(),
        ),
    )
    for suffix, data in arrays:
        sections.append(append_section(payload, f"{name}.{suffix}", data))
    return {
        "name": name,
        "input_groups": HIDDEN_GROUPS,
        "output_groups": HIDDEN_GROUPS,
        "common_domain": "signed_int32",
        "rounding": "round_to_nearest_ties_to_even",
        "sections": sections,
    }


class JointCandidateRuntime:
    def __init__(self) -> None:
        self.prompt_name = ""
        self.generation_index = -1
        self.records: list[dict[str, Any]] = []
        self.pending_layer22_q: Tensor | None = None
        self.pending_layer22_scales: Tensor | None = None
        self.pending_record: dict[str, Any] | None = None

    def set_step(self, prompt_name: str, generation_index: int) -> None:
        if self.pending_layer22_q is not None:
            raise RuntimeError("Layer-22 carry remained unconsumed across generation steps")
        self.prompt_name = prompt_name
        self.generation_index = generation_index

    def publish_layer22(
        self,
        qvalue: Tensor,
        scales: Tensor,
        record: dict[str, Any],
    ) -> None:
        if self.pending_layer22_q is not None:
            raise RuntimeError("Layer-22 carry was published twice")
        self.pending_layer22_q = qvalue.detach().clone()
        self.pending_layer22_scales = scales
        self.pending_record = record

    def consume_layer22(self, hidden_states: Tensor) -> Tensor:
        if self.pending_layer22_q is None or self.pending_layer22_scales is None:
            raise RuntimeError("Layer-22 RMSNorm lacks the Layer-21 integer carry")
        expected = dequantize_grouped(
            self.pending_layer22_q,
            self.pending_layer22_scales,
            hidden_states.dtype,
        )
        if self.pending_record is None:
            raise AssertionError("Layer-22 carry record disappeared")
        self.pending_record["layer22_carry_interface"] = {
            "consumed": True,
            "transport_relative_l2_error": relative_l2(expected, hidden_states),
            "qvalue_sha256": tensor_sha256(self.pending_layer22_q),
        }
        qvalue = self.pending_layer22_q
        self.pending_layer22_q = None
        self.pending_layer22_scales = None
        self.pending_record = None
        return qvalue


class Layer22CarryRMSNorm(nn.Module):
    def __init__(self, norm: PerGroupRMSNorm, runtime: JointCandidateRuntime) -> None:
        super().__init__()
        self.norm = norm
        self.runtime = runtime
        self.input_scale = norm.input_scale
        self.output_scale = norm.output_scale

    def forward(self, hidden_states: Tensor) -> Tensor:
        qinput = self.runtime.consume_layer22(hidden_states)
        output, _trace = self.norm.forward_quantized(qinput)
        return output.to(hidden_states.dtype)


class JointLayer21Candidate(nn.Module):
    def __init__(
        self,
        fixed_layer: Any,
        source_layer: Any,
        calibration: dict[str, dict[str, Any]],
        rms_input: PerGroupRMSNorm,
        rms_post_attention: PerGroupRMSNorm,
        runtime: JointCandidateRuntime,
        shadow_runtime: fixed_point.AllProjectionResidualFusionRuntime,
    ) -> None:
        super().__init__()
        self.attention_type = fixed_layer.attention_type
        self.self_attn = fixed_layer.self_attn
        self.mlp = fixed_layer.mlp
        self.source_layer = source_layer
        self.input_layernorm = rms_input
        self.post_attention_layernorm = rms_post_attention
        self.runtime = runtime
        self.shadow_runtime = shadow_runtime
        self.scalar_input_scale = float(fixed_layer.input_scale)
        self.scalar_post_attention_scale = float(fixed_layer.post_attention_scale)
        self.scalar_post_mlp_scale = float(fixed_layer.post_mlp_scale)
        # PrefixSubstitution records the next layer's legacy scalar consumer
        # descriptor even though this candidate consumes the packed group table.
        self.input_scale = self.scalar_input_scale
        self.post_attention_scale = self.scalar_post_attention_scale
        self.post_mlp_scale = self.scalar_post_mlp_scale
        self.register_buffer(
            "input_scale32",
            calibration["layer_input"]["scale32_records"],
            persistent=True,
        )
        self.register_buffer(
            "input_scales", calibration["layer_input"]["scales"], persistent=True
        )
        self.register_buffer(
            "post_attention_scale32",
            calibration["post_attention_residual"]["scale32_records"],
            persistent=True,
        )
        self.register_buffer(
            "post_attention_scales",
            calibration["post_attention_residual"]["scales"],
            persistent=True,
        )
        self.register_buffer(
            "gated_scales",
            calibration["gated_down_input"]["scales"],
            persistent=True,
        )
        self.register_buffer(
            "post_mlp_scale32",
            calibration["post_mlp_output"]["scale32_records"],
            persistent=True,
        )
        self.register_buffer(
            "post_mlp_scales",
            calibration["post_mlp_output"]["scales"],
            persistent=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: Any = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        if position_embeddings is None:
            raise ValueError("position embeddings are required")
        captured: dict[str, Tensor] = {}

        def capture_exact_attention(
            _module: Any, args: tuple[Any, ...]
        ) -> None:
            captured["attention"] = args[0].detach().clone()

        handle = self.source_layer.post_attention_layernorm.register_forward_pre_hook(
            capture_exact_attention
        )
        try:
            exact_output = self.source_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        finally:
            handle.remove()
        if "attention" not in captured or not isinstance(exact_output, Tensor):
            raise RuntimeError("exact Layer-21 boundary capture failed")

        input_q, input_trace = quantize_grouped(hidden_states, self.input_scales)
        normalized_q, input_norm_trace = self.input_layernorm.forward_quantized(input_q)
        attention, _ = self.self_attn(
            hidden_states=normalized_q.to(hidden_states.dtype),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        del attention
        attention_accumulator = self.self_attn.last_o_projection_accumulator
        if attention_accumulator is None:
            raise RuntimeError("candidate attention lacks the o-projection accumulator")
        attention_q, attention_trace = fuse_grouped_residual(
            attention_accumulator,
            input_q,
            self.self_attn.o_proj.native_scale32_records,
            self.input_scale32,
            self.post_attention_scale32,
        )
        candidate_attention = dequantize_grouped(
            attention_q, self.post_attention_scales, hidden_states.dtype
        )
        self.shadow_runtime.apply(
            "attention",
            LAYER_INDEX,
            attention_accumulator,
            fixed_point.quantize_int8(hidden_states, self.scalar_input_scale),
            self.self_attn.o_proj.native_scale32_records,
            self.scalar_input_scale,
            self.scalar_post_attention_scale,
        )

        normalized_post_q, post_norm_trace = (
            self.post_attention_layernorm.forward_quantized(attention_q)
        )
        gate = self.mlp.gate_proj.forward_hardware_input(
            normalized_post_q.to(hidden_states.dtype)
        )
        up = self.mlp.up_proj.forward_hardware_input(
            normalized_post_q.to(hidden_states.dtype)
        )
        gated, gated_trace, _silu_multiplier, _silu_shift = grouped_silu_gate_raw(
            gate,
            up,
            float(self.mlp.gate_proj.output_scale),
            float(self.mlp.up_proj.output_scale),
            self.gated_scales,
        )
        down_accumulator = self.mlp.down_proj.accumulator_quantized(gated).to(
            torch.int32
        ).reshape(*gated.shape[:-1], HIDDEN_SIZE)
        output_q, output_trace = fuse_grouped_residual(
            down_accumulator,
            attention_q,
            self.mlp.down_proj.native_scale32_records,
            self.post_attention_scale32,
            self.post_mlp_scale32,
        )
        candidate_output = dequantize_grouped(
            output_q, self.post_mlp_scales, hidden_states.dtype
        )
        self.shadow_runtime.apply(
            "mlp",
            LAYER_INDEX,
            down_accumulator,
            fixed_point.quantize_int8(
                candidate_attention, self.scalar_post_attention_scale
            ),
            self.mlp.down_proj.native_scale32_records,
            self.scalar_post_attention_scale,
            self.scalar_post_mlp_scale,
        )

        record = {
            "prompt": self.runtime.prompt_name,
            "generation_index": self.runtime.generation_index,
            "layer_index": LAYER_INDEX,
            "input_quantization": {
                "relative_l2_error": relative_l2(
                    hidden_states,
                    dequantize_grouped(input_q, self.input_scales, hidden_states.dtype),
                ),
                "saturation": input_trace,
            },
            "input_rmsnorm_saturation": input_norm_trace,
            "attention_join": {
                "candidate_vs_exact_relative_l2_error": relative_l2(
                    captured["attention"], candidate_attention
                ),
                "saturation": attention_trace,
                "output_s8_sha256": tensor_sha256(attention_q),
            },
            "post_attention_rmsnorm_saturation": post_norm_trace,
            "gated_activation_saturation": gated_trace,
            "mlp_join": {
                "candidate_vs_exact_relative_l2_error": relative_l2(
                    exact_output, candidate_output
                ),
                "saturation": output_trace,
                "output_s8_sha256": tensor_sha256(output_q),
            },
        }
        self.runtime.records.append(record)
        self.runtime.publish_layer22(output_q, self.post_mlp_scales, record)
        return candidate_output


def activation_manifest(
    calibration: dict[str, dict[str, Any]],
    rms_modules: list[tuple[str, PerGroupRMSNorm]],
    gated_scales: Tensor,
    payload: bytearray,
) -> dict[str, Any]:
    scale_sections = []
    scale_records = {}
    for name, item in calibration.items():
        section = append_section(
            payload,
            f"activation.{name}.scale32_u32",
            item["scale32_records"].cpu().numpy().astype("<u4").tobytes(),
        )
        scale_sections.append(section)
        scale_records[name] = {
            "groups": int(item["scale32_records"].numel()),
            "group_size": GROUP_SIZE,
            "absmax_min": float(item["absmax"].min()),
            "absmax_max": float(item["absmax"].max()),
            "scale_min": float(item["scales"].min()),
            "scale_max": float(item["scales"].max()),
            "scale32_sha256": tensor_sha256(item["scale32_records"]),
        }
    rms = [append_rms_manifest(name, module, payload) for name, module in rms_modules]
    silu_multiplier, silu_shift = fixed_point.derive_multiplier(
        1.0 / ((1 << 21) * gated_scales)
    )
    silu_sections = [
        append_section(
            payload,
            "activation.layer21_silu.output_multiplier_s32",
            silu_multiplier.cpu().numpy().astype("<i4").tobytes(),
        ),
        append_section(
            payload,
            "activation.layer21_silu.output_right_shift_u8",
            silu_shift.cpu().numpy().astype("u1").tobytes(),
        ),
    ]
    return {
        "group_size": GROUP_SIZE,
        "scale_encoding": "normalized_nonzero_Scale32_uint32",
        "fixed_scale_records": scale_records,
        "scale_sections": scale_sections,
        "rmsnorm": rms,
        "silu_sections": silu_sections,
    }


def greedy_generate_candidate(
    model: Any,
    tokenizer: Any,
    prompt_name: str,
    prompt_tokens: list[int],
    termination_token_ids: set[int],
    *,
    max_new_tokens: int,
    top_k_count: int,
    prefix: PrefixSubstitution,
    runtime: JointCandidateRuntime,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    generated: list[int] = []
    steps = []
    with torch.inference_mode():
        for generation_index in range(max_new_tokens):
            prefix.set_step(prompt_name, generation_index)
            runtime.set_step(prompt_name, generation_index)
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1]
            candidates = top_k(tokenizer, logits, top_k_count)
            selected = int(candidates[0]["token_id"])
            generated.append(selected)
            terminated = selected in termination_token_ids
            steps.append(
                {
                    "generation_index": generation_index,
                    "selected_token_id": selected,
                    "selected_piece": token_piece(tokenizer, selected),
                    "terminated": terminated,
                    "top_k": candidates,
                }
            )
            input_ids = torch.cat(
                [input_ids, torch.tensor([[selected]], dtype=torch.long)], dim=1
            )
            if terminated:
                break
    if runtime.pending_layer22_q is not None:
        raise RuntimeError("final Layer-22 carry was not consumed")
    decoded = decode_generated(tokenizer, generated)
    readability = readability_gate(decoded, generated)
    return {
        "prompt_token_count": len(prompt_tokens),
        "prompt_tokens_sha256": sha256_bytes(
            torch.tensor(prompt_tokens, dtype=torch.int32).numpy().tobytes()
        ),
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "readability_gate": readability,
        "coherence_gate": prompt_coherence(prompt_name, decoded, readability),
        "terminated": bool(steps and steps[-1]["terminated"]),
        "steps": steps,
        "wall_seconds": time.perf_counter() - started,
    }


def aggregate_boundaries(
    records: list[dict[str, Any]], prompt_name: str
) -> dict[str, Any]:
    selected = [record for record in records if record["prompt"] == prompt_name]
    if not selected:
        raise RuntimeError(f"joint candidate produced no boundaries for {prompt_name}")
    if any(
        not record.get("layer22_carry_interface", {}).get("consumed", False)
        for record in selected
    ):
        raise RuntimeError("a Layer-21 output did not reach Layer-22 RMSNorm")
    return {
        "calls": len(selected),
        "first_generation_step": selected[0],
        "attention_join": {
            "relative_l2_error_first": selected[0]["attention_join"][
                "candidate_vs_exact_relative_l2_error"
            ],
            "relative_l2_error_max": max(
                record["attention_join"]["candidate_vs_exact_relative_l2_error"]
                for record in selected
            ),
            "saturation": merge_saturation(
                [record["attention_join"]["saturation"] for record in selected]
            ),
        },
        "mlp_join": {
            "relative_l2_error_first": selected[0]["mlp_join"][
                "candidate_vs_exact_relative_l2_error"
            ],
            "relative_l2_error_max": max(
                record["mlp_join"]["candidate_vs_exact_relative_l2_error"]
                for record in selected
            ),
            "saturation": merge_saturation(
                [record["mlp_join"]["saturation"] for record in selected]
            ),
        },
        "layer22_carry_transport_relative_l2_error_max": max(
            record["layer22_carry_interface"]["transport_relative_l2_error"]
            for record in selected
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pack-output", type=Path, required=True)
    parser.add_argument("--second-prompt", default="Write one word: blue")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    started = time.perf_counter()
    scales = json.loads(FROZEN_SCALES.read_text(encoding="utf-8"))
    ranges, operator_ranges = frozen_ranges(scales)
    tokenizer = load_tokenizer()
    tokenizations = {
        "hello_world": tokenize_prompt(tokenizer, "Hello world", ""),
        "second_prompt": tokenize_prompt(tokenizer, args.second_prompt, ""),
    }
    source_model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    fixed_model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    calibration, trajectories = collect_fixed_group_scales(
        source_model, tokenizations
    )
    fixed_point.replace_linears(
        fixed_model, ranges, rope_diagnostic_mechanism=ROPE_MECHANISM
    )
    fixed_point.replace_fixed_operators(
        fixed_model,
        operator_ranges,
        rope_diagnostic_mechanism=ROPE_MECHANISM,
        all_projection_residual_fusion=True,
    )
    frozen_binding_before_repair = validate_frozen_scale_binding(fixed_model, scales)
    shadow_runtime = fixed_model.ace2_all_projection_residual_fusion_runtime
    fixed_layer21 = fixed_model.model.layers[LAYER_INDEX]
    source_layer21 = source_model.model.layers[LAYER_INDEX]
    payload = bytearray()
    projection_repair = install_projection_repair(
        fixed_layer21, source_layer21, calibration, payload
    )

    input_norm = PerGroupRMSNorm(
        source_layer21.input_layernorm,
        calibration["layer_input"]["scales"],
        calibration["input_norm_output"]["scales"],
    )
    post_attention_norm = PerGroupRMSNorm(
        source_layer21.post_attention_layernorm,
        calibration["post_attention_residual"]["scales"],
        calibration["post_attention_norm_output"]["scales"],
    )
    fixed_layer22 = fixed_model.model.layers[LAYER_INDEX + 1]
    source_layer22 = source_model.model.layers[LAYER_INDEX + 1]
    layer22_output_scales = torch.full(
        (HIDDEN_GROUPS,),
        float(fixed_layer22.input_layernorm.output_scale),
        dtype=torch.float64,
    )
    layer22_norm = PerGroupRMSNorm(
        source_layer22.input_layernorm,
        calibration["post_mlp_output"]["scales"],
        layer22_output_scales,
    )
    activation_repair = activation_manifest(
        calibration,
        [
            ("activation.layer21_input_rmsnorm", input_norm),
            ("activation.layer21_post_attention_rmsnorm", post_attention_norm),
            ("activation.layer22_input_rmsnorm", layer22_norm),
        ],
        calibration["gated_down_input"]["scales"],
        payload,
    )
    runtime = JointCandidateRuntime()
    fixed_model.model.layers[LAYER_INDEX] = JointLayer21Candidate(
        fixed_layer21,
        source_layer21,
        calibration,
        input_norm,
        post_attention_norm,
        runtime,
        shadow_runtime,
    )
    fixed_layer22.input_layernorm = Layer22CarryRMSNorm(layer22_norm, runtime)

    pack_output = args.pack_output.resolve()
    pack_output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(pack_output, bytes(payload))
    if sha256_file(pack_output) != hashlib.sha256(bytes(payload)).hexdigest():
        raise RuntimeError("written joint-repair pack hash differs")

    prefix = PrefixSubstitution(fixed_model, source_model, LAYER_INDEX)
    prefix.install()
    termination_token_ids = {
        int(tokenizer.eos_token_id),
        int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
    }
    generations = {}
    try:
        for prompt_name, tokenization in tokenizations.items():
            generations[prompt_name] = greedy_generate_candidate(
                fixed_model,
                tokenizer,
                prompt_name,
                tokenization["chat_template_token_ids"],
                termination_token_ids,
                max_new_tokens=args.max_new_tokens,
                top_k_count=args.top_k,
                prefix=prefix,
                runtime=runtime,
            )
            generation = generations[prompt_name]
            print(
                "ACE2_LAYER21_JOINT_PROMPT "
                f"prompt={prompt_name} tokens={generation['generated_token_ids']} "
                f"decoded={generation['decoded_text']!r} "
                f"readable={generation['readability_gate']['passed']} "
                f"semantic={generation['coherence_gate']['passed']}",
                flush=True,
            )
    finally:
        prefix.remove()

    boundary = {
        prompt_name: aggregate_boundaries(runtime.records, prompt_name)
        for prompt_name in generations
    }
    execution = summarize_execution(shadow_runtime)
    readable = all(
        generation["readability_gate"]["passed"]
        for generation in generations.values()
    )
    semantic = all(
        generation["coherence_gate"]["passed"]
        for generation in generations.values()
    )
    passed = readable and semantic
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "chat_v2_layer21_joint_group32_activation_residual_repair_candidate",
        "contract": {
            "acceptance_candidate": False,
            "hardware_realizable_repair_candidate": True,
            "conditioned_exact_prefix_layers": 21,
            "repaired_layer": LAYER_INDEX,
            "repaired_projections": 7,
            "weight_format": "signed_int4_two_values_per_byte",
            "activation_format": "signed_int8_fixed_per_32_channels",
            "activation_scale_derivation": (
                "One fixed Scale32 table takes the per-group absolute maximum over "
                "the two frozen prompt trajectories recorded by the exact prefix-22 "
                "artifact. The table is frozen before candidate generation and shared "
                "by both prompts and every generation step."
            ),
            "residual_join_arithmetic": (
                "Both Layer-21 joins add a signed-32 projection accumulator and a "
                "signed-int8 residual in an exact Scale32 common domain, round once "
                "to nearest ties-to-even, and saturate to signed int8."
            ),
            "layer22_carry": (
                "The post-MLP signed-int8 payload and its fixed 28-entry Scale32 table "
                "are consumed directly by Layer-22 RMSNorm before the frozen Layer-22 "
                "projections."
            ),
            "readability_and_semantic_gates_are_independent": True,
            "scope_limit": (
                "This tests one packable Layer-21 repair under exact Layers 0 through "
                "20. Layers 22 and 23, final RMSNorm, and LM head remain frozen W4A8."
            ),
        },
        "invocation": {"argv": exact_process_argv(), "cwd": str(Path.cwd())},
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": REVISION,
            "snapshot": str(TOKENIZER_SNAPSHOT.resolve()),
        },
        "prompts": {
            "hello_world": "Hello world",
            "second_prompt": args.second_prompt,
        },
        "calibration_trajectories": trajectories,
        "projection_repair_manifest": projection_repair,
        "activation_repair_manifest": activation_repair,
        "pack": {
            "path": str(pack_output),
            "sha256": sha256_file(pack_output),
            "bytes": pack_output.stat().st_size,
            "section_count": sum(
                len(item["sections"]) for item in projection_repair
            )
            + len(activation_repair["scale_sections"])
            + sum(len(item["sections"]) for item in activation_repair["rmsnorm"])
            + len(activation_repair["silu_sections"]),
        },
        "generations": generations,
        "boundary": boundary,
        "boundary_records": runtime.records,
        "frozen_binding_before_repair": frozen_binding_before_repair,
        "shadow_residual_execution": execution,
        "two_prompt_readability_passed": readable,
        "two_prompt_semantic_gate_passed": semantic,
        "two_prompt_joint_gate_passed": passed,
        "status": (
            "PASS_LAYER21_JOINT_GROUP32_TWO_PROMPT_GATES"
            if passed
            else "FAIL_LAYER21_JOINT_GROUP32_TWO_PROMPT_GATES"
        ),
        "artifacts": {
            "prefix21": {"path": str(PREFIX21), "sha256": sha256_file(PREFIX21)},
            "prefix22": {"path": str(PREFIX22), "sha256": sha256_file(PREFIX22)},
            "frozen_scales": {
                "path": str(FROZEN_SCALES),
                "sha256": sha256_file(FROZEN_SCALES),
            },
            "fixed_point_source": {
                "path": str(ROOT / "tools/ace2_full_model_fixed_point.py"),
                "sha256": sha256_file(ROOT / "tools/ace2_full_model_fixed_point.py"),
            },
            "runner_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "wall_seconds": time.perf_counter() - started,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, canonical_bytes(result))
    print(
        "ACE2_LAYER21_JOINT_RESULT "
        f"status={result['status']} wall_seconds={result['wall_seconds']:.6f} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_LAYER21_JOINT_FAIL detail={error}", file=sys.stderr)
        raise
