#!/usr/bin/env python3
"""Closed, mechanism-disabled full-model baseline executor for Dynamic Scale32."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from ace2_software_identity import require_distribution_versions


SCHEMA_VERSION = 1
CONTRACT_ID = "shared_token_group_dynamic_scale32_v1"
MODEL = {
    "repository": "Qwen/Qwen2.5-0.5B",
    "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
}
DATASET_NAMES = ("wikitext2", "c4_en_512")
OUTPUT_NAMES = (
    "ordered_layer_traces.json",
    "final_outputs.json",
    "results.json",
    "run_contract.json",
    "input_observations.json",
)
SUPPORTED_PREFIX = (
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
)
FIRST_UNSUPPORTED = "layer_0.rope_q"
EXPECTED_SOFTWARE = {
    "datasets": "4.8.5",
    "lm_eval": "0.4.9.2",
    "torch": "2.11.0",
    "transformers": "4.57.6",
}
RMS_HIDDEN_SIZE = 896
RMS_GAIN_FRAC = 8
RMS_INV_FRAC = 30
ROPE_HEAD_DIM = 64
SOFTMAX_PROB_FRAC = 15
SOFTMAX_EXP_STEP = 64
SOFTMAX_EXP_ROUND = 32
INT32_MAX = (1 << 31) - 1
INT64_MAX = (1 << 63) - 1
SCALE32_SIGNIFICAND_MIN = 0x8000
SCALE32_SIGNIFICAND_MAX = 0xFFFF
SCALE32_EXPONENT_MIN = -24
SCALE32_EXPONENT_MAX = 4
SCALE32_ALL_ZERO_RECORD = 0x00E88000
SOFTMAX_EXP_LUT = (
    32768, 28918, 25520, 22521, 19875, 17539, 15479, 13660,
    12055, 10638, 9388, 8285, 7312, 6452, 5694, 5025,
    4435, 3914, 3454, 3048, 2690, 2374, 2095, 1849,
    1631, 1440, 1271, 1121, 990, 873, 771, 680,
    600, 530, 467, 412, 364, 321, 283, 250,
    221, 195, 172, 152, 134, 118, 104, 92,
    81, 72, 63, 56, 49, 43, 38, 34,
    30, 26, 23, 21, 18, 16, 14, 12, 11,
)
SILU_LUT = tuple(
    max(
        -32768,
        min(
            32767,
            round((index / 8.0) / (1.0 + math.exp(-index / 8.0)) * (1 << 12)),
        ),
    )
    for index in range(-64, 65)
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(json_bytes(value))


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys differ: {sorted(value)} != {sorted(expected)}")


def pack_scale32(significand: int, exponent: int) -> int:
    if not SCALE32_SIGNIFICAND_MIN <= significand <= SCALE32_SIGNIFICAND_MAX:
        raise ValueError("Scale32 significand is not normalized unsigned Q1.15")
    if not SCALE32_EXPONENT_MIN <= exponent <= SCALE32_EXPONENT_MAX:
        raise ValueError("Scale32 exponent is outside the frozen range")
    return significand | ((exponent & 0xFF) << 16)


def unpack_scale32(record: int) -> tuple[int, int]:
    if not 0 <= record <= 0xFFFFFFFF or record >> 24:
        raise ValueError("Scale32 reserved byte is nonzero")
    significand = record & 0xFFFF
    exponent_u8 = (record >> 16) & 0xFF
    exponent = exponent_u8 - 256 if exponent_u8 & 0x80 else exponent_u8
    pack_scale32(significand, exponent)
    return significand, exponent


def ceil_scale32_from_ratio(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("Scale32 ratio is invalid")
    if numerator == 0:
        return SCALE32_ALL_ZERO_RECORD
    for exponent in range(SCALE32_EXPONENT_MIN, SCALE32_EXPONENT_MAX + 1):
        shift = 15 - exponent
        scaled_numerator = numerator << max(shift, 0)
        scaled_denominator = denominator << max(-shift, 0)
        significand = (scaled_numerator + scaled_denominator - 1) // scaled_denominator
        significand = max(significand, SCALE32_SIGNIFICAND_MIN)
        if significand <= SCALE32_SIGNIFICAND_MAX:
            return pack_scale32(significand, exponent)
    raise OverflowError("positive scale exceeds the frozen Scale32 range")


def ceil_scale32_from_float(value: float) -> int:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Scale32 input must be finite and positive")
    return ceil_scale32_from_ratio(*value.as_integer_ratio())


def scale32_ratio(record: int) -> tuple[int, int]:
    significand, exponent = unpack_scale32(record)
    if exponent >= 15:
        return significand << (exponent - 15), 1
    return significand, 1 << (15 - exponent)


def dynamic_rope_output_scale(producer_record: int, maximum: int) -> int:
    producer_sig, producer_exp = unpack_scale32(producer_record)
    if maximum < 0:
        raise ValueError("RoPE magnitude is negative")
    if maximum == 0:
        return SCALE32_ALL_ZERO_RECORD
    numerator = producer_sig * maximum
    denominator = 127 << 30
    if producer_exp >= 0:
        numerator <<= producer_exp
    else:
        denominator <<= -producer_exp
    return ceil_scale32_from_ratio(numerator, denominator)


def selected_texts(load_dataset: Callable[..., Any], spec: dict[str, Any], limit: int | None) -> list[str]:
    dataset = load_dataset(
        spec["repository"],
        spec["config"],
        split=spec["split"],
        revision=spec["revision"],
        streaming=True,
    )
    start = spec["indices"]["start"]
    stop = spec["indices"]["stop"]
    if limit is not None:
        stop = min(stop, start + limit)
    records: list[str] = []
    for index, row in enumerate(dataset):
        if index >= stop:
            break
        if index >= start:
            records.append(row[spec["field"]])
    if len(records) != stop - start:
        raise RuntimeError(f"frozen dataset slice returned {len(records)} records, expected {stop - start}")
    return records


def hash_records(records: Iterable[str]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for record in records:
        encoded = record.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        count += 1
    return digest.hexdigest(), count


def load_contracts(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "benchmark/quality/PROMPT_MANIFEST.json"
    config_path = root / "benchmark/quality/QUALITY_CONFIG.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if manifest.get("model") != MODEL:
        raise ValueError("prompt manifest model identity differs")
    if config.get("baseline") != {
        "dtype": "bfloat16",
        "model_repository": MODEL["repository"],
        "model_revision": MODEL["revision"],
    }:
        raise ValueError("quality baseline identity differs")
    if config.get("software") != EXPECTED_SOFTWARE:
        raise ValueError("software pins differ")
    if tuple(name for name in DATASET_NAMES if name in manifest.get("datasets", {})) != DATASET_NAMES:
        raise ValueError("required dataset contracts are missing")
    if config.get("weight_quantization", {}).get("packing_group_size") != 128:
        raise ValueError("W4 packing group differs")
    if config.get("arithmetic", {}).get("rounding") != "round_to_nearest_ties_to_even":
        raise ValueError("rounding contract differs")
    if config.get("arithmetic", {}).get("accumulator") != "signed_int32_across_the_complete_projection_reduction":
        raise ValueError("projection accumulator contract differs")
    return manifest, config


def validate_mode(mode: str, smoke_token_limit: int | None) -> dict[str, Any]:
    if mode == "smoke":
        if not isinstance(smoke_token_limit, int) or not 2 <= smoke_token_limit <= 512:
            raise ValueError("smoke token limit must be in 2..512")
        return {
            "mode": mode,
            "record_limits": {"c4_calibration": 1, "wikitext2": 16, "c4_en_512": 1},
            "token_limit": smoke_token_limit,
        }
    if smoke_token_limit is not None:
        raise ValueError("official mode cannot set a smoke token limit")
    return {
        "mode": mode,
        "record_limits": {"c4_calibration": None, "wikitext2": None, "c4_en_512": None},
        "token_limit": None,
    }


def seed_everything(config: dict[str, Any], np: Any, torch: Any) -> None:
    seeds = config["determinism"]
    exact_keys(
        seeds,
        {"datasets_seed", "numpy_seed", "python_seed", "torch_deterministic_algorithms", "torch_seed"},
        "determinism",
    )
    os.environ["PYTHONHASHSEED"] = str(seeds["python_seed"])
    os.environ["HF_DATASETS_RANDOM_SEED"] = str(seeds["datasets_seed"])
    random.seed(seeds["python_seed"])
    np.random.seed(seeds["numpy_seed"])
    torch.manual_seed(seeds["torch_seed"])
    torch.use_deterministic_algorithms(seeds["torch_deterministic_algorithms"])


def tensor_descriptor(torch: Any, value: Any) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    encoded = tensor.view(torch.uint8).numpy().tobytes()
    floating = tensor.to(torch.float64)
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "sha256": sha256_bytes(encoded),
        "minimum": float(floating.min()) if tensor.numel() else None,
        "maximum": float(floating.max()) if tensor.numel() else None,
        "squared_l2": float((floating * floating).sum()),
    }


class TraceCollector:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.sequences: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None

    def begin(self, dataset: str, index: int, input_ids: Any) -> None:
        if self.current is not None:
            raise RuntimeError("trace sequence already active")
        self.current = {
            "dataset": dataset,
            "sequence_index": index,
            "input_ids": tensor_descriptor(self.torch, input_ids),
            "traces": [],
        }

    def record(self, operator: str, value: Any) -> None:
        if self.current is not None:
            self.current["traces"].append({
                "ordinal": len(self.current["traces"]),
                "operator": operator,
                "tensor": tensor_descriptor(self.torch, value),
            })

    def end(self) -> None:
        if self.current is None:
            raise RuntimeError("no active trace sequence")
        self.sequences.append(self.current)
        self.current = None


@dataclass
class CalibrationRange:
    input_absmax: float = 0.0
    output_absmax: float = 0.0


def make_fixed_runtime(torch: Any, nn: Any, config: dict[str, Any], collector: TraceCollector) -> dict[str, Any]:
    def round_shift_even(value: Any, shift: Any) -> Any:
        shift_tensor = torch.as_tensor(shift, dtype=torch.int64, device=value.device)
        if torch.any(shift_tensor < 0) or torch.any(shift_tensor > 63):
            raise ValueError("right shift is outside 0..63")
        safe = shift_tensor.clamp(min=1)
        magnitude = value.abs()
        base = torch.bitwise_right_shift(magnitude, safe)
        shifted_one = torch.bitwise_left_shift(torch.ones_like(safe), safe)
        mask = torch.where(safe == 63, torch.full_like(safe, INT64_MAX), shifted_one - 1)
        remainder = torch.bitwise_and(magnitude, mask)
        half = torch.bitwise_left_shift(torch.ones_like(safe), safe - 1)
        increment = (remainder > half) | ((remainder == half) & ((base & 1) == 1))
        rounded = base + increment.to(base.dtype)
        signed = torch.where(value < 0, -rounded, rounded)
        return torch.where(shift_tensor == 0, value, signed)

    def round_div_even_unsigned_tensor(numerator: Any, denominator: Any) -> Any:
        quotient = torch.div(numerator, denominator, rounding_mode="floor")
        remainder = numerator - quotient * denominator
        doubled = remainder * 2
        return quotient + ((doubled > denominator) | ((doubled == denominator) & ((quotient & 1) == 1))).to(torch.int64)

    def derive_multiplier(real_multiplier: Any) -> tuple[Any, Any]:
        real = real_multiplier.detach().to(torch.float64)
        multiplier = torch.zeros_like(real, dtype=torch.int64)
        right_shift = torch.full_like(real, -1, dtype=torch.int64)
        for shift in range(63, -1, -1):
            rounded = torch.round(real * math.ldexp(1.0, shift))
            select = (right_shift < 0) & (rounded <= INT32_MAX)
            multiplier = torch.where(select, rounded.to(torch.int64), multiplier)
            right_shift = torch.where(select, torch.full_like(right_shift, shift), right_shift)
        if torch.any(right_shift < 0):
            raise OverflowError("requantization multiplier is not representable")
        return multiplier, right_shift

    def positive_scale(absmax: float) -> float:
        if not math.isfinite(absmax) or absmax <= 0:
            raise ValueError("calibration range must be finite and positive")
        return absmax / 127.0

    def quantize_int8(value: Any, scale: float) -> Any:
        return torch.round(value.to(torch.float64) / scale).clamp(-128, 127).to(torch.int8)

    class W4A8Linear(nn.Module):
        def __init__(self, source: Any, calibration: CalibrationRange, *, scale32_output: bool = False) -> None:
            super().__init__()
            weight = source.weight.detach().to(torch.float64)
            self.in_features = source.in_features
            self.out_features = source.out_features
            self.input_scale = positive_scale(calibration.input_absmax)
            output_scale = positive_scale(calibration.output_absmax)
            self.output_scale32_record = ceil_scale32_from_float(output_scale) if scale32_output else None
            if self.output_scale32_record is not None:
                numerator, denominator = scale32_ratio(self.output_scale32_record)
                output_scale = numerator / denominator
            self.output_scale = output_scale
            weight_scale = weight.abs().amax(dim=1) / 7.0
            weight_scale = torch.where(weight_scale > 0, weight_scale, torch.ones_like(weight_scale))
            qweight = torch.round(weight / weight_scale[:, None]).clamp(-8, 7).to(torch.int8)
            self.register_buffer("qweight_transposed", qweight.transpose(0, 1).contiguous())
            self.register_buffer("weight_scale", weight_scale)
            self.register_buffer("source_bias", source.bias.detach().to(torch.float64) if source.bias is not None else None)
            self.quantized_input = False
            self.bind_hardware_input_scale(self.input_scale)

        def bind_hardware_input_scale(self, input_scale: float) -> None:
            self.hardware_input_scale = input_scale
            real_multiplier = input_scale * self.weight_scale / self.output_scale
            multiplier, right_shift = derive_multiplier(real_multiplier)
            bias = None if self.source_bias is None else torch.round(self.source_bias / (input_scale * self.weight_scale)).to(torch.int64)
            for name, value in (("multiplier", multiplier), ("right_shift", right_shift), ("bias_accumulator", bias)):
                if hasattr(self, name):
                    current = getattr(self, name)
                    if current is not None and value is not None:
                        current.copy_(value)
                else:
                    self.register_buffer(name, value)

        def bind_quantized_input_scale(self, input_scale: float) -> None:
            self.bind_hardware_input_scale(input_scale)
            self.quantized_input = True

        def accumulator_quantized(self, qinput: Any) -> Any:
            flat = qinput.reshape(-1, self.in_features).contiguous()
            accumulator = torch._int_mm(flat, self.qweight_transposed).to(torch.int64)
            if self.bias_accumulator is not None:
                accumulator = accumulator + self.bias_accumulator
            if torch.any(accumulator < -(1 << 31)) or torch.any(accumulator >= (1 << 31)):
                raise OverflowError("projection accumulator exceeds signed-32")
            return accumulator

        def forward_hardware_input(self, inputs: Any) -> Any:
            qinput = inputs if inputs.dtype == torch.int8 else inputs.to(torch.int8)
            accumulator = self.accumulator_quantized(qinput)
            product = accumulator * self.multiplier
            output = round_shift_even(product, self.right_shift).clamp(-128, 127).to(torch.int8)
            return output.reshape(*qinput.shape[:-1], self.out_features)

        def forward(self, inputs: Any) -> Any:
            qinput = inputs.to(torch.int8) if self.quantized_input else quantize_int8(inputs, self.input_scale)
            raw = self.forward_hardware_input(qinput)
            return raw.to(inputs.dtype) * self.output_scale

    class FixedRMSNorm(nn.Module):
        def __init__(self, source: Any, input_scale: float, output_scale: float, operator: str) -> None:
            super().__init__()
            gain = torch.round(source.weight.detach().to(torch.float64) / output_scale * (1 << RMS_GAIN_FRAC))
            if torch.any(gain < -32768) or torch.any(gain > 32767):
                raise OverflowError("RMSNorm gain is not signed-int16")
            self.input_scale = input_scale
            self.output_scale = output_scale
            self.operator = operator
            self.register_buffer("scaled_gains_q8", gain.to(torch.int16))

        def forward(self, hidden_states: Any) -> Any:
            activations = quantize_int8(hidden_states, self.input_scale)
            values = activations.to(torch.int64)
            sumsq = (values * values).sum(dim=-1, keepdim=True)
            mean_square = (sumsq + RMS_HIDDEN_SIZE // 2) // RMS_HIDDEN_SIZE
            root = torch.floor(torch.sqrt(mean_square.to(torch.float64))).to(torch.int64)
            root = (root + (root * root < mean_square).to(torch.int64)).clamp(min=1)
            product = values * self.scaled_gains_q8.to(torch.int64) * ((1 << RMS_INV_FRAC) // root)
            raw = round_shift_even(product, RMS_INV_FRAC + RMS_GAIN_FRAC).clamp(-128, 127).to(torch.int8)
            collector.record(self.operator, raw)
            return raw.to(hidden_states.dtype)

    def dynamic_rope_head_raw(activations: Any, producer_record: int, cos: Any, sin: Any) -> tuple[Any, Any]:
        producer_sig, producer_exp = unpack_scale32(producer_record)
        cos_q15 = torch.round(cos.to(torch.float64) * 32767.0).clamp(-32768, 32767).to(torch.int64).unsqueeze(1)
        sin_q15 = torch.round(sin.to(torch.float64) * 32767.0).clamp(-32768, 32767).to(torch.int64).unsqueeze(1)
        values = activations.to(torch.int64)
        first, second = values[..., :32], values[..., 32:]
        rotated = torch.cat((first * cos_q15[..., :32] - second * sin_q15[..., :32], second * cos_q15[..., 32:] + first * sin_q15[..., 32:]), dim=-1)
        maximum = rotated.abs().amax(dim=-1)
        records = torch.tensor([dynamic_rope_output_scale(producer_record, int(item)) for item in maximum.reshape(-1).tolist()], dtype=torch.int64, device=activations.device).reshape(maximum.shape)
        output_sig = torch.bitwise_and(records, 0xFFFF)
        output_exp_u8 = torch.bitwise_and(torch.bitwise_right_shift(records, 16), 0xFF)
        output_exp = torch.where(output_exp_u8 >= 128, output_exp_u8 - 256, output_exp_u8)
        exponent_delta = producer_exp - output_exp - 15
        numerator = torch.bitwise_left_shift(rotated * producer_sig, exponent_delta.clamp(min=0).unsqueeze(-1))
        denominator = torch.bitwise_left_shift(output_sig, (-exponent_delta).clamp(min=0)).unsqueeze(-1)
        rounded = round_div_even_unsigned_tensor(numerator.abs(), denominator)
        rounded = torch.where(numerator < 0, -rounded, rounded)
        if torch.any(rounded < -127) or torch.any(rounded > 127):
            raise OverflowError("dynamic RoPE output exceeded symmetric int8")
        return rounded.to(torch.int8), records

    def repeat_kv(hidden_states: Any, repeats: int) -> Any:
        if repeats == 1:
            return hidden_states
        batch, heads, sequence, head_dim = hidden_states.shape
        return hidden_states[:, :, None, :, :].expand(batch, heads, repeats, sequence, head_dim).reshape(batch, heads * repeats, sequence, head_dim)

    def dynamic_scores(query: Any, key: Any, query_scale32: Any, key_scale32: Any, mask: Any) -> Any:
        accumulator = torch.matmul(query.to(torch.int32), key.to(torch.int32).transpose(2, 3)).to(torch.int64)
        query_sig = torch.bitwise_and(query_scale32, 0xFFFF).unsqueeze(-1)
        key_sig = torch.bitwise_and(key_scale32, 0xFFFF).unsqueeze(-2)
        pair_sig = round_div_even_unsigned_tensor(query_sig * key_sig, 1 << 15)
        query_exp_u8 = torch.bitwise_and(torch.bitwise_right_shift(query_scale32, 16), 0xFF).unsqueeze(-1)
        key_exp_u8 = torch.bitwise_and(torch.bitwise_right_shift(key_scale32, 16), 0xFF).unsqueeze(-2)
        query_exp = torch.where(query_exp_u8 >= 128, query_exp_u8 - 256, query_exp_u8)
        key_exp = torch.where(key_exp_u8 >= 128, key_exp_u8 - 256, key_exp_u8)
        scores = round_shift_even(accumulator * pair_sig, 9 - (query_exp + key_exp))
        if mask is not None:
            valid = mask[:, :, :, : key.shape[-2]] >= 0
            row_max = torch.where(valid, scores, torch.full_like(scores, torch.iinfo(torch.int64).min)).amax(dim=-1, keepdim=True)
            scores = torch.where(valid, scores - row_max, torch.full_like(scores, -32768))
        else:
            scores = scores - scores.amax(dim=-1, keepdim=True)
        return scores.clamp(-32768, 0).to(torch.int16)

    def fixed_softmax(scores: Any) -> Any:
        magnitude = scores.to(torch.int64).amax(dim=-1, keepdim=True) - scores.to(torch.int64)
        table_index = torch.div(magnitude + SOFTMAX_EXP_ROUND, SOFTMAX_EXP_STEP, rounding_mode="floor")
        table = torch.tensor(SOFTMAX_EXP_LUT, dtype=torch.int64, device=scores.device)
        weights = table[table_index.clamp(max=len(SOFTMAX_EXP_LUT) - 1)]
        weights = torch.where(table_index < len(SOFTMAX_EXP_LUT), weights, torch.zeros_like(weights))
        return round_div_even_unsigned_tensor(weights << SOFTMAX_PROB_FRAC, weights.sum(dim=-1, keepdim=True))

    class FixedAttention(nn.Module):
        def __init__(self, source: Any, normalized_input_scale: float) -> None:
            super().__init__()
            self.layer_idx = source.layer_idx
            self.head_dim = source.head_dim
            self.num_key_value_groups = source.num_key_value_groups
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = source.q_proj, source.k_proj, source.v_proj, source.o_proj
            for module in (self.q_proj, self.k_proj, self.v_proj):
                module.bind_hardware_input_scale(normalized_input_scale)
            if self.q_proj.output_scale32_record is None or self.k_proj.output_scale32_record is None:
                raise ValueError("Q/K projections lack Scale32 producer records")
            self.o_proj.bind_hardware_input_scale(self.v_proj.output_scale)

        def forward(self, hidden_states: Any, position_embeddings: tuple[Any, Any], attention_mask: Any, past_key_values: Any = None, **_: Any) -> tuple[Any, None]:
            if past_key_values is not None:
                raise RuntimeError("cache-free scoring is required")
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query = self.q_proj.forward_hardware_input(hidden_states).view(hidden_shape).transpose(1, 2)
            key = self.k_proj.forward_hardware_input(hidden_states).view(hidden_shape).transpose(1, 2)
            value = self.v_proj.forward_hardware_input(hidden_states).view(hidden_shape).transpose(1, 2)
            prefix = f"layer_{self.layer_idx}"
            collector.record(f"{prefix}.q_proj", query)
            collector.record(f"{prefix}.k_proj", key)
            collector.record(f"{prefix}.v_proj", value)
            cos, sin = position_embeddings
            query, query_scale32 = dynamic_rope_head_raw(query, self.q_proj.output_scale32_record, cos, sin)
            key, key_scale32 = dynamic_rope_head_raw(key, self.k_proj.output_scale32_record, cos, sin)
            collector.record(f"{prefix}.rope_q", query)
            collector.record(f"{prefix}.rope_k", key)
            key = repeat_kv(key, self.num_key_value_groups)
            value = repeat_kv(value, self.num_key_value_groups)
            key_scale32 = key_scale32.repeat_interleave(self.num_key_value_groups, dim=1)
            scores = dynamic_scores(query, key, query_scale32, key_scale32, attention_mask)
            collector.record(f"{prefix}.attention_scores", scores)
            probabilities = fixed_softmax(scores)
            collector.record(f"{prefix}.softmax", probabilities)
            attention_value = round_shift_even(torch.matmul(probabilities, value.to(torch.int64)), SOFTMAX_PROB_FRAC).clamp(-128, 127).to(torch.int8)
            collector.record(f"{prefix}.attention_value", attention_value)
            flattened = attention_value.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            projected = self.o_proj.forward_hardware_input(flattened)
            collector.record(f"{prefix}.o_proj", projected)
            return projected.to(hidden_states.dtype) * self.o_proj.output_scale, None

    class FixedMLP(nn.Module):
        def __init__(self, source: Any, normalized_input_scale: float, layer_index: int) -> None:
            super().__init__()
            self.gate_proj, self.up_proj, self.down_proj = source.gate_proj, source.up_proj, source.down_proj
            self.layer_index = layer_index
            for module in (self.gate_proj, self.up_proj):
                module.bind_hardware_input_scale(normalized_input_scale)
            multiplier, right_shift = derive_multiplier(torch.tensor([1.0 / ((1 << 21) * self.down_proj.input_scale)], dtype=torch.float64))
            self.register_buffer("silu_multiplier", multiplier)
            self.register_buffer("silu_right_shift", right_shift)

        def forward(self, hidden_states: Any) -> Any:
            prefix = f"layer_{self.layer_index}"
            gate = self.gate_proj.forward_hardware_input(hidden_states)
            up = self.up_proj.forward_hardware_input(hidden_states)
            collector.record(f"{prefix}.gate_proj", gate)
            collector.record(f"{prefix}.up_proj", up)
            gate_q6_9 = torch.round(gate.to(torch.float64) * self.gate_proj.output_scale * (1 << 9)).clamp(-32768, 32767).to(torch.int64)
            up_q6_9 = torch.round(up.to(torch.float64) * self.up_proj.output_scale * (1 << 9)).clamp(-32768, 32767).to(torch.int64)
            table = torch.tensor(SILU_LUT, dtype=torch.int64, device=gate.device)
            silu_q3_12 = table[torch.bitwise_right_shift(gate_q6_9, 6).clamp(-64, 64) + 64]
            gated = round_shift_even(silu_q3_12 * up_q6_9 * self.silu_multiplier, self.silu_right_shift).clamp(-128, 127).to(torch.int8)
            collector.record(f"{prefix}.silu_gate", gated)
            down = self.down_proj.forward_hardware_input(gated)
            collector.record(f"{prefix}.down_proj", down)
            return down.to(hidden_states.dtype) * self.down_proj.output_scale

    class FixedDecoderLayer(nn.Module):
        def __init__(self, source: Any, layer_index: int, scales: dict[str, float]) -> None:
            super().__init__()
            self.attention_type = source.attention_type
            self.layer_index = layer_index
            self.input_layernorm = FixedRMSNorm(source.input_layernorm, scales["input"], scales["input_norm"], f"layer_{layer_index}.input_rmsnorm")
            self.post_attention_layernorm = FixedRMSNorm(source.post_attention_layernorm, scales["post_attention"], scales["post_attention_norm"], f"layer_{layer_index}.post_attention_rmsnorm")
            self.self_attn = FixedAttention(source.self_attn, self.input_layernorm.output_scale)
            self.mlp = FixedMLP(source.mlp, self.post_attention_layernorm.output_scale, layer_index)
            self.post_attention_scale = scales["post_attention"]
            self.post_mlp_scale = scales["post_mlp"]

        def residual_add(self, lhs: Any, rhs: Any, scale: float) -> Any:
            raw = (quantize_int8(lhs, scale).to(torch.int16) + quantize_int8(rhs, scale).to(torch.int16)).clamp(-128, 127).to(torch.int8)
            return raw.to(lhs.dtype) * scale

        def forward(self, hidden_states: Any, attention_mask: Any = None, past_key_values: Any = None, position_embeddings: tuple[Any, Any] | None = None, **kwargs: Any) -> Any:
            if position_embeddings is None:
                raise ValueError("position embeddings are required")
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
            hidden_states = self.residual_add(residual, hidden_states, self.post_attention_scale)
            collector.record(f"layer_{self.layer_index}.post_attention_residual", quantize_int8(hidden_states, self.post_attention_scale))
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.residual_add(residual, hidden_states, self.post_mlp_scale)
            collector.record(f"layer_{self.layer_index}.post_mlp_residual", quantize_int8(hidden_states, self.post_mlp_scale))
            return hidden_states

    return {
        "W4A8Linear": W4A8Linear,
        "FixedRMSNorm": FixedRMSNorm,
        "FixedDecoderLayer": FixedDecoderLayer,
        "positive_scale": positive_scale,
    }


def execute_real(mode_contract: dict[str, Any], root: Path, manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    validated_software = require_distribution_versions(EXPECTED_SOFTWARE)
    import datasets
    import numpy as np
    import torch
    import transformers
    from datasets import load_dataset
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_everything(config, np, torch)
    limits = mode_contract["record_limits"]
    text = {
        name: selected_texts(load_dataset, manifest["datasets"][name], limits[name])
        for name in ("c4_calibration", *DATASET_NAMES)
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL["repository"], revision=MODEL["revision"])
    if tokenizer.init_kwargs.get("_commit_hash") not in (None, MODEL["revision"]):
        raise RuntimeError("tokenizer resolved revision differs")

    def tokenize(name: str) -> list[Any]:
        spec = manifest["datasets"][name]
        if name == "wikitext2":
            ids = tokenizer(spec["join"].join(text[name]), add_special_tokens=False, return_tensors="pt")["input_ids"]
            prompts = [ids[:, start : start + spec["token_limit"]] for start in range(0, ids.shape[1], spec["token_limit"])]
            prompts = [item for item in prompts if item.shape[1] >= 2]
        else:
            prompts = [tokenizer(item, add_special_tokens=False, truncation=True, max_length=spec["token_limit"], return_tensors="pt")["input_ids"] for item in text[name]]
        if mode_contract["mode"] == "smoke":
            prompts = [prompts[0][:, : mode_contract["token_limit"]]]
        if not prompts or any(item.shape[1] < 2 for item in prompts):
            raise RuntimeError(f"{name} produced no scoreable prompts")
        return prompts

    prompts = {name: tokenize(name) for name in ("c4_calibration", *DATASET_NAMES)}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL["repository"],
        revision=MODEL["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    if getattr(model.config, "_commit_hash", None) != MODEL["revision"]:
        raise RuntimeError("model resolved revision differs")
    geometry = {
        "hidden_size": model.config.hidden_size,
        "head_dim": model.config.head_dim,
        "num_attention_heads": model.config.num_attention_heads,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_key_value_heads": model.config.num_key_value_heads,
    }
    if geometry != {"hidden_size": 896, "head_dim": 64, "num_attention_heads": 14, "num_hidden_layers": 24, "num_key_value_heads": 2}:
        raise RuntimeError(f"model geometry differs: {geometry}")

    linear_ranges = {name: CalibrationRange() for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    operator_ranges: dict[str, float] = {}
    hooks: list[Any] = []

    def observe(key: str, value: Any) -> None:
        operator_ranges[key] = max(operator_ranges.get(key, 0.0), float(value.detach().abs().amax()))

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            def linear_hook(_module: Any, inputs: tuple[Any, ...], output: Any, *, key: str = name) -> None:
                linear_ranges[key].input_absmax = max(linear_ranges[key].input_absmax, float(inputs[0].detach().abs().amax()))
                linear_ranges[key].output_absmax = max(linear_ranges[key].output_absmax, float(output.detach().abs().amax()))
            hooks.append(module.register_forward_hook(linear_hook))
    for index, layer in enumerate(model.model.layers):
        hooks.append(layer.input_layernorm.register_forward_pre_hook(lambda _m, x, key=f"layer_{index}.input": observe(key, x[0])))
        hooks.append(layer.input_layernorm.register_forward_hook(lambda _m, _x, y, key=f"layer_{index}.input_norm": observe(key, y)))
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda _m, x, key=f"layer_{index}.post_attention": observe(key, x[0])))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(lambda _m, _x, y, key=f"layer_{index}.post_attention_norm": observe(key, y)))
        hooks.append(layer.register_forward_hook(lambda _m, _x, y, key=f"layer_{index}.post_mlp": observe(key, y)))
    hooks.append(model.model.norm.register_forward_pre_hook(lambda _m, x: observe("final.input", x[0])))
    hooks.append(model.model.norm.register_forward_hook(lambda _m, _x, y: observe("final.output", y)))
    try:
        with torch.inference_mode():
            for input_ids in prompts["c4_calibration"]:
                model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    collector = TraceCollector(torch)
    runtime = make_fixed_runtime(torch, nn, config, collector)
    W4A8Linear = runtime["W4A8Linear"]
    replacements = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    for name, module in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, W4A8Linear(module, linear_ranges[name], scale32_output=name.endswith((".q_proj", ".k_proj"))))
    fixed_layers = []
    for index, layer in enumerate(model.model.layers):
        input_norm_scale = max(runtime["positive_scale"](operator_ranges[f"layer_{index}.input_norm"]), float(layer.input_layernorm.weight.detach().to(torch.float64).abs().amax()) * (1 << RMS_GAIN_FRAC) / 32767.0)
        post_norm_scale = max(runtime["positive_scale"](operator_ranges[f"layer_{index}.post_attention_norm"]), float(layer.post_attention_layernorm.weight.detach().to(torch.float64).abs().amax()) * (1 << RMS_GAIN_FRAC) / 32767.0)
        fixed_layers.append(runtime["FixedDecoderLayer"](layer, index, {
            "input": runtime["positive_scale"](operator_ranges[f"layer_{index}.input"]),
            "input_norm": input_norm_scale,
            "post_attention": runtime["positive_scale"](operator_ranges[f"layer_{index}.post_attention"]),
            "post_attention_norm": post_norm_scale,
            "post_mlp": runtime["positive_scale"](operator_ranges[f"layer_{index}.post_mlp"]),
        }))
    model.model.layers = nn.ModuleList(fixed_layers)
    final_norm_source = model.model.norm
    final_output_scale = max(runtime["positive_scale"](operator_ranges["final.output"]), float(final_norm_source.weight.detach().to(torch.float64).abs().amax()) * (1 << RMS_GAIN_FRAC) / 32767.0)
    model.model.norm = runtime["FixedRMSNorm"](final_norm_source, runtime["positive_scale"](operator_ranges["final.input"]), final_output_scale, "final_rmsnorm")
    model.lm_head.bind_quantized_input_scale(model.model.norm.output_scale)

    final_outputs: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    with torch.inference_mode():
        for dataset_name in DATASET_NAMES:
            total_nll = 0.0
            scored_tokens = 0
            sequences = []
            for index, input_ids in enumerate(prompts[dataset_name]):
                collector.begin(dataset_name, index, input_ids)
                output = model(input_ids=input_ids, use_cache=False)
                logits = output.logits[:, :-1, :].to(torch.float32)
                collector.record("lm_head", logits)
                collector.end()
                labels = input_ids[:, 1:]
                nll = float(nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"))
                total_nll += nll
                scored_tokens += labels.numel()
                descriptor = tensor_descriptor(torch, logits)
                sequence = {
                    "sequence_index": index,
                    "input_ids": tensor_descriptor(torch, input_ids),
                    "logits": descriptor,
                    "negative_log_likelihood": nll,
                    "scored_tokens": labels.numel(),
                    "predicted_token_ids": logits.argmax(dim=-1).reshape(-1).tolist(),
                }
                sequences.append(sequence)
                final_outputs.append({"dataset": dataset_name, **sequence})
            mean_nll = total_nll / scored_tokens
            metrics[dataset_name] = {
                "mean_negative_log_likelihood": mean_nll,
                "negative_log_likelihood": total_nll,
                "perplexity": math.exp(mean_nll),
                "scored_tokens": scored_tokens,
                "sequences": sequences,
            }
    observations = {}
    for name in ("c4_calibration", *DATASET_NAMES):
        digest, count = hash_records(text[name])
        spec = manifest["datasets"][name]
        observations[name] = {
            "config": spec["config"],
            "indices": spec["indices"],
            "record_count": count,
            "record_sha256": digest,
            "repository": spec["repository"],
            "revision": spec["revision"],
            "split": spec["split"],
            "token_limit": spec["token_limit"],
        }
    return {
        "ordered_traces": collector.sequences,
        "final_outputs": final_outputs,
        "metrics": metrics,
        "observations": observations,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "software": validated_software,
        "geometry": geometry,
    }


def validate_payload(payload: dict[str, Any]) -> None:
    exact_keys(payload, {"ordered_traces", "final_outputs", "metrics", "observations", "resolved_model_revision", "resolved_tokenizer_revision", "software", "geometry"}, "execution payload")
    if payload["resolved_model_revision"] != MODEL["revision"]:
        raise ValueError("payload model revision differs")
    if payload["resolved_tokenizer_revision"] not in (None, MODEL["revision"]):
        raise ValueError("payload tokenizer revision differs")
    if payload["software"] != EXPECTED_SOFTWARE:
        raise ValueError("payload software differs")
    if not isinstance(payload["ordered_traces"], list) or not isinstance(payload["final_outputs"], list):
        raise ValueError("payload traces or outputs are not lists")


def emit_outputs(
    output_dir: Path,
    mode_contract: dict[str, Any],
    root: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    validate_payload(payload)
    output_dir.mkdir(parents=True, exist_ok=False)
    common = {"schema_version": SCHEMA_VERSION, "contract_id": CONTRACT_ID}
    traces = {
        **common,
        "kind": "dynamic_scale32_disabled_baseline_ordered_layer_traces",
        "ordering": "dataset_then_sequence_then_forward_execution",
        "sequences": payload["ordered_traces"],
    }
    final_outputs = {
        **common,
        "kind": "dynamic_scale32_disabled_baseline_final_outputs",
        "datasets": list(DATASET_NAMES),
        "sequences": payload["final_outputs"],
    }
    input_observations = {
        **common,
        "kind": "dynamic_scale32_disabled_baseline_input_observations",
        "datasets": payload["observations"],
        "environment": dict(os.environ),
        "environment_sha256": sha256_bytes(canonical_json_bytes(dict(os.environ))),
        "mode_scope": mode_contract,
    }
    results = {
        **common,
        "kind": "dynamic_scale32_disabled_baseline_results",
        "classification": "baseline_measurement_only",
        "gate_passed": False,
        "mechanism_disabled": True,
        "mode": mode_contract["mode"],
        "model": {**MODEL, "resolved_revision": payload["resolved_model_revision"]},
        "metrics": payload["metrics"],
        "runtime": {
            "device": "cpu",
            "packages": payload["software"],
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "seeds": config["determinism"],
    }
    values = {
        "ordered_layer_traces.json": traces,
        "final_outputs.json": final_outputs,
        "results.json": results,
        "input_observations.json": input_observations,
    }
    for name, value in values.items():
        write_new_json(output_dir / name, value)
    artifact_hashes = {
        name: sha256_file(output_dir / name)
        for name in sorted(values)
    }
    run_contract = {
        **common,
        "kind": "dynamic_scale32_disabled_baseline_run_contract",
        "artifact_hashes": artifact_hashes,
        "architecture_boundary": {
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "mode": "ADVANCE",
            "ordered_supported_layer_operator_prefix": list(SUPPORTED_PREFIX),
        },
        "baseline_model": 1,
        "candidate_model": 0,
        "datasets": {name: manifest["datasets"][name] for name in ("c4_calibration", *DATASET_NAMES)},
        "determinism": config["determinism"],
        "mechanism": CONTRACT_ID,
        "mechanism_disabled": True,
        "mode_scope": mode_contract,
        "model": {
            **MODEL,
            "resolved_revision": payload["resolved_model_revision"],
            "tokenizer_resolved_revision": payload["resolved_tokenizer_revision"],
        },
        "numerical_contract": {
            "activation_quantization": config["activation_quantization"],
            "arithmetic": config["arithmetic"],
            "full_model_scope": config["full_model_scope"],
            "weight_quantization": config["weight_quantization"],
        },
        "output_schema": {
            "files": list(OUTPUT_NAMES),
            "schema_version": SCHEMA_VERSION,
        },
        "software_versions": payload["software"],
        "source_contract_hashes": {
            "executor": sha256_file(Path(__file__)),
            "prompt_manifest": sha256_file(root / "benchmark/quality/PROMPT_MANIFEST.json"),
            "quality_config": sha256_file(root / "benchmark/quality/QUALITY_CONFIG.json"),
            "requirements": sha256_file(root / "benchmark/quality/requirements.txt"),
        },
        "status": "disabled_mechanism_baseline_complete",
    }
    write_new_json(output_dir / "run_contract.json", run_contract)
    if {path.name for path in output_dir.iterdir()} != set(OUTPUT_NAMES):
        raise RuntimeError("executor output set differs from the frozen five artifacts")


def execute(
    mode: str,
    output_dir: Path,
    smoke_token_limit: int | None,
) -> None:
    mode_contract = validate_mode(mode, smoke_token_limit)
    root = Path(__file__).resolve().parents[1]
    manifest, config = load_contracts(root)
    payload = execute_real(mode_contract, root, manifest, config)
    emit_outputs(output_dir, mode_contract, root, manifest, config, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "official"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-token-limit", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    execute(args.mode, args.output_dir, args.smoke_token_limit)
    print(f"DYNAMIC_SCALE32_DISABLED_BASELINE_COMPLETE output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
