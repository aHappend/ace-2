#!/usr/bin/env python3
"""Independent Scale32 reference for the selected layer-16 group-4 path."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from safetensors import safe_open

from ace2_quality_contracts import (
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    round_divide_even_signed,
    round_divide_even_unsigned,
    scale32_ratio,
    unpack_scale32,
)
from ace2_rmsnorm_reference import derive_scaled_gains_q8, reference_rmsnorm
from run_lora_v4_layer0_full_rtl import ADAPTER, LORA_ALPHA, LORA_RANK, MODEL


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "evidence/candidates/w4a8-layer16-source-grouped-activation-repair-v1/"
    "candidate-0001/groups/group-004"
)
HIDDEN_SIZE = 896
GROUP_SIZE = 4
GROUP_COUNT = HIDDEN_SIZE // GROUP_SIZE
LANES = 16
Q_OUTPUTS = 896
Q_GROUPS = HIDDEN_SIZE // GROUP_SIZE
LAYER = 17


@dataclass(frozen=True)
class QuantizedGroup:
    attention_s8: tuple[int, ...]
    down_s8: tuple[int, ...]
    sum_s8: tuple[int, ...]
    attention_saturation: tuple[bool, ...]
    down_saturation: tuple[bool, ...]
    sum_saturation: tuple[bool, ...]


@dataclass(frozen=True)
class ModelVectors:
    attention_source_s8: tuple[int, ...]
    down_source_s8: tuple[int, ...]
    attention_base_scale32: int
    down_base_scale32: int
    attention_group_scale32: tuple[int, ...]
    down_group_scale32: tuple[int, ...]
    sum_scale32: int
    attention_group_s8: tuple[int, ...]
    down_group_s8: tuple[int, ...]
    sum_s8: tuple[int, ...]
    rms_gain_s16_q8: tuple[int, ...]
    rms_output_s8: tuple[int, ...]
    rms_sumsq: int
    rms_inv_q30: int
    rms_saturation: bool
    q_input_scale32: int
    q_output_scale32: int
    q_weight_s4: tuple[tuple[int, ...], ...]
    q_weight_scale32: tuple[int, ...]
    q_multiplier_s32: tuple[int, ...]
    q_shift_u6: tuple[int, ...]
    q_accumulator_s32: tuple[int, ...]
    q_output_s8: tuple[int, ...]
    q_saturation: tuple[bool, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _saturate_s8(value: int) -> tuple[int, bool]:
    return max(-128, min(127, value)), value < -128 or value > 127


def requantize_scale32_s8(value: int, source_scale32: int, target_scale32: int) -> tuple[int, bool]:
    """Convert one signed-A8 sample between two exact Scale32 domains."""
    if not -128 <= value <= 127:
        raise ValueError("source sample is outside signed A8")
    source_sig, source_exp = unpack_scale32(source_scale32)
    target_sig, target_exp = unpack_scale32(target_scale32)
    common_exp = min(source_exp, target_exp)
    numerator = value * source_sig * (1 << (source_exp - common_exp))
    denominator = target_sig * (1 << (target_exp - common_exp))
    return _saturate_s8(round_divide_even_signed(numerator, denominator))


def quantize_aligned_sum_group(
    attention_source_s8: Iterable[int],
    down_source_s8: Iterable[int],
    attention_base_scale32: int,
    down_base_scale32: int,
    attention_group_scale32: int,
    down_group_scale32: int,
    sum_scale32: int,
) -> QuantizedGroup:
    attention_source = tuple(attention_source_s8)
    down_source = tuple(down_source_s8)
    if len(attention_source) != GROUP_SIZE or len(down_source) != GROUP_SIZE:
        raise ValueError("group-4 source width differs")

    attention_group: list[int] = []
    down_group: list[int] = []
    attention_saturation: list[bool] = []
    down_saturation: list[bool] = []
    for attention, down in zip(attention_source, down_source, strict=True):
        converted, saturated = requantize_scale32_s8(
            attention, attention_base_scale32, attention_group_scale32
        )
        attention_group.append(converted)
        attention_saturation.append(saturated)
        converted, saturated = requantize_scale32_s8(
            down, down_base_scale32, down_group_scale32
        )
        down_group.append(converted)
        down_saturation.append(saturated)

    attention_sig, attention_exp = unpack_scale32(attention_group_scale32)
    down_sig, down_exp = unpack_scale32(down_group_scale32)
    sum_sig, sum_exp = unpack_scale32(sum_scale32)
    common_exp = min(attention_exp, down_exp, sum_exp)
    denominator = sum_sig * (1 << (sum_exp - common_exp))
    sum_group: list[int] = []
    sum_saturation: list[bool] = []
    for attention, down in zip(attention_group, down_group, strict=True):
        numerator = (
            attention * attention_sig * (1 << (attention_exp - common_exp))
            + down * down_sig * (1 << (down_exp - common_exp))
        )
        converted, saturated = _saturate_s8(
            round_divide_even_signed(numerator, denominator)
        )
        sum_group.append(converted)
        sum_saturation.append(saturated)

    return QuantizedGroup(
        attention_s8=tuple(attention_group),
        down_s8=tuple(down_group),
        sum_s8=tuple(sum_group),
        attention_saturation=tuple(attention_saturation),
        down_saturation=tuple(down_saturation),
        sum_saturation=tuple(sum_saturation),
    )


def derive_scale32_multiplier(
    input_scale32: int, weight_scale32: int, output_scale32: int
) -> tuple[int, int]:
    input_num, input_den = scale32_ratio(input_scale32)
    weight_num, weight_den = scale32_ratio(weight_scale32)
    output_num, output_den = scale32_ratio(output_scale32)
    numerator = input_num * weight_num * output_den
    denominator = input_den * weight_den * output_num
    for shift in range(63, -1, -1):
        multiplier = round_divide_even_unsigned(numerator << shift, denominator)
        if multiplier <= (1 << 31) - 1:
            return multiplier, shift
    raise OverflowError("Scale32 projection multiplier is not representable")


def _infer_source_a8(path: Path) -> tuple[np.ndarray, float]:
    source = np.fromfile(path, dtype="<f8")
    nonzero = np.unique(np.abs(source[source != 0.0]))
    if source.size != HIDDEN_SIZE or nonzero.size == 0:
        raise ValueError(f"invalid source artifact: {path}")
    scale = float(nonzero[0])
    quantized = np.rint(source / scale).astype(np.int64)
    if np.any(quantized < -128) or np.any(quantized > 127):
        raise ValueError(f"inferred source A8 escaped range: {path}")
    if not np.array_equal(quantized.astype(np.float64) * scale, source):
        raise ValueError(f"source artifact is not an exact A8/scale product: {path}")
    return quantized, scale


def _group_scales(source_s8: np.ndarray, base_scale32: int) -> tuple[int, ...]:
    base_num, base_den = scale32_ratio(base_scale32)
    records = []
    for start in range(0, HIDDEN_SIZE, GROUP_SIZE):
        maximum = int(np.max(np.abs(source_s8[start : start + GROUP_SIZE])))
        records.append(ceil_scale32_from_ratio(maximum * base_num, 127 * base_den))
    return tuple(records)


def _sum_scale32(
    attention_group_s8: tuple[int, ...],
    down_group_s8: tuple[int, ...],
    attention_scales: tuple[int, ...],
    down_scales: tuple[int, ...],
) -> int:
    maximum = Fraction(0, 1)
    for index, (attention, down) in enumerate(
        zip(attention_group_s8, down_group_s8, strict=True)
    ):
        attention_num, attention_den = scale32_ratio(attention_scales[index // GROUP_SIZE])
        down_num, down_den = scale32_ratio(down_scales[index // GROUP_SIZE])
        value = Fraction(attention * attention_num, attention_den) + Fraction(
            down * down_num, down_den
        )
        maximum = max(maximum, abs(value))
    return ceil_scale32_from_ratio(maximum.numerator, maximum.denominator * 127)


def _q_projection_metadata(
    rms_output: tuple[int, ...], q_input_scale32: int, q_output_scale32: int
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[bool, ...],
]:
    with safe_open(MODEL, framework="pt", device="cpu") as weights, safe_open(
        ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        name = f"model.layers.{LAYER}.self_attn.q_proj"
        base = weights.get_tensor(name + ".weight").to(torch.float32)
        prefix = "base_model.model." + name
        lora_a = adapter.get_tensor(prefix + ".lora_A.weight").to(torch.float32)
        lora_b = adapter.get_tensor(prefix + ".lora_B.weight").to(torch.float32)
        merged = base + (float(LORA_ALPHA) / float(LORA_RANK)) * torch.matmul(
            lora_b, lora_a
        )

    weight = merged.to(torch.float64)
    weight_scale = weight.abs().amax(dim=1) / 7.0
    weight_scale = torch.where(weight_scale > 0, weight_scale, torch.ones_like(weight_scale))
    qweight = torch.round(weight / weight_scale[:, None]).clamp(-8, 7).to(torch.int8)
    activation = torch.tensor(rms_output, dtype=torch.int64)
    accumulators = torch.mv(qweight.to(torch.int64), activation)

    weight_scale32: list[int] = []
    multipliers: list[int] = []
    shifts: list[int] = []
    outputs: list[int] = []
    saturations: list[bool] = []
    for accumulator, scale in zip(
        accumulators.tolist(), weight_scale.tolist(), strict=True
    ):
        scale32 = ceil_scale32_from_float(float(scale))
        multiplier, shift = derive_scale32_multiplier(
            q_input_scale32, scale32, q_output_scale32
        )
        rounded = round_divide_even_signed(int(accumulator) * multiplier, 1 << shift)
        output, saturated = _saturate_s8(rounded)
        weight_scale32.append(scale32)
        multipliers.append(multiplier)
        shifts.append(shift)
        outputs.append(output)
        saturations.append(saturated)

    return (
        tuple(tuple(int(value) for value in row.tolist()) for row in qweight),
        tuple(weight_scale32),
        tuple(multipliers),
        tuple(shifts),
        tuple(int(value) for value in accumulators.tolist()),
        tuple(outputs),
        tuple(saturations),
    )


def derive_model_vectors() -> ModelVectors:
    attention_source, attention_scale = _infer_source_a8(CANDIDATE / "attention_source.bin")
    down_source, down_scale = _infer_source_a8(CANDIDATE / "down_source.bin")
    attention_base_scale32 = ceil_scale32_from_float(attention_scale)
    down_base_scale32 = ceil_scale32_from_float(down_scale)
    attention_scales = _group_scales(attention_source, attention_base_scale32)
    down_scales = _group_scales(down_source, down_base_scale32)

    attention_group: list[int] = []
    down_group: list[int] = []
    for group in range(GROUP_COUNT):
        for source, base_scale, target_scale, output in (
            (
                attention_source,
                attention_base_scale32,
                attention_scales[group],
                attention_group,
            ),
            (down_source, down_base_scale32, down_scales[group], down_group),
        ):
            for value in source[group * GROUP_SIZE : (group + 1) * GROUP_SIZE]:
                output.append(requantize_scale32_s8(int(value), base_scale, target_scale)[0])

    sum_scale32 = _sum_scale32(
        tuple(attention_group), tuple(down_group), attention_scales, down_scales
    )
    sum_values: list[int] = []
    attention_saturations = 0
    down_saturations = 0
    sum_saturations = 0
    for group in range(GROUP_COUNT):
        result = quantize_aligned_sum_group(
            attention_source[group * GROUP_SIZE : (group + 1) * GROUP_SIZE],
            down_source[group * GROUP_SIZE : (group + 1) * GROUP_SIZE],
            attention_base_scale32,
            down_base_scale32,
            attention_scales[group],
            down_scales[group],
            sum_scale32,
        )
        if result.attention_s8 != tuple(
            attention_group[group * GROUP_SIZE : (group + 1) * GROUP_SIZE]
        ) or result.down_s8 != tuple(
            down_group[group * GROUP_SIZE : (group + 1) * GROUP_SIZE]
        ):
            raise AssertionError("group conversion is internally inconsistent")
        sum_values.extend(result.sum_s8)
        attention_saturations += sum(result.attention_saturation)
        down_saturations += sum(result.down_saturation)
        sum_saturations += sum(result.sum_saturation)
    if attention_saturations or down_saturations or sum_saturations:
        raise AssertionError("model source path unexpectedly saturates before RMSNorm")

    candidate_result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
    q_input_scale32 = ceil_scale32_from_float(
        float(np.fromfile(CANDIDATE / "q_input_scales.bin", dtype="<f8")[0])
    )
    q_output_scale32 = ceil_scale32_from_float(
        float(candidate_result["metrics"]["q_output_scale"])
    )
    q_input_num, q_input_den = scale32_ratio(q_input_scale32)
    q_input_scale = q_input_num / q_input_den
    with safe_open(MODEL, framework="pt", device="cpu") as weights:
        gain = (
            weights.get_tensor(f"model.layers.{LAYER}.input_layernorm.weight")
            .to(torch.float32)
            .to(torch.float64)
            .tolist()
        )
    gains = tuple(derive_scaled_gains_q8(gain, q_input_scale))
    rms = reference_rmsnorm(sum_values, list(gains))
    (
        qweight,
        weight_scale32,
        multipliers,
        shifts,
        accumulators,
        q_outputs,
        q_saturations,
    ) = _q_projection_metadata(tuple(rms.outputs), q_input_scale32, q_output_scale32)

    return ModelVectors(
        attention_source_s8=tuple(int(value) for value in attention_source.tolist()),
        down_source_s8=tuple(int(value) for value in down_source.tolist()),
        attention_base_scale32=attention_base_scale32,
        down_base_scale32=down_base_scale32,
        attention_group_scale32=attention_scales,
        down_group_scale32=down_scales,
        sum_scale32=sum_scale32,
        attention_group_s8=tuple(attention_group),
        down_group_s8=tuple(down_group),
        sum_s8=tuple(sum_values),
        rms_gain_s16_q8=gains,
        rms_output_s8=tuple(rms.outputs),
        rms_sumsq=rms.sumsq,
        rms_inv_q30=rms.inv_rms_q30,
        rms_saturation=rms.saturation_seen,
        q_input_scale32=q_input_scale32,
        q_output_scale32=q_output_scale32,
        q_weight_s4=qweight,
        q_weight_scale32=weight_scale32,
        q_multiplier_s32=multipliers,
        q_shift_u6=shifts,
        q_accumulator_s32=accumulators,
        q_output_s8=q_outputs,
        q_saturation=q_saturations,
    )


def _pack(values: Iterable[int], width: int) -> int:
    packed = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        packed |= (int(value) & mask) << (lane * width)
    return packed


def _write_hex(path: Path, values: Iterable[int], width: int) -> None:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    path.write_text(
        "".join(f"{int(value) & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )


def _chunks(values: tuple[int, ...], size: int) -> Iterable[tuple[int, ...]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def write_model_vectors(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    vectors = derive_model_vectors()
    files: dict[str, tuple[Iterable[int], int]] = {
        "attention_source_s8.hex": (
            (_pack(chunk, 8) for chunk in _chunks(vectors.attention_source_s8, GROUP_SIZE)),
            32,
        ),
        "down_source_s8.hex": (
            (_pack(chunk, 8) for chunk in _chunks(vectors.down_source_s8, GROUP_SIZE)),
            32,
        ),
        "attention_group_scale32.hex": (vectors.attention_group_scale32, 32),
        "down_group_scale32.hex": (vectors.down_group_scale32, 32),
        "expected_attention_group_s8.hex": (
            (_pack(chunk, 8) for chunk in _chunks(vectors.attention_group_s8, GROUP_SIZE)),
            32,
        ),
        "expected_down_group_s8.hex": (
            (_pack(chunk, 8) for chunk in _chunks(vectors.down_group_s8, GROUP_SIZE)),
            32,
        ),
        "expected_sum_s8.hex": (
            (_pack(chunk, 8) for chunk in _chunks(vectors.sum_s8, GROUP_SIZE)),
            32,
        ),
        "rms_gain_s16_q8.hex": (
            (_pack(chunk, 16) for chunk in _chunks(vectors.rms_gain_s16_q8, LANES)),
            LANES * 16,
        ),
        "expected_rms_s8.hex": (
            (_pack(chunk, 8) for chunk in _chunks(vectors.rms_output_s8, LANES)),
            LANES * 8,
        ),
        "q_weight_s4.hex": (
            (
                _pack(row[start : start + GROUP_SIZE], 4)
                for row in vectors.q_weight_s4
                for start in range(0, HIDDEN_SIZE, GROUP_SIZE)
            ),
            GROUP_SIZE * 4,
        ),
        "q_multiplier_s32.hex": (vectors.q_multiplier_s32, 32),
        "q_shift_u6.hex": (vectors.q_shift_u6, 8),
        "q_expected_accumulator_s32.hex": (vectors.q_accumulator_s32, 32),
        "q_expected_s8.hex": (vectors.q_output_s8, 8),
        "q_expected_saturation.hex": ((int(value) for value in vectors.q_saturation), 8),
    }
    for name, (values, width) in files.items():
        _write_hex(output / name, values, width)

    constants = output / "ace2_layer16_group4_scale32_constants.svh"
    constants.write_text(
        "\n".join(
            (
                f"localparam integer MODEL_HIDDEN_SIZE = {HIDDEN_SIZE};",
                f"localparam integer MODEL_GROUP_COUNT = {GROUP_COUNT};",
                f"localparam integer MODEL_Q_OUTPUTS = {Q_OUTPUTS};",
                f"localparam logic [31:0] MODEL_ATTENTION_BASE_SCALE32 = 32'h{vectors.attention_base_scale32:08x};",
                f"localparam logic [31:0] MODEL_DOWN_BASE_SCALE32 = 32'h{vectors.down_base_scale32:08x};",
                f"localparam logic [31:0] MODEL_SUM_SCALE32 = 32'h{vectors.sum_scale32:08x};",
                f"localparam logic [31:0] MODEL_Q_INPUT_SCALE32 = 32'h{vectors.q_input_scale32:08x};",
                f"localparam logic [31:0] MODEL_Q_OUTPUT_SCALE32 = 32'h{vectors.q_output_scale32:08x};",
                f"localparam logic [47:0] MODEL_RMS_SUMSQ = 48'd{vectors.rms_sumsq};",
                f"localparam logic [31:0] MODEL_RMS_INV_Q30 = 32'd{vectors.rms_inv_q30};",
                f"localparam integer MODEL_Q_SATURATION_COUNT = {sum(vectors.q_saturation)};",
                "",
            )
        ),
        encoding="ascii",
    )

    candidate_q_input = np.fromfile(CANDIDATE / "q_input_q.bin", dtype="i1").astype(int)
    candidate_q_output = np.fromfile(CANDIDATE / "q_output_q.bin", dtype="i1").astype(int)
    manifest = {
        "schema_version": 1,
        "classification": "candidate_only_focused_rtl_vectors_no_official_attempt",
        "boundary": "layer16_group4_source_quantize_common_scale_sum_layer17_rmsnorm_q_proj",
        "scale_semantics": {
            "encoding": "Scale32 = normalized unsigned Q1.15 significand times 2^signed_exponent",
            "range": {"significand": [0x8000, 0xFFFF], "exponent": [-24, 4]},
            "model_float_to_scale32": "smallest legal Scale32 value greater than or equal to the model-derived positive scale",
            "requantization": "exact common-exponent integer ratio, signed round-to-nearest ties-to-even, signed-A8 saturation",
            "aligned_sum": "source group values are aligned exactly and rounded once into one tensor-wide common Scale32 before RMSNorm",
        },
        "shape": {
            "hidden": HIDDEN_SIZE,
            "source_group": GROUP_SIZE,
            "source_groups": GROUP_COUNT,
            "q_outputs": Q_OUTPUTS,
            "q_mac_lanes": GROUP_SIZE,
        },
        "expected": {
            "source_saturation_count": 0,
            "sum_saturation_count": 0,
            "rms_saturation": vectors.rms_saturation,
            "rms_sumsq": vectors.rms_sumsq,
            "rms_inv_q30": vectors.rms_inv_q30,
            "q_saturation_count": sum(vectors.q_saturation),
            "q_saturation_indices": [
                index for index, value in enumerate(vectors.q_saturation) if value
            ],
            "scale32_vs_frozen_binary32_q_input_mismatches": int(
                np.count_nonzero(np.asarray(vectors.rms_output_s8) != candidate_q_input)
            ),
            "scale32_vs_frozen_binary32_q_output_mismatches": int(
                np.count_nonzero(np.asarray(vectors.q_output_s8) != candidate_q_output)
            ),
        },
        "scales": {
            "attention_base_scale32": f"0x{vectors.attention_base_scale32:08x}",
            "down_base_scale32": f"0x{vectors.down_base_scale32:08x}",
            "sum_scale32": f"0x{vectors.sum_scale32:08x}",
            "q_input_scale32": f"0x{vectors.q_input_scale32:08x}",
            "q_output_scale32": f"0x{vectors.q_output_scale32:08x}",
        },
        "bindings": {
            "candidate_group_result": _artifact(CANDIDATE / "result.json"),
            "attention_source": _artifact(CANDIDATE / "attention_source.bin"),
            "down_source": _artifact(CANDIDATE / "down_source.bin"),
            "candidate_q_input": _artifact(CANDIDATE / "q_input_q.bin"),
            "candidate_q_output": _artifact(CANDIDATE / "q_output_q.bin"),
            "model": {"path": str(MODEL), "sha256": _sha256(MODEL)},
            "adapter": {"path": str(ADAPTER), "sha256": _sha256(ADAPTER)},
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )

    members = sorted(path for path in output.iterdir() if path.is_file())
    sums = output / "SHA256SUMS"
    sums.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in members if path != sums),
        encoding="ascii",
    )
    return manifest


def verify_model_vectors(output: Path) -> dict[str, object]:
    temporary = output.parent / f".{output.name}.check"
    if temporary.exists():
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
    expected = write_model_vectors(temporary)
    try:
        expected_files = sorted(path.name for path in temporary.iterdir())
        observed_files = sorted(path.name for path in output.iterdir())
        if expected_files != observed_files:
            raise RuntimeError("generated vector member list differs")
        for name in expected_files:
            if (temporary / name).read_bytes() != (output / name).read_bytes():
                raise RuntimeError(f"generated vector differs: {name}")
    finally:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
    return expected
