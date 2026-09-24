#!/usr/bin/env python3
"""Bounded fixed-point reference for layer-16 post-attention RMSNorm output."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

try:
    from tools.ace2_quality_contracts import (
        ceil_scale32_from_float,
        round_divide_even_signed,
        scale32_ratio,
        unpack_scale32,
    )
    from tools.ace2_rmsnorm_reference import (
        derive_rmsnorm_output_scale,
        derive_scaled_gains_q8,
    )
except ModuleNotFoundError:
    from ace2_quality_contracts import (  # type: ignore[no-redef]
        ceil_scale32_from_float,
        round_divide_even_signed,
        scale32_ratio,
        unpack_scale32,
    )
    from ace2_rmsnorm_reference import (  # type: ignore[no-redef]
        derive_rmsnorm_output_scale,
        derive_scaled_gains_q8,
    )


HIDDEN_SIZE = 896
GAIN_FRAC = 8
INV_RMS_FRAC = 30
OUTPUT_SHIFT = GAIN_FRAC + INV_RMS_FRAC
MAX_REBASED_MAGNITUDE = 2047


@dataclass(frozen=True)
class GroupedRmsnormOutputResult:
    aligned: list[int]
    gains_q8: list[int]
    output_scale32: list[int]
    outputs: list[int]
    saturation: list[int]
    sumsq: int
    inv_rms_q30: int
    common_exponent: int
    rebase_shift: int


def _saturate_int8(value: int) -> tuple[int, int]:
    clipped = max(-128, min(127, value))
    return clipped, int(clipped != value)


def _rebased_values(
    activations: list[int], input_scale32: list[int]
) -> tuple[list[int], int, int]:
    if len(activations) != HIDDEN_SIZE or len(input_scale32) != HIDDEN_SIZE:
        raise ValueError(f"expected {HIDDEN_SIZE} grouped RMSNorm inputs")
    unpacked = [unpack_scale32(int(record)) for record in input_scale32]
    common_exponent = min(exponent for _significand, exponent in unpacked)
    exact = [
        int(value) * significand * (1 << (exponent - common_exponent))
        for value, (significand, exponent) in zip(
            activations, unpacked, strict=True
        )
    ]
    maximum = max(abs(value) for value in exact)
    shift = max(0, maximum.bit_length() - MAX_REBASED_MAGNITUDE.bit_length())
    while True:
        denominator = 1 << shift
        aligned = [round_divide_even_signed(value, denominator) for value in exact]
        if max(abs(value) for value in aligned) <= MAX_REBASED_MAGNITUDE:
            return aligned, common_exponent, shift
        shift += 1


def fixed_real_rmsnorm(
    activations: Iterable[int],
    input_scale32: Iterable[int],
    weights: Iterable[float],
) -> list[float]:
    values = []
    for activation, record in zip(activations, input_scale32, strict=True):
        numerator, denominator = scale32_ratio(int(record))
        values.append(int(activation) * numerator / denominator)
    gain_values = [float(value) for value in weights]
    if len(values) != HIDDEN_SIZE or len(gain_values) != HIDDEN_SIZE:
        raise ValueError(f"expected {HIDDEN_SIZE} RMSNorm values and gains")
    mean_square = sum(value * value for value in values) / HIDDEN_SIZE
    rms = math.sqrt(mean_square)
    if not math.isfinite(rms) or rms <= 0.0:
        raise ValueError("fixed grouped RMSNorm input has no finite nonzero RMS")
    return [
        value * gain / rms
        for value, gain in zip(values, gain_values, strict=True)
    ]


def derive_grouped_output_metadata(
    calibrated_output: list[float], weights: list[float], group_size: int
) -> tuple[list[int], list[int]]:
    if HIDDEN_SIZE % group_size:
        raise ValueError("RMSNorm output group must divide 896 channels")
    if len(calibrated_output) != HIDDEN_SIZE or len(weights) != HIDDEN_SIZE:
        raise ValueError(f"expected {HIDDEN_SIZE} calibrated outputs and gains")
    scales: list[int] = []
    gains: list[int] = []
    for start in range(0, HIDDEN_SIZE, group_size):
        stop = start + group_size
        group_weights = weights[start:stop]
        scale = derive_rmsnorm_output_scale(
            group_weights,
            max(max(abs(value) for value in calibrated_output[start:stop]), 1.0e-12),
        )
        scale32 = ceil_scale32_from_float(math.nextafter(scale, math.inf))
        numerator, denominator = scale32_ratio(scale32)
        try:
            group_gains = derive_scaled_gains_q8(
                group_weights, numerator / denominator
            )
        except OverflowError:
            scale32 = ceil_scale32_from_float(scale * (1.0 + 2.0**-20))
            numerator, denominator = scale32_ratio(scale32)
            group_gains = derive_scaled_gains_q8(
                group_weights, numerator / denominator
            )
        scales.append(scale32)
        gains.extend(group_gains)
    return scales, gains


def reference_grouped_rmsnorm_output(
    activations: Iterable[int],
    input_scale32: Iterable[int],
    weights: Iterable[float],
    output_group_size: int,
) -> GroupedRmsnormOutputResult:
    activation_values = [int(value) for value in activations]
    scale_values = [int(value) for value in input_scale32]
    weight_values = [float(value) for value in weights]
    calibrated = fixed_real_rmsnorm(
        activation_values, scale_values, weight_values
    )
    output_scales, gains = derive_grouped_output_metadata(
        calibrated, weight_values, output_group_size
    )
    aligned, common_exponent, rebase_shift = _rebased_values(
        activation_values, scale_values
    )
    sumsq = sum(value * value for value in aligned)
    mean_square = (sumsq + HIDDEN_SIZE // 2) // HIDDEN_SIZE
    rms_ceil = max(math.isqrt(mean_square), 1)
    if rms_ceil * rms_ceil < mean_square:
        rms_ceil += 1
    inv_rms_q30 = (1 << INV_RMS_FRAC) // rms_ceil
    if inv_rms_q30 == 0:
        raise OverflowError("bounded RMSNorm reciprocal underflowed")

    outputs: list[int] = []
    saturation: list[int] = []
    for value, gain in zip(aligned, gains, strict=True):
        rounded = round_divide_even_signed(
            value * int(gain) * inv_rms_q30, 1 << OUTPUT_SHIFT
        )
        clipped, saturated = _saturate_int8(rounded)
        outputs.append(clipped)
        saturation.append(saturated)
    return GroupedRmsnormOutputResult(
        aligned=aligned,
        gains_q8=gains,
        output_scale32=output_scales,
        outputs=outputs,
        saturation=saturation,
        sumsq=sumsq,
        inv_rms_q30=inv_rms_q30,
        common_exponent=common_exponent,
        rebase_shift=rebase_shift,
    )
