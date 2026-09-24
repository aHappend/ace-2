#!/usr/bin/env python3
"""Independent integer oracle for layer-23 V input-scale conversion."""

from __future__ import annotations

import math
from typing import Sequence


INT32_MAX = (1 << 31) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def _round_fraction_even(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("oracle fraction must be finite and nonnegative")
    quotient, remainder = divmod(numerator, denominator)
    twice = remainder * 2
    if twice > denominator or (twice == denominator and (quotient & 1)):
        quotient += 1
    return quotient


def derive_multiplier(real_multiplier: float) -> tuple[int, int]:
    if not math.isfinite(real_multiplier) or real_multiplier < 0.0:
        raise ValueError("oracle multiplier input must be finite and nonnegative")
    numerator, denominator = real_multiplier.as_integer_ratio()
    for shift in range(63, -1, -1):
        multiplier = _round_fraction_even(numerator << shift, denominator)
        if multiplier <= INT32_MAX:
            return multiplier, shift
    raise OverflowError("oracle multiplier is not representable")


def round_shift_even(value: int, shift: int) -> int:
    if not 0 <= shift <= 63:
        raise ValueError("oracle shift is not u6")
    if shift == 0:
        return value
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    return -quotient if value < 0 else quotient


def requantize_s8(
    values: Sequence[int],
    source_scale: float,
    target_scale: float,
) -> dict[str, object]:
    if (
        not math.isfinite(source_scale)
        or not math.isfinite(target_scale)
        or source_scale <= 0.0
        or target_scale <= 0.0
    ):
        raise ValueError("oracle scales must be finite and positive")
    source = [int(value) for value in values]
    if any(value < -128 or value > 127 for value in source):
        raise ValueError("oracle source escaped signed int8")
    real_multiplier = source_scale / target_scale
    multiplier, shift = derive_multiplier(real_multiplier)
    converted: list[int] = []
    saturation: list[bool] = []
    for value in source:
        product = value * multiplier
        if not INT64_MIN <= product <= INT64_MAX:
            raise OverflowError("oracle product escaped signed int64")
        rounded = round_shift_even(product, shift)
        saturated = rounded < -128 or rounded > 127
        converted.append(max(-128, min(127, rounded)))
        saturation.append(saturated)
    return {
        "real_multiplier_f64": real_multiplier,
        "multiplier_s32": multiplier,
        "right_shift_u6": shift,
        "converted_s8": converted,
        "saturation": saturation,
    }
