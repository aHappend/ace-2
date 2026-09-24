#!/usr/bin/env python3
"""Independent checks for the selected group-1 exact Scale32 payload."""

from __future__ import annotations

import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ace2_layer16_group4_scale32_reference import (
    derive_scale32_multiplier,
    requantize_scale32_s8,
)
from ace2_quality_contracts import (
    ceil_scale32_from_ratio,
    round_divide_even_signed,
    scale32_ratio,
    unpack_scale32,
)
from ace2_rmsnorm_reference import reference_rmsnorm

CANDIDATE = (
    ROOT
    / "evidence/candidates/w4a8-layer16-source-grouped-activation-repair-v1/"
    "candidate-0002"
)
GROUP = CANDIDATE / "groups/group-001"
HIDDEN = 896
Q_OUTPUTS = 896


def read(name: str, dtype: str, count: int) -> np.ndarray:
    value = np.fromfile(GROUP / f"{name}.bin", dtype=dtype)
    if value.size != count:
        raise AssertionError(f"{name} count differs: {value.size} != {count}")
    return value


def scalar(name: str) -> int:
    return int(read(name, "<i8", 1)[0])


class Group1Scale32ReferenceTest(unittest.TestCase):
    def test_selected_candidate_is_an_honest_improvement(self) -> None:
        result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["selected_source_group_size"], 1)
        self.assertEqual(result["selected_reference_rank"], 40807)
        self.assertLess(
            result["selected_reference_rank"], result["baseline_reference_rank"]
        )
        self.assertFalse(result["first_token_restored"])

    def test_source_groups_and_common_sum_are_exact(self) -> None:
        attention_source = read("attention_source_s8", "i1", HIDDEN).astype(np.int64)
        down_source = read("down_source_s8", "i1", HIDDEN).astype(np.int64)
        attention_base = scalar("attention_base_scale32")
        down_base = scalar("down_base_scale32")
        attention_scales = read(
            "attention_group_scale32", "<i8", HIDDEN
        ).astype(np.int64)
        down_scales = read("down_group_scale32", "<i8", HIDDEN).astype(np.int64)
        attention_group = read("attention_group_s8", "i1", HIDDEN).astype(np.int64)
        down_group = read("down_group_s8", "i1", HIDDEN).astype(np.int64)
        expected_sum = read("sum_s8", "i1", HIDDEN).astype(np.int64)
        sum_scale = scalar("sum_scale32")

        attention_num, attention_den = scale32_ratio(attention_base)
        down_num, down_den = scale32_ratio(down_base)
        maximum = Fraction(0, 1)
        for index in range(HIDDEN):
            self.assertEqual(
                int(attention_scales[index]),
                ceil_scale32_from_ratio(
                    abs(int(attention_source[index])) * attention_num,
                    127 * attention_den,
                ),
            )
            self.assertEqual(
                int(down_scales[index]),
                ceil_scale32_from_ratio(
                    abs(int(down_source[index])) * down_num,
                    127 * down_den,
                ),
            )
            self.assertEqual(
                int(attention_group[index]),
                requantize_scale32_s8(
                    int(attention_source[index]),
                    attention_base,
                    int(attention_scales[index]),
                )[0],
            )
            self.assertEqual(
                int(down_group[index]),
                requantize_scale32_s8(
                    int(down_source[index]), down_base, int(down_scales[index])
                )[0],
            )
            a_num, a_den = scale32_ratio(int(attention_scales[index]))
            d_num, d_den = scale32_ratio(int(down_scales[index]))
            value = Fraction(int(attention_group[index]) * a_num, a_den) + Fraction(
                int(down_group[index]) * d_num, d_den
            )
            maximum = max(maximum, abs(value))

        self.assertEqual(
            sum_scale,
            ceil_scale32_from_ratio(maximum.numerator, maximum.denominator * 127),
        )
        sum_sig, sum_exp = unpack_scale32(sum_scale)
        for index in range(HIDDEN):
            attention_sig, attention_exp = unpack_scale32(
                int(attention_scales[index])
            )
            down_sig, down_exp = unpack_scale32(int(down_scales[index]))
            common_exp = min(attention_exp, down_exp, sum_exp)
            numerator = (
                int(attention_group[index])
                * attention_sig
                * (1 << (attention_exp - common_exp))
                + int(down_group[index])
                * down_sig
                * (1 << (down_exp - common_exp))
            )
            denominator = sum_sig * (1 << (sum_exp - common_exp))
            rounded = round_divide_even_signed(numerator, denominator)
            self.assertEqual(int(expected_sum[index]), max(-128, min(127, rounded)))

        self.assertEqual(int(read("attention_saturation", "u1", HIDDEN).sum()), 0)
        self.assertEqual(int(read("down_saturation", "u1", HIDDEN).sum()), 0)
        self.assertEqual(int(read("sum_saturation", "u1", HIDDEN).sum()), 0)

    def test_rmsnorm_and_q_projection_are_exact(self) -> None:
        sum_s8 = read("sum_s8", "i1", HIDDEN).astype(np.int64).tolist()
        gains = read("rms_gain_s16_q8", "<i2", HIDDEN).astype(np.int64).tolist()
        rms = reference_rmsnorm(sum_s8, gains)
        np.testing.assert_array_equal(
            np.asarray(rms.outputs, dtype=np.int8),
            read("rms_output_s8", "i1", HIDDEN),
        )
        self.assertEqual(rms.sumsq, scalar("rms_sumsq"))
        self.assertEqual(rms.inv_rms_q30, scalar("rms_inv_q30"))
        self.assertEqual(rms.saturation_seen, bool(read("rms_saturation", "u1", 1)[0]))

        activation = np.asarray(rms.outputs, dtype=np.int64)
        qweight = read("q_weight_s4", "i1", Q_OUTPUTS * HIDDEN).reshape(
            Q_OUTPUTS, HIDDEN
        ).astype(np.int64)
        accumulator = qweight @ activation
        np.testing.assert_array_equal(
            accumulator, read("q_accumulator_s32", "<i8", Q_OUTPUTS)
        )
        input_scale = scalar("q_input_scale32")
        output_scale = scalar("q_output_scale32")
        weight_scales = read("q_weight_scale32", "<i8", Q_OUTPUTS)
        expected_multiplier = read("q_multiplier_s32", "<i8", Q_OUTPUTS)
        expected_shift = read("q_shift_u6", "<i8", Q_OUTPUTS)
        expected_output = read("q_output_s8", "i1", Q_OUTPUTS)
        expected_saturation = read("q_saturation", "u1", Q_OUTPUTS)
        for row in range(Q_OUTPUTS):
            multiplier, shift = derive_scale32_multiplier(
                input_scale, int(weight_scales[row]), output_scale
            )
            self.assertEqual(multiplier, int(expected_multiplier[row]))
            self.assertEqual(shift, int(expected_shift[row]))
            rounded = round_divide_even_signed(
                int(accumulator[row]) * multiplier, 1 << shift
            )
            self.assertEqual(int(expected_output[row]), max(-128, min(127, rounded)))
            self.assertEqual(
                int(expected_saturation[row]), int(rounded < -128 or rounded > 127)
            )


if __name__ == "__main__":
    unittest.main()
