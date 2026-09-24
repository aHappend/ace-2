from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ace2_layer16_group4_scale32_reference import (
    derive_scale32_multiplier,
    quantize_aligned_sum_group,
    requantize_scale32_s8,
)
from ace2_quality_contracts import pack_scale32, scale32_ratio


class Layer16Group4Scale32ReferenceTest(unittest.TestCase):
    def test_signed_ties_to_even(self) -> None:
        unit = pack_scale32(0x8000, 0)
        two = pack_scale32(0x8000, 1)
        self.assertEqual(requantize_scale32_s8(5, unit, two), (2, False))
        self.assertEqual(requantize_scale32_s8(7, unit, two), (4, False))
        self.assertEqual(requantize_scale32_s8(-5, unit, two), (-2, False))
        self.assertEqual(requantize_scale32_s8(-7, unit, two), (-4, False))

    def test_sum_rounding_and_asymmetric_saturation(self) -> None:
        unit = pack_scale32(0x8000, 0)
        two = pack_scale32(0x8000, 1)
        rounded = quantize_aligned_sum_group(
            (5, 7, -5, -7),
            (0, 0, 0, 0),
            unit,
            unit,
            unit,
            unit,
            two,
        )
        self.assertEqual(rounded.sum_s8, (2, 4, -2, -4))
        saturated = quantize_aligned_sum_group(
            (127, -128, 127, -128),
            (127, -128, 127, -128),
            two,
            two,
            two,
            two,
            unit,
        )
        self.assertEqual(saturated.sum_s8, (127, -128, 127, -128))
        self.assertEqual(saturated.sum_saturation, (True, True, True, True))

    def test_exact_multiplier_approximates_scale32_ratio(self) -> None:
        rng = random.Random(0xACE21604)
        for _ in range(1000):
            records = [
                pack_scale32(rng.randint(0x8000, 0xFFFF), rng.randint(-12, 2))
                for _ in range(3)
            ]
            multiplier, shift = derive_scale32_multiplier(*records)
            input_num, input_den = scale32_ratio(records[0])
            weight_num, weight_den = scale32_ratio(records[1])
            output_num, output_den = scale32_ratio(records[2])
            exact = input_num * weight_num * output_den / (
                input_den * weight_den * output_num
            )
            represented = multiplier / (1 << shift)
            self.assertLessEqual(abs(exact - represented), 0.5 / (1 << shift))
            self.assertLessEqual(multiplier, (1 << 31) - 1)


if __name__ == "__main__":
    unittest.main()
