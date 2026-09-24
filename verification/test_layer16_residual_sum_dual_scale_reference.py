from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "evidence/candidates/w4a8-layer16-residual-sum-dual-scale-repair-v1/"
    "candidate-0006"
)
HIDDEN = 896
DOWN_GROUP_SIZE = 4


def unpack_scale32(record: int) -> tuple[int, int]:
    if (record >> 24) != 0:
        raise ValueError("reserved Scale32 bits are nonzero")
    significand = record & 0xFFFF
    exponent = (record >> 16) & 0xFF
    if exponent >= 128:
        exponent -= 256
    if significand < 0x8000 or not (-24 <= exponent <= 4):
        raise ValueError("Scale32 record is outside the bounded RTL domain")
    return significand, exponent


def round_divide_even_signed(numerator: int, denominator: int) -> int:
    negative = numerator < 0
    magnitude = abs(numerator)
    quotient, remainder = divmod(magnitude, denominator)
    if remainder * 2 > denominator or (
        remainder * 2 == denominator and quotient & 1
    ):
        quotient += 1
    return -quotient if negative else quotient


class Layer16ResidualSumDualScaleReferenceTest(unittest.TestCase):
    def test_candidate_control_and_best_null_are_frozen(self) -> None:
        result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "PARTIAL_LAYER16_RESIDUAL_SUM_DUAL_SCALE_NO_IMPROVEMENT")
        self.assertEqual(result["ordered_output_group_sizes"], [896, 32, 16, 8, 4, 1])
        self.assertEqual(result["control"]["reference_rank"], 31879)
        self.assertEqual(result["control"]["top_token_id"], 37865)
        self.assertEqual(result["best_non_control_output_group_size"], 32)
        self.assertEqual(result["best_non_control_reference_rank"], 34409)
        self.assertEqual(result["rtl_validation_output_group_size"], 32)
        self.assertFalse(result["first_token_restored"])

    def test_model_vectors_match_one_add_one_round_reference(self) -> None:
        group = CANDIDATE / "contracts/group-032"
        residual = np.fromfile(group / "residual_group_s8.bin", dtype="i1")
        residual_scale = np.fromfile(group / "residual_group_scale32.bin", dtype="<i8")
        down = np.fromfile(group / "down_output_s8.bin", dtype="i1")
        down_group_scale = np.fromfile(group / "down_output_group_scale32.bin", dtype="<i8")
        output_group_scale = np.fromfile(group / "sum_output_group_scale32.bin", dtype="<i8")
        expected = np.fromfile(group / "sum_s8.bin", dtype="i1")
        saturation = np.fromfile(group / "sum_saturation.bin", dtype="u1")

        self.assertEqual(residual.size, HIDDEN)
        self.assertEqual(residual_scale.size, HIDDEN)
        self.assertEqual(down.size, HIDDEN)
        self.assertEqual(down_group_scale.size, HIDDEN // DOWN_GROUP_SIZE)
        self.assertEqual(output_group_scale.size, HIDDEN // 32)
        self.assertEqual(expected.size, HIDDEN)
        self.assertEqual(saturation.size, HIDDEN)

        observed = []
        observed_saturation = []
        for index in range(HIDDEN):
            residual_sig, residual_exp = unpack_scale32(int(residual_scale[index]))
            down_sig, down_exp = unpack_scale32(
                int(down_group_scale[index // DOWN_GROUP_SIZE])
            )
            output_sig, output_exp = unpack_scale32(
                int(output_group_scale[index // 32])
            )
            common_exp = min(residual_exp, down_exp, output_exp)
            residual_term = int(residual[index]) * residual_sig << (residual_exp - common_exp)
            down_term = int(down[index]) * down_sig << (down_exp - common_exp)
            numerator = residual_term + down_term
            denominator = output_sig << (output_exp - common_exp)
            rounded = round_divide_even_signed(numerator, denominator)
            clipped = max(-128, min(127, rounded))
            observed.append(clipped)
            observed_saturation.append(int(clipped != rounded))

        np.testing.assert_array_equal(np.asarray(observed, dtype=np.int8), expected)
        np.testing.assert_array_equal(
            np.asarray(observed_saturation, dtype=np.uint8), saturation
        )

    def test_directed_ties_and_saturation(self) -> None:
        unit = 0x00008000
        half = 0x00FF8000

        def evaluate(residual: int, residual_scale: int, down: int, down_scale: int) -> int:
            residual_sig, residual_exp = unpack_scale32(residual_scale)
            down_sig, down_exp = unpack_scale32(down_scale)
            output_sig, output_exp = unpack_scale32(unit)
            common_exp = min(residual_exp, down_exp, output_exp)
            numerator = (
                (residual * residual_sig << (residual_exp - common_exp))
                + (down * down_sig << (down_exp - common_exp))
            )
            denominator = output_sig << (output_exp - common_exp)
            return round_divide_even_signed(numerator, denominator)

        self.assertEqual(evaluate(1, half, 0, unit), 0)
        self.assertEqual(evaluate(3, half, 0, unit), 2)
        self.assertEqual(evaluate(-1, half, 0, unit), 0)
        self.assertEqual(evaluate(-3, half, 0, unit), -2)
        self.assertEqual(evaluate(1, half, 1, half), 1)
        self.assertGreater(evaluate(127, 0x00018000, 0, unit), 127)
        self.assertLess(evaluate(-128, 0x00018000, 0, unit), -128)


if __name__ == "__main__":
    unittest.main()
