from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "evidence/candidates/w4a8-layer16-post-attention-residual-sum-scale-alignment-repair-v1/"
    "candidate-0002"
)
HIDDEN = 896


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


class Layer16PostAttentionResidualSumScaleAlignmentReferenceTest(unittest.TestCase):
    def test_candidate_control_and_selection_are_frozen(self) -> None:
        result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["ordered_output_group_sizes"], [896, 32, 16, 8, 4, 1])
        self.assertEqual(result["control"]["reference_rank"], 31879)
        self.assertEqual(result["control"]["top_token_id"], 37865)
        validation = result["rtl_validation_output_group_size"]
        self.assertIn(validation, {item["output_group_size"] for item in result["trace"]})
        if result["selected_output_group_size"] is not None:
            self.assertEqual(validation, result["selected_output_group_size"])
            self.assertTrue(
                result["selected_reference_rank"] < 31879
                or result["selected_top_token_id"] == 9707
            )
        else:
            self.assertEqual(validation, result["best_non_control_output_group_size"])

    def test_model_vectors_match_one_add_one_round_reference(self) -> None:
        result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
        group_size = int(result["rtl_validation_output_group_size"])
        group = CANDIDATE / f"contracts/group-{group_size:03d}"
        residual = np.fromfile(group / "residual_s8.bin", dtype="i1")
        residual_scale = np.fromfile(
            group / "residual_scale32_expanded.bin", dtype="<i8"
        )
        attention = np.fromfile(group / "attention_s8.bin", dtype="i1")
        attention_scale = np.fromfile(
            group / "attention_scale32_expanded.bin", dtype="<i8"
        )
        output_group_scale = np.fromfile(
            group / "post_attention_output_group_scale32.bin", dtype="<i8"
        )
        expected = np.fromfile(group / "post_attention_sum_s8.bin", dtype="i1")
        saturation = np.fromfile(
            group / "post_attention_sum_saturation.bin", dtype="u1"
        )

        self.assertEqual(residual.size, HIDDEN)
        self.assertEqual(residual_scale.size, HIDDEN)
        self.assertEqual(attention.size, HIDDEN)
        self.assertEqual(attention_scale.size, HIDDEN)
        self.assertEqual(output_group_scale.size, HIDDEN // group_size)
        self.assertEqual(expected.size, HIDDEN)
        self.assertEqual(saturation.size, HIDDEN)

        observed = []
        observed_saturation = []
        for index in range(HIDDEN):
            residual_sig, residual_exp = unpack_scale32(int(residual_scale[index]))
            attention_sig, attention_exp = unpack_scale32(
                int(attention_scale[index])
            )
            output_sig, output_exp = unpack_scale32(
                int(output_group_scale[index // group_size])
            )
            common_exp = min(residual_exp, attention_exp, output_exp)
            residual_term = (
                int(residual[index]) * residual_sig << (residual_exp - common_exp)
            )
            attention_term = (
                int(attention[index])
                * attention_sig
                << (attention_exp - common_exp)
            )
            rounded = round_divide_even_signed(
                residual_term + attention_term,
                output_sig << (output_exp - common_exp),
            )
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

        def evaluate(residual: int, residual_scale: int, attention: int, attention_scale: int) -> int:
            residual_sig, residual_exp = unpack_scale32(residual_scale)
            attention_sig, attention_exp = unpack_scale32(attention_scale)
            output_sig, output_exp = unpack_scale32(unit)
            common_exp = min(residual_exp, attention_exp, output_exp)
            numerator = (
                (residual * residual_sig << (residual_exp - common_exp))
                + (attention * attention_sig << (attention_exp - common_exp))
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
