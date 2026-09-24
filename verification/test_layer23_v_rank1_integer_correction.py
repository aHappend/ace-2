from __future__ import annotations

import unittest

from tools.ace2_layer23_v_rank1_integer_correction_reference import (
    INPUT_WIDTH,
    OUTPUT_WIDTH,
    POSITION_COUNT,
    apply_rank1_correction,
    load_frozen_config,
    load_frozen_position_vectors,
    round_shift_even_signed,
    saturating_add_s8,
    synthetic_cases,
    synthetic_config,
    verify_generalization_vectors,
    verify_frozen_vectors,
)


class Layer23VRank1IntegerCorrectionTest(unittest.TestCase):
    def test_rounding_and_saturating_add(self) -> None:
        self.assertEqual(round_shift_even_signed(5, 1), 2)
        self.assertEqual(round_shift_even_signed(7, 1), 4)
        self.assertEqual(round_shift_even_signed(-5, 1), -2)
        self.assertEqual(round_shift_even_signed(-7, 1), -4)
        self.assertEqual(saturating_add_s8(120, 20), (127, True))
        self.assertEqual(saturating_add_s8(-120, -20), (-128, True))
        self.assertEqual(saturating_add_s8(40, -20), (20, False))

    def test_synthetic_cases_cover_nominal_tie_and_saturation(self) -> None:
        names = {case["name"] for case in synthetic_cases()}
        self.assertIn("nominal_positive", names)
        self.assertIn("negative_tie_to_even", names)
        self.assertIn("positive_saturation", names)
        self.assertIn("negative_saturation", names)

        negative = next(case for case in synthetic_cases() if case["name"] == "negative_tie_to_even")
        self.assertEqual(negative["expected"]["rank_intermediate_s8"], -3)
        self.assertEqual(negative["expected"]["correction0_s8"], -2)
        self.assertFalse(negative["expected"]["correction_saturation0"])

        positive_sat = next(case for case in synthetic_cases() if case["name"] == "positive_saturation")
        self.assertTrue(positive_sat["expected"]["rank_saturation"])
        self.assertTrue(positive_sat["expected"]["correction_saturation0"])
        self.assertTrue(positive_sat["expected"]["add_saturation0"])

    def test_frozen_payload_geometry(self) -> None:
        config = load_frozen_config()
        vectors = load_frozen_position_vectors()
        self.assertEqual(len(config.input_to_rank_s8), INPUT_WIDTH)
        self.assertEqual(len(config.rank_to_channel_s8), OUTPUT_WIDTH)
        self.assertEqual(len(config.second_stage_multiplier_s32), OUTPUT_WIDTH)
        self.assertEqual(len(config.second_stage_right_shift_u6), OUTPUT_WIDTH)
        self.assertEqual(len(vectors), POSITION_COUNT)
        self.assertEqual(len(vectors[0].input_s8), INPUT_WIDTH)
        self.assertEqual(len(vectors[0].baseline_v_s8), OUTPUT_WIDTH)

    def test_frozen_all_34_positions_match_byte_for_byte(self) -> None:
        summary = verify_frozen_vectors()
        self.assertEqual(summary["positions_verified"], POSITION_COUNT)
        self.assertTrue(summary["intermediate_rank_scalar_byte_equal"])
        self.assertTrue(summary["correction_output_byte_equal"])
        self.assertTrue(summary["corrected_v_output_byte_equal"])
        self.assertEqual(
            summary["rank_intermediate_s8_sha256"],
            "655234ba95521d2c5811b8472d5105d6f831475bd7c4a35401bb037f1d4feae6",
        )
        self.assertEqual(
            summary["correction_s8_sha256"],
            "9483aea03d458933121f22afc02eea8643548715b2513652f6807d62fb29c6c3",
        )
        self.assertEqual(
            summary["corrected_v_output_s8_sha256"],
            "69a7b299e7b69f19238bbb04eeef86a12eb0ba3858c8d0ab9f0e47fa7e5116c3",
        )

    def test_frozen_multi_prompt_generalization_records_match_byte_for_byte(self) -> None:
        summary = verify_generalization_vectors()
        self.assertEqual(summary["records_verified"], 8)
        self.assertEqual(summary["prompt_count"], 4)
        self.assertEqual(summary["positions_verified"], 290)
        self.assertEqual(
            summary["input_payload_s8_sha256"],
            "a4b01d98e6c426e2021afa85d31dfcda99669dfea4871d7842d2cf3c609233dc",
        )
        self.assertEqual(
            summary["correction_s8_sha256"],
            "6fde4861bb3562681733b347b533c77a7a7181bde1206376ebfa892a4adbd629",
        )
        self.assertEqual(
            summary["corrected_v_output_s8_sha256"],
            "ba4f5c732cc7ceb1913f8f62f2e890b7ddad9755c37e3e68b0def6bcf7341b6a",
        )

    def test_standalone_custom_config_uses_same_reference_path(self) -> None:
        config = synthetic_config(
            input_to_rank_0=4,
            first_multiplier=3,
            first_shift=1,
            rank_to_channel_0=-5,
            second_multiplier_0=1,
            second_shift_0=1,
        )
        input_payload = (5,) + (0,) * (INPUT_WIDTH - 1)
        baseline = (-10,) + (0,) * (OUTPUT_WIDTH - 1)
        result = apply_rank1_correction(input_payload, baseline, config)
        self.assertEqual(result.rank_accumulator_s32, 20)
        self.assertEqual(result.rank_rounded_s32, 30)
        self.assertEqual(result.rank_intermediate_s8, 30)
        self.assertEqual(result.correction_accumulator_s32[0], -150)
        self.assertEqual(result.correction_rounded_s32[0], -75)
        self.assertEqual(result.correction_s8[0], -75)
        self.assertEqual(result.corrected_v_s8[0], -85)


if __name__ == "__main__":
    unittest.main()
