from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ace2_full_model_fixed_point import (  # noqa: E402
    CalibrationRange,
    FixedRMSNorm,
    W4A8Linear,
    quantize_int8,
    rmsnorm_output_scale,
    round_shift_even,
)
from ace2_rmsnorm_reference import (  # noqa: E402
    derive_scaled_gains_q8,
    reference_rmsnorm,
)
from localize_layer0_paired_divergence import (  # noqa: E402
    BOUNDARY_ORDER,
    exact_difference,
)


class RmsNormNumericalBisectTest(unittest.TestCase):
    def test_signed_round_shift_uses_ties_to_even(self) -> None:
        values = torch.tensor(
            [-15, -14, -13, -11, -10, -9, -7, -5, -3, -1,
              1, 3, 5, 7, 9, 10, 11, 13, 14, 15],
            dtype=torch.int64,
        )

        def scalar_oracle(value: int, shift: int) -> int:
            magnitude = abs(value)
            base, remainder = divmod(magnitude, 1 << shift)
            half = 1 << (shift - 1)
            if remainder > half or (remainder == half and base % 2 == 1):
                base += 1
            return -base if value < 0 else base

        for shift in (1, 2, 3):
            expected = torch.tensor(
                [scalar_oracle(int(value), shift) for value in values],
                dtype=torch.int64,
            )
            self.assertTrue(torch.equal(expected, round_shift_even(values, shift)))

    def test_activation_quantization_rounds_ties_even_then_saturates(self) -> None:
        raw_values = torch.tensor(
            [
                -129.5,
                -128.5,
                -127.5,
                -126.5,
                -2.5,
                -1.5,
                -0.5,
                0.5,
                1.5,
                2.5,
                126.5,
                127.5,
                128.5,
            ],
            dtype=torch.float64,
        )
        scale = 0.5
        expected = torch.tensor(
            [-128, -128, -128, -126, -2, -2, 0, 0, 2, 2, 126, 127, 127],
            dtype=torch.int8,
        )
        actual = quantize_int8(raw_values * scale, scale)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(5, int(((actual == -128) | (actual == 127)).sum()))

    def test_projection_requantization_rounds_and_clamps_adversarial_values(self) -> None:
        source = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            source.weight.fill_(1.0)
        projection = W4A8Linear(
            source,
            CalibrationRange(input_absmax=127.0, output_absmax=127.0),
        )
        projection.multiplier.fill_(1)
        projection.right_shift.fill_(1)
        accumulator = torch.tensor(
            [-300, -257, -255, -7, -5, -3, -1, 1, 3, 5, 7, 255, 257, 300],
            dtype=torch.int64,
        ).reshape(-1, 1)
        expected = torch.tensor(
            [-128, -128, -128, -4, -2, -2, 0, 0, 2, 2, 4, 127, 127, 127],
            dtype=torch.int8,
        )
        actual = projection.requantize_accumulator(
            accumulator,
            (accumulator.shape[0],),
        ).reshape(-1)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(6, int(((actual == -128) | (actual == 127)).sum()))

    def test_frozen_ramp_vector_distinguishes_repaired_static_scale(self) -> None:
        activations = [((index * 17 + 11) % 256) - 128 for index in range(896)]
        weights = [0.875 + (index % 17) / 64.0 for index in range(896)]
        output_scale = 5.0 / 128.0
        input_scale = 1.0 / 8.0

        source = nn.Module()
        source.register_parameter(
            "weight",
            nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False),
        )
        self.assertEqual(
            output_scale,
            rmsnorm_output_scale(source, output_scale * 127.0),
        )

        independent_gains = derive_scaled_gains_q8(weights, output_scale)
        independent = reference_rmsnorm(activations, independent_gains)
        fixed = FixedRMSNorm(source, input_scale, output_scale)
        hidden_states = (
            torch.tensor(activations, dtype=torch.float32).reshape(1, 1, -1)
            * input_scale
        )
        fixed_raw = fixed(hidden_states).to(torch.int8).reshape(-1).tolist()
        self.assertEqual(independent.outputs, fixed_raw)

        legacy_gains = [round(weight * (1 << 8)) for weight in weights]
        legacy = reference_rmsnorm(activations, legacy_gains)
        first_difference = next(
            index
            for index, (repaired_raw, legacy_raw) in enumerate(
                zip(independent.outputs, legacy.outputs, strict=True)
            )
            if repaired_raw * output_scale != float(legacy_raw)
        )
        self.assertEqual(0, first_difference)
        self.assertEqual(-35, independent.outputs[first_difference])
        self.assertEqual(-1, legacy.outputs[first_difference])
        self.assertEqual(5734, independent_gains[first_difference])
        self.assertEqual(224, legacy_gains[first_difference])
        self.assertEqual(4_889_152, independent.sumsq)
        self.assertEqual(14_510_024, independent.inv_rms_q30)
        self.assertFalse(independent.saturation_seen)
        self.assertFalse(legacy.saturation_seen)
        print(
            "ACE2_RMSNORM_BISECT_PASS "
            "case=ramp coordinate=[0,0,0] activation_s8=-117 weight=0.875 "
            "expected_repaired_raw=-35 expected_repaired=-1.3671875 "
            "legacy_actual_raw=-1 legacy_actual=-1.0 input_scale=0.125 "
            "output_scale=0.0390625 repaired_gain_q7_8=5734 "
            "legacy_gain_q7_8=224 inv_rms_q30=14510024 "
            "rounding=nearest_ties_to_even repaired_saturation=0 legacy_saturation=0"
        )

    def test_exact_difference_records_coordinate_and_provenance(self) -> None:
        reference = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64)
        candidate = reference.clone()
        candidate[0, 1, 0] = 2.75
        raw = torch.tensor([[[8, 16], [22, 32]]], dtype=torch.int8)
        record = exact_difference(
            reference,
            candidate,
            raw,
            torch.full_like(candidate, 0.125),
            {
                "rounding": "nearest_ties_to_even",
                "saturation_range": [-128, 127],
                "scale": 0.125,
                "storage": "signed_int8",
            },
        )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual([0, 1, 0], record["coordinate"])
        self.assertEqual(3.0, record["expected_reference"])
        self.assertEqual(2.75, record["actual_fixed"])
        self.assertEqual(22, record["candidate_raw"])
        self.assertFalse(record["candidate_at_storage_limit"])

    def test_v_projection_precedes_rope_and_score(self) -> None:
        self.assertEqual(
            (
                "model.layers.0.q_projection",
                "model.layers.0.k_projection",
                "model.layers.0.v_projection",
                "model.layers.0.q_post_rope",
                "model.layers.0.k_post_rope",
                "model.layers.0.score",
            ),
            BOUNDARY_ORDER[1:],
        )


if __name__ == "__main__":
    unittest.main()
