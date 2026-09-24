#!/usr/bin/env python3
"""Independent checks for grouped layer-17 RMSNorm-output Scale32."""

from __future__ import annotations

import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ace2_quality_contracts import (
    ceil_scale32_from_float,
    ceil_scale32_from_ratio,
    round_divide_even_signed,
    scale32_ratio,
)
from ace2_rmsnorm_reference import (
    derive_rmsnorm_output_scale,
    derive_scaled_gains_q8,
    reference_rmsnorm,
)


CANDIDATE = (
    ROOT
    / "evidence/candidates/w4a8-layer17-rmsnorm-output-source-grouped-repair-v1/"
    "candidate-0001"
)
PREDECESSOR = (
    ROOT
    / "evidence/candidates/w4a8-layer17-q-output-source-grouped-repair-v1/"
    "candidate-0001/groups/group-128"
)
SEARCH_REVIEW = ROOT / "build/l2-review-layer17-q-output-search-r1/result.json"
RTL_REVIEW = ROOT / "build/l2-review-layer17-q-output-rtl-r1/result.json"
HIDDEN = 896
Q_OUTPUT_GROUP_SIZE = 128


def selected() -> tuple[dict[str, object], Path, int]:
    result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
    group_size = int(result["rtl_validation_rms_output_group_size"])
    return result, CANDIDATE / f"groups/group-{group_size:03d}", group_size


def read(group: Path, name: str, dtype: str, count: int) -> np.ndarray:
    value = np.fromfile(group / f"{name}.bin", dtype=dtype)
    if value.size != count:
        raise AssertionError(f"{name} count differs: {value.size} != {count}")
    return value


def scalar(group: Path, name: str) -> int:
    return int(read(group, name, "<i8", 1)[0])


class Layer17RmsnormOutputGroupedScale32ReferenceTest(unittest.TestCase):
    def test_dependencies_and_scope_are_frozen(self) -> None:
        result, _group, group_size = selected()
        freeze = json.loads((CANDIDATE / "candidate-freeze.json").read_text(encoding="utf-8"))
        search_review = json.loads(SEARCH_REVIEW.read_text(encoding="utf-8"))
        rtl_review = json.loads(RTL_REVIEW.read_text(encoding="utf-8"))
        self.assertIn(group_size, freeze["ordered_rms_output_group_sizes"])
        self.assertEqual(search_review["selected_q_output_group_size"], 128)
        self.assertEqual(search_review["selected_reference_rank"], 33375)
        self.assertEqual(
            rtl_review["status"],
            "PASS_FOCUSED_LAYER17_Q_OUTPUT_GROUPED_SCALE32_RTL_REFERENCE_AGREEMENT",
        )
        self.assertEqual(
            freeze["numeric_contract"]["carried_activation_samples"], "signed A8"
        )
        self.assertFalse(freeze["numeric_contract"]["bf16_runtime_sidecar"])
        self.assertFalse(freeze["numeric_contract"]["wider_activation_payload"])
        self.assertFalse(freeze["numeric_contract"]["token_or_logit_override"])
        self.assertFalse(result["official_run_launched"])
        self.assertFalse(result["official_attempt_consumed"])

    def test_selection_policy_matches_all_preserved_outcomes(self) -> None:
        result, _group, _group_size = selected()
        order = {
            group: index
            for index, group in enumerate(result["ordered_rms_output_group_sizes"])
        }
        trace = result["trace"]
        best = min(
            trace,
            key=lambda item: (
                int(item["reference_rank"]),
                order[int(item["rms_output_group_size"])],
            ),
        )
        self.assertEqual(
            int(result["best_observed_rms_output_group_size"]),
            int(best["rms_output_group_size"]),
        )
        self.assertEqual(
            int(result["best_observed_reference_rank"]), int(best["reference_rank"])
        )
        selected_group = result["selected_rms_output_group_size"]
        if selected_group is None:
            self.assertGreaterEqual(
                int(best["reference_rank"]), int(result["baseline_reference_rank"])
            )
        else:
            selected_record = next(
                item
                for item in trace
                if int(item["rms_output_group_size"]) == int(selected_group)
            )
            self.assertTrue(
                int(selected_record["top_token_id"]) == 9707
                or int(selected_record["reference_rank"])
                < int(result["baseline_reference_rank"])
            )

    def test_group_scale_gain_and_rms_output_are_exact(self) -> None:
        _result, group, group_size = selected()
        group_count = HIDDEN // group_size
        calibrated = read(group, "binary_input_norm", "<f4", HIDDEN)
        weights = read(group, "rms_weight_f32", "<f4", HIDDEN)
        observed_scales = read(
            group, "rms_output_group_scale32", "<i8", group_count
        )
        observed_gains = read(group, "rms_gain_s16_q8", "<i2", HIDDEN)
        expected_gains: list[int] = []
        for group_id, start in enumerate(range(0, HIDDEN, group_size)):
            stop = start + group_size
            scale = derive_rmsnorm_output_scale(
                weights[start:stop].astype(float).tolist(),
                max(float(np.max(np.abs(calibrated[start:stop]))), 1.0e-12),
            )
            expected_scale = ceil_scale32_from_float(scale)
            self.assertEqual(int(observed_scales[group_id]), expected_scale)
            numerator, denominator = scale32_ratio(expected_scale)
            expected_gains.extend(
                derive_scaled_gains_q8(
                    weights[start:stop].astype(float).tolist(),
                    numerator / denominator,
                )
            )
        self.assertEqual(observed_gains.astype(int).tolist(), expected_gains)

        source = read(group, "sum_s8", "i1", HIDDEN).astype(int).tolist()
        expected = reference_rmsnorm(source, expected_gains)
        observed_output = read(group, "rms_output_s8", "i1", HIDDEN)
        self.assertEqual(observed_output.astype(int).tolist(), expected.outputs)
        self.assertEqual(scalar(group, "rms_sumsq"), expected.sumsq)
        self.assertEqual(scalar(group, "rms_inv_q30"), expected.inv_rms_q30)
        self.assertEqual(
            int(read(group, "rms_saturation", "u1", 1)[0]),
            int(expected.saturation_seen),
        )
        self.assertTrue(np.all(observed_output >= -128))
        self.assertTrue(np.all(observed_output <= 127))

    def test_grouped_q_projection_consumes_scale32_exactly(self) -> None:
        _result, group, group_size = selected()
        input_group_count = HIDDEN // group_size
        partials = read(
            group,
            "q_partial_accumulator_s32",
            "<i8",
            HIDDEN * input_group_count,
        ).reshape(HIDDEN, input_group_count)
        input_scales = read(
            group, "rms_output_group_scale32", "<i8", input_group_count
        )
        weight_scales = read(group, "q_weight_scale32", "<i8", HIDDEN)
        output_scales = read(
            group,
            "q_output_group_scale32",
            "<i8",
            HIDDEN // Q_OUTPUT_GROUP_SIZE,
        )
        expected_q = read(group, "q_output_s8", "i1", HIDDEN)
        observed_sat = read(group, "q_saturation", "u1", HIDDEN)

        real_values: list[Fraction] = []
        for row in range(HIDDEN):
            activation = Fraction(0, 1)
            for input_group in range(input_group_count):
                numerator, denominator = scale32_ratio(
                    int(input_scales[input_group])
                )
                activation += Fraction(
                    int(partials[row, input_group]) * numerator, denominator
                )
            weight_numerator, weight_denominator = scale32_ratio(
                int(weight_scales[row])
            )
            real_values.append(
                activation * Fraction(weight_numerator, weight_denominator)
            )

        for output_group, start in enumerate(
            range(0, HIDDEN, Q_OUTPUT_GROUP_SIZE)
        ):
            maximum = max(
                abs(value)
                for value in real_values[start : start + Q_OUTPUT_GROUP_SIZE]
            )
            self.assertEqual(
                int(output_scales[output_group]),
                ceil_scale32_from_ratio(
                    maximum.numerator, maximum.denominator * 127
                ),
            )

        for row, value in enumerate(real_values):
            scale_numerator, scale_denominator = scale32_ratio(
                int(output_scales[row // Q_OUTPUT_GROUP_SIZE])
            )
            rounded = round_divide_even_signed(
                value.numerator * scale_denominator,
                value.denominator * scale_numerator,
            )
            self.assertEqual(int(expected_q[row]), max(-128, min(127, rounded)))
            self.assertEqual(
                int(observed_sat[row]), int(rounded < -128 or rounded > 127)
            )

    def test_w4_payload_is_unchanged(self) -> None:
        _result, group, _group_size = selected()
        self.assertEqual(
            (group / "q_weight_s4.bin").read_bytes(),
            (PREDECESSOR / "q_weight_s4.bin").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
