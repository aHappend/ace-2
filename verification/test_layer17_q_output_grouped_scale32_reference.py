#!/usr/bin/env python3
"""Independent checks for the selected grouped layer-17 Q-output payload."""

from __future__ import annotations

import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ace2_layer16_group4_scale32_reference import derive_scale32_multiplier
from ace2_quality_contracts import (
    ceil_scale32_from_ratio,
    round_divide_even_signed,
    scale32_ratio,
)


CANDIDATE = (
    ROOT
    / "evidence/candidates/w4a8-layer17-q-output-source-grouped-repair-v1/"
    "candidate-0001"
)
PREDECESSOR_GROUP = (
    ROOT
    / "evidence/candidates/w4a8-layer16-source-grouped-activation-repair-v1/"
    "candidate-0002/groups/group-001"
)
Q_OUTPUTS = 896
HEAD_DIM = 64


def selected() -> tuple[dict[str, object], Path, int]:
    result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
    group_size = int(result["selected_q_output_group_size"])
    return result, CANDIDATE / f"groups/group-{group_size:03d}", group_size


def read(group: Path, name: str, dtype: str, count: int) -> np.ndarray:
    value = np.fromfile(group / f"{name}.bin", dtype=dtype)
    if value.size != count:
        raise AssertionError(f"{name} count differs: {value.size} != {count}")
    return value


def scalar(group: Path, name: str) -> int:
    return int(read(group, name, "<i8", 1)[0])


class Layer17QOutputGroupedScale32ReferenceTest(unittest.TestCase):
    def test_selected_candidate_is_honest_and_scope_bounded(self) -> None:
        result, _group, group_size = selected()
        freeze = json.loads((CANDIDATE / "candidate-freeze.json").read_text(encoding="utf-8"))
        self.assertIn(group_size, freeze["ordered_q_output_group_sizes"])
        self.assertLess(
            int(result["selected_reference_rank"]),
            int(result["baseline_reference_rank"]),
        )
        self.assertFalse(freeze["numeric_contract"]["bf16_runtime_sidecar"])
        self.assertFalse(freeze["numeric_contract"]["token_or_logit_override"])
        self.assertEqual(freeze["numeric_contract"]["carried_q_samples"], "signed A8")
        self.assertFalse(result["official_run_launched"])
        self.assertFalse(result["official_attempt_consumed"])

    def test_group_scales_are_exactly_model_derived(self) -> None:
        _result, group, group_size = selected()
        group_count = Q_OUTPUTS // group_size
        accumulator = read(group, "q_accumulator_s32", "<i8", Q_OUTPUTS)
        input_scale = scalar(group, "q_input_scale32")
        weight_scales = read(group, "q_weight_scale32", "<i8", Q_OUTPUTS)
        observed_scales = read(
            group, "q_output_group_scale32", "<i8", group_count
        )
        input_num, input_den = scale32_ratio(input_scale)
        for group_id, start in enumerate(range(0, Q_OUTPUTS, group_size)):
            maximum = Fraction(0, 1)
            for row in range(start, start + group_size):
                weight_num, weight_den = scale32_ratio(int(weight_scales[row]))
                value = Fraction(
                    abs(int(accumulator[row])) * input_num * weight_num,
                    input_den * weight_den,
                )
                maximum = max(maximum, value)
            expected = ceil_scale32_from_ratio(
                maximum.numerator, maximum.denominator * 127
            )
            self.assertEqual(int(observed_scales[group_id]), expected)

        head_scales = read(group, "q_head_scale32", "<i8", Q_OUTPUTS // HEAD_DIM)
        self.assertEqual(group_size % HEAD_DIM, 0)
        for head in range(Q_OUTPUTS // HEAD_DIM):
            self.assertEqual(
                int(head_scales[head]),
                int(observed_scales[(head * HEAD_DIM) // group_size]),
            )

    def test_accumulator_to_grouped_a8_rounding_is_exact(self) -> None:
        _result, group, group_size = selected()
        accumulator = read(group, "q_accumulator_s32", "<i8", Q_OUTPUTS)
        input_scale = scalar(group, "q_input_scale32")
        weight_scales = read(group, "q_weight_scale32", "<i8", Q_OUTPUTS)
        group_scales = read(
            group, "q_output_group_scale32", "<i8", Q_OUTPUTS // group_size
        )
        multipliers = read(group, "q_multiplier_s32", "<i8", Q_OUTPUTS)
        shifts = read(group, "q_shift_u6", "<i8", Q_OUTPUTS)
        expected_q = read(group, "q_output_s8", "i1", Q_OUTPUTS)
        expected_sat = read(group, "q_saturation", "u1", Q_OUTPUTS)
        for row in range(Q_OUTPUTS):
            scale = int(group_scales[row // group_size])
            multiplier, shift = derive_scale32_multiplier(
                input_scale, int(weight_scales[row]), scale
            )
            self.assertEqual(int(multipliers[row]), multiplier)
            self.assertEqual(int(shifts[row]), shift)
            rounded = round_divide_even_signed(
                int(accumulator[row]) * multiplier, 1 << shift
            )
            self.assertEqual(int(expected_q[row]), max(-128, min(127, rounded)))
            self.assertEqual(
                int(expected_sat[row]), int(rounded < -128 or rounded > 127)
            )

    def test_w4_weights_and_layer16_source_payload_are_unchanged(self) -> None:
        _result, group, _group_size = selected()
        self.assertEqual(
            (group / "q_weight_s4.bin").read_bytes(),
            (PREDECESSOR_GROUP / "q_weight_s4.bin").read_bytes(),
        )
        for name in (
            "attention_source_s8.bin",
            "down_source_s8.bin",
            "sum_s8.bin",
            "rms_output_s8.bin",
            "q_accumulator_s32.bin",
        ):
            self.assertEqual(
                (group / name).read_bytes(),
                (PREDECESSOR_GROUP / name).read_bytes(),
                name,
            )


if __name__ == "__main__":
    unittest.main()
