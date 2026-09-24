#!/usr/bin/env python3
"""Independent checks for the best-null grouped RMSNorm-input payload."""

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
    ceil_scale32_from_ratio,
    round_divide_even_signed,
    scale32_ratio,
)


CANDIDATE = (
    ROOT
    / "evidence/candidates/w4a8-layer16-output-to-layer17-rmsnorm-input-repair-v1/"
    "candidate-0002"
)
Q_PREDECESSOR = (
    ROOT
    / "evidence/candidates/w4a8-layer17-q-output-source-grouped-repair-v1/"
    "candidate-0001/groups/group-128"
)
LAYER16_PREDECESSOR = (
    ROOT
    / "evidence/candidates/w4a8-layer16-source-grouped-activation-repair-v1/"
    "candidate-0002/groups/group-001"
)
HIDDEN = 896


def selected() -> tuple[dict[str, object], Path, int]:
    result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
    group_size = int(result["rtl_validation_rms_input_group_size"])
    return result, CANDIDATE / f"groups/group-{group_size:03d}", group_size


def read(group: Path, name: str, dtype: str, count: int) -> np.ndarray:
    values = np.fromfile(group / f"{name}.bin", dtype=dtype)
    if values.size != count:
        raise AssertionError(f"{name} count differs: {values.size} != {count}")
    return values


class Layer17RmsnormInputGroupedScale32ReferenceTest(unittest.TestCase):
    def test_best_null_selection_and_all_outcomes_are_preserved(self) -> None:
        result, _group, group_size = selected()
        freeze = json.loads((CANDIDATE / "candidate-freeze.json").read_text(encoding="utf-8"))
        self.assertIsNone(result["selected_rms_input_group_size"])
        self.assertEqual(group_size, 32)
        self.assertGreaterEqual(
            int(result["best_observed_reference_rank"]),
            int(result["baseline_reference_rank"]),
        )
        self.assertEqual([int(item["rms_input_group_size"]) for item in result["trace"]], [32, 16])
        self.assertEqual([int(item["rms_input_group_size"]) for item in result["failures"]], [8, 4, 1])
        self.assertEqual(freeze["prompt"]["chat_token_count"], 30)
        self.assertFalse(freeze["numeric_contract"]["bf16_runtime_sidecar"])
        self.assertFalse(freeze["numeric_contract"]["wider_carried_activation_payload"])
        self.assertFalse(freeze["numeric_contract"]["token_or_logit_override"])
        self.assertFalse(result["official_run_launched"])
        self.assertFalse(result["official_attempt_consumed"])

    def test_group_scales_are_exactly_derived_from_the_two_a8_sources(self) -> None:
        _result, group, group_size = selected()
        attention = read(group, "attention_group_s8", "i1", HIDDEN)
        attention_scale = read(group, "attention_group_scale32", "<i8", HIDDEN)
        down = read(group, "down_group_s8", "i1", HIDDEN)
        down_scale = read(group, "down_group_scale32", "<i8", HIDDEN)
        observed = read(group, "rms_input_group_scale32", "<i8", HIDDEN // group_size)
        for group_id, start in enumerate(range(0, HIDDEN, group_size)):
            maximum = Fraction(0, 1)
            for index in range(start, start + group_size):
                a_num, a_den = scale32_ratio(int(attention_scale[index]))
                d_num, d_den = scale32_ratio(int(down_scale[index]))
                value = Fraction(int(attention[index]) * a_num, a_den) + Fraction(
                    int(down[index]) * d_num, d_den
                )
                maximum = max(maximum, abs(value))
            expected = ceil_scale32_from_ratio(
                maximum.numerator, maximum.denominator * 127
            )
            self.assertEqual(int(observed[group_id]), expected)

    def test_boundary_rounding_is_exact_signed_ties_to_even(self) -> None:
        _result, group, group_size = selected()
        attention = read(group, "attention_group_s8", "i1", HIDDEN)
        attention_scale = read(group, "attention_group_scale32", "<i8", HIDDEN)
        down = read(group, "down_group_s8", "i1", HIDDEN)
        down_scale = read(group, "down_group_scale32", "<i8", HIDDEN)
        output_scale = read(group, "rms_input_group_scale32", "<i8", HIDDEN // group_size)
        expected = read(group, "rms_input_s8", "i1", HIDDEN)
        saturation = read(group, "rms_input_saturation", "u1", HIDDEN)
        for index in range(HIDDEN):
            a_num, a_den = scale32_ratio(int(attention_scale[index]))
            d_num, d_den = scale32_ratio(int(down_scale[index]))
            o_num, o_den = scale32_ratio(int(output_scale[index // group_size]))
            value = Fraction(int(attention[index]) * a_num, a_den) + Fraction(
                int(down[index]) * d_num, d_den
            )
            rounded = round_divide_even_signed(
                value.numerator * o_den, value.denominator * o_num
            )
            clipped = max(-128, min(127, rounded))
            self.assertEqual(int(expected[index]), clipped)
            self.assertEqual(int(saturation[index]), int(clipped != rounded))

    def test_w4_and_accepted_source_payloads_are_unchanged(self) -> None:
        _result, group, _group_size = selected()
        self.assertEqual(
            (group / "q_weight_s4.bin").read_bytes(),
            (Q_PREDECESSOR / "q_weight_s4.bin").read_bytes(),
        )
        for name in (
            "attention_group_s8.bin",
            "attention_group_scale32.bin",
            "down_group_s8.bin",
            "down_group_scale32.bin",
        ):
            self.assertEqual(
                (group / name).read_bytes(),
                (LAYER16_PREDECESSOR / name).read_bytes(),
                name,
            )


if __name__ == "__main__":
    unittest.main()
