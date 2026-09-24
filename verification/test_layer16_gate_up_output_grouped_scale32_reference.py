#!/usr/bin/env python3
"""Independent reference checks for layer-16 grouped gate/up output A8."""

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
    round_divide_even_unsigned,
    scale32_ratio,
)


CANDIDATE = ROOT / (
    "evidence/candidates/w4a8-layer16-gate-up-output-grouped-a8-repair-v1/"
    "candidate-0004"
)
POSITIONS = 30
CHANNELS = 4864


def selected() -> tuple[dict[str, object], Path, int]:
    result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
    selected_id = str(result["selected_contract_id"])
    matches = [
        (index, item)
        for index, item in enumerate(result["ordered_gate_up_output_contracts"])
        if item["contract_id"] == selected_id
    ]
    if len(matches) != 1:
        raise AssertionError("selected gate/up contract is not unique")
    index, item = matches[0]
    return result, CANDIDATE / "contracts" / f"{index:02d}-{selected_id}", int(item["output_group_size"])


def read(path: Path, dtype: str, count: int) -> np.ndarray:
    value = np.fromfile(path, dtype=dtype)
    if value.size != count:
        raise AssertionError(f"{path.name} count differs: {value.size} != {count}")
    return value


def derive_multiplier(input_scale: float, weight_scale: float, output_scale32: int) -> tuple[int, int]:
    input_num, input_den = float(input_scale).as_integer_ratio()
    weight_num, weight_den = float(weight_scale).as_integer_ratio()
    output_num, output_den = scale32_ratio(output_scale32)
    numerator = input_num * weight_num * output_den
    denominator = input_den * weight_den * output_num
    for shift in range(63, -1, -1):
        multiplier = round_divide_even_unsigned(numerator << shift, denominator)
        if multiplier <= (1 << 31) - 1:
            return multiplier, shift
    raise AssertionError("multiplier is not representable")


class Layer16GateUpOutputGroupedScale32ReferenceTest(unittest.TestCase):
    def test_dependency_gate_and_scope_are_frozen(self) -> None:
        result, _directory, group_size = selected()
        freeze = json.loads((CANDIDATE / "candidate-freeze.json").read_text(encoding="utf-8"))
        gate = freeze["fresh_l2_gate"]
        self.assertEqual(gate["control_reference_rank"], 3252)
        self.assertEqual(gate["control_top_token_id"], 12)
        self.assertEqual(gate["post_attention_rmsnorm_reference_rank"], 3426)
        self.assertEqual(gate["gate_up_projection_reference_rank"], 2843)
        self.assertTrue(result["control_exact_repeat"]["passed"])
        self.assertTrue(result["selected_exact_repeat_passed"])
        self.assertEqual(group_size, int(result["selected_gate_output_group_size"]))
        self.assertEqual(group_size, int(result["selected_up_output_group_size"]))
        self.assertFalse(freeze["numeric_contract"]["bf16_runtime_sidecar"])
        self.assertFalse(result["runtime_bf16_gate_up_sidecar"])
        self.assertFalse(result["official_run_launched"])
        self.assertFalse(result["official_attempt_consumed"])

    def test_all_frozen_outcomes_are_preserved(self) -> None:
        result, _directory, _group_size = selected()
        frozen = result["ordered_gate_up_output_contracts"]
        self.assertEqual(len(result["trace"]), len(frozen))
        self.assertEqual(
            [item["contract_id"] for item in result["trace"]],
            [item["contract_id"] for item in frozen],
        )
        for index, item in enumerate(frozen):
            directory = CANDIDATE / "contracts" / f"{index:02d}-{item['contract_id']}"
            self.assertTrue((directory / "result.json").is_file())

    def test_selected_group_scales_and_streams_are_independent(self) -> None:
        result, directory, group_size = selected()
        group_count = CHANNELS // group_size
        self.assertEqual(int(result["selected_gate_output_group_size"]), group_size)
        for stream in ("gate", "up"):
            input_scales = read(
                directory / f"{stream}_input_scale_f64.bin", "<f8", POSITIONS
            )
            weight_scales = read(
                directory / f"{stream}_weight_scale_f64.bin", "<f8", CHANNELS
            )
            accumulators = read(
                directory / f"{stream}_accumulator_s32.bin",
                "<i8",
                POSITIONS * CHANNELS,
            ).reshape(POSITIONS, CHANNELS)
            scales = read(
                directory / f"{stream}_output_group_scale32.bin",
                "<i8",
                POSITIONS * group_count,
            ).reshape(POSITIONS, group_count)
            control_scales = read(
                directory / f"{stream}_control_output_scale_f64.bin",
                "<f8",
                POSITIONS,
            )
            for position in (0, POSITIONS - 1):
                if result["selected_contract_id"] == "control":
                    self.assertEqual(
                        int(scales[position, 0]),
                        ceil_scale32_from_float(float(control_scales[position])),
                    )
                    continue
                input_num, input_den = float(input_scales[position]).as_integer_ratio()
                for group_id, start in enumerate(range(0, CHANNELS, group_size)):
                    maximum = Fraction(0, 1)
                    for row in range(start, start + group_size):
                        weight_num, weight_den = float(weight_scales[row]).as_integer_ratio()
                        maximum = max(
                            maximum,
                            Fraction(
                                abs(int(accumulators[position, row]))
                                * input_num
                                * weight_num,
                                input_den * weight_den,
                            ),
                        )
                    expected = ceil_scale32_from_ratio(
                        maximum.numerator, maximum.denominator * 127
                    )
                    self.assertEqual(int(scales[position, group_id]), expected)

        gate_scales = (directory / "gate_output_group_scale32.bin").read_bytes()
        up_scales = (directory / "up_output_group_scale32.bin").read_bytes()
        self.assertNotEqual(gate_scales, up_scales)

    def test_selected_integer_requantization_is_exact(self) -> None:
        result, directory, group_size = selected()
        group_count = CHANNELS // group_size
        for stream in ("gate", "up"):
            input_scales = read(
                directory / f"{stream}_input_scale_f64.bin", "<f8", POSITIONS
            )
            weight_scales = read(
                directory / f"{stream}_weight_scale_f64.bin", "<f8", CHANNELS
            )
            accumulators = read(
                directory / f"{stream}_accumulator_s32.bin",
                "<i8",
                POSITIONS * CHANNELS,
            ).reshape(POSITIONS, CHANNELS)
            scales = read(
                directory / f"{stream}_output_group_scale32.bin",
                "<i8",
                POSITIONS * group_count,
            ).reshape(POSITIONS, group_count)
            multipliers = read(
                directory / f"{stream}_multiplier_s32.bin",
                "<i8",
                POSITIONS * CHANNELS,
            ).reshape(POSITIONS, CHANNELS)
            shifts = read(
                directory / f"{stream}_shift_u6.bin",
                "<i8",
                POSITIONS * CHANNELS,
            ).reshape(POSITIONS, CHANNELS)
            expected_q = read(
                directory / f"{stream}_output_s8.bin",
                "i1",
                POSITIONS * CHANNELS,
            ).reshape(POSITIONS, CHANNELS)
            expected_sat = read(
                directory / f"{stream}_saturation.bin",
                "u1",
                POSITIONS * CHANNELS,
            ).reshape(POSITIONS, CHANNELS)
            for position in (0, POSITIONS - 1):
                for row in range(CHANNELS):
                    multiplier = int(multipliers[position, row])
                    shift = int(shifts[position, row])
                    if result["selected_contract_id"] != "control":
                        group_start = (row // group_size) * group_size
                        group_stop = group_start + group_size
                        if not np.any(accumulators[position, group_start:group_stop]):
                            derived = (0, 0)
                        else:
                            derived = derive_multiplier(
                                float(input_scales[position]),
                                float(weight_scales[row]),
                                int(scales[position, row // group_size]),
                            )
                        self.assertEqual((multiplier, shift), derived)
                    rounded = round_divide_even_signed(
                        int(accumulators[position, row]) * multiplier,
                        1 << shift,
                    )
                    self.assertEqual(int(expected_q[position, row]), max(-128, min(127, rounded)))
                    self.assertEqual(
                        int(expected_sat[position, row]),
                        int(rounded < -128 or rounded > 127),
                    )


if __name__ == "__main__":
    unittest.main()
