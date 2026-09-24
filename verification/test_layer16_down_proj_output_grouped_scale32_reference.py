#!/usr/bin/env python3
"""Independent checks for the grouped layer-16 down-projection output payload."""

from __future__ import annotations

import json
import sys
import unittest
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
    / "evidence/candidates/w4a8-layer16-down-proj-output-residual-source-repair-v1/"
    "candidate-0001"
)
LAYER16_PREDECESSOR = (
    ROOT
    / "evidence/candidates/w4a8-layer16-source-grouped-activation-repair-v1/"
    "candidate-0002/groups/group-001"
)
Q_PREDECESSOR = (
    ROOT
    / "evidence/candidates/w4a8-layer17-q-output-source-grouped-repair-v1/"
    "candidate-0001/groups/group-128"
)
HIDDEN = 896


def selected() -> tuple[dict[str, object], Path, int]:
    result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
    group_size = int(result["rtl_validation_down_output_group_size"])
    return result, CANDIDATE / f"groups/group-{group_size:03d}", group_size


def read(group: Path, name: str, dtype: str, count: int) -> np.ndarray:
    values = np.fromfile(group / f"{name}.bin", dtype=dtype)
    if values.size != count:
        raise AssertionError(f"{name} count differs: {values.size} != {count}")
    return values


class Layer16DownProjOutputGroupedScale32ReferenceTest(unittest.TestCase):
    def test_selection_policy_and_all_frozen_outcomes_are_preserved(self) -> None:
        result, _group, group_size = selected()
        freeze = json.loads((CANDIDATE / "candidate-freeze.json").read_text(encoding="utf-8"))
        observed = [int(item["down_output_group_size"]) for item in result["trace"]]
        failed = [int(item["down_output_group_size"]) for item in result["failures"]]
        self.assertEqual(sorted(observed + failed, key=[32, 16, 8, 4, 1].index), [32, 16, 8, 4, 1])
        self.assertIn(group_size, observed)
        if result["selected_down_output_group_size"] is None:
            self.assertGreaterEqual(
                int(result["best_observed_reference_rank"]),
                int(result["baseline_reference_rank"]),
            )
        else:
            self.assertTrue(
                bool(result["first_token_restored"])
                or int(result["selected_reference_rank"]) < int(result["baseline_reference_rank"])
            )
        self.assertEqual(freeze["baseline"]["layer16_source_group_size"], 1)
        self.assertEqual(freeze["baseline"]["layer17_q_output_group_size"], 128)
        self.assertFalse(freeze["numeric_contract"]["bf16_runtime_sidecar"])
        self.assertFalse(freeze["numeric_contract"]["wider_carried_activation_payload"])
        self.assertFalse(freeze["numeric_contract"]["token_or_logit_override"])
        self.assertFalse(result["official_run_launched"])
        self.assertFalse(result["official_attempt_consumed"])

    def test_group_scales_are_exactly_derived_from_the_down_a8_source(self) -> None:
        _result, group, group_size = selected()
        source = read(group, "down_source_s8", "i1", HIDDEN)
        base_scale32 = int(read(group, "down_base_scale32", "<i8", 1)[0])
        observed = read(group, "down_group_scale32", "<i8", HIDDEN // group_size)
        base_num, base_den = scale32_ratio(base_scale32)
        for group_id, start in enumerate(range(0, HIDDEN, group_size)):
            maximum = int(np.abs(source[start : start + group_size].astype(np.int16)).max())
            expected = ceil_scale32_from_ratio(maximum * base_num, 127 * base_den)
            self.assertEqual(int(observed[group_id]), expected)

    def test_boundary_rounding_is_exact_signed_ties_to_even(self) -> None:
        _result, group, group_size = selected()
        source = read(group, "down_source_s8", "i1", HIDDEN)
        base_scale32 = int(read(group, "down_base_scale32", "<i8", 1)[0])
        output_scale = read(group, "down_group_scale32", "<i8", HIDDEN // group_size)
        expected = read(group, "down_group_s8", "i1", HIDDEN)
        saturation = read(group, "down_saturation", "u1", HIDDEN)
        base_num, base_den = scale32_ratio(base_scale32)
        for index in range(HIDDEN):
            output_num, output_den = scale32_ratio(int(output_scale[index // group_size]))
            rounded = round_divide_even_signed(
                int(source[index]) * base_num * output_den,
                base_den * output_num,
            )
            clipped = max(-128, min(127, rounded))
            self.assertEqual(int(expected[index]), clipped)
            self.assertEqual(int(saturation[index]), int(clipped != rounded))

    def test_only_down_output_representation_changes(self) -> None:
        _result, group, _group_size = selected()
        for name in (
            "attention_source_s8.bin",
            "attention_group_s8.bin",
            "attention_group_scale32.bin",
            "down_source_s8.bin",
            "down_base_scale32.bin",
        ):
            self.assertEqual(
                (group / name).read_bytes(),
                (LAYER16_PREDECESSOR / name).read_bytes(),
                name,
            )
        self.assertEqual(
            (group / "q_weight_s4.bin").read_bytes(),
            (Q_PREDECESSOR / "q_weight_s4.bin").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
