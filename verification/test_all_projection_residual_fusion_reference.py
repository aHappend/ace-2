#!/usr/bin/env python3
"""Focused state/error checks for the reference-only residual fusion runtime."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from ace2_full_model_fixed_point import (  # noqa: E402
    AllProjectionResidualFusionRuntime,
    RMS_HIDDEN_SIZE,
    ceil_scale32_from_float,
)


class AllProjectionResidualFusionRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.accumulator = torch.zeros((1, RMS_HIDDEN_SIZE), dtype=torch.int32)
        self.residual = torch.zeros((1, RMS_HIDDEN_SIZE), dtype=torch.int8)
        self.scales = torch.full(
            (RMS_HIDDEN_SIZE,),
            ceil_scale32_from_float(1.0),
            dtype=torch.int64,
        )

    def apply_complete_pass(self, runtime: AllProjectionResidualFusionRuntime) -> None:
        for layer in range(24):
            runtime.apply(
                "attention",
                layer,
                self.accumulator,
                self.residual,
                self.scales,
                1.0,
                1.0,
            )
            runtime.apply(
                "mlp",
                layer,
                self.accumulator,
                self.residual,
                self.scales,
                1.0,
                1.0,
            )

    def test_complete_pass_restarts_cleanly(self) -> None:
        runtime = AllProjectionResidualFusionRuntime()
        self.apply_complete_pass(runtime)
        self.apply_complete_pass(runtime)
        summary = runtime.summary()
        self.assertEqual(summary["completed_passes"], 2)
        self.assertTrue(
            all(
                item["attention_joins"] == 24 and item["mlp_joins"] == 24
                for item in summary["passes"]
            )
        )

    def test_new_runtime_discards_incomplete_pass(self) -> None:
        runtime = AllProjectionResidualFusionRuntime()
        runtime.apply(
            "attention",
            0,
            self.accumulator,
            self.residual,
            self.scales,
            1.0,
            1.0,
        )
        runtime = AllProjectionResidualFusionRuntime()
        self.apply_complete_pass(runtime)
        self.assertEqual(runtime.summary()["completed_passes"], 1)

    def test_out_of_order_join_is_rejected(self) -> None:
        runtime = AllProjectionResidualFusionRuntime()
        with self.assertRaisesRegex(RuntimeError, "did not begin"):
            runtime.apply(
                "mlp",
                0,
                self.accumulator,
                self.residual,
                self.scales,
                1.0,
                1.0,
            )

    def test_incomplete_pass_cannot_be_summarized(self) -> None:
        runtime = AllProjectionResidualFusionRuntime()
        runtime.apply(
            "attention",
            0,
            self.accumulator,
            self.residual,
            self.scales,
            1.0,
            1.0,
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete pass"):
            runtime.summary()

    def test_malformed_scale_geometry_is_rejected(self) -> None:
        runtime = AllProjectionResidualFusionRuntime()
        with self.assertRaisesRegex(ValueError, "do not cover 896 lanes"):
            runtime.apply(
                "attention",
                0,
                self.accumulator,
                self.residual,
                self.scales[:-1],
                1.0,
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
