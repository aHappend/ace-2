import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/diagnose_w4a8_c02_numerical_path.py"
)
SPEC = importlib.util.spec_from_file_location("c02_numerical", MODULE_PATH)
assert SPEC and SPEC.loader
c02 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c02)


class C02NumericalPathTest(unittest.TestCase):
    def test_independent_oracle_matches_frozen_quantizer_on_ties(self) -> None:
        candidate = {
            "candidate_id": "c01-mse-clip-grid",
            "weight_generation": {
                "clip_ratio_denominator": 1000,
                "clip_ratio_numerators": [1000, 990, 975, 950, 925, 900],
            },
        }
        weight = torch.tensor(
            [[-1.0, -0.5, 0.0, 0.5, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.bfloat16,
        )
        observed = c02.oracle_mse_clip_grid_quantization_scale32(weight, candidate)
        expected = c02.evaluator.mse_clip_grid_quantization_scale32(weight, candidate)
        for actual, reference in zip(observed, expected, strict=True):
            self.assertTrue(torch.equal(actual, reference))

    def test_tensor_bundle_is_deterministic(self) -> None:
        tensors = {
            "b": torch.tensor([1, -2], dtype=torch.int8),
            "a": torch.tensor([[1.5]], dtype=torch.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.bin"
            second = Path(directory) / "second.bin"
            c02.write_tensor_bundle(first, tensors)
            c02.write_tensor_bundle(second, tensors)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_boundary_classification_prefers_severe_stage(self) -> None:
        stages = [
            {
                "boundary_class": "attention",
                "metrics": {"severity": "MATERIAL"},
                "ordinal": 0,
                "stage": "layer0.attention_value",
            },
            {
                "boundary_class": "projection_requantization",
                "metrics": {"severity": "SEVERE"},
                "ordinal": 1,
                "stage": "layer0.requantization",
            },
        ]
        result = c02.classify_boundary(
            {"status": "PASS_EXACT_FROZEN_SOURCE_RECONSTRUCTION"}, stages
        )
        self.assertEqual(result["earliest_divergence"], "layer0.attention_value")
        self.assertEqual(
            result["remediation_class"],
            "FIXED_POINT_ATTENTION_VALUE_PATH_OR_SCALE_ALIGNMENT_REPAIR_PTQ_CREDIBLE",
        )


if __name__ == "__main__":
    unittest.main()
