import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/diagnose_w4a8_c02_attention_substages.py"
)
SPEC = importlib.util.spec_from_file_location("c02_attention_substages", MODULE_PATH)
assert SPEC and SPEC.loader
c02 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c02)


class C02AttentionSubstageTest(unittest.TestCase):
    def test_classification_uses_first_material_stage(self) -> None:
        stages = [
            {
                "metrics": {"severity": "TRACKING"},
                "ordinal": 0,
                "stage": "layer0.q_rope",
            },
            {
                "metrics": {"severity": "MATERIAL"},
                "ordinal": 1,
                "stage": "layer0.softmax",
            },
            {
                "metrics": {"severity": "SEVERE"},
                "ordinal": 2,
                "stage": "layer0.value",
            },
        ]
        result = c02.classify_model(stages)
        self.assertEqual(result["earliest_material_primitive"], "layer0.softmax")
        self.assertEqual(result["first_severe_stage"]["stage"], "layer0.value")

    def test_scale32_decode_and_even_rounding(self) -> None:
        record = c02.fixed.ceil_scale32_from_float(0.25)
        self.assertEqual(float(c02.decoded_scale32(record)), 0.25)
        values = torch.tensor([-128.5, -2.5, -1.5, 1.5, 2.5, 127.5])
        observed = c02.stable_round_even_int8(values)
        expected = torch.tensor([-128, -2, -2, 2, 2, 127], dtype=torch.int8)
        self.assertTrue(torch.equal(observed, expected))

    def test_shared_classification_preserves_exact_o_proj_chain(self) -> None:
        def result() -> dict:
            return {
                "classification": {
                    "earliest_material_primitive": (
                        "model.layers.0.self_attn.softmax_probabilities"
                    )
                },
                "scale_alignment_invariants": {
                    "attention_output_code_equals_o_proj_input_code": True,
                    "consumer_scale_exactly_equals_decoded_scale32": True,
                    "consumer_scale_exactly_equals_v_projection_output_scale": True,
                    "score_perturbation_probability_metrics": {
                        "severity": "TRACKING"
                    },
                    "softmax_realization_local_metrics": {
                        "severity": "MATERIAL"
                    },
                },
            }

        observed = c02.shared_classification(
            {"base": result(), "checkpoint-176": result()}
        )
        self.assertEqual(observed["status"], "PASS_SHARED_PRIMITIVE_LOCALIZED")
        self.assertEqual(
            observed["shared_repair_invariant"],
            "CENTERED_Q6_9_TO_Q0_15_SOFTMAX_REALIZATION",
        )
        self.assertTrue(observed["o_proj_scale_and_code_chain_exact_for_both_models"])


if __name__ == "__main__":
    unittest.main()
