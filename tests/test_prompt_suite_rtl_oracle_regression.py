from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts import run_prompt_suite_rtl_oracle_regression as regression


class PromptSuiteRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = (
            Path(__file__).with_name("prompt_suite_rtl_oracle_strict_fresh_v1.json")
        )
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_suite_is_fixed_distinct_and_fresh(self) -> None:
        regression.validate_config(self.config)
        self.assertEqual(4, self.config["max_new_tokens"])
        self.assertEqual(3, len(self.config["cases"]))
        self.assertEqual(
            3,
            len({case["prompt"] for case in self.config["cases"]}),
        )
        self.assertFalse(any("reuse" in case for case in self.config["cases"]))

    def test_duplicate_prompt_is_rejected(self) -> None:
        invalid = copy.deepcopy(self.config)
        invalid["cases"][2]["prompt"] = invalid["cases"][1]["prompt"]
        with self.assertRaisesRegex(regression.RegressionError, "duplicate prompt"):
            regression.validate_config(invalid)

    def test_reused_attempt_is_rejected(self) -> None:
        invalid = copy.deepcopy(self.config)
        invalid["cases"][0]["reuse"] = {
            "attempt": "reports/verification/sealed",
            "oracle": "reports/verification/sealed-oracle",
        }
        with self.assertRaisesRegex(regression.RegressionError, "not a fresh execution"):
            regression.validate_config(invalid)

    def test_exactly_three_prompts_are_required(self) -> None:
        invalid = copy.deepcopy(self.config)
        invalid["cases"].pop()
        with self.assertRaisesRegex(regression.RegressionError, "exactly three"):
            regression.validate_config(invalid)

    def test_continuous_24_layer_kv_growth_is_accepted(self) -> None:
        generated = [10, 11, 12, 13]
        executions = []
        for position in range(5):
            executions.append(
                {
                    "absolute_position": position,
                    "layers": [
                        {
                            "layer_id": layer,
                            "cache_length_before": position,
                            "cache_length_after": position + 1,
                            "kv_append_bytes": 256,
                        }
                        for layer in range(24)
                    ],
                }
            )
        summary = {
            "context_contract": {"prompt_token_count": 2},
            "generated_token_ids": generated,
            "software_transformer_or_logits_fallback": False,
            "token_executions": executions,
        }
        coverage = regression.validate_kv_growth("case", generated, summary)
        self.assertTrue(coverage["continuous_24_layer_kv_growth"])
        self.assertEqual(3, coverage["decode_transitions_checked"])
        self.assertEqual(72, coverage["decode_layer_growth_checks"])

        summary["token_executions"][4]["layers"][23]["cache_length_after"] = 7
        with self.assertRaisesRegex(regression.RegressionError, "discontinuous K/V"):
            regression.validate_kv_growth("case", generated, summary)

    def test_fallback_is_rejected(self) -> None:
        with self.assertRaisesRegex(regression.RegressionError, "software transformer"):
            regression.validate_kv_growth(
                "case",
                [10, 11, 12, 13],
                {
                    "software_transformer_or_logits_fallback": True,
                    "generated_token_ids": [10, 11, 12, 13],
                },
            )

    def test_pass_coverage_requires_exact_acceptance_contract(self) -> None:
        coverage = {
            "configured_prompts": 3,
            "completed_prompts": 3,
            "fresh_rtl_prompts": 3,
            "generated_tokens": 12,
            "continuous_24_layer_kv_growth_prompts": 3,
            "decode_transitions_checked": 9,
            "decode_layer_growth_checks": 216,
            "kv_append_bytes_compared": 1,
            "full_cache_bytes_compared": 1,
            "quantized_layer_output_bytes_compared": 1,
            "full_vocabulary_logit_bytes_compared": 1,
            "full_vocabulary_head_steps_checked": 12,
            "integer_byte_mismatches": 0,
            "selected_token_mismatches": 0,
            "software_transformer_or_logits_fallback": False,
        }
        regression.validate_pass_coverage(coverage)
        coverage["generated_tokens"] = 11
        with self.assertRaisesRegex(regression.RegressionError, "exactly 12"):
            regression.validate_pass_coverage(coverage)

    def test_oracle_prompt_mismatch_is_rejected_before_aggregation(self) -> None:
        case = self.config["cases"][0]
        result = {
            "status": "PASS",
            "reused_inputs": {"prompt": {"sha256": "0" * 64}},
            "tokenization": {},
            "agreement": {},
        }
        with self.assertRaisesRegex(regression.RegressionError, "prompt hash differs"):
            regression.validate_oracle_result(case, regression.ROOT, result)


if __name__ == "__main__":
    unittest.main()
