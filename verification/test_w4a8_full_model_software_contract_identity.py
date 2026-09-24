#!/usr/bin/env python3
"""Negative regression for the W4A8 BF16 full-prefix identity oracle."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_w4a8_full_model_software_contract_repair as runner  # noqa: E402


class FullPrefixIdentityOracleTest(unittest.TestCase):
    def test_cache_hash_substitution_is_rejected(self) -> None:
        fields = [
            "chat_template_input_ids",
            "generated_token_ids",
            "per_step_selected_token",
            "per_step_full_bf16_logit_tensor_sha256",
            "termination_reason",
        ]
        cache_hash = "a" * 64
        full_prefix_hash = "b" * 64
        prompt = {
            "cache_equivalence": [
                {
                    "cache_bf16_logit_tensor_sha256": cache_hash,
                    "full_prefix_bf16_logit_tensor_sha256": full_prefix_hash,
                    "full_prefix_selected_token": 42,
                    "logit_exact": False,
                    "logit_max_abs_difference": 0.125,
                    "logit_mismatch_count": 1,
                    "selected_token_exact": True,
                    "step_index": 0,
                }
            ],
            "decoded_text": "fixture",
            "identity": {
                "chat_template_input_ids": [1, 2, 3],
                "generated_token_ids": [42],
                "per_step_full_bf16_logit_tensor_sha256": [cache_hash],
                "per_step_selected_token": [42],
                "termination_reason": "max_new_tokens",
            },
            "prompt_id": "cache-full-prefix-substitution-fixture",
        }
        repaired = runner.repair_prompt_record(
            prompt,
            {"max_new_tokens": 1, "termination_token_ids": [99]},
            fields,
        )
        self.assertEqual(
            repaired["identity"]["per_step_full_bf16_logit_tensor_sha256"],
            [full_prefix_hash],
        )
        runner.verify_prompt_identity(repaired, fields)

        substituted = json.loads(json.dumps(repaired))
        substituted["identity"]["per_step_full_bf16_logit_tensor_sha256"] = [
            cache_hash
        ]
        with self.assertRaisesRegex(
            RuntimeError,
            "prompt identity does not match raw records",
        ):
            runner.verify_prompt_identity(substituted, fields)


if __name__ == "__main__":
    unittest.main()
