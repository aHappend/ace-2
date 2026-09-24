#!/usr/bin/env python3
"""Regression for the non-consuming V2 launch guard."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "pilot/qwen25_05b_bf16_lora_product_v2"
sys.path.insert(0, str(PACKAGE))

import common  # noqa: E402


class Qwen25Bf16LoraProductV2PreflightTest(unittest.TestCase):
    def test_exact_isolated_environment_is_ready_without_consumption(self) -> None:
        result = common.audit_launch_readiness()
        self.assertEqual(result["status"], "PASS_READY_TO_CONSUME_PREFLIGHT")
        self.assertIsNone(result["first_failed_gate"])
        self.assertIsNone(result["failure_taxonomy"])
        self.assertTrue(result["checks"]["package_versions_exact"])
        self.assertTrue(result["checks"]["generator_reproduction_pass"])
        self.assertTrue(result["checks"]["cpu_bf16_semantics"])
        self.assertTrue(result["checks"]["retention_score_policy_frozen"])
        self.assertTrue(result["checks"]["evaluator_classification_frozen"])
        self.assertTrue(result["checks"]["accepted_l2_amendment_review"])
        self.assertFalse(result["official_namespace_exists"])
        self.assertEqual(result["v2_preflight_marker_count"], 0)
        self.assertEqual(result["v2_attempt_marker_count"], 0)


if __name__ == "__main__":
    unittest.main()
