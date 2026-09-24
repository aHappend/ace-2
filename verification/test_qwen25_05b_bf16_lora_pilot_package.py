#!/usr/bin/env python3
"""Focused non-consuming regression for the A100 LoRA pilot package."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_qwen25_05b_bf16_lora_pilot_package as package_audit  # noqa: E402


class Qwen25Bf16LoraPilotPackageTest(unittest.TestCase):
    def test_package_is_build_ready_without_consuming_one_shot_paths(self) -> None:
        result = package_audit.audit()
        self.assertEqual(result["status"], "PASS")
        self.assertGreaterEqual(result["check_count"], 50)
        self.assertFalse(package_audit.common.PREFLIGHT_DIR.exists())
        self.assertFalse(package_audit.common.ATTEMPT_DIR.exists())
        self.assertIn("No A100 connectivity probe", result["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
