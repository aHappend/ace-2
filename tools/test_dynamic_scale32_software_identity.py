#!/usr/bin/env python3
"""Synthetic-only coverage for Dynamic Scale32 software identity remediation."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from ace2_software_identity import (  # noqa: E402
    evaluate_distribution_versions,
    require_distribution_versions,
)
from run_dynamic_scale32_verification_baseline import (  # noqa: E402
    AUTHORITY_REL,
    DynamicScale32BaselineRunner,
)


class DynamicScale32SoftwareIdentityTests(unittest.TestCase):
    def test_01_torch_local_build_matches_frozen_public_version(self) -> None:
        observed = {
            "datasets": "4.8.5",
            "lm-eval": "0.4.9.2",
            "torch": "2.11.0+cu130",
            "transformers": "4.57.6",
        }
        frozen = {
            "datasets": "4.8.5",
            "lm_eval": "0.4.9.2",
            "torch": "2.11.0",
            "transformers": "4.57.6",
        }
        records = evaluate_distribution_versions(frozen, version_getter=observed.__getitem__)
        self.assertTrue(all(record["matches"] for record in records.values()))
        self.assertEqual(records["torch"]["observed_public"], "2.11.0")
        self.assertEqual(records["torch"]["observed_local"], "cu130")
        self.assertEqual(
            require_distribution_versions(frozen, version_getter=observed.__getitem__),
            frozen,
        )

    def test_02_incompatible_public_versions_fail_closed(self) -> None:
        frozen = {"datasets": "4.8.5", "torch": "2.11.0"}
        for incompatible in ("2.11.1", "2.12.0", "2.11.0rc1"):
            with self.subTest(incompatible=incompatible):
                observed = {"datasets": "4.8.5", "torch": incompatible}
                with self.assertRaisesRegex(RuntimeError, "public_version_mismatch"):
                    require_distribution_versions(frozen, version_getter=observed.__getitem__)
        unrelated_drift = {"datasets": "4.8.6", "torch": "2.11.0+cu130"}
        with self.assertRaisesRegex(RuntimeError, "datasets"):
            require_distribution_versions(frozen, version_getter=unrelated_drift.__getitem__)

    def test_03_runner_probe_uses_same_policy_without_reservation(self) -> None:
        frozen = {
            "datasets": "4.8.5",
            "lm_eval": "0.4.9.2",
            "torch": "2.11.0",
            "transformers": "4.57.6",
        }
        observed = {
            "datasets": "4.8.5",
            "lm-eval": "0.4.9.2",
            "torch": "2.11.0+cu130",
            "transformers": "4.57.6",
        }
        with tempfile.TemporaryDirectory(prefix="ace2-scale32-version-probe-") as temporary:
            root = Path(temporary)
            authority_path = root / AUTHORITY_REL
            authority_path.parent.mkdir(parents=True, exist_ok=True)
            authority_path.write_text(
                json.dumps(
                    {
                        "run_id": "dynamic-scale32-baseline-recovery-fixture",
                        "software_versions": frozen,
                        "model_counts": {"baseline_model": 1, "candidate_model": 0},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            tools_dir = root / "tools"
            tools_dir.mkdir(parents=True, exist_ok=True)
            for name in ("ace2_dynamic_scale32_baseline.py", "ace2_software_identity.py"):
                shutil.copy2(TOOLS / name, tools_dir / name)
            runner = DynamicScale32BaselineRunner(root)
            from unittest import mock

            with mock.patch("ace2_software_identity.metadata.version", side_effect=observed.__getitem__):
                self.assertEqual(runner._runtime_version_blockers(frozen), [])
            self.assertEqual(runner._executor_blockers(), [])
            self.assertFalse(runner.state_root.exists())
            self.assertFalse(runner.ledger_path.exists())
            self.assertFalse(runner.published_root.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
