#!/usr/bin/env python3
"""Fail-closed package regressions for additive c01 command preparation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_w4a8_full_model_candidate as runner  # noqa: E402


class CandidatePackageTest(unittest.TestCase):
    def test_sha_companion_rejects_payload_and_companion_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "command.json"
            companion = root / "command.json.sha256"
            artifact.write_text("{}\n", encoding="utf-8")
            digest = runner.evaluator.sha256_file(artifact)
            companion.write_text(f"{digest}  command.json\n", encoding="utf-8")
            self.assertEqual(
                runner.verify_sha256_companion(
                    artifact,
                    companion,
                    "command.json",
                ),
                digest,
            )
            artifact.write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
                runner.verify_sha256_companion(
                    artifact,
                    companion,
                    "command.json",
                )
            artifact.write_text("{}\n", encoding="utf-8")
            companion.write_text(f"{digest}  wrong.json\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA companion differs"):
                runner.verify_sha256_companion(
                    artifact,
                    companion,
                    "command.json",
                )

    def test_bound_source_record_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            source.write_text("sealed\n", encoding="utf-8")
            record = runner.record_for(source)
            runner.verify_source_records([record])
            source.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source size differs|source hash differs"):
                runner.verify_source_records([record])

    def test_c00_execution_entrypoint_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "sealed terminal/unselectable"):
            runner.evaluator.execute_c00_attempt({}, "", {}, {})

    def test_synthetic_smoke_is_deterministic_and_quality_isolated(self) -> None:
        first = runner.synthetic_smoke("c01-mse-clip-grid")
        second = runner.synthetic_smoke("c01-mse-clip-grid")
        self.assertEqual(first, second)
        self.assertFalse(first["quality_data_accessed"])
        self.assertFalse(first["scoring_performed"])
        self.assertEqual(
            first["status"],
            "PASS_DETERMINISTIC_SYNTHETIC_NON_SCORING",
        )

    def test_original_c01_bundles_remain_the_frozen_superseded_inputs(self) -> None:
        expected = {
            "base": "41e4188a642ea547fd7f08689bcd1de8168706dc3af16d872ef51da5a54b17a8",
            "checkpoint-176": "53319f81b18a8c383e51a8ff6438884ba78d26b05a7404526dd813bed9092af3",
        }
        contract, _contract_sha256 = runner.frozen.load_contract()
        for alias, digest in expected.items():
            spec = runner.frozen.model_spec(contract, alias)
            root = runner.superseded_command_directory(
                spec,
                "c01-mse-clip-grid",
            )
            self.assertEqual(
                runner.verify_sha256_companion(
                    root / runner.COMMAND_NAME,
                    root / runner.COMMAND_SHA_NAME,
                    runner.COMMAND_NAME,
                ),
                digest,
            )
            self.assertNotEqual(
                root,
                runner.command_directory(spec, "c01-mse-clip-grid"),
            )


if __name__ == "__main__":
    unittest.main()
