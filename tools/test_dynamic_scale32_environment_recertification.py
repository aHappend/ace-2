#!/usr/bin/env python3
"""Static/synthetic tests for the no-execution environment recertifier."""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import recertify_dynamic_scale32_baseline_environment as recert  # noqa: E402


class DynamicScale32EnvironmentRecertificationTests(unittest.TestCase):
    def test_01_recertifier_does_not_import_runner_or_executor(self) -> None:
        source = Path(recert.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=recert.__file__)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("run_dynamic_scale32_verification_baseline", imported)
        self.assertNotIn("ace2_dynamic_scale32_baseline", imported)

    def test_02_static_output_contract_is_exactly_five(self) -> None:
        runner = tuple(recert.literal_assignment(recert.SOURCE_PATHS["runner"], "REQUIRED_OUTPUTS"))
        executor = tuple(recert.literal_assignment(recert.SOURCE_PATHS["executor"], "OUTPUT_NAMES"))
        self.assertEqual(runner, executor)
        self.assertEqual(len(runner), 5)

    def test_03_synthetic_filesystem_probe_is_temporary_and_atomic(self) -> None:
        required = tuple(recert.literal_assignment(recert.SOURCE_PATHS["runner"], "REQUIRED_OUTPUTS"))
        with tempfile.TemporaryDirectory(prefix="ace2-recert-test-") as temporary:
            root = Path(temporary)
            observed, gates = recert.filesystem_probe(root, required)
            self.assertTrue(all(gates.values()), observed)
            self.assertEqual(sorted(path.name for path in root.iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
