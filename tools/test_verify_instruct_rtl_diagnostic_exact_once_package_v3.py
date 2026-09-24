#!/usr/bin/env python3
"""Zero-process regression for the diagnostic-Instruct v3 static verifier."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import ace2_exact_once_runtime_supervisor as supervisor  # noqa: E402
import verify_instruct_rtl_diagnostic_exact_once_package_v3 as verifier  # noqa: E402


class StaticV3VerifierTests(unittest.TestCase):
    def test_static_verifier_never_reserves_or_starts(self) -> None:
        absent_before = {
            path: os.path.lexists(path)
            for path in (
                verifier.STATE_DIR,
                verifier.RUNTIME_OUTPUT,
                verifier.STDOUT_PATH,
                verifier.STDERR_PATH,
                verifier.TERMINAL_PATH,
                verifier.AUTHORITY_RECORD_PATH,
            )
        }
        v2_before = supervisor.measure_source_tree(verifier.V2_EVIDENCE)
        with (
            mock.patch.object(supervisor, "run_once", side_effect=AssertionError("run_once called")),
            mock.patch.object(supervisor, "validate_run", side_effect=AssertionError("validate_run called")),
            mock.patch.object(supervisor, "_reserve_pre_start_intent", side_effect=AssertionError("reservation called")),
            mock.patch.object(subprocess, "Popen", side_effect=AssertionError("Popen called")),
        ):
            report = verifier.verify()
        v2_after = supervisor.measure_source_tree(verifier.V2_EVIDENCE)
        absent_after = {path: os.path.lexists(path) for path in absent_before}
        self.assertEqual(report["status"], "PASS_STATIC_ZERO_AUTHORITY")
        self.assertEqual(report["process_starts"], 0)
        self.assertEqual(report["state_reservations"], 0)
        self.assertEqual(absent_before, absent_after)
        self.assertFalse(any(absent_after.values()))
        self.assertEqual(v2_before, v2_after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
