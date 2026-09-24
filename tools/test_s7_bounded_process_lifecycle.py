#!/usr/bin/python3
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s7_bounded_process_lifecycle import run_bounded_process, sha256_bytes, sha256_file


TOOLS = Path(__file__).resolve().parent


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


diagnostic = import_file(
    "target_lifecycle_diagnostic_under_test", TOOLS / "run_s7_target_lifecycle_diagnostic.py"
)


class LifecycleFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected_executable = sha256_file(Path(sys.executable).resolve())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def run_fixture(
        self,
        source: str,
        *,
        natural: float = 0.12,
        timeout: float = 1.0,
        evaluator=None,
    ):
        if evaluator is None:
            evaluator = lambda stdout, stderr, return_code: {
                "passed": return_code == 0 and stdout == b"ok\n" and stderr == b""
            }
        return run_bounded_process(
            (sys.executable, "-I", "-B", "-c", source),
            cwd=self.temp.name,
            environment=os.environ.copy(),
            evaluator=evaluator,
            expected_executable_sha256s={self.expected_executable},
            identity_salt=self.id(),
            timeout_seconds=timeout,
            natural_quiescence_seconds=natural,
            term_grace_seconds=0.12,
            kill_grace_seconds=0.12,
            poll_interval_seconds=0.005,
        )

    def test_immediate_exit(self):
        record = self.run_fixture("print('ok', flush=True)")
        self.assertTrue(record["passed"])
        self.assertFalse(record["process_lifecycle"]["term_sent"])
        self.assertFalse(record["process_lifecycle"]["kill_sent"])

    def test_short_natural_child_quiescence(self):
        source = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable,'-I','-B','-c','import time; time.sleep(0.05)']); "
            "print('ok', flush=True)"
        )
        record = self.run_fixture(source, natural=0.20)
        self.assertTrue(record["passed"])
        self.assertGreaterEqual(record["process_tree"]["observed_process_count"], 2)
        self.assertIsNotNone(record["process_lifecycle"]["natural_quiescence_observed_ms"])
        self.assertFalse(record["process_lifecycle"]["term_sent"])

    def test_persistent_child_requires_term_and_fails(self):
        source = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable,'-I','-B','-c','import time; time.sleep(30)']); "
            "print('ok', flush=True)"
        )
        record = self.run_fixture(source, natural=0.04)
        self.assertFalse(record["passed"])
        self.assertEqual(record["process_exception_type"], "NaturalQuiescenceTimeout")
        self.assertTrue(record["process_lifecycle"]["term_sent"])
        self.assertFalse(record["process_lifecycle"]["kill_sent"])
        self.assertTrue(record["process_lifecycle"]["group_absent_after_cleanup"])

    def test_stubborn_child_requires_kill_and_fails(self):
        source = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable,'-I','-B','-c',"
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']); "
            "print('ok', flush=True)"
        )
        record = self.run_fixture(source, natural=0.04)
        self.assertFalse(record["passed"])
        self.assertEqual(record["process_exception_type"], "NaturalQuiescenceTimeout")
        self.assertTrue(record["process_lifecycle"]["term_sent"])
        self.assertTrue(record["process_lifecycle"]["kill_sent"])
        self.assertTrue(record["process_lifecycle"]["group_absent_after_cleanup"])

    def test_escaped_child_is_detected_cleaned_and_rejected(self):
        escaped_source = (
            "import os,signal,time; os.setsid(); "
            "signal.signal(signal.SIGTERM, signal.SIG_DFL); time.sleep(30)"
        )
        source = (
            "import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable,'-I','-B','-c',{escaped_source!r}]); "
            "time.sleep(0.08); print('ok', flush=True)"
        )
        record = self.run_fixture(source, natural=0.04)
        self.assertFalse(record["passed"])
        self.assertEqual(record["process_exception_type"], "EscapedDescendantProcess")
        self.assertTrue(record["process_lifecycle"]["escaped_descendant_detected"])
        self.assertTrue(record["process_lifecycle"]["group_absent_after_cleanup"])

    def test_nonzero_root_fails(self):
        record = self.run_fixture("import sys; print('ok', flush=True); sys.exit(7)")
        self.assertFalse(record["passed"])
        self.assertIsNone(record["process_exception_type"])
        self.assertEqual(record["sanitized_streams"]["return_code"], 7)

    def test_unexpected_surviving_executable_lineage_fails(self):
        source = (
            "import subprocess; "
            "subprocess.Popen(['/bin/sleep','0.06']); print('ok', flush=True)"
        )
        record = self.run_fixture(source, natural=0.20)
        self.assertFalse(record["passed"])
        self.assertEqual(record["process_exception_type"], "UnexpectedExecutableLineage")
        self.assertTrue(record["process_lifecycle"]["unexpected_executable_lineage"])
        self.assertFalse(record["process_lifecycle"]["term_sent"])

    def test_timeout_cleans_full_group(self):
        record = self.run_fixture("import time; time.sleep(30)", timeout=0.04)
        self.assertFalse(record["passed"])
        self.assertEqual(record["process_exception_type"], "TimeoutExpired")
        self.assertTrue(record["process_lifecycle"]["term_sent"])
        self.assertTrue(record["process_lifecycle"]["group_absent_after_cleanup"])


class ExactTargetEvaluationTests(unittest.TestCase):
    def test_exact_output_metadata_and_semantics(self):
        row = {
            "cuda": "",
            "location": "",
            "name": "msrresrchbasicvc",
            "resource_group": "gcr-singularity",
            "service": "sing",
            "sku": "",
            "subscription": "Singularity Shared",
            "vc": "",
            "workspace_name": "",
        }
        stdout = (json.dumps([row], sort_keys=True, separators=(",", ":")) + "\n").encode()
        result = diagnostic.evaluate_target_output(
            stdout,
            b"",
            0,
            expected_stdout_bytes=len(stdout),
            expected_stdout_sha256=sha256_bytes(stdout),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(diagnostic.EXPECTED_TARGET_STDOUT_BYTES, 201)
        self.assertEqual(
            diagnostic.EXPECTED_TARGET_STDOUT_SHA256,
            "cdc580d647f9ecae10bc60ca1e1d1a701a1d437f7db973c5f3a875080b98c426",
        )
        self.assertFalse(diagnostic.evaluate_target_output(stdout, b"", 0)["passed"])


if __name__ == "__main__":
    unittest.main()
