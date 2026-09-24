#!/usr/bin/env python3
"""Harmless fake-child tests for the exact-once runtime supervisor."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import ace2_exact_once_runtime_supervisor as supervisor  # noqa: E402


EXACT_PYTHON = Path("/usr/bin/python3")


FAKE_CHILD = """#!/usr/bin/env python3
import os
import signal
import sys
from pathlib import Path

mode = sys.argv[1]
counter = Path(sys.argv[2])
with counter.open("a", encoding="utf-8") as stream:
    stream.write("started\\n")
    stream.flush()
    os.fsync(stream.fileno())
print(f"fake-child-stdout:{mode}", flush=True)
print(f"fake-child-stderr:{mode}", file=sys.stderr, flush=True)
if mode == "success":
    raise SystemExit(0)
if mode == "nonzero":
    raise SystemExit(7)
if mode == "signal":
    os.kill(os.getpid(), signal.SIGTERM)
raise SystemExit(99)
"""


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(supervisor.canonical_json_bytes(value, pretty=True))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExactOnceRuntimeSupervisorTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ace2-exact-once-supervisor-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "immutable-source"
        self.config = self.root / "config"
        self.outputs = self.root / "outputs"
        self.state_parent = self.root / "state"
        for path in (self.source, self.config, self.outputs, self.state_parent):
            path.mkdir()
        self.child = self.source / "fake_child.py"
        self.child.write_text(FAKE_CHILD, encoding="utf-8")
        self.tree_context = self.source / "tree_context.txt"
        self.tree_context.write_text("tree-a\n", encoding="utf-8")
        self.python = EXACT_PYTHON.resolve(strict=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_run(self, name: str, mode: str) -> dict[str, Any]:
        counter = self.outputs / f"{name}.counter"
        stdout = self.outputs / f"{name}.stdout"
        stderr = self.outputs / f"{name}.stderr"
        terminal = self.outputs / f"{name}.terminal.json"
        state_dir = self.state_parent / name
        argv = [str(self.python), str(self.child), mode, str(counter)]
        bindings = {
            "bound_files": [
                supervisor.file_binding(self.python),
                supervisor.file_binding(self.child),
            ],
            "source_tree": supervisor.source_tree_binding(self.source),
            "cwd": str(self.source),
            "argv": argv,
            "argv_sha256": supervisor.canonical_value_sha256(argv),
            "environment": {
                "LC_ALL": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "required_absent_outputs": [
                str(counter),
                str(stdout),
                str(stderr),
                str(terminal),
            ],
        }
        package = {
            "schema": supervisor.PACKAGE_SCHEMA,
            "package_id": f"harmless-fake-child-{name}",
            "bindings": bindings,
        }
        package_path = self.config / f"{name}.package.json"
        write_json(package_path, package)
        package_measurement = supervisor.measure_file(package_path)
        authority = {
            "schema": supervisor.AUTHORITY_SCHEMA,
            "authority_id": f"test-only-fake-child-authority-{name}",
            "decision": "execute_once",
            "package": {
                "path": str(package_path),
                "byte_count": package_measurement.byte_count,
                "sha256": package_measurement.sha256,
                "package_id": package["package_id"],
            },
            "bindings": copy.deepcopy(bindings),
            "runtime": {
                "state_dir": str(state_dir),
                "stdout_path": str(stdout),
                "stderr_path": str(stderr),
                "terminal_record_path": str(terminal),
            },
        }
        authority_path = self.config / f"{name}.authority.json"
        write_json(authority_path, authority)
        return {
            "package": package,
            "package_path": package_path,
            "authority": authority,
            "authority_path": authority_path,
            "counter": counter,
            "stdout": stdout,
            "stderr": stderr,
            "terminal": terminal,
            "state_dir": state_dir,
        }

    def starts(self, run: dict[str, Any]) -> int:
        if not run["counter"].exists():
            return 0
        return run["counter"].read_text(encoding="utf-8").splitlines().count("started")

    def assert_complete_terminal(
        self,
        run: dict[str, Any],
        record: dict[str, Any],
        *,
        child_outcome: str,
        exit_code: int | None,
        terminating_signal: int | None,
    ) -> None:
        primary = json.loads(run["terminal"].read_text(encoding="utf-8"))
        durable = json.loads(
            (run["state_dir"] / "terminal_record.json").read_text(encoding="utf-8")
        )
        self.assertEqual(primary, record)
        self.assertEqual(durable, record)
        self.assertEqual(record["schema"], supervisor.TERMINAL_SCHEMA)
        self.assertTrue(record["process_started"])
        self.assertEqual(record["process_start_count"], 1)
        process = record["process"]
        self.assertIsInstance(process["pid"], int)
        self.assertEqual(process["child_outcome"], child_outcome)
        self.assertEqual(process["exit_code"], exit_code)
        self.assertEqual(process["signal"], terminating_signal)
        self.assertTrue(math.isfinite(process["wall_time_seconds"]))
        self.assertGreaterEqual(process["wall_time_seconds"], 0.0)
        self.assertIsInstance(process["peak_rss_kib"], int)
        self.assertGreaterEqual(process["peak_rss_kib"], 0)
        for name in ("stdout", "stderr"):
            capture = record["captures"][name]
            path = run[name]
            self.assertTrue(capture["published"])
            self.assertEqual(capture["byte_count"], path.stat().st_size)
            self.assertEqual(capture["sha256"], sha256_file(path))
        # Re-serialization explicitly exercises allow_nan=False with valid Python values.
        serialized = supervisor.canonical_json_bytes(record, pretty=True)
        self.assertEqual(json.loads(serialized), record)

    def test_01_success_one_start_and_complete_terminal(self) -> None:
        run = self.make_run("success", "success")
        completed = subprocess.run(
            [
                str(EXACT_PYTHON),
                str(Path(supervisor.__file__).resolve()),
                "--package",
                str(run["package_path"]),
                "--authority",
                str(run["authority_path"]),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["process_start_count"], 1)
        record = json.loads(run["terminal"].read_text(encoding="utf-8"))
        self.assertEqual(record["outcome"], "success")
        self.assertEqual(self.starts(run), 1)
        self.assertEqual(run["stdout"].read_text(), "fake-child-stdout:success\n")
        self.assertEqual(run["stderr"].read_text(), "fake-child-stderr:success\n")
        self.assert_complete_terminal(
            run, record, child_outcome="success", exit_code=0, terminating_signal=None
        )

    def test_02_nonzero_exit_is_terminally_sealed(self) -> None:
        run = self.make_run("nonzero", "nonzero")
        record = supervisor.run_once(run["package_path"], run["authority_path"])
        self.assertEqual(record["outcome"], "child_nonzero_exit")
        self.assertEqual(self.starts(run), 1)
        self.assert_complete_terminal(
            run, record, child_outcome="nonzero_exit", exit_code=7, terminating_signal=None
        )

    def test_03_signal_is_terminally_sealed(self) -> None:
        run = self.make_run("signal", "signal")
        record = supervisor.run_once(run["package_path"], run["authority_path"])
        self.assertEqual(record["outcome"], "child_signaled")
        self.assertEqual(self.starts(run), 1)
        self.assert_complete_terminal(
            run,
            record,
            child_outcome="signal",
            exit_code=None,
            terminating_signal=signal.SIGTERM,
        )

    def test_04_metadata_serialization_failure_never_abandons_child(self) -> None:
        run = self.make_run("metadata-failure", "success")
        with mock.patch.object(
            supervisor,
            "_write_started_metadata",
            side_effect=TypeError("injected JSON metadata serialization failure"),
        ):
            record = supervisor.run_once(run["package_path"], run["authority_path"])
        self.assertEqual(record["outcome"], "supervisor_post_start_error")
        self.assertEqual(record["process"]["child_outcome"], "success")
        self.assertEqual(self.starts(run), 1)
        self.assertEqual(
            record["post_start_errors"],
            [
                {
                    "stage": "process_started_metadata",
                    "type": "TypeError",
                    "message": "injected JSON metadata serialization failure",
                }
            ],
        )
        self.assert_complete_terminal(
            run, record, child_outcome="success", exit_code=0, terminating_signal=None
        )

    def test_05_duplicate_restart_is_rejected_by_durable_intent(self) -> None:
        run = self.make_run("duplicate", "success")
        first = supervisor.run_once(run["package_path"], run["authority_path"])
        with self.assertRaisesRegex(supervisor.DuplicateExecution, "pre-start intent"):
            supervisor.run_once(run["package_path"], run["authority_path"])
        self.assertEqual(self.starts(run), 1)
        self.assert_complete_terminal(
            run, first, child_outcome="success", exit_code=0, terminating_signal=None
        )

    def test_06_exact_argv_mismatch_is_zero_start(self) -> None:
        run = self.make_run("argv-mismatch", "success")
        authority = json.loads(run["authority_path"].read_text(encoding="utf-8"))
        authority["bindings"]["argv"][2] = "nonzero"
        authority["bindings"]["argv_sha256"] = supervisor.canonical_value_sha256(
            authority["bindings"]["argv"]
        )
        write_json(run["authority_path"], authority)
        with self.assertRaisesRegex(supervisor.SupervisorError, "bindings mismatch package"):
            supervisor.run_once(run["package_path"], run["authority_path"])
        self.assertEqual(self.starts(run), 0)
        self.assertFalse(run["state_dir"].exists())
        self.assertFalse(run["terminal"].exists())

    def test_07_preexisting_output_is_zero_start(self) -> None:
        run = self.make_run("preexisting-output", "success")
        run["stdout"].write_text("must-not-overwrite\n", encoding="utf-8")
        with self.assertRaisesRegex(supervisor.SupervisorError, "required output already exists"):
            supervisor.run_once(run["package_path"], run["authority_path"])
        self.assertEqual(self.starts(run), 0)
        self.assertEqual(run["stdout"].read_text(), "must-not-overwrite\n")
        self.assertFalse(run["state_dir"].exists())
        self.assertFalse(run["terminal"].exists())

    def test_08_source_tree_digest_preflight_failure_is_zero_start(self) -> None:
        run = self.make_run("hash-failure", "success")
        self.tree_context.write_text("tree-b\n", encoding="utf-8")
        with self.assertRaisesRegex(supervisor.SupervisorError, "source_tree SHA-256 mismatch"):
            supervisor.run_once(run["package_path"], run["authority_path"])
        self.assertEqual(self.starts(run), 0)
        self.assertFalse(run["state_dir"].exists())
        self.assertFalse(run["terminal"].exists())

    def test_09_exact_python310_import_pycompile_and_help(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        imported = subprocess.run(
            [
                str(EXACT_PYTHON),
                "-c",
                (
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "import ace2_exact_once_runtime_supervisor; "
                    "print('.'.join(str(part) for part in sys.version_info[:3]))"
                ),
                str(TOOLS),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(imported.returncode, 0, imported.stderr)
        self.assertEqual(imported.stdout.strip(), "3.10.12")

        compiled_path = self.root / "ace2_exact_once_runtime_supervisor.pyc"
        compiled = subprocess.run(
            [
                str(EXACT_PYTHON),
                "-c",
                (
                    "import py_compile, sys; "
                    "py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)"
                ),
                str(Path(supervisor.__file__).resolve()),
                str(compiled_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        self.assertTrue(compiled_path.is_file())

        help_result = subprocess.run(
            [str(EXACT_PYTHON), str(Path(supervisor.__file__).resolve()), "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--package", help_result.stdout)
        self.assertIn("--authority", help_result.stdout)

    def test_10_disposable_runtime_topology_validation_is_nonconsuming(self) -> None:
        run = self.make_run("validation-only", "success")
        required_outputs, runtime = supervisor.prepare_runtime_bindings(
            source_root=self.source,
            state_dir=run["state_dir"],
            stdout_path=run["stdout"],
            stderr_path=run["stderr"],
            terminal_record_path=run["terminal"],
            additional_outputs=(run["counter"],),
        )
        self.assertEqual(
            required_outputs,
            run["package"]["bindings"]["required_absent_outputs"],
        )
        self.assertEqual(runtime, run["authority"]["runtime"])
        self.assertNotIn(str(run["state_dir"]), required_outputs)
        with self.assertRaisesRegex(
            supervisor.SupervisorError,
            "required output must be outside state_dir",
        ):
            supervisor.prepare_runtime_bindings(
                source_root=self.source,
                state_dir=run["state_dir"],
                stdout_path=run["stdout"],
                stderr_path=run["stderr"],
                terminal_record_path=run["terminal"],
                additional_outputs=(run["state_dir"],),
            )
        validated = supervisor.validate_run(run["package_path"], run["authority_path"])
        self.assertEqual(validated.state_dir, run["state_dir"])
        self.assertEqual(self.starts(run), 0)
        self.assertFalse(run["state_dir"].exists())
        for path in (run["counter"], run["stdout"], run["stderr"], run["terminal"]):
            self.assertFalse(os.path.lexists(path))

    def test_11_concurrent_cli_attempts_start_exactly_one_fake_child(self) -> None:
        run = self.make_run("concurrent", "success")
        command = [
            str(EXACT_PYTHON),
            str(Path(supervisor.__file__).resolve()),
            "--package",
            str(run["package_path"]),
            "--authority",
            str(run["authority_path"]),
        ]
        processes = [
            subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for _ in range(2)
        ]
        results = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            results.append((process.returncode, stdout, stderr))

        self.assertEqual(sum(result[0] == 0 for result in results), 1, results)
        self.assertEqual(sum(result[0] in (20, 21) for result in results), 1, results)
        winner = next(result for result in results if result[0] == 0)
        rejected = next(result for result in results if result[0] in (20, 21))
        self.assertEqual(json.loads(winner[1])["process_start_count"], 1)
        rejected_summary = json.loads(rejected[2])
        self.assertIn(
            rejected_summary["status"],
            (supervisor.SupervisorError.category, supervisor.DuplicateExecution.category),
        )
        self.assertEqual(rejected_summary["process_start_count"], 0)
        self.assertEqual(self.starts(run), 1)

        record = json.loads(run["terminal"].read_text(encoding="utf-8"))
        self.assert_complete_terminal(
            run, record, child_outcome="success", exit_code=0, terminating_signal=None
        )

    def test_12_optional_phase_binding_is_strict_and_nonconsuming(self) -> None:
        run = self.make_run("phase-binding", "success")
        certificate = self.config / "phase-binding.certificate.json"
        regression = self.config / "phase-binding.regression.json"
        write_json(certificate, {"binding_core": {"attempt": "test"}})
        write_json(regression, {"status": "PASS"})
        envelope = {"argv_policy": "freeze-hash-at-launch", "cwd": str(self.source)}
        phase_binding = {
            "certificate": supervisor.file_binding(certificate),
            "binding_core_sha256": supervisor.canonical_value_sha256(
                {"attempt": "test"}
            ),
            "execution_envelope": envelope,
            "execution_envelope_sha256": supervisor.canonical_value_sha256(envelope),
            "phase_transition_regression": supervisor.file_binding(regression),
        }
        for record in (run["package"], run["authority"]):
            record["bindings"]["bound_files"].extend(
                [
                    supervisor.file_binding(certificate),
                    supervisor.file_binding(regression),
                ]
            )
            record["bindings"]["phase_binding"] = copy.deepcopy(phase_binding)
        write_json(run["package_path"], run["package"])
        package_measurement = supervisor.measure_file(run["package_path"])
        run["authority"]["package"]["byte_count"] = package_measurement.byte_count
        run["authority"]["package"]["sha256"] = package_measurement.sha256
        write_json(run["authority_path"], run["authority"])

        supervisor.validate_run(run["package_path"], run["authority_path"])
        self.assertEqual(self.starts(run), 0)
        self.assertFalse(run["state_dir"].exists())

        run["package"]["bindings"]["phase_binding"]["execution_envelope_sha256"] = "0" * 64
        run["authority"]["bindings"]["phase_binding"]["execution_envelope_sha256"] = "0" * 64
        write_json(run["package_path"], run["package"])
        package_measurement = supervisor.measure_file(run["package_path"])
        run["authority"]["package"]["byte_count"] = package_measurement.byte_count
        run["authority"]["package"]["sha256"] = package_measurement.sha256
        write_json(run["authority_path"], run["authority"])
        with self.assertRaisesRegex(supervisor.SupervisorError, "envelope digest mismatch"):
            supervisor.validate_run(run["package_path"], run["authority_path"])
        self.assertEqual(self.starts(run), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
