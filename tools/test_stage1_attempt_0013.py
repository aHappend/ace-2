from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    loaded = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loaded)
    return loaded


validator = load_module("_attempt0013_validator", PACKAGE / "validate_package.py")
runner = load_module("_attempt0013_runner", PACKAGE / "terminal_runner.py")


class Attempt0013CandidateTests(unittest.TestCase):
    def test_package_native_validation(self) -> None:
        validator.validate()

    def test_capacity_gate_accepts_exact_threshold_and_rejects_one_byte_below(
        self,
    ) -> None:
        required = runner.CAPACITY["required_available_bytes"]
        runner.check_capacity(available_bytes=required)
        with self.assertRaisesRegex(runner.GateError, "insufficient capacity"):
            runner.check_capacity(available_bytes=required - 1)

    def test_fresh_fixture_terminal_is_attempt_0013_bound(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".attempt0013-focused-", dir=PACKAGE.parent
        ) as temporary:
            state = Path(temporary) / "state"
            self.assertEqual(runner.execute_fixture("nonzero", state), 7)
            terminal = json.loads(
                (state / "terminal-status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(terminal["status"], "SEALED_TERMINAL_NO_RETRY")
            self.assertEqual(terminal["classification"], "FIXTURE_NONZERO")
            self.assertEqual(
                terminal["consumed_identity"]["package_run_identity"],
                runner.CONTRACT["package_run_identity"],
            )
            self.assertEqual(
                terminal["attempt_0013_retry_replay_resume_relaunch"],
                "FORBIDDEN",
            )
            self.assertNotIn(
                "attempt_0012_retry_replay_resume_relaunch", terminal
            )
            self.assertTrue(terminal["consumption_remains_permanent"])
            self.assertFalse(terminal["second_model_invocation_permitted"])

    def test_complete_run_bound_mutations_fail_closed(self) -> None:
        proof = json.loads(
            (PACKAGE / "capacity.json").read_text(encoding="utf-8")
        )
        proof["safety_margin_bytes"] -= 1
        with self.assertRaisesRegex(
            validator.ValidationError,
            "corrected complete-run capacity arithmetic changed",
        ):
            validator.validate_capacity(proof)


if __name__ == "__main__":
    unittest.main()
