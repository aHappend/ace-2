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


runner = load_module("_attempt0014_r5_runner", PACKAGE / "terminal_runner.py")
regression = load_module(
    "_attempt0014_r5_regression", PACKAGE / "lifecycle_regression.py"
)


class Attempt0014R5Tests(unittest.TestCase):
    def test_inert_24_layer_lifecycle_and_negative_controls(self) -> None:
        result = regression.run_all(runner)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["layer_ids"], list(range(24)))
        self.assertTrue(result["create_use_prune_order_exact"])
        self.assertFalse(result["model_executed"])
        self.assertFalse(result["rtl_executed"])
        self.assertEqual(
            result["legacy_failure_reproduction"]["exception_type"],
            "FileNotFoundError",
        )
        self.assertEqual(
            result["negative_controls"]["premature_prune"]["status"],
            "REJECTED",
        )
        self.assertEqual(
            result["negative_controls"]["missing_generated_input"]["status"],
            "REJECTED",
        )
        self.assertEqual(
            result["negative_controls"]["exception_record_publication_failure"][
                "status"
            ],
            "PASS_FALLBACK_DURABLE_BEFORE_TERMINAL",
        )

    def test_exception_is_published_before_terminal_seal(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".attempt0014-r5-exception-", dir=PACKAGE.parent
        ) as temporary:
            state = Path(temporary) / "state"
            output = Path(temporary) / "output"
            missing = Path(temporary) / "missing-program"
            state.mkdir()
            exit_code = runner.run_lifecycle(
                [missing.as_posix()],
                state=state,
                output=output,
                prefix="INERT",
            )
            self.assertEqual(exit_code, 2)
            terminal = json.loads(
                (state / "terminal-status.json").read_text(encoding="utf-8")
            )
            binding = terminal["exception_record"]
            record_path = state / binding["path"]
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["exception_type"], "FileNotFoundError")
            self.assertEqual(record["phase"], "child_spawn")
            self.assertEqual(
                record["missing_path"],
                missing.resolve().relative_to(runner.ROOT).as_posix(),
            )
            self.assertTrue(record["traceback"]["frames"])
            self.assertEqual(runner.sha256_file(record_path), binding["sha256"])
            self.assertLessEqual(
                record_path.stat().st_mtime_ns,
                (state / "terminal-status.json").stat().st_mtime_ns,
            )


if __name__ == "__main__":
    unittest.main()
