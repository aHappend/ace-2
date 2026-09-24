from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PACKAGE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    loaded = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loaded)
    return loaded


runner = load_module("_repair0006_runner", PACKAGE / "continuation_runner.py")
launcher = load_module("_repair0006_launcher", PACKAGE / "atomic_launcher.py")


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, key: str, generated: Path) -> dict[str, object]:
        self.calls.append(key)
        return {
            "output": (key + ":" + runner.sha256_file(generated)).encode("ascii"),
            "validation_status": "PASS_RTL_REFERENCE_AUTHENTICATED",
            "evidence": {
                "rtl_reference_agreement": True,
                "test_double": True,
            },
        }


class Repair0006NoExecutionTests(unittest.TestCase):
    def make_runtime(self, temporary: str) -> Path:
        root = Path(temporary) / "runtime"
        root.mkdir()
        shutil.copyfile(
            PACKAGE / "checkpoint-manifest.json",
            root / "checkpoint-manifest.json",
        )
        shutil.copytree(PACKAGE / "salvage-state", root / "salvage-state")
        return root / "checkpoint-manifest.json"

    def test_only_layers_11_through_23_are_selected_and_prefix_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".ace2-r6-clean-") as temporary:
            checkpoint_path = self.make_runtime(temporary)
            before = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            prefix = copy.deepcopy(before["units"][:11])
            executor = FakeExecutor()
            result = runner.run_continuation(checkpoint_path, executor)
            self.assertEqual(
                executor.calls,
                [f"position-00/layer-{index:02d}" for index in range(11, 24)],
            )
            self.assertEqual(result["units"][:11], prefix)
            self.assertEqual(result["next_incomplete_unit"], None)

    def test_crash_after_publication_adopts_without_reexecution(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".ace2-r6-after-") as temporary:
            checkpoint_path = self.make_runtime(temporary)
            first = FakeExecutor()
            with self.assertRaises(runner.SimulatedCrash):
                runner.run_continuation(
                    checkpoint_path,
                    first,
                    fault="crash_after_publication:position-00/layer-11",
                )
            second = FakeExecutor()
            runner.run_continuation(checkpoint_path, second)
            self.assertEqual(first.calls, ["position-00/layer-11"])
            self.assertNotIn("position-00/layer-11", second.calls)

    def test_crash_before_publication_retries_incomplete_unit(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".ace2-r6-before-") as temporary:
            checkpoint_path = self.make_runtime(temporary)
            first = FakeExecutor()
            with self.assertRaises(runner.SimulatedCrash):
                runner.run_continuation(
                    checkpoint_path,
                    first,
                    fault="crash_before_publication:position-00/layer-11",
                )
            second = FakeExecutor()
            runner.run_continuation(checkpoint_path, second)
            self.assertEqual(first.calls, ["position-00/layer-11"])
            self.assertEqual(second.calls[0], "position-00/layer-11")

    def test_corrupt_checkpoint_and_missing_dependency_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".ace2-r6-corrupt-") as temporary:
            checkpoint_path = self.make_runtime(temporary)
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["units"][0]["output"]["sha256"] = "0" * 64
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            executor = FakeExecutor()
            with self.assertRaises(runner.CheckpointError):
                runner.run_continuation(checkpoint_path, executor)
            self.assertEqual(executor.calls, [])
        with tempfile.TemporaryDirectory(prefix=".ace2-r6-missing-") as temporary:
            checkpoint_path = self.make_runtime(temporary)
            (checkpoint_path.parent / "salvage-state/layer-10-hidden-s8.bin").unlink()
            executor = FakeExecutor()
            with self.assertRaises(runner.CheckpointError):
                runner.run_continuation(checkpoint_path, executor)
            self.assertEqual(executor.calls, [])

    def test_duplicate_runner_and_authority_consumption_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".ace2-r6-duplicate-") as temporary:
            checkpoint_path = self.make_runtime(temporary)
            with runner.hold_runner_lock(checkpoint_path):
                with self.assertRaises(runner.DuplicateStartError):
                    runner.run_continuation(checkpoint_path, FakeExecutor())
            consumption = Path(temporary) / "consumption"
            record = {
                "schema": "test-only-consumption",
                "authority_identity": "test-only",
                "authority_sha256": "1" * 64,
                "package_identity": launcher.IDENTITY,
                "package_tree_root_sha256": "2" * 64,
            }
            self.assertEqual(
                launcher._atomic_consume(consumption, record, recover=False),
                record,
            )
            with self.assertRaises(launcher.DuplicateConsumptionError):
                launcher._atomic_consume(consumption, record, recover=False)
            self.assertEqual(
                launcher._atomic_consume(consumption, record, recover=True),
                record,
            )

    def test_external_executor_substitution_breaks_package_seal(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".ace2-r6-tamper-") as temporary:
            copied = Path(temporary) / "package"
            shutil.copytree(PACKAGE, copied)
            executor = copied / "production_unit_executor.py"
            executor.chmod(0o644)
            executor.write_bytes(executor.read_bytes() + b"\n")
            with self.assertRaises(launcher.LaunchError):
                launcher.validate_package(copied)

    def test_authority_cannot_be_reused_through_alternate_namespaces(self) -> None:
        manifest = {
            "future_namespaces": {
                "consumption": (
                    launcher.CANONICAL_CONSUMPTION_NAMESPACE.as_posix()
                ),
                "runtime": launcher.CANONICAL_RUNTIME_NAMESPACE.as_posix(),
            }
        }
        canonical_consumption = (
            launcher.ROOT / launcher.CANONICAL_CONSUMPTION_NAMESPACE
        )
        canonical_runtime = launcher.ROOT / launcher.CANONICAL_RUNTIME_NAMESPACE
        alternate_consumption = canonical_consumption.with_name(
            canonical_consumption.name + "-alternate"
        )
        alternate_runtime = canonical_runtime.with_name(
            canonical_runtime.name + "-alternate"
        )
        authority = {"authority_identity": "test-only-authority"}
        patches = (
            mock.patch.object(
                launcher,
                "validate_package",
                return_value=(manifest, "2" * 64),
            ),
            mock.patch.object(launcher, "validate_external_sources"),
            mock.patch.object(launcher, "validate_pending_candidate"),
            mock.patch.object(
                launcher,
                "validate_review_and_authority",
                return_value=(authority, "1" * 64),
            ),
            mock.patch.object(
                launcher,
                "_atomic_consume",
                side_effect=RuntimeError("stop before consumption"),
            ),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4] as consume:
            with self.assertRaisesRegex(RuntimeError, "stop before consumption"):
                launcher.execute(
                    Path("review"),
                    Path("authority"),
                    canonical_consumption,
                    canonical_runtime,
                    recover=False,
                )
            consume.assert_called_once()
            consume.reset_mock()
            for consumption, runtime in (
                (alternate_consumption, canonical_runtime),
                (canonical_consumption, alternate_runtime),
            ):
                with self.assertRaises(launcher.LaunchError):
                    launcher.execute(
                        Path("review"),
                        Path("authority"),
                        consumption,
                        runtime,
                        recover=False,
                    )
            consume.assert_not_called()


if __name__ == "__main__":
    unittest.main()
