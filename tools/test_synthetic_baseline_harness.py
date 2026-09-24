#!/usr/bin/env python3
"""Synthetic-only tests for all seven baseline-harness hardening properties."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
HARNESS = TOOLS / "synthetic_baseline_harness.py"
sys.path.insert(0, str(TOOLS))

from synthetic_baseline_harness import (  # noqa: E402
    ARTIFACT_NAME,
    COMPANION_NAME,
    ENTRYPOINT_KIND,
    OPERATION,
    REQUIRED_STAGE,
    sha256_file,
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class SyntheticBaselineHarnessTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ace2-synthetic-harness-")
        self.root = Path(self.temporary.name)
        self.fixture_rel = Path("fixtures/input.json")
        self.authority_rel = Path("fixtures/authority.json")
        self.fixture_path = self.root / self.fixture_rel
        self.authority_path = self.root / self.authority_rel
        self.ledger = self.root / "state/run-ledger.json"
        self.output = self.root / "published"
        write_json(
            self.fixture_path,
            {
                "schema_version": 1,
                "payload": {
                    "kind": "dry-synthetic-fixture",
                    "samples": [3, 1, 4, 1, 5],
                },
            },
        )
        self.write_authority()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_authority(
        self,
        *,
        stage: str = REQUIRED_STAGE,
        authorized: bool = True,
        dry_fixture_only: bool = True,
        candidate_or_model_execution_permitted: bool = False,
    ) -> None:
        write_json(
            self.authority_path,
            {
                "schema_version": 1,
                "current_stage": stage,
                "synthetic_harness_execution_authorized": authorized,
                "dry_fixture_only": dry_fixture_only,
                "candidate_or_model_execution_permitted": candidate_or_model_execution_permitted,
            },
        )

    def binding(self, path: Path, relative: Path) -> dict[str, Any]:
        return {
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    def make_spec(self, run_id: str) -> Path:
        spec = {
            "schema_version": 1,
            "run_id": run_id,
            "entrypoint_kind": ENTRYPOINT_KIND,
            "operation": OPERATION,
            "fixture": self.binding(self.fixture_path, self.fixture_rel),
            "authority": {
                **self.binding(self.authority_path, self.authority_rel),
                "required_stage": REQUIRED_STAGE,
            },
            "provenance": {
                "contract_id": "ace2-synthetic-baseline-harness-test-v1",
                "harness_sha256": sha256_file(HARNESS),
            },
        }
        path = self.root / "specs" / f"{run_id}.json"
        write_json(path, spec)
        return path

    def command(
        self,
        action: str,
        *,
        spec: Path | None = None,
        run_id: str | None = None,
        failpoint: str | None = None,
    ) -> list[str]:
        command = [
            sys.executable,
            str(HARNESS),
            "--synthetic-root",
            str(self.root),
            "--ledger",
            str(self.ledger),
            "--output-root",
            str(self.output),
            action,
        ]
        if spec is not None:
            command.extend(["--spec", str(spec)])
        if run_id is not None:
            command.extend(["--run-id", run_id])
        if failpoint is not None:
            command.extend(["--failpoint", failpoint])
        return command

    def invoke(
        self,
        action: str,
        *,
        spec: Path | None = None,
        run_id: str | None = None,
        failpoint: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["ACE2_SYNTHETIC_HARNESS_TEST_MODE"] = "1"
        return subprocess.run(
            self.command(action, spec=spec, run_id=run_id, failpoint=failpoint),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def load_ledger(self) -> dict[str, Any]:
        return json.loads(self.ledger.read_text(encoding="utf-8"))

    def bundle(self, run_id: str) -> Path:
        return self.output / f"{run_id}.bundle"

    def test_01_atomic_artifact_and_companion_hash_commit(self) -> None:
        interrupted = self.make_spec("atomic-interrupted")
        result = self.invoke("run", spec=interrupted, failpoint="before_bundle_publish")
        self.assertEqual(result.returncode, 97, result.stderr)
        self.assertFalse(self.bundle("atomic-interrupted").exists())
        recovery = self.invoke("recover", spec=interrupted)
        self.assertEqual(recovery.returncode, 2)
        self.assertEqual(
            self.load_ledger()["runs"]["atomic-interrupted"]["state"],
            "interrupted_fail_closed",
        )

        successful = self.make_spec("atomic-success")
        result = self.invoke("run", spec=successful)
        self.assertEqual(result.returncode, 0, result.stderr)
        bundle = self.bundle("atomic-success")
        self.assertEqual({path.name for path in bundle.iterdir()}, {ARTIFACT_NAME, COMPANION_NAME})
        fields = (bundle / COMPANION_NAME).read_text(encoding="ascii").split()
        self.assertEqual(fields, [sha256_file(bundle / ARTIFACT_NAME), ARTIFACT_NAME])

    def test_02_exact_provenance_binding(self) -> None:
        spec = self.make_spec("provenance-ok")
        expected_spec_sha256 = sha256_file(spec)
        expected_fixture_sha256 = sha256_file(self.fixture_path)
        expected_authority_sha256 = sha256_file(self.authority_path)
        result = self.invoke("run", spec=spec)
        self.assertEqual(result.returncode, 0, result.stderr)
        artifact = json.loads(
            (self.bundle("provenance-ok") / ARTIFACT_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(
            artifact["provenance"],
            {
                "contract_id": "ace2-synthetic-baseline-harness-test-v1",
                "spec_sha256": expected_spec_sha256,
                "fixture_sha256": expected_fixture_sha256,
                "authority_sha256": expected_authority_sha256,
                "harness_sha256": sha256_file(HARNESS),
                "authorized_stage": REQUIRED_STAGE,
            },
        )

        stale = self.make_spec("provenance-stale")
        write_json(self.fixture_path, {"schema_version": 1, "payload": {"changed": True}})
        rejected = self.invoke("run", spec=stale)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("fixture byte count mismatch", rejected.stderr)
        self.assertNotIn("provenance-stale", self.load_ledger()["runs"])

    def test_03_durable_exactly_one_reservation_and_run_ledger(self) -> None:
        spec = self.make_spec("exactly-one")
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["ACE2_SYNTHETIC_HARNESS_TEST_MODE"] = "1"
        command = self.command("run", spec=spec)
        first = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
        second = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
        first_output = first.communicate(timeout=15)
        second_output = second.communicate(timeout=15)
        self.assertEqual(sorted([first.returncode, second.returncode]), [0, 2], (first_output, second_output))
        record = self.load_ledger()["runs"]["exactly-one"]
        self.assertEqual(record["state"], "committed")
        self.assertEqual(record["execution_count"], 1)
        retry = self.invoke("run", spec=spec)
        self.assertEqual(retry.returncode, 2)
        self.assertEqual(self.load_ledger()["runs"]["exactly-one"]["execution_count"], 1)

    def test_04_stage_authorization_before_execution(self) -> None:
        self.write_authority(stage="rtl")
        wrong_stage = self.make_spec("stage-wrong")
        result = self.invoke("run", spec=wrong_stage)
        self.assertEqual(result.returncode, 2)
        self.assertIn("stage authorization denied", result.stderr)
        self.assertFalse(self.ledger.exists())

        self.write_authority(candidate_or_model_execution_permitted=True)
        unsafe_authority = self.make_spec("stage-unsafe-authority")
        result = self.invoke("run", spec=unsafe_authority)
        self.assertEqual(result.returncode, 2)
        self.assertIn("authority permits candidate/model execution", result.stderr)
        self.assertFalse(self.ledger.exists())

        self.write_authority()
        authorized = self.make_spec("stage-authorized")
        result = self.invoke("run", spec=authorized)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.load_ledger()["runs"]["stage-authorized"]["execution_count"], 1)

    def test_05_candidate_entrypoint_interlock(self) -> None:
        unknown_command = self.make_spec("interlock-command")
        value = json.loads(unknown_command.read_text(encoding="utf-8"))
        value["command"] = ["python", "candidate.py"]
        write_json(unknown_command, value)
        result = self.invoke("run", spec=unknown_command)
        self.assertEqual(result.returncode, 2)
        self.assertIn("run spec keys must be exactly", result.stderr)

        candidate_kind = self.make_spec("interlock-kind")
        value = json.loads(candidate_kind.read_text(encoding="utf-8"))
        value["entrypoint_kind"] = "candidate_model_v1"
        write_json(candidate_kind, value)
        result = self.invoke("run", spec=candidate_kind)
        self.assertEqual(result.returncode, 2)
        self.assertIn("candidate/unknown entrypoint rejected", result.stderr)

        protected_path = self.make_spec("interlock-path")
        value = json.loads(protected_path.read_text(encoding="utf-8"))
        value["fixture"] = {
            "path": "rtl/candidate_model.json",
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
        write_json(protected_path, value)
        result = self.invoke("run", spec=protected_path)
        self.assertEqual(result.returncode, 2)
        self.assertIn("protected/candidate surface", result.stderr)
        self.assertFalse(self.ledger.exists())

    def test_06_crash_recovery(self) -> None:
        pre_execution = self.make_spec("crash-reservation")
        result = self.invoke("run", spec=pre_execution, failpoint="after_reservation")
        self.assertEqual(result.returncode, 97)
        self.assertEqual(self.load_ledger()["runs"]["crash-reservation"]["execution_count"], 0)
        recovery = self.invoke("recover", spec=pre_execution)
        self.assertEqual(recovery.returncode, 0, recovery.stderr)
        record = self.load_ledger()["runs"]["crash-reservation"]
        self.assertEqual(record["state"], "committed")
        self.assertEqual(record["execution_count"], 1)

        post_publish = self.make_spec("crash-published")
        result = self.invoke("run", spec=post_publish, failpoint="after_bundle_publish")
        self.assertEqual(result.returncode, 97)
        artifact_sha256 = sha256_file(self.bundle("crash-published") / ARTIFACT_NAME)
        record = self.load_ledger()["runs"]["crash-published"]
        self.assertEqual(record["state"], "running")
        self.assertEqual(record["execution_count"], 1)
        recovery = self.invoke("recover", spec=post_publish)
        self.assertEqual(recovery.returncode, 0, recovery.stderr)
        record = self.load_ledger()["runs"]["crash-published"]
        self.assertTrue(record["recovered_after_crash"])
        self.assertEqual(record["execution_count"], 1)
        self.assertEqual(record["artifact_sha256"], artifact_sha256)

    def test_07_missing_or_mismatched_artifact_fail_closed(self) -> None:
        missing_companion = self.make_spec("missing-companion")
        result = self.invoke("run", spec=missing_companion)
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.bundle("missing-companion") / COMPANION_NAME).unlink()
        verification = self.invoke("verify", run_id="missing-companion")
        self.assertEqual(verification.returncode, 2)
        self.assertEqual(self.load_ledger()["runs"]["missing-companion"]["execution_count"], 1)

        mismatched_artifact = self.make_spec("mismatched-artifact")
        result = self.invoke("run", spec=mismatched_artifact)
        self.assertEqual(result.returncode, 0, result.stderr)
        artifact_path = self.bundle("mismatched-artifact") / ARTIFACT_NAME
        artifact_path.write_text("{}\n", encoding="utf-8")
        verification = self.invoke("verify", run_id="mismatched-artifact")
        self.assertEqual(verification.returncode, 2)
        self.assertIn("artifact companion hash mismatch", verification.stderr)
        self.assertEqual(self.load_ledger()["runs"]["mismatched-artifact"]["execution_count"], 1)

        missing_fixture = self.make_spec("missing-fixture")
        self.fixture_path.unlink()
        result = self.invoke("run", spec=missing_fixture)
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing fixture", result.stderr)
        self.assertNotIn("missing-fixture", self.load_ledger()["runs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
