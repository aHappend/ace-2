#!/usr/bin/env python3
"""Focused tests for the verification-only Dynamic Scale32 baseline runner."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from run_dynamic_scale32_verification_baseline import (  # noqa: E402
    AUTHORITY_KIND,
    AUTHORITY_REL,
    CONTRACT_ID,
    ENVIRONMENT_KEYS,
    LAUNCH_ENVIRONMENT_NAME,
    MANAGER_TRANSITION_AT,
    REQUIRED_OUTPUTS,
    DynamicScale32BaselineRunner,
    BaselineError,
    canonical_integrity,
    sha256_file,
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_companion(path: Path) -> None:
    path.with_suffix(".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n",
        encoding="ascii",
    )


class DynamicScale32BaselineRunnerTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ace2-scale32-baseline-")
        self.root = Path(self.temporary.name)
        self.seeds = {
            "datasets_seed": 20260729,
            "numpy_seed": 20260729,
            "python_seed": 20260729,
            "torch_deterministic_algorithms": True,
            "torch_seed": 20260729,
        }
        self.software = {"pip": importlib.metadata.version("pip")}
        self.model = {
            "repository": "Qwen/Qwen2.5-0.5B",
            "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        }
        self._write_contract_fixture()
        self.runner = DynamicScale32BaselineRunner(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_contract_fixture(self) -> None:
        target_tools = self.root / "tools"
        target_tools.mkdir(parents=True, exist_ok=True)
        for name in (
            "run_dynamic_scale32_verification_baseline.py",
            "ace2_software_identity.py",
        ):
            shutil.copy2(TOOLS / name, target_tools / name)
        self._write_safe_executor()

        pipeline = {
            "current_stage": "verification",
            "stage_history": [
                {
                    "at": MANAGER_TRANSITION_AT,
                    "by": "manager",
                    "direction": "advance",
                    "from_stage": "rtl",
                    "to_stage": "verification",
                    "reason": "test fixture",
                }
            ],
        }
        write_json(self.root / "research/PIPELINE_STATE.json", pipeline)

        review = {
            "contract_id": CONTRACT_ID,
            "stage": "rtl",
            "stage_closing": True,
            "verdict": "go",
            "status": "review_complete_stage_closing_true",
            "integrity": {
                "algorithm": "sha256-canonical-json-v1",
                "canonical_sha256": None,
            },
        }
        review["integrity"]["canonical_sha256"] = canonical_integrity(review)
        review_path = (
            self.root
            / "evidence/review/rtl_stage_closing_shared_token_group_dynamic_scale32_v1/decision-v2.json"
        )
        write_json(review_path, review)
        write_companion(review_path)

        proposal_path = (
            self.root
            / "evidence/shared_token_group_dynamic_scale32_v1/architecture/PROPOSAL.json"
        )
        write_json(proposal_path, {"contract_id": CONTRACT_ID})
        write_companion(proposal_path)
        write_json(
            self.root
            / "evidence/shared_token_group_dynamic_scale32_v1/architecture/MANAGER_FREEZE.json",
            {"contract_id": CONTRACT_ID},
        )

        write_json(
            self.root / "benchmark/quality/PROMPT_MANIFEST.json",
            {
                "model": self.model,
                "datasets": {
                    "wikitext2": {"revision": "wiki-test"},
                    "c4_en_512": {"revision": "c4-test"},
                },
            },
        )
        write_json(
            self.root / "benchmark/quality/QUALITY_CONFIG.json",
            {
                "baseline": {
                    "model_repository": self.model["repository"],
                    "model_revision": self.model["revision"],
                },
                "determinism": self.seeds,
                "software": self.software,
            },
        )
        requirements = self.root / "benchmark/quality/requirements.txt"
        requirements.parent.mkdir(parents=True, exist_ok=True)
        requirements.write_text(f"pip=={self.software['pip']}\n", encoding="utf-8")

    def _write_safe_executor(
        self,
        *,
        exit_status: int = 0,
        create_outputs: bool = True,
        sleep_seconds: int = 0,
        extra_output: bool = False,
    ) -> None:
        source = f'''#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

PROMPT = json.loads(Path("benchmark/quality/PROMPT_MANIFEST.json").read_text())
QUALITY = json.loads(Path("benchmark/quality/QUALITY_CONFIG.json").read_text())
MODEL = PROMPT["model"]
SEEDS = QUALITY["determinism"]
SOFTWARE = QUALITY["software"]
OUTPUT_NAMES = {list(REQUIRED_OUTPUTS)!r}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["smoke", "official"], required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--smoke-token-limit", type=int)
args = parser.parse_args()
if {sleep_seconds}:
    time.sleep({sleep_seconds})
if {exit_status}:
    raise SystemExit({exit_status})
values = {{
    "ordered_layer_traces.json": {{
        "schema_version": 1,
        "contract_id": {CONTRACT_ID!r},
        "kind": "dynamic_scale32_disabled_baseline_ordered_layer_traces",
        "ordering": "dataset_then_sequence_then_forward_execution",
        "sequences": [],
    }},
    "final_outputs.json": {{
        "schema_version": 1,
        "contract_id": {CONTRACT_ID!r},
        "kind": "dynamic_scale32_disabled_baseline_final_outputs",
        "datasets": ["wikitext2", "c4_en_512"],
        "sequences": [],
    }},
    "results.json": {{
        "schema_version": 1,
        "contract_id": {CONTRACT_ID!r},
        "kind": "dynamic_scale32_disabled_baseline_results",
        "classification": "baseline_measurement_only",
        "gate_passed": False,
        "mechanism_disabled": True,
        "mode": args.mode,
        "model": MODEL,
        "metrics": {{}},
        "seeds": SEEDS,
        "runtime": {{"device": "cpu", "packages": SOFTWARE}},
    }},
    "input_observations.json": {{
        "schema_version": 1,
        "contract_id": {CONTRACT_ID!r},
        "kind": "dynamic_scale32_disabled_baseline_input_observations",
        "datasets": {{}},
        "environment": dict(os.environ),
        "environment_sha256": hashlib.sha256(
            json.dumps(dict(os.environ), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "mode_scope": {{"mode": args.mode}},
    }},
}}
if {create_outputs!r}:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in OUTPUT_NAMES:
        if name == "run_contract.json":
            continue
        (args.output_dir / name).write_text(json.dumps(values[name], sort_keys=True) + "\\n")
    artifact_hashes = {{
        name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
        for name in OUTPUT_NAMES
        if name != "run_contract.json"
    }}
    values["run_contract.json"] = {{
        "schema_version": 1,
        "contract_id": {CONTRACT_ID!r},
        "kind": "dynamic_scale32_disabled_baseline_run_contract",
        "artifact_hashes": artifact_hashes,
        "architecture_boundary": {{
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "mode": "ADVANCE",
            "ordered_supported_layer_operator_prefix": [
                "layer_0.input_rmsnorm",
                "layer_0.q_proj",
                "layer_0.k_proj",
                "layer_0.v_proj",
            ],
        }},
        "baseline_model": 1,
        "candidate_model": 0,
        "datasets": {{}},
        "determinism": SEEDS,
        "mechanism": {CONTRACT_ID!r},
        "mechanism_disabled": True,
        "mode_scope": {{"mode": args.mode}},
        "model": MODEL,
        "numerical_contract": {{}},
        "output_schema": {{"files": OUTPUT_NAMES, "schema_version": 1}},
        "software_versions": SOFTWARE,
        "source_contract_hashes": {{}},
        "status": "disabled_mechanism_baseline_complete",
    }}
    (args.output_dir / "run_contract.json").write_text(
        json.dumps(values["run_contract.json"], sort_keys=True) + "\\n"
    )
    if {extra_output!r}:
        (args.output_dir / "unexpected.json").write_text("{{}}\\n")
'''
        path = self.root / "tools/ace2_dynamic_scale32_baseline.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    def _write_authority(
        self,
        run_id: str = "dynamic-scale32-baseline-test",
        *,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        stage = self.runner._stage_authority()
        frozen = self.runner._frozen_inputs()
        scope = {
            "mode": "smoke",
            "mechanism": CONTRACT_ID,
            "mechanism_disabled": True,
            "model_repository": self.model["repository"],
            "model_revision": self.model["revision"],
            "datasets": ["wikitext2", "c4_en_512"],
            "layer_count": 24,
            "required_outputs": list(REQUIRED_OUTPUTS),
            "smoke_token_limit": 128,
        }
        authority = {
            "schema_version": 1,
            "kind": AUTHORITY_KIND,
            "contract_id": CONTRACT_ID,
            "stage": "verification",
            "run_id": run_id,
            "execution_authorized": True,
            "command": self.runner._expected_command(scope),
            "scope": scope,
            "inputs": frozen["inputs"],
            "seeds": self.seeds,
            "device": "cpu",
            "software_versions": self.software,
            "environment": frozen["launch_environment"]["variables"],
            "environment_sha256": frozen["launch_environment"]["sha256"],
            "authority_binding_sha256": stage["authority_binding_sha256"],
            "timeout_seconds": timeout_seconds,
            "model_counts": {"baseline_model": 1, "candidate_model": 0},
            "integrity": {
                "algorithm": "sha256-canonical-json-v1",
                "canonical_sha256": None,
            },
        }
        authority["integrity"]["canonical_sha256"] = canonical_integrity(authority)
        write_json(self.root / AUTHORITY_REL, authority)
        return authority

    def _ledger(self) -> dict[str, Any]:
        return json.loads(self.runner.ledger_path.read_text(encoding="utf-8"))

    def _assert_packet(self, run_id: str, event: str, label: str) -> dict[str, Any]:
        bundle = self.runner._event_bundle(run_id, event)
        verified = self.runner._verify_event_packet(bundle, label)
        self.assertFalse(verified["report"]["stage_closing"])
        return verified["report"]

    def _invoke(self, action: str, *, run_id: str | None = None, failpoint: str | None = None) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(self.root / "tools/run_dynamic_scale32_verification_baseline.py"),
            "--root",
            str(self.root),
            action,
        ]
        if run_id is not None:
            command.extend(["--run-id", run_id])
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if failpoint is not None:
            environment["ACE2_BASELINE_TEST_MODE"] = "1"
            environment["ACE2_BASELINE_TEST_FAILPOINT"] = failpoint
        return subprocess.run(command, capture_output=True, text=True, check=False, env=environment)

    def test_00_frozen_input_inventory_is_complete_without_reservation(self) -> None:
        frozen = self.runner._frozen_inputs()
        self.assertEqual(
            {record["path"] for record in frozen["inputs"]},
            {
                "benchmark/quality/PROMPT_MANIFEST.json",
                "benchmark/quality/QUALITY_CONFIG.json",
                "benchmark/quality/requirements.txt",
                "tools/ace2_dynamic_scale32_baseline.py",
                "tools/ace2_software_identity.py",
            },
        )
        self.assertFalse(self.runner.state_root.exists())
        self.assertFalse(self.runner.ledger_path.exists())
        self.assertFalse(self.runner.published_root.exists())
        self.assertFalse(self.runner.blocker_root.exists())

    def test_01_missing_command_authority_blocks_before_reservation(self) -> None:
        report = self.runner.preflight()
        codes = {item["code"] for item in report["blockers"]}
        self.assertIn("missing_manager_frozen_baseline_command_authority", codes)
        blocked = self.runner._publish_blocker(report)
        self.assertTrue(blocked["bundle"].is_dir())
        self.assertFalse(self.runner.ledger_path.exists())
        self.assertEqual(blocked["report"]["execution_counts"], {"baseline_model": 0, "candidate_model": 0})

    def test_02_candidate_surface_interlock_blocks_before_reservation(self) -> None:
        executor = self.root / "tools/ace2_dynamic_scale32_baseline.py"
        executor.write_text(
            executor.read_text(encoding="utf-8")
            + "\n# --candidate-evidence\n# --diagnostic-rope-mechanism\n",
            encoding="utf-8",
        )
        report = self.runner.preflight()
        codes = {item["code"] for item in report["blockers"]}
        self.assertIn("existing_executor_has_candidate_surface", codes)
        self.assertFalse(self.runner.ledger_path.exists())

    def test_03_exactly_once_atomic_publication_and_hash_verification(self) -> None:
        self._write_authority()
        result = self.runner.run()
        self.assertTrue(result["bundle"].is_dir())
        self.assertEqual(
            {path.name for path in result["bundle"].iterdir()},
            {
                "raw",
                "stdout.log",
                "EXECUTION_AUTHORITY.json",
                "EXECUTION_AUTHORITY.sha256",
                "RESERVATION.json",
                "RESERVATION.sha256",
                LAUNCH_ENVIRONMENT_NAME,
                Path(LAUNCH_ENVIRONMENT_NAME).with_suffix(".sha256").name,
                "reserved_inputs",
                "FINAL_RUN_LEDGER.json",
                "FINAL_RUN_LEDGER.sha256",
                "PROVENANCE.json",
                "HANDOFF.json",
                "MANIFEST.json",
                "MANIFEST.sha256",
            },
        )
        for name in (
            "EXECUTION_AUTHORITY.json",
            "RESERVATION.json",
            "FINAL_RUN_LEDGER.json",
            "MANIFEST.json",
        ):
            companion = (result["bundle"] / name).with_suffix(".sha256")
            self.assertTrue(companion.is_file())
        record = self._ledger()["runs"]["dynamic-scale32-baseline-test"]
        self.assertEqual(record["state"], "committed")
        self.assertEqual(record["execution_count"], 1)
        self.assertEqual(record["model_counts"], {"baseline_model": 1, "candidate_model": 0})
        verified = self.runner.verify("dynamic-scale32-baseline-test")
        self.assertEqual(verified["manifest_sha256"], result["manifest_sha256"])
        self.assertEqual(
            verified["final_ledger"]["runs"]["dynamic-scale32-baseline-test"],
            record,
        )
        with self.assertRaisesRegex(BaselineError, "retry forbidden"):
            self.runner.run()
        duplicate = self._assert_packet(
            "dynamic-scale32-baseline-test",
            "duplicate",
            "DUPLICATE",
        )
        self.assertEqual(duplicate["execution_count"], 1)
        self.assertEqual(
            self._ledger()["runs"]["dynamic-scale32-baseline-test"]["execution_count"],
            1,
        )

    def test_04_crash_after_reservation_never_executes_on_recovery(self) -> None:
        self._write_authority("dynamic-scale32-baseline-crash-reserved")
        result = self._invoke("run", failpoint="after_reservation")
        self.assertEqual(result.returncode, 97, result.stderr)
        record = self._ledger()["runs"]["dynamic-scale32-baseline-crash-reserved"]
        self.assertEqual(record["execution_count"], 0)
        with self.assertRaisesRegex(BaselineError, "retry forbidden"):
            self.runner.recover("dynamic-scale32-baseline-crash-reserved")
        record = self._ledger()["runs"]["dynamic-scale32-baseline-crash-reserved"]
        self.assertEqual(record["state"], "interrupted_before_execution_fail_closed")
        self.assertTrue(self.runner._state_dir("dynamic-scale32-baseline-crash-reserved").is_dir())
        packet = self._assert_packet(
            "dynamic-scale32-baseline-crash-reserved",
            "recovery",
            "RECOVERY",
        )
        self.assertEqual(packet["execution_count"], 0)

    def test_05_crash_after_publication_finalizes_without_rerun(self) -> None:
        run_id = "dynamic-scale32-baseline-crash-published"
        self._write_authority(run_id)
        result = self._invoke("run", failpoint="after_bundle_publish")
        self.assertEqual(result.returncode, 97, result.stderr)
        record = self._ledger()["runs"][run_id]
        self.assertEqual(record["state"], "running")
        self.assertEqual(record["execution_count"], 1)
        recovered = self.runner.recover(run_id)
        self.assertTrue(recovered["bundle"].is_dir())
        record = self._ledger()["runs"][run_id]
        self.assertEqual(record["state"], "committed")
        self.assertEqual(record["execution_count"], 1)
        packet = self._assert_packet(run_id, "recovery", "RECOVERY")
        self.assertEqual(packet["terminal_state"], "committed")
        self.assertEqual(recovered["final_ledger"]["runs"][run_id], record)

    def test_06_missing_or_mismatched_artifact_fails_closed(self) -> None:
        run_id = "dynamic-scale32-baseline-artifact-check"
        self._write_authority(run_id)
        result = self.runner.run()
        (result["bundle"] / "MANIFEST.sha256").unlink()
        with self.assertRaisesRegex(BaselineError, "missing companion"):
            self.runner.verify(run_id)
        self.assertEqual(self._ledger()["runs"][run_id]["execution_count"], 1)

    def test_07_failed_command_is_terminal_and_not_retried(self) -> None:
        self._write_safe_executor(exit_status=3)
        self._write_authority("dynamic-scale32-baseline-command-failure")
        with self.assertRaisesRegex(BaselineError, "failed with exit 3"):
            self.runner.run()
        record = self._ledger()["runs"]["dynamic-scale32-baseline-command-failure"]
        self.assertEqual(record["state"], "failed_fail_closed")
        self.assertEqual(record["execution_count"], 1)
        packet = self._assert_packet(
            "dynamic-scale32-baseline-command-failure",
            "terminal",
            "TERMINAL",
        )
        self.assertEqual(packet["terminal_state"], "failed_fail_closed")
        with self.assertRaisesRegex(BaselineError, "retry forbidden"):
            self.runner.run()
        self.assertEqual(
            self._ledger()["runs"]["dynamic-scale32-baseline-command-failure"]["execution_count"],
            1,
        )

    def test_08_authority_change_after_preflight_never_executes_or_reserves(self) -> None:
        self._write_authority("dynamic-scale32-baseline-authority-race")
        marker = self.root / "MUTATED_COMMAND_EXECUTED"
        original_preflight = self.runner.preflight

        def preflight_then_mutate() -> dict[str, Any]:
            report = original_preflight()
            authority_path = self.root / AUTHORITY_REL
            authority = json.loads(authority_path.read_text(encoding="utf-8"))
            authority["command"] = [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('executed')",
            ]
            authority["integrity"]["canonical_sha256"] = canonical_integrity(authority)
            write_json(authority_path, authority)
            return report

        self.runner.preflight = preflight_then_mutate  # type: ignore[method-assign]
        with self.assertRaisesRegex(BaselineError, "changed before reservation"):
            self.runner.run()
        self.assertFalse(marker.exists())
        self.assertFalse(self.runner.ledger_path.exists())
        self.assertFalse(self.runner.state_root.exists())

    def test_09_missing_raw_directory_is_terminal_and_companion_hashed(self) -> None:
        run_id = "dynamic-scale32-baseline-missing-raw"
        self._write_safe_executor(create_outputs=False)
        self._write_authority(run_id)
        with self.assertRaisesRegex(BaselineError, "output directory is missing"):
            self.runner.run()
        record = self._ledger()["runs"][run_id]
        self.assertEqual(record["state"], "validation_fail_closed")
        self.assertEqual(record["execution_count"], 1)
        packet = self._assert_packet(run_id, "terminal", "TERMINAL")
        self.assertEqual(packet["terminal_state"], "validation_fail_closed")

    def test_10_timeout_is_terminal_and_companion_hashed(self) -> None:
        run_id = "dynamic-scale32-baseline-timeout"
        self._write_safe_executor(sleep_seconds=2)
        self._write_authority(run_id, timeout_seconds=1)
        with self.assertRaisesRegex(BaselineError, "timed out"):
            self.runner.run()
        record = self._ledger()["runs"][run_id]
        self.assertEqual(record["state"], "timeout_fail_closed")
        packet = self._assert_packet(run_id, "terminal", "TERMINAL")
        self.assertEqual(packet["execution_count"], 1)

    def test_11_publication_failure_is_terminal_and_companion_hashed(self) -> None:
        run_id = "dynamic-scale32-baseline-publication-failure"
        self._write_authority(run_id)

        def fail_publication(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise BaselineError("synthetic publication failure")

        self.runner._publish_success = fail_publication  # type: ignore[method-assign]
        with self.assertRaisesRegex(BaselineError, "publication failed"):
            self.runner.run()
        record = self._ledger()["runs"][run_id]
        self.assertEqual(record["state"], "publication_fail_closed")
        packet = self._assert_packet(run_id, "terminal", "TERMINAL")
        self.assertIn("synthetic publication failure", packet["detail"])

    def test_12_different_run_id_cannot_create_second_baseline_reservation(self) -> None:
        first = "dynamic-scale32-baseline-first"
        second = "dynamic-scale32-baseline-second"
        self._write_authority(first)
        self.runner.run()
        self._write_authority(second)
        with self.assertRaisesRegex(BaselineError, "reservation already exists"):
            self.runner.run()
        ledger = self._ledger()
        self.assertEqual(set(ledger["runs"]), {first})
        self.assertEqual(ledger["runs"][first]["execution_count"], 1)
        self.assertFalse(self.runner._state_dir(second).exists())
        packet = self._assert_packet(first, "duplicate", "DUPLICATE")
        self.assertEqual(packet["attempted_run_id"], second)

    def test_13_crash_after_execution_mark_recovers_without_rerun(self) -> None:
        run_id = "dynamic-scale32-baseline-crash-running"
        self._write_authority(run_id)
        result = self._invoke("run", failpoint="after_execution_mark")
        self.assertEqual(result.returncode, 97, result.stderr)
        self.assertEqual(self._ledger()["runs"][run_id]["execution_count"], 1)
        with self.assertRaisesRegex(BaselineError, "retry forbidden"):
            self.runner.recover(run_id)
        record = self._ledger()["runs"][run_id]
        self.assertEqual(record["state"], "interrupted_after_execution_fail_closed")
        packet = self._assert_packet(run_id, "recovery", "RECOVERY")
        self.assertEqual(packet["execution_count"], 1)

    def test_14_unexpected_extra_output_fails_closed(self) -> None:
        run_id = "dynamic-scale32-baseline-extra-output"
        self._write_safe_executor(extra_output=True)
        self._write_authority(run_id)
        with self.assertRaisesRegex(BaselineError, "output set differs"):
            self.runner.run()
        record = self._ledger()["runs"][run_id]
        self.assertEqual(record["state"], "validation_fail_closed")
        self._assert_packet(run_id, "terminal", "TERMINAL")

    def test_15_orphaned_reservation_record_recovers_without_execution(self) -> None:
        run_id = "dynamic-scale32-baseline-orphaned-reservation"
        self._write_authority(run_id)
        result = self._invoke("run", failpoint="after_reservation_record")
        self.assertEqual(result.returncode, 97, result.stderr)
        self.assertFalse(self.runner.ledger_path.exists())
        self.assertTrue(self.runner._state_dir(run_id).is_dir())
        with self.assertRaisesRegex(BaselineError, "retry forbidden"):
            self.runner.recover(run_id)
        record = self._ledger()["runs"][run_id]
        self.assertEqual(record["state"], "interrupted_before_execution_fail_closed")
        self.assertEqual(record["execution_count"], 0)
        packet = self._assert_packet(run_id, "recovery", "RECOVERY")
        self.assertEqual(packet["execution_count"], 0)

    def test_16_post_reservation_live_input_mutation_executes_only_snapshots(self) -> None:
        run_id = "dynamic-scale32-baseline-reserved-inputs"
        self._write_authority(run_id)
        marker = self.root / "MUTATED_LIVE_EXECUTOR_RAN"
        original_create_reservation = self.runner._create_reservation

        def reserve_then_mutate(*args: Any, **kwargs: Any) -> tuple[Path, dict[str, Any], dict[str, Any]]:
            result = original_create_reservation(*args, **kwargs)
            (self.root / "tools/ace2_dynamic_scale32_baseline.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('unsafe live executor ran')\n",
                encoding="utf-8",
            )
            quality_path = self.root / "benchmark/quality/QUALITY_CONFIG.json"
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            quality["determinism"]["python_seed"] = 7
            write_json(quality_path, quality)
            return result

        self.runner._create_reservation = reserve_then_mutate  # type: ignore[method-assign]
        result = self.runner.run()
        self.assertFalse(marker.exists())
        self.assertEqual(self._ledger()["runs"][run_id]["state"], "committed")
        observations = json.loads((result["bundle"] / "raw/input_observations.json").read_text())
        self.assertEqual(observations["environment"]["PYTHONHASHSEED"], str(self.seeds["python_seed"]))
        reservation = json.loads((result["bundle"] / "RESERVATION.json").read_text())
        self.assertEqual(
            {record["source_path"] for record in reservation["reserved_inputs"]},
            {
                "benchmark/quality/PROMPT_MANIFEST.json",
                "benchmark/quality/QUALITY_CONFIG.json",
                "benchmark/quality/requirements.txt",
                "tools/ace2_dynamic_scale32_baseline.py",
                "tools/ace2_software_identity.py",
            },
        )

    def test_17_post_reservation_ambient_environment_drift_is_not_inherited(self) -> None:
        run_id = "dynamic-scale32-baseline-environment-drift"
        self._write_authority(run_id)
        original_create_reservation = self.runner._create_reservation

        def reserve_then_drift(*args: Any, **kwargs: Any) -> tuple[Path, dict[str, Any], dict[str, Any]]:
            result = original_create_reservation(*args, **kwargs)
            os.environ["ACE2_AMBIENT_DRIFT"] = "after-reservation"
            os.environ["PYTHONHASHSEED"] = "999"
            return result

        self.runner._create_reservation = reserve_then_drift  # type: ignore[method-assign]
        with mock.patch.dict(
            os.environ,
            {"ACE2_AMBIENT_DRIFT": "before-reservation", "PYTHONHASHSEED": "111"},
            clear=False,
        ):
            result = self.runner.run()
        observations = json.loads((result["bundle"] / "raw/input_observations.json").read_text())
        environment = observations["environment"]
        self.assertEqual(set(environment), set(ENVIRONMENT_KEYS))
        self.assertNotIn("ACE2_AMBIENT_DRIFT", environment)
        self.assertEqual(environment["PYTHONHASHSEED"], str(self.seeds["python_seed"]))
        launch_environment = json.loads((result["bundle"] / LAUNCH_ENVIRONMENT_NAME).read_text())
        self.assertEqual(environment, launch_environment["variables"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
