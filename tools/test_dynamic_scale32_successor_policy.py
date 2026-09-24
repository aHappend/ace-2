#!/usr/bin/env python3
"""Adversarial synthetic tests for the Dynamic Scale32 successor policy."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

from run_dynamic_scale32_verification_baseline import (  # noqa: E402
    ACCEPTED_ENVIRONMENT_AUDIT_REL,
    ACCEPTED_RECOVERY_HANDOFF_REL,
    AUTHORITY_REL,
    CONTRACT_ID,
    ENVIRONMENT_REPRODUCTION_REL,
    EXPECTED_EXECUTOR_SHA256,
    EXPECTED_PREDECESSOR_LEDGER_SHA256,
    EXPECTED_PREDECESSOR_TERMINAL_SHA256,
    EXPECTED_SOFTWARE_IDENTITY_SHA256,
    EXPECTED_SUCCESSOR_DECISION_SHA256,
    EXECUTOR_REL,
    INPUT_RELS,
    MANAGER_FREEZE_REL,
    POLICY_MIGRATION_HANDOFF_REL,
    POST_POLICY_AUDIT_REL,
    POST_POLICY_REVIEW_REL,
    PREDECESSOR_LEDGER_REL,
    PREDECESSOR_RUN_ID,
    PREDECESSOR_TERMINAL_REL,
    PROPOSAL_COMPANION_REL,
    PROPOSAL_REL,
    RTL_REVIEW_COMPANION_REL,
    RTL_REVIEW_REL,
    SOFTWARE_IDENTITY_REL,
    SUCCESSOR_AUTHORITY_BUNDLE_FILES,
    SUCCESSOR_AUTHORITY_BUNDLE_REL,
    SUCCESSOR_AUTHORITY_COMPANION_REL,
    SUCCESSOR_AUTHORITY_REL,
    SUCCESSOR_AUTHORITY_KIND,
    SUCCESSOR_DECISION_REL,
    SUCCESSOR_GENERATION,
    SUCCESSOR_LEDGER_REL,
    SUCCESSOR_MANAGER_ISSUANCE_REL,
    V1_MANAGER_ISSUANCE_REL,
    V1_AUTHORITY_ARCHIVE_REL,
    BaselineError,
    DynamicScale32BaselineRunner,
    canonical_integrity,
    json_bytes,
    sha256_file,
)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


class DynamicScale32SuccessorPolicyTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ace2-scale32-successor-")
        self.root = Path(self.temporary.name)
        self._copy_fixture(self.root)
        self.runner = DynamicScale32BaselineRunner(self.root)
        platform.platform()
        self.process_patch = mock.patch(
            "run_dynamic_scale32_verification_baseline.subprocess.run",
            side_effect=AssertionError("successor policy tests must not invoke a process"),
        )
        self.process_mock = self.process_patch.start()

    def tearDown(self) -> None:
        self.process_patch.stop()
        self.temporary.cleanup()

    @staticmethod
    def _copy_one(source_root: Path, target_root: Path, relative: Path) -> None:
        source = source_root / relative
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    @classmethod
    def _copy_fixture(cls, target_root: Path) -> None:
        relatives = (
            Path("tools/run_dynamic_scale32_verification_baseline.py"),
            Path("tools/test_dynamic_scale32_successor_policy.py"),
            EXECUTOR_REL,
            SOFTWARE_IDENTITY_REL,
            Path("research/PIPELINE_STATE.json"),
            RTL_REVIEW_REL,
            RTL_REVIEW_COMPANION_REL,
            PROPOSAL_REL,
            PROPOSAL_COMPANION_REL,
            MANAGER_FREEZE_REL,
            *INPUT_RELS,
            SUCCESSOR_DECISION_REL,
            SUCCESSOR_DECISION_REL.with_suffix(".sha256"),
            ACCEPTED_ENVIRONMENT_AUDIT_REL,
            ACCEPTED_ENVIRONMENT_AUDIT_REL.with_suffix(".sha256"),
            ENVIRONMENT_REPRODUCTION_REL,
            ENVIRONMENT_REPRODUCTION_REL.with_suffix(".sha256"),
            ACCEPTED_RECOVERY_HANDOFF_REL,
            ACCEPTED_RECOVERY_HANDOFF_REL.with_suffix(".sha256"),
            PREDECESSOR_LEDGER_REL,
            PREDECESSOR_TERMINAL_REL,
            AUTHORITY_REL,
            AUTHORITY_REL.with_suffix(".sha256"),
            V1_MANAGER_ISSUANCE_REL,
            V1_AUTHORITY_ARCHIVE_REL,
            V1_AUTHORITY_ARCHIVE_REL.with_suffix(".sha256"),
        )
        for relative in relatives:
            cls._copy_one(ROOT, target_root, relative)

    def _migrated(self) -> dict[str, object]:
        self.runner.migrate_successor_ledger()
        return self.runner._load_successor_ledger_readonly()

    def _write_audit_and_review(self) -> tuple[str, str, str]:
        runner_sha256 = sha256_file(self.root / "tools/run_dynamic_scale32_verification_baseline.py")
        audit = {
            "schema_version": 1,
            "kind": "dynamic_scale32_successor_post_policy_environment_audit_v1",
            "contract_id": CONTRACT_ID,
            "status": "pass",
            "authority": False,
            "execution": False,
            "stage_closing": False,
            "source_hashes": {
                "runner": runner_sha256,
                "executor": EXPECTED_EXECUTOR_SHA256,
                "software_identity_helper": EXPECTED_SOFTWARE_IDENTITY_SHA256,
            },
            "observed_counts": {"baseline_model": 1, "candidate_model": 0},
            "no_execution_state": {
                "authority_created": False,
                "reservation_created": False,
                "run_created": False,
                "candidate_task_created": False,
                "process_invoked": False,
            },
        }
        write_json(self.root / POST_POLICY_AUDIT_REL, audit)
        audit_sha256 = sha256_file(self.root / POST_POLICY_AUDIT_REL)
        handoff = {
            "schema_version": 1,
            "kind": "dynamic_scale32_successor_policy_migration_handoff_v1",
            "contract_id": CONTRACT_ID,
            "status": "synthetic_test_only",
            "authority": False,
            "execution": False,
            "stage_closing": False,
            "policy_audit": {"sha256": audit_sha256},
            "source_and_test_hashes": {
                "runner": runner_sha256,
                "successor_policy_test": sha256_file(
                    self.root / "tools/test_dynamic_scale32_successor_policy.py"
                ),
            },
        }
        write_json(self.root / POLICY_MIGRATION_HANDOFF_REL, handoff)
        handoff_sha256 = sha256_file(self.root / POLICY_MIGRATION_HANDOFF_REL)
        review = {
            "schema_version": 1,
            "kind": "dynamic_scale32_successor_policy_independent_review_v1",
            "contract_id": CONTRACT_ID,
            "verdict": "go",
            "independent_reviewer": True,
            "stage_closing": False,
            "bindings": {
                "post_policy_environment_audit_sha256": audit_sha256,
                "runner_sha256": runner_sha256,
            },
        }
        write_json(self.root / POST_POLICY_REVIEW_REL, review)
        return audit_sha256, sha256_file(self.root / POST_POLICY_REVIEW_REL), handoff_sha256

    def _authority(self) -> dict[str, object]:
        audit_sha256, review_sha256, _ = self._write_audit_and_review()
        authority = json.loads((self.root / AUTHORITY_REL).read_text(encoding="utf-8"))
        stage_authority = self.runner._stage_authority()
        frozen = self.runner._frozen_inputs()
        authority.update({
            "kind": SUCCESSOR_AUTHORITY_KIND,
            "run_id": "dynamic-scale32-baseline-successor-v1",
            "authority_binding_sha256": stage_authority["authority_binding_sha256"],
            "inputs": frozen["inputs"],
            "seeds": frozen["seeds"],
            "software_versions": frozen["software_versions"],
            "environment": frozen["launch_environment"]["variables"],
            "environment_sha256": frozen["launch_environment"]["sha256"],
            "command": self.runner._expected_command(authority["scope"]),
            "model_counts": {"baseline_model": 1, "candidate_model": 0},
            "successor_policy": {
                "recovery_generation": SUCCESSOR_GENERATION,
                "decision_sha256": EXPECTED_SUCCESSOR_DECISION_SHA256,
                "post_policy_environment_audit_sha256": audit_sha256,
                "independent_review_sha256": review_sha256,
                "runner_sha256": sha256_file(
                    self.root / "tools/run_dynamic_scale32_verification_baseline.py"
                ),
                "executor_sha256": EXPECTED_EXECUTOR_SHA256,
                "software_identity_helper_sha256": EXPECTED_SOFTWARE_IDENTITY_SHA256,
                "predecessor_ledger_sha256": EXPECTED_PREDECESSOR_LEDGER_SHA256,
                "predecessor_terminal_sha256": EXPECTED_PREDECESSOR_TERMINAL_SHA256,
            },
        })
        authority["integrity"]["canonical_sha256"] = canonical_integrity(authority)
        return authority

    def _bundle_content(self) -> dict[str, bytes]:
        authority = self._authority()
        authority_bytes = json_bytes(authority)
        authority_sha256 = hashlib.sha256(authority_bytes).hexdigest()
        companion_bytes = f"{authority_sha256}  {SUCCESSOR_AUTHORITY_REL.name}\n".encode("ascii")
        policy = authority["successor_policy"]
        issuance = {
            "schema_version": 1,
            "kind": "manager_dynamic_scale32_successor_authority_issuance_v1",
            "contract_id": CONTRACT_ID,
            "stage": "verification",
            "stage_closing": False,
            "recovery_generation": SUCCESSOR_GENERATION,
            "run_id": authority["run_id"],
            "authority": {
                "final_path": SUCCESSOR_AUTHORITY_REL.as_posix(),
                "sha256": authority_sha256,
            },
            "companion": {
                "final_path": SUCCESSOR_AUTHORITY_COMPANION_REL.as_posix(),
                "sha256": hashlib.sha256(companion_bytes).hexdigest(),
            },
            "final_path": SUCCESSOR_MANAGER_ISSUANCE_REL.as_posix(),
            "slot_identity": {
                "ledger_path": SUCCESSOR_LEDGER_REL.as_posix(),
                "ledger_sha256": sha256_file(self.root / SUCCESSOR_LEDGER_REL),
                "recovery_generation": SUCCESSOR_GENERATION,
                "slot_index": 0,
                "status_required_before_and_after_publication": "available",
            },
            "immutable_bindings": {
                "accepted_policy_migration_handoff_sha256": sha256_file(
                    self.root / POLICY_MIGRATION_HANDOFF_REL
                ),
                "accepted_post_policy_environment_audit_sha256": policy[
                    "post_policy_environment_audit_sha256"
                ],
                "candidate_free_executor_sha256": EXPECTED_EXECUTOR_SHA256,
                "independent_policy_review_projection_sha256": policy[
                    "independent_review_sha256"
                ],
                "manager_successor_decision_sha256": EXPECTED_SUCCESSOR_DECISION_SHA256,
                "runner_sha256": sha256_file(
                    self.root / "tools/run_dynamic_scale32_verification_baseline.py"
                ),
                "sealed_v1_authority_sha256": sha256_file(self.root / AUTHORITY_REL),
                "sealed_v1_run_ledger_sha256": EXPECTED_PREDECESSOR_LEDGER_SHA256,
                "sealed_v1_terminal_sha256": EXPECTED_PREDECESSOR_TERMINAL_SHA256,
                "software_identity_helper_sha256": EXPECTED_SOFTWARE_IDENTITY_SHA256,
                "successor_policy_tests_sha256": sha256_file(
                    self.root / "tools/test_dynamic_scale32_successor_policy.py"
                ),
                "versioned_successor_ledger_sha256": sha256_file(
                    self.root / SUCCESSOR_LEDGER_REL
                ),
            },
            "publication": {
                "allowed_artifact_count": 3,
                "atomic_required": True,
                "becomes_effective_only_at_final_paths": True,
                "manager_self_review_can_close_gate": False,
                "requires_fresh_independent_reviewer_acceptance_before_publication": True,
            },
            "authority_effect": {
                "candidate_model": 0,
                "cumulative_baseline_process_attempt_ceiling": 2,
                "execution_authorized": True,
                "publication_is_execution": False,
                "publication_is_l2_acceptance": False,
                "publication_reserves_or_consumes_slot": False,
            },
            "exact_command": authority["command"],
            "conditional_option": "B",
            "predecessor_relationship": "synthetic generation-1 successor; never a v1 retry",
            "claim_boundary": "synthetic publication test only; no reservation or execution",
        }
        return {
            SUCCESSOR_AUTHORITY_REL.name: authority_bytes,
            SUCCESSOR_AUTHORITY_COMPANION_REL.name: companion_bytes,
            SUCCESSOR_MANAGER_ISSUANCE_REL.name: json_bytes(issuance),
        }

    def _write_bundle_directory(
        self,
        content: dict[str, bytes],
        directory: Path | None = None,
    ) -> Path:
        target = directory or (self.root / SUCCESSOR_AUTHORITY_BUNDLE_REL)
        target.mkdir(parents=True, exist_ok=False)
        for name, value in content.items():
            (target / name).write_bytes(value)
        return target

    def _reserve(self, ledger: dict[str, object], run_id: str = "dynamic-scale32-baseline-successor-v1") -> None:
        self.runner.reserve_successor_slot(
            ledger,
            run_id=run_id,
            authority_sha256="a" * 64,
            reservation_sha256="b" * 64,
            reserved_at_utc="2026-08-02T12:00:00.000000Z",
        )

    def test_atomic_directory_publish_success(self) -> None:
        self._migrated()
        ledger_before = self.runner.successor_ledger_path.read_bytes()
        result = self.runner.publish_successor_authority_bundle(self._bundle_content())
        self.assertTrue(result["created"])
        self.assertEqual(
            {path.name for path in self.runner.successor_authority_bundle_path.iterdir()},
            SUCCESSOR_AUTHORITY_BUNDLE_FILES,
        )
        self.assertEqual(
            result["source"]["bundle"]["path"],
            SUCCESSOR_AUTHORITY_BUNDLE_REL.as_posix(),
        )
        self.assertEqual(self.runner.successor_ledger_path.read_bytes(), ledger_before)
        self.process_mock.assert_not_called()

    def test_crash_before_directory_rename_leaves_no_authority(self) -> None:
        self._migrated()

        def crash(point: str) -> None:
            if point == "before_successor_authority_bundle_publish":
                raise RuntimeError("synthetic crash before rename")

        with mock.patch.object(self.runner, "_failpoint", side_effect=crash):
            with self.assertRaisesRegex(RuntimeError, "before rename"):
                self.runner.publish_successor_authority_bundle(self._bundle_content())
        self.assertFalse(self.runner.successor_authority_bundle_path.exists())
        self.assertTrue(any(path.name.startswith(".staging-") for path in self.runner.recovery_root.iterdir()))
        with self.assertRaisesRegex(BaselineError, "bundle missing"):
            self.runner._authority_snapshot()
        self.process_mock.assert_not_called()

    def test_crash_after_directory_rename_exposes_complete_bundle(self) -> None:
        self._migrated()

        def crash(point: str) -> None:
            if point == "after_successor_authority_bundle_publish":
                raise RuntimeError("synthetic crash after rename")

        with mock.patch.object(self.runner, "_failpoint", side_effect=crash):
            with self.assertRaisesRegex(RuntimeError, "after rename"):
                self.runner.publish_successor_authority_bundle(self._bundle_content())
        authority, _, source = self.runner._authority_snapshot()
        self.assertEqual(authority["run_id"], "dynamic-scale32-baseline-successor-v1")
        self.assertEqual(len(source["bundle"]["files"]), 3)
        self.assertEqual(
            {path.name for path in self.runner.successor_authority_bundle_path.iterdir()},
            SUCCESSOR_AUTHORITY_BUNDLE_FILES,
        )
        self.process_mock.assert_not_called()

    def test_partial_and_extra_bundle_rejection(self) -> None:
        self._migrated()
        content = self._bundle_content()
        for label, candidate in (
            ("partial", {key: value for key, value in content.items() if key != SUCCESSOR_MANAGER_ISSUANCE_REL.name}),
            ("extra", {**content, "EXTRA.json": b"{}\n"}),
        ):
            with self.subTest(label=label):
                self._write_bundle_directory(candidate)
                with self.assertRaisesRegex(BaselineError, "contents differ"):
                    self.runner._authority_snapshot()
                shutil.rmtree(self.runner.successor_authority_bundle_path)
        self.process_mock.assert_not_called()

    def test_root_and_split_bundle_rejection(self) -> None:
        self._migrated()
        content = self._bundle_content()
        self.runner.publish_successor_authority_bundle(content)
        self.runner.legacy_root_successor_authority_path.write_bytes(
            content[SUCCESSOR_AUTHORITY_REL.name]
        )
        self.runner.legacy_root_successor_companion_path.write_bytes(
            content[SUCCESSOR_AUTHORITY_COMPANION_REL.name]
        )
        with self.assertRaisesRegex(BaselineError, "unversioned root"):
            self.runner._authority_snapshot()
        self.runner.legacy_root_successor_authority_path.unlink()
        self.runner.legacy_root_successor_companion_path.unlink()
        (self.runner.successor_authority_bundle_path / SUCCESSOR_MANAGER_ISSUANCE_REL.name).unlink()
        split = self.runner.recovery_root / "MANAGER_ISSUANCE.json"
        split.write_bytes(content[SUCCESSOR_MANAGER_ISSUANCE_REL.name])
        with self.assertRaisesRegex(BaselineError, "contents differ"):
            self.runner._authority_snapshot()
        self.process_mock.assert_not_called()

    def test_stale_staging_directory_is_ignored(self) -> None:
        self._migrated()
        stale = self.runner.recovery_root / ".staging-successor-authority-generation-1-v1"
        self._write_bundle_directory(self._bundle_content(), stale)
        report = self.runner.preflight()
        self.assertIsNone(report["execution_authority"])
        self.assertTrue(
            any(item["code"] == "missing_manager_frozen_baseline_command_authority" for item in report["blockers"])
        )
        self.assertTrue(stale.is_dir())
        self.assertFalse(self.runner.successor_authority_bundle_path.exists())
        self.process_mock.assert_not_called()

    def test_v1_root_authority_cannot_authorize_generation_1(self) -> None:
        self._migrated()
        self.assertTrue((self.root / AUTHORITY_REL).is_file())
        self.assertTrue((self.root / V1_MANAGER_ISSUANCE_REL).is_file())
        with self.assertRaisesRegex(BaselineError, "bundle missing"):
            self.runner._authority_snapshot()
        ledger = self.runner._load_successor_ledger_readonly()
        self.assertEqual(ledger["successor_reservations"], 0)
        self.process_mock.assert_not_called()

    def test_duplicate_publication_is_idempotent_only_when_byte_identical(self) -> None:
        self._migrated()
        content = self._bundle_content()
        first = self.runner.publish_successor_authority_bundle(content)
        second = self.runner.publish_successor_authority_bundle(content)
        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        conflicting = dict(content)
        conflicting[SUCCESSOR_MANAGER_ISSUANCE_REL.name] += b" "
        with self.assertRaisesRegex(BaselineError, "conflicting"):
            self.runner.publish_successor_authority_bundle(conflicting)
        self.process_mock.assert_not_called()

    def test_read_only_preflight_inspects_bundle_without_reservation(self) -> None:
        self._migrated()
        self.runner.publish_successor_authority_bundle(self._bundle_content())
        ledger_before = self.runner.successor_ledger_path.read_bytes()
        report = self.runner.preflight()
        self.assertIsNotNone(report["execution_authority"])
        self.assertFalse(
            any(item["code"] == "execution_authority_invalid" for item in report["blockers"]),
            report["blockers"],
        )
        self.assertFalse(report["reservation_created"])
        self.assertEqual(self.runner.successor_ledger_path.read_bytes(), ledger_before)
        self.assertFalse(self.runner._state_dir("dynamic-scale32-baseline-successor-v1").exists())
        self.process_mock.assert_not_called()

    def test_reservation_consumes_slot_only_after_final_bundle_validation(self) -> None:
        self._migrated()
        self.runner.publish_successor_authority_bundle(self._bundle_content())
        authority, _, source = self.runner._authority_snapshot()
        ledger = self.runner._load_successor_ledger_readonly()
        self.runner.reserve_successor_slot(
            ledger,
            run_id=authority["run_id"],
            authority_sha256=source["sha256"],
            reservation_sha256="b" * 64,
            reserved_at_utc="2026-08-02T12:00:00.000000Z",
        )
        self.assertEqual(ledger["recovery_generation_slots"][0]["status"], "consumed")
        disk_ledger = self.runner._load_successor_ledger_readonly()
        authority_path = self.runner.successor_authority_path
        authority_path.write_bytes(authority_path.read_bytes() + b"\n")
        before = copy.deepcopy(disk_ledger)
        with self.assertRaises(BaselineError):
            self.runner._authority_snapshot()
        self.assertEqual(disk_ledger, before)
        self.assertEqual(disk_ledger["recovery_generation_slots"][0]["status"], "available")
        self.process_mock.assert_not_called()

    def test_post_publication_mutation_fails_before_reservation(self) -> None:
        self._migrated()
        self.runner.publish_successor_authority_bundle(self._bundle_content())
        preflight = self.runner.preflight()
        issuance_path = self.runner.successor_manager_issuance_path
        issuance = json.loads(issuance_path.read_text(encoding="utf-8"))
        issuance["run_id"] = "dynamic-scale32-baseline-conflicting-v1"
        write_json(issuance_path, issuance)
        ledger_before = self.runner.successor_ledger_path.read_bytes()
        with self.assertRaisesRegex(BaselineError, "changed after preflight|run_id differs"):
            self.runner._validated_authority_snapshot(preflight)
        self.assertEqual(self.runner.successor_ledger_path.read_bytes(), ledger_before)
        self.assertEqual(
            self.runner._load_successor_ledger_readonly()["successor_reservations"],
            0,
        )
        self.process_mock.assert_not_called()

    def test_mutation_after_validated_snapshot_fails_at_locked_reservation_boundary(self) -> None:
        self._migrated()
        self.runner.publish_successor_authority_bundle(self._bundle_content())
        ledger_before = self.runner.successor_ledger_path.read_bytes()
        validated_snapshot = self.runner._validated_authority_snapshot

        def snapshot_then_mutate(
            preflight: dict[str, object],
        ) -> tuple[dict[str, object], bytes, dict[str, object]]:
            snapshot = validated_snapshot(preflight)
            issuance_path = self.runner.successor_manager_issuance_path
            issuance_path.write_bytes(issuance_path.read_bytes() + b" ")
            return snapshot

        with mock.patch.object(
            self.runner,
            "_validated_authority_snapshot",
            side_effect=snapshot_then_mutate,
        ):
            with self.assertRaisesRegex(
                BaselineError,
                "complete successor authority bundle changed at reservation boundary",
            ):
                self.runner._run_successor()

        self.assertEqual(self.runner.successor_ledger_path.read_bytes(), ledger_before)
        ledger = self.runner._load_successor_ledger_readonly()
        self.assertEqual(ledger["successor_reservations"], 0)
        self.assertEqual(ledger["cumulative_baseline_process_attempts"], 1)
        self.assertEqual(
            ledger["cumulative_model_counts"],
            {"baseline_model": 1, "candidate_model": 0},
        )
        self.assertEqual(ledger["recovery_generation_slots"][0]["status"], "available")
        self.assertFalse(
            self.runner._state_dir("dynamic-scale32-baseline-successor-v1").exists()
        )
        self.assertEqual(
            [
                path.name
                for path in self.runner.state_root.iterdir()
                if path.name.startswith(".staging-reservation-")
            ],
            [],
        )
        self.process_mock.assert_not_called()

    def test_sealed_v1_hash_gate(self) -> None:
        targets = (
            PREDECESSOR_LEDGER_REL,
            PREDECESSOR_TERMINAL_REL,
            AUTHORITY_REL,
        )
        for relative in targets:
            with self.subTest(relative=relative.as_posix()):
                with tempfile.TemporaryDirectory(prefix="ace2-scale32-hash-gate-") as temporary:
                    root = Path(temporary)
                    self._copy_fixture(root)
                    path = root / relative
                    path.write_bytes(path.read_bytes() + b"\n")
                    with self.assertRaisesRegex(BaselineError, "hash mismatch"):
                        DynamicScale32BaselineRunner(root).migrate_successor_ledger()

    def test_idempotent_versioned_migration(self) -> None:
        protected = {
            relative: (self.root / relative).read_bytes()
            for relative in (PREDECESSOR_LEDGER_REL, PREDECESSOR_TERMINAL_REL, AUTHORITY_REL)
        }
        first = self.runner.migrate_successor_ledger()
        first_bytes = self.runner.successor_ledger_path.read_bytes()
        second = self.runner.migrate_successor_ledger()
        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(first_bytes, self.runner.successor_ledger_path.read_bytes())
        ledger = self.runner._load_successor_ledger_readonly()
        predecessor = json.loads((self.root / PREDECESSOR_LEDGER_REL).read_text())
        self.assertEqual(ledger["runs"][PREDECESSOR_RUN_ID], predecessor["runs"][PREDECESSOR_RUN_ID])
        self.assertEqual(ledger["migration_bindings"]["predecessor_ledger"]["sha256"], EXPECTED_PREDECESSOR_LEDGER_SHA256)
        self.assertEqual(
            protected,
            {relative: (self.root / relative).read_bytes() for relative in protected},
        )

    def test_v1_permanent_rejection(self) -> None:
        ledger = self._migrated()
        with self.assertRaisesRegex(BaselineError, "sealed v1"):
            self._reserve(ledger, PREDECESSOR_RUN_ID)
        with self.assertRaisesRegex(BaselineError, "sealed v1"):
            self.runner.mark_successor_invoked(ledger, PREDECESSOR_RUN_ID)
        with self.assertRaisesRegex(BaselineError, "sealed v1 recovery"):
            self.runner.recover(PREDECESSOR_RUN_ID)
        with self.assertRaisesRegex(BaselineError, "sealed v1 verify-as-run"):
            self.runner.verify(PREDECESSOR_RUN_ID)
        self.process_mock.assert_not_called()

    def test_unbound_successor_rejection(self) -> None:
        ledger = self._migrated()
        authority = self._authority()
        self.runner._validate_successor_authority_policy(authority, ledger)
        mutations = {
            "decision": ("decision_sha256", "0" * 64),
            "audit": ("post_policy_environment_audit_sha256", "0" * 64),
            "review": ("independent_review_sha256", "0" * 64),
            "runner": ("runner_sha256", "0" * 64),
            "executor": ("executor_sha256", "0" * 64),
            "helper": ("software_identity_helper_sha256", "0" * 64),
            "generation": ("recovery_generation", 2),
        }
        for label, (field, value) in mutations.items():
            with self.subTest(label=label):
                candidate = copy.deepcopy(authority)
                candidate["successor_policy"][field] = value
                with self.assertRaises(BaselineError):
                    self.runner._validate_successor_authority_policy(candidate, ledger)
        reused = copy.deepcopy(authority)
        reused["run_id"] = PREDECESSOR_RUN_ID
        with self.assertRaisesRegex(BaselineError, "sealed v1"):
            self.runner._validate_successor_authority_policy(reused, ledger)

    def test_single_successor_reservation(self) -> None:
        ledger = self._migrated()
        authority = self._authority()
        self.runner._validate_successor_authority_policy(authority, ledger)
        self._reserve(ledger)
        self.runner._save_successor_ledger(ledger)
        self.assertEqual(set(ledger["runs"]), {PREDECESSOR_RUN_ID, authority["run_id"]})
        self.assertEqual(ledger["successor_reservations"], 1)
        self.assertEqual(ledger["cumulative_baseline_process_attempts"], 1)
        self.assertEqual(ledger["cumulative_model_counts"], {"baseline_model": 1, "candidate_model": 0})
        self.assertEqual(ledger["recovery_generation_slots"][0]["status"], "consumed")
        self.process_mock.assert_not_called()

    def test_no_third_attempt(self) -> None:
        ledger = self._migrated()
        self._reserve(ledger)
        before = copy.deepcopy(ledger)
        with self.assertRaisesRegex(BaselineError, "already consumed"):
            self._reserve(ledger, "dynamic-scale32-baseline-successor-v2")
        self.assertEqual(ledger, before)
        self.assertNotIn("dynamic-scale32-baseline-successor-v2", ledger["runs"])

    def test_successor_duplicate_fail_closed(self) -> None:
        ledger = self._migrated()
        self._reserve(ledger)
        before = copy.deepcopy(ledger)
        with self.assertRaisesRegex(BaselineError, "duplicate"):
            self._reserve(ledger)
        self.assertEqual(ledger, before)
        self.assertEqual(ledger["cumulative_baseline_process_attempts"], 1)
        self.process_mock.assert_not_called()

    def test_successor_recovery_never_reruns(self) -> None:
        states = (
            "reserved",
            "running",
            "failed_fail_closed",
            "validation_fail_closed",
            "publication_fail_closed",
            "committed",
        )
        for state in states:
            with self.subTest(state=state):
                ledger = self._migrated()
                self._reserve(ledger)
                run_id = "dynamic-scale32-baseline-successor-v1"
                if state != "reserved":
                    self.runner.mark_successor_invoked(ledger, run_id)
                if state not in {"reserved", "running"}:
                    self.runner.terminalize_successor_policy(ledger, run_id, state, "synthetic terminal")
                before_attempts = ledger["cumulative_baseline_process_attempts"]
                result = self.runner.recover_successor_policy(ledger, run_id)
                self.assertIn("no_process_invocation", result)
                self.assertEqual(ledger["cumulative_baseline_process_attempts"], before_attempts)
                self.assertEqual(ledger["recovery_generation_slots"][0]["status"], "consumed")
        self.process_mock.assert_not_called()

    def test_cumulative_count_transition(self) -> None:
        ledger = self._migrated()
        authority = self._authority()
        self.runner._validate_successor_authority_policy(authority, ledger)
        self.assertEqual(ledger["cumulative_baseline_process_attempts"], 1)
        self._reserve(ledger)
        self.assertEqual(ledger["cumulative_baseline_process_attempts"], 1)
        self.runner.mark_successor_invoked(ledger, authority["run_id"])
        self.assertEqual(ledger["cumulative_baseline_process_attempts"], 2)
        self.assertEqual(ledger["cumulative_model_counts"], {"baseline_model": 2, "candidate_model": 0})
        with self.assertRaisesRegex(BaselineError, "cannot invoke"):
            self.runner.mark_successor_invoked(ledger, authority["run_id"])

    def test_publication_does_not_reopen_or_accept_l2(self) -> None:
        ledger = self._migrated()
        self._reserve(ledger)
        run_id = "dynamic-scale32-baseline-successor-v1"
        self.runner.mark_successor_invoked(ledger, run_id)
        self.runner.terminalize_successor_policy(ledger, run_id, "committed", "synthetic publication")
        self.assertEqual(
            ledger["publication"],
            {"successor_slot_reopened": False, "baseline_l2_accepted": False},
        )
        self.assertEqual(ledger["recovery_generation_slots"][0]["status"], "consumed")
        reopened = copy.deepcopy(ledger)
        reopened["publication"]["successor_slot_reopened"] = True
        with self.assertRaisesRegex(BaselineError, "publication"):
            self.runner._validate_successor_ledger(reopened)

    def test_candidate_surface_interlock(self) -> None:
        ledger = self._migrated()
        authority = self._authority()
        candidate_surfaces = (
            {"candidate_import": "candidate.module"},
            {"candidate_flag": "--candidate-evidence"},
            {"candidate_task": {"enabled": True}},
            {"candidate_output": "candidate-results.json"},
            {"model_counts": {"baseline_model": 1, "candidate_model": 1}},
        )
        for surface in candidate_surfaces:
            with self.subTest(surface=surface):
                candidate = copy.deepcopy(authority)
                candidate.update(surface)
                with self.assertRaisesRegex(BaselineError, "candidate"):
                    self.runner._validate_successor_authority_policy(candidate, ledger)
        self.process_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
