#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from argus_skill.life.memory import Backlog, BacklogItem


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "native_consumer",
    ROOT / "scripts/run_prompt_recovery_native_consumer.py",
)
assert SPEC and SPEC.loader
consumer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(consumer)


class NativeConsumerTest(unittest.TestCase):
    def backlog(self, root: Path, owner: str = consumer.OLD_OWNER) -> Backlog:
        backlog = Backlog(root / "backlog.jsonl")
        backlog._save(
            [
                BacklogItem(
                    id=consumer.TASK_ID,
                    ts=time.time(),
                    title="recovery",
                    objective="recover",
                    status="running",
                    running_owner=owner,
                )
            ]
        )
        return backlog

    def test_cli_exposes_no_science_or_general_backlog_action(self) -> None:
        choices = consumer.parser()._subparsers._group_actions[0].choices
        self.assertEqual(
            set(choices),
            {"launch-consumer", "_consume", "consumer-status"},
        )

    def test_worker_argv_is_locally_derived(self) -> None:
        self.assertEqual(
            consumer.expected_worker_argv(),
            [
                str(Path(consumer.sys.executable).resolve()),
                "-B",
                str(
                    (
                        ROOT / "scripts/run_prompt_recovery_native_consumer.py"
                    ).resolve()
                ),
                "_consume",
            ],
        )
        self.assertNotEqual(
            consumer.expected_worker_argv(),
            ["python", "science-command"],
        )

    def test_transfer_is_exact_single_field_cas(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            backlog = self.backlog(root)
            state = root / "state"
            continuation = root / "continuation"
            state.mkdir()
            continuation.mkdir()
            authority = root / "authority.json"
            authority.write_text("{}")
            (continuation / "adoption.json").write_text("{}")
            (continuation / "continuation-owner.json").write_text("{}")
            with (
                mock.patch.object(consumer, "STATE_DIR", state),
                mock.patch.object(consumer, "CONTINUATION_STATE", continuation),
                mock.patch.object(consumer, "AUTHORITY_PATH", authority),
                mock.patch.object(consumer, "identity_live", return_value=True),
                mock.patch.object(consumer, "lock_holder_pid", return_value=1),
                mock.patch.object(consumer, "emit") as emit,
            ):
                consumer.transfer_owner(
                    backlog,
                    {"status": "adopted"},
                    {"pid": 1, "start_ticks": 2, "session_id": 1},
                )
            item = backlog.all()[0]
            self.assertEqual(item.status, "running")
            self.assertEqual(item.running_owner, consumer.NEW_OWNER)
            emit.assert_called_once()

    def test_transfer_rejects_unexpected_owner(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            backlog = self.backlog(root, owner="other")
            state = root / "state"
            continuation = root / "continuation"
            state.mkdir()
            continuation.mkdir()
            authority = root / "authority.json"
            authority.write_text("{}")
            (continuation / "adoption.json").write_text("{}")
            (continuation / "continuation-owner.json").write_text("{}")
            with (
                mock.patch.object(consumer, "STATE_DIR", state),
                mock.patch.object(consumer, "CONTINUATION_STATE", continuation),
                mock.patch.object(consumer, "AUTHORITY_PATH", authority),
                mock.patch.object(consumer, "identity_live", return_value=True),
                mock.patch.object(consumer, "lock_holder_pid", return_value=1),
            ):
                with self.assertRaisesRegex(
                    consumer.ConsumerError,
                    "neither expected owner",
                ):
                    consumer.transfer_owner(
                        backlog,
                        {},
                        {"pid": 1, "start_ticks": 2, "session_id": 1},
                    )

    def test_transfer_rejects_new_owner_without_prior_intent(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            backlog = self.backlog(root, owner=consumer.NEW_OWNER)
            state = root / "state"
            continuation = root / "continuation"
            state.mkdir()
            continuation.mkdir()
            (continuation / "adoption.json").write_text("{}")
            (continuation / "continuation-owner.json").write_text("{}")
            with (
                mock.patch.object(consumer, "STATE_DIR", state),
                mock.patch.object(consumer, "CONTINUATION_STATE", continuation),
                mock.patch.object(consumer, "identity_live", return_value=True),
                mock.patch.object(consumer, "lock_holder_pid", return_value=1),
            ):
                with self.assertRaisesRegex(
                    consumer.ConsumerError,
                    "no prior native CAS intent",
                ):
                    consumer.transfer_owner(
                        backlog,
                        {},
                        {"pid": 1, "start_ticks": 2, "session_id": 1},
                    )

    def test_transfer_rechecks_live_lock_holder_inside_backlog_lock(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            backlog = self.backlog(root)
            state = root / "state"
            continuation = root / "continuation"
            original = root / "original"
            state.mkdir()
            continuation.mkdir()
            original.mkdir()
            (original / "owner.lock").touch()
            (continuation / "adoption.json").write_text("{}")
            (continuation / "continuation-owner.json").write_text("{}")
            owner = {"pid": 10, "start_ticks": 20, "session_id": 10}
            with (
                mock.patch.object(consumer, "STATE_DIR", state),
                mock.patch.object(consumer, "CONTINUATION_STATE", continuation),
                mock.patch.object(consumer, "ORIGINAL_STATE", original),
                mock.patch.object(consumer, "identity_live", return_value=True),
                mock.patch.object(consumer, "lock_holder_pid", return_value=None),
            ):
                with self.assertRaisesRegex(
                    consumer.ConsumerError,
                    "released original owner.lock",
                ):
                    consumer.transfer_owner(backlog, {}, owner)
            self.assertEqual(backlog.all()[0].running_owner, consumer.OLD_OWNER)

    def test_committed_transfer_remains_valid_after_owner_exit(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            backlog = self.backlog(root, owner=consumer.NEW_OWNER)
            state = root / "state"
            continuation = root / "continuation"
            state.mkdir()
            continuation.mkdir()
            (continuation / "adoption.json").write_text("{}")
            intent = state / "owner-transfer-intent.json"
            intent.write_text('{"status":"VALIDATED_FOR_EXACT_CAS"}')
            committed = state / "owner-transfer-committed.json"
            committed.write_text(
                consumer.json.dumps(
                    {
                        "status": "COMMITTED_UNDER_NATIVE_BACKLOG_LOCK",
                        "task_id": consumer.TASK_ID,
                        "old_owner": consumer.OLD_OWNER,
                        "new_owner": consumer.NEW_OWNER,
                        "intent_sha256": consumer.sha256_file(intent),
                        "adoption_sha256": consumer.sha256_file(
                            continuation / "adoption.json"
                        ),
                    }
                )
            )
            with (
                mock.patch.object(consumer, "STATE_DIR", state),
                mock.patch.object(consumer, "CONTINUATION_STATE", continuation),
                mock.patch.object(
                    consumer,
                    "validate_adoption",
                    side_effect=AssertionError("must not require live owner"),
                ),
            ):
                self.assertTrue(
                    consumer.observe_owner_transition(backlog, transferred=True)
                )

    def test_observer_identity_accepts_approved_receipt_shapes(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            state = Path(raw)
            identity = {
                "pid": 10,
                "start_ticks": 20,
                "session_id": 10,
                "argv": ["worker"],
                "cwd": str(ROOT),
            }
            process = state / "observer-process.json"
            process.write_text(consumer.json.dumps(identity))
            ready = {
                **identity,
                "authority_sha256": consumer.EXPECTED_CONTINUATION_AUTHORITY_SHA256,
            }
            (state / "observer-ready.json").write_text(
                consumer.json.dumps(ready)
            )
            claim = {
                **identity,
                "observer_process_sha256": consumer.sha256_file(process),
            }
            (state / "worker-claim.json").write_text(
                consumer.json.dumps(claim)
            )
            with mock.patch.object(consumer, "CONTINUATION_STATE", state):
                self.assertEqual(consumer.observer_identity(), identity)

    def test_child_fails_closed_when_parent_receipt_is_absent(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            missing = Path(raw) / "consumer-process.json"
            with self.assertRaisesRegex(
                consumer.ConsumerError,
                "parent consumer process receipt is absent",
            ):
                consumer.wait_for_parent_receipt(missing, timeout_seconds=0)

    def test_fast_post_adoption_blocked_reports_missed_live_transfer(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            backlog = self.backlog(root)
            state = root / "continuation"
            consumer_state = root / "consumer"
            state.mkdir()
            consumer_state.mkdir()
            terminal_path = state / "continuation-terminal.json"
            terminal_path.write_text(
                consumer.json.dumps(
                    {
                        "status": "BLOCKED_NO_FURTHER_LAUNCH",
                        "error": "post-adoption blocker",
                        "case2_rtl_execution_count": 0,
                        "case2_replay": False,
                    }
                )
            )
            (state / "adoption.json").write_text("{}")
            (state / "continuation-owner.json").write_text("{}")
            with (
                mock.patch.object(consumer, "CONTINUATION_STATE", state),
                mock.patch.object(consumer, "STATE_DIR", consumer_state),
                mock.patch.object(consumer, "observer_identity", return_value={}),
                mock.patch.object(consumer, "identity_live", return_value=False),
                mock.patch.object(consumer, "validate_adoption_records"),
                mock.patch.object(consumer, "settle") as settle,
                mock.patch.object(consumer, "emit") as emit,
                mock.patch.object(
                    consumer,
                    "validate_adoption",
                    side_effect=AssertionError("live transfer must not run"),
                ),
            ):
                self.assertTrue(
                    consumer.settle_post_exit_blocked(backlog, terminal_path)
                )
            settle.assert_not_called()
            emit.assert_called_once()
            self.assertTrue(
                (consumer_state / "consumer-blocked.json").exists()
            )

    def test_pre_adoption_blocked_waits_for_original_owner_exit(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            backlog = self.backlog(root)
            state = root / "continuation"
            state.mkdir()
            terminal_path = state / "continuation-terminal.json"
            terminal_path.write_text(
                consumer.json.dumps(
                    {"status": "BLOCKED_NO_FURTHER_LAUNCH"}
                )
            )
            with (
                mock.patch.object(consumer, "CONTINUATION_STATE", state),
                mock.patch.object(consumer, "observer_identity", return_value={}),
                mock.patch.object(consumer, "identity_live", side_effect=[False, True]),
                mock.patch.object(consumer, "settle") as settle,
            ):
                self.assertFalse(
                    consumer.settle_post_exit_blocked(backlog, terminal_path)
                )
            settle.assert_not_called()

    def test_blocked_terminal_uses_native_mark_failed(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            backlog = self.backlog(root, owner=consumer.OLD_OWNER)
            state = root / "state"
            continuation = root / "continuation"
            state.mkdir()
            continuation.mkdir()
            authority = root / "authority.json"
            authority.write_text("{}")
            (continuation / "continuation-terminal.json").write_text("{}")
            terminal = {
                "status": "BLOCKED_NO_FURTHER_LAUNCH",
                "error": "exact technical blocker",
                "case2_rtl_execution_count": 0,
                "case2_replay": False,
            }
            with (
                mock.patch.object(consumer, "STATE_DIR", state),
                mock.patch.object(consumer, "CONTINUATION_STATE", continuation),
                mock.patch.object(consumer, "AUTHORITY_PATH", authority),
                mock.patch.object(consumer, "emit") as emit,
            ):
                consumer.settle(backlog, terminal)
            item = backlog.all()[0]
            self.assertEqual(item.status, "failed")
            self.assertEqual(item.last_error, "exact technical blocker")
            emit.assert_called_once()

    def test_success_requires_real_done_l2_before_native_mark_done(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            backlog = self.backlog(root, owner=consumer.NEW_OWNER)
            state = root / "state"
            continuation = root / "continuation"
            state.mkdir()
            continuation.mkdir()
            authority = root / "authority.json"
            authority.write_text("{}")
            (continuation / "continuation-terminal.json").write_text("{}")
            terminal = {
                "status": "PASS_PENDING_FINAL_INDEPENDENT_L2",
                "case2_rtl_execution_count": 0,
                "case2_oracle_execution_count": 1,
                "case3_rtl_execution_count": 1,
                "case3_oracle_execution_count": 1,
            }
            decision = mock.Mock(status="done", reason="independent evidence accepted")
            review = state / "final-l2-review.json"
            review.write_text("{}")
            with (
                mock.patch.object(consumer, "STATE_DIR", state),
                mock.patch.object(consumer, "CONTINUATION_STATE", continuation),
                mock.patch.object(consumer, "AUTHORITY_PATH", authority),
                mock.patch.object(consumer, "reviewer_decision", return_value=decision),
                mock.patch.object(consumer, "emit") as emit,
            ):
                consumer.settle(backlog, terminal)
            item = backlog.all()[0]
            self.assertEqual(item.status, "done")
            self.assertTrue(item.outcome["final_submission_certified"])
            emit.assert_called_once()

    def test_non_done_l2_blocks_native_task(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            backlog = self.backlog(root, owner=consumer.NEW_OWNER)
            state = root / "state"
            continuation = root / "continuation"
            state.mkdir()
            continuation.mkdir()
            authority = root / "authority.json"
            authority.write_text("{}")
            (continuation / "continuation-terminal.json").write_text("{}")
            (state / "final-l2-review.json").write_text("{}")
            terminal = {
                "status": "PASS_PENDING_FINAL_INDEPENDENT_L2",
                "case2_rtl_execution_count": 0,
                "case2_oracle_execution_count": 1,
                "case3_rtl_execution_count": 1,
                "case3_oracle_execution_count": 1,
            }
            decision = mock.Mock(status="blocked", reason="evidence differs")
            with (
                mock.patch.object(consumer, "STATE_DIR", state),
                mock.patch.object(consumer, "CONTINUATION_STATE", continuation),
                mock.patch.object(consumer, "AUTHORITY_PATH", authority),
                mock.patch.object(consumer, "reviewer_decision", return_value=decision),
                mock.patch.object(consumer, "emit"),
            ):
                consumer.settle(backlog, terminal)
            item = backlog.all()[0]
            self.assertEqual(item.status, "failed")
            self.assertFalse(item.outcome["final_submission_certified"])

    def test_final_l2_cannot_settle_a_concurrently_reassigned_task(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            backlog = self.backlog(root, owner=consumer.NEW_OWNER)
            state = root / "state"
            continuation = root / "continuation"
            authority = root / "authority.json"
            state.mkdir()
            continuation.mkdir()
            authority.write_text("{}")
            (continuation / "continuation-terminal.json").write_text("{}")
            (state / "final-l2-review.json").write_text("{}")
            terminal = {
                "status": "PASS_PENDING_FINAL_INDEPENDENT_L2",
                "case2_rtl_execution_count": 0,
                "case2_oracle_execution_count": 1,
                "case3_rtl_execution_count": 1,
                "case3_oracle_execution_count": 1,
            }
            decision = mock.Mock(status="done", reason="accepted")

            def reassign(*_args):
                backlog.update(consumer.TASK_ID, running_owner="other")
                return decision

            with (
                mock.patch.object(consumer, "STATE_DIR", state),
                mock.patch.object(consumer, "CONTINUATION_STATE", continuation),
                mock.patch.object(consumer, "AUTHORITY_PATH", authority),
                mock.patch.object(consumer, "reviewer_decision", side_effect=reassign),
            ):
                with self.assertRaisesRegex(
                    consumer.ConsumerError,
                    "owner changed during final settlement",
                ):
                    consumer.settle(backlog, terminal)
            self.assertEqual(backlog.all()[0].status, "running")

    def test_partial_terminal_state_recovers_missing_event_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            backlog = self.backlog(root, owner=consumer.NEW_OWNER)
            state = root / "state"
            state.mkdir()
            outcome = {"native_consumer": True}
            intent = {
                "settlement_id": "a" * 64,
                "task_id": consumer.TASK_ID,
                "title": "recovery",
                "objective": "recover",
                "success": True,
                "backlog_status": "done",
                "event_status": "done",
                "summary": "accepted",
                "review_status": "done",
                "expected_owner": consumer.NEW_OWNER,
                "finished_ts": time.time(),
                "last_error": "",
                "outcome": outcome,
                "continuation_terminal_sha256": "b" * 64,
            }
            (state / "settlement-intent.json").write_text(
                consumer.json.dumps(intent)
            )
            item = backlog.all()[0]
            item.status = "done"
            item.outcome = outcome
            item.last_error = ""
            backlog._save([item])
            with (
                mock.patch.object(consumer, "STATE_DIR", state),
                mock.patch.object(consumer, "event_already_emitted", return_value=False),
                mock.patch.object(consumer, "emit") as emit,
                mock.patch.object(consumer, "AUTHORITY_PATH", state / "authority.json"),
                mock.patch.object(consumer, "sha256_file", return_value="c" * 64),
            ):
                self.assertTrue(consumer.recover_pending_settlement(backlog))
            emit.assert_called_once()
            self.assertTrue((state / "native-settlement.json").exists())


if __name__ == "__main__":
    unittest.main()
