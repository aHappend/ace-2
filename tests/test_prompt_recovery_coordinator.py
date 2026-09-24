from __future__ import annotations

import json
import os
import signal
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

from scripts import run_prompt_recovery_coordinator as coordinator


class PromptRecoveryCoordinatorTest(unittest.TestCase):
    def run_case_to_prelaunch_guard(
        self,
        root: Path,
        *,
        cancellation: str | None = None,
        observed_time: float = 10.0,
        overall_deadline: float = 100.0,
        current_free_bytes: int = 100,
    ) -> tuple[
        tuple[dict[str, object] | None, dict[str, object] | None],
        mock.Mock,
    ]:
        DiskUsage = namedtuple("DiskUsage", "total used free")
        authority = root / "authority.json"
        contract = root / "contract.json"
        authority.write_text("{}\n", encoding="utf-8")
        contract.write_text("{}\n", encoding="utf-8")
        with mock.patch.object(
            coordinator.shutil,
            "disk_usage",
            side_effect=[
                DiskUsage(100, 0, 100),
                DiskUsage(100, 100 - current_free_bytes, current_free_bytes),
            ],
        ), mock.patch.object(
            coordinator.time,
            "monotonic",
            return_value=observed_time,
        ), mock.patch.object(
            coordinator.stage1.generation,
            "validate_max_new_tokens",
        ), mock.patch.object(
            coordinator.stage1.product,
            "verify_accepted_position01",
            return_value={},
        ), mock.patch.object(
            coordinator.stage1.generation,
            "resolve_snapshot",
            return_value=root / "model",
        ), mock.patch.object(
            coordinator.stage1.generation,
            "tokenizer_record",
            return_value=({}, None),
        ), mock.patch.object(
            coordinator.stage1,
            "file_record",
            return_value={"path": "injected", "sha256": "0" * 64},
        ), mock.patch.object(
            coordinator.stage1,
            "source_binding",
            return_value={"sha256": "0" * 64},
        ), mock.patch.object(
            coordinator,
            "execution_source_binding",
            return_value={"sha256": "0" * 64},
        ), mock.patch.object(
            coordinator.stage1,
            "tool_record",
            return_value={},
        ), mock.patch.object(
            coordinator,
            "launch_at_scientific_boundary",
        ) as launch:
            result = coordinator.run_case(
                case={"id": "verification-question", "prompt": "question"},
                output=root / "output",
                state_dir=root / "state",
                timeout_seconds=10,
                oracle_timeout_seconds=10,
                cadence_seconds=1,
                overall_deadline=overall_deadline,
                minimum_free_bytes=1,
                cancel_requested=lambda: cancellation,
                authority_paths=(authority, contract),
                contract_path=contract,
                approved_bindings={},
                control_bindings={},
            )
        return result, launch

    def test_contract_is_narrow_and_preserves_first_case(self) -> None:
        contract = coordinator.validate_contract(
            coordinator.ROOT / "tests/prompt_recovery_coordinator_v1.json"
        )
        self.assertEqual("multilingual-greeting", contract["preserved_case_id"])
        self.assertEqual("verification-question", contract["case2_id"])
        self.assertEqual("kv-cache-question", contract["case3_id"])
        self.assertEqual(
            "reports/verification/prompt-suite-rtl-oracle-recovery-0001/"
            "scientific-output",
            contract["output"],
        )
        self.assertGreaterEqual(contract["case_timeout_seconds"], 90_000)
        self.assertGreater(
            contract["overall_timeout_seconds"],
            2 * contract["case_timeout_seconds"],
        )
        preserved = coordinator.load_json(
            coordinator.project_path(contract["preserved_oracle_result"])
        )
        self.assertEqual(
            preserved["source_binding"]["sha256"],
            coordinator.stage1.source_binding()["sha256"],
        )

    def test_dependency_binding_covers_executed_testbenches_and_arithmetic(self) -> None:
        binding = coordinator.execution_source_binding()
        paths = {record["path"] for record in binding["files"]}
        self.assertIn(
            "tools/run_lora_v4_two_token_24layer_rtl_authorized.py",
            paths,
        )
        self.assertIn("tools/run_lora_v4_lm_head_top_token_rtl.py", paths)
        self.assertTrue(
            any(path.startswith("verification/tb/") for path in paths)
        )

    def test_process_stat_parser_binds_start_ticks(self) -> None:
        raw = "44 (worker with ) in name) " + " ".join(
            ["S", "1"] + ["0"] * 17 + ["987654", "0"]
        )
        parsed = coordinator.parse_process_stat(raw, 44)
        self.assertEqual(44, parsed["pid"])
        self.assertEqual(1, parsed["ppid"])
        self.assertEqual(987654, parsed["start_ticks"])

    def test_case2_launch_ticket_is_exclusive_and_non_consuming(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authority = root / "authority.json"
            contract = root / "contract.json"
            prompt = root / "prompt.utf8"
            authority.write_text("{}\n", encoding="utf-8")
            contract.write_text("{}\n", encoding="utf-8")
            prompt.write_text("question", encoding="utf-8")
            first = coordinator.create_case2_launch_ticket(
                root,
                authority,
                contract,
                prompt,
                root / "attempt",
                ["python", "--worker"],
            )
            self.assertEqual(
                "READY_FOR_EXACTLY_ONE_INNER_POPEN",
                first["status"],
            )
            self.assertEqual(["python", "--worker"], first["worker_argv"])
            self.assertEqual(str(prompt.resolve()), first["prompt"])
            self.assertEqual(
                coordinator.execution_source_binding()["sha256"],
                first["source_and_rtl_sha256"],
            )
            self.assertFalse((root / "authority-consumed.json").exists())
            with self.assertRaises(coordinator.DuplicateLaunch):
                coordinator.create_case2_launch_ticket(
                    root,
                    authority,
                    contract,
                    prompt,
                    root / "attempt",
                    ["python", "--worker"],
                )

    def test_authority_issuance_is_review_bound_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_path = root / "contract.json"
            review_path = root / "review.json"
            authority_path = root / "authority.json"
            state_dir = root / "state"
            contract_path.write_text('{"contract": true}\n', encoding="utf-8")
            review_path.write_text('{"review": true}\n', encoding="utf-8")
            contract = {
                "authority_id": "authority-1",
                "scope_authorization_id": "scope-1",
            }
            bindings = {"implementation.py": "0" * 64}
            with mock.patch.object(
                coordinator,
                "validate_contract",
                return_value=contract,
            ), mock.patch.object(
                coordinator,
                "validate_review",
                return_value={},
            ) as validate_review, mock.patch.object(
                coordinator,
                "reviewed_bindings",
                return_value=bindings,
            ):
                authority = coordinator.issue_authority(
                    contract_path,
                    review_path,
                    authority_path,
                    state_dir,
                )
                self.assertEqual(1, authority["authority_cardinality"])
                self.assertEqual(1, authority["case2_execution_limit"])
                self.assertEqual(bindings, authority["reviewed_bindings"])
                self.assertEqual(str(state_dir.resolve()), authority["state_dir"])
                validate_review.assert_called_once_with(
                    review_path,
                    contract_path,
                )
                with self.assertRaisesRegex(
                    coordinator.CoordinationError,
                    "single-use authority already exists",
                ):
                    coordinator.issue_authority(
                        contract_path,
                        review_path,
                        authority_path,
                        state_dir,
                    )

    def test_owner_lock_rejects_a_second_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary)
            descriptor = coordinator.acquire_owner_lock(state_dir)
            try:
                with self.assertRaisesRegex(
                    coordinator.CoordinationError,
                    "another coordinator owner holds the lock",
                ):
                    coordinator.acquire_owner_lock(state_dir)
            finally:
                os.close(descriptor)

    def test_wrapper_signal_is_recorded_without_process_termination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "signals.jsonl"
            cancellation: dict[str, str] = {}
            with mock.patch.object(os, "killpg") as killpg:
                coordinator.signal_recorder(path, cancellation)(
                    signal.SIGTERM,
                    object(),
                )
            killpg.assert_not_called()
            self.assertEqual({}, cancellation)
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                "RECORDED_WITHOUT_ABANDONING_OR_KILLING_STAGE",
                record["action"],
            )

    def test_consumption_is_immediately_before_scientific_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authority = root / "authority.json"
            contract = root / "contract.json"
            prompt = root / "prompt.utf8"
            authority.write_text("{}\n", encoding="utf-8")
            contract.write_text("{}\n", encoding="utf-8")
            prompt.write_text("question", encoding="utf-8")
            observed = []
            worker_command = [
                str(Path(os.sys.executable).resolve()),
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
            ]
            attempt = root / "attempt"
            boundary_path = root / "case2-rtl-popen-boundary.json"
            boundary_role = "verification-question-gated-launcher"
            spawn_command = coordinator.gate_launch_command(
                gate_path=root / "science-gate.json",
                boundary_record_path=boundary_path,
                boundary_role=boundary_role,
                prompt=prompt,
                output=attempt,
                worker_command=worker_command,
            )

            def spawn() -> mock.Mock:
                receipt = json.loads(
                    (root / "authority-consumed.json").read_text(encoding="utf-8")
                )
                observed.append(receipt["status"])
                self.assertFalse((root / "case2-process-started.json").exists())
                coordinator.publish_child_boundary_identity(
                    boundary_record_path=boundary_path,
                    boundary_role=boundary_role,
                    prompt=prompt,
                    output=attempt,
                    worker_command=worker_command,
                )
                return mock.Mock(pid=123)

            with mock.patch.object(
                coordinator,
                "process_identity",
                return_value={
                    "pid": 123,
                    "start_ticks": 456,
                    "argv": spawn_command,
                    "cwd": str(coordinator.ROOT),
                    "session_id": 123,
                    "state": "S",
                },
            ):
                coordinator.launch_at_scientific_boundary(
                    state_dir=root,
                    authority_paths=(authority, contract),
                    prompt=prompt,
                    attempt=attempt,
                    worker_command=worker_command,
                    spawn_command=spawn_command,
                    boundary_record_path=boundary_path,
                    boundary_role=boundary_role,
                    spawn=spawn,
                )
            self.assertEqual(
                ["CONSUMED_IMMEDIATELY_BEFORE_INNER_RTL_POPEN"],
                observed,
            )
            popen = json.loads(
                (root / "case2-popen-succeeded.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, popen["case2_popen_success_count"])
            boundary = json.loads(
                boundary_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                "DETACHED_CHILD_PUBLISHED_IDENTITY_BEFORE_GATE_RELEASE",
                boundary["status"],
            )

    def test_scientific_spawn_failure_consumes_without_process_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authority = root / "authority.json"
            contract = root / "contract.json"
            prompt = root / "prompt.utf8"
            authority.write_text("{}\n", encoding="utf-8")
            contract.write_text("{}\n", encoding="utf-8")
            prompt.write_text("question", encoding="utf-8")
            with self.assertRaisesRegex(OSError, "injected spawn failure"):
                coordinator.launch_at_scientific_boundary(
                    state_dir=root,
                    authority_paths=(authority, contract),
                    prompt=prompt,
                    attempt=root / "attempt",
                    worker_command=["python", "--worker"],
                    spawn_command=["python", "--worker"],
                    boundary_record_path=root / "case2-rtl-popen-boundary.json",
                    boundary_role="verification-question-gated-launcher",
                    spawn=mock.Mock(side_effect=OSError("injected spawn failure")),
                )
            self.assertTrue((root / "authority-consumed.json").is_file())
            self.assertFalse((root / "case2-process-started.json").exists())
            with self.assertRaises(coordinator.DuplicateLaunch):
                coordinator.launch_at_scientific_boundary(
                    state_dir=root,
                    authority_paths=(authority, contract),
                    prompt=prompt,
                    attempt=root / "attempt",
                    worker_command=["python", "--worker"],
                    spawn_command=["python", "--worker"],
                    boundary_record_path=root / "case2-rtl-popen-boundary.json",
                    boundary_role="verification-question-gated-launcher",
                    spawn=mock.Mock(),
                )

    def test_capacity_preflight_failure_does_not_prepare_or_launch(self) -> None:
        DiskUsage = namedtuple("DiskUsage", "total used free")
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                coordinator.shutil,
                "disk_usage",
                return_value=DiskUsage(100, 100, 0),
            ), mock.patch.object(
                coordinator,
                "launch_at_scientific_boundary",
            ) as launch:
                result, failure = coordinator.run_case(
                    case={"id": "verification-question", "prompt": "question"},
                    output=Path(temporary) / "output",
                    state_dir=Path(temporary) / "state",
                    timeout_seconds=10,
                    oracle_timeout_seconds=10,
                    cadence_seconds=1,
                    overall_deadline=100,
                    minimum_free_bytes=1,
                    cancel_requested=lambda: None,
                    authority_paths=None,
                    contract_path=Path(temporary) / "contract.json",
                    approved_bindings={},
                    control_bindings={},
                )
            self.assertIsNone(result)
            self.assertEqual("CAPACITY_FLOOR_NOT_MET_NO_LAUNCH", failure["status"])
            launch.assert_not_called()

    def test_prelaunch_cancellation_does_not_consume_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (result, failure), launch = self.run_case_to_prelaunch_guard(
                root,
                cancellation="injected cancellation",
            )
            self.assertIsNone(result)
            self.assertEqual(
                "CANCELLED_AT_SCIENTIFIC_POPEN_GUARD_NO_LAUNCH_"
                "AUTHORITY_UNCONSUMED",
                failure["status"],
            )
            self.assertEqual("injected cancellation", failure["signal"])
            self.assertFalse((root / "state/authority-consumed.json").exists())
            self.assertFalse((root / "state/case2-launch-ticket.json").exists())
            launch.assert_not_called()

    def test_prelaunch_overall_deadline_does_not_consume_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (result, failure), launch = self.run_case_to_prelaunch_guard(
                root,
                observed_time=100.0,
                overall_deadline=100.0,
            )
            self.assertIsNone(result)
            self.assertEqual(
                "OVERALL_DEADLINE_EXPIRED_AT_SCIENTIFIC_POPEN_GUARD_"
                "NO_LAUNCH_AUTHORITY_UNCONSUMED",
                failure["status"],
            )
            self.assertFalse((root / "state/authority-consumed.json").exists())
            self.assertFalse((root / "state/case2-launch-ticket.json").exists())
            launch.assert_not_called()

    def test_prelaunch_capacity_loss_does_not_consume_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (result, failure), launch = self.run_case_to_prelaunch_guard(
                root,
                current_free_bytes=0,
            )
            self.assertIsNone(result)
            self.assertEqual(
                "CAPACITY_FLOOR_NOT_MET_AT_SCIENTIFIC_POPEN_GUARD_"
                "NO_LAUNCH_AUTHORITY_UNCONSUMED",
                failure["status"],
            )
            self.assertEqual(0, failure["disk_free_bytes"])
            self.assertFalse((root / "state/authority-consumed.json").exists())
            self.assertFalse((root / "state/case2-launch-ticket.json").exists())
            launch.assert_not_called()

    def test_termination_rejects_changed_process_identity(self) -> None:
        identity = {
            "pid": 123,
            "start_ticks": 456,
            "argv": ["python", "--worker"],
            "cwd": str(coordinator.ROOT),
            "session_id": 123,
        }
        with mock.patch.object(coordinator, "identity_live", return_value=False):
            with mock.patch.object(os, "killpg") as killpg:
                with self.assertRaisesRegex(
                    coordinator.CoordinationError,
                    "identity changed",
                ):
                    coordinator.terminate_identity(identity)
        killpg.assert_not_called()

    def test_deadline_terminates_only_exact_identity_bound_worker(self) -> None:
        worker = {
            "pid": 123,
            "start_ticks": 456,
            "argv": ["python", "--worker"],
            "cwd": str(coordinator.ROOT),
        }
        with mock.patch.object(
            coordinator,
            "identity_live",
            return_value=True,
        ), mock.patch.object(
            coordinator.time,
            "monotonic",
            return_value=10.0,
        ), mock.patch.object(
            coordinator,
            "terminate_identity",
        ) as terminate:
            failure = coordinator.monitor_rtl_worker(
                case_id="verification-question",
                worker=worker,
                deadline=9.0,
                cadence_seconds=180,
                heartbeat=mock.Mock(),
                cancel_requested=lambda: None,
            )
        self.assertEqual(
            "BOUNDED_TIMEOUT_NO_RETRY_EXACT_RTL_WORKER_TERMINATED",
            failure["status"],
        )
        terminate.assert_called_once_with(worker)

    def test_launch_readiness_timeout_preserves_live_owner_without_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = root / "contract.json"
            authority = root / "authority.json"
            state = root / "state"
            contract.write_text("{}\n", encoding="utf-8")
            authority.write_text("{}\n", encoding="utf-8")
            process = mock.Mock(pid=123)
            identity = {
                "pid": 123,
                "start_ticks": 456,
                "argv": ["python", "--owner"],
                "cwd": str(coordinator.ROOT),
                "session_id": 123,
            }
            with mock.patch.object(
                coordinator,
                "validate_contract",
                return_value={"output": "injected-output"},
            ), mock.patch.object(
                coordinator,
                "project_path",
                return_value=root / "output",
            ), mock.patch.object(
                coordinator,
                "validate_authority",
                return_value={},
            ), mock.patch.object(
                coordinator,
                "sha256_file",
                return_value="0" * 64,
            ), mock.patch.object(
                coordinator,
                "implementation_binding",
                return_value={},
            ), mock.patch.object(
                coordinator.subprocess,
                "Popen",
                return_value=process,
            ) as popen, mock.patch.object(
                coordinator,
                "process_identity",
                return_value=identity,
            ), mock.patch.object(
                coordinator,
                "identity_live",
                return_value=True,
            ), mock.patch.object(
                coordinator.time,
                "monotonic",
                side_effect=[0.0, 31.0],
            ), mock.patch.object(
                coordinator,
                "terminate_identity",
            ) as terminate:
                result = coordinator.launch(contract, authority, state)
                with self.assertRaises(coordinator.DuplicateLaunch):
                    coordinator.launch(contract, authority, state)

            self.assertEqual(
                "DETACHED_COORDINATOR_OWNER_RUNNING_READINESS_PENDING",
                result["status"],
            )
            self.assertEqual("PENDING", result["readiness"])
            self.assertFalse(result["automatic_retry"])
            self.assertEqual(identity["pid"], result["pid"])
            self.assertEqual(identity["start_ticks"], result["start_ticks"])
            self.assertTrue((state / "owner-process.json").is_file())
            self.assertFalse((state / "terminal.json").exists())
            terminate.assert_not_called()
            popen.assert_called_once()

    def test_status_reports_owner_loss_without_automatic_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            owner = {
                "schema": coordinator.PROCESS_SCHEMA,
                "pid": 1,
                "start_ticks": 1,
                "argv": ["owner"],
                "cwd": str(coordinator.ROOT),
            }
            coordinator.write_exclusive_json(
                state / "owner-process.json",
                owner,
            )
            with mock.patch.object(coordinator, "identity_live", return_value=False):
                result = coordinator.status(state)
            self.assertEqual("TERMINAL", result["state"])
            self.assertEqual(
                "OWNER_LOST_NO_LIVE_CHILD_TERMINAL_NO_AUTOMATIC_RETRY",
                result["terminal"]["status"],
            )
            self.assertEqual(owner, result["terminal"]["owner"])
            self.assertEqual([], result["terminal"]["live_children_at_settlement"])
            self.assertEqual(
                "PERMANENTLY_FORBIDDEN",
                result["terminal"]["retry_replay_relaunch"],
            )
            self.assertTrue((state / "terminal.json").is_file())
            self.assertFalse(result["automatic_retry"])

    def test_owner_loss_preserves_live_child_and_blocks_relaunch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            state.mkdir()
            prompt = root / "prompt.utf8"
            prompt.write_text("question", encoding="utf-8")
            owner = {
                "schema": coordinator.PROCESS_SCHEMA,
                "pid": 1,
                "start_ticks": 1,
                "argv": ["owner"],
                "cwd": str(coordinator.ROOT),
            }
            child = {
                "pid": 2,
                "start_ticks": 3,
                "argv": ["python", "--worker"],
                "cwd": str(coordinator.ROOT),
                "session_id": 2,
                "state": "S",
            }
            worker_command = [
                str(Path(os.sys.executable).resolve()),
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
            ]
            boundary_path = (
                state / "verification-question-rtl-popen-boundary.json"
            )
            boundary_role = "verification-question-gated-launcher"
            spawn_command = coordinator.gate_launch_command(
                gate_path=state / "verification-question-science-gate.json",
                boundary_record_path=boundary_path,
                boundary_role=boundary_role,
                prompt=prompt,
                output=root / "output",
                worker_command=worker_command,
            )
            child["argv"] = spawn_command
            coordinator.write_exclusive_json(state / "owner-process.json", owner)
            owner_lost = False

            def inject_owner_loss_before_boundary_write() -> None:
                nonlocal owner_lost
                self.assertFalse(boundary_path.exists())
                owner_lost = True

            with mock.patch.object(
                coordinator,
                "process_identity",
                return_value=child,
            ), mock.patch.object(
                coordinator.time,
                "monotonic",
                side_effect=[0.0, 61.0],
            ):
                with self.assertRaisesRegex(
                    coordinator.CoordinationError,
                    "scientific execution gate was not released",
                ):
                    coordinator.gate_exec(
                        state / "verification-question-science-gate.json",
                        boundary_path,
                        boundary_role,
                        prompt,
                        root / "output",
                        worker_command,
                        before_boundary_write=inject_owner_loss_before_boundary_write,
                    )
            self.assertTrue(owner_lost)
            self.assertFalse(
                (state / "verification-question-gate-process.json").exists()
            )
            boundary_child = coordinator.load_json(boundary_path)
            self.assertEqual(
                "DETACHED_CHILD_PUBLISHED_IDENTITY_BEFORE_GATE_RELEASE",
                boundary_child["status"],
            )
            with mock.patch.object(
                coordinator,
                "identity_live",
                side_effect=lambda identity: identity["pid"] == child["pid"],
            ):
                result = coordinator.status(state)

            self.assertEqual("TERMINAL", result["state"])
            self.assertEqual(
                "OWNER_LOST_WITH_LIVE_CHILD_TERMINAL_NO_AUTOMATIC_RETRY",
                result["terminal"]["status"],
            )
            self.assertEqual(
                [boundary_child],
                result["terminal"]["live_children_at_settlement"],
            )
            self.assertEqual([boundary_child], result["live_children"])
            terminal_bytes = (state / "terminal.json").read_bytes()

            contract = root / "contract.json"
            authority = root / "authority.json"
            contract.write_text("{}\n", encoding="utf-8")
            authority.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                coordinator,
                "validate_contract",
                return_value={"output": "injected-output"},
            ), mock.patch.object(
                coordinator,
                "validate_authority",
                return_value={},
            ), mock.patch.object(
                coordinator,
                "project_path",
                return_value=root / "output",
            ), mock.patch.object(
                coordinator.subprocess,
                "Popen",
            ) as popen:
                with self.assertRaisesRegex(
                    coordinator.DuplicateLaunch,
                    "state namespace already exists",
                ):
                    coordinator.launch(contract, authority, state)
            popen.assert_not_called()

            with mock.patch.object(coordinator, "identity_live", return_value=False):
                settled = coordinator.status(state)
            self.assertEqual([], settled["live_children"])
            self.assertEqual(
                [boundary_child],
                settled["terminal"]["live_children_at_settlement"],
            )
            self.assertEqual(terminal_bytes, (state / "terminal.json").read_bytes())

    def test_terminal_status_keeps_live_child_visible_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            coordinator.write_exclusive_json(
                state / "terminal.json",
                {
                    "schema": coordinator.TERMINAL_SCHEMA,
                    "status": "FAILED_NO_AUTOMATIC_RETRY",
                },
            )
            coordinator.write_exclusive_json(
                state / "verification-question-inner-worker-process.json",
                {"pid": 2, "start_ticks": 3},
            )
            with mock.patch.object(coordinator, "identity_live", return_value=True):
                result = coordinator.status(state)
            self.assertEqual("TERMINAL", result["state"])
            self.assertEqual(1, len(result["live_children"]))
            self.assertFalse(result["automatic_retry"])

    def test_combined_coverage_requires_two_new_and_one_preserved(self) -> None:
        coverage = {
            "configured_prompts": 3,
            "completed_prompts": 3,
            "fresh_rtl_prompts": 2,
            "generated_tokens": 12,
            "continuous_24_layer_kv_growth_prompts": 3,
            "decode_transitions_checked": 9,
            "decode_layer_growth_checks": 216,
            "integer_byte_mismatches": 0,
            "selected_token_mismatches": 0,
            "software_transformer_or_logits_fallback": False,
            "kv_append_bytes_compared": 1,
            "full_cache_bytes_compared": 1,
            "quantized_layer_output_bytes_compared": 1,
            "full_vocabulary_logit_bytes_compared": 1,
            "full_vocabulary_head_steps_checked": 12,
        }
        coordinator.validate_combined_coverage({"coverage": coverage})
        coverage["fresh_rtl_prompts"] = 3
        with self.assertRaisesRegex(
            coordinator.CoordinationError,
            "preserve exactly one",
        ):
            coordinator.validate_combined_coverage({"coverage": coverage})


if __name__ == "__main__":
    unittest.main()
