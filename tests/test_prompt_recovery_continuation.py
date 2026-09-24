#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prompt_recovery_continuation",
    ROOT / "scripts/run_prompt_recovery_continuation.py",
)
assert SPEC is not None and SPEC.loader is not None
continuation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continuation)


class PromptRecoveryContinuationTest(unittest.TestCase):
    def test_cli_exposes_no_original_case2_launch_or_authority_issuer(self) -> None:
        parser = continuation.build_parser()
        choices = next(
            action.choices
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(
            {
                "launch-continuation",
                "_continue_worker",
                "_gate_exec",
                "continuation-status",
            },
            set(choices),
        )

    def test_deadline_derivation_is_same_domain_and_conservative(self) -> None:
        source = ROOT / (
            "reports/verification/prompt-recovery-controller-custody-0001/"
            "deadline-derivation.json"
        )
        derivation = json.loads(source.read_text(encoding="utf-8"))
        expected_margin = (
            derivation["max_boottime_minus_monotonic_upper_seconds"]
            + derivation["tick_quantization_seconds"]
            + derivation["max_sampling_skew_seconds"]
            + derivation["extra_safety_margin_seconds"]
        )
        self.assertAlmostEqual(
            expected_margin,
            derivation["total_subtracted_margin_seconds"],
            places=15,
        )
        authority = {
            "deadline_derivation": {
                "path": str(source),
                "sha256": continuation.sha256_file(source),
            },
            "absolute_deadline_monotonic_seconds": derivation[
                "absolute_deadline_monotonic_seconds"
            ],
        }
        with mock.patch.object(
            continuation.time,
            "monotonic",
            return_value=derivation["absolute_deadline_monotonic_seconds"] - 1,
        ):
            self.assertEqual(
                derivation["absolute_deadline_monotonic_seconds"],
                continuation.validate_inherited_deadline(authority),
            )

    def test_deadline_expiry_permanently_rejects(self) -> None:
        source = ROOT / (
            "reports/verification/prompt-recovery-controller-custody-0001/"
            "deadline-derivation.json"
        )
        derivation = json.loads(source.read_text(encoding="utf-8"))
        authority = {
            "deadline_derivation": {
                "path": str(source),
                "sha256": continuation.sha256_file(source),
            },
            "absolute_deadline_monotonic_seconds": derivation[
                "absolute_deadline_monotonic_seconds"
            ],
        }
        with mock.patch.object(
            continuation.time,
            "monotonic",
            return_value=derivation["absolute_deadline_monotonic_seconds"],
        ):
            with self.assertRaisesRegex(
                continuation.CoordinationError,
                "inherited overall deadline expired",
            ):
                continuation.validate_inherited_deadline(authority)

    def _adoption_fixture(
        self,
        root: Path,
        *,
        terminal_error: str = "reviewed bytes changed before scientific exec gate",
        owner_live: bool = False,
        create_oracle_result: bool = False,
        create_case3: bool = False,
    ) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
        original_state = root / "original-state"
        continuation_state = root / "continuation-state"
        original_state.mkdir()
        continuation_state.mkdir()
        terminal = {
            "status": "FAILED_NO_AUTOMATIC_RETRY",
            "error": terminal_error,
            "authority_consumed": True,
            "case2_launch_attempt_count": 1,
            "case2_retry_count": 0,
        }
        (original_state / "terminal.json").write_text(json.dumps(terminal))
        (original_state / "owner-process.json").write_text(
            json.dumps({"pid": 10, "start_ticks": 20})
        )
        output_relative = Path(root.name) / "scientific-output"
        output = root / "scientific-output"
        attempt = output / "cases/case2/rtl-attempt"
        attempt.mkdir(parents=True)
        (attempt / "attempt-result.json").write_text(json.dumps({"status": "PASS"}))
        (attempt / "worker-result.json").write_text(
            json.dumps({"readability": {"accepted": True}})
        )
        (attempt / "TREE_ROOT.sha256").write_text("fixture\n")
        if create_oracle_result:
            oracle = output / "cases/case2/full-chain-oracle"
            oracle.mkdir()
            (oracle / "result.json").write_text("{}")
        if create_case3:
            (output / "cases/case3").mkdir()
        contract_path = root / "contract.json"
        contract = {
            "output": output_relative.as_posix(),
            "case2_id": "case2",
            "case3_id": "case3",
            "minimum_launch_free_bytes": 1,
        }
        contract_path.write_text(json.dumps(contract))
        authority = {
            "original_state_dir": str(original_state),
            "contract": {"path": str(contract_path)},
        }
        validated = {"deadline": 123.0, "bindings": {}}
        patches = {
            "validate_continuation_authority": mock.patch.object(
                continuation,
                "validate_continuation_authority",
                return_value=(authority, validated),
            ),
            "identity_live": mock.patch.object(
                continuation,
                "identity_live",
                return_value=owner_live,
            ),
            "validate_original_receipts": mock.patch.object(
                continuation,
                "validate_original_receipts",
            ),
            "live_child_identities": mock.patch.object(
                continuation,
                "live_child_identities",
                return_value=[],
            ),
            "verify_attempt_seal": mock.patch.object(
                continuation,
                "verify_attempt_seal",
            ),
        }
        return original_state, continuation_state, authority, patches

    def test_adoption_accepts_only_exact_fail_closed_boundary(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            _original, state, _authority, patches = self._adoption_fixture(root)
            with (
                patches["validate_continuation_authority"],
                patches["identity_live"],
                patches["validate_original_receipts"],
                patches["live_child_identities"],
                patches["verify_attempt_seal"],
            ):
                authority, contract, record = continuation.validate_adoption(
                    root / "authority.json",
                    state,
                )
            self.assertEqual("case2", contract["case2_id"])
            self.assertEqual(
                "ADOPTED_AFTER_ORIGINAL_OWNER_LOCK_RELEASE_AND_FAIL_CLOSED_TERMINAL",
                record["status"],
            )
            self.assertFalse(record["case2_rtl_replayed"])
            self.assertEqual(state.parent / "original-state", Path(authority["original_state_dir"]))

    def test_adoption_rejects_nonmatching_terminal(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            _original, state, _authority, patches = self._adoption_fixture(
                root,
                terminal_error="different failure",
            )
            with (
                patches["validate_continuation_authority"],
                patches["identity_live"],
            ):
                with self.assertRaisesRegex(
                    continuation.CoordinationError,
                    "terminal error",
                ):
                    continuation.validate_adoption(root / "authority.json", state)
            self.assertFalse((state / "adoption.json").exists())

    def test_adoption_rejects_live_original_owner(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            _original, state, _authority, patches = self._adoption_fixture(
                root,
                owner_live=True,
            )
            with (
                patches["validate_continuation_authority"],
                patches["identity_live"],
            ):
                with self.assertRaisesRegex(
                    continuation.CoordinationError,
                    "original owner is still live",
                ):
                    continuation.validate_adoption(root / "authority.json", state)
            self.assertFalse((state / "adoption.json").exists())

    def test_adoption_rejects_existing_oracle_or_case3(self) -> None:
        for oracle, case3, message in (
            (True, False, "case2 oracle already executed"),
            (False, True, "case3 already exists"),
        ):
            with self.subTest(oracle=oracle, case3=case3):
                with tempfile.TemporaryDirectory(dir=ROOT) as raw:
                    root = Path(raw)
                    _original, state, _authority, patches = self._adoption_fixture(
                        root,
                        create_oracle_result=oracle,
                        create_case3=case3,
                    )
                    with (
                        patches["validate_continuation_authority"],
                        patches["identity_live"],
                        patches["validate_original_receipts"],
                        patches["live_child_identities"],
                        patches["verify_attempt_seal"],
                    ):
                        with self.assertRaisesRegex(
                            continuation.CoordinationError,
                            message,
                        ):
                            continuation.validate_adoption(
                                root / "authority.json",
                                state,
                            )
                    self.assertFalse((state / "adoption.json").exists())

    def test_child_publishes_identity_before_waiting_for_gate(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            prompt = root / "prompt"
            prompt.write_text("prompt")
            worker = [
                str(Path(os.sys.executable).resolve()),
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
            ]
            child = {
                "pid": 123,
                "state": "R",
                "ppid": 1,
                "start_ticks": 456,
                "argv": [
                    str(Path(os.sys.executable).resolve()),
                    "-B",
                    str(Path(continuation.__file__).resolve()),
                ],
                "cwd": str(ROOT),
                "session_id": 123,
            }
            boundary = root / "boundary.json"
            observed: list[bool] = []

            def before_write() -> None:
                observed.append(boundary.exists())

            with mock.patch.object(continuation, "process_identity", return_value=child):
                record = continuation.publish_child_boundary_identity(
                    boundary_record_path=boundary,
                    boundary_role="test-gate",
                    prompt=prompt,
                    output=root / "output",
                    worker_command=worker,
                    before_boundary_write=before_write,
                )
            self.assertEqual([False], observed)
            self.assertTrue(boundary.is_file())
            self.assertEqual(
                "DETACHED_CHILD_PUBLISHED_IDENTITY_BEFORE_GATE_RELEASE",
                record["status"],
            )

    def test_arbitrary_authority_path_is_rejected_before_loading(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            path = Path(raw) / "fabricated-authority.json"
            path.write_text("{}")
            with self.assertRaisesRegex(
                continuation.CoordinationError,
                "authority path is not trusted",
            ):
                continuation.validate_continuation_authority(path)

    def test_gate_rejects_case2_rtl_stage(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            authority = root / "authority.json"
            authority.write_text("{}")
            prompt = root / "prompt"
            prompt.write_text("case2")
            capability = root / "capability.json"
            gate = root / "gate.json"
            boundary = root / "boundary.json"
            output = root / "output"
            worker = [
                str(Path(os.sys.executable).resolve()),
                "-B",
                str(ROOT / "tools/ace2_v73_stage1_chat_attempt.py"),
            ]
            capability.write_text(
                json.dumps(
                    {
                        "schema": "ace2-prompt-recovery-continuation-gate-capability-v1",
                        "status": "ISSUED_UNDER_CONTINUATION_OWNER_AND_LAUNCH_LOCK",
                        "authority_path": str(authority),
                        "authority_sha256": continuation.sha256_file(authority),
                        "stage": "case2-rtl",
                        "case_id": "verification-question",
                        "gate_path": str(gate),
                        "boundary_record_path": str(boundary),
                        "boundary_role": "case2-gate",
                        "prompt": str(prompt),
                        "prompt_sha256": continuation.sha256_file(prompt),
                        "output": str(output),
                        "worker_argv": worker,
                        "worker_argv_sha256": continuation.sha256_bytes(
                            continuation.canonical_bytes(worker)
                        ),
                    }
                )
            )
            with (
                mock.patch.object(
                    continuation,
                    "EXPECTED_CONTINUATION_STATE",
                    root,
                ),
                mock.patch.object(
                    continuation,
                    "EXPECTED_CONTINUATION_AUTHORITY",
                    authority,
                ),
            ):
                with self.assertRaisesRegex(
                    continuation.CoordinationError,
                    "gate stage is not allowlisted",
                ):
                    continuation.validate_gate_capability(
                        capability_path=capability,
                        gate_path=gate,
                        boundary_record_path=boundary,
                        boundary_role="case2-gate",
                        prompt=prompt,
                        output=output,
                        worker_command=worker,
                    )

    def test_direct_worker_without_parent_intent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            state = Path(raw)
            authority_path = state / "authority.json"
            authority_path.write_text("{}")
            authority = {"original_state_dir": str(state / "original")}
            with (
                mock.patch.object(
                    continuation,
                    "EXPECTED_CONTINUATION_STATE",
                    state,
                ),
                mock.patch.object(
                    continuation,
                    "EXPECTED_CONTINUATION_AUTHORITY",
                    authority_path,
                ),
                mock.patch.object(
                    continuation,
                    "validate_continuation_authority",
                    return_value=(authority, {"deadline": 100.0, "bindings": {}}),
                ),
            ):
                with self.assertRaisesRegex(
                    continuation.CoordinationError,
                    "missing regular JSON",
                ):
                    continuation.run_continuation_owner(authority_path, state)

    def test_adopted_oracle_post_popen_failure_reaps_exact_child(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            output = root / "output"
            case_dir = output / "cases/case2"
            attempt = case_dir / "rtl-attempt"
            attempt.mkdir(parents=True)
            prompt = attempt / "prompt.utf8"
            prompt.write_text("case2")
            manifest = {
                "source_and_rtl": {"sha256": "source"},
                "execution_dependency_binding": {"sha256": "dependencies"},
            }
            (attempt / "attempt-manifest.json").write_text(json.dumps(manifest))
            (attempt / "TREE_ROOT.sha256").write_text("fixture\n")
            state = root / "state"
            state.mkdir()
            process = mock.Mock(pid=123)
            identity = {
                "pid": 123,
                "start_ticks": 456,
                "argv": ["gate"],
                "cwd": str(ROOT),
                "session_id": 123,
            }
            with (
                mock.patch.object(continuation, "verify_attempt_seal"),
                mock.patch.object(continuation, "validate_execution_gate_bindings"),
                mock.patch.object(continuation, "create_gate_capability"),
                mock.patch.object(
                    continuation,
                    "launch_at_scientific_boundary",
                    return_value=(process, identity, ((0.0, 0.0), 0.0)),
                ),
                mock.patch.object(
                    continuation,
                    "record_process",
                    side_effect=continuation.CoordinationError("post-Popen failure"),
                ),
                mock.patch.object(
                    continuation,
                    "terminate_and_reap",
                ) as terminate,
            ):
                with self.assertRaisesRegex(
                    continuation.CoordinationError,
                    "post-Popen failure",
                ):
                    continuation.run_adopted_case2_oracle(
                        case={"id": "case2"},
                        output=output,
                        state_dir=state,
                        overall_deadline=continuation.time.monotonic() + 60,
                        oracle_timeout_seconds=10,
                        cadence_seconds=1,
                        minimum_free_bytes=1,
                        contract_path=root / "contract.json",
                        approved_bindings={},
                        control_bindings={
                            "continuation_authority_path": str(
                                continuation.EXPECTED_CONTINUATION_AUTHORITY
                            )
                        },
                        original_bindings={},
                    )
            terminate.assert_called_once()

    def test_final_success_cannot_publish_after_inherited_deadline(self) -> None:
        with mock.patch.object(
            continuation,
            "publish_continuation_terminal",
        ) as publish:
            with self.assertRaisesRegex(
                continuation.CoordinationError,
                "overall deadline expired before continuation final",
            ):
                continuation.finalize_continuation_success(
                    authority_path=Path("/authority"),
                    authority={},
                    contract={},
                    adoption={},
                    output=Path("/output"),
                    config_path=Path("/config"),
                    cases={},
                    case2={},
                    case3={},
                    state_dir=Path("/state"),
                    overall_deadline=continuation.time.monotonic(),
                )
        publish.assert_not_called()

    def test_adopted_oracle_cannot_popen_at_inherited_deadline(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            root = Path(raw)
            output = root / "output"
            attempt = output / "cases/case2/rtl-attempt"
            attempt.mkdir(parents=True)
            (attempt / "prompt.utf8").write_text("case2")
            (attempt / "attempt-manifest.json").write_text(
                json.dumps(
                    {
                        "source_and_rtl": {"sha256": "source"},
                        "execution_dependency_binding": {
                            "sha256": "dependencies"
                        },
                    }
                )
            )
            (attempt / "TREE_ROOT.sha256").write_text("fixture\n")
            state = root / "state"
            state.mkdir()
            with (
                mock.patch.object(continuation, "verify_attempt_seal"),
                mock.patch.object(
                    continuation.subprocess,
                    "Popen",
                ) as popen,
            ):
                with self.assertRaisesRegex(
                    continuation.CoordinationError,
                    "overall deadline expired before adopted case2 oracle",
                ):
                    continuation.run_adopted_case2_oracle(
                        case={"id": "case2"},
                        output=output,
                        state_dir=state,
                        overall_deadline=continuation.time.monotonic(),
                        oracle_timeout_seconds=10,
                        cadence_seconds=1,
                        minimum_free_bytes=1,
                        contract_path=root / "contract.json",
                        approved_bindings={},
                        control_bindings={
                            "continuation_authority_path": str(
                                continuation.EXPECTED_CONTINUATION_AUTHORITY
                            )
                        },
                        original_bindings={},
                    )
            popen.assert_not_called()

    def test_deadline_handoff_reaps_child_before_restoring_alarm(self) -> None:
        process = mock.Mock()
        identity = {"pid": 4321, "start_ticks": 1234}
        with (
            mock.patch.object(
                continuation.time,
                "monotonic",
                return_value=101.0,
            ),
            mock.patch.object(
                continuation,
                "terminate_and_reap",
            ) as terminate,
            mock.patch.object(continuation.signal, "setitimer") as setitimer,
        ):
            with self.assertRaisesRegex(
                continuation.CoordinationError,
                "absolute deadline expired during raw Popen",
            ):
                continuation.resume_deadline_after_child_handoff(
                    ((0.5, 0.0), 100.0),
                    process,
                    identity,
                    [["launcher"], ["worker"]],
                )
        terminate.assert_called_once_with(
            process,
            identity,
            [["launcher"], ["worker"]],
        )
        setitimer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
