from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scripts import verify_full_chain_independent_oracle as full_chain_oracle
from tools import rtl_arbitrary_text_generation_backend as rtl_backend
from tools.run_rtl_arbitrary_text_generation import (
    CANONICAL_HELLO_CHAT_TOKEN_IDS,
    CHAT_TEMPLATE_SHA256,
    REVISION,
    OfficialAuthorizationError,
    PreflightError,
    assert_no_machine_private_paths,
    canonical_chat_token_ids,
    directory_tree_record,
    file_record,
    fresh_l2_authorization_contract,
    identity_sha256,
    parse_sha256_manifest,
    records_aggregate,
    read_prompt,
    require_aggregate_match,
    resolve_snapshot,
    snapshot_candidates,
    runtime_source_aggregate,
    run_official,
    tokenizer_record,
    validate_generation_bounds,
    validate_max_new_tokens,
    verify_directory_identity,
    write_sha256_manifest,
)
from tools.rtl_arbitrary_text_generation_backend import (
    BackendError,
    ProcessTreeRssTracker,
    SimulatorProcessError,
    compile_and_run,
    display_command,
    included_execution_source,
    process_tree_rss_tracking,
    render_residual_vectors,
    render_softmax_vectors,
    residual_execution_source,
    runtime_path,
    timing_record,
    tokenizer_domain_decision,
    tracked_run,
    validate_context_request,
)


OFFICIAL_ATTEMPT = Path(
    "evidence/verification/rtl-arbitrary-text-generation-v1/attempt-0001"
).resolve()
PROJECTION_FIXTURE = Path(
    "verification/fixtures/projection_position00_layer23_v.json"
).resolve()


def official_attempt_identity() -> tuple[str, str, str] | None:
    if not OFFICIAL_ATTEMPT.exists():
        return None
    return tuple(
        hashlib.sha256((OFFICIAL_ATTEMPT / name).read_bytes()).hexdigest()
        for name in ("status.json", "manifest.json", "SHA256SUMS")
    )


class ProjectionStdoutContractTest(unittest.TestCase):
    def projection_case(self) -> tuple[dict, list[str], dict]:
        fixture = json.loads(PROJECTION_FIXTURE.read_text(encoding="utf-8"))
        baseline_v = bytes.fromhex(fixture["baseline_v_u8_hex"])
        corrected_v = bytes.fromhex(fixture["corrected_v_u8_hex"])
        self.assertEqual(hashlib.sha256(baseline_v).hexdigest(), fixture["baseline_v_u8_sha256"])
        self.assertEqual(hashlib.sha256(corrected_v).hexdigest(), fixture["corrected_v_u8_sha256"])
        self.assertEqual(
            sum(left != right for left, right in zip(baseline_v, corrected_v, strict=True)),
            fixture["baseline_corrected_mismatch_count"],
        )

        projections = {}
        lines = []
        for operator, key in enumerate(rtl_backend.PROJECTION_ORDER):
            channels = rtl_backend.PROJECTION_CHANNELS[key]
            outputs = (
                list(baseline_v)
                if key == "v"
                else [(operator * 17 + channel) & 0xFF for channel in range(channels)]
            )
            accumulators = [-(operator * 1000 + channel) for channel in range(channels)]
            if key == "q":
                accumulators[0] = -41
            elif key == "v":
                accumulators[0] = -286
            for channel, (output, accumulator) in enumerate(
                zip(outputs, accumulators, strict=True)
            ):
                lines.append(
                    "RTL_PROJ_RESULT "
                    f"operator={operator} channel={channel} out_u8={output} "
                    f"acc={accumulator} saturation=0 overflow=0"
                )
            projection = {
                "output_q": torch.tensor(
                    list(corrected_v) if key == "v" else outputs,
                    dtype=torch.int64,
                ),
                "accumulator": torch.tensor(accumulators, dtype=torch.int64),
                "saturation": torch.zeros(channels, dtype=torch.int64),
            }
            if key == "v":
                projection["baseline_output_q"] = torch.tensor(outputs, dtype=torch.int64)
            projections[key] = projection
        self.assertEqual(lines[0], fixture["archived_rows"]["q_channel_0"])
        self.assertEqual(
            lines[sum(rtl_backend.PROJECTION_CHANNELS[key] for key in ("q", "k"))],
            fixture["archived_rows"]["v_channel_0"],
        )
        return {"projections": projections}, lines, fixture

    def test_archived_baseline_rows_are_verified_before_sidecar_correction(self) -> None:
        position, lines, fixture = self.projection_case()
        stdout = "\n".join(lines)
        rows = rtl_backend.parse_projection_stdout(stdout)
        self.assertEqual(rows[1024], (2, 0, 238, -286, 0, 0))
        self.assertEqual(fixture["baseline_corrected_mismatch_count"], 69)
        corrected_only_position = {
            "projections": {
                key: dict(projection)
                for key, projection in position["projections"].items()
            }
        }
        corrected_only_position["projections"]["v"].pop("baseline_output_q")
        with self.assertRaisesRegex(BackendError, r"'output': 69"):
            rtl_backend.verify_projection_stdout(stdout, corrected_only_position)
        self.assertEqual(
            rtl_backend.verify_projection_stdout(stdout, position),
            {
                "results": 12672,
                "address_mismatches": 0,
                "output_mismatches": 0,
                "accumulator_mismatches": 0,
                "saturation_mismatches": 0,
                "overflow_mismatches": 0,
            },
        )

    def test_projection_parser_rejects_invalid_rows(self) -> None:
        _position, valid_lines, _fixture = self.projection_case()
        invalid_cases = {}

        missing = valid_lines.copy()
        missing.pop(0)
        invalid_cases["missing"] = missing

        duplicate = valid_lines.copy()
        duplicate[1] = duplicate[0]
        invalid_cases["duplicate"] = duplicate

        malformed = valid_lines.copy()
        malformed[0] += " trailing"
        invalid_cases["malformed"] = malformed

        out_of_range = valid_lines.copy()
        out_of_range[0] = out_of_range[0].replace("out_u8=0", "out_u8=256")
        invalid_cases["out_of_range"] = out_of_range

        operator_misaligned = valid_lines.copy()
        operator_misaligned[0] = operator_misaligned[0].replace("operator=0", "operator=1")
        invalid_cases["operator_misaligned"] = operator_misaligned

        channel_misaligned = valid_lines.copy()
        channel_misaligned[0] = channel_misaligned[0].replace("channel=0", "channel=1")
        invalid_cases["channel_misaligned"] = channel_misaligned

        for name, lines in invalid_cases.items():
            with self.subTest(name=name), self.assertRaises(BackendError):
                rtl_backend.parse_projection_stdout("\n".join(lines))


class RtlArbitraryTextGenerationPreflightTest(unittest.TestCase):
    def test_transient_artifacts_are_pruned_after_claim_evidence_is_durable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            working = Path(temporary)
            retained = {
                working / "rtl_execution.json": b'{"status":"PASS"}\n',
                working / "logs/simulator.stdout.log": b"RTL_PASS\n",
                working / "rtl_observed/cache.bin": bytes(range(16)),
            }
            transient = {
                working / "vectors/input.hex": b"00\n",
                working / "tensors/reference.bin": b"\x01\x02",
                working / "execution_sources/testbench.sv": b"module tb; endmodule\n",
                working / "sim/test.vvp": b"compiled simulator",
            }
            for path, raw in (retained | transient).items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)

            result = rtl_backend.prune_transient_execution_artifacts(working)

            self.assertEqual(
                result["status"],
                "TRANSIENT_ARTIFACTS_PRUNED_AFTER_DURABLE_RESULT",
            )
            self.assertEqual(result["removed_files"], len(transient))
            self.assertEqual(
                result["removed_logical_bytes"],
                sum(len(raw) for raw in transient.values()),
            )
            for path, raw in retained.items():
                self.assertEqual(path.read_bytes(), raw)
            for path in transient:
                self.assertFalse(path.exists())

    def test_bounded_generation_count_accepts_two_through_four(self) -> None:
        self.assertEqual(validate_max_new_tokens(2), 2)
        self.assertEqual(validate_max_new_tokens(3), 3)
        self.assertEqual(validate_max_new_tokens(4), 4)
        for value in (0, 1, 5, 256):
            with self.subTest(value=value), self.assertRaises(PreflightError):
                validate_max_new_tokens(value)

    def test_prompt_is_arbitrary_utf8_but_not_blank(self) -> None:
        self.assertEqual(read_prompt("Grüße, 世界", None), "Grüße, 世界")
        with self.assertRaises(PreflightError):
            read_prompt(" \n\t", None)

    def test_sha256_manifest_rejects_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "SHA256SUMS"
            manifest.write_text(f"{'0' * 64}  ../escape\n", encoding="utf-8")
            with self.assertRaises(PreflightError):
                parse_sha256_manifest(root, manifest)

    def test_sha256_manifest_accepts_safe_relative_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = root / "member.bin"
            member.write_bytes(b"ace2")
            digest = hashlib.sha256(member.read_bytes()).hexdigest()
            manifest = root / "SHA256SUMS"
            manifest.write_text(f"{digest}  member.bin\n", encoding="utf-8")
            self.assertEqual(parse_sha256_manifest(root, manifest), [(digest, member)])

    def test_prompt_and_feedback_share_the_rtl_context_bound(self) -> None:
        self.assertEqual(
            validate_generation_bounds(30, 30, 4),
            {
                "prompt_token_count": 30,
                "max_prompt_tokens": 30,
                "max_new_tokens": 4,
                "total_context_tokens": 34,
                "processed_positions": 33,
                "rtl_context_bound": 43,
            },
        )
        self.assertEqual(validate_generation_bounds(39, 39, 4)["total_context_tokens"], 43)
        with self.assertRaises(PreflightError):
            validate_generation_bounds(40, 40, 4)

    def test_backend_rejects_context_overflow_before_execution(self) -> None:
        self.assertEqual(
            validate_context_request([1] * 36, 4)["total_context_tokens"],
            40,
        )
        self.assertEqual(
            validate_context_request([1] * 39, 4)["total_context_tokens"],
            43,
        )
        with self.assertRaises(BackendError):
            validate_context_request([1] * 40, 4)

    def test_canonical_qwen_hello_chat_template_identity(self) -> None:
        self.assertEqual(
            CHAT_TEMPLATE_SHA256,
            "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
        )
        self.assertEqual(len(CANONICAL_HELLO_CHAT_TOKEN_IDS), 30)
        snapshot = resolve_snapshot()
        record, tokenizer = tokenizer_record("Hello", 30, 4, snapshot)
        self.assertEqual(
            canonical_chat_token_ids(tokenizer, "Hello"),
            list(CANONICAL_HELLO_CHAT_TOKEN_IDS),
        )
        self.assertEqual(
            record["canonical_hello_chat_token_ids"],
            list(CANONICAL_HELLO_CHAT_TOKEN_IDS),
        )
        self.assertEqual(record["generation_bounds"]["total_context_tokens"], 34)

    def test_generated_artifact_paths_are_repository_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            external = Path(temporary) / "generated" / "artifact.svh"
            logical = runtime_path(external)
            self.assertFalse(Path(logical).is_absolute())
            self.assertNotIn("/home/", logical)
            self.assertEqual((Path.cwd() / logical).resolve(), external.resolve())
            artifact_root = Path(temporary) / "artifact-root"
            artifact_root.mkdir()
            (artifact_root / "record.json").write_text(
                json.dumps({"path": logical}) + "\n", encoding="utf-8"
            )
            assert_no_machine_private_paths(artifact_root)
            (artifact_root / "record.json").write_text(
                json.dumps({"path": str(Path.home() / "private")}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(PreflightError):
                assert_no_machine_private_paths(artifact_root)

    def test_process_tree_paths_have_no_absolute_or_parent_escape(self) -> None:
        command = display_command(
            [
                str(Path("rtl/ace2_shell.sv").resolve()),
                "/usr/bin/iverilog",
                "-g2012",
            ]
        )
        self.assertEqual(command, ["rtl/ace2_shell.sv", "iverilog", "-g2012"])
        for value in command:
            self.assertFalse(Path(value).is_absolute())
            self.assertNotIn("..", Path(value).parts)

    def test_snapshot_resolution_is_explicit_or_cache_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / REVISION
            snapshot.mkdir()
            for name in (
                "model.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ):
                (snapshot / name).write_bytes(b"x")
            self.assertEqual(resolve_snapshot(snapshot), snapshot.resolve())
            with mock.patch.dict(os.environ, {"ACE2_QWEN25_SNAPSHOT": str(snapshot)}, clear=False):
                self.assertEqual(resolve_snapshot(), snapshot.resolve())

    def test_bf16_portable_snapshot_precedes_cache_candidates(self) -> None:
        expected = (
            Path(__file__).resolve().parents[1]
            / "build/ace2_chat_demo/cf20-bf16-portable-snapshot"
            / REVISION
        ).resolve()
        with mock.patch.dict(os.environ, {"ACE2_QWEN25_SNAPSHOT": ""}, clear=False):
            self.assertEqual(snapshot_candidates()[0], expected)

    def test_tokenizer_domain_policy_rejects_unmapped_and_empty_rows(self) -> None:
        class Tokenizer:
            vocab_size = 10

            def __len__(self) -> int:
                return 12

            def decode(self, token_ids, **_kwargs):
                return "A" if token_ids == [3] else ""

        record = tokenizer_domain_decision(Tokenizer(), 3)
        self.assertTrue(record["feedback_authorized"])
        with self.assertRaises(BackendError):
            tokenizer_domain_decision(Tokenizer(), 12)
        with self.assertRaises(BackendError):
            tokenizer_domain_decision(Tokenizer(), 4)

    def test_manifest_covers_nested_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "nested/b.txt").write_text("b", encoding="utf-8")
            sums = write_sha256_manifest(root)
            self.assertEqual(
                [path.relative_to(root).as_posix() for _, path in parse_sha256_manifest(root, sums)],
                ["a.txt", "nested/b.txt"],
            )

    def test_runtime_source_aggregate_binds_exact_helper_and_rtl_inputs(self) -> None:
        aggregate = runtime_source_aggregate()
        paths = {record["path"] for record in aggregate["records"]}
        self.assertIn("tools/run_lora_v4_two_token_24layer_rtl_authorized.py", paths)
        self.assertNotIn("tools/run_lora_v4_two_token_24layer_rtl_cycle_repair.py", paths)
        self.assertIn("tools/run_rtl_arbitrary_text_generation.py", paths)
        self.assertIn("tools/ace2_attention_compose_reference.py", paths)
        self.assertIn("rtl/ace2_shell.sv", paths)
        self.assertIn("rtl/generated/ace2_exp_q31_lut.svh", paths)
        self.assertIn("verification/tb/ace2_shell_tb.sv", paths)
        self.assertEqual(aggregate["member_count"], len(paths))

    def test_runtime_source_drift_is_rejected_by_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "runtime.py"
            source.write_text("value = 1\n", encoding="utf-8")
            frozen = records_aggregate([file_record(source)])["sha256"]
            source.write_text("value = 2\n", encoding="utf-8")
            current = records_aggregate([file_record(source)])["sha256"]
            with self.assertRaises(PreflightError):
                require_aggregate_match(frozen, current, "current runtime source")

    def test_file_and_tree_identities_detect_model_and_adapter_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            adapter = root / "adapter"
            model.mkdir()
            adapter.mkdir()
            (model / "model.safetensors").write_bytes(b"model-v1")
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter-v1")
            frozen_model = directory_tree_record(model)["sha256"]
            frozen_adapter = directory_tree_record(adapter)["sha256"]
            (model / "config.json").write_text('{"drift":true}\n', encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter-v2")
            self.assertNotEqual(frozen_model, directory_tree_record(model)["sha256"])
            self.assertNotEqual(frozen_adapter, directory_tree_record(adapter)["sha256"])

    def test_checkpoint_identity_rejects_missing_malformed_and_drifted_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-176"
            checkpoint.mkdir()
            member = checkpoint / "adapter.bin"
            member.write_bytes(b"adapter")
            name = "adapter.bin"
            metadata = {"bytes": member.stat().st_size, "sha256": hashlib.sha256(member.read_bytes()).hexdigest()}
            digest = hashlib.sha256()
            encoded = name.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
            digest.update(bytes.fromhex(metadata["sha256"]))
            digest.update(int(metadata["bytes"]).to_bytes(8, "big"))
            identity = root / "checkpoint_identity.json"
            identity.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "path": str(checkpoint),
                        "files": {name: metadata},
                        "tree_sha256": digest.hexdigest(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(verify_directory_identity(identity)["tree_sha256"], digest.hexdigest())
            member.write_bytes(b"drift")
            with self.assertRaises(PreflightError):
                verify_directory_identity(identity)
            identity.write_text('{"schema_version":1}\n', encoding="utf-8")
            with self.assertRaises(PreflightError):
                verify_directory_identity(identity)

    def test_authorization_contract_hash_binds_every_fresh_l2_authority(self) -> None:
        digest = "a" * 64
        args = Namespace(max_new_tokens=4)
        bindings = {
            "preflight_0004": {
                "preflight": {"sha256": digest},
                "sha256s": {"sha256": "b" * 64},
                "identity_sha256": "c" * 64,
            },
            "checkpoint_176": {
                "model": {"sha256": "d" * 64},
                "parent_model_tree": {"sha256": "e" * 64},
                "adapter": {"sha256": "f" * 64},
                "adapter_checkpoint_tree": {"tree_sha256": "1" * 64},
                "checkpoint_identity": {"sha256": "2" * 64},
                "identity_sha256": "3" * 64,
            },
            "tokenizer": {
                "prompt_sha256": "4" * 64,
                "prompt_token_ids": [1, 2],
                "source_files_sha256": "5" * 64,
                "behavior": {"sha256": "6" * 64},
                "domain": {"sha256": "7" * 64},
                "identity_sha256": "8" * 64,
            },
            "toolchain": {
                "identity_sha256": "9" * 64,
                "python": {"identity_sha256": "a" * 64},
                "packages": {
                    "torch": {"identity_sha256": "b" * 64},
                    "transformers": {"identity_sha256": "c" * 64},
                    "safetensors": {"identity_sha256": "d" * 64},
                },
                "iverilog": {"identity_sha256": "e" * 64},
                "vvp": {"identity_sha256": "f" * 64},
            },
            "runtime_source_aggregate": {"sha256": "1" * 64},
            "protected_files_aggregate": {"sha256": "2" * 64},
            "identity_sha256": "3" * 64,
        }
        contract = fresh_l2_authorization_contract(
            args=args,
            output=Path("preflight-0005"),
            official_output=Path("attempt-0001"),
            bindings=bindings,
        )
        for key in (
            "parent_model_file_sha256",
            "parent_model_tree_sha256",
            "adapter_file_sha256",
            "adapter_tree_sha256",
            "checkpoint_identity_file_sha256",
            "tokenizer_files_sha256",
            "tokenizer_behavior_sha256",
            "tokenizer_domain_sha256",
            "python_identity_sha256",
            "torch_identity_sha256",
            "transformers_identity_sha256",
            "safetensors_identity_sha256",
            "iverilog_identity_sha256",
            "vvp_identity_sha256",
            "runtime_source_aggregate_sha256",
            "protected_files_aggregate_sha256",
        ):
            self.assertIn(key, contract)
        tokenizer_before = identity_sha256(bindings["tokenizer"])
        bindings["tokenizer"]["behavior"]["sha256"] = "0" * 64
        self.assertNotEqual(tokenizer_before, identity_sha256(bindings["tokenizer"]))

    def test_per_step_timing_schema_separates_cpu_and_icarus_wall(self) -> None:
        timing = timing_record(12.5, 7.25, 2.0)
        self.assertEqual(timing["total_wall_seconds"], 12.5)
        self.assertEqual(timing["icarus_wall_seconds"], 7.25)
        self.assertEqual(timing["cpu_wall_seconds"], 5.25)
        self.assertEqual(timing["parent_process_cpu_seconds"], 2.0)
        self.assertIn("iverilog", timing["icarus_wall_definition"])

    def test_recovery_timing_uses_retained_summary_when_wrapper_is_null(self) -> None:
        measured = {
            "compile_wall_seconds": 1.0,
            "model_wall_seconds": 2.0,
            "simulation_wall_seconds": 3.0,
            "total_wall_seconds": 6.0,
        }
        timing, source = full_chain_oracle.retained_backend_timing(
            {"backend_latency": None},
            {"latency": measured},
            recover_failed_attempt=True,
        )
        self.assertEqual(timing, measured)
        self.assertEqual(
            source,
            "sealed_runtime_output_run_summary_latency_due_to_null_wrapper_record",
        )

    def test_oracle_payload_comparison_reads_retained_artifact_bytes(self) -> None:
        payload = b"\x01\x02"
        record = {
            "path": "retained.bin",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            retained = Path(temporary) / "retained.bin"
            retained.write_bytes(payload)
            with mock.patch.object(
                full_chain_oracle.sealed_verifier,
                "verify_file_record",
                return_value=retained,
            ):
                self.assertEqual(
                    full_chain_oracle.require_payload_matches_record(
                        payload,
                        record,
                        "test payload",
                    ),
                    len(payload),
                )

            retained.write_bytes(b"\x01\x03")
            with (
                mock.patch.object(
                    full_chain_oracle.sealed_verifier,
                    "verify_file_record",
                    return_value=retained,
                ),
                self.assertRaisesRegex(
                    full_chain_oracle.OracleError,
                    "test payload retained bytes differ",
                ),
            ):
                full_chain_oracle.require_payload_matches_record(
                    payload,
                    record,
                    "test payload",
                )

    def test_normal_oracle_rejects_failed_attempt_admission(self) -> None:
        with self.assertRaisesRegex(
            full_chain_oracle.OracleError,
            "attempt result is not PASS",
        ):
            full_chain_oracle.verify_attempt_admission(
                Path.cwd(),
                {"status": "FAIL"},
                {},
                recover_failed_attempt=False,
            )

    def test_recovery_admission_requires_exact_rss_failure(self) -> None:
        summary = {
            "status": "PASS_STAGE1_PRODUCT_RTL_GENERATION_UNCHECKED",
            "generated_token_ids": [1, 2, 3, 4],
            "token_executions": [{}, {}],
            "context_contract": {"processed_positions": 2},
            "termination": {
                "reason": "max_new_tokens",
                "generated_token_count": 4,
            },
        }
        attempt_result = {
            "status": "FAIL",
            "exit_code": 3,
            "timed_out": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            (attempt / "attempt-result.json").write_text(
                json.dumps(attempt_result),
                encoding="utf-8",
            )
            (attempt / "stderr.raw.log").write_text(
                full_chain_oracle.RECOVERABLE_RSS_FAILURE + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                full_chain_oracle,
                "file_record",
                side_effect=lambda path: {"path": path.name},
            ):
                admission = full_chain_oracle.verify_attempt_admission(
                    attempt,
                    attempt_result,
                    summary,
                    recover_failed_attempt=True,
                )
            self.assertTrue(admission["numerical_execution_complete"])
            self.assertTrue(admission["original_attempt_remains_failed"])
            (attempt / "stderr.raw.log").write_text(
                "different failure\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                full_chain_oracle.OracleError,
                "not the exact process-tree RSS classification",
            ):
                with mock.patch.object(
                    full_chain_oracle,
                    "file_record",
                    side_effect=lambda path: {"path": path.name},
                ):
                    full_chain_oracle.verify_attempt_admission(
                        attempt,
                        attempt_result,
                        summary,
                        recover_failed_attempt=True,
                    )

    def test_process_tree_rss_observes_iverilog_and_vvp_children(self) -> None:
        with process_tree_rss_tracking(interval_seconds=0.001) as tracker:
            compiled, compile_record = tracked_run(["iverilog", "-V"], cwd=Path.cwd())
            simulated, simulate_record = tracked_run(["vvp", "-V"], cwd=Path.cwd())
        summary = tracker.summary(require_complete=True)
        self.assertEqual(compiled.returncode, 0)
        self.assertEqual(simulated.returncode, 0)
        self.assertIsNotNone(compile_record)
        self.assertIsNotNone(simulate_record)
        self.assertEqual(summary["icarus_child_count"], 2)
        self.assertTrue(summary["all_icarus_children_observed"])
        self.assertTrue(summary["all_icarus_children_terminally_accounted"])
        self.assertEqual(
            summary["sampled_icarus_child_count"]
            + summary["unsampled_icarus_child_count"],
            summary["icarus_child_count"],
        )
        self.assertGreater(summary["peak_rss_kib"], 0)

    def test_process_tree_rss_accepts_terminal_children_missed_by_sampler(self) -> None:
        tracker = ProcessTreeRssTracker(interval_seconds=60.0)
        tracker.start()
        try:
            for pid, program in ((999998, "iverilog"), (999999, "vvp")):
                process = SimpleNamespace(pid=pid, returncode=0)
                tracker.note_spawn(process, [program, "-V"])
                tracker.note_exit(process)
            summary = tracker.summary(require_complete=True)
        finally:
            tracker.stop()

        self.assertTrue(summary["complete"])
        self.assertTrue(summary["all_icarus_children_observed"])
        self.assertTrue(summary["all_icarus_children_terminally_accounted"])
        self.assertFalse(summary["all_icarus_children_sampled"])
        self.assertEqual(summary["sampled_icarus_child_count"], 0)
        self.assertEqual(summary["unsampled_icarus_child_count"], 2)
        for child in summary["children"]:
            self.assertEqual(child["returncode"], 0)
            self.assertEqual(child["observed_samples"], 0)
            self.assertIsNone(child["peak_process_rss_kib"])
            self.assertIsNone(child["peak_subtree_rss_kib"])

    def test_process_tree_rss_rejects_missing_required_program(self) -> None:
        tracker = ProcessTreeRssTracker(interval_seconds=0.001)
        tracker.start()
        try:
            process = SimpleNamespace(pid=999999, returncode=0)
            tracker.note_spawn(process, ["/missing/vvp"])
            tracker.note_exit(process)
            with self.assertRaisesRegex(
                BackendError,
                "missed required Icarus programs: iverilog",
            ):
                tracker.summary(require_complete=True)
        finally:
            tracker.stop()

    def test_process_tree_rss_rejects_sampling_errors(self) -> None:
        tracker = ProcessTreeRssTracker(interval_seconds=60.0)
        tracker.start()
        try:
            for pid, program in ((999998, "iverilog"), (999999, "vvp")):
                process = SimpleNamespace(pid=pid, returncode=0)
                tracker.note_spawn(process, [program, "-V"])
                tracker.note_exit(process)
            with mock.patch.object(tracker, "_snapshot", side_effect=OSError):
                tracker.sample()
            with self.assertRaisesRegex(
                BackendError,
                "sampling reported errors",
            ):
                tracker.summary(require_complete=True)
        finally:
            tracker.stop()

    def test_process_tree_rss_preserves_history_across_pid_reuse(self) -> None:
        tracker = ProcessTreeRssTracker(interval_seconds=0.001)
        first = SimpleNamespace(pid=999998, returncode=0)
        second = SimpleNamespace(pid=999998, returncode=7)

        tracker.note_spawn(first, ["first-child"])
        first_record = tracker.note_exit(first)
        tracker.note_spawn(second, ["second-child"])
        second_record = tracker.note_exit(second)

        summary = tracker.summary(require_complete=False)
        self.assertEqual(first_record["returncode"], 0)
        self.assertEqual(second_record["returncode"], 7)
        self.assertEqual(summary["spawned_process_count"], 2)
        self.assertEqual(
            [record["program"] for record in summary["children"]],
            ["first-child", "second-child"],
        )
        self.assertEqual(
            [record["pid"] for record in summary["children"]],
            [999998, 999998],
        )

    def test_process_tree_rss_rejects_duplicate_active_registration(self) -> None:
        tracker = ProcessTreeRssTracker(interval_seconds=0.001)
        process = SimpleNamespace(pid=999997, returncode=0)
        tracker.note_spawn(process, ["child"])

        with self.assertRaisesRegex(
            BackendError, "duplicate tracked child pid: 999997"
        ):
            tracker.note_spawn(process, ["child"])

        tracker.note_exit(process)

    def test_compile_failure_is_classified_as_simulator_no_execution(self) -> None:
        failed = subprocess.CompletedProcess(["iverilog"], 1, "", "compile failed")
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "tools.rtl_arbitrary_text_generation_backend.run_process",
            return_value=failed,
        ) as run_process_mock:
            with self.assertRaises(SimulatorProcessError) as captured:
                compile_and_run(Path(temporary), "projection", [], "projection_tb")
        self.assertEqual(
            captured.exception.classification,
            "SIMULATOR_COMPILE_FAILURE_NO_EXECUTION",
        )
        self.assertFalse(captured.exception.rtl_executed)
        self.assertEqual(run_process_mock.call_count, 1)

    def test_missing_cycle_marker_is_not_claimed_as_rtl_execution(self) -> None:
        completed = subprocess.CompletedProcess(["tool"], 0, "", "")
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "tools.rtl_arbitrary_text_generation_backend.run_process",
            side_effect=[completed, completed],
        ):
            with self.assertRaises(SimulatorProcessError) as captured:
                compile_and_run(Path(temporary), "projection", [], "projection_tb")
        self.assertEqual(
            captured.exception.classification,
            "SIMULATOR_NO_EXECUTION_EVIDENCE",
        )
        self.assertFalse(captured.exception.rtl_executed)

    def test_context_40_softmax_compile_and_runtime_match_reference(self) -> None:
        scores = [0] * 40
        softmax = rtl_backend.canonical.reference_softmax(
            rtl_backend.canonical.SoftmaxCase("context40", scores),
            context_max=40,
        )
        score = SimpleNamespace(core_scores_q6_9=scores)
        derived = {
            "cached_k": [[] for _ in range(40)],
            "positions": [
                {
                    "scores": [score for _ in range(14)],
                    "probabilities": [softmax for _ in range(14)],
                }
            ],
        }
        with tempfile.TemporaryDirectory(dir="build") as temporary:
            working = Path(temporary)
            vectors = working / "vectors"
            vectors.mkdir(parents=True)
            (vectors / "softmax_vectors.svh").write_text(
                render_softmax_vectors(derived), encoding="utf-8"
            )
            source = included_execution_source(
                working,
                "softmax",
                rtl_backend.layer_runner.SOFTMAX_TB,
                "softmax_vectors.svh",
                "    ace2_softmax_core #(",
            )
            execution = compile_and_run(
                working,
                "softmax",
                [rtl_backend.layer_runner.CORES["softmax"], source],
                "ace2_softmax_tb",
            )
            self.assertIn(
                "ACE2_SOFTMAX_TB_PASS cases=14 context_max=43",
                execution["stdout_text"],
            )
            self.assertNotIn("MISMATCH", execution["stdout_text"])
            self.assertNotIn("FAIL", execution["stdout_text"])

    def test_retained_mlp_residual_case_one_is_warning_and_x_clean(self) -> None:
        zeros = torch.zeros(896, dtype=torch.int8)
        position = {
            "attention_residual": {
                "lhs": zeros,
                "rhs": zeros,
                "output": zeros,
                "saturation": False,
            },
            "final_residual": {
                "down": zeros,
                "stream": zeros,
                "output": zeros,
                "saturation": False,
            },
        }
        final_vectors = render_residual_vectors(position, final=True)
        self.assertIn("localparam integer MLP_RESIDUAL_CASE_COUNT = 2;", final_vectors)
        self.assertIn("mlp_residual_expected_saturation[1] = 1'b0;", final_vectors)
        with tempfile.TemporaryDirectory(dir="build") as temporary:
            working = Path(temporary)
            vectors = working / "vectors"
            vectors.mkdir(parents=True)
            (vectors / "residual_vectors.svh").write_text(
                render_residual_vectors(position, final=False), encoding="utf-8"
            )
            (vectors / "mlp_residual_vectors.svh").write_text(
                final_vectors, encoding="utf-8"
            )
            source = residual_execution_source(working)
            rtl_sources = [Path("rtl/ace2_pkg.sv")] + sorted(
                path for path in Path("rtl").glob("*.sv") if path.name != "ace2_pkg.sv"
            )
            execution = compile_and_run(
                working,
                "residual",
                [*rtl_sources, source],
                "ace2_shell_tb",
                ["+ACE2_GENERATION_SINGLE_TOKEN_RESIDUAL_REPLAY"],
            )
            self.assertIn(
                "ACE2_GENERATION_SINGLE_TOKEN_RESIDUAL_PASS stages=2 beats=112",
                execution["stdout_text"],
            )
            observations = [
                line
                for line in execution["stdout_text"].splitlines()
                if line.startswith("RTL_GENERATION_RESIDUAL_BEAT")
            ]
            self.assertEqual(len(observations), 112)
            self.assertFalse(any(any(char in line for char in "xXzZ") for line in observations))
            compile_stderr = (working / "logs/residual.iverilog.stderr.log").read_text(
                encoding="utf-8"
            )
            self.assertNotRegex(compile_stderr.lower(), r"out[- ]of[- ]bounds?")
            scan = assert_no_machine_private_paths(working)
            self.assertEqual(scan["machine_private_path_matches"], 0)

    def test_failed_child_is_recorded_with_returncode(self) -> None:
        with process_tree_rss_tracking(interval_seconds=0.001):
            completed, record = tracked_run(
                [sys.executable, "-c", "raise SystemExit(7)"],
                cwd=Path.cwd(),
            )
        self.assertEqual(completed.returncode, 7)
        self.assertIsNotNone(record)
        self.assertEqual(record["returncode"], 7)

    def test_official_success_persists_complete_icarus_process_tree(self) -> None:
        sealed_official_identity = official_attempt_identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "official-success-fixture"
            authority_path = root / "authorization-source.json"
            authority_path.write_text('{"authorized":true}\n', encoding="utf-8")
            args = Namespace(
                preflight_output=None,
                output=output,
                authorization=authority_path,
                max_new_tokens=3,
                max_prompt_tokens=8,
                snapshot=None,
            )
            tokenization = {
                "prompt_sha256": "a" * 64,
                "prompt_token_ids": [9707],
            }
            request = {
                "snapshot": root / "snapshot",
                "tokenizer": object(),
                "tokenization": tokenization,
            }
            source_record = {"path": "tools/runtime.py", "bytes": 1, "sha256": "b" * 64}
            protected_record = {"path": "design/SPEC.md", "bytes": 1, "sha256": "c" * 64}
            launch_bindings = {
                "preflight_verification": {"members_verified": 1},
                "runtime_source_aggregate": {
                    "sha256": "d" * 64,
                    "member_count": 1,
                    "total_bytes": 1,
                    "records": [source_record],
                },
                "protected_files_aggregate": {
                    "sha256": "e" * 64,
                    "member_count": 1,
                    "total_bytes": 1,
                    "records": [protected_record],
                },
            }

            def succeed_after_children(target, *_args):
                target.mkdir(parents=True)
                for command in (["iverilog", "-V"], ["vvp", "-V"]):
                    completed, _record = tracked_run(command, cwd=Path.cwd())
                    self.assertEqual(completed.returncode, 0)
                summary = {
                    "schema_version": 1,
                    "status": "PASS_SYNTHETIC_OFFICIAL_BACKEND",
                    "generated_token_ids": [9707],
                    "decoded_text": "synthetic",
                    "peak_rss_kib": 1,
                }
                (target / "run_summary.json").write_text(
                    json.dumps(summary) + "\n",
                    encoding="utf-8",
                )
                return summary

            with mock.patch(
                "tools.run_rtl_arbitrary_text_generation.validate_authorization",
                return_value=({}, authority_path, launch_bindings),
            ), mock.patch(
                "tools.run_rtl_arbitrary_text_generation.backend.run_generation",
                side_effect=succeed_after_children,
            ):
                self.assertEqual(run_official(args, request), 0)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["status"].startswith("PASS_OFFICIAL_"))
            self.assertEqual(manifest["process_start_count"], 1)
            self.assertTrue(manifest["official_attempt_consumed"])
            self.assertEqual(manifest["rss_authority"], "full_official_backend_process_tree")
            self.assertTrue(manifest["process_tree_rss"]["complete"])
            self.assertTrue(manifest["all_icarus_children_observed"])
            self.assertEqual(
                {record["program"] for record in manifest["process_tree_children"]},
                {"iverilog", "vvp"},
            )
            self.assertGreater(manifest["peak_rss_kib"], 1)
            self.assertEqual(summary["peak_rss_kib"], manifest["peak_rss_kib"])
            self.assertTrue(summary["official_result_valid"])
            self.assertTrue(summary["all_icarus_children_observed"])
            for child in manifest["process_tree_children"]:
                self.assertEqual(child["returncode"], 0)
                self.assertGreaterEqual(child["observed_samples"], 1)
        self.assertEqual(official_attempt_identity(), sealed_official_identity)

    def test_consumed_failure_preserves_complete_icarus_process_tree(self) -> None:
        sealed_official_identity = official_attempt_identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "official-failure-fixture"
            authority_path = root / "authorization-source.json"
            authority_path.write_text('{"authorized":true}\n', encoding="utf-8")
            args = Namespace(
                preflight_output=None,
                output=output,
                authorization=authority_path,
                max_new_tokens=3,
                max_prompt_tokens=8,
                snapshot=None,
            )
            request = {
                "snapshot": root / "snapshot",
                "tokenizer": object(),
                "tokenization": {
                    "prompt_sha256": "a" * 64,
                    "prompt_token_ids": [9707],
                },
            }
            source_record = {"path": "tools/runtime.py", "bytes": 1, "sha256": "b" * 64}
            protected_record = {"path": "design/SPEC.md", "bytes": 1, "sha256": "c" * 64}
            launch_bindings = {
                "preflight_verification": {"members_verified": 1},
                "runtime_source_aggregate": {
                    "sha256": "d" * 64,
                    "member_count": 1,
                    "total_bytes": 1,
                    "records": [source_record],
                },
                "protected_files_aggregate": {
                    "sha256": "e" * 64,
                    "member_count": 1,
                    "total_bytes": 1,
                    "records": [protected_record],
                },
            }

            def fail_after_children(target, *_args):
                target.mkdir(parents=True)
                for command in (["iverilog", "-V"], ["vvp", "-V"]):
                    completed, _record = tracked_run(command, cwd=Path.cwd())
                    self.assertEqual(completed.returncode, 0)
                (target / "failure.json").write_text(
                    json.dumps({"status": "SEALED_FAIL_OFFICIAL_RTL_GENERATION", "peak_rss_kib": 1})
                    + "\n",
                    encoding="utf-8",
                )
                raise BackendError("synthetic consumed failure")

            with mock.patch(
                "tools.run_rtl_arbitrary_text_generation.validate_authorization",
                return_value=({}, authority_path, launch_bindings),
            ), mock.patch(
                "tools.run_rtl_arbitrary_text_generation.backend.run_generation",
                side_effect=fail_after_children,
            ), self.assertRaises(BackendError):
                run_official(args, request)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "SEALED_FAIL_OFFICIAL_RTL_GENERATION")
            self.assertEqual(manifest["process_start_count"], 1)
            self.assertTrue(manifest["official_attempt_consumed"])
            self.assertEqual(manifest["source_bindings"], [source_record])
            self.assertEqual(manifest["protected_files"], [protected_record])
            self.assertEqual(manifest["error_type"], "BackendError")
            self.assertTrue(manifest["all_icarus_children_observed"])
            self.assertEqual(
                {record["program"] for record in manifest["process_tree_children"]},
                {"iverilog", "vvp"},
            )
            self.assertEqual(failure["process_tree_children"], manifest["process_tree_children"])
            self.assertEqual(failure["peak_rss_kib"], manifest["peak_rss_kib"])
            self.assertFalse(failure["official_result_valid"])
            self.assertTrue((output / "authorization.json").is_file())
            self.assertTrue((output / "launch_provenance.json").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())
        self.assertEqual(official_attempt_identity(), sealed_official_identity)

    def test_official_process_tree_drift_seals_failure_without_pass_shaped_rss(self) -> None:
        sealed_official_identity = official_attempt_identity()
        drift_summary = {
            "root_pid": os.getpid(),
            "sampling_interval_seconds": 0.01,
            "sample_count": 1,
            "sampling_errors": 0,
            "sampled_process_tree_peak_rss_kib": 4096,
            "sampled_root_peak_rss_kib": 2048,
            "root_ru_maxrss_kib": 2048,
            "peak_rss_kib": 4096,
            "spawned_process_count": 1,
            "icarus_child_count": 1,
            "required_icarus_programs": ["iverilog", "vvp"],
            "observed_icarus_programs": ["iverilog"],
            "missing_icarus_programs": ["vvp"],
            "all_icarus_children_observed": False,
            "complete": False,
            "children": [
                {
                    "pid": 1234,
                    "program": "iverilog",
                    "executable": "/usr/bin/iverilog",
                    "command": ["iverilog"],
                    "started_monotonic": 1.0,
                    "finished_monotonic": 2.0,
                    "returncode": 0,
                    "observed_samples": 1,
                    "peak_process_rss_kib": 1024,
                    "peak_subtree_rss_kib": 1024,
                }
            ],
        }

        class DriftTracker:
            def summary(self, *, require_complete=True):
                if require_complete:
                    raise BackendError("process-tree RSS missed required Icarus programs: vvp")
                return drift_summary

        @contextmanager
        def drift_tracking(*, interval_seconds):
            self.assertEqual(interval_seconds, 0.01)
            yield DriftTracker()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "official-drift-fixture"
            authority_path = root / "authorization-source.json"
            authority_path.write_text('{"authorized":true}\n', encoding="utf-8")
            args = Namespace(
                preflight_output=None,
                output=output,
                authorization=authority_path,
                max_new_tokens=3,
                max_prompt_tokens=8,
                snapshot=None,
            )
            request = {
                "snapshot": root / "snapshot",
                "tokenizer": object(),
                "tokenization": {
                    "prompt_sha256": "a" * 64,
                    "prompt_token_ids": [9707],
                },
            }
            launch_bindings = {
                "preflight_verification": {"members_verified": 1},
                "runtime_source_aggregate": {
                    "sha256": "d" * 64,
                    "member_count": 0,
                    "total_bytes": 0,
                    "records": [],
                },
                "protected_files_aggregate": {
                    "sha256": "e" * 64,
                    "member_count": 0,
                    "total_bytes": 0,
                    "records": [],
                },
            }

            def success_shaped_parent_only_summary(target, *_args):
                target.mkdir(parents=True)
                summary = {
                    "schema_version": 1,
                    "status": "PASS_SYNTHETIC_OFFICIAL_BACKEND",
                    "generated_token_ids": [9707],
                    "decoded_text": "synthetic",
                    "peak_rss_kib": 2048,
                }
                (target / "run_summary.json").write_text(
                    json.dumps(summary) + "\n", encoding="utf-8"
                )
                return summary

            with mock.patch(
                "tools.run_rtl_arbitrary_text_generation.validate_authorization",
                return_value=({}, authority_path, launch_bindings),
            ), mock.patch(
                "tools.run_rtl_arbitrary_text_generation.backend.process_tree_rss_tracking",
                side_effect=drift_tracking,
            ), mock.patch(
                "tools.run_rtl_arbitrary_text_generation.backend.run_generation",
                side_effect=success_shaped_parent_only_summary,
            ), self.assertRaises(BackendError):
                run_official(args, request)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "SEALED_FAIL_OFFICIAL_RTL_GENERATION")
            self.assertFalse(manifest["all_icarus_children_observed"])
            self.assertEqual(manifest["process_tree_rss"]["missing_icarus_programs"], ["vvp"])
            self.assertEqual(
                summary["status"],
                "INVALIDATED_OFFICIAL_RTL_GENERATION_PROCESS_TREE_RSS",
            )
            self.assertFalse(summary["official_result_valid"])
            self.assertEqual(summary["peak_rss_kib"], 4096)
            self.assertFalse(failure["official_result_valid"])
            self.assertFalse(failure["all_icarus_children_observed"])
        self.assertEqual(official_attempt_identity(), sealed_official_identity)

    def test_official_mode_cannot_create_output_without_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "unauthorized-official-fixture"
            args = Namespace(
                preflight_output=None,
                output=output,
                authorization=None,
            )
            with self.assertRaises(OfficialAuthorizationError):
                run_official(args, {})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
