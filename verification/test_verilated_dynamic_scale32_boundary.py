from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import run_full_qwen_command_schedule_runtime as accepted_runtime
from tools.ace2_chat_demo import (
    DYNAMIC_SCALE32_FLAG,
    DYNAMIC_SCALE32_HOST_PLAN_LAYER,
    DYNAMIC_SCALE32_HOST_PLAN_OPCODE,
    PACKAGE_V2_DYNAMIC_RECORD,
    PACKAGE_V2_HEADER,
    PACKAGE_V2_HEADER_BYTES,
    PINNED_EMBEDDING_OFFSET,
    audit_prompt_dynamic_scale32_applicability,
    build_ace2rt2_package,
    build_full_prompt_commands,
    build_prompt_package,
    dynamic_scale32_model_identity,
    projection_reference,
)
from tools.ace2_rmsnorm_reference import reference_rmsnorm
from verification.test_verilated_rmsnorm_runtime import (
    HIDDEN,
    RMSNORM_GAIN_BYTES,
    RMSNORM_SCALE_OFFSET,
    image_file_offset,
    parse_first_journal_record,
    parse_journal_records,
    quantized_embedding,
    read_at,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "build/verilator_full_qwen_runtime/Vace2_shell_runtime_harness"
SEED_TOKEN = 89779
sys.path.insert(0, str(ROOT / "tools"))

from ace2_dynamic_scale32_reference import build_sidecar, quantize_for_delta  # noqa: E402
from ace2_quality_contracts import pack_scale32  # noqa: E402


class VerilatedDynamicScale32BoundaryTest(unittest.TestCase):
    def build_package(self, path: Path) -> tuple[bytearray, list[dict[str, object]]]:
        schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
        commands = build_full_prompt_commands(
            schedule,
            prompt_token_count=1,
            max_new_tokens=1,
        )
        command = commands[0]
        for dynamic_command in commands[:4]:
            dynamic_command["flags"] = int(dynamic_command["flags"]) | DYNAMIC_SCALE32_FLAG
        plan = audit_prompt_dynamic_scale32_applicability(
            [SEED_TOKEN],
            accepted_runtime.embedding_tensor_offset(accepted_runtime.MODEL)[0],
        )["cases"][0]
        base_scale = pack_scale32(0x8000, 0)
        sidecar = build_sidecar(
            payload_addr=int(command["src0_addr"]),
            group_lanes=128,
            producer_tag=0,
            layer_id=DYNAMIC_SCALE32_HOST_PLAN_LAYER,
            producer_opcode=DYNAMIC_SCALE32_HOST_PLAN_OPCODE,
            tensor_elements=HIDDEN,
            model_identity=dynamic_scale32_model_identity(
                accepted_runtime.EXPECTED_MODEL_SHA256
            ),
            deltas=plan["selected_deltas"],
            base_scales=(base_scale,) * 7,
        )
        embedding_offset, embedding_shape = accepted_runtime.embedding_tensor_offset(
            accepted_runtime.MODEL
        )
        build_ace2rt2_package(
            path,
            prompt_token_ids=[SEED_TOKEN],
            max_new_tokens=1,
            commands=commands,
            embedding_offset=embedding_offset,
            embedding_shape=embedding_shape,
            dynamic_scale32_sidecars=[
                {"payload_addr": int(command["src0_addr"]), "sidecar": sidecar}
            ],
        )
        return bytearray(path.read_bytes()), commands

    @staticmethod
    def run_runtime(
        package: Path,
        output: Path,
        *,
        stop_after: int = 1,
        read_latency: int = 1,
        timeout_cycles: int = 1_000_000,
        extra_args: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        command = [
            str(RUNTIME),
            "--package",
            str(package),
            "--image",
            str(accepted_runtime.IMAGE),
            "--model",
            str(accepted_runtime.MODEL),
            "--output",
            str(output),
            "--stop-after",
            str(stop_after),
            "--timeout-cycles",
            str(timeout_cycles),
            "--read-response-latency-cycles",
            str(read_latency),
        ]
        command.extend(extra_args)
        return subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )

    def test_staged_initial_sidecar_gates_bit_exact_rmsnorm(self) -> None:
        self.assertTrue(RUNTIME.is_file(), "run `make full-qwen-runtime-build` first")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_path = root / "dynamic.bin"
            raw, commands = self.build_package(package_path)
            command = commands[0]
            completed = self.run_runtime(package_path, root / "positive")
            self.assertEqual(completed.returncode, 0, completed.stderr)

            scale_raw = read_at(
                accepted_runtime.IMAGE,
                image_file_offset(
                    int(command["scale_addr"]) + RMSNORM_SCALE_OFFSET,
                    8,
                ),
                8,
            )
            embedding_scale = np.float32(struct.unpack("<d", scale_raw)[0])
            gains_raw = read_at(
                accepted_runtime.IMAGE,
                image_file_offset(int(command["scale_addr"]), RMSNORM_GAIN_BYTES),
                RMSNORM_GAIN_BYTES,
            )
            gains = np.frombuffer(gains_raw, dtype="<i2").astype(int).tolist()
            embedding_offset = int(PACKAGE_V2_HEADER.unpack_from(raw)[11])
            activations = quantized_embedding(SEED_TOKEN, embedding_offset, embedding_scale)
            rms_outputs = reference_rmsnorm(activations, gains).outputs
            dynamic_outputs = []
            sidecar_start = (
                int(PACKAGE_V2_HEADER.unpack_from(raw)[2])
                + 4
                + 8
                + int(PACKAGE_V2_HEADER.unpack_from(raw)[10]) * 128
            )
            host_plan = PACKAGE_V2_DYNAMIC_RECORD.unpack_from(raw, sidecar_start)[1]
            selected_deltas = [
                value - 256 if value & 0x80 else value for value in host_plan[24:31]
            ]
            for group, delta in enumerate(selected_deltas):
                values = rms_outputs[group * 128 : (group + 1) * 128]
                dynamic_outputs.extend(quantize_for_delta(value, delta) for value in values)
            expected = bytes(value & 0xFF for value in dynamic_outputs)
            expected_sidecar = build_sidecar(
                payload_addr=int(command["dst_addr"]),
                group_lanes=128,
                producer_tag=int(command["completion_tag"]),
                layer_id=int(command["layer_id"]),
                producer_opcode=int(command["opcode"]),
                tensor_elements=HIDDEN,
                model_identity=dynamic_scale32_model_identity(
                    accepted_runtime.EXPECTED_MODEL_SHA256
                ),
                deltas=selected_deltas,
                base_scales=(pack_scale32(0x8000, 0),) * 7,
            )

            record = parse_first_journal_record(root / "positive" / "progress.journal")
            self.assertEqual(record["read_beats"], 228)
            self.assertEqual(record["write_beats"], HIDDEN // 16 + 4)
            self.assertFalse(record["done_error"])
            payload_writes = record["writes"][: HIDDEN // 16]
            sidecar_writes = record["writes"][HIDDEN // 16 :]
            self.assertEqual(b"".join(write[2] for write in payload_writes), expected)
            self.assertEqual(
                [write[0] for write in sidecar_writes],
                [int(command["dst_addr"]) - 64 + beat * 16 for beat in range(4)],
            )
            self.assertEqual(b"".join(write[2] for write in sidecar_writes), expected_sidecar)

    def test_qkv_consumers_apply_position_plan_under_backpressure(self) -> None:
        self.assertTrue(RUNTIME.is_file(), "run `make full-qwen-runtime-build` first")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_path = root / "dynamic-q.bin"
            raw, commands = self.build_package(package_path)
            completed = self.run_runtime(
                package_path,
                root / "q",
                stop_after=4,
                read_latency=7,
                timeout_cycles=10_000_000,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            rms = commands[0]
            scale_raw = read_at(
                accepted_runtime.IMAGE,
                image_file_offset(int(rms["scale_addr"]) + RMSNORM_SCALE_OFFSET, 8),
                8,
            )
            embedding_scale = np.float32(struct.unpack("<d", scale_raw)[0])
            gains_raw = read_at(
                accepted_runtime.IMAGE,
                image_file_offset(int(rms["scale_addr"]), RMSNORM_GAIN_BYTES),
                RMSNORM_GAIN_BYTES,
            )
            gains = np.frombuffer(gains_raw, dtype="<i2").astype(int).tolist()
            embedding_offset = int(PACKAGE_V2_HEADER.unpack_from(raw)[11])
            activations = quantized_embedding(
                SEED_TOKEN, embedding_offset, embedding_scale
            )
            rms_outputs = reference_rmsnorm(activations, gains).outputs
            records = parse_journal_records(root / "q" / "progress.journal")
            self.assertEqual(len(records), 4)
            for ordinal, projection_command in enumerate(commands[1:4], start=1):
                with self.subTest(operator=projection_command["operator"]):
                    expected = bytes(
                        value & 0xFF
                        for value in projection_reference(projection_command, rms_outputs)
                    )
                    record = records[ordinal]
                    self.assertEqual(record["ordinal"], ordinal)
                    self.assertGreater(record["read_beats"], 4)
                    self.assertEqual(record["write_beats"], int(projection_command["n"]) // 16)
                    self.assertFalse(record["done_error"])
                    self.assertEqual(
                        b"".join(write[2] for write in record["writes"]), expected
                    )

    def test_two_nonzero_user_content_positions_match_reference(self) -> None:
        self.assertTrue(RUNTIME.is_file(), "run `make full-qwen-runtime-build` first")
        prompt_tokens = [151644, 872, 198, 840, 3170]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_path = root / "prompt-dynamic.bin"
            package = build_prompt_package(package_path, prompt_tokens, 1)
            commands = package.pop("_position3_reference_commands")
            position_four_commands = [
                command
                for command in build_full_prompt_commands(
                    json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8")),
                    prompt_token_count=len(prompt_tokens),
                    max_new_tokens=1,
                )
                if int(command["token_step"]) == 4 and int(command["layer_id"]) == 0
            ][:4]
            packaged_commands = package.pop("_position0_reference_commands")
            del packaged_commands
            all_commands = package.pop("_position1_reference_commands")
            all_commands = package.pop("_position2_reference_commands")
            del all_commands
            first_commands = package.pop("_first_reference_commands")
            del first_commands
            package.pop("_all_reference_commands")
            package.pop("_ds32_applicability")

            scale_raw = read_at(
                accepted_runtime.IMAGE,
                image_file_offset(0x0000000300000000 + RMSNORM_SCALE_OFFSET, 8),
                8,
            )
            embedding_scale = np.float32(struct.unpack("<d", scale_raw)[0])
            gains_raw = read_at(
                accepted_runtime.IMAGE,
                image_file_offset(0x0000000300000000, RMSNORM_GAIN_BYTES),
                RMSNORM_GAIN_BYTES,
            )
            gains = np.frombuffer(gains_raw, dtype="<i2").astype(int).tolist()

            for position, token_id in ((3, 840), (4, 3170)):
                with self.subTest(position=position, token_id=token_id):
                    output = root / f"position-{position}"
                    completed = self.run_runtime(
                        package_path,
                        output,
                        stop_after=4,
                        timeout_cycles=10_000_000,
                        extra_args=("--focus-ds32-token-step", str(position)),
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    target = (
                        [
                            command
                            for command in commands
                            if int(command["token_step"]) == position
                            and int(command["layer_id"]) == 0
                        ][:4]
                        if position == 3
                        else position_four_commands
                    )
                    embedding = quantized_embedding(
                        token_id, PINNED_EMBEDDING_OFFSET, embedding_scale
                    )
                    rms_outputs = reference_rmsnorm(embedding, gains).outputs
                    records = parse_journal_records(output / "progress.journal")
                    self.assertEqual(len(records), 4)
                    self.assertEqual(records[0]["done_tag"], int(target[0]["completion_tag"]))
                    for record, command in zip(records[1:], target[1:], strict=True):
                        expected = bytes(
                            value & 0xFF
                            for value in projection_reference(command, rms_outputs)
                        )
                        self.assertEqual(record["done_tag"], int(command["completion_tag"]))
                        self.assertFalse(record["done_error"])
                        self.assertEqual(
                            b"".join(write[2] for write in record["writes"]), expected
                        )

    def test_structural_corruption_reaches_rtl_validator(self) -> None:
        self.assertTrue(RUNTIME.is_file(), "run `make full-qwen-runtime-build` first")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_path = root / "dynamic-corrupt.bin"
            self.build_package(package_path)
            completed = self.run_runtime(
                package_path,
                root / "corrupt",
                extra_args=("--corrupt-staged-sidecar-byte", "0"),
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            record = parse_first_journal_record(
                root / "corrupt" / "progress.journal"
            )
            self.assertEqual(record["read_beats"], 4)
            self.assertEqual(record["write_beats"], 0)
            self.assertTrue(record["done_error"])

    def test_sidecar_response_error_and_tag_mismatch_fail_closed(self) -> None:
        self.assertTrue(RUNTIME.is_file(), "run `make full-qwen-runtime-build` first")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_path = root / "dynamic-response.bin"
            _, commands = self.build_package(package_path)
            sidecar_address = int(commands[0]["src0_addr"]) - 64
            for name, option in (
                ("error", "--read-error-address"),
                ("tag", "--read-tag-mismatch-address"),
            ):
                with self.subTest(name=name):
                    completed = self.run_runtime(
                        package_path,
                        root / name,
                        extra_args=(option, hex(sidecar_address)),
                    )
                    self.assertEqual(completed.returncode, 2, completed.stderr)
                    record = parse_first_journal_record(
                        root / name / "progress.journal"
                    )
                    self.assertEqual(record["write_beats"], 0)
                    self.assertTrue(record["done_error"])

    def test_reset_before_publication_accepts_no_sidecar_write(self) -> None:
        self.assertTrue(RUNTIME.is_file(), "run `make full-qwen-runtime-build` first")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_path = root / "dynamic-reset.bin"
            _, commands = self.build_package(package_path)
            sidecar_address = int(commands[0]["dst_addr"]) - 64
            output = root / "reset"
            completed = self.run_runtime(
                package_path,
                output,
                extra_args=("--reset-before-write-address", hex(sidecar_address)),
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn("category=reset_during_busy", completed.stderr)
            failure = json.loads((output / "first_failure.json").read_text())
            self.assertEqual(failure["category"], "reset_during_busy")
            self.assertEqual(parse_journal_records(output / "progress.journal"), [])

    def test_unrepresentable_host_plan_fails_without_sidecar_publication(self) -> None:
        self.assertTrue(RUNTIME.is_file(), "run `make full-qwen-runtime-build` first")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_path = root / "dynamic-nonzero.bin"
            raw, commands = self.build_package(package_path)
            header = PACKAGE_V2_HEADER.unpack_from(raw)
            header_bytes = int(header[2])
            prompt_count = int(header[5])
            termination_count = int(header[7])
            rope_count = int(header[10])
            manifest_start = header_bytes + prompt_count * 4 + termination_count * 4
            sidecar_start = manifest_start + rope_count * 128
            raw[sidecar_start + 8 + 24] = (-24) & 0xFF
            sidecar_records = bytes(
                raw[sidecar_start : sidecar_start + PACKAGE_V2_DYNAMIC_RECORD.size]
            )
            raw[
                PACKAGE_V2_HEADER_BYTES + 24 : PACKAGE_V2_HEADER_BYTES + 56
            ] = hashlib.sha256(sidecar_records).digest()
            raw[56:88] = hashlib.sha256(bytes(raw[manifest_start:])).digest()
            package_path.write_bytes(raw)

            completed = self.run_runtime(package_path, root / "negative")
            self.assertEqual(completed.returncode, 2, completed.stderr)
            record = parse_first_journal_record(root / "negative" / "progress.journal")
            self.assertGreater(record["read_beats"], 4)
            sidecar_base = int(commands[0]["dst_addr"]) - 64
            self.assertFalse(
                any(sidecar_base <= write[0] < sidecar_base + 64 for write in record["writes"])
            )
            self.assertTrue(record["done_error"])
            self.assertFalse(record["saturation"])

    def test_rtl_rejects_frozen_address_and_low_flag_mutations(self) -> None:
        self.assertTrue(RUNTIME.is_file(), "run `make full-qwen-runtime-build` first")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_path = root / "dynamic-frozen-tuples.bin"
            self.build_package(package_path)
            cases = (
                ("rms_address", 0, "--rtl-mutate-frozen-address"),
                ("q_address", 1, "--rtl-mutate-frozen-address"),
                ("k_address", 2, "--rtl-mutate-frozen-address"),
                ("v_address", 3, "--rtl-mutate-frozen-address"),
                ("rms_low_flags", 0, "--rtl-mutate-frozen-low-flags"),
                ("q_low_flags", 1, "--rtl-mutate-frozen-low-flags"),
                ("k_low_flags", 2, "--rtl-mutate-frozen-low-flags"),
                ("v_low_flags", 3, "--rtl-mutate-frozen-low-flags"),
            )
            for name, ordinal, option in cases:
                with self.subTest(name=name):
                    output = root / name
                    completed = self.run_runtime(
                        package_path,
                        output,
                        stop_after=ordinal + 1,
                        timeout_cycles=10_000_000,
                        extra_args=(option, str(ordinal)),
                    )
                    self.assertEqual(completed.returncode, 2, completed.stderr)
                    records = parse_journal_records(output / "progress.journal")
                    self.assertEqual(len(records), ordinal + 1)
                    record = records[ordinal]
                    self.assertEqual(record["read_beats"], 0)
                    self.assertEqual(record["write_beats"], 0)
                    self.assertTrue(record["done_error"])
                    self.assertFalse(record["saturation"])


if __name__ == "__main__":
    unittest.main()
