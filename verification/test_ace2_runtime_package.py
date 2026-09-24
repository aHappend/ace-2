from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from tools import run_full_qwen_command_schedule_runtime as accepted_runtime
from tools.ace2_chat_demo import (
    DYNAMIC_SCALE32_FLAG,
    DYNAMIC_SCALE32_HOST_PLAN_LAYER,
    DYNAMIC_SCALE32_HOST_PLAN_OPCODE,
    PACKAGE_V2_HEADER,
    PACKAGE_V2_HEADER_BYTES,
    PACKAGE_V2_DYNAMIC_EXTENSION_BYTES,
    PACKAGE_V2_DYNAMIC_RECORD,
    PINNED_EMBEDDING_OFFSET,
    QKV_SCHEDULE_FUSED,
    QKV_SCHEDULE_LEGACY,
    TERMINATION_TOKEN_IDS,
    build_ace2rt2_package,
    build_full_prompt_commands,
    build_rope_records,
    dynamic_scale32_model_identity,
    read_ace2rt2_package_metadata,
    reconstruct_ace2rt2_schedule,
    sha256_bytes,
    tokenizer_identity_record,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "build/verilator_full_qwen_runtime/Vace2_shell_runtime_harness"
sys.path.insert(0, str(ROOT / "tools"))

from ace2_dynamic_scale32_reference import build_sidecar  # noqa: E402
from ace2_quality_contracts import pack_scale32  # noqa: E402


def synthetic_command(ordinal: int, token_step: int) -> dict[str, int | str | None]:
    return {
        "ordinal": ordinal,
        "token_step": token_step,
        "layer_id": 0,
        "operator": "input_rmsnorm",
        "opcode": 2,
        "flags": 0,
        "m": 1,
        "n": 896,
        "k": 896,
        "sequence_position": token_step,
        "completion_tag": ordinal,
        "query_head": None,
        "context_token": None,
        "vocab_tile": None,
        "src0_addr": 0x1000 + token_step * 0x1000,
        "src1_addr": 0,
        "dst_addr": 0x8000 + token_step * 0x1000,
        "scale_addr": 0x0000000300000000,
        "scratch_addr": 0,
    }


class Ace2RuntimePackageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(RUNTIME.is_file(), "run `make full-qwen-runtime-build` first")

    def build_package(
        self,
        path: Path,
        *,
        dynamic_scale32: bool = False,
        qkv_schedule_mode: str = QKV_SCHEDULE_LEGACY,
        command_mutator: Callable[[list[dict[str, object]]], None] | None = None,
    ) -> dict[str, object]:
        schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
        commands = build_full_prompt_commands(
            schedule,
            prompt_token_count=2,
            max_new_tokens=2,
            qkv_schedule_mode=qkv_schedule_mode,
            preserve_prompt_layer0_qkv=dynamic_scale32,
        )
        sidecars = None
        if dynamic_scale32:
            for command in commands:
                if (
                    int(command["token_step"]) < 2
                    and int(command["layer_id"]) == 0
                    and command["operator"] in {
                        "input_rmsnorm",
                        "q_proj",
                        "k_proj",
                        "v_proj",
                    }
                ):
                    command["flags"] = int(command["flags"]) | DYNAMIC_SCALE32_FLAG
            producer = commands[0]
            group_count = (int(producer["n"]) + 127) // 128
            base_scale = pack_scale32(0x8000, 0)
            model_identity = dynamic_scale32_model_identity(
                accepted_runtime.EXPECTED_MODEL_SHA256
            )
            sidecars = [
                {
                    "payload_addr": int(producer["src0_addr"]),
                    "sidecar": build_sidecar(
                        payload_addr=int(producer["src0_addr"]),
                        group_lanes=128,
                        producer_tag=position,
                        layer_id=DYNAMIC_SCALE32_HOST_PLAN_LAYER,
                        producer_opcode=DYNAMIC_SCALE32_HOST_PLAN_OPCODE,
                        tensor_elements=int(producer["n"]),
                        model_identity=model_identity,
                        deltas=(0,) * group_count,
                        base_scales=(base_scale,) * group_count,
                    ),
                }
                for position in range(2)
            ]
        if command_mutator is not None:
            command_mutator(commands)
        return build_ace2rt2_package(
            path,
            prompt_token_ids=[17, 23],
            max_new_tokens=2,
            commands=commands,
            embedding_offset=PINNED_EMBEDDING_OFFSET,
            embedding_shape=[151936, 896],
            dynamic_scale32_sidecars=sidecars,
        )

    def inspect(self, path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(RUNTIME), "--package", str(path), "--inspect-package"],
            check=False,
            text=True,
            capture_output=True,
        )

    def inspect_journal(self, package: Path, journal: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(RUNTIME),
                "--package",
                str(package),
                "--inspect-journal",
                str(journal),
            ],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_tokenizer_identity_serialization_is_exact(self) -> None:
        record = tokenizer_identity_record()
        self.assertTrue(record.endswith(b"\n"))
        self.assertNotIn(b"\r", record)
        self.assertEqual(
            sha256_bytes(record),
            "f2e682984c6fbad1c922bdc0810c8c38b37f2f2be01aa588f3b4ed18e6e20bdb",
        )

    def test_python_builder_cpp_loader_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package_path = Path(temporary) / "runtime.bin"
            built = self.build_package(package_path)
            completed = self.inspect(package_path)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            inspected = json.loads(completed.stdout)
            self.assertEqual(inspected["format"], "ACE2RT2")
            self.assertEqual(inspected["commands"], built["commands"])
            self.assertEqual(inspected["prompt_token_ids"], [17, 23])
            self.assertEqual(inspected["max_new_tokens"], 2)
            self.assertEqual(inspected["termination_token_ids"], list(TERMINATION_TOKEN_IDS))
            self.assertEqual(inspected["rope_position_count"], 3)
            self.assertEqual(inspected["rope_records_sha256"], built["rope_records_sha256"])
            self.assertEqual(inspected["schedule_sha256"], built["schedule_sha256"])
            self.assertEqual(inspected["tokenizer_sha256"], built["tokenizer_sha256"])
            self.assertEqual(inspected["prompt_tokens_sha256"], built["prompt_tokens_sha256"])
            self.assertEqual(inspected["package_sha256"], built["sha256"])
            raw = package_path.read_bytes()
            rope_start = PACKAGE_V2_HEADER_BYTES + 2 * 4 + 2 * 4
            self.assertEqual(
                raw[rope_start + 2 * 128 : rope_start + 3 * 128],
                build_rope_records(3)[2 * 128 : 3 * 128],
            )

    def test_loaders_accept_fused_qkv_and_reject_noncanonical_bases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_path = root / "fused.bin"
            built = self.build_package(
                package_path,
                qkv_schedule_mode=QKV_SCHEDULE_FUSED,
            )
            completed = self.inspect(package_path)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                built["qkv_schedule_mode"],
                QKV_SCHEDULE_FUSED,
            )

            valid = package_path.read_bytes()
            rope_start = PACKAGE_V2_HEADER_BYTES + 2 * 4 + 2 * 4
            command_start = rope_start + 3 * 128
            fused_record = command_start + accepted_runtime.COMMAND.size
            cases = {
                "weight": (36, 0x0000000100000010),
                "output": (44, 0x0000001000000A90),
                "metadata": (52, 0x0000000200000010),
            }
            for name, (field_offset, wrong_base) in cases.items():
                with self.subTest(name=name):
                    raw = bytearray(valid)
                    struct.pack_into(
                        "<Q",
                        raw,
                        fused_record + field_offset,
                        wrong_base,
                    )
                    raw[56:88] = hashlib.sha256(bytes(raw[rope_start:])).digest()
                    corrupted = root / f"fused-bad-{name}.bin"
                    corrupted.write_bytes(raw)
                    with self.assertRaisesRegex(
                        RuntimeError, "fused-QKV descriptor contract differs"
                    ):
                        read_ace2rt2_package_metadata(corrupted)
                    rejected = self.inspect(corrupted)
                    self.assertEqual(rejected.returncode, 3, rejected.stdout)
                    self.assertIn(
                        "fused-QKV descriptor geometry or address differs",
                        rejected.stderr,
                    )

    def test_cpp_loader_rejects_malformed_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid_path = root / "valid.bin"
            self.build_package(valid_path)
            valid = valid_path.read_bytes()
            rope_start = PACKAGE_V2_HEADER_BYTES + 2 * 4 + 2 * 4
            command_start = rope_start + 3 * 128

            cases: dict[str, tuple[bytes, str]] = {}

            rope_count = bytearray(valid)
            struct.pack_into("<I", rope_count, 44, 0)
            cases["rope_count"] = (bytes(rope_count), "ACE2RT2 header contract differs")

            embedding_offset = bytearray(valid)
            struct.pack_into("<Q", embedding_offset, 48, PINNED_EMBEDDING_OFFSET + 1)
            cases["embedding_offset"] = (
                bytes(embedding_offset),
                "ACE2RT2 pinned model or tokenizer identity differs",
            )

            tokenizer_identity = bytearray(valid)
            tokenizer_identity[152] ^= 1
            cases["tokenizer_identity"] = (
                bytes(tokenizer_identity),
                "ACE2RT2 pinned model or tokenizer identity differs",
            )

            prompt_digest = bytearray(valid)
            prompt_digest[PACKAGE_V2_HEADER_BYTES] ^= 1
            cases["prompt_digest"] = (
                bytes(prompt_digest),
                "ACE2RT2 prompt-token digest differs",
            )

            termination = bytearray(valid)
            struct.pack_into("<I", termination, PACKAGE_V2_HEADER_BYTES + 8, 1)
            cases["termination"] = (
                bytes(termination),
                "ACE2RT2 termination token list differs",
            )

            schedule_digest = bytearray(valid)
            schedule_digest[command_start + 20] ^= 1
            cases["schedule_digest"] = (
                bytes(schedule_digest),
                "ACE2RT2 schedule digest differs",
            )

            rope_record = bytearray(valid)
            rope_record[rope_start + 2 * 128] ^= 1
            cases["rope_record"] = (
                bytes(rope_record),
                "ACE2RT2 schedule digest differs",
            )

            ordinal = bytearray(valid)
            struct.pack_into("<I", ordinal, command_start, 9)
            cases["ordinal"] = (
                bytes(ordinal),
                "runtime command package is not ordinally canonical",
            )

            step_gap = bytearray(valid)
            second_record = command_start + accepted_runtime.COMMAND.size
            struct.pack_into("<H", step_gap, second_record + 4, 2)
            struct.pack_into("<H", step_gap, second_record + 16, 2)
            manifest_bytes = bytes(step_gap[rope_start:])
            step_gap[56:88] = hashlib.sha256(manifest_bytes).digest()
            cases["step_gap"] = (
                bytes(step_gap),
                "ACE2RT2 token steps are not contiguous and nondecreasing",
            )

            cases["trailing"] = (valid + b"x", "runtime package has trailing bytes")
            cases["truncated"] = (valid[:-1], "runtime package command array is truncated")

            for name, (raw, expected_error) in cases.items():
                with self.subTest(name=name):
                    path = root / f"{name}.bin"
                    path.write_bytes(raw)
                    completed = self.inspect(path)
                    self.assertEqual(completed.returncode, 3, completed.stdout)
                    self.assertIn(expected_error, completed.stderr)

    def test_dynamic_scale32_extension_schedule_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package_path = Path(temporary) / "dynamic.bin"
            built = self.build_package(
                package_path,
                dynamic_scale32=True,
                qkv_schedule_mode=QKV_SCHEDULE_FUSED,
            )
            completed = self.inspect(package_path)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            inspected = read_ace2rt2_package_metadata(package_path)
            self.assertEqual(inspected["qkv_schedule_mode"], QKV_SCHEDULE_FUSED)
            reconstructed = reconstruct_ace2rt2_schedule(package_path, inspected)
            self.assertEqual(
                reconstructed["schedule_sha256"],
                built["schedule_sha256"],
            )
            self.assertEqual(
                reconstructed["dynamic_scale32_sidecar_records_sha256"],
                built["dynamic_scale32_sidecar_records_sha256"],
            )
            dynamic_commands = [
                command
                for command in reconstructed["commands"]
                if int(command["flags"]) & DYNAMIC_SCALE32_FLAG
            ]
            self.assertEqual(len(dynamic_commands), 8)
            self.assertEqual(
                [command["operator"] for command in dynamic_commands],
                ["input_rmsnorm", "q_proj", "k_proj", "v_proj"] * 2,
            )
            self.assertEqual(
                sum(
                    command["operator"] == "fused_qkv"
                    for command in reconstructed["commands"]
                ),
                24 * 3 - 2,
            )

    def test_dynamic_scale32_extension_round_trip_and_malformed_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid_path = root / "dynamic.bin"
            built = self.build_package(valid_path, dynamic_scale32=True)
            python_inspected = read_ace2rt2_package_metadata(valid_path)
            completed = self.inspect(valid_path)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            inspected = json.loads(completed.stdout)
            self.assertTrue(inspected["dynamic_scale32_enabled"])
            self.assertEqual(inspected["dynamic_scale32_sidecar_count"], 2)
            self.assertEqual(
                inspected["dynamic_scale32_sidecar_records_sha256"],
                built["dynamic_scale32_sidecar_records_sha256"],
            )
            self.assertEqual(
                inspected["dynamic_scale32_model_identity64"],
                built["dynamic_scale32_model_identity64"],
            )
            self.assertEqual(
                inspected["dynamic_scale32_payload_addresses"],
                built["dynamic_scale32_payload_addresses"],
            )
            self.assertEqual(
                python_inspected["dynamic_scale32_sidecar_records_sha256"],
                built["dynamic_scale32_sidecar_records_sha256"],
            )
            self.assertEqual(
                python_inspected["command_records_offset"],
                PACKAGE_V2_HEADER_BYTES
                + PACKAGE_V2_DYNAMIC_EXTENSION_BYTES
                + 2 * 4
                + 2 * 4
                + 3 * 128
                + 2 * PACKAGE_V2_DYNAMIC_RECORD.size,
            )

            valid = valid_path.read_bytes()
            header_bytes = PACKAGE_V2_HEADER_BYTES + PACKAGE_V2_DYNAMIC_EXTENSION_BYTES
            rope_start = header_bytes + 2 * 4 + 2 * 4
            sidecar_start = rope_start + 3 * 128
            command_start = sidecar_start + 2 * PACKAGE_V2_DYNAMIC_RECORD.size

            cases: dict[str, tuple[bytes, str]] = {}

            extension_magic = bytearray(valid)
            extension_magic[PACKAGE_V2_HEADER_BYTES] ^= 1
            cases["extension_magic"] = (
                bytes(extension_magic),
                "ACE2RT2 dynamic Scale32 extension contract differs",
            )

            sidecar_digest = bytearray(valid)
            sidecar_digest[sidecar_start + 8 + 24] ^= 1
            cases["sidecar_digest"] = (
                bytes(sidecar_digest),
                "ACE2RT2 dynamic Scale32 sidecar digest differs",
            )

            sidecar_magic = bytearray(valid)
            sidecar_magic[sidecar_start + 8] ^= 1
            sidecar_records = bytes(sidecar_magic[sidecar_start:command_start])
            sidecar_magic[PACKAGE_V2_HEADER_BYTES + 24 : PACKAGE_V2_HEADER_BYTES + 56] = (
                hashlib.sha256(sidecar_records).digest()
            )
            sidecar_magic[56:88] = hashlib.sha256(
                bytes(sidecar_magic[rope_start:])
            ).digest()
            cases["sidecar_magic"] = (
                bytes(sidecar_magic),
                "ACE2RT2 dynamic Scale32 sidecar magic or schema differs",
            )

            producer_tag = bytearray(valid)
            producer_tag[sidecar_start + 8 + 8] ^= 1
            sidecar_records = bytes(producer_tag[sidecar_start:command_start])
            producer_tag[
                PACKAGE_V2_HEADER_BYTES + 24 : PACKAGE_V2_HEADER_BYTES + 56
            ] = hashlib.sha256(sidecar_records).digest()
            producer_tag[56:88] = hashlib.sha256(
                bytes(producer_tag[rope_start:])
            ).digest()
            cases["producer_tag"] = (
                bytes(producer_tag),
                "ACE2RT2 dynamic Scale32 producer binding is duplicated",
            )

            missing_flag = bytearray(valid)
            missing_flag[command_start + 9] &= ~DYNAMIC_SCALE32_FLAG
            missing_flag[56:88] = hashlib.sha256(
                bytes(missing_flag[rope_start:])
            ).digest()
            cases["missing_flag"] = (
                bytes(missing_flag),
                "ACE2RT2 prompt DS32 tranche requires one host plan and RMS/Q/K/V per prompt position",
            )

            missing_q_flag = bytearray(valid)
            missing_q_flag[command_start + accepted_runtime.COMMAND.size + 9] &= (
                ~DYNAMIC_SCALE32_FLAG
            )
            missing_q_flag[56:88] = hashlib.sha256(
                bytes(missing_q_flag[rope_start:])
            ).digest()
            cases["missing_q_flag"] = (
                bytes(missing_q_flag),
                "ACE2RT2 prompt DS32 tranche requires one host plan and RMS/Q/K/V per prompt position",
            )

            relocated_pair = bytearray(valid)
            struct.pack_into(
                "<Q",
                relocated_pair,
                command_start + 44,
                struct.unpack_from("<Q", relocated_pair, command_start + 44)[0]
                + 0x1000,
            )
            struct.pack_into(
                "<Q",
                relocated_pair,
                command_start + accepted_runtime.COMMAND.size + 28,
                struct.unpack_from(
                    "<Q",
                    relocated_pair,
                    command_start + accepted_runtime.COMMAND.size + 28,
                )[0]
                + 0x1000,
            )
            relocated_pair[56:88] = hashlib.sha256(
                bytes(relocated_pair[rope_start:])
            ).digest()
            cases["relocated_pair"] = (
                bytes(relocated_pair),
                "ACE2RT2 prompt DS32 command contract differs",
            )

            producer_low_flags = bytearray(valid)
            producer_low_flags[command_start + 9] ^= 0x01
            producer_low_flags[56:88] = hashlib.sha256(
                bytes(producer_low_flags[rope_start:])
            ).digest()
            cases["producer_low_flags"] = (
                bytes(producer_low_flags),
                "ACE2RT2 prompt DS32 command contract differs",
            )

            consumer_low_flags = bytearray(valid)
            consumer_low_flags[command_start + accepted_runtime.COMMAND.size + 9] ^= 0x01
            consumer_low_flags[56:88] = hashlib.sha256(
                bytes(consumer_low_flags[rope_start:])
            ).digest()
            cases["consumer_low_flags"] = (
                bytes(consumer_low_flags),
                "ACE2RT2 prompt DS32 command contract differs",
            )

            for name, (raw, expected_error) in cases.items():
                with self.subTest(name=name):
                    path = root / f"{name}.bin"
                    path.write_bytes(raw)
                    with self.assertRaises(RuntimeError):
                        read_ace2rt2_package_metadata(path)
                    malformed = self.inspect(path)
                    self.assertEqual(malformed.returncode, 3, malformed.stdout)
                    self.assertIn(expected_error, malformed.stderr)

    def test_dynamic_scale32_builder_rejects_frozen_address_and_low_flag_mutations(
        self,
    ) -> None:
        def relocate_pair(commands: list[dict[str, object]]) -> None:
            commands[0]["dst_addr"] = int(commands[0]["dst_addr"]) + 0x1000
            commands[1]["src0_addr"] = int(commands[1]["src0_addr"]) + 0x1000

        def mutate_producer_flags(commands: list[dict[str, object]]) -> None:
            commands[0]["flags"] = int(commands[0]["flags"]) ^ 0x01

        def mutate_consumer_flags(commands: list[dict[str, object]]) -> None:
            commands[1]["flags"] = int(commands[1]["flags"]) ^ 0x01

        mutations = (
            (
                "relocated_pair",
                relocate_pair,
                "ACE2RT2 prompt DS32 RMS command contract differs",
            ),
            (
                "producer_low_flags",
                mutate_producer_flags,
                "ACE2RT2 prompt DS32 command flags differ",
            ),
            (
                "consumer_low_flags",
                mutate_consumer_flags,
                "ACE2RT2 prompt DS32 command flags differ",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, mutator, expected_error in mutations:
                with self.subTest(name=name):
                    with self.assertRaisesRegex(RuntimeError, expected_error):
                        self.build_package(
                            root / f"{name}.bin",
                            dynamic_scale32=True,
                            command_mutator=mutator,
                        )

    def test_generalized_journal_token_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_path = root / "runtime.bin"
            self.build_package(package_path)
            generated_tokens = [101, 102]
            body = bytearray(
                struct.pack(
                    "<IQIIHBBiiI",
                    0,
                    123,
                    0,
                    0,
                    0,
                    0,
                    0,
                    103,
                    7,
                    len(generated_tokens),
                )
            )
            body.extend(struct.pack("<2I", *generated_tokens))
            body.extend(b"\0")
            body.extend(bytes(32))
            body.extend(bytes(32))
            body.extend(struct.pack("<I", 0))
            frame = (
                struct.pack("<I", len(body))
                + body
                + hashlib.sha256(body).digest()
                + struct.pack("<I", len(body))
            )
            journal_path = root / "progress.journal"
            journal_path.write_bytes(
                b"ACE2J2\0\0"
                + struct.pack("<I", 2)
                + hashlib.sha256(package_path.read_bytes()).digest()
                + frame
            )
            completed = self.inspect_journal(package_path, journal_path)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            inspected = json.loads(completed.stdout)
            self.assertEqual(inspected["records"], 1)
            self.assertEqual(inspected["generated_token_ids"], generated_tokens)
            self.assertFalse(inspected["terminated"])

    def test_header_struct_matches_specified_width(self) -> None:
        self.assertEqual(PACKAGE_V2_HEADER.size, PACKAGE_V2_HEADER_BYTES)


if __name__ == "__main__":
    unittest.main()
