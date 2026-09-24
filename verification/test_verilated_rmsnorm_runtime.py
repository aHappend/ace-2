from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import run_full_qwen_command_schedule_runtime as accepted_runtime
from tools.ace2_rmsnorm_reference import reference_rmsnorm


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "build/verilator_full_qwen_runtime/Vace2_shell_runtime_harness"
SEED_TOKEN = 89779
HIDDEN = 896
RMSNORM_GAIN_BYTES = HIDDEN * 2
RMSNORM_SCALE_OFFSET = RMSNORM_GAIN_BYTES + 8
JOURNAL_HEADER_BYTES = 8 + 4 + 32
COMPLETED_PREFIX = struct.Struct("<IQIIHBBiiI")
WRITE_RECORD = struct.Struct("<QH16s")
IMAGE_REGIONS = (
    (0x0000000100000000, 246_980_608, 0),
    (0x0000000200000000, 7_297_024, 246_980_608),
    (0x0000000300000000, 88_592, 254_277_632),
    (0x0000000310000000, 55_296, 254_366_224),
)


def image_file_offset(address: int, size: int) -> int:
    for base, region_size, file_offset in IMAGE_REGIONS:
        if base <= address and address - base + size <= region_size:
            return file_offset + address - base
    raise AssertionError(f"address is outside the accepted image: 0x{address:016x}")


def read_at(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(size)
    if len(data) != size:
        raise AssertionError(f"short read from {path}: expected {size}, got {len(data)}")
    return data


def parse_journal_records(path: Path) -> list[dict[str, object]]:
    raw = path.read_bytes()
    if raw[:8] != b"ACE2J2\0\0":
        raise AssertionError("runtime journal magic differs")
    offset = JOURNAL_HEADER_BYTES
    records = []
    while offset < len(raw):
        body_size = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        body = raw[offset : offset + body_size]
        offset += body_size
        if hashlib.sha256(body).digest() != raw[offset : offset + 32]:
            raise AssertionError("runtime journal frame digest differs")
        offset += 32
        if struct.unpack_from("<I", raw, offset)[0] != body_size:
            raise AssertionError("runtime journal frame trailer differs")
        offset += 4

        prefix = COMPLETED_PREFIX.unpack_from(body)
        body_offset = COMPLETED_PREFIX.size
        generated_tokens = []
        for _ in range(prefix[9]):
            generated_tokens.append(struct.unpack_from("<I", body, body_offset)[0])
            body_offset += 4
        terminated = bool(body[body_offset])
        body_offset += 1
        source_sha = body[body_offset : body_offset + 32]
        body_offset += 32
        destination_sha = body[body_offset : body_offset + 32]
        body_offset += 32
        write_count = struct.unpack_from("<I", body, body_offset)[0]
        body_offset += 4
        writes = []
        for _ in range(write_count):
            writes.append(WRITE_RECORD.unpack_from(body, body_offset))
            body_offset += WRITE_RECORD.size
        if body_offset != len(body):
            raise AssertionError("runtime journal record width differs")
        records.append(
            {
                "ordinal": prefix[0],
                "cycles": prefix[1],
                "read_beats": prefix[2],
                "write_beats": prefix[3],
                "done_tag": prefix[4],
                "done_error": bool(prefix[5]),
                "saturation": bool(prefix[6]),
                "generated_tokens": generated_tokens,
                "terminated": terminated,
                "source_sha": source_sha,
                "destination_sha": destination_sha,
                "writes": writes,
            }
        )
    return records


def parse_first_journal_record(path: Path) -> dict[str, object]:
    records = parse_journal_records(path)
    if len(records) != 1:
        raise AssertionError("runtime journal does not contain exactly one frame")
    return records[0]


def quantized_embedding(seed_token: int, embedding_offset: int, scale: np.float32) -> list[int]:
    raw = read_at(
        accepted_runtime.MODEL,
        embedding_offset + seed_token * HIDDEN * 2,
        HIDDEN * 2,
    )
    bf16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    values = (bf16 << 16).view(np.float32)
    return (
        np.clip(np.rint(values / scale), -128, 127)
        .astype(np.int8)
        .astype(int)
        .tolist()
    )


class VerilatedRmsnormRuntimeTest(unittest.TestCase):
    def test_one_cycle_read_response_matches_fixed_point_reference(self) -> None:
        self.assertTrue(RUNTIME.is_file(), "run `make full-qwen-runtime-build` first")
        schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
        embedding_offset, embedding_shape = accepted_runtime.embedding_tensor_offset(
            accepted_runtime.MODEL
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_path = root / "runtime_package.bin"
            output = root / "rtl"
            accepted_runtime.build_package(
                schedule, package_path, embedding_offset, embedding_shape
            )
            package = bytearray(package_path.read_bytes())
            struct.pack_into("<I", package, 20, SEED_TOKEN)
            package_path.write_bytes(package)

            completed = subprocess.run(
                [
                    str(RUNTIME),
                    "--package",
                    str(package_path),
                    "--image",
                    str(accepted_runtime.IMAGE),
                    "--model",
                    str(accepted_runtime.MODEL),
                    "--output",
                    str(output),
                    "--stop-after",
                    "1",
                    "--timeout-cycles",
                    "1000000",
                    "--read-response-latency-cycles",
                    "1",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            resumed = subprocess.run(
                [
                    str(RUNTIME),
                    "--package",
                    str(package_path),
                    "--image",
                    str(accepted_runtime.IMAGE),
                    "--model",
                    str(accepted_runtime.MODEL),
                    "--output",
                    str(output),
                    "--stop-after",
                    "1",
                    "--timeout-cycles",
                    "1000000",
                    "--read-response-latency-cycles",
                    "1",
                    "--resume",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)

            header = accepted_runtime.HEADER.unpack_from(package)
            command = accepted_runtime.COMMAND.unpack_from(
                package, accepted_runtime.HEADER.size
            )
            destination = int(command[-3])
            scale_address = int(command[-2])
            scale_raw = read_at(
                accepted_runtime.IMAGE,
                image_file_offset(scale_address + RMSNORM_SCALE_OFFSET, 8),
                8,
            )
            embedding_scale = np.float32(struct.unpack("<d", scale_raw)[0])
            gains_raw = read_at(
                accepted_runtime.IMAGE,
                image_file_offset(scale_address, RMSNORM_GAIN_BYTES),
                RMSNORM_GAIN_BYTES,
            )
            gains = np.frombuffer(gains_raw, dtype="<i2").astype(int).tolist()
            activations = quantized_embedding(
                SEED_TOKEN, int(header[7]), embedding_scale
            )
            expected = reference_rmsnorm(activations, gains)
            expected_payload = bytes(value & 0xFF for value in expected.outputs)

            record = parse_first_journal_record(output / "progress.journal")
            self.assertEqual(record["ordinal"], 0)
            self.assertEqual(record["read_beats"], 224)
            self.assertEqual(record["write_beats"], HIDDEN // 16)
            self.assertFalse(record["done_error"])
            self.assertEqual(record["saturation"], expected.saturation_seen)
            self.assertEqual(record["generated_tokens"], [])
            self.assertFalse(record["terminated"])

            actual = bytearray()
            destination_digest = hashlib.sha256()
            for beat, (address, strobe, data) in enumerate(record["writes"]):
                self.assertEqual(address, destination + beat * 16)
                self.assertEqual(strobe, 0xFFFF)
                actual.extend(data)
                destination_digest.update(struct.pack("<QH", address, strobe))
                destination_digest.update(data)

            self.assertEqual(bytes(actual), expected_payload)
            self.assertEqual(
                sum(value != 0 for value in actual),
                sum(value != 0 for value in expected_payload),
            )
            self.assertEqual(destination_digest.digest(), record["destination_sha"])


if __name__ == "__main__":
    unittest.main()
