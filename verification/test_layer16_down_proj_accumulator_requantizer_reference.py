"""Independent arithmetic checks for layer-16 down_proj accumulator requantization."""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "evidence/candidates/w4a8-layer16-down-proj-accumulator-requantization-repair-v1/"
    "candidate-0002"
)


def read_i64(path: Path) -> list[int]:
    return [item[0] for item in struct.iter_unpack("<q", path.read_bytes())]


def read_i8(path: Path) -> list[int]:
    return [value - 256 if value >= 128 else value for value in path.read_bytes()]


def round_shift_even(value: int, shift: int) -> int:
    if shift == 0:
        return value
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    base = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (base & 1)):
        base += 1
    return sign * base


def valid_scale32(record: int) -> bool:
    return 0 <= record < (1 << 24) and (record & 0xFFFF) != 0


def requantize(accumulator: int, multiplier: int, shift: int, scale32: int) -> tuple[int, int]:
    if multiplier < 0 or not valid_scale32(scale32):
        return 0, 1
    rounded = round_shift_even(accumulator * multiplier, shift)
    clipped = max(-128, min(127, rounded))
    return clipped, int(clipped != rounded)


class DownProjAccumulatorRequantizerReferenceTest(unittest.TestCase):
    def test_directed_rounding_saturation_and_invalid_scale(self) -> None:
        self.assertEqual(requantize(0, 1, 0, 0x00FB0001), (0, 0))
        self.assertEqual(requantize(1, 1, 1, 0x00FB0002), (0, 0))
        self.assertEqual(requantize(3, 1, 1, 0x00FB0003), (2, 0))
        self.assertEqual(requantize(-3, 1, 1, 0x00FB0004), (-2, 0))
        self.assertEqual(requantize(200, 1, 0, 0x00FB0005), (127, 1))
        self.assertEqual(requantize(-200, 1, 0, 0x00FB0006), (-128, 1))
        self.assertEqual(requantize(7, 1, 0, 0x01000001), (0, 1))

    def test_checkpoint176_model_vectors(self) -> None:
        result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
        group_size = int(result["rtl_validation_down_output_group_size"])
        group = CANDIDATE / f"groups/group-{group_size:03d}"
        accumulator = read_i64(group / "down_accumulator_s32.bin")
        multiplier = read_i64(group / "down_multiplier_s32.bin")
        shifts = read_i64(group / "down_shift_u6.bin")
        scales = read_i64(group / "down_output_group_scale32.bin")
        expected = read_i8(group / "down_output_s8.bin")
        expected_saturation = list((group / "down_saturation.bin").read_bytes())
        self.assertEqual(len(expected), 896)
        observed = [
            requantize(acc, mult, shift, scales[index // group_size])
            for index, (acc, mult, shift) in enumerate(
                zip(accumulator, multiplier, shifts, strict=True)
            )
        ]
        self.assertEqual([item[0] for item in observed], expected)
        self.assertEqual([item[1] for item in observed], expected_saturation)

    def test_w4_a8_scale32_bounds(self) -> None:
        result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
        group_size = int(result["rtl_validation_down_output_group_size"])
        group = CANDIDATE / f"groups/group-{group_size:03d}"
        weights = read_i8(group / "down_weight_s4.bin")
        inputs = read_i8(group / "down_input_s8.bin")
        weight_scales = read_i64(group / "down_weight_scale32.bin")
        output_scales = read_i64(group / "down_output_group_scale32.bin")
        self.assertEqual(len(weights), 896 * 4864)
        self.assertTrue(all(-8 <= value <= 7 for value in weights))
        self.assertEqual(len(inputs), 4864)
        self.assertTrue(all(-128 <= value <= 127 for value in inputs))
        self.assertEqual(len(weight_scales), 896)
        self.assertTrue(all(valid_scale32(value) for value in weight_scales))
        self.assertTrue(all(valid_scale32(value) for value in output_scales))


if __name__ == "__main__":
    unittest.main()
