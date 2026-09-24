#!/usr/bin/env python3
"""Independent checks for the layer-16 post-attention RMSNorm output repair."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ace2_layer16_post_attention_rmsnorm_output_reference import (
    MAX_REBASED_MAGNITUDE,
    OUTPUT_SHIFT,
    reference_grouped_rmsnorm_output,
)
from ace2_quality_contracts import round_divide_even_signed, scale32_ratio
import rtl_arbitrary_text_generation_backend as backend


SOURCE_CANDIDATE = ROOT / (
    "evidence/candidates/w4a8-layer16-gate-up-output-grouped-a8-repair-v1/"
    "candidate-0004"
)
SOURCE = SOURCE_CANDIDATE / "contracts/00-control"
GENERATED = ROOT / (
    "verification/generated/"
    "ace2_layer16_post_attention_rmsnorm_output_grouped_scale32"
)
POSITIONS = 30
HIDDEN = 896
GROUP_SIZE = 128


def read(path: Path, dtype: str, count: int) -> np.ndarray:
    value = np.fromfile(path, dtype=dtype)
    if value.size != count:
        raise AssertionError(f"{path.name} count differs: {value.size} != {count}")
    return value


class Layer16PostAttentionRmsnormOutputReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (GENERATED / "manifest.json").read_text(encoding="utf-8")
        )
        with safe_open(backend.canonical.MODEL, framework="pt", device="cpu") as weights:
            cls.gain = (
                weights.get_tensor(
                    "model.layers.16.post_attention_layernorm.weight"
                )
                .float()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

    def test_scope_and_control_are_frozen(self) -> None:
        result = json.loads(
            (SOURCE_CANDIDATE / "result.json").read_text(encoding="utf-8")
        )
        self.assertEqual(int(result["selected_reference_rank"]), 3252)
        self.assertEqual(int(result["selected_top_token_id"]), 12)
        self.assertTrue(result["control_exact_repeat"]["passed"])
        self.assertEqual(
            self.manifest["boundary"],
            "model.layers.16.post_attention_layernorm.output_to_signed_a8",
        )
        self.assertEqual(self.manifest["executed_contract"], "group-128")
        self.assertFalse(
            self.manifest["numeric_contract"]["runtime_bf16_sidecar"]
        )
        self.assertEqual(
            self.manifest["numeric_contract"]["carried_output"],
            "signed A8 plus explicit Scale32",
        )
        self.assertFalse(self.manifest["scope_guards"]["rank_selection_completed"])

    def test_all_model_vectors_match_bounded_integer_reference(self) -> None:
        source = read(
            SOURCE / "post_attention_sum_s8.bin", "i1", POSITIONS * HIDDEN
        ).reshape(POSITIONS, HIDDEN)
        source_scales = read(
            SOURCE / "post_attention_output_group_scale32.bin",
            "<i8",
            POSITIONS * HIDDEN,
        ).reshape(POSITIONS, HIDDEN)
        aligned_hex = np.asarray(
            [
                int(line, 16)
                for line in (GENERATED / "aligned_activation_s12.hex")
                .read_text(encoding="ascii")
                .splitlines()
            ],
            dtype=np.int64,
        )
        aligned_hex = np.where(aligned_hex & 0x800, aligned_hex - 0x1000, aligned_hex)
        gains_hex = np.asarray(
            [
                int(line, 16)
                for line in (GENERATED / "gain_s16_q8.hex")
                .read_text(encoding="ascii")
                .splitlines()
            ],
            dtype=np.int64,
        )
        gains_hex = np.where(gains_hex & 0x8000, gains_hex - 0x10000, gains_hex)
        expected_hex = np.asarray(
            [
                int(line, 16)
                for line in (GENERATED / "expected_s8.hex")
                .read_text(encoding="ascii")
                .splitlines()
            ],
            dtype=np.int64,
        )
        expected_hex = np.where(expected_hex & 0x80, expected_hex - 0x100, expected_hex)
        scale_hex = np.asarray(
            [
                int(line, 16)
                for line in (GENERATED / "output_group_scale32.hex")
                .read_text(encoding="ascii")
                .splitlines()
            ],
            dtype=np.int64,
        )
        inv_hex = np.asarray(
            [
                int(line, 16)
                for line in (GENERATED / "inv_rms_q30.hex")
                .read_text(encoding="ascii")
                .splitlines()
            ],
            dtype=np.int64,
        )

        for position in range(POSITIONS):
            result = reference_grouped_rmsnorm_output(
                source[position].astype(int).tolist(),
                source_scales[position].astype(int).tolist(),
                self.gain.tolist(),
                GROUP_SIZE,
            )
            start = position * HIDDEN
            stop = start + HIDDEN
            scale_start = position * (HIDDEN // GROUP_SIZE)
            scale_stop = scale_start + HIDDEN // GROUP_SIZE
            self.assertEqual(aligned_hex[start:stop].tolist(), result.aligned)
            self.assertEqual(gains_hex[start:stop].tolist(), result.gains_q8)
            self.assertEqual(expected_hex[start:stop].tolist(), result.outputs)
            self.assertEqual(
                scale_hex[scale_start:scale_stop].tolist(), result.output_scale32
            )
            self.assertEqual(int(inv_hex[position]), result.inv_rms_q30)

        self.assertGreater(int(np.count_nonzero(expected_hex)), 0)
        self.assertLessEqual(int(np.max(np.abs(aligned_hex))), MAX_REBASED_MAGNITUDE)
        self.assertTrue(np.all(inv_hex > 0))

    def test_q7_8_gain_floor_is_representable(self) -> None:
        gains = [
            int(line, 16)
            for line in (GENERATED / "gain_s16_q8.hex")
            .read_text(encoding="ascii")
            .splitlines()
        ]
        gains = [value - 0x10000 if value & 0x8000 else value for value in gains]
        self.assertTrue(all(-(1 << 15) <= value < (1 << 15) for value in gains))
        scales = [
            int(line, 16)
            for line in (GENERATED / "output_group_scale32.hex")
            .read_text(encoding="ascii")
            .splitlines()
        ]
        for position in range(POSITIONS):
            for group in range(HIDDEN // GROUP_SIZE):
                scale = scales[position * (HIDDEN // GROUP_SIZE) + group]
                numerator, denominator = scale32_ratio(scale)
                for channel in range(group * GROUP_SIZE, (group + 1) * GROUP_SIZE):
                    ideal = round(
                        float(self.gain[channel])
                        / (numerator / denominator)
                        * 256
                    )
                    observed = gains[position * HIDDEN + channel]
                    self.assertEqual(observed, ideal)

    def test_directed_ties_to_even_and_saturation(self) -> None:
        half = 1 << (OUTPUT_SHIFT - 1)
        self.assertEqual(round_divide_even_signed(half, 1 << OUTPUT_SHIFT), 0)
        self.assertEqual(round_divide_even_signed(3 * half, 1 << OUTPUT_SHIFT), 2)
        self.assertEqual(round_divide_even_signed(-half, 1 << OUTPUT_SHIFT), 0)
        self.assertEqual(round_divide_even_signed(-3 * half, 1 << OUTPUT_SHIFT), -2)
        self.assertGreater(2047 * 32767 * (1 << 30), 127 << OUTPUT_SHIFT)


if __name__ == "__main__":
    unittest.main()
