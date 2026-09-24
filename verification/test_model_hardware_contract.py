from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from tools import model_hardware_contract as contract


ROOT = Path(__file__).resolve().parents[1]


class ModelHardwareContractTest(unittest.TestCase):
    def test_all_generated_descriptors_validate_with_exact_estimates(self) -> None:
        expected = {
            "qwen2.5-0.5b": (64, 272, 581_138_960, 213_909_504),
            "qwen2.5-1.5b": (128, 528, 1_431_760_912, 1_937_768_448),
            "qwen2.5-3b": (128, 528, 2_551_366_928, 622_854_144),
            "qwen2.5-7b": (128, 1056, 5_509_576_144, 3_875_536_896),
        }
        for model_id, values in expected.items():
            with self.subTest(model_id=model_id):
                descriptor = contract.load_descriptor(model_id)
                derived = descriptor["derived"]
                self.assertEqual(
                    (
                        derived["head_dim"],
                        derived["kv_bytes_per_token_per_layer"],
                        derived["estimated_weight_bytes"],
                        derived["maximum_kv_bytes"],
                    ),
                    values,
                )

    def test_projection_metadata_matches_grouped_int4_geometry(self) -> None:
        for config in contract.QWEN_CONFIGS:
            with self.subTest(model_id=config.model_id):
                descriptor = contract.build_descriptor(config)
                layout = descriptor["weight_layout"]
                derived = descriptor["derived"]
                self.assertEqual(layout["input_group_size"], 32)
                self.assertEqual(layout["packed_group_bytes"], 16)
                self.assertEqual(layout["scale_record_bytes"], 4)
                self.assertEqual(
                    layout["weight_scale_granularity"],
                    "per_output_channel_per_input_group",
                )
                self.assertEqual(
                    derived["projection_scale_record_count"],
                    derived["packed_linear_weight_elements"] // 32,
                )
                self.assertEqual(
                    derived["packed_w4_bytes"],
                    derived["projection_scale_record_count"] * 16,
                )
                self.assertEqual(
                    derived["projection_metadata_bytes"],
                    derived["projection_scale_record_count"] * 4,
                )

    def test_incompatible_descriptors_are_rejected(self) -> None:
        base = contract.build_descriptor(contract.CONFIG_BY_ID["qwen2.5-0.5b"])
        cases = {
            "hidden_divisibility": (
                ("dimensions", "hidden_size"),
                895,
                "divisible",
            ),
            "gqa_geometry": (
                ("dimensions", "num_key_value_heads"),
                3,
                "divisible",
            ),
            "precision": (
                ("precision", "supported_modes"),
                ["w8a8"],
                "unsupported precision",
            ),
            "memory_estimate": (
                ("derived", "maximum_kv_bytes"),
                base["derived"]["maximum_kv_bytes"] + 1,
                "derived memory",
            ),
            "grouped_scale_geometry": (
                ("weight_layout", "input_group_size"),
                64,
                "weight layout",
            ),
        }
        for name, (path, replacement, error) in cases.items():
            with self.subTest(name=name):
                value = copy.deepcopy(base)
                value[path[0]][path[1]] = replacement
                with self.assertRaisesRegex(contract.ContractError, error):
                    contract.validate_descriptor(value)

    def test_runtime_preflight_binds_existing_0p5b_geometry(self) -> None:
        result = contract.runtime_preflight(
            model_id="qwen2.5-0.5b",
            embedding_shape=[151936, 896],
            max_sequence_positions=32768,
            kv_bytes_per_token_per_layer=272,
        )
        self.assertEqual(result["status"], "PASS_MODEL_HARDWARE_CONTRACT")
        self.assertEqual(
            result["quantization_policy"]["status"],
            "PASS_CURRENT_RTL_QUANTIZATION_POLICY",
        )
        self.assertEqual(result["quantization_policy"]["policy_id"], "w4a8")
        self.assertTrue(result["claims"]["package_runtime_compatibility"])
        self.assertFalse(result["claims"]["full_model_rtl_execution"])
        with self.assertRaisesRegex(contract.ContractError, "embedding shape"):
            contract.runtime_preflight(
                model_id="qwen2.5-0.5b",
                embedding_shape=[151936, 1536],
                max_sequence_positions=32768,
                kv_bytes_per_token_per_layer=272,
            )
        with self.assertRaisesRegex(contract.ContractError, "structural compatibility"):
            contract.runtime_preflight(
                model_id="qwen2.5-1.5b",
                embedding_shape=[151936, 1536],
                max_sequence_positions=131072,
                kv_bytes_per_token_per_layer=528,
            )

    def test_runtime_preflight_rejects_malformed_geometry_types(self) -> None:
        valid = {
            "model_id": "qwen2.5-0.5b",
            "embedding_shape": [151936, 896],
            "max_sequence_positions": 32768,
            "kv_bytes_per_token_per_layer": 272,
        }
        cases = {
            "embedding_vocab_boolean": ("embedding_shape", [True, 896]),
            "embedding_vocab_string": ("embedding_shape", ["151936", 896]),
            "embedding_vocab_float": ("embedding_shape", [151936.0, 896]),
            "embedding_hidden_boolean": ("embedding_shape", [151936, True]),
            "embedding_hidden_string": ("embedding_shape", [151936, "896"]),
            "embedding_hidden_float": ("embedding_shape", [151936, 896.0]),
            "sequence_boolean": ("max_sequence_positions", True),
            "sequence_string": ("max_sequence_positions", "32768"),
            "sequence_float": ("max_sequence_positions", 32768.0),
            "kv_stride_boolean": ("kv_bytes_per_token_per_layer", True),
            "kv_stride_string": ("kv_bytes_per_token_per_layer", "272"),
            "kv_stride_float": ("kv_bytes_per_token_per_layer", 272.0),
        }
        for name, (field, malformed) in cases.items():
            with self.subTest(name=name):
                values = copy.deepcopy(valid)
                values[field] = malformed
                with self.assertRaisesRegex(contract.ContractError, "positive integer"):
                    contract.runtime_preflight(**values)

    def test_cli_output_is_deterministic_and_public_safe(self) -> None:
        command = [
            sys.executable,
            "tools/model_hardware_contract.py",
            "--check",
        ]
        first = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        second = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        self.assertEqual(first.stdout, second.stdout)
        self.assertNotIn(str(ROOT).encode(), first.stdout)
        report = json.loads(first.stdout)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["descriptors"]), 4)
        self.assertFalse(report["claims"]["larger_model_rtl_execution"])
        self.assertFalse(report["claims"]["full_model_rtl_execution_performed"])


if __name__ == "__main__":
    unittest.main()
