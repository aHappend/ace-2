from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from tools import model_hardware_contract as base_contract
from tools import model_hardware_contract_phase2 as phase2


ROOT = Path(__file__).resolve().parents[1]


class Phase2ModelHardwareContractTest(unittest.TestCase):
    def test_generated_descriptors_are_canonical_and_hash_bound(self) -> None:
        expected_hashes = {
            "qwen2.5-0.5b": "3c57e17298861fe78b804f97d20de4f9e2293252db8b33d21c54db5fb82a324c",
            "qwen2.5-1.5b": "96d0b9cc1bb7a012184b3c967abadc1d1260a521c0cdc70b3a72fffc268266a9",
            "qwen2.5-3b": "23ca78ebeb3735bfe9b7e212ccdd0a7b9d603e8f911bc1768149b1bac80e718c",
            "qwen2.5-7b": "6179929fb337158d7179094bef1976a8c62f1d36cd5928dcb25319b92f410221",
        }
        for config in base_contract.QWEN_CONFIGS:
            with self.subTest(model_id=config.model_id):
                path = phase2.descriptor_path(config.model_id)
                raw = path.read_bytes()
                descriptor = json.loads(raw)
                self.assertEqual(
                    raw,
                    phase2.canonical_bytes(phase2.build_descriptor(config)),
                )
                self.assertEqual(
                    phase2.validate_descriptor(descriptor),
                    phase2.build_descriptor(config),
                )
                self.assertEqual(phase2.sha256_bytes(raw), expected_hashes[config.model_id])

    def test_product_tiers_and_required_geometry_match_directive(self) -> None:
        expected = {
            "qwen2.5-0.5b": ("ace2_nano", ["ace2_nano"], 1, 1, 15),
            "qwen2.5-1.5b": ("ace2_nano", ["ace2_nano", "ace2_edge"], 2, 2, 17),
            "qwen2.5-3b": ("ace2_edge", ["ace2_edge", "ace2_pro"], 4, 4, 15),
            "qwen2.5-7b": ("ace2_pro", ["ace2_pro"], 8, 8, 17),
        }
        for model_id, values in expected.items():
            with self.subTest(model_id=model_id):
                descriptor = phase2.build_descriptor(base_contract.CONFIG_BY_ID[model_id])
                product = descriptor["product"]
                derived = descriptor["derived"]
                self.assertEqual(
                    (
                        product["primary_tier"],
                        product["compatible_tiers"],
                        product["compute_tiles"],
                        product["memory_channels"],
                        derived["required_sequence_position_bits"],
                    ),
                    values,
                )
                self.assertEqual(
                    descriptor["hardware_interface"]["required_command_widths"][
                        "sequence_position_bits"
                    ],
                    17,
                )

    def test_generated_sv_parameters_match_0p5b_descriptor(self) -> None:
        descriptor_path = phase2.descriptor_path(phase2.SV_PARAMETER_MODEL_ID)
        descriptor = json.loads(descriptor_path.read_bytes())
        expected_values = {
            "ACE2_NUM_LAYERS": 24,
            "ACE2_HIDDEN_SIZE": 896,
            "ACE2_INTERMEDIATE_SIZE": 4864,
            "ACE2_ATTENTION_HEADS": 14,
            "ACE2_KV_HEADS": 2,
            "ACE2_HEAD_DIM": 64,
            "ACE2_KV_WIDTH": 128,
            "ACE2_VOCAB_SIZE": 151936,
            "ACE2_MAX_CONTEXT": 32768,
            "ACE2_LM_HEAD_TILE_SIZE": 32,
            "ACE2_VECTOR_LANES": 16,
        }
        self.assertEqual(phase2.sv_parameter_values(descriptor), expected_values)
        self.assertEqual(
            phase2.SV_PARAMETER_PATH.read_bytes(),
            phase2.build_sv_parameter_include(descriptor),
        )
        package_text = (ROOT / "rtl/ace2_pkg.sv").read_text(encoding="utf-8")
        self.assertIn(
            '`include "generated/ace2_model_parameters.svh"',
            package_text,
        )
        makefile_text = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn(
            "RTL_GENERATED_HEADERS := rtl/generated/ace2_model_parameters.svh",
            makefile_text,
        )
        self.assertEqual(makefile_text.count("$(RTL_GENERATED_HEADERS) \\"), 2)

    def test_projection_metadata_uses_one_scale32_per_output_group(self) -> None:
        for config in base_contract.QWEN_CONFIGS:
            with self.subTest(model_id=config.model_id):
                descriptor = phase2.build_descriptor(config)
                formats = descriptor["formats"]
                derived = descriptor["derived"]
                self.assertEqual(formats["weight_input_group_size"], 32)
                self.assertEqual(formats["packed_weight_group_bytes"], 16)
                self.assertEqual(formats["weight_scale_record_bytes"], 4)
                self.assertEqual(
                    derived["per_layer_projection_scale_record_count"],
                    derived["per_layer_linear_weight_elements"] // 32,
                )
                self.assertEqual(
                    derived["per_layer_projection_metadata_bytes"],
                    derived["per_layer_projection_scale_record_count"] * 4,
                )
                self.assertEqual(
                    derived["per_layer_packed_w4_bytes"],
                    derived["per_layer_projection_scale_record_count"] * 16,
                )

    def test_0p5b_is_gap_free_and_larger_models_record_parameterization_gaps(self) -> None:
        expected_gap_counts = {
            "qwen2.5-0.5b": 0,
            "qwen2.5-1.5b": 10,
            "qwen2.5-3b": 9,
            "qwen2.5-7b": 12,
        }
        for model_id, gap_count in expected_gap_counts.items():
            with self.subTest(model_id=model_id):
                descriptor = phase2.build_descriptor(base_contract.CONFIG_BY_ID[model_id])
                self.assertEqual(
                    len(descriptor["current_parameterization_gaps"]),
                    gap_count,
                )
                if model_id == "qwen2.5-0.5b":
                    self.assertEqual(descriptor["current_parameterization_gaps"], [])
                else:
                    gap_ids = {
                        gap["id"] for gap in descriptor["current_parameterization_gaps"]
                    }
                    self.assertIn("runtime_artifact_identity", gap_ids)

    def test_hard_coded_0p5b_inventory_is_bound_to_active_surfaces(self) -> None:
        inventory = phase2.build_inventory()
        self.assertEqual(inventory["status"], "PASS_INVENTORY_BOUND")
        self.assertEqual(len(inventory["bindings"]), 34)
        self.assertEqual(
            inventory["surface_counts"],
            {
                "command_generation": 4,
                "host_runtime": 3,
                "model_packer": 4,
                "rtl": 23,
            },
        )
        binding_ids = {record["id"] for record in inventory["bindings"]}
        for required in (
            "rtl.shell.intermediate_size",
            "rtl.core.attention_compose.head_dim_default",
            "rtl.core.attention_compose.context_default",
            "rtl.core.absolute_rope_online_attention.lane_count_default",
            "command.chat.fused_qkv_offsets",
            "host.cpp.runtime_constants",
        ):
            with self.subTest(binding_id=required):
                self.assertIn(required, binding_ids)

    def test_descriptor_and_inventory_mutations_fail_closed(self) -> None:
        descriptor = phase2.build_descriptor(base_contract.CONFIG_BY_ID["qwen2.5-0.5b"])
        mutated = copy.deepcopy(descriptor)
        mutated["dimensions"]["hidden_size"] = 1536
        with self.assertRaisesRegex(
            phase2.Phase2ContractError,
            "descriptor is not canonical",
        ):
            phase2.validate_descriptor(mutated)

        missing = phase2.Binding(
            "test.missing.fragment",
            "rtl",
            "rtl/ace2_pkg.sv",
            "missing",
            ("hidden_size",),
            "896",
            "__missing_phase2_fragment__",
            "test failure path",
        )
        with mock.patch.object(phase2, "INVENTORY_BINDINGS", (missing,)):
            with self.assertRaisesRegex(
                phase2.Phase2ContractError,
                "expected occurrence",
            ):
                phase2.build_inventory()

    def test_cli_output_is_deterministic_and_public_safe(self) -> None:
        command = [
            sys.executable,
            "tools/model_hardware_contract_phase2.py",
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
        self.assertEqual(report["inventory"]["binding_count"], 34)
        self.assertEqual(len(report["descriptors"]), 4)
        self.assertEqual(
            report["rtl_parameters"]["values"]["ACE2_HIDDEN_SIZE"],
            896,
        )
        self.assertEqual(
            report["rtl_parameters"]["model_id"],
            "qwen2.5-0.5b",
        )
        self.assertFalse(report["claims"]["larger_model_rtl_execution"])
        self.assertFalse(report["claims"]["full_model_rtl_execution"])
        self.assertFalse(report["claims"]["fpga_or_u280_deployment"])


if __name__ == "__main__":
    unittest.main()
