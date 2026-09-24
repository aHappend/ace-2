from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from tools import ace2_quality_contracts as quality
from tools import export_production_w4a8_package as exporter
from tools import grouped_w4a8_payload_conformance as conformance
from tools import model_hardware_contract as hardware
from tools import qualify_stage1_w4a8_software as runtime
from verification.test_grouped_w4a8_payload_conformance import (
    CompactPayloadDecoderStackModel,
)


class ExportProductionW4A8PackageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptor = copy.deepcopy(
            hardware.load_descriptor("qwen2.5-0.5b")
        )
        self.descriptor["dimensions"].update(
            {
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 32,
            }
        )
        unit_record = quality.pack_scale32(0x8000, 0)
        self.payloads = {}
        for index, name in enumerate(runtime.full_qwen_payload_names()):
            weights = [[0] * 32 for _ in range(32)]
            for output, row in enumerate(weights):
                row[(output + index) % 32] = -1 if index & 1 else 1
            self.payloads[name] = conformance.encode_payload(
                self.descriptor,
                32,
                weights,
                [[unit_record] for _ in range(32)],
            )

    def grouped_model(self) -> nn.Module:
        model = CompactPayloadDecoderStackModel(self.descriptor)
        runtime.install_full_qwen_payload_candidate(
            model,
            self.payloads,
            self.descriptor,
        )
        return model

    def test_execute_exports_valid_canonical_package_without_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "qwen.ace2w4m1"
            with mock.patch.object(
                exporter.hardware,
                "load_descriptor",
                return_value=self.descriptor,
            ):
                result = exporter.execute(
                    destination,
                    model_builder=lambda _descriptor, _contract: self.grouped_model(),
                )
            package = destination.read_bytes()
            decoded = conformance.decode_payload_package(
                package,
                runtime.full_qwen_payload_names(),
            )
            self.assertEqual(decoded, self.payloads)
            self.assertEqual(result["member_count"], 169)
            self.assertEqual(result["byte_count"], len(package))
            self.assertEqual(result["status"], "PASS")

            builder = mock.Mock()
            with self.assertRaises(FileExistsError):
                exporter.execute(destination, model_builder=builder)
            builder.assert_not_called()

    def test_build_uses_pinned_frozen_inputs_and_grouped_install_path(
        self,
    ) -> None:
        contract, _digest = exporter.chat.load_contract()
        manifest = exporter.evaluator.require_canonical_json(
            exporter.evaluator.PROMPT_MANIFEST_PATH
        )
        source_model = mock.Mock(spec=nn.Module)
        fixed_model = mock.Mock(spec=nn.Module)
        fixed_model.eval.return_value = fixed_model
        tokenizer = mock.Mock()
        tokenizer.chat_template = "template"

        with (
            mock.patch.object(exporter.chat, "verify_local_model") as verify_model,
            mock.patch.object(exporter.chat, "set_determinism") as set_determinism,
            mock.patch.object(
                exporter.evaluator,
                "validate_contract_support",
            ) as validate_contract,
            mock.patch.object(
                exporter.evaluator,
                "load_tokenizer",
                return_value=tokenizer,
            ),
            mock.patch.object(exporter, "validate_tokenizer") as validate_tokenizer,
            mock.patch.object(
                exporter.evaluator,
                "selected_local_texts",
                return_value=["frozen text"],
            ) as selected_texts,
            mock.patch.object(
                exporter.evaluator,
                "_tokenize_calibration",
                return_value=[torch.tensor([[1, 2]])],
            ) as tokenize_calibration,
            mock.patch.object(
                exporter.evaluator,
                "load_model",
                return_value=source_model,
            ),
            mock.patch.object(
                exporter.evaluator.fixed,
                "calibrate",
                return_value=({"linear": object()}, {"operator": object()}),
            ) as calibrate,
            mock.patch.object(
                exporter.copy,
                "deepcopy",
                return_value=fixed_model,
            ),
            mock.patch.object(
                exporter.evaluator,
                "replace_linears_for_candidate",
            ) as replace_linears,
            mock.patch.object(
                exporter.evaluator,
                "replace_fixed_operators_for_candidate",
            ) as replace_operators,
            mock.patch.object(
                exporter.runtime,
                "install_candidate",
            ) as install_candidate,
        ):
            observed = exporter.build_grouped_model(self.descriptor, contract)

        self.assertIs(observed, fixed_model)
        verify_model.assert_called_once()
        set_determinism.assert_called_once_with()
        validate_contract.assert_called_once()
        validate_tokenizer.assert_called_once_with(tokenizer, contract)
        selected_texts.assert_called_once_with(
            manifest["datasets"]["c4_calibration"]
        )
        tokenize_calibration.assert_called_once()
        calibrate.assert_called_once()
        replace_linears.assert_called_once()
        replace_operators.assert_called_once()
        install_candidate.assert_called_once_with(fixed_model, source_model)

    def test_calibration_manifest_drift_is_rejected(self) -> None:
        contract, _digest = exporter.chat.load_contract()
        manifest = exporter.evaluator.require_canonical_json(
            exporter.evaluator.PROMPT_MANIFEST_PATH
        )
        changed = copy.deepcopy(manifest)
        changed["datasets"]["c4_calibration"]["indices"]["stop"] += 1
        with self.assertRaisesRegex(RuntimeError, "calibration manifest differs"):
            exporter.validate_calibration_binding(
                contract,
                changed,
                exporter.chat.base_model_spec(contract),
            )


if __name__ == "__main__":
    unittest.main()
