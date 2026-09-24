from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import run_full_qwen_command_schedule_runtime as accepted_runtime
from tools.ace2_chat_demo import (
    DIAGNOSTIC_INSTRUCT_CHAT_TEMPLATE_SHA256,
    DIAGNOSTIC_INSTRUCT_IMAGE_SHA256,
    DIAGNOSTIC_INSTRUCT_PREPARE_STATUS,
    DIAGNOSTIC_INSTRUCT_REPOSITORY,
    DIAGNOSTIC_INSTRUCT_REVISION,
    DIAGNOSTIC_INSTRUCT_RUNTIME_SOURCE_SHA256,
    DIAGNOSTIC_INSTRUCT_SOURCE_SHA256,
    PACKAGE_V2_HEADER,
    PACKAGE_V2_HEADER_BYTES,
    PACKAGE_V2_MAGIC,
    PINNED_EMBEDDING_OFFSET,
    TOKENIZER_REPOSITORY,
    TOKENIZER_SNAPSHOT,
    authenticate_diagnostic_instruct_prepare,
    diagnostic_instruct_expected_file_hashes,
    diagnostic_instruct_tokenizer_identity_record,
    load_diagnostic_instruct_schedule_inputs,
    main,
    prepare_profile,
    sha256_bytes,
    tokenizer_identity_record,
    verify_diagnostic_instruct_package_binding,
)


class FakeInstructTokenizer:
    vocab_size = 151643

    def __init__(self, chat_template: str) -> None:
        self.chat_template = chat_template

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encoded_text = text
        self.add_special_tokens = add_special_tokens
        return [101, 202, 303]

    def apply_chat_template(
        self,
        messages,
        *,
        chat_template: str,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        self.messages = messages
        self.applied_template = chat_template
        self.template_args = (tokenize, add_generation_prompt)
        return [151644, 8948, 198, 151645, 198]


class DiagnosticInstructPrepareTest(unittest.TestCase):
    @staticmethod
    def expected_hash_side_effect(
        *,
        corrupt: Path | None = None,
    ):
        expected = {
            path.resolve(): digest
            for path, digest in diagnostic_instruct_expected_file_hashes().items()
        }

        def fake_sha256(path: Path) -> str:
            resolved = Path(path).resolve()
            if corrupt is not None and resolved == corrupt.resolve():
                return "0" * 64
            if resolved in expected:
                return expected[resolved]
            if resolved == accepted_runtime.SCHEDULE.resolve():
                return accepted_runtime.EXPECTED_SCHEDULE_SHA256
            runtime_source = (
                Path(__file__).resolve().parents[1]
                / "tools/run_full_qwen_command_schedule_runtime.py"
            ).resolve()
            if resolved == runtime_source:
                return DIAGNOSTIC_INSTRUCT_RUNTIME_SOURCE_SHA256
            digest = hashlib.sha256()
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        return fake_sha256

    def authenticated_without_large_hash_reads(self) -> dict[str, object]:
        with mock.patch(
            "tools.ace2_chat_demo.sha256_file",
            side_effect=self.expected_hash_side_effect(),
        ):
            return authenticate_diagnostic_instruct_prepare()

    def test_flag_requires_prepare_only(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                "tools/ace2_chat_demo.py",
                "--diagnostic-instruct-prepare",
                "--prompt",
                "café λ",
            ],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 3, completed.stdout)
        self.assertIn(
            "--diagnostic-instruct-prepare is valid only with --prepare-only",
            completed.stderr,
        )

    def test_default_profile_and_tokenizer_identity_remain_base(self) -> None:
        profile = prepare_profile(False)
        self.assertFalse(profile["diagnostic_only"])
        self.assertEqual(profile["repository"], TOKENIZER_REPOSITORY)
        self.assertEqual(profile["revision"], accepted_runtime.REVISION)
        self.assertEqual(profile["snapshot"], TOKENIZER_SNAPSHOT)
        self.assertEqual(profile["model"], accepted_runtime.MODEL)
        self.assertEqual(profile["image"], accepted_runtime.IMAGE)
        self.assertEqual(
            sha256_bytes(tokenizer_identity_record()),
            "f2e682984c6fbad1c922bdc0810c8c38b37f2f2be01aa588f3b4ed18e6e20bdb",
        )

    def test_all_pinned_input_hashes_and_chat_template_fail_closed(self) -> None:
        for path in diagnostic_instruct_expected_file_hashes():
            with self.subTest(path=path.name), mock.patch(
                "tools.ace2_chat_demo.sha256_file",
                side_effect=self.expected_hash_side_effect(corrupt=path),
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 changed"):
                    authenticate_diagnostic_instruct_prepare()

        with mock.patch(
            "tools.ace2_chat_demo.sha256_file",
            side_effect=self.expected_hash_side_effect(),
        ), mock.patch(
            "tools.ace2_chat_demo.DIAGNOSTIC_INSTRUCT_CHAT_TEMPLATE_SHA256",
            "0" * 64,
        ):
            with self.assertRaisesRegex(RuntimeError, "chat template SHA-256 changed"):
                authenticate_diagnostic_instruct_prepare()

        authenticated = self.authenticated_without_large_hash_reads()
        for path in (
            accepted_runtime.SCHEDULE,
            Path(__file__).resolve().parents[1]
            / "tools/run_full_qwen_command_schedule_runtime.py",
        ):
            with self.subTest(path=path.name), mock.patch(
                "tools.ace2_chat_demo.sha256_file",
                side_effect=self.expected_hash_side_effect(corrupt=path),
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 changed"):
                    load_diagnostic_instruct_schedule_inputs(authenticated)

    @staticmethod
    def fake_schedule_inputs() -> dict[str, dict[str, object]]:
        return {
            "accepted_schedule": {
                "artifact": "evidence/verification/schedule.json",
                "bytes": accepted_runtime.EXPECTED_SCHEDULE_BYTES,
                "sha256": accepted_runtime.EXPECTED_SCHEDULE_SHA256,
            },
            "schedule_validation": {
                "artifact": "evidence/verification/schedule_validation.json",
                "bytes": 1,
                "sha256": "1" * 64,
            },
            "schedule_source": {
                "artifact": "tools/run_full_qwen_command_schedule_runtime.py",
                "bytes": 1,
                "sha256": DIAGNOSTIC_INSTRUCT_RUNTIME_SOURCE_SHA256,
            },
        }

    @staticmethod
    def fake_package_builder(package_path: Path, *_args, **_kwargs) -> dict[str, object]:
        tokenizer_sha256 = sha256_bytes(
            diagnostic_instruct_tokenizer_identity_record()
        )
        raw = PACKAGE_V2_HEADER.pack(
            PACKAGE_V2_MAGIC,
            2,
            PACKAGE_V2_HEADER_BYTES,
            accepted_runtime.COMMAND.size,
            1,
            5,
            8,
            2,
            151936,
            896,
            12,
            PINNED_EMBEDDING_OFFSET,
            bytes(32),
            bytes.fromhex(DIAGNOSTIC_INSTRUCT_IMAGE_SHA256),
            bytes.fromhex(DIAGNOSTIC_INSTRUCT_SOURCE_SHA256["model.safetensors"]),
            bytes.fromhex(tokenizer_sha256),
            bytes(32),
        )
        package_path.write_bytes(raw)
        return {
            "format": "ACE2RT2",
            "version": 2,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "record_bytes": accepted_runtime.COMMAND.size,
            "commands": 1,
            "prompt_token_count": 5,
            "max_new_tokens": 8,
            "termination_token_ids": [151643, 151645],
            "rope_position_count": 12,
            "tokenizer_sha256": tokenizer_sha256,
            "image_sha256": DIAGNOSTIC_INSTRUCT_IMAGE_SHA256,
            "model_sha256": DIAGNOSTIC_INSTRUCT_SOURCE_SHA256[
                "model.safetensors"
            ],
            "_ds32_applicability": {
                "status": "PASS",
                "scope": {
                    "runtime_executed": False,
                    "rtl_agreement_claim": False,
                },
                "earliest_mismatch": None,
            },
            "_first_reference_commands": [],
            "_position0_reference_commands": [],
            "_position1_reference_commands": [],
            "_position2_reference_commands": [],
            "_position3_reference_commands": [],
            "_all_reference_commands": [],
        }

    def test_provenance_is_complete_and_prepare_creates_no_runtime_outputs(self) -> None:
        authenticated = self.authenticated_without_large_hash_reads()
        chat_template = str(authenticated["chat_template"])
        tokenizer = FakeInstructTokenizer(chat_template)
        schedule_inputs = self.fake_schedule_inputs()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "instruct-prepare"
            argv = [
                "ace2_chat_demo.py",
                "--prepare-only",
                "--diagnostic-instruct-prepare",
                "--prompt",
                "Explain café λ without saying Hi.",
                "--output",
                str(output),
            ]
            captured = io.StringIO()
            with mock.patch.object(sys, "argv", argv), mock.patch(
                "tools.ace2_chat_demo.authenticate_diagnostic_instruct_prepare",
                return_value=authenticated,
            ), mock.patch(
                "tools.ace2_chat_demo.load_diagnostic_instruct_schedule_inputs",
                return_value=({}, schedule_inputs, PINNED_EMBEDDING_OFFSET, [151936, 896]),
            ), mock.patch(
                "tools.ace2_chat_demo.load_tokenizer",
                return_value=tokenizer,
            ), mock.patch(
                "tools.ace2_chat_demo.build_prompt_package",
                side_effect=self.fake_package_builder,
            ), mock.patch(
                "tools.ace2_chat_demo.subprocess.run"
            ) as runtime, contextlib.redirect_stdout(captured):
                self.assertEqual(main(), 0)

            runtime.assert_not_called()
            self.assertIn("runtime_executed=false", captured.getvalue())
            self.assertFalse((output / "runtime_invocation.json").exists())
            self.assertFalse((output / "rtl").exists())
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "ds32_prompt_applicability.json",
                    "product_contract_preflight.json",
                    "provenance.json",
                    "runtime_package.bin",
                },
            )

            provenance = json.loads(
                (output / "provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["status"], "PREPARED")
            self.assertFalse(provenance["product_acceptance_eligible"])
            self.assertFalse(provenance["completion_claim"])
            self.assertTrue(provenance["diagnostic_only"])
            self.assertFalse(provenance["runtime_executed"])
            self.assertEqual(
                provenance["tokenizer"]["repository"],
                DIAGNOSTIC_INSTRUCT_REPOSITORY,
            )
            self.assertEqual(
                provenance["tokenizer"]["revision"],
                DIAGNOSTIC_INSTRUCT_REVISION,
            )
            self.assertEqual(
                provenance["bindings"]["chat_template"]["sha256"],
                DIAGNOSTIC_INSTRUCT_CHAT_TEMPLATE_SHA256,
            )
            for key in (
                "model",
                "tokenizer",
                "tokenizer_config",
                "image",
                "image_contract",
                "validation_report",
                "source",
                "package",
            ):
                self.assertIn(key, provenance["bindings"])
            self.assertEqual(
                provenance["product_contract_preflight"]["status"],
                DIAGNOSTIC_INSTRUCT_PREPARE_STATUS,
            )
            self.assertNotIn(
                "Explain café λ without saying Hi.",
                (output / "provenance.json").read_text(encoding="utf-8"),
            )

            package_path = output / "runtime_package.bin"
            package = provenance["runtime_package"]
            verify_diagnostic_instruct_package_binding(package_path, package)
            raw = bytearray(package_path.read_bytes())
            raw[-1] ^= 1
            package_path.write_bytes(raw)
            with self.assertRaisesRegex(RuntimeError, "package SHA-256 changed"):
                verify_diagnostic_instruct_package_binding(package_path, package)


if __name__ == "__main__":
    unittest.main()
