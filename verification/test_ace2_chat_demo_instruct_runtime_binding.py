from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import ace2_chat_demo as chat
from tools import run_full_qwen_command_schedule_runtime as accepted_runtime


class DiagnosticInstructRuntimeBindingTest(unittest.TestCase):
    @staticmethod
    def expected_hash_side_effect(*, corrupt: Path | None = None):
        expected = {
            path.resolve(): digest
            for path, digest in {
                **chat.diagnostic_instruct_expected_file_hashes(),
                **chat.diagnostic_instruct_runtime_binding_expected_file_hashes(),
            }.items()
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
                return chat.DIAGNOSTIC_INSTRUCT_RUNTIME_SOURCE_SHA256
            digest = hashlib.sha256()
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        return fake_sha256

    def authenticate_without_large_hash_reads(self) -> dict[str, object]:
        with mock.patch(
            "tools.ace2_chat_demo.sha256_file",
            side_effect=self.expected_hash_side_effect(),
        ):
            return chat.authenticate_diagnostic_instruct_runtime_binding()

    def test_help_and_default_preserve_base_and_state_zero_authority(self) -> None:
        parser = chat.build_argument_parser()
        help_text = parser.format_help()
        normalized_help = " ".join(help_text.split())
        for required in (
            "--diagnostic-instruct-runtime-binding",
            "diagnostic_only=true",
            "product_acceptance_eligible=false",
            "completion_claim=false",
            "separate external authority",
            "exact package, source, constraints, and execution-tree hashes",
        ):
            self.assertIn(required, normalized_help)

        args = parser.parse_args([])
        self.assertFalse(args.diagnostic_instruct_runtime_binding)
        self.assertFalse(args.diagnostic_instruct_prepare)
        profile = chat.prepare_profile(False)
        self.assertEqual(profile["repository"], chat.TOKENIZER_REPOSITORY)
        self.assertEqual(profile["model"], accepted_runtime.MODEL)
        self.assertEqual(profile["image"], accepted_runtime.IMAGE)

    def test_binding_mode_rejects_all_execution_or_prompt_options(self) -> None:
        cases = (
            ["--prompt", "not used"],
            ["--prompt-file", "prompt.txt"],
            ["--conversation-file", "conversation.json"],
            ["--system-prompt", "changed"],
            ["--max-new-tokens", "9"],
            ["--stop-after", "1"],
            ["--timeout-cycles", "9"],
            ["--prepare-only"],
            ["--prepare-only", "--diagnostic-instruct-prepare"],
            ["--resume"],
            ["--diagnostic-allow-base-model-chat-mismatch"],
            ["--prefill-skip-intermediate-lm-head"],
        )
        for extra in cases:
            with self.subTest(extra=extra), mock.patch(
                "tools.ace2_chat_demo.authenticate_diagnostic_instruct_runtime_binding"
            ) as authenticate, self.assertRaisesRegex(
                RuntimeError,
                "standalone non-execution mode",
            ):
                chat.main(["--diagnostic-instruct-runtime-binding", *extra])
            authenticate.assert_not_called()

    def test_successful_binding_emits_only_zero_authority_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "binding"
            captured = io.StringIO()
            with mock.patch(
                "tools.ace2_chat_demo.sha256_file",
                side_effect=self.expected_hash_side_effect(),
            ), mock.patch(
                "tools.ace2_chat_demo.subprocess.run"
            ) as runtime, mock.patch(
                "tools.ace2_chat_demo.load_tokenizer"
            ) as tokenizer, mock.patch(
                "tools.ace2_chat_demo.build_prompt_package"
            ) as package_builder, contextlib.redirect_stdout(captured):
                self.assertEqual(
                    chat.main(
                        [
                            "--diagnostic-instruct-runtime-binding",
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )

            runtime.assert_not_called()
            tokenizer.assert_not_called()
            package_builder.assert_not_called()
            self.assertIn("runtime_executed=false", captured.getvalue())
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {chat.DIAGNOSTIC_INSTRUCT_RUNTIME_BINDING_FILENAME},
            )
            self.assertFalse((output / "runtime_invocation.json").exists())
            self.assertFalse((output / "rtl").exists())

            binding = json.loads(
                (
                    output / chat.DIAGNOSTIC_INSTRUCT_RUNTIME_BINDING_FILENAME
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(binding["status"], "BOUND_DIAGNOSTIC_ONLY")
            self.assertTrue(binding["diagnostic_only"])
            self.assertFalse(binding["product_acceptance_eligible"])
            self.assertFalse(binding["completion_claim"])
            self.assertFalse(binding["runtime_invocation_authorized"])
            self.assertFalse(binding["runtime_invocation_constructed"])
            self.assertFalse(binding["runtime_binary_accessed"])
            self.assertFalse(binding["runtime_executed"])
            self.assertEqual(
                binding["accepted_prepare"]["package"]["sha256"],
                chat.DIAGNOSTIC_INSTRUCT_ACCEPTED_PACKAGE_SHA256,
            )
            self.assertFalse(
                binding["accepted_prepare"]["fresh_l2_acceptance"][
                    "execution_authority"
                ]
            )
            authority = binding["future_execution_authority"]
            self.assertTrue(authority["required"])
            self.assertEqual(authority["status"], "NOT_BOUND")
            self.assertFalse(authority["authority_present"])
            self.assertFalse(authority["runtime_invocation_authorized"])
            self.assertEqual(
                authority["requirement"],
                chat.DIAGNOSTIC_INSTRUCT_FUTURE_AUTHORITY_REQUIREMENT,
            )
            for key in (
                "accepted_package_sha256",
                "binding_host_source_sha256",
                "model_sha256",
                "tokenizer_sha256",
                "chat_template_sha256",
                "w4a8_image_sha256",
                "image_contract_sha256",
                "validation_report_sha256",
                "constraint_tree_sha256",
                "rtl_tree_sha256",
                "runtime_simulation_source_tree_sha256",
            ):
                self.assertIn(key, authority["must_bind_exact"])
            self.assertEqual(
                binding["claims"],
                {
                    "latency": False,
                    "output_readability": False,
                    "product_completion": False,
                    "rtl_reference_agreement": False,
                    "stage_closure": False,
                },
            )

    def test_all_accepted_package_files_fail_closed_on_hash_drift(self) -> None:
        for path in chat.diagnostic_instruct_runtime_binding_expected_file_hashes():
            with self.subTest(path=path.name), mock.patch(
                "tools.ace2_chat_demo.sha256_file",
                side_effect=self.expected_hash_side_effect(corrupt=path),
            ), self.assertRaisesRegex(RuntimeError, "SHA-256 changed"):
                chat.authenticate_diagnostic_instruct_runtime_binding()

    def test_semantic_rejections_cover_policy_host_hi_and_execution_tree(self) -> None:
        authenticated_binding = self.authenticate_without_large_hash_reads()
        cases = []

        def policy_drift(provenance, _preflight, _ds32):
            provenance["diagnostic_only"] = False

        cases.append(("policy", policy_drift, "provenance policy changed"))

        def missing_validation(provenance, _preflight, _ds32):
            provenance["bindings"].pop("validation_report")

        cases.append(("validation", missing_validation, "artifact binding changed"))

        def stale_host(provenance, _preflight, _ds32):
            provenance["bindings"]["source"]["host"]["sha256"] = "0" * 64

        cases.append(("host", stale_host, "host-source provenance changed"))

        def hi_prompt(provenance, _preflight, _ds32):
            provenance["prompt"]["prompt_sha256"] = hashlib.sha256(b"Hi").hexdigest()

        cases.append(("Hi", hi_prompt, "non-Hi prompt binding changed"))

        def authority_drift(_provenance, preflight, _ds32):
            preflight["execution"]["authorized"] = True

        cases.append(("authority", authority_drift, "preflight authority changed"))

        def ds32_execution(_provenance, _preflight, ds32):
            ds32["scope"]["runtime_executed"] = True

        cases.append(("DS32", ds32_execution, "DS32 scope changed"))

        def execution_tree(provenance, _preflight, _ds32):
            provenance["runtime_package"]["rtl_binding"][
                "constraint_tree_sha256"
            ] = "short"

        cases.append(("tree", execution_tree, "execution-tree binding changed"))

        for name, mutate, expected_error in cases:
            provenance = copy.deepcopy(authenticated_binding["provenance"])
            preflight = copy.deepcopy(authenticated_binding["preflight"])
            ds32 = copy.deepcopy(authenticated_binding["ds32_applicability"])
            mutate(provenance, preflight, ds32)
            with self.subTest(name=name), self.assertRaisesRegex(
                RuntimeError,
                expected_error,
            ):
                chat.validate_diagnostic_instruct_runtime_binding_provenance(
                    provenance=provenance,
                    preflight=preflight,
                    ds32_applicability=ds32,
                    authenticated=authenticated_binding["authenticated"],
                )

    def test_noncanonical_package_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "accepted"
            root.mkdir()
            outside = Path(temporary) / "outside.bin"
            outside.write_bytes(b"package")
            escaped = root / "runtime_package.bin"
            escaped.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "path is noncanonical"):
                chat._require_canonical_file(
                    escaped,
                    root,
                    identity="accepted diagnostic Instruct package",
                )

    def test_binding_source_contains_no_runtime_or_package_construction(self) -> None:
        source = "\n".join(
            (
                inspect.getsource(chat.authenticate_diagnostic_instruct_runtime_binding),
                inspect.getsource(chat.build_diagnostic_instruct_runtime_binding),
                inspect.getsource(chat.run_diagnostic_instruct_runtime_binding),
            )
        )
        for forbidden in (
            "subprocess.run",
            "accepted_runtime.DEFAULT_BINARY",
            "load_tokenizer",
            "build_prompt_package",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
