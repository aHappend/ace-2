from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import run_fresh_multitoken_rtl_oracle_diagnostic as diagnostic


class FreshMultitokenRtlOracleDiagnosticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt = "Name one K/V cache invariant."
        self.generated_tokens = 6
        prompt_ids = [10, 11]
        generated = [20, 21, 22, 23, 24, 25]
        positions = len(prompt_ids) + len(generated) - 1
        self.summary = {
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": generated,
            "software_transformer_or_logits_fallback": False,
            "token_executions": [
                {
                    "absolute_position": position,
                    "phase": "prefill" if position < len(prompt_ids) else "decode",
                    "layers": [
                        {
                            "layer_id": layer_id,
                            "cache_length_before": position,
                            "cache_length_after": position + 1,
                        }
                        for layer_id in range(diagnostic.EXPECTED_LAYERS)
                    ],
                }
                for position in range(positions)
            ],
            "head_steps": [
                {"generation_index": index}
                for index in range(self.generated_tokens)
            ],
            "context_contract": {
                "prompt_token_count": len(prompt_ids),
                "max_new_tokens": self.generated_tokens,
                "total_context_tokens": len(prompt_ids) + len(generated),
                "processed_positions": positions,
                "backend_context_bound": max(
                    diagnostic.MAX_CONTEXT_TOKENS,
                    diagnostic.backend.MAX_CONTEXT_TOKENS,
                ),
            },
        }
        self.product = {
            "status": "PASS_STAGE1_RTL_CHAT_PRODUCT_COHERENT",
            "generated_token_ids": generated,
            "decoded_text": "K/V grows",
            "readability": {"accepted": True},
            "software_transformer_or_logits_fallback": False,
        }
        self.oracle = {
            "status": "PASS",
            "numerical_status": "PASS",
            "reused_inputs": {
                "prompt": {
                    "sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
                }
            },
            "constraints": {
                "model_output_domain": diagnostic.EXPECTED_MODEL_OUTPUTS,
            },
            "agreement": {
                "generated_token_ids": generated,
                "decoded_text": "K/V grows",
                "positions_checked": positions,
                "layers_checked": positions * diagnostic.EXPECTED_LAYERS,
                "kv_append_bytes_compared": 1,
                "full_cache_bytes_compared": 1,
                "quantized_layer_output_bytes_compared": 1,
                "quantized_layer_output_scale_bytes_compared": 1,
                "final_rmsnorm_bytes_compared": 1,
                "final_rmsnorm_scale_bytes_compared": 1,
                "full_vocabulary_head_steps_checked": self.generated_tokens,
                "full_vocabulary_logit_bytes_compared": 1,
                "full_vocabulary_logit_scale_bytes_compared": 1,
                "retained_payload_records_compared": 1,
                "retained_payload_bytes_compared": 6,
                "integer_byte_mismatches": 0,
                "selected_token_mismatches": 0,
            },
        }

    def test_records_continuous_all_layer_growth_and_five_decode_transitions(self) -> None:
        coverage = diagnostic.validate_success(
            self.prompt,
            self.oracle,
            self.summary,
            self.product,
            self.generated_tokens,
        )
        self.assertEqual(5, coverage["decode_transitions"])
        self.assertEqual(24, len(coverage["layer_cache_growth"]))
        self.assertEqual(
            list(range(1, 8)),
            [
                position["cache_length_after"]
                for position in coverage["layer_cache_growth"][23]["positions"]
            ],
        )
        self.assertFalse(coverage["software_transformer_or_logits_fallback"])

    def test_cache_length_gap_fails_closed(self) -> None:
        summary = copy.deepcopy(self.summary)
        summary["token_executions"][3]["layers"][7]["cache_length_before"] = 2
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError,
            "layer 7 cache length is discontinuous",
        ):
            diagnostic.validate_success(
                self.prompt,
                self.oracle,
                summary,
                self.product,
                self.generated_tokens,
            )

    def test_oracle_mismatch_fails_closed(self) -> None:
        oracle = copy.deepcopy(self.oracle)
        oracle["agreement"]["integer_byte_mismatches"] = 1
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError,
            "integer-byte mismatches",
        ):
            diagnostic.validate_success(
                self.prompt,
                oracle,
                self.summary,
                self.product,
                self.generated_tokens,
            )

    def test_software_fallback_fails_closed(self) -> None:
        summary = copy.deepcopy(self.summary)
        summary["software_transformer_or_logits_fallback"] = True
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError,
            "software transformer/logits fallback",
        ):
            diagnostic.validate_success(
                self.prompt,
                self.oracle,
                summary,
                self.product,
                self.generated_tokens,
            )

    def test_prompt_incompatible_product_fails_closed(self) -> None:
        product = copy.deepcopy(self.product)
        product["readability"]["accepted"] = False
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError,
            "prompt-compatible readability",
        ):
            diagnostic.validate_success(
                self.prompt,
                self.oracle,
                self.summary,
                product,
                self.generated_tokens,
            )

    def test_incomplete_retained_payload_fails_closed(self) -> None:
        oracle = copy.deepcopy(self.oracle)
        oracle["agreement"]["retained_payload_bytes_compared"] = 5
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError,
            "retained-byte comparison is incomplete",
        ):
            diagnostic.validate_success(
                self.prompt,
                oracle,
                self.summary,
                self.product,
                self.generated_tokens,
            )

    def test_launch_routes_six_tokens_to_diagnostic_worker(self) -> None:
        command = diagnostic.build_attempt_worker_command(
            Path("/tmp/prompt.utf8"),
            Path("/tmp/runtime-output"),
            Path("/tmp/worker-result.json"),
            prompt_token_count=2,
            max_new_tokens=self.generated_tokens,
        )
        self.assertEqual(
            str(Path(diagnostic.__file__).resolve()),
            command[2],
        )
        self.assertNotIn("ace2_v73_stage1_chat_attempt.py", command)
        token_index = command.index("--max-new-tokens")
        self.assertEqual("6", command[token_index + 1])

    def test_launch_routes_six_tokens_to_independent_oracle(self) -> None:
        command = diagnostic.build_independent_oracle_command(
            Path("/tmp/rtl-attempt"),
            Path("/tmp/independent-oracle"),
            self.generated_tokens,
        )
        token_index = command.index("--expected-generated-tokens")
        self.assertEqual("6", command[token_index + 1])

    def test_launch_rejects_six_tokens_beyond_context_bound(self) -> None:
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError,
            "diagnostic context bound is 40",
        ):
            diagnostic.build_attempt_worker_command(
                Path("/tmp/prompt.utf8"),
                Path("/tmp/runtime-output"),
                Path("/tmp/worker-result.json"),
                prompt_token_count=35,
                max_new_tokens=self.generated_tokens,
            )

    def test_all_supported_counts_are_accepted(self) -> None:
        for value in range(4, 9):
            with self.subTest(value=value):
                self.assertEqual(value, diagnostic.validate_generated_tokens(value))

    def test_malformed_and_out_of_range_counts_are_rejected(self) -> None:
        for value in (3, 9, 4.0, True, False, "4", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    diagnostic.DiagnosticError,
                    "integer from 4 through 8",
                ):
                    diagnostic.validate_generated_tokens(value)

    def test_both_verifiers_use_strict_four_through_eight_validation(self) -> None:
        from scripts import verify_full_chain_independent_oracle as oracle
        from scripts import verify_persistent_kv_multitoken_rtl_chat as persistent

        for value in range(4, 9):
            with self.subTest(verifier="persistent", value=value):
                self.assertEqual(
                    value,
                    persistent.validate_expected_generated_tokens(value),
                )
            with self.subTest(verifier="oracle", value=value):
                self.assertEqual(
                    value,
                    oracle.validate_expected_generated_tokens(value),
                )
        for value in (3, 9, 4.0, True, False, "4", None):
            with self.subTest(verifier="persistent", value=value):
                with self.assertRaises(persistent.VerificationError):
                    persistent.validate_expected_generated_tokens(value)
            with self.subTest(verifier="oracle", value=value):
                with self.assertRaises(persistent.VerificationError):
                    oracle.validate_expected_generated_tokens(value)

    def test_ace2_chat_routes_six_tokens_through_all_three_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "calls.log"
            shim = root / "python-shim"
            shim.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$ACE2_TEST_LOG\"\n",
                encoding="utf-8",
            )
            shim.chmod(0o755)
            environment = dict(os.environ)
            environment["ACE2_TEST_LOG"] = str(log)
            completed = subprocess.run(
                [
                    "make",
                    "--no-print-directory",
                    "ace2-chat",
                    f"PYTHON={shim}",
                    "ACE2_CHAT_MAX_NEW_TOKENS=6",
                    f"ACE2_CHAT_OUTPUT={root / 'attempt'}",
                    f"ACE2_CHAT_VERIFICATION_OUTPUT={root / 'persistent'}",
                    f"ACE2_CHAT_FULL_CHAIN_ORACLE_OUTPUT={root / 'oracle'}",
                ],
                cwd=diagnostic.ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            calls = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(3, len(calls))
            self.assertIn(
                "scripts/run_fresh_multitoken_rtl_oracle_diagnostic.py",
                calls[0],
            )
            self.assertIn("--max-new-tokens 6", calls[0])
            self.assertIn(
                "scripts/verify_persistent_kv_multitoken_rtl_chat.py",
                calls[1],
            )
            self.assertIn("--expected-generated-tokens 6", calls[1])
            self.assertIn(
                "scripts/verify_full_chain_independent_oracle.py",
                calls[2],
            )
            self.assertIn("--expected-generated-tokens 6", calls[2])

    def test_ace2_chat_rejects_malformed_and_out_of_range_counts(self) -> None:
        for value in ("", "3", "9", "4.0", "true"):
            with self.subTest(value=value):
                completed = subprocess.run(
                    [
                        "make",
                        "--no-print-directory",
                        "ace2-chat",
                        "PYTHON=/bin/true",
                        f"ACE2_CHAT_MAX_NEW_TOKENS={value}",
                    ],
                    cwd=diagnostic.ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(2, completed.returncode)
                self.assertIn(
                    "must be an integer from 4 through 8",
                    completed.stderr,
                )


if __name__ == "__main__":
    unittest.main()
