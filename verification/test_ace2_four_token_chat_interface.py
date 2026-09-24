from __future__ import annotations

import copy
import hashlib
import subprocess
import unittest
from pathlib import Path

from scripts import verify_persistent_kv_multitoken_rtl_chat as verifier
from tools.ace2_v73_stage1_chat_attempt import build_parser


ROOT = Path(__file__).resolve().parents[1]


class Ace2FourTokenChatInterfaceTest(unittest.TestCase):
    @staticmethod
    def generation_fixture(tokens: int) -> tuple[dict[str, object], dict[str, object]]:
        prompt_ids = [10, 11]
        total_context = len(prompt_ids) + tokens
        argv = ["python3", "worker.py", "--max-new-tokens", str(tokens)]
        manifest = {
            "tokenization": {
                "generation_bounds": {
                    "max_new_tokens": tokens,
                    "max_prompt_tokens": 40 - tokens,
                    "prompt_token_count": len(prompt_ids),
                    "total_context_tokens": total_context,
                    "processed_positions": total_context - 1,
                    "rtl_context_bound": 40,
                }
            },
            "command": {
                "argv": argv,
                "argv_sha256": hashlib.sha256(
                    verifier.canonical_bytes(argv)
                ).hexdigest(),
            },
        }
        summary = {
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": list(range(tokens)),
            "max_new_tokens": tokens,
            "token_executions": [{} for _ in range(total_context - 1)],
            "context_contract": {
                "max_new_tokens": tokens,
                "prompt_token_count": len(prompt_ids),
                "total_context_tokens": total_context,
                "processed_positions": total_context - 1,
                "backend_context_bound": 43,
            },
        }
        return manifest, summary

    def test_attempt_defaults_to_four_generated_tokens(self) -> None:
        args = build_parser().parse_args(["--prompt", "bounded"])
        self.assertEqual(args.max_new_tokens, 4)

    def test_primary_target_runs_attempt_and_both_verifiers(self) -> None:
        completed = subprocess.run(
            [
                "make",
                "--dry-run",
                "ace2-chat",
                "ACE2_CHAT_ARGS=--prompt bounded",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        commands = completed.stdout
        self.assertIn("tools/ace2_v73_stage1_chat_attempt.py", commands)
        self.assertIn('--max-new-tokens "4"', commands)
        self.assertIn(
            "scripts/verify_persistent_kv_multitoken_rtl_chat.py",
            commands,
        )
        self.assertIn('--expected-generated-tokens "4"', commands)
        self.assertIn("scripts/verify_full_chain_independent_oracle.py", commands)

    def test_primary_target_routes_six_tokens_to_fail_closed_acceptance(self) -> None:
        completed = subprocess.run(
            [
                "make",
                "--dry-run",
                "ace2-chat",
                "ACE2_CHAT_MAX_NEW_TOKENS=6",
                "ACE2_CHAT_ARGS=--prompt bounded",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        commands = completed.stdout
        self.assertIn("run_fresh_multitoken_rtl_oracle_diagnostic.py", commands)
        self.assertIn("--attempt", commands)
        self.assertIn('--expected-generated-tokens "6"', commands)

    def test_verifier_defaults_to_four_and_supports_four_through_eight(self) -> None:
        self.assertEqual(verifier.REQUIRED_GENERATED_TOKENS, 4)
        self.assertEqual(verifier.MIN_DECODE_TRANSITIONS, 3)
        self.assertEqual(verifier.MIN_GENERATED_TOKENS, 4)
        self.assertEqual(verifier.MAX_GENERATED_TOKENS, 8)
        for tokens in range(4, 9):
            manifest, summary = self.generation_fixture(tokens)
            contract = verifier.verify_generation_contract(manifest, summary, tokens)
            self.assertEqual(contract["max_new_tokens"], tokens)
            self.assertEqual(contract["decode_transitions"], tokens - 1)

    def test_verifier_rejects_inconsistent_manifest_request(self) -> None:
        manifest, summary = self.generation_fixture(8)
        manifest["tokenization"]["generation_bounds"]["max_new_tokens"] = 4
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "manifest generated-token request differs",
        ):
            verifier.verify_generation_contract(manifest, summary, 8)

    def test_verifier_rejects_inconsistent_command_request(self) -> None:
        manifest, summary = self.generation_fixture(8)
        manifest["command"]["argv"][-1] = "4"
        manifest["command"]["argv_sha256"] = hashlib.sha256(
            verifier.canonical_bytes(manifest["command"]["argv"])
        ).hexdigest()
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "command generated-token request differs",
        ):
            verifier.verify_generation_contract(manifest, summary, 8)

    def test_verifier_rejects_inconsistent_runtime_count(self) -> None:
        manifest, summary = self.generation_fixture(8)
        summary["generated_token_ids"] = summary["generated_token_ids"][:-1]
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "runtime generated-token count differs",
        ):
            verifier.verify_generation_contract(manifest, summary, 8)

    def test_verifier_rejects_noncausal_execution_count(self) -> None:
        manifest, summary = self.generation_fixture(8)
        summary["token_executions"] = summary["token_executions"][:-1]
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "runtime execution count is not causal",
        ):
            verifier.verify_generation_contract(manifest, summary, 8)

    def test_verifier_rejects_unsupported_count(self) -> None:
        manifest, summary = self.generation_fixture(3)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "integer from 4 through 8",
        ):
            verifier.verify_generation_contract(manifest, summary, 3)

    def test_verifier_rejects_duplicate_request_flag(self) -> None:
        manifest, summary = self.generation_fixture(8)
        manifest = copy.deepcopy(manifest)
        manifest["command"]["argv"].extend(["--max-new-tokens", "8"])
        manifest["command"]["argv_sha256"] = hashlib.sha256(
            verifier.canonical_bytes(manifest["command"]["argv"])
        ).hexdigest()
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "must contain one generated-token request",
        ):
            verifier.verify_generation_contract(manifest, summary, 8)


if __name__ == "__main__":
    unittest.main()
