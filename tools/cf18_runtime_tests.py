from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = load_module("_cf18_runner_test", PACKAGE / "authority_runner.py")
validator = load_module("_cf18_validator_test", PACKAGE / "validate_package.py")


class Cf18PackageTests(unittest.TestCase):
    def test_package_closure_predecessors_and_zero_state(self) -> None:
        validator.validate_package()

    def test_full_schedule_and_only_persistence_cadence_changes(self) -> None:
        candidate = json.loads(
            (PACKAGE / "authority-candidate.json").read_text(encoding="utf-8")
        )
        argv = candidate["exact_endpoint_invocation"]["argv"]
        self.assertIn("--persistence-batch-commands", argv)
        self.assertNotIn("--stop-after", argv)
        self.assertNotIn("--resume", argv)
        self.assertNotIn("--prefill-skip-intermediate-lm-head", argv)
        self.assertEqual(candidate["generated_token_claim"], "NONE_PREPARATION_ONLY")

    def test_missing_separate_authority_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            runner.AuthorityError, "separate CF18 execution authority is absent"
        ):
            runner.validate_external_authority()

    def test_duplicate_claim_is_rejected_in_temporary_namespace(self) -> None:
        candidate = json.loads(
            (PACKAGE / "authority-candidate.json").read_text(encoding="utf-8")
        )
        authority = {
            "identity": candidate["identity"],
            "nonce": candidate["nonce"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            original = runner.CANDIDATE
            candidate_path = Path(temporary) / "candidate.json"
            local_candidate = dict(candidate)
            local_candidate["state_namespace"] = str(state)
            candidate_path.write_text(json.dumps(local_candidate), encoding="utf-8")
            authority_path = Path(temporary) / "execution-authority.json"
            authority_path.write_text(json.dumps(authority), encoding="utf-8")
            runner.CANDIDATE = candidate_path
            original_authority = runner.AUTHORITY
            runner.AUTHORITY = authority_path
            try:
                runner.claim(authority, state)
                with self.assertRaisesRegex(
                    runner.AuthorityError, "already or ambiguously consumed"
                ):
                    runner.claim(authority, state)
            finally:
                runner.CANDIDATE = original
                runner.AUTHORITY = original_authority

    def test_semantic_comparison_rejects_kv_mismatch_before_decode(self) -> None:
        evidence = {
            "status": "HOST_AND_ENDPOINT_RECEIPTS_FSYNCED_NO_DECODE",
            "host": {
                "positions": [
                    {
                        "ordinal": index,
                        "selected_token_id": 10 + index,
                        "per_layer_kv_digests": ["a"] * 24,
                    }
                    for index in range(4)
                ]
            },
            "endpoint": {
                "rtl_positions": [
                    {
                        "ordinal": index,
                        "selected_token_id": 10 + index,
                        "per_layer_kv_digests": ["b"] * 24,
                    }
                    for index in range(4)
                ]
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.json"
            path.write_text(json.dumps(evidence), encoding="utf-8")
            with self.assertRaisesRegex(
                runner.AuthorityError, "semantic comparison failed"
            ):
                runner.validate_and_decode(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
