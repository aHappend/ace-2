#!/usr/bin/env python3
"""Synthetic tests for successor-aware Dynamic Scale32 recertification."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import recertify_dynamic_scale32_successor_environment as recert  # noqa: E402


class DynamicScale32SuccessorEnvironmentRecertificationTests(unittest.TestCase):
    def test_01_does_not_import_runner_or_executor(self) -> None:
        source = Path(recert.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=recert.__file__)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("run_dynamic_scale32_verification_baseline", imported)
        self.assertNotIn("ace2_dynamic_scale32_baseline", imported)

    def test_02_exactly_one_gate_is_replaced(self) -> None:
        gates = {f"gate_{index:02d}": True for index in range(34)}
        gates[recert.SOURCE_GATE] = False
        composed = recert.compose_required_gates(gates, True)
        self.assertEqual(len(composed), 35)
        self.assertTrue(all(composed.values()))
        self.assertFalse(gates[recert.SOURCE_GATE])

    def test_03_runner_mismatch_must_be_exact(self) -> None:
        summary = {
            "required_gate_count": 35,
            "passing_gate_count": 34,
            "failed_required_gate_ids": [recert.SOURCE_GATE],
            "source_mismatch_roles": ["runner"],
            "runner": {
                "accepted_recovery_sha256": recert.HISTORICAL_RUNNER_SHA256,
                "sha256": recert.EXPECTED_RUNNER_SHA256,
            },
            "no_execution_counts": {
                "before": {"attempts": 1},
                "after": {"attempts": 1},
                "runner_or_executor_invoked": False,
                "model_or_data_loaded": False,
                "model_or_data_payload_downloaded": False,
            },
        }
        self.assertTrue(recert.runner_only_historical_mismatch(summary))
        summary["source_mismatch_roles"] = ["runner", "executor"]
        self.assertFalse(recert.runner_only_historical_mismatch(summary))

    def test_04_absent_to_exactly_bound_accepted_review_transition(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ace2-review-transition-test-") as temporary:
            decision_path = Path(temporary) / "decision.json"
            sealed = recert.independent_review_state(decision_path)
            self.assertEqual(sealed["status"], "pending_fresh_reviewer")

            decision = {
                "schema_version": 1,
                "kind": "dynamic_scale32_successor_policy_independent_review_v1",
                "contract_id": "shared_token_group_dynamic_scale32_v1",
                "reviewer_status": "done",
                "status": "accepted",
                "verdict": "go",
                "independent_reviewer": True,
                "stage_closing": False,
                "bindings": recert.expected_review_bindings(),
            }

            def write_decision(value: dict[str, object]) -> None:
                content = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
                decision_path.write_bytes(content)
                digest = hashlib.sha256(content).hexdigest()
                decision_path.with_suffix(".sha256").write_text(
                    f"{digest}  {decision_path.name}\n",
                    encoding="ascii",
                )

            write_decision(decision)
            live = recert.independent_review_state(decision_path)
            self.assertEqual(live["status"], "accepted")
            self.assertTrue(recert.valid_review_transition(sealed, live, decision_path))

            mutations = {
                "missing_audit": lambda value: value["bindings"].pop(
                    "post_policy_environment_audit_sha256"
                ),
                "wrong_audit": lambda value: value["bindings"].update(
                    post_policy_environment_audit_sha256="0" * 64
                ),
                "missing_runner": lambda value: value["bindings"].pop("runner_sha256"),
                "wrong_runner": lambda value: value["bindings"].update(runner_sha256="0" * 64),
                "extra_binding": lambda value: value["bindings"].update(extra="0" * 64),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    candidate = copy.deepcopy(decision)
                    mutate(candidate)
                    write_decision(candidate)
                    invalid = recert.independent_review_state(decision_path)
                    self.assertEqual(invalid["status"], "invalid")
                    self.assertFalse(recert.valid_review_transition(sealed, invalid, decision_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
