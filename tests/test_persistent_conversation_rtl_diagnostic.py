from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_persistent_conversation_rtl_diagnostic as diagnostic
from tools import rtl_arbitrary_text_generation_backend as backend


def summary(
    start: int,
    count: int,
    retained: list[int],
    generated_count: int = 2,
) -> dict[str, object]:
    retained = list(retained)
    executed = list(range(100 + start, 100 + start + count))
    output = retained + executed
    return {
        "status": "PASS_STAGE1_PRODUCT_RTL_GENERATION_UNCHECKED",
        "software_transformer_or_logits_fallback": False,
        "layers": 24,
        "max_new_tokens": generated_count,
        "generated_token_ids": list(range(200, 200 + generated_count)),
        "termination": {
            "reason": "max_new_tokens",
            "generated_token_count": generated_count,
        },
        "head_steps": [
            {"generation_index": index}
            for index in range(generated_count)
        ],
        "prompt_token_ids": output + [999],
        "token_executions": [
            {
                "absolute_position": position,
                "input_token_id": executed[offset],
                "layers": [
                    {
                        "layer_id": layer_id,
                        "cache_length_before": position,
                        "cache_length_after": position + 1,
                    }
                    for layer_id in range(24)
                ],
            }
            for offset, position in enumerate(range(start, start + count))
        ],
        "carried_state": {
            "layer_count": 24,
            "prefix_recomputed": False,
            "retained_context_token_ids": retained,
            "retained_positions": start,
            "turn_executed_positions": count,
            "turn_start_absolute_position": start,
            "output_context_token_ids": output,
            "output_positions": len(output),
        },
    }


class PersistentConversationRtlDiagnosticTest(unittest.TestCase):
    def test_compact_conversation_suppresses_default_system_prompt(self) -> None:
        self.assertEqual(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Define cache."},
            ],
            diagnostic.conversation_messages(),
        )
        self.assertEqual(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Define cache."},
                {"role": "assistant", "content": "Cached."},
                {"role": "user", "content": "Why?"},
            ],
            diagnostic.conversation_messages(["Cached."]),
        )
        self.assertEqual(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Define cache."},
                {"role": "assistant", "content": "Cached."},
                {"role": "user", "content": "Why?"},
                {"role": "assistant", "content": "It avoids recomputation."},
                {"role": "user", "content": "Example?"},
            ],
            diagnostic.conversation_messages(
                ["Cached.", "It avoids recomputation."]
            ),
        )

    def test_continuation_preserves_noncanonical_generated_token_ids(self) -> None:
        previous = {
            "prompt_token_ids": [10, 11],
            "generated_token_ids": [20, 21, 22],
            "carried_state": {
                "output_context_token_ids": [10, 11, 20, 21],
            },
        }
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.return_value = (
            "prefix"
            + diagnostic.ASSISTANT_CONTENT_MARKER
            + "<|im_end|>\nnext-turn"
        )
        tokenizer.encode.return_value = [30, 31]

        prompt = diagnostic.render_continuation(
            tokenizer,
            previous,
            ["decoded text that retokenizes differently"],
        )

        self.assertEqual([10, 11, 20, 21, 22, 30, 31], prompt)
        tokenizer.encode.assert_called_once_with(
            "<|im_end|>\nnext-turn",
            add_special_tokens=False,
        )

    def test_continuation_rejects_inexact_carried_context(self) -> None:
        previous = {
            "prompt_token_ids": [10],
            "generated_token_ids": [20, 21],
            "carried_state": {"output_context_token_ids": [10, 99]},
        }
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError,
            "carried context differs from exact generated tokens",
        ):
            diagnostic.render_continuation(
                mock.Mock(),
                previous,
                ["reply"],
            )

    def test_accepts_turn_two_starting_at_exact_carried_boundary(self) -> None:
        first = summary(0, 3, [])
        second = summary(3, 2, first["carried_state"]["output_context_token_ids"])
        coverage = diagnostic.validate_two_turn_evidence(first, second, 2)
        self.assertEqual(24, coverage["retained_layers"])
        self.assertEqual(3, coverage["retained_positions"])
        self.assertFalse(coverage["prefix_recomputed"])

    def test_mismatched_retained_tokens_fail_closed(self) -> None:
        first = summary(0, 3, [])
        second = summary(3, 2, first["carried_state"]["output_context_token_ids"])
        second["carried_state"]["retained_context_token_ids"][1] += 1
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError,
            "retained token evidence differs",
        ):
            diagnostic.validate_two_turn_evidence(first, second, 2)

    def test_three_turn_evidence_checks_every_carried_transition(self) -> None:
        first = summary(0, 3, [])
        second = summary(3, 2, first["carried_state"]["output_context_token_ids"])
        third = summary(5, 4, second["carried_state"]["output_context_token_ids"])
        coverage = diagnostic.validate_conversation_evidence(
            [first, second, third],
            2,
        )
        self.assertEqual(3, coverage["turns_checked"])
        self.assertEqual(2, coverage["retained_transitions"])
        self.assertEqual([3, 5], [
            transition["retained_positions"]
            for transition in coverage["transitions"]
        ])
        self.assertTrue(all(
            transition["retained_layers"] == 24
            and transition["prefix_recomputed"] is False
            for transition in coverage["transitions"]
        ))

    def test_three_turn_evidence_rejects_stale_third_turn_prefix(self) -> None:
        first = summary(0, 3, [])
        second = summary(3, 2, first["carried_state"]["output_context_token_ids"])
        third = summary(5, 2, second["carried_state"]["output_context_token_ids"])
        third["carried_state"]["retained_context_token_ids"][0] += 1
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError,
            "turn three retained token evidence differs",
        ):
            diagnostic.validate_conversation_evidence([first, second, third], 2)

    def test_incomplete_turn_two_layer_evidence_fails_closed(self) -> None:
        first = summary(0, 3, [])
        second = summary(3, 2, first["carried_state"]["output_context_token_ids"])
        second["token_executions"][0]["layers"].pop()
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError,
            "does not cover 24 layers",
        ):
            diagnostic.validate_two_turn_evidence(first, second, 2)

    def test_prefix_recomputation_claim_fails_closed(self) -> None:
        first = summary(0, 3, [])
        second = summary(3, 2, first["carried_state"]["output_context_token_ids"])
        second["carried_state"]["prefix_recomputed"] = True
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError,
            "recomputed a retained prefix",
        ):
            diagnostic.validate_two_turn_evidence(first, second, 2)

    def test_backend_rejects_mismatched_and_incomplete_carried_state(self) -> None:
        state = backend.new_carried_generation_state("stage1_product")
        state.context_token_ids = [1, 2]
        state.templates = [{} for _ in range(backend.LAYERS)]
        for cache in state.caches:
            for field in cache:
                cache[field] = [object(), object()]
        snapshot = {
            "model": {"sha256": "a" * 64},
            "config": {"sha256": "b" * 64},
        }
        self.assertEqual(
            2,
            backend.validate_carried_generation_state(
                state,
                [1, 2, 3],
                snapshot,
                "stage1_product",
            ),
        )
        mismatched = copy.deepcopy(state)
        with self.assertRaisesRegex(backend.BackendError, "turn prefix differs"):
            backend.validate_carried_generation_state(
                mismatched,
                [1, 9, 3],
                snapshot,
                "stage1_product",
            )
        incomplete = copy.deepcopy(state)
        incomplete.caches.pop()
        with self.assertRaisesRegex(backend.BackendError, "24 layer caches"):
            backend.validate_carried_generation_state(
                incomplete,
                [1, 2, 3],
                snapshot,
                "stage1_product",
            )

    def test_oracle_agreement_requires_every_exact_surface(self) -> None:
        evidence = summary(0, 3, [])
        evidence["generated_token_ids"] = [7, 8]
        agreement = {
            field: 1 for field in diagnostic.REQUIRED_BYTE_COUNTS
        } | {
            "positions_checked": 3,
            "layers_checked": 72,
            "full_vocabulary_head_steps_checked": 2,
            "integer_byte_mismatches": 0,
            "full_vocabulary_logit_byte_mismatches": 0,
            "selected_token_mismatches": 0,
        }
        diagnostic.validate_oracle_agreement(evidence, agreement, "turn one")
        for field in diagnostic.REQUIRED_BYTE_COUNTS:
            with self.subTest(field=field):
                broken = copy.deepcopy(agreement)
                broken[field] = 0
                with self.assertRaisesRegex(
                    diagnostic.DiagnosticError,
                    field,
                ):
                    diagnostic.validate_oracle_agreement(
                        evidence,
                        broken,
                        "turn one",
                    )

    def test_oracle_agreement_rejects_token_mismatch(self) -> None:
        evidence = summary(0, 1, [])
        evidence["generated_token_ids"] = [7]
        agreement = {
            field: 1 for field in diagnostic.REQUIRED_BYTE_COUNTS
        } | {
            "positions_checked": 1,
            "layers_checked": 24,
            "full_vocabulary_head_steps_checked": 1,
            "integer_byte_mismatches": 0,
            "full_vocabulary_logit_byte_mismatches": 0,
            "selected_token_mismatches": 1,
        }
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError,
            "selected-token mismatches",
        ):
            diagnostic.validate_oracle_agreement(evidence, agreement, "turn one")

    def test_all_supported_token_counts_are_accepted(self) -> None:
        for value in range(2, 9):
            with self.subTest(value=value):
                self.assertEqual(value, diagnostic.validate_generated_tokens(value))

    def test_malformed_and_out_of_range_token_counts_fail_closed(self) -> None:
        for value in (1, 9, 2.0, True, False, "4", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    diagnostic.DiagnosticError,
                    "integer from 2 through 8",
                ):
                    diagnostic.validate_generated_tokens(value)

    def test_two_turn_default_and_three_turn_option_are_valid(self) -> None:
        self.assertEqual(2, diagnostic.DEFAULT_TURNS)
        self.assertEqual(2, diagnostic.validate_turns(2))
        self.assertEqual(3, diagnostic.validate_turns(3))
        for value in (1, 4, 2.0, True, "3", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    diagnostic.DiagnosticError,
                    "integer from 2 through 3",
                ):
                    diagnostic.validate_turns(value)

    def test_three_turn_context_bound_expands_without_changing_default(self) -> None:
        with (
            mock.patch.object(
                diagnostic.backend,
                "MAX_CONTEXT_TOKENS",
                diagnostic.DEFAULT_CONTEXT_TOKENS,
            ),
            mock.patch.object(
                diagnostic.generation,
                "MAX_CONTEXT_TOKENS",
                diagnostic.DEFAULT_CONTEXT_TOKENS,
            ),
        ):
            self.assertEqual(
                diagnostic.DEFAULT_CONTEXT_TOKENS,
                diagnostic.configure_context_bound(2),
            )
            self.assertEqual(
                diagnostic.DEFAULT_CONTEXT_TOKENS,
                diagnostic.backend.MAX_CONTEXT_TOKENS,
            )
            self.assertEqual(
                diagnostic.THREE_TURN_CONTEXT_TOKENS,
                diagnostic.configure_context_bound(3),
            )
            self.assertEqual(
                diagnostic.THREE_TURN_CONTEXT_TOKENS,
                diagnostic.backend.MAX_CONTEXT_TOKENS,
            )
            self.assertEqual(
                diagnostic.THREE_TURN_CONTEXT_TOKENS,
                diagnostic.generation.MAX_CONTEXT_TOKENS,
            )

    def test_invalid_token_count_is_rejected_before_rtl_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            with mock.patch.object(
                diagnostic.backend,
                "run_generation_with_state",
            ) as run_generation:
                with self.assertRaisesRegex(
                    diagnostic.DiagnosticError,
                    "integer from 2 through 8",
                ):
                    diagnostic.run(output, 9)
            run_generation.assert_not_called()
            self.assertFalse(output.exists())

    def test_invalid_turn_count_is_rejected_before_rtl_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            with mock.patch.object(
                diagnostic.backend,
                "run_generation_with_state",
            ) as run_generation:
                with self.assertRaisesRegex(
                    diagnostic.DiagnosticError,
                    "integer from 2 through 3",
                ):
                    diagnostic.run(output, 2, 4)
            run_generation.assert_not_called()
            self.assertFalse(output.exists())

    def test_oracle_aggregation_accepts_two_or_three_turns_only(self) -> None:
        agreement = {
            field: 1 for field in diagnostic.REQUIRED_BYTE_COUNTS
        } | {
            "positions_checked": 1,
            "layers_checked": 24,
            "full_vocabulary_head_steps_checked": 2,
        }
        self.assertEqual(
            2,
            diagnostic.aggregate_agreements([agreement, agreement])["turns_checked"],
        )
        self.assertEqual(
            3,
            diagnostic.aggregate_agreements(
                [agreement, agreement, agreement]
            )["turns_checked"],
        )

    def test_cache_prefixes_accumulate_prior_turn_bytes(self) -> None:
        evidence = summary(0, 1, [])
        for layer in evidence["token_executions"][0]["layers"]:
            layer["rtl_execution"] = {"path": "unused"}
        initial = [(b"prior-k", b"prior-v") for _ in range(24)]
        rtl_result = {
            "kv_cache": {
                "rtl_observed_k": {"path": "k"},
                "rtl_observed_v": {"path": "v"},
            }
        }
        with (
            mock.patch.object(
                diagnostic.oracle.sealed_verifier,
                "verify_file_record",
                return_value=Path("unused"),
            ),
            mock.patch.object(
                diagnostic.oracle.sealed_verifier,
                "load_json",
                return_value=rtl_result,
            ),
            mock.patch.object(
                diagnostic.oracle,
                "exact_artifact_bytes",
                side_effect=[value for _ in range(24) for value in (b"k", b"v")],
            ),
        ):
            accumulated = diagnostic.cache_prefixes(evidence, initial)
        self.assertEqual(
            [(b"prior-kk", b"prior-vv") for _ in range(24)],
            accumulated,
        )

    def test_turn_token_count_mismatch_fails_closed(self) -> None:
        first = summary(0, 5, [], generated_count=4)
        second = summary(
            5,
            4,
            first["carried_state"]["output_context_token_ids"],
            generated_count=3,
        )
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError,
            "turn two requested token count differs",
        ):
            diagnostic.validate_two_turn_evidence(first, second, 4)


if __name__ == "__main__":
    unittest.main()
