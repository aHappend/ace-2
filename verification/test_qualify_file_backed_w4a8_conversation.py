from __future__ import annotations

import copy
import hashlib
import unittest

from tools import qualify_file_backed_w4a8_conversation as qualification
from tools import w4a8_full_model_evaluator as evaluator


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def generation(turn_id: str, seed: int = 1) -> dict:
    input_ids = [seed, seed + 1, seed + 2]
    generated_ids = [seed + 3, seed + 4]
    caches = []
    for step in range(len(generated_ids)):
        layers = [
            {
                "key_scale32_sha256": digest(
                    f"scale-{seed}-{step}-{layer}"
                ),
                "key_sha256": digest(f"key-{seed}-{step}-{layer}"),
                "layer_index": layer,
                "sequence_length": len(input_ids) + step,
                "value_sha256": digest(f"value-{seed}-{step}-{layer}"),
            }
            for layer in range(24)
        ]
        caches.append(
            {
                "layers": layers,
                "sha256": evaluator.canonical_sha256(layers),
            }
        )
    text = f"answer-{seed}"
    return {
        "decoded_text": text,
        "decoded_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "generated_token_ids": generated_ids,
        "generated_token_ids_sha256": evaluator.canonical_sha256(generated_ids),
        "input_token_ids": input_ids,
        "per_step_integer_logit_tensor_sha256": [
            digest(f"logits-{seed}-{step}")
            for step in range(len(generated_ids))
        ],
        "per_step_kv_cache": caches,
        "per_step_kv_cache_sha256": [
            cache["sha256"] for cache in caches
        ],
        "prompt_id": turn_id,
        "termination_reason": "max_new_tokens",
        "use_cache": True,
    }


class QualifyFileBackedW4A8ConversationTest(unittest.TestCase):
    def test_exact_token_logit_and_scale32_records_are_qualified(self) -> None:
        persistent = generation("definition")
        record = qualification.compare_turn(
            persistent,
            copy.deepcopy(persistent),
            "definition",
        )

        self.assertEqual(record["turn_id"], "definition")
        self.assertEqual(
            record["generated_token_ids_sha256"],
            persistent["generated_token_ids_sha256"],
        )
        self.assertTrue(qualification.is_sha256(record["scale32_record_sha256"]))
        self.assertTrue(
            qualification.is_sha256(
                record["integer_logit_hash_record_sha256"]
            )
        )

    def test_malformed_hash_evidence_is_rejected(self) -> None:
        malformed = generation("definition")
        malformed["per_step_integer_logit_tensor_sha256"][0] = "not-a-hash"

        with self.assertRaisesRegex(RuntimeError, "integer-logit hash evidence"):
            qualification.compare_turn(
                malformed,
                generation("definition"),
                "definition",
            )

    def test_scale32_record_disagreement_is_rejected(self) -> None:
        persistent = generation("definition")
        fresh = copy.deepcopy(persistent)
        fresh["per_step_kv_cache"][0]["layers"][0][
            "key_scale32_sha256"
        ] = digest("different-scale32")
        fresh["per_step_kv_cache"][0]["sha256"] = evaluator.canonical_sha256(
            fresh["per_step_kv_cache"][0]["layers"]
        )
        fresh["per_step_kv_cache_sha256"][0] = fresh[
            "per_step_kv_cache"
        ][0]["sha256"]

        with self.assertRaisesRegex(RuntimeError, "records differ.*per_step_kv"):
            qualification.compare_turn(persistent, fresh, "definition")

    def test_generated_token_disagreement_is_rejected(self) -> None:
        persistent = generation("definition")
        fresh = generation("definition")
        fresh["generated_token_ids"][0] += 1
        fresh["generated_token_ids_sha256"] = evaluator.canonical_sha256(
            fresh["generated_token_ids"]
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "records differ.*generated_token_ids",
        ):
            qualification.compare_turn(persistent, fresh, "definition")

    def test_host_phase_metrics_are_explicit_and_validate_rss(self) -> None:
        clock_values = iter([10.0, 12.5])
        value, metrics = qualification.measure_host_phase(
            lambda: "result",
            clock=lambda: next(clock_values),
            rss_reader=lambda: 4096,
        )

        self.assertEqual(value, "result")
        self.assertEqual(metrics["wall_seconds"], 2.5)
        self.assertEqual(metrics["process_peak_rss_kib_after_phase"], 4096)
        with self.assertRaisesRegex(RuntimeError, "peak RSS"):
            qualification.measure_host_phase(
                lambda: None,
                clock=lambda: 1.0,
                rss_reader=lambda: 0,
            )


if __name__ == "__main__":
    unittest.main()
