from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from tools import ace2_chat_demo as chat
from tools import ace2_stage1_chat_product as product
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation


ROOT = Path(__file__).resolve().parents[1]


def synthetic_summary(*, decoded_text: str = "Sure, I can help.") -> dict:
    executions = []
    corrected = []
    generated = [1, 2, 3, 4]
    prompt_token_ids = [9]
    for position in range(4):
        layers = []
        for layer_id in range(backend.LAYERS):
            source = {
                "layer_id": layer_id,
                "absolute_position": position,
                "cache_length_after": position + 1,
                "k_source": "icarus_rope_stdout",
                "v_source": (
                    "icarus_ace2_shell_mem_wdata_layer23_rank1_corrected_v"
                    if layer_id == backend.LAYERS - 1
                    else "icarus_projection_stdout"
                ),
                "rtl_observed_k_sha256": "a" * 64,
                "rtl_observed_v_sha256": "b" * 64,
                "host_cache_append_replaced": True,
                "downstream_consumer_position": position + 1,
                "rank1_sidecar_executed": layer_id == backend.LAYERS - 1,
            }
            layers.append(
                {
                    "layer_id": layer_id,
                    "cache_length_before": position,
                    "cache_length_after": position + 1,
                    "rtl_cache_append": source,
                    "integer_boundary_mismatches": {
                        "rmsnorm": 0,
                        "projection": 0,
                        "rope": 0,
                        "attention_score": 0,
                        "softmax": 0,
                        "attention_compose": 0,
                        "silu": 0,
                        "residual": 0,
                        "kv_cache": 0,
                    },
                }
            )
        executions.append(
            {
                "absolute_position": position,
                "phase": "prefill" if position == 0 else "decode",
                "input_token_id": (
                    prompt_token_ids[0] if position == 0 else generated[position - 1]
                ),
                "layers": layers,
            }
        )
        corrected.append(
            {
                **layers[-1]["rtl_cache_append"],
                "downstream_consumed": position < 3,
            }
        )
    return {
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated,
        "decoded_text": decoded_text,
        "token_executions": executions,
        "head_steps": [
            {"rtl_selected_token_agreement": True} for _ in generated
        ],
        "rtl_corrected_v_provenance": corrected,
        "software_transformer_or_logits_fallback": False,
        "detokenization_wall_seconds": 0.1,
        "latency": {
            "compile_wall_seconds": 1.0,
            "model_wall_seconds": 2.0,
            "simulation_wall_seconds": 3.0,
            "total_wall_seconds": 6.0,
        },
    }


class Stage1ChatProductTest(unittest.TestCase):
    def test_plain_utf8_cli_dispatches_one_product_session(self) -> None:
        accepted = {
            "status": "PASS_STAGE1_RTL_CHAT_PRODUCT_COHERENT",
            "generated_token_ids": [1, 2, 3],
            "decoded_text": "Sure, I can help.",
            "readability": {"accepted": True},
            "latency": {
                "compile_wall_seconds": 1.0,
                "model_wall_seconds": 2.0,
                "simulation_wall_seconds": 3.0,
                "total_wall_seconds": 6.0,
            },
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "tools.ace2_chat_demo.stage1_product.run_chat",
            return_value=accepted,
        ) as run_chat, contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = chat.main(
                [
                    "--prompt",
                    "Grüße, 世界",
                    "--output",
                    str(Path(temporary) / "fresh"),
                ]
            )
        self.assertEqual(result, 0)
        run_chat.assert_called_once_with(
            "Grüße, 世界",
            Path(temporary) / "fresh",
            max_new_tokens=4,
        )
        self.assertIn("Assistant: Sure, I can help.", stdout.getvalue())

    def test_rtl_cache_bytes_replace_host_rows_and_remain_causal(self) -> None:
        cache = backend.empty_layer_cache()
        for position in range(2):
            cache["k"].append([0] * (backend.KV_HEADS * backend.HEAD_DIM))
            cache["v"].append([0] * (backend.KV_HEADS * backend.HEAD_DIM))
            rtl = {
                "_rtl_observed_cache_append": {
                    "k_s8": [position + 1] * (backend.KV_HEADS * backend.HEAD_DIM),
                    "v_s8": [position + 2] * (backend.KV_HEADS * backend.HEAD_DIM),
                },
                "kv_cache": {
                    "rtl_observed_k": {"sha256": "a" * 64},
                    "rtl_observed_v": {"sha256": "b" * 64},
                },
                "rank1_sidecar": {
                    "v_source": "icarus_ace2_shell_mem_wdata_layer23_rank1_corrected_v"
                },
            }
            record = backend.consume_rtl_cache_append(
                cache,
                rtl,
                layer_id=backend.LAYERS - 1,
                absolute_position=position,
            )
            self.assertTrue(record["host_cache_append_replaced"])
            self.assertTrue(record["rank1_sidecar_executed"])
            self.assertEqual(cache["v"][position][0], position + 2)
            self.assertEqual(len(cache["v"]), position + 1)
            if position == 1:
                self.assertEqual(cache["v"][0][0], 2)

    def test_sidecar_stdout_bytes_feed_the_next_position_cache(self) -> None:
        corrected = bytes(range(128))
        baseline = bytes(128)
        lines = []
        for beat in range(8):
            start = beat * 16
            stop = start + 16
            lines.append(
                "ACE2_HYBRID_V_BEAT "
                f"beat={beat} rank_acc=9 rank_rounded=3 rank_s8=3 "
                f"baseline={bytes(reversed(baseline[start:stop])).hex()} "
                f"corrected={bytes(reversed(corrected[start:stop])).hex()}"
            )
        observed = backend.parse_rank1_sidecar_stdout("\n".join(lines))
        cache = backend.empty_layer_cache()
        cache["k"].append([0] * (backend.KV_HEADS * backend.HEAD_DIM))
        cache["v"].append([0] * (backend.KV_HEADS * backend.HEAD_DIM))
        record = backend.consume_rtl_cache_append(
            cache,
            {
                "_rtl_observed_cache_append": {
                    "k_s8": [0] * (backend.KV_HEADS * backend.HEAD_DIM),
                    "v_s8": [
                        value - 256 if value & 0x80 else value
                        for value in observed["corrected"]
                    ],
                },
                "kv_cache": {
                    "rtl_observed_k": {"sha256": "a" * 64},
                    "rtl_observed_v": {
                        "sha256": backend.sha256_bytes(observed["corrected"])
                    },
                },
                "rank1_sidecar": {
                    "v_source": "icarus_ace2_shell_mem_wdata_layer23_rank1_corrected_v"
                },
            },
            layer_id=backend.LAYERS - 1,
            absolute_position=0,
        )
        next_position_cached_v = cache["v"][0]
        self.assertEqual(bytes(value & 0xFF for value in next_position_cached_v), corrected)
        self.assertEqual(record["downstream_consumer_position"], 1)
        self.assertTrue(record["rank1_sidecar_executed"])

    def test_run_generation_initializes_and_reuses_stage1_sidecar(self) -> None:
        sidecar = mock.Mock(compile_wall_seconds=0.125)
        weights = mock.Mock()
        tensor = mock.Mock()
        tensor.contiguous.return_value = tensor
        weights.get_tensor.return_value = tensor
        adapter = mock.Mock()
        tokenizer = mock.Mock()
        tokenizer.decode.return_value = "Hello world"
        derived_sidecars = []
        layer_call = 0

        def derive_layer(
            layer_id,
            state,
            cache,
            template,
            current_weights,
            current_adapter,
            *,
            rank1_sidecar=None,
            sidecar_working=None,
        ):
            derived_sidecars.append((rank1_sidecar, sidecar_working))
            return {}, state, template

        def run_layer(_working, _derived):
            nonlocal layer_call
            position = layer_call
            layer_call += 1
            return {
                "kv_cache": {
                    "cache_length_before": position,
                    "cache_length_after": position + 1,
                    "append_bytes": 256,
                },
                "integer_boundary_mismatches": 0,
                "timing": {},
                "compile_wall_seconds": 0.0,
                "simulation_wall_seconds": 0.0,
                "cpu_wall_seconds": 0.0,
                "icarus_wall_seconds": 0.0,
            }

        def consume_cache(cache, _rtl, *, layer_id, absolute_position):
            cache["k"].append([absolute_position])
            cache["v"].append([absolute_position])
            return {
                "layer_id": layer_id,
                "absolute_position": absolute_position,
                "cache_length_after": absolute_position + 1,
                "k_source": "icarus_rope_stdout",
                "v_source": "icarus_ace2_shell_mem_wdata_layer23_rank1_corrected_v",
                "rtl_observed_k_sha256": "a" * 64,
                "rtl_observed_v_sha256": "b" * 64,
                "host_cache_append_replaced": True,
                "downstream_consumer_position": absolute_position + 1,
                "rank1_sidecar_executed": True,
            }

        head_call = 0

        def run_head(*_args):
            nonlocal head_call
            selected = 10 + head_call
            head_call += 1
            return {
                "lm_head": {"top_token": selected, "top_logit_s8": 7},
                "timing": {},
                "compile_wall_seconds": 0.0,
                "simulation_wall_seconds": 0.0,
                "icarus_wall_seconds": 0.0,
            }

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fresh"
            with (
                mock.patch.object(backend, "LAYERS", 1),
                mock.patch.object(
                    backend, "Rank1SidecarSession", return_value=sidecar
                ) as construct_sidecar,
                mock.patch.object(
                    backend,
                    "configure_snapshot",
                    return_value={
                        "model": {"sha256": "a" * 64},
                        "config": {"sha256": "b" * 64},
                    },
                ),
                mock.patch.object(
                    backend,
                    "safe_open",
                    side_effect=[
                        contextlib.nullcontext(weights),
                        contextlib.nullcontext(adapter),
                    ],
                ),
                mock.patch.object(
                    backend,
                    "derive_lm_head_weights",
                    return_value={"artifacts": {}},
                ),
                mock.patch.object(backend, "embedding_state", return_value=object()),
                mock.patch.object(
                    backend, "derive_layer_token", side_effect=derive_layer
                ),
                mock.patch.object(backend, "run_layer_rtl", side_effect=run_layer),
                mock.patch.object(
                    backend, "consume_rtl_cache_append", side_effect=consume_cache
                ),
                mock.patch.object(
                    backend, "run_final_head_rtl", side_effect=run_head
                ),
                mock.patch.object(
                    backend,
                    "tokenizer_domain_decision",
                    return_value={"decoded_piece": "word"},
                ),
                mock.patch.object(
                    backend,
                    "file_record",
                    side_effect=lambda path: {"path": path.as_posix()},
                ),
            ):
                summary = backend.run_generation(
                    output,
                    Path("/snapshot"),
                    tokenizer,
                    [7],
                    2,
                    execution_mode="stage1_product",
                )

        construct_sidecar.assert_called_once_with(output / "shared/rank1-sidecar")
        self.assertEqual(len(derived_sidecars), 2)
        self.assertTrue(all(item[0] is sidecar for item in derived_sidecars))
        self.assertEqual(
            [item[1].relative_to(output).as_posix() for item in derived_sidecars],
            [
                "rank1-sidecar-steps/position-00/layer-00",
                "rank1-sidecar-steps/position-01/layer-00",
            ],
        )
        self.assertEqual(summary["generated_token_ids"], [10, 11])
        self.assertTrue(
            summary["rtl_corrected_v_provenance"][0]["downstream_consumed"]
        )

    def test_summary_requires_multi_token_kv_and_rtl_provenance(self) -> None:
        summary = synthetic_summary()
        self.assertTrue(
            product.validate_backend_summary(summary, "Can you help?")["accepted"]
        )
        summary["rtl_corrected_v_provenance"][0]["downstream_consumed"] = False
        summary["rtl_corrected_v_provenance"][1]["downstream_consumed"] = False
        summary["rtl_corrected_v_provenance"][2]["downstream_consumed"] = False
        with self.assertRaisesRegex(product.ProductError, "did not feed"):
            product.validate_backend_summary(summary, "Can you help?")

    def test_software_fallback_is_rejected(self) -> None:
        summary = synthetic_summary()
        summary["software_transformer_or_logits_fallback"] = True
        with self.assertRaisesRegex(product.ProductError, "fallback is forbidden"):
            product.validate_backend_summary(summary, "Can you help?")

    def test_readability_accepts_unicode_and_rejects_controls(self) -> None:
        accepted = product.readable_output_acceptance(
            "Grüße, 世界",
            "Antwort 世界",
            [31, 32],
        )
        rejected = product.readable_output_acceptance(
            "Grüße, 世界",
            "Antwort\u0000",
            [31, 32],
        )
        self.assertTrue(accepted["accepted"])
        self.assertFalse(rejected["accepted"])

    def test_product_preflight_requires_at_least_two_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            generation.PreflightError,
            "between 2 and 4",
        ):
            product.run_chat(
                "Why do registers store state?",
                Path(temporary) / "fresh",
                max_new_tokens=1,
            )

    def test_product_result_rejects_missing_latency(self) -> None:
        summary = synthetic_summary()
        summary["latency"].pop("simulation_wall_seconds")
        with self.assertRaisesRegex(product.ProductError, "latency is incomplete"):
            product.validate_backend_summary(summary, "Can you help?")

    def test_terminal_artifacts_are_exclusive(self) -> None:
        prompt = "Why do registers store state?"
        readable_summary = synthetic_summary(
            decoded_text="Registers retain circuit state"
        )
        readable = product.readable_output_acceptance(
            prompt,
            readable_summary["decoded_text"],
            readable_summary["generated_token_ids"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            product.write_terminal_result(
                output,
                {"status": "PASS_STAGE1_RTL_CHAT_PRODUCT_COHERENT"},
                readable_summary,
                readable,
            )
            self.assertTrue((output / "product_result.json").is_file())
            self.assertFalse((output / "root_cause.json").exists())

        unreadable_summary = synthetic_summary(
            decoded_text="bad\u0000text"
        )
        unreadable = product.readable_output_acceptance(
            prompt,
            unreadable_summary["decoded_text"],
            unreadable_summary["generated_token_ids"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            product.write_terminal_result(
                output,
                {"status": "BLOCKED_INCOHERENT_STAGE1_RTL_CHAT_OUTPUT"},
                unreadable_summary,
                unreadable,
            )
            self.assertFalse((output / "product_result.json").exists())
            self.assertTrue((output / "root_cause.json").is_file())


if __name__ == "__main__":
    unittest.main()
