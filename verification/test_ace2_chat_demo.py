from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.ace2_attention_compose_reference import (
    AttentionComposeCase,
    AttentionComposePhaseReplay,
    reference_attention_compose,
)
from tools.ace2_chat_demo import (
    PACKAGE_MAGIC,
    PACKAGE_SEED_OFFSET,
    QKV_SCHEDULE_FUSED,
    QKV_SCHEDULE_LEGACY,
    canonical_bytes,
    build_full_prompt_commands,
    certified_rtl_binding,
    dynamic_scale32_prompt_scope,
    evaluate_dynamic_scale32_rms_outputs,
    focused_complete_layer0_commands,
    focused_complete_layer_prefix_commands,
    live_rtl_source_binding,
    patch_package_seed,
    parse_journal_v2,
    pinned_chat_product_preflight,
    prefill_skip_intermediate_lm_head_commands,
    resolve_position_token,
    silu_scale_source_binding,
    summarize_attention_value_to_o_projection,
    tokenize_prompt,
    validate_full_prompt_schedule,
    verify_positions_through_argmax,
    verify_position_zero_through_argmax,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.last_text = text
        self.last_add_special_tokens = add_special_tokens
        return [17, 23, 41]

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool):
        self.messages = messages
        self.template_args = (tokenize, add_generation_prompt)
        return [1, 2, 3, 4, 5]


class Ace2ChatDemoTest(unittest.TestCase):
    def test_pinned_base_completion_preflight_authorizes_product_candidate(self) -> None:
        from tools import run_full_qwen_command_schedule_runtime as accepted_runtime

        result = pinned_chat_product_preflight(
            observed_model_sha256=accepted_runtime.EXPECTED_MODEL_SHA256,
            diagnostic_reason=None,
        )
        self.assertEqual(
            result["status"], "PASS_PINNED_BASE_COMPLETION_CONTRACT"
        )
        self.assertTrue(result["execution"]["authorized"])
        self.assertTrue(result["execution"]["product_acceptance_eligible"])
        self.assertEqual(result["execution"]["mode"], "product_candidate")
        self.assertEqual(
            result["contract"]["minimum_visible_nonterminating_tokens"], 2
        )
        self.assertFalse(result["contract"]["instruction_following_claim"])
        self.assertEqual(result["evidence"]["observed_cases"], 3)
        self.assertEqual(
            result["model_hardware_contract"]["status"],
            "PASS_MODEL_HARDWARE_CONTRACT",
        )
        self.assertEqual(
            result["model_hardware_contract"]["geometry"][
                "kv_bytes_per_token_per_layer"
            ],
            272,
        )
        self.assertFalse(
            result["model_hardware_contract"]["claims"]["full_model_rtl_execution"]
        )
        self.assertEqual(
            result["model_hardware_contract"]["quantization_policy"]["status"],
            "PASS_CURRENT_RTL_QUANTIZATION_POLICY",
        )
        self.assertNotIn(b"decoded_text", canonical_bytes(result))

    def test_pinned_base_chat_product_preflight_allows_diagnostics_only(self) -> None:
        from tools import run_full_qwen_command_schedule_runtime as accepted_runtime

        result = pinned_chat_product_preflight(
            observed_model_sha256=accepted_runtime.EXPECTED_MODEL_SHA256,
            diagnostic_reason="bounded_stop_after",
        )
        self.assertTrue(result["execution"]["authorized"])
        self.assertEqual(result["execution"]["mode"], "diagnostic_only")
        self.assertEqual(
            result["execution"]["diagnostic_reason"], "bounded_stop_after"
        )
        self.assertFalse(result["execution"]["product_acceptance_eligible"])

    def test_pinned_base_chat_product_preflight_fails_closed_on_audit_drift(self) -> None:
        source = Path(
            "build/ace2_chat_diagnostics/"
            "reviewer-r1-bf16-production-chat-contract-20260805.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            drifted = Path(temporary) / source.name
            shutil.copyfile(source, drifted)
            with drifted.open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(RuntimeError, "audit SHA-256 changed"):
                pinned_chat_product_preflight(
                    observed_model_sha256="88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
                    diagnostic_reason=None,
                    audit_path=drifted,
                )

    def test_ordinal54_scaled_silu_is_independently_nonzero(self) -> None:
        records = parse_journal_v2(
            Path(
                "build/ace2_chat_demo/mlp-gate-integrated-20260805/rtl/progress.journal"
            )
        )

        def contiguous_payload(ordinal: int, length: int) -> bytes:
            writes = records[ordinal]["writes"]
            self.assertTrue(writes)
            base = int(writes[0][0])
            payload = bytearray(length)
            for beat, (address, strobe, data) in enumerate(writes):
                self.assertEqual(int(address), base + beat * 16)
                self.assertEqual(int(strobe), 0xFFFF)
                payload[beat * 16 : beat * 16 + 16] = data
            return bytes(payload)

        gate_raw = contiguous_payload(52, 4864)
        up_raw = contiguous_payload(53, 4864)
        self.assertEqual(
            hashlib.sha256(gate_raw).hexdigest(),
            "5cc8582c419055fdb1334151c37be410f17d95b85deed17ffc057a7b4d700802",
        )
        self.assertEqual(
            hashlib.sha256(up_raw).hexdigest(),
            "80144904396603f94be137cfabf5ae0cebc872950433b6b94f51565d632ace18",
        )

        scales = json.loads(
            Path(
                "evidence/layer0_tile_bfp_score_attention_v1/"
                "paired-smoke-20260801-v1/derived_scales.json"
            ).read_text(encoding="utf-8")
        )["linears"]
        gate_scale = float(scales["model.layers.0.mlp.gate_proj"]["output_scale"])
        up_scale = float(scales["model.layers.0.mlp.up_proj"]["output_scale"])
        signed = lambda raw: [value - 256 if value & 0x80 else value for value in raw]
        gate_q6_9 = [round(value * gate_scale * 512) for value in signed(gate_raw)]
        up_q6_9 = [round(value * up_scale * 512) for value in signed(up_raw)]
        self.assertEqual((min(gate_q6_9), max(gate_q6_9)), (-1401, 1745))
        self.assertEqual((min(up_q6_9), max(up_q6_9)), (-1177, 1919))

        manifest = json.loads(
            Path(
                "evidence/verification/build-full-qwen-packed-w4-metadata-image-v2/manifest.json"
            ).read_text(encoding="utf-8")
        )
        silu_record = next(
            record
            for record in manifest["operator_aux_records"]
            if int(record["layer"]) == 0 and record["name"] == "silu_record"
        )
        with Path(
            "evidence/verification/build-full-qwen-packed-w4-metadata-image-v2/full_model_image.bin"
        ).open("rb") as image:
            image.seek(int(silu_record["file_offset"]))
            metadata = image.read(16)
        multiplier = struct.unpack_from("<i", metadata, 0)[0]
        right_shift = metadata[4] & 0x3F
        zero_point = struct.unpack_from("<b", metadata, 5)[0]

        lut = {
            index: max(
                -32768,
                min(
                    32767,
                    round(
                        ((index / 8.0) / (1.0 + math.exp(-(index / 8.0))))
                        * 4096
                    ),
                ),
            )
            for index in range(-64, 65)
        }

        def round_shift_even(value: int, shift: int) -> int:
            if shift == 0:
                return value
            sign = -1 if value < 0 else 1
            magnitude = abs(value)
            base = magnitude >> shift
            remainder = magnitude & ((1 << shift) - 1)
            half = 1 << (shift - 1)
            if remainder > half or (remainder == half and (base & 1)):
                base += 1
            return sign * base

        outputs = []
        for gate, up in zip(gate_q6_9, up_q6_9, strict=True):
            index = max(-64, min(64, gate >> 6))
            value = round_shift_even(lut[index] * up * multiplier, right_shift)
            outputs.append(max(-128, min(127, value + zero_point)))
        output_raw = bytes(value & 0xFF for value in outputs)
        self.assertEqual(sum(value != 0 for value in output_raw), 2587)
        self.assertEqual(
            hashlib.sha256(output_raw).hexdigest(),
            "f66aa25e510b9d5cb90426bef0fe5486edef87a4af6cdd9130a168f8c68d0bbf",
        )

    def test_ds32_rms_output_plan_is_lossless_without_fixed_vector_dependency(self) -> None:
        result = evaluate_dynamic_scale32_rms_outputs([0] * 896)
        self.assertEqual(result["selected_deltas"], [0] * 7)
        self.assertEqual(result["frozen_deltas"], [0, -1, -1, -1, -1, -1, 0])
        self.assertIsNone(result["mismatch_category"])
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["selected_restored_exact"])
        self.assertTrue(result["frozen_representable"])
        self.assertTrue(result["rms_restored_exact"])

    def test_context_four_attention_compose_phase_five_replay_is_bit_exact(self) -> None:
        scores = [96, -32, 64, 16]
        values = [
            [((lane * 3 + 11) % 256) - 128 for lane in range(64)],
            [((lane * 9 + 47) % 256) - 128 for lane in range(64)],
            [((lane * 13 + 101) % 256) - 128 for lane in range(64)],
            [((lane * 17 + 151) % 256) - 128 for lane in range(64)],
        ]
        expected = reference_attention_compose(
            AttentionComposeCase("context_four", scores, values)
        )
        replay = AttentionComposePhaseReplay()
        phases = [
            replay.apply(0, scores[:1]),
            replay.apply(1, scores[1:2]),
            replay.apply(1, scores[2:3]),
            replay.apply(1, scores[3:]),
            replay.apply(2, scores[:1]),
            replay.apply(3, scores[1:2]),
            replay.apply(3, scores[2:3]),
            replay.apply(3, scores[3:]),
            replay.apply(4, scores[:1], values[:1]),
            replay.apply(5, scores[1:2], values[1:2]),
            replay.apply(5, scores[2:3], values[2:3]),
            replay.apply(6, scores[3:], values[3:]),
        ]
        self.assertTrue(all(result.outputs is None for result in phases[:-1]))
        self.assertEqual(phases[9].value_count, 2)
        self.assertEqual(phases[10].value_count, 3)
        self.assertEqual(phases[-1].outputs, expected.outputs)
        self.assertEqual(phases[-1].saturation_seen, expected.saturation_seen)
        self.assertEqual(phases[-1].max_score_q6_9, expected.max_score_q6_9)
        self.assertEqual(phases[-1].exp_sum_q15, expected.exp_sum_q15)
        self.assertEqual(phases[-1].max_count, 4)
        self.assertEqual(phases[-1].sum_count, 4)
        self.assertEqual(phases[-1].value_count, 4)

    def test_context_three_attention_compose_phase_five_replay_is_bit_exact(self) -> None:
        scores = [96, -32, 64]
        values = [
            [((lane * 3 + 11) % 256) - 128 for lane in range(64)],
            [((lane * 9 + 47) % 256) - 128 for lane in range(64)],
            [((lane * 13 + 101) % 256) - 128 for lane in range(64)],
        ]
        expected = reference_attention_compose(
            AttentionComposeCase("context_three", scores, values)
        )
        replay = AttentionComposePhaseReplay()
        phases = [
            replay.apply(0, scores[:1]),
            replay.apply(1, scores[1:2]),
            replay.apply(1, scores[2:]),
            replay.apply(2, scores[:1]),
            replay.apply(3, scores[1:2]),
            replay.apply(3, scores[2:]),
            replay.apply(4, scores[:1], values[:1]),
            replay.apply(5, scores[1:2], values[1:2]),
            replay.apply(6, scores[2:], values[2:]),
        ]
        self.assertTrue(all(result.outputs is None for result in phases[:-1]))
        self.assertEqual(phases[7].value_count, 2)
        self.assertEqual(phases[-1].outputs, expected.outputs)
        self.assertEqual(phases[-1].saturation_seen, expected.saturation_seen)
        self.assertEqual(phases[-1].max_score_q6_9, expected.max_score_q6_9)
        self.assertEqual(phases[-1].exp_sum_q15, expected.exp_sum_q15)
        self.assertEqual(phases[-1].max_count, 3)
        self.assertEqual(phases[-1].sum_count, 3)
        self.assertEqual(phases[-1].value_count, 3)

    def test_context_two_attention_compose_phase_replay_is_bit_exact(self) -> None:
        scores = [0, -64]
        values = [
            [((lane * 5 + 17) % 256) - 128 for lane in range(64)],
            [((lane * 7 + 91) % 256) - 128 for lane in range(64)],
        ]
        expected = reference_attention_compose(
            AttentionComposeCase("context_two", scores, values)
        )
        replay = AttentionComposePhaseReplay()
        phases = [
            replay.apply(0, scores[:1]),
            replay.apply(1, scores[1:]),
            replay.apply(2, scores[:1]),
            replay.apply(3, scores[1:]),
            replay.apply(4, scores[:1], values[:1]),
            replay.apply(6, scores[1:], values[1:]),
        ]
        self.assertTrue(all(result.outputs is None for result in phases[:-1]))
        self.assertEqual(phases[-1].outputs, expected.outputs)
        self.assertEqual(phases[-1].saturation_seen, expected.saturation_seen)
        self.assertEqual(phases[-1].max_score_q6_9, expected.max_score_q6_9)
        self.assertEqual(phases[-1].exp_sum_q15, expected.exp_sum_q15)
        self.assertEqual(phases[-1].max_count, 2)
        self.assertEqual(phases[-1].sum_count, 2)
        self.assertEqual(phases[-1].value_count, 2)

    def test_runtime_reports_certified_baseline_and_live_source_trees(self) -> None:
        binding = certified_rtl_binding()

        def tree_sha(patterns: tuple[str, ...]) -> tuple[int, str]:
            paths = sorted(
                {
                    path
                    for pattern in patterns
                    for path in Path(".").glob(pattern)
                    if path.is_file()
                }
            )
            manifest = "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}\n"
                for path in paths
            ).encode()
            return len(paths), hashlib.sha256(manifest).hexdigest()

        rtl_count, rtl_sha = tree_sha(("rtl/**/*.sv", "rtl/**/*.svh"))
        constraint_count, constraint_sha = tree_sha(("constraints/**/*",))
        self.assertEqual(len(binding["certificate_sha256"]), 64)
        self.assertEqual(len(binding["certified_rtl_tree_sha256"]), 64)
        self.assertEqual(binding["rtl_tree_sha256"], rtl_sha)
        self.assertEqual(binding["rtl_tree_file_count"], rtl_count)
        self.assertEqual(binding["rtl_tree_patterns"], ["rtl/**/*.sv", "rtl/**/*.svh"])
        self.assertIn(
            "rtl/generated/ace2_silu_input_scale_table.svh",
            binding["rtl_transitive_include_paths"],
        )
        self.assertEqual(
            binding["silu_scale_source_binding"]["accepted_layer_count"],
            24,
        )
        simulation_binding = binding["runtime_simulation_source_binding"]
        simulation_paths = {
            record["path"] for record in simulation_binding["source_hashes"]
        }
        self.assertEqual(simulation_binding["file_count"], rtl_count + 3)
        self.assertIn("Makefile", simulation_paths)
        self.assertIn(
            "verification/verilator/ace2_shell_runtime_harness.sv",
            simulation_paths,
        )
        self.assertIn(
            "verification/verilator/ace2_shell_runtime_main.cpp",
            simulation_paths,
        )
        self.assertEqual(binding["constraint_tree_sha256"], constraint_sha)
        self.assertEqual(binding["constraint_tree_file_count"], constraint_count)
        self.assertEqual(len(binding["certified_shell_sha256"]), 64)
        self.assertEqual(len(binding["live_shell_sha256"]), 64)
        self.assertEqual(
            binding["live_shell_matches_certified_baseline"],
            binding["certified_shell_sha256"] == binding["live_shell_sha256"],
        )

    def test_live_rtl_binding_fails_closed_on_missing_transitive_include(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rtl").mkdir()
            (root / "rtl/top.sv").write_text(
                '`include "generated/missing.svh"\nmodule top; endmodule\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "resolves to 0 files"):
                live_rtl_source_binding(root=root)

    def test_silu_scale_binding_fails_closed_on_generator_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "tools/gen_silu_gate_vectors.py",
                "rtl/generated/ace2_silu_input_scale_table.svh",
                "verification/generated/silu_input_scale_table.json",
                "evidence/verification/rtl-full-qwen-autoregressive-integration-v1/integration_inventory.json",
                "evidence/layer0_tile_bfp_score_attention_v1/paired-smoke-20260801-v1/derived_scales.json",
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(relative), destination)
            with (root / "tools/gen_silu_gate_vectors.py").open("ab") as handle:
                handle.write(b"# drift\n")
            with self.assertRaisesRegex(RuntimeError, "bound SiLU scale input differs"):
                silu_scale_source_binding(root=root)

    def test_ds32_scope_reports_live_rtl_policy_and_next_boundary(self) -> None:
        scope = dynamic_scale32_prompt_scope(
            positions_checked=3,
            unique_token_ids_checked=2,
        )
        self.assertEqual(scope["positions_checked"], 3)
        self.assertEqual(scope["unique_token_ids_checked"], 2)
        self.assertTrue(scope["reference_plan_construction"])
        self.assertTrue(scope["rtl_extended"])
        self.assertTrue(scope["later_dynamic_scale32_policy_adopted"])
        self.assertIn("q_proj", scope["rtl_execution_boundary"])
        self.assertTrue(scope["attention_value_to_o_projection_position0"])
        self.assertFalse(
            scope["attention_value_to_o_projection_user_content_positions"]
        )
        self.assertEqual(
            scope["unsupported_later_boundary"],
            "attention_residual_add_publication_into_post_attention_rmsnorm",
        )

    def test_attention_value_to_o_summary_passes_at_exact_stop(self) -> None:
        commands = [
            {
                "ordinal": head,
                "token_step": 0,
                "layer_id": 0,
                "operator": "attention_value",
                "dst_addr": 0x1000 + 64 * head,
            }
            for head in range(14)
        ]
        commands.append(
            {
                "ordinal": 14,
                "token_step": 0,
                "layer_id": 0,
                "operator": "o_proj",
                "src0_addr": 0x1000,
                "k": 896,
            }
        )
        summary = summarize_attention_value_to_o_projection(
            commands,
            {
                "status": "NOT_REACHED",
                "commands_compared": 15,
                "next_command_ordinal": 15,
                "next_operator": "attention_residual_add",
                "operator_totals": {
                    "attention_value": {"commands": 14, "bytes": 896},
                    "o_proj": {"commands": 1, "bytes": 896},
                },
                "last_verified": {
                    "command_ordinal": 14,
                    "operator": "o_proj",
                    "expected_sha256": "a" * 64,
                    "actual_sha256": "a" * 64,
                },
                "operator_last_verified": {
                    "o_proj": {
                        "command_ordinal": 185,
                        "operator": "o_proj",
                        "expected_sha256": "b" * 64,
                        "actual_sha256": "b" * 64,
                    },
                },
                "layer_operator_last_verified": {
                    "0.o_proj": {
                        "command_ordinal": 14,
                        "operator": "o_proj",
                        "expected_sha256": "a" * 64,
                        "actual_sha256": "a" * 64,
                    },
                },
            },
        )
        self.assertEqual(summary["status"], "PASS_BIT_EXACT")
        self.assertTrue(summary["publication_addresses_match_o_proj_input"])
        self.assertEqual(summary["attention_value_bytes"], 896)
        self.assertEqual(summary["o_proj_bytes"], 896)
        self.assertEqual(summary["commands_compared_through_boundary"], 15)
        self.assertEqual(summary["commands_compared_total"], 15)
        self.assertEqual(summary["o_proj_expected_sha256"], "a" * 64)
        self.assertEqual(summary["o_proj_actual_sha256"], "a" * 64)
        self.assertEqual(summary["next_unverified_ordinal"], 15)
        self.assertEqual(summary["next_unsupported_operator"], "attention_residual_add")

    def test_attention_value_to_o_summary_preserves_pass_for_extended_prefix(self) -> None:
        commands = [
            {
                "ordinal": head,
                "token_step": 0,
                "layer_id": 0,
                "operator": "attention_value",
                "dst_addr": 0x1000 + 64 * head,
            }
            for head in range(14)
        ]
        commands.extend(
            [
                {
                    "ordinal": 14,
                    "token_step": 0,
                    "layer_id": 0,
                    "operator": "o_proj",
                    "src0_addr": 0x1000,
                    "k": 896,
                },
                {
                    "ordinal": 15,
                    "token_step": 0,
                    "layer_id": 0,
                    "operator": "attention_residual_add",
                },
                {
                    "ordinal": 16,
                    "token_step": 0,
                    "layer_id": 0,
                    "operator": "post_attention_rmsnorm",
                },
                {
                    "ordinal": 17,
                    "token_step": 0,
                    "layer_id": 0,
                    "operator": "mlp_gate_proj",
                },
            ]
        )
        summary = summarize_attention_value_to_o_projection(
            commands,
            {
                "status": "NOT_REACHED",
                "commands_compared": 17,
                "next_command_ordinal": 17,
                "next_operator": "mlp_gate_proj",
                "operator_totals": {
                    "attention_value": {"commands": 14, "bytes": 896},
                    "o_proj": {"commands": 1, "bytes": 896},
                    "attention_residual_add": {"commands": 1, "bytes": 896},
                    "post_attention_rmsnorm": {"commands": 1, "bytes": 896},
                },
                "last_verified": {
                    "command_ordinal": 16,
                    "operator": "post_attention_rmsnorm",
                    "expected_sha256": "b" * 64,
                    "actual_sha256": "b" * 64,
                },
                "operator_last_verified": {
                    "o_proj": {
                        "command_ordinal": 185,
                        "operator": "o_proj",
                        "expected_sha256": "b" * 64,
                        "actual_sha256": "b" * 64,
                    },
                },
                "layer_operator_last_verified": {
                    "0.o_proj": {
                        "command_ordinal": 14,
                        "operator": "o_proj",
                        "expected_sha256": "a" * 64,
                        "actual_sha256": "a" * 64,
                    },
                    "post_attention_rmsnorm": {
                        "command_ordinal": 16,
                        "operator": "post_attention_rmsnorm",
                        "expected_sha256": "b" * 64,
                        "actual_sha256": "b" * 64,
                    },
                },
            },
        )
        self.assertEqual(summary["status"], "PASS_BIT_EXACT")
        self.assertEqual(summary["commands_compared_through_boundary"], 15)
        self.assertEqual(summary["commands_compared_total"], 17)
        self.assertEqual(summary["o_proj_expected_sha256"], "a" * 64)
        self.assertEqual(summary["o_proj_actual_sha256"], "a" * 64)
        self.assertEqual(summary["next_unverified_ordinal"], 17)
        self.assertEqual(summary["next_unsupported_operator"], "mlp_gate_proj")

    def test_attention_value_to_o_summary_preserves_pass_across_multiple_layers(self) -> None:
        commands = [
            {
                "ordinal": head,
                "token_step": 0,
                "layer_id": 0,
                "operator": "attention_value",
                "dst_addr": 0x1000 + 64 * head,
            }
            for head in range(14)
        ]
        commands.append(
            {
                "ordinal": 14,
                "token_step": 0,
                "layer_id": 0,
                "operator": "o_proj",
                "src0_addr": 0x1000,
                "k": 896,
            }
        )
        summary = summarize_attention_value_to_o_projection(
            commands,
            {
                "status": "NOT_REACHED",
                "commands_compared": 228,
                "next_command_ordinal": 228,
                "next_operator": "input_rmsnorm",
                "operator_totals": {
                    "attention_value": {"commands": 56, "bytes": 3584},
                    "o_proj": {"commands": 4, "bytes": 3584},
                },
                "operator_last_verified": {
                    "o_proj": {
                        "command_ordinal": 185,
                        "operator": "o_proj",
                        "expected_sha256": "b" * 64,
                        "actual_sha256": "b" * 64,
                    },
                },
                "layer_operator_last_verified": {
                    "0.o_proj": {
                        "command_ordinal": 14,
                        "operator": "o_proj",
                        "expected_sha256": "a" * 64,
                        "actual_sha256": "a" * 64,
                    },
                },
            },
        )
        self.assertEqual(summary["status"], "PASS_BIT_EXACT")
        self.assertEqual(summary["attention_value_commands"], 14)
        self.assertEqual(summary["aggregate_attention_value_commands"], 56)
        self.assertEqual(summary["o_proj_commands"], 1)
        self.assertEqual(summary["aggregate_o_proj_commands"], 4)

    def test_direct_entrypoint_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, "tools/ace2_chat_demo.py", "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("full-prompt ACE2RT2", completed.stdout)
        verifier = subprocess.run(
            [sys.executable, "tools/ace2_chat_verify.py", "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(verifier.returncode, 0, verifier.stderr)
        self.assertIn("existing ACE2RT2 journal", verifier.stdout)

    def test_full_prompt_schedule_covers_all_layers_kv_and_lm_tiles(self) -> None:
        from tools import run_full_qwen_command_schedule_runtime as accepted_runtime

        schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
        commands = build_full_prompt_commands(
            schedule,
            prompt_token_count=4,
            max_new_tokens=1,
        )
        self.assertEqual({command["token_step"] for command in commands}, {0, 1, 2, 3})
        for token_step in (0, 1, 2, 3):
            kv = [
                command
                for command in commands
                if command["token_step"] == token_step and command["operator"] == "kv_write"
            ]
            lm = [
                command
                for command in commands
                if command["token_step"] == token_step and command["operator"] == "lm_head_tile"
            ]
            self.assertEqual([command["layer_id"] for command in kv], list(range(24)))
            self.assertEqual([command["vocab_tile"] for command in lm], list(range(4748)))
        position_three = [command for command in commands if command["token_step"] == 3]
        self.assertEqual(position_three[0]["ordinal"], 23055)
        self.assertEqual(position_three[-1]["ordinal"], 33539)
        self.assertEqual(len(position_three), 10485)
        first_compose = [
            command
            for command in position_three
            if command["operator"] == "attention_compose"
        ][:12]
        self.assertEqual([command["ordinal"] for command in first_compose], list(range(23066, 23078)))
        self.assertEqual([command["flags"] for command in first_compose], [0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 6])
        self.assertEqual([command["context_token"] for command in first_compose], [0, 1, 2, 3] * 3)

    def test_fused_qkv_schedule_preserves_explicit_legacy_mode(self) -> None:
        from tools import run_full_qwen_command_schedule_runtime as accepted_runtime

        schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
        legacy = build_full_prompt_commands(
            schedule,
            prompt_token_count=2,
            max_new_tokens=1,
            qkv_schedule_mode=QKV_SCHEDULE_LEGACY,
        )
        fused = build_full_prompt_commands(
            schedule,
            prompt_token_count=2,
            max_new_tokens=1,
            qkv_schedule_mode=QKV_SCHEDULE_FUSED,
        )
        protected = build_full_prompt_commands(
            schedule,
            prompt_token_count=2,
            max_new_tokens=1,
            qkv_schedule_mode=QKV_SCHEDULE_FUSED,
            preserve_prompt_layer0_qkv=True,
        )
        self.assertEqual(len(legacy) - len(fused), 2 * 24 * 2)
        self.assertEqual(len(legacy) - len(protected), 2 * (24 * 2 - 2))
        self.assertEqual(
            sum(command["operator"] == "fused_qkv" for command in fused),
            24 * 2,
        )
        self.assertEqual(
            sum(command["operator"] == "fused_qkv" for command in protected),
            24 * 2 - 2,
        )
        protected_layer0 = [
            command["operator"]
            for command in protected
            if command["token_step"] == 0 and command["layer_id"] == 0
        ][:4]
        self.assertEqual(
            protected_layer0,
            ["input_rmsnorm", "q_proj", "k_proj", "v_proj"],
        )

        fused_index = next(
            index
            for index, command in enumerate(fused)
            if command["operator"] == "fused_qkv"
        )
        for field in ("src1_addr", "scale_addr", "dst_addr"):
            with self.subTest(field=field):
                corrupted = [dict(command) for command in fused]
                corrupted[fused_index][field] += 16
                with self.assertRaisesRegex(
                    RuntimeError, "fused-QKV descriptor contract differs"
                ):
                    validate_full_prompt_schedule(
                        corrupted,
                        prompt_token_count=2,
                        max_new_tokens=1,
                        qkv_schedule_mode=QKV_SCHEDULE_FUSED,
                    )

    def test_focused_complete_layer0_prefix_preserves_all_prior_positions(self) -> None:
        from tools import run_full_qwen_command_schedule_runtime as accepted_runtime

        schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
        commands = build_full_prompt_commands(
            schedule,
            prompt_token_count=3,
            max_new_tokens=1,
        )
        focused = focused_complete_layer0_commands(commands, through_position=2)
        self.assertEqual([command["ordinal"] for command in focused], list(range(len(focused))))
        self.assertEqual({command["token_step"] for command in focused}, {0, 1, 2})
        self.assertTrue(all(command["layer_id"] == 0 for command in focused))
        for position in range(3):
            position_commands = [
                command for command in focused if command["token_step"] == position
            ]
            self.assertEqual(position_commands[0]["operator"], "input_rmsnorm")
            self.assertEqual(position_commands[-1]["operator"], "mlp_residual_add")
            self.assertEqual(
                [command["context_token"] for command in position_commands if command["operator"] == "attention_score"][: position + 1],
                list(range(position + 1)),
            )

    def test_focused_complete_two_layer_prefix_preserves_each_layer(self) -> None:
        from tools import run_full_qwen_command_schedule_runtime as accepted_runtime

        schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
        commands = build_full_prompt_commands(
            schedule,
            prompt_token_count=2,
            max_new_tokens=1,
        )
        focused = focused_complete_layer_prefix_commands(
            commands,
            through_position=1,
            layer_count=2,
        )
        self.assertEqual([command["ordinal"] for command in focused], list(range(len(focused))))
        self.assertEqual({command["token_step"] for command in focused}, {0, 1})
        self.assertEqual({command["layer_id"] for command in focused}, {0, 1})
        for position in range(2):
            for layer in range(2):
                layer_commands = [
                    command
                    for command in focused
                    if command["token_step"] == position
                    and command["layer_id"] == layer
                ]
                self.assertEqual(layer_commands[0]["operator"], "input_rmsnorm")
                self.assertEqual(layer_commands[-1]["operator"], "mlp_residual_add")

    def test_layer_prefix_runtime_binding_distinguishes_live_rtl_from_package(self) -> None:
        from tools.run_chat_layer0_scaleout import runtime_rtl_binding_record

        source = {
            "rtl_tree_sha256": "a" * 64,
            "constraint_tree_sha256": "b" * 64,
        }
        live = {
            "rtl_tree_sha256": "c" * 64,
            "constraint_tree_sha256": "b" * 64,
        }
        binding = runtime_rtl_binding_record(source, live)
        self.assertEqual(binding["live"], live)
        self.assertFalse(binding["rtl_tree_matches_source_package"])
        self.assertTrue(binding["constraint_tree_matches_source_package"])

    def test_prefill_skip_omits_only_intermediate_prompt_lm_head_tiles(self) -> None:
        from tools import run_full_qwen_command_schedule_runtime as accepted_runtime

        schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
        commands = build_full_prompt_commands(
            schedule,
            prompt_token_count=3,
            max_new_tokens=2,
        )
        selected = prefill_skip_intermediate_lm_head_commands(
            commands,
            prompt_token_count=3,
        )
        self.assertEqual([command["ordinal"] for command in selected], list(range(len(selected))))
        self.assertEqual(len(commands) - len(selected), 2 * 4748)
        for position in (0, 1):
            position_commands = [
                command for command in selected if command["token_step"] == position
            ]
            self.assertEqual(position_commands[-1]["operator"], "final_rmsnorm")
            self.assertFalse(any(command["operator"] == "lm_head_tile" for command in position_commands))
        for position in (2, 3):
            position_commands = [
                command for command in selected if command["token_step"] == position
            ]
            self.assertEqual(position_commands[-1]["operator"], "lm_head_tile")
            self.assertEqual(
                sum(command["operator"] == "lm_head_tile" for command in position_commands),
                4748,
            )
        first_after_skip = next(command for command in selected if command["token_step"] == 1)
        self.assertGreater(first_after_skip["source_ordinal"], first_after_skip["ordinal"])

    def test_position_zero_wrapper_preserves_full_prompt_length(self) -> None:
        prompt_tokens = list(range(15))
        with mock.patch(
            "tools.ace2_chat_demo.verify_positions_through_argmax",
            return_value={"status": "PASS_BIT_EXACT"},
        ) as verifier:
            result = verify_position_zero_through_argmax(
                [],
                prompt_tokens=prompt_tokens,
                runtime_output=Path("unused"),
            )
        self.assertEqual(result["status"], "PASS_BIT_EXACT")
        self.assertEqual(verifier.call_args.kwargs["prompt_tokens"], prompt_tokens)
        self.assertEqual(verifier.call_args.kwargs["through_position"], 0)

    def test_generated_position_uses_prior_argmax_and_reuses_published_kv(self) -> None:
        from tools import run_full_qwen_command_schedule_runtime as accepted_runtime

        schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
        commands = build_full_prompt_commands(
            schedule,
            prompt_token_count=2,
            max_new_tokens=2,
        )
        self.assertEqual(resolve_position_token([11, 22], [], 1), (22, "prompt", None))
        self.assertEqual(
            resolve_position_token([11, 22], [198], 2),
            (198, "generated_argmax", 0),
        )
        with self.assertRaisesRegex(RuntimeError, "before its argmax completed"):
            resolve_position_token([11, 22], [], 2)

        generated = [command for command in commands if command["token_step"] == 2]
        self.assertEqual(len([c for c in generated if c["operator"] == "kv_write"]), 24)
        scores = [
            c
            for c in generated
            if c["operator"] == "attention_score"
            and c["layer_id"] == 0
            and c["query_head"] == 0
        ]
        values = [
            c
            for c in generated
            if c["operator"] == "attention_compose"
            and c["layer_id"] == 0
            and c["query_head"] == 0
            and c["flags"] >= 4
        ]
        self.assertEqual([c["context_token"] for c in scores], [0, 1, 2])
        self.assertEqual([c["context_token"] for c in values], [0, 1, 2])
        self.assertEqual([c["flags"] for c in values], [4, 5, 6])
        self.assertEqual(generated[-1]["operator"], "lm_head_tile")
        self.assertEqual(generated[-1]["vocab_tile"], 4747)

    def test_all_position_reference_fails_at_first_mismatch_after_position_four(self) -> None:
        from tools import run_full_qwen_command_schedule_runtime as accepted_runtime

        payload = bytes(32)
        argmax_token = 4747 * 32
        commands = [
            {
                "ordinal": position,
                "token_step": position,
                "sequence_position": position,
                "operator": "lm_head_tile",
                "layer_id": 24,
                "flags": 0,
                "vocab_tile": 4747,
                "completion_tag": position,
                "src0_addr": 0x1000,
                "dst_addr": 0x2000 + position * 0x100,
                "scale_addr": 0,
                "n": 32,
                "k": 896,
            }
            for position in range(6)
        ]
        records = []
        generated_tokens: list[int] = []
        for command in commands:
            position = int(command["token_step"])
            if position >= 4:
                generated_tokens.append(argmax_token)
            writes = [
                (int(command["dst_addr"]) + offset, 0xFFFF, payload[offset : offset + 16])
                for offset in range(0, len(payload), 16)
            ]
            destination_digest = hashlib.sha256()
            for address, strobe, data in writes:
                destination_digest.update(struct.pack("<QH", address, strobe))
                destination_digest.update(data)
            records.append(
                {
                    "writes": writes,
                    "destination_sha": destination_digest.digest(),
                    "done_tag": (
                        int(command["completion_tag"]) + 1
                        if position == 5
                        else int(command["completion_tag"])
                    ),
                    "done_error": False,
                    "saturation": False,
                    "argmax_token": argmax_token,
                    "argmax_logit": 0,
                    "generated_tokens": list(generated_tokens),
                    "terminated": False,
                }
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.bin"
            image.write_bytes(bytes(4096))
            with (
                mock.patch.object(accepted_runtime, "IMAGE", image),
                mock.patch(
                    "tools.ace2_chat_demo.audit_prompt_dynamic_scale32_applicability",
                    return_value={"cases": []},
                ),
                mock.patch(
                    "tools.ace2_chat_demo._PositionReferenceMemory.read",
                    side_effect=lambda address, size: bytes(size),
                ),
                mock.patch(
                    "tools.ace2_chat_demo.quantized_embedding",
                    return_value=[0],
                ),
                mock.patch(
                    "tools.ace2_chat_demo._position_expected_payload",
                    side_effect=lambda command, memory, compose_states: (
                        payload,
                        False,
                        int(command["dst_addr"]),
                    ),
                ),
            ):
                result = verify_positions_through_argmax(
                    commands,
                    prompt_tokens=[1, 2, 3, 4, 5],
                    through_position=5,
                    runtime_output=root,
                    journal_records=records,
                    max_new_tokens=2,
                )

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["commands_compared"], 6)
        self.assertEqual(result["first_mismatch_ordinal"], 5)
        self.assertEqual(result["first_mismatch"]["sequence_position"], 5)
        self.assertFalse(result["first_mismatch"]["completion_tag_matches"])
        self.assertEqual(result["generated_token_ids"], [argmax_token, argmax_token])

    def test_tokenization_metadata_redacts_prompt(self) -> None:
        prompt = "non-frozen synthetic prompt"
        metadata = tokenize_prompt(FakeTokenizer(), prompt, "system")
        self.assertEqual(metadata["last_user_content_token_id"], 41)
        self.assertEqual(metadata["user_token_count"], 3)
        self.assertEqual(metadata["chat_template_token_count"], 5)
        serialized = canonical_bytes(metadata)
        self.assertNotIn(prompt.encode(), serialized)

    def test_package_patch_changes_only_seed_word(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.bin"
            raw = bytearray(PACKAGE_MAGIC + bytes(64))
            struct.pack_into("<I", raw, PACKAGE_SEED_OFFSET, 151643)
            path.write_bytes(raw)
            before = path.read_bytes()
            record = patch_package_seed(path, 777)
            after = path.read_bytes()
            self.assertEqual(record["accepted_schedule_seed_token_id"], 151643)
            self.assertEqual(struct.unpack_from("<I", after, PACKAGE_SEED_OFFSET)[0], 777)
            self.assertEqual(before[:PACKAGE_SEED_OFFSET], after[:PACKAGE_SEED_OFFSET])
            self.assertEqual(before[PACKAGE_SEED_OFFSET + 4 :], after[PACKAGE_SEED_OFFSET + 4 :])


if __name__ == "__main__":
    unittest.main()
