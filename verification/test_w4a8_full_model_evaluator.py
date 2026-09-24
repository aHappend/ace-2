#!/usr/bin/env python3
"""Regression coverage for frozen W4A8 generation validators and quality gates."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import w4a8_full_model_evaluator as evaluator  # noqa: E402


class FakeTokenizer:
    all_special_ids = [99]

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join({1: "alpha", 2: " beta", 3: " gamma"}.get(item, "") for item in token_ids)


class FrozenEvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = evaluator.require_canonical_json(
            ROOT / "design/QWEN25_05B_W4A8_FULL_MODEL_SOFTWARE_CONTRACT_V1.json"
        )
        cls.config = evaluator.load_json(
            ROOT / "benchmark/quality/QUALITY_CONFIG.json"
        )

    def test_contract_support_is_complete(self) -> None:
        evaluator.validate_contract_support(self.contract, self.config)
        validators = {
            prompt["validator"]
            for prompt in self.contract["evaluation_contract"][
                "canonical_generation_prompts"
            ]
        }
        self.assertEqual(validators, evaluator.SUPPORTED_VALIDATORS)

    def test_c01_scale32_regression_changes_the_reported_tie_code(self) -> None:
        weight = torch.tensor(
            [[0.29296875, -0.1318359375]],
            dtype=torch.bfloat16,
        )
        qweight, scale, records, _error = evaluator._clip_grid_row_candidate(
            weight,
            weight.abs().amax(dim=1),
            900,
            1000,
        )
        raw_scale = float(weight.abs().amax()) * 900.0 / 1000.0 / 7.0
        raw_code = int(torch.round(weight[0, 1].to(torch.float64) / raw_scale))
        self.assertEqual(raw_code, -4)
        self.assertEqual(int(qweight[0, 1]), -3)
        self.assertEqual(float(scale[0]), 0.03766822814941406)
        self.assertEqual(float(scale[0]), evaluator.scale32_value(int(records[0])))

    def test_c01_exact_signed_ties_to_even(self) -> None:
        record, realized = evaluator.realize_positive_scale32(0.25)
        self.assertEqual(realized, 0.25)
        weight = torch.tensor(
            [[0.625, -0.625, 0.875, -0.875]],
            dtype=torch.bfloat16,
        )
        qweight = evaluator._quantize_bf16_with_scale32(
            weight,
            torch.tensor([record], dtype=torch.int64),
        )
        self.assertEqual(qweight.tolist(), [[2, -2, 4, -4]])

    def test_c01_selector_is_deterministic_and_returns_realized_scale32(self) -> None:
        candidate = evaluator.candidate_spec(self.contract, "c01-mse-clip-grid")
        weight = torch.tensor(
            [
                [0.0] * 21,
                [10.0] + [1.0] * 20,
            ],
            dtype=torch.bfloat16,
        )
        first = evaluator.mse_clip_grid_quantization_scale32(weight, candidate)
        second = evaluator.mse_clip_grid_quantization_scale32(weight, candidate)
        for left, right in zip(first, second, strict=True):
            self.assertTrue(torch.equal(left, right))
        qweight, scales, numerators, records = first
        self.assertEqual(qweight[0].tolist(), [0] * 21)
        self.assertEqual(int(numerators[0]), 1000)
        self.assertEqual(
            scales.tolist(),
            [evaluator.scale32_value(int(record)) for record in records],
        )

    def test_c01_linear_realizes_every_consumed_projection_scale(self) -> None:
        candidate = evaluator.candidate_spec(self.contract, "c01-mse-clip-grid")
        source = nn.Linear(3, 2, bias=True, dtype=torch.bfloat16)
        with torch.no_grad():
            source.weight.copy_(
                torch.tensor(
                    [[0.29296875, -0.1318359375, 0.0], [0.625, -0.625, 0.25]],
                    dtype=torch.bfloat16,
                )
            )
            source.bias.copy_(torch.tensor([0.125, -0.25], dtype=torch.bfloat16))
        module = evaluator.MSEClipGridW4A8Linear(
            source,
            evaluator.fixed.CalibrationRange(input_absmax=1.0, output_absmax=2.0),
            candidate,
        )
        for value in (
            module.input_scale,
            module.hardware_input_scale,
            module.output_scale,
            *module.weight_scale.tolist(),
        ):
            record = evaluator.ceil_scale32_from_float(float(value))
            self.assertEqual(float(value), evaluator.scale32_value(record))
        module.bind_hardware_input_scale(0.12345)
        hardware_record = evaluator.ceil_scale32_from_float(module.hardware_input_scale)
        self.assertEqual(
            module.hardware_input_scale,
            evaluator.scale32_value(hardware_record),
        )

    def test_c01_operator_calibration_is_realized_without_mutating_observations(self) -> None:
        model = SimpleNamespace(
            model=SimpleNamespace(
                norm=nn.LayerNorm(2),
                layers=[
                    SimpleNamespace(
                        input_layernorm=nn.LayerNorm(2),
                        post_attention_layernorm=nn.LayerNorm(2),
                    )
                ],
            )
        )
        observations = {
            "model.norm.output": evaluator.fixed.ObservedRange(absmax=1.3),
            "model.layers.0.input_layernorm.output": evaluator.fixed.ObservedRange(absmax=2.7),
            "model.layers.0.post_attention_layernorm.output": evaluator.fixed.ObservedRange(absmax=3.1),
            "model.layers.0.input_layernorm.input": evaluator.fixed.ObservedRange(absmax=4.2),
        }
        runtime = evaluator._scale32_runtime_operator_ranges(model, observations)
        self.assertEqual(observations["model.norm.output"].absmax, 1.3)
        for name, observed in runtime.items():
            source = {
                "model.norm.output": model.model.norm,
                "model.layers.0.input_layernorm.output": model.model.layers[0].input_layernorm,
                "model.layers.0.post_attention_layernorm.output": model.model.layers[0].post_attention_layernorm,
            }.get(name)
            scale = (
                evaluator.fixed.rmsnorm_output_scale(source, observed.absmax)
                if source is not None
                else evaluator.fixed.positive_scale(observed.absmax)
            )
            self.assertEqual(
                scale,
                evaluator.scale32_value(evaluator.ceil_scale32_from_float(scale)),
            )

    def test_every_prompt_validator_passes_and_fails(self) -> None:
        cases = {
            "strip_ascii_whitespace_then_exactly_19": ("\t19\n", "19.0"),
            "readability_only": ("A sentence.", "1234"),
            "parse_json_then_exact_object_status_ok_count_3": (
                '{"status":"ok","count":3}',
                '{"status":"ok","count":4}',
            ),
            "exactly_three_nonempty_lines_each_containing_a_unicode_letter": (
                "red\ngreen\nblue",
                "red\n2\nblue",
            ),
            "unicode_casefold_strip_terminal_punctuation_in_buenos_dias_set": (
                "BUENOS DÍAS!",
                "buenas noches",
            ),
            "exactly_two_list_items_with_unicode_letters": (
                "- First\n* Second",
                "- First\nSecond",
            ),
        }
        self.assertEqual(set(cases), evaluator.SUPPORTED_VALIDATORS)
        for name, (passing, failing) in cases.items():
            with self.subTest(name=name, result="pass"):
                self.assertTrue(evaluator.prompt_validator(name, passing)["passed"])
            with self.subTest(name=name, result="fail"):
                self.assertFalse(evaluator.prompt_validator(name, failing)["passed"])

    def test_global_generation_gates_fail_closed(self) -> None:
        gate = self.contract["evaluation_contract"]["generation_gate"]
        generation = self.contract["shared_w4a8_contract"]["generation"]
        prompt = {"short_answer_exception": False}
        tokenizer = FakeTokenizer()
        passing = evaluator.global_generation_checks(
            tokenizer, [1, 2], "alpha beta", prompt, generation, gate
        )
        self.assertTrue(passing["passed"])

        punctuation = evaluator.global_generation_checks(
            tokenizer, [1, 2], "''''", prompt, generation, gate
        )
        self.assertFalse(punctuation["checks"]["punctuation_only_rejected"])
        invalid = evaluator.global_generation_checks(
            tokenizer, [1, 2], "alpha\x01", prompt, generation, gate
        )
        self.assertFalse(
            invalid["checks"]["replacement_surrogate_or_control_character_rejected"]
        )
        repeated = evaluator.global_generation_checks(
            tokenizer, [1, 1, 1, 1, 1], "alpha alpha", prompt, generation, gate
        )
        self.assertFalse(repeated["checks"]["identical_token_run"])
        low_unique = evaluator.global_generation_checks(
            tokenizer, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "alpha", prompt, generation, gate
        )
        self.assertFalse(low_unique["checks"]["unique_token_fraction"])
        invisible = evaluator.global_generation_checks(
            tokenizer, [99], "", prompt, generation, gate
        )
        self.assertFalse(invisible["checks"]["minimum_visible_tokens"])

    def test_every_quality_gate_passes_at_bound_and_fails_above_bound(self) -> None:
        thresholds = evaluator.quality_thresholds_from_contract(self.contract)
        baseline = {
            "lm_eval": {
                "average_normalized_accuracy": 0.8,
                "tasks": {"a": 0.8, "b": 0.8},
            },
            "perplexity": {
                "c4_en_512": {"perplexity": 10.0},
                "wikitext2": {"perplexity": 20.0},
            },
        }
        passing = {
            "lm_eval": {
                "average_normalized_accuracy": 0.78,
                "tasks": {"a": 0.73, "b": 0.8},
            },
            "perplexity": {
                "c4_en_512": {"perplexity": 11.0},
                "wikitext2": {"perplexity": 22.0},
            },
        }
        self.assertTrue(
            evaluator.evaluate_quality(baseline, passing, thresholds, ["a", "b"])[
                "passed"
            ]
        )
        failures = {
            "c4": ("perplexity", "c4_en_512", "perplexity", 11.01),
            "wikitext2": ("perplexity", "wikitext2", "perplexity", 22.01),
            "average": ("lm_eval", "average_normalized_accuracy", None, 0.76),
            "individual": ("lm_eval", "tasks", "a", 0.70),
        }
        for name, (group, field, nested, value) in failures.items():
            candidate = copy.deepcopy(passing)
            if nested is None:
                candidate[group][field] = value
            else:
                candidate[group][field][nested] = value
            with self.subTest(gate=name):
                self.assertFalse(
                    evaluator.evaluate_quality(
                        baseline, candidate, thresholds, ["a", "b"]
                    )["passed"]
                )


if __name__ == "__main__":
    unittest.main()
