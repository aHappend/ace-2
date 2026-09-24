import importlib.util
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/run_w4a8_c02_qk_score_rank_margin_repair_design.py"
)
SPEC = importlib.util.spec_from_file_location("qk_rank_margin_runner", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def unpack_scale32(record: int) -> tuple[int, int]:
    significand = record & 0xFFFF
    exponent_u8 = (record >> 16) & 0xFF
    return significand, exponent_u8 - 256 if exponent_u8 >= 128 else exponent_u8


def round_fraction_even(value: Fraction) -> int:
    magnitude = abs(value.numerator)
    quotient, remainder = divmod(magnitude, value.denominator)
    increment = 2 * remainder > value.denominator or (
        2 * remainder == value.denominator and quotient & 1
    )
    rounded = quotient + int(increment)
    return -rounded if value < 0 else rounded


def arbitrary_precision_group8_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    query_base_records: torch.Tensor,
    key_base_records: torch.Tensor,
    query_gain_records: torch.Tensor,
    key_gain_records: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    sequence = query.shape[2]
    result = torch.full((1, runner.QUERY_HEADS, sequence, sequence), -32768, dtype=torch.int16)
    clamp_events = 0
    for head in range(runner.QUERY_HEADS):
        for query_position in range(sequence):
            exact_scores = []
            for key_position in range(query_position + 1):
                total = Fraction(0)
                for group in range(runner.GROUP_COUNT):
                    start = group * runner.GROUP_SIZE
                    stop = start + runner.GROUP_SIZE
                    dot = sum(
                        int(query[0, head, query_position, lane])
                        * int(key[0, head, key_position, lane])
                        for lane in range(start, stop)
                    )
                    records = (
                        int(query_base_records[0, head, query_position]),
                        int(key_base_records[0, head, key_position]),
                        int(query_gain_records[head, group]),
                        int(key_gain_records[head, group]),
                    )
                    components = [unpack_scale32(record) for record in records]
                    numerator = dot
                    exponent = -63
                    for significand, scale_exponent in components:
                        numerator *= significand
                        exponent += scale_exponent
                    total += Fraction(numerator << exponent, 1) if exponent >= 0 else Fraction(numerator, 1 << -exponent)
                exact_scores.append(total)
            maximum = max(exact_scores)
            for key_position, score in enumerate(exact_scores):
                rounded = round_fraction_even((score - maximum) * (1 << runner.SCORE_FRAC))
                clamp_events += int(rounded < -32768 or rounded > 0)
                result[0, head, query_position, key_position] = max(-32768, min(0, rounded))
    return result, clamp_events


def wide_from_python(values: list[int]) -> tuple[torch.Tensor, ...]:
    rows = [[] for _ in range(runner.WIDE_LIMBS)]
    for value in values:
        remaining = value
        for limb in range(runner.WIDE_LIMBS - 1):
            digit = remaining % runner.WIDE_LIMB_BASE
            rows[limb].append(digit)
            remaining = (remaining - digit) // runner.WIDE_LIMB_BASE
        rows[-1].append(remaining)
    return tuple(torch.tensor(row, dtype=torch.int64).reshape(1, 1, 1, -1) for row in rows)


class QkRankMarginRunnerTest(unittest.TestCase):
    def test_metric_accumulator_recomputes_tracking_metrics(self) -> None:
        accumulator = runner.MetricAccumulator()
        reference = torch.tensor([[[[1.0, 2.0]]]])
        candidate = torch.tensor([[[[1.0, 2.0]]]])
        valid = torch.ones_like(reference, dtype=torch.bool)
        self.assertEqual(accumulator.update(reference, candidate, valid), 0)
        result = accumulator.result()
        self.assertEqual(result["metrics"]["severity"], "TRACKING")
        self.assertEqual(
            result["metrics"],
            runner.MetricAccumulator.metrics_from_raw(result["raw_accumulator"]),
        )

    def test_scale32_pair_conversion_matches_exact_power_of_two(self) -> None:
        record = runner.packed_scale32(0.25)
        dots = torch.tensor([[[[8]]]], dtype=torch.int64)
        records = torch.tensor([[[record]]], dtype=torch.int64)
        converted, overflow, _width = runner.scale32_pair_to_q20_44(
            dots, records, records
        )
        self.assertEqual(overflow, 0)
        self.assertEqual(int(converted), int(round((8 * 0.25 * 0.25 / 8) * (1 << 44))))

    def test_score_realization_counts_clamps(self) -> None:
        valid = torch.ones((1, 1, 1, 2), dtype=torch.bool)
        centered = torch.tensor([[[[0, -100 * (1 << 44)]]]], dtype=torch.int64)
        realized, clamps = runner.score32_from_q20_44(centered, valid)
        self.assertEqual(clamps, 1)
        self.assertEqual(realized.tolist(), [[[[0, -32768]]]])

    def test_zero_energy_nnls_uses_canonical_minimum_scale32(self) -> None:
        record = runner.packed_scale32(0.0)
        self.assertEqual(record, runner.fixed.SCALE32_ALL_ZERO_RECORD)

    def test_wide_q6_9_rounding_matches_arbitrary_precision_ties_to_even(self) -> None:
        shift = 70
        magnitudes = [
            (2 << shift) + (1 << (shift - 1)),
            (3 << shift) + (1 << (shift - 1)),
            (4 << shift) + (1 << (shift - 1)),
            (5 << shift) + (1 << (shift - 1)),
            (32768 << shift) + (1 << (shift - 1)),
            32769 << shift,
        ]
        wide = wide_from_python([-value for value in magnitudes])
        valid = torch.ones((1, 1, 1, len(magnitudes)), dtype=torch.bool)
        observed, clamps = runner.wide_nonpositive_to_q6_9(
            wide,
            torch.tensor([[[[shift]]]], dtype=torch.int64),
            valid,
        )
        self.assertEqual(observed.tolist(), [[[[-2, -4, -4, -6, -32768, -32768]]]])
        self.assertEqual(clamps, 1)

    def test_group8_scores_match_independent_arbitrary_precision_reference(self) -> None:
        sequence = 3
        values = torch.arange(runner.QUERY_HEADS * sequence * runner.HEAD_DIM, dtype=torch.int64)
        query = ((values * 37 + 11) % 255 - 127).to(torch.int8).reshape(1, runner.QUERY_HEADS, sequence, runner.HEAD_DIM)
        key = ((values * 53 + 19) % 255 - 127).to(torch.int8).reshape(1, runner.QUERY_HEADS, sequence, runner.HEAD_DIM)

        exponents = (-24, 4, -7)
        query_base = torch.empty((1, runner.QUERY_HEADS, sequence), dtype=torch.int64)
        key_base = torch.empty_like(query_base)
        for head in range(runner.QUERY_HEADS):
            for position in range(sequence):
                query_base[0, head, position] = runner.fixed.pack_scale32(
                    0x8000 + ((head * 97 + position * 31) & 0x7FFF),
                    exponents[(head + position) % len(exponents)],
                )
                key_base[0, head, position] = runner.fixed.pack_scale32(
                    0x8000 + ((head * 43 + position * 71 + 1) & 0x7FFF),
                    exponents[(2 * head + position + 1) % len(exponents)],
                )

        query_gain = torch.empty((runner.QUERY_HEADS, runner.GROUP_COUNT), dtype=torch.int64)
        key_gain = torch.empty_like(query_gain)
        gain_exponents = (-24, 4, -3, 1, -12, 0, -6, 3)
        for head in range(runner.QUERY_HEADS):
            for group, exponent in enumerate(gain_exponents):
                query_gain[head, group] = runner.fixed.pack_scale32(
                    0x8000 + ((head * 29 + group * 101) & 0x7FFF),
                    exponent,
                )
                key_gain[head, group] = runner.fixed.pack_scale32(
                    0x8000 + ((head * 83 + group * 47 + 3) & 0x7FFF),
                    gain_exponents[-1 - group],
                )

        valid, _ = runner.causal_mask(sequence, device=query.device)
        expected, expected_clamps = arbitrary_precision_group8_reference(
            query, key, query_base, key_base, query_gain, key_gain
        )
        safety = runner.SafetyCounters()
        observed = runner.exact_group8_scores(
            query,
            key,
            query_base,
            key_base,
            query_gain,
            key_gain,
            valid,
            safety,
        )
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(safety.clamp_events, expected_clamps)
        self.assertEqual(safety.overflow_events, 0)
        self.assertGreater(safety.maximum_signed_arithmetic_width, 64)

    def test_validation_checks_contain_only_frozen_predicates(self) -> None:
        task = runner.load_json(runner.TASK_PATH)
        metrics = {"metrics": {"severity": "TRACKING"}}
        safety = {
            "clamp_events": 0,
            "maximum_signed_arithmetic_width": 1,
            "non_finite_values": 0,
            "overflow_events": 0,
            "reserved_negative_128_payloads": 0,
        }
        validation = {
            "corrected_centered_score": metrics,
            "corrected_probability": metrics,
            "fixed_softmax_realization": metrics,
            "integer_safety": safety,
            "top_key_agreement": {
                "absolute_improvement": 0.2,
                "corrected_top_key_index_equal_fraction": 0.9,
                "head_regressions": [],
            },
        }
        diagnostic = {
            "corrected_centered_score": metrics,
            "corrected_probability": metrics,
            "fixed_softmax_realization": metrics,
            "integer_safety": safety,
            "top_key_agreement": {"corrected_top_key_index_equal_fraction": 1.0},
        }
        checks = runner.checks_for_result(task, validation, diagnostic)
        self.assertNotIn("validation_fixed_softmax_tracking", checks)
        self.assertTrue(all(checks.values()))

    def test_selection_is_fail_closed_when_neither_alternative_passes(self) -> None:
        task = runner.load_json(runner.TASK_PATH)

        def model(alias: str) -> dict:
            return {
                "alternatives": [
                    {
                        "alternative_id": alternative,
                        "fit": {"parameter_bytes_per_model": 4},
                        "fixed_diagnostic": {"integer_safety": {"maximum_signed_arithmetic_width": 8}},
                        "pass": False,
                        "validation": {
                            "corrected_probability": {"metrics": {"relative_rmse": 1.0}},
                            "integer_safety": {"maximum_signed_arithmetic_width": 8},
                        },
                    }
                    for alternative in runner.ALTERNATIVE_ORDER
                ],
                "model_alias": alias,
            }

        observed = runner.select_result(
            task, {alias: model(alias) for alias in runner.MODEL_ORDER}
        )
        self.assertEqual(observed["verdict"], "PRE_CANDIDATE_NO_GO")
        self.assertIsNone(observed["selected_alternative_id"])


if __name__ == "__main__":
    unittest.main()
