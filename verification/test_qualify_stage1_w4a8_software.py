from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch
from torch import nn

from tools import ace2_full_model_fixed_point as fixed
from tools import qualify_stage1_w4a8_software as quality


def projection(in_features: int, out_features: int) -> quality.GroupedW4A8Linear:
    module = quality.GroupedW4A8Linear.__new__(quality.GroupedW4A8Linear)
    nn.Module.__init__(module)
    module.in_features = in_features
    module.out_features = out_features
    module.groups = in_features // quality.GROUP_SIZE
    module.register_buffer(
        "qweight",
        torch.zeros((out_features, module.groups, quality.GROUP_SIZE), dtype=torch.int8),
    )
    record = quality.ceil_scale32_from_float(1.0)
    module.register_buffer(
        "weight_scale32_records",
        torch.full((out_features, module.groups), record, dtype=torch.int64),
    )
    module.register_buffer(
        "weight_scale",
        torch.ones((out_features, module.groups), dtype=torch.float64),
    )
    module.register_buffer(
        "output_scale_per_channel",
        torch.ones(out_features, dtype=torch.float64),
    )
    module.register_buffer("bias_output", None)
    module.register_buffer(
        "hardware_input_scale32_records",
        torch.full(
            (module.groups,),
            quality.ceil_scale32_from_float(1.0),
            dtype=torch.int64,
        ),
    )
    module.saturation_events = 0
    module.output_elements = 0
    module.hardware_input_calls = 0
    module.exact_float_carrier_conversions = 0
    module.bound_scale32_associations = 0
    module.dynamic_input_calls = 0
    return module


def round_shift_even_scalar(value: int, shift: int) -> int:
    magnitude = abs(value)
    quotient = magnitude >> shift
    if shift:
        remainder = magnitude & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if remainder > half or (remainder == half and quotient & 1):
            quotient += 1
    return -quotient if value < 0 else quotient


def prop03_scalar(products: list[int], shifts: list[int]) -> int:
    common_shift = max(shifts)
    numerator = sum(
        product << (common_shift - shift)
        for product, shift in zip(products, shifts, strict=True)
    )
    return round_shift_even_scalar(numerator, common_shift)


def unit_scale32_records(qinput: torch.Tensor, groups: int) -> torch.Tensor:
    return torch.full(
        (*qinput.shape[:-1], groups),
        quality.ceil_scale32_from_float(1.0),
        dtype=torch.int64,
    )


def accumulate_with_requantization(
    module: quality.GroupedW4A8Linear,
    qinput: torch.Tensor,
    multipliers: list[list[int]],
    shifts: list[list[int]],
) -> torch.Tensor:
    rows = qinput.shape[0] * qinput.shape[1]
    multiplier = torch.tensor(multipliers, dtype=torch.int64).expand(rows, -1, -1)
    right_shift = torch.tensor(shifts, dtype=torch.int64).expand(rows, -1, -1)
    with mock.patch.object(
        quality.fixed,
        "derive_multiplier",
        return_value=(multiplier, right_shift),
    ):
        return module._accumulate(
            qinput,
            unit_scale32_records(qinput, module.groups),
        )


class Stage1W4A8SoftwareQualityTest(unittest.TestCase):
    def test_all_qwen_projection_families_use_canonical_axes(self) -> None:
        geometries = {
            "q_proj": (896, 896),
            "k_proj": (896, 128),
            "v_proj": (896, 128),
            "o_proj": (896, 896),
            "gate_proj": (896, 4864),
            "up_proj": (896, 4864),
            "down_proj": (4864, 896),
        }
        for name, (in_features, out_features) in geometries.items():
            with self.subTest(name=name):
                module = projection(in_features, out_features)
                for sequence in (1, 3):
                    qinput = torch.zeros((2, sequence, in_features), dtype=torch.int8)
                    records = torch.full(
                        (2, sequence, in_features // quality.GROUP_SIZE),
                        quality.ceil_scale32_from_float(1.0),
                        dtype=torch.int64,
                    )
                    self.assertEqual(
                        module._validate_projection_layout(qinput, records),
                        (2, sequence),
                    )

    def test_projection_maps_batch_and_sequence_to_rows(self) -> None:
        module = projection(64, 3)
        module.qweight[:, :, 0] = torch.tensor([[1, 2], [3, 4], [-1, -2]])
        qinput = torch.zeros((2, 3, 64), dtype=torch.int8)
        qinput[:, :, 0] = torch.tensor([[1, 2, 3], [4, 5, 6]])
        qinput[:, :, 32] = 2
        records = torch.full(
            (2, 3, 2),
            quality.ceil_scale32_from_float(1.0),
            dtype=torch.int64,
        )
        observed = module._project(qinput, records)
        expected = torch.tensor(
            [
                [[5, 11, -5], [6, 14, -6], [7, 17, -7]],
                [[8, 20, -8], [9, 23, -9], [10, 26, -10]],
            ],
            dtype=torch.int8,
        )
        self.assertTrue(torch.equal(observed, expected))

    def test_hardware_input_accepts_int8_and_exact_float_carrier(self) -> None:
        module = projection(64, 3)
        module.qweight[:, :, 0] = torch.tensor([[1, 2], [3, 4], [-1, -2]])
        qinput = torch.zeros((1, 2, 64), dtype=torch.int8)
        qinput[:, :, 0] = torch.tensor([[1, 2]])
        qinput[:, :, 32] = 2
        native = module.forward_hardware_input(qinput)
        exact_float = module.forward_hardware_input(qinput.to(torch.bfloat16))
        self.assertTrue(torch.equal(native, exact_float))
        self.assertEqual(module.hardware_input_calls, 2)
        self.assertEqual(module.exact_float_carrier_conversions, 1)
        self.assertEqual(module.bound_scale32_associations, 2)

    def test_accumulator_interface_preserves_grouped_scale32_requantization(self) -> None:
        module = projection(64, 3)
        module.qweight[:, :, 0] = torch.tensor([[1, 2], [3, 4], [-1, -2]])
        qinput = torch.zeros((1, 2, 64), dtype=torch.int8)
        qinput[:, :, 0] = torch.tensor([[1, 2]])
        qinput[:, :, 32] = 2

        accumulator = module.accumulator_quantized(qinput)

        self.assertEqual(accumulator.dtype, torch.int64)
        self.assertEqual(accumulator.tolist(), [[5, 11, -5], [6, 14, -6]])
        self.assertEqual(
            module.requantize_accumulator(accumulator, qinput.shape[:-1]).tolist(),
            [[[5, 11, -5], [6, 14, -6]]],
        )
        self.assertEqual(module.hardware_input_calls, 1)
        self.assertEqual(module.bound_scale32_associations, 1)

    def test_accumulator_interface_fails_closed(self) -> None:
        module = projection(64, 3)
        qinput = torch.zeros((1, 1, 64), dtype=torch.int8)
        with self.assertRaisesRegex(RuntimeError, "signed int8"):
            module.accumulator_quantized(qinput.to(torch.float32))
        with self.assertRaisesRegex(RuntimeError, "activation geometry"):
            module.accumulator_quantized(qinput[0])
        records = module.hardware_input_scale32_records
        module.hardware_input_scale32_records = None
        with self.assertRaisesRegex(RuntimeError, "metadata is missing"):
            module.accumulator_quantized(qinput)
        module.hardware_input_scale32_records = records
        with self.assertRaisesRegex(RuntimeError, "signed int64"):
            module.requantize_accumulator(torch.zeros((1, 3), dtype=torch.int32), (1, 1))
        with self.assertRaisesRegex(RuntimeError, "accumulator geometry"):
            module.requantize_accumulator(torch.zeros((2, 3), dtype=torch.int64), (1, 1))

    def test_hardware_input_rejects_malformed_float_carriers(self) -> None:
        module = projection(64, 3)
        valid = torch.zeros((1, 2, 64), dtype=torch.float32)
        malformed = {
            "finite": valid.index_fill(-1, torch.tensor([0]), torch.nan),
            "outside signed int8": valid.index_fill(-1, torch.tensor([0]), 128.0),
            "exact integers": valid.index_fill(-1, torch.tensor([0]), 0.5),
            "activation geometry": valid[0],
            "input width": valid[..., :-1],
            "exact float container": valid.to(torch.int16),
        }
        for message, carrier in malformed.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    module.forward_hardware_input(carrier)

    def test_hardware_input_requires_bound_scale32_metadata(self) -> None:
        module = projection(64, 3)
        carrier = torch.zeros((1, 1, 64), dtype=torch.float32)
        records = module.hardware_input_scale32_records
        module.hardware_input_scale32_records = None
        with self.assertRaisesRegex(RuntimeError, "metadata is missing"):
            module.forward_hardware_input(carrier)
        module.hardware_input_scale32_records = records.to(torch.int32)
        with self.assertRaisesRegex(RuntimeError, "not int64"):
            module.forward_hardware_input(carrier)
        module.hardware_input_scale32_records = records[:-1]
        with self.assertRaisesRegex(RuntimeError, "record geometry"):
            module.forward_hardware_input(carrier)
        module.hardware_input_scale32_records = records
        module.hardware_input_scale32_records[0] = -1
        with self.assertRaisesRegex(ValueError, "Scale32"):
            module.forward_hardware_input(carrier)

    def test_packed_int4_is_low_nibble_first_and_round_trips(self) -> None:
        values = torch.tensor(list(range(-8, 8)) * 4, dtype=torch.int8).reshape(1, 2, 32)
        packed = quality.pack_signed_int4(values)
        self.assertEqual(int(packed[0, 0, 0]), 0x98)
        self.assertTrue(torch.equal(quality.unpack_signed_int4(packed, 1, 2), values))

    def test_projection_rejects_transposed_or_incompatible_metadata(self) -> None:
        module = projection(64, 3)
        qinput = torch.zeros((1, 3, 64), dtype=torch.int8)
        records = torch.full(
            (1, 3, 2),
            quality.ceil_scale32_from_float(1.0),
            dtype=torch.int64,
        )
        with self.assertRaisesRegex(RuntimeError, "Scale32 geometry"):
            module._project(qinput, records.transpose(1, 2))
        with self.assertRaisesRegex(RuntimeError, "not int64"):
            module._project(qinput, records.to(torch.int32))
        invalid_records = records.clone()
        invalid_records[0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, "Scale32"):
            module._project(qinput, invalid_records)
        module.weight_scale = module.weight_scale.transpose(0, 1)
        with self.assertRaisesRegex(RuntimeError, "weight scale geometry"):
            module._project(qinput, records)
        module = projection(64, 3)
        module.weight_scale32_records = module.weight_scale32_records.transpose(0, 1)
        with self.assertRaisesRegex(RuntimeError, "record geometry"):
            module._project(qinput, records)

    def test_packed_int4_rejects_transposed_geometry(self) -> None:
        packed = torch.zeros((3, 2, quality.GROUP_SIZE // 2), dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            quality.unpack_signed_int4(packed.transpose(0, 1), 3, 2)

    def test_projection_saturates_once_after_group_merge(self) -> None:
        module = projection(64, 2)
        module.qweight.fill_(7)
        qinput = torch.stack(
            (
                torch.full((64,), 127, dtype=torch.int8),
                torch.full((64,), -128, dtype=torch.int8),
            )
        )[None, :, :]
        records = torch.full(
            (1, 2, 2),
            quality.ceil_scale32_from_float(1.0),
            dtype=torch.int64,
        )
        self.assertEqual(module._project(qinput, records).tolist(), [[[127, 127], [-128, -128]]])
        self.assertEqual(module.saturation_events, 4)

    def test_common_shift_matches_prop03_scalar_equation(self) -> None:
        cases = (
            ("unequal shifts", [3, 3], [1, 2]),
            ("positive tie", [3, 4], [1, 2]),
            ("negative tie", [-3, -4], [1, 2]),
            ("cancellation", [5, -10], [1, 2]),
        )
        for name, products, shifts in cases:
            with self.subTest(name=name):
                module = projection(64, 1)
                module.qweight[0, :, 0] = 1
                qinput = torch.zeros((1, 1, 64), dtype=torch.int8)
                qinput[0, 0, (0, 32)] = torch.tensor(products, dtype=torch.int8)
                observed = accumulate_with_requantization(
                    module,
                    qinput,
                    [[1, 1]],
                    [shifts],
                )
                self.assertEqual(
                    int(observed[0, 0]),
                    prop03_scalar(products, shifts),
                )

    def test_common_shift_rescues_group_underflow(self) -> None:
        module = projection(96, 1)
        module.qweight[0, :, 0] = 1
        qinput = torch.zeros((1, 1, 96), dtype=torch.int8)
        qinput[0, 0, (0, 32, 64)] = 1
        observed = accumulate_with_requantization(
            module,
            qinput,
            [[1, 1, 1]],
            [[2, 2, 2]],
        )
        self.assertEqual(int(observed[0, 0]), 1)

    def test_common_shift_preserves_multi_output_group_layout(self) -> None:
        module = projection(64, 2)
        module.qweight[:, :, 0] = torch.tensor([[1, 1], [-1, 2]])
        qinput = torch.zeros((1, 1, 64), dtype=torch.int8)
        qinput[0, 0, (0, 32)] = torch.tensor([3, 4], dtype=torch.int8)
        shifts = [[1, 2], [2, 1]]
        products = [[3, 4], [-3, 8]]
        observed = accumulate_with_requantization(
            module,
            qinput,
            [[1, 1], [1, 1]],
            shifts,
        )
        expected = [
            prop03_scalar(output_products, output_shifts)
            for output_products, output_shifts in zip(products, shifts, strict=True)
        ]
        self.assertEqual(observed.tolist(), [expected])

    def test_common_shift_same_shift_exact_groups_match_per_group_rounding(self) -> None:
        module = projection(64, 1)
        module.qweight[0, :, 0] = 1
        qinput = torch.zeros((1, 1, 64), dtype=torch.int8)
        products = [4, -2]
        qinput[0, 0, (0, 32)] = torch.tensor(products, dtype=torch.int8)
        observed = accumulate_with_requantization(
            module,
            qinput,
            [[1, 1]],
            [[1, 1]],
        )
        per_group = sum(round_shift_even_scalar(product, 1) for product in products)
        self.assertEqual(int(observed[0, 0]), per_group)
        self.assertEqual(int(observed[0, 0]), prop03_scalar(products, [1, 1]))

    def test_common_shift_keeps_final_signed_int8_saturation(self) -> None:
        module = projection(32, 1)
        module.qweight[0, 0, 0] = 2
        qinput = torch.zeros((1, 2, 32), dtype=torch.int8)
        qinput[0, :, 0] = torch.tensor([100, -100], dtype=torch.int8)
        with mock.patch.object(
            quality.fixed,
            "derive_multiplier",
            return_value=(
                torch.ones((2, 1, 1), dtype=torch.int64),
                torch.zeros((2, 1, 1), dtype=torch.int64),
            ),
        ):
            observed = module._project(
                qinput,
                unit_scale32_records(qinput, module.groups),
            )
        self.assertEqual(observed.tolist(), [[[127], [-128]]])
        self.assertEqual(module.saturation_events, 2)

    def test_frozen_inputs_reject_prompt_regression_duplication(self) -> None:
        suite = quality.load_json(quality.SUITE_PATH)
        calibration = quality.load_json(quality.CALIBRATION_PATH)
        quality.validate_frozen_inputs(suite, calibration)
        duplicated = copy.deepcopy(suite)
        duplicated["cases"][1]["prompt"] = "What are registers used for?"
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            quality.validate_frozen_inputs(duplicated, calibration)

    def test_layer_activation_error_is_recorded(self) -> None:
        def run(selected: int, hidden: torch.Tensor) -> dict[str, object]:
            return {
                "_layer_final_hidden_values": {"0": hidden},
                "steps": [
                    {
                        "selected_token_id": selected,
                        "top_k": [{"logit": 1.0, "token_id": selected}],
                    }
                ],
            }

        comparison = quality.compare_runs(
            run(1, torch.tensor([1.0, 2.0])),
            run(2, torch.tensor([1.0, 4.0])),
        )
        self.assertEqual(comparison["earliest_layer_with_activation_divergence"], 0)
        self.assertEqual(comparison["layer_activation_error"][0]["max_abs_error"], 2.0)
        self.assertFalse(comparison["steps"][0]["argmax_agreement"])

    def test_dynamic_silu_is_int8_scale32_and_deterministic(self) -> None:
        gate = torch.tensor([[[0, 12, -9, 127] * 8]], dtype=torch.int8)
        up = torch.tensor([[[3, -7, 11, 20] * 8]], dtype=torch.int8)
        first = quality.dynamic_silu_groups(gate, up, 0.125, 0.25)
        second = quality.dynamic_silu_groups(gate, up, 0.125, 0.25)
        self.assertEqual(first[0].dtype, torch.int8)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(first[1].shape, (1, 1, 1))
        for record in first[1].reshape(-1).tolist():
            quality.unpack_scale32(record)

    def test_ties_to_even_and_saturation_semantics(self) -> None:
        values = torch.tensor([-257, -255, -1, 1, 255, 257], dtype=torch.int64)
        shifts = torch.ones_like(values)
        self.assertEqual(
            fixed.round_shift_even(values, shifts).tolist(),
            [-128, -128, 0, 0, 128, 128],
        )
        self.assertEqual(
            fixed.round_shift_even(values, shifts).clamp(-128, 127).to(torch.int8).tolist(),
            [-128, -128, 0, 0, 127, 127],
        )


if __name__ == "__main__":
    unittest.main()
