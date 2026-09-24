#!/usr/bin/env python3
"""Exact all-layer Scale32 hook for down-projection residual fusion.

This architecture-stage reference deliberately uses Python integers for the
aligned numerator and denominator.  It therefore implements the frozen
signed-96/unsigned-64 contract without silently narrowing the intermediate
values to a host tensor dtype.  It does not load model or dataset metadata and
does not authorize or perform a focused quality run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import Tensor

from ace2_quality_contracts import (
    pack_scale32,
    round_divide_even_signed,
    unpack_scale32,
)


CONTRACT_ID = "shared_down_projection_residual_fusion_v1"
LAYER_COUNT = 24
HIDDEN_WIDTH = 896
SIGNED32_MIN = -(1 << 31)
SIGNED32_MAX = (1 << 31) - 1
SIGNED96_MIN = -(1 << 95)
SIGNED96_MAX = (1 << 95) - 1
UNSIGNED64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class FusionMetadata:
    """Immutable Scale32 records in layer-major, output-lane-major order."""

    accumulator_scale32: tuple[tuple[int, ...], ...]
    residual_scale32: tuple[int, ...]
    destination_scale32: tuple[int, ...]

    @classmethod
    def from_sequences(
        cls,
        accumulator_scale32: Sequence[Sequence[int]],
        residual_scale32: Sequence[int],
        destination_scale32: Sequence[int],
    ) -> "FusionMetadata":
        metadata = cls(
            accumulator_scale32=tuple(
                tuple(int(record) for record in layer_records)
                for layer_records in accumulator_scale32
            ),
            residual_scale32=tuple(int(record) for record in residual_scale32),
            destination_scale32=tuple(
                int(record) for record in destination_scale32
            ),
        )
        metadata.validate()
        return metadata

    def validate(self) -> None:
        if len(self.accumulator_scale32) != LAYER_COUNT:
            raise ValueError("fusion metadata must contain exactly 24 layers")
        if len(self.residual_scale32) != LAYER_COUNT:
            raise ValueError("fusion metadata must contain 24 residual Scale32 records")
        if len(self.destination_scale32) != LAYER_COUNT:
            raise ValueError(
                "fusion metadata must contain 24 destination Scale32 records"
            )
        for layer_index, records in enumerate(self.accumulator_scale32):
            if len(records) != HIDDEN_WIDTH:
                raise ValueError(
                    f"layer {layer_index} must contain exactly 896 accumulator records"
                )
            for record in records:
                unpack_scale32(record)
            unpack_scale32(self.residual_scale32[layer_index])
            unpack_scale32(self.destination_scale32[layer_index])


@dataclass(frozen=True)
class LayerConstructTrace:
    layer_index: int
    rows: int
    lanes: int
    positive_saturations: int
    negative_saturations: int
    maximum_absolute_numerator: int
    maximum_denominator: int
    common_exponent_minimum: int
    common_exponent_maximum: int


def fuse_scale32_lane_exact(
    accumulator: int,
    residual: int,
    accumulator_scale32: int,
    residual_scale32: int,
    destination_scale32: int,
) -> tuple[int, int, int, int]:
    """Return (signed-int8 output, numerator, denominator, common exponent)."""
    accumulator = int(accumulator)
    residual = int(residual)
    if not SIGNED32_MIN <= accumulator <= SIGNED32_MAX:
        raise OverflowError("down-projection accumulator is outside signed-32")
    if not -128 <= residual <= 127:
        raise ValueError("post-attention residual is outside signed-int8")

    accumulator_sig, accumulator_exp = unpack_scale32(accumulator_scale32)
    residual_sig, residual_exp = unpack_scale32(residual_scale32)
    destination_sig, destination_exp = unpack_scale32(destination_scale32)
    common_exp = min(accumulator_exp, residual_exp, destination_exp)

    accumulator_term = (
        accumulator
        * accumulator_sig
        * (1 << (accumulator_exp - common_exp))
    )
    residual_term = residual * residual_sig * (1 << (residual_exp - common_exp))
    numerator = accumulator_term + residual_term
    denominator = destination_sig * (1 << (destination_exp - common_exp))

    if not SIGNED96_MIN <= numerator <= SIGNED96_MAX:
        raise OverflowError("fused numerator exceeds the frozen signed-96 workspace")
    if not 1 <= denominator <= UNSIGNED64_MAX:
        raise OverflowError(
            "fusion denominator exceeds the frozen unsigned-64 workspace"
        )

    rounded = round_divide_even_signed(numerator, denominator)
    output = max(-128, min(127, rounded))
    return output, numerator, denominator, common_exp


def fuse_scale32_layer_exact(
    accumulator: Tensor,
    residual: Tensor,
    accumulator_scale32: Sequence[int],
    residual_scale32: int,
    destination_scale32: int,
    *,
    layer_index: int,
) -> tuple[Tensor, LayerConstructTrace]:
    """Apply the exact frozen equation to every row and all 896 output lanes."""
    if accumulator.dtype not in (torch.int32, torch.int64):
        raise TypeError("down-projection accumulator must be signed int32 or int64")
    if residual.dtype != torch.int8:
        raise TypeError("post-attention residual must be signed int8")
    if accumulator.shape != residual.shape or accumulator.ndim < 1:
        raise ValueError("accumulator and residual tensor shapes must match")
    if accumulator.numel() == 0:
        raise ValueError("fusion tensors must contain at least one row")
    if accumulator.shape[-1] != HIDDEN_WIDTH:
        raise ValueError("fusion tensors must have 896 output lanes")
    if len(accumulator_scale32) != HIDDEN_WIDTH:
        raise ValueError("accumulator Scale32 records must cover all 896 lanes")
    if not 0 <= layer_index < LAYER_COUNT:
        raise ValueError("layer index must be in 0..23")

    accumulator_rows = accumulator.detach().cpu().reshape(-1, HIDDEN_WIDTH).tolist()
    residual_rows = residual.detach().cpu().reshape(-1, HIDDEN_WIDTH).tolist()
    output_rows: list[list[int]] = []
    positive_saturations = 0
    negative_saturations = 0
    maximum_absolute_numerator = 0
    maximum_denominator = 0
    common_exponent_minimum = 127
    common_exponent_maximum = -128

    for accumulator_row, residual_row in zip(
        accumulator_rows, residual_rows, strict=True
    ):
        output_row: list[int] = []
        for lane, (accumulator_value, residual_value) in enumerate(
            zip(accumulator_row, residual_row, strict=True)
        ):
            output, numerator, denominator, common_exp = fuse_scale32_lane_exact(
                accumulator_value,
                residual_value,
                int(accumulator_scale32[lane]),
                int(residual_scale32),
                int(destination_scale32),
            )
            rounded = round_divide_even_signed(numerator, denominator)
            positive_saturations += int(rounded > 127)
            negative_saturations += int(rounded < -128)
            maximum_absolute_numerator = max(
                maximum_absolute_numerator, abs(numerator)
            )
            maximum_denominator = max(maximum_denominator, denominator)
            common_exponent_minimum = min(common_exponent_minimum, common_exp)
            common_exponent_maximum = max(common_exponent_maximum, common_exp)
            output_row.append(output)
        output_rows.append(output_row)

    output_tensor = torch.tensor(output_rows, dtype=torch.int8).reshape(residual.shape)
    output_tensor = output_tensor.to(device=residual.device)
    trace = LayerConstructTrace(
        layer_index=layer_index,
        rows=len(output_rows),
        lanes=accumulator.numel(),
        positive_saturations=positive_saturations,
        negative_saturations=negative_saturations,
        maximum_absolute_numerator=maximum_absolute_numerator,
        maximum_denominator=maximum_denominator,
        common_exponent_minimum=common_exponent_minimum,
        common_exponent_maximum=common_exponent_maximum,
    )
    return output_tensor, trace


class ExactScale32AllLayerHook:
    """Stateful hook that enforces one ordered invocation for every model layer."""

    def __init__(self, metadata: FusionMetadata) -> None:
        metadata.validate()
        self.metadata = metadata
        self._next_layer = 0
        self._traces: list[LayerConstructTrace] = []
        self._numeric_overflow = False

    def begin_pass(self) -> None:
        self._next_layer = 0
        self._traces.clear()
        self._numeric_overflow = False

    def apply_layer(
        self,
        layer_index: int,
        accumulator: Tensor,
        residual: Tensor,
    ) -> Tensor:
        if layer_index != self._next_layer:
            raise RuntimeError(
                f"fusion hook expected layer {self._next_layer}, got {layer_index}"
            )
        try:
            output, trace = fuse_scale32_layer_exact(
                accumulator,
                residual,
                self.metadata.accumulator_scale32[layer_index],
                self.metadata.residual_scale32[layer_index],
                self.metadata.destination_scale32[layer_index],
                layer_index=layer_index,
            )
        except OverflowError:
            self._numeric_overflow = True
            raise
        self._traces.append(trace)
        self._next_layer += 1
        return output

    @property
    def sticky_numeric_overflow(self) -> bool:
        return self._numeric_overflow

    def finish_pass(self) -> dict[str, object]:
        if self._next_layer != LAYER_COUNT:
            raise RuntimeError(
                f"fusion hook executed {self._next_layer} layers, expected 24"
            )
        return {
            "contract_id": CONTRACT_ID,
            "layers_executed": self._next_layer,
            "positive_saturations": sum(
                trace.positive_saturations for trace in self._traces
            ),
            "negative_saturations": sum(
                trace.negative_saturations for trace in self._traces
            ),
            "sticky_numeric_overflow": self._numeric_overflow,
            "layers": [asdict(trace) for trace in self._traces],
        }


def self_test() -> None:
    unit = pack_scale32(0x8000, 0)
    half = pack_scale32(0x8000, -1)
    assert fuse_scale32_lane_exact(1, 2, unit, unit, unit)[0] == 3
    assert fuse_scale32_lane_exact(1, 0, half, unit, unit)[0] == 0
    assert fuse_scale32_lane_exact(3, 0, half, unit, unit)[0] == 2
    assert fuse_scale32_lane_exact(-3, 0, half, unit, unit)[0] == -2
    assert fuse_scale32_lane_exact(127, 1, unit, unit, unit)[0] == 127
    assert fuse_scale32_lane_exact(-128, -1, unit, unit, unit)[0] == -128

    metadata = FusionMetadata.from_sequences(
        [[unit] * HIDDEN_WIDTH for _ in range(LAYER_COUNT)],
        [unit] * LAYER_COUNT,
        [unit] * LAYER_COUNT,
    )
    hook = ExactScale32AllLayerHook(metadata)
    hook.begin_pass()
    accumulator = torch.zeros((1, HIDDEN_WIDTH), dtype=torch.int32)
    residual = torch.zeros((1, HIDDEN_WIDTH), dtype=torch.int8)
    for layer_index in range(LAYER_COUNT):
        output = hook.apply_layer(layer_index, accumulator, residual)
        if not torch.equal(output, residual):
            raise AssertionError("all-layer zero-vector fusion mismatch")
    summary = hook.finish_pass()
    traces = summary["layers"]
    if not isinstance(traces, list):
        raise AssertionError("all-layer construct trace is not a list")
    if [trace["layer_index"] for trace in traces] != list(range(LAYER_COUNT)):
        raise AssertionError("all-layer construct trace order mismatch")
    if summary["sticky_numeric_overflow"]:
        raise AssertionError("zero-vector fusion unexpectedly overflowed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        raise SystemExit("only --self-test is available; dataset flows are not authorized")
    self_test()
    print("ACE2_EXACT_SCALE32_FUSION_HOOK_SELF_TEST PASS layers=24 lanes=896")


if __name__ == "__main__":
    main()
