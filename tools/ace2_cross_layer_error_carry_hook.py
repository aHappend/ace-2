#!/usr/bin/env python3
"""Construct-faithful all-layer and final-output QECR software hook."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from ace2_cross_layer_error_carry_reference import producer_lane, rmsnorm
from ace2_quality_contracts import unpack_scale32


CONTRACT_ID = "cross_layer_quantization_error_carry_final_output_v1"
LAYER_COUNT = 24
HIDDEN_WIDTH = 896


def _sha256_tensor(value: Tensor) -> str:
    payload = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _scale32_float(record: int) -> float:
    significand, exponent = unpack_scale32(record)
    return float(significand) * (2.0 ** (exponent - 15))


@dataclass(frozen=True)
class CrossLayerCarryMetadata:
    accumulator_scale32: tuple[tuple[int, ...], ...]
    residual_scale32: tuple[int, ...]
    destination_scale32: tuple[int, ...]

    @classmethod
    def from_sequences(
        cls,
        accumulator_scale32: Sequence[Sequence[int]],
        residual_scale32: Sequence[int],
        destination_scale32: Sequence[int],
    ) -> "CrossLayerCarryMetadata":
        metadata = cls(
            accumulator_scale32=tuple(
                tuple(int(record) for record in layer)
                for layer in accumulator_scale32
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
            raise ValueError("QECR metadata must contain exactly 24 layers")
        if len(self.residual_scale32) != LAYER_COUNT:
            raise ValueError("QECR residual metadata must contain 24 records")
        if len(self.destination_scale32) != LAYER_COUNT:
            raise ValueError("QECR destination metadata must contain 24 records")
        for layer_index, records in enumerate(self.accumulator_scale32):
            if len(records) != HIDDEN_WIDTH:
                raise ValueError(
                    f"QECR layer {layer_index} must contain 896 accumulator records"
                )
            for record in records:
                unpack_scale32(record)
            unpack_scale32(self.residual_scale32[layer_index])
            unpack_scale32(self.destination_scale32[layer_index])

    def destination_float(self, layer_index: int) -> float:
        return _scale32_float(self.destination_scale32[layer_index])


@dataclass(frozen=True)
class ProducerTrace:
    layer_index: int
    rows: int
    lanes: int
    maximum_absolute_numerator: int
    maximum_denominator: int
    maximum_absolute_carry_q15: int
    common_exponent_minimum: int
    common_exponent_maximum: int


@dataclass(frozen=True)
class ConsumerTrace:
    consumer_layer_id: int
    producer_layer_index: int
    rows: int
    lanes: int
    maximum_absolute_carry_q15: int
    maximum_absolute_reconstructed_q15: int
    rows_with_output_saturation: int


def produce_layer_exact(
    accumulator: Tensor,
    residual: Tensor,
    metadata: CrossLayerCarryMetadata,
    layer_index: int,
) -> tuple[Tensor, Tensor, ProducerTrace]:
    if accumulator.dtype != torch.int32 or residual.dtype != torch.int8:
        raise TypeError("QECR producer requires signed-int32 and signed-int8 inputs")
    if accumulator.shape != residual.shape or accumulator.shape[-1] != HIDDEN_WIDTH:
        raise ValueError("QECR producer geometry differs from the frozen 896-lane shape")
    if not 0 <= layer_index < LAYER_COUNT:
        raise ValueError("QECR producer layer index must be in 0..23")

    accumulator_rows = accumulator.detach().cpu().reshape(-1, HIDDEN_WIDTH).tolist()
    residual_rows = residual.detach().cpu().reshape(-1, HIDDEN_WIDTH).tolist()
    hidden_rows: list[list[int]] = []
    carry_rows: list[list[int]] = []
    maximum_absolute_numerator = 0
    maximum_denominator = 0
    maximum_absolute_carry = 0
    common_exponent_minimum = 127
    common_exponent_maximum = -128
    accumulator_records = metadata.accumulator_scale32[layer_index]
    residual_record = metadata.residual_scale32[layer_index]
    destination_record = metadata.destination_scale32[layer_index]

    for accumulator_row, residual_row in zip(
        accumulator_rows,
        residual_rows,
        strict=True,
    ):
        hidden_row: list[int] = []
        carry_row: list[int] = []
        for lane, (accumulator_value, residual_value) in enumerate(
            zip(accumulator_row, residual_row, strict=True)
        ):
            result = producer_lane(
                accumulator_value,
                residual_value,
                accumulator_records[lane],
                residual_record,
                destination_record,
            )
            hidden_row.append(result.hidden_s8)
            carry_row.append(result.carry_s16_q15)
            maximum_absolute_numerator = max(
                maximum_absolute_numerator,
                abs(result.numerator_s96),
            )
            maximum_denominator = max(maximum_denominator, result.denominator_u64)
            maximum_absolute_carry = max(
                maximum_absolute_carry,
                abs(result.carry_s16_q15),
            )
            common_exponent_minimum = min(
                common_exponent_minimum,
                result.common_exponent,
            )
            common_exponent_maximum = max(
                common_exponent_maximum,
                result.common_exponent,
            )
        hidden_rows.append(hidden_row)
        carry_rows.append(carry_row)

    hidden = torch.tensor(hidden_rows, dtype=torch.int8).reshape(residual.shape)
    carry = torch.tensor(carry_rows, dtype=torch.int16).reshape(residual.shape)
    trace = ProducerTrace(
        layer_index=layer_index,
        rows=len(hidden_rows),
        lanes=accumulator.numel(),
        maximum_absolute_numerator=maximum_absolute_numerator,
        maximum_denominator=maximum_denominator,
        maximum_absolute_carry_q15=maximum_absolute_carry,
        common_exponent_minimum=common_exponent_minimum,
        common_exponent_maximum=common_exponent_maximum,
    )
    return hidden.to(residual.device), carry.to(residual.device), trace


def consume_rmsnorm_exact(
    hidden: Tensor,
    carry: Tensor,
    scaled_gains_q8: Tensor,
    *,
    consumer_layer_id: int,
    producer_layer_index: int,
) -> tuple[Tensor, ConsumerTrace]:
    if hidden.dtype != torch.int8 or carry.dtype != torch.int16:
        raise TypeError("QECR consumer requires signed-int8 hidden and signed-int16 carry")
    if hidden.shape != carry.shape or hidden.shape[-1] != HIDDEN_WIDTH:
        raise ValueError("QECR consumer geometry differs from the frozen 896-lane shape")
    if scaled_gains_q8.dtype != torch.int16 or scaled_gains_q8.shape != (
        HIDDEN_WIDTH,
    ):
        raise ValueError("QECR consumer gain metadata differs from signed-Q7.8 x896")
    if not 1 <= consumer_layer_id <= LAYER_COUNT:
        raise ValueError("QECR consumer layer id must be in 1..24")

    hidden_rows = hidden.detach().cpu().reshape(-1, HIDDEN_WIDTH).tolist()
    carry_rows = carry.detach().cpu().reshape(-1, HIDDEN_WIDTH).tolist()
    gains = scaled_gains_q8.detach().cpu().tolist()
    output_rows: list[list[int]] = []
    maximum_absolute_carry = 0
    maximum_absolute_reconstructed = 0
    rows_with_saturation = 0
    for hidden_row, carry_row in zip(hidden_rows, carry_rows, strict=True):
        result = rmsnorm(hidden_row, carry_row, gains)
        output_rows.append(list(result.outputs_s8))
        maximum_absolute_carry = max(
            maximum_absolute_carry,
            max(abs(value) for value in carry_row),
        )
        maximum_absolute_reconstructed = max(
            maximum_absolute_reconstructed,
            max(abs(value) for value in result.reconstructed_q15),
        )
        rows_with_saturation += int(result.saturation_seen)

    output = torch.tensor(output_rows, dtype=torch.int8).reshape(hidden.shape)
    trace = ConsumerTrace(
        consumer_layer_id=consumer_layer_id,
        producer_layer_index=producer_layer_index,
        rows=len(output_rows),
        lanes=hidden.numel(),
        maximum_absolute_carry_q15=maximum_absolute_carry,
        maximum_absolute_reconstructed_q15=maximum_absolute_reconstructed,
        rows_with_output_saturation=rows_with_saturation,
    )
    return output.to(hidden.device), trace


class CrossLayerErrorCarryRuntime:
    """Enforce 24 ordered producers and 24 exactly-once RMSNorm consumers."""

    def __init__(self, metadata: CrossLayerCarryMetadata) -> None:
        metadata.validate()
        self.metadata = metadata
        self._next_producer = 0
        self._expected_consumer: int | None = None
        self._hidden: Tensor | None = None
        self._carry: Tensor | None = None
        self._producer_records: list[dict[str, Any]] = []
        self._consumer_records: list[dict[str, Any]] = []
        self._completed: list[dict[str, Any]] = []
        self._sticky_numeric_overflow = False

    def _begin_pass(self) -> None:
        self._next_producer = 0
        self._expected_consumer = None
        self._hidden = None
        self._carry = None
        self._producer_records = []
        self._consumer_records = []
        self._sticky_numeric_overflow = False

    def destination_scale(self, layer_index: int) -> float:
        return self.metadata.destination_float(layer_index)

    def produce(
        self,
        layer_index: int,
        accumulator: Tensor,
        residual: Tensor,
        baseline_down_projection: Tensor,
        baseline_post_mlp: Tensor,
    ) -> Tensor:
        if layer_index == 0:
            if self._expected_consumer is not None:
                raise RuntimeError("QECR pass restarted with live unconsumed carry")
            if self._next_producer not in {0, LAYER_COUNT}:
                raise RuntimeError("QECR pass restarted before 24 ordered producers")
            self._begin_pass()
        if layer_index != self._next_producer:
            raise RuntimeError(
                f"QECR runtime expected producer {self._next_producer}, got {layer_index}"
            )
        if self._expected_consumer is not None or self._carry is not None:
            raise RuntimeError("QECR producer attempted while prior carry remains valid")
        try:
            hidden, carry, trace = produce_layer_exact(
                accumulator,
                residual,
                self.metadata,
                layer_index,
            )
        except OverflowError:
            self._sticky_numeric_overflow = True
            raise
        self._hidden = hidden
        self._carry = carry
        self._expected_consumer = layer_index + 1
        self._next_producer += 1
        self._producer_records.append(
            {
                **asdict(trace),
                "accumulator_s32_sha256": _sha256_tensor(accumulator),
                "residual_s8_sha256": _sha256_tensor(residual),
                "baseline_down_projection_s8_sha256": _sha256_tensor(
                    baseline_down_projection
                ),
                "baseline_post_mlp_s8_sha256": _sha256_tensor(baseline_post_mlp),
                "hidden_s8_sha256": _sha256_tensor(hidden),
                "carry_s16_q15_sha256": _sha256_tensor(carry),
            }
        )
        return hidden

    def consume(self, consumer_layer_id: int, scaled_gains_q8: Tensor) -> Tensor:
        if self._expected_consumer != consumer_layer_id:
            raise RuntimeError(
                f"QECR runtime expected consumer {self._expected_consumer}, "
                f"got {consumer_layer_id}"
            )
        if self._hidden is None or self._carry is None:
            raise RuntimeError("QECR consumer lacks a valid carry payload")
        producer_layer_index = consumer_layer_id - 1
        output, trace = consume_rmsnorm_exact(
            self._hidden,
            self._carry,
            scaled_gains_q8,
            consumer_layer_id=consumer_layer_id,
            producer_layer_index=producer_layer_index,
        )
        self._consumer_records.append(
            {
                **asdict(trace),
                "hidden_s8_sha256": _sha256_tensor(self._hidden),
                "carry_s16_q15_sha256": _sha256_tensor(self._carry),
                "rmsnorm_output_s8_sha256": _sha256_tensor(output),
            }
        )
        self._hidden = None
        self._carry = None
        self._expected_consumer = None
        if consumer_layer_id == LAYER_COUNT:
            if self._next_producer != LAYER_COUNT:
                raise RuntimeError("QECR final consumer arrived before all producers")
            self._completed.append(self._summary(completed=True))
        return output

    def _summary(self, *, completed: bool) -> dict[str, Any]:
        records = {
            "producers": self._producer_records,
            "consumers": self._consumer_records,
        }
        encoded = json.dumps(records, indent=2, sort_keys=True).encode()
        return {
            "contract_id": CONTRACT_ID,
            "completed": completed,
            "layer0_zero_carry_implicit": True,
            "producers_executed": len(self._producer_records),
            "consumers_executed": len(self._consumer_records),
            "ordered_producer_layers": [
                record["layer_index"] for record in self._producer_records
            ],
            "ordered_consumer_layer_ids": [
                record["consumer_layer_id"] for record in self._consumer_records
            ],
            "final_output_consumed": bool(
                self._consumer_records
                and self._consumer_records[-1]["consumer_layer_id"] == LAYER_COUNT
            ),
            "carry_valid_after_pass": self._carry is not None,
            "sticky_numeric_overflow": self._sticky_numeric_overflow,
            "ordered_trace_sha256": hashlib.sha256(encoded).hexdigest(),
            **records,
        }

    def snapshot(self) -> dict[str, Any]:
        return self._summary(completed=False)

    def consume_completed_pass(self) -> dict[str, Any]:
        if len(self._completed) != 1:
            raise RuntimeError("QECR runtime lacks exactly one completed pass")
        return self._completed.pop()
