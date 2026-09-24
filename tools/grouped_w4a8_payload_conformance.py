#!/usr/bin/env python3
"""Build and verify deterministic grouped-W4A8 projection payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

if __package__:
    from tools import ace2_quality_contracts as quality
    from tools import model_hardware_contract as hardware
    from tools import quantization_policy as quantization
else:
    import ace2_quality_contracts as quality
    import model_hardware_contract as hardware
    import quantization_policy as quantization


MAGIC = b"ACE2W4P1"
SCHEMA_VERSION = 1
HEADER = struct.Struct("<8sIIIHHHHI")
PACKAGE_MAGIC = b"ACE2W4M1"
PACKAGE_SCHEMA_VERSION = 1
PACKAGE_HEADER = struct.Struct("<8sII")
PACKAGE_ENTRY_HEADER = struct.Struct("<HI")


class PayloadError(ValueError):
    """Raised when a grouped-W4A8 projection payload is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PayloadError(message)


def _positive_integer(value: Any, name: str) -> int:
    _require(type(value) is int and value > 0, f"{name} must be a positive integer")
    return value


def _layout(descriptor: Mapping[str, Any]) -> tuple[int, int, int]:
    try:
        layout = descriptor["weight_layout"]
        group_size = _positive_integer(layout["input_group_size"], "input group size")
        packed_group_bytes = _positive_integer(
            layout["packed_group_bytes"], "packed group bytes"
        )
        scale_record_bytes = _positive_integer(
            layout["scale_record_bytes"], "scale record bytes"
        )
    except (KeyError, TypeError) as exc:
        raise PayloadError("model descriptor weight layout is malformed") from exc
    _require(group_size % 2 == 0, "input group size must be even")
    _require(
        packed_group_bytes * 2 == group_size,
        "packed group bytes differ from signed-int4 geometry",
    )
    _require(scale_record_bytes == 4, "Scale32 record width differs")
    _require(
        layout.get("packing") == "even_input_low_nibble_odd_input_high_nibble",
        "signed-int4 packing order differs",
    )
    _require(
        layout.get("scale_record_format") == "scale32_little_endian_u32",
        "Scale32 record format differs",
    )
    return group_size, packed_group_bytes, scale_record_bytes


def projection_input_channels(
    descriptor: Mapping[str, Any], projection: str
) -> int:
    try:
        dimensions = descriptor["dimensions"]
        hidden_size = _positive_integer(dimensions["hidden_size"], "hidden size")
        intermediate_size = _positive_integer(
            dimensions["intermediate_size"], "intermediate size"
        )
    except (KeyError, TypeError) as exc:
        raise PayloadError("model descriptor dimensions are malformed") from exc
    if projection in {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "lm_head",
    }:
        return hidden_size
    if projection == "down_proj":
        return intermediate_size
    raise PayloadError(f"unsupported projection family: {projection}")


def _validate_matrix(
    weights: Sequence[Sequence[int]],
    scale32_records: Sequence[Sequence[int]],
    input_channels: int,
    group_count: int,
) -> int:
    _require(
        isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)),
        "weights must be a sequence",
    )
    output_channels = len(weights)
    _require(output_channels > 0, "payload must contain at least one output channel")
    _require(
        isinstance(scale32_records, Sequence)
        and not isinstance(scale32_records, (str, bytes))
        and len(scale32_records) == output_channels,
        "Scale32 output geometry differs",
    )
    for output, (row, records) in enumerate(
        zip(weights, scale32_records, strict=True)
    ):
        _require(
            isinstance(row, Sequence)
            and not isinstance(row, (str, bytes))
            and len(row) == input_channels,
            f"weight row {output} geometry differs",
        )
        for value in row:
            _require(
                type(value) is int and -8 <= value <= 7,
                "signed-int4 weight code is outside -8..7",
            )
        _require(
            isinstance(records, Sequence)
            and not isinstance(records, (str, bytes))
            and len(records) == group_count,
            f"Scale32 row {output} geometry differs",
        )
        for record in records:
            _require(type(record) is int, "Scale32 record must be an integer")
            try:
                quality.unpack_scale32(record)
            except ValueError as exc:
                raise PayloadError(f"invalid Scale32 record: {exc}") from exc
    return output_channels


def encode_payload(
    descriptor: Mapping[str, Any],
    input_channels: int,
    weights: Sequence[Sequence[int]],
    scale32_records: Sequence[Sequence[int]],
) -> bytes:
    group_size, packed_group_bytes, scale_record_bytes = _layout(descriptor)
    input_channels = _positive_integer(input_channels, "input channels")
    _require(
        input_channels % group_size == 0,
        "input channels are not divisible by the descriptor group size",
    )
    group_count = input_channels // group_size
    output_channels = _validate_matrix(
        weights, scale32_records, input_channels, group_count
    )

    packed_weights = bytearray()
    for row in weights:
        for group in range(group_count):
            start = group * group_size
            packed = quantization.pack_signed_int4(row[start : start + group_size])
            _require(
                len(packed) == packed_group_bytes,
                "packed signed-int4 group width differs",
            )
            packed_weights.extend(packed)

    packed_scales = bytearray()
    for records in scale32_records:
        for record in records:
            packed_scales.extend(struct.pack("<I", record))

    body_bytes = len(packed_weights) + len(packed_scales)
    header = HEADER.pack(
        MAGIC,
        SCHEMA_VERSION,
        input_channels,
        output_channels,
        group_size,
        packed_group_bytes,
        scale_record_bytes,
        0,
        body_bytes,
    )
    return header + packed_weights + packed_scales


def decode_payload(
    payload: bytes,
    descriptor: Mapping[str, Any],
    *,
    expected_input_channels: int | None = None,
    expected_output_channels: int | None = None,
) -> dict[str, Any]:
    _require(type(payload) is bytes, "projection payload must be bytes")
    _require(len(payload) >= HEADER.size, "projection payload header is truncated")
    (
        magic,
        schema_version,
        input_channels,
        output_channels,
        group_size,
        packed_group_bytes,
        scale_record_bytes,
        reserved,
        body_bytes,
    ) = HEADER.unpack_from(payload)
    expected_layout = _layout(descriptor)
    _require(magic == MAGIC, "projection payload magic differs")
    _require(schema_version == SCHEMA_VERSION, "projection payload schema differs")
    _require(reserved == 0, "projection payload reserved field is nonzero")
    _require(
        (group_size, packed_group_bytes, scale_record_bytes) == expected_layout,
        "projection payload layout differs from the model descriptor",
    )
    _positive_integer(input_channels, "payload input channels")
    _positive_integer(output_channels, "payload output channels")
    _require(
        input_channels % group_size == 0,
        "payload input channels are not divisible by the descriptor group size",
    )
    if expected_input_channels is not None:
        _require(
            input_channels == expected_input_channels,
            "projection payload input geometry differs",
        )
    if expected_output_channels is not None:
        _require(
            output_channels == expected_output_channels,
            "projection payload output geometry differs",
        )
    group_count = input_channels // group_size
    weight_bytes = output_channels * group_count * packed_group_bytes
    scale_bytes = output_channels * group_count * scale_record_bytes
    expected_body_bytes = weight_bytes + scale_bytes
    _require(body_bytes == expected_body_bytes, "projection payload body size differs")
    _require(
        len(payload) == HEADER.size + expected_body_bytes,
        "projection payload length differs",
    )

    weights: list[list[int]] = []
    offset = HEADER.size
    for _ in range(output_channels):
        row: list[int] = []
        for _ in range(group_count):
            packed = payload[offset : offset + packed_group_bytes]
            offset += packed_group_bytes
            row.extend(quantization.unpack_signed_int4(packed, group_size))
        weights.append(row)

    scale32_records: list[list[int]] = []
    for _ in range(output_channels):
        records = []
        for _ in range(group_count):
            record = struct.unpack_from("<I", payload, offset)[0]
            offset += scale_record_bytes
            try:
                quality.unpack_scale32(record)
            except ValueError as exc:
                raise PayloadError(f"invalid Scale32 record: {exc}") from exc
            records.append(record)
        scale32_records.append(records)
    _require(offset == len(payload), "projection payload parser did not consume the body")
    return {
        "group_count": group_count,
        "input_channels": input_channels,
        "output_channels": output_channels,
        "scale32_records": scale32_records,
        "weights": weights,
    }


def encode_payload_package(
    payloads: Mapping[str, bytes],
    expected_names: Sequence[str],
) -> bytes:
    _require(isinstance(payloads, Mapping), "payload package members must be a mapping")
    names = tuple(expected_names)
    _require(
        names
        and all(type(name) is str and name for name in names)
        and len(set(names)) == len(names),
        "payload package names are invalid",
    )
    _require(set(payloads) == set(names), "payload package member set differs")
    encoded = bytearray(
        PACKAGE_HEADER.pack(
            PACKAGE_MAGIC,
            PACKAGE_SCHEMA_VERSION,
            len(names),
        )
    )
    for name in names:
        try:
            encoded_name = name.encode("ascii")
        except UnicodeEncodeError as exc:
            raise PayloadError("payload package member name is not ASCII") from exc
        payload = payloads[name]
        _require(type(payload) is bytes, "payload package member must be bytes")
        _require(
            len(encoded_name) <= 0xFFFF,
            "payload package member name is too long",
        )
        _require(
            len(payload) <= 0xFFFFFFFF,
            "payload package member is too large",
        )
        encoded.extend(PACKAGE_ENTRY_HEADER.pack(len(encoded_name), len(payload)))
        encoded.extend(encoded_name)
        encoded.extend(payload)
    return bytes(encoded)


def decode_payload_package(
    package: bytes,
    expected_names: Sequence[str],
) -> dict[str, bytes]:
    _require(type(package) is bytes, "payload package must be bytes")
    _require(len(package) >= PACKAGE_HEADER.size, "payload package header is truncated")
    magic, schema_version, member_count = PACKAGE_HEADER.unpack_from(package)
    names = tuple(expected_names)
    _require(magic == PACKAGE_MAGIC, "payload package magic differs")
    _require(
        schema_version == PACKAGE_SCHEMA_VERSION,
        "payload package schema differs",
    )
    _require(member_count == len(names), "payload package member count differs")
    payloads: dict[str, bytes] = {}
    offset = PACKAGE_HEADER.size
    for expected_name in names:
        _require(
            len(package) - offset >= PACKAGE_ENTRY_HEADER.size,
            "payload package entry header is truncated",
        )
        name_bytes, payload_bytes = PACKAGE_ENTRY_HEADER.unpack_from(package, offset)
        offset += PACKAGE_ENTRY_HEADER.size
        entry_bytes = name_bytes + payload_bytes
        _require(
            len(package) - offset >= entry_bytes,
            "payload package entry is truncated",
        )
        encoded_name = package[offset : offset + name_bytes]
        offset += name_bytes
        try:
            name = encoded_name.decode("ascii")
        except UnicodeDecodeError as exc:
            raise PayloadError("payload package member name is not ASCII") from exc
        _require(name == expected_name, "payload package member order differs")
        payloads[name] = package[offset : offset + payload_bytes]
        offset += payload_bytes
    _require(offset == len(package), "payload package has trailing bytes")
    return payloads


def evaluate_projection(
    activations: Sequence[int],
    weights: Sequence[Sequence[int]],
    scale32_records: Sequence[Sequence[int]],
    group_size: int,
) -> list[Fraction]:
    input_channels = len(activations)
    _positive_integer(group_size, "group size")
    _require(
        input_channels > 0 and input_channels % group_size == 0,
        "activation geometry differs",
    )
    for value in activations:
        _require(
            type(value) is int and -128 <= value <= 127,
            "activation code is outside signed-int8",
        )
    group_count = input_channels // group_size
    _validate_matrix(weights, scale32_records, input_channels, group_count)
    outputs = []
    for row, records in zip(weights, scale32_records, strict=True):
        result = Fraction(0)
        for group, record in enumerate(records):
            start = group * group_size
            dot = sum(
                activation * weight
                for activation, weight in zip(
                    activations[start : start + group_size],
                    row[start : start + group_size],
                    strict=True,
                )
            )
            numerator, denominator = quality.scale32_ratio(record)
            result += Fraction(dot * numerator, denominator)
        outputs.append(result)
    return outputs


def _representative_case(
    descriptor: Mapping[str, Any],
    projection: str,
    output_channels: int,
) -> dict[str, Any]:
    group_size, _, _ = _layout(descriptor)
    input_channels = projection_input_channels(descriptor, projection)
    group_count = input_channels // group_size
    weights = [
        [((output * 5 + lane * 3 + (lane // group_size) * 7) % 16) - 8
         for lane in range(input_channels)]
        for output in range(output_channels)
    ]
    records = [
        [
            quality.pack_scale32(
                0x8000 + ((output * 257 + group * 17) % 0x8000),
                -((output + group) % 5),
            )
            for group in range(group_count)
        ]
        for output in range(output_channels)
    ]
    activations = [((lane * 11 + 3) % 255) - 127 for lane in range(input_channels)]
    payload = encode_payload(descriptor, input_channels, weights, records)
    decoded = decode_payload(
        payload,
        descriptor,
        expected_input_channels=input_channels,
        expected_output_channels=output_channels,
    )
    expected = evaluate_projection(activations, weights, records, group_size)
    observed = evaluate_projection(
        activations,
        decoded["weights"],
        decoded["scale32_records"],
        group_size,
    )
    _require(observed == expected, "decoded projection is numerically inequivalent")
    return {
        "group_count": group_count,
        "input_channels": input_channels,
        "numerical_output": [
            {"denominator": value.denominator, "numerator": value.numerator}
            for value in observed
        ],
        "output_channels": output_channels,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "projection": projection,
    }


def run_conformance(model_id: str = "qwen2.5-0.5b") -> dict[str, Any]:
    descriptor = hardware.load_descriptor(model_id)
    cases = [
        _representative_case(descriptor, "q_proj", 3),
        _representative_case(descriptor, "down_proj", 2),
    ]
    return {
        "cases": cases,
        "contract_id": descriptor["contract_id"],
        "model_id": model_id,
        "schema_version": 1,
        "status": "PASS_GROUPED_W4A8_PAYLOAD_CONFORMANCE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model", default="qwen2.5-0.5b")
    args = parser.parse_args()
    if not args.check:
        parser.error("--check is required")
    try:
        report = run_conformance(args.model)
    except (hardware.ContractError, PayloadError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
