#!/usr/bin/env python3
"""Static-only V8 c02 framing and validated cross-parser surface.

The module has no CLI, filesystem reader, publisher, or execution entry point.
Both parsers consume caller-provided bytes.  Each parser independently checks
framing, binding digests, BF16 payload geometry, complete-record finiteness,
and EOF before returning records to any future transform.
"""

from __future__ import annotations

import hashlib
import io
import math
import struct
from typing import Any


C02_MAGIC = b"ACE2-C02-TENSORS-V1\n"
MAX_RECORDS = 256
MAX_NAME_BYTES = 1024
MAX_DTYPE_BYTES = 64
MAX_RANK = 8
MAX_DIMENSION = 1 << 20
MAX_PAYLOAD_BYTES = 1 << 31


class C02Error(ValueError):
    """The supplied c02 bytes or immutable bindings are not acceptable."""


def _frame_require(condition: bool, message: str) -> None:
    if not condition:
        raise C02Error(message)


def record_sha256(dtype: str, shape: tuple[int, ...], payload: bytes) -> str:
    """Fixture-side digest helper; neither parser delegates to this helper."""

    _frame_require(type(dtype) is str, "digest dtype")
    _frame_require(type(shape) is tuple, "digest shape")
    _frame_require(type(payload) is bytes, "digest payload")
    try:
        dtype_bytes = dtype.encode("ascii")
    except UnicodeError as error:
        raise C02Error("digest dtype encoding") from error
    digest = hashlib.sha256()
    digest.update(dtype_bytes)
    digest.update(b"\0")
    digest.update(struct.pack(">I", len(shape)))
    for dimension in shape:
        _frame_require(type(dimension) is int and dimension >= 0, "digest dimension")
        digest.update(struct.pack(">Q", dimension))
    digest.update(payload)
    return digest.hexdigest()


def produce_c02_bundle(records: list[dict[str, Any]]) -> bytes:
    """Produce the exact c02 byte grammar from in-memory synthetic records."""

    _frame_require(type(records) is list and 1 <= len(records) <= MAX_RECORDS, "c02 record count")
    for index, value in enumerate(records):
        _frame_require(
            type(value) is dict and set(value) == {"dtype", "name", "payload", "shape"},
            f"record[{index}] exact keys",
        )
        _frame_require(type(value["name"]) is str, f"record[{index}] name")
    names: set[str] = set()
    output = bytearray(C02_MAGIC)
    output.extend(struct.pack(">I", len(records)))
    for index, value in enumerate(sorted(records, key=lambda item: item["name"])):
        name = value["name"]
        dtype = value["dtype"]
        shape = value["shape"]
        payload = value["payload"]
        _frame_require(type(name) is str and name not in names, f"record[{index}] name")
        _frame_require(type(dtype) is str, f"record[{index}] dtype")
        _frame_require(type(shape) is tuple and 1 <= len(shape) <= MAX_RANK, f"record[{index}] shape")
        _frame_require(type(payload) is bytes and len(payload) <= MAX_PAYLOAD_BYTES, f"record[{index}] payload")
        _frame_require(
            all(type(dimension) is int and 1 <= dimension <= MAX_DIMENSION for dimension in shape),
            f"record[{index}] dimensions",
        )
        try:
            name_bytes = name.encode("utf-8")
            dtype_bytes = dtype.encode("ascii")
        except UnicodeError as error:
            raise C02Error(f"record[{index}] text encoding") from error
        _frame_require(1 <= len(name_bytes) <= MAX_NAME_BYTES, f"record[{index}] name bytes")
        _frame_require(1 <= len(dtype_bytes) <= MAX_DTYPE_BYTES, f"record[{index}] dtype bytes")
        names.add(name)
        output.extend(struct.pack(">H", len(name_bytes)))
        output.extend(name_bytes)
        output.append(len(dtype_bytes))
        output.extend(dtype_bytes)
        output.append(len(shape))
        for dimension in shape:
            output.extend(struct.pack(">Q", dimension))
        output.extend(struct.pack(">Q", len(payload)))
        output.extend(payload)
    return bytes(output)


def parse_c02_producer_bytes(data: bytes, bindings: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Cursor parser mirroring the c02 producer grammar without delegation."""

    if type(data) is not bytes:
        raise C02Error("producer bytes type")
    if type(bindings) is not dict:
        raise C02Error("producer bindings type")
    offset = 0

    def take(size: int) -> bytes:
        nonlocal offset
        if type(size) is not int or size < 0 or offset + size > len(data):
            raise C02Error("producer truncated c02 bytes")
        value = data[offset : offset + size]
        offset += size
        return value

    if take(len(C02_MAGIC)) != C02_MAGIC:
        raise C02Error("producer c02 magic")
    count = int.from_bytes(take(4), "big")
    if not 1 <= count <= MAX_RECORDS:
        raise C02Error("producer c02 record count")
    records: dict[str, dict[str, Any]] = {}
    previous_name: str | None = None
    for _ in range(count):
        name_size = int.from_bytes(take(2), "big")
        if not 1 <= name_size <= MAX_NAME_BYTES:
            raise C02Error("producer c02 name size")
        try:
            name = take(name_size).decode("utf-8", "strict")
        except UnicodeError as error:
            raise C02Error("producer c02 name encoding") from error
        if previous_name is not None and name <= previous_name:
            raise C02Error("producer c02 record name order")
        if name in records:
            raise C02Error("producer duplicate c02 name")
        previous_name = name
        dtype_size = take(1)[0]
        if not 1 <= dtype_size <= MAX_DTYPE_BYTES:
            raise C02Error("producer c02 dtype size")
        try:
            dtype = take(dtype_size).decode("ascii", "strict")
        except UnicodeError as error:
            raise C02Error("producer c02 dtype encoding") from error
        rank = take(1)[0]
        if not 1 <= rank <= MAX_RANK:
            raise C02Error("producer c02 rank")
        shape = tuple(int.from_bytes(take(8), "big") for _ in range(rank))
        if not all(1 <= dimension <= MAX_DIMENSION for dimension in shape):
            raise C02Error("producer c02 shape")
        payload_size = int.from_bytes(take(8), "big")
        if payload_size > MAX_PAYLOAD_BYTES:
            raise C02Error("producer c02 payload size")
        payload = take(payload_size)

        binding = bindings.get(name)
        if type(binding) is not dict or set(binding) != {"dtype", "sha256", "shape"}:
            raise C02Error("producer c02 binding exact keys")
        expected_shape = binding["shape"]
        expected_sha256 = binding["sha256"]
        if binding["dtype"] != dtype:
            raise C02Error("producer c02 binding dtype")
        if type(expected_shape) is not list or any(type(item) is not int for item in expected_shape) or tuple(expected_shape) != shape:
            raise C02Error("producer c02 binding shape")
        if type(expected_sha256) is not str or len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
            raise C02Error("producer c02 binding digest syntax")
        if dtype == "torch.bfloat16":
            if payload_size != 2 * math.prod(shape):
                raise C02Error("producer BF16 payload boundary")

        digest = hashlib.sha256()
        digest.update(dtype.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(shape).to_bytes(4, "big"))
        for dimension in shape:
            digest.update(dimension.to_bytes(8, "big"))
        digest.update(payload)
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise C02Error("producer c02 binding digest")

        if dtype == "torch.bfloat16":
            for word_index, (word,) in enumerate(struct.iter_unpack("<H", payload)):
                if ((word >> 7) & 0xFF) == 0xFF:
                    raise C02Error(f"producer nonfinite BF16 at complete-record word {word_index}")
        records[name] = {"dtype": dtype, "payload": payload, "sha256": observed_sha256, "shape": shape}
    if offset != len(data):
        raise C02Error("producer trailing c02 bytes")
    if set(records) != set(bindings):
        raise C02Error("producer c02 binding record set")
    return records


def parse_c02_accepted_reader(data: bytes, bindings: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Independent BytesIO/Struct accepted-reader implementation."""

    if type(data) is not bytes:
        raise C02Error("accepted-reader bytes type")
    if type(bindings) is not dict:
        raise C02Error("accepted-reader bindings type")
    stream = io.BytesIO(data)

    def read_exact(size: int) -> bytes:
        if type(size) is not int or size < 0:
            raise C02Error("accepted-reader read size")
        value = stream.read(size)
        if len(value) != size:
            raise C02Error("accepted-reader truncated c02 bytes")
        return value

    if read_exact(len(C02_MAGIC)) != C02_MAGIC:
        raise C02Error("accepted-reader c02 magic")
    count = struct.Struct(">I").unpack(read_exact(4))[0]
    if count < 1 or count > MAX_RECORDS:
        raise C02Error("accepted-reader c02 record count")
    records: dict[str, dict[str, Any]] = {}
    last_name: str | None = None
    for _ in range(count):
        name_size = struct.Struct(">H").unpack(read_exact(2))[0]
        if name_size == 0 or name_size > MAX_NAME_BYTES:
            raise C02Error("accepted-reader c02 name size")
        try:
            name = str(read_exact(name_size), "utf-8", "strict")
        except UnicodeError as error:
            raise C02Error("accepted-reader c02 name encoding") from error
        if last_name is not None and not last_name < name:
            raise C02Error("accepted-reader c02 record name order")
        if records.__contains__(name):
            raise C02Error("accepted-reader duplicate c02 name")
        last_name = name
        dtype_size = struct.Struct("B").unpack(read_exact(1))[0]
        if dtype_size == 0 or dtype_size > MAX_DTYPE_BYTES:
            raise C02Error("accepted-reader c02 dtype size")
        try:
            dtype = str(read_exact(dtype_size), "ascii", "strict")
        except UnicodeError as error:
            raise C02Error("accepted-reader c02 dtype encoding") from error
        rank = struct.Struct("B").unpack(read_exact(1))[0]
        if rank == 0 or rank > MAX_RANK:
            raise C02Error("accepted-reader c02 rank")
        dimensions: list[int] = []
        for _dimension_index in range(rank):
            dimension = struct.Struct(">Q").unpack(read_exact(8))[0]
            if dimension == 0 or dimension > MAX_DIMENSION:
                raise C02Error("accepted-reader c02 shape")
            dimensions.append(dimension)
        shape = tuple(dimensions)
        payload_size = struct.Struct(">Q").unpack(read_exact(8))[0]
        if payload_size > MAX_PAYLOAD_BYTES:
            raise C02Error("accepted-reader c02 payload size")
        payload = read_exact(payload_size)

        if name not in bindings:
            raise C02Error("accepted-reader missing c02 binding")
        binding = bindings[name]
        if type(binding) is not dict:
            raise C02Error("accepted-reader c02 binding type")
        if sorted(binding) != ["dtype", "sha256", "shape"]:
            raise C02Error("accepted-reader c02 binding exact keys")
        expected_dtype = binding["dtype"]
        expected_shape = binding["shape"]
        expected_digest = binding["sha256"]
        if type(expected_dtype) is not str or expected_dtype != dtype:
            raise C02Error("accepted-reader c02 binding dtype")
        if type(expected_shape) is not list or len(expected_shape) != len(shape):
            raise C02Error("accepted-reader c02 binding shape type")
        for expected_dimension, observed_dimension in zip(expected_shape, shape):
            if type(expected_dimension) is not int or expected_dimension != observed_dimension:
                raise C02Error("accepted-reader c02 binding shape")
        if type(expected_digest) is not str or len(expected_digest) != 64:
            raise C02Error("accepted-reader c02 binding digest syntax")
        if not all(character in "0123456789abcdef" for character in expected_digest):
            raise C02Error("accepted-reader c02 binding digest alphabet")
        if dtype == "torch.bfloat16" and len(payload) != 2 * math.prod(dimensions):
            raise C02Error("accepted-reader BF16 payload boundary")

        digest_prefix = dtype.encode("ascii") + b"\0" + struct.pack(">I", rank)
        dimension_bytes = b"".join(struct.pack(">Q", dimension) for dimension in dimensions)
        observed_digest = hashlib.sha256(digest_prefix + dimension_bytes + payload).hexdigest()
        if observed_digest != expected_digest:
            raise C02Error("accepted-reader c02 binding digest")

        if dtype == "torch.bfloat16":
            for word_index in range(len(payload) // 2):
                low = payload[word_index * 2]
                high = payload[word_index * 2 + 1]
                exponent_field = (((high << 8) | low) >> 7) & 0xFF
                if exponent_field == 0xFF:
                    raise C02Error(f"accepted-reader nonfinite BF16 at complete-record word {word_index}")
        records[name] = {"dtype": dtype, "payload": payload, "sha256": observed_digest, "shape": shape}
    if stream.read(1) != b"":
        raise C02Error("accepted-reader trailing c02 bytes")
    if len(records) != len(bindings) or any(name not in records for name in bindings):
        raise C02Error("accepted-reader c02 binding record set")
    return records
