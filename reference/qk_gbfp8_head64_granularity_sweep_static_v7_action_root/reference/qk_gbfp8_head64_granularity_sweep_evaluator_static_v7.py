#!/usr/bin/env python3
"""Pure static V7 grammar, arithmetic, and synthetic result construction.

There is deliberately no evaluator CLI, filesystem payload reader, or result
publisher in this module.  All helpers consume caller-provided immutable bytes
and exist so the static verifier can exercise the future evaluator contract
without opening the sealed tensor or creating live state.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
from copy import deepcopy
from typing import Any


PACKAGE_ID = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_PACKAGE"
RESULT_SCHEMA_ID = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_RESULT_V1"
GENERATION_ID = "qk-gbfp8-head64-granularity-sweep-v7-base"
SEALED_SET_ID = "w4a8-c02-attention-substage-trace-v2"
C02_MAGIC = b"ACE2-C02-TENSORS-V1\n"
HEAD_LANES = 64
MANTISSA_MIN = -127
MANTISSA_MAX = 127
RESERVED_MANTISSA = -128
MIN_INT16 = -(1 << 15)
MAX_INT16 = (1 << 15) - 1
MIN_INT64 = -(1 << 63)
MAX_INT64 = (1 << 63) - 1
CANDIDATES = (
    {"bytes_per_head": 80, "exponent_bytes_per_head": 16, "group_count": 8, "group_size": 8, "label": "G8", "mantissa_bytes_per_head": 64},
    {"bytes_per_head": 96, "exponent_bytes_per_head": 32, "group_count": 16, "group_size": 4, "label": "G4", "mantissa_bytes_per_head": 64},
    {"bytes_per_head": 128, "exponent_bytes_per_head": 64, "group_count": 32, "group_size": 2, "label": "G2", "mantissa_bytes_per_head": 64},
    {"bytes_per_head": 192, "exponent_bytes_per_head": 128, "group_count": 64, "group_size": 1, "label": "G1", "mantissa_bytes_per_head": 64},
)
HARD_GATES = {
    "cross_lane_record_count_maximum": 0,
    "invalid_or_non_finite_value_count_maximum": 0,
    "normalization_rejection_count_maximum": 0,
    "positive_centered_realized_score_count_maximum": 0,
    "rank_margin_violation_count_maximum": 0,
    "saturation_event_count_maximum": 0,
    "top_key_matching_fraction_minimum": {"denominator": 1, "numerator": 1},
    "top_key_mismatch_count_maximum": 0,
    "unique_oracle_positive_margin_preserved_fraction_minimum": {"denominator": 1, "numerator": 1},
}


class EvaluationError(ValueError):
    """A caller-provided static fixture violates the frozen V7 contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    require(type(value) is dict and set(value) == expected, f"{context} exact keys")
    return value


def tensor_record_sha256(dtype: str, shape: tuple[int, ...], payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack(">I", len(shape)))
    for dimension in shape:
        digest.update(struct.pack(">Q", dimension))
    digest.update(payload)
    return digest.hexdigest()


def _canonical_record(dtype: str, shape: tuple[int, ...], payload: bytes) -> dict[str, Any]:
    return {
        "dtype": dtype,
        "payload": payload,
        "sha256": tensor_record_sha256(dtype, shape, payload),
        "shape": shape,
    }


def produce_c02_bundle(records: list[dict[str, Any]]) -> bytes:
    """Produce the exact accepted c02 framing from in-memory record bytes."""

    require(type(records) is list and 1 <= len(records) <= 256, "c02 record count")
    names: set[str] = set()
    framed = [C02_MAGIC, struct.pack(">I", len(records))]
    for index, raw_record in enumerate(records):
        record = exact_keys(raw_record, {"dtype", "name", "payload", "shape"}, f"c02 record[{index}]")
        name = record["name"]
        dtype = record["dtype"]
        shape = record["shape"]
        payload = record["payload"]
        require(type(name) is str and name not in names, f"c02 record[{index}] name")
        require(type(dtype) is str, f"c02 record[{index}] dtype")
        require(type(shape) is tuple and 1 <= len(shape) <= 8, f"c02 record[{index}] shape")
        require(type(payload) is bytes and len(payload) <= (1 << 31), f"c02 record[{index}] payload")
        require(all(type(dimension) is int and 1 <= dimension <= (1 << 20) for dimension in shape), f"c02 record[{index}] dimensions")
        try:
            name_bytes = name.encode("utf-8")
            dtype_bytes = dtype.encode("ascii")
        except UnicodeError as error:
            raise EvaluationError(f"c02 record[{index}] text encoding") from error
        require(1 <= len(name_bytes) <= 1024, f"c02 record[{index}] name size")
        require(1 <= len(dtype_bytes) <= 64, f"c02 record[{index}] dtype size")
        names.add(name)
        framed.extend(
            [
                struct.pack(">H", len(name_bytes)),
                name_bytes,
                bytes((len(dtype_bytes),)),
                dtype_bytes,
                bytes((len(shape),)),
                b"".join(struct.pack(">Q", dimension) for dimension in shape),
                struct.pack(">Q", len(payload)),
                payload,
            ]
        )
    return b"".join(framed)


def parse_c02_producer_bytes(data: bytes) -> dict[str, dict[str, Any]]:
    """Parse c02 bytes using a cursor/int.from_bytes implementation."""

    require(type(data) is bytes, "c02 producer bytes type")
    offset = 0

    def take(size: int) -> bytes:
        nonlocal offset
        require(type(size) is int and size >= 0 and offset + size <= len(data), "truncated c02 bytes")
        value = data[offset : offset + size]
        offset += size
        return value

    require(take(len(C02_MAGIC)) == C02_MAGIC, "c02 magic")
    count = int.from_bytes(take(4), "big")
    require(1 <= count <= 256, "c02 record count")
    records: dict[str, dict[str, Any]] = {}
    for _ in range(count):
        name_size = int.from_bytes(take(2), "big")
        require(1 <= name_size <= 1024, "c02 name size")
        try:
            name = take(name_size).decode("utf-8")
        except UnicodeError as error:
            raise EvaluationError("c02 name encoding") from error
        require(name not in records, "duplicate c02 name")
        dtype_size = take(1)[0]
        require(1 <= dtype_size <= 64, "c02 dtype size")
        try:
            dtype = take(dtype_size).decode("ascii")
        except UnicodeError as error:
            raise EvaluationError("c02 dtype encoding") from error
        rank = take(1)[0]
        require(1 <= rank <= 8, "c02 rank")
        shape = tuple(int.from_bytes(take(8), "big") for _ in range(rank))
        require(all(1 <= dimension <= (1 << 20) for dimension in shape), "c02 shape")
        payload_size = int.from_bytes(take(8), "big")
        require(payload_size <= (1 << 31), "c02 payload size")
        payload = take(payload_size)
        records[name] = _canonical_record(dtype, shape, payload)
    require(offset == len(data), "trailing c02 bytes")
    return records


def parse_c02_accepted_reader(data: bytes) -> dict[str, dict[str, Any]]:
    """Independently cross-parse accepted c02 bytes with BytesIO/Struct."""

    require(type(data) is bytes, "c02 accepted-reader bytes type")
    handle = io.BytesIO(data)

    def read_exact(size: int) -> bytes:
        value = handle.read(size)
        require(len(value) == size, "accepted-reader truncated c02 bytes")
        return value

    require(read_exact(len(C02_MAGIC)) == C02_MAGIC, "accepted-reader c02 magic")
    count = struct.Struct(">I").unpack(read_exact(4))[0]
    require(1 <= count <= 256, "accepted-reader c02 record count")
    records: dict[str, dict[str, Any]] = {}
    for _ in range(count):
        name_size = struct.Struct(">H").unpack(read_exact(2))[0]
        require(1 <= name_size <= 1024, "accepted-reader c02 name size")
        try:
            name = read_exact(name_size).decode("utf-8")
        except UnicodeError as error:
            raise EvaluationError("accepted-reader c02 name encoding") from error
        require(name not in records, "accepted-reader duplicate c02 name")
        dtype_size = read_exact(1)[0]
        require(1 <= dtype_size <= 64, "accepted-reader c02 dtype size")
        try:
            dtype = read_exact(dtype_size).decode("ascii")
        except UnicodeError as error:
            raise EvaluationError("accepted-reader c02 dtype encoding") from error
        rank = read_exact(1)[0]
        require(1 <= rank <= 8, "accepted-reader c02 rank")
        shape = tuple(struct.Struct(">Q").unpack(read_exact(8))[0] for _ in range(rank))
        require(all(1 <= dimension <= (1 << 20) for dimension in shape), "accepted-reader c02 shape")
        payload_size = struct.Struct(">Q").unpack(read_exact(8))[0]
        require(payload_size <= (1 << 31), "accepted-reader c02 payload size")
        payload = read_exact(payload_size)
        records[name] = _canonical_record(dtype, shape, payload)
    require(handle.read(1) == b"", "accepted-reader trailing c02 bytes")
    return records


def validate_record_binding(record: dict[str, Any], binding: dict[str, Any]) -> None:
    value = exact_keys(record, {"dtype", "payload", "sha256", "shape"}, "c02 parsed record")
    expected = exact_keys(binding, {"dtype", "sha256", "shape"}, "c02 record binding")
    require(value["dtype"] == expected["dtype"], "c02 record dtype binding")
    require(value["shape"] == tuple(expected["shape"]), "c02 record shape binding")
    require(value["sha256"] == expected["sha256"], "c02 record digest binding")
    require(
        tensor_record_sha256(value["dtype"], value["shape"], value["payload"]) == value["sha256"],
        "c02 record digest recomputation",
    )


def finite_bf16_record_words(record: dict[str, Any]) -> tuple[int, ...]:
    """Validate a complete BF16 record before returning any transform input."""

    value = exact_keys(record, {"dtype", "payload", "sha256", "shape"}, "BF16 record")
    require(value["dtype"] == "torch.bfloat16", "BF16 record dtype")
    shape = value["shape"]
    payload = value["payload"]
    require(type(shape) is tuple and all(type(item) is int and item > 0 for item in shape), "BF16 record shape")
    require(type(payload) is bytes and len(payload) == 2 * math.prod(shape), "BF16 record payload bytes")
    require(tensor_record_sha256(value["dtype"], shape, payload) == value["sha256"], "BF16 record digest")
    words = tuple(word for (word,) in struct.iter_unpack("<H", payload))
    for index, word in enumerate(words):
        require(((word >> 7) & 0xFF) != 0xFF, f"nonfinite BF16 at complete-record word {index}")
    return words


def _bf16_components(word: int) -> tuple[bool, int, int]:
    require(type(word) is int and 0 <= word <= 0xFFFF, "BF16 word")
    exponent_field = (word >> 7) & 0xFF
    require(exponent_field != 0xFF, "nonfinite BF16")
    fraction = word & 0x7F
    coefficient, exponent = (fraction, -133) if exponent_field == 0 else (128 + fraction, exponent_field - 134)
    return bool(word & 0x8000), coefficient, exponent


def _round_unsigned_power_of_two(coefficient: int, shift: int) -> int:
    if shift >= 0:
        return coefficient << shift
    divisor = 1 << (-shift)
    retained, remainder = divmod(coefficient, divisor)
    half = divisor >> 1
    if remainder > half or (remainder == half and (retained & 1)):
        retained += 1
    return retained


def round_even_signed(value: int, right_shift: int) -> int:
    require(type(value) is int and type(right_shift) is int and right_shift > 0, "signed rounding request")
    magnitude = -value if value < 0 else value
    rounded = _round_unsigned_power_of_two(magnitude, -right_shift)
    return -rounded if value < 0 else rounded


def _select_group_exponent(components: tuple[tuple[bool, int, int], ...]) -> int:
    nonzero = tuple((coefficient, exponent) for _, coefficient, exponent in components if coefficient)
    if not nonzero:
        return 0
    candidate = min(exponent - 8 for coefficient, exponent in nonzero)
    require(MIN_INT16 <= candidate <= MAX_INT16, "group exponent search range")
    while True:
        fits = all(
            _round_unsigned_power_of_two(coefficient, source_exponent - candidate) <= MANTISSA_MAX
            for coefficient, source_exponent in nonzero
        )
        if fits:
            require(candidate == MIN_INT16 or any(
                _round_unsigned_power_of_two(coefficient, source_exponent - (candidate - 1)) > MANTISSA_MAX
                for coefficient, source_exponent in nonzero
            ), "group exponent minimality")
            return candidate
        candidate += 1
        require(candidate <= MAX_INT16, "finite BF16 group exponent overflow")


def encode_grouped_head(words: tuple[int, ...], group_size: int) -> dict[str, Any]:
    """Encode one finite 64-lane head with canonical per-group exponents."""

    require(type(words) is tuple and len(words) == HEAD_LANES, "grouped head lanes")
    require(group_size in (8, 4, 2, 1), "grouped head group size")
    components = tuple(_bf16_components(word) for word in words)
    mantissas: list[int] = []
    exponents: list[int] = []
    for offset in range(0, HEAD_LANES, group_size):
        group = components[offset : offset + group_size]
        exponent = _select_group_exponent(group)
        group_mantissas = []
        for negative, coefficient, source_exponent in group:
            magnitude = _round_unsigned_power_of_two(coefficient, source_exponent - exponent)
            mantissa = -magnitude if negative and magnitude else magnitude
            require(MANTISSA_MIN <= mantissa <= MANTISSA_MAX, "grouped mantissa range")
            require(mantissa != RESERVED_MANTISSA, "reserved grouped mantissa")
            group_mantissas.append(mantissa)
        require(exponent == 0 if not any(group_mantissas) else MIN_INT16 <= exponent <= MAX_INT16, "canonical grouped exponent")
        mantissas.extend(group_mantissas)
        exponents.append(exponent)
    return {"exponents": tuple(exponents), "group_size": group_size, "mantissas": tuple(mantissas)}


def encode_grouped_head_from_record(record: dict[str, Any], head_word_offset: int, group_size: int) -> dict[str, Any]:
    words = finite_bf16_record_words(record)
    require(type(head_word_offset) is int and 0 <= head_word_offset <= len(words) - HEAD_LANES, "head word offset")
    return encode_grouped_head(words[head_word_offset : head_word_offset + HEAD_LANES], group_size)


def validate_grouped_head(encoded: dict[str, Any]) -> dict[str, Any]:
    value = exact_keys(encoded, {"exponents", "group_size", "mantissas"}, "grouped head")
    group_size = value["group_size"]
    mantissas = value["mantissas"]
    exponents = value["exponents"]
    require(group_size in (8, 4, 2, 1), "grouped head group size")
    require(type(mantissas) is tuple and len(mantissas) == HEAD_LANES, "grouped mantissa count")
    require(type(exponents) is tuple and len(exponents) == HEAD_LANES // group_size, "grouped exponent count")
    for index, mantissa in enumerate(mantissas):
        require(type(mantissa) is int and MANTISSA_MIN <= mantissa <= MANTISSA_MAX, f"grouped mantissa[{index}]")
        require(mantissa != RESERVED_MANTISSA, f"reserved grouped mantissa[{index}]")
    for group_index, exponent in enumerate(exponents):
        require(type(exponent) is int and MIN_INT16 <= exponent <= MAX_INT16, f"grouped exponent[{group_index}]")
        group = mantissas[group_index * group_size : (group_index + 1) * group_size]
        require(exponent == 0 if not any(group) else True, f"zero grouped exponent[{group_index}]")
    return value


def pack_grouped_head(encoded: dict[str, Any]) -> bytes:
    value = validate_grouped_head(encoded)
    mantissa_bytes = bytes(mantissa & 0xFF for mantissa in value["mantissas"])
    exponent_bytes = b"".join(exponent.to_bytes(2, "little", signed=True) for exponent in value["exponents"])
    return mantissa_bytes + exponent_bytes


def unpack_grouped_head(payload: bytes, group_size: int) -> dict[str, Any]:
    require(type(payload) is bytes and group_size in (8, 4, 2, 1), "packed grouped head request")
    group_count = HEAD_LANES // group_size
    require(len(payload) == HEAD_LANES + 2 * group_count, "packed grouped head bytes")
    mantissas = tuple(value - 256 if value >= 128 else value for value in payload[:HEAD_LANES])
    exponents = tuple(
        int.from_bytes(payload[HEAD_LANES + 2 * index : HEAD_LANES + 2 * index + 2], "little", signed=True)
        for index in range(group_count)
    )
    return validate_grouped_head({"exponents": exponents, "group_size": group_size, "mantissas": mantissas})


def grouped_dot_score_pair(query: dict[str, Any], key: dict[str, Any]) -> tuple[int, int]:
    q = validate_grouped_head(query)
    k = validate_grouped_head(key)
    require(q["group_size"] == k["group_size"], "grouped dot geometry")
    group_size = q["group_size"]
    terms: list[tuple[int, int]] = []
    for group_index, (q_exponent, k_exponent) in enumerate(zip(q["exponents"], k["exponents"])):
        start = group_index * group_size
        stop = start + group_size
        coefficient = sum(
            q_mantissa * k_mantissa
            for q_mantissa, k_mantissa in zip(q["mantissas"][start:stop], k["mantissas"][start:stop])
        )
        if coefficient:
            terms.append((coefficient, q_exponent + k_exponent))
    if not terms:
        return 0, 0
    common_exponent = min(exponent for _, exponent in terms)
    coefficient = sum(value << (exponent - common_exponent) for value, exponent in terms)
    if coefficient == 0:
        return 0, 0
    while coefficient % 2 == 0:
        coefficient //= 2
        common_exponent += 1
    return coefficient, common_exponent


def realize_score_pair_q12_20(score_pair: tuple[int, int]) -> int:
    require(type(score_pair) is tuple and len(score_pair) == 2, "score pair")
    coefficient, exponent = score_pair
    require(type(coefficient) is int and type(exponent) is int, "score pair integers")
    if coefficient == 0:
        require(exponent == 0, "zero score pair exponent")
        return 0
    shift = exponent + 17
    realized = coefficient << shift if shift >= 0 else round_even_signed(coefficient, -shift)
    require(MIN_INT64 <= realized <= MAX_INT64, "Q12.20 score overflow")
    return realized


def normalize_score_row_q12_20(
    score_pairs: tuple[tuple[int, int], ...], valid_mask: tuple[bool, ...]
) -> tuple[tuple[int, ...], int, int]:
    require(type(score_pairs) is tuple and 1 <= len(score_pairs) <= 512, "score row")
    require(type(valid_mask) is tuple and len(valid_mask) == len(score_pairs), "score mask")
    require(all(type(valid) is bool for valid in valid_mask) and any(valid_mask), "score mask values")
    canonical: list[tuple[int, int]] = []
    for index, pair in enumerate(score_pairs):
        require(type(pair) is tuple and len(pair) == 2, f"score pair[{index}]")
        coefficient, exponent = pair
        require(type(coefficient) is int and type(exponent) is int, f"score pair[{index}] integers")
        canonical.append((0, 0) if coefficient == 0 else (coefficient, exponent))
    common_exponent = min(
        (exponent for (coefficient, exponent), valid in zip(canonical, valid_mask) if valid and coefficient),
        default=0,
    )
    aligned = [
        None if not valid else 0 if coefficient == 0 else coefficient << (exponent - common_exponent)
        for (coefficient, exponent), valid in zip(canonical, valid_mask)
    ]
    top = next(index for index, valid in enumerate(valid_mask) if valid)
    maximum = aligned[top]
    for index, valid in enumerate(valid_mask):
        if valid and aligned[index] > maximum:
            top = index
            maximum = aligned[index]
    realized: list[int] = []
    for value, valid in zip(aligned, valid_mask):
        if not valid:
            realized.append(0)
            continue
        centered = value - maximum
        shift = common_exponent + 17
        score = centered << shift if shift >= 0 else round_even_signed(centered, -shift)
        require(MIN_INT64 <= score <= 0, "centered Q12.20 score overflow or positivity")
        realized.append(score)
    require(realized[top] == 0, "Q12.20 top score")
    return tuple(realized), common_exponent, top


def kv_head_for_query(query_head: int) -> int:
    require(type(query_head) is int and 0 <= query_head < 14, "query head index")
    return query_head // 7


def compact_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def seal_result(result: dict[str, Any]) -> dict[str, Any]:
    sealed = deepcopy(result)
    sealed.pop("result_sha256", None)
    sealed["result_sha256"] = hashlib.sha256(compact_bytes(sealed)).hexdigest()
    return sealed


def _fraction_at_least(actual: dict[str, int], limit: dict[str, int]) -> bool:
    return actual["numerator"] * limit["denominator"] >= limit["numerator"] * actual["denominator"]


def _thresholds(metrics: dict[str, Any]) -> dict[str, Any]:
    invalid = metrics["invalid_accounting"]
    rank = metrics["rank_margin"]
    top = metrics["top_key"]
    actuals: dict[str, Any] = {
        "cross_lane_record_count_maximum": invalid["cross_lane_record_count"],
        "invalid_or_non_finite_value_count_maximum": invalid["invalid_or_non_finite_value_count"],
        "normalization_rejection_count_maximum": invalid["normalization_rejection_count"],
        "positive_centered_realized_score_count_maximum": invalid["positive_centered_realized_score_count"],
        "rank_margin_violation_count_maximum": rank["violation_count"],
        "saturation_event_count_maximum": invalid["saturation_event_count"],
        "top_key_matching_fraction_minimum": top["matching_fraction"],
        "top_key_mismatch_count_maximum": top["mismatch_count"],
        "unique_oracle_positive_margin_preserved_fraction_minimum": rank["preserved_positive_margin_fraction"],
    }
    checks: dict[str, Any] = {}
    for name, limit in HARD_GATES.items():
        actual = actuals[name]
        passed = _fraction_at_least(actual, limit) if type(limit) is dict else actual <= limit
        checks[name] = {"actual": actual, "limit": limit, "pass": passed}
    checks["all_hard_gates_pass"] = all(check["pass"] for check in checks.values())
    return checks


def _metrics(passes: bool, ordinal: int) -> dict[str, Any]:
    rows = 8
    matching = rows if passes else rows - 1
    preserved = rows if passes else rows - 1
    return {
        "invalid_accounting": {
            "cross_lane_record_count": 0,
            "invalid_or_non_finite_value_count": 0,
            "normalization_rejection_count": 0,
            "positive_centered_realized_score_count": 0,
            "saturation_event_count": 0,
        },
        "rank_margin": {
            "minimum_realized_margin_q12_20_lsb": 1 + ordinal if passes else -(1 + ordinal),
            "preserved_positive_margin_fraction": {"denominator": rows, "numerator": preserved},
            "preserved_positive_margin_row_count": preserved,
            "unique_oracle_top_row_count": rows,
            "violation_count": 0 if passes else 1,
        },
        "score_error": {
            "maximum_absolute_error_q12_20_lsb": 16 + ordinal,
            "sum_absolute_error_q12_20_lsb": 32 + ordinal,
            "sum_signed_error_q12_20_lsb": ordinal - 2,
            "sum_squared_error_q40_40_lsb2": 64 + ordinal,
            "valid_value_count": 64,
        },
        "top_key": {
            "matching_fraction": {"denominator": rows, "numerator": matching},
            "matching_row_count": matching,
            "mismatch_count": rows - matching,
            "row_count": rows,
        },
    }


def build_synthetic_result(
    *,
    package_sha256: str,
    evaluator_sha256: str,
    first_passing_candidate: str | None,
) -> dict[str, Any]:
    labels = [candidate["label"] for candidate in CANDIDATES]
    if first_passing_candidate is not None and first_passing_candidate not in labels:
        raise ValueError("unknown synthetic passing candidate")
    first_pass_index = None if first_passing_candidate is None else labels.index(first_passing_candidate)
    candidate_results = []
    passing_candidates = []
    for index, candidate in enumerate(CANDIDATES):
        passes = first_pass_index is not None and index >= first_pass_index
        metrics = _metrics(passes, index)
        thresholds = _thresholds(metrics)
        item = dict(candidate)
        item.update({"metrics": metrics, "threshold_evaluation": thresholds})
        candidate_results.append(item)
        if thresholds["all_hard_gates_pass"]:
            passing_candidates.append(candidate["label"])
    selected = passing_candidates[0] if passing_candidates else None
    succeeded = selected is not None
    result = {
        "authority_sha256": "1" * 64,
        "candidate_results": candidate_results,
        "consumed_ledger_sha256": "2" * 64,
        "evaluator_sha256": evaluator_sha256,
        "fresh_l2_acceptance_sha256": "3" * 64,
        "generation_id": GENERATION_ID,
        "input_bindings": {
            "sealed_set_id": SEALED_SET_ID,
            "tensor_bundle_sha256": "4" * 64,
            "tensor_record_count": 25,
        },
        "invocation_sha256": "5" * 64,
        "irreversible_action_id": "ace2:v7:synthetic-schema-validation:0001",
        "lane_label": "Base",
        "model_identity_sha256": "6" * 64,
        "namespace_label": "base",
        "package_id": PACKAGE_ID,
        "package_sha256": package_sha256,
        "schema_id": RESULT_SCHEMA_ID,
        "selected_candidate": selected,
        "selection": {
            "passing_candidates": passing_candidates,
            "policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER",
            "selected_candidate": selected,
        },
        "terminal": {
            "first_record_immutable": True,
            "invocation_count_performed": 1,
            "metrics_published": True,
            "reason_code": "HARD_THRESHOLDS_PASSED" if succeeded else "HARD_THRESHOLD_FAILED",
            "retry_replay_resume_repair_permitted": False,
            "status": "SUCCEEDED_TERMINAL" if succeeded else "FAILED_TERMINAL",
            "tensor_open_count": 1,
            "thresholds_evaluated": True,
        },
    }
    return seal_result(result)


def main() -> int:
    raise RuntimeError("STATIC_ONLY: evaluator execution requires a separate irreversible action ID")


if __name__ == "__main__":
    raise SystemExit(main())
