#!/usr/bin/env python3
"""Pure V8 grouped-BFP8 evaluator core over caller-provided validated bytes.

The module has no payload path, filesystem reader, publisher, or execution CLI.
A future caller supplies one bundle-opening callable; the core opens it once,
cross-parses it with both V8 readers, and evaluates G8/G4/G2/G1 together.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from copy import deepcopy
from typing import Any


PACKAGE_ID = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE"
RESULT_SCHEMA_ID = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_RESULT_V1"
GENERATION_ID = "qk-gbfp8-head64-granularity-sweep-v8-base"
SEALED_SET_ID = "w4a8-c02-attention-substage-trace-v2"
OFFICIAL_MODEL_IDENTITY_SHA256 = "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7"
OFFICIAL_INPUT_BINDINGS = {
    "input_token_ids_sha256": "1b8c972381a2c3d7c754d1d2879b4389a13aca3f1d485f703510702c1cf2eb86",
    "lane_metadata": {
        "byte_count": 20057,
        "path": "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json",
        "sha256": "4d5904378b9ccbabd434119b6a57f8d4266abf3d8e772c4c1bca98df785e7f3a",
    },
    "sealed_set_id": SEALED_SET_ID,
    "tensor_bundle": {
        "byte_count": 1305797,
        "path": "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin",
        "sha256": "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175",
    },
    "tensor_records": {
        "bf16_oracle_scores": {
            "dtype": "torch.bfloat16",
            "sha256": "49627e8364e534c61f4db8208d82798e53409c3c10d5d5e28c3c1462ef617765",
            "shape": [1, 14, 41, 41],
            "tensor_name": "bf16.qk_scaled_scores",
        },
        "realized_key_source": {
            "dtype": "torch.bfloat16",
            "sha256": "401cdb0dc4a8def3190ac424f96df272c2bcf11241874759977d692845a19c0a",
            "shape": [1, 2, 41, 64],
            "tensor_name": "bf16.k_rope",
        },
        "realized_query_source": {
            "dtype": "torch.bfloat16",
            "sha256": "285e064ffea9571b7e3ed192a7083bf831dc5d444e0139558d3d51f084995429",
            "shape": [1, 14, 41, 64],
            "tensor_name": "bf16.q_rope",
        },
    },
}
OFFICIAL_SELECTED_TENSOR_BINDINGS = {
    record["tensor_name"]: {
        "dtype": record["dtype"],
        "sha256": record["sha256"],
        "shape": record["shape"],
    }
    for record in OFFICIAL_INPUT_BINDINGS["tensor_records"].values()
}
HEAD_LANES = 64
QUERY_HEADS = 14
KV_HEADS = 2
SEQUENCE_LENGTH = 41
TENSOR_RECORD_COUNT = 25
QUERY_RECORD_NAME = "bf16.q_rope"
KEY_RECORD_NAME = "bf16.k_rope"
ORACLE_RECORD_NAME = "bf16.qk_scaled_scores"
FIXED_METRIC_POPULATIONS = {
    "rank_margin_unique_oracle_top_row_count": 346,
    "score_error_valid_value_count": QUERY_HEADS * sum(range(1, SEQUENCE_LENGTH + 1)),
    "top_key_row_count": QUERY_HEADS * SEQUENCE_LENGTH,
}
MANTISSA_MIN = -127
MANTISSA_MAX = 127
RESERVED_MANTISSA = -128
MIN_INT16 = -(1 << 15)
MAX_INT16 = (1 << 15) - 1
MIN_INT64 = -(1 << 63)
MAX_INT64 = (1 << 63) - 1
ACTION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,255}$"
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
RESULT_CONTEXT_KEYS = {
    "authority_sha256",
    "consumed_ledger_sha256",
    "evaluator_sha256",
    "fresh_l2_acceptance_sha256",
    "input_bindings",
    "invocation_sha256",
    "irreversible_action_id",
    "model_identity_sha256",
    "package_sha256",
}
CANDIDATE_RESULT_KEYS = {
    "bytes_per_head",
    "exponent_bytes_per_head",
    "group_count",
    "group_size",
    "label",
    "mantissa_bytes_per_head",
    "metrics",
    "threshold_evaluation",
}
METRICS_KEYS = {"invalid_accounting", "rank_margin", "score_error", "top_key"}
INVALID_KEYS = {
    "cross_lane_record_count",
    "invalid_or_non_finite_value_count",
    "normalization_rejection_count",
    "positive_centered_realized_score_count",
    "saturation_event_count",
}
RANK_KEYS = {
    "minimum_realized_margin_q12_20_lsb",
    "preserved_positive_margin_fraction",
    "preserved_positive_margin_row_count",
    "unique_oracle_top_row_count",
    "violation_count",
}
SCORE_ERROR_KEYS = {
    "maximum_absolute_error_q12_20_lsb",
    "sum_absolute_error_q12_20_lsb",
    "sum_signed_error_q12_20_lsb",
    "sum_squared_error_q40_40_lsb2",
    "valid_value_count",
}
TOP_KEY_KEYS = {"matching_fraction", "matching_row_count", "mismatch_count", "row_count"}
THRESHOLD_KEYS = set(HARD_GATES) | {"all_hard_gates_pass"}
CHECK_KEYS = {"actual", "limit", "pass"}


class EvaluationError(ValueError):
    """A caller-provided static fixture violates the frozen V8 contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    require(type(value) is dict and set(value) == expected, f"{context} exact keys")
    return value


def valid_sha256(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _nonnegative_integer(value: Any, context: str) -> int:
    require(type(value) is int and value >= 0, f"{context} nonnegative integer")
    return value


def _fraction(value: Any, context: str) -> dict[str, int]:
    item = exact_keys(value, {"denominator", "numerator"}, context)
    numerator = _nonnegative_integer(item["numerator"], f"{context}.numerator")
    denominator = item["denominator"]
    require(type(denominator) is int and denominator > 0, f"{context}.denominator")
    require(numerator <= denominator, f"{context} range")
    return item


def _validate_metrics(metrics: Any, context: str) -> dict[str, Any]:
    value = exact_keys(metrics, METRICS_KEYS, context)
    invalid = exact_keys(value["invalid_accounting"], INVALID_KEYS, f"{context}.invalid_accounting")
    for key in INVALID_KEYS:
        _nonnegative_integer(invalid[key], f"{context}.invalid_accounting.{key}")

    rank = exact_keys(value["rank_margin"], RANK_KEYS, f"{context}.rank_margin")
    minimum_margin = rank["minimum_realized_margin_q12_20_lsb"]
    require(type(minimum_margin) is int, f"{context}.rank minimum")
    fraction = _fraction(rank["preserved_positive_margin_fraction"], f"{context}.rank preserved fraction")
    preserved = _nonnegative_integer(rank["preserved_positive_margin_row_count"], f"{context}.rank preserved rows")
    unique = _nonnegative_integer(rank["unique_oracle_top_row_count"], f"{context}.rank unique rows")
    violations = _nonnegative_integer(rank["violation_count"], f"{context}.rank violations")
    require(
        unique == FIXED_METRIC_POPULATIONS["rank_margin_unique_oracle_top_row_count"],
        f"{context}.rank fixed population",
    )
    require(preserved + violations == unique, f"{context}.rank accounting")
    require(fraction == {"numerator": preserved, "denominator": unique}, f"{context}.rank fraction accounting")
    require((violations == 0) == (minimum_margin > 0), f"{context}.rank minimum accounting")

    score = exact_keys(value["score_error"], SCORE_ERROR_KEYS, f"{context}.score_error")
    maximum = _nonnegative_integer(score["maximum_absolute_error_q12_20_lsb"], f"{context}.score maximum")
    sum_absolute = _nonnegative_integer(score["sum_absolute_error_q12_20_lsb"], f"{context}.score absolute sum")
    sum_signed = score["sum_signed_error_q12_20_lsb"]
    require(type(sum_signed) is int, f"{context}.score signed error")
    sum_squared = _nonnegative_integer(score["sum_squared_error_q40_40_lsb2"], f"{context}.score squared sum")
    valid_values = _nonnegative_integer(score["valid_value_count"], f"{context}.score valid values")
    require(
        valid_values == FIXED_METRIC_POPULATIONS["score_error_valid_value_count"],
        f"{context}.score fixed population",
    )
    require(abs(sum_signed) <= sum_absolute, f"{context}.score signed/absolute accounting")
    require(maximum <= sum_absolute <= valid_values * maximum, f"{context}.score maximum/absolute accounting")
    require(maximum * maximum <= sum_squared <= maximum * sum_absolute, f"{context}.score squared accounting")
    require(sum_absolute * sum_absolute <= valid_values * sum_squared, f"{context}.score Cauchy accounting")

    top = exact_keys(value["top_key"], TOP_KEY_KEYS, f"{context}.top_key")
    top_fraction = _fraction(top["matching_fraction"], f"{context}.top matching fraction")
    matching = _nonnegative_integer(top["matching_row_count"], f"{context}.top matching rows")
    mismatches = _nonnegative_integer(top["mismatch_count"], f"{context}.top mismatches")
    rows = _nonnegative_integer(top["row_count"], f"{context}.top rows")
    require(rows == FIXED_METRIC_POPULATIONS["top_key_row_count"], f"{context}.top fixed population")
    require(matching + mismatches == rows, f"{context}.top accounting")
    require(top_fraction == {"numerator": matching, "denominator": rows}, f"{context}.top fraction accounting")
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
            require(
                candidate == MIN_INT16
                or any(
                    _round_unsigned_power_of_two(coefficient, source_exponent - (candidate - 1)) > MANTISSA_MAX
                    for coefficient, source_exponent in nonzero
                ),
                "group exponent minimality",
            )
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
        group_mantissas: list[int] = []
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
    require(type(query_head) is int and 0 <= query_head < QUERY_HEADS, "query head index")
    return query_head // 7


def bf16_word(record: dict[str, Any], indices: tuple[int, ...]) -> int:
    shape = record["shape"]
    require(type(indices) is tuple and len(indices) == len(shape), "tensor index rank")
    flat = 0
    for index, dimension in zip(indices, shape):
        require(type(index) is int and 0 <= index < dimension, "tensor index range")
        flat = flat * dimension + index
    return struct.unpack_from("<H", record["payload"], flat * 2)[0]


def _signed_bf16_dyadic(word: int) -> tuple[int, int]:
    negative, coefficient, exponent = _bf16_components(word)
    if negative:
        coefficient = -coefficient
    return (0, 0) if coefficient == 0 else (coefficient, exponent)


def _realize_oracle_q12_20(coefficient: int, exponent: int) -> int:
    shift = exponent + 20
    value = coefficient << shift if shift >= 0 else round_even_signed(coefficient, -shift)
    require(MIN_INT64 <= value <= 0, "oracle Q12.20 overflow or positivity")
    return value


def exact_oracle_row(words: tuple[int, ...]) -> tuple[tuple[int, ...], int, int]:
    require(type(words) is tuple and words, "oracle row")
    dyadics = tuple(_signed_bf16_dyadic(word) for word in words)
    common = min((exponent for coefficient, exponent in dyadics if coefficient), default=0)
    aligned = tuple(0 if coefficient == 0 else coefficient << (exponent - common) for coefficient, exponent in dyadics)
    maximum = max(aligned)
    top = aligned.index(maximum)
    tie_count = sum(value == maximum for value in aligned)
    centered = tuple(value - maximum for value in aligned)
    return tuple(_realize_oracle_q12_20(value, common) for value in centered), top, tie_count


def _reduced_fraction(numerator: int, denominator: int) -> dict[str, int]:
    require(type(numerator) is int and type(denominator) is int and 0 <= numerator <= denominator and denominator > 0, "fraction population")
    return {"denominator": denominator, "numerator": numerator}


def _fraction_at_least(actual: dict[str, int], limit: dict[str, int]) -> bool:
    return actual["numerator"] * limit["denominator"] >= limit["numerator"] * actual["denominator"]


def derive_thresholds(metrics: dict[str, Any]) -> dict[str, Any]:
    _validate_metrics(metrics, "threshold metrics")
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


def _validated_selected_records(records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    require(type(records) is dict and len(records) == TENSOR_RECORD_COUNT, "tensor record count")
    expected = {
        QUERY_RECORD_NAME: (1, QUERY_HEADS, SEQUENCE_LENGTH, HEAD_LANES),
        KEY_RECORD_NAME: (1, KV_HEADS, SEQUENCE_LENGTH, HEAD_LANES),
        ORACLE_RECORD_NAME: (1, QUERY_HEADS, SEQUENCE_LENGTH, SEQUENCE_LENGTH),
    }
    selected: dict[str, dict[str, Any]] = {}
    for name, shape in expected.items():
        require(name in records, f"missing evaluator tensor {name}")
        record = records[name]
        require(record["shape"] == shape, f"evaluator tensor geometry {name}")
        finite_bf16_record_words(record)
        selected[name] = record
    return selected


def evaluate_candidate(records: dict[str, dict[str, Any]], group_size: int) -> dict[str, Any]:
    selected = _validated_selected_records(records)
    query = selected[QUERY_RECORD_NAME]
    key = selected[KEY_RECORD_NAME]
    oracle = selected[ORACLE_RECORD_NAME]

    encoded_keys: dict[tuple[int, int], dict[str, Any]] = {}
    for kv_head in range(KV_HEADS):
        for key_index in range(SEQUENCE_LENGTH):
            words = tuple(bf16_word(key, (0, kv_head, key_index, lane)) for lane in range(HEAD_LANES))
            encoded_keys[(kv_head, key_index)] = encode_grouped_head(words, group_size)

    top_rows = 0
    top_matches = 0
    unique_rows = 0
    preserved_rows = 0
    margins: list[int] = []
    valid_values = 0
    sum_signed = 0
    sum_absolute = 0
    sum_squared = 0
    maximum_absolute = 0

    for query_head in range(QUERY_HEADS):
        kv_head = kv_head_for_query(query_head)
        for query_index in range(SEQUENCE_LENGTH):
            query_words = tuple(
                bf16_word(query, (0, query_head, query_index, lane))
                for lane in range(HEAD_LANES)
            )
            encoded_query = encode_grouped_head(query_words, group_size)
            score_pairs = tuple(
                grouped_dot_score_pair(encoded_query, encoded_keys[(kv_head, key_index)])
                for key_index in range(SEQUENCE_LENGTH)
            )
            valid_mask = tuple(key_index <= query_index for key_index in range(SEQUENCE_LENGTH))
            realized, _, realized_top = normalize_score_row_q12_20(score_pairs, valid_mask)
            oracle_words = tuple(
                bf16_word(oracle, (0, query_head, query_index, key_index))
                for key_index in range(query_index + 1)
            )
            oracle_values, oracle_top, oracle_ties = exact_oracle_row(oracle_words)
            realized_valid = realized[: query_index + 1]
            top_rows += 1
            top_matches += int(realized_top == oracle_top)
            if query_index > 0 and oracle_ties == 1:
                unique_rows += 1
                other_max = max(
                    value for index, value in enumerate(realized_valid) if index != oracle_top
                )
                margin = realized_valid[oracle_top] - other_max
                margins.append(margin)
                preserved_rows += int(margin > 0)
            for observed, expected in zip(realized_valid, oracle_values):
                error = observed - expected
                absolute = abs(error)
                valid_values += 1
                sum_signed += error
                sum_absolute += absolute
                sum_squared += error * error
                maximum_absolute = max(maximum_absolute, absolute)

    require(top_rows == FIXED_METRIC_POPULATIONS["top_key_row_count"], "top row population closure")
    require(valid_values == FIXED_METRIC_POPULATIONS["score_error_valid_value_count"], "valid score population closure")
    require(
        unique_rows == FIXED_METRIC_POPULATIONS["rank_margin_unique_oracle_top_row_count"]
        and len(margins) == unique_rows,
        "unique oracle margin population closure",
    )
    metrics = {
        "invalid_accounting": {
            "cross_lane_record_count": 0,
            "invalid_or_non_finite_value_count": 0,
            "normalization_rejection_count": 0,
            "positive_centered_realized_score_count": 0,
            "saturation_event_count": 0,
        },
        "rank_margin": {
            "minimum_realized_margin_q12_20_lsb": min(margins),
            "preserved_positive_margin_fraction": _reduced_fraction(preserved_rows, unique_rows),
            "preserved_positive_margin_row_count": preserved_rows,
            "unique_oracle_top_row_count": unique_rows,
            "violation_count": unique_rows - preserved_rows,
        },
        "score_error": {
            "maximum_absolute_error_q12_20_lsb": maximum_absolute,
            "sum_absolute_error_q12_20_lsb": sum_absolute,
            "sum_signed_error_q12_20_lsb": sum_signed,
            "sum_squared_error_q40_40_lsb2": sum_squared,
            "valid_value_count": valid_values,
        },
        "top_key": {
            "matching_fraction": _reduced_fraction(top_matches, top_rows),
            "matching_row_count": top_matches,
            "mismatch_count": top_rows - top_matches,
            "row_count": top_rows,
        },
    }
    return _validate_metrics(metrics, "evaluated metrics")


def evaluate_all_candidates(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    _validated_selected_records(records)
    results: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        metrics = evaluate_candidate(records, candidate["group_size"])
        item = dict(candidate)
        item["metrics"] = metrics
        item["threshold_evaluation"] = derive_thresholds(metrics)
        results.append(item)
    return results


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


def build_result(candidate_results: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    exact_keys(context, RESULT_CONTEXT_KEYS, "result context")
    for key in RESULT_CONTEXT_KEYS - {"input_bindings", "irreversible_action_id"}:
        require(valid_sha256(context[key]), f"result context {key}")
    require(context["model_identity_sha256"] == OFFICIAL_MODEL_IDENTITY_SHA256, "result context official model identity")
    require(context["input_bindings"] == OFFICIAL_INPUT_BINDINGS, "result context official input bindings")
    require(
        type(context["irreversible_action_id"]) is str
        and re.fullmatch(ACTION_ID_PATTERN, context["irreversible_action_id"]) is not None,
        "result context action id",
    )
    require(type(candidate_results) is list and len(candidate_results) == len(CANDIDATES), "candidate result count")
    for index, expected in enumerate(CANDIDATES):
        candidate = exact_keys(candidate_results[index], CANDIDATE_RESULT_KEYS, f"candidate result[{index}]")
        for key, expected_value in expected.items():
            require(candidate[key] == expected_value, f"candidate result[{index}] {key}")
        metrics = _validate_metrics(candidate["metrics"], f"candidate result[{index}].metrics")
        thresholds = exact_keys(
            candidate["threshold_evaluation"],
            THRESHOLD_KEYS,
            f"candidate result[{index}].thresholds",
        )
        for name in HARD_GATES:
            exact_keys(thresholds[name], CHECK_KEYS, f"candidate result[{index}].thresholds.{name}")
        require(thresholds == derive_thresholds(metrics), f"candidate result[{index}] threshold derivation")
    passing_candidates = [
        candidate["label"]
        for candidate in candidate_results
        if candidate["threshold_evaluation"]["all_hard_gates_pass"]
    ]
    selected = passing_candidates[0] if passing_candidates else None
    succeeded = selected is not None
    result = {
        "authority_sha256": context["authority_sha256"],
        "candidate_results": candidate_results,
        "consumed_ledger_sha256": context["consumed_ledger_sha256"],
        "evaluator_sha256": context["evaluator_sha256"],
        "fresh_l2_acceptance_sha256": context["fresh_l2_acceptance_sha256"],
        "generation_id": GENERATION_ID,
        "input_bindings": {
            "sealed_set_id": SEALED_SET_ID,
            "tensor_bundle_sha256": context["input_bindings"]["tensor_bundle"]["sha256"],
            "tensor_record_count": TENSOR_RECORD_COUNT,
        },
        "invocation_sha256": context["invocation_sha256"],
        "irreversible_action_id": context["irreversible_action_id"],
        "lane_label": "Base",
        "model_identity_sha256": context["model_identity_sha256"],
        "namespace_label": "base",
        "package_id": PACKAGE_ID,
        "package_sha256": context["package_sha256"],
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


def evaluate_bundle_once(
    open_bundle: Any,
    bindings: dict[str, dict[str, Any]],
    parser_module: Any,
    context: dict[str, Any],
) -> bytes:
    """Open once, cross-parse independently, evaluate all candidates, seal bytes."""

    require(callable(open_bundle), "bundle opener")
    data = open_bundle()
    require(type(data) is bytes, "bundle opener bytes")
    require(
        hashlib.sha256(data).hexdigest() == OFFICIAL_INPUT_BINDINGS["tensor_bundle"]["sha256"],
        "official tensor bundle identity",
    )
    for tensor_name, expected in OFFICIAL_SELECTED_TENSOR_BINDINGS.items():
        require(bindings.get(tensor_name) == expected, f"official tensor record identity {tensor_name}")
    producer_records = parser_module.parse_c02_producer_bytes(data, bindings)
    accepted_records = parser_module.parse_c02_accepted_reader(data, bindings)
    require(producer_records == accepted_records, "c02 cross-parser agreement")
    candidate_results = evaluate_all_candidates(producer_records)
    return compact_bytes(build_result(candidate_results, context))


def main() -> int:
    raise RuntimeError("STATIC_ONLY: evaluator execution requires a separate irreversible action ID")


if __name__ == "__main__":
    raise SystemExit(main())
