#!/usr/bin/env python3
"""Deterministic G16 Q/K diagnostic V5 reference for a later authorized run.

Importing this module is synthetic-only.  The CLI refuses to open a tensor
bundle until a separate lane-specific cardinality-one authority has been
validated and consumed into a create-only ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import struct
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_REL = "design/QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_CONTRACT.json"
PACKAGE_REL = "reference/QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_PACKAGE.json"
SCHEMA_REL = "reference/QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_SCHEMA.json"
G16_REFERENCE_REL = "reference/qk_grouped_bfp8_head64_successor_static_v1.py"
OUTPUT_ROOT_REL = "build/qk-gbfp8-g16-head64-successor-diagnostic-v5"
AUTHORITY_ROOT_REL = "build/qk-gbfp8-g16-head64-successor-diagnostic-v5-authority"
PACKAGE_ID = "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_STATIC_V1_PACKAGE"
RESULT_SCHEMA_ID = "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_RESULT_V1"
ALTERNATIVE_ID = "QK_GBFP8_G16_E16_HEAD64_V1"
TENSOR_BUNDLE_MAGIC = b"ACE2-C02-TENSORS-V1\n"
EXACT_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}
SELECTED_ROLES = (
    "bf16_oracle_scores",
    "realized_key_source",
    "realized_query_source",
)
PRE_METRIC_FAILURE_REASON_CODES = frozenset(
    {
        "INPUT_BINDING_REJECTED",
        "INPUT_SCHEMA_REJECTED",
        "NON_FINITE_INPUT_REJECTED",
        "NORMALIZATION_REJECTED",
        "REPRESENTATION_REJECTED",
        "RESULT_SCHEMA_REJECTED",
    }
)
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

sys.dont_write_bytecode = True


class EvaluationError(RuntimeError):
    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


class PublicationError(RuntimeError):
    reason_code = "OUTPUT_PUBLICATION_REJECTED"


def require(condition: bool, message: str, reason: str = "INPUT_SCHEMA_REJECTED") -> None:
    if not condition:
        raise EvaluationError(reason, message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    payload = path.read_bytes()
    return json.loads(payload.decode("ascii"))


def load_canonical_json(path: Path) -> Any:
    payload = path.read_bytes()
    value = json.loads(payload.decode("ascii"))
    require(canonical_json_bytes(value) == payload, f"noncanonical JSON: {path}")
    return value


def exact_keys(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    require(type(value) is dict and set(value) == keys, f"{context} exact keys")
    return value


def lowercase_sha(value: Any, context: str) -> str:
    require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{context} SHA-256",
    )
    return value


def exact_int(value: Any, context: str) -> int:
    require(type(value) is int, f"{context} exact integer")
    return value


def import_g16_reference(expected_sha256: str) -> Any:
    path = ROOT / G16_REFERENCE_REL
    require(sha256_file(path) == expected_sha256, "G16 reference hash", "INPUT_BINDING_REJECTED")
    spec = importlib.util.spec_from_file_location("accepted_g16_reference", path)
    require(spec is not None and spec.loader is not None, "G16 reference import", "INPUT_BINDING_REJECTED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_record_sha256(dtype: str, shape: tuple[int, ...], payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack(">I", len(shape)))
    for dimension in shape:
        digest.update(struct.pack(">Q", dimension))
    digest.update(payload)
    return digest.hexdigest()


def read_exact(handle: Any, size: int) -> bytes:
    data = handle.read(size)
    require(len(data) == size, "truncated tensor container")
    return data


def read_tensor_bundle(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    verify_plain_path(path)
    require(path.is_file() and not path.is_symlink(), "tensor bundle plain file", "INPUT_BINDING_REJECTED")
    with path.open("rb") as handle:
        require(read_exact(handle, len(TENSOR_BUNDLE_MAGIC)) == TENSOR_BUNDLE_MAGIC, "tensor magic")
        count = struct.unpack(">I", read_exact(handle, 4))[0]
        require(1 <= count <= 256, "tensor record count")
        for _ in range(count):
            name_size = struct.unpack(">H", read_exact(handle, 2))[0]
            require(1 <= name_size <= 1024, "tensor name size")
            name = read_exact(handle, name_size).decode("utf-8")
            require(name not in records, "duplicate tensor name")
            dtype_size = read_exact(handle, 1)[0]
            require(1 <= dtype_size <= 64, "tensor dtype size")
            dtype = read_exact(handle, dtype_size).decode("ascii")
            rank = read_exact(handle, 1)[0]
            require(1 <= rank <= 8, "tensor rank")
            shape = tuple(struct.unpack(">Q", read_exact(handle, 8))[0] for _ in range(rank))
            require(all(1 <= dimension <= (1 << 20) for dimension in shape), "tensor shape")
            payload_size = struct.unpack(">Q", read_exact(handle, 8))[0]
            require(payload_size <= (1 << 31), "tensor payload size")
            payload = read_exact(handle, payload_size)
            records[name] = {
                "dtype": dtype,
                "payload": payload,
                "sha256": tensor_record_sha256(dtype, shape, payload),
                "shape": shape,
            }
        require(handle.read(1) == b"", "trailing tensor container bytes")
    return records


def validate_selected_records(
    records: dict[str, dict[str, Any]], lane: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    bindings = lane["authoritative_tensors"]
    require(set(bindings) == set(SELECTED_ROLES), "selected role set", "INPUT_BINDING_REJECTED")
    for role in SELECTED_ROLES:
        binding = bindings[role]
        name = binding["tensor_name"]
        require(name in records, f"missing selected tensor {name}", "INPUT_BINDING_REJECTED")
        record = records[name]
        require(record["dtype"] == binding["dtype"], f"dtype {name}", "INPUT_BINDING_REJECTED")
        require(list(record["shape"]) == binding["shape"], f"shape {name}", "INPUT_BINDING_REJECTED")
        require(record["sha256"] == binding["sha256"], f"record hash {name}", "INPUT_BINDING_REJECTED")
        require(len(record["payload"]) == 2 * math.prod(record["shape"]), f"payload size {name}")
        selected[role] = record
    return selected


def validate_selected_record_finiteness(selected: dict[str, dict[str, Any]]) -> None:
    require(set(selected) == set(SELECTED_ROLES), "selected finite-scan role set")
    for role in SELECTED_ROLES:
        record = selected[role]
        require(record.get("dtype") == "torch.bfloat16", f"selected dtype {role}")
        payload = record.get("payload")
        require(type(payload) is bytes and len(payload) % 2 == 0, f"selected payload {role}")
        for word_index in range(len(payload) // 2):
            word = struct.unpack_from("<H", payload, word_index * 2)[0]
            if ((word >> 7) & 0xFF) == 0xFF:
                raise EvaluationError(
                    "NON_FINITE_INPUT_REJECTED",
                    f"nonfinite BF16 in complete selected {role} record at flat index {word_index}",
                )


def bf16_word(record: dict[str, Any], indices: tuple[int, ...]) -> int:
    shape = record["shape"]
    require(len(indices) == len(shape), "tensor index rank")
    flat = 0
    for index, dimension in zip(indices, shape):
        require(type(index) is int and 0 <= index < dimension, "tensor index range")
        flat = flat * dimension + index
    return struct.unpack_from("<H", record["payload"], flat * 2)[0]


def signed_bf16_dyadic(word: int) -> tuple[int, int]:
    word = exact_int(word, "BF16 word")
    require(0 <= word <= 0xFFFF, "BF16 word range")
    sign = -1 if word & 0x8000 else 1
    exponent_field = (word >> 7) & 0xFF
    fraction = word & 0x7F
    require(exponent_field != 0xFF, "nonfinite BF16", "NON_FINITE_INPUT_REJECTED")
    if exponent_field == 0:
        return (0, 0) if fraction == 0 else (sign * fraction, -133)
    return (sign * (128 + fraction), exponent_field - 134)


def round_signed_pow2(value: int, binary_shift: int) -> int:
    value = exact_int(value, "round value")
    binary_shift = exact_int(binary_shift, "round shift")
    if binary_shift >= 0:
        return value << binary_shift
    magnitude = abs(value)
    divisor = 1 << (-binary_shift)
    quotient, remainder = divmod(magnitude, divisor)
    doubled = remainder * 2
    if doubled > divisor or (doubled == divisor and (quotient & 1)):
        quotient += 1
    return -quotient if value < 0 else quotient


def exact_oracle_row(words: tuple[int, ...]) -> tuple[tuple[int, ...], int, int]:
    require(type(words) is tuple and len(words) >= 1, "oracle row")
    decoded = tuple(signed_bf16_dyadic(word) for word in words)
    nonzero_exponents = [exponent for coefficient, exponent in decoded if coefficient != 0]
    common = min(nonzero_exponents) if nonzero_exponents else 0
    aligned = tuple(
        0 if coefficient == 0 else coefficient << (exponent - common)
        for coefficient, exponent in decoded
    )
    top = max(range(len(aligned)), key=lambda index: (aligned[index], -index))
    maximum = aligned[top]
    tie_count = sum(value == maximum for value in aligned)
    realized = tuple(round_signed_pow2(value - maximum, common + 20) for value in aligned)
    require(all(INT64_MIN <= value <= 0 for value in realized), "oracle Q12.20 range")
    return realized, top, tie_count


def reduced_fraction(numerator: int, denominator: int) -> dict[str, int] | None:
    numerator = exact_int(numerator, "fraction numerator")
    denominator = exact_int(denominator, "fraction denominator")
    if denominator == 0:
        return None
    require(0 <= numerator <= denominator, "fraction range")
    divisor = math.gcd(numerator, denominator)
    return {"denominator": denominator // divisor, "numerator": numerator // divisor}


def evaluate_selected_records(
    selected: dict[str, dict[str, Any]], g16: Any
) -> dict[str, Any]:
    validate_selected_record_finiteness(selected)
    return evaluate_finite_selected_records(selected, g16)


def evaluate_selected_records_from_reference(
    selected: dict[str, dict[str, Any]], expected_g16_sha256: str
) -> dict[str, Any]:
    validate_selected_record_finiteness(selected)
    g16 = import_g16_reference(expected_g16_sha256)
    return evaluate_finite_selected_records(selected, g16)


def evaluate_finite_selected_records(
    selected: dict[str, dict[str, Any]], g16: Any
) -> dict[str, Any]:
    oracle = selected["bf16_oracle_scores"]
    key = selected["realized_key_source"]
    query = selected["realized_query_source"]

    encoded_keys: dict[tuple[int, int], Any] = {}
    try:
        for kv_head in range(2):
            for key_index in range(41):
                words = tuple(bf16_word(key, (0, kv_head, key_index, lane)) for lane in range(64))
                encoded_keys[(kv_head, key_index)] = g16.encode_head_bf16(words, ALTERNATIVE_ID)
    except (ValueError, TypeError, AssertionError) as exc:
        raise EvaluationError("REPRESENTATION_REJECTED", str(exc)) from exc

    row_count = 0
    top_matches = 0
    singleton_rows = 0
    unique_rows = 0
    tied_rows = 0
    preserved_rows = 0
    violations = 0
    minimum_margin: int | None = None
    valid_values = 0
    sum_signed = 0
    sum_absolute = 0
    sum_squared = 0
    maximum_absolute: int | None = None
    saturation_events = 0

    for query_head in range(14):
        kv_head = query_head // 7
        for query_index in range(41):
            query_words = tuple(
                bf16_word(query, (0, query_head, query_index, lane)) for lane in range(64)
            )
            try:
                encoded_query = g16.encode_head_bf16(query_words, ALTERNATIVE_ID)
                keys = tuple(encoded_keys[(kv_head, key_index)] for key_index in range(41))
                valid_mask = tuple(key_index <= query_index for key_index in range(41))
                realized, _common, realized_top, saturations = g16.normalize_score_row(
                    encoded_query, keys, valid_mask, ALTERNATIVE_ID
                )
            except (ValueError, TypeError, AssertionError) as exc:
                raise EvaluationError("NORMALIZATION_REJECTED", str(exc)) from exc
            require(all(realized[index] <= 0 for index in range(query_index + 1)), "positive realized")
            saturation_events += saturations
            oracle_words = tuple(
                bf16_word(oracle, (0, query_head, query_index, key_index))
                for key_index in range(query_index + 1)
            )
            oracle_values, oracle_top, oracle_ties = exact_oracle_row(oracle_words)
            realized_valid = realized[: query_index + 1]
            row_count += 1
            top_matches += int(realized_top == oracle_top)
            if len(realized_valid) == 1:
                singleton_rows += 1
            elif oracle_ties == 1:
                unique_rows += 1
                other_max = max(
                    value for index, value in enumerate(realized_valid) if index != oracle_top
                )
                margin = realized_valid[oracle_top] - other_max
                if minimum_margin is None or margin < minimum_margin:
                    minimum_margin = margin
                if margin > 0:
                    preserved_rows += 1
                else:
                    violations += 1
            else:
                tied_rows += 1
            for observed, expected in zip(realized_valid, oracle_values):
                error = observed - expected
                absolute = abs(error)
                valid_values += 1
                sum_signed += error
                sum_absolute += absolute
                sum_squared += error * error
                if maximum_absolute is None or absolute > maximum_absolute:
                    maximum_absolute = absolute

    return {
        "invalid_accounting": {
            "cross_lane_record_count": 0,
            "invalid_or_non_finite_value_count": 0,
            "normalization_rejection_count": 0,
            "positive_centered_realized_score_count": 0,
            "saturation_event_count": saturation_events,
        },
        "rank_margin": {
            "minimum_realized_margin_q12_20_lsb": minimum_margin,
            "oracle_tied_row_count": tied_rows,
            "preserved_positive_margin_fraction": reduced_fraction(preserved_rows, unique_rows),
            "preserved_positive_margin_row_count": preserved_rows,
            "singleton_valid_key_row_count": singleton_rows,
            "unique_oracle_top_row_count": unique_rows,
            "violation_count": violations,
        },
        "score_error": {
            "maximum_absolute_error_q12_20_lsb": maximum_absolute,
            "sum_absolute_error_q12_20_lsb": sum_absolute,
            "sum_signed_error_q12_20_lsb": sum_signed,
            "sum_squared_error_q40_40_lsb2": sum_squared,
            "valid_value_count": valid_values,
        },
        "top_key": {
            "matching_fraction": reduced_fraction(top_matches, row_count),
            "matching_row_count": top_matches,
            "mismatch_count": row_count - top_matches,
            "row_count": row_count,
        },
    }


def integer_maximum(actual: int, limit: int) -> dict[str, Any]:
    return {"actual": actual, "limit": limit, "pass": actual <= limit}


def fraction_minimum(
    actual: dict[str, int] | None, limit: dict[str, int]
) -> dict[str, Any]:
    passed = actual is not None and (
        actual["numerator"] * limit["denominator"]
        >= limit["numerator"] * actual["denominator"]
    )
    return {"actual": actual, "limit": limit, "pass": passed}


def evaluate_thresholds(metrics: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    invalid = metrics["invalid_accounting"]
    rank = metrics["rank_margin"]
    top = metrics["top_key"]
    result = {
        "cross_lane_record_count_maximum": integer_maximum(
            invalid["cross_lane_record_count"], limits["cross_lane_record_count_maximum"]
        ),
        "invalid_or_non_finite_value_count_maximum": integer_maximum(
            invalid["invalid_or_non_finite_value_count"],
            limits["invalid_or_non_finite_value_count_maximum"],
        ),
        "normalization_rejection_count_maximum": integer_maximum(
            invalid["normalization_rejection_count"], limits["normalization_rejection_count_maximum"]
        ),
        "positive_centered_realized_score_count_maximum": integer_maximum(
            invalid["positive_centered_realized_score_count"],
            limits["positive_centered_realized_score_count_maximum"],
        ),
        "rank_margin_violation_count_maximum": integer_maximum(
            rank["violation_count"], limits["rank_margin_violation_count_maximum"]
        ),
        "saturation_event_count_maximum": integer_maximum(
            invalid["saturation_event_count"], limits["saturation_event_count_maximum"]
        ),
        "top_key_matching_fraction_minimum": fraction_minimum(
            top["matching_fraction"], limits["top_key_matching_fraction_minimum"]
        ),
        "top_key_mismatch_count_maximum": integer_maximum(
            top["mismatch_count"], limits["top_key_mismatch_count_maximum"]
        ),
        "unique_oracle_positive_margin_preserved_fraction_minimum": fraction_minimum(
            rank["preserved_positive_margin_fraction"],
            limits["unique_oracle_positive_margin_preserved_fraction_minimum"],
        ),
    }
    result["all_hard_thresholds_pass"] = all(check["pass"] for check in result.values())
    return result


def lane_input_bindings(lane: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_token_ids_sha256": lane["input_token_ids_sha256"],
        "lane_metadata": lane["metadata"],
        "sealed_set_id": contract["input_set"]["sealed_set_id"],
        "tensor_bundle": lane["tensor_bundle"],
        "tensor_records": lane["authoritative_tensors"],
    }


def result_object(
    lane: dict[str, Any],
    contract: dict[str, Any],
    package_sha256: str,
    evaluator_sha256: str,
    authority_sha256: str,
    invocation_sha256: str,
    metrics: dict[str, Any] | None,
    thresholds: dict[str, Any] | None,
    reason_code: str,
) -> dict[str, Any]:
    succeeded = thresholds is not None and thresholds["all_hard_thresholds_pass"]
    result = {
        "authority_sha256": authority_sha256,
        "evaluator_sha256": evaluator_sha256,
        "generation_id": lane["generation_id"],
        "input_bindings": lane_input_bindings(lane, contract),
        "invocation_sha256": invocation_sha256,
        "lane_label": lane["label"],
        "metrics": metrics,
        "model_identity_sha256": lane["model_identity_sha256"],
        "namespace_label": lane["namespace_label"],
        "package_id": PACKAGE_ID,
        "package_sha256": package_sha256,
        "schema_id": RESULT_SCHEMA_ID,
        "terminal": {
            "first_record_immutable": True,
            "invocation_count_performed": 1,
            "metrics_published": metrics is not None,
            "reason_code": reason_code,
            "retry_replay_resume_repair_permitted": False,
            "status": "SUCCEEDED_TERMINAL" if succeeded else "FAILED_TERMINAL",
            "thresholds_evaluated": thresholds is not None,
        },
        "threshold_evaluation": thresholds,
    }
    result["result_sha256"] = object_sha256(result)
    return result


def validate_metrics(metrics: dict[str, Any]) -> None:
    exact_keys(metrics, {"invalid_accounting", "rank_margin", "score_error", "top_key"}, "metrics")
    top = exact_keys(
        metrics["top_key"],
        {"matching_fraction", "matching_row_count", "mismatch_count", "row_count"},
        "top_key",
    )
    rank = exact_keys(
        metrics["rank_margin"],
        {
            "minimum_realized_margin_q12_20_lsb",
            "oracle_tied_row_count",
            "preserved_positive_margin_fraction",
            "preserved_positive_margin_row_count",
            "singleton_valid_key_row_count",
            "unique_oracle_top_row_count",
            "violation_count",
        },
        "rank_margin",
    )
    score = exact_keys(
        metrics["score_error"],
        {
            "maximum_absolute_error_q12_20_lsb",
            "sum_absolute_error_q12_20_lsb",
            "sum_signed_error_q12_20_lsb",
            "sum_squared_error_q40_40_lsb2",
            "valid_value_count",
        },
        "score_error",
    )
    invalid = exact_keys(
        metrics["invalid_accounting"],
        {
            "cross_lane_record_count",
            "invalid_or_non_finite_value_count",
            "normalization_rejection_count",
            "positive_centered_realized_score_count",
            "saturation_event_count",
        },
        "invalid_accounting",
    )
    require(top["matching_row_count"] + top["mismatch_count"] == top["row_count"], "top population")
    require(
        rank["singleton_valid_key_row_count"]
        + rank["unique_oracle_top_row_count"]
        + rank["oracle_tied_row_count"]
        == top["row_count"],
        "rank population",
    )
    require(
        rank["preserved_positive_margin_row_count"] + rank["violation_count"]
        == rank["unique_oracle_top_row_count"],
        "margin population",
    )
    require((rank["minimum_realized_margin_q12_20_lsb"] is None) == (rank["unique_oracle_top_row_count"] == 0), "margin nullability")
    require((rank["preserved_positive_margin_fraction"] is None) == (rank["unique_oracle_top_row_count"] == 0), "margin fraction nullability")
    require((score["maximum_absolute_error_q12_20_lsb"] is None) == (score["valid_value_count"] == 0), "score nullability")
    require(all(type(value) is int and value >= 0 for value in invalid.values()), "invalid counts")


def validate_fraction(value: Any, context: str, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    fraction = exact_keys(value, {"denominator", "numerator"}, context)
    denominator = exact_int(fraction["denominator"], f"{context}.denominator")
    numerator = exact_int(fraction["numerator"], f"{context}.numerator")
    require(denominator > 0 and 0 <= numerator <= denominator, f"{context} range")
    require(math.gcd(numerator, denominator) == 1, f"{context} reduced")


def validate_threshold_evaluation(value: dict[str, Any]) -> None:
    fields = {
        "cross_lane_record_count_maximum",
        "invalid_or_non_finite_value_count_maximum",
        "normalization_rejection_count_maximum",
        "positive_centered_realized_score_count_maximum",
        "rank_margin_violation_count_maximum",
        "saturation_event_count_maximum",
        "top_key_matching_fraction_minimum",
        "top_key_mismatch_count_maximum",
        "unique_oracle_positive_margin_preserved_fraction_minimum",
    }
    checks = exact_keys(value, fields | {"all_hard_thresholds_pass"}, "threshold_evaluation")
    for name in fields:
        check = exact_keys(checks[name], {"actual", "limit", "pass"}, name)
        if name.endswith("fraction_minimum"):
            validate_fraction(check["actual"], f"{name}.actual", nullable=True)
            validate_fraction(check["limit"], f"{name}.limit")
            expected = check["actual"] is not None and (
                check["actual"]["numerator"] * check["limit"]["denominator"]
                >= check["limit"]["numerator"] * check["actual"]["denominator"]
            )
        else:
            actual = exact_int(check["actual"], f"{name}.actual")
            limit = exact_int(check["limit"], f"{name}.limit")
            require(actual >= 0 and limit >= 0, f"{name} range")
            expected = actual <= limit
        require(type(check["pass"]) is bool and check["pass"] == expected, f"{name} pass")
    require(
        type(checks["all_hard_thresholds_pass"]) is bool
        and checks["all_hard_thresholds_pass"] == all(checks[name]["pass"] for name in fields),
        "all hard thresholds",
    )


def _validate_result(result: dict[str, Any]) -> None:
    expected_keys = {
        "authority_sha256",
        "evaluator_sha256",
        "generation_id",
        "input_bindings",
        "invocation_sha256",
        "lane_label",
        "metrics",
        "model_identity_sha256",
        "namespace_label",
        "package_id",
        "package_sha256",
        "result_sha256",
        "schema_id",
        "terminal",
        "threshold_evaluation",
    }
    exact_keys(result, expected_keys, "result")
    require(result["package_id"] == PACKAGE_ID and result["schema_id"] == RESULT_SCHEMA_ID, "result identity")
    for field in ("authority_sha256", "evaluator_sha256", "invocation_sha256", "model_identity_sha256", "package_sha256", "result_sha256"):
        lowercase_sha(result[field], field)
    terminal = exact_keys(
        result["terminal"],
        {
            "first_record_immutable",
            "invocation_count_performed",
            "metrics_published",
            "reason_code",
            "retry_replay_resume_repair_permitted",
            "status",
            "thresholds_evaluated",
        },
        "terminal",
    )
    require(terminal["first_record_immutable"] is True, "terminal immutability")
    require(terminal["retry_replay_resume_repair_permitted"] is False, "terminal replay")
    require(terminal["invocation_count_performed"] in (0, 1), "terminal invocation count")
    if result["metrics"] is not None:
        validate_metrics(result["metrics"])
    if result["threshold_evaluation"] is not None:
        validate_threshold_evaluation(result["threshold_evaluation"])
    require(terminal["metrics_published"] == (result["metrics"] is not None), "metrics publication")
    require(terminal["thresholds_evaluated"] == (result["threshold_evaluation"] is not None), "threshold publication")
    if terminal["status"] == "SUCCEEDED_TERMINAL":
        require(result["metrics"] is not None and result["threshold_evaluation"] is not None, "success payload")
        require(result["threshold_evaluation"]["all_hard_thresholds_pass"] is True, "success thresholds")
        require(terminal["reason_code"] == "HARD_THRESHOLDS_PASSED", "success reason")
        require(terminal["invocation_count_performed"] == 1, "success count")
    elif terminal["status"] == "NO_EXECUTION_TERMINAL":
        require(terminal["invocation_count_performed"] == 0, "no-execution count")
        require(result["metrics"] is None and result["threshold_evaluation"] is None, "no-execution payload")
        require(terminal["reason_code"] == "NO_EXECUTION_AUTHORITY", "no-execution reason")
    else:
        require(terminal["status"] == "FAILED_TERMINAL", "terminal status")
        require(terminal["invocation_count_performed"] == 1, "failure count")
        if result["threshold_evaluation"] is not None:
            require(result["metrics"] is not None, "threshold failure metrics")
            require(result["threshold_evaluation"]["all_hard_thresholds_pass"] is False, "failure thresholds")
            require(terminal["reason_code"] == "HARD_THRESHOLD_FAILED", "threshold failure reason")
        else:
            require(result["metrics"] is None, "pre-metric failure payload")
            require(
                terminal["reason_code"] in PRE_METRIC_FAILURE_REASON_CODES,
                "pre-metric failure reason",
            )
    copy = dict(result)
    expected_hash = copy.pop("result_sha256")
    require(object_sha256(copy) == expected_hash, "result self hash")


def validate_result(result: dict[str, Any]) -> None:
    try:
        _validate_result(result)
    except EvaluationError as exc:
        raise EvaluationError("RESULT_SCHEMA_REJECTED", str(exc)) from exc


def authorized_result(
    lane: dict[str, Any],
    contract: dict[str, Any],
    package_sha256: str,
    evaluator_sha256: str,
    authority_sha256: str,
    invocation_sha256: str,
    evaluate: Callable[[], tuple[dict[str, Any], dict[str, Any], str]],
) -> dict[str, Any]:
    try:
        try:
            metrics, thresholds, reason = evaluate()
        except EvaluationError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, struct.error) as exc:
            raise EvaluationError("INPUT_BINDING_REJECTED", str(exc)) from exc
        result = result_object(
            lane,
            contract,
            package_sha256,
            evaluator_sha256,
            authority_sha256,
            invocation_sha256,
            metrics,
            thresholds,
            reason,
        )
        validate_result(result)
    except EvaluationError as exc:
        if exc.reason_code not in PRE_METRIC_FAILURE_REASON_CODES:
            raise
        result = result_object(
            lane,
            contract,
            package_sha256,
            evaluator_sha256,
            authority_sha256,
            invocation_sha256,
            None,
            None,
            exc.reason_code,
        )
        _validate_result(result)
    return result


def selected_lane(contract: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [lane for lane in contract["input_set"]["lanes"] if lane["label"] == label]
    require(len(matches) == 1, "selected lane", "CROSS_LANE_REJECTED")
    return matches[0]


def selected_invocation(contract: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [item for item in contract["invocations"] if item["lane_label"] == label]
    require(len(matches) == 1, "selected invocation", "CROSS_LANE_REJECTED")
    return matches[0]


def verify_plain_path(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path(".")
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            require(not current.is_symlink(), f"symlink component {current}", "INPUT_BINDING_REJECTED")


def resolve_argv_artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    raw_paths = {
        "package": args.package,
        "authority": args.authority,
        "ledger": args.ledger,
        "metadata": args.metadata,
        "tensor_bundle": args.tensor_bundle,
        "output": args.output,
    }
    resolved: dict[str, Path] = {}
    for name, raw_path in raw_paths.items():
        require(type(raw_path) is str and raw_path != "", f"{name} argv path", "INPUT_BINDING_REJECTED")
        require("\\" not in raw_path and "\x00" not in raw_path, f"{name} plain argv path", "INPUT_BINDING_REJECTED")
        path = Path(raw_path)
        require(not path.is_absolute(), "argv paths must be repository-relative", "INPUT_BINDING_REJECTED")
        require(
            path.as_posix() == raw_path
            and path.parts
            and all(part not in ("", ".", "..") for part in path.parts),
            f"{name} plain repository-relative argv path",
            "INPUT_BINDING_REJECTED",
        )
        artifact_path = ROOT / path
        verify_plain_path(artifact_path)
        resolved[name] = artifact_path
    require(args.package == PACKAGE_REL, "package argv identity", "INPUT_BINDING_REJECTED")
    return resolved


def write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(descriptor, payload[offset:])
        except InterruptedError:
            continue
        if written <= 0 or written > len(payload) - offset:
            raise OSError("invalid zero-length or overlong write")
        offset += written


def best_effort_unlink(path: Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def create_only_commit(path: Path, payload: bytes) -> None:
    staging_path = path.with_name(f".{path.name}.incomplete")
    descriptor: int | None = None
    directory_descriptor: int | None = None
    staging_created = False
    final_linked = False
    try:
        descriptor = os.open(
            staging_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
        )
        staging_created = True
        write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None

        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        os.link(staging_path, path, follow_symlinks=False)
        final_linked = True
        os.unlink(staging_path)
        staging_created = False
        os.fsync(directory_descriptor)
    except BaseException:
        if final_linked:
            best_effort_unlink(path)
        if staging_created:
            best_effort_unlink(staging_path)
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if directory_descriptor is not None:
            try:
                os.close(directory_descriptor)
            except OSError:
                pass


def consume_authority(
    authority_path: Path,
    ledger_path: Path,
    authority: dict[str, Any],
    authority_sha256: str,
) -> None:
    require(not ledger_path.exists() and not ledger_path.is_symlink(), "authority ledger collision", "INPUT_BINDING_REJECTED")
    require(authority_path.parent == ledger_path.parent, "authority and ledger namespace", "INPUT_BINDING_REJECTED")
    require(authority_path.is_file() and not authority_path.is_symlink(), "authority file", "INPUT_BINDING_REJECTED")
    require(
        {entry.name for entry in authority_path.parent.iterdir()} == {authority_path.name},
        "authority namespace contents",
        "INPUT_BINDING_REJECTED",
    )
    ledger = {
        "authority_id": authority["authority_id"],
        "authority_sha256": authority_sha256,
        "invocation_sha256": authority["invocation_sha256"],
        "lane_label": authority["lane_label"],
        "schema_id": "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_AUTHORITY_LEDGER_V1",
        "state": "CONSUMED",
    }
    create_only_commit(ledger_path, canonical_json_bytes(ledger))


def publication_require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def prepare_output(output: Path, lane: dict[str, Any]) -> None:
    expected = ROOT / lane["output_path"]
    publication_require(output == expected, "output identity")
    try:
        verify_plain_path(output)
    except EvaluationError as exc:
        raise PublicationError(str(exc)) from exc
    publication_require(not output.exists() and not output.is_symlink(), "output collision")
    publication_require(not output.parent.exists(), "result namespace collision")
    try:
        output.parent.mkdir(parents=True, mode=0o700)
    except OSError as exc:
        raise PublicationError(str(exc)) from exc


def exclusive_publish(output: Path, result: dict[str, Any]) -> None:
    payload = canonical_json_bytes(result)
    try:
        create_only_commit(output, payload)
    except OSError as exc:
        raise PublicationError(str(exc)) from exc


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(allow_abbrev=False)
    value.add_argument("--package", required=True)
    value.add_argument("--authority", required=True)
    value.add_argument("--ledger", required=True)
    value.add_argument("--lane", required=True, choices=("Base", "checkpoint-176"))
    value.add_argument("--metadata", required=True)
    value.add_argument("--tensor-bundle", required=True)
    value.add_argument("--output", required=True)
    return value


def _main() -> int:
    args = parser().parse_args()
    require(Path.cwd() == ROOT, "canonical cwd", "INPUT_BINDING_REJECTED")
    require(dict(os.environ) == EXACT_ENVIRONMENT, "exact environment", "INPUT_BINDING_REJECTED")
    artifact_paths = resolve_argv_artifact_paths(args)
    package_path = artifact_paths["package"]
    authority_path = artifact_paths["authority"]
    ledger_path = artifact_paths["ledger"]
    metadata_path = artifact_paths["metadata"]
    tensor_path = artifact_paths["tensor_bundle"]
    output_path = artifact_paths["output"]

    package = load_json(package_path)
    package_sha256 = sha256_file(package_path)
    bindings = package["artifacts"]
    contract_binding = bindings["contract"]
    schema_binding = bindings["result_schema"]
    evaluator_binding = bindings["reference_evaluator"]
    contract_path = ROOT / contract_binding["path"]
    contract = load_json(contract_path)
    require(sha256_file(contract_path) == contract_binding["sha256"], "contract binding", "INPUT_BINDING_REJECTED")
    require(sha256_file(ROOT / schema_binding["path"]) == schema_binding["sha256"], "schema binding", "INPUT_BINDING_REJECTED")
    evaluator_sha256 = sha256_file(Path(__file__))
    require(evaluator_sha256 == evaluator_binding["sha256"], "evaluator binding", "INPUT_BINDING_REJECTED")

    lane = selected_lane(contract, args.lane)
    invocation = selected_invocation(contract, args.lane)
    actual_argv = [sys.executable, "-I", "-B", *sys.argv]
    require(actual_argv == invocation["argv"], "exact argv", "INPUT_BINDING_REJECTED")
    require(
        sha256_file(Path(sys.executable)) == invocation["interpreter"]["sha256"],
        "interpreter hash",
        "INPUT_BINDING_REJECTED",
    )
    require(
        sys.version.split()[0] == invocation["interpreter"]["version"],
        "interpreter version",
        "INPUT_BINDING_REJECTED",
    )
    require(args.metadata == lane["metadata"]["path"], "metadata argv", "CROSS_LANE_REJECTED")
    require(args.tensor_bundle == lane["tensor_bundle"]["path"], "tensor argv", "CROSS_LANE_REJECTED")
    require(args.output == lane["output_path"], "output argv", "CROSS_LANE_REJECTED")

    authority = load_canonical_json(authority_path)
    authority_sha256 = sha256_file(authority_path)
    exact_keys(
        authority,
        {
            "authority_id",
            "authority_cardinality",
            "evaluator_sha256",
            "invocation_sha256",
            "lane_label",
            "output_path",
            "package_sha256",
            "result_schema_sha256",
            "state",
        },
        "authority",
    )
    require(authority["authority_cardinality"] == 1, "authority cardinality", "INPUT_BINDING_REJECTED")
    require(authority["state"] == "READY_UNCONSUMED", "authority state", "INPUT_BINDING_REJECTED")
    require(authority["lane_label"] == args.lane, "authority lane", "CROSS_LANE_REJECTED")
    require(authority["package_sha256"] == package_sha256, "authority package", "INPUT_BINDING_REJECTED")
    require(authority["evaluator_sha256"] == evaluator_sha256, "authority evaluator", "INPUT_BINDING_REJECTED")
    require(authority["result_schema_sha256"] == schema_binding["sha256"], "authority schema", "INPUT_BINDING_REJECTED")
    require(authority["invocation_sha256"] == invocation["invocation_sha256"], "authority invocation", "INPUT_BINDING_REJECTED")
    require(authority["output_path"] == lane["output_path"], "authority output", "CROSS_LANE_REJECTED")
    prepare_output(output_path, lane)
    consume_authority(authority_path, ledger_path, authority, authority_sha256)

    def evaluate() -> tuple[dict[str, Any], dict[str, Any], str]:
        require(metadata_path.stat().st_size == lane["metadata"]["byte_count"], "metadata bytes", "INPUT_BINDING_REJECTED")
        require(sha256_file(metadata_path) == lane["metadata"]["sha256"], "metadata hash", "INPUT_BINDING_REJECTED")
        require(tensor_path.stat().st_size == lane["tensor_bundle"]["byte_count"], "bundle bytes", "INPUT_BINDING_REJECTED")
        require(sha256_file(tensor_path) == lane["tensor_bundle"]["sha256"], "bundle hash", "INPUT_BINDING_REJECTED")
        records = read_tensor_bundle(tensor_path)
        selected = validate_selected_records(records, lane)
        metrics = evaluate_selected_records_from_reference(
            selected, contract["accepted_g16"]["pure_reference"]["sha256"]
        )
        thresholds = evaluate_thresholds(metrics, contract["thresholds"])
        reason = "HARD_THRESHOLDS_PASSED" if thresholds["all_hard_thresholds_pass"] else "HARD_THRESHOLD_FAILED"
        return metrics, thresholds, reason

    result = authorized_result(
        lane,
        contract,
        package_sha256,
        evaluator_sha256,
        authority_sha256,
        invocation["invocation_sha256"],
        evaluate,
    )
    exclusive_publish(output_path, result)
    return 0 if result["terminal"]["status"] == "SUCCEEDED_TERMINAL" else 1


def cli_rejection(error: Exception, reason_code: str) -> None:
    payload = {
        "message": str(error),
        "reason_code": reason_code,
        "result_published": False,
        "schema_id": "QK_GBFP8_G16_HEAD64_SUCCESSOR_DIAGNOSTIC_CLI_REJECTION_V1",
    }
    sys.stderr.buffer.write(canonical_json_bytes(payload))


def main() -> int:
    try:
        return _main()
    except PublicationError as exc:
        cli_rejection(exc, exc.reason_code)
        return 3
    except EvaluationError as exc:
        cli_rejection(exc, exc.reason_code)
        return 2
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        cli_rejection(exc, "INPUT_BINDING_REJECTED")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
