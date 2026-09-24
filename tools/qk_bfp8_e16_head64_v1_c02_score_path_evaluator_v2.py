#!/usr/bin/env python3
"""Candidate-free evaluator for the accepted c02 QK score-path contract.

The executable reads the deterministic sealed tensor container directly.  It
does not import model, tokenizer, generation, calibration, or candidate code.
All Q/K encoding and score normalization use the checksum-bound accepted pure
references after their source identities have been verified.
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
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PACKAGE_REL = (
    "reference/QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATION_PACKAGE.json"
)
EVALUATION_PACKAGE_SHA256 = (
    "0bd51407f20950a93f47a520718a3bb4d5a5006d990d60aef626d16a5f9d976d"
)
REPRESENTATION_REL = "reference/qk_bfp8_e16_head64_v1.py"
REPRESENTATION_SHA256 = (
    "bfb8b0d95ef8a8b939694fe8fec710d4acb1abff8b38b4bbbb4a43bdea5bb704"
)
NORMALIZATION_REL = "reference/qk_bfp8_e16_head64_v1_score_normalization.py"
NORMALIZATION_SHA256 = (
    "8b1db0b13ee3a4534848814e75d13cea9946e380cd09b8bec0e76a527df098a3"
)
AMENDMENT_REL = (
    "reference/"
    "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_AUTHORITATIVE_AMENDMENT_V2.json"
)
OUTPUT_ROOT_REL = "build/qk-bfp8-e16-head64-v1-c02-score-path-evaluation-v1"
LIVE_ROOT_REL = "build/qk-bfp8-e16-head64-v1-c02-live-authority-surface-v2"
TENSOR_BUNDLE_MAGIC = b"ACE2-C02-TENSORS-V1\n"
RESULT_SCHEMA_ID = "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_RESULT_V1"
PACKAGE_ID = "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATION_PACKAGE_V1"
EVALUATOR_SPEC_SHA256 = (
    "b719d18f571e664b25c926ecb972f9972f2a13c4b8013d89f8bf970cf8a99756"
)
EXACT_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}
MIN_INT64 = -(1 << 63)
MAX_INT64 = (1 << 63) - 1
SELECTED_RECORD_ROLES = (
    "bf16_oracle_scores",
    "realized_query_source",
    "realized_key_source",
)


class EvaluationError(RuntimeError):
    """A deterministic accepted-contract rejection."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


def require(condition: bool, message: str, reason: str = "INPUT_SCHEMA_REJECTED") -> None:
    if not condition:
        raise EvaluationError(reason, message)


def compact_json_bytes(value: Any) -> bytes:
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


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(compact_json_bytes(value)[:-1]).hexdigest()


def lexical_exists(path: Path) -> bool:
    return os.path.lexists(path)


def verify_no_symlink_components(path: Path) -> None:
    require(path.is_absolute(), f"non-absolute path: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        require(not stat.S_ISLNK(info.st_mode), f"symlinked path component: {current}")


def verify_plain_file(path: Path) -> None:
    verify_no_symlink_components(path)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise EvaluationError("INPUT_BINDING_REJECTED", f"missing file: {path}") from exc
    require(stat.S_ISREG(info.st_mode), f"not a regular file: {path}", "INPUT_BINDING_REJECTED")


def sha256_file(path: Path) -> str:
    verify_plain_file(path)
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def load_pretty_json(path: Path) -> Any:
    verify_plain_file(path)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError("INPUT_SCHEMA_REJECTED", f"invalid JSON: {path}") from exc
    require(raw == pretty_json_bytes(value), f"noncanonical pretty JSON: {path}")
    return value


def load_compact_json(path: Path) -> Any:
    verify_plain_file(path)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError("INPUT_SCHEMA_REJECTED", f"invalid JSON: {path}") from exc
    require(raw == compact_json_bytes(value), f"noncanonical compact JSON: {path}")
    return value


def import_bound_reference(relative: str, expected_sha256: str, module_name: str) -> Any:
    path = ROOT / relative
    require(
        sha256_file(path) == expected_sha256,
        f"bound reference hash mismatch: {relative}",
        "INPUT_BINDING_REJECTED",
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, "reference import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_relative_path(text: Any, expected: str) -> None:
    require(type(text) is str and text == expected, "path binding mismatch", "INPUT_BINDING_REJECTED")
    pure = PurePosixPath(text)
    require(not pure.is_absolute() and ".." not in pure.parts, "path escape", "INPUT_BINDING_REJECTED")


def recursive_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                found.append(child)
            found.extend(recursive_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(recursive_values(child, key))
    return found


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
    """Parse the deterministic tensor container without pickle or torch."""

    verify_plain_file(path)
    records: dict[str, dict[str, Any]] = {}
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


def validate_metadata(metadata: dict[str, Any], lane: dict[str, Any]) -> None:
    require(metadata.get("model_identity_sha256") == lane["model_identity_sha256"], "model identity", "INPUT_BINDING_REJECTED")
    require(metadata.get("status") == "PASS_ATTENTION_SUBSTAGE_LOCALIZED_NON_SCORING", "metadata status", "INPUT_BINDING_REJECTED")
    require(
        recursive_values(metadata, "input_token_ids_sha256") == [lane["input_token_ids_sha256"]],
        "input token identity",
        "INPUT_BINDING_REJECTED",
    )
    bundle = metadata.get("tensor_bundle", {})
    require(bundle.get("sha256") == lane["tensor_bundle"]["sha256"], "metadata bundle hash", "INPUT_BINDING_REJECTED")
    require(bundle.get("bytes") == lane["tensor_bundle"]["byte_count"], "metadata bundle bytes", "INPUT_BINDING_REJECTED")
    tensor_records = bundle.get("tensors")
    require(isinstance(tensor_records, dict), "metadata tensor records", "INPUT_BINDING_REJECTED")
    for binding in lane["authoritative_tensors"].values():
        name = binding["tensor_name"]
        require(
            tensor_records.get(name)
            == {"dtype": binding["dtype"], "shape": binding["shape"], "sha256": binding["sha256"]},
            f"metadata tensor identity: {name}",
            "INPUT_BINDING_REJECTED",
        )


def validate_selected_records(
    records: dict[str, dict[str, Any]], lane: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for role, binding in lane["authoritative_tensors"].items():
        name = binding["tensor_name"]
        require(name in records, f"missing tensor: {name}", "INPUT_BINDING_REJECTED")
        record = records[name]
        require(record["dtype"] == binding["dtype"], f"dtype: {name}", "INPUT_BINDING_REJECTED")
        require(list(record["shape"]) == binding["shape"], f"shape: {name}", "INPUT_BINDING_REJECTED")
        require(record["sha256"] == binding["sha256"], f"tensor hash: {name}", "INPUT_BINDING_REJECTED")
        require(len(record["payload"]) == 2 * math.prod(record["shape"]), f"payload bytes: {name}")
        selected[role] = record
    return selected


def bf16_word(record: dict[str, Any], indices: tuple[int, ...]) -> int:
    shape = record["shape"]
    require(len(indices) == len(shape), "tensor index rank")
    flat = 0
    for index, dimension in zip(indices, shape):
        require(0 <= index < dimension, "tensor index range")
        flat = flat * dimension + index
    return struct.unpack_from("<H", record["payload"], flat * 2)[0]


def validate_selected_record_finiteness(selected: dict[str, dict[str, Any]]) -> None:
    require(
        set(selected) == set(SELECTED_RECORD_ROLES),
        "selected tensor role set",
        "INPUT_BINDING_REJECTED",
    )
    for role in SELECTED_RECORD_ROLES:
        record = selected[role]
        payload = record.get("payload")
        require(record.get("dtype") == "torch.bfloat16", f"selected dtype: {role}")
        require(type(payload) is bytes and len(payload) % 2 == 0, f"selected payload: {role}")
        for word_index, (word,) in enumerate(struct.iter_unpack("<H", payload)):
            if ((word >> 7) & 0xFF) == 0xFF:
                raise EvaluationError(
                    "NON_FINITE_ORACLE_REJECTED",
                    f"non-finite BF16 in complete selected {role} record at flat index {word_index}",
                )


def signed_bf16_dyadic(word: int) -> tuple[int, int]:
    exponent_field = (word >> 7) & 0xFF
    fraction = word & 0x7F
    require(exponent_field != 0xFF, "non-finite BF16", "NON_FINITE_ORACLE_REJECTED")
    if exponent_field == 0:
        coefficient, exponent = fraction, -133
    else:
        coefficient, exponent = 128 + fraction, exponent_field - 134
    if word & 0x8000:
        coefficient = -coefficient
    return (0, 0) if coefficient == 0 else (coefficient, exponent)


def round_even_signed(value: int, right_shift: int) -> int:
    require(right_shift > 0, "rounding shift")
    negative = value < 0
    magnitude = -value if negative else value
    retained, remainder = divmod(magnitude, 1 << right_shift)
    half = 1 << (right_shift - 1)
    if remainder > half or (remainder == half and (retained & 1)):
        retained += 1
    return -retained if negative else retained


def realize_dyadic_q12_20(coefficient: int, exponent: int) -> int:
    shift = exponent + 20
    value = coefficient << shift if shift >= 0 else round_even_signed(coefficient, -shift)
    require(MIN_INT64 <= value <= MAX_INT64, "oracle Q12.20 overflow", "NON_FINITE_ORACLE_REJECTED")
    require(value <= 0, "positive centered oracle", "NON_FINITE_ORACLE_REJECTED")
    return value


def exact_oracle_row(words: tuple[int, ...]) -> tuple[tuple[int, ...], int, int]:
    dyadics = tuple(signed_bf16_dyadic(word) for word in words)
    common = min((exponent for coefficient, exponent in dyadics if coefficient), default=0)
    aligned = tuple(0 if coefficient == 0 else coefficient << (exponent - common) for coefficient, exponent in dyadics)
    maximum = max(aligned)
    top = aligned.index(maximum)
    tie_count = sum(value == maximum for value in aligned)
    centered = tuple(value - maximum for value in aligned)
    return tuple(realize_dyadic_q12_20(value, common) for value in centered), top, tie_count


def realized_row_details(
    score_pairs: tuple[tuple[int, int], ...], valid_mask: tuple[bool, ...]
) -> tuple[int, int]:
    canonical = tuple((0, 0) if mantissa == 0 else (mantissa, exponent) for mantissa, exponent in score_pairs)
    common = min(
        (exponent for (mantissa, exponent), valid in zip(canonical, valid_mask) if valid and mantissa),
        default=0,
    )
    aligned: list[int | None] = []
    for (mantissa, exponent), valid in zip(canonical, valid_mask):
        aligned.append(None if not valid else 0 if mantissa == 0 else mantissa << (exponent - common))
    maximum = max(value for value in aligned if value is not None)
    positive = 0
    saturation = 0
    for value in aligned:
        if value is None:
            continue
        centered = value - maximum
        shift = common + 17
        raw = centered << shift if shift >= 0 else round_even_signed(centered, -shift)
        positive += int(raw > 0)
        saturation += int(raw < MIN_INT64)
    return positive, saturation


def reduced_fraction(numerator: int, denominator: int) -> dict[str, int] | None:
    if denominator == 0:
        return None
    divisor = math.gcd(numerator, denominator)
    return {"denominator": denominator // divisor, "numerator": numerator // divisor}


def lane_input_bindings(lane: dict[str, Any], package: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_token_ids_sha256": lane["input_token_ids_sha256"],
        "lane_metadata": lane["metadata"],
        "normalization_contract_sha256": package["normalization_bindings"]["contract"]["sha256"],
        "representation_contract_sha256": package["representation_bindings"]["contract"]["sha256"],
        "sealed_set_id": package["input_set"]["sealed_set_id"],
        "tensor_bundle": lane["tensor_bundle"],
        "tensor_records": lane["authoritative_tensors"],
    }


def evaluate_selected_records(
    selected: dict[str, dict[str, Any]], lane: dict[str, Any], package: dict[str, Any]
) -> dict[str, Any]:
    validate_selected_record_finiteness(selected)
    representation = import_bound_reference(REPRESENTATION_REL, REPRESENTATION_SHA256, "qk_rep_v2_bound")
    normalization = import_bound_reference(NORMALIZATION_REL, NORMALIZATION_SHA256, "qk_norm_v2_bound")
    oracle = selected["bf16_oracle_scores"]
    query = selected["realized_query_source"]
    key = selected["realized_key_source"]

    encoded_keys: dict[tuple[int, int], tuple[tuple[int, ...], int]] = {}
    for kv_head in range(2):
        for key_index in range(41):
            words = tuple(bf16_word(key, (0, kv_head, key_index, lane_index)) for lane_index in range(64))
            try:
                encoded_keys[(kv_head, key_index)] = representation.encode_head(words)
            except ValueError as exc:
                raise EvaluationError("REPRESENTATION_REJECTED", str(exc)) from exc

    top_rows = 0
    top_matches = 0
    singleton_rows = 0
    unique_rows = 0
    tied_rows = 0
    preserved_rows = 0
    margins: list[int] = []
    valid_values = 0
    sum_signed = 0
    sum_absolute = 0
    sum_squared = 0
    maximum_absolute: int | None = None
    positive_count = 0
    saturation_count = 0
    normalization_rejections = 0

    for query_head in range(14):
        kv_head = query_head // 7
        for query_index in range(41):
            query_words = tuple(
                bf16_word(query, (0, query_head, query_index, lane_index))
                for lane_index in range(64)
            )
            try:
                q_mantissas, q_exponent = representation.encode_head(query_words)
            except ValueError as exc:
                raise EvaluationError("REPRESENTATION_REJECTED", str(exc)) from exc
            score_pairs: list[tuple[int, int]] = []
            for key_index in range(41):
                k_mantissas, k_exponent = encoded_keys[(kv_head, key_index)]
                try:
                    score_pairs.append(
                        representation.dot_product(
                            q_mantissas, q_exponent, k_mantissas, k_exponent
                        )
                    )
                except ValueError as exc:
                    raise EvaluationError("REPRESENTATION_REJECTED", str(exc)) from exc
            valid_mask = tuple(index <= query_index for index in range(41))
            try:
                realized, _, realized_top = normalization.normalize_score_row(
                    tuple(score_pairs), valid_mask
                )
            except ValueError as exc:
                normalization_rejections += 1
                raise EvaluationError("NORMALIZATION_REJECTED", str(exc)) from exc
            row_positive, row_saturation = realized_row_details(tuple(score_pairs), valid_mask)
            positive_count += row_positive
            saturation_count += row_saturation

            oracle_words = tuple(
                bf16_word(oracle, (0, query_head, query_index, key_index))
                for key_index in range(query_index + 1)
            )
            oracle_values, oracle_top, oracle_ties = exact_oracle_row(oracle_words)
            realized_valid = realized[: query_index + 1]
            top_rows += 1
            top_matches += int(realized_top == oracle_top)
            if query_index == 0:
                singleton_rows += 1
            elif oracle_ties == 1:
                unique_rows += 1
                other_max = max(
                    value for index, value in enumerate(realized_valid) if index != oracle_top
                )
                margin = realized_valid[oracle_top] - other_max
                margins.append(margin)
                preserved_rows += int(margin > 0)
            else:
                tied_rows += 1

            for observed, expected in zip(realized_valid, oracle_values):
                error = observed - expected
                absolute = abs(error)
                valid_values += 1
                sum_signed += error
                sum_absolute += absolute
                sum_squared += error * error
                maximum_absolute = absolute if maximum_absolute is None else max(maximum_absolute, absolute)

    top_fraction = reduced_fraction(top_matches, top_rows)
    preserved_fraction = reduced_fraction(preserved_rows, unique_rows)
    metrics = {
        "invalid_accounting": {
            "cross_lane_record_count": 0,
            "invalid_or_non_finite_value_count": 0,
            "normalization_rejection_count": normalization_rejections,
            "positive_centered_realized_score_count": positive_count,
            "saturation_event_count": saturation_count,
        },
        "rank_margin": {
            "minimum_realized_margin_q12_20_lsb": min(margins) if margins else None,
            "oracle_tied_row_count": tied_rows,
            "preserved_positive_margin_fraction": preserved_fraction,
            "preserved_positive_margin_row_count": preserved_rows,
            "singleton_valid_key_row_count": singleton_rows,
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
            "matching_fraction": top_fraction,
            "matching_row_count": top_matches,
            "mismatch_count": top_rows - top_matches,
            "row_count": top_rows,
        },
    }
    require(singleton_rows + unique_rows + tied_rows == top_rows, "rank population closure", "RESULT_SCHEMA_REJECTED")
    require(valid_values == 14 * sum(range(1, 42)), "valid population closure", "RESULT_SCHEMA_REJECTED")
    return metrics


def integer_maximum(actual: int, limit: int) -> dict[str, Any]:
    return {"actual": actual, "limit": limit, "pass": actual <= limit}


def fraction_minimum(actual: dict[str, int] | None, limit: dict[str, int]) -> dict[str, Any]:
    passed = False
    if actual is not None:
        passed = actual["numerator"] * limit["denominator"] >= limit["numerator"] * actual["denominator"]
    return {"actual": actual, "limit": limit, "pass": passed}


def evaluate_thresholds(metrics: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    invalid = metrics["invalid_accounting"]
    rank = metrics["rank_margin"]
    top = metrics["top_key"]
    result = {
        "cross_lane_record_count_maximum": integer_maximum(invalid["cross_lane_record_count"], limits["cross_lane_record_count_maximum"]),
        "invalid_or_non_finite_value_count_maximum": integer_maximum(invalid["invalid_or_non_finite_value_count"], limits["invalid_or_non_finite_value_count_maximum"]),
        "normalization_rejection_count_maximum": integer_maximum(invalid["normalization_rejection_count"], limits["normalization_rejection_count_maximum"]),
        "positive_centered_realized_score_count_maximum": integer_maximum(invalid["positive_centered_realized_score_count"], limits["positive_centered_realized_score_count_maximum"]),
        "rank_margin_violation_count_maximum": integer_maximum(rank["violation_count"], limits["rank_margin_violation_count_maximum"]),
        "saturation_event_count_maximum": integer_maximum(invalid["saturation_event_count"], limits["saturation_event_count_maximum"]),
        "top_key_matching_fraction_minimum": fraction_minimum(top["matching_fraction"], limits["top_key_matching_fraction_minimum"]),
        "top_key_mismatch_count_maximum": integer_maximum(top["mismatch_count"], limits["top_key_mismatch_count_maximum"]),
        "unique_oracle_positive_margin_preserved_fraction_minimum": fraction_minimum(rank["preserved_positive_margin_fraction"], limits["unique_oracle_positive_margin_preserved_fraction_minimum"]),
    }
    result["all_hard_thresholds_pass"] = all(
        check["pass"] for name, check in result.items() if name != "all_hard_thresholds_pass"
    )
    return result


def result_object(
    lane: dict[str, Any],
    package: dict[str, Any],
    metrics: dict[str, Any] | None,
    thresholds: dict[str, Any] | None,
    reason_code: str,
) -> dict[str, Any]:
    succeeded = thresholds is not None and thresholds["all_hard_thresholds_pass"]
    result = {
        "evaluator_spec_sha256": EVALUATOR_SPEC_SHA256,
        "generation_id": lane["generation_id"],
        "input_bindings": lane_input_bindings(lane, package),
        "invocation_sha256": lane["invocation_sha256"],
        "lane_label": lane["label"],
        "metrics": metrics,
        "model_identity_sha256": lane["model_identity_sha256"],
        "namespace_label": lane["namespace_label"],
        "package_id": PACKAGE_ID,
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


def prepare_output_path(output: Path, lane_label: str) -> None:
    expected = ROOT / OUTPUT_ROOT_REL / ("base" if lane_label == "Base" else "checkpoint-176") / "result.json"
    require(output == expected, "output path identity", "OUTPUT_PUBLICATION_REJECTED")
    verify_no_symlink_components(output)
    root = ROOT / OUTPUT_ROOT_REL
    namespace = output.parent
    if not lexical_exists(root):
        root.mkdir(mode=0o700)
        descriptor = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(descriptor)
        os.close(descriptor)
    else:
        info = root.lstat()
        require(stat.S_ISDIR(info.st_mode) and not root.is_symlink(), "output root type", "OUTPUT_PUBLICATION_REJECTED")
    allowed = set() if lane_label == "Base" else {"base"}
    require({entry.name for entry in root.iterdir()} == allowed, "unrelated output entry", "OUTPUT_PUBLICATION_REJECTED")
    require(not lexical_exists(namespace), "lane output namespace collision", "OUTPUT_PUBLICATION_REJECTED")
    namespace.mkdir(mode=0o700)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)


def exclusive_publish(output: Path, result: dict[str, Any]) -> None:
    payload = compact_json_bytes(result)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)


def selected_lane(package: dict[str, Any], lane_label: str) -> dict[str, Any]:
    lanes = package.get("input_set", {}).get("lanes")
    require(isinstance(lanes, list) and len(lanes) == 2, "accepted lane list", "INPUT_BINDING_REJECTED")
    matches = [lane for lane in lanes if lane.get("label") == lane_label]
    require(len(matches) == 1, "selected lane identity", "INPUT_BINDING_REJECTED")
    invocation = next(
        (item for item in package["invocations"] if item["descriptor"]["lane_label"] == lane_label),
        None,
    )
    require(invocation is not None, "selected invocation", "INPUT_BINDING_REJECTED")
    lane = dict(matches[0])
    lane["invocation_sha256"] = invocation["invocation_sha256"]
    return lane


def validate_consumed_ledger(path: Path, lane_label: str, amendment_sha256: str) -> None:
    ledger = load_pretty_json(path)
    require(ledger.get("schema_id") == "QK_BFP8_E16_HEAD64_V1_C02_LIVE_AUTHORITY_LEDGER_V2", "ledger schema", "INPUT_BINDING_REJECTED")
    require(ledger.get("state") == "CONSUMED", "authority not consumed", "INPUT_BINDING_REJECTED")
    require(ledger.get("lane_label") == lane_label, "ledger lane", "CROSS_LANE_REJECTED")
    require(ledger.get("amendment_sha256") == amendment_sha256, "ledger amendment", "INPUT_BINDING_REJECTED")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(allow_abbrev=False)
    value.add_argument("--lane", required=True, choices=("Base", "checkpoint-176"))
    value.add_argument("--amendment", required=True)
    value.add_argument("--authority-ledger", required=True)
    value.add_argument("--evaluation-package", required=True)
    value.add_argument("--input-metadata", required=True)
    value.add_argument("--tensor-bundle", required=True)
    value.add_argument("--output", required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    require(Path.cwd() == ROOT, "canonical cwd", "INPUT_BINDING_REJECTED")
    require(dict(os.environ) == EXACT_ENVIRONMENT, "exact minimal environment", "INPUT_BINDING_REJECTED")
    package_path = Path(args.evaluation_package)
    amendment_path = Path(args.amendment)
    ledger_path = Path(args.authority_ledger)
    metadata_path = Path(args.input_metadata)
    tensor_path = Path(args.tensor_bundle)
    output_path = Path(args.output)
    for path in (package_path, amendment_path, ledger_path, metadata_path, tensor_path, output_path):
        require(not path.is_absolute(), "argv path must be repository-relative", "INPUT_BINDING_REJECTED")
    package_path = ROOT / package_path
    amendment_path = ROOT / amendment_path
    ledger_path = ROOT / ledger_path
    metadata_path = ROOT / metadata_path
    tensor_path = ROOT / tensor_path
    output_path = ROOT / output_path

    validate_relative_path(args.evaluation_package, EVALUATION_PACKAGE_REL)
    validate_relative_path(args.amendment, AMENDMENT_REL)
    require(sha256_file(package_path) == EVALUATION_PACKAGE_SHA256, "evaluation package hash", "INPUT_BINDING_REJECTED")
    package = load_pretty_json(package_path)
    amendment = load_pretty_json(amendment_path)
    amendment_sha256 = sha256_file(amendment_path)
    require(amendment.get("amendment_id") == "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_AUTHORITATIVE_AMENDMENT_V2", "amendment id", "INPUT_BINDING_REJECTED")
    lane = selected_lane(package, args.lane)
    amendment_lane = amendment["execution_package"]["lanes"][args.lane]
    require(amendment_lane["accepted_invocation_sha256"] == lane["invocation_sha256"], "amendment invocation", "INPUT_BINDING_REJECTED")
    validate_relative_path(args.input_metadata, lane["metadata"]["path"])
    validate_relative_path(args.tensor_bundle, lane["tensor_bundle"]["path"])
    validate_relative_path(args.output, lane["output_path"])
    validate_relative_path(
        args.authority_ledger,
        LIVE_ROOT_REL + "/lanes/" + lane["namespace_label"] + "/authority-ledger.json",
    )
    validate_consumed_ledger(ledger_path, args.lane, amendment_sha256)
    prepare_output_path(output_path, args.lane)

    try:
        require(sha256_file(metadata_path) == lane["metadata"]["sha256"], "metadata hash", "INPUT_BINDING_REJECTED")
        metadata = load_pretty_json(metadata_path)
        validate_metadata(metadata, lane)
        require(sha256_file(tensor_path) == lane["tensor_bundle"]["sha256"], "bundle hash", "INPUT_BINDING_REJECTED")
        records = read_tensor_bundle(tensor_path)
        selected = validate_selected_records(records, lane)
        metrics = evaluate_selected_records(selected, lane, package)
        thresholds = evaluate_thresholds(metrics, package["evaluation_contract"]["thresholds"])
        reason = "HARD_THRESHOLDS_PASSED" if thresholds["all_hard_thresholds_pass"] else "HARD_THRESHOLD_FAILED"
        result = result_object(lane, package, metrics, thresholds, reason)
    except EvaluationError as exc:
        result = result_object(lane, package, None, None, exc.reason_code)
    exclusive_publish(output_path, result)
    return 0 if result["terminal"]["status"] == "SUCCEEDED_TERMINAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
