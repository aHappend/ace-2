#!/usr/bin/env python3
"""V2 exactly-once transaction core for the ordered c02 QK score lanes.

The command-line entry point is intentionally disabled: this package freezes an
executable surface but does not materialize a live credential or READY record.
Static verification replaces the launch backend and uses temporary roots only.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, NoReturn


CANONICAL_ROOT = Path(__file__).resolve().parents[1]
ROOT = CANONICAL_ROOT
AMENDMENT_REL = (
    "reference/"
    "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_AUTHORITATIVE_AMENDMENT_V2.json"
)
AMENDMENT_SHA256 = "ba78ea65ba764df0009132fe7e29b9e7d0e4ec92077e22ab9af318970eaa02bb"
LIVE_ROOT_REL = "build/qk-bfp8-e16-head64-v1-c02-live-authority-surface-v2"
OUTPUT_ROOT_REL = "build/qk-bfp8-e16-head64-v1-c02-score-path-evaluation-v1"
EVALUATION_PACKAGE_REL = (
    "reference/QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATION_PACKAGE.json"
)
RESULT_SCHEMA_ID = "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_RESULT_V1"
EVALUATION_PACKAGE_ID = "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATION_PACKAGE_V1"
READY_SCHEMA_ID = "QK_BFP8_E16_HEAD64_V1_C02_LIVE_AUTHORITY_READY_V2"
LEDGER_SCHEMA_ID = "QK_BFP8_E16_HEAD64_V1_C02_LIVE_AUTHORITY_LEDGER_V2"
TERMINAL_SCHEMA_ID = "QK_BFP8_E16_HEAD64_V1_C02_LIVE_AUTHORITY_TERMINAL_V2"
READY_STATE = "READY_UNCONSUMED"
CONSUMED_STATE = "CONSUMED"
INVALIDATED_STATE = "INVALIDATED_TERMINAL"
TERMINAL_STATES = {"FAILED_TERMINAL", "SUCCEEDED_TERMINAL"}


class SurfaceError(RuntimeError):
    """A fail-closed authority or filesystem rejection."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SurfaceError(message)


def fail_live(message: str) -> NoReturn:
    print(f"LIVE_SURFACE_V2_BLOCKED_NO_AUTHORITY_NO_EXECUTION: {message}", file=sys.stderr)
    raise SystemExit(2)


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


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(compact_json_bytes(value)).hexdigest()


def is_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def lexical_exists(path: Path) -> bool:
    return os.path.lexists(path)


def live_root() -> Path:
    return ROOT / LIVE_ROOT_REL


def output_root() -> Path:
    return ROOT / OUTPUT_ROOT_REL


def verify_no_symlink_components(path: Path) -> None:
    require(path.is_absolute(), f"path is not absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        require(not stat.S_ISLNK(info.st_mode), f"symlinked path component: {current}")


def verify_plain_file(path: Path, mode: int | None = None) -> None:
    verify_no_symlink_components(path)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise SurfaceError(f"missing regular file: {path}") from exc
    require(stat.S_ISREG(info.st_mode), f"not a regular file: {path}")
    if mode is not None:
        require(stat.S_IMODE(info.st_mode) == mode, f"file mode mismatch: {path}")


def verify_plain_directory(path: Path, mode: int | None = 0o700) -> None:
    verify_no_symlink_components(path)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise SurfaceError(f"missing directory: {path}") from exc
    require(stat.S_ISDIR(info.st_mode), f"not a directory: {path}")
    if mode is not None:
        require(stat.S_IMODE(info.st_mode) == mode, f"directory mode mismatch: {path}")


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
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SurfaceError(f"invalid JSON: {path}") from exc
    require(raw == pretty_json_bytes(value), f"noncanonical pretty JSON: {path}")
    return value


def load_compact_json(path: Path) -> Any:
    verify_plain_file(path)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SurfaceError(f"invalid compact JSON: {path}") from exc
    require(
        raw
        == (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
        f"noncanonical compact JSON: {path}",
    )
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_write_json(path: Path, value: Any, mode: int = 0o400) -> None:
    verify_no_symlink_components(path)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        mode,
    )
    try:
        payload = pretty_json_bytes(value)
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def atomic_replace_json(path: Path, value: Any) -> None:
    temporary = path.parent / ("." + path.name + ".replace")
    require(not lexical_exists(temporary), f"replacement collision: {temporary}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o400,
    )
    try:
        payload = pretty_json_bytes(value)
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def lane_slug(lane_label: str) -> str:
    require(lane_label in {"Base", "checkpoint-176"}, "unknown lane")
    return "base" if lane_label == "Base" else "checkpoint-176"


def lane_paths(lane_label: str) -> dict[str, Path]:
    root = live_root() / "lanes" / lane_slug(lane_label)
    return {
        "root": root,
        "ledger": root / "authority-ledger.json",
        "lock": root / ".transition.lock",
        "terminal": root / "first-terminal.json",
    }


def load_amendment() -> dict[str, Any]:
    path = CANONICAL_ROOT / AMENDMENT_REL
    require(sha256_file(path) == AMENDMENT_SHA256, "amendment SHA-256 mismatch")
    amendment = load_pretty_json(path)
    require(
        amendment.get("amendment_id")
        == "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_AUTHORITATIVE_AMENDMENT_V2",
        "amendment identity",
    )
    require(amendment.get("authority_state") == "STATIC_FROZEN_NO_LIVE_CREDENTIAL", "live authority state")
    return amendment


def verify_bound_artifacts(amendment: dict[str, Any]) -> None:
    for group in amendment["accepted_bindings"].values():
        for binding in group.values():
            path_text = binding["path"]
            path = (
                Path(path_text)
                if Path(path_text).is_absolute()
                else CANONICAL_ROOT / path_text
            )
            require(sha256_file(path) == binding["sha256"], f"bound artifact mutation: {path_text}")
    evaluator = amendment["execution_package"]["evaluator"]
    require(
        sha256_file(CANONICAL_ROOT / evaluator["path"]) == evaluator["sha256"],
        "evaluator mutation",
    )
    interpreter = amendment["execution_package"]["interpreter"]
    interpreter_path = Path(interpreter["path"])
    require(interpreter_path.is_absolute(), "interpreter path")
    verify_plain_file(interpreter_path)
    require(os.access(interpreter_path, os.X_OK), "interpreter is not executable")
    require(sha256_file(interpreter_path) == interpreter["sha256"], "interpreter mutation")


def validate_lane_descriptor(amendment: dict[str, Any], lane_label: str) -> dict[str, Any]:
    lane = amendment["execution_package"]["lanes"][lane_label]
    argv = lane["argv"]
    require(type(argv) is list and all(type(item) is str and item for item in argv), "argv schema")
    require(argv[0] == amendment["execution_package"]["interpreter"]["path"], "argv interpreter")
    require(argv[1] == amendment["execution_package"]["evaluator"]["path"], "argv evaluator")
    require(object_sha256(argv) == lane["materialized_argv_sha256"], "materialized argv hash")
    require(lane["environment"] == amendment["execution_package"]["minimal_environment"], "lane environment")
    require(
        lane["historical_logical_command_descriptor_sha256"]
        != lane["materialized_argv_sha256"],
        "logical descriptor falsely equated to argv hash",
    )
    return lane


def validate_runtime_roots() -> None:
    require(ROOT.is_absolute(), "repository root not absolute")
    require(ROOT == Path(os.path.normpath(str(ROOT))), "repository root noncanonical")
    if ROOT == CANONICAL_ROOT:
        require(ROOT == CANONICAL_ROOT, "canonical root mismatch")
    verify_no_symlink_components(ROOT)
    verify_no_symlink_components(live_root())
    verify_no_symlink_components(output_root())


def directory_names(path: Path) -> set[str]:
    return {entry.name for entry in path.iterdir()}


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    require(type(value) is dict, f"{label} object")
    require(set(value) == expected, f"{label} keys")
    return value


def require_integer(value: Any, label: str, *, nonnegative: bool = False) -> int:
    require(type(value) is int, f"{label} integer")
    if nonnegative:
        require(value >= 0, f"{label} nonnegative")
    return value


def validate_fraction(value: Any, label: str, *, nullable: bool) -> dict[str, int] | None:
    if value is None:
        require(nullable, f"{label} null")
        return None
    fraction = require_exact_keys(value, {"denominator", "numerator"}, label)
    denominator = require_integer(fraction["denominator"], f"{label} denominator")
    numerator = require_integer(fraction["numerator"], f"{label} numerator", nonnegative=True)
    require(denominator > 0, f"{label} denominator positive")
    require(numerator <= denominator, f"{label} proper fraction")
    require(math.gcd(numerator, denominator) == 1, f"{label} reduced fraction")
    return fraction


def reduced_fraction(numerator: int, denominator: int) -> dict[str, int] | None:
    if denominator == 0:
        return None
    divisor = math.gcd(numerator, denominator)
    return {"denominator": denominator // divisor, "numerator": numerator // divisor}


def load_evaluation_package(amendment: dict[str, Any]) -> dict[str, Any]:
    binding = amendment["accepted_bindings"]["evaluation"]["package"]
    require(binding["path"] == EVALUATION_PACKAGE_REL, "evaluation package path")
    path = CANONICAL_ROOT / EVALUATION_PACKAGE_REL
    require(sha256_file(path) == binding["sha256"], "evaluation package mutation")
    package = load_pretty_json(path)
    require(package.get("package_id") == EVALUATION_PACKAGE_ID, "evaluation package identity")
    return package


def selected_package_lane(package: dict[str, Any], lane_label: str) -> dict[str, Any]:
    selected = [lane for lane in package["input_set"]["lanes"] if lane.get("label") == lane_label]
    require(len(selected) == 1, "evaluation package lane cardinality")
    return selected[0]


def expected_input_bindings(package: dict[str, Any], package_lane: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_token_ids_sha256": package_lane["input_token_ids_sha256"],
        "lane_metadata": package_lane["metadata"],
        "normalization_contract_sha256": package["normalization_bindings"]["contract"]["sha256"],
        "representation_contract_sha256": package["representation_bindings"]["contract"]["sha256"],
        "sealed_set_id": package["input_set"]["sealed_set_id"],
        "tensor_bundle": package_lane["tensor_bundle"],
        "tensor_records": package_lane["authoritative_tensors"],
    }


def validate_metrics(metrics: Any, package: dict[str, Any]) -> dict[str, Any]:
    value = require_exact_keys(
        metrics,
        {"invalid_accounting", "rank_margin", "score_error", "top_key"},
        "metrics",
    )
    invalid = require_exact_keys(
        value["invalid_accounting"],
        {
            "cross_lane_record_count",
            "invalid_or_non_finite_value_count",
            "normalization_rejection_count",
            "positive_centered_realized_score_count",
            "saturation_event_count",
        },
        "invalid accounting metrics",
    )
    for name, item in invalid.items():
        require_integer(item, f"invalid accounting {name}", nonnegative=True)

    rank = require_exact_keys(
        value["rank_margin"],
        {
            "minimum_realized_margin_q12_20_lsb",
            "oracle_tied_row_count",
            "preserved_positive_margin_fraction",
            "preserved_positive_margin_row_count",
            "singleton_valid_key_row_count",
            "unique_oracle_top_row_count",
            "violation_count",
        },
        "rank margin metrics",
    )
    for name in (
        "oracle_tied_row_count",
        "preserved_positive_margin_row_count",
        "singleton_valid_key_row_count",
        "unique_oracle_top_row_count",
        "violation_count",
    ):
        require_integer(rank[name], f"rank margin {name}", nonnegative=True)
    minimum_margin = rank["minimum_realized_margin_q12_20_lsb"]
    if minimum_margin is not None:
        require_integer(minimum_margin, "minimum realized margin")
    validate_fraction(
        rank["preserved_positive_margin_fraction"],
        "preserved positive margin fraction",
        nullable=True,
    )

    score = require_exact_keys(
        value["score_error"],
        {
            "maximum_absolute_error_q12_20_lsb",
            "sum_absolute_error_q12_20_lsb",
            "sum_signed_error_q12_20_lsb",
            "sum_squared_error_q40_40_lsb2",
            "valid_value_count",
        },
        "score error metrics",
    )
    maximum_absolute = score["maximum_absolute_error_q12_20_lsb"]
    if maximum_absolute is not None:
        require_integer(maximum_absolute, "maximum absolute error", nonnegative=True)
    require_integer(score["sum_absolute_error_q12_20_lsb"], "sum absolute error", nonnegative=True)
    require_integer(score["sum_signed_error_q12_20_lsb"], "sum signed error")
    require_integer(score["sum_squared_error_q40_40_lsb2"], "sum squared error", nonnegative=True)
    require_integer(score["valid_value_count"], "valid value count", nonnegative=True)

    top = require_exact_keys(
        value["top_key"],
        {"matching_fraction", "matching_row_count", "mismatch_count", "row_count"},
        "top key metrics",
    )
    for name in ("matching_row_count", "mismatch_count", "row_count"):
        require_integer(top[name], f"top key {name}", nonnegative=True)
    validate_fraction(top["matching_fraction"], "top key matching fraction", nullable=True)

    require(top["matching_row_count"] + top["mismatch_count"] == top["row_count"], "top key population closure")
    require(
        rank["singleton_valid_key_row_count"]
        + rank["unique_oracle_top_row_count"]
        + rank["oracle_tied_row_count"]
        == top["row_count"],
        "rank population closure",
    )
    require(
        rank["preserved_positive_margin_row_count"] + rank["violation_count"]
        == rank["unique_oracle_top_row_count"],
        "rank preservation closure",
    )
    require(
        top["matching_fraction"]
        == reduced_fraction(top["matching_row_count"], top["row_count"]),
        "top key fraction identity",
    )
    require(
        rank["preserved_positive_margin_fraction"]
        == reduced_fraction(
            rank["preserved_positive_margin_row_count"],
            rank["unique_oracle_top_row_count"],
        ),
        "rank margin fraction identity",
    )
    if rank["unique_oracle_top_row_count"] == 0:
        require(minimum_margin is None, "empty rank margin minimum")
    else:
        require(minimum_margin is not None, "nonempty rank margin minimum")
        require(
            (rank["violation_count"] == 0 and minimum_margin > 0)
            or (rank["violation_count"] > 0 and minimum_margin <= 0),
            "rank margin violation identity",
        )

    if score["valid_value_count"] == 0:
        require(maximum_absolute is None, "empty score maximum")
        require(
            score["sum_absolute_error_q12_20_lsb"] == 0
            and score["sum_signed_error_q12_20_lsb"] == 0
            and score["sum_squared_error_q40_40_lsb2"] == 0,
            "empty score sums",
        )
    else:
        require(maximum_absolute is not None, "nonempty score maximum")
        require(score["sum_absolute_error_q12_20_lsb"] >= maximum_absolute, "score maximum bound")
        require(
            score["sum_absolute_error_q12_20_lsb"] >= abs(score["sum_signed_error_q12_20_lsb"]),
            "score signed sum bound",
        )
        require(
            (score["sum_absolute_error_q12_20_lsb"] == 0)
            == (score["sum_squared_error_q40_40_lsb2"] == 0),
            "score zero-sum identity",
        )

    roles = package["evaluation_contract"]["authoritative_tensor_mapping"]["roles"]
    query_heads = roles["bf16_oracle_scores"]["shape"][1]
    sequence_length = package["evaluation_contract"]["authoritative_tensor_mapping"]["sequence_length"]
    require(top["row_count"] == query_heads * sequence_length, "complete top-key row population")
    require(rank["singleton_valid_key_row_count"] == query_heads, "singleton row population")
    require(
        score["valid_value_count"] == query_heads * sequence_length * (sequence_length + 1) // 2,
        "complete valid-value population",
    )
    return value


def validate_threshold_evaluation(
    thresholds: Any, metrics: dict[str, Any], package: dict[str, Any]
) -> dict[str, Any]:
    require(type(metrics) is dict, "threshold metrics object")
    value = require_exact_keys(
        thresholds,
        {
            "all_hard_thresholds_pass",
            "cross_lane_record_count_maximum",
            "invalid_or_non_finite_value_count_maximum",
            "normalization_rejection_count_maximum",
            "positive_centered_realized_score_count_maximum",
            "rank_margin_violation_count_maximum",
            "saturation_event_count_maximum",
            "top_key_matching_fraction_minimum",
            "top_key_mismatch_count_maximum",
            "unique_oracle_positive_margin_preserved_fraction_minimum",
        },
        "threshold evaluation",
    )
    limits = package["evaluation_contract"]["thresholds"]
    integer_actuals = {
        "cross_lane_record_count_maximum": metrics["invalid_accounting"]["cross_lane_record_count"],
        "invalid_or_non_finite_value_count_maximum": metrics["invalid_accounting"]["invalid_or_non_finite_value_count"],
        "normalization_rejection_count_maximum": metrics["invalid_accounting"]["normalization_rejection_count"],
        "positive_centered_realized_score_count_maximum": metrics["invalid_accounting"]["positive_centered_realized_score_count"],
        "rank_margin_violation_count_maximum": metrics["rank_margin"]["violation_count"],
        "saturation_event_count_maximum": metrics["invalid_accounting"]["saturation_event_count"],
        "top_key_mismatch_count_maximum": metrics["top_key"]["mismatch_count"],
    }
    passes: list[bool] = []
    for name, actual in integer_actuals.items():
        check = require_exact_keys(value[name], {"actual", "limit", "pass"}, name)
        require_integer(check["actual"], f"{name} actual", nonnegative=True)
        require_integer(check["limit"], f"{name} limit", nonnegative=True)
        require(type(check["pass"]) is bool, f"{name} pass boolean")
        require(check["actual"] == actual, f"{name} actual identity")
        require(check["limit"] == limits[name], f"{name} limit identity")
        require(check["pass"] == (actual <= limits[name]), f"{name} comparison")
        passes.append(check["pass"])

    fraction_actuals = {
        "top_key_matching_fraction_minimum": metrics["top_key"]["matching_fraction"],
        "unique_oracle_positive_margin_preserved_fraction_minimum": metrics["rank_margin"][
            "preserved_positive_margin_fraction"
        ],
    }
    for name, actual in fraction_actuals.items():
        check = require_exact_keys(value[name], {"actual", "limit", "pass"}, name)
        validate_fraction(check["actual"], f"{name} actual", nullable=True)
        validate_fraction(check["limit"], f"{name} limit", nullable=False)
        require(type(check["pass"]) is bool, f"{name} pass boolean")
        require(check["actual"] == actual, f"{name} actual identity")
        require(check["limit"] == limits[name], f"{name} limit identity")
        expected_pass = False
        if actual is not None:
            expected_pass = (
                actual["numerator"] * limits[name]["denominator"]
                >= limits[name]["numerator"] * actual["denominator"]
            )
        require(check["pass"] == expected_pass, f"{name} comparison")
        passes.append(check["pass"])

    require(type(value["all_hard_thresholds_pass"]) is bool, "all thresholds pass boolean")
    require(value["all_hard_thresholds_pass"] == all(passes), "all thresholds pass identity")
    return value


def validate_result_terminal(
    terminal: Any, metrics: Any, thresholds: Any
) -> dict[str, Any]:
    value = require_exact_keys(
        terminal,
        {
            "first_record_immutable",
            "invocation_count_performed",
            "metrics_published",
            "reason_code",
            "retry_replay_resume_repair_permitted",
            "status",
            "thresholds_evaluated",
        },
        "result terminal",
    )
    require(value["first_record_immutable"] is True, "result terminal immutability")
    require(value["retry_replay_resume_repair_permitted"] is False, "result terminal retry")
    require(
        type(value["invocation_count_performed"]) is int
        and value["invocation_count_performed"] == 1,
        "result invocation count",
    )
    require(type(value["metrics_published"]) is bool, "result metrics published boolean")
    require(type(value["thresholds_evaluated"]) is bool, "result thresholds evaluated boolean")
    require(type(value["reason_code"]) is str, "result reason code")
    require(
        type(value["status"]) is str
        and value["status"] in {"SUCCEEDED_TERMINAL", "FAILED_TERMINAL"},
        "result terminal state",
    )
    if metrics is None or thresholds is None:
        require(metrics is None and thresholds is None, "result pre-metric null pairing")
        require(value["status"] == "FAILED_TERMINAL", "result pre-metric terminal state")
        require(value["metrics_published"] is False, "result pre-metric publication")
        require(value["thresholds_evaluated"] is False, "result pre-metric threshold flag")
        require(
            value["reason_code"]
            in {
                "CROSS_LANE_REJECTED",
                "INPUT_BINDING_REJECTED",
                "INPUT_SCHEMA_REJECTED",
                "NON_FINITE_ORACLE_REJECTED",
                "NORMALIZATION_REJECTED",
                "OUTPUT_PUBLICATION_REJECTED",
                "REPRESENTATION_REJECTED",
                "RESULT_SCHEMA_REJECTED",
            },
            "result pre-metric reason",
        )
    else:
        require(value["metrics_published"] is True, "result metrics publication")
        require(value["thresholds_evaluated"] is True, "result threshold evaluation flag")
        succeeded = thresholds["all_hard_thresholds_pass"]
        require(
            value["status"] == ("SUCCEEDED_TERMINAL" if succeeded else "FAILED_TERMINAL"),
            "result terminal threshold state",
        )
        require(
            value["reason_code"] == ("HARD_THRESHOLDS_PASSED" if succeeded else "HARD_THRESHOLD_FAILED"),
            "result terminal threshold reason",
        )
    return value


def validate_result(
    path: Path,
    amendment: dict[str, Any],
    lane_label: str,
    lane: dict[str, Any],
) -> dict[str, Any]:
    result = load_compact_json(path)
    required = {
        "evaluator_spec_sha256",
        "generation_id",
        "input_bindings",
        "invocation_sha256",
        "lane_label",
        "metrics",
        "model_identity_sha256",
        "namespace_label",
        "package_id",
        "result_sha256",
        "schema_id",
        "terminal",
        "threshold_evaluation",
    }
    require_exact_keys(result, required, "result top-level schema")
    result_without_hash = dict(result)
    result_hash = result_without_hash.pop("result_sha256")
    require(is_sha256(result_hash), "result self hash encoding")
    require(result_hash == object_sha256(result_without_hash), "result self hash")
    package = load_evaluation_package(amendment)
    package_lane = selected_package_lane(package, lane_label)
    invocations = [
        item
        for item in package["invocations"]
        if item["descriptor"].get("lane_label") == lane_label
    ]
    require(len(invocations) == 1, "evaluation package invocation cardinality")
    invocation = invocations[0]
    require(package_lane["generation_id"] == lane["generation_id"], "package/amendment generation")
    require(package_lane["namespace_label"] == lane["namespace_label"], "package/amendment namespace")
    require(package_lane["model_identity_sha256"] == lane["model_identity_sha256"], "package/amendment model")
    require(package_lane["metadata"]["sha256"] == lane["input_metadata_sha256"], "package/amendment metadata")
    require(package_lane["tensor_bundle"]["sha256"] == lane["input_tensor_bundle_sha256"], "package/amendment tensor bundle")
    require(
        {name: record["sha256"] for name, record in package_lane["authoritative_tensors"].items()}
        == lane["tensor_record_sha256"],
        "package/amendment tensor records",
    )
    require(invocation["invocation_sha256"] == lane["accepted_invocation_sha256"], "package/amendment invocation")
    require(
        package["evaluator"]["spec_sha256"]
        == amendment["execution_package"]["evaluator"]["evaluator_spec_sha256"],
        "package/amendment evaluator spec",
    )
    require(result["schema_id"] == RESULT_SCHEMA_ID, "result schema id")
    require(result["package_id"] == EVALUATION_PACKAGE_ID, "result package id")
    require(result["evaluator_spec_sha256"] == package["evaluator"]["spec_sha256"], "result evaluator spec")
    require(result["lane_label"] == lane_label, "result lane")
    require(result["namespace_label"] == lane_slug(lane_label), "result namespace")
    require(result["generation_id"] == package_lane["generation_id"], "result generation")
    require(result["invocation_sha256"] == lane["accepted_invocation_sha256"], "result invocation")
    require(result["model_identity_sha256"] == lane["model_identity_sha256"], "result model")
    require(
        result["input_bindings"] == expected_input_bindings(package, package_lane),
        "result input bindings",
    )
    metrics = None if result["metrics"] is None else validate_metrics(result["metrics"], package)
    thresholds = (
        None
        if result["threshold_evaluation"] is None
        else validate_threshold_evaluation(result["threshold_evaluation"], metrics, package)
    )
    validate_result_terminal(result["terminal"], metrics, thresholds)
    return result


def validate_terminal_self_hash(record: dict[str, Any]) -> None:
    require(type(record) is dict, "terminal record object")
    value = dict(record)
    terminal_sha256 = value.pop("terminal_sha256", None)
    require(is_sha256(terminal_sha256), "terminal self hash encoding")
    require(object_sha256(value) == terminal_sha256, "terminal self hash")


def validate_live_lane_namespace(lane_label: str, *, include_terminal: bool) -> None:
    paths = lane_paths(lane_label)
    expected = {".transition.lock", "authority-ledger.json"}
    if include_terminal:
        expected.add("first-terminal.json")
    verify_plain_directory(paths["root"])
    require(directory_names(paths["root"]) == expected, f"{lane_label} live namespace contents")
    verify_plain_file(paths["lock"], 0o600)
    verify_plain_file(paths["ledger"], 0o400)
    if include_terminal:
        verify_plain_file(paths["terminal"], 0o400)
    else:
        require(not lexical_exists(paths["terminal"]), f"{lane_label} terminal collision")


def validate_output_lane_namespace(lane_label: str) -> Path:
    namespace = output_root() / lane_slug(lane_label)
    verify_plain_directory(namespace)
    require(directory_names(namespace) == {"result.json"}, f"{lane_label} result namespace contents")
    result_path = namespace / "result.json"
    verify_plain_file(result_path, 0o400)
    return result_path


def validate_terminal_record(record: dict[str, Any], ledger: dict[str, Any]) -> None:
    require_exact_keys(
        record,
        {
            "amendment_sha256",
            "consumption_committed",
            "execution_started",
            "first_record_immutable",
            "invocation_sha256",
            "lane_label",
            "reason_code",
            "record_id",
            "result_file_sha256",
            "result_sha256",
            "retry_replay_resume_repair_permitted",
            "schema_id",
            "status",
            "terminal_sha256",
        },
        "authority terminal",
    )
    validate_terminal_self_hash(record)
    require(record["schema_id"] == TERMINAL_SCHEMA_ID, "authority terminal schema")
    require(record["amendment_sha256"] == ledger["amendment_sha256"], "authority terminal amendment")
    require(record["invocation_sha256"] == ledger["invocation_sha256"], "authority terminal invocation")
    require(record["lane_label"] == ledger["lane_label"], "authority terminal lane")
    require(record["record_id"] == ledger["record_id"], "authority terminal record")
    require(record["first_record_immutable"] is True, "authority terminal immutability")
    require(
        record["retry_replay_resume_repair_permitted"] is False,
        "authority terminal retry",
    )
    require(type(record["consumption_committed"]) is bool, "authority terminal consumption flag")
    require(type(record["execution_started"]) is bool, "authority terminal execution flag")
    require(type(record["reason_code"]) is str and record["reason_code"], "authority terminal reason")
    require(
        type(record["status"]) is str and record["status"] in TERMINAL_STATES,
        "authority terminal status",
    )
    require(not record["execution_started"] or record["consumption_committed"], "execution requires consumption")
    result_file_sha256 = record["result_file_sha256"]
    result_sha256 = record["result_sha256"]
    require(
        (result_file_sha256 is None and result_sha256 is None)
        or (is_sha256(result_file_sha256) and is_sha256(result_sha256)),
        "authority terminal result hash pairing",
    )
    if record["status"] == "SUCCEEDED_TERMINAL":
        require(record["consumption_committed"] is True, "successful terminal consumption")
        require(record["execution_started"] is True, "successful terminal execution")
        require(
            record["reason_code"] == "PROCESS_EXIT_0_IMMUTABLE_RESULT_VERIFIED",
            "successful terminal reason",
        )
        require(is_sha256(result_file_sha256) and is_sha256(result_sha256), "successful terminal result hashes")


def validate_base_prerequisite(amendment: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    base_lane = validate_lane_descriptor(amendment, "Base")
    paths = lane_paths("Base")
    validate_live_lane_namespace("Base", include_terminal=True)
    ledger = load_pretty_json(paths["ledger"])
    terminal = load_pretty_json(paths["terminal"])
    validate_ledger(ledger, amendment, "Base")
    validate_terminal_record(terminal, ledger)
    require(ledger["state"] == CONSUMED_STATE, "Base ledger not consumed")
    require(terminal["status"] == "SUCCEEDED_TERMINAL", "Base terminal not successful")
    result_path = validate_output_lane_namespace("Base")
    result = validate_result(result_path, amendment, "Base", base_lane)
    require(result["terminal"]["status"] == "SUCCEEDED_TERMINAL", "Base result terminal")
    require(terminal["result_file_sha256"] == sha256_file(result_path), "Base result file hash")
    require(terminal["result_sha256"] == result["result_sha256"], "Base result identity")
    return terminal, result


def ensure_output_state(amendment: dict[str, Any], lane_label: str) -> None:
    validate_runtime_roots()
    root = output_root()
    lane_namespace = root / lane_slug(lane_label)
    require(not lexical_exists(lane_namespace), f"{lane_label} result namespace collision")
    if not lexical_exists(root):
        require(lane_label == "Base", "checkpoint requires Base output root")
        return
    verify_plain_directory(root)
    if lane_label == "Base":
        require(directory_names(root) == set(), "unrelated entry before Base")
    else:
        require(directory_names(root) == {"base"}, "unrelated or missing Base output entry")
        validate_base_prerequisite(amendment)


def ensure_live_state_for_materialization(amendment: dict[str, Any], lane_label: str) -> None:
    root = live_root()
    if not lexical_exists(root):
        require(lane_label == "Base", "checkpoint live root missing")
        return
    verify_plain_directory(root)
    require(directory_names(root) == {"lanes"}, "unrelated live-root entry")
    lanes = root / "lanes"
    verify_plain_directory(lanes)
    if lane_label == "Base":
        require(directory_names(lanes) == set(), "unrelated lane before Base")
    else:
        require(directory_names(lanes) == {"base"}, "checkpoint lane ordering/collision")
        validate_base_prerequisite(amendment)


def make_ready_record(
    amendment: dict[str, Any], lane_label: str, credential_sha256: str
) -> dict[str, Any]:
    require(is_sha256(credential_sha256), "credential identity encoding")
    lane = validate_lane_descriptor(amendment, lane_label)
    return {
        "amendment_sha256": AMENDMENT_SHA256,
        "authority_credential_sha256": credential_sha256,
        "authority_record_cardinality": 1,
        "environment": lane["environment"],
        "first_terminal_create_only": True,
        "historical_logical_command_descriptor_sha256": lane[
            "historical_logical_command_descriptor_sha256"
        ],
        "invocation_sha256": lane["accepted_invocation_sha256"],
        "lane_label": lane_label,
        "materialized_argv_sha256": lane["materialized_argv_sha256"],
        "namespace_label": lane_slug(lane_label),
        "no_retry_replay_resume_repair": True,
        "record_id": lane["record_id"],
        "schema_id": READY_SCHEMA_ID,
        "shell_argv": lane["argv"],
        "state": READY_STATE,
    }


def validate_ready_record(
    record: dict[str, Any], amendment: dict[str, Any], lane_label: str
) -> None:
    require(type(record) is dict, "READY record object")
    expected_keys = set(make_ready_record(amendment, lane_label, record.get("authority_credential_sha256", "")))
    require(set(record) == expected_keys, "READY record keys")
    require(record["schema_id"] == READY_SCHEMA_ID, "READY schema")
    require(record["state"] == READY_STATE, "READY state")
    expected = make_ready_record(amendment, lane_label, record["authority_credential_sha256"])
    require(record == expected, "READY identity closure")


def validate_ledger(record: dict[str, Any], amendment: dict[str, Any], lane_label: str) -> None:
    require(type(record) is dict, "ledger object")
    require(record.get("schema_id") == LEDGER_SCHEMA_ID, "ledger schema")
    require(record.get("state") in {CONSUMED_STATE, INVALIDATED_STATE}, "ledger state")
    timestamp_key = (
        "consumed_at_unix_ns" if record["state"] == CONSUMED_STATE else "invalidated_at_unix_ns"
    )
    ready_keys = set(make_ready_record(amendment, lane_label, record.get("authority_credential_sha256", "")))
    expected_keys = (ready_keys - {"schema_id", "state"}) | {
        "schema_id",
        "state",
        "ready_record_sha256",
        timestamp_key,
    }
    require(set(record) == expected_keys, "ledger keys")
    require(is_sha256(record["ready_record_sha256"]), "READY hash encoding")
    require_integer(record[timestamp_key], f"ledger {timestamp_key}", nonnegative=True)
    ready = dict(record)
    ready.pop("ready_record_sha256", None)
    ready.pop("consumed_at_unix_ns", None)
    ready.pop("invalidated_at_unix_ns", None)
    ready["schema_id"] = READY_SCHEMA_ID
    ready["state"] = READY_STATE
    validate_ready_record(ready, amendment, lane_label)
    require(record["ready_record_sha256"] == object_sha256(ready), "READY hash closure")


def materialize_lane(lane_label: str, authority_credential_sha256: str) -> Path:
    """Materialize one isolated READY record; never called by this package's CLI."""

    amendment = load_amendment()
    verify_bound_artifacts(amendment)
    validate_lane_descriptor(amendment, lane_label)
    ensure_output_state(amendment, lane_label)
    ensure_live_state_for_materialization(amendment, lane_label)
    root = live_root()
    if not lexical_exists(root):
        verify_plain_directory(root.parent, None)
        root.mkdir(mode=0o700)
        fsync_directory(root.parent)
        (root / "lanes").mkdir(mode=0o700)
        fsync_directory(root)
    lanes = root / "lanes"
    paths = lane_paths(lane_label)
    require(not lexical_exists(paths["root"]), f"{lane_label} lane collision")
    paths["root"].mkdir(mode=0o700)
    fsync_directory(lanes)
    lock = os.open(
        paths["lock"],
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    os.fsync(lock)
    os.close(lock)
    fsync_directory(paths["root"])
    exclusive_write_json(
        paths["ledger"],
        make_ready_record(amendment, lane_label, authority_credential_sha256),
    )
    validate_live_lane_namespace(lane_label, include_terminal=False)
    return paths["ledger"]


def terminal_record(
    ledger: dict[str, Any],
    status: str,
    reason_code: str,
    *,
    consumption_committed: bool,
    execution_started: bool,
    result_file_sha256: str | None,
    result_sha256: str | None,
) -> dict[str, Any]:
    require(status in TERMINAL_STATES, "terminal status")
    record = {
        "amendment_sha256": ledger["amendment_sha256"],
        "consumption_committed": consumption_committed,
        "execution_started": execution_started,
        "first_record_immutable": True,
        "invocation_sha256": ledger["invocation_sha256"],
        "lane_label": ledger["lane_label"],
        "reason_code": reason_code,
        "record_id": ledger["record_id"],
        "result_file_sha256": result_file_sha256,
        "result_sha256": result_sha256,
        "retry_replay_resume_repair_permitted": False,
        "schema_id": TERMINAL_SCHEMA_ID,
        "status": status,
    }
    record["terminal_sha256"] = object_sha256(record)
    validate_terminal_record(record, ledger)
    return record


def seal_first_terminal(path: Path, record: dict[str, Any]) -> None:
    require(not lexical_exists(path), "first terminal already exists")
    validate_terminal_self_hash(record)
    validate_live_lane_namespace(record["lane_label"], include_terminal=False)
    exclusive_write_json(path, record)
    validate_live_lane_namespace(record["lane_label"], include_terminal=True)


def invalidate_ready(lane_label: str, reason_code: str) -> dict[str, Any]:
    amendment = load_amendment()
    paths = lane_paths(lane_label)
    validate_live_lane_namespace(lane_label, include_terminal=False)
    descriptor = os.open(paths["lock"], os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    try:
        require(not lexical_exists(paths["terminal"]), "terminal collision")
        ready = load_pretty_json(paths["ledger"])
        validate_ready_record(ready, amendment, lane_label)
        invalidated = dict(ready)
        invalidated["schema_id"] = LEDGER_SCHEMA_ID
        invalidated["state"] = INVALIDATED_STATE
        invalidated["ready_record_sha256"] = object_sha256(ready)
        invalidated["invalidated_at_unix_ns"] = time.time_ns()
        atomic_replace_json(paths["ledger"], invalidated)
        terminal = terminal_record(
            invalidated,
            "FAILED_TERMINAL",
            reason_code,
            consumption_committed=False,
            execution_started=False,
            result_file_sha256=None,
            result_sha256=None,
        )
        seal_first_terminal(paths["terminal"], terminal)
        return terminal
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def default_launch(argv: tuple[str, ...], environment: dict[str, str]) -> Any:
    return subprocess.run(
        argv,
        check=False,
        close_fds=True,
        cwd=ROOT,
        env=environment,
        shell=False,
    )


LAUNCH_BACKEND: Callable[[tuple[str, ...], dict[str, str]], Any] = default_launch


def consume_and_run(lane_label: str) -> dict[str, Any]:
    """Durably consume, launch once, validate immutable output, and seal terminal."""

    amendment = load_amendment()
    verify_bound_artifacts(amendment)
    lane = validate_lane_descriptor(amendment, lane_label)
    ensure_output_state(amendment, lane_label)
    paths = lane_paths(lane_label)
    validate_live_lane_namespace(lane_label, include_terminal=False)
    descriptor = os.open(paths["lock"], os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    try:
        require(not lexical_exists(paths["terminal"]), "terminal collision")
        ready = load_pretty_json(paths["ledger"])
        validate_ready_record(ready, amendment, lane_label)
        if lane_label == "checkpoint-176":
            validate_base_prerequisite(amendment)
        consumed = dict(ready)
        consumed["schema_id"] = LEDGER_SCHEMA_ID
        consumed["state"] = CONSUMED_STATE
        consumed["ready_record_sha256"] = object_sha256(ready)
        consumed["consumed_at_unix_ns"] = time.time_ns()
        atomic_replace_json(paths["ledger"], consumed)
        durable = load_pretty_json(paths["ledger"])
        validate_ledger(durable, amendment, lane_label)
        require(durable == consumed, "consumed ledger readback")
        validate_live_lane_namespace(lane_label, include_terminal=False)

        try:
            require(load_amendment() == amendment, "amendment changed after consumption")
            verify_bound_artifacts(amendment)
            ensure_output_state(amendment, lane_label)
            if lane_label == "checkpoint-176":
                validate_base_prerequisite(amendment)
            completed = LAUNCH_BACKEND(tuple(lane["argv"]), dict(lane["environment"]))
            returncode = completed.returncode
            require(type(returncode) is int, "launcher return code")
        except BaseException as exc:
            terminal = terminal_record(
                consumed,
                "FAILED_TERMINAL",
                "LAUNCH_OR_POST_CONSUME_FAILURE_" + type(exc).__name__.upper(),
                consumption_committed=True,
                execution_started=True,
                result_file_sha256=None,
                result_sha256=None,
            )
            seal_first_terminal(paths["terminal"], terminal)
            raise

        result: dict[str, Any] | None = None
        result_file_sha256: str | None = None
        result_sha256: str | None = None
        try:
            root = output_root()
            verify_plain_directory(root)
            expected_output_names = {"base"} if lane_label == "Base" else {"base", "checkpoint-176"}
            require(directory_names(root) == expected_output_names, "post-launch output root contents")
            result_path = validate_output_lane_namespace(lane_label)
            if lane_label == "checkpoint-176":
                validate_base_prerequisite(amendment)
            validate_live_lane_namespace(lane_label, include_terminal=False)
            result = validate_result(result_path, amendment, lane_label, lane)
            result_file_sha256 = sha256_file(result_path)
            result_sha256 = result["result_sha256"]
        except SurfaceError:
            result = None

        succeeded = (
            returncode == 0
            and result is not None
            and result["terminal"]["status"] == "SUCCEEDED_TERMINAL"
        )
        status = "SUCCEEDED_TERMINAL" if succeeded else "FAILED_TERMINAL"
        reason = "PROCESS_EXIT_0_IMMUTABLE_RESULT_VERIFIED" if succeeded else f"PROCESS_EXIT_{returncode}_OR_RESULT_REJECTED"
        terminal = terminal_record(
            consumed,
            status,
            reason,
            consumption_committed=True,
            execution_started=True,
            result_file_sha256=result_file_sha256,
            result_sha256=result_sha256,
        )
        seal_first_terminal(paths["terminal"], terminal)
        return terminal
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def main() -> int:
    fail_live(
        "the V2 amendment freezes argv and implementation identities but contains "
        "no live credential; external authority must materialize a cardinality-one "
        "READY record before this transaction core is eligible"
    )


if __name__ == "__main__":
    raise SystemExit(main())
