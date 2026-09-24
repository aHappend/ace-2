#!/usr/bin/env python3
"""Bit-exact integer reference for the layer-23 rank-1 V correction candidate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = (
    ROOT
    / "evidence/candidates/stage1-layer23-v-rank1-integer-correction-v1/"
    "nonofficial-candidate-0001"
)
FREEZE = CANDIDATE_DIR / "candidate-freeze.json"
RESULT = CANDIDATE_DIR / "result.json"
FRESH_L2_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/32444a7fc5ab/round-0001.json"
)
GENERALIZATION_DIR = (
    ROOT
    / "evidence/candidates/stage1-layer23-v-rank1-integer-correction-v1/"
    "nonofficial-generalization-0001"
)
GENERALIZATION_RESULT = GENERALIZATION_DIR / "result.json"
GENERALIZATION_FREEZE = GENERALIZATION_DIR / "generalization-freeze.json"
GENERALIZATION_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/8e6ad3ce7e45/round-0001.json"
)

CONTRACT_ID = "stage1_layer23_v_rank1_integer_correction_rtl_v1"
MISSION_ID = "stage1vrank1int01"
INPUT_WIDTH = 896
OUTPUT_WIDTH = 128
POSITION_COUNT = 34
INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
RETAINED_OUTPUT_SCALE = 0.06586176203930472

FREEZE_SHA256 = "c00e607a567ebc1f02854e9763f9722eeb985bfaf97962401f48414c73dfd0f4"
RESULT_SHA256 = "fa4547a5884a070966a5c0c1fea8997b2aef51d22d40431df34f719cce6ca037"
FRESH_L2_REVIEW_SHA256 = (
    "80881054b5dc3b06f83027dd548446a8b177b7c56add0cdfd306c4f737cbfc06"
)
GENERALIZATION_RESULT_SHA256 = "3201ac208ff856fa331c6c9544b4abda5557d3701f0cdf153f1dc0f2cf665a90"
GENERALIZATION_FREEZE_SHA256 = "a9ae7e1c2368aeb3ca0a0a2673b16403e8d2e1d1b8266495e0c42005c67acc83"
GENERALIZATION_REVIEW_SHA256 = (
    "08df2f4ce013161c279dcbedcbb1243e6bc7841f14e44d6227bc5d6cc749e90d"
)


@dataclass(frozen=True)
class FrozenRank1Config:
    input_to_rank_s8: tuple[int, ...]
    rank_to_channel_s8: tuple[int, ...]
    first_stage_multiplier_s32: int
    first_stage_right_shift_u6: int
    second_stage_multiplier_s32: tuple[int, ...]
    second_stage_right_shift_u6: tuple[int, ...]


@dataclass(frozen=True)
class Rank1PositionVector:
    position: int
    input_s8: tuple[int, ...]
    baseline_v_s8: tuple[int, ...]
    expected_rank_accumulator_s32: int
    expected_rank_rounded_s32: int
    expected_rank_intermediate_s8: int
    expected_correction_accumulator_s32: tuple[int, ...]
    expected_correction_rounded_s32: tuple[int, ...]
    expected_correction_s8: tuple[int, ...]
    expected_corrected_v_s8: tuple[int, ...]


@dataclass(frozen=True)
class NamedRank1PositionVector:
    source_id: str
    vector: Rank1PositionVector


@dataclass(frozen=True)
class Rank1CorrectionResult:
    rank_accumulator_s32: int
    rank_rounded_s32: int
    rank_intermediate_s8: int
    correction_accumulator_s32: tuple[int, ...]
    correction_rounded_s32: tuple[int, ...]
    correction_s8: tuple[int, ...]
    corrected_v_s8: tuple[int, ...]
    input_saturation: bool
    rank_saturation: bool
    correction_saturation: tuple[bool, ...]
    add_saturation: tuple[bool, ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def artifact_path(record: dict[str, Any]) -> Path:
    value = Path(record["path"])
    return value if value.is_absolute() else ROOT / value


def verify_record(record: dict[str, Any]) -> None:
    path = artifact_path(record)
    require(path.is_file(), f"missing artifact: {record['path']}")
    require(path.stat().st_size == int(record["bytes"]), f"artifact size changed: {record['path']}")
    require(sha256_file(path) == record["sha256"], f"artifact hash changed: {record['path']}")


def verify_candidate_binding(*, check_review: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    require(FREEZE.is_file(), "rank-1 integer candidate freeze is missing")
    require(RESULT.is_file(), "rank-1 integer candidate result is missing")
    require(sha256_file(FREEZE) == FREEZE_SHA256, "rank-1 integer freeze hash changed")
    require(sha256_file(RESULT) == RESULT_SHA256, "rank-1 integer result hash changed")
    if check_review:
        require(FRESH_L2_REVIEW.is_file(), "Fresh-L2 review handoff is missing")
        require(sha256_file(FRESH_L2_REVIEW) == FRESH_L2_REVIEW_SHA256, "Fresh-L2 review hash changed")
        review = load_json(FRESH_L2_REVIEW)
        reason = review.get("review", {}).get("reason", "")
        require(review.get("producer_role") == "reviewer", "Fresh-L2 handoff role changed")
        require(review.get("review", {}).get("status") == "done", "Fresh-L2 review is not done")
        require("reproduced" in reason and "rank-1" in reason, "Fresh-L2 review does not reproduce the rank-1 candidate")
    freeze = load_json(FREEZE)
    result = load_json(RESULT)
    require(
        result.get("bindings", {}).get("freeze", {}).get("sha256") == FREEZE_SHA256,
        "result no longer binds the frozen candidate hash",
    )
    require(
        result.get("bindings", {}).get("runner", {}).get("sha256")
        == "e29892e9fe9af5b22959c37121364cc6b83608892098692464baf20fa6839c0a",
        "result no longer binds the frozen execution runner hash",
    )
    require(
        result.get("decision", {}).get("accepted_integer_candidate") is True,
        "integer candidate is no longer accepted",
    )
    require(
        result.get("decision", {}).get("hardware_representable") is True,
        "integer candidate is no longer hardware-representable",
    )
    require(
        result.get("decision", {}).get("existing_w4a8_path_byte_match") is True,
        "baseline W4A8 path byte-match is no longer true",
    )
    for record in freeze.get("artifacts", {}).values():
        verify_record(record)
    for record in (
        result.get("integer_candidate", {})
        .get("integer_evidence", {})
        .get("artifacts", {})
        .values()
    ):
        verify_record(record)
    return freeze, result


def verify_generalization_binding(*, check_review: bool = True) -> dict[str, Any]:
    require(GENERALIZATION_FREEZE.is_file(), "rank-1 generalization freeze is missing")
    require(GENERALIZATION_RESULT.is_file(), "rank-1 generalization result is missing")
    require(
        sha256_file(GENERALIZATION_FREEZE) == GENERALIZATION_FREEZE_SHA256,
        "rank-1 generalization freeze hash changed",
    )
    require(
        sha256_file(GENERALIZATION_RESULT) == GENERALIZATION_RESULT_SHA256,
        "rank-1 generalization result hash changed",
    )
    if check_review:
        require(GENERALIZATION_REVIEW.is_file(), "Fresh-L2 generalization handoff is missing")
        require(
            sha256_file(GENERALIZATION_REVIEW) == GENERALIZATION_REVIEW_SHA256,
            "Fresh-L2 generalization handoff hash changed",
        )
        review = load_json(GENERALIZATION_REVIEW)
        reason = review.get("review", {}).get("reason", "")
        require(review.get("producer_role") == "reviewer", "Fresh-L2 generalization role changed")
        require(review.get("review", {}).get("status") == "done", "Fresh-L2 generalization is not done")
        require(
            GENERALIZATION_RESULT_SHA256 in reason and "all eight records" in reason,
            "Fresh-L2 generalization handoff does not reproduce the frozen records",
        )
    result = load_json(GENERALIZATION_RESULT)
    require(
        result.get("bindings", {}).get("freeze", {}).get("sha256") == GENERALIZATION_FREEZE_SHA256,
        "generalization result no longer binds the frozen prompt manifest",
    )
    require(
        result.get("aggregate", {}).get("accepted_nonofficial_generalization") is True,
        "generalization result is no longer accepted",
    )
    require(
        int(result.get("aggregate", {}).get("evaluation_record_count", -1)) == 8,
        "generalization record count changed",
    )
    return result


def signed_int8(byte: int) -> int:
    return byte - 256 if byte & 0x80 else byte


def check_signed(value: int, bits: int, label: str) -> int:
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if not minimum <= value <= maximum:
        raise OverflowError(f"{label} does not fit signed-{bits}: {value}")
    return value


def check_unsigned(value: int, bits: int, label: str) -> int:
    if not 0 <= value < (1 << bits):
        raise OverflowError(f"{label} does not fit unsigned-{bits}: {value}")
    return value


def read_s8(record: dict[str, Any], expected_shape: Sequence[int]) -> tuple[int, ...]:
    require(list(record["shape"]) == list(expected_shape), f"shape mismatch for {record['path']}")
    data = artifact_path(record).read_bytes()
    require(len(data) == int(record["bytes"]), f"byte count mismatch for {record['path']}")
    return tuple(signed_int8(value) for value in data)


def read_u8(record: dict[str, Any], expected_shape: Sequence[int]) -> tuple[int, ...]:
    require(list(record["shape"]) == list(expected_shape), f"shape mismatch for {record['path']}")
    data = artifact_path(record).read_bytes()
    require(len(data) == int(record["bytes"]), f"byte count mismatch for {record['path']}")
    return tuple(data)


def read_s32le(record: dict[str, Any], expected_shape: Sequence[int]) -> tuple[int, ...]:
    require(list(record["shape"]) == list(expected_shape), f"shape mismatch for {record['path']}")
    data = artifact_path(record).read_bytes()
    require(len(data) == int(record["bytes"]) and len(data) % 4 == 0, f"byte count mismatch for {record['path']}")
    return tuple(value[0] for value in struct.iter_unpack("<i", data))


def read_f64le(record: dict[str, Any], expected_shape: Sequence[int]) -> tuple[float, ...]:
    require(list(record["shape"]) == list(expected_shape), f"shape mismatch for {record['path']}")
    data = artifact_path(record).read_bytes()
    require(
        len(data) == int(record["bytes"]) and len(data) % 8 == 0,
        f"byte count mismatch for {record['path']}",
    )
    return tuple(value[0] for value in struct.iter_unpack("<d", data))


def quantize_s8_from_f64(values: Sequence[float], scale: float, label: str) -> tuple[int, ...]:
    require(scale > 0.0, f"{label} scale must be positive")
    quantized: list[int] = []
    for index, value in enumerate(values):
        rounded = int(round(float(value) / scale))
        if rounded > INT8_MAX:
            rounded = INT8_MAX
        elif rounded < INT8_MIN:
            rounded = INT8_MIN
        quantized.append(check_signed(rounded, 8, f"{label}[{index}]"))
    return tuple(quantized)


def dequantize_s8(values: Sequence[int], scale: float) -> tuple[float, ...]:
    return tuple(float(value) * scale for value in values)


def assert_dequantized_close(actual: Sequence[float], expected: Sequence[float], label: str) -> None:
    require(len(actual) == len(expected), f"{label} length changed")
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        require(abs(left - right) <= 1.0e-12, f"{label}[{index}] no longer byte-scale aligned")


def pack_s8(values: Iterable[int]) -> bytes:
    return bytes((check_signed(int(value), 8, "signed int8") & 0xFF) for value in values)


def pack_s32le(values: Iterable[int]) -> bytes:
    return b"".join(struct.pack("<i", check_signed(int(value), 32, "signed int32")) for value in values)


def round_shift_even_signed(value: int, shift: int) -> int:
    check_unsigned(shift, 6, "right shift")
    if shift == 0:
        return value
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    return -quotient if value < 0 else quotient


def saturate_s8(value: int) -> tuple[int, bool]:
    if value > INT8_MAX:
        return INT8_MAX, True
    if value < INT8_MIN:
        return INT8_MIN, True
    return value, False


def saturating_add_s8(left: int, right: int) -> tuple[int, bool]:
    check_signed(left, 8, "baseline V output")
    check_signed(right, 8, "correction output")
    return saturate_s8(left + right)


def load_frozen_config(*, check_review: bool = True) -> FrozenRank1Config:
    freeze, _ = verify_candidate_binding(check_review=check_review)
    artifacts = freeze["artifacts"]
    first_multiplier = read_s32le(artifacts["first_stage_multiplier_s32le"], [1])[0]
    first_shift = read_u8(artifacts["first_stage_right_shift_u8"], [1])[0]
    second_shift = read_u8(artifacts["second_stage_right_shift_u8"], [OUTPUT_WIDTH])
    require(0 <= first_shift <= 63, "first stage shift is not u6")
    require(all(0 <= value <= 63 for value in second_shift), "second stage shift is not u6")
    return FrozenRank1Config(
        input_to_rank_s8=read_s8(artifacts["input_to_rank_s8"], [INPUT_WIDTH, 1]),
        rank_to_channel_s8=read_s8(artifacts["rank_to_channel_s8"], [1, OUTPUT_WIDTH]),
        first_stage_multiplier_s32=first_multiplier,
        first_stage_right_shift_u6=first_shift,
        second_stage_multiplier_s32=read_s32le(
            artifacts["second_stage_multiplier_s32le"], [OUTPUT_WIDTH]
        ),
        second_stage_right_shift_u6=second_shift,
    )


def apply_rank1_correction(
    input_s8: Sequence[int],
    baseline_v_s8: Sequence[int],
    config: FrozenRank1Config,
) -> Rank1CorrectionResult:
    require(len(input_s8) == INPUT_WIDTH, "rank-1 correction requires 896 input bytes")
    require(len(baseline_v_s8) == OUTPUT_WIDTH, "rank-1 correction requires 128 baseline V bytes")
    require(len(config.input_to_rank_s8) == INPUT_WIDTH, "input_to_rank coefficient count changed")
    require(len(config.rank_to_channel_s8) == OUTPUT_WIDTH, "rank_to_channel coefficient count changed")
    require(len(config.second_stage_multiplier_s32) == OUTPUT_WIDTH, "second multiplier count changed")
    require(len(config.second_stage_right_shift_u6) == OUTPUT_WIDTH, "second shift count changed")

    input_saturation = False
    rank_accumulator = 0
    for index, (sample, factor) in enumerate(zip(input_s8, config.input_to_rank_s8, strict=True)):
        check_signed(sample, 8, f"input_s8[{index}]")
        check_signed(factor, 8, f"input_to_rank_s8[{index}]")
        rank_accumulator = check_signed(
            rank_accumulator + sample * factor,
            32,
            "rank-1 first-stage accumulator",
        )

    check_signed(config.first_stage_multiplier_s32, 32, "first-stage multiplier")
    rank_product = check_signed(
        rank_accumulator * config.first_stage_multiplier_s32,
        64,
        "rank-1 first-stage product",
    )
    rank_rounded = check_signed(
        round_shift_even_signed(rank_product, config.first_stage_right_shift_u6),
        32,
        "rank-1 rounded intermediate",
    )
    rank_s8, rank_saturation = saturate_s8(rank_rounded)

    correction_accumulators: list[int] = []
    correction_rounded: list[int] = []
    correction_s8: list[int] = []
    corrected_v_s8: list[int] = []
    correction_saturation: list[bool] = []
    add_saturation: list[bool] = []
    for channel in range(OUTPUT_WIDTH):
        factor = config.rank_to_channel_s8[channel]
        check_signed(factor, 8, f"rank_to_channel_s8[{channel}]")
        accumulator = check_signed(
            rank_s8 * factor,
            32,
            f"rank-1 second-stage accumulator[{channel}]",
        )
        multiplier = check_signed(
            config.second_stage_multiplier_s32[channel],
            32,
            f"second-stage multiplier[{channel}]",
        )
        product = check_signed(
            accumulator * multiplier,
            64,
            f"rank-1 second-stage product[{channel}]",
        )
        rounded = check_signed(
            round_shift_even_signed(product, config.second_stage_right_shift_u6[channel]),
            32,
            f"rank-1 rounded correction[{channel}]",
        )
        correction, saturated = saturate_s8(rounded)
        corrected, add_saturated = saturating_add_s8(baseline_v_s8[channel], correction)
        correction_accumulators.append(accumulator)
        correction_rounded.append(rounded)
        correction_s8.append(correction)
        corrected_v_s8.append(corrected)
        correction_saturation.append(saturated)
        add_saturation.append(add_saturated)

    return Rank1CorrectionResult(
        rank_accumulator_s32=rank_accumulator,
        rank_rounded_s32=rank_rounded,
        rank_intermediate_s8=rank_s8,
        correction_accumulator_s32=tuple(correction_accumulators),
        correction_rounded_s32=tuple(correction_rounded),
        correction_s8=tuple(correction_s8),
        corrected_v_s8=tuple(corrected_v_s8),
        input_saturation=input_saturation,
        rank_saturation=rank_saturation,
        correction_saturation=tuple(correction_saturation),
        add_saturation=tuple(add_saturation),
    )


def _chunks(values: Sequence[int], width: int) -> tuple[tuple[int, ...], ...]:
    require(len(values) % width == 0, "flat vector length is not a multiple of row width")
    return tuple(tuple(values[index : index + width]) for index in range(0, len(values), width))


def load_frozen_position_vectors(*, check_review: bool = True) -> tuple[Rank1PositionVector, ...]:
    _, result = verify_candidate_binding(check_review=check_review)
    artifacts = result["integer_candidate"]["integer_evidence"]["artifacts"]
    input_rows = _chunks(read_s8(artifacts["input_payload_s8"], [POSITION_COUNT, INPUT_WIDTH]), INPUT_WIDTH)
    rank_acc = read_s32le(artifacts["rank_intermediate_accumulator_s32le"], [POSITION_COUNT])
    rank_rounded = read_s32le(artifacts["rank_intermediate_rounded_s32le"], [POSITION_COUNT])
    rank_s8 = read_s8(artifacts["rank_intermediate_s8"], [POSITION_COUNT])
    corr_acc = _chunks(
        read_s32le(artifacts["correction_accumulator_s32le"], [POSITION_COUNT, OUTPUT_WIDTH]),
        OUTPUT_WIDTH,
    )
    corr_rounded = _chunks(
        read_s32le(artifacts["correction_rounded_s32le"], [POSITION_COUNT, OUTPUT_WIDTH]),
        OUTPUT_WIDTH,
    )
    corr_s8 = _chunks(read_s8(artifacts["correction_s8"], [POSITION_COUNT, OUTPUT_WIDTH]), OUTPUT_WIDTH)
    corrected_s8 = _chunks(
        read_s8(artifacts["corrected_v_output_s8"], [POSITION_COUNT, OUTPUT_WIDTH]),
        OUTPUT_WIDTH,
    )
    baseline_rows: list[tuple[int, ...]] = []
    for position in range(POSITION_COUNT):
        row: list[int] = []
        for channel in range(OUTPUT_WIDTH):
            baseline = corrected_s8[position][channel] - corr_s8[position][channel]
            check_signed(baseline, 8, f"derived baseline_v_s8[{position}][{channel}]")
            recomputed, saturated = saturating_add_s8(baseline, corr_s8[position][channel])
            require(not saturated, f"frozen final add unexpectedly saturated at position {position} channel {channel}")
            require(
                recomputed == corrected_s8[position][channel],
                f"derived baseline does not reproduce corrected output at position {position} channel {channel}",
            )
            row.append(baseline)
        baseline_rows.append(tuple(row))
    return tuple(
        Rank1PositionVector(
            position=position,
            input_s8=input_rows[position],
            baseline_v_s8=baseline_rows[position],
            expected_rank_accumulator_s32=rank_acc[position],
            expected_rank_rounded_s32=rank_rounded[position],
            expected_rank_intermediate_s8=rank_s8[position],
            expected_correction_accumulator_s32=corr_acc[position],
            expected_correction_rounded_s32=corr_rounded[position],
            expected_correction_s8=corr_s8[position],
            expected_corrected_v_s8=corrected_s8[position],
        )
        for position in range(POSITION_COUNT)
    )


def load_generalization_position_vectors(
    *, check_review: bool = True
) -> tuple[NamedRank1PositionVector, ...]:
    freeze, _ = verify_candidate_binding(check_review=check_review)
    result = verify_generalization_binding(check_review=check_review)
    config = load_frozen_config(check_review=check_review)
    input_scale = float(freeze["quantization"]["input_payload"]["scale32"]["decoded_value"])
    named_vectors: list[NamedRank1PositionVector] = []
    global_position = 0
    for record in result["records"]:
        artifacts = record["artifacts"]
        input_record = artifacts["input_dequantized"]
        baseline_record = artifacts["baseline_v_output"]
        predicted_record = artifacts["predicted_residual"]
        corrected_record = artifacts["corrected_v_output"]
        input_shape = input_record["shape"]
        require(len(input_shape) == 2 and int(input_shape[1]) == INPUT_WIDTH, "generalization input shape changed")
        row_count = int(input_shape[0])
        require(
            baseline_record["shape"] == [row_count, OUTPUT_WIDTH]
            and predicted_record["shape"] == [row_count, OUTPUT_WIDTH]
            and corrected_record["shape"] == [row_count, OUTPUT_WIDTH],
            "generalization V artifact shape changed",
        )
        input_rows = _chunks(
            quantize_s8_from_f64(
                read_f64le(input_record, [row_count, INPUT_WIDTH]),
                input_scale,
                "generalization input",
            ),
            INPUT_WIDTH,
        )
        baseline_f64 = read_f64le(baseline_record, [row_count, OUTPUT_WIDTH])
        predicted_f64 = read_f64le(predicted_record, [row_count, OUTPUT_WIDTH])
        corrected_f64 = read_f64le(corrected_record, [row_count, OUTPUT_WIDTH])
        baseline_rows = _chunks(
            quantize_s8_from_f64(
                baseline_f64,
                RETAINED_OUTPUT_SCALE,
                "generalization baseline V",
            ),
            OUTPUT_WIDTH,
        )
        predicted_rows = _chunks(
            quantize_s8_from_f64(
                predicted_f64,
                RETAINED_OUTPUT_SCALE,
                "generalization predicted residual",
            ),
            OUTPUT_WIDTH,
        )
        corrected_rows = _chunks(
            quantize_s8_from_f64(
                corrected_f64,
                RETAINED_OUTPUT_SCALE,
                "generalization corrected V",
            ),
            OUTPUT_WIDTH,
        )
        assert_dequantized_close(
            dequantize_s8([value for row in baseline_rows for value in row], RETAINED_OUTPUT_SCALE),
            baseline_f64,
            "generalization baseline V",
        )
        assert_dequantized_close(
            dequantize_s8([value for row in predicted_rows for value in row], RETAINED_OUTPUT_SCALE),
            predicted_f64,
            "generalization predicted residual",
        )
        assert_dequantized_close(
            dequantize_s8([value for row in corrected_rows for value in row], RETAINED_OUTPUT_SCALE),
            corrected_f64,
            "generalization corrected V",
        )
        for row in range(row_count):
            actual = apply_rank1_correction(input_rows[row], baseline_rows[row], config)
            require(
                actual.correction_s8 == predicted_rows[row],
                f"generalization correction mismatch at {record['prompt_id']} step {record['decode_step']} row {row}",
            )
            require(
                actual.corrected_v_s8 == corrected_rows[row],
                f"generalization corrected V mismatch at {record['prompt_id']} step {record['decode_step']} row {row}",
            )
            source_id = f"{record['prompt_id']}/step-{int(record['decode_step']):02d}/pos-{row:03d}"
            named_vectors.append(
                NamedRank1PositionVector(
                    source_id=source_id,
                    vector=Rank1PositionVector(
                        position=global_position,
                        input_s8=input_rows[row],
                        baseline_v_s8=baseline_rows[row],
                        expected_rank_accumulator_s32=actual.rank_accumulator_s32,
                        expected_rank_rounded_s32=actual.rank_rounded_s32,
                        expected_rank_intermediate_s8=actual.rank_intermediate_s8,
                        expected_correction_accumulator_s32=actual.correction_accumulator_s32,
                        expected_correction_rounded_s32=actual.correction_rounded_s32,
                        expected_correction_s8=actual.correction_s8,
                        expected_corrected_v_s8=actual.corrected_v_s8,
                    ),
                )
            )
            global_position += 1
    return tuple(named_vectors)


def verify_frozen_vectors(*, check_review: bool = True) -> dict[str, Any]:
    config = load_frozen_config(check_review=check_review)
    vectors = load_frozen_position_vectors(check_review=check_review)
    rank_accumulators: list[int] = []
    rank_rounded: list[int] = []
    rank_s8: list[int] = []
    correction_accumulators: list[int] = []
    correction_rounded: list[int] = []
    correction_s8: list[int] = []
    corrected_s8: list[int] = []
    for vector in vectors:
        actual = apply_rank1_correction(vector.input_s8, vector.baseline_v_s8, config)
        require(
            actual.rank_accumulator_s32 == vector.expected_rank_accumulator_s32,
            f"rank accumulator mismatch at position {vector.position}",
        )
        require(
            actual.rank_rounded_s32 == vector.expected_rank_rounded_s32,
            f"rank rounded mismatch at position {vector.position}",
        )
        require(
            actual.rank_intermediate_s8 == vector.expected_rank_intermediate_s8,
            f"rank byte mismatch at position {vector.position}",
        )
        require(
            actual.correction_accumulator_s32 == vector.expected_correction_accumulator_s32,
            f"correction accumulator mismatch at position {vector.position}",
        )
        require(
            actual.correction_rounded_s32 == vector.expected_correction_rounded_s32,
            f"correction rounded mismatch at position {vector.position}",
        )
        require(
            actual.correction_s8 == vector.expected_correction_s8,
            f"correction byte mismatch at position {vector.position}",
        )
        require(
            actual.corrected_v_s8 == vector.expected_corrected_v_s8,
            f"corrected V byte mismatch at position {vector.position}",
        )
        require(not actual.rank_saturation, f"unexpected frozen rank saturation at position {vector.position}")
        require(not any(actual.correction_saturation), f"unexpected frozen correction saturation at position {vector.position}")
        require(not any(actual.add_saturation), f"unexpected frozen add saturation at position {vector.position}")
        rank_accumulators.append(actual.rank_accumulator_s32)
        rank_rounded.append(actual.rank_rounded_s32)
        rank_s8.append(actual.rank_intermediate_s8)
        correction_accumulators.extend(actual.correction_accumulator_s32)
        correction_rounded.extend(actual.correction_rounded_s32)
        correction_s8.extend(actual.correction_s8)
        corrected_s8.extend(actual.corrected_v_s8)
    return {
        "contract_id": CONTRACT_ID,
        "candidate_freeze_sha256": FREEZE_SHA256,
        "candidate_result_sha256": RESULT_SHA256,
        "fresh_l2_review_sha256": FRESH_L2_REVIEW_SHA256 if check_review else None,
        "positions_verified": len(vectors),
        "input_width": INPUT_WIDTH,
        "output_width": OUTPUT_WIDTH,
        "intermediate_rank_scalar_byte_equal": True,
        "correction_output_byte_equal": True,
        "corrected_v_output_byte_equal": True,
        "rank_intermediate_accumulator_s32le_sha256": sha256_bytes(pack_s32le(rank_accumulators)),
        "rank_intermediate_rounded_s32le_sha256": sha256_bytes(pack_s32le(rank_rounded)),
        "rank_intermediate_s8_sha256": sha256_bytes(pack_s8(rank_s8)),
        "correction_accumulator_s32le_sha256": sha256_bytes(pack_s32le(correction_accumulators)),
        "correction_rounded_s32le_sha256": sha256_bytes(pack_s32le(correction_rounded)),
        "correction_s8_sha256": sha256_bytes(pack_s8(correction_s8)),
        "corrected_v_output_s8_sha256": sha256_bytes(pack_s8(corrected_s8)),
    }


def verify_generalization_vectors(*, check_review: bool = True) -> dict[str, Any]:
    named_vectors = load_generalization_position_vectors(check_review=check_review)
    require(len(named_vectors) > 0, "no generalization vectors loaded")
    input_s8: list[int] = []
    baseline_s8: list[int] = []
    rank_accumulators: list[int] = []
    rank_rounded: list[int] = []
    rank_s8: list[int] = []
    correction_accumulators: list[int] = []
    correction_rounded: list[int] = []
    correction_s8: list[int] = []
    corrected_s8: list[int] = []
    prompts = set()
    for named in named_vectors:
        prompt_id = named.source_id.split("/", 1)[0]
        prompts.add(prompt_id)
        vector = named.vector
        input_s8.extend(vector.input_s8)
        baseline_s8.extend(vector.baseline_v_s8)
        rank_accumulators.append(vector.expected_rank_accumulator_s32)
        rank_rounded.append(vector.expected_rank_rounded_s32)
        rank_s8.append(vector.expected_rank_intermediate_s8)
        correction_accumulators.extend(vector.expected_correction_accumulator_s32)
        correction_rounded.extend(vector.expected_correction_rounded_s32)
        correction_s8.extend(vector.expected_correction_s8)
        corrected_s8.extend(vector.expected_corrected_v_s8)
    return {
        "contract_id": CONTRACT_ID,
        "generalization_freeze_sha256": GENERALIZATION_FREEZE_SHA256,
        "generalization_result_sha256": GENERALIZATION_RESULT_SHA256,
        "generalization_review_sha256": GENERALIZATION_REVIEW_SHA256 if check_review else None,
        "records_verified": 8,
        "prompt_count": len(prompts),
        "positions_verified": len(named_vectors),
        "input_width": INPUT_WIDTH,
        "output_width": OUTPUT_WIDTH,
        "input_payload_s8_sha256": sha256_bytes(pack_s8(input_s8)),
        "baseline_v_output_s8_sha256": sha256_bytes(pack_s8(baseline_s8)),
        "rank_intermediate_accumulator_s32le_sha256": sha256_bytes(pack_s32le(rank_accumulators)),
        "rank_intermediate_rounded_s32le_sha256": sha256_bytes(pack_s32le(rank_rounded)),
        "rank_intermediate_s8_sha256": sha256_bytes(pack_s8(rank_s8)),
        "correction_accumulator_s32le_sha256": sha256_bytes(pack_s32le(correction_accumulators)),
        "correction_rounded_s32le_sha256": sha256_bytes(pack_s32le(correction_rounded)),
        "correction_s8_sha256": sha256_bytes(pack_s8(correction_s8)),
        "corrected_v_output_s8_sha256": sha256_bytes(pack_s8(corrected_s8)),
    }


def synthetic_config(
    *,
    input_to_rank_0: int,
    first_multiplier: int,
    first_shift: int,
    rank_to_channel_0: int,
    second_multiplier_0: int,
    second_shift_0: int,
) -> FrozenRank1Config:
    check_signed(input_to_rank_0, 8, "synthetic input_to_rank_0")
    check_signed(rank_to_channel_0, 8, "synthetic rank_to_channel_0")
    check_unsigned(first_shift, 6, "synthetic first shift")
    check_unsigned(second_shift_0, 6, "synthetic second shift")
    return FrozenRank1Config(
        input_to_rank_s8=(input_to_rank_0,) + (0,) * (INPUT_WIDTH - 1),
        rank_to_channel_s8=(rank_to_channel_0,) + (0,) * (OUTPUT_WIDTH - 1),
        first_stage_multiplier_s32=first_multiplier,
        first_stage_right_shift_u6=first_shift,
        second_stage_multiplier_s32=(second_multiplier_0,) + (0,) * (OUTPUT_WIDTH - 1),
        second_stage_right_shift_u6=(second_shift_0,) + (0,) * (OUTPUT_WIDTH - 1),
    )


def synthetic_cases() -> tuple[dict[str, Any], ...]:
    definitions = [
        (
            "nominal_positive",
            3,
            2,
            1,
            0,
            5,
            1,
            0,
            10,
        ),
        (
            "negative_tie_to_even",
            -3,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
        ),
        (
            "positive_saturation",
            127,
            127,
            1,
            0,
            127,
            1,
            0,
            100,
        ),
        (
            "negative_saturation",
            -128,
            127,
            1,
            0,
            127,
            1,
            0,
            -100,
        ),
    ]
    cases: list[dict[str, Any]] = []
    for (
        name,
        input0,
        input_to_rank0,
        first_multiplier,
        first_shift,
        rank_to_channel0,
        second_multiplier0,
        second_shift0,
        baseline0,
    ) in definitions:
        config = synthetic_config(
            input_to_rank_0=input_to_rank0,
            first_multiplier=first_multiplier,
            first_shift=first_shift,
            rank_to_channel_0=rank_to_channel0,
            second_multiplier_0=second_multiplier0,
            second_shift_0=second_shift0,
        )
        input_payload = (input0,) + (0,) * (INPUT_WIDTH - 1)
        baseline = (baseline0,) + (0,) * (OUTPUT_WIDTH - 1)
        actual = apply_rank1_correction(input_payload, baseline, config)
        cases.append(
            {
                "name": name,
                "input0_s8": input0,
                "input_to_rank0_s8": input_to_rank0,
                "first_multiplier_s32": first_multiplier,
                "first_shift_u6": first_shift,
                "rank_to_channel0_s8": rank_to_channel0,
                "second_multiplier0_s32": second_multiplier0,
                "second_shift0_u6": second_shift0,
                "baseline0_s8": baseline0,
                "expected": {
                    "rank_accumulator_s32": actual.rank_accumulator_s32,
                    "rank_rounded_s32": actual.rank_rounded_s32,
                    "rank_intermediate_s8": actual.rank_intermediate_s8,
                    "correction_accumulator0_s32": actual.correction_accumulator_s32[0],
                    "correction_rounded0_s32": actual.correction_rounded_s32[0],
                    "correction0_s8": actual.correction_s8[0],
                    "corrected0_s8": actual.corrected_v_s8[0],
                    "rank_saturation": actual.rank_saturation,
                    "correction_saturation0": actual.correction_saturation[0],
                    "add_saturation0": actual.add_saturation[0],
                },
            }
        )
    return tuple(cases)


if __name__ == "__main__":
    print(json.dumps(verify_frozen_vectors(), indent=2, sort_keys=True))
