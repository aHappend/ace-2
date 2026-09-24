#!/usr/bin/env python3
"""Pure integer reference for the static grouped-BFP8 Q/K successors.

This module has no model, tensor, evaluator, filesystem, or execution-authority
entry point.  Inputs are synthetic BF16 bit patterns or already-packed records.
"""

from __future__ import annotations

from typing import Iterable


CONTRACT_ID = "QK_GROUPED_BFP8_HEAD64_SUCCESSOR_STATIC_V1"
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
MIN_GROUP_EXPONENT = -139
MAX_GROUP_EXPONENT = 122
MIN_SCORE_EXPONENT = -278
MAX_SCORE_EXPONENT = 244
MAX_ROW_LENGTH = 512
OUTPUT_MAGIC = b"QKSPOU01"

ALTERNATIVES = {
    "QK_GBFP8_G16_E16_HEAD64_V1": {
        "group_count": 4,
        "group_size": 16,
        "partial_width": 19,
        "record_bytes": 72,
    },
    "QK_GBFP8_G8_E16_HEAD64_V1": {
        "group_count": 8,
        "group_size": 8,
        "partial_width": 18,
        "record_bytes": 80,
    },
}


class ContractError(ValueError):
    pass


def _exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ContractError(f"{name} must be an exact integer")
    return value


def _spec(alternative_id: str) -> dict[str, int]:
    if type(alternative_id) is not str or alternative_id not in ALTERNATIVES:
        raise ContractError("unknown alternative_id")
    return ALTERNATIVES[alternative_id]


def decode_bf16_bits(bits: int) -> tuple[int, int]:
    """Return exact (signed_integer, power_of_two_exponent)."""
    bits = _exact_int(bits, "bf16 bits")
    if bits < 0 or bits > 0xFFFF:
        raise ContractError("BF16 bits outside unsigned-16 range")
    sign = -1 if bits & 0x8000 else 1
    exponent_field = (bits >> 7) & 0xFF
    fraction = bits & 0x7F
    if exponent_field == 0xFF:
        raise ContractError("BF16 NaN or infinity")
    if exponent_field == 0:
        if fraction == 0:
            return (0, 0)
        return (sign * fraction, -133)
    return (sign * (128 + fraction), exponent_field - 134)


def _abs_le_scaled(value: int, value_exponent: int, limit: int, limit_exponent: int) -> bool:
    if value < 0 or limit < 0:
        raise AssertionError("absolute comparison requires nonnegative operands")
    delta = value_exponent - limit_exponent
    if delta >= 0:
        return (value << delta) <= limit
    return value <= (limit << (-delta))


def _round_nonnegative_pow2(value: int, binary_shift: int) -> int:
    if value < 0:
        raise AssertionError("rounding helper requires nonnegative magnitude")
    if binary_shift >= 0:
        return value << binary_shift
    discarded = -binary_shift
    quotient, remainder = divmod(value, 1 << discarded)
    half = 1 << (discarded - 1)
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    return quotient


def round_signed_pow2(value: int, binary_shift: int) -> int:
    value = _exact_int(value, "rounding value")
    binary_shift = _exact_int(binary_shift, "binary shift")
    magnitude = _round_nonnegative_pow2(abs(value), binary_shift)
    return -magnitude if value < 0 else magnitude


def _choose_group_exponent(decoded_group: tuple[tuple[int, int], ...]) -> int:
    if all(numerator == 0 for numerator, _ in decoded_group):
        return 0
    for exponent in range(MIN_GROUP_EXPONENT, MAX_GROUP_EXPONENT + 1):
        if all(
            numerator == 0
            or _abs_le_scaled(abs(numerator), source_exponent, 127, exponent)
            for numerator, source_exponent in decoded_group
        ):
            return exponent
    raise ContractError("finite BF16 group has no legal exponent")


def encode_head_bf16(bits_by_lane: tuple[int, ...], alternative_id: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    spec = _spec(alternative_id)
    if type(bits_by_lane) is not tuple or len(bits_by_lane) != 64:
        raise ContractError("head must be an immutable tuple of exactly 64 BF16 words")
    decoded = tuple(decode_bf16_bits(bits) for bits in bits_by_lane)
    mantissas: list[int] = []
    exponents: list[int] = []
    group_size = spec["group_size"]
    for group_index in range(spec["group_count"]):
        group = decoded[group_index * group_size : (group_index + 1) * group_size]
        exponent = _choose_group_exponent(group)
        group_mantissas = tuple(
            round_signed_pow2(numerator, source_exponent - exponent)
            for numerator, source_exponent in group
        )
        if all(mantissa == 0 for mantissa in group_mantissas):
            if exponent != 0:
                raise AssertionError("zero group failed canonicalization")
        else:
            maximum = max(abs(mantissa) for mantissa in group_mantissas)
            if maximum < 64 or maximum > 127:
                raise AssertionError("least-exponent invariant failed")
        mantissas.extend(group_mantissas)
        exponents.append(exponent)
    encoded = (tuple(mantissas), tuple(exponents))
    validate_encoded_head(encoded, alternative_id)
    return encoded


def validate_encoded_head(encoded: tuple[tuple[int, ...], tuple[int, ...]], alternative_id: str) -> None:
    spec = _spec(alternative_id)
    if type(encoded) is not tuple or len(encoded) != 2:
        raise ContractError("encoded head must be a two-tuple")
    mantissas, exponents = encoded
    if type(mantissas) is not tuple or len(mantissas) != 64:
        raise ContractError("encoded mantissas must be a 64-tuple")
    if type(exponents) is not tuple or len(exponents) != spec["group_count"]:
        raise ContractError("encoded exponents have wrong group count")
    group_size = spec["group_size"]
    for lane, mantissa in enumerate(mantissas):
        mantissa = _exact_int(mantissa, f"mantissa[{lane}]")
        if mantissa < -127 or mantissa > 127:
            raise ContractError("mantissa outside canonical signed-int8 range")
    for group_index, exponent in enumerate(exponents):
        exponent = _exact_int(exponent, f"exponent[{group_index}]")
        group = mantissas[group_index * group_size : (group_index + 1) * group_size]
        maximum = max(abs(value) for value in group)
        if maximum == 0:
            if exponent != 0:
                raise ContractError("zero group has nonzero exponent")
        else:
            if exponent < MIN_GROUP_EXPONENT or exponent > MAX_GROUP_EXPONENT:
                raise ContractError("nonzero group exponent outside canonical range")
            if maximum < 64:
                raise ContractError("nonzero group is not least-exponent canonical")


def pack_head_record(encoded: tuple[tuple[int, ...], tuple[int, ...]], alternative_id: str) -> bytes:
    validate_encoded_head(encoded, alternative_id)
    mantissas, exponents = encoded
    payload = bytearray((value & 0xFF) for value in mantissas)
    for exponent in exponents:
        payload.extend(exponent.to_bytes(2, "little", signed=True))
    if len(payload) != ALTERNATIVES[alternative_id]["record_bytes"]:
        raise AssertionError("head record length mismatch")
    return bytes(payload)


def unpack_head_record(payload: bytes, alternative_id: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    spec = _spec(alternative_id)
    if type(payload) is not bytes or len(payload) != spec["record_bytes"]:
        raise ContractError("head payload has wrong type or exact length")
    mantissas = tuple(byte - 256 if byte >= 128 else byte for byte in payload[:64])
    exponents = tuple(
        int.from_bytes(payload[64 + 2 * group : 66 + 2 * group], "little", signed=True)
        for group in range(spec["group_count"])
    )
    encoded = (mantissas, exponents)
    validate_encoded_head(encoded, alternative_id)
    return encoded


def group_dot_terms(
    query: tuple[tuple[int, ...], tuple[int, ...]],
    key: tuple[tuple[int, ...], tuple[int, ...]],
    alternative_id: str,
) -> tuple[tuple[int, int], ...]:
    spec = _spec(alternative_id)
    validate_encoded_head(query, alternative_id)
    validate_encoded_head(key, alternative_id)
    q_mantissas, q_exponents = query
    k_mantissas, k_exponents = key
    group_size = spec["group_size"]
    terms = []
    partial_limit = group_size * 127 * 127
    for group in range(spec["group_count"]):
        start = group * group_size
        stop = start + group_size
        partial = sum(q_mantissas[lane] * k_mantissas[lane] for lane in range(start, stop))
        exponent = q_exponents[group] + k_exponents[group]
        if partial < -partial_limit or partial > partial_limit:
            raise AssertionError("group partial width bound violated")
        if exponent < MIN_SCORE_EXPONENT or exponent > MAX_SCORE_EXPONENT:
            raise AssertionError("group product exponent bound violated")
        terms.append((partial, exponent))
    return tuple(terms)


def _validated_mask(valid_mask: tuple[bool, ...], row_length: int) -> tuple[bool, ...]:
    if type(valid_mask) is not tuple or len(valid_mask) != row_length:
        raise ContractError("valid mask has wrong type or length")
    if any(type(bit) is not bool for bit in valid_mask):
        raise ContractError("valid mask entries must be exact booleans")
    if not any(valid_mask):
        raise ContractError("all-masked row")
    return valid_mask


def normalize_score_row(
    query: tuple[tuple[int, ...], tuple[int, ...]],
    keys: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    valid_mask: tuple[bool, ...],
    alternative_id: str,
) -> tuple[tuple[int, ...], int, int, int]:
    _spec(alternative_id)
    validate_encoded_head(query, alternative_id)
    if type(keys) is not tuple or not (1 <= len(keys) <= MAX_ROW_LENGTH):
        raise ContractError("keys must be an immutable row of length 1 through 512")
    for key in keys:
        validate_encoded_head(key, alternative_id)
    valid_mask = _validated_mask(valid_mask, len(keys))
    all_terms = tuple(group_dot_terms(query, key, alternative_id) for key in keys)
    nonzero_exponents = [
        exponent
        for key_index, terms in enumerate(all_terms)
        if valid_mask[key_index]
        for partial, exponent in terms
        if partial != 0
    ]
    common_exponent = min(nonzero_exponents) if nonzero_exponents else 0
    aligned_scores: list[int] = []
    for key_index, terms in enumerate(all_terms):
        if not valid_mask[key_index]:
            aligned_scores.append(0)
            continue
        aligned = sum(
            partial << (exponent - common_exponent)
            for partial, exponent in terms
            if partial != 0
        )
        if aligned < -(1 << 542) or aligned > (1 << 542) - 1:
            raise AssertionError("signed-543 aligned score bound violated")
        aligned_scores.append(aligned)
    argmax_index = next(index for index, valid in enumerate(valid_mask) if valid)
    row_max = aligned_scores[argmax_index]
    for index, valid in enumerate(valid_mask):
        if valid and aligned_scores[index] > row_max:
            row_max = aligned_scores[index]
            argmax_index = index
    output: list[int] = []
    saturation_count = 0
    for index, valid in enumerate(valid_mask):
        if not valid:
            output.append(0)
            continue
        centered = aligned_scores[index] - row_max
        if centered < -(1 << 543) or centered > (1 << 543) - 1:
            raise AssertionError("signed-544 centered score bound violated")
        if centered > 0:
            raise ContractError("positive centered score")
        realized = round_signed_pow2(centered, common_exponent + 17)
        if realized > 0:
            raise ContractError("positive realized score")
        if realized < INT64_MIN:
            realized = INT64_MIN
            saturation_count += 1
        output.append(realized)
    return (tuple(output), common_exponent, argmax_index, saturation_count)


def _pack_mask(valid_mask: tuple[bool, ...]) -> bytes:
    packed = bytearray((len(valid_mask) + 7) // 8)
    for index, valid in enumerate(valid_mask):
        if valid:
            packed[index // 8] |= 1 << (index % 8)
    return bytes(packed)


def _pack_normalized_frame(
    scores_q12_20: tuple[int, ...],
    common_exponent: int,
    argmax_index: int,
    valid_mask: tuple[bool, ...],
) -> bytes:
    if type(scores_q12_20) is not tuple or not (1 <= len(scores_q12_20) <= MAX_ROW_LENGTH):
        raise ContractError("scores must be an immutable row of length 1 through 512")
    valid_mask = _validated_mask(valid_mask, len(scores_q12_20))
    common_exponent = _exact_int(common_exponent, "common exponent")
    argmax_index = _exact_int(argmax_index, "argmax index")
    if common_exponent < MIN_SCORE_EXPONENT or common_exponent > MAX_SCORE_EXPONENT:
        raise ContractError("common exponent outside canonical range")
    if argmax_index < 0 or argmax_index >= len(scores_q12_20) or not valid_mask[argmax_index]:
        raise ContractError("argmax index is not a valid key")
    for index, score in enumerate(scores_q12_20):
        score = _exact_int(score, f"score[{index}]")
        if score < INT64_MIN or score > 0:
            raise ContractError("score outside signed-int64 nonpositive range")
        if not valid_mask[index] and score != 0:
            raise ContractError("masked score must be deterministic zero")
    if scores_q12_20[argmax_index] != 0:
        raise ContractError("argmax score must be canonical zero")
    header = bytearray(OUTPUT_MAGIC)
    header.extend((2).to_bytes(2, "little"))
    header.extend((0).to_bytes(2, "little"))
    header.extend(len(scores_q12_20).to_bytes(2, "little"))
    header.extend((8).to_bytes(2, "little"))
    header.extend(common_exponent.to_bytes(2, "little", signed=True))
    header.extend(argmax_index.to_bytes(2, "little"))
    header.extend((0).to_bytes(4, "little"))
    for score in scores_q12_20:
        header.extend(score.to_bytes(8, "little", signed=True))
    header.extend(_pack_mask(valid_mask))
    return bytes(header)


def pack_normalized_row(
    query: tuple[tuple[int, ...], tuple[int, ...]],
    keys: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    valid_mask: tuple[bool, ...],
    alternative_id: str,
) -> bytes:
    """Validate and normalize an encoded Q/K row before canonical packing."""
    scores, common_exponent, argmax_index, _ = normalize_score_row(
        query, keys, valid_mask, alternative_id
    )
    return _pack_normalized_frame(scores, common_exponent, argmax_index, valid_mask)


def unpack_normalized_row(payload: bytes) -> tuple[tuple[int, ...], int, int, tuple[bool, ...]]:
    if type(payload) is not bytes or len(payload) < 33:
        raise ContractError("normalized payload has wrong type or is too short")
    if payload[:8] != OUTPUT_MAGIC:
        raise ContractError("wrong output magic")
    if int.from_bytes(payload[8:10], "little") != 2:
        raise ContractError("wrong output schema version")
    if int.from_bytes(payload[10:12], "little") != 0:
        raise ContractError("nonzero output flags")
    row_length = int.from_bytes(payload[12:14], "little")
    if not (1 <= row_length <= MAX_ROW_LENGTH):
        raise ContractError("output row length outside range")
    if int.from_bytes(payload[14:16], "little") != 8:
        raise ContractError("wrong output score width")
    common_exponent = int.from_bytes(payload[16:18], "little", signed=True)
    argmax_index = int.from_bytes(payload[18:20], "little")
    if payload[20:24] != b"\x00\x00\x00\x00":
        raise ContractError("nonzero output reserved field")
    mask_bytes = (row_length + 7) // 8
    expected_length = 24 + 8 * row_length + mask_bytes
    if len(payload) != expected_length:
        raise ContractError("output payload length or trailing bytes invalid")
    score_offset = 24
    scores = tuple(
        int.from_bytes(payload[score_offset + 8 * index : score_offset + 8 * (index + 1)], "little", signed=True)
        for index in range(row_length)
    )
    mask_payload = payload[-mask_bytes:]
    if row_length % 8 and mask_payload[-1] >> (row_length % 8):
        raise ContractError("nonzero output mask tail bits")
    valid_mask = tuple(bool(mask_payload[index // 8] & (1 << (index % 8))) for index in range(row_length))
    if _pack_normalized_frame(scores, common_exponent, argmax_index, valid_mask) != payload:
        raise ContractError("noncanonical normalized payload")
    return (scores, common_exponent, argmax_index, valid_mask)


def decode_head_values(
    encoded: tuple[tuple[int, ...], tuple[int, ...]], alternative_id: str
) -> tuple[tuple[int, int], ...]:
    """Return exact dyadic lane values for synthetic conformance tests."""
    validate_encoded_head(encoded, alternative_id)
    mantissas, exponents = encoded
    group_size = ALTERNATIVES[alternative_id]["group_size"]
    return tuple((mantissa, exponents[lane // group_size]) for lane, mantissa in enumerate(mantissas))
