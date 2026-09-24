"""NO_EXECUTION_AUTHORITY pure score-pair normalization reference."""

AUTHORITY = "NO_EXECUTION_AUTHORITY"
CONTRACT_ID = "QK_BFP8_E16_HEAD64_V1_SCORE_NORMALIZATION_V2"

MAX_ROW_LENGTH = 512
SCORE_MANTISSA_MIN = -1032256
SCORE_MANTISSA_MAX = 1032256
SCORE_EXPONENT_MIN = -278
SCORE_EXPONENT_MAX = 244
MIN_Q12_20 = -(1 << 63)
MASKED_Q12_20 = 0
INPUT_MAGIC = b"QKSPIN01"
OUTPUT_MAGIC = b"QKSPOU01"
SCHEMA_VERSION = 2


class ReferenceViolation(ValueError):
    """A request or packed row violates the frozen normalization contract."""


def _require(condition, message):
    if not condition:
        raise ReferenceViolation(message)


def _require_int(value, name):
    _require(type(value) is int, f"{name} must be an exact integer")


def _validate_score_pair_fields(pair, name="score pair"):
    _require(type(pair) is tuple and len(pair) == 2, f"{name} must be a two-integer tuple")
    mantissa, exponent = pair
    _require_int(mantissa, f"{name} mantissa")
    _require_int(exponent, f"{name} exponent")
    _require(
        SCORE_MANTISSA_MIN <= mantissa <= SCORE_MANTISSA_MAX,
        f"{name} mantissa is outside the exact score bound",
    )
    _require(
        SCORE_EXPONENT_MIN <= exponent <= SCORE_EXPONENT_MAX,
        f"{name} exponent is outside the exact producer range",
    )
    return mantissa, exponent


def canonicalize_producer_score_pair(pair):
    """Canonicalize one exact bound-producer score pair at the interface."""

    mantissa, exponent = _validate_score_pair_fields(pair, "producer score pair")
    return (0, 0) if mantissa == 0 else (mantissa, exponent)


def _validate_canonical_score_pair(pair, name="score pair"):
    mantissa, exponent = _validate_score_pair_fields(pair, name)
    if mantissa == 0:
        _require(exponent == 0, f"{name} zero must use exponent zero")
    return mantissa, exponent


def _validate_row(score_pairs, valid_mask):
    _require(type(score_pairs) is tuple, "score_pairs must be an immutable tuple")
    _require(type(valid_mask) is tuple, "valid_mask must be an immutable tuple")
    _require(1 <= len(score_pairs) <= MAX_ROW_LENGTH, "row length must be 1 through 512")
    _require(len(valid_mask) == len(score_pairs), "mask length differs from score row length")
    canonical = []
    for index, pair in enumerate(score_pairs):
        try:
            canonical.append(canonicalize_producer_score_pair(pair))
        except ReferenceViolation as error:
            raise ReferenceViolation(f"score_pairs[{index}] {error}") from error
    for index, valid in enumerate(valid_mask):
        _require(type(valid) is bool, f"valid_mask[{index}] must be an exact boolean")
    _require(any(valid_mask), "all-masked rows are rejected")
    return tuple(canonical)


def compare_score_pairs(left_pair, right_pair):
    """Return exact represented-value ordering as -1, 0, or +1."""

    left_mantissa, left_exponent = canonicalize_producer_score_pair(left_pair)
    right_mantissa, right_exponent = canonicalize_producer_score_pair(right_pair)
    if left_mantissa == 0 and right_mantissa == 0:
        return 0
    if left_mantissa < 0 <= right_mantissa:
        return -1
    if right_mantissa < 0 <= left_mantissa:
        return 1
    if left_mantissa == 0:
        return -1 if right_mantissa > 0 else 1
    if right_mantissa == 0:
        return 1 if left_mantissa > 0 else -1
    common_exponent = min(left_exponent, right_exponent)
    left_value = left_mantissa << (left_exponent - common_exponent)
    right_value = right_mantissa << (right_exponent - common_exponent)
    return -1 if left_value < right_value else 1 if left_value > right_value else 0


def _round_shift_even_signed(value, shift):
    _require_int(value, "rounding value")
    _require_int(shift, "rounding shift")
    _require(shift > 0, "rounding shift must be positive")
    negative = value < 0
    magnitude = -value if negative else value
    retained = magnitude >> shift
    remainder = magnitude - (retained << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (retained & 1)):
        retained += 1
    return -retained if negative else retained


def _realize_centered_q12_20(centered, common_exponent):
    _require_int(centered, "centered score")
    _require(centered <= 0, "centered score became positive")
    binary_shift = common_exponent + 17
    if binary_shift >= 0:
        realized = centered << binary_shift
    else:
        realized = _round_shift_even_signed(centered, -binary_shift)
    _require(realized <= 0, "realized score became positive")
    return max(MIN_Q12_20, realized)


def normalize_score_row(score_pairs, valid_mask):
    """Return (scores_q12_20, common_exponent, lowest-index exact argmax)."""

    canonical = _validate_row(score_pairs, valid_mask)
    nonzero_exponents = [
        exponent
        for (mantissa, exponent), valid in zip(canonical, valid_mask)
        if valid and mantissa != 0
    ]
    common_exponent = min(nonzero_exponents) if nonzero_exponents else 0
    aligned = []
    for (mantissa, exponent), valid in zip(canonical, valid_mask):
        if not valid:
            aligned.append(None)
        elif mantissa == 0:
            aligned.append(0)
        else:
            aligned.append(mantissa << (exponent - common_exponent))

    argmax_index = next(index for index, valid in enumerate(valid_mask) if valid)
    row_max = aligned[argmax_index]
    for index, valid in enumerate(valid_mask):
        if valid and aligned[index] > row_max:
            argmax_index = index
            row_max = aligned[index]

    scores_q12_20 = []
    for index, valid in enumerate(valid_mask):
        if not valid:
            scores_q12_20.append(MASKED_Q12_20)
            continue
        centered = aligned[index] - row_max
        scores_q12_20.append(_realize_centered_q12_20(centered, common_exponent))
    _require(scores_q12_20[argmax_index] == 0, "argmax did not realize to exact zero")
    return tuple(scores_q12_20), common_exponent, argmax_index


def _pack_mask(valid_mask):
    packed = bytearray((len(valid_mask) + 7) // 8)
    for index, valid in enumerate(valid_mask):
        if valid:
            packed[index // 8] |= 1 << (index % 8)
    return bytes(packed)


def _unpack_mask(payload, row_length):
    mask_bytes = (row_length + 7) // 8
    _require(len(payload) == mask_bytes, "mask payload length differs")
    if row_length % 8:
        used = (1 << (row_length % 8)) - 1
        _require(payload[-1] & ~used == 0, "unused high mask bits must be zero")
    return tuple(bool(payload[index // 8] & (1 << (index % 8))) for index in range(row_length))


def pack_input_row(score_pairs, valid_mask):
    canonical = _validate_row(score_pairs, valid_mask)
    header = (
        INPUT_MAGIC
        + SCHEMA_VERSION.to_bytes(2, "little")
        + (0).to_bytes(2, "little")
        + len(canonical).to_bytes(2, "little")
        + (6).to_bytes(2, "little")
    )
    records = b"".join(
        mantissa.to_bytes(4, "little", signed=True)
        + exponent.to_bytes(2, "little", signed=True)
        for mantissa, exponent in canonical
    )
    return header + records + _pack_mask(valid_mask)


def unpack_input_row(payload):
    _require(type(payload) is bytes, "input payload must be immutable bytes")
    _require(len(payload) >= 17, "input payload is shorter than one-row framing")
    _require(payload[:8] == INPUT_MAGIC, "input magic differs")
    _require(int.from_bytes(payload[8:10], "little") == SCHEMA_VERSION, "input schema differs")
    _require(int.from_bytes(payload[10:12], "little") == 0, "input flags must be zero")
    row_length = int.from_bytes(payload[12:14], "little")
    _require(1 <= row_length <= MAX_ROW_LENGTH, "input row length is outside 1 through 512")
    _require(int.from_bytes(payload[14:16], "little") == 6, "input record_bytes differs")
    mask_bytes = (row_length + 7) // 8
    expected_length = 16 + 6 * row_length + mask_bytes
    _require(len(payload) == expected_length, "input payload length or trailing bytes differ")
    score_pairs = []
    offset = 16
    for index in range(row_length):
        mantissa = int.from_bytes(payload[offset : offset + 4], "little", signed=True)
        exponent = int.from_bytes(payload[offset + 4 : offset + 6], "little", signed=True)
        score_pairs.append(
            _validate_canonical_score_pair((mantissa, exponent), f"packed score[{index}]")
        )
        offset += 6
    valid_mask = _unpack_mask(payload[offset:], row_length)
    canonical = _validate_row(tuple(score_pairs), valid_mask)
    return canonical, valid_mask


def pack_normalized_row(score_pairs, valid_mask):
    scores_q12_20, common_exponent, argmax_index = normalize_score_row(score_pairs, valid_mask)
    header = (
        OUTPUT_MAGIC
        + SCHEMA_VERSION.to_bytes(2, "little")
        + (0).to_bytes(2, "little")
        + len(scores_q12_20).to_bytes(2, "little")
        + (8).to_bytes(2, "little")
        + common_exponent.to_bytes(2, "little", signed=True)
        + argmax_index.to_bytes(2, "little")
        + (0).to_bytes(4, "little")
    )
    scores = b"".join(score.to_bytes(8, "little", signed=True) for score in scores_q12_20)
    return header + scores + _pack_mask(valid_mask)


def unpack_normalized_row(payload):
    _require(type(payload) is bytes, "output payload must be immutable bytes")
    _require(len(payload) >= 27, "output payload is shorter than one-row framing")
    _require(payload[:8] == OUTPUT_MAGIC, "output magic differs")
    _require(int.from_bytes(payload[8:10], "little") == SCHEMA_VERSION, "output schema differs")
    _require(int.from_bytes(payload[10:12], "little") == 0, "output flags must be zero")
    row_length = int.from_bytes(payload[12:14], "little")
    _require(1 <= row_length <= MAX_ROW_LENGTH, "output row length is outside 1 through 512")
    _require(int.from_bytes(payload[14:16], "little") == 8, "output score_bytes differs")
    common_exponent = int.from_bytes(payload[16:18], "little", signed=True)
    _require(
        SCORE_EXPONENT_MIN <= common_exponent <= SCORE_EXPONENT_MAX,
        "output common exponent is outside the canonical range",
    )
    argmax_index = int.from_bytes(payload[18:20], "little")
    _require(argmax_index < row_length, "output argmax index is outside the row")
    _require(int.from_bytes(payload[20:24], "little") == 0, "output reserved field must be zero")
    mask_bytes = (row_length + 7) // 8
    expected_length = 24 + 8 * row_length + mask_bytes
    _require(len(payload) == expected_length, "output payload length or trailing bytes differ")
    scores_q12_20 = tuple(
        int.from_bytes(payload[24 + 8 * index : 32 + 8 * index], "little", signed=True)
        for index in range(row_length)
    )
    valid_mask = _unpack_mask(payload[24 + 8 * row_length :], row_length)
    _require(any(valid_mask), "packed output cannot be all masked")
    _require(valid_mask[argmax_index], "packed output argmax must be valid")
    _require(scores_q12_20[argmax_index] == 0, "packed output argmax score must be zero")
    for index, (score, valid) in enumerate(zip(scores_q12_20, valid_mask)):
        if valid:
            _require(
                MIN_Q12_20 <= score <= 0,
                f"packed valid score[{index}] is outside the Q12.20 softmax range",
            )
        else:
            _require(score == MASKED_Q12_20, f"packed masked score[{index}] lacks zero filler")
    return scores_q12_20, valid_mask, common_exponent, argmax_index
