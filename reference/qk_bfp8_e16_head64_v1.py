"""NO_EXECUTION_AUTHORITY pure reference for QK_BFP8_E16_HEAD64_V1."""

AUTHORITY = "NO_EXECUTION_AUTHORITY"
CONTRACT_ID = "QK_BFP8_E16_HEAD64_V1"

HEAD_LANES = 64
MANTISSA_MIN = -127
MANTISSA_MAX = 127
RESERVED_MANTISSA = -128
HEAD_EXPONENT_MIN = -139
HEAD_EXPONENT_MAX = 122
SCORE_EXPONENT_MIN = -278
SCORE_EXPONENT_MAX = 244
LANE_PRODUCT_MIN = -16129
LANE_PRODUCT_MAX = 16129
SCORE_MANTISSA_MIN = -1032256
SCORE_MANTISSA_MAX = 1032256
HEAD_RECORD_BYTES = 66


class ReferenceViolation(ValueError):
    """A source word, encoded record, or arithmetic request is non-canonical."""


def _require(condition, message):
    if not condition:
        raise ReferenceViolation(message)


def _require_int(value, name):
    _require(type(value) is int, f"{name} must be an integer")


def bf16_components(word):
    """Return (negative, unsigned_significand, power_of_two) exactly."""

    _require_int(word, "BF16 word")
    _require(0 <= word <= 0xFFFF, "BF16 word is outside 16 bits")
    negative = bool(word & 0x8000)
    exponent_field = (word >> 7) & 0xFF
    fraction = word & 0x7F
    _require(exponent_field != 0xFF, "non-finite BF16 source value")
    if exponent_field == 0:
        return negative, fraction, -133
    return negative, 128 + fraction, exponent_field - 134


def _magnitude_leq(left_coefficient, left_exponent, right_coefficient, right_exponent):
    if left_exponent >= right_exponent:
        return (left_coefficient << (left_exponent - right_exponent)) <= right_coefficient
    return left_coefficient <= (right_coefficient << (right_exponent - left_exponent))


def select_head_exponent(words):
    """Select the least canonical exponent satisfying max_abs <= 127 * 2^e."""

    values = tuple(words)
    _require(len(values) == HEAD_LANES, "a head must contain exactly 64 BF16 words")
    components = tuple(bf16_components(word) for word in values)
    nonzero = tuple((significand, exponent) for _, significand, exponent in components if significand)
    if not nonzero:
        return 0
    for exponent in range(HEAD_EXPONENT_MIN, HEAD_EXPONENT_MAX + 1):
        if all(
            _magnitude_leq(significand, source_exponent, MANTISSA_MAX, exponent)
            for significand, source_exponent in nonzero
        ):
            return exponent
    raise ReferenceViolation("finite BF16 head has no canonical exponent")


def _round_unsigned_power_of_two(significand, shift):
    if shift >= 0:
        return significand << shift
    divisor = 1 << (-shift)
    quotient, remainder = divmod(significand, divisor)
    half = divisor >> 1
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    return quotient


def quantize_bf16(word, exponent):
    """Compute round-to-nearest-ties-to-even(BF16 * 2^-exponent)."""

    _require_int(exponent, "head exponent")
    _require(
        HEAD_EXPONENT_MIN <= exponent <= HEAD_EXPONENT_MAX,
        "head exponent is outside the canonical finite range",
    )
    negative, significand, source_exponent = bf16_components(word)
    magnitude = _round_unsigned_power_of_two(significand, source_exponent - exponent)
    value = -magnitude if negative and magnitude else magnitude
    _require(MANTISSA_MIN <= value <= MANTISSA_MAX, "quantized mantissa escaped canonical range")
    _require(value != RESERVED_MANTISSA, "reserved mantissa was produced")
    return value


def validate_head(mantissas, exponent):
    """Return a canonical immutable head or reject it."""

    values = tuple(mantissas)
    _require(len(values) == HEAD_LANES, "a head must contain exactly 64 mantissas")
    _require_int(exponent, "head exponent")
    for lane, value in enumerate(values):
        _require_int(value, f"mantissa[{lane}]")
        _require(MANTISSA_MIN <= value <= MANTISSA_MAX, f"mantissa[{lane}] is non-canonical")
        _require(value != RESERVED_MANTISSA, f"mantissa[{lane}] uses the reserved payload")
    if any(values):
        _require(
            HEAD_EXPONENT_MIN <= exponent <= HEAD_EXPONENT_MAX,
            "nonzero head exponent is non-canonical",
        )
    else:
        _require(exponent == 0, "all-zero head must use exponent zero")
    return values, exponent


def encode_head(words):
    """Encode one 64-lane finite BF16 Q or K head without clamping."""

    values = tuple(words)
    exponent = select_head_exponent(values)
    mantissas = tuple(quantize_bf16(word, exponent) for word in values)
    return validate_head(mantissas, exponent)


def decode_head_dyadics(mantissas, exponent):
    """Return exact reconstructed lanes as (signed_integer, power_of_two)."""

    values, canonical_exponent = validate_head(mantissas, exponent)
    return tuple((value, canonical_exponent) for value in values)


def pack_head(mantissas, exponent):
    """Pack 64 signed int8 lanes followed by a signed int16 little-endian exponent."""

    values, canonical_exponent = validate_head(mantissas, exponent)
    payload = bytes(value & 0xFF for value in values)
    return payload + canonical_exponent.to_bytes(2, byteorder="little", signed=True)


def unpack_head(payload):
    """Unpack one canonical 66-byte Q/K head record."""

    _require(isinstance(payload, (bytes, bytearray)), "packed head must be bytes")
    raw = bytes(payload)
    _require(len(raw) == HEAD_RECORD_BYTES, "packed head must contain exactly 66 bytes")
    mantissas = tuple(value - 256 if value >= 128 else value for value in raw[:HEAD_LANES])
    exponent = int.from_bytes(raw[HEAD_LANES:], byteorder="little", signed=True)
    return validate_head(mantissas, exponent)


def dot_product(q_mantissas, q_exponent, k_mantissas, k_exponent):
    """Return the exact represented pre-scale dot product as (mantissa, exponent)."""

    q_values, q_canonical_exponent = validate_head(q_mantissas, q_exponent)
    k_values, k_canonical_exponent = validate_head(k_mantissas, k_exponent)
    score_mantissa = sum(q * k for q, k in zip(q_values, k_values))
    score_exponent = q_canonical_exponent + k_canonical_exponent
    _require(
        SCORE_MANTISSA_MIN <= score_mantissa <= SCORE_MANTISSA_MAX,
        "64-lane score mantissa escaped its exact range",
    )
    _require(
        SCORE_EXPONENT_MIN <= score_exponent <= SCORE_EXPONENT_MAX,
        "score exponent escaped its exact range",
    )
    return score_mantissa, score_exponent


def storage_cost():
    """Return the frozen topology-dependent byte accounting."""

    layers = 24
    tokens = 512
    query_heads = 14
    key_value_heads = 2
    value_bytes_per_token_per_layer = 128
    baseline_key_head_bytes = 68
    contract_key_head_bytes = HEAD_RECORD_BYTES
    baseline_total_per_token_per_layer = (
        key_value_heads * baseline_key_head_bytes + value_bytes_per_token_per_layer
    )
    contract_total_per_token_per_layer = (
        key_value_heads * contract_key_head_bytes + value_bytes_per_token_per_layer
    )
    return {
        "baseline_total_bytes_for_24_layers_and_512_tokens": (
            baseline_total_per_token_per_layer * layers * tokens
        ),
        "baseline_total_bytes_per_token_per_layer": baseline_total_per_token_per_layer,
        "contract_total_bytes_for_24_layers_and_512_tokens": (
            contract_total_per_token_per_layer * layers * tokens
        ),
        "contract_total_bytes_per_token_per_layer": contract_total_per_token_per_layer,
        "delta_bytes_for_24_layers_and_512_tokens": (
            (contract_total_per_token_per_layer - baseline_total_per_token_per_layer)
            * layers
            * tokens
        ),
        "delta_bytes_per_token_across_24_layers": (
            contract_total_per_token_per_layer - baseline_total_per_token_per_layer
        )
        * layers,
        "delta_bytes_per_token_per_layer": (
            contract_total_per_token_per_layer - baseline_total_per_token_per_layer
        ),
        "query_transient_bytes_per_token_per_layer": query_heads * contract_key_head_bytes,
    }
