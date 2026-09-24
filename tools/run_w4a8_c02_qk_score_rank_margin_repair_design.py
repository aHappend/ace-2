#!/usr/bin/env python3
"""Execute the frozen c02 Q/K rank-margin PTQ repair-design alternatives.

The campaign is diagnostic-only.  It reconstructs the immutable c01 model in
memory, captures the exact frozen C4 calibration slice, fits both preregistered
alternatives, evaluates the frozen validation split and sealed diagnostic, and
runs the complete procedure twice in fresh processes.  It never writes below a
candidate/attempt namespace and never mutates the sealed c01 artifacts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers.models.qwen2 import modeling_qwen2


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import ace2_full_model_fixed_point as fixed  # noqa: E402
import diagnose_w4a8_c02_numerical_path as coarse  # noqa: E402
import run_w4a8_full_model_software_contract as frozen  # noqa: E402
import verify_w4a8_c02_qk_score_rank_margin_repair_design as static_preflight  # noqa: E402
import w4a8_full_model_evaluator as evaluator  # noqa: E402


TASK_PATH = ROOT / "design/W4A8_C02_QK_SCORE_RANK_MARGIN_REPAIR_DESIGN_V1_TASK.json"
DEFAULT_OUTPUT = ROOT / "evidence/diagnostics/w4a8-c02-qk-score-rank-margin-repair-design-v1"
SCRIPT_PATH = Path(__file__).resolve()
MODEL_ORDER = ("base", "checkpoint-176")
ALTERNATIVE_ORDER = (
    "qk_s4_residual_cross_term_scale32_v1",
    "qk_group8_diagonal_scale32_rebalance_v1",
)
RUN_NAMES = ("run-0001", "run-0002")
RUN_MEMBERS = (
    "run-contract.json",
    "base/calibration-capture-manifest.json",
    "base/alternative-results.json",
    "checkpoint-176/calibration-capture-manifest.json",
    "checkpoint-176/alternative-results.json",
    "selection.json",
    "SHA256SUMS",
)
HEAD_DIM = 64
QUERY_HEADS = 14
KV_HEADS = 2
GROUP_COUNT = 8
GROUP_SIZE = 8
SCORE_FRAC = fixed.ATTENTION_SCORE_FRAC
Q20_44_TO_Q6_9_SHIFT = fixed.TAGGED_SCORE_FRAC - SCORE_FRAC
WIDE_LIMB_BITS = 30
WIDE_LIMB_BASE = 1 << WIDE_LIMB_BITS
WIDE_LIMB_MASK = WIDE_LIMB_BASE - 1
WIDE_LIMBS = 7
WIDE_SAFE_SIGNED_BITS = (WIDE_LIMBS - 1) * WIDE_LIMB_BITS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return coarse.tensor_sha256(value.detach().cpu().contiguous())


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def scale32_value(record: int | torch.Tensor) -> float | torch.Tensor:
    if isinstance(record, int):
        return float(coarse._scale32_values(torch.tensor(record, dtype=torch.int64)))
    return coarse._scale32_values(record.to(torch.int64))


def packed_scale32(value: float) -> int:
    require(math.isfinite(value) and value >= 0.0, "Scale32 fit must be finite and nonnegative")
    record = int(
        fixed.ceil_scale32_from_ratio(0, 1)
        if value == 0.0
        else fixed.ceil_scale32_from_float(value)
    )
    realized = float(scale32_value(record))
    require(realized >= value, "ceil Scale32 fit did not bound the requested value")
    return record


def float_hex(value: float) -> str:
    return float(value).hex()


def reshape_heads(value: torch.Tensor, heads: int) -> torch.Tensor:
    require(value.ndim == 3, "projection tensor must be [batch, sequence, channels]")
    require(value.shape[-1] == heads * HEAD_DIM, "projection channel geometry differs")
    return value.reshape(value.shape[0], value.shape[1], heads, HEAD_DIM).transpose(1, 2).contiguous()


def causal_mask(sequence: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.ones(sequence, sequence, dtype=torch.bool, device=device).tril()
    attention_mask = torch.where(
        valid,
        torch.zeros((), dtype=torch.float32, device=device),
        torch.full((), float("-inf"), dtype=torch.float32, device=device),
    ).reshape(1, 1, sequence, sequence)
    return valid.reshape(1, 1, sequence, sequence), attention_mask


def repeat_kv(value: torch.Tensor) -> torch.Tensor:
    return fixed.repeat_kv(value, QUERY_HEADS // KV_HEADS)


def score32_from_q20_44(centered: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, int]:
    require(centered.dtype == torch.int64, "Q20.44 score tensor must be signed int64")
    rounded = fixed.round_shift_even(centered, Q20_44_TO_Q6_9_SHIFT)
    active = valid.expand_as(rounded)
    clamp_events = int(((rounded < -32768) | (rounded > 0))[active].sum())
    realized = torch.where(
        active,
        rounded.clamp(-32768, 0),
        torch.full_like(rounded, -32768),
    )
    return realized.to(torch.int16), clamp_events


def center_q20_44(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    active = valid.expand_as(scores)
    maximum = torch.where(
        active,
        scores,
        torch.full_like(scores, torch.iinfo(torch.int64).min),
    ).amax(dim=-1, keepdim=True)
    centered = torch.where(active, scores - maximum, torch.zeros_like(scores))
    require(not torch.any(centered[active] > 0), "centered Q20.44 score became positive")
    return centered


def scale32_pair_to_q20_44(
    dots: torch.Tensor,
    query_records: torch.Tensor,
    key_records: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """Convert per-query/per-key Scale32 dot products to signed Q20.44."""
    require(dots.dtype == torch.int64 and dots.ndim == 4, "dot tensor geometry differs")
    require(query_records.shape == dots.shape[:3], "query Scale32 geometry differs")
    require(key_records.shape == (dots.shape[0], dots.shape[1], dots.shape[3]), "key Scale32 geometry differs")
    qsig = torch.bitwise_and(query_records, 0xFFFF).unsqueeze(-1)
    ksig = torch.bitwise_and(key_records, 0xFFFF).unsqueeze(-2)
    qexp_u8 = torch.bitwise_and(torch.bitwise_right_shift(query_records, 16), 0xFF)
    kexp_u8 = torch.bitwise_and(torch.bitwise_right_shift(key_records, 16), 0xFF)
    qexp = torch.where(qexp_u8 >= 128, qexp_u8 - 256, qexp_u8).unsqueeze(-1)
    kexp = torch.where(kexp_u8 >= 128, kexp_u8 - 256, kexp_u8).unsqueeze(-2)
    product = dots * qsig * ksig
    shift = qexp + kexp + 11
    positive = shift.clamp(min=0)
    negative = (-shift).clamp(min=0)
    limit = torch.bitwise_right_shift(
        torch.full_like(product, torch.iinfo(torch.int64).max),
        positive.clamp(max=62),
    )
    overflow = (shift > 62) | (product > limit) | (product < -limit - 1)
    overflow_events = int(overflow.sum())
    require(overflow_events == 0, "Scale32 Q20.44 conversion exceeded signed int64")
    shifted = torch.bitwise_left_shift(product, positive)
    converted = torch.where(shift >= 0, shifted, fixed.round_shift_even(product, negative))
    maximum_magnitude = int(converted.abs().max())
    maximum_width = max(1, maximum_magnitude.bit_length() + 1)
    return converted, overflow_events, maximum_width


def scale32_components(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode legal Scale32 records without passing through floating point."""
    packed = records.to(torch.int64)
    significand = torch.bitwise_and(packed, 0xFFFF)
    exponent_u8 = torch.bitwise_and(torch.bitwise_right_shift(packed, 16), 0xFF)
    exponent = torch.where(exponent_u8 >= 128, exponent_u8 - 256, exponent_u8)
    require(not torch.any(torch.bitwise_right_shift(packed, 24) != 0), "Scale32 reserved byte is nonzero")
    require(not torch.any((significand < 0x8000) | (significand > 0xFFFF)), "Scale32 significand is not normalized")
    require(not torch.any((exponent < -24) | (exponent > 4)), "Scale32 exponent is outside the frozen range")
    return significand, exponent


def wide_normalize(coefficients: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Normalize signed base-2^30 coefficients with a signed top limb."""
    require(len(coefficients) == WIDE_LIMBS, "wide coefficient count differs")
    normalized = [coefficient.to(torch.int64) for coefficient in coefficients]
    for index in range(WIDE_LIMBS - 1):
        carry = torch.div(normalized[index], WIDE_LIMB_BASE, rounding_mode="floor")
        normalized[index] = normalized[index] - carry * WIDE_LIMB_BASE
        normalized[index + 1] = normalized[index + 1] + carry
    return tuple(normalized)


def wide_multiply_small(value: tuple[torch.Tensor, ...], factor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    require(not torch.any((factor < 0) | (factor > 0xFFFF)), "wide small multiplier is outside unsigned 16-bit")
    return wide_normalize([digit * factor for digit in value])


def wide_left_shift_small(value: tuple[torch.Tensor, ...], shift: torch.Tensor) -> tuple[torch.Tensor, ...]:
    require(not torch.any((shift < 0) | (shift >= WIDE_LIMB_BITS)), "wide small left shift is outside one limb")
    factor = torch.bitwise_left_shift(torch.ones_like(shift), shift)
    return wide_normalize([digit * factor for digit in value])


def wide_subtract(left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return wide_normalize([left_digit - right_digit for left_digit, right_digit in zip(left, right, strict=True)])


def wide_negate(value: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return wide_normalize([-digit for digit in value])


def wide_row_max(value: tuple[torch.Tensor, ...], valid: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return an exact lexicographic row maximum for normalized wide integers."""
    active = valid.expand_as(value[0])
    top_maximum = torch.where(
        active,
        value[-1],
        torch.full_like(value[-1], torch.iinfo(torch.int64).min),
    ).amax(dim=-1, keepdim=True)
    candidates = active & (value[-1] == top_maximum)
    for digit in reversed(value[:-1]):
        maximum = torch.where(candidates, digit, torch.full_like(digit, -1)).amax(dim=-1, keepdim=True)
        candidates &= digit == maximum
    index = torch.argmax(candidates.to(torch.int64), dim=-1, keepdim=True)
    return tuple(torch.gather(digit, -1, index) for digit in value)


def wide_select_limb(value: tuple[torch.Tensor, ...], index: torch.Tensor) -> torch.Tensor:
    selected = torch.zeros_like(value[0])
    for limb, digit in enumerate(value):
        selected = torch.where(index == limb, digit, selected)
    return selected


def wide_extract_low16(value: tuple[torch.Tensor, ...], start: torch.Tensor) -> torch.Tensor:
    limb = torch.div(start, WIDE_LIMB_BITS, rounding_mode="floor")
    offset = torch.remainder(start, WIDE_LIMB_BITS)
    current = wide_select_limb(value, limb)
    following = wide_select_limb(value, limb + 1)
    combined = torch.bitwise_right_shift(current, offset) | torch.bitwise_left_shift(following, WIDE_LIMB_BITS - offset)
    return torch.bitwise_and(combined, 0xFFFF)


def wide_extract_bit(value: tuple[torch.Tensor, ...], position: torch.Tensor) -> torch.Tensor:
    limb = torch.div(position, WIDE_LIMB_BITS, rounding_mode="floor")
    offset = torch.remainder(position, WIDE_LIMB_BITS)
    return torch.bitwise_and(torch.bitwise_right_shift(wide_select_limb(value, limb), offset), 1).to(torch.bool)


def wide_any_bits_at_or_above(value: tuple[torch.Tensor, ...], start: torch.Tensor) -> torch.Tensor:
    limb = torch.div(start, WIDE_LIMB_BITS, rounding_mode="floor")
    offset = torch.remainder(start, WIDE_LIMB_BITS)
    observed = torch.bitwise_right_shift(wide_select_limb(value, limb), offset) != 0
    for candidate, digit in enumerate(value):
        observed |= (limb < candidate) & (digit != 0)
    return observed


def wide_any_bits_below(value: tuple[torch.Tensor, ...], stop: torch.Tensor) -> torch.Tensor:
    limb = torch.div(stop, WIDE_LIMB_BITS, rounding_mode="floor")
    offset = torch.remainder(stop, WIDE_LIMB_BITS)
    mask = torch.bitwise_left_shift(torch.ones_like(offset), offset) - 1
    observed = torch.bitwise_and(wide_select_limb(value, limb), mask) != 0
    for candidate, digit in enumerate(value):
        observed |= (limb > candidate) & (digit != 0)
    return observed


def wide_nonpositive_to_q6_9(
    value: tuple[torch.Tensor, ...],
    right_shift: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Round exact non-positive dyadics to Q6.9 with ties to even."""
    lower_nonzero = torch.zeros_like(value[0], dtype=torch.bool)
    positive = value[-1] > 0
    for digit in value[:-1]:
        lower_nonzero |= digit != 0
    positive |= (value[-1] == 0) & lower_nonzero
    require(not torch.any(positive), "centered group8 score became positive")
    require(not torch.any(right_shift <= 0), "group8 Q6.9 conversion unexpectedly requires a left shift")

    magnitude = wide_negate(value)
    require(not torch.any(magnitude[-1] < 0), "group8 magnitude exceeded the declared wide representation")
    quotient = wide_extract_low16(magnitude, right_shift)
    high_nonzero = wide_any_bits_at_or_above(magnitude, right_shift + 16)
    half = wide_extract_bit(magnitude, right_shift - 1)
    below_half = wide_any_bits_below(magnitude, right_shift - 1)
    increment = half & (below_half | ((quotient & 1) == 1))
    rounded = torch.where(high_nonzero, torch.full_like(quotient, 65536), quotient + increment.to(torch.int64))

    active = valid.expand_as(rounded)
    clamp_events = int((active & (rounded > 32768)).sum())
    realized = torch.where(
        active,
        -rounded.clamp(max=32768),
        torch.full_like(rounded, -32768),
    )
    return realized.to(torch.int16), clamp_events


def exact_group8_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    query_base_records: torch.Tensor,
    key_base_records: torch.Tensor,
    query_gain_records: torch.Tensor,
    key_gain_records: torch.Tensor,
    valid: torch.Tensor,
    safety: SafetyCounters,
) -> torch.Tensor:
    """Accumulate all group partial products exactly before Q6.9 realization."""
    require(query.dtype == torch.int8 and key.dtype == torch.int8, "group8 Q/K tensors must be signed int8")
    require(query.shape == key.shape and query.shape[-1] == HEAD_DIM, "group8 Q/K geometry differs")
    require(query_base_records.shape == query.shape[:3], "group8 query Scale32 geometry differs")
    require(key_base_records.shape == key.shape[:3], "group8 key Scale32 geometry differs")
    require(query_gain_records.shape == (QUERY_HEADS, GROUP_COUNT), "group8 query-gain geometry differs")
    require(key_gain_records.shape == (QUERY_HEADS, GROUP_COUNT), "group8 key-gain geometry differs")

    query_sig, query_exp = scale32_components(query_base_records)
    key_sig, key_exp = scale32_components(key_base_records)
    query_gain_sig, query_gain_exp = scale32_components(query_gain_records)
    key_gain_sig, key_gain_exp = scale32_components(key_gain_records)
    gain_sig_product = query_gain_sig * key_gain_sig
    gain_exp_sum = query_gain_exp + key_gain_exp
    gain_exp_minimum = gain_exp_sum.amin(dim=-1)
    gain_shift = gain_exp_sum - gain_exp_minimum.unsqueeze(-1)
    require(not torch.any((gain_shift < 0) | (gain_shift > 2 * 28)), "group8 gain exponent alignment differs")

    sequence = query.shape[2]
    score_shape = (query.shape[0], query.shape[1], sequence, sequence)
    accumulator = [torch.zeros(score_shape, dtype=torch.int64, device=query.device) for _ in range(WIDE_LIMBS)]
    query_grouped = query.reshape(query.shape[0], QUERY_HEADS, sequence, GROUP_COUNT, GROUP_SIZE)
    key_grouped = key.reshape(key.shape[0], QUERY_HEADS, sequence, GROUP_COUNT, GROUP_SIZE)
    for group in range(GROUP_COUNT):
        dots = torch.matmul(
            query_grouped[..., group, :].to(torch.int32),
            key_grouped[..., group, :].to(torch.int32).transpose(2, 3),
        ).to(torch.int64)
        magnitude = dots.abs() * gain_sig_product[:, group].reshape(1, QUERY_HEADS, 1, 1)
        sign = torch.sign(dots)
        shift = gain_shift[:, group].reshape(1, QUERY_HEADS, 1, 1)
        offset = torch.div(shift, WIDE_LIMB_BITS, rounding_mode="floor")
        residual_shift = torch.remainder(shift, WIDE_LIMB_BITS)
        chunk0 = torch.bitwise_and(magnitude, WIDE_LIMB_MASK)
        chunk1 = torch.bitwise_right_shift(magnitude, WIDE_LIMB_BITS)
        shifted0 = torch.bitwise_left_shift(chunk0, residual_shift)
        shifted1 = torch.bitwise_left_shift(chunk1, residual_shift)
        parts = (
            torch.bitwise_and(shifted0, WIDE_LIMB_MASK),
            torch.bitwise_right_shift(shifted0, WIDE_LIMB_BITS) + torch.bitwise_and(shifted1, WIDE_LIMB_MASK),
            torch.bitwise_right_shift(shifted1, WIDE_LIMB_BITS),
        )
        for base_limb in (0, 1):
            selected = offset == base_limb
            for part_index, part in enumerate(parts):
                accumulator[base_limb + part_index] += torch.where(selected, sign * part, torch.zeros_like(part))
    accumulated = wide_normalize(accumulator)

    key_exp_minimum = key_exp.amin(dim=-1, keepdim=True)
    key_shift = key_exp - key_exp_minimum
    require(not torch.any((key_shift < 0) | (key_shift >= WIDE_LIMB_BITS)), "group8 key exponent alignment exceeds one limb")
    key_weighted = wide_multiply_small(accumulated, key_sig.unsqueeze(-2))
    key_weighted = wide_left_shift_small(key_weighted, key_shift.unsqueeze(-2))
    maximum = wide_row_max(key_weighted, valid)
    centered = wide_subtract(key_weighted, maximum)
    active = valid.expand_as(centered[0])
    centered = tuple(torch.where(active, digit, torch.zeros_like(digit)) for digit in centered)
    fully_weighted = wide_multiply_small(centered, query_sig.unsqueeze(-1))

    maximum_group_dot = GROUP_SIZE * 127 * 127
    widths = []
    for head in range(QUERY_HEADS):
        aligned_group_bound = sum(
            maximum_group_dot * int(gain_sig_product[head, group]) << int(gain_shift[head, group])
            for group in range(GROUP_COUNT)
        )
        maximum_key_factor = max(
            int(key_sig[0, head, position]) << int(key_shift[0, head, position])
            for position in range(sequence)
        )
        maximum_query_sig = int(query_sig[0, head].max())
        centered_bound = 2 * aligned_group_bound * maximum_key_factor * maximum_query_sig
        widths.append(max(1, centered_bound.bit_length() + 1))
    capacity_events = sum(width > WIDE_SAFE_SIGNED_BITS for width in widths)
    safety.overflow_events += capacity_events
    require(capacity_events == 0, "group8 exact accumulator exceeds its declared signed width")
    safety.merge_width(max(widths))

    right_shift = 54 - (
        query_exp.unsqueeze(-1)
        + gain_exp_minimum.reshape(1, QUERY_HEADS, 1, 1)
        + key_exp_minimum.reshape(1, QUERY_HEADS, 1, 1)
    )
    scores, clamps = wide_nonpositive_to_q6_9(fully_weighted, right_shift, valid)
    safety.clamp_events += clamps
    return scores


def rotate_residual(
    residual: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    require(residual.dtype == torch.int8 and residual.shape[-1] == HEAD_DIM, "residual geometry differs")
    cos_q15 = torch.round(cosine.to(torch.float64) * 32767.0).clamp(-32768, 32767).to(torch.int64).unsqueeze(1)
    sin_q15 = torch.round(sine.to(torch.float64) * 32767.0).clamp(-32768, 32767).to(torch.int64).unsqueeze(1)
    values = residual.to(torch.int64)
    first, second = values[..., :32], values[..., 32:]
    first_acc = first * cos_q15[..., :32] - second * sin_q15[..., :32]
    second_acc = second * cos_q15[..., 32:] + first * sin_q15[..., 32:]
    overflow_events = int(
        ((first_acc < -(1 << 21)) | (first_acc >= (1 << 21))).sum()
        + ((second_acc < -(1 << 21)) | (second_acc >= (1 << 21))).sum()
    )
    require(overflow_events == 0, "residual RoPE exceeded signed-22 staging")
    rotated = torch.cat(
        (fixed.round_shift_even(first_acc, 15), fixed.round_shift_even(second_acc, 15)),
        dim=-1,
    )
    output_overflow = int(((rotated < -128) | (rotated > 127)).sum())
    require(output_overflow == 0, "residual RoPE exceeded signed int8")
    maximum_magnitude = max(int(first_acc.abs().max()), int(second_acc.abs().max()))
    width = max(1, maximum_magnitude.bit_length() + 1)
    return rotated.to(torch.int8), overflow_events + output_overflow, width


@dataclass
class MetricAccumulator:
    count: int = 0
    candidate_zero_count: int = 0
    sum_reference_squared: float = 0.0
    sum_candidate_squared: float = 0.0
    sum_error_squared: float = 0.0
    sum_dot: float = 0.0
    maximum_absolute_error: float = 0.0
    reference_absmax: float = 0.0
    candidate_absmax: float = 0.0

    def update(self, reference: torch.Tensor, candidate: torch.Tensor, valid: torch.Tensor) -> int:
        active = valid.expand_as(reference)
        ref = reference[active].detach().to(torch.float64)
        observed = candidate[active].detach().to(torch.float64)
        require(ref.shape == observed.shape, "metric comparison geometry differs")
        finite = torch.isfinite(ref) & torch.isfinite(observed)
        non_finite = int((~finite).sum())
        if not torch.all(finite):
            ref = ref[finite]
            observed = observed[finite]
        if ref.numel() == 0:
            return non_finite
        error = observed - ref
        self.count += ref.numel()
        self.candidate_zero_count += int((observed == 0).sum())
        self.sum_reference_squared += float(torch.sum(ref * ref))
        self.sum_candidate_squared += float(torch.sum(observed * observed))
        self.sum_error_squared += float(torch.sum(error * error))
        self.sum_dot += float(torch.sum(ref * observed))
        self.maximum_absolute_error = max(self.maximum_absolute_error, float(error.abs().max()))
        self.reference_absmax = max(self.reference_absmax, float(ref.abs().max()))
        self.candidate_absmax = max(self.candidate_absmax, float(observed.abs().max()))
        return non_finite

    def raw(self) -> dict[str, Any]:
        return {
            "candidate_absmax": self.candidate_absmax,
            "candidate_zero_count": self.candidate_zero_count,
            "count": self.count,
            "maximum_absolute_error": self.maximum_absolute_error,
            "reference_absmax": self.reference_absmax,
            "sum_candidate_squared": self.sum_candidate_squared,
            "sum_dot": self.sum_dot,
            "sum_error_squared": self.sum_error_squared,
            "sum_reference_squared": self.sum_reference_squared,
        }

    @staticmethod
    def metrics_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
        count = int(raw["count"])
        require(count > 0, "metric accumulator is empty")
        ref_energy = raw["sum_reference_squared"] / count
        error_energy = raw["sum_error_squared"] / count
        relative_rmse = math.sqrt(error_energy) / math.sqrt(ref_energy) if ref_energy > 0 else (0.0 if error_energy == 0 else math.inf)
        denominator = math.sqrt(raw["sum_reference_squared"] * raw["sum_candidate_squared"])
        cosine = raw["sum_dot"] / denominator if denominator > 0 else (1.0 if raw["sum_error_squared"] == 0 else 0.0)
        severity = "TRACKING"
        if relative_rmse >= 1.0 or cosine <= 0.5:
            severity = "SEVERE"
        elif relative_rmse >= 0.25 or cosine <= 0.9:
            severity = "MATERIAL"
        return {
            "candidate_absmax": raw["candidate_absmax"],
            "candidate_zero_fraction": raw["candidate_zero_count"] / count,
            "cosine_similarity": cosine,
            "maximum_absolute_error": raw["maximum_absolute_error"],
            "reference_absmax": raw["reference_absmax"],
            "relative_rmse": relative_rmse,
            "saturation_fraction": None,
            "severity": severity,
        }

    def result(self) -> dict[str, Any]:
        raw = self.raw()
        return {"metrics": self.metrics_from_raw(raw), "raw_accumulator": raw}


@dataclass
class TopAgreement:
    baseline_equal: int = 0
    corrected_equal: int = 0
    rows: int = 0
    per_head: dict[str, dict[str, int]] = field(default_factory=dict)

    def update(
        self,
        layer: int,
        reference: torch.Tensor,
        baseline: torch.Tensor,
        corrected: torch.Tensor,
    ) -> None:
        reference_top = torch.argmax(reference, dim=-1).reshape(QUERY_HEADS, -1)
        baseline_top = torch.argmax(baseline, dim=-1).reshape(QUERY_HEADS, -1)
        corrected_top = torch.argmax(corrected, dim=-1).reshape(QUERY_HEADS, -1)
        for head in range(QUERY_HEADS):
            key = f"layer-{layer:02d}/head-{head:02d}"
            base_count = int((reference_top[head] == baseline_top[head]).sum())
            corrected_count = int((reference_top[head] == corrected_top[head]).sum())
            row_count = reference_top.shape[1]
            entry = self.per_head.setdefault(key, {"baseline_equal": 0, "corrected_equal": 0, "rows": 0})
            entry["baseline_equal"] += base_count
            entry["corrected_equal"] += corrected_count
            entry["rows"] += row_count
            self.baseline_equal += base_count
            self.corrected_equal += corrected_count
            self.rows += row_count

    def result(self) -> dict[str, Any]:
        require(self.rows > 0, "top-key accumulator is empty")
        per_head = []
        regressions = []
        for key, value in sorted(self.per_head.items()):
            baseline = value["baseline_equal"] / value["rows"]
            corrected = value["corrected_equal"] / value["rows"]
            row = {
                "baseline_fraction": baseline,
                "corrected_fraction": corrected,
                "head": key,
                "improvement": corrected - baseline,
                "rows": value["rows"],
            }
            per_head.append(row)
            if corrected < baseline:
                regressions.append(key)
        baseline = self.baseline_equal / self.rows
        corrected = self.corrected_equal / self.rows
        return {
            "absolute_improvement": corrected - baseline,
            "baseline_top_key_index_equal_fraction": baseline,
            "corrected_top_key_index_equal_fraction": corrected,
            "head_regressions": regressions,
            "per_layer_head": per_head,
            "row_head_count": self.rows,
        }


@dataclass
class SafetyCounters:
    clamp_events: int = 0
    non_finite_values: int = 0
    overflow_events: int = 0
    reserved_negative_128_payloads: int = 0
    maximum_signed_arithmetic_width: int = 1

    def merge_width(self, width: int) -> None:
        self.maximum_signed_arithmetic_width = max(self.maximum_signed_arithmetic_width, width)

    def result(self) -> dict[str, int]:
        return {
            "clamp_events": self.clamp_events,
            "maximum_signed_arithmetic_width": self.maximum_signed_arithmetic_width,
            "non_finite_values": self.non_finite_values,
            "overflow_events": self.overflow_events,
            "reserved_negative_128_payloads": self.reserved_negative_128_payloads,
        }


@dataclass
class EvaluationState:
    centered_score: MetricAccumulator = field(default_factory=MetricAccumulator)
    corrected_probability: MetricAccumulator = field(default_factory=MetricAccumulator)
    fixed_softmax_realization: MetricAccumulator = field(default_factory=MetricAccumulator)
    top: TopAgreement = field(default_factory=TopAgreement)
    safety: SafetyCounters = field(default_factory=SafetyCounters)
    baseline_score_hash_matches: int = 0
    baseline_score_hash_checks: int = 0

    def result(self) -> dict[str, Any]:
        return {
            "baseline_score_hash_checks": {
                "exact_matches": self.baseline_score_hash_matches,
                "total": self.baseline_score_hash_checks,
            },
            "corrected_centered_score": self.centered_score.result(),
            "corrected_probability": self.corrected_probability.result(),
            "fixed_softmax_realization": self.fixed_softmax_realization.result(),
            "integer_safety": self.safety.result(),
            "top_key_agreement": self.top.result(),
        }


class FitState:
    def __init__(self) -> None:
        self.residual_max = {
            "q_proj": torch.zeros((24, QUERY_HEADS), dtype=torch.float64),
            "k_proj": torch.zeros((24, KV_HEADS), dtype=torch.float64),
        }
        self.gain_numerator = {
            "q_proj": torch.zeros((24, QUERY_HEADS, GROUP_COUNT), dtype=torch.float64),
            "k_proj": torch.zeros((24, KV_HEADS, GROUP_COUNT), dtype=torch.float64),
        }
        self.gain_denominator = {
            "q_proj": torch.zeros((24, QUERY_HEADS, GROUP_COUNT), dtype=torch.float64),
            "k_proj": torch.zeros((24, KV_HEADS, GROUP_COUNT), dtype=torch.float64),
        }

    def update(self, bf16: list[dict[str, Any]], w4: list[dict[str, Any]], model: torch.nn.Module) -> None:
        for layer, (bcap, wcap) in enumerate(zip(bf16, w4, strict=True)):
            attention = model.model.layers[layer].self_attn
            for projection, heads in (("q_proj", QUERY_HEADS), ("k_proj", KV_HEADS)):
                bf_pre = reshape_heads(bcap[f"{projection}_output"], heads).to(torch.float64)
                w_pre = reshape_heads(wcap[f"{projection}_s8"], heads).to(torch.float64)
                producer_record = int(getattr(attention, projection).output_scale32_record)
                residual = bf_pre - w_pre * float(scale32_value(producer_record))
                maximum = residual.abs().amax(dim=(0, 2, 3))
                self.residual_max[projection][layer] = torch.maximum(self.residual_max[projection][layer], maximum)

                bf_post = bcap[projection[0] + "_rope"].to(torch.float64)
                w_post = wcap[projection[0] + "_rope_s8"].to(torch.float64)
                w_scale = scale32_value(wcap[projection[0] + "_rope_scale32"]).unsqueeze(-1)
                dequant = (w_post * w_scale).reshape(1, heads, w_post.shape[2], GROUP_COUNT, GROUP_SIZE)
                target = bf_post.reshape(1, heads, bf_post.shape[2], GROUP_COUNT, GROUP_SIZE)
                self.gain_numerator[projection][layer] += torch.sum(dequant * target, dim=(0, 2, 4))
                self.gain_denominator[projection][layer] += torch.sum(dequant * dequant, dim=(0, 2, 4))

    def finalize(self) -> dict[str, Any]:
        residual_records: dict[str, torch.Tensor] = {}
        gain_records: dict[str, torch.Tensor] = {}
        residual_rows = []
        gain_rows = []
        for projection, heads in (("q_proj", QUERY_HEADS), ("k_proj", KV_HEADS)):
            residual_records[projection] = torch.empty((24, heads), dtype=torch.int64)
            gain_records[projection] = torch.empty((24, heads, GROUP_COUNT), dtype=torch.int64)
            for layer in range(24):
                for head in range(heads):
                    maximum = float(self.residual_max[projection][layer, head])
                    target = maximum / 7.0
                    record = packed_scale32(target)
                    residual_records[projection][layer, head] = record
                    residual_rows.append(
                        {
                            "decoded_scale_hex": float_hex(float(scale32_value(record))),
                            "head": head,
                            "layer": layer,
                            "packed_u32": record,
                            "projection": projection,
                            "residual_absmax_hex": float_hex(maximum),
                        }
                    )
                    for group in range(GROUP_COUNT):
                        numerator = float(self.gain_numerator[projection][layer, head, group])
                        denominator = float(self.gain_denominator[projection][layer, head, group])
                        raw_gain = (
                            max(0.0, numerator / denominator)
                            if denominator > 0.0
                            else 0.0
                        )
                        gain_record = packed_scale32(raw_gain)
                        gain_records[projection][layer, head, group] = gain_record
                        gain_rows.append(
                            {
                                "decoded_gain_hex": float_hex(float(scale32_value(gain_record))),
                                "denominator_hex": float_hex(denominator),
                                "group": group,
                                "head": head,
                                "layer": layer,
                                "numerator_hex": float_hex(numerator),
                                "packed_u32": gain_record,
                                "projection": projection,
                                "raw_nnls_gain_hex": float_hex(raw_gain),
                            }
                        )
        return {
            "gain_records": gain_records,
            "residual_records": residual_records,
            "serialized": {
                "qk_group8_diagonal_scale32_rebalance_v1": {
                    "parameter_bytes_per_model": len(gain_rows) * 4,
                    "records": gain_rows,
                },
                "qk_s4_residual_cross_term_scale32_v1": {
                    "parameter_bytes_per_model": len(residual_rows) * 4,
                    "records": residual_rows,
                },
            },
        }


def capture_bf16(model: torch.nn.Module, input_ids: torch.Tensor, *, retain_scores: bool) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = [dict() for _ in range(24)]
    hooks = []
    for layer, module in enumerate(model.model.layers):
        for projection in ("q_proj", "k_proj"):
            linear = getattr(module.self_attn, projection)

            def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor, *, index: int = layer, name: str = projection) -> None:
                layers[index][f"{name}_output"] = output.detach().cpu().contiguous()

            hooks.append(linear.register_forward_hook(hook))

    original_rope = modeling_qwen2.apply_rotary_pos_emb
    original_eager = modeling_qwen2.eager_attention_forward
    rope_calls = 0
    attention_calls = 0

    def rope_wrapper(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal rope_calls
        result = original_rope(q, k, cos, sin, unsqueeze_dim)
        require(rope_calls < 24, "BF16 RoPE call count exceeded 24")
        layers[rope_calls]["q_rope"] = result[0].detach().cpu().contiguous()
        layers[rope_calls]["k_rope"] = result[1].detach().cpu().contiguous()
        rope_calls += 1
        return result

    def eager_wrapper(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal attention_calls
        result = original_eager(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
        require(attention_calls < 24, "BF16 attention call count exceeded 24")
        mapped_key = modeling_qwen2.repeat_kv(key, module.num_key_value_groups)
        scores = torch.matmul(query, mapped_key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : mapped_key.shape[-2]]
        centered = scores - scores.amax(dim=-1, keepdim=True)
        entry = layers[attention_calls]
        entry["centered_score_sha256"] = tensor_hash(centered)
        entry["probability_sha256"] = tensor_hash(result[1])
        if retain_scores:
            entry["centered_scores"] = centered.detach().cpu().contiguous()
            entry["probabilities"] = result[1].detach().cpu().contiguous()
        attention_calls += 1
        return result

    modeling_qwen2.apply_rotary_pos_emb = rope_wrapper
    modeling_qwen2.eager_attention_forward = eager_wrapper
    try:
        with torch.inference_mode():
            model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    finally:
        modeling_qwen2.apply_rotary_pos_emb = original_rope
        modeling_qwen2.eager_attention_forward = original_eager
        for hook in hooks:
            hook.remove()
    require(rope_calls == 24 and attention_calls == 24, "BF16 capture did not cover 24 layers")
    return layers


def capture_w4(model: torch.nn.Module, input_ids: torch.Tensor) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = [dict() for _ in range(24)]
    restorers: list[Callable[[], None]] = []
    for layer, module in enumerate(model.model.layers):
        for projection in ("q_proj", "k_proj"):
            linear = getattr(module.self_attn, projection)
            original = linear.forward_hardware_input

            def wrapper(self: fixed.W4A8Linear, inputs: torch.Tensor, *, index: int = layer, name: str = projection, call: Callable[[torch.Tensor], torch.Tensor] = original) -> torch.Tensor:
                result = call(inputs)
                layers[index][f"{name}_s8"] = result.detach().cpu().contiguous()
                return result

            linear.forward_hardware_input = types.MethodType(wrapper, linear)
            restorers.append(lambda target=linear, saved=original: setattr(target, "forward_hardware_input", saved))

    original_rope = fixed.dynamic_rope_head_raw
    original_scores = fixed.fixed_dynamic_attention_scores_raw
    original_softmax = fixed.fixed_softmax_raw
    rope_calls = 0
    score_calls = 0
    softmax_calls = 0

    def rope_wrapper(activations: torch.Tensor, producer_scale32: int, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        nonlocal rope_calls
        result = original_rope(activations, producer_scale32, cos, sin)
        require(rope_calls < 48, "W4 dynamic RoPE call count exceeded 48")
        layer = rope_calls // 2
        prefix = "q" if rope_calls % 2 == 0 else "k"
        layers[layer][f"{prefix}_rope_s8"] = result[0].detach().cpu().contiguous()
        layers[layer][f"{prefix}_rope_scale32"] = result[1].detach().cpu().contiguous()
        layers[layer][f"{prefix}_cos"] = cos.detach().cpu().contiguous()
        layers[layer][f"{prefix}_sin"] = sin.detach().cpu().contiguous()
        rope_calls += 1
        return result

    def score_wrapper(query: torch.Tensor, key: torch.Tensor, query_scale32: torch.Tensor, key_scale32: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        nonlocal score_calls
        result = original_scores(query, key, query_scale32, key_scale32, attention_mask)
        require(score_calls < 24, "W4 score call count exceeded 24")
        layers[score_calls]["centered_score_sha256"] = tensor_hash(result)
        score_calls += 1
        return result

    def softmax_wrapper(scores: torch.Tensor) -> torch.Tensor:
        nonlocal softmax_calls
        result = original_softmax(scores)
        require(softmax_calls < 24, "W4 softmax call count exceeded 24")
        layers[softmax_calls]["probability_sha256"] = tensor_hash(result)
        softmax_calls += 1
        return result

    fixed.dynamic_rope_head_raw = rope_wrapper
    fixed.fixed_dynamic_attention_scores_raw = score_wrapper
    fixed.fixed_softmax_raw = softmax_wrapper
    try:
        with torch.inference_mode():
            model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    finally:
        fixed.dynamic_rope_head_raw = original_rope
        fixed.fixed_dynamic_attention_scores_raw = original_scores
        fixed.fixed_softmax_raw = original_softmax
        for restore in restorers:
            restore()
    require(rope_calls == 48 and score_calls == 24 and softmax_calls == 24, "W4 capture did not cover 24 layers")
    return layers


def capture_manifest_record(
    record_index: int | str,
    split: str,
    input_ids: torch.Tensor,
    bf16: list[dict[str, Any]],
    w4: list[dict[str, Any]],
) -> dict[str, Any]:
    layer_rows = []
    for layer, (bcap, wcap) in enumerate(zip(bf16, w4, strict=True)):
        entries = {
            "bf16.centered_score": bcap["centered_score_sha256"],
            "bf16.k_proj": tensor_hash(bcap["k_proj_output"]),
            "bf16.k_rope": tensor_hash(bcap["k_rope"]),
            "bf16.probability": bcap["probability_sha256"],
            "bf16.q_proj": tensor_hash(bcap["q_proj_output"]),
            "bf16.q_rope": tensor_hash(bcap["q_rope"]),
            "w4.centered_score": wcap["centered_score_sha256"],
            "w4.k_proj_s8": tensor_hash(wcap["k_proj_s8"]),
            "w4.k_rope_s8": tensor_hash(wcap["k_rope_s8"]),
            "w4.k_rope_scale32": tensor_hash(wcap["k_rope_scale32"]),
            "w4.probability": wcap["probability_sha256"],
            "w4.q_proj_s8": tensor_hash(wcap["q_proj_s8"]),
            "w4.q_rope_s8": tensor_hash(wcap["q_rope_s8"]),
            "w4.q_rope_scale32": tensor_hash(wcap["q_rope_scale32"]),
            "w4.rope_cos": tensor_hash(wcap["q_cos"]),
            "w4.rope_sin": tensor_hash(wcap["q_sin"]),
        }
        layer_rows.append({"capture_sha256": canonical_sha256(entries), "layer": layer, "tensor_hashes": entries})
    return {
        "capture_sha256": canonical_sha256(layer_rows),
        "input_token_count": int(input_ids.shape[1]),
        "input_token_ids_sha256": tensor_hash(input_ids),
        "layer_captures": layer_rows,
        "record_index": record_index,
        "split": split,
    }


def baseline_scores(wcap: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = wcap["q_rope_s8"]
    key = repeat_kv(wcap["k_rope_s8"])
    query_records = wcap["q_rope_scale32"]
    key_records = repeat_kv(wcap["k_rope_scale32"].unsqueeze(-1)).squeeze(-1)
    sequence = query.shape[2]
    valid, mask = causal_mask(sequence, device=query.device)
    scores = fixed.fixed_dynamic_attention_scores_raw(query, key, query_records, key_records, mask)
    return scores, valid, mask


def residual_scores(
    layer: int,
    bcap: dict[str, Any],
    wcap: dict[str, Any],
    model: torch.nn.Module,
    parameters: dict[str, torch.Tensor],
    safety: SafetyCounters,
) -> torch.Tensor:
    attention = model.model.layers[layer].self_attn
    residual_rotated = {}
    for projection, heads, prefix in (("q_proj", QUERY_HEADS, "q"), ("k_proj", KV_HEADS, "k")):
        bf_pre = reshape_heads(bcap[f"{projection}_output"], heads).to(torch.float64)
        w_pre = reshape_heads(wcap[f"{projection}_s8"], heads)
        producer_record = int(getattr(attention, projection).output_scale32_record)
        residual_physical = bf_pre - w_pre.to(torch.float64) * float(scale32_value(producer_record))
        records = parameters[projection][layer]
        scales = scale32_value(records).reshape(1, heads, 1, 1)
        unbounded = torch.round(residual_physical / scales)
        safety.clamp_events += int(((unbounded < -7) | (unbounded > 7)).sum())
        codes = unbounded.clamp(-7, 7).to(torch.int8)
        safety.reserved_negative_128_payloads += int((codes == -128).sum())
        rotated, overflow, width = rotate_residual(codes, wcap[f"{prefix}_cos"], wcap[f"{prefix}_sin"])
        safety.overflow_events += overflow
        safety.merge_width(width)
        residual_rotated[projection] = rotated

    query = wcap["q_rope_s8"]
    key = repeat_kv(wcap["k_rope_s8"])
    qres = residual_rotated["q_proj"]
    kres = repeat_kv(residual_rotated["k_proj"])
    qbase_records = wcap["q_rope_scale32"]
    kbase_records = repeat_kv(wcap["k_rope_scale32"].unsqueeze(-1)).squeeze(-1)
    qres_records = parameters["q_proj"][layer].reshape(1, QUERY_HEADS, 1).expand_as(qbase_records)
    kres_records = parameters["k_proj"][layer].repeat_interleave(QUERY_HEADS // KV_HEADS).reshape(1, QUERY_HEADS, 1).expand_as(kbase_records)
    terms = (
        (query, key, qbase_records, kbase_records),
        (query, kres, qbase_records, kres_records),
        (qres, key, qres_records, kbase_records),
        (qres, kres, qres_records, kres_records),
    )
    total = None
    for left, right, left_records, right_records in terms:
        dots = torch.matmul(left.to(torch.int32), right.to(torch.int32).transpose(2, 3)).to(torch.int64)
        converted, overflow, width = scale32_pair_to_q20_44(dots, left_records, right_records)
        safety.overflow_events += overflow
        safety.merge_width(width)
        if total is None:
            total = converted
        else:
            positive_overflow = (converted > 0) & (total > torch.iinfo(torch.int64).max - converted)
            negative_overflow = (converted < 0) & (total < torch.iinfo(torch.int64).min - converted)
            events = int((positive_overflow | negative_overflow).sum())
            safety.overflow_events += events
            require(events == 0, "residual corrected score exceeded signed int64")
            total = total + converted
    assert total is not None
    valid, _ = causal_mask(total.shape[-1], device=total.device)
    centered = center_q20_44(total, valid)
    scores, clamps = score32_from_q20_44(centered, valid)
    safety.clamp_events += clamps
    safety.merge_width(64)
    return scores


def group_gain_scores(
    layer: int,
    wcap: dict[str, Any],
    parameters: dict[str, torch.Tensor],
    safety: SafetyCounters,
) -> torch.Tensor:
    query = wcap["q_rope_s8"]
    key = repeat_kv(wcap["k_rope_s8"])
    sequence = query.shape[2]
    valid, _ = causal_mask(sequence, device=query.device)
    return exact_group8_scores(
        query,
        key,
        wcap["q_rope_scale32"],
        repeat_kv(wcap["k_rope_scale32"].unsqueeze(-1)).squeeze(-1),
        parameters["q_proj"][layer],
        parameters["k_proj"][layer].repeat_interleave(QUERY_HEADS // KV_HEADS, dim=0),
        valid,
        safety,
    )


def update_evaluation(
    state: EvaluationState,
    layer: int,
    reference_scores: torch.Tensor,
    reference_probabilities: torch.Tensor,
    baseline: torch.Tensor,
    corrected: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    corrected_float = corrected.to(torch.float64) / (1 << SCORE_FRAC)
    candidate_probabilities = fixed.fixed_softmax_raw(corrected).to(torch.float64) / (1 << fixed.SOFTMAX_PROB_FRAC)
    ideal_scores = torch.where(
        valid.expand_as(corrected),
        corrected.to(torch.float32) / (1 << SCORE_FRAC),
        torch.full_like(corrected.to(torch.float32), float("-inf")),
    )
    ideal_probabilities = torch.softmax(ideal_scores, dim=-1).to(torch.float64)
    state.safety.non_finite_values += state.centered_score.update(reference_scores, corrected_float, valid)
    state.safety.non_finite_values += state.corrected_probability.update(reference_probabilities, candidate_probabilities, valid)
    state.safety.non_finite_values += state.fixed_softmax_realization.update(ideal_probabilities, candidate_probabilities, valid)
    state.top.update(layer, reference_scores, baseline.to(torch.float64), corrected_float)


def evaluate_capture(
    bf16: list[dict[str, Any]],
    w4: list[dict[str, Any]],
    model: torch.nn.Module,
    fitted: dict[str, Any],
    states: dict[str, EvaluationState],
) -> None:
    for layer, (bcap, wcap) in enumerate(zip(bf16, w4, strict=True)):
        baseline, valid, _ = baseline_scores(wcap)
        baseline_hash_match = tensor_hash(baseline) == wcap["centered_score_sha256"]
        for state in states.values():
            state.baseline_score_hash_checks += 1
            state.baseline_score_hash_matches += int(baseline_hash_match)
        require(baseline_hash_match, f"c01 baseline score parity failed at layer {layer}")
        reference_scores = bcap["centered_scores"].to(torch.float64)
        reference_probabilities = bcap["probabilities"].to(torch.float64)

        residual = residual_scores(layer, bcap, wcap, model, fitted["residual_records"], states[ALTERNATIVE_ORDER[0]].safety)
        update_evaluation(states[ALTERNATIVE_ORDER[0]], layer, reference_scores, reference_probabilities, baseline, residual, valid)

        gains = group_gain_scores(layer, wcap, fitted["gain_records"], states[ALTERNATIVE_ORDER[1]].safety)
        update_evaluation(states[ALTERNATIVE_ORDER[1]], layer, reference_scores, reference_probabilities, baseline, gains, valid)


def checks_for_result(task: dict[str, Any], validation: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, bool]:
    validation_gate = task["acceptance"]["frozen_calibration_validation_gate_per_model"]
    diagnostic_gate = task["acceptance"]["fixed_diagnostic_gate_per_model"]
    safety_gate = task["acceptance"]["integer_safety"]
    return {
        "diagnostic_centered_score_tracking": diagnostic["corrected_centered_score"]["metrics"]["severity"] == diagnostic_gate["corrected_centered_score_severity"],
        "diagnostic_fixed_softmax_tracking": diagnostic["fixed_softmax_realization"]["metrics"]["severity"] == diagnostic_gate["fixed_softmax_realization_severity"],
        "diagnostic_probability_tracking": diagnostic["corrected_probability"]["metrics"]["severity"] == diagnostic_gate["corrected_probability_severity"],
        "diagnostic_top_key_exact": diagnostic["top_key_agreement"]["corrected_top_key_index_equal_fraction"] == diagnostic_gate["top_key_index_equal_fraction"],
        "integer_clamp_free": validation["integer_safety"]["clamp_events"] + diagnostic["integer_safety"]["clamp_events"] == safety_gate["clamp_events"],
        "integer_non_finite_free": validation["integer_safety"]["non_finite_values"] + diagnostic["integer_safety"]["non_finite_values"] == safety_gate["non_finite_values"],
        "integer_overflow_free": validation["integer_safety"]["overflow_events"] + diagnostic["integer_safety"]["overflow_events"] == safety_gate["overflow_events"],
        "integer_reserved_payload_free": validation["integer_safety"]["reserved_negative_128_payloads"] + diagnostic["integer_safety"]["reserved_negative_128_payloads"] == safety_gate["reserved_negative_128_payloads"],
        "validation_centered_score_tracking": validation["corrected_centered_score"]["metrics"]["severity"] == validation_gate["corrected_centered_score_severity"],
        "validation_minimum_improvement": validation["top_key_agreement"]["absolute_improvement"] >= validation_gate["minimum_absolute_top_key_agreement_improvement"],
        "validation_no_head_regression": not validation["top_key_agreement"]["head_regressions"],
        "validation_probability_tracking": validation["corrected_probability"]["metrics"]["severity"] == validation_gate["corrected_probability_severity"],
        "validation_top_key_fraction": validation["top_key_agreement"]["corrected_top_key_index_equal_fraction"] >= validation_gate["minimum_top_key_index_equal_fraction"],
    }


def model_result(
    alias: str,
    task: dict[str, Any],
    contract: dict[str, Any],
    prompts: list[torch.Tensor],
    input_observation: dict[str, Any],
    fixed_input: torch.Tensor,
    fixed_input_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = frozen.model_spec(contract, alias)
    frozen.verify_local_model_inputs(contract, spec)
    sealed_root = coarse.SEALED_ROOTS[alias]
    bf16_model = frozen.load_model(spec)
    w4_model = frozen.load_model(spec)
    source_reconstruction, _ = coarse.reconstruct_model_from_source_oracle(w4_model, sealed_root, contract)
    require(source_reconstruction["status"] == "PASS_EXACT_FROZEN_SOURCE_RECONSTRUCTION", f"source reconstruction failed: {alias}")
    evaluator.enable_w4a8_kv_cache(w4_model)

    fit = FitState()
    manifest_records = []
    fit_start = task["calibration_protocol"]["fit_records"]["start"]
    fit_stop = task["calibration_protocol"]["fit_records"]["stop"]
    validation_start = task["calibration_protocol"]["validation_records"]["start"]
    validation_stop = task["calibration_protocol"]["validation_records"]["stop"]
    for index in range(fit_start, fit_stop):
        bf16 = capture_bf16(bf16_model, prompts[index], retain_scores=False)
        w4 = capture_w4(w4_model, prompts[index])
        fit.update(bf16, w4, w4_model)
        manifest_records.append(capture_manifest_record(index, "fit", prompts[index], bf16, w4))
        print(f"CAPTURE alias={alias} split=fit record={index} tokens={prompts[index].shape[1]}", flush=True)
        del bf16, w4
        gc.collect()
    fitted = fit.finalize()

    validation_states = {alternative: EvaluationState() for alternative in ALTERNATIVE_ORDER}
    for index in range(validation_start, validation_stop):
        bf16 = capture_bf16(bf16_model, prompts[index], retain_scores=True)
        w4 = capture_w4(w4_model, prompts[index])
        evaluate_capture(bf16, w4, w4_model, fitted, validation_states)
        manifest_records.append(capture_manifest_record(index, "validation", prompts[index], bf16, w4))
        print(f"EVALUATE alias={alias} split=validation record={index} tokens={prompts[index].shape[1]}", flush=True)
        del bf16, w4
        gc.collect()

    diagnostic_bf16 = capture_bf16(bf16_model, fixed_input, retain_scores=True)
    diagnostic_w4 = capture_w4(w4_model, fixed_input)
    diagnostic_states = {alternative: EvaluationState() for alternative in ALTERNATIVE_ORDER}
    evaluate_capture(diagnostic_bf16, diagnostic_w4, w4_model, fitted, diagnostic_states)
    diagnostic_manifest = capture_manifest_record("g00-arithmetic-short", "fixed_diagnostic", fixed_input, diagnostic_bf16, diagnostic_w4)
    print(f"EVALUATE alias={alias} split=fixed_diagnostic tokens={fixed_input.shape[1]}", flush=True)

    alternatives = []
    for alternative in ALTERNATIVE_ORDER:
        validation = validation_states[alternative].result()
        diagnostic = diagnostic_states[alternative].result()
        checks = checks_for_result(task, validation, diagnostic)
        alternatives.append(
            {
                "alternative_id": alternative,
                "checks": checks,
                "fixed_diagnostic": diagnostic,
                "fit": fitted["serialized"][alternative],
                "pass": all(checks.values()),
                "validation": validation,
            }
        )

    manifest = {
        "artifact_kind": "w4a8_c02_qk_score_rank_margin_calibration_capture_manifest",
        "calibration_input_observation": input_observation,
        "capture_digest_sha256": canonical_sha256(manifest_records),
        "capture_fields": [
            "Q/K projection outputs",
            "Q/K post-RoPE tensors and Scale32 records",
            "centered score tensors",
            "ideal/fixed softmax probability tensors",
        ],
        "diagnostic_capture": diagnostic_manifest,
        "fit_record_range": [fit_start, fit_stop],
        "model_alias": alias,
        "model_identity_sha256": task["frozen_inputs"]["c01"][alias]["model_identity_sha256"],
        "records": manifest_records,
        "schema_version": 1,
        "source_reconstruction": source_reconstruction,
        "validation_record_range": [validation_start, validation_stop],
    }
    result = {
        "alternatives": alternatives,
        "artifact_kind": "w4a8_c02_qk_score_rank_margin_alternative_results",
        "claim_boundary": task["claim_boundary"],
        "fixed_input": fixed_input_record,
        "model_alias": alias,
        "model_identity_sha256": task["frozen_inputs"]["c01"][alias]["model_identity_sha256"],
        "schema_version": 1,
    }
    del bf16_model, w4_model, diagnostic_bf16, diagnostic_w4
    gc.collect()
    return manifest, result


def select_result(task: dict[str, Any], model_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for alternative in ALTERNATIVE_ORDER:
        rows = []
        for alias in MODEL_ORDER:
            row = next(item for item in model_results[alias]["alternatives"] if item["alternative_id"] == alternative)
            rows.append(row)
        passed = all(row["pass"] for row in rows)
        candidates.append(
            {
                "alternative_id": alternative,
                "declared_maximum_signed_arithmetic_width": max(
                    max(row["validation"]["integer_safety"]["maximum_signed_arithmetic_width"], row["fixed_diagnostic"]["integer_safety"]["maximum_signed_arithmetic_width"])
                    for row in rows
                ),
                "model_results": {alias: rows[index]["pass"] for index, alias in enumerate(MODEL_ORDER)},
                "model_specific_metadata_bytes": sum(row["fit"]["parameter_bytes_per_model"] for row in rows),
                "pass": passed,
                "validation_corrected_probability_relative_rmse_sum": sum(row["validation"]["corrected_probability"]["metrics"]["relative_rmse"] for row in rows),
            }
        )
    passing = [row for row in candidates if row["pass"]]
    selected = None
    if passing:
        selected = min(
            passing,
            key=lambda row: (
                row["validation_corrected_probability_relative_rmse_sum"],
                row["model_specific_metadata_bytes"],
                row["declared_maximum_signed_arithmetic_width"],
                row["alternative_id"],
            ),
        )["alternative_id"]
    verdict = (
        task["acceptance"]["fail_closed_outcomes"]["one_alternative_passes"]
        if selected is not None
        else task["acceptance"]["fail_closed_outcomes"]["no_alternative_passes"]
    )
    return {
        "alternatives": candidates,
        "artifact_kind": "w4a8_c02_qk_score_rank_margin_selection",
        "candidate_package_created": False,
        "schema_version": 1,
        "selected_alternative_id": selected,
        "selection_tie_break": task["acceptance"]["selection_tie_break"],
        "verdict": verdict,
    }


def source_hashes() -> dict[str, str]:
    paths = (
        SCRIPT_PATH,
        TASK_PATH,
        ROOT / "tools/ace2_full_model_fixed_point.py",
        ROOT / "tools/diagnose_w4a8_c02_numerical_path.py",
        ROOT / "tools/w4a8_full_model_evaluator.py",
        ROOT / "tools/run_w4a8_full_model_software_contract.py",
    )
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def run_contract(task: dict[str, Any], seeds: dict[str, Any], threads: int) -> dict[str, Any]:
    return {
        "alternatives": task["alternatives"],
        "artifact_kind": "w4a8_c02_qk_score_rank_margin_frozen_run_contract",
        "calibration_protocol": task["calibration_protocol"],
        "commands": {
            "logical_single_run": ".venv/bin/python tools/run_w4a8_c02_qk_score_rank_margin_repair_design.py --single-run <isolated-run-root> --threads 8",
            "verify": ".venv/bin/python tools/run_w4a8_c02_qk_score_rank_margin_repair_design.py --verify-existing",
        },
        "compute": {
            "device": "cpu",
            "interop_threads": 1,
            "threads": threads,
        },
        "environment": {
            "datasets": package_version("datasets"),
            "peft": package_version("peft"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
        },
        "frozen_inputs": task["frozen_inputs"],
        "model_order": list(MODEL_ORDER),
        "required_outputs": task["artifact_isolation"]["required_outputs"],
        "schema_version": 1,
        "seeds": seeds,
        "source_hashes": source_hashes(),
        "task": {"path": TASK_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(TASK_PATH)},
        "thresholds": task["acceptance"],
    }


def write_sums(root: Path) -> None:
    members = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    body = "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in members)
    (root / "SHA256SUMS").write_text(body, encoding="ascii")


def run_once(output_root: Path, threads: int) -> None:
    require(not output_root.exists(), f"isolated run root already exists: {output_root}")
    require(ROOT.resolve() in output_root.resolve().parents, "output must remain repository-local")
    require(not {"candidate", "candidates", "attempt", "attempts"}.intersection(output_root.parts), "candidate/attempt namespace is forbidden")
    task = load_json(TASK_PATH)
    static = static_preflight.build_result(task)
    require(static["status"] == "PASS_REPAIR_DESIGN_STATIC_PREFLIGHT", "static preflight did not pass")
    require(1 <= threads <= 32, "thread count must be in 1..32")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    seeds = frozen.set_determinism()
    contract, contract_sha256 = frozen.load_contract()
    require(contract_sha256 == task["frozen_inputs"]["software_contract"]["sha256"], "software contract hash differs")
    base_spec = frozen.model_spec(contract, "base")
    frozen.verify_local_model_inputs(contract, base_spec)
    tokenizer = frozen.load_tokenizer(contract, base_spec)
    prompt_manifest = load_json(evaluator.PROMPT_MANIFEST_PATH)
    calibration_spec = prompt_manifest["datasets"]["c4_calibration"]
    texts = evaluator.selected_local_texts(calibration_spec)
    prompts = evaluator._tokenize_calibration(
        tokenizer,
        texts,
        contract["shared_w4a8_contract"]["generation"],
        contract["evaluation_contract"]["calibration_set"]["token_limit"],
    )
    record_sha, record_count = fixed.hash_records(texts)
    token_sha, sequence_count, token_count = fixed.hash_token_sequences(prompts)
    input_observation = {
        "config": calibration_spec["config"],
        "record_count": record_count,
        "record_sha256": record_sha,
        "repository": calibration_spec["repository"],
        "revision": calibration_spec["revision"],
        "split": calibration_spec["split"],
        "tokenized": {
            "sequence_count": sequence_count,
            "token_count": token_count,
            "token_sequence_sha256": token_sha,
        },
    }
    sealed_observation = load_json(coarse.SEALED_ROOTS["base"] / "input-observations.json")["c4_calibration"]
    require(input_observation == sealed_observation, "live calibration inputs differ from sealed c01 observations")
    fixed_input, fixed_input_record = coarse.sealed_fixed_input()

    output_root.mkdir(parents=True)
    write_json(output_root / "run-contract.json", run_contract(task, seeds, threads))
    results = {}
    for alias in MODEL_ORDER:
        model_root = output_root / alias
        model_root.mkdir()
        manifest, result = model_result(alias, task, contract, prompts, input_observation, fixed_input, fixed_input_record)
        write_json(model_root / "calibration-capture-manifest.json", manifest)
        write_json(model_root / "alternative-results.json", result)
        results[alias] = result
    write_json(output_root / "selection.json", select_result(task, results))
    write_sums(output_root)
    print(load_json(output_root / "selection.json")["verdict"], flush=True)


def verify_sums(root: Path) -> None:
    sums = root / "SHA256SUMS"
    require(sums.is_file(), f"SHA256SUMS is missing: {root}")
    for line in sums.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        path = root / name
        require(path.is_file(), f"hashed evidence member is missing: {name}")
        require(sha256_file(path) == digest, f"hashed evidence member differs: {name}")


def recompute_model_result(task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(result))
    for alternative in clone["alternatives"]:
        for split in ("validation", "fixed_diagnostic"):
            for metric_name in ("corrected_centered_score", "corrected_probability", "fixed_softmax_realization"):
                record = alternative[split][metric_name]
                record["metrics"] = MetricAccumulator.metrics_from_raw(record["raw_accumulator"])
        alternative["checks"] = checks_for_result(task, alternative["validation"], alternative["fixed_diagnostic"])
        alternative["pass"] = all(alternative["checks"].values())
    return clone


def verify_run(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    verify_sums(root)
    contract = load_json(root / "run-contract.json")
    require(contract["task"]["sha256"] == sha256_file(TASK_PATH), "run task hash differs")
    for path, digest in contract["source_hashes"].items():
        require(sha256_file(ROOT / path) == digest, f"run source hash differs: {path}")
    results = {}
    for alias in MODEL_ORDER:
        manifest = load_json(root / alias / "calibration-capture-manifest.json")
        require(manifest["capture_digest_sha256"] == canonical_sha256(manifest["records"]), f"capture digest differs: {alias}")
        live = load_json(root / alias / "alternative-results.json")
        recomputed = recompute_model_result(task, live)
        require(recomputed == live, f"metrics or checks do not recompute exactly: {alias}")
        results[alias] = live
    selection = load_json(root / "selection.json")
    require(selection == select_result(task, results), "selection does not recompute from model results")
    return selection


def assemble_campaign(output_root: Path) -> str:
    task = load_json(TASK_PATH)
    selections = {name: verify_run(output_root / name, task) for name in RUN_NAMES}
    compared = []
    deterministic = True
    for member in RUN_MEMBERS:
        first = output_root / RUN_NAMES[0] / member
        second = output_root / RUN_NAMES[1] / member
        equal = first.read_bytes() == second.read_bytes()
        compared.append({
            "byte_identical": equal,
            "member": member,
            "run_0001_sha256": sha256_file(first),
            "run_0002_sha256": sha256_file(second),
        })
        deterministic &= equal
    require(selections[RUN_NAMES[0]] == selections[RUN_NAMES[1]], "selection objects differ across independent runs")
    for member in RUN_MEMBERS[:-1]:
        destination = output_root / member
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_root / RUN_NAMES[0] / member, destination)
    verdict = selections[RUN_NAMES[0]]["verdict"] if deterministic else task["acceptance"]["fail_closed_outcomes"]["tie_or_nondeterminism"]
    write_json(
        output_root / "determinism-check.json",
        {
            "artifact_kind": "w4a8_c02_qk_score_rank_margin_determinism_check",
            "compared_members": compared,
            "independent_repeat_count": 2,
            "runs_byte_identical": deterministic,
            "schema_version": 1,
            "verdict": verdict,
        },
    )
    write_sums(output_root)
    return verdict


def run_campaign(output_root: Path, threads: int) -> int:
    require(not output_root.exists(), f"campaign output root already exists: {output_root}")
    require(ROOT.resolve() in output_root.resolve().parents, "campaign output must remain repository-local")
    require(
        output_root == DEFAULT_OUTPUT or DEFAULT_OUTPUT.resolve() in output_root.parents,
        "campaign output root differs from the frozen diagnostic namespace",
    )
    output_root.mkdir(parents=True)
    for run_name in RUN_NAMES:
        stdout_path = output_root / f"{run_name}.stdout.log"
        stderr_path = output_root / f"{run_name}.stderr.log"
        command = [sys.executable, str(SCRIPT_PATH), "--single-run", str(output_root / run_name), "--threads", str(threads)]
        print(f"START {run_name}", flush=True)
        started = time.monotonic()
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout_handle, stderr=stderr_handle, env=os.environ.copy())
            while process.poll() is None:
                time.sleep(10)
                print(f"ACTIVE {run_name} elapsed_seconds={int(time.monotonic() - started)}", flush=True)
        if process.returncode != 0:
            write_json(
                output_root / f"{run_name}.failure.json",
                {
                    "failure_taxonomy": "diagnostic_execution_failure",
                    "regression": "rerun the focused unit tests and the exact single-run command after repairing the first stderr failure",
                    "root_cause_hypothesis": "the first concrete exception in the preserved stderr log",
                    "run": run_name,
                    "stderr": stderr_path.name,
                    "stdout": stdout_path.name,
                },
            )
            return process.returncode
        print(f"DONE {run_name} elapsed_seconds={int(time.monotonic() - started)}", flush=True)
    verdict = assemble_campaign(output_root)
    print(verdict, flush=True)
    return 0


def verify_campaign(output_root: Path) -> str:
    verify_sums(output_root)
    verdict = assemble_verify_only(output_root)
    print(f"{verdict} verified_existing=true")
    return verdict


def assemble_verify_only(output_root: Path) -> str:
    task = load_json(TASK_PATH)
    selections = {name: verify_run(output_root / name, task) for name in RUN_NAMES}
    check = load_json(output_root / "determinism-check.json")
    require(check["runs_byte_identical"] is True, "independent runs are not byte-identical")
    for row in check["compared_members"]:
        member = row["member"]
        first = output_root / RUN_NAMES[0] / member
        second = output_root / RUN_NAMES[1] / member
        require(first.read_bytes() == second.read_bytes(), f"independent run member differs: {member}")
        if member != "SHA256SUMS":
            require((output_root / member).read_bytes() == first.read_bytes(), f"canonical campaign copy differs: {member}")
    require(selections[RUN_NAMES[0]] == selections[RUN_NAMES[1]], "independent selection differs")
    require(check["verdict"] == selections[RUN_NAMES[0]]["verdict"], "determinism verdict differs from selection")
    return check["verdict"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--single-run", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.single_run is not None:
        run_once(args.single_run.resolve(), args.threads)
        return 0
    output_root = args.output_root.resolve()
    if args.verify_existing:
        verify_campaign(output_root)
        return 0
    return run_campaign(output_root, args.threads)


if __name__ == "__main__":
    raise SystemExit(main())
