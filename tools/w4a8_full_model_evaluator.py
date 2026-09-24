#!/usr/bin/env python3
"""Deterministic evaluator for bounded candidates in the frozen W4A8 contract."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
import types
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    from tools import ace2_full_model_fixed_point as fixed
    from tools.ace2_quality_contracts import (
        ceil_scale32_from_float,
        ceil_scale32_from_ratio,
        scale32_ratio,
        unpack_scale32,
    )
else:
    import ace2_full_model_fixed_point as fixed
    from ace2_quality_contracts import (
        ceil_scale32_from_float,
        ceil_scale32_from_ratio,
        scale32_ratio,
        unpack_scale32,
    )


ROOT = Path(__file__).resolve().parents[1]
QUALITY_CONFIG_PATH = ROOT / "benchmark/quality/QUALITY_CONFIG.json"
PROMPT_MANIFEST_PATH = ROOT / "benchmark/quality/PROMPT_MANIFEST.json"
ASCII_WHITESPACE = " \t\n\r\f\v"
ACCEPTED_GOOD_MORNING_SPANISH = frozenset(
    {"buenos días", "buenos dias", "buen día", "buen dia"}
)
LIST_PREFIX = re.compile(r"^(?:[-*•]|[12][.)])\s+", re.UNICODE)
SUPPORTED_VALIDATORS = frozenset(
    {
        "strip_ascii_whitespace_then_exactly_19",
        "readability_only",
        "parse_json_then_exact_object_status_ok_count_3",
        "exactly_three_nonempty_lines_each_containing_a_unicode_letter",
        "unicode_casefold_strip_terminal_punctuation_in_buenos_dias_set",
        "exactly_two_list_items_with_unicode_letters",
    }
)
GENERATION_GATE_FIELDS = frozenset(
    {
        "all_prompt_validators_must_pass",
        "maximum_identical_token_run",
        "minimum_alphanumeric_character_fraction",
        "minimum_unique_token_fraction_when_at_least_8_tokens",
        "minimum_visible_tokens",
        "punctuation_only_output_rejected",
        "replacement_surrogate_or_control_character_rejected",
        "short_answer_minimum_visible_tokens",
    }
)
QUALITY_GATE_FIELDS = frozenset(
    {
        "applies_independently_to_both_models",
        "c4_perplexity_ratio_to_model_specific_bf16_max",
        "lm_eval_average_normalized_accuracy_drop_percentage_points_max",
        "lm_eval_individual_task_drop_percentage_points_max",
        "mission_local_not_project_final_emnlp_gate",
        "wikitext2_perplexity_ratio_to_model_specific_bf16_max",
    }
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def require_canonical_value(path: Path) -> Any:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(path.read_bytes() == canonical_bytes(value), f"JSON is not canonical: {path}")
    return value


def require_canonical_json(path: Path) -> dict[str, Any]:
    value = require_canonical_value(path)
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def quality_thresholds_from_contract(contract: dict[str, Any]) -> dict[str, Any]:
    gate = contract["evaluation_contract"]["quality_gate"]
    require(set(gate) == QUALITY_GATE_FIELDS, "frozen quality-gate fields differ")
    return {
        "applies_independently_to_both_models": gate[
            "applies_independently_to_both_models"
        ],
        "c4_en_512_perplexity_ratio_max": gate[
            "c4_perplexity_ratio_to_model_specific_bf16_max"
        ],
        "lm_eval_average_normalized_accuracy_drop_percentage_points_max": gate[
            "lm_eval_average_normalized_accuracy_drop_percentage_points_max"
        ],
        "lm_eval_individual_task_drop_percentage_points_max": gate[
            "lm_eval_individual_task_drop_percentage_points_max"
        ],
        "mission_local_not_project_final_emnlp_gate": gate[
            "mission_local_not_project_final_emnlp_gate"
        ],
        "wikitext2_perplexity_ratio_max": gate[
            "wikitext2_perplexity_ratio_to_model_specific_bf16_max"
        ],
    }


def validate_contract_support(
    contract: dict[str, Any],
    quality_config: dict[str, Any],
) -> None:
    evaluation = contract["evaluation_contract"]
    validators = [
        prompt["validator"] for prompt in evaluation["canonical_generation_prompts"]
    ]
    require(set(validators) == SUPPORTED_VALIDATORS, "canonical validator set differs")
    require(len(validators) == 8, "canonical prompt count differs")
    require(
        set(evaluation["generation_gate"]) == GENERATION_GATE_FIELDS,
        "frozen generation-gate fields differ",
    )
    require(
        evaluation["generation_gate"]["punctuation_only_output_rejected"] is True,
        "punctuation-only rejection is not enabled",
    )
    require(
        evaluation["generation_gate"][
            "replacement_surrogate_or_control_character_rejected"
        ]
        is True,
        "invalid-character rejection is not enabled",
    )
    require(
        quality_thresholds_from_contract(contract)
        == quality_config["acceptance_thresholds"],
        "quality thresholds differ from the frozen contract",
    )
    require(
        contract["candidate_matrix"][0]
        == {
            "candidate_id": "c00-rtn-absmax",
            "description": "Deterministic signed-W4 round-to-nearest-even control with one full-row scale per output channel.",
            "ordered_index": 0,
            "weight_generation": {
                "code_selection": "round_to_nearest_ties_to_even_then_saturate_signed_w4",
                "scale": "maximum_absolute_bf16_weight_in_complete_output_row_divided_by_7",
                "search": "none",
            },
        },
        "c00 candidate definition differs",
    )
    require(
        contract["candidate_matrix"][1]
        == {
            "candidate_id": "c01-mse-clip-grid",
            "description": "Predeclared per-row clipping grid selected by exact BF16 weight reconstruction MSE.",
            "ordered_index": 1,
            "weight_generation": {
                "clip_ratio_denominator": 1000,
                "clip_ratio_numerators": [1000, 990, 975, 950, 925, 900],
                "code_selection": "round_to_nearest_ties_to_even_then_saturate_signed_w4",
                "objective": "sum_of_squared_bf16_weight_error_accumulated_in_binary64",
                "tie_break": "larger_clip_ratio_then_lower_candidate_index",
            },
        },
        "c01 candidate definition differs",
    )
    shared = contract["shared_w4a8_contract"]
    require(shared["weight_w4"]["integer_range"] == [-8, 7], "W4 range differs")
    require(shared["activation_a8"]["integer_range"] == [-128, 127], "A8 range differs")
    require(shared["accumulation"]["width"] == 32, "accumulator width differs")
    require(shared["kv_cache"]["width"] == 8, "KV-cache width differs")
    require(
        shared["generation"]["decode"] == "greedy_one_token_at_a_time"
        and shared["generation"]["sampling"] is False
        and shared["generation"]["use_cache"] is True,
        "generation policy differs",
    )


def candidate_spec(contract: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in contract["candidate_matrix"]
        if candidate["candidate_id"] == candidate_id
    ]
    require(len(matches) == 1, f"candidate identity count differs: {candidate_id}")
    return matches[0]


def scale32_value(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def realize_positive_scale32(value: float) -> tuple[int, float]:
    record = ceil_scale32_from_float(value)
    realized = scale32_value(record)
    require(realized >= value, "Scale32 realization is below the required scale")
    return record, realized


def _scale32_values(records: torch.Tensor) -> torch.Tensor:
    integer_records = records.detach().to(torch.int64)
    significand = torch.bitwise_and(integer_records, 0xFFFF)
    exponent_u8 = torch.bitwise_and(torch.bitwise_right_shift(integer_records, 16), 0xFF)
    exponent = torch.where(exponent_u8 >= 128, exponent_u8 - 256, exponent_u8)
    return torch.ldexp(significand.to(torch.float64), exponent - 15)


def _clip_scale32_records(
    row_absmax_bf16: torch.Tensor,
    numerator: int,
    denominator: int,
) -> torch.Tensor:
    require(row_absmax_bf16.dtype == torch.bfloat16, "c01 row absmax must be BF16")
    require(row_absmax_bf16.ndim == 1, "c01 row absmax must be rank one")
    require(numerator > 0 and denominator > 0, "c01 clip ratio must be positive")
    unique_absmax, inverse = torch.unique(
        row_absmax_bf16.detach().cpu(),
        sorted=True,
        return_inverse=True,
    )
    unique_records: list[int] = []
    for value in unique_absmax.to(torch.float64).tolist():
        absmax_numerator, absmax_denominator = float(value).as_integer_ratio()
        unique_records.append(
            ceil_scale32_from_ratio(
                absmax_numerator * numerator,
                absmax_denominator * denominator * 7,
            )
        )
    records = torch.tensor(unique_records, dtype=torch.int64)[inverse]
    return records.to(row_absmax_bf16.device)


def _quantize_bf16_with_scale32(
    weight_bf16: torch.Tensor,
    scale32_records: torch.Tensor,
) -> torch.Tensor:
    """Quantize BF16 rows with exact signed ties-to-even Scale32 division."""
    require(weight_bf16.dtype == torch.bfloat16, "c01 quantization source must be BF16")
    require(weight_bf16.ndim == 2, "c01 quantization source must be a matrix")
    require(
        scale32_records.shape == (weight_bf16.shape[0],),
        "c01 Scale32 row count differs",
    )
    bits = torch.bitwise_and(weight_bf16.view(torch.int16).to(torch.int64), 0xFFFF)
    exponent_bits = torch.bitwise_and(torch.bitwise_right_shift(bits, 7), 0xFF)
    require(not torch.any(exponent_bits == 0xFF).item(), "c01 BF16 source is non-finite")
    fraction = torch.bitwise_and(bits, 0x7F)
    mantissa = torch.where(exponent_bits == 0, fraction, fraction + 0x80)
    weight_exponent = torch.where(exponent_bits == 0, -133, exponent_bits - 134)

    records = scale32_records.to(device=weight_bf16.device, dtype=torch.int64)
    scale_significand = torch.bitwise_and(records, 0xFFFF)
    scale_exponent_u8 = torch.bitwise_and(torch.bitwise_right_shift(records, 16), 0xFF)
    scale_exponent = torch.where(
        scale_exponent_u8 >= 128,
        scale_exponent_u8 - 256,
        scale_exponent_u8,
    )
    shift = weight_exponent - scale_exponent[:, None] + 15
    nonzero = mantissa != 0
    require(
        not torch.any(nonzero & (shift > 20)).item(),
        "c01 exact W4 division exceeded the bounded shift",
    )
    active = nonzero & (shift >= 6)
    safe_shift = shift.clamp(min=0, max=20)
    exact_numerator = torch.bitwise_left_shift(mantissa, safe_shift)
    exact_denominator = scale_significand[:, None]
    quotient = torch.div(exact_numerator, exact_denominator, rounding_mode="floor")
    remainder = exact_numerator - quotient * exact_denominator
    doubled_remainder = remainder * 2
    increment = (doubled_remainder > exact_denominator) | (
        (doubled_remainder == exact_denominator) & ((quotient & 1) == 1)
    )
    rounded = torch.where(active, quotient + increment.to(torch.int64), 0)
    signed = torch.where((bits & 0x8000) != 0, -rounded, rounded)
    return signed.clamp(-8, 7).to(torch.int8)


def _clip_grid_row_candidate(
    weight_bf16: torch.Tensor,
    row_absmax_bf16: torch.Tensor,
    numerator: int,
    denominator: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    records = _clip_scale32_records(row_absmax_bf16, numerator, denominator)
    scale = _scale32_values(records)
    qweight = _quantize_bf16_with_scale32(weight_bf16, records)
    weight = weight_bf16.to(torch.float64)
    reconstruction = qweight.to(torch.float64) * scale[:, None]
    error = torch.sum(
        (weight - reconstruction) * (weight - reconstruction),
        dim=1,
        dtype=torch.float64,
    )
    return qweight, scale, records, error


def mse_clip_grid_quantization_scale32(
    source_weight: torch.Tensor,
    candidate: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    require(candidate["candidate_id"] == "c01-mse-clip-grid", "c01 candidate identity differs")
    generation = candidate["weight_generation"]
    numerators = generation["clip_ratio_numerators"]
    denominator = generation["clip_ratio_denominator"]
    require(numerators == [1000, 990, 975, 950, 925, 900], "c01 clip grid differs")
    require(denominator == 1000, "c01 clip denominator differs")
    require(
        generation.get("code_selection")
        in {None, "round_to_nearest_ties_to_even_then_saturate_signed_w4"},
        "c01 code-selection semantics differ",
    )
    require(
        generation.get("objective")
        in {None, "sum_of_squared_bf16_weight_error_accumulated_in_binary64"},
        "c01 objective semantics differ",
    )
    require(
        generation.get("tie_break")
        in {None, "larger_clip_ratio_then_lower_candidate_index"},
        "c01 tie-break semantics differ",
    )
    weight_bf16 = source_weight.detach().to(torch.bfloat16)
    require(weight_bf16.ndim == 2, "c01 source weight must be a matrix")
    row_absmax_bf16 = weight_bf16.abs().amax(dim=1)
    best_error = torch.full(
        (weight_bf16.shape[0],),
        float("inf"),
        dtype=torch.float64,
        device=weight_bf16.device,
    )
    best_scale = torch.ones_like(best_error)
    best_records = torch.zeros_like(best_error, dtype=torch.int64)
    best_qweight = torch.zeros_like(weight_bf16, dtype=torch.int8)
    best_numerator = torch.full_like(best_records, numerators[0])
    for numerator in numerators:
        qweight, scale, records, error = _clip_grid_row_candidate(
            weight_bf16,
            row_absmax_bf16,
            numerator,
            denominator,
        )
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale, best_scale)
        best_records = torch.where(better, records, best_records)
        best_qweight = torch.where(better[:, None], qweight, best_qweight)
        best_numerator = torch.where(
            better,
            torch.full_like(best_numerator, numerator),
            best_numerator,
        )
    require(torch.equal(best_scale, _scale32_values(best_records)), "c01 Scale32 values drifted")
    return best_qweight, best_scale, best_numerator, best_records


def mse_clip_grid_quantization(
    source_weight: torch.Tensor,
    candidate: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qweight, scale, numerator, _records = mse_clip_grid_quantization_scale32(
        source_weight,
        candidate,
    )
    return qweight, scale, numerator


class MSEClipGridW4A8Linear(fixed.W4A8Linear):
    def __init__(
        self,
        source: nn.Linear,
        calibration: fixed.CalibrationRange,
        candidate: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(source, calibration, **kwargs)
        self.input_scale = realize_positive_scale32(self.input_scale)[1]
        self.hardware_input_scale = self.input_scale
        if self.output_scale is None:
            output_head_scales = torch.tensor(
                [
                    realize_positive_scale32(float(value))[1]
                    for value in self.output_head_scales.detach().cpu().tolist()
                ],
                dtype=torch.float64,
                device=self.output_head_scales.device,
            )
            self.output_head_scales.copy_(output_head_scales)
            self.output_scale_per_channel.copy_(
                output_head_scales.repeat_interleave(self.output_head_size)
            )
        else:
            self.output_scale32_record, self.output_scale = realize_positive_scale32(
                self.output_scale
            )
            self.output_scale_per_channel.fill_(self.output_scale)
        (
            qweight,
            weight_scale,
            clip_ratio_numerator,
            weight_scale32_records,
        ) = mse_clip_grid_quantization_scale32(
            source.weight,
            candidate,
        )
        self.qweight.copy_(qweight)
        self.qweight_transposed.copy_(qweight.transpose(0, 1).contiguous())
        self.weight_scale.copy_(weight_scale)
        self.register_buffer(
            "weight_scale32_records",
            weight_scale32_records,
            persistent=True,
        )
        self.register_buffer(
            "clip_ratio_numerator",
            clip_ratio_numerator,
            persistent=True,
        )
        self.native_scale32_records.copy_(
            self._native_scale32_for_input_scale(self.input_scale)
        )
        multiplier, right_shift, bias_accumulator = self._metadata_for_input_scale(
            self.input_scale
        )
        self.multiplier.copy_(multiplier)
        self.right_shift.copy_(right_shift)
        if self.bias_accumulator is not None:
            require(bias_accumulator is not None, "c01 bias metadata disappeared")
            self.bias_accumulator.copy_(bias_accumulator)

    def bind_hardware_input_scale(self, input_scale: float) -> None:
        super().bind_hardware_input_scale(realize_positive_scale32(input_scale)[1])


def replace_linears_for_candidate(
    model: nn.Module,
    ranges: dict[str, fixed.CalibrationRange],
    candidate: dict[str, Any],
    *,
    rope_diagnostic_mechanism: str | None,
) -> None:
    require(
        candidate["candidate_id"] == "c01-mse-clip-grid",
        "c00 linear regeneration/execution is terminally prohibited",
    )
    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    for name, module in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        use_per_head_output_scales, emit_scale32 = fixed.rope_linear_contract(
            name,
            rope_diagnostic_mechanism,
        )
        frozen_output_head_absmax = fixed.qk_residual_projection_output_absmax(
            name,
            rope_diagnostic_mechanism,
        )
        setattr(
            parent,
            child_name,
            MSEClipGridW4A8Linear(
                module,
                ranges[name],
                candidate,
                output_head_absmax=(
                    frozen_output_head_absmax
                    if frozen_output_head_absmax is not None
                    else ranges[name].output_head_absmax
                    if use_per_head_output_scales
                    else None
                ),
                output_head_size=(fixed.ROPE_HEAD_DIM if use_per_head_output_scales else None),
                scale32_output=emit_scale32,
            ),
        )


def _scale32_runtime_operator_ranges(
    model: nn.Module,
    operator_ranges: dict[str, fixed.ObservedRange],
) -> dict[str, fixed.ObservedRange]:
    norm_outputs: dict[str, nn.Module] = {
        "model.norm.output": model.model.norm,
    }
    for index, layer in enumerate(model.model.layers):
        norm_outputs[f"model.layers.{index}.input_layernorm.output"] = (
            layer.input_layernorm
        )
        norm_outputs[f"model.layers.{index}.post_attention_layernorm.output"] = (
            layer.post_attention_layernorm
        )
    require(
        set(norm_outputs) <= set(operator_ranges),
        "c01 RMSNorm calibration observations are incomplete",
    )
    runtime_ranges: dict[str, fixed.ObservedRange] = {}
    for name, observed in operator_ranges.items():
        required_scale = (
            fixed.rmsnorm_output_scale(norm_outputs[name], observed.absmax)
            if name in norm_outputs
            else fixed.positive_scale(observed.absmax)
        )
        realized_scale = realize_positive_scale32(required_scale)[1]
        runtime_ranges[name] = fixed.ObservedRange(
            absmax=realized_scale * 127.0,
            percentile_absmax=None,
        )
        reproduced = (
            fixed.rmsnorm_output_scale(
                norm_outputs[name],
                runtime_ranges[name].absmax,
            )
            if name in norm_outputs
            else fixed.positive_scale(runtime_ranges[name].absmax)
        )
        require(reproduced == realized_scale, f"c01 operator Scale32 drifted: {name}")
    return runtime_ranges


def replace_fixed_operators_for_candidate(
    model: nn.Module,
    operator_ranges: dict[str, fixed.ObservedRange],
    candidate: dict[str, Any],
    *,
    rope_diagnostic_mechanism: str | None,
) -> dict[str, fixed.ObservedRange]:
    require(
        candidate["candidate_id"] == "c01-mse-clip-grid",
        "c00 operator execution is terminally prohibited",
    )
    runtime_ranges = _scale32_runtime_operator_ranges(model, operator_ranges)
    fixed.replace_fixed_operators(
        model,
        runtime_ranges,
        rope_diagnostic_mechanism=rope_diagnostic_mechanism,
    )
    return runtime_ranges


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack(">I", tensor.ndim))
    for dimension in tensor.shape:
        digest.update(struct.pack(">Q", int(dimension)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _contains_unicode_letter(value: str) -> bool:
    return any(unicodedata.category(character).startswith("L") for character in value)


def _invalid_output_characters(value: str) -> list[str]:
    invalid: list[str] = []
    for character in value:
        codepoint = ord(character)
        category = unicodedata.category(character)
        if character == "\ufffd" or 0xD800 <= codepoint <= 0xDFFF:
            invalid.append(f"U+{codepoint:04X}")
        elif category == "Cc" and character not in "\t\n\r":
            invalid.append(f"U+{codepoint:04X}")
    return invalid


def _visible_token_ids(
    tokenizer: Any,
    token_ids: list[int],
    termination_token_ids: set[int],
) -> list[int]:
    visible: list[int] = []
    special_ids = set(tokenizer.all_special_ids) | termination_token_ids
    for token_id in token_ids:
        if token_id in special_ids:
            continue
        fragment = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if any(not character.isspace() for character in fragment):
            visible.append(token_id)
    return visible


def _maximum_identical_run(token_ids: list[int]) -> int:
    maximum = 0
    current = 0
    previous: int | None = None
    for token_id in token_ids:
        current = current + 1 if token_id == previous else 1
        maximum = max(maximum, current)
        previous = token_id
    return maximum


def global_generation_checks(
    tokenizer: Any,
    token_ids: list[int],
    decoded_text: str,
    prompt: dict[str, Any],
    generation: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    termination_ids = set(generation["termination_token_ids"])
    visible_ids = _visible_token_ids(tokenizer, token_ids, termination_ids)
    visible_characters = [character for character in decoded_text if not character.isspace()]
    alphanumeric = sum(
        unicodedata.category(character).startswith(("L", "N"))
        for character in visible_characters
    )
    alphanumeric_fraction = (
        alphanumeric / len(visible_characters) if visible_characters else 0.0
    )
    unique_fraction = (
        len(set(visible_ids)) / len(visible_ids) if visible_ids else 0.0
    )
    invalid_characters = _invalid_output_characters(decoded_text)
    punctuation_only = bool(visible_characters) and all(
        unicodedata.category(character).startswith(("P", "S"))
        for character in visible_characters
    )
    minimum_visible = (
        gate["short_answer_minimum_visible_tokens"]
        if prompt["short_answer_exception"]
        else gate["minimum_visible_tokens"]
    )
    checks = {
        "alphanumeric_character_fraction": (
            alphanumeric_fraction >= gate["minimum_alphanumeric_character_fraction"]
        ),
        "identical_token_run": (
            _maximum_identical_run(visible_ids) <= gate["maximum_identical_token_run"]
        ),
        "minimum_visible_tokens": len(visible_ids) >= minimum_visible,
        "punctuation_only_rejected": not punctuation_only,
        "replacement_surrogate_or_control_character_rejected": not invalid_characters,
        "unique_token_fraction": (
            len(visible_ids) < 8
            or unique_fraction >= gate["minimum_unique_token_fraction_when_at_least_8_tokens"]
        ),
    }
    return {
        "alphanumeric_character_fraction": alphanumeric_fraction,
        "checks": checks,
        "invalid_characters": invalid_characters,
        "maximum_identical_token_run": _maximum_identical_run(visible_ids),
        "passed": all(checks.values()),
        "punctuation_only": punctuation_only,
        "unique_token_fraction": unique_fraction,
        "visible_token_count": len(visible_ids),
        "visible_token_ids": visible_ids,
    }


def _strip_terminal_unicode_punctuation(value: str) -> str:
    stripped = value.strip()
    while stripped and unicodedata.category(stripped[-1]).startswith("P"):
        stripped = stripped[:-1].rstrip()
    return stripped


def prompt_validator(name: str, decoded_text: str) -> dict[str, Any]:
    detail: dict[str, Any] = {"name": name}
    if name == "strip_ascii_whitespace_then_exactly_19":
        normalized = decoded_text.strip(ASCII_WHITESPACE)
        passed = normalized == "19"
        detail["normalized"] = normalized
    elif name == "readability_only":
        passed = bool(decoded_text.strip()) and _contains_unicode_letter(decoded_text)
    elif name == "parse_json_then_exact_object_status_ok_count_3":
        try:
            parsed = json.loads(decoded_text)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        passed = type(parsed) is dict and parsed == {"status": "ok", "count": 3}
        detail["parsed"] = parsed
    elif name == "exactly_three_nonempty_lines_each_containing_a_unicode_letter":
        lines = decoded_text.splitlines()
        passed = len(lines) == 3 and all(
            line.strip() and _contains_unicode_letter(line) for line in lines
        )
        detail["lines"] = lines
    elif name == "unicode_casefold_strip_terminal_punctuation_in_buenos_dias_set":
        normalized = unicodedata.normalize("NFC", decoded_text).casefold()
        normalized = _strip_terminal_unicode_punctuation(normalized)
        passed = normalized in ACCEPTED_GOOD_MORNING_SPANISH
        detail["accepted_set"] = sorted(ACCEPTED_GOOD_MORNING_SPANISH)
        detail["normalized"] = normalized
    elif name == "exactly_two_list_items_with_unicode_letters":
        lines = [line.strip() for line in decoded_text.splitlines() if line.strip()]
        contents = [LIST_PREFIX.sub("", line, count=1) for line in lines]
        passed = (
            len(lines) == 2
            and all(LIST_PREFIX.match(line) for line in lines)
            and all(_contains_unicode_letter(content) for content in contents)
        )
        detail["items"] = contents
        detail["lines"] = lines
    else:
        raise ValueError(f"unsupported canonical generation validator: {name}")
    detail["passed"] = bool(passed)
    return detail


def evaluate_generation(
    tokenizer: Any,
    prompt_results: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    generation: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    prompt_by_id = {prompt["prompt_id"]: prompt for prompt in prompts}
    require(len(prompt_by_id) == len(prompts), "canonical prompt IDs alias")
    evaluations: list[dict[str, Any]] = []
    for result in prompt_results:
        prompt = prompt_by_id[result["prompt_id"]]
        global_result = global_generation_checks(
            tokenizer,
            result["generated_token_ids"],
            result["decoded_text"],
            prompt,
            generation,
            gate,
        )
        validator_result = prompt_validator(prompt["validator"], result["decoded_text"])
        evaluations.append(
            {
                "global": global_result,
                "passed": global_result["passed"] and validator_result["passed"],
                "prompt_id": result["prompt_id"],
                "validator": validator_result,
            }
        )
    require(
        {item["prompt_id"] for item in evaluations} == set(prompt_by_id),
        "canonical generation results are incomplete",
    )
    return {
        "all_prompt_validators_must_pass": gate["all_prompt_validators_must_pass"],
        "passed": all(item["passed"] for item in evaluations),
        "prompts": evaluations,
    }


def evaluate_quality(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: dict[str, Any],
    tasks: list[str],
) -> dict[str, Any]:
    require(
        thresholds["applies_independently_to_both_models"] is True,
        "quality gate is not model-independent",
    )
    require(
        thresholds["mission_local_not_project_final_emnlp_gate"] is True,
        "quality gate boundary differs",
    )
    metrics: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for dataset, threshold_name in (
        ("wikitext2", "wikitext2_perplexity_ratio_max"),
        ("c4_en_512", "c4_en_512_perplexity_ratio_max"),
    ):
        bf16 = float(baseline["perplexity"][dataset]["perplexity"])
        w4a8 = float(candidate["perplexity"][dataset]["perplexity"])
        require(math.isfinite(bf16) and bf16 > 0, f"invalid BF16 {dataset} perplexity")
        require(math.isfinite(w4a8) and w4a8 > 0, f"invalid W4A8 {dataset} perplexity")
        ratio = w4a8 / bf16
        metrics[dataset] = {"bf16": bf16, "ratio": ratio, "w4a8": w4a8}
        checks[f"{dataset}_perplexity_ratio"] = ratio <= thresholds[threshold_name]

    baseline_tasks = baseline["lm_eval"]["tasks"]
    candidate_tasks = candidate["lm_eval"]["tasks"]
    require(set(baseline_tasks) == set(candidate_tasks) == set(tasks), "lm-eval task set differs")
    task_drops: dict[str, float] = {}
    task_checks: dict[str, bool] = {}
    for task in tasks:
        bf16 = float(baseline_tasks[task])
        w4a8 = float(candidate_tasks[task])
        require(
            math.isfinite(bf16) and math.isfinite(w4a8) and 0.0 <= bf16 <= 1.0 and 0.0 <= w4a8 <= 1.0,
            f"invalid lm-eval metric for {task}",
        )
        drop = 100.0 * (bf16 - w4a8)
        task_drops[task] = drop
        task_checks[task] = (
            drop <= thresholds["lm_eval_individual_task_drop_percentage_points_max"]
        )
    average_drop = 100.0 * (
        float(baseline["lm_eval"]["average_normalized_accuracy"])
        - float(candidate["lm_eval"]["average_normalized_accuracy"])
    )
    checks["lm_eval_average_normalized_accuracy_drop"] = (
        average_drop
        <= thresholds["lm_eval_average_normalized_accuracy_drop_percentage_points_max"]
    )
    checks["lm_eval_individual_task_drop"] = all(task_checks.values())
    metrics["lm_eval"] = {
        "average_drop_percentage_points": average_drop,
        "individual_task_checks": task_checks,
        "individual_task_drop_percentage_points": task_drops,
    }
    return {"checks": checks, "metrics": metrics, "passed": all(checks.values())}


def snapshot_path(repository: str, revision: str) -> Path:
    return (
        Path.home()
        / ".cache/huggingface/hub"
        / f"models--{repository.replace('/', '--')}"
        / "snapshots"
        / revision
    )


def dataset_snapshot_path(repository: str, revision: str) -> Path:
    return (
        Path.home()
        / ".cache/huggingface/hub"
        / f"datasets--{repository.replace('/', '--')}"
        / "snapshots"
        / revision
    )


def local_dataset_files(spec: dict[str, Any]) -> list[Path]:
    root = dataset_snapshot_path(spec["repository"], spec["revision"])
    require(root.is_dir(), f"local dataset snapshot is missing: {spec['repository']}")
    if spec["repository"] == "allenai/c4":
        require(spec["config"] == "en", "C4 config differs")
        files = sorted((root / "en").glob(f"c4-{spec['split']}.*.json.gz"))
    elif spec["repository"] == "Salesforce/wikitext":
        files = sorted((root / spec["config"]).glob(f"{spec['split']}-*.parquet"))
    else:
        raise ValueError(f"unsupported bounded local dataset: {spec['repository']}")
    require(files, f"local dataset files are missing: {spec['repository']}")
    require(all(path.is_file() for path in files), "local dataset snapshot is incomplete")
    return files


def selected_local_texts(spec: dict[str, Any]) -> list[str]:
    files = local_dataset_files(spec)
    builder = "json" if spec["repository"] == "allenai/c4" else "parquet"
    dataset = load_dataset(
        builder,
        data_files={spec["split"]: [str(path) for path in files]},
        split=spec["split"],
        streaming=True,
    )
    start = spec["indices"]["start"]
    stop = spec["indices"]["stop"]
    texts: list[str] = []
    for index, row in enumerate(dataset):
        if index >= stop:
            break
        if index >= start:
            texts.append(row[spec["field"]])
    require(
        len(texts) == stop - start,
        f"local {spec['repository']} slice returned {len(texts)} records",
    )
    return texts


def load_tokenizer(spec: dict[str, Any]) -> Any:
    return AutoTokenizer.from_pretrained(
        spec["repository"],
        revision=spec["revision"],
        local_files_only=True,
    )


def load_model(spec: dict[str, Any]) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        spec["repository"],
        revision=spec["revision"],
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=None,
    )
    if spec["alias"] == "checkpoint-176":
        from peft import PeftModel

        adapter_path = ROOT / spec["adapter"]["adapter_config"]["path"]
        model = PeftModel.from_pretrained(
            model,
            adapter_path.parent,
            is_trainable=False,
            local_files_only=True,
        ).merge_and_unload()
    return model.eval()


def _tokenize_calibration(
    tokenizer: Any,
    texts: list[str],
    generation: dict[str, Any],
    token_limit: int,
) -> list[torch.Tensor]:
    prompts: list[torch.Tensor] = []
    for text in texts:
        require(bool(text.strip()), "calibration C4 record is empty")
        input_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": generation["canonical_system_message"]},
                {"role": "user", "content": text},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )[:, :token_limit]
        require(input_ids.shape[1] >= 2, "calibration prompt has fewer than two tokens")
        prompts.append(input_ids.to(dtype=torch.long, device="cpu"))
    return prompts


def load_bounded_inputs(
    tokenizer: Any,
    contract: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, list[torch.Tensor]], dict[str, Any]]:
    generation = contract["shared_w4a8_contract"]["generation"]
    calibration_texts = selected_local_texts(manifest["datasets"]["c4_calibration"])
    c4_texts = selected_local_texts(manifest["datasets"]["c4_en_512"])
    wiki_texts = selected_local_texts(manifest["datasets"]["wikitext2"])
    inputs = {
        "calibration": _tokenize_calibration(
            tokenizer,
            calibration_texts,
            generation,
            contract["evaluation_contract"]["calibration_set"]["token_limit"],
        ),
        "c4_en_512": fixed.tokenize_prompts(
            tokenizer,
            c4_texts,
            manifest["datasets"]["c4_en_512"]["token_limit"],
        ),
        "wikitext2": fixed.tokenize_wikitext(
            tokenizer,
            wiki_texts,
            manifest["datasets"]["wikitext2"]["token_limit"],
            manifest["datasets"]["wikitext2"]["join"],
        )[: contract["evaluation_contract"]["bounded_quality_sets"]["wikitext2_perplexity"]["first_complete_windows"]],
    }
    observations: dict[str, Any] = {}
    for name, texts in (
        ("c4_calibration", calibration_texts),
        ("c4_en_512", c4_texts),
        ("wikitext2", wiki_texts),
    ):
        record_sha, record_count = fixed.hash_records(texts)
        observations[name] = {
            "config": manifest["datasets"][name]["config"],
            "record_count": record_count,
            "record_sha256": record_sha,
            "repository": manifest["datasets"][name]["repository"],
            "revision": manifest["datasets"][name]["revision"],
            "split": manifest["datasets"][name]["split"],
        }
    observation_names = {
        "calibration": "c4_calibration",
        "c4_en_512": "c4_en_512",
        "wikitext2": "wikitext2",
    }
    for name, prompts in inputs.items():
        token_sha, sequence_count, token_count = fixed.hash_token_sequences(prompts)
        observations[observation_names[name]]["tokenized"] = {
            "sequence_count": sequence_count,
            "token_count": token_count,
            "token_sequence_sha256": token_sha,
        }
    return inputs, observations


def _perplexity_pair(model: nn.Module, inputs: dict[str, list[torch.Tensor]]) -> dict[str, Any]:
    return {
        "c4_en_512": fixed.perplexity(model, inputs["c4_en_512"]),
        "wikitext2": fixed.perplexity(model, inputs["wikitext2"]),
    }


def _quality_measurement(
    model: nn.Module,
    tokenizer: Any,
    inputs: dict[str, list[torch.Tensor]],
    manifest: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    label: str,
) -> dict[str, Any]:
    perplexity = _perplexity_pair(model, inputs)
    write_json(output_root / f"perplexity-{label}.json", perplexity)
    fixed.seed_everything(config)
    lm_eval, lm_eval_raw = fixed.run_lm_eval(model, tokenizer, manifest, config)
    fixed.write_json(
        output_root / f"lm-eval-{label}-raw.json",
        lm_eval_raw,
        default=fixed.lm_eval_json_default,
    )
    result = {"lm_eval": lm_eval, "perplexity": perplexity}
    write_json(output_root / f"quality-{label}.json", result)
    return result


def render_conversation_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> torch.Tensor:
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(dtype=torch.long, device="cpu")


def _render_prompt_ids(tokenizer: Any, prompt: dict[str, Any], generation: dict[str, Any]) -> torch.Tensor:
    return render_conversation_ids(
        tokenizer,
        [
            {"role": "system", "content": generation["canonical_system_message"]},
            {"role": "user", "content": prompt["user"]},
        ],
    )


def _cache_aware_dynamic_attention_forward(
    self: fixed.FixedAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    if past_key_values is None:
        return self._ace2_cache_free_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
            **kwargs,
        )
    require(self.dynamic_rope_head_scale, "cache runtime requires dynamic RoPE scales")
    require(
        not any(
            (
                self.layer0_fixed_q7,
                self.layer0_relative_rope_score_fusion,
                self.layer0_absolute_rope_online_attention,
                self.layer0_projection_shadow_staged_attention,
                self.layer0_tile_max_delta_attention,
                self.layer0_tile_bfp_score_attention,
                self.shared_native_accumulator_tagged_attention,
                self.shared_q20_44_score_mode,
            )
        ),
        "cache runtime supports only the frozen dynamic-RoPE mechanism",
    )
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query = (
        self.q_proj.forward_hardware_input(hidden_states)
        .view(hidden_shape)
        .transpose(1, 2)
    )
    key = (
        self.k_proj.forward_hardware_input(hidden_states)
        .view(hidden_shape)
        .transpose(1, 2)
    )
    value = (
        self.v_proj.forward_hardware_input(hidden_states)
        .view(hidden_shape)
        .transpose(1, 2)
    )
    cos, sin = position_embeddings
    self.query_rope_output_elements += query.numel()
    query, query_scale32, query_saturations = fixed.dynamic_rope_head_raw(
        query, self.query_producer_scale32, cos, sin
    )
    self.key_rope_output_elements += key.numel()
    key, key_scale32, key_saturations = fixed.dynamic_rope_head_raw(
        key, self.key_producer_scale32, cos, sin
    )
    self.query_rope_output_saturations += query_saturations
    self.key_rope_output_saturations += key_saturations
    query_min = int(query_scale32.min())
    query_max = int(query_scale32.max())
    key_min = int(key_scale32.min())
    key_max = int(key_scale32.max())
    self.query_dynamic_scale_min = (
        query_min
        if self.query_dynamic_scale_min is None
        else min(self.query_dynamic_scale_min, query_min)
    )
    self.query_dynamic_scale_max = (
        query_max
        if self.query_dynamic_scale_max is None
        else max(self.query_dynamic_scale_max, query_max)
    )
    self.key_dynamic_scale_min = (
        key_min
        if self.key_dynamic_scale_min is None
        else min(self.key_dynamic_scale_min, key_min)
    )
    self.key_dynamic_scale_max = (
        key_max
        if self.key_dynamic_scale_max is None
        else max(self.key_dynamic_scale_max, key_max)
    )

    past_length = past_key_values.get_seq_length(self.layer_idx)
    if past_length == 0:
        key_scale_cache = key_scale32
    else:
        previous_scales = getattr(self, "ace2_key_scale32_cache", None)
        require(previous_scales is not None, "KV key Scale32 cache is missing")
        require(
            previous_scales.shape[-1] == past_length,
            "KV key Scale32 cache length differs",
        )
        key_scale_cache = torch.cat((previous_scales, key_scale32), dim=2)
    full_key, full_value = past_key_values.update(
        key,
        value,
        self.layer_idx,
        {"cache_position": kwargs.get("cache_position")},
    )
    require(full_key.dtype == torch.int8, "cached key payload is not signed int8")
    require(full_value.dtype == torch.int8, "cached value payload is not signed int8")
    require(
        full_key.shape[-2] == key_scale_cache.shape[-1],
        "KV key payload and Scale32 lengths differ",
    )
    self.ace2_key_scale32_cache = key_scale_cache
    mapped_key = fixed.repeat_kv(full_key, self.num_key_value_groups)
    mapped_value = fixed.repeat_kv(full_value, self.num_key_value_groups)
    mapped_key_scales = key_scale_cache.repeat_interleave(
        self.num_key_value_groups, dim=1
    )
    scores = fixed.fixed_dynamic_attention_scores_raw(
        query,
        mapped_key,
        query_scale32,
        mapped_key_scales,
        attention_mask,
    )
    probabilities = fixed.fixed_softmax_raw(scores)
    output = fixed.fixed_attention_value_raw(probabilities, mapped_value)
    output = output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return self._project_attention_output(output, hidden_states), None


def _traced_w4a8_linear_forward(
    self: fixed.W4A8Linear,
    inputs: torch.Tensor,
) -> torch.Tensor:
    raw = (
        self.forward_hardware_input(inputs)
        if self.input_is_quantized
        else self.forward_raw(inputs)
    )
    self.last_raw_output = raw
    return raw.to(inputs.dtype) * self.output_scale_per_channel.to(inputs.dtype)


def enable_w4a8_kv_cache(model: nn.Module) -> None:
    for layer in model.model.layers:
        attention = layer.self_attn
        require(
            isinstance(attention, fixed.FixedAttention),
            "KV-cache binding requires fixed attention modules",
        )
        require(
            attention.rope_diagnostic_mechanism == fixed.ACTIVE_ROPE_MECHANISM,
            "KV-cache binding mechanism differs",
        )
        if not hasattr(attention, "_ace2_cache_free_forward"):
            attention._ace2_cache_free_forward = attention.forward
            attention.forward = types.MethodType(
                _cache_aware_dynamic_attention_forward, attention
            )
    require(isinstance(model.lm_head, fixed.W4A8Linear), "integer LM head is missing")
    if not hasattr(model.lm_head, "last_raw_output"):
        model.lm_head.last_raw_output = None
        model.lm_head.forward = types.MethodType(
            _traced_w4a8_linear_forward, model.lm_head
        )


def kv_cache_step_record(model: nn.Module, past_key_values: Any) -> dict[str, Any]:
    require(len(past_key_values.layers) == len(model.model.layers) == 24, "KV layer count differs")
    layers: list[dict[str, Any]] = []
    for index, (cache_layer, decoder_layer) in enumerate(
        zip(past_key_values.layers, model.model.layers, strict=True)
    ):
        scales = decoder_layer.self_attn.ace2_key_scale32_cache
        require(cache_layer.keys.dtype == torch.int8, "KV key cache dtype differs")
        require(cache_layer.values.dtype == torch.int8, "KV value cache dtype differs")
        require(
            scales.shape == cache_layer.keys.shape[:-1],
            "KV key Scale32 geometry differs",
        )
        for record in scales.detach().cpu().reshape(-1).tolist():
            unpack_scale32(record)
        layers.append(
            {
                "key_scale32_sha256": tensor_sha256(scales),
                "key_sha256": tensor_sha256(cache_layer.keys),
                "layer_index": index,
                "sequence_length": cache_layer.keys.shape[-2],
                "value_sha256": tensor_sha256(cache_layer.values),
            }
        )
    return {
        "layers": layers,
        "sha256": canonical_sha256(layers),
    }


@dataclass
class W4A8ContinuationState:
    token_sequence: torch.Tensor
    past_key_values: Any
    key_scale32_by_layer: tuple[torch.Tensor, ...]


def _continuation_cache_length(state: W4A8ContinuationState) -> int:
    require(
        len(state.past_key_values.layers)
        == len(state.key_scale32_by_layer)
        == 24,
        "continuation state layer count differs",
    )
    lengths = {
        int(layer.keys.shape[-2]) for layer in state.past_key_values.layers
    }
    require(len(lengths) == 1, "continuation K/V cache lengths differ")
    cache_length = lengths.pop()
    require(
        all(
            int(scales.shape[-1]) == cache_length
            for scales in state.key_scale32_by_layer
        ),
        "continuation Scale32 cache lengths differ",
    )
    return cache_length


def _snapshot_continuation_state(
    model: nn.Module,
    sequence: torch.Tensor,
    past_key_values: Any,
) -> W4A8ContinuationState:
    require(past_key_values is not None, "continuation K/V cache is missing")
    scales = tuple(
        layer.self_attn.ace2_key_scale32_cache.detach().clone()
        for layer in model.model.layers
    )
    state = W4A8ContinuationState(
        token_sequence=sequence.detach().clone(),
        past_key_values=past_key_values,
        key_scale32_by_layer=scales,
    )
    _continuation_cache_length(state)
    return state


def _restore_continuation_scales(
    model: nn.Module,
    state: W4A8ContinuationState,
) -> None:
    require(
        len(model.model.layers) == len(state.key_scale32_by_layer) == 24,
        "continuation model layer count differs",
    )
    for layer, scales in zip(
        model.model.layers,
        state.key_scale32_by_layer,
        strict=True,
    ):
        layer.self_attn.ace2_key_scale32_cache = scales.detach().clone()


def generate_w4a8_continuation(
    model: nn.Module,
    tokenizer: Any,
    prompt_id: str,
    input_ids: torch.Tensor,
    generation: dict[str, Any],
    state: W4A8ContinuationState | None = None,
) -> tuple[dict[str, Any], W4A8ContinuationState]:
    require(
        input_ids.dtype == torch.long
        and input_ids.device.type == "cpu"
        and input_ids.ndim == 2
        and input_ids.shape[0] == 1
        and input_ids.shape[1] > 0,
        "conversation input token geometry differs",
    )
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    sequence = input_ids.clone()
    if state is None:
        current_ids = input_ids
        past_key_values = None
    else:
        carried_length = int(state.token_sequence.shape[-1])
        require(
            state.token_sequence.dtype == torch.long
            and state.token_sequence.device.type == "cpu"
            and state.token_sequence.shape[0] == 1,
            "continuation token state geometry differs",
        )
        require(
            input_ids.shape[-1] >= carried_length
            and torch.equal(
                input_ids[:, :carried_length],
                state.token_sequence,
            ),
            "conversation prefix/template drift detected",
        )
        cache_length = _continuation_cache_length(state)
        require(
            cache_length <= carried_length
            and carried_length - cache_length <= 1,
            "continuation token and K/V cache lengths differ",
        )
        _restore_continuation_scales(model, state)
        current_ids = input_ids[:, cache_length:]
        require(
            current_ids.shape[-1] > 0,
            "conversation continuation has no uncached tokens",
        )
        past_key_values = state.past_key_values

    generated: list[int] = []
    kv_cache_records: list[dict[str, Any]] = []
    kv_cache_hashes: list[str] = []
    raw_logit_hashes: list[str] = []
    termination_reason = "max_new_tokens"
    for _step in range(generation["max_new_tokens"]):
        attention_mask = torch.ones_like(sequence, dtype=torch.long)
        output = model(
            input_ids=current_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        require(
            torch.isfinite(output.logits).all().item(),
            "W4A8 generation produced non-finite logits",
        )
        raw_logits = model.lm_head.last_raw_output
        require(
            raw_logits is not None and raw_logits.dtype == torch.int8,
            "integer lm-head logits are missing",
        )
        raw_last = raw_logits[:, -1, :].contiguous()
        selected = int(torch.argmax(raw_last, dim=-1).item())
        generated.append(selected)
        raw_logit_hashes.append(tensor_sha256(raw_last))
        past_key_values = output.past_key_values
        require(
            past_key_values is not None,
            "W4A8 production decode did not return a KV cache",
        )
        cache_record = kv_cache_step_record(model, past_key_values)
        kv_cache_records.append(cache_record)
        kv_cache_hashes.append(cache_record["sha256"])
        next_token = torch.tensor([[selected]], dtype=torch.long)
        sequence = torch.cat((sequence, next_token), dim=1)
        current_ids = next_token
        if selected in generation["termination_token_ids"]:
            termination_reason = f"termination_token_id:{selected}"
            break
    decoded = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=generation[
            "clean_up_tokenization_spaces"
        ],
    )
    result = {
        "decoded_text": decoded,
        "decoded_text_sha256": hashlib.sha256(
            decoded.encode("utf-8")
        ).hexdigest(),
        "generated_token_ids": generated,
        "generated_token_ids_sha256": canonical_sha256(generated),
        "input_token_ids": input_ids.reshape(-1).tolist(),
        "per_step_integer_logit_tensor_sha256": raw_logit_hashes,
        "per_step_kv_cache": kv_cache_records,
        "per_step_kv_cache_sha256": kv_cache_hashes,
        "prompt_id": prompt_id,
        "termination_reason": termination_reason,
        "use_cache": True,
    }
    return result, _snapshot_continuation_state(
        model,
        sequence,
        past_key_values,
    )


def generate_w4a8(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    generation: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for prompt in prompts:
            input_ids = _render_prompt_ids(tokenizer, prompt, generation)
            result, _state = generate_w4a8_continuation(
                model,
                tokenizer,
                prompt["prompt_id"],
                input_ids,
                generation,
            )
            results.append(result)
    return results


def _calibration_observations(
    ranges: dict[str, fixed.CalibrationRange],
    operator_ranges: dict[str, fixed.ObservedRange],
) -> dict[str, Any]:
    return {
        "linears": {
            name: {
                "input_absmax": value.input_absmax,
                "output_absmax": value.output_absmax,
                "output_head_absmax": value.output_head_absmax,
            }
            for name, value in sorted(ranges.items())
        },
        "operators": {
            name: {"absmax": value.absmax}
            for name, value in sorted(operator_ranges.items())
        },
    }


def _write_packed_w4(model: nn.Module, path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    with path.open("xb") as handle:
        for name, module in sorted(model.named_modules()):
            if not isinstance(module, fixed.W4A8Linear):
                continue
            qweight = module.qweight.detach().cpu().contiguous()
            require(qweight.shape[1] % 2 == 0, f"W4 packing requires an even input width: {name}")
            unsigned = torch.bitwise_and(qweight.to(torch.int16), 0xF).to(torch.uint8)
            packed = unsigned[:, 0::2] | torch.bitwise_left_shift(unsigned[:, 1::2], 4)
            payload = packed.contiguous().numpy().tobytes(order="C")
            handle.write(payload)
            records.append(
                {
                    "bytes": len(payload),
                    "in_features": module.in_features,
                    "name": name,
                    "offset": offset,
                    "out_features": module.out_features,
                    "qweight_sha256": tensor_sha256(qweight),
                }
            )
            offset += len(payload)
    return records


def _exact_realized_scale32_record(value: float, context: str) -> int:
    record = ceil_scale32_from_float(float(value))
    require(
        scale32_value(record) == float(value),
        f"runtime scale is not the exported Scale32 value: {context}",
    )
    return record


def _write_scale_artifacts(
    model: nn.Module,
    ranges: dict[str, fixed.CalibrationRange],
    operator_ranges: dict[str, fixed.ObservedRange],
    output_root: Path,
) -> dict[str, str]:
    table = fixed.derived_scale_table(ranges, operator_ranges, model)
    linears = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, fixed.W4A8Linear)
    }
    require(set(linears) == set(table["linears"]), "derived linear scale set differs")
    weight: dict[str, Any] = {}
    activation: dict[str, Any] = {}
    for name, module in sorted(linears.items()):
        if isinstance(module, MSEClipGridW4A8Linear):
            weight_records = module.weight_scale32_records.detach().cpu().tolist()
            require(
                module.weight_scale.detach().cpu().tolist()
                == [scale32_value(record) for record in weight_records],
                f"c01 runtime and exported weight scales differ: {name}",
            )
        else:
            weight_records = [
                ceil_scale32_from_float(float(value))
                for value in module.weight_scale.detach().cpu().tolist()
            ]
        for record in weight_records:
            unpack_scale32(record)
        exact_c01 = isinstance(module, MSEClipGridW4A8Linear)
        output_head_records = [
            (
                _exact_realized_scale32_record(float(value), f"{name}.output_head")
                if exact_c01
                else ceil_scale32_from_float(float(value))
            )
            for value in module.output_head_scales.detach().cpu().tolist()
        ]
        output_record = (
            None
            if module.output_scale is None
            else (
                _exact_realized_scale32_record(
                    float(module.output_scale),
                    f"{name}.output",
                )
                if exact_c01
                else ceil_scale32_from_float(float(module.output_scale))
            )
        )
        input_record = (
            _exact_realized_scale32_record(float(module.input_scale), f"{name}.input")
            if exact_c01
            else ceil_scale32_from_float(float(module.input_scale))
        )
        hardware_input_record = (
            _exact_realized_scale32_record(
                float(module.hardware_input_scale),
                f"{name}.hardware_input",
            )
            if exact_c01
            else ceil_scale32_from_float(float(module.hardware_input_scale))
        )
        for record in (input_record, hardware_input_record, *output_head_records):
            unpack_scale32(record)
        if output_record is not None:
            unpack_scale32(output_record)
        weight[name] = {
            "granularity": "one_positive_Scale32_per_output_channel_across_the_complete_projection_reduction",
            "records": weight_records,
        }
        activation[name] = {
            "hardware_input_scale32_record": hardware_input_record,
            "input_scale32_record": input_record,
            "output_head_scale32_records": output_head_records,
            "output_scale32_record": output_record,
        }
    exact_c01_model = bool(linears) and all(
        isinstance(module, MSEClipGridW4A8Linear) for module in linears.values()
    )
    operator = {}
    for name, entry in table["operators"].items():
        record = (
            _exact_realized_scale32_record(entry["scale"], f"operator.{name}")
            if exact_c01_model
            else ceil_scale32_from_float(entry["scale"])
        )
        operator[name] = {**entry, "scale32_record": record}
    for entry in operator.values():
        unpack_scale32(entry["scale32_record"])
    kv_cache = {
        f"layer-{index:02d}": {
            "key_payload": "signed_int8_post_rope_dynamic_scale32_per_token_per_kv_head",
            "key_producer_scale32": layer.self_attn.key_producer_scale32,
            "value_output_scale32": (
                _exact_realized_scale32_record(
                    float(layer.self_attn.v_proj.output_scale),
                    f"layer-{index:02d}.value_output",
                )
                if exact_c01_model
                else ceil_scale32_from_float(
                    float(layer.self_attn.v_proj.output_scale)
                )
            ),
        }
        for index, layer in enumerate(model.model.layers)
    }
    for layer in kv_cache.values():
        require(layer["key_producer_scale32"] is not None, "KV key Scale32 is missing")
        unpack_scale32(layer["key_producer_scale32"])
        unpack_scale32(layer["value_output_scale32"])
    paths = {
        "activation_scale32": output_root / "activation-scale32.json",
        "kv_cache_scale32": output_root / "kv-cache-scale32.json",
        "operator_scale32": output_root / "operator-scale32.json",
        "weight_scale32": output_root / "weight-scale32.json",
    }
    write_json(paths["weight_scale32"], weight)
    write_json(paths["activation_scale32"], activation)
    write_json(paths["operator_scale32"], operator)
    write_json(paths["kv_cache_scale32"], kv_cache)
    write_json(output_root / "derived-scales.json", table)
    return {name: sha256_file(path) for name, path in paths.items()}


def execute_candidate_attempt(
    contract: dict[str, Any],
    contract_sha256: str,
    spec: dict[str, Any],
    source_hashes: dict[str, str],
    candidate_id: str,
) -> Path:
    candidate_definition = candidate_spec(contract, candidate_id)
    require(
        candidate_id == "c01-mse-clip-grid",
        "only additive c01 execution is supported; c00 is terminally prohibited",
    )
    target = (
        ROOT
        / spec["artifact_root"]
        / candidate_id
        / contract["artifact_policy"]["candidate_attempt_name"]
    )
    require(
        not target.exists(),
        f"immutable candidate attempt already exists: {target.relative_to(ROOT)}",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".attempt-0001-", dir=target.parent))
    try:
        config = load_json(QUALITY_CONFIG_PATH)
        manifest = load_json(PROMPT_MANIFEST_PATH)
        validate_contract_support(contract, config)
        fixed.seed_everything(config)
        tokenizer = load_tokenizer(spec)
        inputs, input_observations = load_bounded_inputs(tokenizer, contract, manifest)
        input_observations["lm_eval"] = fixed.observe_lm_eval_datasets(manifest)
        write_json(temporary / "input-observations.json", input_observations)

        model = load_model(spec)
        ranges, operator_ranges = fixed.calibrate(model, inputs["calibration"])
        calibration = _calibration_observations(ranges, operator_ranges)
        write_json(temporary / "calibration-observations.json", calibration)
        baseline = _quality_measurement(
            model,
            tokenizer,
            inputs,
            manifest,
            config,
            temporary,
            "bf16",
        )

        replace_linears_for_candidate(
            model,
            ranges,
            candidate_definition,
            rope_diagnostic_mechanism=fixed.ACTIVE_ROPE_MECHANISM,
        )
        runtime_operator_ranges = replace_fixed_operators_for_candidate(
            model,
            operator_ranges,
            candidate_definition,
            rope_diagnostic_mechanism=fixed.ACTIVE_ROPE_MECHANISM,
        )
        enable_w4a8_kv_cache(model)
        packed_path = temporary / "packed-w4.bin"
        packed_records = _write_packed_w4(model, packed_path)
        write_json(temporary / "packed-w4-layout.json", packed_records)
        scale_hashes = _write_scale_artifacts(
            model,
            ranges,
            runtime_operator_ranges,
            temporary,
        )

        generation_results = generate_w4a8(
            model,
            tokenizer,
            contract["evaluation_contract"]["canonical_generation_prompts"],
            contract["shared_w4a8_contract"]["generation"],
        )
        write_json(temporary / "canonical-generation.json", generation_results)
        generation_score = evaluate_generation(
            tokenizer,
            generation_results,
            contract["evaluation_contract"]["canonical_generation_prompts"],
            contract["shared_w4a8_contract"]["generation"],
            contract["evaluation_contract"]["generation_gate"],
        )

        candidate_quality = _quality_measurement(
            model,
            tokenizer,
            inputs,
            manifest,
            config,
            temporary,
            "w4a8",
        )
        quality_score = evaluate_quality(
            baseline,
            candidate_quality,
            quality_thresholds_from_contract(contract),
            contract["evaluation_contract"]["bounded_quality_sets"]["lm_eval"]["tasks"],
        )
        result = {
            "candidate_id": candidate_id,
            "classification": "PASS" if generation_score["passed"] and quality_score["passed"] else "NEGATIVE",
            "contract_sha256": contract_sha256,
            "created_at_utc": utc_now(),
            "generation": generation_score,
            "model": {
                "alias": spec["alias"],
                "model_id": spec["model_id"],
                "model_identity_sha256": canonical_sha256(spec["contract_identity"]),
            },
            "quality": quality_score,
            "schema_version": 1,
            "source_hashes": source_hashes,
            "status": "PASS" if generation_score["passed"] and quality_score["passed"] else "FAIL",
        }
        write_json(temporary / "results.json", result)
        configuration = {
            "candidate": candidate_definition,
            "evaluation_contract": contract["evaluation_contract"],
            "model": result["model"],
            "shared_w4a8_contract": contract["shared_w4a8_contract"],
            "source_hashes": source_hashes,
        }
        write_json(temporary / "configuration.json", configuration)
        raw_output = {
            "calibration_observations_sha256": sha256_file(temporary / "calibration-observations.json"),
            "canonical_generation_sha256": sha256_file(temporary / "canonical-generation.json"),
            "derived_scales_sha256": sha256_file(temporary / "derived-scales.json"),
            "input_observations_sha256": sha256_file(temporary / "input-observations.json"),
            "lm_eval_bf16_raw_sha256": sha256_file(temporary / "lm-eval-bf16-raw.json"),
            "lm_eval_w4a8_raw_sha256": sha256_file(temporary / "lm-eval-w4a8-raw.json"),
            "packed_w4_layout_sha256": sha256_file(temporary / "packed-w4-layout.json"),
            "perplexity_bf16_sha256": sha256_file(temporary / "perplexity-bf16.json"),
            "perplexity_w4a8_sha256": sha256_file(temporary / "perplexity-w4a8.json"),
            "quality_bf16_sha256": sha256_file(temporary / "quality-bf16.json"),
            "quality_w4a8_sha256": sha256_file(temporary / "quality-w4a8.json"),
            "results_sha256": sha256_file(temporary / "results.json"),
        }
        write_json(temporary / "raw-output.json", raw_output)
        manifest_record = {
            "activation_scale32_sha256": scale_hashes["activation_scale32"],
            "calibration_input_sha256": canonical_sha256(input_observations["c4_calibration"]),
            "calibration_observations_sha256": sha256_file(temporary / "calibration-observations.json"),
            "candidate_id": candidate_id,
            "configuration_sha256": sha256_file(temporary / "configuration.json"),
            "contract_sha256": contract_sha256,
            "created_at_utc": utc_now(),
            "input_observations_sha256": sha256_file(temporary / "input-observations.json"),
            "kv_cache_scale32_sha256": scale_hashes["kv_cache_scale32"],
            "model_alias": spec["alias"],
            "model_identity_sha256": result["model"]["model_identity_sha256"],
            "operator_scale32_sha256": scale_hashes["operator_scale32"],
            "packed_w4_sha256": sha256_file(packed_path),
            "raw_output_sha256": sha256_file(temporary / "raw-output.json"),
            "schema_version": 1,
            "source_hashes": source_hashes,
            "status": result["status"],
            "weight_scale32_sha256": scale_hashes["weight_scale32"],
        }
        write_json(temporary / "manifest.json", manifest_record)
        temporary.rename(target)
        del model
        gc.collect()
        return target
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_candidate_attempt(
    contract: dict[str, Any],
    contract_sha256: str,
    spec: dict[str, Any],
    source_hashes: dict[str, str],
    candidate_id: str,
) -> dict[str, Any]:
    candidate_definition = candidate_spec(contract, candidate_id)
    require(candidate_id in {"c00-rtn-absmax", "c01-mse-clip-grid"}, "unsupported candidate verification")
    target = (
        ROOT
        / spec["artifact_root"]
        / candidate_id
        / contract["artifact_policy"]["candidate_attempt_name"]
    )
    require(target.is_dir(), f"candidate attempt is missing: {target.relative_to(ROOT)}")
    config = load_json(QUALITY_CONFIG_PATH)
    validate_contract_support(contract, config)
    manifest = require_canonical_json(target / "manifest.json")
    required_manifest_fields = set(contract["artifact_policy"]["manifest_required_fields"])
    require(required_manifest_fields <= set(manifest), "candidate manifest fields are incomplete")
    require(manifest["contract_sha256"] == contract_sha256, "candidate contract binding differs")
    require(manifest["source_hashes"] == source_hashes, "candidate source hashes differ")
    require(manifest["model_alias"] == spec["alias"], "candidate model alias differs")
    require(manifest["candidate_id"] == candidate_id, "candidate identity differs")
    model_identity_sha256 = canonical_sha256(spec["contract_identity"])
    require(
        manifest["model_identity_sha256"] == model_identity_sha256,
        "c00 model identity hash differs",
    )
    files = {
        "activation_scale32_sha256": "activation-scale32.json",
        "calibration_observations_sha256": "calibration-observations.json",
        "configuration_sha256": "configuration.json",
        "input_observations_sha256": "input-observations.json",
        "kv_cache_scale32_sha256": "kv-cache-scale32.json",
        "operator_scale32_sha256": "operator-scale32.json",
        "packed_w4_sha256": "packed-w4.bin",
        "raw_output_sha256": "raw-output.json",
        "weight_scale32_sha256": "weight-scale32.json",
    }
    for field, relative in files.items():
        path = target / relative
        require(path.is_file(), f"candidate artifact is missing: {relative}")
        require(
            sha256_file(path) == manifest[field],
            f"candidate artifact hash differs: {relative}",
        )

    weight_scales = require_canonical_json(target / "weight-scale32.json")
    activation_scales = require_canonical_json(target / "activation-scale32.json")
    operator_scales = require_canonical_json(target / "operator-scale32.json")
    kv_scales = require_canonical_json(target / "kv-cache-scale32.json")
    derived_scales = require_canonical_json(target / "derived-scales.json")
    require(set(weight_scales) == set(activation_scales), "linear Scale32 sets differ")
    for name, entry in weight_scales.items():
        require(
            set(entry) == {"granularity", "records"},
            f"weight Scale32 fields differ: {name}",
        )
        require(entry["records"], f"weight Scale32 records are empty: {name}")
        for record in entry["records"]:
            unpack_scale32(record)
    for name, entry in activation_scales.items():
        require(
            set(entry)
            == {
                "hardware_input_scale32_record",
                "input_scale32_record",
                "output_head_scale32_records",
                "output_scale32_record",
            },
            f"activation Scale32 fields differ: {name}",
        )
        unpack_scale32(entry["hardware_input_scale32_record"])
        unpack_scale32(entry["input_scale32_record"])
        for record in entry["output_head_scale32_records"]:
            unpack_scale32(record)
        if entry["output_scale32_record"] is not None:
            unpack_scale32(entry["output_scale32_record"])
    for name, entry in operator_scales.items():
        unpack_scale32(entry["scale32_record"])
        require(float(entry["scale"]) > 0.0, f"operator scale is not positive: {name}")
    require(len(kv_scales) == 24, "KV Scale32 layer count differs")
    for name, entry in kv_scales.items():
        require(
            entry["key_payload"]
            == "signed_int8_post_rope_dynamic_scale32_per_token_per_kv_head",
            f"KV key payload differs: {name}",
        )
        unpack_scale32(entry["key_producer_scale32"])
        unpack_scale32(entry["value_output_scale32"])
    if candidate_id == "c01-mse-clip-grid":
        require(
            set(weight_scales) == set(derived_scales["linears"]),
            "c01 derived linear scale set differs",
        )
        for name, line in derived_scales["linears"].items():
            require(
                [scale32_value(record) for record in weight_scales[name]["records"]]
                == line["weight_scale"],
                f"c01 derived weight scales differ: {name}",
            )
            activation = activation_scales[name]
            require(
                scale32_value(activation["input_scale32_record"])
                == line["input_scale"],
                f"c01 derived input scale differs: {name}",
            )
            require(
                scale32_value(activation["hardware_input_scale32_record"])
                == line["hardware_input_scale"],
                f"c01 derived hardware-input scale differs: {name}",
            )
            require(
                [
                    scale32_value(record)
                    for record in activation["output_head_scale32_records"]
                ]
                == line["output_head_scales"],
                f"c01 derived output-head scales differ: {name}",
            )
            require(
                (
                    None
                    if activation["output_scale32_record"] is None
                    else scale32_value(activation["output_scale32_record"])
                )
                == line["output_scale"],
                f"c01 derived output scale differs: {name}",
            )
        require(
            set(operator_scales) == set(derived_scales["operators"]),
            "c01 derived operator scale set differs",
        )
        for name, entry in operator_scales.items():
            require(
                scale32_value(entry["scale32_record"])
                == derived_scales["operators"][name]["scale"],
                f"c01 derived operator scale differs: {name}",
            )
        for index in range(24):
            layer_name = f"layer-{index:02d}"
            attention = derived_scales["attention"][
                f"model.layers.{index}.self_attn"
            ]
            require(
                kv_scales[layer_name]["key_producer_scale32"]
                == attention["dynamic_rope_head_scale"]["key_producer_scale32"],
                f"c01 KV key producer Scale32 differs: {layer_name}",
            )
            require(
                scale32_value(kv_scales[layer_name]["value_output_scale32"])
                == attention["attention_value_output_scale"],
                f"c01 KV value scale differs: {layer_name}",
            )

    packed_layout = require_canonical_value(target / "packed-w4-layout.json")
    require(isinstance(packed_layout, list) and packed_layout, "packed W4 layout is empty")
    offset = 0
    for record in packed_layout:
        require(record["offset"] == offset, "packed W4 offsets are not contiguous")
        require(record["in_features"] % 2 == 0, "packed W4 input width is odd")
        expected_bytes = record["out_features"] * record["in_features"] // 2
        require(record["bytes"] == expected_bytes, "packed W4 byte count differs")
        require(
            len(weight_scales[record["name"]]["records"]) == record["out_features"],
            "packed W4 row and weight-scale counts differ",
        )
        offset += record["bytes"]
    require((target / "packed-w4.bin").stat().st_size == offset, "packed W4 size differs")

    raw_output = require_canonical_json(target / "raw-output.json")
    raw_files = {
        "calibration_observations_sha256": "calibration-observations.json",
        "canonical_generation_sha256": "canonical-generation.json",
        "derived_scales_sha256": "derived-scales.json",
        "input_observations_sha256": "input-observations.json",
        "lm_eval_bf16_raw_sha256": "lm-eval-bf16-raw.json",
        "lm_eval_w4a8_raw_sha256": "lm-eval-w4a8-raw.json",
        "packed_w4_layout_sha256": "packed-w4-layout.json",
        "perplexity_bf16_sha256": "perplexity-bf16.json",
        "perplexity_w4a8_sha256": "perplexity-w4a8.json",
        "quality_bf16_sha256": "quality-bf16.json",
        "quality_w4a8_sha256": "quality-w4a8.json",
        "results_sha256": "results.json",
    }
    require(set(raw_output) == set(raw_files), "raw-output hash fields differ")
    for field, relative in raw_files.items():
        path = target / relative
        require(path.is_file(), f"raw candidate artifact is missing: {relative}")
        require(
            sha256_file(path) == raw_output[field],
            f"raw candidate hash differs: {relative}",
        )

    input_observations = require_canonical_json(target / "input-observations.json")
    require(
        manifest["calibration_input_sha256"]
        == canonical_sha256(input_observations["c4_calibration"]),
        "calibration input binding differs",
    )
    prompt_results = require_canonical_value(target / "canonical-generation.json")
    require(isinstance(prompt_results, list), "canonical generation must be a list")
    tokenizer = load_tokenizer(spec)
    generation_score = evaluate_generation(
        tokenizer,
        prompt_results,
        contract["evaluation_contract"]["canonical_generation_prompts"],
        contract["shared_w4a8_contract"]["generation"],
        contract["evaluation_contract"]["generation_gate"],
    )
    baseline = require_canonical_json(target / "quality-bf16.json")
    candidate_quality = require_canonical_json(target / "quality-w4a8.json")
    require(
        baseline["perplexity"] == require_canonical_json(target / "perplexity-bf16.json"),
        "BF16 perplexity summary differs",
    )
    require(
        candidate_quality["perplexity"]
        == require_canonical_json(target / "perplexity-w4a8.json"),
        "W4A8 perplexity summary differs",
    )
    quality_score = evaluate_quality(
        baseline,
        candidate_quality,
        quality_thresholds_from_contract(contract),
        contract["evaluation_contract"]["bounded_quality_sets"]["lm_eval"]["tasks"],
    )

    results = require_canonical_json(target / "results.json")
    require(results["contract_sha256"] == contract_sha256, "candidate result contract differs")
    require(results["source_hashes"] == source_hashes, "candidate result source binding differs")
    require(results["candidate_id"] == candidate_id, "candidate result identity differs")
    require(results["generation"] == generation_score, "generation validators do not reproduce")
    require(results["quality"] == quality_score, "quality gates do not reproduce")
    expected_pass = generation_score["passed"] and quality_score["passed"]
    require(results["status"] == ("PASS" if expected_pass else "FAIL"), "c00 status differs")
    require(
        results["classification"] == ("PASS" if expected_pass else "NEGATIVE"),
        "c00 classification differs",
    )
    require(manifest["status"] == results["status"], "manifest and result status differ")
    configuration = require_canonical_json(target / "configuration.json")
    require(
        configuration
        == {
            "candidate": candidate_definition,
            "evaluation_contract": contract["evaluation_contract"],
            "model": results["model"],
            "shared_w4a8_contract": contract["shared_w4a8_contract"],
            "source_hashes": source_hashes,
        },
        "candidate configuration binding differs",
    )
    return {
        "alias": spec["alias"],
        "classification": results["classification"],
        "generation_passed": results["generation"]["passed"],
        "path": target.relative_to(ROOT).as_posix(),
        "quality_passed": results["quality"]["passed"],
        "status": results["status"],
    }


def execute_c00_attempt(
    contract: dict[str, Any],
    contract_sha256: str,
    spec: dict[str, Any],
    source_hashes: dict[str, str],
) -> Path:
    del contract, contract_sha256, spec, source_hashes
    raise RuntimeError(
        "c00 is sealed terminal/unselectable and cannot create or replay an attempt"
    )


def verify_c00_attempt(
    contract: dict[str, Any],
    contract_sha256: str,
    spec: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return verify_candidate_attempt(
        contract,
        contract_sha256,
        spec,
        source_hashes,
        "c00-rtn-absmax",
    )


def self_test() -> None:
    cases = {
        "strip_ascii_whitespace_then_exactly_19": (" 19\n", "20"),
        "readability_only": ("Readable sentence.", "1234"),
        "parse_json_then_exact_object_status_ok_count_3": (
            '{"count":3,"status":"ok"}',
            '{"count":4,"status":"ok"}',
        ),
        "exactly_three_nonempty_lines_each_containing_a_unicode_letter": (
            "red\ngreen\nblue",
            "red\ngreen",
        ),
        "unicode_casefold_strip_terminal_punctuation_in_buenos_dias_set": (
            "Buen día.",
            "Buenas tardes",
        ),
        "exactly_two_list_items_with_unicode_letters": (
            "1. First\n2. Second",
            "First\nSecond",
        ),
    }
    require(set(cases) == SUPPORTED_VALIDATORS, "self-test validator coverage differs")
    for name, (passing, failing) in cases.items():
        require(prompt_validator(name, passing)["passed"], f"validator did not pass: {name}")
        require(not prompt_validator(name, failing)["passed"], f"validator did not fail: {name}")

    c01_candidate = {
        "candidate_id": "c01-mse-clip-grid",
        "weight_generation": {
            "clip_ratio_denominator": 1000,
            "clip_ratio_numerators": [1000, 990, 975, 950, 925, 900],
        },
    }
    c01_weight = torch.tensor(
        [
            [0.0] * 21,
            [10.0] + [1.0] * 20,
        ],
        dtype=torch.bfloat16,
    )
    c01_qweight, c01_scale, c01_numerator, c01_records = (
        mse_clip_grid_quantization_scale32(c01_weight, c01_candidate)
    )
    require(torch.equal(c01_qweight[0], torch.zeros(21, dtype=torch.int8)), "c01 zero row codes differ")
    require(
        float(c01_scale[0]) == scale32_value(int(c01_records[0])),
        "c01 zero row Scale32 differs",
    )
    require(int(c01_numerator[0]) == 1000, "c01 equal-error tie break differs")
    require(int(c01_numerator[1]) < 1000, "c01 clipping grid did not improve the outlier row")
    require(torch.all(c01_scale > 0), "c01 selected a non-positive scale")

    divergence_weight = torch.tensor(
        [[0.29296875, -0.1318359375]],
        dtype=torch.bfloat16,
    )
    divergence_absmax = divergence_weight.abs().amax(dim=1)
    divergence_qweight, divergence_scale, _records, _error = (
        _clip_grid_row_candidate(
            divergence_weight,
            divergence_absmax,
            900,
            1000,
        )
    )
    raw_scale = float(divergence_absmax[0]) * 900.0 / 1000.0 / 7.0
    raw_code = int(torch.round(divergence_weight[0, 1].to(torch.float64) / raw_scale))
    require(raw_code == -4, "c01 raw-scale divergence oracle changed")
    require(
        int(divergence_qweight[0, 1]) == -3,
        "c01 Scale32 divergence repair changed",
    )
    require(
        float(divergence_scale[0]) == 0.03766822814941406,
        "c01 Scale32 divergence scale changed",
    )

    baseline = {
        "lm_eval": {
            "average_normalized_accuracy": 0.5,
            "tasks": {"task": 0.5},
        },
        "perplexity": {
            "c4_en_512": {"perplexity": 10.0},
            "wikitext2": {"perplexity": 20.0},
        },
    }
    candidate = {
        "lm_eval": {
            "average_normalized_accuracy": 0.475,
            "tasks": {"task": 0.43},
        },
        "perplexity": {
            "c4_en_512": {"perplexity": 11.0},
            "wikitext2": {"perplexity": 22.0},
        },
    }
    thresholds = {
        "applies_independently_to_both_models": True,
        "c4_en_512_perplexity_ratio_max": 1.1,
        "lm_eval_average_normalized_accuracy_drop_percentage_points_max": 3.0,
        "lm_eval_individual_task_drop_percentage_points_max": 8.0,
        "mission_local_not_project_final_emnlp_gate": True,
        "wikitext2_perplexity_ratio_max": 1.1,
    }
    require(
        evaluate_quality(baseline, candidate, thresholds, ["task"])["passed"],
        "quality boundary did not pass",
    )
    candidate["perplexity"]["c4_en_512"]["perplexity"] = 11.0001
    require(
        not evaluate_quality(baseline, candidate, thresholds, ["task"])["passed"],
        "quality excess did not fail",
    )


if __name__ == "__main__":
    self_test()
    print("ACE2_W4A8_EVALUATOR_SELF_TEST status=pass")
