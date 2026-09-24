#!/usr/bin/env python3
"""Non-scoring c02 source-oracle reconstruction and fixed-input stage trace.

This diagnostic is deliberately outside every candidate/official attempt namespace.
It reconstructs c01 W4/Scale32 artifacts from the frozen BF16 source, then runs one
sealed canonical prompt as a cache-free and cache-split microdiagnostic.  It never
scores a benchmark, generates a continuation, or invokes an execute-command path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import ace2_full_model_fixed_point as fixed
import run_w4a8_full_model_software_contract as frozen
import w4a8_full_model_evaluator as evaluator
from ace2_quality_contracts import ceil_scale32_from_ratio


DEFAULT_OUTPUT = (
    ROOT
    / "evidence/diagnostics/w4a8-c02-source-oracle-stage-trace-v2"
)
SEALED_ROOTS = {
    "base": ROOT
    / "build/w4a8-full-model-software-contract-v1/base/c01-mse-clip-grid/attempt-0001",
    "checkpoint-176": ROOT
    / "build/w4a8-full-model-software-contract-v1/checkpoint-176/c01-mse-clip-grid/attempt-0001",
}
MODEL_ORDER = ("base", "checkpoint-176")
TENSOR_BUNDLE_MAGIC = b"ACE2-C02-TENSORS-V1\n"
CLIP_NUMERATORS = (1000, 990, 975, 950, 925, 900)
CLIP_DENOMINATOR = 1000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    payload_tensor = (
        tensor.view(torch.uint16) if tensor.dtype == torch.bfloat16 else tensor
    )
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack(">I", tensor.ndim))
    for dimension in tensor.shape:
        digest.update(struct.pack(">Q", int(dimension)))
    digest.update(payload_tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def last_row(value: torch.Tensor) -> torch.Tensor:
    require(value.ndim >= 1, "captured tensor must have at least one dimension")
    if value.ndim == 1:
        return value.detach().cpu().contiguous()
    return value.detach().reshape(-1, value.shape[-1])[-1].cpu().contiguous()


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"no tensor in captured value: {type(value)!r}")


def _scale32_values(records: torch.Tensor) -> torch.Tensor:
    integer_records = records.detach().to(torch.int64)
    significand = torch.bitwise_and(integer_records, 0xFFFF)
    exponent_u8 = torch.bitwise_and(
        torch.bitwise_right_shift(integer_records, 16), 0xFF
    )
    exponent = torch.where(exponent_u8 >= 128, exponent_u8 - 256, exponent_u8)
    return torch.ldexp(significand.to(torch.float64), exponent - 15)


def oracle_clip_scale32_records(
    row_absmax_bf16: torch.Tensor,
    numerator: int,
    denominator: int,
) -> torch.Tensor:
    """Independent c02 realization of the frozen row clip-to-Scale32 rule."""
    require(row_absmax_bf16.dtype == torch.bfloat16, "oracle absmax must be BF16")
    unique_absmax, inverse = torch.unique(
        row_absmax_bf16.detach().cpu(), sorted=True, return_inverse=True
    )
    records: list[int] = []
    for value in unique_absmax.to(torch.float64).tolist():
        absmax_numerator, absmax_denominator = float(value).as_integer_ratio()
        records.append(
            ceil_scale32_from_ratio(
                absmax_numerator * numerator,
                absmax_denominator * denominator * 7,
            )
        )
    return torch.tensor(records, dtype=torch.int64)[inverse].to(
        row_absmax_bf16.device
    )


def oracle_quantize_bf16_with_scale32(
    weight_bf16: torch.Tensor,
    records: torch.Tensor,
) -> torch.Tensor:
    """Independent exact BF16/Scale32 ties-to-even signed-W4 division."""
    require(weight_bf16.dtype == torch.bfloat16, "oracle weight must be BF16")
    bits = torch.bitwise_and(weight_bf16.view(torch.int16).to(torch.int64), 0xFFFF)
    exponent_bits = torch.bitwise_and(torch.bitwise_right_shift(bits, 7), 0xFF)
    require(not torch.any(exponent_bits == 0xFF).item(), "oracle source is non-finite")
    fraction = torch.bitwise_and(bits, 0x7F)
    mantissa = torch.where(exponent_bits == 0, fraction, fraction + 0x80)
    weight_exponent = torch.where(exponent_bits == 0, -133, exponent_bits - 134)

    records_i64 = records.to(device=weight_bf16.device, dtype=torch.int64)
    scale_significand = torch.bitwise_and(records_i64, 0xFFFF)
    scale_exponent_u8 = torch.bitwise_and(
        torch.bitwise_right_shift(records_i64, 16), 0xFF
    )
    scale_exponent = torch.where(
        scale_exponent_u8 >= 128,
        scale_exponent_u8 - 256,
        scale_exponent_u8,
    )
    shift = weight_exponent - scale_exponent[:, None] + 15
    nonzero = mantissa != 0
    require(
        not torch.any(nonzero & (shift > 20)).item(),
        "oracle exact W4 division exceeded bounded shift",
    )
    active = nonzero & (shift >= 6)
    exact_numerator = torch.bitwise_left_shift(mantissa, shift.clamp(0, 20))
    exact_denominator = scale_significand[:, None]
    quotient = torch.div(exact_numerator, exact_denominator, rounding_mode="floor")
    remainder = exact_numerator - quotient * exact_denominator
    doubled = remainder * 2
    increment = (doubled > exact_denominator) | (
        (doubled == exact_denominator) & ((quotient & 1) == 1)
    )
    rounded = torch.where(active, quotient + increment.to(torch.int64), 0)
    signed = torch.where((bits & 0x8000) != 0, -rounded, rounded)
    return signed.clamp(-8, 7).to(torch.int8)


def oracle_mse_clip_grid_quantization_scale32(
    source_weight: torch.Tensor,
    candidate: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct every c01 row without calling the frozen c01 quantizer."""
    generation = candidate["weight_generation"]
    require(candidate["candidate_id"] == "c01-mse-clip-grid", "candidate differs")
    require(
        tuple(generation["clip_ratio_numerators"]) == CLIP_NUMERATORS,
        "clip grid differs",
    )
    require(
        generation["clip_ratio_denominator"] == CLIP_DENOMINATOR,
        "clip denominator differs",
    )
    weight_bf16 = source_weight.detach().to(torch.bfloat16)
    row_absmax = weight_bf16.abs().amax(dim=1)
    best_error = torch.full(
        (weight_bf16.shape[0],),
        float("inf"),
        dtype=torch.float64,
        device=weight_bf16.device,
    )
    best_records = torch.zeros_like(best_error, dtype=torch.int64)
    best_qweight = torch.zeros_like(weight_bf16, dtype=torch.int8)
    best_numerator = torch.full_like(best_records, CLIP_NUMERATORS[0])
    weight_f64 = weight_bf16.to(torch.float64)
    for numerator in CLIP_NUMERATORS:
        records = oracle_clip_scale32_records(
            row_absmax, numerator, CLIP_DENOMINATOR
        )
        scale = _scale32_values(records)
        qweight = oracle_quantize_bf16_with_scale32(weight_bf16, records)
        reconstruction = qweight.to(torch.float64) * scale[:, None]
        error = torch.sum(
            (weight_f64 - reconstruction) ** 2,
            dim=1,
            dtype=torch.float64,
        )
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_records = torch.where(better, records, best_records)
        best_qweight = torch.where(better[:, None], qweight, best_qweight)
        best_numerator = torch.where(
            better,
            torch.full_like(best_numerator, numerator),
            best_numerator,
        )
    return (
        best_qweight,
        _scale32_values(best_records),
        best_numerator,
        best_records,
    )


def write_tensor_bundle(
    path: Path,
    tensors: dict[str, torch.Tensor],
    *,
    public_path: str | None = None,
) -> dict[str, Any]:
    """Write a deterministic, timestamp-free exact tensor container."""
    records: dict[str, Any] = {}
    with path.open("wb") as handle:
        handle.write(TENSOR_BUNDLE_MAGIC)
        handle.write(struct.pack(">I", len(tensors)))
        for name in sorted(tensors):
            tensor = tensors[name].detach().cpu().contiguous()
            name_raw = name.encode("utf-8")
            dtype_raw = str(tensor.dtype).encode("ascii")
            payload_tensor = (
                tensor.view(torch.uint16)
                if tensor.dtype == torch.bfloat16
                else tensor
            )
            payload = payload_tensor.numpy().tobytes(order="C")
            handle.write(struct.pack(">H", len(name_raw)))
            handle.write(name_raw)
            handle.write(struct.pack(">B", len(dtype_raw)))
            handle.write(dtype_raw)
            handle.write(struct.pack(">B", tensor.ndim))
            for dimension in tensor.shape:
                handle.write(struct.pack(">Q", int(dimension)))
            handle.write(struct.pack(">Q", len(payload)))
            handle.write(payload)
            records[name] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": tensor_sha256(tensor),
            }
    if public_path is None:
        try:
            public_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            public_path = str(path)
    return {
        "bytes": path.stat().st_size,
        "path": public_path,
        "sha256": sha256_file(path),
        "tensor_count": len(records),
        "tensors": records,
    }


def numerical_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    raw_int8: torch.Tensor | None = None,
) -> dict[str, Any]:
    ref = reference.detach().to(torch.float64).reshape(-1)
    observed = candidate.detach().to(torch.float64).reshape(-1)
    require(ref.shape == observed.shape, "comparison geometry differs")
    ref_energy = torch.mean(ref * ref)
    error = observed - ref
    error_rms = torch.sqrt(torch.mean(error * error))
    relative_rmse = (
        float(error_rms / torch.sqrt(ref_energy))
        if float(ref_energy) > 0.0
        else (0.0 if float(error_rms) == 0.0 else math.inf)
    )
    ref_norm = torch.linalg.vector_norm(ref)
    observed_norm = torch.linalg.vector_norm(observed)
    cosine = (
        float(torch.dot(ref, observed) / (ref_norm * observed_norm))
        if float(ref_norm) > 0.0 and float(observed_norm) > 0.0
        else (1.0 if torch.equal(ref, observed) else 0.0)
    )
    saturation_fraction = None
    if raw_int8 is not None:
        raw = raw_int8.detach().to(torch.int16).reshape(-1)
        saturation_fraction = float(((raw <= -128) | (raw >= 127)).double().mean())
    severity = "TRACKING"
    if (
        relative_rmse >= 1.0
        or cosine <= 0.5
        or (saturation_fraction is not None and saturation_fraction >= 0.25)
    ):
        severity = "SEVERE"
    elif (
        relative_rmse >= 0.25
        or cosine <= 0.9
        or (saturation_fraction is not None and saturation_fraction >= 0.05)
    ):
        severity = "MATERIAL"
    return {
        "candidate_absmax": float(observed.abs().max()),
        "candidate_zero_fraction": float((observed == 0).double().mean()),
        "cosine_similarity": cosine,
        "maximum_absolute_error": float(error.abs().max()),
        "reference_absmax": float(ref.abs().max()),
        "relative_rmse": relative_rmse,
        "saturation_fraction": saturation_fraction,
        "severity": severity,
    }


def stable_top(values: torch.Tensor, count: int = 8) -> list[dict[str, Any]]:
    flat = values.detach().to(torch.float64).reshape(-1)
    ids = torch.argsort(flat, descending=True, stable=True)[:count].tolist()
    return [{"score": float(flat[index]), "token_id": int(index)} for index in ids]


def rank_of(values: torch.Tensor, token_id: int) -> int:
    flat = values.detach().to(torch.float64).reshape(-1)
    selected = flat[token_id]
    ids = torch.arange(flat.numel(), dtype=torch.int64)
    return int(((flat > selected) | ((flat == selected) & (ids < token_id))).sum()) + 1


def calibration_ranges(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = load_json(root / "calibration-observations.json")
    linears = {
        name: fixed.CalibrationRange(
            input_absmax=float(entry["input_absmax"]),
            output_absmax=float(entry["output_absmax"]),
            output_head_absmax=entry["output_head_absmax"],
        )
        for name, entry in observed["linears"].items()
    }
    operators = {
        name: fixed.ObservedRange(absmax=float(entry["absmax"]))
        for name, entry in observed["operators"].items()
    }
    return linears, operators


def sealed_fixed_input() -> tuple[torch.Tensor, dict[str, Any]]:
    records = {}
    token_lists = []
    for alias, root in SEALED_ROOTS.items():
        generation = load_json(root / "canonical-generation.json")
        first = generation[0]
        require(first["prompt_id"] == "g00-arithmetic-short", "fixed prompt differs")
        token_lists.append(first["input_token_ids"])
        records[alias] = {
            "first_generated_token_id": first["generated_token_ids"][0],
            "first_integer_logit_sha256": first[
                "per_step_integer_logit_tensor_sha256"
            ][0],
            "prompt_id": first["prompt_id"],
        }
    require(token_lists[0] == token_lists[1], "model fixed-input token IDs differ")
    ids = torch.tensor([token_lists[0]], dtype=torch.long)
    return ids, {
        "input_token_count": ids.shape[1],
        "input_token_ids": token_lists[0],
        "input_token_ids_sha256": canonical_sha256(token_lists[0]),
        "sealed_first_step": records,
    }


def register_bf16_hooks(
    model: nn.Module,
    captures: dict[str, torch.Tensor],
) -> list[Any]:
    hooks: list[Any] = []

    def capture(key: str, value: Any) -> None:
        captures[key] = last_row(first_tensor(value))

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(
                module.register_forward_pre_hook(
                    lambda _m, inputs, n=name: capture(f"bf16.{n}.input", inputs[0])
                )
            )
            hooks.append(
                module.register_forward_hook(
                    lambda _m, _inputs, output, n=name: capture(
                        f"bf16.{n}.output", output
                    )
                )
            )

    hooks.append(
        model.model.embed_tokens.register_forward_hook(
            lambda _m, _inputs, output: capture("bf16.model.embed_tokens.output", output)
        )
    )
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        for suffix, module in (
            ("input_layernorm", layer.input_layernorm),
            ("post_attention_layernorm", layer.post_attention_layernorm),
        ):
            name = f"{prefix}.{suffix}"
            hooks.append(
                module.register_forward_pre_hook(
                    lambda _m, inputs, n=name: capture(f"bf16.{n}.input", inputs[0])
                )
            )
            hooks.append(
                module.register_forward_hook(
                    lambda _m, _inputs, output, n=name: capture(
                        f"bf16.{n}.output", output
                    )
                )
            )
        hooks.append(
            layer.self_attn.register_forward_hook(
                lambda _m, _inputs, output, n=prefix: capture(
                    f"bf16.{n}.self_attn.output", output
                )
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                lambda _m, _inputs, output, n=prefix: capture(
                    f"bf16.{n}.mlp.output", output
                )
            )
        )
        hooks.append(
            layer.register_forward_hook(
                lambda _m, _inputs, output, n=prefix: capture(
                    f"bf16.{n}.output", output
                )
            )
        )
    hooks.append(
        model.model.norm.register_forward_pre_hook(
            lambda _m, inputs: capture("bf16.model.norm.input", inputs[0])
        )
    )
    hooks.append(
        model.model.norm.register_forward_hook(
            lambda _m, _inputs, output: capture("bf16.model.norm.output", output)
        )
    )
    return hooks


def extract_cache_tensors(
    model: nn.Module,
    past_key_values: Any,
    captures: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, Any]:
    summaries = {}
    for index in (0, len(model.model.layers) - 1):
        cache_layer = past_key_values.layers[index]
        record = {}
        for field in ("keys", "values"):
            value = getattr(cache_layer, field, None)
            if isinstance(value, torch.Tensor):
                key = f"{prefix}.layer-{index:02d}.{field}"
                captures[key] = value.detach().cpu().contiguous()
                record[field] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": tensor_sha256(value),
                }
        attention = model.model.layers[index].self_attn
        key_scales = getattr(attention, "ace2_key_scale32_cache", None)
        if isinstance(key_scales, torch.Tensor):
            key = f"{prefix}.layer-{index:02d}.key_scale32"
            captures[key] = key_scales.detach().cpu().contiguous()
            record["key_scale32"] = {
                "dtype": str(key_scales.dtype),
                "shape": list(key_scales.shape),
                "sha256": tensor_sha256(key_scales),
            }
        summaries[f"layer-{index:02d}"] = record
    return summaries


def run_bf16_trace(
    model: nn.Module,
    input_ids: torch.Tensor,
    captures: dict[str, torch.Tensor],
) -> dict[str, Any]:
    hooks = register_bf16_hooks(model, captures)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
        logits = output.logits[0, -1].detach().cpu().contiguous()
        captures["bf16.ranking.logits"] = logits
    finally:
        for hook in hooks:
            hook.remove()

    prefix = input_ids[:, :-1]
    final_token = input_ids[:, -1:]
    with torch.inference_mode():
        prefill = model(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            use_cache=True,
        )
        require(prefill.past_key_values is not None, "BF16 prefill cache is missing")
        split = model(
            input_ids=final_token,
            attention_mask=torch.ones_like(input_ids),
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
    split_logits = split.logits[0, -1].detach().cpu().contiguous()
    captures["bf16.cache_split.logits"] = split_logits
    cache_summary = extract_cache_tensors(
        model, split.past_key_values, captures, "bf16.cache"
    )
    return {
        "cache_full_prefix_max_abs_difference": float(
            (split_logits.float() - logits.float()).abs().max()
        ),
        "cache_full_prefix_selected_token_equal": int(torch.argmax(split_logits))
        == int(torch.argmax(logits)),
        "cache_layers": cache_summary,
        "full_prefix_selected_token_id": int(torch.argmax(logits)),
        "full_prefix_top": stable_top(logits),
        "split_selected_token_id": int(torch.argmax(split_logits)),
    }


def reconstruct_model_from_source_oracle(
    model: nn.Module,
    sealed_root: Path,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ranges, operator_ranges = calibration_ranges(sealed_root)
    candidate = evaluator.candidate_spec(contract, "c01-mse-clip-grid")
    original_quantizer = evaluator.mse_clip_grid_quantization_scale32
    evaluator.mse_clip_grid_quantization_scale32 = (
        oracle_mse_clip_grid_quantization_scale32
    )
    try:
        evaluator.replace_linears_for_candidate(
            model,
            ranges,
            candidate,
            rope_diagnostic_mechanism=fixed.ACTIVE_ROPE_MECHANISM,
        )
    finally:
        evaluator.mse_clip_grid_quantization_scale32 = original_quantizer
    runtime_operator_ranges = evaluator.replace_fixed_operators_for_candidate(
        model,
        operator_ranges,
        candidate,
        rope_diagnostic_mechanism=fixed.ACTIVE_ROPE_MECHANISM,
    )

    clip_counts: Counter[int] = Counter()
    row_count = 0
    for module in model.modules():
        if isinstance(module, evaluator.MSEClipGridW4A8Linear):
            values = module.clip_ratio_numerator.detach().cpu().tolist()
            clip_counts.update(int(value) for value in values)
            row_count += len(values)

    with tempfile.TemporaryDirectory(prefix="ace2-c02-reconstruct-") as directory:
        temporary = Path(directory)
        packed = temporary / "packed-w4.bin"
        layout = evaluator._write_packed_w4(model, packed)
        evaluator.write_json(temporary / "packed-w4-layout.json", layout)
        evaluator._write_scale_artifacts(
            model,
            ranges,
            runtime_operator_ranges,
            temporary,
        )
        compared = (
            "packed-w4.bin",
            "packed-w4-layout.json",
            "weight-scale32.json",
            "activation-scale32.json",
            "operator-scale32.json",
            "kv-cache-scale32.json",
            "derived-scales.json",
        )
        artifacts = {}
        mismatch_names = []
        for name in compared:
            observed = sha256_file(temporary / name)
            sealed = sha256_file(sealed_root / name)
            exact = observed == sealed
            if not exact:
                mismatch_names.append(name)
            artifacts[name] = {
                "exact_byte_match": exact,
                "reconstructed_sha256": observed,
                "sealed_sha256": sealed,
            }
        sealed_layout = {entry["name"]: entry for entry in load_json(sealed_root / "packed-w4-layout.json")}
        module_hash_mismatches = [
            entry["name"]
            for entry in layout
            if entry["qweight_sha256"]
            != sealed_layout[entry["name"]]["qweight_sha256"]
        ]
    return (
        {
            "artifact_exact_match_count": len(artifacts) - len(mismatch_names),
            "artifacts": artifacts,
            "clip_ratio_numerator_row_counts": {
                str(value): clip_counts[value] for value in CLIP_NUMERATORS
            },
            "mismatch_artifacts": mismatch_names,
            "module_qweight_hash_mismatches": module_hash_mismatches,
            "oracle": (
                "independent c02 BF16 bitfield division, exact ties-to-even W4 codes, "
                "binary64 row MSE, frozen larger-ratio tie break"
            ),
            "oracle_row_count": row_count,
            "packed_qweight_element_count": sum(
                entry["in_features"] * entry["out_features"] for entry in layout
            ),
            "status": (
                "PASS_EXACT_FROZEN_SOURCE_RECONSTRUCTION"
                if not mismatch_names and not module_hash_mismatches
                else "FAIL_SOURCE_RECONSTRUCTION"
            ),
        },
        {"ranges": ranges, "runtime_operator_ranges": runtime_operator_ranges},
    )


def install_w4_capture(
    model: nn.Module,
    captures: dict[str, torch.Tensor],
) -> tuple[list[Any], list[Callable[[], None]]]:
    hooks: list[Any] = []
    restorers: list[Callable[[], None]] = []

    def capture(key: str, value: Any) -> None:
        captures[key] = last_row(first_tensor(value))

    for name, module in model.named_modules():
        if not isinstance(module, fixed.W4A8Linear):
            continue
        original_accumulator = module.accumulator_quantized
        original_requantize = module.requantize_accumulator

        def accumulator_wrapper(
            self: fixed.W4A8Linear,
            qinput: torch.Tensor,
            *,
            n: str = name,
            original: Callable[[torch.Tensor], torch.Tensor] = original_accumulator,
        ) -> torch.Tensor:
            result = original(qinput)
            captures[f"w4.{n}.input_s8"] = last_row(qinput)
            captures[f"w4.{n}.accumulator_s64"] = last_row(result)
            return result

        def requantize_wrapper(
            self: fixed.W4A8Linear,
            accumulator: torch.Tensor,
            original_shape: tuple[int, ...],
            *,
            n: str = name,
            original: Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] = original_requantize,
        ) -> torch.Tensor:
            result = original(accumulator, original_shape)
            captures[f"w4.{n}.requant_s8"] = last_row(result)
            return result

        module.accumulator_quantized = types.MethodType(accumulator_wrapper, module)
        module.requantize_accumulator = types.MethodType(requantize_wrapper, module)

        def restore(
            target: fixed.W4A8Linear = module,
            acc: Callable[..., Any] = original_accumulator,
            req: Callable[..., Any] = original_requantize,
        ) -> None:
            target.accumulator_quantized = acc
            target.requantize_accumulator = req

        restorers.append(restore)

    hooks.append(
        model.model.embed_tokens.register_forward_hook(
            lambda _m, _inputs, output: capture("w4.model.embed_tokens.output", output)
        )
    )
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        for suffix, module in (
            ("input_layernorm", layer.input_layernorm),
            ("post_attention_layernorm", layer.post_attention_layernorm),
        ):
            name = f"{prefix}.{suffix}"
            hooks.append(
                module.register_forward_pre_hook(
                    lambda _m, inputs, n=name: capture(f"w4.{n}.input", inputs[0])
                )
            )
            hooks.append(
                module.register_forward_hook(
                    lambda _m, _inputs, output, n=name: capture(
                        f"w4.{n}.output_s8", output
                    )
                )
            )
        hooks.append(
            layer.self_attn.register_forward_hook(
                lambda _m, _inputs, output, n=prefix: capture(
                    f"w4.{n}.self_attn.output", output
                )
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                lambda _m, _inputs, output, n=prefix: capture(
                    f"w4.{n}.mlp.output", output
                )
            )
        )
        hooks.append(
            layer.register_forward_hook(
                lambda _m, _inputs, output, n=prefix: capture(
                    f"w4.{n}.output", output
                )
            )
        )
    hooks.append(
        model.model.norm.register_forward_pre_hook(
            lambda _m, inputs: capture("w4.model.norm.input", inputs[0])
        )
    )
    hooks.append(
        model.model.norm.register_forward_hook(
            lambda _m, _inputs, output: capture("w4.model.norm.output_s8", output)
        )
    )
    return hooks, restorers


def run_w4_trace(
    model: nn.Module,
    input_ids: torch.Tensor,
    captures: dict[str, torch.Tensor],
    sealed_step: dict[str, Any],
) -> dict[str, Any]:
    evaluator.enable_w4a8_kv_cache(model)
    hooks, restorers = install_w4_capture(model, captures)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
        raw = captures["w4.lm_head.requant_s8"].contiguous()
        captures["w4.ranking.logits_s8"] = raw
        full_logits = output.logits[0, -1].detach().cpu().contiguous()
        captures["w4.ranking.logits_dequantized"] = full_logits
    finally:
        for hook in hooks:
            hook.remove()
        for restore in restorers:
            restore()

    prefix = input_ids[:, :-1]
    final_token = input_ids[:, -1:]
    with torch.inference_mode():
        prefill = model(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            use_cache=True,
            logits_to_keep=1,
        )
        require(prefill.past_key_values is not None, "W4 prefill cache is missing")
        split = model(
            input_ids=final_token,
            attention_mask=torch.ones_like(input_ids),
            past_key_values=prefill.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
    split_raw_2d = (
        model.lm_head.last_raw_output[:, -1, :].detach().cpu().contiguous()
    )
    split_raw = split_raw_2d[0]
    split_logits = split.logits[0, -1].detach().cpu().contiguous()
    captures["w4.cache_split.logits_s8"] = split_raw
    captures["w4.cache_split.logits_dequantized"] = split_logits
    cache_summary = extract_cache_tensors(
        model, split.past_key_values, captures, "w4.cache"
    )
    kv_record = evaluator.kv_cache_step_record(model, split.past_key_values)
    return {
        "cache_full_prefix_integer_logits_equal": torch.equal(split_raw, raw),
        "cache_full_prefix_selected_token_equal": int(torch.argmax(split_raw))
        == int(torch.argmax(raw)),
        "cache_layers": cache_summary,
        "fixed_input_reproduces_sealed_first_logit_hash": tensor_sha256(split_raw_2d)
        == sealed_step["first_integer_logit_sha256"],
        "fixed_input_reproduces_sealed_first_token": int(torch.argmax(split_raw))
        == sealed_step["first_generated_token_id"],
        "full_prefix_selected_token_id": int(torch.argmax(raw)),
        "full_prefix_top": stable_top(raw),
        "kv_cache_record_sha256": kv_record["sha256"],
        "sealed_first_integer_logit_sha256": sealed_step[
            "first_integer_logit_sha256"
        ],
        "sealed_first_token_id": sealed_step["first_generated_token_id"],
        "split_integer_logit_sha256": tensor_sha256(split_raw_2d),
        "split_selected_token_id": int(torch.argmax(split_raw)),
    }


def compare_stage(
    records: list[dict[str, Any]],
    stage: str,
    boundary_class: str,
    reference: torch.Tensor,
    observed: torch.Tensor,
    *,
    raw_int8: torch.Tensor | None = None,
) -> None:
    records.append(
        {
            "boundary_class": boundary_class,
            "metrics": numerical_metrics(reference, observed, raw_int8=raw_int8),
            "ordinal": len(records),
            "stage": stage,
        }
    )


def build_stage_comparisons(
    model: nn.Module,
    captures: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    compare_stage(
        records,
        "model.embed_tokens.output",
        "activation",
        captures["bf16.model.embed_tokens.output"],
        captures["w4.model.embed_tokens.output"],
    )
    linear_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, fixed.W4A8Linear)
    }
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        input_norm = layer.input_layernorm
        compare_stage(
            records,
            f"{prefix}.input_layernorm.input",
            "activation",
            captures[f"bf16.{prefix}.input_layernorm.input"],
            captures[f"w4.{prefix}.input_layernorm.input"],
        )
        input_norm_raw = captures[f"w4.{prefix}.input_layernorm.output_s8"]
        compare_stage(
            records,
            f"{prefix}.input_layernorm.output",
            "normalization",
            captures[f"bf16.{prefix}.input_layernorm.output"],
            input_norm_raw.to(torch.float64) * input_norm.output_scale,
            raw_int8=input_norm_raw.to(torch.int8),
        )
        for suffix in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ):
            name = f"{prefix}.{suffix}"
            module = linear_modules[name]
            qinput = captures[f"w4.{name}.input_s8"]
            accumulator = captures[f"w4.{name}.accumulator_s64"]
            requant = captures[f"w4.{name}.requant_s8"]
            bf16_input = captures[f"bf16.{name}.input"]
            bf16_output = captures[f"bf16.{name}.output"]
            input_dequant = qinput.to(torch.float64) * module.hardware_input_scale
            accumulator_dequant = (
                accumulator.to(torch.float64)
                * module.hardware_input_scale
                * module.weight_scale.detach().cpu()
            )
            requant_dequant = (
                requant.to(torch.float64)
                * module.output_scale_per_channel.detach().cpu()
            )
            compare_stage(
                records,
                f"{name}.activation_quantization",
                (
                    "attention"
                    if suffix == "self_attn.o_proj"
                    else "activation_quantization"
                ),
                bf16_input,
                input_dequant,
                raw_int8=qinput,
            )
            compare_stage(
                records,
                f"{name}.accumulator",
                "packed_weight_accumulator",
                bf16_output,
                accumulator_dequant,
            )
            compare_stage(
                records,
                f"{name}.requantization",
                "projection_requantization",
                bf16_output,
                requant_dequant,
                raw_int8=requant,
            )
        compare_stage(
            records,
            f"{prefix}.self_attn.output",
            "attention",
            captures[f"bf16.{prefix}.self_attn.output"],
            captures[f"w4.{prefix}.self_attn.output"],
        )
        post_norm = layer.post_attention_layernorm
        compare_stage(
            records,
            f"{prefix}.post_attention_residual",
            "residual",
            captures[f"bf16.{prefix}.post_attention_layernorm.input"],
            captures[f"w4.{prefix}.post_attention_layernorm.input"],
        )
        post_norm_raw = captures[f"w4.{prefix}.post_attention_layernorm.output_s8"]
        compare_stage(
            records,
            f"{prefix}.post_attention_layernorm.output",
            "normalization",
            captures[f"bf16.{prefix}.post_attention_layernorm.output"],
            post_norm_raw.to(torch.float64) * post_norm.output_scale,
            raw_int8=post_norm_raw.to(torch.int8),
        )
        for suffix in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            name = f"{prefix}.{suffix}"
            module = linear_modules[name]
            qinput = captures[f"w4.{name}.input_s8"]
            accumulator = captures[f"w4.{name}.accumulator_s64"]
            requant = captures[f"w4.{name}.requant_s8"]
            bf16_input = captures[f"bf16.{name}.input"]
            bf16_output = captures[f"bf16.{name}.output"]
            boundary = (
                "nonlinear"
                if suffix == "mlp.down_proj"
                else "activation_quantization"
            )
            compare_stage(
                records,
                f"{name}.activation_quantization",
                boundary,
                bf16_input,
                qinput.to(torch.float64) * module.hardware_input_scale,
                raw_int8=qinput,
            )
            compare_stage(
                records,
                f"{name}.accumulator",
                "packed_weight_accumulator",
                bf16_output,
                accumulator.to(torch.float64)
                * module.hardware_input_scale
                * module.weight_scale.detach().cpu(),
            )
            compare_stage(
                records,
                f"{name}.requantization",
                "projection_requantization",
                bf16_output,
                requant.to(torch.float64)
                * module.output_scale_per_channel.detach().cpu(),
                raw_int8=requant,
            )
        compare_stage(
            records,
            f"{prefix}.mlp.output",
            "nonlinear",
            captures[f"bf16.{prefix}.mlp.output"],
            captures[f"w4.{prefix}.mlp.output"],
        )
        compare_stage(
            records,
            f"{prefix}.output",
            "residual",
            captures[f"bf16.{prefix}.output"],
            captures[f"w4.{prefix}.output"],
        )

    final_norm = model.model.norm
    compare_stage(
        records,
        "model.norm.input",
        "activation",
        captures["bf16.model.norm.input"],
        captures["w4.model.norm.input"],
    )
    final_norm_raw = captures["w4.model.norm.output_s8"]
    compare_stage(
        records,
        "model.norm.output",
        "normalization",
        captures["bf16.model.norm.output"],
        final_norm_raw.to(torch.float64) * final_norm.output_scale,
        raw_int8=final_norm_raw.to(torch.int8),
    )
    lm_head = linear_modules["lm_head"]
    for suffix, boundary_class, reference_key, observed, raw in (
        (
            "activation_quantization",
            "activation_quantization",
            "bf16.lm_head.input",
            captures["w4.lm_head.input_s8"].to(torch.float64)
            * lm_head.hardware_input_scale,
            captures["w4.lm_head.input_s8"],
        ),
        (
            "accumulator",
            "packed_weight_accumulator",
            "bf16.lm_head.output",
            captures["w4.lm_head.accumulator_s64"].to(torch.float64)
            * lm_head.hardware_input_scale
            * lm_head.weight_scale.detach().cpu(),
            None,
        ),
        (
            "requantization",
            "projection_requantization",
            "bf16.lm_head.output",
            captures["w4.lm_head.requant_s8"].to(torch.float64)
            * lm_head.output_scale_per_channel.detach().cpu(),
            captures["w4.lm_head.requant_s8"],
        ),
    ):
        compare_stage(
            records,
            f"lm_head.{suffix}",
            boundary_class,
            captures[reference_key],
            observed,
            raw_int8=raw,
        )
    return records


def classify_boundary(
    source_reconstruction: dict[str, Any],
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    if source_reconstruction["status"] != "PASS_EXACT_FROZEN_SOURCE_RECONSTRUCTION":
        return {
            "earliest_divergence": "packed-weight/Scale32 numerical realization",
            "first_material_stage": None,
            "first_severe_stage": None,
            "remediation_class": "PACKAGING_OR_QUANTIZER_REALIZATION_REPAIR",
        }
    material = next(
        (record for record in stages if record["metrics"]["severity"] != "TRACKING"),
        None,
    )
    severe = next(
        (record for record in stages if record["metrics"]["severity"] == "SEVERE"),
        None,
    )
    decisive = material
    if decisive is None:
        remediation = "NO_PRE_RANKING_MATERIAL_DIVERGENCE"
        earliest = "decode/logit ranking"
    else:
        boundary = decisive["boundary_class"]
        earliest = decisive["stage"]
        if boundary in {"activation_quantization", "projection_requantization"}:
            remediation = "STRUCTURALLY_DIFFERENT_PTQ_SCALE_TOPOLOGY_REMAINS_CREDIBLE"
        elif boundary == "packed_weight_accumulator":
            remediation = "WEIGHT_REPRESENTATION_REQUIRES_NEW_PTQ_CODEBOOK_OR_QAT"
        elif boundary == "attention":
            remediation = "FIXED_POINT_ATTENTION_VALUE_PATH_OR_SCALE_ALIGNMENT_REPAIR_PTQ_CREDIBLE"
        elif boundary in {"normalization", "nonlinear", "residual"}:
            remediation = "FIXED_POINT_OPERATOR_OR_SCALE_ALIGNMENT_REPAIR"
        else:
            remediation = "BOUNDARY_REMAINS_UNCLASSIFIED"
    interval = None
    if material is not None:
        previous = stages[material["ordinal"] - 1] if material["ordinal"] else None
        interval = {
            "first_material_stage": material["stage"],
            "last_tracking_stage": previous["stage"] if previous is not None else None,
        }
    return {
        "earliest_divergence": earliest,
        "earliest_material_interval": interval,
        "first_material_stage": material,
        "first_severe_stage": severe,
        "remediation_class": remediation,
    }


def run_model(
    alias: str,
    contract: dict[str, Any],
    input_ids: torch.Tensor,
    fixed_input: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    sealed_root = SEALED_ROOTS[alias]
    before_hashes = {
        path.name: sha256_file(path)
        for path in sorted(sealed_root.iterdir())
        if path.is_file()
    }
    spec = frozen.model_spec(contract, alias)
    frozen.verify_local_model_inputs(contract, spec)
    model = frozen.load_model(spec)
    captures: dict[str, torch.Tensor] = {}
    bf16 = run_bf16_trace(model, input_ids, captures)
    source_reconstruction, _runtime = reconstruct_model_from_source_oracle(
        model, sealed_root, contract
    )
    w4 = run_w4_trace(
        model,
        input_ids,
        captures,
        fixed_input["sealed_first_step"][alias],
    )
    stages = build_stage_comparisons(model, captures)
    boundary = classify_boundary(source_reconstruction, stages)

    bf16_logits = captures["bf16.ranking.logits"]
    w4_raw = captures["w4.ranking.logits_s8"]
    lm_head = model.lm_head
    w4_logits = w4_raw.to(torch.float64) * lm_head.output_scale_per_channel.detach().cpu()
    bf16_top = int(torch.argmax(bf16_logits))
    w4_top = int(torch.argmax(w4_raw))
    ranking = {
        "bf16_selected_token_id": bf16_top,
        "bf16_selected_token_rank_in_w4": rank_of(w4_raw, bf16_top),
        "bf16_top": stable_top(bf16_logits),
        "selected_token_equal": bf16_top == w4_top,
        "w4_selected_token_id": w4_top,
        "w4_selected_token_rank_in_bf16": rank_of(bf16_logits, w4_top),
        "w4_top": stable_top(w4_raw),
        "logit_metrics": numerical_metrics(bf16_logits, w4_logits, raw_int8=w4_raw),
    }

    model_root = output_root / alias
    model_root.mkdir(parents=True, exist_ok=False)
    bundle = write_tensor_bundle(
        model_root / "fixed-input-tensors.bin",
        captures,
        public_path=f"{alias}/fixed-input-tensors.bin",
    )
    after_hashes = {
        path.name: sha256_file(path)
        for path in sorted(sealed_root.iterdir())
        if path.is_file()
    }
    require(before_hashes == after_hashes, "sealed c01 artifacts changed during diagnostic")
    result = {
        "alias": alias,
        "boundary_classification": boundary,
        "bf16_cache_diagnostic": bf16,
        "fixed_input": fixed_input,
        "model_identity_sha256": canonical_sha256(spec["contract_identity"]),
        "prohibitions_observed": {
            "benchmark_scoring_performed": False,
            "candidate_bundle_created": False,
            "candidate_or_official_namespace_created": False,
            "continuation_generation_performed": False,
            "execute_command_invoked": False,
            "frozen_bf16_control_mutated": False,
            "sealed_c01_mutated": False,
        },
        "ranking": ranking,
        "schema_version": 1,
        "sealed_attempt_root": sealed_root.relative_to(ROOT).as_posix(),
        "source_reconstruction": source_reconstruction,
        "stage_comparisons": stages,
        "status": (
            "PASS_LOCALIZED_NON_SCORING_DIAGNOSTIC"
            if source_reconstruction["status"]
            == "PASS_EXACT_FROZEN_SOURCE_RECONSTRUCTION"
            and w4["fixed_input_reproduces_sealed_first_token"]
            and w4["fixed_input_reproduces_sealed_first_logit_hash"]
            else "FAIL_DIAGNOSTIC_REPRODUCTION"
        ),
        "tensor_bundle": bundle,
        "w4_cache_diagnostic": w4,
    }
    (model_root / "result.json").write_bytes(canonical_bytes(result))
    del model
    gc.collect()
    return result


def aggregate_results(
    results: dict[str, dict[str, Any]], output_root: Path
) -> dict[str, Any]:
    exact_source = all(
        result["source_reconstruction"]["status"]
        == "PASS_EXACT_FROZEN_SOURCE_RECONSTRUCTION"
        for result in results.values()
    )
    reproduces = all(
        result["w4_cache_diagnostic"]["fixed_input_reproduces_sealed_first_token"]
        and result["w4_cache_diagnostic"][
            "fixed_input_reproduces_sealed_first_logit_hash"
        ]
        for result in results.values()
    )
    classes = {
        result["boundary_classification"]["remediation_class"]
        for result in results.values()
    }
    return {
        "boundary_classification": {
            "earliest_exonerated_boundary": (
                "frozen-source c01 qweight, clip-selected Scale32, packed bytes, activation/operator/KV Scale32 exports"
                if exact_source
                else None
            ),
            "model_isolated_first_divergence": {
                alias: result["boundary_classification"] for alias, result in results.items()
            },
            "sealed_collapse_reproduced_on_fixed_input": reproduces,
        },
        "models": {
            alias: {
                "result_path": f"{alias}/result.json",
                "result_sha256": sha256_file(output_root / alias / "result.json")
                if (output_root / alias / "result.json").is_file()
                else None,
                "status": result["status"],
            }
            for alias, result in results.items()
        },
        "prohibitions_observed": {
            "benchmark_scoring_performed": False,
            "candidate_bundle_created": False,
            "candidate_or_official_namespace_created": False,
            "continuation_generation_performed": False,
            "execute_command_invoked": False,
            "retraining_performed": False,
        },
        "remediation_classification": {
            "decision": (
                next(iter(classes)) if len(classes) == 1 else "MODEL_SPECIFIC_REMEDIATION_CLASSES"
            ),
            "full_qat_or_quantization_aware_lora_required": (
                all(
                    value == "WEIGHT_REPRESENTATION_REQUIRES_NEW_PTQ_CODEBOOK_OR_QAT"
                    for value in classes
                )
            ),
            "reason": (
                "The source reconstruction removes artifact realization as a cause. The first material fixed-input stage bounds the earliest cause, while the first severe stage records model-specific downstream amplification."
            ),
        },
        "severity_rule": {
            "material": "relative_rmse>=0.25 or cosine<=0.9 or int8_saturation_fraction>=0.05",
            "severe": "relative_rmse>=1.0 or cosine<=0.5 or int8_saturation_fraction>=0.25",
        },
        "schema_version": 1,
        "status": (
            "PASS_EARLIEST_BOUNDARY_LOCALIZED"
            if exact_source and reproduces
            else "FAIL_C02_NUMERICAL_PATH_DIAGNOSTIC"
        ),
    }


def write_sha256s(output_root: Path) -> None:
    members = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    body = "".join(
        f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}\n"
        for path in members
    )
    (output_root / "SHA256SUMS").write_text(body, encoding="ascii")


def verify_existing(output_root: Path) -> int:
    sums = output_root / "SHA256SUMS"
    require(sums.is_file(), "SHA256SUMS is missing")
    for line in sums.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        path = output_root / name
        require(path.is_file(), f"evidence member is missing: {name}")
        require(sha256_file(path) == digest, f"evidence member hash differs: {name}")
    aggregate = load_json(output_root / "result.json")
    require(aggregate["status"] == "PASS_EARLIEST_BOUNDARY_LOCALIZED", "aggregate status differs")
    print(f"PASS {output_root.relative_to(ROOT)} {sha256_file(output_root / 'result.json')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    require(ROOT.resolve() in output_root.parents, "output must remain repository-local")
    require("candidate" not in output_root.parts, "candidate namespace is forbidden")
    if args.verify_existing:
        return verify_existing(output_root)
    require(not output_root.exists(), f"diagnostic output already exists: {output_root}")
    require(1 <= args.threads <= 32, "thread count must be in 1..32")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    contract, contract_sha256 = frozen.load_contract()
    input_ids, fixed_input = sealed_fixed_input()
    output_root.mkdir(parents=True)
    try:
        results = {
            alias: run_model(
                alias, contract, input_ids, fixed_input, output_root
            )
            for alias in MODEL_ORDER
        }
        aggregate = aggregate_results(results, output_root)
        aggregate["contract_sha256"] = contract_sha256
        (output_root / "result.json").write_bytes(canonical_bytes(aggregate))
        write_sha256s(output_root)
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    print(
        f"WROTE {output_root.relative_to(ROOT)} "
        f"{sha256_file(output_root / 'result.json')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
