#!/usr/bin/env python3
"""Trace frozen BF16/W4A8 boundaries from attention score through LM head."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

import ace2_full_model_fixed_point as fixed_point
from ace2_full_model_fixed_point import (
    FixedRMSNorm,
    PROMPT_MANIFEST,
    QUALITY_CONFIG,
    ROOT,
    W4A8Linear,
    calibrate,
    hash_records,
    hash_token_sequences,
    load_contracts,
    repeat_kv,
    replace_fixed_operators,
    replace_linears,
    seed_everything,
    selected_texts,
    sha256_file,
    tokenize_prompts,
    tokenize_wikitext,
    utc_now,
    validate_runtime,
)
from localize_quality_divergence import compare_tensor, write_sha256s


def _tensor_output(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, Tensor):
                return item
    raise TypeError(f"boundary output does not contain a tensor: {type(value).__name__}")


def _copy(value: Tensor) -> Tensor:
    return value.detach().to(device="cpu").contiguous().clone()


def capture_module_boundaries(
    model: nn.Module,
    input_ids: Tensor,
    *,
    capture_attention_inputs: bool,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    hooks: list[Any] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):

            def capture_linear(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                boundary_name: str = name,
            ) -> None:
                captured[boundary_name] = _copy(_tensor_output(output))

            hooks.append(module.register_forward_hook(capture_linear))

    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"

        def capture_post_attention_input(
            _module: nn.Module,
            inputs: tuple[Any, ...],
            *,
            boundary_name: str = f"{prefix}.post_attention_residual",
        ) -> None:
            captured[boundary_name] = _copy(_tensor_output(inputs))

        def capture_post_attention_norm(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            boundary_name: str = f"{prefix}.post_attention_layernorm",
        ) -> None:
            captured[boundary_name] = _copy(_tensor_output(output))

        def capture_layer_output(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            boundary_name: str = f"{prefix}.post_mlp_residual",
        ) -> None:
            captured[boundary_name] = _copy(_tensor_output(output))

        hooks.extend(
            [
                layer.post_attention_layernorm.register_forward_pre_hook(
                    capture_post_attention_input
                ),
                layer.post_attention_layernorm.register_forward_hook(
                    capture_post_attention_norm
                ),
                layer.register_forward_hook(capture_layer_output),
            ]
        )
        if capture_attention_inputs:

            def capture_attention_input(
                _module: nn.Module,
                _args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                boundary_name: str = f"{prefix}.self_attn.inputs",
            ) -> None:
                position = kwargs.get("position_embeddings")
                if not isinstance(position, tuple) or len(position) != 2:
                    raise RuntimeError(f"{boundary_name} lacks position embeddings")
                mask = kwargs.get("attention_mask")
                captured[boundary_name] = {
                    "attention_mask": None if mask is None else _copy(mask),
                    "position_embeddings": tuple(_copy(value) for value in position),
                }

            hooks.append(
                layer.self_attn.register_forward_pre_hook(
                    capture_attention_input,
                    with_kwargs=True,
                )
            )

    def capture_final_norm(
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        captured["model.norm"] = _copy(_tensor_output(output))

    hooks.append(model.model.norm.register_forward_hook(capture_final_norm))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    return captured


def reconstruct_baseline_boundaries(
    model: nn.Module,
    captured: dict[str, Any],
) -> dict[str, Tensor]:
    boundaries: dict[str, Tensor] = {}
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        attention = layer.self_attn
        query_projection = captured[f"{prefix}.self_attn.q_proj"]
        key_projection = captured[f"{prefix}.self_attn.k_proj"]
        value_projection = captured[f"{prefix}.self_attn.v_proj"]
        input_shape = query_projection.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query = query_projection.view(hidden_shape).transpose(1, 2)
        key = key_projection.view(hidden_shape).transpose(1, 2)
        value = value_projection.view(hidden_shape).transpose(1, 2)
        attention_inputs = captured[f"{prefix}.self_attn.inputs"]
        query, key = apply_rotary_pos_emb(
            query,
            key,
            *attention_inputs["position_embeddings"],
        )
        key = repeat_kv(key, attention.num_key_value_groups)
        value = repeat_kv(value, attention.num_key_value_groups)
        score = torch.matmul(query, key.transpose(2, 3)) * attention.scaling
        masked_score = score
        attention_mask = attention_inputs["attention_mask"]
        if attention_mask is not None:
            masked_score = score + attention_mask[:, :, :, : key.shape[-2]]
            valid = attention_mask[:, :, :, : key.shape[-2]] >= 0
            score = score - torch.where(
                valid,
                score,
                torch.full_like(score, float("-inf")),
            ).amax(dim=-1, keepdim=True)
        else:
            score = score - score.amax(dim=-1, keepdim=True)
        probability = nn.functional.softmax(
            masked_score,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)
        attention_value = torch.matmul(probability, value)
        boundaries[f"{prefix}.score"] = score
        boundaries[f"{prefix}.softmax_from_score_exact"] = probability
        boundaries[f"{prefix}.softmax"] = probability
        boundaries[f"{prefix}.attention_value"] = attention_value
        boundaries[f"{prefix}.o_projection"] = captured[
            f"{prefix}.self_attn.o_proj"
        ]
        boundaries[f"{prefix}.post_attention_residual"] = captured[
            f"{prefix}.post_attention_residual"
        ]
        boundaries[f"{prefix}.post_attention_layernorm"] = captured[
            f"{prefix}.post_attention_layernorm"
        ]
        gate = captured[f"{prefix}.mlp.gate_proj"]
        up = captured[f"{prefix}.mlp.up_proj"]
        boundaries[f"{prefix}.gate_projection"] = gate
        boundaries[f"{prefix}.up_projection"] = up
        boundaries[f"{prefix}.silu_gate"] = layer.mlp.act_fn(gate) * up
        boundaries[f"{prefix}.down_projection"] = captured[
            f"{prefix}.mlp.down_proj"
        ]
        boundaries[f"{prefix}.post_mlp_residual"] = captured[
            f"{prefix}.post_mlp_residual"
        ]
    boundaries["model.norm"] = captured["model.norm"]
    boundaries["lm_head"] = captured["lm_head"]
    return boundaries


@contextmanager
def trace_fixed_raw(
    model: nn.Module,
) -> Iterator[dict[str, Tensor]]:
    raw: dict[str, Tensor] = {}
    linear_names = {
        id(module): name
        for name, module in model.named_modules()
        if isinstance(module, W4A8Linear)
    }
    counters = {
        "score": 0,
        "softmax": 0,
        "attention_value": 0,
        "silu": 0,
        "residual": 0,
        "down_projection_fusion": 0,
        "cross_layer_error_carry": 0,
    }
    original_linear = W4A8Linear.forward_quantized
    original_score = fixed_point.fixed_attention_scores_raw
    original_shadow_score = fixed_point.projection_shadow_staged_attention_scores_raw
    original_tile_score = fixed_point.tile_max_delta_attention_scores_raw
    original_bfp_score = fixed_point.tile_bfp_attention_scores_raw
    original_native_score = fixed_point.native_accumulator_tagged_attention_scores_raw
    original_qk_residual_score = fixed_point.qk_residual_cross_term_scores_raw
    original_v_residual_score = fixed_point.v_residual_baseline_scores_raw
    original_softmax = fixed_point.fixed_softmax_raw
    original_tile_softmax = fixed_point.tile_max_delta_softmax_raw
    original_bfp_softmax = fixed_point.tile_bfp_softmax_raw
    original_native_softmax = fixed_point.native_accumulator_tagged_softmax_raw
    original_attention_value = fixed_point.fixed_attention_value_raw
    original_v_residual_attention_value = fixed_point.v_residual_attention_value_raw
    original_silu = fixed_point.fixed_silu_gate_raw
    original_residual = fixed_point.fixed_residual_add
    original_down_projection_fusion = fixed_point.down_projection_residual_fusion_raw
    original_cross_layer_error_carry = fixed_point.cross_layer_error_carry_produce_raw

    def trace_linear(module: W4A8Linear, qinput: Tensor) -> Tensor:
        output = original_linear(module, qinput)
        raw[linear_names[id(module)]] = _copy(output)
        return output

    def trace_score(*args: Any, **kwargs: Any) -> Tensor:
        output = original_score(*args, **kwargs)
        index = counters["score"]
        raw[f"model.layers.{index}.score"] = _copy(output)
        counters["score"] += 1
        return output

    def trace_shadow_score(*args: Any, **kwargs: Any) -> Tensor:
        output = original_shadow_score(*args, **kwargs)
        index = counters["score"]
        raw[f"model.layers.{index}.score"] = _copy(output)
        counters["score"] += 1
        return output

    def trace_tile_score(*args: Any, **kwargs: Any) -> Tensor:
        output = original_tile_score(*args, **kwargs)
        index = counters["score"]
        raw[f"model.layers.{index}.score"] = _copy(output)
        counters["score"] += 1
        return output

    def trace_bfp_score(*args: Any, **kwargs: Any) -> Tensor:
        output = original_bfp_score(*args, **kwargs)
        index = counters["score"]
        raw[f"model.layers.{index}.score"] = _copy(output)
        counters["score"] += 1
        return output

    def trace_native_score(*args: Any, **kwargs: Any) -> Tensor:
        output = original_native_score(*args, **kwargs)
        index = counters["score"]
        raw[f"model.layers.{index}.score"] = _copy(output)
        counters["score"] += 1
        return output

    def trace_qk_residual_score(*args: Any, **kwargs: Any) -> Tensor:
        output = original_qk_residual_score(*args, **kwargs)
        index = counters["score"]
        raw[f"model.layers.{index}.score"] = _copy(output)
        counters["score"] += 1
        return output

    def trace_v_residual_score(*args: Any, **kwargs: Any) -> Tensor:
        output = original_v_residual_score(*args, **kwargs)
        index = counters["score"]
        raw[f"model.layers.{index}.score"] = _copy(output)
        counters["score"] += 1
        return output

    def trace_softmax(*args: Any, **kwargs: Any) -> Tensor:
        output = original_softmax(*args, **kwargs)
        index = counters["softmax"]
        raw[f"model.layers.{index}.softmax"] = _copy(output)
        counters["softmax"] += 1
        return output

    def trace_tile_softmax(*args: Any, **kwargs: Any) -> Tensor:
        output = original_tile_softmax(*args, **kwargs)
        index = counters["softmax"]
        raw[f"model.layers.{index}.softmax"] = _copy(output)
        counters["softmax"] += 1
        return output

    def trace_bfp_softmax(*args: Any, **kwargs: Any) -> Tensor:
        output = original_bfp_softmax(*args, **kwargs)
        index = counters["softmax"]
        raw[f"model.layers.{index}.softmax"] = _copy(output)
        counters["softmax"] += 1
        return output

    def trace_native_softmax(*args: Any, **kwargs: Any) -> Tensor:
        output = original_native_softmax(*args, **kwargs)
        index = counters["softmax"]
        raw[f"model.layers.{index}.softmax"] = _copy(output)
        counters["softmax"] += 1
        return output

    def trace_attention_value(*args: Any, **kwargs: Any) -> Tensor:
        output = original_attention_value(*args, **kwargs)
        index = counters["attention_value"]
        raw[f"model.layers.{index}.attention_value"] = _copy(output)
        counters["attention_value"] += 1
        return output

    def trace_v_residual_attention_value(*args: Any, **kwargs: Any) -> Tensor:
        output = original_v_residual_attention_value(*args, **kwargs)
        index = counters["attention_value"]
        raw[f"model.layers.{index}.attention_value"] = _copy(output)
        counters["attention_value"] += 1
        return output

    def trace_silu(*args: Any, **kwargs: Any) -> Tensor:
        output = original_silu(*args, **kwargs)
        index = counters["silu"]
        raw[f"model.layers.{index}.silu_gate"] = _copy(output)
        counters["silu"] += 1
        return output

    def trace_residual(*args: Any, **kwargs: Any) -> Tensor:
        output = original_residual(*args, **kwargs)
        call = counters["residual"]
        index = call // 2
        suffix = "post_attention_residual" if call % 2 == 0 else "post_mlp_residual"
        raw[f"model.layers.{index}.{suffix}"] = _copy(output)
        counters["residual"] += 1
        return output

    def trace_down_projection_fusion(
        runtime: fixed_point.DownProjectionResidualFusionRuntime,
        layer_index: int,
        accumulator: Tensor,
        residual: Tensor,
        baseline_down_projection: Tensor,
        baseline_post_mlp: Tensor,
    ) -> Tensor:
        output = original_down_projection_fusion(
            runtime,
            layer_index,
            accumulator,
            residual,
            baseline_down_projection,
            baseline_post_mlp,
        )
        raw[f"model.layers.{layer_index}.mlp.down_proj"] = _copy(
            baseline_down_projection
        )
        if runtime.mode == "candidate":
            layer = model.model.layers[layer_index]
            raw[f"model.layers.{layer_index}.post_mlp_residual"] = _copy(
                output.to(torch.float64) * layer.post_mlp_scale
            )
        counters["down_projection_fusion"] += 1
        return output

    def trace_cross_layer_error_carry(
        runtime: fixed_point.CrossLayerErrorCarryRuntime,
        layer_index: int,
        accumulator: Tensor,
        residual: Tensor,
        baseline_down_projection: Tensor,
        baseline_post_mlp: Tensor,
    ) -> Tensor:
        output = original_cross_layer_error_carry(
            runtime,
            layer_index,
            accumulator,
            residual,
            baseline_down_projection,
            baseline_post_mlp,
        )
        raw[f"model.layers.{layer_index}.mlp.down_proj"] = _copy(
            baseline_down_projection
        )
        raw[f"model.layers.{layer_index}.post_mlp_residual"] = _copy(
            output.to(torch.float64) * runtime.destination_scale(layer_index)
        )
        counters["cross_layer_error_carry"] += 1
        return output

    W4A8Linear.forward_quantized = trace_linear
    fixed_point.fixed_attention_scores_raw = trace_score
    fixed_point.projection_shadow_staged_attention_scores_raw = trace_shadow_score
    fixed_point.tile_max_delta_attention_scores_raw = trace_tile_score
    fixed_point.tile_bfp_attention_scores_raw = trace_bfp_score
    fixed_point.native_accumulator_tagged_attention_scores_raw = trace_native_score
    fixed_point.qk_residual_cross_term_scores_raw = trace_qk_residual_score
    fixed_point.v_residual_baseline_scores_raw = trace_v_residual_score
    fixed_point.fixed_softmax_raw = trace_softmax
    fixed_point.tile_max_delta_softmax_raw = trace_tile_softmax
    fixed_point.tile_bfp_softmax_raw = trace_bfp_softmax
    fixed_point.native_accumulator_tagged_softmax_raw = trace_native_softmax
    fixed_point.fixed_attention_value_raw = trace_attention_value
    fixed_point.v_residual_attention_value_raw = trace_v_residual_attention_value
    fixed_point.fixed_silu_gate_raw = trace_silu
    fixed_point.fixed_residual_add = trace_residual
    fixed_point.down_projection_residual_fusion_raw = trace_down_projection_fusion
    fixed_point.cross_layer_error_carry_produce_raw = trace_cross_layer_error_carry
    try:
        yield raw
        fusion_runtime = getattr(model, "ace2_down_projection_fusion_runtime", None)
        fusion_active = fusion_runtime is not None
        carry_runtime = getattr(model, "ace2_cross_layer_error_carry_runtime", None)
        carry_active = carry_runtime is not None
        expected = {
            "score": 24,
            "softmax": 24,
            "attention_value": 24,
            "silu": 24,
            "residual": 48,
            "down_projection_fusion": 24 if fusion_active else 0,
            "cross_layer_error_carry": 24 if carry_active else 0,
        }
        if counters != expected:
            raise RuntimeError(f"fixed trace call counts differ: {counters} != {expected}")
    finally:
        W4A8Linear.forward_quantized = original_linear
        fixed_point.fixed_attention_scores_raw = original_score
        fixed_point.projection_shadow_staged_attention_scores_raw = original_shadow_score
        fixed_point.tile_max_delta_attention_scores_raw = original_tile_score
        fixed_point.tile_bfp_attention_scores_raw = original_bfp_score
        fixed_point.native_accumulator_tagged_attention_scores_raw = original_native_score
        fixed_point.qk_residual_cross_term_scores_raw = original_qk_residual_score
        fixed_point.v_residual_baseline_scores_raw = original_v_residual_score
        fixed_point.fixed_softmax_raw = original_softmax
        fixed_point.tile_max_delta_softmax_raw = original_tile_softmax
        fixed_point.tile_bfp_softmax_raw = original_bfp_softmax
        fixed_point.native_accumulator_tagged_softmax_raw = original_native_softmax
        fixed_point.fixed_attention_value_raw = original_attention_value
        fixed_point.v_residual_attention_value_raw = original_v_residual_attention_value
        fixed_point.fixed_silu_gate_raw = original_silu
        fixed_point.fixed_residual_add = original_residual
        fixed_point.down_projection_residual_fusion_raw = original_down_projection_fusion
        fixed_point.cross_layer_error_carry_produce_raw = original_cross_layer_error_carry


def dequantize_fixed_boundaries(
    model: nn.Module,
    captured: dict[str, Any],
    raw: dict[str, Tensor],
) -> dict[str, Tensor]:
    boundaries: dict[str, Tensor] = {}
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        attention = layer.self_attn
        mlp = layer.mlp
        score_fraction = (
            44
            if (
                attention.shared_native_accumulator_tagged_attention
                or attention.shared_qk_residual_mode
                or attention.shared_v_residual_mode
            )
            else 17
            if (
                attention.layer0_tile_max_delta_attention
                or attention.layer0_tile_bfp_score_attention
            )
            else 9
        )
        boundaries[f"{prefix}.score"] = raw[f"{prefix}.score"].to(torch.float64) / (
            1 << score_fraction
        )
        boundaries[f"{prefix}.softmax_from_score_exact"] = nn.functional.softmax(
            boundaries[f"{prefix}.score"],
            dim=-1,
            dtype=torch.float64,
        )
        boundaries[f"{prefix}.softmax"] = raw[f"{prefix}.softmax"].to(torch.float64) / (
            1 << 15
        )
        boundaries[f"{prefix}.attention_value"] = (
            raw[f"{prefix}.attention_value"].to(torch.float64)
            * attention.v_proj.output_scale
        )
        boundaries[f"{prefix}.o_projection"] = (
            raw[f"{prefix}.self_attn.o_proj"].to(torch.float64)
            * attention.o_proj.output_scale
        )
        boundaries[f"{prefix}.post_attention_residual"] = raw[
            f"{prefix}.post_attention_residual"
        ].to(torch.float64)
        boundaries[f"{prefix}.post_attention_layernorm"] = (
            captured[f"{prefix}.post_attention_layernorm"].to(torch.float64)
            * layer.post_attention_layernorm.output_scale
        )
        boundaries[f"{prefix}.gate_projection"] = (
            raw[f"{prefix}.mlp.gate_proj"].to(torch.float64)
            * mlp.gate_proj.output_scale
        )
        boundaries[f"{prefix}.up_projection"] = (
            raw[f"{prefix}.mlp.up_proj"].to(torch.float64)
            * mlp.up_proj.output_scale
        )
        boundaries[f"{prefix}.silu_gate"] = (
            raw[f"{prefix}.silu_gate"].to(torch.float64)
            * mlp.down_proj.input_scale
        )
        boundaries[f"{prefix}.down_projection"] = (
            raw[f"{prefix}.mlp.down_proj"].to(torch.float64)
            * mlp.down_proj.output_scale
        )
        boundaries[f"{prefix}.post_mlp_residual"] = raw[
            f"{prefix}.post_mlp_residual"
        ].to(torch.float64)
    if not isinstance(model.model.norm, FixedRMSNorm):
        raise TypeError("fixed final RMSNorm is missing")
    boundaries["model.norm"] = (
        captured["model.norm"].to(torch.float64) * model.model.norm.output_scale
    )
    boundaries["lm_head"] = (
        raw["lm_head"].to(torch.float64) * model.lm_head.output_scale
    )
    return boundaries


def compare_boundaries(
    baseline: dict[str, Tensor],
    candidate: dict[str, Tensor],
) -> tuple[dict[str, Any], str | None]:
    if list(baseline) != list(candidate):
        raise RuntimeError("baseline and candidate boundary order differs")
    comparisons: dict[str, Any] = {}
    first_material: str | None = None
    for name in baseline:
        reference = baseline[name].to(torch.float64)
        fixed = candidate[name].to(torch.float64)
        if name.endswith(".score"):
            sequence = reference.shape[-1]
            valid = torch.ones(
                sequence,
                sequence,
                dtype=torch.bool,
            ).tril().view(1, 1, sequence, sequence).expand_as(reference)
            reference = reference[valid]
            fixed = fixed[valid]
        metrics = compare_tensor(reference, fixed)
        comparisons[name] = metrics
        if first_material is None and (
            metrics["relative_l2_error"] >= 0.25
            or (
                metrics["cosine_similarity"] is not None
                and metrics["cosine_similarity"] <= 0.95
            )
        ):
            first_material = name
    return comparisons, first_material


def run(
    output_dir: Path,
    diagnostic_rope_mechanism: str | None = None,
    capture_token_limit: int = 32,
) -> dict[str, Any]:
    manifest, config, rtl_binding = load_contracts(require_rtl_binding=False)
    versions = validate_runtime(config)
    seed_everything(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    calibration_spec = manifest["datasets"]["c4_calibration"]
    c4_spec = manifest["datasets"]["c4_en_512"]
    wiki_spec = manifest["datasets"]["wikitext2"]
    calibration_text = selected_texts(calibration_spec, limit=1)
    c4_text = selected_texts(c4_spec, limit=1)
    wiki_text = selected_texts(wiki_spec, limit=16)
    model_spec = manifest["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
    )
    calibration_prompt = tokenize_prompts(
        tokenizer,
        calibration_text,
        calibration_spec["token_limit"],
    )[0][:, :32]
    if not 2 <= capture_token_limit <= 128:
        raise ValueError("capture token limit must be in 2..128")
    prompts = {
        "c4_en_512": tokenize_prompts(
            tokenizer,
            c4_text,
            c4_spec["token_limit"],
        )[0][:, :capture_token_limit],
        "wikitext2": tokenize_wikitext(
            tokenizer,
            wiki_text,
            wiki_spec["token_limit"],
            wiki_spec["join"],
        )[0][:, :capture_token_limit],
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision != model_spec["revision"]:
        raise RuntimeError(
            f"model resolved to {resolved_revision}, expected {model_spec['revision']}"
        )
    ranges, operator_ranges = calibrate(model, [calibration_prompt])
    baseline: dict[str, dict[str, Tensor]] = {}
    for dataset, prompt in prompts.items():
        seed_everything(config)
        captured = capture_module_boundaries(
            model,
            prompt,
            capture_attention_inputs=True,
        )
        baseline[dataset] = reconstruct_baseline_boundaries(model, captured)
    replace_linears(
        model,
        ranges,
        rope_diagnostic_mechanism=diagnostic_rope_mechanism,
    )
    replace_fixed_operators(
        model,
        operator_ranges,
        rope_diagnostic_mechanism=diagnostic_rope_mechanism,
    )
    comparisons: dict[str, Any] = {}
    first_material: dict[str, str | None] = {}
    fixed_boundary_hashes: dict[str, dict[str, str]] = {}
    fusion_passes: dict[str, dict[str, Any]] = {}
    carry_passes: dict[str, dict[str, Any]] = {}
    candidate_execution_failure: dict[str, Any] | None = None
    for dataset, prompt in prompts.items():
        seed_everything(config)
        try:
            with trace_fixed_raw(model) as raw:
                captured = capture_module_boundaries(
                    model,
                    prompt,
                    capture_attention_inputs=False,
                )
        except (OverflowError, RuntimeError, ValueError) as exc:
            if (
                diagnostic_rope_mechanism
                != "cross_layer_quantization_error_carry_final_output_v1"
            ):
                raise
            carry_runtime = getattr(
                model,
                "ace2_cross_layer_error_carry_runtime",
                None,
            )
            if carry_runtime is not None:
                carry_passes[dataset] = carry_runtime.snapshot()
            candidate_execution_failure = {
                "dataset": dataset,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            break
        candidate = dequantize_fixed_boundaries(model, captured, raw)
        comparisons[dataset], first_material[dataset] = compare_boundaries(
            baseline[dataset],
            candidate,
        )
        fixed_boundary_hashes[dataset] = {
            name: fixed_point.sha256_tensor(value)
            for name, value in candidate.items()
        }
        fusion_runtime = getattr(model, "ace2_down_projection_fusion_runtime", None)
        if fusion_runtime is not None:
            fusion_passes[dataset] = fusion_runtime.consume_completed_pass()
        carry_runtime = getattr(model, "ace2_cross_layer_error_carry_runtime", None)
        if carry_runtime is not None:
            carry_passes[dataset] = carry_runtime.consume_completed_pass()
    observations = {}
    for dataset, texts in {
        "c4_calibration": calibration_text,
        "c4_en_512": c4_text,
        "wikitext2": wiki_text,
    }.items():
        record_hash, record_count = hash_records(texts)
        prompt = calibration_prompt if dataset == "c4_calibration" else prompts[dataset]
        token_hash, sequence_count, token_count = hash_token_sequences([prompt])
        observations[dataset] = {
            "record_count": record_count,
            "record_sha256": record_hash,
            "token_count": token_count,
            "token_sequence_count": sequence_count,
            "token_sequence_sha256": token_hash,
        }
    source_paths = [
        PROMPT_MANIFEST,
        QUALITY_CONFIG,
        ROOT / "tools" / "ace2_full_model_fixed_point.py",
        Path(__file__),
    ]
    qk_residual_contract = None
    if diagnostic_rope_mechanism in {
        "shared_qk_residual_cross_term_baseline_v1",
        "shared_qk_residual_cross_term_attention_v1",
    }:
        contract_paths = [
            ROOT / "reference" / "generated" / "qk_residual_scale32_metadata.json",
            ROOT / "reference" / "qk_residual_cross_term_full_model_hook.json",
            ROOT / "tools" / "ace2_qk_residual_cross_term_reference.py",
            ROOT / "rtl" / "ace2_qk_residual_cross_term_core.sv",
            ROOT
            / "evidence"
            / "shared_qk_residual_cross_term_attention_v1"
            / "latest"
            / "PRECHECK.json",
        ]
        source_paths.extend(contract_paths)
        layers = []
        for layer_index, layer in enumerate(model.model.layers):
            attention = layer.self_attn
            checks = dict(attention.baseline_equality_checks)
            if any(value != len(prompts) for value in checks.values()):
                raise RuntimeError(
                    f"layer {layer_index} baseline equality checks differ: {checks}"
                )
            layers.append(
                {
                    "layer": layer_index,
                    "baseline_equality_checks": checks,
                    "query_residual_positive_clamps": attention.query_residual_positive_clamps,
                    "query_residual_negative_clamps": attention.query_residual_negative_clamps,
                    "key_residual_positive_clamps": attention.key_residual_positive_clamps,
                    "key_residual_negative_clamps": attention.key_residual_negative_clamps,
                }
            )
        qk_residual_contract = {
            "contract_id": "shared_qk_residual_cross_term_attention_v1",
            "mode": (
                "candidate_three_corrections"
                if diagnostic_rope_mechanism
                == "shared_qk_residual_cross_term_attention_v1"
                else "authoritative_baseline_only"
            ),
            "layer_scope": "all_24_layers",
            "baseline_equality_boundaries": [
                "q_projection_int8",
                "k_projection_int8",
                "q_absolute_rope_int8",
                "k_absolute_rope_int8",
                "base_score_before_residual_correction",
            ],
            "all_baseline_equality_checks_passed": True,
            "layers": layers,
        }
    v_residual_contract = None
    if diagnostic_rope_mechanism in {
        "shared_v_residual_value_correction_baseline_v1",
        "shared_v_residual_value_correction_attention_v1",
    }:
        contract_paths = [
            ROOT / "reference" / "generated" / "qk_residual_scale32_metadata.json",
            ROOT / "reference" / "generated" / "v_residual_scale32_metadata.json",
            ROOT / "reference" / "v_residual_value_correction_full_model_hook.json",
            ROOT / "tools" / "ace2_v_residual_value_correction_reference.py",
            ROOT / "rtl" / "ace2_v_residual_value_correction_core.sv",
            ROOT
            / "evidence"
            / "shared_v_residual_value_correction_attention_v1"
            / "latest"
            / "PRECHECK.json",
        ]
        source_paths.extend(contract_paths)
        layers = []
        for layer_index, layer in enumerate(model.model.layers):
            attention = layer.self_attn
            checks = dict(attention.baseline_equality_checks)
            if any(value != len(prompts) for value in checks.values()):
                raise RuntimeError(
                    f"layer {layer_index} V-residual baseline equality checks differ: {checks}"
                )
            layers.append(
                {
                    "layer": layer_index,
                    "baseline_equality_checks": checks,
                    "v_residual_positive_clamps": attention.v_residual_positive_clamps,
                    "v_residual_negative_clamps": attention.v_residual_negative_clamps,
                }
            )
        v_residual_contract = {
            "contract_id": "shared_v_residual_value_correction_attention_v1",
            "mode": (
                "candidate_value_correction"
                if diagnostic_rope_mechanism
                == "shared_v_residual_value_correction_attention_v1"
                else "authoritative_baseline_value_only"
            ),
            "layer_scope": "all_24_layers",
            "baseline_equality_boundaries": [
                "q_projection_int8",
                "k_projection_int8",
                "v_projection_int8",
                "q_absolute_rope_int8",
                "k_absolute_rope_int8",
                "base_score_q20_44",
                "softmax_probability_q0_15",
            ],
            "all_baseline_equality_checks_passed": True,
            "layers": layers,
        }
    down_projection_residual_fusion_contract = None
    if diagnostic_rope_mechanism in {
        "shared_down_projection_residual_fusion_baseline_v1",
        "shared_down_projection_residual_fusion_v1",
    }:
        contract_paths = [
            ROOT / "reference" / "generated" / "down_projection_residual_fusion_metadata.json",
            ROOT / "reference" / "generated" / "down_projection_residual_fusion_metadata.bin",
            ROOT / "tools" / "ace2_down_projection_residual_fusion_hook.py",
            ROOT / "tools" / "ace2_down_projection_residual_fusion_reference.py",
            ROOT / "rtl" / "ace2_down_projection_residual_fusion_core.sv",
            ROOT
            / "evidence"
            / "shared_down_projection_residual_fusion_v1"
            / "latest"
            / "PRECHECK.json",
        ]
        source_paths.extend(contract_paths)
        if set(fusion_passes) != set(prompts):
            raise RuntimeError("down-projection fusion passes do not cover both datasets")
        for dataset, summary in fusion_passes.items():
            if summary["layers_executed"] != 24:
                raise RuntimeError(f"{dataset} fusion pass did not execute all 24 layers")
            if not summary["all_lane_independent_oracle_match"]:
                raise RuntimeError(f"{dataset} fusion pass failed independent replay")
            if summary["sticky_numeric_overflow"]:
                raise RuntimeError(f"{dataset} fusion pass overflowed")
        down_projection_residual_fusion_contract = {
            "contract_id": "shared_down_projection_residual_fusion_v1",
            "mode": (
                "composed_recurrent_candidate"
                if diagnostic_rope_mechanism
                == "shared_down_projection_residual_fusion_v1"
                else "isolated_baseline_capture_replay"
            ),
            "layer_scope": "all_24_layers",
            "all_lane_independent_oracle_match": True,
            "baseline_projection_diagnostic_not_used_by_candidate_output": True,
            "passes": fusion_passes,
        }
    cross_layer_error_carry_contract = None
    candidate_gate_passed = False
    if (
        diagnostic_rope_mechanism
        == "cross_layer_quantization_error_carry_final_output_v1"
    ):
        frozen_baseline_path = (
            ROOT
            / "evidence"
            / "cross_layer_quantization_error_carry_final_output_v1"
            / "latest"
            / "baseline_reproducibility"
            / "results.json"
        )
        baseline_acceptance_path = (
            ROOT
            / "evidence"
            / "cross_layer_quantization_error_carry_final_output_v1"
            / "latest"
            / "BASELINE_REPRODUCIBILITY.json"
        )
        contract_paths = [
            ROOT / "reference" / "generated" / "cross_layer_error_carry_metadata.json",
            ROOT / "reference" / "generated" / "cross_layer_error_carry_metadata.bin",
            ROOT / "tools" / "ace2_cross_layer_error_carry_hook.py",
            ROOT / "tools" / "ace2_cross_layer_error_carry_reference.py",
            ROOT / "rtl" / "ace2_cross_layer_error_carry_core.sv",
            ROOT
            / "evidence"
            / "cross_layer_quantization_error_carry_final_output_v1"
            / "latest"
            / "PRECHECK.json",
            baseline_acceptance_path,
            frozen_baseline_path,
        ]
        source_paths.extend(contract_paths)
        frozen_baseline = json.loads(frozen_baseline_path.read_text(encoding="utf-8"))
        expected_producers = list(range(24))
        expected_consumers = list(range(1, 25))
        pass_checks: dict[str, dict[str, bool]] = {}
        for dataset, summary in carry_passes.items():
            pass_checks[dataset] = {
                "completed": summary["completed"] is True,
                "all_24_producers": summary["producers_executed"] == 24,
                "all_24_consumers": summary["consumers_executed"] == 24,
                "producer_order_exact": summary["ordered_producer_layers"]
                == expected_producers,
                "consumer_order_exact": summary["ordered_consumer_layer_ids"]
                == expected_consumers,
                "final_output_consumed": summary["final_output_consumed"] is True,
                "carry_cleared": summary["carry_valid_after_pass"] is False,
                "no_numeric_overflow": summary["sticky_numeric_overflow"] is False,
            }
        all_layer_final_output_hook_passed = (
            candidate_execution_failure is None
            and set(carry_passes) == set(prompts)
            and all(all(checks.values()) for checks in pass_checks.values())
        )
        metric_names = [
            "model.layers.0.score",
            "model.layers.0.post_mlp_residual",
            "lm_head",
        ]
        strict_improvement: dict[str, dict[str, Any]] = {}
        for dataset in prompts:
            if dataset not in comparisons:
                continue
            strict_improvement[dataset] = {}
            for name in metric_names:
                baseline_error = frozen_baseline["comparisons"][dataset][name][
                    "relative_l2_error"
                ]
                candidate_error = comparisons[dataset][name]["relative_l2_error"]
                strict_improvement[dataset][name] = {
                    "baseline_relative_l2_error": baseline_error,
                    "candidate_relative_l2_error": candidate_error,
                    "strictly_improved": candidate_error < baseline_error,
                }
        frozen_order = frozen_baseline["boundary_order"]
        pre_injection_end = frozen_order.index("model.layers.0.down_projection")
        pre_injection_boundaries = frozen_order[: pre_injection_end + 1]
        pre_injection_exact: dict[str, Any] = {}
        for dataset in prompts:
            if dataset not in fixed_boundary_hashes:
                continue
            mismatches = [
                name
                for name in pre_injection_boundaries
                if fixed_boundary_hashes[dataset][name]
                != frozen_baseline["fixed_boundary_hashes"][dataset][name]
            ]
            pre_injection_exact[dataset] = {
                "boundary_count": len(pre_injection_boundaries),
                "exact": not mismatches,
                "mismatches": mismatches,
            }
        all_required_metrics_strictly_improved = (
            set(strict_improvement) == set(prompts)
            and all(
                metric["strictly_improved"]
                for dataset in strict_improvement.values()
                for metric in dataset.values()
            )
        )
        all_pre_injection_boundaries_exact = (
            set(pre_injection_exact) == set(prompts)
            and all(check["exact"] for check in pre_injection_exact.values())
        )
        candidate_gate_passed = (
            all_layer_final_output_hook_passed
            and all_pre_injection_boundaries_exact
            and all_required_metrics_strictly_improved
        )
        cross_layer_error_carry_contract = {
            "contract_id": "cross_layer_quantization_error_carry_final_output_v1",
            "mode": "recurrent_all_24_layers_plus_final_rmsnorm",
            "candidate_run_count": 1,
            "baseline_run_count_in_this_invocation": 0,
            "candidate_capability_accepted": False,
            "frozen_baseline": {
                "path": frozen_baseline_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(frozen_baseline_path),
                "independently_accepted": True,
            },
            "execution_failure": candidate_execution_failure,
            "all_layer_final_output_hook_passed": all_layer_final_output_hook_passed,
            "pass_checks": pass_checks,
            "pre_injection_exact": pre_injection_exact,
            "strict_improvement": strict_improvement,
            "all_required_metrics_strictly_improved": (
                all_required_metrics_strictly_improved
            ),
            "candidate_gate_passed": candidate_gate_passed,
            "passes": carry_passes,
        }
    result_boundary_order = (
        list(next(iter(comparisons.values())))
        if comparisons
        else list(next(iter(baseline.values())))
    )
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": (
            "focused_candidate_discriminator_pending_independent_review"
            if cross_layer_error_carry_contract is not None
            else "diagnostic_frozen_score_to_lm_head_trace_not_acceptance_evidence"
        ),
        "gate_passed": candidate_gate_passed,
        "candidate_capability_accepted": False,
        "material_divergence_policy": {
            "cosine_similarity_max": 0.95,
            "relative_l2_error_min": 0.25,
        },
        "first_material_divergence": first_material,
        "boundary_order": result_boundary_order,
        "comparisons": comparisons,
        "input_observations": observations,
        "model": {**model_spec, "resolved_revision": resolved_revision},
        "historical_rtl_binding": rtl_binding,
        "diagnostic_rope_mechanism": diagnostic_rope_mechanism,
        "capture_token_limit": capture_token_limit,
        "fixed_boundary_hashes": fixed_boundary_hashes,
        "qk_residual_contract": qk_residual_contract,
        "v_residual_contract": v_residual_contract,
        "down_projection_residual_fusion_contract": down_projection_residual_fusion_contract,
        "cross_layer_error_carry_contract": cross_layer_error_carry_contract,
        "candidate_execution_failure": candidate_execution_failure,
        "runtime": {
            "device": "cpu",
            "packages": versions,
            "torch": torch.__version__,
        },
        "sources": [
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
    }
    result_path = output_dir / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sha256s = write_sha256s(output_dir)
    print(
        "ACE2_SCORE_TO_LM_HEAD_LOCALIZATION "
        f"c4_first={first_material.get('c4_en_512')} "
        f"wiki_first={first_material.get('wikitext2')} "
        f"sha256s_sha256={sha256_file(sha256s)}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-rope-mechanism",
        choices=[
            "layer0_projection_shadow_staged_attention_v1",
            "layer0_tile_max_delta_attention_v1",
            "layer0_tile_bfp_score_attention_v1",
            "shared_native_accumulator_tagged_attention_v1",
            "shared_qk_residual_cross_term_baseline_v1",
            "shared_qk_residual_cross_term_attention_v1",
            "shared_v_residual_value_correction_baseline_v1",
            "shared_v_residual_value_correction_attention_v1",
            "shared_down_projection_residual_fusion_baseline_v1",
            "shared_down_projection_residual_fusion_v1",
            "cross_layer_quantization_error_carry_final_output_v1",
        ],
    )
    parser.add_argument("--capture-token-limit", type=int, default=32)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    if ROOT.resolve() not in args.output_dir.resolve().parents:
        raise SystemExit("--output-dir must be below the repository root")
    run(
        args.output_dir,
        args.diagnostic_rope_mechanism,
        args.capture_token_limit,
    )


if __name__ == "__main__":
    main()
