#!/usr/bin/env python3
"""Additive non-scoring c02 layer-0 attention-substage localization.

This tool is deliberately outside candidate and official-attempt namespaces.  It
loads the frozen BF16 sources, reconstructs the sealed c01 W4A8 model with the
already reviewed independent oracle, and observes only one fixed input.  Runtime
wrappers record layer-0 Q/K RoPE, QK score scaling, softmax, V composition,
attention-output requantization, and the o_proj input Scale32/code boundary.
No frozen implementation module is edited or replaced on disk.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import ace2_full_model_fixed_point as fixed
import diagnose_w4a8_c02_numerical_path as coarse
import run_w4a8_full_model_software_contract as frozen
import w4a8_full_model_evaluator as evaluator
from transformers.models.qwen2 import modeling_qwen2


DEFAULT_OUTPUT = (
    ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2"
)
PREDECESSOR_ROOT = (
    ROOT / "evidence/diagnostics/w4a8-c02-source-oracle-stage-trace-v2"
)
SUPERSEDED_TRACE_ROOT = (
    ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v1"
)
MODEL_ORDER = coarse.MODEL_ORDER
SOFTMAX_SCALE = 1 << fixed.SOFTMAX_PROB_FRAC
SCORE_SCALE = 1 << fixed.ATTENTION_SCORE_FRAC


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_copy(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous()


def last_query_heads(value: torch.Tensor) -> torch.Tensor:
    require(value.ndim == 4, "attention tensor must be rank 4")
    return tensor_copy(value[:, :, -1, :])


def last_query_sequence(value: torch.Tensor) -> torch.Tensor:
    require(value.ndim == 4, "score/probability tensor must be rank 4")
    return tensor_copy(value[:, :, -1:, :])


def last_token(value: torch.Tensor) -> torch.Tensor:
    require(value.ndim == 3, "projection input must be rank 3")
    return tensor_copy(value[:, -1, :])


def decoded_scale32(records: torch.Tensor | int) -> torch.Tensor:
    value = (
        torch.tensor(records, dtype=torch.int64)
        if isinstance(records, int)
        else records.detach().to(torch.int64).cpu()
    )
    return coarse._scale32_values(value)


def stable_round_even_int8(value: torch.Tensor) -> torch.Tensor:
    return torch.round(value.to(torch.float64)).clamp(-128, 127).to(torch.int8)


def metric_record(
    stage: str,
    boundary_class: str,
    reference: torch.Tensor,
    observed: torch.Tensor,
    *,
    raw_int8: torch.Tensor | None = None,
    local_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "boundary_class": boundary_class,
        "metrics": coarse.numerical_metrics(
            reference, observed, raw_int8=raw_int8
        ),
        "stage": stage,
    }
    if local_metrics is not None:
        record["local_realization_metrics"] = local_metrics
    return record


def capture_bf16_layer0(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    captures: dict[str, torch.Tensor] = {}
    original_rope = modeling_qwen2.apply_rotary_pos_emb
    original_eager = modeling_qwen2.eager_attention_forward
    rope_calls = 0
    attention_calls = 0

    def rope_wrapper(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal rope_calls
        result = original_rope(q, k, cos, sin, unsqueeze_dim)
        if rope_calls == 0:
            captures["q_rope"] = tensor_copy(result[0])
            captures["k_rope"] = tensor_copy(result[1])
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
        result = original_eager(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
        if attention_calls == 0:
            mapped_key = modeling_qwen2.repeat_kv(key, module.num_key_value_groups)
            mapped_value = modeling_qwen2.repeat_kv(
                value, module.num_key_value_groups
            )
            scaled_scores = torch.matmul(
                query, mapped_key.transpose(2, 3)
            ) * scaling
            masked_scores = scaled_scores
            if attention_mask is not None:
                masked_scores = masked_scores + attention_mask[
                    :, :, :, : mapped_key.shape[-2]
                ]
            centered_scores = masked_scores - masked_scores.amax(
                dim=-1, keepdim=True
            )
            captures["qk_scaled_scores"] = tensor_copy(scaled_scores)
            captures["qk_centered_scores"] = tensor_copy(centered_scores)
            captures["probabilities"] = tensor_copy(result[1])
            captures["mapped_value"] = tensor_copy(mapped_value)
            captures["attention_composition"] = tensor_copy(result[0])
        attention_calls += 1
        return result

    def o_proj_input_hook(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        captures["o_proj_input"] = tensor_copy(inputs[0])

    hook = model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(
        o_proj_input_hook
    )
    modeling_qwen2.apply_rotary_pos_emb = rope_wrapper
    modeling_qwen2.eager_attention_forward = eager_wrapper
    try:
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
    finally:
        hook.remove()
        modeling_qwen2.apply_rotary_pos_emb = original_rope
        modeling_qwen2.eager_attention_forward = original_eager
    require(rope_calls >= 1, "BF16 layer-0 RoPE was not observed")
    require(attention_calls >= 1, "BF16 layer-0 eager attention was not observed")
    require(
        torch.equal(
            captures["attention_composition"].reshape_as(captures["o_proj_input"]),
            captures["o_proj_input"],
        ),
        "BF16 attention composition and o_proj input differ",
    )
    return captures


def capture_w4_layer0(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    captures: dict[str, torch.Tensor] = {}
    attention = model.model.layers[0].self_attn
    require(isinstance(attention, fixed.FixedAttention), "layer-0 attention is not fixed")
    require(attention.dynamic_rope_head_scale, "layer-0 dynamic RoPE is not active")
    evaluator.enable_w4a8_kv_cache(model)

    original_rope = fixed.dynamic_rope_head_raw
    original_scores = fixed.fixed_dynamic_attention_scores_raw
    original_softmax = fixed.fixed_softmax_raw
    original_value = fixed.fixed_attention_value_raw
    original_o_accumulator = attention.o_proj.accumulator_quantized
    rope_calls = 0
    score_calls = 0
    softmax_calls = 0
    value_calls = 0

    def rope_wrapper(
        activations: torch.Tensor,
        producer_scale32: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        nonlocal rope_calls
        result = original_rope(activations, producer_scale32, cos, sin)
        if rope_calls < 2:
            prefix = "q" if rope_calls == 0 else "k"
            captures[f"{prefix}_rope_input_s8"] = tensor_copy(activations)
            captures[f"{prefix}_rope_output_s8"] = tensor_copy(result[0])
            captures[f"{prefix}_rope_scale32"] = tensor_copy(result[1])
        rope_calls += 1
        return result

    def score_wrapper(
        query: torch.Tensor,
        key: torch.Tensor,
        query_scale32: torch.Tensor,
        key_scale32: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        nonlocal score_calls
        result = original_scores(
            query, key, query_scale32, key_scale32, attention_mask
        )
        if score_calls == 0:
            captures["score_query_s8"] = tensor_copy(query)
            captures["score_key_s8"] = tensor_copy(key)
            captures["score_query_scale32"] = tensor_copy(query_scale32)
            captures["score_key_scale32"] = tensor_copy(key_scale32)
            captures["scores_q6_9"] = tensor_copy(result)
        score_calls += 1
        return result

    def softmax_wrapper(scores: torch.Tensor) -> torch.Tensor:
        nonlocal softmax_calls
        result = original_softmax(scores)
        if softmax_calls == 0:
            captures["softmax_input_q6_9"] = tensor_copy(scores)
            captures["probabilities_q0_15"] = tensor_copy(result)
        softmax_calls += 1
        return result

    def value_wrapper(
        probabilities: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal value_calls
        result = original_value(probabilities, values)
        if value_calls == 0:
            accumulator = torch.matmul(
                probabilities.to(torch.int64), values.to(torch.int64)
            )
            captures["composition_accumulator"] = tensor_copy(accumulator)
            captures["mapped_value_s8"] = tensor_copy(values)
            captures["attention_output_s8"] = tensor_copy(result)
        value_calls += 1
        return result

    def o_accumulator_wrapper(
        self: fixed.W4A8Linear,
        qinput: torch.Tensor,
    ) -> torch.Tensor:
        captures["o_proj_input_s8"] = tensor_copy(qinput)
        return original_o_accumulator(qinput)

    fixed.dynamic_rope_head_raw = rope_wrapper
    fixed.fixed_dynamic_attention_scores_raw = score_wrapper
    fixed.fixed_softmax_raw = softmax_wrapper
    fixed.fixed_attention_value_raw = value_wrapper
    attention.o_proj.accumulator_quantized = types.MethodType(
        o_accumulator_wrapper, attention.o_proj
    )
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
        logits_raw = tensor_copy(model.lm_head.last_raw_output[:, -1, :])
        require(output.logits.shape[-1] == logits_raw.shape[-1], "logit shape differs")
    finally:
        fixed.dynamic_rope_head_raw = original_rope
        fixed.fixed_dynamic_attention_scores_raw = original_scores
        fixed.fixed_softmax_raw = original_softmax
        fixed.fixed_attention_value_raw = original_value
        attention.o_proj.accumulator_quantized = original_o_accumulator

    require(rope_calls >= 2, "W4 layer-0 Q/K RoPE was not observed")
    require(score_calls >= 1, "W4 layer-0 score conversion was not observed")
    require(softmax_calls >= 1, "W4 layer-0 softmax was not observed")
    require(value_calls >= 1, "W4 layer-0 value composition was not observed")
    expected_o_input = captures["attention_output_s8"].transpose(1, 2).reshape(
        input_ids.shape[0], input_ids.shape[1], -1
    )
    require(
        torch.equal(expected_o_input, captures["o_proj_input_s8"]),
        "W4 attention output and o_proj input codes differ",
    )
    return captures, logits_raw


def build_substage_result(
    model: torch.nn.Module,
    bf16: dict[str, torch.Tensor],
    w4: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attention = model.model.layers[0].self_attn
    q_scale = decoded_scale32(w4["q_rope_scale32"]).unsqueeze(-1)
    k_scale = decoded_scale32(w4["k_rope_scale32"]).unsqueeze(-1)
    q_dequant = w4["q_rope_output_s8"].to(torch.float64) * q_scale
    k_dequant = w4["k_rope_output_s8"].to(torch.float64) * k_scale

    score_q_scale = decoded_scale32(w4["score_query_scale32"]).unsqueeze(-1)
    score_k_scale = decoded_scale32(w4["score_key_scale32"]).unsqueeze(-1)
    q_score_dequant = w4["score_query_s8"].to(torch.float64) * score_q_scale
    k_score_dequant = w4["score_key_s8"].to(torch.float64) * score_k_scale
    physical_scores = torch.matmul(
        q_score_dequant, k_score_dequant.transpose(2, 3)
    ) / math.sqrt(attention.head_dim)
    physical_centered = physical_scores - physical_scores.amax(
        dim=-1, keepdim=True
    )
    realized_scores = w4["scores_q6_9"].to(torch.float64) / SCORE_SCALE
    probabilities = w4["probabilities_q0_15"].to(torch.float64) / SOFTMAX_SCALE
    ideal_realized_probabilities = torch.softmax(
        realized_scores.to(torch.float32), dim=-1
    ).to(torch.float64)

    value_scale_record = attention.v_proj.output_scale32_record
    require(value_scale_record is not None, "V projection Scale32 record is missing")
    value_scale = float(decoded_scale32(value_scale_record))
    composition_pre_requant = (
        w4["composition_accumulator"].to(torch.float64)
        * value_scale
        / SOFTMAX_SCALE
    )
    attention_output = w4["attention_output_s8"].to(torch.float64) * value_scale
    o_proj_input = w4["o_proj_input_s8"].to(torch.float64) * float(
        attention.o_proj.hardware_input_scale
    )

    bf16_q = last_query_heads(bf16["q_rope"])
    bf16_k = tensor_copy(bf16["k_rope"])
    bf16_scores = last_query_sequence(bf16["qk_scaled_scores"])
    bf16_centered = last_query_sequence(bf16["qk_centered_scores"])
    bf16_probabilities = last_query_sequence(bf16["probabilities"])
    require(
        bf16["attention_composition"].ndim == 4,
        "BF16 attention composition must be [batch, sequence, heads, lanes]",
    )
    bf16_composition = tensor_copy(bf16["attention_composition"][:, -1, :, :])
    bf16_o_input = last_token(bf16["o_proj_input"])

    local_score_metrics = coarse.numerical_metrics(
        last_query_sequence(physical_centered),
        last_query_sequence(realized_scores),
    )
    score_induced_probability_metrics = coarse.numerical_metrics(
        bf16_probabilities,
        last_query_sequence(ideal_realized_probabilities),
    )
    local_softmax_metrics = coarse.numerical_metrics(
        last_query_sequence(ideal_realized_probabilities),
        last_query_sequence(probabilities),
    )
    local_requant_metrics = coarse.numerical_metrics(
        last_query_heads(composition_pre_requant),
        last_query_heads(attention_output),
        raw_int8=last_query_heads(w4["attention_output_s8"]),
    )
    stages = [
        metric_record(
            "model.layers.0.self_attn.q_rope",
            "qk_rope",
            bf16_q,
            last_query_heads(q_dequant),
            raw_int8=last_query_heads(w4["q_rope_output_s8"]),
        ),
        metric_record(
            "model.layers.0.self_attn.k_rope",
            "qk_rope",
            bf16_k,
            k_dequant,
            raw_int8=w4["k_rope_output_s8"],
        ),
        metric_record(
            "model.layers.0.self_attn.qk_scaled_dot",
            "qk_score_scaling",
            bf16_scores,
            last_query_sequence(physical_scores),
        ),
        metric_record(
            "model.layers.0.self_attn.qk_score_q6_9_realization",
            "qk_score_scaling",
            bf16_centered,
            last_query_sequence(realized_scores),
            local_metrics=local_score_metrics,
        ),
        metric_record(
            "model.layers.0.self_attn.softmax_probabilities",
            "softmax",
            bf16_probabilities,
            last_query_sequence(probabilities),
            local_metrics=local_softmax_metrics,
        ),
        metric_record(
            "model.layers.0.self_attn.v_weighted_composition",
            "attention_value_composition",
            bf16_composition,
            last_query_heads(composition_pre_requant),
        ),
        metric_record(
            "model.layers.0.self_attn.attention_output_requantization",
            "attention_output_requantization",
            bf16_composition,
            last_query_heads(attention_output),
            raw_int8=last_query_heads(w4["attention_output_s8"]),
            local_metrics=local_requant_metrics,
        ),
        metric_record(
            "model.layers.0.self_attn.o_proj_input_scale32_activation_coding",
            "o_proj_input_scale32",
            bf16_o_input,
            last_token(o_proj_input),
            raw_int8=last_token(w4["o_proj_input_s8"]),
        ),
    ]
    for ordinal, stage in enumerate(stages):
        stage["ordinal"] = ordinal

    producer_scale = float(attention.v_proj.output_scale)
    consumer_scale = float(attention.o_proj.hardware_input_scale)
    bf16_target_codes = stable_round_even_int8(bf16["o_proj_input"] / consumer_scale)
    actual_codes = w4["o_proj_input_s8"]
    bf16_centered_last = last_query_sequence(bf16["qk_centered_scores"])
    realized_scores_last = last_query_sequence(realized_scores)
    bf16_score_top = torch.argmax(bf16_centered_last, dim=-1)
    realized_score_top = torch.argmax(realized_scores_last, dim=-1)
    bf16_sorted = torch.sort(
        bf16_centered_last.to(torch.float64), dim=-1, descending=True
    ).values
    realized_sorted = torch.sort(
        realized_scores_last.to(torch.float64), dim=-1, descending=True
    ).values
    invariants = {
        "attention_output_code_equals_o_proj_input_code": torch.equal(
            w4["attention_output_s8"].transpose(1, 2).reshape_as(actual_codes),
            actual_codes,
        ),
        "bf16_nearest_code_disagreement_fraction": float(
            (bf16_target_codes != actual_codes).to(torch.float64).mean()
        ),
        "consumer_hardware_input_scale": consumer_scale,
        "consumer_scale_exactly_equals_decoded_scale32": consumer_scale
        == value_scale,
        "consumer_scale_exactly_equals_v_projection_output_scale": consumer_scale
        == producer_scale,
        "producer_scale32_decoded": value_scale,
        "producer_scale32_record": int(value_scale_record),
        "producer_v_projection_output_scale": producer_scale,
        "score_rank_margin": {
            "bf16_realized_top_index_equal_fraction": float(
                (bf16_score_top == realized_score_top).to(torch.float64).mean()
            ),
            "bf16_top2_margin_max": float((bf16_sorted[..., 0] - bf16_sorted[..., 1]).max()),
            "bf16_top2_margin_mean": float((bf16_sorted[..., 0] - bf16_sorted[..., 1]).mean()),
            "bf16_top2_margin_min": float((bf16_sorted[..., 0] - bf16_sorted[..., 1]).min()),
            "realized_top2_margin_max": float(
                (realized_sorted[..., 0] - realized_sorted[..., 1]).max()
            ),
            "realized_top2_margin_mean": float(
                (realized_sorted[..., 0] - realized_sorted[..., 1]).mean()
            ),
            "realized_top2_margin_min": float(
                (realized_sorted[..., 0] - realized_sorted[..., 1]).min()
            ),
        },
        "qk_score_realization_local_metrics": local_score_metrics,
        "requantization_local_metrics": local_requant_metrics,
        "score_perturbation_probability_metrics": score_induced_probability_metrics,
        "softmax_realization_local_metrics": local_softmax_metrics,
    }
    return stages, invariants


def classify_model(stages: list[dict[str, Any]]) -> dict[str, Any]:
    first_material = next(
        (stage for stage in stages if stage["metrics"]["severity"] != "TRACKING"),
        None,
    )
    first_severe = next(
        (stage for stage in stages if stage["metrics"]["severity"] == "SEVERE"),
        None,
    )
    return {
        "earliest_material_primitive": (
            first_material["stage"] if first_material is not None else None
        ),
        "first_material_stage": first_material,
        "first_severe_stage": first_severe,
    }


def shared_classification(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    earliest = {
        alias: result["classification"]["earliest_material_primitive"]
        for alias, result in results.items()
    }
    same_primitive = len(set(earliest.values())) == 1
    scale_chain_exact = all(
        result["scale_alignment_invariants"][
            "consumer_scale_exactly_equals_decoded_scale32"
        ]
        and result["scale_alignment_invariants"][
            "consumer_scale_exactly_equals_v_projection_output_scale"
        ]
        and result["scale_alignment_invariants"][
            "attention_output_code_equals_o_proj_input_code"
        ]
        for result in results.values()
    )
    primitive = next(iter(earliest.values())) if same_primitive else None
    softmax_local_material = all(
        result["scale_alignment_invariants"]["softmax_realization_local_metrics"][
            "severity"
        ]
        != "TRACKING"
        for result in results.values()
    )
    score_perturbation_material = all(
        result["scale_alignment_invariants"][
            "score_perturbation_probability_metrics"
        ]["severity"]
        != "TRACKING"
        for result in results.values()
    )
    if same_primitive and primitive is not None:
        if primitive.endswith("q_rope") or primitive.endswith("k_rope"):
            invariant = "DYNAMIC_QK_ROPE_SCALE32_CODE_REALIZATION"
        elif "qk_" in primitive:
            invariant = "QK_SCALE32_TO_CENTERED_Q6_9_SCORE_ALIGNMENT"
        elif "softmax" in primitive and softmax_local_material:
            invariant = "CENTERED_Q6_9_TO_Q0_15_SOFTMAX_REALIZATION"
        elif "softmax" in primitive and score_perturbation_material:
            invariant = "QK_SCORE_RANK_MARGIN_PRESERVATION_BEFORE_SOFTMAX"
        elif "v_weighted" in primitive:
            invariant = "Q0_15_BY_V_SCALE32_ATTENTION_COMPOSITION"
        else:
            invariant = "V_SCALE32_ATTENTION_OUTPUT_TO_O_PROJ_INPUT_ALIGNMENT"
    else:
        invariant = None
    return {
        "earliest_primitives": earliest,
        "one_shared_earliest_primitive": same_primitive,
        "o_proj_scale_and_code_chain_exact_for_both_models": scale_chain_exact,
        "one_shared_scale_alignment_violation_explains_both_models": bool(
            invariant
            and invariant
            in {
                "DYNAMIC_QK_ROPE_SCALE32_CODE_REALIZATION",
                "QK_SCALE32_TO_CENTERED_Q6_9_SCORE_ALIGNMENT",
                "CENTERED_Q6_9_TO_Q0_15_SOFTMAX_REALIZATION",
                "V_SCALE32_ATTENTION_OUTPUT_TO_O_PROJ_INPUT_ALIGNMENT",
            }
        ),
        "score_perturbation_is_material_for_both_models": score_perturbation_material,
        "shared_repair_invariant": invariant,
        "softmax_realization_is_material_for_both_models": softmax_local_material,
        "status": (
            "PASS_SHARED_PRIMITIVE_LOCALIZED"
            if same_primitive and primitive is not None
            else "BOUND_REMAINS_MODEL_SPECIFIC_OR_UNRESOLVED"
        ),
    }


def run_model(
    alias: str,
    contract: dict[str, Any],
    input_ids: torch.Tensor,
    fixed_input: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    sealed_root = coarse.SEALED_ROOTS[alias]
    before_hashes = {
        path.name: sha256_file(path)
        for path in sorted(sealed_root.iterdir())
        if path.is_file()
    }
    spec = frozen.model_spec(contract, alias)
    frozen.verify_local_model_inputs(contract, spec)
    model = frozen.load_model(spec)
    bf16 = capture_bf16_layer0(model, input_ids)
    source_reconstruction, _ = coarse.reconstruct_model_from_source_oracle(
        model, sealed_root, contract
    )
    require(
        source_reconstruction["status"]
        == "PASS_EXACT_FROZEN_SOURCE_RECONSTRUCTION",
        f"source reconstruction failed for {alias}",
    )
    w4, logits_raw = capture_w4_layer0(model, input_ids)
    sealed_step = fixed_input["sealed_first_step"][alias]
    require(
        coarse.tensor_sha256(logits_raw)
        == sealed_step["first_integer_logit_sha256"],
        f"sealed first integer-logit hash was not reproduced for {alias}",
    )
    require(
        int(torch.argmax(logits_raw)) == sealed_step["first_generated_token_id"],
        f"sealed first token was not reproduced for {alias}",
    )
    stages, invariants = build_substage_result(model, bf16, w4)
    classification = classify_model(stages)
    predecessor = coarse.load_json(PREDECESSOR_ROOT / alias / "result.json")
    result = {
        "alias": alias,
        "classification": classification,
        "fixed_input": fixed_input,
        "model_identity_sha256": coarse.canonical_sha256(spec["contract_identity"]),
        "predecessor_downstream_classification": {
            "first_severe_stage": predecessor["boundary_classification"][
                "first_severe_stage"
            ],
            "result_sha256": sha256_file(PREDECESSOR_ROOT / alias / "result.json"),
        },
        "prohibitions_observed": {
            "benchmark_scoring_performed": False,
            "candidate_bundle_created": False,
            "candidate_or_official_namespace_created": False,
            "continuation_generation_performed": False,
            "execute_command_invoked": False,
            "frozen_kernel_evaluator_runner_mutated": False,
            "retraining_performed": False,
            "sealed_c00_or_c01_mutated": False,
        },
        "scale_alignment_invariants": invariants,
        "schema_version": 1,
        "source_reconstruction": source_reconstruction,
        "stage_comparisons": stages,
        "status": "PASS_ATTENTION_SUBSTAGE_LOCALIZED_NON_SCORING",
    }
    model_root = output_root / alias
    model_root.mkdir(parents=True, exist_ok=False)
    tensor_bundle = {}
    for prefix, tensors in (("bf16", bf16), ("w4", w4)):
        for name, tensor in tensors.items():
            tensor_bundle[f"{prefix}.{name}"] = tensor
    result["tensor_bundle"] = coarse.write_tensor_bundle(
        model_root / "attention-substage-tensors.bin",
        tensor_bundle,
        public_path=f"{alias}/attention-substage-tensors.bin",
    )
    (model_root / "result.json").write_bytes(canonical_bytes(result))
    after_hashes = {
        path.name: sha256_file(path)
        for path in sorted(sealed_root.iterdir())
        if path.is_file()
    }
    require(before_hashes == after_hashes, f"sealed c01 artifacts changed for {alias}")
    del model
    gc.collect()
    return result


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
    aggregate = coarse.load_json(output_root / "result.json")
    require(
        aggregate["status"] == "PASS_C02_ATTENTION_SUBSTAGE_LOCALIZATION",
        "aggregate status differs",
    )
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
    require(
        not {"candidate", "attempt"}.intersection(output_root.parts),
        "candidate/attempt namespaces are forbidden",
    )
    if args.verify_existing:
        return verify_existing(output_root)
    require(not output_root.exists(), f"diagnostic output already exists: {output_root}")
    require(1 <= args.threads <= 32, "thread count must be in 1..32")
    require(PREDECESSOR_ROOT.is_dir(), "accepted predecessor evidence is missing")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    contract, contract_sha256 = frozen.load_contract()
    input_ids, fixed_input = coarse.sealed_fixed_input()
    output_root.mkdir(parents=True)
    try:
        results = {
            alias: run_model(alias, contract, input_ids, fixed_input, output_root)
            for alias in MODEL_ORDER
        }
        shared = shared_classification(results)
        aggregate = {
            "contract_sha256": contract_sha256,
            "models": {
                alias: {
                    "classification": result["classification"],
                    "result_path": f"{alias}/result.json",
                    "result_sha256": sha256_file(output_root / alias / "result.json"),
                    "status": result["status"],
                }
                for alias, result in results.items()
            },
            "predecessor": {
                "path": PREDECESSOR_ROOT.relative_to(ROOT).as_posix(),
                "result_sha256": sha256_file(PREDECESSOR_ROOT / "result.json"),
                "status": coarse.load_json(PREDECESSOR_ROOT / "result.json")["status"],
            },
            "superseded_trace": {
                "disposition": (
                    "PRESERVED_NON_ACCEPTED_INCOMPLETE_SOFTMAX_CAUSAL_DISCRIMINATION"
                ),
                "path": SUPERSEDED_TRACE_ROOT.relative_to(ROOT).as_posix(),
                "result_sha256": sha256_file(SUPERSEDED_TRACE_ROOT / "result.json"),
            },
            "prohibitions_observed": {
                "benchmark_scoring_performed": False,
                "candidate_bundle_created": False,
                "candidate_or_official_namespace_created": False,
                "continuation_generation_performed": False,
                "execute_command_invoked": False,
                "retraining_performed": False,
                "rtl_spec_manifest_or_u280_changed": False,
            },
            "schema_version": 1,
            "shared_classification": shared,
            "status": "PASS_C02_ATTENTION_SUBSTAGE_LOCALIZATION",
        }
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
