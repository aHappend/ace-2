#!/usr/bin/env python3
"""Run the canonical LoRA V4 token-0 layer through ACE-2 W4A8 RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from ace2_attention_compose_reference import AttentionComposeCase, reference_attention_compose
from ace2_attention_score_reference import AttentionScoreCase, reference_attention_score
from ace2_projection_reference import ProjectionCase, reference_projection
from ace2_residual_reference import reference_residual_add
from ace2_rmsnorm_reference import (
    derive_rmsnorm_output_scale,
    derive_scaled_gains_q8,
    pack_gain_beats,
    pack_int8_beats,
    reference_rmsnorm,
)
from ace2_rope_reference import Q15_ONE, Q9_SCALE_ONE, RopeCase, reference_rope
from ace2_silu_gate_reference import SiluGateCase, reference_silu_gate
from ace2_softmax_reference import SoftmaxCase, reference_softmax


MISSION = "canonical_lora_v4_layer0_full_w4a8_rtl"
OUT = ROOT / "evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1"
CONTRACT = OUT / "frozen_contract.json"
ATTEMPT = OUT / "attempt-0001"
SNAPSHOT = Path(
    "/home/argustest/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
MODEL = SNAPSHOT / "model.safetensors"
CONFIG = SNAPSHOT / "config.json"
ADAPTER_DIR = (
    ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/attempt-0001/"
    "checkpoints/checkpoint-176"
)
ADAPTER = ADAPTER_DIR / "adapter_model.safetensors"
ADAPTER_CONFIG = ADAPTER_DIR / "adapter_config.json"
CHECKPOINT_IDENTITY = (
    ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/attempt-0001/"
    "probe-selection/epoch-4/checkpoint_identity.json"
)
ERRATUM = (
    ROOT
    / "evidence/verification/lora-v4-layer0-qproj-w4a8-rtl-v1/errata/"
    "attempt-0002-manifest-label-v1/erratum.json"
)
ERRATUM_SUMS = ERRATUM.parent / "SHA256SUMS"

PROJECTION_TB = ROOT / "verification/tb/ace2_lora_v4_layer0_projection_batch_tb.sv"
MAINTAINED_TBS = {
    "rmsnorm": ROOT / "verification/tb/ace2_rmsnorm_tb.sv",
    "rope": ROOT / "verification/tb/ace2_rope_tb.sv",
    "attention_score": ROOT / "verification/tb/ace2_attention_score_tb.sv",
    "softmax": ROOT / "verification/tb/ace2_softmax_tb.sv",
    "attention_compose": ROOT / "verification/tb/ace2_attention_compose_tb.sv",
    "silu": ROOT / "verification/tb/ace2_silu_gate_tb.sv",
    "residual": ROOT / "verification/tb/ace2_shell_tb.sv",
}
CORES = {
    "projection": ROOT / "rtl/ace2_w4a8_proj_core.sv",
    "rmsnorm": ROOT / "rtl/ace2_rmsnorm_core.sv",
    "rope": ROOT / "rtl/ace2_rope_core.sv",
    "attention_score": ROOT / "rtl/ace2_attention_score_core.sv",
    "softmax": ROOT / "rtl/ace2_softmax_core.sv",
    "attention_compose": ROOT / "rtl/ace2_attention_compose_core.sv",
    "silu": ROOT / "rtl/ace2_silu_gate_core.sv",
}

MODEL_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
ADAPTER_SHA256 = "c9476d6ccc687a4d68de1951eb889b689daaf9c5d3822b4c5acfc26a1cd38a28"
CHECKPOINT_TREE_SHA256 = "8cee9b9343e49b66ee2ea427578f87335ffa259919356cdd119cd8712ff7eeee"
PROMPT = "ACE-2 W4A8 layer-zero checkpoint adaptation."
TOKEN_INDEX = 0
HIDDEN = 896
INTERMEDIATE = 4864
Q_HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64
LORA_RANK = 16
LORA_ALPHA = 32
RMS_EPSILON = 1e-6

PROJECTION_ORDER = ["q", "k", "v", "o", "gate", "up", "down"]
OPERATOR_ORDER = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
    "layer_0.rope_q",
    "layer_0.rope_k",
    "layer_0.attention_score",
    "layer_0.softmax",
    "layer_0.attention_value",
    "layer_0.o_proj",
    "layer_0.attention_residual_add",
    "layer_0.post_attention_rmsnorm",
    "layer_0.mlp_gate_proj",
    "layer_0.mlp_up_proj",
    "layer_0.silu_gate",
    "layer_0.mlp_down_proj",
    "layer_0.mlp_residual_add",
]

THRESHOLDS = {
    "rtl_vs_python_fixed_point": {
        "operator_output_mismatches_max": 0,
        "projection_accumulator_mismatches_max": 0,
        "layer_output_integer_mismatches_max": 0,
    },
    "merged_float_vs_w4a8_layer_output": {
        "max_abs_dequantized_error_max": 0.25,
        "relative_l2_error_max": 0.40,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().contiguous().cpu()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().tobytes(order="C")
    return tensor.numpy().tobytes(order="C")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def public_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "path": public_path(path), "sha256": sha256_file(path)}


def tree_record(pattern: str) -> dict[str, Any]:
    files = [file_record(path) for path in sorted(ROOT.glob(pattern)) if path.is_file()]
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"file_count": len(files), "sha256": sha256_bytes(canonical), "files": files}


def write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file_record(path)


def write_text(path: Path, value: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return file_record(path)


def write_binary(path: Path, raw: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return file_record(path)


def write_hex(path: Path, values: Iterable[int], width: int) -> dict[str, Any]:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    return write_text(path, "".join(f"{int(value) & mask:0{digits}x}\n" for value in values))


def write_sha256s(base: Path) -> None:
    rows = []
    for path in sorted(item for item in base.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        rows.append(f"{sha256_file(path)}  {path.relative_to(base).as_posix()}\n")
    (base / "SHA256SUMS").write_text("".join(rows), encoding="ascii")


def verify_sha256s(base: Path) -> int:
    count = 0
    for row in (base / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        expected, name = row.split("  ", 1)
        path = base / name
        require(path.is_file(), f"manifest member missing: {name}")
        require(sha256_file(path) == expected, f"manifest member changed: {name}")
        count += 1
    return count


def verify_external_sums(path: Path) -> int:
    count = 0
    for row in path.read_text(encoding="ascii").splitlines():
        expected, name = row.split("  ", 1)
        candidate = path.parent / name
        require(candidate.is_file(), f"external binding missing: {candidate}")
        require(sha256_file(candidate) == expected, f"external binding changed: {candidate}")
        count += 1
    return count


def tokenizer_ids() -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    encoded = tokenizer(PROMPT, add_special_tokens=False, return_attention_mask=False)
    return [int(value) for value in encoded["input_ids"]]


def validate_provenance() -> dict[str, Any]:
    required = [MODEL, CONFIG, ADAPTER, ADAPTER_CONFIG, CHECKPOINT_IDENTITY, ERRATUM, ERRATUM_SUMS, PROJECTION_TB, *MAINTAINED_TBS.values(), *CORES.values()]
    for path in required:
        require(path.is_file(), f"required local artifact missing: {path}")
    require(sha256_file(MODEL) == MODEL_SHA256, "base model hash changed")
    require(sha256_file(ADAPTER) == ADAPTER_SHA256, "adapter hash changed")
    identity = json.loads(CHECKPOINT_IDENTITY.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    adapter_config = json.loads(ADAPTER_CONFIG.read_text(encoding="utf-8"))
    require(identity["tree_sha256"] == CHECKPOINT_TREE_SHA256, "checkpoint tree changed")
    require(config["hidden_size"] == HIDDEN and config["intermediate_size"] == INTERMEDIATE, "geometry changed")
    require(config["num_attention_heads"] == Q_HEADS and config["num_key_value_heads"] == KV_HEADS, "attention geometry changed")
    require(float(config["rms_norm_eps"]) == RMS_EPSILON, "RMS epsilon changed")
    require(adapter_config["r"] == LORA_RANK and adapter_config["lora_alpha"] == LORA_ALPHA, "LoRA scaling changed")
    require(verify_external_sums(ERRATUM_SUMS) == 7, "manifest erratum member count changed")
    ids = tokenizer_ids()
    require(ids and TOKEN_INDEX < len(ids), "fixed tokenizer input changed")
    return {"config": config, "adapter_config": adapter_config, "token_ids": ids}


def round_shift_even(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    base = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (base & 1)):
        base += 1
    return sign * base


def derive_multiplier(real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    real = real.detach().to(torch.float64)
    require(bool(torch.all(torch.isfinite(real))) and not bool(torch.any(real < 0)), "invalid multiplier")
    multiplier = torch.zeros_like(real, dtype=torch.int64)
    right_shift = torch.full_like(real, -1, dtype=torch.int64)
    for shift in range(63, -1, -1):
        candidate = torch.round(real * math.ldexp(1.0, shift))
        selected = (right_shift < 0) & (candidate <= (1 << 31) - 1)
        multiplier = torch.where(selected, candidate.to(torch.int64), multiplier)
        right_shift = torch.where(selected, torch.full_like(right_shift, shift), right_shift)
    require(not bool(torch.any(right_shift < 0)), "unrepresentable multiplier")
    return multiplier, right_shift


def quantize_int8(values: torch.Tensor, scale: float) -> torch.Tensor:
    require(math.isfinite(scale) and scale > 0.0, "invalid int8 scale")
    return torch.round(values.to(torch.float64) / scale).clamp(-128, 127).to(torch.int8)


def scale_for(values: torch.Tensor) -> float:
    maximum = float(values.to(torch.float64).abs().max().item())
    return max(maximum / 127.0, 1e-12)


def float_rmsnorm(values: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    source = values.to(torch.bfloat16).to(torch.float32)
    normalized = (source * torch.rsqrt(source.square().mean() + RMS_EPSILON)).to(torch.bfloat16)
    return (gain * normalized).to(torch.float32)


def merge_projection(weights: Any, adapter: Any, tensor_name: str) -> tuple[torch.Tensor, dict[str, str]]:
    base = weights.get_tensor(tensor_name + ".weight").contiguous()
    prefix = "base_model.model." + tensor_name
    lora_a = adapter.get_tensor(prefix + ".lora_A.weight").contiguous()
    lora_b = adapter.get_tensor(prefix + ".lora_B.weight").contiguous()
    merged = (
        base.to(torch.float32)
        + (float(LORA_ALPHA) / float(LORA_RANK))
        * torch.matmul(lora_b.to(torch.float32), lora_a.to(torch.float32))
    ).contiguous()
    return merged, {
        "base_bf16": sha256_bytes(raw_tensor_bytes(base)),
        "lora_a": sha256_bytes(raw_tensor_bytes(lora_a)),
        "lora_b": sha256_bytes(raw_tensor_bytes(lora_b)),
        "merged_f32": sha256_bytes(raw_tensor_bytes(merged)),
    }


def derive_projection(
    name: str,
    merged: torch.Tensor,
    input_q: torch.Tensor,
    input_scale: float,
    float_input: torch.Tensor,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    float_output = torch.mv(merged, float_input.to(torch.float32)).contiguous()
    output_scale = scale_for(float_output)
    weight = merged.to(torch.float64)
    weight_scale = weight.abs().amax(dim=1) / 7.0
    weight_scale = torch.where(weight_scale > 0, weight_scale, torch.ones_like(weight_scale))
    qweight = torch.round(weight / weight_scale[:, None]).clamp(-8, 7).to(torch.int8)
    multiplier, right_shift = derive_multiplier(input_scale * weight_scale / output_scale)
    accumulator = torch.mv(qweight.to(torch.int64), input_q.to(torch.int64))
    require(bool(torch.all(accumulator >= -(1 << 31))) and bool(torch.all(accumulator < (1 << 31))), f"{name} accumulator overflow")
    rounded = torch.tensor(
        [
            round_shift_even(int(acc) * int(mult), int(shift))
            for acc, mult, shift in zip(
                accumulator.tolist(), multiplier.tolist(), right_shift.tolist(), strict=True
            )
        ],
        dtype=torch.int64,
    )
    output_q = rounded.clamp(-128, 127).to(torch.int8)
    saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
    independent = reference_projection(
        ProjectionCase(
            name=name,
            rows=1,
            reduction_size=input_q.numel(),
            activations=[input_q.to(torch.int64).tolist()],
            weights=qweight.to(torch.int64).tolist(),
            multipliers=multiplier.tolist(),
            right_shifts=right_shift.tolist(),
            output_zero_points=[0] * output_q.numel(),
            bias_accumulators=[0] * output_q.numel(),
        )
    )
    require(independent.outputs == [output_q.to(torch.int64).tolist()], f"{name} reference differs")
    return {
        "name": name,
        "input_q": input_q,
        "input_scale": input_scale,
        "qweight": qweight,
        "weight_scale": weight_scale,
        "multiplier": multiplier,
        "right_shift": right_shift,
        "accumulator": accumulator,
        "rounded": rounded,
        "output_q": output_q,
        "output_scale": output_scale,
        "saturation": saturation,
        "float_output": float_output,
        "source_hashes": source_hashes,
    }


def derive() -> dict[str, Any]:
    provenance = validate_provenance()
    token_id = provenance["token_ids"][TOKEN_INDEX]
    with safe_open(MODEL, framework="pt", device="cpu") as weights, safe_open(ADAPTER, framework="pt", device="cpu") as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight")[token_id].contiguous()
        input_gain = weights.get_tensor("model.layers.0.input_layernorm.weight").contiguous()
        post_gain = weights.get_tensor("model.layers.0.post_attention_layernorm.weight").contiguous()

        float_input_norm = float_rmsnorm(embedding, input_gain)
        embedding_scale = scale_for(embedding.to(torch.float32))
        embedding_q = quantize_int8(embedding, embedding_scale)
        input_norm_scale = derive_rmsnorm_output_scale(
            input_gain.to(torch.float64).tolist(),
            float(float_input_norm.to(torch.float64).abs().max().item()),
        )
        input_gains = derive_scaled_gains_q8(input_gain.to(torch.float64).tolist(), input_norm_scale)
        input_norm = reference_rmsnorm(embedding_q.to(torch.int64).tolist(), input_gains)
        input_norm_q = torch.tensor(input_norm.outputs, dtype=torch.int8)

        projections: dict[str, dict[str, Any]] = {}
        for key, tensor_name in (
            ("q", "model.layers.0.self_attn.q_proj"),
            ("k", "model.layers.0.self_attn.k_proj"),
            ("v", "model.layers.0.self_attn.v_proj"),
        ):
            merged, hashes = merge_projection(weights, adapter, tensor_name)
            projections[key] = derive_projection(
                key, merged, input_norm_q, input_norm_scale, float_input_norm, hashes
            )

        q_values = projections["q"]["output_q"].to(torch.int64).tolist()
        k_values = projections["k"]["output_q"].to(torch.int64).tolist()
        rope_q = reference_rope(
            RopeCase("q_position0", 0, q_values, [Q9_SCALE_ONE] * HIDDEN, [Q15_ONE] * HIDDEN, [0] * HIDDEN)
        )
        rope_k = reference_rope(
            RopeCase("k_position0", 0, k_values, [Q9_SCALE_ONE] * len(k_values), [Q15_ONE] * len(k_values), [0] * len(k_values))
        )
        require(rope_q.outputs == q_values and rope_k.outputs == k_values, "position-0 RoPE is not identity")

        scores = []
        probabilities = []
        attention_heads = []
        v_q = projections["v"]["output_q"].to(torch.int64).reshape(KV_HEADS, HEAD_DIM)
        for head in range(Q_HEADS):
            kv_head = head // (Q_HEADS // KV_HEADS)
            q_head = rope_q.outputs[head * HEAD_DIM : (head + 1) * HEAD_DIM]
            k_head = rope_k.outputs[kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM]
            score = reference_attention_score(
                AttentionScoreCase(
                    f"head_{head}", q_head, [k_head],
                    projections["q"]["output_scale"], projections["k"]["output_scale"],
                )
            )
            softmax = reference_softmax(SoftmaxCase(f"head_{head}", score.core_scores_q6_9))
            compose = reference_attention_compose(
                AttentionComposeCase(f"head_{head}", score.core_scores_q6_9, [v_q[kv_head].tolist()])
            )
            require(softmax.probabilities_q0_15[0] == 32768, "singleton softmax changed")
            require(compose.outputs == v_q[kv_head].tolist(), "singleton attention compose changed V")
            scores.append(score)
            probabilities.append(softmax)
            attention_heads.extend(compose.outputs)
        attention_q = torch.tensor(attention_heads, dtype=torch.int8)
        float_v = projections["v"]["float_output"].reshape(KV_HEADS, HEAD_DIM)
        float_attention = float_v.repeat_interleave(Q_HEADS // KV_HEADS, dim=0).reshape(HIDDEN)

        o_merged, o_hashes = merge_projection(weights, adapter, "model.layers.0.self_attn.o_proj")
        projections["o"] = derive_projection(
            "o", o_merged, attention_q, projections["v"]["output_scale"], float_attention, o_hashes
        )
        float_attention_residual = embedding.to(torch.float32) + projections["o"]["float_output"]
        attention_residual_scale = scale_for(float_attention_residual)
        attention_residual_source = quantize_int8(embedding, attention_residual_scale)
        attention_residual_o = quantize_int8(
            projections["o"]["output_q"].to(torch.float64) * projections["o"]["output_scale"],
            attention_residual_scale,
        )
        attention_residual_values, attention_residual_sat = reference_residual_add(
            attention_residual_source.to(torch.int64).tolist(), attention_residual_o.to(torch.int64).tolist()
        )
        attention_residual_q = torch.tensor(attention_residual_values, dtype=torch.int8)

        float_post_norm = float_rmsnorm(float_attention_residual, post_gain)
        post_norm_scale = derive_rmsnorm_output_scale(
            post_gain.to(torch.float64).tolist(),
            float(float_post_norm.to(torch.float64).abs().max().item()),
        )
        post_gains = derive_scaled_gains_q8(post_gain.to(torch.float64).tolist(), post_norm_scale)
        post_norm = reference_rmsnorm(attention_residual_values, post_gains)
        post_norm_q = torch.tensor(post_norm.outputs, dtype=torch.int8)

        for key, tensor_name in (
            ("gate", "model.layers.0.mlp.gate_proj"),
            ("up", "model.layers.0.mlp.up_proj"),
        ):
            merged, hashes = merge_projection(weights, adapter, tensor_name)
            projections[key] = derive_projection(
                key, merged, post_norm_q, post_norm_scale, float_post_norm, hashes
            )

        float_silu = F.silu(projections["gate"]["float_output"]) * projections["up"]["float_output"]
        gate_q6 = torch.round(
            projections["gate"]["output_q"].to(torch.float64)
            * projections["gate"]["output_scale"]
            * (1 << 9)
        ).clamp(-32768, 32767).to(torch.int16)
        up_q6 = torch.round(
            projections["up"]["output_q"].to(torch.float64)
            * projections["up"]["output_scale"]
            * (1 << 9)
        ).clamp(-32768, 32767).to(torch.int16)
        silu_scale = scale_for(float_silu)
        silu_multiplier, silu_shift = derive_multiplier(torch.tensor([1.0 / (silu_scale * (1 << 21))], dtype=torch.float64))
        silu = reference_silu_gate(
            SiluGateCase(
                "layer0_silu", gate_q6.to(torch.int64).tolist(), up_q6.to(torch.int64).tolist(),
                int(silu_multiplier[0]), int(silu_shift[0]), 0,
            )
        )
        silu_q = torch.tensor(silu.outputs, dtype=torch.int8)

        down_merged, down_hashes = merge_projection(weights, adapter, "model.layers.0.mlp.down_proj")
        projections["down"] = derive_projection(
            "down", down_merged, silu_q, silu_scale, float_silu, down_hashes
        )
        float_layer_output = float_attention_residual + projections["down"]["float_output"]
        layer_output_scale = scale_for(float_layer_output)
        final_residual_stream = quantize_int8(
            attention_residual_q.to(torch.float64) * attention_residual_scale, layer_output_scale
        )
        final_residual_down = quantize_int8(
            projections["down"]["output_q"].to(torch.float64) * projections["down"]["output_scale"],
            layer_output_scale,
        )
        layer_values, layer_sat = reference_residual_add(
            final_residual_down.to(torch.int64).tolist(), final_residual_stream.to(torch.int64).tolist()
        )
        layer_q = torch.tensor(layer_values, dtype=torch.int8)
        dequant = layer_q.to(torch.float64) * layer_output_scale
        error = dequant - float_layer_output.to(torch.float64)
        relative_l2 = float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(float_layer_output.to(torch.float64)))

    return {
        "provenance": provenance,
        "token_id": token_id,
        "embedding": embedding,
        "embedding_q": embedding_q,
        "embedding_scale": embedding_scale,
        "input_gain": input_gain,
        "input_gains": input_gains,
        "input_norm": input_norm,
        "input_norm_q": input_norm_q,
        "input_norm_scale": input_norm_scale,
        "projections": projections,
        "rope_q": rope_q,
        "rope_k": rope_k,
        "scores": scores,
        "probabilities": probabilities,
        "attention_q": attention_q,
        "attention_scale": projections["v"]["output_scale"],
        "attention_residual": {
            "lhs": attention_residual_source,
            "rhs": attention_residual_o,
            "output": attention_residual_q,
            "scale": attention_residual_scale,
            "saturation": attention_residual_sat,
        },
        "post_gain": post_gain,
        "post_gains": post_gains,
        "post_norm": post_norm,
        "post_norm_q": post_norm_q,
        "post_norm_scale": post_norm_scale,
        "silu": {
            "gate_q6": gate_q6,
            "up_q6": up_q6,
            "output": silu_q,
            "scale": silu_scale,
            "multiplier": int(silu_multiplier[0]),
            "right_shift": int(silu_shift[0]),
            "result": silu,
        },
        "final_residual": {
            "down": final_residual_down,
            "stream": final_residual_stream,
            "output": layer_q,
            "scale": layer_output_scale,
            "saturation": layer_sat,
        },
        "float_layer_output": float_layer_output,
        "metrics": {
            "max_abs_dequantized_error": float(error.abs().max().item()),
            "mean_abs_dequantized_error": float(error.abs().mean().item()),
            "relative_l2_error": relative_l2,
            "float_output_absmax": float(float_layer_output.to(torch.float64).abs().max().item()),
            "layer_output_scale": layer_output_scale,
        },
    }


def pack(values: Iterable[int], width: int) -> int:
    result = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        result |= (int(value) & mask) << (lane * width)
    return result


def chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def hex_literal(value: int, width: int) -> str:
    return f"{width}'h{value & ((1 << width) - 1):0{width // 4}x}"


def render_rmsnorm_vectors(derived: dict[str, Any]) -> str:
    cases = [
        (derived["embedding_q"].to(torch.int64).tolist(), derived["input_gains"], derived["input_norm"]),
        (derived["attention_residual"]["output"].to(torch.int64).tolist(), derived["post_gains"], derived["post_norm"]),
    ]
    lines = [
        "// Canonical LoRA V4 layer-0 RMSNorm vectors.",
        "localparam integer TEST_COUNT = 2;",
        "localparam integer TEST_BEATS = 56;",
        "reg [8*16-1:0] test_input_beats [0:TEST_COUNT*TEST_BEATS-1];",
        "reg [16*16-1:0] test_gain_beats [0:TEST_COUNT*TEST_BEATS-1];",
        "reg [8*16-1:0] test_expected_beats [0:TEST_COUNT*TEST_BEATS-1];",
        "reg [47:0] test_expected_sumsq [0:TEST_COUNT-1];",
        "reg [31:0] test_expected_inv [0:TEST_COUNT-1];",
        "reg test_expected_saturation [0:TEST_COUNT-1];",
        "initial begin",
    ]
    for case, (inputs, gains, result) in enumerate(cases):
        lines += [
            f"  test_expected_sumsq[{case}] = {hex_literal(result.sumsq, 48)};",
            f"  test_expected_inv[{case}] = {hex_literal(result.inv_rms_q30, 32)};",
            f"  test_expected_saturation[{case}] = 1'b{int(result.saturation_seen)};",
        ]
        for beat, (input_word, gain_word, output_word) in enumerate(
            zip(pack_int8_beats(inputs), pack_gain_beats(gains), pack_int8_beats(result.outputs), strict=True)
        ):
            flat = case * 56 + beat
            lines += [
                f"  test_input_beats[{flat}] = {hex_literal(input_word, 128)};",
                f"  test_gain_beats[{flat}] = {hex_literal(gain_word, 256)};",
                f"  test_expected_beats[{flat}] = {hex_literal(output_word, 128)};",
            ]
    return "\n".join(lines + ["end", ""])


def render_rope_vectors(derived: dict[str, Any]) -> str:
    q = derived["projections"]["q"]["output_q"].to(torch.int64).tolist()
    k = derived["projections"]["k"]["output_q"].to(torch.int64).tolist() + [0] * (HIDDEN - KV_HEADS * HEAD_DIM)
    cases = [q, k]
    lines = [
        "// Canonical LoRA V4 position-0 RoPE vectors.",
        "localparam integer ROPE_CASE_COUNT = 2;",
        "localparam integer ROPE_HIDDEN_SIZE = 896;",
        "localparam integer ROPE_LANES = 16;",
        "localparam integer ROPE_BEATS = 56;",
        "reg [15:0] rope_sequence_position [0:ROPE_CASE_COUNT-1];",
        "reg rope_expected_saturation [0:ROPE_CASE_COUNT-1];",
        "reg [127:0] rope_input_beats [0:ROPE_CASE_COUNT*ROPE_BEATS-1];",
        "reg [127:0] rope_scale_beats [0:ROPE_CASE_COUNT*ROPE_BEATS*2-1];",
        "reg [127:0] rope_cos_beats [0:ROPE_CASE_COUNT*ROPE_BEATS*2-1];",
        "reg [127:0] rope_sin_beats [0:ROPE_CASE_COUNT*ROPE_BEATS*2-1];",
        "reg [127:0] rope_expected_beats [0:ROPE_CASE_COUNT*ROPE_BEATS-1];",
        "initial begin",
    ]
    scale_half = pack([Q9_SCALE_ONE] * 8, 16)
    cos_half = pack([Q15_ONE] * 8, 16)
    for case, values in enumerate(cases):
        lines += [f"  rope_sequence_position[{case}] = 16'd0;", f"  rope_expected_saturation[{case}] = 1'b0;"]
        for beat, group in enumerate(chunks(values, 16)):
            flat = case * 56 + beat
            lines += [
                f"  rope_input_beats[{flat}] = {hex_literal(pack(group, 8), 128)};",
                f"  rope_expected_beats[{flat}] = {hex_literal(pack(group, 8), 128)};",
            ]
            for half in range(2):
                table = case * 112 + beat * 2 + half
                lines += [
                    f"  rope_scale_beats[{table}] = {hex_literal(scale_half, 128)};",
                    f"  rope_cos_beats[{table}] = {hex_literal(cos_half, 128)};",
                    f"  rope_sin_beats[{table}] = 128'h00000000000000000000000000000000;",
                ]
    return "\n".join(lines + ["end", ""])


def render_attention_score_vectors(derived: dict[str, Any]) -> str:
    lines = [
        "// Canonical LoRA V4 singleton attention-score vectors.",
        "localparam integer ATTN_SCORE_CASE_COUNT = 14;",
        "localparam integer ATTN_SCORE_HEAD_DIM = 64;",
        "localparam integer ATTN_SCORE_MAC_LANES = 1;",
        "localparam integer ATTN_SCORE_CONTEXT_MAX = 8;",
        "localparam integer ATTN_SCORE_BEATS_PER_VECTOR = 4;",
        "localparam [127:0] ATTN_SCORE_INVALID_NEGATIVE_MULTIPLIER = 128'd0;",
        "localparam [127:0] ATTN_SCORE_INVALID_RESERVED_BITS = 128'd0;",
        "reg [15:0] attn_score_context_count [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg signed [31:0] attn_score_multiplier [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg [5:0] attn_score_right_shift [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg attn_score_expected_saturation [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg attn_score_expected_saturation_token [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_CONTEXT_MAX-1];",
        "reg attn_score_expected_core_saturation_token [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_CONTEXT_MAX-1];",
        "reg [31:0] attn_score_expected_acc [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_CONTEXT_MAX-1];",
        "reg [127:0] attn_score_expected_word [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg [127:0] attn_score_expected_core_word [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg [127:0] attn_score_q_beats [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_BEATS_PER_VECTOR-1];",
        "reg [127:0] attn_score_k_beats [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_CONTEXT_MAX*ATTN_SCORE_BEATS_PER_VECTOR-1];",
        "initial begin",
    ]
    q = derived["rope_q"].outputs
    k = derived["rope_k"].outputs
    for head, score in enumerate(derived["scores"]):
        kv_head = head // 7
        lines += [
            f"  attn_score_context_count[{head}] = 16'd1;",
            f"  attn_score_multiplier[{head}] = 32'sd{score.multiplier};",
            f"  attn_score_right_shift[{head}] = 6'd{score.right_shift};",
            f"  attn_score_expected_saturation[{head}] = 1'b0;",
            f"  attn_score_expected_word[{head}] = {hex_literal(pack(score.scores_q6_9 + [0] * 7, 16), 128)};",
            f"  attn_score_expected_core_word[{head}] = {hex_literal(pack(score.core_scores_q6_9 + [0] * 7, 16), 128)};",
        ]
        for beat in range(4):
            q_group = q[head * 64 + beat * 16 : head * 64 + (beat + 1) * 16]
            k_group = k[kv_head * 64 + beat * 16 : kv_head * 64 + (beat + 1) * 16]
            lines.append(f"  attn_score_q_beats[{head * 4 + beat}] = {hex_literal(pack(q_group, 8), 128)};")
            lines.append(f"  attn_score_k_beats[{head * 32 + beat}] = {hex_literal(pack(k_group, 8), 128)};")
        for token in range(8):
            flat = head * 8 + token
            acc = score.accumulators[0] if token == 0 else 0
            core_sat = score.core_saturation_by_token[0] if token == 0 else False
            centered_sat = score.saturation_by_token[0] if token == 0 else False
            lines += [
                f"  attn_score_expected_acc[{flat}] = {hex_literal(acc, 32)};",
                f"  attn_score_expected_saturation_token[{flat}] = 1'b{int(centered_sat)};",
                f"  attn_score_expected_core_saturation_token[{flat}] = 1'b{int(core_sat)};",
            ]
            if token != 0:
                for beat in range(4):
                    lines.append(f"  attn_score_k_beats[{head * 32 + token * 4 + beat}] = 128'd0;")
    return "\n".join(lines + ["end", ""])


def render_softmax_vectors(derived: dict[str, Any]) -> str:
    lines = [
        "// Canonical LoRA V4 singleton softmax vectors.",
        "localparam integer SOFTMAX_CASE_COUNT = 14;",
        "localparam integer SOFTMAX_CONTEXT_MAX = 8;",
        "reg [15:0] softmax_context_count [0:SOFTMAX_CASE_COUNT-1];",
        "reg [127:0] softmax_score_word [0:SOFTMAX_CASE_COUNT-1];",
        "reg [127:0] softmax_expected_exp_word [0:SOFTMAX_CASE_COUNT-1];",
        "reg [18:0] softmax_expected_exp_sum [0:SOFTMAX_CASE_COUNT-1];",
        "reg [127:0] softmax_expected_word [0:SOFTMAX_CASE_COUNT-1];",
        "initial begin",
    ]
    for case, (score, softmax) in enumerate(zip(derived["scores"], derived["probabilities"], strict=True)):
        lines += [
            f"  softmax_context_count[{case}] = 16'd1;",
            f"  softmax_score_word[{case}] = {hex_literal(pack(score.core_scores_q6_9 + [0] * 7, 16), 128)};",
            f"  softmax_expected_exp_word[{case}] = {hex_literal(pack(softmax.exp_weights_q15, 16), 128)};",
            f"  softmax_expected_exp_sum[{case}] = 19'd{softmax.exp_sum_q15};",
            f"  softmax_expected_word[{case}] = {hex_literal(pack(softmax.probabilities_q0_15, 16), 128)};",
        ]
    return "\n".join(lines + ["end", ""])


def render_attention_compose_vectors(derived: dict[str, Any]) -> str:
    v = derived["projections"]["v"]["output_q"].to(torch.int64).reshape(2, 64)
    lines = [
        "// Canonical LoRA V4 singleton attention represented as two protocol tiles.",
        "localparam integer ATTN_COMPOSE_CASE_COUNT = 14;",
        "localparam integer ATTN_COMPOSE_TILE = 8;",
        "localparam integer ATTN_COMPOSE_MAX_CONTEXT = 9;",
        "localparam integer ATTN_COMPOSE_MAX_TILES = 2;",
        "localparam integer ATTN_COMPOSE_BEATS = 4;",
        "reg [15:0] attn_compose_context_count [0:ATTN_COMPOSE_CASE_COUNT-1];",
        "reg [127:0] attn_compose_score_tiles [0:ATTN_COMPOSE_CASE_COUNT*ATTN_COMPOSE_MAX_TILES-1];",
        "reg [127:0] attn_compose_value_beats [0:ATTN_COMPOSE_CASE_COUNT*ATTN_COMPOSE_MAX_CONTEXT*ATTN_COMPOSE_BEATS-1];",
        "reg [127:0] attn_compose_expected_beats [0:ATTN_COMPOSE_CASE_COUNT*ATTN_COMPOSE_BEATS-1];",
        "reg attn_compose_expected_saturation [0:ATTN_COMPOSE_CASE_COUNT-1];",
        "initial begin",
    ]
    for head, score in enumerate(derived["scores"]):
        kv_head = head // 7
        values = v[kv_head].tolist()
        protocol_scores = [score.core_scores_q6_9[0]] + [-32768] * 8
        protocol_values = [values] + [[0] * HEAD_DIM for _ in range(8)]
        compose = reference_attention_compose(
            AttentionComposeCase(f"head_{head}_protocol_padding", protocol_scores, protocol_values)
        )
        require(compose.outputs == values, "attention-compose protocol padding changed singleton output")
        lines += [
            f"  attn_compose_context_count[{head}] = 16'd9;",
            f"  attn_compose_score_tiles[{head * 2}] = {hex_literal(pack(protocol_scores[:8], 16), 128)};",
            f"  attn_compose_score_tiles[{head * 2 + 1}] = {hex_literal(pack(protocol_scores[8:] + [0] * 7, 16), 128)};",
            f"  attn_compose_expected_saturation[{head}] = 1'b{int(compose.saturation_seen)};",
        ]
        for token, row in enumerate(protocol_values):
            for beat, group in enumerate(chunks(row, 16)):
                flat = head * 9 * 4 + token * 4 + beat
                lines.append(f"  attn_compose_value_beats[{flat}] = {hex_literal(pack(group, 8), 128)};")
        for beat, group in enumerate(chunks(values, 16)):
            lines.append(
                f"  attn_compose_expected_beats[{head * 4 + beat}] = {hex_literal(pack(group, 8), 128)};"
            )
    return "\n".join(lines + ["end", ""])


def render_silu_vectors(derived: dict[str, Any]) -> str:
    gate = derived["silu"]["gate_q6"].to(torch.int64).tolist()
    up = derived["silu"]["up_q6"].to(torch.int64).tolist()
    output = derived["silu"]["output"].to(torch.int64).tolist()
    result = derived["silu"]["result"]
    lines = [
        "// Canonical LoRA V4 layer-0 SiLU vectors.",
        "localparam integer SILU_CASE_COUNT = 1;",
        "localparam integer SILU_MAX_INPUT_BEATS = 608;",
        "localparam integer SILU_MAX_OUTPUT_BEATS = 304;",
        "localparam [5:0] SILU_REQUIRED_BOUNDARY_COVERAGE = 6'd0;",
        "reg [15:0] silu_case_length [0:SILU_CASE_COUNT-1];",
        "reg signed [31:0] silu_case_multiplier [0:SILU_CASE_COUNT-1];",
        "reg [5:0] silu_case_right_shift [0:SILU_CASE_COUNT-1];",
        "reg signed [7:0] silu_case_zero_point [0:SILU_CASE_COUNT-1];",
        "reg silu_expected_saturation [0:SILU_CASE_COUNT-1];",
        "reg [5:0] silu_case_boundary_coverage [0:SILU_CASE_COUNT-1];",
        "reg [127:0] silu_gate_beats [0:SILU_CASE_COUNT*SILU_MAX_INPUT_BEATS-1];",
        "reg [127:0] silu_up_beats [0:SILU_CASE_COUNT*SILU_MAX_INPUT_BEATS-1];",
        "reg [127:0] silu_expected_beats [0:SILU_CASE_COUNT*SILU_MAX_OUTPUT_BEATS-1];",
        "initial begin",
        f"  silu_case_length[0] = 16'd{INTERMEDIATE};",
        f"  silu_case_multiplier[0] = 32'sd{derived['silu']['multiplier']};",
        f"  silu_case_right_shift[0] = 6'd{derived['silu']['right_shift']};",
        "  silu_case_zero_point[0] = 8'sd0;",
        f"  silu_expected_saturation[0] = 1'b{int(result.saturation_seen)};",
        "  silu_case_boundary_coverage[0] = 6'd0;",
    ]
    for beat in range(608):
        lines += [
            f"  silu_gate_beats[{beat}] = {hex_literal(pack(gate[beat * 8 : (beat + 1) * 8], 16), 128)};",
            f"  silu_up_beats[{beat}] = {hex_literal(pack(up[beat * 8 : (beat + 1) * 8], 16), 128)};",
        ]
    for beat, group in enumerate(chunks(output, 16)):
        lines.append(f"  silu_expected_beats[{beat}] = {hex_literal(pack(group, 8), 128)};")
    return "\n".join(lines + ["end", ""])


def render_residual_vectors(values: dict[str, Any], *, mlp: bool) -> str:
    if mlp:
        lhs = values["down"].to(torch.int64).tolist()
        rhs = values["stream"].to(torch.int64).tolist()
        output = values["output"].to(torch.int64).tolist()
        prefix = "mlp_residual"
        header = [
            "localparam integer MLP_RESIDUAL_CASE_COUNT = 1;",
            "localparam integer MLP_RESIDUAL_BEATS = 56;",
            "reg [127:0] mlp_residual_down_beats [0:55];",
            "reg [127:0] mlp_residual_stream_beats [0:55];",
            "reg [127:0] mlp_residual_expected_beats [0:55];",
            "reg mlp_residual_expected_saturation [0:0];",
            "initial begin",
            f"  mlp_residual_expected_saturation[0] = 1'b{int(values['saturation'])};",
        ]
        names = ("down", "stream")
    else:
        lhs = values["lhs"].to(torch.int64).tolist()
        rhs = values["rhs"].to(torch.int64).tolist()
        output = values["output"].to(torch.int64).tolist()
        prefix = "residual"
        header = [
            "localparam integer RESIDUAL_CASE_COUNT = 1;",
            "localparam integer RESIDUAL_BEATS = 56;",
            "reg [127:0] residual_lhs_beats [0:55];",
            "reg [127:0] residual_rhs_beats [0:55];",
            "reg [127:0] residual_expected_beats [0:55];",
            "reg residual_expected_saturation [0:0];",
            "initial begin",
            f"  residual_expected_saturation[0] = 1'b{int(values['saturation'])};",
        ]
        names = ("lhs", "rhs")
    lines = ["// Canonical LoRA V4 residual vectors.", *header]
    for beat, (lhs_group, rhs_group, out_group) in enumerate(
        zip(chunks(lhs, 16), chunks(rhs, 16), chunks(output, 16), strict=True)
    ):
        lines += [
            f"  {prefix}_{names[0]}_beats[{beat}] = {hex_literal(pack(lhs_group, 8), 128)};",
            f"  {prefix}_{names[1]}_beats[{beat}] = {hex_literal(pack(rhs_group, 8), 128)};",
            f"  {prefix}_expected_beats[{beat}] = {hex_literal(pack(out_group, 8), 128)};",
        ]
    return "\n".join(lines + ["end", ""])


def persist_vectors(derived: dict[str, Any]) -> dict[str, Any]:
    vector_dir = ATTEMPT / "vectors"
    artifacts: dict[str, Any] = {}
    activation_words: list[int] = []
    weight_words: list[int] = []
    multipliers: list[int] = []
    shifts: list[int] = []
    outputs: list[int] = []
    accumulators: list[int] = []
    saturation: list[int] = []
    for key in PROJECTION_ORDER:
        projection = derived["projections"][key]
        input_values = projection["input_q"].to(torch.int64).tolist()
        qweight = projection["qweight"].to(torch.int64)
        activation_words.extend(pack(group, 8) for group in chunks(input_values, 4))
        for row in qweight:
            weight_words.extend(pack(group, 4) for group in chunks(row.tolist(), 4))
        multipliers.extend(projection["multiplier"].tolist())
        shifts.extend(projection["right_shift"].tolist())
        outputs.extend(projection["output_q"].to(torch.int64).tolist())
        accumulators.extend(projection["accumulator"].tolist())
        saturation.extend(projection["saturation"].to(torch.int64).tolist())
    artifacts["projection_activation"] = write_hex(vector_dir / "projection_activation.hex", activation_words, 32)
    artifacts["projection_weight"] = write_hex(vector_dir / "projection_weight.hex", weight_words, 16)
    artifacts["projection_multiplier"] = write_hex(vector_dir / "projection_multiplier.hex", multipliers, 32)
    artifacts["projection_shift"] = write_hex(vector_dir / "projection_shift.hex", shifts, 8)
    artifacts["projection_expected"] = write_hex(vector_dir / "projection_expected.hex", outputs, 8)
    artifacts["projection_accumulator"] = write_hex(vector_dir / "projection_accumulator.hex", accumulators, 32)
    artifacts["projection_saturation"] = write_hex(vector_dir / "projection_saturation.hex", saturation, 8)
    artifacts["rmsnorm"] = write_text(vector_dir / "rmsnorm_vectors.svh", render_rmsnorm_vectors(derived))
    artifacts["rope"] = write_text(vector_dir / "rope_vectors.svh", render_rope_vectors(derived))
    artifacts["attention_score"] = write_text(vector_dir / "attention_score_vectors.svh", render_attention_score_vectors(derived))
    artifacts["softmax"] = write_text(vector_dir / "softmax_vectors.svh", render_softmax_vectors(derived))
    artifacts["attention_compose"] = write_text(vector_dir / "attention_compose_vectors.svh", render_attention_compose_vectors(derived))
    artifacts["silu"] = write_text(vector_dir / "silu_gate_vectors.svh", render_silu_vectors(derived))
    artifacts["residual"] = write_text(vector_dir / "residual_vectors.svh", render_residual_vectors(derived["attention_residual"], mlp=False))
    artifacts["mlp_residual"] = write_text(vector_dir / "mlp_residual_vectors.svh", render_residual_vectors(derived["final_residual"], mlp=True))
    return artifacts


def execution_source(name: str, include_name: str) -> Path:
    source = MAINTAINED_TBS[name].read_text(encoding="utf-8")
    old = {
        "rmsnorm": '../generated/rmsnorm_vectors.svh',
        "rope": '../generated/rope_vectors.svh',
        "attention_score": '../generated/attention_score_vectors.svh',
        "softmax": '../generated/softmax_vectors.svh',
        "attention_compose": '../generated/attention_compose_vectors.svh',
        "silu": '../generated/silu_gate_vectors.svh',
    }[name]
    source = source.replace(f'`include "{old}"', f'`include "{relative(ATTEMPT / "vectors" / include_name)}"', 1)
    path = ATTEMPT / "execution_sources" / f"{name}_tb.sv"
    write_text(path, source)
    return path


def projection_execution_source() -> Path:
    source = PROJECTION_TB.read_text(encoding="utf-8")
    frozen_prefix = "evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1/attempt-0001/vectors"
    source = source.replace(frozen_prefix, relative(ATTEMPT / "vectors"))
    path = ATTEMPT / "execution_sources/projection_tb.sv"
    write_text(path, source)
    return path


def residual_execution_source() -> Path:
    source = MAINTAINED_TBS["residual"].read_text(encoding="utf-8")
    source = source.replace(
        '`include "../generated/residual_vectors.svh"',
        f'`include "{relative(ATTEMPT / "vectors/residual_vectors.svh")}"',
        1,
    ).replace(
        '`include "../generated/mlp_residual_vectors.svh"',
        f'`include "{relative(ATTEMPT / "vectors/mlp_residual_vectors.svh")}"',
        1,
    )
    branch = r'''
        if ($test$plusargs("LORA_V4_LAYER0_RESIDUAL_REPLAY")) begin
            send_residual_cmd_full(0, 16'd896, RESIDUAL_LHS_BASE, RESIDUAL_RHS_BASE,
                                   RESIDUAL_OUT_BASE, 16'h7a00);
            wait_residual_done_and_compare(0, residual_expected_saturation[0], 16'h7a00);
            for (case_index = 0; case_index < observed_count; case_index = case_index + 1)
                $display("RTL_RESIDUAL_BEAT stage=attention beat=%0d data=%032x", case_index, observed_output[case_index]);
            if (residual_expected_saturation[0])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            send_mlp_residual_cmd_full(0, 16'd896, MLP_RESIDUAL_DOWN_BASE,
                                       MLP_RESIDUAL_STREAM_BASE, MLP_RESIDUAL_OUT_BASE, 16'h7a01);
            wait_mlp_residual_done_and_compare(0, mlp_residual_expected_saturation[0], 16'h7a01);
            for (case_index = 0; case_index < observed_count; case_index = case_index + 1)
                $display("RTL_RESIDUAL_BEAT stage=final beat=%0d data=%032x", case_index, observed_output[case_index]);
            if (failures != 0)
                $fatal(1, "ACE2_LORA_V4_LAYER0_RESIDUAL_REPLAY_FAIL failures=%0d", failures);
            $display("ACE2_LORA_V4_LAYER0_RESIDUAL_REPLAY_PASS stages=2 beats=112");
            $finish;
        end

'''
    needle = "        if (qproj_stride_only_mode) begin"
    require(needle in source, "shell residual insertion point changed")
    source = source.replace(needle, branch + needle, 1)
    path = ATTEMPT / "execution_sources/residual_tb.sv"
    write_text(path, source)
    return path


def run_process(name: str, command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    logs = ATTEMPT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (logs / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed


def compile_and_run(name: str, sources: list[Path], top: str, plusargs: list[str] | None = None) -> dict[str, Any]:
    binary = ATTEMPT / "sim" / f"{name}.vvp"
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "iverilog", "-g2012", "-I.", "-Irtl", "-Irtl/generated", "-Iverification/tb",
        "-s", top, "-o", relative(binary), *[relative(path) for path in sources],
    ]
    compiled = run_process(f"{name}.iverilog", command)
    require(compiled.returncode == 0, f"{name} compile failed")
    simulate = ["vvp", relative(binary), *(plusargs or [])]
    replayed = run_process(f"{name}.vvp", simulate)
    require(replayed.returncode == 0, f"{name} simulation failed")
    return {
        "compile_command": command,
        "simulate_command": simulate,
        "binary": file_record(binary),
        "stdout": file_record(ATTEMPT / f"logs/{name}.vvp.stdout.log"),
        "stderr": file_record(ATTEMPT / f"logs/{name}.vvp.stderr.log"),
        "stdout_text": replayed.stdout,
    }


def run_rtl(derived: dict[str, Any]) -> dict[str, Any]:
    sources = {
        "projection": projection_execution_source(),
        "rmsnorm": execution_source("rmsnorm", "rmsnorm_vectors.svh"),
        "rope": execution_source("rope", "rope_vectors.svh"),
        "attention_score": execution_source("attention_score", "attention_score_vectors.svh"),
        "softmax": execution_source("softmax", "softmax_vectors.svh"),
        "attention_compose": execution_source("attention_compose", "attention_compose_vectors.svh"),
        "silu": execution_source("silu", "silu_gate_vectors.svh"),
        "residual": residual_execution_source(),
    }
    results = {
        "projection": compile_and_run("projection", [CORES["projection"], sources["projection"]], "ace2_lora_v4_layer0_projection_batch_tb"),
        "rmsnorm": compile_and_run("rmsnorm", [CORES["rmsnorm"], sources["rmsnorm"]], "ace2_rmsnorm_tb"),
        "rope": compile_and_run("rope", [CORES["rope"], sources["rope"]], "ace2_rope_tb"),
        "attention_score": compile_and_run("attention_score", [CORES["attention_score"], sources["attention_score"]], "ace2_attention_score_tb"),
        "softmax": compile_and_run("softmax", [CORES["softmax"], sources["softmax"]], "ace2_softmax_tb"),
        "attention_compose": compile_and_run("attention_compose", [CORES["attention_compose"], sources["attention_compose"]], "ace2_attention_compose_tb"),
        "silu": compile_and_run("silu", [CORES["silu"], sources["silu"]], "ace2_silu_gate_tb"),
    }
    rtl_sources = [ROOT / "rtl/ace2_pkg.sv"] + sorted(
        path for path in (ROOT / "rtl").glob("*.sv") if path.name != "ace2_pkg.sv"
    )
    results["residual"] = compile_and_run(
        "residual", [*rtl_sources, sources["residual"]], "ace2_shell_tb", ["+LORA_V4_LAYER0_RESIDUAL_REPLAY"]
    )

    markers = {
        "projection": "ACE2_LORA_V4_LAYER0_PROJECTION_BATCH_PASS operators=7 channels=12672",
        "rmsnorm": "ACE2_RMSNORM_TB_PASS cases=2 beats_per_case=56",
        "rope": "ACE2_ROPE_TB_PASS cases=2 beats_per_case=56",
        "attention_score": "ACE2_ATTN_SCORE_TB_PASS cases=14 context_max=8",
        "softmax": "ACE2_SOFTMAX_TB_PASS cases=14 context_max=8",
        "attention_compose": "ACE2_ATTN_COMPOSE_TB_PASS cases=14",
        "silu": "ACE2_SILU_GATE_TB_PASS cases=1",
        "residual": "ACE2_LORA_V4_LAYER0_RESIDUAL_REPLAY_PASS stages=2 beats=112",
    }
    for name, marker in markers.items():
        require(marker in results[name]["stdout_text"], f"{name} PASS marker missing")
    projection_rows = re.findall(r"RTL_PROJ_RESULT operator=(\d+) channel=(\d+) out_u8=(\d+) acc=(-?\d+) saturation=(\d+) overflow=(\d+)", results["projection"]["stdout_text"])
    require(len(projection_rows) == 12672, "projection RTL output count changed")
    final_beats = re.findall(r"RTL_RESIDUAL_BEAT stage=final beat=(\d+) data=([0-9a-fA-F]+)", results["residual"]["stdout_text"])
    require([int(index) for index, _ in final_beats] == list(range(56)), "final residual beat order changed")
    final_raw = b"".join(int(data, 16).to_bytes(16, "little") for _, data in final_beats)
    expected_raw = raw_tensor_bytes(derived["final_residual"]["output"])
    require(final_raw == expected_raw, "final RTL layer output differs from fixed-point golden")
    for value in results.values():
        value.pop("stdout_text", None)
    return {"executions": results, "projection_results": len(projection_rows), "final_layer_output_sha256": sha256_bytes(final_raw)}


def persist_tensors(derived: dict[str, Any]) -> dict[str, Any]:
    directory = ATTEMPT / "tensors"
    payloads = {
        "embedding_bf16le.bin": raw_tensor_bytes(derived["embedding"]),
        "input_rmsnorm_s8.bin": raw_tensor_bytes(derived["input_norm_q"]),
        "k_proj_output_s8.bin": raw_tensor_bytes(derived["projections"]["k"]["output_q"]),
        "v_proj_output_s8.bin": raw_tensor_bytes(derived["projections"]["v"]["output_q"]),
        "attention_output_s8.bin": raw_tensor_bytes(derived["attention_q"]),
        "attention_residual_s8.bin": raw_tensor_bytes(derived["attention_residual"]["output"]),
        "post_attention_rmsnorm_s8.bin": raw_tensor_bytes(derived["post_norm_q"]),
        "silu_output_s8.bin": raw_tensor_bytes(derived["silu"]["output"]),
        "layer0_output_s8.bin": raw_tensor_bytes(derived["final_residual"]["output"]),
        "layer0_output_float_f32le.bin": raw_tensor_bytes(derived["float_layer_output"]),
        "layer0_output_scale_f64le.bin": struct.pack("<d", derived["final_residual"]["scale"]),
    }
    for key in PROJECTION_ORDER:
        projection = derived["projections"][key]
        payloads[f"{key}_qweight_s4_in_s8.bin"] = raw_tensor_bytes(projection["qweight"])
        payloads[f"{key}_output_s8.bin"] = raw_tensor_bytes(projection["output_q"])
        payloads[f"{key}_accumulator_s32le.bin"] = raw_tensor_bytes(projection["accumulator"].to(torch.int32))
    return {name: write_binary(directory / name, raw) for name, raw in payloads.items()}


def conversion_summary(derived: dict[str, Any], tensors: dict[str, Any], vectors: dict[str, Any]) -> dict[str, Any]:
    projections = {}
    for key in PROJECTION_ORDER:
        value = derived["projections"][key]
        projections[key] = {
            "input_elements": int(value["input_q"].numel()),
            "output_channels": int(value["output_q"].numel()),
            "weight_elements": int(value["qweight"].numel()),
            "input_scale": value["input_scale"],
            "output_scale": value["output_scale"],
            "accumulator_range": [int(value["accumulator"].min()), int(value["accumulator"].max())],
            "saturation_count": int(value["saturation"].sum()),
            "source_hashes": value["source_hashes"],
        }
    return {
        "schema_version": 1,
        "status": "PASS_DERIVATION",
        "token_id": derived["token_id"],
        "token_index": TOKEN_INDEX,
        "projection_order": PROJECTION_ORDER,
        "projections": projections,
        "rmsnorm": {
            "input_sumsq": derived["input_norm"].sumsq,
            "input_inv_rms_q30": derived["input_norm"].inv_rms_q30,
            "post_sumsq": derived["post_norm"].sumsq,
            "post_inv_rms_q30": derived["post_norm"].inv_rms_q30,
        },
        "singleton_attention": {
            "query_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "softmax_probability_q15": 32768,
            "position": 0,
            "compose_protocol": "one real token plus eight exact-zero-weight/value padding entries across FIRST/LAST tiles",
        },
        "scales": {
            "embedding": derived["embedding_scale"],
            "input_rmsnorm": derived["input_norm_scale"],
            "attention_residual": derived["attention_residual"]["scale"],
            "post_attention_rmsnorm": derived["post_norm_scale"],
            "silu": derived["silu"]["scale"],
            "layer_output": derived["final_residual"]["scale"],
        },
        "tensor_artifacts": tensors,
        "vector_artifacts": vectors,
    }


def comparison(derived: dict[str, Any], rtl: dict[str, Any]) -> dict[str, Any]:
    metrics = derived["metrics"]
    exact = {
        "operator_output_mismatches": 0,
        "projection_accumulator_mismatches": 0,
        "layer_output_integer_mismatches": 0,
        "projection_results": rtl["projection_results"],
        "final_layer_output_sha256": rtl["final_layer_output_sha256"],
    }
    passed = (
        exact["operator_output_mismatches"] <= THRESHOLDS["rtl_vs_python_fixed_point"]["operator_output_mismatches_max"]
        and exact["projection_accumulator_mismatches"] <= THRESHOLDS["rtl_vs_python_fixed_point"]["projection_accumulator_mismatches_max"]
        and exact["layer_output_integer_mismatches"] <= THRESHOLDS["rtl_vs_python_fixed_point"]["layer_output_integer_mismatches_max"]
        and metrics["max_abs_dequantized_error"] <= THRESHOLDS["merged_float_vs_w4a8_layer_output"]["max_abs_dequantized_error_max"]
        and metrics["relative_l2_error"] <= THRESHOLDS["merged_float_vs_w4a8_layer_output"]["relative_l2_error_max"]
    )
    return {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "thresholds": THRESHOLDS,
        "rtl_vs_python_fixed_point": exact,
        "merged_float_vs_w4a8_layer_output": metrics,
    }


def prepare() -> None:
    require(not CONTRACT.exists(), f"contract already exists: {CONTRACT}")
    provenance = validate_provenance()
    contract = {
        "schema_version": 1,
        "mission_increment": MISSION,
        "claim": "canonical LoRA V4 complete layer-0 token-0 W4A8 RTL comparison",
        "canonical_checkpoint": {
            "path": relative(ADAPTER_DIR),
            "adapter_sha256": ADAPTER_SHA256,
            "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
            "base_model_sha256": MODEL_SHA256,
            "quality_claim_boundary": "independent terminal NO-GO 36/56 below frozen 48-pass minimum",
        },
        "fixed_input": {
            "text": PROMPT,
            "token_ids": provenance["token_ids"],
            "selected_token_index": TOKEN_INDEX,
            "selected_token_id": provenance["token_ids"][TOKEN_INDEX],
            "attention_context": "causal singleton token at sequence position 0",
        },
        "architecture": {
            "geometry": {"hidden": HIDDEN, "intermediate": INTERMEDIATE, "query_heads": Q_HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM},
            "operator_order": OPERATOR_ORDER,
            "lora_merge": "W_base_fp32 + (alpha/r) * (B_fp32 @ A_fp32) for Q/K/V/O/gate/up/down",
        },
        "quantization": {
            "projection_weight": "signed W4 symmetric per output channel absmax/7",
            "activation": "signed A8 per operator tensor",
            "projection_accumulator": "signed int32 exact complete reduction",
            "rounding": "round-to-nearest ties-to-even",
            "rmsnorm": "ACE-2 int8 input plus signed Q7.8 gains",
            "rope": "position-0 Q9 scale one and Q15 cosine one",
            "attention": "14 singleton query-head replays with grouped 2-head K/V mapping",
            "attention_compose_protocol": "one real token plus eight exact-zero-weight/value protocol-padding entries across FIRST/LAST tiles",
            "residual": "ACE-2 shell residual opcode at declared common tensor scales",
        },
        "thresholds": THRESHOLDS,
        "manifest_erratum": file_record(ERRATUM),
        "source_bindings": {
            "runner": file_record(Path(__file__)),
            "projection_testbench": file_record(PROJECTION_TB),
            "maintained_testbenches": {name: file_record(path) for name, path in MAINTAINED_TBS.items()},
            "rtl_cores": {name: file_record(path) for name, path in CORES.items()},
            "shell": file_record(ROOT / "rtl/ace2_shell.sv"),
            "rtl_tree": tree_record("rtl/*.sv"),
            "verification_generated_tree": tree_record("verification/generated/*.svh"),
            "config": file_record(CONFIG),
            "adapter_config": file_record(ADAPTER_CONFIG),
            "checkpoint_identity": file_record(CHECKPOINT_IDENTITY),
        },
        "commands": {
            "preflight": ".venv/bin/python tools/run_lora_v4_layer0_full_rtl.py --preflight",
            "prepare": ".venv/bin/python tools/run_lora_v4_layer0_full_rtl.py --prepare",
            "run": ".venv/bin/python tools/run_lora_v4_layer0_full_rtl.py --run",
            "check": ".venv/bin/python tools/run_lora_v4_layer0_full_rtl.py --check",
        },
        "attempt_policy": {"official_attempt": "attempt-0001", "immutable_after_execution": True},
    }
    write_json(CONTRACT, contract)
    print(f"FROZEN {relative(CONTRACT)} sha256={sha256_file(CONTRACT)}")


def validate_contract() -> dict[str, Any]:
    require(CONTRACT.is_file(), "full-layer contract is absent")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    validate_provenance()
    require(contract["source_bindings"]["runner"]["sha256"] == sha256_file(Path(__file__)), "runner changed after freeze")
    require(contract["source_bindings"]["projection_testbench"]["sha256"] == sha256_file(PROJECTION_TB), "projection TB changed after freeze")
    for name, path in MAINTAINED_TBS.items():
        require(contract["source_bindings"]["maintained_testbenches"][name]["sha256"] == sha256_file(path), f"{name} TB changed")
    for name, path in CORES.items():
        require(contract["source_bindings"]["rtl_cores"][name]["sha256"] == sha256_file(path), f"{name} core changed")
    require(contract["source_bindings"]["rtl_tree"]["sha256"] == tree_record("rtl/*.sv")["sha256"], "RTL tree changed")
    require(
        contract["source_bindings"]["verification_generated_tree"]["sha256"]
        == tree_record("verification/generated/*.svh")["sha256"],
        "generated verification tree changed",
    )
    require(contract["fixed_input"]["token_ids"] == tokenizer_ids(), "fixed tokenizer IDs changed")
    return contract


def run() -> None:
    require(not ATTEMPT.exists(), f"official attempt already exists: {ATTEMPT}")
    contract = validate_contract()
    require(shutil.which("iverilog") is not None and shutil.which("vvp") is not None, "Icarus tools unavailable")
    ATTEMPT.mkdir(parents=True)
    derived = derive()
    tensors = persist_tensors(derived)
    vectors = persist_vectors(derived)
    write_json(ATTEMPT / "conversion.json", conversion_summary(derived, tensors, vectors))
    rtl = run_rtl(derived)
    write_json(ATTEMPT / "rtl_execution.json", rtl)
    result = comparison(derived, rtl)
    write_json(ATTEMPT / "comparison.json", result)
    manifest = {
        "schema_version": 1,
        "status": result["status"],
        "claim": contract["claim"],
        "contract": file_record(CONTRACT),
        "canonical_checkpoint": contract["canonical_checkpoint"],
        "fixed_input": contract["fixed_input"],
        "operator_order": OPERATOR_ORDER,
        "thresholds": THRESHOLDS,
        "conversion": file_record(ATTEMPT / "conversion.json"),
        "rtl_execution": file_record(ATTEMPT / "rtl_execution.json"),
        "comparison": file_record(ATTEMPT / "comparison.json"),
        "layer_output": {
            "integer_tensor": tensors["layer0_output_s8.bin"],
            "floating_reference": tensors["layer0_output_float_f32le.bin"],
            "scale": tensors["layer0_output_scale_f64le.bin"],
            "rtl_output_sha256": rtl["final_layer_output_sha256"],
        },
        "manifest_erratum": contract["manifest_erratum"],
        "commands": contract["commands"],
    }
    write_json(ATTEMPT / "manifest.json", manifest)
    write_sha256s(ATTEMPT)
    print(json.dumps(result, sort_keys=True))
    require(result["status"] == "PASS", "full layer comparison failed frozen thresholds")


def check() -> None:
    contract = validate_contract()
    require(ATTEMPT.is_dir(), "full-layer attempt is absent")
    members = verify_sha256s(ATTEMPT)
    manifest = json.loads((ATTEMPT / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((ATTEMPT / "comparison.json").read_text(encoding="utf-8"))
    execution = json.loads((ATTEMPT / "rtl_execution.json").read_text(encoding="utf-8"))
    require(manifest["status"] == result["status"] == "PASS", "full-layer status is not PASS")
    require(manifest["contract"]["sha256"] == sha256_file(CONTRACT), "contract binding changed")
    require(result["thresholds"] == contract["thresholds"], "thresholds changed")
    require(execution["projection_results"] == 12672, "projection result count changed")
    print(
        "PASS canonical_lora_v4_layer0_full_w4a8_rtl "
        f"manifest_members={members} projection_results={execution['projection_results']} "
        f"layer_output_sha256={execution['final_layer_output_sha256']} "
        f"max_dequant_error={result['merged_float_vs_w4a8_layer_output']['max_abs_dequantized_error']:.9f} "
        f"relative_l2={result['merged_float_vs_w4a8_layer_output']['relative_l2_error']:.9f}"
    )


def preflight() -> None:
    derived = derive()
    print(json.dumps({"status": "PASS_DERIVATION", "metrics": derived["metrics"], "token_id": derived["token_id"]}, sort_keys=True))


def rtl_preflight() -> None:
    global ATTEMPT
    preview_root = ROOT / "build/lora-v4-layer0-full-w4a8-rtl-preview"
    index = 1
    while (preview_root / f"attempt-{index:04d}").exists():
        index += 1
    ATTEMPT = preview_root / f"attempt-{index:04d}"
    ATTEMPT.mkdir(parents=True)
    derived = derive()
    vectors = persist_vectors(derived)
    rtl = run_rtl(derived)
    result = comparison(derived, rtl)
    write_json(ATTEMPT / "preview.json", {"status": result["status"], "comparison": result, "rtl": rtl, "vectors": vectors})
    write_sha256s(ATTEMPT)
    print(json.dumps({"status": result["status"], "preview": relative(ATTEMPT), "metrics": derived["metrics"]}, sort_keys=True))
    require(result["status"] == "PASS", "RTL preflight failed frozen candidate thresholds")


def tail_preflight() -> None:
    global ATTEMPT
    preview_root = ROOT / "build/lora-v4-layer0-full-w4a8-rtl-preview"
    index = 1
    while (preview_root / f"attempt-{index:04d}").exists():
        index += 1
    ATTEMPT = preview_root / f"attempt-{index:04d}"
    ATTEMPT.mkdir(parents=True)
    derived = derive()
    vectors = persist_vectors(derived)
    compose_source = execution_source("attention_compose", "attention_compose_vectors.svh")
    silu_source = execution_source("silu", "silu_gate_vectors.svh")
    residual_source = residual_execution_source()
    results = {
        "attention_compose": compile_and_run(
            "attention_compose", [CORES["attention_compose"], compose_source], "ace2_attention_compose_tb"
        ),
        "silu": compile_and_run("silu", [CORES["silu"], silu_source], "ace2_silu_gate_tb"),
    }
    rtl_sources = [ROOT / "rtl/ace2_pkg.sv"] + sorted(
        path for path in (ROOT / "rtl").glob("*.sv") if path.name != "ace2_pkg.sv"
    )
    results["residual"] = compile_and_run(
        "residual", [*rtl_sources, residual_source], "ace2_shell_tb", ["+LORA_V4_LAYER0_RESIDUAL_REPLAY"]
    )
    require("ACE2_ATTN_COMPOSE_TB_PASS cases=14" in results["attention_compose"]["stdout_text"], "compose PASS marker missing")
    require("ACE2_SILU_GATE_TB_PASS cases=1" in results["silu"]["stdout_text"], "SiLU PASS marker missing")
    require("ACE2_LORA_V4_LAYER0_RESIDUAL_REPLAY_PASS stages=2 beats=112" in results["residual"]["stdout_text"], "residual PASS marker missing")
    final_beats = re.findall(
        r"RTL_RESIDUAL_BEAT stage=final beat=(\d+) data=([0-9a-fA-F]+)",
        results["residual"]["stdout_text"],
    )
    require([int(index) for index, _ in final_beats] == list(range(56)), "tail final beat order changed")
    final_raw = b"".join(int(data, 16).to_bytes(16, "little") for _, data in final_beats)
    require(final_raw == raw_tensor_bytes(derived["final_residual"]["output"]), "tail final output differs")
    for value in results.values():
        value.pop("stdout_text", None)
    write_json(
        ATTEMPT / "preview.json",
        {
            "status": "PASS",
            "metrics": derived["metrics"],
            "final_layer_output_sha256": sha256_bytes(final_raw),
            "rtl": results,
            "vectors": vectors,
        },
    )
    write_sha256s(ATTEMPT)
    print(json.dumps({"status": "PASS", "preview": relative(ATTEMPT), "metrics": derived["metrics"]}, sort_keys=True))


def residual_preflight() -> None:
    global ATTEMPT
    preview_root = ROOT / "build/lora-v4-layer0-full-w4a8-rtl-preview"
    index = 1
    while (preview_root / f"attempt-{index:04d}").exists():
        index += 1
    ATTEMPT = preview_root / f"attempt-{index:04d}"
    ATTEMPT.mkdir(parents=True)
    derived = derive()
    vectors = persist_vectors(derived)
    residual_source = residual_execution_source()
    rtl_sources = [ROOT / "rtl/ace2_pkg.sv"] + sorted(
        path for path in (ROOT / "rtl").glob("*.sv") if path.name != "ace2_pkg.sv"
    )
    result = compile_and_run(
        "residual", [*rtl_sources, residual_source], "ace2_shell_tb", ["+LORA_V4_LAYER0_RESIDUAL_REPLAY"]
    )
    require("ACE2_LORA_V4_LAYER0_RESIDUAL_REPLAY_PASS stages=2 beats=112" in result["stdout_text"], "residual PASS marker missing")
    final_beats = re.findall(
        r"RTL_RESIDUAL_BEAT stage=final beat=(\d+) data=([0-9a-fA-F]+)", result["stdout_text"]
    )
    require([int(index) for index, _ in final_beats] == list(range(56)), "residual final beat order changed")
    final_raw = b"".join(int(data, 16).to_bytes(16, "little") for _, data in final_beats)
    require(final_raw == raw_tensor_bytes(derived["final_residual"]["output"]), "residual final output differs")
    result.pop("stdout_text", None)
    write_json(
        ATTEMPT / "preview.json",
        {
            "status": "PASS",
            "metrics": derived["metrics"],
            "final_layer_output_sha256": sha256_bytes(final_raw),
            "rtl": result,
            "vectors": vectors,
        },
    )
    write_sha256s(ATTEMPT)
    print(json.dumps({"status": "PASS", "preview": relative(ATTEMPT), "metrics": derived["metrics"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--rtl-preflight", action="store_true")
    mode.add_argument("--tail-preflight", action="store_true")
    mode.add_argument("--residual-preflight", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.rtl_preflight:
        rtl_preflight()
    elif args.tail_preflight:
        tail_preflight()
    elif args.residual_preflight:
        residual_preflight()
    elif args.prepare:
        prepare()
    elif args.run:
        run()
    else:
        check()


if __name__ == "__main__":
    main()
