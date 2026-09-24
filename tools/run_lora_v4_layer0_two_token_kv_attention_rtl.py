#!/usr/bin/env python3
"""Verify two-position layer-0 KV reuse through the ACE-2 attention RTL path.

This is a bounded attention-frontier increment.  It deliberately does not claim
the position-1 O projection, MLP, residual boundaries, or final layer output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_lora_v4_layer0_full_rtl as canonical
from ace2_attention_compose_reference import AttentionComposeCase, reference_attention_compose
from ace2_attention_score_reference import AttentionScoreCase, reference_attention_score
from ace2_projection_reference import ProjectionCase, reference_projection
from ace2_rmsnorm_reference import (
    derive_scaled_gains_q8,
    pack_gain_beats,
    pack_int8_beats,
    reference_rmsnorm,
)
from ace2_rope_reference import (
    Q15_ONE,
    Q9_SCALE_ONE,
    RopeCase,
    round_shift_even as rope_round_shift_even,
    reference_rope,
)
from ace2_softmax_reference import SoftmaxCase, reference_softmax


MISSION = "lora_v4_layer0_two_token_kv_attention_rtl"
OUT = ROOT / "evidence/verification/lora-v4-layer0-two-token-kv-attention-v1"
CONTRACT = OUT / "frozen_contract.json"
OFFICIAL = OUT / "attempt-0001"
PREVIEW_ROOT = ROOT / "build/lora-v4-layer0-two-token-kv-attention-preview"
ACCEPTED = ROOT / "evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1/attempt-0001"
ACCEPTED_CONTRACT = ACCEPTED.parent / "frozen_contract.json"
ERRATUM = ACCEPTED.parent / "errata/float-fidelity-post-preview-v1/erratum.json"

RMS_TB = ROOT / "verification/tb/ace2_rmsnorm_tb.sv"
PROJECTION_TB = ROOT / "verification/tb/ace2_lora_v4_layer0_projection_batch_tb.sv"
ROPE_TB = ROOT / "verification/tb/ace2_rope_tb.sv"
SCORE_TB = ROOT / "verification/tb/ace2_attention_score_tb.sv"
SOFTMAX_TB = ROOT / "verification/tb/ace2_softmax_tb.sv"
COMPOSE_TB = ROOT / "verification/tb/ace2_attention_compose_tb.sv"

CORES = {
    "rmsnorm": ROOT / "rtl/ace2_rmsnorm_core.sv",
    "projection": ROOT / "rtl/ace2_w4a8_proj_core.sv",
    "rope": ROOT / "rtl/ace2_rope_core.sv",
    "score": ROOT / "rtl/ace2_attention_score_core.sv",
    "softmax": ROOT / "rtl/ace2_softmax_core.sv",
    "compose": ROOT / "rtl/ace2_attention_compose_core.sv",
}

HIDDEN = 896
Q_HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64
ROPE_THETA = 1_000_000.0
PROJECTION_ORDER = ("q", "k", "v")
PROJECTION_CHANNELS = {"q": 896, "k": 128, "v": 128}
PROMPT_ALIAS = "canonical_lora_v4_layer0_adaptation_prompt"

ACTIVE_ATTEMPT = OFFICIAL


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


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "path": relative(path), "sha256": sha256_file(path)}


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


def raw_int8(values: Iterable[int]) -> bytes:
    return bytes(int(value) & 0xFF for value in values)


def raw_int32(values: Iterable[int]) -> bytes:
    return b"".join((int(value) & 0xFFFF_FFFF).to_bytes(4, "little") for value in values)


def load_s8(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    return torch.tensor([value - 256 if value >= 128 else value for value in raw], dtype=torch.int8)


def load_s32(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    require(len(raw) % 4 == 0, f"unaligned int32 tensor: {path}")
    values = [int.from_bytes(raw[index:index + 4], "little", signed=True) for index in range(0, len(raw), 4)]
    return torch.tensor(values, dtype=torch.int64)


def read_hex(path: Path, width: int, *, signed: bool = False) -> list[int]:
    values = [int(line.strip(), 16) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if signed:
        sign = 1 << (width - 1)
        modulus = 1 << width
        values = [value - modulus if value & sign else value for value in values]
    return values


def write_hex(path: Path, values: Iterable[int], width: int) -> dict[str, Any]:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    return write_text(path, "".join(f"{int(value) & mask:0{digits}x}\n" for value in values))


def chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def pack(values: Iterable[int], width: int) -> int:
    result = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        result |= (int(value) & mask) << (lane * width)
    return result


def hex_literal(value: int, width: int) -> str:
    return f"{width}'h{value & ((1 << width) - 1):0{width // 4}x}"


def next_preview() -> Path:
    index = 1
    while (PREVIEW_ROOT / f"attempt-{index:04d}").exists():
        index += 1
    return PREVIEW_ROOT / f"attempt-{index:04d}"


def rope_coefficients(position: int, heads: int) -> tuple[list[int], list[int]]:
    frequencies = [ROPE_THETA ** (-(2.0 * index) / HEAD_DIM) for index in range(HEAD_DIM // 2)]
    cosine_half = [max(-32768, min(32767, round(math.cos(position * value) * (1 << 15)))) for value in frequencies]
    sine_half = [max(-32768, min(32767, round(math.sin(position * value) * (1 << 15)))) for value in frequencies]
    one_head_cos = cosine_half + cosine_half
    one_head_sin = sine_half + sine_half
    return one_head_cos * heads, one_head_sin * heads


def rope_saturation_by_beat(inputs: list[int], cosines: list[int], sines: list[int]) -> list[bool]:
    q9 = [int(value) * Q9_SCALE_ONE for value in inputs]
    saturated: list[bool] = []
    for beat in range(HIDDEN // 16):
        beat_saturated = False
        for index in range(beat * 16, (beat + 1) * 16):
            head_base = (index // HEAD_DIM) * HEAD_DIM
            dim = index % HEAD_DIM
            pair_index = head_base + dim + 32 if dim < 32 else head_base + dim - 32
            if dim < 32:
                rotated = q9[index] * cosines[index] - q9[pair_index] * sines[index]
            else:
                rotated = q9[index] * cosines[index] + q9[pair_index] * sines[index]
            rounded = rope_round_shift_even(rotated, 24)
            beat_saturated = beat_saturated or rounded < -128 or rounded > 127
        saturated.append(beat_saturated)
    return saturated


def projection_from_fixed_metadata(
    name: str,
    input_q: torch.Tensor,
    input_scale: float,
    qweight: torch.Tensor,
    multiplier: torch.Tensor,
    right_shift: torch.Tensor,
    output_scale: float,
) -> dict[str, Any]:
    accumulator = torch.mv(qweight.to(torch.int64), input_q.to(torch.int64))
    require(bool(torch.all(accumulator >= -(1 << 31))) and bool(torch.all(accumulator < (1 << 31))), f"{name} accumulator overflow")
    rounded = torch.tensor(
        [
            canonical.round_shift_even(int(acc), int(shift))
            for acc, shift in zip((accumulator * multiplier).tolist(), right_shift.tolist(), strict=True)
        ],
        dtype=torch.int64,
    )
    output = rounded.clamp(-128, 127).to(torch.int8)
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
            output_zero_points=[0] * output.numel(),
            bias_accumulators=[0] * output.numel(),
        )
    )
    require(independent.outputs == [output.to(torch.int64).tolist()], f"{name} independent projection mismatch")
    return {
        "name": name,
        "input_q": input_q,
        "input_scale": input_scale,
        "qweight": qweight,
        "multiplier": multiplier,
        "right_shift": right_shift,
        "accumulator": accumulator,
        "output_q": output,
        "output_scale": output_scale,
        "saturation": saturation,
    }


def derive() -> dict[str, Any]:
    provenance = canonical.validate_provenance()
    token_ids = provenance["token_ids"]
    require(len(token_ids) >= 2, "canonical tokenizer sequence no longer has two positions")
    conversion = json.loads((ACCEPTED / "conversion.json").read_text(encoding="utf-8"))
    input_scale = float(conversion["scales"]["input_rmsnorm"])

    multiplier_all = read_hex(ACCEPTED / "vectors/projection_multiplier.hex", 32, signed=True)
    shift_all = read_hex(ACCEPTED / "vectors/projection_shift.hex", 8)
    offsets = {"q": 0, "k": 896, "v": 1024}

    accepted_qweights = {
        key: load_s8(ACCEPTED / f"tensors/{key}_qweight_s4_in_s8.bin").reshape(PROJECTION_CHANNELS[key], HIDDEN)
        for key in PROJECTION_ORDER
    }
    accepted_outputs = {key: load_s8(ACCEPTED / f"tensors/{key}_output_s8.bin") for key in PROJECTION_ORDER}
    accepted_accumulators = {key: load_s32(ACCEPTED / f"tensors/{key}_accumulator_s32le.bin") for key in PROJECTION_ORDER}

    with safe_open(canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(canonical.ADAPTER, framework="pt", device="cpu") as adapter:
        input_gain = weights.get_tensor("model.layers.0.input_layernorm.weight").contiguous()
        fixed_gains = derive_scaled_gains_q8(input_gain.to(torch.float64).tolist(), input_scale)
        merged: dict[str, torch.Tensor] = {}
        source_hashes: dict[str, dict[str, str]] = {}
        metadata: dict[str, dict[str, Any]] = {}
        for key, tensor_name in (
            ("q", "model.layers.0.self_attn.q_proj"),
            ("k", "model.layers.0.self_attn.k_proj"),
            ("v", "model.layers.0.self_attn.v_proj"),
        ):
            merged[key], source_hashes[key] = canonical.merge_projection(weights, adapter, tensor_name)
            require(source_hashes[key] == conversion["projections"][key]["source_hashes"], f"{key} merged checkpoint source changed")
            weight = merged[key].to(torch.float64)
            weight_scale = weight.abs().amax(dim=1) / 7.0
            weight_scale = torch.where(weight_scale > 0, weight_scale, torch.ones_like(weight_scale))
            independently_quantized = torch.round(weight / weight_scale[:, None]).clamp(-8, 7).to(torch.int8)
            require(torch.equal(independently_quantized, accepted_qweights[key]), f"{key} accepted W4 tensor changed")
            output_scale = float(conversion["projections"][key]["output_scale"])
            multiplier, right_shift = canonical.derive_multiplier(input_scale * weight_scale / output_scale)
            start = offsets[key]
            stop = start + PROJECTION_CHANNELS[key]
            require(multiplier.tolist() == multiplier_all[start:stop], f"{key} multiplier metadata changed")
            require(right_shift.tolist() == shift_all[start:stop], f"{key} shift metadata changed")
            metadata[key] = {
                "qweight": accepted_qweights[key],
                "multiplier": multiplier,
                "right_shift": right_shift,
                "output_scale": output_scale,
            }

        positions: list[dict[str, Any]] = []
        for position in (0, 1):
            embedding = weights.get_tensor("model.embed_tokens.weight")[token_ids[position]].contiguous()
            embedding_scale = canonical.scale_for(embedding.to(torch.float32))
            embedding_q = canonical.quantize_int8(embedding, embedding_scale)
            input_norm = reference_rmsnorm(embedding_q.to(torch.int64).tolist(), fixed_gains)
            input_norm_q = torch.tensor(input_norm.outputs, dtype=torch.int8)
            projections = {
                key: projection_from_fixed_metadata(
                    f"position{position}_{key}",
                    input_norm_q,
                    input_scale,
                    metadata[key]["qweight"],
                    metadata[key]["multiplier"],
                    metadata[key]["right_shift"],
                    metadata[key]["output_scale"],
                )
                for key in PROJECTION_ORDER
            }
            if position == 0:
                require(torch.equal(input_norm_q, load_s8(ACCEPTED / "tensors/input_rmsnorm_s8.bin")), "position-0 RMSNorm differs from accepted attempt")
                for key in PROJECTION_ORDER:
                    require(torch.equal(projections[key]["output_q"], accepted_outputs[key]), f"position-0 {key} output differs from accepted attempt")
                    require(torch.equal(projections[key]["accumulator"], accepted_accumulators[key]), f"position-0 {key} accumulator differs from accepted attempt")

            q_values = projections["q"]["output_q"].to(torch.int64).tolist()
            k_values = projections["k"]["output_q"].to(torch.int64).tolist()
            q_cos, q_sin = rope_coefficients(position, Q_HEADS)
            k_cos, k_sin = rope_coefficients(position, KV_HEADS)
            rope_q = reference_rope(RopeCase(f"position{position}_q", position, q_values, [Q9_SCALE_ONE] * HIDDEN, q_cos, q_sin))
            rope_k = reference_rope(RopeCase(f"position{position}_k", position, k_values, [Q9_SCALE_ONE] * (KV_HEADS * HEAD_DIM), k_cos, k_sin))
            if position == 0:
                require(rope_q.outputs == q_values and rope_k.outputs == k_values, "position-0 RoPE is not identity")
            else:
                require(any(value != source for value, source in zip(rope_q.outputs, q_values, strict=True)), "position-1 Q RoPE remained identity")
                require(any(value != source for value, source in zip(rope_k.outputs, k_values, strict=True)), "position-1 K RoPE remained identity")
            positions.append({
                "position": position,
                "token_id": token_ids[position],
                "embedding_q": embedding_q,
                "input_gains": fixed_gains,
                "input_norm": input_norm,
                "input_norm_q": input_norm_q,
                "projections": projections,
                "rope_q": rope_q,
                "rope_k": rope_k,
                "rope_q_cos": q_cos,
                "rope_q_sin": q_sin,
                "rope_k_cos": k_cos,
                "rope_k_sin": k_sin,
            })

    cached_k = [positions[0]["rope_k"].outputs, positions[1]["rope_k"].outputs]
    cached_v = [
        positions[0]["projections"]["v"]["output_q"].to(torch.int64).tolist(),
        positions[1]["projections"]["v"]["output_q"].to(torch.int64).tolist(),
    ]
    nonzero_cached_probability_heads = 0
    cached_value_effect_heads = 0
    for position in positions:
        scores = []
        probabilities = []
        composes = []
        attention_heads: list[int] = []
        context = position["position"] + 1
        for head in range(Q_HEADS):
            kv_head = head // (Q_HEADS // KV_HEADS)
            q_head = position["rope_q"].outputs[head * HEAD_DIM:(head + 1) * HEAD_DIM]
            keys = [cached_k[token][kv_head * HEAD_DIM:(kv_head + 1) * HEAD_DIM] for token in range(context)]
            values = [cached_v[token][kv_head * HEAD_DIM:(kv_head + 1) * HEAD_DIM] for token in range(context)]
            score = reference_attention_score(
                AttentionScoreCase(
                    f"position{position['position']}_head{head}",
                    q_head,
                    keys,
                    positions[0]["projections"]["q"]["output_scale"],
                    positions[0]["projections"]["k"]["output_scale"],
                )
            )
            softmax = reference_softmax(SoftmaxCase(f"position{position['position']}_head{head}", score.core_scores_q6_9))
            compose = reference_attention_compose(AttentionComposeCase(f"position{position['position']}_head{head}", score.core_scores_q6_9, values))
            if position["position"] == 1:
                if softmax.probabilities_q0_15[0] != 0:
                    nonzero_cached_probability_heads += 1
                zero_cached = reference_attention_compose(
                    AttentionComposeCase(
                        f"position1_head{head}_zero_cached_v",
                        score.core_scores_q6_9,
                        [[0] * HEAD_DIM, values[1]],
                    )
                )
                if zero_cached.outputs != compose.outputs:
                    cached_value_effect_heads += 1
            scores.append(score)
            probabilities.append(softmax)
            composes.append(compose)
            attention_heads.extend(compose.outputs)
        position["scores"] = scores
        position["probabilities"] = probabilities
        position["composes"] = composes
        position["attention_q"] = torch.tensor(attention_heads, dtype=torch.int8)

    require(nonzero_cached_probability_heads > 0, "position-0 cache received zero probability in every decode head")
    require(cached_value_effect_heads > 0, "zeroing cached position-0 V did not alter any decode head")
    return {
        "token_ids": token_ids,
        "positions": positions,
        "cached_k": cached_k,
        "cached_v": cached_v,
        "source_hashes": source_hashes,
        "cache_reuse": {
            "decode_heads": Q_HEADS,
            "cached_position0_probability_nonzero_heads": nonzero_cached_probability_heads,
            "compose_output_changes_when_cached_v_zeroed_heads": cached_value_effect_heads,
        },
    }


def render_rmsnorm_vectors(derived: dict[str, Any]) -> str:
    lines = [
        "// Tokenizer-derived positions 0 and 1 with frozen canonical layer-0 gain metadata.",
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
    for case, position in enumerate(derived["positions"]):
        result = position["input_norm"]
        lines += [
            f"  test_expected_sumsq[{case}] = {hex_literal(result.sumsq, 48)};",
            f"  test_expected_inv[{case}] = {hex_literal(result.inv_rms_q30, 32)};",
            f"  test_expected_saturation[{case}] = 1'b{int(result.saturation_seen)};",
        ]
        for beat, (input_word, gain_word, output_word) in enumerate(zip(
            pack_int8_beats(position["embedding_q"].to(torch.int64).tolist()),
            pack_gain_beats(position["input_gains"]),
            pack_int8_beats(result.outputs),
            strict=True,
        )):
            flat = case * 56 + beat
            lines += [
                f"  test_input_beats[{flat}] = {hex_literal(input_word, 128)};",
                f"  test_gain_beats[{flat}] = {hex_literal(gain_word, 256)};",
                f"  test_expected_beats[{flat}] = {hex_literal(output_word, 128)};",
            ]
    return "\n".join(lines + ["end", ""])


def render_rope_vectors(derived: dict[str, Any]) -> str:
    cases: list[tuple[int, list[int], list[int], list[int], list[int], Any]] = []
    for position in derived["positions"]:
        cases.append((position["position"], position["projections"]["q"]["output_q"].to(torch.int64).tolist(), position["rope_q_cos"], position["rope_q_sin"], position["rope_q"].outputs, position["rope_q"]))
        k_input = position["projections"]["k"]["output_q"].to(torch.int64).tolist() + [0] * (HIDDEN - KV_HEADS * HEAD_DIM)
        k_cos = position["rope_k_cos"] + [Q15_ONE] * (HIDDEN - KV_HEADS * HEAD_DIM)
        k_sin = position["rope_k_sin"] + [0] * (HIDDEN - KV_HEADS * HEAD_DIM)
        k_output = position["rope_k"].outputs + [0] * (HIDDEN - KV_HEADS * HEAD_DIM)
        cases.append((position["position"], k_input, k_cos, k_sin, k_output, position["rope_k"]))
    lines = [
        "// Position-0 identity and non-identity position-1 Q/K RoPE.",
        "localparam integer ROPE_CASE_COUNT = 4;",
        "localparam integer ROPE_HIDDEN_SIZE = 896;",
        "localparam integer ROPE_LANES = 16;",
        "localparam integer ROPE_BEATS = 56;",
        "reg [15:0] rope_sequence_position [0:ROPE_CASE_COUNT-1];",
        "reg rope_expected_saturation [0:ROPE_CASE_COUNT-1];",
        "reg rope_expected_saturation_by_beat [0:ROPE_CASE_COUNT*ROPE_BEATS-1];",
        "reg [127:0] rope_input_beats [0:ROPE_CASE_COUNT*ROPE_BEATS-1];",
        "reg [127:0] rope_scale_beats [0:ROPE_CASE_COUNT*ROPE_BEATS*2-1];",
        "reg [127:0] rope_cos_beats [0:ROPE_CASE_COUNT*ROPE_BEATS*2-1];",
        "reg [127:0] rope_sin_beats [0:ROPE_CASE_COUNT*ROPE_BEATS*2-1];",
        "reg [127:0] rope_expected_beats [0:ROPE_CASE_COUNT*ROPE_BEATS-1];",
        "initial begin",
    ]
    for case, (sequence_position, inputs, cosines, sines, outputs, result) in enumerate(cases):
        beat_saturation = rope_saturation_by_beat(inputs, cosines, sines)
        require(any(beat_saturation) == result.saturation_seen, f"RoPE saturation reduction changed for case {case}")
        lines += [f"  rope_sequence_position[{case}] = 16'd{sequence_position};", f"  rope_expected_saturation[{case}] = 1'b{int(result.saturation_seen)};"]
        for beat in range(56):
            flat = case * 56 + beat
            input_group = inputs[beat * 16:(beat + 1) * 16]
            output_group = outputs[beat * 16:(beat + 1) * 16]
            lines += [
                f"  rope_input_beats[{flat}] = {hex_literal(pack(input_group, 8), 128)};",
                f"  rope_expected_beats[{flat}] = {hex_literal(pack(output_group, 8), 128)};",
                f"  rope_expected_saturation_by_beat[{flat}] = 1'b{int(beat_saturation[beat])};",
            ]
            for half in range(2):
                table = case * 112 + beat * 2 + half
                start = beat * 16 + half * 8
                lines += [
                    f"  rope_scale_beats[{table}] = {hex_literal(pack([Q9_SCALE_ONE] * 8, 16), 128)};",
                    f"  rope_cos_beats[{table}] = {hex_literal(pack(cosines[start:start + 8], 16), 128)};",
                    f"  rope_sin_beats[{table}] = {hex_literal(pack(sines[start:start + 8], 16), 128)};",
                ]
    return "\n".join(lines + ["end", ""])


def attention_cases(derived: dict[str, Any]) -> list[tuple[dict[str, Any], int, Any, Any, Any]]:
    return [
        (position, head, position["scores"][head], position["probabilities"][head], position["composes"][head])
        for position in derived["positions"]
        for head in range(Q_HEADS)
    ]


def render_score_vectors(derived: dict[str, Any]) -> str:
    cases = attention_cases(derived)
    lines = [
        "// Position-0 prefill and position-1 two-entry causal score vectors.",
        "localparam integer ATTN_SCORE_CASE_COUNT = 28;",
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
    for case, (position, head, score, _, _) in enumerate(cases):
        context = position["position"] + 1
        kv_head = head // 7
        q_head = position["rope_q"].outputs[head * 64:(head + 1) * 64]
        lines += [
            f"  attn_score_context_count[{case}] = 16'd{context};",
            f"  attn_score_multiplier[{case}] = 32'sd{score.multiplier};",
            f"  attn_score_right_shift[{case}] = 6'd{score.right_shift};",
            f"  attn_score_expected_saturation[{case}] = 1'b{int(score.saturation_seen)};",
            f"  attn_score_expected_word[{case}] = {hex_literal(pack(score.scores_q6_9 + [0] * (8 - context), 16), 128)};",
            f"  attn_score_expected_core_word[{case}] = {hex_literal(pack(score.core_scores_q6_9 + [0] * (8 - context), 16), 128)};",
        ]
        for beat in range(4):
            lines.append(f"  attn_score_q_beats[{case * 4 + beat}] = {hex_literal(pack(q_head[beat * 16:(beat + 1) * 16], 8), 128)};")
        for token in range(8):
            flat = case * 8 + token
            real = token < context
            lines += [
                f"  attn_score_expected_acc[{flat}] = {hex_literal(score.accumulators[token] if real else 0, 32)};",
                f"  attn_score_expected_saturation_token[{flat}] = 1'b{int(score.saturation_by_token[token]) if real else 0};",
                f"  attn_score_expected_core_saturation_token[{flat}] = 1'b{int(score.core_saturation_by_token[token]) if real else 0};",
            ]
            key = derived["cached_k"][token][kv_head * 64:(kv_head + 1) * 64] if real else [0] * 64
            for beat in range(4):
                lines.append(f"  attn_score_k_beats[{case * 32 + token * 4 + beat}] = {hex_literal(pack(key[beat * 16:(beat + 1) * 16], 8), 128)};")
    return "\n".join(lines + ["end", ""])


def render_softmax_vectors(derived: dict[str, Any]) -> str:
    cases = attention_cases(derived)
    lines = [
        "// Position-0 singleton and position-1 two-entry softmax vectors.",
        "localparam integer SOFTMAX_CASE_COUNT = 28;",
        "localparam integer SOFTMAX_CONTEXT_MAX = 8;",
        "reg [15:0] softmax_context_count [0:SOFTMAX_CASE_COUNT-1];",
        "reg [127:0] softmax_score_word [0:SOFTMAX_CASE_COUNT-1];",
        "reg [127:0] softmax_expected_exp_word [0:SOFTMAX_CASE_COUNT-1];",
        "reg [18:0] softmax_expected_exp_sum [0:SOFTMAX_CASE_COUNT-1];",
        "reg [127:0] softmax_expected_word [0:SOFTMAX_CASE_COUNT-1];",
        "initial begin",
    ]
    for case, (position, _, score, softmax, _) in enumerate(cases):
        context = position["position"] + 1
        lines += [
            f"  softmax_context_count[{case}] = 16'd{context};",
            f"  softmax_score_word[{case}] = {hex_literal(pack(score.core_scores_q6_9 + [0] * (8 - context), 16), 128)};",
            f"  softmax_expected_exp_word[{case}] = {hex_literal(pack(softmax.exp_weights_q15, 16), 128)};",
            f"  softmax_expected_exp_sum[{case}] = 19'd{softmax.exp_sum_q15};",
            f"  softmax_expected_word[{case}] = {hex_literal(pack(softmax.probabilities_q0_15, 16), 128)};",
        ]
    return "\n".join(lines + ["end", ""])


def render_compose_vectors(derived: dict[str, Any]) -> str:
    cases = attention_cases(derived)
    lines = [
        "// Two protocol tiles: one/two real causal entries plus exact-zero padding.",
        "localparam integer ATTN_COMPOSE_CASE_COUNT = 28;",
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
    for case, (position, head, score, _, compose) in enumerate(cases):
        context = position["position"] + 1
        kv_head = head // 7
        real_values = [derived["cached_v"][token][kv_head * 64:(kv_head + 1) * 64] for token in range(context)]
        protocol_scores = score.core_scores_q6_9 + [-32768] * (9 - context)
        protocol_values = real_values + [[0] * 64 for _ in range(9 - context)]
        protocol_compose = reference_attention_compose(AttentionComposeCase(f"protocol_case{case}", protocol_scores, protocol_values))
        require(protocol_compose.outputs == compose.outputs, f"compose protocol padding changed case {case}")
        lines += [
            f"  attn_compose_context_count[{case}] = 16'd9;",
            f"  attn_compose_score_tiles[{case * 2}] = {hex_literal(pack(protocol_scores[:8], 16), 128)};",
            f"  attn_compose_score_tiles[{case * 2 + 1}] = {hex_literal(pack(protocol_scores[8:] + [0] * 7, 16), 128)};",
            f"  attn_compose_expected_saturation[{case}] = 1'b{int(protocol_compose.saturation_seen)};",
        ]
        for token, row in enumerate(protocol_values):
            for beat in range(4):
                flat = case * 36 + token * 4 + beat
                lines.append(f"  attn_compose_value_beats[{flat}] = {hex_literal(pack(row[beat * 16:(beat + 1) * 16], 8), 128)};")
        for beat in range(4):
            lines.append(f"  attn_compose_expected_beats[{case * 4 + beat}] = {hex_literal(pack(compose.outputs[beat * 16:(beat + 1) * 16], 8), 128)};")
    return "\n".join(lines + ["end", ""])


def render_projection_source() -> str:
    source = PROJECTION_TB.read_text(encoding="utf-8")
    source = source.replace("localparam integer CASES = 7;", "localparam integer CASES = 6;")
    source = source.replace("localparam integer TOTAL_INPUT_GROUPS = 2560;", "localparam integer TOTAL_INPUT_GROUPS = 1344;")
    source = source.replace("localparam integer TOTAL_WEIGHT_GROUPS = 3727360;", "localparam integer TOTAL_WEIGHT_GROUPS = 258048;")
    source = source.replace("localparam integer TOTAL_OUTPUTS = 12672;", "localparam integer TOTAL_OUTPUTS = 2304;")
    old = re.search(r"    task select_case;.*?    endtask\n", source, flags=re.S)
    require(old is not None, "projection select_case block changed")
    replacement = """    task select_case;
        input integer selected_case;
        begin
            case (selected_case)
                0: begin groups = 224; outputs = 896; input_offset = 0;    weight_offset = 0;      output_offset = 0;    end
                1: begin groups = 224; outputs = 128; input_offset = 224;  weight_offset = 200704; output_offset = 896;  end
                2: begin groups = 224; outputs = 128; input_offset = 448;  weight_offset = 229376; output_offset = 1024; end
                3: begin groups = 224; outputs = 896; input_offset = 672;  weight_offset = 0;      output_offset = 1152; end
                4: begin groups = 224; outputs = 128; input_offset = 896;  weight_offset = 200704; output_offset = 2048; end
                default: begin groups = 224; outputs = 128; input_offset = 1120; weight_offset = 229376; output_offset = 2176; end
            endcase
        end
    endtask
"""
    source = source[:old.start()] + replacement + source[old.end():]
    frozen = "evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1/attempt-0001/vectors"
    source = source.replace(frozen, relative(ACTIVE_ATTEMPT / "vectors"))
    return source


def persist_vectors(derived: dict[str, Any]) -> dict[str, Any]:
    directory = ACTIVE_ATTEMPT / "vectors"
    activation_words: list[int] = []
    multipliers: list[int] = []
    shifts: list[int] = []
    outputs: list[int] = []
    accumulators: list[int] = []
    saturation: list[int] = []
    for position in derived["positions"]:
        for key in PROJECTION_ORDER:
            projection = position["projections"][key]
            activation_words.extend(pack(group, 8) for group in chunks(projection["input_q"].to(torch.int64).tolist(), 4))
            multipliers.extend(projection["multiplier"].tolist())
            shifts.extend(projection["right_shift"].tolist())
            outputs.extend(projection["output_q"].to(torch.int64).tolist())
            accumulators.extend(projection["accumulator"].tolist())
            saturation.extend(projection["saturation"].to(torch.int64).tolist())
    weight_words: list[int] = []
    for key in PROJECTION_ORDER:
        for row in derived["positions"][0]["projections"][key]["qweight"].to(torch.int64):
            weight_words.extend(pack(group, 4) for group in chunks(row.tolist(), 4))
    return {
        "projection_activation": write_hex(directory / "projection_activation.hex", activation_words, 32),
        "projection_weight": write_hex(directory / "projection_weight.hex", weight_words, 16),
        "projection_multiplier": write_hex(directory / "projection_multiplier.hex", multipliers, 32),
        "projection_shift": write_hex(directory / "projection_shift.hex", shifts, 8),
        "projection_expected": write_hex(directory / "projection_expected.hex", outputs, 8),
        "projection_accumulator": write_hex(directory / "projection_accumulator.hex", accumulators, 32),
        "projection_saturation": write_hex(directory / "projection_saturation.hex", saturation, 8),
        "rmsnorm": write_text(directory / "rmsnorm_vectors.svh", render_rmsnorm_vectors(derived)),
        "rope": write_text(directory / "rope_vectors.svh", render_rope_vectors(derived)),
        "score": write_text(directory / "attention_score_vectors.svh", render_score_vectors(derived)),
        "softmax": write_text(directory / "softmax_vectors.svh", render_softmax_vectors(derived)),
        "compose": write_text(directory / "attention_compose_vectors.svh", render_compose_vectors(derived)),
    }


def execution_source(name: str, source_path: Path, generated_name: str) -> Path:
    source = source_path.read_text(encoding="utf-8")
    source = source.replace(f'`include "../generated/{generated_name}"', f'`include "{relative(ACTIVE_ATTEMPT / "vectors" / generated_name)}"', 1)
    if name == "rope":
        source = source.replace(
            "saturation_seen !== rope_expected_saturation[selected_case]",
            "saturation_seen !== rope_expected_saturation_by_beat[selected_case*ROPE_BEATS + selected_beat]",
            1,
        ).replace(
            "saturation_seen, rope_expected_saturation[selected_case]",
            "saturation_seen, rope_expected_saturation_by_beat[selected_case*ROPE_BEATS + selected_beat]",
            1,
        )
    path = ACTIVE_ATTEMPT / "execution_sources" / f"{name}_tb.sv"
    write_text(path, source)
    return path


def run_process(name: str, command: list[str]) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    elapsed = time.monotonic() - started
    logs = ACTIVE_ATTEMPT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (logs / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    write_json(logs / f"{name}.result.json", {"command": command, "elapsed_seconds": elapsed, "returncode": completed.returncode})
    return completed


def compile_and_run(name: str, core: Path, source: Path, top: str) -> dict[str, Any]:
    binary = ACTIVE_ATTEMPT / "sim" / f"{name}.vvp"
    binary.parent.mkdir(parents=True, exist_ok=True)
    compile_command = ["iverilog", "-g2012", "-I.", "-Irtl", "-Irtl/generated", "-Iverification/tb", "-s", top, "-o", relative(binary), relative(core), relative(source)]
    compiled = run_process(f"{name}.iverilog", compile_command)
    require(compiled.returncode == 0, f"{name} compile failed")
    simulate_command = ["vvp", relative(binary)]
    simulated = run_process(f"{name}.vvp", simulate_command)
    require(simulated.returncode == 0, f"{name} simulation failed")
    return {
        "compile_command": compile_command,
        "simulate_command": simulate_command,
        "binary": file_record(binary),
        "stdout": file_record(ACTIVE_ATTEMPT / f"logs/{name}.vvp.stdout.log"),
        "stderr": file_record(ACTIVE_ATTEMPT / f"logs/{name}.vvp.stderr.log"),
        "stdout_text": simulated.stdout,
    }


def run_rtl() -> dict[str, Any]:
    projection_source = ACTIVE_ATTEMPT / "execution_sources/projection_tb.sv"
    write_text(projection_source, render_projection_source())
    sources = {
        "rmsnorm": execution_source("rmsnorm", RMS_TB, "rmsnorm_vectors.svh"),
        "rope": execution_source("rope", ROPE_TB, "rope_vectors.svh"),
        "score": execution_source("score", SCORE_TB, "attention_score_vectors.svh"),
        "softmax": execution_source("softmax", SOFTMAX_TB, "softmax_vectors.svh"),
        "compose": execution_source("compose", COMPOSE_TB, "attention_compose_vectors.svh"),
    }
    results = {
        "rmsnorm": compile_and_run("rmsnorm", CORES["rmsnorm"], sources["rmsnorm"], "ace2_rmsnorm_tb"),
        "projection": compile_and_run("projection", CORES["projection"], projection_source, "ace2_lora_v4_layer0_projection_batch_tb"),
        "rope": compile_and_run("rope", CORES["rope"], sources["rope"], "ace2_rope_tb"),
        "score": compile_and_run("score", CORES["score"], sources["score"], "ace2_attention_score_tb"),
        "softmax": compile_and_run("softmax", CORES["softmax"], sources["softmax"], "ace2_softmax_tb"),
        "compose": compile_and_run("compose", CORES["compose"], sources["compose"], "ace2_attention_compose_tb"),
    }
    markers = {
        "rmsnorm": "ACE2_RMSNORM_TB_PASS cases=2 beats_per_case=56",
        "projection": "ACE2_LORA_V4_LAYER0_PROJECTION_BATCH_PASS operators=6 channels=2304",
        "rope": "ACE2_ROPE_TB_PASS cases=4 beats_per_case=56",
        "score": "ACE2_ATTN_SCORE_TB_PASS cases=28 context_max=8",
        "softmax": "ACE2_SOFTMAX_TB_PASS cases=28 context_max=8",
        "compose": "ACE2_ATTN_COMPOSE_TB_PASS cases=28",
    }
    for name, marker in markers.items():
        require(marker in results[name]["stdout_text"], f"{name} PASS marker missing")
    projection_rows = re.findall(r"RTL_PROJ_RESULT operator=(\d+) channel=(\d+) out_u8=(\d+) acc=(-?\d+)", results["projection"]["stdout_text"])
    require(len(projection_rows) == 2304, "projection result count changed")
    for value in results.values():
        value.pop("stdout_text", None)
    return {"executions": results, "projection_results": len(projection_rows)}


def persist_tensors(derived: dict[str, Any]) -> dict[str, Any]:
    directory = ACTIVE_ATTEMPT / "tensors"
    records: dict[str, Any] = {}
    for position in derived["positions"]:
        prefix = f"position{position['position']}"
        records[f"{prefix}_input_rmsnorm_s8.bin"] = write_binary(directory / f"{prefix}_input_rmsnorm_s8.bin", raw_int8(position["input_norm_q"].tolist()))
        records[f"{prefix}_rope_q_s8.bin"] = write_binary(directory / f"{prefix}_rope_q_s8.bin", raw_int8(position["rope_q"].outputs))
        records[f"{prefix}_rope_k_s8.bin"] = write_binary(directory / f"{prefix}_rope_k_s8.bin", raw_int8(position["rope_k"].outputs))
        records[f"{prefix}_attention_s8.bin"] = write_binary(directory / f"{prefix}_attention_s8.bin", raw_int8(position["attention_q"].tolist()))
        for key in PROJECTION_ORDER:
            projection = position["projections"][key]
            records[f"{prefix}_{key}_accumulator_s32le.bin"] = write_binary(directory / f"{prefix}_{key}_accumulator_s32le.bin", raw_int32(projection["accumulator"].tolist()))
            records[f"{prefix}_{key}_output_s8.bin"] = write_binary(directory / f"{prefix}_{key}_output_s8.bin", raw_int8(projection["output_q"].tolist()))
    records["kv_cache_k_s8.bin"] = write_binary(directory / "kv_cache_k_s8.bin", raw_int8(derived["cached_k"][0] + derived["cached_k"][1]))
    records["kv_cache_v_s8.bin"] = write_binary(directory / "kv_cache_v_s8.bin", raw_int8(derived["cached_v"][0] + derived["cached_v"][1]))
    return records


def comparison(derived: dict[str, Any], rtl: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "PASS_EXACT_TWO_POSITION_ATTENTION_FRONTIER",
        "claim_boundary": "positions 0 and 1 through input RMSNorm, Q/K/V projections, RoPE, causal score, softmax, and attention compose only",
        "zero_mismatch_policy": True,
        "rtl_vs_independent_python_fixed_point": {
            "rmsnorm_output_mismatches": 0,
            "projection_accumulator_mismatches": 0,
            "projection_output_mismatches": 0,
            "rope_output_mismatches": 0,
            "attention_score_accumulator_mismatches": 0,
            "attention_score_output_mismatches": 0,
            "softmax_probability_mismatches": 0,
            "attention_compose_output_mismatches": 0,
            "kv_cache_tensor_mismatches": 0,
            "projection_results": rtl["projection_results"],
        },
        "cache_reuse": derived["cache_reuse"],
        "position1_rope_non_identity": True,
        "float_fidelity": {
            "quality_pass_claimed": False,
            "reason": "the prior 0.25/0.40 limits were selected after preview and are descriptive only",
            "provenance_erratum": file_record(ERRATUM),
        },
        "remaining_for_full_mission": [
            "position-1 O projection",
            "position-1 attention residual",
            "position-1 post-attention RMSNorm",
            "position-1 gate/up/SiLU/down MLP",
            "position-1 final residual and 896-byte layer output",
            "combined two-position full-layer immutable official attempt",
        ],
    }


def write_sha256s(base: Path) -> None:
    members = [path for path in sorted(base.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"]
    write_text(base / "SHA256SUMS", "".join(f"{sha256_file(path)}  {path.relative_to(base).as_posix()}\n" for path in members))


def verify_sha256s(base: Path) -> int:
    lines = (base / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in lines:
        digest, name = line.split("  ", 1)
        require(sha256_file(base / name) == digest, f"immutable member changed: {name}")
    return len(lines)


def ensure_erratum() -> None:
    if ERRATUM.exists():
        return
    write_json(ERRATUM, {
        "schema_version": 1,
        "kind": "non_mutating_provenance_erratum",
        "applies_to": relative(ACCEPTED),
        "accepted_exact_rtl_fixed_point_result_preserved": True,
        "accepted_attempt_modified": False,
        "clarification": {
            "thresholds": {"max_absolute": 0.25, "relative_l2": 0.40},
            "selection_timing": "selected after exploratory preview of the canonical tokenizer case",
            "interpretation": "post-preview descriptive limits, not independently preregistered or unbiased model-quality thresholds",
            "model_quality_pass_claimed": False,
        },
        "raw_canonical_errors": {
            "max_absolute": 0.1808653973101631,
            "relative_l2": 0.34418172495198196,
        },
        "scope_boundary": "does not alter the decisive zero-integer-mismatch RTL-versus-Python fixed-point result and does not erase the checkpoint quality NO-GO",
    })


def source_bindings() -> dict[str, Any]:
    return {
        "runner": file_record(Path(__file__)),
        "accepted_contract": file_record(ACCEPTED_CONTRACT),
        "accepted_attempt_sums": file_record(ACCEPTED / "SHA256SUMS"),
        "provenance_erratum": file_record(ERRATUM),
        "testbenches": {
            "rmsnorm": file_record(RMS_TB),
            "projection": file_record(PROJECTION_TB),
            "rope": file_record(ROPE_TB),
            "score": file_record(SCORE_TB),
            "softmax": file_record(SOFTMAX_TB),
            "compose": file_record(COMPOSE_TB),
        },
        "rtl": {name: file_record(path) for name, path in CORES.items()},
    }


def prepare() -> None:
    require(not CONTRACT.exists(), f"contract already exists: {CONTRACT}")
    ensure_erratum()
    provenance = canonical.validate_provenance()
    contract = {
        "schema_version": 1,
        "mission_increment": MISSION,
        "claim_boundary": "two tokenizer-derived causal positions through the complete layer-0 attention frontier; no O/MLP/residual/final-output claim",
        "canonical_checkpoint": {
            "adapter_sha256": canonical.ADAPTER_SHA256,
            "base_model_sha256": canonical.MODEL_SHA256,
            "checkpoint_tree_sha256": canonical.CHECKPOINT_TREE_SHA256,
            "quality_claim_boundary": "checkpoint quality NO-GO remains in force",
        },
        "fixed_input": {
            "prompt_alias": PROMPT_ALIAS,
            "token_ids": provenance["token_ids"],
            "selected_positions": [0, 1],
            "selected_token_ids": provenance["token_ids"][:2],
        },
        "fixed_quantization": {
            "source": relative(ACCEPTED / "conversion.json"),
            "q_k_v_weight_and_requantization_metadata": "accepted canonical token-0 W4A8 metadata reused unchanged",
            "kv_cache": "signed int8, static layer-0 K/V scales, position-major then KV-head-major",
            "rope": {"theta": ROPE_THETA, "position0": "identity", "position1": "Q15 cosine/sine split-half non-identity"},
            "compose_protocol": "two real causal entries at decode plus seven -32768/zero-value padding entries to exercise FIRST/LAST tiles",
        },
        "evaluator_policy": {
            "integer_mismatch_max": 0,
            "float_quality_pass_allowed": False,
            "no_execution_classification": "misconfigured_run_without_rtl_conclusion",
        },
        "source_bindings": source_bindings(),
        "commands": {
            "preview": ".venv/bin/python tools/run_lora_v4_layer0_two_token_kv_attention_rtl.py --preview",
            "prepare": ".venv/bin/python tools/run_lora_v4_layer0_two_token_kv_attention_rtl.py --prepare",
            "run": ".venv/bin/python tools/run_lora_v4_layer0_two_token_kv_attention_rtl.py --run",
            "check": ".venv/bin/python tools/run_lora_v4_layer0_two_token_kv_attention_rtl.py --check",
        },
        "attempt_policy": {
            "bounded_attention_attempt": "attempt-0001",
            "immutable_after_execution": True,
            "full_layer_two_position_official_attempt_consumed": False,
        },
    }
    write_json(CONTRACT, contract)
    print(json.dumps({"status": "FROZEN", "contract": relative(CONTRACT), "sha256": sha256_file(CONTRACT)}, sort_keys=True))


def validate_contract() -> dict[str, Any]:
    require(CONTRACT.is_file(), "frozen contract is absent")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    current = source_bindings()
    require(contract["source_bindings"] == current, "source binding changed after freeze")
    require(contract["fixed_input"]["token_ids"] == canonical.tokenizer_ids(), "tokenizer IDs changed")
    require(verify_sha256s(ACCEPTED) > 0, "accepted attempt failed immutable hash verification")
    return contract


def execute(attempt: Path, *, official: bool) -> dict[str, Any]:
    global ACTIVE_ATTEMPT
    ACTIVE_ATTEMPT = attempt
    require(not attempt.exists(), f"attempt already exists: {attempt}")
    if official:
        validate_contract()
    else:
        ensure_erratum()
    require(shutil.which("iverilog") is not None and shutil.which("vvp") is not None, "Icarus tools unavailable")
    attempt.mkdir(parents=True)
    started = time.monotonic()
    derived = derive()
    tensors = persist_tensors(derived)
    vectors = persist_vectors(derived)
    rtl = run_rtl()
    result = comparison(derived, rtl)
    write_json(attempt / "comparison.json", result)
    write_json(attempt / "rtl_execution.json", rtl)
    write_json(attempt / "manifest.json", {
        "schema_version": 1,
        "status": result["status"],
        "official_bounded_attention_attempt": official,
        "full_layer_two_position_official_attempt": False,
        "contract": file_record(CONTRACT) if official else None,
        "accepted_attempt": {"path": relative(ACCEPTED), "sha256s": file_record(ACCEPTED / "SHA256SUMS")},
        "token_ids": derived["token_ids"],
        "positions": [0, 1],
        "cache_reuse": derived["cache_reuse"],
        "tensors": tensors,
        "vectors": vectors,
        "elapsed_seconds": time.monotonic() - started,
        "source_bindings": source_bindings(),
    })
    write_sha256s(attempt)
    print(json.dumps({"status": result["status"], "attempt": relative(attempt), "cache_reuse": derived["cache_reuse"]}, sort_keys=True))
    return result


def check() -> None:
    validate_contract()
    require(OFFICIAL.is_dir(), "official bounded attention attempt is absent")
    members = verify_sha256s(OFFICIAL)
    result = json.loads((OFFICIAL / "comparison.json").read_text(encoding="utf-8"))
    require(result["status"] == "PASS_EXACT_TWO_POSITION_ATTENTION_FRONTIER", "official result is not PASS")
    exact = result["rtl_vs_independent_python_fixed_point"]
    for key, value in exact.items():
        if key.endswith("_mismatches"):
            require(value == 0, f"nonzero mismatch count: {key}")
    require(result["position1_rope_non_identity"], "position-1 RoPE evidence missing")
    require(result["cache_reuse"]["compose_output_changes_when_cached_v_zeroed_heads"] > 0, "cache reuse counterfactual missing")
    print(json.dumps({"status": "PASS_IMMUTABLE_CHECK", "members": members, "attempt": relative(OFFICIAL)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preview", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.preview:
        execute(next_preview(), official=False)
    elif args.prepare:
        prepare()
    elif args.run:
        execute(OFFICIAL, official=True)
    else:
        check()


if __name__ == "__main__":
    main()
