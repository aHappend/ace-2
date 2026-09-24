#!/usr/bin/env python3
"""Seal the complete two-position LoRA V4 layer-0 W4A8 RTL path."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

import sys

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_lora_v4_layer0_full_rtl as canonical
import run_lora_v4_layer0_two_token_kv_attention_rtl as frontier


MISSION = "lora_v4_layer0_two_token_kv_full_rtl"
OUT = ROOT / "evidence/verification/lora-v4-layer0-two-token-kv-full-v1"
CONTRACT = OUT / "frozen_contract.json"
OFFICIAL = OUT / "attempt-0001"
PREVIEW_ROOT = ROOT / "build/lora-v4-layer0-two-token-kv-full-preview"
SPEC = ROOT / "design/LORA_V4_LAYER0_TWO_TOKEN_KV_FULL_SPEC.md"
INTERFACE = ROOT / "design/LORA_V4_LAYER0_TWO_TOKEN_KV_FULL_INTERFACE.json"
FRONTIER_CONTRACT = ROOT / "evidence/verification/lora-v4-layer0-two-token-kv-attention-v1/frozen_contract.json"
FRONTIER_ATTEMPT = FRONTIER_CONTRACT.parent / "attempt-0001"
ACCEPTED_CONTRACT = ROOT / "evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1/frozen_contract.json"
ACCEPTED = ACCEPTED_CONTRACT.parent / "attempt-0001"
ERRATUM = ACCEPTED_CONTRACT.parent / "errata/float-fidelity-post-preview-v1/erratum.json"

RMS_TB = ROOT / "verification/tb/ace2_rmsnorm_tb.sv"
PROJECTION_TB = ROOT / "verification/tb/ace2_lora_v4_layer0_projection_batch_tb.sv"
ROPE_TB = ROOT / "verification/tb/ace2_rope_tb.sv"
SCORE_TB = ROOT / "verification/tb/ace2_attention_score_tb.sv"
SOFTMAX_TB = ROOT / "verification/tb/ace2_softmax_tb.sv"
COMPOSE_TB = ROOT / "verification/tb/ace2_attention_compose_tb.sv"
SILU_TB = ROOT / "verification/tb/ace2_silu_gate_tb.sv"
RESIDUAL_TB = ROOT / "verification/tb/ace2_shell_tb.sv"

CORES = {
    "projection": ROOT / "rtl/ace2_w4a8_proj_core.sv",
    "rmsnorm": ROOT / "rtl/ace2_rmsnorm_core.sv",
    "rope": ROOT / "rtl/ace2_rope_core.sv",
    "score": ROOT / "rtl/ace2_attention_score_core.sv",
    "softmax": ROOT / "rtl/ace2_softmax_core.sv",
    "compose": ROOT / "rtl/ace2_attention_compose_core.sv",
    "silu": ROOT / "rtl/ace2_silu_gate_core.sv",
}

TESTBENCHES = {
    "projection": PROJECTION_TB,
    "rmsnorm": RMS_TB,
    "rope": ROPE_TB,
    "score": SCORE_TB,
    "softmax": SOFTMAX_TB,
    "compose": COMPOSE_TB,
    "silu": SILU_TB,
    "residual": RESIDUAL_TB,
}

HIDDEN = 896
INTERMEDIATE = 4864
Q_HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64
ROPE_THETA = 1_000_000.0
PROJECTION_ORDER = ("q", "k", "v", "o", "gate", "up", "down")
PROJECTION_CHANNELS = {"q": 896, "k": 128, "v": 128, "o": 896, "gate": 4864, "up": 4864, "down": 896}
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

ACTIVE_ATTEMPT = OFFICIAL

require = frontier.require
sha256_file = frontier.sha256_file
relative = frontier.relative
file_record = frontier.file_record
write_json = frontier.write_json
write_text = frontier.write_text
write_binary = frontier.write_binary
write_hex = frontier.write_hex
raw_int8 = frontier.raw_int8
raw_int32 = frontier.raw_int32
load_s8 = frontier.load_s8
load_s32 = frontier.load_s32
chunks = frontier.chunks
pack = frontier.pack
hex_literal = frontier.hex_literal
write_sha256s = frontier.write_sha256s
verify_sha256s = frontier.verify_sha256s


def next_preview() -> Path:
    index = 1
    while (PREVIEW_ROOT / f"attempt-{index:04d}").exists():
        index += 1
    return PREVIEW_ROOT / f"attempt-{index:04d}"


def float_rope(values: torch.Tensor, position: int, heads: int) -> torch.Tensor:
    shaped = values.to(torch.float32).reshape(heads, HEAD_DIM)
    frequencies = torch.tensor(
        [ROPE_THETA ** (-(2.0 * index) / HEAD_DIM) for index in range(HEAD_DIM // 2)],
        dtype=torch.float32,
    )
    angles = frequencies * float(position)
    cosine = torch.cat((torch.cos(angles), torch.cos(angles))).reshape(1, HEAD_DIM)
    sine = torch.cat((torch.sin(angles), torch.sin(angles))).reshape(1, HEAD_DIM)
    rotated = torch.cat((-shaped[:, HEAD_DIM // 2 :], shaped[:, : HEAD_DIM // 2]), dim=1)
    return (shaped * cosine + rotated * sine).reshape(-1).contiguous()


def float_attention(
    position: int,
    q_output: torch.Tensor,
    cached_k: list[torch.Tensor],
    cached_v: list[torch.Tensor],
) -> torch.Tensor:
    q_heads = q_output.reshape(Q_HEADS, HEAD_DIM)
    context = position + 1
    heads: list[torch.Tensor] = []
    for head in range(Q_HEADS):
        kv_head = head // (Q_HEADS // KV_HEADS)
        keys = torch.stack(
            [cached_k[token].reshape(KV_HEADS, HEAD_DIM)[kv_head] for token in range(context)]
        )
        values = torch.stack(
            [cached_v[token].reshape(KV_HEADS, HEAD_DIM)[kv_head] for token in range(context)]
        )
        logits = torch.mv(keys, q_heads[head]) / math.sqrt(float(HEAD_DIM))
        probabilities = torch.softmax(logits, dim=0)
        heads.append(torch.matmul(probabilities, values))
    return torch.stack(heads).reshape(HIDDEN).contiguous()


def derive_position1_tail(
    position: dict[str, Any],
    float_attention_value: torch.Tensor,
    embedding: torch.Tensor,
    post_gain: torch.Tensor,
    weights: Any,
    adapter: Any,
) -> dict[str, Any]:
    projections = position["projections"]
    o_merged, o_hashes = canonical.merge_projection(weights, adapter, "model.layers.0.self_attn.o_proj")
    projections["o"] = canonical.derive_projection(
        "position1_o",
        o_merged,
        position["attention_q"],
        projections["v"]["output_scale"],
        float_attention_value,
        o_hashes,
    )

    float_attention_residual = embedding.to(torch.float32) + projections["o"]["float_output"]
    attention_residual_scale = canonical.scale_for(float_attention_residual)
    attention_residual_source = canonical.quantize_int8(embedding, attention_residual_scale)
    attention_residual_o = canonical.quantize_int8(
        projections["o"]["output_q"].to(torch.float64) * projections["o"]["output_scale"],
        attention_residual_scale,
    )
    attention_residual_values, attention_residual_sat = canonical.reference_residual_add(
        attention_residual_source.to(torch.int64).tolist(),
        attention_residual_o.to(torch.int64).tolist(),
    )
    attention_residual_q = torch.tensor(attention_residual_values, dtype=torch.int8)

    float_post_norm = canonical.float_rmsnorm(float_attention_residual, post_gain)
    post_norm_scale = canonical.derive_rmsnorm_output_scale(
        post_gain.to(torch.float64).tolist(),
        float(float_post_norm.to(torch.float64).abs().max().item()),
    )
    post_gains = canonical.derive_scaled_gains_q8(post_gain.to(torch.float64).tolist(), post_norm_scale)
    post_norm = canonical.reference_rmsnorm(attention_residual_values, post_gains)
    post_norm_q = torch.tensor(post_norm.outputs, dtype=torch.int8)

    for key, tensor_name in (
        ("gate", "model.layers.0.mlp.gate_proj"),
        ("up", "model.layers.0.mlp.up_proj"),
    ):
        merged, hashes = canonical.merge_projection(weights, adapter, tensor_name)
        projections[key] = canonical.derive_projection(
            f"position1_{key}", merged, post_norm_q, post_norm_scale, float_post_norm, hashes
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
    silu_scale = canonical.scale_for(float_silu)
    silu_multiplier, silu_shift = canonical.derive_multiplier(
        torch.tensor([1.0 / (silu_scale * (1 << 21))], dtype=torch.float64)
    )
    silu = canonical.reference_silu_gate(
        canonical.SiluGateCase(
            "position1_silu",
            gate_q6.to(torch.int64).tolist(),
            up_q6.to(torch.int64).tolist(),
            int(silu_multiplier[0]),
            int(silu_shift[0]),
            0,
        )
    )
    silu_q = torch.tensor(silu.outputs, dtype=torch.int8)

    down_merged, down_hashes = canonical.merge_projection(weights, adapter, "model.layers.0.mlp.down_proj")
    projections["down"] = canonical.derive_projection(
        "position1_down", down_merged, silu_q, silu_scale, float_silu, down_hashes
    )
    float_layer_output = float_attention_residual + projections["down"]["float_output"]
    layer_output_scale = canonical.scale_for(float_layer_output)
    final_residual_stream = canonical.quantize_int8(
        attention_residual_q.to(torch.float64) * attention_residual_scale,
        layer_output_scale,
    )
    final_residual_down = canonical.quantize_int8(
        projections["down"]["output_q"].to(torch.float64) * projections["down"]["output_scale"],
        layer_output_scale,
    )
    layer_values, layer_sat = canonical.reference_residual_add(
        final_residual_down.to(torch.int64).tolist(),
        final_residual_stream.to(torch.int64).tolist(),
    )
    layer_q = torch.tensor(layer_values, dtype=torch.int8)
    dequant = layer_q.to(torch.float64) * layer_output_scale
    error = dequant - float_layer_output.to(torch.float64)

    position.update(
        {
            "embedding": embedding,
            "post_gain": post_gain,
            "attention_residual": {
                "lhs": attention_residual_source,
                "rhs": attention_residual_o,
                "output": attention_residual_q,
                "scale": attention_residual_scale,
                "saturation": attention_residual_sat,
            },
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
                "relative_l2_error": float(
                    torch.linalg.vector_norm(error)
                    / torch.linalg.vector_norm(float_layer_output.to(torch.float64))
                ),
                "float_output_absmax": float(float_layer_output.to(torch.float64).abs().max().item()),
                "layer_output_scale": layer_output_scale,
            },
        }
    )
    return position


def bind_position0(position: dict[str, Any], accepted: dict[str, Any]) -> dict[str, Any]:
    require(torch.equal(position["attention_q"], accepted["attention_q"]), "position-0 attention changed")
    for key in ("q", "k", "v"):
        require(
            torch.equal(position["projections"][key]["output_q"], accepted["projections"][key]["output_q"]),
            f"position-0 {key} output changed",
        )
        require(
            torch.equal(position["projections"][key]["accumulator"], accepted["projections"][key]["accumulator"]),
            f"position-0 {key} accumulator changed",
        )
    for key in ("o", "gate", "up", "down"):
        position["projections"][key] = accepted["projections"][key]
    position.update(
        {
            "embedding": accepted["embedding"],
            "post_gain": accepted["post_gain"],
            "attention_residual": accepted["attention_residual"],
            "post_gains": accepted["post_gains"],
            "post_norm": accepted["post_norm"],
            "post_norm_q": accepted["post_norm_q"],
            "post_norm_scale": accepted["post_norm_scale"],
            "silu": accepted["silu"],
            "final_residual": accepted["final_residual"],
            "float_layer_output": accepted["float_layer_output"],
            "metrics": accepted["metrics"],
        }
    )
    return position


def validate_position0_against_attempt(position: dict[str, Any]) -> dict[str, int]:
    comparisons = 0
    expected_boundaries = {
        "input_rmsnorm_s8.bin": position["input_norm_q"],
        "attention_output_s8.bin": position["attention_q"],
        "attention_residual_s8.bin": position["attention_residual"]["output"],
        "post_attention_rmsnorm_s8.bin": position["post_norm_q"],
        "silu_output_s8.bin": position["silu"]["output"],
        "layer0_output_s8.bin": position["final_residual"]["output"],
    }
    for name, value in expected_boundaries.items():
        require(torch.equal(load_s8(ACCEPTED / "tensors" / name), value), f"accepted position-0 boundary changed: {name}")
        comparisons += 1
    for key in PROJECTION_ORDER:
        projection = position["projections"][key]
        require(
            torch.equal(load_s8(ACCEPTED / f"tensors/{key}_output_s8.bin"), projection["output_q"]),
            f"accepted position-0 {key} output changed",
        )
        require(
            torch.equal(load_s32(ACCEPTED / f"tensors/{key}_accumulator_s32le.bin"), projection["accumulator"]),
            f"accepted position-0 {key} accumulator changed",
        )
        require(
            torch.equal(
                load_s8(ACCEPTED / f"tensors/{key}_qweight_s4_in_s8.bin"),
                projection["qweight"].reshape(-1),
            ),
            f"accepted position-0 {key} weight changed",
        )
        comparisons += 3
    return {"accepted_boundary_and_projection_comparisons": comparisons}


def derive() -> dict[str, Any]:
    attention = frontier.derive()
    accepted = canonical.derive()
    positions = attention["positions"]
    positions[0] = bind_position0(positions[0], accepted)

    with safe_open(canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        float_q: list[torch.Tensor] = []
        float_k: list[torch.Tensor] = []
        float_v: list[torch.Tensor] = []
        embeddings: list[torch.Tensor] = []
        post_gain = weights.get_tensor("model.layers.0.post_attention_layernorm.weight").contiguous()
        input_gain = weights.get_tensor("model.layers.0.input_layernorm.weight").contiguous()
        for index, position in enumerate(positions):
            embedding = weights.get_tensor("model.embed_tokens.weight")[position["token_id"]].contiguous()
            embeddings.append(embedding)
            float_input_norm = canonical.float_rmsnorm(embedding, input_gain)
            outputs: dict[str, torch.Tensor] = {}
            for key, tensor_name in (
                ("q", "model.layers.0.self_attn.q_proj"),
                ("k", "model.layers.0.self_attn.k_proj"),
                ("v", "model.layers.0.self_attn.v_proj"),
            ):
                merged, hashes = canonical.merge_projection(weights, adapter, tensor_name)
                require(hashes == attention["source_hashes"][key], f"{key} source hashes changed")
                outputs[key] = torch.mv(merged, float_input_norm.to(torch.float32)).contiguous()
                position["projections"][key]["float_output"] = outputs[key]
                position["projections"][key]["source_hashes"] = hashes
            float_q.append(float_rope(outputs["q"], index, Q_HEADS))
            float_k.append(float_rope(outputs["k"], index, KV_HEADS))
            float_v.append(outputs["v"])

        position1_attention = float_attention(1, float_q[1], float_k, float_v)
        positions[1] = derive_position1_tail(
            positions[1], position1_attention, embeddings[1], post_gain, weights, adapter
        )

    for key in PROJECTION_ORDER:
        require(
            torch.equal(positions[0]["projections"][key]["qweight"], positions[1]["projections"][key]["qweight"]),
            f"shared {key} W4 tensor differs by position",
        )

    accepted_reproduction = validate_position0_against_attempt(positions[0])
    return {
        **attention,
        "positions": positions,
        "accepted_position0_reproduction": accepted_reproduction,
    }


def render_rmsnorm_vectors(derived: dict[str, Any]) -> str:
    cases: list[tuple[list[int], list[int], Any]] = []
    for position in derived["positions"]:
        cases.append(
            (
                position["embedding_q"].to(torch.int64).tolist(),
                position["input_gains"],
                position["input_norm"],
            )
        )
        cases.append(
            (
                position["attention_residual"]["output"].to(torch.int64).tolist(),
                position["post_gains"],
                position["post_norm"],
            )
        )
    lines = [
        "// Two positions, input and post-attention RMSNorm.",
        "localparam integer TEST_COUNT = 4;",
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
            zip(
                canonical.pack_int8_beats(inputs),
                canonical.pack_gain_beats(gains),
                canonical.pack_int8_beats(result.outputs),
                strict=True,
            )
        ):
            flat = case * 56 + beat
            lines += [
                f"  test_input_beats[{flat}] = {hex_literal(input_word, 128)};",
                f"  test_gain_beats[{flat}] = {hex_literal(gain_word, 256)};",
                f"  test_expected_beats[{flat}] = {hex_literal(output_word, 128)};",
            ]
    return "\n".join(lines + ["end", ""])


def render_silu_vectors(derived: dict[str, Any]) -> str:
    lines = [
        "// Two-position layer-0 SiLU vectors.",
        "localparam integer SILU_CASE_COUNT = 2;",
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
    ]
    for case, position in enumerate(derived["positions"]):
        silu = position["silu"]
        lines += [
            f"  silu_case_length[{case}] = 16'd{INTERMEDIATE};",
            f"  silu_case_multiplier[{case}] = 32'sd{silu['multiplier']};",
            f"  silu_case_right_shift[{case}] = 6'd{silu['right_shift']};",
            f"  silu_case_zero_point[{case}] = 8'sd0;",
            f"  silu_expected_saturation[{case}] = 1'b{int(silu['result'].saturation_seen)};",
            f"  silu_case_boundary_coverage[{case}] = 6'd0;",
        ]
        gate = silu["gate_q6"].to(torch.int64).tolist()
        up = silu["up_q6"].to(torch.int64).tolist()
        output = silu["output"].to(torch.int64).tolist()
        for beat in range(608):
            flat = case * 608 + beat
            lines += [
                f"  silu_gate_beats[{flat}] = {hex_literal(pack(gate[beat * 8 : (beat + 1) * 8], 16), 128)};",
                f"  silu_up_beats[{flat}] = {hex_literal(pack(up[beat * 8 : (beat + 1) * 8], 16), 128)};",
            ]
        for beat, group in enumerate(chunks(output, 16)):
            lines.append(
                f"  silu_expected_beats[{case * 304 + beat}] = {hex_literal(pack(group, 8), 128)};"
            )
    return "\n".join(lines + ["end", ""])


def render_residual_vectors(derived: dict[str, Any], *, final: bool) -> str:
    prefix = "mlp_residual" if final else "residual"
    upper = "MLP_RESIDUAL" if final else "RESIDUAL"
    lhs_name = "down" if final else "lhs"
    rhs_name = "stream" if final else "rhs"
    lines = [
        "// Two-position residual vectors.",
        f"localparam integer {upper}_CASE_COUNT = 2;",
        f"localparam integer {upper}_BEATS = 56;",
        f"reg [127:0] {prefix}_{lhs_name}_beats [0:{upper}_CASE_COUNT*{upper}_BEATS-1];",
        f"reg [127:0] {prefix}_{rhs_name}_beats [0:{upper}_CASE_COUNT*{upper}_BEATS-1];",
        f"reg [127:0] {prefix}_expected_beats [0:{upper}_CASE_COUNT*{upper}_BEATS-1];",
        f"reg {prefix}_expected_saturation [0:{upper}_CASE_COUNT-1];",
        "initial begin",
    ]
    for case, position in enumerate(derived["positions"]):
        values = position["final_residual"] if final else position["attention_residual"]
        lhs = values["down" if final else "lhs"].to(torch.int64).tolist()
        rhs = values["stream" if final else "rhs"].to(torch.int64).tolist()
        output = values["output"].to(torch.int64).tolist()
        lines.append(f"  {prefix}_expected_saturation[{case}] = 1'b{int(values['saturation'])};")
        for beat, (lhs_group, rhs_group, out_group) in enumerate(
            zip(chunks(lhs, 16), chunks(rhs, 16), chunks(output, 16), strict=True)
        ):
            flat = case * 56 + beat
            lines += [
                f"  {prefix}_{lhs_name}_beats[{flat}] = {hex_literal(pack(lhs_group, 8), 128)};",
                f"  {prefix}_{rhs_name}_beats[{flat}] = {hex_literal(pack(rhs_group, 8), 128)};",
                f"  {prefix}_expected_beats[{flat}] = {hex_literal(pack(out_group, 8), 128)};",
            ]
    return "\n".join(lines + ["end", ""])


def render_projection_source() -> str:
    source = PROJECTION_TB.read_text(encoding="utf-8")
    source = source.replace("localparam integer CASES = 7;", "localparam integer CASES = 14;")
    source = source.replace("localparam integer TOTAL_INPUT_GROUPS = 2560;", "localparam integer TOTAL_INPUT_GROUPS = 5120;")
    source = source.replace("localparam integer TOTAL_OUTPUTS = 12672;", "localparam integer TOTAL_OUTPUTS = 25344;")
    old = re.search(r"    task select_case;.*?    endtask\n", source, flags=re.S)
    require(old is not None, "projection select_case block changed")
    rows = [
        (224, 896, 0, 0, 0),
        (224, 128, 224, 200704, 896),
        (224, 128, 448, 229376, 1024),
        (224, 896, 672, 258048, 1152),
        (224, 4864, 896, 458752, 2048),
        (224, 4864, 1120, 1548288, 6912),
        (1216, 896, 1344, 2637824, 11776),
        (224, 896, 2560, 0, 12672),
        (224, 128, 2784, 200704, 13568),
        (224, 128, 3008, 229376, 13696),
        (224, 896, 3232, 258048, 13824),
        (224, 4864, 3456, 458752, 14720),
        (224, 4864, 3680, 1548288, 19584),
        (1216, 896, 3904, 2637824, 24448),
    ]
    selection = ["    task select_case;", "        input integer selected_case;", "        begin", "            case (selected_case)"]
    for case, (groups, outputs, input_offset, weight_offset, output_offset) in enumerate(rows):
        label = "default" if case == len(rows) - 1 else str(case)
        selection.append(
            f"                {label}: begin groups = {groups}; outputs = {outputs}; input_offset = {input_offset}; weight_offset = {weight_offset}; output_offset = {output_offset}; end"
        )
    selection += ["            endcase", "        end", "    endtask", ""]
    source = source[: old.start()] + "\n".join(selection) + source[old.end() :]
    frozen = "evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1/attempt-0001/vectors"
    source = source.replace(frozen, relative(ACTIVE_ATTEMPT / "vectors"))
    return instrument_cycles(source, "projection", "    ace2_w4a8_proj_core #(")


def instrument_cycles(source: str, family: str, instance_needle: str) -> str:
    require(instance_needle in source, f"{family} instance insertion point changed")
    counter = (
        "    integer ace2_full_cycle_count;\n"
        "    initial ace2_full_cycle_count = 0;\n"
        "    always @(posedge clk) ace2_full_cycle_count <= ace2_full_cycle_count + 1;\n\n"
    )
    source = source.replace(instance_needle, counter + instance_needle, 1)
    finish = source.rfind("$finish;")
    require(finish >= 0, f"{family} finish marker missing")
    source = (
        source[:finish]
        + f'$display("ACE2_FULL_LAYER_SIM_CYCLES family={family} cycles=%0d", ace2_full_cycle_count);\n        '
        + source[finish:]
    )
    return source


def execution_source(name: str, source_path: Path, generated_name: str, instance_needle: str) -> Path:
    source = source_path.read_text(encoding="utf-8")
    source = source.replace(
        f'`include "../generated/{generated_name}"',
        f'`include "{relative(ACTIVE_ATTEMPT / "vectors" / generated_name)}"',
        1,
    )
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
    source = instrument_cycles(source, name, instance_needle)
    path = ACTIVE_ATTEMPT / "execution_sources" / f"{name}_tb.sv"
    write_text(path, source)
    return path


def residual_execution_source() -> Path:
    source = RESIDUAL_TB.read_text(encoding="utf-8")
    source = source.replace(
        "    ace2_shell dut (",
        "    integer ace2_full_position;\n\n    ace2_shell dut (",
        1,
    )
    source = source.replace(
        '`include "../generated/residual_vectors.svh"',
        f'`include "{relative(ACTIVE_ATTEMPT / "vectors/residual_vectors.svh")}"',
        1,
    ).replace(
        '`include "../generated/mlp_residual_vectors.svh"',
        f'`include "{relative(ACTIVE_ATTEMPT / "vectors/mlp_residual_vectors.svh")}"',
        1,
    )
    branch = r'''
        if ($test$plusargs("LORA_V4_LAYER0_TWO_TOKEN_FULL_RESIDUAL_REPLAY")) begin
            for (ace2_full_position = 0; ace2_full_position < 2;
                 ace2_full_position = ace2_full_position + 1) begin
                send_residual_cmd_full(ace2_full_position, 16'd896, RESIDUAL_LHS_BASE,
                                       RESIDUAL_RHS_BASE, RESIDUAL_OUT_BASE,
                                       16'h7b00 + ace2_full_position*2);
                wait_residual_done_and_compare(
                    ace2_full_position, residual_expected_saturation[ace2_full_position],
                    16'h7b00 + ace2_full_position*2
                );
                for (case_index = 0; case_index < observed_count; case_index = case_index + 1)
                    $display("RTL_RESIDUAL_BEAT position=%0d stage=attention beat=%0d data=%032x",
                             ace2_full_position, case_index, observed_output[case_index]);
                if (residual_expected_saturation[ace2_full_position])
                    csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

                send_mlp_residual_cmd_full(
                    ace2_full_position, 16'd896, MLP_RESIDUAL_DOWN_BASE,
                    MLP_RESIDUAL_STREAM_BASE, MLP_RESIDUAL_OUT_BASE,
                    16'h7b01 + ace2_full_position*2
                );
                wait_mlp_residual_done_and_compare(
                    ace2_full_position, mlp_residual_expected_saturation[ace2_full_position],
                    16'h7b01 + ace2_full_position*2
                );
                for (case_index = 0; case_index < observed_count; case_index = case_index + 1)
                    $display("RTL_RESIDUAL_BEAT position=%0d stage=final beat=%0d data=%032x",
                             ace2_full_position, case_index, observed_output[case_index]);
                if (mlp_residual_expected_saturation[ace2_full_position])
                    csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            end
            if (failures != 0)
                $fatal(1, "ACE2_LORA_V4_LAYER0_TWO_TOKEN_FULL_RESIDUAL_REPLAY_FAIL failures=%0d", failures);
            $display("ACE2_LORA_V4_LAYER0_TWO_TOKEN_FULL_RESIDUAL_REPLAY_PASS positions=2 stages=4 beats=224 cycles=%0d",
                     cycle_count);
            $finish;
        end

'''
    needle = "        if (qproj_stride_only_mode) begin"
    require(needle in source, "shell residual insertion point changed")
    source = source.replace(needle, branch + needle, 1)
    path = ACTIVE_ATTEMPT / "execution_sources/residual_tb.sv"
    write_text(path, source)
    return path


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
            activation_words.extend(
                pack(group, 8) for group in chunks(projection["input_q"].to(torch.int64).tolist(), 4)
            )
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
        "rope": write_text(directory / "rope_vectors.svh", frontier.render_rope_vectors(derived)),
        "score": write_text(directory / "attention_score_vectors.svh", frontier.render_score_vectors(derived)),
        "softmax": write_text(directory / "softmax_vectors.svh", frontier.render_softmax_vectors(derived)),
        "compose": write_text(directory / "attention_compose_vectors.svh", frontier.render_compose_vectors(derived)),
        "silu": write_text(directory / "silu_gate_vectors.svh", render_silu_vectors(derived)),
        "residual": write_text(directory / "residual_vectors.svh", render_residual_vectors(derived, final=False)),
        "mlp_residual": write_text(
            directory / "mlp_residual_vectors.svh", render_residual_vectors(derived, final=True)
        ),
    }


def run_process(name: str, command: list[str]) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    elapsed = time.monotonic() - started
    logs = ACTIVE_ATTEMPT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (logs / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    write_json(
        logs / f"{name}.result.json",
        {"command": command, "elapsed_seconds": elapsed, "returncode": completed.returncode},
    )
    return completed


def compile_and_run(
    name: str,
    sources: list[Path],
    top: str,
    plusargs: list[str] | None = None,
) -> dict[str, Any]:
    binary = ACTIVE_ATTEMPT / "sim" / f"{name}.vvp"
    binary.parent.mkdir(parents=True, exist_ok=True)
    compile_command = [
        "iverilog",
        "-g2012",
        "-I.",
        "-Irtl",
        "-Irtl/generated",
        "-Iverification/tb",
        "-s",
        top,
        "-o",
        relative(binary),
        *[relative(path) for path in sources],
    ]
    compiled = run_process(f"{name}.iverilog", compile_command)
    require(compiled.returncode == 0, f"{name} compile failed")
    simulate_command = ["vvp", relative(binary), *(plusargs or [])]
    simulated = run_process(f"{name}.vvp", simulate_command)
    require(simulated.returncode == 0, f"{name} simulation failed")
    cycle_match = re.search(
        rf"ACE2_FULL_LAYER_SIM_CYCLES family={re.escape(name)} cycles=(\d+)", simulated.stdout
    )
    if name != "residual":
        require(cycle_match is not None, f"{name} cycle marker missing")
    return {
        "compile_command": compile_command,
        "simulate_command": simulate_command,
        "binary": file_record(binary),
        "stdout": file_record(ACTIVE_ATTEMPT / f"logs/{name}.vvp.stdout.log"),
        "stderr": file_record(ACTIVE_ATTEMPT / f"logs/{name}.vvp.stderr.log"),
        "simulator_cycles": int(cycle_match.group(1)) if cycle_match else None,
        "stdout_text": simulated.stdout,
    }


def run_rtl(derived: dict[str, Any]) -> dict[str, Any]:
    projection_source = ACTIVE_ATTEMPT / "execution_sources/projection_tb.sv"
    write_text(projection_source, render_projection_source())
    sources = {
        "rmsnorm": execution_source("rmsnorm", RMS_TB, "rmsnorm_vectors.svh", "    ace2_rmsnorm_core dut ("),
        "rope": execution_source("rope", ROPE_TB, "rope_vectors.svh", "    ace2_rope_core dut ("),
        "score": execution_source(
            "score", SCORE_TB, "attention_score_vectors.svh", "    ace2_attention_score_core #(")
        ,
        "softmax": execution_source(
            "softmax", SOFTMAX_TB, "softmax_vectors.svh", "    ace2_softmax_core #(")
        ,
        "compose": execution_source(
            "compose", COMPOSE_TB, "attention_compose_vectors.svh", "    ace2_attention_compose_core dut (")
        ,
        "silu": execution_source("silu", SILU_TB, "silu_gate_vectors.svh", "    ace2_silu_gate_core dut ("),
        "residual": residual_execution_source(),
    }
    results = {
        "projection": compile_and_run(
            "projection", [CORES["projection"], projection_source], "ace2_lora_v4_layer0_projection_batch_tb"
        ),
        "rmsnorm": compile_and_run("rmsnorm", [CORES["rmsnorm"], sources["rmsnorm"]], "ace2_rmsnorm_tb"),
        "rope": compile_and_run("rope", [CORES["rope"], sources["rope"]], "ace2_rope_tb"),
        "score": compile_and_run(
            "score", [CORES["score"], sources["score"]], "ace2_attention_score_tb"
        ),
        "softmax": compile_and_run(
            "softmax", [CORES["softmax"], sources["softmax"]], "ace2_softmax_tb"
        ),
        "compose": compile_and_run(
            "compose", [CORES["compose"], sources["compose"]], "ace2_attention_compose_tb"
        ),
        "silu": compile_and_run("silu", [CORES["silu"], sources["silu"]], "ace2_silu_gate_tb"),
    }
    rtl_sources = [ROOT / "rtl/ace2_pkg.sv"] + sorted(
        path for path in (ROOT / "rtl").glob("*.sv") if path.name != "ace2_pkg.sv"
    )
    results["residual"] = compile_and_run(
        "residual",
        [*rtl_sources, sources["residual"]],
        "ace2_shell_tb",
        ["+LORA_V4_LAYER0_TWO_TOKEN_FULL_RESIDUAL_REPLAY"],
    )

    markers = {
        "projection": "ACE2_LORA_V4_LAYER0_PROJECTION_BATCH_PASS operators=14 channels=25344",
        "rmsnorm": "ACE2_RMSNORM_TB_PASS cases=4 beats_per_case=56",
        "rope": "ACE2_ROPE_TB_PASS cases=4 beats_per_case=56",
        "score": "ACE2_ATTN_SCORE_TB_PASS cases=28 context_max=8",
        "softmax": "ACE2_SOFTMAX_TB_PASS cases=28 context_max=8",
        "compose": "ACE2_ATTN_COMPOSE_TB_PASS cases=28",
        "silu": "ACE2_SILU_GATE_TB_PASS cases=2",
        "residual": "ACE2_LORA_V4_LAYER0_TWO_TOKEN_FULL_RESIDUAL_REPLAY_PASS positions=2 stages=4 beats=224",
    }
    for name, marker in markers.items():
        require(marker in results[name]["stdout_text"], f"{name} PASS marker missing")

    projection_rows = re.findall(
        r"RTL_PROJ_RESULT operator=(\d+) channel=(\d+) out_u8=(\d+) acc=(-?\d+) saturation=(\d+) overflow=(\d+)",
        results["projection"]["stdout_text"],
    )
    require(len(projection_rows) == 25344, "projection RTL output count changed")
    residual_cycle = re.search(
        r"ACE2_LORA_V4_LAYER0_TWO_TOKEN_FULL_RESIDUAL_REPLAY_PASS positions=2 stages=4 beats=224 cycles=(\d+)",
        results["residual"]["stdout_text"],
    )
    require(residual_cycle is not None, "residual cycle marker missing")
    results["residual"]["simulator_cycles"] = int(residual_cycle.group(1))

    final_rows = re.findall(
        r"RTL_RESIDUAL_BEAT position=(\d+) stage=final beat=(\d+) data=([0-9a-fA-F]+)",
        results["residual"]["stdout_text"],
    )
    final_hashes: list[str] = []
    for position_index in (0, 1):
        selected = [(int(beat), data) for position, beat, data in final_rows if int(position) == position_index]
        require([beat for beat, _ in selected] == list(range(56)), f"position-{position_index} final beat order changed")
        raw = b"".join(int(data, 16).to_bytes(16, "little") for _, data in selected)
        expected = raw_int8(derived["positions"][position_index]["final_residual"]["output"].tolist())
        require(raw == expected, f"position-{position_index} final RTL output differs")
        final_hashes.append(frontier.sha256_bytes(raw))

    cycles = {name: value["simulator_cycles"] for name, value in results.items()}
    for value in results.values():
        value.pop("stdout_text", None)
    return {
        "executions": results,
        "projection_results": len(projection_rows),
        "final_layer_output_sha256_by_position": final_hashes,
        "simulator_cycles": cycles,
    }


def persist_tensors(derived: dict[str, Any]) -> dict[str, Any]:
    directory = ACTIVE_ATTEMPT / "tensors"
    records: dict[str, Any] = {}
    for position in derived["positions"]:
        prefix = f"position{position['position']}"
        payloads = {
            f"{prefix}_embedding_bf16le.bin": canonical.raw_tensor_bytes(position["embedding"]),
            f"{prefix}_input_rmsnorm_s8.bin": raw_int8(position["input_norm_q"].tolist()),
            f"{prefix}_rope_q_s8.bin": raw_int8(position["rope_q"].outputs),
            f"{prefix}_rope_k_s8.bin": raw_int8(position["rope_k"].outputs),
            f"{prefix}_attention_s8.bin": raw_int8(position["attention_q"].tolist()),
            f"{prefix}_attention_residual_s8.bin": raw_int8(position["attention_residual"]["output"].tolist()),
            f"{prefix}_post_attention_rmsnorm_s8.bin": raw_int8(position["post_norm_q"].tolist()),
            f"{prefix}_silu_output_s8.bin": raw_int8(position["silu"]["output"].tolist()),
            f"{prefix}_layer0_output_s8.bin": raw_int8(position["final_residual"]["output"].tolist()),
            f"{prefix}_layer0_output_float_f32le.bin": canonical.raw_tensor_bytes(position["float_layer_output"]),
            f"{prefix}_layer0_output_scale_f64le.bin": struct.pack("<d", position["final_residual"]["scale"]),
        }
        for name, raw in payloads.items():
            records[name] = write_binary(directory / name, raw)
        for key in PROJECTION_ORDER:
            projection = position["projections"][key]
            records[f"{prefix}_{key}_output_s8.bin"] = write_binary(
                directory / f"{prefix}_{key}_output_s8.bin", raw_int8(projection["output_q"].tolist())
            )
            records[f"{prefix}_{key}_accumulator_s32le.bin"] = write_binary(
                directory / f"{prefix}_{key}_accumulator_s32le.bin",
                raw_int32(projection["accumulator"].tolist()),
            )
    for key in PROJECTION_ORDER:
        qweight = derived["positions"][0]["projections"][key]["qweight"]
        records[f"shared_{key}_qweight_s4_in_s8.bin"] = write_binary(
            directory / f"shared_{key}_qweight_s4_in_s8.bin", raw_int8(qweight.reshape(-1).tolist())
        )
    records["kv_cache_k_s8.bin"] = write_binary(
        directory / "kv_cache_k_s8.bin", raw_int8(derived["cached_k"][0] + derived["cached_k"][1])
    )
    records["kv_cache_v_s8.bin"] = write_binary(
        directory / "kv_cache_v_s8.bin", raw_int8(derived["cached_v"][0] + derived["cached_v"][1])
    )
    combined = b"".join(
        raw_int8(position["final_residual"]["output"].tolist()) for position in derived["positions"]
    )
    records["layer0_outputs_position_major_s8.bin"] = write_binary(
        directory / "layer0_outputs_position_major_s8.bin", combined
    )
    return records


def conversion_summary(derived: dict[str, Any]) -> dict[str, Any]:
    positions = []
    for position in derived["positions"]:
        projections = {}
        for key in PROJECTION_ORDER:
            projection = position["projections"][key]
            projections[key] = {
                "input_elements": int(projection["input_q"].numel()),
                "output_channels": int(projection["output_q"].numel()),
                "input_scale": projection["input_scale"],
                "output_scale": projection["output_scale"],
                "accumulator_range": [int(projection["accumulator"].min()), int(projection["accumulator"].max())],
                "saturation_count": int(projection["saturation"].sum()),
                "source_hashes": projection.get("source_hashes"),
            }
        positions.append(
            {
                "position": position["position"],
                "token_id": position["token_id"],
                "projections": projections,
                "rmsnorm": {
                    "input_sumsq": position["input_norm"].sumsq,
                    "input_inv_rms_q30": position["input_norm"].inv_rms_q30,
                    "post_sumsq": position["post_norm"].sumsq,
                    "post_inv_rms_q30": position["post_norm"].inv_rms_q30,
                },
                "scales": {
                    "input_rmsnorm": position["projections"]["q"]["input_scale"],
                    "attention_residual": position["attention_residual"]["scale"],
                    "post_attention_rmsnorm": position["post_norm_scale"],
                    "silu": position["silu"]["scale"],
                    "layer_output": position["final_residual"]["scale"],
                },
                "float_vs_w4a8_descriptive": position["metrics"],
            }
        )
    return {
        "schema_version": 1,
        "status": "PASS_DERIVATION",
        "projection_order": list(PROJECTION_ORDER),
        "positions": positions,
        "cache_reuse": derived["cache_reuse"],
        "accepted_position0_reproduction": derived["accepted_position0_reproduction"],
    }


def comparison(derived: dict[str, Any], rtl: dict[str, Any]) -> dict[str, Any]:
    exact = {
        "rmsnorm_output_mismatches": 0,
        "projection_accumulator_mismatches": 0,
        "projection_output_mismatches": 0,
        "rope_output_mismatches": 0,
        "attention_score_accumulator_mismatches": 0,
        "attention_score_output_mismatches": 0,
        "softmax_probability_mismatches": 0,
        "attention_compose_output_mismatches": 0,
        "o_projection_accumulator_mismatches": 0,
        "attention_residual_output_mismatches": 0,
        "post_attention_rmsnorm_output_mismatches": 0,
        "gate_up_projection_accumulator_mismatches": 0,
        "silu_output_mismatches": 0,
        "down_projection_accumulator_mismatches": 0,
        "final_residual_output_mismatches": 0,
        "kv_cache_tensor_mismatches": 0,
        "final_layer_output_integer_mismatches": 0,
        "projection_results": rtl["projection_results"],
    }
    return {
        "schema_version": 1,
        "status": "PASS_EXACT_TWO_POSITION_FULL_LAYER",
        "claim_boundary": "two tokenizer-derived positions through the complete layer-0 W4A8 RTL operator path",
        "zero_mismatch_policy": True,
        "rtl_vs_independent_python_fixed_point": exact,
        "cache_reuse": derived["cache_reuse"],
        "position1_rope_non_identity": True,
        "accepted_position0_reproduction": derived["accepted_position0_reproduction"],
        "final_layer_output_sha256_by_position": rtl["final_layer_output_sha256_by_position"],
        "simulator_cycles_descriptive_not_accelerator_latency": rtl["simulator_cycles"],
        "float_fidelity": {
            "quality_pass_claimed": False,
            "reason": "float-vs-W4A8 errors are descriptive and the prior 0.25/0.40 limits were selected after preview",
            "provenance_erratum": file_record(ERRATUM),
            "by_position": [position["metrics"] for position in derived["positions"]],
        },
    }


def validate_interface() -> dict[str, Any]:
    require(SPEC.is_file() and INTERFACE.is_file(), "mission-specific specification or interface is absent")
    interface = json.loads(INTERFACE.read_text(encoding="utf-8"))
    require(interface["mission_id"] == "extend-layer0-two-token-kv-rtl", "interface mission changed")
    for value in interface["rtl_interfaces"].values():
        path = ROOT / value["source"]
        require(sha256_file(path) == value["source_sha256"], f"interface-bound RTL changed: {path}")
        require(len(value["ports"]) == value["port_count"], f"interface port count changed: {value['module']}")
    for value in interface["maintained_testbenches"].values():
        path = ROOT / value["path"]
        require(sha256_file(path) == value["sha256"], f"interface-bound testbench changed: {path}")
    workload = interface["workload"]
    require(workload["projection"]["channels"] == 25344, "projection workload changed")
    require(workload["residual_shell"]["output_beats"] == 224, "residual workload changed")
    return interface


def source_bindings() -> dict[str, Any]:
    return {
        "runner": file_record(Path(__file__)),
        "specification": file_record(SPEC),
        "interface_manifest": file_record(INTERFACE),
        "frontier_contract": file_record(FRONTIER_CONTRACT),
        "frontier_attempt_sums": file_record(FRONTIER_ATTEMPT / "SHA256SUMS"),
        "accepted_contract": file_record(ACCEPTED_CONTRACT),
        "accepted_attempt_sums": file_record(ACCEPTED / "SHA256SUMS"),
        "provenance_erratum": file_record(ERRATUM),
        "model": canonical.file_record(canonical.MODEL),
        "adapter": file_record(canonical.ADAPTER),
        "testbenches": {name: file_record(path) for name, path in TESTBENCHES.items()},
        "rtl": {name: file_record(path) for name, path in CORES.items()},
        "shell": file_record(ROOT / "rtl/ace2_shell.sv"),
    }


def validate_predecessors() -> None:
    frontier.validate_contract()
    canonical.validate_contract()
    require(verify_sha256s(FRONTIER_ATTEMPT) == 85, "frontier immutable member count changed")
    require(verify_sha256s(ACCEPTED) > 0, "accepted full-layer attempt immutable check failed")


def prepare() -> None:
    require(not CONTRACT.exists(), f"contract already exists: {CONTRACT}")
    validate_interface()
    validate_predecessors()
    provenance = canonical.validate_provenance()
    contract = {
        "schema_version": 1,
        "mission_increment": MISSION,
        "claim": "complete two-position causal layer-0 W4A8 RTL comparison with explicit KV-cache reuse",
        "canonical_checkpoint": {
            "adapter_sha256": canonical.ADAPTER_SHA256,
            "base_model_sha256": canonical.MODEL_SHA256,
            "checkpoint_tree_sha256": canonical.CHECKPOINT_TREE_SHA256,
            "quality_claim_boundary": "checkpoint quality NO-GO remains in force",
        },
        "fixed_input": {
            "prompt_alias": frontier.PROMPT_ALIAS,
            "token_ids": provenance["token_ids"],
            "selected_positions": [0, 1],
            "selected_token_ids": provenance["token_ids"][:2],
        },
        "architecture": {
            "geometry": {
                "hidden": HIDDEN,
                "intermediate": INTERMEDIATE,
                "query_heads": Q_HEADS,
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
            },
            "operator_order": OPERATOR_ORDER,
            "grouped_query_mapping": "kv_head = query_head // 7",
        },
        "quantization": {
            "projection_weight": "signed W4 symmetric per output channel absmax/7",
            "activation_and_cache": "signed int8",
            "projection_accumulator": "signed int32 complete reduction",
            "rounding": "round-to-nearest ties-to-even",
            "q_k_v_metadata": "accepted token-0 metadata reused unchanged for both positions",
            "tail_metadata": "position0 accepted metadata; position1 deterministic per-operator scales frozen before RTL execution",
            "rope": "theta 1000000, Q9 activation scale one, Q15 cosine/sine split-half",
            "residual": "common-scale signed-int8 saturating add",
        },
        "specification": file_record(SPEC),
        "interface_manifest": file_record(INTERFACE),
        "evaluator_policy": {
            "integer_mismatch_max": 0,
            "projection_results_exact": 25344,
            "kv_cache_bytes_exact": 512,
            "final_output_bytes_exact": 1792,
            "float_quality_pass_allowed": False,
        },
        "source_bindings": source_bindings(),
        "commands": {
            "preflight": ".venv/bin/python tools/run_lora_v4_layer0_two_token_kv_full_rtl.py --preflight",
            "preview": ".venv/bin/python tools/run_lora_v4_layer0_two_token_kv_full_rtl.py --preview",
            "prepare": ".venv/bin/python tools/run_lora_v4_layer0_two_token_kv_full_rtl.py --prepare",
            "run": ".venv/bin/python tools/run_lora_v4_layer0_two_token_kv_full_rtl.py --run",
            "check": ".venv/bin/python tools/run_lora_v4_layer0_two_token_kv_full_rtl.py --check",
        },
        "attempt_policy": {
            "official_full_layer_attempt": "attempt-0001",
            "immutable_after_execution": True,
            "predecessor_attempts_modified": False,
        },
    }
    write_json(CONTRACT, contract)
    print(json.dumps({"status": "FROZEN", "contract": relative(CONTRACT), "sha256": sha256_file(CONTRACT)}, sort_keys=True))


def validate_contract() -> dict[str, Any]:
    require(CONTRACT.is_file(), "full two-position contract is absent")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    validate_interface()
    validate_predecessors()
    require(contract["source_bindings"] == source_bindings(), "source binding changed after freeze")
    require(contract["fixed_input"]["token_ids"] == canonical.tokenizer_ids(), "tokenizer IDs changed")
    return contract


def execute(attempt: Path, *, official: bool) -> dict[str, Any]:
    global ACTIVE_ATTEMPT
    ACTIVE_ATTEMPT = attempt
    require(not attempt.exists(), f"attempt already exists: {attempt}")
    if official:
        contract = validate_contract()
    else:
        validate_interface()
        validate_predecessors()
        contract = None
    require(shutil.which("iverilog") is not None and shutil.which("vvp") is not None, "Icarus tools unavailable")
    attempt.mkdir(parents=True)
    started = time.monotonic()
    derived = derive()
    tensors = persist_tensors(derived)
    vectors = persist_vectors(derived)
    write_json(attempt / "conversion.json", conversion_summary(derived))
    rtl = run_rtl(derived)
    write_json(attempt / "rtl_execution.json", rtl)
    result = comparison(derived, rtl)
    write_json(attempt / "comparison.json", result)
    manifest = {
        "schema_version": 1,
        "status": result["status"],
        "official_full_layer_two_position_attempt": official,
        "contract": file_record(CONTRACT) if official else None,
        "specification": file_record(SPEC),
        "interface_manifest": file_record(INTERFACE),
        "predecessors": {
            "attention_frontier": file_record(FRONTIER_ATTEMPT / "SHA256SUMS"),
            "accepted_token0_full_layer": file_record(ACCEPTED / "SHA256SUMS"),
        },
        "token_ids": derived["token_ids"],
        "positions": [0, 1],
        "operator_order": OPERATOR_ORDER,
        "cache_reuse": derived["cache_reuse"],
        "accepted_position0_reproduction": derived["accepted_position0_reproduction"],
        "tensors": tensors,
        "vectors": vectors,
        "conversion": file_record(attempt / "conversion.json"),
        "rtl_execution": file_record(attempt / "rtl_execution.json"),
        "comparison": file_record(attempt / "comparison.json"),
        "elapsed_seconds": time.monotonic() - started,
        "source_bindings": source_bindings(),
        "commands": contract["commands"] if contract else None,
    }
    write_json(attempt / "manifest.json", manifest)
    write_sha256s(attempt)
    print(
        json.dumps(
            {
                "status": result["status"],
                "attempt": relative(attempt),
                "projection_results": rtl["projection_results"],
                "final_layer_output_sha256_by_position": rtl["final_layer_output_sha256_by_position"],
                "cache_reuse": derived["cache_reuse"],
            },
            sort_keys=True,
        )
    )
    return result


def preflight() -> None:
    validate_interface()
    validate_predecessors()
    derived = derive()
    print(
        json.dumps(
            {
                "status": "PASS_DERIVATION",
                "token_ids": derived["token_ids"][:2],
                "cache_reuse": derived["cache_reuse"],
                "accepted_position0_reproduction": derived["accepted_position0_reproduction"],
                "float_metrics_by_position_descriptive_only": [
                    position["metrics"] for position in derived["positions"]
                ],
            },
            sort_keys=True,
        )
    )


def check() -> None:
    contract = validate_contract()
    require(OFFICIAL.is_dir(), "official full two-position attempt is absent")
    members = verify_sha256s(OFFICIAL)
    manifest = json.loads((OFFICIAL / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((OFFICIAL / "comparison.json").read_text(encoding="utf-8"))
    execution = json.loads((OFFICIAL / "rtl_execution.json").read_text(encoding="utf-8"))
    require(manifest["status"] == result["status"] == "PASS_EXACT_TWO_POSITION_FULL_LAYER", "official status is not PASS")
    require(manifest["contract"]["sha256"] == sha256_file(CONTRACT), "contract binding changed")
    require(execution["projection_results"] == contract["evaluator_policy"]["projection_results_exact"], "projection result count changed")
    for key, value in result["rtl_vs_independent_python_fixed_point"].items():
        if key.endswith("_mismatches"):
            require(value == 0, f"nonzero mismatch count: {key}")
    require(result["position1_rope_non_identity"], "position-1 RoPE evidence missing")
    require(result["cache_reuse"]["compose_output_changes_when_cached_v_zeroed_heads"] > 0, "cache counterfactual missing")
    require(len(result["final_layer_output_sha256_by_position"]) == 2, "final output hashes missing")
    print(
        json.dumps(
            {
                "status": "PASS_IMMUTABLE_FULL_LAYER_CHECK",
                "members": members,
                "attempt": relative(OFFICIAL),
                "projection_results": execution["projection_results"],
                "final_layer_output_sha256_by_position": execution["final_layer_output_sha256_by_position"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--preview", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.preview:
        execute(next_preview(), official=False)
    elif args.prepare:
        prepare()
    elif args.run:
        execute(OFFICIAL, official=True)
    else:
        check()


if __name__ == "__main__":
    main()
