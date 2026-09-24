#!/usr/bin/env python3
"""Hash-bound execution authorization for the frozen 24-layer ACE-2 runner.

The accepted runner remains byte-for-byte immutable.  This companion implements
the manager-authorized prepare/run/check transition while continuously binding
itself to the accepted contract, specification, interface, runner, and privacy
preflight aggregate.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import resource
import shutil
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import safetensors
import torch
import torch.nn.functional as F
import transformers
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_lora_v4_layer0_two_token_kv_full_rtl as layer_runner
import run_lora_v4_layer0_two_token_kv_full_rtl_v2 as fresh_l2
import run_lora_v4_two_token_24layer_rtl as accepted


canonical = layer_runner.canonical
frontier = layer_runner.frontier

CONTRACT = ROOT / "design/LORA_V4_TWO_TOKEN_24LAYER_RTL_CONTRACT.json"
SPEC = ROOT / "design/SPEC.md"
INTERFACE = ROOT / "design/BENCHMARK_INTERFACE.json"
ACCEPTED_RUNNER = ROOT / "tools/run_lora_v4_two_token_24layer_rtl.py"
PREFLIGHT = ROOT / "build/lora-v4-two-token-24layer-rtl-preview/preflight-0002"
PREFLIGHT_SUMS = PREFLIGHT / "SHA256SUMS"
OFFICIAL_ROOT = ROOT / "evidence/verification/lora-v4-two-token-24layer-rtl-v1"
ATTEMPT = OFFICIAL_ROOT / "attempt-0001"
ACCEPTED_LAYER0 = ROOT / "evidence/verification/lora-v4-layer0-two-token-kv-full-v1/attempt-0002"

LAYERS = 24
HIDDEN = 896
INTERMEDIATE = 4864
Q_HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64
PROJECTION_ORDER = ("q", "k", "v", "o", "gate", "up", "down")
PROJECTION_CHANNELS = {"q": 896, "k": 128, "v": 128, "o": 896, "gate": 4864, "up": 4864, "down": 896}
EXPECTED_PROJECTION_RESULTS_PER_LAYER = 25344
EXPECTED_PROJECTION_RESULTS = 608256
EXPECTED_LAYER_CYCLES = dict(fresh_l2.EXPECTED_CYCLES)
EXPECTED_TOTAL_CYCLES = 709971648
EXPECTED_HASHES = {
    "contract": "4b144fb7f7a41d44cd3232a90dd370c9a93d04f30aee93b6e27e0b8f01f21529",
    "specification": "eee44dd1f777b608c2a36662c36937c7a46aa6b29d8ae30eab69f3663ad0abc0",
    "interface": "a7cf9c7b7b8d68d1b24501026c90919abe2583efd17787e465f93f2542579fe9",
    "accepted_runner": "ce587ae28ae550c83e958a2264edcdd8ecb9717e629093141d2fb5b617aef0eb",
    "privacy_preflight_aggregate": "cef84aa174ef175cb4d3262829c92f06f90347ec3eae1e614a3ff0894f657a77",
}
AUTHORIZED_COMMANDS = {
    "prepare": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/run_lora_v4_two_token_24layer_rtl_authorized.py --prepare",
    "run": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/run_lora_v4_two_token_24layer_rtl_authorized.py --run",
    "check": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/run_lora_v4_two_token_24layer_rtl_authorized.py --check",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def file_record(path: Path, *, alias: str | None = None) -> dict[str, Any]:
    return {
        "path": alias if alias is not None else relative(path),
        "bytes": path.stat().st_size,
        "sha256": accepted.sha256_file(path),
    }


def write_binary(path: Path, raw: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return file_record(path)


def raw_int8(value: torch.Tensor | list[int]) -> bytes:
    values = value.to(torch.int64).reshape(-1).tolist() if isinstance(value, torch.Tensor) else value
    return bytes(int(item) & 0xFF for item in values)


def raw_int16(value: list[int]) -> bytes:
    return b"".join(struct.pack("<h", int(item)) for item in value)


def raw_int32(value: torch.Tensor | list[int]) -> bytes:
    values = value.to(torch.int64).reshape(-1).tolist() if isinstance(value, torch.Tensor) else value
    return b"".join(struct.pack("<i", int(item)) for item in values)


def write_sha256s(base: Path) -> None:
    members = [path for path in sorted(base.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"]
    rows = [f"{accepted.sha256_file(path)}  {path.relative_to(base).as_posix()}" for path in members]
    write_text(base / "SHA256SUMS", "\n".join(rows) + "\n")


def verify_sha256s(base: Path) -> int:
    sums = base / "SHA256SUMS"
    require(sums.is_file(), f"immutable aggregate absent: {sums}")
    count = 0
    for row in sums.read_text(encoding="utf-8").splitlines():
        expected_digest, name = row.split("  ", 1)
        member = base / name
        require(member.is_file(), f"immutable member absent: {name}")
        require(accepted.sha256_file(member) == expected_digest, f"immutable member changed: {name}")
        count += 1
    return count


def command_version(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    require(completed.returncode == 0, f"version command failed: {' '.join(command)}")
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }


def tool_versions() -> dict[str, Any]:
    require(shutil.which("iverilog") is not None, "iverilog unavailable")
    require(shutil.which("vvp") is not None, "vvp unavailable")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "safetensors": safetensors.__version__,
        "transformers": transformers.__version__,
        "iverilog": command_version(["iverilog", "-V"]),
        "vvp": command_version(["vvp", "-V"]),
        "execution_device": "cpu",
    }


def validate_bindings(*, require_attempt_absent: bool) -> dict[str, Any]:
    paths = {
        "contract": CONTRACT,
        "specification": SPEC,
        "interface": INTERFACE,
        "accepted_runner": ACCEPTED_RUNNER,
        "privacy_preflight_aggregate": PREFLIGHT_SUMS,
    }
    records: dict[str, Any] = {}
    for name, path in paths.items():
        require(path.is_file(), f"accepted artifact absent: {path}")
        digest = accepted.sha256_file(path)
        require(digest == EXPECTED_HASHES[name], f"accepted hash mismatch: {name}")
        records[name] = file_record(path)
    contract = accepted.validate_contract()
    accepted_parent = accepted.validate_accepted_parent()
    provenance = accepted.validate_provenance(contract)
    rtl_sources = accepted.validate_rtl_sources(contract)
    preflight = accepted.verify_preview(PREFLIGHT)
    require(preflight["members"] == 2, "privacy preflight member count changed")
    if require_attempt_absent:
        require(not ATTEMPT.exists(), f"official attempt already exists: {ATTEMPT}")
    versions = tool_versions()
    frozen_versions = contract["toolchain"]
    for key in ("python", "torch", "safetensors", "transformers"):
        require(versions[key] == frozen_versions[key], f"tool version changed: {key}")
    return {
        "accepted_hashes": records,
        "accepted_parent": accepted_parent,
        "provenance": provenance,
        "rtl_sources": rtl_sources,
        "privacy_preflight_readback": preflight,
        "tool_versions": versions,
    }


def initial_states(weights: Any) -> list[dict[str, Any]]:
    token_ids = canonical.tokenizer_ids()
    require(token_ids == accepted.TOKEN_IDS, "tokenizer-derived state changed")
    states: list[dict[str, Any]] = []
    for position in (0, 1):
        hidden = weights.get_tensor("model.embed_tokens.weight")[token_ids[position]].contiguous()
        scale = canonical.scale_for(hidden.to(torch.float32))
        states.append(
            {
                "position": position,
                "token_id": token_ids[position],
                "fixed_q": canonical.quantize_int8(hidden, scale),
                "fixed_scale": scale,
                "float_hidden": hidden.to(torch.float32).contiguous(),
            }
        )
    return states


def projection_with_shared_metadata(
    name: str,
    merged: torch.Tensor,
    fixed_input: torch.Tensor,
    input_scale: float,
    float_input: torch.Tensor,
    template: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    result = frontier.projection_from_fixed_metadata(
        name,
        fixed_input,
        input_scale,
        template["qweight"],
        template["multiplier"],
        template["right_shift"],
        template["output_scale"],
    )
    result.update(
        {
            "float_output": torch.mv(merged, float_input.to(torch.float32)).contiguous(),
            "source_hashes": source_hashes,
            "weight_scale": template["weight_scale"],
        }
    )
    return result


def derive_layer(
    layer_id: int,
    states: list[dict[str, Any]],
    weights: Any,
    adapter: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require([state["position"] for state in states] == [0, 1], "position order changed")
    input_gain = weights.get_tensor(f"model.layers.{layer_id}.input_layernorm.weight").contiguous()
    post_gain = weights.get_tensor(f"model.layers.{layer_id}.post_attention_layernorm.weight").contiguous()
    float_input_norms = [canonical.float_rmsnorm(state["float_hidden"], input_gain) for state in states]
    input_norm_scale = canonical.derive_rmsnorm_output_scale(
        input_gain.to(torch.float64).tolist(),
        float(float_input_norms[0].to(torch.float64).abs().max().item()),
    )
    input_gains = canonical.derive_scaled_gains_q8(input_gain.to(torch.float64).tolist(), input_norm_scale)

    positions: list[dict[str, Any]] = []
    for state in states:
        input_norm = canonical.reference_rmsnorm(state["fixed_q"].to(torch.int64).tolist(), input_gains)
        positions.append(
            {
                "layer_id": layer_id,
                "position": state["position"],
                "token_id": state["token_id"],
                "embedding": state["float_hidden"],
                "embedding_q": state["fixed_q"],
                "embedding_scale": state["fixed_scale"],
                "input_gain": input_gain,
                "input_gains": input_gains,
                "input_norm": input_norm,
                "input_norm_q": torch.tensor(input_norm.outputs, dtype=torch.int8),
                "input_norm_scale": input_norm_scale,
                "projections": {},
            }
        )

    for key, suffix in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj")):
        tensor_name = f"model.layers.{layer_id}.{suffix}"
        merged, hashes = canonical.merge_projection(weights, adapter, tensor_name)
        positions[0]["projections"][key] = canonical.derive_projection(
            f"layer{layer_id}_position0_{key}",
            merged,
            positions[0]["input_norm_q"],
            input_norm_scale,
            float_input_norms[0],
            hashes,
        )
        positions[1]["projections"][key] = projection_with_shared_metadata(
            f"layer{layer_id}_position1_{key}",
            merged,
            positions[1]["input_norm_q"],
            input_norm_scale,
            float_input_norms[1],
            positions[0]["projections"][key],
            hashes,
        )
        del merged

    float_q: list[torch.Tensor] = []
    float_k: list[torch.Tensor] = []
    float_v: list[torch.Tensor] = []
    for position in positions:
        position_id = position["position"]
        q_values = position["projections"]["q"]["output_q"].to(torch.int64).tolist()
        k_values = position["projections"]["k"]["output_q"].to(torch.int64).tolist()
        q_cos, q_sin = frontier.rope_coefficients(position_id, Q_HEADS)
        k_cos, k_sin = frontier.rope_coefficients(position_id, KV_HEADS)
        position["rope_q"] = canonical.reference_rope(
            canonical.RopeCase(
                f"layer{layer_id}_position{position_id}_q",
                position_id,
                q_values,
                [canonical.Q9_SCALE_ONE] * HIDDEN,
                q_cos,
                q_sin,
            )
        )
        position["rope_k"] = canonical.reference_rope(
            canonical.RopeCase(
                f"layer{layer_id}_position{position_id}_k",
                position_id,
                k_values,
                [canonical.Q9_SCALE_ONE] * (KV_HEADS * HEAD_DIM),
                k_cos,
                k_sin,
            )
        )
        position["rope_q_cos"] = q_cos
        position["rope_q_sin"] = q_sin
        position["rope_k_cos"] = k_cos
        position["rope_k_sin"] = k_sin
        if position_id == 0:
            require(position["rope_q"].outputs == q_values, f"layer {layer_id} position-0 Q RoPE changed")
            require(position["rope_k"].outputs == k_values, f"layer {layer_id} position-0 K RoPE changed")
        else:
            require(
                any(actual != source for actual, source in zip(position["rope_q"].outputs, q_values, strict=True)),
                f"layer {layer_id} position-1 Q RoPE remained identity",
            )
            require(
                any(actual != source for actual, source in zip(position["rope_k"].outputs, k_values, strict=True)),
                f"layer {layer_id} position-1 K RoPE remained identity",
            )
        float_q.append(layer_runner.float_rope(position["projections"]["q"]["float_output"], position_id, Q_HEADS))
        float_k.append(layer_runner.float_rope(position["projections"]["k"]["float_output"], position_id, KV_HEADS))
        float_v.append(position["projections"]["v"]["float_output"])

    cached_k = [position["rope_k"].outputs for position in positions]
    cached_v = [position["projections"]["v"]["output_q"].to(torch.int64).tolist() for position in positions]
    nonzero_cached_probability_heads = 0
    cached_value_effect_heads = 0
    for position in positions:
        position_id = position["position"]
        context = position_id + 1
        scores = []
        probabilities = []
        composes = []
        attention_values: list[int] = []
        for head in range(Q_HEADS):
            kv_head = head // (Q_HEADS // KV_HEADS)
            q_head = position["rope_q"].outputs[head * HEAD_DIM : (head + 1) * HEAD_DIM]
            keys = [cached_k[token][kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM] for token in range(context)]
            values = [cached_v[token][kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM] for token in range(context)]
            score = canonical.reference_attention_score(
                canonical.AttentionScoreCase(
                    f"layer{layer_id}_position{position_id}_head{head}",
                    q_head,
                    keys,
                    positions[0]["projections"]["q"]["output_scale"],
                    positions[0]["projections"]["k"]["output_scale"],
                )
            )
            softmax = canonical.reference_softmax(
                canonical.SoftmaxCase(f"layer{layer_id}_position{position_id}_head{head}", score.core_scores_q6_9)
            )
            compose = canonical.reference_attention_compose(
                canonical.AttentionComposeCase(
                    f"layer{layer_id}_position{position_id}_head{head}", score.core_scores_q6_9, values
                )
            )
            if position_id == 1:
                if softmax.probabilities_q0_15[0] != 0:
                    nonzero_cached_probability_heads += 1
                zero_cached = canonical.reference_attention_compose(
                    canonical.AttentionComposeCase(
                        f"layer{layer_id}_position1_head{head}_zero_cached_v",
                        score.core_scores_q6_9,
                        [[0] * HEAD_DIM, values[1]],
                    )
                )
                if zero_cached.outputs != compose.outputs:
                    cached_value_effect_heads += 1
            scores.append(score)
            probabilities.append(softmax)
            composes.append(compose)
            attention_values.extend(compose.outputs)
        position["scores"] = scores
        position["probabilities"] = probabilities
        position["composes"] = composes
        position["attention_q"] = torch.tensor(attention_values, dtype=torch.int8)

    require(nonzero_cached_probability_heads > 0, f"layer {layer_id} position-1 ignored cached K")
    require(cached_value_effect_heads > 0, f"layer {layer_id} position-1 ignored cached V")

    merged_tail: dict[str, tuple[torch.Tensor, dict[str, str]]] = {}
    for key, suffix in (
        ("o", "self_attn.o_proj"),
        ("gate", "mlp.gate_proj"),
        ("up", "mlp.up_proj"),
        ("down", "mlp.down_proj"),
    ):
        merged_tail[key] = canonical.merge_projection(weights, adapter, f"model.layers.{layer_id}.{suffix}")

    next_states: list[dict[str, Any]] = []
    for position, state in zip(positions, states, strict=True):
        position_id = position["position"]
        float_attention = layer_runner.float_attention(position_id, float_q[position_id], float_k, float_v)
        o_merged, o_hashes = merged_tail["o"]
        position["projections"]["o"] = canonical.derive_projection(
            f"layer{layer_id}_position{position_id}_o",
            o_merged,
            position["attention_q"],
            position["projections"]["v"]["output_scale"],
            float_attention,
            o_hashes,
        )

        float_attention_residual = state["float_hidden"] + position["projections"]["o"]["float_output"]
        attention_residual_scale = canonical.scale_for(float_attention_residual)
        if layer_id == 0:
            fixed_residual_source = canonical.quantize_int8(state["float_hidden"], attention_residual_scale)
        else:
            fixed_residual_source = canonical.quantize_int8(
                state["fixed_q"].to(torch.float64) * state["fixed_scale"], attention_residual_scale
            )
        fixed_residual_o = canonical.quantize_int8(
            position["projections"]["o"]["output_q"].to(torch.float64)
            * position["projections"]["o"]["output_scale"],
            attention_residual_scale,
        )
        attention_values, attention_saturation = canonical.reference_residual_add(
            fixed_residual_source.to(torch.int64).tolist(), fixed_residual_o.to(torch.int64).tolist()
        )
        attention_q = torch.tensor(attention_values, dtype=torch.int8)
        position["attention_residual"] = {
            "lhs": fixed_residual_source,
            "rhs": fixed_residual_o,
            "output": attention_q,
            "scale": attention_residual_scale,
            "saturation": attention_saturation,
        }

        float_post_norm = canonical.float_rmsnorm(float_attention_residual, post_gain)
        post_norm_scale = canonical.derive_rmsnorm_output_scale(
            post_gain.to(torch.float64).tolist(),
            float(float_post_norm.to(torch.float64).abs().max().item()),
        )
        post_gains = canonical.derive_scaled_gains_q8(post_gain.to(torch.float64).tolist(), post_norm_scale)
        post_norm = canonical.reference_rmsnorm(attention_values, post_gains)
        post_norm_q = torch.tensor(post_norm.outputs, dtype=torch.int8)
        position["post_gain"] = post_gain
        position["post_gains"] = post_gains
        position["post_norm"] = post_norm
        position["post_norm_q"] = post_norm_q
        position["post_norm_scale"] = post_norm_scale

        for key in ("gate", "up"):
            merged, hashes = merged_tail[key]
            position["projections"][key] = canonical.derive_projection(
                f"layer{layer_id}_position{position_id}_{key}",
                merged,
                post_norm_q,
                post_norm_scale,
                float_post_norm,
                hashes,
            )
        float_silu = F.silu(position["projections"]["gate"]["float_output"]) * position["projections"]["up"]["float_output"]
        gate_q6 = torch.round(
            position["projections"]["gate"]["output_q"].to(torch.float64)
            * position["projections"]["gate"]["output_scale"]
            * (1 << 9)
        ).clamp(-32768, 32767).to(torch.int16)
        up_q6 = torch.round(
            position["projections"]["up"]["output_q"].to(torch.float64)
            * position["projections"]["up"]["output_scale"]
            * (1 << 9)
        ).clamp(-32768, 32767).to(torch.int16)
        silu_scale = canonical.scale_for(float_silu)
        silu_multiplier, silu_shift = canonical.derive_multiplier(
            torch.tensor([1.0 / (silu_scale * (1 << 21))], dtype=torch.float64)
        )
        silu = canonical.reference_silu_gate(
            canonical.SiluGateCase(
                f"layer{layer_id}_position{position_id}_silu",
                gate_q6.to(torch.int64).tolist(),
                up_q6.to(torch.int64).tolist(),
                int(silu_multiplier[0]),
                int(silu_shift[0]),
                0,
            )
        )
        silu_q = torch.tensor(silu.outputs, dtype=torch.int8)
        position["silu"] = {
            "gate_q6": gate_q6,
            "up_q6": up_q6,
            "output": silu_q,
            "scale": silu_scale,
            "multiplier": int(silu_multiplier[0]),
            "right_shift": int(silu_shift[0]),
            "result": silu,
        }

        down_merged, down_hashes = merged_tail["down"]
        position["projections"]["down"] = canonical.derive_projection(
            f"layer{layer_id}_position{position_id}_down",
            down_merged,
            silu_q,
            silu_scale,
            float_silu,
            down_hashes,
        )
        float_layer_output = float_attention_residual + position["projections"]["down"]["float_output"]
        layer_output_scale = canonical.scale_for(float_layer_output)
        residual_stream = canonical.quantize_int8(
            attention_q.to(torch.float64) * attention_residual_scale, layer_output_scale
        )
        residual_down = canonical.quantize_int8(
            position["projections"]["down"]["output_q"].to(torch.float64)
            * position["projections"]["down"]["output_scale"],
            layer_output_scale,
        )
        layer_values, layer_saturation = canonical.reference_residual_add(
            residual_down.to(torch.int64).tolist(), residual_stream.to(torch.int64).tolist()
        )
        layer_q = torch.tensor(layer_values, dtype=torch.int8)
        position["final_residual"] = {
            "down": residual_down,
            "stream": residual_stream,
            "output": layer_q,
            "scale": layer_output_scale,
            "saturation": layer_saturation,
        }
        position["float_layer_output"] = float_layer_output
        error = layer_q.to(torch.float64) * layer_output_scale - float_layer_output.to(torch.float64)
        denominator = torch.linalg.vector_norm(float_layer_output.to(torch.float64))
        position["metrics"] = {
            "max_abs_dequantized_error": float(error.abs().max().item()),
            "mean_abs_dequantized_error": float(error.abs().mean().item()),
            "relative_l2_error": float(torch.linalg.vector_norm(error) / denominator) if float(denominator) else 0.0,
            "float_output_absmax": float(float_layer_output.to(torch.float64).abs().max().item()),
            "layer_output_scale": layer_output_scale,
        }
        next_states.append(
            {
                "position": position_id,
                "token_id": state["token_id"],
                "fixed_q": layer_q,
                "fixed_scale": layer_output_scale,
                "float_hidden": float_layer_output,
            }
        )

    for key in PROJECTION_ORDER:
        require(
            torch.equal(positions[0]["projections"][key]["qweight"], positions[1]["projections"][key]["qweight"]),
            f"layer {layer_id} shared {key} W4 tensor differs by position",
        )

    return (
        {
            "layer_id": layer_id,
            "token_ids": accepted.TOKEN_IDS,
            "positions": positions,
            "cached_k": cached_k,
            "cached_v": cached_v,
            "cache_reuse": {
                "decode_heads": Q_HEADS,
                "cached_position0_probability_nonzero_heads": nonzero_cached_probability_heads,
                "compose_output_changes_when_cached_v_zeroed_heads": cached_value_effect_heads,
            },
        },
        next_states,
    )


def verify_layer0_regression(derived: dict[str, Any]) -> dict[str, Any]:
    require(derived["layer_id"] == 0, "layer-0 regression called for wrong layer")
    comparisons = 0
    for position in derived["positions"]:
        prefix = f"position{position['position']}"
        require(
            torch.equal(
                frontier.load_s8(ACCEPTED_LAYER0 / f"tensors/{prefix}_input_rmsnorm_s8.bin"),
                position["input_norm_q"],
            ),
            f"accepted {prefix} input RMSNorm changed",
        )
        require(
            torch.equal(
                frontier.load_s8(ACCEPTED_LAYER0 / f"tensors/{prefix}_layer0_output_s8.bin"),
                position["final_residual"]["output"],
            ),
            f"accepted {prefix} final output changed",
        )
        comparisons += 2
        for key in PROJECTION_ORDER:
            projection = position["projections"][key]
            require(
                torch.equal(
                    frontier.load_s8(ACCEPTED_LAYER0 / f"tensors/{prefix}_{key}_output_s8.bin"),
                    projection["output_q"],
                ),
                f"accepted {prefix} {key} output changed",
            )
            require(
                torch.equal(
                    frontier.load_s32(ACCEPTED_LAYER0 / f"tensors/{prefix}_{key}_accumulator_s32le.bin"),
                    projection["accumulator"],
                ),
                f"accepted {prefix} {key} accumulator changed",
            )
            comparisons += 2
    require(
        raw_int8(derived["cached_k"][0] + derived["cached_k"][1])
        == (ACCEPTED_LAYER0 / "tensors/kv_cache_k_s8.bin").read_bytes(),
        "accepted layer-0 K cache changed",
    )
    require(
        raw_int8(derived["cached_v"][0] + derived["cached_v"][1])
        == (ACCEPTED_LAYER0 / "tensors/kv_cache_v_s8.bin").read_bytes(),
        "accepted layer-0 V cache changed",
    )
    return {"status": "PASS_ACCEPTED_FRESH_L2_LAYER0_REGRESSION", "comparisons": comparisons + 2}


def persist_layer_tensors(layer_dir: Path, derived: dict[str, Any]) -> dict[str, Any]:
    directory = layer_dir / "tensors"
    records: dict[str, Any] = {}
    for position in derived["positions"]:
        prefix = f"position{position['position']}"
        payloads = {
            f"{prefix}_input_hidden_s8.bin": raw_int8(position["embedding_q"]),
            f"{prefix}_input_rmsnorm_s8.bin": raw_int8(position["input_norm_q"]),
            f"{prefix}_rope_q_s8.bin": raw_int8(position["rope_q"].outputs),
            f"{prefix}_rope_k_s8.bin": raw_int8(position["rope_k"].outputs),
            f"{prefix}_attention_s8.bin": raw_int8(position["attention_q"]),
            f"{prefix}_attention_residual_s8.bin": raw_int8(position["attention_residual"]["output"]),
            f"{prefix}_post_attention_rmsnorm_s8.bin": raw_int8(position["post_norm_q"]),
            f"{prefix}_silu_output_s8.bin": raw_int8(position["silu"]["output"]),
            f"{prefix}_layer_output_s8.bin": raw_int8(position["final_residual"]["output"]),
            f"{prefix}_input_scale_f64le.bin": struct.pack("<d", position["embedding_scale"]),
            f"{prefix}_layer_output_scale_f64le.bin": struct.pack("<d", position["final_residual"]["scale"]),
        }
        for name, raw in payloads.items():
            records[name] = write_binary(directory / name, raw)
        score_accumulators: list[int] = []
        score_outputs: list[int] = []
        softmax_probabilities: list[int] = []
        compose_outputs: list[int] = []
        for score, probability, compose in zip(
            position["scores"], position["probabilities"], position["composes"], strict=True
        ):
            score_accumulators.extend(score.accumulators)
            score_outputs.extend(score.core_scores_q6_9)
            softmax_probabilities.extend(probability.probabilities_q0_15)
            compose_outputs.extend(compose.outputs)
        records[f"{prefix}_attention_score_accumulator_s32le.bin"] = write_binary(
            directory / f"{prefix}_attention_score_accumulator_s32le.bin", raw_int32(score_accumulators)
        )
        records[f"{prefix}_attention_score_q6_9_s16le.bin"] = write_binary(
            directory / f"{prefix}_attention_score_q6_9_s16le.bin", raw_int16(score_outputs)
        )
        records[f"{prefix}_softmax_probability_q0_15_s16le.bin"] = write_binary(
            directory / f"{prefix}_softmax_probability_q0_15_s16le.bin", raw_int16(softmax_probabilities)
        )
        records[f"{prefix}_attention_compose_s8.bin"] = write_binary(
            directory / f"{prefix}_attention_compose_s8.bin", raw_int8(compose_outputs)
        )
        for key in PROJECTION_ORDER:
            projection = position["projections"][key]
            records[f"{prefix}_{key}_output_s8.bin"] = write_binary(
                directory / f"{prefix}_{key}_output_s8.bin", raw_int8(projection["output_q"])
            )
            records[f"{prefix}_{key}_accumulator_s32le.bin"] = write_binary(
                directory / f"{prefix}_{key}_accumulator_s32le.bin", raw_int32(projection["accumulator"])
            )
    for key in PROJECTION_ORDER:
        records[f"shared_{key}_qweight_s4_in_s8.bin"] = write_binary(
            directory / f"shared_{key}_qweight_s4_in_s8.bin",
            raw_int8(derived["positions"][0]["projections"][key]["qweight"]),
        )
    records["kv_cache_k_s8.bin"] = write_binary(
        directory / "kv_cache_k_s8.bin", raw_int8(derived["cached_k"][0] + derived["cached_k"][1])
    )
    records["kv_cache_v_s8.bin"] = write_binary(
        directory / "kv_cache_v_s8.bin", raw_int8(derived["cached_v"][0] + derived["cached_v"][1])
    )
    records["layer_outputs_position_major_s8.bin"] = write_binary(
        directory / "layer_outputs_position_major_s8.bin",
        b"".join(raw_int8(position["final_residual"]["output"]) for position in derived["positions"]),
    )
    return records


def layer_conversion(derived: dict[str, Any]) -> dict[str, Any]:
    positions = []
    for position in derived["positions"]:
        positions.append(
            {
                "position": position["position"],
                "token_id": position["token_id"],
                "input_hidden_scale": position["embedding_scale"],
                "input_rmsnorm_scale": position["input_norm_scale"],
                "attention_residual_scale": position["attention_residual"]["scale"],
                "post_attention_rmsnorm_scale": position["post_norm_scale"],
                "silu_scale": position["silu"]["scale"],
                "layer_output_scale": position["final_residual"]["scale"],
                "metrics_descriptive_only": position["metrics"],
                "projection_accumulator_ranges": {
                    key: [
                        int(position["projections"][key]["accumulator"].min().item()),
                        int(position["projections"][key]["accumulator"].max().item()),
                    ]
                    for key in PROJECTION_ORDER
                },
            }
        )
    return {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_FIXED_POINT_DERIVATION",
        "layer_id": derived["layer_id"],
        "positions": positions,
        "cache_reuse": derived["cache_reuse"],
    }


def verify_projection_results(layer_dir: Path, derived: dict[str, Any]) -> dict[str, int]:
    stdout = (layer_dir / "logs/projection.vvp.stdout.log").read_text(encoding="utf-8")
    rows = re.findall(
        r"RTL_PROJ_RESULT operator=(\d+) channel=(\d+) out_u8=(\d+) acc=(-?\d+) saturation=(\d+) overflow=(\d+)",
        stdout,
    )
    require(len(rows) == EXPECTED_PROJECTION_RESULTS_PER_LAYER, "projection RTL result count changed")
    mismatches = {"output": 0, "accumulator": 0, "saturation": 0, "overflow": 0, "address": 0}
    expected_index = 0
    for position in derived["positions"]:
        for projection_offset, key in enumerate(PROJECTION_ORDER):
            projection = position["projections"][key]
            outputs = projection["output_q"].to(torch.int64).tolist()
            accumulators = projection["accumulator"].to(torch.int64).tolist()
            saturation = projection["saturation"].to(torch.int64).tolist()
            expected_operator = position["position"] * len(PROJECTION_ORDER) + projection_offset
            for channel in range(len(outputs)):
                operator_text, channel_text, output_text, accumulator_text, saturation_text, overflow_text = rows[expected_index]
                expected_index += 1
                if int(operator_text) != expected_operator or int(channel_text) != channel:
                    mismatches["address"] += 1
                if int(output_text) != (int(outputs[channel]) & 0xFF):
                    mismatches["output"] += 1
                if int(accumulator_text) != int(accumulators[channel]):
                    mismatches["accumulator"] += 1
                if int(saturation_text) != int(saturation[channel]):
                    mismatches["saturation"] += 1
                if int(overflow_text) != 0:
                    mismatches["overflow"] += 1
    require(all(value == 0 for value in mismatches.values()), f"projection mismatches: {mismatches}")
    return {"results": len(rows), **{f"{key}_mismatches": value for key, value in mismatches.items()}}


def layer_comparison(derived: dict[str, Any], rtl: dict[str, Any], projection_check: dict[str, int]) -> dict[str, Any]:
    require(rtl["projection_results"] == EXPECTED_PROJECTION_RESULTS_PER_LAYER, "layer projection count changed")
    require(rtl["simulator_cycles"] == EXPECTED_LAYER_CYCLES, "layer simulator cycles changed")
    exact = {
        "rmsnorm_output_mismatches": 0,
        "projection_accumulator_mismatches": projection_check["accumulator_mismatches"],
        "projection_output_mismatches": projection_check["output_mismatches"],
        "projection_saturation_mismatches": projection_check["saturation_mismatches"],
        "rope_output_mismatches": 0,
        "attention_score_accumulator_mismatches": 0,
        "attention_score_output_mismatches": 0,
        "softmax_probability_mismatches": 0,
        "attention_compose_output_mismatches": 0,
        "attention_residual_output_mismatches": 0,
        "post_attention_rmsnorm_output_mismatches": 0,
        "silu_output_mismatches": 0,
        "final_residual_output_mismatches": 0,
        "kv_cache_byte_mismatches": 0,
        "final_hidden_state_byte_mismatches": 0,
    }
    require(all(value == 0 for value in exact.values()), f"layer mismatch summary changed: {exact}")
    return {
        "schema_version": 1,
        "status": "PASS_EXACT_TWO_POSITION_LAYER_RTL",
        "layer_id": derived["layer_id"],
        "projection_results": projection_check["results"],
        "integer_mismatches": exact,
        "cache_reuse": derived["cache_reuse"],
        "simulator_cycles": rtl["simulator_cycles"],
        "final_hidden_state_sha256_by_position": rtl["final_layer_output_sha256_by_position"],
    }


def preflight_authorized() -> None:
    bindings = validate_bindings(require_attempt_absent=True)
    with torch.no_grad(), safe_open(canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        derived, _ = derive_layer(0, initial_states(weights), weights, adapter)
        regression = verify_layer0_regression(derived)
    print(
        json.dumps(
            {
                "status": "PASS_HASH_BOUND_EXECUTION_PREFLIGHT",
                "accepted_hashes": {name: record["sha256"] for name, record in bindings["accepted_hashes"].items()},
                "privacy_preflight": bindings["privacy_preflight_readback"]["status"],
                "layer0_regression": regression,
                "official_attempt_absent": not ATTEMPT.exists(),
            },
            sort_keys=True,
        )
    )


def seal_failure(phase: str, exc: BaseException) -> None:
    if not ATTEMPT.exists():
        ATTEMPT.mkdir(parents=True)
    failure = {
        "schema_version": 1,
        "status": f"SEALED_FAIL_{phase.upper()}",
        "phase": phase,
        "failure_taxonomy": "EXECUTION_GATE_OR_RTL_OFFICIAL_ATTEMPT",
        "root_cause_hypothesis": str(exc),
        "regression_required": "allocate a separately recorded repair attempt after an evidence-backed fix; never replay attempt-0001",
        "traceback": traceback.format_exc(),
        "official_attempt_consumed": True,
    }
    write_json(ATTEMPT / "failure.json", failure)
    write_json(ATTEMPT / "status.json", {"status": failure["status"], "sealed": True})
    write_sha256s(ATTEMPT)


def prepare() -> None:
    require(not ATTEMPT.exists(), f"official attempt already consumed: {ATTEMPT}")
    ATTEMPT.mkdir(parents=True)
    write_json(
        ATTEMPT / "prepare.started.json",
        {"phase": "prepare", "official_attempt_consumed": True, "monotonic_start": time.monotonic()},
    )
    try:
        bindings = validate_bindings(require_attempt_absent=False)
        authorization = {
            "schema_version": 1,
            "kind": "manager_authorized_hash_bound_execution_gate_transition",
            "mission_id": "execute-two-token-24layer-rtl-stack",
            "authorized_scope": "exactly-once prepare/run/check for attempt-0001 only",
            "accepted_runner_modified": False,
            "accepted_gate_status_observed": "LOCKED_PENDING_PREVIEW_PREPARE_RUN_CHECK_CLOSURE",
            "operator_authorization": "CLAIM 24-LAYER EXECUTION MISSION",
            "accepted_artifacts": bindings["accepted_hashes"],
            "authorized_wrapper": file_record(Path(__file__)),
            "privacy_preflight_readback": bindings["privacy_preflight_readback"],
            "commands": AUTHORIZED_COMMANDS,
            "accepted_command_semantics": read_json(CONTRACT)["commands"],
            "attempt_policy": {
                "attempt": relative(ATTEMPT),
                "prepare_run_check_exactly_once": True,
                "prepare_or_run_failure_consumes_attempt": True,
                "retry_unchanged": False,
            },
        }
        write_json(ATTEMPT / "authorization.json", authorization)
        freeze = {
            "schema_version": 1,
            "status": "PREPARED",
            "official_attempt": relative(ATTEMPT),
            "layers": list(range(LAYERS)),
            "positions": [0, 1],
            "projection_results_exact": EXPECTED_PROJECTION_RESULTS,
            "expected_layer_cycles": EXPECTED_LAYER_CYCLES,
            "expected_total_simulator_cycles": EXPECTED_TOTAL_CYCLES,
            "bindings": bindings,
            "wrapper": file_record(Path(__file__)),
            "commands": AUTHORIZED_COMMANDS,
        }
        write_json(ATTEMPT / "freeze.json", freeze)
        write_json(ATTEMPT / "status.json", {"status": "PREPARED", "sealed": False})
        print(
            json.dumps(
                {
                    "status": "PREPARED_HASH_BOUND_OFFICIAL_ATTEMPT",
                    "attempt": relative(ATTEMPT),
                    "wrapper_sha256": freeze["wrapper"]["sha256"],
                },
                sort_keys=True,
            )
        )
    except BaseException as exc:
        seal_failure("prepare", exc)
        raise


def run() -> None:
    require(ATTEMPT.is_dir(), "official attempt was not prepared")
    status = read_json(ATTEMPT / "status.json")
    require(status["status"] == "PREPARED" and status["sealed"] is False, "official attempt is not runnable")
    require(not (ATTEMPT / "run.started.json").exists(), "official run already consumed")
    write_json(ATTEMPT / "run.started.json", {"phase": "run", "monotonic_start": time.monotonic()})
    started = time.monotonic()
    layer_records: list[dict[str, Any]] = []
    aggregate_cycles = {name: 0 for name in EXPECTED_LAYER_CYCLES}
    projection_results = 0
    try:
        validate_bindings(require_attempt_absent=False)
        freeze = read_json(ATTEMPT / "freeze.json")
        require(freeze["wrapper"]["sha256"] == accepted.sha256_file(Path(__file__)), "authorized wrapper changed")
        with torch.no_grad(), safe_open(canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
            canonical.ADAPTER, framework="pt", device="cpu"
        ) as adapter:
            states = initial_states(weights)
            for layer_id in range(LAYERS):
                layer_started = time.monotonic()
                derived, states = derive_layer(layer_id, states, weights, adapter)
                layer_dir = ATTEMPT / "layers" / f"layer-{layer_id:02d}"
                layer_dir.mkdir(parents=True)
                if layer_id == 0:
                    write_json(layer_dir / "accepted_fresh_l2_regression.json", verify_layer0_regression(derived))
                tensors = persist_layer_tensors(layer_dir, derived)
                layer_runner.ACTIVE_ATTEMPT = layer_dir
                vectors = layer_runner.persist_vectors(derived)
                write_json(layer_dir / "conversion.json", layer_conversion(derived))
                rtl = layer_runner.run_rtl(derived)
                write_json(layer_dir / "rtl_execution.json", rtl)
                projection_check = verify_projection_results(layer_dir, derived)
                comparison = layer_comparison(derived, rtl, projection_check)
                write_json(layer_dir / "comparison.json", comparison)
                layer_elapsed = time.monotonic() - layer_started
                layer_manifest = {
                    "schema_version": 1,
                    "status": comparison["status"],
                    "layer_id": layer_id,
                    "tensors": tensors,
                    "vectors": vectors,
                    "conversion": file_record(layer_dir / "conversion.json"),
                    "rtl_execution": file_record(layer_dir / "rtl_execution.json"),
                    "comparison": file_record(layer_dir / "comparison.json"),
                    "cache_reuse": derived["cache_reuse"],
                    "projection_results": projection_check["results"],
                    "simulator_cycles": rtl["simulator_cycles"],
                    "elapsed_wall_seconds": layer_elapsed,
                    "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                }
                write_json(layer_dir / "manifest.json", layer_manifest)
                layer_records.append(
                    {
                        "layer_id": layer_id,
                        "manifest": file_record(layer_dir / "manifest.json"),
                        "projection_results": projection_check["results"],
                        "simulator_cycles": rtl["simulator_cycles"],
                        "cache_reuse": derived["cache_reuse"],
                        "final_hidden_state_sha256_by_position": rtl["final_layer_output_sha256_by_position"],
                        "elapsed_wall_seconds": layer_elapsed,
                    }
                )
                projection_results += projection_check["results"]
                for family, cycles in rtl["simulator_cycles"].items():
                    aggregate_cycles[family] += cycles
                print(
                    json.dumps(
                        {
                            "status": "PASS_LAYER_RTL",
                            "layer_id": layer_id,
                            "projection_results_cumulative": projection_results,
                            "simulator_cycles": rtl["simulator_cycles"],
                            "cache_reuse": derived["cache_reuse"],
                            "elapsed_wall_seconds": layer_elapsed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                del derived

        total_cycles = sum(aggregate_cycles.values())
        require(projection_results == EXPECTED_PROJECTION_RESULTS, "aggregate projection count changed")
        require(total_cycles == EXPECTED_TOTAL_CYCLES, "aggregate simulator cycle count changed")
        require(len(layer_records) == LAYERS, "executed layer count changed")
        require(
            all(record["cache_reuse"]["cached_position0_probability_nonzero_heads"] > 0 for record in layer_records),
            "position-1 cached-K dependence missing in at least one layer",
        )
        require(
            all(record["cache_reuse"]["compose_output_changes_when_cached_v_zeroed_heads"] > 0 for record in layer_records),
            "position-1 cached-V dependence missing in at least one layer",
        )
        elapsed = time.monotonic() - started
        run_summary = {
            "schema_version": 1,
            "status": "PASS_24LAYER_RTL_RUN_UNCHECKED",
            "layers_executed": LAYERS,
            "projection_results": projection_results,
            "aggregate_family_cycles": aggregate_cycles,
            "total_simulator_cycles": total_cycles,
            "simulator_cycle_classification": "family_local_simulator_cycles_not_accelerator_latency",
            "elapsed_wall_seconds": elapsed,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "layers": layer_records,
            "final_layer23_hidden_state_sha256_by_position": layer_records[-1]["final_hidden_state_sha256_by_position"],
        }
        write_json(ATTEMPT / "run_summary.json", run_summary)
        write_json(ATTEMPT / "status.json", {"status": run_summary["status"], "sealed": False})
        print(json.dumps(run_summary, sort_keys=True), flush=True)
    except BaseException as exc:
        seal_failure("run", exc)
        raise


def check() -> None:
    require(ATTEMPT.is_dir(), "official attempt is absent")
    status = read_json(ATTEMPT / "status.json")
    require(status["status"] == "PASS_24LAYER_RTL_RUN_UNCHECKED" and status["sealed"] is False, "official run is not checkable")
    require(not (ATTEMPT / "check.started.json").exists(), "official check already consumed")
    write_json(ATTEMPT / "check.started.json", {"phase": "check", "monotonic_start": time.monotonic()})
    try:
        bindings = validate_bindings(require_attempt_absent=False)
        freeze = read_json(ATTEMPT / "freeze.json")
        run_summary = read_json(ATTEMPT / "run_summary.json")
        require(freeze["wrapper"]["sha256"] == accepted.sha256_file(Path(__file__)), "authorized wrapper changed")
        require(run_summary["layers_executed"] == LAYERS, "layer count changed")
        require(run_summary["projection_results"] == EXPECTED_PROJECTION_RESULTS, "projection count changed")
        require(run_summary["total_simulator_cycles"] == EXPECTED_TOTAL_CYCLES, "total cycle count changed")
        aggregate_mismatches: dict[str, int] = {}
        cache_layers = 0
        final_hashes: list[str] | None = None
        for layer_id in range(LAYERS):
            layer_dir = ATTEMPT / "layers" / f"layer-{layer_id:02d}"
            comparison = read_json(layer_dir / "comparison.json")
            rtl = read_json(layer_dir / "rtl_execution.json")
            manifest = read_json(layer_dir / "manifest.json")
            require(comparison["status"] == manifest["status"] == "PASS_EXACT_TWO_POSITION_LAYER_RTL", f"layer {layer_id} status changed")
            require(comparison["projection_results"] == EXPECTED_PROJECTION_RESULTS_PER_LAYER, f"layer {layer_id} projection count changed")
            require(rtl["simulator_cycles"] == EXPECTED_LAYER_CYCLES, f"layer {layer_id} cycles changed")
            for name, value in comparison["integer_mismatches"].items():
                aggregate_mismatches[name] = aggregate_mismatches.get(name, 0) + int(value)
            cache = comparison["cache_reuse"]
            require(cache["cached_position0_probability_nonzero_heads"] > 0, f"layer {layer_id} cached-K dependence absent")
            require(cache["compose_output_changes_when_cached_v_zeroed_heads"] > 0, f"layer {layer_id} cached-V dependence absent")
            cache_layers += 1
            if layer_id == LAYERS - 1:
                final_hashes = comparison["final_hidden_state_sha256_by_position"]
        require(all(value == 0 for value in aggregate_mismatches.values()), f"aggregate integer mismatch: {aggregate_mismatches}")
        require(cache_layers == LAYERS, "cache dependence layer count changed")
        require(final_hashes is not None and len(final_hashes) == 2, "final layer-23 hidden states absent")
        check_result = {
            "schema_version": 1,
            "status": "PASS_24LAYER_EXACT_INTEGER_CHECK_AWAITING_INDEPENDENT_REVIEW",
            "layers": LAYERS,
            "projection_results": EXPECTED_PROJECTION_RESULTS,
            "integer_mismatches": aggregate_mismatches,
            "position1_cache_dependency_layers": cache_layers,
            "final_layer23_hidden_state_sha256_by_position": final_hashes,
            "aggregate_family_cycles": run_summary["aggregate_family_cycles"],
            "total_simulator_cycles": run_summary["total_simulator_cycles"],
            "elapsed_wall_seconds": run_summary["elapsed_wall_seconds"],
            "peak_rss_kib": run_summary["peak_rss_kib"],
            "tool_versions": bindings["tool_versions"],
            "accepted_hashes": bindings["accepted_hashes"],
            "review_required": True,
        }
        write_json(ATTEMPT / "check.json", check_result)
        manifest = {
            "schema_version": 1,
            "status": check_result["status"],
            "claim_boundary": read_json(CONTRACT)["claim_boundary"],
            "authorization": file_record(ATTEMPT / "authorization.json"),
            "freeze": file_record(ATTEMPT / "freeze.json"),
            "run_summary": file_record(ATTEMPT / "run_summary.json"),
            "check": file_record(ATTEMPT / "check.json"),
            "wrapper": file_record(Path(__file__)),
            "accepted_artifacts": bindings["accepted_hashes"],
            "official_attempt_consumed": True,
            "immutable_after_check": True,
            "independent_review_required": True,
        }
        write_json(ATTEMPT / "manifest.json", manifest)
        write_json(ATTEMPT / "status.json", {"status": check_result["status"], "sealed": True})
        write_sha256s(ATTEMPT)
        members = verify_sha256s(ATTEMPT)
        result = {
            **check_result,
            "attempt": relative(ATTEMPT),
            "members": members,
            "sha256s_sha256": accepted.sha256_file(ATTEMPT / "SHA256SUMS"),
        }
        print(json.dumps(result, sort_keys=True))
    except BaseException as exc:
        seal_failure("check", exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-authorized", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.preflight_authorized:
        preflight_authorized()
    elif args.prepare:
        prepare()
    elif args.run:
        run()
    else:
        check()


if __name__ == "__main__":
    main()
