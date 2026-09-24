#!/usr/bin/env python3
"""Run checkpoint-176 final RMSNorm plus tied LM-head through ACE-2 RTL."""

from __future__ import annotations

import argparse
import hashlib
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
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_lora_v4_layer0_full_rtl as canonical
from ace2_projection_reference import ProjectionCase, reference_projection
from ace2_rmsnorm_reference import (
    derive_rmsnorm_output_scale,
    derive_scaled_gains_q8,
    pack_gain_beats,
    pack_int8_beats,
    reference_rmsnorm,
)


MISSION = "canonical_checkpoint176_lm_head_top_token_rtl"
OUT = ROOT / "evidence/verification/lora-v4-lm-head-top-token-rtl-v1"
ATTEMPT = OUT / "attempt-0001"
PREFLIGHT = ROOT / "build/lora-v4-lm-head-top-token-rtl-preview/preflight-0001"

UPSTREAM = ROOT / "evidence/verification/lora-v4-two-token-24layer-rtl-v1/attempt-0003"
UPSTREAM_SUMS_SHA256 = "0d3cc8064f947183c83ae24add4631cade31b164a7331a1507b86d27d5a29c85"
LAYER23 = UPSTREAM / "layers/layer-23/tensors"
POSITION_INPUTS = {
    0: {
        "q": LAYER23 / "position0_layer_output_s8.bin",
        "scale": LAYER23 / "position0_layer_output_scale_f64le.bin",
        "q_sha256": "77184fd7c719f19d80893085b48eafa08202bcb814c8df244a8ad0c4502ba0fd",
        "scale_sha256": "530cfb7aa47e7cbdc7027106b943cc45ff937ef704f6f33e9442e52a23c89a33",
    },
    1: {
        "q": LAYER23 / "position1_layer_output_s8.bin",
        "scale": LAYER23 / "position1_layer_output_scale_f64le.bin",
        "q_sha256": "8be5035825c44c7e0ee7379d57add671ad8d3da70948cd2d3382337893274225",
        "scale_sha256": "f67bbde827ac25725d59f5ae2d65c8419477dc1b81eb774444b9bb0fc8aad0b4",
    },
}

SNAPSHOT = canonical.SNAPSHOT
MODEL = canonical.MODEL
CONFIG = canonical.CONFIG
ADAPTER = canonical.ADAPTER
ADAPTER_CONFIG = canonical.ADAPTER_CONFIG
CHECKPOINT_IDENTITY = canonical.CHECKPOINT_IDENTITY
MODEL_SHA256 = canonical.MODEL_SHA256
ADAPTER_SHA256 = canonical.ADAPTER_SHA256
CHECKPOINT_TREE_SHA256 = canonical.CHECKPOINT_TREE_SHA256

HIDDEN = 896
VOCAB = 151936
TILE_OUTPUTS = 32
GROUPS = HIDDEN // 4
RMS_EPSILON = canonical.RMS_EPSILON
RMS_TB = ROOT / "verification/tb/ace2_rmsnorm_tb.sv"
RMS_CORE = ROOT / "rtl/ace2_rmsnorm_core.sv"
PROJ_CORE = ROOT / "rtl/ace2_w4a8_proj_core.sv"


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


def tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().contiguous().cpu()
    return tensor.numpy().tobytes(order="C")


def raw_tensor_sha256(value: torch.Tensor) -> str:
    return sha256_bytes(canonical.raw_tensor_bytes(value))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def public_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def file_record(path: Path) -> dict[str, Any]:
    return {"path": public_path(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


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


def write_sha256s(base: Path) -> None:
    rows = []
    for path in sorted(item for item in base.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        rows.append(f"{sha256_file(path)}  {path.relative_to(base).as_posix()}\n")
    (base / "SHA256SUMS").write_text("".join(rows), encoding="ascii")


def verify_sha256s(base: Path) -> int:
    count = 0
    for row in (base / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        expected, name = row.split("  ", 1)
        candidate = base / name
        require(candidate.is_file(), f"manifest member missing: {name}")
        require(sha256_file(candidate) == expected, f"manifest member changed: {name}")
        count += 1
    return count


def tool_line(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return None
    text = (completed.stdout or completed.stderr).splitlines()
    return text[0] if text else f"returncode={completed.returncode}"


def tool_record(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    return {
        "path": path,
        "sha256": sha256_file(Path(path)) if path else None,
    }


def tool_versions() -> dict[str, Any]:
    import safetensors

    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "safetensors": safetensors.__version__,
        "iverilog": tool_line(["iverilog", "-V"]),
        "vvp": tool_line(["vvp", "-V"]),
        "verilator": tool_line(["verilator", "--version"]),
        "binaries": {name: tool_record(name) for name in ("python3", "iverilog", "vvp", "verilator")},
    }


def rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def verify_upstream() -> dict[str, Any]:
    required = [UPSTREAM / "check.json", UPSTREAM / "run_summary.json", UPSTREAM / "SHA256SUMS"]
    for path in required:
        require(path.is_file(), f"required upstream artifact missing: {path}")
    require(sha256_file(UPSTREAM / "SHA256SUMS") == UPSTREAM_SUMS_SHA256, "upstream SHA256SUMS changed")
    check = json.loads((UPSTREAM / "check.json").read_text(encoding="utf-8"))
    run = json.loads((UPSTREAM / "run_summary.json").read_text(encoding="utf-8"))
    require(check.get("status") == "PASS_24LAYER_EXACT_INTEGER_CYCLE_REPAIR_CHECK_AWAITING_INDEPENDENT_REVIEW", "upstream check is not accepted attempt-0003 status")
    require(run.get("status") == "PASS_24LAYER_RTL_RUN_UNCHECKED", "upstream run status changed")
    inputs: dict[str, Any] = {}
    for position, binding in POSITION_INPUTS.items():
        q_path = binding["q"]
        scale_path = binding["scale"]
        require(q_path.is_file() and scale_path.is_file(), f"upstream layer-23 position {position} tensor missing")
        require(q_path.stat().st_size == HIDDEN, f"upstream layer-23 position {position} hidden width changed")
        require(scale_path.stat().st_size == 8, f"upstream layer-23 position {position} scale width changed")
        require(sha256_file(q_path) == binding["q_sha256"], f"upstream position {position} hidden hash changed")
        require(sha256_file(scale_path) == binding["scale_sha256"], f"upstream position {position} scale hash changed")
        inputs[str(position)] = {"hidden": file_record(q_path), "scale": file_record(scale_path)}
    return {
        "attempt": public_path(UPSTREAM),
        "sha256s": file_record(UPSTREAM / "SHA256SUMS"),
        "check_status": check["status"],
        "run_status": run["status"],
        "position_inputs": inputs,
    }


def validate_model_bindings() -> dict[str, Any]:
    for path in (MODEL, CONFIG, ADAPTER, ADAPTER_CONFIG, CHECKPOINT_IDENTITY, RMS_TB, RMS_CORE, PROJ_CORE):
        require(path.is_file(), f"required local artifact missing: {path}")
    require(sha256_file(MODEL) == MODEL_SHA256, "base model hash changed")
    require(sha256_file(ADAPTER) == ADAPTER_SHA256, "adapter hash changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    identity = json.loads(CHECKPOINT_IDENTITY.read_text(encoding="utf-8"))
    adapter_config = json.loads(ADAPTER_CONFIG.read_text(encoding="utf-8"))
    require(identity["tree_sha256"] == CHECKPOINT_TREE_SHA256, "checkpoint tree changed")
    require(config["model_type"] == "qwen2", "unexpected model type")
    require(config["hidden_size"] == HIDDEN, "hidden width changed")
    require(config["vocab_size"] == VOCAB, "vocab size changed")
    require(config["tie_word_embeddings"] is True, "LM-head is no longer tied to embeddings")
    require(float(config["rms_norm_eps"]) == RMS_EPSILON, "RMS epsilon changed")

    with safe_open(MODEL, framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        require("model.embed_tokens.weight" in keys, "tied embedding tensor missing")
        require("model.norm.weight" in keys, "final RMSNorm gain tensor missing")
        require("lm_head.weight" not in keys, "unexpected separate LM-head tensor present")
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        require(list(embedding.shape) == [VOCAB, HIDDEN], "embedding/LM-head shape changed")
        require(list(norm_gain.shape) == [HIDDEN], "final RMSNorm gain shape changed")

    return {
        "config": file_record(CONFIG),
        "model": file_record(MODEL),
        "adapter": file_record(ADAPTER),
        "adapter_config": file_record(ADAPTER_CONFIG),
        "checkpoint_identity": file_record(CHECKPOINT_IDENTITY),
        "checkpoint_tree_sha256": identity["tree_sha256"],
        "adapter_lora": {
            "r": adapter_config.get("r"),
            "lora_alpha": adapter_config.get("lora_alpha"),
        },
        "lm_head_source": {
            "kind": "tied_embedding",
            "tensor": "model.embed_tokens.weight",
            "shape": [VOCAB, HIDDEN],
            "dtype": str(embedding.dtype),
            "raw_tensor_sha256": raw_tensor_sha256(embedding),
        },
        "final_rmsnorm_source": {
            "tensor": "model.norm.weight",
            "shape": [HIDDEN],
            "dtype": str(norm_gain.dtype),
            "raw_tensor_sha256": raw_tensor_sha256(norm_gain),
        },
    }


def frozen_contract(output_count: int) -> dict[str, Any]:
    require(output_count % TILE_OUTPUTS == 0, "output_count must align to LM-head tile size")
    return {
        "mission": MISSION,
        "version": 1,
        "scope": "final RMSNorm plus tied W4A8 LM-head top-token RTL check from accepted layer-23 hidden states",
        "non_goals": [
            "arbitrary-text chat completion",
            "Stage-1 productization closure",
            "U280 deployment",
            "PPA replay",
        ],
        "upstream": verify_upstream(),
        "model_bindings": validate_model_bindings(),
        "interface": {
            "positions": [0, 1],
            "hidden": HIDDEN,
            "vocab": VOCAB,
            "official_output_count": output_count,
            "tile_outputs": TILE_OUTPUTS,
            "k": HIDDEN,
            "w4_groups": GROUPS,
            "lm_head_layer_id": 24,
            "shell_descriptor": "opcode W4A8 projection, m=1, n=32, k=896, layer_id=24",
        },
        "quantization": {
            "final_rmsnorm": "ACE-2 static per-tensor int8 RMSNorm with Q7.8 gains and RNE saturation",
            "lm_head_weight": "signed W4 symmetric per output token row absmax/7, torch.round ties-to-even, clamp [-8, 7]",
            "lm_head_activation": "signed int8 final RMSNorm output",
            "lm_head_accumulator": "signed int32 dot product",
            "lm_head_output": "signed int8 projection result with zero point 0, bias 0, RNE right shift, clamp [-128,127]",
            "top_token_tiebreak": "lowest token id wins equal signed-int8 top logit",
        },
    }


def read_layer23_position(position: int) -> tuple[torch.Tensor, float]:
    binding = POSITION_INPUTS[position]
    raw = binding["q"].read_bytes()
    values = [byte if byte < 128 else byte - 256 for byte in raw]
    hidden_q = torch.tensor(values, dtype=torch.int8)
    (scale,) = struct.unpack("<d", binding["scale"].read_bytes())
    require(hidden_q.numel() == HIDDEN, "layer-23 hidden tensor width mismatch")
    require(math.isfinite(scale) and scale > 0.0, "layer-23 hidden scale invalid")
    return hidden_q, float(scale)


def hex_literal(value: int, width: int) -> str:
    return f"{width}'h{int(value) & ((1 << width) - 1):0{width // 4}x}"


def render_rmsnorm_vectors(positions: dict[int, dict[str, Any]]) -> str:
    lines = [
        "// Generated by tools/run_lora_v4_lm_head_top_token_rtl.py.",
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
    for case, position in enumerate((0, 1)):
        item = positions[position]
        result = item["rmsnorm_result"]
        lines += [
            f"  test_expected_sumsq[{case}] = {hex_literal(result.sumsq, 48)};",
            f"  test_expected_inv[{case}] = {hex_literal(result.inv_rms_q30, 32)};",
            f"  test_expected_saturation[{case}] = 1'b{int(result.saturation_seen)};",
        ]
        for beat, (input_word, gain_word, output_word) in enumerate(
            zip(
                pack_int8_beats(item["input_q"].to(torch.int64).tolist()),
                pack_gain_beats(item["gains_q8"]),
                pack_int8_beats(result.outputs),
                strict=True,
            )
        ):
            flat = case * 56 + beat
            lines += [
                f"  test_input_beats[{flat}] = {hex_literal(input_word, 128)};",
                f"  test_gain_beats[{flat}] = {hex_literal(gain_word, 256)};",
                f"  test_expected_beats[{flat}] = {hex_literal(output_word, 128)};",
            ]
    lines += ["end", ""]
    return "\n".join(lines)


def pack_i8_groups(values: torch.Tensor) -> torch.Tensor:
    require(values.numel() == HIDDEN, "activation packing width mismatch")
    lanes = values.to(torch.int64).view(GROUPS, 4) & 0xFF
    return (lanes[:, 0] | (lanes[:, 1] << 8) | (lanes[:, 2] << 16) | (lanes[:, 3] << 24)).to(torch.int64)


def packed_w4_groups(rows: torch.Tensor) -> torch.Tensor:
    lanes = rows.to(torch.int64).view(rows.shape[0], GROUPS, 4) & 0xF
    packed = lanes[:, :, 0] | (lanes[:, :, 1] << 4) | (lanes[:, :, 2] << 8) | (lanes[:, :, 3] << 12)
    return packed.reshape(-1).to(torch.int64)


def write_hex_tensor(path: Path, values: torch.Tensor, width: int, chunk: int = 65536) -> dict[str, Any]:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    lookup = None
    if width in (8, 16):
        lookup = [f"{value:0{digits}x}\n" for value in range(1 << width)]
    flat = values.detach().reshape(-1).cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        for start in range(0, flat.numel(), chunk):
            items = flat[start : start + chunk].to(torch.int64).tolist()
            if lookup is None:
                handle.write("".join(f"{int(value) & mask:0{digits}x}\n" for value in items))
            else:
                handle.write("".join(lookup[int(value) & mask] for value in items))
    return file_record(path)


def derive_final_rmsnorm(working: Path, norm_gain: torch.Tensor) -> dict[int, dict[str, Any]]:
    positions: dict[int, dict[str, Any]] = {}
    norm_gain_f64 = norm_gain.to(torch.float64)
    for position in (0, 1):
        input_q, input_scale = read_layer23_position(position)
        float_input = input_q.to(torch.float64) * input_scale
        float_norm = canonical.float_rmsnorm(float_input.to(torch.float32), norm_gain).contiguous()
        output_scale = derive_rmsnorm_output_scale(
            norm_gain_f64.tolist(),
            float(float_norm.to(torch.float64).abs().max().item()),
        )
        gains_q8 = derive_scaled_gains_q8(norm_gain_f64.tolist(), output_scale)
        result = reference_rmsnorm(input_q.to(torch.int64).tolist(), gains_q8)
        output_q = torch.tensor(result.outputs, dtype=torch.int8)
        vectors = working / "vectors" / f"position{position}"
        tensors = working / "tensors"
        artifacts = {
            "input_q": file_record(POSITION_INPUTS[position]["q"]),
            "input_scale": file_record(POSITION_INPUTS[position]["scale"]),
            "final_rmsnorm_s8": write_binary(tensors / f"position{position}_final_rmsnorm_s8.bin", tensor_bytes(output_q)),
            "final_rmsnorm_scale_f64le": write_binary(
                tensors / f"position{position}_final_rmsnorm_scale_f64le.bin",
                struct.pack("<d", float(output_scale)),
            ),
            "final_rmsnorm_gains_s16le": write_binary(
                tensors / f"position{position}_final_rmsnorm_gains_s16le.bin",
                torch.tensor(gains_q8, dtype=torch.int16).numpy().tobytes(order="C"),
            ),
            "lm_head_activation_hex": write_hex_tensor(vectors / "lm_head_activation.hex", pack_i8_groups(output_q), 32),
        }
        positions[position] = {
            "input_q": input_q,
            "input_scale": input_scale,
            "float_norm": float_norm,
            "final_q": output_q,
            "final_scale": float(output_scale),
            "gains_q8": gains_q8,
            "rmsnorm_result": result,
            "artifacts": artifacts,
            "summary": {
                "position": position,
                "input_scale": input_scale,
                "final_rmsnorm_scale": float(output_scale),
                "sumsq": int(result.sumsq),
                "inv_rms_q30": int(result.inv_rms_q30),
                "saturation_seen": bool(result.saturation_seen),
                "final_output_sha256": sha256_bytes(tensor_bytes(output_q)),
            },
        }
    write_text(working / "vectors/rmsnorm_vectors.svh", render_rmsnorm_vectors(positions))
    return positions


def derive_lm_head_weights(working: Path, embedding: torch.Tensor, output_count: int) -> dict[str, Any]:
    qweight = torch.empty((output_count, HIDDEN), dtype=torch.int8)
    weight_scale = torch.empty((output_count,), dtype=torch.float64)
    for start in range(0, output_count, 2048):
        stop = min(output_count, start + 2048)
        chunk = embedding[start:stop].to(torch.float64)
        scale = chunk.abs().amax(dim=1) / 7.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        qweight[start:stop] = torch.round(chunk / scale[:, None]).clamp(-8, 7).to(torch.int8)
        weight_scale[start:stop] = scale
    weight_path = working / "vectors/lm_head_weight.hex"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    with weight_path.open("w", encoding="ascii") as handle:
        lookup = [f"{value:04x}\n" for value in range(1 << 16)]
        for start in range(0, output_count, 512):
            stop = min(output_count, start + 512)
            packed = packed_w4_groups(qweight[start:stop]).tolist()
            handle.write("".join(lookup[int(value) & 0xFFFF] for value in packed))
    return {
        "qweight": qweight,
        "weight_scale": weight_scale,
        "artifacts": {
            "lm_head_weight_hex": file_record(weight_path),
            "lm_head_qweight_raw_sha256": sha256_bytes(tensor_bytes(qweight)),
        },
    }


def round_outputs(accumulator: torch.Tensor, multiplier: torch.Tensor, right_shift: torch.Tensor) -> torch.Tensor:
    values = [
        canonical.round_shift_even(int(acc) * int(mult), int(shift))
        for acc, mult, shift in zip(accumulator.tolist(), multiplier.tolist(), right_shift.tolist(), strict=True)
    ]
    return torch.tensor(values, dtype=torch.int64)


def top_token(values: torch.Tensor) -> tuple[int, int]:
    best_token = 0
    best_value = -129
    for token, value in enumerate(values.to(torch.int64).tolist()):
        if value > best_value:
            best_token = token
            best_value = int(value)
    return best_token, best_value


def derive_lm_head_position(
    working: Path,
    position: int,
    item: dict[str, Any],
    embedding: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_count: int,
) -> dict[str, Any]:
    float_outputs = []
    final_float = item["float_norm"].to(torch.float32)
    for start in range(0, output_count, 4096):
        stop = min(output_count, start + 4096)
        float_outputs.append(torch.mv(embedding[start:stop].to(torch.float32), final_float))
    float_output = torch.cat(float_outputs).contiguous()
    output_scale = canonical.scale_for(float_output)
    multiplier, right_shift = canonical.derive_multiplier(item["final_scale"] * weight_scale / output_scale)

    accumulator = torch.empty((output_count,), dtype=torch.int64)
    act = item["final_q"].to(torch.int32)
    for start in range(0, output_count, 4096):
        stop = min(output_count, start + 4096)
        accumulator[start:stop] = (qweight[start:stop].to(torch.int32) * act).sum(dim=1, dtype=torch.int64)

    require(bool(torch.all(accumulator >= -(1 << 31))) and bool(torch.all(accumulator < (1 << 31))), "LM-head accumulator overflow")
    rounded = round_outputs(accumulator, multiplier, right_shift)
    output_q = rounded.clamp(-128, 127).to(torch.int8)
    saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
    sample_indices = sorted(
        set(
            index
            for index in (0, 1, 31, 32, 63, output_count // 2, output_count - 2, output_count - 1)
            if 0 <= index < output_count
        )
    )
    for index in sample_indices:
        independent = reference_projection(
            ProjectionCase(
                name=f"lm_head_position{position}_token{index}",
                rows=1,
                reduction_size=HIDDEN,
                activations=[item["final_q"].to(torch.int64).tolist()],
                weights=[qweight[index].to(torch.int64).tolist()],
                multipliers=[int(multiplier[index])],
                right_shifts=[int(right_shift[index])],
                output_zero_points=[0],
                bias_accumulators=[0],
            )
        )
        require(
            independent.outputs == [[int(output_q[index])]],
            f"independent sampled LM-head projection reference differs for position {position} token {index}",
        )

    vectors = working / "vectors" / f"position{position}"
    artifacts = {
        "lm_head_multiplier_hex": write_hex_tensor(vectors / "lm_head_multiplier.hex", multiplier, 32),
        "lm_head_shift_hex": write_hex_tensor(vectors / "lm_head_shift.hex", right_shift, 8),
        "lm_head_expected_hex": write_hex_tensor(vectors / "lm_head_expected.hex", output_q.to(torch.int64), 8),
        "lm_head_accumulator_hex": write_hex_tensor(vectors / "lm_head_accumulator.hex", accumulator, 32),
        "lm_head_saturation_hex": write_hex_tensor(vectors / "lm_head_saturation.hex", saturation.to(torch.int64), 8),
        "lm_head_output_s8": write_binary(working / "tensors" / f"position{position}_lm_head_output_s8.bin", tensor_bytes(output_q)),
        "lm_head_output_scale_f64le": write_binary(
            working / "tensors" / f"position{position}_lm_head_output_scale_f64le.bin",
            struct.pack("<d", float(output_scale)),
        ),
    }
    token, logit = top_token(output_q)
    return {
        "position": position,
        "output_count": output_count,
        "lm_head_output_scale": float(output_scale),
        "top_token": int(token),
        "top_logit_s8": int(logit),
        "saturation_count": int(saturation.sum().item()),
        "output_sha256": sha256_bytes(tensor_bytes(output_q)),
        "accumulator_min": int(accumulator.min().item()),
        "accumulator_max": int(accumulator.max().item()),
        "independent_reference_sampled_tokens": sample_indices,
        "artifacts": artifacts,
    }


def render_lm_head_tb(working: Path, position: int, output_count: int) -> str:
    vector = working / "vectors" / f"position{position}"
    return f"""`timescale 1ns/1ps
`default_nettype none

module ace2_lm_head_position{position}_tb;
    localparam integer POSITION = {position};
    localparam integer TOKEN_OFFSET = 0;
    localparam integer TOTAL_OUTPUTS = {output_count};
    localparam integer GROUPS = {GROUPS};
    localparam integer MAC_LANES = 4;
    localparam integer GROUP_INDEX_WIDTH = $clog2(GROUPS + 1);

    reg clk;
    reg rst_n;
    reg start_valid;
    wire start_ready;
    reg pair_valid;
    wire pair_ready;
    reg [31:0] act_data;
    reg [15:0] weight_data;
    reg [GROUP_INDEX_WIDTH-1:0] last_group;
    reg meta_valid;
    wire meta_ready;
    reg signed [31:0] multiplier;
    reg [5:0] right_shift;
    wire out_valid;
    reg out_ready;
    wire [7:0] out_data;
    wire signed [31:0] acc;
    wire accumulator_overflow;
    wire saturation_seen;

    reg [31:0] activation_mem [0:GROUPS-1];
    reg [15:0] weight_mem [0:TOTAL_OUTPUTS*GROUPS-1];
    reg [31:0] multiplier_mem [0:TOTAL_OUTPUTS-1];
    reg [7:0] shift_mem [0:TOTAL_OUTPUTS-1];
    reg [7:0] expected_mem [0:TOTAL_OUTPUTS-1];
    reg [31:0] expected_acc_mem [0:TOTAL_OUTPUTS-1];
    reg [7:0] expected_saturation_mem [0:TOTAL_OUTPUTS-1];

    integer channel;
    integer group_index;
    integer guard;
    integer failures;
    integer cycles;
    integer top_token;
    integer top_logit_s8;
    reg top_valid;
    reg signed [7:0] out_signed;

    ace2_w4a8_proj_core #(
        .K_SIZE({HIDDEN}),
        .MAC_LANES(MAC_LANES)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(1'b0),
        .start_valid_i(start_valid),
        .start_ready_o(start_ready),
        .last_group_i(last_group),
        .pair_valid_i(pair_valid),
        .pair_ready_o(pair_ready),
        .act_data_i(act_data),
        .weight_data_i(weight_data),
        .meta_valid_i(meta_valid),
        .meta_ready_o(meta_ready),
        .multiplier_i(multiplier),
        .right_shift_i(right_shift),
        .output_zero_point_i(8'sd0),
        .bias_accumulator_i(32'sd0),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_data_o(out_data),
        .acc_o(acc),
        .accumulator_overflow_o(accumulator_overflow),
        .saturation_seen_o(saturation_seen)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk) begin
        if (rst_n)
            cycles = cycles + 1;
    end

    task apply_reset;
        begin
            rst_n = 1'b0;
            start_valid = 1'b0;
            pair_valid = 1'b0;
            act_data = 32'd0;
            weight_data = 16'd0;
            last_group = 0;
            meta_valid = 1'b0;
            multiplier = 32'sd0;
            right_shift = 6'd0;
            out_ready = 1'b0;
            cycles = 0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task run_channel;
        input integer selected_channel;
        begin
            while (!start_ready) @(posedge clk);
            last_group = GROUPS - 1;
            @(negedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;

            for (group_index = 0; group_index < GROUPS; group_index = group_index + 1) begin
                while (!pair_ready) @(posedge clk);
                act_data = activation_mem[group_index];
                weight_data = weight_mem[selected_channel*GROUPS + group_index];
                @(negedge clk);
                pair_valid = 1'b1;
                @(posedge clk);
                @(negedge clk);
                pair_valid = 1'b0;
            end

            while (!meta_ready) @(posedge clk);
            multiplier = $signed(multiplier_mem[selected_channel]);
            right_shift = shift_mem[selected_channel][5:0];
            @(negedge clk);
            meta_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            meta_valid = 1'b0;

            guard = 0;
            while (!out_valid && guard < 8192) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid) begin
                $display("RTL_LM_HEAD_TIMEOUT position=%0d token=%0d", POSITION, TOKEN_OFFSET + selected_channel);
                failures = failures + 1;
            end else begin
                out_signed = $signed(out_data);
                if ((out_data !== expected_mem[selected_channel]) ||
                    (acc !== $signed(expected_acc_mem[selected_channel])) ||
                    (saturation_seen !== expected_saturation_mem[selected_channel][0]) ||
                    accumulator_overflow) begin
                    $display("RTL_LM_HEAD_MISMATCH position=%0d token=%0d got=%0d expected=%0d got_acc=%0d expected_acc=%0d got_sat=%0d expected_sat=%0d overflow=%0d",
                             POSITION, TOKEN_OFFSET + selected_channel, out_signed,
                             $signed(expected_mem[selected_channel]), acc,
                             $signed(expected_acc_mem[selected_channel]), saturation_seen,
                             expected_saturation_mem[selected_channel][0], accumulator_overflow);
                    failures = failures + 1;
                end
                if (!top_valid || (out_signed > top_logit_s8) ||
                    ((out_signed == top_logit_s8) && ((TOKEN_OFFSET + selected_channel) < top_token))) begin
                    top_valid = 1'b1;
                    top_logit_s8 = out_signed;
                    top_token = TOKEN_OFFSET + selected_channel;
                end
                out_ready = 1'b1;
                @(posedge clk);
                @(negedge clk);
                out_ready = 1'b0;
            end
        end
    endtask

    initial begin
        $readmemh("{relative(vector / "lm_head_activation.hex")}", activation_mem);
        $readmemh("{relative(working / "vectors/lm_head_weight.hex")}", weight_mem);
        $readmemh("{relative(vector / "lm_head_multiplier.hex")}", multiplier_mem);
        $readmemh("{relative(vector / "lm_head_shift.hex")}", shift_mem);
        $readmemh("{relative(vector / "lm_head_expected.hex")}", expected_mem);
        $readmemh("{relative(vector / "lm_head_accumulator.hex")}", expected_acc_mem);
        $readmemh("{relative(vector / "lm_head_saturation.hex")}", expected_saturation_mem);
        failures = 0;
        top_token = 0;
        top_logit_s8 = -129;
        top_valid = 1'b0;
        apply_reset();
        for (channel = 0; channel < TOTAL_OUTPUTS; channel = channel + 1)
            run_channel(channel);
        if (failures != 0)
            $fatal(1, "ACE2_LM_HEAD_PROJ_FAIL position=%0d failures=%0d", POSITION, failures);
        $display("ACE2_LM_HEAD_PROJ_PASS position=%0d outputs=%0d token_offset=%0d top_token=%0d top_logit_s8=%0d cycles=%0d",
                 POSITION, TOTAL_OUTPUTS, TOKEN_OFFSET, top_token, top_logit_s8, cycles);
        $finish;
    end
endmodule

`default_nettype wire
"""


def rmsnorm_execution_source(working: Path) -> Path:
    source = RMS_TB.read_text(encoding="utf-8")
    source = source.replace(
        '`include "../generated/rmsnorm_vectors.svh"',
        f'`include "{relative(working / "vectors/rmsnorm_vectors.svh")}"',
        1,
    )
    path = working / "execution_sources/rmsnorm_tb.sv"
    write_text(path, source)
    return path


def lm_head_execution_source(working: Path, position: int, output_count: int) -> Path:
    path = working / "execution_sources" / f"lm_head_position{position}_tb.sv"
    write_text(path, render_lm_head_tb(working, position, output_count))
    return path


def run_process(working: Path, name: str, command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    logs = working / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (logs / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed


def compile_and_run(working: Path, name: str, sources: list[Path], top: str) -> dict[str, Any]:
    binary = working / "sim" / f"{name}.vvp"
    binary.parent.mkdir(parents=True, exist_ok=True)
    compile_cmd = [
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
        *[relative(source) for source in sources],
    ]
    start = time.time()
    compiled = run_process(working, f"{name}.iverilog", compile_cmd)
    require(compiled.returncode == 0, f"{name} compile failed; see {working / 'logs' / (name + '.iverilog.stderr.log')}")
    sim_cmd = ["vvp", relative(binary)]
    replayed = run_process(working, f"{name}.vvp", sim_cmd)
    elapsed = time.time() - start
    require(replayed.returncode == 0, f"{name} simulation failed; see {working / 'logs' / (name + '.vvp.stdout.log')}")
    return {
        "compile_command": compile_cmd,
        "simulate_command": sim_cmd,
        "elapsed_wall_seconds": elapsed,
        "binary": file_record(binary),
        "stdout": file_record(working / "logs" / f"{name}.vvp.stdout.log"),
        "stderr": file_record(working / "logs" / f"{name}.vvp.stderr.log"),
        "stdout_text": replayed.stdout,
    }


def parse_lm_head_marker(stdout: str, position: int, output_count: int) -> dict[str, int]:
    pattern = re.compile(
        r"ACE2_LM_HEAD_PROJ_PASS position=(?P<position>\d+) outputs=(?P<outputs>\d+) "
        r"token_offset=(?P<offset>\d+) top_token=(?P<token>\d+) top_logit_s8=(?P<logit>-?\d+) cycles=(?P<cycles>\d+)"
    )
    matches = list(pattern.finditer(stdout))
    require(len(matches) == 1, f"missing or duplicated LM-head RTL PASS marker for position {position}")
    data = {key: int(value) for key, value in matches[0].groupdict().items()}
    require(data["position"] == position and data["outputs"] == output_count and data["offset"] == 0, "LM-head RTL marker geometry mismatch")
    return data


def run_rtl(working: Path, output_count: int) -> dict[str, Any]:
    rms_source = rmsnorm_execution_source(working)
    rms = compile_and_run(working, "final_rmsnorm", [RMS_CORE, rms_source], "ace2_rmsnorm_tb")
    require("ACE2_RMSNORM_TB_PASS cases=2 beats_per_case=56" in rms["stdout_text"], "final RMSNorm RTL PASS marker missing")
    results: dict[str, Any] = {
        "final_rmsnorm": {
            key: value for key, value in rms.items() if key != "stdout_text"
        }
    }
    for position in (0, 1):
        source = lm_head_execution_source(working, position, output_count)
        top = f"ace2_lm_head_position{position}_tb"
        result = compile_and_run(working, f"lm_head_position{position}", [PROJ_CORE, source], top)
        marker = parse_lm_head_marker(result["stdout_text"], position, output_count)
        results[f"lm_head_position{position}"] = {
            key: value for key, value in result.items() if key != "stdout_text"
        } | {"marker": marker}
    return results


def derive_and_run(working: Path, output_count: int, official: bool) -> dict[str, Any]:
    started = time.time()
    verify_upstream()
    validate_model_bindings()
    with safe_open(MODEL, framework="pt", device="cpu") as weights:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
    require(output_count <= embedding.shape[0], "requested output_count exceeds vocab")
    positions = derive_final_rmsnorm(working, norm_gain)
    weights = derive_lm_head_weights(working, embedding, output_count)
    summaries: dict[str, Any] = {}
    for position in (0, 1):
        summaries[str(position)] = derive_lm_head_position(
            working,
            position,
            positions[position],
            embedding,
            weights["qweight"],
            weights["weight_scale"],
            output_count,
        )
    rtl = run_rtl(working, output_count)
    for position in (0, 1):
        marker = rtl[f"lm_head_position{position}"]["marker"]
        require(marker["token"] == summaries[str(position)]["top_token"], f"position {position} RTL top-token mismatch")
        require(marker["logit"] == summaries[str(position)]["top_logit_s8"], f"position {position} RTL top-logit mismatch")
    elapsed = time.time() - started
    return {
        "status": "PASS_LM_HEAD_TOP_TOKEN_RTL_RUN_UNCHECKED" if official else "PASS_LM_HEAD_TOP_TOKEN_PREFLIGHT_UNOFFICIAL",
        "mission": MISSION,
        "official_attempt": official,
        "output_count": output_count,
        "elapsed_wall_seconds": elapsed,
        "peak_rss_kib": rss_kib(),
        "tool_versions": tool_versions(),
        "rmsnorm_positions": {str(position): positions[position]["summary"] for position in (0, 1)},
        "lm_head_weight": weights["artifacts"],
        "lm_head_positions": summaries,
        "rtl": rtl,
    }


def phase_log(working: Path, phase: str, status: str, started: float, extra: dict[str, Any] | None = None) -> None:
    write_json(
        working / "phase-logs" / f"{phase}.result.json",
        {
            "phase": phase,
            "status": status,
            "elapsed_wall_seconds": time.time() - started,
            "peak_rss_kib": rss_kib(),
            **(extra or {}),
        },
    )


def do_preflight() -> None:
    if PREFLIGHT.exists():
        shutil.rmtree(PREFLIGHT)
    PREFLIGHT.mkdir(parents=True)
    write_json(PREFLIGHT / "frozen_contract.json", frozen_contract(64))
    summary = derive_and_run(PREFLIGHT, 64, official=False)
    write_json(PREFLIGHT / "run_summary.json", summary)
    write_sha256s(PREFLIGHT)
    print(json.dumps({"status": summary["status"], "path": public_path(PREFLIGHT)}, sort_keys=True))


def do_prepare() -> None:
    started = time.time()
    require(not ATTEMPT.exists(), f"official attempt already exists: {ATTEMPT}")
    ATTEMPT.mkdir(parents=True)
    write_json(ATTEMPT / "phase-logs/prepare.started.json", {"phase": "prepare", "started_unix": started})
    contract = frozen_contract(VOCAB)
    write_json(ATTEMPT / "frozen_contract.json", contract)
    write_json(ATTEMPT / "status.json", {"status": "PREPARED_AWAITING_RUN", "mission": MISSION})
    phase_log(ATTEMPT, "prepare", "PASS_PREPARED_AWAITING_RUN", started, {"output_count": VOCAB})
    print(json.dumps({"status": "PREPARED_AWAITING_RUN", "path": public_path(ATTEMPT)}, sort_keys=True))


def do_run() -> None:
    started = time.time()
    require(ATTEMPT.is_dir(), "official attempt has not been prepared")
    require((ATTEMPT / "frozen_contract.json").is_file(), "frozen contract missing")
    require(not (ATTEMPT / "phase-logs/run.started.json").exists(), "official run phase already started")
    write_json(ATTEMPT / "phase-logs/run.started.json", {"phase": "run", "started_unix": started})
    try:
        summary = derive_and_run(ATTEMPT, VOCAB, official=True)
        write_json(ATTEMPT / "run_summary.json", summary)
        write_json(ATTEMPT / "status.json", {"status": summary["status"], "mission": MISSION})
        phase_log(ATTEMPT, "run", summary["status"], started, {"output_count": VOCAB})
    except Exception as exc:
        write_json(
            ATTEMPT / "run_summary.json",
            {
                "status": "FAIL_LM_HEAD_TOP_TOKEN_RTL_RUN",
                "mission": MISSION,
                "error": repr(exc),
                "elapsed_wall_seconds": time.time() - started,
                "peak_rss_kib": rss_kib(),
            },
        )
        phase_log(ATTEMPT, "run", "FAIL_LM_HEAD_TOP_TOKEN_RTL_RUN", started, {"error": repr(exc)})
        raise
    print(json.dumps({"status": summary["status"], "path": public_path(ATTEMPT)}, sort_keys=True))


def do_check() -> None:
    started = time.time()
    require(ATTEMPT.is_dir(), "official attempt has not been prepared")
    require(not (ATTEMPT / "phase-logs/check.started.json").exists(), "official check phase already started")
    write_json(ATTEMPT / "phase-logs/check.started.json", {"phase": "check", "started_unix": started})
    summary = json.loads((ATTEMPT / "run_summary.json").read_text(encoding="utf-8"))
    require(summary.get("status") == "PASS_LM_HEAD_TOP_TOKEN_RTL_RUN_UNCHECKED", "run summary is not a passing unchecked official run")
    require(summary.get("output_count") == VOCAB, "run output count is not full vocab")
    checks = []
    for position in (0, 1):
        lm = summary["lm_head_positions"][str(position)]
        marker = summary["rtl"][f"lm_head_position{position}"]["marker"]
        require(marker["token"] == lm["top_token"], f"position {position} checked top-token mismatch")
        require(marker["logit"] == lm["top_logit_s8"], f"position {position} checked top-logit mismatch")
        require(marker["outputs"] == VOCAB, f"position {position} RTL output count mismatch")
        checks.append(
            {
                "position": position,
                "top_token": lm["top_token"],
                "top_logit_s8": lm["top_logit_s8"],
                "rtl_cycles": marker["cycles"],
                "saturation_count": lm["saturation_count"],
                "output_sha256": lm["output_sha256"],
            }
        )
    check = {
        "status": "PASS_LM_HEAD_TOP_TOKEN_EXACT_CHECK_AWAITING_FRESH_L2",
        "mission": MISSION,
        "output_count": VOCAB,
        "checks": checks,
        "elapsed_wall_seconds": time.time() - started,
        "peak_rss_kib": rss_kib(),
        "run_summary": file_record(ATTEMPT / "run_summary.json"),
    }
    write_json(ATTEMPT / "check.json", check)
    write_json(ATTEMPT / "status.json", {"status": check["status"], "mission": MISSION})
    phase_log(ATTEMPT, "check", check["status"], started, {"output_count": VOCAB})
    write_sha256s(ATTEMPT)
    check["sha256_manifest_members"] = verify_sha256s(ATTEMPT)
    write_json(ATTEMPT / "check.json", check)
    write_json(ATTEMPT / "status.json", {"status": check["status"], "mission": MISSION})
    write_sha256s(ATTEMPT)
    print(json.dumps({"status": check["status"], "path": public_path(ATTEMPT)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true", help="run an unofficial 64-token rehearsal under build/")
    group.add_argument("--prepare", action="store_true", help="freeze the official exact-once attempt")
    group.add_argument("--run", action="store_true", help="execute the official exact-once run")
    group.add_argument("--check", action="store_true", help="check and seal the official attempt")
    args = parser.parse_args()
    if args.preflight:
        do_preflight()
    elif args.prepare:
        do_prepare()
    elif args.run:
        do_run()
    elif args.check:
        do_check()


if __name__ == "__main__":
    main()
