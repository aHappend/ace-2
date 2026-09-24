#!/usr/bin/env python3
"""Adapt the canonical local LoRA V4 layer-0 Q projection to ACE-2 W4A8 RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/verification/lora-v4-layer0-qproj-w4a8-rtl-v1"
CONTRACT = OUT / "frozen_contract.json"
FAILED_ATTEMPT = OUT / "attempt-0001"
REPAIR_CONTRACT = OUT / "repair-0002-contract.json"
ATTEMPT = OUT / "attempt-0002"
SNAPSHOT = Path(
    "/home/argustest/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
MODEL = SNAPSHOT / "model.safetensors"
CONFIG = SNAPSHOT / "config.json"
TOKENIZER_CONFIG = SNAPSHOT / "tokenizer_config.json"
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
TRAINING_RESULT = (
    ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/attempt-0001/"
    "training_result.json"
)
TERMINAL_REVIEW = (
    ROOT
    / "research/raw/specification/"
    "qwen25-05b-instruct-bf16-lora-product-v4-terminal-fresh-review.json"
)
RTL = ROOT / "rtl/ace2_w4a8_proj_core.sv"
TB = ROOT / "verification/tb/ace2_lora_v4_layer0_qproj_tb.sv"
REFERENCE = ROOT / "tools/ace2_projection_reference.py"
SELF = Path(__file__).resolve()

MODEL_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
ADAPTER_SHA256 = "c9476d6ccc687a4d68de1951eb889b689daaf9c5d3822b4c5acfc26a1cd38a28"
CHECKPOINT_TREE_SHA256 = "8cee9b9343e49b66ee2ea427578f87335ffa259919356cdd119cd8712ff7eeee"
PROMPT = "ACE-2 W4A8 layer-zero checkpoint adaptation."
TOKEN_INDEX = 0
HIDDEN = 896
MAC_LANES = 4
GROUPS = HIDDEN // MAC_LANES
LORA_RANK = 16
LORA_ALPHA = 32
RMS_EPSILON = 1e-6

THRESHOLDS = {
    "rtl_vs_python_fixed_point": {
        "output_integer_mismatches_max": 0,
        "accumulator_integer_mismatches_max": 0,
        "max_abs_integer_error": 0,
        "max_abs_dequantized_error": 0.0,
    },
    "merged_float_vs_w4a8": {
        "max_abs_dequantized_error_max": 1.0,
        "relative_l2_error_max": 0.20,
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def public_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def command_version(command: list[str]) -> str:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    output = (completed.stdout + completed.stderr).strip().splitlines()
    return output[0] if output else f"exit={completed.returncode}"


def tree_record(base: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"file_count": len(rows), "sha256": sha256_bytes(canonical), "files": rows}


def checkpoint_inventory() -> dict[str, Any]:
    files = []
    for path in sorted(item for item in ADAPTER_DIR.iterdir() if item.is_file()):
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"path": str(ADAPTER_DIR.resolve()), "files": files, "inventory_sha256": sha256_bytes(canonical)}


def tokenizer_ids() -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    encoded = tokenizer(PROMPT, add_special_tokens=False, return_attention_mask=False)
    return [int(value) for value in encoded["input_ids"]]


def architecture_mapping() -> list[dict[str, Any]]:
    return [
        {"operator": "layer_0.input_rmsnorm", "rtl": "ace2_rmsnorm_core", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.q_proj", "rtl": "ace2_w4a8_proj_core", "status": "official_attempt_0001"},
        {"operator": "layer_0.k_proj", "rtl": "ace2_w4a8_proj_core", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.v_proj", "rtl": "ace2_w4a8_proj_core", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.rope_qk", "rtl": "ace2_rope_core", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.attention_score", "rtl": "ace2_attention_score_core", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.softmax", "rtl": "ace2_softmax_core", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.attention_value", "rtl": "ace2_attention_compose_core", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.o_proj", "rtl": "ace2_w4a8_proj_core", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.attention_residual_add", "rtl": "ACE2_OPCODE_RESIDUAL_ADD", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.post_attention_rmsnorm", "rtl": "ace2_rmsnorm_core", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.mlp_gate_proj", "rtl": "ace2_w4a8_proj_core", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.mlp_up_proj", "rtl": "ace2_w4a8_proj_core", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.silu_gate", "rtl": "ace2_silu_gate_core", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.mlp_down_proj", "rtl": "ace2_w4a8_proj_core", "status": "mapped_not_executed_this_increment"},
        {"operator": "layer_0.mlp_residual_add", "rtl": "ACE2_OPCODE_RESIDUAL_ADD", "status": "mapped_not_executed_this_increment"},
    ]


def validate_provenance() -> dict[str, Any]:
    required = [MODEL, CONFIG, TOKENIZER_CONFIG, ADAPTER, ADAPTER_CONFIG, CHECKPOINT_IDENTITY, TRAINING_RESULT, TERMINAL_REVIEW, RTL, TB, REFERENCE, SELF]
    for path in required:
        require(path.is_file(), f"required local artifact missing: {path}")
    require(sha256_file(MODEL) == MODEL_SHA256, "base model hash changed")
    require(sha256_file(ADAPTER) == ADAPTER_SHA256, "canonical adapter hash changed")
    identity = read_json(CHECKPOINT_IDENTITY)
    training = read_json(TRAINING_RESULT)
    review = read_json(TERMINAL_REVIEW)
    config = read_json(CONFIG)
    adapter_config = read_json(ADAPTER_CONFIG)
    require(identity["path"].endswith("checkpoints/checkpoint-176"), "selected checkpoint path changed")
    require(identity["tree_sha256"] == CHECKPOINT_TREE_SHA256, "selected checkpoint tree changed")
    require(identity["files"]["adapter_model.safetensors"]["sha256"] == ADAPTER_SHA256, "checkpoint identity adapter hash changed")
    require(training["selected_epoch"] == 4, "probe-locked epoch changed")
    require(training["status"] == "PASS_CHECKPOINT_LOCKED_BEFORE_DEV", "training terminal checkpoint status changed")
    require(review["review"]["status"] == "done", "independent terminal review status changed")
    require("terminal NO-GO" in review["review"]["reason"], "quality NO-GO claim boundary changed")
    require(config["architectures"] == ["Qwen2ForCausalLM"], "Qwen architecture changed")
    require(config["hidden_size"] == HIDDEN, "hidden size changed")
    require(config["intermediate_size"] == 4864, "intermediate size changed")
    require(config["num_attention_heads"] == 14 and config["num_key_value_heads"] == 2, "attention geometry changed")
    require(config["num_hidden_layers"] == 24, "layer count changed")
    require(config["hidden_act"] == "silu", "MLP activation changed")
    require(float(config["rms_norm_eps"]) == RMS_EPSILON, "RMSNorm epsilon changed")
    require(adapter_config["peft_type"] == "LORA", "adapter type changed")
    require(adapter_config["r"] == LORA_RANK and adapter_config["lora_alpha"] == LORA_ALPHA, "LoRA scaling changed")
    require("q_proj" in adapter_config["target_modules"], "layer-0 Q projection is not adapted")
    return {
        "checkpoint_identity": identity,
        "training_status": training["status"],
        "selected_epoch": training["selected_epoch"],
        "quality_claim_boundary": "independent_terminal_no_go_36_of_56_below_48_pass_minimum",
        "model_config": config,
        "adapter_config": adapter_config,
    }


def prepare() -> None:
    require(not CONTRACT.exists(), f"frozen contract already exists: {CONTRACT}")
    require(not ATTEMPT.exists(), f"official attempt already exists: {ATTEMPT}")
    provenance = validate_provenance()
    ids = tokenizer_ids()
    require(ids, "fixed prompt tokenized to an empty sequence")
    require(TOKEN_INDEX < len(ids), "fixed token index is outside tokenizer output")
    contract = {
        "schema_version": 1,
        "mission_increment": "canonical_lora_v4_layer0_q_proj_w4a8_rtl",
        "official_attempt_policy": {"first_attempt": "attempt-0001", "immutable_after_execution": True, "repairs_use_new_attempt_directory": True},
        "canonical_checkpoint": {
            "selection": "V4 probe-locked epoch 4 checkpoint selected before dev evaluation",
            "inventory": checkpoint_inventory(),
            "adapter": file_record(ADAPTER),
            "base_model": file_record(MODEL),
            "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
            "quality_claim_boundary": provenance["quality_claim_boundary"],
        },
        "architecture": {
            "family": "Qwen2.5-compatible Qwen2ForCausalLM",
            "geometry": {"layers": 24, "hidden": 896, "intermediate": 4864, "attention_heads": 14, "kv_heads": 2, "head_dim": 64},
            "lora_merge": "W_effective = W_base_bf16_to_fp32 + (lora_alpha / r) * (B_fp32 @ A_fp32)",
            "operator_mapping": architecture_mapping(),
        },
        "fixed_input": {
            "text": PROMPT,
            "tokenizer_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "tokenizer_add_special_tokens": False,
            "token_ids": ids,
            "selected_token_index": TOKEN_INDEX,
            "selected_token_id": ids[TOKEN_INDEX],
        },
        "quantization": {
            "activation": "symmetric signed-int8 per selected token, scale=max(abs(input_rmsnorm))/127, round-to-nearest-even, clamp[-128,127]",
            "weight": "merged layer-0 q_proj symmetric signed-int4 per output channel, scale=max(abs(row))/7, round-to-nearest-even, clamp[-8,7], low-nibble-first lanes",
            "accumulator": "signed-int32 exact dot product",
            "requantization": "signed-int32 multiplier plus unsigned-6 right shift, round-to-nearest-even, output zero-point 0, signed-int8 saturation",
            "output_scale": "per selected token scalar max(abs(merged_float_q_proj))/127",
        },
        "thresholds": THRESHOLDS,
        "public_rtl_contract": {
            "module": "ace2_w4a8_proj_core",
            "parameters": {"K_SIZE": HIDDEN, "MAC_LANES": MAC_LANES, "ACT_WIDTH": 8, "WGT_WIDTH": 4, "ACC_WIDTH": 32},
            "clock_period_ns": 10.0,
            "reset": "asynchronous active-low rst_ni",
            "handshake": "start_valid/start_ready, pair_valid/pair_ready, meta_valid/meta_ready, out_valid/out_ready",
            "source": public_file_record(RTL),
            "testbench": public_file_record(TB),
        },
        "source_bindings": {
            "runner": public_file_record(SELF),
            "independent_projection_reference": public_file_record(REFERENCE),
            "config": file_record(CONFIG),
            "tokenizer_config": file_record(TOKENIZER_CONFIG),
            "checkpoint_identity": public_file_record(CHECKPOINT_IDENTITY),
            "training_result": public_file_record(TRAINING_RESULT),
            "terminal_review": public_file_record(TERMINAL_REVIEW),
            "rtl_tree": tree_record(ROOT / "rtl"),
            "constraints_tree": tree_record(ROOT / "constraints"),
        },
        "tools": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "safetensors": __import__("safetensors").__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "iverilog": command_version(["iverilog", "-V"]),
            "vvp": command_version(["vvp", "-V"]),
        },
        "commands": {
            "prepare": ".venv/bin/python tools/run_lora_v4_layer0_qproj_rtl.py --prepare",
            "official_attempt_0001": ".venv/bin/python tools/run_lora_v4_layer0_qproj_rtl.py --run",
            "decisive_verifier": ".venv/bin/python tools/run_lora_v4_layer0_qproj_rtl.py --check",
        },
        "expected_artifacts": [
            "attempt-0001/manifest.json",
            "attempt-0001/conversion.json",
            "attempt-0001/comparison.json",
            "attempt-0001/tensors",
            "attempt-0001/vectors",
            "attempt-0001/logs/iverilog.stdout.log",
            "attempt-0001/logs/iverilog.stderr.log",
            "attempt-0001/logs/vvp.stdout.log",
            "attempt-0001/logs/vvp.stderr.log",
            "attempt-0001/SHA256SUMS",
        ],
    }
    write_json(CONTRACT, contract)
    print(f"FROZEN contract={CONTRACT.relative_to(ROOT)} sha256={sha256_file(CONTRACT)} token_ids={ids}")


def validate_contract(*, allow_repaired_harness: bool = False) -> dict[str, Any]:
    require(CONTRACT.is_file(), "frozen contract is absent; run --prepare first")
    contract = read_json(CONTRACT)
    provenance = validate_provenance()
    require(contract["canonical_checkpoint"]["adapter"]["sha256"] == ADAPTER_SHA256, "frozen adapter hash differs")
    require(contract["canonical_checkpoint"]["base_model"]["sha256"] == MODEL_SHA256, "frozen base hash differs")
    require(contract["fixed_input"]["token_ids"] == tokenizer_ids(), "tokenizer-derived fixed input changed")
    source_checks = (("independent_projection_reference", REFERENCE),) if allow_repaired_harness else (("runner", SELF), ("independent_projection_reference", REFERENCE))
    for name, path in source_checks:
        require(contract["source_bindings"][name]["sha256"] == sha256_file(path), f"frozen {name} changed")
    require(contract["public_rtl_contract"]["source"]["sha256"] == sha256_file(RTL), "frozen RTL source changed")
    if not allow_repaired_harness:
        require(contract["public_rtl_contract"]["testbench"]["sha256"] == sha256_file(TB), "frozen testbench changed")
    require(contract["source_bindings"]["rtl_tree"]["sha256"] == tree_record(ROOT / "rtl")["sha256"], "frozen RTL tree changed")
    require(contract["source_bindings"]["constraints_tree"]["sha256"] == tree_record(ROOT / "constraints")["sha256"], "frozen constraints tree changed")
    contract["validated_provenance"] = provenance
    return contract


def prepare_repair() -> None:
    require(FAILED_ATTEMPT.is_dir(), "failed attempt-0001 directory is absent")
    require(not REPAIR_CONTRACT.exists(), f"repair contract already exists: {REPAIR_CONTRACT}")
    require(not ATTEMPT.exists(), f"repair attempt already exists: {ATTEMPT}")
    contract = validate_contract(allow_repaired_harness=True)
    derived = derive_tensors(contract)
    require(derived["output_q"].numel() == HIDDEN, "derivation preflight did not produce all Q-projection channels")
    failure = {
        "status": "NO_EXECUTION_REFERENCE_GENERATION_FAILURE",
        "failure_taxonomy": "experiment_harness_reference_generation_defect",
        "root_cause_hypothesis": "round_shift_even is a two-argument helper but attempt-0001 passed accumulator, multiplier, and shift as three arguments",
        "exception": {
            "type": "TypeError",
            "message": "round_shift_even() takes 2 positional arguments but 3 were given",
            "source_line_at_failure": 427,
        },
        "rtl_execution": False,
        "compile_execution": False,
        "vectors_written": False,
        "correctness_conclusion": "none",
        "regression": "derive all 896 fixed-point Q-projection outputs in memory before freezing repair attempt-0002",
        "command": ".venv/bin/python tools/run_lora_v4_layer0_qproj_rtl.py --run",
    }
    write_json(FAILED_ATTEMPT / "terminal.json", failure)
    write_sha256s(FAILED_ATTEMPT)
    repair = {
        "schema_version": 1,
        "attempt": "attempt-0002",
        "parent_frozen_contract": public_file_record(CONTRACT),
        "failed_attempt": {
            "terminal": public_file_record(FAILED_ATTEMPT / "terminal.json"),
            "sha256s": public_file_record(FAILED_ATTEMPT / "SHA256SUMS"),
        },
        "repair_scope": "fix reference round-to-even call arity and retarget the dedicated testbench vector directory; no RTL, checkpoint, input, quantization, or threshold change",
        "unchanged": {
            "canonical_checkpoint": contract["canonical_checkpoint"],
            "fixed_input": contract["fixed_input"],
            "quantization": contract["quantization"],
            "thresholds": contract["thresholds"],
            "public_rtl_source": contract["public_rtl_contract"]["source"],
            "rtl_tree": contract["source_bindings"]["rtl_tree"],
            "constraints_tree": contract["source_bindings"]["constraints_tree"],
        },
        "repaired_harness": {
            "runner": public_file_record(SELF),
            "testbench": public_file_record(TB),
            "derivation_preflight": {"status": "PASS", "channels": int(derived["output_q"].numel())},
        },
        "commands": {
            "prepare_repair": ".venv/bin/python tools/run_lora_v4_layer0_qproj_rtl.py --prepare-repair",
            "attempt_0002": ".venv/bin/python tools/run_lora_v4_layer0_qproj_rtl.py --run",
            "decisive_verifier": ".venv/bin/python tools/run_lora_v4_layer0_qproj_rtl.py --check",
        },
    }
    write_json(REPAIR_CONTRACT, repair)
    print(f"FROZEN_REPAIR contract={REPAIR_CONTRACT.relative_to(ROOT)} sha256={sha256_file(REPAIR_CONTRACT)} preflight_channels={HIDDEN}")


def validate_repair_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = validate_contract(allow_repaired_harness=True)
    require(REPAIR_CONTRACT.is_file(), "repair-0002 contract is absent; run --prepare-repair first")
    repair = read_json(REPAIR_CONTRACT)
    require(repair["parent_frozen_contract"]["sha256"] == sha256_file(CONTRACT), "repair parent contract binding changed")
    require(repair["repaired_harness"]["runner"]["sha256"] == sha256_file(SELF), "repaired runner changed")
    require(repair["repaired_harness"]["testbench"]["sha256"] == sha256_file(TB), "repaired testbench changed")
    require(repair["unchanged"]["fixed_input"] == contract["fixed_input"], "repair changed fixed input")
    require(repair["unchanged"]["quantization"] == contract["quantization"], "repair changed quantization")
    require(repair["unchanged"]["thresholds"] == contract["thresholds"], "repair changed thresholds")
    return contract, repair


def raw_tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().tobytes(order="C")
    return value.numpy().tobytes(order="C")


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
    real = real.to(torch.float64)
    multiplier = torch.zeros_like(real, dtype=torch.int64)
    right_shift = torch.full_like(real, -1, dtype=torch.int64)
    for shift in range(63, -1, -1):
        candidate = torch.round(real * math.ldexp(1.0, shift))
        selected = (right_shift < 0) & (candidate <= (1 << 31) - 1)
        multiplier = torch.where(selected, candidate.to(torch.int64), multiplier)
        right_shift = torch.where(selected, torch.full_like(right_shift, shift), right_shift)
    require(not bool(torch.any(right_shift < 0)), "unrepresentable projection multiplier")
    return multiplier, right_shift


def pack_s4(values: torch.Tensor) -> bytes:
    flat = values.reshape(-1).to(torch.int16)
    require(flat.numel() % 2 == 0, "signed-int4 tensor element count must be even")
    packed = (flat[0::2] & 0xF) | ((flat[1::2] & 0xF) << 4)
    return packed.to(torch.uint8).numpy().tobytes(order="C")


def write_binary(path: Path, raw: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return public_file_record(path)


def write_hex(path: Path, values: Iterable[int], width: int) -> dict[str, Any]:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{int(value) & mask:0{digits}x}\n" for value in values), encoding="ascii")
    return public_file_record(path)


def derive_tensors(contract: dict[str, Any]) -> dict[str, Any]:
    token_id = int(contract["fixed_input"]["selected_token_id"])
    with safe_open(MODEL, framework="pt", device="cpu") as weights:
        embedding = weights.get_tensor("model.embed_tokens.weight")[token_id].contiguous()
        rms_gain = weights.get_tensor("model.layers.0.input_layernorm.weight").contiguous()
        base_weight = weights.get_tensor("model.layers.0.self_attn.q_proj.weight").contiguous()
    with safe_open(ADAPTER, framework="pt", device="cpu") as adapter:
        lora_a = adapter.get_tensor("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight").contiguous()
        lora_b = adapter.get_tensor("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight").contiguous()
    require(tuple(embedding.shape) == (HIDDEN,), "embedding geometry changed")
    require(tuple(rms_gain.shape) == (HIDDEN,), "RMS gain geometry changed")
    require(tuple(base_weight.shape) == (HIDDEN, HIDDEN), "base Q projection geometry changed")
    require(tuple(lora_a.shape) == (LORA_RANK, HIDDEN), "LoRA A geometry changed")
    require(tuple(lora_b.shape) == (HIDDEN, LORA_RANK), "LoRA B geometry changed")

    embedding_fp32 = embedding.to(torch.float32)
    variance = embedding_fp32.square().mean()
    normalized_bf16 = (embedding_fp32 * torch.rsqrt(variance + RMS_EPSILON)).to(torch.bfloat16)
    rms_output_bf16 = (rms_gain * normalized_bf16).contiguous()
    merged_weight_fp32 = (
        base_weight.to(torch.float32)
        + (float(LORA_ALPHA) / float(LORA_RANK)) * torch.matmul(lora_b.to(torch.float32), lora_a.to(torch.float32))
    ).contiguous()
    float_output_fp32 = torch.mv(merged_weight_fp32, rms_output_bf16.to(torch.float32)).contiguous()

    activation_scale = float(rms_output_bf16.to(torch.float64).abs().max().item()) / 127.0
    output_scale = float(float_output_fp32.to(torch.float64).abs().max().item()) / 127.0
    require(activation_scale > 0.0 and output_scale > 0.0, "zero quantization scale")
    input_q = torch.round(rms_output_bf16.to(torch.float64) / activation_scale).clamp(-128, 127).to(torch.int8)
    weight_fp64 = merged_weight_fp32.to(torch.float64)
    weight_scale = weight_fp64.abs().amax(dim=1) / 7.0
    weight_scale = torch.where(weight_scale > 0.0, weight_scale, torch.ones_like(weight_scale))
    qweight = torch.round(weight_fp64 / weight_scale[:, None]).clamp(-8, 7).to(torch.int8)
    multiplier, right_shift = derive_multiplier(activation_scale * weight_scale / output_scale)
    accumulator = torch.mv(qweight.to(torch.int64), input_q.to(torch.int64))
    require(bool(torch.all(accumulator >= -(1 << 31))) and bool(torch.all(accumulator < (1 << 31))), "projection accumulator exceeds signed-32")
    rounded = torch.tensor(
        [round_shift_even(int(acc) * int(mult), int(shift)) for acc, mult, shift in zip(accumulator.tolist(), multiplier.tolist(), right_shift.tolist(), strict=True)],
        dtype=torch.int64,
    )
    output_q = rounded.clamp(-128, 127).to(torch.int8)
    saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)

    sys.path.insert(0, str(ROOT / "tools"))
    from ace2_projection_reference import ProjectionCase, reference_projection  # pylint: disable=import-outside-toplevel

    independent = reference_projection(
        ProjectionCase(
            name="canonical_lora_v4_layer0_q_proj",
            rows=1,
            reduction_size=HIDDEN,
            activations=[input_q.to(torch.int64).tolist()],
            weights=qweight.to(torch.int64).tolist(),
            multipliers=multiplier.tolist(),
            right_shifts=right_shift.tolist(),
            output_zero_points=[0] * HIDDEN,
            bias_accumulators=[0] * HIDDEN,
        )
    )
    require(independent.outputs == [output_q.to(torch.int64).tolist()], "independent fixed-point reference disagrees")

    return {
        "embedding_bf16": embedding,
        "rms_gain_bf16": rms_gain,
        "rms_output_bf16": rms_output_bf16,
        "merged_weight_fp32": merged_weight_fp32,
        "float_output_fp32": float_output_fp32,
        "activation_scale": activation_scale,
        "input_q": input_q,
        "weight_scale": weight_scale,
        "qweight": qweight,
        "output_scale": output_scale,
        "multiplier": multiplier,
        "right_shift": right_shift,
        "accumulator": accumulator,
        "rounded": rounded,
        "output_q": output_q,
        "saturation": saturation,
        "source_tensor_hashes": {
            "embedding_bf16": sha256_bytes(raw_tensor_bytes(embedding)),
            "input_layernorm_weight_bf16": sha256_bytes(raw_tensor_bytes(rms_gain)),
            "base_q_proj_weight_bf16": sha256_bytes(raw_tensor_bytes(base_weight)),
            "lora_A_fp32": sha256_bytes(raw_tensor_bytes(lora_a)),
            "lora_B_fp32": sha256_bytes(raw_tensor_bytes(lora_b)),
        },
    }


def persist_tensors(derived: dict[str, Any]) -> dict[str, Any]:
    directory = ATTEMPT / "tensors"
    payloads = {
        "embedding_bf16le.bin": raw_tensor_bytes(derived["embedding_bf16"]),
        "input_layernorm_gain_bf16le.bin": raw_tensor_bytes(derived["rms_gain_bf16"]),
        "input_rmsnorm_bf16le.bin": raw_tensor_bytes(derived["rms_output_bf16"]),
        "input_scale_f64le.bin": struct.pack("<d", derived["activation_scale"]),
        "input_q_s8.bin": raw_tensor_bytes(derived["input_q"]),
        "q_proj_merged_f32le.bin": raw_tensor_bytes(derived["merged_weight_fp32"]),
        "q_proj_weight_scales_f64le.bin": raw_tensor_bytes(derived["weight_scale"]),
        "q_proj_qweight_s4_packed.bin": pack_s4(derived["qweight"]),
        "q_proj_qweight_s4_in_s8.bin": raw_tensor_bytes(derived["qweight"]),
        "output_scale_f64le.bin": struct.pack("<d", derived["output_scale"]),
        "multiplier_s32le.bin": b"".join(struct.pack("<i", int(value)) for value in derived["multiplier"].tolist()),
        "right_shift_u6.bin": bytes(int(value) for value in derived["right_shift"].tolist()),
        "accumulator_s32le.bin": b"".join(struct.pack("<i", int(value)) for value in derived["accumulator"].tolist()),
        "rounded_s64le.bin": b"".join(struct.pack("<q", int(value)) for value in derived["rounded"].tolist()),
        "output_q_s8.bin": raw_tensor_bytes(derived["output_q"]),
        "output_float_f32le.bin": raw_tensor_bytes(derived["float_output_fp32"]),
    }
    return {name: write_binary(directory / name, raw) for name, raw in payloads.items()}


def persist_vectors(derived: dict[str, Any]) -> dict[str, Any]:
    directory = ATTEMPT / "vectors"
    input_values = derived["input_q"].to(torch.int64).tolist()
    qweight = derived["qweight"].to(torch.int64)
    activation_words = []
    weight_words = []
    for group in range(GROUPS):
        word = 0
        for lane in range(MAC_LANES):
            word |= (input_values[group * MAC_LANES + lane] & 0xFF) << (lane * 8)
        activation_words.append(word)
    for channel in range(HIDDEN):
        for group in range(GROUPS):
            word = 0
            for lane in range(MAC_LANES):
                word |= (int(qweight[channel, group * MAC_LANES + lane]) & 0xF) << (lane * 4)
            weight_words.append(word)
    return {
        "activation": write_hex(directory / "activation.hex", activation_words, 32),
        "qweight": write_hex(directory / "qweight.hex", weight_words, 16),
        "multiplier": write_hex(directory / "multiplier.hex", derived["multiplier"].tolist(), 32),
        "right_shift": write_hex(directory / "right_shift.hex", derived["right_shift"].tolist(), 8),
        "expected_output": write_hex(directory / "expected_output.hex", derived["output_q"].to(torch.int64).tolist(), 8),
        "expected_accumulator": write_hex(directory / "expected_accumulator.hex", derived["accumulator"].tolist(), 32),
        "expected_saturation": write_hex(directory / "expected_saturation.hex", derived["saturation"].tolist(), 8),
    }


def run_process(command: list[str], log_name: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    logs = ATTEMPT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{log_name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (logs / f"{log_name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed


def parse_rtl(stdout: str) -> list[dict[str, int]]:
    pattern = re.compile(r"RTL_RESULT channel=(\d+) out_u8=(\d+) acc=(-?\d+) saturation=(\d+) overflow=(\d+)")
    rows = []
    for match in pattern.finditer(stdout):
        rows.append({
            "channel": int(match.group(1)),
            "out_u8": int(match.group(2)),
            "acc": int(match.group(3)),
            "saturation": int(match.group(4)),
            "overflow": int(match.group(5)),
        })
    return rows


def compare(derived: dict[str, Any], rtl_rows: list[dict[str, int]]) -> dict[str, Any]:
    expected_output = derived["output_q"].to(torch.int64).tolist()
    expected_acc = derived["accumulator"].to(torch.int64).tolist()
    expected_sat = derived["saturation"].to(torch.int64).tolist()
    require(len(rtl_rows) == HIDDEN, f"RTL emitted {len(rtl_rows)} results, expected {HIDDEN}")
    output_errors = []
    accumulator_errors = []
    saturation_mismatches = 0
    overflow_count = 0
    rtl_signed_output = []
    for channel, row in enumerate(rtl_rows):
        require(row["channel"] == channel, "RTL channel order changed")
        signed_output = row["out_u8"] - 256 if row["out_u8"] >= 128 else row["out_u8"]
        rtl_signed_output.append(signed_output)
        output_errors.append(signed_output - expected_output[channel])
        accumulator_errors.append(row["acc"] - expected_acc[channel])
        saturation_mismatches += int(row["saturation"] != expected_sat[channel])
        overflow_count += int(row["overflow"] != 0)

    output_error_tensor = torch.tensor(output_errors, dtype=torch.int64)
    accumulator_error_tensor = torch.tensor(accumulator_errors, dtype=torch.int64)
    rtl_dequantized = torch.tensor(rtl_signed_output, dtype=torch.float64) * float(derived["output_scale"])
    golden_dequantized = derived["output_q"].to(torch.float64) * float(derived["output_scale"])
    float_reference = derived["float_output_fp32"].to(torch.float64)
    rtl_golden_error = (rtl_dequantized - golden_dequantized).abs()
    quant_error = rtl_dequantized - float_reference
    relative_l2 = float(torch.linalg.vector_norm(quant_error) / torch.linalg.vector_norm(float_reference))
    metrics = {
        "rtl_vs_python_fixed_point": {
            "output_integer_mismatches": int((output_error_tensor != 0).sum().item()),
            "accumulator_integer_mismatches": int((accumulator_error_tensor != 0).sum().item()),
            "max_abs_integer_error": int(output_error_tensor.abs().max().item()),
            "max_abs_accumulator_error": int(accumulator_error_tensor.abs().max().item()),
            "max_abs_dequantized_error": float(rtl_golden_error.max().item()),
            "saturation_flag_mismatches": saturation_mismatches,
            "overflow_count": overflow_count,
        },
        "merged_float_vs_w4a8": {
            "max_abs_dequantized_error": float(quant_error.abs().max().item()),
            "mean_abs_dequantized_error": float(quant_error.abs().mean().item()),
            "rmse_dequantized": float(torch.sqrt(torch.mean(quant_error.square())).item()),
            "relative_l2_error": relative_l2,
            "float_reference_absmax": float(float_reference.abs().max().item()),
            "output_scale": float(derived["output_scale"]),
        },
    }
    exact = metrics["rtl_vs_python_fixed_point"]
    quant = metrics["merged_float_vs_w4a8"]
    passed = (
        exact["output_integer_mismatches"] <= THRESHOLDS["rtl_vs_python_fixed_point"]["output_integer_mismatches_max"]
        and exact["accumulator_integer_mismatches"] <= THRESHOLDS["rtl_vs_python_fixed_point"]["accumulator_integer_mismatches_max"]
        and exact["max_abs_integer_error"] <= THRESHOLDS["rtl_vs_python_fixed_point"]["max_abs_integer_error"]
        and exact["max_abs_dequantized_error"] <= THRESHOLDS["rtl_vs_python_fixed_point"]["max_abs_dequantized_error"]
        and exact["saturation_flag_mismatches"] == 0
        and exact["overflow_count"] == 0
        and quant["max_abs_dequantized_error"] <= THRESHOLDS["merged_float_vs_w4a8"]["max_abs_dequantized_error_max"]
        and quant["relative_l2_error"] <= THRESHOLDS["merged_float_vs_w4a8"]["relative_l2_error_max"]
    )
    return {"status": "PASS" if passed else "FAIL", "thresholds": THRESHOLDS, "metrics": metrics}


def write_sha256s(base: Path) -> None:
    rows = []
    for path in sorted(item for item in base.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        rows.append(f"{sha256_file(path)}  {path.relative_to(base)}\n")
    (base / "SHA256SUMS").write_text("".join(rows), encoding="ascii")


def run() -> None:
    require(not ATTEMPT.exists(), f"official attempt is immutable and already exists: {ATTEMPT}")
    contract, repair = validate_repair_contract()
    ATTEMPT.mkdir(parents=True)
    derived = derive_tensors(contract)
    tensor_artifacts = persist_tensors(derived)
    vector_artifacts = persist_vectors(derived)
    conversion = {
        "status": "PASS",
        "selected_token_id": contract["fixed_input"]["selected_token_id"],
        "source_tensor_hashes": derived["source_tensor_hashes"],
        "scales": {"activation": derived["activation_scale"], "output": derived["output_scale"]},
        "ranges": {
            "activation_q": [int(derived["input_q"].min().item()), int(derived["input_q"].max().item())],
            "qweight": [int(derived["qweight"].min().item()), int(derived["qweight"].max().item())],
            "accumulator": [int(derived["accumulator"].min().item()), int(derived["accumulator"].max().item())],
            "multiplier": [int(derived["multiplier"].min().item()), int(derived["multiplier"].max().item())],
            "right_shift": [int(derived["right_shift"].min().item()), int(derived["right_shift"].max().item())],
        },
        "tensor_artifacts": tensor_artifacts,
        "vector_artifacts": vector_artifacts,
    }
    write_json(ATTEMPT / "conversion.json", conversion)

    binary = ATTEMPT / "sim/ace2_lora_v4_layer0_qproj_tb.vvp"
    binary.parent.mkdir(parents=True)
    compile_command = ["iverilog", "-g2012", "-o", str(binary), str(RTL), str(TB)]
    compiled = run_process(compile_command, "iverilog")
    if compiled.returncode != 0:
        write_json(ATTEMPT / "comparison.json", {"status": "NO_EXECUTION_COMPILE_FAILURE", "returncode": compiled.returncode})
        write_sha256s(ATTEMPT)
        raise RuntimeError("official RTL attempt did not execute because compilation failed")
    simulated = run_process(["vvp", str(binary)], "vvp")
    if simulated.returncode != 0:
        write_json(ATTEMPT / "comparison.json", {"status": "RTL_EXECUTION_FAILURE", "returncode": simulated.returncode, "parsed_results": len(parse_rtl(simulated.stdout))})
        write_sha256s(ATTEMPT)
        raise RuntimeError("official RTL simulation failed")
    rtl_rows = parse_rtl(simulated.stdout)
    comparison = compare(derived, rtl_rows)
    write_json(ATTEMPT / "rtl_outputs.json", rtl_rows)
    write_json(ATTEMPT / "comparison.json", comparison)
    manifest = {
        "schema_version": 1,
        "status": comparison["status"],
        "claim": "canonical LoRA V4 layer-0 q_proj only; no full-layer or model-quality claim",
        "contract": public_file_record(CONTRACT),
        "repair_contract": public_file_record(REPAIR_CONTRACT),
        "repair_scope": repair["repair_scope"],
        "canonical_checkpoint": contract["canonical_checkpoint"],
        "fixed_input": contract["fixed_input"],
        "architecture": contract["architecture"],
        "quantization": contract["quantization"],
        "thresholds": contract["thresholds"],
        "commands": {"compile": compile_command, "simulate": ["vvp", str(binary)], "verify": contract["commands"]["decisive_verifier"]},
        "conversion": public_file_record(ATTEMPT / "conversion.json"),
        "rtl_outputs": public_file_record(ATTEMPT / "rtl_outputs.json"),
        "comparison": public_file_record(ATTEMPT / "comparison.json"),
        "logs": {
            "iverilog_stdout": public_file_record(ATTEMPT / "logs/iverilog.stdout.log"),
            "iverilog_stderr": public_file_record(ATTEMPT / "logs/iverilog.stderr.log"),
            "vvp_stdout": public_file_record(ATTEMPT / "logs/vvp.stdout.log"),
            "vvp_stderr": public_file_record(ATTEMPT / "logs/vvp.stderr.log"),
        },
    }
    write_json(ATTEMPT / "manifest.json", manifest)
    write_sha256s(ATTEMPT)
    print(json.dumps(comparison, sort_keys=True))
    require(comparison["status"] == "PASS", "official RTL comparison failed frozen thresholds")


def verify_manifest(base: Path) -> int:
    count = 0
    for line in (base / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        path = base / relative
        require(path.is_file(), f"manifest member missing: {relative}")
        require(sha256_file(path) == digest, f"manifest member hash changed: {relative}")
        count += 1
    return count


def check() -> None:
    contract, repair = validate_repair_contract()
    require(ATTEMPT.is_dir(), "repair attempt-0002 is absent")
    member_count = verify_manifest(ATTEMPT)
    comparison = read_json(ATTEMPT / "comparison.json")
    manifest = read_json(ATTEMPT / "manifest.json")
    rtl_rows = read_json(ATTEMPT / "rtl_outputs.json")
    require(comparison["status"] == "PASS", "comparison status is not PASS")
    require(manifest["status"] == "PASS", "manifest status is not PASS")
    require(manifest["contract"]["sha256"] == sha256_file(CONTRACT), "manifest contract binding changed")
    require(manifest["repair_contract"]["sha256"] == sha256_file(REPAIR_CONTRACT), "manifest repair contract binding changed")
    require(len(rtl_rows) == HIDDEN, "RTL output count changed")
    require(contract["thresholds"] == comparison["thresholds"], "comparison thresholds differ from frozen contract")
    print(
        "PASS canonical_lora_v4_layer0_qproj_w4a8_rtl "
        f"manifest_members={member_count} rtl_channels={len(rtl_rows)} "
        f"max_int_error={comparison['metrics']['rtl_vs_python_fixed_point']['max_abs_integer_error']} "
        f"max_dequant_error={comparison['metrics']['merged_float_vs_w4a8']['max_abs_dequantized_error']:.9f} "
        f"relative_l2={comparison['metrics']['merged_float_vs_w4a8']['relative_l2_error']:.9f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--prepare-repair", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.prepare_repair:
        prepare_repair()
    elif args.run:
        run()
    else:
        check()


if __name__ == "__main__":
    main()
