#!/usr/bin/env python3
"""Prepare and audit a mission-local zero-forward Qwen integration slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from huggingface_hub.constants import HF_HUB_CACHE
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
MISSION_ID = "rtl-full-qwen-autoregressive-integration-v1"
DEFAULT_OUT = ROOT / "evidence" / "verification" / MISSION_ID
MISSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    f"{MISSION_ID}/mission.json"
)
CHECKPOINT = MISSION.with_name("CHECKPOINT.md")
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
SNAPSHOT = (
    Path(HF_HUB_CACHE)
    / "models--Qwen--Qwen2.5-0.5B"
    / "snapshots"
    / REVISION
)
CONFIG = SNAPSHOT / "config.json"
MODEL = SNAPSHOT / "model.safetensors"
GENERATION_CONFIG = SNAPSHOT / "generation_config.json"
SCALES = (
    ROOT
    / "evidence/layer0_tile_bfp_score_attention_v1/paired-smoke-20260801-v1/derived_scales.json"
)
UPSTREAM_RUN_CONTRACT = SCALES.with_name("run_contract.json")
UPSTREAM_RESULTS = SCALES.with_name("results.json")
PKG = ROOT / "rtl/ace2_pkg.sv"
SHELL = ROOT / "rtl/ace2_shell.sv"
SHELL_TB = ROOT / "verification/tb/ace2_shell_tb.sv"

HIDDEN = 896
INTERMEDIATE = 4864
HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64
LAYERS = 24
VOCAB = 151936
LM_TILE = 32
CONTEXT_TILE = 8
KV_BYTES_PER_TOKEN = 272

W4_BASE = 0x0000_0001_0000_0000
META_BASE = 0x0000_0002_0000_0000
NORM_BASE = 0x0000_0003_0000_0000
AUX_BASE = 0x0000_0003_1000_0000
ROPE_TABLE_BASE = 0x0000_0004_0000_0000
KV_BASE = 0x0000_0008_0000_0000
BUFFER_BASE = 0x0000_0010_0000_0000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def align(value: int, boundary: int = 16) -> int:
    return (value + boundary - 1) // boundary * boundary


def tensor_spec(weights: Any, key: str) -> dict[str, Any]:
    require(key in weights.keys(), f"missing raw safetensors key: {key}")
    tensor = weights.get_slice(key)
    return {
        "key": key,
        "dtype": str(tensor.get_dtype()),
        "shape": list(tensor.get_shape()),
    }


def expected_linear_shapes(layer: int) -> list[tuple[str, tuple[int, int]]]:
    prefix = f"model.layers.{layer}"
    return [
        (f"{prefix}.self_attn.q_proj", (HIDDEN, HIDDEN)),
        (f"{prefix}.self_attn.k_proj", (KV_HEADS * HEAD_DIM, HIDDEN)),
        (f"{prefix}.self_attn.v_proj", (KV_HEADS * HEAD_DIM, HIDDEN)),
        (f"{prefix}.self_attn.o_proj", (HIDDEN, HIDDEN)),
        (f"{prefix}.mlp.gate_proj", (INTERMEDIATE, HIDDEN)),
        (f"{prefix}.mlp.up_proj", (INTERMEDIATE, HIDDEN)),
        (f"{prefix}.mlp.down_proj", (HIDDEN, INTERMEDIATE)),
    ]


def metadata_summary(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sha256": sha256_json(value),
        "keys": sorted(value),
    }
    for key in (
        "input_scale",
        "hardware_input_scale",
        "output_scale",
        "qweight_sha256",
    ):
        if key in value:
            result[key] = value[key]
    for key in ("weight_scale", "multiplier", "right_shift", "bias_accumulator"):
        if key in value:
            item = value[key]
            result[f"{key}_count"] = len(item) if isinstance(item, list) else 0
    return result


def build_model_inventory(config: dict[str, Any], scales: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(config["hidden_size"] == HIDDEN, "hidden size changed")
    require(config["intermediate_size"] == INTERMEDIATE, "intermediate size changed")
    require(config["num_hidden_layers"] == LAYERS, "layer count changed")
    require(config["num_attention_heads"] == HEADS, "attention head count changed")
    require(config["num_key_value_heads"] == KV_HEADS, "KV head count changed")
    require(config["vocab_size"] == VOCAB, "vocabulary size changed")
    require(config["tie_word_embeddings"] is True, "embedding/lm-head tie changed")
    require(len(scales.get("linears", {})) == LAYERS * 7 + 1, "full linear metadata inventory is incomplete")
    require(len(scales.get("attention", {})) == LAYERS, "full attention metadata inventory is incomplete")

    linear_layout: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    w4_cursor = W4_BASE
    meta_cursor = META_BASE
    norm_cursor = NORM_BASE
    aux_cursor = AUX_BASE

    with safe_open(MODEL, framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        embed = tensor_spec(weights, "model.embed_tokens.weight")
        require(embed["shape"] == [VOCAB, HIDDEN], "embedding geometry changed")
        require(embed["dtype"] == "BF16", "embedding dtype changed")
        require("lm_head.weight" not in keys, "unexpected untied lm-head tensor appeared")

        for layer in range(LAYERS):
            prefix = f"model.layers.{layer}"
            input_norm = tensor_spec(weights, f"{prefix}.input_layernorm.weight")
            post_norm = tensor_spec(weights, f"{prefix}.post_attention_layernorm.weight")
            require(input_norm["shape"] == [HIDDEN], f"layer {layer} input norm geometry changed")
            require(post_norm["shape"] == [HIDDEN], f"layer {layer} post norm geometry changed")
            norm_records = []
            for kind, spec in (("input_rmsnorm", input_norm), ("post_attention_rmsnorm", post_norm)):
                record_bytes = 16 + HIDDEN * 2
                norm_records.append(
                    {
                        "operator": kind,
                        "tensor": spec,
                        "image_addr": norm_cursor,
                        "image_bytes": record_bytes,
                        "layout": "16-byte activation-scale record followed by 896 signed-Q7.8 gains",
                    }
                )
                norm_cursor = align(norm_cursor + record_bytes)

            linears = []
            for name, shape in expected_linear_shapes(layer):
                spec = tensor_spec(weights, f"{name}.weight")
                require(spec["shape"] == list(shape), f"{name} geometry changed")
                require(spec["dtype"] == "BF16", f"{name} dtype changed")
                meta = scales["linears"].get(name)
                require(isinstance(meta, dict), f"missing full-model metadata for {name}")
                output_count, input_count = shape
                packed_bytes = output_count * input_count // 2
                metadata_bytes = output_count * 16
                record = {
                    "name": name,
                    "weight": spec,
                    "packed_w4_addr": w4_cursor,
                    "packed_w4_bytes": packed_bytes,
                    "projection_metadata_addr": meta_cursor,
                    "projection_metadata_bytes": metadata_bytes,
                    "projection_metadata": metadata_summary(meta),
                }
                linears.append(record)
                linear_layout.append(record)
                w4_cursor = align(w4_cursor + packed_bytes)
                meta_cursor = align(meta_cursor + metadata_bytes)

            attention_name = f"{prefix}.self_attn"
            attention_meta = scales["attention"].get(attention_name)
            require(isinstance(attention_meta, dict), f"missing attention metadata for layer {layer}")
            aux_records = {}
            for name, size, source in (
                ("rope_q_scale_records", (HIDDEN // 16) * 32, attention_name),
                ("rope_k_scale_records", ((KV_HEADS * HEAD_DIM) // 16) * 32, attention_name),
                ("kv_scale_record", 16, attention_name),
                ("attention_score_records", HEADS * 16, attention_name),
                ("silu_record", 16, f"{prefix}.post_attention_layernorm.output"),
            ):
                aux_records[name] = {
                    "image_addr": aux_cursor,
                    "image_bytes": size,
                    "source_metadata": source,
                }
                aux_cursor = align(aux_cursor + size)
            layers.append(
                {
                    "layer_id": layer,
                    "norms": norm_records,
                    "linears": linears,
                    "attention_metadata_sha256": sha256_json(attention_meta),
                    "aux_metadata": aux_records,
                    "operator_metadata": {
                        key: metadata_summary(scales["operators"][key])
                        for key in (
                            f"{prefix}.input_layernorm.input",
                            f"{prefix}.input_layernorm.output",
                            f"{prefix}.post_attention_residual",
                            f"{prefix}.post_attention_layernorm.output",
                            f"{prefix}.post_mlp_residual",
                        )
                    },
                }
            )

        final_norm = tensor_spec(weights, "model.norm.weight")
        require(final_norm["shape"] == [HIDDEN], "final norm geometry changed")
        final_norm_record = {
            "operator": "final_rmsnorm",
            "tensor": final_norm,
            "image_addr": norm_cursor,
            "image_bytes": 16 + HIDDEN * 2,
            "layout": "16-byte activation-scale record followed by 896 signed-Q7.8 gains",
        }
        norm_cursor = align(norm_cursor + final_norm_record["image_bytes"])

        lm_meta = scales["linears"].get("lm_head")
        require(isinstance(lm_meta, dict), "missing lm-head metadata")
        lm_packed_bytes = VOCAB * HIDDEN // 2
        lm_metadata_bytes = VOCAB * 16
        lm_head_record = {
            "name": "lm_head",
            "tied_weight_tensor": embed,
            "packed_w4_addr": w4_cursor,
            "packed_w4_bytes": lm_packed_bytes,
            "projection_metadata_addr": meta_cursor,
            "projection_metadata_bytes": lm_metadata_bytes,
            "projection_metadata": metadata_summary(lm_meta),
        }
        linear_layout.append(lm_head_record)
        w4_cursor = align(w4_cursor + lm_packed_bytes)
        meta_cursor = align(meta_cursor + lm_metadata_bytes)

    inventory = {
        "schema_version": 1,
        "model": {
            "name": "Qwen2.5-0.5B",
            "revision": REVISION,
            "config": config,
            "embedding": embed,
            "final_norm": final_norm_record,
            "lm_head": lm_head_record,
        },
        "layers": layers,
        "image_layout": {
            "packed_w4": {"base": W4_BASE, "end": w4_cursor, "bytes": w4_cursor - W4_BASE},
            "projection_metadata": {"base": META_BASE, "end": meta_cursor, "bytes": meta_cursor - META_BASE},
            "rmsnorm_metadata": {"base": NORM_BASE, "end": norm_cursor, "bytes": norm_cursor - NORM_BASE},
            "operator_aux_metadata": {"base": AUX_BASE, "end": aux_cursor, "bytes": aux_cursor - AUX_BASE},
            "status": "address_map_only_no_full_model_binary_image_present",
        },
    }
    return inventory, linear_layout


def parse_opcodes() -> dict[str, int]:
    text = PKG.read_text(encoding="utf-8")
    rows = re.findall(r"localparam \[7:0\] ACE2_OPCODE_([A-Z0-9_]+)\s*=\s*8'h([0-9a-fA-F]+);", text)
    result = {name.lower(): int(value, 16) for name, value in rows}
    require(len(result) == 10, "unexpected shell opcode inventory")
    return result


def source_line(path: Path, pattern: str) -> int:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if pattern in line:
            return number
    raise RuntimeError(f"source marker not found: {path}:{pattern}")


def build_surface_inventory(opcodes: dict[str, int]) -> dict[str, Any]:
    shell_text = SHELL.read_text(encoding="utf-8")
    runtime_candidates = sorted(
        str(path.relative_to(ROOT))
        for directory in (ROOT / "tools", ROOT / "reference", ROOT / "verification")
        for path in directory.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.resolve() != Path(__file__).resolve()
        and re.search(r"(autoregressive|command_stream|descriptor_stream|generation_runtime|token_select)", path.name)
    )
    return {
        "shell": {
            "path": str(SHELL.relative_to(ROOT)),
            "sha256": sha256_file(SHELL),
            "direct_command_ingress_line": source_line(SHELL, "input  wire                              cmd_valid_i"),
            "layer_id_ingress_line": source_line(SHELL, "cmd_layer_id_i"),
            "sequence_position_ingress_line": source_line(SHELL, "cmd_sequence_position_i"),
            "attention_context_limit": CONTEXT_TILE,
            "attention_context_limit_line": source_line(SHELL, "ATTN_CONTEXT_MAX = 8"),
            "kv_sequence_stride_expression_line": source_line(SHELL, "kv_sequence_offset_w"),
            "lm_head_shape_line": source_line(SHELL, "proj_lm_head_shape_w"),
            "opcodes": opcodes,
            "contains_embedding_opcode": "EMBED" in shell_text,
            "contains_token_selection_opcode": "ARGMAX" in shell_text or "TOKEN_SELECT" in shell_text,
            "contains_internal_layer_loop": False,
            "contains_internal_generation_loop": False,
        },
        "maintained_interfaces": [
            {"surface": "token_embedding", "owner": "host/runtime preload", "rtl_opcode": None, "status": "interface_plan_only"},
            {"surface": "per_layer_weight_metadata_selection", "owner": "host/runtime addresses", "rtl_opcode": "w4a8_proj/rmsnorm/attention metadata", "status": "address_map_prepared_no_binary_image"},
            {"surface": "layer_to_layer_state", "owner": "host/runtime ping-pong addresses", "rtl_opcode": "all operator descriptors", "status": "schedule_prepared"},
            {"surface": "cross_token_kv_cache", "owner": "shell KV_WRITE plus host-selected per-layer base", "rtl_opcode": "kv_write", "status": "implemented_descriptor_surface"},
            {"surface": "final_rmsnorm", "owner": "shell", "rtl_opcode": "rmsnorm layer_id=24", "status": "implemented_descriptor_surface"},
            {"surface": "lm_head", "owner": "shell plus host tile loop", "rtl_opcode": "w4a8_proj layer_id=24 n=32", "status": "implemented_tile_surface"},
            {"surface": "token_selection", "owner": "host/runtime per frozen workload", "rtl_opcode": None, "status": "raw_safetensors_argmax_witness_prepared_without_model_forward_in_this_mission"},
            {"surface": "generation_control", "owner": "host/runtime", "rtl_opcode": None, "status": "two_token_schedule_prepared_no_executor"},
        ],
        "maintained_runtime_candidates": runtime_candidates,
    }


def buffer_map() -> dict[str, int]:
    names_sizes = [
        ("hidden_a", HIDDEN),
        ("hidden_b", HIDDEN),
        ("norm", HIDDEN),
        ("q", HIDDEN),
        ("k", KV_HEADS * HEAD_DIM),
        ("v", KV_HEADS * HEAD_DIM),
        ("rope_q", HIDDEN),
        ("rope_k", KV_HEADS * HEAD_DIM),
        ("score", HEADS * CONTEXT_TILE * 16),
        ("prob", HEADS * 16),
        ("attn_value", HIDDEN),
        ("o_proj", HIDDEN),
        ("post_norm", HIDDEN),
        ("gate", INTERMEDIATE),
        ("up", INTERMEDIATE),
        ("silu", INTERMEDIATE),
        ("down", HIDDEN),
        ("final_norm", HIDDEN),
        ("logit_tile", LM_TILE),
    ]
    result: dict[str, int] = {}
    cursor = BUFFER_BASE
    for name, size in names_sizes:
        result[name] = cursor
        cursor = align(cursor + size)
    return result


def command(
    commands: list[dict[str, Any]],
    *,
    token_step: int,
    layer: int,
    operator: str,
    opcode: int,
    m: int,
    n: int,
    k: int,
    sequence_position: int,
    src0: int,
    src1: int,
    dst: int,
    scale: int,
    scratch: int = 0,
    flags: int = 9,
    tensor: str | None = None,
    head: int | None = None,
    context_token: int | None = None,
) -> None:
    commands.append(
        {
            "ordinal": len(commands),
            "kind": "shell_command",
            "token_step": token_step,
            "layer_id": layer,
            "operator": operator,
            "opcode": opcode,
            "flags": flags,
            "m": m,
            "n": n,
            "k": k,
            "sequence_position": sequence_position,
            "completion_tag": len(commands) & 0xFFFF,
            "src0_addr": src0,
            "src1_addr": src1,
            "dst_addr": dst,
            "scale_addr": scale,
            "scratch_addr": scratch,
            "weight_tensor": tensor,
            "query_head": head,
            "context_token": context_token,
        }
    )


def linear_by_name(layout: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in layout}


def build_schedule(opcodes: dict[str, int], layout: list[dict[str, Any]], inventory: dict[str, Any]) -> dict[str, Any]:
    linears = linear_by_name(layout)
    buffers = buffer_map()
    commands: list[dict[str, Any]] = []
    host_steps: list[dict[str, Any]] = []
    positions = [0, 1]
    selected_tokens = [151643, None]

    norm_records = {
        (layer["layer_id"], item["operator"]): item
        for layer in inventory["layers"]
        for item in layer["norms"]
    }
    final_norm = inventory["model"]["final_norm"]

    for token_step, position in enumerate(positions):
        hidden_in = buffers["hidden_a"]
        hidden_out = buffers["hidden_b"]
        host_steps.append(
            {
                "ordinal_before_command": len(commands),
                "token_step": token_step,
                "operation": "embedding_row_preload",
                "token_id": selected_tokens[token_step],
                "source": "model.embed_tokens.weight raw BF16 row",
                "transform": "round_ties_to_even(row / layer0_input_scale), then signed-int8 saturation",
                "destination_addr": hidden_in,
                "status": "host_runtime_step",
            }
        )
        for layer in range(LAYERS):
            prefix = f"model.layers.{layer}"
            input_norm = norm_records[(layer, "input_rmsnorm")]
            post_norm = norm_records[(layer, "post_attention_rmsnorm")]
            aux = inventory["layers"][layer]["aux_metadata"]
            command(commands, token_step=token_step, layer=layer, operator="input_rmsnorm", opcode=opcodes["rmsnorm"], m=1, n=HIDDEN, k=0, sequence_position=position, src0=hidden_in, src1=0, dst=buffers["norm"], scale=input_norm["image_addr"])
            for short, n_value, dst in (("q_proj", HIDDEN, buffers["q"]), ("k_proj", KV_HEADS * HEAD_DIM, buffers["k"]), ("v_proj", KV_HEADS * HEAD_DIM, buffers["v"])):
                item = linears[f"{prefix}.self_attn.{short[0]}_proj"]
                command(commands, token_step=token_step, layer=layer, operator=short, opcode=opcodes["w4a8_proj"], m=1, n=n_value, k=HIDDEN, sequence_position=position, src0=buffers["norm"], src1=item["packed_w4_addr"], dst=dst, scale=item["projection_metadata_addr"], tensor=item["name"])
            command(commands, token_step=token_step, layer=layer, operator="rope_q", opcode=opcodes["rope"], m=1, n=HIDDEN, k=0, sequence_position=position, src0=buffers["q"], src1=ROPE_TABLE_BASE, dst=buffers["rope_q"], scale=aux["rope_q_scale_records"]["image_addr"], scratch=0, flags=45)
            command(commands, token_step=token_step, layer=layer, operator="rope_k", opcode=opcodes["rope"], m=1, n=KV_HEADS * HEAD_DIM, k=0, sequence_position=position, src0=buffers["k"], src1=ROPE_TABLE_BASE, dst=buffers["rope_k"], scale=aux["rope_k_scale_records"]["image_addr"], scratch=0, flags=45)
            layer_kv = KV_BASE + layer * 32768 * KV_BYTES_PER_TOKEN
            command(commands, token_step=token_step, layer=layer, operator="kv_write", opcode=opcodes["kv_write"], m=1, n=KV_HEADS * HEAD_DIM, k=0, sequence_position=position, src0=buffers["rope_k"], src1=buffers["v"], dst=0, scale=aux["kv_scale_record"]["image_addr"], scratch=layer_kv)
            context = position + 1
            require(context <= CONTEXT_TILE, "two-token witness unexpectedly exceeds the focused compose tile count")
            for head in range(HEADS):
                kv_head = head // (HEADS // KV_HEADS)
                q_addr = buffers["rope_q"] + head * HEAD_DIM
                score_addr = buffers["score"] + head * CONTEXT_TILE * 16
                prob_addr = buffers["prob"] + head * 16
                value_addr = buffers["attn_value"] + head * HEAD_DIM
                if context == 1:
                    record = layer_kv
                    k_addr = record + kv_head * HEAD_DIM
                    v_addr = record + KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
                    command(commands, token_step=token_step, layer=layer, operator="attention_score", opcode=opcodes["attn_score"], m=1, n=1, k=HEAD_DIM, sequence_position=position, src0=q_addr, src1=k_addr, dst=score_addr, scale=aux["attention_score_records"]["image_addr"] + head * 16, head=head, context_token=0)
                    command(commands, token_step=token_step, layer=layer, operator="softmax", opcode=opcodes["softmax"], m=1, n=1, k=0, sequence_position=position, src0=score_addr, src1=0, dst=prob_addr, scale=0, head=head)
                    command(commands, token_step=token_step, layer=layer, operator="attention_value", opcode=opcodes["attn_value"], m=1, n=1, k=HEAD_DIM, sequence_position=position, src0=prob_addr, src1=v_addr, dst=value_addr, scale=0, head=head, context_token=0)
                else:
                    tile_records = []
                    for context_token in range(context):
                        record = layer_kv + context_token * KV_BYTES_PER_TOKEN
                        k_addr = record + kv_head * HEAD_DIM
                        v_addr = record + KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
                        tile_score_addr = score_addr + context_token * 16
                        command(commands, token_step=token_step, layer=layer, operator="attention_score", opcode=opcodes["attn_score"], m=1, n=1, k=HEAD_DIM, sequence_position=position, src0=q_addr, src1=k_addr, dst=tile_score_addr, scale=aux["attention_score_records"]["image_addr"] + head * 16, head=head, context_token=context_token)
                        tile_records.append((tile_score_addr, v_addr, context_token))
                    phases = (
                        (0, tile_records[0]),
                        (1, tile_records[1]),
                        (2, tile_records[0]),
                        (3, tile_records[1]),
                        (4, tile_records[0]),
                        (6, tile_records[1]),
                    )
                    for phase, (tile_score_addr, v_addr, context_token) in phases:
                        command(commands, token_step=token_step, layer=layer, operator="attention_compose", opcode=opcodes["attn_compose"], m=1, n=1, k=HEAD_DIM, sequence_position=position, src0=tile_score_addr, src1=v_addr, dst=value_addr, scale=0, flags=phase, head=head, context_token=context_token)
            item = linears[f"{prefix}.self_attn.o_proj"]
            command(commands, token_step=token_step, layer=layer, operator="o_proj", opcode=opcodes["w4a8_proj"], m=1, n=HIDDEN, k=HIDDEN, sequence_position=position, src0=buffers["attn_value"], src1=item["packed_w4_addr"], dst=buffers["o_proj"], scale=item["projection_metadata_addr"], tensor=item["name"])
            command(commands, token_step=token_step, layer=layer, operator="attention_residual_add", opcode=opcodes["residual_add"], m=1, n=HIDDEN, k=0, sequence_position=position, src0=hidden_in, src1=buffers["o_proj"], dst=hidden_out, scale=0)
            command(commands, token_step=token_step, layer=layer, operator="post_attention_rmsnorm", opcode=opcodes["rmsnorm"], m=1, n=HIDDEN, k=0, sequence_position=position, src0=hidden_out, src1=0, dst=buffers["post_norm"], scale=post_norm["image_addr"])
            for short, dst in (("gate_proj", buffers["gate"]), ("up_proj", buffers["up"])):
                item = linears[f"{prefix}.mlp.{short}"]
                command(commands, token_step=token_step, layer=layer, operator=f"mlp_{short}", opcode=opcodes["w4a8_proj"], m=1, n=INTERMEDIATE, k=HIDDEN, sequence_position=position, src0=buffers["post_norm"], src1=item["packed_w4_addr"], dst=dst, scale=item["projection_metadata_addr"], tensor=item["name"])
            command(commands, token_step=token_step, layer=layer, operator="silu_gate", opcode=opcodes["silu_gate"], m=1, n=INTERMEDIATE, k=0, sequence_position=position, src0=buffers["gate"], src1=buffers["up"], dst=buffers["silu"], scale=aux["silu_record"]["image_addr"])
            item = linears[f"{prefix}.mlp.down_proj"]
            command(commands, token_step=token_step, layer=layer, operator="mlp_down_proj", opcode=opcodes["w4a8_proj"], m=1, n=HIDDEN, k=INTERMEDIATE, sequence_position=position, src0=buffers["silu"], src1=item["packed_w4_addr"], dst=buffers["down"], scale=item["projection_metadata_addr"], tensor=item["name"])
            command(commands, token_step=token_step, layer=layer, operator="mlp_residual_add", opcode=opcodes["residual_add"], m=1, n=HIDDEN, k=0, sequence_position=position, src0=buffers["down"], src1=hidden_out, dst=hidden_in, scale=0)

        command(commands, token_step=token_step, layer=24, operator="final_rmsnorm", opcode=opcodes["rmsnorm"], m=1, n=HIDDEN, k=0, sequence_position=position, src0=hidden_in, src1=0, dst=buffers["final_norm"], scale=final_norm["image_addr"])
        lm = linears["lm_head"]
        for tile in range(math.ceil(VOCAB / LM_TILE)):
            command(commands, token_step=token_step, layer=24, operator="lm_head_tile", opcode=opcodes["w4a8_proj"], m=1, n=LM_TILE, k=HIDDEN, sequence_position=position, src0=buffers["final_norm"], src1=lm["packed_w4_addr"] + tile * LM_TILE * (HIDDEN // 2), dst=buffers["logit_tile"], scale=lm["projection_metadata_addr"] + tile * LM_TILE * 16, tensor="lm_head")
            commands[-1]["vocab_tile"] = tile
        host_steps.append(
            {
                "ordinal_after_command": len(commands),
                "token_step": token_step,
                "operation": "stable_argmax_or_sampling",
                "source": "4748 ordered lm-head tiles",
                "status": "host_runtime_step_per_frozen_workload",
            }
        )

    counts_per_token = {
        str(token_step): sum(1 for item in commands if item["token_step"] == token_step)
        for token_step in range(len(positions))
    }
    require(counts_per_token == {"0": 6117, "1": 7797}, "unexpected command schedule length")
    return {
        "schema_version": 1,
        "classification": "deterministic_two_token_decode_command_plan_without_model_forward_in_this_mission",
        "positions": positions,
        "buffer_map": buffer_map(),
        "kv_cache": {
            "base": KV_BASE,
            "bytes_per_layer": 32768 * KV_BYTES_PER_TOKEN,
            "bytes_all_layers": LAYERS * 32768 * KV_BYTES_PER_TOKEN,
            "address_rule": "base + layer*32768*272 + position*272",
        },
        "command_count_per_token": counts_per_token,
        "command_count": len(commands),
        "host_steps": host_steps,
        "commands": commands,
    }


def raw_lm_head_witness(seed_token: int, layer0_input_scale: float) -> dict[str, Any]:
    with safe_open(MODEL, framework="pt", device="cpu") as weights:
        table = weights.get_tensor("model.embed_tokens.weight")
        hidden = table[seed_token].to(torch.float32)
        best: list[tuple[float, int]] = []
        chunk = 4096
        for start in range(0, VOCAB, chunk):
            stop = min(VOCAB, start + chunk)
            scores = torch.mv(table[start:stop].to(torch.float32), hidden)
            values, indices = torch.topk(scores, min(8, stop - start))
            best.extend((float(value), start + int(index)) for value, index in zip(values, indices, strict=True))
            best = sorted(best, key=lambda row: (-row[0], row[1]))[:8]
        selected = best[0][1]
        seed_q = torch.round(hidden / layer0_input_scale).clamp(-128, 127).to(torch.int8)
        selected_q = torch.round(table[selected].to(torch.float32) / layer0_input_scale).clamp(-128, 127).to(torch.int8)
        seed_raw = table[seed_token].contiguous().view(torch.uint8).numpy().tobytes()
        selected_raw = table[selected].contiguous().view(torch.uint8).numpy().tobytes()
    return {
        "classification": "raw_safetensors_tied_embedding_lm_head_control_witness_not_decoder_output",
        "seed_token_id": seed_token,
        "seed_embedding_bf16_sha256": hashlib.sha256(seed_raw).hexdigest(),
        "layer0_input_scale": layer0_input_scale,
        "seed_embedding_s8_sha256": hashlib.sha256(seed_q.numpy().tobytes()).hexdigest(),
        "top8_raw_bf16_dot": [{"token_id": token, "score": score} for score, token in best],
        "stable_argmax_token_id": selected,
        "selected_embedding_bf16_sha256": hashlib.sha256(selected_raw).hexdigest(),
        "selected_embedding_s8_sha256": hashlib.sha256(selected_q.numpy().tobytes()).hexdigest(),
        "model_module_calls": 0,
        "model_forward_calls": 0,
    }


def validate_schedule(schedule: dict[str, Any], opcodes: dict[str, int]) -> dict[str, Any]:
    commands = schedule["commands"]
    errors: list[str] = []
    layer_ops = {layer: [] for layer in range(LAYERS)}
    kv_positions: dict[int, set[int]] = {layer: set() for layer in range(LAYERS)}
    for item in commands:
        if item["kind"] != "shell_command":
            continue
        opcode = item["opcode"]
        layer = item["layer_id"]
        if opcode == opcodes["rmsnorm"]:
            if not (item["m"] == 1 and item["n"] == HIDDEN and item["k"] == 0 and layer <= 24):
                errors.append(f"bad RMSNorm descriptor at {item['ordinal']}")
        elif opcode == opcodes["w4a8_proj"]:
            shape = (item["m"], item["n"], item["k"], layer)
            valid = (
                item["m"] == 1
                and ((layer < 24 and ((item["k"] == HIDDEN and item["n"] in {HIDDEN, KV_HEADS * HEAD_DIM, INTERMEDIATE}) or (item["k"] == INTERMEDIATE and item["n"] == HIDDEN))) or (layer == 24 and item["n"] == LM_TILE and item["k"] == HIDDEN))
            )
            if not valid:
                errors.append(f"bad projection descriptor {shape} at {item['ordinal']}")
        elif opcode == opcodes["rope"]:
            if not (layer < 24 and item["m"] == 1 and item["n"] in {HIDDEN, KV_HEADS * HEAD_DIM} and item["k"] == 0 and item["flags"] == 45):
                errors.append(f"bad RoPE descriptor at {item['ordinal']}")
        elif opcode == opcodes["kv_write"]:
            if not (layer < 24 and item["m"] == 1 and item["n"] == KV_HEADS * HEAD_DIM and item["k"] == 0):
                errors.append(f"bad KV descriptor at {item['ordinal']}")
            kv_positions[layer].add(item["sequence_position"])
        elif opcode in {opcodes["attn_score"], opcodes["attn_value"]}:
            if not (layer < 24 and item["m"] == 1 and 1 <= item["n"] <= CONTEXT_TILE and item["k"] == HEAD_DIM):
                errors.append(f"bad attention descriptor at {item['ordinal']}")
        elif opcode == opcodes["softmax"]:
            if not (layer < 24 and item["m"] == 1 and 1 <= item["n"] <= CONTEXT_TILE and item["k"] == 0):
                errors.append(f"bad softmax descriptor at {item['ordinal']}")
        elif opcode == opcodes["attn_compose"]:
            if not (layer < 24 and item["m"] == 1 and item["n"] == 1 and item["k"] == HEAD_DIM and item["flags"] in {0, 1, 2, 3, 4, 6}):
                errors.append(f"bad attention-compose descriptor at {item['ordinal']}")
        elif opcode in {opcodes["residual_add"], opcodes["silu_gate"]}:
            pass
        else:
            errors.append(f"unknown opcode at {item['ordinal']}")
        if layer < 24:
            layer_ops[layer].append(item["operator"])
    require(not errors, "; ".join(errors[:8]))
    require(all(positions == {0, 1} for positions in kv_positions.values()), "cross-token per-layer KV updates are incomplete")
    require(all(any(op == "mlp_residual_add" for op in ops) for ops in layer_ops.values()), "not all layers reach MLP residual")
    for layer in range(LAYERS):
        for head in range(HEADS):
            score_tokens = {
                item["context_token"]
                for item in commands
                if item["token_step"] == 1
                and item["layer_id"] == layer
                and item["query_head"] == head
                and item["operator"] == "attention_score"
            }
            compose_phases = [
                item["flags"]
                for item in commands
                if item["token_step"] == 1
                and item["layer_id"] == layer
                and item["query_head"] == head
                and item["operator"] == "attention_compose"
            ]
            require(score_tokens == {0, 1}, f"layer {layer} head {head} lacks exact cross-token score reads")
            require(compose_phases == [0, 1, 2, 3, 4, 6], f"layer {layer} head {head} compose phases differ")
    return {
        "status": "PASS",
        "validated_commands": len(commands),
        "layers": LAYERS,
        "kv_positions_per_layer": [0, 1],
        "lm_head_tiles_per_token": math.ceil(VOCAB / LM_TILE),
        "first_command": commands[0]["operator"],
        "last_command": commands[-1]["operator"],
    }


def collect_rtl_execution(out: Path) -> dict[str, Any]:
    specs = {
        "compile": (out / "rtl/compile.log", None),
        "high_layer_sweep": (out / "rtl/layer_sweep.log", "ACE2_SHELL_LAYER_SWEEP_TB_PASS"),
        "attention_compose": (out / "rtl/attention_compose.log", "ACE2_SHELL_ATTN_COMPOSE_ONLY_TB_PASS"),
        "final_rmsnorm": (out / "rtl/final_rmsnorm.log", "ACE2_SHELL_FINAL_RMSNORM_TB_PASS"),
        "lm_head_tile": (out / "rtl/lm_head.log", "ACE2_SHELL_LM_HEAD_TB_PASS"),
    }
    records: dict[str, Any] = {}
    for name, (path, marker) in specs.items():
        if not path.is_file():
            records[name] = {"status": "not_run", "path": str(path.relative_to(ROOT))}
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        status = "PASS" if marker is None or marker in text else "FAIL"
        records[name] = {
            "status": status,
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "pass_marker": marker,
            "last_line": text.strip().splitlines()[-1] if text.strip() else "",
        }
    require(all(record["status"] == "PASS" for record in records.values()), "fresh focused RTL execution is incomplete")
    return {
        "schema_version": 1,
        "classification": "fresh_current_rtl_focused_surface_execution_not_end_to_end_decoder",
        "records": records,
    }


def boundary_report(
    inventory: dict[str, Any],
    surface_inventory: dict[str, Any],
    schedule: dict[str, Any],
    rtl_execution: dict[str, Any],
) -> dict[str, Any]:
    runtime_candidates = surface_inventory["maintained_runtime_candidates"]
    return {
        "schema_version": 1,
        "status": "STOPPED_AT_FIRST_GENUINE_INTEGRATION_BOUNDARY",
        "maximal_ordered_path_prepared": {
            "tokens": 2,
            "layers_per_token": LAYERS,
            "commands": schedule["command_count"],
            "final_rmsnorm_per_token": 1,
            "lm_head_tiles_per_token": math.ceil(VOCAB / LM_TILE),
            "cross_token_kv_positions": [0, 1],
        },
        "fresh_rtl_surface_execution": rtl_execution,
        "first_boundary": {
            "id": "full_model_w4_metadata_image_missing",
            "category": "implementation",
            "description": (
                "The live shell has descriptor surfaces for the transformer operators, KV writes, final RMSNorm, and one 32-logit lm-head tile, and this package now supplies the complete ordered 24-layer, per-head, per-tile, cross-token descriptor schedule. The first executable-data boundary is that the repository has no maintained full-model packed-W4/metadata memory image at the prepared addresses; existing generated packed weights are limited to Layer-0/operator evidence."
            ),
            "address_map_prepared": True,
            "packed_w4_bytes_required": inventory["image_layout"]["packed_w4"]["bytes"],
            "projection_metadata_bytes_required": inventory["image_layout"]["projection_metadata"]["bytes"],
            "rmsnorm_metadata_bytes_required": inventory["image_layout"]["rmsnorm_metadata"]["bytes"],
            "operator_aux_metadata_bytes_required": inventory["image_layout"]["operator_aux_metadata"]["bytes"],
            "runtime_candidate_paths": runtime_candidates,
            "later_known_gap_not_yet_reached": "No executable host/runtime submitter consumes command_schedule.json after the model image exists.",
            "synthesis_impacting_rtl_change_required_now": False,
        },
        "rtl_change_policy": "No RTL was changed. Operator approval remains required before any synthesis-impacting controller, embedding, token-selection, or generation-loop RTL is introduced.",
        "not_claimed": [
            "No complete decoder numerical output was produced.",
            "No model forward, capture, or calibration was run by this integration mission; its frozen numerical metadata came from the bound upstream smoke run.",
            "No full-model W4 binary image was generated.",
            "No end-to-end RTL autoregressive token was executed.",
            "No synthesis, STA, PPA, or publication artifact was changed.",
        ],
    }


def write_checksums(out: Path, paths: list[Path]) -> None:
    rows = [f"{sha256_file(path)}  {path.relative_to(out).as_posix()}" for path in sorted(paths)]
    (out / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--generated-at-utc",
        help="Explicit YYYY-MM-DDTHH:MM:SSZ provenance timestamp for byte-stable regeneration",
    )
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    generated_at_utc = args.generated_at_utc or utc_now()
    require(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", generated_at_utc) is not None,
        "--generated-at-utc must use YYYY-MM-DDTHH:MM:SSZ",
    )

    for path in (
        MISSION,
        CHECKPOINT,
        CONFIG,
        MODEL,
        GENERATION_CONFIG,
        SCALES,
        UPSTREAM_RUN_CONTRACT,
        UPSTREAM_RESULTS,
        PKG,
        SHELL,
        SHELL_TB,
    ):
        require(path.is_file(), f"missing required input: {path}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION_CONFIG.read_text(encoding="utf-8"))
    scales = json.loads(SCALES.read_text(encoding="utf-8"))
    upstream_run_contract = json.loads(UPSTREAM_RUN_CONTRACT.read_text(encoding="utf-8"))
    upstream_results = json.loads(UPSTREAM_RESULTS.read_text(encoding="utf-8"))
    scale_sha256 = sha256_file(SCALES)
    run_contract_sha256 = sha256_file(UPSTREAM_RUN_CONTRACT)
    require(
        upstream_results["artifacts"]["derived_scales_sha256"] == scale_sha256,
        "upstream results do not bind the frozen derived scales",
    )
    require(
        upstream_results["artifacts"]["run_contract_sha256"] == run_contract_sha256,
        "upstream results do not bind the retained run contract",
    )
    require(upstream_run_contract["mode"] == "smoke", "upstream metadata was not produced by the retained smoke run")
    require(upstream_results["mode"] == "smoke", "upstream results mode differs from the retained smoke run")
    require(
        upstream_run_contract["model"]["revision"] == REVISION
        and upstream_results["model"]["resolved_revision"] == REVISION,
        "upstream metadata model revision differs from the frozen model snapshot",
    )
    opcodes = parse_opcodes()
    inventory, layout = build_model_inventory(config, scales)
    surfaces = build_surface_inventory(opcodes)
    schedule = build_schedule(opcodes, layout, inventory)
    schedule_validation = validate_schedule(schedule, opcodes)
    seed_token = int(generation.get("bos_token_id", config["bos_token_id"]))
    layer0_input_scale = float(scales["operators"]["model.layers.0.input_layernorm.input"]["scale"])
    witness = raw_lm_head_witness(seed_token, layer0_input_scale)
    schedule["host_steps"][2]["token_id"] = witness["stable_argmax_token_id"]
    rtl_execution = collect_rtl_execution(out)
    boundary = boundary_report(inventory, surfaces, schedule, rtl_execution)
    activation_quantization = upstream_run_contract["numerical_contract"]["activation_quantization"]
    provenance = {
        "schema_version": 2,
        "generated_at_utc": generated_at_utc,
        "mission_id": MISSION_ID,
        "classification": "zero_model_forward_in_this_mission_with_upstream_calibration_derived_metadata",
        "claim_scope": {
            "no_forward_in_this_integration_mission": True,
            "forward_independent_inputs": False,
            "statement": (
                "This integration mission ran no model forward, capture, or calibration. "
                "The frozen projection and operator metadata was produced by the bound upstream "
                "smoke calibration/model-execution run and is not forward-independent."
            ),
        },
        "inputs": {
            str(path): file_record(path)
            for path in (
                MISSION,
                CONFIG,
                MODEL,
                GENERATION_CONFIG,
                SCALES,
                UPSTREAM_RUN_CONTRACT,
                UPSTREAM_RESULTS,
                PKG,
                SHELL,
                SHELL_TB,
                Path(__file__),
            )
        },
        "frozen_numerical_metadata": {
            "classification": "upstream_calibration_and_model_execution_derived",
            "consumption": "derived_scales_consumed_byte_for_byte_without_regeneration",
            "derived_scales": {
                "path": str(SCALES.relative_to(ROOT)),
                **file_record(SCALES),
            },
            "upstream_run_contract": {
                "path": str(UPSTREAM_RUN_CONTRACT.relative_to(ROOT)),
                **file_record(UPSTREAM_RUN_CONTRACT),
            },
            "upstream_results": {
                "path": str(UPSTREAM_RESULTS.relative_to(ROOT)),
                **file_record(UPSTREAM_RESULTS),
            },
            "upstream_execution": {
                "created_at_utc": upstream_run_contract["created_at_utc"],
                "mode": upstream_run_contract["mode"],
                "command": upstream_run_contract["command"],
                "model": upstream_run_contract["model"],
                "calibration_dataset": activation_quantization["calibration_dataset"],
                "activation_scale_derivation": activation_quantization["scale"],
                "calibration_and_model_execution_occurred_upstream": True,
            },
        },
        "mission_local_guards": {
            "transformers_imported": False,
            "model_instantiated": False,
            "model_forward_calls": 0,
            "capture_hooks": 0,
            "calibration_runs": 0,
            "subprocess_model_execution": 0,
        },
        "determinism": {
            "generated_at_utc_source": "explicit_cli" if args.generated_at_utc else "wall_clock_default",
            "frozen_input_hashes_required": True,
        },
    }

    outputs = {
        "integration_inventory.json": inventory,
        "maintained_surfaces.json": surfaces,
        "command_schedule.json": schedule,
        "command_schedule_validation.json": schedule_validation,
        "raw_tied_embedding_lm_head_witness.json": witness,
        "rtl_execution.json": rtl_execution,
        "boundary_report.json": boundary,
        "provenance.json": provenance,
    }
    written = []
    for name, value in outputs.items():
        path = out / name
        write_json(path, value)
        written.append(path)
    written.extend(path for path in (out / "rtl").glob("*.log") if path.is_file())
    write_checksums(out, written)
    print(
        "ACE2_FULL_QWEN_INTEGRATION_PREP_PASS "
        f"layers={LAYERS} tokens=2 commands={schedule['command_count']} "
        f"lm_tiles={math.ceil(VOCAB / LM_TILE)} boundary={boundary['first_boundary']['id']}"
    )


if __name__ == "__main__":
    main()
