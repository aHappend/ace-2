#!/usr/bin/env python3
"""Generate ACE-2 Phase-2 model hardware contracts and binding inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import model_hardware_contract as base_contract  # noqa: E402


SCHEMA_VERSION = 2
PRECISION_MODE = base_contract.PRECISION_MODE
SCHEMA_PATH = ROOT / "design/model_hardware_contract_phase2.schema.json"
DESCRIPTOR_DIR = ROOT / "design/model_hardware_contracts/phase2"
INVENTORY_PATH = DESCRIPTOR_DIR / "HARD_CODED_0P5B_INVENTORY.json"
MANIFEST_PATH = DESCRIPTOR_DIR / "MANIFEST.json"
PLAN_PATH = ROOT / "design/MODEL_HARDWARE_PARAMETERIZATION_PHASE2.md"
SV_PARAMETER_PATH = ROOT / "rtl/generated/ace2_model_parameters.svh"
SV_PARAMETER_MODEL_ID = "qwen2.5-0.5b"


class Phase2ContractError(ValueError):
    """Raised when a Phase-2 hardware contract artifact is inconsistent."""


@dataclass(frozen=True)
class ProductConfig:
    primary_tier: str
    compatible_tiers: tuple[str, ...]
    compute_tiles: int
    memory_channels: int


PRODUCT_CONFIGS = {
    "qwen2.5-0.5b": ProductConfig("ace2_nano", ("ace2_nano",), 1, 1),
    "qwen2.5-1.5b": ProductConfig(
        "ace2_nano", ("ace2_nano", "ace2_edge"), 2, 2
    ),
    "qwen2.5-3b": ProductConfig("ace2_edge", ("ace2_edge", "ace2_pro"), 4, 4),
    "qwen2.5-7b": ProductConfig("ace2_pro", ("ace2_pro",), 8, 8),
}

CURRENT_0P5B_BINDING = {
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_layers": 24,
    "attention_heads": 14,
    "kv_heads": 2,
    "head_dim": 64,
    "kv_width_elements": 128,
    "vocab_size": 151936,
    "max_context": 32768,
    "compute_tiles": 1,
    "memory_channels": 1,
}
CURRENT_COMMAND_WIDTHS = {
    "layer_id_bits": 8,
    "m_bits": 16,
    "n_bits": 16,
    "k_bits": 16,
    "sequence_position_bits": 16,
    "completion_tag_bits": 16,
    "address_bits": 64,
}
PHASE2_MINIMUM_COMMAND_WIDTHS = {
    **CURRENT_COMMAND_WIDTHS,
    "sequence_position_bits": 17,
}


@dataclass(frozen=True)
class Binding:
    binding_id: str
    surface: str
    path: str
    symbol: str
    contract_fields: tuple[str, ...]
    current_binding: str
    fragment: str
    successor_action: str
    occurrence: int = 1


INVENTORY_SCOPE = {
    "included": (
        "active top-level rtl/*.sv sources",
        "active model image packers and independent image verifiers under tools/",
        "active command-package and chat-runtime generation tools",
        "the Verilated host runtime bridge used by make full-qwen-runtime-build",
    ),
    "excluded": (
        "historical build/, evidence/, reference/, runtime authority packages",
        "unit tests and one-off diagnosis scripts that are not packer, command generation, host runtime, or public RTL surfaces",
        "numeric literals that encode protocol widths, fixed-point widths, opcodes, addresses, hashes, or token IDs rather than Qwen model dimensions",
    ),
}

INVENTORY_BINDINGS = (
    Binding(
        "rtl.pkg.hidden_size",
        "rtl",
        "rtl/generated/ace2_model_parameters.svh",
        "ACE2_HIDDEN_SIZE",
        ("hidden_size",),
        "896",
        "localparam integer ACE2_HIDDEN_SIZE = 896;",
        "Generated from the selected descriptor with 896 retained as the 0.5B value.",
    ),
    Binding(
        "rtl.pkg.vocab_size",
        "rtl",
        "rtl/generated/ace2_model_parameters.svh",
        "ACE2_VOCAB_SIZE",
        ("vocab_size", "lm_head_tile_count"),
        "151936",
        "localparam integer ACE2_VOCAB_SIZE = 151936;",
        "Generated from the selected descriptor and used to derive the LM-head tile count.",
    ),
    Binding(
        "rtl.shell.intermediate_size",
        "rtl",
        "rtl/ace2_shell.sv",
        "MLP_INTERMEDIATE_SIZE",
        ("intermediate_size",),
        "4864",
        "localparam integer MLP_INTERMEDIATE_SIZE = ACE2_INTERMEDIATE_SIZE;",
        "Resolved through the descriptor-generated package parameter used by projection and SiLU loops.",
    ),
    Binding(
        "rtl.shell.kv_width",
        "rtl",
        "rtl/ace2_shell.sv",
        "QKV_KV_OUTPUTS",
        ("kv_heads", "head_dim", "kv_width_elements", "weight_tiling"),
        "128",
        "localparam integer QKV_KV_OUTPUTS = ACE2_KV_WIDTH;",
        "Resolved through the descriptor-generated K/V width used by QKV offsets and output spans.",
    ),
    Binding(
        "rtl.shell.head_dim",
        "rtl",
        "rtl/ace2_shell.sv",
        "ATTN_HEAD_DIM",
        ("head_dim", "attention_heads", "gqa_mapping"),
        "64",
        "localparam integer ATTN_HEAD_DIM = ACE2_HEAD_DIM;",
        "Resolved through the descriptor-generated head dimension used by attention loops.",
    ),
    Binding(
        "rtl.shell.max_context",
        "rtl",
        "rtl/ace2_shell.sv",
        "ROPE_MAX_SEQUENCE_POSITION",
        ("max_context", "sequence_position_bits"),
        "32767",
        "localparam [15:0] ROPE_MAX_SEQUENCE_POSITION = 16'(ACE2_MAX_CONTEXT - 1);",
        "Resolved for the 0.5B public 16-bit command interface through the descriptor-generated context bound.",
    ),
    Binding(
        "rtl.core.rmsnorm.hidden_default",
        "rtl",
        "rtl/ace2_rmsnorm_core.sv",
        "HIDDEN_SIZE parameter default",
        ("hidden_size",),
        "896",
        "parameter integer HIDDEN_SIZE = 896,",
        "Drive the default from the selected model contract while preserving override support.",
    ),
    Binding(
        "rtl.core.projection.k_default",
        "rtl",
        "rtl/ace2_w4a8_proj_core.sv",
        "K_SIZE parameter default",
        ("hidden_size", "intermediate_size"),
        "896",
        "parameter integer K_SIZE = 896,",
        "Instantiate projection cores with descriptor-specific reduction widths.",
    ),
    Binding(
        "rtl.core.attention_score.head_dim_default",
        "rtl",
        "rtl/ace2_attention_score_core.sv",
        "HEAD_DIM parameter default",
        ("head_dim",),
        "64",
        "parameter integer HEAD_DIM = 64,",
        "Instantiate score cores with descriptor head_dim.",
    ),
    Binding(
        "rtl.core.attention_compose.head_dim_default",
        "rtl",
        "rtl/ace2_attention_compose_core.sv",
        "HEAD_DIM parameter default",
        ("head_dim",),
        "64",
        "parameter integer HEAD_DIM = 64,",
        "Bind compose accumulation width to descriptor head_dim.",
    ),
    Binding(
        "rtl.core.attention_compose.context_default",
        "rtl",
        "rtl/ace2_attention_compose_core.sv",
        "CONTEXT_MAX parameter default",
        ("max_context",),
        "32768",
        "parameter integer CONTEXT_MAX = 32768,",
        "Bind compose context checks to descriptor max_context.",
    ),
    Binding(
        "rtl.core.qk_residual_cross_term.head_dim_default",
        "rtl",
        "rtl/ace2_qk_residual_cross_term_core.sv",
        "HEAD_DIM parameter default",
        ("head_dim",),
        "64",
        "parameter integer HEAD_DIM = 64",
        "Bind residual cross-term score helpers to descriptor head_dim.",
    ),
    Binding(
        "rtl.core.v_residual_value_correction.context_default",
        "rtl",
        "rtl/ace2_v_residual_value_correction_core.sv",
        "CONTEXT_MAX parameter default",
        ("max_context",),
        "32768",
        "parameter integer CONTEXT_MAX = 32768",
        "Bind value-correction helper context storage to descriptor max_context.",
    ),
    Binding(
        "rtl.core.absolute_rope_online_attention.lane_count_default",
        "rtl",
        "rtl/ace2_absolute_rope_online_attention_core.sv",
        "LANE_COUNT parameter default",
        ("head_dim",),
        "64",
        "parameter integer LANE_COUNT = 64",
        "Bind absolute-RoPE online attention helpers to descriptor head_dim.",
    ),
    Binding(
        "rtl.core.native_accumulator_tagged_attention.max_lanes_default",
        "rtl",
        "rtl/ace2_native_accumulator_tagged_attention_core.sv",
        "MAX_LANES parameter default",
        ("head_dim",),
        "64",
        "parameter integer MAX_LANES = 64",
        "Bind tagged attention helper lane capacity to descriptor head_dim.",
    ),
    Binding(
        "rtl.core.tile_bfp_score_attention.max_keys_default",
        "rtl",
        "rtl/ace2_tile_bfp_score_attention_core.sv",
        "MAX_KEYS parameter default",
        ("head_dim",),
        "64",
        "parameter integer MAX_KEYS = 64",
        "Bind tile-BFP score helper capacity to descriptor head_dim or retire it before multi-size RTL acceptance.",
    ),
    Binding(
        "rtl.core.tile_max_delta_attention.max_keys_default",
        "rtl",
        "rtl/ace2_tile_max_delta_attention_core.sv",
        "MAX_KEYS parameter default",
        ("head_dim",),
        "64",
        "parameter integer MAX_KEYS = 64",
        "Bind tile-max-delta attention helper capacity to descriptor head_dim or retire it before multi-size RTL acceptance.",
    ),
    Binding(
        "rtl.core.error_carry.hidden_default",
        "rtl",
        "rtl/ace2_cross_layer_error_carry_core.sv",
        "ace2_error_carry_state_core HIDDEN_SIZE default",
        ("hidden_size",),
        "896",
        "parameter integer HIDDEN_SIZE = 896,",
        "Make checked-in successor/helper RTL defaults descriptor-bound before admitting multi-size checks.",
    ),
    Binding(
        "rtl.core.carry_aware_rmsnorm.hidden_default",
        "rtl",
        "rtl/ace2_cross_layer_error_carry_core.sv",
        "ace2_carry_aware_rmsnorm_core HIDDEN_SIZE default",
        ("hidden_size",),
        "896",
        "parameter integer HIDDEN_SIZE = 896",
        "Make carry-aware RMSNorm helper defaults descriptor-bound before admitting multi-size checks.",
        occurrence=2,
    ),
    Binding(
        "rtl.helper.layer16_runtime_metadata.hidden_default",
        "rtl",
        "rtl/ace2_layer16_post_attention_rmsnorm_runtime_metadata_core.sv",
        "HIDDEN_SIZE parameter default",
        ("hidden_size",),
        "896",
        "parameter integer HIDDEN_SIZE = 896,",
        "Retire or descriptor-parameterize layer-specific runtime-metadata helper cores before multi-size RTL acceptance.",
    ),
    Binding(
        "rtl.helper.layer16_group1.hidden_default",
        "rtl",
        "rtl/ace2_layer16_group1_scale32_core.sv",
        "HIDDEN_SIZE parameter default",
        ("hidden_size",),
        "896",
        "parameter integer HIDDEN_SIZE = 896,",
        "Retire or descriptor-parameterize layer-specific grouped-Scale32 helper cores before multi-size RTL acceptance.",
    ),
    Binding(
        "rtl.helper.layer16_group4.hidden_default",
        "rtl",
        "rtl/ace2_layer16_group4_scale32_core.sv",
        "HIDDEN_SIZE parameter default",
        ("hidden_size",),
        "896",
        "parameter integer HIDDEN_SIZE = 896,",
        "Retire or descriptor-parameterize layer-specific grouped-Scale32 helper cores before multi-size RTL acceptance.",
    ),
    Binding(
        "rtl.helper.layer17_rmsnorm_output.hidden_default",
        "rtl",
        "rtl/ace2_layer17_rmsnorm_output_grouped_scale32_core.sv",
        "HIDDEN_SIZE parameter default",
        ("hidden_size",),
        "896",
        "parameter integer HIDDEN_SIZE = 896,",
        "Retire or descriptor-parameterize layer-specific grouped-Scale32 helper cores before multi-size RTL acceptance.",
    ),
    Binding(
        "packer.base.model_identity",
        "model_packer",
        "tools/build_full_qwen_image_v2.py",
        "Qwen2.5-0.5B source and core dimensions",
        ("num_layers", "hidden_size", "head_dim", "kv_heads"),
        "Qwen/Qwen2.5-0.5B, hidden=896, head_dim=64, kv_heads=2, layers=24",
        '    / "models--Qwen--Qwen2.5-0.5B"',
        "Load source identity and dimensions from the Phase-2 descriptor rather than a single pinned base model path.",
    ),
    Binding(
        "packer.base.region_bytes",
        "model_packer",
        "tools/build_full_qwen_image_v2.py",
        "EXPECTED_REGION_BYTES",
        ("weight_tiling", "rmsnorm_record_bytes", "operator_aux_metadata_bytes"),
        "0.5B region byte totals",
        '    "packed_w4": 246_980_608,',
        "Compute packed-W4, projection-metadata, RMSNorm, and operator-aux regions from descriptor-derived rows and record sizes.",
    ),
    Binding(
        "packer.verify.model_identity",
        "model_packer",
        "tools/verify_full_qwen_image_v2.py",
        "independent verifier source and dimensions",
        ("num_layers", "hidden_size", "head_dim", "kv_heads"),
        "Qwen/Qwen2.5-0.5B, hidden=896, head_dim=64, kv_heads=2, layers=24",
        '    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots"',
        "Mirror descriptor-derived source and geometry in the independent packer verifier.",
    ),
    Binding(
        "packer.audit.rmsnorm_layout",
        "model_packer",
        "tools/audit_full_qwen_image_contract.py",
        "RMSNorm/gain and accepted command-count audit",
        ("hidden_size", "num_layers", "weight_tiling"),
        "896 gains, 13,914 accepted two-token commands",
        "gains_per_record = 896",
        "Replace frozen 0.5B audit constants with descriptor-derived structural checks for new packages.",
    ),
    Binding(
        "command.accepted_runtime.model_path",
        "command_generation",
        "tools/run_full_qwen_command_schedule_runtime.py",
        "accepted runtime model path and package dimensions",
        ("hidden_size", "vocab_size", "num_layers", "lm_head_tile_count"),
        "Qwen2.5-0.5B, [151936, 896], layers=24, tiles=4748",
        '    / "models--Qwen--Qwen2.5-0.5B"',
        "Make schedule/package construction consume the descriptor and emit model_id plus contract hash.",
    ),
    Binding(
        "command.chat.constants",
        "command_generation",
        "tools/ace2_chat_demo.py",
        "ACE2RT2 package and fused-QKV constants",
        ("hidden_size", "max_context", "kv_width_elements", "lm_head_tile_count"),
        "hidden=896, max_context=32768, kv_bytes=272, LM tiles=4748",
        "KV_BYTES_PER_TOKEN = 272",
        "Derive ACE2RT package header fields, KV stride, LM-head tile count, and fused-QKV spans from the contract.",
    ),
    Binding(
        "command.chat.fused_qkv_offsets",
        "command_generation",
        "tools/ace2_chat_demo.py",
        "FUSED_QKV offsets/strides",
        ("hidden_size", "kv_width_elements", "weight_tiling"),
        "0.5B Q/K/V byte offsets and 0x71c000 layer stride",
        "FUSED_QKV_WEIGHT_OFFSETS = (0, 401_408, 458_752)",
        "Compute Q/K/V weight, metadata, and output offsets from hidden_size and kv_width_elements.",
    ),
    Binding(
        "command.fused_prefix.preflight",
        "command_generation",
        "tools/run_fused_qkv_runtime_prefix.py",
        "fused-prefix runtime_preflight geometry",
        ("hidden_size", "vocab_size", "max_context", "kv_bytes_per_token_per_layer"),
        "[151936, 896], max_context=32768, kv_stride=272",
        '        model_id="qwen2.5-0.5b",',
        "Turn the prefix runner into a descriptor-selected structural/operator check after RTL parameterization.",
    ),
    Binding(
        "host.cpp.runtime_constants",
        "host_runtime",
        "verification/verilator/ace2_shell_runtime_main.cpp",
        "ACE2RT2 C++ geometry constants",
        ("num_layers", "hidden_size", "vocab_size", "max_context"),
        "layers=24, hidden=896, vocab=151936, max_context=32768",
        "constexpr std::uint32_t kHidden = 896;",
        "Parse geometry from the package contract hash instead of compiling one C++ runtime geometry.",
    ),
    Binding(
        "host.cpp.schedule_loops",
        "host_runtime",
        "verification/verilator/ace2_shell_runtime_main.cpp",
        "C++ schedule loops and fused-QKV legality",
        ("attention_heads", "hidden_size", "weight_tiling", "lm_head_tile_count"),
        "24 layers, 14 heads, fused n/k=896, fixed spans",
        "for (std::uint8_t layer = 0; layer < 24; ++layer) {",
        "Replace schedule validation loops and fused-QKV range checks with descriptor-derived values.",
    ),
    Binding(
        "host.identity_profile.embedding_offset",
        "host_runtime",
        "verification/verilator/ace2_runtime_identity_profile.h",
        "runtime identity profile embedding offset",
        ("weight_layout", "embedding_storage"),
        "0.5B image/model hashes and embedding_offset=32288",
        "std::uint64_t embedding_offset;",
        "Add model_id and descriptor hash to identity matching so offsets are per generated image contract.",
    ),
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase2ContractError(message)


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _positive_integer() -> dict[str, Any]:
    return {"type": "integer", "minimum": 1}


def _string_array(values: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 1,
        "items": {"enum": list(values)},
        "uniqueItems": True,
    }


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": (
        "https://github.com/Argus-AiTeam/ace-2/"
        "design/model_hardware_contract_phase2.schema.json"
    ),
    "title": "ACE-2 Phase-2 Qwen2.5 Unified Model Hardware Contract",
    **_strict_object(
        {
            "schema_version": {"const": SCHEMA_VERSION},
            "contract_id": {"type": "string", "minLength": 1},
            "model_id": {"enum": [config.model_id for config in base_contract.QWEN_CONFIGS]},
            "source": _strict_object(
                {
                    "repository": {"type": "string", "minLength": 1},
                    "revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                    "config_path": {"const": "config.json"},
                    "config_url": {"type": "string", "format": "uri"},
                }
            ),
            "product": _strict_object(
                {
                    "primary_tier": {"enum": ["ace2_nano", "ace2_edge", "ace2_pro"]},
                    "compatible_tiers": _string_array(
                        ("ace2_nano", "ace2_edge", "ace2_pro")
                    ),
                    "compute_tiles": _positive_integer(),
                    "memory_channels": _positive_integer(),
                    "claim_boundary": {"type": "string", "minLength": 1},
                }
            ),
            "dimensions": _strict_object(
                {
                    "num_layers": _positive_integer(),
                    "hidden_size": _positive_integer(),
                    "intermediate_size": _positive_integer(),
                    "attention_heads": _positive_integer(),
                    "kv_heads": _positive_integer(),
                    "head_dim": _positive_integer(),
                    "vocab_size": _positive_integer(),
                    "max_context": _positive_integer(),
                    "tie_word_embeddings": {"type": "boolean"},
                }
            ),
            "formats": _strict_object(
                {
                    "weight_format": {"const": "signed_int4_packed_w4"},
                    "weight_input_group_size": {"const": 32},
                    "packed_weight_group_bytes": {"const": 16},
                    "weight_scale_granularity": {
                        "const": "per_output_channel_per_input_group"
                    },
                    "weight_scale_format": {"const": "scale32_little_endian_u32"},
                    "weight_scale_record_bytes": {"const": 4},
                    "activation_format": {"const": "signed_int8"},
                    "accumulator_format": {"const": "signed_int32_dot_product"},
                    "kv_format": {"const": "signed_int8_payload_scale32_per_kv_head"},
                    "scale_format": {"const": "scale32"},
                    "embedding_storage": {"const": "bf16"},
                }
            ),
            "hardware_interface": _strict_object(
                {
                    "address_bits": {"const": 64},
                    "data_width_bits": {"const": 128},
                    "lm_head_tile_size": {"const": 32},
                    "projection_tile_m": {"const": 1},
                    "projection_tile_n": {"const": 32},
                    "projection_tile_k": {"const": 32},
                    "vector_lanes": {"const": 16},
                    "required_command_widths": _strict_object(
                        {
                            "layer_id_bits": _positive_integer(),
                            "m_bits": _positive_integer(),
                            "n_bits": _positive_integer(),
                            "k_bits": _positive_integer(),
                            "sequence_position_bits": _positive_integer(),
                            "completion_tag_bits": _positive_integer(),
                            "address_bits": _positive_integer(),
                        }
                    ),
                    "current_shell_command_widths": _strict_object(
                        {
                            "layer_id_bits": {"const": 8},
                            "m_bits": {"const": 16},
                            "n_bits": {"const": 16},
                            "k_bits": {"const": 16},
                            "sequence_position_bits": {"const": 16},
                            "completion_tag_bits": {"const": 16},
                            "address_bits": {"const": 64},
                        }
                    ),
                }
            ),
            "derived": {"type": "object"},
            "static_capacity_checks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "pass", "evidence"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "pass": {"type": "boolean"},
                        "evidence": {"type": "string", "minLength": 1},
                    },
                },
            },
            "current_parameterization_gaps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "current", "required", "blocking_surfaces"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "current": {"type": ["integer", "string"]},
                        "required": {"type": ["integer", "string"]},
                        "blocking_surfaces": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "compatibility": _strict_object(
                {
                    "existing_0p5b_runtime_preflight": {"type": "boolean"},
                    "full_model_rtl_execution_validated": {"const": False},
                    "larger_model_rtl_execution_validated": {"const": False},
                    "synthesis_timing_or_u280_validated": {"const": False},
                }
            ),
            "claim_boundary": {"type": "string", "minLength": 1},
        }
    ),
}


def descriptor_path(model_id: str) -> Path:
    if model_id not in base_contract.CONFIG_BY_ID:
        raise Phase2ContractError(f"unsupported model_id: {model_id}")
    return DESCRIPTOR_DIR / f"{model_id}.json"


def sv_parameter_values(descriptor: Mapping[str, Any]) -> dict[str, int]:
    dimensions = descriptor["dimensions"]
    derived = descriptor["derived"]
    hardware_interface = descriptor["hardware_interface"]
    values = {
        "ACE2_NUM_LAYERS": int(dimensions["num_layers"]),
        "ACE2_HIDDEN_SIZE": int(dimensions["hidden_size"]),
        "ACE2_INTERMEDIATE_SIZE": int(dimensions["intermediate_size"]),
        "ACE2_ATTENTION_HEADS": int(dimensions["attention_heads"]),
        "ACE2_KV_HEADS": int(dimensions["kv_heads"]),
        "ACE2_HEAD_DIM": int(dimensions["head_dim"]),
        "ACE2_KV_WIDTH": int(derived["kv_width_elements"]),
        "ACE2_VOCAB_SIZE": int(dimensions["vocab_size"]),
        "ACE2_MAX_CONTEXT": int(dimensions["max_context"]),
        "ACE2_LM_HEAD_TILE_SIZE": int(hardware_interface["lm_head_tile_size"]),
        "ACE2_VECTOR_LANES": int(hardware_interface["vector_lanes"]),
    }
    require(
        values["ACE2_KV_WIDTH"]
        == values["ACE2_KV_HEADS"] * values["ACE2_HEAD_DIM"],
        "SV parameter K/V width is inconsistent",
    )
    require(
        values["ACE2_HIDDEN_SIZE"]
        == values["ACE2_ATTENTION_HEADS"] * values["ACE2_HEAD_DIM"],
        "SV parameter attention geometry is inconsistent",
    )
    return values


def build_sv_parameter_include(descriptor: Mapping[str, Any]) -> bytes:
    model_id = str(descriptor["model_id"])
    require(
        model_id == SV_PARAMETER_MODEL_ID,
        f"unsupported SV parameter model: {model_id}",
    )
    descriptor_sha256 = sha256_bytes(canonical_bytes(descriptor))
    lines = [
        "// Generated by tools/model_hardware_contract_phase2.py --write.",
        f"// Selected model: {model_id}",
        f"// Descriptor SHA-256: {descriptor_sha256}",
        (
            "localparam [255:0] ACE2_MODEL_DESCRIPTOR_SHA256 = "
            f"256'h{descriptor_sha256};"
        ),
    ]
    lines.extend(
        f"localparam integer {name} = {value};"
        for name, value in sv_parameter_values(descriptor).items()
    )
    return ("\n".join(lines) + "\n").encode()


def bits_for_count(count: int) -> int:
    require(count > 0, "bit-width count must be positive")
    return max(1, (count - 1).bit_length())


def qwen_source(config: base_contract.QwenConfig) -> dict[str, Any]:
    return {
        "repository": config.repository,
        "revision": config.revision,
        "config_path": "config.json",
        "config_url": (
            f"https://huggingface.co/{config.repository}/"
            f"blob/{config.revision}/config.json"
        ),
    }


def product_config(model_id: str) -> ProductConfig:
    try:
        return PRODUCT_CONFIGS[model_id]
    except KeyError as error:
        raise Phase2ContractError(f"missing product config for {model_id}") from error


def derive_phase2(config: base_contract.QwenConfig) -> dict[str, int]:
    base = base_contract.derive(config)
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    head_dim = base["head_dim"]
    kv_width = base["kv_width_elements"]
    projection_tile_n = 32
    lm_head_tile = 32
    q_weight_bytes = hidden * hidden // 2
    kv_weight_bytes = kv_width * hidden // 2
    group_size = base_contract.WEIGHT_LAYOUT["input_group_size"]
    scale_record_bytes = base_contract.WEIGHT_LAYOUT["scale_record_bytes"]
    q_scale_records = hidden * (hidden // group_size)
    kv_scale_records = kv_width * (hidden // group_size)
    q_metadata_bytes = q_scale_records * scale_record_bytes
    kv_metadata_bytes = kv_scale_records * scale_record_bytes
    per_layer_projection_rows = (
        3 * hidden + 2 * kv_width + 2 * intermediate
    )
    per_layer_linear_elements = (
        2 * hidden * hidden
        + 2 * hidden * kv_width
        + 3 * hidden * intermediate
    )
    per_layer_packed_w4_bytes = per_layer_linear_elements // 2
    lm_head_tile_count = config.vocab_size // lm_head_tile
    return {
        **base,
        "num_layers": config.num_hidden_layers,
        "max_context": config.max_position_embeddings,
        "per_layer_linear_weight_elements": per_layer_linear_elements,
        "per_layer_packed_w4_bytes": per_layer_packed_w4_bytes,
        "per_layer_projection_rows": per_layer_projection_rows,
        "per_layer_projection_scale_record_count": (
            per_layer_linear_elements // group_size
        ),
        "per_layer_projection_metadata_bytes": (
            per_layer_linear_elements // group_size * scale_record_bytes
        ),
        "qkv_q_weight_bytes": q_weight_bytes,
        "qkv_k_weight_bytes": kv_weight_bytes,
        "qkv_v_weight_bytes": kv_weight_bytes,
        "qkv_weight_span_bytes": q_weight_bytes + 2 * kv_weight_bytes,
        "qkv_q_scale_record_count": q_scale_records,
        "qkv_k_scale_record_count": kv_scale_records,
        "qkv_v_scale_record_count": kv_scale_records,
        "qkv_q_metadata_bytes": q_metadata_bytes,
        "qkv_k_metadata_bytes": kv_metadata_bytes,
        "qkv_v_metadata_bytes": kv_metadata_bytes,
        "qkv_metadata_span_bytes": q_metadata_bytes + 2 * kv_metadata_bytes,
        "qkv_output_span_bytes": hidden + 2 * kv_width,
        "qkv_weight_offsets_q_k_v": (0, q_weight_bytes, q_weight_bytes + kv_weight_bytes),
        "qkv_metadata_offsets_q_k_v": (
            0,
            q_metadata_bytes,
            q_metadata_bytes + kv_metadata_bytes,
        ),
        "qkv_output_offsets_q_k_v": (0, hidden, hidden + kv_width),
        "rmsnorm_record_bytes": hidden * 2 + 16,
        "lm_head_input_elements": hidden,
        "lm_head_output_elements": config.vocab_size,
        "lm_head_tile_size": lm_head_tile,
        "lm_head_tile_count": lm_head_tile_count,
        "required_layer_id_bits": bits_for_count(config.num_hidden_layers + 1),
        "required_sequence_position_bits": bits_for_count(config.max_position_embeddings),
        "required_hidden_index_bits": bits_for_count(hidden),
        "required_intermediate_index_bits": bits_for_count(intermediate),
        "required_vocab_index_bits": bits_for_count(config.vocab_size),
        "required_lm_head_tile_index_bits": bits_for_count(lm_head_tile_count),
        "required_kv_width_index_bits": bits_for_count(kv_width),
    }


def capacity_checks(
    config: base_contract.QwenConfig,
    product: ProductConfig,
    derived: Mapping[str, int],
) -> list[dict[str, Any]]:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    return [
        {
            "id": "hidden-divisible-by-attention-heads",
            "pass": hidden % config.num_attention_heads == 0,
            "evidence": f"{hidden} % {config.num_attention_heads} == 0",
        },
        {
            "id": "attention-heads-divisible-by-kv-heads",
            "pass": config.num_attention_heads % config.num_key_value_heads == 0,
            "evidence": (
                f"{config.num_attention_heads} % {config.num_key_value_heads} == 0"
            ),
        },
        {
            "id": "hidden-aligns-to-vector-and-projection-k",
            "pass": hidden % 16 == 0 and hidden % 32 == 0,
            "evidence": f"hidden_size={hidden}, vector_lanes=16, projection_k=32",
        },
        {
            "id": "intermediate-aligns-to-projection-n",
            "pass": intermediate % 32 == 0,
            "evidence": f"intermediate_size={intermediate}, projection_n=32",
        },
        {
            "id": "vocab-aligns-to-lm-head-tile",
            "pass": config.vocab_size % 32 == 0,
            "evidence": f"vocab_size={config.vocab_size}, lm_head_tile=32",
        },
        {
            "id": "phase2-command-fields-cover-required-shapes",
            "pass": (
                derived["required_layer_id_bits"]
                <= PHASE2_MINIMUM_COMMAND_WIDTHS["layer_id_bits"]
                and max(hidden, intermediate, derived["kv_width_elements"]) < (1 << 16)
                and derived["required_sequence_position_bits"]
                <= PHASE2_MINIMUM_COMMAND_WIDTHS["sequence_position_bits"]
            ),
            "evidence": (
                "required bits layer="
                f"{derived['required_layer_id_bits']}, sequence="
                f"{derived['required_sequence_position_bits']}, max_mnk="
                f"{max(hidden, intermediate, derived['kv_width_elements'])}"
            ),
        },
        {
            "id": "address-and-byte-counts-fit-64-bit-space",
            "pass": derived["maximum_weight_plus_kv_bytes"] < (1 << 64),
            "evidence": (
                "maximum_weight_plus_kv_bytes="
                f"{derived['maximum_weight_plus_kv_bytes']}"
            ),
        },
        {
            "id": "qkv-spans-are-16-byte-aligned",
            "pass": (
                derived["qkv_weight_span_bytes"] % 16 == 0
                and derived["qkv_metadata_span_bytes"] % 16 == 0
                and derived["qkv_output_span_bytes"] % 16 == 0
            ),
            "evidence": (
                f"qkv_weight={derived['qkv_weight_span_bytes']}, "
                f"qkv_metadata={derived['qkv_metadata_span_bytes']}, "
                f"qkv_output={derived['qkv_output_span_bytes']}"
            ),
        },
        {
            "id": "product-compute-and-memory-channels-positive",
            "pass": product.compute_tiles > 0 and product.memory_channels > 0,
            "evidence": (
                f"compute_tiles={product.compute_tiles}, "
                f"memory_channels={product.memory_channels}"
            ),
        },
    ]


def parameterization_gaps(
    config: base_contract.QwenConfig,
    product: ProductConfig,
    derived: Mapping[str, int],
) -> list[dict[str, Any]]:
    required = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_layers": config.num_hidden_layers,
        "attention_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": derived["head_dim"],
        "kv_width_elements": derived["kv_width_elements"],
        "vocab_size": config.vocab_size,
        "max_context": config.max_position_embeddings,
        "compute_tiles": product.compute_tiles,
        "memory_channels": product.memory_channels,
    }
    surface_map = {
        "hidden_size": ("rtl", "model_packer", "command_generation", "host_runtime"),
        "intermediate_size": ("rtl", "model_packer", "command_generation"),
        "num_layers": ("rtl", "model_packer", "command_generation", "host_runtime"),
        "attention_heads": ("rtl", "command_generation", "host_runtime"),
        "kv_heads": ("model_packer", "command_generation"),
        "head_dim": ("rtl", "model_packer", "command_generation", "host_runtime"),
        "kv_width_elements": ("rtl", "model_packer", "command_generation"),
        "vocab_size": ("rtl", "model_packer", "command_generation", "host_runtime"),
        "max_context": ("rtl", "command_generation", "host_runtime"),
        "compute_tiles": ("rtl", "host_runtime"),
        "memory_channels": ("rtl", "model_packer", "host_runtime"),
    }
    gaps: list[dict[str, Any]] = []
    for field, required_value in required.items():
        current_value = CURRENT_0P5B_BINDING[field]
        if current_value == required_value:
            continue
        gaps.append(
            {
                "id": field,
                "current": current_value,
                "required": required_value,
                "blocking_surfaces": list(surface_map[field]),
            }
        )
    if config.model_id != "qwen2.5-0.5b":
        gaps.append(
            {
                "id": "runtime_artifact_identity",
                "current": "sealed_0p5b_ace2rt2_package_and_image",
                "required": f"{config.model_id}_descriptor_hash_bound_package_and_image",
                "blocking_surfaces": [
                    "model_packer",
                    "command_generation",
                    "host_runtime",
                ],
            }
        )
    return gaps


def build_descriptor(config: base_contract.QwenConfig) -> dict[str, Any]:
    product = product_config(config.model_id)
    derived = derive_phase2(config)
    checks = capacity_checks(config, product, derived)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": f"ace2-phase2-{config.model_id}-{PRECISION_MODE}-v1",
        "model_id": config.model_id,
        "source": qwen_source(config),
        "product": {
            "primary_tier": product.primary_tier,
            "compatible_tiers": list(product.compatible_tiers),
            "compute_tiles": product.compute_tiles,
            "memory_channels": product.memory_channels,
            "claim_boundary": (
                "Phase-2 configuration contract only; compute_tiles and "
                "memory_channels are specification parameters, not measured "
                "throughput, timing, synthesis, FPGA, or U280 evidence."
            ),
        },
        "dimensions": {
            "num_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "attention_heads": config.num_attention_heads,
            "kv_heads": config.num_key_value_heads,
            "head_dim": derived["head_dim"],
            "vocab_size": config.vocab_size,
            "max_context": config.max_position_embeddings,
            "tie_word_embeddings": config.tie_word_embeddings,
        },
        "formats": {
            "weight_format": "signed_int4_packed_w4",
            "weight_input_group_size": 32,
            "packed_weight_group_bytes": 16,
            "weight_scale_granularity": "per_output_channel_per_input_group",
            "weight_scale_format": "scale32_little_endian_u32",
            "weight_scale_record_bytes": 4,
            "activation_format": "signed_int8",
            "accumulator_format": "signed_int32_dot_product",
            "kv_format": "signed_int8_payload_scale32_per_kv_head",
            "scale_format": "scale32",
            "embedding_storage": "bf16",
        },
        "hardware_interface": {
            "address_bits": 64,
            "data_width_bits": 128,
            "lm_head_tile_size": 32,
            "projection_tile_m": 1,
            "projection_tile_n": 32,
            "projection_tile_k": 32,
            "vector_lanes": 16,
            "required_command_widths": {
                "layer_id_bits": max(
                    PHASE2_MINIMUM_COMMAND_WIDTHS["layer_id_bits"],
                    derived["required_layer_id_bits"],
                ),
                "m_bits": 16,
                "n_bits": 16,
                "k_bits": 16,
                "sequence_position_bits": max(
                    PHASE2_MINIMUM_COMMAND_WIDTHS["sequence_position_bits"],
                    derived["required_sequence_position_bits"],
                ),
                "completion_tag_bits": 16,
                "address_bits": 64,
            },
            "current_shell_command_widths": dict(CURRENT_COMMAND_WIDTHS),
        },
        "derived": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in derived.items()
        },
        "static_capacity_checks": checks,
        "current_parameterization_gaps": parameterization_gaps(
            config, product, derived
        ),
        "compatibility": {
            "existing_0p5b_runtime_preflight": config.model_id == "qwen2.5-0.5b",
            "full_model_rtl_execution_validated": False,
            "larger_model_rtl_execution_validated": False,
            "synthesis_timing_or_u280_validated": False,
        },
        "claim_boundary": (
            "Canonical descriptor, derived-parameter arithmetic, and static "
            "capacity planning only. This artifact does not elaborate RTL, run "
            "a simulator/formal tool, execute a model, build a bitstream, or "
            "claim product completion."
        ),
    }


def _record(value: Any, path: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def validate_descriptor(value: Any) -> dict[str, Any]:
    descriptor = _record(value, "descriptor")
    model_id = descriptor.get("model_id")
    require(
        isinstance(model_id, str) and model_id in base_contract.CONFIG_BY_ID,
        "unsupported model_id",
    )
    expected = build_descriptor(base_contract.CONFIG_BY_ID[model_id])
    require(dict(descriptor) == expected, f"{model_id} descriptor is not canonical")
    checks = descriptor["static_capacity_checks"]
    require(checks and all(item["pass"] is True for item in checks), "capacity check failed")
    return dict(descriptor)


def locate_binding(binding: Binding) -> dict[str, Any]:
    path = ROOT / binding.path
    require(path.is_file(), f"inventory source missing: {binding.path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [
        (index + 1, line.rstrip())
        for index, line in enumerate(lines)
        if binding.fragment in line
    ]
    require(
        len(matches) >= binding.occurrence,
        f"{binding.binding_id} expected occurrence {binding.occurrence}, found {len(matches)}",
    )
    line_number, line_text = matches[binding.occurrence - 1]
    return {
        "id": binding.binding_id,
        "surface": binding.surface,
        "path": binding.path,
        "line": line_number,
        "symbol": binding.symbol,
        "contract_fields": list(binding.contract_fields),
        "current_0p5b_binding": binding.current_binding,
        "evidence_fragment": binding.fragment,
        "observed_line": line_text.strip(),
        "source_sha256": sha256_file(path),
        "successor_action": binding.successor_action,
    }


def build_inventory() -> dict[str, Any]:
    records = [locate_binding(binding) for binding in INVENTORY_BINDINGS]
    surfaces = sorted({record["surface"] for record in records})
    return {
        "schema_version": 1,
        "inventory_id": "ace2-phase2-hard-coded-0p5b-binding-inventory-v1",
        "scope": INVENTORY_SCOPE,
        "classification": (
            "static source inventory; no RTL/model/simulator/formal/synthesis/"
            "FPGA/U280 execution"
        ),
        "status": "PASS_INVENTORY_BOUND",
        "surface_counts": {
            surface: sum(record["surface"] == surface for record in records)
            for surface in surfaces
        },
        "bindings": records,
        "claim_boundary": (
            "This inventory identifies current active 0.5B dimension bindings "
            "that must be removed or descriptor-bound by successor work; it does "
            "not claim those removals have been implemented."
        ),
    }


def successor_plan() -> list[dict[str, Any]]:
    return [
        {
            "order": 1,
            "task": "descriptor-bound-sv-parameter-package",
            "objective": (
                "Generate or hand-maintain one RTL package surface carrying "
                "hidden/intermediate/head/KV/vocab/context parameters and exact "
                "0.5B defaults without changing the public ace2_shell port list."
            ),
            "acceptance_check": (
                "0.5B focused regressions still pass and generated parameter "
                "values match design/model_hardware_contracts/phase2/qwen2.5-0.5b.json."
            ),
        },
        {
            "order": 2,
            "task": "qkv-gqa-and-attention-geometry-parameterization",
            "objective": (
                "Replace QKV_KV_OUTPUTS, ATTN_HEAD_DIM, per-layer Q/K/V spans, "
                "GQA head loops, and fused-QKV legality checks with descriptor "
                "derived values."
            ),
            "acceptance_check": (
                "Single-layer Q/K/V, RoPE, attention-score, attention-value, "
                "and fused-QKV reference checks pass for all four descriptors."
            ),
        },
        {
            "order": 3,
            "task": "packer-and-command-layout-parameterization",
            "objective": (
                "Make image builders, independent verifiers, command schedules, "
                "KV strides, LM-head tile counts, and DS32 sidecars consume the "
                "same descriptor hash instead of 0.5B literals."
            ),
            "acceptance_check": (
                "No-execution package/layout checks generate consistent address "
                "maps for 0.5B, 1.5B, 3B, and 7B and reject a descriptor/image mismatch."
            ),
        },
        {
            "order": 4,
            "task": "host-runtime-contract-hash-binding",
            "objective": (
                "Extend the ACE2RT package/runtime identity path so the C++ host "
                "runtime validates model_id, descriptor hash, derived widths, "
                "and region map before accepting commands."
            ),
            "acceptance_check": (
                "Runtime package metadata reconstructs the descriptor hash and "
                "fails closed on hidden_size, kv_width, max_context, or image-region drift."
            ),
        },
        {
            "order": 5,
            "task": "representative-multisize-rtl-reference-checks",
            "objective": (
                "After independent review and RTL-stage admission, run bounded "
                "single-layer/operator RTL/reference checks for every descriptor "
                "without attempting full-model 1.5B/3B/7B Icarus runs."
            ),
            "acceptance_check": (
                "Each descriptor passes representative focused RTL/reference "
                "checks and existing 0.5B regressions remain green after every RTL change."
            ),
        },
    ]


def build_manifest() -> dict[str, Any]:
    schema_raw = canonical_bytes(SCHEMA)
    descriptors = []
    for config in base_contract.QWEN_CONFIGS:
        path = descriptor_path(config.model_id)
        descriptors.append(
            {
                "model_id": config.model_id,
                "artifact": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "primary_tier": PRODUCT_CONFIGS[config.model_id].primary_tier,
            }
        )
    inventory = {
        "artifact": INVENTORY_PATH.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(INVENTORY_PATH),
    }
    rtl_parameters = {
        "artifact": SV_PARAMETER_PATH.relative_to(ROOT).as_posix(),
        "model_id": SV_PARAMETER_MODEL_ID,
        "descriptor_sha256": sha256_file(descriptor_path(SV_PARAMETER_MODEL_ID)),
        "sha256": sha256_file(SV_PARAMETER_PATH),
    }
    plan = {
        "artifact": PLAN_PATH.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(PLAN_PATH),
    }
    descriptor_manifest = "".join(
        f"{item['sha256']}  {item['artifact']}\n" for item in descriptors
    ).encode()
    return {
        "schema_version": 1,
        "manifest_id": "ace2-phase2-unified-model-hardware-contract-v1",
        "status": "PASS_MANIFEST_BOUND",
        "schema": {
            "artifact": SCHEMA_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(schema_raw),
        },
        "descriptors": descriptors,
        "descriptor_set_sha256": sha256_bytes(descriptor_manifest),
        "inventory": inventory,
        "rtl_parameters": rtl_parameters,
        "implementation_plan": plan,
        "claims": {
            "canonical_descriptors": True,
            "descriptor_bound_sv_parameters": True,
            "hard_coded_0p5b_inventory_bound": True,
            "successor_plan_bounded": True,
            "larger_model_rtl_execution": False,
            "full_model_rtl_execution": False,
            "synthesis_timing_ppa": False,
            "fpga_or_u280_deployment": False,
            "product_completion": False,
        },
    }


def build_plan_markdown() -> str:
    inventory = build_inventory()
    descriptor_rows = []
    for config in base_contract.QWEN_CONFIGS:
        descriptor = build_descriptor(config)
        dimensions = descriptor["dimensions"]
        product = descriptor["product"]
        derived = descriptor["derived"]
        descriptor_rows.append(
            "| {model} | {tier} | {layers} | {hidden} | {intermediate} | {heads}/{kv} | {head_dim} | {vocab} | {context} | {tiles}/{channels} | {gaps} |".format(
                model=config.model_id,
                tier=product["primary_tier"],
                layers=dimensions["num_layers"],
                hidden=dimensions["hidden_size"],
                intermediate=dimensions["intermediate_size"],
                heads=dimensions["attention_heads"],
                kv=dimensions["kv_heads"],
                head_dim=dimensions["head_dim"],
                vocab=dimensions["vocab_size"],
                context=dimensions["max_context"],
                tiles=product["compute_tiles"],
                channels=product["memory_channels"],
                gaps=len(descriptor["current_parameterization_gaps"]),
            )
        )
    binding_rows = [
        "| {surface} | {path}:{line} | {symbol} | {binding} |".format(
            surface=record["surface"],
            path=record["path"],
            line=record["line"],
            symbol=record["symbol"],
            binding=record["current_0p5b_binding"].replace("|", "/"),
        )
        for record in inventory["bindings"]
    ]
    plan_rows = [
        f"| {item['order']} | {item['task']} | {item['acceptance_check']} |"
        for item in successor_plan()
    ]
    return "\n".join(
        [
            "# ACE-2 Phase-2 model hardware parameterization contract",
            "",
            "This is a specification-stage, additive successor contract. It preserves",
            "the accepted 0.5B v1 model-hardware descriptors and records no RTL, model,",
            "simulator, synthesis, FPGA, U280, PPA, timing, or product-completion result.",
            "",
            "## Unified descriptor matrix",
            "",
            "| Model | Primary tier | Layers | Hidden | MLP | Q heads/KV heads | Head dim | Vocab | Max context | Compute tiles/memory channels | Current gaps |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |",
            *descriptor_rows,
            "",
            "## Bound 0.5B hard-coded inventory",
            "",
            "| Surface | Source | Symbol | Current binding |",
            "| --- | --- | --- | --- |",
            *binding_rows,
            "",
            "## Narrow successor implementation plan",
            "",
            "| Order | Task | Decisive acceptance check |",
            "| ---: | --- | --- |",
            *plan_rows,
            "",
            "## Claim boundary",
            "",
            "The descriptor and inventory checks are static and deterministic. Larger-model",
            "records are configuration contracts only until successor RTL, packer,",
            "command generation, host runtime, and focused RTL/reference checks are",
            "implemented and independently reviewed.",
            "",
        ]
    )


def write_artifacts() -> None:
    SCHEMA_PATH.write_bytes(canonical_bytes(SCHEMA))
    DESCRIPTOR_DIR.mkdir(parents=True, exist_ok=True)
    for config in base_contract.QWEN_CONFIGS:
        descriptor_path(config.model_id).write_bytes(
            canonical_bytes(build_descriptor(config))
        )
    sv_descriptor = build_descriptor(
        base_contract.CONFIG_BY_ID[SV_PARAMETER_MODEL_ID]
    )
    SV_PARAMETER_PATH.parent.mkdir(parents=True, exist_ok=True)
    SV_PARAMETER_PATH.write_bytes(build_sv_parameter_include(sv_descriptor))
    INVENTORY_PATH.write_bytes(canonical_bytes(build_inventory()))
    PLAN_PATH.write_text(build_plan_markdown(), encoding="utf-8")
    MANIFEST_PATH.write_bytes(canonical_bytes(build_manifest()))


def validate_repository() -> dict[str, Any]:
    require(SCHEMA_PATH.is_file(), "missing Phase-2 schema")
    schema_raw = SCHEMA_PATH.read_bytes()
    require(schema_raw == canonical_bytes(SCHEMA), "Phase-2 schema is stale")
    descriptors = []
    descriptor_by_model_id = {}
    for config in base_contract.QWEN_CONFIGS:
        path = descriptor_path(config.model_id)
        require(path.is_file(), f"missing descriptor: {path.relative_to(ROOT)}")
        raw = path.read_bytes()
        descriptor = validate_descriptor(json.loads(raw))
        descriptor_by_model_id[config.model_id] = descriptor
        require(
            raw == canonical_bytes(build_descriptor(config)),
            f"descriptor is not canonical: {config.model_id}",
        )
        descriptors.append(
            {
                "model_id": config.model_id,
                "artifact": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(raw),
                "primary_tier": descriptor["product"]["primary_tier"],
                "current_parameterization_gap_count": len(
                    descriptor["current_parameterization_gaps"]
                ),
                "required_sequence_position_bits": descriptor["derived"][
                    "required_sequence_position_bits"
                ],
                "maximum_weight_plus_kv_bytes": descriptor["derived"][
                    "maximum_weight_plus_kv_bytes"
                ],
            }
        )
    sv_descriptor = descriptor_by_model_id[SV_PARAMETER_MODEL_ID]
    expected_sv_parameters = build_sv_parameter_include(sv_descriptor)
    require(SV_PARAMETER_PATH.is_file(), "missing generated SV parameters")
    require(
        SV_PARAMETER_PATH.read_bytes() == expected_sv_parameters,
        "generated SV parameters are stale",
    )
    expected_inventory_raw = canonical_bytes(build_inventory())
    require(INVENTORY_PATH.is_file(), "missing inventory")
    require(
        INVENTORY_PATH.read_bytes() == expected_inventory_raw,
        "hard-coded 0.5B inventory is stale",
    )
    require(PLAN_PATH.is_file(), "missing implementation plan")
    require(
        PLAN_PATH.read_text(encoding="utf-8") == build_plan_markdown(),
        "implementation plan is stale",
    )
    expected_manifest = build_manifest()
    require(MANIFEST_PATH.is_file(), "missing manifest")
    require(
        MANIFEST_PATH.read_bytes() == canonical_bytes(expected_manifest),
        "Phase-2 manifest is stale",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "ace2_phase2_model_hardware_contract_static_validation",
        "status": "PASS",
        "schema": {
            "artifact": SCHEMA_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(schema_raw),
        },
        "descriptors": descriptors,
        "inventory": {
            "artifact": INVENTORY_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(expected_inventory_raw),
            "binding_count": len(INVENTORY_BINDINGS),
            "surface_counts": build_inventory()["surface_counts"],
        },
        "rtl_parameters": {
            "artifact": SV_PARAMETER_PATH.relative_to(ROOT).as_posix(),
            "model_id": SV_PARAMETER_MODEL_ID,
            "descriptor_sha256": sha256_file(
                descriptor_path(SV_PARAMETER_MODEL_ID)
            ),
            "sha256": sha256_bytes(expected_sv_parameters),
            "values": sv_parameter_values(sv_descriptor),
        },
        "manifest": {
            "artifact": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(MANIFEST_PATH),
        },
        "claims": expected_manifest["claims"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.write:
            write_artifacts()
        if not args.write or args.check:
            sys.stdout.buffer.write(canonical_bytes(validate_repository()))
    except (OSError, json.JSONDecodeError, Phase2ContractError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
