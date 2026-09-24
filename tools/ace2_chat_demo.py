#!/usr/bin/env python3
"""Prompt-redacted local chat front end for the ACE-2 RTL runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import re
import struct
import subprocess
import sys
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoTokenizer

from tools import run_full_qwen_command_schedule_runtime as accepted_runtime
from tools import ace2_stage1_chat_product as stage1_product
from tools.model_hardware_contract import runtime_preflight
from tools.ace2_attention_compose_reference import AttentionComposePhaseReplay
from tools.ace2_attention_value_reference import (
    AttentionValueCase,
    reference_attention_value,
)
from tools.ace2_dynamic_scale32_reference import (
    dynamic_group,
    quantize_for_delta,
    round_shift_even_signed,
)
from tools.ace2_projection_reference import ProjectionCase, reference_projection
from tools.ace2_quality_contracts import pack_scale32
from tools.ace2_residual_reference import reference_residual_add
from tools.ace2_rmsnorm_reference import reference_rmsnorm
from tools.ace2_rope_reference import RopeCase, reference_rope
from tools.ace2_silu_gate_reference import (
    SiluGateCase,
    reference_silu_gate_scaled_int8,
)
from tools.ace2_softmax_reference import SoftmaxCase, reference_softmax


REVISION = accepted_runtime.REVISION
TOKENIZER_SNAPSHOT = (
    Path(HF_HUB_CACHE)
    / "models--Qwen--Qwen2.5-0.5B"
    / "snapshots"
    / REVISION
)
DEFAULT_OUTPUT = ROOT / "build/ace2_chat_demo/latest"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
MIN_NEW_TOKENS = 1
MAX_NEW_TOKENS = 256
DEFAULT_PERSISTENCE_BATCH_COMMANDS = 64
PACKAGE_SEED_OFFSET = 8 + 4 + 4 + 4
PACKAGE_MAGIC = b"ACE2RT1\0"
PACKAGE_V2_MAGIC = b"ACE2RT2\0"
PACKAGE_V2_HEADER = struct.Struct("<8s10IQ32s32s32s32s32s")
PACKAGE_V2_HEADER_BYTES = 216
PACKAGE_V2_ROPE_RECORD = struct.Struct("<32h32h")
PACKAGE_V2_DYNAMIC_EXTENSION_MAGIC = b"DS32"
PACKAGE_V2_DYNAMIC_EXTENSION = struct.Struct("<4sIIIQ32sII")
PACKAGE_V2_DYNAMIC_EXTENSION_BYTES = 64
PACKAGE_V2_DYNAMIC_RECORD = struct.Struct("<Q64s")
DYNAMIC_SCALE32_FEATURE_SIDECARS = 1
DYNAMIC_SCALE32_FLAG = 1 << 6
DYNAMIC_SCALE32_HOST_PLAN_LAYER = 0xFF
DYNAMIC_SCALE32_HOST_PLAN_OPCODE = 0xFE
DYNAMIC_SCALE32_INITIAL_GROUP_LANES = 128
DYNAMIC_SCALE32_INITIAL_GROUP_COUNT = 7
DYNAMIC_SCALE32_INITIAL_ELEMENTS = 896
DYNAMIC_SCALE32_RMS_OUTPUT_DELTAS = (0, -1, -1, -1, -1, -1, 0)
DYNAMIC_SCALE32_RMSNORM_GAIN_BYTES = DYNAMIC_SCALE32_INITIAL_ELEMENTS * 2
DYNAMIC_SCALE32_RMSNORM_SCALE_OFFSET = DYNAMIC_SCALE32_RMSNORM_GAIN_BYTES + 8
DYNAMIC_SCALE32_FROZEN_FLAGS = DYNAMIC_SCALE32_FLAG | 0x09
DYNAMIC_SCALE32_PROMPT_OPERATORS = (
    "input_rmsnorm",
    "q_proj",
    "k_proj",
    "v_proj",
)
QKV_SCHEDULE_FUSED = "fused"
QKV_SCHEDULE_LEGACY = "legacy"
FUSED_QKV_OPCODE = 0x0B
FUSED_QKV_OPERATOR_ID = len(accepted_runtime.OPERATOR_IDS)
RT2_OPERATOR_IDS = {
    **accepted_runtime.OPERATOR_IDS,
    "fused_qkv": FUSED_QKV_OPERATOR_ID,
}
FUSED_QKV_WEIGHT_OFFSETS = (0, 401_408, 458_752)
FUSED_QKV_METADATA_OFFSETS = (0, 14_336, 16_384)
FUSED_QKV_OUTPUT_OFFSETS = (0, 896, 1_024)
FUSED_QKV_WEIGHT_BASE = 0x0000000100000000
FUSED_QKV_WEIGHT_LAYER_STRIDE = 0x71C000
FUSED_QKV_METADATA_BASE = 0x0000000200000000
FUSED_QKV_METADATA_LAYER_STRIDE = 0x31800
FUSED_QKV_WEIGHT_BYTES = 516_096
FUSED_QKV_METADATA_BYTES = 18_432
FUSED_QKV_OUTPUT_BYTES = 1_152
FUSED_QKV_ACTIVATION_ADDR = 0x0000001000000700
FUSED_QKV_OUTPUT_ADDR = 0x0000001000000A80
TERMINATION_TOKEN_IDS = (151643, 151645)
PINNED_EMBEDDING_OFFSET = 32288
MAX_SEQUENCE_POSITIONS = 32768
KV_BYTES_PER_TOKEN = 272
LM_HEAD_TILE_COUNT = 4748
RT2_SCORE_WORKSPACE_BASE = 0x0000001001000000
RT2_SCORE_HEAD_STRIDE = MAX_SEQUENCE_POSITIONS * 16
JOURNAL_V2_HEADER_BYTES = 8 + 4 + 32
JOURNAL_COMPLETED_PREFIX = struct.Struct("<IQIIHBBiiI")
JOURNAL_WRITE_RECORD = struct.Struct("<QH16s")
IMAGE_REGIONS = (
    (0x0000000100000000, 246_980_608, 0),
    (0x0000000200000000, 7_297_024, 246_980_608),
    (0x0000000300000000, 88_592, 254_277_632),
    (0x0000000310000000, 55_296, 254_366_224),
)
TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-0.5B"
TOKENIZER_FILE_SHA256 = {
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
}
DIAGNOSTIC_INSTRUCT_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct"
DIAGNOSTIC_INSTRUCT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DIAGNOSTIC_INSTRUCT_SNAPSHOT = (
    Path(HF_HUB_CACHE)
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / DIAGNOSTIC_INSTRUCT_REVISION
)
DIAGNOSTIC_INSTRUCT_SOURCE_SHA256 = {
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
}
DIAGNOSTIC_INSTRUCT_SOURCE_BYTES = {
    "config.json": 659,
    "model.safetensors": 988_097_824,
    "tokenizer.json": 7_031_645,
    "tokenizer_config.json": 7_305,
}
DIAGNOSTIC_INSTRUCT_CACHE_MANIFEST_FILENAME = (
    "diagnostic_instruct_cache_manifest.json"
)
DIAGNOSTIC_INSTRUCT_CHAT_TEMPLATE_SHA256 = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
DIAGNOSTIC_INSTRUCT_IMAGE_DIR = (
    ROOT / "evidence/verification/build-qwen2.5-0.5b-instruct-w4a8-image-v1"
)
DIAGNOSTIC_INSTRUCT_IMAGE = DIAGNOSTIC_INSTRUCT_IMAGE_DIR / "full_model_image.bin"
DIAGNOSTIC_INSTRUCT_IMAGE_CONTRACT = (
    DIAGNOSTIC_INSTRUCT_IMAGE_DIR / "image_contract_v2.json"
)
DIAGNOSTIC_INSTRUCT_VALIDATION_REPORT = (
    DIAGNOSTIC_INSTRUCT_IMAGE_DIR / "validation_report.json"
)
DIAGNOSTIC_INSTRUCT_IMAGE_SHA256 = (
    "ce1f94d930bf4d195b33c6aab4b18736419a34bb3671ea4d23ef2fe6be08ad42"
)
DIAGNOSTIC_INSTRUCT_IMAGE_CONTRACT_SHA256 = (
    "c8aa96b6fe660ff78f659eadfca7bf1785d391fb8b3c00ee4a940813604f466e"
)
DIAGNOSTIC_INSTRUCT_VALIDATION_REPORT_SHA256 = (
    "d29779283c32a6093f833c6df3074daacfcf532b0ea0f4bf9d1d738f2d3d8a79"
)
DIAGNOSTIC_INSTRUCT_IMAGE_CONTRACT_BODY_SHA256 = (
    "5afe2bad158001c7b493913a946e0ddbd3471fa52983b0627a283c41343aff76"
)
DIAGNOSTIC_INSTRUCT_GAP_REPORT = (
    ROOT
    / "research/raw/specification/stage1-product-gap-audit-20260809T194126Z.md"
)
DIAGNOSTIC_INSTRUCT_GAP_REPORT_SHA256 = (
    "1340bc56a3f20b347742669ae25fe6fcc3a3dfb7483cc57ca30174b4ae10542f"
)
DIAGNOSTIC_INSTRUCT_PREPARE_STATUS = (
    "PASS_PINNED_INSTRUCT_DIAGNOSTIC_PREPARE_CONTRACT"
)
DIAGNOSTIC_INSTRUCT_ACCEPTED_PREPARE_RELATIVE = Path(
    "build/audit-arbitrary-text-chat-product-gap-v1/instruct-prepare"
)
DIAGNOSTIC_INSTRUCT_ACCEPTED_PACKAGE_SHA256 = (
    "3e3eae19d69d0b24b7bdfd0b5eb0f35b1cf663d22554f988f6035b99f8191295"
)
DIAGNOSTIC_INSTRUCT_ACCEPTED_PROVENANCE_SHA256 = (
    "e8105f946895a25d89ee2f1eb6dfc02f7f8e81befb65e0ea6595aff13671d9d7"
)
DIAGNOSTIC_INSTRUCT_ACCEPTED_PREFLIGHT_SHA256 = (
    "b1c0c6342ec77ce61bfba0626aca6146aabdc680a92440a497f45543fbc5d9d1"
)
DIAGNOSTIC_INSTRUCT_ACCEPTED_DS32_SHA256 = (
    "bdaa2b15200f09268c454d3daf92797fe85ccb97f729ec2a2d6cb8773eda9ed5"
)
DIAGNOSTIC_INSTRUCT_ACCEPTED_PACKAGE_BYTES = 92_917_676
DIAGNOSTIC_INSTRUCT_ACCEPTED_PACKAGE_COMMANDS = 1_366_317
DIAGNOSTIC_INSTRUCT_ACCEPTED_PROMPT_SHA256 = (
    "5f85c0fedfd298bfc262408927bc4fb5a714f0de236d06d06224a58aa8aae1bc"
)
DIAGNOSTIC_INSTRUCT_ACCEPTED_PREPARE_HOST_SHA256 = (
    "e17b737f1b37ab99d6f90981279f96a69942050bc57a75d0078cb836a636e027"
)
DIAGNOSTIC_INSTRUCT_ACCEPTED_PREPARE_HOST_BYTES = 201_731
DIAGNOSTIC_INSTRUCT_RUNTIME_BINDING_FILENAME = (
    "diagnostic_instruct_runtime_binding.json"
)
DIAGNOSTIC_INSTRUCT_FUTURE_AUTHORITY_REQUIREMENT = (
    "future invocation requires separate external authority binding the exact "
    "package, source, constraints, and execution-tree hashes"
)
DIAGNOSTIC_INSTRUCT_RUNTIME_SOURCE_SHA256 = (
    "61b94d528fa256cba53faa398419b2c118b06cca32dbb8531be1bd0dc4c8ebb3"
)
PINNED_CHAT_SEMANTICS_AUDIT = (
    ROOT
    / "build/ace2_chat_diagnostics/"
    "reviewer-r1-bf16-production-chat-contract-20260805.json"
)
PINNED_CHAT_SEMANTICS_AUDIT_SHA256 = (
    "8281623c82fac57c9c0f430b278adc0fbd3f70295d5fc6823a114c2f71a4664f"
)
PINNED_CHAT_SEMANTICS_AUDIT_STATUS = "FAIL_PINNED_BF16_BASE_CHAT_CONTRACT"
PINNED_BASE_COMPLETION_CONTRACT_STATUS = "PASS_PINNED_BASE_COMPLETION_CONTRACT"
FINAL_CERTIFICATE = ROOT / "research/FINAL_PRODUCT_CERTIFICATE.json"
FINAL_CERTIFICATE_SUM = ROOT / "research/FINAL_PRODUCT_CERTIFICATE.sha256"
FINAL_RTL_TREE_MANIFEST = (
    ROOT
    / "evidence/verification/rtl-rmsnorm-final-sumsq-carry-cone-repair-v1/rtl_tree_manifest.sha256"
)
SILU_INPUT_SCALE_TABLE = ROOT / "verification/generated/silu_input_scale_table.json"
SILU_INPUT_SCALE_TABLE_RTL = ROOT / "rtl/generated/ace2_silu_input_scale_table.svh"
SILU_INPUT_SCALE_GENERATOR = ROOT / "tools/gen_silu_gate_vectors.py"
SILU_SCALE_INVENTORY = (
    ROOT
    / "evidence/verification/rtl-full-qwen-autoregressive-integration-v1"
    / "integration_inventory.json"
)
SILU_SCALE_SOURCE = (
    ROOT
    / "evidence/layer0_tile_bfp_score_attention_v1"
    / "paired-smoke-20260801-v1/derived_scales.json"
)
RTL_SOURCE_PATTERNS = ("rtl/**/*.sv", "rtl/**/*.svh")
RUNTIME_SIMULATION_SOURCE_PATHS = (
    "Makefile",
    "verification/verilator/ace2_shell_runtime_harness.sv",
    "verification/verilator/ace2_shell_runtime_main.cpp",
)
SILU_SCALE_BOUND_SHA256 = {
    "tools/gen_silu_gate_vectors.py": "fde5fd2ff77bb2a98e434b97baaa7dbbac165a5b90623358080c3308d24fc172",
    "rtl/generated/ace2_silu_input_scale_table.svh": "7ecf3573856a262160f76a4d8aff1d91f894a3731e80ffdeb4890d597ab6d455",
    "verification/generated/silu_input_scale_table.json": "b135093651b5d32b1c74ed1af84768f26d744efb51ffb89332a4944cd817086c",
    "evidence/verification/rtl-full-qwen-autoregressive-integration-v1/integration_inventory.json": "6472b1460c9908255655fcca45d0e6d5eadc065f6debf5b6605e9107568d1950",
    "evidence/layer0_tile_bfp_score_attention_v1/paired-smoke-20260801-v1/derived_scales.json": "78280eb606e0c14ea74163f72f45cbb045670fb26f400c054013b9726266ebaa",
}
RTL_INCLUDE_PATTERN = re.compile(r'^\s*`include\s+"([^"]+)"')


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _verified_file_record(
    path: Path,
    expected_sha256: str,
    *,
    identity: str,
    artifact: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {identity}: {path.name}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(f"{identity} SHA-256 changed")
    return {
        "artifact": artifact if artifact is not None else _artifact_label(path),
        "bytes": path.stat().st_size,
        "sha256": observed_sha256,
    }


def prepare_profile(diagnostic_instruct: bool) -> dict[str, Any]:
    if diagnostic_instruct:
        return {
            "diagnostic_only": True,
            "repository": DIAGNOSTIC_INSTRUCT_REPOSITORY,
            "revision": DIAGNOSTIC_INSTRUCT_REVISION,
            "snapshot": DIAGNOSTIC_INSTRUCT_SNAPSHOT,
            "model": DIAGNOSTIC_INSTRUCT_SNAPSHOT / "model.safetensors",
            "image": DIAGNOSTIC_INSTRUCT_IMAGE,
        }
    return {
        "diagnostic_only": False,
        "repository": TOKENIZER_REPOSITORY,
        "revision": REVISION,
        "snapshot": TOKENIZER_SNAPSHOT,
        "model": accepted_runtime.MODEL,
        "image": accepted_runtime.IMAGE,
    }


def diagnostic_instruct_expected_file_hashes() -> dict[Path, str]:
    return {
        **{
            DIAGNOSTIC_INSTRUCT_SNAPSHOT / name: expected
            for name, expected in DIAGNOSTIC_INSTRUCT_SOURCE_SHA256.items()
        },
        DIAGNOSTIC_INSTRUCT_IMAGE: DIAGNOSTIC_INSTRUCT_IMAGE_SHA256,
        DIAGNOSTIC_INSTRUCT_IMAGE_CONTRACT: (
            DIAGNOSTIC_INSTRUCT_IMAGE_CONTRACT_SHA256
        ),
        DIAGNOSTIC_INSTRUCT_VALIDATION_REPORT: (
            DIAGNOSTIC_INSTRUCT_VALIDATION_REPORT_SHA256
        ),
        DIAGNOSTIC_INSTRUCT_GAP_REPORT: DIAGNOSTIC_INSTRUCT_GAP_REPORT_SHA256,
    }


def diagnostic_instruct_source_inventory(snapshot: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for name, expected_sha256 in DIAGNOSTIC_INSTRUCT_SOURCE_SHA256.items():
        path = snapshot / name
        record: dict[str, Any] = {
            "artifact": (
                f"hf-cache://{DIAGNOSTIC_INSTRUCT_REPOSITORY}@"
                f"{DIAGNOSTIC_INSTRUCT_REVISION}/{name}"
            ),
            "expected_bytes": DIAGNOSTIC_INSTRUCT_SOURCE_BYTES[name],
            "expected_sha256": expected_sha256,
            "present": path.is_file(),
        }
        if path.is_file():
            record["observed_bytes"] = path.stat().st_size
            record["observed_sha256"] = sha256_file(path)
            record["matches"] = (
                record["observed_bytes"] == record["expected_bytes"]
                and record["observed_sha256"] == record["expected_sha256"]
            )
        else:
            record["matches"] = False
        files.append(record)
    present_count = sum(record["present"] for record in files)
    valid_count = sum(record["matches"] for record in files)
    if valid_count == len(files):
        status = "VALID"
    elif present_count == 0:
        status = "ABSENT"
    else:
        status = "INCOMPLETE_OR_INVALID"
    fingerprint_body = {
        "repository": DIAGNOSTIC_INSTRUCT_REPOSITORY,
        "revision": DIAGNOSTIC_INSTRUCT_REVISION,
        "files": files,
    }
    return {
        **fingerprint_body,
        "status": status,
        "source_fingerprint_sha256": sha256_bytes(canonical_bytes(fingerprint_body)),
    }


def inspect_diagnostic_instruct_metadata(snapshot: Path) -> dict[str, Any]:
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    expected_config = {
        "model_type": "qwen2",
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_attention_heads": 14,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "max_position_embeddings": MAX_SEQUENCE_POSITIONS,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError("pinned Instruct model config geometry changed")

    model_path = snapshot / "model.safetensors"
    with model_path.open("rb") as handle:
        raw_header_bytes = handle.read(8)
        if len(raw_header_bytes) != 8:
            raise RuntimeError("pinned Instruct safetensors header is truncated")
        header_bytes = struct.unpack("<Q", raw_header_bytes)[0]
        if not 2 <= header_bytes <= 16 << 20:
            raise RuntimeError("pinned Instruct safetensors header length is invalid")
        raw_header = handle.read(header_bytes)
    if len(raw_header) != header_bytes:
        raise RuntimeError("pinned Instruct safetensors header is truncated")
    try:
        safetensors_header = json.loads(raw_header)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "pinned Instruct safetensors header is not canonical JSON"
        ) from error
    tensor_records = {
        name: record
        for name, record in safetensors_header.items()
        if name != "__metadata__"
    }
    embedding = tensor_records.get("model.embed_tokens.weight")
    if not isinstance(embedding, dict) or (
        embedding.get("dtype") != "BF16"
        or embedding.get("shape") != [151936, 896]
    ):
        raise RuntimeError("pinned Instruct embedding tensor metadata changed")
    layer_indices = sorted(
        {
            int(match.group(1))
            for name in tensor_records
            if (match := re.match(r"^model\.layers\.(\d+)\.", name))
        }
    )
    if layer_indices != list(range(24)):
        raise RuntimeError("pinned Instruct safetensors layer metadata changed")

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer_config = json.loads(
        (snapshot / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str):
        raise RuntimeError("pinned Instruct chat template is missing")
    verify_loaded_diagnostic_instruct_tokenizer(tokenizer, chat_template)
    if (
        tokenizer.vocab_size != 151643
        or len(tokenizer) != 151665
        or tokenizer.eos_token_id != 151645
        or tokenizer.pad_token_id != 151643
    ):
        raise RuntimeError("pinned Instruct tokenizer metadata changed")
    return {
        "status": "AVAILABLE_AND_BOUND",
        "model_config": expected_config,
        "safetensors": {
            "header_bytes": header_bytes,
            "tensor_count": len(tensor_records),
            "embedding": {
                "name": "model.embed_tokens.weight",
                "dtype": embedding["dtype"],
                "shape": embedding["shape"],
            },
            "layer_indices": layer_indices,
            "weights_loaded": False,
            "model_constructed": False,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "base_vocab_size": tokenizer.vocab_size,
            "mapped_decode_domain": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "chat_template_sha256": sha256_bytes(chat_template.encode("utf-8")),
            "local_files_only": True,
            "tokenization_executed": False,
            "decode_executed": False,
        },
    }


def prepare_diagnostic_instruct_cache(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            "diagnostic Instruct cache preparation requires a fresh empty --output"
        )
    before = diagnostic_instruct_source_inventory(DIAGNOSTIC_INSTRUCT_SNAPSHOT)
    retrieval_performed = before["status"] != "VALID"
    if retrieval_performed:
        downloaded = Path(
            snapshot_download(
                repo_id=DIAGNOSTIC_INSTRUCT_REPOSITORY,
                revision=DIAGNOSTIC_INSTRUCT_REVISION,
                allow_patterns=sorted(DIAGNOSTIC_INSTRUCT_SOURCE_SHA256),
                cache_dir=HF_HUB_CACHE,
                max_workers=1,
            )
        )
        if downloaded.resolve() != DIAGNOSTIC_INSTRUCT_SNAPSHOT.resolve():
            raise RuntimeError("pinned Instruct retrieval resolved an unexpected snapshot")
    after = diagnostic_instruct_source_inventory(DIAGNOSTIC_INSTRUCT_SNAPSHOT)
    if after["status"] != "VALID":
        raise RuntimeError("pinned Instruct retrieval did not produce the exact snapshot")

    resolved_blobs: set[Path] = set()
    repository_cache = DIAGNOSTIC_INSTRUCT_SNAPSHOT.parents[1].resolve()
    blob_root = (repository_cache / "blobs").resolve()
    for name in DIAGNOSTIC_INSTRUCT_SOURCE_SHA256:
        snapshot_file = DIAGNOSTIC_INSTRUCT_SNAPSHOT / name
        resolved = snapshot_file.resolve()
        if not snapshot_file.is_symlink() or resolved.parent != blob_root:
            raise RuntimeError(
                "pinned Instruct cache is not using the shared blob/snapshot layout"
            )
        resolved_blobs.add(resolved)
    retained_unique_blob_bytes = sum(path.stat().st_size for path in resolved_blobs)
    expected_logical_bytes = sum(DIAGNOSTIC_INSTRUCT_SOURCE_BYTES.values())
    if retained_unique_blob_bytes != expected_logical_bytes:
        raise RuntimeError("pinned Instruct cache retained-byte bound changed")

    metadata = inspect_diagnostic_instruct_metadata(DIAGNOSTIC_INSTRUCT_SNAPSHOT)
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "diagnostic_instruct_pinned_cache_binding",
        "status": "READY_PINNED_LOCAL_SNAPSHOT",
        "source_before": before,
        "source_after": after,
        "cache": {
            "repository": DIAGNOSTIC_INSTRUCT_REPOSITORY,
            "revision": DIAGNOSTIC_INSTRUCT_REVISION,
            "snapshot": (
                f"hf-cache://{DIAGNOSTIC_INSTRUCT_REPOSITORY}@"
                f"{DIAGNOSTIC_INSTRUCT_REVISION}/"
            ),
            "retrieval_performed": retrieval_performed,
            "snapshot_download": {
                "revision_is_full_commit": True,
                "allow_patterns": sorted(DIAGNOSTIC_INSTRUCT_SOURCE_SHA256),
                "max_workers": 1,
            },
            "bounded_storage": {
                "requested_payload_file_count": len(
                    DIAGNOSTIC_INSTRUCT_SOURCE_SHA256
                ),
                "expected_logical_bytes": expected_logical_bytes,
                "retained_unique_blob_count": len(resolved_blobs),
                "retained_unique_blob_bytes": retained_unique_blob_bytes,
                "snapshot_files_are_shared_blob_links": True,
                "model_snapshot_copied_into_runtime_output": False,
                "tokenizer_cache_copied_into_runtime_output": False,
            },
        },
        "metadata_availability": metadata,
        "execution": {
            "model_weights_loaded": False,
            "model_constructed": False,
            "model_executed": False,
            "generation_executed": False,
            "decode_executed": False,
            "rtl_executed": False,
            "benchmark_executed": False,
            "official_attempt_consumed": False,
        },
        "claims": {
            "completion_claim": False,
            "rtl_agreement_claim": False,
            "latency_claim": False,
            "ppa_claim": False,
            "fpga_or_u280_claim": False,
            "stage_advancement_claim": False,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_atomic(
        output / DIAGNOSTIC_INSTRUCT_CACHE_MANIFEST_FILENAME,
        canonical_bytes(result),
    )
    return result


def authenticate_diagnostic_instruct_prepare() -> dict[str, Any]:
    source_files: dict[str, dict[str, Any]] = {}
    for name, expected_sha256 in DIAGNOSTIC_INSTRUCT_SOURCE_SHA256.items():
        source_files[name] = _verified_file_record(
            DIAGNOSTIC_INSTRUCT_SNAPSHOT / name,
            expected_sha256,
            identity=f"pinned Instruct {name}",
            artifact=(
                f"hf-cache://{DIAGNOSTIC_INSTRUCT_REPOSITORY}@"
                f"{DIAGNOSTIC_INSTRUCT_REVISION}/{name}"
            ),
        )

    tokenizer_config = json.loads(
        (DIAGNOSTIC_INSTRUCT_SNAPSHOT / "tokenizer_config.json").read_text(
            encoding="utf-8"
        )
    )
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str):
        raise RuntimeError("pinned Instruct chat template is missing")
    chat_template_sha256 = sha256_bytes(chat_template.encode("utf-8"))
    if chat_template_sha256 != DIAGNOSTIC_INSTRUCT_CHAT_TEMPLATE_SHA256:
        raise RuntimeError("pinned Instruct chat template SHA-256 changed")

    image = _verified_file_record(
        DIAGNOSTIC_INSTRUCT_IMAGE,
        DIAGNOSTIC_INSTRUCT_IMAGE_SHA256,
        identity="pinned Instruct W4A8 image",
    )
    image_contract = _verified_file_record(
        DIAGNOSTIC_INSTRUCT_IMAGE_CONTRACT,
        DIAGNOSTIC_INSTRUCT_IMAGE_CONTRACT_SHA256,
        identity="pinned Instruct image contract",
    )
    validation_report = _verified_file_record(
        DIAGNOSTIC_INSTRUCT_VALIDATION_REPORT,
        DIAGNOSTIC_INSTRUCT_VALIDATION_REPORT_SHA256,
        identity="pinned Instruct validation report",
    )
    gap_report = _verified_file_record(
        DIAGNOSTIC_INSTRUCT_GAP_REPORT,
        DIAGNOSTIC_INSTRUCT_GAP_REPORT_SHA256,
        identity="Fresh-L2-accepted Stage-1 gap report",
    )

    contract_wrapper = json.loads(
        DIAGNOSTIC_INSTRUCT_IMAGE_CONTRACT.read_text(encoding="utf-8")
    )
    if set(contract_wrapper) != {"contract", "contract_sha256"}:
        raise RuntimeError("pinned Instruct image contract wrapper changed")
    contract = contract_wrapper.get("contract")
    if not isinstance(contract, dict):
        raise RuntimeError("pinned Instruct image contract body changed")
    if (
        contract_wrapper.get("contract_sha256")
        != DIAGNOSTIC_INSTRUCT_IMAGE_CONTRACT_BODY_SHA256
        or sha256_bytes(canonical_bytes(contract))
        != DIAGNOSTIC_INSTRUCT_IMAGE_CONTRACT_BODY_SHA256
    ):
        raise RuntimeError("pinned Instruct image contract body SHA-256 changed")
    contract_model = contract.get("source_model", {})
    contract_safetensors = contract_model.get("safetensors", {})
    contract_scope = contract.get("scope_guards", {})
    if (
        contract_model.get("repository") != DIAGNOSTIC_INSTRUCT_REPOSITORY
        or contract_model.get("revision") != DIAGNOSTIC_INSTRUCT_REVISION
        or contract_safetensors.get("sha256")
        != DIAGNOSTIC_INSTRUCT_SOURCE_SHA256["model.safetensors"]
        or contract.get("option_b_identity_inputs", {}).get(
            "chat_template_sha256"
        )
        != DIAGNOSTIC_INSTRUCT_CHAT_TEMPLATE_SHA256
        or contract.get("full_image", {}).get("bytes") != image["bytes"]
        or contract_scope.get("command_runtime_or_decoder_execution") is not False
        or contract_scope.get("rtl_or_testbench_change") is not False
    ):
        raise RuntimeError("pinned Instruct image contract identity changed")

    validation = json.loads(
        DIAGNOSTIC_INSTRUCT_VALIDATION_REPORT.read_text(encoding="utf-8")
    )
    validation_scope = validation.get("scope_guards", {})
    if (
        validation.get("status") != "PASS"
        or validation.get("contract_sha256")
        != DIAGNOSTIC_INSTRUCT_IMAGE_CONTRACT_BODY_SHA256
        or validation_scope.get("command_runtime_or_decoder_execution") is not False
        or validation_scope.get("rtl_or_testbench_change") is not False
        or validation_scope.get("model_called") is not False
    ):
        raise RuntimeError("pinned Instruct validation report identity changed")

    return {
        "repository": DIAGNOSTIC_INSTRUCT_REPOSITORY,
        "revision": DIAGNOSTIC_INSTRUCT_REVISION,
        "snapshot": DIAGNOSTIC_INSTRUCT_SNAPSHOT,
        "model_path": DIAGNOSTIC_INSTRUCT_SNAPSHOT / "model.safetensors",
        "image_path": DIAGNOSTIC_INSTRUCT_IMAGE,
        "source_files": source_files,
        "chat_template": chat_template,
        "chat_template_sha256": chat_template_sha256,
        "image": image,
        "image_contract": image_contract,
        "validation_report": validation_report,
        "gap_report": gap_report,
        "metadata_availability": inspect_diagnostic_instruct_metadata(
            DIAGNOSTIC_INSTRUCT_SNAPSHOT
        ),
    }


def diagnostic_instruct_runtime_binding_expected_file_hashes() -> dict[Path, str]:
    prepare_root = ROOT / DIAGNOSTIC_INSTRUCT_ACCEPTED_PREPARE_RELATIVE
    return {
        prepare_root / "runtime_package.bin": (
            DIAGNOSTIC_INSTRUCT_ACCEPTED_PACKAGE_SHA256
        ),
        prepare_root / "provenance.json": (
            DIAGNOSTIC_INSTRUCT_ACCEPTED_PROVENANCE_SHA256
        ),
        prepare_root / "product_contract_preflight.json": (
            DIAGNOSTIC_INSTRUCT_ACCEPTED_PREFLIGHT_SHA256
        ),
        prepare_root / "ds32_prompt_applicability.json": (
            DIAGNOSTIC_INSTRUCT_ACCEPTED_DS32_SHA256
        ),
    }


def _require_canonical_file(path: Path, root: Path, *, identity: str) -> Path:
    if root.is_symlink():
        raise RuntimeError(f"{identity} root is noncanonical")
    resolved_root = root.resolve()
    if path.is_symlink() or path.resolve().parent != resolved_root:
        raise RuntimeError(f"{identity} path is noncanonical")
    return path


def _accepted_diagnostic_instruct_prepare_paths() -> dict[str, Path]:
    prepare_root = ROOT / DIAGNOSTIC_INSTRUCT_ACCEPTED_PREPARE_RELATIVE
    expected_root = ROOT.resolve() / DIAGNOSTIC_INSTRUCT_ACCEPTED_PREPARE_RELATIVE
    if prepare_root.resolve() != expected_root or prepare_root.is_symlink():
        raise RuntimeError("accepted diagnostic Instruct prepare root is noncanonical")
    return {
        "root": prepare_root,
        "package": _require_canonical_file(
            prepare_root / "runtime_package.bin",
            prepare_root,
            identity="accepted diagnostic Instruct package",
        ),
        "provenance": _require_canonical_file(
            prepare_root / "provenance.json",
            prepare_root,
            identity="accepted diagnostic Instruct provenance",
        ),
        "preflight": _require_canonical_file(
            prepare_root / "product_contract_preflight.json",
            prepare_root,
            identity="accepted diagnostic Instruct preflight",
        ),
        "ds32": _require_canonical_file(
            prepare_root / "ds32_prompt_applicability.json",
            prepare_root,
            identity="accepted diagnostic Instruct DS32 applicability",
        ),
    }


def _load_authenticated_json(path: Path, *, identity: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{identity} is not canonical UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{identity} top level changed")
    return value


def validate_diagnostic_instruct_runtime_binding_provenance(
    *,
    provenance: dict[str, Any],
    preflight: dict[str, Any],
    ds32_applicability: dict[str, Any],
    authenticated: dict[str, Any],
) -> None:
    expected_false = (
        "product_acceptance_eligible",
        "completion_claim",
        "runtime_executed",
        "generation_executed",
        "decode_executed",
    )
    if (
        provenance.get("schema_version") != 1
        or provenance.get("status") != "PREPARED"
        or provenance.get("classification")
        != "diagnostic_instruct_ace2rt2_host_package_preparation"
        or provenance.get("diagnostic_only") is not True
        or any(provenance.get(key) is not False for key in expected_false)
    ):
        raise RuntimeError("accepted diagnostic Instruct provenance policy changed")

    tokenizer = provenance.get("tokenizer")
    if not isinstance(tokenizer, dict) or (
        tokenizer.get("repository") != DIAGNOSTIC_INSTRUCT_REPOSITORY
        or tokenizer.get("revision") != DIAGNOSTIC_INSTRUCT_REVISION
        or tokenizer.get("chat_template_sha256")
        != DIAGNOSTIC_INSTRUCT_CHAT_TEMPLATE_SHA256
        or tokenizer.get("chat_template_applied") is not True
        or tokenizer.get("add_generation_prompt") is not True
    ):
        raise RuntimeError("accepted diagnostic Instruct tokenizer binding changed")

    bindings = provenance.get("bindings")
    if not isinstance(bindings, dict):
        raise RuntimeError("accepted diagnostic Instruct bindings are missing")
    expected_bindings = {
        "model": authenticated["source_files"]["model.safetensors"],
        "model_config": authenticated["source_files"]["config.json"],
        "tokenizer": authenticated["source_files"]["tokenizer.json"],
        "tokenizer_config": authenticated["source_files"]["tokenizer_config.json"],
        "image": authenticated["image"],
        "image_contract": authenticated["image_contract"],
        "validation_report": authenticated["validation_report"],
    }
    if any(bindings.get(key) != value for key, value in expected_bindings.items()):
        raise RuntimeError("accepted diagnostic Instruct artifact binding changed")
    if bindings.get("chat_template") != {
        "sha256": DIAGNOSTIC_INSTRUCT_CHAT_TEMPLATE_SHA256,
        "encoding": "UTF-8",
        "applied": True,
        "add_generation_prompt": True,
    }:
        raise RuntimeError("accepted diagnostic Instruct chat-template binding changed")

    source = bindings.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("accepted diagnostic Instruct source binding is missing")
    if source.get("host") != {
        "artifact": "tools/ace2_chat_demo.py",
        "bytes": DIAGNOSTIC_INSTRUCT_ACCEPTED_PREPARE_HOST_BYTES,
        "sha256": DIAGNOSTIC_INSTRUCT_ACCEPTED_PREPARE_HOST_SHA256,
    }:
        raise RuntimeError("accepted diagnostic Instruct host-source provenance changed")
    if source.get("stage1_gap_report") != authenticated["gap_report"]:
        raise RuntimeError("accepted diagnostic Instruct gap-report binding changed")

    expected_package_binding = {
        "artifact": "runtime_package.bin",
        "bytes": DIAGNOSTIC_INSTRUCT_ACCEPTED_PACKAGE_BYTES,
        "sha256": DIAGNOSTIC_INSTRUCT_ACCEPTED_PACKAGE_SHA256,
        "format": "ACE2RT2",
        "version": 2,
    }
    if bindings.get("package") != expected_package_binding:
        raise RuntimeError("accepted diagnostic Instruct package binding changed")

    runtime_package = provenance.get("runtime_package")
    if not isinstance(runtime_package, dict) or (
        runtime_package.get("sha256")
        != DIAGNOSTIC_INSTRUCT_ACCEPTED_PACKAGE_SHA256
        or runtime_package.get("bytes")
        != DIAGNOSTIC_INSTRUCT_ACCEPTED_PACKAGE_BYTES
        or runtime_package.get("commands")
        != DIAGNOSTIC_INSTRUCT_ACCEPTED_PACKAGE_COMMANDS
        or runtime_package.get("format") != "ACE2RT2"
        or runtime_package.get("version") != 2
        or runtime_package.get("model_sha256")
        != DIAGNOSTIC_INSTRUCT_SOURCE_SHA256["model.safetensors"]
        or runtime_package.get("image_sha256")
        != DIAGNOSTIC_INSTRUCT_IMAGE_SHA256
    ):
        raise RuntimeError("accepted diagnostic Instruct runtime-package identity changed")
    rtl_binding = runtime_package.get("rtl_binding")
    if not isinstance(rtl_binding, dict) or any(
        not isinstance(rtl_binding.get(key), str)
        or len(rtl_binding[key]) != 64
        for key in (
            "rtl_tree_sha256",
            "constraint_tree_sha256",
        )
    ):
        raise RuntimeError("accepted diagnostic Instruct execution-tree binding changed")
    runtime_source_binding = rtl_binding.get("runtime_simulation_source_binding")
    if (
        not isinstance(runtime_source_binding, dict)
        or not isinstance(runtime_source_binding.get("sha256"), str)
        or len(runtime_source_binding["sha256"]) != 64
    ):
        raise RuntimeError("accepted diagnostic Instruct runtime-source binding changed")

    prompt = provenance.get("prompt")
    hi_hashes = {
        sha256_bytes(b"Hi"),
        sha256_bytes(b"Hi\n"),
        sha256_bytes(b"Hi\r\n"),
    }
    if not isinstance(prompt, dict) or (
        prompt.get("prompt_sha256")
        != DIAGNOSTIC_INSTRUCT_ACCEPTED_PROMPT_SHA256
        or prompt.get("prompt_sha256") in hi_hashes
        or prompt.get("prompt_utf8_bytes") != 58
        or prompt.get("chat_template_token_count") != 34
        or prompt.get("chat_template_sha256")
        != DIAGNOSTIC_INSTRUCT_CHAT_TEMPLATE_SHA256
        or prompt.get("chat_template_applied") is not True
    ):
        raise RuntimeError("accepted diagnostic Instruct non-Hi prompt binding changed")

    execution_policy = provenance.get("execution_policy")
    if not isinstance(execution_policy, dict) or (
        execution_policy.get("mode") != "package_preparation_only"
        or execution_policy.get("execution_commands") != 0
        or execution_policy.get("runtime_binary_required") is not False
        or execution_policy.get("runtime_executed") is not False
    ):
        raise RuntimeError("accepted diagnostic Instruct execution policy changed")

    if (
        preflight.get("schema_version") != 1
        or preflight.get("status") != DIAGNOSTIC_INSTRUCT_PREPARE_STATUS
        or preflight.get("classification")
        != "ace2_diagnostic_instruct_prepare_preflight"
        or preflight.get("requested_mode")
        != "diagnostic_instruct_host_package_preparation_only"
    ):
        raise RuntimeError("accepted diagnostic Instruct preflight changed")
    preflight_execution = preflight.get("execution")
    if not isinstance(preflight_execution, dict) or (
        preflight_execution.get("authorized") is not False
        or preflight_execution.get("product_acceptance_eligible") is not False
        or preflight_execution.get("runtime_executed") is not False
        or preflight_execution.get("generation_executed") is not False
        or preflight_execution.get("decode_executed") is not False
    ):
        raise RuntimeError("accepted diagnostic Instruct preflight authority changed")

    ds32_scope = ds32_applicability.get("scope")
    if (
        ds32_applicability.get("status") != "PASS"
        or not isinstance(ds32_scope, dict)
        or ds32_scope.get("runtime_executed") is not False
        or ds32_scope.get("generation_or_decode_executed") is not False
        or ds32_scope.get("rtl_agreement_claim") is not False
    ):
        raise RuntimeError("accepted diagnostic Instruct DS32 scope changed")


def authenticate_diagnostic_instruct_runtime_binding() -> dict[str, Any]:
    paths = _accepted_diagnostic_instruct_prepare_paths()
    expected_hashes = diagnostic_instruct_runtime_binding_expected_file_hashes()
    accepted_files = {
        key: _verified_file_record(
            paths[key],
            expected_hashes[paths[key]],
            identity=f"accepted diagnostic Instruct {key}",
        )
        for key in ("package", "provenance", "preflight", "ds32")
    }
    if accepted_files["package"]["bytes"] != DIAGNOSTIC_INSTRUCT_ACCEPTED_PACKAGE_BYTES:
        raise RuntimeError("accepted diagnostic Instruct package byte count changed")

    provenance = _load_authenticated_json(
        paths["provenance"],
        identity="accepted diagnostic Instruct provenance",
    )
    preflight = _load_authenticated_json(
        paths["preflight"],
        identity="accepted diagnostic Instruct preflight",
    )
    ds32_applicability = _load_authenticated_json(
        paths["ds32"],
        identity="accepted diagnostic Instruct DS32 applicability",
    )
    authenticated = authenticate_diagnostic_instruct_prepare()
    validate_diagnostic_instruct_runtime_binding_provenance(
        provenance=provenance,
        preflight=preflight,
        ds32_applicability=ds32_applicability,
        authenticated=authenticated,
    )
    verify_diagnostic_instruct_package_binding(
        paths["package"],
        provenance["runtime_package"],
    )
    return {
        "paths": paths,
        "accepted_files": accepted_files,
        "provenance": provenance,
        "preflight": preflight,
        "ds32_applicability": ds32_applicability,
        "authenticated": authenticated,
    }


def path_sorted_tree_binding(
    patterns: str | tuple[str, ...],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    normalized_patterns = (patterns,) if isinstance(patterns, str) else patterns
    paths = sorted(
        {
            path
            for pattern in normalized_patterns
            for path in root.glob(pattern)
            if path.is_file()
        }
    )
    if not paths:
        raise RuntimeError(
            "source tree patterns have no files: " + ", ".join(normalized_patterns)
        )
    source_hashes = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    manifest = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in source_hashes
    ).encode()
    return {
        "file_count": len(source_hashes),
        "sha256": sha256_bytes(manifest),
        "hash_method": (
            "sha256 of UTF-8 path-sorted per-file SHA256 manifest for "
            + ", ".join(normalized_patterns)
        ),
        "patterns": list(normalized_patterns),
        "source_hashes": source_hashes,
    }


def path_sorted_source_binding(
    relative_paths: list[str] | tuple[str, ...],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    paths = sorted(set(relative_paths))
    if not paths:
        raise RuntimeError("source binding path list is empty")
    records = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"source binding input is missing: {relative}")
        records.append({"path": relative, "sha256": sha256_file(path)})
    manifest = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in records
    ).encode()
    return {
        "file_count": len(records),
        "sha256": sha256_bytes(manifest),
        "hash_method": "sha256 of UTF-8 path-sorted explicit per-file SHA256 manifest",
        "source_hashes": records,
    }


def _resolve_rtl_include(root: Path, source: Path, include: str) -> Path:
    candidates = [
        source.parent / include,
        root / "rtl" / include,
        root / "rtl/generated" / include,
    ]
    resolved = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and candidate not in resolved:
            resolved.append(candidate)
    if len(resolved) != 1:
        raise RuntimeError(
            f"RTL include {include!r} from {source.relative_to(root)} resolves to "
            f"{len(resolved)} files"
        )
    return resolved[0]


def live_rtl_source_binding(*, root: Path = ROOT) -> dict[str, Any]:
    binding = path_sorted_tree_binding(RTL_SOURCE_PATTERNS, root=root)
    bound_paths = {record["path"] for record in binding["source_hashes"]}
    include_paths: set[str] = set()
    for record in binding["source_hashes"]:
        source = root / record["path"]
        for line in source.read_text(encoding="utf-8").splitlines():
            match = RTL_INCLUDE_PATTERN.match(line)
            if match is None:
                continue
            included = _resolve_rtl_include(root, source, match.group(1))
            relative = included.relative_to(root).as_posix()
            if relative not in bound_paths:
                raise RuntimeError(f"transitive RTL include is omitted from binding: {relative}")
            include_paths.add(relative)
    scale_table_path = SILU_INPUT_SCALE_TABLE_RTL.relative_to(ROOT).as_posix()
    if root == ROOT and scale_table_path not in bound_paths:
        raise RuntimeError("SiLU input scale table is omitted from live RTL binding")
    return {
        **binding,
        "transitive_include_paths": sorted(include_paths),
        "transitive_include_count": len(include_paths),
    }


def silu_scale_source_binding(*, root: Path = ROOT) -> dict[str, Any]:
    records = []
    for relative, expected_sha256 in sorted(SILU_SCALE_BOUND_SHA256.items()):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"bound SiLU scale input is missing: {relative}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"bound SiLU scale input differs: {relative}")
        records.append(
            {
                "path": relative,
                "sha256": actual_sha256,
            }
        )
    table_relative = SILU_INPUT_SCALE_TABLE.relative_to(ROOT).as_posix()
    table = json.loads((root / table_relative).read_text(encoding="utf-8"))
    if table.get("generator") != SILU_INPUT_SCALE_GENERATOR.relative_to(ROOT).as_posix():
        raise RuntimeError("generated SiLU scale table generator binding differs")
    if table.get("source") != SILU_SCALE_SOURCE.relative_to(ROOT).as_posix():
        raise RuntimeError("generated SiLU scale table source binding differs")
    if table.get("source_sha256") != SILU_SCALE_BOUND_SHA256[table["source"]]:
        raise RuntimeError("generated SiLU scale table source hash differs")
    layers = table.get("layers")
    if not isinstance(layers, list) or [item.get("layer_id") for item in layers] != list(range(24)):
        raise RuntimeError("generated SiLU scale table layer inventory differs")
    manifest = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in records
    ).encode()
    return {
        "file_count": len(records),
        "sha256": sha256_bytes(manifest),
        "hash_method": "sha256 of fixed path-sorted accepted SiLU scale-source manifest",
        "source_hashes": records,
        "accepted_layer_count": len(layers),
    }


def dynamic_scale32_model_identity(model_sha256: str) -> int:
    try:
        digest = bytes.fromhex(model_sha256)
    except ValueError as error:
        raise RuntimeError("ACE2RT2 model SHA-256 is not hexadecimal") from error
    if len(digest) != 32:
        raise RuntimeError("ACE2RT2 model SHA-256 width differs")
    return int.from_bytes(digest[:8], "little")


def dynamic_scale32_host_plan_tag(sequence_position: int) -> int:
    if not 0 <= sequence_position < MAX_SEQUENCE_POSITIONS:
        raise RuntimeError("dynamic Scale32 host plan position is outside 0..32767")
    return sequence_position


def _validate_dynamic_scale32_sidecar(
    sidecar: bytes,
    *,
    payload_addr: int,
    model_identity: int,
) -> None:
    if len(sidecar) != 64:
        raise RuntimeError("ACE2RT2 dynamic Scale32 sidecar width differs")
    if payload_addr < 64 or payload_addr % 64:
        raise RuntimeError("ACE2RT2 dynamic Scale32 payload address is invalid")
    if sidecar[0:4] != b"BFP1" or sidecar[4] != 1:
        raise RuntimeError("ACE2RT2 dynamic Scale32 sidecar magic or schema differs")
    group_lanes = sidecar[5]
    group_count = sidecar[6]
    tensor_elements = int.from_bytes(sidecar[12:16], "little")
    if group_lanes not in (64, 128) or not 1 <= group_count <= 38:
        raise RuntimeError("ACE2RT2 dynamic Scale32 sidecar shape differs")
    expected_groups = (tensor_elements + group_lanes - 1) // group_lanes
    if tensor_elements == 0 or expected_groups != group_count or sidecar[7] != 0:
        raise RuntimeError("ACE2RT2 dynamic Scale32 sidecar shape differs")
    if int.from_bytes(sidecar[16:24], "little") != model_identity:
        raise RuntimeError("ACE2RT2 dynamic Scale32 sidecar model identity differs")
    if sidecar[62:64] != b"\0\0" or any(sidecar[24 + group_count : 62]):
        raise RuntimeError("ACE2RT2 dynamic Scale32 sidecar reserved bytes differ")
    for raw_delta in sidecar[24 : 24 + group_count]:
        delta = raw_delta - 256 if raw_delta & 0x80 else raw_delta
        if not -24 <= delta <= 24:
            raise RuntimeError("ACE2RT2 dynamic Scale32 sidecar delta is outside [-24,24]")


def _serialize_dynamic_scale32_sidecars(
    records: list[dict[str, Any]] | None,
    *,
    model_identity: int,
) -> bytes:
    raw = bytearray()
    seen_bindings: set[tuple[int, int, int, int]] = set()
    for index, record in enumerate(records or []):
        try:
            payload_addr = int(record["payload_addr"])
            sidecar = bytes(record["sidecar"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"ACE2RT2 dynamic Scale32 record {index} is not encodable"
            ) from error
        _validate_dynamic_scale32_sidecar(
            sidecar,
            payload_addr=payload_addr,
            model_identity=model_identity,
        )
        binding = (
            payload_addr,
            int.from_bytes(sidecar[8:10], "little"),
            sidecar[10],
            sidecar[11],
        )
        if binding in seen_bindings:
            raise RuntimeError("ACE2RT2 dynamic Scale32 producer binding is duplicated")
        seen_bindings.add(binding)
        raw.extend(PACKAGE_V2_DYNAMIC_RECORD.pack(payload_addr, sidecar))
    return bytes(raw)


def _validate_dynamic_scale32_initial_binding(
    records: list[dict[str, Any]] | None,
    commands: list[dict[str, Any]],
    prompt_token_count: int,
) -> None:
    dynamic_commands = [
        command for command in commands if int(command["flags"]) & DYNAMIC_SCALE32_FLAG
    ]
    sidecars = records or []
    if not sidecars:
        if dynamic_commands:
            raise RuntimeError(
                "ACE2RT2 flags[6] initial boundary lacks its DS32 sidecar"
            )
        return
    if len(sidecars) != prompt_token_count:
        raise RuntimeError(
            "ACE2RT2 prompt DS32 tranche requires one host plan per prompt position"
        )
    expected_dynamic = [
        command
        for command in commands
        if int(command["token_step"]) < prompt_token_count
        and int(command["layer_id"]) == 0
        and command.get("operator") in DYNAMIC_SCALE32_PROMPT_OPERATORS
    ]
    if dynamic_commands != expected_dynamic or len(dynamic_commands) != 4 * prompt_token_count:
        raise RuntimeError(
            "ACE2RT2 flags[6] is only legal on prompt-position layer-0 RMS/Q/K/V"
        )

    for position, record in enumerate(sidecars):
        tranche = [
            command
            for command in expected_dynamic
            if int(command["token_step"]) == position
        ]
        if [command.get("operator") for command in tranche] != list(
            DYNAMIC_SCALE32_PROMPT_OPERATORS
        ):
            raise RuntimeError("ACE2RT2 prompt DS32 operator order differs")
        rms, q_proj, k_proj, v_proj = tranche
        if any(int(command["flags"]) != DYNAMIC_SCALE32_FROZEN_FLAGS for command in tranche):
            raise RuntimeError("ACE2RT2 prompt DS32 command flags differ")
        if (
            int(rms["m"]) != 1
            or int(rms["n"]) != 896
            or int(rms["k"]) != 0
            or int(rms["src0_addr"]) != 0x0000001000000000
            or int(rms["src1_addr"]) != 0
            or int(rms["dst_addr"]) != 0x0000001000000700
            or int(rms["scale_addr"]) != 0x0000000300000000
            or int(rms["scratch_addr"]) != 0
        ):
            raise RuntimeError("ACE2RT2 prompt DS32 RMS command contract differs")
        projection_contracts = (
            (q_proj, 896, 0x0000000100000000, 0x0000001000000A80, 0x0000000200000000),
            (k_proj, 128, 0x0000000100062000, 0x0000001000000E00, 0x0000000200003800),
            (v_proj, 128, 0x0000000100070000, 0x0000001000000E80, 0x0000000200004000),
        )
        for command, outputs, weights, destination, metadata in projection_contracts:
            if (
                int(command["m"]) != 1
                or int(command["n"]) != outputs
                or int(command["k"]) != 896
                or int(command["src0_addr"]) != int(rms["dst_addr"])
                or int(command["src1_addr"]) != weights
                or int(command["dst_addr"]) != destination
                or int(command["scale_addr"]) != metadata
                or int(command["scratch_addr"]) != 0
            ):
                raise RuntimeError("ACE2RT2 prompt DS32 Q/K/V consumer contract differs")

        try:
            payload_addr = int(record["payload_addr"])
            sidecar = bytes(record["sidecar"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("ACE2RT2 prompt DS32 host plan is not encodable") from error
        if payload_addr != int(rms["src0_addr"]):
            raise RuntimeError("ACE2RT2 prompt DS32 host-plan payload binding differs")
        if (
            len(sidecar) != 64
            or sidecar[5] != DYNAMIC_SCALE32_INITIAL_GROUP_LANES
            or sidecar[6] != DYNAMIC_SCALE32_INITIAL_GROUP_COUNT
            or int.from_bytes(sidecar[8:10], "little")
            != dynamic_scale32_host_plan_tag(position)
            or sidecar[10] != DYNAMIC_SCALE32_HOST_PLAN_LAYER
            or sidecar[11] != DYNAMIC_SCALE32_HOST_PLAN_OPCODE
            or int.from_bytes(sidecar[12:16], "little")
            != DYNAMIC_SCALE32_INITIAL_ELEMENTS
        ):
            raise RuntimeError("ACE2RT2 prompt DS32 host-plan producer contract differs")


def _apply_prompt_dynamic_scale32_flags(
    commands: list[dict[str, Any]], prompt_token_count: int
) -> None:
    for command in commands:
        if (
            int(command["token_step"]) < prompt_token_count
            and int(command["layer_id"]) == 0
            and command.get("operator") in DYNAMIC_SCALE32_PROMPT_OPERATORS
        ):
            command["flags"] = int(command["flags"]) | DYNAMIC_SCALE32_FLAG


def _build_prompt_dynamic_scale32_sidecar(
    *,
    payload_addr: int,
    model_identity: int,
    sequence_position: int,
    deltas: list[int] | tuple[int, ...],
) -> bytes:
    if payload_addr < 64 or payload_addr % 64:
        raise RuntimeError("ACE2RT2 initial dynamic Scale32 payload address is invalid")
    sidecar = bytearray(64)
    sidecar[0:4] = b"BFP1"
    sidecar[4] = 1
    sidecar[5] = DYNAMIC_SCALE32_INITIAL_GROUP_LANES
    sidecar[6] = DYNAMIC_SCALE32_INITIAL_GROUP_COUNT
    sidecar[8:10] = dynamic_scale32_host_plan_tag(sequence_position).to_bytes(2, "little")
    sidecar[10] = DYNAMIC_SCALE32_HOST_PLAN_LAYER
    sidecar[11] = DYNAMIC_SCALE32_HOST_PLAN_OPCODE
    sidecar[12:16] = DYNAMIC_SCALE32_INITIAL_ELEMENTS.to_bytes(4, "little")
    sidecar[16:24] = model_identity.to_bytes(8, "little")
    if len(deltas) != DYNAMIC_SCALE32_INITIAL_GROUP_COUNT:
        raise RuntimeError("ACE2RT2 prompt DS32 host plan requires seven deltas")
    for group, delta in enumerate(deltas):
        if not -24 <= int(delta) <= 24:
            raise RuntimeError("ACE2RT2 prompt DS32 host-plan delta is outside [-24,24]")
        sidecar[24 + group] = int(delta) & 0xFF
    return bytes(sidecar)


def write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_non_whitespace(value: str, *, identity: str) -> str:
    if not value.strip():
        raise RuntimeError(f"{identity} must contain non-whitespace UTF-8 text")
    return value


def _decode_utf8(raw: bytes, *, identity: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{identity} must contain valid UTF-8") from error


def read_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt_file is not None:
        value = _decode_utf8(prompt_file.read_bytes(), identity="prompt file")
        return _require_non_whitespace(value, identity="prompt")
    elif prompt is not None:
        value = prompt
    elif not sys.stdin.isatty():
        value = sys.stdin.read()
    else:
        value = input("User: ")
    value = value.strip()
    return _require_non_whitespace(value, identity="prompt")


def single_turn_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
    _require_non_whitespace(prompt, identity="prompt")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def normalize_conversation_messages(
    raw_messages: Any,
    *,
    default_system_prompt: str,
) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise RuntimeError("conversation messages must be a non-empty array")

    messages: list[dict[str, str]] = []
    for index, item in enumerate(raw_messages):
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise RuntimeError(
                f"conversation message {index} must contain exactly role and content"
            )
        role = item["role"]
        content = item["content"]
        if not isinstance(role, str) or not isinstance(content, str):
            raise RuntimeError(
                f"conversation message {index} role and content must be strings"
            )
        if role not in {"system", "user", "assistant"}:
            raise RuntimeError(f"conversation message {index} has an unknown role")
        _require_non_whitespace(
            content,
            identity=f"conversation message {index} content",
        )
        messages.append({"role": role, "content": content})

    body_start = 0
    if messages[0]["role"] == "system":
        body_start = 1
    else:
        messages.insert(0, {"role": "system", "content": default_system_prompt})
        body_start = 1

    body = messages[body_start:]
    if not body:
        raise RuntimeError("conversation must contain at least one user message")
    for offset, message in enumerate(body):
        expected_role = "user" if offset % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            raise RuntimeError(
                "conversation roles must alternate user/assistant beginning with user"
            )
    if body[-1]["role"] != "user":
        raise RuntimeError("conversation must end with a user message")
    return messages


def read_conversation_file(
    conversation_file: Path,
    *,
    default_system_prompt: str,
) -> list[dict[str, str]]:
    raw = conversation_file.read_bytes()
    text = _decode_utf8(raw, identity="conversation file")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError("conversation file must contain valid JSON") from error
    if not isinstance(document, dict) or set(document) != {"messages"}:
        raise RuntimeError(
            "conversation file must contain exactly one messages field"
        )
    return normalize_conversation_messages(
        document["messages"],
        default_system_prompt=default_system_prompt,
    )


def validate_max_new_tokens(value: int) -> int:
    if not MIN_NEW_TOKENS <= value <= MAX_NEW_TOKENS:
        raise RuntimeError(
            f"--max-new-tokens must be between {MIN_NEW_TOKENS} and {MAX_NEW_TOKENS}"
        )
    return value


def load_tokenizer(snapshot: Path = TOKENIZER_SNAPSHOT) -> Any:
    required = ["tokenizer.json", "tokenizer_config.json"]
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeError("pinned local tokenizer snapshot is incomplete: " + ", ".join(missing))
    return AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )


def tokenizer_identity_record(snapshot: Path = TOKENIZER_SNAPSHOT) -> bytes:
    for name, expected_sha256 in TOKENIZER_FILE_SHA256.items():
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError(f"pinned tokenizer identity file is missing: {name}")
        if sha256_file(path) != expected_sha256:
            raise RuntimeError(f"pinned tokenizer identity file differs: {name}")
    return (
        "ace2-tokenizer-identity-v1\n"
        f"repository={TOKENIZER_REPOSITORY}\n"
        f"revision={REVISION}\n"
        f"tokenizer.json.sha256={TOKENIZER_FILE_SHA256['tokenizer.json']}\n"
        "tokenizer_config.json.sha256="
        f"{TOKENIZER_FILE_SHA256['tokenizer_config.json']}\n"
    ).encode("utf-8")


def diagnostic_instruct_tokenizer_identity_record() -> bytes:
    return (
        "ace2-tokenizer-identity-v2\n"
        f"repository={DIAGNOSTIC_INSTRUCT_REPOSITORY}\n"
        f"revision={DIAGNOSTIC_INSTRUCT_REVISION}\n"
        "tokenizer.json.sha256="
        f"{DIAGNOSTIC_INSTRUCT_SOURCE_SHA256['tokenizer.json']}\n"
        "tokenizer_config.json.sha256="
        f"{DIAGNOSTIC_INSTRUCT_SOURCE_SHA256['tokenizer_config.json']}\n"
        f"chat_template.sha256={DIAGNOSTIC_INSTRUCT_CHAT_TEMPLATE_SHA256}\n"
    ).encode("utf-8")


def verify_loaded_diagnostic_instruct_tokenizer(
    tokenizer: Any,
    chat_template: str,
) -> None:
    loaded_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(loaded_template, str):
        raise RuntimeError("loaded pinned Instruct tokenizer has no chat template")
    if loaded_template != chat_template:
        raise RuntimeError("loaded pinned Instruct tokenizer chat template differs")
    if sha256_bytes(loaded_template.encode("utf-8")) != (
        DIAGNOSTIC_INSTRUCT_CHAT_TEMPLATE_SHA256
    ):
        raise RuntimeError("loaded pinned Instruct tokenizer chat template SHA-256 changed")


def _conversation_identity_bytes(messages: list[dict[str, str]]) -> bytes:
    raw = bytearray(b"ace2-conversation-v1\0")
    for message in messages:
        role = message["role"].encode("utf-8")
        content = message["content"].encode("utf-8")
        raw.extend(struct.pack("<II", len(role), len(content)))
        raw.extend(role)
        raw.extend(content)
    return bytes(raw)


def tokenize_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    chat_template: str | None = None,
) -> dict[str, Any]:
    if not messages or messages[-1].get("role") != "user":
        raise RuntimeError("tokenization requires a conversation ending in user")
    user_id_groups = [
        list(tokenizer.encode(message["content"], add_special_tokens=False))
        for message in messages
        if message["role"] == "user"
    ]
    if not user_id_groups or any(not ids for ids in user_id_groups):
        raise RuntimeError("tokenizer produced no user-content tokens")
    final_user_ids = user_id_groups[-1]
    if chat_template is None:
        chat_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        chat_ids = tokenizer.apply_chat_template(
            messages,
            chat_template=chat_template,
            tokenize=True,
            add_generation_prompt=True,
        )
    if hasattr(chat_ids, "tolist"):
        chat_ids = chat_ids.tolist()
    if chat_ids and isinstance(chat_ids[0], list):
        if len(chat_ids) != 1:
            raise RuntimeError("unexpected batched chat-template tokenization")
        chat_ids = chat_ids[0]
    chat_ids = [int(token) for token in chat_ids]
    if not chat_ids:
        raise RuntimeError("chat template produced no tokens")
    final_user_raw = messages[-1]["content"].encode("utf-8")
    conversation_raw = _conversation_identity_bytes(messages)
    result = {
        "prompt_sha256": sha256_bytes(final_user_raw),
        "prompt_utf8_bytes": len(final_user_raw),
        "conversation_sha256": sha256_bytes(conversation_raw),
        "conversation_utf8_bytes": sum(
            len(message["content"].encode("utf-8")) for message in messages
        ),
        "message_count": len(messages),
        "user_message_count": len(user_id_groups),
        "assistant_message_count": sum(
            message["role"] == "assistant" for message in messages
        ),
        "user_token_count": sum(len(ids) for ids in user_id_groups),
        "final_user_token_count": len(final_user_ids),
        "chat_template_token_count": len(chat_ids),
        "chat_template_token_ids": chat_ids,
        "last_user_content_token_id": int(final_user_ids[-1]),
    }
    if chat_template is not None:
        result["chat_template_sha256"] = sha256_bytes(
            chat_template.encode("utf-8")
        )
        result["chat_template_applied"] = True
        result["add_generation_prompt"] = True
    return result


def tokenize_prompt(
    tokenizer: Any,
    prompt: str,
    system_prompt: str,
    *,
    chat_template: str | None = None,
) -> dict[str, Any]:
    return tokenize_messages(
        tokenizer,
        single_turn_messages(prompt, system_prompt),
        chat_template=chat_template,
    )


def load_diagnostic_instruct_schedule_inputs(
    authenticated: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], int, list[int]]:
    schedule_record = _verified_file_record(
        accepted_runtime.SCHEDULE,
        accepted_runtime.EXPECTED_SCHEDULE_SHA256,
        identity="accepted ACE2RT2 template schedule",
    )
    if schedule_record["bytes"] != accepted_runtime.EXPECTED_SCHEDULE_BYTES:
        raise RuntimeError("accepted ACE2RT2 template schedule byte count changed")
    runtime_source = _verified_file_record(
        ROOT / "tools/run_full_qwen_command_schedule_runtime.py",
        DIAGNOSTIC_INSTRUCT_RUNTIME_SOURCE_SHA256,
        identity="accepted ACE2RT2 schedule source",
    )
    if not accepted_runtime.SCHEDULE_VALIDATION.is_file():
        raise RuntimeError("accepted ACE2RT2 schedule validation is missing")
    schedule_validation = json.loads(
        accepted_runtime.SCHEDULE_VALIDATION.read_text(encoding="utf-8")
    )
    if schedule_validation != {
        "first_command": "input_rmsnorm",
        "kv_positions_per_layer": [0, 1],
        "last_command": "lm_head_tile",
        "layers": 24,
        "lm_head_tiles_per_token": 4748,
        "status": "PASS",
        "validated_commands": accepted_runtime.EXPECTED_COMMANDS,
    }:
        raise RuntimeError("accepted ACE2RT2 schedule validation changed")
    validation_record = {
        "artifact": _artifact_label(accepted_runtime.SCHEDULE_VALIDATION),
        "bytes": accepted_runtime.SCHEDULE_VALIDATION.stat().st_size,
        "sha256": sha256_file(accepted_runtime.SCHEDULE_VALIDATION),
    }
    schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
    if (
        schedule.get("command_count") != accepted_runtime.EXPECTED_COMMANDS
        or schedule.get("command_count_per_token")
        != accepted_runtime.EXPECTED_TOKEN_COUNTS
        or len(schedule.get("commands", [])) != accepted_runtime.EXPECTED_COMMANDS
    ):
        raise RuntimeError("accepted ACE2RT2 template schedule changed")
    for ordinal, command in enumerate(schedule["commands"]):
        if (
            command.get("ordinal") != ordinal
            or command.get("completion_tag") != (ordinal & 0xFFFF)
            or command.get("operator") not in accepted_runtime.OPERATOR_IDS
        ):
            raise RuntimeError(
                f"accepted ACE2RT2 template schedule differs at command {ordinal}"
            )
    embedding_offset, embedding_shape = accepted_runtime.embedding_tensor_offset(
        authenticated["model_path"]
    )
    return (
        schedule,
        {
            "accepted_schedule": schedule_record,
            "schedule_validation": validation_record,
            "schedule_source": runtime_source,
        },
        embedding_offset,
        embedding_shape,
    )


def diagnostic_instruct_prepare_preflight(
    authenticated: dict[str, Any],
    schedule_inputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "classification": "ace2_diagnostic_instruct_prepare_preflight",
        "status": DIAGNOSTIC_INSTRUCT_PREPARE_STATUS,
        "requested_mode": "diagnostic_instruct_host_package_preparation_only",
        "model": {
            "repository": authenticated["repository"],
            "revision": authenticated["revision"],
            "config": authenticated["source_files"]["config.json"],
            "model_safetensors": authenticated["source_files"]["model.safetensors"],
        },
        "tokenizer": {
            "tokenizer_json": authenticated["source_files"]["tokenizer.json"],
            "tokenizer_config": authenticated["source_files"][
                "tokenizer_config.json"
            ],
            "chat_template_sha256": authenticated["chat_template_sha256"],
            "chat_template_applied": True,
            "add_generation_prompt": True,
        },
        "w4a8_image": {
            "image": authenticated["image"],
            "contract": authenticated["image_contract"],
            "validation_report": authenticated["validation_report"],
        },
        "source": {
            "stage1_gap_report": authenticated["gap_report"],
            **schedule_inputs,
        },
        "execution": {
            "authorized": False,
            "mode": "diagnostic_only_prepare",
            "product_acceptance_eligible": False,
            "runtime_executed": False,
            "generation_executed": False,
            "decode_executed": False,
        },
        "claims": {
            "completion_claim": False,
            "rtl_agreement_claim": False,
            "latency_claim": False,
            "fpga_or_u280_claim": False,
            "stage_advancement_claim": False,
        },
    }


def pinned_chat_product_preflight(
    *,
    observed_model_sha256: str,
    diagnostic_reason: str | None,
    audit_path: Path = PINNED_CHAT_SEMANTICS_AUDIT,
) -> dict[str, Any]:
    if not audit_path.is_file():
        raise RuntimeError(
            "missing authenticated pinned-Base chat-semantics audit: "
            f"{audit_path.name}"
        )
    audit_sha256 = sha256_file(audit_path)
    if audit_sha256 != PINNED_CHAT_SEMANTICS_AUDIT_SHA256:
        raise RuntimeError("pinned-Base chat-semantics audit SHA-256 changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != 1:
        raise RuntimeError("pinned-Base chat-semantics audit schema changed")
    if audit.get("classification") != "reviewer_bf16_production_chat_contract_audit":
        raise RuntimeError("pinned-Base chat-semantics audit classification changed")
    if audit.get("status") != PINNED_CHAT_SEMANTICS_AUDIT_STATUS:
        raise RuntimeError("pinned-Base chat-semantics audit status changed")
    model = audit.get("model")
    expected_model = {
        "repository": TOKENIZER_REPOSITORY,
        "revision": REVISION,
        "model_safetensors_sha256": accepted_runtime.EXPECTED_MODEL_SHA256,
    }
    if not isinstance(model, dict) or any(
        model.get(key) != value for key, value in expected_model.items()
    ):
        raise RuntimeError("pinned-Base chat-semantics audit model identity changed")
    if observed_model_sha256 != accepted_runtime.EXPECTED_MODEL_SHA256:
        raise RuntimeError("pinned model safetensors SHA-256 changed")
    contract = audit.get("contract")
    if not isinstance(contract, dict) or any(
        contract.get(key) != value
        for key, value in {
            "chat_template": True,
            "add_generation_prompt": True,
            "greedy_argmax": True,
            "max_new_tokens": 8,
            "termination_token_ids": list(TERMINATION_TOKEN_IDS),
            "use_cache": True,
            "acceptance_candidate": False,
            "accelerator_or_w4a8_executed": False,
        }.items()
    ):
        raise RuntimeError("pinned-Base chat-semantics audit contract changed")
    cases = audit.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise RuntimeError("pinned-Base chat-semantics audit cases changed")
    for case_name, case in cases.items():
        if not isinstance(case, dict):
            raise RuntimeError(
                f"pinned-Base chat-semantics audit case changed: {case_name}"
            )
        generated_token_ids = case.get("generated_token_ids")
        if (
            not isinstance(generated_token_ids, list)
            or len(generated_token_ids) < 2
            or any(int(token) in TERMINATION_TOKEN_IDS for token in generated_token_ids)
            or not str(case.get("decoded_text", "")).strip()
        ):
            raise RuntimeError(
                "pinned-Base audit no longer establishes visible multi-token "
                f"completion feasibility: {case_name}"
            )
    product_candidate = diagnostic_reason is None
    hardware_contract = runtime_preflight(
        model_id="qwen2.5-0.5b",
        embedding_shape=[151936, 896],
        max_sequence_positions=MAX_SEQUENCE_POSITIONS,
        kv_bytes_per_token_per_layer=KV_BYTES_PER_TOKEN,
    )
    try:
        artifact = audit_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        artifact = audit_path.name
    return {
        "schema_version": 1,
        "classification": "ace2_chat_product_model_preflight",
        "status": PINNED_BASE_COMPLETION_CONTRACT_STATUS,
        "requested_product_mode": "readable_multi_token_base_completion",
        "model": expected_model,
        "model_hardware_contract": hardware_contract,
        "contract": {
            "semantic_scope": "decoded continuation readability only",
            "instruction_following_claim": False,
            "minimum_visible_nonterminating_tokens": 2,
            "greedy_argmax": True,
            "termination_token_ids": list(TERMINATION_TOKEN_IDS),
            "accelerator_and_quantized_reference_required": True,
        },
        "evidence": {
            "artifact": artifact,
            "sha256": audit_sha256,
            "status": audit["status"],
            "observed_cases": len(cases),
            "interpretation": (
                "feasibility evidence only; the retained BF16 audit is not "
                "accelerator or W4A8 product acceptance"
            ),
        },
        "execution": {
            "authorized": True,
            "mode": "product_candidate" if product_candidate else "diagnostic_only",
            "diagnostic_reason": diagnostic_reason,
            "product_acceptance_eligible": product_candidate,
        },
        "required_runtime_evidence": (
            "complete ACE2RT2 execution, readable multi-token decode, and passing "
            "declared quantized-reference boundaries"
        ),
    }


def patch_package_seed(package_path: Path, seed_token_id: int) -> dict[str, Any]:
    raw = bytearray(package_path.read_bytes())
    if len(raw) < PACKAGE_SEED_OFFSET + 4 or bytes(raw[:8]) != PACKAGE_MAGIC:
        raise RuntimeError("accepted runtime package header is invalid")
    accepted_seed = struct.unpack_from("<I", raw, PACKAGE_SEED_OFFSET)[0]
    struct.pack_into("<I", raw, PACKAGE_SEED_OFFSET, seed_token_id)
    write_atomic(package_path, bytes(raw))
    return {
        "accepted_schedule_seed_token_id": accepted_seed,
        "prompt_seed_token_id": seed_token_id,
        "bytes": len(raw),
        "sha256": sha256_bytes(bytes(raw)),
        "mutation": "package_header_seed_token_only",
    }


def nested_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            if item_key == key:
                found.append(item_value)
            found.extend(nested_values(item_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(nested_values(item, key))
    return found


def certified_rtl_binding() -> dict[str, Any]:
    certificate_raw = FINAL_CERTIFICATE.read_bytes()
    certificate_sha = sha256_bytes(certificate_raw)
    sum_fields = FINAL_CERTIFICATE_SUM.read_text(encoding="utf-8").split()
    if len(sum_fields) < 1 or sum_fields[0] != certificate_sha:
        raise RuntimeError("final product certificate checksum companion differs")
    certificate = json.loads(certificate_raw)
    if certificate.get("certificate_status") != "CERTIFIED" or certificate.get("decision") != "FORMAL_PRODUCT_CERTIFIED":
        raise RuntimeError("final product certificate is not in the certified state")

    manifest_raw = FINAL_RTL_TREE_MANIFEST.read_bytes()
    certified_rtl_tree_sha = sha256_bytes(manifest_raw)
    bound_tree_hashes = {
        str(item)
        for key in ("rtl_tree_sha256", "publication_rtl_tree_sha256")
        for item in nested_values(certificate, key)
    }
    if certified_rtl_tree_sha not in bound_tree_hashes:
        raise RuntimeError("final RTL tree manifest is not bound by the product certificate")

    shell_hashes = []
    for line in manifest_raw.decode("utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == "rtl/ace2_shell.sv":
            shell_hashes.append(fields[0])
    if len(shell_hashes) != 1:
        raise RuntimeError("final RTL tree manifest lacks one canonical shell record")
    certified_shell_sha = shell_hashes[0]
    live_shell_sha = sha256_file(ROOT / "rtl/ace2_shell.sv")
    live_matches_certified = live_shell_sha == certified_shell_sha
    live_rtl_tree = live_rtl_source_binding()
    runtime_simulation_sources = path_sorted_source_binding(
        [
            *(record["path"] for record in live_rtl_tree["source_hashes"]),
            *RUNTIME_SIMULATION_SOURCE_PATHS,
        ]
    )
    live_constraint_tree = path_sorted_tree_binding("constraints/**/*")
    scale_source_binding = silu_scale_source_binding()
    return {
        "certificate_sha256": certificate_sha,
        "certified_rtl_tree_sha256": certified_rtl_tree_sha,
        "certified_rtl_tree_manifest": FINAL_RTL_TREE_MANIFEST.relative_to(ROOT).as_posix(),
        "rtl_tree_sha256": live_rtl_tree["sha256"],
        "rtl_tree_file_count": live_rtl_tree["file_count"],
        "rtl_tree_hash_method": live_rtl_tree["hash_method"],
        "rtl_tree_patterns": live_rtl_tree["patterns"],
        "rtl_source_hashes": live_rtl_tree["source_hashes"],
        "rtl_transitive_include_count": live_rtl_tree["transitive_include_count"],
        "rtl_transitive_include_paths": live_rtl_tree["transitive_include_paths"],
        "runtime_simulation_source_binding": runtime_simulation_sources,
        "silu_scale_source_binding": scale_source_binding,
        "constraint_tree_sha256": live_constraint_tree["sha256"],
        "constraint_tree_file_count": live_constraint_tree["file_count"],
        "constraint_tree_hash_method": live_constraint_tree["hash_method"],
        "constraint_source_hashes": live_constraint_tree["source_hashes"],
        "certified_shell_sha256": certified_shell_sha,
        "live_shell_sha256": live_shell_sha,
        "live_shell_matches_certified_baseline": live_matches_certified,
        "runtime_classification": (
            "certificate_bound_baseline"
            if live_matches_certified
            else "post_certificate_focused_runtime_repair"
        ),
    }


def _append_rt2_command(
    commands: list[dict[str, Any]],
    template: dict[str, Any],
    *,
    token_step: int,
    **updates: Any,
) -> None:
    command = dict(template)
    command.update(updates)
    command["ordinal"] = len(commands)
    command["completion_tag"] = len(commands) & 0xFFFF
    command["token_step"] = token_step
    command["sequence_position"] = token_step
    commands.append(command)


def _validate_qkv_trio(
    q_proj: dict[str, Any],
    k_proj: dict[str, Any],
    v_proj: dict[str, Any],
) -> None:
    trio = (q_proj, k_proj, v_proj)
    if [command.get("operator") for command in trio] != [
        "q_proj",
        "k_proj",
        "v_proj",
    ]:
        raise RuntimeError("ACE2RT2 Q/K/V operator order differs")
    common_fields = (
        "token_step",
        "layer_id",
        "sequence_position",
        "src0_addr",
        "flags",
    )
    for field in common_fields:
        if len({int(command[field]) for command in trio}) != 1:
            raise RuntimeError(f"ACE2RT2 Q/K/V {field} differs")
    if any(int(command["opcode"]) != 1 for command in trio):
        raise RuntimeError("ACE2RT2 Q/K/V legacy opcode differs")
    if [
        (int(command["m"]), int(command["n"]), int(command["k"]))
        for command in trio
    ] != [(1, 896, 896), (1, 128, 896), (1, 128, 896)]:
        raise RuntimeError("ACE2RT2 Q/K/V geometry differs")
    if any(
        command.get(field) is not None
        for command in trio
        for field in ("query_head", "context_token", "vocab_tile")
    ):
        raise RuntimeError("ACE2RT2 Q/K/V selector fields differ")
    if any(int(command["scratch_addr"]) != 0 for command in trio):
        raise RuntimeError("ACE2RT2 Q/K/V scratch address differs")

    weight_addresses = tuple(int(command["src1_addr"]) for command in trio)
    metadata_addresses = tuple(int(command["scale_addr"]) for command in trio)
    output_addresses = tuple(int(command["dst_addr"]) for command in trio)
    for addresses, offsets, identity in (
        (weight_addresses, FUSED_QKV_WEIGHT_OFFSETS, "weight"),
        (metadata_addresses, FUSED_QKV_METADATA_OFFSETS, "metadata"),
        (output_addresses, FUSED_QKV_OUTPUT_OFFSETS, "output"),
    ):
        base = addresses[0]
        if any(address != base + offset for address, offset in zip(addresses, offsets)):
            raise RuntimeError(f"ACE2RT2 Q/K/V {identity} bases are not contiguous")
        if any(address <= 0 or address & 0xF for address in addresses):
            raise RuntimeError(f"ACE2RT2 Q/K/V {identity} base is not 16-byte aligned")

    src0 = int(q_proj["src0_addr"])
    if src0 != FUSED_QKV_ACTIVATION_ADDR or output_addresses[0] != FUSED_QKV_OUTPUT_ADDR:
        raise RuntimeError("ACE2RT2 Q/K/V activation or output base differs")
    image_file_offset(weight_addresses[0], FUSED_QKV_WEIGHT_BYTES)
    image_file_offset(metadata_addresses[0], FUSED_QKV_METADATA_BYTES)


def _validate_fused_qkv_descriptor(command: dict[str, Any]) -> None:
    layer_id = int(command["layer_id"])
    expected_weight_base = (
        FUSED_QKV_WEIGHT_BASE + layer_id * FUSED_QKV_WEIGHT_LAYER_STRIDE
    )
    expected_metadata_base = (
        FUSED_QKV_METADATA_BASE + layer_id * FUSED_QKV_METADATA_LAYER_STRIDE
    )
    if (
        command.get("operator") != "fused_qkv"
        or int(command["opcode"]) != FUSED_QKV_OPCODE
        or int(command["flags"]) != 0
        or not 0 <= layer_id < 24
        or (int(command["m"]), int(command["n"]), int(command["k"]))
        != (1, 896, 896)
        or int(command["src0_addr"]) != FUSED_QKV_ACTIVATION_ADDR
        or int(command["src1_addr"]) != expected_weight_base
        or int(command["dst_addr"]) != FUSED_QKV_OUTPUT_ADDR
        or int(command["scale_addr"]) != expected_metadata_base
        or int(command["scratch_addr"]) != 0
        or any(
            command.get(field) is not None
            for field in ("query_head", "context_token", "vocab_tile")
        )
    ):
        raise RuntimeError("ACE2RT2 fused-QKV descriptor contract differs")
    addresses = (
        int(command["src0_addr"]),
        int(command["src1_addr"]),
        int(command["dst_addr"]),
        int(command["scale_addr"]),
    )
    if any(address <= 0 or address & 0xF for address in addresses):
        raise RuntimeError("ACE2RT2 fused-QKV address is not 16-byte aligned")
    image_file_offset(int(command["src1_addr"]), FUSED_QKV_WEIGHT_BYTES)
    image_file_offset(int(command["scale_addr"]), FUSED_QKV_METADATA_BYTES)


def _fuse_qkv_trio(
    q_proj: dict[str, Any],
    k_proj: dict[str, Any],
    v_proj: dict[str, Any],
) -> dict[str, Any]:
    _validate_qkv_trio(q_proj, k_proj, v_proj)
    if int(q_proj["flags"]) & DYNAMIC_SCALE32_FLAG:
        raise RuntimeError("ACE2RT2 dynamic-Scale32 Q/K/V cannot be fused")
    fused = dict(q_proj)
    fused.update(
        {
            "operator": "fused_qkv",
            "operator_id": FUSED_QKV_OPERATOR_ID,
            "opcode": FUSED_QKV_OPCODE,
            "flags": 0,
            "m": 1,
            "n": 896,
            "k": 896,
            "weight_tensor": (
                q_proj.get("weight_tensor"),
                k_proj.get("weight_tensor"),
                v_proj.get("weight_tensor"),
            ),
        }
    )
    _validate_fused_qkv_descriptor(fused)
    return fused


def _expect_operator(
    commands: list[dict[str, Any]],
    index: int,
    *,
    token_step: int,
    layer: int,
    operator: str,
    query_head: int | None = None,
    context_token: int | None = None,
    flags: int | None = None,
    vocab_tile: int | None = None,
) -> int:
    if index >= len(commands):
        raise RuntimeError("ACE2RT2 full schedule is truncated")
    command = commands[index]
    expected = {
        "ordinal": index,
        "token_step": token_step,
        "sequence_position": token_step,
        "layer_id": layer,
        "operator": operator,
        "completion_tag": index & 0xFFFF,
        "query_head": query_head,
        "context_token": context_token,
    }
    for key, value in expected.items():
        if command.get(key) != value:
            raise RuntimeError(
                f"ACE2RT2 full schedule differs at command {index}: {key}"
            )
    if flags is not None and int(command["flags"]) != flags:
        raise RuntimeError(f"ACE2RT2 compose phase differs at command {index}")
    if vocab_tile is not None and int(command.get("vocab_tile", -1)) != vocab_tile:
        raise RuntimeError(f"ACE2RT2 LM-head tile differs at command {index}")
    return index + 1


def validate_full_prompt_schedule(
    commands: list[dict[str, Any]],
    *,
    prompt_token_count: int,
    max_new_tokens: int,
    qkv_schedule_mode: str = QKV_SCHEDULE_LEGACY,
    preserve_prompt_layer0_qkv: bool = False,
) -> None:
    if qkv_schedule_mode not in {QKV_SCHEDULE_FUSED, QKV_SCHEDULE_LEGACY}:
        raise RuntimeError("ACE2RT2 QKV schedule mode differs")
    step_count = prompt_token_count + max_new_tokens - 1
    index = 0
    suffix = (
        "o_proj",
        "attention_residual_add",
        "post_attention_rmsnorm",
        "mlp_gate_proj",
        "mlp_up_proj",
        "silu_gate",
        "mlp_down_proj",
        "mlp_residual_add",
    )
    for token_step in range(step_count):
        for layer in range(24):
            index = _expect_operator(
                commands,
                index,
                token_step=token_step,
                layer=layer,
                operator="input_rmsnorm",
            )
            use_fused_qkv = (
                qkv_schedule_mode == QKV_SCHEDULE_FUSED
                and not (
                    preserve_prompt_layer0_qkv
                    and token_step < prompt_token_count
                    and layer == 0
                )
            )
            if use_fused_qkv:
                if index >= len(commands):
                    raise RuntimeError("ACE2RT2 full schedule is truncated")
                _validate_fused_qkv_descriptor(commands[index])
                index = _expect_operator(
                    commands,
                    index,
                    token_step=token_step,
                    layer=layer,
                    operator="fused_qkv",
                )
            else:
                for operator in ("q_proj", "k_proj", "v_proj"):
                    index = _expect_operator(
                        commands,
                        index,
                        token_step=token_step,
                        layer=layer,
                        operator=operator,
                    )
            for operator in ("rope_q", "rope_k", "kv_write"):
                index = _expect_operator(
                    commands,
                    index,
                    token_step=token_step,
                    layer=layer,
                    operator=operator,
                )
            if token_step == 0:
                for head in range(14):
                    index = _expect_operator(
                        commands,
                        index,
                        token_step=token_step,
                        layer=layer,
                        operator="attention_score",
                        query_head=head,
                        context_token=0,
                    )
                    index = _expect_operator(
                        commands,
                        index,
                        token_step=token_step,
                        layer=layer,
                        operator="softmax",
                        query_head=head,
                    )
                    index = _expect_operator(
                        commands,
                        index,
                        token_step=token_step,
                        layer=layer,
                        operator="attention_value",
                        query_head=head,
                        context_token=0,
                    )
            else:
                context_count = token_step + 1
                for head in range(14):
                    for context_token in range(context_count):
                        index = _expect_operator(
                            commands,
                            index,
                            token_step=token_step,
                            layer=layer,
                            operator="attention_score",
                            query_head=head,
                            context_token=context_token,
                        )
                    for first_flag, more_flag in ((0, 1), (2, 3)):
                        for context_token in range(context_count):
                            index = _expect_operator(
                                commands,
                                index,
                                token_step=token_step,
                                layer=layer,
                                operator="attention_compose",
                                query_head=head,
                                context_token=context_token,
                                flags=first_flag if context_token == 0 else more_flag,
                            )
                    for context_token in range(context_count):
                        phase = 4 if context_token == 0 else 6 if context_token + 1 == context_count else 5
                        index = _expect_operator(
                            commands,
                            index,
                            token_step=token_step,
                            layer=layer,
                            operator="attention_compose",
                            query_head=head,
                            context_token=context_token,
                            flags=phase,
                        )
            for operator in suffix:
                index = _expect_operator(
                    commands,
                    index,
                    token_step=token_step,
                    layer=layer,
                    operator=operator,
                )
        index = _expect_operator(
            commands,
            index,
            token_step=token_step,
            layer=24,
            operator="final_rmsnorm",
        )
        for vocab_tile in range(LM_HEAD_TILE_COUNT):
            index = _expect_operator(
                commands,
                index,
                token_step=token_step,
                layer=24,
                operator="lm_head_tile",
                vocab_tile=vocab_tile,
            )
    if index != len(commands):
        raise RuntimeError("ACE2RT2 full schedule has trailing commands")


def build_full_prompt_commands(
    accepted_schedule: dict[str, Any],
    *,
    prompt_token_count: int,
    max_new_tokens: int,
    qkv_schedule_mode: str = QKV_SCHEDULE_LEGACY,
    preserve_prompt_layer0_qkv: bool = False,
) -> list[dict[str, Any]]:
    step_count = prompt_token_count + max_new_tokens - 1
    if prompt_token_count < 1:
        raise RuntimeError("ACE2RT2 prompt or generation count is outside its legal range")
    validate_max_new_tokens(max_new_tokens)
    if step_count > MAX_SEQUENCE_POSITIONS:
        raise RuntimeError("ACE2RT2 prompt plus generation positions exceed 32768")
    if qkv_schedule_mode not in {QKV_SCHEDULE_FUSED, QKV_SCHEDULE_LEGACY}:
        raise RuntimeError("ACE2RT2 QKV schedule mode differs")

    accepted_commands = list(accepted_schedule["commands"])
    token0 = [command for command in accepted_commands if int(command["token_step"]) == 0]
    token1 = [command for command in accepted_commands if int(command["token_step"]) == 1]
    commands: list[dict[str, Any]] = []
    for token_step in range(step_count):
        for layer in range(24):
            layer0 = [command for command in token0 if int(command["layer_id"]) == layer]
            layer1 = [command for command in token1 if int(command["layer_id"]) == layer]
            o_proj_index = next(
                index for index, command in enumerate(layer0) if command["operator"] == "o_proj"
            )
            prefix = layer0[:7]
            simple_attention = layer0[7:o_proj_index]
            suffix = layer0[o_proj_index:]
            if [command["operator"] for command in prefix] != [
                "input_rmsnorm",
                "q_proj",
                "k_proj",
                "v_proj",
                "rope_q",
                "rope_k",
                "kv_write",
            ]:
                raise RuntimeError(f"accepted layer-{layer} prefix template differs")
            use_fused_qkv = (
                qkv_schedule_mode == QKV_SCHEDULE_FUSED
                and not (
                    preserve_prompt_layer0_qkv
                    and token_step < prompt_token_count
                    and layer == 0
                )
            )
            _append_rt2_command(commands, prefix[0], token_step=token_step)
            if use_fused_qkv:
                _append_rt2_command(
                    commands,
                    _fuse_qkv_trio(prefix[1], prefix[2], prefix[3]),
                    token_step=token_step,
                )
            else:
                for template in prefix[1:4]:
                    _append_rt2_command(commands, template, token_step=token_step)
            for template in prefix[4:]:
                _append_rt2_command(commands, template, token_step=token_step)

            if token_step == 0:
                for template in simple_attention:
                    _append_rt2_command(commands, template, token_step=token_step)
            else:
                kv_base = int(prefix[-1]["scratch_addr"])
                score_templates = {
                    int(command["query_head"]): command
                    for command in layer0
                    if command["operator"] == "attention_score"
                }
                compose_templates = {
                    (int(command["query_head"]), int(command["flags"])): command
                    for command in layer1
                    if command["operator"] == "attention_compose"
                }
                for head in range(14):
                    score_template = score_templates[head]
                    kv_head = head // 7
                    score_base = (
                        RT2_SCORE_WORKSPACE_BASE
                        + (layer * 14 + head) * RT2_SCORE_HEAD_STRIDE
                    )
                    for context_token in range(token_step + 1):
                        record = kv_base + context_token * KV_BYTES_PER_TOKEN
                        _append_rt2_command(
                            commands,
                            score_template,
                            token_step=token_step,
                            src1_addr=record + kv_head * 64,
                            dst_addr=score_base + context_token * 16,
                            context_token=context_token,
                        )
                    for first_flag, more_flag in ((0, 1), (2, 3)):
                        for context_token in range(token_step + 1):
                            phase = first_flag if context_token == 0 else more_flag
                            template = compose_templates[(head, phase)]
                            record = kv_base + context_token * KV_BYTES_PER_TOKEN
                            _append_rt2_command(
                                commands,
                                template,
                                token_step=token_step,
                                flags=phase,
                                src0_addr=score_base + context_token * 16,
                                src1_addr=record + 128 + kv_head * 64,
                                context_token=context_token,
                            )
                    for context_token in range(token_step + 1):
                        phase = 4 if context_token == 0 else 6 if context_token == token_step else 5
                        template_key = (head, phase if phase != 5 else 4)
                        template = compose_templates.get(template_key, compose_templates[(head, 4)])
                        record = kv_base + context_token * KV_BYTES_PER_TOKEN
                        _append_rt2_command(
                            commands,
                            template,
                            token_step=token_step,
                            flags=phase,
                            src0_addr=score_base + context_token * 16,
                            src1_addr=record + 128 + kv_head * 64,
                            context_token=context_token,
                        )
            for template in suffix:
                _append_rt2_command(commands, template, token_step=token_step)

        final_commands = [
            command for command in token0 if int(command["layer_id"]) == 24
        ]
        for template in final_commands:
            _append_rt2_command(commands, template, token_step=token_step)

    validate_full_prompt_schedule(
        commands,
        prompt_token_count=prompt_token_count,
        max_new_tokens=max_new_tokens,
        qkv_schedule_mode=qkv_schedule_mode,
        preserve_prompt_layer0_qkv=preserve_prompt_layer0_qkv,
    )
    return commands


def build_rope_records(position_count: int) -> bytes:
    if not 1 <= position_count <= MAX_SEQUENCE_POSITIONS:
        raise RuntimeError("ACE2RT2 RoPE position count is outside 1..32768")
    payload = bytearray()
    for position in range(position_count):
        cosine, sine = accepted_runtime.absolute_coefficients_q15(position)
        payload.extend(PACKAGE_V2_ROPE_RECORD.pack(*cosine, *sine))
    return bytes(payload)


def _ace2rt2_command_fields(command: dict[str, Any], ordinal: int) -> tuple[int, ...]:
    if int(command.get("ordinal", ordinal)) != ordinal:
        raise RuntimeError(f"ACE2RT2 command ordinal differs at {ordinal}")
    operator_id = command.get("operator_id")
    if operator_id is None:
        operator = command.get("operator")
        if operator not in RT2_OPERATOR_IDS:
            raise RuntimeError(f"ACE2RT2 command operator differs at {ordinal}")
        operator_id = RT2_OPERATOR_IDS[operator]
    query_head = -1 if command.get("query_head") is None else int(command["query_head"])
    context_token = -1 if command.get("context_token") is None else int(command["context_token"])
    vocab_tile = -1 if command.get("vocab_tile") is None else int(command["vocab_tile"])
    return (
        ordinal,
        int(command["token_step"]),
        int(command["layer_id"]),
        int(operator_id),
        int(command["opcode"]),
        int(command["flags"]),
        int(command["m"]),
        int(command["n"]),
        int(command["k"]),
        int(command["sequence_position"]),
        int(command["completion_tag"]),
        query_head,
        context_token,
        vocab_tile,
        int(command["src0_addr"]),
        int(command["src1_addr"]),
        int(command["dst_addr"]),
        int(command["scale_addr"]),
        int(command["scratch_addr"]),
    )


def _serialize_ace2rt2_commands(commands: list[dict[str, Any]]) -> bytes:
    if not commands:
        raise RuntimeError("ACE2RT2 requires at least one command")
    raw = bytearray()
    for ordinal, command in enumerate(commands):
        try:
            raw.extend(accepted_runtime.COMMAND.pack(*_ace2rt2_command_fields(command, ordinal)))
        except (KeyError, OverflowError, struct.error, ValueError) as error:
            raise RuntimeError(f"ACE2RT2 command {ordinal} is not encodable: {error}") from error
    return bytes(raw)


def _validate_ace2rt2_steps(
    command_records: bytes,
    prompt_token_count: int,
    max_new_tokens: int,
) -> None:
    previous_step: int | None = None
    highest_step = -1
    for offset in range(0, len(command_records), accepted_runtime.COMMAND.size):
        fields = accepted_runtime.COMMAND.unpack_from(command_records, offset)
        token_step = int(fields[1])
        sequence_position = int(fields[9])
        if sequence_position != token_step:
            raise RuntimeError("ACE2RT2 command sequence position differs from token step")
        if previous_step is None:
            if token_step != 0:
                raise RuntimeError("ACE2RT2 token steps must begin at zero")
        elif token_step < previous_step or token_step > previous_step + 1:
            raise RuntimeError("ACE2RT2 token steps are not contiguous and nondecreasing")
        previous_step = token_step
        highest_step = max(highest_step, token_step)
    expected_highest = prompt_token_count + max_new_tokens - 2
    if highest_step != expected_highest:
        raise RuntimeError("ACE2RT2 command stream does not cover the full generation bound")


def build_ace2rt2_package(
    output: Path,
    *,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    commands: list[dict[str, Any]],
    embedding_offset: int,
    embedding_shape: list[int],
    image_sha256: str = accepted_runtime.EXPECTED_IMAGE_SHA256,
    model_sha256: str = accepted_runtime.EXPECTED_MODEL_SHA256,
    tokenizer_snapshot: Path = TOKENIZER_SNAPSHOT,
    tokenizer_record: bytes | None = None,
    dynamic_scale32_sidecars: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(embedding_shape) != 2 or list(map(int, embedding_shape)) != [151936, 896]:
        raise RuntimeError("ACE2RT2 embedding geometry differs")
    if int(embedding_offset) != PINNED_EMBEDDING_OFFSET:
        raise RuntimeError("ACE2RT2 embedding offset differs")
    if not prompt_token_ids:
        raise RuntimeError("ACE2RT2 requires at least one prompt token")
    validate_max_new_tokens(max_new_tokens)
    prompt_tokens = [int(token) for token in prompt_token_ids]
    if any(token < 0 or token >= int(embedding_shape[0]) for token in prompt_tokens):
        raise RuntimeError("ACE2RT2 prompt token is outside the embedding table")

    qkv_schedule_mode = (
        QKV_SCHEDULE_FUSED
        if any(command.get("operator") == "fused_qkv" for command in commands)
        else QKV_SCHEDULE_LEGACY
    )
    preserve_prompt_layer0_qkv = bool(dynamic_scale32_sidecars)
    validate_full_prompt_schedule(
        commands,
        prompt_token_count=len(prompt_tokens),
        max_new_tokens=max_new_tokens,
        qkv_schedule_mode=qkv_schedule_mode,
        preserve_prompt_layer0_qkv=preserve_prompt_layer0_qkv,
    )
    command_records = _serialize_ace2rt2_commands(commands)
    _validate_ace2rt2_steps(command_records, len(prompt_tokens), max_new_tokens)
    rope_position_count = len(prompt_tokens) + max_new_tokens - 1
    rope_records = build_rope_records(rope_position_count)
    prompt_bytes = struct.pack(f"<{len(prompt_tokens)}I", *prompt_tokens)
    termination_bytes = struct.pack("<2I", *TERMINATION_TOKEN_IDS)
    if tokenizer_record is None:
        tokenizer_record = tokenizer_identity_record(tokenizer_snapshot)
    try:
        image_digest = bytes.fromhex(image_sha256)
        model_digest = bytes.fromhex(model_sha256)
    except ValueError as error:
        raise RuntimeError("ACE2RT2 image or model SHA-256 is not hexadecimal") from error
    if len(image_digest) != 32 or len(model_digest) != 32:
        raise RuntimeError("ACE2RT2 image or model SHA-256 width differs")

    model_identity = dynamic_scale32_model_identity(model_sha256)
    _validate_dynamic_scale32_initial_binding(
        dynamic_scale32_sidecars,
        commands,
        len(prompt_tokens),
    )
    sidecar_records = _serialize_dynamic_scale32_sidecars(
        dynamic_scale32_sidecars,
        model_identity=model_identity,
    )
    sidecar_count = len(sidecar_records) // PACKAGE_V2_DYNAMIC_RECORD.size
    sidecar_payload_addresses = [
        PACKAGE_V2_DYNAMIC_RECORD.unpack_from(sidecar_records, offset)[0]
        for offset in range(0, len(sidecar_records), PACKAGE_V2_DYNAMIC_RECORD.size)
    ]
    header_bytes = PACKAGE_V2_HEADER_BYTES
    dynamic_extension = b""
    if sidecar_count:
        header_bytes += PACKAGE_V2_DYNAMIC_EXTENSION_BYTES
        dynamic_extension = PACKAGE_V2_DYNAMIC_EXTENSION.pack(
            PACKAGE_V2_DYNAMIC_EXTENSION_MAGIC,
            1,
            PACKAGE_V2_DYNAMIC_RECORD.size,
            sidecar_count,
            model_identity,
            hashlib.sha256(sidecar_records).digest(),
            DYNAMIC_SCALE32_FEATURE_SIDECARS,
            0,
        )

    header = PACKAGE_V2_HEADER.pack(
        PACKAGE_V2_MAGIC,
        2,
        header_bytes,
        accepted_runtime.COMMAND.size,
        len(commands),
        len(prompt_tokens),
        max_new_tokens,
        len(TERMINATION_TOKEN_IDS),
        int(embedding_shape[0]),
        int(embedding_shape[1]),
        rope_position_count,
        int(embedding_offset),
        hashlib.sha256(rope_records + sidecar_records + command_records).digest(),
        image_digest,
        model_digest,
        hashlib.sha256(tokenizer_record).digest(),
        hashlib.sha256(prompt_bytes).digest(),
    )
    if len(header) != PACKAGE_V2_HEADER_BYTES:
        raise RuntimeError("ACE2RT2 header width differs")
    raw = (
        header
        + dynamic_extension
        + prompt_bytes
        + termination_bytes
        + rope_records
        + sidecar_records
        + command_records
    )
    write_atomic(output, raw)
    return {
        "format": "ACE2RT2",
        "version": 2,
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "record_bytes": accepted_runtime.COMMAND.size,
        "commands": len(commands),
        "qkv_schedule_mode": qkv_schedule_mode,
        "prompt_token_count": len(prompt_tokens),
        "max_new_tokens": max_new_tokens,
        "termination_token_ids": list(TERMINATION_TOKEN_IDS),
        "rope_position_count": rope_position_count,
        "rope_records_sha256": hashlib.sha256(rope_records).hexdigest(),
        "dynamic_scale32_enabled": bool(sidecar_count),
        "dynamic_scale32_model_identity64": f"{model_identity:016x}",
        "dynamic_scale32_sidecar_count": sidecar_count,
        "dynamic_scale32_sidecar_records_sha256": hashlib.sha256(
            sidecar_records
        ).hexdigest(),
        "dynamic_scale32_payload_addresses": sidecar_payload_addresses,
        "embedding_tensor_absolute_offset": int(embedding_offset),
        "schedule_sha256": hashlib.sha256(
            rope_records + sidecar_records + command_records
        ).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_record).hexdigest(),
        "prompt_tokens_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
    }


def read_ace2rt2_package_metadata(package_path: Path) -> dict[str, Any]:
    raw = package_path.read_bytes()
    if len(raw) < PACKAGE_V2_HEADER_BYTES:
        raise RuntimeError("ACE2RT2 package header is truncated")
    unpacked = PACKAGE_V2_HEADER.unpack_from(raw)
    (
        magic,
        version,
        header_bytes,
        record_bytes,
        command_count,
        prompt_token_count,
        max_new_tokens,
        termination_token_count,
        embedding_rows,
        embedding_cols,
        rope_position_count,
        embedding_offset,
        schedule_digest,
        image_digest,
        model_digest,
        tokenizer_digest,
        prompt_digest,
    ) = unpacked
    if (
        magic != PACKAGE_V2_MAGIC
        or version != 2
        or header_bytes
        not in (
            PACKAGE_V2_HEADER_BYTES,
            PACKAGE_V2_HEADER_BYTES + PACKAGE_V2_DYNAMIC_EXTENSION_BYTES,
        )
        or record_bytes != accepted_runtime.COMMAND.size
        or prompt_token_count < 1
        or termination_token_count != len(TERMINATION_TOKEN_IDS)
        or [embedding_rows, embedding_cols] != [151936, 896]
        or rope_position_count != prompt_token_count + max_new_tokens - 1
        or embedding_offset != PINNED_EMBEDDING_OFFSET
    ):
        raise RuntimeError("ACE2RT2 package header contract differs")
    validate_max_new_tokens(max_new_tokens)

    sidecar_count = 0
    sidecar_record_bytes = PACKAGE_V2_DYNAMIC_RECORD.size
    sidecar_digest = hashlib.sha256(b"").digest()
    model_identity = dynamic_scale32_model_identity(model_digest.hex())
    if header_bytes != PACKAGE_V2_HEADER_BYTES:
        if len(raw) < header_bytes:
            raise RuntimeError("ACE2RT2 dynamic Scale32 extension is truncated")
        (
            extension_magic,
            extension_version,
            sidecar_record_bytes,
            sidecar_count,
            extension_model_identity,
            sidecar_digest,
            feature_flags,
            extension_reserved,
        ) = PACKAGE_V2_DYNAMIC_EXTENSION.unpack_from(raw, PACKAGE_V2_HEADER_BYTES)
        if (
            extension_magic != PACKAGE_V2_DYNAMIC_EXTENSION_MAGIC
            or extension_version != 1
            or sidecar_record_bytes != PACKAGE_V2_DYNAMIC_RECORD.size
            or sidecar_count == 0
            or extension_model_identity != model_identity
            or feature_flags != DYNAMIC_SCALE32_FEATURE_SIDECARS
            or extension_reserved != 0
        ):
            raise RuntimeError("ACE2RT2 dynamic Scale32 extension contract differs")
    prompt_start = header_bytes
    prompt_stop = prompt_start + prompt_token_count * 4
    termination_stop = prompt_stop + termination_token_count * 4
    rope_stop = termination_stop + rope_position_count * PACKAGE_V2_ROPE_RECORD.size
    sidecar_stop = rope_stop + sidecar_count * sidecar_record_bytes
    expected_bytes = sidecar_stop + command_count * record_bytes
    if expected_bytes != len(raw):
        raise RuntimeError("ACE2RT2 package byte count differs")
    prompt_raw = raw[prompt_start:prompt_stop]
    if hashlib.sha256(prompt_raw).digest() != prompt_digest:
        raise RuntimeError("ACE2RT2 prompt-token digest differs")
    prompt_tokens = list(struct.unpack(f"<{prompt_token_count}I", prompt_raw))
    if any(token >= embedding_rows for token in prompt_tokens):
        raise RuntimeError("ACE2RT2 prompt token is outside the embedding table")
    termination_tokens = list(
        struct.unpack(
            f"<{termination_token_count}I",
            raw[prompt_stop:termination_stop],
        )
    )
    if termination_tokens != list(TERMINATION_TOKEN_IDS):
        raise RuntimeError("ACE2RT2 termination token list differs")
    sidecar_raw = raw[rope_stop:sidecar_stop]
    if hashlib.sha256(sidecar_raw).digest() != sidecar_digest:
        raise RuntimeError("ACE2RT2 dynamic Scale32 sidecar digest differs")
    sidecar_payload_addresses: list[int] = []
    parsed_sidecars: list[tuple[int, bytes]] = []
    seen_bindings: set[tuple[int, int, int, int]] = set()
    for offset in range(0, len(sidecar_raw), sidecar_record_bytes):
        payload_addr, sidecar = PACKAGE_V2_DYNAMIC_RECORD.unpack_from(sidecar_raw, offset)
        _validate_dynamic_scale32_sidecar(
            sidecar,
            payload_addr=payload_addr,
            model_identity=model_identity,
        )
        binding = (
            payload_addr,
            int.from_bytes(sidecar[8:10], "little"),
            sidecar[10],
            sidecar[11],
        )
        if binding in seen_bindings:
            raise RuntimeError("ACE2RT2 dynamic Scale32 producer binding is duplicated")
        seen_bindings.add(binding)
        sidecar_payload_addresses.append(payload_addr)
        parsed_sidecars.append((payload_addr, sidecar))
    schedule_raw = raw[termination_stop:]
    if hashlib.sha256(schedule_raw).digest() != schedule_digest:
        raise RuntimeError("ACE2RT2 schedule digest differs")
    command_raw = raw[sidecar_stop:]
    command_fields = [
        accepted_runtime.COMMAND.unpack_from(command_raw, offset)
        for offset in range(0, len(command_raw), accepted_runtime.COMMAND.size)
    ]
    fused_fields = [
        fields
        for fields in command_fields
        if fields[3] == FUSED_QKV_OPERATOR_ID or fields[4] == FUSED_QKV_OPCODE
    ]
    for fields in fused_fields:
        if fields[3] != FUSED_QKV_OPERATOR_ID or fields[4] != FUSED_QKV_OPCODE:
            raise RuntimeError("ACE2RT2 fused-QKV operator/opcode binding differs")
        _validate_fused_qkv_descriptor(
            {
                "operator": "fused_qkv",
                "opcode": fields[4],
                "flags": fields[5],
                "layer_id": fields[2],
                "m": fields[6],
                "n": fields[7],
                "k": fields[8],
                "query_head": None if fields[11] == -1 else fields[11],
                "context_token": None if fields[12] == -1 else fields[12],
                "vocab_tile": None if fields[13] == -1 else fields[13],
                "src0_addr": fields[14],
                "src1_addr": fields[15],
                "dst_addr": fields[16],
                "scale_addr": fields[17],
                "scratch_addr": fields[18],
            }
        )
    dynamic_commands = [fields for fields in command_fields if fields[5] & DYNAMIC_SCALE32_FLAG]
    if not parsed_sidecars:
        if dynamic_commands:
            raise RuntimeError("ACE2RT2 flags[6] initial boundary lacks its DS32 sidecar")
    else:
        expected_dynamic = [
            fields
            for fields in command_fields
            if fields[1] < prompt_token_count
            and fields[2] == 0
            and fields[3] in (0, 1, 2, 3)
        ]
        if (
            len(parsed_sidecars) != prompt_token_count
            or dynamic_commands != expected_dynamic
            or len(dynamic_commands) != 4 * prompt_token_count
        ):
            raise RuntimeError(
                "ACE2RT2 prompt DS32 tranche requires one host plan and RMS/Q/K/V per prompt position"
            )
        for position, (payload_addr, sidecar) in enumerate(parsed_sidecars):
            tranche = [fields for fields in expected_dynamic if fields[1] == position]
            if [fields[3] for fields in tranche] != [0, 1, 2, 3]:
                raise RuntimeError("ACE2RT2 prompt DS32 operator order differs")
            rms, q_proj, k_proj, v_proj = tranche
            if any(fields[5] != DYNAMIC_SCALE32_FROZEN_FLAGS for fields in tranche):
                raise RuntimeError("ACE2RT2 prompt DS32 command flags differ")
            if (
                rms[4:9] != (2, DYNAMIC_SCALE32_FROZEN_FLAGS, 1, 896, 0)
                or rms[14:] != (
                    0x0000001000000000,
                    0,
                    0x0000001000000700,
                    0x0000000300000000,
                    0,
                )
            ):
                raise RuntimeError("ACE2RT2 prompt DS32 RMS command contract differs")
            projection_contracts = (
                (q_proj, 896, 0x0000000100000000, 0x0000001000000A80, 0x0000000200000000),
                (k_proj, 128, 0x0000000100062000, 0x0000001000000E00, 0x0000000200003800),
                (v_proj, 128, 0x0000000100070000, 0x0000001000000E80, 0x0000000200004000),
            )
            for projection, outputs, weights, destination, metadata in projection_contracts:
                if (
                    projection[4:9]
                    != (1, DYNAMIC_SCALE32_FROZEN_FLAGS, 1, outputs, 896)
                    or projection[14:]
                    != (
                        int(rms[16]),
                        weights,
                        destination,
                        metadata,
                        0,
                    )
                ):
                    raise RuntimeError("ACE2RT2 prompt DS32 Q/K/V consumer contract differs")
            if payload_addr != rms[14]:
                raise RuntimeError("ACE2RT2 prompt DS32 host-plan payload binding differs")
            if (
                sidecar[5] != DYNAMIC_SCALE32_INITIAL_GROUP_LANES
                or sidecar[6] != DYNAMIC_SCALE32_INITIAL_GROUP_COUNT
                or int.from_bytes(sidecar[8:10], "little")
                != dynamic_scale32_host_plan_tag(position)
                or sidecar[10] != DYNAMIC_SCALE32_HOST_PLAN_LAYER
                or sidecar[11] != DYNAMIC_SCALE32_HOST_PLAN_OPCODE
                or int.from_bytes(sidecar[12:16], "little")
                != DYNAMIC_SCALE32_INITIAL_ELEMENTS
            ):
                raise RuntimeError("ACE2RT2 prompt DS32 host-plan producer contract differs")
    return {
        "path": str(package_path.resolve()),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "version": version,
        "record_bytes": record_bytes,
        "commands": command_count,
        "qkv_schedule_mode": (
            QKV_SCHEDULE_FUSED if fused_fields else QKV_SCHEDULE_LEGACY
        ),
        "prompt_token_count": prompt_token_count,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_sha256": prompt_digest.hex(),
        "max_new_tokens": max_new_tokens,
        "termination_token_ids": termination_tokens,
        "rope_position_count": rope_position_count,
        "dynamic_scale32_enabled": bool(sidecar_count),
        "dynamic_scale32_model_identity64": f"{model_identity:016x}",
        "dynamic_scale32_sidecar_count": sidecar_count,
        "dynamic_scale32_sidecar_records_sha256": sidecar_digest.hex(),
        "dynamic_scale32_payload_addresses": sidecar_payload_addresses,
        "schedule_sha256": schedule_digest.hex(),
        "image_sha256": image_digest.hex(),
        "model_sha256": model_digest.hex(),
        "tokenizer_sha256": tokenizer_digest.hex(),
        "command_records_offset": sidecar_stop,
    }


def reconstruct_ace2rt2_schedule(
    package_path: Path,
    package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    package_path = package_path.resolve()
    metadata = package or read_ace2rt2_package_metadata(package_path)
    schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
    commands = build_full_prompt_commands(
        schedule,
        prompt_token_count=int(metadata["prompt_token_count"]),
        max_new_tokens=int(metadata["max_new_tokens"]),
        qkv_schedule_mode=str(metadata["qkv_schedule_mode"]),
        preserve_prompt_layer0_qkv=bool(metadata["dynamic_scale32_enabled"]),
    )
    if metadata["dynamic_scale32_enabled"]:
        _apply_prompt_dynamic_scale32_flags(
            commands,
            int(metadata["prompt_token_count"]),
        )
    if len(commands) != int(metadata["commands"]):
        raise RuntimeError("reconstructed command count differs from ACE2RT2")

    sidecar_bytes = (
        int(metadata["dynamic_scale32_sidecar_count"])
        * PACKAGE_V2_DYNAMIC_RECORD.size
    )
    sidecar_stop = int(metadata["command_records_offset"])
    sidecar_start = sidecar_stop - sidecar_bytes
    raw = package_path.read_bytes()
    if sidecar_start < 0 or sidecar_stop > len(raw):
        raise RuntimeError("ACE2RT2 dynamic Scale32 sidecar range differs")
    sidecar_records = raw[sidecar_start:sidecar_stop]
    sidecar_records_sha256 = hashlib.sha256(sidecar_records).hexdigest()
    if sidecar_records_sha256 != metadata["dynamic_scale32_sidecar_records_sha256"]:
        raise RuntimeError("ACE2RT2 dynamic Scale32 sidecar digest differs")

    reconstructed_schedule_sha256 = hashlib.sha256(
        build_rope_records(int(metadata["rope_position_count"]))
        + sidecar_records
        + _serialize_ace2rt2_commands(commands)
    ).hexdigest()
    if reconstructed_schedule_sha256 != metadata["schedule_sha256"]:
        raise RuntimeError("reconstructed command schedule differs from ACE2RT2")
    return {
        "commands": commands,
        "schedule_sha256": reconstructed_schedule_sha256,
        "dynamic_scale32_sidecar_records_sha256": sidecar_records_sha256,
    }


def build_prompt_package(
    package_path: Path,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    *,
    diagnostic_instruct: dict[str, Any] | None = None,
    qkv_schedule_mode: str = QKV_SCHEDULE_FUSED,
) -> dict[str, Any]:
    rtl_binding = certified_rtl_binding()
    if diagnostic_instruct is None:
        stale_shell_sha = accepted_runtime.EXPECTED_SHELL_SHA256
        accepted_runtime.EXPECTED_SHELL_SHA256 = rtl_binding["live_shell_sha256"]
        try:
            schedule, inputs, embedding_offset, embedding_shape = accepted_runtime.validate_inputs(
                accepted_runtime.IMAGE.resolve(),
                accepted_runtime.IMAGE_CONTRACT.resolve(),
                accepted_runtime.IMAGE_AUDIT.resolve(),
                accepted_runtime.SCHEDULE.resolve(),
                accepted_runtime.SCHEDULE_VALIDATION.resolve(),
                accepted_runtime.MODEL.resolve(),
            )
        finally:
            accepted_runtime.EXPECTED_SHELL_SHA256 = stale_shell_sha
        image_path = accepted_runtime.IMAGE
        model_path = accepted_runtime.MODEL
        tokenizer_snapshot = TOKENIZER_SNAPSHOT
        tokenizer_record = None
        diagnostic_prepare_only = False
    else:
        schedule = diagnostic_instruct["schedule"]
        schedule_inputs = diagnostic_instruct["schedule_inputs"]
        embedding_offset = int(diagnostic_instruct["embedding_offset"])
        embedding_shape = list(map(int, diagnostic_instruct["embedding_shape"]))
        inputs = {
            "sealed_image": diagnostic_instruct["image"],
            "raw_safetensors": diagnostic_instruct["source_files"][
                "model.safetensors"
            ],
            "accepted_schedule": schedule_inputs["accepted_schedule"],
        }
        image_path = diagnostic_instruct["image_path"]
        model_path = diagnostic_instruct["model_path"]
        tokenizer_snapshot = diagnostic_instruct["snapshot"]
        tokenizer_record = diagnostic_instruct_tokenizer_identity_record()
        diagnostic_prepare_only = True
        qkv_schedule_mode = QKV_SCHEDULE_LEGACY
    commands = build_full_prompt_commands(
        schedule,
        prompt_token_count=len(prompt_token_ids),
        max_new_tokens=max_new_tokens,
        qkv_schedule_mode=qkv_schedule_mode,
        preserve_prompt_layer0_qkv=(
            qkv_schedule_mode == QKV_SCHEDULE_FUSED
            and diagnostic_instruct is None
        ),
    )
    _apply_prompt_dynamic_scale32_flags(commands, len(prompt_token_ids))
    model_identity = dynamic_scale32_model_identity(inputs["raw_safetensors"]["sha256"])
    ds32_applicability = audit_prompt_dynamic_scale32_applicability(
        prompt_token_ids,
        embedding_offset,
        image_path=image_path,
        model_path=model_path,
        diagnostic_prepare_only=diagnostic_prepare_only,
    )
    if any(not case["selected_restored_exact"] for case in ds32_applicability["cases"]):
        raise RuntimeError("ACE2RT2 prompt DS32 selected plan is not lossless")
    dynamic_scale32_sidecars = [
        {
            "payload_addr": int(commands[0]["src0_addr"]),
            "sidecar": _build_prompt_dynamic_scale32_sidecar(
                payload_addr=int(commands[0]["src0_addr"]),
                model_identity=model_identity,
                sequence_position=int(case["chat_position"]),
                deltas=case["selected_deltas"],
            ),
        }
        for case in ds32_applicability["cases"]
    ]
    package = build_ace2rt2_package(
        package_path,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        commands=commands,
        embedding_offset=embedding_offset,
        embedding_shape=embedding_shape,
        image_sha256=inputs["sealed_image"]["sha256"],
        model_sha256=inputs["raw_safetensors"]["sha256"],
        tokenizer_snapshot=tokenizer_snapshot,
        tokenizer_record=tokenizer_record,
        dynamic_scale32_sidecars=dynamic_scale32_sidecars,
    )
    return {
        **package,
        "_ds32_applicability": ds32_applicability,
        "_first_reference_commands": commands[:10],
        "_position0_reference_commands": [
            command for command in commands if int(command["token_step"]) == 0
        ],
        "_position1_reference_commands": [
            command for command in commands if int(command["token_step"]) <= 1
        ],
        "_position2_reference_commands": [
            command for command in commands if int(command["token_step"]) <= 2
        ],
        "_position3_reference_commands": [
            command for command in commands if int(command["token_step"]) <= 3
        ],
        "_all_reference_commands": list(commands),
        "image_sha256": inputs["sealed_image"]["sha256"],
        "model_sha256": inputs["raw_safetensors"]["sha256"],
        "template_schedule_sha256": inputs["accepted_schedule"]["sha256"],
        "rtl_binding": rtl_binding,
    }


def decode_generated(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def resolve_position_token(
    prompt_tokens: list[int],
    generated_tokens: list[int],
    position: int,
) -> tuple[int, str, int | None]:
    if position < 0:
        raise RuntimeError("position token lookup requires a nonnegative position")
    if position < len(prompt_tokens):
        return int(prompt_tokens[position]), "prompt", None
    generated_index = position - len(prompt_tokens)
    if generated_index >= len(generated_tokens):
        raise RuntimeError("generated-token feedback began before its argmax completed")
    return int(generated_tokens[generated_index]), "generated_argmax", generated_index


def image_file_offset(address: int, size: int) -> int:
    for base, region_size, file_offset in IMAGE_REGIONS:
        if base <= address and address - base + size <= region_size:
            return file_offset + address - base
    raise RuntimeError(f"address is outside the accepted image: 0x{address:016x}")


def read_image(
    address: int,
    size: int,
    *,
    image_path: Path = accepted_runtime.IMAGE,
) -> bytes:
    with image_path.open("rb") as handle:
        handle.seek(image_file_offset(address, size))
        raw = handle.read(size)
    if len(raw) != size:
        raise RuntimeError("accepted image read is truncated")
    return raw


def quantized_embedding(
    token: int,
    embedding_offset: int,
    scale: np.float32,
    *,
    model_path: Path = accepted_runtime.MODEL,
) -> list[int]:
    with model_path.open("rb") as handle:
        handle.seek(embedding_offset + token * 896 * 2)
        raw = handle.read(896 * 2)
    if len(raw) != 896 * 2:
        raise RuntimeError("embedding row is truncated")
    bf16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    values = (bf16 << 16).view(np.float32)
    return np.clip(np.rint(values / scale), -128, 127).astype(np.int8).astype(int).tolist()


def _signed_int8_sha256(values: list[int]) -> str:
    return hashlib.sha256(bytes(value & 0xFF for value in values)).hexdigest()


@lru_cache(maxsize=1)
def silu_input_scale_table() -> tuple[dict[str, Any], ...]:
    payload = json.loads(SILU_INPUT_SCALE_TABLE.read_text(encoding="utf-8"))
    layers = payload.get("layers")
    if not isinstance(layers, list) or len(layers) != 24:
        raise RuntimeError("SiLU input scale table must contain exactly 24 layers")
    for layer_id, record in enumerate(layers):
        if not isinstance(record, dict) or int(record.get("layer_id", -1)) != layer_id:
            raise RuntimeError("SiLU input scale table layer ordering differs")
        for key in ("gate_scale_numerator", "up_scale_numerator"):
            value = int(record.get(key, 0))
            if not 0 < value < (1 << 16):
                raise RuntimeError("SiLU input scale table numerator differs")
    return tuple(layers)


def _restore_dynamic_scale32_mantissa(value: int, delta: int) -> int:
    if delta >= 0:
        return value << delta
    return round_shift_even_signed(value, -delta)


def evaluate_dynamic_scale32_rms_outputs(outputs: list[int]) -> dict[str, Any]:
    if len(outputs) != DYNAMIC_SCALE32_INITIAL_ELEMENTS:
        raise RuntimeError(
            "dynamic Scale32 RMSNorm applicability requires exactly 896 outputs"
        )

    base_scale = pack_scale32(0x8000, 0)
    selected_deltas: list[int] = []
    selected_restored_outputs: list[int] = []
    restored_outputs: list[int] = []
    frozen_representable = True
    group_lanes = DYNAMIC_SCALE32_INITIAL_GROUP_LANES

    for group, frozen_delta in enumerate(DYNAMIC_SCALE32_RMS_OUTPUT_DELTAS):
        values = outputs[group * group_lanes : (group + 1) * group_lanes]
        selected = dynamic_group(values, base_scale)
        selected_deltas.append(selected.delta)
        selected_restored_outputs.extend(
            _restore_dynamic_scale32_mantissa(mantissa, selected.delta)
            for mantissa in selected.mantissas
        )
        for value in values:
            mantissa = quantize_for_delta(value, frozen_delta)
            frozen_representable &= -127 <= mantissa <= 127
            restored_outputs.append(
                _restore_dynamic_scale32_mantissa(mantissa, frozen_delta)
            )

    frozen_deltas = list(DYNAMIC_SCALE32_RMS_OUTPUT_DELTAS)
    selected_restored_exact = selected_restored_outputs == outputs
    restored_exact = restored_outputs == outputs
    mismatch_category = (
        None if selected_restored_exact else "selected_plan_round_trip"
    )

    return {
        "selection_rule": "smallest legal delta with all RNE mantissas in [-127,127]",
        "selected_deltas": selected_deltas,
        "selected_restored_exact": selected_restored_exact,
        "frozen_deltas": frozen_deltas,
        "delta_vector_matches": selected_deltas == frozen_deltas,
        "frozen_representable": bool(frozen_representable),
        "rms_restored_exact": restored_exact,
        "rms_s8_sha256": _signed_int8_sha256(outputs),
        "mismatch_category": mismatch_category,
        "status": "PASS" if selected_restored_exact else "MISMATCH",
    }


def dynamic_scale32_prompt_scope(
    *,
    positions_checked: int,
    unique_token_ids_checked: int,
) -> dict[str, Any]:
    return {
        "positions_checked": positions_checked,
        "unique_token_ids_checked": unique_token_ids_checked,
        "reference_plan_construction": True,
        "rtl_extended": True,
        "later_dynamic_scale32_policy_adopted": True,
        "rtl_execution_boundary": (
            "per-position authenticated host plan drives layer-0 input RMSNorm "
            "publication consumed by q_proj, k_proj, and v_proj; a bounded "
            "position-0 live RTL prefix also publishes all attention-value "
            "heads into o_proj"
        ),
        "unsupported_later_boundary": (
            "attention_residual_add_publication_into_post_attention_rmsnorm"
        ),
        "attention_value_to_o_projection_position0": True,
        "attention_value_to_o_projection_user_content_positions": False,
        "absolute_position_semantics": (
            "layer-0 input RMSNorm consumes only the tied embedding row; "
            "RoPE and KV state occur later in the command schedule"
        ),
    }


def audit_prompt_dynamic_scale32_applicability(
    prompt_token_ids: list[int],
    embedding_offset: int,
    *,
    image_path: Path = accepted_runtime.IMAGE,
    model_path: Path = accepted_runtime.MODEL,
    diagnostic_prepare_only: bool = False,
) -> dict[str, Any]:
    if not prompt_token_ids:
        raise RuntimeError("dynamic Scale32 prompt applicability requires prompt tokens")

    scale_address = 0x0000000300000000
    embedding_scale = np.float32(
        struct.unpack(
            "<d",
            read_image(
                scale_address + DYNAMIC_SCALE32_RMSNORM_SCALE_OFFSET,
                8,
                image_path=image_path,
            ),
        )[0]
    )
    gains = np.frombuffer(
        read_image(
            scale_address,
            DYNAMIC_SCALE32_RMSNORM_GAIN_BYTES,
            image_path=image_path,
        ),
        dtype="<i2",
    ).astype(int).tolist()

    token_results: dict[int, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    for chat_position, token_id in enumerate(prompt_token_ids):
        token = int(token_id)
        if token not in token_results:
            embedding = quantized_embedding(
                token,
                embedding_offset,
                embedding_scale,
                model_path=model_path,
            )
            rms_outputs = reference_rmsnorm(embedding, gains).outputs
            token_results[token] = {
                "token_id": token,
                "embedding_s8_sha256": _signed_int8_sha256(embedding),
                **evaluate_dynamic_scale32_rms_outputs(rms_outputs),
            }
        cases.append(
            {
                "chat_position": chat_position,
                **token_results[token],
            }
        )

    earliest = next((case for case in cases if case["status"] == "MISMATCH"), None)
    prompt_bytes = struct.pack(f"<{len(prompt_token_ids)}I", *prompt_token_ids)
    scope = dynamic_scale32_prompt_scope(
        positions_checked=len(cases),
        unique_token_ids_checked=len(token_results),
    )
    if diagnostic_prepare_only:
        scope = {
            "positions_checked": len(cases),
            "unique_token_ids_checked": len(token_results),
            "reference_plan_construction": True,
            "runtime_executed": False,
            "rtl_agreement_claim": False,
            "generation_or_decode_executed": False,
            "scope": (
                "host-side layer-0 input RMSNorm DS32 sidecar construction for "
                "the prepared package only"
            ),
        }
    return {
        "schema_version": 2,
        "generated_at_utc": utc_now(),
        "classification": "full_prompt_layer0_input_rmsnorm_ds32_applicability_audit",
        "prompt_tokens_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "scope": scope,
        "frozen_deltas": list(DYNAMIC_SCALE32_RMS_OUTPUT_DELTAS),
        "cases": cases,
        "earliest_mismatch": (
            None
            if earliest is None
            else {
                "chat_position": earliest["chat_position"],
                "token_id": earliest["token_id"],
                "category": earliest["mismatch_category"],
                "selected_deltas": earliest["selected_deltas"],
                "frozen_deltas": earliest["frozen_deltas"],
                "frozen_representable": earliest["frozen_representable"],
                "rms_restored_exact": earliest["rms_restored_exact"],
            }
        ),
        "status": "MISMATCH_FOUND" if earliest is not None else "PASS",
    }


def projection_reference(command: dict[str, Any], activations: list[int]) -> list[int]:
    outputs = int(command["n"])
    reduction = int(command["k"])
    packed = np.frombuffer(
        read_image(int(command["src1_addr"]), outputs * reduction // 2),
        dtype=np.uint8,
    )
    nibbles = np.empty(packed.size * 2, dtype=np.int8)
    nibbles[0::2] = packed & 0x0F
    nibbles[1::2] = packed >> 4
    nibbles[nibbles >= 8] -= 16
    weights = nibbles.reshape(outputs, reduction).astype(int).tolist()
    metadata = read_image(int(command["scale_addr"]), outputs * 16)
    multipliers: list[int] = []
    right_shifts: list[int] = []
    zero_points: list[int] = []
    biases: list[int] = []
    for output in range(outputs):
        record = metadata[output * 16 : (output + 1) * 16]
        if record[4] & 0xC0 or any(record[10:]):
            raise RuntimeError("projection metadata reserved bits differ")
        multipliers.append(struct.unpack_from("<i", record, 0)[0])
        right_shifts.append(record[4] & 0x3F)
        zero_points.append(struct.unpack_from("<b", record, 5)[0])
        biases.append(struct.unpack_from("<i", record, 6)[0])
    result = reference_projection(
        ProjectionCase(
            name=str(command["operator"]),
            rows=1,
            reduction_size=reduction,
            activations=[activations],
            weights=weights,
            multipliers=multipliers,
            right_shifts=right_shifts,
            output_zero_points=zero_points,
            bias_accumulators=biases,
        )
    )
    return result.outputs[0]


def parse_journal_v2(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if raw[:8] != b"ACE2J2\0\0" or struct.unpack_from("<I", raw, 8)[0] != 2:
        raise RuntimeError("runtime journal is not ACE2J2 version 2")
    offset = JOURNAL_V2_HEADER_BYTES
    records: list[dict[str, Any]] = []
    while offset < len(raw):
        body_size = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        body = raw[offset : offset + body_size]
        offset += body_size
        digest = raw[offset : offset + 32]
        offset += 32
        trailer = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        if len(body) != body_size or trailer != body_size or hashlib.sha256(body).digest() != digest:
            raise RuntimeError("runtime journal frame authentication differs")
        prefix = JOURNAL_COMPLETED_PREFIX.unpack_from(body)
        body_offset = JOURNAL_COMPLETED_PREFIX.size
        generated_tokens = []
        for _ in range(prefix[9]):
            generated_tokens.append(struct.unpack_from("<I", body, body_offset)[0])
            body_offset += 4
        terminated = bool(body[body_offset])
        body_offset += 1
        source_sha = body[body_offset : body_offset + 32]
        body_offset += 32
        destination_sha = body[body_offset : body_offset + 32]
        body_offset += 32
        write_count = struct.unpack_from("<I", body, body_offset)[0]
        body_offset += 4
        writes = []
        for _ in range(write_count):
            writes.append(JOURNAL_WRITE_RECORD.unpack_from(body, body_offset))
            body_offset += JOURNAL_WRITE_RECORD.size
        if body_offset != len(body):
            raise RuntimeError("runtime journal frame width differs")
        records.append(
            {
                "ordinal": prefix[0],
                "cycles": prefix[1],
                "read_beats": prefix[2],
                "write_beats": prefix[3],
                "done_tag": prefix[4],
                "done_error": bool(prefix[5]),
                "saturation": bool(prefix[6]),
                "argmax_token": prefix[7],
                "argmax_logit": prefix[8],
                "generated_tokens": generated_tokens,
                "terminated": terminated,
                "source_sha": source_sha,
                "destination_sha": destination_sha,
                "writes": writes,
            }
        )
    return records


def verify_first_kv_publication(
    commands: list[dict[str, Any]],
    *,
    first_prompt_token: int,
    runtime_output: Path,
    journal_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    kv_command = next(
        command
        for command in commands
        if command["operator"] == "kv_write"
        and int(command["token_step"]) == 0
        and int(command["layer_id"]) == 0
    )
    records = (
        journal_records
        if journal_records is not None
        else parse_journal_v2(runtime_output / "progress.journal")
    )
    ordinal = int(kv_command["ordinal"])
    if len(records) <= ordinal:
        return {
            "status": "NOT_REACHED",
            "command_ordinal": ordinal,
            "commands_completed": len(records),
        }

    first_commands = commands[: ordinal + 1]
    rmsnorm_command = first_commands[0]
    k_command = next(command for command in first_commands if command["operator"] == "k_proj")
    v_command = next(command for command in first_commands if command["operator"] == "v_proj")
    rope_command = next(command for command in first_commands if command["operator"] == "rope_k")
    scale_record = read_image(int(rmsnorm_command["scale_addr"]), 1808)
    gains = np.frombuffer(scale_record[:1792], dtype="<i2").astype(int).tolist()
    embedding_scale = np.float32(struct.unpack_from("<d", scale_record, 1800)[0])
    activations = quantized_embedding(
        first_prompt_token,
        PINNED_EMBEDDING_OFFSET,
        embedding_scale,
    )
    normalized = reference_rmsnorm(activations, gains).outputs
    key = projection_reference(k_command, normalized)
    value = projection_reference(v_command, normalized)
    scales = np.frombuffer(
        read_image(int(rope_command["scale_addr"]), 128 * 2),
        dtype="<i2",
    ).astype(int).tolist()
    cosine, sine = accepted_runtime.absolute_coefficients_q15(0)
    coefficient_cos = (list(cosine) + list(cosine)) * 2
    coefficient_sin = (list(sine) + list(sine)) * 2
    rotated_key = reference_rope(
        RopeCase(
            name="layer0_position0_rope_k",
            sequence_position=0,
            activations=key,
            scales_q9=scales,
            cos_q15=coefficient_cos,
            sin_q15=coefficient_sin,
        )
    ).outputs
    metadata = read_image(int(kv_command["scale_addr"]), 16)
    expected = bytes((value & 0xFF) for value in rotated_key + value) + metadata
    expected_base = int(kv_command["scratch_addr"])
    record = records[ordinal]
    actual = bytearray()
    destination_digest = hashlib.sha256()
    address_mismatch = None
    for beat, (address, strobe, data) in enumerate(record["writes"]):
        expected_address = expected_base + beat * 16
        if address != expected_address or strobe != 0xFFFF:
            address_mismatch = {
                "beat": beat,
                "expected_address": expected_address,
                "actual_address": address,
                "expected_strobe": 0xFFFF,
                "actual_strobe": strobe,
            }
            break
        actual.extend(data)
        destination_digest.update(struct.pack("<QH", address, strobe))
        destination_digest.update(data)
    actual_bytes = bytes(actual)
    mismatch_offset = next(
        (index for index, (left, right) in enumerate(zip(expected, actual_bytes)) if left != right),
        None,
    )
    passed = (
        address_mismatch is None
        and len(actual_bytes) == len(expected)
        and mismatch_offset is None
        and destination_digest.digest() == record["destination_sha"]
        and not record["done_error"]
    )
    return {
        "status": "PASS_BIT_EXACT" if passed else "FAIL",
        "command_ordinal": ordinal,
        "operator": "layer_0.kv_write",
        "sequence_position": 0,
        "bytes_compared": len(expected),
        "expected_sha256": sha256_bytes(expected),
        "actual_sha256": sha256_bytes(actual_bytes),
        "destination_digest_matches_journal": destination_digest.digest() == record["destination_sha"],
        "address_mismatch": address_mismatch,
        "first_payload_mismatch_offset": mismatch_offset,
        "completion_tag": record["done_tag"],
        "completion_error": record["done_error"],
    }


def _pack_int8_bytes(values: list[int]) -> bytes:
    return bytes(value & 0xFF for value in values)


def _round_shift_even_signed(value: int, shift: int) -> int:
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


def _expected_attention_score(
    command: dict[str, Any],
    query: list[int],
    keys: list[list[int]],
) -> tuple[bytes, bool]:
    metadata = read_image(int(command["scale_addr"]), 16)
    if metadata[4] & 0xC0 or any(metadata[5:]):
        raise RuntimeError("attention-score metadata reserved bits differ")
    multiplier = struct.unpack_from("<i", metadata, 0)[0]
    right_shift = metadata[4] & 0x3F
    scaled_scores = []
    for key in keys:
        accumulator = sum(left * right for left, right in zip(query, key, strict=True))
        scaled_scores.append(
            _round_shift_even_signed(accumulator * multiplier, right_shift)
        )
    row_max = max(scaled_scores)
    scores = []
    saturation = False
    for scaled in scaled_scores:
        centered = scaled - row_max
        clipped = max(-32768, min(0, centered))
        saturation |= clipped != centered
        scores.append(clipped)
    payload = struct.pack(f"<{len(scores)}h", *scores)
    return payload + bytes(16 - len(payload)), saturation


def _compare_reference_command(
    command: dict[str, Any],
    record: dict[str, Any],
    expected: bytes,
    *,
    expected_saturation: bool,
    expected_base: int | None = None,
    expected_segments: list[tuple[int, bytes]] | None = None,
    expected_argmax_token: int | None = None,
    expected_argmax_logit: int | None = None,
    expected_generated_tokens: list[int] | None = None,
    expected_terminated: bool | None = None,
) -> dict[str, Any]:
    if expected_base is None:
        expected_base = int(command["dst_addr"])
    if expected_segments is None:
        expected_segments = [(expected_base, expected)]
    expected_stream = b"".join(payload for _, payload in expected_segments)
    expected_writes: list[tuple[int, int, bytes]] = []
    for segment_base, segment_payload in expected_segments:
        for offset in range(0, len(segment_payload), 16):
            chunk = segment_payload[offset : offset + 16]
            expected_writes.append(
                (
                    segment_base + offset,
                    0xFFFF if len(chunk) == 16 else (1 << len(chunk)) - 1,
                    chunk,
                )
            )
    actual = bytearray()
    destination_digest = hashlib.sha256()
    address_mismatch = None
    for beat, (address, strobe, data) in enumerate(record["writes"]):
        if beat >= len(expected_writes):
            address_mismatch = {
                "beat": beat,
                "expected_address": None,
                "actual_address": address,
                "expected_strobe": None,
                "actual_strobe": strobe,
            }
            break
        expected_address, expected_strobe, expected_chunk = expected_writes[beat]
        if address != expected_address or strobe != expected_strobe:
            address_mismatch = {
                "beat": beat,
                "expected_address": expected_address,
                "actual_address": address,
                "expected_strobe": expected_strobe,
                "actual_strobe": strobe,
            }
            break
        actual.extend(data[: len(expected_chunk)])
        destination_digest.update(struct.pack("<QH", address, strobe))
        destination_digest.update(data)
    actual_bytes = bytes(actual)
    mismatch_offset = next(
        (
            index
            for index, (expected_byte, actual_byte) in enumerate(
                zip(expected_stream, actual_bytes, strict=False)
            )
            if expected_byte != actual_byte
        ),
        None,
    )
    completion_tag_matches = record["done_tag"] == int(command["completion_tag"])
    completion_error_matches = record["done_error"] == expected_saturation
    saturation_matches = record["saturation"] == expected_saturation
    argmax_matches = (
        expected_argmax_token is None
        or (
            record["argmax_token"] == expected_argmax_token
            and record["argmax_logit"] == expected_argmax_logit
        )
    )
    generated_tokens_match = (
        expected_generated_tokens is None
        or record["generated_tokens"] == expected_generated_tokens
    )
    terminated_matches = (
        expected_terminated is None or record["terminated"] == expected_terminated
    )
    passed = (
        address_mismatch is None
        and len(record["writes"]) == len(expected_writes)
        and len(actual_bytes) == len(expected_stream)
        and mismatch_offset is None
        and destination_digest.digest() == record["destination_sha"]
        and completion_tag_matches
        and completion_error_matches
        and saturation_matches
        and argmax_matches
        and generated_tokens_match
        and terminated_matches
    )
    return {
        "status": "PASS_BIT_EXACT" if passed else "FAIL",
        "command_ordinal": int(command["ordinal"]),
        "operator": f"layer_{command['layer_id']}.{command['operator']}",
        "sequence_position": int(command["sequence_position"]),
        "bytes_compared": len(expected_stream),
        "expected_sha256": sha256_bytes(expected_stream),
        "actual_sha256": sha256_bytes(actual_bytes),
        "destination_digest_matches_journal": (
            destination_digest.digest() == record["destination_sha"]
        ),
        "address_mismatch": address_mismatch,
        "first_payload_mismatch_offset": mismatch_offset,
        "completion_tag": record["done_tag"],
        "expected_completion_tag": int(command["completion_tag"]),
        "completion_tag_matches": completion_tag_matches,
        "completion_error": record["done_error"],
        "expected_completion_error": expected_saturation,
        "completion_error_matches": completion_error_matches,
        "saturation": record["saturation"],
        "expected_saturation": expected_saturation,
        "saturation_matches": saturation_matches,
        "argmax_token": record["argmax_token"],
        "argmax_logit": record["argmax_logit"],
        "expected_argmax_token": expected_argmax_token,
        "expected_argmax_logit": expected_argmax_logit,
        "argmax_matches": argmax_matches,
        "generated_token_ids": record["generated_tokens"],
        "expected_generated_token_ids": expected_generated_tokens,
        "generated_token_history_matches": generated_tokens_match,
        "terminated": record["terminated"],
        "expected_terminated": expected_terminated,
        "terminated_matches": terminated_matches,
    }


def verify_first_attention_triplet(
    commands: list[dict[str, Any]],
    *,
    first_prompt_token: int,
    runtime_output: Path,
    journal_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(commands) < 10 or [command["operator"] for command in commands[7:10]] != [
        "attention_score",
        "softmax",
        "attention_value",
    ]:
        raise RuntimeError("first attention reference requires commands 0 through 9")
    records = (
        journal_records
        if journal_records is not None
        else parse_journal_v2(runtime_output / "progress.journal")
    )
    if len(records) < 10:
        return {
            "status": "NOT_REACHED",
            "first_command_ordinal": 7,
            "last_command_ordinal": 9,
            "commands_completed": len(records),
        }

    rmsnorm_command = commands[0]
    q_command = commands[1]
    k_command = commands[2]
    v_command = commands[3]
    rope_q_command = commands[4]
    rope_k_command = commands[5]
    scale_record = read_image(int(rmsnorm_command["scale_addr"]), 1808)
    gains = np.frombuffer(scale_record[:1792], dtype="<i2").astype(int).tolist()
    embedding_scale = np.float32(struct.unpack_from("<d", scale_record, 1800)[0])
    activations = quantized_embedding(
        first_prompt_token,
        PINNED_EMBEDDING_OFFSET,
        embedding_scale,
    )
    normalized = reference_rmsnorm(activations, gains).outputs
    query = projection_reference(q_command, normalized)
    key = projection_reference(k_command, normalized)
    value = projection_reference(v_command, normalized)
    cosine, sine = accepted_runtime.absolute_coefficients_q15(0)
    head_cosine = list(cosine) + list(cosine)
    head_sine = list(sine) + list(sine)
    query_scales = np.frombuffer(
        read_image(int(rope_q_command["scale_addr"]), 896 * 2),
        dtype="<i2",
    ).astype(int).tolist()
    key_scales = np.frombuffer(
        read_image(int(rope_k_command["scale_addr"]), 128 * 2),
        dtype="<i2",
    ).astype(int).tolist()
    rotated_query = reference_rope(
        RopeCase(
            name="layer0_position0_rope_q",
            sequence_position=0,
            activations=query,
            scales_q9=query_scales,
            cos_q15=head_cosine * 14,
            sin_q15=head_sine * 14,
        )
    ).outputs
    rotated_key = reference_rope(
        RopeCase(
            name="layer0_position0_rope_k",
            sequence_position=0,
            activations=key,
            scales_q9=key_scales,
            cos_q15=head_cosine * 2,
            sin_q15=head_sine * 2,
        )
    ).outputs

    score_command, softmax_command, value_command = commands[7:10]
    score_payload, score_saturation = _expected_attention_score(
        score_command,
        rotated_query[:64],
        [rotated_key[:64]],
    )
    score_values = list(struct.unpack_from("<h", score_payload, 0))
    softmax_result = reference_softmax(
        SoftmaxCase(name="layer0_position0_head0", scores_q6_9=score_values)
    )
    softmax_payload = struct.pack(
        "<8H", *softmax_result.probabilities_q0_15
    )
    value_result = reference_attention_value(
        AttentionValueCase(
            name="layer0_position0_head0",
            probabilities_q15=softmax_result.probabilities_q0_15[:1],
            values=[value[:64]],
        )
    )
    value_payload = _pack_int8_bytes(value_result.outputs)
    checks = [
        _compare_reference_command(
            score_command,
            records[7],
            score_payload,
            expected_saturation=score_saturation,
        ),
        _compare_reference_command(
            softmax_command,
            records[8],
            softmax_payload,
            expected_saturation=False,
        ),
        _compare_reference_command(
            value_command,
            records[9],
            value_payload,
            expected_saturation=value_result.saturation_seen,
        ),
    ]
    passed = all(check["status"] == "PASS_BIT_EXACT" for check in checks)
    first_mismatch = next(
        (check["command_ordinal"] for check in checks if check["status"] == "FAIL"),
        None,
    )
    return {
        "status": "PASS_BIT_EXACT" if passed else "FAIL",
        "first_command_ordinal": 7,
        "last_command_ordinal": 9,
        "commands_compared": len(checks),
        "bytes_compared": sum(int(check["bytes_compared"]) for check in checks),
        "first_mismatch_ordinal": first_mismatch,
        "checks": checks,
    }


class _PositionReferenceMemory:
    def __init__(self, image: mmap.mmap) -> None:
        self.image = image
        self.dynamic: dict[int, int] = {}

    def read(self, address: int, size: int) -> bytes:
        for base, region_size, file_offset in IMAGE_REGIONS:
            if base <= address and address - base + size <= region_size:
                offset = file_offset + address - base
                return bytes(self.image[offset : offset + size])
        missing = [
            address + index
            for index in range(size)
            if address + index not in self.dynamic
        ]
        if missing:
            raise RuntimeError(
                "position reference read uninitialized dynamic memory at "
                f"0x{missing[0]:016x}"
            )
        return bytes(self.dynamic[address + index] for index in range(size))

    def write(self, address: int, payload: bytes) -> None:
        for index, value in enumerate(payload):
            self.dynamic[address + index] = value


def _signed_int8_list(raw: bytes) -> list[int]:
    return np.frombuffer(raw, dtype=np.int8).astype(int).tolist()


def _dynamic_rms_output_sidecar(
    command: dict[str, Any], host_plan_sidecar: bytes
) -> bytes:
    sidecar = bytearray(64)
    sidecar[0:4] = b"BFP1"
    sidecar[4] = 1
    sidecar[5] = 128
    sidecar[6] = 7
    sidecar[8:10] = int(command["completion_tag"]).to_bytes(2, "little")
    sidecar[10] = int(command["layer_id"])
    sidecar[11] = int(command["opcode"])
    sidecar[12:16] = DYNAMIC_SCALE32_INITIAL_ELEMENTS.to_bytes(4, "little")
    sidecar[16:24] = dynamic_scale32_model_identity(
        accepted_runtime.EXPECTED_MODEL_SHA256
    ).to_bytes(8, "little")
    sidecar[24:31] = host_plan_sidecar[24:31]
    return bytes(sidecar)


def _dynamic_projection_producer_offset(command: dict[str, Any]) -> int:
    operator = command.get("operator")
    if operator == "q_proj":
        return 1
    if operator == "k_proj":
        return 2
    if operator == "v_proj":
        return 3
    raise RuntimeError("dynamic projection sidecar is only defined for Q/K/V")


def _apply_dynamic_projection_sidecar(
    command: dict[str, Any], payload: bytes, sidecar: bytes
) -> list[int]:
    expected_identity = dynamic_scale32_model_identity(
        accepted_runtime.EXPECTED_MODEL_SHA256
    )
    if (
        len(sidecar) != 64
        or sidecar[0:4] != b"BFP1"
        or sidecar[4:8] != bytes((1, 128, 7, 0))
        or int.from_bytes(sidecar[8:10], "little")
        != (
            int(command["completion_tag"])
            - _dynamic_projection_producer_offset(command)
        ) & 0xFFFF
        or sidecar[10] != int(command["layer_id"])
        or sidecar[11] != 2
        or int.from_bytes(sidecar[12:16], "little")
        != DYNAMIC_SCALE32_INITIAL_ELEMENTS
        or int.from_bytes(sidecar[16:24], "little") != expected_identity
        or any(sidecar[31:])
    ):
        raise RuntimeError("dynamic Q/K/V sidecar reference contract differs")
    activations = _signed_int8_list(payload)
    restored: list[int] = []
    for index, value in enumerate(activations):
        raw_delta = sidecar[24 + index // 128]
        delta = raw_delta - 256 if raw_delta & 0x80 else raw_delta
        scaled = (
            value << delta
            if delta >= 0
            else _round_shift_even_signed(value, -delta)
        )
        if not -128 <= scaled <= 127:
            raise RuntimeError("dynamic Q/K/V sidecar application overflows int8")
        restored.append(scaled)
    return restored


def _projection_reference_numpy(
    command: dict[str, Any],
    activations: list[int],
    memory: _PositionReferenceMemory,
) -> tuple[bytes, bool]:
    outputs = int(command["n"])
    reduction = int(command["k"])
    packed = np.frombuffer(
        memory.read(int(command["src1_addr"]), outputs * reduction // 2),
        dtype=np.uint8,
    )
    weights = np.empty(packed.size * 2, dtype=np.int8)
    weights[0::2] = (packed & 0x0F).astype(np.int8)
    weights[1::2] = (packed >> 4).astype(np.int8)
    weights[weights >= 8] -= 16
    matrix = weights.reshape(outputs, reduction).astype(np.int32)
    activation_array = np.asarray(activations, dtype=np.int32)
    accumulators = (matrix @ activation_array).astype(np.int64)
    metadata = memory.read(int(command["scale_addr"]), outputs * 16)
    multipliers = np.ndarray(
        shape=(outputs,), dtype="<i4", buffer=metadata, offset=0, strides=(16,)
    ).astype(np.int64)
    shifts = np.ndarray(
        shape=(outputs,), dtype=np.uint8, buffer=metadata, offset=4, strides=(16,)
    )
    zero_points = np.ndarray(
        shape=(outputs,), dtype=np.int8, buffer=metadata, offset=5, strides=(16,)
    ).astype(np.int64)
    biases = np.ndarray(
        shape=(outputs,), dtype="<i4", buffer=metadata, offset=6, strides=(16,)
    ).astype(np.int64)
    reserved_shift_bits = shifts & 0xC0
    if np.any(reserved_shift_bits) or any(
        metadata[index * 16 + reserved]
        for index in range(outputs)
        for reserved in range(10, 16)
    ):
        raise RuntimeError("projection metadata reserved bits differ")
    shifts = shifts & 0x3F
    accumulators += biases
    values: list[int] = []
    saturation = False
    for accumulator, multiplier, shift, zero_point in zip(
        accumulators.tolist(),
        multipliers.tolist(),
        shifts.tolist(),
        zero_points.tolist(),
        strict=True,
    ):
        scaled = _round_shift_even_signed(accumulator * multiplier, shift)
        shifted = scaled + zero_point
        clipped = max(-128, min(127, shifted))
        saturation |= clipped != shifted
        values.append(clipped)
    return _pack_int8_bytes(values), saturation


def _position_expected_payload(
    command: dict[str, Any],
    memory: _PositionReferenceMemory,
    compose_states: dict[tuple[int, int, int], AttentionComposePhaseReplay],
) -> tuple[bytes, bool, int]:
    operator = str(command["operator"])
    src0 = int(command["src0_addr"])
    src1 = int(command["src1_addr"])
    dst = int(command["dst_addr"])
    n = int(command["n"])
    k = int(command["k"])
    if operator in {"input_rmsnorm", "post_attention_rmsnorm", "final_rmsnorm"}:
        activations = _signed_int8_list(memory.read(src0, n))
        gains = np.frombuffer(
            memory.read(int(command["scale_addr"]), n * 2), dtype="<i2"
        ).astype(int).tolist()
        result = reference_rmsnorm(activations, gains)
        if operator == "input_rmsnorm" and int(command["flags"]) & DYNAMIC_SCALE32_FLAG:
            host_plan = memory.read(src0 - 64, 64)
            if (
                host_plan[0:4] != b"BFP1"
                or host_plan[4:8] != bytes((1, 128, 7, 0))
                or int.from_bytes(host_plan[8:10], "little")
                != dynamic_scale32_host_plan_tag(int(command["sequence_position"]))
                or host_plan[10] != DYNAMIC_SCALE32_HOST_PLAN_LAYER
                or host_plan[11] != DYNAMIC_SCALE32_HOST_PLAN_OPCODE
                or int.from_bytes(host_plan[12:16], "little") != 896
                or int.from_bytes(host_plan[16:24], "little")
                != dynamic_scale32_model_identity(accepted_runtime.EXPECTED_MODEL_SHA256)
                or any(host_plan[31:])
            ):
                raise RuntimeError("dynamic RMS host-plan reference contract differs")
            outputs: list[int] = []
            for group, raw_delta in enumerate(host_plan[24:31]):
                delta = raw_delta - 256 if raw_delta & 0x80 else raw_delta
                group_values = result.outputs[group * 128 : (group + 1) * 128]
                for value in group_values:
                    quantized = quantize_for_delta(value, delta)
                    if not -127 <= quantized <= 127:
                        raise RuntimeError(
                            "dynamic RMSNorm output is not representable"
                        )
                    outputs.append(quantized)
            return _pack_int8_bytes(outputs), result.saturation_seen, dst
        return _pack_int8_bytes(result.outputs), result.saturation_seen, dst
    if int(command["opcode"]) == 1:
        source_payload = memory.read(src0, k)
        activations = (
            _apply_dynamic_projection_sidecar(
                command,
                source_payload,
                memory.read(src0 - 64, 64),
            )
            if int(command["flags"]) & DYNAMIC_SCALE32_FLAG
            else _signed_int8_list(source_payload)
        )
        payload, saturation = _projection_reference_numpy(command, activations, memory)
        return payload, saturation, dst
    if operator in {"rope_q", "rope_k"}:
        activations = _signed_int8_list(memory.read(src0, n))
        scales = np.frombuffer(
            memory.read(int(command["scale_addr"]), n * 2), dtype="<i2"
        ).astype(int).tolist()
        position = int(command["sequence_position"])
        cosine, sine = accepted_runtime.absolute_coefficients_q15(position)
        head_cosine = list(cosine) + list(cosine)
        head_sine = list(sine) + list(sine)
        result = reference_rope(
            RopeCase(
                name=f"position{position}_{operator}",
                sequence_position=position,
                activations=activations,
                scales_q9=scales,
                cos_q15=head_cosine * (n // 64),
                sin_q15=head_sine * (n // 64),
            )
        )
        return _pack_int8_bytes(result.outputs), result.saturation_seen, dst
    if operator == "kv_write":
        payload = (
            memory.read(src0, 128)
            + memory.read(src1, 128)
            + memory.read(int(command["scale_addr"]), 16)
        )
        expected_base = int(command["scratch_addr"]) + int(command["sequence_position"]) * 272
        return payload, False, expected_base
    if operator == "attention_score":
        query = _signed_int8_list(memory.read(src0, 64))
        keys = [
            _signed_int8_list(memory.read(src1 + context * 64, 64))
            for context in range(n)
        ]
        payload, saturation = _expected_attention_score(command, query, keys)
        return payload, saturation, dst
    if operator == "softmax":
        scores = list(struct.unpack(f"<{n}h", memory.read(src0, 16)[: n * 2]))
        result = reference_softmax(
            SoftmaxCase(name="position0_softmax", scores_q6_9=scores)
        )
        return struct.pack("<8H", *result.probabilities_q0_15), False, dst
    if operator == "attention_value":
        probabilities = list(struct.unpack(f"<{n}H", memory.read(src0, 16)[: n * 2]))
        values = [
            _signed_int8_list(memory.read(src1 + context * 64, 64))
            for context in range(n)
        ]
        result = reference_attention_value(
            AttentionValueCase(
                name="position0_attention_value",
                probabilities_q15=probabilities,
                values=values,
            )
        )
        return _pack_int8_bytes(result.outputs), result.saturation_seen, dst
    if operator == "attention_compose":
        head = int(command["query_head"])
        key = (
            int(command["token_step"]),
            int(command["layer_id"]),
            head,
        )
        state = compose_states.setdefault(key, AttentionComposePhaseReplay())
        scores = list(struct.unpack(f"<{n}h", memory.read(src0, 16)[: n * 2]))
        phase = int(command["flags"])
        values = None
        if phase >= 4:
            values = [
                _signed_int8_list(memory.read(src1 + context * 64, 64))
                for context in range(n)
            ]
        result = state.apply(phase, scores, values)
        payload = b"" if result.outputs is None else _pack_int8_bytes(result.outputs)
        return payload, result.saturation_seen, dst
    if operator in {"attention_residual_add", "mlp_residual_add"}:
        lhs = _signed_int8_list(memory.read(src0, n))
        rhs = _signed_int8_list(memory.read(src1, n))
        outputs, saturation = reference_residual_add(lhs, rhs)
        return _pack_int8_bytes(outputs), saturation, dst
    if operator == "silu_gate":
        gate = _signed_int8_list(memory.read(src0, n))
        up = _signed_int8_list(memory.read(src1, n))
        metadata = memory.read(int(command["scale_addr"]), 16)
        if metadata[4] & 0xC0 or any(metadata[6:]):
            raise RuntimeError("SiLU metadata reserved bits differ")
        input_scales = silu_input_scale_table()[int(command["layer_id"])]
        result = reference_silu_gate_scaled_int8(
            SiluGateCase(
                name="position0_silu_gate",
                gate_q6_9=gate,
                up_q6_9=up,
                multiplier=struct.unpack_from("<i", metadata, 0)[0],
                right_shift=metadata[4] & 0x3F,
                output_zero_point=struct.unpack_from("<b", metadata, 5)[0],
            ),
            gate_scale_numerator=int(input_scales["gate_scale_numerator"]),
            up_scale_numerator=int(input_scales["up_scale_numerator"]),
        )
        return _pack_int8_bytes(result.outputs), result.saturation_seen, dst
    raise RuntimeError(f"position reference does not implement {operator}")


def verify_positions_through_argmax(
    commands: list[dict[str, Any]],
    *,
    prompt_tokens: list[int],
    through_position: int,
    runtime_output: Path,
    journal_records: list[dict[str, Any]] | None = None,
    max_new_tokens: int | None = None,
    termination_token_ids: list[int] | tuple[int, ...] = TERMINATION_TOKEN_IDS,
    required_final_operator: str = "lm_head_tile",
) -> dict[str, Any]:
    position_commands = [
        command
        for command in commands
        if int(command["token_step"]) <= through_position
    ]
    final_position_commands = [
        command
        for command in position_commands
        if int(command["token_step"]) == through_position
    ]
    maximum_position = max((int(command["token_step"]) for command in commands), default=-1)
    if (
        through_position < 0
        or not prompt_tokens
        or through_position > maximum_position
        or not final_position_commands
        or final_position_commands[-1]["operator"] != required_final_operator
    ):
        raise RuntimeError("position reference command stream is incomplete")
    generated_position = through_position >= len(prompt_tokens)
    expected_target_kv_publications = sum(
        command["operator"] == "kv_write" for command in final_position_commands
    )
    expected_target_key_reads = sum(
        command["operator"] == "attention_score" for command in final_position_commands
    )
    expected_target_value_reads = sum(
        command["operator"] == "attention_compose" and int(command["flags"]) >= 4
        for command in final_position_commands
    )
    records = (
        journal_records
        if journal_records is not None
        else parse_journal_v2(runtime_output / "progress.journal")
    )
    prompt_ds32_plans = {
        int(case["chat_position"]): case
        for case in audit_prompt_dynamic_scale32_applicability(
            prompt_tokens,
            PINNED_EMBEDDING_OFFSET,
        )["cases"]
    }
    comparison_digest = hashlib.sha256()
    operator_totals: dict[str, dict[str, int]] = {}
    compose_phase_totals: dict[str, dict[str, int]] = {}
    compose_phase_totals_by_position: dict[str, dict[str, dict[str, int]]] = {}
    first_compose_sequence: list[dict[str, Any]] = []
    target_compose_sequence: list[dict[str, Any]] = []
    bytes_compared = 0
    commands_compared = 0
    argmax_token = -1
    argmax_logit = -129
    last_verified: dict[str, Any] | None = None
    operator_last_verified: dict[str, dict[str, Any]] = {}
    layer_operator_last_verified: dict[str, dict[str, Any]] = {}
    expected_generated_tokens: list[int] = []
    expected_terminated = False
    generated_feedback: list[dict[str, Any]] = []
    token_history_transitions: list[dict[str, Any]] = []
    token_history_checks = 0
    token_history_mismatches: list[dict[str, Any]] = []
    kv_publication_bases: dict[tuple[int, int], int] = {}
    target_kv_publications: list[dict[str, Any]] = []
    target_key_reads = 0
    target_value_reads = 0
    target_self_key_reads = 0
    target_self_value_reads = 0
    kv_reuse_address_mismatches: list[dict[str, Any]] = []

    def generation_evidence() -> dict[str, Any]:
        target_kv_reuse_complete = (
            len(target_kv_publications) == expected_target_kv_publications
            and target_key_reads == expected_target_key_reads
            and target_value_reads == expected_target_value_reads
            and target_self_key_reads > 0
            and target_self_value_reads > 0
            and not kv_reuse_address_mismatches
        )
        return {
            "generated_token_ids": list(expected_generated_tokens),
            "terminated": expected_terminated,
            "generated_token_feedback": generated_feedback,
            "journal_token_history": {
                "status": "PASS_APPEND_ONLY_EXACT" if not token_history_mismatches else "FAIL",
                "commands_checked": token_history_checks,
                "transitions": token_history_transitions,
                "final_expected_token_ids": list(expected_generated_tokens),
                "mismatches": token_history_mismatches,
            },
            "generated_position_kv": {
                "status": (
                    "PASS_PUBLICATION_AND_REUSE"
                    if not generated_position or target_kv_reuse_complete
                    else "INCOMPLETE"
                ),
                "position": through_position if generated_position else None,
                "publications_expected": expected_target_kv_publications,
                "publications_checked": len(target_kv_publications),
                "publication_records": target_kv_publications,
                "key_reads_expected": expected_target_key_reads,
                "key_reads_checked": target_key_reads,
                "value_reads_expected": expected_target_value_reads,
                "value_reads_checked": target_value_reads,
                "self_key_reads_checked": target_self_key_reads,
                "self_value_reads_checked": target_self_value_reads,
                "address_mismatches": kv_reuse_address_mismatches,
            },
            "target_position_kv": {
                "status": (
                    "PASS_PUBLICATION_AND_CONTEXT_REUSE"
                    if target_kv_reuse_complete
                    else "INCOMPLETE"
                ),
                "position": through_position,
                "publications_expected": expected_target_kv_publications,
                "publications_checked": len(target_kv_publications),
                "publication_records": target_kv_publications,
                "key_reads_expected": expected_target_key_reads,
                "key_reads_checked": target_key_reads,
                "prior_key_reads_checked": target_key_reads - target_self_key_reads,
                "value_reads_expected": expected_target_value_reads,
                "value_reads_checked": target_value_reads,
                "prior_value_reads_checked": target_value_reads - target_self_value_reads,
                "self_key_reads_checked": target_self_key_reads,
                "self_value_reads_checked": target_self_value_reads,
                "address_mismatches": kv_reuse_address_mismatches,
            },
        }

    with accepted_runtime.IMAGE.open("rb") as image_handle:
        with mmap.mmap(image_handle.fileno(), 0, access=mmap.ACCESS_READ) as image_map:
            memory = _PositionReferenceMemory(image_map)
            scale_record = memory.read(int(position_commands[0]["scale_addr"]), 1808)
            embedding_scale = np.float32(struct.unpack_from("<d", scale_record, 1800)[0])
            compose_states: dict[
                tuple[int, int, int], AttentionComposePhaseReplay
            ] = {}
            loaded_position = -1

            for command in position_commands:
                ordinal = int(command["ordinal"])
                if ordinal >= len(records):
                    return {
                        "status": "NOT_REACHED",
                        "position": through_position,
                        "commands_expected": len(position_commands),
                        "commands_completed": len(records),
                        "commands_compared": commands_compared,
                        "bytes_compared": bytes_compared,
                        "next_command_ordinal": ordinal,
                        "next_operator": command["operator"],
                        "comparison_sha256": comparison_digest.hexdigest(),
                        "operator_totals": operator_totals,
                        "attention_compose_phase_totals": compose_phase_totals,
                        "attention_compose_phase_totals_by_position": compose_phase_totals_by_position,
                        "first_attention_compose_sequence": first_compose_sequence,
                        "target_attention_compose_sequence": target_compose_sequence,
                        "last_verified": last_verified,
                        "operator_last_verified": operator_last_verified,
                        "layer_operator_last_verified": layer_operator_last_verified,
                        "argmax": {
                            "token_id": argmax_token,
                            "logit_s8": argmax_logit,
                        },
                        **generation_evidence(),
                    }
                position = int(command["token_step"])
                if loaded_position != position:
                    token, token_source, generated_index = resolve_position_token(
                        prompt_tokens,
                        expected_generated_tokens,
                        position,
                    )
                    embedding = quantized_embedding(
                        token,
                        PINNED_EMBEDDING_OFFSET,
                        embedding_scale,
                    )
                    embedding_raw = _pack_int8_bytes(embedding)
                    memory.write(
                        int(command["src0_addr"]),
                        embedding_raw,
                    )
                    if (
                        position < len(prompt_tokens)
                        and int(command["flags"]) & DYNAMIC_SCALE32_FLAG
                    ):
                        plan = prompt_ds32_plans[position]
                        host_plan_sidecar = _build_prompt_dynamic_scale32_sidecar(
                            payload_addr=int(command["src0_addr"]),
                            model_identity=dynamic_scale32_model_identity(
                                accepted_runtime.EXPECTED_MODEL_SHA256
                            ),
                            sequence_position=position,
                            deltas=plan["selected_deltas"],
                        )
                        memory.write(int(command["src0_addr"]) - 64, host_plan_sidecar)
                    if token_source == "generated_argmax":
                        generated_feedback.append(
                            {
                                "position": position,
                                "generated_index": generated_index,
                                "token_id": token,
                                "source": token_source,
                                "embedding_s8_sha256": sha256_bytes(embedding_raw),
                                "first_command_ordinal": ordinal,
                                "first_command_operator": command["operator"],
                            }
                        )
                    loaded_position = position
                    argmax_token = -1
                    argmax_logit = -129
                payload, saturation, expected_base = _position_expected_payload(
                    command, memory, compose_states
                )
                dynamic_output_sidecar = None
                expected_segments = None
                if (
                    command["operator"] == "input_rmsnorm"
                    and int(command["flags"]) & DYNAMIC_SCALE32_FLAG
                ):
                    dynamic_output_sidecar = _dynamic_rms_output_sidecar(
                        command,
                        memory.read(int(command["src0_addr"]) - 64, 64),
                    )
                    expected_segments = [
                        (expected_base, payload),
                        (expected_base - 64, dynamic_output_sidecar),
                    ]
                reuse_address_check: dict[str, Any] | None = None
                if command["operator"] == "attention_score":
                    context_token = int(command["context_token"])
                    layer = int(command["layer_id"])
                    kv_head = int(command["query_head"]) // 7
                    publication_base = kv_publication_bases.get((context_token, layer))
                    expected_source = (
                        None if publication_base is None else publication_base + kv_head * 64
                    )
                    reuse_address_check = {
                        "kind": "key",
                        "context_token": context_token,
                        "expected_source": expected_source,
                        "actual_source": int(command["src1_addr"]),
                    }
                elif command["operator"] == "attention_compose" and int(command["flags"]) >= 4:
                    context_token = int(command["context_token"])
                    layer = int(command["layer_id"])
                    kv_head = int(command["query_head"]) // 7
                    publication_base = kv_publication_bases.get((context_token, layer))
                    expected_source = (
                        None
                        if publication_base is None
                        else publication_base + 128 + kv_head * 64
                    )
                    reuse_address_check = {
                        "kind": "value",
                        "context_token": context_token,
                        "expected_source": expected_source,
                        "actual_source": int(command["src1_addr"]),
                    }
                if command["operator"] == "lm_head_tile":
                    if int(command["vocab_tile"]) == 0:
                        argmax_token = -1
                        argmax_logit = -129
                    for lane, raw_logit in enumerate(payload):
                        token = int(command["vocab_tile"]) * 32 + lane
                        logit = raw_logit - 256 if raw_logit >= 128 else raw_logit
                        if (
                            argmax_token < 0
                            or logit > argmax_logit
                            or (logit == argmax_logit and token < argmax_token)
                        ):
                            argmax_token = token
                            argmax_logit = logit
                    if (
                        int(command["vocab_tile"]) == LM_HEAD_TILE_COUNT - 1
                        and position >= len(prompt_tokens) - 1
                    ):
                        generated_index = position - (len(prompt_tokens) - 1)
                        if generated_index != len(expected_generated_tokens):
                            raise RuntimeError("generated-token reference sequence is noncanonical")
                        if (
                            max_new_tokens is not None
                            and len(expected_generated_tokens) >= max_new_tokens
                        ):
                            raise RuntimeError("generated-token reference exceeds package bound")
                        expected_generated_tokens.append(argmax_token)
                        expected_terminated = argmax_token in termination_token_ids
                        token_history_transitions.append(
                            {
                                "command_ordinal": ordinal,
                                "position": position,
                                "argmax_token": argmax_token,
                                "argmax_logit": argmax_logit,
                                "generated_token_ids": list(expected_generated_tokens),
                                "terminated": expected_terminated,
                            }
                        )
                check = _compare_reference_command(
                    command,
                    records[ordinal],
                    payload,
                    expected_saturation=saturation,
                    expected_base=expected_base,
                    expected_segments=expected_segments,
                    expected_argmax_token=argmax_token,
                    expected_argmax_logit=argmax_logit,
                    expected_generated_tokens=expected_generated_tokens,
                    expected_terminated=expected_terminated,
                )
                token_history_checks += 1
                if (
                    not check["generated_token_history_matches"]
                    or not check["terminated_matches"]
                ):
                    token_history_mismatches.append(
                        {
                            "command_ordinal": ordinal,
                            "actual_generated_token_ids": check["generated_token_ids"],
                            "expected_generated_token_ids": check[
                                "expected_generated_token_ids"
                            ],
                            "actual_terminated": check["terminated"],
                            "expected_terminated": check["expected_terminated"],
                        }
                    )
                if reuse_address_check is not None:
                    reuse_address_check["command_ordinal"] = ordinal
                    address_matches = (
                        reuse_address_check["expected_source"]
                        == reuse_address_check["actual_source"]
                    )
                    check["kv_reuse"] = {
                        **reuse_address_check,
                        "address_matches": address_matches,
                    }
                    if not address_matches:
                        check["status"] = "FAIL"
                        kv_reuse_address_mismatches.append(check["kv_reuse"])
                    if position == through_position:
                        if reuse_address_check["kind"] == "key":
                            target_key_reads += 1
                            if int(reuse_address_check["context_token"]) == position:
                                target_self_key_reads += 1
                        else:
                            target_value_reads += 1
                            if int(reuse_address_check["context_token"]) == position:
                                target_self_value_reads += 1
                comparison_digest.update(canonical_bytes(check))
                total = operator_totals.setdefault(
                    str(command["operator"]), {"commands": 0, "bytes": 0}
                )
                total["commands"] += 1
                total["bytes"] += int(check["bytes_compared"])
                if command["operator"] == "attention_compose":
                    phase = str(int(command["flags"]))
                    phase_total = compose_phase_totals.setdefault(
                        phase, {"commands": 0, "bytes": 0}
                    )
                    phase_total["commands"] += 1
                    phase_total["bytes"] += len(payload)
                    position_phase_totals = compose_phase_totals_by_position.setdefault(
                        str(position), {}
                    )
                    position_phase_total = position_phase_totals.setdefault(
                        phase, {"commands": 0, "bytes": 0}
                    )
                    position_phase_total["commands"] += 1
                    position_phase_total["bytes"] += len(payload)
                    if len(first_compose_sequence) < 6:
                        first_compose_sequence.append(
                            {
                                "phase": int(command["flags"]),
                                "context_token": int(command["context_token"]),
                                **check,
                            }
                        )
                    target_sequence_limit = 3 * (through_position + 1)
                    if position == through_position and len(target_compose_sequence) < target_sequence_limit:
                        target_compose_sequence.append(
                            {
                                "phase": int(command["flags"]),
                                "context_token": int(command["context_token"]),
                                **check,
                            }
                        )
                bytes_compared += int(check["bytes_compared"])
                commands_compared += 1
                if check["status"] != "PASS_BIT_EXACT":
                    return {
                        "status": "FAIL",
                        "position": through_position,
                        "commands_expected": len(position_commands),
                        "commands_completed": len(records),
                        "commands_compared": commands_compared,
                        "bytes_compared": bytes_compared,
                        "first_mismatch_ordinal": ordinal,
                        "first_mismatch": check,
                        "comparison_sha256": comparison_digest.hexdigest(),
                        "operator_totals": operator_totals,
                        "attention_compose_phase_totals": compose_phase_totals,
                        "attention_compose_phase_totals_by_position": compose_phase_totals_by_position,
                        "first_attention_compose_sequence": first_compose_sequence,
                        "target_attention_compose_sequence": target_compose_sequence,
                        "last_verified": last_verified,
                        "operator_last_verified": operator_last_verified,
                        "layer_operator_last_verified": layer_operator_last_verified,
                        "argmax": {
                            "token_id": argmax_token,
                            "logit_s8": argmax_logit,
                        },
                        **generation_evidence(),
                    }
                if payload:
                    memory.write(expected_base, payload)
                if dynamic_output_sidecar is not None:
                    memory.write(expected_base - 64, dynamic_output_sidecar)
                if command["operator"] == "kv_write":
                    kv_publication_bases[(position, int(command["layer_id"]))] = expected_base
                    if position == through_position:
                        target_kv_publications.append(
                            {
                                "command_ordinal": ordinal,
                                "layer_id": int(command["layer_id"]),
                                "sequence_position": int(command["sequence_position"]),
                                "destination_base": expected_base,
                                "payload_sha256": sha256_bytes(payload),
                                "status": check["status"],
                            }
                        )
                if (
                    generated_feedback
                    and ordinal == int(generated_feedback[-1]["first_command_ordinal"])
                ):
                    generated_feedback[-1]["first_command_status"] = check["status"]
                last_verified = {
                    "command_ordinal": ordinal,
                    "operator": command["operator"],
                    "expected_sha256": check["expected_sha256"],
                    "actual_sha256": check["actual_sha256"],
                }
                operator_last_verified[str(command["operator"])] = last_verified
                layer_operator_last_verified[
                    f"{int(command['layer_id'])}.{command['operator']}"
                ] = last_verified

    if generated_feedback:
        progress_path = runtime_output / "progress.json"
        if not progress_path.is_file():
            raise RuntimeError("runtime progress metadata is missing")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        feedback = generated_feedback[-1]
        runtime_feedback = {
            "embedding_token": int(progress["embedding_token"]),
            "embedding_s8_sha256": str(progress["embedding_s8_sha256"]),
            "expected_embedding_token": int(feedback["token_id"]),
            "expected_embedding_s8_sha256": str(feedback["embedding_s8_sha256"]),
        }
        runtime_feedback["matches"] = (
            runtime_feedback["embedding_token"]
            == runtime_feedback["expected_embedding_token"]
            and runtime_feedback["embedding_s8_sha256"]
            == runtime_feedback["expected_embedding_s8_sha256"]
        )
        feedback["runtime_progress"] = runtime_feedback
        if not runtime_feedback["matches"]:
            return {
                "status": "FAIL",
                "position": through_position,
                "commands_expected": len(position_commands),
                "commands_completed": len(records),
                "commands_compared": commands_compared,
                "bytes_compared": bytes_compared,
                "first_mismatch_ordinal": None,
                "first_mismatch": {"category": "generated_token_feedback_metadata"},
                "comparison_sha256": comparison_digest.hexdigest(),
                "operator_totals": operator_totals,
                "attention_compose_phase_totals": compose_phase_totals,
                "attention_compose_phase_totals_by_position": compose_phase_totals_by_position,
                "first_attention_compose_sequence": first_compose_sequence,
                "target_attention_compose_sequence": target_compose_sequence,
                "last_verified": last_verified,
                "operator_last_verified": operator_last_verified,
                "layer_operator_last_verified": layer_operator_last_verified,
                "argmax": {"token_id": argmax_token, "logit_s8": argmax_logit},
                **generation_evidence(),
            }

    final_generation_evidence = generation_evidence()
    if (
        generated_position
        and final_generation_evidence["generated_position_kv"]["status"]
        != "PASS_PUBLICATION_AND_REUSE"
    ):
        return {
            "status": "FAIL",
            "position": through_position,
            "commands_expected": len(position_commands),
            "commands_completed": len(records),
            "commands_compared": commands_compared,
            "bytes_compared": bytes_compared,
            "first_mismatch_ordinal": None,
            "first_mismatch": {"category": "generated_position_kv_incomplete"},
            "comparison_sha256": comparison_digest.hexdigest(),
            "operator_totals": operator_totals,
            "attention_compose_phase_totals": compose_phase_totals,
            "attention_compose_phase_totals_by_position": compose_phase_totals_by_position,
            "first_attention_compose_sequence": first_compose_sequence,
            "target_attention_compose_sequence": target_compose_sequence,
            "last_verified": last_verified,
            "operator_last_verified": operator_last_verified,
            "layer_operator_last_verified": layer_operator_last_verified,
            "argmax": {"token_id": argmax_token, "logit_s8": argmax_logit},
            **final_generation_evidence,
        }

    return {
        "status": "PASS_BIT_EXACT",
        "position": through_position,
        "commands_expected": len(position_commands),
        "commands_completed": len(records),
        "commands_compared": commands_compared,
        "bytes_compared": bytes_compared,
        "first_mismatch_ordinal": None,
        "comparison_sha256": comparison_digest.hexdigest(),
        "operator_totals": operator_totals,
        "attention_compose_phase_totals": compose_phase_totals,
        "attention_compose_phase_totals_by_position": compose_phase_totals_by_position,
        "first_attention_compose_sequence": first_compose_sequence,
        "target_attention_compose_sequence": target_compose_sequence,
        "last_verified": last_verified,
        "operator_last_verified": operator_last_verified,
        "layer_operator_last_verified": layer_operator_last_verified,
        "argmax": {"token_id": argmax_token, "logit_s8": argmax_logit},
        **final_generation_evidence,
    }


def focused_complete_layer_prefix_commands(
    commands: list[dict[str, Any]],
    *,
    through_position: int,
    layer_count: int,
) -> list[dict[str, Any]]:
    if not 1 <= layer_count <= 24:
        raise RuntimeError("focused layer-prefix count must be between 1 and 24")
    selected = [
        command
        for command in commands
        if int(command["token_step"]) <= through_position
        and int(command["layer_id"]) < layer_count
    ]
    if (
        through_position < 0
        or not selected
        or int(selected[0]["token_step"]) != 0
        or int(selected[-1]["token_step"]) != through_position
    ):
        raise RuntimeError("focused complete layer-prefix prompt prefix is incomplete")
    for position in range(through_position + 1):
        for layer in range(layer_count):
            layer_commands = [
                command
                for command in selected
                if int(command["token_step"]) == position
                and int(command["layer_id"]) == layer
            ]
            if (
                not layer_commands
                or layer_commands[0]["operator"] != "input_rmsnorm"
                or layer_commands[-1]["operator"] != "mlp_residual_add"
            ):
                raise RuntimeError(
                    "focused complete layer-prefix prompt prefix is incomplete"
                )
    normalized = []
    for ordinal, command in enumerate(selected):
        record = dict(command)
        record["source_ordinal"] = int(command["ordinal"])
        record["ordinal"] = ordinal
        normalized.append(record)
    return normalized


def focused_complete_layer0_commands(
    commands: list[dict[str, Any]],
    *,
    through_position: int,
) -> list[dict[str, Any]]:
    return focused_complete_layer_prefix_commands(
        commands,
        through_position=through_position,
        layer_count=1,
    )


def verify_complete_layer_prefix_prompt_prefix(
    commands: list[dict[str, Any]],
    *,
    prompt_tokens: list[int],
    through_position: int,
    layer_count: int,
    runtime_output: Path,
) -> dict[str, Any]:
    selected = focused_complete_layer_prefix_commands(
        commands,
        through_position=through_position,
        layer_count=layer_count,
    )
    result = verify_positions_through_argmax(
        selected,
        prompt_tokens=prompt_tokens,
        through_position=through_position,
        runtime_output=runtime_output,
        required_final_operator="mlp_residual_add",
    )
    result["execution_mode"] = "focused_complete_layer_prefix_prompt_prefix"
    result["layer_count"] = layer_count
    result["source_ordinal_first"] = selected[0]["source_ordinal"]
    result["source_ordinal_last"] = selected[-1]["source_ordinal"]
    return result


def prefill_skip_intermediate_lm_head_commands(
    commands: list[dict[str, Any]],
    *,
    prompt_token_count: int,
) -> list[dict[str, Any]]:
    if prompt_token_count < 1:
        raise RuntimeError("prefill LM-head skipping requires at least one prompt token")
    final_prompt_position = prompt_token_count - 1
    selected = [
        command
        for command in commands
        if not (
            int(command["token_step"]) < final_prompt_position
            and command["operator"] == "lm_head_tile"
        )
    ]
    if not selected:
        raise RuntimeError("prefill LM-head execution schedule is empty")
    normalized = []
    for ordinal, command in enumerate(selected):
        record = dict(command)
        record["source_ordinal"] = int(command["ordinal"])
        record["ordinal"] = ordinal
        normalized.append(record)
    skipped = len(commands) - len(normalized)
    present_intermediate_positions = {
        int(command["token_step"])
        for command in commands
        if int(command["token_step"]) < final_prompt_position
    }
    expected_skipped = len(present_intermediate_positions) * LM_HEAD_TILE_COUNT
    if skipped != expected_skipped:
        raise RuntimeError("prefill LM-head skip count differs from the frozen schedule")
    for position in sorted(present_intermediate_positions):
        position_commands = [
            command
            for command in normalized
            if int(command["token_step"]) == position
        ]
        if (
            not position_commands
            or position_commands[-1]["operator"] != "final_rmsnorm"
            or any(command["operator"] == "lm_head_tile" for command in position_commands)
        ):
            raise RuntimeError("prefill LM-head skip boundary differs")
    final_prompt_commands = [
        command
        for command in normalized
        if int(command["token_step"]) == final_prompt_position
    ]
    if final_prompt_commands and (
        final_prompt_commands[-1]["operator"] != "lm_head_tile"
        or sum(command["operator"] == "lm_head_tile" for command in final_prompt_commands)
        != LM_HEAD_TILE_COUNT
    ):
        raise RuntimeError("final prompt LM-head schedule is incomplete")
    return normalized


def verify_complete_layer0_prompt_prefix(
    commands: list[dict[str, Any]],
    *,
    prompt_tokens: list[int],
    through_position: int,
    runtime_output: Path,
) -> dict[str, Any]:
    result = verify_complete_layer_prefix_prompt_prefix(
        commands,
        prompt_tokens=prompt_tokens,
        through_position=through_position,
        layer_count=1,
        runtime_output=runtime_output,
    )
    result["execution_mode"] = "focused_complete_layer0_prompt_prefix"
    return result


def verify_position_zero_through_argmax(
    commands: list[dict[str, Any]],
    *,
    prompt_tokens: list[int],
    runtime_output: Path,
    journal_records: list[dict[str, Any]] | None = None,
    required_final_operator: str = "lm_head_tile",
) -> dict[str, Any]:
    if not prompt_tokens:
        raise RuntimeError("position-0 reference requires at least one prompt token")
    return verify_positions_through_argmax(
        commands,
        prompt_tokens=prompt_tokens,
        through_position=0,
        runtime_output=runtime_output,
        journal_records=journal_records,
        required_final_operator=required_final_operator,
    )


def summarize_attention_value_to_o_projection(
    commands: list[dict[str, Any]],
    position_reference: dict[str, Any],
) -> dict[str, Any]:
    position_zero = [
        command for command in commands if int(command["token_step"]) == 0
    ]
    o_command = next(
        (
            command
            for command in position_zero
            if int(command["layer_id"]) == 0 and command["operator"] == "o_proj"
        ),
        None,
    )
    if o_command is None:
        raise RuntimeError("position-0 attention-to-O command stream is incomplete")
    o_ordinal = int(o_command["ordinal"])
    attention_commands = [
        command
        for command in position_zero
        if int(command["layer_id"]) == 0
        and command["operator"] == "attention_value"
        and int(command["ordinal"]) < o_ordinal
    ]
    expected_addresses = [
        int(o_command["src0_addr"]) + 64 * head
        for head in range(len(attention_commands))
    ]
    actual_addresses = [int(command["dst_addr"]) for command in attention_commands]
    publication_contiguous = (
        len(attention_commands) == 14
        and actual_addresses == expected_addresses
        and int(o_command["k"]) == 896
    )

    commands_compared = int(position_reference.get("commands_compared", 0))
    mismatch_ordinal = position_reference.get("first_mismatch_ordinal")
    failed_before_boundary = (
        position_reference.get("status") == "FAIL"
        and mismatch_ordinal is not None
        and int(mismatch_ordinal) <= o_ordinal
    )
    reached = commands_compared > o_ordinal and not failed_before_boundary
    totals = position_reference.get("operator_totals", {})
    attention_total = totals.get("attention_value", {})
    o_total = totals.get("o_proj", {})
    counts_cover_boundary = (
        int(attention_total.get("commands", 0)) >= len(attention_commands)
        and int(attention_total.get("bytes", 0)) >= len(attention_commands) * 64
        and int(o_total.get("commands", 0)) >= 1
        and int(o_total.get("bytes", 0)) >= int(o_command["k"])
    )
    last_verified = position_reference.get("last_verified") or {}
    operator_last_verified = position_reference.get("operator_last_verified") or {}
    layer_operator_last_verified = (
        position_reference.get("layer_operator_last_verified") or {}
    )
    o_verified = layer_operator_last_verified.get("0.o_proj") or operator_last_verified.get("o_proj") or (
        last_verified
        if (
            int(last_verified.get("command_ordinal", -1)) == o_ordinal
            and last_verified.get("operator") == "o_proj"
        )
        else {}
    )
    o_hashes_match = (
        int(o_verified.get("command_ordinal", -1)) == o_ordinal
        and o_verified.get("operator") == "o_proj"
        and bool(o_verified.get("expected_sha256"))
        and o_verified.get("actual_sha256") == o_verified.get("expected_sha256")
    )
    passed = reached and publication_contiguous and counts_cover_boundary and o_hashes_match
    boundary_failed = failed_before_boundary or (reached and not passed)
    next_unverified_ordinal = position_reference.get("next_command_ordinal")
    next_unverified_operator = position_reference.get("next_operator")
    if position_reference.get("status") == "FAIL":
        mismatch_ordinal = position_reference.get("first_mismatch_ordinal")
        if mismatch_ordinal is not None:
            next_unverified_ordinal = int(mismatch_ordinal)
            next_unverified_operator = next(
                (
                    command["operator"]
                    for command in position_zero
                    if int(command["ordinal"]) == int(mismatch_ordinal)
                ),
                None,
            )
    return {
        "status": "PASS_BIT_EXACT" if passed else "FAIL" if boundary_failed else "NOT_REACHED",
        "first_attention_value_ordinal": (
            None if not attention_commands else int(attention_commands[0]["ordinal"])
        ),
        "last_attention_value_ordinal": (
            None if not attention_commands else int(attention_commands[-1]["ordinal"])
        ),
        "o_proj_ordinal": o_ordinal,
        "commands_compared_through_boundary": (
            o_ordinal + 1 if reached else commands_compared
        ),
        "commands_compared_total": commands_compared,
        "attention_value_commands": len(attention_commands),
        "attention_value_bytes": len(attention_commands) * 64,
        "o_proj_commands": 1,
        "o_proj_bytes": int(o_command["k"]),
        "aggregate_attention_value_commands": int(attention_total.get("commands", 0)),
        "aggregate_attention_value_bytes": int(attention_total.get("bytes", 0)),
        "aggregate_o_proj_commands": int(o_total.get("commands", 0)),
        "aggregate_o_proj_bytes": int(o_total.get("bytes", 0)),
        "publication_base": int(o_command["src0_addr"]),
        "publication_addresses_match_o_proj_input": publication_contiguous,
        "o_proj_expected_sha256": o_verified.get("expected_sha256"),
        "o_proj_actual_sha256": o_verified.get("actual_sha256"),
        "next_unverified_ordinal": next_unverified_ordinal,
        "next_unsupported_operator": next_unverified_operator,
        "actual_user_content_position_covered": False,
    }


def verify_position_one_through_argmax(
    commands: list[dict[str, Any]],
    *,
    prompt_tokens: list[int],
    runtime_output: Path,
    required_final_operator: str = "lm_head_tile",
) -> dict[str, Any]:
    if len(prompt_tokens) < 2:
        raise RuntimeError("position-1 reference requires at least two prompt tokens")
    return verify_positions_through_argmax(
        commands,
        prompt_tokens=prompt_tokens,
        through_position=1,
        runtime_output=runtime_output,
        required_final_operator=required_final_operator,
    )


def verify_position_two_through_argmax(
    commands: list[dict[str, Any]],
    *,
    prompt_tokens: list[int],
    runtime_output: Path,
    required_final_operator: str = "lm_head_tile",
) -> dict[str, Any]:
    if len(prompt_tokens) < 3:
        raise RuntimeError("position-2 reference requires at least three prompt tokens")
    return verify_positions_through_argmax(
        commands,
        prompt_tokens=prompt_tokens,
        through_position=2,
        runtime_output=runtime_output,
        required_final_operator=required_final_operator,
    )


def verify_position_three_through_argmax(
    commands: list[dict[str, Any]],
    *,
    prompt_tokens: list[int],
    runtime_output: Path,
    required_final_operator: str = "lm_head_tile",
) -> dict[str, Any]:
    if len(prompt_tokens) < 4:
        raise RuntimeError("position-3 reference requires at least four prompt tokens")
    return verify_positions_through_argmax(
        commands,
        prompt_tokens=prompt_tokens,
        through_position=3,
        runtime_output=runtime_output,
        required_final_operator=required_final_operator,
    )


def ensure_fresh_output(output: Path, resume: bool) -> None:
    runtime_output = output / "rtl"
    progress_names = ("progress.journal", "commands.jsonl", "progress.json", "summary.json", "first_failure.json")
    existing = [name for name in progress_names if (runtime_output / name).exists()]
    if existing and not resume:
        raise RuntimeError(
            "runtime output already contains progress artifacts; choose a fresh --output or use --resume"
        )


def verify_diagnostic_instruct_package_binding(
    package_path: Path,
    package: dict[str, Any],
) -> None:
    if package_path.stat().st_size != int(package["bytes"]):
        raise RuntimeError("prepared diagnostic Instruct package byte count changed")
    if sha256_file(package_path) != package["sha256"]:
        raise RuntimeError("prepared diagnostic Instruct package SHA-256 changed")
    with package_path.open("rb") as handle:
        header_raw = handle.read(PACKAGE_V2_HEADER_BYTES)
    if len(header_raw) != PACKAGE_V2_HEADER_BYTES:
        raise RuntimeError("prepared diagnostic Instruct package header is truncated")
    unpacked = PACKAGE_V2_HEADER.unpack(header_raw)
    if unpacked[0] != PACKAGE_V2_MAGIC or unpacked[1] != 2:
        raise RuntimeError("prepared diagnostic Instruct package format changed")
    image_digest, model_digest, tokenizer_digest = unpacked[13:16]
    if image_digest.hex() != DIAGNOSTIC_INSTRUCT_IMAGE_SHA256:
        raise RuntimeError("prepared diagnostic Instruct package image identity changed")
    if model_digest.hex() != DIAGNOSTIC_INSTRUCT_SOURCE_SHA256["model.safetensors"]:
        raise RuntimeError("prepared diagnostic Instruct package model identity changed")
    expected_tokenizer_digest = sha256_bytes(
        diagnostic_instruct_tokenizer_identity_record()
    )
    if tokenizer_digest.hex() != expected_tokenizer_digest:
        raise RuntimeError("prepared diagnostic Instruct package tokenizer identity changed")
    if package.get("tokenizer_sha256") != expected_tokenizer_digest:
        raise RuntimeError("prepared diagnostic Instruct package metadata changed")


def build_diagnostic_instruct_provenance(
    *,
    args: argparse.Namespace,
    system_prompt: str,
    tokenizer: Any,
    tokenization: dict[str, Any],
    authenticated: dict[str, Any],
    schedule_inputs: dict[str, Any],
    package: dict[str, Any],
    product_preflight_path: Path,
    product_preflight: dict[str, Any],
    ds32_applicability_path: Path,
    ds32_applicability: dict[str, Any],
) -> dict[str, Any]:
    host_source = Path(__file__).resolve()
    host_source_record = {
        "artifact": _artifact_label(host_source),
        "bytes": host_source.stat().st_size,
        "sha256": sha256_file(host_source),
    }
    package_binding = {
        "artifact": "runtime_package.bin",
        "bytes": int(package["bytes"]),
        "sha256": package["sha256"],
        "format": package["format"],
        "version": int(package["version"]),
    }
    bindings = {
        "model": authenticated["source_files"]["model.safetensors"],
        "model_config": authenticated["source_files"]["config.json"],
        "tokenizer": authenticated["source_files"]["tokenizer.json"],
        "tokenizer_config": authenticated["source_files"][
            "tokenizer_config.json"
        ],
        "chat_template": {
            "sha256": authenticated["chat_template_sha256"],
            "encoding": "UTF-8",
            "applied": True,
            "add_generation_prompt": True,
        },
        "image": authenticated["image"],
        "image_contract": authenticated["image_contract"],
        "validation_report": authenticated["validation_report"],
        "source": {
            "host": host_source_record,
            "stage1_gap_report": authenticated["gap_report"],
            "schedule_source": schedule_inputs["schedule_source"],
            "accepted_schedule": schedule_inputs["accepted_schedule"],
            "schedule_validation": schedule_inputs["schedule_validation"],
        },
        "package": package_binding,
    }
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "diagnostic_instruct_ace2rt2_host_package_preparation",
        "status": "PREPARED",
        "product_acceptance_eligible": False,
        "completion_claim": False,
        "diagnostic_only": True,
        "runtime_executed": False,
        "generation_executed": False,
        "decode_executed": False,
        "prompt": tokenization,
        "system_prompt_sha256": sha256_bytes(system_prompt.encode("utf-8")),
        "tokenizer": {
            "repository": authenticated["repository"],
            "revision": authenticated["revision"],
            "class": tokenizer.__class__.__name__,
            "vocab_size": int(tokenizer.vocab_size),
            "identity_record_sha256": package["tokenizer_sha256"],
            "chat_template_sha256": authenticated["chat_template_sha256"],
            "chat_template_applied": True,
            "add_generation_prompt": True,
        },
        "runtime_package": package,
        "bindings": bindings,
        "product_contract_preflight": {
            "artifact": product_preflight_path.name,
            "sha256": sha256_file(product_preflight_path),
            "status": product_preflight["status"],
            "execution": product_preflight["execution"],
        },
        "execution_policy": {
            "mode": "package_preparation_only",
            "package_commands": int(package["commands"]),
            "execution_commands": 0,
            "runtime_binary_required": False,
            "runtime_executed": False,
        },
        "accelerator_boundary": {
            "status": "PREPARED",
            "prompt_token_count": tokenization["chat_template_token_count"],
            "package_contains_complete_prompt_and_bounded_generation_schedule": True,
            "runtime_executed": False,
            "rtl_output_created": False,
            "w4a8_rtl_agreement_claim": False,
        },
        "scope": {
            "nonempty_utf8_prompt_accepted": True,
            "prompt_text_persisted": False,
            "full_prompt_prefill_schedule_present": True,
            "product_acceptance_eligible": False,
            "completion_claim": False,
            "diagnostic_only": True,
            "runtime_executed": False,
            "generation_executed": False,
            "decode_executed": False,
            "latency_claim": False,
            "fpga_or_u280_action": False,
            "stage_advancement": False,
        },
        "dynamic_scale32_prompt_applicability": {
            "artifact": ds32_applicability_path.name,
            "sha256": sha256_file(ds32_applicability_path),
            "status": ds32_applicability["status"],
            "scope": ds32_applicability["scope"],
            "earliest_mismatch": ds32_applicability["earliest_mismatch"],
            "claim": "host-side package preparation only; no runtime or RTL agreement",
        },
    }


def build_diagnostic_instruct_runtime_binding(
    authenticated_binding: dict[str, Any],
) -> dict[str, Any]:
    provenance = authenticated_binding["provenance"]
    accepted_files = authenticated_binding["accepted_files"]
    package_bindings = provenance["bindings"]
    package_source = package_bindings["source"]
    runtime_package = provenance["runtime_package"]
    rtl_binding = runtime_package["rtl_binding"]
    runtime_source_binding = rtl_binding["runtime_simulation_source_binding"]
    host_source = Path(__file__).resolve()
    binding_host_source = {
        "artifact": _artifact_label(host_source),
        "bytes": host_source.stat().st_size,
        "sha256": sha256_file(host_source),
    }
    exact_future_bindings = {
        "accepted_package_sha256": DIAGNOSTIC_INSTRUCT_ACCEPTED_PACKAGE_SHA256,
        "accepted_prepare_provenance_sha256": (
            DIAGNOSTIC_INSTRUCT_ACCEPTED_PROVENANCE_SHA256
        ),
        "accepted_prepare_host_source_sha256": (
            DIAGNOSTIC_INSTRUCT_ACCEPTED_PREPARE_HOST_SHA256
        ),
        "binding_host_source_sha256": binding_host_source["sha256"],
        "model_sha256": DIAGNOSTIC_INSTRUCT_SOURCE_SHA256["model.safetensors"],
        "tokenizer_sha256": DIAGNOSTIC_INSTRUCT_SOURCE_SHA256["tokenizer.json"],
        "tokenizer_config_sha256": (
            DIAGNOSTIC_INSTRUCT_SOURCE_SHA256["tokenizer_config.json"]
        ),
        "chat_template_sha256": DIAGNOSTIC_INSTRUCT_CHAT_TEMPLATE_SHA256,
        "w4a8_image_sha256": DIAGNOSTIC_INSTRUCT_IMAGE_SHA256,
        "image_contract_sha256": DIAGNOSTIC_INSTRUCT_IMAGE_CONTRACT_SHA256,
        "validation_report_sha256": (
            DIAGNOSTIC_INSTRUCT_VALIDATION_REPORT_SHA256
        ),
        "accepted_schedule_sha256": package_source["accepted_schedule"]["sha256"],
        "schedule_source_sha256": package_source["schedule_source"]["sha256"],
        "constraint_tree_sha256": rtl_binding["constraint_tree_sha256"],
        "rtl_tree_sha256": rtl_binding["rtl_tree_sha256"],
        "runtime_simulation_source_tree_sha256": runtime_source_binding["sha256"],
    }
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "diagnostic_instruct_runtime_binding",
        "status": "BOUND_DIAGNOSTIC_ONLY",
        "diagnostic_only": True,
        "product_acceptance_eligible": False,
        "completion_claim": False,
        "runtime_invocation_authorized": False,
        "runtime_invocation_constructed": False,
        "runtime_binary_accessed": False,
        "runtime_executed": False,
        "generation_executed": False,
        "decode_executed": False,
        "accepted_prepare": {
            "root": DIAGNOSTIC_INSTRUCT_ACCEPTED_PREPARE_RELATIVE.as_posix(),
            "fresh_l2_acceptance": {
                "mission_id": "implement-diagnostic-instruct-prepare-v1",
                "reviewer_role": "fresh_independent_l2",
                "status": "accepted",
                "execution_authority": False,
            },
            "package": accepted_files["package"],
            "provenance": accepted_files["provenance"],
            "product_contract_preflight": accepted_files["preflight"],
            "ds32_prompt_applicability": accepted_files["ds32"],
        },
        "bindings": {
            "model": package_bindings["model"],
            "model_config": package_bindings["model_config"],
            "tokenizer": package_bindings["tokenizer"],
            "tokenizer_config": package_bindings["tokenizer_config"],
            "chat_template": package_bindings["chat_template"],
            "w4a8_image": package_bindings["image"],
            "image_contract": package_bindings["image_contract"],
            "validation_report": package_bindings["validation_report"],
            "accepted_prepare_host_source": package_source["host"],
            "binding_host_source": binding_host_source,
            "package": package_bindings["package"],
            "metadata_availability": authenticated_binding["authenticated"][
                "metadata_availability"
            ],
        },
        "execution_policy": {
            "mode": "diagnostic_binding_only",
            "runtime_binary_required": False,
            "runtime_invocation_authorized": False,
            "runtime_invocation_constructed": False,
            "runtime_executed": False,
        },
        "future_execution_authority": {
            "required": True,
            "status": "NOT_BOUND",
            "authority_present": False,
            "runtime_invocation_authorized": False,
            "requirement": DIAGNOSTIC_INSTRUCT_FUTURE_AUTHORITY_REQUIREMENT,
            "must_bind_exact": exact_future_bindings,
            "additional_required_exact_bindings": [
                "external_authorization_id",
                "external_authorization_action",
                "runtime_binary_sha256",
                "exact_invocation_argv_sha256",
                "live_source_constraint_execution_tree_hashes",
            ],
        },
        "claims": {
            "output_readability": False,
            "rtl_reference_agreement": False,
            "latency": False,
            "product_completion": False,
            "stage_closure": False,
        },
    }


def run_diagnostic_instruct_runtime_binding(
    args: argparse.Namespace,
) -> int:
    authenticated_binding = authenticate_diagnostic_instruct_runtime_binding()
    output = args.output.resolve()
    accepted_prepare_root = authenticated_binding["paths"]["root"].resolve()
    if output == accepted_prepare_root or accepted_prepare_root in output.parents:
        raise RuntimeError(
            "diagnostic Instruct runtime binding output must not modify the accepted prepare package"
        )
    binding_path = output / DIAGNOSTIC_INSTRUCT_RUNTIME_BINDING_FILENAME
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            "diagnostic Instruct runtime binding requires a fresh empty --output"
        )
    output.mkdir(parents=True, exist_ok=True)
    binding = build_diagnostic_instruct_runtime_binding(authenticated_binding)
    write_atomic(binding_path, canonical_bytes(binding))
    print(
        "ACE2_CHAT_DIAGNOSTIC_RUNTIME_BINDING_PASS "
        "diagnostic_only=true product_acceptance_eligible=false "
        "completion_claim=false runtime_invocation_authorized=false "
        "runtime_executed=false separate_future_exact_execution_authority=required "
        f"package_sha256={DIAGNOSTIC_INSTRUCT_ACCEPTED_PACKAGE_SHA256}"
    )
    return 0


def run_request(
    args: argparse.Namespace,
    messages: list[dict[str, str]],
    output: Path,
) -> int:
    stage1_product_mode = not any(
        (
            args.prepare_only,
            args.diagnostic_instruct_prepare,
            args.stop_after,
            args.resume,
            args.diagnostic_allow_base_model_chat_mismatch,
            args.prefill_skip_intermediate_lm_head,
            args.legacy_qkv,
        )
    )
    if stage1_product_mode:
        if (
            len(messages) != 2
            or messages[0] != {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
            or messages[1].get("role") != "user"
        ):
            raise RuntimeError(
                "Stage-1 product mode accepts one user prompt with the canonical "
                "Qwen chat-template system message"
            )
        result = stage1_product.run_chat(
            messages[1]["content"],
            output,
            max_new_tokens=args.max_new_tokens,
        )
        if result["readability"]["accepted"]:
            print("Assistant: " + result["decoded_text"])
            print(
                "ACE2_STAGE1_CHAT_RESULT "
                f"status={result['status']} tokens={len(result['generated_token_ids'])} "
                f"compile_seconds={result['latency']['compile_wall_seconds']:.6f} "
                f"model_seconds={result['latency']['model_wall_seconds']:.6f} "
                f"simulation_seconds={result['latency']['simulation_wall_seconds']:.6f} "
                f"total_seconds={result['latency']['total_wall_seconds']:.6f}"
            )
            return 0
        print(
            "ACE2_STAGE1_CHAT_REJECTED "
            f"status={result['status']} root_cause={output.resolve() / 'root_cause.json'}",
            file=sys.stderr,
        )
        return 5
    profile = prepare_profile(args.diagnostic_instruct_prepare)
    diagnostic_instruct: dict[str, Any] | None = None
    schedule_inputs: dict[str, Any] | None = None
    if args.diagnostic_instruct_prepare:
        diagnostic_instruct = authenticate_diagnostic_instruct_prepare()
        (
            diagnostic_schedule,
            schedule_inputs,
            embedding_offset,
            embedding_shape,
        ) = load_diagnostic_instruct_schedule_inputs(diagnostic_instruct)
        diagnostic_instruct.update(
            {
                "schedule": diagnostic_schedule,
                "schedule_inputs": schedule_inputs,
                "embedding_offset": embedding_offset,
                "embedding_shape": embedding_shape,
            }
        )
    tokenizer = load_tokenizer(profile["snapshot"])
    if diagnostic_instruct is not None:
        verify_loaded_diagnostic_instruct_tokenizer(
            tokenizer,
            diagnostic_instruct["chat_template"],
        )
    tokenization = tokenize_messages(
        tokenizer,
        messages,
        chat_template=(
            diagnostic_instruct["chat_template"]
            if diagnostic_instruct is not None
            else None
        ),
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ensure_fresh_output(output, args.resume)
    diagnostic_reason = (
        "prepare_only"
        if args.prepare_only
        else "bounded_stop_after"
        if args.stop_after > 0
        else "explicit_diagnostic_only_override"
        if args.diagnostic_allow_base_model_chat_mismatch
        else None
    )
    if diagnostic_instruct is not None:
        if schedule_inputs is None:
            raise RuntimeError("diagnostic Instruct schedule inputs are missing")
        product_preflight = diagnostic_instruct_prepare_preflight(
            diagnostic_instruct,
            schedule_inputs,
        )
        product_preflight["generated_at_utc"] = utc_now()
    else:
        if not accepted_runtime.MODEL.is_file():
            raise RuntimeError("missing pinned model safetensors")
        product_preflight_start = time.perf_counter()
        product_preflight = pinned_chat_product_preflight(
            observed_model_sha256=sha256_file(accepted_runtime.MODEL),
            diagnostic_reason=diagnostic_reason,
        )
        product_preflight["generated_at_utc"] = utc_now()
        product_preflight["measurement"] = {
            "wall_seconds": time.perf_counter() - product_preflight_start,
            "scope": (
                "pinned model SHA-256 plus authenticated Base-completion feasibility "
                "audit validation"
            ),
            "performance_claim": "measured host preflight wall time only",
        }
    product_preflight_path = output / "product_contract_preflight.json"
    write_atomic(product_preflight_path, canonical_bytes(product_preflight))
    package_path = output / "runtime_package.bin"
    package = build_prompt_package(
        package_path,
        tokenization["chat_template_token_ids"],
        args.max_new_tokens,
        diagnostic_instruct=diagnostic_instruct,
        qkv_schedule_mode=(
            QKV_SCHEDULE_LEGACY if args.legacy_qkv else QKV_SCHEDULE_FUSED
        ),
    )
    ds32_applicability = package.pop("_ds32_applicability")
    first_reference_commands = package.pop("_first_reference_commands")
    position0_reference_commands = package.pop("_position0_reference_commands")
    package.pop("_position1_reference_commands")
    package.pop("_position2_reference_commands")
    package.pop("_position3_reference_commands")
    all_reference_commands = package.pop("_all_reference_commands")
    prompt_token_count = int(tokenization["chat_template_token_count"])
    execution_command_count = (
        0 if diagnostic_instruct is not None else int(package["commands"])
    )
    if args.prefill_skip_intermediate_lm_head and diagnostic_instruct is None:
        execution_command_count -= (prompt_token_count - 1) * LM_HEAD_TILE_COUNT
        position0_reference_commands = prefill_skip_intermediate_lm_head_commands(
            position0_reference_commands,
            prompt_token_count=prompt_token_count,
        )
        all_reference_commands = prefill_skip_intermediate_lm_head_commands(
            all_reference_commands,
            prompt_token_count=prompt_token_count,
        )
    if diagnostic_instruct is None and args.stop_after > execution_command_count:
        raise RuntimeError("--stop-after exceeds the selected RTL execution command count")
    ds32_applicability_path = output / "ds32_prompt_applicability.json"
    write_atomic(ds32_applicability_path, canonical_bytes(ds32_applicability))
    provenance_path = output / "provenance.json"

    if diagnostic_instruct is not None:
        if schedule_inputs is None:
            raise RuntimeError("diagnostic Instruct schedule inputs are missing")
        verify_diagnostic_instruct_package_binding(package_path, package)
        provenance = build_diagnostic_instruct_provenance(
            args=args,
            system_prompt=messages[0]["content"],
            tokenizer=tokenizer,
            tokenization=tokenization,
            authenticated=diagnostic_instruct,
            schedule_inputs=schedule_inputs,
            package=package,
            product_preflight_path=product_preflight_path,
            product_preflight=product_preflight,
            ds32_applicability_path=ds32_applicability_path,
            ds32_applicability=ds32_applicability,
        )
    else:
        provenance = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "full_prompt_ace2rt2_rtl_schedule_with_authenticated_rope",
        "prompt": tokenization,
        "system_prompt_sha256": sha256_bytes(
            messages[0]["content"].encode("utf-8")
        ),
        "tokenizer": {
            "repository": "Qwen/Qwen2.5-0.5B",
            "revision": REVISION,
            "class": tokenizer.__class__.__name__,
            "vocab_size": int(tokenizer.vocab_size),
        },
        "runtime_package": package,
        "product_contract_preflight": {
            "artifact": product_preflight_path.name,
            "sha256": sha256_file(product_preflight_path),
            "status": product_preflight["status"],
            "execution": product_preflight["execution"],
        },
        "execution_policy": {
            "mode": (
                "prefill_skip_intermediate_lm_head"
                if args.prefill_skip_intermediate_lm_head
                else "full_package_schedule"
            ),
            "package_commands": int(package["commands"]),
            "execution_commands": execution_command_count,
            "skipped_intermediate_lm_head_tiles": (
                int(package["commands"]) - execution_command_count
            ),
            "final_rmsnorm_executed_at_all_positions": True,
            "final_prompt_and_generated_lm_head_executed": True,
        },
        "accelerator_boundary": {
            "status": "PREPARED",
            "prompt_token_count": tokenization["chat_template_token_count"],
            "embedding_source": "pinned raw BF16 tied embedding table per input position",
            "rtl_command_schedule": "complete ACE2RT2 prompt and bounded generation schedule",
            "rope_source": "SHA-256-bound canonical signed-Q1.15 records carried in ACE2RT2",
        },
        "scope": {
            "arbitrary_utf8_prompt_accepted": True,
            "full_prompt_prefill_schedule_present": True,
            "quantized_reference_boundary": (
                "every selected RTL command destination and completion field across "
                "all prompt and generated positions reached; intermediate prompt LM-head tiles are "
                "excluded only when the recorded prefill optimization is selected"
            ),
            "completion_claim": False,
        },
        "dynamic_scale32_prompt_applicability": {
            "artifact": ds32_applicability_path.name,
            "sha256": sha256_file(ds32_applicability_path),
            "status": ds32_applicability["status"],
            "scope": ds32_applicability["scope"],
            "positions_checked": ds32_applicability["scope"]["positions_checked"],
            "unique_token_ids_checked": ds32_applicability["scope"][
                "unique_token_ids_checked"
            ],
            "earliest_mismatch": ds32_applicability["earliest_mismatch"],
            "claim": (
                "authenticated per-position host plans execute in live RTL through "
                "layer-0 input RMSNorm and Q/K/V; the bounded position-0 "
                "attention-value publication into O projection is separately "
                "reference-checked, while actual user-content positions remain open"
            ),
        },
        "status": "PREPARED",
        }
    write_atomic(provenance_path, canonical_bytes(provenance))

    if args.prepare_only:
        if diagnostic_instruct is not None:
            print(
                "ACE2_CHAT_PREPARE_PASS "
                "mode=diagnostic_instruct "
                f"prompt_tokens={tokenization['user_token_count']} "
                f"chat_tokens={tokenization['chat_template_token_count']} "
                f"commands={package['commands']} "
                f"rope_positions={package['rope_position_count']} "
                "execution_commands=0 "
                f"ds32_applicability={ds32_applicability['status']} "
                f"product_contract={product_preflight['status']} "
                "product_acceptance_eligible=false completion_claim=false "
                "diagnostic_only=true runtime_executed=false "
                "full_prompt_schedule=true prompt_text_persisted=false"
            )
            return 0
        print(
            "ACE2_CHAT_PREPARE_PASS "
            f"prompt_tokens={tokenization['user_token_count']} "
            f"chat_tokens={tokenization['chat_template_token_count']} "
            f"commands={package['commands']} rope_positions={package['rope_position_count']} "
            f"execution_commands={execution_command_count} "
            f"ds32_applicability={ds32_applicability['status']} "
            f"product_contract={product_preflight['status']} "
            "product_acceptance_eligible=false "
            "full_prompt_schedule=true prompt_text_persisted=false"
        )
        return 0

    binary = accepted_runtime.DEFAULT_BINARY.resolve()
    if not binary.is_file():
        raise RuntimeError("missing Verilator runtime binary; run `make full-qwen-runtime-build`")
    runtime_output = output / "rtl"
    command = [
        str(binary),
        "--package",
        str(package_path),
        "--image",
        str(accepted_runtime.IMAGE.resolve()),
        "--model",
        str(accepted_runtime.MODEL.resolve()),
        "--output",
        str(runtime_output),
        "--stop-after",
        str(args.stop_after),
        "--timeout-cycles",
        str(args.timeout_cycles),
        "--persistence-batch-commands",
        str(args.persistence_batch_commands),
    ]
    if args.resume:
        command.append("--resume")
    if args.prefill_skip_intermediate_lm_head:
        command.append("--prefill-skip-intermediate-lm-head")
    runtime_invocation_path = output / "runtime_invocation.json"
    prior_cumulative_wall_seconds = 0.0
    if args.resume and runtime_invocation_path.is_file():
        prior_invocation = json.loads(runtime_invocation_path.read_text(encoding="utf-8"))
        prior_cumulative_wall_seconds = float(
            prior_invocation.get(
                "cumulative_wall_seconds",
                prior_invocation.get(
                    "wall_seconds",
                    prior_invocation.get("prior_cumulative_wall_seconds", 0.0),
                ),
            )
        )
    runtime_invocation = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "cwd": str(Path.cwd().resolve()),
        "argv": command,
        "runtime_binary_sha256": sha256_file(binary),
        "runtime_package_sha256": sha256_file(package_path),
        "image_sha256": sha256_file(accepted_runtime.IMAGE.resolve()),
        "model_safetensors_sha256": sha256_file(accepted_runtime.MODEL.resolve()),
        "requested_stop_after": args.stop_after,
        "execution_policy": provenance["execution_policy"],
        "timeout_cycles_per_command": args.timeout_cycles,
        "persistence_batch_commands": args.persistence_batch_commands,
        "resume": args.resume,
        "prior_cumulative_wall_seconds": prior_cumulative_wall_seconds,
        "status": "RECORDED_BEFORE_EXECUTION",
    }
    write_atomic(runtime_invocation_path, canonical_bytes(runtime_invocation))
    provenance["runtime_invocation"] = runtime_invocation
    write_atomic(provenance_path, canonical_bytes(provenance))
    start = time.perf_counter()
    completed = subprocess.run(command, check=False)
    wall_seconds = time.perf_counter() - start
    runtime_invocation["completed_at_utc"] = utc_now()
    runtime_invocation["runtime_exit_code"] = completed.returncode
    runtime_invocation["wall_seconds"] = wall_seconds
    runtime_invocation["segment_wall_seconds"] = wall_seconds
    runtime_invocation["cumulative_wall_seconds"] = (
        prior_cumulative_wall_seconds + wall_seconds
    )
    runtime_invocation["status"] = "EXECUTED"
    write_atomic(runtime_invocation_path, canonical_bytes(runtime_invocation))
    summary_path = runtime_output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    journal_path = runtime_output / "progress.journal"
    journal_records = parse_journal_v2(journal_path) if journal_path.is_file() else []
    reference = verify_first_kv_publication(
        first_reference_commands[:7],
        first_prompt_token=tokenization["chat_template_token_ids"][0],
        runtime_output=runtime_output,
        journal_records=journal_records,
    ) if journal_records else {"status": "NOT_REACHED"}
    reference_path = output / "first_kv_reference.json"
    write_atomic(reference_path, canonical_bytes(reference))
    attention_reference = verify_first_attention_triplet(
        first_reference_commands,
        first_prompt_token=tokenization["chat_template_token_ids"][0],
        runtime_output=runtime_output,
        journal_records=journal_records,
    ) if journal_records else {"status": "NOT_REACHED"}
    attention_reference_path = output / "first_attention_reference.json"
    write_atomic(attention_reference_path, canonical_bytes(attention_reference))
    position0_required_final = position0_reference_commands[-1]["operator"]
    position0_reference = verify_position_zero_through_argmax(
        position0_reference_commands,
        prompt_tokens=tokenization["chat_template_token_ids"],
        runtime_output=runtime_output,
        journal_records=journal_records,
        required_final_operator=position0_required_final,
    ) if journal_records else {"status": "NOT_REACHED"}
    position0_reference_path = output / "position0_reference.json"
    write_atomic(position0_reference_path, canonical_bytes(position0_reference))
    attention_to_o_reference = summarize_attention_value_to_o_projection(
        position0_reference_commands,
        position0_reference,
    )
    attention_to_o_reference_path = output / "attention_value_to_o_reference.json"
    write_atomic(
        attention_to_o_reference_path,
        canonical_bytes(attention_to_o_reference),
    )
    maximum_reference_position = max(
        int(command["token_step"]) for command in all_reference_commands
    )
    if len(journal_records) > len(all_reference_commands):
        all_positions_reference = {
            "status": "FAIL",
            "position": maximum_reference_position,
            "commands_expected": len(all_reference_commands),
            "commands_completed": len(journal_records),
            "first_mismatch_ordinal": len(all_reference_commands),
            "first_mismatch": {"category": "journal_exceeds_selected_schedule"},
        }
    elif journal_records:
        reference_through_position = int(
            all_reference_commands[len(journal_records) - 1]["token_step"]
        )
        target_position_commands = [
            command
            for command in all_reference_commands
            if int(command["token_step"]) == reference_through_position
        ]
        all_positions_reference = verify_positions_through_argmax(
            all_reference_commands,
            prompt_tokens=tokenization["chat_template_token_ids"],
            through_position=reference_through_position,
            runtime_output=runtime_output,
            journal_records=journal_records,
            max_new_tokens=args.max_new_tokens,
            required_final_operator=str(target_position_commands[-1]["operator"]),
        )
        all_positions_reference["maximum_scheduled_position"] = maximum_reference_position
        all_positions_reference["completed_journal_records"] = len(journal_records)
    else:
        all_positions_reference = {
            "status": "NOT_REACHED",
            "position": None,
            "maximum_scheduled_position": maximum_reference_position,
            "commands_expected": len(all_reference_commands),
            "commands_completed": 0,
        }
    all_positions_reference_path = output / "all_positions_reference.json"
    write_atomic(
        all_positions_reference_path,
        canonical_bytes(all_positions_reference),
    )
    token_ids = [int(token) for token in summary.get("generated_token_ids", [])]
    decoded = decode_generated(tokenizer, token_ids)
    provenance["accelerator_boundary"] = {
        **provenance["accelerator_boundary"],
        "status": summary.get("status", "NO_SUMMARY"),
        "commands_completed": int(summary.get("commands_completed", 0)),
        "segment_simulator_cycles": int(
            summary.get("segment_simulator_cycles", summary.get("simulator_cycles", 0))
        ),
        "cumulative_simulator_cycles": int(
            summary.get("cumulative_simulator_cycles", summary.get("simulator_cycles", 0))
        ),
        "generated_token_ids": token_ids,
        "runtime_binary_sha256": sha256_file(binary),
    }
    provenance["measurement"] = {
        "wall_seconds": wall_seconds,
        "segment_wall_seconds": wall_seconds,
        "prior_cumulative_wall_seconds": prior_cumulative_wall_seconds,
        "cumulative_wall_seconds": prior_cumulative_wall_seconds + wall_seconds,
        "performance_claim": "measured Verilator wall time only; no interactive-speed claim",
    }
    provenance["decode"] = {
        "generated_token_count": len(token_ids),
        "nonterminating_token_count": sum(
            token not in TERMINATION_TOKEN_IDS for token in token_ids
        ),
        "text_utf8_bytes": len(decoded.encode("utf-8")),
        "readable_text_observed": bool(decoded.strip()),
    }
    provenance["runtime_exit_code"] = completed.returncode
    provenance["quantized_reference"] = {
        "first_kv": reference,
        "first_attention_triplet": attention_reference,
        "attention_value_to_o_projection": attention_to_o_reference,
        "position0_through_argmax": position0_reference,
        "all_executed_positions_through_argmax": all_positions_reference,
    }
    reference_failed = (
        reference.get("status") == "FAIL"
        or attention_reference.get("status") == "FAIL"
        or attention_to_o_reference.get("status") == "FAIL"
        or position0_reference.get("status") == "FAIL"
        or all_positions_reference.get("status") == "FAIL"
    )
    reference_statuses = {
        "first_kv": reference.get("status", "NOT_REACHED"),
        "first_attention_triplet": attention_reference.get("status", "NOT_REACHED"),
        "attention_value_to_o_projection": attention_to_o_reference.get(
            "status", "NOT_REACHED"
        ),
        "position0_through_argmax": position0_reference.get("status", "NOT_REACHED"),
        "all_executed_positions_through_argmax": all_positions_reference.get(
            "status", "NOT_REACHED"
        ),
    }
    reference_complete = all(
        str(status).startswith("PASS") for status in reference_statuses.values()
    )
    nonterminating_token_count = int(
        provenance["decode"]["nonterminating_token_count"]
    )
    product_acceptance_candidate = bool(
        product_preflight["execution"]["product_acceptance_eligible"]
        and completed.returncode == 0
        and reference_complete
        and nonterminating_token_count >= 2
        and decoded.strip()
    )
    provenance["decode"]["meets_readable_multi_token_threshold"] = bool(
        nonterminating_token_count >= 2 and decoded.strip()
    )
    provenance["product_acceptance"] = {
        "candidate_for_independent_review": product_acceptance_candidate,
        "preflight_eligible": product_preflight["execution"][
            "product_acceptance_eligible"
        ],
        "runtime_exit_zero": completed.returncode == 0,
        "quantized_reference_complete": reference_complete,
        "reference_statuses": reference_statuses,
        "minimum_visible_nonterminating_tokens": 2,
    }
    provenance["scope"]["completion_claim"] = product_acceptance_candidate
    provenance["status"] = (
        "REFERENCE_MISMATCH"
        if reference_failed
        else "PRODUCT_ACCEPTANCE_CANDIDATE"
        if product_acceptance_candidate
        else "PRODUCT_OUTPUT_REJECTED"
        if product_preflight["execution"]["product_acceptance_eligible"]
        and completed.returncode == 0
        else "RTL_EXECUTED"
        if completed.returncode == 0
        else "RTL_STOPPED"
    )
    write_atomic(provenance_path, canonical_bytes(provenance))

    if token_ids:
        print("Assistant: " + (decoded if decoded else "<decoded output contains no visible text>"))
    else:
        print("Assistant: <no token selected in the requested RTL command prefix>")
    print(
        "ACE2_CHAT_RTL_RESULT "
        f"status={summary.get('status', 'NO_SUMMARY')} "
        f"commands={summary.get('commands_completed', 0)} "
        f"tokens={len(token_ids)} wall_seconds={wall_seconds:.6f} "
        f"full_prompt_schedule=true first_kv_reference={reference.get('status', 'NOT_REACHED')} "
        "first_attention_reference="
        f"{attention_reference.get('status', 'NOT_REACHED')} "
        "attention_to_o_reference="
        f"{attention_to_o_reference.get('status', 'NOT_REACHED')} "
        "position0_reference="
        f"{position0_reference.get('status', 'NOT_REACHED')} "
        "all_positions_reference="
        f"{all_positions_reference.get('status', 'NOT_REACHED')} "
        f"product_contract={product_preflight['status']} "
        "product_acceptance_candidate="
        f"{str(product_acceptance_candidate).lower()}"
    )
    if reference_failed:
        return 2
    if (
        product_preflight["execution"]["product_acceptance_eligible"]
        and completed.returncode == 0
        and not product_acceptance_candidate
    ):
        return 5
    return completed.returncode


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "The full-prompt ACE2RT2 legacy diagnostics remain available; "
            "plain text runs one persistent canonical Qwen2.5-0.5B-Instruct "
            "W4A8 Icarus-backed Stage-1 chat session"
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--prompt",
        help="UTF-8 user text; omitted means stdin or repeated interactive prompts",
    )
    source.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 user text file read byte-for-byte",
    )
    source.add_argument(
        "--conversation-file",
        type=Path,
        help="UTF-8 JSON messages transcript ending in a user message",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help=(
            "legacy diagnostic system text; the Stage-1 product path requires "
            "the canonical default"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
        help="Stage-1 greedy generation count, 3 or 4 (default: 4)",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="RTL command prefix length; 0 runs the complete RT2 prompt/generation schedule",
    )
    parser.add_argument("--timeout-cycles", type=int, default=100_000_000)
    parser.add_argument(
        "--persistence-batch-commands",
        type=int,
        default=DEFAULT_PERSISTENCE_BATCH_COMMANDS,
        help="durably commit runtime progress every N commands (default: 64)",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--legacy-qkv",
        action="store_true",
        help="emit the original three-descriptor Q/K/V schedule instead of opcode 0x0b",
    )
    parser.add_argument(
        "--diagnostic-instruct-prepare",
        action="store_true",
        help=(
            "prepare an authenticated pinned-Instruct ACE2RT2 package without "
            "runtime execution; valid only with --prepare-only"
        ),
    )
    parser.add_argument(
        "--prepare-diagnostic-instruct-cache",
        action="store_true",
        help=(
            "retrieve only the four checksum-pinned Instruct source files into "
            "the shared Hugging Face cache and emit a metadata-only cache manifest"
        ),
    )
    parser.add_argument(
        "--diagnostic-instruct-runtime-binding",
        action="store_true",
        help=(
            "bind only the Fresh-L2-accepted non-Hi Instruct ACE2RT2 package; "
            "diagnostic_only=true, product_acceptance_eligible=false, "
            "completion_claim=false, and no runtime invocation. A future invocation "
            "requires separate external authority binding the exact package, source, "
            "constraints, and execution-tree hashes"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--diagnostic-allow-base-model-chat-mismatch",
        action="store_true",
        help=(
            "force a full RTL run to remain diagnostic-only under the pinned-Base "
            "completion contract; never eligible for product acceptance"
        ),
    )
    parser.add_argument(
        "--prefill-skip-intermediate-lm-head",
        action="store_true",
        help=(
            "retain the frozen full package schedule but omit stateless LM-head "
            "tiles before the final prompt position"
        ),
    )
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.stop_after < 0:
        raise RuntimeError("--stop-after must be nonnegative")
    if args.persistence_batch_commands < 1:
        raise RuntimeError("--persistence-batch-commands must be at least one")
    validate_max_new_tokens(args.max_new_tokens)
    if args.diagnostic_instruct_prepare and not args.prepare_only:
        raise RuntimeError(
            "--diagnostic-instruct-prepare is valid only with --prepare-only"
        )
    if (
        args.diagnostic_instruct_runtime_binding
        and args.prepare_diagnostic_instruct_cache
    ):
        raise RuntimeError(
            "--diagnostic-instruct-runtime-binding and "
            "--prepare-diagnostic-instruct-cache are mutually exclusive"
        )
    if (
        args.diagnostic_instruct_runtime_binding
        or args.prepare_diagnostic_instruct_cache
    ):
        conflicts = []
        for active, option in (
            (args.prompt is not None, "--prompt"),
            (args.prompt_file is not None, "--prompt-file"),
            (args.conversation_file is not None, "--conversation-file"),
            (args.system_prompt != DEFAULT_SYSTEM_PROMPT, "--system-prompt"),
            (args.max_new_tokens != 4, "--max-new-tokens"),
            (args.stop_after != 0, "--stop-after"),
            (args.timeout_cycles != 100_000_000, "--timeout-cycles"),
            (
                args.persistence_batch_commands != DEFAULT_PERSISTENCE_BATCH_COMMANDS,
                "--persistence-batch-commands",
            ),
            (args.prepare_only, "--prepare-only"),
            (args.legacy_qkv, "--legacy-qkv"),
            (args.diagnostic_instruct_prepare, "--diagnostic-instruct-prepare"),
            (args.resume, "--resume"),
            (
                args.diagnostic_allow_base_model_chat_mismatch,
                "--diagnostic-allow-base-model-chat-mismatch",
            ),
            (
                args.prefill_skip_intermediate_lm_head,
                "--prefill-skip-intermediate-lm-head",
            ),
        ):
            if active:
                conflicts.append(option)
        if conflicts:
            raise RuntimeError(
                "the selected cache/binding operation is a standalone non-execution "
                "mode; incompatible option(s): " + ", ".join(conflicts)
            )
    if args.prepare_only and args.resume:
        raise RuntimeError("--prepare-only and --resume are mutually exclusive")


def run_interactive(args: argparse.Namespace) -> int:
    turn = 0
    output_root = args.output.resolve()
    while True:
        try:
            prompt = input("User: ").strip()
        except EOFError:
            return 0
        _require_non_whitespace(prompt, identity="prompt")
        turn += 1
        messages = single_turn_messages(prompt, args.system_prompt)
        result = run_request(
            args,
            messages,
            output_root / f"turn-{turn:04d}",
        )
        if result != 0:
            return result


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    validate_cli_args(args)
    if args.prepare_diagnostic_instruct_cache:
        result = prepare_diagnostic_instruct_cache(args.output)
        print(
            f"{result['status']} "
            f"retrieval_performed={str(result['cache']['retrieval_performed']).lower()} "
            "model_executed=false rtl_executed=false"
        )
        return 0
    if args.diagnostic_instruct_runtime_binding:
        return run_diagnostic_instruct_runtime_binding(args)
    if args.conversation_file is not None:
        messages = read_conversation_file(
            args.conversation_file,
            default_system_prompt=args.system_prompt,
        )
        return run_request(args, messages, args.output)
    if (
        args.prompt is not None
        or args.prompt_file is not None
        or not sys.stdin.isatty()
    ):
        prompt = read_prompt(args.prompt, args.prompt_file)
        return run_request(
            args,
            single_turn_messages(prompt, args.system_prompt),
            args.output,
        )
    return run_interactive(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_CHAT_SETUP_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
