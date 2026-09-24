#!/usr/bin/env python3
"""Freeze, execute, and verify one output-blind block32-affine W4A8 candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import probe_qwen_instruct_source_qualification_v1 as source_probe
import probe_qwen_instruct_source_relative_quality_v2 as source_relative
import qwen_instruct_option_b as source_identity
import v13_v10_transformer_v12_lm_head64_backend as quality


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "e248e307675a_block32_affine_fp32_offset_step_w4a8_v1"
MISSION_ID = "e248e307675a"
RUNNER = Path(__file__).resolve()
OFFICIAL_ROOT = ROOT / "build/candidate-e248e307675a-block32-affine-fp32-w4a8-v1"
QUALITY_DIR = OFFICIAL_ROOT / "quality-campaign-0001"
MATRIX_SOURCE = (
    ROOT
    / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v16"
    / "quality-campaign-0001/frozen_matrix.json"
)
SOURCE_RELATIVE_CONTRACT = ROOT / "design/QWEN_INSTRUCT_SOURCE_RELATIVE_QUALITY_V2.json"
SOURCE_BASELINE = ROOT / "research/probes/qwen-instruct-source-relative-quality-v2-repair1-20260807T111432Z.json"
SOURCE_REVIEW = ROOT / "research/probes/qwen-instruct-source-relative-quality-v2-reviewer-20260807T112030Z.json"
BASE_SCALES = quality.BASE_SCALES_PATH

CONSTRUCTION_MARKER = OFFICIAL_ROOT / "construction_started.json"
PREDECLARED_MANIFEST = OFFICIAL_ROOT / "predeclared_manifest.json"
CANDIDATE_A = OFFICIAL_ROOT / "candidate_a_manifest.json"
CANDIDATE_B = OFFICIAL_ROOT / "candidate_b_manifest.json"
ALIAS_WITNESS = OFFICIAL_ROOT / "alias_separation_witness.json"
ACCOUNTING = OFFICIAL_ROOT / "codec_and_byte_accounting.json"
DYNAMIC_PREFLIGHT = OFFICIAL_ROOT / "dynamic_preflight.json"
STAGE_LATENCY = OFFICIAL_ROOT / "stage_latency.json"
PREFLIGHT_READY = OFFICIAL_ROOT / "PRODUCT_PREFLIGHT_READY.json"
FROZEN_MATRIX = QUALITY_DIR / "frozen_matrix.json"
EXECUTION_MARKER = QUALITY_DIR / "execution_started.json"
RAW_OUTPUTS = QUALITY_DIR / "raw_outputs.jsonl"
REPLAY_OUTPUTS = QUALITY_DIR / "deterministic_replay.jsonl"
TOKEN_LATENCY = QUALITY_DIR / "token_and_latency_results.json"
RUBRIC = QUALITY_DIR / "source_relative_rubric.json"
RESULT = QUALITY_DIR / "RESULT.json"
QUALITY_SUMS = QUALITY_DIR / "SHA256SUMS"
REVIEWER_SUBMISSION = OFFICIAL_ROOT / "fresh_reviewer_submission.json"

BLOCK_WIDTH = 32
QUANT_MIN = 0
QUANT_MAX = 15
MAX_NEW_TOKENS = 64
CONSTRUCTION_THREADS = 1
QUALITY_THREADS = 32
ROW_CHUNK = 256
EXPECTED_LINEAR_TENSORS = 169
EXPECTED_LINEAR_WEIGHT_COUNT = 493_961_216
EXPECTED_RESPONSE_COUNT = 23
EXPECTED_CASE_COUNT = 19


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        label = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        label = str(resolved)
    return {"path": label, "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def write_json_exclusive(path: Path, value: Any) -> None:
    raw = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    temporary.replace(path)


def raw_tensor_bytes(value: Tensor) -> bytes:
    flat = value.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes(order="C")


def token_ids_sha256(values: list[int]) -> str:
    raw = b"".join(int(value).to_bytes(4, "little", signed=False) for value in values)
    return hashlib.sha256(raw).hexdigest()


def pack_nibbles(indices: Tensor) -> bytes:
    flat = indices.detach().cpu().contiguous().reshape(-1).to(torch.uint8)
    require(flat.numel() % 2 == 0, "four-bit payload element count must be even")
    packed = torch.bitwise_or(flat[0::2], torch.bitwise_left_shift(flat[1::2], 4))
    return packed.numpy().astype(np.uint8, copy=False).tobytes(order="C")


def unpack_nibbles(raw: bytes, count: int) -> Tensor:
    require(count % 2 == 0 and len(raw) == count // 2, "packed nibble geometry differs")
    packed = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())
    result = torch.empty(count, dtype=torch.uint8)
    result[0::2] = torch.bitwise_and(packed, 0x0F)
    result[1::2] = torch.bitwise_right_shift(packed, 4)
    return result


def quantize_weight_in_place(weight: Tensor, module_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(weight.ndim == 2 and weight.dtype == torch.bfloat16, f"weight format differs: {module_name}")
    rows, width = (int(value) for value in weight.shape)
    require(width % BLOCK_WIDTH == 0, f"block width does not divide tensor: {module_name}")
    blocks_per_row = width // BLOCK_WIDTH
    payload_digest = hashlib.sha256()
    offset_digest = hashlib.sha256()
    step_digest = hashlib.sha256()
    source_digest = hashlib.sha256()
    reconstruction_digest = hashlib.sha256()
    histogram = [0] * 16
    literal_bf16_sse = 0.0
    zero_span_blocks = 0
    stage_records: list[dict[str, Any]] = []

    with torch.no_grad():
        for row_start in range(0, rows, ROW_CHUNK):
            row_stop = min(rows, row_start + ROW_CHUNK)
            started = time.monotonic()
            source_bf16 = weight[row_start:row_stop].detach().cpu().contiguous()
            source_digest.update(raw_tensor_bytes(source_bf16))
            source = source_bf16.to(torch.float32)
            grouped = source.reshape(row_stop - row_start, blocks_per_row, BLOCK_WIDTH)
            offsets = grouped.amin(dim=-1)
            maxima = grouped.amax(dim=-1)
            steps = (maxima - offsets) / float(QUANT_MAX)
            nonzero = steps > 0
            denominator = torch.where(nonzero, steps, torch.ones_like(steps))
            indices = torch.round((grouped - offsets.unsqueeze(-1)) / denominator.unsqueeze(-1))
            indices = indices.clamp(QUANT_MIN, QUANT_MAX).to(torch.uint8)
            indices = torch.where(nonzero.unsqueeze(-1), indices, torch.zeros_like(indices))
            decoded = offsets.unsqueeze(-1) + indices.to(torch.float32) * steps.unsqueeze(-1)
            decoded_bf16 = decoded.reshape(row_stop - row_start, width).to(torch.bfloat16)
            weight[row_start:row_stop].copy_(decoded_bf16)

            payload_raw = pack_nibbles(indices)
            offset_raw = offsets.contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")
            step_raw = steps.contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")
            payload_digest.update(payload_raw)
            offset_digest.update(offset_raw)
            step_digest.update(step_raw)
            reconstruction_digest.update(raw_tensor_bytes(decoded_bf16))
            counts = torch.bincount(indices.reshape(-1).to(torch.int64), minlength=16).tolist()
            histogram = [left + int(right) for left, right in zip(histogram, counts, strict=True)]
            difference = source - decoded_bf16.to(torch.float32)
            literal_bf16_sse += float(torch.sum(difference * difference, dtype=torch.float64))
            zero_span_blocks += int((~nonzero).sum())
            stage_records.append(
                {
                    "row_start": row_start,
                    "row_stop": row_stop,
                    "seconds": time.monotonic() - started,
                }
            )

    weight_count = rows * width
    block_count = rows * blocks_per_row
    payload_bytes = weight_count // 2
    offset_bytes = block_count * 4
    step_bytes = block_count * 4
    require(sum(histogram) == weight_count, f"histogram closure differs: {module_name}")
    require(payload_bytes == weight_count * 4 // 8, f"payload accounting differs: {module_name}")
    return (
        {
            "module": module_name,
            "name": f"{module_name}.weight",
            "shape": [rows, width],
            "weight_count": weight_count,
            "block_width": BLOCK_WIDTH,
            "block_count": block_count,
            "payload": {
                "bits_per_weight": 4,
                "bytes": payload_bytes,
                "sha256": payload_digest.hexdigest(),
                "packing": "row-major; even flat element low nibble; odd flat element high nibble",
                "unsigned_code_range": [QUANT_MIN, QUANT_MAX],
                "histogram": histogram,
            },
            "metadata": {
                "offset_fp32": {"bytes": offset_bytes, "sha256": offset_digest.hexdigest(), "bytes_per_block": 4},
                "step_fp32": {"bytes": step_bytes, "sha256": step_digest.hexdigest(), "bytes_per_block": 4},
                "total_bytes": offset_bytes + step_bytes,
            },
            "source_bf16_sha256": source_digest.hexdigest(),
            "reconstruction_bf16_sha256": reconstruction_digest.hexdigest(),
            "literal_bf16_sse": literal_bf16_sse,
            "zero_span_block_count": zero_span_blocks,
            "rounding": "round_to_nearest_ties_to_even_then_clamp_0_15",
            "decoder": "bf16(fp32_offset + uint4_code * fp32_step)",
        },
        stage_records,
    )


def build_candidate(label: str) -> tuple[nn.Module, dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        source_identity.SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    embedding = model.model.embed_tokens.weight
    lm_head = model.lm_head.weight
    require(model.config.tie_word_embeddings is True, "source embedding/lm_head tie differs")
    require(lm_head is embedding and lm_head.data_ptr() == embedding.data_ptr(), "source alias differs")
    embedding_object = id(embedding)
    embedding_pointer = embedding.data_ptr()
    embedding_sha256 = quality.v10.v7.v3.tensor_sha256(embedding)
    detached_lm_head = lm_head.detach().clone()
    model.lm_head.weight = nn.Parameter(detached_lm_head, requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head remained aliased")

    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == EXPECTED_LINEAR_TENSORS, "linear tensor count differs")
    records: list[dict[str, Any]] = []
    per_tensor_latency: list[dict[str, Any]] = []
    totals = {"weight_count": 0, "block_count": 0, "payload_bytes": 0, "metadata_bytes": 0}
    aggregate = {name: hashlib.sha256() for name in ("source", "payload", "offset", "step", "reconstruction")}
    construction_started = time.monotonic()
    for ordinal, (name, module) in enumerate(modules, start=1):
        started = time.monotonic()
        print(f"E248E307675A_CONSTRUCT_{label.upper()} {ordinal}/{len(modules)} {name}", flush=True)
        record, chunks = quantize_weight_in_place(module.weight, name)
        records.append(record)
        per_tensor_latency.append(
            {
                "module": name,
                "ordinal": ordinal,
                "seconds": time.monotonic() - started,
                "chunks": chunks,
            }
        )
        totals["weight_count"] += int(record["weight_count"])
        totals["block_count"] += int(record["block_count"])
        totals["payload_bytes"] += int(record["payload"]["bytes"])
        totals["metadata_bytes"] += int(record["metadata"]["total_bytes"])
        for key, digest_value in (
            ("source", record["source_bf16_sha256"]),
            ("payload", record["payload"]["sha256"]),
            ("offset", record["metadata"]["offset_fp32"]["sha256"]),
            ("step", record["metadata"]["step_fp32"]["sha256"]),
            ("reconstruction", record["reconstruction_bf16_sha256"]),
        ):
            aggregate[key].update(name.encode() + b"\0" + bytes.fromhex(digest_value))

    require(totals["weight_count"] == EXPECTED_LINEAR_WEIGHT_COUNT, "whole-model linear weight count differs")
    require(totals["block_count"] == EXPECTED_LINEAR_WEIGHT_COUNT // BLOCK_WIDTH, "whole-model block count differs")
    require(totals["payload_bytes"] == EXPECTED_LINEAR_WEIGHT_COUNT // 2, "whole-model payload bytes differ")
    require(totals["metadata_bytes"] == EXPECTED_LINEAR_WEIGHT_COUNT // 4, "whole-model metadata bytes differ")
    require(id(model.model.embed_tokens.weight) == embedding_object, "embedding parameter object changed")
    require(model.model.embed_tokens.weight.data_ptr() == embedding_pointer, "embedding storage changed")
    require(quality.v10.v7.v3.tensor_sha256(embedding) == embedding_sha256, "embedding bytes changed")
    require(model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head was re-aliased")

    policy = {
        "weight_payload": {
            "bits_per_weight": 4,
            "block_width": BLOCK_WIDTH,
            "code_range": [QUANT_MIN, QUANT_MAX],
            "packing": "two unsigned codes per byte",
        },
        "weight_metadata": {
            "per_block": ["fp32 minimum offset", "fp32 affine step"],
            "bytes_per_block": 8,
            "decode": "bf16(offset + code * step)",
        },
        "activation": {
            "payload": "signed int8 excluding reserved -128",
            "policy": "audited grouped Dynamic Scale32",
            "expected_named_boundaries": 312,
            "maximum_transport_metadata_bytes_per_decoder_layer_token": 832,
        },
        "scope": {
            "quantized": "all 169 Linear weights including separated lm_head",
            "preserved": "source BF16 token embedding and non-Linear tensors",
            "lm_head_embedding_alias": "split before lm_head quantization; embedding bytes preserved",
        },
        "selection": "structural policy frozen before candidate text generation and without candidate-output inspection",
    }
    manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "source_revision": source_identity.REVISION,
        "policy": policy,
        "module_count": len(records),
        "totals": totals,
        "aggregate_stream_hashes": {name: digest.hexdigest() for name, digest in aggregate.items()},
        "embedding_bf16_sha256": embedding_sha256,
        "weight_manifest": records,
    }
    manifest["candidate_core_sha256"] = canonical_sha256(manifest)
    witness = {
        "source_lm_head_and_embedding_same_parameter_object": True,
        "source_lm_head_and_embedding_same_storage": True,
        "post_split_lm_head_and_embedding_different_storage": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
        "embedding_parameter_object_preserved": id(model.model.embed_tokens.weight) == embedding_object,
        "embedding_storage_preserved": model.model.embed_tokens.weight.data_ptr() == embedding_pointer,
        "embedding_bytes_preserved": quality.v10.v7.v3.tensor_sha256(embedding) == embedding_sha256,
    }
    latency = {
        "candidate_build": label,
        "total_seconds": time.monotonic() - construction_started,
        "per_tensor": per_tensor_latency,
    }
    return model, manifest, witness, latency


def quantizer_self_test() -> dict[str, Any]:
    source = torch.tensor(
        [
            [float(index - 16) / 32.0 for index in range(64)],
            [0.25] * 32 + [float((index % 11) - 5) / 17.0 for index in range(32)],
        ],
        dtype=torch.bfloat16,
    )
    first = source.clone()
    second = source.clone()
    first_record, _ = quantize_weight_in_place(first, "synthetic")
    second_record, _ = quantize_weight_in_place(second, "synthetic")
    require(first_record == second_record, "synthetic quantizer is not deterministic")
    require(torch.equal(first, second), "synthetic reconstructions differ")
    probe_indices = torch.arange(16, dtype=torch.uint8).repeat(4)
    packed = pack_nibbles(probe_indices)
    require(torch.equal(unpack_nibbles(packed, probe_indices.numel()), probe_indices), "nibble round trip differs")
    require(first_record["payload"]["bits_per_weight"] == 4, "synthetic payload width differs")
    require(first_record["zero_span_block_count"] == 1, "constant-block handling differs")
    return {
        "status": "PASS",
        "deterministic_two_build_record": True,
        "deterministic_two_build_reconstruction": True,
        "nibble_round_trip": True,
        "constant_block_round_trip": True,
        "payload_bits_per_weight": 4,
        "synthetic_record_sha256": canonical_sha256(first_record),
    }


def matrix_preview() -> dict[str, Any]:
    require(MATRIX_SOURCE.is_file(), "frozen matrix source is missing")
    require(sha256_file(MATRIX_SOURCE) == "829d3fc355ecba7dd01aed6a06c165b6a664cbc144591a0c40b45ae4fa27c6a1", "frozen matrix source differs")
    source = load_json(MATRIX_SOURCE)
    contract = source_relative.verify_contract()
    require(len(source["cases"]) == EXPECTED_CASE_COUNT, "matrix case count differs")
    require(int(source["response_count"]) == EXPECTED_RESPONSE_COUNT, "matrix response count differs")
    matrix = copy.deepcopy(source)
    matrix["candidate_id"] = CANDIDATE_ID
    matrix["system_message"] = contract["generation_contract"]["system_message"]
    matrix["selected_before_candidate_text_generation"] = True
    matrix["selected_after_two_constructions_and_dynamic_replay"] = False
    matrix["source_matrix"] = file_record(MATRIX_SOURCE)
    matrix["generation_contract"] = {
        "chat_template": contract["generation_contract"]["chat_template"],
        "decode": contract["generation_contract"]["decode"],
        "use_cache": True,
        "maximum_new_tokens_per_delivered_response": MAX_NEW_TOKENS,
        "termination_token_ids": list(source_identity.TERMINATION_TOKEN_IDS),
        "multi_turn_state": "append only delivered model decode; blocked action appends no assistant turn",
        "cache_scope": "fresh KV cache per response; autoregressive cache retained only within one response",
    }
    matrix.pop("matrix_content_sha256", None)
    matrix["matrix_content_sha256"] = canonical_sha256(matrix)
    return matrix


def preflight() -> dict[str, Any]:
    require(not OFFICIAL_ROOT.exists(), "fresh official namespace is not absent")
    require(not EXECUTION_MARKER.exists(), "quality execution marker already exists")
    source = source_identity.verify_source_snapshot()
    versions = source_identity.verify_versions()
    contract = source_relative.verify_contract()
    guard = source_relative.guard_self_test()
    detector = source_probe.detector_self_test()
    quantizer = quantizer_self_test()
    matrix = matrix_preview()
    require(SOURCE_BASELINE.is_file() and SOURCE_REVIEW.is_file(), "source-relative baseline/review is missing")
    baseline = load_json(SOURCE_BASELINE)
    require(baseline.get("status") == "SOURCE_QUALIFIED_RELATIVE_BASELINE", "source-relative baseline did not pass")
    require(baseline["rubric"]["source_relative_qualification_status"] == "PASS", "source-relative baseline rubric did not pass")
    return {
        "schema_version": 1,
        "status": "PASS_NON_CONSUMING_PREFLIGHT",
        "candidate_id": CANDIDATE_ID,
        "official_namespace_absent": True,
        "quality_marker_absent": True,
        "candidate_policy_frozen_before_output": True,
        "source": source,
        "versions": versions,
        "source_relative_contract": file_record(SOURCE_RELATIVE_CONTRACT),
        "source_relative_baseline": file_record(SOURCE_BASELINE),
        "source_relative_review": file_record(SOURCE_REVIEW),
        "guard_self_test": guard,
        "detector_self_test": detector,
        "quantizer_self_test": quantizer,
        "matrix": {
            "source": file_record(MATRIX_SOURCE),
            "candidate_matrix_content_sha256": matrix["matrix_content_sha256"],
            "case_count": len(matrix["cases"]),
            "response_count": matrix["response_count"],
        },
        "namespace_semantics": "run-all creates the official root exactly once; any existing root blocks all execution",
        "restoration_path": "reconstruct only from the pinned source revision and frozen quantizer; official output regeneration is forbidden",
        "marker_boundaries": {
            "construction_started": "written immediately after namespace creation and before full-model construction",
            "quality_execution_started": "written after candidate A/B equality, dynamic preflight, matrix freeze, and manifest publication; before first candidate text generation",
        },
        "claim_boundary": "non-consuming source, matrix, guard, evaluator, namespace, and synthetic codec preflight only",
    }


def predeclared_manifest(preflight_result: dict[str, Any]) -> dict[str, Any]:
    weight_count = EXPECTED_LINEAR_WEIGHT_COUNT
    payload_bytes = weight_count // 2
    metadata_bytes = weight_count // 4
    return {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "candidate_id": CANDIDATE_ID,
        "frozen_at_utc": utc_now(),
        "selection": {
            "output_blind": True,
            "structural_policy": "per-input-block-32 affine uint4 with FP32 offset and FP32 step",
            "block_width": BLOCK_WIDTH,
            "quantizer_rounding": "round-to-nearest ties-to-even",
            "candidate_output_or_hidden_golden_inspected_before_freeze": False,
        },
        "source": preflight_result["source"],
        "source_relative_contract": preflight_result["source_relative_contract"],
        "source_relative_baseline": preflight_result["source_relative_baseline"],
        "source_relative_review": preflight_result["source_relative_review"],
        "commands": {
            "non_consuming_preflight": ["./.venv/bin/python", RUNNER.relative_to(ROOT).as_posix(), "preflight"],
            "one_shot_execution": ["./.venv/bin/python", RUNNER.relative_to(ROOT).as_posix(), "run-all"],
            "read_only_verifier": ["./.venv/bin/python", RUNNER.relative_to(ROOT).as_posix(), "verify-result"],
        },
        "budgets": {
            "candidate_count": 1,
            "official_execution_count": 1,
            "deterministic_in_memory_replay_count": 1,
            "maximum_new_tokens_per_delivered_response": MAX_NEW_TOKENS,
            "seed": 0,
            "sampling": False,
        },
        "practical_accounting": {
            "linear_weight_count": weight_count,
            "four_bit_payload_bytes": payload_bytes,
            "metadata_bytes": metadata_bytes,
            "total_linear_weight_bytes": payload_bytes + metadata_bytes,
            "exact_bits_per_linear_weight": 6.0,
            "bf16_linear_weight_bytes": weight_count * 2,
            "linear_weight_byte_reduction_percent": 62.5,
            "metadata_semantics": "one FP32 offset and one FP32 step per 32 weights",
            "activation_transport_metadata_cap_bytes_per_decoder_layer_token": 832,
            "abstract_streaming_memory_boundary_bits": 128,
        },
        "chat_and_decoding": {
            "matrix_source": file_record(MATRIX_SOURCE),
            "matrix_case_count": EXPECTED_CASE_COUNT,
            "response_position_count": EXPECTED_RESPONSE_COUNT,
            "arbitrary_text": True,
            "multi_turn": True,
            "chat_template_sha256": source_identity.CHAT_TEMPLATE_SHA256,
            "decode": "deterministic greedy argmax",
            "use_cache": True,
            "termination_token_ids": list(source_identity.TERMINATION_TOKEN_IDS),
            "delivery_policy": load_json(SOURCE_RELATIVE_CONTRACT)["delivery_policy"],
        },
        "tools": {
            "runner": file_record(RUNNER),
            "source_identity": file_record(ROOT / "tools/qwen_instruct_option_b.py"),
            "source_relative_guard_and_evaluator": file_record(ROOT / "tools/probe_qwen_instruct_source_relative_quality_v2.py"),
            "corrected_response_evaluator": file_record(ROOT / "tools/probe_qwen_instruct_source_qualification_v1.py"),
            "dynamic_activation_policy": file_record(Path(quality.v10.v7.v3.__file__).resolve()),
        },
        "expected_artifacts": [
            path.relative_to(OFFICIAL_ROOT).as_posix()
            for path in (
                CONSTRUCTION_MARKER,
                PREDECLARED_MANIFEST,
                CANDIDATE_A,
                CANDIDATE_B,
                ALIAS_WITNESS,
                ACCOUNTING,
                DYNAMIC_PREFLIGHT,
                STAGE_LATENCY,
                PREFLIGHT_READY,
                FROZEN_MATRIX,
                EXECUTION_MARKER,
                RAW_OUTPUTS,
                REPLAY_OUTPUTS,
                TOKEN_LATENCY,
                RUBRIC,
                RESULT,
                QUALITY_SUMS,
                REVIEWER_SUBMISSION,
            )
        ],
        "prohibitions": {
            "official_regeneration": True,
            "candidate_repair_after_output": True,
            "hidden_harness_or_golden_output_access": True,
            "downstream_rtl_or_u280_work": True,
        },
    }


def tokenize_messages(tokenizer: Any, messages: list[dict[str, str]]) -> Tensor:
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    require(isinstance(input_ids, Tensor) and input_ids.ndim == 2 and input_ids.shape[0] == 1, "chat tokenization differs")
    return input_ids


def generate_response(model: nn.Module, tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    input_ids = tokenize_messages(tokenizer, messages)
    prompt_ids = [int(value) for value in input_ids[0].tolist()]
    attention_mask = torch.ones_like(input_ids)
    current_ids = input_ids
    past_key_values: Any | None = None
    generated: list[int] = []
    all_logits_finite = True
    cache_returned_every_step = True
    started = time.monotonic()
    with torch.inference_mode():
        for _index in range(MAX_NEW_TOKENS):
            output = model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = output.logits[0, -1]
            finite = bool(torch.isfinite(logits).all())
            require(finite, "candidate generation produced non-finite logits")
            all_logits_finite = all_logits_finite and finite
            token = int(torch.argmax(logits))
            generated.append(token)
            past_key_values = output.past_key_values
            cache_returned_every_step = cache_returned_every_step and past_key_values is not None
            if token in source_identity.TERMINATION_TOKEN_IDS:
                break
            current_ids = torch.tensor([[token]], dtype=input_ids.dtype)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)), dim=1)
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    termination_reason = "termination_token" if generated and generated[-1] in source_identity.TERMINATION_TOKEN_IDS else "maximum_new_tokens"
    terminating_token = generated[-1] if termination_reason == "termination_token" else None
    nonterminating = generated[:-1] if terminating_token is not None else generated
    return {
        "input_token_ids": prompt_ids,
        "input_token_ids_sha256": token_ids_sha256(prompt_ids),
        "generated_token_ids": generated,
        "generated_token_ids_sha256": token_ids_sha256(generated),
        "visible_nonterminating_generated_token_count": len(nonterminating),
        "terminating_token_id": terminating_token,
        "termination_reason": termination_reason,
        "decoded_text": decoded,
        "decoded_text_sha256": hashlib.sha256(decoded.encode()).hexdigest(),
        "wall_seconds": time.monotonic() - started,
        "model_forward_count": len(generated),
        "all_logits_finite": all_logits_finite,
        "use_cache": True,
        "cache_returned_every_step": cache_returned_every_step,
        "cache_scope": "fresh per response; retained across autoregressive token steps only",
    }


def write_jsonl_record(descriptor: int, record: dict[str, Any]) -> None:
    raw = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]
    os.fsync(descriptor)


def run_matrix(
    model: nn.Module,
    tokenizer: Any,
    matrix: dict[str, Any],
    run_kind: str,
    output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(run_kind in ("campaign", "deterministic_replay"), "run kind differs")
    policy = quality.v10.v7.v3.AuditedDynamicPolicy(load_json(BASE_SCALES))
    policy.install(model)
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    records: list[dict[str, Any]] = []
    total_forwards = 0
    started = time.monotonic()
    try:
        for case in matrix["cases"]:
            messages: list[dict[str, str]] = [{"role": "system", "content": matrix["system_message"]}]
            for turn_index, user_text in enumerate(case["user_turns"], start=1):
                messages.append({"role": "user", "content": user_text})
                guard = source_relative.unauthorized_access_guard(user_text)
                common = {
                    "schema_version": 1,
                    "candidate_id": CANDIDATE_ID,
                    "run_kind": run_kind,
                    "case_id": case["id"],
                    "category": case["category"],
                    "turn_index": turn_index,
                    "is_final_turn": turn_index == len(case["user_turns"]),
                    "raw_user_prompt": user_text,
                    "required_observable": case["required_observable"] if turn_index == len(case["user_turns"]) else None,
                    "delivery_policy": guard,
                    "software_answer_replacement": False,
                }
                if guard["blocked"]:
                    record = {
                        **common,
                        "delivery_action": "BLOCKED_NO_ANSWER",
                        "messages_before_policy_guard": [dict(item) for item in messages],
                        "model_forward_count": 0,
                        "generated_token_ids": [],
                        "generated_token_ids_sha256": token_ids_sha256([]),
                        "decoded_text": "",
                        "decoded_text_sha256": hashlib.sha256(b"").hexdigest(),
                        "visible_nonterminating_generated_token_count": 0,
                        "termination_reason": "policy_guard_block",
                        "terminating_token_id": None,
                        "wall_seconds": 0.0,
                        "all_logits_finite": True,
                        "use_cache": False,
                        "cache_returned_every_step": True,
                        "cache_scope": "not applicable; guard blocked before tokenization and model forward",
                    }
                else:
                    generated = generate_response(model, tokenizer, messages)
                    record = {
                        **common,
                        "delivery_action": "DELIVER_MODEL_OUTPUT",
                        "messages_before_generation": [dict(item) for item in messages],
                        **generated,
                    }
                    total_forwards += int(record["model_forward_count"])
                    messages.append({"role": "assistant", "content": record["decoded_text"]})
                records.append(record)
                write_jsonl_record(descriptor, record)
                print(
                    f"E248E307675A_{run_kind.upper()} {case['id']} turn={turn_index} "
                    f"action={record['delivery_action']} forwards={record['model_forward_count']} "
                    f"seconds={record['wall_seconds']:.3f}",
                    flush=True,
                )
        activation = policy.audited_summary(total_forwards)
    finally:
        os.close(descriptor)
        policy.uninstall()
    require(len(records) == EXPECTED_RESPONSE_COUNT, "quality response count differs")
    activation = dict(activation)
    activation["response_count"] = len(records)
    activation["model_forward_count"] = total_forwards
    activation["blocked_response_positions"] = sum(item["delivery_action"] == "BLOCKED_NO_ANSWER" for item in records)
    return records, {"wall_seconds": time.monotonic() - started, "activation_execution": activation}


def build_candidate_rubric(records: list[dict[str, Any]]) -> dict[str, Any]:
    source = load_json(SOURCE_BASELINE)
    source_rubric = source["rubric"]
    source_passes = {
        (item["case_id"], int(item["turn_index"]))
        for item in source_rubric["responses"]
        if item["action_pass"]
    }
    candidate_passes: set[tuple[str, int]] = set()
    responses: list[dict[str, Any]] = []
    blocked_count = 0
    delivered_count = 0
    delivered_integrity_count = 0
    critical_safety_failures = 0
    visible_single = 0
    visible_multi = 0
    holdouts = 0

    for record in records:
        key = (record["case_id"], int(record["turn_index"]))
        guard = record["delivery_policy"]
        if record["delivery_action"] == "BLOCKED_NO_ANSWER":
            blocked_count += 1
            checks = {
                "guard_matched": guard["blocked"] is True,
                "zero_model_forwards": record["model_forward_count"] == 0,
                "no_generated_tokens": record["generated_token_ids"] == [],
                "no_visible_model_text": record["decoded_text"] == "",
                "no_software_replacement": record["software_answer_replacement"] is False,
            }
            observable = record["is_final_turn"] and all(checks.values())
        else:
            delivered_count += 1
            if guard["blocked"]:
                critical_safety_failures += 1
            checks = source_relative.generic_delivery_checks(record)
            delivered_integrity_count += int(all(checks.values()))
            observable = source_probe.case_observable(record["case_id"], record["decoded_text"]) if record["is_final_turn"] else True
            checks["case_specific_required_observable"] = observable
        action_pass = all(checks.values())
        if action_pass:
            candidate_passes.add(key)
        if record["is_final_turn"] and observable:
            if record["category"] == "visible_single_turn":
                visible_single += 1
            elif record["category"] == "visible_multi_turn":
                visible_multi += 1
            elif record["category"].startswith("fresh_reviewer_holdout"):
                holdouts += 1
        responses.append(
            {
                "case_id": record["case_id"],
                "turn_index": record["turn_index"],
                "delivery_action": record["delivery_action"],
                "checks": checks,
                "action_pass": action_pass,
                "source_action_pass": key in source_passes,
            }
        )

    regressions = [
        {"case_id": case_id, "turn_index": turn_index}
        for case_id, turn_index in sorted(source_passes - candidate_passes)
    ]
    aggregate_noninferior = all(
        (
            visible_single >= int(source_rubric["visible_single_turn_required_observable_count"]),
            visible_multi >= int(source_rubric["visible_multi_turn_required_observable_count"]),
            holdouts >= int(source_rubric["public_holdout_required_observable_count"]),
        )
    )
    passed = all(
        (
            len(records) == EXPECTED_RESPONSE_COUNT,
            blocked_count == 1,
            delivered_count == 22,
            delivered_integrity_count == delivered_count,
            critical_safety_failures == 0,
            not regressions,
            aggregate_noninferior,
        )
    )
    return {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "engineer_preliminary_status": "PASS_FOR_FRESH_REVIEW" if passed else "SOURCE_RELATIVE_QUALITY_NO_GO",
        "response_position_count": len(records),
        "blocked_response_position_count": blocked_count,
        "delivered_response_position_count": delivered_count,
        "delivered_integrity_pass_count": delivered_integrity_count,
        "critical_safety_failure_count": critical_safety_failures,
        "candidate_action_pass_count": len(candidate_passes),
        "source_action_pass_count": len(source_passes),
        "source_action_pass_regressions": regressions,
        "zero_source_action_pass_regressions": not regressions,
        "visible_single_turn_required_observable_count": visible_single,
        "visible_multi_turn_required_observable_count": visible_multi,
        "public_holdout_required_observable_count": holdouts,
        "source_aggregate_observable_counts": {
            "visible_single_turn": source_rubric["visible_single_turn_required_observable_count"],
            "visible_multi_turn": source_rubric["visible_multi_turn_required_observable_count"],
            "public_holdout": source_rubric["public_holdout_required_observable_count"],
        },
        "aggregate_observable_noninferiority": aggregate_noninferior,
        "responses": responses,
        "fresh_reviewer": {
            "status": "PENDING_INDEPENDENT_REVIEW",
            "dimensions": ["correctness", "instruction_following", "safety", "readability", "multi_turn_context"],
        },
        "claim_boundary": "engineer source-relative non-inferiority screen only; independent Fresh Reviewer decides acceptance",
    }


def dynamic_preflight(model: nn.Module, tokenizer: Any) -> dict[str, Any]:
    require(not EXECUTION_MARKER.exists(), "quality marker exists before activation preflight")
    calibration = load_json(source_identity.CALIBRATION_DIR / "calibration_contract.json")
    messages = calibration["chat"]["prompts"][0]["messages"]
    input_ids = tokenize_messages(tokenizer, messages)
    policy = quality.v10.v7.v3.AuditedDynamicPolicy(load_json(BASE_SCALES))
    started = time.monotonic()
    policy.install(model)
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, use_cache=False)
        require(bool(torch.isfinite(output.logits).all()), "activation preflight produced non-finite logits")
        summary = policy.audited_summary(1)
    finally:
        policy.uninstall()
    require(summary["event_count"] == 312, "activation event count differs")
    require(summary["all_312_layer_event_names_present"] is True, "activation event closure differs")
    require(summary["saturation_count"] == 0 and summary["clipping_count"] == 0, "activation clipping differs")
    require(summary["reserved_minus_128_produced"] is False, "reserved activation payload was produced")
    token_ids = [int(value) for value in input_ids[0].tolist()]
    return {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "status": "PASS_ALL_312_DYNAMIC_SCALE32_EVENTS",
        "quality_campaign_started": False,
        "input_token_ids_sha256": token_ids_sha256(token_ids),
        "input_token_count": len(token_ids),
        "candidate_text_or_token_output_recorded": False,
        "base_scales": file_record(BASE_SCALES),
        "activation_execution": summary,
        "wall_seconds": time.monotonic() - started,
    }


def write_quality_sums() -> None:
    paths = [FROZEN_MATRIX, EXECUTION_MARKER, RAW_OUTPUTS, REPLAY_OUTPUTS, TOKEN_LATENCY, RUBRIC, RESULT]
    QUALITY_SUMS.write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in paths), encoding="utf-8")


def run_all() -> int:
    require(not OFFICIAL_ROOT.exists(), "fresh official namespace must be absent")
    torch.manual_seed(0)
    torch.set_num_threads(CONSTRUCTION_THREADS)
    torch.set_num_interop_threads(1)
    preflight_result = preflight()
    OFFICIAL_ROOT.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(
        CONSTRUCTION_MARKER,
        {
            "schema_version": 1,
            "mission_id": MISSION_ID,
            "candidate_id": CANDIDATE_ID,
            "started_at_utc": utc_now(),
            "namespace_absent_before_creation": True,
            "quality_campaign_started": False,
            "seed": 0,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {
                    **preflight_result["versions"],
                    "numpy": importlib.metadata.version("numpy"),
                },
                "construction_torch_threads": CONSTRUCTION_THREADS,
                "quality_torch_threads": QUALITY_THREADS,
            },
        },
    )
    write_json_exclusive(PREDECLARED_MANIFEST, predeclared_manifest(preflight_result))
    model_b: nn.Module | None = None
    stage_latency: dict[str, Any] = {"schema_version": 1, "candidate_id": CANDIDATE_ID, "stages": {}}
    try:
        model_a, manifest_a, witness_a, latency_a = build_candidate("candidate_a")
        write_json_exclusive(CANDIDATE_A, manifest_a)
        del model_a
        model_b, manifest_b, witness_b, latency_b = build_candidate("candidate_b")
        write_json_exclusive(CANDIDATE_B, manifest_b)
        require(manifest_a == manifest_b, "independent candidate manifests differ")
        require(witness_a == witness_b and all(witness_b.values()), "alias witnesses differ or fail")
        write_json_exclusive(
            ALIAS_WITNESS,
            {
                "schema_version": 1,
                "candidate_id": CANDIDATE_ID,
                "candidate_a": witness_a,
                "candidate_b": witness_b,
                "all_alias_proofs_pass": True,
            },
        )
        totals = manifest_b["totals"]
        accounting = {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "status": "PASS_EXACT_FOUR_BIT_PAYLOAD_AND_PRACTICAL_METADATA_CLOSURE",
            "linear_weight_count": totals["weight_count"],
            "block_count": totals["block_count"],
            "four_bit_payload_bytes": totals["payload_bytes"],
            "metadata_bytes": totals["metadata_bytes"],
            "total_linear_weight_bytes": totals["payload_bytes"] + totals["metadata_bytes"],
            "exact_payload_bits_per_weight": 4.0,
            "exact_total_bits_per_weight": 6.0,
            "bf16_linear_weight_bytes": totals["weight_count"] * 2,
            "linear_weight_byte_reduction_percent": 62.5,
            "metadata": {
                "offset_fp32_bytes": totals["block_count"] * 4,
                "step_fp32_bytes": totals["block_count"] * 4,
                "bytes_per_block": 8,
                "weights_per_block": BLOCK_WIDTH,
            },
            "activation_transport_metadata_cap_bytes_per_decoder_layer_token": 832,
            "abstract_streaming_memory_boundary_bits": 128,
            "software_reference_preserves_bf16_embedding": True,
            "hardware_weight_scope": "shared embedding/lm_head table is represented by the quantized lm_head payload; software keeps BF16 embedding to isolate Linear-weight fidelity",
        }
        write_json_exclusive(ACCOUNTING, accounting)
        tokenizer = AutoTokenizer.from_pretrained(source_identity.SNAPSHOT, local_files_only=True, trust_remote_code=False)
        dynamic = dynamic_preflight(model_b, tokenizer)
        write_json_exclusive(DYNAMIC_PREFLIGHT, dynamic)
        matrix = matrix_preview()
        QUALITY_DIR.mkdir(parents=False, exist_ok=False)
        write_json_exclusive(FROZEN_MATRIX, matrix)
        stage_latency["stages"]["candidate_a_construction"] = latency_a
        stage_latency["stages"]["candidate_b_construction"] = latency_b
        stage_latency["stages"]["dynamic_activation_preflight"] = {"wall_seconds": dynamic["wall_seconds"]}
        write_json_exclusive(
            PREFLIGHT_READY,
            {
                "schema_version": 1,
                "status": "PRODUCT_PREFLIGHT_READY",
                "candidate_id": CANDIDATE_ID,
                "candidate_manifest_sha256": sha256_file(CANDIDATE_B),
                "candidate_a_b_equal": True,
                "four_bit_payload_accounting_passed": True,
                "dynamic_event_count": dynamic["activation_execution"]["event_count"],
                "matrix": file_record(FROZEN_MATRIX),
                "quality_attempts_consumed": 0,
            },
        )
        write_json_exclusive(
            EXECUTION_MARKER,
            {
                "schema_version": 1,
                "mission_id": MISSION_ID,
                "candidate_id": CANDIDATE_ID,
                "campaign_id": "quality-campaign-0001",
                "started_at_utc": utc_now(),
                "attempt_budget": 1,
                "attempt_number": 1,
                "matrix": file_record(FROZEN_MATRIX),
                "candidate_manifest": file_record(CANDIDATE_B),
                "seed": 0,
                "sampling": False,
            },
        )

        torch.set_num_threads(QUALITY_THREADS)
        campaign, campaign_timing = run_matrix(model_b, tokenizer, matrix, "campaign", RAW_OUTPUTS)
        replay, replay_timing = run_matrix(model_b, tokenizer, matrix, "deterministic_replay", REPLAY_OUTPUTS)
        require(len(campaign) == len(replay) == EXPECTED_RESPONSE_COUNT, "campaign/replay response count differs")
        replay_checks: list[dict[str, Any]] = []
        for first, second in zip(campaign, replay, strict=True):
            match = all(
                (
                    first["case_id"] == second["case_id"],
                    first["turn_index"] == second["turn_index"],
                    first["delivery_action"] == second["delivery_action"],
                    first["generated_token_ids"] == second["generated_token_ids"],
                    first["decoded_text"] == second["decoded_text"],
                    first["model_forward_count"] == second["model_forward_count"],
                    first["cache_returned_every_step"] == second["cache_returned_every_step"],
                )
            )
            require(match, f"deterministic replay differs: {first['case_id']} turn {first['turn_index']}")
            replay_checks.append(
                {
                    "case_id": first["case_id"],
                    "turn_index": first["turn_index"],
                    "delivery_action_match": True,
                    "token_ids_match": True,
                    "decoded_text_match": True,
                    "model_forward_count_match": True,
                    "kv_cache_behavior_match": True,
                    "campaign_wall_seconds": first["wall_seconds"],
                    "replay_wall_seconds": second["wall_seconds"],
                }
            )
        token_latency = {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "response_count": len(campaign),
            "all_deterministic_replays_match": True,
            "responses": replay_checks,
            "campaign": campaign_timing,
            "deterministic_replay": replay_timing,
            "raw_latency_evidence": "per-response wall time plus aggregate campaign/replay wall time; no throughput claim",
        }
        write_json_exclusive(TOKEN_LATENCY, token_latency)
        rubric = build_candidate_rubric(campaign)
        write_json_exclusive(RUBRIC, rubric)
        ready = rubric["engineer_preliminary_status"] == "PASS_FOR_FRESH_REVIEW"
        result = {
            "schema_version": 1,
            "status": "READY_FOR_FRESH_REVIEW" if ready else "SOURCE_RELATIVE_QUALITY_NO_GO",
            "candidate_id": CANDIDATE_ID,
            "campaign_id": "quality-campaign-0001",
            "response_count": len(campaign),
            "deterministic_replay_passed": True,
            "zero_source_action_pass_regressions": rubric["zero_source_action_pass_regressions"],
            "engineer_preliminary_status": rubric["engineer_preliminary_status"],
            "fresh_reviewer_status": "PENDING",
            "selected_policy_id": None,
            "claim_boundary": "quantized software reference and source-relative quality evidence only; no RTL, U280, PPA, or stage-advance claim",
        }
        if not ready:
            result["failure_taxonomy"] = "SOURCE_RELATIVE_QUALITY_REGRESSION"
            result["root_cause_hypothesis"] = "block32 affine weight reconstruction perturbed one or more source-passing response actions or aggregate observables"
            result["required_regression"] = "preserve the immutable campaign and add a non-consuming synthetic/source-independent quantizer regression before any separately authorized structural successor"
        write_json_exclusive(RESULT, result)
        stage_latency["stages"]["quality_campaign"] = {"wall_seconds": campaign_timing["wall_seconds"]}
        stage_latency["stages"]["deterministic_replay"] = {"wall_seconds": replay_timing["wall_seconds"]}
        stage_latency["recorded_at_utc"] = utc_now()
        write_json_exclusive(STAGE_LATENCY, stage_latency)
        write_quality_sums()
        submission = {
            "schema_version": 1,
            "status": "READY_FOR_FRESH_REVIEW",
            "mission_id": MISSION_ID,
            "candidate_id": CANDIDATE_ID,
            "requested_review": "independently adjudicate source-passing regression closure, arbitrary-text/multi-turn quality and safety, deterministic replay, latency evidence, and four-bit payload accounting",
            "predeclared_manifest": file_record(PREDECLARED_MANIFEST),
            "candidate_manifest": file_record(CANDIDATE_B),
            "accounting": file_record(ACCOUNTING),
            "dynamic_preflight": file_record(DYNAMIC_PREFLIGHT),
            "frozen_matrix": file_record(FROZEN_MATRIX),
            "raw_outputs": file_record(RAW_OUTPUTS),
            "deterministic_replay_outputs": file_record(REPLAY_OUTPUTS),
            "token_and_latency_results": file_record(TOKEN_LATENCY),
            "stage_latency": file_record(STAGE_LATENCY),
            "source_relative_rubric": file_record(RUBRIC),
            "quality_result": file_record(RESULT),
            "quality_sha256s": file_record(QUALITY_SUMS),
            "verification_command": ["./.venv/bin/python", RUNNER.relative_to(ROOT).as_posix(), "verify-result"],
            "official_campaign_regeneration_forbidden": True,
            "selected_policy_id_before_review": None,
        }
        write_json_exclusive(REVIEWER_SUBMISSION, submission)
        return 0 if ready else 4
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "TERMINAL_ONE_SHOT_FAILURE",
            "candidate_id": CANDIDATE_ID,
            "failure_taxonomy": "EVALUATOR_EXECUTION_FAILURE" if EXECUTION_MARKER.exists() else "CANDIDATE_PREFLIGHT_FAILURE",
            "root_cause_hypothesis": str(exc),
            "required_regression": "reproduce the failing operation outside the official namespace without model text generation before any separately authorized successor",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "quality_campaign_started": EXECUTION_MARKER.exists(),
            "failed_at_utc": utc_now(),
            "seed": 0,
        }
        target = QUALITY_DIR / "failure.json" if QUALITY_DIR.exists() else OFFICIAL_ROOT / "failure.json"
        if not target.exists():
            write_json_exclusive(target, failure)
        print(f"E248E307675A_RUN_FAILED {type(exc).__name__}: {exc}", flush=True)
        return 5 if EXECUTION_MARKER.exists() else 2
    finally:
        if model_b is not None:
            del model_b


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_result() -> dict[str, Any]:
    required = (
        CONSTRUCTION_MARKER,
        PREDECLARED_MANIFEST,
        CANDIDATE_A,
        CANDIDATE_B,
        ALIAS_WITNESS,
        ACCOUNTING,
        DYNAMIC_PREFLIGHT,
        STAGE_LATENCY,
        PREFLIGHT_READY,
        FROZEN_MATRIX,
        EXECUTION_MARKER,
        RAW_OUTPUTS,
        REPLAY_OUTPUTS,
        TOKEN_LATENCY,
        RUBRIC,
        RESULT,
        QUALITY_SUMS,
        REVIEWER_SUBMISSION,
    )
    for path in required:
        require(path.is_file(), f"required artifact is missing: {path}")
    require(CANDIDATE_A.read_bytes() == CANDIDATE_B.read_bytes(), "candidate A/B manifest bytes differ")
    manifest = load_json(CANDIDATE_B)
    accounting = load_json(ACCOUNTING)
    require(manifest["totals"]["weight_count"] == EXPECTED_LINEAR_WEIGHT_COUNT, "verified weight count differs")
    require(accounting["four_bit_payload_bytes"] == EXPECTED_LINEAR_WEIGHT_COUNT // 2, "verified payload bytes differ")
    require(accounting["metadata_bytes"] == EXPECTED_LINEAR_WEIGHT_COUNT // 4, "verified metadata bytes differ")
    require(accounting["exact_payload_bits_per_weight"] == 4.0, "verified payload width differs")
    require(accounting["exact_total_bits_per_weight"] == 6.0, "verified practical bit cost differs")
    dynamic = load_json(DYNAMIC_PREFLIGHT)
    require(dynamic["activation_execution"]["event_count"] == 312, "verified activation event count differs")
    require(dynamic["activation_execution"]["saturation_count"] == 0, "verified activation saturation differs")

    sums: dict[str, str] = {}
    for line in QUALITY_SUMS.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    for path in (FROZEN_MATRIX, EXECUTION_MARKER, RAW_OUTPUTS, REPLAY_OUTPUTS, TOKEN_LATENCY, RUBRIC, RESULT):
        require(sums.get(path.name) == sha256_file(path), f"quality checksum differs: {path.name}")
    campaign = parse_jsonl(RAW_OUTPUTS)
    replay = parse_jsonl(REPLAY_OUTPUTS)
    require(len(campaign) == len(replay) == EXPECTED_RESPONSE_COUNT, "verified response count differs")
    for first, second in zip(campaign, replay, strict=True):
        require(first["delivery_action"] == second["delivery_action"], "verified replay action differs")
        require(first["generated_token_ids"] == second["generated_token_ids"], "verified replay tokens differ")
        require(first["decoded_text"] == second["decoded_text"], "verified replay text differs")
        require(first["model_forward_count"] == second["model_forward_count"], "verified replay forward count differs")
    stored_rubric = load_json(RUBRIC)
    require(stored_rubric == build_candidate_rubric(campaign), "stored source-relative rubric differs")
    result = load_json(RESULT)
    latency = load_json(TOKEN_LATENCY)
    require(latency["all_deterministic_replays_match"] is True, "stored deterministic replay status differs")
    require(load_json(REVIEWER_SUBMISSION)["official_campaign_regeneration_forbidden"] is True, "review submission permits regeneration")
    require("v17" not in canonical_bytes(load_json(PREDECLARED_MANIFEST)).decode().lower(), "predeclared manifest references forbidden V17 state")
    return {
        "status": "VERIFIED_IMMUTABLE_ONE_SHOT_BUNDLE",
        "candidate_id": CANDIDATE_ID,
        "quality_status": result["status"],
        "response_count": len(campaign),
        "deterministic_replay_passed": True,
        "zero_source_action_pass_regressions": stored_rubric["zero_source_action_pass_regressions"],
        "four_bit_payload_bytes": accounting["four_bit_payload_bytes"],
        "metadata_bytes": accounting["metadata_bytes"],
        "exact_total_bits_per_weight": accounting["exact_total_bits_per_weight"],
        "fresh_reviewer_status": result["fresh_reviewer_status"],
        "fresh_reviewer_submission_sha256": sha256_file(REVIEWER_SUBMISSION),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "preflight", "run-all", "verify-result"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "self-test":
        result = {
            "schema_version": 1,
            "status": "PASS_NON_CONSUMING_SELF_TEST",
            "candidate_id": CANDIDATE_ID,
            "quantizer": quantizer_self_test(),
            "guard": source_relative.guard_self_test(),
            "official_namespace_absent": not OFFICIAL_ROOT.exists(),
            "quality_marker_absent": not EXECUTION_MARKER.exists(),
            "claim_boundary": "synthetic quantizer and guard tests only; no model construction or text generation",
        }
    elif args.command == "preflight":
        result = preflight()
    elif args.command == "run-all":
        return run_all()
    else:
        result = verify_result()
    if args.output is not None:
        output = args.output.resolve()
        require(output.is_relative_to(ROOT), "output must remain inside the repository")
        require(not output.is_relative_to(OFFICIAL_ROOT), "non-consuming output may not enter the official namespace")
        output.parent.mkdir(parents=True, exist_ok=True)
        require(not output.exists(), f"output already exists: {output}")
        write_json_exclusive(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
