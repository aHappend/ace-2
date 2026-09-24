#!/usr/bin/env python3
"""Execute exactly one frozen post-V19 block16-affine BF16 W4A8 campaign."""

from __future__ import annotations

import argparse
import builtins
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_e248e307675a_block32_affine_fp32_w4a8 as backend


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve()
CANDIDATE_ID = "option_b_post_v19_block16_affine_bf16_offset_step_w4a8_v20"
MISSION_LABEL = "post-v19-block16-affine-bf16-w4a8-v20"
TASK_ID = "job-post-v19-b16bf16-v20"
OFFICIAL_ROOT = ROOT / "build/stage1-option-b-post-v19-block16-affine-bf16-w4a8-v20"
QUALITY_DIR = OFFICIAL_ROOT / "quality-campaign-0001"

SELF_TEST_OUTPUT = ROOT / "build/post-v19-block16-affine-bf16-w4a8-v20-self-test.json"
PREFLIGHT_OUTPUT = ROOT / "build/post-v19-block16-affine-bf16-w4a8-v20-preflight.json"
PREEXECUTION_CONTRACT = ROOT / "build/post-v19-block16-affine-bf16-w4a8-v20-preexecution-contract.json"
GEOMETRY_FIXTURE = ROOT / "build/post-v19-block16-affine-bf16-w4a8-v20-production-geometry-preexecution.json"
GEOMETRY_TOOL = ROOT / "tools/audit_post_v19_block16_geometry.py"
PLANNER_GEOMETRY = ROOT / "research/probes/post-v19-block16-full-production-geometry-20260807.json"
PLAN = ROOT / "design/OPTION_B_POST_V19_BLOCK16_AFFINE_BF16_W4A8_V20_PLAN.json"
TASK = ROOT / "design/OPTION_B_POST_V19_BLOCK16_AFFINE_BF16_W4A8_V20_ENGINEER_TASK.json"
WEIGHT_PROBE = ROOT / "research/probes/post-e248-weight-only-quantizer-comparison-20260807.json"
WEIGHT_PROBE_TOOL = ROOT / "tools/probe_post_e248_weight_only_quantizers.py"
PREDECESSOR_REVIEW = ROOT / "research/raw/specification/v19-block16-affine-bf16-construction-preflight-no-go-review-replan-20260807T151344Z.json"
PREDECESSOR_MANIFEST = ROOT / "build/candidate-e248e307675a-block32-affine-fp32-w4a8-v1/candidate_b_manifest.json"
BENCHMARK_INTERFACE = ROOT / "design/BENCHMARK_INTERFACE.json"
SPEC = ROOT / "design/SPEC.md"
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"

CONSTRUCTION_MARKER = OFFICIAL_ROOT / "construction_started.json"
PREDECLARED_MANIFEST = OFFICIAL_ROOT / "predeclared_manifest.json"
CANDIDATE_A = OFFICIAL_ROOT / "candidate_a_manifest.json"
CANDIDATE_B = OFFICIAL_ROOT / "candidate_b_manifest.json"
PAYLOAD_IMAGE = OFFICIAL_ROOT / "linear_weight_payload.bin"
METADATA_IMAGE = OFFICIAL_ROOT / "linear_weight_metadata.bin"
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

BLOCK_WIDTH = 16
ROW_CHUNK = 256
EXPECTED_LINEAR_TENSORS = 169
EXPECTED_LINEAR_WEIGHT_COUNT = 493_961_216
EXPECTED_BLOCK_COUNT = 30_872_576
EXPECTED_PAYLOAD_BYTES = 246_980_608
EXPECTED_METADATA_BYTES = 123_490_304
EXPECTED_TOTAL_BYTES = 370_470_912
EXPECTED_WEIGHT_SSE = 766.2914187726616

PLAN_SHA256 = "52e21cde0d57a20f5f72c03b7ccabebc0222b12d1802a5736896e98bc1e33b3a"
TASK_SHA256 = "b6c5c7bb1123409c5a1eae59b9b1e81383e8e7de0744246b20b207010042c1ef"
GEOMETRY_TOOL_SHA256 = "861b5072f0c2a9a38f1f2eb068a206dfa4ba5455b3c3b364704d00208fba62f4"
PLANNER_GEOMETRY_SHA256 = "9aa96f2269ef70db320af348647f6b4bc029f5421d28f17c4a7cc29c1d4e28eb"
PROBE_SHA256 = "d02028573a617b39a95985f570935a84279344ecdfbe72a305b8c84aea0b326e"
PROBE_TOOL_SHA256 = "f99e59eb09e81eb30ec77c39e9126e5c6849aec63a88f92e070baecad5c97cac"
PREDECESSOR_REVIEW_SHA256 = "ed38ca11690bf8f69864e47cab6accb081f638b1cfafcda8bc9aefb5bb3fe28e"
MATRIX_SHA256 = "829d3fc355ecba7dd01aed6a06c165b6a664cbc144591a0c40b45ae4fa27c6a1"

BASE_WRITE_EXCLUSIVE = backend.write_json_exclusive
BASE_PREDECLARED_MANIFEST = backend.predeclared_manifest
BASE_VERIFY_RESULT = backend.verify_result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


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
    return {
        "path": resolved.relative_to(ROOT.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def validated_geometry_fixture() -> dict[str, Any]:
    require(GEOMETRY_TOOL.is_file(), "production geometry tool is missing")
    require(PLANNER_GEOMETRY.is_file(), "planner production geometry result is missing")
    require(GEOMETRY_FIXTURE.is_file(), "fresh production geometry fixture is missing")
    require(sha256_file(GEOMETRY_TOOL) == GEOMETRY_TOOL_SHA256, "production geometry tool differs")
    require(sha256_file(PLANNER_GEOMETRY) == PLANNER_GEOMETRY_SHA256, "planner production geometry result differs")
    require(not OFFICIAL_ROOT.exists(), "official root exists while validating marker-free geometry")

    planner = load_json(PLANNER_GEOMETRY)
    fresh = load_json(GEOMETRY_FIXTURE)
    expected = {
        "linear_tensor_count": EXPECTED_LINEAR_TENSORS,
        "linear_weight_count": EXPECTED_LINEAR_WEIGHT_COUNT,
        "metadata_record_count": EXPECTED_BLOCK_COUNT,
        "payload_bytes": EXPECTED_PAYLOAD_BYTES,
        "metadata_bytes": EXPECTED_METADATA_BYTES,
        "total_bytes": EXPECTED_TOTAL_BYTES,
    }
    require(
        planner.get("status") == fresh.get("status") == "PASS_NON_CONSUMING_FULL_169_TENSOR_PRODUCTION_GEOMETRY",
        "production geometry status differs",
    )
    require(fresh.get("totals") == expected and fresh.get("expected") == expected, "fresh production geometry totals differ")
    require(planner.get("totals") == expected and planner.get("expected") == expected, "planner production geometry totals differ")
    for key in ("source", "package_versions", "representation", "invariants", "geometry_manifest_sha256", "tensors"):
        require(fresh.get(key) == planner.get(key), f"fresh production geometry field differs: {key}")
    require(fresh["geometry_manifest_sha256"] == "72f7f78f08b8dac134bee45a69dd60514e79856d7e6e3b35d658a6c2c7edab9c", "geometry manifest hash differs")
    accounting = fresh.get("execution_accounting", {})
    require(accounting.get("model_forward_count") == 0, "geometry fixture performed a model forward")
    require(accounting.get("candidate_text_or_token_output_generated") is False, "geometry fixture generated candidate output")
    require(accounting.get("quantized_candidate_constructed") is False, "geometry fixture constructed a candidate")
    require(accounting.get("official_namespace_created") is False, "geometry fixture created the official namespace")
    return {
        "tool": file_record(GEOMETRY_TOOL),
        "planner_result": file_record(PLANNER_GEOMETRY),
        "fresh_result": file_record(GEOMETRY_FIXTURE),
        "status": fresh["status"],
        "geometry_manifest_sha256": fresh["geometry_manifest_sha256"],
        "totals": fresh["totals"],
        "source": fresh["source"],
        "package_versions": fresh["package_versions"],
        "official_namespace_absent_after_fixture": not OFFICIAL_ROOT.exists(),
    }


def write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]


def bf16_bytes(values: Tensor) -> bytes:
    bits = values.detach().cpu().contiguous().view(torch.uint16).numpy()
    return bits.astype("<u2", copy=False).tobytes(order="C")


def encode_grouped(grouped: Tensor) -> dict[str, Any]:
    require(grouped.dtype == torch.float32 and grouped.shape[-1] == BLOCK_WIDTH, "codec input differs")
    minima = grouped.amin(dim=-1)
    maxima = grouped.amax(dim=-1)
    offsets_bf16 = minima.to(torch.bfloat16)
    offsets = offsets_bf16.to(torch.float32)
    raw_steps = (maxima - offsets) / 15.0
    nonconstant = maxima != offsets
    steps_bf16 = torch.where(nonconstant, raw_steps.to(torch.bfloat16), torch.zeros_like(raw_steps).to(torch.bfloat16))
    steps = steps_bf16.to(torch.float32)
    require(bool(torch.isfinite(offsets).all()), "non-finite BF16 offset")
    require(bool(torch.isfinite(steps).all()), "non-finite BF16 step")
    require(bool((steps >= 0).all()), "negative BF16 step")
    require(bool((steps[nonconstant] > 0).all()), "nonconstant block rounded to a zero BF16 step")
    denominator = torch.where(nonconstant, steps, torch.ones_like(steps))
    indices = torch.round((grouped - offsets.unsqueeze(-1)) / denominator.unsqueeze(-1))
    indices = indices.clamp(0, 15).to(torch.uint8)
    indices = torch.where(nonconstant.unsqueeze(-1), indices, torch.zeros_like(indices))
    decoded = offsets.unsqueeze(-1) + indices.to(torch.float32) * steps.unsqueeze(-1)
    decoded_bf16 = decoded.to(torch.bfloat16)
    offset_bits = offsets_bf16.contiguous().view(torch.uint16)
    step_bits = steps_bf16.contiguous().view(torch.uint16)
    metadata_pairs = torch.stack((offset_bits, step_bits), dim=-1)
    metadata_raw = metadata_pairs.numpy().astype("<u2", copy=False).tobytes(order="C")
    return {
        "indices": indices,
        "offsets_bf16": offsets_bf16,
        "steps_bf16": steps_bf16,
        "decoded_bf16": decoded_bf16,
        "metadata_raw": metadata_raw,
        "nonconstant": nonconstant,
    }


def quantize_weight_in_place(
    weight: Tensor,
    module_name: str,
    payload_descriptor: int | None = None,
    metadata_descriptor: int | None = None,
    payload_stream_digest: Any | None = None,
    metadata_stream_digest: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(weight.ndim == 2 and weight.dtype == torch.bfloat16, f"weight format differs: {module_name}")
    rows, width = (int(value) for value in weight.shape)
    require(width % BLOCK_WIDTH == 0, f"block width does not divide tensor: {module_name}")
    blocks_per_row = width // BLOCK_WIDTH
    payload_digest = hashlib.sha256()
    metadata_digest = hashlib.sha256()
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
            source_digest.update(backend.raw_tensor_bytes(source_bf16))
            source = source_bf16.to(torch.float32)
            grouped = source.reshape(row_stop - row_start, blocks_per_row, BLOCK_WIDTH)
            encoded = encode_grouped(grouped)
            indices = encoded["indices"]
            decoded_bf16 = encoded["decoded_bf16"].reshape(row_stop - row_start, width)
            weight[row_start:row_stop].copy_(decoded_bf16)

            payload_raw = backend.pack_nibbles(indices)
            metadata_raw = encoded["metadata_raw"]
            offset_raw = bf16_bytes(encoded["offsets_bf16"])
            step_raw = bf16_bytes(encoded["steps_bf16"])
            payload_digest.update(payload_raw)
            metadata_digest.update(metadata_raw)
            offset_digest.update(offset_raw)
            step_digest.update(step_raw)
            if payload_stream_digest is not None:
                payload_stream_digest.update(payload_raw)
            if metadata_stream_digest is not None:
                metadata_stream_digest.update(metadata_raw)
            if payload_descriptor is not None:
                write_all(payload_descriptor, payload_raw)
            if metadata_descriptor is not None:
                write_all(metadata_descriptor, metadata_raw)
            reconstruction_digest.update(backend.raw_tensor_bytes(decoded_bf16))
            counts = torch.bincount(indices.reshape(-1).to(torch.int64), minlength=16).tolist()
            histogram = [left + int(right) for left, right in zip(histogram, counts, strict=True)]
            difference = source - decoded_bf16.to(torch.float32)
            literal_bf16_sse += float(torch.sum(difference * difference, dtype=torch.float64))
            zero_span_blocks += int((~encoded["nonconstant"]).sum())
            stage_records.append({"row_start": row_start, "row_stop": row_stop, "seconds": time.monotonic() - started})

    weight_count = rows * width
    block_count = rows * blocks_per_row
    payload_bytes = weight_count // 2
    metadata_bytes = block_count * 4
    require(sum(histogram) == weight_count, f"histogram closure differs: {module_name}")
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
                "packing": "row-major; earlier flat element low nibble; later flat element high nibble",
                "unsigned_code_range": [0, 15],
                "histogram": histogram,
            },
            "metadata": {
                "record_order": "offset_u16_le,step_u16_le per row-local block",
                "record_count": block_count,
                "bytes_per_block": 4,
                "total_bytes": metadata_bytes,
                "sha256": metadata_digest.hexdigest(),
                "offset_bf16": {"bytes": block_count * 2, "sha256": offset_digest.hexdigest()},
                "step_bf16": {"bytes": block_count * 2, "sha256": step_digest.hexdigest()},
            },
            "source_bf16_sha256": source_digest.hexdigest(),
            "reconstruction_bf16_sha256": reconstruction_digest.hexdigest(),
            "literal_bf16_sse": literal_bf16_sse,
            "zero_span_block_count": zero_span_blocks,
            "rounding": "BF16 offset and step; round-to-nearest-ties-to-even uint4 code; clamp 0..15",
            "decoder": "bf16(fp32(bf16_offset) + uint4_code * fp32(bf16_step))",
        },
        stage_records,
    )


def build_candidate(label: str) -> tuple[nn.Module, dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        backend.source_identity.SNAPSHOT,
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
    embedding_sha256 = backend.quality.v10.v7.v3.tensor_sha256(embedding)
    model.lm_head.weight = nn.Parameter(lm_head.detach().clone(), requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head remained aliased")

    persist = label == "candidate_b"
    payload_descriptor = None
    metadata_descriptor = None
    payload_stream_digest = hashlib.sha256()
    metadata_stream_digest = hashlib.sha256()
    if persist:
        payload_descriptor = os.open(PAYLOAD_IMAGE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        metadata_descriptor = os.open(METADATA_IMAGE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)

    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == EXPECTED_LINEAR_TENSORS, "linear tensor count differs")
    records: list[dict[str, Any]] = []
    per_tensor_latency: list[dict[str, Any]] = []
    totals = {"weight_count": 0, "block_count": 0, "payload_bytes": 0, "metadata_bytes": 0}
    aggregate = {name: hashlib.sha256() for name in ("source", "payload", "metadata", "reconstruction")}
    construction_started = time.monotonic()
    try:
        for ordinal, (name, module) in enumerate(modules, start=1):
            started = time.monotonic()
            print(
                f"V20_BLOCK16_BF16_CONSTRUCT_{label.upper()} {ordinal}/{len(modules)} {name}",
                file=sys.stderr if label == "chat" else sys.stdout,
                flush=True,
            )
            record, chunks = quantize_weight_in_place(
                module.weight,
                name,
                payload_descriptor,
                metadata_descriptor,
                payload_stream_digest,
                metadata_stream_digest,
            )
            records.append(record)
            per_tensor_latency.append({"module": name, "ordinal": ordinal, "seconds": time.monotonic() - started, "chunks": chunks})
            totals["weight_count"] += int(record["weight_count"])
            totals["block_count"] += int(record["block_count"])
            totals["payload_bytes"] += int(record["payload"]["bytes"])
            totals["metadata_bytes"] += int(record["metadata"]["total_bytes"])
            for key, digest_value in (
                ("source", record["source_bf16_sha256"]),
                ("payload", record["payload"]["sha256"]),
                ("metadata", record["metadata"]["sha256"]),
                ("reconstruction", record["reconstruction_bf16_sha256"]),
            ):
                aggregate[key].update(name.encode() + b"\0" + bytes.fromhex(digest_value))
        if payload_descriptor is not None:
            os.fsync(payload_descriptor)
        if metadata_descriptor is not None:
            os.fsync(metadata_descriptor)
    finally:
        if payload_descriptor is not None:
            os.close(payload_descriptor)
        if metadata_descriptor is not None:
            os.close(metadata_descriptor)

    require(totals == {
        "weight_count": EXPECTED_LINEAR_WEIGHT_COUNT,
        "block_count": EXPECTED_BLOCK_COUNT,
        "payload_bytes": EXPECTED_PAYLOAD_BYTES,
        "metadata_bytes": EXPECTED_METADATA_BYTES,
    }, "whole-model byte geometry differs")
    require(id(model.model.embed_tokens.weight) == embedding_object, "embedding parameter object changed")
    require(model.model.embed_tokens.weight.data_ptr() == embedding_pointer, "embedding storage changed")
    require(backend.quality.v10.v7.v3.tensor_sha256(embedding) == embedding_sha256, "embedding bytes changed")

    streams = {
        "payload": {"path": PAYLOAD_IMAGE.relative_to(ROOT).as_posix(), "bytes": EXPECTED_PAYLOAD_BYTES, "sha256": payload_stream_digest.hexdigest()},
        "metadata": {"path": METADATA_IMAGE.relative_to(ROOT).as_posix(), "bytes": EXPECTED_METADATA_BYTES, "sha256": metadata_stream_digest.hexdigest()},
    }
    if persist:
        require(file_record(PAYLOAD_IMAGE) == streams["payload"], "persisted payload stream differs")
        require(file_record(METADATA_IMAGE) == streams["metadata"], "persisted metadata stream differs")
    policy = {
        "weight_payload": {"bits_per_weight": 4, "block_width": BLOCK_WIDTH, "code_range": [0, 15], "packing": "two unsigned codes per byte"},
        "weight_metadata": {"per_block": ["BF16 minimum offset", "BF16 affine step"], "bytes_per_block": 4, "decode": "bf16(fp32(offset) + code * fp32(step))"},
        "activation": {"payload": "signed int8 excluding reserved -128", "policy": "audited grouped Dynamic Scale32", "expected_named_boundaries": 312, "maximum_transport_metadata_bytes_per_decoder_layer_token": 832},
        "scope": {"quantized": "all 169 Linear weights including separated lm_head", "preserved": "source BF16 token embedding and non-Linear tensors", "lm_head_embedding_alias": "split before lm_head quantization; embedding bytes preserved"},
        "selection": "source-weight-only structural policy frozen before candidate text generation",
    }
    manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "task_id": TASK_ID,
        "source_revision": backend.source_identity.REVISION,
        "policy": policy,
        "module_count": len(records),
        "totals": totals,
        "complete_streams": streams,
        "aggregate_tensor_hashes": {name: digest.hexdigest() for name, digest in aggregate.items()},
        "embedding_bf16_sha256": embedding_sha256,
        "weight_manifest": records,
    }
    manifest["candidate_core_sha256"] = backend.canonical_sha256(manifest)
    witness = {
        "source_lm_head_and_embedding_same_parameter_object": True,
        "source_lm_head_and_embedding_same_storage": True,
        "post_split_lm_head_and_embedding_different_storage": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
        "embedding_parameter_object_preserved": id(model.model.embed_tokens.weight) == embedding_object,
        "embedding_storage_preserved": model.model.embed_tokens.weight.data_ptr() == embedding_pointer,
        "embedding_bytes_preserved": backend.quality.v10.v7.v3.tensor_sha256(embedding) == embedding_sha256,
    }
    latency = {"candidate_build": label, "total_seconds": time.monotonic() - construction_started, "per_tensor": per_tensor_latency}
    return model, manifest, witness, latency


def parse_conversation(value: Any) -> list[dict[str, str]]:
    require(isinstance(value, list) and value, "conversation must be a nonempty messages array")
    result: list[dict[str, str]] = []
    expected = "user"
    for index, item in enumerate(value):
        require(isinstance(item, dict) and set(item) == {"role", "content"}, "conversation message keys differ")
        role = item["role"]
        content = item["content"]
        require(role in {"system", "user", "assistant"} and isinstance(content, str) and content.strip(), "invalid conversation message")
        if index == 0 and role == "system":
            result.append({"role": role, "content": content})
            continue
        require(role == expected, "conversation roles must alternate user/assistant")
        result.append({"role": role, "content": content})
        expected = "assistant" if role == "user" else "user"
    require(result[-1]["role"] == "user", "conversation must end in a user message")
    return result


def parser_self_test() -> dict[str, bool]:
    single = parse_conversation([{"role": "user", "content": "hello"}])
    multi = parse_conversation([
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ])
    rejected = 0
    for invalid in (
        [],
        [{"role": "assistant", "content": "bad"}],
        [{"role": "user", "content": "x", "extra": 1}],
        [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}],
        [{"role": "user", "content": ""}],
    ):
        try:
            parse_conversation(invalid)
        except RuntimeError:
            rejected += 1
    return {"single_turn_valid": len(single) == 1, "multi_turn_valid": len(multi) == 4, "invalid_forms_rejected": rejected == 5}


def quantizer_self_test() -> dict[str, Any]:
    source = torch.tensor(
        [
            [0.25] * 16,
            [float(index - 11) / 7.0 for index in range(16)],
            [-1.0, -0.75, -0.2, 0.0, 0.1, 0.9, 1.7, 3.25, -0.5, 2.0, 1.0, -0.1, 0.4, 0.7, 2.7, 3.0],
        ],
        dtype=torch.bfloat16,
    )
    first = source.clone()
    second = source.clone()
    first_record, _ = quantize_weight_in_place(first, "synthetic")
    second_record, _ = quantize_weight_in_place(second, "synthetic")
    require(first_record == second_record and torch.equal(first, second), "synthetic codec is not deterministic")

    grouped = source.to(torch.float32).reshape(3, 1, 16)
    encoded = encode_grouped(grouped)
    require(bool((encoded["indices"][1:].amin(dim=-1) == 0).all()), "code-zero edge missing")
    require(bool((encoded["indices"][1:].amax(dim=-1) == 15).all()), "code-fifteen edge missing")
    require(int(encoded["steps_bf16"][0].view(torch.uint16)) == 0, "constant block does not use +0 BF16 step")
    require(float(encoded["offsets_bf16"][1]) < 0, "negative offset vector failed")
    expected_step = ((grouped[2].amax() - grouped[2].amin()) / 15.0).to(torch.bfloat16)
    require(int(encoded["steps_bf16"][2].view(torch.uint16)) == int(expected_step.view(torch.uint16)), "BF16 step rounding differs")
    explicit = (encoded["offsets_bf16"].to(torch.float32).unsqueeze(-1) + encoded["indices"].to(torch.float32) * encoded["steps_bf16"].to(torch.float32).unsqueeze(-1)).to(torch.bfloat16)
    require(torch.equal(explicit, encoded["decoded_bf16"]), "exact BF16 reconstruction differs")

    nibble_probe = torch.tensor([0, 1, 2, 3, 14, 15], dtype=torch.uint8)
    packed = backend.pack_nibbles(nibble_probe)
    require(packed == bytes((0x10, 0x32, 0xFE)), "nibble packing order differs")
    require(torch.equal(backend.unpack_nibbles(packed, 6), nibble_probe), "nibble unpack differs")
    metadata_probe = encode_grouped(torch.tensor([[[-1.0] + [0.5] * 15]], dtype=torch.float32))["metadata_raw"]
    require(metadata_probe[:2] == bytes((0x80, 0xBF)), "BF16 offset is not little-endian")
    expected_metadata = bf16_bytes(torch.tensor([[-1.0]], dtype=torch.bfloat16)) + bf16_bytes(
        torch.tensor([[(0.5 - (-1.0)) / 15.0]], dtype=torch.bfloat16)
    )
    require(metadata_probe[:4] == expected_metadata, "BF16 offset/step metadata order differs")
    return {
        "status": "PASS",
        "constant_block": True,
        "negative_offset": True,
        "asymmetric_range": True,
        "code_0_and_code_15_edges": True,
        "bf16_step_rounding": True,
        "nibble_packing_order": True,
        "little_endian_metadata_order": True,
        "exact_bf16_reconstruction": True,
        "deterministic_two_build_record": True,
        "parser": parser_self_test(),
        "synthetic_record_sha256": backend.canonical_sha256(first_record),
    }


def source_weight_reproduction() -> dict[str, Any]:
    predecessor = load_json(PREDECESSOR_MANIFEST)
    predecessor_by_module = {item["module"]: float(item["literal_bf16_sse"]) for item in predecessor["weight_manifest"]}
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        backend.source_identity.SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    embedding = model.model.embed_tokens.weight
    require(model.lm_head.weight is embedding and model.lm_head.weight.data_ptr() == embedding.data_ptr(), "source alias differs")
    embedding_hash = backend.quality.v10.v7.v3.tensor_sha256(embedding)
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone(), requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight.data_ptr() != embedding.data_ptr(), "preflight alias separation failed")

    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == EXPECTED_LINEAR_TENSORS, "preflight Linear tensor count differs")
    aggregate = 0.0
    strict = 0
    weight_count = 0
    with torch.no_grad():
        for ordinal, (name, module) in enumerate(modules, start=1):
            print(f"V20_BLOCK16_BF16_SOURCE_WEIGHT_PREFLIGHT {ordinal}/{len(modules)} {name}", flush=True)
            source_weight = module.weight.detach().cpu()
            rows, width = (int(value) for value in source_weight.shape)
            weight_count += rows * width
            module_sse = 0.0
            for row_start in range(0, rows, ROW_CHUNK):
                row_stop = min(rows, row_start + ROW_CHUNK)
                source = source_weight[row_start:row_stop].to(torch.float32)
                grouped = source.reshape(row_stop - row_start, width // BLOCK_WIDTH, BLOCK_WIDTH)
                reconstruction = encode_grouped(grouped)["decoded_bf16"].reshape(row_stop - row_start, width).to(torch.float32)
                difference = source - reconstruction
                module_sse += float(torch.sum(difference * difference, dtype=torch.float64))
            aggregate += module_sse
            strict += int(module_sse < predecessor_by_module[name])
    require(weight_count == EXPECTED_LINEAR_WEIGHT_COUNT, "preflight weight count differs")
    require(abs(aggregate - EXPECTED_WEIGHT_SSE) <= 1e-9, "source-weight aggregate SSE differs")
    require(strict == EXPECTED_LINEAR_TENSORS, "source-weight per-tensor strict improvement differs")
    require(backend.quality.v10.v7.v3.tensor_sha256(embedding) == embedding_hash, "preflight changed embedding bytes")
    return {
        "classification": "non_consuming_source_weight_only_reproduction",
        "linear_tensor_count": len(modules),
        "linear_weight_count": weight_count,
        "aggregate_literal_bf16_sse": aggregate,
        "per_tensor_strictly_improved_vs_predecessor": strict,
        "source_embedding_bytes_preserved": True,
        "lm_head_storage_detached": True,
        "candidate_text_or_token_output_generated": False,
        "wall_seconds": time.monotonic() - started,
    }


def expected_terminal_artifacts() -> list[str]:
    return [
        path.relative_to(OFFICIAL_ROOT).as_posix()
        for path in (
            CONSTRUCTION_MARKER, PREDECLARED_MANIFEST, CANDIDATE_A, CANDIDATE_B,
            PAYLOAD_IMAGE, METADATA_IMAGE, ALIAS_WITNESS, ACCOUNTING, DYNAMIC_PREFLIGHT,
            STAGE_LATENCY, PREFLIGHT_READY, FROZEN_MATRIX, EXECUTION_MARKER, RAW_OUTPUTS,
            REPLAY_OUTPUTS, TOKEN_LATENCY, RUBRIC, RESULT, QUALITY_SUMS, REVIEWER_SUBMISSION,
        )
    ]


def assemble_preexecution_contract(
    source: dict[str, Any], versions: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    bindings = {
        "plan": file_record(PLAN),
        "task": file_record(TASK),
        "runner": file_record(RUNNER),
        "production_geometry_tool": file_record(GEOMETRY_TOOL),
        "planner_production_geometry": file_record(PLANNER_GEOMETRY),
        "fresh_production_geometry": file_record(GEOMETRY_FIXTURE),
        "delegated_backend": file_record(Path(backend.__file__).resolve()),
        "matrix": file_record(backend.MATRIX_SOURCE),
        "source_relative_contract": file_record(backend.SOURCE_RELATIVE_CONTRACT),
        "source_relative_baseline": file_record(backend.SOURCE_BASELINE),
        "source_relative_review": file_record(backend.SOURCE_REVIEW),
        "corrected_response_evaluator": file_record(ROOT / "tools/probe_qwen_instruct_source_qualification_v1.py"),
        "guard_and_source_relative_evaluator": file_record(ROOT / "tools/probe_qwen_instruct_source_relative_quality_v2.py"),
        "source_identity": file_record(ROOT / "tools/qwen_instruct_option_b.py"),
        "activation_scales": file_record(backend.BASE_SCALES),
        "dynamic_activation_backend": file_record(Path(backend.quality.v10.v7.v3.__file__).resolve()),
        "predecessor_review": file_record(PREDECESSOR_REVIEW),
        "predecessor_manifest_for_source_weight_reproduction": file_record(PREDECESSOR_MANIFEST),
        "source_weight_probe": file_record(WEIGHT_PROBE),
        "source_weight_probe_tool": file_record(WEIGHT_PROBE_TOOL),
        "benchmark_interface": file_record(BENCHMARK_INTERFACE),
        "public_interface_spec": file_record(SPEC),
        "pipeline_state": file_record(PIPELINE_STATE),
        "non_consuming_self_test": file_record(SELF_TEST_OUTPUT),
    }
    return {
        "schema_version": 1,
        "status": "FROZEN_PREEXECUTION_CONTRACT",
        "task_id": TASK_ID,
        "candidate_id": CANDIDATE_ID,
        "mission_label": MISSION_LABEL,
        "source": {"repository": source["repository"], "revision": source["revision"], "chat_template_sha256": source["chat_template_sha256"]},
        "versions": versions,
        "production_geometry": geometry,
        "bindings": bindings,
        "namespace": {
            "official_root": OFFICIAL_ROOT.relative_to(ROOT).as_posix(),
            "must_be_absent_before_construction": True,
            "construction_marker": CONSTRUCTION_MARKER.relative_to(ROOT).as_posix(),
            "quality_marker": EXECUTION_MARKER.relative_to(ROOT).as_posix(),
            "construction_attempts_authorized": 1,
            "quality_campaigns_authorized": 1,
            "resume_replay_repair_or_second_campaign_permitted": False,
        },
        "marker_boundaries": {
            "construction_started": "after the fresh 169-tensor geometry rerun and frozen preexecution contract, then exclusive official-root creation, and before full-model construction",
            "quality_execution_started": "after two equal constructions, complete stream persistence, dynamic preflight, frozen matrix, and preflight-ready publication; before first candidate text generation",
        },
        "commands": {
            "self_test": [".venv/bin/python", RUNNER.relative_to(ROOT).as_posix(), "self-test", "--output", SELF_TEST_OUTPUT.relative_to(ROOT).as_posix()],
            "preflight": [".venv/bin/python", RUNNER.relative_to(ROOT).as_posix(), "preflight", "--output", PREFLIGHT_OUTPUT.relative_to(ROOT).as_posix()],
            "authority_consuming": [".venv/bin/python", RUNNER.relative_to(ROOT).as_posix(), "run-all"],
            "terminal_verification": [".venv/bin/python", RUNNER.relative_to(ROOT).as_posix(), "verify-result"],
            "candidate_local_chat": [".venv/bin/python", RUNNER.relative_to(ROOT).as_posix(), "chat", "[one input source]", "--max-new-tokens", "N"],
        },
        "output_path_semantics": {
            "self_test": SELF_TEST_OUTPUT.relative_to(ROOT).as_posix(),
            "preflight": PREFLIGHT_OUTPUT.relative_to(ROOT).as_posix(),
            "preexecution_contract": PREEXECUTION_CONTRACT.relative_to(ROOT).as_posix(),
            "all_three_non_consuming_paths_outside_official_root": True,
            "exclusive_create_no_overwrite": True,
            "chat": "stdout JSON only; no official artifact mutation",
        },
        "expected_terminal_artifacts": expected_terminal_artifacts(),
        "read_only_verifier_rules": {
            "complete_payload_and_metadata_hash_and_size_match": True,
            "candidate_a_b_manifest_byte_identity": True,
            "campaign_replay_token_text_forward_count_identity": True,
            "selected_policy_id_must_remain_null": True,
            "fresh_reviewer_status_must_remain_pending_until_external_adjudication": True,
        },
        "claim_boundary": "Frozen software-reference execution inputs, exact production geometry, and exactly-once boundaries only; no candidate text, quality conclusion, RTL, PPA, U280, or stage-advance claim.",
    }


def validate_frozen_contract(require_official_root_absent: bool = True) -> dict[str, Any]:
    require(PREEXECUTION_CONTRACT.is_file(), "preexecution contract is missing")
    contract = load_json(PREEXECUTION_CONTRACT)
    require(contract.get("status") == "FROZEN_PREEXECUTION_CONTRACT", "preexecution contract status differs")
    require(contract["candidate_id"] == CANDIDATE_ID and contract["task_id"] == TASK_ID, "preexecution identity differs")
    for record in contract["bindings"].values():
        path = ROOT / record["path"]
        require(path.is_file() and file_record(path) == record, f"preexecution binding drifted: {record['path']}")
    if require_official_root_absent:
        require(not OFFICIAL_ROOT.exists(), "official root exists before authority consumption")
    return contract


def preflight() -> dict[str, Any]:
    if PREFLIGHT_OUTPUT.is_file():
        validate_frozen_contract()
        result = load_json(PREFLIGHT_OUTPUT)
        require(result.get("status") == "PASS_NON_CONSUMING_PREFLIGHT", "stored preflight did not pass")
        return result

    require(not OFFICIAL_ROOT.exists(), "fresh official namespace is not absent")
    require(not PREEXECUTION_CONTRACT.exists(), "preexecution contract exists without the frozen preflight output")
    require(sha256_file(PLAN) == PLAN_SHA256 and sha256_file(TASK) == TASK_SHA256, "plan/task hash differs")
    geometry = validated_geometry_fixture()
    require(sha256_file(WEIGHT_PROBE) == PROBE_SHA256 and sha256_file(WEIGHT_PROBE_TOOL) == PROBE_TOOL_SHA256, "source-weight probe binding differs")
    require(sha256_file(PREDECESSOR_REVIEW) == PREDECESSOR_REVIEW_SHA256, "predecessor review hash differs")
    require(sha256_file(backend.MATRIX_SOURCE) == MATRIX_SHA256, "frozen matrix hash differs")
    source = backend.source_identity.verify_source_snapshot()
    versions = backend.source_identity.verify_versions()
    contract = backend.source_relative.verify_contract()
    guard = backend.source_relative.guard_self_test()
    detector = backend.source_probe.detector_self_test()
    quantizer = quantizer_self_test()
    matrix = backend.matrix_preview()
    benchmark = load_json(BENCHMARK_INTERFACE)
    require(benchmark.get("stage") == "specification" and benchmark.get("applies") is False, "benchmark/stage boundary differs")
    require(benchmark.get("product_compression_gate", {}).get("selected_policy_id") is None, "selected_policy_id is not null")
    task = load_json(TASK)
    plan = load_json(PLAN)
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "task identity differs")
    require(plan["candidate_id"] == CANDIDATE_ID and plan["stage"] == "specification", "plan identity/stage differs")
    require(plan["fresh_identity"]["task_id"] == TASK_ID, "plan task identity differs")
    require(task["mandatory_marker_free_geometry_fixture"]["required_totals"] == {
        "linear_tensor_count": EXPECTED_LINEAR_TENSORS,
        "linear_weight_count": EXPECTED_LINEAR_WEIGHT_COUNT,
        "metadata_record_count": EXPECTED_BLOCK_COUNT,
        "payload_bytes": EXPECTED_PAYLOAD_BYTES,
        "metadata_bytes": EXPECTED_METADATA_BYTES,
        "total_bytes": EXPECTED_TOTAL_BYTES,
    }, "task production geometry differs")
    weight_reproduction = source_weight_reproduction()
    preexecution = assemble_preexecution_contract(source, versions, geometry)
    preexecution_raw = canonical_bytes(preexecution)
    preexecution_record = {
        "path": PREEXECUTION_CONTRACT.relative_to(ROOT).as_posix(),
        "bytes": len(preexecution_raw),
        "sha256": hashlib.sha256(preexecution_raw).hexdigest(),
    }
    preliminary = {
        "source": source,
        "source_relative_contract": file_record(backend.SOURCE_RELATIVE_CONTRACT),
        "source_relative_baseline": file_record(backend.SOURCE_BASELINE),
        "source_relative_review": file_record(backend.SOURCE_REVIEW),
        "preexecution_contract": preexecution_record,
    }
    manifest_preview = predeclared_manifest(preliminary)
    require(manifest_preview["candidate_id"] == CANDIDATE_ID, "predeclared manifest identity differs")
    require(manifest_preview["expected_artifacts"] == expected_terminal_artifacts(), "predeclared artifact list differs")
    BASE_WRITE_EXCLUSIVE(PREEXECUTION_CONTRACT, preexecution)
    require(file_record(PREEXECUTION_CONTRACT) == preexecution_record, "written preexecution contract differs")
    return {
        "schema_version": 1,
        "status": "PASS_NON_CONSUMING_PREFLIGHT",
        "candidate_id": CANDIDATE_ID,
        "task_id": TASK_ID,
        "official_namespace_absent": True,
        "construction_marker_absent": True,
        "quality_marker_absent": True,
        "selected_policy_id": None,
        "stage": "specification",
        "source": source,
        "versions": versions,
        "production_geometry": geometry,
        "source_relative_contract": file_record(backend.SOURCE_RELATIVE_CONTRACT),
        "source_relative_baseline": file_record(backend.SOURCE_BASELINE),
        "source_relative_review": file_record(backend.SOURCE_REVIEW),
        "guard_self_test": guard,
        "detector_self_test": detector,
        "quantizer_self_test": quantizer,
        "source_weight_reproduction": weight_reproduction,
        "matrix": {"source": file_record(backend.MATRIX_SOURCE), "candidate_matrix_content_sha256": matrix["matrix_content_sha256"], "case_count": len(matrix["cases"]), "response_count": matrix["response_count"]},
        "preexecution_contract": preexecution_record,
        "predeclared_manifest_preview_sha256": backend.canonical_sha256(manifest_preview),
        "predeclared_manifest_preview_valid": True,
        "marker_boundaries": preexecution["marker_boundaries"],
        "output_path_semantics": preexecution["output_path_semantics"],
        "claim_boundary": "Non-consuming production geometry, restoration, source-weight, codec, parser, matrix, evaluator, guard, manifest, marker, namespace, and binding checks only; no candidate text generation.",
    }


def predeclared_manifest(preflight_result: dict[str, Any]) -> dict[str, Any]:
    manifest = BASE_PREDECLARED_MANIFEST(preflight_result)
    manifest["mission_id"] = MISSION_LABEL
    manifest["task_id"] = TASK_ID
    manifest["candidate_id"] = CANDIDATE_ID
    manifest["selection"].update({
        "structural_policy": "row-local block-16 affine uint4 with BF16 offset and BF16 step",
        "block_width": BLOCK_WIDTH,
        "candidate_output_or_hidden_golden_inspected_before_freeze": False,
    })
    manifest["practical_accounting"].update({
        "linear_weight_count": EXPECTED_LINEAR_WEIGHT_COUNT,
        "four_bit_payload_bytes": EXPECTED_PAYLOAD_BYTES,
        "metadata_record_count": EXPECTED_BLOCK_COUNT,
        "metadata_bytes": EXPECTED_METADATA_BYTES,
        "total_linear_weight_bytes": EXPECTED_TOTAL_BYTES,
        "metadata_semantics": "one little-endian BF16 offset and one little-endian BF16 step per 16 weights",
    })
    if "preexecution_contract" in preflight_result:
        manifest["preexecution_contract"] = copy.deepcopy(preflight_result["preexecution_contract"])
    else:
        manifest["preexecution_contract"] = file_record(PREEXECUTION_CONTRACT)
    manifest["tools"]["delegated_construction_quality_backend"] = file_record(Path(backend.__file__).resolve())
    manifest["expected_artifacts"] = expected_terminal_artifacts()
    manifest["complete_streams"] = {
        "payload": PAYLOAD_IMAGE.relative_to(ROOT).as_posix(),
        "metadata": METADATA_IMAGE.relative_to(ROOT).as_posix(),
    }
    return manifest


def transformed_write_json_exclusive(path: Path, value: Any) -> None:
    updated = copy.deepcopy(value)
    if path == ACCOUNTING:
        updated["metadata"] = {
            "offset_bf16_bytes": EXPECTED_BLOCK_COUNT * 2,
            "step_bf16_bytes": EXPECTED_BLOCK_COUNT * 2,
            "bytes_per_block": 4,
            "weights_per_block": BLOCK_WIDTH,
            "record_order": "offset_u16_le,step_u16_le",
        }
        updated["complete_streams"] = {"payload": file_record(PAYLOAD_IMAGE), "metadata": file_record(METADATA_IMAGE)}
    elif path == PREFLIGHT_READY:
        updated["complete_payload"] = file_record(PAYLOAD_IMAGE)
        updated["complete_metadata"] = file_record(METADATA_IMAGE)
    elif path == RESULT:
        if updated.get("status") == "READY_FOR_FRESH_REVIEW":
            updated["status"] = "SOURCE_RELATIVE_QUALITY_READY_FOR_FRESH_REVIEW"
        if "root_cause_hypothesis" in updated:
            updated["root_cause_hypothesis"] = "block16 affine BF16 weight reconstruction perturbed one or more source-passing response actions or aggregate observables"
    elif path == REVIEWER_SUBMISSION:
        updated["task_id"] = TASK_ID
        updated["preexecution_contract"] = file_record(PREEXECUTION_CONTRACT)
        updated["complete_payload"] = file_record(PAYLOAD_IMAGE)
        updated["complete_metadata"] = file_record(METADATA_IMAGE)
        updated["candidate_local_chat_command"] = [".venv/bin/python", RUNNER.relative_to(ROOT).as_posix(), "chat", "[one input source]", "--max-new-tokens", "N"]
    BASE_WRITE_EXCLUSIVE(path, updated)


def write_quality_sums() -> None:
    paths = [
        CONSTRUCTION_MARKER, PREDECLARED_MANIFEST, CANDIDATE_A, CANDIDATE_B,
        PAYLOAD_IMAGE, METADATA_IMAGE, ALIAS_WITNESS, ACCOUNTING, DYNAMIC_PREFLIGHT,
        PREFLIGHT_READY, FROZEN_MATRIX, EXECUTION_MARKER, RAW_OUTPUTS, REPLAY_OUTPUTS,
        TOKEN_LATENCY, RUBRIC, RESULT, STAGE_LATENCY,
    ]
    raw = "".join(f"{sha256_file(path)}  {path.name}\n" for path in paths).encode()
    descriptor = os.open(QUALITY_SUMS, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        write_all(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def translated_print(*args: Any, **kwargs: Any) -> None:
    if args and isinstance(args[0], str):
        args = (args[0].replace("E248E307675A", "V20_BLOCK16_BF16"), *args[1:])
    builtins.print(*args, **kwargs)


def bind_backend() -> None:
    replacements = {
        "CANDIDATE_ID": CANDIDATE_ID,
        "MISSION_ID": MISSION_LABEL,
        "RUNNER": RUNNER,
        "OFFICIAL_ROOT": OFFICIAL_ROOT,
        "QUALITY_DIR": QUALITY_DIR,
        "CONSTRUCTION_MARKER": CONSTRUCTION_MARKER,
        "PREDECLARED_MANIFEST": PREDECLARED_MANIFEST,
        "CANDIDATE_A": CANDIDATE_A,
        "CANDIDATE_B": CANDIDATE_B,
        "ALIAS_WITNESS": ALIAS_WITNESS,
        "ACCOUNTING": ACCOUNTING,
        "DYNAMIC_PREFLIGHT": DYNAMIC_PREFLIGHT,
        "STAGE_LATENCY": STAGE_LATENCY,
        "PREFLIGHT_READY": PREFLIGHT_READY,
        "FROZEN_MATRIX": FROZEN_MATRIX,
        "EXECUTION_MARKER": EXECUTION_MARKER,
        "RAW_OUTPUTS": RAW_OUTPUTS,
        "REPLAY_OUTPUTS": REPLAY_OUTPUTS,
        "TOKEN_LATENCY": TOKEN_LATENCY,
        "RUBRIC": RUBRIC,
        "RESULT": RESULT,
        "QUALITY_SUMS": QUALITY_SUMS,
        "REVIEWER_SUBMISSION": REVIEWER_SUBMISSION,
        "BLOCK_WIDTH": BLOCK_WIDTH,
        "quantize_weight_in_place": quantize_weight_in_place,
        "build_candidate": build_candidate,
        "quantizer_self_test": quantizer_self_test,
        "preflight": preflight,
        "predeclared_manifest": predeclared_manifest,
        "write_json_exclusive": transformed_write_json_exclusive,
        "write_quality_sums": write_quality_sums,
        "print": translated_print,
    }
    for name, value in replacements.items():
        setattr(backend, name, value)


def verify_result() -> dict[str, Any]:
    result = BASE_VERIFY_RESULT()
    contract = validate_frozen_contract(require_official_root_absent=False)
    require(contract["bindings"]["runner"] == file_record(RUNNER), "verified runner differs from preexecution freeze")
    require(contract["production_geometry"]["totals"] == {
        "linear_tensor_count": EXPECTED_LINEAR_TENSORS,
        "linear_weight_count": EXPECTED_LINEAR_WEIGHT_COUNT,
        "metadata_record_count": EXPECTED_BLOCK_COUNT,
        "payload_bytes": EXPECTED_PAYLOAD_BYTES,
        "metadata_bytes": EXPECTED_METADATA_BYTES,
        "total_bytes": EXPECTED_TOTAL_BYTES,
    }, "verified production geometry differs")
    require(contract["production_geometry"]["geometry_manifest_sha256"] == "72f7f78f08b8dac134bee45a69dd60514e79856d7e6e3b35d658a6c2c7edab9c", "verified geometry manifest differs")
    manifest = load_json(CANDIDATE_B)
    require(PAYLOAD_IMAGE.stat().st_size == EXPECTED_PAYLOAD_BYTES, "verified payload image size differs")
    require(METADATA_IMAGE.stat().st_size == EXPECTED_METADATA_BYTES, "verified metadata image size differs")
    require(file_record(PAYLOAD_IMAGE) == manifest["complete_streams"]["payload"], "verified payload image hash differs")
    require(file_record(METADATA_IMAGE) == manifest["complete_streams"]["metadata"], "verified metadata image hash differs")
    require(manifest["totals"]["block_count"] == EXPECTED_BLOCK_COUNT, "verified metadata record count differs")
    accounting = load_json(ACCOUNTING)
    require(accounting["total_linear_weight_bytes"] == EXPECTED_TOTAL_BYTES, "verified total bytes differ")
    require(accounting["metadata"]["bytes_per_block"] == 4, "verified BF16 metadata width differs")
    stored_result = load_json(RESULT)
    require(stored_result.get("selected_policy_id") is None, "verified selected_policy_id is not null")
    result["task_id"] = TASK_ID
    result["payload_image_sha256"] = sha256_file(PAYLOAD_IMAGE)
    result["metadata_image_sha256"] = sha256_file(METADATA_IMAGE)
    result["preexecution_contract_sha256"] = sha256_file(PREEXECUTION_CONTRACT)
    return result


def chat_messages(args: argparse.Namespace) -> list[dict[str, str]]:
    explicit = [args.prompt is not None, args.prompt_file is not None, args.conversation_file is not None]
    require(sum(explicit) <= 1, "choose exactly one explicit chat input source")
    if args.conversation_file is not None:
        raw = args.conversation_file.read_text(encoding="utf-8")
        return parse_conversation(json.loads(raw))
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    elif args.prompt is not None:
        prompt = args.prompt
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read()
    else:
        prompt = input("User: ")
    require(prompt.strip() != "", "chat prompt must be nonempty")
    system = backend.source_relative.verify_contract()["generation_contract"]["system_message"]
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def run_chat(args: argparse.Namespace) -> dict[str, Any]:
    validate_frozen_contract(require_official_root_absent=False)
    messages = chat_messages(args)
    last_user = next(item["content"] for item in reversed(messages) if item["role"] == "user")
    guard = backend.source_relative.unauthorized_access_guard(last_user)
    if guard["blocked"]:
        return {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "delivery_action": "BLOCKED_NO_ANSWER",
            "generated_token_ids": [],
            "decoded_text": "",
            "wall_seconds": 0.0,
            "model_forward_count": 0,
            "delivery_policy": guard,
        }
    torch.manual_seed(0)
    torch.set_num_threads(backend.QUALITY_THREADS)
    model, _manifest, _witness, construction_latency = build_candidate("chat")
    tokenizer = AutoTokenizer.from_pretrained(backend.source_identity.SNAPSHOT, local_files_only=True, trust_remote_code=False)
    policy = backend.quality.v10.v7.v3.AuditedDynamicPolicy(load_json(backend.BASE_SCALES))
    old_limit = backend.MAX_NEW_TOKENS
    backend.MAX_NEW_TOKENS = args.max_new_tokens
    policy.install(model)
    try:
        generated = backend.generate_response(model, tokenizer, messages)
    finally:
        policy.uninstall()
        backend.MAX_NEW_TOKENS = old_limit
    return {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "delivery_action": "DELIVER_MODEL_OUTPUT",
        "delivery_policy": guard,
        "construction_wall_seconds": construction_latency["total_seconds"],
        **generated,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("self-test", "preflight"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--output", required=True, type=Path)
    subparsers.add_parser("run-all")
    subparsers.add_parser("verify-result")
    chat = subparsers.add_parser("chat")
    chat.add_argument("--prompt")
    chat.add_argument("--prompt-file", type=Path)
    chat.add_argument("--conversation-file", type=Path)
    chat.add_argument("--max-new-tokens", required=True, type=int, choices=range(1, 33), metavar="N")
    args = parser.parse_args()

    bind_backend()
    if args.command == "self-test":
        output = args.output.resolve()
        require(output == SELF_TEST_OUTPUT.resolve(), "self-test output path differs from frozen command")
        require(not output.exists(), "self-test output already exists")
        require(not OFFICIAL_ROOT.exists(), "self-test may not run after official namespace creation")
        result = {
            "schema_version": 1,
            "status": "PASS_NON_CONSUMING_SELF_TEST",
            "candidate_id": CANDIDATE_ID,
            "task_id": TASK_ID,
            "quantizer": quantizer_self_test(),
            "guard": backend.source_relative.guard_self_test(),
            "official_namespace_absent": True,
            "candidate_text_or_token_output_generated": False,
            "claim_boundary": "synthetic codec, parser, and guard tests only",
        }
        BASE_WRITE_EXCLUSIVE(output, result)
    elif args.command == "preflight":
        output = args.output.resolve()
        require(output == PREFLIGHT_OUTPUT.resolve(), "preflight output path differs from frozen command")
        require(not output.exists(), "preflight output already exists")
        result = preflight()
        BASE_WRITE_EXCLUSIVE(output, result)
    elif args.command == "run-all":
        require(SELF_TEST_OUTPUT.is_file() and PREFLIGHT_OUTPUT.is_file(), "frozen self-test/preflight outputs are missing")
        validate_frozen_contract()
        return backend.run_all()
    elif args.command == "verify-result":
        result = verify_result()
    else:
        result = run_chat(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
