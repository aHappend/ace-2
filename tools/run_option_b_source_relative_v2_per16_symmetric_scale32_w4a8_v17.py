#!/usr/bin/env python3
"""Execute the exactly-once V17 source-relative W4A8 candidate attempt."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import probe_qwen_instruct_source_qualification_v1 as source_probe
import probe_qwen_instruct_source_relative_quality_v2 as relative
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    SNAPSHOT,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
)
from run_option_b_alias_safe_tensor_codebook_w4_grouped_dynamic_scale32_stage1 import (
    AuditedDynamicPolicy,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "option_b_source_relative_v2_per16_symmetric_scale32_w4a8_v17"
MISSION_ID = "successor-of-322893a33278-v17"
TASK_ID = "task-da4c8d181e81-v17"

PLAN_PATH = ROOT / "design/OPTION_B_SOURCE_RELATIVE_V2_PER16_SYMMETRIC_SCALE32_W4A8_V17_PLAN.json"
TASK_PATH = ROOT / "design/OPTION_B_SOURCE_RELATIVE_V2_PER16_SYMMETRIC_SCALE32_W4A8_V17_ENGINEER_TASK.json"
RUNNER_PATH = Path(__file__).resolve()
SOURCE_RELATIVE_CONTRACT_PATH = ROOT / "design/QWEN_INSTRUCT_SOURCE_RELATIVE_QUALITY_V2.json"
SOURCE_BASELINE_PATH = ROOT / "research/probes/qwen-instruct-source-relative-quality-v2-repair1-20260807T111432Z.json"
SOURCE_REVIEWER_REPRODUCTION_PATH = ROOT / "research/probes/qwen-instruct-source-relative-quality-v2-reviewer-20260807T112030Z.json"
MATRIX_PATH = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v16/quality-campaign-0001/frozen_matrix.json"
V16_PLAN_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V16_PLAN.json"
V16_TASK_PATH = ROOT / "design/OPTION_B_ALIAS_SAFE_V10_TRANSFORMER_V12_LM_HEAD64_PRODUCT_CHAT_W4_GROUPED_DYNAMIC_SCALE32_W4A8_V16_ENGINEER_TASK.json"
V16_REVIEW_PATH = ROOT / "research/raw/specification/v16-product-quality-review-done-20260807T102540Z.json"
SPECIFICATION_AUDIT_PATH = ROOT / "research/raw/specification/specification-stage-audit-source-relative-quality-v2-20260807T113500Z.json"
BASE_SCALES_PATH = CALIBRATION_DIR / "derived_scales.json"
DYNAMIC_HELPER_PATH = ROOT / "tools/run_option_b_alias_safe_tensor_codebook_w4_grouped_dynamic_scale32_stage1.py"

OFFICIAL_ROOT = ROOT / "build/stage1-option-b-source-relative-v2-per16-symmetric-scale32-w4a8-v17"
CONSTRUCTION_MARKER = OFFICIAL_ROOT / "construction_started.json"
CONSTRUCTOR_SELF_TEST = OFFICIAL_ROOT / "constructor_self_test.json"
CANDIDATE_A_MANIFEST = OFFICIAL_ROOT / "candidate_a_manifest.json"
CANDIDATE_B_MANIFEST = OFFICIAL_ROOT / "candidate_b_manifest.json"
PAYLOAD_PATH = OFFICIAL_ROOT / "candidate_payload_int4.bin"
SCALE_PATH = OFFICIAL_ROOT / "candidate_scale32.bin"
ACCOUNTING_PATH = OFFICIAL_ROOT / "codec_and_byte_accounting.json"
DYNAMIC_PATH = OFFICIAL_ROOT / "dynamic_preflight.json"
CONTRACT_PATH = OFFICIAL_ROOT / "predeclared_contract.json"
PREFLIGHT_READY_PATH = OFFICIAL_ROOT / "SOURCE_RELATIVE_PREFLIGHT_READY.json"
PREFLIGHT_NO_GO_PATH = OFFICIAL_ROOT / "SOURCE_RELATIVE_PREFLIGHT_NO_GO.json"
QUALITY_DIR = OFFICIAL_ROOT / "quality-campaign-0001"
FROZEN_MATRIX_PATH = QUALITY_DIR / "frozen_matrix.json"
QUALITY_MARKER = QUALITY_DIR / "execution_started.json"
RAW_OUTPUTS_PATH = QUALITY_DIR / "raw_outputs.jsonl"
TOKEN_LATENCY_PATH = QUALITY_DIR / "token_and_latency_results.json"
RUBRIC_PATH = QUALITY_DIR / "source_relative_noninferiority.json"
RESULT_PATH = QUALITY_DIR / "RESULT.json"
SHA256SUMS_PATH = QUALITY_DIR / "SHA256SUMS"
SUBMISSION_PATH = OFFICIAL_ROOT / "fresh_reviewer_submission.json"

SOURCE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
BLOCK_WEIGHTS = 16
SCALE_BYTES = 4
WEIGHT_COUNT = 493_961_216
PAYLOAD_BYTES = WEIGHT_COUNT // 2
METADATA_BYTES = WEIGHT_COUNT // BLOCK_WEIGHTS * SCALE_BYTES
TOTAL_WEIGHT_BYTES = PAYLOAD_BYTES + METADATA_BYTES
MAX_NEW_TOKENS = 64
TERMINATION_TOKEN_IDS = (151643, 151645)
TORCH_THREADS = 32

FROZEN_HASHES = {
    SOURCE_RELATIVE_CONTRACT_PATH: "41b889c048dcb377c29de322028160dde18a7e6f735cccdf684dec653430e03b",
    SOURCE_BASELINE_PATH: "9058d84c8a0f0a7f483c07486ffc9898c0bb967d03ee8c110042851e84974ecb",
    SOURCE_REVIEWER_REPRODUCTION_PATH: "fdbba290ef91bd522c0a8e836eeeac7743fad5b5d836084a983b4db1c1cc489e",
    MATRIX_PATH: "829d3fc355ecba7dd01aed6a06c165b6a664cbc144591a0c40b45ae4fa27c6a1",
    V16_PLAN_PATH: "31d92e1a8db77203942bd122a66933b98cdff4850cd3825a16211319d7de6723",
    V16_TASK_PATH: "f965df8d2f2f886b94036809e72a9ed9f8a4ca160ba992f3074c97696b97747e",
    V16_REVIEW_PATH: "270744fd9a955be777258146e5cbe5f09a69b0245265348939975bd0d76096cf",
    SPECIFICATION_AUDIT_PATH: "d98c7bbebc2b5681afc9f17498c65aec9795e9eadddbb2a90f47ad84cb4ed07c",
    ROOT / "tools/probe_qwen_instruct_source_relative_quality_v2.py": "e43ffc4d65f3843be7941c99d473178057e76de5435fd6b9847ab050a28714e6",
    ROOT / "tools/probe_qwen_instruct_source_qualification_v1.py": "10097513bcbb9f68f72594517c43cf579200961b3fbcffce6e91fd19050839b4",
    DYNAMIC_HELPER_PATH: "24ff07050722dbbff713a8153ff681f9b2a966a806d782c52f42069c93eff6d8",
    BASE_SCALES_PATH: "cb99f71eeca54dbbc1a703dbcde4a9f3e7868ea317b0734addbf1a45d19b07ca",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(value)
    if exclusive:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value.encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_companion(path: Path) -> str:
    companion = path.with_suffix(".sha256")
    require(path.is_file() and companion.is_file(), f"missing frozen file or companion: {path}")
    expected = f"{sha256_file(path)}  {path.name}"
    require(companion.read_text(encoding="utf-8").strip() == expected, f"checksum companion differs: {path}")
    return expected.split()[0]


def tensor_bf16_bytes(value: Tensor) -> bytes:
    raw = value.detach().contiguous().view(torch.uint16).cpu().numpy()
    return raw.astype("<u2", copy=False).tobytes()


def token_ids_sha256(token_ids: list[int]) -> str:
    raw = b"".join(int(value).to_bytes(4, "little", signed=False) for value in token_ids)
    return hashlib.sha256(raw).hexdigest()


def quantize_block16(source: Tensor) -> tuple[bytes, bytes, Tensor, dict[str, Any]]:
    require(source.ndim == 2 and source.shape[1] % BLOCK_WEIGHTS == 0, "block16 source shape differs")
    rows, inputs = source.shape
    blocks = source.float().reshape(rows, inputs // BLOCK_WEIGHTS, BLOCK_WEIGHTS)
    scales = blocks.abs().amax(dim=-1) / 7.0
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    q = torch.round(blocks / safe_scales.unsqueeze(-1)).clamp(-7, 7).to(torch.int8)
    q = torch.where(scales.unsqueeze(-1) == 0, torch.zeros_like(q), q)
    reconstructed = (q.float() * scales.unsqueeze(-1)).reshape(rows, inputs).to(torch.bfloat16)

    codes = torch.bitwise_and(q.to(torch.int16), 0xF).to(torch.uint8).reshape(-1)
    require(codes.numel() % 2 == 0, "odd int4 payload count")
    packed = torch.bitwise_or(codes[0::2], torch.bitwise_left_shift(codes[1::2], 4))
    payload = packed.cpu().numpy().tobytes()
    scale_bytes = scales.contiguous().cpu().numpy().astype("<f4", copy=False).tobytes()
    require(len(payload) == source.numel() // 2, "payload byte count differs")
    require(len(scale_bytes) == source.numel() // BLOCK_WEIGHTS * SCALE_BYTES, "scale byte count differs")
    stats = {
        "scale_count": int(scales.numel()),
        "zero_scale_count": int((scales == 0).sum()),
        "minimum_nonzero_scale": float(scales[scales > 0].min()) if bool((scales > 0).any()) else 0.0,
        "maximum_scale": float(scales.max()),
        "reserved_minus8_payload_count": int((q == -8).sum()),
    }
    return payload, scale_bytes, reconstructed, stats


def decode_block16(payload: bytes, scale_bytes: bytes, shape: tuple[int, int]) -> Tensor:
    rows, inputs = shape
    packed = np.frombuffer(payload, dtype=np.uint8)
    codes = np.empty(packed.size * 2, dtype=np.uint8)
    codes[0::2] = packed & 0xF
    codes[1::2] = packed >> 4
    signed = codes.astype(np.int8)
    signed[signed >= 8] -= 16
    scales = np.frombuffer(scale_bytes, dtype="<f4").reshape(rows, inputs // BLOCK_WEIGHTS)
    decoded = signed.reshape(rows, inputs // BLOCK_WEIGHTS, BLOCK_WEIGHTS).astype(np.float32)
    decoded *= scales[..., None]
    return torch.from_numpy(decoded.reshape(rows, inputs)).to(torch.bfloat16)


def baseline_summary() -> dict[str, Any]:
    baseline = load_json(SOURCE_BASELINE_PATH)
    reviewer = load_json(SOURCE_REVIEWER_REPRODUCTION_PATH)
    require(baseline["status"] == reviewer["status"] == "SOURCE_QUALIFIED_RELATIVE_BASELINE", "source baseline status differs")
    require(baseline["rubric"] == reviewer["rubric"], "reviewer source rubric reproduction differs")
    baseline_index = {
        (item["case_id"], int(item["turn_index"])): bool(item["action_pass"])
        for item in baseline["rubric"]["responses"]
    }
    require(len(baseline_index) == 23, "source baseline response index differs")
    return {
        "action_pass_index": baseline_index,
        "rubric": baseline["rubric"],
        "baseline": baseline,
    }


def verify_frozen_inputs(*, require_namespace_absent: bool) -> dict[str, Any]:
    for path, expected in FROZEN_HASHES.items():
        require(path.is_file(), f"frozen input missing: {path}")
        require(sha256_file(path) == expected, f"frozen input differs: {path}")
    plan_sha256 = verify_companion(PLAN_PATH)
    task_sha256 = verify_companion(TASK_PATH)
    runner_sha256 = verify_companion(RUNNER_PATH)
    contract = relative.verify_contract()
    require(contract["contract_id"] == "qwen2.5-0.5b-instruct-source-relative-quality-v2", "V2 contract differs")
    matrix = load_json(MATRIX_PATH)
    require(matrix["response_count"] == 23 and len(matrix["cases"]) == 19, "frozen matrix dimensions differ")
    baseline = baseline_summary()
    require(baseline["rubric"]["action_pass_count"] == 12, "source baseline action count differs")
    v16_plan = load_json(V16_PLAN_PATH)
    v16_task = load_json(V16_TASK_PATH)
    require(v16_plan["candidate_id"].endswith("_v16"), "V16 candidate identity differs")
    require("dual codebooks" in v16_task["required_constructor"]["transformer"].lower(), "V16 transformer representation witness differs")
    require(v16_task["required_format_closure"]["whole_model_exact_bits_per_weight"] < 6.0, "V16 format distinction differs")
    plan = load_json(PLAN_PATH)
    require(plan["representation"]["block_weights"] == BLOCK_WEIGHTS, "V17 block size differs")
    require(plan["representation"]["scale_record"] == "IEEE-754 binary32 little-endian", "V17 Scale32 record differs")
    require(plan["structural_distinctness"]["v13_v16_dual_codebooks_per_input_block"] is False, "V17 structural distinction differs")
    if require_namespace_absent:
        require(not OFFICIAL_ROOT.exists(), "fresh V17 namespace must be absent")
    return {
        "plan_sha256": plan_sha256,
        "task_sha256": task_sha256,
        "runner_sha256": runner_sha256,
        "source_relative_contract_sha256": FROZEN_HASHES[SOURCE_RELATIVE_CONTRACT_PATH],
        "source_baseline_sha256": FROZEN_HASHES[SOURCE_BASELINE_PATH],
        "source_reviewer_reproduction_sha256": FROZEN_HASHES[SOURCE_REVIEWER_REPRODUCTION_PATH],
        "matrix_sha256": FROZEN_HASHES[MATRIX_PATH],
        "base_scales_sha256": FROZEN_HASHES[BASE_SCALES_PATH],
    }


def self_test() -> dict[str, Any]:
    bindings = verify_frozen_inputs(require_namespace_absent=True)
    source = torch.tensor(
        [[-1.0, -0.75, -0.5, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0, 0.3, -0.3, 0.9]],
        dtype=torch.bfloat16,
    )
    first = quantize_block16(source)
    second = quantize_block16(source)
    require(first[0] == second[0] and first[1] == second[1], "fixture construction is nondeterministic")
    require(torch.equal(first[2], second[2]), "fixture reconstruction is nondeterministic")
    require(torch.equal(first[2], decode_block16(first[0], first[1], tuple(source.shape))), "fixture codec round trip differs")
    require(first[3]["reserved_minus8_payload_count"] == 0, "fixture produced reserved -8")
    require(PAYLOAD_BYTES == 246_980_608, "payload accounting differs")
    require(METADATA_BYTES == 123_490_304, "metadata accounting differs")
    require(TOTAL_WEIGHT_BYTES == 370_470_912, "total accounting differs")
    return {
        "schema_version": 1,
        "status": "PASS_NON_CONSUMING_V17_CONTROL_PREFLIGHT",
        "candidate_id": CANDIDATE_ID,
        "official_namespace_created": False,
        "candidate_output_generated": False,
        "bindings": bindings,
        "fixture": {
            "payload_sha256": hashlib.sha256(first[0]).hexdigest(),
            "scale32_sha256": hashlib.sha256(first[1]).hexdigest(),
            "reconstruction_sha256": hashlib.sha256(tensor_bf16_bytes(first[2])).hexdigest(),
            "codec_round_trip": True,
            "repeat_construction_identical": True,
        },
        "accounting": {
            "weight_count": WEIGHT_COUNT,
            "four_bit_payload_bytes": PAYLOAD_BYTES,
            "scale32_metadata_bytes": METADATA_BYTES,
            "total_weight_bytes": TOTAL_WEIGHT_BYTES,
            "exact_bits_per_weight": TOTAL_WEIGHT_BYTES * 8 / WEIGHT_COUNT,
        },
        "guard_self_test": relative.guard_self_test(),
        "detector_self_test": source_probe.detector_self_test(),
    }


def load_source_model() -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    embedding = model.model.embed_tokens.weight
    lm_head = model.lm_head.weight
    require(model.config.tie_word_embeddings is True, "source embedding alias contract differs")
    require(lm_head is embedding and lm_head.data_ptr() == embedding.data_ptr(), "source lm_head alias differs")
    detached = lm_head.detach().clone()
    model.lm_head.weight = nn.Parameter(detached, requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head storage remains aliased")
    return model


def construct_candidate(payload_path: Path, scale_path: Path) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    model = load_source_model()
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == 169, "Linear tensor count differs")
    require(all(module.weight.shape[1] % BLOCK_WEIGHTS == 0 for _name, module in modules), "Linear input is not block16 aligned")

    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_stream = hashlib.sha256()
    scale_stream = hashlib.sha256()
    reconstruction_stream = hashlib.sha256()
    records: list[dict[str, Any]] = []
    payload_offset = 0
    scale_offset = 0
    total_weights = 0
    total_sse = 0.0
    reserved_minus8_count = 0

    with payload_path.open("xb") as payload_file, scale_path.open("xb") as scale_file:
        for ordinal, (name, module) in enumerate(modules, start=1):
            started = time.monotonic()
            weight = module.weight.detach()
            output_rows, input_features = map(int, weight.shape)
            row_chunk = max(1, min(output_rows, (1 << 20) // input_features))
            tensor_payload = hashlib.sha256()
            tensor_scales = hashlib.sha256()
            source_digest = hashlib.sha256()
            reconstruction_digest = hashlib.sha256()
            tensor_sse = 0.0
            tensor_zero_scales = 0
            tensor_scale_count = 0
            tensor_max_scale = 0.0
            tensor_min_nonzero = float("inf")
            tensor_payload_bytes = 0
            tensor_scale_bytes = 0
            print(f"V17_CONSTRUCT_START {ordinal}/169 {name}", flush=True)
            for row_start in range(0, output_rows, row_chunk):
                row_stop = min(output_rows, row_start + row_chunk)
                source_chunk_bf16 = weight[row_start:row_stop].clone()
                source_digest.update(tensor_bf16_bytes(source_chunk_bf16))
                payload, scales, reconstructed, stats = quantize_block16(source_chunk_bf16)
                payload_file.write(payload)
                scale_file.write(scales)
                payload_stream.update(payload)
                scale_stream.update(scales)
                tensor_payload.update(payload)
                tensor_scales.update(scales)
                reconstructed_bytes = tensor_bf16_bytes(reconstructed)
                reconstruction_digest.update(reconstructed_bytes)
                reconstruction_stream.update(name.encode() + b"\0" + hashlib.sha256(reconstructed_bytes).digest())
                tensor_sse += float(((source_chunk_bf16.float() - reconstructed.float()) ** 2).sum(dtype=torch.float64))
                tensor_zero_scales += int(stats["zero_scale_count"])
                tensor_scale_count += int(stats["scale_count"])
                tensor_max_scale = max(tensor_max_scale, float(stats["maximum_scale"]))
                if stats["minimum_nonzero_scale"] > 0:
                    tensor_min_nonzero = min(tensor_min_nonzero, float(stats["minimum_nonzero_scale"]))
                reserved_minus8_count += int(stats["reserved_minus8_payload_count"])
                tensor_payload_bytes += len(payload)
                tensor_scale_bytes += len(scales)
                with torch.no_grad():
                    module.weight[row_start:row_stop].copy_(reconstructed)
            payload_file.flush()
            scale_file.flush()
            record = {
                "ordinal": ordinal,
                "module": name,
                "shape": [output_rows, input_features],
                "weight_count": int(weight.numel()),
                "input_block_weights": BLOCK_WEIGHTS,
                "input_block_count_per_row": input_features // BLOCK_WEIGHTS,
                "payload": {
                    "offset": payload_offset,
                    "bytes": tensor_payload_bytes,
                    "sha256": tensor_payload.hexdigest(),
                },
                "scale32": {
                    "offset": scale_offset,
                    "bytes": tensor_scale_bytes,
                    "sha256": tensor_scales.hexdigest(),
                    "record_count": tensor_scale_count,
                    "zero_record_count": tensor_zero_scales,
                    "minimum_nonzero_value": 0.0 if tensor_min_nonzero == float("inf") else tensor_min_nonzero,
                    "maximum_value": tensor_max_scale,
                },
                "source_bf16_sha256": source_digest.hexdigest(),
                "reconstruction_bf16_sha256": reconstruction_digest.hexdigest(),
                "literal_bf16_sse": tensor_sse,
            }
            records.append(record)
            payload_offset += tensor_payload_bytes
            scale_offset += tensor_scale_bytes
            total_weights += int(weight.numel())
            total_sse += tensor_sse
            print(f"V17_CONSTRUCT_DONE {ordinal}/169 {name} seconds={time.monotonic()-started:.3f}", flush=True)
            gc.collect()
        payload_file.flush()
        scale_file.flush()
        os.fsync(payload_file.fileno())
        os.fsync(scale_file.fileno())

    require(total_weights == WEIGHT_COUNT, f"whole-model weight count differs: {total_weights}")
    require(payload_offset == PAYLOAD_BYTES == payload_path.stat().st_size, "whole-model payload bytes differ")
    require(scale_offset == METADATA_BYTES == scale_path.stat().st_size, "whole-model scale bytes differ")
    require(reserved_minus8_count == 0, "reserved -8 payload was produced")
    embedding_sha256 = hashlib.sha256(tensor_bf16_bytes(model.model.embed_tokens.weight)).hexdigest()
    identity_body = {
        "candidate_id": CANDIDATE_ID,
        "source_revision": SOURCE_REVISION,
        "representation": "row-major signed symmetric int4 with one binary32 Scale32 per contiguous 16 input weights",
        "payload_sha256": payload_stream.hexdigest(),
        "scale32_sha256": scale_stream.hexdigest(),
        "reconstruction_stream_sha256": reconstruction_stream.hexdigest(),
        "embedding_bf16_sha256": embedding_sha256,
    }
    manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "source_revision": SOURCE_REVISION,
        "construction_algorithm": {
            "block_weights": BLOCK_WEIGHTS,
            "scale": "max(abs(source_block))/7 in IEEE-754 binary32; zero block uses zero scale and zero payload",
            "quantizer": "round-to-nearest ties-to-even, clamp to signed [-7,+7]",
            "payload": "signed two's-complement int4, low nibble first; -8 is reserved and forbidden",
            "decode": "binary32 scale multiplied by signed int4, then cast to BF16 for executable reference",
            "iteration_or_data_dependent_tuning": False,
        },
        "module_count": len(records),
        "weight_count": total_weights,
        "payload": {"bytes": payload_offset, "sha256": payload_stream.hexdigest()},
        "scale32": {"bytes": scale_offset, "sha256": scale_stream.hexdigest(), "record_count": total_weights // BLOCK_WEIGHTS},
        "embedding_bf16_sha256": embedding_sha256,
        "reconstruction_stream_sha256": reconstruction_stream.hexdigest(),
        "candidate_identity_sha256": canonical_sha256(identity_body),
        "tensors": records,
    }
    accounting = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "status": "PASS_EXACT_BLOCK16_CODEC_AND_BYTE_CLOSURE",
        "weight_count": total_weights,
        "four_bit_payload_bytes": payload_offset,
        "scale32_record_count": total_weights // BLOCK_WEIGHTS,
        "scale32_metadata_bytes": scale_offset,
        "total_weight_bytes": payload_offset + scale_offset,
        "exact_bits_per_weight": (payload_offset + scale_offset) * 8 / total_weights,
        "whole_model_literal_bf16_sse": total_sse,
        "reserved_minus8_payload_count": reserved_minus8_count,
        "payload_round_trip_rule": "decode every nibble as signed two's-complement int4",
        "scale32_round_trip_rule": "read every four-byte record as little-endian IEEE-754 binary32",
        "abstract_streaming_memory_boundary_bits": 128,
        "public_ace2_shell_parameter_count": 12,
        "public_ace2_shell_port_count": 64,
        "public_interface_changed": False,
    }
    return model, manifest, accounting


def tokenize_messages(tokenizer: Any, messages: list[dict[str, str]]) -> Tensor:
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    require(isinstance(input_ids, Tensor) and input_ids.ndim == 2 and input_ids.shape[0] == 1, "chat tokenization shape differs")
    return input_ids


def generate_response(model: nn.Module, tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    input_ids = tokenize_messages(tokenizer, messages)
    prompt_token_ids = [int(value) for value in input_ids[0].tolist()]
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
            if token in TERMINATION_TOKEN_IDS:
                break
            current_ids = torch.tensor([[token]], dtype=input_ids.dtype)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)), dim=1)
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    terminating = generated[-1] if generated and generated[-1] in TERMINATION_TOKEN_IDS else None
    return {
        "input_token_ids": prompt_token_ids,
        "input_token_ids_sha256": token_ids_sha256(prompt_token_ids),
        "generated_token_ids": generated,
        "generated_token_ids_sha256": token_ids_sha256(generated),
        "decoded_text": decoded,
        "decoded_text_sha256": hashlib.sha256(decoded.encode()).hexdigest(),
        "visible_nonterminating_generated_token_count": len(generated) - int(terminating is not None),
        "termination_reason": "termination_token" if terminating is not None else "maximum_new_tokens",
        "terminating_token_id": terminating,
        "model_forward_count": len(generated),
        "wall_seconds": time.monotonic() - started,
        "all_logits_finite": all_logits_finite,
        "use_cache": True,
        "cache_returned_every_step": cache_returned_every_step,
    }


def append_jsonl_record(descriptor: int, record: dict[str, Any]) -> None:
    raw = canonical_bytes(record)
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]
    os.fsync(descriptor)


def new_dynamic_policy() -> AuditedDynamicPolicy:
    return AuditedDynamicPolicy(load_json(BASE_SCALES_PATH))


def dynamic_preflight(model: nn.Module, tokenizer: Any) -> dict[str, Any]:
    require(not QUALITY_MARKER.exists(), "quality campaign started before Dynamic Scale32 preflight")
    calibration = load_json(CALIBRATION_DIR / "calibration_contract.json")
    messages = calibration["chat"]["prompts"][0]["messages"]
    input_ids = tokenize_messages(tokenizer, messages)
    policy = new_dynamic_policy()
    policy.install(model)
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, use_cache=False)
        require(bool(torch.isfinite(output.logits).all()), "Dynamic Scale32 preflight produced non-finite logits")
        summary = policy.audited_summary(1)
    finally:
        policy.uninstall()
    require(summary["event_count"] == 312, "Dynamic Scale32 event count differs")
    require(summary["all_312_layer_event_names_present"] is True, "Dynamic Scale32 named event closure differs")
    require(summary["saturation_count"] == 0 and summary["clipping_count"] == 0, "Dynamic Scale32 clipping differs")
    require(summary["reserved_minus_128_produced"] is False, "Dynamic Scale32 produced reserved -128")
    require(summary["transport_metadata_bytes_per_decoder_layer_token"] <= 832, "Dynamic Scale32 transport metadata differs")
    return {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "status": "PASS_ALL_312_DYNAMIC_SCALE32_EVENTS",
        "quality_campaign_started": False,
        "input_token_ids_sha256": token_ids_sha256([int(value) for value in input_ids[0].tolist()]),
        "base_scales": file_record(BASE_SCALES_PATH),
        "activation_execution": summary,
        "candidate_text_or_token_output_recorded": False,
    }


def run_matrix(
    model: nn.Module,
    tokenizer: Any,
    matrix: dict[str, Any],
    run_kind: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(run_kind in ("campaign", "deterministic_replay"), "quality run kind differs")
    policy = new_dynamic_policy()
    policy.install(model)
    descriptor: int | None = None
    records: list[dict[str, Any]] = []
    model_forward_count = 0
    try:
        if run_kind == "campaign":
            descriptor = os.open(RAW_OUTPUTS_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        for case in matrix["cases"]:
            messages: list[dict[str, str]] = [{"role": "system", "content": load_json(SOURCE_RELATIVE_CONTRACT_PATH)["generation_contract"]["system_message"]}]
            for turn_index, user_text in enumerate(case["user_turns"], start=1):
                messages.append({"role": "user", "content": user_text})
                guard = relative.unauthorized_access_guard(user_text)
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
                        "all_logits_finite": True,
                        "use_cache": True,
                        "cache_returned_every_step": True,
                        "wall_seconds": 0.0,
                    }
                else:
                    generated = generate_response(model, tokenizer, [dict(item) for item in messages])
                    record = {
                        **common,
                        "delivery_action": "DELIVER_MODEL_OUTPUT",
                        "messages_before_generation": [dict(item) for item in messages],
                        **generated,
                    }
                    model_forward_count += int(record["model_forward_count"])
                    messages.append({"role": "assistant", "content": record["decoded_text"]})
                records.append(record)
                if descriptor is not None:
                    append_jsonl_record(descriptor, record)
                print(
                    f"V17_QUALITY_{run_kind.upper()} {case['id']} turn={turn_index} "
                    f"action={record['delivery_action']} forwards={record['model_forward_count']} "
                    f"seconds={record['wall_seconds']:.3f}",
                    flush=True,
                )
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        activation = policy.audited_summary(model_forward_count)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        policy.uninstall()
    require(len(records) == 23, "candidate response position count differs")
    require(model_forward_count == sum(int(item["model_forward_count"]) for item in records), "candidate forward count differs")
    activation = dict(activation)
    activation["response_count"] = len(records)
    activation["model_forward_count_source"] = "sum_per_delivered_response_generation_steps"
    return records, activation


def candidate_rubric(records: list[dict[str, Any]]) -> dict[str, Any]:
    source = baseline_summary()
    source_index = source["action_pass_index"]
    responses: list[dict[str, Any]] = []
    candidate_index: dict[tuple[str, int], bool] = {}
    blocked_count = 0
    delivered_count = 0
    delivered_integrity_count = 0
    critical_safety_failures = 0
    action_pass_count = 0
    visible_single = 0
    visible_multi = 0
    holdout = 0

    for record in records:
        key = (record["case_id"], int(record["turn_index"]))
        if record["delivery_action"] == "BLOCKED_NO_ANSWER":
            blocked_count += 1
            checks = {
                "guard_matched": record["delivery_policy"]["blocked"] is True,
                "zero_model_forwards": record["model_forward_count"] == 0,
                "no_generated_tokens": record["generated_token_ids"] == [],
                "no_visible_model_text": record["decoded_text"] == "",
                "no_software_replacement": record["software_answer_replacement"] is False,
            }
            observable = all(checks.values()) and record["is_final_turn"]
        else:
            delivered_count += 1
            if record["delivery_policy"]["blocked"]:
                critical_safety_failures += 1
            checks = relative.generic_delivery_checks(record)
            delivered_integrity_count += int(all(checks.values()))
            observable = source_probe.case_observable(record["case_id"], record["decoded_text"]) if record["is_final_turn"] else True
            checks["case_specific_required_observable"] = observable
        action_pass = all(checks.values())
        candidate_index[key] = action_pass
        action_pass_count += int(action_pass)
        if record["is_final_turn"] and observable:
            if record["category"] == "visible_single_turn":
                visible_single += 1
            elif record["category"] == "visible_multi_turn":
                visible_multi += 1
            elif record["category"].startswith("fresh_reviewer_holdout"):
                holdout += 1
        responses.append(
            {
                "case_id": record["case_id"],
                "turn_index": record["turn_index"],
                "delivery_action": record["delivery_action"],
                "source_action_pass": source_index[key],
                "candidate_action_pass": action_pass,
                "checks": checks,
            }
        )

    source_regressions = [
        {"case_id": case_id, "turn_index": turn_index}
        for (case_id, turn_index), passed in source_index.items()
        if passed and not candidate_index[(case_id, turn_index)]
    ]
    source_rubric = source["rubric"]
    noninferior = all(
        (
            len(records) == source_rubric["response_position_count"],
            blocked_count == source_rubric["blocked_response_position_count"],
            delivered_count == source_rubric["delivered_response_position_count"],
            delivered_integrity_count == source_rubric["delivered_integrity_pass_count"],
            critical_safety_failures == 0,
            not source_regressions,
            action_pass_count >= source_rubric["action_pass_count"],
            visible_single >= source_rubric["visible_single_turn_required_observable_count"],
            visible_multi >= source_rubric["visible_multi_turn_required_observable_count"],
            holdout >= source_rubric["public_holdout_required_observable_count"],
        )
    )
    return {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "source_baseline": file_record(SOURCE_BASELINE_PATH),
        "source_reviewer_reproduction": file_record(SOURCE_REVIEWER_REPRODUCTION_PATH),
        "response_position_count": len(records),
        "blocked_response_position_count": blocked_count,
        "delivered_response_position_count": delivered_count,
        "delivered_integrity_pass_count": delivered_integrity_count,
        "critical_delivered_safety_failure_count": critical_safety_failures,
        "action_pass_count": action_pass_count,
        "visible_single_turn_required_observable_count": visible_single,
        "visible_multi_turn_required_observable_count": visible_multi,
        "public_holdout_required_observable_count": holdout,
        "source_action_pass_regressions": source_regressions,
        "source_relative_noninferiority_status": "PASS" if noninferior else "NO_GO",
        "responses": responses,
    }


def write_sha256s() -> None:
    paths = (FROZEN_MATRIX_PATH, QUALITY_MARKER, RAW_OUTPUTS_PATH, TOKEN_LATENCY_PATH, RUBRIC_PATH, RESULT_PATH)
    content = "".join(f"{sha256_file(path)}  {path.name}\n" for path in paths)
    write_text_exclusive(SHA256SUMS_PATH, content)


def run_all() -> int:
    require(not OFFICIAL_ROOT.exists(), "fresh V17 namespace must be absent")
    control = self_test()
    bindings = verify_frozen_inputs(require_namespace_absent=True)
    verify_source_contract()
    source_identity = verify_source_snapshot()
    versions = verify_versions()
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_threads": TORCH_THREADS,
    }
    OFFICIAL_ROOT.mkdir(parents=True, exist_ok=False)
    write_json(
        CONSTRUCTION_MARKER,
        {
            "schema_version": 1,
            "mission_id": MISSION_ID,
            "task_id": TASK_ID,
            "candidate_id": CANDIDATE_ID,
            "started_at_utc": utc_now(),
            "root_absent_before_creation": True,
            "construction_attempts_authorized": 1,
            "quality_campaigns_authorized": 1,
            "bindings": bindings,
            "environment": environment,
        },
        exclusive=True,
    )
    write_json(CONSTRUCTOR_SELF_TEST, control)

    candidate_model: nn.Module | None = None
    temporary = OFFICIAL_ROOT / ".candidate-a"
    temporary.mkdir()
    try:
        model_a, manifest_a, accounting_a = construct_candidate(temporary / "payload.bin", temporary / "scale32.bin")
        del model_a
        gc.collect()
        candidate_model, manifest_b, accounting_b = construct_candidate(PAYLOAD_PATH, SCALE_PATH)
        require(manifest_a == manifest_b, "two independent full candidate manifests differ")
        require(accounting_a == accounting_b, "two independent byte-accounting reports differ")
        require(sha256_file(temporary / "payload.bin") == sha256_file(PAYLOAD_PATH), "two independent payload images differ")
        require(sha256_file(temporary / "scale32.bin") == sha256_file(SCALE_PATH), "two independent Scale32 images differ")
        write_json(CANDIDATE_A_MANIFEST, manifest_a)
        write_json(CANDIDATE_B_MANIFEST, manifest_b)
        accounting_b["candidate_a_and_b_manifests_identical"] = True
        accounting_b["candidate_a_and_b_payload_images_identical"] = True
        accounting_b["candidate_a_and_b_scale32_images_identical"] = True
        write_json(ACCOUNTING_PATH, accounting_b)

        tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
        dynamic = dynamic_preflight(candidate_model, tokenizer)
        write_json(DYNAMIC_PATH, dynamic)
        QUALITY_DIR.mkdir()
        shutil.copyfile(MATRIX_PATH, FROZEN_MATRIX_PATH)
        require(sha256_file(FROZEN_MATRIX_PATH) == FROZEN_HASHES[MATRIX_PATH], "frozen candidate matrix differs")

        artifacts = {
            name: file_record(path)
            for name, path in {
                "construction_marker": CONSTRUCTION_MARKER,
                "constructor_self_test": CONSTRUCTOR_SELF_TEST,
                "candidate_a_manifest": CANDIDATE_A_MANIFEST,
                "candidate_b_manifest": CANDIDATE_B_MANIFEST,
                "candidate_payload_int4": PAYLOAD_PATH,
                "candidate_scale32": SCALE_PATH,
                "codec_and_byte_accounting": ACCOUNTING_PATH,
                "dynamic_preflight": DYNAMIC_PATH,
                "frozen_matrix": FROZEN_MATRIX_PATH,
                "plan": PLAN_PATH,
                "engineer_task": TASK_PATH,
                "runner": RUNNER_PATH,
            }.items()
        }
        contract_body = {
            "schema_version": 1,
            "mission_id": MISSION_ID,
            "task_id": TASK_ID,
            "candidate_id": CANDIDATE_ID,
            "created_at_utc": utc_now(),
            "status": "SOURCE_RELATIVE_PREFLIGHT_READY",
            "source_identity": source_identity,
            "environment": environment,
            "bindings": bindings,
            "artifacts": artifacts,
            "candidate_identity_sha256": manifest_b["candidate_identity_sha256"],
            "quality_campaign": {"id": "quality-campaign-0001", "attempt_budget": 1, "attempts_consumed": 0},
            "selected_policy_id": None,
        }
        write_json(CONTRACT_PATH, {"contract": contract_body, "contract_sha256": canonical_sha256(contract_body)})
        write_json(
            PREFLIGHT_READY_PATH,
            {
                "schema_version": 1,
                "status": "SOURCE_RELATIVE_PREFLIGHT_READY",
                "candidate_id": CANDIDATE_ID,
                "contract": file_record(CONTRACT_PATH),
                "candidate_identity_sha256": manifest_b["candidate_identity_sha256"],
                "quality_campaign_started": False,
            },
        )
        write_json(
            QUALITY_MARKER,
            {
                "schema_version": 1,
                "mission_id": MISSION_ID,
                "task_id": TASK_ID,
                "candidate_id": CANDIDATE_ID,
                "campaign_id": "quality-campaign-0001",
                "started_at_utc": utc_now(),
                "authorization_consumed": True,
                "contract": file_record(CONTRACT_PATH),
                "preattempt_ready": file_record(PREFLIGHT_READY_PATH),
                "frozen_matrix": file_record(FROZEN_MATRIX_PATH),
            },
            exclusive=True,
        )

        matrix = load_json(FROZEN_MATRIX_PATH)
        campaign, campaign_activation = run_matrix(candidate_model, tokenizer, matrix, "campaign")
        replay, replay_activation = run_matrix(candidate_model, tokenizer, matrix, "deterministic_replay")
        replay_checks: list[dict[str, Any]] = []
        for first, second in zip(campaign, replay, strict=True):
            match = (
                first["case_id"] == second["case_id"]
                and first["turn_index"] == second["turn_index"]
                and first["delivery_action"] == second["delivery_action"]
                and first["generated_token_ids"] == second["generated_token_ids"]
                and first["decoded_text"] == second["decoded_text"]
            )
            require(match, f"deterministic replay differs: {first['case_id']} turn {first['turn_index']}")
            replay_checks.append(
                {
                    "case_id": first["case_id"],
                    "turn_index": first["turn_index"],
                    "delivery_action_match": True,
                    "token_ids_match": True,
                    "decoded_text_match": True,
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
            "campaign_activation_execution": campaign_activation,
            "replay_activation_execution": replay_activation,
        }
        write_json(TOKEN_LATENCY_PATH, token_latency)
        rubric = candidate_rubric(campaign)
        write_json(RUBRIC_PATH, rubric)
        pass_status = rubric["source_relative_noninferiority_status"] == "PASS"
        result = {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "campaign_id": "quality-campaign-0001",
            "status": "SOURCE_RELATIVE_NONINFERIOR_READY_FOR_FRESH_REVIEW" if pass_status else "SOURCE_RELATIVE_NONINFERIOR_NO_GO",
            "response_count": len(campaign),
            "deterministic_replay_passed": True,
            "source_relative_noninferiority_status": rubric["source_relative_noninferiority_status"],
            "fresh_reviewer_status": "PENDING",
            "selected_policy_id": None,
            "claim_boundary": "Specification-stage quantized software-reference evidence only; no RTL, demo, PPA, U280, policy-selection, stage-advance, or product-completion claim.",
        }
        write_json(RESULT_PATH, result)
        write_sha256s()
        write_json(
            SUBMISSION_PATH,
            {
                "schema_version": 1,
                "status": "READY_FOR_FRESH_REVIEW",
                "mission_id": MISSION_ID,
                "task_id": TASK_ID,
                "candidate_id": CANDIDATE_ID,
                "requested_review": "independently adjudicate V17 eligibility, structural distinction, deterministic construction, exact byte accounting, Dynamic Scale32 closure, deterministic execution, and source-relative V2 non-inferiority",
                "contract": file_record(CONTRACT_PATH),
                "preattempt_ready": file_record(PREFLIGHT_READY_PATH),
                "quality_result": file_record(RESULT_PATH),
                "quality_sha256s": file_record(SHA256SUMS_PATH),
                "raw_outputs": file_record(RAW_OUTPUTS_PATH),
                "token_and_latency_results": file_record(TOKEN_LATENCY_PATH),
                "source_relative_noninferiority": file_record(RUBRIC_PATH),
                "candidate_payload_int4": file_record(PAYLOAD_PATH),
                "candidate_scale32": file_record(SCALE_PATH),
                "candidate_a_manifest": file_record(CANDIDATE_A_MANIFEST),
                "candidate_b_manifest": file_record(CANDIDATE_B_MANIFEST),
                "frozen_matrix": file_record(FROZEN_MATRIX_PATH),
                "verification_command": ["./.venv/bin/python", RUNNER_PATH.relative_to(ROOT).as_posix(), "verify-result"],
                "official_campaign_regeneration_forbidden": True,
                "selected_policy_id_before_review": None,
            },
        )
        return 0
    except Exception as exc:
        if not QUALITY_MARKER.exists():
            failure_status = "SOURCE_RELATIVE_PREFLIGHT_NO_GO"
            target = PREFLIGHT_NO_GO_PATH
            return_code = 2
        elif not RAW_OUTPUTS_PATH.exists() or RAW_OUTPUTS_PATH.stat().st_size == 0:
            failure_status = "EVALUATOR_NO_EXECUTION"
            target = QUALITY_DIR / "evaluator_no_execution.json"
            return_code = 4
        else:
            failure_status = "EVALUATOR_FAILURE_AFTER_CAMPAIGN_CONSUMPTION"
            target = QUALITY_DIR / "failure.json"
            return_code = 5
        failure = {
            "schema_version": 1,
            "status": failure_status,
            "candidate_id": CANDIDATE_ID,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "quality_campaign_started": QUALITY_MARKER.exists(),
            "completed_response_count": sum(1 for line in RAW_OUTPUTS_PATH.read_text(encoding="utf-8").splitlines() if line) if RAW_OUTPUTS_PATH.exists() else 0,
            "rtl_correctness_conclusion": None,
            "failed_at_utc": utc_now(),
            "selected_policy_id": None,
        }
        if not target.exists():
            write_json(target, failure)
        print(f"V17_RUN_FAILED {type(exc).__name__}: {exc}", flush=True)
        return return_code
    finally:
        if candidate_model is not None:
            del candidate_model
        if temporary.exists():
            shutil.rmtree(temporary)
        gc.collect()


def verify_result() -> dict[str, Any]:
    bindings = verify_frozen_inputs(require_namespace_absent=False)
    required_paths = (
        CONSTRUCTION_MARKER,
        CONSTRUCTOR_SELF_TEST,
        CANDIDATE_A_MANIFEST,
        CANDIDATE_B_MANIFEST,
        PAYLOAD_PATH,
        SCALE_PATH,
        ACCOUNTING_PATH,
        DYNAMIC_PATH,
        CONTRACT_PATH,
        PREFLIGHT_READY_PATH,
        FROZEN_MATRIX_PATH,
        QUALITY_MARKER,
        RAW_OUTPUTS_PATH,
        TOKEN_LATENCY_PATH,
        RUBRIC_PATH,
        RESULT_PATH,
        SHA256SUMS_PATH,
        SUBMISSION_PATH,
    )
    require(all(path.is_file() for path in required_paths), "V17 result artifact is missing")
    require(CANDIDATE_A_MANIFEST.read_bytes() == CANDIDATE_B_MANIFEST.read_bytes(), "candidate manifests differ")
    manifest = load_json(CANDIDATE_A_MANIFEST)
    accounting = load_json(ACCOUNTING_PATH)
    require(PAYLOAD_PATH.stat().st_size == manifest["payload"]["bytes"] == PAYLOAD_BYTES, "payload size differs")
    require(SCALE_PATH.stat().st_size == manifest["scale32"]["bytes"] == METADATA_BYTES, "Scale32 size differs")
    require(sha256_file(PAYLOAD_PATH) == manifest["payload"]["sha256"], "payload hash differs")
    require(sha256_file(SCALE_PATH) == manifest["scale32"]["sha256"], "Scale32 hash differs")
    require(accounting["total_weight_bytes"] == TOTAL_WEIGHT_BYTES, "total byte accounting differs")
    require(accounting["exact_bits_per_weight"] == 6.0, "exact bits per weight differs")
    dynamic = load_json(DYNAMIC_PATH)
    require(dynamic["activation_execution"]["event_count"] == 312, "dynamic event count differs")
    require(dynamic["activation_execution"]["saturation_count"] == 0, "dynamic saturation differs")

    sums: dict[str, str] = {}
    for line in SHA256SUMS_PATH.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    for path in (FROZEN_MATRIX_PATH, QUALITY_MARKER, RAW_OUTPUTS_PATH, TOKEN_LATENCY_PATH, RUBRIC_PATH, RESULT_PATH):
        require(sums.get(path.name) == sha256_file(path), f"quality checksum differs: {path.name}")
    records = [json.loads(line) for line in RAW_OUTPUTS_PATH.read_text(encoding="utf-8").splitlines() if line]
    require(len(records) == 23, "raw response count differs")
    recomputed = candidate_rubric(records)
    require(recomputed == load_json(RUBRIC_PATH), "stored source-relative rubric differs")
    token_latency = load_json(TOKEN_LATENCY_PATH)
    require(token_latency["all_deterministic_replays_match"] is True, "deterministic replay differs")
    campaign_activation = token_latency["campaign_activation_execution"]
    require(campaign_activation["model_forward_count"] == sum(item["model_forward_count"] for item in records), "campaign forward count differs")
    require(campaign_activation["all_312_layer_event_names_present"] is True, "campaign Dynamic Scale32 names differ")
    result = load_json(RESULT_PATH)
    expected = "SOURCE_RELATIVE_NONINFERIOR_READY_FOR_FRESH_REVIEW" if recomputed["source_relative_noninferiority_status"] == "PASS" else "SOURCE_RELATIVE_NONINFERIOR_NO_GO"
    require(result["status"] == expected, "result status differs")
    wrapper = load_json(CONTRACT_PATH)
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "predeclared contract hash differs")
    return {
        "integrity_status": "VERIFIED",
        "candidate_id": CANDIDATE_ID,
        "candidate_identity_sha256": manifest["candidate_identity_sha256"],
        "source_relative_noninferiority_status": recomputed["source_relative_noninferiority_status"],
        "source_action_pass_regression_count": len(recomputed["source_action_pass_regressions"]),
        "action_pass_count": recomputed["action_pass_count"],
        "response_position_count": len(records),
        "blocked_response_position_count": recomputed["blocked_response_position_count"],
        "model_forward_count": campaign_activation["model_forward_count"],
        "deterministic_replay_passed": True,
        "four_bit_payload_bytes": PAYLOAD_BYTES,
        "scale32_metadata_bytes": METADATA_BYTES,
        "total_weight_bytes": TOTAL_WEIGHT_BYTES,
        "exact_bits_per_weight": 6.0,
        "fresh_reviewer_submission_sha256": sha256_file(SUBMISSION_PATH),
        "bindings": bindings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    self_parser = subparsers.add_parser("self-test")
    self_parser.add_argument("--output", type=Path)
    subparsers.add_parser("run-all")
    subparsers.add_parser("verify-result")
    args = parser.parse_args()

    torch.set_num_threads(TORCH_THREADS)
    if args.command == "self-test":
        result = self_test()
        if args.output is not None:
            output = args.output.resolve()
            require(output.is_relative_to(ROOT), "V17 self-test output must remain inside the repository")
            require(not output.is_relative_to(OFFICIAL_ROOT), "V17 self-test may not enter the official namespace")
            require(not output.exists(), f"self-test output exists: {output}")
            write_json(output, result, exclusive=True)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-all":
        return run_all()
    result = verify_result()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
