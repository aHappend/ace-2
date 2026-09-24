#!/usr/bin/env python3
"""Exact V10 packed-Scale24/per-row-selector constructor and evidence flow."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_option_b_alias_safe_packed_scale24_per_row_selector_dual_input_block_codebook_w4_grouped_dynamic_scale32_stage1 as reference
import v7_input_block_full_model_backend as v7
import v9_dual_input_block_codebook_row_group_selector_backend as v9
from ace2_quality_contracts import (
    SCALE32_ALL_ZERO_RECORD,
    SCALE32_EXPONENT_MAX,
    SCALE32_EXPONENT_MIN,
    SCALE32_SIGNIFICAND_MAX,
    SCALE32_SIGNIFICAND_MIN,
    pack_scale32,
    unpack_scale32,
)
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    IMAGE_DIR,
    SNAPSHOT,
    file_record,
    load_json,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
    write_json,
)
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM
from run_all_transformer_per32_stage1 import (
    aggregate_prompts,
    generate_reference,
    log_message,
    prompt_binding,
    sequence_disagreements,
    utc_now,
)
from run_option_b_alias_safe_post_w4_grouped_dynamic_scale32_stage1 import (
    parse_sums,
    seal_attempt,
    verify_companion as verify_checksum_companion,
    write_checksum_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = reference.CANDIDATE_ID
MISSION_ID = reference.MISSION_ID
TASK_ID = reference.TASK_ID
TASK_PATH = reference.TASK_PATH
TASK_COMPANION = reference.TASK_COMPANION
PLAN_PATH = reference.PLAN_PATH
PLAN_COMPANION = reference.PLAN_COMPANION
RUNNER_PATH = reference.RUNNER_PATH
BACKEND_PATH = Path(__file__).resolve()
OUTPUT_DIR = reference.OFFICIAL_ROOT
ATTEMPT_DIR = OUTPUT_DIR / "attempt-0001"
MARKER_PATH = reference.OFFICIAL_MARKER
PREATTEMPT_DIR = OUTPUT_DIR / "preattempt"
INPUT_MANIFEST_PATH = PREATTEMPT_DIR / "non_evaluator_inputs.json"
CANDIDATE_A_PATH = PREATTEMPT_DIR / "candidate_a_manifest.json"
CANDIDATE_B_PATH = PREATTEMPT_DIR / "candidate_b_manifest.json"
ALIAS_WITNESS_PATH = PREATTEMPT_DIR / "alias_separation_witness.json"
RECONSTRUCTION_PATH = PREATTEMPT_DIR / "model_only_v9_relative_reconstruction.json"
QUALITY_PATH = PREATTEMPT_DIR / "non_evaluator_quality.json"
DYNAMIC_PATH = PREATTEMPT_DIR / "dynamic_preflight.json"
SELF_TEST_PATH = PREATTEMPT_DIR / "runner_self_test.json"
CHRONOLOGY_PATH = PREATTEMPT_DIR / "preparation_chronology.json"
RECOVERY_PATH = PREATTEMPT_DIR / "preparation_recovery.json"
CONTRACT_PATH = OUTPUT_DIR / "predeclared_contract.json"
READY_PATH = OUTPUT_DIR / "PREATTEMPT_READY.json"
NO_GO_PATH = OUTPUT_DIR / "PREATTEMPT_NO_GO.json"

SOURCE_CONTRACT_PATH = ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
CALIBRATION_CONTRACT_PATH = CALIBRATION_DIR / "calibration_contract.json"
CALIBRATION_REPORT_PATH = CALIBRATION_DIR / "calibration_report.json"
BASE_SCALES_PATH = CALIBRATION_DIR / "derived_scales.json"
MATRIX_PATH = ROOT / "build/option-b-v2-weight-policy-20260806/full_sequence_discrimination.json"
MATRIX_SHA256 = "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
SOURCE_EMBEDDING_SHA256 = "4e96b0df6d274768cbb7e72404011853d23349999b658dc2f4dfb3c431ea223f"
V9_RECONSTRUCTION_PATH = ROOT / "build/stage1-option-b-alias-safe-dual-input-block-codebook-row-group-selector-w4-grouped-dynamic-scale32-w4a8-v9/preattempt/model_only_dual_codebook_reconstruction.json"
V9_RECONSTRUCTION_SHA256 = "f8aefa12f62396a65d0e41a8414a83cfcce30b890a5598750bda6dbd2ad13c49"
V9_AGGREGATE_BF16_SSE = 2176.5828747795154
V9_REVIEW_PATH = ROOT / "research/raw/specification/dual-input-block-codebook-row-group-selector-v9-review-done-20260806T231214Z.json"
V9_REVIEW_SHA256 = "340e38b4b58ead6ff8639c5d3fd7930b0d199631b81ac8f5953b7f973799ac29"

INPUT_BLOCK_LANES = reference.INPUT_BLOCK_LANES
CODEBOOK_ENTRIES = reference.CODEBOOK_ENTRIES
OUTER_ITERATIONS = reference.OUTER_ITERATIONS
ALIGNMENT_BYTES = reference.ALIGNMENT_BYTES
TORCH_THREADS = 32
CONSTRUCTION_TORCH_THREADS = 1
EXPECTED_EVENTS = v7.grouped.LAYERS * len(v7.grouped.EVENT_FAMILIES)
STATE_MATCH_KEYS = (
    "embedding_bf16_sha256",
    "payload_manifest_sha256",
    "packed_scale24_manifest_sha256",
    "dual_codebook_manifest_sha256",
    "selector_manifest_sha256",
    "reconstruction_manifest_sha256",
    "objective_manifest_sha256",
    "iteration_trace_manifest_sha256",
    "weight_manifest_sha256",
    "candidate_state_manifest_sha256",
)


def require(condition: bool, message: str) -> None:
    reference.require(condition, message)


def canonical_sha256(value: Any) -> str:
    return reference.canonical_sha256(value)


def atomic_write_json(path: Path, value: Any) -> None:
    raw = reference.canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _configure_quality_helpers() -> None:
    v7.v3.CANDIDATE_ID = CANDIDATE_ID
    v7.v3.OUTPUT_DIR = OUTPUT_DIR
    v7.v3.PREATTEMPT_DIR = PREATTEMPT_DIR
    v7.v3.ATTEMPT_DIR = ATTEMPT_DIR
    v7.v3.MARKER_PATH = MARKER_PATH
    v7.v3.TORCH_THREADS = TORCH_THREADS


def _pack_scale24_record(record: int) -> bytes:
    unpack_scale32(record)
    raw = int(record).to_bytes(4, "little", signed=False)
    require(raw[3] == 0, "Scale32 reserved high byte is nonzero")
    return raw[:3]


def _unpack_scale24_record(raw: bytes) -> int:
    require(len(raw) == 3, "packed Scale24 record must contain exactly three bytes")
    record = int.from_bytes(raw + b"\x00", "little", signed=False)
    unpack_scale32(record)
    require(_pack_scale24_record(record) == raw, "packed Scale24 record does not round-trip")
    return record


def _pack_scale24_stream(records: Tensor) -> bytes:
    require(records.ndim == 1, "packed Scale24 source must be a one-dimensional record stream")
    return b"".join(_pack_scale24_record(int(record)) for record in records.tolist())


def _unpack_scale24_stream(raw: bytes, record_count: int) -> Tensor:
    require(record_count >= 0 and len(raw) == record_count * 3, "packed Scale24 stream length differs")
    return torch.tensor(
        [_unpack_scale24_record(raw[offset : offset + 3]) for offset in range(0, len(raw), 3)],
        dtype=torch.int64,
    )


def _row_ranges(rows: int) -> tuple[tuple[int, int], ...]:
    require(rows > 0, "V10 tensor has no output rows")
    return tuple((row, row + 1) for row in range(rows))


def construct_packed_scale24_state(source: Tensor, module_name: str) -> dict[str, Any]:
    require(source.ndim == 2 and source.dtype == torch.bfloat16, f"source weight format differs: {module_name}")
    rows, width = (int(value) for value in source.shape)
    blocks = reference.input_block_ranges(width)
    fresh_v9 = v9.construct_dual_codebook_state(source, module_name)
    predecessor_group_size = int(fresh_v9["row_group_size"])
    records = fresh_v9["final_records"].clone()
    assignments = fresh_v9["final_assignments"].clone()
    codebooks = tuple((tuple(banks[0]), tuple(banks[1])) for banks in fresh_v9["final_codebooks"])
    predecessor_selectors = fresh_v9["final_selectors"]
    row_indices = torch.arange(rows, dtype=torch.int64)
    selectors = predecessor_selectors[torch.div(row_indices, predecessor_group_size, rounding_mode="floor")].clone()
    require(selectors.shape == (rows, len(blocks)), f"V10 initial selector geometry differs: {module_name}")

    packed_scale24 = _pack_scale24_stream(records)
    require(torch.equal(_unpack_scale24_stream(packed_scale24, rows), records), f"V10 initial packed scale round-trip differs: {module_name}")
    fresh_v9_digest = hashlib.sha256()
    initial_digest = hashlib.sha256()
    for row_start, row_stop in v7.v4._row_chunks(rows, width):
        predecessor_reconstruction = v9._decoded_chunk_v9(
            fresh_v9["final_records"],
            fresh_v9["final_assignments"],
            fresh_v9["final_codebooks"],
            fresh_v9["final_selectors"],
            predecessor_group_size,
            row_start,
            row_stop,
        )
        initial_reconstruction = v9._decoded_chunk_v9(
            records, assignments, codebooks, selectors, 1, row_start, row_stop
        )
        require(torch.equal(initial_reconstruction, predecessor_reconstruction), f"V10 initialization is not byte-identical to V9: {module_name}")
        fresh_v9_digest.update(v7.v3.raw_tensor_bytes(predecessor_reconstruction))
        initial_digest.update(v7.v3.raw_tensor_bytes(initial_reconstruction))
    initial_v9_sha256 = initial_digest.hexdigest()
    require(initial_v9_sha256 == fresh_v9_digest.hexdigest(), f"V10 initial V9 hash differs: {module_name}")
    require(v7.v3.pack_codebook_indices(assignments) == fresh_v9["payload"], f"V10 initial payload differs from V9: {module_name}")
    require(v9._dual_codebook_raw(codebooks) == fresh_v9["codebook_raw"], f"V10 initial codebooks differ from V9: {module_name}")

    objective_exponent = int(fresh_v9["final_objective"]["binary_exponent"])
    previous_objective = v9._objective(
        source,
        records,
        assignments,
        codebooks,
        selectors,
        1,
        fresh_v9["fresh_v7_state"]["shift_min"],
        objective_exponent,
    )
    require(previous_objective == int(fresh_v9["final_objective"]["scaled_integer"]), f"V10 expanded V9 objective differs: {module_name}")
    trace: list[dict[str, Any]] = []
    for iteration in range(1, OUTER_ITERATIONS + 1):
        selectors, selector_objective, selector_summary = v9._assign_selectors(
            source,
            records,
            assignments,
            codebooks,
            1,
            fresh_v9["fresh_v7_state"]["shift_min"],
            objective_exponent,
        )
        require(selector_objective <= previous_objective, f"V10 selector assignment increased exact source SSE: {module_name}")
        assignments, payload_statistics = v9._assign_payload(
            source,
            records,
            codebooks,
            selectors,
            1,
            fresh_v9["fresh_v7_state"]["shift_min"],
            fresh_v9["fresh_v7_state"]["shift_max"],
        )
        payload_objective = v9._objective_from_statistics(
            records, payload_statistics, fresh_v9["fresh_v7_state"]["shift_min"], objective_exponent
        )
        require(payload_objective <= selector_objective, f"V10 payload assignment increased exact source SSE: {module_name}")
        records, row_summary = v9._update_row_records(
            source,
            assignments,
            codebooks,
            selectors,
            1,
            records,
            fresh_v9["fresh_v7_state"]["shift_min"],
            objective_exponent,
            payload_statistics,
        )
        assignments, row_statistics = v9._assign_payload(
            source,
            records,
            codebooks,
            selectors,
            1,
            fresh_v9["fresh_v7_state"]["shift_min"],
            fresh_v9["fresh_v7_state"]["shift_max"],
        )
        row_objective = v9._objective_from_statistics(
            records, row_statistics, fresh_v9["fresh_v7_state"]["shift_min"], objective_exponent
        )
        require(row_objective <= payload_objective, f"V10 row-scale step increased exact source SSE: {module_name}")
        codebooks, codebook_summary = v9._update_codebooks(
            source,
            records,
            assignments,
            codebooks,
            selectors,
            1,
            fresh_v9["fresh_v7_state"]["shift_min"],
            fresh_v9["fresh_v7_state"]["shift_max"],
            objective_exponent,
        )
        assignments, final_statistics = v9._assign_payload(
            source,
            records,
            codebooks,
            selectors,
            1,
            fresh_v9["fresh_v7_state"]["shift_min"],
            fresh_v9["fresh_v7_state"]["shift_max"],
        )
        final_objective = v9._objective_from_statistics(
            records, final_statistics, fresh_v9["fresh_v7_state"]["shift_min"], objective_exponent
        )
        require(final_objective <= row_objective, f"V10 codebook step increased exact source SSE: {module_name}")
        require(final_objective <= previous_objective, f"V10 outer iteration increased exact source SSE: {module_name}")
        payload = v7.v3.pack_codebook_indices(assignments)
        packed_scale24 = _pack_scale24_stream(records)
        codebook_raw = v9._dual_codebook_raw(codebooks)
        selector_raw = v9._pack_selectors(selectors)
        trace.append(
            {
                "iteration": iteration,
                "prior_objective": v7.v4._objective_record(previous_objective, objective_exponent),
                "after_selector_objective": v7.v4._objective_record(selector_objective, objective_exponent),
                "after_payload_objective": v7.v4._objective_record(payload_objective, objective_exponent),
                "after_row_scale_objective": v7.v4._objective_record(row_objective, objective_exponent),
                "final_objective": v7.v4._objective_record(final_objective, objective_exponent),
                "selector_assignment": selector_summary,
                "row_scale_update": row_summary,
                "dual_codebook_updates": codebook_summary,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "packed_scale24_stream_sha256": hashlib.sha256(packed_scale24).hexdigest(),
                "dual_codebook_stream_sha256": hashlib.sha256(codebook_raw).hexdigest(),
                "selector_stream_sha256": hashlib.sha256(selector_raw).hexdigest(),
            }
        )
        previous_objective = final_objective

    payload = v7.v3.pack_codebook_indices(assignments)
    packed_scale24 = _pack_scale24_stream(records)
    codebook_raw = v9._dual_codebook_raw(codebooks)
    selector_raw = v9._pack_selectors(selectors)
    return {
        "module_name": module_name,
        "shape": [rows, width],
        "input_block_lanes": INPUT_BLOCK_LANES,
        "input_block_count": len(blocks),
        "row_group_size": 1,
        "row_group_count": rows,
        "fresh_v9_state": fresh_v9,
        "fresh_v9_exact_reproduction": True,
        "fresh_v9_predecessor_group_size": predecessor_group_size,
        "initial_v9_objective": fresh_v9["final_objective"],
        "initial_v9_reconstruction_bf16_sha256": initial_v9_sha256,
        "outer_iterations_completed": len(trace),
        "final_records": records,
        "final_assignments": assignments,
        "final_codebooks": codebooks,
        "final_selectors": selectors,
        "payload": payload,
        "packed_scale24": packed_scale24,
        "codebook_raw": codebook_raw,
        "selector_raw": selector_raw,
        "final_objective": v7.v4._objective_record(previous_objective, objective_exponent),
        "iteration_trace": trace,
        "iteration_trace_sha256": canonical_sha256(trace),
    }


def _decode(state: dict[str, Any]) -> Tensor:
    records = _unpack_scale24_stream(state["packed_scale24"], int(state["shape"][0]))
    require(torch.equal(records, state["final_records"]), "V10 decoded packed scales differ from state records")
    return v9._decode(records, state["final_assignments"], state["final_codebooks"], state["final_selectors"], 1)


def _state_hashes(state: dict[str, Any]) -> dict[str, str]:
    reconstruction = _decode(state)
    return {
        "payload_sha256": hashlib.sha256(state["payload"]).hexdigest(),
        "packed_scale24_stream_sha256": hashlib.sha256(state["packed_scale24"]).hexdigest(),
        "dual_codebook_stream_sha256": hashlib.sha256(state["codebook_raw"]).hexdigest(),
        "selector_stream_sha256": hashlib.sha256(state["selector_raw"]).hexdigest(),
        "reconstruction_bf16_sha256": v7.v3.tensor_sha256(reconstruction),
    }


def _validate_state(state: dict[str, Any]) -> dict[str, Any]:
    rows, width = state["shape"]
    blocks = reference.input_block_ranges(width)
    codebooks = v9._validate_codebooks(state["final_codebooks"], len(blocks))
    require(state["row_group_size"] == 1 and state["row_group_count"] == rows, "V10 selector row geometry differs")
    require(state["final_assignments"].shape == (rows, width), "V10 payload geometry differs")
    require(bool(torch.all(state["final_assignments"] < CODEBOOK_ENTRIES)), "V10 payload escaped four bits")
    require(state["final_selectors"].shape == (rows, len(blocks)), "V10 selector geometry differs")
    require(torch.equal(v9._unpack_selectors(state["selector_raw"], rows, len(blocks)), state["final_selectors"]), "V10 selector packing round-trip differs")
    require(torch.equal(_unpack_scale24_stream(state["packed_scale24"], rows), state["final_records"]), "V10 packed Scale24 round-trip differs")
    require(len(state["payload"]) == rows * width // 2, "V10 payload byte count differs")
    require(len(state["packed_scale24"]) == rows * 3, "V10 packed Scale24 byte count differs")
    require(len(state["codebook_raw"]) == len(blocks) * 2 * CODEBOOK_ENTRIES, "V10 dual-codebook byte count differs")
    require(len(state["selector_raw"]) % ALIGNMENT_BYTES == 0, "V10 selector stream is not 16-byte aligned")
    return {
        "rows": rows,
        "input_features": width,
        "input_block_count": len(blocks),
        "row_group_size": 1,
        "row_group_count": rows,
        "selector_bit_count": rows * len(blocks),
        "selector_aligned_bytes": len(state["selector_raw"]),
        "packed_scale24_bytes": len(state["packed_scale24"]),
        "dual_codebook_bytes": len(state["codebook_raw"]),
        "all_codebooks_strictly_increasing": all(
            all(left < right for left, right in zip(bank, bank[1:])) for banks in codebooks for bank in banks
        ),
    }


def combined_self_test() -> dict[str, Any]:
    environment = {
        **reference.require_project_python(),
        "platform": platform.platform(),
        "packages": {"torch": importlib.metadata.version("torch")},
        "backend": "exact BF16 dyadic V9-relative packed-Scale24 per-row-selector constructor",
    }
    reference.verify_companion(TASK_PATH, TASK_COMPANION)
    reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    task = load_json(TASK_PATH)
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "frozen V10 Engineer authority differs")
    root_existed_before = OUTPUT_DIR.exists()
    marker_existed_before = MARKER_PATH.exists()
    require(not ATTEMPT_DIR.exists() and not marker_existed_before, "V10 official attempt must remain absent during constructor self-test")

    extrema = (
        pack_scale32(SCALE32_SIGNIFICAND_MIN, SCALE32_EXPONENT_MIN),
        pack_scale32(SCALE32_SIGNIFICAND_MAX, SCALE32_EXPONENT_MAX),
        SCALE32_ALL_ZERO_RECORD,
    )
    encoded_extrema = [_pack_scale24_record(record) for record in extrema]
    require([_unpack_scale24_record(raw) for raw in encoded_extrema] == list(extrema), "packed Scale24 legal extrema do not round-trip")
    require(encoded_extrema[0] == b"\x00\x80\xe8", "packed Scale24 byte order differs")
    reserved_rejected = reference.expect_rejected(lambda: _pack_scale24_record(extrema[0] | 0x01000000), "nonzero Scale32 reserved byte was accepted")
    truncated_rejected = reference.expect_rejected(lambda: _unpack_scale24_record(encoded_extrema[0][:2]), "truncated packed Scale24 record was accepted")
    extended_rejected = reference.expect_rejected(lambda: _unpack_scale24_record(encoded_extrema[0] + b"\x00"), "extended packed Scale24 record was accepted")
    reorder_probe = _pack_scale24_record(pack_scale32(0x8020, SCALE32_EXPONENT_MIN))
    reordered_rejected = reference.expect_rejected(
        lambda: _unpack_scale24_record(bytes(reversed(reorder_probe))),
        "structurally illegal reordered packed Scale24 record was accepted",
    )
    order_records = (
        _pack_scale24_record(pack_scale32(0x8020, SCALE32_EXPONENT_MIN)),
        _pack_scale24_record(pack_scale32(0x9001, -7)),
        _pack_scale24_record(pack_scale32(0xFFFE, SCALE32_EXPONENT_MAX)),
    )
    ordered_stream = b"".join(order_records)
    reordered_stream = b"".join(reversed(order_records))
    stream_order_hash_rejected = hashlib.sha256(ordered_stream).digest() != hashlib.sha256(reordered_stream).digest()
    require(stream_order_hash_rejected, "packed Scale24 stream-order authentication did not detect reordering")

    require(len(reference.input_block_ranges(896)) == 7 and len(reference.input_block_ranges(4864)) == 38, "V10 input-block geometry differs")
    illegal_width_rejected = reference.expect_rejected(lambda: reference.input_block_ranges(1024), "illegal V10 width was accepted")
    require(all(reference.row_group_size(name) == 1 for name in ("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj", "lm_head")), "V10 row-group size differs")

    lanes = torch.arange(896, dtype=torch.int64)
    row0 = ((lanes % 47) - 23).to(torch.float64) / 256.0
    row1 = torch.where((lanes % 5) == 0, row0 * 0.25, row0 * 1.5)
    row2 = torch.where((lanes % 7) < 3, -row0 * 0.75, row0 * 0.125)
    row3 = torch.zeros_like(row0)
    fixture = torch.stack((row0, row1, row2, row3)).to(torch.bfloat16)
    candidate_a = construct_packed_scale24_state(fixture, "self_test.down_proj")
    candidate_b = construct_packed_scale24_state(fixture, "self_test.down_proj")
    geometry = _validate_state(candidate_a)
    require(candidate_a["fresh_v9_predecessor_group_size"] == 2, "V10 fixture did not exercise V9 grouped-selector expansion")
    require(_state_hashes(candidate_a) == _state_hashes(candidate_b), "independent V10 fixture state hashes differ")
    require(candidate_a["iteration_trace"] == candidate_b["iteration_trace"], "independent V10 fixture traces differ")
    require(candidate_a["fresh_v9_exact_reproduction"] is True, "V10 fixture did not reproduce fresh V9")
    require(candidate_a["outer_iterations_completed"] == OUTER_ITERATIONS, "V10 fixture iteration count differs")
    require(int(candidate_a["final_objective"]["scaled_integer"]) <= int(candidate_a["initial_v9_objective"]["scaled_integer"]), "V10 fixture regressed against fresh V9")

    synthetic_selectors = torch.tensor([[0, 1, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0]], dtype=torch.uint8)
    synthetic_raw = v9._pack_selectors(synthetic_selectors)
    require(torch.equal(v9._unpack_selectors(synthetic_raw, 2, 7), synthetic_selectors), "V10 synthetic selector round-trip differs")
    tampered_padding = bytearray(synthetic_raw)
    tampered_padding[-1] |= 0x80
    padding_rejected = reference.expect_rejected(lambda: v9._unpack_selectors(bytes(tampered_padding), 2, 7), "nonzero V10 selector padding was accepted")
    require(synthetic_raw[0] == 0b11010110, "V10 selector bit order is not row-major/block-major LSB-first")

    empty_bank_selectors = torch.zeros_like(candidate_a["final_selectors"])
    retained_codebooks, empty_summary = v9._update_codebooks(
        fixture,
        candidate_a["final_records"],
        candidate_a["final_assignments"],
        candidate_a["final_codebooks"],
        empty_bank_selectors,
        1,
        candidate_a["fresh_v9_state"]["fresh_v7_state"]["shift_min"],
        candidate_a["fresh_v9_state"]["fresh_v7_state"]["shift_max"],
        int(candidate_a["fresh_v9_state"]["fresh_v7_state"]["final_objective"]["binary_exponent"]),
    )
    require(all(retained_codebooks[index][1] == candidate_a["final_codebooks"][index][1] for index in range(len(retained_codebooks))), "V10 empty bank did not retain its codebook")
    require(all(block["banks"][1]["empty_bank_retained"] for block in empty_summary), "V10 empty-bank summary differs")

    build_root = ROOT / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v10-hash-rejection-", dir=build_root) as temporary_directory:
        temporary_root = Path(temporary_directory)
        copied_artifact = temporary_root / "copied_task.json"
        copied_companion = temporary_root / "copied_task.sha256"
        copied_artifact.write_bytes(TASK_PATH.read_bytes())
        copied_companion.write_text(f"{reference.sha256_file(copied_artifact)}  {copied_artifact.relative_to(ROOT).as_posix()}\n", encoding="utf-8")
        reference.verify_companion(copied_artifact, copied_companion)
        copied_artifact.write_bytes(copied_artifact.read_bytes() + b" ")
        hash_mismatch_rejected = reference.expect_rejected(lambda: reference.verify_companion(copied_artifact, copied_companion), "tampered copied V10 artifact passed companion verification")

    require(OUTPUT_DIR.exists() == root_existed_before, "self-test changed the V10 namespace")
    require(MARKER_PATH.exists() == marker_existed_before is False, "self-test changed the V10 execution marker")
    reconstruction = _decode(candidate_a)
    fixture_sse = float(torch.sum((fixture.to(torch.float64) - reconstruction.to(torch.float64)) ** 2))
    return {
        "schema_version": 1,
        "status": "PASS",
        "candidate_id": CANDIDATE_ID,
        "mission_id": MISSION_ID,
        "task_id": TASK_ID,
        "environment": environment,
        "runner_sha256": reference.sha256_file(RUNNER_PATH),
        "backend_sha256": reference.sha256_file(BACKEND_PATH),
        "task_sha256": reference.sha256_file(TASK_PATH),
        "plan_sha256": reference.sha256_file(PLAN_PATH),
        "checks": {
            "packed_scale24_legal_extrema_and_all_zero_round_trip": True,
            "packed_scale24_little_endian": True,
            "reserved_high_byte_rejected": reserved_rejected,
            "truncated_record_rejected": truncated_rejected,
            "extended_record_rejected": extended_rejected,
            "reordered_record_rejected": reordered_rejected,
            "reordered_stream_hash_rejected": stream_order_hash_rejected,
            "legal_width_block_counts": {"896": 7, "4864": 38},
            "illegal_width_rejected": illegal_width_rejected,
            "per_row_selector_geometry": True,
            "fresh_v9_payload_scale_codebook_selector_and_reconstruction_reproduction": True,
            "independent_candidate_a_b_state_and_trace_match": True,
            "outer_iterations_completed": OUTER_ITERATIONS,
            "monotonic_exact_source_objective": True,
            "non_regression_against_fresh_v9_fixture": True,
            "dual_codebooks_strictly_increasing": geometry["all_codebooks_strictly_increasing"],
            "selector_row_major_block_major_lsb_first": True,
            "selector_zero_padding": True,
            "nonzero_selector_padding_rejected": padding_rejected,
            "payload_nibble_packing_reused_from_reviewed_v9": True,
            "scale32_bracketing_neighbors_ties_range_and_zero_rows_reused_from_reviewed_v9": True,
            "empty_bank_retains_prior_codebook": True,
            "bound_hash_mismatch_rejected": hash_mismatch_rejected,
        },
        "fixture": {
            "shape": list(fixture.shape),
            "geometry": geometry,
            "bf16_reconstruction_sse": fixture_sse,
            "initial_v9_objective": candidate_a["initial_v9_objective"],
            "final_v10_objective": candidate_a["final_objective"],
            "iteration_trace_sha256": candidate_a["iteration_trace_sha256"],
            **_state_hashes(candidate_a),
        },
        "official_build_namespace_changed": False,
        "official_execution_marker_exists": False,
        "claim_boundary": "constructor kernel and synthetic fixture self-test only; no 169-tensor, quality, Dynamic Scale32, official-attempt, RTL, or PPA conclusion",
    }


def _measure_and_materialize(
    module: nn.Linear,
    source: Tensor,
    state: dict[str, Any],
    sealed_v9_record: dict[str, Any],
) -> dict[str, Any]:
    rows, width = source.shape
    fresh_v9 = state["fresh_v9_state"]
    fresh_v9_sse = 0.0
    v10_sse = 0.0
    fresh_v9_digest = hashlib.sha256()
    v10_digest = hashlib.sha256()
    decoded_records = _unpack_scale24_stream(state["packed_scale24"], int(rows))
    with torch.no_grad():
        for row_start, row_stop in v7.v4._row_chunks(int(rows), int(width)):
            decoded_v9 = v9._decoded_chunk_v9(
                fresh_v9["final_records"],
                fresh_v9["final_assignments"],
                fresh_v9["final_codebooks"],
                fresh_v9["final_selectors"],
                fresh_v9["row_group_size"],
                row_start,
                row_stop,
            )
            decoded_v10 = v9._decoded_chunk_v9(
                decoded_records,
                state["final_assignments"],
                state["final_codebooks"],
                state["final_selectors"],
                1,
                row_start,
                row_stop,
            )
            source64 = source[row_start:row_stop].to(torch.float64)
            v9_difference = source64 - decoded_v9.to(torch.float64)
            v10_difference = source64 - decoded_v10.to(torch.float64)
            fresh_v9_sse += float(torch.sum(v9_difference * v9_difference))
            v10_sse += float(torch.sum(v10_difference * v10_difference))
            fresh_v9_digest.update(v7.v3.raw_tensor_bytes(decoded_v9))
            v10_digest.update(v7.v3.raw_tensor_bytes(decoded_v10))
            module.weight[row_start:row_stop].copy_(decoded_v10)
    fresh_v9_sha = fresh_v9_digest.hexdigest()
    v10_sha = v10_digest.hexdigest()
    sealed_v9_sse = float(sealed_v9_record["v9_dual_codebook_bf16_sse"])
    require(fresh_v9_sse >= 0.0 and v10_sse >= 0.0, f"negative BF16 reconstruction SSE: {sealed_v9_record['module']}")
    require(fresh_v9_sha == sealed_v9_record["reconstruction_bf16_sha256"], f"fresh V9 reconstruction hash differs: {sealed_v9_record['module']}")
    require(math.isclose(fresh_v9_sse, sealed_v9_sse, rel_tol=1e-15, abs_tol=1e-12), f"fresh V9 reconstruction SSE differs: {sealed_v9_record['module']}")
    require(state["initial_v9_reconstruction_bf16_sha256"] == fresh_v9_sha, f"V10 initial V9 reproduction differs: {sealed_v9_record['module']}")
    require(v10_sha == v7.v3.tensor_sha256(module.weight), f"V10 materialized reconstruction hash differs: {sealed_v9_record['module']}")
    return {
        "module": sealed_v9_record["module"],
        "weight_count": int(source.numel()),
        "fresh_v9_bf16_sse": sealed_v9_sse,
        "fresh_v9_remeasured_bf16_sse": fresh_v9_sse,
        "fresh_v9_remeasurement_delta": fresh_v9_sse - sealed_v9_sse,
        "fresh_v9_matches_sealed": True,
        "fresh_v9_reconstruction_bf16_sha256": fresh_v9_sha,
        "v10_packed_scale24_per_row_selector_bf16_sse": v10_sse,
        "v10_non_regressing_against_fresh_v9": v10_sse <= fresh_v9_sse,
        "v10_strictly_improved_against_fresh_v9": v10_sse < fresh_v9_sse,
        "reconstruction_bf16_sha256": v10_sha,
    }


def _policy_and_format() -> dict[str, Any]:
    body = {
        "payload": {
            "bits_per_weight": 4,
            "semantics": "unsigned selected-bank codebook index 0 through 15",
            "packing": "row-major; even flat element low nibble; odd flat element high nibble",
        },
        "row_scale": {
            "mathematical_format": "Scale32",
            "stored_format": "packed_scale24",
            "bytes_per_output_row": 3,
            "byte_order": "significand-low, significand-high, signed-exponent",
            "decode": "append one zero reserved high byte before Scale32 validation and arithmetic",
            "lossless": True,
        },
        "dual_codebook": {
            "banks_per_input_block": 2,
            "input_block_lanes": INPUT_BLOCK_LANES,
            "entries_per_bank": CODEBOOK_ENTRIES,
            "bytes_per_input_block": 32,
            "order": "bank A then bank B for each increasing input block",
            "format": "strictly increasing signed Q4.4 int8",
        },
        "selector": {
            "bits_per_output_row_per_input_block": 1,
            "packing": "output-row-major then input-block-major, least-significant-bit first, 16-byte tensor alignment, zero padding",
            "row_group_sizes": reference.ROW_GROUP_SIZE_BY_FAMILY,
        },
        "construction": {
            "initialization": "fresh V9 payload, exact Scale32, dual codebooks, and predecessor selectors repeated over each former row group",
            "outer_iterations": OUTER_ITERATIONS,
            "optimizer_cost": "constant-elided exact dyadic linear-plus-quadratic ordering cost",
            "reported_model_gate": "literal nonnegative BF16-materialized reconstruction SSE",
        },
        "activation": {
            "policy": "grouped Dynamic Scale32",
            "event_count": EXPECTED_EVENTS,
            "maximum_transport_metadata_bytes_per_decoder_layer_token": 832,
        },
        "storage": {
            "whole_model_total_weight_bytes": 248892848,
            "whole_model_exact_bits_per_weight": 4.030969880841819,
            "maximum_total_stored_bits_per_weight": 4.036,
        },
        "predecessor_v9_backend_sha256": reference.sha256_file(Path(v9.__file__).resolve()),
        "native_exact_assignment_source_sha256": reference.sha256_file(v9.NATIVE_SOURCE_PATH),
    }
    return {**body, "policy_and_format_sha256": canonical_sha256(body)}


def _quantize_model(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {record["module"]: record for record in v7.grouped.expected_transformer_tensors()}
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == 169, "Qwen linear count differs")
    require([name for name, _module in modules if name != "lm_head"] == list(expected), "transformer linear order differs")
    require(reference.sha256_file(V9_RECONSTRUCTION_PATH) == V9_RECONSTRUCTION_SHA256, "sealed V9 reconstruction hash differs")
    sealed_report = load_json(V9_RECONSTRUCTION_PATH)
    sealed_records = {item["module"]: item for item in sealed_report["tensors"]}
    require(list(sealed_records) == [name for name, _module in modules], "sealed V9 tensor order differs")

    manifest_names = (
        "payload",
        "packed_scale24",
        "dual_codebook",
        "selector",
        "reconstruction",
        "objective",
        "iteration_trace",
    )
    manifests: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest_names}
    weight_manifest: list[dict[str, Any]] = []
    streams = {name: hashlib.sha256() for name in ("payload", "packed_scale24", "dual_codebook", "selector", "reconstruction")}
    totals = {"weight_count": 0, "payload": 0, "packed_scale24": 0, "dual_codebook": 0, "selector": 0}

    for ordinal, (name, module) in enumerate(modules, start=1):
        started = time.monotonic()
        print(f"V10_CONSTRUCT_START {ordinal}/169 {name}", flush=True)
        source = module.weight.detach()
        source_sha = v7.v3.tensor_sha256(source)
        state = construct_packed_scale24_state(source, name)
        geometry = _validate_state(state)
        reconstruction = _measure_and_materialize(module, source, state, sealed_records[name])
        raw = {
            "payload": state["payload"],
            "packed_scale24": state["packed_scale24"],
            "dual_codebook": state["codebook_raw"],
            "selector": state["selector_raw"],
        }
        component_records = {
            component: {"module": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            for component, value in raw.items()
        }
        component_records["packed_scale24"].update(
            {
                "record_count": int(source.shape[0]),
                "bytes_per_record": 3,
                "reserved_high_byte_restored_zero": True,
            }
        )
        component_records["dual_codebook"].update(
            {
                "input_block_count": state["input_block_count"],
                "bank_count": state["input_block_count"] * 2,
                "q4_4_banks_by_input_block": [[list(bank) for bank in banks] for banks in state["final_codebooks"]],
            }
        )
        component_records["selector"].update(
            {
                "row_group_size": 1,
                "output_row_count": int(source.shape[0]),
                "selector_bit_count": int(source.shape[0]) * state["input_block_count"],
                "alignment_bytes": ALIGNMENT_BYTES,
                "padding_bits_zero": True,
            }
        )
        for component in ("payload", "packed_scale24", "dual_codebook", "selector"):
            manifests[component].append(component_records[component])
            streams[component].update(name.encode() + b"\0" + raw[component])
            totals[component] += len(raw[component])
        streams["reconstruction"].update(name.encode() + b"\0" + bytes.fromhex(reconstruction["reconstruction_bf16_sha256"]))
        manifests["reconstruction"].append(reconstruction)
        objective_record = {
            "module": name,
            "kind": "constant_elided_optimizer_ordering_cost_not_literal_sse",
            "initial_v9_optimizer_cost": state["initial_v9_objective"],
            "final_v10_optimizer_cost": state["final_objective"],
            "actual_nonnegative_bf16_sse": {
                "fresh_v9": reconstruction["fresh_v9_bf16_sse"],
                "v10": reconstruction["v10_packed_scale24_per_row_selector_bf16_sse"],
            },
        }
        manifests["objective"].append(objective_record)
        trace_record = {
            "module": name,
            "source_bf16_sha256": source_sha,
            "shift_min": state["fresh_v9_state"]["fresh_v7_state"]["shift_min"],
            "shift_max": state["fresh_v9_state"]["fresh_v7_state"]["shift_max"],
            "fresh_v9_predecessor_group_size": state["fresh_v9_predecessor_group_size"],
            "row_group_size": 1,
            "input_block_count": state["input_block_count"],
            "fresh_v9_exact_reproduction": state["fresh_v9_exact_reproduction"],
            "outer_iterations_completed": state["outer_iterations_completed"],
            "iteration_trace": state["iteration_trace"],
            "iteration_trace_sha256": state["iteration_trace_sha256"],
        }
        manifests["iteration_trace"].append(trace_record)
        record = {
            "module": name,
            "name": f"{name}.weight",
            "shape": list(source.shape),
            "scope": "separated_lm_head" if name == "lm_head" else "transformer",
            "geometry": geometry,
            "components": component_records,
            "reconstruction": reconstruction,
            "objective": objective_record,
            "iteration_trace_sha256": state["iteration_trace_sha256"],
            "metadata_sequence": ["dual_codebook_bank_a", "dual_codebook_bank_b", "selector", "payload", "packed_scale24"],
        }
        weight_manifest.append(record)
        totals["weight_count"] += int(source.numel())
        print(
            f"V10_CONSTRUCT_DONE {ordinal}/169 {name} seconds={time.monotonic()-started:.3f} "
            f"v9_sse={reconstruction['fresh_v9_bf16_sse']:.12g} v10_sse={reconstruction['v10_packed_scale24_per_row_selector_bf16_sse']:.12g}",
            flush=True,
        )
        del state
        gc.collect()

    aggregate_v9 = float(sum(item["fresh_v9_bf16_sse"] for item in manifests["reconstruction"]))
    aggregate_v10 = float(sum(item["v10_packed_scale24_per_row_selector_bf16_sse"] for item in manifests["reconstruction"]))
    fresh_count = sum(bool(item["fresh_v9_matches_sealed"]) for item in manifests["reconstruction"])
    non_regressing_count = sum(bool(item["v10_non_regressing_against_fresh_v9"]) for item in manifests["reconstruction"])
    improved_count = sum(bool(item["v10_strictly_improved_against_fresh_v9"]) for item in manifests["reconstruction"])
    require(aggregate_v9 == V9_AGGREGATE_BF16_SSE, "fresh aggregate V9 BF16 SSE differs")
    require(all(item["fresh_v9_bf16_sse"] >= 0.0 and item["v10_packed_scale24_per_row_selector_bf16_sse"] >= 0.0 for item in manifests["reconstruction"]), "V10 report contains negative BF16 SSE")
    reconstruction_body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "measurement": "literal nonnegative sum-squared error from pinned BF16 source to BF16-materialized reconstruction",
        "optimizer_cost_note": "scaled_integer optimizer fields omit the source-squared constant and are not reported as SSE",
        "reference": {
            "candidate_id": v9.CANDIDATE_ID,
            "artifact": file_record(V9_RECONSTRUCTION_PATH),
            "aggregate_bf16_sse": V9_AGGREGATE_BF16_SSE,
            "fresh_reconstruction_required": True,
        },
        "tensor_count": len(manifests["reconstruction"]),
        "fresh_v9_exact_reproduction_count": fresh_count,
        "per_tensor_non_regressing_count": non_regressing_count,
        "per_tensor_strictly_improved_count": improved_count,
        "all_169_fresh_v9_reproductions_match": fresh_count == 169,
        "all_169_tensors_non_regressing_against_fresh_v9": non_regressing_count == 169,
        "aggregate_fresh_v9_bf16_sse": aggregate_v9,
        "aggregate_v10_bf16_sse": aggregate_v10,
        "aggregate_strictly_improved_against_fresh_v9": aggregate_v10 < aggregate_v9,
        "tensors": manifests["reconstruction"],
    }
    passed = (
        reconstruction_body["all_169_fresh_v9_reproductions_match"]
        and reconstruction_body["all_169_tensors_non_regressing_against_fresh_v9"]
        and reconstruction_body["aggregate_strictly_improved_against_fresh_v9"]
    )
    reconstruction_report = {
        **reconstruction_body,
        "status": "PASS_FRESH_V9_AND_169_OF_169_WITH_STRICT_AGGREGATE_IMPROVEMENT" if passed else "NO_GO_V9_RELATIVE_MODEL_ONLY_RECONSTRUCTION_GATE",
        "reconstruction_report_sha256": canonical_sha256(reconstruction_body),
    }
    require(
        totals
        == {
            "weight_count": 493961216,
            "payload": 246980608,
            "packed_scale24": 1368192,
            "dual_codebook": 61664,
            "selector": 482384,
        },
        f"whole-model V10 storage geometry differs: {totals}",
    )
    hashes = {f"{name}_manifest_sha256": canonical_sha256(values) for name, values in manifests.items()}
    hashes.update(
        {
            "weight_manifest_sha256": canonical_sha256(weight_manifest),
            "reconstruction_report_sha256": reconstruction_report["reconstruction_report_sha256"],
        }
    )
    hashes.update({f"{name}_stream_sha256": digest.hexdigest() for name, digest in streams.items()})
    storage = {
        "total_weight_count": totals["weight_count"],
        "total_payload_bytes": totals["payload"],
        "total_packed_scale24_bytes": totals["packed_scale24"],
        "total_dual_codebook_bytes": totals["dual_codebook"],
        "total_selector_bytes": totals["selector"],
        "whole_model_total_weight_bytes": sum(totals[name] for name in ("payload", "packed_scale24", "dual_codebook", "selector")),
        "whole_model_exact_bits_per_weight": 4.030969880841819,
        "standard_decoder_layer_exact_bits_per_weight": 4.029584478021978,
        "lm_head_exact_bits_per_weight": 4.034611377722951,
        "maximum_total_stored_bits_per_weight": 4.036,
    }
    return weight_manifest, {"manifests": manifests, "hashes": hashes, "storage": storage, "reconstruction": reconstruction_report}


def _load_candidate(label: str) -> tuple[nn.Module, dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    embedding = model.model.embed_tokens.weight
    lm_head = model.lm_head.weight
    require(model.config.tie_word_embeddings is True, "source tie_word_embeddings differs")
    require(lm_head is embedding and lm_head.data_ptr() == embedding.data_ptr(), "source lm_head alias differs")
    source_shape = list(embedding.shape)
    source_dtype = str(embedding.dtype)
    source_object_id = id(embedding)
    source_pointer = embedding.data_ptr()
    source_hash = v7.v3.tensor_sha256(embedding)
    require(source_hash == SOURCE_EMBEDDING_SHA256, "source embedding BF16 hash differs")
    cloned = lm_head.detach().clone()
    require(v7.v3.tensor_sha256(cloned) == source_hash, "detached lm_head clone differs before split")
    model.lm_head.weight = nn.Parameter(cloned, requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight is not embedding and model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head split failed")
    require(v7.v3.tensor_sha256(model.lm_head.weight) == source_hash, "lm_head clone bytes differ after split")
    require(id(model.model.embed_tokens.weight) == source_object_id and model.model.embed_tokens.weight.data_ptr() == source_pointer, "embedding identity changed during split")

    weight_manifest, construction = _quantize_model(model)
    require(model.config.tie_word_embeddings is False, "candidate config was retied")
    require(model.model.embed_tokens.weight is embedding and embedding.data_ptr() == source_pointer, "embedding identity changed after V10 construction")
    require(list(embedding.shape) == source_shape and str(embedding.dtype) == source_dtype, "embedding shape or dtype changed")
    embedding_hash = v7.v3.tensor_sha256(embedding)
    require(embedding_hash == source_hash == SOURCE_EMBEDDING_SHA256, "embedding bytes changed after V10 construction")
    state_manifest = v7.v3.state_manifest(model)
    identity = {
        "embedding_bf16_sha256": embedding_hash,
        **construction["hashes"],
        "candidate_state_manifest_sha256": state_manifest["candidate_state_manifest_sha256"],
    }
    manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "construction_label": label,
        "source_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "config_tie_word_embeddings_after_split": False,
        "module_counts": {"transformer_linears": 168, "lm_head": 1, "total_dual_codebook_linears": 169},
        "identity": identity,
        "storage": construction["storage"],
        **{f"{name}_manifest": values for name, values in construction["manifests"].items()},
        "weight_manifest": weight_manifest,
        "state_manifest": state_manifest,
    }
    witness = {
        "construction_label": label,
        "source": {
            "same_parameter_object": True,
            "same_storage_pointer": True,
            "embedding_object_id": source_object_id,
            "embedding_storage_pointer": source_pointer,
            "embedding_shape": source_shape,
            "embedding_dtype": source_dtype,
            "embedding_bf16_sha256": source_hash,
        },
        "post_split": {
            "different_parameter_objects": model.lm_head.weight is not embedding,
            "different_storage_pointers": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
            "embedding_object_preserved": id(embedding) == source_object_id,
            "embedding_storage_preserved": embedding.data_ptr() == source_pointer,
            "config_tie_word_embeddings": bool(model.config.tie_word_embeddings),
        },
        "post_v10": {
            "embedding_bf16_sha256": embedding_hash,
            "embedding_hash_matches_source": embedding_hash == SOURCE_EMBEDDING_SHA256,
            "lm_head_storage_is_separate": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
            "transformer_dual_codebook_count": 168,
            "lm_head_dual_codebook_count": 1,
        },
    }
    return model, manifest, witness, construction["reconstruction"]


def _compare_candidates(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> dict[str, Any]:
    comparisons = {key: candidate_a["identity"][key] == candidate_b["identity"][key] for key in STATE_MATCH_KEYS}
    require(all(comparisons.values()), f"candidate A/B identity differs: {comparisons}")
    manifest_fields = tuple(
        f"{name}_manifest"
        for name in ("payload", "packed_scale24", "dual_codebook", "selector", "reconstruction", "objective", "iteration_trace")
    ) + ("weight_manifest", "state_manifest")
    field_matches = {field: candidate_a[field] == candidate_b[field] for field in manifest_fields}
    require(all(field_matches.values()), f"candidate A/B whole-model manifests differ: {field_matches}")
    return {"identity_matches": comparisons, "manifest_matches": field_matches, "all_match": True}


def _validate_reconstruction(report: dict[str, Any]) -> bool:
    return (
        report["fresh_v9_exact_reproduction_count"] == 169
        and report["per_tensor_non_regressing_count"] == 169
        and report["aggregate_fresh_v9_bf16_sse"] == V9_AGGREGATE_BF16_SSE
        and report["aggregate_v10_bf16_sse"] < V9_AGGREGATE_BF16_SSE
        and report["all_169_fresh_v9_reproductions_match"] is True
        and report["all_169_tensors_non_regressing_against_fresh_v9"] is True
        and report["aggregate_strictly_improved_against_fresh_v9"] is True
    )


def _validate_quality(report: dict[str, Any]) -> bool:
    return report["exact_bf16_sequence_match_count"] == 8 and report["response_gate_outcome_match_count"] == 8


def _validate_dynamic(report: dict[str, Any]) -> bool:
    activation = report["activation_execution"]
    return (
        activation["event_count"] == EXPECTED_EVENTS
        and activation["all_312_layer_event_names_present"] is True
        and activation["saturation_count"] == 0
        and activation["clipping_count"] == 0
        and activation["reserved_minus_128_produced"] is False
        and activation["transport_metadata_bytes_per_decoder_layer_token"] <= 832
    )


def _audit_attempt_namespace(expect_root_absent: bool) -> dict[str, Any]:
    root_existed = OUTPUT_DIR.exists()
    if expect_root_absent:
        require(not root_existed, "fresh V10 namespace already exists")
    markers = list(OUTPUT_DIR.rglob("execution_started.json")) if root_existed else []
    require(not markers and not ATTEMPT_DIR.exists(), "V10 official attempt or marker exists")
    return {
        "root_existed_before_preparation": root_existed,
        "attempt_directory_exists": False,
        "execution_marker_exists": False,
        "attempt_budget": 1,
        "attempts_consumed": 0,
    }


def _recover_nonqualifying_preparation() -> dict[str, Any]:
    if not OUTPUT_DIR.exists():
        return {"recovered": False, "fresh_namespace": True, "archived_artifacts": []}
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "cannot recover a V10 namespace with an official attempt")
    entries = sorted(OUTPUT_DIR.iterdir())
    if not entries:
        OUTPUT_DIR.rmdir()
        return {"recovered": True, "fresh_namespace": True, "archived_artifacts": []}
    require(not CONTRACT_PATH.exists() and not READY_PATH.exists() and not NO_GO_PATH.exists(), "completed V10 preflight may not be recovered")
    failure_root = OUTPUT_DIR / "preflight-failures"
    failure_root.mkdir(exist_ok=True)
    recovery_dir = failure_root / f"recovered-{time.time_ns()}"
    recovery_dir.mkdir()
    archived: list[str] = []
    for path in entries:
        if path == failure_root:
            continue
        destination = recovery_dir / path.name
        shutil.move(path, destination)
        archived.append(destination.relative_to(ROOT).as_posix())
    return {"recovered": True, "fresh_namespace": False, "archived_artifacts": archived}


def _artifact_records() -> dict[str, Any]:
    paths = {
        "engineer_task": TASK_PATH,
        "engineer_task_companion": TASK_COMPANION,
        "planning_requirements": PLAN_PATH,
        "planning_requirements_companion": PLAN_COMPANION,
        "v9_review_record": V9_REVIEW_PATH,
        "v9_reconstruction_reference": V9_RECONSTRUCTION_PATH,
        "source_contract": SOURCE_CONTRACT_PATH,
        "calibration_contract": CALIBRATION_CONTRACT_PATH,
        "calibration_report": CALIBRATION_REPORT_PATH,
        "base_scales": BASE_SCALES_PATH,
        "image_manifest": IMAGE_DIR / "manifest.json",
        "official_matrix_hash_only": MATRIX_PATH,
        "runner_source": RUNNER_PATH,
        "scalable_backend_source": BACKEND_PATH,
        "reviewed_v9_runner": Path(v9.reference.__file__).resolve(),
        "reviewed_v9_backend": Path(v9.__file__).resolve(),
        "native_exact_assignment_source": v9.NATIVE_SOURCE_PATH,
        "native_exact_assignment_library": v9.NATIVE_LIBRARY_PATH,
        "reviewed_v7_backend": Path(v7.__file__).resolve(),
        "reviewed_quality_helper": Path(v7.v3.__file__).resolve(),
        "reviewed_grouped_dynamic_runner": ROOT / "tools/run_option_b_grouped_scale32_stage1.py",
        "response_gate_source": ROOT / "tools/qwen_instruct_response_gate.py",
        "identity_helper_source": ROOT / "tools/qwen_instruct_option_b.py",
        "quality_contracts": ROOT / "tools/ace2_quality_contracts.py",
        "non_evaluator_inputs": INPUT_MANIFEST_PATH,
        "candidate_a_manifest": CANDIDATE_A_PATH,
        "candidate_b_manifest": CANDIDATE_B_PATH,
        "alias_separation_witness": ALIAS_WITNESS_PATH,
        "model_only_reconstruction": RECONSTRUCTION_PATH,
        "non_evaluator_quality": QUALITY_PATH,
        "dynamic_preflight": DYNAMIC_PATH,
        "runner_self_test": SELF_TEST_PATH,
        "preparation_chronology": CHRONOLOGY_PATH,
    }
    if RECOVERY_PATH.is_file():
        paths["preparation_recovery"] = RECOVERY_PATH
    failure_root = OUTPUT_DIR / "preflight-failures"
    if failure_root.is_dir():
        for index, path in enumerate(sorted(item for item in failure_root.rglob("*") if item.is_file())):
            paths[f"nonqualifying_preflight_failure_{index:02d}"] = path
    return {name: file_record(path) for name, path in paths.items()}


def _build_contract(
    environment: dict[str, Any],
    source_identity: dict[str, Any],
    namespace_audit: dict[str, Any],
    state_comparison: dict[str, Any],
    gate_passes: dict[str, bool],
) -> dict[str, Any]:
    task = load_json(TASK_PATH)
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "V10 Engineer authority differs")
    require(task["fresh_attempt_namespace"]["attempt_budget"] == 1 and task["fresh_attempt_namespace"]["attempts_consumed_at_task_freeze"] == 0, "V10 attempt accounting differs")
    require(reference.sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    artifacts = _artifact_records()
    all_pass = all(gate_passes.values())
    contract = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "task_id": TASK_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "status": "PREATTEMPT_GATES_PASS" if all_pass else "PREATTEMPT_NO_GO",
        "authority": {
            "engineer_task": artifacts["engineer_task"],
            "planning_requirements": artifacts["planning_requirements"],
            "terminal_v9_review": artifacts["v9_review_record"],
            "attempt_id": "attempt-0001",
            "exact_authorized_candidate_attempts": 1,
            "attempts_consumed": 0,
            "selected_policy_id": None,
        },
        "source_model": source_identity,
        "environment": environment,
        "format_and_policy": _policy_and_format(),
        "construction": {
            "backend": "fresh exact V9 reconstruction plus exact per-row selector/payload/codebook aggregation and Python-integer Scale32 optimization",
            "source_tied_then_runtime_split_lm_head": True,
            "source_embedding_bf16_sha256": SOURCE_EMBEDDING_SHA256,
            "embedding_bytes_preserved": True,
            "transformer_linears": 168,
            "separated_lm_head_linears": 1,
            "outer_iterations": OUTER_ITERATIONS,
            "candidate_a_and_b_whole_model_manifests_match": state_comparison,
            "packed_scale24_round_trip_required": True,
            "optimizer_cost_is_not_literal_sse": True,
            "model_gate_uses_actual_nonnegative_bf16_sse": True,
        },
        "pre_attempt_gates": {
            "fresh_v9_and_169_model_closure": {"passed": gate_passes["fresh_v9_and_169_model_closure"], "artifact": artifacts["model_only_reconstruction"]},
            "non_evaluator_quality_8_of_8": {"passed": gate_passes["non_evaluator_quality_8_of_8"], "artifact": artifacts["non_evaluator_quality"]},
            "dynamic_312_event_closure": {"passed": gate_passes["dynamic_312_event_closure"], "artifact": artifacts["dynamic_preflight"]},
            "two_build_alias_codec_and_manifest_closure": {"passed": gate_passes["two_build_alias_codec_and_manifest_closure"], "artifact": artifacts["alias_separation_witness"]},
            "all_pass": all_pass,
        },
        "official_matrix_binding": {
            "artifact": artifacts["official_matrix_hash_only"],
            "sha256": MATRIX_SHA256,
            "prompt_count": 9,
            "official_input_content_accessed_during_preflight": False,
        },
        "attempt": {
            "score_policy": "binary only: 9/9 complete token arrays exactly equal BF16 and 9/9 response-gate outcomes equal BF16",
            "attempt_budget": 1,
            "attempts_consumed": 0,
        },
        "official_execution": {
            "mode_exposed_by_runner": True,
            "gated_by_verified_preattempt_ready": True,
            "attempt_directory_exists": False,
            "execution_marker_exists": False,
        },
        "namespace_audit": namespace_audit,
        "artifacts": artifacts,
        "claim_boundary": {
            "attempts_consumed": 0,
            "numerical_conclusion": None,
            "selected_policy_id": None,
            "rtl_or_ppa_claimed": False,
            "public_ace2_shell_ports_changed": False,
            "current_stage_remains": "specification",
        },
    }
    return {"contract": contract, "contract_sha256": canonical_sha256(contract)}


def _verify_bound_record(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    require(path.is_file(), f"bound artifact is missing: {record['path']}")
    require(path.stat().st_size == record["bytes"], f"bound artifact size differs: {record['path']}")
    require(reference.sha256_file(path) == record["sha256"], f"bound artifact hash differs: {record['path']}")


def load_verified_closure(*, require_unconsumed: bool) -> dict[str, Any]:
    _configure_quality_helpers()
    reference.require_project_python()
    require(CONTRACT_PATH.is_file(), "V10 predeclared contract is missing")
    require(READY_PATH.is_file() != NO_GO_PATH.is_file(), "exactly one V10 pre-attempt terminal record is required")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "V10 predeclared contract wrapper differs")
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "V10 predeclared contract canonical hash differs")
    contract = wrapper["contract"]
    require(contract["candidate_id"] == CANDIDATE_ID and contract["task_id"] == TASK_ID, "V10 predeclared authority differs")
    require(contract["artifacts"]["runner_source"]["sha256"] == reference.sha256_file(RUNNER_PATH), "V10 runner changed after freeze")
    require(contract["artifacts"]["scalable_backend_source"]["sha256"] == reference.sha256_file(BACKEND_PATH), "V10 backend changed after freeze")
    for record in contract["artifacts"].values():
        _verify_bound_record(record)
    reference.verify_companion(TASK_PATH, TASK_COMPANION)
    reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    require(reference.sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(reference.sha256_file(V9_RECONSTRUCTION_PATH) == V9_RECONSTRUCTION_SHA256, "sealed V9 reconstruction hash differs")
    require(reference.sha256_file(V9_REVIEW_PATH) == V9_REVIEW_SHA256, "sealed V9 review hash differs")
    require(verify_source_snapshot() == contract["source_model"], "source snapshot identity differs")
    packages = {**verify_versions(), "numpy": importlib.metadata.version("numpy")}
    require(packages == contract["environment"]["packages"], "package versions differ")
    require(combined_self_test() == load_json(SELF_TEST_PATH), "V10 runner self-test result differs")
    candidate_a = load_json(CANDIDATE_A_PATH)
    candidate_b = load_json(CANDIDATE_B_PATH)
    comparison = _compare_candidates(candidate_a, candidate_b)
    witness = load_json(ALIAS_WITNESS_PATH)
    require(witness["state_comparison"] == comparison, "V10 alias witness comparison differs")
    gate_passes = {
        "fresh_v9_and_169_model_closure": _validate_reconstruction(load_json(RECONSTRUCTION_PATH)),
        "non_evaluator_quality_8_of_8": _validate_quality(load_json(QUALITY_PATH)),
        "dynamic_312_event_closure": _validate_dynamic(load_json(DYNAMIC_PATH)),
        "two_build_alias_codec_and_manifest_closure": comparison["all_match"] and witness["embedding_preserved_in_both"] and witness["lm_head_separate_in_both"] and witness["packed_scale24_codec_passed"],
    }
    require(contract["pre_attempt_gates"]["all_pass"] == all(gate_passes.values()), "V10 contract aggregate gate status differs")
    for name, passed in gate_passes.items():
        require(contract["pre_attempt_gates"][name]["passed"] == passed, f"V10 contract gate differs: {name}")
    terminal_path = READY_PATH if READY_PATH.is_file() else NO_GO_PATH
    terminal = load_json(terminal_path)
    require(terminal["candidate_id"] == CANDIDATE_ID, "V10 terminal candidate differs")
    require(terminal["contract"]["sha256"] == reference.sha256_file(CONTRACT_PATH), "V10 terminal contract file hash differs")
    require(terminal["contract_sha256"] == wrapper["contract_sha256"], "V10 terminal contract canonical hash differs")
    require(terminal["bound_artifacts"] == contract["artifacts"], "V10 terminal artifact snapshot differs")
    require((terminal_path == READY_PATH) == all(gate_passes.values()), "V10 terminal disposition differs from gates")
    if require_unconsumed or terminal_path == NO_GO_PATH:
        require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "V10 official attempt exists before authorization consumption")
    quality = load_json(QUALITY_PATH)
    dynamic = load_json(DYNAMIC_PATH)
    reconstruction = load_json(RECONSTRUCTION_PATH)
    return {
        "status": terminal["status"],
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": wrapper["contract_sha256"],
        "gate_passes": gate_passes,
        "fresh_v9_exact_reproduction_count": reconstruction["fresh_v9_exact_reproduction_count"],
        "per_tensor_non_regressing_count": reconstruction["per_tensor_non_regressing_count"],
        "aggregate_fresh_v9_bf16_sse": reconstruction["aggregate_fresh_v9_bf16_sse"],
        "aggregate_v10_bf16_sse": reconstruction["aggregate_v10_bf16_sse"],
        "exact_sequence_matches": quality["exact_bf16_sequence_match_count"],
        "gate_outcome_matches": quality["response_gate_outcome_match_count"],
        "event_count": dynamic["activation_execution"]["event_count"],
        "attempts_consumed": int(MARKER_PATH.exists()),
        "official_execution_mode_exposed": True,
    }


def prepare() -> dict[str, Any]:
    _configure_quality_helpers()
    base_environment = reference.require_project_python()
    environment = {
        **base_environment,
        "platform": platform.platform(),
        "packages": {**verify_versions(), "numpy": importlib.metadata.version("numpy")},
        "construction_torch_num_threads": CONSTRUCTION_TORCH_THREADS,
        "quality_torch_num_threads": TORCH_THREADS,
        "backend": "exact V9-relative per-row selector construction with lossless packed Scale24",
        "native_build": v9._native_build_identity(),
    }
    recovery = _recover_nonqualifying_preparation()
    namespace_audit = _audit_attempt_namespace(expect_root_absent=recovery["fresh_namespace"])
    namespace_audit["preparation_recovery"] = recovery
    reference.verify_companion(TASK_PATH, TASK_COMPANION)
    reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    verify_source_contract()
    source_identity = verify_source_snapshot()
    require(reference.sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(reference.sha256_file(V9_RECONSTRUCTION_PATH) == V9_RECONSTRUCTION_SHA256, "sealed V9 reconstruction hash differs")
    require(reference.sha256_file(V9_REVIEW_PATH) == V9_REVIEW_SHA256, "sealed V9 review hash differs")
    torch.set_num_threads(CONSTRUCTION_TORCH_THREADS)
    torch.set_num_interop_threads(1)
    self_test = combined_self_test()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    probe = OUTPUT_DIR / ".preflight-write-probe"
    descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, b"preflight\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(probe.read_bytes() == b"preflight\n", "V10 output-path readback differs")
    probe.unlink()

    chronology: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "preparation_started_at_utc": utc_now(),
        "namespace_audit": namespace_audit,
        "official_matrix_content_accessed": False,
        "output_path_preflight": {"exclusive_create": True, "readback": True, "collision_behavior": "O_EXCL", "execution_marker_created": False},
    }
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    inputs, input_manifest = v7.v3.non_evaluator_inputs(tokenizer)

    chronology["candidate_a_started_at_utc"] = utc_now()
    print("V10_CANDIDATE_A_START", flush=True)
    candidate_a, manifest_a, witness_a, reconstruction_a = _load_candidate("candidate_a_independent_construction")
    chronology["candidate_a_complete_at_utc"] = utc_now()
    print("V10_CANDIDATE_A_DONE", flush=True)
    del candidate_a
    gc.collect()

    chronology["candidate_b_started_at_utc"] = utc_now()
    print("V10_CANDIDATE_B_START", flush=True)
    candidate_b, manifest_b, witness_b, reconstruction_b = _load_candidate("candidate_b_independent_quality_and_dynamic_preflight")
    state_comparison = _compare_candidates(manifest_a, manifest_b)
    require(reconstruction_a == reconstruction_b, "V10 candidate A/B reconstruction reports differ")
    chronology["candidate_b_constructed_at_utc"] = utc_now()
    print("V10_CANDIDATE_B_CONSTRUCTED quality_dynamic_start", flush=True)
    torch.set_num_threads(TORCH_THREADS)
    quality, dynamic = v7.v3.non_evaluator_quality_and_dynamic(candidate_b, tokenizer, inputs, input_manifest)
    chronology["candidate_b_quality_and_dynamic_complete_at_utc"] = utc_now()
    print("V10_QUALITY_DYNAMIC_DONE", flush=True)
    del candidate_b
    gc.collect()

    gate_passes = {
        "fresh_v9_and_169_model_closure": _validate_reconstruction(reconstruction_b),
        "non_evaluator_quality_8_of_8": _validate_quality(quality),
        "dynamic_312_event_closure": _validate_dynamic(dynamic),
        "two_build_alias_codec_and_manifest_closure": (
            state_comparison["all_match"]
            and witness_a["post_v10"]["embedding_hash_matches_source"]
            and witness_b["post_v10"]["embedding_hash_matches_source"]
            and witness_a["post_v10"]["lm_head_storage_is_separate"]
            and witness_b["post_v10"]["lm_head_storage_is_separate"]
            and self_test["checks"]["packed_scale24_legal_extrema_and_all_zero_round_trip"]
        ),
    }
    require(not MARKER_PATH.exists() and not ATTEMPT_DIR.exists(), "V10 official attempt appeared before artifact freeze")

    PREATTEMPT_DIR.mkdir(parents=False, exist_ok=False)
    write_json(INPUT_MANIFEST_PATH, input_manifest)
    write_json(CANDIDATE_A_PATH, manifest_a)
    write_json(CANDIDATE_B_PATH, manifest_b)
    write_json(
        ALIAS_WITNESS_PATH,
        {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "candidate_a": witness_a,
            "candidate_b": witness_b,
            "state_comparison": state_comparison,
            "embedding_preserved_in_both": witness_a["post_v10"]["embedding_hash_matches_source"] and witness_b["post_v10"]["embedding_hash_matches_source"],
            "lm_head_separate_in_both": witness_a["post_v10"]["lm_head_storage_is_separate"] and witness_b["post_v10"]["lm_head_storage_is_separate"],
            "packed_scale24_codec_passed": self_test["checks"]["packed_scale24_legal_extrema_and_all_zero_round_trip"],
        },
    )
    write_json(RECONSTRUCTION_PATH, reconstruction_b)
    write_json(QUALITY_PATH, quality)
    write_json(DYNAMIC_PATH, dynamic)
    write_json(SELF_TEST_PATH, self_test)
    if recovery["recovered"]:
        write_json(RECOVERY_PATH, {"schema_version": 1, "candidate_id": CANDIDATE_ID, **recovery})
    chronology["pre_attempt_artifacts_frozen_at_utc"] = utc_now()
    chronology["gate_passes"] = gate_passes
    write_json(CHRONOLOGY_PATH, chronology)

    wrapper = _build_contract(environment, source_identity, namespace_audit, state_comparison, gate_passes)
    write_json(CONTRACT_PATH, wrapper)
    all_pass = all(gate_passes.values())
    terminal_path = READY_PATH if all_pass else NO_GO_PATH
    terminal = {
        "schema_version": 1,
        "status": "PREATTEMPT_READY" if all_pass else "PREATTEMPT_NO_GO",
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": utc_now(),
        "contract": file_record(CONTRACT_PATH),
        "contract_sha256": wrapper["contract_sha256"],
        "bound_artifacts": wrapper["contract"]["artifacts"],
        "gate_passes": gate_passes,
        "actual_nonnegative_bf16_sse": {"fresh_v9": reconstruction_b["aggregate_fresh_v9_bf16_sse"], "v10": reconstruction_b["aggregate_v10_bf16_sse"]},
        "attempt_budget": 1,
        "attempts_consumed": 0,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": True,
        "disposition": (
            "all marker-free V10 gates pass; PREATTEMPT_READY is sealed and the single official attempt may now be consumed"
            if all_pass
            else "preserve V10 no-go evidence, leave attempt-0001 absent and unconsumed, keep selected_policy_id null, and submit for Fresh Review"
        ),
    }
    atomic_write_json(terminal_path, terminal)
    verified = load_verified_closure(require_unconsumed=True)
    for path in list(PREATTEMPT_DIR.iterdir()) + [CONTRACT_PATH, terminal_path]:
        if path.is_file():
            path.chmod(0o444)
    return verified


def execute_once() -> int:
    _configure_quality_helpers()
    closure = load_verified_closure(require_unconsumed=True)
    require(closure["status"] == "PREATTEMPT_READY", "V10 official execution requires verified PREATTEMPT_READY")
    wrapper = load_json(CONTRACT_PATH)
    ATTEMPT_DIR.mkdir(parents=False, exist_ok=False)
    marker = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "task_id": TASK_ID,
        "candidate_id": CANDIDATE_ID,
        "attempt_id": "attempt-0001",
        "started_at_utc": utc_now(),
        "contract": file_record(CONTRACT_PATH),
        "contract_sha256": wrapper["contract_sha256"],
        "preattempt_ready": file_record(READY_PATH),
        "runner_sha256": reference.sha256_file(RUNNER_PATH),
        "backend_sha256": reference.sha256_file(BACKEND_PATH),
        "matrix_sha256": MATRIX_SHA256,
        "selected_policy_id_before_execution": None,
        "authorization_consumed": True,
    }
    atomic_write_json(MARKER_PATH, marker)
    started = time.monotonic()
    run_log_path = ATTEMPT_DIR / "run.log"
    candidate_prompt_count = 0
    result_path: Path | None = None
    reviewer_path: Path | None = None
    failure_path: Path | None = None
    return_code = 5
    terminal_class = "EVALUATOR_NO_EXECUTION"
    policy: Any | None = None
    run_log = run_log_path.open("w", encoding="utf-8")
    try:
        log_message(run_log, f"official_attempt_started candidate={CANDIDATE_ID} attempt=attempt-0001")
        torch.set_num_threads(TORCH_THREADS)
        source_contract = verify_source_contract()
        source_identity = verify_source_snapshot()
        versions = verify_versions()
        matrix = load_json(MATRIX_PATH)
        prompt_specs = {record["case_id"]: record for record in prompt_binding(matrix)}
        from discriminate_qwen_instruct_w4_weight_policies import PROMPTS, chat_input

        require(len(PROMPTS) == 9, "official prompt count differs")
        tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
        inputs = {prompt.case_id: chat_input(tokenizer, prompt.prompt, DEFAULT_SYSTEM) for prompt in PROMPTS}
        reference_model = AutoModelForCausalLM.from_pretrained(
            SNAPSHOT,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval()
        references: dict[str, dict[str, Any]] = {}
        for prompt in PROMPTS:
            bf16 = generate_reference(reference_model, tokenizer, inputs[prompt.case_id], prompt.expected)
            frozen = prompt_specs[prompt.case_id]
            require(bf16["generated_token_ids"] == frozen["bf16_generated_token_ids"], f"BF16 tokens differ: {prompt.case_id}")
            require(bf16["response_gate"]["status"] == frozen["bf16_response_gate_status"], f"BF16 response gate differs: {prompt.case_id}")
            references[prompt.case_id] = bf16
        del reference_model
        gc.collect()

        candidate_model, official_manifest, official_witness, _reconstruction = _load_candidate("official_attempt_candidate")
        comparison = _compare_candidates(load_json(CANDIDATE_B_PATH), official_manifest)
        require(comparison["all_match"], "official V10 candidate differs from preflight")
        policy = v7.v3.AuditedDynamicPolicy(load_json(BASE_SCALES_PATH))
        policy.install(candidate_model)
        prompt_results: list[dict[str, Any]] = []
        model_forwards = 0
        for prompt in PROMPTS:
            bf16 = references[prompt.case_id]
            candidate, forwards = v7.grouped.generate_candidate(
                candidate_model, tokenizer, inputs[prompt.case_id], prompt.expected, bf16
            )
            model_forwards += forwards
            candidate_prompt_count += 1
            disagreements = sequence_disagreements(bf16["generated_token_ids"], candidate["generated_token_ids"])
            gate_match = candidate["response_gate"]["status"] == bf16["response_gate"]["status"]
            public_bf16 = dict(bf16)
            public_bf16.pop("_logits")
            prompt_results.append(
                {
                    "case_id": prompt.case_id,
                    "prompt_sha256": prompt_specs[prompt.case_id]["prompt_sha256"],
                    "expected_sha256": prompt_specs[prompt.case_id]["expected_sha256"],
                    "bf16": public_bf16,
                    "candidate": candidate,
                    "token_disagreements": disagreements,
                    "response_gate_outcome_match": gate_match,
                }
            )
        activation_summary = policy.audited_summary(model_forwards)
        policy.uninstall()
        policy = None
        aggregate = aggregate_prompts(prompt_results)
        disagreement_set = [
            {
                "case_id": item["case_id"],
                "token_disagreements": item["token_disagreements"],
                "bf16_response_gate_status": item["bf16"]["response_gate"]["status"],
                "candidate_response_gate_status": item["candidate"]["response_gate"]["status"],
                "response_gate_outcome_match": item["response_gate_outcome_match"],
            }
            for item in prompt_results
            if item["token_disagreements"] or not item["response_gate_outcome_match"]
        ]
        eligible = aggregate["exact_bf16_sequence_match_count"] == 9 and aggregate["response_gate_outcome_match_count"] == 9
        numerical_status = "PASS_ELIGIBLE_FOR_FRESH_REVIEW" if eligible else "BLOCKED_EXACT_BF16_DISAGREEMENT"
        result = {
            "schema_version": 1,
            "classification": "qwen_instruct_stage1_option_b_alias_safe_packed_scale24_per_row_selector_single_candidate",
            "status": numerical_status,
            "mission_id": MISSION_ID,
            "task_id": TASK_ID,
            "candidate_id": CANDIDATE_ID,
            "attempt_id": "attempt-0001",
            "contract": file_record(CONTRACT_PATH),
            "preattempt_ready": file_record(READY_PATH),
            "matrix": file_record(MATRIX_PATH),
            "base_scales": file_record(BASE_SCALES_PATH),
            "source_model": source_identity,
            "source_contract_status": source_contract["status"],
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {**versions, "numpy": importlib.metadata.version("numpy")},
                "torch_num_threads": torch.get_num_threads(),
            },
            "official_candidate_manifest": official_manifest,
            "official_alias_witness": official_witness,
            "preflight_state_match": comparison,
            "activation_execution": activation_summary,
            "prompts": prompt_results,
            "aggregate": aggregate,
            "disagreement_set": disagreement_set,
            "eligibility": {
                "rule": "binary 9/9 exact BF16 token arrays and 9/9 response-gate outcomes",
                "eligible": eligible,
                "eligible_candidate_id": CANDIDATE_ID if eligible else None,
                "selected_policy_id": None,
                "fresh_reviewer_required": True,
            },
            "failure_analysis": None
            if eligible
            else {
                "failure_taxonomy": "NUMERICAL_EXACT_BF16_DISAGREEMENT",
                "root_cause_hypothesis": "the V10 four-bit reconstruction changed one or more greedy logits relative to BF16",
                "regression": "preserve the immutable disagreement set; do not rerun the consumed candidate",
            },
            "scope_guards": {
                "candidate_attempt_count": 1,
                "alternate_policy_executed": False,
                "official_inputs_used_before_marker": False,
                "saturation_or_clipping_used": False,
                "runner_changed_after_freeze": False,
                "rtl_mutated": False,
                "ppa_executed": False,
                "network_access_performed": False,
            },
            "claim_boundary": "Numerical-policy attempt only; no RTL, latency, timing, PPA, demo, U280, product-selection, or stage-advance claim.",
            "timing": {"completed_at_utc": utc_now(), "elapsed_seconds": time.monotonic() - started},
        }
        result_path = ATTEMPT_DIR / "results.json"
        write_json(result_path, result)
        reviewer = {
            "schema_version": 1,
            "status": "READY_FOR_FRESH_REVIEW",
            "mission_id": MISSION_ID,
            "task_id": TASK_ID,
            "candidate_id": CANDIDATE_ID,
            "numerical_result": numerical_status,
            "contract": file_record(CONTRACT_PATH),
            "preattempt_ready": file_record(READY_PATH),
            "results": file_record(result_path),
            "selected_policy_id_before_adjudication": None,
            "independent_acceptance_required": True,
            "requested_review": [
                "task and hash closure",
                "lossless packed Scale24 and per-row selector accounting",
                "deterministic 169-tensor V9-relative construction",
                "8/8 preflight exactness",
                "312 legal Dynamic Scale32 events",
                "one consumed attempt",
                "nine exact BF16 token arrays and response-gate comparisons",
            ],
            "verification_command": ["./.venv/bin/python", RUNNER_PATH.relative_to(ROOT).as_posix(), "verify-result"],
            "official_attempt_regeneration_forbidden": True,
        }
        reviewer_path = ATTEMPT_DIR / "fresh_reviewer_input.json"
        write_json(reviewer_path, reviewer)
        terminal_class = "NUMERICAL_SUCCESS" if eligible else "NUMERICAL_FAILURE"
        return_code = 0 if eligible else 4
        log_message(run_log, f"official_attempt_complete status={numerical_status} exact_sequences={aggregate['exact_bf16_sequence_match_count']}/9 gate_matches={aggregate['response_gate_outcome_match_count']}/9")
    except Exception as exc:
        if policy is not None:
            policy.uninstall()
        terminal_class = "EVALUATOR_NO_EXECUTION" if candidate_prompt_count == 0 else "EVALUATOR_PARTIAL_EXECUTION_FAILURE"
        failure = {
            "schema_version": 1,
            "status": "BLOCKED_" + terminal_class,
            "classification": terminal_class,
            "failure_taxonomy": terminal_class,
            "root_cause_hypothesis": f"official runner raised {type(exc).__name__}: {exc}",
            "regression": "preserve the immutable failure, marker, log, traceback, and completed-prompt count; do not replay",
            "mission_id": MISSION_ID,
            "task_id": TASK_ID,
            "candidate_id": CANDIDATE_ID,
            "attempt_id": "attempt-0001",
            "candidate_prompt_results_completed": candidate_prompt_count,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "numerical_correctness_conclusion": None,
            "selected_policy_id": None,
            "authorization_consumed": True,
            "failed_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
        }
        failure_path = ATTEMPT_DIR / "failure.json"
        write_json(failure_path, failure)
        log_message(run_log, f"official_attempt_failed classification={terminal_class} error_type={type(exc).__name__} error={exc}")
        return_code = 5
    finally:
        run_log.flush()
        os.fsync(run_log.fileno())
        run_log.close()
    terminal_path = ATTEMPT_DIR / "terminal_status.json"
    write_json(
        terminal_path,
        {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "attempt_id": "attempt-0001",
            "classification": terminal_class,
            "return_code": return_code,
            "candidate_prompt_results_completed": candidate_prompt_count,
            "completed_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    evidence = [CONTRACT_PATH, READY_PATH, MARKER_PATH, run_log_path, terminal_path]
    for path in (result_path, reviewer_path, failure_path):
        if path is not None:
            evidence.append(path)
    write_checksum_bundle(evidence)
    seal_attempt([MARKER_PATH, run_log_path, terminal_path] + [path for path in (result_path, reviewer_path, failure_path) if path is not None])
    return return_code


def verify_result() -> dict[str, Any]:
    load_verified_closure(require_unconsumed=False)
    attempts = sorted(path.name for path in OUTPUT_DIR.glob("attempt-*") if path.is_dir())
    require(attempts == ["attempt-0001"], "official attempt set differs")
    marker = load_json(MARKER_PATH)
    require(marker["authorization_consumed"] is True, "attempt authorization was not consumed")
    require(marker["runner_sha256"] == reference.sha256_file(RUNNER_PATH), "executed runner hash differs")
    require(marker["backend_sha256"] == reference.sha256_file(BACKEND_PATH), "executed backend hash differs")
    sums = ATTEMPT_DIR / "SHA256SUMS"
    companion = ATTEMPT_DIR / "SHA256SUMS.sha256"
    verify_checksum_companion(sums, companion)
    for digest, path in parse_sums(sums):
        require(path.is_file() and reference.sha256_file(path) == digest, f"official evidence hash differs: {path}")
    terminal = load_json(ATTEMPT_DIR / "terminal_status.json")
    failure_path = ATTEMPT_DIR / "failure.json"
    if failure_path.exists():
        failure = load_json(failure_path)
        require(failure["numerical_correctness_conclusion"] is None, "execution failure drew a numerical conclusion")
        return {
            "status": failure["status"],
            "classification": failure["classification"],
            "candidate_id": CANDIDATE_ID,
            "attempt_count": 1,
            "candidate_prompt_results_completed": failure["candidate_prompt_results_completed"],
            "eligible": None,
            "selected_policy_id": None,
            "failure_sha256": reference.sha256_file(failure_path),
            "terminal_return_code": terminal["return_code"],
        }
    result = load_json(ATTEMPT_DIR / "results.json")
    require(result["candidate_id"] == CANDIDATE_ID and len(result["prompts"]) == 9, "official result shape differs")
    activation = result["activation_execution"]
    require(activation["event_count"] == 312 and activation["all_312_layer_event_names_present"] is True, "official dynamic event closure differs")
    exact_count = 0
    gate_count = 0
    disagreements: list[dict[str, Any]] = []
    for item in result["prompts"]:
        token_disagreements = sequence_disagreements(item["bf16"]["generated_token_ids"], item["candidate"]["generated_token_ids"])
        gate_match = item["bf16"]["response_gate"]["status"] == item["candidate"]["response_gate"]["status"]
        require(token_disagreements == item["token_disagreements"], f"token disagreement record differs: {item['case_id']}")
        require(gate_match == item["response_gate_outcome_match"], f"response-gate record differs: {item['case_id']}")
        exact_count += int(not token_disagreements)
        gate_count += int(gate_match)
        if token_disagreements or not gate_match:
            disagreements.append(
                {
                    "case_id": item["case_id"],
                    "token_disagreements": token_disagreements,
                    "bf16_response_gate_status": item["bf16"]["response_gate"]["status"],
                    "candidate_response_gate_status": item["candidate"]["response_gate"]["status"],
                    "response_gate_outcome_match": gate_match,
                }
            )
    require(disagreements == result["disagreement_set"], "official disagreement set differs")
    eligible = exact_count == 9 and gate_count == 9
    require(result["eligibility"]["eligible"] == eligible, "official eligibility differs")
    reviewer = load_json(ATTEMPT_DIR / "fresh_reviewer_input.json")
    require(reviewer["official_attempt_regeneration_forbidden"] is True, "review packet permits regeneration")
    return {
        "status": result["status"],
        "candidate_id": CANDIDATE_ID,
        "attempt_count": 1,
        "eligible": eligible,
        "exact_sequence_matches": exact_count,
        "gate_outcome_matches": gate_count,
        "disagreement_case_ids": [item["case_id"] for item in disagreements],
        "selected_policy_id": None,
        "results_sha256": reference.sha256_file(ATTEMPT_DIR / "results.json"),
        "reviewer_input_sha256": reference.sha256_file(ATTEMPT_DIR / "fresh_reviewer_input.json"),
        "terminal_return_code": terminal["return_code"],
    }


def record_preparation_failure(exc: Exception) -> None:
    if MARKER_PATH.exists():
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failure = OUTPUT_DIR / f"preparation_failure_{time.time_ns()}.json"
    write_json(
        failure,
        {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "classification": "PREATTEMPT_FAILURE",
            "failure_taxonomy": "PREATTEMPT_EXECUTION_FAILURE",
            "root_cause_hypothesis": f"runner raised {type(exc).__name__}: {exc}",
            "regression": "repair only the isolated marker-free V10 runner and rerun pre-attempt preparation; official authorization remains unconsumed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runner_sha256": reference.sha256_file(RUNNER_PATH),
            "backend_sha256": reference.sha256_file(BACKEND_PATH),
            "traceback": traceback.format_exc(),
            "attempt_authorization_consumed": False,
            "official_execution_marker_exists": False,
            "failed_at_utc": utc_now(),
        },
    )
