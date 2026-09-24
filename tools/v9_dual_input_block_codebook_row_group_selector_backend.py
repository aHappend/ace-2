#!/usr/bin/env python3
"""Exact marker-free V9 dual-codebook constructor and fixture self-test."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_option_b_alias_safe_dual_input_block_codebook_row_group_selector_w4_grouped_dynamic_scale32_stage1 as reference
import v7_input_block_full_model_backend as v7
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
from run_all_transformer_per32_stage1 import utc_now


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
NATIVE_SOURCE_PATH = ROOT / "tools/v9_dual_codebook_exact_assignment.cpp"
NATIVE_BUILD_DIR = ROOT / "build/v9-dual-codebook-native-exact-assignment"
NATIVE_LIBRARY_PATH = NATIVE_BUILD_DIR / "libv9_dual_codebook_exact_assignment.so"
OUTPUT_DIR = reference.OFFICIAL_ROOT
ATTEMPT_DIR = OUTPUT_DIR / "attempt-0001"
MARKER_PATH = reference.OFFICIAL_MARKER
PREATTEMPT_DIR = OUTPUT_DIR / "preattempt"
INPUT_MANIFEST_PATH = PREATTEMPT_DIR / "non_evaluator_inputs.json"
CANDIDATE_A_PATH = PREATTEMPT_DIR / "candidate_a_manifest.json"
CANDIDATE_B_PATH = PREATTEMPT_DIR / "candidate_b_manifest.json"
ALIAS_WITNESS_PATH = PREATTEMPT_DIR / "alias_separation_witness.json"
RECONSTRUCTION_PATH = PREATTEMPT_DIR / "model_only_dual_codebook_reconstruction.json"
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
V7_RECONSTRUCTION_PATH = ROOT / "build/stage1-option-b-alias-safe-input-block-codebook-w4-grouped-dynamic-scale32-w4a8-v7/preattempt/model_only_input_block_reconstruction.json"
V7_RECONSTRUCTION_SHA256 = "6e320fc03ccbbae91b253eaab231ddd5e4a69b99fba44aa2079f28c7ee3f38c5"
V7_AGGREGATE_BF16_SSE = 2303.224223882074

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
    "row_scale_manifest_sha256",
    "dual_codebook_manifest_sha256",
    "selector_manifest_sha256",
    "reconstruction_manifest_sha256",
    "objective_manifest_sha256",
    "iteration_trace_manifest_sha256",
    "weight_manifest_sha256",
    "candidate_state_manifest_sha256",
)
_NATIVE_LIBRARY: ctypes.CDLL | None = None


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


def _native_build_identity() -> dict[str, Any]:
    compiler = shutil.which("g++")
    require(compiler is not None, "g++ is required for the exact V9 assignment backend")
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-fopenmp",
        str(NATIVE_SOURCE_PATH),
        "-o",
        str(NATIVE_LIBRARY_PATH),
    ]
    version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    return {
        "source_sha256": reference.sha256_file(NATIVE_SOURCE_PATH),
        "compiler": compiler,
        "compiler_version": version,
        "command": command,
    }


def _load_native_library() -> ctypes.CDLL:
    global _NATIVE_LIBRARY
    if _NATIVE_LIBRARY is not None:
        return _NATIVE_LIBRARY
    identity = _native_build_identity()
    NATIVE_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    rebuild = not NATIVE_LIBRARY_PATH.is_file() or NATIVE_LIBRARY_PATH.stat().st_mtime_ns < NATIVE_SOURCE_PATH.stat().st_mtime_ns
    if rebuild:
        subprocess.run(identity["command"], check=True)
    library = ctypes.CDLL(str(NATIVE_LIBRARY_PATH))
    library.v9_assign_selectors_exact.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.v9_assign_selectors_exact.restype = ctypes.c_int
    library.v9_assign_payload_exact.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    library.v9_assign_payload_exact.restype = ctypes.c_int
    library.v9_accumulate_codebook_aggregates.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.v9_accumulate_codebook_aggregates.restype = ctypes.c_int
    _NATIVE_LIBRARY = library
    return library


def _native_arrays(
    source: Tensor,
    records: Tensor,
    assignments: Tensor | None,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    source_bits = source.contiguous().view(torch.uint16).numpy()
    record_values = records.to(torch.int64).contiguous().numpy().astype(np.uint32, copy=False)
    assignment_values = None if assignments is None else assignments.contiguous().numpy()
    codebook_values = np.asarray(codebooks, dtype=np.int8).reshape(-1)
    selector_values = None if selectors is None else selectors.contiguous().numpy()
    return source_bits, record_values, assignment_values, codebook_values, selector_values


def _signed_i128(low: int, high: int) -> int:
    return (int(high) << 64) | int(low)


def _block_ranges(width: int) -> tuple[tuple[int, int], ...]:
    return reference.input_block_ranges(width)


def _row_group_ranges(rows: int, group_size: int) -> tuple[tuple[int, int], ...]:
    require(group_size > 0 and rows % group_size == 0, "V9 output rows are not divisible by the frozen row-group size")
    return tuple((start, start + group_size) for start in range(0, rows, group_size))


def _validate_codebooks(codebooks: Sequence[Sequence[Sequence[int]]], block_count: int) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    require(len(codebooks) == block_count, "V9 dual-codebook block count differs")
    validated = []
    for banks in codebooks:
        require(len(banks) == 2, "V9 input block does not contain exactly two codebook banks")
        validated.append((v7.v4.reference.validate_codebook(banks[0]), v7.v4.reference.validate_codebook(banks[1])))
    return tuple(validated)


def _dual_codebook_raw(codebooks: Sequence[Sequence[Sequence[int]]]) -> bytes:
    validated = _validate_codebooks(codebooks, len(codebooks))
    return b"".join(bytes(point & 0xFF for point in bank) for banks in validated for bank in banks)


def _pack_selectors(selectors: Tensor) -> bytes:
    require(selectors.ndim == 2 and selectors.dtype == torch.uint8, "V9 selector tensor format differs")
    flat = [int(value) for value in selectors.reshape(-1).tolist()]
    require(all(value in (0, 1) for value in flat), "V9 selector escaped one bit")
    byte_count = (len(flat) + 7) // 8
    aligned_count = ((byte_count + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES) * ALIGNMENT_BYTES
    raw = bytearray(aligned_count)
    for ordinal, value in enumerate(flat):
        raw[ordinal // 8] |= value << (ordinal % 8)
    return bytes(raw)


def _unpack_selectors(raw: bytes, group_count: int, block_count: int) -> Tensor:
    bit_count = group_count * block_count
    byte_count = (bit_count + 7) // 8
    aligned_count = ((byte_count + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES) * ALIGNMENT_BYTES
    require(len(raw) == aligned_count, "V9 selector stream alignment differs")
    values = [(raw[ordinal // 8] >> (ordinal % 8)) & 1 for ordinal in range(bit_count)]
    for ordinal in range(bit_count, len(raw) * 8):
        require(((raw[ordinal // 8] >> (ordinal % 8)) & 1) == 0, "V9 selector padding bit is nonzero")
    return torch.tensor(values, dtype=torch.uint8).reshape(group_count, block_count)


def _selected_points(
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor,
    group_size: int,
    row_start: int,
    row_stop: int,
) -> Tensor:
    width = int(assignments.shape[1])
    selected = torch.empty((row_stop - row_start, width), dtype=torch.int64)
    row_indices = torch.arange(row_start, row_stop, dtype=torch.int64)
    group_indices = torch.div(row_indices, group_size, rounding_mode="floor")
    for block_index, (start, stop) in enumerate(_block_ranges(width)):
        bank_by_row = selectors[group_indices, block_index].to(torch.bool)
        bank_a = torch.tensor(codebooks[block_index][0], dtype=torch.int64)
        bank_b = torch.tensor(codebooks[block_index][1], dtype=torch.int64)
        indices = assignments[row_start:row_stop, start:stop].to(torch.int64)
        points_a = bank_a[indices]
        points_b = bank_b[indices]
        selected[:, start:stop] = torch.where(bank_by_row[:, None], points_b, points_a)
    return selected


def _row_statistics(
    source: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor,
    group_size: int,
    base_shift: int,
) -> tuple[list[int], list[int], list[int]]:
    rows, width = source.shape
    weight_codepoint: list[int] = []
    codepoint_squared: list[int] = []
    absolute_maximum: list[int] = []
    for row_start, row_stop in v7.v4._row_chunks(rows, width):
        significands, shifts = v7.v4._decode_bf16(source[row_start:row_stop])
        delta = torch.where(significands == 0, torch.zeros_like(shifts), shifts - base_shift)
        require(bool(torch.all((delta >= 0) & (delta <= 54))), "BF16 tensor lattice span exceeds V9 row update")
        exact = torch.bitwise_left_shift(significands, delta)
        points = _selected_points(assignments, codebooks, selectors, group_size, row_start, row_stop)
        maximum = int(torch.abs(exact).max())
        require(maximum * 127 * width < (1 << 63), "V9 row-update reduction may overflow int64")
        weight_codepoint.extend(int(value) for value in torch.sum(exact * points, dim=1).tolist())
        codepoint_squared.extend(int(value) for value in torch.sum(points * points, dim=1).tolist())
        absolute_maximum.extend(int(value) for value in torch.abs(exact).amax(dim=1).tolist())
    return weight_codepoint, codepoint_squared, absolute_maximum


def _objective(
    source: Tensor,
    records: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor,
    group_size: int,
    base_shift: int,
    objective_exponent: int,
) -> int:
    weight_codepoint, codepoint_squared, _absolute_maximum = _row_statistics(
        source, assignments, codebooks, selectors, group_size, base_shift
    )
    return sum(
        v7.v4._row_objective(int(record), wp, p2, base_shift, objective_exponent)
        for record, wp, p2 in zip(records.tolist(), weight_codepoint, codepoint_squared, strict=True)
    )


def _objective_from_statistics(
    records: Tensor,
    statistics: tuple[Sequence[int], Sequence[int], Sequence[int]],
    base_shift: int,
    objective_exponent: int,
) -> int:
    weight_codepoint, codepoint_squared, _absolute_maximum = statistics
    return sum(
        v7.v4._row_objective(int(record), int(wp), int(p2), base_shift, objective_exponent)
        for record, wp, p2 in zip(records.tolist(), weight_codepoint, codepoint_squared, strict=True)
    )


def _update_row_records(
    source: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor,
    group_size: int,
    prior_records: Tensor,
    base_shift: int,
    objective_exponent: int,
    statistics: tuple[Sequence[int], Sequence[int], Sequence[int]] | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    if statistics is None:
        statistics = _row_statistics(source, assignments, codebooks, selectors, group_size, base_shift)
    weight_codepoint, codepoint_squared, absolute_maximum = statistics
    selected: list[int] = []
    candidate_counts: list[int] = []
    changed = 0
    all_zero = 0
    for wp, p2, maximum, previous in zip(
        weight_codepoint, codepoint_squared, absolute_maximum, prior_records.tolist(), strict=True
    ):
        wp = int(wp)
        p2 = int(p2)
        maximum = int(maximum)
        candidates, summary = v7.v4._candidate_row_records(wp, p2, maximum, base_shift, int(previous))
        if summary["all_zero"]:
            choice = v7.v4.SCALE32_ALL_ZERO_RECORD
            all_zero += 1
        else:
            choice = min(
                candidates,
                key=lambda record: (
                    v7.v4._row_objective(record, wp, p2, base_shift, objective_exponent),
                    *v7.v4._record_scale_key(record),
                ),
            )
        selected.append(choice)
        candidate_counts.append(summary["candidate_count"])
        changed += int(choice != int(previous))
    return torch.tensor(selected, dtype=torch.int64), {
        "row_count": len(selected),
        "changed_row_count": changed,
        "all_zero_row_count": all_zero,
        "minimum_candidate_count": min(candidate_counts),
        "maximum_candidate_count": max(candidate_counts),
        "mean_candidate_count": sum(candidate_counts) / len(candidate_counts),
    }


def _decoded_chunk_v9(
    records: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor,
    group_size: int,
    row_start: int,
    row_stop: int,
) -> Tensor:
    significands, exponents = v7.v4._records_components(records[row_start:row_stop])
    scales = torch.ldexp(significands.to(torch.float64), exponents.to(torch.int32) - 15)
    points = _selected_points(assignments, codebooks, selectors, group_size, row_start, row_stop)
    return (points.to(torch.float64) * (scales[:, None] / 16.0)).to(torch.bfloat16)


def _decode(
    records: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor,
    group_size: int,
) -> Tensor:
    rows, width = assignments.shape
    decoded = torch.empty((rows, width), dtype=torch.bfloat16)
    for row_start, row_stop in v7.v4._row_chunks(rows, width):
        decoded[row_start:row_stop] = _decoded_chunk_v9(
            records, assignments, codebooks, selectors, group_size, row_start, row_stop
        )
    return decoded


def _group_block_bf16_sse(
    source: Tensor,
    records: Tensor,
    assignments: Tensor,
    codebook: Sequence[int],
    row_start: int,
    row_stop: int,
    block_start: int,
    block_stop: int,
) -> float:
    significands, exponents = v7.v4._records_components(records[row_start:row_stop])
    scales = torch.ldexp(significands.to(torch.float64), exponents.to(torch.int32) - 15)
    points = torch.tensor(codebook, dtype=torch.float64)
    decoded = points[assignments[row_start:row_stop, block_start:block_stop].to(torch.int64)] * (scales[:, None] / 16.0)
    decoded_bf16 = decoded.to(torch.bfloat16).to(torch.float64)
    difference = source[row_start:row_stop, block_start:block_stop].to(torch.float64) - decoded_bf16
    return float(torch.sum(difference * difference))


def _initialize_bank_b(
    source: Tensor,
    records: Tensor,
    assignments: Tensor,
    v7_codebooks: Sequence[Sequence[int]],
    group_size: int,
    shift_min: int,
    shift_max: int,
) -> tuple[tuple[tuple[tuple[int, ...], tuple[int, ...]], ...], list[dict[str, Any]]]:
    groups = _row_group_ranges(int(source.shape[0]), group_size)
    selected_banks = []
    summaries = []
    for block_index, (block_start, block_stop) in enumerate(_block_ranges(int(source.shape[1]))):
        bank_a = tuple(int(value) for value in v7_codebooks[block_index])
        row_sse = torch.empty(int(source.shape[0]), dtype=torch.float64)
        points = torch.tensor(bank_a, dtype=torch.float64)
        for row_start, row_stop in v7.v4._row_chunks(int(source.shape[0]), INPUT_BLOCK_LANES):
            significands, exponents = v7.v4._records_components(records[row_start:row_stop])
            scales = torch.ldexp(significands.to(torch.float64), exponents.to(torch.int32) - 15)
            decoded = points[assignments[row_start:row_stop, block_start:block_stop].to(torch.int64)] * (
                scales[:, None] / 16.0
            )
            decoded_bf16 = decoded.to(torch.bfloat16).to(torch.float64)
            difference = source[row_start:row_stop, block_start:block_stop].to(torch.float64) - decoded_bf16
            row_sse[row_start:row_stop] = torch.sum(difference * difference, dim=1)
        group_sse = row_sse.reshape(len(groups), group_size).sum(dim=1)
        ranked = sorted(
            ((float(value), group_index) for group_index, value in enumerate(group_sse.tolist())),
            key=lambda item: (-item[0], item[1]),
        )
        selected_group_ordinals = tuple(sorted(group_index for _sse, group_index in ranked[: (len(groups) + 1) // 2]))
        selected_rows = [row for group_index in selected_group_ordinals for row in range(*groups[group_index])]
        source_subset = source[selected_rows, block_start:block_stop]
        records_subset = records[selected_rows]
        _initial_assignments, aggregates = v7.v4._assign_and_aggregate(
            source_subset, records_subset, bank_a, shift_min, shift_max
        )
        bank_b, update = v7.v4._update_codebook(aggregates, bank_a)
        selected_banks.append((bank_a, bank_b))
        summaries.append({
            "block_index": block_index,
            "selected_group_count": len(selected_group_ordinals),
            "selected_group_ordinals_sha256": hashlib.sha256(b"".join(struct.pack("<I", value) for value in selected_group_ordinals)).hexdigest(),
            "highest_ranked_group": ranked[0][1],
            "highest_ranked_group_bf16_sse": ranked[0][0],
            "bank_b_update": update,
        })
    return tuple(selected_banks), summaries


def _assign_selectors(
    source: Tensor,
    records: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    group_size: int,
    shift_min: int,
    objective_exponent: int,
) -> tuple[Tensor, int, dict[str, Any]]:
    groups = _row_group_ranges(int(source.shape[0]), group_size)
    blocks = _block_ranges(int(source.shape[1]))
    selectors = torch.empty((len(groups), len(blocks)), dtype=torch.uint8)
    source_bits, record_values, assignment_values, codebook_values, _unused = _native_arrays(
        source, records, assignments, codebooks, None
    )
    require(assignment_values is not None, "V9 native selector assignments are missing")
    selector_values = selectors.numpy()
    chosen_cost_low = np.zeros(selectors.numel(), dtype=np.uint64)
    chosen_cost_high = np.zeros(selectors.numel(), dtype=np.int64)
    bank_b_count = ctypes.c_uint64()
    tie_count = ctypes.c_uint64()
    status = _load_native_library().v9_assign_selectors_exact(
        source_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        int(source.shape[0]),
        int(source.shape[1]),
        record_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        assignment_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        codebook_values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        group_size,
        shift_min,
        objective_exponent,
        selector_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        chosen_cost_low.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        chosen_cost_high.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(bank_b_count),
        ctypes.byref(tie_count),
    )
    require(status == 0, f"V9 native selector assignment failed: {status}")
    objective = sum(_signed_i128(low, high) for low, high in zip(chosen_cost_low, chosen_cost_high, strict=True))
    return selectors, objective, {
        "selector_count": selectors.numel(),
        "bank_a_count": selectors.numel() - bank_b_count.value,
        "bank_b_count": bank_b_count.value,
        "exact_tie_to_bank_a_count": tie_count.value,
    }


def _assign_payload(
    source: Tensor,
    records: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor,
    group_size: int,
    shift_min: int,
    shift_max: int,
) -> tuple[Tensor, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    assignments = torch.empty(source.shape, dtype=torch.uint8)
    source_bits, record_values, _unused_assignments, codebook_values, selector_values = _native_arrays(
        source, records, None, codebooks, selectors
    )
    require(selector_values is not None, "V9 native payload selectors are missing")
    assignment_values = assignments.numpy()
    weight_codepoint = np.zeros(int(source.shape[0]), dtype=np.int64)
    codepoint_squared = np.zeros(int(source.shape[0]), dtype=np.int64)
    absolute_maximum = np.zeros(int(source.shape[0]), dtype=np.int64)
    status = _load_native_library().v9_assign_payload_exact(
        source_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        int(source.shape[0]),
        int(source.shape[1]),
        record_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        codebook_values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        selector_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        group_size,
        shift_min,
        assignment_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        weight_codepoint.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        codepoint_squared.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        absolute_maximum.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    require(status == 0, f"V9 native payload assignment failed: {status}")
    require(
        all(int(maximum) * 127 * int(source.shape[1]) < (1 << 63) for maximum in absolute_maximum),
        "V9 native row-update reduction may overflow int64",
    )
    return assignments, (weight_codepoint, codepoint_squared, absolute_maximum)


def _update_codebooks(
    source: Tensor,
    records: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor,
    group_size: int,
    shift_min: int,
    shift_max: int,
    objective_exponent: int,
) -> tuple[tuple[tuple[tuple[int, ...], tuple[int, ...]], ...], list[dict[str, Any]]]:
    groups = _row_group_ranges(int(source.shape[0]), group_size)
    blocks = _block_ranges(int(source.shape[1]))
    source_bits, record_values, assignment_values, _unused_codebooks, selector_values = _native_arrays(
        source, records, assignments, codebooks, selectors
    )
    require(assignment_values is not None and selector_values is not None, "V9 native codebook inputs are missing")
    value_count = len(blocks) * 2 * CODEBOOK_ENTRIES
    linear_low = np.zeros(value_count, dtype=np.uint64)
    linear_high = np.zeros(value_count, dtype=np.int64)
    quadratic_low = np.zeros(value_count, dtype=np.uint64)
    quadratic_high = np.zeros(value_count, dtype=np.int64)
    counts = np.zeros(value_count, dtype=np.uint64)
    status = _load_native_library().v9_accumulate_codebook_aggregates(
        source_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        int(source.shape[0]),
        int(source.shape[1]),
        record_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        assignment_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        selector_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        group_size,
        shift_min,
        objective_exponent,
        linear_low.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        linear_high.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        quadratic_low.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        quadratic_high.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    require(status == 0, f"V9 native codebook aggregation failed: {status}")

    def integer_at(low: np.ndarray, high: np.ndarray, index: int) -> int:
        return (int(high[index]) << 64) | int(low[index])

    updated_blocks = []
    summaries = []
    for block_index, (_block_start, _block_stop) in enumerate(blocks):
        updated_banks = []
        bank_summaries = []
        for bank in (0, 1):
            selected_groups = [group_index for group_index in range(len(groups)) if int(selectors[group_index, block_index]) == bank]
            prior = codebooks[block_index][bank]
            selected_row_count = len(selected_groups) * group_size
            if selected_row_count == 0:
                updated = tuple(prior)
                summary = {"selected_row_count": 0, "empty_bank_retained": True}
            else:
                base = (block_index * 2 + bank) * CODEBOOK_ENTRIES
                aggregates = {
                    "objective_binary_exponent": objective_exponent,
                    "linear_scaled": [integer_at(linear_low, linear_high, base + index) for index in range(CODEBOOK_ENTRIES)],
                    "quadratic_scaled": [integer_at(quadratic_low, quadratic_high, base + index) for index in range(CODEBOOK_ENTRIES)],
                    "cluster_counts": [int(counts[base + index]) for index in range(CODEBOOK_ENTRIES)],
                }
                updated, update = v7.v4._update_codebook(aggregates, prior)
                summary = {"selected_row_count": selected_row_count, "empty_bank_retained": False, **update}
            updated_banks.append(updated)
            bank_summaries.append({"bank": "A" if bank == 0 else "B", **summary})
        updated_blocks.append((updated_banks[0], updated_banks[1]))
        summaries.append({"block_index": block_index, "banks": bank_summaries})
    return tuple(updated_blocks), summaries


def construct_dual_codebook_state(
    source: Tensor,
    module_name: str,
    *,
    group_size_override: int | None = None,
) -> dict[str, Any]:
    require(source.ndim == 2 and source.dtype == torch.bfloat16, f"source weight format differs: {module_name}")
    group_size = reference.row_group_size(module_name) if group_size_override is None else group_size_override
    groups = _row_group_ranges(int(source.shape[0]), group_size)
    blocks = _block_ranges(int(source.shape[1]))
    fresh_v7 = v7._construct_input_block_state(source, module_name)
    records = fresh_v7["final_records"].clone()
    assignments = fresh_v7["final_assignments"].clone()
    codebooks = tuple((tuple(points), tuple(points)) for points in fresh_v7["final_codebooks"])
    selectors = torch.zeros((len(groups), len(blocks)), dtype=torch.uint8)
    objective_exponent = int(fresh_v7["final_objective"]["binary_exponent"])
    initial_objective = int(fresh_v7["final_objective"]["scaled_integer"])
    initial_digest = hashlib.sha256()
    fresh_v7_digest = hashlib.sha256()
    for row_start, row_stop in v7.v4._row_chunks(int(source.shape[0]), int(source.shape[1])):
        initial_reconstruction = _decoded_chunk_v9(
            records, assignments, codebooks, selectors, group_size, row_start, row_stop
        )
        fresh_v7_reconstruction = v7._decoded_chunk(
            fresh_v7["final_records"], fresh_v7["final_assignments"], fresh_v7["final_codebooks"], row_start, row_stop
        )
        require(torch.equal(initial_reconstruction, fresh_v7_reconstruction), f"V9 initialization is not byte-identical to V7: {module_name}")
        initial_digest.update(v7.v3.raw_tensor_bytes(initial_reconstruction))
        fresh_v7_digest.update(v7.v3.raw_tensor_bytes(fresh_v7_reconstruction))
    initial_v7_reconstruction_sha256 = initial_digest.hexdigest()
    require(initial_v7_reconstruction_sha256 == fresh_v7_digest.hexdigest(), f"V9 initial V7 hash differs: {module_name}")
    initial_payload = v7.v3.pack_codebook_indices(assignments)
    require(initial_payload == fresh_v7["payload"], f"V9 initial payload differs from V7: {module_name}")
    initial_scale_raw = b"".join(struct.pack("<I", int(value)) for value in records.tolist())
    require(initial_scale_raw == fresh_v7["scale_raw"], f"V9 initial row-scale stream differs from V7: {module_name}")

    codebooks, bank_b_initialization = _initialize_bank_b(
        source,
        records,
        assignments,
        fresh_v7["final_codebooks"],
        group_size,
        fresh_v7["shift_min"],
        fresh_v7["shift_max"],
    )
    require(not bool(torch.any(selectors)), "V9 selectors changed before the first assignment")
    previous_objective = initial_objective
    trace = []
    for iteration in range(1, OUTER_ITERATIONS + 1):
        selectors, selector_objective, selector_summary = _assign_selectors(
            source, records, assignments, codebooks, group_size, fresh_v7["shift_min"], objective_exponent
        )
        require(selector_objective <= previous_objective, f"V9 selector assignment increased exact source SSE: {module_name}")
        assignments, payload_statistics = _assign_payload(
            source, records, codebooks, selectors, group_size, fresh_v7["shift_min"], fresh_v7["shift_max"]
        )
        payload_objective = _objective_from_statistics(
            records, payload_statistics, fresh_v7["shift_min"], objective_exponent
        )
        require(payload_objective <= selector_objective, f"V9 payload assignment increased exact source SSE: {module_name}")
        records, row_summary = _update_row_records(
            source,
            assignments,
            codebooks,
            selectors,
            group_size,
            records,
            fresh_v7["shift_min"],
            objective_exponent,
            payload_statistics,
        )
        assignments, row_statistics = _assign_payload(
            source, records, codebooks, selectors, group_size, fresh_v7["shift_min"], fresh_v7["shift_max"]
        )
        row_objective = _objective_from_statistics(
            records, row_statistics, fresh_v7["shift_min"], objective_exponent
        )
        require(row_objective <= payload_objective, f"V9 row-scale step increased exact source SSE: {module_name}")
        codebooks, codebook_summary = _update_codebooks(
            source,
            records,
            assignments,
            codebooks,
            selectors,
            group_size,
            fresh_v7["shift_min"],
            fresh_v7["shift_max"],
            objective_exponent,
        )
        assignments, final_statistics = _assign_payload(
            source, records, codebooks, selectors, group_size, fresh_v7["shift_min"], fresh_v7["shift_max"]
        )
        final_objective = _objective_from_statistics(
            records, final_statistics, fresh_v7["shift_min"], objective_exponent
        )
        require(final_objective <= row_objective, f"V9 codebook step increased exact source SSE: {module_name}")
        require(final_objective <= previous_objective, f"V9 outer iteration increased exact source SSE: {module_name}")
        payload = v7.v3.pack_codebook_indices(assignments)
        scale_raw = b"".join(struct.pack("<I", int(value)) for value in records.tolist())
        codebook_raw = _dual_codebook_raw(codebooks)
        selector_raw = _pack_selectors(selectors)
        trace.append({
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
            "row_scale_stream_sha256": hashlib.sha256(scale_raw).hexdigest(),
            "dual_codebook_stream_sha256": hashlib.sha256(codebook_raw).hexdigest(),
            "selector_stream_sha256": hashlib.sha256(selector_raw).hexdigest(),
        })
        previous_objective = final_objective

    payload = v7.v3.pack_codebook_indices(assignments)
    scale_raw = b"".join(struct.pack("<I", int(value)) for value in records.tolist())
    codebook_raw = _dual_codebook_raw(codebooks)
    selector_raw = _pack_selectors(selectors)
    return {
        "module_name": module_name,
        "shape": list(source.shape),
        "input_block_lanes": INPUT_BLOCK_LANES,
        "input_block_count": len(blocks),
        "row_group_size": group_size,
        "row_group_count": len(groups),
        "fresh_v7_state": fresh_v7,
        "fresh_v7_exact_reproduction": True,
        "initial_v7_objective": fresh_v7["final_objective"],
        "initial_v7_reconstruction_bf16_sha256": initial_v7_reconstruction_sha256,
        "bank_b_initialization": bank_b_initialization,
        "outer_iterations_completed": len(trace),
        "final_records": records,
        "final_assignments": assignments,
        "final_codebooks": codebooks,
        "final_selectors": selectors,
        "payload": payload,
        "scale_raw": scale_raw,
        "codebook_raw": codebook_raw,
        "selector_raw": selector_raw,
        "final_objective": v7.v4._objective_record(previous_objective, objective_exponent),
        "iteration_trace": trace,
        "iteration_trace_sha256": canonical_sha256(trace),
    }


def _state_hashes(state: dict[str, Any]) -> dict[str, str]:
    reconstruction = _decode(
        state["final_records"],
        state["final_assignments"],
        state["final_codebooks"],
        state["final_selectors"],
        state["row_group_size"],
    )
    return {
        "payload_sha256": hashlib.sha256(state["payload"]).hexdigest(),
        "row_scale_stream_sha256": hashlib.sha256(state["scale_raw"]).hexdigest(),
        "dual_codebook_stream_sha256": hashlib.sha256(state["codebook_raw"]).hexdigest(),
        "selector_stream_sha256": hashlib.sha256(state["selector_raw"]).hexdigest(),
        "reconstruction_bf16_sha256": v7.v3.tensor_sha256(reconstruction),
    }


def _validate_state(state: dict[str, Any]) -> dict[str, Any]:
    rows, width = state["shape"]
    blocks = _block_ranges(width)
    groups = _row_group_ranges(rows, state["row_group_size"])
    codebooks = _validate_codebooks(state["final_codebooks"], len(blocks))
    require(state["final_assignments"].shape == (rows, width), "V9 payload geometry differs")
    require(bool(torch.all(state["final_assignments"] < CODEBOOK_ENTRIES)), "V9 payload escaped four bits")
    require(state["final_selectors"].shape == (len(groups), len(blocks)), "V9 selector geometry differs")
    require(torch.equal(_unpack_selectors(state["selector_raw"], len(groups), len(blocks)), state["final_selectors"]), "V9 selector packing round-trip differs")
    require(len(state["payload"]) == rows * width // 2, "V9 payload byte count differs")
    require(len(state["scale_raw"]) == rows * 4, "V9 row-scale byte count differs")
    require(len(state["codebook_raw"]) == len(blocks) * 2 * CODEBOOK_ENTRIES, "V9 dual-codebook byte count differs")
    require(len(state["selector_raw"]) % ALIGNMENT_BYTES == 0, "V9 selector stream is not 16-byte aligned")
    return {
        "rows": rows,
        "input_features": width,
        "input_block_count": len(blocks),
        "row_group_size": state["row_group_size"],
        "row_group_count": len(groups),
        "selector_bit_count": len(groups) * len(blocks),
        "selector_aligned_bytes": len(state["selector_raw"]),
        "dual_codebook_bytes": len(state["codebook_raw"]),
        "all_codebooks_strictly_increasing": all(
            all(left < right for left, right in zip(bank, bank[1:])) for banks in codebooks for bank in banks
        ),
    }


def combined_self_test() -> dict[str, Any]:
    environment = {
        **reference.require_project_python(),
        "platform": platform.platform(),
        "packages": {
            "torch": importlib.metadata.version("torch"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "backend": "exact BF16 dyadic V7-relative dual-codebook constructor",
    }
    reference.verify_companion(TASK_PATH, TASK_COMPANION)
    reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "frozen V9 Engineer authority differs")
    root_existed_before = OUTPUT_DIR.exists()
    marker_existed_before = MARKER_PATH.exists()
    require(not ATTEMPT_DIR.exists() and not marker_existed_before, "V9 official attempt must remain absent during constructor self-test")

    require(len(_block_ranges(896)) == 7 and len(_block_ranges(4864)) == 38, "V9 input-block geometry differs")
    illegal_width_rejected = reference.expect_rejected(lambda: _block_ranges(1024), "illegal V9 width was accepted")
    require(reference.row_group_size("model.layers.0.self_attn.q_proj") == 1, "V9 q_proj row-group size differs")
    require(reference.row_group_size("model.layers.0.mlp.down_proj") == 2, "V9 down_proj row-group size differs")
    require(reference.row_group_size("lm_head") == 32, "V9 lm_head row-group size differs")
    require(len(_row_group_ranges(151936, 32)) == 4748, "V9 lm_head row-group count differs")

    lanes = torch.arange(896, dtype=torch.int64)
    row0 = ((lanes % 47) - 23).to(torch.float64) / 256.0
    row1 = torch.where((lanes % 5) == 0, row0 * 0.25, row0 * 1.5)
    row2 = torch.where((lanes % 7) < 3, -row0 * 0.75, row0 * 0.125)
    row3 = torch.zeros_like(row0)
    fixture = torch.stack((row0, row1, row2, row3)).to(torch.bfloat16)
    candidate_a = construct_dual_codebook_state(fixture, "self_test.q_proj", group_size_override=1)
    candidate_b = construct_dual_codebook_state(fixture, "self_test.q_proj", group_size_override=1)
    geometry = _validate_state(candidate_a)
    require(_state_hashes(candidate_a) == _state_hashes(candidate_b), "independent V9 fixture state hashes differ")
    require(candidate_a["iteration_trace"] == candidate_b["iteration_trace"], "independent V9 fixture iteration traces differ")
    require(candidate_a["final_objective"] == candidate_b["final_objective"], "independent V9 fixture objectives differ")
    require(candidate_a["fresh_v7_exact_reproduction"] is True, "V9 fixture did not prove fresh V7 reproduction")
    require(candidate_a["outer_iterations_completed"] == OUTER_ITERATIONS, "V9 fixture iteration count differs")
    require(
        all(int(item["final_objective"]["scaled_integer"]) <= int(item["prior_objective"]["scaled_integer"]) for item in candidate_a["iteration_trace"]),
        "V9 fixture exact objective is not monotonic",
    )
    require(int(candidate_a["final_objective"]["scaled_integer"]) <= int(candidate_a["initial_v7_objective"]["scaled_integer"]), "V9 fixture regressed against fresh V7")

    synthetic_selectors = torch.tensor([[0, 1, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0]], dtype=torch.uint8)
    synthetic_raw = _pack_selectors(synthetic_selectors)
    require(torch.equal(_unpack_selectors(synthetic_raw, 2, 7), synthetic_selectors), "V9 synthetic selector round-trip differs")
    tampered_padding = bytearray(synthetic_raw)
    tampered_padding[-1] |= 0x80
    padding_rejected = reference.expect_rejected(
        lambda: _unpack_selectors(bytes(tampered_padding), 2, 7),
        "nonzero V9 selector padding was accepted",
    )
    require(synthetic_raw[0] == 0b11010110, "V9 selector bit order is not row-group-major/block-major LSB-first")

    empty_bank_selectors = torch.zeros_like(candidate_a["final_selectors"])
    retained_codebooks, empty_bank_summary = _update_codebooks(
        fixture,
        candidate_a["final_records"],
        candidate_a["final_assignments"],
        candidate_a["final_codebooks"],
        empty_bank_selectors,
        1,
        candidate_a["fresh_v7_state"]["shift_min"],
        candidate_a["fresh_v7_state"]["shift_max"],
        int(candidate_a["fresh_v7_state"]["final_objective"]["binary_exponent"]),
    )
    require(
        all(retained_codebooks[index][1] == candidate_a["final_codebooks"][index][1] for index in range(len(retained_codebooks))),
        "V9 empty bank did not retain its prior codebook",
    )
    require(all(block["banks"][1]["empty_bank_retained"] for block in empty_bank_summary), "V9 empty-bank summary differs")

    metadata_sequence = [
        {"block_index": block_index, "bank": bank, "beat_bytes": 16}
        for block_index in range(7)
        for bank in ("A", "B")
    ]
    require(
        [(item["block_index"], item["bank"]) for item in metadata_sequence[:4]] == [(0, "A"), (0, "B"), (1, "A"), (1, "B")],
        "V9 dual-codebook metadata order differs",
    )

    build_root = ROOT / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v9-hash-rejection-", dir=build_root) as temporary_directory:
        temporary_root = Path(temporary_directory)
        copied_artifact = temporary_root / "copied_task.json"
        copied_companion = temporary_root / "copied_task.sha256"
        copied_artifact.write_bytes(TASK_PATH.read_bytes())
        copied_companion.write_text(
            f"{reference.sha256_file(copied_artifact)}  {copied_artifact.relative_to(ROOT).as_posix()}\n",
            encoding="utf-8",
        )
        reference.verify_companion(copied_artifact, copied_companion)
        copied_artifact.write_bytes(copied_artifact.read_bytes() + b" ")
        hash_mismatch_rejected = reference.expect_rejected(
            lambda: reference.verify_companion(copied_artifact, copied_companion),
            "tampered copied V9 artifact passed companion verification",
        )

    require(OUTPUT_DIR.exists() == root_existed_before, "self-test changed the V9 namespace")
    require(MARKER_PATH.exists() == marker_existed_before is False, "self-test changed the V9 execution marker")
    hashes = _state_hashes(candidate_a)
    final_reconstruction = _decode(
        candidate_a["final_records"],
        candidate_a["final_assignments"],
        candidate_a["final_codebooks"],
        candidate_a["final_selectors"],
        candidate_a["row_group_size"],
    )
    fixture_sse = float(torch.sum((fixture.to(torch.float64) - final_reconstruction.to(torch.float64)) ** 2))
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
            "legal_width_block_counts": {"896": 7, "4864": 38},
            "illegal_width_rejected": illegal_width_rejected,
            "row_group_sizes": {"transformer_default": 1, "down_proj": 2, "lm_head": 32},
            "lm_head_row_group_count": 4748,
            "fresh_v7_payload_scale_codebook_and_reconstruction_reproduction": True,
            "independent_candidate_a_b_state_and_trace_match": True,
            "outer_iterations_completed": OUTER_ITERATIONS,
            "monotonic_exact_source_objective": True,
            "non_regression_against_fresh_v7_fixture": True,
            "dual_codebooks_strictly_increasing": geometry["all_codebooks_strictly_increasing"],
            "selector_row_group_major_block_major_lsb_first": True,
            "selector_zero_padding": True,
            "nonzero_selector_padding_rejected": padding_rejected,
            "payload_nibble_packing_reused_from_reviewed_v7": True,
            "scale32_bracketing_neighbors_ties_range_and_zero_rows_reused_from_reviewed_v7": True,
            "empty_bank_retains_prior_codebook": True,
            "metadata_sequence": metadata_sequence,
            "bound_hash_mismatch_rejected": hash_mismatch_rejected,
        },
        "fixture": {
            "shape": list(fixture.shape),
            "geometry": geometry,
            "bf16_reconstruction_sse": fixture_sse,
            "initial_v7_objective": candidate_a["initial_v7_objective"],
            "final_v9_objective": candidate_a["final_objective"],
            "iteration_trace_sha256": candidate_a["iteration_trace_sha256"],
            **hashes,
        },
        "official_build_namespace_changed": False,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": False,
        "claim_boundary": "constructor kernel and synthetic fixture self-test only; no 169-tensor, aggregate fresh-V7, 8/8 quality, 312-event, PREATTEMPT_READY, official-attempt, or RTL/PPA conclusion",
    }


def _measure_and_materialize(
    module: nn.Linear,
    source: Tensor,
    state: dict[str, Any],
    sealed_v7_record: dict[str, Any],
) -> dict[str, Any]:
    rows, width = source.shape
    fresh_v7 = state["fresh_v7_state"]
    fresh_v7_sse = 0.0
    v9_sse = 0.0
    fresh_v7_digest = hashlib.sha256()
    v9_digest = hashlib.sha256()
    with torch.no_grad():
        for row_start, row_stop in v7.v4._row_chunks(rows, width):
            decoded_v7 = v7._decoded_chunk(
                fresh_v7["final_records"],
                fresh_v7["final_assignments"],
                fresh_v7["final_codebooks"],
                row_start,
                row_stop,
            )
            decoded_v9 = _decoded_chunk_v9(
                state["final_records"],
                state["final_assignments"],
                state["final_codebooks"],
                state["final_selectors"],
                state["row_group_size"],
                row_start,
                row_stop,
            )
            source64 = source[row_start:row_stop].to(torch.float64)
            v7_difference = source64 - decoded_v7.to(torch.float64)
            v9_difference = source64 - decoded_v9.to(torch.float64)
            fresh_v7_sse += float(torch.sum(v7_difference * v7_difference))
            v9_sse += float(torch.sum(v9_difference * v9_difference))
            fresh_v7_digest.update(v7.v3.raw_tensor_bytes(decoded_v7))
            v9_digest.update(v7.v3.raw_tensor_bytes(decoded_v9))
            module.weight[row_start:row_stop].copy_(decoded_v9)
    fresh_v7_sha = fresh_v7_digest.hexdigest()
    v9_sha = v9_digest.hexdigest()
    sealed_v7_sse = float(sealed_v7_record["v7_input_block_bf16_sse"])
    require(fresh_v7_sse >= 0.0 and v9_sse >= 0.0, f"negative BF16 reconstruction SSE: {sealed_v7_record['module']}")
    require(fresh_v7_sha == sealed_v7_record["reconstruction_bf16_sha256"], f"fresh V7 reconstruction hash differs: {sealed_v7_record['module']}")
    require(math.isclose(fresh_v7_sse, sealed_v7_sse, rel_tol=1e-15, abs_tol=1e-12), f"fresh V7 reconstruction SSE differs: {sealed_v7_record['module']}")
    require(state["initial_v7_reconstruction_bf16_sha256"] == fresh_v7_sha, f"V9 initial reproduction hash differs: {sealed_v7_record['module']}")
    require(v9_sha == v7.v3.tensor_sha256(module.weight), f"V9 materialized reconstruction hash differs: {sealed_v7_record['module']}")
    return {
        "module": sealed_v7_record["module"],
        "weight_count": int(source.numel()),
        "fresh_v7_bf16_sse": sealed_v7_sse,
        "fresh_v7_remeasured_bf16_sse": fresh_v7_sse,
        "fresh_v7_remeasurement_delta": fresh_v7_sse - sealed_v7_sse,
        "sealed_v7_bf16_sse": sealed_v7_sse,
        "fresh_v7_matches_sealed": True,
        "fresh_v7_reconstruction_bf16_sha256": fresh_v7_sha,
        "v9_dual_codebook_bf16_sse": v9_sse,
        "v9_non_regressing_against_fresh_v7": v9_sse <= fresh_v7_sse,
        "v9_strictly_improved_against_fresh_v7": v9_sse < fresh_v7_sse,
        "reconstruction_bf16_sha256": v9_sha,
    }


def _policy_and_format() -> dict[str, Any]:
    body = {
        "payload": {
            "bits_per_weight": 4,
            "semantics": "unsigned selected-bank codebook index 0 through 15",
            "packing": "row-major; even flat element low nibble; odd flat element high nibble",
        },
        "row_scale": {"format": "Scale32", "bytes_per_output_row": 4, "additional_records": 0},
        "dual_codebook": {
            "banks_per_input_block": 2,
            "input_block_lanes": INPUT_BLOCK_LANES,
            "entries_per_bank": CODEBOOK_ENTRIES,
            "bytes_per_input_block": 32,
            "order": "bank A then bank B for each increasing input block",
            "format": "strictly increasing signed Q4.4 int8",
        },
        "selector": {
            "bits_per_row_group_per_input_block": 1,
            "packing": "row-group-major then input-block-major, least-significant-bit first, 16-byte tensor alignment, zero padding",
            "row_group_sizes": reference.ROW_GROUP_SIZE_BY_FAMILY,
        },
        "construction": {
            "initialization": "fresh V7 payload, row Scale32, and duplicated V7 codebook banks with all selectors A",
            "bank_b_seed": "upper half of row groups by decreasing exact V7 block BF16 SSE and increasing ordinal ties",
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
            "whole_model_total_weight_bytes": 249169056,
            "whole_model_exact_bits_per_weight": 4.035443236094066,
            "maximum_total_stored_bits_per_weight": 4.036,
        },
        "native_exact_assignment_source_sha256": reference.sha256_file(NATIVE_SOURCE_PATH),
    }
    return {**body, "policy_and_format_sha256": canonical_sha256(body)}


def _quantize_model(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {record["module"]: record for record in v7.grouped.expected_transformer_tensors()}
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == 169, "Qwen linear count differs")
    require([name for name, _module in modules if name != "lm_head"] == list(expected), "transformer linear order differs")
    require(reference.sha256_file(V7_RECONSTRUCTION_PATH) == V7_RECONSTRUCTION_SHA256, "sealed V7 reconstruction hash differs")
    sealed_report = load_json(V7_RECONSTRUCTION_PATH)
    sealed_records = {item["module"]: item for item in sealed_report["tensors"]}
    require(list(sealed_records) == [name for name, _module in modules], "sealed V7 tensor order differs")

    manifest_names = ("payload", "row_scale", "dual_codebook", "selector", "reconstruction", "objective", "iteration_trace")
    manifests: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest_names}
    weight_manifest: list[dict[str, Any]] = []
    streams = {name: hashlib.sha256() for name in ("payload", "row_scale", "dual_codebook", "selector", "reconstruction")}
    totals = {"weight_count": 0, "payload": 0, "row_scale": 0, "dual_codebook": 0, "selector": 0}

    for ordinal, (name, module) in enumerate(modules, start=1):
        started = time.monotonic()
        print(f"V9_CONSTRUCT_START {ordinal}/169 {name}", flush=True)
        source = module.weight.detach()
        source_sha = v7.v3.tensor_sha256(source)
        state = construct_dual_codebook_state(source, name)
        geometry = _validate_state(state)
        reconstruction = _measure_and_materialize(module, source, state, sealed_records[name])
        raw = {
            "payload": state["payload"],
            "row_scale": state["scale_raw"],
            "dual_codebook": state["codebook_raw"],
            "selector": state["selector_raw"],
        }
        component_records = {
            component: {"module": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            for component, value in raw.items()
        }
        component_records["row_scale"]["record_count"] = int(source.shape[0])
        component_records["dual_codebook"].update({
            "input_block_count": state["input_block_count"],
            "bank_count": state["input_block_count"] * 2,
            "q4_4_banks_by_input_block": [[list(bank) for bank in banks] for banks in state["final_codebooks"]],
        })
        component_records["selector"].update({
            "row_group_size": state["row_group_size"],
            "row_group_count": state["row_group_count"],
            "selector_bit_count": state["row_group_count"] * state["input_block_count"],
            "alignment_bytes": ALIGNMENT_BYTES,
            "padding_bits_zero": True,
        })
        for component in ("payload", "row_scale", "dual_codebook", "selector"):
            manifests[component].append(component_records[component])
            streams[component].update(name.encode() + b"\0" + raw[component])
            totals[component] += len(raw[component])
        streams["reconstruction"].update(name.encode() + b"\0" + bytes.fromhex(reconstruction["reconstruction_bf16_sha256"]))
        manifests["reconstruction"].append(reconstruction)
        objective_record = {
            "module": name,
            "kind": "constant_elided_optimizer_ordering_cost_not_literal_sse",
            "initial_v7_optimizer_cost": state["initial_v7_objective"],
            "final_v9_optimizer_cost": state["final_objective"],
            "actual_nonnegative_bf16_sse": {
                "fresh_v7": reconstruction["fresh_v7_bf16_sse"],
                "v9": reconstruction["v9_dual_codebook_bf16_sse"],
            },
        }
        manifests["objective"].append(objective_record)
        trace_record = {
            "module": name,
            "source_bf16_sha256": source_sha,
            "shift_min": state["fresh_v7_state"]["shift_min"],
            "shift_max": state["fresh_v7_state"]["shift_max"],
            "row_group_size": state["row_group_size"],
            "input_block_count": state["input_block_count"],
            "fresh_v7_exact_reproduction": state["fresh_v7_exact_reproduction"],
            "bank_b_initialization": state["bank_b_initialization"],
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
            "metadata_sequence": ["dual_codebook_bank_a", "dual_codebook_bank_b", "selector", "payload", "row_scale"],
        }
        weight_manifest.append(record)
        totals["weight_count"] += int(source.numel())
        print(
            f"V9_CONSTRUCT_DONE {ordinal}/169 {name} seconds={time.monotonic()-started:.3f} "
            f"v7_sse={reconstruction['fresh_v7_bf16_sse']:.12g} v9_sse={reconstruction['v9_dual_codebook_bf16_sse']:.12g}",
            flush=True,
        )
        del state
        gc.collect()

    aggregate_v7 = float(sum(item["fresh_v7_bf16_sse"] for item in manifests["reconstruction"]))
    aggregate_v9 = float(sum(item["v9_dual_codebook_bf16_sse"] for item in manifests["reconstruction"]))
    fresh_count = sum(bool(item["fresh_v7_matches_sealed"]) for item in manifests["reconstruction"])
    non_regressing_count = sum(bool(item["v9_non_regressing_against_fresh_v7"]) for item in manifests["reconstruction"])
    improved_count = sum(bool(item["v9_strictly_improved_against_fresh_v7"]) for item in manifests["reconstruction"])
    require(aggregate_v7 == V7_AGGREGATE_BF16_SSE, "fresh aggregate V7 BF16 SSE differs")
    require(all(item["fresh_v7_bf16_sse"] >= 0.0 and item["v9_dual_codebook_bf16_sse"] >= 0.0 for item in manifests["reconstruction"]), "V9 model report contains negative BF16 SSE")
    reconstruction_body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "measurement": "literal nonnegative sum-squared error from pinned BF16 source to BF16-materialized reconstruction",
        "optimizer_cost_note": "scaled_integer optimizer fields omit the source-squared constant and are not reported as SSE",
        "reference": {
            "candidate_id": v7.CANDIDATE_ID,
            "artifact": file_record(V7_RECONSTRUCTION_PATH),
            "aggregate_bf16_sse": V7_AGGREGATE_BF16_SSE,
            "fresh_reconstruction_required": True,
        },
        "tensor_count": len(manifests["reconstruction"]),
        "fresh_v7_exact_reproduction_count": fresh_count,
        "per_tensor_non_regressing_count": non_regressing_count,
        "per_tensor_strictly_improved_count": improved_count,
        "all_169_fresh_v7_reproductions_match": fresh_count == 169,
        "all_169_tensors_non_regressing_against_fresh_v7": non_regressing_count == 169,
        "aggregate_fresh_v7_bf16_sse": aggregate_v7,
        "aggregate_v9_dual_codebook_bf16_sse": aggregate_v9,
        "aggregate_strictly_improved_against_fresh_v7": aggregate_v9 < aggregate_v7,
        "tensors": manifests["reconstruction"],
    }
    passed = (
        reconstruction_body["all_169_fresh_v7_reproductions_match"]
        and reconstruction_body["all_169_tensors_non_regressing_against_fresh_v7"]
        and reconstruction_body["aggregate_strictly_improved_against_fresh_v7"]
    )
    reconstruction_report = {
        **reconstruction_body,
        "status": "PASS_FRESH_V7_AND_169_OF_169_WITH_STRICT_AGGREGATE_IMPROVEMENT" if passed else "NO_GO_V7_RELATIVE_MODEL_ONLY_RECONSTRUCTION_GATE",
        "reconstruction_report_sha256": canonical_sha256(reconstruction_body),
    }
    require(
        totals == {
            "weight_count": 493961216,
            "payload": 246980608,
            "row_scale": 1824256,
            "dual_codebook": 61664,
            "selector": 302528,
        },
        f"whole-model V9 storage geometry differs: {totals}",
    )
    hashes = {f"{name}_manifest_sha256": canonical_sha256(values) for name, values in manifests.items()}
    hashes.update({"weight_manifest_sha256": canonical_sha256(weight_manifest), "reconstruction_report_sha256": reconstruction_report["reconstruction_report_sha256"]})
    hashes.update({f"{name}_stream_sha256": digest.hexdigest() for name, digest in streams.items()})
    storage = {
        "total_weight_count": totals["weight_count"],
        "total_payload_bytes": totals["payload"],
        "total_row_scale_bytes": totals["row_scale"],
        "total_dual_codebook_bytes": totals["dual_codebook"],
        "total_selector_bytes": totals["selector"],
        "whole_model_total_weight_bytes": sum(totals[name] for name in ("payload", "row_scale", "dual_codebook", "selector")),
        "whole_model_exact_bits_per_weight": 4.035443236094066,
        "standard_decoder_layer_exact_bits_per_weight": 4.035242101648351,
        "lm_head_exact_bits_per_weight": 4.035971912985919,
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
    require(model.model.embed_tokens.weight is embedding and embedding.data_ptr() == source_pointer, "embedding identity changed after V9 construction")
    require(list(embedding.shape) == source_shape and str(embedding.dtype) == source_dtype, "embedding shape or dtype changed")
    embedding_hash = v7.v3.tensor_sha256(embedding)
    require(embedding_hash == source_hash == SOURCE_EMBEDDING_SHA256, "embedding bytes changed after V9 construction")
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
        "post_v9": {
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
    manifest_fields = tuple(f"{name}_manifest" for name in ("payload", "row_scale", "dual_codebook", "selector", "reconstruction", "objective", "iteration_trace")) + ("weight_manifest", "state_manifest")
    field_matches = {field: candidate_a[field] == candidate_b[field] for field in manifest_fields}
    require(all(field_matches.values()), f"candidate A/B whole-model manifests differ: {field_matches}")
    return {
        "state_match_keys": list(STATE_MATCH_KEYS),
        "identity_comparisons": comparisons,
        "manifest_byte_equivalent_fields": list(manifest_fields),
        "manifest_comparisons": field_matches,
        "all_match": True,
    }


def _validate_reconstruction(report: dict[str, Any]) -> bool:
    require(report["tensor_count"] == len(report["tensors"]) == 169, "V9 reconstruction tensor count differs")
    require(report["fresh_v7_exact_reproduction_count"] == sum(bool(item["fresh_v7_matches_sealed"]) for item in report["tensors"]), "V9 fresh V7 reproduction count differs")
    require(report["per_tensor_non_regressing_count"] == sum(bool(item["v9_non_regressing_against_fresh_v7"]) for item in report["tensors"]), "V9 non-regressing count differs")
    require(report["aggregate_fresh_v7_bf16_sse"] == V7_AGGREGATE_BF16_SSE, "V9 aggregate V7 reference differs")
    require(all(item["fresh_v7_bf16_sse"] >= 0.0 and item["v9_dual_codebook_bf16_sse"] >= 0.0 for item in report["tensors"]), "V9 reconstruction report contains negative BF16 SSE")
    passed = (
        report["all_169_fresh_v7_reproductions_match"] is True
        and report["all_169_tensors_non_regressing_against_fresh_v7"] is True
        and report["aggregate_strictly_improved_against_fresh_v7"] is True
    )
    require((report["status"].startswith("PASS_")) == passed, "V9 reconstruction status differs")
    body = {key: value for key, value in report.items() if key not in {"status", "reconstruction_report_sha256"}}
    require(report["reconstruction_report_sha256"] == canonical_sha256(body), "V9 reconstruction report hash differs")
    return passed


def _validate_quality(report: dict[str, Any]) -> bool:
    return v7.v3.validate_quality(report)


def _validate_dynamic(report: dict[str, Any]) -> bool:
    passed = v7.v3.validate_dynamic(report)
    activation = report["activation_execution"]
    require(activation["named_boundary_count"] == EXPECTED_EVENTS, "V9 named Dynamic Scale32 boundary count differs")
    require(activation["transport_metadata_bytes_per_decoder_layer_token"] <= 832, "V9 activation metadata cap differs")
    return passed


def _audit_attempt_namespace(expect_root_absent: bool) -> dict[str, Any]:
    root_existed = OUTPUT_DIR.exists()
    if expect_root_absent:
        require(not root_existed, "V9 namespace existed before fresh pre-attempt preparation")
    markers = list(OUTPUT_DIR.rglob("execution_started.json")) if root_existed else []
    require(not markers and not ATTEMPT_DIR.exists(), "V9 official attempt or marker exists")
    return {
        "candidate_id": CANDIDATE_ID,
        "root_existed_at_audit": root_existed,
        "attempt_directories": [],
        "matching_execution_markers": [],
        "authorization_consumed": False,
    }


def _recover_nonqualifying_preparation() -> dict[str, Any]:
    if not OUTPUT_DIR.exists():
        return {"recovered": False, "fresh_namespace": True, "archived_artifacts": []}
    complete = PREATTEMPT_DIR.is_dir() and CONTRACT_PATH.is_file() and (READY_PATH.is_file() or NO_GO_PATH.is_file())
    if complete:
        raise RuntimeError("sealed V9 pre-attempt closure already exists; use verify")
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "cannot recover a V9 namespace with an official attempt")
    candidates = [path for path in OUTPUT_DIR.iterdir() if path.name != "preflight-failures"]
    require(candidates, "existing V9 namespace has ambiguous provenance")
    archive_root = OUTPUT_DIR / "preflight-failures"
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / f"recovered-{time.time_ns()}"
    destination.mkdir()
    archived: list[str] = []
    for path in candidates:
        target = destination / path.name
        path.rename(target)
        archived.append(target.relative_to(ROOT).as_posix())
    return {"recovered": True, "fresh_namespace": False, "archived_artifacts": archived}


def _artifact_records() -> dict[str, Any]:
    paths = {
        "engineer_task": TASK_PATH,
        "engineer_task_companion": TASK_COMPANION,
        "planning_requirements": PLAN_PATH,
        "planning_requirements_companion": PLAN_COMPANION,
        "source_contract": SOURCE_CONTRACT_PATH,
        "calibration_contract": CALIBRATION_CONTRACT_PATH,
        "calibration_report": CALIBRATION_REPORT_PATH,
        "base_scales": BASE_SCALES_PATH,
        "image_manifest": IMAGE_DIR / "manifest.json",
        "fresh_v7_reconstruction_reference": V7_RECONSTRUCTION_PATH,
        "official_matrix_hash_only": MATRIX_PATH,
        "runner_source": RUNNER_PATH,
        "scalable_backend_source": BACKEND_PATH,
        "native_exact_assignment_source": NATIVE_SOURCE_PATH,
        "native_exact_assignment_library": NATIVE_LIBRARY_PATH,
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
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "V9 Engineer authority differs")
    require(task["fresh_attempt_namespace"]["attempt_budget"] == 1 and task["fresh_attempt_namespace"]["attempts_consumed_at_task_freeze"] == 0, "V9 attempt accounting differs")
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
            "attempt_id": "attempt-0001",
            "exact_authorized_candidate_attempts": 1,
            "attempts_consumed": 0,
            "selected_policy_id": None,
        },
        "source_model": source_identity,
        "environment": environment,
        "format_and_policy": _policy_and_format(),
        "construction": {
            "backend": "fresh exact V7 reconstruction plus exact native selector/payload/codebook aggregation and Python-integer Scale32/codebook optimization",
            "source_tied_then_runtime_split_lm_head": True,
            "source_embedding_bf16_sha256": SOURCE_EMBEDDING_SHA256,
            "embedding_bytes_preserved": True,
            "transformer_linears": 168,
            "separated_lm_head_linears": 1,
            "outer_iterations": OUTER_ITERATIONS,
            "candidate_a_and_b_whole_model_manifests_match": state_comparison,
            "optimizer_cost_is_not_literal_sse": True,
            "model_gate_uses_actual_nonnegative_bf16_sse": True,
        },
        "pre_attempt_gates": {
            "fresh_v7_and_169_model_closure": {"passed": gate_passes["fresh_v7_and_169_model_closure"], "artifact": artifacts["model_only_reconstruction"]},
            "non_evaluator_quality_8_of_8": {"passed": gate_passes["non_evaluator_quality_8_of_8"], "artifact": artifacts["non_evaluator_quality"]},
            "dynamic_312_event_closure": {"passed": gate_passes["dynamic_312_event_closure"], "artifact": artifacts["dynamic_preflight"]},
            "two_build_alias_and_manifest_closure": {"passed": gate_passes["two_build_alias_and_manifest_closure"], "artifact": artifacts["alias_separation_witness"]},
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
            "mode_exposed_by_runner": False,
            "attempt_directory_exists": False,
            "execution_marker_exists": False,
            "conditional_execution_rule": "create attempt-0001 only after this exact PREATTEMPT_READY closure verifies",
        },
        "namespace_audit": namespace_audit,
        "artifacts": artifacts,
        "claim_boundary": {
            "attempts_consumed": 0,
            "numerical_conclusion": None,
            "selected_policy_id": None,
            "rtl_or_ppa_claimed": False,
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


def load_verified_closure() -> dict[str, Any]:
    _configure_quality_helpers()
    reference.require_project_python()
    require(CONTRACT_PATH.is_file(), "V9 predeclared contract is missing")
    require(READY_PATH.is_file() != NO_GO_PATH.is_file(), "exactly one V9 pre-attempt terminal record is required")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "V9 predeclared contract wrapper differs")
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "V9 predeclared contract canonical hash differs")
    contract = wrapper["contract"]
    require(contract["candidate_id"] == CANDIDATE_ID and contract["task_id"] == TASK_ID, "V9 predeclared authority differs")
    require(contract["artifacts"]["runner_source"]["sha256"] == reference.sha256_file(RUNNER_PATH), "V9 runner changed after freeze")
    require(contract["artifacts"]["scalable_backend_source"]["sha256"] == reference.sha256_file(BACKEND_PATH), "V9 backend changed after freeze")
    require(contract["artifacts"]["native_exact_assignment_source"]["sha256"] == reference.sha256_file(NATIVE_SOURCE_PATH), "V9 native source changed after freeze")
    for record in contract["artifacts"].values():
        _verify_bound_record(record)
    reference.verify_companion(TASK_PATH, TASK_COMPANION)
    reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    require(reference.sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(reference.sha256_file(V7_RECONSTRUCTION_PATH) == V7_RECONSTRUCTION_SHA256, "sealed V7 reconstruction hash differs")
    require(verify_source_snapshot() == contract["source_model"], "source snapshot identity differs")
    packages = {**verify_versions(), "numpy": importlib.metadata.version("numpy")}
    require(packages == contract["environment"]["packages"], "package versions differ")
    require(combined_self_test() == load_json(SELF_TEST_PATH), "V9 runner self-test result differs")
    candidate_a = load_json(CANDIDATE_A_PATH)
    candidate_b = load_json(CANDIDATE_B_PATH)
    comparison = _compare_candidates(candidate_a, candidate_b)
    witness = load_json(ALIAS_WITNESS_PATH)
    require(witness["state_comparison"] == comparison, "V9 alias witness comparison differs")
    gate_passes = {
        "fresh_v7_and_169_model_closure": _validate_reconstruction(load_json(RECONSTRUCTION_PATH)),
        "non_evaluator_quality_8_of_8": _validate_quality(load_json(QUALITY_PATH)),
        "dynamic_312_event_closure": _validate_dynamic(load_json(DYNAMIC_PATH)),
        "two_build_alias_and_manifest_closure": comparison["all_match"] and witness["embedding_preserved_in_both"] and witness["lm_head_separate_in_both"],
    }
    require(contract["pre_attempt_gates"]["all_pass"] == all(gate_passes.values()), "V9 contract aggregate gate status differs")
    for name, passed in gate_passes.items():
        require(contract["pre_attempt_gates"][name]["passed"] == passed, f"V9 contract gate differs: {name}")
    terminal_path = READY_PATH if READY_PATH.is_file() else NO_GO_PATH
    terminal = load_json(terminal_path)
    require(terminal["candidate_id"] == CANDIDATE_ID, "V9 terminal candidate differs")
    require(terminal["contract"]["sha256"] == reference.sha256_file(CONTRACT_PATH), "V9 terminal contract file hash differs")
    require(terminal["contract_sha256"] == wrapper["contract_sha256"], "V9 terminal contract canonical hash differs")
    require(terminal["bound_artifacts"] == contract["artifacts"], "V9 terminal artifact snapshot differs")
    require(terminal["official_execution_marker_exists"] is False and terminal["attempts_consumed"] == 0, "V9 terminal record claims an official attempt")
    require((terminal_path == READY_PATH) == all(gate_passes.values()), "V9 terminal disposition differs from gates")
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "V9 official attempt exists after marker-free closure")
    quality = load_json(QUALITY_PATH)
    dynamic = load_json(DYNAMIC_PATH)
    reconstruction = load_json(RECONSTRUCTION_PATH)
    return {
        "status": terminal["status"],
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": wrapper["contract_sha256"],
        "gate_passes": gate_passes,
        "fresh_v7_exact_reproduction_count": reconstruction["fresh_v7_exact_reproduction_count"],
        "per_tensor_non_regressing_count": reconstruction["per_tensor_non_regressing_count"],
        "aggregate_fresh_v7_bf16_sse": reconstruction["aggregate_fresh_v7_bf16_sse"],
        "aggregate_v9_dual_codebook_bf16_sse": reconstruction["aggregate_v9_dual_codebook_bf16_sse"],
        "exact_sequence_matches": quality["exact_bf16_sequence_match_count"],
        "gate_outcome_matches": quality["response_gate_outcome_match_count"],
        "event_count": dynamic["activation_execution"]["event_count"],
        "attempts_consumed": 0,
        "official_execution_mode_exposed": False,
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
        "backend": "exact native selector/payload/codebook aggregation with Python-integer Scale32/codebook optimization",
        "native_build": _native_build_identity(),
    }
    recovery = _recover_nonqualifying_preparation()
    namespace_audit = _audit_attempt_namespace(expect_root_absent=recovery["fresh_namespace"])
    namespace_audit["preparation_recovery"] = recovery
    reference.verify_companion(TASK_PATH, TASK_COMPANION)
    reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    verify_source_contract()
    source_identity = verify_source_snapshot()
    require(reference.sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(reference.sha256_file(V7_RECONSTRUCTION_PATH) == V7_RECONSTRUCTION_SHA256, "sealed V7 reconstruction hash differs")
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
    require(probe.read_bytes() == b"preflight\n", "V9 output-path readback differs")
    probe.unlink()

    chronology: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "preparation_started_at_utc": utc_now(),
        "namespace_audit": namespace_audit,
        "official_matrix_content_accessed": False,
        "output_path_preflight": {
            "exclusive_create": True,
            "readback": True,
            "collision_behavior": "O_EXCL",
            "execution_marker_created": False,
        },
    }
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    inputs, input_manifest = v7.v3.non_evaluator_inputs(tokenizer)

    chronology["candidate_a_started_at_utc"] = utc_now()
    print("V9_CANDIDATE_A_START", flush=True)
    candidate_a, manifest_a, witness_a, reconstruction_a = _load_candidate("candidate_a_independent_construction")
    chronology["candidate_a_complete_at_utc"] = utc_now()
    print("V9_CANDIDATE_A_DONE", flush=True)
    del candidate_a
    gc.collect()

    chronology["candidate_b_started_at_utc"] = utc_now()
    print("V9_CANDIDATE_B_START", flush=True)
    candidate_b, manifest_b, witness_b, reconstruction_b = _load_candidate("candidate_b_independent_quality_and_dynamic_preflight")
    state_comparison = _compare_candidates(manifest_a, manifest_b)
    require(reconstruction_a == reconstruction_b, "V9 candidate A/B reconstruction reports differ")
    chronology["candidate_b_constructed_at_utc"] = utc_now()
    print("V9_CANDIDATE_B_CONSTRUCTED quality_dynamic_start", flush=True)
    torch.set_num_threads(TORCH_THREADS)
    quality, dynamic = v7.v3.non_evaluator_quality_and_dynamic(candidate_b, tokenizer, inputs, input_manifest)
    chronology["candidate_b_quality_and_dynamic_complete_at_utc"] = utc_now()
    print("V9_QUALITY_DYNAMIC_DONE", flush=True)
    del candidate_b
    gc.collect()

    gate_passes = {
        "fresh_v7_and_169_model_closure": _validate_reconstruction(reconstruction_b),
        "non_evaluator_quality_8_of_8": _validate_quality(quality),
        "dynamic_312_event_closure": _validate_dynamic(dynamic),
        "two_build_alias_and_manifest_closure": (
            state_comparison["all_match"]
            and witness_a["post_v9"]["embedding_hash_matches_source"]
            and witness_b["post_v9"]["embedding_hash_matches_source"]
            and witness_a["post_v9"]["lm_head_storage_is_separate"]
            and witness_b["post_v9"]["lm_head_storage_is_separate"]
        ),
    }
    require(not MARKER_PATH.exists() and not ATTEMPT_DIR.exists(), "V9 official attempt appeared before artifact freeze")

    PREATTEMPT_DIR.mkdir(parents=False, exist_ok=False)
    write_json(INPUT_MANIFEST_PATH, input_manifest)
    write_json(CANDIDATE_A_PATH, manifest_a)
    write_json(CANDIDATE_B_PATH, manifest_b)
    write_json(ALIAS_WITNESS_PATH, {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "candidate_a": witness_a,
        "candidate_b": witness_b,
        "state_comparison": state_comparison,
        "embedding_preserved_in_both": witness_a["post_v9"]["embedding_hash_matches_source"] and witness_b["post_v9"]["embedding_hash_matches_source"],
        "lm_head_separate_in_both": witness_a["post_v9"]["lm_head_storage_is_separate"] and witness_b["post_v9"]["lm_head_storage_is_separate"],
    })
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
        "actual_nonnegative_bf16_sse": {
            "fresh_v7": reconstruction_b["aggregate_fresh_v7_bf16_sse"],
            "v9": reconstruction_b["aggregate_v9_dual_codebook_bf16_sse"],
        },
        "attempt_budget": 1,
        "attempts_consumed": 0,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": False,
        "disposition": (
            "all marker-free V9 gates pass; PREATTEMPT_READY is sealed and the single official attempt may now be consumed"
            if all_pass
            else "preserve V9 no-go evidence, leave attempt-0001 absent and unconsumed, keep selected_policy_id null, and submit for Fresh Review"
        ),
    }
    atomic_write_json(terminal_path, terminal)
    verified = load_verified_closure()
    for path in list(PREATTEMPT_DIR.iterdir()) + [CONTRACT_PATH, terminal_path]:
        if path.is_file():
            path.chmod(0o444)
    return verified


def record_preparation_failure(exc: Exception) -> None:
    if MARKER_PATH.exists():
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failure = OUTPUT_DIR / f"preparation_failure_{time.time_ns()}.json"
    write_json(failure, {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "classification": "PREATTEMPT_FAILURE",
        "failure_taxonomy": "PREATTEMPT_EXECUTION_FAILURE",
        "root_cause_hypothesis": f"runner raised {type(exc).__name__}: {exc}",
        "regression": "repair only the isolated marker-free V9 runner and rerun pre-attempt preparation; official authorization remains unconsumed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "runner_sha256": reference.sha256_file(RUNNER_PATH),
        "backend_sha256": reference.sha256_file(BACKEND_PATH),
        "native_source_sha256": reference.sha256_file(NATIVE_SOURCE_PATH),
        "traceback": traceback.format_exc(),
        "attempt_authorization_consumed": False,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": False,
        "failed_at_utc": utc_now(),
    })
