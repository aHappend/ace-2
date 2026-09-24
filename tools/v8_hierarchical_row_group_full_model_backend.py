#!/usr/bin/env python3
"""Exact marker-free full-model backend for the V8 hierarchical W4 format."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import importlib.metadata
import json
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

import run_option_b_alias_safe_hierarchical_row_group_scale_w4_grouped_dynamic_scale32_stage1 as reference
import v4_joint_full_model_backend as v4
import v7_input_block_full_model_backend as v7
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
NATIVE_SOURCE_PATH = ROOT / "tools/v8_hierarchical_exact_assignment.cpp"
NATIVE_BUILD_DIR = ROOT / "build/v8-hierarchical-native-exact-assignment"
NATIVE_LIBRARY_PATH = NATIVE_BUILD_DIR / "libv8_hierarchical_exact_assignment.so"
OUTPUT_DIR = reference.OFFICIAL_ROOT
MARKER_PATH = reference.OFFICIAL_MARKER
PREATTEMPT_DIR = OUTPUT_DIR / "preattempt"
INPUT_MANIFEST_PATH = PREATTEMPT_DIR / "non_evaluator_inputs.json"
CANDIDATE_A_PATH = PREATTEMPT_DIR / "candidate_a_manifest.json"
CANDIDATE_B_PATH = PREATTEMPT_DIR / "candidate_b_manifest.json"
ALIAS_WITNESS_PATH = PREATTEMPT_DIR / "alias_separation_witness.json"
RECONSTRUCTION_PATH = PREATTEMPT_DIR / "model_only_hierarchical_reconstruction.json"
QUALITY_PATH = PREATTEMPT_DIR / "non_evaluator_quality.json"
DYNAMIC_PATH = PREATTEMPT_DIR / "dynamic_preflight.json"
SELF_TEST_PATH = PREATTEMPT_DIR / "runner_self_test.json"
CHRONOLOGY_PATH = PREATTEMPT_DIR / "preparation_chronology.json"
RECOVERY_PATH = PREATTEMPT_DIR / "preparation_recovery.json"
CONTRACT_PATH = OUTPUT_DIR / "predeclared_contract.json"
READY_PATH = OUTPUT_DIR / "PREATTEMPT_READY.json"
NO_GO_PATH = OUTPUT_DIR / "PREATTEMPT_NO_GO.json"
ATTEMPT_DIR = OUTPUT_DIR / "attempt-0001"

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

WEIGHT_GROUP_LANES = reference.WEIGHT_GROUP_LANES
TABLE_ENTRIES = reference.TABLE_ENTRIES
SIGNED_INT4_MIN = reference.SIGNED_INT4_MIN
SIGNED_INT4_MAX = reference.SIGNED_INT4_MAX
GROUP_DELTA_MIN = reference.GROUP_DELTA_MIN
GROUP_DELTA_MAX = reference.GROUP_DELTA_MAX
OUTER_ITERATIONS = reference.OUTER_ITERATIONS
LEGAL_RECORD_COUNT = (SCALE32_EXPONENT_MAX - SCALE32_EXPONENT_MIN + 1) * (
    SCALE32_SIGNIFICAND_MAX - SCALE32_SIGNIFICAND_MIN + 1
)
TORCH_THREADS = 32
EXPECTED_EVENTS = v7.grouped.LAYERS * len(v7.grouped.EVENT_FAMILIES)
STATE_MATCH_KEYS = (
    "embedding_bf16_sha256",
    "payload_manifest_sha256",
    "row_selector_manifest_sha256",
    "group_delta_manifest_sha256",
    "row_base_table_manifest_sha256",
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
    require(compiler is not None, "g++ is required for the exact V8 assignment backend")
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
    return {
        "source_sha256": reference.sha256_file(NATIVE_SOURCE_PATH),
        "compiler": compiler,
        "compiler_version": subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0],
        "command": command,
    }


def _load_native_library() -> ctypes.CDLL:
    global _NATIVE_LIBRARY
    if _NATIVE_LIBRARY is not None:
        return _NATIVE_LIBRARY
    identity = _native_build_identity()
    metadata_path = NATIVE_BUILD_DIR / "build_identity.json"
    rebuild = True
    if NATIVE_LIBRARY_PATH.is_file() and metadata_path.is_file():
        metadata = load_json(metadata_path)
        rebuild = metadata.get("source_sha256") != identity["source_sha256"]
    if rebuild:
        NATIVE_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        temporary = NATIVE_BUILD_DIR / f"libv8_hierarchical_exact_assignment.{os.getpid()}.tmp.so"
        command = [*identity["command"][:-1], str(temporary)]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        require(completed.returncode == 0, f"V8 native backend compilation failed: {completed.stderr}")
        os.replace(temporary, NATIVE_LIBRARY_PATH)
        metadata = {
            **identity,
            "library_sha256": reference.sha256_file(NATIVE_LIBRARY_PATH),
        }
        temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
        temporary_metadata.write_bytes(reference.canonical_bytes(metadata))
        os.replace(temporary_metadata, metadata_path)
    library = ctypes.CDLL(str(NATIVE_LIBRARY_PATH))
    function = library.v8_assign_exact
    function.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int8),
    ]
    function.restype = ctypes.c_int
    _NATIVE_LIBRARY = library
    return library


def _record_key(record: int) -> tuple[int, int, int]:
    significand, exponent = unpack_scale32(int(record))
    return exponent, significand, int(record)


def _record_ordinal(record: int) -> int:
    significand, exponent = unpack_scale32(int(record))
    return (exponent - SCALE32_EXPONENT_MIN) * (SCALE32_SIGNIFICAND_MAX - SCALE32_SIGNIFICAND_MIN + 1) + (
        significand - SCALE32_SIGNIFICAND_MIN
    )


def _ordinal_record(ordinal: int) -> int:
    require(0 <= ordinal < LEGAL_RECORD_COUNT, "Scale32 ordinal escaped legal range")
    span = SCALE32_SIGNIFICAND_MAX - SCALE32_SIGNIFICAND_MIN + 1
    exponent_offset, significand_offset = divmod(ordinal, span)
    return pack_scale32(SCALE32_SIGNIFICAND_MIN + significand_offset, SCALE32_EXPONENT_MIN + exponent_offset)


def _group_ranges(width: int) -> tuple[tuple[int, int], ...]:
    return reference.weight_group_ranges(width)


def _pack_signed_nibbles(values: Tensor | Sequence[int]) -> bytes:
    flat = [int(value) for value in (values.reshape(-1).tolist() if isinstance(values, Tensor) else values)]
    require(all(SIGNED_INT4_MIN <= value <= SIGNED_INT4_MAX for value in flat), "signed int4 value escaped range")
    if len(flat) & 1:
        flat.append(0)
    return bytes((flat[index] & 0xF) | ((flat[index + 1] & 0xF) << 4) for index in range(0, len(flat), 2))


def _unpack_signed_nibbles(raw: bytes, count: int) -> tuple[int, ...]:
    values: list[int] = []
    for byte in raw:
        for nibble in (byte & 0xF, byte >> 4):
            values.append(nibble - 16 if nibble >= 8 else nibble)
    return tuple(values[:count])


def _pack_unsigned_nibbles(values: Tensor | Sequence[int]) -> bytes:
    flat = [int(value) for value in (values.reshape(-1).tolist() if isinstance(values, Tensor) else values)]
    require(all(0 <= value < TABLE_ENTRIES for value in flat), "row selector escaped four bits")
    if len(flat) & 1:
        flat.append(0)
    return bytes(flat[index] | (flat[index + 1] << 4) for index in range(0, len(flat), 2))


def _round_ratio_tie_smaller(numerator: Tensor, denominator: int) -> Tensor:
    require(numerator.dtype == torch.int64 and denominator > 0, "exact assignment ratio differs")
    floor = torch.div(numerator, denominator, rounding_mode="floor")
    remainder = numerator - floor * denominator
    increment = (remainder * 2 > denominator).to(torch.int64)
    return floor + increment


def _payload_for_effective_scale(exact_source: Tensor, base_shift: int, record: int, delta: int) -> tuple[Tensor, int]:
    significand, exponent = unpack_scale32(record)
    effective_exponent = exponent + delta
    require(GROUP_DELTA_MIN <= delta <= GROUP_DELTA_MAX, "group delta escaped signed four bits")
    require(SCALE32_EXPONENT_MIN <= effective_exponent <= SCALE32_EXPONENT_MAX, "effective Scale32 exponent escaped range")
    scale_shift = effective_exponent - 15
    if base_shift <= scale_shift:
        numerator = exact_source
        denominator = significand << (scale_shift - base_shift)
    else:
        numerator = torch.bitwise_left_shift(exact_source, base_shift - scale_shift)
        denominator = significand
    payload = torch.clamp(
        _round_ratio_tie_smaller(numerator, denominator),
        min=SIGNED_INT4_MIN,
        max=SIGNED_INT4_MAX,
    ).to(torch.int8)
    objective_exponent = 2 * min(base_shift, SCALE32_EXPONENT_MIN - 15)
    linear = int(torch.sum(exact_source * payload.to(torch.int64)))
    quadratic = int(torch.sum(payload.to(torch.int64) ** 2))
    linear_shift = base_shift + scale_shift - objective_exponent
    quadratic_shift = 2 * scale_shift - objective_exponent
    require(linear_shift >= 0 and quadratic_shift >= 0, "assignment objective exponent differs")
    variable_cost = ((-2 * significand * linear) << linear_shift) + (
        (significand * significand * quadratic) << quadratic_shift
    )
    return payload, variable_cost


def _initial_table(seed_records: Sequence[int]) -> tuple[int, ...]:
    require(bool(seed_records), "row-base seed list is empty")
    ordered = sorted((int(record) for record in seed_records), key=_record_key)
    targets = [ordered[min(len(ordered) - 1, ((2 * index + 1) * len(ordered)) // (2 * TABLE_ENTRIES))] for index in range(TABLE_ENTRIES)]
    target_ordinals = [_record_ordinal(record) for record in targets]
    candidate_sets: list[list[int]] = []
    for target in target_ordinals:
        candidates = {
            ordinal
            for distance in range(TABLE_ENTRIES + 1)
            for ordinal in (target - distance, target + distance)
            if 0 <= ordinal < LEGAL_RECORD_COUNT
        }
        candidate_sets.append(sorted(candidates))

    states: dict[int, tuple[int, tuple[int, ...]]] = {
        ordinal: (abs(ordinal - target_ordinals[0]), (ordinal,)) for ordinal in candidate_sets[0]
    }
    for index, candidates in enumerate(candidate_sets[1:], start=1):
        current: dict[int, tuple[int, tuple[int, ...]]] = {}
        previous = sorted(states.items())
        best: tuple[int, tuple[int, ...]] | None = None
        cursor = 0
        for ordinal in candidates:
            while cursor < len(previous) and previous[cursor][0] < ordinal:
                candidate = previous[cursor][1]
                if best is None or candidate < best:
                    best = candidate
                cursor += 1
            if best is not None:
                current[ordinal] = (best[0] + abs(ordinal - target_ordinals[index]), best[1] + (ordinal,))
        require(bool(current), "initial row-base table duplicate repair has no strict path")
        states = current
    _, ordinals = min(states.values())
    table = tuple(_ordinal_record(ordinal) for ordinal in ordinals)
    require(all(_record_key(left) < _record_key(right) for left, right in zip(table, table[1:])), "initial row-base table is not strictly increasing")
    return table


def _tensor_exact_lattice(source: Tensor) -> tuple[Tensor, int, int, int]:
    shift_min, shift_max, subnormals = v4._tensor_shift_range(source)
    require(subnormals == 0, "pinned tensor contains BF16 subnormals")
    require(shift_max - shift_min <= 54, "BF16 tensor exponent span exceeds exact int64 backend")
    significands, shifts = v4._decode_bf16(source)
    deltas = torch.where(significands == 0, torch.zeros_like(shifts), shifts - shift_min)
    require(bool(torch.all((deltas >= 0) & (deltas <= 54))), "BF16 tensor lattice shift escaped range")
    exact = torch.bitwise_left_shift(significands, deltas)
    return exact, shift_min, shift_max, subnormals


def _assign_rows_groups_oracle(exact: Tensor, base_shift: int, table: Sequence[int]) -> tuple[Tensor, Tensor, Tensor, int, dict[str, Any]]:
    rows, width = exact.shape
    groups = _group_ranges(width)
    payload = torch.empty((rows, width), dtype=torch.int8)
    selectors = torch.empty(rows, dtype=torch.uint8)
    deltas = torch.empty((rows, len(groups)), dtype=torch.int8)
    total_cost = 0
    delta_histogram = {str(delta): 0 for delta in range(GROUP_DELTA_MIN, GROUP_DELTA_MAX + 1)}
    for row in range(rows):
        selector_candidates: list[tuple[int, int, list[Tensor], list[int]]] = []
        for selector, record in enumerate(table):
            _significand, exponent = unpack_scale32(record)
            group_payloads: list[Tensor] = []
            group_deltas: list[int] = []
            row_cost = 0
            legal = True
            for start, stop in groups:
                candidates: list[tuple[int, int, bytes, Tensor]] = []
                for delta in range(GROUP_DELTA_MIN, GROUP_DELTA_MAX + 1):
                    if not SCALE32_EXPONENT_MIN <= exponent + delta <= SCALE32_EXPONENT_MAX:
                        continue
                    assigned, cost = _payload_for_effective_scale(exact[row, start:stop], base_shift, record, delta)
                    candidates.append((cost, delta, _pack_signed_nibbles(assigned), assigned))
                if not candidates:
                    legal = False
                    break
                cost, delta, _raw, assigned = min(candidates, key=lambda item: (item[0], item[1], item[2]))
                row_cost += cost
                group_payloads.append(assigned)
                group_deltas.append(delta)
            if legal:
                selector_candidates.append((row_cost, selector, group_payloads, group_deltas))
        require(bool(selector_candidates), "row has no legal base-selector/group-delta assignment")
        row_cost, selector, group_payloads, group_deltas = min(selector_candidates, key=lambda item: (item[0], item[1]))
        selectors[row] = selector
        total_cost += row_cost
        for group_index, ((start, stop), assigned, delta) in enumerate(zip(groups, group_payloads, group_deltas, strict=True)):
            payload[row, start:stop] = assigned
            deltas[row, group_index] = delta
            delta_histogram[str(delta)] += 1
    return payload, selectors, deltas, total_cost, {
        "row_count": rows,
        "group_count": rows * len(groups),
        "groups_per_row": len(groups),
        "delta_histogram": delta_histogram,
    }


def _prepare_assignment_cache(exact: Tensor, base_shift: int) -> dict[str, Any]:
    rows, width = exact.shape
    groups = len(_group_ranges(width))
    common_shift = min(base_shift, SCALE32_EXPONENT_MIN - 15)
    source_shift = base_shift - common_shift
    require(source_shift >= 0, "V8 common assignment lattice differs")
    exact_common = torch.bitwise_left_shift(exact, source_shift).contiguous()
    maximum = int(torch.abs(exact_common).max())
    require(maximum <= (1 << 52), "V8 common assignment lattice exceeds exact int64 native bounds")
    grouped = exact_common.reshape(rows, groups, WEIGHT_GROUP_LANES)
    sorted_exact = torch.sort(grouped, dim=2).values.contiguous()
    return {
        "common_shift": common_shift,
        "source_shift": source_shift,
        "exact_common": exact_common,
        "sorted_exact": sorted_exact,
    }


def _assignment_statistics(
    cache: dict[str, Any],
    payload: Tensor,
    selectors: Tensor,
    deltas: Tensor,
    table: Sequence[int],
) -> tuple[int, list[dict[str, int]], dict[str, Any]]:
    exact_common = cache["exact_common"]
    common_shift = int(cache["common_shift"])
    source_shift = int(cache["source_shift"])
    rows, width = exact_common.shape
    groups = len(_group_ranges(width))
    linear_bins = [0] * (TABLE_ENTRIES * 16)
    quadratic_bins = [0] * (TABLE_ENTRIES * 16)
    group_bins = [0] * (TABLE_ENTRIES * 16)
    rows_per_chunk = max(1, 1_000_000 // width)
    for start in range(0, rows, rows_per_chunk):
        stop = min(rows, start + rows_per_chunk)
        exact_grouped = exact_common[start:stop].reshape(-1, groups, WEIGHT_GROUP_LANES)
        payload_grouped = payload[start:stop].to(torch.int64).reshape(-1, groups, WEIGHT_GROUP_LANES)
        maximum = int(torch.abs(exact_grouped).max())
        require(maximum * 8 * WEIGHT_GROUP_LANES < (1 << 63), "V8 group reduction may overflow int64")
        linear = torch.sum(exact_grouped * payload_grouped, dim=2)
        quadratic = torch.sum(payload_grouped * payload_grouped, dim=2)
        keys = selectors[start:stop].to(torch.int64)[:, None] * 16 + deltas[start:stop].to(torch.int64) + 8
        chunk_linear = torch.zeros(TABLE_ENTRIES * 16, dtype=torch.int64)
        chunk_quadratic = torch.zeros(TABLE_ENTRIES * 16, dtype=torch.int64)
        chunk_groups = torch.zeros(TABLE_ENTRIES * 16, dtype=torch.int64)
        chunk_linear.scatter_add_(0, keys.reshape(-1), linear.reshape(-1))
        chunk_quadratic.scatter_add_(0, keys.reshape(-1), quadratic.reshape(-1))
        chunk_groups.scatter_add_(0, keys.reshape(-1), torch.ones_like(keys).reshape(-1))
        for index, value in enumerate(chunk_linear.tolist()):
            linear_bins[index] += int(value)
        for index, value in enumerate(chunk_quadratic.tolist()):
            quadratic_bins[index] += int(value)
        for index, value in enumerate(chunk_groups.tolist()):
            group_bins[index] += int(value)

    row_counts = torch.bincount(selectors.to(torch.int64), minlength=TABLE_ENTRIES).tolist()
    statistics: list[dict[str, int]] = []
    objective = 0
    delta_histogram = {str(delta): 0 for delta in range(GROUP_DELTA_MIN, GROUP_DELTA_MAX + 1)}
    for selector, record in enumerate(table):
        significand, exponent = unpack_scale32(int(record))
        selector_linear = 0
        selector_quadratic = 0
        used_deltas: list[int] = []
        for delta in range(GROUP_DELTA_MIN, GROUP_DELTA_MAX + 1):
            index = selector * 16 + delta + 8
            linear = linear_bins[index]
            require(linear % (1 << source_shift) == 0, "V8 table statistic escaped source lattice")
            source_linear = linear >> source_shift
            quadratic = quadratic_bins[index]
            count = group_bins[index]
            if count:
                used_deltas.append(delta)
                delta_histogram[str(delta)] += count
            selector_linear += source_linear << (delta - GROUP_DELTA_MIN)
            selector_quadratic += quadratic << (2 * (delta - GROUP_DELTA_MIN))
            if linear or quadratic:
                scale_shift = exponent + delta - 15 - common_shift
                require(scale_shift >= 0, "V8 selected scale escaped common assignment lattice")
                scale = significand << scale_shift
                objective += -2 * scale * linear + scale * scale * quadratic
        statistics.append(
            {
                "row_count": int(row_counts[selector]),
                "linear": selector_linear,
                "quadratic": selector_quadratic,
                "minimum_delta": min(used_deltas) if used_deltas else GROUP_DELTA_MAX,
                "maximum_delta": max(used_deltas) if used_deltas else GROUP_DELTA_MIN,
            }
        )
    return objective, statistics, {
        "row_count": rows,
        "group_count": rows * groups,
        "groups_per_row": groups,
        "delta_histogram": delta_histogram,
        "assignment_backend": "native exact sorted-prefix Scale32 candidate evaluation",
        "objective_binary_exponent": 2 * common_shift,
    }


def _assign_rows_groups(
    exact: Tensor,
    base_shift: int,
    table: Sequence[int],
    cache: dict[str, Any] | None = None,
) -> tuple[Tensor, Tensor, Tensor, int, dict[str, Any], list[dict[str, int]], dict[str, Any]]:
    cache = _prepare_assignment_cache(exact, base_shift) if cache is None else cache
    exact_common = cache["exact_common"]
    sorted_exact = cache["sorted_exact"]
    rows, width = exact_common.shape
    groups = len(_group_ranges(width))
    significands: list[int] = []
    exponents: list[int] = []
    for record in table:
        significand, exponent = unpack_scale32(int(record))
        significands.append(significand)
        exponents.append(exponent)
    exact_numpy = exact_common.numpy()
    sorted_numpy = sorted_exact.numpy()
    significand_numpy = np.asarray(significands, dtype=np.int64)
    exponent_numpy = np.asarray(exponents, dtype=np.int8)
    payload_numpy = np.empty((rows, width), dtype=np.int8)
    selector_numpy = np.empty(rows, dtype=np.uint8)
    delta_numpy = np.empty((rows, groups), dtype=np.int8)
    library = _load_native_library()
    status = library.v8_assign_exact(
        exact_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        sorted_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        rows,
        groups,
        significand_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        exponent_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        int(cache["common_shift"]),
        TORCH_THREADS,
        payload_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        selector_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        delta_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    require(status == 0, f"native exact V8 assignment failed with status {status}")
    payload = torch.from_numpy(payload_numpy)
    selectors = torch.from_numpy(selector_numpy)
    deltas = torch.from_numpy(delta_numpy)
    objective, statistics, summary = _assignment_statistics(cache, payload, selectors, deltas, table)
    return payload, selectors, deltas, objective, summary, statistics, cache


def _cluster_statistics(exact: Tensor, payload: Tensor, selectors: Tensor, deltas: Tensor) -> list[dict[str, int]]:
    rows, width = exact.shape
    groups = _group_ranges(width)
    statistics = [
        {
            "row_count": 0,
            "linear": 0,
            "quadratic": 0,
            "minimum_delta": GROUP_DELTA_MAX,
            "maximum_delta": GROUP_DELTA_MIN,
        }
        for _ in range(TABLE_ENTRIES)
    ]
    for row in range(rows):
        selector = int(selectors[row])
        statistics[selector]["row_count"] += 1
        for group_index, (start, stop) in enumerate(groups):
            delta = int(deltas[row, group_index])
            q = payload[row, start:stop].to(torch.int64)
            statistics[selector]["minimum_delta"] = min(statistics[selector]["minimum_delta"], delta)
            statistics[selector]["maximum_delta"] = max(statistics[selector]["maximum_delta"], delta)
            statistics[selector]["linear"] += int(torch.sum(exact[row, start:stop] * q)) << (delta - GROUP_DELTA_MIN)
            statistics[selector]["quadratic"] += int(torch.sum(q * q)) << (2 * (delta - GROUP_DELTA_MIN))
    return statistics


def _table_record_cost(record: int, statistic: dict[str, int], base_shift: int) -> int:
    significand, exponent = unpack_scale32(record)
    linear_exponent = base_shift + GROUP_DELTA_MIN
    quadratic_exponent = 2 * GROUP_DELTA_MIN
    scale_shift = exponent - 15
    objective_exponent = min(
        base_shift + GROUP_DELTA_MIN + SCALE32_EXPONENT_MIN - 15,
        2 * GROUP_DELTA_MIN + 2 * (SCALE32_EXPONENT_MIN - 15),
    )
    linear_shift = linear_exponent + scale_shift - objective_exponent
    quadratic_shift = quadratic_exponent + 2 * scale_shift - objective_exponent
    require(linear_shift >= 0 and quadratic_shift >= 0, "table objective exponent differs")
    return ((-2 * significand * statistic["linear"]) << linear_shift) + (
        (significand * significand * statistic["quadratic"]) << quadratic_shift
    )


def _table_candidates(statistic: dict[str, int], prior_record: int, base_shift: int) -> tuple[int, ...]:
    if statistic["row_count"] == 0 or statistic["quadratic"] == 0:
        return (int(prior_record),)
    numerator = statistic["linear"]
    denominator = statistic["quadratic"]
    shift = base_shift - GROUP_DELTA_MIN
    require(numerator > 0 and denominator > 0, "nonempty selector has nonpositive ideal base scale")
    minimum = pack_scale32(SCALE32_SIGNIFICAND_MIN, SCALE32_EXPONENT_MIN)
    maximum = pack_scale32(SCALE32_SIGNIFICAND_MAX, SCALE32_EXPONENT_MAX)
    if v4._compare_record_to_dyadic(minimum, numerator, denominator, shift) >= 0:
        floor_record = ceil_record = minimum
    elif v4._compare_record_to_dyadic(maximum, numerator, denominator, shift) <= 0:
        floor_record = ceil_record = maximum
    else:
        ceil_record = v4._ceil_scale32_dyadic(numerator, denominator, shift)
        floor_record = ceil_record if v4._compare_record_to_dyadic(ceil_record, numerator, denominator, shift) == 0 else v4.reference.previous_scale32_record(ceil_record)
        require(floor_record is not None, "Scale32 ideal floor is missing")
    records = {int(prior_record), int(floor_record), int(ceil_record)}
    for record in (int(floor_record), int(ceil_record)):
        previous = v4.reference.previous_scale32_record(record)
        following = v4.reference.next_scale32_record(record)
        if previous is not None:
            records.add(previous)
        if following is not None:
            records.add(following)
    minimum_delta = statistic["minimum_delta"]
    maximum_delta = statistic["maximum_delta"]
    filtered = tuple(
        sorted(
            (
                record
                for record in records
                if SCALE32_EXPONENT_MIN <= unpack_scale32(record)[1] + minimum_delta
                and unpack_scale32(record)[1] + maximum_delta <= SCALE32_EXPONENT_MAX
            ),
            key=_record_key,
        )
    )
    require(int(prior_record) in filtered, "prior row-base record became illegal for fixed group deltas")
    return filtered


def _strict_scale32_table_dp(costs: Sequence[dict[int, int]]) -> tuple[tuple[int, ...], int]:
    require(len(costs) == TABLE_ENTRIES, "row-base table DP cluster count differs")
    states: dict[int, tuple[int, tuple[int, ...]]] = {
        record: (cost, (record,)) for record, cost in sorted(costs[0].items(), key=lambda item: _record_key(item[0]))
    }
    require(bool(states), "first row-base table cluster has no candidate")
    for cluster in costs[1:]:
        previous = sorted(states.items(), key=lambda item: _record_key(item[0]))
        current: dict[int, tuple[int, tuple[int, ...]]] = {}
        best: tuple[int, tuple[int, ...]] | None = None
        cursor = 0
        for record, record_cost in sorted(cluster.items(), key=lambda item: _record_key(item[0])):
            while cursor < len(previous) and _record_key(previous[cursor][0]) < _record_key(record):
                candidate = previous[cursor][1]
                if best is None or (candidate[0], tuple(_record_key(item) for item in candidate[1])) < (
                    best[0], tuple(_record_key(item) for item in best[1])
                ):
                    best = candidate
                cursor += 1
            if best is not None:
                current[record] = (best[0] + record_cost, best[1] + (record,))
        require(bool(current), "strict row-base table DP has no feasible path")
        states = current
    optimum, table = min(states.values(), key=lambda item: (item[0], tuple(_record_key(record) for record in item[1])))
    return table, optimum


def _update_table(
    exact: Tensor,
    payload: Tensor,
    selectors: Tensor,
    deltas: Tensor,
    prior_table: Sequence[int],
    base_shift: int,
    statistics: list[dict[str, int]] | None = None,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    statistics = _cluster_statistics(exact, payload, selectors, deltas) if statistics is None else statistics
    costs: list[dict[int, int]] = []
    candidate_counts: list[int] = []
    for statistic, prior in zip(statistics, prior_table, strict=True):
        candidates = _table_candidates(statistic, int(prior), base_shift)
        candidate_counts.append(len(candidates))
        costs.append({record: _table_record_cost(record, statistic, base_shift) for record in candidates})
    selected, optimum = _strict_scale32_table_dp(costs)
    require(
        all(statistic["row_count"] != 0 or selected[index] == prior_table[index] for index, statistic in enumerate(statistics)),
        "empty selector cluster did not retain prior table entry",
    )
    return selected, {
        "selector_row_counts": [statistic["row_count"] for statistic in statistics],
        "empty_selector_count": sum(statistic["row_count"] == 0 for statistic in statistics),
        "candidate_counts": candidate_counts,
        "fixed_assignment_optimum": str(optimum),
    }


def _raw_table(table: Sequence[int]) -> bytes:
    require(len(table) == TABLE_ENTRIES, "row-base table entry count differs")
    require(all(_record_key(left) < _record_key(right) for left, right in zip(table, table[1:])), "row-base table is not strictly increasing")
    return b"".join(struct.pack("<I", int(record)) for record in table)


def _state_raw(state: dict[str, Any]) -> dict[str, bytes]:
    return {
        "payload": _pack_signed_nibbles(state["payload"]),
        "row_selector": _pack_unsigned_nibbles(state["row_selectors"]),
        "group_delta": _pack_signed_nibbles(state["group_deltas"]),
        "row_base_table": _raw_table(state["row_base_table"]),
    }


def _state_hashes(state: dict[str, Any]) -> dict[str, str]:
    return {f"{name}_sha256": hashlib.sha256(raw).hexdigest() for name, raw in _state_raw(state).items()}


def _serialize_tensor_stream(module_name: str, state: dict[str, Any]) -> tuple[bytes, list[dict[str, Any]]]:
    raw = _state_raw(state)
    pieces = [raw["row_base_table"][index : index + 16] for index in range(0, 64, 16)] + [
        raw["row_selector"],
        raw["group_delta"],
        raw["payload"],
    ]
    kinds = ["row_base_table"] * 4 + ["row_selector", "group_delta", "payload"]
    descriptors: list[dict[str, Any]] = []
    offset = 0
    for ordinal, (kind, piece) in enumerate(zip(kinds, pieces, strict=True)):
        descriptors.append(
            {
                "ordinal": ordinal,
                "module": module_name,
                "kind": kind,
                "offset": offset,
                "bytes": len(piece),
                "sha256": hashlib.sha256(piece).hexdigest(),
            }
        )
        offset += len(piece)
    stream = b"".join(pieces)
    require(offset == len(stream), "V8 packed stream offset accounting differs")
    return stream, descriptors


def _parse_tensor_stream(
    module_name: str,
    stream: bytes,
    descriptors: Sequence[dict[str, Any]],
    shape: Sequence[int],
) -> dict[str, bytes]:
    rows, width = map(int, shape)
    groups = len(_group_ranges(width))
    expected_kinds = ["row_base_table"] * 4 + ["row_selector", "group_delta", "payload"]
    expected_lengths = [16, 16, 16, 16, (rows + 1) // 2, (rows * groups + 1) // 2, (rows * width + 1) // 2]
    require(len(descriptors) == len(expected_kinds), "V8 packed stream descriptor count differs")
    pieces: list[bytes] = []
    offset = 0
    for ordinal, (descriptor, kind, length) in enumerate(zip(descriptors, expected_kinds, expected_lengths, strict=True)):
        require(descriptor["ordinal"] == ordinal, "V8 packed stream ordinal differs")
        require(descriptor["module"] == module_name, "V8 packed stream tensor identity differs")
        require(descriptor["kind"] == kind, "V8 packed stream kind ordering differs")
        require(descriptor["offset"] == offset and descriptor["bytes"] == length, "V8 packed stream offset or length differs")
        piece = stream[offset : offset + length]
        require(len(piece) == length, "V8 packed stream ended early")
        require(hashlib.sha256(piece).hexdigest() == descriptor["sha256"], "V8 packed stream component hash differs")
        pieces.append(piece)
        offset += length
    require(offset == len(stream), "V8 packed stream has trailing bytes")
    return {
        "row_base_table": b"".join(pieces[:4]),
        "row_selector": pieces[4],
        "group_delta": pieces[5],
        "payload": pieces[6],
    }


def _all_zero_state(source: Tensor) -> dict[str, Any]:
    rows, width = source.shape
    groups = len(_group_ranges(width))
    table = tuple(_ordinal_record(index) for index in range(TABLE_ENTRIES))
    state = {
        "payload": torch.zeros(source.shape, dtype=torch.int8),
        "row_selectors": torch.zeros(rows, dtype=torch.uint8),
        "group_deltas": torch.zeros((rows, groups), dtype=torch.int8),
        "row_base_table": table,
        "final_objective": {"scaled_integer": "0", "binary_exponent": 0},
        "iteration_trace": [],
        "outer_iterations_completed": OUTER_ITERATIONS,
        "all_zero_tensor": True,
    }
    for iteration in range(1, OUTER_ITERATIONS + 1):
        state["iteration_trace"].append({"iteration": iteration, "prior_objective": "0", "final_objective": "0", **_state_hashes(state)})
    state["iteration_trace_sha256"] = canonical_sha256(state["iteration_trace"])
    return state


def construct_hierarchical_state(source: Tensor, seed_records: Sequence[int], module_name: str) -> dict[str, Any]:
    require(source.ndim == 2 and source.dtype == torch.bfloat16, f"source weight format differs: {module_name}")
    require(len(seed_records) == source.shape[0], f"V7 seed row count differs: {module_name}")
    if not bool(torch.any(source != 0)):
        return _all_zero_state(source)
    exact, shift_min, shift_max, subnormals = _tensor_exact_lattice(source)
    table = _initial_table(seed_records)
    assignment_cache = _prepare_assignment_cache(exact, shift_min)
    payload, selectors, deltas, previous_objective, assignment_summary, statistics, assignment_cache = _assign_rows_groups(
        exact, shift_min, table, assignment_cache
    )
    objective_exponent = 2 * min(shift_min, SCALE32_EXPONENT_MIN - 15)
    trace: list[dict[str, Any]] = []
    for iteration in range(1, OUTER_ITERATIONS + 1):
        prior_objective = previous_objective
        prior_table = table
        table, table_summary = _update_table(
            exact, payload, selectors, deltas, prior_table, shift_min, statistics
        )
        if table == prior_table:
            assignment_summary = {**assignment_summary, "assignment_reused_at_exact_fixed_point": True}
        else:
            payload, selectors, deltas, previous_objective, assignment_summary, statistics, assignment_cache = _assign_rows_groups(
                exact, shift_min, table, assignment_cache
            )
            assignment_summary = {**assignment_summary, "assignment_reused_at_exact_fixed_point": False}
        require(previous_objective <= prior_objective, f"V8 outer iteration increased exact source SSE: {module_name}")
        state = {
            "payload": payload,
            "row_selectors": selectors,
            "group_deltas": deltas,
            "row_base_table": table,
        }
        trace.append(
            {
                "iteration": iteration,
                "prior_objective": {"scaled_integer": str(prior_objective), "binary_exponent": objective_exponent},
                "final_objective": {"scaled_integer": str(previous_objective), "binary_exponent": objective_exponent},
                "assignment": assignment_summary,
                "table_update": table_summary,
                **_state_hashes(state),
            }
        )
    return {
        "shift_min": shift_min,
        "shift_max": shift_max,
        "shift_span": shift_max - shift_min,
        "subnormal_count": subnormals,
        "weight_group_lanes": WEIGHT_GROUP_LANES,
        "groups_per_row": len(_group_ranges(int(source.shape[1]))),
        "payload": payload,
        "row_selectors": selectors,
        "group_deltas": deltas,
        "row_base_table": table,
        "final_objective": {"scaled_integer": str(previous_objective), "binary_exponent": objective_exponent},
        "iteration_trace": trace,
        "iteration_trace_sha256": canonical_sha256(trace),
        "outer_iterations_completed": len(trace),
        "all_zero_tensor": False,
    }


def _decode_state(state: dict[str, Any]) -> Tensor:
    payload = state["payload"]
    rows, width = payload.shape
    decoded = torch.empty((rows, width), dtype=torch.float64)
    groups = _group_ranges(width)
    for row in range(rows):
        record = state["row_base_table"][int(state["row_selectors"][row])]
        significand, exponent = unpack_scale32(record)
        for group_index, (start, stop) in enumerate(groups):
            delta = int(state["group_deltas"][row, group_index])
            scale = torch.ldexp(
                torch.tensor(float(significand), dtype=torch.float64),
                torch.tensor(exponent + delta - 15, dtype=torch.int32),
            )
            decoded[row, start:stop] = payload[row, start:stop].to(torch.float64) * scale
    return decoded.to(torch.bfloat16)


def _validate_state(state: dict[str, Any], shape: Sequence[int]) -> dict[str, Any]:
    rows, width = map(int, shape)
    groups = len(_group_ranges(width))
    require(tuple(state["payload"].shape) == (rows, width), "V8 payload shape differs")
    require(tuple(state["row_selectors"].shape) == (rows,), "V8 row-selector shape differs")
    require(tuple(state["group_deltas"].shape) == (rows, groups), "V8 group-delta shape differs")
    require(bool(torch.all((state["payload"] >= SIGNED_INT4_MIN) & (state["payload"] <= SIGNED_INT4_MAX))), "V8 payload range differs")
    require(bool(torch.all((state["row_selectors"] >= 0) & (state["row_selectors"] < TABLE_ENTRIES))), "V8 row-selector range differs")
    require(bool(torch.all((state["group_deltas"] >= GROUP_DELTA_MIN) & (state["group_deltas"] <= GROUP_DELTA_MAX))), "V8 group-delta range differs")
    require(len(state["row_base_table"]) == TABLE_ENTRIES, "V8 row-base table count differs")
    require(all(_record_key(left) < _record_key(right) for left, right in zip(state["row_base_table"], state["row_base_table"][1:])), "V8 row-base table ordering differs")
    effective: list[int] = []
    for row in range(rows):
        _significand, exponent = unpack_scale32(state["row_base_table"][int(state["row_selectors"][row])])
        effective.extend(exponent + int(delta) for delta in state["group_deltas"][row])
    require(all(SCALE32_EXPONENT_MIN <= exponent <= SCALE32_EXPONENT_MAX for exponent in effective), "V8 effective exponent legality differs")
    raw = _state_raw(state)
    require(len(raw["payload"]) == (rows * width + 1) // 2, "V8 payload byte count differs")
    require(len(raw["row_selector"]) == (rows + 1) // 2, "V8 selector byte count differs")
    require(len(raw["group_delta"]) == (rows * groups + 1) // 2, "V8 delta byte count differs")
    require(len(raw["row_base_table"]) == 64, "V8 row-base table byte count differs")
    return {
        "row_count": rows,
        "weight_count": rows * width,
        "group_count": rows * groups,
        "groups_per_row": groups,
        "minimum_effective_exponent": min(effective),
        "maximum_effective_exponent": max(effective),
        "bytes": {name: len(value) for name, value in raw.items()},
    }


def _environment() -> dict[str, Any]:
    base = reference.require_project_python()
    return {
        **base,
        "platform": platform.platform(),
        "packages": {
            "torch": importlib.metadata.version("torch"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "backend": "exact BF16 dyadic lattice with integer assignment objectives and strict Scale32 table DP",
    }


def combined_self_test() -> dict[str, Any]:
    environment = _environment()
    reference.verify_companion(TASK_PATH, TASK_COMPANION)
    reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "frozen V8 Engineer authority differs")
    root_existed_before = OUTPUT_DIR.exists()
    marker_existed_before = MARKER_PATH.exists()
    require(not ATTEMPT_DIR.exists() and not marker_existed_before, "V8 official attempt must remain absent during constructor self-test")

    ranges_896 = _group_ranges(896)
    ranges_4864 = _group_ranges(4864)
    require(len(ranges_896) == 7 and len(ranges_4864) == 38, "V8 group geometry differs")
    illegal_width_rejected = reference.expect_rejected(lambda: _group_ranges(1024), "illegal V8 width was accepted")

    source_row = ((torch.arange(896, dtype=torch.int64) % 29) - 14).to(torch.float64) / 256.0
    fixture = torch.stack((source_row, source_row / 2, -source_row, torch.zeros_like(source_row))).to(torch.bfloat16)
    v7_seed = v7._construct_input_block_state(fixture, "self_test.linear")["final_records"].tolist()
    candidate_a = construct_hierarchical_state(fixture, v7_seed, "self_test.linear")
    candidate_b = construct_hierarchical_state(fixture, v7_seed, "self_test.linear")
    geometry = _validate_state(candidate_a, fixture.shape)
    require(_state_raw(candidate_a) == _state_raw(candidate_b), "independent V8 fixture structural outputs differ")
    require(candidate_a["iteration_trace"] == candidate_b["iteration_trace"], "independent V8 fixture iteration traces differ")
    require(candidate_a["final_objective"] == candidate_b["final_objective"], "independent V8 fixture objectives differ")
    require(candidate_a["outer_iterations_completed"] == OUTER_ITERATIONS, "V8 fixture iteration count differs")
    require(
        all(int(item["final_objective"]["scaled_integer"]) <= int(item["prior_objective"]["scaled_integer"]) for item in candidate_a["iteration_trace"]),
        "V8 fixture exact objective is not monotonic",
    )
    exact_fixture, fixture_shift, _fixture_shift_max, _fixture_subnormals = _tensor_exact_lattice(fixture)
    oracle_table = _initial_table(v7_seed)
    oracle_payload, oracle_selectors, oracle_deltas, oracle_objective, _oracle_summary = _assign_rows_groups_oracle(
        exact_fixture, fixture_shift, oracle_table
    )
    for _iteration in range(OUTER_ITERATIONS):
        oracle_table, _oracle_table_summary = _update_table(
            exact_fixture,
            oracle_payload,
            oracle_selectors,
            oracle_deltas,
            oracle_table,
            fixture_shift,
        )
        oracle_payload, oracle_selectors, oracle_deltas, oracle_objective, _oracle_summary = _assign_rows_groups_oracle(
            exact_fixture, fixture_shift, oracle_table
        )
    require(torch.equal(candidate_a["payload"], oracle_payload), "native exact payload differs from Python oracle")
    require(torch.equal(candidate_a["row_selectors"], oracle_selectors), "native exact row selectors differ from Python oracle")
    require(torch.equal(candidate_a["group_deltas"], oracle_deltas), "native exact group deltas differ from Python oracle")
    require(candidate_a["row_base_table"] == oracle_table, "native exact row-base table differs from Python oracle")
    require(int(candidate_a["final_objective"]["scaled_integer"]) == oracle_objective, "native exact objective differs from Python oracle")

    raw = _state_raw(candidate_a)
    require(_unpack_signed_nibbles(raw["payload"], fixture.numel()) == tuple(int(value) for value in candidate_a["payload"].reshape(-1).tolist()), "signed payload nibble round-trip differs")
    require(_unpack_signed_nibbles(raw["group_delta"], candidate_a["group_deltas"].numel()) == tuple(int(value) for value in candidate_a["group_deltas"].reshape(-1).tolist()), "signed delta nibble round-trip differs")

    unit_record = pack_scale32(SCALE32_SIGNIFICAND_MIN, 0)
    tie_values = torch.tensor([1, -1], dtype=torch.int64)
    tie_payload, _cost = _payload_for_effective_scale(tie_values, -1, unit_record, 0)
    require(tie_payload.tolist() == [0, -1], "exact half-way payload tie did not choose smaller signed int4")

    synthetic_prior = tuple(_ordinal_record(100 + index * 4) for index in range(TABLE_ENTRIES))
    synthetic_costs = []
    for index, prior in enumerate(synthetic_prior):
        if index == 7:
            synthetic_costs.append({prior: 0})
        else:
            synthetic_costs.append({prior: 1, _ordinal_record(_record_ordinal(prior) + 1): 0})
    selected_table, _optimum = _strict_scale32_table_dp(synthetic_costs)
    require(selected_table[7] == synthetic_prior[7], "empty-cluster table entry did not remain fixed")
    require(all(_record_key(left) < _record_key(right) for left, right in zip(selected_table, selected_table[1:])), "strict Scale32 table DP ordering differs")

    zero_fixture = torch.zeros((2, 896), dtype=torch.bfloat16)
    zero_state = construct_hierarchical_state(zero_fixture, [SCALE32_ALL_ZERO_RECORD] * 2, "self_test.zero")
    zero_geometry = _validate_state(zero_state, zero_fixture.shape)
    require(zero_state["row_base_table"] == tuple(_ordinal_record(index) for index in range(TABLE_ENTRIES)), "all-zero tensor table rule differs")
    require(not bool(torch.any(zero_state["payload"])) and not bool(torch.any(zero_state["group_deltas"])), "all-zero tensor emitted nonzero payload or deltas")

    reconstruction = _decode_state(candidate_a)
    fixture_sse = float(torch.sum((fixture.to(torch.float64) - reconstruction.to(torch.float64)) ** 2))
    stream, metadata_sequence = _serialize_tensor_stream("self_test.linear", candidate_a)
    parsed = _parse_tensor_stream("self_test.linear", stream, metadata_sequence, fixture.shape)
    require(parsed == raw, "V8 serialized stream parser did not recover exact packed components")
    build_root = ROOT / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v8-hash-rejection-", dir=build_root) as temporary_directory:
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
            "tampered copied artifact passed companion verification",
        )
    require(OUTPUT_DIR.exists() == root_existed_before, "self-test changed the V8 namespace")
    require(MARKER_PATH.exists() == marker_existed_before is False, "self-test changed the V8 execution marker")

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
            "legal_width_group_counts": {"896": 7, "4864": 38},
            "illegal_width_rejected": illegal_width_rejected,
            "independent_candidate_a_b_structural_match": True,
            "native_exact_assignment_matches_python_oracle": True,
            "outer_iterations_completed": OUTER_ITERATIONS,
            "monotonic_exact_source_objective": True,
            "signed_payload_nibble_round_trip": True,
            "signed_delta_nibble_round_trip": True,
            "exact_payload_tie_chooses_smaller_signed_value": True,
            "effective_exponents_legal": True,
            "strict_row_base_table_dp": True,
            "empty_selector_retains_prior_table_entry": True,
            "all_zero_tensor_rule": True,
            "metadata_sequence": metadata_sequence,
            "metadata_serializer_parser_round_trip": True,
            "bound_hash_mismatch_rejected": hash_mismatch_rejected,
        },
        "fixture": {
            "shape": list(fixture.shape),
            "geometry": geometry,
            "zero_geometry": zero_geometry,
            "bf16_reconstruction_sse": fixture_sse,
            "iteration_trace_sha256": candidate_a["iteration_trace_sha256"],
            **_state_hashes(candidate_a),
        },
        "official_build_namespace_changed": False,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": False,
        "claim_boundary": "constructor kernel and fixture self-test only; no 169-tensor, fresh-V7 aggregate, 8/8 quality, or 312-event conclusion",
    }


def _fresh_v7_measurement(source: Tensor, state: dict[str, Any], sealed: dict[str, Any]) -> dict[str, Any]:
    rows, width = source.shape
    sse = 0.0
    digest = hashlib.sha256()
    for start, stop in v7.v4._row_chunks(rows, width):
        decoded = v7._decoded_chunk(
            state["final_records"], state["final_assignments"], state["final_codebooks"], start, stop
        )
        difference = source[start:stop].to(torch.float64) - decoded.to(torch.float64)
        sse += float(torch.sum(difference * difference))
        digest.update(v7.v3.raw_tensor_bytes(decoded))
    reconstruction_sha = digest.hexdigest()
    require(reconstruction_sha == sealed["reconstruction_bf16_sha256"], f"fresh V7 reconstruction hash differs: {sealed['module']}")
    require(sse == float(sealed["v7_input_block_bf16_sse"]), f"fresh V7 reconstruction SSE differs: {sealed['module']}")
    return {
        "fresh_v7_bf16_sse": sse,
        "fresh_v7_reconstruction_bf16_sha256": reconstruction_sha,
        "fresh_v7_matches_sealed": True,
    }


def _materialize_v8(module: nn.Linear, source: Tensor, state: dict[str, Any]) -> dict[str, Any]:
    rows, width = source.shape
    groups = len(_group_ranges(width))
    table_significands = torch.tensor([unpack_scale32(record)[0] for record in state["row_base_table"]], dtype=torch.float64)
    table_exponents = torch.tensor([unpack_scale32(record)[1] for record in state["row_base_table"]], dtype=torch.int32)
    sse = 0.0
    digest = hashlib.sha256()
    with torch.no_grad():
        for start, stop in v7.v4._row_chunks(rows, width):
            selected = state["row_selectors"][start:stop].to(torch.int64)
            significands = table_significands[selected]
            exponents = table_exponents[selected, None] + state["group_deltas"][start:stop].to(torch.int32)
            scales = torch.ldexp(significands[:, None].expand_as(exponents), exponents - 15)
            decoded = (
                state["payload"][start:stop]
                .to(torch.float64)
                .reshape(stop - start, groups, WEIGHT_GROUP_LANES)
                * scales[:, :, None]
            ).reshape(stop - start, width).to(torch.bfloat16)
            difference = source[start:stop].to(torch.float64) - decoded.to(torch.float64)
            sse += float(torch.sum(difference * difference))
            digest.update(v7.v3.raw_tensor_bytes(decoded))
            module.weight[start:stop].copy_(decoded)
    reconstruction_sha = digest.hexdigest()
    require(reconstruction_sha == v7.v3.tensor_sha256(module.weight), "V8 materialized reconstruction hash differs")
    return {"v8_hierarchical_bf16_sse": sse, "reconstruction_bf16_sha256": reconstruction_sha}


def _policy_and_format() -> dict[str, Any]:
    body = {
        "payload": {"bits_per_weight": 4, "semantics": "signed int4 -8 through +7", "packing": "low nibble first"},
        "row_selector": {"bits_per_output_row": 4, "table_entries": 16, "packing": "low nibble first"},
        "group_delta": {"bits_per_128_weight_group": 4, "range": [-8, 7], "packing": "low nibble first"},
        "row_base_table": {"entries": 16, "bytes_per_tensor": 64, "ordering": "strictly increasing Scale32"},
        "metadata_stream_order": ["row_base_table_beat_0", "row_base_table_beat_1", "row_base_table_beat_2", "row_base_table_beat_3", "row_selector", "group_delta", "payload"],
        "activation": {"policy": "grouped Dynamic Scale32", "event_count": EXPECTED_EVENTS, "maximum_transport_metadata_bytes_per_decoder_layer_token": 832},
        "outer_iterations": OUTER_ITERATIONS,
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

    names = ("payload", "row_selector", "group_delta", "row_base_table", "reconstruction", "objective", "iteration_trace")
    manifests: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    weight_manifest: list[dict[str, Any]] = []
    streams = {name: hashlib.sha256() for name in ("payload", "row_selector", "group_delta", "row_base_table", "packed_tensor", "reconstruction")}
    totals = {"weight_count": 0, "payload": 0, "row_selector": 0, "group_delta": 0, "row_base_table": 0}

    for ordinal, (name, module) in enumerate(modules, start=1):
        started = time.monotonic()
        print(f"V8_CONSTRUCT_START {ordinal}/169 {name}", flush=True)
        source = module.weight.detach()
        source_sha = v7.v3.tensor_sha256(source)
        fresh_v7_state = v7._construct_input_block_state(source, name)
        fresh_v7 = _fresh_v7_measurement(source, fresh_v7_state, sealed_records[name])
        state = construct_hierarchical_state(source, fresh_v7_state["final_records"].tolist(), name)
        geometry = _validate_state(state, source.shape)
        packed_stream, descriptors = _serialize_tensor_stream(name, state)
        require(_parse_tensor_stream(name, packed_stream, descriptors, source.shape) == _state_raw(state), f"V8 stream round trip differs: {name}")
        reconstruction = {"module": name, "weight_count": int(source.numel()), **fresh_v7, **_materialize_v8(module, source, state)}
        reconstruction["v8_strictly_improved_against_fresh_v7"] = reconstruction["v8_hierarchical_bf16_sse"] < reconstruction["fresh_v7_bf16_sse"]
        raw = _state_raw(state)
        component_records = {
            component: {"module": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            for component, value in raw.items()
        }
        component_records["row_selector"]["row_count"] = int(source.shape[0])
        component_records["group_delta"]["group_count"] = geometry["group_count"]
        component_records["row_base_table"]["record_count"] = 16
        for component in ("payload", "row_selector", "group_delta", "row_base_table"):
            manifests[component].append(component_records[component])
            streams[component].update(name.encode() + b"\0" + raw[component])
            totals[component] += len(raw[component])
        streams["packed_tensor"].update(name.encode() + b"\0" + packed_stream)
        streams["reconstruction"].update(name.encode() + b"\0" + bytes.fromhex(reconstruction["reconstruction_bf16_sha256"]))
        manifests["reconstruction"].append(reconstruction)
        manifests["objective"].append({"module": name, "final_objective": state["final_objective"]})
        manifests["iteration_trace"].append({
            "module": name,
            "source_bf16_sha256": source_sha,
            "shift_min": state.get("shift_min"),
            "shift_max": state.get("shift_max"),
            "outer_iterations_completed": state["outer_iterations_completed"],
            "iteration_trace": state["iteration_trace"],
            "iteration_trace_sha256": state["iteration_trace_sha256"],
        })
        record = {
            "module": name,
            "name": f"{name}.weight",
            "shape": list(source.shape),
            "scope": "separated_lm_head" if name == "lm_head" else "transformer",
            "geometry": geometry,
            "components": component_records,
            "packed_stream": {"bytes": len(packed_stream), "sha256": hashlib.sha256(packed_stream).hexdigest(), "descriptors": descriptors},
            "reconstruction": reconstruction,
            "objective": state["final_objective"],
            "iteration_trace_sha256": state["iteration_trace_sha256"],
        }
        weight_manifest.append(record)
        totals["weight_count"] += int(source.numel())
        print(f"V8_CONSTRUCT_DONE {ordinal}/169 {name} seconds={time.monotonic()-started:.3f} v7_sse={fresh_v7['fresh_v7_bf16_sse']:.12g} v8_sse={reconstruction['v8_hierarchical_bf16_sse']:.12g}", flush=True)
        del state, fresh_v7_state
        gc.collect()

    aggregate_v7 = float(sum(item["fresh_v7_bf16_sse"] for item in manifests["reconstruction"]))
    aggregate_v8 = float(sum(item["v8_hierarchical_bf16_sse"] for item in manifests["reconstruction"]))
    fresh_count = sum(bool(item["fresh_v7_matches_sealed"]) for item in manifests["reconstruction"])
    improved_count = sum(bool(item["v8_strictly_improved_against_fresh_v7"]) for item in manifests["reconstruction"])
    require(aggregate_v7 == V7_AGGREGATE_BF16_SSE, "fresh aggregate V7 BF16 SSE differs")
    reconstruction_body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "measurement": "sum-squared error from pinned BF16 source to BF16-materialized reconstruction",
        "reference": {"candidate_id": v7.CANDIDATE_ID, "artifact": file_record(V7_RECONSTRUCTION_PATH), "aggregate_bf16_sse": V7_AGGREGATE_BF16_SSE, "fresh_reconstruction_required": True},
        "tensor_count": len(manifests["reconstruction"]),
        "fresh_v7_exact_reproduction_count": fresh_count,
        "all_169_tensors_legal": len(weight_manifest) == 169,
        "per_tensor_strictly_improved_count": improved_count,
        "aggregate_fresh_v7_bf16_sse": aggregate_v7,
        "aggregate_v8_hierarchical_bf16_sse": aggregate_v8,
        "aggregate_strictly_improved_against_fresh_v7": aggregate_v8 < aggregate_v7,
        "tensors": manifests["reconstruction"],
    }
    reconstruction_report = {
        **reconstruction_body,
        "status": "PASS_FRESH_V7_AND_169_LEGAL_WITH_STRICT_AGGREGATE_IMPROVEMENT" if fresh_count == 169 and aggregate_v8 < aggregate_v7 else "NO_GO_V7_RELATIVE_MODEL_ONLY_RECONSTRUCTION_GATE",
        "reconstruction_report_sha256": canonical_sha256(reconstruction_body),
    }
    require(totals == {"weight_count": 493961216, "payload": 246980608, "row_selector": 228032, "group_delta": 1929536, "row_base_table": 10816}, "whole-model V8 storage geometry differs")
    hashes = {f"{name}_manifest_sha256": canonical_sha256(values) for name, values in manifests.items()}
    hashes.update({"weight_manifest_sha256": canonical_sha256(weight_manifest), "reconstruction_report_sha256": reconstruction_report["reconstruction_report_sha256"]})
    hashes.update({f"{name}_stream_sha256": digest.hexdigest() for name, digest in streams.items()})
    storage = {
        "total_weight_count": totals["weight_count"],
        "total_payload_bytes": totals["payload"],
        "total_row_selector_bytes": totals["row_selector"],
        "total_group_delta_bytes": totals["group_delta"],
        "total_row_base_table_bytes": totals["row_base_table"],
        "whole_model_total_weight_bytes": sum(totals[name] for name in ("payload", "row_selector", "group_delta", "row_base_table")),
        "whole_model_exact_bits_per_weight": 4.03511828750539,
        "standard_decoder_layer_exact_bits_per_weight": 4.03489010989011,
        "lm_head_exact_bits_per_weight": 4.035718046696354,
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
    model.lm_head.weight = nn.Parameter(cloned, requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight is not embedding and model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head split failed")
    require(v7.v3.tensor_sha256(model.lm_head.weight) == source_hash, "lm_head clone differs")
    require(id(embedding) == source_object_id and embedding.data_ptr() == source_pointer, "embedding identity changed during split")

    weight_manifest, construction = _quantize_model(model)
    require(model.config.tie_word_embeddings is False, "candidate config was retied")
    require(model.model.embed_tokens.weight is embedding and embedding.data_ptr() == source_pointer, "embedding identity changed after V8 construction")
    require(list(embedding.shape) == source_shape and str(embedding.dtype) == source_dtype, "embedding shape or dtype changed")
    embedding_hash = v7.v3.tensor_sha256(embedding)
    require(embedding_hash == source_hash == SOURCE_EMBEDDING_SHA256, "embedding bytes changed after V8 construction")
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
        "module_counts": {"transformer_linears": 168, "lm_head": 1, "total_hierarchical_linears": 169},
        "identity": identity,
        "storage": construction["storage"],
        **{f"{name}_manifest": values for name, values in construction["manifests"].items()},
        "weight_manifest": weight_manifest,
        "state_manifest": state_manifest,
    }
    witness = {
        "construction_label": label,
        "source": {"same_parameter_object": True, "same_storage_pointer": True, "embedding_object_id": source_object_id, "embedding_storage_pointer": source_pointer, "embedding_shape": source_shape, "embedding_dtype": source_dtype, "embedding_bf16_sha256": source_hash},
        "post_split": {"different_parameter_objects": model.lm_head.weight is not embedding, "different_storage_pointers": model.lm_head.weight.data_ptr() != embedding.data_ptr(), "embedding_object_preserved": id(embedding) == source_object_id, "embedding_storage_preserved": embedding.data_ptr() == source_pointer, "config_tie_word_embeddings": bool(model.config.tie_word_embeddings)},
        "post_v8": {"embedding_bf16_sha256": embedding_hash, "embedding_hash_matches_source": embedding_hash == SOURCE_EMBEDDING_SHA256, "lm_head_storage_is_separate": model.lm_head.weight.data_ptr() != embedding.data_ptr(), "transformer_hierarchical_count": 168, "lm_head_hierarchical_count": 1},
    }
    return model, manifest, witness, construction["reconstruction"]


def _compare_candidates(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> dict[str, Any]:
    comparisons = {key: candidate_a["identity"][key] == candidate_b["identity"][key] for key in STATE_MATCH_KEYS}
    require(all(comparisons.values()), f"candidate A/B identity differs: {comparisons}")
    manifest_fields = tuple(f"{name}_manifest" for name in ("payload", "row_selector", "group_delta", "row_base_table", "reconstruction", "objective", "iteration_trace")) + ("weight_manifest", "state_manifest")
    field_matches = {field: candidate_a[field] == candidate_b[field] for field in manifest_fields}
    require(all(field_matches.values()), f"candidate A/B whole-model manifests differ: {field_matches}")
    return {"state_match_keys": list(STATE_MATCH_KEYS), "identity_comparisons": comparisons, "manifest_byte_equivalent_fields": list(manifest_fields), "manifest_comparisons": field_matches, "all_match": True}


def _validate_reconstruction(report: dict[str, Any]) -> bool:
    require(report["tensor_count"] == len(report["tensors"]) == 169, "V8 reconstruction tensor count differs")
    require(report["fresh_v7_exact_reproduction_count"] == 169, "V8 fresh V7 reproduction count differs")
    require(report["all_169_tensors_legal"] is True, "V8 legality count differs")
    require(report["aggregate_fresh_v7_bf16_sse"] == V7_AGGREGATE_BF16_SSE, "V8 aggregate V7 reference differs")
    passed = report["aggregate_strictly_improved_against_fresh_v7"] is True
    require((report["status"].startswith("PASS_")) == passed, "V8 reconstruction status differs")
    body = {key: value for key, value in report.items() if key not in {"status", "reconstruction_report_sha256"}}
    require(report["reconstruction_report_sha256"] == canonical_sha256(body), "V8 reconstruction report hash differs")
    return passed


def _validate_quality(report: dict[str, Any]) -> bool:
    return v7.v3.validate_quality(report)


def _validate_dynamic(report: dict[str, Any]) -> bool:
    passed = v7.v3.validate_dynamic(report)
    activation = report["activation_execution"]
    require(activation["named_boundary_count"] == EXPECTED_EVENTS, "V8 named Dynamic Scale32 boundary count differs")
    require(activation["transport_metadata_bytes_per_decoder_layer_token"] <= 832, "V8 activation metadata cap differs")
    return passed


def _audit_attempt_namespace(expect_root_absent: bool) -> dict[str, Any]:
    root_existed = OUTPUT_DIR.exists()
    if expect_root_absent:
        require(not root_existed, "V8 namespace existed before fresh pre-attempt preparation")
    markers = list(OUTPUT_DIR.rglob("execution_started.json")) if root_existed else []
    require(not markers and not ATTEMPT_DIR.exists(), "V8 official attempt or marker exists")
    return {"candidate_id": CANDIDATE_ID, "root_existed_at_audit": root_existed, "attempt_directories": [], "matching_execution_markers": [], "authorization_consumed": False}


def _recover_nonqualifying_preparation() -> dict[str, Any]:
    if not OUTPUT_DIR.exists():
        return {"recovered": False, "fresh_namespace": True, "archived_artifacts": []}
    complete = PREATTEMPT_DIR.is_dir() and CONTRACT_PATH.is_file() and (READY_PATH.is_file() or NO_GO_PATH.is_file())
    if complete:
        raise RuntimeError("sealed V8 pre-attempt closure already exists; use verify")
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "cannot recover a V8 namespace with an official attempt")
    candidates = [path for path in OUTPUT_DIR.iterdir() if path.name != "preflight-failures"]
    require(candidates, "existing V8 namespace has ambiguous provenance")
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
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "V8 Engineer authority differs")
    require(task["fresh_attempt_namespace"]["attempt_budget"] == 1 and task["fresh_attempt_namespace"]["attempts_consumed_at_task_freeze"] == 0, "V8 attempt accounting differs")
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
        "authority": {"engineer_task": artifacts["engineer_task"], "planning_requirements": artifacts["planning_requirements"], "attempt_id": "attempt-0001", "exact_authorized_candidate_attempts": 1, "attempts_consumed": 0, "selected_policy_id": None},
        "source_model": source_identity,
        "environment": environment,
        "format_and_policy": _policy_and_format(),
        "construction": {"backend": "fresh exact V7 reconstruction plus native exact hierarchical row/group assignment and strict Scale32 table DP", "source_tied_then_runtime_split_lm_head": True, "source_embedding_bf16_sha256": SOURCE_EMBEDDING_SHA256, "embedding_bytes_preserved": True, "transformer_linears": 168, "separated_lm_head_linears": 1, "weight_group_lanes": 128, "outer_iterations": 8, "candidate_a_and_b_whole_model_manifests_match": state_comparison},
        "pre_attempt_gates": {
            "fresh_v7_and_169_model_closure": {"passed": gate_passes["fresh_v7_and_169_model_closure"], "artifact": artifacts["model_only_reconstruction"]},
            "non_evaluator_quality_8_of_8": {"passed": gate_passes["non_evaluator_quality_8_of_8"], "artifact": artifacts["non_evaluator_quality"]},
            "dynamic_312_event_closure": {"passed": gate_passes["dynamic_312_event_closure"], "artifact": artifacts["dynamic_preflight"]},
            "two_build_alias_and_manifest_closure": {"passed": gate_passes["two_build_alias_and_manifest_closure"], "artifact": artifacts["alias_separation_witness"]},
            "all_pass": all_pass,
        },
        "official_matrix_binding": {"artifact": artifacts["official_matrix_hash_only"], "sha256": MATRIX_SHA256, "prompt_count": 9, "official_input_content_accessed_during_preflight": False},
        "attempt": {"score_policy": "binary only: 9/9 complete token arrays exactly equal BF16 and 9/9 response-gate outcomes equal BF16", "attempt_budget": 1, "attempts_consumed": 0},
        "official_execution": {"mode_exposed_by_runner": False, "attempt_directory_exists": False, "execution_marker_exists": False, "conditional_execution_rule": "consume attempt-0001 only after this exact PREATTEMPT_READY closure verifies"},
        "namespace_audit": namespace_audit,
        "artifacts": artifacts,
        "claim_boundary": {"attempts_consumed": 0, "numerical_conclusion": None, "selected_policy_id": None, "rtl_or_ppa_claimed": False, "current_stage_remains": "specification"},
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
    require(CONTRACT_PATH.is_file(), "V8 predeclared contract is missing")
    require(READY_PATH.is_file() != NO_GO_PATH.is_file(), "exactly one V8 pre-attempt terminal record is required")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "V8 predeclared contract wrapper differs")
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "V8 predeclared contract canonical hash differs")
    contract = wrapper["contract"]
    require(contract["candidate_id"] == CANDIDATE_ID and contract["task_id"] == TASK_ID, "V8 predeclared authority differs")
    require(contract["artifacts"]["runner_source"]["sha256"] == reference.sha256_file(RUNNER_PATH), "V8 runner changed after freeze")
    require(contract["artifacts"]["scalable_backend_source"]["sha256"] == reference.sha256_file(BACKEND_PATH), "V8 backend changed after freeze")
    for record in contract["artifacts"].values():
        _verify_bound_record(record)
    reference.verify_companion(TASK_PATH, TASK_COMPANION)
    reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    require(reference.sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(reference.sha256_file(V7_RECONSTRUCTION_PATH) == V7_RECONSTRUCTION_SHA256, "sealed V7 reconstruction hash differs")
    require(verify_source_snapshot() == contract["source_model"], "source snapshot identity differs")
    packages = {**verify_versions(), "numpy": importlib.metadata.version("numpy")}
    require(packages == contract["environment"]["packages"], "package versions differ")
    require(combined_self_test() == load_json(SELF_TEST_PATH), "V8 runner self-test result differs")
    candidate_a = load_json(CANDIDATE_A_PATH)
    candidate_b = load_json(CANDIDATE_B_PATH)
    comparison = _compare_candidates(candidate_a, candidate_b)
    witness = load_json(ALIAS_WITNESS_PATH)
    require(witness["state_comparison"] == comparison, "V8 alias witness comparison differs")
    gate_passes = {
        "fresh_v7_and_169_model_closure": _validate_reconstruction(load_json(RECONSTRUCTION_PATH)),
        "non_evaluator_quality_8_of_8": _validate_quality(load_json(QUALITY_PATH)),
        "dynamic_312_event_closure": _validate_dynamic(load_json(DYNAMIC_PATH)),
        "two_build_alias_and_manifest_closure": comparison["all_match"] and witness["embedding_preserved_in_both"] and witness["lm_head_separate_in_both"],
    }
    require(contract["pre_attempt_gates"]["all_pass"] == all(gate_passes.values()), "V8 contract aggregate gate status differs")
    for name, passed in gate_passes.items():
        require(contract["pre_attempt_gates"][name]["passed"] == passed, f"V8 contract gate differs: {name}")
    terminal_path = READY_PATH if READY_PATH.is_file() else NO_GO_PATH
    terminal = load_json(terminal_path)
    require(terminal["candidate_id"] == CANDIDATE_ID, "V8 terminal candidate differs")
    require(terminal["contract"]["sha256"] == reference.sha256_file(CONTRACT_PATH), "V8 terminal contract file hash differs")
    require(terminal["contract_sha256"] == wrapper["contract_sha256"], "V8 terminal contract canonical hash differs")
    require(terminal["bound_artifacts"] == contract["artifacts"], "V8 terminal artifact snapshot differs")
    require(terminal["official_execution_marker_exists"] is False and terminal["attempts_consumed"] == 0, "V8 terminal record claims an official attempt")
    require((terminal_path == READY_PATH) == all(gate_passes.values()), "V8 terminal disposition differs from gates")
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "V8 official attempt exists after marker-free closure")
    quality = load_json(QUALITY_PATH)
    dynamic = load_json(DYNAMIC_PATH)
    reconstruction = load_json(RECONSTRUCTION_PATH)
    return {
        "status": terminal["status"],
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": wrapper["contract_sha256"],
        "gate_passes": gate_passes,
        "fresh_v7_exact_reproduction_count": reconstruction["fresh_v7_exact_reproduction_count"],
        "aggregate_fresh_v7_bf16_sse": reconstruction["aggregate_fresh_v7_bf16_sse"],
        "aggregate_v8_hierarchical_bf16_sse": reconstruction["aggregate_v8_hierarchical_bf16_sse"],
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
        "torch_num_threads": TORCH_THREADS,
        "backend": "native exact sorted-prefix selector/delta assignment with Python-integer Scale32 table DP",
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
    self_test = combined_self_test()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    probe = OUTPUT_DIR / ".preflight-write-probe"
    descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, b"preflight\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(probe.read_bytes() == b"preflight\n", "V8 output-path readback differs")
    probe.unlink()

    chronology: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "preparation_started_at_utc": utc_now(),
        "namespace_audit": namespace_audit,
        "official_matrix_content_accessed": False,
        "output_path_preflight": {"exclusive_create": True, "readback": True, "collision_behavior": "O_EXCL", "execution_marker_created": False},
    }
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    inputs, input_manifest = v7.v3.non_evaluator_inputs(tokenizer)

    chronology["candidate_a_started_at_utc"] = utc_now()
    print("V8_CANDIDATE_A_START", flush=True)
    candidate_a, manifest_a, witness_a, reconstruction_a = _load_candidate("candidate_a_independent_construction")
    chronology["candidate_a_complete_at_utc"] = utc_now()
    print("V8_CANDIDATE_A_DONE", flush=True)
    del candidate_a
    gc.collect()

    chronology["candidate_b_started_at_utc"] = utc_now()
    print("V8_CANDIDATE_B_START", flush=True)
    candidate_b, manifest_b, witness_b, reconstruction_b = _load_candidate("candidate_b_independent_quality_and_dynamic_preflight")
    state_comparison = _compare_candidates(manifest_a, manifest_b)
    require(reconstruction_a == reconstruction_b, "V8 candidate A/B reconstruction reports differ")
    chronology["candidate_b_constructed_at_utc"] = utc_now()
    print("V8_CANDIDATE_B_CONSTRUCTED quality_dynamic_start", flush=True)
    quality, dynamic = v7.v3.non_evaluator_quality_and_dynamic(candidate_b, tokenizer, inputs, input_manifest)
    chronology["candidate_b_quality_and_dynamic_complete_at_utc"] = utc_now()
    print("V8_QUALITY_DYNAMIC_DONE", flush=True)
    del candidate_b
    gc.collect()

    gate_passes = {
        "fresh_v7_and_169_model_closure": _validate_reconstruction(reconstruction_b),
        "non_evaluator_quality_8_of_8": _validate_quality(quality),
        "dynamic_312_event_closure": _validate_dynamic(dynamic),
        "two_build_alias_and_manifest_closure": state_comparison["all_match"] and witness_a["post_v8"]["embedding_hash_matches_source"] and witness_b["post_v8"]["embedding_hash_matches_source"] and witness_a["post_v8"]["lm_head_storage_is_separate"] and witness_b["post_v8"]["lm_head_storage_is_separate"],
    }
    require(not MARKER_PATH.exists() and not ATTEMPT_DIR.exists(), "V8 official attempt appeared before artifact freeze")

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
        "embedding_preserved_in_both": witness_a["post_v8"]["embedding_hash_matches_source"] and witness_b["post_v8"]["embedding_hash_matches_source"],
        "lm_head_separate_in_both": witness_a["post_v8"]["lm_head_storage_is_separate"] and witness_b["post_v8"]["lm_head_storage_is_separate"],
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
        "attempt_budget": 1,
        "attempts_consumed": 0,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": False,
        "disposition": "all marker-free V8 gates pass; PREATTEMPT_READY is sealed for Fresh Review" if all_pass else "preserve V8 no-go evidence, leave attempt-0001 absent and unconsumed, keep selected_policy_id null, and submit for Fresh Review",
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
        "regression": "repair only the isolated marker-free V8 runner and rerun pre-attempt preparation; official authorization remains unconsumed",
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
