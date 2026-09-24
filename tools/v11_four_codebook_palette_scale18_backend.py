#!/usr/bin/env python3
"""Exact V11 four-codebook/quaternary-selector/palette-Scale18 kernel."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import shutil
import struct
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_option_b_alias_safe_four_input_block_codebook_quaternary_row_selector_palette_scale18_w4_grouped_dynamic_scale32_stage1 as reference
import v4_joint_full_model_backend as v4
import v7_input_block_full_model_backend as v7
import v10_packed_scale24_per_row_selector_backend as v10
from ace2_quality_contracts import SCALE32_SIGNIFICAND_MAX, SCALE32_SIGNIFICAND_MIN, pack_scale32
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
RECONSTRUCTION_PATH = PREATTEMPT_DIR / "model_only_v10_relative_reconstruction.json"
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
V10_RECONSTRUCTION_PATH = ROOT / "build/stage1-option-b-alias-safe-packed-scale24-per-row-selector-dual-input-block-codebook-w4-grouped-dynamic-scale32-w4a8-v10/preattempt/model_only_v9_relative_reconstruction.json"
V10_RECONSTRUCTION_SHA256 = "47a276395a2cf7e574f1d15ec61942ad7c60679ff0129c31ead3d75b944938de"
V10_AGGREGATE_BF16_SSE = 2153.4047236346078
V10_REVIEW_PATH = ROOT / "research/raw/specification/packed-scale24-per-row-selector-v10-review-done-20260807T011340Z.json"
V10_REVIEW_SHA256 = "013018e47424a30198917b8b8ac6dcc8d19617b7c5e2286353c1cc58797e5d04"

INPUT_BLOCK_LANES = reference.INPUT_BLOCK_LANES
CODEBOOK_BANKS = reference.CODEBOOK_BANKS
CODEBOOK_ENTRIES = reference.CODEBOOK_ENTRIES
SELECTOR_BITS = reference.SELECTOR_BITS
PALETTE_ENTRIES = reference.PALETTE_ENTRIES
ROW_RECORD_BITS = reference.ROW_RECORD_BITS
OUTER_ITERATIONS = reference.OUTER_ITERATIONS
ALIGNMENT_BYTES = reference.ALIGNMENT_BYTES
EXPONENT_MIN = reference.EXPONENT_MIN
EXPONENT_MAX = reference.EXPONENT_MAX
LEGAL_EXPONENTS = tuple(range(EXPONENT_MIN, EXPONENT_MAX + 1))
TORCH_THREADS = 32
CONSTRUCTION_TORCH_THREADS = 1
EXPECTED_EVENTS = v7.grouped.LAYERS * len(v7.grouped.EVENT_FAMILIES)
STATE_MATCH_KEYS = (
    "embedding_bf16_sha256",
    "payload_manifest_sha256",
    "palette_scale18_manifest_sha256",
    "four_codebook_manifest_sha256",
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


def _align(value: int) -> int:
    return ((value + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES) * ALIGNMENT_BYTES


def _validate_palette(palette: Sequence[int]) -> tuple[int, int, int, int]:
    values = tuple(int(value) for value in palette)
    require(len(values) == PALETTE_ENTRIES, "V11 exponent palette does not contain four entries")
    require(all(EXPONENT_MIN <= value <= EXPONENT_MAX for value in values), "V11 exponent palette escaped the legal range")
    require(all(left < right for left, right in zip(values, values[1:])), "V11 exponent palette is not strictly increasing")
    return values  # type: ignore[return-value]


def _validate_codebooks(
    codebooks: Sequence[Sequence[Sequence[int]]], block_count: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    require(len(codebooks) == block_count, "V11 four-codebook block count differs")
    validated = []
    for banks in codebooks:
        require(len(banks) == CODEBOOK_BANKS, "V11 input block does not contain four codebook banks")
        validated.append(tuple(v4.reference.validate_codebook(bank) for bank in banks))
    return tuple(validated)


def _codebook_raw(codebooks: Sequence[Sequence[Sequence[int]]]) -> bytes:
    validated = _validate_codebooks(codebooks, len(codebooks))
    return b"".join(bytes(point & 0xFF for point in bank) for banks in validated for bank in banks)


def _pack_selectors(selectors: Tensor) -> bytes:
    require(selectors.ndim == 2 and selectors.dtype == torch.uint8, "V11 selector tensor format differs")
    flat = [int(value) for value in selectors.reshape(-1).tolist()]
    require(all(0 <= value < CODEBOOK_BANKS for value in flat), "V11 selector escaped two bits")
    byte_count = (len(flat) * SELECTOR_BITS + 7) // 8
    raw = bytearray(_align(byte_count))
    for ordinal, value in enumerate(flat):
        bit = ordinal * SELECTOR_BITS
        raw[bit // 8] |= value << (bit % 8)
    return bytes(raw)


def _unpack_selectors(raw: bytes, rows: int, block_count: int) -> Tensor:
    field_count = rows * block_count
    used_bits = field_count * SELECTOR_BITS
    require(len(raw) == _align((used_bits + 7) // 8), "V11 selector stream alignment differs")
    values = []
    for ordinal in range(field_count):
        bit = ordinal * SELECTOR_BITS
        values.append((raw[bit // 8] >> (bit % 8)) & 0x3)
    for bit in range(used_bits, len(raw) * 8):
        require(((raw[bit // 8] >> (bit % 8)) & 1) == 0, "V11 selector padding bit is nonzero")
    return torch.tensor(values, dtype=torch.uint8).reshape(rows, block_count)


def _pack_palette_scale18(palette: Sequence[int], records: Tensor) -> bytes:
    palette_values = _validate_palette(palette)
    require(records.ndim == 1 and records.dtype == torch.int64, "V11 row-record tensor format differs")
    values = [int(value) for value in records.tolist()]
    require(all(0 <= value < (1 << ROW_RECORD_BITS) for value in values), "V11 row record escaped 18 bits")
    for value in values:
        require(SCALE32_SIGNIFICAND_MIN <= (value & 0xFFFF) <= SCALE32_SIGNIFICAND_MAX, "V11 row significand escaped normalized Q1.15")
        require(((value >> 16) & 0x3) < PALETTE_ENTRIES, "V11 row palette index escaped two bits")
    used_bits = len(values) * ROW_RECORD_BITS
    raw = bytearray(_align(PALETTE_ENTRIES + (used_bits + 7) // 8))
    for index, exponent in enumerate(palette_values):
        raw[index] = exponent & 0xFF
    for ordinal, value in enumerate(values):
        for offset in range(ROW_RECORD_BITS):
            if (value >> offset) & 1:
                bit = ordinal * ROW_RECORD_BITS + offset
                raw[PALETTE_ENTRIES + bit // 8] |= 1 << (bit % 8)
    return bytes(raw)


def _unpack_palette_scale18(raw: bytes, row_count: int) -> tuple[tuple[int, int, int, int], Tensor]:
    used_bits = row_count * ROW_RECORD_BITS
    expected = _align(PALETTE_ENTRIES + (used_bits + 7) // 8)
    require(len(raw) == expected, "V11 palette-Scale18 stream alignment differs")
    palette = _validate_palette(tuple(int.from_bytes(raw[index : index + 1], "little", signed=True) for index in range(PALETTE_ENTRIES)))
    values = []
    for ordinal in range(row_count):
        value = 0
        for offset in range(ROW_RECORD_BITS):
            bit = ordinal * ROW_RECORD_BITS + offset
            value |= ((raw[PALETTE_ENTRIES + bit // 8] >> (bit % 8)) & 1) << offset
        require(SCALE32_SIGNIFICAND_MIN <= (value & 0xFFFF) <= SCALE32_SIGNIFICAND_MAX, "decoded V11 row significand escaped normalized Q1.15")
        values.append(value)
    for bit in range(used_bits, (len(raw) - PALETTE_ENTRIES) * 8):
        require(((raw[PALETTE_ENTRIES + bit // 8] >> (bit % 8)) & 1) == 0, "V11 palette-Scale18 padding bit is nonzero")
    return palette, torch.tensor(values, dtype=torch.int64)


def _scale32_records(palette: Sequence[int], records: Tensor) -> Tensor:
    palette_values = _validate_palette(palette)
    result = []
    for value in records.tolist():
        value = int(value)
        result.append(pack_scale32(value & 0xFFFF, palette_values[(value >> 16) & 0x3]))
    return torch.tensor(result, dtype=torch.int64)


def _block_ranges(width: int) -> tuple[tuple[int, int], ...]:
    return reference.input_block_ranges(width)


def _objective_exponent(base_shift: int) -> int:
    return min(base_shift + EXPONENT_MIN - 19, 2 * EXPONENT_MIN - 38)


def _exact_block_energies(source: Tensor, block_start: int, block_stop: int) -> list[int]:
    significands, shifts = v4._decode_bf16(source[:, block_start:block_stop])
    nonzero = significands != 0
    if bool(torch.any(nonzero)):
        base = int(shifts[nonzero].min())
    else:
        base = 0
    energies = []
    for row_significands, row_shifts in zip(significands.tolist(), shifts.tolist(), strict=True):
        energies.append(sum(int(sig) * int(sig) << (2 * (int(shift) - base)) for sig, shift in zip(row_significands, row_shifts, strict=True) if sig))
    return energies


def _energy_bucket_selectors(source: Tensor) -> tuple[Tensor, list[dict[str, Any]]]:
    rows, width = (int(value) for value in source.shape)
    blocks = _block_ranges(width)
    selectors = torch.empty((rows, len(blocks)), dtype=torch.uint8)
    summaries = []
    for block_index, (start, stop) in enumerate(blocks):
        energies = _exact_block_energies(source, start, stop)
        ranked = sorted(range(rows), key=lambda row: (energies[row], row))
        buckets = [[] for _ in range(CODEBOOK_BANKS)]
        for rank, row in enumerate(ranked):
            bank = min(CODEBOOK_BANKS - 1, rank * CODEBOOK_BANKS // rows)
            selectors[row, block_index] = bank
            buckets[bank].append(row)
        summaries.append({
            "block_index": block_index,
            "bucket_sizes": [len(bucket) for bucket in buckets],
            "ranked_rows_sha256": hashlib.sha256(b"".join(struct.pack("<I", row) for row in ranked)).hexdigest(),
        })
    return selectors, summaries


def _initial_codebooks(source: Tensor, scale32_records: Tensor, selectors: Tensor, base_shift: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    rows, width = (int(value) for value in source.shape)
    blocks = _block_ranges(width)
    codebooks = []
    fallback = tuple(range(-8, 8))
    for block_index, (start, stop) in enumerate(blocks):
        banks = []
        for bank in range(CODEBOOK_BANKS):
            selected_rows = [row for row in range(rows) if int(selectors[row, block_index]) == bank]
            if not selected_rows:
                banks.append(fallback)
                continue
            histogram = v4._initial_histogram(source[selected_rows, start:stop], scale32_records[selected_rows], base_shift)
            banks.append(tuple(v4.reference.construct_initial_codebook(histogram)["codepoints"]))
        codebooks.append(tuple(banks))
    return _validate_codebooks(codebooks, len(blocks))


def _selected_points(assignments: Tensor, codebooks: Sequence[Sequence[Sequence[int]]], selectors: Tensor) -> Tensor:
    rows, width = (int(value) for value in assignments.shape)
    points = torch.empty((rows, width), dtype=torch.int64)
    for block_index, (start, stop) in enumerate(_block_ranges(width)):
        for bank in range(CODEBOOK_BANKS):
            row_mask = selectors[:, block_index] == bank
            if bool(torch.any(row_mask)):
                codebook = torch.tensor(codebooks[block_index][bank], dtype=torch.int64)
                points[row_mask, start:stop] = codebook[assignments[row_mask, start:stop].to(torch.int64)]
    return points


def _assign_payload(
    source: Tensor,
    palette: Sequence[int],
    records: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor,
) -> Tensor:
    return _assign_payload_from_scale32(source, _scale32_records(palette, records), codebooks, selectors)


def _assign_payload_from_scale32(
    source: Tensor,
    scale_records: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor,
) -> Tensor:
    rows, width = (int(value) for value in source.shape)
    assignments = torch.empty((rows, width), dtype=torch.uint8)
    significands, exponents = v4._records_components(scale_records)
    scales = torch.ldexp(significands.to(torch.float64), exponents.to(torch.int32) - 15)
    for block_index, (start, stop) in enumerate(_block_ranges(width)):
        for bank in range(CODEBOOK_BANKS):
            row_indices = torch.nonzero(selectors[:, block_index] == bank, as_tuple=False).reshape(-1)
            if row_indices.numel() == 0:
                continue
            points = codebooks[block_index][bank]
            midpoint_sums = torch.tensor([left + right for left, right in zip(points, points[1:])], dtype=torch.float64)
            boundaries = scales[row_indices, None] * (midpoint_sums[None, :] / 32.0)
            selected = torch.searchsorted(boundaries.contiguous(), source[row_indices, start:stop].to(torch.float64).contiguous(), right=False)
            require(bool(torch.all((selected >= 0) & (selected < CODEBOOK_ENTRIES))), "V11 payload assignment escaped four bits")
            assignments[row_indices, start:stop] = selected.to(torch.uint8)
    return assignments


def _row_statistics(source: Tensor, assignments: Tensor, codebooks: Sequence[Sequence[Sequence[int]]], selectors: Tensor, base_shift: int) -> tuple[list[int], list[int], list[int]]:
    rows, width = (int(value) for value in source.shape)
    weight_codepoint = []
    codepoint_squared = []
    absolute_maximum = []
    for row_start, row_stop in v4._row_chunks(rows, width):
        significands, shifts = v4._decode_bf16(source[row_start:row_stop])
        delta = torch.where(significands == 0, torch.zeros_like(shifts), shifts - base_shift)
        require(bool(torch.all((delta >= 0) & (delta <= 54))), "BF16 tensor lattice span exceeds V11 row update")
        exact = torch.bitwise_left_shift(significands, delta)
        points = _selected_points(assignments[row_start:row_stop], codebooks, selectors[row_start:row_stop])
        maximum = int(torch.abs(exact).max())
        require(maximum * 127 * width < (1 << 63), "V11 row-update reduction may overflow int64")
        weight_codepoint.extend(int(value) for value in torch.sum(exact * points, dim=1).tolist())
        codepoint_squared.extend(int(value) for value in torch.sum(points * points, dim=1).tolist())
        absolute_maximum.extend(int(value) for value in torch.abs(exact).amax(dim=1).tolist())
    return weight_codepoint, codepoint_squared, absolute_maximum


def _compare_fixed_record(significand: int, exponent: int, numerator: int, denominator: int, shift: int) -> int:
    return v4._compare_dyadic(significand, 1, exponent - 15, numerator, denominator, shift)


def _best_record_for_exponent(
    exponent: int,
    weight_codepoint: int,
    codepoint_squared: int,
    absolute_maximum: int,
    base_shift: int,
    objective_exponent: int,
    prior_significand: int | None,
) -> tuple[int, int] | None:
    lower_numerator = absolute_maximum * 16
    lower_denominator = 127
    ideal_numerator = weight_codepoint * 16
    ideal_denominator = codepoint_squared
    if ideal_denominator <= 0 or ideal_numerator <= 0 or v4._compare_dyadic(ideal_numerator, ideal_denominator, base_shift, lower_numerator, lower_denominator, base_shift) < 0:
        target_numerator, target_denominator = lower_numerator, lower_denominator
    else:
        target_numerator, target_denominator = ideal_numerator, ideal_denominator
    shift = base_shift + 15 - exponent
    numerator = target_numerator << max(shift, 0)
    denominator = target_denominator << max(-shift, 0)
    floor_value, remainder = divmod(numerator, denominator)
    candidates = {
        SCALE32_SIGNIFICAND_MIN,
        SCALE32_SIGNIFICAND_MAX,
        floor_value - 1,
        floor_value,
        floor_value + int(remainder != 0),
        floor_value + 1,
    }
    if prior_significand is not None:
        candidates.add(prior_significand)
    filtered = sorted(
        value
        for value in candidates
        if SCALE32_SIGNIFICAND_MIN <= value <= SCALE32_SIGNIFICAND_MAX
        and _compare_fixed_record(value, exponent, lower_numerator, lower_denominator, base_shift) >= 0
    )
    if not filtered:
        return None
    choice = min(
        filtered,
        key=lambda significand: (
            v4._row_objective(pack_scale32(significand, exponent), weight_codepoint, codepoint_squared, base_shift, objective_exponent),
            significand,
        ),
    )
    return choice, v4._row_objective(pack_scale32(choice, exponent), weight_codepoint, codepoint_squared, base_shift, objective_exponent)


def _record_cost_table(
    statistics: tuple[Sequence[int], Sequence[int], Sequence[int]],
    prior_palette: Sequence[int] | None,
    prior_records: Tensor | None,
    base_shift: int,
    objective_exponent: int,
) -> list[dict[int, tuple[int, int]]]:
    prior_palette_values = _validate_palette(prior_palette) if prior_palette is not None else None
    prior_values = prior_records.tolist() if prior_records is not None else [None] * len(statistics[0])
    tables = []
    for row, (wp, p2, maximum, prior) in enumerate(zip(*statistics, prior_values, strict=True)):
        prior_by_exponent: dict[int, int] = {}
        if prior is not None and prior_palette_values is not None:
            prior = int(prior)
            prior_by_exponent[prior_palette_values[(prior >> 16) & 0x3]] = prior & 0xFFFF
        table = {}
        for exponent in LEGAL_EXPONENTS:
            choice = _best_record_for_exponent(
                exponent,
                int(wp),
                int(p2),
                int(maximum),
                base_shift,
                objective_exponent,
                prior_by_exponent.get(exponent),
            )
            if choice is not None:
                table[exponent] = choice
        require(table, f"V11 row cost table has no feasible exponent: {row}")
        tables.append(table)
    return tables


def _records_for_palette(tables: Sequence[dict[int, tuple[int, int]]], palette: Sequence[int]) -> tuple[Tensor, int] | None:
    palette_values = _validate_palette(palette)
    records = []
    objective = 0
    for table in tables:
        feasible = [(index, exponent) for index, exponent in enumerate(palette_values) if exponent in table]
        if not feasible:
            return None
        palette_index, exponent = min(
            feasible,
            key=lambda item: (table[item[1]][1], item[0], table[item[1]][0]),
        )
        significand, cost = table[exponent]
        records.append(significand | (palette_index << 16))
        objective += cost
    return torch.tensor(records, dtype=torch.int64), objective


def _initial_palette(tables: Sequence[dict[int, tuple[int, int]]]) -> tuple[int, int, int, int]:
    winners = [min(table, key=lambda exponent: (table[exponent][1], exponent, table[exponent][0])) for table in tables]
    counts = Counter(winners)
    selected = [exponent for exponent, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:PALETTE_ENTRIES]]
    while len(selected) < PALETTE_ENTRIES:
        remaining = [value for value in LEGAL_EXPONENTS if value not in selected]
        choice = min(remaining, key=lambda value: (min(abs(value - existing) for existing in selected) if selected else 0, value))
        selected.append(choice)
    # Legal row exponents form contiguous suffixes because the only feasibility
    # restriction is the exact no-overflow lower bound.  Frequency selection can
    # therefore miss a rare high-minimum row; minimally raise the largest entry
    # so every row remains encodable while retaining the other three winners.
    for table in tables:
        feasible = tuple(sorted(table))
        require(feasible == tuple(range(feasible[0], EXPONENT_MAX + 1)), "V11 row exponent feasibility is not a contiguous suffix")
    required_minimum = max(min(table) for table in tables)
    if required_minimum > max(selected):
        selected[selected.index(max(selected))] = required_minimum
    return _validate_palette(sorted(selected))


def _update_palette(tables: Sequence[dict[int, tuple[int, int]]], prior_palette: Sequence[int]) -> tuple[tuple[int, int, int, int], Tensor, int, dict[str, Any]]:
    palette = list(_validate_palette(prior_palette))
    changes = 0
    for coordinate in range(PALETTE_ENTRIES):
        candidates = []
        for exponent in LEGAL_EXPONENTS:
            proposal = palette.copy()
            proposal[coordinate] = exponent
            if len(set(proposal)) != PALETTE_ENTRIES or proposal != sorted(proposal):
                continue
            result = _records_for_palette(tables, proposal)
            if result is not None:
                records, objective = result
                candidates.append((objective, tuple(proposal), records))
        require(candidates, f"V11 palette coordinate has no legal candidate: {coordinate}")
        objective, selected, _records = min(candidates, key=lambda item: (item[0], item[1]))
        changes += int(tuple(palette) != selected)
        palette = list(selected)
    result = _records_for_palette(tables, palette)
    require(result is not None, "V11 selected palette cannot encode every row")
    records, objective = result
    return _validate_palette(palette), records, objective, {"coordinate_changes": changes, "palette": palette}


def _objective_from_statistics(palette: Sequence[int], records: Tensor, statistics: tuple[Sequence[int], Sequence[int], Sequence[int]], base_shift: int, objective_exponent: int) -> int:
    scale32 = _scale32_records(palette, records)
    return sum(
        v4._row_objective(int(record), int(wp), int(p2), base_shift, objective_exponent)
        for record, wp, p2 in zip(scale32.tolist(), statistics[0], statistics[1], strict=True)
    )


def _assign_selectors(
    source: Tensor,
    palette: Sequence[int],
    records: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    base_shift: int,
    objective_exponent: int,
) -> tuple[Tensor, int, dict[str, Any]]:
    rows, width = (int(value) for value in source.shape)
    selectors = torch.empty((rows, len(_block_ranges(width))), dtype=torch.uint8)
    scale32 = _scale32_records(palette, records)
    counts = [0] * CODEBOOK_BANKS
    ties = 0
    objective = 0
    for block_index, (start, stop) in enumerate(_block_ranges(width)):
        significands, shifts = v4._decode_bf16(source[:, start:stop])
        delta = torch.where(significands == 0, torch.zeros_like(shifts), shifts - base_shift)
        exact = torch.bitwise_left_shift(significands, delta)
        for row in range(rows):
            costs = []
            indices = assignments[row, start:stop].to(torch.int64)
            for bank in range(CODEBOOK_BANKS):
                points = torch.tensor(codebooks[block_index][bank], dtype=torch.int64)[indices]
                wp = int(torch.sum(exact[row] * points))
                p2 = int(torch.sum(points * points))
                costs.append(v4._row_objective(int(scale32[row]), wp, p2, base_shift, objective_exponent))
            choice = min(range(CODEBOOK_BANKS), key=lambda bank: (costs[bank], bank))
            ties += int(sum(cost == costs[choice] for cost in costs) > 1)
            selectors[row, block_index] = choice
            counts[choice] += 1
            objective += costs[choice]
    return selectors, objective, {"bank_counts": counts, "exact_tie_to_lower_bank_count": ties}


def _update_codebooks(
    source: Tensor,
    palette: Sequence[int],
    records: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor,
    shift_min: int,
    shift_max: int,
) -> tuple[tuple[tuple[int, ...], ...], ..., list[dict[str, Any]]]:
    rows, width = (int(value) for value in source.shape)
    scale32 = _scale32_records(palette, records)
    updated_blocks = []
    summaries = []
    for block_index, (start, stop) in enumerate(_block_ranges(width)):
        updated_banks = []
        bank_summaries = []
        for bank in range(CODEBOOK_BANKS):
            selected_rows = [row for row in range(rows) if int(selectors[row, block_index]) == bank]
            prior = codebooks[block_index][bank]
            if not selected_rows:
                updated = tuple(prior)
                summary = {"selected_row_count": 0, "empty_bank_retained": True}
            else:
                aggregates = v4._exact_aggregates(
                    source[selected_rows, start:stop],
                    scale32[selected_rows],
                    assignments[selected_rows, start:stop],
                    shift_min,
                    shift_max,
                )
                updated, update = v4._update_codebook(aggregates, prior)
                summary = {"selected_row_count": len(selected_rows), "empty_bank_retained": False, **update}
            updated_banks.append(updated)
            bank_summaries.append({"bank": bank, **summary})
        updated_blocks.append(tuple(updated_banks))
        summaries.append({"block_index": block_index, "banks": bank_summaries})
    return _validate_codebooks(updated_blocks, len(updated_blocks)), summaries


def _decode(state: dict[str, Any]) -> Tensor:
    records = _scale32_records(state["final_palette"], state["final_records"])
    significands, exponents = v4._records_components(records)
    scales = torch.ldexp(significands.to(torch.float64), exponents.to(torch.int32) - 15)
    points = _selected_points(state["final_assignments"], state["final_codebooks"], state["final_selectors"])
    return (points.to(torch.float64) * (scales[:, None] / 16.0)).to(torch.bfloat16)


def construct_four_codebook_state(source: Tensor, module_name: str) -> dict[str, Any]:
    require(source.ndim == 2 and source.dtype == torch.bfloat16, f"source weight format differs: {module_name}")
    rows, width = (int(value) for value in source.shape)
    blocks = _block_ranges(width)
    shift_min, shift_max, subnormal_count = v4._tensor_shift_range(source)
    objective_exponent = _objective_exponent(shift_min)
    initial_scale32, zero_rows = v4._initial_row_records(source, shift_min)
    selectors, energy_summary = _energy_bucket_selectors(source)
    codebooks = _initial_codebooks(source, initial_scale32, selectors, shift_min)

    assignments = _assign_payload_from_scale32(source, initial_scale32, codebooks, selectors)
    statistics = _row_statistics(source, assignments, codebooks, selectors, shift_min)
    tables = _record_cost_table(statistics, None, None, shift_min, objective_exponent)
    palette = _initial_palette(tables)
    initial_result = _records_for_palette(tables, palette)
    require(initial_result is not None, f"V11 initial palette cannot encode every row: {module_name}")
    records, _initial_palette_objective = initial_result
    assignments = _assign_payload(source, palette, records, codebooks, selectors)
    statistics = _row_statistics(source, assignments, codebooks, selectors, shift_min)
    previous_objective = _objective_from_statistics(palette, records, statistics, shift_min, objective_exponent)
    initial_objective = previous_objective
    trace = []

    for iteration in range(1, OUTER_ITERATIONS + 1):
        selectors, selector_objective, selector_summary = _assign_selectors(
            source, palette, records, assignments, codebooks, shift_min, objective_exponent
        )
        require(selector_objective <= previous_objective, f"V11 selector assignment increased exact source SSE: {module_name}")
        assignments = _assign_payload(source, palette, records, codebooks, selectors)
        payload_statistics = _row_statistics(source, assignments, codebooks, selectors, shift_min)
        payload_objective = _objective_from_statistics(palette, records, payload_statistics, shift_min, objective_exponent)
        require(payload_objective <= selector_objective, f"V11 payload assignment increased exact source SSE: {module_name}")

        tables = _record_cost_table(payload_statistics, palette, records, shift_min, objective_exponent)
        palette, records, palette_objective, palette_summary = _update_palette(tables, palette)
        require(palette_objective <= payload_objective, f"V11 palette/row-record step increased exact source SSE: {module_name}")
        assignments = _assign_payload(source, palette, records, codebooks, selectors)
        row_statistics = _row_statistics(source, assignments, codebooks, selectors, shift_min)
        row_objective = _objective_from_statistics(palette, records, row_statistics, shift_min, objective_exponent)
        require(row_objective <= palette_objective, f"V11 row-record payload refresh increased exact source SSE: {module_name}")

        codebooks, codebook_summary = _update_codebooks(
            source, palette, records, assignments, codebooks, selectors, shift_min, shift_max
        )
        assignments = _assign_payload(source, palette, records, codebooks, selectors)
        final_statistics = _row_statistics(source, assignments, codebooks, selectors, shift_min)
        final_objective = _objective_from_statistics(palette, records, final_statistics, shift_min, objective_exponent)
        require(final_objective <= row_objective, f"V11 codebook update increased exact source SSE: {module_name}")
        require(final_objective <= previous_objective, f"V11 outer iteration increased exact source SSE: {module_name}")

        payload = v7.v3.pack_codebook_indices(assignments)
        palette_scale18_raw = _pack_palette_scale18(palette, records)
        codebook_raw = _codebook_raw(codebooks)
        selector_raw = _pack_selectors(selectors)
        trace.append({
            "iteration": iteration,
            "prior_objective": v4._objective_record(previous_objective, objective_exponent),
            "after_selector_objective": v4._objective_record(selector_objective, objective_exponent),
            "after_payload_objective": v4._objective_record(payload_objective, objective_exponent),
            "after_palette_row_record_objective": v4._objective_record(row_objective, objective_exponent),
            "final_objective": v4._objective_record(final_objective, objective_exponent),
            "selector_assignment": selector_summary,
            "palette_update": palette_summary,
            "four_codebook_updates": codebook_summary,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "palette_scale18_stream_sha256": hashlib.sha256(palette_scale18_raw).hexdigest(),
            "four_codebook_stream_sha256": hashlib.sha256(codebook_raw).hexdigest(),
            "selector_stream_sha256": hashlib.sha256(selector_raw).hexdigest(),
        })
        previous_objective = final_objective

    payload = v7.v3.pack_codebook_indices(assignments)
    palette_scale18_raw = _pack_palette_scale18(palette, records)
    codebook_raw = _codebook_raw(codebooks)
    selector_raw = _pack_selectors(selectors)
    return {
        "module_name": module_name,
        "shape": [rows, width],
        "shift_min": shift_min,
        "shift_max": shift_max,
        "subnormal_count": subnormal_count,
        "zero_row_count": zero_rows,
        "input_block_count": len(blocks),
        "energy_bucket_initialization": energy_summary,
        "initial_objective": v4._objective_record(initial_objective, objective_exponent),
        "outer_iterations_completed": len(trace),
        "final_palette": palette,
        "final_records": records,
        "final_codebooks": codebooks,
        "final_selectors": selectors,
        "final_assignments": assignments,
        "payload": payload,
        "palette_scale18_raw": palette_scale18_raw,
        "codebook_raw": codebook_raw,
        "selector_raw": selector_raw,
        "final_objective": v4._objective_record(previous_objective, objective_exponent),
        "iteration_trace": trace,
        "iteration_trace_sha256": canonical_sha256(trace),
    }


def _state_hashes(state: dict[str, Any]) -> dict[str, str]:
    reconstruction = _decode(state)
    return {
        "payload_sha256": hashlib.sha256(state["payload"]).hexdigest(),
        "palette_scale18_sha256": hashlib.sha256(state["palette_scale18_raw"]).hexdigest(),
        "four_codebook_sha256": hashlib.sha256(state["codebook_raw"]).hexdigest(),
        "selector_sha256": hashlib.sha256(state["selector_raw"]).hexdigest(),
        "reconstruction_bf16_sha256": v7.v3.tensor_sha256(reconstruction),
        "objective_sha256": canonical_sha256(state["final_objective"]),
        "iteration_trace_sha256": state["iteration_trace_sha256"],
    }


def _tensor_storage_bytes(rows: int, width: int) -> dict[str, int]:
    blocks = len(_block_ranges(width))
    payload = rows * width // 2
    palette_scale18 = _align(PALETTE_ENTRIES + (rows * ROW_RECORD_BITS + 7) // 8)
    codebooks = blocks * CODEBOOK_BANKS * CODEBOOK_ENTRIES
    selectors = _align((rows * blocks * SELECTOR_BITS + 7) // 8)
    return {
        "payload": payload,
        "palette_scale18": palette_scale18,
        "four_codebook": codebooks,
        "selector": selectors,
        "total": payload + palette_scale18 + codebooks + selectors,
    }


def _storage_closure() -> dict[str, Any]:
    # Qwen2.5-0.5B uses 896-wide Q/O projections and 128-row GQA K/V
    # projections; treating K/V as 896 rows silently overstates every stream.
    standard_shapes = [(896, 896)] * 2 + [(128, 896)] * 2 + [(4864, 896)] * 2 + [(896, 4864)]
    standard_parts = [_tensor_storage_bytes(rows, width) for rows, width in standard_shapes]
    standard = {key: sum(item[key] for item in standard_parts) for key in standard_parts[0]}
    lm_head = _tensor_storage_bytes(151936, 896)
    whole = {key: standard[key] * 24 + lm_head[key] for key in standard}
    require(standard == {"payload": 7454720, "palette_scale18": 28624, "four_codebook": 5120, "selector": 29120, "total": 7517584}, "V11 standard-layer byte closure differs")
    require(lm_head == {"payload": 68067328, "palette_scale18": 341872, "four_codebook": 448, "selector": 265888, "total": 68675536}, "V11 lm_head byte closure differs")
    require(whole == {"payload": 246980608, "palette_scale18": 1028848, "four_codebook": 123328, "selector": 964768, "total": 249097552}, "V11 whole-model byte closure differs")
    return {"standard_decoder_layer": standard, "lm_head": lm_head, "whole_model": whole}


def _validate_state(state: dict[str, Any]) -> dict[str, Any]:
    rows, width = state["shape"]
    blocks = _block_ranges(width)
    codebooks = _validate_codebooks(state["final_codebooks"], len(blocks))
    palette, records = _unpack_palette_scale18(state["palette_scale18_raw"], rows)
    require(palette == state["final_palette"] and torch.equal(records, state["final_records"]), "V11 palette-Scale18 round trip differs")
    require(torch.equal(_unpack_selectors(state["selector_raw"], rows, len(blocks)), state["final_selectors"]), "V11 selector round trip differs")
    require(state["final_assignments"].shape == (rows, width), "V11 payload geometry differs")
    require(bool(torch.all(state["final_assignments"] < CODEBOOK_ENTRIES)), "V11 payload escaped four bits")
    require(len(state["payload"]) == rows * width // 2, "V11 payload byte count differs")
    require(len(state["codebook_raw"]) == len(blocks) * CODEBOOK_BANKS * CODEBOOK_ENTRIES, "V11 codebook byte count differs")
    return {
        "rows": rows,
        "input_features": width,
        "input_block_count": len(blocks),
        "palette_scale18_bytes": len(state["palette_scale18_raw"]),
        "four_codebook_bytes": len(state["codebook_raw"]),
        "selector_bytes": len(state["selector_raw"]),
        "all_codebooks_strictly_increasing": all(
            all(left < right for left, right in zip(bank, bank[1:])) for banks in codebooks for bank in banks
        ),
    }


def combined_self_test() -> dict[str, Any]:
    environment = {
        **reference.require_project_python(),
        "platform": platform.platform(),
        "packages": {"torch": importlib.metadata.version("torch")},
        "backend": "exact BF16 dyadic four-bank/palette-Scale18 constructor kernel",
    }
    reference.verify_companion(TASK_PATH, TASK_COMPANION)
    reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "frozen V11 Engineer authority differs")
    root_existed_before = OUTPUT_DIR.exists()
    marker_existed_before = MARKER_PATH.exists()
    require(not ATTEMPT_DIR.exists() and not marker_existed_before, "V11 official attempt must remain absent during constructor self-test")

    storage = _storage_closure()
    require(len(_block_ranges(896)) == 7 and len(_block_ranges(4864)) == 38, "V11 input-block geometry differs")
    illegal_width_rejected = reference.expect_rejected(lambda: _block_ranges(1024), "illegal V11 width was accepted")

    palette = (-24, -8, 0, 4)
    synthetic_records = torch.tensor([
        SCALE32_SIGNIFICAND_MIN | (0 << 16),
        SCALE32_SIGNIFICAND_MAX | (1 << 16),
        0x9234 | (2 << 16),
        0xCDEF | (3 << 16),
        0x8001 | (1 << 16),
    ], dtype=torch.int64)
    scale18_raw = _pack_palette_scale18(palette, synthetic_records)
    unpacked_palette, unpacked_records = _unpack_palette_scale18(scale18_raw, len(synthetic_records))
    require(unpacked_palette == palette and torch.equal(unpacked_records, synthetic_records), "synthetic V11 Scale18 round trip differs")
    tampered_scale18 = bytearray(scale18_raw)
    tampered_scale18[-1] |= 0x80
    nonzero_scale18_padding_rejected = reference.expect_rejected(
        lambda: _unpack_palette_scale18(bytes(tampered_scale18), len(synthetic_records)),
        "nonzero V11 Scale18 padding was accepted",
    )
    truncated_scale18_rejected = reference.expect_rejected(
        lambda: _unpack_palette_scale18(scale18_raw[:-1], len(synthetic_records)),
        "truncated V11 Scale18 stream was accepted",
    )
    extended_scale18_rejected = reference.expect_rejected(
        lambda: _unpack_palette_scale18(scale18_raw + b"\x00" * ALIGNMENT_BYTES, len(synthetic_records)),
        "extended V11 Scale18 stream was accepted",
    )

    synthetic_selectors = torch.tensor([[0, 1, 2, 3, 3, 2, 1], [1, 2, 3, 0, 0, 3, 2]], dtype=torch.uint8)
    selector_raw = _pack_selectors(synthetic_selectors)
    require(torch.equal(_unpack_selectors(selector_raw, 2, 7), synthetic_selectors), "synthetic V11 selector round trip differs")
    require(selector_raw[0] == 0b11100100, "V11 selector field order is not row-major/block-major LSB-first")
    tampered_selector = bytearray(selector_raw)
    tampered_selector[-1] |= 0x80
    nonzero_selector_padding_rejected = reference.expect_rejected(
        lambda: _unpack_selectors(bytes(tampered_selector), 2, 7),
        "nonzero V11 selector padding was accepted",
    )

    lanes = torch.arange(896, dtype=torch.int64)
    fixture_rows = []
    for row in range(8):
        base = (((lanes * (row + 3)) % 59) - 29).to(torch.float64) / (128.0 * (1 + row % 3))
        fixture_rows.append(torch.where((lanes % (row + 5)) == 0, -base * (row + 1) / 4.0, base))
    fixture_rows[-1] = torch.zeros_like(fixture_rows[-1])
    fixture = torch.stack(fixture_rows).to(torch.bfloat16)
    candidate_a = construct_four_codebook_state(fixture, "self_test.q_proj")
    candidate_b = construct_four_codebook_state(fixture, "self_test.q_proj")
    geometry = _validate_state(candidate_a)
    hashes_a = _state_hashes(candidate_a)
    require(hashes_a == _state_hashes(candidate_b), "independent V11 fixture state hashes differ")
    require(candidate_a["iteration_trace"] == candidate_b["iteration_trace"], "independent V11 fixture iteration traces differ")
    require(candidate_a["outer_iterations_completed"] == OUTER_ITERATIONS, "V11 fixture iteration count differs")
    require(
        all(int(item["final_objective"]["scaled_integer"]) <= int(item["prior_objective"]["scaled_integer"]) for item in candidate_a["iteration_trace"]),
        "V11 fixture exact objective is not monotonic",
    )

    duplicate_codebooks = tuple(tuple(tuple(range(-8, 8)) for _bank in range(CODEBOOK_BANKS)) for _block in range(7))
    tie_assignments = torch.full(fixture.shape, 8, dtype=torch.uint8)
    tie_records = torch.full((fixture.shape[0],), SCALE32_SIGNIFICAND_MIN, dtype=torch.int64)
    tie_selectors, _tie_objective, tie_summary = _assign_selectors(
        torch.zeros_like(fixture), (-24, -23, -22, -21), tie_records, tie_assignments, duplicate_codebooks, -20, _objective_exponent(-20)
    )
    require(not bool(torch.any(tie_selectors)), "V11 exact selector ties did not choose bank zero")

    empty_selectors = torch.zeros_like(candidate_a["final_selectors"])
    retained_codebooks, empty_summary = _update_codebooks(
        fixture,
        candidate_a["final_palette"],
        candidate_a["final_records"],
        candidate_a["final_assignments"],
        candidate_a["final_codebooks"],
        empty_selectors,
        candidate_a["shift_min"],
        candidate_a["shift_max"],
    )
    require(
        all(retained_codebooks[block][bank] == candidate_a["final_codebooks"][block][bank] for block in range(7) for bank in (1, 2, 3)),
        "V11 empty bank did not retain its prior codebook",
    )
    require(all(all(bank["empty_bank_retained"] for bank in block["banks"][1:]) for block in empty_summary), "V11 empty-bank summary differs")

    metadata_sequence = [
        {"block_index": block, "bank": bank, "beat_bytes": 16}
        for block in range(7)
        for bank in range(CODEBOOK_BANKS)
    ]
    require([(item["block_index"], item["bank"]) for item in metadata_sequence[:6]] == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)], "V11 codebook metadata order differs")

    build_root = ROOT / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v11-hash-rejection-", dir=build_root) as temporary_directory:
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
            "tampered copied V11 artifact passed companion verification",
        )

    require(OUTPUT_DIR.exists() == root_existed_before, "self-test changed the V11 namespace")
    require(MARKER_PATH.exists() == marker_existed_before is False, "self-test changed the V11 execution marker")
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
            "legal_width_block_counts": {"896": 7, "4864": 38},
            "illegal_width_rejected": illegal_width_rejected,
            "exact_storage_byte_closure": storage,
            "palette_order_range_and_signed_extrema": True,
            "scale18_lsb_first_cross_byte_round_trip": True,
            "scale18_truncation_rejected": truncated_scale18_rejected,
            "scale18_extension_rejected": extended_scale18_rejected,
            "scale18_nonzero_padding_rejected": nonzero_scale18_padding_rejected,
            "selector_row_major_block_major_lsb_first": True,
            "selector_nonzero_padding_rejected": nonzero_selector_padding_rejected,
            "energy_bucket_initialization": True,
            "four_codebooks_strictly_increasing": geometry["all_codebooks_strictly_increasing"],
            "independent_candidate_a_b_state_and_trace_match": True,
            "outer_iterations_completed": OUTER_ITERATIONS,
            "monotonic_exact_source_objective": True,
            "payload_nibble_packing_reused_from_reviewed_v7": True,
            "all_zero_row_supported": candidate_a["zero_row_count"] >= 1,
            "palette_coordinate_updates_exercised": True,
            "exact_selector_ties_choose_lower_bank": tie_summary["exact_tie_to_lower_bank_count"] == tie_selectors.numel(),
            "empty_banks_retain_prior_codebooks": True,
            "metadata_sequence": metadata_sequence,
            "bound_hash_mismatch_rejected": hash_mismatch_rejected,
        },
        "fixture": {
            "shape": list(fixture.shape),
            "geometry": geometry,
            "bf16_reconstruction_sse": fixture_sse,
            "initial_objective": candidate_a["initial_objective"],
            "final_objective": candidate_a["final_objective"],
            **hashes_a,
        },
        "official_build_namespace_changed": False,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": True,
        "claim_boundary": "constructor kernel, exact format/byte closure, and synthetic fixture self-test only; no 169-tensor, 8/8 quality, 312-event, PREATTEMPT_READY, official-attempt, RTL, synthesis, timing, or PPA conclusion",
    }


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


def _decoded_chunk_v11(state: dict[str, Any], row_start: int, row_stop: int) -> Tensor:
    records = _scale32_records(state["final_palette"], state["final_records"][row_start:row_stop])
    significands, exponents = v4._records_components(records)
    scales = torch.ldexp(significands.to(torch.float64), exponents.to(torch.int32) - 15)
    points = _selected_points(
        state["final_assignments"][row_start:row_stop],
        state["final_codebooks"],
        state["final_selectors"][row_start:row_stop],
    )
    return (points.to(torch.float64) * (scales[:, None] / 16.0)).to(torch.bfloat16)


def _measure_and_materialize(
    module: nn.Linear,
    source: Tensor,
    state: dict[str, Any],
    sealed_v10_record: dict[str, Any],
) -> dict[str, Any]:
    rows, width = (int(value) for value in source.shape)
    v11_sse = 0.0
    v11_digest = hashlib.sha256()
    with torch.no_grad():
        for row_start, row_stop in v4._row_chunks(rows, width):
            decoded = _decoded_chunk_v11(state, row_start, row_stop)
            source64 = source[row_start:row_stop].to(torch.float64)
            difference = source64 - decoded.to(torch.float64)
            v11_sse += float(torch.sum(difference * difference))
            v11_digest.update(v7.v3.raw_tensor_bytes(decoded))
            module.weight[row_start:row_stop].copy_(decoded)
    v11_sha = v11_digest.hexdigest()
    v10_sse = float(sealed_v10_record["v10_packed_scale24_per_row_selector_bf16_sse"])
    require(v10_sse >= 0.0 and v11_sse >= 0.0, f"negative BF16 reconstruction SSE: {sealed_v10_record['module']}")
    require(v11_sha == v7.v3.tensor_sha256(module.weight), f"V11 materialized reconstruction hash differs: {sealed_v10_record['module']}")
    return {
        "module": sealed_v10_record["module"],
        "weight_count": int(source.numel()),
        "sealed_v10_bf16_sse": v10_sse,
        "sealed_v10_reconstruction_bf16_sha256": sealed_v10_record["reconstruction_bf16_sha256"],
        "v11_four_codebook_palette_scale18_bf16_sse": v11_sse,
        "v11_non_regressing_against_sealed_v10": v11_sse <= v10_sse,
        "v11_strictly_improved_against_sealed_v10": v11_sse < v10_sse,
        "reconstruction_bf16_sha256": v11_sha,
    }


def _policy_and_format() -> dict[str, Any]:
    body = {
        "payload": {
            "bits_per_weight": 4,
            "semantics": "unsigned selected-bank codebook index 0 through 15",
            "packing": "row-major; even flat element low nibble; odd flat element high nibble",
        },
        "palette_scale18": {
            "palette_entries_per_tensor": PALETTE_ENTRIES,
            "palette_entry_format": "strictly increasing signed int8 Scale32 exponents in [-24,+4]",
            "row_record_bits": ROW_RECORD_BITS,
            "row_record_format": "16-bit normalized unsigned Q1.15 significand plus two-bit palette index",
            "packing": "four palette bytes followed by row-major LSB-first 18-bit records, 16-byte tensor alignment, zero padding",
        },
        "four_codebook": {
            "banks_per_input_block": CODEBOOK_BANKS,
            "input_block_lanes": INPUT_BLOCK_LANES,
            "entries_per_bank": CODEBOOK_ENTRIES,
            "bytes_per_input_block": CODEBOOK_BANKS * CODEBOOK_ENTRIES,
            "order": "bank 0, bank 1, bank 2, bank 3 for each increasing input block",
            "format": "strictly increasing signed Q4.4 int8",
        },
        "selector": {
            "bits_per_output_row_per_input_block": SELECTOR_BITS,
            "packing": "output-row-major then input-block-major, least-significant-bit first, 16-byte tensor alignment, zero padding",
        },
        "construction": {
            "source": "pinned BF16 weights only; sealed V10 values are comparison ledger data",
            "outer_iterations": OUTER_ITERATIONS,
            "optimizer_cost": "constant-elided exact dyadic source-space ordering cost",
            "reported_model_gate": "literal nonnegative BF16-materialized reconstruction SSE",
        },
        "activation": {
            "policy": "grouped Dynamic Scale32",
            "event_count": EXPECTED_EVENTS,
            "maximum_transport_metadata_bytes_per_decoder_layer_token": 832,
        },
        "storage": {
            "whole_model_total_weight_bytes": 249097552,
            "whole_model_exact_bits_per_weight": 4.034285185661216,
            "maximum_total_stored_bits_per_weight": 4.036,
        },
        "sealed_v10_backend_sha256": reference.sha256_file(Path(v10.__file__).resolve()),
    }
    return {**body, "policy_and_format_sha256": canonical_sha256(body)}


def _quantize_model(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {record["module"]: record for record in v7.grouped.expected_transformer_tensors()}
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == 169, "Qwen linear count differs")
    require([name for name, _module in modules if name != "lm_head"] == list(expected), "transformer linear order differs")
    require(reference.sha256_file(V10_RECONSTRUCTION_PATH) == V10_RECONSTRUCTION_SHA256, "sealed V10 reconstruction hash differs")
    sealed_report = load_json(V10_RECONSTRUCTION_PATH)
    sealed_records = {item["module"]: item for item in sealed_report["tensors"]}
    require(list(sealed_records) == [name for name, _module in modules], "sealed V10 tensor order differs")

    manifest_names = (
        "payload",
        "palette_scale18",
        "four_codebook",
        "selector",
        "reconstruction",
        "objective",
        "iteration_trace",
    )
    manifests: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest_names}
    weight_manifest: list[dict[str, Any]] = []
    streams = {name: hashlib.sha256() for name in ("payload", "palette_scale18", "four_codebook", "selector", "reconstruction")}
    totals = {"weight_count": 0, "payload": 0, "palette_scale18": 0, "four_codebook": 0, "selector": 0}

    for ordinal, (name, module) in enumerate(modules, start=1):
        started = time.monotonic()
        print(f"V11_CONSTRUCT_START {ordinal}/169 {name}", flush=True)
        source = module.weight.detach()
        source_sha = v7.v3.tensor_sha256(source)
        state = construct_four_codebook_state(source, name)
        geometry = _validate_state(state)
        reconstruction = _measure_and_materialize(module, source, state, sealed_records[name])
        raw = {
            "payload": state["payload"],
            "palette_scale18": state["palette_scale18_raw"],
            "four_codebook": state["codebook_raw"],
            "selector": state["selector_raw"],
        }
        component_records = {
            component: {"module": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            for component, value in raw.items()
        }
        component_records["palette_scale18"].update(
            {
                "palette": list(state["final_palette"]),
                "row_record_count": int(source.shape[0]),
                "row_record_bits": ROW_RECORD_BITS,
                "alignment_bytes": ALIGNMENT_BYTES,
                "padding_bits_zero": True,
            }
        )
        component_records["four_codebook"].update(
            {
                "input_block_count": state["input_block_count"],
                "bank_count": state["input_block_count"] * CODEBOOK_BANKS,
                "q4_4_banks_by_input_block": [[list(bank) for bank in banks] for banks in state["final_codebooks"]],
            }
        )
        component_records["selector"].update(
            {
                "output_row_count": int(source.shape[0]),
                "selector_bit_count": int(source.shape[0]) * state["input_block_count"] * SELECTOR_BITS,
                "alignment_bytes": ALIGNMENT_BYTES,
                "padding_bits_zero": True,
            }
        )
        for component in ("payload", "palette_scale18", "four_codebook", "selector"):
            manifests[component].append(component_records[component])
            streams[component].update(name.encode() + b"\0" + raw[component])
            totals[component] += len(raw[component])
        streams["reconstruction"].update(name.encode() + b"\0" + bytes.fromhex(reconstruction["reconstruction_bf16_sha256"]))
        manifests["reconstruction"].append(reconstruction)
        objective_record = {
            "module": name,
            "kind": "constant_elided_exact_source_space_optimizer_ordering_cost_not_literal_sse",
            "initial_v11_optimizer_cost": state["initial_objective"],
            "final_v11_optimizer_cost": state["final_objective"],
            "actual_nonnegative_bf16_sse": {
                "sealed_v10": reconstruction["sealed_v10_bf16_sse"],
                "v11": reconstruction["v11_four_codebook_palette_scale18_bf16_sse"],
            },
        }
        manifests["objective"].append(objective_record)
        trace_record = {
            "module": name,
            "source_bf16_sha256": source_sha,
            "shift_min": state["shift_min"],
            "shift_max": state["shift_max"],
            "input_block_count": state["input_block_count"],
            "final_palette": list(state["final_palette"]),
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
            "metadata_sequence": ["four_codebook_banks_0_through_3_by_input_block", "quaternary_selector", "payload", "palette_scale18"],
        }
        weight_manifest.append(record)
        totals["weight_count"] += int(source.numel())
        print(
            f"V11_CONSTRUCT_DONE {ordinal}/169 {name} seconds={time.monotonic()-started:.3f} "
            f"v10_sse={reconstruction['sealed_v10_bf16_sse']:.12g} v11_sse={reconstruction['v11_four_codebook_palette_scale18_bf16_sse']:.12g}",
            flush=True,
        )
        del state
        gc.collect()

    aggregate_v10 = float(sum(item["sealed_v10_bf16_sse"] for item in manifests["reconstruction"]))
    aggregate_v11 = float(sum(item["v11_four_codebook_palette_scale18_bf16_sse"] for item in manifests["reconstruction"]))
    non_regressing_count = sum(bool(item["v11_non_regressing_against_sealed_v10"]) for item in manifests["reconstruction"])
    improved_count = sum(bool(item["v11_strictly_improved_against_sealed_v10"]) for item in manifests["reconstruction"])
    require(aggregate_v10 == V10_AGGREGATE_BF16_SSE, "sealed aggregate V10 BF16 SSE differs")
    require(all(item["sealed_v10_bf16_sse"] >= 0.0 and item["v11_four_codebook_palette_scale18_bf16_sse"] >= 0.0 for item in manifests["reconstruction"]), "V11 report contains negative BF16 SSE")
    reconstruction_body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "measurement": "literal nonnegative sum-squared error from pinned BF16 source to BF16-materialized reconstruction",
        "optimizer_cost_note": "scaled_integer optimizer fields omit the source-squared constant and are not reported as SSE",
        "reference": {
            "candidate_id": v10.CANDIDATE_ID,
            "artifact": file_record(V10_RECONSTRUCTION_PATH),
            "aggregate_bf16_sse": V10_AGGREGATE_BF16_SSE,
            "sealed_ledger_only": True,
        },
        "tensor_count": len(manifests["reconstruction"]),
        "per_tensor_non_regressing_count": non_regressing_count,
        "per_tensor_strictly_improved_count": improved_count,
        "all_169_tensors_non_regressing_against_sealed_v10": non_regressing_count == 169,
        "aggregate_sealed_v10_bf16_sse": aggregate_v10,
        "aggregate_v11_bf16_sse": aggregate_v11,
        "aggregate_strictly_improved_against_sealed_v10": aggregate_v11 < aggregate_v10,
        "tensors": manifests["reconstruction"],
    }
    passed = reconstruction_body["all_169_tensors_non_regressing_against_sealed_v10"] and reconstruction_body["aggregate_strictly_improved_against_sealed_v10"]
    reconstruction_report = {
        **reconstruction_body,
        "status": "PASS_169_OF_169_V10_LEDGER_WITH_STRICT_AGGREGATE_IMPROVEMENT" if passed else "NO_GO_V10_LEDGER_MODEL_ONLY_RECONSTRUCTION_GATE",
        "reconstruction_report_sha256": canonical_sha256(reconstruction_body),
    }
    require(
        totals
        == {
            "weight_count": 493961216,
            "payload": 246980608,
            "palette_scale18": 1028848,
            "four_codebook": 123328,
            "selector": 964768,
        },
        f"whole-model V11 storage geometry differs: {totals}",
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
        "total_palette_scale18_bytes": totals["palette_scale18"],
        "total_four_codebook_bytes": totals["four_codebook"],
        "total_selector_bytes": totals["selector"],
        "whole_model_total_weight_bytes": sum(totals[name] for name in ("payload", "palette_scale18", "four_codebook", "selector")),
        "whole_model_exact_bits_per_weight": 4.034285185661216,
        "standard_decoder_layer_exact_bits_per_weight": 4.033731112637363,
        "lm_head_exact_bits_per_weight": 4.035741552834276,
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
    require(model.model.embed_tokens.weight is embedding and embedding.data_ptr() == source_pointer, "embedding identity changed after V11 construction")
    require(list(embedding.shape) == source_shape and str(embedding.dtype) == source_dtype, "embedding shape or dtype changed")
    embedding_hash = v7.v3.tensor_sha256(embedding)
    require(embedding_hash == source_hash == SOURCE_EMBEDDING_SHA256, "embedding bytes changed after V11 construction")
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
        "module_counts": {"transformer_linears": 168, "lm_head": 1, "total_four_codebook_linears": 169},
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
        "post_v11": {
            "embedding_bf16_sha256": embedding_hash,
            "embedding_hash_matches_source": embedding_hash == SOURCE_EMBEDDING_SHA256,
            "lm_head_storage_is_separate": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
            "transformer_four_codebook_count": 168,
            "lm_head_four_codebook_count": 1,
        },
    }
    return model, manifest, witness, construction["reconstruction"]


def _construct_candidate_artifacts(manifest_path: str, witness_path: str, reconstruction_path: str) -> None:
    torch.set_num_threads(CONSTRUCTION_TORCH_THREADS)
    torch.set_num_interop_threads(1)
    print("V11_CANDIDATE_A_WORKER_START", flush=True)
    candidate, manifest, witness, reconstruction = _load_candidate("candidate_a_independent_construction")
    write_json(Path(manifest_path), manifest)
    write_json(Path(witness_path), witness)
    write_json(Path(reconstruction_path), reconstruction)
    del candidate
    gc.collect()
    print("V11_CANDIDATE_A_WORKER_DONE", flush=True)


def _compare_candidates(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> dict[str, Any]:
    comparisons = {key: candidate_a["identity"][key] == candidate_b["identity"][key] for key in STATE_MATCH_KEYS}
    require(all(comparisons.values()), f"candidate A/B identity differs: {comparisons}")
    manifest_fields = tuple(
        f"{name}_manifest"
        for name in ("payload", "palette_scale18", "four_codebook", "selector", "reconstruction", "objective", "iteration_trace")
    ) + ("weight_manifest", "state_manifest")
    field_matches = {field: candidate_a[field] == candidate_b[field] for field in manifest_fields}
    require(all(field_matches.values()), f"candidate A/B whole-model manifests differ: {field_matches}")
    return {"identity_matches": comparisons, "manifest_matches": field_matches, "all_match": True}


def _validate_reconstruction(report: dict[str, Any]) -> bool:
    return (
        report["tensor_count"] == 169
        and report["per_tensor_non_regressing_count"] == 169
        and report["aggregate_sealed_v10_bf16_sse"] == V10_AGGREGATE_BF16_SSE
        and report["aggregate_v11_bf16_sse"] < V10_AGGREGATE_BF16_SSE
        and report["all_169_tensors_non_regressing_against_sealed_v10"] is True
        and report["aggregate_strictly_improved_against_sealed_v10"] is True
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
        require(not root_existed, "fresh V11 namespace already exists")
    markers = list(OUTPUT_DIR.rglob("execution_started.json")) if root_existed else []
    require(not markers and not ATTEMPT_DIR.exists(), "V11 official attempt or marker exists")
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
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "cannot recover a V11 namespace with an official attempt")
    entries = sorted(OUTPUT_DIR.iterdir())
    if not entries:
        OUTPUT_DIR.rmdir()
        return {"recovered": True, "fresh_namespace": True, "archived_artifacts": []}
    require(not CONTRACT_PATH.exists() and not READY_PATH.exists() and not NO_GO_PATH.exists(), "completed V11 preflight may not be recovered")
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
        "v10_review_record": V10_REVIEW_PATH,
        "v10_reconstruction_ledger": V10_RECONSTRUCTION_PATH,
        "source_contract": SOURCE_CONTRACT_PATH,
        "calibration_contract": CALIBRATION_CONTRACT_PATH,
        "calibration_report": CALIBRATION_REPORT_PATH,
        "base_scales": BASE_SCALES_PATH,
        "image_manifest": IMAGE_DIR / "manifest.json",
        "official_matrix_hash_only": MATRIX_PATH,
        "runner_source": RUNNER_PATH,
        "scalable_backend_source": BACKEND_PATH,
        "reviewed_v10_runner": Path(v10.reference.__file__).resolve(),
        "reviewed_v10_backend": Path(v10.__file__).resolve(),
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
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "V11 Engineer authority differs")
    require(task["fresh_attempt_namespace"]["attempt_budget"] == 1 and task["fresh_attempt_namespace"]["attempts_consumed_at_task_freeze"] == 0, "V11 attempt accounting differs")
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
            "terminal_v10_review": artifacts["v10_review_record"],
            "attempt_id": "attempt-0001",
            "exact_authorized_candidate_attempts": 1,
            "attempts_consumed": 0,
            "selected_policy_id": None,
        },
        "source_model": source_identity,
        "environment": environment,
        "format_and_policy": _policy_and_format(),
        "construction": {
            "backend": "fresh exact BF16 four-codebook/quaternary-selector/palette-Scale18 construction",
            "source_tied_then_runtime_split_lm_head": True,
            "source_embedding_bf16_sha256": SOURCE_EMBEDDING_SHA256,
            "embedding_bytes_preserved": True,
            "transformer_linears": 168,
            "separated_lm_head_linears": 1,
            "outer_iterations": OUTER_ITERATIONS,
            "candidate_a_and_b_whole_model_manifests_match": state_comparison,
            "palette_scale18_round_trip_required": True,
            "optimizer_cost_is_not_literal_sse": True,
            "model_gate_uses_actual_nonnegative_bf16_sse": True,
        },
        "pre_attempt_gates": {
            "v10_ledger_and_169_model_closure": {"passed": gate_passes["v10_ledger_and_169_model_closure"], "artifact": artifacts["model_only_reconstruction"]},
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
    require(CONTRACT_PATH.is_file(), "V11 predeclared contract is missing")
    require(READY_PATH.is_file() != NO_GO_PATH.is_file(), "exactly one V11 pre-attempt terminal record is required")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "V11 predeclared contract wrapper differs")
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "V11 predeclared contract canonical hash differs")
    contract = wrapper["contract"]
    require(contract["candidate_id"] == CANDIDATE_ID and contract["task_id"] == TASK_ID, "V11 predeclared authority differs")
    require(contract["artifacts"]["runner_source"]["sha256"] == reference.sha256_file(RUNNER_PATH), "V11 runner changed after freeze")
    require(contract["artifacts"]["scalable_backend_source"]["sha256"] == reference.sha256_file(BACKEND_PATH), "V11 backend changed after freeze")
    for record in contract["artifacts"].values():
        _verify_bound_record(record)
    reference.verify_companion(TASK_PATH, TASK_COMPANION)
    reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    require(reference.sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(reference.sha256_file(V10_RECONSTRUCTION_PATH) == V10_RECONSTRUCTION_SHA256, "sealed V10 reconstruction hash differs")
    require(reference.sha256_file(V10_REVIEW_PATH) == V10_REVIEW_SHA256, "sealed V10 review hash differs")
    require(verify_source_snapshot() == contract["source_model"], "source snapshot identity differs")
    packages = {**verify_versions(), "numpy": importlib.metadata.version("numpy")}
    require(packages == contract["environment"]["packages"], "package versions differ")
    require(combined_self_test() == load_json(SELF_TEST_PATH), "V11 runner self-test result differs")
    candidate_a = load_json(CANDIDATE_A_PATH)
    candidate_b = load_json(CANDIDATE_B_PATH)
    comparison = _compare_candidates(candidate_a, candidate_b)
    witness = load_json(ALIAS_WITNESS_PATH)
    require(witness["state_comparison"] == comparison, "V11 alias witness comparison differs")
    gate_passes = {
        "v10_ledger_and_169_model_closure": _validate_reconstruction(load_json(RECONSTRUCTION_PATH)),
        "non_evaluator_quality_8_of_8": _validate_quality(load_json(QUALITY_PATH)),
        "dynamic_312_event_closure": _validate_dynamic(load_json(DYNAMIC_PATH)),
        "two_build_alias_codec_and_manifest_closure": comparison["all_match"] and witness["embedding_preserved_in_both"] and witness["lm_head_separate_in_both"] and witness["palette_scale18_codec_passed"],
    }
    require(contract["pre_attempt_gates"]["all_pass"] == all(gate_passes.values()), "V11 contract aggregate gate status differs")
    for name, passed in gate_passes.items():
        require(contract["pre_attempt_gates"][name]["passed"] == passed, f"V11 contract gate differs: {name}")
    terminal_path = READY_PATH if READY_PATH.is_file() else NO_GO_PATH
    terminal = load_json(terminal_path)
    require(terminal["candidate_id"] == CANDIDATE_ID, "V11 terminal candidate differs")
    require(terminal["contract"]["sha256"] == reference.sha256_file(CONTRACT_PATH), "V11 terminal contract file hash differs")
    require(terminal["contract_sha256"] == wrapper["contract_sha256"], "V11 terminal contract canonical hash differs")
    require(terminal["bound_artifacts"] == contract["artifacts"], "V11 terminal artifact snapshot differs")
    require((terminal_path == READY_PATH) == all(gate_passes.values()), "V11 terminal disposition differs from gates")
    if require_unconsumed or terminal_path == NO_GO_PATH:
        require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "V11 official attempt exists before authorization consumption")
    quality = load_json(QUALITY_PATH)
    dynamic = load_json(DYNAMIC_PATH)
    reconstruction = load_json(RECONSTRUCTION_PATH)
    return {
        "status": terminal["status"],
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": wrapper["contract_sha256"],
        "gate_passes": gate_passes,
        "per_tensor_non_regressing_count": reconstruction["per_tensor_non_regressing_count"],
        "aggregate_sealed_v10_bf16_sse": reconstruction["aggregate_sealed_v10_bf16_sse"],
        "aggregate_v11_bf16_sse": reconstruction["aggregate_v11_bf16_sse"],
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
        "backend": "exact BF16 four-codebook/quaternary-selector/palette-Scale18 construction",
    }
    recovery = _recover_nonqualifying_preparation()
    namespace_audit = _audit_attempt_namespace(expect_root_absent=recovery["fresh_namespace"])
    namespace_audit["preparation_recovery"] = recovery
    reference.verify_companion(TASK_PATH, TASK_COMPANION)
    reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    verify_source_contract()
    source_identity = verify_source_snapshot()
    require(reference.sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(reference.sha256_file(V10_RECONSTRUCTION_PATH) == V10_RECONSTRUCTION_SHA256, "sealed V10 reconstruction hash differs")
    require(reference.sha256_file(V10_REVIEW_PATH) == V10_REVIEW_SHA256, "sealed V10 review hash differs")
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
    require(probe.read_bytes() == b"preflight\n", "V11 output-path readback differs")
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

    candidate_a_manifest_tmp = OUTPUT_DIR / ".candidate_a_manifest.json"
    candidate_a_witness_tmp = OUTPUT_DIR / ".candidate_a_witness.json"
    candidate_a_reconstruction_tmp = OUTPUT_DIR / ".candidate_a_reconstruction.json"
    chronology["candidate_a_started_at_utc"] = utc_now()
    print("V11_CANDIDATE_A_START parallel_worker", flush=True)
    context = multiprocessing.get_context("spawn")
    candidate_a_process = context.Process(
        target=_construct_candidate_artifacts,
        args=(str(candidate_a_manifest_tmp), str(candidate_a_witness_tmp), str(candidate_a_reconstruction_tmp)),
        name="v11-candidate-a",
    )
    candidate_a_process.start()
    chronology["candidate_b_started_at_utc"] = utc_now()
    print("V11_CANDIDATE_B_START parallel_parent", flush=True)
    try:
        candidate_b, manifest_b, witness_b, reconstruction_b = _load_candidate("candidate_b_independent_quality_and_dynamic_preflight")
    except BaseException:
        if candidate_a_process.is_alive():
            candidate_a_process.terminate()
        candidate_a_process.join()
        raise
    candidate_a_process.join()
    require(candidate_a_process.exitcode == 0, f"V11 candidate A worker failed: exitcode={candidate_a_process.exitcode}")
    manifest_a = load_json(candidate_a_manifest_tmp)
    witness_a = load_json(candidate_a_witness_tmp)
    reconstruction_a = load_json(candidate_a_reconstruction_tmp)
    candidate_a_manifest_tmp.unlink()
    candidate_a_witness_tmp.unlink()
    candidate_a_reconstruction_tmp.unlink()
    chronology["candidate_a_complete_at_utc"] = utc_now()
    print("V11_CANDIDATE_A_DONE", flush=True)
    state_comparison = _compare_candidates(manifest_a, manifest_b)
    require(reconstruction_a == reconstruction_b, "V11 candidate A/B reconstruction reports differ")
    chronology["candidate_b_constructed_at_utc"] = utc_now()
    print("V11_CANDIDATE_B_CONSTRUCTED quality_dynamic_start", flush=True)
    torch.set_num_threads(TORCH_THREADS)
    quality, dynamic = v7.v3.non_evaluator_quality_and_dynamic(candidate_b, tokenizer, inputs, input_manifest)
    chronology["candidate_b_quality_and_dynamic_complete_at_utc"] = utc_now()
    print("V11_QUALITY_DYNAMIC_DONE", flush=True)
    del candidate_b
    gc.collect()

    gate_passes = {
        "v10_ledger_and_169_model_closure": _validate_reconstruction(reconstruction_b),
        "non_evaluator_quality_8_of_8": _validate_quality(quality),
        "dynamic_312_event_closure": _validate_dynamic(dynamic),
        "two_build_alias_codec_and_manifest_closure": (
            state_comparison["all_match"]
            and witness_a["post_v11"]["embedding_hash_matches_source"]
            and witness_b["post_v11"]["embedding_hash_matches_source"]
            and witness_a["post_v11"]["lm_head_storage_is_separate"]
            and witness_b["post_v11"]["lm_head_storage_is_separate"]
            and self_test["checks"]["scale18_lsb_first_cross_byte_round_trip"]
            and self_test["checks"]["selector_row_major_block_major_lsb_first"]
        ),
    }
    require(not MARKER_PATH.exists() and not ATTEMPT_DIR.exists(), "V11 official attempt appeared before artifact freeze")

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
            "embedding_preserved_in_both": witness_a["post_v11"]["embedding_hash_matches_source"] and witness_b["post_v11"]["embedding_hash_matches_source"],
            "lm_head_separate_in_both": witness_a["post_v11"]["lm_head_storage_is_separate"] and witness_b["post_v11"]["lm_head_storage_is_separate"],
            "palette_scale18_codec_passed": self_test["checks"]["scale18_lsb_first_cross_byte_round_trip"],
            "quaternary_selector_codec_passed": self_test["checks"]["selector_row_major_block_major_lsb_first"],
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
        "actual_nonnegative_bf16_sse": {"sealed_v10": reconstruction_b["aggregate_sealed_v10_bf16_sse"], "v11": reconstruction_b["aggregate_v11_bf16_sse"]},
        "attempt_budget": 1,
        "attempts_consumed": 0,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": True,
        "disposition": (
            "all marker-free V11 gates pass; PREATTEMPT_READY is sealed and the single official attempt may now be consumed"
            if all_pass
            else "preserve V11 no-go evidence, leave attempt-0001 absent and unconsumed, keep selected_policy_id null, and submit for Fresh Review"
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
    require(closure["status"] == "PREATTEMPT_READY", "V11 official execution requires verified PREATTEMPT_READY")
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
        require(comparison["all_match"], "official V11 candidate differs from preflight")
        policy = v7.v3.AuditedDynamicPolicy(load_json(BASE_SCALES_PATH))
        policy.install(candidate_model)
        prompt_results: list[dict[str, Any]] = []
        model_forwards = 0
        for prompt in PROMPTS:
            bf16 = references[prompt.case_id]
            candidate, forwards = v7.grouped.generate_candidate(candidate_model, tokenizer, inputs[prompt.case_id], prompt.expected, bf16)
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
            "classification": "qwen_instruct_stage1_option_b_alias_safe_four_codebook_palette_scale18_single_candidate",
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
                "root_cause_hypothesis": "the V11 four-bit reconstruction changed one or more greedy logits relative to BF16",
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
                "palette-Scale18, four-codebook, and quaternary-selector accounting",
                "deterministic 169-tensor V10-ledger construction",
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
            "regression": "repair only the isolated marker-free V11 runner and rerun pre-attempt preparation; official authorization remains unconsumed",
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
