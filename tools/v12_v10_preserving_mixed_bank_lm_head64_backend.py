#!/usr/bin/env python3
"""Exact V12 V10-preserving mixed-bank constructor and evidence flow."""

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
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_option_b_alias_safe_v10_preserving_mixed_four_bank_lm_head64_grouped_dual_codebook_packed_scale24_w4_grouped_dynamic_scale32_stage1 as reference
import v10_packed_scale24_per_row_selector_backend as v10
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
V10_SEED_PATH = PREATTEMPT_DIR / "v10_compatible_seed_reproduction.json"
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
V10_CANDIDATE_MANIFEST_PATH = ROOT / "build/stage1-option-b-alias-safe-packed-scale24-per-row-selector-dual-input-block-codebook-w4-grouped-dynamic-scale32-w4a8-v10/preattempt/candidate_b_manifest.json"
V10_CANDIDATE_MANIFEST_SHA256 = "59bb66df93445ab03a13281775d36d041adbd9f2069f052558f4fe8582f2df10"
V10_REVIEW_PATH = ROOT / "research/raw/specification/packed-scale24-per-row-selector-v10-review-done-20260807T011340Z.json"
V10_REVIEW_SHA256 = "013018e47424a30198917b8b8ac6dcc8d19617b7c5e2286353c1cc58797e5d04"
V11_REVIEW_PATH = ROOT / "research/raw/specification/four-codebook-palette-scale18-v11-review-done-20260807T034150Z.json"
V11_REVIEW_SHA256 = "14ed1885ddb736a3b4e046bef9de2a084b51eb5afc41f6acfb9849e2730d4cab"
V10_AGGREGATE_BF16_SSE = 2153.4047236346078

v7 = v10.v7
v4 = v7.v4

INPUT_BLOCK_LANES = reference.INPUT_BLOCK_LANES
CODEBOOK_ENTRIES = reference.CODEBOOK_ENTRIES
FOUR_BANK_COUNT = reference.FOUR_BANK_COUNT
DUAL_BANK_COUNT = reference.DUAL_BANK_COUNT
FOUR_BANK_SELECTOR_BITS = reference.FOUR_BANK_SELECTOR_BITS
DUAL_BANK_SELECTOR_BITS = reference.DUAL_BANK_SELECTOR_BITS
LM_HEAD_ROWS = reference.LM_HEAD_ROWS
LM_HEAD_WIDTH = reference.LM_HEAD_WIDTH
LM_HEAD_GROUP_COUNT = reference.LM_HEAD_GROUP_COUNT
LM_HEAD_ROWS_PER_GROUP = reference.LM_HEAD_ROWS_PER_GROUP
OUTER_ITERATIONS = reference.OUTER_ITERATIONS
ALIGNMENT_BYTES = reference.ALIGNMENT_BYTES
TORCH_THREADS = 32
CONSTRUCTION_TORCH_THREADS = 1
EXPECTED_EVENTS = v7.grouped.LAYERS * len(v7.grouped.EVENT_FAMILIES)
STATE_MATCH_KEYS = (
    "embedding_bf16_sha256",
    "payload_manifest_sha256",
    "packed_scale24_manifest_sha256",
    "mixed_codebook_manifest_sha256",
    "selector_manifest_sha256",
    "v10_seed_manifest_sha256",
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
    return (value + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES * ALIGNMENT_BYTES


def _block_ranges(width: int) -> tuple[tuple[int, int], ...]:
    return reference.input_block_ranges(width)


def _group_ranges(rows: int, group_size: int) -> tuple[tuple[int, int], ...]:
    require(rows > 0 and group_size > 0 and rows % group_size == 0, "V12 row-group geometry differs")
    return tuple((start, start + group_size) for start in range(0, rows, group_size))


def lm_head_group_ranges(rows: int = LM_HEAD_ROWS) -> tuple[tuple[int, int], ...]:
    groups = _group_ranges(rows, LM_HEAD_ROWS_PER_GROUP)
    require(len(groups) == LM_HEAD_GROUP_COUNT, "V12 lm_head group count differs")
    require(groups[0] == (0, LM_HEAD_ROWS_PER_GROUP), "V12 first lm_head group differs")
    require(groups[-1] == (LM_HEAD_ROWS - LM_HEAD_ROWS_PER_GROUP, LM_HEAD_ROWS), "V12 final lm_head group differs")
    return groups


def _validate_bank(bank: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in bank)
    require(len(values) == CODEBOOK_ENTRIES, "V12 codebook entry count differs")
    require(all(-128 <= value <= 127 for value in values), "V12 codebook entry escaped signed Q4.4")
    require(all(left < right for left, right in zip(values, values[1:])), "V12 codebook is not strictly increasing")
    return values


def _validate_standard_codebooks(
    codebooks: Sequence[Sequence[Sequence[int]]], block_count: int, bank_count: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    require(len(codebooks) == block_count, "V12 standard codebook block count differs")
    result = []
    for banks in codebooks:
        require(len(banks) == bank_count, "V12 standard bank count differs")
        result.append(tuple(_validate_bank(bank) for bank in banks))
    return tuple(result)


def _validate_grouped_codebooks(
    codebooks: Sequence[Sequence[Sequence[Sequence[int]]]], group_count: int, block_count: int
) -> tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]:
    require(len(codebooks) == group_count, "V12 grouped codebook group count differs")
    return tuple(_validate_standard_codebooks(group, block_count, DUAL_BANK_COUNT) for group in codebooks)


def _codebook_raw_standard(codebooks: Sequence[Sequence[Sequence[int]]]) -> bytes:
    return bytes(int(value) & 0xFF for banks in codebooks for bank in banks for value in bank)


def _codebook_raw_grouped(codebooks: Sequence[Sequence[Sequence[Sequence[int]]]]) -> bytes:
    return bytes(
        int(value) & 0xFF
        for group in codebooks
        for banks in group
        for bank in banks
        for value in bank
    )


def _pack_selectors(selectors: Tensor, bits_per_selector: int) -> bytes:
    require(selectors.ndim == 2 and selectors.dtype == torch.uint8, "V12 selector source format differs")
    require(bits_per_selector in (1, 2), "V12 selector width is illegal")
    limit = 1 << bits_per_selector
    require(bool(torch.all(selectors < limit)), "V12 selector value exceeds its width")
    flat = selectors.reshape(-1).tolist()
    raw = bytearray(_align((len(flat) * bits_per_selector + 7) // 8))
    for ordinal, value in enumerate(flat):
        bit_offset = ordinal * bits_per_selector
        shift = bit_offset & 7
        require(shift + bits_per_selector <= 8, "V12 selector unexpectedly crosses a byte")
        raw[bit_offset >> 3] |= int(value) << shift
    return bytes(raw)


def _unpack_selectors(raw: bytes, rows: int, block_count: int, bits_per_selector: int) -> Tensor:
    require(rows >= 0 and block_count > 0, "V12 selector geometry is illegal")
    expected = _align((rows * block_count * bits_per_selector + 7) // 8)
    require(len(raw) == expected, "V12 selector stream length differs")
    mask = (1 << bits_per_selector) - 1
    values = []
    for ordinal in range(rows * block_count):
        bit_offset = ordinal * bits_per_selector
        values.append((raw[bit_offset >> 3] >> (bit_offset & 7)) & mask)
    result = torch.tensor(values, dtype=torch.uint8).reshape(rows, block_count)
    require(_pack_selectors(result, bits_per_selector) == raw, "V12 selector padding or ordering differs")
    return result


def _scale_values(records: Tensor) -> Tensor:
    significands, exponents = v4._records_components(records)
    return torch.ldexp(significands.to(torch.float64), exponents.to(torch.int32) - 15)


def _selected_points_standard(
    assignments: Tensor, codebooks: Sequence[Sequence[Sequence[int]]], selectors: Tensor
) -> Tensor:
    rows, width = (int(value) for value in assignments.shape)
    points = torch.empty((rows, width), dtype=torch.int64)
    for block_index, (start, stop) in enumerate(_block_ranges(width)):
        for bank in range(len(codebooks[block_index])):
            row_indices = torch.nonzero(selectors[:, block_index] == bank, as_tuple=False).reshape(-1)
            if row_indices.numel() == 0:
                continue
            bank_points = torch.tensor(codebooks[block_index][bank], dtype=torch.int64)
            points[row_indices, start:stop] = bank_points[assignments[row_indices, start:stop].to(torch.int64)]
    return points


def _selected_points_grouped(
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[Sequence[int]]]],
    selectors: Tensor,
    group_size: int,
) -> Tensor:
    rows, width = (int(value) for value in assignments.shape)
    points = torch.empty((rows, width), dtype=torch.int64)
    for group_index, (row_start, row_stop) in enumerate(_group_ranges(rows, group_size)):
        for block_index, (start, stop) in enumerate(_block_ranges(width)):
            for bank in range(DUAL_BANK_COUNT):
                local = torch.nonzero(selectors[row_start:row_stop, block_index] == bank, as_tuple=False).reshape(-1)
                if local.numel() == 0:
                    continue
                row_indices = local + row_start
                bank_points = torch.tensor(codebooks[group_index][block_index][bank], dtype=torch.int64)
                points[row_indices, start:stop] = bank_points[assignments[row_indices, start:stop].to(torch.int64)]
    return points


def _decode_standard(
    records: Tensor, assignments: Tensor, codebooks: Sequence[Sequence[Sequence[int]]], selectors: Tensor
) -> Tensor:
    scales = _scale_values(records)
    points = _selected_points_standard(assignments, codebooks, selectors)
    return (points.to(torch.float64) * (scales[:, None] / 16.0)).to(torch.bfloat16)


def _decode_grouped(
    records: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[Sequence[int]]]],
    selectors: Tensor,
    group_size: int,
) -> Tensor:
    scales = _scale_values(records)
    points = _selected_points_grouped(assignments, codebooks, selectors, group_size)
    return (points.to(torch.float64) * (scales[:, None] / 16.0)).to(torch.bfloat16)


def _literal_bf16_sse(source: Tensor, reconstruction: Tensor) -> float:
    difference = source.to(torch.float64) - reconstruction.to(torch.float64)
    return float(torch.sum(difference * difference))


def _assign_payload_standard(
    source: Tensor,
    records: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor,
) -> Tensor:
    rows, width = (int(value) for value in source.shape)
    assignments = torch.empty((rows, width), dtype=torch.uint8)
    scales = _scale_values(records)
    for block_index, (start, stop) in enumerate(_block_ranges(width)):
        for bank in range(len(codebooks[block_index])):
            row_indices = torch.nonzero(selectors[:, block_index] == bank, as_tuple=False).reshape(-1)
            if row_indices.numel() == 0:
                continue
            points = codebooks[block_index][bank]
            midpoint_sums = torch.tensor(
                [left + right for left, right in zip(points, points[1:])], dtype=torch.float64
            )
            boundaries = scales[row_indices, None] * (midpoint_sums[None, :] / 32.0)
            selected = torch.searchsorted(
                boundaries.contiguous(), source[row_indices, start:stop].to(torch.float64).contiguous(), right=False
            )
            require(bool(torch.all(selected < CODEBOOK_ENTRIES)), "V12 payload assignment escaped four bits")
            assignments[row_indices, start:stop] = selected.to(torch.uint8)
    return assignments


def _assign_payload_grouped(
    source: Tensor,
    records: Tensor,
    codebooks: Sequence[Sequence[Sequence[Sequence[int]]]],
    selectors: Tensor,
    group_size: int,
) -> Tensor:
    rows, width = (int(value) for value in source.shape)
    assignments = torch.empty((rows, width), dtype=torch.uint8)
    scales = _scale_values(records)
    for group_index, (row_start, row_stop) in enumerate(_group_ranges(rows, group_size)):
        for block_index, (start, stop) in enumerate(_block_ranges(width)):
            for bank in range(DUAL_BANK_COUNT):
                local = torch.nonzero(selectors[row_start:row_stop, block_index] == bank, as_tuple=False).reshape(-1)
                if local.numel() == 0:
                    continue
                row_indices = local + row_start
                points = codebooks[group_index][block_index][bank]
                midpoint_sums = torch.tensor(
                    [left + right for left, right in zip(points, points[1:])], dtype=torch.float64
                )
                boundaries = scales[row_indices, None] * (midpoint_sums[None, :] / 32.0)
                selected = torch.searchsorted(
                    boundaries.contiguous(), source[row_indices, start:stop].to(torch.float64).contiguous(), right=False
                )
                require(bool(torch.all(selected < CODEBOOK_ENTRIES)), "V12 grouped payload escaped four bits")
                assignments[row_indices, start:stop] = selected.to(torch.uint8)
    return assignments


def _row_bank_costs(
    source_block: Tensor,
    records: Tensor,
    assignment_block: Tensor,
    banks: Sequence[Sequence[int]],
    shift_min: int,
    objective_exponent: int,
) -> list[list[int]]:
    significands, shifts = v4._decode_bf16(source_block)
    delta = torch.where(significands == 0, torch.zeros_like(shifts), shifts - shift_min)
    require(bool(torch.all((delta >= 0) & (delta <= 54))), "V12 BF16 lattice span differs")
    exact = torch.bitwise_left_shift(significands, delta)
    indices = assignment_block.to(torch.int64)
    costs: list[list[int]] = []
    for bank in banks:
        points = torch.tensor(bank, dtype=torch.int64)[indices]
        weight_codepoint = torch.sum(exact * points, dim=1).tolist()
        codepoint_squared = torch.sum(points * points, dim=1).tolist()
        costs.append(
            [
                v4._row_objective(int(record), int(wp), int(p2), shift_min, objective_exponent)
                for record, wp, p2 in zip(records.tolist(), weight_codepoint, codepoint_squared, strict=True)
            ]
        )
    return costs


def _assign_selectors_standard(
    source: Tensor,
    records: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    shift_min: int,
    objective_exponent: int,
) -> tuple[Tensor, dict[str, Any]]:
    rows, width = (int(value) for value in source.shape)
    selectors = torch.empty((rows, len(_block_ranges(width))), dtype=torch.uint8)
    counts = [0] * len(codebooks[0])
    ties = 0
    for block_index, (start, stop) in enumerate(_block_ranges(width)):
        costs = _row_bank_costs(
            source[:, start:stop], records, assignments[:, start:stop], codebooks[block_index], shift_min, objective_exponent
        )
        for row in range(rows):
            choice = min(range(len(costs)), key=lambda bank: (costs[bank][row], bank))
            ties += int(sum(bank_cost[row] == costs[choice][row] for bank_cost in costs) > 1)
            selectors[row, block_index] = choice
            counts[choice] += 1
    return selectors, {"bank_counts": counts, "exact_tie_to_lower_bank_count": ties}


def _assign_selectors_grouped(
    source: Tensor,
    records: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[Sequence[int]]]],
    group_size: int,
    shift_min: int,
    objective_exponent: int,
) -> tuple[Tensor, dict[str, Any]]:
    rows, width = (int(value) for value in source.shape)
    selectors = torch.empty((rows, len(_block_ranges(width))), dtype=torch.uint8)
    counts = [0, 0]
    ties = 0
    for group_index, (row_start, row_stop) in enumerate(_group_ranges(rows, group_size)):
        for block_index, (start, stop) in enumerate(_block_ranges(width)):
            costs = _row_bank_costs(
                source[row_start:row_stop, start:stop],
                records[row_start:row_stop],
                assignments[row_start:row_stop, start:stop],
                codebooks[group_index][block_index],
                shift_min,
                objective_exponent,
            )
            for local_row in range(row_stop - row_start):
                choice = min((0, 1), key=lambda bank: (costs[bank][local_row], bank))
                ties += int(costs[0][local_row] == costs[1][local_row])
                selectors[row_start + local_row, block_index] = choice
                counts[choice] += 1
    return selectors, {"bank_counts": counts, "exact_tie_to_lower_bank_count": ties}


def _update_standard_codebooks(
    source: Tensor,
    records: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[int]]],
    selectors: Tensor,
    shift_min: int,
    shift_max: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    updated = []
    for block_index, (start, stop) in enumerate(_block_ranges(int(source.shape[1]))):
        banks = []
        for bank_index, prior in enumerate(codebooks[block_index]):
            row_indices = torch.nonzero(selectors[:, block_index] == bank_index, as_tuple=False).reshape(-1)
            if row_indices.numel() == 0:
                banks.append(tuple(prior))
                continue
            aggregates = v4._exact_aggregates(
                source[row_indices, start:stop], records[row_indices], assignments[row_indices, start:stop], shift_min, shift_max
            )
            selected, _summary = v4._update_codebook(aggregates, prior)
            banks.append(selected)
        updated.append(tuple(banks))
    return _validate_standard_codebooks(updated, len(updated), len(updated[0]))


def _update_grouped_codebooks(
    source: Tensor,
    records: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[Sequence[Sequence[int]]]],
    selectors: Tensor,
    group_size: int,
    shift_min: int,
    shift_max: int,
) -> tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]:
    rows, width = (int(value) for value in source.shape)
    updated_groups = []
    for group_index, (row_start, row_stop) in enumerate(_group_ranges(rows, group_size)):
        updated_blocks = []
        for block_index, (start, stop) in enumerate(_block_ranges(width)):
            updated_banks = []
            for bank in range(DUAL_BANK_COUNT):
                local = torch.nonzero(selectors[row_start:row_stop, block_index] == bank, as_tuple=False).reshape(-1)
                prior = codebooks[group_index][block_index][bank]
                if local.numel() == 0:
                    updated_banks.append(tuple(prior))
                    continue
                row_indices = local + row_start
                aggregates = v4._exact_aggregates(
                    source[row_indices, start:stop],
                    records[row_indices],
                    assignments[row_indices, start:stop],
                    shift_min,
                    shift_max,
                )
                selected, _summary = v4._update_codebook(aggregates, prior)
                updated_banks.append(selected)
            updated_blocks.append(tuple(updated_banks))
        updated_groups.append(tuple(updated_blocks))
    return _validate_grouped_codebooks(updated_groups, len(updated_groups), len(updated_groups[0]))


def _seed_summary(seed: dict[str, Any]) -> dict[str, Any]:
    reconstruction = v10._decode(seed)
    return {
        "payload_sha256": hashlib.sha256(seed["payload"]).hexdigest(),
        "packed_scale24_stream_sha256": hashlib.sha256(seed["packed_scale24"]).hexdigest(),
        "dual_codebook_stream_sha256": hashlib.sha256(seed["codebook_raw"]).hexdigest(),
        "selector_stream_sha256": hashlib.sha256(seed["selector_raw"]).hexdigest(),
        "reconstruction_bf16_sha256": v7.v3.tensor_sha256(reconstruction),
        "literal_bf16_sse": None,
    }


def _state_hashes(state: dict[str, Any]) -> dict[str, str]:
    reconstruction = _decode_state(state)
    return {
        "payload_sha256": hashlib.sha256(state["payload"]).hexdigest(),
        "packed_scale24_stream_sha256": hashlib.sha256(state["packed_scale24"]).hexdigest(),
        "mixed_codebook_stream_sha256": hashlib.sha256(state["codebook_raw"]).hexdigest(),
        "selector_stream_sha256": hashlib.sha256(state["selector_raw"]).hexdigest(),
        "reconstruction_bf16_sha256": v7.v3.tensor_sha256(reconstruction),
        "iteration_trace_sha256": canonical_sha256(state["iteration_trace"]),
    }


def _construct_four_bank_from_seed(source: Tensor, module_name: str, seed: dict[str, Any]) -> dict[str, Any]:
    rows, width = (int(value) for value in source.shape)
    blocks = _block_ranges(width)
    records = seed["final_records"].clone()
    assignments = seed["final_assignments"].clone()
    codebooks = tuple(
        (tuple(banks[0]), tuple(banks[1]), tuple(banks[0]), tuple(banks[1])) for banks in seed["final_codebooks"]
    )
    selectors = seed["final_selectors"].clone()
    require(selectors.shape == (rows, len(blocks)), "V12 four-bank selector seed geometry differs")
    initial = _decode_standard(records, assignments, codebooks, selectors)
    seed_reconstruction = v10._decode(seed)
    require(torch.equal(initial, seed_reconstruction), f"V12 four-bank seed does not decode as V10: {module_name}")
    previous_sse = _literal_bf16_sse(source, initial)
    shift_min = int(seed["fresh_v9_state"]["fresh_v7_state"]["shift_min"])
    shift_max = int(seed["fresh_v9_state"]["fresh_v7_state"]["shift_max"])
    objective_exponent = int(seed["final_objective"]["binary_exponent"])
    trace = []
    for iteration in range(1, OUTER_ITERATIONS + 1):
        prior = (assignments, codebooks, selectors)
        candidate_selectors, selector_summary = _assign_selectors_standard(
            source, records, assignments, codebooks, shift_min, objective_exponent
        )
        candidate_assignments = _assign_payload_standard(source, records, codebooks, candidate_selectors)
        candidate_codebooks = _update_standard_codebooks(
            source, records, candidate_assignments, codebooks, candidate_selectors, shift_min, shift_max
        )
        candidate_assignments = _assign_payload_standard(source, records, candidate_codebooks, candidate_selectors)
        candidate_reconstruction = _decode_standard(records, candidate_assignments, candidate_codebooks, candidate_selectors)
        candidate_sse = _literal_bf16_sse(source, candidate_reconstruction)
        accepted = candidate_sse <= previous_sse
        if accepted:
            assignments, codebooks, selectors = candidate_assignments, candidate_codebooks, candidate_selectors
            previous_sse = candidate_sse
        else:
            assignments, codebooks, selectors = prior
        trace.append(
            {
                "iteration": iteration,
                "candidate_literal_bf16_sse": candidate_sse,
                "accepted_literal_bf16_non_regression": accepted,
                "final_literal_bf16_sse": previous_sse,
                "selector_assignment": selector_summary,
                "payload_sha256": hashlib.sha256(v7.v3.pack_codebook_indices(assignments)).hexdigest(),
                "codebook_sha256": hashlib.sha256(_codebook_raw_standard(codebooks)).hexdigest(),
                "selector_sha256": hashlib.sha256(_pack_selectors(selectors, FOUR_BANK_SELECTOR_BITS)).hexdigest(),
            }
        )
    return {
        "module_name": module_name,
        "mode": "standard_four_bank",
        "shape": [rows, width],
        "input_block_count": len(blocks),
        "bank_count": FOUR_BANK_COUNT,
        "selector_bits": FOUR_BANK_SELECTOR_BITS,
        "final_records": records,
        "final_assignments": assignments,
        "final_codebooks": codebooks,
        "final_selectors": selectors,
        "payload": v7.v3.pack_codebook_indices(assignments),
        "packed_scale24": seed["packed_scale24"],
        "codebook_raw": _codebook_raw_standard(codebooks),
        "selector_raw": _pack_selectors(selectors, FOUR_BANK_SELECTOR_BITS),
        "v10_seed": _seed_summary(seed),
        "initial_v10_literal_bf16_sse": _literal_bf16_sse(source, seed_reconstruction),
        "final_v12_literal_bf16_sse": previous_sse,
        "outer_iterations_completed": len(trace),
        "iteration_trace": trace,
    }


def _construct_grouped_lm_head_from_seed(
    source: Tensor, module_name: str, seed: dict[str, Any], group_size: int
) -> dict[str, Any]:
    rows, width = (int(value) for value in source.shape)
    blocks = _block_ranges(width)
    groups = _group_ranges(rows, group_size)
    records = seed["final_records"].clone()
    assignments = seed["final_assignments"].clone()
    codebooks = tuple(
        tuple((tuple(banks[0]), tuple(banks[1])) for banks in seed["final_codebooks"]) for _group in groups
    )
    selectors = seed["final_selectors"].clone()
    initial = _decode_grouped(records, assignments, codebooks, selectors, group_size)
    seed_reconstruction = v10._decode(seed)
    require(torch.equal(initial, seed_reconstruction), f"V12 grouped seed does not decode as V10: {module_name}")
    previous_sse = _literal_bf16_sse(source, initial)
    shift_min = int(seed["fresh_v9_state"]["fresh_v7_state"]["shift_min"])
    shift_max = int(seed["fresh_v9_state"]["fresh_v7_state"]["shift_max"])
    objective_exponent = int(seed["final_objective"]["binary_exponent"])
    trace = []
    for iteration in range(1, OUTER_ITERATIONS + 1):
        prior = (assignments, codebooks, selectors)
        candidate_selectors, selector_summary = _assign_selectors_grouped(
            source, records, assignments, codebooks, group_size, shift_min, objective_exponent
        )
        candidate_assignments = _assign_payload_grouped(source, records, codebooks, candidate_selectors, group_size)
        candidate_codebooks = _update_grouped_codebooks(
            source,
            records,
            candidate_assignments,
            codebooks,
            candidate_selectors,
            group_size,
            shift_min,
            shift_max,
        )
        candidate_assignments = _assign_payload_grouped(
            source, records, candidate_codebooks, candidate_selectors, group_size
        )
        candidate_reconstruction = _decode_grouped(
            records, candidate_assignments, candidate_codebooks, candidate_selectors, group_size
        )
        candidate_sse = _literal_bf16_sse(source, candidate_reconstruction)
        accepted = candidate_sse <= previous_sse
        if accepted:
            assignments, codebooks, selectors = candidate_assignments, candidate_codebooks, candidate_selectors
            previous_sse = candidate_sse
        else:
            assignments, codebooks, selectors = prior
        trace.append(
            {
                "iteration": iteration,
                "candidate_literal_bf16_sse": candidate_sse,
                "accepted_literal_bf16_non_regression": accepted,
                "final_literal_bf16_sse": previous_sse,
                "selector_assignment": selector_summary,
                "payload_sha256": hashlib.sha256(v7.v3.pack_codebook_indices(assignments)).hexdigest(),
                "codebook_sha256": hashlib.sha256(_codebook_raw_grouped(codebooks)).hexdigest(),
                "selector_sha256": hashlib.sha256(_pack_selectors(selectors, DUAL_BANK_SELECTOR_BITS)).hexdigest(),
            }
        )
    return {
        "module_name": module_name,
        "mode": "lm_head_grouped_dual_bank",
        "shape": [rows, width],
        "input_block_count": len(blocks),
        "group_size": group_size,
        "group_count": len(groups),
        "bank_count_per_group_block": DUAL_BANK_COUNT,
        "selector_bits": DUAL_BANK_SELECTOR_BITS,
        "group_selector_metadata_bits": 0,
        "final_records": records,
        "final_assignments": assignments,
        "final_codebooks": codebooks,
        "final_selectors": selectors,
        "payload": v7.v3.pack_codebook_indices(assignments),
        "packed_scale24": seed["packed_scale24"],
        "codebook_raw": _codebook_raw_grouped(codebooks),
        "selector_raw": _pack_selectors(selectors, DUAL_BANK_SELECTOR_BITS),
        "v10_seed": _seed_summary(seed),
        "initial_v10_literal_bf16_sse": _literal_bf16_sse(source, seed_reconstruction),
        "final_v12_literal_bf16_sse": previous_sse,
        "outer_iterations_completed": len(trace),
        "iteration_trace": trace,
    }


def _construct_preserved_down_proj(source: Tensor, module_name: str, seed: dict[str, Any]) -> dict[str, Any]:
    rows, width = (int(value) for value in source.shape)
    reconstruction = v10._decode(seed)
    return {
        "module_name": module_name,
        "mode": "down_proj_v10_dual_bank_preserved",
        "shape": [rows, width],
        "input_block_count": len(_block_ranges(width)),
        "bank_count": DUAL_BANK_COUNT,
        "selector_bits": DUAL_BANK_SELECTOR_BITS,
        "final_records": seed["final_records"].clone(),
        "final_assignments": seed["final_assignments"].clone(),
        "final_codebooks": seed["final_codebooks"],
        "final_selectors": seed["final_selectors"].clone(),
        "payload": seed["payload"],
        "packed_scale24": seed["packed_scale24"],
        "codebook_raw": seed["codebook_raw"],
        "selector_raw": seed["selector_raw"],
        "v10_seed": _seed_summary(seed),
        "initial_v10_literal_bf16_sse": _literal_bf16_sse(source, reconstruction),
        "final_v12_literal_bf16_sse": _literal_bf16_sse(source, reconstruction),
        "outer_iterations_completed": 0,
        "iteration_trace": [],
    }


def construct_v12_state(source: Tensor, module_name: str) -> dict[str, Any]:
    require(source.ndim == 2 and source.dtype == torch.bfloat16, f"source weight format differs: {module_name}")
    family = reference.module_family(module_name)
    seed = v10.construct_packed_scale24_state(source, module_name)
    if family in reference.FOUR_BANK_FAMILIES:
        return _construct_four_bank_from_seed(source, module_name, seed)
    if family == "down_proj":
        return _construct_preserved_down_proj(source, module_name, seed)
    require(list(source.shape) == [LM_HEAD_ROWS, LM_HEAD_WIDTH], "V12 lm_head shape differs")
    return _construct_grouped_lm_head_from_seed(source, module_name, seed, LM_HEAD_ROWS_PER_GROUP)


def _decode_state(state: dict[str, Any]) -> Tensor:
    mode = state["mode"]
    if mode == "standard_four_bank":
        return _decode_standard(
            state["final_records"], state["final_assignments"], state["final_codebooks"], state["final_selectors"]
        )
    if mode == "lm_head_grouped_dual_bank":
        return _decode_grouped(
            state["final_records"],
            state["final_assignments"],
            state["final_codebooks"],
            state["final_selectors"],
            int(state["group_size"]),
        )
    require(mode == "down_proj_v10_dual_bank_preserved", "V12 state mode differs")
    return v10.v9._decode(
        state["final_records"], state["final_assignments"], state["final_codebooks"], state["final_selectors"], 1
    )


def _storage_closure() -> dict[str, Any]:
    standard = {
        "payload": 7454720,
        "packed_scale24": 38016,
        "mixed_codebook": 3904,
        "selector": 24864,
    }
    standard["total"] = sum(standard.values())
    standard["weight_count"] = 14909440
    standard["bits_per_weight"] = standard["total"] * 8 / standard["weight_count"]
    lm_head = {
        "payload": LM_HEAD_ROWS * LM_HEAD_WIDTH // 2,
        "packed_scale24": LM_HEAD_ROWS * 3,
        "grouped_dual_codebook": LM_HEAD_GROUP_COUNT * 7 * DUAL_BANK_COUNT * CODEBOOK_ENTRIES,
        "selector": _align(LM_HEAD_ROWS * 7 // 8),
    }
    lm_head["total"] = sum(lm_head.values())
    lm_head["weight_count"] = LM_HEAD_ROWS * LM_HEAD_WIDTH
    lm_head["bits_per_weight"] = lm_head["total"] * 8 / lm_head["weight_count"]
    whole = {
        "payload": standard["payload"] * 24 + lm_head["payload"],
        "packed_scale24": standard["packed_scale24"] * 24 + lm_head["packed_scale24"],
        "mixed_codebook": standard["mixed_codebook"] * 24 + lm_head["grouped_dual_codebook"],
        "selector": standard["selector"] * 24 + lm_head["selector"],
        "weight_count": standard["weight_count"] * 24 + lm_head["weight_count"],
    }
    whole["total"] = whole["payload"] + whole["packed_scale24"] + whole["mixed_codebook"] + whole["selector"]
    whole["bits_per_weight"] = whole["total"] * 8 / whole["weight_count"]
    require(standard["total"] == 7521504, "V12 standard-layer total bytes differ")
    require(lm_head["total"] == 68670416, "V12 lm_head total bytes differ")
    require(whole["total"] == 249186512, "V12 whole-model total bytes differ")
    require(standard["bits_per_weight"] == 4.035834478021978, "V12 standard-layer bits/weight differs")
    require(lm_head["bits_per_weight"] == 4.035440674268865, "V12 lm_head bits/weight differs")
    require(whole["bits_per_weight"] == 4.035725946548808, "V12 whole-model bits/weight differs")
    require(whole["bits_per_weight"] < 4.036, "V12 whole-model storage cap failed")
    return {"standard_decoder_layer": standard, "lm_head": lm_head, "whole_model": whole}


def combined_self_test() -> dict[str, Any]:
    environment = {
        **reference.require_project_python(),
        "platform": platform.platform(),
        "packages": {"torch": importlib.metadata.version("torch")},
        "backend": "V10-preserving fixed-Scale24 mixed four/dual-bank constructor kernel",
    }
    reference.verify_companion(TASK_PATH, TASK_COMPANION)
    reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    task = load_json(TASK_PATH)
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "frozen V12 Engineer authority differs")
    require(task["fresh_namespace"]["attempt_budget"] == 1, "V12 attempt budget differs")
    root_existed_before = OUTPUT_DIR.exists()
    marker_existed_before = MARKER_PATH.exists()
    require(not ATTEMPT_DIR.exists() and not marker_existed_before, "V12 official attempt must remain absent during self-test")

    require(len(_block_ranges(896)) == 7 and len(_block_ranges(4864)) == 38, "V12 input-block geometry differs")
    illegal_width_rejected = reference.expect_rejected(lambda: _block_ranges(1024), "illegal V12 width was accepted")
    groups = lm_head_group_ranges()
    require(all(stop - start == LM_HEAD_ROWS_PER_GROUP for start, stop in groups), "V12 lm_head group size differs")
    require(all(groups[index][1] == groups[index + 1][0] for index in range(len(groups) - 1)), "V12 lm_head groups are not contiguous")

    one_bit = torch.tensor([[0, 1, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0]], dtype=torch.uint8)
    one_bit_raw = _pack_selectors(one_bit, DUAL_BANK_SELECTOR_BITS)
    require(torch.equal(_unpack_selectors(one_bit_raw, 2, 7, DUAL_BANK_SELECTOR_BITS), one_bit), "V12 one-bit selector round-trip differs")
    require(one_bit_raw[0] == 0b11010110, "V12 one-bit selector order differs")
    two_bit = torch.tensor([[0, 1, 2, 3, 0, 1, 2], [3, 2, 1, 0, 3, 2, 1]], dtype=torch.uint8)
    two_bit_raw = _pack_selectors(two_bit, FOUR_BANK_SELECTOR_BITS)
    require(torch.equal(_unpack_selectors(two_bit_raw, 2, 7, FOUR_BANK_SELECTOR_BITS), two_bit), "V12 two-bit selector round-trip differs")
    require(two_bit_raw[:2] == bytes((0b11100100, 0b11100100)), "V12 two-bit selector order differs")
    tampered_padding = bytearray(two_bit_raw)
    tampered_padding[-1] = 1
    padding_rejected = reference.expect_rejected(
        lambda: _unpack_selectors(bytes(tampered_padding), 2, 7, FOUR_BANK_SELECTOR_BITS),
        "nonzero V12 selector padding was accepted",
    )

    lanes = torch.arange(896, dtype=torch.int64)
    row0 = ((lanes % 47) - 23).to(torch.float64) / 256.0
    standard_fixture = torch.stack(
        (
            row0,
            torch.where((lanes % 5) == 0, row0 * 0.25, row0 * 1.5),
            torch.where((lanes % 7) < 3, -row0 * 0.75, row0 * 0.125),
            torch.zeros_like(row0),
        )
    ).to(torch.bfloat16)
    standard_a = construct_v12_state(standard_fixture, "self_test.q_proj")
    standard_b = construct_v12_state(standard_fixture, "self_test.q_proj")
    require(_state_hashes(standard_a) == _state_hashes(standard_b), "independent V12 four-bank fixture hashes differ")
    require(standard_a["iteration_trace"] == standard_b["iteration_trace"], "independent V12 four-bank traces differ")
    require(standard_a["bank_count"] == FOUR_BANK_COUNT and standard_a["selector_bits"] == FOUR_BANK_SELECTOR_BITS, "V12 four-bank format differs")
    require(standard_a["packed_scale24"] == v10._pack_scale24_stream(standard_a["final_records"]), "V12 Scale24 stream changed")
    require(standard_a["final_v12_literal_bf16_sse"] <= standard_a["initial_v10_literal_bf16_sse"], "V12 standard fixture regressed")
    require(standard_a["outer_iterations_completed"] == OUTER_ITERATIONS, "V12 standard iteration count differs")

    down = construct_v12_state(standard_fixture, "self_test.down_proj")
    require(down["mode"] == "down_proj_v10_dual_bank_preserved", "V12 down_proj mode differs")
    require(hashlib.sha256(down["payload"]).hexdigest() == down["v10_seed"]["payload_sha256"], "V12 down_proj payload changed")
    require(
        hashlib.sha256(down["packed_scale24"]).hexdigest() == down["v10_seed"]["packed_scale24_stream_sha256"],
        "V12 down_proj packed Scale24 changed",
    )
    require(
        hashlib.sha256(down["codebook_raw"]).hexdigest() == down["v10_seed"]["dual_codebook_stream_sha256"],
        "V12 down_proj codebooks changed",
    )
    require(
        hashlib.sha256(down["selector_raw"]).hexdigest() == down["v10_seed"]["selector_stream_sha256"],
        "V12 down_proj selectors changed",
    )
    require(down["initial_v10_literal_bf16_sse"] == down["final_v12_literal_bf16_sse"], "V12 down_proj changed")

    grouped_fixture = torch.stack(
        tuple(
            torch.where((lanes % (5 + row)) < (2 + row % 3), row0 * (0.5 + row / 8.0), -row0 * (0.25 + row / 16.0))
            for row in range(64)
        )
    ).to(torch.bfloat16)
    grouped_seed_a = v10.construct_packed_scale24_state(grouped_fixture, "lm_head")
    grouped_seed_b = v10.construct_packed_scale24_state(grouped_fixture, "lm_head")
    grouped_a = _construct_grouped_lm_head_from_seed(grouped_fixture, "self_test.lm_head", grouped_seed_a, 32)
    grouped_b = _construct_grouped_lm_head_from_seed(grouped_fixture, "self_test.lm_head", grouped_seed_b, 32)
    require(_state_hashes(grouped_a) == _state_hashes(grouped_b), "independent V12 grouped fixture hashes differ")
    require(grouped_a["iteration_trace"] == grouped_b["iteration_trace"], "independent V12 grouped traces differ")
    require(grouped_a["group_count"] == 2 and grouped_a["group_selector_metadata_bits"] == 0, "V12 grouped fixture geometry differs")
    require(len(grouped_a["codebook_raw"]) == 2 * 7 * 2 * 16, "V12 grouped codebook bytes differ")
    require(grouped_a["final_v12_literal_bf16_sse"] <= grouped_a["initial_v10_literal_bf16_sse"], "V12 grouped fixture regressed")

    full_lm_selectors = torch.zeros((LM_HEAD_ROWS, 7), dtype=torch.uint8)
    full_lm_selector_raw = _pack_selectors(full_lm_selectors, DUAL_BANK_SELECTOR_BITS)
    require(len(full_lm_selector_raw) == 132944, "V12 full lm_head selector bytes differ")
    require(
        torch.equal(
            _unpack_selectors(full_lm_selector_raw, LM_HEAD_ROWS, 7, DUAL_BANK_SELECTOR_BITS), full_lm_selectors
        ),
        "V12 full lm_head selector round-trip differs",
    )
    storage = _storage_closure()

    build_root = ROOT / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v12-hash-rejection-", dir=build_root) as temporary_directory:
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
            "tampered copied V12 artifact passed companion verification",
        )

    require(OUTPUT_DIR.exists() == root_existed_before, "V12 self-test changed the official namespace")
    require(MARKER_PATH.exists() == marker_existed_before is False, "V12 self-test changed the execution marker")
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
            "lm_head_exact_64_groups_of_2374_rows": True,
            "lm_head_group_transitions_contiguous": True,
            "lm_head_group_selector_metadata_bits": 0,
            "one_bit_selector_row_major_block_major_lsb_first": True,
            "two_bit_selector_row_major_block_major_lsb_first": True,
            "selector_alignment_and_zero_padding": True,
            "nonzero_selector_padding_rejected": padding_rejected,
            "v10_seed_decode_preserved_before_standard_optimization": True,
            "selected_standard_family_four_bank_two_bit_format": True,
            "down_proj_exact_v10_dual_bank_preservation": True,
            "grouped_lm_head_dual_bank_kernel_deterministic": True,
            "fixed_packed_scale24_records_preserved": True,
            "eight_iterations_completed": True,
            "literal_bf16_non_regression_acceptance": True,
            "independent_candidate_a_b_state_and_trace_match": True,
            "exact_storage_and_sub_4_036_closure": True,
            "bound_hash_mismatch_rejected": hash_mismatch_rejected,
        },
        "standard_fixture": {
            "shape": list(standard_fixture.shape),
            "initial_v10_literal_bf16_sse": standard_a["initial_v10_literal_bf16_sse"],
            "final_v12_literal_bf16_sse": standard_a["final_v12_literal_bf16_sse"],
            **_state_hashes(standard_a),
        },
        "grouped_fixture": {
            "shape": list(grouped_fixture.shape),
            "group_size": grouped_a["group_size"],
            "group_count": grouped_a["group_count"],
            "initial_v10_literal_bf16_sse": grouped_a["initial_v10_literal_bf16_sse"],
            "final_v12_literal_bf16_sse": grouped_a["final_v12_literal_bf16_sse"],
            **_state_hashes(grouped_a),
        },
        "storage": storage,
        "official_build_namespace_changed": False,
        "official_execution_marker_exists": False,
        "claim_boundary": "constructor kernel and synthetic fixture self-test only; no 169-tensor, quality, Dynamic Scale32, official-attempt, RTL, or PPA conclusion",
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


def _validate_full_state(state: dict[str, Any]) -> dict[str, Any]:
    rows, width = (int(value) for value in state["shape"])
    blocks = _block_ranges(width)
    require(state["final_assignments"].shape == (rows, width), "V12 payload geometry differs")
    require(bool(torch.all(state["final_assignments"] < CODEBOOK_ENTRIES)), "V12 payload escaped four bits")
    require(state["final_selectors"].shape == (rows, len(blocks)), "V12 selector geometry differs")
    require(
        torch.equal(
            _unpack_selectors(state["selector_raw"], rows, len(blocks), int(state["selector_bits"])),
            state["final_selectors"],
        ),
        "V12 selector round trip differs",
    )
    require(
        torch.equal(v10._unpack_scale24_stream(state["packed_scale24"], rows), state["final_records"]),
        "V12 packed Scale24 round trip differs",
    )
    require(len(state["payload"]) == rows * width // 2, "V12 payload byte count differs")
    require(len(state["packed_scale24"]) == rows * 3, "V12 packed Scale24 byte count differs")
    require(len(state["selector_raw"]) % ALIGNMENT_BYTES == 0, "V12 selector stream is not aligned")
    mode = state["mode"]
    if mode == "standard_four_bank":
        codebooks = _validate_standard_codebooks(state["final_codebooks"], len(blocks), FOUR_BANK_COUNT)
        expected_codebook_bytes = len(blocks) * FOUR_BANK_COUNT * CODEBOOK_ENTRIES
        group_count = 1
        group_size = rows
        group_selector_metadata_bits = 0
    elif mode == "down_proj_v10_dual_bank_preserved":
        codebooks = _validate_standard_codebooks(state["final_codebooks"], len(blocks), DUAL_BANK_COUNT)
        expected_codebook_bytes = len(blocks) * DUAL_BANK_COUNT * CODEBOOK_ENTRIES
        group_count = 1
        group_size = rows
        group_selector_metadata_bits = 0
    else:
        require(mode == "lm_head_grouped_dual_bank", "V12 state mode differs")
        codebooks = _validate_grouped_codebooks(state["final_codebooks"], LM_HEAD_GROUP_COUNT, len(blocks))
        expected_codebook_bytes = LM_HEAD_GROUP_COUNT * len(blocks) * DUAL_BANK_COUNT * CODEBOOK_ENTRIES
        group_count = LM_HEAD_GROUP_COUNT
        group_size = LM_HEAD_ROWS_PER_GROUP
        group_selector_metadata_bits = int(state["group_selector_metadata_bits"])
        require(group_selector_metadata_bits == 0, "V12 lm_head group selector metadata appeared")
    require(len(state["codebook_raw"]) == expected_codebook_bytes, "V12 codebook byte count differs")
    return {
        "mode": mode,
        "rows": rows,
        "input_features": width,
        "input_block_count": len(blocks),
        "bank_count": int(state.get("bank_count", state.get("bank_count_per_group_block", DUAL_BANK_COUNT))),
        "selector_bits": int(state["selector_bits"]),
        "selector_bit_count": rows * len(blocks) * int(state["selector_bits"]),
        "selector_aligned_bytes": len(state["selector_raw"]),
        "packed_scale24_bytes": len(state["packed_scale24"]),
        "mixed_codebook_bytes": len(state["codebook_raw"]),
        "group_count": group_count,
        "group_size": group_size,
        "group_selector_metadata_bits": group_selector_metadata_bits,
        "all_codebooks_strictly_increasing": all(
            all(left < right for left, right in zip(bank, bank[1:]))
            for group in (codebooks if mode == "lm_head_grouped_dual_bank" else (codebooks,))
            for banks in group
            for bank in banks
        ),
    }


def _decoded_chunk(state: dict[str, Any], row_start: int, row_stop: int) -> Tensor:
    mode = state["mode"]
    if mode == "standard_four_bank":
        return _decode_standard(
            state["final_records"][row_start:row_stop],
            state["final_assignments"][row_start:row_stop],
            state["final_codebooks"],
            state["final_selectors"][row_start:row_stop],
        )
    if mode == "down_proj_v10_dual_bank_preserved":
        return v10.v9._decoded_chunk_v9(
            state["final_records"],
            state["final_assignments"],
            state["final_codebooks"],
            state["final_selectors"],
            1,
            row_start,
            row_stop,
        )
    require(mode == "lm_head_grouped_dual_bank", "V12 chunk mode differs")
    group_size = int(state["group_size"])
    group_index = row_start // group_size
    require(row_start == group_index * group_size and row_stop == row_start + group_size, "V12 lm_head chunk is not one fixed group")
    return _decode_grouped(
        state["final_records"][row_start:row_stop],
        state["final_assignments"][row_start:row_stop],
        (state["final_codebooks"][group_index],),
        state["final_selectors"][row_start:row_stop],
        group_size,
    )


def _measure_and_materialize(
    module: nn.Linear,
    source: Tensor,
    state: dict[str, Any],
    sealed_v10_record: dict[str, Any],
    sealed_v10_manifest_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows, width = (int(value) for value in source.shape)
    expected_components = sealed_v10_manifest_record["components"]
    seed = state["v10_seed"]
    seed_checks = {
        "payload": seed["payload_sha256"] == expected_components["payload"]["sha256"],
        "packed_scale24": seed["packed_scale24_stream_sha256"] == expected_components["packed_scale24"]["sha256"],
        "dual_codebook": seed["dual_codebook_stream_sha256"] == expected_components["dual_codebook"]["sha256"],
        "selector": seed["selector_stream_sha256"] == expected_components["selector"]["sha256"],
        "reconstruction": seed["reconstruction_bf16_sha256"] == sealed_v10_record["reconstruction_bf16_sha256"],
        "literal_bf16_sse": math.isclose(
            float(state["initial_v10_literal_bf16_sse"]),
            float(sealed_v10_record["v10_packed_scale24_per_row_selector_bf16_sse"]),
            rel_tol=1e-15,
            abs_tol=1e-12,
        ),
    }
    seed_record = {
        "module": sealed_v10_record["module"],
        "weight_count": int(source.numel()),
        "component_hash_matches": seed_checks,
        "all_component_and_reconstruction_hashes_match": all(seed_checks.values()),
        "sealed_v10_literal_bf16_sse": float(sealed_v10_record["v10_packed_scale24_per_row_selector_bf16_sse"]),
        "fresh_v10_literal_bf16_sse": float(state["initial_v10_literal_bf16_sse"]),
        "fresh_v10": seed,
        "sealed_v10": {
            "payload_sha256": expected_components["payload"]["sha256"],
            "packed_scale24_stream_sha256": expected_components["packed_scale24"]["sha256"],
            "dual_codebook_stream_sha256": expected_components["dual_codebook"]["sha256"],
            "selector_stream_sha256": expected_components["selector"]["sha256"],
            "reconstruction_bf16_sha256": sealed_v10_record["reconstruction_bf16_sha256"],
        },
    }

    v12_sse = 0.0
    v12_digest = hashlib.sha256()
    chunks = (
        _group_ranges(rows, LM_HEAD_ROWS_PER_GROUP)
        if state["mode"] == "lm_head_grouped_dual_bank"
        else tuple(v4._row_chunks(rows, width))
    )
    with torch.no_grad():
        for row_start, row_stop in chunks:
            decoded = _decoded_chunk(state, row_start, row_stop)
            difference = source[row_start:row_stop].to(torch.float64) - decoded.to(torch.float64)
            v12_sse += float(torch.sum(difference * difference))
            v12_digest.update(v7.v3.raw_tensor_bytes(decoded))
            module.weight[row_start:row_stop].copy_(decoded)
    v12_sha = v12_digest.hexdigest()
    sealed_v10_sse = float(sealed_v10_record["v10_packed_scale24_per_row_selector_bf16_sse"])
    require(v12_sse >= 0.0 and sealed_v10_sse >= 0.0, f"negative BF16 reconstruction SSE: {sealed_v10_record['module']}")
    require(v12_sha == v7.v3.tensor_sha256(module.weight), f"V12 materialized reconstruction hash differs: {sealed_v10_record['module']}")
    reconstruction = {
        "module": sealed_v10_record["module"],
        "weight_count": int(source.numel()),
        "sealed_v10_bf16_sse": sealed_v10_sse,
        "sealed_v10_reconstruction_bf16_sha256": sealed_v10_record["reconstruction_bf16_sha256"],
        "v10_seed_exact_reproduction": seed_record["all_component_and_reconstruction_hashes_match"],
        "v12_bf16_sse": v12_sse,
        "v12_non_regressing_against_sealed_v10": v12_sse <= sealed_v10_sse,
        "v12_strictly_improved_against_sealed_v10": v12_sse < sealed_v10_sse,
        "reconstruction_bf16_sha256": v12_sha,
    }
    return seed_record, reconstruction


def _policy_and_format() -> dict[str, Any]:
    body = {
        "payload": {
            "bits_per_weight": 4,
            "semantics": "unsigned selected-bank codebook index 0 through 15",
            "packing": "row-major; even flat element low nibble; odd flat element high nibble",
        },
        "row_scale": {
            "stored_format": "packed_scale24",
            "bytes_per_output_row": 3,
            "decode": "restore one zero reserved high byte before Scale32 validation and arithmetic",
            "preserved_from_v10": True,
        },
        "standard_layers": {
            "four_bank_families": list(reference.FOUR_BANK_FAMILIES),
            "four_bank_selector_bits": FOUR_BANK_SELECTOR_BITS,
            "dual_bank_families": list(reference.DUAL_BANK_FAMILIES),
            "dual_bank_selector_bits": DUAL_BANK_SELECTOR_BITS,
            "input_block_lanes": INPUT_BLOCK_LANES,
            "entries_per_bank": CODEBOOK_ENTRIES,
        },
        "lm_head": {
            "group_count": LM_HEAD_GROUP_COUNT,
            "rows_per_group": LM_HEAD_ROWS_PER_GROUP,
            "banks_per_group_block": DUAL_BANK_COUNT,
            "selector_bits": DUAL_BANK_SELECTOR_BITS,
            "group_selector_metadata_bits": 0,
        },
        "construction": {
            "v10_seed": "fresh exact reviewed V10 construction under V12 namespace",
            "outer_iterations": OUTER_ITERATIONS,
            "reported_model_gate": "literal nonnegative BF16-materialized reconstruction SSE",
        },
        "activation": {
            "policy": "grouped Dynamic Scale32",
            "event_count": EXPECTED_EVENTS,
            "maximum_transport_metadata_bytes_per_decoder_layer_token": 832,
        },
        "storage": _storage_closure(),
        "reviewed_v10_runner_sha256": reference.sha256_file(Path(v10.reference.__file__).resolve()),
        "reviewed_v10_backend_sha256": reference.sha256_file(Path(v10.__file__).resolve()),
    }
    return {**body, "policy_and_format_sha256": canonical_sha256(body)}


def _quantize_model(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {record["module"]: record for record in v7.grouped.expected_transformer_tensors()}
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == 169, "Qwen linear count differs")
    require([name for name, _module in modules if name != "lm_head"] == list(expected), "transformer linear order differs")
    require(reference.sha256_file(V10_RECONSTRUCTION_PATH) == V10_RECONSTRUCTION_SHA256, "sealed V10 reconstruction hash differs")
    require(reference.sha256_file(V10_CANDIDATE_MANIFEST_PATH) == V10_CANDIDATE_MANIFEST_SHA256, "sealed V10 candidate manifest hash differs")
    sealed_report = load_json(V10_RECONSTRUCTION_PATH)
    sealed_records = {item["module"]: item for item in sealed_report["tensors"]}
    sealed_manifest = load_json(V10_CANDIDATE_MANIFEST_PATH)
    sealed_manifest_records = {item["module"]: item for item in sealed_manifest["weight_manifest"]}
    module_names = [name for name, _module in modules]
    require(list(sealed_records) == module_names, "sealed V10 reconstruction tensor order differs")
    require(list(sealed_manifest_records) == module_names, "sealed V10 manifest tensor order differs")

    manifest_names = (
        "payload",
        "packed_scale24",
        "mixed_codebook",
        "selector",
        "v10_seed",
        "reconstruction",
        "objective",
        "iteration_trace",
    )
    manifests: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest_names}
    weight_manifest: list[dict[str, Any]] = []
    streams = {name: hashlib.sha256() for name in ("payload", "packed_scale24", "mixed_codebook", "selector", "reconstruction")}
    totals = {"weight_count": 0, "payload": 0, "packed_scale24": 0, "mixed_codebook": 0, "selector": 0}

    for ordinal, (name, module) in enumerate(modules, start=1):
        started = time.monotonic()
        print(f"V12_CONSTRUCT_START {ordinal}/169 {name}", flush=True)
        source = module.weight.detach()
        source_sha = v7.v3.tensor_sha256(source)
        state = construct_v12_state(source, name)
        geometry = _validate_full_state(state)
        seed_record, reconstruction = _measure_and_materialize(
            module, source, state, sealed_records[name], sealed_manifest_records[name]
        )
        raw = {
            "payload": state["payload"],
            "packed_scale24": state["packed_scale24"],
            "mixed_codebook": state["codebook_raw"],
            "selector": state["selector_raw"],
        }
        component_records = {
            component: {"module": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            for component, value in raw.items()
        }
        component_records["packed_scale24"].update(
            {"record_count": int(source.shape[0]), "bytes_per_record": 3, "preserved_from_v10": True}
        )
        component_records["mixed_codebook"].update(
            {
                "mode": state["mode"],
                "input_block_count": state["input_block_count"],
                "group_count": int(state.get("group_count", 1)),
                "banks_per_block": int(state.get("bank_count", state.get("bank_count_per_group_block", DUAL_BANK_COUNT))),
                "q4_4_banks": json.loads(json.dumps(state["final_codebooks"])),
            }
        )
        component_records["selector"].update(
            {
                "output_row_count": int(source.shape[0]),
                "selector_bits_per_row_block": int(state["selector_bits"]),
                "selector_bit_count": int(source.shape[0]) * state["input_block_count"] * int(state["selector_bits"]),
                "alignment_bytes": ALIGNMENT_BYTES,
                "padding_bits_zero": True,
                "group_selector_metadata_bits": int(state.get("group_selector_metadata_bits", 0)),
            }
        )
        for component in ("payload", "packed_scale24", "mixed_codebook", "selector"):
            manifests[component].append(component_records[component])
            streams[component].update(name.encode() + b"\0" + raw[component])
            totals[component] += len(raw[component])
        manifests["v10_seed"].append(seed_record)
        streams["reconstruction"].update(name.encode() + b"\0" + bytes.fromhex(reconstruction["reconstruction_bf16_sha256"]))
        manifests["reconstruction"].append(reconstruction)
        objective_record = {
            "module": name,
            "kind": "literal BF16 source-space SSE",
            "initial_v10_literal_bf16_sse": state["initial_v10_literal_bf16_sse"],
            "final_v12_literal_bf16_sse": state["final_v12_literal_bf16_sse"],
        }
        manifests["objective"].append(objective_record)
        trace_record = {
            "module": name,
            "source_bf16_sha256": source_sha,
            "mode": state["mode"],
            "input_block_count": state["input_block_count"],
            "outer_iterations_completed": state["outer_iterations_completed"],
            "iteration_trace": state["iteration_trace"],
            "iteration_trace_sha256": canonical_sha256(state["iteration_trace"]),
        }
        manifests["iteration_trace"].append(trace_record)
        record = {
            "module": name,
            "name": f"{name}.weight",
            "shape": list(source.shape),
            "scope": "separated_lm_head" if name == "lm_head" else "transformer",
            "geometry": geometry,
            "components": component_records,
            "v10_seed": seed_record,
            "reconstruction": reconstruction,
            "objective": objective_record,
            "iteration_trace_sha256": trace_record["iteration_trace_sha256"],
            "metadata_sequence": (
                ["grouped_dual_codebooks_by_fixed_lm_head_group", "one_bit_selector", "payload", "packed_scale24"]
                if name == "lm_head"
                else ["mixed_codebooks_by_input_block", "selector", "payload", "packed_scale24"]
            ),
        }
        weight_manifest.append(record)
        totals["weight_count"] += int(source.numel())
        print(
            f"V12_CONSTRUCT_DONE {ordinal}/169 {name} seconds={time.monotonic()-started:.3f} "
            f"v10_sse={reconstruction['sealed_v10_bf16_sse']:.12g} v12_sse={reconstruction['v12_bf16_sse']:.12g} "
            f"seed_exact={int(seed_record['all_component_and_reconstruction_hashes_match'])}",
            flush=True,
        )
        del state
        gc.collect()

    aggregate_v10 = float(sum(item["sealed_v10_bf16_sse"] for item in manifests["reconstruction"]))
    aggregate_v12 = float(sum(item["v12_bf16_sse"] for item in manifests["reconstruction"]))
    seed_count = sum(bool(item["all_component_and_reconstruction_hashes_match"]) for item in manifests["v10_seed"])
    non_regressing_count = sum(bool(item["v12_non_regressing_against_sealed_v10"]) for item in manifests["reconstruction"])
    improved_count = sum(bool(item["v12_strictly_improved_against_sealed_v10"]) for item in manifests["reconstruction"])
    require(aggregate_v10 == V10_AGGREGATE_BF16_SSE, "sealed aggregate V10 BF16 SSE differs")
    seed_body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "reference": {"candidate_id": v10.CANDIDATE_ID, "reconstruction": file_record(V10_RECONSTRUCTION_PATH), "candidate_manifest": file_record(V10_CANDIDATE_MANIFEST_PATH)},
        "tensor_count": 169,
        "v10_seed_exact_reproduction_count": seed_count,
        "all_169_v10_seed_reproductions_match": seed_count == 169,
        "aggregate_sealed_v10_bf16_sse": aggregate_v10,
        "tensors": manifests["v10_seed"],
    }
    seed_report = {
        **seed_body,
        "status": "PASS_169_OF_169_EXACT_V10_SEED_REPRODUCTION" if seed_count == 169 else "NO_GO_V10_SEED_REPRODUCTION_GATE",
        "seed_report_sha256": canonical_sha256(seed_body),
    }
    reconstruction_body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "measurement": "literal nonnegative sum-squared error from pinned BF16 source to BF16-materialized reconstruction",
        "reference": {"candidate_id": v10.CANDIDATE_ID, "artifact": file_record(V10_RECONSTRUCTION_PATH), "aggregate_bf16_sse": V10_AGGREGATE_BF16_SSE},
        "tensor_count": 169,
        "per_tensor_non_regressing_count": non_regressing_count,
        "per_tensor_strictly_improved_count": improved_count,
        "all_169_tensors_non_regressing_against_sealed_v10": non_regressing_count == 169,
        "aggregate_sealed_v10_bf16_sse": aggregate_v10,
        "aggregate_v12_bf16_sse": aggregate_v12,
        "aggregate_strictly_improved_against_sealed_v10": aggregate_v12 < aggregate_v10,
        "tensors": manifests["reconstruction"],
    }
    model_pass = non_regressing_count == 169 and aggregate_v12 < aggregate_v10
    reconstruction_report = {
        **reconstruction_body,
        "status": "PASS_169_OF_169_V10_LEDGER_WITH_STRICT_AGGREGATE_IMPROVEMENT" if model_pass else "NO_GO_V10_LEDGER_MODEL_ONLY_RECONSTRUCTION_GATE",
        "reconstruction_report_sha256": canonical_sha256(reconstruction_body),
    }
    require(
        totals
        == {
            "weight_count": 493961216,
            "payload": 246980608,
            "packed_scale24": 1368192,
            "mixed_codebook": 108032,
            "selector": 729680,
        },
        f"whole-model V12 storage geometry differs: {totals}",
    )
    hashes = {f"{name}_manifest_sha256": canonical_sha256(values) for name, values in manifests.items()}
    hashes.update(
        {
            "weight_manifest_sha256": canonical_sha256(weight_manifest),
            "v10_seed_report_sha256": seed_report["seed_report_sha256"],
            "reconstruction_report_sha256": reconstruction_report["reconstruction_report_sha256"],
        }
    )
    hashes.update({f"{name}_stream_sha256": digest.hexdigest() for name, digest in streams.items()})
    closure = _storage_closure()
    storage = {
        "total_weight_count": totals["weight_count"],
        "total_payload_bytes": totals["payload"],
        "total_packed_scale24_bytes": totals["packed_scale24"],
        "total_mixed_codebook_bytes": totals["mixed_codebook"],
        "total_selector_bytes": totals["selector"],
        "whole_model_total_weight_bytes": sum(totals[name] for name in ("payload", "packed_scale24", "mixed_codebook", "selector")),
        "whole_model_exact_bits_per_weight": closure["whole_model"]["bits_per_weight"],
        "standard_decoder_layer_exact_bits_per_weight": closure["standard_decoder_layer"]["bits_per_weight"],
        "lm_head_exact_bits_per_weight": closure["lm_head"]["bits_per_weight"],
        "maximum_total_stored_bits_per_weight": 4.036,
    }
    return weight_manifest, {
        "manifests": manifests,
        "hashes": hashes,
        "storage": storage,
        "v10_seed": seed_report,
        "reconstruction": reconstruction_report,
    }


def _load_candidate(label: str) -> tuple[nn.Module, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    require(model.model.embed_tokens.weight is embedding and embedding.data_ptr() == source_pointer, "embedding identity changed after V12 construction")
    require(list(embedding.shape) == source_shape and str(embedding.dtype) == source_dtype, "embedding shape or dtype changed")
    embedding_hash = v7.v3.tensor_sha256(embedding)
    require(embedding_hash == source_hash == SOURCE_EMBEDDING_SHA256, "embedding bytes changed after V12 construction")
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
        "module_counts": {"transformer_linears": 168, "standard_four_bank_linears": 144, "preserved_down_proj_linears": 24, "grouped_lm_head_linears": 1},
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
        "post_v12": {
            "embedding_bf16_sha256": embedding_hash,
            "embedding_hash_matches_source": embedding_hash == SOURCE_EMBEDDING_SHA256,
            "lm_head_storage_is_separate": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
            "standard_four_bank_count": 144,
            "down_proj_v10_preserved_count": 24,
            "lm_head_grouped_dual_bank_count": 1,
        },
    }
    return model, manifest, witness, construction["v10_seed"], construction["reconstruction"]


def _construct_candidate_artifacts(
    manifest_path: str, witness_path: str, seed_path: str, reconstruction_path: str
) -> None:
    torch.set_num_threads(CONSTRUCTION_TORCH_THREADS)
    torch.set_num_interop_threads(1)
    print("V12_CANDIDATE_A_WORKER_START", flush=True)
    candidate, manifest, witness, seed, reconstruction = _load_candidate("candidate_a_independent_construction")
    write_json(Path(manifest_path), manifest)
    write_json(Path(witness_path), witness)
    write_json(Path(seed_path), seed)
    write_json(Path(reconstruction_path), reconstruction)
    del candidate
    gc.collect()
    print("V12_CANDIDATE_A_WORKER_DONE", flush=True)


def _compare_candidates(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> dict[str, Any]:
    comparisons = {key: candidate_a["identity"][key] == candidate_b["identity"][key] for key in STATE_MATCH_KEYS}
    require(all(comparisons.values()), f"candidate A/B identity differs: {comparisons}")
    manifest_fields = tuple(
        f"{name}_manifest"
        for name in ("payload", "packed_scale24", "mixed_codebook", "selector", "v10_seed", "reconstruction", "objective", "iteration_trace")
    ) + ("weight_manifest", "state_manifest")
    field_matches = {field: candidate_a[field] == candidate_b[field] for field in manifest_fields}
    require(all(field_matches.values()), f"candidate A/B whole-model manifests differ: {field_matches}")
    return {"identity_matches": comparisons, "manifest_matches": field_matches, "all_match": True}


def _validate_seed(report: dict[str, Any]) -> bool:
    return report["tensor_count"] == 169 and report["v10_seed_exact_reproduction_count"] == 169 and report["all_169_v10_seed_reproductions_match"] is True


def _validate_reconstruction(report: dict[str, Any]) -> bool:
    return (
        report["tensor_count"] == 169
        and report["per_tensor_non_regressing_count"] == 169
        and report["aggregate_sealed_v10_bf16_sse"] == V10_AGGREGATE_BF16_SSE
        and report["aggregate_v12_bf16_sse"] < V10_AGGREGATE_BF16_SSE
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
        require(not root_existed, "fresh V12 namespace already exists")
    markers = list(OUTPUT_DIR.rglob("execution_started.json")) if root_existed else []
    require(not markers and not ATTEMPT_DIR.exists(), "V12 official attempt or marker exists")
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
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "cannot recover a V12 namespace with an official attempt")
    entries = sorted(OUTPUT_DIR.iterdir())
    if not entries:
        OUTPUT_DIR.rmdir()
        return {"recovered": True, "fresh_namespace": True, "archived_artifacts": []}
    require(not CONTRACT_PATH.exists() and not READY_PATH.exists() and not NO_GO_PATH.exists(), "completed V12 preflight may not be recovered")
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
        "terminal_v11_review": V11_REVIEW_PATH,
        "reviewed_v10_review": V10_REVIEW_PATH,
        "reviewed_v10_reconstruction_ledger": V10_RECONSTRUCTION_PATH,
        "reviewed_v10_candidate_manifest": V10_CANDIDATE_MANIFEST_PATH,
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
        "v10_compatible_seed_reproduction": V10_SEED_PATH,
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
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "V12 Engineer authority differs")
    require(
        task["fresh_namespace"]["attempt_budget"] == 1
        and task["fresh_namespace"]["attempts_consumed_at_task_freeze"] == 0,
        "V12 attempt accounting differs",
    )
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
            "terminal_v11_review": artifacts["terminal_v11_review"],
            "reviewed_v10_review": artifacts["reviewed_v10_review"],
            "attempt_id": "attempt-0001",
            "exact_authorized_candidate_attempts": 1,
            "attempts_consumed": 0,
            "selected_policy_id": None,
        },
        "source_model": source_identity,
        "environment": environment,
        "format_and_policy": _policy_and_format(),
        "construction": {
            "backend": "fresh exact V10 seed plus V12 mixed four/dual-bank and fixed lm_head64-group optimization",
            "source_tied_then_runtime_split_lm_head": True,
            "source_embedding_bf16_sha256": SOURCE_EMBEDDING_SHA256,
            "embedding_bytes_preserved": True,
            "transformer_linears": 168,
            "separated_lm_head_linears": 1,
            "outer_iterations": OUTER_ITERATIONS,
            "candidate_a_and_b_whole_model_manifests_match": state_comparison,
            "packed_scale24_records_preserved": True,
            "model_gate_uses_actual_nonnegative_bf16_sse": True,
        },
        "pre_attempt_gates": {
            "v10_seed_exact_169": {"passed": gate_passes["v10_seed_exact_169"], "artifact": artifacts["v10_compatible_seed_reproduction"]},
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
    require(CONTRACT_PATH.is_file(), "V12 predeclared contract is missing")
    require(READY_PATH.is_file() != NO_GO_PATH.is_file(), "exactly one V12 pre-attempt terminal record is required")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "V12 predeclared contract wrapper differs")
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "V12 predeclared contract canonical hash differs")
    contract = wrapper["contract"]
    require(contract["candidate_id"] == CANDIDATE_ID and contract["task_id"] == TASK_ID, "V12 predeclared authority differs")
    require(contract["artifacts"]["runner_source"]["sha256"] == reference.sha256_file(RUNNER_PATH), "V12 runner changed after freeze")
    require(contract["artifacts"]["scalable_backend_source"]["sha256"] == reference.sha256_file(BACKEND_PATH), "V12 backend changed after freeze")
    for record in contract["artifacts"].values():
        _verify_bound_record(record)
    reference.verify_companion(TASK_PATH, TASK_COMPANION)
    reference.verify_companion(PLAN_PATH, PLAN_COMPANION)
    require(reference.sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(reference.sha256_file(V10_RECONSTRUCTION_PATH) == V10_RECONSTRUCTION_SHA256, "sealed V10 reconstruction hash differs")
    require(reference.sha256_file(V10_CANDIDATE_MANIFEST_PATH) == V10_CANDIDATE_MANIFEST_SHA256, "sealed V10 candidate manifest hash differs")
    require(reference.sha256_file(V10_REVIEW_PATH) == V10_REVIEW_SHA256, "sealed V10 review hash differs")
    require(reference.sha256_file(V11_REVIEW_PATH) == V11_REVIEW_SHA256, "terminal V11 review hash differs")
    require(verify_source_snapshot() == contract["source_model"], "source snapshot identity differs")
    packages = {**verify_versions(), "numpy": importlib.metadata.version("numpy")}
    require(packages == contract["environment"]["packages"], "package versions differ")
    stored_self_test = load_json(SELF_TEST_PATH)
    require(stored_self_test["runner_sha256"] == reference.sha256_file(RUNNER_PATH), "stored V12 self-test runner hash differs")
    require(stored_self_test["backend_sha256"] == reference.sha256_file(BACKEND_PATH), "stored V12 self-test backend hash differs")
    if require_unconsumed:
        require(combined_self_test() == stored_self_test, "V12 runner self-test result differs")
    candidate_a = load_json(CANDIDATE_A_PATH)
    candidate_b = load_json(CANDIDATE_B_PATH)
    comparison = _compare_candidates(candidate_a, candidate_b)
    witness = load_json(ALIAS_WITNESS_PATH)
    require(witness["state_comparison"] == comparison, "V12 alias witness comparison differs")
    gate_passes = {
        "v10_seed_exact_169": _validate_seed(load_json(V10_SEED_PATH)),
        "v10_ledger_and_169_model_closure": _validate_reconstruction(load_json(RECONSTRUCTION_PATH)),
        "non_evaluator_quality_8_of_8": _validate_quality(load_json(QUALITY_PATH)),
        "dynamic_312_event_closure": _validate_dynamic(load_json(DYNAMIC_PATH)),
        "two_build_alias_codec_and_manifest_closure": (
            comparison["all_match"]
            and witness["embedding_preserved_in_both"]
            and witness["lm_head_separate_in_both"]
            and witness["packed_scale24_codec_passed"]
            and witness["mixed_selector_codecs_passed"]
            and witness["lm_head_group_schedule_passed"]
        ),
    }
    require(contract["pre_attempt_gates"]["all_pass"] == all(gate_passes.values()), "V12 contract aggregate gate status differs")
    for name, passed in gate_passes.items():
        require(contract["pre_attempt_gates"][name]["passed"] == passed, f"V12 contract gate differs: {name}")
    terminal_path = READY_PATH if READY_PATH.is_file() else NO_GO_PATH
    terminal = load_json(terminal_path)
    require(terminal["candidate_id"] == CANDIDATE_ID, "V12 terminal candidate differs")
    require(terminal["contract"]["sha256"] == reference.sha256_file(CONTRACT_PATH), "V12 terminal contract file hash differs")
    require(terminal["contract_sha256"] == wrapper["contract_sha256"], "V12 terminal contract canonical hash differs")
    require(terminal["bound_artifacts"] == contract["artifacts"], "V12 terminal artifact snapshot differs")
    require((terminal_path == READY_PATH) == all(gate_passes.values()), "V12 terminal disposition differs from gates")
    if require_unconsumed or terminal_path == NO_GO_PATH:
        require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "V12 official attempt exists before authorization consumption")
    seed = load_json(V10_SEED_PATH)
    quality = load_json(QUALITY_PATH)
    dynamic = load_json(DYNAMIC_PATH)
    reconstruction = load_json(RECONSTRUCTION_PATH)
    return {
        "status": terminal["status"],
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": wrapper["contract_sha256"],
        "gate_passes": gate_passes,
        "v10_seed_exact_reproduction_count": seed["v10_seed_exact_reproduction_count"],
        "per_tensor_non_regressing_count": reconstruction["per_tensor_non_regressing_count"],
        "aggregate_sealed_v10_bf16_sse": reconstruction["aggregate_sealed_v10_bf16_sse"],
        "aggregate_v12_bf16_sse": reconstruction["aggregate_v12_bf16_sse"],
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
        "backend": "exact V10-preserving mixed four/dual-bank and fixed lm_head64-group construction",
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
    require(reference.sha256_file(V10_CANDIDATE_MANIFEST_PATH) == V10_CANDIDATE_MANIFEST_SHA256, "sealed V10 candidate manifest hash differs")
    require(reference.sha256_file(V10_REVIEW_PATH) == V10_REVIEW_SHA256, "sealed V10 review hash differs")
    require(reference.sha256_file(V11_REVIEW_PATH) == V11_REVIEW_SHA256, "terminal V11 review hash differs")
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
    require(probe.read_bytes() == b"preflight\n", "V12 output-path readback differs")
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
    candidate_a_seed_tmp = OUTPUT_DIR / ".candidate_a_seed.json"
    candidate_a_reconstruction_tmp = OUTPUT_DIR / ".candidate_a_reconstruction.json"
    chronology["candidate_a_started_at_utc"] = utc_now()
    print("V12_CANDIDATE_A_START parallel_worker", flush=True)
    context = multiprocessing.get_context("spawn")
    candidate_a_process = context.Process(
        target=_construct_candidate_artifacts,
        args=(str(candidate_a_manifest_tmp), str(candidate_a_witness_tmp), str(candidate_a_seed_tmp), str(candidate_a_reconstruction_tmp)),
        name="v12-candidate-a",
    )
    candidate_a_process.start()
    chronology["candidate_b_started_at_utc"] = utc_now()
    print("V12_CANDIDATE_B_START parallel_parent", flush=True)
    try:
        candidate_b, manifest_b, witness_b, seed_b, reconstruction_b = _load_candidate(
            "candidate_b_independent_quality_and_dynamic_preflight"
        )
    except BaseException:
        if candidate_a_process.is_alive():
            candidate_a_process.terminate()
        candidate_a_process.join()
        raise
    candidate_a_process.join()
    require(candidate_a_process.exitcode == 0, f"V12 candidate A worker failed: exitcode={candidate_a_process.exitcode}")
    manifest_a = load_json(candidate_a_manifest_tmp)
    witness_a = load_json(candidate_a_witness_tmp)
    seed_a = load_json(candidate_a_seed_tmp)
    reconstruction_a = load_json(candidate_a_reconstruction_tmp)
    for path in (candidate_a_manifest_tmp, candidate_a_witness_tmp, candidate_a_seed_tmp, candidate_a_reconstruction_tmp):
        path.unlink()
    chronology["candidate_a_complete_at_utc"] = utc_now()
    print("V12_CANDIDATE_A_DONE", flush=True)
    state_comparison = _compare_candidates(manifest_a, manifest_b)
    require(seed_a == seed_b, "V12 candidate A/B V10 seed reports differ")
    require(reconstruction_a == reconstruction_b, "V12 candidate A/B reconstruction reports differ")
    chronology["candidate_b_constructed_at_utc"] = utc_now()
    print("V12_CANDIDATE_B_CONSTRUCTED quality_dynamic_start", flush=True)
    torch.set_num_threads(TORCH_THREADS)
    quality, dynamic = v7.v3.non_evaluator_quality_and_dynamic(candidate_b, tokenizer, inputs, input_manifest)
    chronology["candidate_b_quality_and_dynamic_complete_at_utc"] = utc_now()
    print("V12_QUALITY_DYNAMIC_DONE", flush=True)
    del candidate_b
    gc.collect()

    gate_passes = {
        "v10_seed_exact_169": _validate_seed(seed_b),
        "v10_ledger_and_169_model_closure": _validate_reconstruction(reconstruction_b),
        "non_evaluator_quality_8_of_8": _validate_quality(quality),
        "dynamic_312_event_closure": _validate_dynamic(dynamic),
        "two_build_alias_codec_and_manifest_closure": (
            state_comparison["all_match"]
            and witness_a["post_v12"]["embedding_hash_matches_source"]
            and witness_b["post_v12"]["embedding_hash_matches_source"]
            and witness_a["post_v12"]["lm_head_storage_is_separate"]
            and witness_b["post_v12"]["lm_head_storage_is_separate"]
            and self_test["checks"]["fixed_packed_scale24_records_preserved"]
            and self_test["checks"]["one_bit_selector_row_major_block_major_lsb_first"]
            and self_test["checks"]["two_bit_selector_row_major_block_major_lsb_first"]
            and self_test["checks"]["lm_head_exact_64_groups_of_2374_rows"]
        ),
    }
    require(not MARKER_PATH.exists() and not ATTEMPT_DIR.exists(), "V12 official attempt appeared before artifact freeze")

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
            "embedding_preserved_in_both": witness_a["post_v12"]["embedding_hash_matches_source"] and witness_b["post_v12"]["embedding_hash_matches_source"],
            "lm_head_separate_in_both": witness_a["post_v12"]["lm_head_storage_is_separate"] and witness_b["post_v12"]["lm_head_storage_is_separate"],
            "packed_scale24_codec_passed": self_test["checks"]["fixed_packed_scale24_records_preserved"],
            "mixed_selector_codecs_passed": self_test["checks"]["one_bit_selector_row_major_block_major_lsb_first"] and self_test["checks"]["two_bit_selector_row_major_block_major_lsb_first"],
            "lm_head_group_schedule_passed": self_test["checks"]["lm_head_exact_64_groups_of_2374_rows"],
        },
    )
    write_json(V10_SEED_PATH, seed_b)
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
        "actual_nonnegative_bf16_sse": {"sealed_v10": reconstruction_b["aggregate_sealed_v10_bf16_sse"], "v12": reconstruction_b["aggregate_v12_bf16_sse"]},
        "attempt_budget": 1,
        "attempts_consumed": 0,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": True,
        "disposition": (
            "all marker-free V12 gates pass; PREATTEMPT_READY is sealed and the single official attempt may now be consumed"
            if all_pass
            else "preserve V12 no-go evidence, leave attempt-0001 absent and unconsumed, keep selected_policy_id null, and submit for Fresh Review"
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
    require(closure["status"] == "PREATTEMPT_READY", "V12 official execution requires verified PREATTEMPT_READY")
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

        candidate_model, official_manifest, official_witness, _seed, _reconstruction = _load_candidate(
            "official_attempt_candidate"
        )
        comparison = _compare_candidates(load_json(CANDIDATE_B_PATH), official_manifest)
        require(comparison["all_match"], "official V12 candidate differs from preflight")
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
            "classification": "qwen_instruct_stage1_option_b_alias_safe_v10_preserving_mixed_bank_lm_head64_single_candidate",
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
                "root_cause_hypothesis": "the V12 four-bit reconstruction changed one or more greedy logits relative to BF16",
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
                "169/169 exact V10 seed reproduction",
                "mixed-bank, packed-Scale24, selector, and lm_head64-group accounting",
                "169/169 non-regression and strict aggregate SSE improvement",
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
        log_message(
            run_log,
            f"official_attempt_complete status={numerical_status} exact_sequences={aggregate['exact_bf16_sequence_match_count']}/9 gate_matches={aggregate['response_gate_outcome_match_count']}/9",
        )
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
            "regression": "repair only the isolated marker-free V12 runner and rerun pre-attempt preparation; official authorization remains unconsumed",
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
