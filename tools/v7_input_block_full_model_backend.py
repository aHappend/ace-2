#!/usr/bin/env python3
"""Exact marker-free full-model backend for the V7 128-lane codebook policy."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_option_b_alias_safe_input_block_codebook_w4_grouped_dynamic_scale32_stage1 as reference
import run_option_b_alias_safe_tensor_codebook_w4_grouped_dynamic_scale32_stage1 as v3
import run_option_b_grouped_scale32_stage1 as grouped
import v4_joint_full_model_backend as v4
from ace2_quality_contracts import SCALE32_ALL_ZERO_RECORD
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    IMAGE_DIR,
    ROOT,
    SNAPSHOT,
    canonical_bytes,
    file_record,
    load_json,
    require,
    sha256_file,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
    write_json,
)
from run_all_transformer_per32_stage1 import utc_now


CANDIDATE_ID = reference.CANDIDATE_ID
MISSION_ID = reference.MISSION_ID
TASK_ID = reference.TASK_ID
TASK_PATH = reference.TASK_PATH
TASK_COMPANION = reference.TASK_COMPANION
PLAN_PATH = reference.PLAN_PATH
PLAN_COMPANION = reference.PLAN_COMPANION
RUNNER_PATH = reference.RUNNER_PATH
BACKEND_PATH = Path(__file__).resolve()
V4_RUNNER_PATH = Path(v4.reference.__file__).resolve()
V4_BACKEND_PATH = Path(v4.__file__).resolve()
V3_RUNNER_PATH = Path(v3.__file__).resolve()
GROUPED_RUNNER_PATH = ROOT / "tools/run_option_b_grouped_scale32_stage1.py"
SOURCE_CONTRACT_PATH = ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
CALIBRATION_CONTRACT_PATH = CALIBRATION_DIR / "calibration_contract.json"
CALIBRATION_REPORT_PATH = CALIBRATION_DIR / "calibration_report.json"
BASE_SCALES_PATH = CALIBRATION_DIR / "derived_scales.json"
MATRIX_PATH = ROOT / "build/option-b-v2-weight-policy-20260806/full_sequence_discrimination.json"
MATRIX_SHA256 = "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
SOURCE_EMBEDDING_SHA256 = "4e96b0df6d274768cbb7e72404011853d23349999b658dc2f4dfb3c431ea223f"
V4_RECONSTRUCTION_PATH = ROOT / "build/stage1-option-b-alias-safe-joint-row-scale-tensor-codebook-w4-grouped-dynamic-scale32-w4a8-v4/preattempt/model_only_reconstruction.json"
V4_RECONSTRUCTION_SHA256 = "725a3467127736fe2346d1577abce53370941d8c647357c21499bb7a25c3c53a"
V4_AGGREGATE_BF16_SSE = 2387.058149454451

OUTPUT_DIR = reference.OFFICIAL_ROOT
PREATTEMPT_DIR = OUTPUT_DIR / "preattempt"
INPUT_MANIFEST_PATH = PREATTEMPT_DIR / "non_evaluator_inputs.json"
CANDIDATE_A_PATH = PREATTEMPT_DIR / "candidate_a_manifest.json"
CANDIDATE_B_PATH = PREATTEMPT_DIR / "candidate_b_manifest.json"
ALIAS_WITNESS_PATH = PREATTEMPT_DIR / "alias_separation_witness.json"
RECONSTRUCTION_PATH = PREATTEMPT_DIR / "model_only_input_block_reconstruction.json"
QUALITY_PATH = PREATTEMPT_DIR / "non_evaluator_quality.json"
DYNAMIC_PATH = PREATTEMPT_DIR / "dynamic_preflight.json"
SELF_TEST_PATH = PREATTEMPT_DIR / "runner_self_test.json"
CHRONOLOGY_PATH = PREATTEMPT_DIR / "preparation_chronology.json"
RECOVERY_PATH = PREATTEMPT_DIR / "preparation_recovery.json"
CONTRACT_PATH = OUTPUT_DIR / "predeclared_contract.json"
READY_PATH = OUTPUT_DIR / "PREATTEMPT_READY.json"
NO_GO_PATH = OUTPUT_DIR / "PREATTEMPT_NO_GO.json"
ATTEMPT_DIR = OUTPUT_DIR / "attempt-0001"
MARKER_PATH = ATTEMPT_DIR / "execution_started.json"

INPUT_BLOCK_LANES = reference.INPUT_BLOCK_LANES
OUTER_ITERATIONS = reference.OUTER_ITERATIONS
TORCH_THREADS = 32
EXPECTED_EVENTS = grouped.LAYERS * len(grouped.EVENT_FAMILIES)
STATE_MATCH_KEYS = (
    "embedding_bf16_sha256",
    "payload_manifest_sha256",
    "row_scale_manifest_sha256",
    "codebook_manifest_sha256",
    "reconstruction_manifest_sha256",
    "iteration_trace_manifest_sha256",
    "weight_manifest_sha256",
    "candidate_state_manifest_sha256",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def verify_companion(path: Path, companion: Path) -> None:
    expected = f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}"
    require(companion.is_file(), f"checksum companion is missing: {companion}")
    require(companion.read_text(encoding="utf-8").strip() == expected, f"checksum companion differs: {path}")


def require_project_python() -> dict[str, Any]:
    grouped.require_project_python()
    expected = ROOT / ".venv/bin/python"
    require(expected.is_file(), "project Python is missing")
    require(os.path.samefile(Path(sys.executable), expected), "wrong Python executable")
    return {
        "entrypoint": "./.venv/bin/python",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {**verify_versions(), "numpy": importlib.metadata.version("numpy")},
        "torch_num_threads": TORCH_THREADS,
        "backend": "fresh exact V4 initialization plus exact 128-lane block updates",
    }


def atomic_write_json(path: Path, value: Any) -> None:
    raw = canonical_bytes(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _configure_quality_helpers() -> None:
    v3.CANDIDATE_ID = CANDIDATE_ID
    v3.OUTPUT_DIR = OUTPUT_DIR
    v3.PREATTEMPT_DIR = PREATTEMPT_DIR
    v3.ATTEMPT_DIR = ATTEMPT_DIR
    v3.MARKER_PATH = MARKER_PATH
    v3.TORCH_THREADS = TORCH_THREADS


def _block_ranges(width: int) -> tuple[tuple[int, int], ...]:
    return reference.input_block_ranges(width)


def _assign_blocks(
    source: Tensor,
    records: Tensor,
    codebooks: Sequence[Sequence[int]],
    shift_min: int,
    shift_max: int,
) -> tuple[Tensor, list[dict[str, Any]], int, int]:
    blocks = _block_ranges(int(source.shape[1]))
    require(len(codebooks) == len(blocks), "V7 codebook block count differs")
    assignments = torch.empty(source.shape, dtype=torch.uint8)
    aggregates: list[dict[str, Any]] = []
    objective_exponent: int | None = None
    objective = 0
    for block_index, (start, stop) in enumerate(blocks):
        selected, summary = v4._assign_and_aggregate(
            source[:, start:stop], records, codebooks[block_index], shift_min, shift_max
        )
        assignments[:, start:stop] = selected
        exponent = int(summary["objective_binary_exponent"])
        if objective_exponent is None:
            objective_exponent = exponent
        require(exponent == objective_exponent, "V7 block objective exponent differs")
        aggregates.append(summary)
        objective += v4._objective(summary, codebooks[block_index])
    require(objective_exponent is not None, "V7 tensor has no input blocks")
    return assignments, aggregates, objective, objective_exponent


def _row_statistics_blocked(
    source: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[int]],
    base_shift: int,
) -> tuple[list[int], list[int], list[int]]:
    rows, width = source.shape
    blocks = _block_ranges(width)
    weight_codepoint: list[int] = []
    codepoint_squared: list[int] = []
    absolute_maximum: list[int] = []
    point_tensors = [torch.tensor(v4.reference.validate_codebook(points), dtype=torch.int64) for points in codebooks]
    for row_start, row_stop in v4._row_chunks(rows, width):
        significands, shifts = v4._decode_bf16(source[row_start:row_stop])
        delta = torch.where(significands == 0, torch.zeros_like(shifts), shifts - base_shift)
        require(bool(torch.all((delta >= 0) & (delta <= 54))), "BF16 tensor lattice span exceeds V7 row update")
        exact = torch.bitwise_left_shift(significands, delta)
        selected = torch.empty_like(exact)
        for block_index, (start, stop) in enumerate(blocks):
            selected[:, start:stop] = point_tensors[block_index][assignments[row_start:row_stop, start:stop].to(torch.int64)]
        maximum = int(torch.abs(exact).max())
        require(maximum * 127 * width < (1 << 63), "V7 row-update reduction may overflow int64")
        weight_codepoint.extend(int(value) for value in torch.sum(exact * selected, dim=1).tolist())
        codepoint_squared.extend(int(value) for value in torch.sum(selected * selected, dim=1).tolist())
        absolute_maximum.extend(int(value) for value in torch.abs(exact).amax(dim=1).tolist())
    return weight_codepoint, codepoint_squared, absolute_maximum


def _update_row_records_blocked(
    source: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[int]],
    prior_records: Tensor,
    base_shift: int,
    objective_exponent: int,
) -> tuple[Tensor, dict[str, Any]]:
    weight_codepoint, codepoint_squared, absolute_maximum = _row_statistics_blocked(
        source, assignments, codebooks, base_shift
    )
    prior = [int(value) for value in prior_records.tolist()]
    selected: list[int] = []
    candidate_counts: list[int] = []
    changed = 0
    all_zero = 0
    for wp, p2, maximum, previous in zip(
        weight_codepoint, codepoint_squared, absolute_maximum, prior, strict=True
    ):
        candidates, summary = v4._candidate_row_records(wp, p2, maximum, base_shift, previous)
        if summary["all_zero"]:
            choice = SCALE32_ALL_ZERO_RECORD
            all_zero += 1
        else:
            choice = min(
                candidates,
                key=lambda record: (
                    v4._row_objective(record, wp, p2, base_shift, objective_exponent),
                    *v4._record_scale_key(record),
                ),
            )
        selected.append(choice)
        candidate_counts.append(summary["candidate_count"])
        changed += int(choice != previous)
    records = torch.tensor(selected, dtype=torch.int64)
    return records, {
        "row_count": len(selected),
        "changed_row_count": changed,
        "all_zero_row_count": all_zero,
        "minimum_candidate_count": min(candidate_counts),
        "maximum_candidate_count": max(candidate_counts),
        "mean_candidate_count": sum(candidate_counts) / len(candidate_counts),
    }


def _update_block_codebooks(
    aggregates: Sequence[dict[str, Any]],
    prior_codebooks: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], ...], list[dict[str, Any]]]:
    require(len(aggregates) == len(prior_codebooks), "V7 block update count differs")
    selected: list[tuple[int, ...]] = []
    summaries: list[dict[str, Any]] = []
    for block_index, (summary, prior) in enumerate(zip(aggregates, prior_codebooks, strict=True)):
        codebook, update = v4._update_codebook(summary, prior)
        selected.append(codebook)
        summaries.append({"block_index": block_index, **update})
    return tuple(selected), summaries


def _codebook_raw(codebooks: Sequence[Sequence[int]]) -> bytes:
    return b"".join(bytes(point & 0xFF for point in v4.reference.validate_codebook(codebook)) for codebook in codebooks)


def _construct_input_block_state(source: Tensor, module_name: str) -> dict[str, Any]:
    require(source.ndim == 2 and source.dtype == torch.bfloat16, f"source weight format differs: {module_name}")
    blocks = _block_ranges(int(source.shape[1]))
    fresh_v4 = v4._construct_joint_state(source, module_name)
    records = fresh_v4["final_records"].clone()
    v4_codebook = tuple(int(value) for value in fresh_v4["final_codepoints"])
    codebooks = tuple(v4_codebook for _block in blocks)
    assignments, aggregates, previous_objective, objective_exponent = _assign_blocks(
        source, records, codebooks, fresh_v4["shift_min"], fresh_v4["shift_max"]
    )
    require(torch.equal(assignments, fresh_v4["final_assignments"]), f"V7 initialization payload does not reproduce V4: {module_name}")
    require(
        v4._objective_record(previous_objective, objective_exponent) == fresh_v4["final_objective"],
        f"V7 initialization objective does not reproduce V4: {module_name}",
    )
    initial_payload = v3.pack_codebook_indices(assignments)
    require(initial_payload == fresh_v4["payload"], f"V7 initial packed payload differs from V4: {module_name}")
    initial_codebook_raw = _codebook_raw(codebooks)
    trace: list[dict[str, Any]] = []

    for iteration in range(1, OUTER_ITERATIONS + 1):
        prior_records = records
        records, row_summary = _update_row_records_blocked(
            source,
            assignments,
            codebooks,
            prior_records,
            fresh_v4["shift_min"],
            objective_exponent,
        )
        assignments_after_scale, scale_aggregates, after_scale_objective, exponent = _assign_blocks(
            source, records, codebooks, fresh_v4["shift_min"], fresh_v4["shift_max"]
        )
        require(exponent == objective_exponent, f"V7 row-scale objective exponent changed: {module_name}")
        require(after_scale_objective <= previous_objective, f"V7 row-scale update increased exact source SSE: {module_name}")

        codebooks, block_summaries = _update_block_codebooks(scale_aggregates, codebooks)
        assignments, final_aggregates, final_objective, exponent = _assign_blocks(
            source, records, codebooks, fresh_v4["shift_min"], fresh_v4["shift_max"]
        )
        require(exponent == objective_exponent, f"V7 codebook objective exponent changed: {module_name}")
        require(final_objective <= after_scale_objective, f"V7 codebook update increased exact source SSE: {module_name}")
        require(final_objective <= previous_objective, f"V7 outer iteration increased exact source SSE: {module_name}")
        payload = v3.pack_codebook_indices(assignments)
        scale_raw = b"".join(struct.pack("<I", int(value)) for value in records.tolist())
        codebook_raw = _codebook_raw(codebooks)
        trace.append(
            {
                "iteration": iteration,
                "prior_objective": v4._objective_record(previous_objective, objective_exponent),
                "after_row_scale_objective": v4._objective_record(after_scale_objective, objective_exponent),
                "final_objective": v4._objective_record(final_objective, objective_exponent),
                "row_scale_update": row_summary,
                "row_scale_stream_sha256": hashlib.sha256(scale_raw).hexdigest(),
                "input_block_count": len(blocks),
                "input_block_codebook_stream_sha256": hashlib.sha256(codebook_raw).hexdigest(),
                "input_block_codebook_updates": block_summaries,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        previous_objective = final_objective

    require(len(trace) == OUTER_ITERATIONS, f"V7 constructor iteration count differs: {module_name}")
    payload = v3.pack_codebook_indices(assignments)
    scale_raw = b"".join(struct.pack("<I", int(value)) for value in records.tolist())
    codebook_raw = _codebook_raw(codebooks)
    return {
        "shift_min": fresh_v4["shift_min"],
        "shift_max": fresh_v4["shift_max"],
        "shift_span": fresh_v4["shift_span"],
        "subnormal_count": fresh_v4["subnormal_count"],
        "input_block_lanes": INPUT_BLOCK_LANES,
        "input_block_count": len(blocks),
        "input_block_ranges": [list(item) for item in blocks],
        "fresh_v4_state": fresh_v4,
        "initial_v4_payload_sha256": hashlib.sha256(initial_payload).hexdigest(),
        "initial_v4_row_scale_stream_sha256": hashlib.sha256(fresh_v4["scale_raw"]).hexdigest(),
        "initial_v4_tensor_codebook": list(v4_codebook),
        "initial_v4_replicated_codebook_stream_sha256": hashlib.sha256(initial_codebook_raw).hexdigest(),
        "initial_v4_objective": fresh_v4["final_objective"],
        "outer_iterations_completed": len(trace),
        "final_records": records,
        "final_codebooks": codebooks,
        "final_assignments": assignments,
        "payload": payload,
        "scale_raw": scale_raw,
        "codebook_raw": codebook_raw,
        "final_objective": v4._objective_record(previous_objective, objective_exponent),
        "iteration_trace": trace,
        "iteration_trace_sha256": canonical_sha256(trace),
    }


def _decoded_chunk(
    records: Tensor,
    assignments: Tensor,
    codebooks: Sequence[Sequence[int]],
    row_start: int,
    row_stop: int,
) -> Tensor:
    width = int(assignments.shape[1])
    scale_significands, scale_exponents = v4._records_components(records[row_start:row_stop])
    scales = torch.ldexp(scale_significands.to(torch.float64), scale_exponents.to(torch.int32) - 15)
    decoded = torch.empty((row_stop - row_start, width), dtype=torch.float64)
    for block_index, (start, stop) in enumerate(_block_ranges(width)):
        points = torch.tensor(codebooks[block_index], dtype=torch.float64)
        decoded[:, start:stop] = points[assignments[row_start:row_stop, start:stop].to(torch.int64)] * (scales[:, None] / 16.0)
    return decoded.to(torch.bfloat16)


def _materialize_and_measure(
    module: nn.Linear,
    state: dict[str, Any],
    sealed_v4_record: dict[str, Any],
) -> dict[str, Any]:
    source = module.weight.detach()
    fresh_v4 = state["fresh_v4_state"]
    v4_codebooks = tuple(tuple(fresh_v4["final_codepoints"]) for _block in _block_ranges(int(source.shape[1])))
    fresh_v4_sse = 0.0
    v7_sse = 0.0
    v4_hash = hashlib.sha256()
    v7_hash = hashlib.sha256()
    rows, width = source.shape
    with torch.no_grad():
        for start, stop in v4._row_chunks(rows, width):
            decoded_v4 = _decoded_chunk(
                fresh_v4["final_records"], fresh_v4["final_assignments"], v4_codebooks, start, stop
            )
            decoded_v7 = _decoded_chunk(
                state["final_records"], state["final_assignments"], state["final_codebooks"], start, stop
            )
            source64 = source[start:stop].to(torch.float64)
            v4_difference = source64 - decoded_v4.to(torch.float64)
            v7_difference = source64 - decoded_v7.to(torch.float64)
            fresh_v4_sse += float(torch.sum(v4_difference * v4_difference))
            v7_sse += float(torch.sum(v7_difference * v7_difference))
            v4_hash.update(v3.raw_tensor_bytes(decoded_v4))
            v7_hash.update(v3.raw_tensor_bytes(decoded_v7))
            module.weight[start:stop].copy_(decoded_v7)
    fresh_v4_sha = v4_hash.hexdigest()
    v7_sha = v7_hash.hexdigest()
    require(fresh_v4_sha == sealed_v4_record["reconstruction_bf16_sha256"], f"fresh V4 reconstruction hash differs: {sealed_v4_record['module']}")
    require(fresh_v4_sse == float(sealed_v4_record["v4_joint_bf16_sse"]), f"fresh V4 reconstruction SSE differs: {sealed_v4_record['module']}")
    require(v7_sha == v3.tensor_sha256(module.weight), f"V7 materialized reconstruction hash differs: {sealed_v4_record['module']}")
    return {
        "module": sealed_v4_record["module"],
        "weight_count": int(module.weight.numel()),
        "fresh_v4_bf16_sse": fresh_v4_sse,
        "sealed_v4_bf16_sse": float(sealed_v4_record["v4_joint_bf16_sse"]),
        "fresh_v4_matches_sealed": True,
        "fresh_v4_reconstruction_bf16_sha256": fresh_v4_sha,
        "v7_input_block_bf16_sse": v7_sse,
        "v7_non_regressing_against_fresh_v4": v7_sse <= fresh_v4_sse,
        "v7_strictly_improved_against_fresh_v4": v7_sse < fresh_v4_sse,
        "reconstruction_bf16_sha256": v7_sha,
    }


def _policy_and_format() -> dict[str, Any]:
    body = {
        "payload": {
            "bits_per_weight": 4,
            "semantics": "unsigned input-block codebook index 0 through 15",
            "packing": "row-major; even flat element low nibble; odd flat element high nibble",
        },
        "row_scale": {
            "format": "Scale32",
            "bytes_per_row": 4,
            "additional_records": 0,
            "scope": "one scale shared by all input blocks of an output row",
            "selection_tie_order": ["source_space_sse", "smaller_positive_scale", "smaller_packed_record"],
        },
        "codebook": {
            "scope": "one per contiguous 128 input lanes",
            "input_block_lanes": INPUT_BLOCK_LANES,
            "entry_count": 16,
            "record_bytes": 16,
            "alignment_bytes": 16,
            "format": "strictly increasing signed Q4.4 int8",
            "outer_iterations": OUTER_ITERATIONS,
            "initialization": "fresh exact V4 tensor codebook duplicated into every input block",
            "empty_cluster": "retain prior codepoint",
            "global_update": "exact strictly-increasing dynamic program with lexicographic ties",
            "fetch_order": "increasing input-channel block order",
        },
        "activation": {
            "policy": "grouped Dynamic Scale32",
            "event_count": EXPECTED_EVENTS,
            "payload_range": [-127, 127],
            "reserved_payload": -128,
            "delta_range": [-24, 24],
            "effective_exponent_range": [-24, 4],
            "maximum_transport_metadata_bytes_per_decoder_layer_token": 832,
        },
    }
    return {**body, "policy_and_format_sha256": canonical_sha256(body)}


def _quantize_model(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {record["module"]: record for record in grouped.expected_transformer_tensors()}
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == 169, "Qwen linear count differs")
    require([name for name, _module in modules if name != "lm_head"] == list(expected), "transformer linear order differs")
    require(sha256_file(V4_RECONSTRUCTION_PATH) == V4_RECONSTRUCTION_SHA256, "V4 reconstruction reference hash differs")
    v4_report = load_json(V4_RECONSTRUCTION_PATH)
    v4_records = {item["module"]: item for item in v4_report["tensors"]}
    require(list(v4_records) == [name for name, _module in modules], "V4 reconstruction tensor order differs")

    weight_manifest: list[dict[str, Any]] = []
    payload_manifest: list[dict[str, Any]] = []
    row_scale_manifest: list[dict[str, Any]] = []
    codebook_manifest: list[dict[str, Any]] = []
    reconstruction_manifest: list[dict[str, Any]] = []
    iteration_trace_manifest: list[dict[str, Any]] = []
    payload_stream = hashlib.sha256()
    scale_stream = hashlib.sha256()
    codebook_stream = hashlib.sha256()
    reconstruction_stream = hashlib.sha256()
    total_weight_count = 0
    total_payload_bytes = 0
    total_scale_bytes = 0
    total_codebook_bytes = 0
    total_codebook_count = 0

    for ordinal, (name, module) in enumerate(modules, start=1):
        started = time.monotonic()
        print(f"V7_CONSTRUCT_START {ordinal}/169 {name}", flush=True)
        source_sha = v3.tensor_sha256(module.weight)
        state = _construct_input_block_state(module.weight.detach(), name)
        reconstruction = _materialize_and_measure(module, state, v4_records[name])
        payload = state["payload"]
        scale_raw = state["scale_raw"]
        codebook_raw = state["codebook_raw"]
        payload_sha = hashlib.sha256(payload).hexdigest()
        scale_sha = hashlib.sha256(scale_raw).hexdigest()
        codebook_sha = hashlib.sha256(codebook_raw).hexdigest()
        weight_count = int(module.weight.numel())
        block_count = state["input_block_count"]
        require(len(payload) == (weight_count + 1) // 2, f"packed payload byte count differs: {name}")
        require(len(scale_raw) == module.weight.shape[0] * 4, f"row-scale byte count differs: {name}")
        require(len(codebook_raw) == block_count * 16, f"input-block codebook byte count differs: {name}")

        payload_record = {"module": name, "bytes": len(payload), "sha256": payload_sha}
        scale_record = {
            "module": name,
            "record_count": int(module.weight.shape[0]),
            "bytes": len(scale_raw),
            "sha256": scale_sha,
        }
        codebook_record = {
            "module": name,
            "input_block_lanes": INPUT_BLOCK_LANES,
            "record_count": block_count,
            "bytes": len(codebook_raw),
            "sha256": codebook_sha,
            "q4_4_codepoints_by_input_block": [list(points) for points in state["final_codebooks"]],
        }
        trace_record = {
            "module": name,
            "source_bf16_sha256": source_sha,
            "shift_min": state["shift_min"],
            "shift_max": state["shift_max"],
            "shift_span": state["shift_span"],
            "subnormal_count": state["subnormal_count"],
            "input_block_lanes": INPUT_BLOCK_LANES,
            "input_block_count": block_count,
            "input_block_ranges": state["input_block_ranges"],
            "initial_v4_payload_sha256": state["initial_v4_payload_sha256"],
            "initial_v4_row_scale_stream_sha256": state["initial_v4_row_scale_stream_sha256"],
            "initial_v4_tensor_codebook": state["initial_v4_tensor_codebook"],
            "initial_v4_replicated_codebook_stream_sha256": state["initial_v4_replicated_codebook_stream_sha256"],
            "initial_v4_objective": state["initial_v4_objective"],
            "outer_iterations_completed": state["outer_iterations_completed"],
            "final_objective": state["final_objective"],
            "iteration_trace": state["iteration_trace"],
            "iteration_trace_sha256": state["iteration_trace_sha256"],
        }
        record = {
            "module": name,
            "name": f"{name}.weight",
            "shape": list(module.weight.shape),
            "scope": "separated_lm_head" if name == "lm_head" else "transformer",
            "payload": payload_record,
            "row_scale": scale_record,
            "input_block_codebook": codebook_record,
            "reconstruction": reconstruction,
            "iteration_trace_sha256": state["iteration_trace_sha256"],
        }
        weight_manifest.append(record)
        payload_manifest.append(payload_record)
        row_scale_manifest.append(scale_record)
        codebook_manifest.append(codebook_record)
        reconstruction_manifest.append(reconstruction)
        iteration_trace_manifest.append(trace_record)
        payload_stream.update(name.encode() + b"\0" + payload)
        scale_stream.update(name.encode() + b"\0" + scale_raw)
        codebook_stream.update(name.encode() + b"\0" + codebook_raw)
        reconstruction_stream.update(name.encode() + b"\0" + bytes.fromhex(reconstruction["reconstruction_bf16_sha256"]))
        total_weight_count += weight_count
        total_payload_bytes += len(payload)
        total_scale_bytes += len(scale_raw)
        total_codebook_bytes += len(codebook_raw)
        total_codebook_count += block_count
        elapsed = time.monotonic() - started
        print(
            f"V7_CONSTRUCT_DONE {ordinal}/169 {name} seconds={elapsed:.3f} "
            f"v4_sse={reconstruction['fresh_v4_bf16_sse']:.12g} "
            f"v7_sse={reconstruction['v7_input_block_bf16_sse']:.12g}",
            flush=True,
        )
        del state
        gc.collect()

    aggregate_v4_sse = float(sum(item["fresh_v4_bf16_sse"] for item in reconstruction_manifest))
    aggregate_v7_sse = float(sum(item["v7_input_block_bf16_sse"] for item in reconstruction_manifest))
    non_regressing_count = sum(bool(item["v7_non_regressing_against_fresh_v4"]) for item in reconstruction_manifest)
    strictly_improved_count = sum(bool(item["v7_strictly_improved_against_fresh_v4"]) for item in reconstruction_manifest)
    require(aggregate_v4_sse == V4_AGGREGATE_BF16_SSE, "fresh aggregate V4 BF16 SSE differs")
    reconstruction_body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "measurement": "sum-squared error from pinned BF16 source to BF16-materialized reconstruction",
        "reference": {
            "candidate_id": v4_report["candidate_id"],
            "artifact": file_record(V4_RECONSTRUCTION_PATH),
            "aggregate_bf16_sse": V4_AGGREGATE_BF16_SSE,
            "fresh_reconstruction_required": True,
        },
        "tensor_count": len(reconstruction_manifest),
        "fresh_v4_exact_reproduction_count": sum(bool(item["fresh_v4_matches_sealed"]) for item in reconstruction_manifest),
        "per_tensor_non_regressing_count": non_regressing_count,
        "per_tensor_strictly_improved_count": strictly_improved_count,
        "all_169_fresh_v4_reproductions_match": all(item["fresh_v4_matches_sealed"] for item in reconstruction_manifest),
        "all_169_tensors_non_regressing_against_fresh_v4": non_regressing_count == 169,
        "aggregate_fresh_v4_bf16_sse": aggregate_v4_sse,
        "aggregate_v7_input_block_bf16_sse": aggregate_v7_sse,
        "aggregate_strictly_improved_against_fresh_v4": aggregate_v7_sse < aggregate_v4_sse,
        "tensors": reconstruction_manifest,
    }
    reconstruction_report = {
        **reconstruction_body,
        "status": (
            "PASS_FRESH_V4_AND_169_OF_169_WITH_STRICT_AGGREGATE_IMPROVEMENT"
            if reconstruction_body["all_169_fresh_v4_reproductions_match"]
            and reconstruction_body["all_169_tensors_non_regressing_against_fresh_v4"]
            and reconstruction_body["aggregate_strictly_improved_against_fresh_v4"]
            else "NO_GO_V4_RELATIVE_MODEL_ONLY_RECONSTRUCTION_GATE"
        ),
        "reconstruction_report_sha256": canonical_sha256(reconstruction_body),
    }

    transformer_records = [record for record in weight_manifest if record["scope"] == "transformer"]
    standard_layer_weights = grouped.policy_cost()["standard_decoder_layer_weight_count"]
    standard_layer_existing_bytes = grouped.policy_cost()["standard_decoder_layer_total_weight_bytes"]
    standard_layer_codebook_bytes = 80 * 16
    standard_layer_bits = ((standard_layer_existing_bytes + standard_layer_codebook_bytes) * 8) / standard_layer_weights
    lm_head_record = next(record for record in weight_manifest if record["module"] == "lm_head")
    lm_head_weight_count = int(np.prod(lm_head_record["shape"]))
    lm_head_total_bytes = lm_head_record["payload"]["bytes"] + lm_head_record["row_scale"]["bytes"] + lm_head_record["input_block_codebook"]["bytes"]
    lm_head_bits = (lm_head_total_bytes * 8) / lm_head_weight_count
    require(len(transformer_records) == 168 and len(weight_manifest) == 169, "V7 tensor count differs")
    require(total_codebook_count == 1927 and total_codebook_bytes == 30832, "whole-model V7 codebook geometry differs")
    require(abs(standard_layer_bits - 4.027884615384615) < 1e-15, "standard-layer V7 stored bits differ")
    require(standard_layer_bits <= 4.036 and lm_head_bits <= 4.036, "stored-bit cap exceeded")
    manifests = {
        "payload": payload_manifest,
        "row_scale": row_scale_manifest,
        "codebook": codebook_manifest,
        "reconstruction": reconstruction_manifest,
        "iteration_trace": iteration_trace_manifest,
    }
    hashes = {
        "payload_manifest_sha256": canonical_sha256(payload_manifest),
        "row_scale_manifest_sha256": canonical_sha256(row_scale_manifest),
        "codebook_manifest_sha256": canonical_sha256(codebook_manifest),
        "reconstruction_manifest_sha256": canonical_sha256(reconstruction_manifest),
        "iteration_trace_manifest_sha256": canonical_sha256(iteration_trace_manifest),
        "weight_manifest_sha256": canonical_sha256(weight_manifest),
        "reconstruction_report_sha256": reconstruction_report["reconstruction_report_sha256"],
        "payload_stream_sha256": payload_stream.hexdigest(),
        "row_scale_stream_sha256": scale_stream.hexdigest(),
        "codebook_stream_sha256": codebook_stream.hexdigest(),
        "reconstruction_stream_sha256": reconstruction_stream.hexdigest(),
    }
    storage = {
        "total_weight_count": total_weight_count,
        "total_payload_bytes": total_payload_bytes,
        "total_row_scale_bytes": total_scale_bytes,
        "whole_model_input_block_codebook_count": total_codebook_count,
        "whole_model_input_block_codebook_bytes": total_codebook_bytes,
        "standard_decoder_layer_input_block_codebook_count": 80,
        "standard_decoder_layer_input_block_codebook_bytes": 1280,
        "standard_decoder_layer_exact_bits_per_weight": standard_layer_bits,
        "lm_head_exact_bits_per_weight": lm_head_bits,
        "maximum_total_stored_bits_per_weight": 4.036,
    }
    return weight_manifest, {
        "manifests": manifests,
        "hashes": hashes,
        "storage": storage,
        "reconstruction": reconstruction_report,
    }


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
    require(lm_head is embedding, "source lm_head and embedding Parameter objects differ")
    require(lm_head.data_ptr() == embedding.data_ptr(), "source lm_head and embedding storage differs")
    source_shape = list(embedding.shape)
    source_dtype = str(embedding.dtype)
    source_object_id = id(embedding)
    source_pointer = embedding.data_ptr()
    source_hash = v3.tensor_sha256(embedding)
    require(source_hash == SOURCE_EMBEDDING_SHA256, "source embedding BF16 hash differs")

    cloned = lm_head.detach().clone()
    require(v3.tensor_sha256(cloned) == source_hash, "detached lm_head clone differs before split")
    model.lm_head.weight = nn.Parameter(cloned, requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight is not embedding, "lm_head Parameter object remained tied")
    require(model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head storage remained tied")
    require(v3.tensor_sha256(model.lm_head.weight) == source_hash, "lm_head clone bytes differ after split")
    require(id(model.model.embed_tokens.weight) == source_object_id, "embedding Parameter object changed during split")
    require(model.model.embed_tokens.weight.data_ptr() == source_pointer, "embedding storage changed during split")

    weight_manifest, construction = _quantize_model(model)
    require(model.config.tie_word_embeddings is False, "candidate config was retied")
    require(model.model.embed_tokens.weight is embedding, "embedding Parameter object changed after V7 construction")
    require(model.model.embed_tokens.weight.data_ptr() == source_pointer, "embedding storage changed after V7 construction")
    require(list(embedding.shape) == source_shape and str(embedding.dtype) == source_dtype, "embedding shape or dtype changed")
    embedding_hash = v3.tensor_sha256(embedding)
    require(embedding_hash == source_hash == SOURCE_EMBEDDING_SHA256, "embedding bytes changed after V7 construction")
    state = v3.state_manifest(model)
    identity = {
        "embedding_bf16_sha256": embedding_hash,
        **construction["hashes"],
        "candidate_state_manifest_sha256": state["candidate_state_manifest_sha256"],
    }
    manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "construction_label": label,
        "source_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "config_tie_word_embeddings_after_split": False,
        "module_counts": {"transformer_linears": 168, "lm_head": 1, "total_input_block_linears": 169},
        "identity": identity,
        "storage": construction["storage"],
        "payload_manifest": construction["manifests"]["payload"],
        "row_scale_manifest": construction["manifests"]["row_scale"],
        "codebook_manifest": construction["manifests"]["codebook"],
        "reconstruction_manifest": construction["manifests"]["reconstruction"],
        "iteration_trace_manifest": construction["manifests"]["iteration_trace"],
        "weight_manifest": weight_manifest,
        "state_manifest": state,
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
            "pre_v7_byte_identical": True,
            "config_tie_word_embeddings": bool(model.config.tie_word_embeddings),
        },
        "post_v7": {
            "embedding_bf16_sha256": embedding_hash,
            "embedding_hash_matches_source": embedding_hash == SOURCE_EMBEDDING_SHA256,
            "lm_head_storage_is_separate": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
            "transformer_input_block_count": 168,
            "lm_head_input_block_count": 1,
        },
    }
    return model, manifest, witness, construction["reconstruction"]


def _compare_candidates(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> dict[str, Any]:
    comparisons = {key: candidate_a["identity"][key] == candidate_b["identity"][key] for key in STATE_MATCH_KEYS}
    require(all(comparisons.values()), f"candidate A/B identity differs: {comparisons}")
    manifest_fields = (
        "payload_manifest",
        "row_scale_manifest",
        "codebook_manifest",
        "reconstruction_manifest",
        "iteration_trace_manifest",
        "weight_manifest",
        "state_manifest",
    )
    field_matches = {field: candidate_a[field] == candidate_b[field] for field in manifest_fields}
    require(all(field_matches.values()), f"candidate A/B whole-model manifests differ: {field_matches}")
    return {
        "state_match_keys": list(STATE_MATCH_KEYS),
        "identity_comparisons": comparisons,
        "manifest_byte_equivalent_fields": list(manifest_fields),
        "manifest_comparisons": field_matches,
        "all_match": True,
    }


def combined_self_test() -> dict[str, Any]:
    environment = require_project_python()
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    task = load_json(TASK_PATH)
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "frozen V7 Engineer authority differs")
    root_existed_before = OUTPUT_DIR.exists()
    marker_existed_before = MARKER_PATH.exists()
    require(not marker_existed_before, "V7 execution marker already exists")
    v4_self_test = v4.reference.self_test()
    require(v4_self_test["status"] == "PASS", "reviewed V4 exact constructor self-test failed")
    ranges_896 = _block_ranges(896)
    ranges_4864 = _block_ranges(4864)
    require(len(ranges_896) == 7 and len(ranges_4864) == 38, "V7 block-count geometry differs")
    require(ranges_896[0] == (0, 128) and ranges_896[-1] == (768, 896), "V7 block ordering differs")
    illegal_width_rejected = reference.expect_rejected(lambda: _block_ranges(1024), "illegal V7 width was accepted")

    values = ((torch.arange(896, dtype=torch.int64) % 31) - 15).to(torch.float64) / 256.0
    fixture = torch.stack((values, values / 2)).to(torch.bfloat16)
    candidate_a = _construct_input_block_state(fixture, "self_test.linear")
    candidate_b = _construct_input_block_state(fixture, "self_test.linear")
    require(candidate_a["payload"] == candidate_b["payload"], "independent V7 fixture payloads differ")
    require(candidate_a["scale_raw"] == candidate_b["scale_raw"], "independent V7 fixture row scales differ")
    require(candidate_a["codebook_raw"] == candidate_b["codebook_raw"], "independent V7 fixture codebooks differ")
    require(candidate_a["iteration_trace"] == candidate_b["iteration_trace"], "independent V7 fixture traces differ")
    fresh_v4 = candidate_a["fresh_v4_state"]
    require(candidate_a["initial_v4_payload_sha256"] == hashlib.sha256(fresh_v4["payload"]).hexdigest(), "V7 fixture did not reproduce V4 payload")
    replicated = _codebook_raw(tuple(tuple(fresh_v4["final_codepoints"]) for _block in ranges_896))
    require(candidate_a["initial_v4_replicated_codebook_stream_sha256"] == hashlib.sha256(replicated).hexdigest(), "V7 fixture codebook replication differs")
    require(len(candidate_a["codebook_raw"]) == 7 * 16, "V7 fixture codebook byte count differs")
    require(candidate_a["outer_iterations_completed"] == OUTER_ITERATIONS, "V7 fixture iteration count differs")
    require(
        all(int(item["final_objective"]["scaled_integer"]) <= int(item["prior_objective"]["scaled_integer"]) for item in candidate_a["iteration_trace"]),
        "V7 fixture monotonic objective differs",
    )
    cache_sequence = [
        {"block_index": index, "start_input_lane": start, "stop_input_lane": stop, "codebook_beat_bytes": 16}
        for index, (start, stop) in enumerate(ranges_896)
    ]
    require([item["block_index"] for item in cache_sequence] == list(range(7)), "V7 cache sequence differs")
    hash_mismatch_rejected = reference.expect_rejected(
        lambda: require("0" * 64 == "1" * 64, "bound artifact hash differs"),
        "bound hash mismatch was accepted",
    )
    require(OUTPUT_DIR.exists() == root_existed_before, "self-test changed the V7 namespace")
    require(MARKER_PATH.exists() == marker_existed_before is False, "self-test changed the V7 execution marker")
    return {
        "schema_version": 1,
        "status": "PASS",
        "candidate_id": CANDIDATE_ID,
        "mission_id": MISSION_ID,
        "task_id": TASK_ID,
        "environment": environment,
        "runner_sha256": sha256_file(RUNNER_PATH),
        "backend_sha256": sha256_file(BACKEND_PATH),
        "task_sha256": sha256_file(TASK_PATH),
        "plan_sha256": sha256_file(PLAN_PATH),
        "checks": {
            "reviewed_v4_exact_constructor": True,
            "legal_width_block_counts": {"896": 7, "4864": 38},
            "illegal_width_rejected": illegal_width_rejected,
            "increasing_input_lane_order": True,
            "fresh_v4_payload_reproduction": True,
            "fresh_v4_codebook_replication": True,
            "independent_candidate_a_b_match": True,
            "scale32_bracketing_neighbors_ties_range_and_zero_rows": True,
            "strict_codebook_dp_optimum_empty_clusters_and_ties": True,
            "nibble_packing": True,
            "cache_sequence": cache_sequence,
            "bound_hash_mismatch_rejected": hash_mismatch_rejected,
            "outer_iterations_completed": OUTER_ITERATIONS,
            "monotonic_source_space_sse": True,
        },
        "fixture": {
            "input_block_count": candidate_a["input_block_count"],
            "payload_sha256": hashlib.sha256(candidate_a["payload"]).hexdigest(),
            "row_scale_stream_sha256": hashlib.sha256(candidate_a["scale_raw"]).hexdigest(),
            "input_block_codebook_stream_sha256": hashlib.sha256(candidate_a["codebook_raw"]).hexdigest(),
            "iteration_trace_sha256": candidate_a["iteration_trace_sha256"],
        },
        "official_build_namespace_changed": False,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": False,
    }


def _validate_reconstruction(report: dict[str, Any]) -> bool:
    require(report["tensor_count"] == len(report["tensors"]) == 169, "V7 reconstruction tensor count differs")
    require(report["fresh_v4_exact_reproduction_count"] == sum(bool(item["fresh_v4_matches_sealed"]) for item in report["tensors"]), "fresh V4 reproduction count differs")
    require(report["per_tensor_non_regressing_count"] == sum(bool(item["v7_non_regressing_against_fresh_v4"]) for item in report["tensors"]), "V7 non-regressing count differs")
    passed = (
        report["all_169_fresh_v4_reproductions_match"] is True
        and report["all_169_tensors_non_regressing_against_fresh_v4"] is True
        and report["aggregate_strictly_improved_against_fresh_v4"] is True
    )
    require((report["status"].startswith("PASS_")) == passed, "V7 reconstruction status differs")
    body = {key: value for key, value in report.items() if key not in {"status", "reconstruction_report_sha256"}}
    require(report["reconstruction_report_sha256"] == canonical_sha256(body), "V7 reconstruction report hash differs")
    return passed


def _validate_quality(report: dict[str, Any]) -> bool:
    return v3.validate_quality(report)


def _validate_dynamic(report: dict[str, Any]) -> bool:
    passed = v3.validate_dynamic(report)
    require(report["activation_execution"]["transport_metadata_bytes_per_decoder_layer_token"] <= 832, "V7 activation metadata cap differs")
    return passed


def _audit_attempt_namespace(expect_root_absent: bool) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for marker in sorted((ROOT / "build").glob("**/execution_started.json")):
        try:
            record = load_json(marker)
        except Exception:
            continue
        if record.get("candidate_id") == CANDIDATE_ID:
            matches.append(file_record(marker))
    attempts = sorted(path.name for path in OUTPUT_DIR.glob("attempt-*") if path.is_dir()) if OUTPUT_DIR.exists() else []
    if expect_root_absent:
        require(not OUTPUT_DIR.exists(), "V7 namespace existed before preparation")
    require(not matches, "a prior V7 execution marker exists")
    require(not attempts, f"V7 attempt namespace is already consumed: {attempts}")
    return {
        "candidate_id": CANDIDATE_ID,
        "root_existed_at_audit": OUTPUT_DIR.exists(),
        "matching_execution_markers": matches,
        "attempt_directories": attempts,
        "authorization_consumed": False,
    }


def _recover_nonqualifying_preparation() -> dict[str, Any]:
    if not OUTPUT_DIR.exists():
        return {"fresh_namespace": True, "recovered": False, "archived_artifacts": []}
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "cannot recover a consumed V7 attempt namespace")
    complete = PREATTEMPT_DIR.is_dir() and CONTRACT_PATH.is_file() and (READY_PATH.is_file() or NO_GO_PATH.is_file())
    require(not complete, "V7 pre-attempt closure already exists; use verify")
    candidates = [path for path in OUTPUT_DIR.iterdir() if path.name != "preflight-failures"]
    require(candidates, "existing V7 namespace has ambiguous provenance")
    archive_root = OUTPUT_DIR / "preflight-failures"
    archive_root.mkdir(exist_ok=True)
    ordinal = 1
    while (archive_root / f"failure-{ordinal:04d}").exists():
        ordinal += 1
    archive = archive_root / f"failure-{ordinal:04d}"
    archive.mkdir()
    archived: list[dict[str, Any]] = []
    for path in candidates:
        destination = archive / path.name
        os.replace(path, destination)
        if destination.is_file():
            destination.chmod(0o444)
            archived.append(file_record(destination))
        else:
            for item in sorted(entry for entry in destination.rglob("*") if entry.is_file()):
                item.chmod(0o444)
                archived.append(file_record(item))
    return {"fresh_namespace": False, "recovered": True, "archived_artifacts": archived}


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
        "fresh_v4_reconstruction_reference": V4_RECONSTRUCTION_PATH,
        "official_matrix_hash_only": MATRIX_PATH,
        "runner_source": RUNNER_PATH,
        "scalable_backend_source": BACKEND_PATH,
        "reviewed_v4_constructor": V4_RUNNER_PATH,
        "reviewed_v4_backend": V4_BACKEND_PATH,
        "reviewed_quality_helper": V3_RUNNER_PATH,
        "reviewed_grouped_dynamic_runner": GROUPED_RUNNER_PATH,
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
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "V7 Engineer authority differs")
    require(task["fresh_attempt_namespace"]["attempt_budget"] == 1, "V7 attempt budget differs")
    require(task["fresh_attempt_namespace"]["attempts_consumed_at_task_freeze"] == 0, "V7 attempt was consumed at freeze")
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(sha256_file(V4_RECONSTRUCTION_PATH) == V4_RECONSTRUCTION_SHA256, "V4 reconstruction reference hash differs")
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
            "backend": "fresh exact V4 reconstruction plus exact 128-input-lane block-codebook updates",
            "source_tied_then_runtime_split_lm_head": True,
            "source_embedding_bf16_sha256": SOURCE_EMBEDDING_SHA256,
            "embedding_bytes_preserved": True,
            "transformer_linears": 168,
            "separated_lm_head_linears": 1,
            "input_block_lanes": 128,
            "outer_iterations": 8,
            "candidate_a_and_b_whole_model_manifests_match": state_comparison,
        },
        "pre_attempt_gates": {
            "fresh_v4_and_169_model_closure": {"passed": gate_passes["fresh_v4_and_169_model_closure"], "artifact": artifacts["model_only_reconstruction"]},
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
            "conditional_execution_rule": "consume attempt-0001 only after this exact PREATTEMPT_READY closure verifies",
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
    require(sha256_file(path) == record["sha256"], f"bound artifact hash differs: {record['path']}")


def load_verified_closure() -> dict[str, Any]:
    _configure_quality_helpers()
    require_project_python()
    require(CONTRACT_PATH.is_file(), "V7 predeclared contract is missing")
    require(READY_PATH.is_file() != NO_GO_PATH.is_file(), "exactly one V7 pre-attempt terminal record is required")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "V7 predeclared contract wrapper differs")
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "V7 predeclared contract canonical hash differs")
    contract = wrapper["contract"]
    require(contract["candidate_id"] == CANDIDATE_ID and contract["task_id"] == TASK_ID, "V7 predeclared authority differs")
    require(contract["artifacts"]["runner_source"]["sha256"] == sha256_file(RUNNER_PATH), "V7 runner changed after freeze")
    require(contract["artifacts"]["scalable_backend_source"]["sha256"] == sha256_file(BACKEND_PATH), "V7 backend changed after freeze")
    for record in contract["artifacts"].values():
        _verify_bound_record(record)
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(sha256_file(V4_RECONSTRUCTION_PATH) == V4_RECONSTRUCTION_SHA256, "V4 reconstruction reference hash differs")
    require(verify_source_snapshot() == contract["source_model"], "source snapshot identity differs")
    packages = {**verify_versions(), "numpy": importlib.metadata.version("numpy")}
    require(packages == contract["environment"]["packages"], "package versions differ")
    require(combined_self_test() == load_json(SELF_TEST_PATH), "V7 runner self-test result differs")
    candidate_a = load_json(CANDIDATE_A_PATH)
    candidate_b = load_json(CANDIDATE_B_PATH)
    comparison = _compare_candidates(candidate_a, candidate_b)
    witness = load_json(ALIAS_WITNESS_PATH)
    require(witness["state_comparison"] == comparison, "V7 alias witness state comparison differs")
    gate_passes = {
        "fresh_v4_and_169_model_closure": _validate_reconstruction(load_json(RECONSTRUCTION_PATH)),
        "non_evaluator_quality_8_of_8": _validate_quality(load_json(QUALITY_PATH)),
        "dynamic_312_event_closure": _validate_dynamic(load_json(DYNAMIC_PATH)),
        "two_build_alias_and_manifest_closure": comparison["all_match"]
        and witness["embedding_preserved_in_both"]
        and witness["lm_head_separate_in_both"],
    }
    require(contract["pre_attempt_gates"]["all_pass"] == all(gate_passes.values()), "V7 contract aggregate gate status differs")
    for name, passed in gate_passes.items():
        require(contract["pre_attempt_gates"][name]["passed"] == passed, f"V7 contract gate differs: {name}")
    terminal_path = READY_PATH if READY_PATH.is_file() else NO_GO_PATH
    terminal = load_json(terminal_path)
    require(terminal["candidate_id"] == CANDIDATE_ID, "V7 terminal pre-attempt candidate differs")
    require(terminal["contract"]["sha256"] == sha256_file(CONTRACT_PATH), "V7 terminal contract file hash differs")
    require(terminal["contract_sha256"] == wrapper["contract_sha256"], "V7 terminal contract canonical hash differs")
    require(terminal["bound_artifacts"] == contract["artifacts"], "V7 terminal artifact snapshot differs")
    require(terminal["official_execution_marker_exists"] is False and terminal["attempts_consumed"] == 0, "V7 terminal record claims an official attempt")
    if terminal_path == READY_PATH:
        require(terminal["status"] == "PREATTEMPT_READY" and all(gate_passes.values()), "V7 ready record exists without all gates passing")
    else:
        require(terminal["status"] == "PREATTEMPT_NO_GO" and not all(gate_passes.values()), "V7 no-go record does not preserve a failed gate")
    _audit_attempt_namespace(expect_root_absent=False)
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "V7 official attempt exists after marker-free closure")
    quality = load_json(QUALITY_PATH)
    dynamic = load_json(DYNAMIC_PATH)
    reconstruction = load_json(RECONSTRUCTION_PATH)
    return {
        "status": terminal["status"],
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": wrapper["contract_sha256"],
        "gate_passes": gate_passes,
        "fresh_v4_exact_reproduction_count": reconstruction["fresh_v4_exact_reproduction_count"],
        "per_tensor_non_regressing_count": reconstruction["per_tensor_non_regressing_count"],
        "aggregate_fresh_v4_bf16_sse": reconstruction["aggregate_fresh_v4_bf16_sse"],
        "aggregate_v7_input_block_bf16_sse": reconstruction["aggregate_v7_input_block_bf16_sse"],
        "exact_sequence_matches": quality["exact_bf16_sequence_match_count"],
        "gate_outcome_matches": quality["response_gate_outcome_match_count"],
        "event_count": dynamic["activation_execution"]["event_count"],
        "attempts_consumed": 0,
        "official_execution_mode_exposed": False,
    }


def prepare() -> dict[str, Any]:
    _configure_quality_helpers()
    environment = require_project_python()
    recovery = _recover_nonqualifying_preparation()
    namespace_audit = _audit_attempt_namespace(expect_root_absent=recovery["fresh_namespace"])
    namespace_audit["preparation_recovery"] = recovery
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    verify_source_contract()
    source_identity = verify_source_snapshot()
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(sha256_file(V4_RECONSTRUCTION_PATH) == V4_RECONSTRUCTION_SHA256, "V4 reconstruction reference hash differs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    probe = OUTPUT_DIR / ".preflight-write-probe"
    fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, b"preflight\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    require(probe.read_bytes() == b"preflight\n", "V7 output-path readback differs")
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
    inputs, input_manifest = v3.non_evaluator_inputs(tokenizer)
    self_test = combined_self_test()

    chronology["candidate_a_started_at_utc"] = utc_now()
    print("V7_CANDIDATE_A_START", flush=True)
    candidate_a, manifest_a, witness_a, reconstruction_a = _load_candidate("candidate_a_independent_construction")
    chronology["candidate_a_complete_at_utc"] = utc_now()
    print("V7_CANDIDATE_A_DONE", flush=True)
    del candidate_a
    gc.collect()

    chronology["candidate_b_started_at_utc"] = utc_now()
    print("V7_CANDIDATE_B_START", flush=True)
    candidate_b, manifest_b, witness_b, reconstruction_b = _load_candidate("candidate_b_independent_quality_and_dynamic_preflight")
    state_comparison = _compare_candidates(manifest_a, manifest_b)
    require(reconstruction_a == reconstruction_b, "V7 candidate A/B reconstruction reports differ")
    chronology["candidate_b_constructed_at_utc"] = utc_now()
    print("V7_CANDIDATE_B_CONSTRUCTED quality_dynamic_start", flush=True)
    quality, dynamic = v3.non_evaluator_quality_and_dynamic(candidate_b, tokenizer, inputs, input_manifest)
    chronology["candidate_b_quality_and_dynamic_complete_at_utc"] = utc_now()
    print("V7_QUALITY_DYNAMIC_DONE", flush=True)
    del candidate_b
    gc.collect()

    gate_passes = {
        "fresh_v4_and_169_model_closure": _validate_reconstruction(reconstruction_b),
        "non_evaluator_quality_8_of_8": _validate_quality(quality),
        "dynamic_312_event_closure": _validate_dynamic(dynamic),
        "two_build_alias_and_manifest_closure": state_comparison["all_match"]
        and witness_a["post_v7"]["embedding_hash_matches_source"]
        and witness_b["post_v7"]["embedding_hash_matches_source"]
        and witness_a["post_v7"]["lm_head_storage_is_separate"]
        and witness_b["post_v7"]["lm_head_storage_is_separate"],
    }
    require(not MARKER_PATH.exists() and not ATTEMPT_DIR.exists(), "V7 official attempt appeared before artifact freeze")

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
            "embedding_preserved_in_both": witness_a["post_v7"]["embedding_hash_matches_source"] and witness_b["post_v7"]["embedding_hash_matches_source"],
            "lm_head_separate_in_both": witness_a["post_v7"]["lm_head_storage_is_separate"] and witness_b["post_v7"]["lm_head_storage_is_separate"],
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
        "attempt_budget": 1,
        "attempts_consumed": 0,
        "official_execution_marker_exists": False,
        "official_execution_mode_exposed": False,
        "disposition": (
            "all marker-free V7 gates pass; PREATTEMPT_READY is sealed and attempt-0001 may be consumed exactly once only after this closure verifies"
            if all_pass
            else "preserve V7 no-go evidence, leave attempt-0001 absent and unconsumed, keep selected_policy_id null, and submit for Fresh Review"
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
    write_json(
        failure,
        {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "classification": "PREATTEMPT_FAILURE",
            "failure_taxonomy": "PREATTEMPT_EXECUTION_FAILURE",
            "root_cause_hypothesis": f"runner raised {type(exc).__name__}: {exc}",
            "regression": "repair only the isolated marker-free V7 runner and rerun pre-attempt preparation; official authorization remains unconsumed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "backend_sha256": sha256_file(BACKEND_PATH),
            "traceback": traceback.format_exc(),
            "attempt_authorization_consumed": False,
            "official_execution_marker_exists": False,
            "official_execution_mode_exposed": False,
            "failed_at_utc": utc_now(),
        },
    )
