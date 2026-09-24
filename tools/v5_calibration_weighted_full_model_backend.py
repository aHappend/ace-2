#!/usr/bin/env python3
"""Exact full-model backend for the activation-weighted V5 candidate."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
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

import run_option_b_alias_safe_calibration_weighted_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_stage1 as reference
import run_option_b_alias_safe_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_stage1 as v4_reference
import run_option_b_alias_safe_tensor_codebook_w4_grouped_dynamic_scale32_stage1 as v3
import run_option_b_grouped_scale32_stage1 as grouped
import v4_joint_full_model_backend as v4_backend
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
    ROOT,
    SNAPSHOT,
    TERMINATION_TOKEN_IDS,
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


CANDIDATE_ID = reference.CANDIDATE_ID
MISSION_ID = reference.MISSION_ID
TASK_ID = reference.TASK_ID
TASK_PATH = reference.TASK_PATH
TASK_COMPANION = reference.TASK_COMPANION
PLAN_PATH = reference.PLAN_PATH
PLAN_COMPANION = reference.PLAN_COMPANION
RUNNER_PATH = reference.RUNNER_PATH
BACKEND_PATH = Path(__file__).resolve()
V4_RUNNER_PATH = Path(v4_reference.__file__).resolve()
V4_BACKEND_PATH = Path(v4_backend.__file__).resolve()
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

OUTPUT_DIR = reference.OFFICIAL_ROOT
PREATTEMPT_DIR = OUTPUT_DIR / "preattempt"
INPUT_MANIFEST_PATH = PREATTEMPT_DIR / "non_evaluator_inputs.json"
CALIBRATION_A_PATH = PREATTEMPT_DIR / "calibration_statistics_a.json"
CALIBRATION_B_PATH = PREATTEMPT_DIR / "calibration_statistics_b.json"
CALIBRATION_EQUALITY_PATH = PREATTEMPT_DIR / "calibration_statistics_equality.json"
CANDIDATE_A_PATH = PREATTEMPT_DIR / "candidate_a_manifest.json"
CANDIDATE_B_PATH = PREATTEMPT_DIR / "candidate_b_manifest.json"
ALIAS_WITNESS_PATH = PREATTEMPT_DIR / "alias_separation_witness.json"
RECONSTRUCTION_PATH = PREATTEMPT_DIR / "model_only_weighted_reconstruction.json"
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

Q4_4_MIN = reference.Q4_4_MIN
Q4_4_MAX = reference.Q4_4_MAX
CODEBOOK_ENTRIES = reference.CODEBOOK_ENTRIES
OUTER_ITERATIONS = reference.OUTER_ITERATIONS
TORCH_THREADS = 32
WEIGHTED_CHUNK_VALUES = 1 << 13
ENERGY_LIMB_BITS = 16
ENERGY_LIMB_MASK = (1 << ENERGY_LIMB_BITS) - 1
MAX_NEW_TOKENS = 6
EXPECTED_EVENTS = grouped.LAYERS * len(grouped.EVENT_FAMILIES)
STATE_MATCH_KEYS = (
    "calibration_statistics_manifest_sha256",
    "embedding_bf16_sha256",
    "payload_manifest_sha256",
    "row_scale_manifest_sha256",
    "codebook_manifest_sha256",
    "reconstruction_manifest_sha256",
    "weighted_objective_manifest_sha256",
    "iteration_trace_manifest_sha256",
    "weight_manifest_sha256",
    "candidate_state_manifest_sha256",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


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
        "backend": "exact BF16 dyadic activation capture and 16-bit-limb weighted integer reductions",
    }


def verify_companion(path: Path, companion: Path) -> None:
    expected = f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}"
    require(companion.is_file(), f"checksum companion is missing: {companion}")
    require(companion.read_text(encoding="utf-8").strip() == expected, f"checksum companion differs: {path}")


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


def _configure_v3_helpers() -> None:
    v3.CANDIDATE_ID = CANDIDATE_ID
    v3.OUTPUT_DIR = OUTPUT_DIR
    v3.PREATTEMPT_DIR = PREATTEMPT_DIR
    v3.ATTEMPT_DIR = ATTEMPT_DIR
    v3.MARKER_PATH = MARKER_PATH
    v3.TORCH_THREADS = TORCH_THREADS


def _row_chunks(rows: int, width: int):
    rows_per_chunk = max(1, WEIGHTED_CHUNK_VALUES // width)
    for start in range(0, rows, rows_per_chunk):
        yield start, min(rows, start + rows_per_chunk)


def _energy_limbs(values: Sequence[int]) -> tuple[tuple[int, Tensor], ...]:
    maximum = max(values)
    limbs: list[tuple[int, Tensor]] = []
    offset = 0
    while maximum:
        limbs.append(
            (
                offset,
                torch.tensor([(value >> offset) & ENERGY_LIMB_MASK for value in values], dtype=torch.int64),
            )
        )
        maximum >>= ENERGY_LIMB_BITS
        offset += ENERGY_LIMB_BITS
    require(bool(limbs), "activation-energy limb decomposition is empty")
    return tuple(limbs)


def _objective_record(value: int, exponent: int) -> dict[str, Any]:
    return {"scaled_integer": str(value), "binary_exponent": exponent}


def _objective_value(record: dict[str, Any]) -> tuple[int, int]:
    return int(record["scaled_integer"]), int(record["binary_exponent"])


def _aggregate_objectives(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    exponent = min(int(record["binary_exponent"]) for record in records)
    total = sum(
        int(record["scaled_integer"]) << (int(record["binary_exponent"]) - exponent)
        for record in records
    )
    return _objective_record(total, exponent)


def _assign(source: Tensor, records: Tensor, codepoints: Sequence[int]) -> Tensor:
    points = reference.v4.validate_codebook(codepoints)
    rows, width = source.shape
    assignments = torch.empty(source.shape, dtype=torch.uint8)
    midpoint_sums = torch.tensor(
        [left + right for left, right in zip(points, points[1:])], dtype=torch.float64
    )
    for start, stop in _row_chunks(rows, width):
        scale_significands, scale_exponents = v4_backend._records_components(records[start:stop])
        scales = torch.ldexp(scale_significands.to(torch.float64), scale_exponents.to(torch.int32) - 15)
        boundaries = scales[:, None] * (midpoint_sums[None, :] / 32.0)
        selected = torch.searchsorted(
            boundaries.contiguous(), source[start:stop].to(torch.float64).contiguous(), right=False
        )
        require(bool(torch.all((selected >= 0) & (selected < CODEBOOK_ENTRIES))), "weighted assignment escaped four bits")
        assignments[start:stop] = selected.to(torch.uint8)
    return assignments


def _weighted_exact_aggregates(
    source: Tensor,
    records: Tensor,
    assignments: Tensor,
    shift_min: int,
    shift_max: int,
    limbs: Sequence[tuple[int, Tensor]],
) -> dict[str, Any]:
    rows, width = source.shape
    l_min = shift_min + SCALE32_EXPONENT_MIN - 19
    l_max = shift_max + SCALE32_EXPONENT_MAX - 19
    q_min = 2 * SCALE32_EXPONENT_MIN - 38
    q_max = 2 * SCALE32_EXPONENT_MAX - 38
    c_min = 2 * shift_min
    c_max = 2 * shift_max
    objective_exponent = min(c_min, l_min, q_min)
    l_range = l_max - l_min + 1
    q_range = q_max - q_min + 1
    c_range = c_max - c_min + 1
    l_bins = [0] * (CODEBOOK_ENTRIES * l_range)
    q_bins = [0] * (CODEBOOK_ENTRIES * q_range)
    c_bins = [0] * c_range
    counts = torch.zeros(CODEBOOK_ENTRIES, dtype=torch.int64)

    for start, stop in _row_chunks(rows, width):
        chunk = source[start:stop]
        indices = assignments[start:stop].to(torch.int64)
        weight_significands, weight_shifts = v4_backend._decode_bf16(chunk)
        weight_shifts = torch.where(
            weight_significands == 0, torch.full_like(weight_shifts, shift_min), weight_shifts
        )
        scale_significands, scale_exponents = v4_backend._records_components(records[start:stop])
        l_exponents = weight_shifts + scale_exponents[:, None] - 19
        l_keys = indices * l_range + (l_exponents - l_min)
        q_exponents = 2 * scale_exponents - 38
        q_keys = indices * q_range + (q_exponents[:, None] - q_min)
        c_keys = 2 * weight_shifts - c_min
        counts += torch.bincount(indices.reshape(-1), minlength=CODEBOOK_ENTRIES)

        for limb_offset, full_limb in limbs:
            energy = full_limb[None, :].expand(stop - start, width)
            local_l = torch.zeros(CODEBOOK_ENTRIES * l_range, dtype=torch.int64)
            local_q = torch.zeros(CODEBOOK_ENTRIES * q_range, dtype=torch.int64)
            local_c = torch.zeros(c_range, dtype=torch.int64)
            l_values = weight_significands * scale_significands[:, None] * energy
            q_values = torch.square(scale_significands)[:, None] * energy
            c_values = torch.square(weight_significands) * energy
            local_l.scatter_add_(0, l_keys.reshape(-1), l_values.reshape(-1))
            local_q.scatter_add_(0, q_keys.reshape(-1), q_values.reshape(-1))
            local_c.scatter_add_(0, c_keys.reshape(-1), c_values.reshape(-1))
            for index, value in enumerate(local_l.tolist()):
                l_bins[index] += int(value) << limb_offset
            for index, value in enumerate(local_q.tolist()):
                q_bins[index] += int(value) << limb_offset
            for index, value in enumerate(local_c.tolist()):
                c_bins[index] += int(value) << limb_offset

    constant = sum(
        coefficient << (c_min + offset - objective_exponent)
        for offset, coefficient in enumerate(c_bins)
    )
    linear_scaled: list[int] = []
    quadratic_scaled: list[int] = []
    for cluster in range(CODEBOOK_ENTRIES):
        linear_scaled.append(
            sum(
                coefficient << (l_min + offset - objective_exponent)
                for offset, coefficient in enumerate(
                    l_bins[cluster * l_range : (cluster + 1) * l_range]
                )
            )
        )
        quadratic_scaled.append(
            sum(
                coefficient << (q_min + offset - objective_exponent)
                for offset, coefficient in enumerate(
                    q_bins[cluster * q_range : (cluster + 1) * q_range]
                )
            )
        )
    return {
        "objective_binary_exponent": objective_exponent,
        "constant_scaled": constant,
        "linear_scaled": linear_scaled,
        "quadratic_scaled": quadratic_scaled,
        "cluster_counts": [int(value) for value in counts.tolist()],
    }


def _objective(aggregates: dict[str, Any], codepoints: Sequence[int]) -> int:
    points = reference.v4.validate_codebook(codepoints)
    return aggregates["constant_scaled"] + sum(
        -2 * point * linear + point * point * quadratic
        for point, linear, quadratic in zip(
            points,
            aggregates["linear_scaled"],
            aggregates["quadratic_scaled"],
            strict=True,
        )
    )


def _assign_and_aggregate(
    source: Tensor,
    records: Tensor,
    codepoints: Sequence[int],
    shift_min: int,
    shift_max: int,
    limbs: Sequence[tuple[int, Tensor]],
) -> tuple[Tensor, dict[str, Any]]:
    assignments = _assign(source, records, codepoints)
    return assignments, _weighted_exact_aggregates(
        source, records, assignments, shift_min, shift_max, limbs
    )


def _weighted_row_statistics(
    source: Tensor,
    assignments: Tensor,
    codepoints: Sequence[int],
    base_shift: int,
    shift_max: int,
    limbs: Sequence[tuple[int, Tensor]],
) -> tuple[list[int], list[int], list[int]]:
    points = torch.tensor(reference.v4.validate_codebook(codepoints), dtype=torch.int64)
    rows, width = source.shape
    shift_range = shift_max - base_shift + 1
    weight_codepoint: list[int] = []
    codepoint_squared: list[int] = []
    absolute_maximum: list[int] = []
    for start, stop in _row_chunks(rows, width):
        significands, shifts = v4_backend._decode_bf16(source[start:stop])
        delta = torch.where(significands == 0, torch.zeros_like(shifts), shifts - base_shift)
        require(bool(torch.all((delta >= 0) & (delta < shift_range))), "weighted row lattice span differs")
        exact = torch.bitwise_left_shift(significands, delta)
        absolute_maximum.extend(int(value) for value in torch.abs(exact).amax(dim=1).tolist())
        selected = points[assignments[start:stop].to(torch.int64)]
        row_count = stop - start
        row_wp = [0] * row_count
        row_p2 = [0] * row_count
        row_ids = torch.arange(row_count, dtype=torch.int64)[:, None].expand(row_count, width)
        keys = row_ids * shift_range + delta
        for limb_offset, full_limb in limbs:
            energy = full_limb[None, :].expand(row_count, width)
            p2 = torch.sum(selected * selected * energy, dim=1).tolist()
            bins = torch.zeros(row_count * shift_range, dtype=torch.int64)
            bins.scatter_add_(0, keys.reshape(-1), (significands * selected * energy).reshape(-1))
            values = bins.reshape(row_count, shift_range).tolist()
            for row_index in range(row_count):
                row_p2[row_index] += int(p2[row_index]) << limb_offset
                row_wp[row_index] += sum(
                    int(coefficient) << (shift_offset + limb_offset)
                    for shift_offset, coefficient in enumerate(values[row_index])
                )
        weight_codepoint.extend(row_wp)
        codepoint_squared.extend(row_p2)
    return weight_codepoint, codepoint_squared, absolute_maximum


def _update_row_records(
    source: Tensor,
    assignments: Tensor,
    codepoints: Sequence[int],
    prior_records: Tensor,
    base_shift: int,
    shift_max: int,
    objective_exponent: int,
    limbs: Sequence[tuple[int, Tensor]],
) -> tuple[Tensor, dict[str, Any]]:
    weight_codepoint, codepoint_squared, absolute_maximum = _weighted_row_statistics(
        source, assignments, codepoints, base_shift, shift_max, limbs
    )
    prior = [int(value) for value in prior_records.tolist()]
    selected: list[int] = []
    candidate_counts: list[int] = []
    changed = 0
    all_zero = 0
    for wp, p2, maximum, previous in zip(
        weight_codepoint, codepoint_squared, absolute_maximum, prior, strict=True
    ):
        candidates, summary = v4_backend._candidate_row_records(
            wp, p2, maximum, base_shift, previous
        )
        if summary["all_zero"]:
            choice = SCALE32_ALL_ZERO_RECORD
            all_zero += 1
        else:
            choice = min(
                candidates,
                key=lambda record: (
                    v4_backend._row_objective(
                        record, wp, p2, base_shift, objective_exponent
                    ),
                    *v4_backend._record_scale_key(record),
                ),
            )
        selected.append(choice)
        candidate_counts.append(summary["candidate_count"])
        changed += int(choice != previous)
    return torch.tensor(selected, dtype=torch.int64), {
        "row_count": len(selected),
        "changed_row_count": changed,
        "all_zero_row_count": all_zero,
        "minimum_candidate_count": min(candidate_counts),
        "maximum_candidate_count": max(candidate_counts),
        "mean_candidate_count": sum(candidate_counts) / len(candidate_counts),
    }


def _update_codebook(
    aggregates: dict[str, Any], prior_codebook: Sequence[int]
) -> tuple[tuple[int, ...], dict[str, Any]]:
    prior = reference.v4.validate_codebook(prior_codebook)
    counts = aggregates["cluster_counts"]
    costs: list[dict[int, int]] = []
    for index in range(CODEBOOK_ENTRIES):
        allowed = (prior[index],) if counts[index] == 0 else range(Q4_4_MIN, Q4_4_MAX + 1)
        linear = aggregates["linear_scaled"][index]
        quadratic = aggregates["quadratic_scaled"][index]
        costs.append(
            {point: -2 * point * linear + point * point * quadratic for point in allowed}
        )
    selected, optimum = v4_backend._strict_integer_codebook_dp(costs)
    selected = reference.v4.validate_codebook(selected)
    require(
        all(count != 0 or selected[index] == prior[index] for index, count in enumerate(counts)),
        "weighted empty cluster did not retain its prior codepoint",
    )
    return selected, {
        "cluster_counts": counts,
        "empty_cluster_count": sum(count == 0 for count in counts),
        "fixed_assignment_weighted_optimum": _objective_record(
            optimum, aggregates["objective_binary_exponent"]
        ),
    }


def _decoded_metrics(
    source: Tensor,
    records: Tensor,
    assignments: Tensor,
    codepoints: Sequence[int],
    destination: Tensor | None = None,
) -> dict[str, Any]:
    points = torch.tensor(codepoints, dtype=torch.float64)
    rows, width = source.shape
    digest = hashlib.sha256()
    sse = 0.0
    with torch.no_grad():
        # Match the sealed V4 BF16-SSE reduction order exactly.  The
        # reconstruction hash is independent of chunking, but float reduction
        # is not.
        for start, stop in v4_backend._row_chunks(rows, width):
            scale_significands, scale_exponents = v4_backend._records_components(records[start:stop])
            scales = torch.ldexp(
                scale_significands.to(torch.float64), scale_exponents.to(torch.int32) - 15
            )
            decoded = points[assignments[start:stop].to(torch.int64)] * (
                scales[:, None] / 16.0
            )
            decoded_bf16 = decoded.to(torch.bfloat16)
            digest.update(decoded_bf16.contiguous().view(torch.uint8).numpy().tobytes(order="C"))
            difference = source[start:stop].to(torch.float64) - decoded_bf16.to(torch.float64)
            sse += float(torch.sum(difference * difference))
            if destination is not None:
                destination[start:stop].copy_(decoded_bf16)
    return {"reconstruction_bf16_sha256": digest.hexdigest(), "unweighted_bf16_sse": sse}


def _construct_weighted_state(
    source: Tensor, module_name: str, channel_weights: Sequence[int]
) -> dict[str, Any]:
    require(source.ndim == 2 and source.dtype == torch.bfloat16, f"source weight format differs: {module_name}")
    weights = reference.validate_channel_weights(channel_weights, source.shape[1])
    limbs = _energy_limbs(weights)
    fresh_v4 = v4_backend._construct_joint_state(source, module_name)
    shift_min = int(fresh_v4["shift_min"])
    shift_max = int(fresh_v4["shift_max"])
    records = fresh_v4["final_records"]
    codebook = fresh_v4["final_codepoints"]
    assignments = fresh_v4["final_assignments"]
    baseline_metrics = _decoded_metrics(source, records, assignments, codebook)
    baseline_payload = fresh_v4["payload"]
    baseline_scale_raw = fresh_v4["scale_raw"]
    baseline_codebook_raw = fresh_v4["codebook_raw"]
    aggregates = _weighted_exact_aggregates(
        source, records, assignments, shift_min, shift_max, limbs
    )
    objective_exponent = aggregates["objective_binary_exponent"]
    baseline_objective = _objective(aggregates, codebook)
    previous_objective = baseline_objective
    trace: list[dict[str, Any]] = []
    del fresh_v4

    for iteration in range(1, OUTER_ITERATIONS + 1):
        records, row_summary = _update_row_records(
            source,
            assignments,
            codebook,
            records,
            shift_min,
            shift_max,
            objective_exponent,
            limbs,
        )
        assignments_after_scale, scale_aggregates = _assign_and_aggregate(
            source, records, codebook, shift_min, shift_max, limbs
        )
        after_scale_objective = _objective(scale_aggregates, codebook)
        require(
            after_scale_objective <= previous_objective,
            f"weighted row-scale update increased exact objective: {module_name}",
        )
        codebook, codebook_summary = _update_codebook(scale_aggregates, codebook)
        assignments, final_aggregates = _assign_and_aggregate(
            source, records, codebook, shift_min, shift_max, limbs
        )
        final_objective = _objective(final_aggregates, codebook)
        require(
            final_objective <= after_scale_objective,
            f"weighted codebook update increased exact objective: {module_name}",
        )
        require(
            final_objective <= previous_objective,
            f"weighted outer iteration increased exact objective: {module_name}",
        )
        payload = v3.pack_codebook_indices(assignments)
        scale_raw = b"".join(struct.pack("<I", int(value)) for value in records.tolist())
        codebook_raw = bytes(point & 0xFF for point in codebook)
        trace.append(
            {
                "iteration": iteration,
                "prior_weighted_objective": _objective_record(
                    previous_objective, objective_exponent
                ),
                "after_row_scale_weighted_objective": _objective_record(
                    after_scale_objective, objective_exponent
                ),
                "final_weighted_objective": _objective_record(
                    final_objective, objective_exponent
                ),
                "row_scale_update": row_summary,
                "row_scale_stream_sha256": hashlib.sha256(scale_raw).hexdigest(),
                "codepoints": list(codebook),
                "codebook_record_sha256": hashlib.sha256(codebook_raw).hexdigest(),
                "codebook_update": codebook_summary,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        previous_objective = final_objective

    payload = v3.pack_codebook_indices(assignments)
    scale_raw = b"".join(struct.pack("<I", int(value)) for value in records.tolist())
    codebook_raw = bytes(point & 0xFF for point in codebook)
    return {
        "shift_min": shift_min,
        "shift_max": shift_max,
        "shift_span": shift_max - shift_min,
        "activation_energy_limb_count": len(limbs),
        "activation_energy_sha256": canonical_sha256([str(value) for value in weights]),
        "fresh_v4_payload_sha256": hashlib.sha256(baseline_payload).hexdigest(),
        "fresh_v4_row_scale_stream_sha256": hashlib.sha256(baseline_scale_raw).hexdigest(),
        "fresh_v4_codebook_record_sha256": hashlib.sha256(baseline_codebook_raw).hexdigest(),
        "fresh_v4_reconstruction": baseline_metrics,
        "v4_baseline_weighted_objective": _objective_record(
            baseline_objective, objective_exponent
        ),
        "outer_iterations_completed": len(trace),
        "final_records": records,
        "final_codepoints": codebook,
        "final_assignments": assignments,
        "payload": payload,
        "scale_raw": scale_raw,
        "codebook_raw": codebook_raw,
        "final_weighted_objective": _objective_record(
            previous_objective, objective_exponent
        ),
        "iteration_trace": trace,
        "iteration_trace_sha256": canonical_sha256(trace),
    }


class _EnergyAccumulator:
    def __init__(self, width: int):
        self.width = width
        self.base_exponent: int | None = None
        self.values = [0] * width
        self.position_count = 0

    def add(self, values: Tensor) -> None:
        require(values.dtype == torch.bfloat16 and values.shape[-1] == self.width, "captured Linear input format differs")
        flat = values.detach().reshape(-1, self.width).contiguous()
        self.position_count += int(flat.shape[0])
        significands, shifts = v4_backend._decode_bf16(flat)
        nonzero = significands != 0
        if not bool(torch.any(nonzero)):
            return
        minimum = int(shifts[nonzero].min())
        maximum = int(shifts[nonzero].max())
        call_exponent = 2 * minimum
        delta = 2 * torch.where(nonzero, shifts - minimum, torch.zeros_like(shifts))
        maximum_term = (int(torch.abs(significands).max()) ** 2) << (2 * (maximum - minimum))
        if maximum_term * max(1, int(flat.shape[0])) < (1 << 62):
            exact = torch.bitwise_left_shift(torch.square(significands), delta)
            call_values = [int(value) for value in torch.sum(exact, dim=0).tolist()]
        else:
            call_values = [0] * self.width
            for shift in range(minimum, maximum + 1):
                partial = torch.sum(
                    torch.where(shifts == shift, torch.square(significands), torch.zeros_like(significands)),
                    dim=0,
                ).tolist()
                offset = 2 * (shift - minimum)
                for index, value in enumerate(partial):
                    call_values[index] += int(value) << offset
        if self.base_exponent is None:
            self.base_exponent = call_exponent
            self.values = call_values
        elif call_exponent < self.base_exponent:
            shift = self.base_exponent - call_exponent
            self.values = [(value << shift) + call for value, call in zip(self.values, call_values, strict=True)]
            self.base_exponent = call_exponent
        else:
            shift = call_exponent - self.base_exponent
            self.values = [value + (call << shift) for value, call in zip(self.values, call_values, strict=True)]

    def record(self, module_name: str) -> dict[str, Any]:
        require(self.base_exponent is not None and self.position_count > 0, f"no activation statistics captured: {module_name}")
        floored = [max(value, 1) for value in self.values]
        divisor = math.gcd(*floored)
        normalized = [value // divisor for value in floored]
        require(math.gcd(*normalized) == 1 and all(value > 0 for value in normalized), f"activation-energy reduction differs: {module_name}")
        strings = [str(value) for value in normalized]
        return {
            "module": module_name,
            "input_channels": self.width,
            "captured_position_count": self.position_count,
            "module_local_lattice_binary_exponent": self.base_exponent,
            "whole_vector_gcd": str(divisor),
            "minimum_normalized_energy": str(min(normalized)),
            "maximum_normalized_energy": str(max(normalized)),
            "normalized_h": strings,
            "normalized_h_sha256": canonical_sha256(strings),
        }


class _CalibrationCollector:
    def __init__(self, model: nn.Module):
        modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        require(len(modules) == 169, "calibration Linear count differs")
        self.accumulators = {
            name: _EnergyAccumulator(int(module.in_features)) for name, module in modules
        }
        self.mode = "all"
        self.handles = [module.register_forward_pre_hook(self._hook(name)) for name, module in modules]

    def _hook(self, name: str):
        def capture(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            require(bool(inputs) and isinstance(inputs[0], Tensor), f"Linear input missing: {name}")
            value = inputs[0]
            selected = value if self.mode == "all" else value[..., -1:, :]
            self.accumulators[name].add(selected)

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def records(self) -> list[dict[str, Any]]:
        require(not self.handles, "calibration hooks remain installed")
        return [accumulator.record(name) for name, accumulator in self.accumulators.items()]


def _capture_calibration_statistics(
    model: nn.Module, tokenizer: Any
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    require(sha256_file(CALIBRATION_CONTRACT_PATH) == "084ab27de7e19b3f66cbcacbd7c219b67e7ef1cbc51c0d92e6da67af7e6df763", "calibration contract hash differs")
    contract = load_json(CALIBRATION_CONTRACT_PATH)
    prompts = contract["chat"]["prompts"]
    require(len(prompts) == 4, "calibration prompt count differs")
    collector = _CalibrationCollector(model)
    generated: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for prompt in prompts:
                token_ids = list(prompt["token_ids"])
                encoded = tokenizer.apply_chat_template(
                    prompt["messages"], tokenize=True, add_generation_prompt=True
                )
                require(encoded == token_ids, f"calibration prompt tokenization differs: {prompt['ordinal']}")
                input_ids = torch.tensor([token_ids], dtype=torch.long)
                collector.mode = "all"
                logits = model(input_ids=input_ids, use_cache=False).logits[:, -1, :]
                output_tokens: list[int] = []
                for _step in range(MAX_NEW_TOKENS):
                    token = int(torch.argmax(logits, dim=-1).item())
                    output_tokens.append(token)
                    input_ids = torch.cat(
                        (input_ids, torch.tensor([[token]], dtype=torch.long)), dim=1
                    )
                    collector.mode = "last"
                    logits = model(input_ids=input_ids, use_cache=False).logits[:, -1, :]
                    if token in TERMINATION_TOKEN_IDS:
                        break
                generated.append(
                    {
                        "ordinal": int(prompt["ordinal"]),
                        "prompt_token_ids_sha256": prompt["token_ids_sha256"],
                        "prompt_token_count": len(token_ids),
                        "generated_token_ids": output_tokens,
                        "generated_token_ids_sha256": canonical_sha256(output_tokens),
                        "generated_token_count": len(output_tokens),
                        "captured_position_count": len(token_ids) + len(output_tokens),
                    }
                )
    finally:
        collector.close()
    modules = collector.records()
    expected_positions = sum(item["captured_position_count"] for item in generated)
    require(
        all(item["captured_position_count"] == expected_positions for item in modules),
        "calibration module position counts differ",
    )
    body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "source_model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "calibration_contract": file_record(CALIBRATION_CONTRACT_PATH),
        "capture_arithmetic": "exact BF16 significand-square accumulation on a module-local dyadic integer lattice; max(H_j,1); whole-vector GCD reduction only",
        "prompt_count": 4,
        "linear_count": len(modules),
        "generated_calibration": generated,
        "modules": modules,
        "official_inputs_accessed": False,
        "additional_holdout_inputs_accessed": False,
        "floating_point_statistic_accumulation_used": False,
    }
    manifest_sha = canonical_sha256(body)
    weights = {
        item["module"]: tuple(int(value) for value in item["normalized_h"])
        for item in modules
    }
    return {**body, "calibration_statistics_manifest_sha256": manifest_sha}, weights


def _policy_and_format() -> dict[str, Any]:
    body = {
        "payload": {
            "bits_per_weight": 4,
            "semantics": "unsigned codebook index 0 through 15",
            "packing": "row-major; even flat element low nibble; odd flat element high nibble",
        },
        "row_scale": {
            "format": "Scale32",
            "source": "fresh V4 initialization followed by eight exact activation-energy-weighted updates",
            "bytes_per_row": 4,
            "additional_records": 0,
        },
        "codebook": {
            "scope": "one per Linear tensor",
            "entry_count": 16,
            "bytes": 16,
            "alignment_bytes": 16,
            "format": "strictly increasing signed Q4.4 int8",
            "outer_iterations": OUTER_ITERATIONS,
            "empty_cluster": "retain prior codepoint",
        },
        "calibration_statistics": {
            "product_bytes": 0,
            "runtime_bytes": 0,
            "transport_bytes": 0,
        },
        "activation": {
            "policy": "grouped Dynamic Scale32",
            "event_count": EXPECTED_EVENTS,
            "payload_range": [-127, 127],
            "reserved_payload": -128,
            "delta_range": [-24, 24],
            "effective_exponent_range": [-24, 4],
        },
    }
    return {**body, "policy_and_format_sha256": canonical_sha256(body)}


def _quantize_model_weighted(
    model: nn.Module,
    calibration: dict[str, Any],
    weights: dict[str, tuple[int, ...]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {record["module"]: record for record in grouped.expected_transformer_tensors()}
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == 169, "Qwen linear count differs")
    require([name for name, _module in modules if name != "lm_head"] == list(expected), "transformer linear order differs")
    require(list(weights) == [name for name, _module in modules], "calibration statistic module order differs")
    require(sha256_file(V4_RECONSTRUCTION_PATH) == V4_RECONSTRUCTION_SHA256, "V4 reconstruction reference hash differs")
    v4_report = load_json(V4_RECONSTRUCTION_PATH)
    v4_records = {item["module"]: item for item in v4_report["tensors"]}
    require(list(v4_records) == [name for name, _module in modules], "V4 reconstruction tensor order differs")

    weight_manifest: list[dict[str, Any]] = []
    payload_manifest: list[dict[str, Any]] = []
    row_scale_manifest: list[dict[str, Any]] = []
    codebook_manifest: list[dict[str, Any]] = []
    reconstruction_manifest: list[dict[str, Any]] = []
    weighted_objective_manifest: list[dict[str, Any]] = []
    iteration_trace_manifest: list[dict[str, Any]] = []
    payload_stream = hashlib.sha256()
    scale_stream = hashlib.sha256()
    codebook_stream = hashlib.sha256()
    reconstruction_stream = hashlib.sha256()
    total_weight_count = 0
    total_payload_bytes = 0
    total_scale_bytes = 0
    total_codebook_bytes = 0

    for ordinal, (name, module) in enumerate(modules, start=1):
        started = time.monotonic()
        print(f"V5_CONSTRUCT_START {ordinal}/169 {name}", flush=True)
        source_sha = v3.tensor_sha256(module.weight)
        state = _construct_weighted_state(module.weight.detach(), name, weights[name])
        v4_reference_record = v4_records[name]
        require(
            state["fresh_v4_reconstruction"]["reconstruction_bf16_sha256"]
            == v4_reference_record["reconstruction_bf16_sha256"],
            f"fresh V4 reconstruction hash differs: {name}",
        )
        require(
            state["fresh_v4_reconstruction"]["unweighted_bf16_sse"]
            == v4_reference_record["v4_joint_bf16_sse"],
            f"fresh V4 reconstruction SSE differs: {name}",
        )
        final_metrics = _decoded_metrics(
            module.weight.detach(),
            state["final_records"],
            state["final_assignments"],
            state["final_codepoints"],
            module.weight,
        )
        baseline_value, baseline_exp = _objective_value(
            state["v4_baseline_weighted_objective"]
        )
        final_value, final_exp = _objective_value(state["final_weighted_objective"])
        require(baseline_exp == final_exp, f"weighted objective exponent differs: {name}")
        non_regressing = final_value <= baseline_value
        strict = final_value < baseline_value
        payload = state["payload"]
        scale_raw = state["scale_raw"]
        codebook_raw = state["codebook_raw"]
        payload_record = {
            "module": name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        scale_record = {
            "module": name,
            "record_count": int(module.weight.shape[0]),
            "bytes": len(scale_raw),
            "sha256": hashlib.sha256(scale_raw).hexdigest(),
        }
        codebook_record = {
            "module": name,
            "bytes": 16,
            "sha256": hashlib.sha256(codebook_raw).hexdigest(),
            "q4_4_codepoints": list(state["final_codepoints"]),
        }
        reconstruction_record = {
            "module": name,
            "weight_count": int(module.weight.numel()),
            "fresh_v4_unweighted_bf16_sse": state["fresh_v4_reconstruction"]["unweighted_bf16_sse"],
            "v5_unweighted_bf16_sse": final_metrics["unweighted_bf16_sse"],
            "fresh_v4_reconstruction_bf16_sha256": state["fresh_v4_reconstruction"]["reconstruction_bf16_sha256"],
            "v5_reconstruction_bf16_sha256": final_metrics["reconstruction_bf16_sha256"],
        }
        objective_record = {
            "module": name,
            "activation_energy_sha256": state["activation_energy_sha256"],
            "fresh_v4_weighted_objective": state["v4_baseline_weighted_objective"],
            "v5_weighted_objective": state["final_weighted_objective"],
            "v5_non_regressing_against_fresh_v4": non_regressing,
            "v5_strictly_improved_against_fresh_v4": strict,
        }
        trace_record = {
            "module": name,
            "source_bf16_sha256": source_sha,
            "shift_min": state["shift_min"],
            "shift_max": state["shift_max"],
            "shift_span": state["shift_span"],
            "activation_energy_limb_count": state["activation_energy_limb_count"],
            "activation_energy_sha256": state["activation_energy_sha256"],
            "fresh_v4_payload_sha256": state["fresh_v4_payload_sha256"],
            "fresh_v4_row_scale_stream_sha256": state["fresh_v4_row_scale_stream_sha256"],
            "fresh_v4_codebook_record_sha256": state["fresh_v4_codebook_record_sha256"],
            "outer_iterations_completed": state["outer_iterations_completed"],
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
            "codebook": codebook_record,
            "reconstruction": reconstruction_record,
            "weighted_objective": objective_record,
            "iteration_trace_sha256": state["iteration_trace_sha256"],
        }
        weight_manifest.append(record)
        payload_manifest.append(payload_record)
        row_scale_manifest.append(scale_record)
        codebook_manifest.append(codebook_record)
        reconstruction_manifest.append(reconstruction_record)
        weighted_objective_manifest.append(objective_record)
        iteration_trace_manifest.append(trace_record)
        payload_stream.update(name.encode() + b"\0" + payload)
        scale_stream.update(name.encode() + b"\0" + scale_raw)
        codebook_stream.update(name.encode() + b"\0" + codebook_raw)
        reconstruction_stream.update(
            name.encode() + b"\0" + bytes.fromhex(final_metrics["reconstruction_bf16_sha256"])
        )
        weight_count = int(module.weight.numel())
        total_weight_count += weight_count
        total_payload_bytes += len(payload)
        total_scale_bytes += len(scale_raw)
        total_codebook_bytes += len(codebook_raw)
        require(len(payload) == (weight_count + 1) // 2, f"packed payload byte count differs: {name}")
        require(len(scale_raw) == module.weight.shape[0] * 4, f"row-scale byte count differs: {name}")
        require(len(codebook_raw) == 16, f"codebook byte count differs: {name}")
        print(
            f"V5_CONSTRUCT_DONE {ordinal}/169 {name} seconds={time.monotonic() - started:.3f} "
            f"weighted_non_regressing={non_regressing} weighted_strict={strict}",
            flush=True,
        )
        del state
        gc.collect()

    baseline_records = [item["fresh_v4_weighted_objective"] for item in weighted_objective_manifest]
    final_records = [item["v5_weighted_objective"] for item in weighted_objective_manifest]
    aggregate_baseline = _aggregate_objectives(baseline_records)
    aggregate_final = _aggregate_objectives(final_records)
    baseline_value, baseline_exp = _objective_value(aggregate_baseline)
    final_value, final_exp = _objective_value(aggregate_final)
    require(baseline_exp == final_exp, "aggregate weighted objective exponent differs")
    non_regressing_count = sum(
        bool(item["v5_non_regressing_against_fresh_v4"])
        for item in weighted_objective_manifest
    )
    strict_count = sum(
        bool(item["v5_strictly_improved_against_fresh_v4"])
        for item in weighted_objective_manifest
    )
    report_body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "measurement": "exact activation-energy-weighted rational reconstruction objective",
        "reference": {
            "candidate_id": v4_report["candidate_id"],
            "artifact": file_record(V4_RECONSTRUCTION_PATH),
            "freshly_reconstructed_not_imported": True,
        },
        "calibration_statistics_manifest_sha256": calibration[
            "calibration_statistics_manifest_sha256"
        ],
        "tensor_count": len(weighted_objective_manifest),
        "per_tensor_non_regressing_count": non_regressing_count,
        "per_tensor_strictly_improved_count": strict_count,
        "all_169_tensors_non_regressing_against_fresh_v4": non_regressing_count == 169,
        "aggregate_fresh_v4_weighted_objective": aggregate_baseline,
        "aggregate_v5_weighted_objective": aggregate_final,
        "aggregate_strictly_improved_against_fresh_v4": final_value < baseline_value,
        "aggregate_fresh_v4_unweighted_bf16_sse": sum(
            item["fresh_v4_unweighted_bf16_sse"] for item in reconstruction_manifest
        ),
        "aggregate_v5_unweighted_bf16_sse": sum(
            item["v5_unweighted_bf16_sse"] for item in reconstruction_manifest
        ),
        "tensors": weighted_objective_manifest,
    }
    report = {
        **report_body,
        "status": (
            "PASS_169_OF_169_AND_STRICT_AGGREGATE_WEIGHTED_IMPROVEMENT"
            if report_body["all_169_tensors_non_regressing_against_fresh_v4"]
            and report_body["aggregate_strictly_improved_against_fresh_v4"]
            else "NO_GO_V4_RELATIVE_WEIGHTED_RECONSTRUCTION_GATE"
        ),
        "reconstruction_report_sha256": canonical_sha256(report_body),
    }

    transformer_records = [record for record in weight_manifest if record["scope"] == "transformer"]
    standard_layer_weights = grouped.policy_cost()["standard_decoder_layer_weight_count"]
    standard_layer_existing_bytes = grouped.policy_cost()["standard_decoder_layer_total_weight_bytes"]
    standard_layer_bits = ((standard_layer_existing_bytes + 7 * 16) * 8) / standard_layer_weights
    lm_head_record = next(record for record in weight_manifest if record["module"] == "lm_head")
    lm_head_weight_count = int(np.prod(lm_head_record["shape"]))
    lm_head_total_bytes = lm_head_record["payload"]["bytes"] + lm_head_record["row_scale"]["bytes"] + 16
    lm_head_bits = (lm_head_total_bytes * 8) / lm_head_weight_count
    require(len(transformer_records) == 168 and len(weight_manifest) == 169, "weighted tensor count differs")
    require(total_codebook_bytes == 2704, "whole-model codebook byte count differs")
    require(abs(standard_layer_bits - 4.027257898351649) < 1e-15, "standard-layer stored bits differ")
    require(abs(lm_head_bits - 4.035715225959803) < 1e-15, "lm_head stored bits differ")
    require(standard_layer_bits <= 4.036 and lm_head_bits <= 4.036, "stored-bit cap exceeded")
    manifests = {
        "payload": payload_manifest,
        "row_scale": row_scale_manifest,
        "codebook": codebook_manifest,
        "reconstruction": reconstruction_manifest,
        "weighted_objective": weighted_objective_manifest,
        "iteration_trace": iteration_trace_manifest,
    }
    hashes = {
        "payload_manifest_sha256": canonical_sha256(payload_manifest),
        "row_scale_manifest_sha256": canonical_sha256(row_scale_manifest),
        "codebook_manifest_sha256": canonical_sha256(codebook_manifest),
        "reconstruction_manifest_sha256": canonical_sha256(reconstruction_manifest),
        "weighted_objective_manifest_sha256": canonical_sha256(weighted_objective_manifest),
        "iteration_trace_manifest_sha256": canonical_sha256(iteration_trace_manifest),
        "weight_manifest_sha256": canonical_sha256(weight_manifest),
        "reconstruction_report_sha256": report["reconstruction_report_sha256"],
        "payload_stream_sha256": payload_stream.hexdigest(),
        "row_scale_stream_sha256": scale_stream.hexdigest(),
        "codebook_stream_sha256": codebook_stream.hexdigest(),
        "reconstruction_stream_sha256": reconstruction_stream.hexdigest(),
    }
    storage = {
        "total_weight_count": total_weight_count,
        "total_payload_bytes": total_payload_bytes,
        "total_row_scale_bytes": total_scale_bytes,
        "whole_model_codebook_bytes": total_codebook_bytes,
        "standard_decoder_layer_exact_bits_per_weight": standard_layer_bits,
        "lm_head_exact_bits_per_weight": lm_head_bits,
        "maximum_total_stored_bits_per_weight": 4.036,
        "calibration_statistics_product_bytes": 0,
    }
    return weight_manifest, {
        "manifests": manifests,
        "hashes": hashes,
        "storage": storage,
        "reconstruction": report,
    }


def _load_candidate(
    label: str, tokenizer: Any
) -> tuple[nn.Module, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
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

    calibration, weights = _capture_calibration_statistics(model, tokenizer)
    require(model.lm_head.weight is embedding, "calibration capture changed source tying")
    require(v3.tensor_sha256(embedding) == source_hash, "calibration capture changed embedding bytes")

    cloned = lm_head.detach().clone()
    require(v3.tensor_sha256(cloned) == source_hash, "detached lm_head clone differs before split")
    model.lm_head.weight = nn.Parameter(cloned, requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight is not embedding, "lm_head Parameter object remained tied")
    require(model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head storage remained tied")
    require(v3.tensor_sha256(model.lm_head.weight) == source_hash, "lm_head clone bytes differ after split")
    require(id(model.model.embed_tokens.weight) == source_object_id, "embedding Parameter object changed during split")
    require(model.model.embed_tokens.weight.data_ptr() == source_pointer, "embedding storage changed during split")

    weight_manifest, construction = _quantize_model_weighted(model, calibration, weights)
    require(model.config.tie_word_embeddings is False, "candidate config was retied")
    require(model.model.embed_tokens.weight is embedding, "embedding Parameter object changed after construction")
    require(model.model.embed_tokens.weight.data_ptr() == source_pointer, "embedding storage changed after construction")
    require(list(embedding.shape) == source_shape and str(embedding.dtype) == source_dtype, "embedding shape or dtype changed")
    embedding_hash = v3.tensor_sha256(embedding)
    require(embedding_hash == source_hash == SOURCE_EMBEDDING_SHA256, "embedding bytes changed after construction")
    state = v3.state_manifest(model)
    identity = {
        "calibration_statistics_manifest_sha256": calibration[
            "calibration_statistics_manifest_sha256"
        ],
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
        "module_counts": {"transformer_linears": 168, "lm_head": 1, "total_joint_linears": 169},
        "identity": identity,
        "storage": construction["storage"],
        "payload_manifest": construction["manifests"]["payload"],
        "row_scale_manifest": construction["manifests"]["row_scale"],
        "codebook_manifest": construction["manifests"]["codebook"],
        "reconstruction_manifest": construction["manifests"]["reconstruction"],
        "weighted_objective_manifest": construction["manifests"]["weighted_objective"],
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
            "pre_joint_byte_identical": True,
            "config_tie_word_embeddings": bool(model.config.tie_word_embeddings),
        },
        "post_joint": {
            "embedding_bf16_sha256": embedding_hash,
            "embedding_hash_matches_source": embedding_hash == SOURCE_EMBEDDING_SHA256,
            "lm_head_storage_is_separate": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
            "transformer_joint_count": 168,
            "lm_head_joint_count": 1,
        },
    }
    return model, calibration, manifest, witness, construction["reconstruction"]


def _compare_calibration(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    require(a == b, "independent calibration-statistic captures differ")
    require(a["linear_count"] == 169 and len(a["modules"]) == 169, "calibration statistic count differs")
    require(a["prompt_count"] == 4, "calibration prompt count differs")
    return {
        "byte_identical": canonical_bytes(a) == canonical_bytes(b),
        "manifest_sha256": a["calibration_statistics_manifest_sha256"],
        "linear_count": 169,
        "prompt_count": 4,
        "generated_token_arrays_match": True,
        "position_counts_match": True,
        "normalized_h_vectors_match": True,
    }


def _compare_candidates(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    comparisons = {key: a["identity"][key] == b["identity"][key] for key in STATE_MATCH_KEYS}
    require(all(comparisons.values()), f"candidate A/B identity differs: {comparisons}")
    fields = (
        "payload_manifest",
        "row_scale_manifest",
        "codebook_manifest",
        "reconstruction_manifest",
        "weighted_objective_manifest",
        "iteration_trace_manifest",
        "weight_manifest",
        "state_manifest",
    )
    field_matches = {field: a[field] == b[field] for field in fields}
    require(all(field_matches.values()), f"candidate A/B whole-model manifests differ: {field_matches}")
    return {
        "state_match_keys": list(STATE_MATCH_KEYS),
        "identity_comparisons": comparisons,
        "manifest_byte_equivalent_fields": list(fields),
        "manifest_comparisons": field_matches,
        "all_match": True,
    }


def combined_self_test() -> dict[str, Any]:
    environment = require_project_python()
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    task = load_json(TASK_PATH)
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "frozen V5 Engineer authority differs")
    root_existed_before = OUTPUT_DIR.exists()
    marker_existed_before = MARKER_PATH.exists()
    require(not marker_existed_before, "V5 execution marker already exists")
    exact_reference = reference.self_test()
    fixture = reference.v4.faithful_fixture()
    source = torch.tensor([[float(value) for value in row] for row in fixture], dtype=torch.bfloat16)
    weights = reference.normalize_activation_energies(tuple((index + 1) ** 2 for index in range(source.shape[1])))
    scalable_a = _construct_weighted_state(source, "scalable_weighted_fixture_a", weights)
    scalable_b = _construct_weighted_state(source, "scalable_weighted_fixture_b", weights)
    expected = exact_reference["weighted_fixture"]
    require(scalable_a["final_records"].tolist() == expected["final_row_scale_records"], "scalable weighted row scales differ")
    require(list(scalable_a["final_codepoints"]) == expected["final_codepoints"], "scalable weighted codebook differs")
    require(scalable_a["payload"].hex() == expected["payload_hex"], "scalable weighted payload differs")
    require(scalable_a["scale_raw"] == scalable_b["scale_raw"], "independent scalable weighted row scales differ")
    require(scalable_a["codebook_raw"] == scalable_b["codebook_raw"], "independent scalable weighted codebooks differ")
    require(scalable_a["payload"] == scalable_b["payload"], "independent scalable weighted payloads differ")
    require(scalable_a["iteration_trace"] == scalable_b["iteration_trace"], "independent scalable weighted traces differ")
    require(OUTPUT_DIR.exists() == root_existed_before, "combined self-test changed the V5 namespace")
    require(MARKER_PATH.exists() == marker_existed_before is False, "combined self-test changed the V5 marker")
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
        "reference_self_test_sha256": canonical_sha256(exact_reference),
        "checks": {
            "exact_weighted_reference_self_test_passed": exact_reference["status"] == "PASS",
            "scalable_exact_weighted_dyadic_lattice": True,
            "scalable_matches_fraction_reference_final_row_scales": True,
            "scalable_matches_fraction_reference_final_codebook": True,
            "scalable_matches_fraction_reference_final_payload": True,
            "independent_scalable_fixture_a_b_match": True,
            "energy_limb_bits": ENERGY_LIMB_BITS,
            "weighted_chunk_values": WEIGHTED_CHUNK_VALUES,
            "official_execution_marker_exists": False,
        },
        "scalable_fixture": {
            "shape": list(source.shape),
            "final_row_scale_stream_sha256": hashlib.sha256(scalable_a["scale_raw"]).hexdigest(),
            "final_codebook_record_sha256": hashlib.sha256(scalable_a["codebook_raw"]).hexdigest(),
            "final_payload_sha256": hashlib.sha256(scalable_a["payload"]).hexdigest(),
            "iteration_trace_sha256": scalable_a["iteration_trace_sha256"],
        },
        "official_build_namespace_changed": False,
        "official_execution_marker_exists": False,
    }


def _validate_reconstruction(report: dict[str, Any]) -> bool:
    require(report["tensor_count"] == len(report["tensors"]) == 169, "V5 weighted tensor count differs")
    non_regressing = sum(bool(item["v5_non_regressing_against_fresh_v4"]) for item in report["tensors"])
    strict = sum(bool(item["v5_strictly_improved_against_fresh_v4"]) for item in report["tensors"])
    require(non_regressing == report["per_tensor_non_regressing_count"], "V5 weighted non-regression aggregate differs")
    require(strict == report["per_tensor_strictly_improved_count"], "V5 weighted strict aggregate differs")
    baseline, baseline_exp = _objective_value(report["aggregate_fresh_v4_weighted_objective"])
    final, final_exp = _objective_value(report["aggregate_v5_weighted_objective"])
    passed = (
        report["all_169_tensors_non_regressing_against_fresh_v4"] is True
        and non_regressing == 169
        and baseline_exp == final_exp
        and final < baseline
        and report["aggregate_strictly_improved_against_fresh_v4"] is True
    )
    require(report["status"].startswith("PASS_") == passed, "V5 weighted reconstruction status differs")
    body = {key: value for key, value in report.items() if key not in {"status", "reconstruction_report_sha256"}}
    require(report["reconstruction_report_sha256"] == canonical_sha256(body), "V5 weighted reconstruction hash differs")
    return passed


def _validate_quality(report: dict[str, Any]) -> bool:
    _configure_v3_helpers()
    return v3.validate_quality(report)


def _validate_dynamic(report: dict[str, Any]) -> bool:
    _configure_v3_helpers()
    return v3.validate_dynamic(report)


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
        require(not OUTPUT_DIR.exists(), "V5 namespace existed before preparation")
    require(not matches, "a prior V5 execution marker exists")
    require(not attempts, f"V5 attempt namespace is already consumed: {attempts}")
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
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "cannot recover a consumed V5 attempt namespace")
    complete = PREATTEMPT_DIR.is_dir() and CONTRACT_PATH.is_file() and (READY_PATH.is_file() or NO_GO_PATH.is_file())
    require(not complete, "V5 pre-attempt closure already exists; use verify")
    candidates = [path for path in OUTPUT_DIR.iterdir() if path.name != "preflight-failures"]
    require(candidates, "existing V5 namespace has ambiguous provenance")
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
        "v4_reconstruction_reference": V4_RECONSTRUCTION_PATH,
        "official_matrix_hash_only": MATRIX_PATH,
        "runner_source": RUNNER_PATH,
        "scalable_backend_source": BACKEND_PATH,
        "v4_reference_runner": V4_RUNNER_PATH,
        "v4_reference_backend": V4_BACKEND_PATH,
        "reviewed_v3_preflight_helper": V3_RUNNER_PATH,
        "reviewed_grouped_dynamic_runner": GROUPED_RUNNER_PATH,
        "response_gate_source": ROOT / "tools/qwen_instruct_response_gate.py",
        "identity_helper_source": ROOT / "tools/qwen_instruct_option_b.py",
        "quality_contracts": ROOT / "tools/ace2_quality_contracts.py",
        "non_evaluator_inputs": INPUT_MANIFEST_PATH,
        "calibration_statistics_a": CALIBRATION_A_PATH,
        "calibration_statistics_b": CALIBRATION_B_PATH,
        "calibration_statistics_equality": CALIBRATION_EQUALITY_PATH,
        "candidate_a_manifest": CANDIDATE_A_PATH,
        "candidate_b_manifest": CANDIDATE_B_PATH,
        "alias_separation_witness": ALIAS_WITNESS_PATH,
        "model_only_weighted_reconstruction": RECONSTRUCTION_PATH,
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
    calibration_equality: dict[str, Any],
    state_comparison: dict[str, Any],
    gate_passes: dict[str, bool],
) -> dict[str, Any]:
    task = load_json(TASK_PATH)
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "V5 Engineer authority differs")
    require(task["fresh_attempt_namespace"]["attempt_budget"] == 1, "V5 attempt budget differs")
    require(task["fresh_attempt_namespace"]["attempts_consumed_at_task_freeze"] == 0, "V5 attempt was consumed at freeze")
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
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
            "backend": "exact BF16 dyadic calibration plus exact 16-bit-limb weighted integer optimization",
            "fresh_v4_initialization": True,
            "calibration_duplicate_capture": calibration_equality,
            "source_tied_then_runtime_split_lm_head": True,
            "source_embedding_bf16_sha256": SOURCE_EMBEDDING_SHA256,
            "embedding_bytes_preserved": True,
            "transformer_joint_linears": 168,
            "separated_lm_head_joint_linears": 1,
            "candidate_a_and_b_whole_model_manifests_match": state_comparison,
        },
        "pre_attempt_gates": {
            "duplicate_calibration_statistics": {"passed": gate_passes["duplicate_calibration_statistics"], "artifact": artifacts["calibration_statistics_equality"]},
            "v4_relative_weighted_model_169_of_169": {"passed": gate_passes["v4_relative_weighted_model_169_of_169"], "artifact": artifacts["model_only_weighted_reconstruction"]},
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
        "official_execution": {
            "mode_exposed_by_runner": True,
            "attempt_directory_exists": False,
            "execution_marker_exists": False,
            "conditional_execution_rule": "consume attempt-0001 only after this PREATTEMPT_READY closure verifies",
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


def load_verified_closure(require_unconsumed: bool = True) -> dict[str, Any]:
    _configure_v3_helpers()
    require_project_python()
    require(CONTRACT_PATH.is_file(), "V5 predeclared contract is missing")
    require(READY_PATH.is_file() != NO_GO_PATH.is_file(), "exactly one V5 pre-attempt terminal record is required")
    wrapper = load_json(CONTRACT_PATH)
    require(set(wrapper) == {"contract", "contract_sha256"}, "V5 predeclared contract wrapper differs")
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "V5 predeclared contract canonical hash differs")
    contract = wrapper["contract"]
    require(contract["candidate_id"] == CANDIDATE_ID and contract["task_id"] == TASK_ID, "V5 predeclared authority differs")
    require(contract["artifacts"]["runner_source"]["sha256"] == sha256_file(RUNNER_PATH), "V5 runner changed after freeze")
    require(contract["artifacts"]["scalable_backend_source"]["sha256"] == sha256_file(BACKEND_PATH), "V5 backend changed after freeze")
    for record in contract["artifacts"].values():
        _verify_bound_record(record)
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(verify_source_snapshot() == contract["source_model"], "source snapshot identity differs")
    packages = {**verify_versions(), "numpy": importlib.metadata.version("numpy")}
    require(packages == contract["environment"]["packages"], "package versions differ")
    require(combined_self_test() == load_json(SELF_TEST_PATH), "V5 runner self-test result differs")
    calibration_a = load_json(CALIBRATION_A_PATH)
    calibration_b = load_json(CALIBRATION_B_PATH)
    calibration_equality = _compare_calibration(calibration_a, calibration_b)
    require(calibration_equality == load_json(CALIBRATION_EQUALITY_PATH), "V5 calibration equality record differs")
    candidate_a = load_json(CANDIDATE_A_PATH)
    candidate_b = load_json(CANDIDATE_B_PATH)
    comparison = _compare_candidates(candidate_a, candidate_b)
    witness = load_json(ALIAS_WITNESS_PATH)
    require(witness["state_comparison"] == comparison, "V5 alias witness state comparison differs")
    reconstruction = load_json(RECONSTRUCTION_PATH)
    quality = load_json(QUALITY_PATH)
    dynamic = load_json(DYNAMIC_PATH)
    gate_passes = {
        "duplicate_calibration_statistics": calibration_equality["byte_identical"],
        "v4_relative_weighted_model_169_of_169": _validate_reconstruction(reconstruction),
        "non_evaluator_quality_8_of_8": _validate_quality(quality),
        "dynamic_312_event_closure": _validate_dynamic(dynamic),
        "two_build_alias_and_manifest_closure": comparison["all_match"]
        and witness["embedding_preserved_in_both"]
        and witness["lm_head_separate_in_both"],
    }
    require(contract["pre_attempt_gates"]["all_pass"] == all(gate_passes.values()), "V5 contract aggregate gate status differs")
    for name, passed in gate_passes.items():
        require(contract["pre_attempt_gates"][name]["passed"] == passed, f"V5 contract gate differs: {name}")
    terminal_path = READY_PATH if READY_PATH.is_file() else NO_GO_PATH
    terminal = load_json(terminal_path)
    require(terminal["candidate_id"] == CANDIDATE_ID, "V5 terminal candidate differs")
    require(terminal["contract"]["sha256"] == sha256_file(CONTRACT_PATH), "V5 terminal contract file hash differs")
    require(terminal["contract_sha256"] == wrapper["contract_sha256"], "V5 terminal contract canonical hash differs")
    require(terminal["bound_artifacts"] == contract["artifacts"], "V5 terminal artifact snapshot differs")
    if terminal_path == READY_PATH:
        require(terminal["status"] == "PREATTEMPT_READY" and all(gate_passes.values()), "V5 ready exists without all gates passing")
    else:
        require(terminal["status"] == "PREATTEMPT_NO_GO" and not all(gate_passes.values()), "V5 no-go does not preserve a failed gate")
    if require_unconsumed:
        _audit_attempt_namespace(expect_root_absent=False)
        require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "V5 official attempt already exists")
    else:
        attempts = sorted(path.name for path in OUTPUT_DIR.glob("attempt-*") if path.is_dir())
        require(attempts in ([], ["attempt-0001"]), "V5 official attempt set differs")
    return {
        "status": terminal["status"],
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": wrapper["contract_sha256"],
        "gate_passes": gate_passes,
        "duplicate_calibration_statistics_match": calibration_equality["byte_identical"],
        "per_tensor_non_regressing_count": reconstruction["per_tensor_non_regressing_count"],
        "exact_sequence_matches": quality["exact_bf16_sequence_match_count"],
        "gate_outcome_matches": quality["response_gate_outcome_match_count"],
        "event_count": dynamic["activation_execution"]["event_count"],
        "attempts_consumed": 1 if MARKER_PATH.exists() else 0,
    }


def prepare() -> dict[str, Any]:
    _configure_v3_helpers()
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
    require(probe.read_bytes() == b"preflight\n", "V5 output-path readback differs")
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
    print("V5_CANDIDATE_A_START calibration_and_construction", flush=True)
    candidate_a, calibration_a, manifest_a, witness_a, reconstruction_a = _load_candidate(
        "candidate_a_independent_calibration_and_construction", tokenizer
    )
    chronology["candidate_a_complete_at_utc"] = utc_now()
    print("V5_CANDIDATE_A_DONE", flush=True)
    del candidate_a
    gc.collect()

    chronology["candidate_b_started_at_utc"] = utc_now()
    print("V5_CANDIDATE_B_START calibration_construction_quality_dynamic", flush=True)
    candidate_b, calibration_b, manifest_b, witness_b, reconstruction_b = _load_candidate(
        "candidate_b_independent_calibration_construction_quality_and_dynamic", tokenizer
    )
    calibration_equality = _compare_calibration(calibration_a, calibration_b)
    state_comparison = _compare_candidates(manifest_a, manifest_b)
    require(reconstruction_a == reconstruction_b, "V5 candidate A/B reconstruction reports differ")
    chronology["candidate_b_constructed_at_utc"] = utc_now()
    quality, dynamic = v3.non_evaluator_quality_and_dynamic(
        candidate_b, tokenizer, inputs, input_manifest
    )
    chronology["candidate_b_quality_and_dynamic_complete_at_utc"] = utc_now()
    print("V5_QUALITY_DYNAMIC_DONE", flush=True)
    del candidate_b
    gc.collect()

    gate_passes = {
        "duplicate_calibration_statistics": calibration_equality["byte_identical"],
        "v4_relative_weighted_model_169_of_169": _validate_reconstruction(reconstruction_b),
        "non_evaluator_quality_8_of_8": _validate_quality(quality),
        "dynamic_312_event_closure": _validate_dynamic(dynamic),
        "two_build_alias_and_manifest_closure": state_comparison["all_match"]
        and witness_a["post_joint"]["embedding_hash_matches_source"]
        and witness_b["post_joint"]["embedding_hash_matches_source"]
        and witness_a["post_joint"]["lm_head_storage_is_separate"]
        and witness_b["post_joint"]["lm_head_storage_is_separate"],
    }
    require(not MARKER_PATH.exists() and not ATTEMPT_DIR.exists(), "V5 official attempt appeared before artifact freeze")
    PREATTEMPT_DIR.mkdir(parents=False, exist_ok=False)
    write_json(INPUT_MANIFEST_PATH, input_manifest)
    write_json(CALIBRATION_A_PATH, calibration_a)
    write_json(CALIBRATION_B_PATH, calibration_b)
    write_json(CALIBRATION_EQUALITY_PATH, calibration_equality)
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
            "embedding_preserved_in_both": witness_a["post_joint"]["embedding_hash_matches_source"] and witness_b["post_joint"]["embedding_hash_matches_source"],
            "lm_head_separate_in_both": witness_a["post_joint"]["lm_head_storage_is_separate"] and witness_b["post_joint"]["lm_head_storage_is_separate"],
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
    wrapper = _build_contract(
        environment,
        source_identity,
        namespace_audit,
        calibration_equality,
        state_comparison,
        gate_passes,
    )
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
        "disposition": (
            "all marker-free V5 gates pass; PREATTEMPT_READY is sealed and attempt-0001 may now be consumed exactly once"
            if all_pass
            else "preserve V5 no-go evidence, leave attempt-0001 absent and unconsumed, keep selected_policy_id null, and submit for Fresh Review"
        ),
    }
    atomic_write_json(terminal_path, terminal)
    verified = load_verified_closure()
    for path in list(PREATTEMPT_DIR.iterdir()) + [CONTRACT_PATH, terminal_path]:
        if path.is_file():
            path.chmod(0o444)
    return verified


def execute_once() -> int:
    closure = load_verified_closure(require_unconsumed=True)
    require(closure["status"] == "PREATTEMPT_READY", "V5 official execution requires verified PREATTEMPT_READY")
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
        "runner_sha256": sha256_file(RUNNER_PATH),
        "backend_sha256": sha256_file(BACKEND_PATH),
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
        candidate_model, official_calibration, official_manifest, official_witness, _reconstruction = _load_candidate(
            "official_attempt_candidate", tokenizer
        )
        preflight_manifest = load_json(CANDIDATE_B_PATH)
        comparison = _compare_candidates(preflight_manifest, official_manifest)
        require(
            official_calibration["calibration_statistics_manifest_sha256"]
            == load_json(CALIBRATION_B_PATH)["calibration_statistics_manifest_sha256"],
            "official calibration statistics differ from preflight",
        )
        policy = v3.AuditedDynamicPolicy(load_json(BASE_SCALES_PATH))
        policy.install(candidate_model)
        prompt_results: list[dict[str, Any]] = []
        model_forwards = 0
        for prompt in PROMPTS:
            bf16 = references[prompt.case_id]
            candidate, forwards = grouped.generate_candidate(
                candidate_model, tokenizer, inputs[prompt.case_id], prompt.expected, bf16
            )
            model_forwards += forwards
            candidate_prompt_count += 1
            disagreements = sequence_disagreements(
                bf16["generated_token_ids"], candidate["generated_token_ids"]
            )
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
            "classification": "qwen_instruct_stage1_option_b_alias_safe_calibration_weighted_joint_codebook_single_candidate",
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
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": {**versions, "numpy": importlib.metadata.version("numpy")}, "torch_num_threads": torch.get_num_threads()},
            "official_candidate_manifest": official_manifest,
            "official_alias_witness": official_witness,
            "preflight_state_match": comparison,
            "activation_execution": activation_summary,
            "prompts": prompt_results,
            "aggregate": aggregate,
            "disagreement_set": disagreement_set,
            "eligibility": {"rule": "binary 9/9 exact BF16 token arrays and 9/9 response-gate outcomes", "eligible": eligible, "eligible_candidate_id": CANDIDATE_ID if eligible else None, "selected_policy_id": None, "fresh_reviewer_required": True},
            "failure_analysis": None if eligible else {"failure_taxonomy": "NUMERICAL_EXACT_BF16_DISAGREEMENT", "root_cause_hypothesis": "the activation-weighted four-bit reconstruction changed one or more greedy logits relative to BF16", "regression": "preserve the immutable disagreement set; do not rerun the consumed candidate"},
            "scope_guards": {"candidate_attempt_count": 1, "alternate_policy_executed": False, "official_inputs_used_before_marker": False, "saturation_or_clipping_used": False, "runner_changed_after_freeze": False, "rtl_mutated": False, "ppa_executed": False, "network_access_performed": False},
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
            "requested_review": ["task and hash closure", "duplicate exact calibration-statistic capture", "169/169 weighted closure", "8/8 preflight exactness", "312 legal events", "one consumed attempt", "nine exact BF16 token arrays and response-gate comparisons"],
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
    write_json(terminal_path, {"schema_version": 1, "candidate_id": CANDIDATE_ID, "attempt_id": "attempt-0001", "classification": terminal_class, "return_code": return_code, "candidate_prompt_results_completed": candidate_prompt_count, "completed_at_utc": utc_now(), "elapsed_seconds": time.monotonic() - started})
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
    require(marker["runner_sha256"] == sha256_file(RUNNER_PATH), "executed runner hash differs")
    sums = ATTEMPT_DIR / "SHA256SUMS"
    companion = ATTEMPT_DIR / "SHA256SUMS.sha256"
    verify_checksum_companion(sums, companion)
    for digest, path in parse_sums(sums):
        require(path.is_file() and sha256_file(path) == digest, f"official evidence hash differs: {path}")
    terminal = load_json(ATTEMPT_DIR / "terminal_status.json")
    failure_path = ATTEMPT_DIR / "failure.json"
    if failure_path.exists():
        failure = load_json(failure_path)
        require(failure["numerical_correctness_conclusion"] is None, "execution failure drew a numerical conclusion")
        return {"status": failure["status"], "classification": failure["classification"], "candidate_id": CANDIDATE_ID, "attempt_count": 1, "candidate_prompt_results_completed": failure["candidate_prompt_results_completed"], "eligible": None, "selected_policy_id": None, "failure_sha256": sha256_file(failure_path), "terminal_return_code": terminal["return_code"]}
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
            disagreements.append({"case_id": item["case_id"], "token_disagreements": token_disagreements, "bf16_response_gate_status": item["bf16"]["response_gate"]["status"], "candidate_response_gate_status": item["candidate"]["response_gate"]["status"], "response_gate_outcome_match": gate_match})
    require(disagreements == result["disagreement_set"], "official disagreement set differs")
    eligible = exact_count == 9 and gate_count == 9
    require(result["eligibility"]["eligible"] == eligible, "official eligibility differs")
    reviewer = load_json(ATTEMPT_DIR / "fresh_reviewer_input.json")
    require(reviewer["official_attempt_regeneration_forbidden"] is True, "review packet permits regeneration")
    return {"status": result["status"], "candidate_id": CANDIDATE_ID, "attempt_count": 1, "eligible": eligible, "exact_sequence_matches": exact_count, "gate_outcome_matches": gate_count, "disagreement_case_ids": [item["case_id"] for item in disagreements], "selected_policy_id": None, "results_sha256": sha256_file(ATTEMPT_DIR / "results.json"), "reviewer_input_sha256": sha256_file(ATTEMPT_DIR / "fresh_reviewer_input.json"), "terminal_return_code": terminal["return_code"]}


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
            "regression": "repair only the isolated marker-free V5 runner and rerun pre-attempt preparation; official authorization remains unconsumed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "backend_sha256": sha256_file(BACKEND_PATH),
            "traceback": traceback.format_exc(),
            "attempt_authorization_consumed": False,
            "official_execution_marker_exists": False,
            "failed_at_utc": utc_now(),
        },
    )
