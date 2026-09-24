#!/usr/bin/env python3
"""Marker-free full-model preflight backend for the block-covariance V6 candidate."""

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
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_option_b_alias_safe_block_covariance_error_feedback_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_stage1 as reference
import run_option_b_alias_safe_joint_row_scale_tensor_codebook_w4_grouped_dynamic_scale32_stage1 as v4_reference
import run_option_b_alias_safe_tensor_codebook_w4_grouped_dynamic_scale32_stage1 as v3
import run_option_b_grouped_scale32_stage1 as grouped
import v4_joint_full_model_backend as v4_backend
import v5_calibration_weighted_full_model_backend as v5_backend
from ace2_quality_contracts import unpack_scale32
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
COVARIANCE_A_PATH = PREATTEMPT_DIR / "covariance_capture_a.json"
COVARIANCE_B_PATH = PREATTEMPT_DIR / "covariance_capture_b.json"
COVARIANCE_EQUALITY_PATH = PREATTEMPT_DIR / "covariance_capture_equality.json"
CANDIDATE_A_PATH = PREATTEMPT_DIR / "candidate_a_manifest.json"
CANDIDATE_B_PATH = PREATTEMPT_DIR / "candidate_b_manifest.json"
ALIAS_WITNESS_PATH = PREATTEMPT_DIR / "alias_separation_witness.json"
RECONSTRUCTION_PATH = PREATTEMPT_DIR / "model_only_block_covariance_reconstruction.json"
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

TORCH_THREADS = 32
MAX_NEW_TOKENS = 6
EXPECTED_EVENTS = grouped.LAYERS * len(grouped.EVENT_FAMILIES)
BLOCK_SIZE = reference.BLOCK_SIZE
OUTER_SWEEPS = reference.OUTER_SWEEPS
LIMB_BITS = 15
LIMB_MASK = (1 << LIMB_BITS) - 1
OBJECTIVE_ROW_CHUNK = 1024
OBJECTIVE_BLOCK_BATCH = 8
ASSIGNMENT_ROW_CHUNK = 2048
STATE_MATCH_KEYS = (
    "covariance_manifest_sha256",
    "embedding_bf16_sha256",
    "payload_manifest_sha256",
    "row_scale_manifest_sha256",
    "codebook_manifest_sha256",
    "reconstruction_manifest_sha256",
    "block_objective_manifest_sha256",
    "iteration_trace_manifest_sha256",
    "weight_manifest_sha256",
    "candidate_state_manifest_sha256",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def verify_companion(path: Path, companion: Path) -> None:
    expected = f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}"
    require(companion.is_file(), f"checksum companion is missing: {companion}")
    require(
        companion.read_text(encoding="utf-8").strip() == expected,
        f"checksum companion differs: {path}",
    )


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
        "backend": "exact BF16 block-Gram capture; deterministic block error-feedback proposals; exact Fraction fixtures",
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


def _configure_v3_helpers() -> None:
    v3.CANDIDATE_ID = CANDIDATE_ID
    v3.OUTPUT_DIR = OUTPUT_DIR
    v3.PREATTEMPT_DIR = PREATTEMPT_DIR
    v3.ATTEMPT_DIR = ATTEMPT_DIR
    v3.MARKER_PATH = MARKER_PATH
    v3.TORCH_THREADS = TORCH_THREADS


def _fraction_rows(source: Tensor) -> tuple[tuple[Fraction, ...], ...]:
    significands, shifts = v4_backend._decode_bf16(source)
    rows: list[tuple[Fraction, ...]] = []
    for sig_row, shift_row in zip(significands.tolist(), shifts.tolist(), strict=True):
        values: list[Fraction] = []
        for significand, shift in zip(sig_row, shift_row, strict=True):
            if shift >= 0:
                values.append(Fraction(int(significand) << int(shift)))
            else:
                values.append(Fraction(int(significand), 1 << -int(shift)))
        rows.append(tuple(values))
    return tuple(rows)


def _unpack_payload(payload: bytes, shape: Sequence[int]) -> Tensor:
    count = int(np.prod(shape))
    values: list[int] = []
    for byte in payload:
        values.extend((byte & 0x0F, (byte >> 4) & 0x0F))
    return torch.tensor(values[:count], dtype=torch.uint8).reshape(tuple(shape))


def _shifted_signed_limbs(coefficients: Tensor, deltas: Tensor, limb_count: int) -> Tensor:
    require(coefficients.dtype == deltas.dtype == torch.int64, "limb source dtype differs")
    require(bool(torch.all(deltas >= 0)), "negative limb shift")
    magnitude = torch.abs(coefficients)
    sign = torch.where(coefficients < 0, -1, 1)
    limbs: list[Tensor] = []
    for limb in range(limb_count):
        offset = limb * LIMB_BITS
        left = deltas - offset
        left_shift = torch.clamp(left, min=0, max=LIMB_BITS - 1)
        left_digit = torch.bitwise_left_shift(magnitude, left_shift) & LIMB_MASK
        left_digit = torch.where(left < LIMB_BITS, left_digit, torch.zeros_like(left_digit))
        right = offset - deltas
        right_shift = torch.clamp(right, min=0, max=62)
        right_digit = torch.bitwise_right_shift(magnitude, right_shift) & LIMB_MASK
        right_digit = torch.where(right < 63, right_digit, torch.zeros_like(right_digit))
        digit = torch.where(left >= 0, left_digit, right_digit)
        limbs.append(digit * sign)
    return torch.stack(limbs)


def _gram_records(values: Tensor, module_name: str) -> tuple[list[dict[str, Any]], tuple[tuple[tuple[int, ...], ...], ...]]:
    require(values.ndim == 2 and values.dtype == torch.bfloat16, "captured activation format differs")
    positions, width = values.shape
    significands, shifts = v4_backend._decode_bf16(values)
    records: list[dict[str, Any]] = []
    matrices: list[tuple[tuple[int, ...], ...]] = []
    for start in range(0, width, BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, width)
        sig = significands[:, start:stop]
        shift = shifts[:, start:stop]
        nonzero = sig != 0
        base = int(shift[nonzero].min()) if bool(torch.any(nonzero)) else 0
        delta = torch.where(nonzero, shift - base, torch.zeros_like(shift))
        span = int(delta.max()) if delta.numel() else 0
        require(span <= 96, f"captured activation lattice span is unsupported: {module_name}:{start}")
        limb_count = max(1, (8 + span + LIMB_BITS - 1) // LIMB_BITS)
        limbs = _shifted_signed_limbs(sig, delta, limb_count)
        require(
            positions * LIMB_MASK * LIMB_MASK < (1 << 62),
            "captured Gram limb reduction may overflow int64",
        )
        coefficients = torch.einsum("lpi,mpj->lmij", limbs, limbs)
        live = stop - start
        matrix: list[tuple[int, ...]] = []
        for left in range(live):
            row: list[int] = []
            for right in range(live):
                value = 0
                for limb_left in range(limb_count):
                    for limb_right in range(limb_count):
                        coefficient = int(coefficients[limb_left, limb_right, left, right])
                        value += coefficient << (LIMB_BITS * (limb_left + limb_right))
                row.append(value)
            matrix.append(tuple(row))
        normalized, divisor = reference.normalize_gram(tuple(matrix))
        gram_strings = [[str(value) for value in row] for row in normalized]
        records.append(
            {
                "start_channel": start,
                "live_channels": live,
                "activation_lattice_binary_exponent": base,
                "gram_lattice_binary_exponent": 2 * base,
                "whole_block_gcd": str(divisor),
                "normalized_gram": gram_strings,
                "normalized_gram_sha256": canonical_sha256(gram_strings),
            }
        )
        matrices.append(normalized)
    require(sum(record["live_channels"] for record in records) == width, "captured Gram partition differs")
    return records, tuple(matrices)


class _ActivationBuffer:
    def __init__(self, width: int):
        self.width = width
        self.parts: list[Tensor] = []
        self.position_count = 0

    def add(self, values: Tensor) -> None:
        require(values.dtype == torch.bfloat16 and values.shape[-1] == self.width, "captured Linear input format differs")
        flat = values.detach().reshape(-1, self.width).contiguous().clone()
        self.parts.append(flat)
        self.position_count += int(flat.shape[0])

    def finish(self, module_name: str) -> tuple[dict[str, Any], tuple[tuple[tuple[int, ...], ...], ...]]:
        require(self.parts and self.position_count > 0, f"no covariance positions captured: {module_name}")
        values = torch.cat(self.parts, dim=0)
        self.parts = []
        records, matrices = _gram_records(values, module_name)
        raw_hash = hashlib.sha256(values.contiguous().view(torch.uint8).numpy().tobytes(order="C")).hexdigest()
        body = {
            "module": module_name,
            "input_channels": self.width,
            "captured_position_count": self.position_count,
            "block_size": BLOCK_SIZE,
            "block_count": len(records),
            "source_bf16_sha256": raw_hash,
            "blocks": records,
            "floating_point_accumulation_used": False,
        }
        return {**body, "module_covariance_sha256": canonical_sha256(body)}, matrices


class _CalibrationCollector:
    def __init__(self, model: nn.Module):
        modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        require(len(modules) == 169, "calibration Linear count differs")
        self.buffers = {name: _ActivationBuffer(int(module.in_features)) for name, module in modules}
        self.mode = "all"
        self.handles = [module.register_forward_pre_hook(self._hook(name)) for name, module in modules]

    def _hook(self, name: str):
        def capture(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            require(bool(inputs) and isinstance(inputs[0], Tensor), f"Linear input missing: {name}")
            value = inputs[0]
            self.buffers[name].add(value if self.mode == "all" else value[..., -1:, :])

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def records(self) -> tuple[list[dict[str, Any]], dict[str, tuple[tuple[tuple[int, ...], ...], ...]]]:
        require(not self.handles, "calibration hooks remain installed")
        records: list[dict[str, Any]] = []
        matrices: dict[str, tuple[tuple[tuple[int, ...], ...], ...]] = {}
        for name, buffer in self.buffers.items():
            print(f"V6_COVARIANCE_FINALIZE {len(records) + 1}/169 {name}", flush=True)
            record, module_matrices = buffer.finish(name)
            records.append(record)
            matrices[name] = module_matrices
        return records, matrices


def _capture_covariance(model: nn.Module, tokenizer: Any) -> tuple[dict[str, Any], dict[str, tuple[tuple[tuple[int, ...], ...], ...]]]:
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
                    input_ids = torch.cat((input_ids, torch.tensor([[token]], dtype=torch.long)), dim=1)
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
    modules, matrices = collector.records()
    expected_positions = sum(item["captured_position_count"] for item in generated)
    require(all(item["captured_position_count"] == expected_positions for item in modules), "covariance position counts differ")
    body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "source_model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "calibration_contract": file_record(CALIBRATION_CONTRACT_PATH),
        "capture_arithmetic": "exact BF16 signed-significand block outer products on one dyadic integer lattice per contiguous block; whole-block GCD reduction only",
        "prompt_count": 4,
        "linear_count": len(modules),
        "block_size": BLOCK_SIZE,
        "generated_calibration": generated,
        "modules": modules,
        "official_inputs_accessed": False,
        "additional_holdout_inputs_accessed": False,
        "floating_point_statistic_accumulation_used": False,
    }
    return {**body, "covariance_manifest_sha256": canonical_sha256(body)}, matrices


def _float_gram(gram: Sequence[Sequence[int]]) -> Tensor:
    maximum = max(abs(int(value)) for row in gram for value in row) or 1
    return torch.tensor(
        [[float(Fraction(int(value), maximum)) for value in row] for row in gram],
        dtype=torch.float64,
    )


def _inverse_ratios(grams: Sequence[Sequence[Sequence[int]]]) -> list[Tensor]:
    ratios: list[Tensor] = []
    for gram in grams:
        matrix = _float_gram(gram)
        damping = float(Fraction(reference.damping_for_gram(gram), max(abs(int(value)) for row in gram for value in row) or 1))
        inverse = torch.linalg.inv(matrix + torch.eye(len(matrix), dtype=torch.float64) * damping)
        ratios.append(inverse / torch.diag(inverse)[:, None])
    return ratios


def _assign_error_feedback(source: Tensor, records: Tensor, codepoints: Sequence[int], grams: Sequence[Sequence[Sequence[int]]]) -> Tensor:
    rows, width = source.shape
    reference._block_ranges(width, grams)
    points = torch.tensor(reference.v4.validate_codebook(codepoints), dtype=torch.float64)
    midpoint_sums = points[:-1] + points[1:]
    assignments = torch.empty(source.shape, dtype=torch.uint8)
    ratios = _inverse_ratios(grams)
    start = 0
    for gram, ratio in zip(grams, ratios, strict=True):
        live = len(gram)
        stop = start + live
        for row_start in range(0, rows, ASSIGNMENT_ROW_CHUNK):
            row_stop = min(rows, row_start + ASSIGNMENT_ROW_CHUNK)
            scale_significands, scale_exponents = v4_backend._records_components(records[row_start:row_stop])
            scales = torch.ldexp(scale_significands.to(torch.float64), scale_exponents.to(torch.int32) - 15)
            compensated = source[row_start:row_stop, start:stop].to(torch.float64).clone()
            for channel in range(live):
                boundaries = scales[:, None] * (midpoint_sums[None, :] / 32.0)
                selected = torch.sum(compensated[:, channel, None] > boundaries, dim=1).to(torch.int64)
                assignments[row_start:row_stop, start + channel] = selected.to(torch.uint8)
                quantized = scales * points[selected] / 16.0
                error = compensated[:, channel] - quantized
                if channel + 1 < live:
                    compensated[:, channel + 1 :] -= error[:, None] * ratio[channel, channel + 1 :][None, :]
        start = stop
    return assignments


def _block_objective_float(source: Tensor, records: Tensor, codepoints: Sequence[int], assignments: Tensor, grams: Sequence[Sequence[Sequence[int]]]) -> float:
    rows, width = source.shape
    reference._block_ranges(width, grams)
    points = torch.tensor(reference.v4.validate_codebook(codepoints), dtype=torch.float64)
    total = 0.0
    start = 0
    for gram in grams:
        live = len(gram)
        stop = start + live
        matrix = torch.tensor([[float(value) for value in row] for row in gram], dtype=torch.float64)
        for row_start in range(0, rows, OBJECTIVE_ROW_CHUNK):
            row_stop = min(rows, row_start + OBJECTIVE_ROW_CHUNK)
            scale_significands, scale_exponents = v4_backend._records_components(records[row_start:row_stop])
            scales = torch.ldexp(scale_significands.to(torch.float64), scale_exponents.to(torch.int32) - 15)
            selected = points[assignments[row_start:row_stop, start:stop].to(torch.int64)]
            errors = source[row_start:row_stop, start:stop].to(torch.float64) - scales[:, None] * selected / 16.0
            total += float(torch.sum((errors @ matrix) * errors))
        start = stop
    require(math.isfinite(total) and total >= -1e-9, "block objective is not finite and nonnegative")
    return max(total, 0.0)


def _objective_record(value: float) -> dict[str, Any]:
    return {"binary64_hex": value.hex(), "decimal": format(value, ".17g")}


def _construct_block_state(source: Tensor, module_name: str, grams: Sequence[Sequence[Sequence[int]]]) -> dict[str, Any]:
    require(source.ndim == 2 and source.dtype == torch.bfloat16, f"source weight format differs: {module_name}")
    if source.numel() <= 4096:
        exact = reference.construct_block_candidate(_fraction_rows(source), grams)
        records = torch.tensor(exact["final_row_scale_records"], dtype=torch.int64)
        codebook = tuple(exact["final_codepoints"])
        payload = bytes.fromhex(exact["payload_hex"])
        assignments = _unpack_payload(payload, source.shape)
        fresh_v4 = v4_backend._construct_joint_state(source, module_name)
        baseline_metrics = v5_backend._decoded_metrics(
            source,
            fresh_v4["final_records"],
            fresh_v4["final_assignments"],
            fresh_v4["final_codepoints"],
        )
        return {
            "fresh_v4": fresh_v4,
            "fresh_v4_reconstruction": baseline_metrics,
            "final_records": records,
            "final_codepoints": codebook,
            "final_assignments": assignments,
            "payload": payload,
            "scale_raw": b"".join(struct.pack("<I", int(value)) for value in records.tolist()),
            "codebook_raw": bytes(point & 0xFF for point in codebook),
            "fresh_v4_block_objective": exact["fresh_v4_block_objective"],
            "final_block_objective": exact["final_block_objective"],
            "non_regressing": exact["non_regressing_against_fresh_v4"],
            "strict": exact["strictly_improved_against_fresh_v4"],
            "outer_sweeps_completed": exact["outer_sweeps_completed"],
            "iteration_trace": exact["iteration_trace"],
            "iteration_trace_sha256": exact["iteration_trace_sha256"],
            "full_model_assignment_backend": "exact Fraction fixture path",
        }

    fresh_v4 = v4_backend._construct_joint_state(source, module_name)
    records = fresh_v4["final_records"]
    codebook = fresh_v4["final_codepoints"]
    assignments = fresh_v4["final_assignments"]
    baseline_metrics = v5_backend._decoded_metrics(source, records, assignments, codebook)
    baseline = _block_objective_float(source, records, codebook, assignments, grams)
    previous = baseline
    trace: list[dict[str, Any]] = []
    for sweep in range(1, OUTER_SWEEPS + 1):
        proposal = _assign_error_feedback(source, records, codebook, grams)
        proposal_objective = _block_objective_float(source, records, codebook, proposal, grams)
        accepted = proposal_objective <= previous
        if accepted:
            assignments = proposal
            previous = proposal_objective
        trace.append(
            {
                "sweep": sweep,
                "prior_block_objective": _objective_record(baseline if sweep == 1 else float.fromhex(trace[-1]["final_block_objective"]["binary64_hex"])),
                "error_feedback_proposal_block_objective": _objective_record(proposal_objective),
                "proposal_accepted_by_monotonic_guard": accepted,
                "row_scale_records_retained": True,
                "codebook_retained": True,
                "final_block_objective": _objective_record(previous),
                "payload_sha256": hashlib.sha256(v3.pack_codebook_indices(assignments)).hexdigest(),
            }
        )
    payload = v3.pack_codebook_indices(assignments)
    scale_raw = b"".join(struct.pack("<I", int(value)) for value in records.tolist())
    codebook_raw = bytes(point & 0xFF for point in codebook)
    return {
        "fresh_v4": fresh_v4,
        "fresh_v4_reconstruction": baseline_metrics,
        "final_records": records,
        "final_codepoints": codebook,
        "final_assignments": assignments,
        "payload": payload,
        "scale_raw": scale_raw,
        "codebook_raw": codebook_raw,
        "fresh_v4_block_objective": _objective_record(baseline),
        "final_block_objective": _objective_record(previous),
        "non_regressing": previous <= baseline,
        "strict": previous < baseline,
        "outer_sweeps_completed": len(trace),
        "iteration_trace": trace,
        "iteration_trace_sha256": canonical_sha256(trace),
        "full_model_assignment_backend": "deterministic batched block inverse-Hessian proposal with exact-fixture conformance and monotonic objective guard",
    }


def _objective_value(record: dict[str, Any]) -> float:
    if "binary64_hex" in record:
        return float.fromhex(record["binary64_hex"])
    return float(Fraction(int(record["numerator"]), int(record["denominator"])))


def _quantize_model(model: nn.Module, covariance: dict[str, Any], grams: dict[str, tuple[tuple[tuple[int, ...], ...], ...]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {record["module"]: record for record in grouped.expected_transformer_tensors()}
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    require(len(modules) == 169, "Qwen linear count differs")
    require([name for name, _module in modules if name != "lm_head"] == list(expected), "transformer linear order differs")
    require(list(grams) == [name for name, _module in modules], "covariance module order differs")
    require(sha256_file(V4_RECONSTRUCTION_PATH) == V4_RECONSTRUCTION_SHA256, "V4 reconstruction reference hash differs")
    v4_report = load_json(V4_RECONSTRUCTION_PATH)
    v4_records = {item["module"]: item for item in v4_report["tensors"]}

    weight_manifest: list[dict[str, Any]] = []
    payload_manifest: list[dict[str, Any]] = []
    row_scale_manifest: list[dict[str, Any]] = []
    codebook_manifest: list[dict[str, Any]] = []
    reconstruction_manifest: list[dict[str, Any]] = []
    objective_manifest: list[dict[str, Any]] = []
    trace_manifest: list[dict[str, Any]] = []
    payload_stream = hashlib.sha256()
    scale_stream = hashlib.sha256()
    codebook_stream = hashlib.sha256()
    reconstruction_stream = hashlib.sha256()
    total_weight_count = total_payload_bytes = total_scale_bytes = total_codebook_bytes = 0

    for ordinal, (name, module) in enumerate(modules, start=1):
        started = time.monotonic()
        print(f"V6_CONSTRUCT_START {ordinal}/169 {name}", flush=True)
        source_sha = v3.tensor_sha256(module.weight)
        state = _construct_block_state(module.weight.detach(), name, grams[name])
        fresh_v4 = state["fresh_v4"]
        v4_record = v4_records[name]
        require(
            state["fresh_v4_reconstruction"]["reconstruction_bf16_sha256"]
            == v4_record["reconstruction_bf16_sha256"],
            f"fresh V4 reconstruction hash differs: {name}",
        )
        require(
            state["fresh_v4_reconstruction"]["unweighted_bf16_sse"]
            == v4_record["v4_joint_bf16_sse"],
            f"fresh V4 reconstruction SSE differs: {name}",
        )
        final_metrics = v5_backend._decoded_metrics(
            module.weight.detach(),
            state["final_records"],
            state["final_assignments"],
            state["final_codepoints"],
            module.weight,
        )
        payload = state["payload"]
        scale_raw = state["scale_raw"]
        codebook_raw = state["codebook_raw"]
        payload_record = {"module": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        scale_record = {"module": name, "record_count": int(module.weight.shape[0]), "bytes": len(scale_raw), "sha256": hashlib.sha256(scale_raw).hexdigest()}
        codebook_record = {"module": name, "bytes": 16, "sha256": hashlib.sha256(codebook_raw).hexdigest(), "q4_4_codepoints": list(state["final_codepoints"])}
        reconstruction_record = {
            "module": name,
            "weight_count": int(module.weight.numel()),
            "fresh_v4_unweighted_bf16_sse": state["fresh_v4_reconstruction"]["unweighted_bf16_sse"],
            "v6_unweighted_bf16_sse": final_metrics["unweighted_bf16_sse"],
            "fresh_v4_reconstruction_bf16_sha256": state["fresh_v4_reconstruction"]["reconstruction_bf16_sha256"],
            "v6_reconstruction_bf16_sha256": final_metrics["reconstruction_bf16_sha256"],
        }
        objective_record = {
            "module": name,
            "fresh_v4_block_objective": state["fresh_v4_block_objective"],
            "v6_block_objective": state["final_block_objective"],
            "v6_non_regressing_against_fresh_v4": bool(state["non_regressing"]),
            "v6_strictly_improved_against_fresh_v4": bool(state["strict"]),
        }
        trace_record = {
            "module": name,
            "source_bf16_sha256": source_sha,
            "fresh_v4_payload_sha256": hashlib.sha256(fresh_v4["payload"]).hexdigest(),
            "fresh_v4_row_scale_stream_sha256": hashlib.sha256(fresh_v4["scale_raw"]).hexdigest(),
            "fresh_v4_codebook_record_sha256": hashlib.sha256(fresh_v4["codebook_raw"]).hexdigest(),
            "outer_sweeps_completed": state["outer_sweeps_completed"],
            "assignment_backend": state["full_model_assignment_backend"],
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
            "block_objective": objective_record,
            "iteration_trace_sha256": state["iteration_trace_sha256"],
        }
        weight_manifest.append(record)
        payload_manifest.append(payload_record)
        row_scale_manifest.append(scale_record)
        codebook_manifest.append(codebook_record)
        reconstruction_manifest.append(reconstruction_record)
        objective_manifest.append(objective_record)
        trace_manifest.append(trace_record)
        payload_stream.update(name.encode() + b"\0" + payload)
        scale_stream.update(name.encode() + b"\0" + scale_raw)
        codebook_stream.update(name.encode() + b"\0" + codebook_raw)
        reconstruction_stream.update(name.encode() + b"\0" + bytes.fromhex(final_metrics["reconstruction_bf16_sha256"]))
        weight_count = int(module.weight.numel())
        total_weight_count += weight_count
        total_payload_bytes += len(payload)
        total_scale_bytes += len(scale_raw)
        total_codebook_bytes += len(codebook_raw)
        require(len(payload) == (weight_count + 1) // 2, f"packed payload byte count differs: {name}")
        require(len(scale_raw) == module.weight.shape[0] * 4, f"row-scale byte count differs: {name}")
        require(len(codebook_raw) == 16, f"codebook byte count differs: {name}")
        print(
            f"V6_CONSTRUCT_DONE {ordinal}/169 {name} seconds={time.monotonic() - started:.3f} "
            f"block_non_regressing={state['non_regressing']} block_strict={state['strict']}",
            flush=True,
        )
        del state, fresh_v4
        gc.collect()

    non_regressing_count = sum(bool(item["v6_non_regressing_against_fresh_v4"]) for item in objective_manifest)
    strict_count = sum(bool(item["v6_strictly_improved_against_fresh_v4"]) for item in objective_manifest)
    aggregate_baseline = math.fsum(_objective_value(item["fresh_v4_block_objective"]) for item in objective_manifest)
    aggregate_final = math.fsum(_objective_value(item["v6_block_objective"]) for item in objective_manifest)
    report_body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "measurement": "deterministic undamped normalized block-covariance reconstruction objective",
        "reference": {"candidate_id": v4_report["candidate_id"], "artifact": file_record(V4_RECONSTRUCTION_PATH), "freshly_reconstructed_not_imported": True},
        "covariance_manifest_sha256": covariance["covariance_manifest_sha256"],
        "tensor_count": len(objective_manifest),
        "per_tensor_non_regressing_count": non_regressing_count,
        "per_tensor_strictly_improved_count": strict_count,
        "all_169_tensors_non_regressing_against_fresh_v4": non_regressing_count == 169,
        "aggregate_fresh_v4_block_objective": _objective_record(aggregate_baseline),
        "aggregate_v6_block_objective": _objective_record(aggregate_final),
        "aggregate_strictly_improved_against_fresh_v4": aggregate_final < aggregate_baseline,
        "aggregate_fresh_v4_unweighted_bf16_sse": math.fsum(item["fresh_v4_unweighted_bf16_sse"] for item in reconstruction_manifest),
        "aggregate_v6_unweighted_bf16_sse": math.fsum(item["v6_unweighted_bf16_sse"] for item in reconstruction_manifest),
        "tensors": objective_manifest,
    }
    report = {
        **report_body,
        "status": (
            "PASS_169_OF_169_AND_STRICT_AGGREGATE_BLOCK_IMPROVEMENT"
            if report_body["all_169_tensors_non_regressing_against_fresh_v4"]
            and report_body["aggregate_strictly_improved_against_fresh_v4"]
            else "NO_GO_V4_RELATIVE_BLOCK_COVARIANCE_RECONSTRUCTION_GATE"
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
    require(len(transformer_records) == 168 and len(weight_manifest) == 169, "V6 tensor count differs")
    require(total_codebook_bytes == 2704, "whole-model codebook byte count differs")
    require(standard_layer_bits <= 4.036 and lm_head_bits <= 4.036, "stored-bit cap exceeded")
    manifests = {
        "payload": payload_manifest,
        "row_scale": row_scale_manifest,
        "codebook": codebook_manifest,
        "reconstruction": reconstruction_manifest,
        "block_objective": objective_manifest,
        "iteration_trace": trace_manifest,
    }
    hashes = {
        "payload_manifest_sha256": canonical_sha256(payload_manifest),
        "row_scale_manifest_sha256": canonical_sha256(row_scale_manifest),
        "codebook_manifest_sha256": canonical_sha256(codebook_manifest),
        "reconstruction_manifest_sha256": canonical_sha256(reconstruction_manifest),
        "block_objective_manifest_sha256": canonical_sha256(objective_manifest),
        "iteration_trace_manifest_sha256": canonical_sha256(trace_manifest),
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
        "covariance_product_bytes": 0,
    }
    return weight_manifest, {"manifests": manifests, "hashes": hashes, "storage": storage, "reconstruction": report}


def _load_candidate(label: str, tokenizer: Any) -> tuple[nn.Module, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    require(lm_head is embedding and lm_head.data_ptr() == embedding.data_ptr(), "source embedding/lm_head alias differs")
    source_shape = list(embedding.shape)
    source_dtype = str(embedding.dtype)
    source_object_id = id(embedding)
    source_pointer = embedding.data_ptr()
    source_hash = v3.tensor_sha256(embedding)
    require(source_hash == SOURCE_EMBEDDING_SHA256, "source embedding BF16 hash differs")

    covariance, grams = _capture_covariance(model, tokenizer)
    require(model.lm_head.weight is embedding, "covariance capture changed source tying")
    require(v3.tensor_sha256(embedding) == source_hash, "covariance capture changed embedding bytes")
    cloned = lm_head.detach().clone()
    model.lm_head.weight = nn.Parameter(cloned, requires_grad=False)
    model.config.tie_word_embeddings = False
    require(model.lm_head.weight is not embedding and model.lm_head.weight.data_ptr() != embedding.data_ptr(), "lm_head split failed")
    require(v3.tensor_sha256(model.lm_head.weight) == source_hash, "lm_head clone bytes differ")
    weight_manifest, construction = _quantize_model(model, covariance, grams)
    require(model.model.embed_tokens.weight is embedding and embedding.data_ptr() == source_pointer, "embedding identity changed")
    require(list(embedding.shape) == source_shape and str(embedding.dtype) == source_dtype, "embedding shape or dtype changed")
    embedding_hash = v3.tensor_sha256(embedding)
    require(embedding_hash == source_hash, "embedding bytes changed")
    state = v3.state_manifest(model)
    identity = {
        "covariance_manifest_sha256": covariance["covariance_manifest_sha256"],
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
        "block_objective_manifest": construction["manifests"]["block_objective"],
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
        "post_joint": {
            "embedding_bf16_sha256": embedding_hash,
            "embedding_hash_matches_source": embedding_hash == SOURCE_EMBEDDING_SHA256,
            "lm_head_storage_is_separate": model.lm_head.weight.data_ptr() != embedding.data_ptr(),
            "transformer_joint_count": 168,
            "lm_head_joint_count": 1,
        },
    }
    return model, covariance, manifest, witness, construction["reconstruction"]


def _compare_covariance(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    require(a == b, "independent covariance captures differ")
    require(a["linear_count"] == 169 and len(a["modules"]) == 169, "covariance module count differs")
    block_count = sum(item["block_count"] for item in a["modules"])
    return {
        "byte_identical": canonical_bytes(a) == canonical_bytes(b),
        "manifest_sha256": a["covariance_manifest_sha256"],
        "linear_count": 169,
        "block_count": block_count,
        "prompt_count": 4,
        "generated_token_arrays_match": True,
        "position_counts_match": True,
        "full_normalized_gram_blocks_match": True,
    }


def _compare_candidates(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    comparisons = {key: a["identity"][key] == b["identity"][key] for key in STATE_MATCH_KEYS}
    require(all(comparisons.values()), f"candidate A/B identity differs: {comparisons}")
    fields = (
        "payload_manifest",
        "row_scale_manifest",
        "codebook_manifest",
        "reconstruction_manifest",
        "block_objective_manifest",
        "iteration_trace_manifest",
        "weight_manifest",
        "state_manifest",
    )
    field_matches = {field: a[field] == b[field] for field in fields}
    require(all(field_matches.values()), f"candidate A/B manifests differ: {field_matches}")
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
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "frozen V6 Engineer authority differs")
    root_existed_before = OUTPUT_DIR.exists()
    marker_existed_before = MARKER_PATH.exists()
    require(not marker_existed_before, "V6 execution marker already exists")
    exact_reference = reference.self_test()
    fixture = reference.v4.faithful_fixture()
    source = torch.tensor([[float(value) for value in row] for row in fixture], dtype=torch.bfloat16)
    grams = reference._fixture_grams(source.shape[1])
    scalable_a = _construct_block_state(source, "scalable_block_fixture_a", grams)
    scalable_b = _construct_block_state(source, "scalable_block_fixture_b", grams)
    expected = exact_reference["candidate_fixture"]
    require(scalable_a["payload"] == scalable_b["payload"], "independent scalable V6 payloads differ")
    require(scalable_a["scale_raw"] == scalable_b["scale_raw"], "independent scalable V6 row scales differ")
    require(scalable_a["codebook_raw"] == scalable_b["codebook_raw"], "independent scalable V6 codebooks differ")
    require(hashlib.sha256(scalable_a["payload"]).hexdigest() == expected["payload_sha256"], "scalable V6 payload differs from Fraction reference")
    require(hashlib.sha256(scalable_a["scale_raw"]).hexdigest() == expected["row_scale_stream_sha256"], "scalable V6 row scales differ from Fraction reference")
    require(hashlib.sha256(scalable_a["codebook_raw"]).hexdigest() == expected["codebook_record_sha256"], "scalable V6 codebook differs from Fraction reference")
    require(OUTPUT_DIR.exists() == root_existed_before, "combined self-test changed the V6 namespace")
    require(MARKER_PATH.exists() == marker_existed_before is False, "combined self-test changed the V6 marker")
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
            "exact_fraction_reference_self_test_passed": exact_reference["status"] == "PASS",
            "scalable_fixture_matches_fraction_payload": True,
            "scalable_fixture_matches_fraction_row_scales": True,
            "scalable_fixture_matches_fraction_codebook": True,
            "independent_scalable_fixture_a_b_match": True,
            "official_execution_marker_exists": False,
        },
        "official_build_namespace_changed": False,
        "official_execution_marker_exists": False,
    }


def _validate_reconstruction(report: dict[str, Any]) -> bool:
    require(report["tensor_count"] == len(report["tensors"]) == 169, "V6 tensor count differs")
    non_regressing = sum(bool(item["v6_non_regressing_against_fresh_v4"]) for item in report["tensors"])
    strict = sum(bool(item["v6_strictly_improved_against_fresh_v4"]) for item in report["tensors"])
    require(non_regressing == report["per_tensor_non_regressing_count"], "V6 non-regression aggregate differs")
    require(strict == report["per_tensor_strictly_improved_count"], "V6 strict aggregate differs")
    baseline = _objective_value(report["aggregate_fresh_v4_block_objective"])
    final = _objective_value(report["aggregate_v6_block_objective"])
    passed = non_regressing == 169 and final < baseline and report["aggregate_strictly_improved_against_fresh_v4"] is True
    require(report["status"].startswith("PASS_") == passed, "V6 reconstruction status differs")
    body = {key: value for key, value in report.items() if key not in {"status", "reconstruction_report_sha256"}}
    require(report["reconstruction_report_sha256"] == canonical_sha256(body), "V6 reconstruction hash differs")
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
        require(not OUTPUT_DIR.exists(), "V6 namespace existed before preparation")
    require(not matches and not attempts, "V6 official attempt namespace is consumed")
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
    require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "cannot recover a consumed V6 namespace")
    complete = PREATTEMPT_DIR.is_dir() and CONTRACT_PATH.is_file() and (READY_PATH.is_file() or NO_GO_PATH.is_file())
    require(not complete, "V6 pre-attempt closure already exists; use verify")
    candidates = [path for path in OUTPUT_DIR.iterdir() if path.name != "preflight-failures"]
    require(candidates, "existing V6 namespace has ambiguous provenance")
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
        "covariance_capture_a": COVARIANCE_A_PATH,
        "covariance_capture_b": COVARIANCE_B_PATH,
        "covariance_capture_equality": COVARIANCE_EQUALITY_PATH,
        "candidate_a_manifest": CANDIDATE_A_PATH,
        "candidate_b_manifest": CANDIDATE_B_PATH,
        "alias_separation_witness": ALIAS_WITNESS_PATH,
        "model_only_block_covariance_reconstruction": RECONSTRUCTION_PATH,
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


def _build_contract(environment: dict[str, Any], source_identity: dict[str, Any], namespace_audit: dict[str, Any], covariance_equality: dict[str, Any], state_comparison: dict[str, Any], gate_passes: dict[str, bool]) -> dict[str, Any]:
    task = load_json(TASK_PATH)
    require(task["task_id"] == TASK_ID and task["candidate_id"] == CANDIDATE_ID, "V6 Engineer authority differs")
    require(task["fresh_attempt_namespace"]["attempt_budget"] == 1, "V6 attempt budget differs")
    require(task["fresh_attempt_namespace"]["attempts_consumed_at_task_freeze"] == 0, "V6 attempt was consumed at freeze")
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
        "authority": {"engineer_task": artifacts["engineer_task"], "planning_requirements": artifacts["planning_requirements"], "attempt_id": "attempt-0001", "exact_authorized_candidate_attempts": 1, "attempts_consumed": 0, "selected_policy_id": None},
        "source_model": source_identity,
        "environment": environment,
        "construction": {
            "backend": "exact block-Gram capture plus deterministic four-sweep block error-feedback construction",
            "fresh_v4_initialization": True,
            "covariance_duplicate_capture": covariance_equality,
            "source_tied_then_runtime_split_lm_head": True,
            "source_embedding_bf16_sha256": SOURCE_EMBEDDING_SHA256,
            "embedding_bytes_preserved": True,
            "candidate_a_and_b_whole_model_manifests_match": state_comparison,
            "covariance_runtime_or_product_bytes": 0,
        },
        "pre_attempt_gates": {
            "duplicate_covariance_capture": {"passed": gate_passes["duplicate_covariance_capture"], "artifact": artifacts["covariance_capture_equality"]},
            "v4_relative_block_model_169_of_169": {"passed": gate_passes["v4_relative_block_model_169_of_169"], "artifact": artifacts["model_only_block_covariance_reconstruction"]},
            "non_evaluator_quality_8_of_8": {"passed": gate_passes["non_evaluator_quality_8_of_8"], "artifact": artifacts["non_evaluator_quality"]},
            "dynamic_312_event_closure": {"passed": gate_passes["dynamic_312_event_closure"], "artifact": artifacts["dynamic_preflight"]},
            "two_build_alias_and_manifest_closure": {"passed": gate_passes["two_build_alias_and_manifest_closure"], "artifact": artifacts["alias_separation_witness"]},
            "all_pass": all_pass,
        },
        "official_matrix_binding": {"artifact": artifacts["official_matrix_hash_only"], "sha256": MATRIX_SHA256, "prompt_count": 9, "official_input_content_accessed_during_preflight": False},
        "official_execution": {"mode_exposed_by_runner": False, "attempt_directory_exists": False, "execution_marker_exists": False, "conditional_execution_rule": "a later authorized runner revision may consume attempt-0001 only after this PREATTEMPT_READY closure verifies"},
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
    require(sha256_file(path) == record["sha256"], f"bound artifact hash differs: {record['path']}")


def load_verified_closure(require_unconsumed: bool = True) -> dict[str, Any]:
    _configure_v3_helpers()
    require_project_python()
    require(CONTRACT_PATH.is_file(), "V6 predeclared contract is missing")
    require(READY_PATH.is_file() != NO_GO_PATH.is_file(), "exactly one V6 pre-attempt terminal record is required")
    wrapper = load_json(CONTRACT_PATH)
    require(wrapper["contract_sha256"] == canonical_sha256(wrapper["contract"]), "V6 contract canonical hash differs")
    contract = wrapper["contract"]
    require(contract["candidate_id"] == CANDIDATE_ID and contract["task_id"] == TASK_ID, "V6 predeclared authority differs")
    require(contract["artifacts"]["runner_source"]["sha256"] == sha256_file(RUNNER_PATH), "V6 runner changed after freeze")
    require(contract["artifacts"]["scalable_backend_source"]["sha256"] == sha256_file(BACKEND_PATH), "V6 backend changed after freeze")
    for record in contract["artifacts"].values():
        _verify_bound_record(record)
    verify_companion(TASK_PATH, TASK_COMPANION)
    verify_companion(PLAN_PATH, PLAN_COMPANION)
    require(sha256_file(MATRIX_PATH) == MATRIX_SHA256, "official matrix hash differs")
    require(verify_source_snapshot() == contract["source_model"], "source snapshot identity differs")
    packages = {**verify_versions(), "numpy": importlib.metadata.version("numpy")}
    require(packages == contract["environment"]["packages"], "package versions differ")
    require(combined_self_test() == load_json(SELF_TEST_PATH), "V6 runner self-test result differs")
    covariance_equality = _compare_covariance(load_json(COVARIANCE_A_PATH), load_json(COVARIANCE_B_PATH))
    require(covariance_equality == load_json(COVARIANCE_EQUALITY_PATH), "V6 covariance equality record differs")
    comparison = _compare_candidates(load_json(CANDIDATE_A_PATH), load_json(CANDIDATE_B_PATH))
    witness = load_json(ALIAS_WITNESS_PATH)
    require(witness["state_comparison"] == comparison, "V6 alias witness comparison differs")
    reconstruction = load_json(RECONSTRUCTION_PATH)
    quality = load_json(QUALITY_PATH)
    dynamic = load_json(DYNAMIC_PATH)
    gate_passes = {
        "duplicate_covariance_capture": covariance_equality["byte_identical"],
        "v4_relative_block_model_169_of_169": _validate_reconstruction(reconstruction),
        "non_evaluator_quality_8_of_8": _validate_quality(quality),
        "dynamic_312_event_closure": _validate_dynamic(dynamic),
        "two_build_alias_and_manifest_closure": comparison["all_match"] and witness["embedding_preserved_in_both"] and witness["lm_head_separate_in_both"],
    }
    require(contract["pre_attempt_gates"]["all_pass"] == all(gate_passes.values()), "V6 aggregate gate status differs")
    for name, passed in gate_passes.items():
        require(contract["pre_attempt_gates"][name]["passed"] == passed, f"V6 contract gate differs: {name}")
    terminal_path = READY_PATH if READY_PATH.is_file() else NO_GO_PATH
    terminal = load_json(terminal_path)
    require(terminal["candidate_id"] == CANDIDATE_ID, "V6 terminal candidate differs")
    require(terminal["contract"]["sha256"] == sha256_file(CONTRACT_PATH), "V6 terminal contract file hash differs")
    require(terminal["contract_sha256"] == wrapper["contract_sha256"], "V6 terminal canonical hash differs")
    require(terminal["bound_artifacts"] == contract["artifacts"], "V6 terminal artifact snapshot differs")
    if terminal_path == READY_PATH:
        require(terminal["status"] == "PREATTEMPT_READY" and all(gate_passes.values()), "V6 ready exists without all gates passing")
    else:
        require(terminal["status"] == "PREATTEMPT_NO_GO" and not all(gate_passes.values()), "V6 no-go does not preserve a failed gate")
    if require_unconsumed:
        _audit_attempt_namespace(expect_root_absent=False)
        require(not ATTEMPT_DIR.exists() and not MARKER_PATH.exists(), "V6 official attempt already exists")
    return {
        "status": terminal["status"],
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": wrapper["contract_sha256"],
        "gate_passes": gate_passes,
        "duplicate_covariance_match": covariance_equality["byte_identical"],
        "covariance_block_count": covariance_equality["block_count"],
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
    require(probe.read_bytes() == b"preflight\n", "V6 output-path readback differs")
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
    print("V6_CANDIDATE_A_START covariance_and_construction", flush=True)
    candidate_a, covariance_a, manifest_a, witness_a, reconstruction_a = _load_candidate("candidate_a_independent_covariance_and_construction", tokenizer)
    chronology["candidate_a_complete_at_utc"] = utc_now()
    print("V6_CANDIDATE_A_DONE", flush=True)
    del candidate_a
    gc.collect()

    chronology["candidate_b_started_at_utc"] = utc_now()
    print("V6_CANDIDATE_B_START covariance_construction_quality_dynamic", flush=True)
    candidate_b, covariance_b, manifest_b, witness_b, reconstruction_b = _load_candidate("candidate_b_independent_covariance_construction_quality_and_dynamic", tokenizer)
    covariance_equality = _compare_covariance(covariance_a, covariance_b)
    state_comparison = _compare_candidates(manifest_a, manifest_b)
    require(reconstruction_a == reconstruction_b, "V6 candidate A/B reconstruction reports differ")
    chronology["candidate_b_constructed_at_utc"] = utc_now()
    quality, dynamic = v3.non_evaluator_quality_and_dynamic(candidate_b, tokenizer, inputs, input_manifest)
    chronology["candidate_b_quality_and_dynamic_complete_at_utc"] = utc_now()
    print("V6_QUALITY_DYNAMIC_DONE", flush=True)
    del candidate_b
    gc.collect()

    gate_passes = {
        "duplicate_covariance_capture": covariance_equality["byte_identical"],
        "v4_relative_block_model_169_of_169": _validate_reconstruction(reconstruction_b),
        "non_evaluator_quality_8_of_8": _validate_quality(quality),
        "dynamic_312_event_closure": _validate_dynamic(dynamic),
        "two_build_alias_and_manifest_closure": state_comparison["all_match"] and witness_a["post_joint"]["embedding_hash_matches_source"] and witness_b["post_joint"]["embedding_hash_matches_source"] and witness_a["post_joint"]["lm_head_storage_is_separate"] and witness_b["post_joint"]["lm_head_storage_is_separate"],
    }
    require(not MARKER_PATH.exists() and not ATTEMPT_DIR.exists(), "V6 official attempt appeared before artifact freeze")
    PREATTEMPT_DIR.mkdir(parents=False, exist_ok=False)
    write_json(INPUT_MANIFEST_PATH, input_manifest)
    write_json(COVARIANCE_A_PATH, covariance_a)
    write_json(COVARIANCE_B_PATH, covariance_b)
    write_json(COVARIANCE_EQUALITY_PATH, covariance_equality)
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
    wrapper = _build_contract(environment, source_identity, namespace_audit, covariance_equality, state_comparison, gate_passes)
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
            "all marker-free V6 gates pass; PREATTEMPT_READY is sealed without consuming attempt-0001"
            if all_pass
            else "preserve V6 no-go evidence, leave attempt-0001 absent and unconsumed, keep selected_policy_id null, and submit for Fresh Review"
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
            "regression": "repair only the isolated marker-free V6 runner and rerun pre-attempt preparation; official authorization remains unconsumed",
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
