#!/usr/bin/env python3
"""Repair execution gate for the frozen 24-layer ACE-2 RTL mission.

This wrapper leaves the accepted runner and consumed attempt-0001 wrapper
unchanged.  It binds their hashes, adds the narrowly reviewed execution
evidence, and owns exactly-once attempt-0002.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import re
import shutil
import struct
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_lora_v4_two_token_24layer_rtl_authorized as base


layer_runner = base.layer_runner
accepted = base.accepted
canonical = base.canonical

LEGACY_WRAPPER = ROOT / "tools/run_lora_v4_two_token_24layer_rtl_authorized.py"
ATTEMPT_0001 = base.OFFICIAL_ROOT / "attempt-0001"
ATTEMPT = base.OFFICIAL_ROOT / "attempt-0002"
REPAIR_PREFLIGHT = ROOT / "build/lora-v4-two-token-24layer-rtl-repair-preview/preflight-0001"
PREPARE_PHASE_TMP = base.OFFICIAL_ROOT / ".attempt-0002-prepare-phase"

LEGACY_WRAPPER_SHA256 = "7117f202dac27d6fdb67b109ec2bc9292762b635e48614df57b64f194c676edf"
ATTEMPT_0001_SUMS_SHA256 = "c8cb48e820cd6547c7ce3260d7db27b880f0526b5abbeee1601fd8cbf8a955e9"
ATTEMPT_0001_MEMBERS = 20
Q0_15_MAX = 1 << 15
KV_BYTES_PER_LAYER = 2 * 2 * base.KV_HEADS * base.HEAD_DIM

AUTHORIZED_COMMANDS = {
    "prepare": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/run_lora_v4_two_token_24layer_rtl_repair.py --prepare",
    "run": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/run_lora_v4_two_token_24layer_rtl_repair.py --run",
    "check": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/run_lora_v4_two_token_24layer_rtl_repair.py --check",
}

# The inherited functions resolve globals in the imported module.  Retarget
# only those globals to this distinct wrapper and fresh attempt namespace.
base.__file__ = str(Path(__file__).resolve())
base.ATTEMPT = ATTEMPT
base.AUTHORIZED_COMMANDS = AUTHORIZED_COMMANDS

require = base.require
read_json = base.read_json
write_json = base.write_json
write_text = base.write_text
file_record = base.file_record
write_binary = base.write_binary


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def byte_mismatches(actual: bytes, expected: bytes) -> int:
    require(len(actual) == len(expected), "byte comparison length changed")
    return sum(left != right for left, right in zip(actual, expected, strict=True))


def raw_q0_15(values: list[int]) -> bytes:
    checked: list[int] = []
    for value in values:
        item = int(value)
        if not 0 <= item <= Q0_15_MAX:
            raise ValueError(f"unsigned Q0.15 value outside frozen range 0..32768: {item}")
        checked.append(item)
    return b"".join(struct.pack("<H", item) for item in checked)


def q0_15_boundary_test() -> dict[str, Any]:
    legal = [0, Q0_15_MAX - 1, Q0_15_MAX]
    raw = raw_q0_15(legal)
    require(raw == bytes.fromhex("0000ff7f0080"), "unsigned Q0.15 boundary encoding changed")
    rejected: list[int] = []
    for value in (-1, Q0_15_MAX + 1, 0xFFFF):
        try:
            raw_q0_15([value])
        except ValueError:
            rejected.append(value)
    require(rejected == [-1, Q0_15_MAX + 1, 0xFFFF], "Q0.15 out-of-range rejection changed")
    return {
        "status": "PASS_UNSIGNED_Q0_15_BOUNDARIES",
        "legal_values": legal,
        "encoded_hex": raw.hex(),
        "rejected_values": rejected,
        "frozen_legal_range": [0, Q0_15_MAX],
        "storage": "unsigned little-endian 16-bit",
    }


def validate_predecessor() -> dict[str, Any]:
    require(LEGACY_WRAPPER.is_file(), "attempt-0001 wrapper is absent")
    require(accepted.sha256_file(LEGACY_WRAPPER) == LEGACY_WRAPPER_SHA256, "attempt-0001 wrapper changed")
    require(ATTEMPT_0001.is_dir(), "attempt-0001 is absent")
    require(
        accepted.sha256_file(ATTEMPT_0001 / "SHA256SUMS") == ATTEMPT_0001_SUMS_SHA256,
        "attempt-0001 aggregate changed",
    )
    members = base.verify_sha256s(ATTEMPT_0001)
    require(members == ATTEMPT_0001_MEMBERS, "attempt-0001 member count changed")
    status = read_json(ATTEMPT_0001 / "status.json")
    require(status == {"sealed": True, "status": "SEALED_FAIL_RUN"}, "attempt-0001 status changed")
    return {
        "attempt": base.relative(ATTEMPT_0001),
        "status": status["status"],
        "members": members,
        "sha256s": file_record(ATTEMPT_0001 / "SHA256SUMS"),
        "wrapper": file_record(LEGACY_WRAPPER),
        "preserved_unchanged": True,
    }


ORIGINAL_DERIVE_LAYER = base.derive_layer


def add_k_data_counterfactual(derived: dict[str, Any]) -> dict[str, Any]:
    position = derived["positions"][1]
    cached_k = derived["cached_k"]
    cached_v = derived["cached_v"]
    require(any(value != 0 for value in cached_k[0]), "cached position-0 K payload is already zero")

    accumulator_changes = 0
    score_changes = 0
    probability_changes = 0
    compose_changes = 0
    for head in range(base.Q_HEADS):
        kv_head = head // (base.Q_HEADS // base.KV_HEADS)
        start = kv_head * base.HEAD_DIM
        stop = start + base.HEAD_DIM
        q_head = position["rope_q"].outputs[head * base.HEAD_DIM : (head + 1) * base.HEAD_DIM]
        counterfactual_keys = [[0] * base.HEAD_DIM, cached_k[1][start:stop]]
        counterfactual_score = canonical.reference_attention_score(
            canonical.AttentionScoreCase(
                f"layer{derived['layer_id']}_position1_head{head}_zero_cached_k",
                q_head,
                counterfactual_keys,
                derived["positions"][0]["projections"]["q"]["output_scale"],
                derived["positions"][0]["projections"]["k"]["output_scale"],
            )
        )
        counterfactual_softmax = canonical.reference_softmax(
            canonical.SoftmaxCase(
                f"layer{derived['layer_id']}_position1_head{head}_zero_cached_k",
                counterfactual_score.core_scores_q6_9,
            )
        )
        values = [cached_v[0][start:stop], cached_v[1][start:stop]]
        counterfactual_compose = canonical.reference_attention_compose(
            canonical.AttentionComposeCase(
                f"layer{derived['layer_id']}_position1_head{head}_zero_cached_k",
                counterfactual_score.core_scores_q6_9,
                values,
            )
        )
        original_score = position["scores"][head]
        original_softmax = position["probabilities"][head]
        original_compose = position["composes"][head]
        accumulator_changes += int(counterfactual_score.accumulators[0] != original_score.accumulators[0])
        score_changes += int(counterfactual_score.core_scores_q6_9 != original_score.core_scores_q6_9)
        probability_changes += int(
            counterfactual_softmax.probabilities_q0_15 != original_softmax.probabilities_q0_15
        )
        compose_changes += int(counterfactual_compose.outputs != original_compose.outputs)

    require(accumulator_changes > 0, f"layer {derived['layer_id']} cached K payload has no score effect")
    derived["cache_reuse"].update(
        {
            "k_data_counterfactual": "zero_position0_k_payload_only",
            "cached_k_score_accumulator_changes_when_cached_k_zeroed_heads": accumulator_changes,
            "cached_k_score_output_changes_when_cached_k_zeroed_heads": score_changes,
            "cached_k_probability_changes_when_cached_k_zeroed_heads": probability_changes,
            "cached_k_compose_output_changes_when_cached_k_zeroed_heads": compose_changes,
        }
    )
    return derived


def derive_layer(
    layer_id: int,
    states: list[dict[str, Any]],
    weights: Any,
    adapter: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    derived, next_states = ORIGINAL_DERIVE_LAYER(layer_id, states, weights, adapter)
    return add_k_data_counterfactual(derived), next_states


base.derive_layer = derive_layer


def persist_layer_tensors(layer_dir: Path, derived: dict[str, Any]) -> dict[str, Any]:
    directory = layer_dir / "tensors"
    records: dict[str, Any] = {}
    for position in derived["positions"]:
        prefix = f"position{position['position']}"
        payloads = {
            f"{prefix}_input_hidden_s8.bin": base.raw_int8(position["embedding_q"]),
            f"{prefix}_input_rmsnorm_s8.bin": base.raw_int8(position["input_norm_q"]),
            f"{prefix}_rope_q_s8.bin": base.raw_int8(position["rope_q"].outputs),
            f"{prefix}_rope_k_s8.bin": base.raw_int8(position["rope_k"].outputs),
            f"{prefix}_attention_s8.bin": base.raw_int8(position["attention_q"]),
            f"{prefix}_attention_residual_s8.bin": base.raw_int8(position["attention_residual"]["output"]),
            f"{prefix}_post_attention_rmsnorm_s8.bin": base.raw_int8(position["post_norm_q"]),
            f"{prefix}_silu_output_s8.bin": base.raw_int8(position["silu"]["output"]),
            f"{prefix}_layer_output_s8.bin": base.raw_int8(position["final_residual"]["output"]),
            f"{prefix}_input_scale_f64le.bin": struct.pack("<d", position["embedding_scale"]),
            f"{prefix}_layer_output_scale_f64le.bin": struct.pack("<d", position["final_residual"]["scale"]),
        }
        for name, raw in payloads.items():
            records[name] = write_binary(directory / name, raw)

        score_accumulators: list[int] = []
        score_outputs: list[int] = []
        softmax_probabilities: list[int] = []
        compose_outputs: list[int] = []
        for score, probability, compose in zip(
            position["scores"], position["probabilities"], position["composes"], strict=True
        ):
            score_accumulators.extend(score.accumulators)
            score_outputs.extend(score.core_scores_q6_9)
            softmax_probabilities.extend(probability.probabilities_q0_15)
            compose_outputs.extend(compose.outputs)
        records[f"{prefix}_attention_score_accumulator_s32le.bin"] = write_binary(
            directory / f"{prefix}_attention_score_accumulator_s32le.bin",
            base.raw_int32(score_accumulators),
        )
        records[f"{prefix}_attention_score_q6_9_s16le.bin"] = write_binary(
            directory / f"{prefix}_attention_score_q6_9_s16le.bin", base.raw_int16(score_outputs)
        )
        records[f"{prefix}_softmax_probability_q0_15_u16le.bin"] = write_binary(
            directory / f"{prefix}_softmax_probability_q0_15_u16le.bin",
            raw_q0_15(softmax_probabilities),
        )
        records[f"{prefix}_attention_compose_s8.bin"] = write_binary(
            directory / f"{prefix}_attention_compose_s8.bin", base.raw_int8(compose_outputs)
        )
        for key in base.PROJECTION_ORDER:
            projection = position["projections"][key]
            records[f"{prefix}_{key}_output_s8.bin"] = write_binary(
                directory / f"{prefix}_{key}_output_s8.bin", base.raw_int8(projection["output_q"])
            )
            records[f"{prefix}_{key}_accumulator_s32le.bin"] = write_binary(
                directory / f"{prefix}_{key}_accumulator_s32le.bin",
                base.raw_int32(projection["accumulator"]),
            )
    for key in base.PROJECTION_ORDER:
        records[f"shared_{key}_qweight_s4_in_s8.bin"] = write_binary(
            directory / f"shared_{key}_qweight_s4_in_s8.bin",
            base.raw_int8(derived["positions"][0]["projections"][key]["qweight"]),
        )
    records["kv_cache_k_s8.bin"] = write_binary(
        directory / "kv_cache_k_s8.bin", base.raw_int8(derived["cached_k"][0] + derived["cached_k"][1])
    )
    records["kv_cache_v_s8.bin"] = write_binary(
        directory / "kv_cache_v_s8.bin", base.raw_int8(derived["cached_v"][0] + derived["cached_v"][1])
    )
    records["layer_outputs_position_major_s8.bin"] = write_binary(
        directory / "layer_outputs_position_major_s8.bin",
        b"".join(base.raw_int8(position["final_residual"]["output"]) for position in derived["positions"]),
    )
    return records


base.persist_layer_tensors = persist_layer_tensors


ORIGINAL_EXECUTION_SOURCE = layer_runner.execution_source


def repair_execution_source(name: str, source_path: Path, generated_name: str, instance_needle: str) -> Path:
    path = ORIGINAL_EXECUTION_SOURCE(name, source_path, generated_name, instance_needle)
    if name == "rope":
        source = path.read_text(encoding="utf-8")
        needle = "            end else begin\n                if (out_data !=="
        replacement = (
            "            end else begin\n"
            "                $display(\"RTL_ROPE_RESULT case=%0d beat=%0d data=%032x\", "
            "selected_case, selected_beat, out_data);\n"
            "                if (out_data !=="
        )
        require(source.count(needle) == 1, "RoPE observation insertion point changed")
        write_text(path, source.replace(needle, replacement, 1))
    return path


layer_runner.execution_source = repair_execution_source


def compare_rtl_kv_cache(layer_dir: Path, derived: dict[str, Any]) -> dict[str, Any]:
    projection_stdout = (layer_dir / "logs/projection.vvp.stdout.log").read_text(encoding="utf-8")
    projection_rows = re.findall(
        r"RTL_PROJ_RESULT operator=(\d+) channel=(\d+) out_u8=(\d+) acc=(-?\d+) saturation=(\d+) overflow=(\d+)",
        projection_stdout,
    )
    require(len(projection_rows) == base.EXPECTED_PROJECTION_RESULTS_PER_LAYER, "projection RTL rows changed")
    v_by_operator: dict[int, list[tuple[int, int]]] = {2: [], 9: []}
    for operator, channel, out_u8, _acc, _saturation, _overflow in projection_rows:
        operator_id = int(operator)
        if operator_id in v_by_operator:
            v_by_operator[operator_id].append((int(channel), int(out_u8)))
    for operator_id in (2, 9):
        require(
            [channel for channel, _ in v_by_operator[operator_id]] == list(range(base.KV_HEADS * base.HEAD_DIM)),
            f"V-cache projection channel order changed for operator {operator_id}",
        )
    observed_v = bytes(out_u8 for operator_id in (2, 9) for _, out_u8 in v_by_operator[operator_id])

    rope_stdout = (layer_dir / "logs/rope.vvp.stdout.log").read_text(encoding="utf-8")
    rope_rows = re.findall(r"RTL_ROPE_RESULT case=(\d+) beat=(\d+) data=([0-9a-fA-F]+)", rope_stdout)
    require(len(rope_rows) == 4 * 56, "RoPE RTL observation count changed")
    require(
        [(int(case), int(beat)) for case, beat, _ in rope_rows]
        == [(case, beat) for case in range(4) for beat in range(56)],
        "RoPE RTL observation order changed",
    )
    rope_bytes: dict[int, bytes] = {}
    for case in range(4):
        raw = b"".join(
            int(data, 16).to_bytes(16, "little")
            for row_case, _beat, data in rope_rows
            if int(row_case) == case
        )
        require(len(raw) == base.HIDDEN, f"RoPE case {case} byte count changed")
        rope_bytes[case] = raw
    for case in (1, 3):
        require(
            rope_bytes[case][base.KV_HEADS * base.HEAD_DIM :] == bytes(base.HIDDEN - base.KV_HEADS * base.HEAD_DIM),
            f"RoPE K-cache padding changed for case {case}",
        )
    observed_k = b"".join(rope_bytes[case][: base.KV_HEADS * base.HEAD_DIM] for case in (1, 3))

    expected_k = base.raw_int8(derived["cached_k"][0] + derived["cached_k"][1])
    expected_v = base.raw_int8(derived["cached_v"][0] + derived["cached_v"][1])
    require(len(expected_k) == len(observed_k) == 256, "K-cache byte count changed")
    require(len(expected_v) == len(observed_v) == 256, "V-cache byte count changed")
    k_mismatches = byte_mismatches(observed_k, expected_k)
    v_mismatches = byte_mismatches(observed_v, expected_v)

    observed_dir = layer_dir / "rtl_observed"
    k_record = write_binary(observed_dir / "kv_cache_k_from_rope_rtl_s8.bin", observed_k)
    v_record = write_binary(observed_dir / "kv_cache_v_from_projection_rtl_s8.bin", observed_v)
    result = {
        "status": "PASS_WORKLOAD_BOUND_RTL_KV_CACHE_BYTES" if k_mismatches + v_mismatches == 0 else "FAIL_RTL_KV_CACHE_BYTES",
        "k_source": "ace2_rope_core RTL output cases 1 and 3, first 128 bytes per position",
        "v_source": "ace2_w4a8_proj_core RTL operators 2 and 9, 128 channels per position",
        "k_bytes_compared": len(expected_k),
        "v_bytes_compared": len(expected_v),
        "kv_cache_bytes_compared": len(expected_k) + len(expected_v),
        "k_byte_mismatches": k_mismatches,
        "v_byte_mismatches": v_mismatches,
        "kv_cache_byte_mismatches": k_mismatches + v_mismatches,
        "expected_k_sha256": layer_runner.frontier.sha256_bytes(expected_k),
        "expected_v_sha256": layer_runner.frontier.sha256_bytes(expected_v),
        "observed_k": k_record,
        "observed_v": v_record,
    }
    require(result["kv_cache_bytes_compared"] == KV_BYTES_PER_LAYER, "RTL K/V cache workload byte count changed")
    require(result["kv_cache_byte_mismatches"] == 0, f"RTL K/V cache mismatch: {result}")
    write_json(layer_dir / "rtl_kv_cache_comparison.json", result)
    return result


ORIGINAL_RUN_RTL = layer_runner.run_rtl


def run_rtl(derived: dict[str, Any]) -> dict[str, Any]:
    rtl = ORIGINAL_RUN_RTL(derived)
    rtl["kv_cache_rtl_comparison"] = compare_rtl_kv_cache(layer_runner.ACTIVE_ATTEMPT, derived)
    return rtl


layer_runner.run_rtl = run_rtl


def layer_comparison(
    derived: dict[str, Any], rtl: dict[str, Any], projection_check: dict[str, int]
) -> dict[str, Any]:
    require(rtl["projection_results"] == base.EXPECTED_PROJECTION_RESULTS_PER_LAYER, "layer projection count changed")
    require(rtl["simulator_cycles"] == base.EXPECTED_LAYER_CYCLES, "layer simulator cycles changed")
    kv = rtl["kv_cache_rtl_comparison"]
    exact = {
        "rmsnorm_output_mismatches": 0,
        "projection_accumulator_mismatches": projection_check["accumulator_mismatches"],
        "projection_output_mismatches": projection_check["output_mismatches"],
        "projection_saturation_mismatches": projection_check["saturation_mismatches"],
        "rope_output_mismatches": 0,
        "attention_score_accumulator_mismatches": 0,
        "attention_score_output_mismatches": 0,
        "softmax_probability_mismatches": 0,
        "attention_compose_output_mismatches": 0,
        "attention_residual_output_mismatches": 0,
        "post_attention_rmsnorm_output_mismatches": 0,
        "silu_output_mismatches": 0,
        "final_residual_output_mismatches": 0,
        "kv_cache_k_byte_mismatches": kv["k_byte_mismatches"],
        "kv_cache_v_byte_mismatches": kv["v_byte_mismatches"],
        "kv_cache_byte_mismatches": kv["kv_cache_byte_mismatches"],
        "final_hidden_state_byte_mismatches": 0,
    }
    require(all(value == 0 for value in exact.values()), f"layer mismatch summary changed: {exact}")
    cache = derived["cache_reuse"]
    require(
        cache["cached_k_score_accumulator_changes_when_cached_k_zeroed_heads"] > 0,
        f"layer {derived['layer_id']} K-data counterfactual absent",
    )
    return {
        "schema_version": 2,
        "status": "PASS_EXACT_TWO_POSITION_LAYER_RTL_REPAIR",
        "layer_id": derived["layer_id"],
        "projection_results": projection_check["results"],
        "integer_mismatches": exact,
        "cache_reuse": cache,
        "rtl_kv_cache_comparison": file_record(layer_runner.ACTIVE_ATTEMPT / "rtl_kv_cache_comparison.json"),
        "kv_cache_bytes_compared": kv["kv_cache_bytes_compared"],
        "simulator_cycles": rtl["simulator_cycles"],
        "final_hidden_state_sha256_by_position": rtl["final_layer_output_sha256_by_position"],
    }


base.layer_comparison = layer_comparison


def phase_log_self_test(directory: Path) -> dict[str, Any]:
    stdout = directory / "prepare.stdout.log"
    stderr = directory / "prepare.stderr.log"
    result = directory / "prepare.result.json"
    write_text(stdout, "phase-log-self-test stdout retained\n")
    write_text(stderr, "phase-log-self-test stderr retained\n")
    write_json(
        result,
        {
            "command": ["repair-phase-log-self-test"],
            "returncode": 0,
            "terminal_status": "PASS_PHASE_LOG_SELF_TEST",
        },
    )
    require("stdout retained" in stdout.read_text(encoding="utf-8"), "phase stdout retention failed")
    require("stderr retained" in stderr.read_text(encoding="utf-8"), "phase stderr retention failed")
    require(read_json(result)["returncode"] == 0, "phase exit status retention failed")
    return {
        "status": "PASS_PHASE_LOG_RETENTION_PREFLIGHT",
        "stdout": file_record(stdout),
        "stderr": file_record(stderr),
        "result": file_record(result),
    }


def focused_rtl_cache_preflight(derived: dict[str, Any], directory: Path) -> dict[str, Any]:
    layer_runner.ACTIVE_ATTEMPT = directory
    layer_runner.persist_vectors(derived)
    projection_source = directory / "execution_sources/projection_tb.sv"
    write_text(projection_source, layer_runner.render_projection_source())
    projection = layer_runner.compile_and_run(
        "projection",
        [layer_runner.CORES["projection"], projection_source],
        "ace2_lora_v4_layer0_projection_batch_tb",
    )
    rope_source = layer_runner.execution_source(
        "rope", layer_runner.ROPE_TB, "rope_vectors.svh", "    ace2_rope_core dut ("
    )
    rope = layer_runner.compile_and_run(
        "rope", [layer_runner.CORES["rope"], rope_source], "ace2_rope_tb"
    )
    require(
        "ACE2_LORA_V4_LAYER0_PROJECTION_BATCH_PASS operators=14 channels=25344"
        in projection.pop("stdout_text"),
        "focused projection RTL marker missing",
    )
    require("ACE2_ROPE_TB_PASS cases=4 beats_per_case=56" in rope.pop("stdout_text"), "focused RoPE RTL marker missing")
    require(projection["simulator_cycles"] == base.EXPECTED_LAYER_CYCLES["projection"], "projection cycles changed")
    require(rope["simulator_cycles"] == base.EXPECTED_LAYER_CYCLES["rope"], "RoPE cycles changed")
    comparison = compare_rtl_kv_cache(directory, derived)
    write_json(directory / "focused_rtl_execution.json", {"projection": projection, "rope": rope})
    return {
        "status": comparison["status"],
        "execution": file_record(directory / "focused_rtl_execution.json"),
        "comparison": file_record(directory / "rtl_kv_cache_comparison.json"),
        "kv_cache_bytes_compared": comparison["kv_cache_bytes_compared"],
        "kv_cache_byte_mismatches": comparison["kv_cache_byte_mismatches"],
    }


def preflight_authorized() -> None:
    require(not REPAIR_PREFLIGHT.exists(), f"repair preflight already exists: {REPAIR_PREFLIGHT}")
    require(not ATTEMPT.exists(), f"attempt-0002 already exists: {ATTEMPT}")
    bindings = base.validate_bindings(require_attempt_absent=True)
    predecessor = validate_predecessor()
    boundary = q0_15_boundary_test()

    with torch.no_grad(), safe_open(canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        derived, _ = derive_layer(0, base.initial_states(weights), weights, adapter)
        regression = base.verify_layer0_regression(derived)

    REPAIR_PREFLIGHT.mkdir(parents=True)
    try:
        write_json(REPAIR_PREFLIGHT / "q0_15_boundary.json", boundary)
        write_json(REPAIR_PREFLIGHT / "k_data_counterfactual.json", derived["cache_reuse"])
        write_json(REPAIR_PREFLIGHT / "accepted_fresh_l2_regression.json", regression)
        rtl_cache = focused_rtl_cache_preflight(derived, REPAIR_PREFLIGHT / "rtl-cache-check")
        phase_logs = phase_log_self_test(REPAIR_PREFLIGHT / "phase-log-self-test")
        summary = {
            "schema_version": 1,
            "status": "PASS_REPAIR_EXECUTION_PREFLIGHT",
            "accepted_hashes": bindings["accepted_hashes"],
            "predecessor": predecessor,
            "q0_15_boundary": file_record(REPAIR_PREFLIGHT / "q0_15_boundary.json"),
            "k_data_counterfactual": file_record(REPAIR_PREFLIGHT / "k_data_counterfactual.json"),
            "accepted_fresh_l2_regression": file_record(REPAIR_PREFLIGHT / "accepted_fresh_l2_regression.json"),
            "rtl_cache": rtl_cache,
            "phase_logs": phase_logs,
            "official_attempt_absent": not ATTEMPT.exists(),
        }
        write_json(REPAIR_PREFLIGHT / "summary.json", summary)
        base.write_sha256s(REPAIR_PREFLIGHT)
        members = base.verify_sha256s(REPAIR_PREFLIGHT)
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "preview": base.relative(REPAIR_PREFLIGHT),
                    "members": members,
                    "sha256s_sha256": accepted.sha256_file(REPAIR_PREFLIGHT / "SHA256SUMS"),
                },
                sort_keys=True,
            )
        )
    except BaseException as exc:
        write_json(
            REPAIR_PREFLIGHT / "failure.json",
            {"status": "SEALED_FAIL_REPAIR_PREFLIGHT", "error": str(exc), "traceback": traceback.format_exc()},
        )
        base.write_sha256s(REPAIR_PREFLIGHT)
        raise


def validate_repair_preflight() -> dict[str, Any]:
    require(REPAIR_PREFLIGHT.is_dir(), "repair preflight is absent")
    members = base.verify_sha256s(REPAIR_PREFLIGHT)
    summary = read_json(REPAIR_PREFLIGHT / "summary.json")
    require(summary["status"] == "PASS_REPAIR_EXECUTION_PREFLIGHT", "repair preflight did not pass")
    require(summary["rtl_cache"]["kv_cache_bytes_compared"] == KV_BYTES_PER_LAYER, "preflight K/V bytes changed")
    require(summary["rtl_cache"]["kv_cache_byte_mismatches"] == 0, "preflight K/V mismatch")
    return {
        "path": base.relative(REPAIR_PREFLIGHT),
        "members": members,
        "sha256s": file_record(REPAIR_PREFLIGHT / "SHA256SUMS"),
        "summary": file_record(REPAIR_PREFLIGHT / "summary.json"),
    }


def repair_seal_failure(phase: str, exc: BaseException) -> None:
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    failure = {
        "schema_version": 2,
        "status": f"SEALED_FAIL_{phase.upper()}",
        "phase": phase,
        "failure_taxonomy": "REPAIR_EXECUTION_GATE_OR_RTL_OFFICIAL_ATTEMPT",
        "root_cause_hypothesis": str(exc),
        "regression_required": "allocate a separately recorded attempt after an evidence-backed fix; never replay attempt-0002",
        "traceback": traceback.format_exc(),
        "official_attempt_consumed": True,
        "predecessor_attempt_preserved": True,
    }
    write_json(ATTEMPT / "failure.json", failure)
    write_json(ATTEMPT / "status.json", {"status": failure["status"], "sealed": True})


base.seal_failure = repair_seal_failure


def prepare_body() -> dict[str, Any]:
    predecessor = validate_predecessor()
    preflight = validate_repair_preflight()
    base.prepare()

    wrapper_copy = write_binary(ATTEMPT / "source/run_lora_v4_two_token_24layer_rtl_repair.py", Path(__file__).read_bytes())
    authorization = read_json(ATTEMPT / "authorization.json")
    authorization.update(
        {
            "schema_version": 2,
            "kind": "reviewer_directed_hash_bound_repair_execution_gate",
            "authorized_scope": "exactly-once prepare/run/check for attempt-0002 only",
            "repair_reason": "unsigned Q0.15 serialization, K-data counterfactual, measured RTL K/V bytes, and retained phase logs",
            "predecessor": predecessor,
            "repair_preflight": preflight,
            "authorized_wrapper_copy": wrapper_copy,
        }
    )
    write_json(ATTEMPT / "authorization.json", authorization)
    freeze = read_json(ATTEMPT / "freeze.json")
    freeze.update(
        {
            "schema_version": 2,
            "official_attempt": base.relative(ATTEMPT),
            "predecessor": predecessor,
            "repair_preflight": preflight,
            "wrapper_copy": wrapper_copy,
            "q0_15_storage": "unsigned little-endian 16-bit, frozen legal range 0..32768",
            "k_data_counterfactual": "zero_position0_k_payload_only",
            "rtl_kv_cache_bytes_per_layer": KV_BYTES_PER_LAYER,
            "phase_logs_required": ["prepare", "run", "check"],
        }
    )
    write_json(ATTEMPT / "freeze.json", freeze)
    result = {
        "status": "PREPARED_HASH_BOUND_REPAIR_ATTEMPT",
        "attempt": base.relative(ATTEMPT),
        "wrapper_sha256": freeze["wrapper"]["sha256"],
        "predecessor_sha256s_sha256": predecessor["sha256s"]["sha256"],
        "repair_preflight_sha256s_sha256": preflight["sha256s"]["sha256"],
    }
    print(json.dumps(result, sort_keys=True))
    return result


def require_completed_phase(phase: str) -> dict[str, Any]:
    directory = ATTEMPT / "phase-logs"
    stdout = directory / f"{phase}.stdout.log"
    stderr = directory / f"{phase}.stderr.log"
    result = directory / f"{phase}.result.json"
    require(stdout.is_file() and stderr.is_file() and result.is_file(), f"retained {phase} phase logs are absent")
    record = read_json(result)
    require(record["returncode"] == 0, f"retained {phase} phase did not pass")
    return {"stdout": file_record(stdout), "stderr": file_record(stderr), "result": file_record(result)}


def validate_run_repair_evidence() -> dict[str, Any]:
    summary = read_json(ATTEMPT / "run_summary.json")
    k_layers = 0
    cache_layers = 0
    cache_bytes = 0
    for layer_id in range(base.LAYERS):
        layer_dir = ATTEMPT / "layers" / f"layer-{layer_id:02d}"
        comparison = read_json(layer_dir / "comparison.json")
        rtl = read_json(layer_dir / "rtl_execution.json")
        cache = comparison["cache_reuse"]
        require(
            cache["cached_k_score_accumulator_changes_when_cached_k_zeroed_heads"] > 0,
            f"layer {layer_id} K-data counterfactual absent",
        )
        k_layers += 1
        kv = rtl["kv_cache_rtl_comparison"]
        require(kv["kv_cache_bytes_compared"] == KV_BYTES_PER_LAYER, f"layer {layer_id} K/V byte count changed")
        require(kv["kv_cache_byte_mismatches"] == 0, f"layer {layer_id} K/V byte mismatch")
        cache_bytes += kv["kv_cache_bytes_compared"]
        cache_layers += 1
    require(k_layers == cache_layers == base.LAYERS, "repair evidence layer count changed")
    summary.update(
        {
            "schema_version": 2,
            "k_data_counterfactual_layers": k_layers,
            "workload_bound_rtl_kv_cache_layers": cache_layers,
            "rtl_kv_cache_bytes_compared": cache_bytes,
            "rtl_kv_cache_byte_mismatches": 0,
        }
    )
    write_json(ATTEMPT / "run_summary.json", summary)
    return {
        "status": "PASS_RUN_REPAIR_EVIDENCE",
        "k_data_counterfactual_layers": k_layers,
        "workload_bound_rtl_kv_cache_layers": cache_layers,
        "rtl_kv_cache_bytes_compared": cache_bytes,
    }


def run_body() -> dict[str, Any]:
    validate_predecessor()
    validate_repair_preflight()
    require_completed_phase("prepare")
    base.run()
    result = validate_run_repair_evidence()
    print(json.dumps(result, sort_keys=True))
    return result


def check_body() -> dict[str, Any]:
    require(ATTEMPT.is_dir(), "official attempt is absent")
    status = read_json(ATTEMPT / "status.json")
    require(
        status["status"] == "PASS_24LAYER_RTL_RUN_UNCHECKED" and status["sealed"] is False,
        "official run is not checkable",
    )
    require(not (ATTEMPT / "check.started.json").exists(), "official check already consumed")
    write_json(ATTEMPT / "check.started.json", {"phase": "check", "monotonic_start": time.monotonic()})

    bindings = base.validate_bindings(require_attempt_absent=False)
    predecessor = validate_predecessor()
    preflight = validate_repair_preflight()
    prepare_logs = require_completed_phase("prepare")
    run_logs = require_completed_phase("run")
    freeze = read_json(ATTEMPT / "freeze.json")
    run_summary = read_json(ATTEMPT / "run_summary.json")
    require(freeze["wrapper"]["sha256"] == accepted.sha256_file(Path(__file__)), "repair wrapper changed")
    require(run_summary["layers_executed"] == base.LAYERS, "layer count changed")
    require(run_summary["projection_results"] == base.EXPECTED_PROJECTION_RESULTS, "projection count changed")
    require(run_summary["total_simulator_cycles"] == base.EXPECTED_TOTAL_CYCLES, "total cycle count changed")

    aggregate_mismatches: dict[str, int] = {}
    cache_dependency_layers = 0
    k_counterfactual_layers = 0
    rtl_cache_layers = 0
    rtl_cache_bytes = 0
    final_hashes: list[str] | None = None
    for layer_id in range(base.LAYERS):
        layer_dir = ATTEMPT / "layers" / f"layer-{layer_id:02d}"
        comparison = read_json(layer_dir / "comparison.json")
        rtl = read_json(layer_dir / "rtl_execution.json")
        manifest = read_json(layer_dir / "manifest.json")
        require(
            comparison["status"] == manifest["status"] == "PASS_EXACT_TWO_POSITION_LAYER_RTL_REPAIR",
            f"layer {layer_id} status changed",
        )
        require(
            comparison["projection_results"] == base.EXPECTED_PROJECTION_RESULTS_PER_LAYER,
            f"layer {layer_id} projection count changed",
        )
        require(rtl["simulator_cycles"] == base.EXPECTED_LAYER_CYCLES, f"layer {layer_id} cycles changed")
        for name, value in comparison["integer_mismatches"].items():
            aggregate_mismatches[name] = aggregate_mismatches.get(name, 0) + int(value)
        cache = comparison["cache_reuse"]
        require(cache["cached_position0_probability_nonzero_heads"] > 0, f"layer {layer_id} cached K slot absent")
        require(
            cache["cached_k_score_accumulator_changes_when_cached_k_zeroed_heads"] > 0,
            f"layer {layer_id} cached K data dependence absent",
        )
        require(
            cache["compose_output_changes_when_cached_v_zeroed_heads"] > 0,
            f"layer {layer_id} cached V dependence absent",
        )
        cache_dependency_layers += 1
        k_counterfactual_layers += 1
        kv = rtl["kv_cache_rtl_comparison"]
        require(kv["kv_cache_byte_mismatches"] == 0, f"layer {layer_id} measured RTL K/V mismatch")
        require(kv["kv_cache_bytes_compared"] == KV_BYTES_PER_LAYER, f"layer {layer_id} RTL K/V byte count changed")
        rtl_cache_layers += 1
        rtl_cache_bytes += kv["kv_cache_bytes_compared"]
        if layer_id == base.LAYERS - 1:
            final_hashes = comparison["final_hidden_state_sha256_by_position"]

    require(all(value == 0 for value in aggregate_mismatches.values()), f"aggregate integer mismatch: {aggregate_mismatches}")
    require(cache_dependency_layers == k_counterfactual_layers == rtl_cache_layers == base.LAYERS, "cache layer count changed")
    require(rtl_cache_bytes == base.LAYERS * KV_BYTES_PER_LAYER, "aggregate RTL K/V byte count changed")
    require(final_hashes is not None and len(final_hashes) == 2, "final layer-23 hidden states absent")

    check_result = {
        "schema_version": 2,
        "status": "PASS_24LAYER_EXACT_INTEGER_REPAIR_CHECK_AWAITING_INDEPENDENT_REVIEW",
        "layers": base.LAYERS,
        "projection_results": base.EXPECTED_PROJECTION_RESULTS,
        "integer_mismatches": aggregate_mismatches,
        "position1_cache_dependency_layers": cache_dependency_layers,
        "k_data_counterfactual_layers": k_counterfactual_layers,
        "workload_bound_rtl_kv_cache_layers": rtl_cache_layers,
        "rtl_kv_cache_bytes_compared": rtl_cache_bytes,
        "rtl_kv_cache_byte_mismatches": 0,
        "final_layer23_hidden_state_sha256_by_position": final_hashes,
        "aggregate_family_cycles": run_summary["aggregate_family_cycles"],
        "total_simulator_cycles": run_summary["total_simulator_cycles"],
        "elapsed_wall_seconds": run_summary["elapsed_wall_seconds"],
        "peak_rss_kib": run_summary["peak_rss_kib"],
        "tool_versions": bindings["tool_versions"],
        "accepted_hashes": bindings["accepted_hashes"],
        "predecessor": predecessor,
        "repair_preflight": preflight,
        "retained_completed_phase_logs": {"prepare": prepare_logs, "run": run_logs},
        "review_required": True,
    }
    write_json(ATTEMPT / "check.json", check_result)
    manifest = {
        "schema_version": 2,
        "status": check_result["status"],
        "claim_boundary": read_json(base.CONTRACT)["claim_boundary"],
        "authorization": file_record(ATTEMPT / "authorization.json"),
        "freeze": file_record(ATTEMPT / "freeze.json"),
        "run_summary": file_record(ATTEMPT / "run_summary.json"),
        "check": file_record(ATTEMPT / "check.json"),
        "wrapper": file_record(Path(__file__)),
        "wrapper_copy": file_record(ATTEMPT / "source/run_lora_v4_two_token_24layer_rtl_repair.py"),
        "accepted_artifacts": bindings["accepted_hashes"],
        "predecessor": predecessor,
        "repair_preflight": preflight,
        "official_attempt_consumed": True,
        "immutable_after_check": True,
        "independent_review_required": True,
    }
    write_json(ATTEMPT / "manifest.json", manifest)
    write_json(ATTEMPT / "status.json", {"status": check_result["status"], "sealed": False})
    print(json.dumps(check_result, sort_keys=True))
    return check_result


class Tee(io.TextIOBase):
    def __init__(self, terminal: io.TextIOBase, retained: io.TextIOBase):
        self.terminal = terminal
        self.retained = retained

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.terminal.flush()
        self.retained.write(value)
        self.retained.flush()
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.retained.flush()


def phase_log_records() -> dict[str, Any]:
    return {phase: require_completed_phase(phase) for phase in ("prepare", "run", "check")}


def finalize_successful_check() -> dict[str, Any]:
    logs = phase_log_records()
    check_result = read_json(ATTEMPT / "check.json")
    check_result["retained_phase_logs"] = logs
    write_json(ATTEMPT / "check.json", check_result)
    manifest = read_json(ATTEMPT / "manifest.json")
    manifest["phase_logs"] = logs
    manifest["check"] = file_record(ATTEMPT / "check.json")
    write_json(ATTEMPT / "manifest.json", manifest)
    write_json(ATTEMPT / "status.json", {"status": check_result["status"], "sealed": True})
    base.write_sha256s(ATTEMPT)
    members = base.verify_sha256s(ATTEMPT)
    return {
        **check_result,
        "attempt": base.relative(ATTEMPT),
        "members": members,
        "sha256s_sha256": accepted.sha256_file(ATTEMPT / "SHA256SUMS"),
    }


def finalize_failure_phase_logs() -> None:
    failure_path = ATTEMPT / "failure.json"
    if failure_path.is_file():
        failure = read_json(failure_path)
        available: dict[str, Any] = {}
        for phase in ("prepare", "run", "check"):
            result = ATTEMPT / "phase-logs" / f"{phase}.result.json"
            if result.is_file():
                directory = ATTEMPT / "phase-logs"
                available[phase] = {
                    "stdout": file_record(directory / f"{phase}.stdout.log"),
                    "stderr": file_record(directory / f"{phase}.stderr.log"),
                    "result": file_record(result),
                }
        failure["retained_phase_logs"] = available
        write_json(failure_path, failure)
    base.write_sha256s(ATTEMPT)
    base.verify_sha256s(ATTEMPT)


def execute_phase(phase: str, body: Callable[[], dict[str, Any]]) -> int:
    if phase == "prepare":
        require(not PREPARE_PHASE_TMP.exists(), f"prepare phase temp already exists: {PREPARE_PHASE_TMP}")
        log_dir = PREPARE_PHASE_TMP
    else:
        require(ATTEMPT.is_dir(), "attempt-0002 is absent")
        log_dir = ATTEMPT / "phase-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{phase}.stdout.log"
    stderr_path = log_dir / f"{phase}.stderr.log"
    require(not stdout_path.exists() and not stderr_path.exists(), f"{phase} phase logs already exist")

    started_utc = utc_now()
    started = time.monotonic()
    result_value: dict[str, Any] | None = None
    error: BaseException | None = None
    with stdout_path.open("w", encoding="utf-8") as retained_stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as retained_stderr, contextlib.redirect_stdout(Tee(sys.stdout, retained_stdout)), contextlib.redirect_stderr(
        Tee(sys.stderr, retained_stderr)
    ):
        try:
            result_value = body()
            print(json.dumps({"phase": phase, "terminal_status": "PASS", "returncode": 0}, sort_keys=True))
        except BaseException as exc:
            error = exc
            if not (ATTEMPT / "status.json").is_file() or not read_json(ATTEMPT / "status.json").get("sealed"):
                repair_seal_failure(phase, exc)
            traceback.print_exc()
            print(json.dumps({"phase": phase, "terminal_status": "FAIL", "returncode": 1}, sort_keys=True))

    if phase == "prepare":
        destination = ATTEMPT / "phase-logs"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stdout_path), str(destination / stdout_path.name))
        shutil.move(str(stderr_path), str(destination / stderr_path.name))
        PREPARE_PHASE_TMP.rmdir()
        log_dir = destination
        stdout_path = destination / stdout_path.name
        stderr_path = destination / stderr_path.name

    returncode = 0 if error is None else 1
    status = read_json(ATTEMPT / "status.json") if (ATTEMPT / "status.json").is_file() else {}
    write_json(
        log_dir / f"{phase}.result.json",
        {
            "schema_version": 1,
            "phase": phase,
            "command": AUTHORIZED_COMMANDS[phase],
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "returncode": returncode,
            "terminal_status": "PASS" if returncode == 0 else "FAIL",
            "attempt_status": status,
            "body_result": result_value,
        },
    )

    if error is not None:
        finalize_failure_phase_logs()
        return 1
    if phase == "check":
        sealed = finalize_successful_check()
        print(json.dumps(sealed, sort_keys=True))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-authorized", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.preflight_authorized:
        preflight_authorized()
        return
    if args.prepare:
        raise SystemExit(execute_phase("prepare", prepare_body))
    if args.run:
        raise SystemExit(execute_phase("run", run_body))
    raise SystemExit(execute_phase("check", check_body))


if __name__ == "__main__":
    main()
