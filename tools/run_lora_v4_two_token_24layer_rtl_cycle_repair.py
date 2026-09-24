#!/usr/bin/env python3
"""Workload-bound cycle repair for the frozen 24-layer ACE-2 RTL mission.

This distinct wrapper preserves attempts 0001/0002, binds the prior repair,
predicts every RTL family's per-layer simulator cycles from the exact vectors,
reconciles those predictions against retained raw simulator logs, and owns the
exactly-once attempt-0003 prepare/run/check namespace.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_lora_v4_two_token_24layer_rtl_repair as prior


base = prior.base
accepted = prior.accepted
canonical = prior.canonical
layer_runner = prior.layer_runner

PRIOR_WRAPPER = ROOT / "tools/run_lora_v4_two_token_24layer_rtl_repair.py"
ATTEMPT_0002 = base.OFFICIAL_ROOT / "attempt-0002"
ATTEMPT = base.OFFICIAL_ROOT / "attempt-0003"
CYCLE_PREFLIGHT = ROOT / "build/lora-v4-two-token-24layer-rtl-cycle-repair-preview/preflight-0001"
PREPARE_PHASE_TMP = base.OFFICIAL_ROOT / ".attempt-0003-prepare-phase"

PRIOR_WRAPPER_SHA256 = "e1f558870e0e8e3ccb3398d5259a14b2302a3545d0e97c4b6b33900c56e3ae5f"
ATTEMPT_0002_SUMS_SHA256 = "3d9c028b23c18e35935cd659f9eab5bac39c85512609f412ae82f2e220792b5e"
ATTEMPT_0002_MEMBERS = 320
PRIOR_PREFLIGHT_SUMS_SHA256 = "5e89b5d3299db5484d26ed7d895eb2a2b51f81aa2aaf6149cf1ac0315c680b4d"

CYCLE_FAMILIES = ("projection", "rmsnorm", "rope", "score", "softmax", "compose", "silu", "residual")
FOCUSED_LAYER01_DELTA = {
    "projection": 21488,
    "rmsnorm": -76,
    "rope": 1068,
    "score": 0,
    "softmax": 0,
    "compose": 0,
    "silu": 4864,
    "residual": 8,
}

# Fixed scheduler costs are derived from the hash-bound RTL/testbench state
# machines.  The variable terms below are reconstructed from each layer's
# generated vectors and independently checked against the raw simulator logs.
FIXED_CYCLES = {
    "projection": 25836293,
    "rmsnorm": 248828,
    "rope": 743237,
    "score": 5,
    "softmax": 5,
    "compose": 4,
    "silu": 1316935,
    "residual": 3615,
}

CYCLE_SOURCE_HASHES = {
    "tools/run_lora_v4_layer0_two_token_kv_full_rtl.py": "afe6ad0fae5db7c8d6e7dc9c9bf5136654a867a546363ea38e19644689334e25",
    "rtl/ace2_w4a8_proj_core.sv": "e4e7ff9d458ae823728669ae0a53ba650dbb0633d28c613347e4af2e1e198dfe",
    "verification/tb/ace2_lora_v4_layer0_projection_batch_tb.sv": "ba70be6b912890011db4acf7adef3db8412c6798bdcf423ce7885c34c4042603",
    "rtl/ace2_rmsnorm_core.sv": "b09fe7073fd6509f0ca83d5b0982b1952aa624d4883631690df35c0bfabb014d",
    "verification/tb/ace2_rmsnorm_tb.sv": "848aeebac69c675e132524d0f2ed8791fac0833058f607af8312c8e763466197",
    "rtl/ace2_rope_core.sv": "e5feff02169c5f58b81bfd44350380f6074ca0cbbedf536a0744d8c2f663f3c8",
    "verification/tb/ace2_rope_tb.sv": "876a24f74b80e10d1d79d483f2ba919ca938a7b7307ab9ed5f5f883e26f65297",
    "rtl/ace2_attention_score_core.sv": "400234b1f5b28f379792d5c1e023adc3f2c81e7bea825d5f7683bdc6ccd1dd55",
    "verification/tb/ace2_attention_score_tb.sv": "7ab860d870397da522b72171f12abc7e20f209208001a392904aaa47ff1ec408",
    "rtl/ace2_softmax_core.sv": "33b76b6a840111bf17f3c5324b6c2f71f1ca1a6ca29a67746fc92ffa53043ea6",
    "verification/tb/ace2_softmax_tb.sv": "94eedebf2ca38cbc3344b58006741ab7ac91721dc2ef648688242bd8a1a944be",
    "rtl/ace2_attention_compose_core.sv": "33ca20e48a6f9eb8f93874e95c8e8b73a48c00e435693b84d4a5915da7ca78ff",
    "verification/tb/ace2_attention_compose_tb.sv": "4776095cbbb616a4b78cdbf938c87f436e79f7d7663e34a7f227dfea3f4b7810",
    "rtl/ace2_silu_gate_core.sv": "ace98af2654bb2ea0ccb256e55ee6d755ff4bc34d0fe0d2cdeff6dcd581199ae",
    "verification/tb/ace2_silu_gate_tb.sv": "e940835530544fc313167082dbcb6a77d8a3c5aa8839f99ea704b30e526fc83b",
    "rtl/ace2_shell.sv": "1325b0fa8993f5d760d38ad15bff0065088eeb3e894fda431e5cdcad0a614bb9",
    "verification/tb/ace2_shell_tb.sv": "6e937a3f6d4b0aa2b88eb9a56dd5e60c6f062692ff14ff7642ed3d82763ab571",
}

AUTHORIZED_COMMANDS = {
    "prepare": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/run_lora_v4_two_token_24layer_rtl_cycle_repair.py --prepare",
    "run": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/run_lora_v4_two_token_24layer_rtl_cycle_repair.py --run",
    "check": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/run_lora_v4_two_token_24layer_rtl_cycle_repair.py --check",
}

# Retarget inherited globals to this distinct immutable namespace.  Imported
# functions resolve these module globals at call time.
base.__file__ = str(Path(__file__).resolve())
base.ATTEMPT = ATTEMPT
base.AUTHORIZED_COMMANDS = AUTHORIZED_COMMANDS
prior.ATTEMPT = ATTEMPT
prior.PREPARE_PHASE_TMP = PREPARE_PHASE_TMP
prior.AUTHORIZED_COMMANDS = AUTHORIZED_COMMANDS

require = base.require
read_json = base.read_json
write_json = base.write_json
write_binary = base.write_binary
file_record = base.file_record

ORIGINAL_VALIDATE_ATTEMPT1 = prior.validate_predecessor
ORIGINAL_REPAIR_RUN_RTL = layer_runner.run_rtl


def validate_cycle_sources() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, expected in CYCLE_SOURCE_HASHES.items():
        path = ROOT / name
        require(path.is_file(), f"cycle source absent: {name}")
        require(accepted.sha256_file(path) == expected, f"cycle source changed: {name}")
        records[name] = file_record(path)
    return {"status": "PASS_HASH_BOUND_CYCLE_SOURCES", "members": records}


def validate_predecessors() -> dict[str, Any]:
    attempt1 = ORIGINAL_VALIDATE_ATTEMPT1()
    require(PRIOR_WRAPPER.is_file(), "attempt-0002 repair wrapper is absent")
    require(accepted.sha256_file(PRIOR_WRAPPER) == PRIOR_WRAPPER_SHA256, "attempt-0002 repair wrapper changed")
    require(ATTEMPT_0002.is_dir(), "attempt-0002 is absent")
    require(
        accepted.sha256_file(ATTEMPT_0002 / "SHA256SUMS") == ATTEMPT_0002_SUMS_SHA256,
        "attempt-0002 aggregate changed",
    )
    members = base.verify_sha256s(ATTEMPT_0002)
    require(members == ATTEMPT_0002_MEMBERS, "attempt-0002 member count changed")
    status = read_json(ATTEMPT_0002 / "status.json")
    require(status == {"sealed": True, "status": "SEALED_FAIL_RUN"}, "attempt-0002 status changed")
    prior_preflight_sums = prior.REPAIR_PREFLIGHT / "SHA256SUMS"
    require(prior_preflight_sums.is_file(), "attempt-0002 repair preflight aggregate is absent")
    require(
        accepted.sha256_file(prior_preflight_sums) == PRIOR_PREFLIGHT_SUMS_SHA256,
        "attempt-0002 repair preflight changed",
    )
    return {
        "attempt_0001": attempt1,
        "attempt_0002": {
            "attempt": base.relative(ATTEMPT_0002),
            "status": status["status"],
            "members": members,
            "sha256s": file_record(ATTEMPT_0002 / "SHA256SUMS"),
            "wrapper": file_record(PRIOR_WRAPPER),
            "repair_preflight_sha256s": file_record(prior_preflight_sums),
            "preserved_unchanged": True,
        },
    }


def indexed_decimal_assignments(path: Path, name: str) -> list[int]:
    text = path.read_text(encoding="utf-8")
    rows = [(int(index), int(value)) for index, value in re.findall(
        rf"{re.escape(name)}\[(\d+)\] = (?:\d+'d)?(\d+);", text
    )]
    require(rows, f"no assignments found for {name}")
    rows.sort()
    require([index for index, _ in rows] == list(range(len(rows))), f"non-contiguous assignments for {name}")
    return [value for _, value in rows]


def indexed_bits(path: Path, name: str) -> list[int]:
    text = path.read_text(encoding="utf-8")
    rows = [(int(index), int(value)) for index, value in re.findall(
        rf"{re.escape(name)}\[(\d+)\] = 1'b([01]);", text
    )]
    require(rows, f"no bit assignments found for {name}")
    rows.sort()
    require([index for index, _ in rows] == list(range(len(rows))), f"non-contiguous assignments for {name}")
    return [value for _, value in rows]


def indexed_hex128(path: Path, name: str) -> list[int]:
    text = path.read_text(encoding="utf-8")
    rows = [(int(index), int(value, 16)) for index, value in re.findall(
        rf"{re.escape(name)}\[(\d+)\] = 128'h([0-9a-fA-F]+);", text
    )]
    require(rows, f"no 128-bit assignments found for {name}")
    rows.sort()
    require([index for index, _ in rows] == list(range(len(rows))), f"non-contiguous assignments for {name}")
    return [value for _, value in rows]


def signed(value: int, width: int) -> int:
    return value - (1 << width) if value & (1 << (width - 1)) else value


def lanes_s8(word: int) -> list[int]:
    return [signed(value, 8) for value in word.to_bytes(16, "little")]


def lanes_s16(word: int) -> list[int]:
    raw = word.to_bytes(16, "little")
    return [signed(int.from_bytes(raw[offset : offset + 2], "little"), 16) for offset in range(0, 16, 2)]


def predict_projection(vectors: Path) -> tuple[int, dict[str, Any]]:
    shifts = [int(row, 16) for row in (vectors / "projection_shift.hex").read_text(encoding="utf-8").splitlines() if row]
    require(len(shifts) == base.EXPECTED_PROJECTION_RESULTS_PER_LAYER, "projection shift count changed")
    require(all(0 <= value <= 63 for value in shifts), "projection shift range changed")
    workload = sum(shifts)
    return FIXED_CYCLES["projection"] + workload, {
        "right_shift_values": len(shifts),
        "right_shift_cycle_sum": workload,
        "formula": "25836293 + sum(projection_shift[25344])",
    }


def predict_rmsnorm(vectors: Path) -> tuple[int, dict[str, Any]]:
    sums = []
    text = (vectors / "rmsnorm_vectors.svh").read_text(encoding="utf-8")
    rows = [(int(index), int(value, 16)) for index, value in re.findall(
        r"test_expected_sumsq\[(\d+)\] = 48'h([0-9a-fA-F]+);", text
    )]
    rows.sort()
    require([index for index, _ in rows] == list(range(4)), "RMSNorm case count changed")
    sums = [value for _, value in rows]
    rounded_means = [(value + base.HIDDEN // 2) // base.HIDDEN for value in sums]
    roots = [math.isqrt(value - 1) + 1 if value else 0 for value in rounded_means]
    workload = 2 * sum(roots)
    return FIXED_CYCLES["rmsnorm"] + workload, {
        "sumsq": sums,
        "rounded_mean_square": rounded_means,
        "ceil_sqrt_iterations": roots,
        "sqrt_state_cycles": workload,
        "formula": "248828 + 2*sum(ceil_sqrt(round(sumsq/896)))",
    }


def predict_rope(vectors: Path) -> tuple[int, dict[str, Any]]:
    path = vectors / "rope_vectors.svh"
    input_words = [lanes_s8(value) for value in indexed_hex128(path, "rope_input_beats")]
    scale_words = [lanes_s16(value) for value in indexed_hex128(path, "rope_scale_beats")]
    cos_words = [lanes_s16(value) for value in indexed_hex128(path, "rope_cos_beats")]
    sin_words = [lanes_s16(value) for value in indexed_hex128(path, "rope_sin_beats")]
    require(len(input_words) == 4 * 56, "RoPE input workload changed")
    require(len(scale_words) == len(cos_words) == len(sin_words) == 4 * 56 * 2, "RoPE metadata workload changed")
    counts = [0, 0, 0, 0, 0]
    for case in range(4):
        for beat in range(56):
            pair_beat = beat ^ 2
            second_half = bool((beat >> 1) & 1)
            for lane in range(16):
                lane_group = lane // 8
                lane_offset = lane % 8
                metadata_index = case * 112 + beat * 2 + lane_group
                pair_metadata_index = case * 112 + pair_beat * 2 + lane_group
                act = input_words[case * 56 + beat][lane]
                pair = input_words[case * 56 + pair_beat][lane]
                scale = scale_words[metadata_index][lane_offset]
                pair_scale = scale_words[pair_metadata_index][lane_offset]
                cosine = cos_words[metadata_index][lane_offset]
                sine = sin_words[metadata_index][lane_offset]
                act_scaled = act * scale
                pair_scaled = pair * pair_scale
                prod_cos = act_scaled * cosine
                prod_sin = pair_scaled * sine
                rotated = prod_cos + prod_sin if second_half else prod_cos - prod_sin
                conditions = (
                    (act < 0) ^ (scale < 0),
                    (pair < 0) ^ (pair_scale < 0),
                    (act_scaled < 0) ^ (cosine < 0),
                    (pair_scaled < 0) ^ (sine < 0),
                    rotated < 0,
                )
                counts = [value + int(condition) for value, condition in zip(counts, conditions, strict=True)]
    negative_paths = sum(counts)
    workload = 3 * negative_paths
    return FIXED_CYCLES["rope"] + workload, {
        "negative_path_counts": {
            "act_scale": counts[0],
            "pair_scale": counts[1],
            "cos_product": counts[2],
            "sin_product": counts[3],
            "rotated_result": counts[4],
        },
        "negative_paths": negative_paths,
        "negation_chunk_cycles": workload,
        "formula": "743237 + 3*count(signed negate paths across 3584 lanes)",
    }


def predict_score(vectors: Path) -> tuple[int, dict[str, Any]]:
    contexts = indexed_decimal_assignments(vectors / "attention_score_vectors.svh", "attn_score_context_count")
    require(len(contexts) == 28 and all(value in (1, 2) for value in contexts), "score context workload changed")
    token_runs = sum(contexts)
    return FIXED_CYCLES["score"] + 134 * token_runs, {
        "contexts": contexts,
        "token_runs": token_runs,
        "cycles_per_64-lane_score": 134,
        "formula": "5 + 134*sum(context_count[28])",
    }


def predict_softmax(vectors: Path) -> tuple[int, dict[str, Any]]:
    contexts = indexed_decimal_assignments(vectors / "softmax_vectors.svh", "softmax_context_count")
    require(len(contexts) == 28 and all(value in (1, 2) for value in contexts), "softmax context workload changed")
    expected = FIXED_CYCLES["softmax"] + 36 * len(contexts) + 49 * sum(contexts)
    return expected, {
        "contexts": contexts,
        "cases": len(contexts),
        "active_context_entries": sum(contexts),
        "formula": "5 + 36*cases + 49*sum(context_count)",
    }


def predict_compose(vectors: Path) -> tuple[int, dict[str, Any]]:
    contexts = indexed_decimal_assignments(vectors / "attention_compose_vectors.svh", "attn_compose_context_count")
    require(len(contexts) == 28 and all(value == 9 for value in contexts), "compose protocol workload changed")
    case_cycles = [1214] + [1215 for _ in contexts[1:]]
    return FIXED_CYCLES["compose"] + sum(case_cycles), {
        "contexts": contexts,
        "tiles_per_case": [2 for _ in contexts],
        "predicted_case_cycles": case_cycles,
        "formula": "4 + first context-9 case 1214 + remaining 27 context-9 cases 1215",
    }


def predict_silu(vectors: Path) -> tuple[int, dict[str, Any]]:
    path = vectors / "silu_gate_vectors.svh"
    lengths = indexed_decimal_assignments(path, "silu_case_length")
    shifts = indexed_decimal_assignments(path, "silu_case_right_shift")
    require(len(lengths) == len(shifts) == 2, "SiLU case count changed")
    require(lengths == [base.INTERMEDIATE, base.INTERMEDIATE], "SiLU lane workload changed")
    workload = sum(length * shift for length, shift in zip(lengths, shifts, strict=True))
    return FIXED_CYCLES["silu"] + workload, {
        "lengths": lengths,
        "right_shifts": shifts,
        "right_shift_lane_cycles": workload,
        "formula": "1316935 + sum(case_length*right_shift)",
    }


def predict_residual(vectors: Path) -> tuple[int, dict[str, Any]]:
    attention = indexed_bits(vectors / "residual_vectors.svh", "residual_expected_saturation")
    final = indexed_bits(vectors / "mlp_residual_vectors.svh", "mlp_residual_expected_saturation")
    require(len(attention) == len(final) == 2, "residual case workload changed")
    clear_writes = sum(attention) + sum(final)
    return FIXED_CYCLES["residual"] + 8 * clear_writes, {
        "attention_saturation_by_position": attention,
        "final_saturation_by_position": final,
        "csr_error_clear_writes": clear_writes,
        "cycles_per_clear_write": 8,
        "formula": "3615 + 8*sum(expected_saturation across four residual commands)",
    }


def predict_cycle_vector(layer_dir: Path) -> dict[str, Any]:
    vectors = layer_dir / "vectors"
    require(vectors.is_dir(), f"layer vectors absent: {vectors}")
    predictors = {
        "projection": predict_projection,
        "rmsnorm": predict_rmsnorm,
        "rope": predict_rope,
        "score": predict_score,
        "softmax": predict_softmax,
        "compose": predict_compose,
        "silu": predict_silu,
        "residual": predict_residual,
    }
    expected: dict[str, int] = {}
    workload: dict[str, Any] = {}
    for family in CYCLE_FAMILIES:
        expected[family], workload[family] = predictors[family](vectors)
    return {
        "schema_version": 1,
        "status": "PASS_SOURCE_DERIVED_WORKLOAD_CYCLE_PREDICTION",
        "expected_cycles": expected,
        "workload_terms": workload,
    }


def one_cycle_marker(text: str, pattern: str, family: str) -> int:
    rows = re.findall(pattern, text)
    require(len(rows) == 1, f"{family} raw cycle marker count changed")
    return int(rows[0])


def reconcile_cycles(layer_dir: Path, reported_cycles: dict[str, int]) -> dict[str, Any]:
    require(set(reported_cycles) == set(CYCLE_FAMILIES), "reported cycle family set changed")
    prediction = predict_cycle_vector(layer_dir)
    predicted = prediction["expected_cycles"]
    raw_cycles: dict[str, int] = {}
    raw_logs: dict[str, Any] = {}
    pass_markers = {
        "projection": "ACE2_LORA_V4_LAYER0_PROJECTION_BATCH_PASS operators=14 channels=25344",
        "rmsnorm": "ACE2_RMSNORM_TB_PASS cases=4 beats_per_case=56",
        "rope": "ACE2_ROPE_TB_PASS cases=4 beats_per_case=56",
        "score": "ACE2_ATTN_SCORE_TB_PASS cases=28 context_max=8",
        "softmax": "ACE2_SOFTMAX_TB_PASS cases=28 context_max=8",
        "compose": "ACE2_ATTN_COMPOSE_TB_PASS cases=28 contexts=9,16,17 context_max=32768 tile=8",
        "silu": "ACE2_SILU_GATE_TB_PASS cases=2",
        "residual": "ACE2_LORA_V4_LAYER0_TWO_TOKEN_FULL_RESIDUAL_REPLAY_PASS positions=2 stages=4 beats=224",
    }
    for family in CYCLE_FAMILIES:
        path = layer_dir / "logs" / f"{family}.vvp.stdout.log"
        require(path.is_file(), f"{family} raw simulator stdout absent")
        text = path.read_text(encoding="utf-8")
        require(pass_markers[family] in text, f"{family} raw PASS marker missing")
        if family == "residual":
            raw_cycles[family] = one_cycle_marker(
                text,
                r"ACE2_LORA_V4_LAYER0_TWO_TOKEN_FULL_RESIDUAL_REPLAY_PASS positions=2 stages=4 beats=224 cycles=(\d+)",
                family,
            )
        else:
            raw_cycles[family] = one_cycle_marker(
                text,
                rf"ACE2_FULL_LAYER_SIM_CYCLES family={family} cycles=(\d+)",
                family,
            )
        raw_logs[family] = file_record(path)

    compose_text = (layer_dir / "logs/compose.vvp.stdout.log").read_text(encoding="utf-8")
    compose_rows = [tuple(map(int, row)) for row in re.findall(
        r"ACE2_ATTN_COMPOSE_CASE case=(\d+) context=(\d+) tiles=(\d+) cycles=(\d+)", compose_text
    )]
    contexts = prediction["workload_terms"]["compose"]["contexts"]
    require(len(compose_rows) == len(contexts), "compose raw case cycle count changed")
    predicted_compose_cases = prediction["workload_terms"]["compose"]["predicted_case_cycles"]
    require(
        compose_rows == [
            (index, context, 2, predicted_compose_cases[index])
            for index, context in enumerate(contexts)
        ],
        "compose raw per-case cycle reconciliation changed",
    )

    normalized_reported = {family: int(reported_cycles[family]) for family in CYCLE_FAMILIES}
    require(raw_cycles == normalized_reported, f"reported/raw cycle disagreement: {raw_cycles} != {normalized_reported}")
    require(predicted == raw_cycles, f"predicted/raw cycle disagreement: {predicted} != {raw_cycles}")
    return {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_WORKLOAD_CYCLE_RECONCILIATION",
        "predicted_cycles": predicted,
        "reported_cycles": normalized_reported,
        "raw_log_cycles": raw_cycles,
        "per_family_mismatches": {family: 0 for family in CYCLE_FAMILIES},
        "workload_terms": prediction["workload_terms"],
        "raw_stdout_logs": raw_logs,
        "compose_raw_case_cycles": [
            {"case": case, "context": context, "tiles": tiles, "cycles": cycles}
            for case, context, tiles, cycles in compose_rows
        ],
    }


def run_rtl(derived: dict[str, Any]) -> dict[str, Any]:
    rtl = ORIGINAL_REPAIR_RUN_RTL(derived)
    reconciliation = reconcile_cycles(layer_runner.ACTIVE_ATTEMPT, rtl["simulator_cycles"])
    write_json(layer_runner.ACTIVE_ATTEMPT / "cycle_reconciliation.json", reconciliation)
    rtl["cycle_reconciliation"] = file_record(layer_runner.ACTIVE_ATTEMPT / "cycle_reconciliation.json")
    return rtl


layer_runner.run_rtl = run_rtl


def layer_comparison(
    derived: dict[str, Any], rtl: dict[str, Any], projection_check: dict[str, int]
) -> dict[str, Any]:
    require(rtl["projection_results"] == base.EXPECTED_PROJECTION_RESULTS_PER_LAYER, "layer projection count changed")
    cycle_path = layer_runner.ACTIVE_ATTEMPT / "cycle_reconciliation.json"
    cycle = read_json(cycle_path)
    require(cycle["status"] == "PASS_INDEPENDENT_WORKLOAD_CYCLE_RECONCILIATION", "cycle oracle did not pass")
    require(cycle["reported_cycles"] == rtl["simulator_cycles"], "cycle oracle/report changed")
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
        "cycle_prediction_mismatches": sum(cycle["per_family_mismatches"].values()),
        "final_hidden_state_byte_mismatches": 0,
    }
    require(all(value == 0 for value in exact.values()), f"layer mismatch summary changed: {exact}")
    cache = derived["cache_reuse"]
    require(
        cache["cached_k_score_accumulator_changes_when_cached_k_zeroed_heads"] > 0,
        f"layer {derived['layer_id']} K-data counterfactual absent",
    )
    return {
        "schema_version": 3,
        "status": "PASS_EXACT_TWO_POSITION_LAYER_RTL_CYCLE_REPAIR",
        "layer_id": derived["layer_id"],
        "projection_results": projection_check["results"],
        "integer_mismatches": exact,
        "cache_reuse": cache,
        "rtl_kv_cache_comparison": file_record(layer_runner.ACTIVE_ATTEMPT / "rtl_kv_cache_comparison.json"),
        "kv_cache_bytes_compared": kv["kv_cache_bytes_compared"],
        "cycle_reconciliation": file_record(cycle_path),
        "simulator_cycles": rtl["simulator_cycles"],
        "final_hidden_state_sha256_by_position": rtl["final_layer_output_sha256_by_position"],
    }


base.layer_comparison = layer_comparison


def preflight_authorized() -> None:
    require(not CYCLE_PREFLIGHT.exists(), f"cycle preflight already exists: {CYCLE_PREFLIGHT}")
    require(not ATTEMPT.exists(), f"attempt-0003 already exists: {ATTEMPT}")
    bindings = base.validate_bindings(require_attempt_absent=True)
    predecessors = validate_predecessors()
    sources = validate_cycle_sources()
    layers: dict[str, Any] = {}
    vectors: dict[int, dict[str, int]] = {}
    for layer_id in (0, 1):
        layer_dir = ATTEMPT_0002 / "layers" / f"layer-{layer_id:02d}"
        rtl = read_json(layer_dir / "rtl_execution.json")
        reconciliation = reconcile_cycles(layer_dir, rtl["simulator_cycles"])
        vectors[layer_id] = reconciliation["predicted_cycles"]
        layers[f"layer_{layer_id}"] = reconciliation
    require(vectors[0] == base.EXPECTED_LAYER_CYCLES, "focused layer-0 accepted cycle vector changed")
    delta = {family: vectors[1][family] - vectors[0][family] for family in CYCLE_FAMILIES}
    require(delta == FOCUSED_LAYER01_DELTA, f"focused layer-0/layer-1 discrimination changed: {delta}")
    require(sum(delta.values()) == 27352, "focused total layer delta changed")

    CYCLE_PREFLIGHT.mkdir(parents=True)
    try:
        write_json(CYCLE_PREFLIGHT / "cycle_source_bindings.json", sources)
        write_json(
            CYCLE_PREFLIGHT / "layer01_cycle_discrimination.json",
            {
                "schema_version": 1,
                "status": "PASS_LAYER0_LAYER1_WORKLOAD_CYCLE_DISCRIMINATION",
                "layer_0": layers["layer_0"],
                "layer_1": layers["layer_1"],
                "layer_1_minus_layer_0": delta,
                "total_delta": sum(delta.values()),
                "changed_families": [family for family in CYCLE_FAMILIES if delta[family] != 0],
                "invariant_families": [family for family in CYCLE_FAMILIES if delta[family] == 0],
            },
        )
        summary = {
            "schema_version": 1,
            "status": "PASS_HASH_BOUND_WORKLOAD_CYCLE_REPAIR_PREFLIGHT",
            "accepted_hashes": bindings["accepted_hashes"],
            "predecessors": predecessors,
            "cycle_sources": file_record(CYCLE_PREFLIGHT / "cycle_source_bindings.json"),
            "focused_layer01": file_record(CYCLE_PREFLIGHT / "layer01_cycle_discrimination.json"),
            "cycle_families": list(CYCLE_FAMILIES),
            "official_attempt_absent": not ATTEMPT.exists(),
        }
        write_json(CYCLE_PREFLIGHT / "summary.json", summary)
        base.write_sha256s(CYCLE_PREFLIGHT)
        members = base.verify_sha256s(CYCLE_PREFLIGHT)
        print(json.dumps({
            "status": summary["status"],
            "preview": base.relative(CYCLE_PREFLIGHT),
            "members": members,
            "sha256s_sha256": accepted.sha256_file(CYCLE_PREFLIGHT / "SHA256SUMS"),
        }, sort_keys=True))
    except BaseException as exc:
        write_json(CYCLE_PREFLIGHT / "failure.json", {
            "status": "SEALED_FAIL_CYCLE_REPAIR_PREFLIGHT",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        base.write_sha256s(CYCLE_PREFLIGHT)
        raise


def validate_cycle_preflight() -> dict[str, Any]:
    require(CYCLE_PREFLIGHT.is_dir(), "cycle repair preflight is absent")
    members = base.verify_sha256s(CYCLE_PREFLIGHT)
    summary = read_json(CYCLE_PREFLIGHT / "summary.json")
    require(summary["status"] == "PASS_HASH_BOUND_WORKLOAD_CYCLE_REPAIR_PREFLIGHT", "cycle repair preflight did not pass")
    focused = read_json(CYCLE_PREFLIGHT / "layer01_cycle_discrimination.json")
    require(focused["layer_1_minus_layer_0"] == FOCUSED_LAYER01_DELTA, "focused cycle discrimination changed")
    return {
        "path": base.relative(CYCLE_PREFLIGHT),
        "members": members,
        "sha256s": file_record(CYCLE_PREFLIGHT / "SHA256SUMS"),
        "summary": file_record(CYCLE_PREFLIGHT / "summary.json"),
        "focused_layer01": file_record(CYCLE_PREFLIGHT / "layer01_cycle_discrimination.json"),
    }


def cycle_repair_seal_failure(phase: str, exc: BaseException) -> None:
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    failure = {
        "schema_version": 3,
        "status": f"SEALED_FAIL_{phase.upper()}",
        "phase": phase,
        "failure_taxonomy": "WORKLOAD_CYCLE_REPAIR_EXECUTION_GATE_OR_RTL_OFFICIAL_ATTEMPT",
        "root_cause_hypothesis": str(exc),
        "regression_required": "allocate a separately recorded attempt after an evidence-backed fix; never replay attempt-0003",
        "traceback": traceback.format_exc(),
        "official_attempt_consumed": True,
        "predecessor_attempts_preserved": ["attempt-0001", "attempt-0002"],
    }
    write_json(ATTEMPT / "failure.json", failure)
    write_json(ATTEMPT / "status.json", {"status": failure["status"], "sealed": True})


base.seal_failure = cycle_repair_seal_failure
prior.repair_seal_failure = cycle_repair_seal_failure


def prepare_body() -> dict[str, Any]:
    predecessors = validate_predecessors()
    preflight = validate_cycle_preflight()
    sources = validate_cycle_sources()
    base.prepare()

    wrapper_copy = write_binary(
        ATTEMPT / "source/run_lora_v4_two_token_24layer_rtl_cycle_repair.py",
        Path(__file__).read_bytes(),
    )
    authorization = read_json(ATTEMPT / "authorization.json")
    authorization.update({
        "schema_version": 3,
        "kind": "reviewer_directed_hash_bound_workload_cycle_repair_execution_gate",
        "authorized_scope": "exactly-once prepare/run/check for attempt-0003 only",
        "repair_reason": "source-derived workload cycle prediction plus independent raw-log reconciliation for every RTL family",
        "predecessors": predecessors,
        "cycle_repair_preflight": preflight,
        "cycle_source_bindings": sources,
        "authorized_wrapper_copy": wrapper_copy,
    })
    write_json(ATTEMPT / "authorization.json", authorization)

    freeze = read_json(ATTEMPT / "freeze.json")
    freeze.pop("expected_layer_cycles", None)
    freeze.pop("expected_total_simulator_cycles", None)
    freeze.update({
        "schema_version": 3,
        "official_attempt": base.relative(ATTEMPT),
        "predecessors": predecessors,
        "cycle_repair_preflight": preflight,
        "cycle_source_bindings": sources,
        "wrapper_copy": wrapper_copy,
        "cycle_oracle": {
            "classification": "source-derived exact per-layer family cycles independently reconciled to raw simulator stdout",
            "families": list(CYCLE_FAMILIES),
            "fixed_scheduler_cycles": FIXED_CYCLES,
            "focused_layer_1_minus_layer_0": FOCUSED_LAYER01_DELTA,
            "aggregate_expectation": "sum the exact workload-derived per-layer predictions after all 24 layers execute",
        },
        "phase_logs_required": ["prepare", "run", "check"],
    })
    write_json(ATTEMPT / "freeze.json", freeze)
    result = {
        "status": "PREPARED_HASH_BOUND_WORKLOAD_CYCLE_REPAIR_ATTEMPT",
        "attempt": base.relative(ATTEMPT),
        "wrapper_sha256": freeze["wrapper"]["sha256"],
        "attempt_0002_sha256s_sha256": predecessors["attempt_0002"]["sha256s"]["sha256"],
        "cycle_preflight_sha256s_sha256": preflight["sha256s"]["sha256"],
    }
    print(json.dumps(result, sort_keys=True))
    return result


def run_official() -> None:
    require(ATTEMPT.is_dir(), "official attempt was not prepared")
    status = read_json(ATTEMPT / "status.json")
    require(status["status"] == "PREPARED" and status["sealed"] is False, "official attempt is not runnable")
    require(not (ATTEMPT / "run.started.json").exists(), "official run already consumed")
    write_json(ATTEMPT / "run.started.json", {"phase": "run", "monotonic_start": time.monotonic()})
    started = time.monotonic()
    layer_records: list[dict[str, Any]] = []
    aggregate_cycles = {name: 0 for name in CYCLE_FAMILIES}
    aggregate_predicted = {name: 0 for name in CYCLE_FAMILIES}
    projection_results = 0
    try:
        base.validate_bindings(require_attempt_absent=False)
        validate_predecessors()
        validate_cycle_preflight()
        validate_cycle_sources()
        freeze = read_json(ATTEMPT / "freeze.json")
        require(freeze["wrapper"]["sha256"] == accepted.sha256_file(Path(__file__)), "authorized wrapper changed")
        with torch.no_grad(), safe_open(canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
            canonical.ADAPTER, framework="pt", device="cpu"
        ) as adapter:
            states = base.initial_states(weights)
            for layer_id in range(base.LAYERS):
                layer_started = time.monotonic()
                derived, states = base.derive_layer(layer_id, states, weights, adapter)
                layer_dir = ATTEMPT / "layers" / f"layer-{layer_id:02d}"
                layer_dir.mkdir(parents=True)
                if layer_id == 0:
                    write_json(layer_dir / "accepted_fresh_l2_regression.json", base.verify_layer0_regression(derived))
                tensors = base.persist_layer_tensors(layer_dir, derived)
                layer_runner.ACTIVE_ATTEMPT = layer_dir
                vectors = layer_runner.persist_vectors(derived)
                write_json(layer_dir / "conversion.json", base.layer_conversion(derived))
                rtl = layer_runner.run_rtl(derived)
                write_json(layer_dir / "rtl_execution.json", rtl)
                projection_check = base.verify_projection_results(layer_dir, derived)
                comparison = base.layer_comparison(derived, rtl, projection_check)
                write_json(layer_dir / "comparison.json", comparison)
                cycle = read_json(layer_dir / "cycle_reconciliation.json")
                layer_elapsed = time.monotonic() - layer_started
                layer_manifest = {
                    "schema_version": 3,
                    "status": comparison["status"],
                    "layer_id": layer_id,
                    "tensors": tensors,
                    "vectors": vectors,
                    "conversion": file_record(layer_dir / "conversion.json"),
                    "rtl_execution": file_record(layer_dir / "rtl_execution.json"),
                    "comparison": file_record(layer_dir / "comparison.json"),
                    "cycle_reconciliation": file_record(layer_dir / "cycle_reconciliation.json"),
                    "cache_reuse": derived["cache_reuse"],
                    "projection_results": projection_check["results"],
                    "simulator_cycles": rtl["simulator_cycles"],
                    "elapsed_wall_seconds": layer_elapsed,
                    "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                }
                write_json(layer_dir / "manifest.json", layer_manifest)
                layer_records.append({
                    "layer_id": layer_id,
                    "manifest": file_record(layer_dir / "manifest.json"),
                    "projection_results": projection_check["results"],
                    "simulator_cycles": rtl["simulator_cycles"],
                    "predicted_cycles": cycle["predicted_cycles"],
                    "cache_reuse": derived["cache_reuse"],
                    "final_hidden_state_sha256_by_position": rtl["final_layer_output_sha256_by_position"],
                    "elapsed_wall_seconds": layer_elapsed,
                })
                projection_results += projection_check["results"]
                for family in CYCLE_FAMILIES:
                    aggregate_cycles[family] += rtl["simulator_cycles"][family]
                    aggregate_predicted[family] += cycle["predicted_cycles"][family]
                print(json.dumps({
                    "status": "PASS_LAYER_RTL_WORKLOAD_CYCLES",
                    "layer_id": layer_id,
                    "projection_results_cumulative": projection_results,
                    "simulator_cycles": rtl["simulator_cycles"],
                    "cycle_reconciliation": cycle["status"],
                    "cache_reuse": derived["cache_reuse"],
                    "elapsed_wall_seconds": layer_elapsed,
                }, sort_keys=True), flush=True)
                del derived

        require(projection_results == base.EXPECTED_PROJECTION_RESULTS, "aggregate projection count changed")
        require(aggregate_cycles == aggregate_predicted, "aggregate predicted/measured family cycles differ")
        require(len(layer_records) == base.LAYERS, "executed layer count changed")
        require(
            all(record["cache_reuse"]["cached_position0_probability_nonzero_heads"] > 0 for record in layer_records),
            "position-1 cached-K dependence missing in at least one layer",
        )
        require(
            all(record["cache_reuse"]["compose_output_changes_when_cached_v_zeroed_heads"] > 0 for record in layer_records),
            "position-1 cached-V dependence missing in at least one layer",
        )
        total_cycles = sum(aggregate_cycles.values())
        run_summary = {
            "schema_version": 3,
            "status": "PASS_24LAYER_RTL_RUN_UNCHECKED",
            "layers_executed": base.LAYERS,
            "projection_results": projection_results,
            "aggregate_family_cycles": aggregate_cycles,
            "aggregate_predicted_family_cycles": aggregate_predicted,
            "total_simulator_cycles": total_cycles,
            "total_predicted_simulator_cycles": sum(aggregate_predicted.values()),
            "simulator_cycle_classification": "family_local_simulator_cycles_not_accelerator_latency",
            "cycle_oracle_status": "PASS_24LAYER_SOURCE_DERIVED_AND_RAW_LOG_RECONCILED",
            "elapsed_wall_seconds": time.monotonic() - started,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "layers": layer_records,
            "final_layer23_hidden_state_sha256_by_position": layer_records[-1]["final_hidden_state_sha256_by_position"],
        }
        write_json(ATTEMPT / "run_summary.json", run_summary)
        write_json(ATTEMPT / "status.json", {"status": run_summary["status"], "sealed": False})
        print(json.dumps(run_summary, sort_keys=True), flush=True)
    except BaseException as exc:
        cycle_repair_seal_failure("run", exc)
        raise


def run_body() -> dict[str, Any]:
    validate_predecessors()
    validate_cycle_preflight()
    prior.require_completed_phase("prepare")
    run_official()
    repair = prior.validate_run_repair_evidence()
    summary = read_json(ATTEMPT / "run_summary.json")
    require(summary["aggregate_family_cycles"] == summary["aggregate_predicted_family_cycles"], "run cycle aggregate changed")
    result = {
        "status": "PASS_RUN_WORKLOAD_CYCLE_REPAIR_EVIDENCE",
        "cycle_oracle_status": summary["cycle_oracle_status"],
        "total_simulator_cycles": summary["total_simulator_cycles"],
        "repair_evidence": repair,
    }
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
    predecessors = validate_predecessors()
    preflight = validate_cycle_preflight()
    sources = validate_cycle_sources()
    prepare_logs = prior.require_completed_phase("prepare")
    run_logs = prior.require_completed_phase("run")
    freeze = read_json(ATTEMPT / "freeze.json")
    run_summary = read_json(ATTEMPT / "run_summary.json")
    require(freeze["wrapper"]["sha256"] == accepted.sha256_file(Path(__file__)), "cycle repair wrapper changed")
    require(run_summary["layers_executed"] == base.LAYERS, "layer count changed")
    require(run_summary["projection_results"] == base.EXPECTED_PROJECTION_RESULTS, "projection count changed")
    require(run_summary["aggregate_family_cycles"] == run_summary["aggregate_predicted_family_cycles"], "aggregate cycle oracle changed")
    require(run_summary["total_simulator_cycles"] == sum(run_summary["aggregate_family_cycles"].values()), "total cycle sum changed")

    aggregate_mismatches: dict[str, int] = {}
    aggregate_predicted = {family: 0 for family in CYCLE_FAMILIES}
    cache_dependency_layers = 0
    k_counterfactual_layers = 0
    rtl_cache_layers = 0
    rtl_cache_bytes = 0
    cycle_reconciled_layers = 0
    final_hashes: list[str] | None = None
    for layer_id in range(base.LAYERS):
        layer_dir = ATTEMPT / "layers" / f"layer-{layer_id:02d}"
        comparison = read_json(layer_dir / "comparison.json")
        rtl = read_json(layer_dir / "rtl_execution.json")
        manifest = read_json(layer_dir / "manifest.json")
        require(
            comparison["status"] == manifest["status"] == "PASS_EXACT_TWO_POSITION_LAYER_RTL_CYCLE_REPAIR",
            f"layer {layer_id} status changed",
        )
        require(comparison["projection_results"] == base.EXPECTED_PROJECTION_RESULTS_PER_LAYER, f"layer {layer_id} projection count changed")
        recorded_cycle = read_json(layer_dir / "cycle_reconciliation.json")
        independently_recomputed = reconcile_cycles(layer_dir, rtl["simulator_cycles"])
        require(recorded_cycle == independently_recomputed, f"layer {layer_id} cycle reconciliation record changed")
        for family in CYCLE_FAMILIES:
            aggregate_predicted[family] += independently_recomputed["predicted_cycles"][family]
        cycle_reconciled_layers += 1
        for name, value in comparison["integer_mismatches"].items():
            aggregate_mismatches[name] = aggregate_mismatches.get(name, 0) + int(value)
        cache = comparison["cache_reuse"]
        require(cache["cached_position0_probability_nonzero_heads"] > 0, f"layer {layer_id} cached K slot absent")
        require(cache["cached_k_score_accumulator_changes_when_cached_k_zeroed_heads"] > 0, f"layer {layer_id} cached K data dependence absent")
        require(cache["compose_output_changes_when_cached_v_zeroed_heads"] > 0, f"layer {layer_id} cached V dependence absent")
        cache_dependency_layers += 1
        k_counterfactual_layers += 1
        kv = rtl["kv_cache_rtl_comparison"]
        require(kv["kv_cache_byte_mismatches"] == 0, f"layer {layer_id} measured RTL K/V mismatch")
        require(kv["kv_cache_bytes_compared"] == prior.KV_BYTES_PER_LAYER, f"layer {layer_id} RTL K/V byte count changed")
        rtl_cache_layers += 1
        rtl_cache_bytes += kv["kv_cache_bytes_compared"]
        if layer_id == base.LAYERS - 1:
            final_hashes = comparison["final_hidden_state_sha256_by_position"]

    require(all(value == 0 for value in aggregate_mismatches.values()), f"aggregate integer mismatch: {aggregate_mismatches}")
    require(cache_dependency_layers == k_counterfactual_layers == rtl_cache_layers == cycle_reconciled_layers == base.LAYERS, "evidence layer count changed")
    require(rtl_cache_bytes == base.LAYERS * prior.KV_BYTES_PER_LAYER, "aggregate RTL K/V byte count changed")
    require(aggregate_predicted == run_summary["aggregate_family_cycles"], "check-time aggregate cycle prediction changed")
    require(final_hashes is not None and len(final_hashes) == 2, "final layer-23 hidden states absent")

    check_result = {
        "schema_version": 3,
        "status": "PASS_24LAYER_EXACT_INTEGER_CYCLE_REPAIR_CHECK_AWAITING_INDEPENDENT_REVIEW",
        "layers": base.LAYERS,
        "projection_results": base.EXPECTED_PROJECTION_RESULTS,
        "integer_mismatches": aggregate_mismatches,
        "position1_cache_dependency_layers": cache_dependency_layers,
        "k_data_counterfactual_layers": k_counterfactual_layers,
        "workload_bound_rtl_kv_cache_layers": rtl_cache_layers,
        "cycle_reconciled_layers": cycle_reconciled_layers,
        "rtl_kv_cache_bytes_compared": rtl_cache_bytes,
        "rtl_kv_cache_byte_mismatches": 0,
        "final_layer23_hidden_state_sha256_by_position": final_hashes,
        "aggregate_family_cycles": run_summary["aggregate_family_cycles"],
        "aggregate_predicted_family_cycles": aggregate_predicted,
        "total_simulator_cycles": run_summary["total_simulator_cycles"],
        "elapsed_wall_seconds": run_summary["elapsed_wall_seconds"],
        "peak_rss_kib": run_summary["peak_rss_kib"],
        "tool_versions": bindings["tool_versions"],
        "accepted_hashes": bindings["accepted_hashes"],
        "predecessors": predecessors,
        "cycle_repair_preflight": preflight,
        "cycle_source_bindings": sources,
        "retained_completed_phase_logs": {"prepare": prepare_logs, "run": run_logs},
        "review_required": True,
    }
    write_json(ATTEMPT / "check.json", check_result)
    manifest = {
        "schema_version": 3,
        "status": check_result["status"],
        "claim_boundary": read_json(base.CONTRACT)["claim_boundary"],
        "authorization": file_record(ATTEMPT / "authorization.json"),
        "freeze": file_record(ATTEMPT / "freeze.json"),
        "run_summary": file_record(ATTEMPT / "run_summary.json"),
        "check": file_record(ATTEMPT / "check.json"),
        "wrapper": file_record(Path(__file__)),
        "wrapper_copy": file_record(ATTEMPT / "source/run_lora_v4_two_token_24layer_rtl_cycle_repair.py"),
        "accepted_artifacts": bindings["accepted_hashes"],
        "predecessors": predecessors,
        "cycle_repair_preflight": preflight,
        "cycle_source_bindings": sources,
        "official_attempt_consumed": True,
        "immutable_after_check": True,
        "independent_review_required": True,
    }
    write_json(ATTEMPT / "manifest.json", manifest)
    write_json(ATTEMPT / "status.json", {"status": check_result["status"], "sealed": False})
    print(json.dumps(check_result, sort_keys=True))
    return check_result


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
        raise SystemExit(prior.execute_phase("prepare", prepare_body))
    if args.run:
        raise SystemExit(prior.execute_phase("run", run_body))
    raise SystemExit(prior.execute_phase("check", check_body))


if __name__ == "__main__":
    main()
