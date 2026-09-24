#!/usr/bin/env python3
"""Fail-closed readback for the additive 24-layer RTL cycle erratum.

This checker is intentionally read-only.  It verifies the immutable base
contracts, the pre-attempt cycle oracle and source bindings, the published
24-layer schedule, and the retained attempt-0003 reconciliation without
creating or modifying an attempt namespace.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "design/LORA_V4_TWO_TOKEN_24LAYER_RTL_CONTRACT_ERRATUM_V1.json"
FAMILIES = ("projection", "rmsnorm", "rope", "score", "softmax", "compose", "silu", "residual")
VECTOR_FILES = (
    "projection_shift.hex",
    "rmsnorm_vectors.svh",
    "rope_vectors.svh",
    "attention_score_vectors.svh",
    "softmax_vectors.svh",
    "attention_compose_vectors.svh",
    "silu_gate_vectors.svh",
    "residual_vectors.svh",
    "mlp_residual_vectors.svh",
)


class ErratumCheckError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ErratumCheckError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def verify_record(record: dict[str, Any]) -> Path:
    require(set(record) >= {"path", "bytes", "sha256"}, f"incomplete file record: {record}")
    path = ROOT / str(record["path"])
    require(path.is_file(), f"bound file absent: {record['path']}")
    require(path.stat().st_size == int(record["bytes"]), f"bound byte count changed: {record['path']}")
    require(sha256_file(path) == record["sha256"], f"bound hash changed: {record['path']}")
    return path


def verify_sidecar(path: Path) -> None:
    sidecar = Path(f"{path}.sha256")
    require(sidecar.is_file(), f"checksum sidecar absent: {sidecar.relative_to(ROOT)}")
    rows = sidecar.read_text(encoding="utf-8").splitlines()
    require(len(rows) == 1, f"checksum sidecar must contain one row: {sidecar.relative_to(ROOT)}")
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", rows[0])
    require(match is not None, f"malformed checksum sidecar: {sidecar.relative_to(ROOT)}")
    require(match.group(2) == path.name, f"checksum sidecar basename changed: {sidecar.relative_to(ROOT)}")
    require(match.group(1) == sha256_file(path), f"checksum sidecar mismatch: {path.relative_to(ROOT)}")


def checksum_index(root: Path, expected_sha256: str) -> dict[str, str]:
    sums = root / "SHA256SUMS"
    require(sums.is_file(), f"attempt checksum index absent: {sums.relative_to(ROOT)}")
    require(sha256_file(sums) == expected_sha256, f"attempt checksum index changed: {sums.relative_to(ROOT)}")
    result: dict[str, str] = {}
    for row in sums.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", row)
        require(match is not None, f"malformed attempt checksum row: {row}")
        require(match.group(2) not in result, f"duplicate attempt checksum member: {match.group(2)}")
        result[match.group(2)] = match.group(1)
    return result


def verify_indexed_member(root: Path, index: dict[str, str], relative: str) -> Path:
    require(relative in index, f"consumed attempt member not indexed: {relative}")
    path = root / relative
    require(path.is_file(), f"consumed attempt member absent: {relative}")
    require(sha256_file(path) == index[relative], f"consumed attempt member changed: {relative}")
    return path


def markdown_integer(cell: str) -> int:
    normalized = cell.strip().replace("**", "").replace(",", "")
    require(re.fullmatch(r"-?\d+", normalized) is not None, f"invalid Markdown cycle value: {cell}")
    return int(normalized)


def parse_spec_schedule(path: Path) -> tuple[list[dict[str, int]], dict[str, int]]:
    rows: list[dict[str, int]] = []
    aggregate: dict[str, int] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 10:
            continue
        label = cells[0].replace("**", "")
        if label.isdigit():
            values = [markdown_integer(cell) for cell in cells[1:]]
            rows.append({"layer_id": int(label), **dict(zip((*FAMILIES, "aggregate"), values, strict=True))})
        elif label == "All 24":
            values = [markdown_integer(cell) for cell in cells[1:]]
            aggregate = dict(zip((*FAMILIES, "aggregate"), values, strict=True))
    require(len(rows) == 24 and aggregate is not None, "specification erratum cycle table is incomplete")
    return rows, aggregate


def indexed_decimal_assignments(path: Path, name: str) -> list[int]:
    text = path.read_text(encoding="utf-8")
    rows = [
        (int(index), int(value))
        for index, value in re.findall(rf"{re.escape(name)}\[(\d+)\] = (?:\d+'d)?(\d+);", text)
    ]
    rows.sort()
    require(rows and [index for index, _ in rows] == list(range(len(rows))), f"invalid assignments: {name}")
    return [value for _, value in rows]


def indexed_bits(path: Path, name: str) -> list[int]:
    text = path.read_text(encoding="utf-8")
    rows = [(int(index), int(value)) for index, value in re.findall(rf"{re.escape(name)}\[(\d+)\] = 1'b([01]);", text)]
    rows.sort()
    require(rows and [index for index, _ in rows] == list(range(len(rows))), f"invalid bit assignments: {name}")
    return [value for _, value in rows]


def indexed_hex128(path: Path, name: str) -> list[int]:
    text = path.read_text(encoding="utf-8")
    rows = [
        (int(index), int(value, 16))
        for index, value in re.findall(rf"{re.escape(name)}\[(\d+)\] = 128'h([0-9a-fA-F]+);", text)
    ]
    rows.sort()
    require(rows and [index for index, _ in rows] == list(range(len(rows))), f"invalid 128-bit assignments: {name}")
    return [value for _, value in rows]


def signed(value: int, width: int) -> int:
    return value - (1 << width) if value & (1 << (width - 1)) else value


def lanes_s8(word: int) -> list[int]:
    return [signed(value, 8) for value in word.to_bytes(16, "little")]


def lanes_s16(word: int) -> list[int]:
    raw = word.to_bytes(16, "little")
    return [signed(int.from_bytes(raw[offset : offset + 2], "little"), 16) for offset in range(0, 16, 2)]


def predict_layer(vectors: Path, fixed: dict[str, int]) -> dict[str, int]:
    shifts = [int(row, 16) for row in (vectors / "projection_shift.hex").read_text(encoding="utf-8").splitlines() if row]
    require(len(shifts) == 25344 and all(0 <= value <= 63 for value in shifts), "projection workload changed")
    projection = fixed["projection"] + sum(shifts)

    rms_text = (vectors / "rmsnorm_vectors.svh").read_text(encoding="utf-8")
    rms_rows = [(int(index), int(value, 16)) for index, value in re.findall(r"test_expected_sumsq\[(\d+)\] = 48'h([0-9a-fA-F]+);", rms_text)]
    rms_rows.sort()
    require([index for index, _ in rms_rows] == list(range(4)), "RMSNorm workload changed")
    rounded_means = [(value + 448) // 896 for _, value in rms_rows]
    roots = [math.isqrt(value - 1) + 1 if value else 0 for value in rounded_means]
    rmsnorm = fixed["rmsnorm"] + 2 * sum(roots)

    rope_path = vectors / "rope_vectors.svh"
    input_words = [lanes_s8(value) for value in indexed_hex128(rope_path, "rope_input_beats")]
    scale_words = [lanes_s16(value) for value in indexed_hex128(rope_path, "rope_scale_beats")]
    cos_words = [lanes_s16(value) for value in indexed_hex128(rope_path, "rope_cos_beats")]
    sin_words = [lanes_s16(value) for value in indexed_hex128(rope_path, "rope_sin_beats")]
    require(len(input_words) == 224, "RoPE input workload changed")
    require(len(scale_words) == len(cos_words) == len(sin_words) == 448, "RoPE metadata workload changed")
    negative_paths = 0
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
                negative_paths += sum((
                    (act < 0) ^ (scale < 0),
                    (pair < 0) ^ (pair_scale < 0),
                    (act_scaled < 0) ^ (cosine < 0),
                    (pair_scaled < 0) ^ (sine < 0),
                    rotated < 0,
                ))
    rope = fixed["rope"] + 3 * negative_paths

    score_contexts = indexed_decimal_assignments(vectors / "attention_score_vectors.svh", "attn_score_context_count")
    require(len(score_contexts) == 28 and all(value in (1, 2) for value in score_contexts), "score workload changed")
    score = fixed["score"] + 134 * sum(score_contexts)

    softmax_contexts = indexed_decimal_assignments(vectors / "softmax_vectors.svh", "softmax_context_count")
    require(len(softmax_contexts) == 28 and all(value in (1, 2) for value in softmax_contexts), "softmax workload changed")
    softmax = fixed["softmax"] + 36 * len(softmax_contexts) + 49 * sum(softmax_contexts)

    compose_contexts = indexed_decimal_assignments(vectors / "attention_compose_vectors.svh", "attn_compose_context_count")
    require(len(compose_contexts) == 28 and all(value == 9 for value in compose_contexts), "compose workload changed")
    compose = fixed["compose"] + 1214 + 1215 * 27

    silu_lengths = indexed_decimal_assignments(vectors / "silu_gate_vectors.svh", "silu_case_length")
    silu_shifts = indexed_decimal_assignments(vectors / "silu_gate_vectors.svh", "silu_case_right_shift")
    require(silu_lengths == [4864, 4864] and len(silu_shifts) == 2, "SiLU workload changed")
    silu = fixed["silu"] + sum(length * shift for length, shift in zip(silu_lengths, silu_shifts, strict=True))

    attention_saturation = indexed_bits(vectors / "residual_vectors.svh", "residual_expected_saturation")
    final_saturation = indexed_bits(vectors / "mlp_residual_vectors.svh", "mlp_residual_expected_saturation")
    require(len(attention_saturation) == len(final_saturation) == 2, "residual workload changed")
    residual = fixed["residual"] + 8 * (sum(attention_saturation) + sum(final_saturation))

    return {
        "projection": projection,
        "rmsnorm": rmsnorm,
        "rope": rope,
        "score": score,
        "softmax": softmax,
        "compose": compose,
        "silu": silu,
        "residual": residual,
    }


def main() -> int:
    contract = load_json(CONTRACT)
    require(contract.get("status") == "NORMATIVE_ERRATUM_ACTIVE", "contract erratum is not active")
    require(contract.get("effective_contract_revision") == "2.1", "effective contract revision changed")

    for record in contract["base_contracts"].values():
        path = verify_record(record)
        verify_sidecar(path)
    spec_erratum_path = verify_record(contract["errata"]["specification"])
    interface_path = verify_record(contract["errata"]["interface"])
    checker_path = verify_record(contract["fail_closed_consistency_check"]["checker"])
    require(checker_path.resolve() == Path(__file__).resolve(), "wrong checker entry point")
    verify_sidecar(spec_erratum_path)
    verify_sidecar(interface_path)
    verify_sidecar(CONTRACT)
    verify_sidecar(checker_path)

    interface = load_json(interface_path)
    require(interface.get("status") == "NORMATIVE_ERRATUM_ACTIVE", "interface erratum is not active")
    require(interface.get("effective_contract_revision") == "2.1", "interface revision changed")
    require(interface["normative_precedence"]["conflict_rule"] == "ERRATUM_WINS_FOR_CYCLE_FIELDS_ONLY", "cycle precedence changed")
    require(contract["normative_precedence"]["conflict_rule"] == "ERRATUM_WINS_FOR_CYCLE_FIELDS_ONLY", "contract cycle precedence changed")

    base_interface = load_json(ROOT / contract["base_contracts"]["benchmark_interface"]["path"])
    base_contract = load_json(ROOT / contract["base_contracts"]["mission_contract"]["path"])
    require(base_interface["latency_throughput_expectations"]["all_layers_expected_simulator_cycles_exact"]["aggregate"] == 709971648, "superseded base interface cycle table changed")
    require(base_contract["protocol_and_scenario_contract"]["exact_aggregate_simulator_cycles"] == 709971648, "superseded base contract cycle value changed")
    require(interface["cycle_oracle"]["superseded_repeated_layer0_values"]["aggregate"] == 709971648, "superseded cycle value is not explicit")
    require(interface["acceptance"]["obsolete_aggregate_rejected"] == 709971648, "obsolete cycle value is not rejected")
    require(interface["cycle_oracle"]["aggregate_delta_from_superseded_table"] == -866437, "cycle-table delta changed")

    for record in contract["pre_attempt_oracle_bindings"].values():
        verify_record(record)
    preflight_root = ROOT / contract["pre_attempt_oracle"]["preflight_root"]
    preflight_index = checksum_index(preflight_root, contract["pre_attempt_oracle"]["preflight_sha256s_sha256"])
    for relative in ("summary.json", "layer01_cycle_discrimination.json", "cycle_source_bindings.json"):
        verify_indexed_member(preflight_root, preflight_index, relative)

    source_bindings = load_json(preflight_root / "cycle_source_bindings.json")
    require(source_bindings.get("status") == "PASS_HASH_BOUND_CYCLE_SOURCES", "cycle source binding status changed")
    for record in source_bindings["members"].values():
        verify_record(record)

    fixed = {name: int(value) for name, value in interface["cycle_oracle"]["fixed_scheduler_cycles"].items()}
    require(tuple(fixed) == FAMILIES, "fixed-cycle family ordering changed")
    freeze = load_json(ROOT / contract["pre_attempt_oracle_bindings"]["freeze"]["path"])
    authorization = load_json(ROOT / contract["pre_attempt_oracle_bindings"]["authorization"]["path"])
    require(freeze["cycle_oracle"]["fixed_scheduler_cycles"] == fixed, "freeze/oracle fixed-cycle disagreement")
    require(freeze["cycle_oracle"]["families"] == list(FAMILIES), "freeze/oracle family disagreement")
    require(authorization["authorized_wrapper"]["sha256"] == contract["pre_attempt_oracle_bindings"]["oracle_wrapper"]["sha256"], "authorization wrapper binding changed")
    require(authorization["cycle_repair_preflight"]["sha256s"]["sha256"] == contract["pre_attempt_oracle"]["preflight_sha256s_sha256"], "authorization preflight binding changed")

    attempts = contract["consumed_attempt_integrity"]
    for name in ("attempt_0001", "attempt_0002"):
        root = ROOT / attempts[name]["path"]
        checksum_index(root, attempts[name]["sha256s_sha256"])
    attempt_root = ROOT / attempts["attempt_0003"]["path"]
    attempt_index = checksum_index(attempt_root, attempts["attempt_0003"]["sha256s_sha256"])
    for relative in ("authorization.json", "freeze.json", "run_summary.json", "check.json"):
        verify_indexed_member(attempt_root, attempt_index, relative)
    for record in contract["accepted_reconciliation_bindings"].values():
        verify_record(record)

    schedule_rows = interface["cycle_oracle"]["per_layer_expected_simulator_cycles_exact"]
    require(len(schedule_rows) == 24, "published cycle schedule must contain 24 layers")
    spec_rows, spec_aggregate = parse_spec_schedule(spec_erratum_path)
    require(spec_rows == schedule_rows, "specification/interface per-layer cycle schedule disagreement")
    published: list[dict[str, int]] = []
    aggregate = {family: 0 for family in FAMILIES}
    for layer_id, row in enumerate(schedule_rows):
        require(row["layer_id"] == layer_id, f"published layer ordering changed at layer {layer_id}")
        expected = {family: int(row[family]) for family in FAMILIES}
        require(set(row) == {"layer_id", *FAMILIES, "aggregate"}, f"published layer fields changed at layer {layer_id}")
        require(int(row["aggregate"]) == sum(expected.values()), f"published layer aggregate changed at layer {layer_id}")
        vector_prefix = f"layers/layer-{layer_id:02d}/vectors"
        for filename in VECTOR_FILES:
            verify_indexed_member(attempt_root, attempt_index, f"{vector_prefix}/{filename}")
        recomputed = predict_layer(attempt_root / vector_prefix, fixed)
        require(recomputed == expected, f"pre-attempt oracle disagreement at layer {layer_id}: {recomputed} != {expected}")
        published.append(expected)
        for family in FAMILIES:
            aggregate[family] += expected[family]

    published_aggregate = interface["cycle_oracle"]["all_layers_expected_simulator_cycles_exact"]
    require(set(published_aggregate) == {*FAMILIES, "aggregate"}, "published aggregate fields changed")
    require({family: int(published_aggregate[family]) for family in FAMILIES} == aggregate, "published family aggregates changed")
    require(int(published_aggregate["aggregate"]) == sum(aggregate.values()) == 709105211, "published total cycle oracle changed")
    require(spec_aggregate == published_aggregate, "specification/interface aggregate cycle disagreement")
    require(contract["normative_cycle_acceptance"]["exact_family_simulator_cycles"] == aggregate, "contract/interface family aggregate disagreement")
    require(contract["normative_cycle_acceptance"]["exact_aggregate_simulator_cycles"] == 709105211, "contract total cycle oracle changed")

    run_summary = load_json(attempt_root / "run_summary.json")
    check = load_json(attempt_root / "check.json")
    for layer_id, row in enumerate(run_summary["layers"]):
        require(row["layer_id"] == layer_id, f"run summary layer ordering changed at layer {layer_id}")
        require(row["predicted_cycles"] == published[layer_id], f"run/published prediction disagreement at layer {layer_id}")
        require(row["simulator_cycles"] == published[layer_id], f"run/published measured disagreement at layer {layer_id}")
    require(run_summary["aggregate_family_cycles"] == aggregate, "run/published family aggregate disagreement")
    require(run_summary["aggregate_predicted_family_cycles"] == aggregate, "run/published predicted aggregate disagreement")
    require(run_summary["total_simulator_cycles"] == 709105211, "run total cycle count changed")
    require(check["aggregate_family_cycles"] == aggregate, "check/published family aggregate disagreement")
    require(check["aggregate_predicted_family_cycles"] == aggregate, "check/published predicted aggregate disagreement")
    require(check["total_simulator_cycles"] == 709105211, "check total cycle count changed")

    print(json.dumps({
        "status": "PASS_ERRATUM_V1_FAIL_CLOSED_CONSISTENCY",
        "effective_contract_revision": "2.1",
        "layers": 24,
        "families": list(FAMILIES),
        "exact_aggregate_simulator_cycles": 709105211,
        "consumed_attempts_modified": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ErratumCheckError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"FAIL_ERRATUM_V1_CONSISTENCY: {exc}", file=sys.stderr)
        raise SystemExit(1)
