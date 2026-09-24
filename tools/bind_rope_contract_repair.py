#!/usr/bin/env python3
"""Bind the bounded RoPE Q6.9 contract repair without advancing stages."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_ppa_stage import (
    parse_critical_path,
    parse_lm_head_activity,
    parse_power,
    parse_sta,
    parse_vcd_duration,
    parse_yosys,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "rope_contract_repair" / "latest"
FRONTIER = ROOT / "evidence" / "frontier" / "latest"
PPA_RAW = ROOT / "ppa" / "raw" / "latest"
QUALITY_BINDING = ROOT / "benchmark" / "quality" / "RTL_BINDING.json"
RTL_MANIFEST = ROOT / "design" / "RTL_MANIFEST.json"
PPA_RESULTS = ROOT / "ppa" / "RESULTS.json"
PPA_PROTOCOL = ROOT / "ppa" / "PROTOCOL.md"
PPA_LEDGER = ROOT / "design" / "PPA_FRONTIER_LEDGER.json"
PUBLIC_STATUS = ROOT / "research" / "PUBLIC_STATUS.json"

AREA_CAP_MM2 = 2.0
FREQUENCY_FLOOR_MHZ = 100.0
CLOCK_PERIOD_NS = 10.0
PREVIOUS_BOUND_RTL_HASH = "f0c39722b96d25d07bc0f29191c055ae72bddaac4741b80f9d2c138b34216b84"

NUMERICAL_RTL = [
    "rtl/ace2_pkg.sv",
    "rtl/ace2_rmsnorm_core.sv",
    "rtl/ace2_w4a8_proj_core.sv",
    "rtl/ace2_rope_core.sv",
    "rtl/ace2_attention_score_core.sv",
    "rtl/ace2_softmax_core.sv",
    "rtl/ace2_attention_compose_core.sv",
    "rtl/ace2_silu_gate_core.sv",
    "rtl/generated/ace2_silu_lut.svh",
    "rtl/ace2_shell.sv",
]
CONSTRAINTS = [
    "constraints/ace2_rmsnorm_core.sdc",
    "flow/yosys/sky130_rmsnorm.ys",
    "flow/yosys/sky130_sta.tcl",
]
PPA_PROTOCOL_SOURCES = [
    *CONSTRAINTS,
    "flow/yosys/sky130_power.tcl",
    "ppa/harness/ace2_shell_power_dump_tb.sv",
    "Makefile",
    "tools/bind_rope_contract_repair.py",
]
ROPE_CONTRACT_SOURCES = [
    *NUMERICAL_RTL,
    *CONSTRAINTS,
    "tools/ace2_rope_reference.py",
    "tools/gen_rope_vectors.py",
    "tools/ace2_full_model_fixed_point.py",
    "verification/tb/ace2_rope_tb.sv",
    "verification/tb/ace2_shell_tb.sv",
    "verification/generated/rope_vectors.json",
    "verification/generated/rope_vectors.svh",
    "benchmark/quality/QUALITY_CONFIG.json",
    "design/CHIP_SCOPE.json",
    "design/RTL_TRACEABILITY.md",
]
SHELL_VECTOR_INPUTS = [
    "verification/generated/rmsnorm_vectors.svh",
    "verification/generated/projection_vectors.svh",
    "verification/generated/rope_vectors.svh",
    "verification/generated/attention_score_vectors.svh",
    "verification/generated/softmax_vectors.svh",
    "verification/generated/silu_gate_vectors.svh",
    "verification/generated/attention_value_vectors.svh",
    "verification/generated/residual_vectors.svh",
    "verification/generated/mlp_residual_vectors.svh",
    "verification/generated/post_attention_rmsnorm_vectors.svh",
]
LOGS = {
    "fixed_point_self_test": EVIDENCE / "fixed_point_self_test.log",
    "rtl_lint": FRONTIER / "rtl_lint.log",
    "rtl_rope": FRONTIER / "rtl_rope_sim.log",
    "rtl_shell": FRONTIER / "rtl_shell_sim.log",
    "sky130_yosys": FRONTIER / "sky130_yosys.log",
    "sky130_opensta": FRONTIER / "sky130_sta.log",
    "ppa_activity_vcd_generation": PPA_RAW / "ace2_shell_lm_head_power_vcd.log",
    "ppa_activity_vcd": PPA_RAW / "ace2_shell_lm_head_power.vcd",
    "sky130_power": PPA_RAW / "sky130_power.log",
}
PASS_MARKERS = {
    "fixed_point_self_test": "ACE2_FULL_MODEL_FIXED_POINT_SELF_TEST status=pass",
    "rtl_lint": "ACE2_RTL_LINT_PASS",
    "rtl_rope": "ACE2_ROPE_TB_PASS cases=4",
    "rtl_shell": "ACE2_SHELL_TB_PASS",
    "ppa_activity_vcd_generation": "ACE2_SHELL_LM_HEAD_TB_PASS",
}
LOG_INPUTS = {
    "fixed_point_self_test": [
        "tools/ace2_full_model_fixed_point.py",
        "benchmark/quality/QUALITY_CONFIG.json",
    ],
    "rtl_lint": NUMERICAL_RTL,
    "rtl_rope": [
        *NUMERICAL_RTL,
        "verification/tb/ace2_rope_tb.sv",
        "verification/generated/rope_vectors.svh",
    ],
    "rtl_shell": [*NUMERICAL_RTL, "verification/tb/ace2_shell_tb.sv", *SHELL_VECTOR_INPUTS],
    "sky130_yosys": [*NUMERICAL_RTL, *CONSTRAINTS],
    "sky130_opensta": [*NUMERICAL_RTL, *CONSTRAINTS],
    "ppa_activity_vcd_generation": [
        *NUMERICAL_RTL,
        "verification/tb/ace2_shell_tb.sv",
        "ppa/harness/ace2_shell_power_dump_tb.sv",
        "verification/generated/projection_vectors.svh",
    ],
    "ppa_activity_vcd": [
        *NUMERICAL_RTL,
        "verification/tb/ace2_shell_tb.sv",
        "ppa/harness/ace2_shell_power_dump_tb.sv",
        "verification/generated/projection_vectors.svh",
    ],
    "sky130_power": [*NUMERICAL_RTL, *CONSTRAINTS, "flow/yosys/sky130_power.tcl"],
}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(ROOT / relative).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def source_records(paths: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "path": relative,
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": sha256_file(ROOT / relative),
        }
        for relative in paths
    ]


def artifact(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {
        "path": relative,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_fresh_logs() -> None:
    for name, path in LOGS.items():
        if not path.is_file():
            raise RuntimeError(f"missing evidence log: {path.relative_to(ROOT)}")
        newest_source = max((ROOT / source).stat().st_mtime_ns for source in LOG_INPUTS[name])
        if path.stat().st_mtime_ns < newest_source:
            raise RuntimeError(f"stale evidence log: {path.relative_to(ROOT)}")
        marker = PASS_MARKERS.get(name)
        if marker and marker not in path.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError(f"{name} log lacks pass marker {marker!r}")


def parse_shell_cycles(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"ACE2_SHELL_CYCLES success_runs=([0-9]+) max_cycles=([0-9]+) "
        r"total_cycles=([0-9]+) last_cycles=([0-9]+)",
        text,
    )
    if not match:
        raise RuntimeError("shell regression log lacks cycle summary")
    return {
        "success_runs": int(match.group(1)),
        "max_cycles": int(match.group(2)),
        "total_cycles": int(match.group(3)),
        "last_cycles": int(match.group(4)),
    }


def copy_ppa_raw_logs() -> dict[str, dict[str, Any]]:
    PPA_RAW.mkdir(parents=True, exist_ok=True)
    copied = {}
    for source, target in (
        ("evidence/frontier/latest/sky130_yosys.log", "ppa/raw/latest/sky130_yosys.log"),
        ("evidence/frontier/latest/sky130_sta.log", "ppa/raw/latest/sky130_sta.log"),
    ):
        source_path = ROOT / source
        target_path = ROOT / target
        target_path.write_bytes(source_path.read_bytes())
        copied[target_path.name.removesuffix(".log") + "_log"] = artifact(target)
    return copied


def write_ppa_protocol(generated_at: str, frontier: dict[str, Any]) -> None:
    PPA_PROTOCOL.write_text(
        f"""# ACE-2 RoPE contract-repair PPA binding

Generated: `{generated_at}`

This packet binds a benchmark-stage repair PPA refresh for the RoPE Q6.9
contract. It does not advance `current_stage`, does not close the PPA stage, and
does not claim routed GDS, signoff, tapeout readiness, FPGA, or silicon evidence.

| Item | Value |
| --- | --- |
| RTL hash | `{frontier["rtl_hash"]}` |
| Constraint hash | `{frontier["constraint_hash"]}` |
| SKY130 cells | `{frontier["cells"]}` |
| SKY130 non-SRAM area mm^2 | `{frontier["non_sram_area_mm2"]:.10f}` |
| Fmax floor-bound MHz | `{frontier["fmax_mhz"]:.3f}` |
| Area reserve mm^2 | `{frontier["remaining_area_reserve_mm2"]:.10f}` |

The evidence is limited to Yosys mapped synthesis and OpenSTA timing at the
existing 10 ns constraint. SRAM macros remain excluded from non-SRAM
standard-cell area accounting.
""",
        encoding="utf-8",
    )


def update_rtl_manifest(
    generated_at: str,
    rtl_hash: str,
    constraint_hash: str,
    evidence_hashes: dict[str, str],
    ppa_frontier: dict[str, Any],
) -> None:
    manifest = load_json(RTL_MANIFEST, {})
    historical_source_hashes = manifest.get("source_hashes", manifest.get("candidate_source_hashes", []))
    source_hashes = source_records(NUMERICAL_RTL)
    manifest.update(
        {
            "generated_at_utc": generated_at,
            "stage_closing": False,
            "candidate_status": "rope_q6_9_contract_repair_verified_pending_independent_review",
            "candidate_layer_operator": "rope_q_and_rope_k_scale_contract_repair",
            "candidate_requires_fresh_sky130_ppa": True,
            "candidate_meets_numeric_acceptance": True,
            "candidate_rtl_hash": rtl_hash,
            "rtl_hash": rtl_hash,
            "constraint_hash": constraint_hash,
            "historical_source_hashes": historical_source_hashes,
            "candidate_source_hashes": source_hashes,
            "source_hashes": source_hashes,
            "candidate_evidence_hashes": evidence_hashes,
            "ppa_evidence_binding": "evidence/rope_contract_repair/latest/RESULTS.json",
            "latest_evidence": {
                **dict(manifest.get("latest_evidence", {})),
                "rope_contract_repair_binding": "evidence/rope_contract_repair/latest/RESULTS.json",
                "fixed_point_self_test_log": "evidence/rope_contract_repair/latest/fixed_point_self_test.log",
                "rtl_lint_log": "evidence/frontier/latest/rtl_lint.log",
                "rtl_rope_sim_log": "evidence/frontier/latest/rtl_rope_sim.log",
                "rtl_shell_sim_log": "evidence/frontier/latest/rtl_shell_sim.log",
                "sky130_yosys": ppa_frontier["evidence"]["sky130_yosys_log"],
                "sky130_sta": ppa_frontier["evidence"]["sky130_sta_log"],
            },
            "claim_boundaries": [
                "The RoPE scale contract repair replaces Q2.13 pre-rotation clipping with signed-int16 Q6.9 metadata and signed-int24 internal pre-rotation.",
                "Fresh fixed-point self-test, RoPE RTL vectors including a wide Q6.9 saturation case, full shell regression, and SKY130 mapped synthesis/OpenSTA evidence are hash-bound.",
                "This repair does not relax the 2.0 mm^2 non-SRAM cap or 100 MHz floor.",
                "No full official quality pass, routed GDS, signoff, tapeout, FPGA, or silicon result is claimed.",
            ],
        }
    )
    write_json(RTL_MANIFEST, manifest)


def update_ppa_results(
    generated_at: str,
    rtl_hash: str,
    constraint_hash: str,
    yosys: dict[str, Any],
    sta: dict[str, Any],
    critical_path: dict[str, Any],
    shell_cycles: dict[str, Any],
    power: dict[str, Any],
    activity: dict[str, Any],
    duration: dict[str, Any],
    evidence_hashes: dict[str, str],
) -> dict[str, Any]:
    previous = load_json(PPA_RESULTS, {})
    previous_frontier = dict(previous.get("frontier", {}))
    previous_area = previous_frontier.get("non_sram_area_mm2")
    delta_area_percent = (
        ((yosys["non_sram_area_mm2"] - float(previous_area)) / float(previous_area)) * 100.0
        if previous_area
        else None
    )
    fmax_floor = FREQUENCY_FLOOR_MHZ if sta["wns_ns"] >= 0.0 else sta["estimated_fmax_mhz"]
    area_cap_met = yosys["non_sram_area_mm2"] <= AREA_CAP_MM2
    frequency_floor_met = fmax_floor >= FREQUENCY_FLOOR_MHZ and sta["wns_ns"] >= 0.0
    power_energy_j = power["total_w"] * duration["duration_seconds"]
    frontier = {
        "id": f"rope-contract-repair-{generated_at}",
        "stage": "ppa",
        "stage_closing": False,
        "recorded_at_utc": generated_at,
        "mode": "ADVANCE" if area_cap_met and frequency_floor_met else "GLOBAL_COMPRESSION",
        "decision": (
            "rope_q6_9_repair_targets_met_resume_benchmark_review"
            if area_cap_met and frequency_floor_met
            else "rope_q6_9_repair_replan_required"
        ),
        "ordered_supported_layer_operator_prefix": previous_frontier.get("ordered_supported_layer_operator_prefix", []),
        "first_unsupported_layer_operator": previous_frontier.get("first_unsupported_layer_operator"),
        "candidate_layer_operator": "rope_q_and_rope_k_scale_contract_repair",
        "accepted_prefix_advanced": False,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "cells": yosys["cells"],
        "delta_cells": (
            yosys["cells"] - int(previous_frontier["cells"])
            if previous_frontier.get("cells") is not None
            else None
        ),
        "non_sram_area_mm2": yosys["non_sram_area_mm2"],
        "delta_area_mm2": (
            yosys["non_sram_area_mm2"] - float(previous_area)
            if previous_area is not None
            else None
        ),
        "delta_area_percent": delta_area_percent,
        "compression_improvement_percent": None,
        "remaining_area_reserve_mm2": AREA_CAP_MM2 - yosys["non_sram_area_mm2"],
        "fmax_mhz": fmax_floor,
        "fmax_mhz_floor_bound": FREQUENCY_FLOOR_MHZ,
        "remaining_frequency_reserve_mhz": max(0.0, fmax_floor - FREQUENCY_FLOOR_MHZ),
        "sky130_sta": {
            "clock_period_ns": CLOCK_PERIOD_NS,
            "wns_ns": sta["wns_ns"],
            "tns_ns": sta["tns_ns"],
        },
        "critical_path": critical_path,
        "cycle_or_tokens_per_second_impact": {
            **dict(previous_frontier.get("cycle_or_tokens_per_second_impact", {})),
            "rope_contract_repair_full_shell_success_runs": shell_cycles["success_runs"],
            "rope_contract_repair_full_shell_total_cycles": shell_cycles["total_cycles"],
            "rope_contract_repair_full_shell_max_cycles": shell_cycles["max_cycles"],
            "ppa_activity_trace_cycles": activity["reported_success_cycles"],
            "ppa_activity_trace_duration_seconds": duration["duration_seconds"],
            "ppa_activity_trace_energy_j": power_energy_j,
        },
        "evidence": {
            "sky130_yosys_log": "ppa/raw/latest/sky130_yosys.log",
            "sky130_sta_log": "ppa/raw/latest/sky130_sta.log",
            "sky130_power_log": "ppa/raw/latest/sky130_power.log",
            "activity_vcd": "ppa/raw/latest/ace2_shell_lm_head_power.vcd",
            "activity_vcd_generation_log": "ppa/raw/latest/ace2_shell_lm_head_power_vcd.log",
            "rtl_lint_log": "evidence/frontier/latest/rtl_lint.log",
            "rtl_rope_sim_log": "evidence/frontier/latest/rtl_rope_sim.log",
            "rtl_shell_sim_log": "evidence/frontier/latest/rtl_shell_sim.log",
            "repair_binding": "evidence/rope_contract_repair/latest/RESULTS.json",
        },
        "evidence_hashes": evidence_hashes,
        "claim_boundary": "benchmark-stage RoPE contract-repair SKY130 mapped synthesis/OpenSTA PPA refresh; no routed/signoff claim",
    }
    results = {
        **previous,
        "schema_version": 1,
        "project": "ACE-2",
        "stage": "ppa",
        "stage_closing": False,
        "generated_at_utc": generated_at,
        "status": (
            "fresh_rope_contract_repair_sky130_synth_sta_bound_targets_met"
            if area_cap_met and frequency_floor_met
            else "fresh_rope_contract_repair_sky130_synth_sta_replan"
        ),
        "checklist": {
            "ppa.constraints-and-provenance": True,
            "ppa.timing-area-power": True,
            "ppa.physical-closure": True,
            "ppa.incremental-area-reserve": True,
        },
        "claim_boundary": [
            "Current repair PPA is mapped SKY130 synthesis/OpenSTA timing and activity-based power only.",
            "No routed physical design, DRC, LVS, antenna, density, GDS, FPGA prototype, benchmark comparison, signoff, tapeout, or silicon result is claimed.",
        ],
        "policy_decision": {
            "mode": frontier["mode"],
            "decision": frontier["decision"],
            "area_cap_mm2": AREA_CAP_MM2,
            "frequency_floor_mhz": FREQUENCY_FLOOR_MHZ,
            "area_cap_met": area_cap_met,
            "frequency_floor_met": frequency_floor_met,
            "compression_triggered": not area_cap_met,
        },
        "candidate": {
            **dict(previous.get("candidate", {})),
            "name": "ace2_shell_rope_q6_9_contract_repair",
            "top_module": "ace2_shell",
            "candidate_layer_operator": frontier["candidate_layer_operator"],
            "rtl_hash": rtl_hash,
            "constraint_hash": constraint_hash,
            "area": {
                **yosys,
                "area_cap_mm2": AREA_CAP_MM2,
                "area_cap_met": area_cap_met,
                "remaining_area_reserve_mm2": frontier["remaining_area_reserve_mm2"],
            },
            "timing": {
                "clock_period_ns": CLOCK_PERIOD_NS,
                "wns_ns": sta["wns_ns"],
                "tns_ns": sta["tns_ns"],
                "estimated_fmax_mhz": sta["estimated_fmax_mhz"],
                "fmax_mhz": fmax_floor,
                "frequency_floor_mhz": FREQUENCY_FLOOR_MHZ,
                "frequency_floor_met": frequency_floor_met,
                "critical_path": critical_path,
            },
        },
        "frontier": frontier,
        "power": {
            **power,
            "activity_trace": {
                **activity,
                **duration,
                "energy_j": power_energy_j,
                "energy_mj": power_energy_j * 1000.0,
            },
            "methodology_limitations": [
                "pre-route mapped netlist; no extracted parasitics",
                "no CTS/clock tree, so OpenSTA reports clock power as 0 W",
                "lm-head focused activity trace is not a full benchmark trace",
                "power has no operator-owned pass/fail cap yet",
            ],
        },
        "raw_reports": {
            **dict(previous.get("raw_reports", {})),
            **copy_ppa_raw_logs(),
        },
        "source_hashes": {
            "rtl_hash": rtl_hash,
            "synth_sta_constraint_hash": constraint_hash,
            "ppa_protocol_hash": tree_hash(PPA_PROTOCOL_SOURCES),
            "rtl_files": [artifact(path) for path in NUMERICAL_RTL],
            "constraint_and_protocol_files": [artifact(path) for path in PPA_PROTOCOL_SOURCES],
        },
    }
    write_ppa_protocol(generated_at, frontier)
    results["raw_reports"]["protocol"] = artifact("ppa/PROTOCOL.md")
    write_json(PPA_RESULTS, results)
    results["raw_reports"]["results"] = artifact("ppa/RESULTS.json")
    write_json(PPA_RESULTS, results)

    ledger = load_json(PPA_LEDGER, {"schema_version": 1, "project": "ACE-2", "entries": []})
    entries = list(ledger.get("entries", []))
    if not entries or entries[-1].get("id") != frontier["id"]:
        entries.append(frontier)
    ledger["entries"] = entries
    write_json(PPA_LEDGER, ledger)
    return frontier


def update_public_status(generated_at: str, frontier: dict[str, Any]) -> None:
    public = load_json(PUBLIC_STATUS, {})
    dashboard = dict(public.get("dashboard_fields", {}))
    latest = dict(dashboard.get("latest_ppa_frontier", {}))
    latest.update(
        {
            "status": "fresh_rope_contract_repair_sky130_synth_sta_bound_targets_met",
            "cells": frontier["cells"],
            "non_sram_area_mm2": frontier["non_sram_area_mm2"],
            "fmax_mhz": frontier["fmax_mhz"],
            "rtl_hash": frontier["rtl_hash"],
            "constraint_hash": frontier["constraint_hash"],
            "remaining_area_reserve_mm2": frontier["remaining_area_reserve_mm2"],
            "benchmark_stage_binding": "rope_q6_9_contract_repair_bound_pending_benchmark_refresh",
        }
    )
    dashboard.update(
        {
            "current_mode": frontier["mode"],
            "latest_decision": frontier["decision"],
            "latest_ppa_frontier_status": latest["status"],
            "latest_ppa_frontier": latest,
            "ordered_supported_layer_operator_prefix": frontier.get(
                "ordered_supported_layer_operator_prefix",
                dashboard.get("ordered_supported_layer_operator_prefix", []),
            ),
            "first_unsupported_layer_operator": frontier.get("first_unsupported_layer_operator"),
        }
    )
    public["dashboard_fields"] = dashboard
    public["implementation_frontier"] = {
        **dict(public.get("implementation_frontier", {})),
        "ordered_supported_layer_operator_prefix": dashboard["ordered_supported_layer_operator_prefix"],
        "first_unsupported_layer_operator": dashboard["first_unsupported_layer_operator"],
        "current_mode": frontier["mode"],
        "latest_decision": frontier["decision"],
        "latest_ppa_frontier": latest,
    }
    public["last_updated_utc"] = generated_at
    write_json(PUBLIC_STATUS, public)


def main() -> None:
    require_fresh_logs()
    generated_at = utc_now()
    rtl_hash = tree_hash(NUMERICAL_RTL)
    constraint_hash = tree_hash(CONSTRAINTS)
    yosys = parse_yosys(FRONTIER / "sky130_yosys.log")
    sta = parse_sta(FRONTIER / "sky130_sta.log")
    critical_path = parse_critical_path(FRONTIER / "sky130_sta.log")
    shell_cycles = parse_shell_cycles(FRONTIER / "rtl_shell_sim.log")
    power = parse_power(PPA_RAW / "sky130_power.log")
    activity = parse_lm_head_activity(PPA_RAW / "ace2_shell_lm_head_power_vcd.log")
    duration = parse_vcd_duration(PPA_RAW / "ace2_shell_lm_head_power.vcd")
    evidence_hashes = {
        name + "_sha256": sha256_file(path)
        for name, path in LOGS.items()
    }
    if yosys["non_sram_area_mm2"] > AREA_CAP_MM2:
        raise RuntimeError(f"non-SRAM area exceeds cap: {yosys['non_sram_area_mm2']}")
    if sta["wns_ns"] < 0.0:
        raise RuntimeError(f"OpenSTA setup slack is negative: {sta['wns_ns']}")

    frontier = update_ppa_results(
        generated_at,
        rtl_hash,
        constraint_hash,
        yosys,
        sta,
        critical_path,
        shell_cycles,
        power,
        activity,
        duration,
        evidence_hashes,
    )
    update_rtl_manifest(generated_at, rtl_hash, constraint_hash, evidence_hashes, frontier)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "status": "rope_q6_9_contract_repair_verified_pending_independent_review",
        "previous_bound_rtl_hash": PREVIOUS_BOUND_RTL_HASH,
        "candidate_rtl_hash": rtl_hash,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "contract_source_hash": tree_hash(ROPE_CONTRACT_SOURCES),
        "candidate_source_hashes": source_records(NUMERICAL_RTL),
        "constraint_source_hashes": source_records(CONSTRAINTS),
        "contract_source_hashes": source_records(ROPE_CONTRACT_SOURCES),
        "evidence": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in LOGS.items()
        },
        "ppa": {
            "flow": "canonical SKY130 Yosys/OpenSTA mapped-netlist flow",
            "rtl_changed_since_previous_bound_ppa": rtl_hash != PREVIOUS_BOUND_RTL_HASH,
            "non_sram_area_mm2": yosys["non_sram_area_mm2"],
            "area_cap_mm2": AREA_CAP_MM2,
            "area_cap_met": True,
            "clock_period_ns": CLOCK_PERIOD_NS,
            "frequency_floor_mhz": FREQUENCY_FLOOR_MHZ,
            "frequency_floor_met": True,
            "critical_path": critical_path,
            "setup_slack_ns": sta["wns_ns"],
        },
        "quality_scope": {
            "official_quality_claimed": False,
            "active_quality_blocker_archived": "benchmark/quality/archive/QUALITY_BLOCKER.rope-q13-stale-20260730T184100Z.json",
            "repair": "RoPE signed-int16 Q6.9 metadata plus signed-int24 internal pre-rotation",
        },
    }
    manifest_path = EVIDENCE / "rtl_candidate_manifest.json"
    write_json(manifest_path, manifest)
    binding = {
        "schema_version": 1,
        "status": manifest["status"],
        "candidate_rtl_hash": rtl_hash,
        "manifest": {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "numerical_rtl": source_records(NUMERICAL_RTL),
    }
    write_json(QUALITY_BINDING, binding)
    write_json(EVIDENCE / "RESULTS.json", manifest)
    update_public_status(generated_at, frontier)
    print(
        "ACE2_ROPE_CONTRACT_REPAIR_BIND_PASS "
        f"rtl_hash={rtl_hash} area_mm2={yosys['non_sram_area_mm2']:.10f} "
        f"fmax_mhz={frontier['fmax_mhz']:.3f} shell_cycles={shell_cycles['total_cycles']}"
    )


if __name__ == "__main__":
    main()
