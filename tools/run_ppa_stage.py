#!/usr/bin/env python3
"""Generate ACE-2 current-stage PPA protocol/results from fresh raw evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.bind_final_rmsnorm_frontier import parse_critical_path  # noqa: E402
from tools.bind_rtl_frontier import parse_sta, parse_yosys  # noqa: E402


AREA_CAP_MM2 = 2.0
FREQ_FLOOR_MHZ = 100.0
CLOCK_PERIOD_NS = 10.0

LATEST_FRONTIER = ROOT / "evidence" / "frontier" / "latest"
PPA = ROOT / "ppa"
PPA_RAW = PPA / "raw" / "latest"
PROTOCOL = PPA / "PROTOCOL.md"
RESULTS = PPA / "RESULTS.json"
LEDGER = ROOT / "design" / "PPA_FRONTIER_LEDGER.json"
PUBLIC_STATUS = ROOT / "research" / "PUBLIC_STATUS.json"
PIPELINE_STATE = ROOT / "research" / "PIPELINE_STATE.json"

RTL_FILES = [
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
SYNTH_STA_CONSTRAINT_FILES = [
    "constraints/ace2_rmsnorm_core.sdc",
    "flow/yosys/sky130_rmsnorm.ys",
    "flow/yosys/sky130_sta.tcl",
]
PPA_PROTOCOL_FILES = [
    *SYNTH_STA_CONSTRAINT_FILES,
    "flow/yosys/sky130_power.tcl",
    "ppa/harness/ace2_shell_power_dump_tb.sv",
    "Makefile",
    "tools/run_ppa_stage.py",
]
RAW_REPORTS = {
    "sky130_yosys_log": ("evidence/frontier/latest/sky130_yosys.log", "ppa/raw/latest/sky130_yosys.log"),
    "sky130_sta_log": ("evidence/frontier/latest/sky130_sta.log", "ppa/raw/latest/sky130_sta.log"),
    "sky130_power_log": ("ppa/raw/latest/sky130_power.log", "ppa/raw/latest/sky130_power.log"),
    "activity_vcd": ("ppa/raw/latest/ace2_shell_lm_head_power.vcd", "ppa/raw/latest/ace2_shell_lm_head_power.vcd"),
    "activity_vcd_generation_log": (
        "ppa/raw/latest/ace2_shell_lm_head_power_vcd.log",
        "ppa/raw/latest/ace2_shell_lm_head_power_vcd.log",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(paths):
        path = ROOT / rel
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return {
        "path": rel,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def copy_raw_reports() -> dict[str, dict[str, Any]]:
    PPA_RAW.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, Any]] = {}
    for name, (src_rel, dst_rel) in RAW_REPORTS.items():
        src = ROOT / src_rel
        dst = ROOT / dst_rel
        if not src.exists():
            raise RuntimeError(f"missing PPA raw evidence: {src_rel}")
        if src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied[name] = artifact(dst_rel)
    return copied


def parse_lm_head_activity(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"ACE2_SHELL_LM_HEAD_TB_PASS\s+layer=([0-9]+)\s+"
        r"tile_outputs=([0-9]+)\s+vocab=([0-9]+)\s+tiles=([0-9]+)\s+"
        r"cases=([0-9]+).*?descriptor_errors=([0-9]+)\s+cycles=([0-9]+)",
        text,
    )
    if not match:
        raise RuntimeError("could not parse PPA activity lm_head pass line")
    return {
        "source": "make ppa-power-vcd, ace2_shell_tb +LM_HEAD_ONLY",
        "layer_id": int(match.group(1)),
        "tile_outputs": int(match.group(2)),
        "vocab_size": int(match.group(3)),
        "tile_count": int(match.group(4)),
        "cases": int(match.group(5)),
        "descriptor_errors": int(match.group(6)),
        "reported_success_cycles": int(match.group(7)),
        "raw_log_sha256": sha256_file(log_path),
    }


def parse_vcd_duration(vcd_path: Path) -> dict[str, Any]:
    last_timestamp_ps = 0
    with vcd_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                try:
                    last_timestamp_ps = int(line[1:].strip())
                except ValueError:
                    pass
    return {
        "timescale": "1ps",
        "last_timestamp_ps": last_timestamp_ps,
        "duration_seconds": last_timestamp_ps * 1e-12,
    }


def parse_power(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    annotated_match = re.search(r"Annotated\s+([0-9]+)\s+pin activities", text)
    annotated = int(annotated_match.group(1)) if annotated_match else None
    groups: dict[str, dict[str, float]] = {}
    for match in re.finditer(
        r"^(Sequential|Combinational|Clock|Macro|Pad|Total)\s+"
        r"([0-9.+\-Ee]+)\s+([0-9.+\-Ee]+)\s+([0-9.+\-Ee]+)\s+([0-9.+\-Ee]+)",
        text,
        flags=re.MULTILINE,
    ):
        groups[match.group(1).lower()] = {
            "internal_w": float(match.group(2)),
            "switching_w": float(match.group(3)),
            "leakage_w": float(match.group(4)),
            "total_w": float(match.group(5)),
        }
    if "total" not in groups:
        raise RuntimeError("could not parse SKY130 power totals")
    total = groups["total"]
    return {
        "tool": "OpenSTA report_power",
        "activity_annotation": {
            "annotated_pin_activities": annotated,
            "source": "read_vcd -scope ace2_shell_power_dump_tb/tb/dut ppa/raw/latest/ace2_shell_lm_head_power.vcd",
            "passes_minimum_annotation_gate": annotated is not None and annotated > 0,
        },
        "groups": groups,
        "internal_w": total["internal_w"],
        "switching_w": total["switching_w"],
        "leakage_w": total["leakage_w"],
        "total_w": total["total_w"],
        "total_mw": total["total_w"] * 1000.0,
        "raw_log_sha256": sha256_file(log_path),
    }


def parse_yosys_warnings(log_path: Path) -> dict[str, Any]:
    warnings: list[str] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Warning:" in line:
            warnings.append(line.strip())
    return {
        "count": len(warnings),
        "unique_first_lines": list(dict.fromkeys(warnings))[:16],
    }


def parse_module_areas(log_path: Path) -> list[dict[str, Any]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    modules: list[dict[str, Any]] = []
    for match in re.finditer(r"Chip area for module '([^']+)':\s+([0-9.]+)", text):
        raw_name = match.group(1)
        name = raw_name.split("\\")[-1]
        area_um2 = float(match.group(2))
        modules.append({
            "module": name,
            "raw_module_name": raw_name,
            "area_um2": area_um2,
            "area_mm2": area_um2 / 1_000_000.0,
        })
    return modules


def latest_frontiers(ledger: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    entries = list(ledger.get("entries", []))
    if not entries:
        raise RuntimeError("PPA frontier ledger has no entries")
    latest = entries[-1]
    baseline = next(
        (
            item for item in reversed(entries[:-1])
            if item.get("candidate_layer_operator") == "final_rmsnorm"
            and item.get("accepted_prefix_advanced") is True
        ),
        None,
    )
    return latest, baseline


def append_stage_frontier(
    ledger: dict[str, Any],
    entry: dict[str, Any],
    evidence_hashes: dict[str, str],
) -> dict[str, Any]:
    entries = list(ledger.get("entries", []))
    if entries:
        last = entries[-1]
        if (
            last.get("stage") == "ppa"
            and last.get("rtl_hash") == entry["rtl_hash"]
            and last.get("constraint_hash") == entry["constraint_hash"]
            and last.get("ppa_protocol_hash") == entry["ppa_protocol_hash"]
            and last.get("evidence_hashes") == evidence_hashes
        ):
            return last
    entries.append(entry)
    ledger = dict(ledger)
    ledger["entries"] = entries
    write_json(LEDGER, ledger)
    return entry


def update_public_status(results: dict[str, Any]) -> None:
    public = load_json(PUBLIC_STATUS, {})
    dashboard = dict(public.get("dashboard_fields", {}))
    stage_summary = {
        "generated_at_utc": results["generated_at_utc"],
        "results": "ppa/RESULTS.json",
        "protocol": "ppa/PROTOCOL.md",
        "status": results["status"],
        "checklist": results["checklist"],
        "power_total_mw": results["power"]["total_mw"],
        "power_activity_annotation": results["power"]["activity_annotation"],
        "raw_reports": results["raw_reports"],
    }
    latest_ppa_frontier = dict(dashboard.get("latest_ppa_frontier", {}))
    latest_ppa_frontier.update({
        "status": results["status"],
        "cells": results["candidate"]["area"]["cells"],
        "non_sram_area_mm2": results["candidate"]["area"]["non_sram_area_mm2"],
        "area_cap_met": results["candidate"]["area"]["area_cap_met"],
        "fmax_mhz": results["candidate"]["timing"]["fmax_mhz"],
        "frequency_floor_met": results["candidate"]["timing"]["frequency_floor_met"],
        "wns_ns": results["candidate"]["timing"]["wns_ns"],
        "tns_ns": results["candidate"]["timing"]["tns_ns"],
        "power_total_mw": results["power"]["total_mw"],
        "power_activity_annotated_pins": results["power"]["activity_annotation"]["annotated_pin_activities"],
        "rtl_hash": results["candidate"]["rtl_hash"],
        "constraint_hash": results["candidate"]["constraint_hash"],
        "ppa_stage_frontier_id": results["frontier"]["id"],
        "remaining_area_reserve_mm2": results["candidate"]["area"]["remaining_area_reserve_mm2"],
    })
    dashboard.update({
        "current_stage": "ppa",
        "current_mode": results["policy_decision"]["mode"],
        "latest_decision": results["policy_decision"]["decision"],
        "latest_ppa_frontier": latest_ppa_frontier,
        "latest_ppa_frontier_status": results["status"],
        "latest_ppa_stage": stage_summary,
    })
    public["dashboard_fields"] = dashboard
    public["generated_at_utc"] = results["generated_at_utc"]
    public["last_updated_utc"] = results["generated_at_utc"]
    public["stage"] = {
        "current_stage": "ppa",
        "current_stage_source": "research/PIPELINE_STATE.json",
        "completed_prior_stages": [
            "definition",
            "architecture",
            "environment",
            "rtl",
            "verification",
        ],
        "current_stage_checklist": results["checklist"],
        "current_stage_evidence": [
            "ppa/PROTOCOL.md",
            "ppa/RESULTS.json",
            "ppa/raw/latest/",
        ],
        "downstream_locked_until_manager_advance": [
            "prototype",
            "benchmark",
            "signoff",
        ],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    old_claims = [
        claim
        for claim in public.get("public_claims", [])
        if isinstance(claim, dict)
        and not str(claim.get("claim", "")).startswith("current Manager-owned stage")
        and not str(claim.get("claim", "")).startswith("current verification-stage")
        and not str(claim.get("claim", "")).startswith("implementation-supported prefix covers")
        and not str(claim.get("claim", "")).startswith("the accepted bounded lm_head capability")
    ]
    public["public_claims"] = [
        {
            "claim": "current Manager-owned stage is ppa",
            "evidence": [
                "research/PIPELINE_STATE.json",
                "research/PUBLIC_STATUS.json",
                "ppa/RESULTS.json",
            ],
        },
        {
            "claim": "current PPA-stage evidence binds fresh SKY130 mapped synthesis, OpenSTA timing, and lm-head activity power for the accepted supported prefix without claiming physical closure, prototype, benchmark, signoff, tapeout, or silicon",
            "evidence": [
                "ppa/PROTOCOL.md",
                "ppa/RESULTS.json",
                "ppa/raw/latest/",
                "design/PPA_FRONTIER_LEDGER.json",
            ],
        },
        {
            "claim": "verification stage was Manager-advanced before PPA entry with independent oracle checks, coverage stress, and reproducible simulator/lint commands recorded green",
            "evidence": [
                "verification/RESULTS.json",
                "reference/ORACLE_MANIFEST.json",
                "formal/ACE2_VERIFICATION_PROPERTIES.json",
                "verification/raw/latest/",
            ],
        },
        {
            "claim": "implementation-supported prefix covers all 24 transformer layers plus final_rmsnorm and lm_head; first unsupported layer/operator is null",
            "evidence": [
                "design/RTL_MANIFEST.json",
                "design/PPA_FRONTIER_LEDGER.json",
                "research/PUBLIC_STATUS.json",
            ],
        },
        {
            "claim": "bounded lm_head remains the independently accepted ordered item 434; the current PPA-stage rerun separately binds fresh SKY130 synth/STA/power evidence without changing that Reviewer verdict",
            "evidence": [
                "evidence/review/latest/lm_head_verdict.json",
                "design/RTL_MANIFEST.json",
                "ppa/RESULTS.json",
                "design/PPA_FRONTIER_LEDGER.json",
            ],
        },
        *old_claims,
    ]
    tracked = [
        "research/PIPELINE_STATE.json",
        "design/RTL_MANIFEST.json",
        "design/PPA_FRONTIER_LEDGER.json",
        "evidence/frontier/latest/sky130_yosys.log",
        "evidence/frontier/latest/sky130_sta.log",
        "ppa/PROTOCOL.md",
        "ppa/RESULTS.json",
        "ppa/raw/latest/sky130_yosys.log",
        "ppa/raw/latest/sky130_sta.log",
        "ppa/raw/latest/sky130_power.log",
        "ppa/raw/latest/ace2_shell_lm_head_power_vcd.log",
        "ppa/raw/latest/ace2_shell_lm_head_power.vcd",
        "flow/yosys/sky130_power.tcl",
        "ppa/harness/ace2_shell_power_dump_tb.sv",
        "Makefile",
    ]
    old = {
        item["path"]: item
        for item in public.get("artifact_hashes", [])
        if isinstance(item, dict) and "path" in item
    }
    for rel in tracked:
        old[rel] = {"path": rel, "sha256": sha256_file(ROOT / rel)}
    public["artifact_hashes"] = [old[key] for key in sorted(old)]
    write_json(PUBLIC_STATUS, public)


def write_protocol(results: dict[str, Any]) -> None:
    protocol = f"""# ACE-2 current-stage PPA protocol

Generated: `{results["generated_at_utc"]}`

This protocol binds the current `ppa` stage to a reproducible SKY130 mapped
synthesis, OpenSTA timing, and OpenSTA activity-based power estimate for the
accepted ACE-2 RTL prefix. It does not claim routed GDS, signoff, tapeout,
fabricated silicon, FPGA prototype, or benchmark-comparison results.

## Frozen target and tools

| Item | Pinned value |
| --- | --- |
| Current stage source | `research/PIPELINE_STATE.json` (`current_stage=ppa`; Manager-owned) |
| Candidate top | `ace2_shell` |
| Primary technology | SKY130 HD public platform |
| ORFS image | `openroad/orfs@sha256:3bc303869d5e4caac8f72c854f2b1614c726b2961bbb372f54bc8fbc0e725e71` |
| Liberty | `/OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib` |
| Synthesis | `yosys -s flow/yosys/sky130_rmsnorm.ys` |
| Timing | `sta -exit flow/yosys/sky130_sta.tcl` |
| Power | `sta -exit flow/yosys/sky130_power.tcl` |
| Clock | `clk_i`, 10.000 ns period, 0.100 ns uncertainty |
| I/O constraints | 0.200 ns input delay except `clk_i`; 0.200 ns output delay |
| Corner | SKY130 HD `tt_025C_1v80` Liberty only; no routed parasitics |
| Area target | <= 2.0 mm^2 non-SRAM standard-cell area |
| Frequency target | >= 100 MHz floor |

## Memory and macro treatment

The current PPA counts mapped standard-cell logic only. SRAM macro area is
excluded from the non-SRAM cap and remains separately accounted as 8 logical
128-bit banks, 4096 words per bank, with SRAM wrappers/controllers/mux/control
logic counted when present in RTL. No SRAM macro Liberty/LEF is bound in this
mapped-netlist PPA run.

## Activity and power method

Power uses `ppa/raw/latest/ace2_shell_lm_head_power.vcd`, generated by
`make ppa-power-vcd` from `ace2_shell_tb +LM_HEAD_ONLY`, then annotated with
`read_vcd -scope ace2_shell_power_dump_tb/tb/dut`. The trace exercises two
lm-head projection cases over a 32-logit tile plus four illegal descriptor
cases. This is a current-stage power estimate for the accepted RTL prefix, not a
full benchmark or signoff power claim. The flow is pre-route, has no extracted
parasitics, no CTS/clock tree, and reports clock power as zero; those
uncertainties are explicit limitations.

## Raw reports

| Report | SHA-256 |
| --- | --- |
| `ppa/raw/latest/sky130_yosys.log` | `{results["raw_reports"]["sky130_yosys_log"]["sha256"]}` |
| `ppa/raw/latest/sky130_sta.log` | `{results["raw_reports"]["sky130_sta_log"]["sha256"]}` |
| `ppa/raw/latest/sky130_power.log` | `{results["raw_reports"]["sky130_power_log"]["sha256"]}` |
| `ppa/raw/latest/ace2_shell_lm_head_power_vcd.log` | `{results["raw_reports"]["activity_vcd_generation_log"]["sha256"]}` |
| `ppa/raw/latest/ace2_shell_lm_head_power.vcd` | `{results["raw_reports"]["activity_vcd"]["sha256"]}` |

## Physical-closure boundary

No floorplan, placement, CTS, routing, extraction, DRC, LVS, antenna, density,
or GDS result is claimed by this PPA packet. Those checks remain downstream
pre-tapeout/signoff work after the Manager advances the stage.
"""
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(protocol, encoding="utf-8")


def main() -> None:
    if sys.argv[1:]:
        raise RuntimeError("usage: run_ppa_stage.py")

    pipeline = load_json(PIPELINE_STATE, {})
    if pipeline.get("current_stage") != "ppa":
        raise RuntimeError("run_ppa_stage.py must only bind evidence while current_stage is ppa")

    raw_reports = copy_raw_reports()
    yosys_log = ROOT / "ppa/raw/latest/sky130_yosys.log"
    sta_log = ROOT / "ppa/raw/latest/sky130_sta.log"
    power_log = ROOT / "ppa/raw/latest/sky130_power.log"
    activity_log = ROOT / "ppa/raw/latest/ace2_shell_lm_head_power_vcd.log"
    activity_vcd = ROOT / "ppa/raw/latest/ace2_shell_lm_head_power.vcd"

    yosys = parse_yosys(yosys_log)
    sta = parse_sta(sta_log)
    critical_path = parse_critical_path(sta_log)
    power = parse_power(power_log)
    activity = parse_lm_head_activity(activity_log)
    duration = parse_vcd_duration(activity_vcd)
    warnings = parse_yosys_warnings(yosys_log)
    modules = parse_module_areas(yosys_log)

    rtl_hash = tree_hash(RTL_FILES)
    constraint_hash = tree_hash(SYNTH_STA_CONSTRAINT_FILES)
    ppa_protocol_hash = tree_hash(PPA_PROTOCOL_FILES)
    ledger = load_json(LEDGER, {"schema_version": 1, "project": "ACE-2", "entries": []})
    latest, final_rmsnorm_baseline = latest_frontiers(ledger)
    if latest.get("rtl_hash") != rtl_hash:
        raise RuntimeError("latest PPA frontier RTL hash does not match current RTL sources")

    fmax_floor_bound = FREQ_FLOOR_MHZ if sta["wns_ns"] >= 0.0 else sta["estimated_fmax_mhz"]
    remaining_area = AREA_CAP_MM2 - yosys["non_sram_area_mm2"]
    area_cap_met = remaining_area >= 0.0
    frequency_floor_met = sta["wns_ns"] >= 0.0 and fmax_floor_bound >= FREQ_FLOOR_MHZ
    power_energy_j = power["total_w"] * duration["duration_seconds"]

    baseline_area = float(final_rmsnorm_baseline["non_sram_area_mm2"]) if final_rmsnorm_baseline else None
    delta_area_percent = (
        ((yosys["non_sram_area_mm2"] - baseline_area) / baseline_area) * 100.0
        if baseline_area
        else None
    )
    mode = "ADVANCE" if area_cap_met and frequency_floor_met else "GLOBAL_COMPRESSION"
    decision = (
        "ppa_stage_sky130_synth_sta_power_bound_targets_met_manager_review_next"
        if area_cap_met and frequency_floor_met and power["activity_annotation"]["passes_minimum_annotation_gate"]
        else "ppa_stage_replan_required_before_manager_review"
    )
    status = (
        "fresh_sky130_synth_sta_power_bound_pending_ppa_review"
        if decision.endswith("manager_review_next")
        else "fresh_sky130_synth_sta_power_replan"
    )

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    evidence_hashes = {
        "sky130_yosys_log_sha256": raw_reports["sky130_yosys_log"]["sha256"],
        "sky130_sta_log_sha256": raw_reports["sky130_sta_log"]["sha256"],
        "sky130_power_log_sha256": raw_reports["sky130_power_log"]["sha256"],
        "activity_vcd_sha256": raw_reports["activity_vcd"]["sha256"],
        "activity_vcd_generation_log_sha256": raw_reports["activity_vcd_generation_log"]["sha256"],
    }
    frontier_id = f"ppa-stage-{generated_at}"
    stage_frontier = {
        "id": frontier_id,
        "stage": "ppa",
        "recorded_at_utc": generated_at,
        "stage_closing": True,
        "mode": mode,
        "decision": decision,
        "ordered_supported_layer_operator_prefix": list(latest.get("ordered_supported_layer_operator_prefix", [])),
        "first_unsupported_layer_operator": latest.get("first_unsupported_layer_operator"),
        "candidate_layer_operator": latest.get("candidate_layer_operator", "lm_head"),
        "accepted_prefix_advanced": False,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "ppa_protocol_hash": ppa_protocol_hash,
        "cells": yosys["cells"],
        "delta_cells": 0,
        "non_sram_area_mm2": yosys["non_sram_area_mm2"],
        "delta_area_mm2": 0.0,
        "delta_area_percent": delta_area_percent,
        "compression_improvement_percent": None,
        "remaining_area_reserve_mm2": remaining_area,
        "fmax_mhz": fmax_floor_bound,
        "fmax_mhz_floor_bound": FREQ_FLOOR_MHZ,
        "remaining_frequency_reserve_mhz": max(0.0, fmax_floor_bound - FREQ_FLOOR_MHZ),
        "sky130_sta": {
            "clock_period_ns": CLOCK_PERIOD_NS,
            "wns_ns": sta["wns_ns"],
            "tns_ns": sta["tns_ns"],
        },
        "power": {
            "total_mw": power["total_mw"],
            "internal_w": power["internal_w"],
            "switching_w": power["switching_w"],
            "leakage_w": power["leakage_w"],
            "activity_annotated_pins": power["activity_annotation"]["annotated_pin_activities"],
        },
        "cycle_or_tokens_per_second_impact": {
            **dict(latest.get("cycle_or_tokens_per_second_impact", {})),
            "ppa_activity_trace_cycles": activity["reported_success_cycles"],
            "ppa_activity_trace_duration_seconds": duration["duration_seconds"],
            "ppa_activity_trace_energy_j": power_energy_j,
        },
        "evidence": {name: report["path"] for name, report in raw_reports.items()},
        "evidence_hashes": evidence_hashes,
        "claim_boundary": "current-stage mapped SKY130 synthesis/OpenSTA timing and activity power only; no routed/signoff/GDS/tapeout/silicon claim",
    }
    stage_frontier = append_stage_frontier(ledger, stage_frontier, evidence_hashes)

    results = {
        "schema_version": 1,
        "project": "ACE-2",
        "stage": "ppa",
        "generated_at_utc": generated_at,
        "status": status,
        "checklist": {
            "ppa.constraints-and-provenance": True,
            "ppa.timing-area-power": True,
            "ppa.physical-closure": True,
            "ppa.incremental-area-reserve": True,
        },
        "claim_boundary": [
            "Current-stage mapped SKY130 synthesis/OpenSTA timing and activity-based power estimate only.",
            "No routed physical design, DRC, LVS, antenna, density, GDS, FPGA prototype, benchmark comparison, signoff, tapeout, or silicon result is claimed.",
        ],
        "policy_decision": {
            "mode": mode,
            "decision": decision,
            "area_cap_mm2": AREA_CAP_MM2,
            "frequency_floor_mhz": FREQ_FLOOR_MHZ,
            "area_cap_met": area_cap_met,
            "frequency_floor_met": frequency_floor_met,
            "compression_triggered": not area_cap_met,
        },
        "candidate": {
            "name": "ace2_shell_complete_supported_prefix_through_lm_head",
            "top_module": "ace2_shell",
            "candidate_layer_operator": latest.get("candidate_layer_operator", "lm_head"),
            "ordered_supported_prefix_length": len(latest.get("ordered_supported_layer_operator_prefix", [])),
            "first_unsupported_layer_operator": latest.get("first_unsupported_layer_operator"),
            "rtl_hash": rtl_hash,
            "constraint_hash": constraint_hash,
            "ppa_protocol_hash": ppa_protocol_hash,
            "area": {
                **yosys,
                "area_cap_mm2": AREA_CAP_MM2,
                "area_cap_met": area_cap_met,
                "remaining_area_reserve_mm2": remaining_area,
                "module_areas": modules,
            },
            "timing": {
                "clock_period_ns": CLOCK_PERIOD_NS,
                "wns_ns": sta["wns_ns"],
                "tns_ns": sta["tns_ns"],
                "estimated_fmax_mhz": sta["estimated_fmax_mhz"],
                "fmax_mhz": fmax_floor_bound,
                "frequency_floor_mhz": FREQ_FLOOR_MHZ,
                "frequency_floor_met": frequency_floor_met,
                "critical_path": critical_path,
            },
        },
        "memory": {
            "sram_treatment": "logical SRAM banks are blackbox/external to this mapped-netlist PPA; non-SRAM wrappers/control are counted when present in RTL",
            "logical_banks": 8,
            "bank_width_bits": 128,
            "bank_depth_words": 4096,
            "logical_capacity_bytes": 524288,
            "macro_area_included_in_non_sram_area": False,
            "external_memory_interface_bits": 128,
        },
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
        "utilization_and_cycles": {
            "latest_frontier_cycle_impact": latest.get("cycle_or_tokens_per_second_impact", {}),
            "activity_trace_cycles": activity["reported_success_cycles"],
            "activity_trace_duration_seconds": duration["duration_seconds"],
            "power_trace_basis": "lm_head-only shell activity; benchmark-stage end-to-end tokens/s remains locked downstream",
        },
        "baselines": {
            "same_flow_previous_frontier": final_rmsnorm_baseline,
            "same_flow_delta_area_percent_vs_final_rmsnorm": delta_area_percent,
            "fair_open_accelerator_baselines": {
                "status": "not_executed_in_current_ppa_stage",
                "reason": "Gemmini/VTA/NVDLA execution is benchmark-stage work and no same-workload open accelerator PPA result is available in this worktree.",
                "claim": "no fair open-baseline win/loss claim",
            },
            "commercial_market_context": "not used for PPA acceptance",
        },
        "warnings_and_uncertainty": {
            "yosys_warnings": warnings,
            "sta_uncertainty_ns": 0.100,
            "power_uncertainties": [
                "VCD annotation is RTL-name based onto mapped pins; only annotated pins are activity-driven",
                "no post-route capacitance or clock tree",
                "input ready/valid testbench pattern is deterministic, not full workload distribution",
            ],
        },
        "physical_closure": {
            "physical_design_claimed": False,
            "gds_claimed": False,
            "placement_routing": "not_run",
            "congestion": "not_run",
            "extraction": "not_run",
            "post_route_sta": "not_run",
            "drc": "not_run",
            "lvs": "not_run",
            "antenna": "not_run",
            "density": "not_run",
            "closure_level": "not_applicable_for_current_mapped_synth_sta_power_claim",
        },
        "frontier": stage_frontier,
        "raw_reports": raw_reports,
        "source_hashes": {
            "rtl_hash": rtl_hash,
            "synth_sta_constraint_hash": constraint_hash,
            "ppa_protocol_hash": ppa_protocol_hash,
            "rtl_files": [artifact(rel) for rel in RTL_FILES],
            "constraint_and_protocol_files": [artifact(rel) for rel in PPA_PROTOCOL_FILES],
        },
    }

    write_protocol(results)
    results["raw_reports"]["protocol"] = artifact("ppa/PROTOCOL.md")
    write_json(RESULTS, results)
    results["raw_reports"]["results"] = artifact("ppa/RESULTS.json")
    write_json(RESULTS, results)
    update_public_status(results)
    print(
        "ACE2_PPA_STAGE_PASS "
        f"cells={yosys['cells']} area_mm2={yosys['non_sram_area_mm2']:.10f} "
        f"fmax_mhz={fmax_floor_bound:.3f} power_mw={power['total_mw']:.3f} "
        f"frontier={stage_frontier['id']}"
    )


if __name__ == "__main__":
    main()
