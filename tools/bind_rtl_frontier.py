#!/usr/bin/env python3
"""Bind the current ACE-2 RTL fast-loop evidence into project artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "evidence" / "frontier" / "latest"
MANIFEST = ROOT / "design" / "RTL_MANIFEST.json"
LEDGER = ROOT / "design" / "PPA_FRONTIER_LEDGER.json"
PUBLIC_STATUS = ROOT / "research" / "PUBLIC_STATUS.json"
PIPELINE_STATE = ROOT / "research" / "PIPELINE_STATE.json"
REVIEW_VERDICT = ROOT / "evidence" / "review" / "latest" / "mlp_residual_add_verdict.json"

AREA_CAP_MM2 = 2.0
FREQ_FLOOR_MHZ = 100.0
HIDDEN_SIZE = 896
MLP_INTERMEDIATE_SIZE = 4864
PROJ_MAC_LANES = 4
PROJ_GROUPS = MLP_INTERMEDIATE_SIZE // PROJ_MAC_LANES
PROJ_GROUPS_PER_STORAGE_BEAT = 16 // PROJ_MAC_LANES
PROJ_WEIGHT_STORAGE_BEATS_PER_OUTPUT = MLP_INTERMEDIATE_SIZE // 16
PROJ_LOGICAL_WEIGHT_BYTES_PER_OUTPUT = MLP_INTERMEDIATE_SIZE // 2
PROJ_PADDED_WEIGHT_BYTES_PER_OUTPUT = PROJ_WEIGHT_STORAGE_BEATS_PER_OUTPUT * 16

SYNTHESIZED_MODULES = [
    "ace2_shell",
    "ace2_state_shadow",
    "ace2_rmsnorm_core",
    "ace2_w4a8_proj_core",
    "ace2_rope_core",
    "ace2_softmax_core",
    "ace2_silu_gate_core",
]

PREVIOUS_PREFIX = ["layer_0.input_rmsnorm", "layer_0.q_proj", "layer_0.k_proj", "layer_0.v_proj", "layer_0.rope_q", "layer_0.rope_k", "layer_0.kv_write", "layer_0.attention_score", "layer_0.softmax", "layer_0.attention_value", "layer_0.o_proj", "layer_0.attention_residual_add", "layer_0.post_attention_rmsnorm", "layer_0.mlp_gate_proj", "layer_0.mlp_up_proj", "layer_0.silu_gate", "layer_0.mlp_down_proj"]
SUPPORTED_PREFIX = PREVIOUS_PREFIX + ["layer_0.mlp_residual_add"]
PREVIOUS_FIRST_UNSUPPORTED = "layer_0.mlp_residual_add"
FIRST_UNSUPPORTED = "layer_1.input_rmsnorm"
FIRST_UNSUPPORTED_LABEL = "layer_1_input_rmsnorm"
NEW_OPERATOR = "mlp_residual_add"
CANDIDATE_OPERATOR = "layer_0.mlp_residual_add"
KV_WRITE_BYTES_PER_TOKEN = 272
ROPE_LANES = 2
ATTN_SCORE_HEAD_DIM = 64
ATTN_SCORE_CONTEXT_MAX = 8
ATTN_SCORE_MAC_LANES = 1
SOFTMAX_CONTEXT_MAX = 8
SOFTMAX_EXP_STEP_Q6_9 = 64
ATTN_VALUE_HEAD_DIM = 64
ATTN_VALUE_CONTEXT_MAX = 8
ATTN_VALUE_MAC_LANES = 1
COMPRESSION_MECHANISM = "mlp_residual_add_global_folding"
TIMING_REPAIR_MECHANISM = "mlp_residual_add_structural_timing_repair"

SOURCE_FILES = [
    "rtl/ace2_pkg.sv",
    "rtl/ace2_rmsnorm_core.sv",
    "rtl/ace2_w4a8_proj_core.sv",
    "rtl/ace2_rope_core.sv",
    "rtl/ace2_attention_score_core.sv",
    "rtl/ace2_softmax_core.sv",
    "rtl/ace2_silu_gate_core.sv",
    "rtl/generated/ace2_silu_lut.svh",
    "rtl/ace2_shell.sv",
    "verification/tb/ace2_rmsnorm_tb.sv",
    "verification/tb/ace2_w4a8_proj_tb.sv",
    "verification/tb/ace2_rope_tb.sv",
    "verification/tb/ace2_attention_score_tb.sv",
    "verification/tb/ace2_softmax_tb.sv",
    "verification/tb/ace2_silu_gate_tb.sv",
    "verification/tb/ace2_shell_tb.sv",
    "verification/generated/rmsnorm_vectors.svh",
    "verification/generated/rmsnorm_vectors.json",
    "verification/generated/projection_vectors.svh",
    "verification/generated/projection_vectors.json",
    "verification/generated/rope_vectors.svh",
    "verification/generated/rope_vectors.json",
    "verification/generated/attention_score_vectors.svh",
    "verification/generated/attention_score_vectors.json",
    "verification/generated/softmax_vectors.svh",
    "verification/generated/softmax_vectors.json",
    "verification/generated/silu_gate_vectors.svh",
    "verification/generated/silu_gate_vectors.json",
    "verification/generated/attention_value_vectors.svh",
    "verification/generated/attention_value_vectors.json",
    "verification/generated/residual_vectors.svh",
    "verification/generated/residual_vectors.json",
    "verification/generated/mlp_residual_vectors.svh",
    "verification/generated/mlp_residual_vectors.json",
    "verification/generated/post_attention_rmsnorm_vectors.svh",
    "verification/generated/post_attention_rmsnorm_vectors.json",
    "tools/ace2_rmsnorm_reference.py",
    "tools/ace2_projection_reference.py",
    "tools/ace2_rope_reference.py",
    "tools/ace2_attention_score_reference.py",
    "tools/ace2_softmax_reference.py",
    "tools/ace2_silu_gate_reference.py",
    "tools/ace2_attention_value_reference.py",
    "tools/ace2_residual_reference.py",
    "tools/ace2_mlp_residual_reference.py",
    "tools/gen_rmsnorm_vectors.py",
    "tools/gen_projection_vectors.py",
    "tools/gen_rope_vectors.py",
    "tools/gen_attention_score_vectors.py",
    "tools/gen_softmax_vectors.py",
    "tools/gen_silu_gate_vectors.py",
    "tools/gen_attention_value_vectors.py",
    "tools/gen_residual_vectors.py",
    "tools/gen_mlp_residual_vectors.py",
    "tools/gen_post_attention_rmsnorm_vectors.py",
    "tools/bind_rtl_frontier.py",
    "constraints/ace2_rmsnorm_core.sdc",
    "flow/yosys/sky130_rmsnorm.ys",
    "flow/yosys/sky130_sta.tcl",
    "Makefile",
    "design/RTL_TRACEABILITY.md",
    "design/SPEC.md",
    "design/ARCHITECTURE.md",
]

ARTIFACT_FILES = [
    "MISSION.md",
    "research/PIPELINE_STATE.json",
    "design/RTL_MANIFEST.json",
    "design/PPA_FRONTIER_LEDGER.json",
    "evidence/frontier/latest/rtl_core_sim.log",
    "evidence/frontier/latest/rtl_proj_sim.log",
    "evidence/frontier/latest/rtl_rope_sim.log",
    "evidence/frontier/latest/rtl_attention_score_sim.log",
    "evidence/frontier/latest/rtl_softmax_sim.log",
    "evidence/frontier/latest/rtl_silu_gate_core_sim.log",
    "evidence/frontier/latest/rtl_silu_gate_sim.log",
    "evidence/frontier/latest/rtl_shell_sim.log",
    "evidence/frontier/latest/rtl_qproj_stride_sim.log",
    "evidence/frontier/latest/rtl_mlp_up_sim.log",
    "evidence/frontier/latest/rtl_mlp_down_sim.log",
    "evidence/frontier/latest/rtl_mlp_residual_sim.log",
    "evidence/frontier/latest/sky130_yosys.log",
    "evidence/frontier/latest/sky130_sta.log",
    "evidence/review/latest/mlp_residual_add_verdict.json",
]


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
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_yosys(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    local_cell_count = None
    local_area = None
    total_cell_count = None
    total_area = None
    sections = [match.end() for match in re.finditer(r"^=== ace2_shell ===\s*$", text, flags=re.MULTILINE)]
    if sections:
        start = sections[-1]
        end_match = re.search(r"^(?:=== |\s+Chip area)", text[start:], flags=re.MULTILINE)
        section = text[start:start + end_match.start()] if end_match else text[start:]
        local_cell_count = sum(
            int(match.group(1))
            for match in re.finditer(r"^\s*([0-9]+)\s+[0-9.+Ee-]+\s+sky130_fd_sc_hd__", section, flags=re.MULTILINE)
        )
    if local_cell_count is None or local_cell_count == 0:
        for match in re.finditer(r"Number of cells:\s+([0-9]+)", text):
            local_cell_count = int(match.group(1))
    for match in re.finditer(r"Chip area for module '\\ace2_shell':\s+([0-9.]+)", text):
        local_area = float(match.group(1))
    for match in re.finditer(r"^\s*([0-9]+)\s+[0-9.+Ee-]+\s+ace2_shell\s*$", text, flags=re.MULTILINE):
        total_cell_count = int(match.group(1))
    for match in re.finditer(r"Chip area for top module.*?:\s+([0-9.]+)", text):
        total_area = float(match.group(1))
    if local_cell_count is None or local_area is None:
        raise RuntimeError("could not parse Yosys cell count/area")
    area = total_area if total_area is not None else local_area
    cell_count = total_cell_count if total_cell_count is not None else local_cell_count
    return {
        "cells": cell_count,
        "non_sram_area_um2": area,
        "non_sram_area_mm2": area / 1_000_000.0,
        "area_accounting_basis": "hierarchical_total_including_rtl_submodules" if total_area is not None else "module_local_excluding_submodules",
        "local_cells_excluding_submodules": local_cell_count,
        "local_non_sram_area_um2": local_area,
        "local_non_sram_area_mm2": local_area / 1_000_000.0,
        "log_sha256": sha256_file(log_path),
    }


def parse_sta(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    wns = None
    tns = None
    for match in re.finditer(r"wns\s+(?:max\s+)?([-+0-9.]+)", text, flags=re.IGNORECASE):
        wns = float(match.group(1))
    for match in re.finditer(r"tns\s+(?:max\s+)?([-+0-9.]+)", text, flags=re.IGNORECASE):
        tns = float(match.group(1))
    if wns is None:
        slack_matches = re.findall(r"slack\s+\(?\w*\)?\s+([-+0-9.]+)", text)
        if slack_matches:
            wns = min(float(value) for value in slack_matches)
    if wns is None:
        raise RuntimeError("could not parse STA WNS")
    period_ns = 10.0
    critical_path_ns = period_ns - wns
    fmax_mhz = 1000.0 / critical_path_ns if critical_path_ns > 0 else math.inf
    return {
        "clock_period_ns": period_ns,
        "wns_ns": wns,
        "tns_ns": tns,
        "estimated_fmax_mhz": fmax_mhz,
        "estimated_fmax_mhz_floor_bound": FREQ_FLOOR_MHZ if wns >= 0 else fmax_mhz,
        "log_sha256": sha256_file(log_path),
    }


def parse_shell_sim(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"ACE2_SHELL_CYCLES\s+success_runs=([0-9]+)\s+max_cycles=([0-9]+)\s+total_cycles=([0-9]+)\s+last_cycles=([0-9]+)",
        text,
    )
    if not match:
        raise RuntimeError("could not parse shell simulation cycle summary")
    pass_match = re.search(
        r"ACE2_SHELL_TB_PASS\s+layers=([0-9]+)\s+repeated_layer=([0-9]+)\s+generated_cases=([0-9]+)\s+qproj_cases=([0-9]+)(?:\s+kproj_cases=([0-9]+))?(?:\s+vproj_cases=([0-9]+))?(?:\s+oproj_cases=([0-9]+))?(?:\s+mlp_gate_proj_cases=([0-9]+))?(?:\s+mlp_up_proj_cases=([0-9]+))?(?:\s+mlp_down_proj_cases=([0-9]+))?(?:\s+mlp_residual_cases=([0-9]+))?(?:\s+silu_gate_cases=([0-9]+))?(?:\s+rope_q_cases=([0-9]+))?(?:\s+rope_k_cases=([0-9]+))?(?:\s+kv_write_cases=([0-9]+))?(?:\s+attn_score_cases=([0-9]+))?(?:\s+softmax_cases=([0-9]+))?(?:\s+attn_value_cases=([0-9]+))?(?:\s+residual_cases=([0-9]+))?(?:\s+post_rms_cases=([0-9]+))?\s+descriptor_errors=([0-9]+)\s+memory_errors=([0-9]+)\s+watchdog_errors=([0-9]+)\s+reset_busy=([0-9]+)(?:\s+response_protocol_cases=([0-9]+))?",
        text,
    )
    if not pass_match:
        raise RuntimeError("could not parse shell simulation pass summary")
    success_runs = int(match.group(1))
    total_cycles = int(match.group(3))
    return {
        "success_runs": success_runs,
        "max_cycles_per_success_with_test_stalls": int(match.group(2)),
        "total_success_cycles": total_cycles,
        "last_cycles": int(match.group(4)),
        "average_cycles_per_success": total_cycles / success_runs if success_runs else None,
        "layers": int(pass_match.group(1)),
        "repeated_layer": int(pass_match.group(2)),
        "generated_cases": int(pass_match.group(3)),
        "qproj_cases": int(pass_match.group(4)),
        "kproj_cases": int(pass_match.group(5) or 0),
        "vproj_cases": int(pass_match.group(6) or 0),
        "oproj_cases": int(pass_match.group(7) or 0),
        "mlp_gate_proj_cases": int(pass_match.group(8) or 0),
        "mlp_up_proj_cases": int(pass_match.group(9) or 0),
        "mlp_down_proj_cases": int(pass_match.group(10) or 0),
        "mlp_residual_cases": int(pass_match.group(11) or 0),
        "silu_gate_cases": int(pass_match.group(12) or 0),
        "rope_q_cases": int(pass_match.group(13) or 0),
        "rope_k_cases": int(pass_match.group(14) or 0),
        "kv_write_cases": int(pass_match.group(15) or 0),
        "attn_score_cases": int(pass_match.group(16) or 0),
        "softmax_cases": int(pass_match.group(17) or 0),
        "attn_value_cases": int(pass_match.group(18) or 0),
        "residual_cases": int(pass_match.group(19) or 0),
        "post_rms_cases": int(pass_match.group(20) or 0),
        "descriptor_errors": int(pass_match.group(21)),
        "memory_errors": int(pass_match.group(22)),
        "watchdog_errors": int(pass_match.group(23)),
        "reset_busy": int(pass_match.group(24)),
        "response_protocol_cases": int(pass_match.group(25) or 0),
        "log_sha256": sha256_file(log_path),
    }


def parse_mlp_up_sim(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"ACE2_SHELL_MLP_UP_TB_PASS\s+cases=([0-9]+)\s+writes=([0-9]+)\s+cycles=([0-9]+)",
        text,
    )
    if not match:
        raise RuntimeError("could not parse MLP up shell simulation pass summary")
    return {
        "mlp_up_proj_cases": int(match.group(1)),
        "mlp_up_proj_writes": int(match.group(2)),
        "mlp_up_proj_cycles": int(match.group(3)),
        "log_sha256": sha256_file(log_path),
    }


def parse_mlp_down_sim(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"ACE2_SHELL_MLP_DOWN_TB_PASS\s+cases=([0-9]+)\s+writes=([0-9]+)\s+cycles=([0-9]+)",
        text,
    )
    if not match:
        raise RuntimeError("could not parse MLP down shell simulation pass summary")
    return {
        "mlp_down_proj_cases": int(match.group(1)),
        "mlp_down_proj_writes": int(match.group(2)),
        "mlp_down_proj_cycles": int(match.group(3)),
        "log_sha256": sha256_file(log_path),
    }


def parse_mlp_residual_sim(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"ACE2_SHELL_MLP_RESIDUAL_TB_PASS\s+cases=([0-9]+)\s+writes=([0-9]+)\s+cycles=([0-9]+)\s+saturation_cases=([0-9]+)\s+descriptor_errors=([0-9]+)\s+memory_errors=([0-9]+)\s+protocol_cases=([0-9]+)",
        text,
    )
    if not match:
        raise RuntimeError("could not parse MLP residual shell simulation pass summary")
    return {
        "mlp_residual_cases": int(match.group(1)),
        "mlp_residual_writes": int(match.group(2)),
        "mlp_residual_cycles": int(match.group(3)),
        "saturation_cases": int(match.group(4)),
        "descriptor_error_cases": int(match.group(5)),
        "memory_error_cases": int(match.group(6)),
        "protocol_cases": int(match.group(7)),
        "log_sha256": sha256_file(log_path),
    }


def parse_qproj_stride_sim(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"ACE2_SHELL_QPROJ_STRIDE_TB_PASS\s+cases=([0-9]+)\s+rows=([0-9]+)\s+input_beats_per_row=([0-9]+)\s+writes=([0-9]+)\s+cycles=([0-9]+)",
        text,
    )
    if not match:
        raise RuntimeError("could not parse Q-proj stride simulation pass summary")
    return {
        "qproj_stride_cases": int(match.group(1)),
        "rows": int(match.group(2)),
        "input_beats_per_row": int(match.group(3)),
        "writes": int(match.group(4)),
        "cycles": int(match.group(5)),
        "log_sha256": sha256_file(log_path),
    }


def parse_silu_sim(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"ACE2_SHELL_SILU_GATE_TB_PASS\s+cases=([0-9]+)\s+writes=([0-9]+)\s+cycles=([0-9]+)\s+boundary_mask=([0-9a-fA-F]+)\s+sticky_checks=([0-9]+)",
        text,
    )
    if not match:
        raise RuntimeError("could not parse SiLU gate shell simulation pass summary")
    return {
        "silu_gate_cases": int(match.group(1)),
        "silu_gate_writes": int(match.group(2)),
        "silu_gate_cycles": int(match.group(3)),
        "boundary_coverage_mask": int(match.group(4), 16),
        "sticky_numeric_error_clean_checks": int(match.group(5)),
        "log_sha256": sha256_file(log_path),
    }


def artifact_hashes() -> list[dict[str, str]]:
    return [
        {"path": rel, "sha256": sha256_file(ROOT / rel)}
        for rel in ARTIFACT_FILES
        if (ROOT / rel).exists()
    ]


def source_hashes() -> list[dict[str, str]]:
    return [{"path": rel, "sha256": sha256_file(ROOT / rel)} for rel in SOURCE_FILES]


def reviewer_acceptance(
    rtl_hash: str,
    constraint_hash: str,
    evidence_hashes: dict[str, str],
) -> tuple[str, dict[str, Any] | None]:
    if not REVIEW_VERDICT.exists():
        return "pending", None

    verdict = load_json(REVIEW_VERDICT, None)
    if not isinstance(verdict, dict):
        raise RuntimeError("independent Reviewer verdict must be a JSON object")
    status = verdict.get("status")
    publication_authorized = verdict.get("publication_authorized")
    if status not in {"accepted", "rejected"}:
        raise RuntimeError("independent Reviewer verdict status must be accepted or rejected")
    if status == "rejected":
        if publication_authorized is not False:
            raise RuntimeError("rejected independent Reviewer verdict cannot authorize publication")
        return status, verdict
    if publication_authorized is not True:
        raise RuntimeError("accepted independent Reviewer verdict must authorize publication")

    expected_bindings: dict[str, Any] = {
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
    }
    stale = [
        key for key, expected in expected_bindings.items()
        if verdict.get(key) != expected
    ]
    reviewed_evidence = verdict.get("evidence_hashes")
    if not isinstance(reviewed_evidence, dict):
        raise RuntimeError("accepted independent Reviewer verdict lacks evidence_hashes")
    stale.extend(
        key for key, expected in evidence_hashes.items()
        if reviewed_evidence.get(key) != expected
    )
    findings = verdict.get("findings")
    if not isinstance(findings, list):
        raise RuntimeError("accepted independent Reviewer verdict lacks findings")
    if any(
        isinstance(finding, dict) and finding.get("severity") in {"blocker", "high"}
        for finding in findings
    ):
        raise RuntimeError("accepted independent Reviewer verdict contains blocking findings")
    if stale:
        raise RuntimeError(
            "accepted independent Reviewer verdict is stale for: " + ", ".join(stale)
        )
    return status, verdict


def entry_meets_acceptance(entry: dict[str, Any]) -> bool:
    sta = entry.get("sky130_sta", {})
    return (
        float(sta.get("wns_ns", -1.0)) >= 0.0
        and float(entry.get("fmax_mhz", 0.0)) >= FREQ_FLOOR_MHZ
        and float(entry.get("non_sram_area_mm2", math.inf)) <= AREA_CAP_MM2
    )


def entry_timing_shortfall(entry: dict[str, Any]) -> bool:
    sta = entry.get("sky130_sta", {})
    return (
        str(entry.get("decision", "")).endswith("_timing_shortfall_replan")
        or float(sta.get("wns_ns", 0.0)) < 0.0
        or float(entry.get("fmax_mhz", FREQ_FLOOR_MHZ)) < FREQ_FLOOR_MHZ
    )


def comparison_metrics(yosys: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_basis = baseline.get("area_accounting_basis")
    if baseline_basis == "hierarchical_total_including_rtl_submodules":
        return {
            "basis": baseline_basis,
            "cells": yosys["cells"],
            "baseline_cells": int(baseline["cells"]),
            "area_mm2": yosys["non_sram_area_mm2"],
            "baseline_area_mm2": float(baseline["non_sram_area_mm2"]),
            "limitation": None,
        }
    return {
        "basis": "legacy_module_local_excluding_submodules",
        "cells": yosys["local_cells_excluding_submodules"],
        "baseline_cells": int(baseline["cells"]),
        "area_mm2": yosys["local_non_sram_area_mm2"],
        "baseline_area_mm2": float(baseline["non_sram_area_mm2"]),
        "limitation": (
            "previous accepted same-prefix frontier lacks hierarchical total area because "
            "the binder formerly selected ace2_shell local module area; current total "
            "area is corrected, while compression delta is kept on the legacy local "
            "basis to avoid mixing accounting bases"
        ),
    }


def main() -> None:
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    yosys = parse_yosys(LATEST / "sky130_yosys.log")
    sta = parse_sta(LATEST / "sky130_sta.log")
    shell_sim = parse_shell_sim(LATEST / "rtl_shell_sim.log")
    qproj_stride_sim = parse_qproj_stride_sim(LATEST / "rtl_qproj_stride_sim.log")
    mlp_up_sim = parse_mlp_up_sim(LATEST / "rtl_mlp_up_sim.log")
    mlp_down_sim = parse_mlp_down_sim(LATEST / "rtl_mlp_down_sim.log")
    mlp_residual_sim = parse_mlp_residual_sim(LATEST / "rtl_mlp_residual_sim.log")
    silu_sim = parse_silu_sim(LATEST / "rtl_silu_gate_sim.log")

    rtl_hash = tree_hash([rel for rel in SOURCE_FILES if rel.startswith("rtl/")])
    constraint_hash = tree_hash(["constraints/ace2_rmsnorm_core.sdc", "flow/yosys/sky130_rmsnorm.ys", "flow/yosys/sky130_sta.tcl"])
    current_evidence_hashes = {
        "rtl_core_sim_log_sha256": sha256_file(LATEST / "rtl_core_sim.log"),
        "rtl_proj_sim_log_sha256": sha256_file(LATEST / "rtl_proj_sim.log"),
        "rtl_rope_sim_log_sha256": sha256_file(LATEST / "rtl_rope_sim.log"),
        "rtl_attention_score_sim_log_sha256": sha256_file(LATEST / "rtl_attention_score_sim.log"),
        "rtl_softmax_sim_log_sha256": sha256_file(LATEST / "rtl_softmax_sim.log"),
        "rtl_shell_sim_log_sha256": sha256_file(LATEST / "rtl_shell_sim.log"),
        "rtl_qproj_stride_sim_log_sha256": sha256_file(LATEST / "rtl_qproj_stride_sim.log"),
        "rtl_mlp_up_sim_log_sha256": sha256_file(LATEST / "rtl_mlp_up_sim.log"),
        "rtl_mlp_down_sim_log_sha256": sha256_file(LATEST / "rtl_mlp_down_sim.log"),
        "rtl_mlp_residual_sim_log_sha256": mlp_residual_sim["log_sha256"],
        "rtl_silu_gate_core_sim_log_sha256": sha256_file(LATEST / "rtl_silu_gate_core_sim.log"),
        "rtl_silu_gate_sim_log_sha256": sha256_file(LATEST / "rtl_silu_gate_sim.log"),
        "sky130_yosys_log_sha256": yosys["log_sha256"],
        "sky130_sta_log_sha256": sta["log_sha256"],
    }
    reviewer_status, review_verdict = reviewer_acceptance(
        rtl_hash, constraint_hash, current_evidence_hashes
    )

    ledger = load_json(LEDGER, {"schema_version": 1, "project": "ACE-2", "entries": []})
    ppa_reuse_baseline = next(
        (
            entry for entry in reversed(ledger.get("entries", []))
            if entry.get("rtl_hash") == rtl_hash
            and entry.get("constraint_hash") == constraint_hash
            and entry.get("evidence_hashes", {}).get("sky130_yosys_log_sha256")
            == current_evidence_hashes["sky130_yosys_log_sha256"]
            and entry.get("evidence_hashes", {}).get("sky130_sta_log_sha256")
            == current_evidence_hashes["sky130_sta_log_sha256"]
        ),
        None,
    )
    canonical_ppa_entry = None
    if ppa_reuse_baseline is not None:
        source_frontier_id = ppa_reuse_baseline.get(
            "ppa_evidence_binding", {}
        ).get("source_frontier_id")
        canonical_ppa_entry = next(
            (
                entry for entry in ledger.get("entries", [])
                if entry.get("id") == source_frontier_id
                and entry.get("rtl_hash") == rtl_hash
                and entry.get("constraint_hash") == constraint_hash
                and entry.get("evidence_hashes", {}).get(
                    "sky130_yosys_log_sha256"
                ) == current_evidence_hashes["sky130_yosys_log_sha256"]
                and entry.get("evidence_hashes", {}).get(
                    "sky130_sta_log_sha256"
                ) == current_evidence_hashes["sky130_sta_log_sha256"]
            ),
            None,
        )
        if canonical_ppa_entry is None:
            canonical_ppa_entry = ppa_reuse_baseline
        canonical_ppa_entry = dict(canonical_ppa_entry)
        canonical_cycle_impact = dict(
            canonical_ppa_entry.get("cycle_or_tokens_per_second_impact", {})
        )
        canonical_cycle_impact.update({
            "projection_core_groups_per_output": PROJ_GROUPS,
            "projection_weight_storage_beats_per_output": PROJ_WEIGHT_STORAGE_BEATS_PER_OUTPUT,
            "logical_w4_weight_bytes_per_output": PROJ_LOGICAL_WEIGHT_BYTES_PER_OUTPUT,
            "padded_weight_bytes_per_output": PROJ_PADDED_WEIGHT_BYTES_PER_OUTPUT,
        })
        canonical_ppa_entry["cycle_or_tokens_per_second_impact"] = canonical_cycle_impact
        canonical_ppa_entry["ppa_evidence_binding"] = {
            "mode": "fresh",
            "source_frontier_id": None,
            "reuse_basis": None,
        }
        canonical_ppa_entry["claim_boundary"] = (
            "fresh SKY130 mapped synthesis/OpenSTA PPA for bounded RTL operator "
            "slice; no routed/signoff claim"
        )
    ppa_reuse_source_id = (
        canonical_ppa_entry.get("id")
        if canonical_ppa_entry is not None
        else None
    )
    ppa_evidence_binding = {
        "mode": "reused_canonical" if ppa_reuse_baseline is not None else "fresh",
        "source_frontier_id": ppa_reuse_source_id,
        "reuse_basis": (
            "unchanged RTL hash, constraint hash, synthesis log hash, and STA log hash"
            if ppa_reuse_baseline is not None else None
        ),
    }
    prior_entries = []
    for ledger_entry in ledger.get("entries", []):
        if (
            canonical_ppa_entry is not None
            and ledger_entry.get("id") == canonical_ppa_entry.get("id")
        ):
            prior_entries.append(canonical_ppa_entry)
        elif not all(
            ledger_entry.get("evidence_hashes", {}).get(key) == value
            for key, value in current_evidence_hashes.items()
        ):
            prior_entries.append(ledger_entry)
    prior_baselines = [
        entry for entry in prior_entries
        if entry.get("ordered_supported_layer_operator_prefix") == PREVIOUS_PREFIX
        and entry.get("first_unsupported_layer_operator") == PREVIOUS_FIRST_UNSUPPORTED
        and entry_meets_acceptance(entry)
        and entry.get("accepted_prefix_advanced") is True
    ]
    if not prior_baselines:
        raise RuntimeError("could not find a passing prior frontier baseline for the new operator delta")
    baseline = prior_baselines[-1]
    comparison = comparison_metrics(yosys, baseline)
    baseline_area = comparison["baseline_area_mm2"]
    baseline_cells = comparison["baseline_cells"]
    non_sram_area = yosys["non_sram_area_mm2"]
    comparison_area = comparison["area_mm2"]
    delta_area_percent = ((comparison_area - baseline_area) / baseline_area) * 100.0
    fmax_floor = sta["estimated_fmax_mhz_floor_bound"]
    timing_met = sta["wns_ns"] >= 0 and fmax_floor >= FREQ_FLOOR_MHZ
    area_met = non_sram_area <= AREA_CAP_MM2
    meets_acceptance = timing_met and area_met
    miss_reason = "timing_shortfall" if not timing_met else "area_cap_exceeded"
    compression_baselines = [
        (index, entry) for index, entry in enumerate(prior_entries)
        if (
            entry.get("ordered_supported_layer_operator_prefix") == SUPPORTED_PREFIX
            and entry.get("first_unsupported_layer_operator") == FIRST_UNSUPPORTED
            and entry_meets_acceptance(entry)
            and entry.get("area_accounting_basis") == "hierarchical_total_including_rtl_submodules"
        )
        or (
            entry.get("candidate_layer_operator") == CANDIDATE_OPERATOR
            and entry_meets_acceptance(entry)
            and entry.get("area_accounting_basis") == "hierarchical_total_including_rtl_submodules"
            and (
                str(entry.get("mode", "")).startswith("ADVANCE_PAUSED_REUSE_REVIEW")
                or "pause_for_global_reuse_folding_review" in str(entry.get("decision", ""))
            )
        )
    ]
    latest_same_candidate_passing_index = max(
        (
            index for index, entry in enumerate(prior_entries)
            if entry.get("candidate_layer_operator") == CANDIDATE_OPERATOR
            and entry_meets_acceptance(entry)
        ),
        default=-1,
    )
    timing_shortfall_entries = [
        entry for index, entry in enumerate(prior_entries)
        if (
            entry.get("candidate_layer_operator") == CANDIDATE_OPERATOR
            or str(entry.get("decision", "")).startswith(f"{NEW_OPERATOR}_")
            or str(entry.get("decision", "")).startswith(f"{COMPRESSION_MECHANISM}_")
        )
        and index > latest_same_candidate_passing_index
        and entry_timing_shortfall(entry)
        and not any(
            later.get("timing_repair_from_id") == entry.get("id")
            for later in prior_entries
        )
    ]
    timing_repair_baseline = timing_shortfall_entries[-1] if timing_shortfall_entries else None
    is_timing_repair = timing_repair_baseline is not None and meets_acceptance
    compression_baseline = None if is_timing_repair else (compression_baselines[-1][1] if compression_baselines else None)

    compression_improvement_percent = None
    if compression_baseline is not None:
        compression_improvement_percent = (
            (float(compression_baseline["non_sram_area_mm2"]) - non_sram_area) /
            float(compression_baseline["non_sram_area_mm2"])
        ) * 100.0
    timing_improvement_percent = None
    if timing_repair_baseline is not None:
        prior_fmax = float(timing_repair_baseline.get("fmax_mhz", 0.0))
        if prior_fmax > 0.0:
            timing_improvement_percent = ((fmax_floor - prior_fmax) / prior_fmax) * 100.0
    if is_timing_repair:
        if delta_area_percent < 3.0:
            decision = f"{NEW_OPERATOR}_structural_timing_closure_met_100mhz_area_delta_lt_3pct_continue_advancing"
            current_mode = "ADVANCE"
            next_mode = "ADVANCE"
        else:
            decision = f"{NEW_OPERATOR}_structural_timing_closure_met_100mhz_area_delta_gte_3pct_pause_for_global_reuse_folding_review_before_{FIRST_UNSUPPORTED_LABEL}"
            current_mode = "ADVANCE_PAUSED_REUSE_REVIEW"
            next_mode = "ADVANCE_PAUSED_REUSE_REVIEW"
    elif compression_improvement_percent is not None:
        if meets_acceptance and compression_improvement_percent >= 3.0:
            decision = f"{COMPRESSION_MECHANISM}_improvement_gte_3pct_continue_global_compression"
            current_mode = "GLOBAL_COMPRESSION"
            next_mode = "GLOBAL_COMPRESSION"
        elif meets_acceptance:
            decision = f"{COMPRESSION_MECHANISM}_improvement_lt_3pct_stop_this_compression_direction_resume_advancement"
            current_mode = "GLOBAL_COMPRESSION_STOPPED"
            next_mode = "ADVANCE"
        else:
            decision = f"{COMPRESSION_MECHANISM}_{miss_reason}_replan"
            current_mode = "GLOBAL_COMPRESSION_REPLAN_TIMING" if not timing_met else "GLOBAL_COMPRESSION_REPLAN_AREA"
            next_mode = current_mode
    else:
        decision = (
            f"{NEW_OPERATOR}_bound_area_delta_lt_3pct_continue_advancing"
            if meets_acceptance and delta_area_percent < 3.0
            else f"{NEW_OPERATOR}_bound_area_delta_gte_3pct_pause_for_global_reuse_folding_review_before_{FIRST_UNSUPPORTED_LABEL}"
            if meets_acceptance
            else f"{NEW_OPERATOR}_bound_{miss_reason}_replan"
        )
        current_mode = (
            "ADVANCE"
            if meets_acceptance and delta_area_percent < 3.0
            else "ADVANCE_PAUSED_REUSE_REVIEW"
            if meets_acceptance
            else "ADVANCE_REPLAN_TIMING"
            if not timing_met
            else "ADVANCE_REPLAN_AREA"
        )
        next_mode = current_mode
    status = (
        "fresh_sky130_synth_sta_bound_reviewer_accepted"
        if meets_acceptance and reviewer_status == "accepted"
        else "fresh_sky130_synth_sta_bound_reviewer_rejected"
        if meets_acceptance and reviewer_status == "rejected"
        else "fresh_sky130_synth_sta_bound_meets_floor_awaiting_independent_reviewer"
        if meets_acceptance
        else "fresh_sky130_synth_sta_bound_timing_shortfall"
        if not timing_met
        else "fresh_sky130_synth_sta_bound_area_cap_exceeded"
    )
    reviewer_accepted = reviewer_status == "accepted"
    published = meets_acceptance and reviewer_accepted
    if meets_acceptance and not reviewer_accepted:
        if reviewer_status == "rejected":
            decision = f"{NEW_OPERATOR}_targets_met_independent_reviewer_rejected"
            current_mode = "ADVANCE_REVIEW_REJECTED"
        elif compression_improvement_percent is not None:
            decision = f"{decision}_awaiting_independent_reviewer"
            if current_mode == "GLOBAL_COMPRESSION_STOPPED":
                current_mode = "GLOBAL_COMPRESSION_STOPPED_REVIEW_PENDING"
                next_mode = "ADVANCE_REVIEW_PENDING"
            elif current_mode == "GLOBAL_COMPRESSION":
                current_mode = "GLOBAL_COMPRESSION_REVIEW_PENDING"
                next_mode = current_mode
            else:
                current_mode = f"{current_mode}_REVIEW_PENDING"
                next_mode = current_mode
        elif delta_area_percent >= 3.0:
            decision = (
                f"{NEW_OPERATOR}_targets_met_area_delta_gte_3pct_"
                f"pause_for_global_reuse_folding_review_before_{FIRST_UNSUPPORTED_LABEL}_"
                "awaiting_independent_reviewer"
            )
            current_mode = "ADVANCE_PAUSED_REUSE_REVIEW_PENDING_REVIEW"
        else:
            decision = f"{NEW_OPERATOR}_targets_met_awaiting_independent_reviewer"
            current_mode = "ADVANCE_REVIEW_PENDING"
        next_mode = current_mode
    accepted_prefix = SUPPORTED_PREFIX if published else PREVIOUS_PREFIX
    accepted_first_unsupported = FIRST_UNSUPPORTED if published else PREVIOUS_FIRST_UNSUPPORTED
    accepted_capabilities = [
        "input_rmsnorm",
        "w4a8_projection_q_proj",
        "w4a8_projection_k_proj",
        "w4a8_projection_v_proj",
        "rope_q",
        "rope_k",
        "kv_write",
        "attention_score",
        "softmax",
        "attention_value",
        "w4a8_projection_o_proj",
        "attention_residual_add",
        "post_attention_rmsnorm",
    ]
    accepted_capabilities.extend(["w4a8_projection_mlp_gate_proj", "w4a8_projection_mlp_up_proj", "int16_silu_gate_requant", "w4a8_projection_mlp_down_proj"])
    if published:
        accepted_capabilities.append("mlp_residual_add")

    cycle_impact = {
        "basis": f"verification/tb/ace2_shell_tb.sv direct command-dispatch regression with deterministic ready stalls, compressed {PROJ_MAC_LANES}-lane projection vectors, full-shape MLP/SiLU oracles, a distinct full-shape post-MLP residual oracle, {ROPE_LANES}-lane eighth-beat rope_q/rope_k vectors, kv_write, attention score/value, softmax, and post-attention RMSNorm",
        "operators": ["RMSNORM", "W4A8_PROJ", "O_PROJ", "MLP_GATE_PROJ", "MLP_UP_PROJ", "SILU_GATE", "MLP_DOWN_PROJ", "MLP_RESIDUAL_ADD", "ROPE", "KV_WRITE", "ATTN_SCORE", "SOFTMAX", "ATTN_VALUE", "RESIDUAL_ADD", "POST_ATTENTION_RMSNORM"],
        "projection_mac_lanes": PROJ_MAC_LANES,
        "rope_lanes": ROPE_LANES,
        "projection_core_groups_per_output": PROJ_GROUPS,
        "projection_weight_storage_beats_per_output": PROJ_WEIGHT_STORAGE_BEATS_PER_OUTPUT,
        "logical_w4_weight_bytes_per_output": PROJ_LOGICAL_WEIGHT_BYTES_PER_OUTPUT,
        "padded_weight_bytes_per_output": PROJ_PADDED_WEIGHT_BYTES_PER_OUTPUT,
        "mlp_intermediate_size": MLP_INTERMEDIATE_SIZE,
        "kv_write_bytes_per_token": KV_WRITE_BYTES_PER_TOKEN,
        "attention_score_head_dim": ATTN_SCORE_HEAD_DIM,
        "attention_score_context_max": ATTN_SCORE_CONTEXT_MAX,
        "attention_score_mac_lanes": ATTN_SCORE_MAC_LANES,
        "softmax_context_max": SOFTMAX_CONTEXT_MAX,
        "softmax_exp_step_q6_9": SOFTMAX_EXP_STEP_Q6_9,
        "attention_value_head_dim": ATTN_VALUE_HEAD_DIM,
        "attention_value_context_max": ATTN_VALUE_CONTEXT_MAX,
        "attention_value_mac_lanes": ATTN_VALUE_MAC_LANES,
        "qproj_cases": shell_sim["qproj_cases"],
        "kproj_cases": shell_sim["kproj_cases"],
        "vproj_cases": shell_sim["vproj_cases"],
        "oproj_cases": shell_sim["oproj_cases"],
        "mlp_gate_proj_cases": shell_sim["mlp_gate_proj_cases"],
        "mlp_up_proj_cases": mlp_up_sim["mlp_up_proj_cases"],
        "mlp_up_proj_writes": mlp_up_sim["mlp_up_proj_writes"],
        "mlp_up_proj_cycles": mlp_up_sim["mlp_up_proj_cycles"],
        "mlp_down_proj_cases": mlp_down_sim["mlp_down_proj_cases"],
        "mlp_down_proj_writes": mlp_down_sim["mlp_down_proj_writes"],
        "mlp_down_proj_cycles": mlp_down_sim["mlp_down_proj_cycles"],
        "mlp_residual_cases": mlp_residual_sim["mlp_residual_cases"],
        "mlp_residual_writes": mlp_residual_sim["mlp_residual_writes"],
        "mlp_residual_cycles": mlp_residual_sim["mlp_residual_cycles"],
        "mlp_residual_saturation_cases": mlp_residual_sim["saturation_cases"],
        "mlp_residual_descriptor_error_cases": mlp_residual_sim["descriptor_error_cases"],
        "mlp_residual_memory_error_cases": mlp_residual_sim["memory_error_cases"],
        "mlp_residual_protocol_cases": mlp_residual_sim["protocol_cases"],
        "qproj_stride_cases": qproj_stride_sim["qproj_stride_cases"],
        "qproj_stride_rows": qproj_stride_sim["rows"],
        "qproj_stride_input_beats_per_row": qproj_stride_sim["input_beats_per_row"],
        "silu_gate_cases": silu_sim["silu_gate_cases"],
        "silu_gate_writes": silu_sim["silu_gate_writes"],
        "silu_gate_cycles": silu_sim["silu_gate_cycles"],
        "silu_gate_boundary_coverage_mask": silu_sim["boundary_coverage_mask"],
        "silu_gate_sticky_numeric_error_clean_checks": silu_sim[
            "sticky_numeric_error_clean_checks"
        ],
        "rope_q_cases": shell_sim["rope_q_cases"],
        "rope_k_cases": shell_sim["rope_k_cases"],
        "kv_write_cases": shell_sim["kv_write_cases"],
        "attn_score_cases": shell_sim["attn_score_cases"],
        "softmax_cases": shell_sim["softmax_cases"],
        "attn_value_cases": shell_sim["attn_value_cases"],
        "residual_cases": shell_sim["residual_cases"],
        "post_attention_rmsnorm_cases": shell_sim["post_rms_cases"],
        "response_protocol_cases": shell_sim["response_protocol_cases"],
        "qproj_m_max": 16,
        "shell_test_average_cycles_per_success": shell_sim["average_cycles_per_success"],
        "shell_test_max_cycles_with_ready_stalls": shell_sim["max_cycles_per_success_with_test_stalls"],
        "shell_test_success_runs": shell_sim["success_runs"],
    }
    claim_boundaries = [
        (
            "The independent Reviewer accepted this bounded layer_0.mlp_residual_add candidate, so the complete ordered layer_0 prefix is published."
            if published
            else "This bounded candidate verifies layer_0.mlp_residual_add and meets the numeric PPA contract, but the published prefix remains through mlp_down_proj until independent Reviewer acceptance."
            if meets_acceptance
            else f"This bounded MLP residual candidate does not advance the accepted prefix because canonical SKY130 PPA misses the {'100 MHz floor' if not timing_met else '2.0 mm2 non-SRAM cap'}."
        ),
        f"The q_proj/k_proj/v_proj/o_proj/mlp_gate_proj/mlp_up_proj implementation uses a verified {PROJ_MAC_LANES}-lane global-compression projection datapath with {PROJ_GROUPS_PER_STORAGE_BEAT} folded {PROJ_MAC_LANES}-weight groups per padded 128-bit W4 storage beat.",
        "The prior accepted global compression folds shell RoPE execution to a 2-lane eighth-beat datapath while preserving 128-bit external memory reads/writes; RoPE table/scale addressing, eighth-beat selection, and rotate/round arithmetic are staged to keep the 100 MHz SKY130 timing floor.",
        "The rope_q/rope_k implementation uses one serialized shared RoPE datapath with exact sequence-position table addressing, 3584-byte Q table stride, 512-byte K table stride, round-to-nearest-even int8 saturation, eighth-beat output packing, and reset/soft-reset folded transient payload, table-address, and command-ready staging.",
        "The kv_write implementation reuses the shell DMA path to append 272-byte records at scratch_addr + sequence_position*272: 128-byte signed-int8 K, 128-byte signed-int8 V, and one descriptor-defined 16-byte per-token/per-KV-head scale metadata beat copied bit-for-bit from aligned scale_addr.",
        "The attention_score implementation adds opcode 0x04 for one query-head Q*K score tiles over up to eight K-cache tokens per descriptor, using a shell-inline staged 1-lane Q*K MAC and the shared DMA payload path for one packed Q6.9 score write beat; the standalone attention-score core remains unit-regressed but is not instantiated in the current PPA top after this same-prefix fold.",
        "The softmax implementation adds opcode 0x05 for one up-to-eight-token Q6.9 score beat, max-subtracts active scores, applies a generated fixed-point exp LUT at 1/8-logit steps, sums Q0.15 weights, normalizes with a registered trial-shift/compare/subtract divider pipeline and round-to-nearest-even, and writes one packed Q0.15 probability beat.",
        "The attention_value implementation adds opcode 0x06 for one up-to-eight-token unsigned Q0.15 probability beat and up to eight 64-byte signed-int8 V rows, reuses the shell-inline attention product, signed-int32 accumulator, DMA, and payload storage, rounds to nearest-even after a 15-bit right shift, saturates to signed int8, and writes one 64-byte vector as four 128-bit output beats; it does not apply the stored KV scale metadata.",
        "The o_proj implementation reuses opcode 0x01 W4A8_PROJ with m<=16, n==896, k==896, the same 4-lane projection core, per-output requantization metadata, and 128-bit DMA write path to project the complete attention-output hidden vector.",
        f"The accepted mlp_gate_proj and mlp_up_proj capabilities reuse opcode 0x01 W4A8_PROJ with m<=16, n=={MLP_INTERMEDIATE_SIZE}, k==896, full 64-bit descriptor base plus narrowed projection byte-offset counters, the same 4-lane projection core, per-output requantization metadata, and 128-bit DMA write path.",
        f"The accepted MLP down projection reuses opcode 0x01 W4A8_PROJ with m<=16, n==896, k=={MLP_INTERMEDIATE_SIZE}, the same 4-lane projection core, full-width tensor bases, per-output requantization metadata, and 128-bit DMA writes.",
        "The MLP residual candidate reuses accepted opcode 0x08 and the 16-lane signed-int8 saturating-add DMA path for the full 896-element post-MLP tensor; a distinct oracle covers post-MLP data, write addresses, saturation, descriptor rejection, memory faults, response-tag faults, and backpressure.",
        "The accepted SiLU gate uses opcode 0x07 with signed int16 Q6.9 gate/up inputs, a clipped floor-indexed Q3.12 sigmoid LUT, signed Q9.21 products, descriptor-wide multiplier/shift/zero-point metadata, round-to-nearest-even, and int8 saturation.",
        "The accepted attention_residual_add operator uses opcode 0x08 for 896 signed-int8 elements, reuses the shell DMA and shared 128-bit payload storage, performs 16-lane saturating add, and exposes numeric saturation through completion/error status.",
        "The accepted post_attention_rmsnorm operator reuses opcode 0x02, the existing two-pass RMSNorm core, DMA, payload, and gain paths on separate post-attention oracle vectors.",
        "The PPA evidence is SKY130 mapped synthesis plus OpenSTA timing from ORFS with hierarchical total RTL submodule area accounting, not routed GDS, DRC, LVS, FPGA, signoff, tapeout, or silicon evidence.",
    ]
    if compression_improvement_percent is not None:
        threshold_result = "meets" if compression_improvement_percent >= 3.0 else "is below"
        claim_boundaries.append(
            f"The current same-prefix {COMPRESSION_MECHANISM} step improved non-SRAM area by {compression_improvement_percent:.3f}% versus compression baseline {compression_baseline.get('id')}; this {threshold_result} the 3% continuation threshold."
        )
    if timing_repair_baseline is not None:
        claim_boundaries.append(
            f"The current mlp_residual_add timing repair is bound to {timing_repair_baseline.get('id')} using {TIMING_REPAIR_MECHANISM}."
        )
    if ppa_reuse_baseline is not None:
        claim_boundaries.append(
            f"Canonical SKY130 synthesis/STA evidence is reused from {ppa_reuse_source_id} because RTL, constraints, and both PPA log hashes are unchanged."
        )
    if comparison["limitation"]:
        claim_boundaries.append(comparison["limitation"])
    claim_boundaries.append("Full decoder/model correctness and quality remain unmet until later workload traces are implemented and verified.")

    manifest = {
        "schema_version": 1,
        "project": "ACE-2",
        "stage": "rtl",
        "stage_closing": False,
        "generated_at_utc": generated_at,
        "supported_layer_operator_prefix": accepted_prefix,
        "first_unsupported_layer_operator": accepted_first_unsupported,
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "candidate_status": status,
        "candidate_meets_numeric_acceptance": meets_acceptance,
        "candidate_geometry": {
            "m": 1,
            "n": HIDDEN_SIZE,
            "k": 0,
            "lanes": 16,
            "beats": HIDDEN_SIZE // 16,
        },
        "independent_reviewer_acceptance": reviewer_status,
        "independent_reviewer_verdict": (
            "evidence/review/latest/mlp_residual_add_verdict.json"
            if review_verdict is not None else None
        ),
        "top_module_for_current_ppa": "ace2_shell",
        "ppa_evidence_binding": ppa_evidence_binding,
        "supported_reusable_operator_capabilities": accepted_capabilities,
        "parameters": {
            "HIDDEN_SIZE": HIDDEN_SIZE,
            "LANES": 16,
            "ACT_WIDTH": 8,
            "GAIN_WIDTH": 16,
            "ACC_WIDTH": 48,
            "INV_RMS_FRAC": 30,
            "GAIN_FRAC": 14,
            "PROJ_ACC_WIDTH": 32,
            "PROJ_MAC_LANES": PROJ_MAC_LANES,
            "PROJ_GROUPS": PROJ_GROUPS,
            "PROJ_M_MAX": 16,
            "MLP_INTERMEDIATE_SIZE": MLP_INTERMEDIATE_SIZE,
            "PROJ_ADDRESSING": "full_64_bit_descriptor_base_plus_tensor_offset",
            "PROJ_WEIGHT_STORAGE_BEATS_PER_OUTPUT": PROJ_WEIGHT_STORAGE_BEATS_PER_OUTPUT,
            "PROJ_LOGICAL_WEIGHT_BYTES_PER_OUTPUT": PROJ_LOGICAL_WEIGHT_BYTES_PER_OUTPUT,
            "PROJ_PADDED_WEIGHT_BYTES_PER_OUTPUT": PROJ_PADDED_WEIGHT_BYTES_PER_OUTPUT,
            "ROPE_MAX_SEQUENCE_POSITION": 32767,
            "ROPE_LANES": ROPE_LANES,
            "ROPE_TABLE_FORMAT": "cos/sin Q1.15, 64 bytes per beat pair table slice",
            "KV_WRITE_BYTES_PER_TOKEN": KV_WRITE_BYTES_PER_TOKEN,
            "KV_WRITE_RECORD_LAYOUT": "128B signed-int8 K, 128B signed-int8 V, 16B descriptor-defined per-token/per-KV-head scale metadata",
            "KV_SCALE_METADATA_BEHAVIOR": "bit-exact copy from 16-byte-aligned scale_addr; not reinterpreted by bounded RTL",
            "ATTN_SCORE_HEAD_DIM": ATTN_SCORE_HEAD_DIM,
            "ATTN_SCORE_CONTEXT_MAX": ATTN_SCORE_CONTEXT_MAX,
            "ATTN_SCORE_MAC_LANES": ATTN_SCORE_MAC_LANES,
            "SOFTMAX_CONTEXT_MAX": SOFTMAX_CONTEXT_MAX,
            "SOFTMAX_EXP_STEP_Q6_9": SOFTMAX_EXP_STEP_Q6_9,
            "ATTN_VALUE_HEAD_DIM": ATTN_VALUE_HEAD_DIM,
            "ATTN_VALUE_CONTEXT_MAX": ATTN_VALUE_CONTEXT_MAX,
            "ATTN_VALUE_MAC_LANES": ATTN_VALUE_MAC_LANES,
            "ATTN_VALUE_PROBABILITY_FORMAT": "unsigned Q0.15",
            "ATTN_VALUE_ACCUMULATOR_FORMAT": "signed int32",
            "ATTN_VALUE_REQUANTIZATION": "round-to-nearest-even right shift by 15, then signed-int8 saturation",
            "ATTN_VALUE_OUTPUT_BYTES": 64,
            "SILU_GATE_INPUT_FORMAT": "signed int16 Q6.9",
            "SILU_GATE_LUT_FORMAT": "signed int16 Q3.12, floor-indexed at 1/8 steps and clipped to [-8,8]",
            "SILU_GATE_PRODUCT_FORMAT": "signed int32 Q9.21",
            "SILU_GATE_REQUANTIZATION": "descriptor-wide signed-int32 multiplier, u6 right shift, signed-int8 zero point, round-to-nearest-even, signed-int8 saturation",
        },
        "interfaces": {
            "clock": "clk_i single accelerator clock",
            "reset": "rst_ni active-low asynchronous assertion at the module boundary",
            "csr": "32-bit address, 64-bit data, valid/ready request with valid response",
            "command_dispatch": "direct descriptor ingress; W4A8_PROJ opcode 0x01, RMSNORM opcode 0x02, ROPE opcode 0x03, ATTN_SCORE opcode 0x04, SOFTMAX opcode 0x05, ATTN_VALUE opcode 0x06, SILU_GATE opcode 0x07, RESIDUAL_ADD opcode 0x08, and KV_WRITE opcode 0x0A",
            "external_memory_dma": "abstract 128-bit request/write/read/response ready-valid channels with 8-bit tags and 16-bit burst length",
            "scratchpad": "8 logical 128-bit SRAM-bank ports are exposed as blackbox macro boundary signals and held inactive by the current slice",
            "completion": "cmd_done_valid/cmd_done_ready plus tag, error, sumsq, inv_rms_q30, saturation_seen",
        },
        "layer_execution_coverage": {
            "basis": "verification/tb/ace2_shell_tb.sv direct command-dispatch regression",
            "input_rmsnorm_layer_ids": list(range(24)),
            "q_proj_layer_ids": [0],
            "k_proj_layer_ids": [0],
            "v_proj_layer_ids": [0],
            "rope_q_layer_ids": [0],
            "rope_k_layer_ids": [0],
            "kv_write_layer_ids": [0],
            "attention_score_layer_ids": [0],
            "softmax_layer_ids": [0],
            "attention_value_layer_ids": [0],
            "o_proj_layer_ids": [0],
            "o_proj_candidate_layer_ids": [0],
            "attention_residual_add_layer_ids": [0],
            "attention_residual_add_candidate_layer_ids": [0],
            "post_attention_rmsnorm_layer_ids": [0],
            "post_attention_rmsnorm_candidate_layer_ids": [0],
            "mlp_gate_proj_layer_ids": [0],
            "mlp_up_proj_layer_ids": [0],
            "silu_gate_layer_ids": [0],
            "mlp_down_proj_layer_ids": [0],
            "mlp_residual_add_layer_ids": [0] if published else [],
            "mlp_residual_add_candidate_layer_ids": [0],
            "repeated_layer_indices_tested": [7],
        },
        "synthesized_modules": SYNTHESIZED_MODULES,
        "traceability": [
            {
                "requirement": (
                    "design/WORKLOAD.md layer_0.mlp_residual_add; the final ordered layer_0 operator publishes only after numeric targets and independent Reviewer acceptance"
                ),
                "implementation": "rtl/ace2_shell.sv reusing accepted opcode 0x08, 16-lane signed-int8 saturating add, shared payload storage, and the abstract 128-bit DMA boundary without changing RTL interfaces",
                "verification": f"independent post-MLP Python oracle, {mlp_residual_sim['mlp_residual_cases']} full-shape targeted cases with {mlp_residual_sim['mlp_residual_writes']} writes in the final case, {mlp_residual_sim['descriptor_error_cases']} descriptor errors, {mlp_residual_sim['memory_error_cases']} memory errors, {mlp_residual_sim['protocol_cases']} protocol cases, and the complete accepted-prefix regression plus all dedicated MLP/SiLU targets",
            },
            {
                "requirement": f"design/SPEC.md opcode 0x01 W4A8 projection with q_proj/o_proj n=896, k_proj/v_proj n=128, mlp_gate_proj/mlp_up_proj n={MLP_INTERMEDIATE_SIZE} and k=896, and mlp_down_proj n=896 and k={MLP_INTERMEDIATE_SIZE}; accepted prior opcode 0x02/0x03/0x04/0x05/0x06/0x07/0x08/0x0A operators remain regressed",
                "implementation": f"CSR shell, descriptor validation, streaming activation/weight/metadata/table memory sequencing, {PROJ_MAC_LANES}-lane projection MAC/requant/saturate datapath reused by q_proj/k_proj/v_proj/o_proj/mlp_gate_proj/mlp_up_proj/mlp_down_proj, full 64-bit projection tensor base plus reduction-dependent byte offsets, eighth-beat {ROPE_LANES}-lane shell RoPE sequencing, KV_WRITE cache append, attention-score/value sequencing, softmax, SiLU, residual-add, post-attention RMSNorm, shared payload packing, narrowed CSR control/status storage, and split registered command-load enables for timing-safe descriptor execution",
                "verification": "directed invalid descriptors, 24 legal RMSNorm layer IDs, q_proj/k_proj/v_proj/o_proj layer_0 vectors, full-shape MLP gate/up/down projection and post-MLP residual oracles, a two-row Q-proj memory-stride discriminator, SiLU, rope_q and rope_k vectors including high sequence position, bit-exact kv_write append vectors, attention_score, softmax, attention_value, residual-add, post-attention RMSNorm, memory errors, watchdog timeout, reset, ready stalls, write-address checks, and output comparisons",
            },
            {
                "requirement": "design/SPEC.md command lifecycle and reset semantics for the synthesized shell state",
                "implementation": "rtl/ace2_shell.sv module ace2_state_shadow provides the registered timing-safe state handoff instantiated as u_prefix_state_shadow",
                "verification": "complete shell regression covering command sequencing, busy reset, watchdog, response errors, and ready stalls",
            },
        ],
        "ip_provenance": [
            {"name": "ace2_rmsnorm_core", "kind": "new_project_rtl", "source_revision": "active_worktree", "license": "repository project license not separately declared in this manifest", "third_party": False},
            {"name": "ace2_w4a8_proj_core", "kind": "new_project_rtl", "source_revision": "active_worktree", "license": "repository project license not separately declared in this manifest", "third_party": False},
            {"name": "ace2_rope_core", "kind": "new_project_rtl", "source_revision": "active_worktree", "license": "repository project license not separately declared in this manifest", "third_party": False},
            {"name": "ace2_attention_score_core", "kind": "new_project_rtl", "source_revision": "active_worktree", "license": "repository project license not separately declared in this manifest", "third_party": False},
            {"name": "ace2_softmax_core", "kind": "new_project_rtl", "source_revision": "active_worktree", "license": "repository project license not separately declared in this manifest", "third_party": False},
            {"name": "ace2_silu_gate_core", "kind": "new_project_rtl", "source_revision": "active_worktree", "license": "repository project license not separately declared in this manifest", "third_party": False},
            {"name": "ace2_shell", "kind": "new_project_rtl", "source_revision": "active_worktree", "license": "repository project license not separately declared in this manifest", "third_party": False},
            {"name": "ace2_state_shadow", "kind": "new_project_rtl", "source_revision": "active_worktree", "license": "repository project license not separately declared in this manifest", "third_party": False},
            {"name": "rmsnorm_vectors.svh", "kind": "generated_verification_source", "regeneration_command": "python3 tools/gen_rmsnorm_vectors.py", "generator": "tools/gen_rmsnorm_vectors.py", "reference": "tools/ace2_rmsnorm_reference.py", "third_party": False},
            {"name": "projection_vectors.svh", "kind": "generated_verification_source", "regeneration_command": "python3 tools/gen_projection_vectors.py", "generator": "tools/gen_projection_vectors.py", "reference": "tools/ace2_projection_reference.py", "third_party": False},
            {"name": "rope_vectors.svh", "kind": "generated_verification_source", "regeneration_command": "python3 tools/gen_rope_vectors.py", "generator": "tools/gen_rope_vectors.py", "reference": "tools/ace2_rope_reference.py", "third_party": False},
            {"name": "attention_score_vectors.svh", "kind": "generated_verification_source", "regeneration_command": "python3 tools/gen_attention_score_vectors.py", "generator": "tools/gen_attention_score_vectors.py", "reference": "tools/ace2_attention_score_reference.py", "third_party": False},
            {"name": "softmax_vectors.svh", "kind": "generated_verification_source", "regeneration_command": "python3 tools/gen_softmax_vectors.py", "generator": "tools/gen_softmax_vectors.py", "reference": "tools/ace2_softmax_reference.py", "third_party": False},
            {"name": "silu_gate_vectors.svh", "kind": "generated_verification_source", "regeneration_command": "python3 tools/gen_silu_gate_vectors.py", "generator": "tools/gen_silu_gate_vectors.py", "reference": "tools/ace2_silu_gate_reference.py", "third_party": False},
            {"name": "ace2_silu_lut.svh", "path": "rtl/generated/ace2_silu_lut.svh", "kind": "generated_rtl_source", "regeneration_command": "python3 tools/gen_silu_gate_vectors.py", "generator": "tools/gen_silu_gate_vectors.py", "reference": "tools/ace2_silu_gate_reference.py", "third_party": False},
            {"name": "attention_value_vectors.svh", "kind": "generated_verification_source", "regeneration_command": "python3 tools/gen_attention_value_vectors.py", "generator": "tools/gen_attention_value_vectors.py", "reference": "tools/ace2_attention_value_reference.py", "third_party": False},
            {"name": "residual_vectors.svh", "kind": "generated_verification_source", "regeneration_command": "python3 tools/gen_residual_vectors.py", "generator": "tools/gen_residual_vectors.py", "reference": "tools/ace2_residual_reference.py", "third_party": False},
            {"name": "mlp_residual_vectors.svh", "kind": "generated_verification_source", "regeneration_command": "python3 tools/gen_mlp_residual_vectors.py", "generator": "tools/gen_mlp_residual_vectors.py", "reference": "tools/ace2_mlp_residual_reference.py", "third_party": False},
            {"name": "post_attention_rmsnorm_vectors.svh", "kind": "generated_verification_source", "regeneration_command": "python3 tools/gen_post_attention_rmsnorm_vectors.py", "generator": "tools/gen_post_attention_rmsnorm_vectors.py", "reference": "tools/ace2_rmsnorm_reference.py", "third_party": False},
        ],
        "source_hashes": source_hashes(),
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "latest_evidence": {
            "lint": "make rtl-lint",
            "simulation": "make rtl-sim",
            "rtl_core_sim_log": "evidence/frontier/latest/rtl_core_sim.log",
            "rtl_proj_sim_log": "evidence/frontier/latest/rtl_proj_sim.log",
            "rtl_rope_sim_log": "evidence/frontier/latest/rtl_rope_sim.log",
            "rtl_attention_score_sim_log": "evidence/frontier/latest/rtl_attention_score_sim.log",
            "rtl_softmax_sim_log": "evidence/frontier/latest/rtl_softmax_sim.log",
            "rtl_shell_sim_log": "evidence/frontier/latest/rtl_shell_sim.log",
            "rtl_qproj_stride_sim_log": "evidence/frontier/latest/rtl_qproj_stride_sim.log",
            "rtl_mlp_up_sim_log": "evidence/frontier/latest/rtl_mlp_up_sim.log",
            "rtl_mlp_down_sim_log": "evidence/frontier/latest/rtl_mlp_down_sim.log",
            "rtl_mlp_residual_sim_log": "evidence/frontier/latest/rtl_mlp_residual_sim.log",
            "rtl_silu_gate_core_sim_log": "evidence/frontier/latest/rtl_silu_gate_core_sim.log",
            "rtl_silu_gate_sim_log": "evidence/frontier/latest/rtl_silu_gate_sim.log",
            "sky130_synthesis_log": "evidence/frontier/latest/sky130_yosys.log",
            "sky130_sta_log": "evidence/frontier/latest/sky130_sta.log",
            "independent_reviewer_verdict": (
                "evidence/review/latest/mlp_residual_add_verdict.json"
                if review_verdict is not None else None
            ),
            "shell_sim": shell_sim,
            "qproj_stride_sim": qproj_stride_sim,
            "mlp_up_sim": mlp_up_sim,
            "mlp_down_sim": mlp_down_sim,
            "mlp_residual_sim": mlp_residual_sim,
            "silu_gate_sim": silu_sim,
            "sky130_yosys": yosys,
            "sky130_sta": sta,
            "ppa_evidence_binding": ppa_evidence_binding,
        },
        "claim_boundaries": claim_boundaries,
    }
    write_json(MANIFEST, manifest)

    entry = {
        "id": f"frontier-{generated_at}",
        "recorded_at_utc": generated_at,
        "stage_closing": False,
        "ordered_supported_layer_operator_prefix": accepted_prefix,
        "first_unsupported_layer_operator": accepted_first_unsupported,
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "accepted_prefix_advanced": published,
        "candidate_meets_numeric_acceptance": meets_acceptance,
        "independent_reviewer_acceptance": reviewer_status,
        "independent_reviewer_verdict_sha256": (
            sha256_file(REVIEW_VERDICT) if review_verdict is not None else None
        ),
        "mode": current_mode,
        "decision": decision,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "cells": yosys["cells"],
        "delta_cells": comparison["cells"] - baseline_cells,
        "delta_cells_basis": comparison["basis"],
        "non_sram_area_mm2": non_sram_area,
        "non_sram_area_um2": yosys["non_sram_area_um2"],
        "area_accounting_basis": yosys["area_accounting_basis"],
        "local_cells_excluding_submodules": yosys["local_cells_excluding_submodules"],
        "local_non_sram_area_mm2": yosys["local_non_sram_area_mm2"],
        "delta_area_mm2": comparison_area - baseline_area,
        "delta_area_percent": delta_area_percent,
        "compression_improvement_percent": compression_improvement_percent,
        "compression_mechanism": COMPRESSION_MECHANISM if compression_improvement_percent is not None else None,
        "timing_improvement_percent": timing_improvement_percent,
        "timing_repair_mechanism": TIMING_REPAIR_MECHANISM if timing_repair_baseline is not None else None,
        "timing_repair_from_id": timing_repair_baseline.get("id") if timing_repair_baseline is not None else None,
        "compression_baseline_id": compression_baseline.get("id") if compression_baseline is not None else None,
        "compression_baseline_non_sram_area_mm2": float(compression_baseline["non_sram_area_mm2"]) if compression_baseline is not None else None,
        "comparison_area_accounting_basis": comparison["basis"],
        "comparison_non_sram_area_mm2": comparison_area,
        "comparison_baseline_id": baseline.get("id"),
        "comparison_limitation": comparison["limitation"],
        "fmax_mhz": fmax_floor,
        "fmax_mhz_floor_bound": fmax_floor,
        "frequency_floor_met": timing_met,
        "area_cap_mm2": AREA_CAP_MM2,
        "area_cap_met": area_met,
        "remaining_area_reserve_mm2": AREA_CAP_MM2 - non_sram_area,
        "remaining_frequency_reserve_mhz": max(0.0, fmax_floor - FREQ_FLOOR_MHZ),
        "remaining_frequency_reserve_mhz_floor_bound": 0.0,
        "sky130_sta": {"clock_period_ns": sta["clock_period_ns"], "wns_ns": sta["wns_ns"], "tns_ns": sta["tns_ns"]},
        "cycle_or_tokens_per_second_impact": cycle_impact,
        "evidence": {
            "rtl_manifest": "design/RTL_MANIFEST.json",
            "rtl_core_sim_log": "evidence/frontier/latest/rtl_core_sim.log",
            "rtl_proj_sim_log": "evidence/frontier/latest/rtl_proj_sim.log",
            "rtl_rope_sim_log": "evidence/frontier/latest/rtl_rope_sim.log",
            "rtl_attention_score_sim_log": "evidence/frontier/latest/rtl_attention_score_sim.log",
            "rtl_softmax_sim_log": "evidence/frontier/latest/rtl_softmax_sim.log",
            "rtl_mlp_up_sim_log": "evidence/frontier/latest/rtl_mlp_up_sim.log",
            "rtl_mlp_down_sim_log": "evidence/frontier/latest/rtl_mlp_down_sim.log",
            "rtl_mlp_residual_sim_log": "evidence/frontier/latest/rtl_mlp_residual_sim.log",
            "rtl_qproj_stride_sim_log": "evidence/frontier/latest/rtl_qproj_stride_sim.log",
            "rtl_silu_gate_core_sim_log": "evidence/frontier/latest/rtl_silu_gate_core_sim.log",
            "rtl_silu_gate_sim_log": "evidence/frontier/latest/rtl_silu_gate_sim.log",
            "shell_sim_log": "evidence/frontier/latest/rtl_shell_sim.log",
            "yosys_log": "evidence/frontier/latest/sky130_yosys.log",
            "sta_log": "evidence/frontier/latest/sky130_sta.log",
            "independent_reviewer_verdict": (
                "evidence/review/latest/mlp_residual_add_verdict.json"
                if review_verdict is not None else None
            ),
        },
        "evidence_hashes": current_evidence_hashes,
        "ppa_evidence_binding": ppa_evidence_binding,
        "claim_boundary": (
            "canonical SKY130 mapped synthesis/OpenSTA PPA reused for unchanged RTL and constraints; no routed/signoff claim"
            if ppa_reuse_baseline is not None
            else "fresh SKY130 mapped synthesis/OpenSTA PPA for bounded RTL operator slice; no routed/signoff claim"
        ),
    }
    ledger["entries"] = prior_entries + [entry]
    write_json(LEDGER, ledger)

    latest_frontier = {
        "status": status,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "cells": yosys["cells"],
        "non_sram_area_mm2": non_sram_area,
        "area_accounting_basis": yosys["area_accounting_basis"],
        "local_non_sram_area_mm2": yosys["local_non_sram_area_mm2"],
        "fmax_mhz": fmax_floor,
        "fmax_mhz_floor_bound": fmax_floor,
        "frequency_floor_met": timing_met,
        "area_cap_mm2": AREA_CAP_MM2,
        "area_cap_met": area_met,
        "wns_ns": sta["wns_ns"],
        "tns_ns": sta["tns_ns"],
        "cycle_or_tokens_per_second_impact": cycle_impact,
        "delta_area_percent": delta_area_percent,
        "compression_improvement_percent": compression_improvement_percent,
        "compression_mechanism": COMPRESSION_MECHANISM if compression_improvement_percent is not None else None,
        "timing_improvement_percent": timing_improvement_percent,
        "timing_repair_mechanism": TIMING_REPAIR_MECHANISM if timing_repair_baseline is not None else None,
        "timing_repair_from_id": timing_repair_baseline.get("id") if timing_repair_baseline is not None else None,
        "compression_baseline_id": compression_baseline.get("id") if compression_baseline is not None else None,
        "compression_baseline_non_sram_area_mm2": float(compression_baseline["non_sram_area_mm2"]) if compression_baseline is not None else None,
        "comparison_area_accounting_basis": comparison["basis"],
        "comparison_non_sram_area_mm2": comparison_area,
        "comparison_limitation": comparison["limitation"],
        "remaining_area_reserve_mm2": AREA_CAP_MM2 - non_sram_area,
        "remaining_frequency_reserve_mhz": max(0.0, fmax_floor - FREQ_FLOOR_MHZ),
        "remaining_frequency_reserve_mhz_floor_bound": 0.0,
    }

    public = load_json(PUBLIC_STATUS, {})
    pipeline = load_json(PIPELINE_STATE, {})
    manager_stage = str(pipeline.get("current_stage", "unknown"))
    stage_order = ["definition", "architecture", "environment", "rtl", "verification", "ppa", "prototype", "benchmark", "signoff"]
    completed_stages = [
        stage for stage in stage_order
        if pipeline.get("stages", {}).get(stage, {}).get("status") == "done"
    ]
    current_stage_index = stage_order.index(manager_stage) if manager_stage in stage_order else -1
    downstream_stages = stage_order[current_stage_index + 1:] if current_stage_index >= 0 else []
    public["schema_version"] = 1
    public["project"] = "ACE-2"
    public["vertical"] = "chip_design"
    public["generated_at_utc"] = generated_at
    public["last_updated_utc"] = generated_at
    public["artifact_hashes"] = artifact_hashes()
    if published:
        blockers = []
    elif meets_acceptance:
        blockers = [{
            "id": "mlp_residual_add_independent_review",
            "status": "open",
            "summary": (
                "Independent Reviewer rejected the MLP residual-add candidate."
                if reviewer_status == "rejected"
                else "MLP residual-add candidate meets numeric PPA targets but remains unpublished pending independent Reviewer acceptance."
            ),
        }]
        if delta_area_percent >= 3.0:
            blockers.append({
                "id": "mlp_residual_add_global_reuse_review",
                "status": "open",
                "summary": (
                    "MLP residual-add candidate increases non-SRAM area by at least 3%; "
                    f"pause capability advancement for global reuse/folding review before {FIRST_UNSUPPORTED}."
                ),
            })
    else:
        blockers = [{
            "id": f"mlp_residual_add_sky130_{miss_reason}",
            "status": "open",
            "summary": f"MLP residual-add candidate misses the operator-owned {'100 MHz SKY130 floor' if not timing_met else '2.0 mm2 non-SRAM cap'} and cannot be counted as supported.",
        }]
    public["blockers"] = blockers
    public["privacy_policy"] = {
        "contains_credentials": False,
        "contains_private_paths": False,
        "contains_private_pdk_contents": False,
        "contains_prompts": False,
        "contains_raw_private_logs": False,
        "public_safe": True,
    }
    public["stage"] = {
        "current_stage": manager_stage,
        "completed_prior_stages": completed_stages,
        "current_stage_source": "research/PIPELINE_STATE.json",
        "stage_transition_owner": "Manager",
        "planner_may_advance_stage": False,
        "downstream_locked_until_manager_advance": downstream_stages,
        "current_stage_checklist": {},
        "current_stage_evidence": [],
    }
    public["dashboard_fields"] = {
        "supported_layers": accepted_prefix,
        "supported_reusable_operator_capabilities": accepted_capabilities,
        "input_rmsnorm_layer_execution_coverage": "layer_id_0_through_23_plus_repeated_layer_7",
        "ordered_supported_layer_operator_prefix": accepted_prefix,
        "first_unsupported_layer_operator": accepted_first_unsupported,
        "current_mode": next_mode,
        "latest_decision": decision,
        "latest_ppa_frontier_status": status,
        "latest_ppa_frontier": latest_frontier,
    }
    public["implementation_frontier"] = {
        "tracking_order_definition": "design/WORKLOAD.md#ordered-layer/operator-frontier",
        "ordered_supported_layer_operator_prefix": accepted_prefix,
        "first_unsupported_layer_operator": {
            "identifier": accepted_first_unsupported,
            "layer": None if accepted_first_unsupported == "none" else int(accepted_first_unsupported.split(".", 1)[0].split("_", 1)[1]),
            "operator": (
                "none"
                if accepted_first_unsupported == "none"
                else accepted_first_unsupported.split(".", 1)[1]
            ),
            "reason": (
                "The independently accepted mlp_residual_add candidate completes the ordered layer_0 operator prefix; the next accepted-prefix item is layer_1.input_rmsnorm until shared-layer reuse is independently accepted."
                if published
                else "The mlp_residual_add candidate is unpublished pending independent Reviewer acceptance, so the accepted prefix remains through mlp_down_proj."
                if meets_acceptance
                else f"The mlp_residual_add candidate misses the {'100 MHz floor' if not timing_met else '2.0 mm2 non-SRAM cap'}."
            ),
        },
        "current_mode": next_mode,
        "latest_decision": decision,
        "latest_ppa_frontier": latest_frontier,
        "decision_policy": {
            "area_cap_mm2": AREA_CAP_MM2,
            "frequency_floor_mhz": FREQ_FLOOR_MHZ,
            "target_authority": "operator_only",
        },
    }
    public["public_claims"] = [
        {"claim": f"current Manager-owned stage is {manager_stage}", "evidence": ["research/PIPELINE_STATE.json", "research/PUBLIC_STATUS.json"]},
        {
            "claim": (
                "implementation-supported prefix completes all 18 ordered layer_0 operators through layer_0.mlp_residual_add after independent Reviewer acceptance"
                if published
                else "implementation-supported prefix remains through layer_0.mlp_down_proj; the mlp_residual_add candidate meets numeric targets but awaits independent Reviewer acceptance"
                if meets_acceptance
                else "implementation-supported prefix remains through layer_0.mlp_down_proj; first unsupported layer/operator is layer_0.mlp_residual_add"
            ),
            "evidence": ["design/RTL_MANIFEST.json", "design/PPA_FRONTIER_LEDGER.json", "research/PUBLIC_STATUS.json"],
        },
        {
            "claim": (
                "the bounded RTL regression covers the accepted prefix plus a distinct full-shape post-MLP residual oracle with descriptor, memory-fault, tag-fault, backpressure, address, and saturation checks"
                if meets_acceptance
                else "the bounded RTL regression covers the mlp_residual_add candidate, but support is not claimed unless canonical SKY130 PPA meets 100 MHz and the 2.0 mm2 non-SRAM cap"
            ),
            "evidence": ["evidence/frontier/latest/rtl_mlp_residual_sim.log", "evidence/frontier/latest/rtl_shell_sim.log"],
        },
        {
            "claim": (
                "the independently accepted mlp_residual_add implementation meets the 100 MHz SKY130 floor and 2.0 mm2 hierarchical total non-SRAM cap"
                if published
                else "the mlp_residual_add candidate meets the 100 MHz SKY130 floor and 2.0 mm2 hierarchical total non-SRAM cap but remains unpublished pending independent review"
                if meets_acceptance
                else f"the canonical SKY130 synth/OpenSTA bound for the mlp_residual_add candidate misses the {'100 MHz floor' if not timing_met else '2.0 mm2 non-SRAM cap'} and does not advance the supported prefix"
            ),
            "evidence": ["evidence/frontier/latest/sky130_yosys.log", "evidence/frontier/latest/sky130_sta.log", "design/PPA_FRONTIER_LEDGER.json"],
        },
    ]
    explicit_non_claims = [
        "no routed Open-PDK GDS has been claimed for this RTL slice",
        "no FPGA prototype, full benchmark, signoff, tapeout readiness, or fabricated silicon result has been claimed",
        (
            "the complete ordered layer_0 prefix is published only after independent Reviewer acceptance"
            if published
            else "no mlp_residual_add support is published pending independent Reviewer acceptance"
            if meets_acceptance
            else "no MLP support is claimed beyond the independently accepted mlp_down_proj prefix"
        ),
        "the complete Qwen2.5-0.5B W4A8 workload remains incomplete after this operator slice",
        "the PPA evidence is SKY130 mapped synthesis plus OpenSTA only; it is not routed, signoff, tapeout, or silicon evidence",
    ]
    if comparison["limitation"]:
        explicit_non_claims.append(comparison["limitation"])
    public["explicit_non_claims"] = explicit_non_claims
    write_json(PUBLIC_STATUS, public)

    print(json.dumps({"manifest": str(MANIFEST), "ledger": str(LEDGER), "public_status": str(PUBLIC_STATUS), "entry": entry}, indent=2))


if __name__ == "__main__":
    main()
