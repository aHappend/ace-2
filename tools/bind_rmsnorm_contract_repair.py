#!/usr/bin/env python3
"""Bind reproducible RMSNorm contract-repair evidence without accepting it."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "rmsnorm_repair" / "latest"
FRONTIER = ROOT / "evidence" / "frontier" / "latest"
QUALITY_BINDING = ROOT / "benchmark" / "quality" / "RTL_BINDING.json"
HISTORICAL_RTL_HASH = "b760fe91116e3e95b4979d0b09835740635f670324099171693fa8b9eac9c0d5"
PREVIOUS_BOUND_RTL_HASH = "f0c39722b96d25d07bc0f29191c055ae72bddaac4741b80f9d2c138b34216b84"
AREA_CAP_MM2 = 2.0
FREQUENCY_FLOOR_MHZ = 100.0

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
CONTRACT_SOURCES = [
    *NUMERICAL_RTL,
    *CONSTRAINTS,
    "tools/ace2_rmsnorm_reference.py",
    "tools/ace2_attention_score_reference.py",
    "tools/ace2_full_model_fixed_point.py",
    "tools/gen_rmsnorm_vectors.py",
    "tools/gen_post_attention_rmsnorm_vectors.py",
    "tools/gen_projection_vectors.py",
    "tools/gen_attention_score_vectors.py",
    "verification/tb/ace2_rmsnorm_tb.sv",
    "verification/tb/ace2_w4a8_proj_tb.sv",
    "verification/tb/ace2_attention_score_tb.sv",
    "verification/tb/ace2_shell_tb.sv",
    "verification/generated/rmsnorm_vectors.json",
    "verification/generated/rmsnorm_vectors.svh",
    "verification/generated/post_attention_rmsnorm_vectors.json",
    "verification/generated/post_attention_rmsnorm_vectors.svh",
    "verification/generated/projection_vectors.json",
    "verification/generated/projection_vectors.svh",
    "verification/generated/attention_score_vectors.json",
    "verification/generated/attention_score_vectors.svh",
    "benchmark/quality/QUALITY_CONFIG.json",
    "design/SPEC.md",
]
LOGS = {
    "fixed_point_self_test": EVIDENCE / "fixed_point_self_test.log",
    "rtl_lint": FRONTIER / "rtl_lint.log",
    "rtl_rmsnorm": FRONTIER / "rtl_core_sim.log",
    "rtl_projection_consumer": FRONTIER / "rtl_proj_sim.log",
    "rtl_attention_score": FRONTIER / "rtl_attention_score_sim.log",
    "rtl_final_rmsnorm": FRONTIER / "rtl_final_rmsnorm_sim.log",
    "rtl_lm_head": FRONTIER / "rtl_lm_head_sim.log",
    "sky130_yosys": FRONTIER / "sky130_yosys.log",
    "sky130_opensta": FRONTIER / "sky130_sta.log",
}
PASS_MARKERS = {
    "fixed_point_self_test": "ACE2_FULL_MODEL_FIXED_POINT_SELF_TEST status=pass",
    "rtl_lint": "ACE2_RTL_LINT_PASS",
    "rtl_rmsnorm": "ACE2_RMSNORM_TB_PASS",
    "rtl_projection_consumer": "ACE2_W4A8_PROJ_TB_PASS",
    "rtl_attention_score": "ACE2_ATTN_SCORE_TB_PASS",
    "rtl_final_rmsnorm": "ACE2_SHELL_FINAL_RMSNORM_TB_PASS",
    "rtl_lm_head": "ACE2_SHELL_LM_HEAD_TB_PASS",
}
SOFTWARE_SOURCES = [
    "tools/ace2_rmsnorm_reference.py",
    "tools/ace2_attention_score_reference.py",
    "tools/ace2_full_model_fixed_point.py",
    "benchmark/quality/QUALITY_CONFIG.json",
]
LOG_INPUTS = {
    "fixed_point_self_test": SOFTWARE_SOURCES,
    "rtl_lint": NUMERICAL_RTL,
    "rtl_rmsnorm": [
        *NUMERICAL_RTL,
        "verification/tb/ace2_rmsnorm_tb.sv",
        "verification/generated/rmsnorm_vectors.svh",
    ],
    "rtl_projection_consumer": [
        *NUMERICAL_RTL,
        "verification/tb/ace2_w4a8_proj_tb.sv",
        "verification/generated/projection_vectors.svh",
    ],
    "rtl_attention_score": [
        *NUMERICAL_RTL,
        "verification/tb/ace2_attention_score_tb.sv",
        "verification/generated/attention_score_vectors.svh",
    ],
    "rtl_final_rmsnorm": [
        *NUMERICAL_RTL,
        "verification/tb/ace2_shell_tb.sv",
        "verification/generated/rmsnorm_vectors.svh",
    ],
    "rtl_lm_head": [
        *NUMERICAL_RTL,
        "verification/tb/ace2_shell_tb.sv",
        "verification/generated/projection_vectors.svh",
    ],
    "sky130_yosys": [*NUMERICAL_RTL, *CONSTRAINTS],
    "sky130_opensta": [*NUMERICAL_RTL, *CONSTRAINTS],
}


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


def source_records(paths: list[str]) -> list[dict[str, str]]:
    return [
        {"path": relative, "sha256": sha256_file(ROOT / relative)}
        for relative in paths
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_area() -> float:
    text = LOGS["sky130_yosys"].read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"Chip area for top module '\\ace2_shell': ([0-9]+\.[0-9]+)",
        text,
    )
    if not matches:
        raise RuntimeError("canonical Yosys log lacks top-module area")
    return float(matches[-1]) / 1_000_000.0


def parse_sta() -> tuple[float, float]:
    text = LOGS["sky130_opensta"].read_text(encoding="utf-8", errors="replace")
    synchronous = re.search(
        r"Path Group: clk_i.*?"
        r"([0-9]+\.[0-9]+)\s+data arrival time.*?"
        r"(-?[0-9]+\.[0-9]+)\s+slack \((MET|VIOLATED)\)",
        text,
        flags=re.DOTALL,
    )
    if not synchronous:
        raise RuntimeError("canonical OpenSTA log lacks synchronous timing")
    arrival_ns = float(synchronous.group(1))
    slack_ns = float(synchronous.group(2))
    if synchronous.group(3) != "MET" or slack_ns < 0:
        raise RuntimeError("canonical OpenSTA timing does not meet the 10 ns constraint")
    return arrival_ns, slack_ns


def require_fresh_logs() -> None:
    for name, path in LOGS.items():
        if not path.is_file():
            raise RuntimeError(f"missing evidence log: {path.relative_to(ROOT)}")
        newest_source = max(
            (ROOT / source).stat().st_mtime_ns for source in LOG_INPUTS[name]
        )
        if path.stat().st_mtime_ns < newest_source:
            raise RuntimeError(f"stale evidence log: {path.relative_to(ROOT)}")
        marker = PASS_MARKERS.get(name)
        if marker and marker not in path.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError(f"{name} log lacks pass marker")


def main() -> None:
    require_fresh_logs()
    area_mm2 = parse_area()
    arrival_ns, slack_ns = parse_sta()
    if area_mm2 > AREA_CAP_MM2:
        raise RuntimeError(f"non-SRAM area {area_mm2} exceeds {AREA_CAP_MM2} mm^2")

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    rtl_hash = tree_hash(NUMERICAL_RTL)
    constraint_hash = tree_hash(CONSTRAINTS)
    contract_hash = tree_hash(CONTRACT_SOURCES)
    evidence_records = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in LOGS.items()
    }
    manifest = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "status": "candidate_pending_independent_l2",
        "historical_accepted_rtl_hash": HISTORICAL_RTL_HASH,
        "previous_bound_rtl_hash": PREVIOUS_BOUND_RTL_HASH,
        "candidate_rtl_hash": rtl_hash,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "contract_source_hash": contract_hash,
        "candidate_source_hashes": source_records(NUMERICAL_RTL),
        "constraint_source_hashes": source_records(CONSTRAINTS),
        "contract_source_hashes": source_records(CONTRACT_SOURCES),
        "evidence": evidence_records,
        "ppa": {
            "flow": "canonical SKY130 Yosys/OpenSTA mapped-netlist flow",
            "rtl_changed_since_previous_bound_ppa": rtl_hash != PREVIOUS_BOUND_RTL_HASH,
            "canonical_ppa_reused_for_exact_unchanged_rtl": rtl_hash == PREVIOUS_BOUND_RTL_HASH,
            "orfs_image": (
                "openroad/orfs@sha256:"
                "3bc303869d5e4caac8f72c854f2b1614c726b2961bbb372f54bc8fbc0e725e71"
            ),
            "non_sram_area_mm2": area_mm2,
            "area_cap_mm2": AREA_CAP_MM2,
            "area_cap_met": area_mm2 <= AREA_CAP_MM2,
            "clock_period_ns": 10.0,
            "frequency_floor_mhz": FREQUENCY_FLOOR_MHZ,
            "frequency_floor_met": slack_ns >= 0,
            "critical_data_arrival_ns": arrival_ns,
            "setup_slack_ns": slack_ns,
        },
    }
    manifest_path = EVIDENCE / "rtl_candidate_manifest.json"
    write_json(manifest_path, manifest)
    binding = {
        "schema_version": 1,
        "status": "candidate_pending_independent_l2",
        "candidate_rtl_hash": rtl_hash,
        "manifest": {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "numerical_rtl": source_records(NUMERICAL_RTL),
    }
    write_json(QUALITY_BINDING, binding)
    write_json(EVIDENCE / "RESULTS.json", manifest)
    print(
        "ACE2_RMSNORM_REPAIR_BIND_PASS "
        f"rtl_hash={rtl_hash} constraint_hash={constraint_hash} "
        f"area_mm2={area_mm2:.7f} setup_slack_ns={slack_ns:.4f} "
        "review_status=pending"
    )


if __name__ == "__main__":
    main()
