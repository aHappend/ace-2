#!/usr/bin/env python3
"""Seal the RMSNorm/downstream scale-contract verification and fresh PPA evidence."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_ppa_stage import parse_critical_path, parse_sta, parse_yosys


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "rmsnorm_scale_contract" / "latest"
FRONTIER = ROOT / "evidence" / "frontier" / "latest"
QUALITY_BINDING = ROOT / "benchmark" / "quality" / "RTL_BINDING.json"
PRIOR_MANIFEST = (
    ROOT / "evidence" / "rmsnorm_repair" / "latest" / "rtl_candidate_manifest.json"
)

AREA_CAP_MM2 = 2.0
FREQUENCY_FLOOR_MHZ = 100.0
CLOCK_PERIOD_NS = 10.0

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
STABLE_REGENERATED_PPA_SOURCES = ["rtl/generated/ace2_silu_lut.svh"]
CONSTRAINT_SOURCES = [
    "constraints/ace2_rmsnorm_core.sdc",
    "flow/yosys/sky130_rmsnorm.ys",
    "flow/yosys/sky130_sta.tcl",
]
CONTRACT_SOURCES = [
    *NUMERICAL_RTL,
    *CONSTRAINT_SOURCES,
    "tools/ace2_rmsnorm_reference.py",
    "tools/ace2_rope_reference.py",
    "tools/ace2_attention_score_reference.py",
    "tools/gen_rmsnorm_vectors.py",
    "tools/gen_rope_vectors.py",
    "tools/gen_attention_score_vectors.py",
    "tools/ace2_full_model_fixed_point.py",
    "verification/tb/ace2_rmsnorm_tb.sv",
    "verification/tb/ace2_rope_tb.sv",
    "verification/tb/ace2_attention_score_tb.sv",
    "verification/tb/ace2_shell_tb.sv",
    "verification/generated/rmsnorm_vectors.json",
    "verification/generated/rmsnorm_vectors.svh",
    "verification/generated/rope_vectors.json",
    "verification/generated/rope_vectors.svh",
    "verification/generated/attention_score_vectors.json",
    "verification/generated/attention_score_vectors.svh",
    "benchmark/quality/QUALITY_CONFIG.json",
    "design/CHIP_SCOPE.json",
    "design/SPEC.md",
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
FIXED_POINT_SOURCES = [
    "tools/ace2_full_model_fixed_point.py",
    "tools/ace2_projection_reference.py",
    "tools/ace2_rmsnorm_reference.py",
    "tools/ace2_rope_reference.py",
    "tools/ace2_attention_score_reference.py",
    "tools/ace2_softmax_reference.py",
    "tools/ace2_attention_value_reference.py",
    "tools/ace2_attention_compose_reference.py",
    "tools/ace2_silu_gate_reference.py",
    "tools/ace2_residual_reference.py",
    "tools/ace2_mlp_residual_reference.py",
    "benchmark/quality/QUALITY_CONFIG.json",
]
VERIFICATION_SOURCES = sorted(
    set(
        [
            *NUMERICAL_RTL,
            *SHELL_VECTOR_INPUTS,
            *FIXED_POINT_SOURCES,
            "Makefile",
            "verification/tb/ace2_rmsnorm_tb.sv",
            "verification/tb/ace2_w4a8_proj_tb.sv",
            "verification/tb/ace2_rope_tb.sv",
            "verification/tb/ace2_attention_score_tb.sv",
            "verification/tb/ace2_softmax_tb.sv",
            "verification/tb/ace2_attention_compose_tb.sv",
            "verification/tb/ace2_silu_gate_tb.sv",
            "verification/tb/ace2_shell_tb.sv",
        ]
    )
)

VERIFICATION_LOGS = {
    "fixed_point_self_test": {
        "path": "evidence/rmsnorm_scale_contract/latest/fixed_point_self_test.log",
        "marker": "ACE2_FULL_MODEL_FIXED_POINT_SELF_TEST status=pass",
        "inputs": FIXED_POINT_SOURCES,
    },
    "rtl_lint": {
        "path": "evidence/frontier/latest/rtl_lint.log",
        "marker": "ACE2_RTL_LINT_PASS",
        "inputs": [*NUMERICAL_RTL, "Makefile"],
    },
    "rtl_rmsnorm": {
        "path": "evidence/frontier/latest/rtl_core_sim.log",
        "marker": "ACE2_RMSNORM_TB_PASS cases=15 beats_per_case=56",
        "inputs": [
            *NUMERICAL_RTL,
            "verification/tb/ace2_rmsnorm_tb.sv",
            "verification/generated/rmsnorm_vectors.svh",
            "Makefile",
        ],
    },
    "rtl_projection": {
        "path": "evidence/frontier/latest/rtl_proj_sim.log",
        "marker": "ACE2_W4A8_PROJ_TB_PASS cases=5",
        "inputs": [
            *NUMERICAL_RTL,
            "verification/tb/ace2_w4a8_proj_tb.sv",
            "verification/generated/projection_vectors.svh",
            "Makefile",
        ],
    },
    "rtl_rope": {
        "path": "evidence/frontier/latest/rtl_rope_sim.log",
        "marker": "ACE2_ROPE_TB_PASS cases=5",
        "inputs": [
            *NUMERICAL_RTL,
            "verification/tb/ace2_rope_tb.sv",
            "verification/generated/rope_vectors.svh",
            "Makefile",
        ],
    },
    "rtl_attention_score": {
        "path": "evidence/frontier/latest/rtl_attention_score_sim.log",
        "marker": "ACE2_ATTN_SCORE_TB_PASS cases=4",
        "inputs": [
            *NUMERICAL_RTL,
            "verification/tb/ace2_attention_score_tb.sv",
            "verification/generated/attention_score_vectors.svh",
            "Makefile",
        ],
    },
    "rtl_softmax": {
        "path": "evidence/frontier/latest/rtl_softmax_sim.log",
        "marker": "ACE2_SOFTMAX_TB_PASS cases=5",
        "inputs": [
            *NUMERICAL_RTL,
            "verification/tb/ace2_softmax_tb.sv",
            "verification/generated/softmax_vectors.svh",
            "Makefile",
        ],
    },
    "rtl_attention_compose": {
        "path": "evidence/frontier/latest/rtl_attention_compose_sim.log",
        "marker": "ACE2_ATTN_COMPOSE_TB_PASS cases=3",
        "inputs": [
            *NUMERICAL_RTL,
            "verification/tb/ace2_attention_compose_tb.sv",
            "Makefile",
        ],
    },
    "rtl_silu_gate": {
        "path": "evidence/frontier/latest/rtl_silu_gate_core_sim.log",
        "marker": "ACE2_SILU_GATE_TB_PASS cases=8",
        "inputs": [
            *NUMERICAL_RTL,
            "verification/tb/ace2_silu_gate_tb.sv",
            "verification/generated/silu_gate_vectors.svh",
            "Makefile",
        ],
    },
    "rtl_attention_score_shell": {
        "path": "evidence/frontier/latest/rtl_attention_score_shell_sim.log",
        "marker": (
            "ACE2_SHELL_ATTN_SCORE_TB_PASS cases=4 unequal_scale_case=1 "
            "invalid_metadata_cases=2"
        ),
        "inputs": [
            *NUMERICAL_RTL,
            "verification/tb/ace2_shell_tb.sv",
            *SHELL_VECTOR_INPUTS,
            "Makefile",
        ],
    },
    "rtl_final_rmsnorm": {
        "path": "evidence/frontier/latest/rtl_final_rmsnorm_sim.log",
        "marker": (
            "ACE2_SHELL_FINAL_RMSNORM_TB_PASS layer=24 cases=2 "
            "rejected_layer=25 rejected_m=2 rejected_k=1 descriptor_errors=3"
        ),
        "inputs": [
            *NUMERICAL_RTL,
            "verification/tb/ace2_shell_tb.sv",
            "verification/generated/rmsnorm_vectors.svh",
            "Makefile",
        ],
    },
    "rtl_lm_head": {
        "path": "evidence/frontier/latest/rtl_lm_head_sim.log",
        "marker": (
            "ACE2_SHELL_LM_HEAD_TB_PASS layer=24 tile_outputs=32 vocab=151936 "
            "tiles=4748 cases=2 rejected_layer=23 rejected_m=2 rejected_n=64 "
            "rejected_k=768 descriptor_errors=4"
        ),
        "inputs": [
            *NUMERICAL_RTL,
            "verification/tb/ace2_shell_tb.sv",
            "verification/generated/projection_vectors.svh",
            "verification/generated/rmsnorm_vectors.svh",
            "Makefile",
        ],
    },
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


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def artifact(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": sha256_file(path),
    }


def source_records(paths: list[str]) -> list[dict[str, Any]]:
    return [artifact(relative) for relative in paths]


def executable_record(command: str) -> dict[str, Any]:
    executable = shutil.which(command)
    if executable is None:
        raise RuntimeError(f"missing executable: {command}")
    resolved = Path(executable).resolve()
    return {
        "command": command,
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def command_version(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.splitlines()[0]


def require_fresh(path: Path, inputs: list[str]) -> None:
    freshness_inputs = [
        relative
        for relative in inputs
        if relative not in STABLE_REGENERATED_PPA_SOURCES
    ]
    newest_input = max(
        (ROOT / relative).stat().st_mtime_ns for relative in freshness_inputs
    )
    if path.stat().st_mtime_ns < newest_input:
        raise RuntimeError(f"stale artifact: {path.relative_to(ROOT)}")


def verification_records() -> dict[str, Any]:
    records = {}
    for name, spec in VERIFICATION_LOGS.items():
        path = ROOT / spec["path"]
        if not path.is_file():
            raise RuntimeError(f"missing verification log: {spec['path']}")
        require_fresh(path, spec["inputs"])
        text = path.read_text(encoding="utf-8", errors="replace")
        if spec["marker"] not in text:
            raise RuntimeError(
                f"{spec['path']} lacks pass marker {spec['marker']!r}"
            )
        records[name] = {
            **artifact(spec["path"]),
            "pass_marker": spec["marker"],
            "input_hash": tree_hash(spec["inputs"]),
        }
    return records


def parse_orfs_image() -> str:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^ORFS_IMAGE \?= (\S+)$", makefile, re.MULTILINE)
    if match is None or "@sha256:" not in match.group(1):
        raise RuntimeError("Makefile lacks a digest-pinned ORFS image")
    return match.group(1)


def parse_setup_slack(text: str) -> float:
    matches = re.findall(
        r"^\s*(-?[0-9]+\.[0-9]+)\s+slack \((?:MET|VIOLATED)\)$",
        text,
        re.MULTILINE,
    )
    if not matches:
        raise RuntimeError("OpenSTA log lacks a setup slack result")
    return float(matches[-1])


def validate_stable_regenerated_sources() -> list[dict[str, Any]]:
    prior = json.loads(PRIOR_MANIFEST.read_text(encoding="utf-8"))
    prior_hashes = {
        record["path"]: record["sha256"]
        for record in prior["candidate_source_hashes"]
    }
    records = []
    for relative in STABLE_REGENERATED_PPA_SOURCES:
        current = artifact(relative)
        if prior_hashes.get(relative) != current["sha256"]:
            raise RuntimeError(
                f"post-PPA regenerated source changed content: {relative}"
            )
        records.append(
            {
                **current,
                "prior_manifest": PRIOR_MANIFEST.relative_to(ROOT).as_posix(),
                "prior_manifest_sha256": sha256_file(PRIOR_MANIFEST),
            }
        )
    return records


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    yosys_path = FRONTIER / "sky130_yosys.log"
    sta_path = FRONTIER / "sky130_sta.log"
    ppa_freshness_inputs = [
        *(
            relative
            for relative in NUMERICAL_RTL
            if relative not in STABLE_REGENERATED_PPA_SOURCES
        ),
        *CONSTRAINT_SOURCES,
        "Makefile",
    ]
    stable_regenerated_sources = validate_stable_regenerated_sources()
    require_fresh(yosys_path, ppa_freshness_inputs)
    require_fresh(sta_path, [*CONSTRAINT_SOURCES, "Makefile", "evidence/frontier/latest/sky130_yosys.log"])

    yosys_text = yosys_path.read_text(encoding="utf-8", errors="replace")
    sta_text = sta_path.read_text(encoding="utf-8", errors="replace")
    yosys_version_match = re.search(r"^Yosys .+$", yosys_text, re.MULTILINE)
    sta_version_match = re.search(r"^OpenSTA .+$", sta_text, re.MULTILINE)
    if yosys_version_match is None or sta_version_match is None:
        raise RuntimeError("PPA logs lack tool version identity")

    yosys = parse_yosys(yosys_path)
    sta = parse_sta(sta_path)
    critical_path = parse_critical_path(sta_path)
    setup_slack_ns = parse_setup_slack(sta_text)
    area_cap_met = yosys["non_sram_area_mm2"] <= AREA_CAP_MM2
    frequency_floor_met = setup_slack_ns >= 0.0
    ppa_thresholds_met = area_cap_met and frequency_floor_met

    verification = verification_records()
    toolchain = {
        "ppa": {
            "orfs_image": parse_orfs_image(),
            "yosys": yosys_version_match.group(0),
            "opensta": sta_version_match.group(0),
            "invocation_source": artifact("Makefile"),
        },
        "verification": {
            "iverilog": {
                **executable_record("iverilog"),
                "version": command_version(["iverilog", "-V"]),
            },
            "vvp": {
                **executable_record("vvp"),
                "version": command_version(["vvp", "-V"]),
            },
            "verilator": {
                **executable_record("verilator"),
                "version": command_version(["verilator", "--version"]),
            },
            "python": {
                **executable_record(sys.executable),
                "version": command_version([sys.executable, "--version"]),
            },
        },
        "binding_tool": artifact("tools/bind_rope_contract_evidence.py"),
    }
    toolchain["sha256"] = canonical_hash(toolchain)

    generated_at = utc_now()
    rtl_hash = tree_hash(NUMERICAL_RTL)
    status = (
        "rmsnorm_downstream_scale_contract_pending_paired_smoke_review"
        if ppa_thresholds_met
        else "rmsnorm_downstream_scale_contract_ppa_threshold_miss"
    )
    manifest = {
        "schema_version": 2,
        "generated_at_utc": generated_at,
        "status": status,
        "candidate_rtl_hash": rtl_hash,
        "rtl_hash": rtl_hash,
        "constraint_hash": tree_hash(CONSTRAINT_SOURCES),
        "contract_source_hash": tree_hash(CONTRACT_SOURCES),
        "verification_source_hash": tree_hash(VERIFICATION_SOURCES),
        "candidate_source_hashes": source_records(NUMERICAL_RTL),
        "constraint_source_hashes": source_records(CONSTRAINT_SOURCES),
        "contract_source_hashes": source_records(CONTRACT_SOURCES),
        "verification_source_hashes": source_records(VERIFICATION_SOURCES),
        "content_stable_regenerated_sources": stable_regenerated_sources,
        "toolchain": toolchain,
        "verification": verification,
        "ppa": {
            "flow": "canonical SKY130 Yosys/OpenSTA mapped-netlist flow",
            "reused_existing_logs": False,
            "rerun_performed_by_binding": False,
            "fresh_canonical_run_performed_before_binding": True,
            "source_hash": rtl_hash,
            "constraint_hash": tree_hash(CONSTRAINT_SOURCES),
            "content_stable_regenerated_sources": stable_regenerated_sources,
            "yosys_log": artifact("evidence/frontier/latest/sky130_yosys.log"),
            "opensta_log": artifact("evidence/frontier/latest/sky130_sta.log"),
            "cells": yosys["cells"],
            "non_sram_area_mm2": yosys["non_sram_area_mm2"],
            "area_cap_mm2": AREA_CAP_MM2,
            "area_cap_met": area_cap_met,
            "clock_period_ns": CLOCK_PERIOD_NS,
            "critical_path": critical_path,
            "setup_slack_ns": setup_slack_ns,
            "frequency_floor_mhz": FREQUENCY_FLOOR_MHZ,
            "frequency_floor_met": frequency_floor_met,
            "thresholds_met": ppa_thresholds_met,
            "sta_wns_ns": sta["wns_ns"],
            "sta_tns_ns": sta["tns_ns"],
        },
        "quality_scope": {
            "paired_smoke_run": False,
            "paired_smoke_permitted": ppa_thresholds_met,
            "paired_smoke_blocked_reason": (
                None
                if ppa_thresholds_met
                else "canonical_sky130_ppa_threshold_miss"
            ),
            "official_14_item_evaluation_run": False,
            "official_14_item_evaluation_prohibited": True,
            "thresholds": {
                "c4_en_512_perplexity_ratio_max": 1.05,
                "wikitext2_perplexity_ratio_max": 1.05,
            },
        },
    }
    manifest_path = EVIDENCE / "RESULTS.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    QUALITY_BINDING.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "ACE2_RMSNORM_SCALE_CONTRACT_EVIDENCE_BIND_PASS "
        f"rtl_hash={rtl_hash} "
        f"area_mm2={yosys['non_sram_area_mm2']:.9f} "
        f"setup_slack_ns={setup_slack_ns:.4f} "
        f"ppa_thresholds_met={str(ppa_thresholds_met).lower()} "
        f"verification_logs={len(verification)}"
    )


if __name__ == "__main__":
    main()
