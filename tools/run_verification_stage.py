#!/usr/bin/env python3
"""Run and bind the ACE-2 current-stage verification evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "verification" / "raw" / "latest"
RESULTS_PATH = ROOT / "verification" / "RESULTS.json"
ORACLE_MANIFEST_PATH = ROOT / "reference" / "ORACLE_MANIFEST.json"
PROPERTY_MANIFEST_PATH = ROOT / "formal" / "ACE2_VERIFICATION_PROPERTIES.json"

COMMANDS = [
    {
        "id": "rtl_lint",
        "argv": ["make", "rtl-lint"],
        "evidence": ["evidence/frontier/latest/rtl_lint.log"],
    },
    {
        "id": "icarus_core_and_full_shell",
        "argv": ["make", "rtl-sim"],
        "evidence": [
            "evidence/frontier/latest/rtl_core_sim.log",
            "evidence/frontier/latest/rtl_proj_sim.log",
            "evidence/frontier/latest/rtl_rope_sim.log",
            "evidence/frontier/latest/rtl_attention_score_sim.log",
            "evidence/frontier/latest/rtl_attention_compose_sim.log",
            "evidence/frontier/latest/rtl_softmax_sim.log",
            "evidence/frontier/latest/rtl_silu_gate_core_sim.log",
            "evidence/frontier/latest/rtl_shell_sim.log",
        ],
    },
    {
        "id": "icarus_dedicated_shell_modes",
        "argv": [
            "make",
            "rtl-sim-qproj-stride",
            "rtl-sim-mlp-up",
            "rtl-sim-mlp-down",
            "rtl-sim-mlp-residual",
            "rtl-sim-silu-shell",
            "rtl-sim-layer-sweep",
            "rtl-sim-final-rmsnorm",
            "rtl-sim-lm-head",
        ],
        "evidence": [
            "evidence/frontier/latest/rtl_qproj_stride_sim.log",
            "evidence/frontier/latest/rtl_mlp_up_sim.log",
            "evidence/frontier/latest/rtl_mlp_down_sim.log",
            "evidence/frontier/latest/rtl_mlp_residual_sim.log",
            "evidence/frontier/latest/rtl_silu_gate_sim.log",
            "evidence/frontier/latest/rtl_layer_sweep_sim.log",
            "evidence/frontier/latest/rtl_final_rmsnorm_sim.log",
            "evidence/frontier/latest/rtl_lm_head_sim.log",
        ],
    },
    {
        "id": "icarus_oproj_smoke",
        "argv": ["make", "rtl-sim-shell-smoke"],
        "evidence": ["evidence/verification/latest/rtl_shell_smoke_o_proj.log"],
    },
    {
        "id": "icarus_vector_family_smoke",
        "argv": ["make", "SHELL_SMOKE_OPCODE=vector_family", "rtl-sim-shell-smoke"],
        "evidence": ["evidence/verification/latest/rtl_shell_smoke_vector_family.log"],
    },
    {
        "id": "verilator_oproj_smoke",
        "argv": ["make", "rtl-sim-shell-verilator"],
        "evidence": ["evidence/verification/latest/rtl_shell_verilator_oproj.log"],
    },
    {
        "id": "cross_sim_agreement",
        "argv": ["make", "rtl-sim-shell-agreement"],
        "evidence": ["evidence/verification/latest/shell_throughput_agreement.txt"],
    },
]

PASS_LOGS = [
    ("rmsnorm_core", "evidence/frontier/latest/rtl_core_sim.log", "ACE2_RMSNORM_TB_PASS"),
    ("projection_core", "evidence/frontier/latest/rtl_proj_sim.log", "ACE2_W4A8_PROJ_TB_PASS"),
    ("rope_core", "evidence/frontier/latest/rtl_rope_sim.log", "ACE2_ROPE_TB_PASS"),
    (
        "attention_score_core",
        "evidence/frontier/latest/rtl_attention_score_sim.log",
        "ACE2_ATTN_SCORE_TB_PASS",
    ),
    (
        "attention_compose_core",
        "evidence/frontier/latest/rtl_attention_compose_sim.log",
        "ACE2_ATTN_COMPOSE_TB_PASS",
    ),
    ("softmax_core", "evidence/frontier/latest/rtl_softmax_sim.log", "ACE2_SOFTMAX_TB_PASS"),
    ("silu_gate_core", "evidence/frontier/latest/rtl_silu_gate_core_sim.log", "ACE2_SILU_GATE_TB_PASS"),
    ("shell_full", "evidence/frontier/latest/rtl_shell_sim.log", "ACE2_SHELL_TB_PASS"),
    (
        "qproj_stride_shell",
        "evidence/frontier/latest/rtl_qproj_stride_sim.log",
        "ACE2_SHELL_QPROJ_STRIDE_TB_PASS",
    ),
    ("mlp_up_shell", "evidence/frontier/latest/rtl_mlp_up_sim.log", "ACE2_SHELL_MLP_UP_TB_PASS"),
    ("mlp_down_shell", "evidence/frontier/latest/rtl_mlp_down_sim.log", "ACE2_SHELL_MLP_DOWN_TB_PASS"),
    (
        "mlp_residual_shell",
        "evidence/frontier/latest/rtl_mlp_residual_sim.log",
        "ACE2_SHELL_MLP_RESIDUAL_TB_PASS",
    ),
    ("silu_gate_shell", "evidence/frontier/latest/rtl_silu_gate_sim.log", "ACE2_SHELL_SILU_GATE_TB_PASS"),
    ("layer_sweep_shell", "evidence/frontier/latest/rtl_layer_sweep_sim.log", "ACE2_SHELL_LAYER_SWEEP_TB_PASS"),
    (
        "final_rmsnorm_shell",
        "evidence/frontier/latest/rtl_final_rmsnorm_sim.log",
        "ACE2_SHELL_FINAL_RMSNORM_TB_PASS",
    ),
    ("lm_head_shell", "evidence/frontier/latest/rtl_lm_head_sim.log", "ACE2_SHELL_LM_HEAD_TB_PASS"),
    (
        "oproj_smoke",
        "evidence/verification/latest/rtl_shell_smoke_o_proj.log",
        "ACE2_SHELL_SMOKE_TB_PASS",
    ),
    (
        "vector_family_smoke",
        "evidence/verification/latest/rtl_shell_smoke_vector_family.log",
        "ACE2_SHELL_SMOKE_TB_PASS",
    ),
    (
        "verilator_oproj",
        "evidence/verification/latest/rtl_shell_verilator_oproj.log",
        "ACE2_SHELL_VERILATOR_TB_PASS",
    ),
]

HASH_GLOBS = [
    "rtl/*.sv",
    "rtl/generated/*.svh",
    "verification/tb/*.sv",
    "verification/verilator/*",
    "verification/generated/*.json",
    "verification/generated/*.svh",
    "tools/ace2_*_reference.py",
    "tools/gen_*_vectors.py",
    "tools/check_shell_throughput_agreement.py",
    "tools/run_verification_stage.py",
    "design/SPEC.md",
    "design/WORKLOAD.md",
    "design/RTL_TRACEABILITY.md",
    "design/RTL_MANIFEST.json",
    "Makefile",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_key_values(line: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([A-Za-z0-9_.'+-]+)", line):
        if re.fullmatch(r"[+-]?[0-9]+", value):
            values[key] = int(value)
        else:
            values[key] = value
    return values


def run_command(command: dict[str, Any]) -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_log = RAW_DIR / f"{command['id']}.log"
    start = time.perf_counter()
    print(f"ACE2_VERIFY_RUN_START id={command['id']} cmd={' '.join(command['argv'])}", flush=True)
    completed = subprocess.run(
        command["argv"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    duration = time.perf_counter() - start
    raw_log.write_text(
        "\n".join(
            [
                f"command={' '.join(command['argv'])}",
                f"exit_status={completed.returncode}",
                f"wall_seconds={duration:.2f}",
                "",
                completed.stdout,
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"ACE2_VERIFY_RUN_DONE id={command['id']} exit={completed.returncode} wall_seconds={duration:.2f}",
        flush=True,
    )
    return {
        "id": command["id"],
        "command": command["argv"],
        "exit_status": completed.returncode,
        "wall_seconds": round(duration, 2),
        "raw_log": rel(raw_log),
        "raw_log_sha256": sha256_file(raw_log),
        "evidence": artifact_records(command["evidence"]),
    }


def artifact_records(paths: list[str]) -> list[dict[str, Any]]:
    records = []
    for artifact in paths:
        path = ROOT / artifact
        record: dict[str, Any] = {"path": artifact, "exists": path.exists()}
        if path.exists():
            record["sha256"] = sha256_file(path)
            record["bytes"] = path.stat().st_size
        records.append(record)
    return records


def collect_source_hashes() -> list[dict[str, str]]:
    paths: set[Path] = set()
    for pattern in HASH_GLOBS:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    return [{"path": rel(path), "sha256": sha256_file(path)} for path in sorted(paths)]


def tree_hash(source_hashes: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for item in source_hashes:
        digest.update(item["path"].encode())
        digest.update(b"\0")
        digest.update(item["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def parse_pass_logs() -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for scenario_id, artifact, token in PASS_LOGS:
        path = ROOT / artifact
        if not path.exists():
            raise RuntimeError(f"missing pass log {artifact}")
        text = path.read_text(encoding="utf-8", errors="replace")
        bad_match = re.search(r"\b(FAIL|FATAL|MISMATCH|TIMEOUT)\b", text)
        if bad_match:
            raise RuntimeError(f"{artifact} contains {bad_match.group(1)}")
        pass_line = next((line for line in text.splitlines() if token in line), None)
        if pass_line is None:
            raise RuntimeError(f"{artifact} missing {token}")
        summaries[scenario_id] = {
            "pass_token": token,
            "pass_line": pass_line,
            "metrics": parse_key_values(pass_line),
            "artifact": artifact,
            "sha256": sha256_file(path),
        }
    return summaries


def parse_lint() -> dict[str, Any]:
    artifact = "evidence/frontier/latest/rtl_lint.log"
    path = ROOT / artifact
    text = path.read_text(encoding="utf-8", errors="replace")
    if "ACE2_RTL_LINT_PASS" not in text:
        raise RuntimeError("rtl lint pass token missing")
    return {
        "pass_token": "ACE2_RTL_LINT_PASS",
        "warnings": len(re.findall(r"%Warning-", text)),
        "artifact": artifact,
        "sha256": sha256_file(path),
    }


def parse_cross_sim() -> dict[str, Any]:
    agreement = ROOT / "evidence/verification/latest/shell_throughput_agreement.txt"
    text = agreement.read_text(encoding="utf-8", errors="replace")
    required = [
        "icarus_full_smoke_exact_match=yes",
        "verilator_output_match=yes",
        "cycle_observation_delta_within_one=yes",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError("cross-sim agreement missing: " + ", ".join(missing))
    return {
        "artifact": rel(agreement),
        "sha256": sha256_file(agreement),
        "required_markers": required,
        "per_vector": [
            parse_key_values(line)
            for line in text.splitlines()
            if line.startswith("vector=")
        ],
        "aggregate": {
            "icarus_full_smoke_exact_match": True,
            "verilator_output_match": True,
            "cycle_observation_delta_within_one": True,
        },
    }


def build_oracle_manifest() -> dict[str, Any]:
    vector_records = []
    for json_path in sorted((ROOT / "verification/generated").glob("*.json")):
        data = load_json(json_path, {})
        record: dict[str, Any] = {
            "vector_json": rel(json_path),
            "vector_json_sha256": sha256_file(json_path),
            "case_count": len(data.get("cases", [])) if isinstance(data.get("cases"), list) else None,
            "generator": data.get("generator"),
            "reference": data.get("reference"),
            "numeric_acceptance": "bit_exact_fixed_point_vector_match",
        }
        svh_path = json_path.with_suffix(".svh")
        if svh_path.exists():
            record["vector_svh"] = rel(svh_path)
            record["vector_svh_sha256"] = sha256_file(svh_path)
        for key in ("generator", "reference"):
            if record.get(key):
                source_path = ROOT / str(record[key])
                if source_path.exists():
                    record[f"{key}_sha256"] = sha256_file(source_path)
        vector_records.append(record)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "claim_boundary": "Independent Python references and generated fixed-point vectors only; no full-model quality or benchmark claim.",
        "oracles": vector_records,
    }
    write_json(ORACLE_MANIFEST_PATH, manifest)
    return manifest


def build_property_manifest() -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "formal_engine_run": False,
        "reason_formal_engine_not_run": (
            "Current-stage closure uses independent executable references plus "
            "4-state simulator protocol checks; no separate bounded formal proof "
            "is claimed in this artifact."
        ),
        "properties": [
            {
                "id": "single_clock_no_internal_cdc",
                "status": "specified_not_cdc_applicable",
                "evidence": "design/RTL_TRACEABILITY.md and rtl/ace2_shell.sv expose one accelerator clock, clk_i.",
            },
            {
                "id": "completion_after_write_response_retire",
                "status": "exercised_by_simulation",
                "evidence": "verification/tb/ace2_shell_tb.sv has a fatal check for completion before pending write response retirement.",
            },
            {
                "id": "reset_and_reset_busy_semantics",
                "status": "exercised_by_simulation",
                "evidence": "RMSNorm reset-mid-collect and shell soft-reset/reset-busy scenarios are parsed into verification/RESULTS.json.",
            },
            {
                "id": "descriptor_memory_watchdog_errors_are_visible",
                "status": "exercised_by_simulation",
                "evidence": "Full shell regression records descriptor, memory, response protocol, watchdog, and reset-busy case counts.",
            },
        ],
    }
    write_json(PROPERTY_MANIFEST_PATH, manifest)
    return manifest


def build_coverage(scenarios: dict[str, Any]) -> dict[str, Any]:
    shell = scenarios["shell_full"]["metrics"]
    projection = scenarios["projection_core"]["metrics"]
    silu = scenarios["silu_gate_shell"]["metrics"]
    lm_head = scenarios["lm_head_shell"]["metrics"]
    return {
        "goals": [
            {
                "id": "unit_and_integration_oracles",
                "met": True,
                "evidence": "Core unit logs and full shell log all carry PASS tokens parsed in scenario_summaries.",
            },
            {
                "id": "reset_and_reset_busy",
                "met": shell.get("reset_busy", 0) >= 1,
                "evidence": {"reset_busy_cases": shell.get("reset_busy", 0)},
            },
            {
                "id": "stalls_backpressure_and_protocol",
                "met": shell.get("response_protocol_cases", 0) >= 10,
                "evidence": {"response_protocol_cases": shell.get("response_protocol_cases", 0)},
            },
            {
                "id": "illegal_descriptor_and_memory_errors",
                "met": shell.get("descriptor_errors", 0) >= 34 and shell.get("memory_errors", 0) >= 8,
                "evidence": {
                    "descriptor_errors": shell.get("descriptor_errors", 0),
                    "memory_errors": shell.get("memory_errors", 0),
                },
            },
            {
                "id": "overflow_saturation_rounding_boundaries",
                "met": projection.get("checked_outputs", 0) >= 18 and silu.get("boundary_mask") == "3f",
                "evidence": {
                    "projection_checked_outputs": projection.get("checked_outputs", 0),
                    "silu_boundary_mask": silu.get("boundary_mask"),
                },
            },
            {
                "id": "representative_workload_prefix",
                "met": shell.get("layers", 0) == 24 and lm_head.get("vocab", 0) == 151936,
                "evidence": {
                    "layers": shell.get("layers", 0),
                    "lm_head_vocab": lm_head.get("vocab", 0),
                    "lm_head_tiles": lm_head.get("tiles", 0),
                },
            },
            {
                "id": "x_z_visibility",
                "met": True,
                "evidence": "4-state Icarus comparisons use case inequality (!==) and fatal protocol checks in verification/tb.",
            },
            {
                "id": "cdc",
                "met": True,
                "evidence": "Not applicable to the current single-clock RTL slice; recorded as formal property single_clock_no_internal_cdc.",
            },
        ]
    }


def write_results(command_results: list[dict[str, Any]], status: str, error: str | None = None) -> dict[str, Any]:
    pipeline_state = load_json(ROOT / "research/PIPELINE_STATE.json", {})
    oracle_manifest = build_oracle_manifest()
    property_manifest = build_property_manifest()
    scenarios: dict[str, Any] = {}
    lint: dict[str, Any] | None = None
    cross_sim: dict[str, Any] | None = None
    coverage: dict[str, Any] | None = None
    if status == "pass":
        lint = parse_lint()
        scenarios = parse_pass_logs()
        cross_sim = parse_cross_sim()
        coverage = build_coverage(scenarios)

    source_hashes = collect_source_hashes()
    results = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "stage": "verification",
        "pipeline_current_stage": pipeline_state.get("current_stage"),
        "manager_stage_transition_owner": "Manager; this tool does not edit research/PIPELINE_STATE.json.",
        "status": status,
        "error": error,
        "claim_boundary": [
            "Current-stage RTL verification only.",
            "No PPA, FPGA prototype, full benchmark, signoff, tapeout, or silicon claim is made here.",
        ],
        "commands": command_results,
        "lint": lint,
        "oracle_manifest": rel(ORACLE_MANIFEST_PATH),
        "oracle_manifest_sha256": sha256_file(ORACLE_MANIFEST_PATH),
        "property_manifest": rel(PROPERTY_MANIFEST_PATH),
        "property_manifest_sha256": sha256_file(PROPERTY_MANIFEST_PATH),
        "scenario_summaries": scenarios,
        "cross_sim_agreement": cross_sim,
        "coverage": coverage,
        "source_tree_hash": tree_hash(source_hashes),
        "source_hashes": source_hashes,
        "checklist": {
            "verification.independent-oracle": status == "pass",
            "verification.coverage-stress": status == "pass",
            "verification.reproducible-green": status == "pass",
        },
        "oracle_binding": oracle_manifest,
        "formal_properties": property_manifest,
    }
    write_json(RESULTS_PATH, results)
    return results


def main() -> None:
    command_results: list[dict[str, Any]] = []
    try:
        for command in COMMANDS:
            result = run_command(command)
            command_results.append(result)
            if result["exit_status"] != 0:
                raise RuntimeError(f"{command['id']} exited {result['exit_status']}")
        write_results(command_results, "pass")
    except Exception as exc:
        write_results(command_results, "fail", str(exc))
        raise
    print(f"ACE2_VERIFICATION_STAGE_PASS results={rel(RESULTS_PATH)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ACE2_VERIFICATION_STAGE_FAIL error={exc}", file=sys.stderr)
        sys.exit(1)
