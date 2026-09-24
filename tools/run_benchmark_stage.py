#!/usr/bin/env python3
"""Bind ACE-2 current-stage benchmark evidence."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from audit_benchmark_contract import run_audit


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_STATE = ROOT / "research" / "PIPELINE_STATE.json"
PUBLIC_STATUS = ROOT / "research" / "PUBLIC_STATUS.json"
LIVE_VIEW = ROOT / ".argus" / "live-view.json"
TARGET = ROOT / "design" / "TARGET.json"
CHIP_SCOPE = ROOT / "design" / "CHIP_SCOPE.json"
WORKLOAD = ROOT / "design" / "WORKLOAD.md"
BASELINE_PLAN = ROOT / "design" / "BASELINE_PLAN.md"
MEMORY_MODEL = ROOT / "design" / "MEMORY_MODEL.json"
RTL_MANIFEST = ROOT / "design" / "RTL_MANIFEST.json"
PPA_RESULTS = ROOT / "ppa" / "RESULTS.json"
PROTOTYPE_RESULTS = ROOT / "prototype" / "RESULTS.json"
VERIFICATION_RESULTS = ROOT / "verification" / "RESULTS.json"
BENCHMARK = ROOT / "benchmark"
RAW_DIR = BENCHMARK / "raw" / "latest"
RAW_BINDING = RAW_DIR / "benchmark_binding.json"
PROTOCOL = BENCHMARK / "PROTOCOL.md"
RESULTS = BENCHMARK / "RESULTS.json"
TRACE_AUDIT = RAW_DIR / "trace_capability_audit.json"
QUALITY_AUDIT = RAW_DIR / "quality_gate_preflight.json"
GEMMINI_AUDIT = RAW_DIR / "gemmini_subset_manifest.json"
MAKEFILE_AUDIT = RAW_DIR / "makefile_provenance.json"
PACKET_AUDIT = RAW_DIR / "packet_integrity_audit.json"
CONTRACT_AUDIT = RAW_DIR / "benchmark_contract_audit.json"

CLOCK_HZ = 100_000_000
CLOCK_MHZ = 100.0
STREAM_BYTES_PER_CYCLE = 16
MAC_LANES = 32
KV_RECORD_BYTES_PER_LAYER = 272
TRANSFORMER_LAYERS = 24
STAGE_ORDER = [
    "definition",
    "architecture",
    "environment",
    "rtl",
    "verification",
    "ppa",
    "prototype",
    "benchmark",
    "signoff",
]
TRACKED_ARTIFACTS = [
    "MISSION.md",
    "Makefile",
    "research/PIPELINE_STATE.json",
    "research/PUBLIC_STATUS.json",
    "design/BASELINE_PLAN.md",
    "design/CHIP_SCOPE.json",
    "design/MEMORY_MODEL.json",
    "design/PPA_FRONTIER_LEDGER.json",
    "design/RTL_MANIFEST.json",
    "design/TARGET.json",
    "design/WORKLOAD.md",
    "verification/RESULTS.json",
    "ppa/PROTOCOL.md",
    "ppa/RESULTS.json",
    "prototype/RESULTS.json",
    "benchmark/PROTOCOL.md",
    "benchmark/RESULTS.json",
    "benchmark/raw/latest/benchmark_binding.json",
    "benchmark/raw/latest/benchmark_contract_audit.json",
    "benchmark/raw/latest/gemmini_subset_manifest.json",
    "benchmark/raw/latest/makefile_provenance.json",
    "benchmark/raw/latest/packet_integrity_audit.json",
    "benchmark/raw/latest/quality_gate_preflight.json",
    "benchmark/raw/latest/quality_results.json",
    "benchmark/raw/latest/trace_capability_audit.json",
    "benchmark/baselines/gemmini-transposer/build.sbt",
    "benchmark/baselines/gemmini-transposer/project/build.properties",
    "benchmark/baselines/gemmini-transposer/src/main/scala/gemmini/Util.scala",
    "tools/audit_benchmark_contract.py",
    "tools/run_gemmini_subset.py",
    "tools/run_quality_gate.py",
    "tools/run_benchmark_stage.py",
]
KERNEL_SCENARIOS = [
    "qproj_stride_shell",
    "mlp_up_shell",
    "mlp_down_shell",
    "silu_gate_shell",
    "mlp_residual_shell",
    "layer_sweep_shell",
    "final_rmsnorm_shell",
    "lm_head_shell",
]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_json_sha256(data: Any) -> str:
    payload = json.dumps(data, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def write_self_hashed_json(path: Path, data: dict[str, Any]) -> None:
    data["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": (
            "UTF-8, sorted keys, two-space indentation, trailing newline, with "
            "integrity.canonical_sha256 set to null"
        ),
        "canonical_sha256": None,
    }
    data["integrity"]["canonical_sha256"] = canonical_json_sha256(data)
    write_json(path, data)


def artifact(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return {
        "path": rel,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def tree_hash(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(paths):
        item = artifact(rel)
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def stage_lists(pipeline: dict[str, Any]) -> tuple[list[str], list[str]]:
    current_stage = str(pipeline.get("current_stage", "unknown"))
    completed = [
        stage
        for stage in STAGE_ORDER
        if stage != current_stage and pipeline.get("stages", {}).get(stage, {}).get("status") == "done"
    ]
    index = STAGE_ORDER.index(current_stage) if current_stage in STAGE_ORDER else -1
    downstream = STAGE_ORDER[index + 1 :] if index >= 0 else []
    return completed, downstream


def require_current_stage(pipeline: dict[str, Any]) -> None:
    if pipeline.get("current_stage") != "benchmark":
        raise RuntimeError("run_benchmark_stage.py must only bind evidence while current_stage is benchmark")


def require_prior_evidence(verification: dict[str, Any], ppa: dict[str, Any], prototype: dict[str, Any]) -> None:
    if verification.get("stage") != "verification" or not all(verification.get("checklist", {}).values()):
        raise RuntimeError("benchmark binder requires passed verification evidence")
    if ppa.get("stage") != "ppa" or not all(ppa.get("checklist", {}).values()):
        raise RuntimeError("benchmark binder requires passed PPA evidence")
    if prototype.get("stage") != "prototype" or not all(prototype.get("checklist", {}).values()):
        raise RuntimeError("benchmark binder requires passed prototype evidence")


def require_targets(ppa: dict[str, Any], target: dict[str, Any]) -> None:
    targets = target.get("frozen_numeric_targets", {})
    area_cap = float(targets.get("area_cap_non_sram_mm2", 2.0))
    freq_floor = float(targets.get("frequency_floor_mhz", 100.0))
    frontier = ppa.get("frontier", {})
    area = float(frontier.get("non_sram_area_mm2"))
    fmax = float(frontier.get("fmax_mhz"))
    if area > area_cap:
        raise RuntimeError(f"benchmark binder requires accepted PPA area <= {area_cap} mm^2; got {area}")
    if fmax < freq_floor:
        raise RuntimeError(f"benchmark binder requires accepted PPA fmax >= {freq_floor} MHz; got {fmax}")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_from_cycles(cycles: int, power_w: float, work_items: int | None = None) -> dict[str, Any]:
    seconds = cycles / CLOCK_HZ
    metric: dict[str, Any] = {
        "latency_cycles": cycles,
        "latency_seconds_at_100mhz": seconds,
        "latency_us_at_100mhz": seconds * 1_000_000.0,
        "energy_j": power_w * seconds,
        "energy_uj": power_w * seconds * 1_000_000.0,
    }
    if work_items is not None and work_items > 0:
        metric["throughput_items_per_second_at_100mhz"] = work_items / seconds
    return metric


def build_kernel_results(verification: dict[str, Any], ppa: dict[str, Any]) -> list[dict[str, Any]]:
    power_w = safe_float(ppa.get("power", {}).get("total_w"), safe_float(ppa.get("frontier", {}).get("power", {}).get("total_mw")) / 1000.0)
    scenarios = verification.get("scenario_summaries", {})
    results: list[dict[str, Any]] = []
    for scenario_id in KERNEL_SCENARIOS:
        scenario = scenarios.get(scenario_id, {})
        metrics = dict(scenario.get("metrics", {}))
        cycles = metrics.get("cycles")
        if not isinstance(cycles, int) or cycles <= 0:
            continue
        cases = metrics.get("cases") if isinstance(metrics.get("cases"), int) else None
        writes = metrics.get("writes") if isinstance(metrics.get("writes"), int) else None
        observed_payload_bytes = writes * STREAM_BYTES_PER_CYCLE if writes is not None else None
        bandwidth_bpc = (observed_payload_bytes / cycles) if observed_payload_bytes is not None else None
        kernel = {
            "id": scenario_id,
            "result_type": "cycle_accurate_rtl_simulation_kernel_or_integration_scenario",
            "source": scenario.get("artifact"),
            "source_sha256": scenario.get("sha256"),
            "pass_token": scenario.get("pass_token"),
            "latency_throughput_energy": metric_from_cycles(cycles, power_w, cases),
            "effective_bandwidth": {
                "observed_payload_bytes": observed_payload_bytes,
                "observed_payload_bytes_per_cycle": bandwidth_bpc,
                "observed_payload_utilization_of_128b_stream": (
                    bandwidth_bpc / STREAM_BYTES_PER_CYCLE if bandwidth_bpc is not None else None
                ),
                "basis": "payload bytes visible in the RTL log; not a full external-memory trace",
            },
            "utilization": {
                "mac_lanes": MAC_LANES,
                "dot_mac_utilization": None,
                "reason_dot_mac_utilization_unreported": (
                    "kernel logs expose command cycles and write counts, but not full per-cycle MAC-active counters"
                ),
                "stream_utilization": bandwidth_bpc / STREAM_BYTES_PER_CYCLE if bandwidth_bpc is not None else None,
            },
            "area_resources": full_area_resources(ppa),
            "uncertainty": [
                "Icarus RTL simulation with deterministic vectors; not a routed netlist, FPGA, or silicon timing run",
                "energy uses the current SKY130 mapped-netlist lm-head activity power as a constant-power estimate",
                "payload bandwidth is derived from visible write-count fields where available",
            ],
            "workload_specific_metrics": metrics,
        }
        results.append(kernel)
    return results


def full_area_resources(ppa: dict[str, Any]) -> dict[str, Any]:
    frontier = ppa.get("frontier", {})
    memory = ppa.get("memory", {})
    return {
        "resource_scope": "full ace2_shell mapped SKY130 standard-cell design, not per-kernel floorplanning",
        "cells": frontier.get("cells"),
        "non_sram_area_mm2": frontier.get("non_sram_area_mm2"),
        "area_cap_mm2": 2.0,
        "remaining_area_reserve_mm2": frontier.get("remaining_area_reserve_mm2"),
        "logical_sram_banks": memory.get("logical_banks", 8),
        "logical_sram_capacity_bytes": memory.get("logical_capacity_bytes", 524288),
        "sram_macro_area_included": False,
        "external_stream_width_bits": memory.get("external_memory_interface_bits", 128),
    }


def derive_cycles_at_bandwidth(row: dict[str, Any], bandwidth_bpc: int) -> int:
    cycles_at_1 = int(row["cycles_at_1_byte_per_cycle"])
    cycles_at_16 = int(row["cycles_at_16_bytes_per_cycle"])
    if bandwidth_bpc == 1:
        return cycles_at_1
    if bandwidth_bpc == 16:
        return cycles_at_16
    return max(cycles_at_16, math.ceil(cycles_at_1 / bandwidth_bpc))


def build_prefill_results(memory_model: dict[str, Any], ppa: dict[str, Any]) -> list[dict[str, Any]]:
    power_w = safe_float(ppa.get("power", {}).get("total_w"), safe_float(ppa.get("frontier", {}).get("power", {}).get("total_mw")) / 1000.0)
    decode_weight_bytes = int(memory_model["projection_macs"]["decode_linear_total"]["w4_weight_bytes"])
    results: list[dict[str, Any]] = []
    for row in memory_model.get("prefill_estimates", []):
        prompt_tokens = int(row["prompt_tokens"])
        modeled_bytes = int(row["weight_stream_passes"]) * decode_weight_bytes + int(row["kv_write_bytes"])
        for bandwidth_bpc in memory_model["architecture_a0"]["bandwidth_sweep_bytes_per_cycle"]:
            cycles = derive_cycles_at_bandwidth(row, int(bandwidth_bpc))
            seconds = cycles / CLOCK_HZ
            macs = int(row["linear_macs"]) + int(row["attention_macs"])
            results.append({
                "trace_class": "prefill",
                "result_type": "architecture_model_bound_estimate_bound_to_current_ppa",
                "prompt_tokens": prompt_tokens,
                "effective_bandwidth_budget_bytes_per_cycle": int(bandwidth_bpc),
                "cycles": cycles,
                "latency_seconds_at_100mhz": seconds,
                "tokens_per_second_at_100mhz": prompt_tokens / seconds,
                "modeled_external_bytes": modeled_bytes,
                "effective_bandwidth_bytes_per_cycle": modeled_bytes / cycles,
                "effective_bandwidth_utilization": (modeled_bytes / cycles) / int(bandwidth_bpc),
                "energy_j": power_w * seconds,
                "energy_per_token_j": (power_w * seconds) / prompt_tokens,
                "dot_mac_utilization": min(1.0, macs / (cycles * MAC_LANES)),
                "workload_specific_metrics": {
                    "linear_macs": int(row["linear_macs"]),
                    "attention_macs": int(row["attention_macs"]),
                    "kv_write_bytes": int(row["kv_write_bytes"]),
                    "weight_stream_passes": int(row["weight_stream_passes"]),
                    "dominant_limit": row.get("dominant_limit_at_16_bytes_per_cycle"),
                    "combined_arithmetic_intensity_mac_per_byte": row.get("combined_arithmetic_intensity_mac_per_byte"),
                },
                "uncertainty": [
                    "Derived from design/MEMORY_MODEL.json roofline-style cycle estimates, not a full prefill RTL trace",
                    "2/4/8 B/cycle points use max(cycles_at_16Bpc, ceil(cycles_at_1Bpc / bandwidth)) interpolation",
                    "Energy uses current mapped SKY130 lm-head activity power as a constant-power estimate",
                ],
            })
    return results


def build_decode_results(memory_model: dict[str, Any], ppa: dict[str, Any]) -> list[dict[str, Any]]:
    power_w = safe_float(ppa.get("power", {}).get("total_w"), safe_float(ppa.get("frontier", {}).get("power", {}).get("total_mw")) / 1000.0)
    decode_linear = memory_model["projection_macs"]["decode_linear_total"]
    linear_macs = int(decode_linear["macs"])
    linear_weight_bytes = int(decode_linear["w4_weight_bytes"])
    kv_write_bytes = TRANSFORMER_LAYERS * KV_RECORD_BYTES_PER_LAYER
    generated_runs = [1, 16, 128]
    results: list[dict[str, Any]] = []
    for row in memory_model.get("decode_context_estimates", []):
        context_tokens = int(row["context_tokens"])
        for bandwidth_bpc in memory_model["architecture_a0"]["bandwidth_sweep_bytes_per_cycle"]:
            per_token_cycles = derive_cycles_at_bandwidth(row, int(bandwidth_bpc))
            modeled_external_bytes = linear_weight_bytes + int(row["attention_kv_bytes"])
            for generated_tokens in generated_runs:
                cycles = per_token_cycles * generated_tokens
                seconds = cycles / CLOCK_HZ
                macs = (linear_macs + int(row["attention_macs"])) * generated_tokens
                bytes_for_run = modeled_external_bytes * generated_tokens
                results.append({
                    "trace_class": "decode",
                    "result_type": "architecture_model_bound_estimate_bound_to_current_ppa",
                    "context_tokens": context_tokens,
                    "generated_tokens": generated_tokens,
                    "context_growth_model": "constant_context_per_token_projection; no measured autoregressive context growth trace",
                    "effective_bandwidth_budget_bytes_per_cycle": int(bandwidth_bpc),
                    "cycles": cycles,
                    "cycles_per_generated_token": per_token_cycles,
                    "latency_seconds_at_100mhz": seconds,
                    "tokens_per_second_at_100mhz": generated_tokens / seconds,
                    "modeled_external_bytes": bytes_for_run,
                    "bytes_per_generated_token": bytes_for_run / generated_tokens,
                    "effective_bandwidth_bytes_per_cycle": bytes_for_run / cycles,
                    "effective_bandwidth_utilization": (bytes_for_run / cycles) / int(bandwidth_bpc),
                    "energy_j": power_w * seconds,
                    "energy_per_token_j": (power_w * seconds) / generated_tokens,
                    "dot_mac_utilization": min(1.0, macs / (cycles * MAC_LANES)),
                    "workload_specific_metrics": {
                        "linear_macs_per_token": linear_macs,
                        "attention_macs_per_token": int(row["attention_macs"]),
                        "linear_weight_bytes_per_token": linear_weight_bytes,
                        "attention_kv_bytes_per_token": int(row["attention_kv_bytes"]),
                        "kv_write_bytes_per_token": kv_write_bytes,
                        "dominant_limit": row.get("dominant_limit_at_16_bytes_per_cycle"),
                        "combined_arithmetic_intensity_mac_per_byte": row.get("combined_arithmetic_intensity_mac_per_byte"),
                    },
                    "uncertainty": [
                        "Derived from design/MEMORY_MODEL.json roofline-style cycle estimates, not a full decode RTL trace",
                        "Generated-token runs multiply the per-token context estimate and do not model context growth between tokens",
                        "2/4/8 B/cycle points use max(cycles_at_16Bpc, ceil(cycles_at_1Bpc / bandwidth)) interpolation",
                        "Energy uses current mapped SKY130 lm-head activity power as a constant-power estimate",
                    ],
                })
    return results


def summarize_system(prefill: list[dict[str, Any]], decode: list[dict[str, Any]]) -> dict[str, Any]:
    prefill_16 = [row for row in prefill if row["effective_bandwidth_budget_bytes_per_cycle"] == 16]
    decode_16 = [
        row
        for row in decode
        if row["effective_bandwidth_budget_bytes_per_cycle"] == 16 and row["generated_tokens"] == 1
    ]
    return {
        "prefill_tokens_per_second_at_16Bpc": {
            str(row["prompt_tokens"]): row["tokens_per_second_at_100mhz"] for row in prefill_16
        },
        "decode_tokens_per_second_at_16Bpc_single_token": {
            str(row["context_tokens"]): row["tokens_per_second_at_100mhz"] for row in decode_16
        },
        "lowest_decode_tokens_per_second_at_16Bpc": min(
            (row["tokens_per_second_at_100mhz"] for row in decode_16),
            default=None,
        ),
        "lowest_prefill_tokens_per_second_at_16Bpc": min(
            (row["tokens_per_second_at_100mhz"] for row in prefill_16),
            default=None,
        ),
    }


def probe_baseline_tools() -> dict[str, Any]:
    java = shutil.which("java")
    sbt = shutil.which("sbt")
    mill = shutil.which("mill")
    return {
        "java": {"path": java, "available": java is not None},
        "sbt": {"path": sbt, "available": sbt is not None},
        "mill": {"path": mill, "available": mill is not None},
        "gemmini_execution_ready": java is not None and sbt is not None,
        "claim_boundary": (
            "Host-tool probe only. The separately hash-bound Gemmini transposer subset "
            "uses a pinned SBT Docker image and is not a matched Qwen/W4A8 comparison."
        ),
    }


def build_baseline_scope(tool_probe: dict[str, Any], gemmini_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "fair_open_accelerator_baselines": {
            "gemmini": {
                "status": gemmini_audit["status"],
                "required_by": "design/TARGET.json public_comparison_contract.required_baselines",
                "repository": gemmini_audit["repository"],
                "revision": gemmini_audit["revision"],
                "executed_test": gemmini_audit["test"],
                "results": gemmini_audit["results"],
                "compatibility": gemmini_audit["compatibility"],
                "raw_manifest": "benchmark/raw/latest/gemmini_subset_manifest.json",
                "blocker": (
                    "Executed transposer subset is not the frozen Qwen2.5-0.5B W4A8 "
                    "workload and has no matched memory, host, PPA, or quality boundary."
                ),
                "comparison_claim": "excluded_from_win_loss_tables",
            },
            "vta": {
                "status": "not_executed_optional",
                "comparison_claim": "excluded_from_win_loss_tables",
                "reason": "optional baseline; no workload adapter or same-boundary result exists in this worktree",
            },
            "nvdla": {
                "status": "not_executed_optional",
                "comparison_claim": "excluded_from_win_loss_tables",
                "reason": "optional baseline; transformer/SFU workload gaps are not extrapolated",
            },
        },
        "same_flow_internal_context": {
            "status": "reported_as_internal_frontier_context_only",
            "comparison": "PPA ledger delta against the prior final_rmsnorm frontier is not an accelerator baseline win/loss claim",
        },
        "commercial_market_context": {
            "status": "not_used",
            "reason": "different process nodes, products, and published TOPS/W figures are not direct PPA or benchmark evidence",
        },
        "direct_win_loss_claims": [],
    }


def write_raw_binding(
    generated_at: str,
    pipeline: dict[str, Any],
    benchmark_inputs: dict[str, Any],
    tool_probe: dict[str, Any],
) -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw = {
        "generated_at_utc": generated_at,
        "command": "make benchmark-stage",
        "script": "tools/run_benchmark_stage.py",
        "manager_owned_current_stage": pipeline.get("current_stage"),
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "baseline_tool_probe": tool_probe,
        "binding_inputs": benchmark_inputs,
        "claim_boundary": [
            "Candidate kernel metrics are parsed from existing cycle-accurate RTL simulation logs.",
            "End-to-end prefill/decode metrics are model-bound estimates from design/MEMORY_MODEL.json, not full hardware measurements.",
            "The frozen full-shape trace gate is blocked; BF16/W4A8 quality is not passed or claimed by this benchmark binder.",
            "A pinned Gemmini transposer subset was executed but is explicitly incompatible and excluded from win/loss comparison.",
            "No FPGA, GDS, signoff, tapeout, or silicon benchmark claim is made.",
        ],
    }
    write_json(RAW_BINDING, raw)
    return artifact("benchmark/raw/latest/benchmark_binding.json")


def write_protocol(results: dict[str, Any]) -> None:
    raw_hash = results["raw_artifacts"]["benchmark_binding"]["sha256"]
    first_unsupported = results["candidate"]["first_unsupported_layer_operator"]
    first_unsupported_text = "none" if first_unsupported is None else str(first_unsupported)
    quality_status = results["quality"]["status"]
    quality_ready = quality_status == "ready_for_explicit_official_measurement"
    quality_blocked = quality_status.startswith("blocked_")
    makefile_status = results["contract_audit"].get("makefile_provenance_status")
    if makefile_status == "pass_exact_reconstruction":
        makefile_gate = (
            "| Makefile provenance | **Pass** | Exact byte-for-byte PPA/prototype "
            "reconstruction proves later deltas add only stage binders; "
            "`benchmark/raw/latest/makefile_provenance.json` |"
        )
    else:
        makefile_gate = (
            "| Makefile provenance | **Blocked** | Current Makefile no longer "
            "reconstructs to the immutable PPA/prototype packet hashes; "
            "`benchmark/raw/latest/makefile_provenance.json` |"
        )
    if quality_ready:
        quality_intro = (
            "the BF16/W4A8 quality preflight has no active blocker, but the full "
            "official quality measurement has not been run"
        )
        quality_gate = (
            "| BF16/W4A8 quality | **Ready, official measurement not executed** | "
            "Preflight has no active quality blocker; no official perplexity or lm-eval "
            "pass is claimed; `benchmark/raw/latest/quality_gate_preflight.json` and "
            "`benchmark/raw/latest/quality_results.json` |"
        )
    elif quality_blocked:
        quality_intro = (
            "the BF16/W4A8 quality gate is blocked by the current preflight status"
        )
        quality_gate = (
            "| BF16/W4A8 quality | **Blocked, not executed officially** | "
            "Quality preflight is blocked; official evaluation remains prohibited until "
            "the blocker is resolved; `benchmark/raw/latest/quality_gate_preflight.json` "
            "and `benchmark/raw/latest/quality_results.json` |"
        )
    else:
        quality_intro = "the BF16/W4A8 quality status is not an official pass"
        quality_gate = (
            "| BF16/W4A8 quality | **Not passed** | Current quality status is "
            f"`{quality_status}`; no official quality pass is claimed; "
            "`benchmark/raw/latest/quality_gate_preflight.json` and "
            "`benchmark/raw/latest/quality_results.json` |"
        )
    protocol = f"""# ACE-2 current-stage benchmark protocol

Generated: `{results["generated_at_utc"]}`

This protocol binds the current `benchmark` stage to the accepted ACE-2
pre-tapeout candidate evidence. The stage remains blocked: cycle-accurate RTL
kernel scenarios and architecture estimates do not satisfy the frozen full-shape
trace requirement, and {quality_intro}. No fair open-baseline win/loss, quality
pass, FPGA, GDS, signoff, tapeout, or silicon claim is made.

## Frozen candidate and workload

| Item | Pinned value |
| --- | --- |
| Current stage source | `research/PIPELINE_STATE.json` (`current_stage=benchmark`; Manager-owned) |
| Candidate top | `ace2_shell` |
| Supported prefix | {results["candidate"]["ordered_supported_prefix_length"]} ordered layer/operators through `lm_head` |
| First unsupported layer/operator | `{first_unsupported_text}` |
| Model boundary | Qwen2.5-0.5B, batch 1 causal decoder inference |
| Quantization | W4A8 signed int4 weights, signed int8 activations, int32 accumulators, fixed-point requantization |
| Clock for reported rates | 100 MHz |
| Primary technology/resource scope | SKY130 HD mapped standard-cell PPA, no routed parasitics |
| Host boundary | Tokenization, prompt formatting, sampling, detokenization, application I/O, and external allocation remain host-owned |

## Fairness rules

All candidate and future baseline rows must use the same frozen workload,
quantization/quality contract, abstract 128-bit memory interface, 1/2/4/8/16
bytes-per-cycle bandwidth sweep, 100 MHz reporting point, host-offload boundary,
and evidence-retention rules. A row is excluded from win/loss comparison unless
its raw logs, source/config hashes, and unsupported operators are recorded.

The pinned Gemmini `TransposerUnitTest` subset executed two 8-bit tests
successfully. It is explicitly incompatible with the frozen model,
quantization, memory, host, technology, and measurement boundaries, so it is
engineering context only and excluded from win/loss comparison. VTA and NVDLA
remain optional unexecuted baselines. Commercial market data is not used as
direct evidence.

## Frozen gate status

| Gate | Status | Decisive evidence |
| --- | --- | --- |
| Full-shape cycle-accurate traces | **Blocked** | RTL rejects attention-score, softmax, and attention-value descriptors above context 8 and has no cross-tile exact-softmax merge command/state; `benchmark/raw/latest/trace_capability_audit.json` |
{quality_gate}
| Required Gemmini evidence | **Executed incompatible subset** | Two pinned 8-bit transposer tests passed; not a fair comparison; `benchmark/raw/latest/gemmini_subset_manifest.json` |
{makefile_gate}

## Measurement and estimate methods

| Scope | Method | Synchronization / repetitions | Power and energy |
| --- | --- | --- | --- |
| Kernel scenarios | Existing Icarus cycle-accurate RTL logs parsed from `verification/RESULTS.json` | Command acceptance to completion acceptance where logged; deterministic vector counts in each scenario; no statistical warmup | Energy = cycles / 100 MHz * current SKY130 mapped power |
| End-to-end prefill/decode | `design/MEMORY_MODEL.json` roofline-style model bound to the current PPA area/power | Deterministic calculation for the frozen prompt/context sets; no measured runtime repetitions | Same constant-power estimate; explicitly not full workload activity power |
| Gemmini subset | Pinned upstream Chisel transposer tests through Treadle | Two tests, one run; source/config/log hashes retained | No power or energy comparison |

## Raw evidence

| Artifact | SHA-256 |
| --- | --- |
| `benchmark/raw/latest/benchmark_binding.json` | `{raw_hash}` |
| `verification/RESULTS.json` | `{artifact("verification/RESULTS.json")["sha256"]}` |
| `ppa/RESULTS.json` | `{artifact("ppa/RESULTS.json")["sha256"]}` |
| `prototype/RESULTS.json` | `{artifact("prototype/RESULTS.json")["sha256"]}` |
| `design/MEMORY_MODEL.json` | `{artifact("design/MEMORY_MODEL.json")["sha256"]}` |
| `benchmark/raw/latest/trace_capability_audit.json` | `{artifact("benchmark/raw/latest/trace_capability_audit.json")["sha256"]}` |
| `benchmark/raw/latest/quality_gate_preflight.json` | `{artifact("benchmark/raw/latest/quality_gate_preflight.json")["sha256"]}` |
| `benchmark/raw/latest/quality_results.json` | `{artifact("benchmark/raw/latest/quality_results.json")["sha256"]}` |
| `benchmark/raw/latest/gemmini_subset_manifest.json` | `{artifact("benchmark/raw/latest/gemmini_subset_manifest.json")["sha256"]}` |
| `benchmark/raw/latest/makefile_provenance.json` | `{artifact("benchmark/raw/latest/makefile_provenance.json")["sha256"]}` |
| `benchmark/raw/latest/packet_integrity_audit.json` | `{artifact("benchmark/raw/latest/packet_integrity_audit.json")["sha256"]}` |

## Claim boundary

The benchmark packet reports kernel measurements, architecture estimates, one
incompatible Gemmini subset, and explicit blockers only. Architecture estimates
are not relabeled as cycle-accurate traces. A quality pass, fair open-accelerator
win/loss, FPGA measurement, routed GDS/signoff, tapeout readiness, and
fabricated-silicon measurement are not claimed.
"""
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(protocol, encoding="utf-8")


def update_public_status(results: dict[str, Any], pipeline: dict[str, Any]) -> None:
    public = load_json(PUBLIC_STATUS, {})
    completed, downstream = stage_lists(pipeline)
    dashboard = dict(public.get("dashboard_fields", {}))
    frontier = results["candidate"]["ppa_frontier"]
    latest_ppa_frontier = dict(dashboard.get("latest_ppa_frontier", {}))
    latest_ppa_frontier.update({
        "status": dashboard.get("latest_ppa_frontier_status", "fresh_sky130_synth_sta_power_bound_pending_ppa_review"),
        "cells": frontier.get("cells"),
        "non_sram_area_mm2": frontier.get("non_sram_area_mm2"),
        "fmax_mhz": frontier.get("fmax_mhz"),
        "rtl_hash": frontier.get("rtl_hash"),
        "constraint_hash": frontier.get("constraint_hash"),
        "remaining_area_reserve_mm2": frontier.get("remaining_area_reserve_mm2"),
        "benchmark_stage_binding": "blocked_trace_and_quality_contracts_unmet",
    })
    latest_benchmark_stage = {
        "status": results["status"],
        "checklist": results["checklist"],
        "results": "benchmark/RESULTS.json",
        "protocol": "benchmark/PROTOCOL.md",
        "raw_binding": "benchmark/raw/latest/benchmark_binding.json",
        "comparison_scope": results["comparison_scope"],
        "summary": results["system_summary"],
    }
    dashboard.update({
        "current_stage": "benchmark",
        "current_mode": frontier.get("mode", dashboard.get("current_mode")),
        "latest_decision": results["decision"],
        "latest_ppa_frontier": latest_ppa_frontier,
        "latest_benchmark_stage": latest_benchmark_stage,
        "ordered_supported_layer_operator_prefix": frontier.get(
            "ordered_supported_layer_operator_prefix",
            dashboard.get("ordered_supported_layer_operator_prefix", []),
        ),
        "first_unsupported_layer_operator": frontier.get("first_unsupported_layer_operator"),
    })

    public["schema_version"] = 1
    public["project"] = "ACE-2"
    public["vertical"] = "chip_design"
    public["generated_at_utc"] = results["generated_at_utc"]
    public["last_updated_utc"] = results["generated_at_utc"]
    public["stage"] = {
        "current_stage": "benchmark",
        "completed_prior_stages": completed,
        "current_stage_source": "research/PIPELINE_STATE.json",
        "stage_transition_owner": "Manager",
        "planner_may_advance_stage": False,
        "downstream_locked_until_manager_advance": downstream,
        "current_stage_checklist": results["checklist"],
        "current_stage_evidence": [
            "benchmark/PROTOCOL.md",
            "benchmark/RESULTS.json",
            "benchmark/raw/latest/benchmark_binding.json",
        ],
    }
    public["dashboard_fields"] = dashboard
    public["implementation_frontier"] = {
        "tracking_order_definition": "design/WORKLOAD.md#ordered-layer/operator-frontier",
        "ordered_supported_layer_operator_prefix": frontier.get("ordered_supported_layer_operator_prefix", []),
        "first_unsupported_layer_operator": frontier.get("first_unsupported_layer_operator"),
        "current_mode": frontier.get("mode", "ADVANCE"),
        "latest_decision": results["decision"],
        "latest_ppa_frontier": latest_ppa_frontier,
        "decision_policy": {
            "area_cap_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "target_authority": "operator_only",
        },
    }

    trace_status = results["contract_audit"].get("trace_status", "")
    trace_hash_mismatch = "rtl_hash_mismatch" in trace_status
    quality_status = results["quality"]["status"]
    rope_quality_blocker = "rope_scale_range" in quality_status
    quality_ready = quality_status == "ready_for_explicit_official_measurement"
    makefile_status = results["contract_audit"].get("makefile_provenance_status")

    public["public_claims"] = [
        {
            "claim": "current Manager-owned stage is benchmark",
            "evidence": [
                "research/PIPELINE_STATE.json",
                "research/PUBLIC_STATUS.json",
                "benchmark/RESULTS.json",
            ],
        },
        {
            "claim": (
                "benchmark closure is blocked because the frozen RTL accepts attention "
                "contexts only through eight tokens and no exact cross-tile softmax merge "
                "path exists for the required full-shape traces"
            ),
            "evidence": [
                "benchmark/PROTOCOL.md",
                "benchmark/RESULTS.json",
                "benchmark/raw/latest/trace_capability_audit.json",
                "rtl/ace2_shell.sv",
            ],
        },
        {
            "claim": (
                "the pinned Gemmini 8-bit transposer subset executed two tests successfully "
                "but is incompatible with the frozen Qwen2.5-0.5B W4A8 comparison boundary "
                "and is excluded from win/loss claims"
            ),
            "evidence": [
                "benchmark/RESULTS.json",
                "benchmark/raw/latest/gemmini_subset_manifest.json",
                "design/BASELINE_PLAN.md",
            ],
        },
        {
            "claim": (
                (
                    "the benchmark packet records that current Makefile provenance no longer "
                    "reconstructs exactly to the immutable PPA/prototype packet hashes"
                )
                if makefile_status != "pass_exact_reconstruction"
                else (
                    "the accepted 434-item RTL frontier and canonical PPA packet remain "
                    "unchanged; exact Makefile reconstruction proves later deltas are stage "
                    "orchestration only"
                )
            ),
            "evidence": [
                "design/RTL_MANIFEST.json",
                "ppa/RESULTS.json",
                "benchmark/raw/latest/makefile_provenance.json",
                "benchmark/raw/latest/packet_integrity_audit.json",
            ],
        },
    ]

    public["explicit_non_claims"] = [
        "no full-shape cycle-accurate prefill or decode trace is claimed",
        "no BF16-vs-W4A8 quality, perplexity, or lm-eval pass is claimed",
        "no fair Gemmini, VTA, NVDLA, commercial, or market win/loss is claimed",
        "no FPGA, routed GDS, signoff, tapeout-readiness, or fabricated-silicon benchmark measurement is claimed",
    ]

    public["artifact_hashes"] = [
        artifact(rel)
        for rel in sorted(TRACKED_ARTIFACTS)
        if rel != "research/PUBLIC_STATUS.json"
    ]
    public["blockers"] = [
        {
            "id": (
                "rtl_hash_mismatch_and_full_shape_trace_unmet"
                if trace_hash_mismatch
                else "full_shape_trace_blocked_by_rtl_context_limit"
            ),
            "evidence": "benchmark/raw/latest/trace_capability_audit.json",
            "resolution": (
                (
                    "Resolve the RTL/PPA manifest hash disagreement, then implement exact "
                    "long-context attention composition and rerun canonical PPA before tracing."
                )
                if trace_hash_mismatch
                else (
                    "Implement exact long-context attention composition in RTL/architecture, "
                    "verify it, and rerun canonical PPA before tracing."
                )
            ),
        },
        {
            "id": (
                "quality_gate_blocked_by_rope_scale_range"
                if rope_quality_blocker
                else (
                    "official_quality_measurement_not_run"
                    if quality_ready
                    else "quality_gate_not_executable"
                )
            ),
            "evidence": "benchmark/raw/latest/quality_gate_preflight.json",
            "resolution": (
                (
                    "Open a bounded RoPE/attention scale-range repair or no-go mission; "
                    "do not run official quality measurement while the blocker is active."
                )
                if rope_quality_blocker
                else (
                    (
                        "Official BF16/W4A8 quality remains unmeasured; run the full quality "
                        "gate only when an authorized complete-workload/model release milestone "
                        "requires it."
                    )
                    if quality_ready
                    else (
                        "Freeze scale derivation and prompts, implement the full-model fixed-point "
                        "reference, then run the BF16/W4A8 gate."
                    )
                )
            ),
        },
    ]
    public["privacy_policy"] = {
        "contains_credentials": False,
        "contains_private_paths": False,
        "contains_private_pdk_contents": False,
        "contains_prompts": False,
        "contains_raw_private_logs": False,
        "public_safe": True,
    }
    write_self_hashed_json(PUBLIC_STATUS, public)


def update_live_view() -> None:
    write_json(
        LIVE_VIEW,
        {
            "version": 1,
            "title": "Benchmark evidence bound",
            "paths": [
                "benchmark/PROTOCOL.md",
                "benchmark/RESULTS.json",
                "benchmark/raw/latest/benchmark_binding.json",
                "verification/RESULTS.json",
                "ppa/RESULTS.json",
                "research/PUBLIC_STATUS.json",
            ],
            "reason": (
                "Shows the current benchmark-stage candidate metrics, fairness protocol, "
                "baseline exclusions, and claim boundaries."
            ),
        },
    )


def main() -> None:
    if sys.argv[1:]:
        raise RuntimeError("usage: run_benchmark_stage.py")

    audit_summary = run_audit()
    pipeline = load_json(PIPELINE_STATE, {})
    target = load_json(TARGET, {})
    chip_scope = load_json(CHIP_SCOPE, {})
    memory_model = load_json(MEMORY_MODEL, {})
    verification = load_json(VERIFICATION_RESULTS, {})
    ppa = load_json(PPA_RESULTS, {})
    prototype = load_json(PROTOTYPE_RESULTS, {})
    rtl_manifest = load_json(RTL_MANIFEST, {})
    trace_audit = load_json(TRACE_AUDIT, {})
    quality_audit = load_json(QUALITY_AUDIT, {})
    gemmini_audit = load_json(GEMMINI_AUDIT, {})

    require_current_stage(pipeline)
    require_prior_evidence(verification, ppa, prototype)
    require_targets(ppa, target)

    generated_at = utc_now()
    tool_probe = probe_baseline_tools()
    frontier = dict(ppa.get("frontier", {}))
    benchmark_inputs = {
        "input_artifacts": [
            artifact(rel)
            for rel in [
                "research/PIPELINE_STATE.json",
                "design/TARGET.json",
                "design/CHIP_SCOPE.json",
                "design/WORKLOAD.md",
                "design/BASELINE_PLAN.md",
                "design/MEMORY_MODEL.json",
                "design/RTL_MANIFEST.json",
                "verification/RESULTS.json",
                "ppa/RESULTS.json",
                "prototype/RESULTS.json",
                "benchmark/raw/latest/benchmark_contract_audit.json",
                "benchmark/raw/latest/gemmini_subset_manifest.json",
                "benchmark/raw/latest/makefile_provenance.json",
                "benchmark/raw/latest/packet_integrity_audit.json",
                "benchmark/raw/latest/quality_gate_preflight.json",
                "benchmark/raw/latest/quality_results.json",
                "benchmark/raw/latest/trace_capability_audit.json",
                "benchmark/baselines/gemmini-transposer/build.sbt",
                "benchmark/baselines/gemmini-transposer/project/build.properties",
                "benchmark/baselines/gemmini-transposer/src/main/scala/gemmini/Util.scala",
                "tools/audit_benchmark_contract.py",
                "tools/run_gemmini_subset.py",
                "tools/run_quality_gate.py",
                "tools/run_benchmark_stage.py",
                "Makefile",
            ]
        ],
        "workload_model": chip_scope.get("target_workload", {}).get("model", {}),
        "delivery_contract": target.get("delivery_contract", {}),
        "ppa_frontier": {
            key: frontier.get(key)
            for key in [
                "id",
                "rtl_hash",
                "constraint_hash",
                "cells",
                "non_sram_area_mm2",
                "fmax_mhz",
                "mode",
                "decision",
                "first_unsupported_layer_operator",
            ]
        },
    }
    raw_binding = write_raw_binding(generated_at, pipeline, benchmark_inputs, tool_probe)

    kernel_results = build_kernel_results(verification, ppa)
    prefill_results = build_prefill_results(memory_model, ppa)
    decode_results = build_decode_results(memory_model, ppa)
    comparison_scope = build_baseline_scope(tool_probe, gemmini_audit)
    system_summary = summarize_system(prefill_results, decode_results)

    results = {
        "schema_version": 1,
        "project": "ACE-2",
        "stage": "benchmark",
        "generated_at_utc": generated_at,
        "status": "blocked_at_benchmark_contract_unmet",
        "stage_closing": False,
        "decision": "hold_benchmark_full_shape_trace_unmet_and_quality_not_officially_measured",
        "manager_stage_transition_owner": "Manager; this tool does not edit research/PIPELINE_STATE.json.",
        "checklist": {
            "benchmark.protocol-fairness": True,
            "benchmark.kernel-system-metrics": False,
            "benchmark.claim-boundary": True,
        },
        "contract_audit": audit_summary,
        "candidate": {
            "name": "ace2_shell_complete_supported_prefix_through_lm_head",
            "top_module": "ace2_shell",
            "delivery_level": target.get("delivery_contract", {}).get("delivery_level"),
            "ordered_supported_prefix_length": len(frontier.get("ordered_supported_layer_operator_prefix", [])),
            "first_unsupported_layer_operator": frontier.get("first_unsupported_layer_operator"),
            "rtl_hash": frontier.get("rtl_hash"),
            "constraint_hash": frontier.get("constraint_hash"),
            "ppa_frontier": frontier,
            "rtl_manifest_candidate_status": rtl_manifest.get("candidate_status"),
        },
        "fairness_protocol": {
            "workload": {
                "model": "Qwen2.5-0.5B",
                "batch_size": 1,
                "mode": "causal decoder inference",
                "prefill_prompt_lengths": [16, 128, 512, 2048, 8192, 32768],
                "decode_context_lengths": [1, 128, 1024, 4096, 8192, 32768],
                "decode_generated_token_runs": [1, 16, 128],
            },
            "quantization_and_quality": {
                "candidate_numeric_contract": "W4A8 fixed-point contract from design/WORKLOAD.md and design/CHIP_SCOPE.json",
                "baseline_requirement": "same W4A8 contract or explicitly marked incompatible",
                "quality_gate_status": quality_audit["status"],
            },
            "memory_host_budget": {
                "external_memory_stream_bits": 128,
                "bandwidth_sweep_bytes_per_cycle": memory_model["architecture_a0"]["bandwidth_sweep_bytes_per_cycle"],
                "host_runtime_boundary": "tokenization/sampling/detokenization/application I/O excluded from accelerator metrics",
            },
            "warmup_repetitions_synchronization": {
                "kernel_rtl": "deterministic simulation vectors; command-accept to completion-accept boundaries where logged",
                "system_estimates": "deterministic calculation from MEMORY_MODEL; no runtime warmup or repetitions",
                "baselines": "must match candidate prompt/context sets, synchronization boundary, and evidence retention before comparison",
            },
            "power_method": {
                "candidate": "current SKY130 mapped OpenSTA activity power from ppa/RESULTS.json applied to cycle counts",
                "baselines": "must report matched activity/corner/voltage or be excluded from power/energy win-loss comparison",
            },
            "platform_technology_constraints": {
                "candidate": "SKY130 HD mapped standard-cell PPA at 100 MHz; no routed parasitics",
                "baselines": "must state RTL/PPA/FPGA/software claim type and technology; market data excluded",
            },
        },
        "kernel_results": kernel_results,
        "end_to_end_results": {
            "result_type": "model_bound_estimates_not_full_hardware_measurements",
            "prefill": prefill_results,
            "decode": decode_results,
            "area_resources": full_area_resources(ppa),
            "system_summary": system_summary,
        },
        "system_summary": system_summary,
        "comparison_scope": comparison_scope,
        "quality": {
            "bf16_reference_measured": False,
            "w4a8_quality_measured": False,
            "perplexity_delta": None,
            "lm_eval_delta_percentage_points": None,
            "status": quality_audit["status"],
            "gate_passed": False,
            "preflight": quality_audit,
        },
        "claim_boundary": [
            "Kernel metrics are cycle-accurate RTL simulation results for deterministic supported scenarios, not routed timing, FPGA, or silicon measurements.",
            "End-to-end prefill/decode metrics are model-bound estimates from design/MEMORY_MODEL.json bound to current PPA area/power, not full trace RTL measurements.",
            "The frozen full-shape trace gate is blocked by the RTL attention context limit and no cycle-accurate full-shape result is claimed.",
            "The executed Gemmini transposer subset is incompatible with the frozen workload and excluded from fair win/loss comparison.",
            (
                "The BF16/W4A8 quality preflight has no active blocker, but the official full quality gate has not been run and no perplexity or lm-eval delta is claimed."
                if quality_audit["status"] == "ready_for_explicit_official_measurement"
                else "The BF16/W4A8 quality gate is not passed and no perplexity or lm-eval delta is claimed."
            ),
            "No commercial market comparison, routed GDS, signoff, tapeout readiness, or fabricated silicon claim is made.",
        ],
        "limitations": [
            {
                "id": "full_shape_trace_blocked_by_rtl_context_limit",
                "impact": "The frozen performance and full-workload correctness gate is unmet.",
                "evidence": trace_audit,
            },
            {
                "id": "full_quality_gate_not_run",
                "impact": "Quality target remains unmet; no relaxation is implied.",
                "evidence": quality_audit,
            },
            {
                "id": "gemmini_only_incompatible_subset_executed",
                "impact": "No direct accelerator win/loss claim against Gemmini, VTA, or NVDLA.",
                "evidence": gemmini_audit,
            },
            {
                "id": "end_to_end_estimates_not_full_trace_measurements",
                "impact": "Prefill/decode estimate rows do not satisfy the frozen trace gate.",
            },
        ],
        "raw_artifacts": {
            "benchmark_binding": raw_binding,
            "verification_results": artifact("verification/RESULTS.json"),
            "ppa_results": artifact("ppa/RESULTS.json"),
            "prototype_results": artifact("prototype/RESULTS.json"),
            "memory_model": artifact("design/MEMORY_MODEL.json"),
            "contract_audit": artifact("benchmark/raw/latest/benchmark_contract_audit.json"),
            "trace_capability_audit": artifact("benchmark/raw/latest/trace_capability_audit.json"),
            "quality_gate_preflight": artifact("benchmark/raw/latest/quality_gate_preflight.json"),
            "quality_gate_result": artifact("benchmark/raw/latest/quality_results.json"),
            "gemmini_subset_manifest": artifact("benchmark/raw/latest/gemmini_subset_manifest.json"),
            "makefile_provenance": artifact("benchmark/raw/latest/makefile_provenance.json"),
            "packet_integrity_audit": artifact("benchmark/raw/latest/packet_integrity_audit.json"),
        },
        "source_hashes": {
            "benchmark_binding_hash": tree_hash(
                [
                    "research/PIPELINE_STATE.json",
                    "design/TARGET.json",
                    "design/CHIP_SCOPE.json",
                    "design/WORKLOAD.md",
                    "design/BASELINE_PLAN.md",
                    "design/MEMORY_MODEL.json",
                    "design/RTL_MANIFEST.json",
                    "verification/RESULTS.json",
                    "ppa/RESULTS.json",
                    "prototype/RESULTS.json",
                    "benchmark/raw/latest/benchmark_contract_audit.json",
                    "benchmark/raw/latest/gemmini_subset_manifest.json",
                    "benchmark/raw/latest/makefile_provenance.json",
                    "benchmark/raw/latest/packet_integrity_audit.json",
                    "benchmark/raw/latest/quality_gate_preflight.json",
                    "benchmark/raw/latest/quality_results.json",
                    "benchmark/raw/latest/trace_capability_audit.json",
                    "benchmark/raw/latest/benchmark_binding.json",
                    "benchmark/baselines/gemmini-transposer/build.sbt",
                    "benchmark/baselines/gemmini-transposer/project/build.properties",
                    "benchmark/baselines/gemmini-transposer/src/main/scala/gemmini/Util.scala",
                    "tools/audit_benchmark_contract.py",
                    "tools/run_gemmini_subset.py",
                    "tools/run_quality_gate.py",
                    "tools/run_benchmark_stage.py",
                    "Makefile",
                ]
            ),
            "input_artifacts": benchmark_inputs["input_artifacts"],
        },
        "downstream_locked": ["signoff"],
    }

    write_protocol(results)
    results["raw_artifacts"]["protocol"] = artifact("benchmark/PROTOCOL.md")
    write_self_hashed_json(RESULTS, results)
    update_public_status(results, pipeline)
    update_live_view()
    print(
        "ACE2_BENCHMARK_STAGE_BLOCKED "
        f"kernel_rows={len(kernel_results)} "
        f"prefill_rows={len(prefill_results)} "
        f"decode_rows={len(decode_results)} "
        "gemmini_subset_tests=2 "
        "fair_open_win_loss=none"
    )


if __name__ == "__main__":
    main()
