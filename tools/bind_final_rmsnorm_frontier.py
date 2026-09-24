#!/usr/bin/env python3
"""Bind final-RMSNorm candidate evidence without publishing before review."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import datetime as datetime_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if not hasattr(datetime_module, "UTC"):
    datetime_module.UTC = timezone.utc

from tools.bind_rtl_frontier import parse_shell_sim, parse_sta, parse_yosys

LATEST = ROOT / "evidence" / "frontier" / "latest"
MANIFEST = ROOT / "design" / "RTL_MANIFEST.json"
LEDGER = ROOT / "design" / "PPA_FRONTIER_LEDGER.json"
PUBLIC_STATUS = ROOT / "research" / "PUBLIC_STATUS.json"
REVIEW_VERDICT = ROOT / "evidence" / "review" / "latest" / "final_rmsnorm_verdict.json"
VERIFICATION_BINDING = LATEST / "final_rmsnorm_verification_binding.json"

AREA_CAP_MM2 = 2.0
FREQ_FLOOR_MHZ = 100.0
CANDIDATE_OPERATOR = "final_rmsnorm"
PUBLISHED_FIRST_UNSUPPORTED = "lm_head"

RTL_FILES = [
    "rtl/ace2_pkg.sv",
    "rtl/ace2_rmsnorm_core.sv",
    "rtl/ace2_w4a8_proj_core.sv",
    "rtl/ace2_rope_core.sv",
    "rtl/ace2_attention_score_core.sv",
    "rtl/ace2_softmax_core.sv",
    "rtl/ace2_silu_gate_core.sv",
    "rtl/generated/ace2_silu_lut.svh",
    "rtl/ace2_shell.sv",
]
CONSTRAINT_FILES = [
    "constraints/ace2_rmsnorm_core.sdc",
    "flow/yosys/sky130_rmsnorm.ys",
    "flow/yosys/sky130_sta.tcl",
]
SHELL_VECTOR_FILES = [
    "verification/generated/rmsnorm_vectors.svh",
    "verification/generated/projection_vectors.svh",
    "verification/generated/rope_vectors.svh",
    "verification/generated/attention_score_vectors.svh",
    "verification/generated/softmax_vectors.svh",
    "verification/generated/attention_value_vectors.svh",
    "verification/generated/residual_vectors.svh",
    "verification/generated/mlp_residual_vectors.svh",
    "verification/generated/post_attention_rmsnorm_vectors.svh",
    "verification/generated/silu_gate_vectors.svh",
]
SOURCE_FILES = RTL_FILES + CONSTRAINT_FILES + [
    "verification/tb/ace2_shell_tb.sv",
    *SHELL_VECTOR_FILES,
    "tools/ace2_rmsnorm_reference.py",
    "tools/gen_rmsnorm_vectors.py",
    "tools/bind_final_rmsnorm_frontier.py",
    "tools/bind_rtl_frontier.py",
    "Makefile",
    "design/RTL_TRACEABILITY.md",
    "design/SPEC.md",
    "design/WORKLOAD.md",
    "design/FAST_LOOP_POLICY.json",
]
PUBLIC_ARTIFACT_FILES = [
    "research/PIPELINE_STATE.json",
    "design/RTL_MANIFEST.json",
    "design/PPA_FRONTIER_LEDGER.json",
    "evidence/frontier/latest/rtl_layer_sweep_sim.log",
    "evidence/frontier/latest/rtl_lint.log",
    "evidence/frontier/latest/rtl_final_rmsnorm_sim.log",
    "evidence/frontier/latest/rtl_shell_sim.log",
    "evidence/frontier/latest/sky130_yosys.log",
    "evidence/frontier/latest/sky130_sta.log",
    "evidence/frontier/latest/final_rmsnorm_verification_binding.json",
    "evidence/review/latest/shared_transformer_layers_verdict.json",
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
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(sha256_file(ROOT / rel).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_fresh_evidence(output: Path, inputs: list[str]) -> None:
    newest_input = max((ROOT / rel).stat().st_mtime_ns for rel in inputs)
    if output.stat().st_mtime_ns < newest_input:
        raise RuntimeError(
            f"{output.relative_to(ROOT)} is stale for the current source/constraints"
        )


def refresh_generated_rtl_provenance(
    provenance: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    refreshed = [dict(item) for item in provenance]
    for item in refreshed:
        if (
            item.get("kind") == "generated_rtl_source"
            and item.get("path") == "rtl/generated/ace2_silu_lut.svh"
        ):
            generator = str(item["generator"])
            reference = str(item["reference"])
            item.update({
                "content_sha256": sha256_file(ROOT / str(item["path"])),
                "generator_sha256": sha256_file(ROOT / generator),
                "reference_sha256": sha256_file(ROOT / reference),
                "frozen_configuration": {
                    "input_format": "signed_q6_9",
                    "output_format": "signed_q3_12",
                    "index_shift": 6,
                    "minimum_index": -64,
                    "maximum_index": 64,
                    "table_entries": 129,
                    "indexing": "arithmetic_floor_with_endpoint_clamp",
                    "quantization": "round_ties_to_even_then_signed_int16_saturate",
                },
            })
            return refreshed
    raise RuntimeError("manifest lacks ace2_silu_lut.svh generated RTL provenance")


def pending_ppa_binding(rtl_hash: str, constraint_hash: str) -> dict[str, Any]:
    return {
        "mode": "pending_fresh_canonical",
        "frontier_id": None,
        "source_frontier_id": None,
        "reuse_basis": None,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
    }


def fresh_ppa_binding(
    frontier_id: str,
    rtl_hash: str,
    constraint_hash: str,
    evidence_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "mode": "fresh",
        "frontier_id": frontier_id,
        "source_frontier_id": None,
        "reuse_basis": None,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "sky130_yosys_log_sha256": evidence_hashes[
            "sky130_yosys_log_sha256"
        ],
        "sky130_sta_log_sha256": evidence_hashes["sky130_sta_log_sha256"],
    }


def parse_lint(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "ACE2_RTL_LINT_PASS" not in text or "ACE2_RTL_LINT_FAIL" in text:
        raise RuntimeError("durable RTL lint evidence does not record a pass")
    return {
        "passed": True,
        "warnings": len(re.findall(r"^%Warning-", text, flags=re.MULTILINE)),
        "log_sha256": sha256_file(log_path),
    }


def parse_critical_path(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    startpoints = re.findall(r"^Startpoint:\s+(.+)$", text, flags=re.MULTILINE)
    endpoints = re.findall(r"^Endpoint:\s+(.+)$", text, flags=re.MULTILINE)
    arrivals = re.findall(
        r"^\s*(-?[0-9]+\.[0-9]+)\s+data arrival time$",
        text,
        flags=re.MULTILINE,
    )
    if len(startpoints) < 2 or len(endpoints) < 2 or not arrivals:
        raise RuntimeError("could not parse synchronous STA critical path")
    return {
        "startpoint": startpoints[-1],
        "endpoint": endpoints[-1],
        "data_arrival_ns": max(float(value) for value in arrivals),
    }


def parse_final_rmsnorm(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"ACE2_SHELL_FINAL_RMSNORM_TB_PASS\s+layer=([0-9]+)\s+"
        r"cases=([0-9]+)\s+rejected_layer=([0-9]+)\s+"
        r"rejected_m=([0-9]+)\s+rejected_k=([0-9]+)\s+"
        r"descriptor_errors=([0-9]+)\s+cycles=([0-9]+)",
        text,
    )
    if not match:
        raise RuntimeError("could not parse final RMSNorm pass summary")
    return {
        "layer_id": int(match.group(1)),
        "cases": int(match.group(2)),
        "rejected_layer_id": int(match.group(3)),
        "rejected_m": int(match.group(4)),
        "rejected_k": int(match.group(5)),
        "descriptor_errors": int(match.group(6)),
        "cycles": int(match.group(7)),
        "log_sha256": sha256_file(log_path),
    }


def collect_verification_evidence() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], bool
]:
    require_fresh_evidence(
        LATEST / "rtl_final_rmsnorm_sim.log",
        RTL_FILES + [
            "verification/tb/ace2_shell_tb.sv",
            "verification/generated/rmsnorm_vectors.svh",
        ],
    )
    require_fresh_evidence(
        LATEST / "rtl_lint.log",
        RTL_FILES + ["Makefile"],
    )
    require_fresh_evidence(
        LATEST / "rtl_shell_sim.log",
        RTL_FILES + ["verification/tb/ace2_shell_tb.sv"] + SHELL_VECTOR_FILES,
    )
    lint = parse_lint(LATEST / "rtl_lint.log")
    final_sim = parse_final_rmsnorm(LATEST / "rtl_final_rmsnorm_sim.log")
    shell_sim = parse_shell_sim(LATEST / "rtl_shell_sim.log")
    verification_met = (
        final_sim["layer_id"] == 24
        and final_sim["cases"] >= 2
        and final_sim["rejected_layer_id"] == 25
        and final_sim["rejected_m"] == 2
        and final_sim["rejected_k"] == 1
        and final_sim["descriptor_errors"] >= 3
        and lint["passed"]
        and shell_sim["layers"] == 24
        and shell_sim["success_runs"] >= 66
        and shell_sim["descriptor_errors"] >= 30
        and shell_sim["memory_errors"] >= 8
        and shell_sim["watchdog_errors"] >= 1
        and shell_sim["reset_busy"] >= 1
        and shell_sim["response_protocol_cases"] >= 10
    )
    return lint, final_sim, shell_sim, verification_met


def write_verification_binding(
    generated_at: str,
    rtl_hash: str,
    constraint_hash: str,
    lint: dict[str, Any],
    final_sim: dict[str, Any],
    shell_sim: dict[str, Any],
    evidence_hashes: dict[str, str],
    *,
    ppa_status: str,
    ppa_evidence_binding: dict[str, Any],
    review_status: str = "pending",
    publication_authorized: bool = False,
    yosys: dict[str, Any] | None = None,
    sta: dict[str, Any] | None = None,
) -> None:
    evidence = {
        "rtl_lint_log": "evidence/frontier/latest/rtl_lint.log",
        "rtl_final_rmsnorm_sim_log": "evidence/frontier/latest/rtl_final_rmsnorm_sim.log",
        "rtl_shell_sim_log": "evidence/frontier/latest/rtl_shell_sim.log",
    }
    binding = {
        "schema_version": 1,
        "project": "ACE-2",
        "generated_at_utc": generated_at,
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "candidate_rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "required_geometry": {"m": 1, "n": 896, "k": 0},
        "verification_complete": True,
        "ppa_status": ppa_status,
        "independent_reviewer_acceptance": review_status,
        "publication_authorized": publication_authorized,
        "ppa_evidence_binding": ppa_evidence_binding,
        "lint": lint,
        "final_rmsnorm_sim": final_sim,
        "shell_sim": shell_sim,
        "evidence": evidence,
        "evidence_hashes": evidence_hashes,
    }
    if yosys is not None and sta is not None:
        binding["sky130_yosys"] = yosys
        binding["sky130_sta"] = sta
        binding["evidence"].update({
            "sky130_yosys_log": "evidence/frontier/latest/sky130_yosys.log",
            "sky130_sta_log": "evidence/frontier/latest/sky130_sta.log",
        })
    write_json(VERIFICATION_BINDING, binding)


def capability_sets(
    previous_manifest: dict[str, Any], published: bool
) -> tuple[list[str], list[str], list[str]]:
    accepted = list(
        previous_manifest.get("supported_reusable_operator_capabilities", [])
    )
    candidate = list(accepted)
    if CANDIDATE_OPERATOR not in candidate:
        candidate.append(CANDIDATE_OPERATOR)
    return accepted, candidate, candidate if published else accepted


def reviewer_status(
    rtl_hash: str, constraint_hash: str, evidence_hashes: dict[str, str]
) -> tuple[str, dict[str, Any] | None]:
    if not REVIEW_VERDICT.exists():
        return "pending", None
    verdict = load_json(REVIEW_VERDICT, None)
    if not isinstance(verdict, dict):
        raise RuntimeError("final RMSNorm verdict must be a JSON object")
    status = verdict.get("status")
    if status not in {"accepted", "rejected"}:
        raise RuntimeError("final RMSNorm verdict status must be accepted or rejected")
    if status == "rejected":
        return "rejected", verdict
    expected = {
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
    }
    stale = [key for key, value in expected.items() if verdict.get(key) != value]
    reviewed_evidence = verdict.get("evidence_hashes")
    if not isinstance(reviewed_evidence, dict):
        raise RuntimeError("accepted final RMSNorm verdict lacks evidence hashes")
    stale.extend(
        key for key, value in evidence_hashes.items()
        if reviewed_evidence.get(key) != value
    )
    if stale:
        raise RuntimeError("accepted final RMSNorm verdict is stale for: " + ", ".join(stale))
    if verdict.get("publication_authorized") is not True:
        raise RuntimeError("accepted final RMSNorm verdict must authorize publication")
    return "accepted", verdict


def bind_verification_only() -> None:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    previous_manifest = load_json(MANIFEST, {})
    accepted_prefix = list(previous_manifest.get("supported_layer_operator_prefix", []))
    if previous_manifest.get("first_unsupported_layer_operator") != CANDIDATE_OPERATOR:
        raise RuntimeError("accepted frontier must end immediately before final_rmsnorm")
    if not accepted_prefix or accepted_prefix[-1] != "layer_23.mlp_residual_add":
        raise RuntimeError("accepted transformer-layer prefix is incomplete")

    lint, final_sim, shell_sim, verification_met = collect_verification_evidence()
    if not verification_met:
        raise RuntimeError("final RMSNorm verification evidence misses acceptance coverage")

    rtl_hash = tree_hash(RTL_FILES)
    constraint_hash = tree_hash(CONSTRAINT_FILES)
    ppa_binding = pending_ppa_binding(rtl_hash, constraint_hash)
    accepted_capabilities, candidate_capabilities, _ = capability_sets(
        previous_manifest, False
    )
    evidence_hashes = {
        "rtl_lint_log_sha256": lint["log_sha256"],
        "rtl_final_rmsnorm_sim_log_sha256": final_sim["log_sha256"],
        "rtl_shell_sim_log_sha256": shell_sim["log_sha256"],
    }
    write_verification_binding(
        generated_at,
        rtl_hash,
        constraint_hash,
        lint,
        final_sim,
        shell_sim,
        evidence_hashes,
        ppa_status="pending_fresh_canonical_sky130_yosys_opensta",
        ppa_evidence_binding=ppa_binding,
    )

    manifest = dict(previous_manifest)
    manifest.update({
        "generated_at_utc": generated_at,
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "candidate_status": "verification_bound_fresh_sky130_ppa_pending",
        "candidate_meets_numeric_acceptance": None,
        "candidate_verification_complete": True,
        "candidate_requires_fresh_sky130_ppa": True,
        "candidate_supported_layer_operator_prefix_after_review": (
            accepted_prefix + [CANDIDATE_OPERATOR]
        ),
        "candidate_first_unsupported_layer_operator_after_review": (
            PUBLISHED_FIRST_UNSUPPORTED
        ),
        "candidate_geometry": {
            "m": 1,
            "n": 896,
            "k": 0,
            "lanes": 16,
            "beats": 56,
        },
        "independent_reviewer_acceptance": "pending",
        "independent_reviewer_verdict": None,
        "candidate_rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "candidate_source_hashes": [
            {"path": rel, "sha256": sha256_file(ROOT / rel)}
            for rel in SOURCE_FILES
        ],
        "supported_reusable_operator_capabilities": accepted_capabilities,
        "candidate_reusable_operator_capabilities_after_review": (
            candidate_capabilities
        ),
        "candidate_evidence_hashes": evidence_hashes,
        "candidate_verification_binding": (
            "evidence/frontier/latest/final_rmsnorm_verification_binding.json"
        ),
        "ppa_evidence_binding": ppa_binding,
        "ip_provenance": refresh_generated_rtl_provenance(
            list(previous_manifest.get("ip_provenance", []))
        ),
        "latest_evidence": {
            **dict(previous_manifest.get("latest_evidence", {})),
            "lint": lint,
            "rtl_lint_log": "evidence/frontier/latest/rtl_lint.log",
            "final_rmsnorm_sim": final_sim,
            "rtl_final_rmsnorm_sim_log": "evidence/frontier/latest/rtl_final_rmsnorm_sim.log",
            "shell_sim": shell_sim,
            "rtl_shell_sim_log": "evidence/frontier/latest/rtl_shell_sim.log",
            "final_rmsnorm_verification_binding": (
                "evidence/frontier/latest/final_rmsnorm_verification_binding.json"
            ),
            "ppa_evidence_binding": ppa_binding,
        },
        "layer_execution_coverage": {
            **dict(previous_manifest.get("layer_execution_coverage", {})),
            "basis": "verification/tb/ace2_shell_tb.sv complete command-dispatch regression plus +FINAL_RMSNORM_ONLY layer-24 and shape-boundary coverage",
            "final_rmsnorm_layer_ids": [],
            "final_rmsnorm_candidate_layer_ids": [24],
            "final_rmsnorm_negative_descriptor_coverage": {
                "layer_ids_rejected": [final_sim["rejected_layer_id"]],
                "m_values_rejected": [final_sim["rejected_m"]],
                "k_values_rejected": [final_sim["rejected_k"]],
                "cases": final_sim["descriptor_errors"],
            },
        },
        "claim_boundaries": [
            "The accepted prefix remains through layer_23.mlp_residual_add; final_rmsnorm is not published.",
            "Fresh durable lint, complete-shell, and dedicated final-RMSNorm simulations verify m == 1, n == 896, k == 0 at reserved layer_id 24 and reject layer_id 25, m == 2, and k == 1.",
            "Fresh canonical SKY130 Yosys/OpenSTA evidence and independent Reviewer acceptance remain mandatory before publication.",
        ],
    })
    write_json(MANIFEST, manifest)
    print(
        f"bound {CANDIDATE_OPERATOR} verification: "
        f"rtl_hash={rtl_hash} ppa=pending review=pending"
    )


def main() -> None:
    if sys.argv[1:] == ["--verification-only"]:
        bind_verification_only()
        return
    if sys.argv[1:]:
        raise RuntimeError("usage: bind_final_rmsnorm_frontier.py [--verification-only]")

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    previous_manifest = load_json(MANIFEST, {})
    accepted_prefix = list(previous_manifest.get("supported_layer_operator_prefix", []))
    if previous_manifest.get("first_unsupported_layer_operator") != CANDIDATE_OPERATOR:
        raise RuntimeError("accepted frontier must end immediately before final_rmsnorm")
    if not accepted_prefix or accepted_prefix[-1] != "layer_23.mlp_residual_add":
        raise RuntimeError("accepted transformer-layer prefix is incomplete")

    lint, final_sim, shell_sim, verification_met = collect_verification_evidence()
    require_fresh_evidence(LATEST / "sky130_yosys.log", RTL_FILES + CONSTRAINT_FILES)
    require_fresh_evidence(LATEST / "sky130_sta.log", RTL_FILES + CONSTRAINT_FILES)

    yosys = parse_yosys(LATEST / "sky130_yosys.log")
    sta = parse_sta(LATEST / "sky130_sta.log")
    rtl_hash = tree_hash(RTL_FILES)
    constraint_hash = tree_hash(CONSTRAINT_FILES)
    evidence_hashes = {
        "rtl_lint_log_sha256": lint["log_sha256"],
        "rtl_final_rmsnorm_sim_log_sha256": final_sim["log_sha256"],
        "rtl_shell_sim_log_sha256": shell_sim["log_sha256"],
        "sky130_yosys_log_sha256": yosys["log_sha256"],
        "sky130_sta_log_sha256": sta["log_sha256"],
    }
    frontier_id = f"frontier-{generated_at}"
    ppa_binding = fresh_ppa_binding(
        frontier_id, rtl_hash, constraint_hash, evidence_hashes
    )
    review_status, review = reviewer_status(rtl_hash, constraint_hash, evidence_hashes)
    timing_met = sta["wns_ns"] >= 0.0 and sta["estimated_fmax_mhz_floor_bound"] >= FREQ_FLOOR_MHZ
    area_met = yosys["non_sram_area_mm2"] <= AREA_CAP_MM2
    numeric_targets_met = timing_met and area_met and verification_met
    published = numeric_targets_met and review_status == "accepted"
    candidate_prefix = accepted_prefix + [CANDIDATE_OPERATOR]
    published_prefix = candidate_prefix if published else accepted_prefix
    first_unsupported = PUBLISHED_FIRST_UNSUPPORTED if published else CANDIDATE_OPERATOR
    accepted_capabilities, candidate_capabilities, published_capabilities = (
        capability_sets(previous_manifest, published)
    )

    ledger = load_json(LEDGER, {"schema_version": 1, "project": "ACE-2", "entries": []})
    baseline = next(
        (
            entry for entry in reversed(ledger.get("entries", []))
            if entry.get("accepted_prefix_advanced") is True
            and entry.get("first_unsupported_layer_operator") == CANDIDATE_OPERATOR
        ),
        None,
    )
    if baseline is None:
        raise RuntimeError("could not find accepted pre-final-RMSNorm baseline")
    baseline_area = float(baseline["non_sram_area_mm2"])
    baseline_cells = int(baseline["cells"])
    delta_area = yosys["non_sram_area_mm2"] - baseline_area
    delta_area_percent = (delta_area / baseline_area) * 100.0
    timing_baseline = next(
        (
            item for item in reversed(ledger.get("entries", []))
            if item.get("candidate_layer_operator") == CANDIDATE_OPERATOR
            and item.get("rtl_hash") != rtl_hash
            and item.get("frequency_floor_met") is False
        ),
        None,
    )
    if timing_baseline is None:
        raise RuntimeError("could not find prior final-RMSNorm timing baseline")
    baseline_fmax = float(timing_baseline["fmax_mhz_floor_bound"])
    timing_improvement_percent = (
        (sta["estimated_fmax_mhz_floor_bound"] - baseline_fmax)
        / baseline_fmax
        * 100.0
    )
    critical_path = parse_critical_path(LATEST / "sky130_sta.log")
    timing_repair_mechanism = (
        "banked shared_payload_q updates with kept per-16-bit state/receive "
        "replicas to split full-payload decode fanout"
    )

    if published:
        status = "fresh_sky130_synth_sta_reviewer_accepted"
        decision = "final_rmsnorm_bound_targets_met_continue_to_lm_head"
        mode = "ADVANCE"
    elif numeric_targets_met and review_status == "pending":
        status = "fresh_sky130_synth_sta_meets_floor_awaiting_independent_reviewer"
        decision = "final_rmsnorm_targets_met_awaiting_independent_reviewer"
        mode = "ADVANCE_REVIEW_PENDING"
    elif numeric_targets_met:
        status = "fresh_sky130_synth_sta_independent_reviewer_rejected"
        decision = "final_rmsnorm_targets_met_independent_reviewer_rejected"
        mode = "ADVANCE_REVIEW_REJECTED"
    else:
        status = "fresh_sky130_synth_sta_candidate_replan"
        decision = "final_rmsnorm_bound_replan"
        mode = "ADVANCE_REPLAN"

    write_verification_binding(
        generated_at,
        rtl_hash,
        constraint_hash,
        lint,
        final_sim,
        shell_sim,
        evidence_hashes,
        ppa_status=status,
        ppa_evidence_binding=ppa_binding,
        review_status=review_status,
        publication_authorized=published,
        yosys=yosys,
        sta=sta,
    )

    cycle_impact = {
        "basis": "two bit-exact full hidden-vector RMSNorm cases at reserved layer_id 24 plus illegal layer_id, m, and k rejection",
        "candidate": CANDIDATE_OPERATOR,
        "candidate_supported_prefix_length": len(candidate_prefix),
        "accepted_supported_prefix_length": len(published_prefix),
        "cases": final_sim["cases"],
        "cycles": final_sim["cycles"],
        "layer_id": final_sim["layer_id"],
        "rejected_layer_id": final_sim["rejected_layer_id"],
        "rejected_m": final_sim["rejected_m"],
        "rejected_k": final_sim["rejected_k"],
        "descriptor_errors": final_sim["descriptor_errors"],
        "datapath_reuse": "existing ace2_rmsnorm_core and shared shell DMA/control",
    }
    latest_frontier = {
        "status": status,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "cells": yosys["cells"],
        "non_sram_area_mm2": yosys["non_sram_area_mm2"],
        "area_accounting_basis": yosys["area_accounting_basis"],
        "local_non_sram_area_mm2": yosys["local_non_sram_area_mm2"],
        "fmax_mhz": sta["estimated_fmax_mhz_floor_bound"],
        "fmax_mhz_floor_bound": sta["estimated_fmax_mhz_floor_bound"],
        "frequency_floor_met": timing_met,
        "area_cap_mm2": AREA_CAP_MM2,
        "area_cap_met": area_met,
        "wns_ns": sta["wns_ns"],
        "tns_ns": sta["tns_ns"],
        "cycle_or_tokens_per_second_impact": cycle_impact,
        "delta_area_percent": delta_area_percent,
        "compression_improvement_percent": None,
        "timing_improvement_percent": timing_improvement_percent,
        "timing_repair_mechanism": timing_repair_mechanism,
        "timing_repair_from_id": timing_baseline["id"],
        "critical_path": critical_path,
        "remaining_area_reserve_mm2": AREA_CAP_MM2 - yosys["non_sram_area_mm2"],
        "remaining_frequency_reserve_mhz": max(
            0.0, sta["estimated_fmax_mhz_floor_bound"] - FREQ_FLOOR_MHZ
        ),
        "remaining_frequency_reserve_mhz_floor_bound": 0.0,
    }

    manifest = dict(previous_manifest)
    manifest.update({
        "generated_at_utc": generated_at,
        "supported_layer_operator_prefix": published_prefix,
        "first_unsupported_layer_operator": first_unsupported,
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "candidate_status": status,
        "candidate_meets_numeric_acceptance": numeric_targets_met,
        "candidate_requires_fresh_sky130_ppa": False,
        "candidate_verification_complete": verification_met,
        "candidate_supported_layer_operator_prefix_after_review": candidate_prefix,
        "candidate_first_unsupported_layer_operator_after_review": PUBLISHED_FIRST_UNSUPPORTED,
        "candidate_geometry": {
            "m": 1,
            "n": 896,
            "k": 0,
            "lanes": 16,
            "beats": 56,
        },
        "independent_reviewer_acceptance": review_status,
        "independent_reviewer_verdict": (
            "evidence/review/latest/final_rmsnorm_verdict.json"
            if review is not None else None
        ),
        "candidate_rtl_hash": rtl_hash,
        "candidate_source_hashes": [
            {"path": rel, "sha256": sha256_file(ROOT / rel)}
            for rel in SOURCE_FILES
        ],
        "candidate_evidence_hashes": evidence_hashes,
        "candidate_verification_binding": (
            "evidence/frontier/latest/final_rmsnorm_verification_binding.json"
        ),
        "ppa_evidence_binding": ppa_binding,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "supported_reusable_operator_capabilities": published_capabilities,
        "candidate_reusable_operator_capabilities_after_review": candidate_capabilities,
        "source_hashes": [
            {"path": rel, "sha256": sha256_file(ROOT / rel)}
            for rel in SOURCE_FILES
        ],
        "ip_provenance": refresh_generated_rtl_provenance(
            list(previous_manifest.get("ip_provenance", []))
        ),
        "latest_evidence": {
            **dict(previous_manifest.get("latest_evidence", {})),
            "lint": lint,
            "rtl_lint_log": "evidence/frontier/latest/rtl_lint.log",
            "final_rmsnorm_sim": final_sim,
            "rtl_final_rmsnorm_sim_log": "evidence/frontier/latest/rtl_final_rmsnorm_sim.log",
            "shell_sim": shell_sim,
            "rtl_shell_sim_log": "evidence/frontier/latest/rtl_shell_sim.log",
            "sky130_yosys": yosys,
            "sky130_sta": sta,
            "ppa_evidence_binding": ppa_binding,
        },
        "layer_execution_coverage": {
            **dict(previous_manifest.get("layer_execution_coverage", {})),
            "basis": "verification/tb/ace2_shell_tb.sv complete command-dispatch regression plus +FINAL_RMSNORM_ONLY layer-24 and shape-boundary coverage",
            "final_rmsnorm_layer_ids": [24] if published else [],
            "final_rmsnorm_candidate_layer_ids": [24],
            "final_rmsnorm_negative_descriptor_coverage": {
                "layer_ids_rejected": [final_sim["rejected_layer_id"]],
                "m_values_rejected": [final_sim["rejected_m"]],
                "k_values_rejected": [final_sim["rejected_k"]],
                "cases": final_sim["descriptor_errors"],
            },
        },
        "claim_boundaries": [
            (
                "The accepted prefix includes final_rmsnorm after independent Reviewer acceptance."
                if published else
                "The accepted prefix remains through layer_23.mlp_residual_add while final_rmsnorm awaits independent Reviewer acceptance."
            ),
            "The final-RMSNorm candidate reuses the accepted RMSNorm datapath and requires m == 1, n == 896, k == 0, and reserved layer_id 24 for opcode 0x02; layer_id 25, m == 2, and k == 1 are rejected.",
            "Durable Verilator lint and complete shell-regression logs are fresh and hash-bound with the dedicated final-RMSNorm simulation and canonical PPA evidence.",
            (
                "The first unsupported accepted operator is lm_head."
                if published else
                "The first unsupported accepted operator remains final_rmsnorm; lm_head is also unsupported."
            ),
            "SKY130 synthesis/OpenSTA evidence is canonical mapped PPA, not routed GDS, signoff, tapeout, FPGA, or silicon evidence.",
        ],
    })
    write_json(MANIFEST, manifest)

    entry = {
        "id": frontier_id,
        "recorded_at_utc": generated_at,
        "stage_closing": False,
        "ordered_supported_layer_operator_prefix": published_prefix,
        "first_unsupported_layer_operator": first_unsupported,
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "candidate_supported_layer_operator_prefix_after_review": candidate_prefix,
        "candidate_first_unsupported_layer_operator_after_review": PUBLISHED_FIRST_UNSUPPORTED,
        "accepted_prefix_advanced": published,
        "candidate_meets_numeric_acceptance": numeric_targets_met,
        "independent_reviewer_acceptance": review_status,
        "independent_reviewer_verdict_sha256": (
            sha256_file(REVIEW_VERDICT) if review is not None else None
        ),
        "mode": mode,
        "decision": decision,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "cells": yosys["cells"],
        "delta_cells": int(yosys["cells"]) - baseline_cells,
        "delta_cells_basis": yosys["area_accounting_basis"],
        "non_sram_area_mm2": yosys["non_sram_area_mm2"],
        "non_sram_area_um2": yosys["non_sram_area_um2"],
        "area_accounting_basis": yosys["area_accounting_basis"],
        "local_cells_excluding_submodules": yosys["local_cells_excluding_submodules"],
        "local_non_sram_area_mm2": yosys["local_non_sram_area_mm2"],
        "delta_area_mm2": delta_area,
        "delta_area_percent": delta_area_percent,
        "compression_improvement_percent": None,
        "timing_improvement_percent": timing_improvement_percent,
        "timing_repair_mechanism": timing_repair_mechanism,
        "timing_repair_from_id": timing_baseline["id"],
        "critical_path": critical_path,
        "fmax_mhz": sta["estimated_fmax_mhz_floor_bound"],
        "fmax_mhz_floor_bound": sta["estimated_fmax_mhz_floor_bound"],
        "frequency_floor_met": timing_met,
        "area_cap_mm2": AREA_CAP_MM2,
        "area_cap_met": area_met,
        "remaining_area_reserve_mm2": AREA_CAP_MM2 - yosys["non_sram_area_mm2"],
        "remaining_frequency_reserve_mhz": max(
            0.0, sta["estimated_fmax_mhz_floor_bound"] - FREQ_FLOOR_MHZ
        ),
        "remaining_frequency_reserve_mhz_floor_bound": 0.0,
        "sky130_sta": {
            "clock_period_ns": sta["clock_period_ns"],
            "wns_ns": sta["wns_ns"],
            "tns_ns": sta["tns_ns"],
        },
        "cycle_or_tokens_per_second_impact": cycle_impact,
        "evidence": {
            "rtl_manifest": "design/RTL_MANIFEST.json",
            "rtl_lint_log": "evidence/frontier/latest/rtl_lint.log",
            "rtl_final_rmsnorm_sim_log": "evidence/frontier/latest/rtl_final_rmsnorm_sim.log",
            "rtl_shell_sim_log": "evidence/frontier/latest/rtl_shell_sim.log",
            "yosys_log": "evidence/frontier/latest/sky130_yosys.log",
            "sta_log": "evidence/frontier/latest/sky130_sta.log",
            "independent_reviewer_verdict": (
                "evidence/review/latest/final_rmsnorm_verdict.json"
                if review is not None else None
            ),
        },
        "evidence_hashes": evidence_hashes,
        "ppa_evidence_binding": ppa_binding,
        "claim_boundary": "final-RMSNorm RTL candidate with fresh canonical SKY130 mapped synthesis/OpenSTA PPA; no routed/signoff claim",
    }
    entries = list(ledger.get("entries", []))
    duplicate_binding = (
        bool(entries)
        and entries[-1].get("candidate_layer_operator") == CANDIDATE_OPERATOR
        and entries[-1].get("rtl_hash") == rtl_hash
        and entries[-1].get("constraint_hash") == constraint_hash
        and entries[-1].get("evidence_hashes") == evidence_hashes
    )
    if duplicate_binding:
        entries[-1] = entry
    else:
        entries.append(entry)
    ledger["entries"] = entries
    write_json(LEDGER, ledger)

    public = load_json(PUBLIC_STATUS, {})
    accepted_shared_frontier = next(
        (
            item for item in reversed(entries)
            if item.get("candidate_layer_operator")
            == "layer_1_through_23.shared_transformer_layer_reuse"
            and item.get("accepted_prefix_advanced") is True
            and item.get("independent_reviewer_acceptance") == "accepted"
        ),
        None,
    )
    if accepted_shared_frontier is None:
        raise RuntimeError("could not find accepted shared-layer frontier evidence")
    if published:
        blockers: list[dict[str, str]] = []
    elif numeric_targets_met:
        blockers = [{
            "id": "final_rmsnorm_independent_review",
            "status": "open",
            "summary": (
                "Final RMSNorm meets numeric PPA targets but remains unpublished "
                "pending independent Reviewer acceptance."
            ),
        }]
    else:
        missed_targets = []
        if not timing_met:
            missed_targets.append("WNS/TNS >= 0 at 10 ns")
        if not area_met:
            missed_targets.append("non-SRAM area <= 2.0 mm2")
        blockers = [{
            "id": "final_rmsnorm_numeric_ppa",
            "status": "open",
            "summary": "Final RMSNorm remains unpublished because fresh canonical "
            + "SKY130 evidence misses "
            + " and ".join(missed_targets)
            + ".",
        }]
    candidate_claim = (
        "the current final_rmsnorm candidate meets the 100 MHz SKY130 floor "
        "and 2.0 mm2 hierarchical total non-SRAM cap"
        if numeric_targets_met else
        "the current final_rmsnorm candidate does not meet all numeric SKY130 "
        "acceptance targets and remains unpublished"
    )
    public_claims = [
        {
            "claim": "current Manager-owned stage is rtl",
            "evidence": [
                "research/PIPELINE_STATE.json",
                "research/PUBLIC_STATUS.json",
            ],
        },
        {
            "claim": (
                "implementation-supported prefix covers all 24 transformer "
                "layers through layer_23.mlp_residual_add after independent "
                "Reviewer acceptance"
            ),
            "evidence": [
                "design/RTL_MANIFEST.json",
                "design/PPA_FRONTIER_LEDGER.json",
                "research/PUBLIC_STATUS.json",
            ],
        },
        {
            "claim": (
                "the shared-layer RTL discriminator covers high layer_id 23 "
                "across 13 cases, two projection shapes, and nine opcode families"
            ),
            "evidence": [
                "evidence/frontier/latest/rtl_layer_sweep_sim.log",
                "evidence/review/latest/shared_transformer_layers_verdict.json",
            ],
        },
        {
            "claim": (
                "the independently accepted shared-layer frontier "
                f"{accepted_shared_frontier['id']} meets the 100 MHz SKY130 "
                "floor and 2.0 mm2 hierarchical total non-SRAM cap"
            ),
            "evidence": [
                "design/PPA_FRONTIER_LEDGER.json",
                "evidence/review/latest/shared_transformer_layers_verdict.json",
            ],
        },
        {
            "claim": candidate_claim,
            "evidence": [
                "evidence/frontier/latest/final_rmsnorm_verification_binding.json",
                "evidence/frontier/latest/sky130_yosys.log",
                "evidence/frontier/latest/sky130_sta.log",
                "design/PPA_FRONTIER_LEDGER.json",
            ],
        },
    ]
    first_unsupported_detail = {
        "identifier": first_unsupported,
        "layer": None,
        "operator": first_unsupported,
        "reason": (
            "Final RMSNorm has independent Reviewer acceptance; lm_head is next."
            if published else
            "Final RMSNorm is implemented and PPA-bound but awaits independent Reviewer acceptance."
        ),
    }
    public.update({
        "last_updated_utc": generated_at,
        "generated_at_utc": generated_at,
        "artifact_hashes": [
            {"path": rel, "sha256": sha256_file(ROOT / rel)}
            for rel in PUBLIC_ARTIFACT_FILES
            if (ROOT / rel).exists()
        ],
        "blockers": blockers,
        "public_claims": public_claims,
        "implementation_frontier": {
            **dict(public.get("implementation_frontier", {})),
            "current_mode": mode,
            "ordered_supported_layer_operator_prefix": published_prefix,
            "supported_reusable_operator_capabilities": published_capabilities,
            "candidate_reusable_operator_capabilities_after_review": candidate_capabilities,
            "first_unsupported_layer_operator": first_unsupported_detail,
            "candidate_layer_operator": CANDIDATE_OPERATOR,
            "candidate_supported_layer_operator_prefix_after_review": candidate_prefix,
            "candidate_first_unsupported_layer_operator_after_review": PUBLISHED_FIRST_UNSUPPORTED,
            "latest_decision": decision,
            "latest_ppa_frontier": latest_frontier,
        },
        "dashboard_fields": {
            **dict(public.get("dashboard_fields", {})),
            "ordered_supported_layer_operator_prefix": published_prefix,
            "supported_reusable_operator_capabilities": published_capabilities,
            "candidate_reusable_operator_capabilities_after_review": candidate_capabilities,
            "first_unsupported_layer_operator": first_unsupported,
            "current_mode": mode,
            "latest_decision": decision,
            "latest_ppa_frontier": latest_frontier,
            "latest_ppa_frontier_status": status,
            "final_rmsnorm_candidate": CANDIDATE_OPERATOR,
            "final_rmsnorm_candidate_review_status": review_status,
        },
        "explicit_non_claims": [
            "final_rmsnorm publication remains pending independent Reviewer acceptance"
            if not published else
            "lm_head remains unsupported",
            "the complete Qwen2.5-0.5B workload remains incomplete until lm_head and required full-workload traces are implemented and verified",
            "no routed GDS, FPGA prototype, full benchmark, signoff, tapeout readiness, or fabricated silicon result is claimed",
        ],
    })
    write_json(PUBLIC_STATUS, public)

    print(
        f"bound {CANDIDATE_OPERATOR}: status={status} "
        f"accepted_first_unsupported={first_unsupported} "
        f"candidate_first_unsupported_after_review={PUBLISHED_FIRST_UNSUPPORTED}"
    )


if __name__ == "__main__":
    main()
