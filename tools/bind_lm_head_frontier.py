#!/usr/bin/env python3
"""Bind the bounded lm-head candidate without publishing before review."""

from __future__ import annotations

import datetime as datetime_module
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if not hasattr(datetime_module, "UTC"):
    datetime_module.UTC = timezone.utc

from tools.bind_final_rmsnorm_frontier import (
    parse_critical_path,
    parse_lint,
    refresh_generated_rtl_provenance,
)
from tools.bind_rtl_frontier import parse_shell_sim, parse_sta, parse_yosys


LATEST = ROOT / "evidence" / "frontier" / "latest"
MANIFEST = ROOT / "design" / "RTL_MANIFEST.json"
LEDGER = ROOT / "design" / "PPA_FRONTIER_LEDGER.json"
PUBLIC_STATUS = ROOT / "research" / "PUBLIC_STATUS.json"
REVIEW_VERDICT = ROOT / "evidence" / "review" / "latest" / "lm_head_verdict.json"
VERIFICATION_BINDING = LATEST / "lm_head_verification_binding.json"

AREA_CAP_MM2 = 2.0
FREQ_FLOOR_MHZ = 100.0
CANDIDATE_OPERATOR = "lm_head"
CANDIDATE_CAPABILITY = "w4a8_projection_lm_head_tile"

SYNTHESIZED_MODULES = [
    "ace2_shell",
    "ace2_state_shadow",
    "ace2_attention_accumulator",
    "ace2_rmsnorm_core",
    "ace2_w4a8_proj_core",
    "ace2_rope_core",
    "ace2_softmax_core",
    "ace2_silu_gate_core",
]
ATTENTION_ACCUMULATOR_TRACEABILITY = {
    "requirement": (
        "design/SPEC.md opcodes 0x04 ATTN_SCORE and 0x06 ATTN_VALUE, signed int32 "
        "accumulation, command restart, memory-fault retirement, watchdog, and "
        "soft-reset semantics"
    ),
    "implementation": (
        "rtl/ace2_shell.sv module ace2_attention_accumulator, instantiated as "
        "u_attention_accumulator, isolates the score/value accumulator lifetime "
        "slot from the shell-wide abort cone while preserving reset and hold priority"
    ),
    "verification": (
        "complete shell regression covering score and value additions, soft reset, "
        "response-fault hold, watchdog, command restart, and completion/error retirement"
    ),
}
ATTENTION_ACCUMULATOR_PROVENANCE = {
    "name": "ace2_attention_accumulator",
    "kind": "new_project_rtl",
    "source_revision": "active_worktree",
    "license": "repository project license not separately declared in this manifest",
    "third_party": False,
}

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
    "verification/tb/ace2_w4a8_proj_tb.sv",
    *SHELL_VECTOR_FILES,
    "tools/ace2_projection_reference.py",
    "tools/gen_projection_vectors.py",
    "tools/bind_lm_head_frontier.py",
    "tools/bind_rtl_frontier.py",
    "Makefile",
    "design/RTL_TRACEABILITY.md",
    "design/SPEC.md",
    "design/WORKLOAD.md",
    "design/FAST_LOOP_POLICY.json",
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


def refresh_traceability(traceability: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refreshed = [dict(item) for item in traceability]
    for index, item in enumerate(refreshed):
        if "module ace2_attention_accumulator" in str(item.get("implementation", "")):
            refreshed[index] = dict(ATTENTION_ACCUMULATOR_TRACEABILITY)
            break
    else:
        refreshed.append(dict(ATTENTION_ACCUMULATOR_TRACEABILITY))
    return refreshed


def refresh_ip_provenance(provenance: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refreshed = refresh_generated_rtl_provenance(provenance)
    for index, item in enumerate(refreshed):
        if item.get("name") == "ace2_attention_accumulator":
            refreshed[index] = dict(ATTENTION_ACCUMULATOR_PROVENANCE)
            break
    else:
        insertion_index = next(
            (
                index
                for index, item in enumerate(refreshed)
                if item.get("kind") != "new_project_rtl"
            ),
            len(refreshed),
        )
        refreshed.insert(insertion_index, dict(ATTENTION_ACCUMULATOR_PROVENANCE))
    return refreshed


def require_current(
    output: Path,
    inputs: list[str],
    bound_source_hashes: dict[str, str],
    bound_output_sha256: str | None,
) -> None:
    if not output.exists():
        raise RuntimeError(f"missing evidence: {output.relative_to(ROOT)}")
    newest_input = max((ROOT / rel).stat().st_mtime_ns for rel in inputs)
    if output.stat().st_mtime_ns >= newest_input:
        return
    changed_inputs = [
        rel for rel in inputs
        if bound_source_hashes.get(rel) != sha256_file(ROOT / rel)
    ]
    if changed_inputs:
        raise RuntimeError(
            f"stale evidence: {output.relative_to(ROOT)}; changed inputs: "
            + ", ".join(changed_inputs)
        )
    if bound_output_sha256 != sha256_file(output):
        raise RuntimeError(
            f"stale evidence: {output.relative_to(ROOT)}; output differs from bound hash"
        )


def parse_lm_head(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"ACE2_SHELL_LM_HEAD_TB_PASS\s+layer=([0-9]+)\s+"
        r"tile_outputs=([0-9]+)\s+vocab=([0-9]+)\s+tiles=([0-9]+)\s+"
        r"cases=([0-9]+)\s+rejected_layer=([0-9]+)\s+"
        r"rejected_m=([0-9]+)\s+rejected_n=([0-9]+)\s+"
        r"rejected_k=([0-9]+)\s+descriptor_errors=([0-9]+)\s+"
        r"cycles=([0-9]+)",
        text,
    )
    if not match:
        raise RuntimeError("could not parse lm-head pass summary")
    keys = [
        "layer_id",
        "tile_outputs",
        "vocab_size",
        "tile_count",
        "cases",
        "rejected_layer_id",
        "rejected_m",
        "rejected_n",
        "rejected_k",
        "descriptor_errors",
        "cycles",
    ]
    result = {key: int(value) for key, value in zip(keys, match.groups(), strict=True)}
    result["log_sha256"] = sha256_file(log_path)
    return result


def parse_projection_semantics(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    semantic_match = re.search(
        r"ACE2_PROJ_SEMANTIC_BINS_PASS\s+round_half_ties=([0-9]+)\s+"
        r"shift_0=([0-9]+)\s+shift_63=([0-9]+)\s+semantic_cases=([0-9]+)",
        text,
    )
    pass_match = re.search(
        r"ACE2_W4A8_PROJ_TB_PASS\s+cases=([0-9]+)\s+"
        r"checked_outputs=([0-9]+)\s+max_groups=([0-9]+)",
        text,
    )
    if not semantic_match or not pass_match:
        raise RuntimeError("could not parse projection semantic-bin pass summary")
    return {
        "round_half_ties": int(semantic_match.group(1)),
        "shift_0": int(semantic_match.group(2)),
        "shift_63": int(semantic_match.group(3)),
        "semantic_cases": int(semantic_match.group(4)),
        "vector_cases": int(pass_match.group(1)),
        "checked_outputs": int(pass_match.group(2)),
        "max_groups": int(pass_match.group(3)),
        "log_sha256": sha256_file(log_path),
    }


def reviewer_status(
    rtl_hash: str, constraint_hash: str, evidence_hashes: dict[str, str]
) -> tuple[str, dict[str, Any] | None]:
    if not REVIEW_VERDICT.exists():
        return "pending", None
    verdict = load_json(REVIEW_VERDICT, None)
    if not isinstance(verdict, dict) or verdict.get("status") not in {"accepted", "rejected"}:
        raise RuntimeError("lm-head verdict must be an accepted/rejected JSON object")
    if verdict["status"] == "rejected":
        return "rejected", verdict
    expected = {
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
    }
    stale = [key for key, value in expected.items() if verdict.get(key) != value]
    reviewed = verdict.get("evidence_hashes")
    if not isinstance(reviewed, dict):
        raise RuntimeError("accepted lm-head verdict lacks evidence hashes")
    stale.extend(key for key, value in evidence_hashes.items() if reviewed.get(key) != value)
    if stale:
        raise RuntimeError("accepted lm-head verdict is stale for: " + ", ".join(stale))
    if verdict.get("publication_authorized") is not True:
        raise RuntimeError("accepted lm-head verdict must authorize publication")
    return "accepted", verdict


def main() -> None:
    if sys.argv[1:]:
        raise RuntimeError("usage: bind_lm_head_frontier.py")

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest = load_json(MANIFEST, {})
    current_prefix = list(manifest.get("supported_layer_operator_prefix", []))
    bound_source_hashes = {
        str(item["path"]): str(item["sha256"])
        for item in manifest.get("source_hashes", [])
    }
    bound_evidence_hashes = dict(manifest.get("candidate_evidence_hashes", {}))
    if manifest.get("first_unsupported_layer_operator") == CANDIDATE_OPERATOR:
        accepted_prefix = current_prefix
    elif manifest.get("first_unsupported_layer_operator") is None and current_prefix and current_prefix[-1] == CANDIDATE_OPERATOR:
        accepted_prefix = current_prefix[:-1]
    else:
        raise RuntimeError("accepted frontier must end immediately before or at lm_head")
    if not accepted_prefix or accepted_prefix[-1] != "final_rmsnorm":
        raise RuntimeError("accepted prefix does not include final_rmsnorm")

    require_current(
        LATEST / "rtl_lm_head_sim.log",
        RTL_FILES + ["verification/tb/ace2_shell_tb.sv", "verification/generated/projection_vectors.svh"],
        bound_source_hashes,
        bound_evidence_hashes.get("rtl_lm_head_sim_log_sha256"),
    )
    require_current(
        LATEST / "rtl_proj_sim.log",
        RTL_FILES
        + [
            "verification/tb/ace2_w4a8_proj_tb.sv",
            "verification/generated/projection_vectors.svh",
        ],
        bound_source_hashes,
        bound_evidence_hashes.get("rtl_proj_sim_log_sha256"),
    )
    require_current(
        LATEST / "rtl_lint.log",
        RTL_FILES + ["Makefile"],
        bound_source_hashes,
        bound_evidence_hashes.get("rtl_lint_log_sha256"),
    )
    require_current(
        LATEST / "rtl_shell_sim.log",
        RTL_FILES + ["verification/tb/ace2_shell_tb.sv", *SHELL_VECTOR_FILES],
        bound_source_hashes,
        bound_evidence_hashes.get("rtl_shell_sim_log_sha256"),
    )
    require_current(
        LATEST / "sky130_yosys.log",
        RTL_FILES + CONSTRAINT_FILES,
        bound_source_hashes,
        bound_evidence_hashes.get("sky130_yosys_log_sha256"),
    )
    require_current(
        LATEST / "sky130_sta.log",
        RTL_FILES + CONSTRAINT_FILES,
        bound_source_hashes,
        bound_evidence_hashes.get("sky130_sta_log_sha256"),
    )

    lint = parse_lint(LATEST / "rtl_lint.log")
    lm_head = parse_lm_head(LATEST / "rtl_lm_head_sim.log")
    proj_semantics = parse_projection_semantics(LATEST / "rtl_proj_sim.log")
    shell = parse_shell_sim(LATEST / "rtl_shell_sim.log")
    yosys = parse_yosys(LATEST / "sky130_yosys.log")
    sta = parse_sta(LATEST / "sky130_sta.log")
    critical_path = parse_critical_path(LATEST / "sky130_sta.log")
    verification_met = (
        lint["passed"]
        and lm_head["layer_id"] == 24
        and lm_head["tile_outputs"] == 32
        and lm_head["vocab_size"] == 151936
        and lm_head["tile_count"] == 4748
        and lm_head["cases"] >= 2
        and lm_head["rejected_layer_id"] == 23
        and lm_head["rejected_m"] == 2
        and lm_head["rejected_n"] == 64
        and lm_head["rejected_k"] == 768
        and lm_head["descriptor_errors"] >= 4
        and proj_semantics["round_half_ties"] >= 4
        and proj_semantics["shift_0"] >= 1
        and proj_semantics["shift_63"] >= 1
        and proj_semantics["semantic_cases"] >= 6
        and shell["layers"] == 24
        and shell["success_runs"] >= 68
        and shell["descriptor_errors"] >= 34
        and shell["memory_errors"] >= 8
        and shell["watchdog_errors"] >= 1
        and shell["reset_busy"] >= 1
        and shell["response_protocol_cases"] >= 10
    )
    timing_met = sta["wns_ns"] >= 0.0 and sta["tns_ns"] >= 0.0
    area_met = yosys["non_sram_area_mm2"] <= AREA_CAP_MM2
    numeric_targets_met = verification_met and timing_met and area_met

    rtl_hash = tree_hash(RTL_FILES)
    constraint_hash = tree_hash(CONSTRAINT_FILES)
    evidence_hashes = {
        "rtl_lint_log_sha256": lint["log_sha256"],
        "rtl_lm_head_sim_log_sha256": lm_head["log_sha256"],
        "rtl_proj_sim_log_sha256": proj_semantics["log_sha256"],
        "rtl_shell_sim_log_sha256": shell["log_sha256"],
        "sky130_yosys_log_sha256": yosys["log_sha256"],
        "sky130_sta_log_sha256": sta["log_sha256"],
    }
    review_status, review = reviewer_status(rtl_hash, constraint_hash, evidence_hashes)
    published = numeric_targets_met and review_status == "accepted"
    candidate_prefix = accepted_prefix + [CANDIDATE_OPERATOR]
    published_prefix = candidate_prefix if published else accepted_prefix
    first_unsupported = None if published else CANDIDATE_OPERATOR
    frontier_id = f"frontier-{generated_at}"
    ppa_binding = {
        "mode": "fresh",
        "frontier_id": frontier_id,
        "source_frontier_id": None,
        "reuse_basis": None,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "sky130_yosys_log_sha256": evidence_hashes["sky130_yosys_log_sha256"],
        "sky130_sta_log_sha256": evidence_hashes["sky130_sta_log_sha256"],
    }
    status = (
        "fresh_sky130_synth_sta_reviewer_accepted"
        if published
        else "fresh_sky130_synth_sta_meets_floor_awaiting_independent_reviewer"
        if numeric_targets_met and review_status == "pending"
        else "fresh_sky130_synth_sta_independent_reviewer_rejected"
        if numeric_targets_met
        else "fresh_sky130_synth_sta_candidate_replan"
    )
    decision = (
        "lm_head_bound_targets_met_workload_trace_next"
        if published
        else "lm_head_targets_met_awaiting_independent_reviewer"
        if numeric_targets_met and review_status == "pending"
        else "lm_head_targets_met_independent_reviewer_rejected"
        if numeric_targets_met
        else "lm_head_bound_replan"
    )
    mode = (
        "ADVANCE"
        if published
        else "ADVANCE_REVIEW_PENDING"
        if numeric_targets_met and review_status == "pending"
        else "ADVANCE_REVIEW_REJECTED"
        if numeric_targets_met
        else "ADVANCE_REPLAN"
    )

    binding = {
        "schema_version": 1,
        "project": "ACE-2",
        "generated_at_utc": generated_at,
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "candidate_rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "required_geometry": {
            "m": 1,
            "n": 32,
            "k": 896,
            "layer_id": 24,
            "vocab_size": 151936,
            "tile_count": 4748,
        },
        "verification_complete": verification_met,
        "numeric_targets_met": numeric_targets_met,
        "ppa_status": status,
        "independent_reviewer_acceptance": review_status,
        "publication_authorized": published,
        "ppa_evidence_binding": ppa_binding,
        "lint": lint,
        "lm_head_sim": lm_head,
        "projection_semantic_bins": proj_semantics,
        "shell_sim": shell,
        "sky130_yosys": yosys,
        "sky130_sta": sta,
        "evidence": {
            "rtl_lint_log": "evidence/frontier/latest/rtl_lint.log",
            "rtl_lm_head_sim_log": "evidence/frontier/latest/rtl_lm_head_sim.log",
            "rtl_proj_sim_log": "evidence/frontier/latest/rtl_proj_sim.log",
            "rtl_shell_sim_log": "evidence/frontier/latest/rtl_shell_sim.log",
            "sky130_yosys_log": "evidence/frontier/latest/sky130_yosys.log",
            "sky130_sta_log": "evidence/frontier/latest/sky130_sta.log",
        },
        "evidence_hashes": evidence_hashes,
    }
    write_json(VERIFICATION_BINDING, binding)

    accepted_capabilities = list(manifest.get("supported_reusable_operator_capabilities", []))
    candidate_capabilities = list(accepted_capabilities)
    if CANDIDATE_CAPABILITY not in candidate_capabilities:
        candidate_capabilities.append(CANDIDATE_CAPABILITY)
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
        "candidate_first_unsupported_layer_operator_after_review": None,
        "candidate_geometry": {
            "m": 1,
            "n": 32,
            "k": 896,
            "layer_id": 24,
            "vocab_size": 151936,
            "tile_count": 4748,
            "output_beats_per_tile": 2,
        },
        "independent_reviewer_acceptance": review_status,
        "independent_reviewer_verdict": (
            "evidence/review/latest/lm_head_verdict.json" if review is not None else None
        ),
        "candidate_rtl_hash": rtl_hash,
        "rtl_hash": rtl_hash,
        "constraint_hash": constraint_hash,
        "candidate_source_hashes": [
            {"path": rel, "sha256": sha256_file(ROOT / rel)} for rel in SOURCE_FILES
        ],
        "source_hashes": [
            {"path": rel, "sha256": sha256_file(ROOT / rel)} for rel in SOURCE_FILES
        ],
        "candidate_evidence_hashes": evidence_hashes,
        "candidate_verification_binding": "evidence/frontier/latest/lm_head_verification_binding.json",
        "ppa_evidence_binding": ppa_binding,
        "supported_reusable_operator_capabilities": (
            candidate_capabilities if published else accepted_capabilities
        ),
        "candidate_reusable_operator_capabilities_after_review": candidate_capabilities,
        "synthesized_modules": SYNTHESIZED_MODULES,
        "traceability": refresh_traceability(list(manifest.get("traceability", []))),
        "ip_provenance": refresh_ip_provenance(list(manifest.get("ip_provenance", []))),
        "latest_evidence": {
            **dict(manifest.get("latest_evidence", {})),
            "independent_reviewer_verdict": (
                "evidence/review/latest/lm_head_verdict.json" if review is not None else None
            ),
            "lint": lint,
            "rtl_lint_log": "evidence/frontier/latest/rtl_lint.log",
            "lm_head_sim": lm_head,
            "rtl_lm_head_sim_log": "evidence/frontier/latest/rtl_lm_head_sim.log",
            "projection_semantic_bins": proj_semantics,
            "rtl_proj_sim_log": "evidence/frontier/latest/rtl_proj_sim.log",
            "shell_sim": shell,
            "rtl_shell_sim_log": "evidence/frontier/latest/rtl_shell_sim.log",
            "sky130_yosys": yosys,
            "sky130_sta": sta,
            "ppa_evidence_binding": ppa_binding,
        },
        "layer_execution_coverage": {
            **dict(manifest.get("layer_execution_coverage", {})),
            "lm_head_candidate_layer_ids": [24],
            "lm_head_tile_outputs": 32,
            "lm_head_vocabulary_tiles": 4748,
            "lm_head_negative_descriptor_coverage": {
                "layer_ids_rejected": [lm_head["rejected_layer_id"]],
                "m_values_rejected": [lm_head["rejected_m"]],
                "n_values_rejected": [lm_head["rejected_n"]],
                "k_values_rejected": [lm_head["rejected_k"]],
                "cases": lm_head["descriptor_errors"],
            },
        },
        "claim_boundaries": [
            (
                "The accepted ordered prefix includes lm_head after independent Reviewer acceptance."
                if published
                else "The accepted ordered prefix remains at 433 items through final_rmsnorm while lm_head awaits independent Reviewer acceptance."
            ),
            "The bounded lm_head candidate reuses opcode 0x01 and the shared W4A8 projection datapath for 4748 runtime-scheduled 32-output tiles at reserved layer_id 24.",
            "Fresh lint, focused lm-head simulation, complete shell regression, and canonical SKY130 mapped synthesis/OpenSTA evidence are hash-bound.",
            "The shared projection requantizer has current-source evidence for four signed round-half ties and right-shift endpoints 0 and 63.",
            "No full-workload trace, routed GDS, signoff, tapeout, FPGA, or silicon result is claimed.",
        ],
    })
    write_json(MANIFEST, manifest)

    ledger = load_json(LEDGER, {"schema_version": 1, "project": "ACE-2", "entries": []})
    entries = list(ledger.get("entries", []))
    baseline = next(
        (
            item for item in reversed(entries)
            if item.get("candidate_layer_operator") == "final_rmsnorm"
            and item.get("accepted_prefix_advanced") is True
        ),
        None,
    )
    if baseline is None:
        raise RuntimeError("missing accepted final-RMSNorm PPA baseline")
    delta_area = yosys["non_sram_area_mm2"] - float(baseline["non_sram_area_mm2"])
    cycle_impact = {
        "candidate": CANDIDATE_OPERATOR,
        "basis": "two independent fixed-point projection vectors over one 32-logit tile, four illegal-shape/layer cases, and six shared-requantizer semantic-bin cases",
        "accepted_supported_prefix_length": len(published_prefix),
        "candidate_supported_prefix_length": len(candidate_prefix),
        "cases": lm_head["cases"],
        "cycles": lm_head["cycles"],
        "tile_outputs": 32,
        "vocab_size": 151936,
        "tile_count": 4748,
        "descriptor_errors": lm_head["descriptor_errors"],
        "datapath_reuse": "existing ace2_w4a8_proj_core and shared shell DMA/control",
    }
    entry = {
        "id": frontier_id,
        "recorded_at_utc": generated_at,
        "stage_closing": False,
        "ordered_supported_layer_operator_prefix": published_prefix,
        "first_unsupported_layer_operator": first_unsupported,
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "candidate_supported_layer_operator_prefix_after_review": candidate_prefix,
        "candidate_first_unsupported_layer_operator_after_review": None,
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
        "delta_cells": int(yosys["cells"]) - int(baseline["cells"]),
        "non_sram_area_mm2": yosys["non_sram_area_mm2"],
        "non_sram_area_um2": yosys["non_sram_area_um2"],
        "area_accounting_basis": yosys["area_accounting_basis"],
        "local_cells_excluding_submodules": yosys["local_cells_excluding_submodules"],
        "local_non_sram_area_mm2": yosys["local_non_sram_area_mm2"],
        "delta_area_mm2": delta_area,
        "delta_area_percent": delta_area / float(baseline["non_sram_area_mm2"]) * 100.0,
        "fmax_mhz": sta["estimated_fmax_mhz_floor_bound"],
        "fmax_mhz_floor_bound": sta["estimated_fmax_mhz_floor_bound"],
        "frequency_floor_met": timing_met,
        "area_cap_mm2": AREA_CAP_MM2,
        "area_cap_met": area_met,
        "remaining_area_reserve_mm2": AREA_CAP_MM2 - yosys["non_sram_area_mm2"],
        "remaining_frequency_reserve_mhz": max(
            0.0, sta["estimated_fmax_mhz_floor_bound"] - FREQ_FLOOR_MHZ
        ),
        "sky130_sta": {
            "clock_period_ns": sta["clock_period_ns"],
            "wns_ns": sta["wns_ns"],
            "tns_ns": sta["tns_ns"],
        },
        "critical_path": critical_path,
        "cycle_or_tokens_per_second_impact": cycle_impact,
        "evidence": {
            "rtl_manifest": "design/RTL_MANIFEST.json",
            "rtl_lint_log": "evidence/frontier/latest/rtl_lint.log",
            "rtl_lm_head_sim_log": "evidence/frontier/latest/rtl_lm_head_sim.log",
            "rtl_proj_sim_log": "evidence/frontier/latest/rtl_proj_sim.log",
            "rtl_shell_sim_log": "evidence/frontier/latest/rtl_shell_sim.log",
            "yosys_log": "evidence/frontier/latest/sky130_yosys.log",
            "sta_log": "evidence/frontier/latest/sky130_sta.log",
            "verification_binding": "evidence/frontier/latest/lm_head_verification_binding.json",
            "independent_reviewer_verdict": (
                "evidence/review/latest/lm_head_verdict.json" if review is not None else None
            ),
        },
        "evidence_hashes": evidence_hashes,
        "ppa_evidence_binding": ppa_binding,
        "claim_boundary": "bounded 32-output lm-head tile RTL candidate with fresh canonical SKY130 mapped synthesis/OpenSTA PPA; no publication or routed/signoff claim before review",
    }
    duplicate = (
        bool(entries)
        and entries[-1].get("candidate_layer_operator") == CANDIDATE_OPERATOR
        and entries[-1].get("rtl_hash") == rtl_hash
        and entries[-1].get("constraint_hash") == constraint_hash
        and entries[-1].get("evidence_hashes") == evidence_hashes
    )
    if duplicate:
        entry["id"] = entries[-1]["id"]
        ppa_binding["frontier_id"] = entry["id"]
        binding["ppa_evidence_binding"]["frontier_id"] = entry["id"]
        manifest["ppa_evidence_binding"]["frontier_id"] = entry["id"]
        entries[-1] = entry
        write_json(VERIFICATION_BINDING, binding)
        write_json(MANIFEST, manifest)
    else:
        entries.append(entry)
    ledger["entries"] = entries
    write_json(LEDGER, ledger)

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
        "delta_area_percent": entry["delta_area_percent"],
        "critical_path": critical_path,
        "remaining_area_reserve_mm2": AREA_CAP_MM2 - yosys["non_sram_area_mm2"],
        "remaining_frequency_reserve_mhz": max(
            0.0, sta["estimated_fmax_mhz_floor_bound"] - FREQ_FLOOR_MHZ
        ),
    }
    public = load_json(PUBLIC_STATUS, {})
    dashboard = dict(public.get("dashboard_fields", {}))
    dashboard.pop("final_rmsnorm_candidate", None)
    dashboard.pop("final_rmsnorm_candidate_review_status", None)
    dashboard.update({
        "candidate_reusable_operator_capabilities_after_review": candidate_capabilities,
        "current_mode": mode,
        "lm_head_candidate": CANDIDATE_OPERATOR,
        "lm_head_candidate_review_status": review_status,
        "first_unsupported_layer_operator": first_unsupported,
        "latest_decision": decision,
        "latest_ppa_frontier": latest_frontier,
        "latest_ppa_frontier_status": status,
        "ordered_supported_layer_operator_prefix": published_prefix,
    })
    public["dashboard_fields"] = dashboard
    implementation = dict(public.get("implementation_frontier", {}))
    implementation.update({
        "candidate_first_unsupported_layer_operator_after_review": None,
        "candidate_layer_operator": CANDIDATE_OPERATOR,
        "candidate_reusable_operator_capabilities_after_review": candidate_capabilities,
        "candidate_supported_layer_operator_prefix_after_review": candidate_prefix,
        "current_mode": mode,
        "first_unsupported_layer_operator": (
            None if published else {
                "identifier": CANDIDATE_OPERATOR,
                "layer": None,
                "operator": CANDIDATE_OPERATOR,
                "reason": (
                    "The bounded lm-head candidate awaits independent Reviewer acceptance."
                    if numeric_targets_met
                    else "The bounded lm-head candidate misses the immutable 100 MHz SKY130 floor."
                ),
            }
        ),
        "latest_decision": decision,
        "latest_ppa_frontier": latest_frontier,
        "ordered_supported_layer_operator_prefix": published_prefix,
        "supported_reusable_operator_capabilities": (
            candidate_capabilities if published else accepted_capabilities
        ),
    })
    public["implementation_frontier"] = implementation
    public["blockers"] = (
        []
        if published
        else ["independent Reviewer acceptance for lm_head"]
        if numeric_targets_met
        else [
            "lm_head SKY130 timing misses the 100 MHz floor: "
            f"WNS {sta['wns_ns']:.2f} ns, TNS {sta['tns_ns']:.2f} ns"
        ]
    )
    public["explicit_non_claims"] = [
        (
            "full-workload trace and benchmark evidence remain incomplete"
            if published
            else "lm_head meets the immutable numeric targets but remains unaccepted and unpublished pending independent Reviewer acceptance"
            if numeric_targets_met
            else "lm_head has passing RTL verification but misses the 100 MHz SKY130 floor and is not accepted or published"
        ),
        "no routed GDS, FPGA prototype, full benchmark, signoff, tapeout readiness, or fabricated silicon result is claimed",
    ]
    candidate_claim = {
        "claim": (
            "the bounded lm_head candidate meets the 100 MHz SKY130 floor and "
            "2.0 mm2 hierarchical total non-SRAM cap but remains unpublished "
            "pending independent Reviewer acceptance"
            if numeric_targets_met
            else "the bounded lm_head candidate remains unpublished because it "
            "does not meet all immutable numeric acceptance targets"
        ),
        "evidence": [
            "evidence/frontier/latest/lm_head_verification_binding.json",
            "evidence/frontier/latest/sky130_yosys.log",
            "evidence/frontier/latest/sky130_sta.log",
            "design/PPA_FRONTIER_LEDGER.json",
        ],
    }
    public_claims = [
        claim
        for claim in public.get("public_claims", [])
        if "current final_rmsnorm candidate" not in str(claim.get("claim", ""))
        and "bounded lm_head candidate" not in str(claim.get("claim", ""))
    ]
    public_claims.append(candidate_claim)
    public["public_claims"] = public_claims
    public["generated_at_utc"] = generated_at
    public["last_updated_utc"] = generated_at
    public["artifact_hashes"] = [
        {"path": rel, "sha256": sha256_file(ROOT / rel)}
        for rel in [
            "research/PIPELINE_STATE.json",
            "design/RTL_MANIFEST.json",
            "design/PPA_FRONTIER_LEDGER.json",
            "evidence/frontier/latest/rtl_lint.log",
            "evidence/frontier/latest/rtl_lm_head_sim.log",
            "evidence/frontier/latest/rtl_proj_sim.log",
            "evidence/frontier/latest/rtl_shell_sim.log",
            "evidence/frontier/latest/sky130_yosys.log",
            "evidence/frontier/latest/sky130_sta.log",
            "evidence/frontier/latest/lm_head_verification_binding.json",
        ]
    ]
    write_json(PUBLIC_STATUS, public)
    print(
        f"bound lm_head: rtl_hash={rtl_hash} area_mm2={yosys['non_sram_area_mm2']:.10f} "
        f"wns_ns={sta['wns_ns']:.4f} tns_ns={sta['tns_ns']:.4f} "
        f"review={review_status} published={published}"
    )


if __name__ == "__main__":
    main()
