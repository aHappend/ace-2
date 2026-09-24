#!/usr/bin/env python3
"""Bind the reviewed layer-0 fixed-Q7 bounded no-go into RTL-stage state.

This script does not modify RTL, the Manager-owned pipeline stage, or the
accepted PPA frontier.  It verifies the sealed candidate evidence, records the
rejected diagnostic source in the RTL manifest, and refreshes traceability and
privacy-filtered dashboard state.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
TRACEABILITY = ROOT / "design/RTL_TRACEABILITY.md"
POLICY = ROOT / "design/FAST_LOOP_POLICY.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
LIVE_VIEW = ROOT / ".argus/live-view.json"
LEDGER = ROOT / "design/PPA_FRONTIER_LEDGER.json"
PROPOSAL = ROOT / "design/NUMERICAL_REPLACEMENT_PROPOSAL.md"
REVIEW = ROOT / ".autors/ace-2/wiki/sources/runs/9c79a6b9826c-r001.md"
DECISION = ROOT / "evidence/review/layer0_fixed_q7_rope_score_v1_no_go/decision.json"
BASELINE = ROOT / "benchmark/raw/quality/operator-paired-smoke-20260731T081721Z/results.json"
CANDIDATE = ROOT / "benchmark/raw/quality/layer0-fixed-q7-rope-score-v1-smoke-128-20260731/results.json"
RUN_CONTRACT = ROOT / "benchmark/raw/quality/layer0-fixed-q7-rope-score-v1-smoke-128-20260731/run_contract.json"
SOURCE_LIST = ROOT / "evidence/layer0_fixed_q7_rope_score_v1/latest/source_hashes.txt"
CANDIDATE_EVIDENCE = ROOT / "evidence/layer0_fixed_q7_rope_score_v1/latest/candidate_evidence.json"
BOUNDED_NO_GO = ROOT / "evidence/layer0_fixed_q7_rope_score_v1/latest/BOUNDED_NO_GO.json"
SOFTWARE_LOG = ROOT / "evidence/layer0_fixed_q7_rope_score_v1/latest/software_reference_unittest.log"
RTL_LOG = ROOT / "evidence/layer0_fixed_q7_rope_score_v1/latest/rtl_fixed_q7_rope_score.log"
LINT_LOG = ROOT / "evidence/layer0_fixed_q7_rope_score_v1/latest/rtl_lint.log"
RTL_SOURCE = ROOT / "rtl/ace2_fixed_q7_rope_score_core.sv"
DYNAMIC_RTL_SOURCE = ROOT / "rtl/ace2_dynamic_rope_head_core.sv"

CONTRACT_ID = "layer0_fixed_q7_rope_score_v1"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
HISTORICAL_PPA = {
    "cells": 62199,
    "non_sram_area_mm2": 0.6108746272,
    "setup_slack_ns_at_100mhz": 0.1502,
    "fmax_mhz": 100.0,
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_sha256(value: dict[str, Any]) -> str:
    canonical = json.loads(json.dumps(value))
    canonical.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(canonical, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_source_list() -> dict[str, str]:
    sources: dict[str, str] = {}
    for raw_line in SOURCE_LIST.read_text(encoding="utf-8").splitlines():
        digest, relative = raw_line.split(maxsplit=1)
        relative = relative.strip()
        path = ROOT / relative
        require(path.is_file(), f"missing candidate source: {relative}")
        require(sha256_file(path) == digest, f"candidate source hash changed: {relative}")
        sources[relative] = digest
    require(sources.get(RTL_SOURCE.relative_to(ROOT).as_posix()) == sha256_file(RTL_SOURCE),
            "fixed-Q7 RTL is not bound by the source list")
    return sources


def validate_smokes(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, dict[str, float]]:
    ratios: dict[str, dict[str, float]] = {}
    for dataset in ("wikitext2", "c4_en_512"):
        before = baseline["metrics"][dataset]
        after = candidate["metrics"][dataset]
        require(before["bf16"] == after["bf16"], f"{dataset} BF16 binding changed")
        require(after["ratio"] > before["ratio"], f"{dataset} did not regress as reviewed")
        ratios[dataset] = {
            "baseline": float(before["ratio"]),
            "candidate": float(after["ratio"]),
        }
    require(candidate.get("gate_passed") is False, "candidate smoke unexpectedly passed")
    return ratios


def refresh_public_integrity(public: dict[str, Any]) -> None:
    tracked = {
        item["path"]
        for item in public.get("artifact_hashes", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    tracked.update(
        path.relative_to(ROOT).as_posix()
        for path in (
            PIPELINE,
            MANIFEST,
            TRACEABILITY,
            POLICY,
            CHECKPOINT,
            LIVE_VIEW,
            LEDGER,
            PROPOSAL,
            REVIEW,
            DECISION,
            BASELINE,
            CANDIDATE,
            RUN_CONTRACT,
            SOURCE_LIST,
            CANDIDATE_EVIDENCE,
            BOUNDED_NO_GO,
            SOFTWARE_LOG,
            RTL_LOG,
            LINT_LOG,
            RTL_SOURCE,
            DYNAMIC_RTL_SOURCE,
            ROOT / "tools/bind_fixed_q7_rope_score_no_go.py",
        )
    )
    tracked.discard(PUBLIC.relative_to(ROOT).as_posix())
    public["artifact_hashes"] = [
        artifact(ROOT / relative)
        for relative in sorted(tracked)
        if (ROOT / relative).is_file()
    ]
    public["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonical_sha256": None,
        "canonicalization": (
            "UTF-8, sorted keys, two-space indentation, trailing newline, with "
            "integrity.canonical_sha256 set to null"
        ),
    }
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)


def main() -> None:
    pipeline = load(PIPELINE)
    baseline = load(BASELINE)
    candidate = load(CANDIDATE)
    run_contract = load(RUN_CONTRACT)
    candidate_evidence = load(CANDIDATE_EVIDENCE)
    bounded = load(BOUNDED_NO_GO)
    sources = parse_source_list()
    ratios = validate_smokes(baseline, candidate)

    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(bounded.get("contract_id") == CONTRACT_ID, "bounded result contract changed")
    require(bounded.get("decision") == "bounded_no_go", "bounded no-go decision missing")
    require(bounded.get("focused_software_rtl_checks_pass") is True, "focused checks did not pass")
    require(bounded.get("smoke_gate_passed") is False, "bounded smoke gate unexpectedly passed")
    require(run_contract["candidate"]["candidate_id"] == candidate_evidence["candidate_id"],
            "run contract candidate binding changed")
    require("Reviewer verdict: done" in REVIEW.read_text(encoding="utf-8"),
            "independent reviewer verdict is not done")
    for key in ("full_shell_regression_run", "canonical_sky130_ppa_run", "official_14_item_evaluation_run"):
        require(bounded["expensive_runs"][key] is False, f"forbidden expensive run recorded: {key}")

    review_time = "2026-07-31T15:46:50Z"
    decision = {
        "candidate_capability_accepted": False,
        "contract_id": CONTRACT_ID,
        "decision": "done",
        "disposition": "bounded_no_go",
        "evidence": {
            "baseline_smoke": artifact(BASELINE),
            "bounded_no_go": artifact(BOUNDED_NO_GO),
            "candidate_evidence": artifact(CANDIDATE_EVIDENCE),
            "candidate_smoke": artifact(CANDIDATE),
            "reviewer_verdict": artifact(REVIEW),
            "source_hash_list": artifact(SOURCE_LIST),
        },
        "focused_software_rtl_checks_pass": True,
        "historical_ppa_frontier_preserved": HISTORICAL_PPA,
        "reason": "Focused software/RTL checks pass, but both comparable paired-smoke ratios regress.",
        "reviewed_at_utc": review_time,
        "schema_version": 1,
        "smoke_gate_passed": False,
        "smoke_ratios": ratios,
        "stage": "rtl",
        "stage_closing": False,
        "unrun": [
            "full_shell_regression",
            "canonical_sky130_ppa",
            "official_14_item_evaluation",
        ],
    }
    write(DECISION, decision)

    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    review_binding = {
        "candidate_capability_accepted": False,
        "decision": "done",
        "evidence": DECISION.relative_to(ROOT).as_posix(),
        "evidence_sha256": sha256_file(DECISION),
        "reviewed_at_utc": review_time,
        "scope": "bounded_negative_result_and_expensive_run_blocking_only",
        "stage_closing": False,
    }

    manifest = load(MANIFEST)
    manifest.update(
        {
            "architecture_contract_status": "layer0_fixed_q7_rope_score_v1_completed_rejected_bounded_no_go_no_authorized_replacement",
            "candidate_first_unsupported_layer_operator_after_review": FIRST_UNSUPPORTED,
            "candidate_generated_hashes": {},
            "candidate_geometry": {
                "attention_mac_lanes": 1,
                "fixed_q7_fraction_bits": 7,
                "fixed_q7_payload_bits": 16,
                "layer_scope": 0,
                "multiplier_cycles_per_key": 256,
                "projection_mac_lanes": 4,
                "rope_lanes": 2,
                "wide_kv_record_bytes": 400,
            },
            "candidate_layer_operator": CONTRACT_ID,
            "candidate_meets_numeric_acceptance": False,
            "candidate_requires_fresh_sky130_ppa": False,
            "candidate_review_binding": review_binding,
            "candidate_rtl_hash": sha256_file(RTL_SOURCE),
            "candidate_rtl_hash_scope": "standalone_fixed_q7_rope_score_core_not_shell_integrated_not_accepted_capability",
            "candidate_source_hashes": sources,
            "candidate_status": "rejected_bounded_no_go_both_comparable_smokes_regressed",
            "candidate_supported_layer_operator_prefix_after_review": PREFIX,
            "candidate_verification_complete": False,
            "current_stage": "rtl",
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "generated_at_utc": now,
            "independent_reviewer_acceptance": False,
            "independent_reviewer_verdict": "fixed_q7_bounded_no_go_accepted_candidate_capability_rejected",
            "interfaces_contract_status": "fixed_q7_standalone_candidate_rejected_not_shell_integrated_no_authorized_rope_q_contract",
            "stage_closing": False,
            "supported_layer_operator_prefix": PREFIX,
        }
    )
    manifest["candidate_evidence_hashes"] = {
        "baseline_smoke_sha256": sha256_file(BASELINE),
        "bounded_no_go_sha256": sha256_file(BOUNDED_NO_GO),
        "candidate_evidence_sha256": sha256_file(CANDIDATE_EVIDENCE),
        "candidate_smoke_sha256": sha256_file(CANDIDATE),
        "candidate_source_hash_list_sha256": sha256_file(SOURCE_LIST),
        "focused_rtl_sha256": sha256_file(RTL_LOG),
        "focused_software_sha256": sha256_file(SOFTWARE_LOG),
        "rtl_lint_sha256": sha256_file(LINT_LOG),
    }
    manifest["candidate_interface"] = {
        "module": "ace2_fixed_q7_rope_score_core",
        "parameters": {},
        "ports": {
            "act_data_i": 512,
            "clear_i": 1,
            "clk_i": 1,
            "coefficient_error_o": 1,
            "cos_q15_i": 1024,
            "dot_q7xq7_o": 38,
            "error_valid_o": 1,
            "key_q7_i": 1024,
            "key_scale32_i": 32,
            "multiplier_cycles_o": 9,
            "numeric_overflow_o": 1,
            "out_ready_i": 1,
            "out_valid_o": 1,
            "precenter_q6_9_o": 64,
            "query_q7_i": 1024,
            "query_scale32_i": 32,
            "rope_q7_o": 1024,
            "rst_ni": 1,
            "score_mode_i": 1,
            "sin_q15_i": 1024,
            "start_ready_o": 1,
            "start_valid_i": 1,
        },
        "status": "rejected_diagnostic_not_in_accepted_shell_frontier",
    }
    manifest["diagnostic_rtl_sources"] = [
        {
            "interface": {
                "parameters": {},
                "ports": {
                    "act_data_i": 512,
                    "clear_i": 1,
                    "clk_i": 1,
                    "cos_q15_i": 1024,
                    "error_valid_o": 1,
                    "numeric_overflow_o": 1,
                    "out_data_o": 512,
                    "out_ready_i": 1,
                    "out_valid_o": 1,
                    "output_scale32_o": 32,
                    "producer_scale32_i": 32,
                    "rst_ni": 1,
                    "sin_q15_i": 1024,
                    "start_ready_o": 1,
                    "start_valid_i": 1,
                },
            },
            "module": "ace2_dynamic_rope_head_core",
            "path": DYNAMIC_RTL_SOURCE.relative_to(ROOT).as_posix(),
            "review": "evidence/review/dynamic_rope_head_scale_v1_no_go_l2/decision.json",
            "sha256": sha256_file(DYNAMIC_RTL_SOURCE),
            "status": "rejected_diagnostic_not_in_accepted_shell_frontier",
        },
        {
            "interface": manifest["candidate_interface"],
            "module": "ace2_fixed_q7_rope_score_core",
            "path": RTL_SOURCE.relative_to(ROOT).as_posix(),
            "review": DECISION.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(RTL_SOURCE),
            "status": "rejected_diagnostic_not_in_accepted_shell_frontier",
        },
    ]
    manifest["candidate_verification_binding"] = {
        "classification": "focused_rtl_checks_not_independent_verification_stage",
        "rtl_lint": artifact(LINT_LOG),
        "rtl_simulation": artifact(RTL_LOG),
        "software_reference": artifact(SOFTWARE_LOG),
    }
    manifest["claim_boundaries"] = [
        "This is a bounded negative result, not RTL stage closure or project completion.",
        "The fixed-Q7 RTL is a rejected standalone diagnostic and is not an accepted shell capability.",
        "No candidate full-shell, SKY130 PPA, official evaluation, prototype, signoff, tapeout, or silicon result is claimed.",
        "The historical 0.6108746272 mm^2 and +0.1502 ns at 100 MHz frontier remains unchanged.",
        "A structurally different numerical contract requires architecture-stage refreezing and explicit operator approval.",
    ]
    provenance = [
        item for item in manifest.get("ip_provenance", [])
        if item.get("name") not in {
            "ace2_dynamic_rope_head_core",
            "ace2_fixed_q7_rope_score_core",
        }
    ]
    provenance.append(
        {
            "kind": "first_party_rejected_diagnostic_rtl",
            "license": "repository project license not separately declared in this manifest",
            "name": "ace2_dynamic_rope_head_core",
            "source_revision": sha256_file(DYNAMIC_RTL_SOURCE),
            "third_party": False,
        }
    )
    provenance.append(
        {
            "kind": "first_party_rejected_diagnostic_rtl",
            "license": "repository project license not separately declared in this manifest",
            "name": "ace2_fixed_q7_rope_score_core",
            "source_revision": sha256_file(RTL_SOURCE),
            "third_party": False,
        }
    )
    manifest["ip_provenance"] = provenance
    manifest.setdefault("latest_evidence", {})["fixed_q7_rope_score_no_go"] = artifact(DECISION)
    manifest["proposed_replacement_contract"] = {
        "contract_id": CONTRACT_ID,
        "implementation_authorized": False,
        "review": review_binding,
        "status": "completed_rejected_bounded_no_go",
    }
    manifest["traceability"] = {
        "architecture_contract_gap": {
            "resolution_owner": "Manager routes architecture refreeze; operator approves any replacement contract",
            "status": "active_after_fixed_q7_no_go",
        },
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "rejected_diagnostic_sources_traced": [
            "ace2_dynamic_rope_head_core",
            "ace2_fixed_q7_rope_score_core",
        ],
        "rtl.contract-traceability": False,
        "rtl.hardware-discipline": False,
        "rtl.ip-provenance": True,
        "selected_mechanism": "layer0_fixed_q7_rope_score_v1_rejected_no_authorized_replacement",
        "stage_checklist": {
            "rtl.contract-traceability": False,
            "rtl.hardware-discipline": False,
            "rtl.ip-provenance": True,
        },
    }
    write(MANIFEST, manifest)

    policy = load(POLICY)
    policy["active_repair_authorization"] = {
        "authority": "operator",
        "contract_id": CONTRACT_ID,
        "execution_status": "completed_bounded_no_go_independently_reviewed",
        "implementation_authorized": False,
        "independent_reviewer_gate": review_binding,
        "official_14_item_evaluation": "forbidden_and_not_run",
        "required_manager_action": "route_to_architecture_to_select_and_refreeze_a_structurally_different_contract",
        "required_operator_action": "approve_any_replacement_contract_after_architecture_review",
        "stage_closing": False,
        "status": "consumed_rejected_bounded_no_go",
        "task_count": 1,
        "updated_at_utc": now,
    }
    proposal_gate = policy.setdefault("architecture_proposal_authorization", {})
    proposal_gate.update(
        {
            "contract_id": CONTRACT_ID,
            "implementation_authorized": False,
            "required_manager_action": "route_to_architecture_to_select_and_refreeze_a_structurally_different_contract",
            "required_operator_action": "approve_any_replacement_contract_after_architecture_review",
            "stage_closing": False,
            "status": "consumed_rejected_bounded_no_go",
            "updated_at_utc": now,
        }
    )
    write(POLICY, policy)

    TRACEABILITY.write_text(
        f"""# ACE-2 RTL traceability notes

This packet records the independently reviewed bounded no-go for
`{CONTRACT_ID}`. It is not an accepted capability or RTL-stage closeout.

## Current frontier

- Accepted prefix: `{PREFIX[0]}`, `{PREFIX[1]}`, `{PREFIX[2]}`, `{PREFIX[3]}`.
- First unsupported operator: `{FIRST_UNSUPPORTED}`.
- Mode: `ADVANCE`.
- Rejected standalone RTL: `ace2_fixed_q7_rope_score_core`, SHA-256
  `{sha256_file(RTL_SOURCE)}`.
- WikiText-2 ratio: `{ratios['wikitext2']['baseline']} -> {ratios['wikitext2']['candidate']}`.
- C4-en ratio: `{ratios['c4_en_512']['baseline']} -> {ratios['c4_en_512']['candidate']}`.

## RTL checklist

- `rtl.contract-traceability`: **not satisfied**. The rejected module is now
  source-, interface-, and provenance-traced in `design/RTL_MANIFEST.json`, but
  no accepted `layer_0.rope_q` shell implementation exists.
- `rtl.hardware-discipline`: **not satisfied for stage closure**. Focused
  standalone simulation and Verilator lint pass, including 256 multiplier
  cycles and backpressure/reset-clear checks, but rejected standalone evidence
  cannot certify the missing shell datapath, descriptor, memory, and completion
  integration.
- `rtl.ip-provenance`: **satisfied**. The candidate is first-party RTL with an
  exact source hash and no third-party or generated RTL dependency.

## Evidence boundary

The paired smoke gate failed on both required datasets. Therefore no candidate
full-shell regression, canonical SKY130 PPA, or official 14-item evaluation was
run. The accepted prefix and historical exact-hash PPA prerequisite remain
unchanged at 62,199 cells, `0.6108746272 mm^2`, and `+0.1502 ns` setup slack at
100 MHz.

## Required routing

The Manager-owned stage remains `rtl`; Planner did not edit
`research/PIPELINE_STATE.json`. A qualifying next implementation requires the
Manager to route back to architecture, refreeze one structurally different
numerical contract, and obtain explicit operator approval. Until then,
`layer_0.rope_q` remains unsupported.

Review binding: `{DECISION.relative_to(ROOT).as_posix()}`
""",
        encoding="utf-8",
    )

    CHECKPOINT.write_text(
        f"""# Goal

Advance the complete Qwen2.5-0.5B W4A8 accelerator without relaxing the 1.05x
quality limit, 2.0 mm^2 non-SRAM cap, or 100 MHz SKY130 floor.

# Current State

The Manager-owned stage is `rtl`. The accepted ordered prefix remains through
`layer_0.v_proj`; `{FIRST_UNSUPPORTED}` is first unsupported, and mode is
`ADVANCE`.

The single operator-approved `{CONTRACT_ID}` candidate is complete as an
independently reviewed bounded no-go. Four focused Python tests, standalone
Icarus RTL simulation, and Verilator lint pass. The comparable smoke ratios
regress on both datasets: WikiText-2 `{ratios['wikitext2']['baseline']} ->
{ratios['wikitext2']['candidate']}` and C4-en
`{ratios['c4_en_512']['baseline']} -> {ratios['c4_en_512']['candidate']}`.

No candidate full-shell regression, canonical SKY130 PPA, or official 14-item
evaluation was run. The accepted historical PPA frontier remains 62,199 cells,
`0.6108746272 mm^2`, and `+0.1502 ns` setup slack at 100 MHz.

# Required Routing / Blocker

The authorized fixed-Q7 direction is exhausted and did not produce an accepted
RTL capability. Planner cannot edit `research/PIPELINE_STATE.json`. The Manager
must route to architecture before a structurally different numerical contract
can be selected and refrozen; any replacement still requires explicit operator
approval before RTL implementation.

# RTL Stage Checklist

- `rtl.contract-traceability`: not satisfied; the rejected diagnostic is now
  traced, but no accepted `layer_0.rope_q` implementation exists.
- `rtl.hardware-discipline`: not satisfied for stage closure; focused standalone
  checks pass, but shell/descriptor/memory integration is absent by stop rule.
- `rtl.ip-provenance`: satisfied for current first-party/generated sources.

# Relevant Evidence

- `{DECISION.relative_to(ROOT).as_posix()}`
- `{BOUNDED_NO_GO.relative_to(ROOT).as_posix()}`
- `{CANDIDATE.relative_to(ROOT).as_posix()}`
- `{SOURCE_LIST.relative_to(ROOT).as_posix()}`
- `design/RTL_MANIFEST.json`
- `design/RTL_TRACEABILITY.md`
- `design/FAST_LOOP_POLICY.json`
- `research/PUBLIC_STATUS.json`
""",
        encoding="utf-8",
    )

    live_view = {
        "paths": [
            "research/PIPELINE_STATE.json",
            DECISION.relative_to(ROOT).as_posix(),
            BOUNDED_NO_GO.relative_to(ROOT).as_posix(),
            "design/RTL_TRACEABILITY.md",
            "design/RTL_MANIFEST.json",
            "research/PUBLIC_STATUS.json",
        ],
        "reason": "Shows the rejected fixed-Q7 candidate, unchanged supported prefix and PPA frontier, and the architecture-routing blocker.",
        "title": "RTL blocked: fixed-Q7 candidate failed both smoke gates",
        "version": 1,
    }
    write(LIVE_VIEW, live_view)

    public = load(PUBLIC)
    latest_decision = "layer0_fixed_q7_rope_score_v1_bounded_no_go_architecture_refreeze_required"
    public.update(
        {
            "architecture_proposal_gate": {
                "contract_id": CONTRACT_ID,
                "implementation_authorized": False,
                "review": review_binding,
                "status": "consumed_rejected_bounded_no_go",
            },
            "blockers": [
                {
                    "evidence": DECISION.relative_to(ROOT).as_posix(),
                    "id": "fixed_q7_no_go_requires_architecture_refreeze",
                    "reason": "The only authorized fixed-Q7 candidate regressed both comparable smoke ratios and cannot become an accepted RTL capability.",
                    "required_resolution": "Manager routes to architecture; a structurally different refrozen contract then requires explicit operator approval.",
                    "stage": "rtl",
                    "status": "active",
                }
            ],
            "current_mode": "ADVANCE",
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "generated_at_utc": now,
            "implementation_frontier": {
                "current_mode": "ADVANCE",
                "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
                "latest_decision": latest_decision,
                "latest_quality_diagnostic": {
                    "evidence": DECISION.relative_to(ROOT).as_posix(),
                    "status": "fixed_q7_bounded_no_go_both_smokes_regressed",
                },
                "latest_rtl_candidate": {
                    "evidence": DECISION.relative_to(ROOT).as_posix(),
                    "rtl_hash": sha256_file(RTL_SOURCE),
                    "rtl_hash_scope": "standalone_rejected_diagnostic_not_shell_integration",
                    "stage_closing": False,
                    "status": "rejected_bounded_no_go",
                },
                "ordered_supported_layer_operator_prefix": PREFIX,
                "required_manager_action": "route_to_architecture_for_a_structurally_different_refrozen_contract",
                "required_operator_action": "approve_any_refrozen_replacement_before_rtl",
                "rtl_contract_traceability": False,
            },
            "last_updated_utc": now,
            "latest_decision": latest_decision,
            "latest_ppa_frontier_status": "historical_pre_fixed_q7_candidate_exact_hash_bound_ppa_prerequisite",
            "selected_replacement_contract": {
                "contract_id": CONTRACT_ID,
                "implementation_authorized": False,
                "review": review_binding,
                "source": PROPOSAL.relative_to(ROOT).as_posix(),
                "status": "completed_rejected_bounded_no_go",
            },
            "supported_layer_operator_prefix": PREFIX,
        }
    )
    public["stage"] = {
        "completed_prior_stages": ["definition", "architecture", "environment"],
        "current_stage": "rtl",
        "current_stage_checklist": {
            "rtl.contract-traceability": False,
            "rtl.hardware-discipline": False,
            "rtl.ip-provenance": True,
        },
        "current_stage_evidence": [
            DECISION.relative_to(ROOT).as_posix(),
            BOUNDED_NO_GO.relative_to(ROOT).as_posix(),
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
        ],
        "current_stage_source": "research/PIPELINE_STATE.json",
        "current_stage_status": "blocked_after_fixed_q7_bounded_no_go",
        "downstream_locked_until_manager_advance": [
            "verification", "ppa", "prototype", "benchmark", "signoff"
        ],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    dashboard = public.setdefault("dashboard_fields", {})
    dashboard.update(
        {
            "candidate_mechanism": {
                "contract_id": CONTRACT_ID,
                "implementation_authorized": False,
                "review": review_binding,
                "status": "completed_rejected_bounded_no_go",
            },
            "candidate_rtl_hash": sha256_file(RTL_SOURCE),
            "current_mode": "ADVANCE",
            "current_stage": "rtl",
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "latest_decision": latest_decision,
            "latest_quality_diagnostic": public["implementation_frontier"]["latest_quality_diagnostic"],
            "latest_rtl_candidate": public["implementation_frontier"]["latest_rtl_candidate"],
            "ordered_supported_layer_operator_prefix": PREFIX,
            "required_manager_action": public["implementation_frontier"]["required_manager_action"],
            "required_operator_action": public["implementation_frontier"]["required_operator_action"],
            "routing_status": "rtl_blocked_pending_manager_architecture_refreeze",
            "rtl_contract_traceability": False,
            "supported_layer_operator_prefix": PREFIX,
        }
    )
    if isinstance(dashboard.get("latest_ppa_frontier"), dict):
        dashboard["latest_ppa_frontier"].update(
            {
                "decision": "preserve_historical_hash_bound_ppa_after_fixed_q7_no_go",
                "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
                "ordered_supported_layer_operator_prefix": PREFIX,
                "status": "historical_pre_fixed_q7_candidate_exact_hash_bound_ppa_prerequisite",
            }
        )
    dashboard["operator_policy"] = {
        "active_repair_authorization": policy["active_repair_authorization"],
        "area_cap_non_sram_mm2": policy["area_cap_non_sram_mm2"],
        "authority": policy["authority"],
        "decision_rules": policy["decision_rules"],
        "fast_loop": policy["fast_loop"],
        "frequency_floor_mhz": policy["frequency_floor_mhz"],
        "targets_relaxed": False,
    }
    public["public_claims"] = [
        {
            "claim": "the current Manager-owned stage is rtl",
            "evidence": ["research/PIPELINE_STATE.json"],
        },
        {
            "claim": "the fixed-Q7 candidate passed focused checks but regressed both comparable smoke ratios and is rejected",
            "evidence": [DECISION.relative_to(ROOT).as_posix()],
        },
        {
            "claim": "the supported prefix remains through layer_0.v_proj and layer_0.rope_q is first unsupported",
            "evidence": ["design/RTL_MANIFEST.json", "design/RTL_TRACEABILITY.md"],
        },
        {
            "claim": "no candidate full-shell regression, canonical SKY130 PPA, or official evaluation was run",
            "evidence": [BOUNDED_NO_GO.relative_to(ROOT).as_posix()],
        },
        {
            "claim": "the historical frontier remains 0.6108746272 mm^2 with +0.1502 ns setup slack at 100 MHz",
            "evidence": [CANDIDATE_EVIDENCE.relative_to(ROOT).as_posix(), "design/PPA_FRONTIER_LEDGER.json"],
        },
        {
            "claim": "the immutable 2.0 mm^2, 100 MHz, and 1.05x targets are unchanged",
            "evidence": ["MISSION.md", "design/TARGET.json"],
        },
    ]
    metrics = [
        item for item in public.get("reviewer_certified_metrics", [])
        if item.get("name") != "layer0_fixed_q7_rope_score_v1_bounded_no_go"
    ]
    metrics.append(
        {
            "certified_at_utc": review_time,
            "contract_binding_status": "rejected_bounded_no_go",
            "evidence": [DECISION.relative_to(ROOT).as_posix()],
            "name": "layer0_fixed_q7_rope_score_v1_bounded_no_go",
            "status": "focused_checks_pass_both_smoke_ratios_regress",
        }
    )
    public["reviewer_certified_metrics"] = metrics
    refresh_public_integrity(public)
    write(PUBLIC, public)

    print(
        "ACE2_FIXED_Q7_NO_GO_BIND_PASS "
        f"rtl_sha256={sha256_file(RTL_SOURCE)} "
        f"manifest_sha256={sha256_file(MANIFEST)} "
        f"public_status_sha256={sha256_file(PUBLIC)}"
    )


if __name__ == "__main__":
    main()
