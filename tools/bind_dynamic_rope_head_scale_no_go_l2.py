#!/usr/bin/env python3
"""Bind the independently certified dynamic RoPE head-scale bounded no-go.

This is an RTL-stage evidence/status binder.  It intentionally does not edit
RTL, the Manager-owned pipeline state, or the PPA frontier ledger.
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
DECISION = ROOT / "evidence/review/dynamic_rope_head_scale_v1_no_go_l2/decision.json"
BASELINE = ROOT / "benchmark/raw/quality/operator-paired-smoke-20260731T081721Z/results.json"
CANDIDATE = ROOT / "benchmark/raw/quality/dynamic-rope-head-scale-v1-smoke-128-20260731/results.json"
SOURCE_HASHES = ROOT / "evidence/dynamic_rope_head_scale_v1/latest/SOURCE_HASHES.json"
PPA_REVIEW = ROOT / "evidence/review/latest/ppa_repair_verdict.json"
PPA_CANDIDATE = ROOT / "evidence/candidates/ppa_repair_6388ec6ae7085bd7/candidate_evidence.json"
LEDGER = ROOT / "design/PPA_FRONTIER_LEDGER.json"

CONTRACT_ID = "dynamic_rope_head_scale_v1"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
EXCLUDED_DIRECTIONS = [
    "static_per_head_qk_scale_metadata",
    "fused_qk_rope_basis_rotation",
    "dynamic_post_rope_head_requantization",
]


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
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    canonical = json.loads(json.dumps(value))
    canonical.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(canonical, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def metric(result: dict[str, Any], dataset: str) -> dict[str, Any]:
    value = result.get("metrics", {}).get(dataset)
    require(isinstance(value, dict), f"missing metric: {dataset}")
    return value


def validate_comparable_smokes(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, float]]:
    require(baseline.get("gate_passed") is False, "baseline smoke unexpectedly passed")
    require(candidate.get("gate_passed") is False, "candidate smoke unexpectedly passed")
    ratios: dict[str, dict[str, float]] = {}
    for dataset in ("wikitext2", "c4_en_512"):
        before = metric(baseline, dataset)
        after = metric(candidate, dataset)
        for field in ("negative_log_likelihood", "perplexity", "scored_tokens"):
            require(
                before["bf16"][field] == after["bf16"][field],
                f"{dataset} BF16 {field} changed between comparable smokes",
            )
        require(
            before["bf16"]["sequences"] == after["bf16"]["sequences"],
            f"{dataset} BF16 sequence binding changed",
        )
        require(
            after["ratio"] > before["ratio"],
            f"{dataset} did not regress as certified",
        )
        ratios[dataset] = {
            "baseline": float(before["ratio"]),
            "candidate": float(after["ratio"]),
        }
    return ratios


def validate_source_hashes(source_packet: dict[str, Any]) -> None:
    require(source_packet.get("contract_id") == CONTRACT_ID, "source packet contract changed")
    for section in ("source_hashes", "generated_hashes"):
        for relative, expected in source_packet.get(section, {}).items():
            path = ROOT / relative
            require(path.is_file(), f"missing bound source: {relative}")
            require(sha256_file(path) == expected, f"bound source hash changed: {relative}")
    evidence_dir = SOURCE_HASHES.parent
    for filename, expected in source_packet.get("evidence_hashes", {}).items():
        path = (
            ROOT / "evidence/frontier/latest/rtl_lint.log"
            if filename == "rtl_lint.log"
            else evidence_dir / filename
        )
        require(path.is_file(), f"missing bound evidence: {path}")
        require(sha256_file(path) == expected, f"bound evidence hash changed: {path}")
    excluded = set(source_packet.get("excluded", []))
    require(
        {"full_shell_regression", "sky130_ppa", "official_14_item_evaluation"}
        <= excluded,
        "expensive-run exclusions changed",
    )


def historical_ppa_frontier(ppa_review: dict[str, Any]) -> dict[str, Any]:
    metrics = ppa_review["metrics"]
    return {
        "area_cap_met": metrics["non_sram_area_mm2"] <= 2.0,
        "area_cap_mm2": 2.0,
        "cells": metrics["mapped_cells"],
        "compression_improvement_percent": None,
        "constraint_hash": ppa_review["ordered_constraint_flow_hash_list_sha256"],
        "cycle_or_tokens_per_second_impact": {
            "status": "not_remeasured_for_rejected_dynamic_candidate"
        },
        "decision": "preserve_historical_hash_bound_ppa_prerequisite_after_dynamic_candidate_no_go",
        "delta_area_percent": None,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "fmax_interpretation": "100_mhz_verified_operating_point_not_maximum_frequency_search",
        "fmax_mhz": 100.0,
        "frequency_floor_met": metrics["setup_slack_ns_at_100mhz"] >= 0.0,
        "frequency_floor_mhz": 100.0,
        "id": ppa_review["candidate_id"],
        "mode": "ADVANCE",
        "non_sram_area_mm2": metrics["non_sram_area_mm2"],
        "ordered_supported_layer_operator_prefix": PREFIX,
        "publication_authorized": ppa_review["publication_authorized"],
        "remaining_area_reserve_mm2": 2.0 - metrics["non_sram_area_mm2"],
        "rtl_hash": ppa_review["ordered_source_hash_list_sha256"],
        "setup_slack_ns_at_100mhz": metrics["setup_slack_ns_at_100mhz"],
        "stage_closing": False,
        "status": "historical_pre_dynamic_candidate_exact_hash_bound_ppa_prerequisite",
        "technology": "SKY130 HD public ORFS platform",
    }


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
            DECISION,
            BASELINE,
            CANDIDATE,
            SOURCE_HASHES,
            PPA_REVIEW,
            PPA_CANDIDATE,
            LEDGER,
            ROOT / "tools/bind_dynamic_rope_head_scale_no_go_l2.py",
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
    source_packet = load(SOURCE_HASHES)
    ppa_review = load(PPA_REVIEW)
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    require(ppa_review.get("status") == "accepted", "historical PPA review is not accepted")
    require(ppa_review.get("stage_closing") is False, "historical PPA review scope changed")
    require(ppa_review.get("publication_authorized") is False, "historical PPA publication scope changed")
    ratios = validate_comparable_smokes(baseline, candidate)
    validate_source_hashes(source_packet)

    decision = {
        "candidate_capability_accepted": False,
        "canonical_sky130_ppa_run": False,
        "contract_id": CONTRACT_ID,
        "decision": {
            "disposition": "bounded_no_go",
            "reason": (
                "Focused software/RTL checks pass, but the comparable short smokes "
                "regress on both WikiText-2 and C4-en."
            ),
            "status": "done",
        },
        "earliest_measured_divergence": "model.layers.0.score",
        "evidence": {
            "baseline_smoke": artifact(BASELINE),
            "candidate_smoke": artifact(CANDIDATE),
            "candidate_source_hashes": artifact(SOURCE_HASHES),
            "historical_ppa_review": artifact(PPA_REVIEW),
        },
        "focused_software_rtl_checks_pass": True,
        "full_shell_regression_run": False,
        "historical_frontier_preserved": {
            "mapped_cells": ppa_review["metrics"]["mapped_cells"],
            "non_sram_area_mm2": ppa_review["metrics"]["non_sram_area_mm2"],
            "setup_slack_ns_at_100mhz": ppa_review["metrics"]["setup_slack_ns_at_100mhz"],
        },
        "next_task_constraints": {
            "excluded_directions": EXCLUDED_DIRECTIONS,
            "implementation_authorized": False,
            "required_manager_action": (
                "rollback_to_architecture_then_dispatch_exactly_one_bounded_"
                "layer0_score_analysis_proposal_task"
            ),
            "required_operator_action": (
                "explicitly_approve_any_new_architecture_contract_after_independent_l2"
            ),
            "required_output_count": 1,
            "rtl_modification_authorized": False,
        },
        "official_14_item_evaluation_run": False,
        "review_level": "independent_l2",
        "reviewed_at_utc": "2026-07-31T14:44:13Z",
        "schema_version": 1,
        "smoke_gate_passed": False,
        "smoke_ratios": ratios,
        "source_packet_binding": {
            "mission_sha256": "435470fe6e12c41b578acc27331e3037415b5d5a1a6fdea0aa5b0a3c73b37017",
            "reviewed_round_sha256": "f96f77e324e799e4d8a120b32a11bcd0cd815dff72e4f10daabb04f8643b4ca1",
        },
        "stage": "rtl",
        "stage_closing": False,
    }
    write(DECISION, decision)

    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    review_binding = {
        "accepted_scope": "bounded_negative_result_and_expensive_run_blocking_only",
        "candidate_capability_accepted": False,
        "decision": "done",
        "evidence": DECISION.relative_to(ROOT).as_posix(),
        "evidence_sha256": sha256_file(DECISION),
        "reviewed_at_utc": decision["reviewed_at_utc"],
        "stage_closing": False,
    }

    manifest = load(MANIFEST)
    manifest.update(
        {
            "architecture_contract_status": (
                "dynamic_rope_head_scale_v1_rejected_bounded_no_go_"
                "manager_architecture_rollback_required"
            ),
            "candidate_first_unsupported_layer_operator_after_review": FIRST_UNSUPPORTED,
            "candidate_layer_operator": CONTRACT_ID,
            "candidate_meets_numeric_acceptance": False,
            "candidate_requires_fresh_sky130_ppa": False,
            "candidate_rtl_hash": source_packet["source_hashes"]["rtl/ace2_dynamic_rope_head_core.sv"],
            "candidate_rtl_hash_scope": (
                "standalone_diagnostic_ace2_dynamic_rope_head_core_only_not_shell_integration"
            ),
            "candidate_source_hashes": source_packet["source_hashes"],
            "candidate_generated_hashes": source_packet["generated_hashes"],
            "candidate_status": "rejected_bounded_no_go_both_comparable_smokes_regressed",
            "candidate_supported_layer_operator_prefix_after_review": PREFIX,
            "current_stage": "rtl",
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "generated_at_utc": now,
            "independent_reviewer_acceptance": False,
            "independent_reviewer_verdict": (
                "dynamic_rope_head_scale_bounded_no_go_accepted_candidate_capability_rejected"
            ),
            "interfaces_contract_status": (
                "standalone_dynamic_head_core_is_rejected_diagnostic_only_and_not_"
                "shell_integrated;_no_replacement_interface_contract_is_authorized"
            ),
            "supported_layer_operator_prefix": PREFIX,
        }
    )
    manifest["proposed_replacement_contract"] = {
        "contract_id": CONTRACT_ID,
        "earliest_measured_divergence": "model.layers.0.score",
        "implementation_authorized": False,
        "review": review_binding,
        "status": "rejected_bounded_no_go",
    }
    manifest["dynamic_rope_head_scale_no_go_l2_review"] = review_binding
    manifest.setdefault("latest_evidence", {})["dynamic_rope_head_scale_no_go_l2"] = artifact(DECISION)
    manifest["claim_boundaries"] = [
        "This is a bounded negative result, not RTL stage closure or project completion.",
        "The standalone dynamic-head RTL is diagnostic only and is not an accepted shell capability.",
        "No candidate full-shell, SKY130 PPA, official evaluation, prototype, signoff, tapeout, or silicon result is claimed.",
        "The historical 0.6108746272 mm^2, +0.1502 ns hash-bound PPA prerequisite remains unchanged.",
        "No new numerical or interface contract is authorized until independent L2 and explicit operator approval.",
    ]
    trace = manifest.setdefault("traceability", {})
    trace["architecture_contract_gap"] = {
        "earliest_measured_divergence": "model.layers.0.score",
        "excluded_repeat_directions": EXCLUDED_DIRECTIONS,
        "resolution_owner": (
            "Manager rolls back to architecture and dispatches exactly one bounded "
            "analysis/proposal task; any resulting contract requires independent L2 "
            "and explicit operator approval before RTL changes"
        ),
        "status": "active",
    }
    trace["stage_checklist"] = {
        "rtl.contract-traceability": False,
        "rtl.hardware-discipline": False,
        "rtl.ip-provenance": True,
    }
    write(MANIFEST, manifest)

    policy = load(POLICY)
    active = policy["active_repair_authorization"]
    active.update(
        {
            "execution_status": "completed_bounded_no_go_independently_accepted",
            "implementation_authorized": False,
            "independent_reviewer_gate": review_binding,
            "required_manager_action": (
                "rollback_to_architecture_then_dispatch_exactly_one_bounded_"
                "layer0_score_analysis_proposal_task"
            ),
            "required_operator_action": (
                "approve_replacement_contract_only_after_independent_l2"
            ),
            "status": "consumed_rejected_bounded_no_go",
            "updated_at_utc": now,
        }
    )
    proposal = policy["architecture_proposal_authorization"]
    proposal.update(
        {
            "contract_id": None,
            "excluded_directions": EXCLUDED_DIRECTIONS,
            "implementation_authorized": False,
            "required_manager_action": (
                "rollback_to_architecture_then_dispatch_exactly_one_bounded_"
                "layer0_score_analysis_proposal_task"
            ),
            "required_operator_action": (
                "explicit_approval_of_new_contract_after_independent_l2"
            ),
            "required_output": (
                "one_structurally_different_numerical_mechanism_grounded_in_"
                "model.layers.0.score_divergence_with_interface_cycle_sram_area_"
                "risk_and_minimal_discriminating_test_contract"
            ),
            "required_output_count": 1,
            "scope": "analysis_proposal_only_after_manager_architecture_rollback",
            "status": "manager_architecture_rollback_pending",
            "task_count": 1,
            "updated_at_utc": now,
        }
    )
    write(POLICY, policy)

    TRACEABILITY.write_text(
        f"""# ACE-2 RTL traceability notes

This packet records the independently certified bounded no-go for
`{CONTRACT_ID}`. It is not an accepted capability or stage closeout.

## Current frontier

- Accepted prefix: `{PREFIX[0]}`, `{PREFIX[1]}`, `{PREFIX[2]}`, `{PREFIX[3]}`.
- First unsupported operator: `{FIRST_UNSUPPORTED}`.
- Mode: `ADVANCE`.
- Candidate: `{CONTRACT_ID}` standalone diagnostic core, SHA-256
  `{source_packet['source_hashes']['rtl/ace2_dynamic_rope_head_core.sv']}`.
- WikiText-2 ratio regressed from `{ratios['wikitext2']['baseline']}` to
  `{ratios['wikitext2']['candidate']}`.
- C4-en ratio regressed from `{ratios['c4_en_512']['baseline']}` to
  `{ratios['c4_en_512']['candidate']}`.

## RTL checklist

- `rtl.contract-traceability`: **not satisfied**. The rejected standalone core
  is not integrated into the shell and cannot satisfy the selected contract.
- `rtl.hardware-discipline`: **not certified**. Focused vectors pass, but the
  standalone implementation has top-level lint width/index warnings and does
  not implement the frozen shared reciprocal/encoder/rounder schedule.
- `rtl.ip-provenance`: **satisfied** for the inspected sources. The candidate is
  first-party RTL; exact source and generated-vector hashes are recorded in
  `evidence/dynamic_rope_head_scale_v1/latest/SOURCE_HASHES.json`.

## Evidence boundary

Independent L2 accepted only the bounded negative result and the decision to
block expensive runs. No candidate full-shell regression, canonical SKY130 PPA,
or official 14-item evaluation was run. The historical exact-hash prerequisite
frontier remains 62,199 cells, `0.6108746272 mm^2`, and `+0.1502 ns` setup slack
at 100 MHz; it is not quality acceptance for this rejected candidate.

## Required routing

The Manager-owned stage remains `rtl`. The Manager must roll back to
`architecture` and dispatch exactly one bounded analysis/proposal task grounded
in `model.layers.0.score`. That task must not repeat static per-head metadata,
fused Q/K basis rotation, or dynamic post-RoPE head requantization. No RTL may
change until a structurally different architecture contract receives
independent L2 and explicit operator approval.

Review: `{DECISION.relative_to(ROOT).as_posix()}`
""",
        encoding="utf-8",
    )

    CHECKPOINT.write_text(
        f"""# Goal

Advance the complete Qwen2.5-0.5B W4A8 accelerator without relaxing the 1.05x
quality limit, 2.0 mm^2 non-SRAM cap, or 100 MHz SKY130 floor.

# Current State

The Manager-owned stage is `rtl`. The accepted ordered prefix remains through
`layer_0.v_proj`; `layer_0.rope_q` is first unsupported, and mode is `ADVANCE`.

Independent L2 certified `{CONTRACT_ID}` only as a bounded no-go. Focused
software and standalone RTL checks pass, but the comparable smoke ratios regress
on both datasets: WikiText-2 `{ratios['wikitext2']['baseline']} ->
{ratios['wikitext2']['candidate']}` and C4-en
`{ratios['c4_en_512']['baseline']} -> {ratios['c4_en_512']['candidate']}`.
No candidate full-shell, PPA, or official evaluation was run.

The historical exact-hash PPA prerequisite remains 62,199 cells,
`0.6108746272 mm^2`, and `+0.1502 ns` setup slack at 100 MHz. It is historical
pre-replacement evidence, not acceptance of the rejected dynamic candidate.

# Required Routing / Blocker

Planner cannot edit `research/PIPELINE_STATE.json`. The Manager must roll back
to `architecture` and dispatch exactly one bounded analysis/proposal task for a
structurally different numerical mechanism grounded in
`model.layers.0.score`. Static per-head scale metadata, fused Q/K basis
rotation, and dynamic post-RoPE head requantization are excluded. RTL changes
remain forbidden until the resulting architecture contract receives independent
L2 and explicit operator approval.

# RTL Stage Checklist

- `rtl.contract-traceability`: not satisfied.
- `rtl.hardware-discipline`: not certified for the rejected standalone core.
- `rtl.ip-provenance`: satisfied for the inspected first-party/generated sources.

# Relevant Evidence

- `{DECISION.relative_to(ROOT).as_posix()}`
- `{CANDIDATE.relative_to(ROOT).as_posix()}`
- `{BASELINE.relative_to(ROOT).as_posix()}`
- `{SOURCE_HASHES.relative_to(ROOT).as_posix()}`
- `{PPA_REVIEW.relative_to(ROOT).as_posix()}`
- `design/RTL_MANIFEST.json`
- `design/RTL_TRACEABILITY.md`
- `design/FAST_LOOP_POLICY.json`
- `research/PUBLIC_STATUS.json`
""",
        encoding="utf-8",
    )

    frontier = historical_ppa_frontier(ppa_review)
    public = load(PUBLIC)
    latest_decision = (
        "dynamic_rope_head_scale_v1_bounded_no_go_manager_architecture_rollback_"
        "and_one_analysis_proposal_task_required"
    )
    public.update(
        {
            "architecture_certification_mission": {
                "contract_id": CONTRACT_ID,
                "implementation_authorized": False,
                "mission_count": 1,
                "scope": "historical_architecture_certification_for_rejected_contract",
                "status": "candidate_rejected_after_bounded_rtl_smoke_gate",
            },
            "architecture_proposal_gate": {
                "earliest_measured_divergence": "model.layers.0.score",
                "excluded_directions": EXCLUDED_DIRECTIONS,
                "implementation_authorized": False,
                "required_manager_action": (
                    "rollback_to_architecture_then_dispatch_exactly_one_bounded_"
                    "layer0_score_analysis_proposal_task"
                ),
                "required_operator_action": (
                    "explicit_approval_of_new_contract_after_independent_l2"
                ),
                "required_output": proposal["required_output"],
                "required_output_count": 1,
                "status": "manager_architecture_rollback_pending",
            },
            "blockers": [
                {
                    "evidence": DECISION.relative_to(ROOT).as_posix(),
                    "id": "dynamic_rope_head_scale_no_go_requires_architecture_rollback",
                    "reason": (
                        "The single authorized dynamic candidate regressed both "
                        "comparable smoke ratios and cannot close RTL."
                    ),
                    "required_resolution": (
                        "Manager rollback to architecture and exactly one bounded "
                        "analysis/proposal task; then independent L2 and explicit "
                        "operator approval before RTL modification."
                    ),
                    "stage": "rtl",
                    "status": "active",
                }
            ],
            "current_mode": "ADVANCE",
            "dynamic_rope_head_scale_no_go_l2_review": review_binding,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "generated_at_utc": now,
            "last_updated_utc": now,
            "latest_decision": latest_decision,
            "latest_ppa_frontier_status": frontier["status"],
            "selected_replacement_contract": {
                "contract_id": CONTRACT_ID,
                "implementation_authorized": False,
                "review": review_binding,
                "status": "rejected_bounded_no_go",
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
            CANDIDATE.relative_to(ROOT).as_posix(),
            SOURCE_HASHES.relative_to(ROOT).as_posix(),
            MANIFEST.relative_to(ROOT).as_posix(),
            TRACEABILITY.relative_to(ROOT).as_posix(),
        ],
        "current_stage_source": PIPELINE.relative_to(ROOT).as_posix(),
        "current_stage_status": "bounded_no_go_requires_manager_architecture_rollback",
        "downstream_locked_until_manager_advance": [
            "verification",
            "ppa",
            "prototype",
            "benchmark",
            "signoff",
        ],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    implementation = public.setdefault("implementation_frontier", {})
    implementation.update(
        {
            "candidate_mechanism": {
                "contract_id": CONTRACT_ID,
                "implementation_authorized": False,
                "review": review_binding,
                "status": "rejected_bounded_no_go",
            },
            "candidate_rtl_hash": manifest["candidate_rtl_hash"],
            "current_mode": "ADVANCE",
            "current_stage": "rtl",
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "latest_decision": latest_decision,
            "latest_ppa_frontier": frontier,
            "latest_ppa_frontier_status": frontier["status"],
            "ordered_supported_layer_operator_prefix": PREFIX,
            "required_manager_action": public["architecture_proposal_gate"]["required_manager_action"],
            "required_operator_action": public["architecture_proposal_gate"]["required_operator_action"],
            "routing_status": "manager_architecture_rollback_pending",
            "rtl_contract_traceability": False,
            "supported_layer_operator_prefix": PREFIX,
        }
    )
    dashboard = public.setdefault("dashboard_fields", {})
    dashboard.update(
        {
            "current_mode": "ADVANCE",
            "current_stage": "rtl",
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "latest_decision": latest_decision,
            "latest_ppa_frontier": frontier,
            "latest_ppa_frontier_status": frontier["status"],
            "ordered_supported_layer_operator_prefix": PREFIX,
            "required_manager_action": public["architecture_proposal_gate"]["required_manager_action"],
            "required_operator_action": public["architecture_proposal_gate"]["required_operator_action"],
            "routing_status": "manager_architecture_rollback_pending",
            "rtl_contract_traceability": False,
            "supported_layer_operator_prefix": PREFIX,
        }
    )
    for section in (implementation, dashboard):
        section["candidate_mechanism"] = {
            "contract_id": CONTRACT_ID,
            "implementation_authorized": False,
            "review": review_binding,
            "status": "rejected_bounded_no_go",
        }
        section["candidate_rtl_hash"] = manifest["candidate_rtl_hash"]
        architecture_model = section.get("current_architecture_performance_model")
        if isinstance(architecture_model, dict):
            architecture_model["implementation_authorized"] = False
            architecture_model["status"] = (
                "historical_estimate_for_rejected_dynamic_contract_not_current_architecture"
            )
        environment_stage = section.get("latest_environment_stage")
        if isinstance(environment_stage, dict):
            environment_stage.update(
                {
                    "contract_binding_status": "historical_rejected_dynamic_contract",
                    "reviewer_verdict": "accepted_historical_environment_gate",
                    "status": "historical_completed_then_candidate_rejected",
                }
            )
        section["latest_quality_diagnostic"] = {
            "evidence": DECISION.relative_to(ROOT).as_posix(),
            "first_material_divergence": "model.layers.0.score",
            "status": "dynamic_rope_head_scale_v1_bounded_no_go_both_smokes_regressed",
        }
        section["latest_rtl_candidate"] = {
            "evidence": DECISION.relative_to(ROOT).as_posix(),
            "rtl_hash": manifest["candidate_rtl_hash"],
            "rtl_hash_scope": manifest["candidate_rtl_hash_scope"],
            "stage_closing": False,
            "status": manifest["candidate_status"],
        }
        operator_policy = section.setdefault("operator_policy", {})
        operator_policy["active_repair_authorization"] = policy[
            "active_repair_authorization"
        ]
        operator_policy["architecture_proposal_authorization"] = policy[
            "architecture_proposal_authorization"
        ]
    claims = [
        item
        for item in public.get("public_claims", [])
        if isinstance(item, dict)
        and not str(item.get("claim", "")).startswith("the current Manager-owned stage is ")
        and CONTRACT_ID not in str(item.get("claim", ""))
    ]
    public["public_claims"] = [
        *claims,
        {
            "claim": "the current Manager-owned stage is rtl",
            "evidence": [PIPELINE.relative_to(ROOT).as_posix()],
        },
        {
            "claim": (
                "independent L2 accepted dynamic_rope_head_scale_v1 only as a "
                "bounded no-go after both comparable smoke ratios regressed"
            ),
            "evidence": [DECISION.relative_to(ROOT).as_posix()],
        },
        {
            "claim": (
                "the historical exact-hash SKY130 prerequisite remains 0.6108746272 "
                "mm^2 with +0.1502 ns setup slack at 100 MHz and was not rerun for "
                "the rejected dynamic candidate"
            ),
            "evidence": [PPA_REVIEW.relative_to(ROOT).as_posix()],
        },
    ]
    write(
        LIVE_VIEW,
        {
            "paths": [
                PIPELINE.relative_to(ROOT).as_posix(),
                DECISION.relative_to(ROOT).as_posix(),
                TRACEABILITY.relative_to(ROOT).as_posix(),
                POLICY.relative_to(ROOT).as_posix(),
                PUBLIC.relative_to(ROOT).as_posix(),
            ],
            "reason": (
                "Shows the independently certified dynamic-candidate no-go, immutable "
                "targets, preserved historical PPA prerequisite, and required Manager "
                "architecture rollback before one bounded analysis/proposal task."
            ),
            "title": "RTL no-go: architecture rollback required",
            "version": 1,
        },
    )
    refresh_public_integrity(public)
    write(PUBLIC, public)

    print(
        "ACE2_DYNAMIC_ROPE_HEAD_SCALE_NO_GO_L2_BIND_PASS "
        f"decision_sha256={sha256_file(DECISION)} stage={pipeline['current_stage']}"
    )


if __name__ == "__main__":
    main()
