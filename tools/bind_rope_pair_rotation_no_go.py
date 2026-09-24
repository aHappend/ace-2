#!/usr/bin/env python3
"""Bind the failed two-dataset gate for the selected Q/K basis candidate."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "rope_pair_rotation" / "latest"
BASELINE = ROOT / "benchmark" / "raw" / "quality" / "operator-paired-smoke-20260731T081721Z" / "results.json"
CANDIDATE = ROOT / "benchmark" / "raw" / "quality" / "rope-pair-rotation-candidate-smoke-128" / "results.json"
MANIFEST = ROOT / "design" / "RTL_MANIFEST.json"
TRACEABILITY = ROOT / "design" / "RTL_TRACEABILITY.md"
POLICY = ROOT / "design" / "FAST_LOOP_POLICY.json"
PUBLIC = ROOT / "research" / "PUBLIC_STATUS.json"
PIPELINE = ROOT / "research" / "PIPELINE_STATE.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_sha256(value: dict[str, Any]) -> str:
    canonical = json.loads(json.dumps(value))
    canonical.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(canonical, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def refresh_public_integrity(public: dict[str, Any]) -> None:
    tracked = {
        item["path"]
        for item in public.get("artifact_hashes", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    tracked.update(
        path.relative_to(ROOT).as_posix()
        for path in (
            CHECKPOINT,
            MANIFEST,
            TRACEABILITY,
            POLICY,
            PIPELINE,
            EVIDENCE / "RESULTS.json",
            ROOT / "evidence/per_head_qk_repair/latest/RESULTS.json",
            ROOT / "evidence/review/per_head_qk_no_go_l2/decision.json",
            ROOT / "tools/bind_rope_pair_rotation_no_go.py",
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
    now = utc_now()
    pipeline = json.loads(PIPELINE.read_text(encoding="utf-8"))
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    ratios: dict[str, Any] = {}
    for dataset in ("wikitext2", "c4_en_512"):
        baseline_ratio = baseline["metrics"][dataset]["ratio"]
        candidate_ratio = candidate["metrics"][dataset]["ratio"]
        ratios[dataset] = {
            "baseline": baseline_ratio,
            "candidate": candidate_ratio,
            "delta": candidate_ratio - baseline_ratio,
            "improvement_percent": (baseline_ratio - candidate_ratio) / baseline_ratio * 100.0,
            "strictly_lower": candidate_ratio < baseline_ratio,
        }
    if baseline["input_observations"] != candidate["input_observations"]:
        raise RuntimeError("candidate smoke did not use the frozen baseline inputs")
    if all(item["strictly_lower"] for item in ratios.values()):
        raise RuntimeError("no-go binder cannot bind a passing paired-smoke gate")

    rtl_hash = candidate["artifacts"]["accepted_rtl"]["candidate_rtl_hash"]
    result = {
        "schema_version": 1,
        "generated_at_utc": now,
        "stage": "rtl",
        "stage_closing": False,
        "status": "bounded_no_go_rope_pair_rotation_c4_regressed",
        "accepted_prefix_unchanged": PREFIX,
        "first_unsupported_layer_operator": "layer_0.rope_q",
        "candidate_rtl_hash": rtl_hash,
        "selected_mechanism": {
            "angle_degrees": 22.5,
            "command_flag_half_degree_units": 45,
            "per_head_qk_variant": False,
            "runtime_arithmetic_added": False,
        },
        "selection_probe": {
            "control": artifact(ROOT / "evidence" / "diagnostics" / "rope-structural-probe-control-r3-20260731" / "results.json"),
            "selected": artifact(ROOT / "evidence" / "diagnostics" / "rope-pair-angle-22p5-probe-r2-20260731" / "results.json"),
            "joint_probe_improvement": True,
        },
        "focused_checks": {
            "fixed_point_self_test": "pass",
            "rtl_lint": artifact(ROOT / "evidence" / "frontier" / "latest" / "rtl_lint.log"),
            "rtl_rope_core": artifact(ROOT / "evidence" / "frontier" / "latest" / "rtl_rope_sim.log"),
            "rtl_rope_shell": artifact(ROOT / "evidence" / "frontier" / "latest" / "rtl_rope_shell_sim.log"),
        },
        "smoke_gate": {
            "pass_condition": "strictly_lower_ratio_on_both_datasets",
            "passed": False,
            "same_input_observations": True,
            "same_model_revision": baseline["model"]["resolved_revision"] == candidate["model"]["resolved_revision"],
            "same_seeds": baseline["seeds"] == candidate["seeds"],
            "ratios": ratios,
            "baseline": artifact(BASELINE),
            "candidate": artifact(CANDIDATE),
        },
        "smoke_provenance": {
            "candidate_packet": artifact(EVIDENCE / "candidate_evidence.json"),
            "source_hash_list": artifact(EVIDENCE / "source_hashes.before"),
            "live_manifest_path": "design/RTL_MANIFEST.json",
            "manifest_drift_semantics": "the candidate manifest guard was valid during measurement; the live manifest is intentionally rewritten by this no-go binder while material RTL and numerical-source hashes remain bound by the run contract",
        },
        "expensive_runs": {
            "full_shell_regression_run": False,
            "canonical_sky130_ppa_run": False,
            "reason": "forbidden_by_failed_operator_owned_two_dataset_gate",
        },
        "immutable_targets": {
            "quality_ratio_max": 1.05,
            "area_cap_non_sram_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "quality_target_met": False,
            "targets_relaxed": False,
        },
        "rtl_checklist_evidence": {
            "rtl.contract-traceability": False,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
        },
        "architecture_contract_gap": {
            "accepted_contract": (
                "one static positive-real Q scale per query head, one K scale per "
                "KV head, and a sixteen-record 256-byte attention-head metadata table"
            ),
            "active_candidate": (
                "scalar per-tensor Q/K scales plus an offline +22.5 degree commuting "
                "basis rotation selected by cmd_flags=45"
            ),
            "effect": (
                "the failed candidate is synthesizable and lint-clean but does not "
                "satisfy the independently accepted architecture contract"
            ),
            "resolution_owner": "Manager rollback to architecture; operator approves any replacement contract",
        },
        "claim_boundaries": [
            "This is a bounded negative result, not RTL stage closure or project completion.",
            "No full-shell, synthesis, STA, power, PPA, prototype, benchmark, signoff, tapeout, or silicon result is claimed for this candidate.",
            "The historical accepted frontier and the earlier per-head no-go remain preserved.",
            "Post-measurement live-manifest drift is expected from no-go binding and is not a numerical input change.",
            "The active failed candidate does not close RTL contract traceability against the accepted per-head metadata architecture.",
        ],
        "required_next_action": (
            "manager_rollback_to_architecture_for_operator_approved_structurally_"
            "different_numerical_contract; no additional RTL candidate is authorized"
        ),
    }
    write_json(EVIDENCE / "RESULTS.json", result)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.update(
        {
            "generated_at_utc": now,
            "candidate_status": result["status"],
            "candidate_meets_numeric_acceptance": False,
            "candidate_requires_fresh_sky130_ppa": False,
            "candidate_verification_complete": False,
            "candidate_review_binding": {
                "decision": "pending",
                "level": None,
                "review_status": "rotation_no_go_not_independently_reviewed",
                "result_binding": "evidence/rope_pair_rotation/latest/RESULTS.json",
            },
            "independent_reviewer_acceptance": False,
            "independent_reviewer_verdict": "pending_rotation_no_go_review",
            "ppa_evidence_binding": None,
            "architecture_contract_status": "not_satisfied_by_failed_rotation_candidate",
            "traceability": {
                **manifest.get("traceability", {}),
                "rtl.contract-traceability": False,
                "rtl.hardware-discipline": True,
                "rtl.ip-provenance": True,
                "selected_mechanism": "failed_shared_rope_commuting_qk_pair_rotation_22_5deg",
                "architecture_contract_gap": result["architecture_contract_gap"],
            },
            "latest_evidence": {
                "rope_pair_rotation_no_go": artifact(EVIDENCE / "RESULTS.json"),
                "per_head_qk_no_go": artifact(ROOT / "evidence" / "per_head_qk_repair" / "latest" / "RESULTS.json"),
            },
            "claim_boundaries": result["claim_boundaries"],
        }
    )
    write_json(MANIFEST, manifest)

    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    active = policy["active_repair_authorization"]
    active.update(
        {
            "execution_status": "completed_bounded_no_go_c4_regressed",
            "smoke_gate_passed": False,
            "observed_smoke_ratios": ratios,
            "result_binding": "evidence/rope_pair_rotation/latest/RESULTS.json",
            "full_shell_regression_run": False,
            "canonical_sky130_ppa_run": False,
            "authorization_consumed": True,
            "replacement_numerical_contract_authorized": False,
            "required_manager_action": "rollback_to_architecture",
            "updated_at_utc": now,
        }
    )
    write_json(POLICY, policy)

    TRACEABILITY.write_text(
        f"""# ACE-2 RTL traceability notes

This packet records a failed, non-stage-closing RTL candidate. It is not an
accepted implementation of the frozen architecture contract.

## Current frontier

- Accepted prefix: `layer_0.input_rmsnorm`, `layer_0.q_proj`,
  `layer_0.k_proj`, `layer_0.v_proj`.
- First unsupported operator: `layer_0.rope_q`.
- Active failed candidate: shared +22.5 degree Q/K basis rotation with scalar
  per-tensor Q/K scales and `cmd_flags=45`.
- Candidate RTL hash: `{rtl_hash}`.
- Candidate status: `bounded_no_go_rope_pair_rotation_c4_regressed`.
- WikiText-2 ratio changed from `{ratios['wikitext2']['baseline']}` to
  `{ratios['wikitext2']['candidate']}`; C4-en changed from
  `{ratios['c4_en_512']['baseline']}` to
  `{ratios['c4_en_512']['candidate']}`. The required joint gate failed.

## RTL checklist

- `rtl.contract-traceability`: **not satisfied**. The independently accepted
  architecture requires fourteen query-head scales, two KV-head scales, and a
  sixteen-record 256-byte attention-head metadata table. The active candidate
  instead restores scalar per-tensor Q/K scales and binds an offline basis
  rotation. The candidate therefore cannot close the accepted contract.
- `rtl.hardware-discipline`: supported for this failed candidate by the current
  Verilator lint/elaboration pass and focused RoPE simulations.
- `rtl.ip-provenance`: supported. Design RTL is first-party; the generated SiLU
  LUT retains its recorded generator and regeneration command.

## Evidence boundary

The focused lint, RoPE-core, and RoPE-shell checks pass. The two-dataset quality
gate fails, so no candidate full-shell regression or canonical SKY130 PPA was
run. Historical PPA remains bound to an earlier numerical contract and is not
current evidence. Manager rollback to `architecture` is required before an
operator-approved replacement numerical contract can be implemented.
""",
        encoding="utf-8",
    )

    CHECKPOINT.write_text(
        f"""# Goal

Advance the complete Qwen2.5-0.5B W4A8 accelerator without relaxing the 1.05x
quality limit, 2.0 mm^2 non-SRAM cap, or 100 MHz SKY130 floor.

# Current State

The Manager-owned stage is `rtl`. The accepted ordered prefix remains through
`layer_0.v_proj`; `layer_0.rope_q` is first unsupported.

Two quality-recovery directions are bounded no-gos:

- Per-head Q/K metadata: both 128-token ratios regressed slightly.
- Shared +22.5 degree Q/K basis rotation: WikiText-2 improved, but C4-en
  regressed from `{ratios['c4_en_512']['baseline']}` to
  `{ratios['c4_en_512']['candidate']}`.

The active rotation RTL is lint-clean and passes focused RoPE checks, but it
does not implement the independently accepted per-head 256-byte metadata-table
architecture. `rtl.contract-traceability` is therefore unmet. No completed
candidate full-shell regression or canonical SKY130 PPA exists for either
failed direction.

# Required Routing

No additional RTL numerical candidate is authorized under the consumed repair
contract. The next legal high-value move is Manager rollback to `architecture`
for an operator-approved, structurally different numerical contract. Planner
must not edit `research/PIPELINE_STATE.json`, relax a target, reuse historical
PPA as current evidence, or advance into verification/PPA.

# Relevant Evidence

- `evidence/per_head_qk_repair/latest/RESULTS.json`
- `evidence/rope_pair_rotation/latest/RESULTS.json`
- `evidence/review/per_head_qk_no_go_l2/decision.json`
- `design/RTL_MANIFEST.json`
- `design/RTL_TRACEABILITY.md`
- `research/PUBLIC_STATUS.json`
""",
        encoding="utf-8",
    )

    public = json.loads(PUBLIC.read_text(encoding="utf-8"))
    public["generated_at_utc"] = now
    public["last_updated_utc"] = now
    public.update(
        {
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "current_mode": "ADVANCE",
            "latest_decision": "replan_to_architecture_after_per_head_and_rotation_quality_no_gos",
            "latest_ppa_frontier_status": "historical_pre_selected_basis_contract_no_new_ppa",
        }
    )
    public["blockers"] = [
        {
            "id": "rtl_architecture_contract_mismatch_after_quality_no_gos",
            "stage": "rtl",
            "status": "active",
            "reason": (
                "The active failed rotation candidate uses scalar Q/K scales and "
                "does not implement the accepted per-head metadata-table architecture."
            ),
            "required_resolution": (
                "Manager rollback to architecture and explicit operator approval of "
                "a replacement numerical contract."
            ),
            "evidence": "design/RTL_TRACEABILITY.md",
        },
        {
            "id": "rope_pair_rotation_smoke_gate_failed",
            "stage": "rtl",
            "status": "active",
            "reason": "The selected +22.5 degree fused Q/K basis improved WikiText-2 but regressed C4 on the comparable 128-token gate.",
            "required_resolution": "Independent review and fresh operator direction are required before another numerical mechanism; targets remain unchanged.",
            "evidence": "evidence/rope_pair_rotation/latest/RESULTS.json",
        },
        {
            "id": "per_head_qk_no_go_l2_checkpoint_handoff_blocked",
            "stage": "rtl",
            "status": "active",
            "reason": (
                "The sole independent per-head no-go review supported the negative "
                "result but returned blocked because it could not write CHECKPOINT.md."
            ),
            "required_resolution": (
                "Fresh operator authorization is required before a second independent "
                "review; this does not authorize another RTL mechanism."
            ),
            "evidence": "evidence/review/per_head_qk_no_go_l2/decision.json",
        },
    ]
    stage = public["stage"]
    stage["current_stage_status"] = "rtl_contract_traceability_unmet_requires_architecture_rollback"
    stage["current_stage_checklist"] = {
        "rtl.contract-traceability": False,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
    }
    stage["current_stage_evidence"] = [
        "design/RTL_MANIFEST.json",
        "design/RTL_TRACEABILITY.md",
        "evidence/rope_pair_rotation/latest/RESULTS.json",
        "evidence/per_head_qk_repair/latest/RESULTS.json",
        "evidence/review/per_head_qk_no_go_l2/decision.json",
        "CHECKPOINT.md",
    ]
    frontier = public["implementation_frontier"]
    current_environment = {
        "checklist": {
            "environment.eda-capabilities": True,
            "environment.tool-ip-selection": True,
        },
        "contract_binding": "existing_tools_remain_ready_but_active_failed_rtl_does_not_match_accepted_architecture",
        "dependency_delta": "no_new_eda_pdk_ip_license_board_compiler_or_runtime_dependency",
        "evidence": [
            "research/ENVIRONMENT_AUDIT.json",
            "research/TOOLCHAIN_CANDIDATES.md",
            "research/IP_REUSE_PLAN.md",
        ],
        "primary_fast_loop_ready": True,
        "status": "historical_environment_readiness_carried_no_contract_dependency_change",
    }
    frontier.update(
        {
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "latest_decision": "replan_to_architecture_after_per_head_and_rotation_quality_no_gos",
            "candidate_rtl_hash": rtl_hash,
            "candidate_mechanism": {
                **result["selected_mechanism"],
                "status": result["status"],
                "smoke_ratios": ratios,
            },
            "latest_ppa_frontier_status": "historical_pre_selected_basis_contract_no_new_ppa",
            "rtl_contract_traceability": False,
            "required_manager_action": "rollback_to_architecture",
            "latest_environment_stage": current_environment,
            "latest_rtl_candidate": {
                "status": result["status"],
                "evidence": "evidence/rope_pair_rotation/latest/RESULTS.json",
                "rtl_hash": rtl_hash,
                "stage_closing": False,
            },
            "latest_quality_diagnostic": {
                "status": result["status"],
                "evidence": "evidence/rope_pair_rotation/latest/RESULTS.json",
                "first_material_divergence": "model.layers.0.score",
                "selected_mechanism": result["selected_mechanism"],
                "smoke_ratios": ratios,
            },
        }
    )
    dashboard = public["dashboard_fields"]
    dashboard.update(
        {
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "current_mode": "ADVANCE",
            "latest_decision": "replan_to_architecture_after_per_head_and_rotation_quality_no_gos",
            "candidate_rtl_hash": rtl_hash,
            "candidate_mechanism": frontier["candidate_mechanism"],
            "latest_ppa_frontier_status": "historical_pre_selected_basis_contract_no_new_ppa",
            "rtl_contract_traceability": False,
            "required_manager_action": "rollback_to_architecture",
            "latest_environment_stage": current_environment,
            "latest_rtl_candidate": frontier["latest_rtl_candidate"],
            "latest_quality_diagnostic": frontier["latest_quality_diagnostic"],
        }
    )
    public["public_claims"] = [
        {
            "claim": "the current Manager-owned stage is rtl",
            "evidence": ["research/PIPELINE_STATE.json", "research/PUBLIC_STATUS.json"],
        },
        {
            "claim": "the per-head Q/K direction and the +22.5 degree fused Q/K basis direction are both preserved bounded negative results",
            "evidence": [
                "evidence/per_head_qk_repair/latest/RESULTS.json",
                "evidence/rope_pair_rotation/latest/RESULTS.json",
            ],
        },
        {
            "claim": "the ordered supported prefix remains through layer_0.v_proj and layer_0.rope_q remains first unsupported",
            "evidence": ["design/RTL_MANIFEST.json", "research/PUBLIC_STATUS.json"],
        },
        {
            "claim": "the selected candidate passed focused lint and RoPE RTL checks but failed the required joint 128-token quality gate, so no candidate full-shell or PPA was run",
            "evidence": [
                "evidence/frontier/latest/rtl_lint.log",
                "evidence/frontier/latest/rtl_rope_sim.log",
                "evidence/frontier/latest/rtl_rope_shell_sim.log",
                "evidence/rope_pair_rotation/latest/RESULTS.json",
            ],
        },
        {
            "claim": "RTL contract traceability is unmet because the failed scalar-scale rotation candidate does not implement the accepted per-head metadata-table architecture",
            "evidence": [
                "design/ARCHITECTURE.md",
                "design/RTL_MANIFEST.json",
                "design/RTL_TRACEABILITY.md",
            ],
        },
        {
            "claim": "the 2.0 mm^2, 100 MHz, and 1.05x quality targets are unchanged and are not relaxed by either no-go",
            "evidence": ["MISSION.md", "design/FAST_LOOP_POLICY.json"],
        },
        {
            "claim": "the latest PPA frontier is historical evidence for a prior numerical contract and is not republished for the failed basis candidate",
            "evidence": ["design/PPA_FRONTIER_LEDGER.json", "research/PUBLIC_STATUS.json"],
        },
    ]
    refresh_public_integrity(public)
    write_json(PUBLIC, public)
    print(
        "ACE2_ROPE_PAIR_ROTATION_NO_GO_BOUND "
        f"rtl_hash={rtl_hash} wiki={ratios['wikitext2']['candidate']} "
        f"c4={ratios['c4_en_512']['candidate']}"
    )


if __name__ == "__main__":
    main()
