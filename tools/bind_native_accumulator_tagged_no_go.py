#!/usr/bin/env python3
"""Seal the bounded focused-quality no-go for native tagged attention."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_native_accumulator_tagged_attention_v1"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
BASELINE = ROOT / "evidence/layer0_tile_bfp_score_attention_v1/focused-candidate-full-20260801-v1/results.json"
FOCUSED = ROOT / "evidence/shared_native_accumulator_tagged_attention_v1/focused-candidate-all-layer-20260801-v1/results.json"
LATEST = ROOT / "evidence/shared_native_accumulator_tagged_attention_v1/latest"
CANDIDATE = LATEST / "candidate_evidence.json"
NO_GO = LATEST / "BOUNDED_NO_GO.json"
RTL_REVIEW = ROOT / "evidence/review/rtl_checklist_shared_native_accumulator_tagged_attention_v1/decision.json"
QUALITY_SOURCES = [
    "tools/ace2_full_model_fixed_point.py",
    "tools/localize_score_to_lm_head.py",
    "tools/bind_native_accumulator_tagged_no_go.py",
    "verification/test_native_accumulator_tagged_full_model.py",
]
QUALITY_LOGS = [
    "evidence/shared_native_accumulator_tagged_attention_v1/latest/full_model_self_test.log",
    "evidence/shared_native_accumulator_tagged_attention_v1/latest/full_model_tensor_unittest.log",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(clone, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def refresh_human_routing(
    *, candidate_id: str, no_go_sha256: str, generated_at_utc: str
) -> None:
    mission_path = ROOT / "MISSION.md"
    mission = mission_path.read_text(encoding="utf-8")
    start_marker = "### Active numerical repair implementation authorization\n"
    end_marker = "## Verification and implementation contract\n"
    require(start_marker in mission, "MISSION active-repair section is missing")
    require(end_marker in mission, "MISSION verification section is missing")
    before, remainder = mission.split(start_marker, 1)
    _, after = remainder.split(end_marker, 1)
    active_section = f"""### Active numerical repair implementation authorization

The accepted prefix remains through `layer_0.v_proj`; the first unsupported
operator remains `layer_0.rope_q`. `ADVANCE` mode, the 1.05x quality limit,
the 2.0 mm^2 non-SRAM cap, the 100 MHz floor, and the historical accepted PPA
frontier remain unchanged.

The current contract, `shared_native_accumulator_tagged_attention_v1`, was
implemented as standalone synthesizable RTL candidate `{candidate_id}`. Its
independent RTL checklist passed, including contract/interface traceability,
hardware discipline, and first-party generated-source provenance. The frozen
128-token all-layer discriminator nevertheless failed: WikiText-2 final
`lm_head` relative-L2 improved from `0.9105705798` to `0.9099393324`, while
C4-en regressed from `1.1849780795` to `1.2030940076`. The required strict
improvement on both datasets was therefore not met.

The candidate is sealed by
`evidence/shared_native_accumulator_tagged_attention_v1/latest/BOUNDED_NO_GO.json`.
Its one-time implementation authority is consumed and
`implementation_authorized=false`. No paired smoke, shell admission,
full-shell regression, candidate PPA, official evaluation, prototype, full
benchmark, signoff, tapeout, or silicon result followed.

No structurally distinct successor RTL is authorized while the Manager-owned
stage remains `rtl`. The required next action is a Manager reroute from `rtl`
to `architecture` to freeze and independently review a new numerical and
compute contract. Planner and Engineer must not edit
`research/PIPELINE_STATE.json`.

"""
    mission_path.write_text(
        before + active_section + end_marker + after,
        encoding="utf-8",
    )

    dump(
        ROOT / ".argus/live-view.json",
        {
            "version": 1,
            "title": "Native-accumulator tagged attention bounded no-go",
            "reason": (
                "The frozen C4 final-lm_head gate failed; implementation authority "
                "is consumed and a Manager rtl-to-architecture reroute is required."
            ),
            "candidate_id": candidate_id,
            "evidence_sha256": no_go_sha256,
            "generated_at_utc": generated_at_utc,
            "paths": [
                "research/PIPELINE_STATE.json",
                "CHECKPOINT.md",
                "design/RTL_MANIFEST.json",
                "evidence/shared_native_accumulator_tagged_attention_v1/latest/BOUNDED_NO_GO.json",
                "research/PUBLIC_STATUS.json",
            ],
        },
    )


def refresh_live_contract_copies(value: Any, live_record: dict[str, Any]) -> None:
    if isinstance(value, dict):
        if (
            value.get("contract_id") == CONTRACT
            and (
                value.get("candidate_id") == live_record.get("candidate_id")
                or "implementation_authorized" in value
            )
        ):
            for key in (
                "implementation_authorized",
                "operator_approval_consumed",
                "required_manager_action",
                "result_binding",
                "result_binding_sha256",
                "status",
            ):
                value[key] = copy.deepcopy(live_record[key])
        for child in value.values():
            refresh_live_contract_copies(child, live_record)
    elif isinstance(value, list):
        for child in value:
            refresh_live_contract_copies(child, live_record)


def main() -> None:
    pipeline = load(ROOT / "research/PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage is not rtl")
    manifest = load(ROOT / "design/RTL_MANIFEST.json")
    candidate = load(CANDIDATE)
    review = load(RTL_REVIEW)
    require(candidate.get("contract_id") == CONTRACT, "candidate contract differs")
    require(
        candidate.get("candidate_rtl_hash") == manifest.get("candidate_rtl_hash"),
        "candidate and manifest RTL hashes differ",
    )
    require(review.get("reviewer_status") == "done", "standalone RTL review is not done")
    require(
        review.get("candidate_rtl_hash") == candidate.get("candidate_rtl_hash"),
        "standalone RTL review hash differs",
    )
    require(all(review.get("checklist", {}).values()), "standalone RTL checklist is incomplete")
    for relative in QUALITY_SOURCES + QUALITY_LOGS:
        require((ROOT / relative).is_file(), f"missing quality artifact: {relative}")
    require(
        "SELF_TEST status=pass" in (ROOT / QUALITY_LOGS[0]).read_text(encoding="utf-8"),
        "full-model self-test did not pass",
    )
    require(
        "OK" in (ROOT / QUALITY_LOGS[1]).read_text(encoding="utf-8"),
        "full-model tensor/scalar equality test did not pass",
    )

    baseline = load(BASELINE)
    focused = load(FOCUSED)
    require(
        baseline.get("input_observations") == focused.get("input_observations"),
        "focused candidate did not reproduce the frozen predecessor inputs",
    )
    require(
        focused.get("diagnostic_rope_mechanism") == CONTRACT,
        "focused result is not the native tagged mechanism",
    )
    require(focused.get("capture_token_limit") == 128, "focused result is not 128 tokens")

    comparisons: dict[str, Any] = {}
    all_required_improved = True
    for dataset in ("wikitext2", "c4_en_512"):
        comparisons[dataset] = {
            "first_material_divergence": focused["first_material_divergence"][dataset],
        }
        for boundary in (
            "model.layers.0.score",
            "model.layers.0.attention_value",
            "lm_head",
        ):
            baseline_value = baseline["comparisons"][dataset][boundary]["relative_l2_error"]
            candidate_value = focused["comparisons"][dataset][boundary]["relative_l2_error"]
            improved = candidate_value < baseline_value
            comparisons[dataset][boundary] = {
                "baseline_relative_l2": baseline_value,
                "candidate_relative_l2": candidate_value,
                "delta_relative_l2": candidate_value - baseline_value,
                "strictly_improved": improved,
            }
            all_required_improved = all_required_improved and improved
    require(
        comparisons["c4_en_512"]["lm_head"]["strictly_improved"] is False,
        "expected C4 lm_head discriminator failure is absent",
    )
    require(all_required_improved is False, "focused discriminator unexpectedly passed")
    require(
        all(
            comparisons[dataset]["first_material_divergence"]
            != "model.layers.0.score"
            for dataset in comparisons
        ),
        "focused result introduced a material divergence before the accepted score boundary",
    )

    now = utc_now()
    quality_source_hashes = {
        relative: sha256(ROOT / relative) for relative in QUALITY_SOURCES
    }
    payload = {
        "schema_version": 1,
        "contract_id": CONTRACT,
        "candidate_id": candidate["candidate_id"],
        "candidate_rtl_hash": candidate["candidate_rtl_hash"],
        "sealed_at_utc": now,
        "decision": "bounded_no_go_focused_discriminator_c4_lm_head_failed",
        "accepted_capability": False,
        "accepted_frontier_changed": False,
        "operator_approval_consumed": True,
        "stage_closing": False,
        "manager_owned_stage": "rtl",
        "implementation_adequacy": {
            "standalone_rtl_candidate": artifact(CANDIDATE),
            "independent_rtl_review": artifact(RTL_REVIEW),
            "focused_rtl_logs": [
                artifact(ROOT / relative)
                for relative in (
                    "evidence/shared_native_accumulator_tagged_attention_v1/latest/software_reference_unittest.log",
                    "evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_simulation.log",
                    "evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_rope_lint.log",
                    "evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_score_lint.log",
                    "evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_elaboration.log",
                )
            ],
            "full_model_checks": [artifact(ROOT / relative) for relative in QUALITY_LOGS],
            "quality_source_hashes": quality_source_hashes,
            "construct_fidelity": "The all-24-layer model retains signed-32 Q/K accumulators, emits per-output Scale32 records, applies tagged absolute RoPE, exact exponent-aligned score accumulation, Q20.44 conversion, staged Q1.31/Q0.15 softmax, and existing int8 attention value. Tensor kernels match the scalar reference on deterministic vectors.",
        },
        "focused_discriminator": {
            "run": True,
            "passed": False,
            "baseline": artifact(BASELINE),
            "candidate": artifact(FOCUSED),
            "candidate_sha256s": artifact(FOCUSED.parent / "SHA256SUMS"),
            "input_observations_equal": True,
            "scope": "all_24_layers_one_shared_numerical_mechanism",
            "comparisons": comparisons,
            "finding": "Layer-0 score and attention-value relative-L2 improve on both datasets and WikiText-2 lm_head improves, but C4 lm_head relative-L2 regresses, violating the frozen both-dataset stop rule.",
        },
        "downstream_runs": {
            "paired_smoke": False,
            "full_shell_regression": False,
            "canonical_sky130_ppa": False,
            "official_validation": False,
            "prototype": False,
            "full_benchmark": False,
            "signoff": False,
        },
        "preserved_frontier": {
            "mode": "ADVANCE",
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "cells": 62199,
            "non_sram_area_mm2": 0.6108746272,
            "setup_slack_ns_at_100mhz": 0.1502,
            "area_cap_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "remaining_area_reserve_mm2": 1.3891253728,
            "candidate_ppa_run": False,
        },
        "required_next_action": "Manager reroutes rtl to architecture for one structurally distinct successor; the native-accumulator tagged direction is sealed and must not proceed to paired smoke, full regression, or PPA.",
        "claim_boundary": "Bounded negative focused-quality result only; not accepted RTL capability, PPA, benchmark acceptance, signoff, tapeout, or silicon evidence.",
    }
    dump(NO_GO, payload)
    no_go_sha = sha256(NO_GO)
    refresh_human_routing(
        candidate_id=candidate["candidate_id"],
        no_go_sha256=no_go_sha,
        generated_at_utc=now,
    )

    policy = load(ROOT / "design/FAST_LOOP_POLICY.json")
    for key in (
        "active_repair_authorization",
        "architecture_proposal_authorization",
        "selected_replacement_contract",
    ):
        policy[key].update(
            {
                "implementation_authorized": False,
                "operator_approval_consumed": True,
                "status": "authorization_consumed_rejected_focused_discriminator",
                "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
                "result_binding": NO_GO.relative_to(ROOT).as_posix(),
                "result_binding_sha256": no_go_sha,
            }
        )
    dump(ROOT / "design/FAST_LOOP_POLICY.json", policy)

    target = load(ROOT / "design/TARGET.json")
    target["current_stage"] = "rtl"
    target["current_architecture_contract"] = copy.deepcopy(
        policy["selected_replacement_contract"]
    )
    target["fast_loop_contract"]["manager_recommendation"] = (
        "reroute_rtl_to_architecture_after_native_tagged_focused_no_go"
    )
    for key in ("active_repair_authorization", "architecture_proposal_authorization"):
        target["fast_loop_contract"][key] = copy.deepcopy(policy[key])
    target["generated_at_utc"] = now
    dump(ROOT / "design/TARGET.json", target)

    scope = load(ROOT / "design/CHIP_SCOPE.json")
    scope["stage"] = {
        "current_stage": "rtl",
        "current_stage_status": "candidate_rejected_manager_reroute_required",
        "downstream_stages_locked_until_manager_advance": [
            "verification",
            "ppa",
            "prototype",
            "benchmark",
            "signoff",
        ],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    scope["authority_override"].update(
        {
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "required_next_action": "manager_reroute_rtl_to_architecture",
            "result_binding": NO_GO.relative_to(ROOT).as_posix(),
            "result_binding_sha256": no_go_sha,
        }
    )
    scope["implementation_frontier"]["latest_decision"] = (
        "native_accumulator_tagged_focused_discriminator_bounded_no_go"
    )
    scope["implementation_frontier"]["first_unsupported_layer_operator"]["reason"] = (
        "The accepted prefix remains through layer_0.v_proj; the all-layer native-accumulator tagged successor failed the frozen C4 lm_head focused discriminator."
    )
    scope["operator_owned_execution_policy"]["active_successor_contract"] = copy.deepcopy(
        policy["selected_replacement_contract"]
    )
    scope["authority_override"]["operator_implementation_approval"] = copy.deepcopy(
        policy["selected_replacement_contract"]
    )
    refresh_live_contract_copies(scope, policy["selected_replacement_contract"])
    dump(ROOT / "design/CHIP_SCOPE.json", scope)

    manifest.update(
        {
            "stage": "rtl",
            "current_stage": "rtl",
            "architecture_contract_status": f"{CONTRACT}_rejected_focused_discriminator_manager_reroute_required",
            "candidate_status": "authorization_consumed_rejected_focused_discriminator",
            "candidate_meets_numeric_acceptance": False,
            "candidate_requires_fresh_sky130_ppa": False,
            "candidate_verification_complete": False,
            "candidate_review_binding": {
                "candidate_capability_accepted": False,
                "decision": "bounded_no_go",
                "evidence": NO_GO.relative_to(ROOT).as_posix(),
                "evidence_sha256": no_go_sha,
                "focused_discriminator_passed": False,
                "paired_smoke_run": False,
                "stage_closing": False,
            },
        }
    )
    manifest["candidate_verification_binding"] = {
        "status": "standalone_rtl_pass_focused_discriminator_failed_candidate_rejected",
        "evidence": [artifact(NO_GO), artifact(FOCUSED), artifact(FOCUSED.parent / "SHA256SUMS")],
    }
    manifest["proposed_replacement_contract"].update(
        {
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "status": "rejected_focused_discriminator",
            "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
            "result_binding": NO_GO.relative_to(ROOT).as_posix(),
            "result_binding_sha256": no_go_sha,
        }
    )
    manifest["traceability"]["architecture_contract_gap"] = {
        "resolution_owner": "Manager",
        "status": "candidate_rejected_manager_architecture_reroute_required",
    }
    manifest["traceability"]["rtl.contract-traceability"] = False
    manifest["traceability"]["stage_checklist"] = {
        "rtl.contract-traceability": False,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
    }
    manifest["latest_evidence"]["native_accumulator_tagged_bounded_no_go"] = artifact(NO_GO)
    manifest["claim_boundaries"] = [
        "The native-accumulator tagged candidate remains a disciplined standalone RTL implementation but failed its all-layer focused quality discriminator and is not accepted capability.",
        "The accepted prefix remains through layer_0.v_proj and layer_0.rope_q remains first unsupported.",
        "No paired smoke, full-shell regression, candidate PPA, prototype, full benchmark, signoff, tapeout, or silicon result was run or claimed for this candidate.",
        "The historical 0.6108746272 mm2 and +0.1502 ns at 100 MHz frontier remains unchanged.",
    ]
    manifest["generated_at_utc"] = now
    dump(ROOT / "design/RTL_MANIFEST.json", manifest)

    oracle = load(ROOT / "reference/ORACLE_MANIFEST.json")
    oracle["proposed_architecture_contract"] = copy.deepcopy(
        policy["selected_replacement_contract"]
    )
    oracle["generated_at_utc"] = now
    dump(ROOT / "reference/ORACLE_MANIFEST.json", oracle)

    status = load(ROOT / "research/PUBLIC_STATUS.json")
    status["stage"].update(
        {
            "current_stage": "rtl",
            "current_stage_status": "candidate_rejected_manager_reroute_required",
            "current_stage_checklist": {
                "rtl.contract-traceability": False,
                "rtl.hardware-discipline": True,
                "rtl.ip-provenance": True,
            },
            "current_stage_evidence": [
                "design/RTL_MANIFEST.json",
                "design/RTL_TRACEABILITY.md",
                NO_GO.relative_to(ROOT).as_posix(),
            ],
        }
    )
    status["latest_decision"] = (
        "native_accumulator_tagged_focused_discriminator_bounded_no_go"
    )
    status["selected_replacement_contract"] = {
        "contract_id": CONTRACT,
        "candidate_id": candidate["candidate_id"],
        "candidate_rtl_hash": candidate["candidate_rtl_hash"],
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "status": "rejected_focused_discriminator",
        "result_binding": NO_GO.relative_to(ROOT).as_posix(),
        "result_binding_sha256": no_go_sha,
    }
    status["architecture_proposal_gate"].update(
        {
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
            "required_operator_action": "none_standing_authorization_was_consumed_by_bounded_attempt",
            "result_binding": NO_GO.relative_to(ROOT).as_posix(),
            "result_binding_sha256": no_go_sha,
            "status": "rejected_focused_discriminator",
        }
    )
    status["blockers"] = [
        {
            "id": "manager_architecture_reroute_required",
            "stage": "rtl",
            "status": "active",
            "reason": "The all-layer native-accumulator tagged candidate failed the frozen C4 final-lm_head improvement requirement before paired smoke.",
            "required_resolution": "Manager rolls rtl back to architecture for one structurally distinct successor.",
            "evidence": NO_GO.relative_to(ROOT).as_posix(),
        }
    ]
    latest_quality = {
        "evidence": NO_GO.relative_to(ROOT).as_posix(),
        "focused_discriminator_passed": False,
        "paired_smoke_run": False,
        "comparisons": comparisons,
        "failed_requirement": "c4_en_512.lm_head.strict_relative_l2_improvement",
    }
    latest_rtl = {
        "candidate_id": candidate["candidate_id"],
        "contract_id": CONTRACT,
        "evidence": NO_GO.relative_to(ROOT).as_posix(),
        "rtl_hash": candidate["candidate_rtl_hash"],
        "rtl_hash_scope": "standalone_candidate_rtl_with_all_layer_quality_model_binding",
        "stage_closing": False,
        "status": "rejected_focused_discriminator_c4_lm_head_regressed",
    }
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status[name]
        container["operator_policy"] = copy.deepcopy(policy)
        if isinstance(container.get("candidate_mechanism"), dict):
            container["candidate_mechanism"].update(
                {
                    "implementation_authorized": False,
                    "operator_approval_consumed": True,
                    "required_manager_action": "reroute_rtl_to_architecture_for_structurally_distinct_contract",
                    "result_binding": NO_GO.relative_to(ROOT).as_posix(),
                    "result_binding_sha256": no_go_sha,
                    "status": "rejected_focused_discriminator",
                }
            )
        container.update(
            {
                "current_stage": "rtl",
                "current_mode": "ADVANCE",
                "mode": "ADVANCE",
                "latest_decision": status["latest_decision"],
                "routing_status": "manager_architecture_reroute_required",
                "required_manager_action": "reroute_rtl_to_architecture",
                "rtl_contract_traceability": False,
                "ordered_supported_layer_operator_prefix": PREFIX,
                "supported_layer_operator_prefix": PREFIX,
                "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
                "latest_quality_diagnostic": latest_quality,
                "latest_rtl_candidate": latest_rtl,
                "latest_ppa_frontier_status": "historical_frontier_preserved_no_candidate_ppa",
            }
        )
        if isinstance(container.get("latest_ppa_frontier"), dict):
            container["latest_ppa_frontier"].update(
                {
                    "decision": "preserve_historical_hash_bound_ppa_after_native_tagged_failed_pre_smoke_quality_gate",
                    "status": "historical_preserved_after_native_tagged_failed_pre_ppa_gate",
                    "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
                    "ordered_supported_layer_operator_prefix": PREFIX,
                    "mode": "ADVANCE",
                    "candidate_ppa_run": False,
                }
            )
    refresh_live_contract_copies(status, policy["selected_replacement_contract"])
    status["current_mode"] = "ADVANCE"
    status["supported_layer_operator_prefix"] = PREFIX
    status["first_unsupported_layer_operator"] = FIRST_UNSUPPORTED
    status["latest_ppa_frontier_status"] = "historical_frontier_preserved_no_candidate_ppa"
    existing_paths = {
        item["path"]
        for item in status.get("artifact_hashes", [])
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and (ROOT / item["path"]).is_file()
    }
    existing_paths.update(
        {
            NO_GO.relative_to(ROOT).as_posix(),
            FOCUSED.relative_to(ROOT).as_posix(),
            (FOCUSED.parent / "SHA256SUMS").relative_to(ROOT).as_posix(),
            "tools/ace2_full_model_fixed_point.py",
            "tools/localize_score_to_lm_head.py",
            "tools/bind_native_accumulator_tagged_no_go.py",
            "verification/test_native_accumulator_tagged_full_model.py",
            *QUALITY_LOGS,
        }
    )
    status["artifact_hashes"] = [
        artifact(ROOT / relative) for relative in sorted(existing_paths)
    ]
    status["generated_at_utc"] = now
    status["last_updated_utc"] = now
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": "UTF-8 sorted keys two-space indentation trailing newline integrity hash null during hash",
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump(ROOT / "research/PUBLIC_STATUS.json", status)

    print(
        "ACE2_NATIVE_TAGGED_BOUNDED_NO_GO "
        f"candidate_id={candidate['candidate_id']} no_go_sha256={no_go_sha}"
    )


if __name__ == "__main__":
    main()
