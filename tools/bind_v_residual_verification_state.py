#!/usr/bin/env python3
"""Reconcile project/dashboard state to the bound V-residual verification decision."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_v_residual_value_correction_attention_v1"
CANDIDATE_HASH = "a7eebc3e120268d6f864f2dfa8f4f4adb5b7465c3c38a99d5a90a65ae35a1c7f"
RTL_SHA256 = "73e39521e7a21c7ae2594abcc48946cb0efd00bf0be14c5c21a8dbfe5edee87c"
RTL_REVIEW_SHA256 = "f41815862113092740672303d60a9ee25f6b487744833f418a5af3f9d8c97c33"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
DECISION = ROOT / "evidence" / CONTRACT / "latest" / "VERIFICATION_DECISION.json"
RTL_REVIEW_DIR = (
    ROOT / "evidence/review/rtl_checklist_shared_v_residual_value_correction_attention_v1"
)
RTL_REVIEW = RTL_REVIEW_DIR / "decision.json"
RTL_REVIEW_ARCHIVE = RTL_REVIEW_DIR / "rtl_stage_closing_decision.json"
FOCUSED_L2_REVIEW = (
    ROOT
    / "evidence/review/focused_verification_shared_v_residual_value_correction_attention_v1"
    / "decision.json"
)
RESULTS = ROOT / "verification" / "RESULTS.json"
ORACLE_MANIFEST = ROOT / "reference" / "ORACLE_MANIFEST.json"
PROPERTY_MANIFEST = ROOT / "formal" / "ACE2_VERIFICATION_PROPERTIES.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def dump(path: Path, value: Any) -> None:
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
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(clone, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def bind_existing_state() -> dict[str, Any]:
    pipeline = load(ROOT / "research/PIPELINE_STATE.json")
    if pipeline.get("current_stage") != "verification":
        raise RuntimeError("current stage is not verification")
    decision = load(DECISION)
    results = load(RESULTS)
    rtl_review = load(RTL_REVIEW)
    focused_l2_review = load(FOCUSED_L2_REVIEW)
    if decision.get("contract_id") != CONTRACT or results.get("candidate_rtl_hash") != decision.get(
        "candidate_rtl_hash"
    ):
        raise RuntimeError("verification decision/results binding differs")
    if decision.get("decision") != "bounded_no_go_focused_discriminator_failed":
        raise RuntimeError("state binder is limited to the current bounded no-go")
    require(decision.get("candidate_rtl_hash") == CANDIDATE_HASH, "focused decision candidate differs")
    require(
        decision.get("standalone_verification", {}).get("independent_rtl_review")
        == artifact(RTL_REVIEW),
        "focused decision does not bind the authoritative RTL-stage review",
    )
    require(sha256(RTL_REVIEW) == RTL_REVIEW_SHA256, "authoritative RTL-stage review changed")
    require(
        RTL_REVIEW.read_bytes() == RTL_REVIEW_ARCHIVE.read_bytes(),
        "authoritative RTL-stage review differs from its archive copy",
    )
    require(
        rtl_review.get("stage") == "rtl"
        and rtl_review.get("stage_closing") is True
        and rtl_review.get("quality_discriminator_complete") is False,
        "RTL-stage review role differs",
    )
    require(
        rtl_review.get("candidate_rtl_hash") == CANDIDATE_HASH
        and rtl_review.get("live_rtl_source_sha256") == RTL_SHA256,
        "RTL-stage review candidate binding differs",
    )
    require(
        focused_l2_review.get("stage") == "verification"
        and focused_l2_review.get("stage_closing") is False
        and focused_l2_review.get("decision") == "accept_bounded_no_go",
        "focused L2 review role differs",
    )
    require(
        focused_l2_review.get("candidate_rtl_hash") == CANDIDATE_HASH
        and focused_l2_review.get("live_rtl_source_sha256") == RTL_SHA256,
        "focused L2 review candidate binding differs",
    )
    require(
        focused_l2_review.get("integrity", {}).get("canonical_sha256")
        == canonical_sha256(focused_l2_review),
        "focused L2 canonical SHA-256 differs",
    )
    require(
        focused_l2_review.get("authoritative_focused_decision") == artifact(DECISION),
        "focused L2 authoritative decision binding differs",
    )
    decision_sha = sha256(DECISION)
    now = utc_now()
    manager_action = "rollback_verification_to_architecture_for_structurally_distinct_successor"
    latest_decision = "v_residual_focused_discriminator_bounded_no_go"

    generator = ROOT / "tools/gen_v_residual_value_correction_vectors.py"
    reference = ROOT / "tools/ace2_v_residual_value_correction_reference.py"
    vector_json = ROOT / "verification/generated/v_residual_value_correction_vectors.json"
    vector_svh = ROOT / "verification/generated/v_residual_value_correction_vectors.svh"
    baseline = ROOT / "verification/raw/latest/v_focused_baseline_replay/results.json"
    candidate = ROOT / "verification/raw/latest/v_focused_candidate_replay/results.json"
    require(
        focused_l2_review.get("focused_inputs", {}).get("fresh_baseline_replay")
        == artifact(baseline),
        "focused L2 baseline artifact binding differs",
    )
    require(
        focused_l2_review.get("focused_inputs", {}).get("fresh_candidate_replay")
        == artifact(candidate),
        "focused L2 candidate artifact binding differs",
    )
    rtl_review_artifact = artifact(RTL_REVIEW)
    focused_l2_artifact = artifact(FOCUSED_L2_REVIEW)
    focused_decision_artifact = artifact(DECISION)
    focused_comparisons = copy.deepcopy(
        decision["focused_discriminator"]["comparisons"]
    )
    failed_conditions = copy.deepcopy(
        decision["focused_discriminator"]["failed_conditions"]
    )
    expected_failures = [
        "wikitext2.layer0_score_strict_improvement",
        "c4_en_512.layer0_score_strict_improvement",
        "c4_en_512.lm_head_strict_improvement",
    ]
    require(failed_conditions == expected_failures, "focused decision failure set differs")
    require(
        focused_l2_review.get("mandatory_metrics") == focused_comparisons
        and focused_l2_review.get("failed_conditions") == failed_conditions,
        "focused L2 mandatory metric projection differs",
    )
    oracle = {
        "schema_version": 1,
        "oracles": [
            {
                "case_count": 10,
                "numeric_acceptance": "bit_exact_fixed_point_vector_match",
                "generator": generator.relative_to(ROOT).as_posix(),
                "generator_sha256": sha256(generator),
                "reference": reference.relative_to(ROOT).as_posix(),
                "reference_sha256": sha256(reference),
                "vector_json": vector_json.relative_to(ROOT).as_posix(),
                "vector_json_sha256": sha256(vector_json),
                "vector_svh": vector_svh.relative_to(ROOT).as_posix(),
                "vector_svh_sha256": sha256(vector_svh),
            }
        ],
        "verification_evidence": {
            "generated_at_utc": decision["bound_at_utc"],
            "contract_id": CONTRACT,
            "candidate_id": decision["candidate_id"],
            "candidate_rtl_hash": decision["candidate_rtl_hash"],
            "claim_boundary": "Independent scalar/tensor/RTL and focused-quality verification only; no downstream claim.",
            "runner": artifact(ROOT / "tools/ace2_full_model_fixed_point.py"),
            "trace": artifact(ROOT / "tools/localize_score_to_lm_head.py"),
            "hook": artifact(ROOT / "reference/v_residual_value_correction_full_model_hook.json"),
            "metadata": artifact(ROOT / "reference/generated/v_residual_scale32_metadata.json"),
            "baseline": artifact(baseline),
            "candidate": artifact(candidate),
            "baseline_equality_passed_all_24_layers": True,
            "focused_comparisons": focused_comparisons,
            "quality_gate_passed": False,
        },
    }
    dump(ORACLE_MANIFEST, oracle)

    properties = load(PROPERTY_MANIFEST)
    for item in properties.get("properties", []):
        if item.get("id") == "reset_clear_backpressure":
            item["evidence"] = (
                "The correction core was active and the projection core idle when reset was "
                "asserted; both cores were then observed idle, with post-reset known outputs."
            )
        elif item.get("id") == "x_z_known_outputs":
            item["evidence"] = (
                "Known-output checks cover post-reset, post-backpressure, post-clear, and reset "
                "while correction was active and projection was idle."
            )
    properties["coverage_limitations"] = [
        "projection_busy_midflight_reset_not_exercised",
    ]
    properties["state_rebound_at_utc"] = now
    dump(PROPERTY_MANIFEST, properties)

    source_paths = [ROOT / item["path"] for item in results.get("source_hashes", [])]
    for path in (
        ROOT / "tools/run_v_residual_value_correction_verification.py",
        Path(__file__).resolve(),
    ):
        if path not in source_paths:
            source_paths.append(path)
    results["source_hashes"] = [artifact(path) for path in source_paths]
    results["oracle_manifest"] = artifact(ORACLE_MANIFEST)
    results["property_manifest"] = artifact(PROPERTY_MANIFEST)
    results["focused_l2_review"] = focused_l2_artifact
    results.setdefault("quality_gate", {}).update(
        {
            "passed": False,
            "comparisons": copy.deepcopy(focused_comparisons),
            "failed_conditions": copy.deepcopy(failed_conditions),
            "decision": focused_decision_artifact,
        }
    )
    results.setdefault("checklist_explanation", {})["verification.independent-oracle"] = (
        "Exact arithmetic and baseline equality pass. Mandatory strict improvement fails for "
        "both unchanged layer-0 score relative-L2 metrics and for the regressed C4-en final "
        "lm_head relative-L2 metric."
    )
    stress = results.setdefault("coverage", {}).setdefault("stress", [])
    results["coverage"]["stress"] = [
        (
            "reset_while_correction_active_projection_idle_both_cores_observed_idle_after_reset"
            if item == "reset_and_midflight_reset_projection_and_correction"
            else item
        )
        for item in stress
    ]
    results["coverage"]["limitations"] = [
        "projection_busy_midflight_reset_not_exercised",
    ]
    results.setdefault("checklist_explanation", {})["verification.coverage-stress"] = (
        "Standalone and all-layer stress coverage passed. Reset was asserted while the correction "
        "core was active and the projection core idle; projection-busy reset was not exercised. "
        "Four-state X/Z checks and the single-clock CDC disposition also passed."
    )
    results["state_rebound_at_utc"] = now
    results["state_binding"] = artifact(Path(__file__).resolve())
    dump(RESULTS, results)

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
                "required_manager_action": manager_action,
                "quality_discriminator_complete": True,
                "focused_discriminator_passed": False,
                "result_binding": DECISION.relative_to(ROOT).as_posix(),
                "result_binding_sha256": decision_sha,
            }
        )
    policy["manager_recommendation"] = manager_action
    dump(ROOT / "design/FAST_LOOP_POLICY.json", policy)

    target = load(ROOT / "design/TARGET.json")
    target["current_stage"] = "verification"
    target["current_architecture_contract"] = copy.deepcopy(
        policy["selected_replacement_contract"]
    )
    fast_loop = target["fast_loop_contract"]
    fast_loop["active_repair_authorization"] = copy.deepcopy(
        policy["active_repair_authorization"]
    )
    fast_loop["architecture_proposal_authorization"] = copy.deepcopy(
        policy["architecture_proposal_authorization"]
    )
    fast_loop["manager_recommendation"] = manager_action
    target["generated_at_utc"] = now
    dump(ROOT / "design/TARGET.json", target)

    scope = load(ROOT / "design/CHIP_SCOPE.json")
    scope["stage"].update(
        {
            "current_stage": "verification",
            "current_stage_status": "bounded_no_go_manager_rollback_required",
            "current_stage_checklist": copy.deepcopy(results["checklist"]),
            "current_stage_evidence": [
                "verification/RESULTS.json",
                DECISION.relative_to(ROOT).as_posix(),
            ],
            "downstream_stages_locked_until_manager_advance": [
                "ppa",
                "prototype",
                "benchmark",
                "signoff",
            ],
        }
    )
    scope["authority_override"].update(
        {
            "implementation_authorized": False,
            "operator_approval_consumed": True,
            "required_next_action": manager_action,
            "operator_implementation_approval": copy.deepcopy(
                policy["selected_replacement_contract"]
            ),
            "result_binding": DECISION.relative_to(ROOT).as_posix(),
            "result_binding_sha256": decision_sha,
        }
    )
    scope["numerical_behavior"]["active_v_residual_value_correction_contract"][
        "status"
    ] = "rejected_focused_discriminator"
    scope["numerical_behavior"]["attention_projection_scales"][
        "quality_status"
    ] = "focused_discriminator_failed_score_non_improvements_and_c4_lm_head_regression"
    scope["operator_owned_execution_policy"]["active_successor_contract"] = copy.deepcopy(
        policy["selected_replacement_contract"]
    )
    scope["implementation_frontier"].update(
        {
            "latest_decision": latest_decision,
            "supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        }
    )
    dump(ROOT / "design/CHIP_SCOPE.json", scope)

    manifest = load(ROOT / "design/RTL_MANIFEST.json")
    manifest["independent_reviewer_acceptance"][
        "status"
    ] = "rtl_checklist_accepted_manager_advanced_candidate_rejected_in_verification"
    manifest["independent_reviewer_acceptance"].update(
        {
            "quality_discriminator_complete": True,
            "focused_verification_review": focused_l2_artifact,
        }
    )
    manifest["candidate_review_binding"] = {
        "candidate_capability_accepted": False,
        "decision": "accepted_rtl_stage_closing_for_verification_entry",
        "evidence": rtl_review_artifact,
        "ppa_run": False,
        "quality_discriminator_complete": False,
        "reviewer_status": rtl_review["reviewer_status"],
        "scope": rtl_review["scope"],
        "shell_admitted": False,
        "stage_closing": True,
    }
    manifest.setdefault("candidate_verification_binding", {}).update(
        {
            "status": "bounded_no_go",
            "evidence": focused_decision_artifact,
            "independent_l2_review": focused_l2_artifact,
            "focused_discriminator_passed": False,
            "paired_smoke_run": False,
            "stage_closing": False,
        }
    )
    manifest.setdefault("latest_evidence", {})[
        "v_residual_value_correction_focused_verification_l2"
    ] = focused_l2_artifact
    manifest["latest_evidence"][
        "v_residual_value_correction_verification"
    ] = focused_decision_artifact
    manifest["proposed_replacement_contract"] = copy.deepcopy(
        policy["selected_replacement_contract"]
    )
    manifest["generated_at_utc"] = now
    dump(ROOT / "design/RTL_MANIFEST.json", manifest)

    status = load(ROOT / "research/PUBLIC_STATUS.json")
    status["architecture_proposal_gate"] = {
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "operator_approval_consumed": True,
        "required_manager_action": manager_action,
        "stage_closing": False,
        "status": "rejected_focused_discriminator",
        "result_binding": DECISION.relative_to(ROOT).as_posix(),
        "result_binding_sha256": decision_sha,
    }
    status["selected_replacement_contract"] = copy.deepcopy(
        policy["selected_replacement_contract"]
    )
    for blocker in status.get("blockers", []):
        if isinstance(blocker, dict) and blocker.get("id") == "v_residual_focused_quality_no_go":
            blocker["reason"] = "Focused failed conditions: " + ", ".join(failed_conditions)
            blocker["evidence"] = focused_decision_artifact["path"]
    verification_projection = {
        "status": "bounded_no_go",
        "checklist": copy.deepcopy(results["checklist"]),
        "failed_conditions": copy.deepcopy(failed_conditions),
        "results": "verification/RESULTS.json",
        "decision": focused_decision_artifact,
        "independent_l2_review": focused_l2_artifact,
        "reset_coverage": {
            "active_module_at_reset": "correction",
            "projection_state_at_reset": "idle",
            "post_reset_observation": "both_cores_idle_and_outputs_known",
            "limitation": "projection_busy_midflight_reset_not_exercised",
        },
    }
    rtl_candidate_projection = {
        "status": "rejected_focused_discriminator",
        "stage_closing": False,
        "rtl_review_stage_closing": True,
        "quality_discriminator_complete": True,
        "focused_discriminator_passed": False,
        "review": rtl_review_artifact,
        "verification_decision": focused_decision_artifact,
        "verification_l2_review": focused_l2_artifact,
    }
    if isinstance(status.get("latest_rtl_candidate"), dict):
        status["latest_rtl_candidate"].update(copy.deepcopy(rtl_candidate_projection))
    status["latest_verification_stage"] = copy.deepcopy(verification_projection)
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status.get(name)
        if not isinstance(container, dict):
            continue
        container.update(
            {
                "current_stage": "verification",
                "supported_layer_operator_prefix": PREFIX,
                "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
                "current_mode": "ADVANCE",
                "latest_decision": latest_decision,
                "routing_status": manager_action,
                "required_manager_action": manager_action,
                "required_operator_action": "none",
                "operator_policy": copy.deepcopy(policy),
                "verification_checklist": copy.deepcopy(results["checklist"]),
            }
        )
        if isinstance(container.get("candidate_mechanism"), dict):
            container["candidate_mechanism"]["status"] = "rejected_focused_discriminator"
        if isinstance(container.get("latest_rtl_candidate"), dict):
            container["latest_rtl_candidate"].update(copy.deepcopy(rtl_candidate_projection))
        container["latest_verification_stage"] = copy.deepcopy(verification_projection)
    projected_paths = {
        item.get("path") for item in status.get("artifact_hashes", []) if isinstance(item, dict)
    }
    for value in (rtl_review_artifact, focused_decision_artifact, focused_l2_artifact):
        if value["path"] not in projected_paths:
            status.setdefault("artifact_hashes", []).append(value)
    status["artifact_hashes"] = [
        artifact(ROOT / item["path"])
        for item in status.get("artifact_hashes", [])
        if isinstance(item, dict) and (ROOT / item.get("path", "")).is_file()
    ]
    status["public_claims"] = [
        {
            "claim": "the Manager-owned current stage remains verification",
            "evidence": ["research/PIPELINE_STATE.json"],
        },
        {
            "claim": "the immutable RTL-stage review accepted the exact V-residual candidate for verification entry",
            "evidence": [rtl_review_artifact["path"], "evidence/shared_v_residual_value_correction_attention_v1/latest/PRECHECK.json"],
        },
        {
            "claim": "focused verification produced a bounded NO_GO independently accepted by the verification-specific L2 review",
            "evidence": [focused_decision_artifact["path"], focused_l2_artifact["path"]],
        },
        {
            "claim": "the supported prefix and historical PPA frontier are unchanged, and no downstream run followed",
            "evidence": ["design/CHIP_SCOPE.json", focused_decision_artifact["path"]],
        },
    ]
    status["routing_status"] = manager_action
    status["routing_authorized"] = "manager_only"
    dump(ROOT / "research/PUBLIC_STATUS.json", status)

    rebound_manifest = load(ROOT / "design/RTL_MANIFEST.json")
    rebound_status = load(ROOT / "research/PUBLIC_STATUS.json")
    require(
        rebound_manifest.get("candidate_review_binding", {}).get("evidence")
        == rtl_review_artifact,
        "manifest RTL review projection differs",
    )
    require(
        rebound_manifest.get("candidate_verification_binding", {}).get("independent_l2_review")
        == focused_l2_artifact,
        "manifest focused L2 projection differs",
    )
    public_rtl_reviews = [
        rebound_status.get("latest_rtl_candidate", {}).get("review"),
        rebound_status.get("dashboard_fields", {}).get("latest_rtl_candidate", {}).get("review"),
        rebound_status.get("implementation_frontier", {}).get("latest_rtl_candidate", {}).get("review"),
    ]
    require(
        all(value == rtl_review_artifact for value in public_rtl_reviews),
        "public RTL review projection differs",
    )
    require(
        any(item == rtl_review_artifact for item in rebound_status.get("artifact_hashes", [])),
        "public RTL review artifact hash projection differs",
    )
    public_l2_reviews = [
        rebound_status.get("latest_verification_stage", {}).get("independent_l2_review"),
        rebound_status.get("dashboard_fields", {}).get("latest_verification_stage", {}).get("independent_l2_review"),
        rebound_status.get("implementation_frontier", {}).get("latest_verification_stage", {}).get("independent_l2_review"),
    ]
    require(
        all(value == focused_l2_artifact for value in public_l2_reviews),
        "public focused L2 projection differs",
    )
    require(
        "reset_and_midflight_reset_projection_and_correction"
        not in results.get("coverage", {}).get("stress", []),
        "broad reset claim remains in verification results",
    )

    return {
        "status": "pass",
        "decision_sha256": decision_sha,
        "rtl_review_sha256": rtl_review_artifact["sha256"],
        "focused_l2_sha256": focused_l2_artifact["sha256"],
        "manager_action": manager_action,
    }


def main() -> int:
    result = bind_existing_state()
    print(
        "ACE2_V_RESIDUAL_STATE_BIND_PASS "
        f"decision_sha256={result['decision_sha256']} "
        f"rtl_review_sha256={result['rtl_review_sha256']} "
        f"focused_l2_sha256={result['focused_l2_sha256']} "
        f"manager_action={result['manager_action']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
