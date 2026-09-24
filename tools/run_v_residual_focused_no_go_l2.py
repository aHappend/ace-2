#!/usr/bin/env python3
"""Independently bind the frozen V-residual focused-discriminator no-go."""

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
FOCUSED_DECISION_SHA256 = "e5f018f90e8c0b65858572fc9fb21c665094178161608ac0712902b9eaec15a6"
PRIOR_RTL_REVIEW_SHA256 = "f41815862113092740672303d60a9ee25f6b487744833f418a5af3f9d8c97c33"

LATEST = ROOT / "evidence" / CONTRACT / "latest"
PRECHECK = LATEST / "PRECHECK.json"
FOCUSED_DECISION = LATEST / "VERIFICATION_DECISION.json"
BASELINE = ROOT / "verification/raw/latest/v_focused_baseline_replay/results.json"
CANDIDATE = ROOT / "verification/raw/latest/v_focused_candidate_replay/results.json"
HISTORICAL_BASELINE = (
    ROOT
    / "evidence/shared_qk_residual_cross_term_attention_v1"
    / "focused-baseline-all-layer-20260801-v1/results.json"
)
RTL_REVIEW_DIR = (
    ROOT / "evidence/review/rtl_checklist_shared_v_residual_value_correction_attention_v1"
)
PRIOR_RTL_REVIEW = RTL_REVIEW_DIR / "decision.json"
PRIOR_RTL_REVIEW_ARCHIVE = RTL_REVIEW_DIR / "rtl_stage_closing_decision.json"
REVIEW_DIR = (
    ROOT / "evidence/review/focused_verification_shared_v_residual_value_correction_attention_v1"
)
OUT = REVIEW_DIR / "decision.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
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


def verify_canonical(value: dict[str, Any], label: str) -> None:
    expected = value.get("integrity", {}).get("canonical_sha256")
    require(expected == canonical_sha256(value), f"{label} canonical SHA-256 differs")


def verify_sha256s(result_path: Path) -> None:
    manifest = result_path.parent / "SHA256SUMS"
    require(manifest.is_file(), f"missing SHA256SUMS beside {result_path}")
    expected, separator, name = manifest.read_text(encoding="utf-8").strip().partition("  ")
    require(separator == "  " and name == result_path.name, f"malformed {manifest}")
    require(expected == sha256(result_path), f"SHA256SUMS mismatch for {result_path}")


def source_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = result.get("sources")
    require(isinstance(sources, list), "focused result sources are missing")
    mapped = {str(item.get("path")): item for item in sources}
    require(len(mapped) == len(sources), "focused result source paths are not unique")
    return mapped


def verify_bound_source(
    baseline_sources: dict[str, dict[str, Any]],
    candidate_sources: dict[str, dict[str, Any]],
    relative: str,
) -> dict[str, Any]:
    path = ROOT / relative
    current = artifact(path)
    require(baseline_sources.get(relative) == current, f"baseline source binding differs: {relative}")
    require(candidate_sources.get(relative) == current, f"candidate source binding differs: {relative}")
    return current


def recompute_comparisons(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in ("wikitext2", "c4_en_512"):
        baseline_metrics = baseline["comparisons"][dataset]
        candidate_metrics = candidate["comparisons"][dataset]
        baseline_score = baseline_metrics["model.layers.0.score"]["relative_l2_error"]
        candidate_score = candidate_metrics["model.layers.0.score"]["relative_l2_error"]
        baseline_value = baseline_metrics["model.layers.0.attention_value"]["relative_l2_error"]
        candidate_value = candidate_metrics["model.layers.0.attention_value"]["relative_l2_error"]
        baseline_lm = baseline_metrics["lm_head"]["relative_l2_error"]
        candidate_lm = candidate_metrics["lm_head"]["relative_l2_error"]
        comparisons[dataset] = {
            "layer0_score_strict_improvement": {
                "id": f"{dataset}.layer0_score_strict_improvement",
                "passed": candidate_score < baseline_score,
                "baseline_relative_l2": baseline_score,
                "candidate_relative_l2": candidate_score,
                "delta_relative_l2": candidate_score - baseline_score,
            },
            "layer0_attention_value_strict_improvement": {
                "id": f"{dataset}.layer0_attention_value_strict_improvement",
                "passed": candidate_value < baseline_value,
                "baseline_relative_l2": baseline_value,
                "candidate_relative_l2": candidate_value,
                "delta_relative_l2": candidate_value - baseline_value,
            },
            "lm_head_strict_improvement": {
                "id": f"{dataset}.lm_head_strict_improvement",
                "passed": candidate_lm < baseline_lm,
                "baseline_relative_l2": baseline_lm,
                "candidate_relative_l2": candidate_lm,
                "delta_relative_l2": candidate_lm - baseline_lm,
            },
        }
    return comparisons


def main() -> int:
    require(sha256(FOCUSED_DECISION) == FOCUSED_DECISION_SHA256, "authoritative focused decision changed")
    require(sha256(ROOT / "rtl/ace2_v_residual_value_correction_core.sv") == RTL_SHA256, "live RTL changed")

    precheck = load(PRECHECK)
    focused = load(FOCUSED_DECISION)
    baseline = load(BASELINE)
    candidate = load(CANDIDATE)
    historical = load(HISTORICAL_BASELINE)
    prior_review = load(PRIOR_RTL_REVIEW)

    require(prior_review.get("reviewer_role") == "independent_l2", "prior review role differs")
    require(prior_review.get("stage") == "rtl", "prior review is not the RTL-stage decision")
    require(prior_review.get("quality_discriminator_complete") is False, "focused review already replaced prior review")
    require(
        sha256(PRIOR_RTL_REVIEW) == PRIOR_RTL_REVIEW_SHA256,
        "prior RTL review SHA-256 differs",
    )
    require(
        PRIOR_RTL_REVIEW.read_bytes() == PRIOR_RTL_REVIEW_ARCHIVE.read_bytes(),
        "authoritative RTL review differs from its archive copy",
    )

    require(precheck.get("contract_id") == CONTRACT, "precheck contract differs")
    require(precheck.get("candidate_rtl_hash") == CANDIDATE_HASH, "precheck candidate hash differs")
    require(precheck.get("candidate_rtl_source", {}).get("sha256") == RTL_SHA256, "precheck RTL SHA-256 differs")
    verify_canonical(precheck, "precheck")

    require(focused.get("contract_id") == CONTRACT, "focused decision contract differs")
    require(focused.get("candidate_rtl_hash") == CANDIDATE_HASH, "focused decision candidate hash differs")
    require(focused.get("decision") == "bounded_no_go_focused_discriminator_failed", "focused decision is not bounded NO_GO")
    require(focused.get("accepted_capability") is False, "focused decision accepts the candidate")
    require(focused.get("accepted_frontier_changed") is False, "focused decision changes the frontier")
    require(focused.get("manager_owned_stage") == "verification", "focused decision stage differs")
    require(all(value is False for value in focused.get("downstream_runs", {}).values()), "prohibited downstream run recorded")
    verify_canonical(focused, "focused decision")

    for path in (BASELINE, CANDIDATE, HISTORICAL_BASELINE):
        verify_sha256s(path)
    require(artifact(BASELINE) == focused["focused_discriminator"]["fresh_baseline_replay"], "baseline artifact binding differs")
    require(artifact(CANDIDATE) == focused["focused_discriminator"]["fresh_candidate_replay"], "candidate artifact binding differs")
    require(artifact(HISTORICAL_BASELINE) == focused["focused_discriminator"]["historical_authoritative_baseline"], "historical baseline binding differs")

    require(baseline.get("capture_token_limit") == 128, "baseline token limit differs")
    require(candidate.get("capture_token_limit") == 128, "candidate token limit differs")
    require(baseline.get("diagnostic_rope_mechanism") == "shared_v_residual_value_correction_baseline_v1", "baseline mechanism differs")
    require(candidate.get("diagnostic_rope_mechanism") == CONTRACT, "candidate mechanism differs")
    require(baseline.get("input_observations") == candidate.get("input_observations"), "baseline/candidate inputs differ")
    require(baseline.get("input_observations") == historical.get("input_observations"), "authoritative baseline inputs differ")
    require(baseline.get("comparisons") == historical.get("comparisons"), "authoritative baseline metrics differ")
    require(baseline.get("runtime") == candidate.get("runtime"), "baseline/candidate runtime differs")

    expected_runtime = {
        "device": "cpu",
        "packages": {
            "datasets": "4.8.5",
            "lm_eval": "0.4.9.2",
            "torch": "2.11.0",
            "transformers": "4.57.6",
        },
        "torch": "2.11.0+cu130",
    }
    require(baseline.get("runtime") == expected_runtime, "frozen runtime binding differs")

    for label, result in (("baseline", baseline), ("candidate", candidate)):
        contract = result.get("v_residual_contract", {})
        require(contract.get("contract_id") == CONTRACT, f"{label} contract differs")
        require(contract.get("layer_scope") == "all_24_layers", f"{label} layer scope differs")
        require(contract.get("all_baseline_equality_checks_passed") is True, f"{label} baseline equality failed")
        layers = contract.get("layers", [])
        require([row.get("layer") for row in layers] == list(range(24)), f"{label} layer rows differ")
        require(
            contract.get("baseline_equality_boundaries")
            == [
                "q_projection_int8",
                "k_projection_int8",
                "v_projection_int8",
                "q_absolute_rope_int8",
                "k_absolute_rope_int8",
                "base_score_q20_44",
                "softmax_probability_q0_15",
            ],
            f"{label} baseline equality boundaries differ",
        )
        for row in layers:
            require(all(count > 0 for count in row.get("baseline_equality_checks", {}).values()), f"{label} layer equality count is empty")

    metadata_path = ROOT / "reference/generated/v_residual_scale32_metadata.json"
    metadata = load(metadata_path)
    require(metadata.get("contract_id") == CONTRACT, "metadata contract differs")
    require(metadata.get("calibration_provenance", {}).get("quality_metrics_executed") is False, "metadata calibration ran quality metrics")
    require(
        [(row.get("layer"), row.get("kv_head")) for row in metadata.get("records", [])]
        == [(layer, head) for layer in range(24) for head in range(2)],
        "metadata is not the frozen ordered 48-row table",
    )
    verify_canonical(metadata, "metadata")

    hook_path = ROOT / "reference/v_residual_value_correction_full_model_hook.json"
    hook = load(hook_path)
    require(hook.get("candidate_contract") == CONTRACT, "full-model hook contract differs")
    require(hook.get("layer_scope") == "all_24_layers", "full-model hook layer scope differs")
    require(hook.get("required_runner") == "tools/ace2_full_model_fixed_point.py", "full-model runner differs")
    require(hook.get("scalar_reference") == "tools/ace2_v_residual_value_correction_reference.py", "scalar reference differs")

    vectors_path = ROOT / "verification/generated/v_residual_value_correction_vectors.json"
    vectors_svh_path = ROOT / "verification/generated/v_residual_value_correction_vectors.svh"
    vectors = load(vectors_path)
    require(vectors.get("contract_id") == CONTRACT, "vector contract differs")
    require(len(vectors.get("projection_cases", [])) == 4, "projection vector count differs")
    require(len(vectors.get("correction_cases", [])) == 6, "correction vector count differs")

    baseline_sources = source_map(baseline)
    candidate_sources = source_map(candidate)
    bound_sources = {}
    for relative in (
        "tools/ace2_full_model_fixed_point.py",
        "tools/localize_score_to_lm_head.py",
        "tools/ace2_v_residual_value_correction_reference.py",
        "reference/generated/v_residual_scale32_metadata.json",
        "reference/v_residual_value_correction_full_model_hook.json",
        "rtl/ace2_v_residual_value_correction_core.sv",
        "evidence/shared_v_residual_value_correction_attention_v1/latest/PRECHECK.json",
    ):
        bound_sources[relative] = verify_bound_source(baseline_sources, candidate_sources, relative)

    comparisons = recompute_comparisons(baseline, candidate)
    require(comparisons == focused["focused_discriminator"]["comparisons"], "focused mandatory metrics differ")
    failures = [
        item["id"]
        for dataset in comparisons.values()
        for item in dataset.values()
        if not item["passed"]
    ]
    require(
        failures
        == [
            "wikitext2.layer0_score_strict_improvement",
            "c4_en_512.layer0_score_strict_improvement",
            "c4_en_512.lm_head_strict_improvement",
        ],
        "focused failure set differs",
    )
    require(focused["focused_discriminator"].get("failed_conditions") == failures, "recorded failure set differs")
    require(focused["focused_discriminator"].get("passed") is False, "focused gate is marked passed")

    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "reviewed_at_utc": now,
        "reviewer_role": "independent_l2",
        "reviewer_status": "done",
        "stage": "verification",
        "scope": "focused_discriminator_bounded_no_go_only",
        "stage_closing": False,
        "manager_stage_transition_owner": "Manager",
        "contract_id": CONTRACT,
        "candidate_id": focused["candidate_id"],
        "candidate_rtl_hash": CANDIDATE_HASH,
        "live_rtl_source_sha256": RTL_SHA256,
        "decision": "accept_bounded_no_go",
        "candidate_capability_accepted": False,
        "quality_discriminator_complete": True,
        "focused_discriminator_passed": False,
        "reason": (
            "Independent evidence-only L2 recomputation accepts the bounded NO_GO. "
            "WikiText-2 layer-0 score relative-L2 remains 0.15514931718110408 and C4-en layer-0 "
            "score relative-L2 remains 0.18302257326653032, so both mandatory score metrics fail "
            "strict improvement. Layer-0 attention-value relative-L2 improves on both datasets and "
            "WikiText-2 final lm_head improves, but C4-en final lm_head regresses from "
            "1.1934673932274116 to 1.1979960520084465. The operator-owned all-metrics-on-both-datasets "
            "gate therefore fails. No focused rerun or downstream work was performed."
        ),
        "authoritative_focused_decision": artifact(FOCUSED_DECISION),
        "prior_rtl_stage_review": {
            **artifact(PRIOR_RTL_REVIEW),
            "archive_copy": artifact(PRIOR_RTL_REVIEW_ARCHIVE),
            "path_status": "authoritative_immutable_rtl_stage_closing_verdict",
        },
        "focused_inputs": {
            "historical_authoritative_baseline": artifact(HISTORICAL_BASELINE),
            "fresh_baseline_replay": artifact(BASELINE),
            "fresh_candidate_replay": artifact(CANDIDATE),
            "input_observations": baseline["input_observations"],
            "runtime": baseline["runtime"],
        },
        "mandatory_metrics": comparisons,
        "failed_conditions": failures,
        "bindings": {
            "precheck": artifact(PRECHECK),
            "metadata": artifact(metadata_path),
            "full_model_hook": artifact(hook_path),
            "vectors_json": artifact(vectors_path),
            "vectors_svh": artifact(vectors_svh_path),
            "focused_sources": bound_sources,
        },
        "contract_preservation": {
            "ordered_supported_layer_operator_prefix": [
                "layer_0.input_rmsnorm",
                "layer_0.q_proj",
                "layer_0.k_proj",
                "layer_0.v_proj",
            ],
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "non_sram_area_cap_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "abstract_streaming_memory_width_bits": 128,
            "sealed_predecessor_contract_id": "shared_qk_residual_cross_term_attention_v1",
            "historical_ppa_frontier": {
                "cells": 62199,
                "non_sram_area_mm2": 0.6108746272,
                "setup_slack_ns_at_100mhz": 0.1502,
                "candidate_ppa_claim": False,
            },
        },
        "downstream_runs": copy.deepcopy(focused["downstream_runs"]),
        "required_manager_action": "rollback_verification_to_architecture_for_structurally_distinct_successor",
        "claim_boundary": (
            "Independent L2 acceptance of the focused bounded NO_GO only; no capability, shell, PPA, "
            "prototype, benchmark, signoff, tapeout, or silicon acceptance."
        ),
        "integrity": {"canonical_sha256": None},
    }
    payload["integrity"]["canonical_sha256"] = canonical_sha256(payload)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUT)
    print(
        "ACE2_V_RESIDUAL_FOCUSED_NO_GO_L2_PASS "
        f"decision_sha256={sha256(OUT)} "
        f"failed_conditions={','.join(failures)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
