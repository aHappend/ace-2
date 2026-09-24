#!/usr/bin/env python3
"""Fail-closed static audit for the non-consuming S7 specification package."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_SUCCESSOR_S7_DORA_SFT_DPO_SPECIFICATION_PACKAGE.json"
MANIFEST = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_SUCCESSOR_S7_PACKAGE_MANIFEST.json"
SPEC_SUM = SPEC.with_suffix(SPEC.suffix + ".sha256")
MANIFEST_SUM = MANIFEST.with_suffix(MANIFEST.suffix + ".sha256")
AUDIT = ROOT / "build/bf16-successor-s7-specification-package/package-audit.json"

ANCHORS = {
    "latest.json": "ac0a0560b586fc371306d3a1500f9a69d66fece051e71748a22118434934df5c",
    "research/FINAL_PRODUCT_CERTIFICATE.json": "a5d1ea92dff045b1887c7bfb12f1c8985d1dbeb4f6c3be097e650fdfd3bd9537",
    "design/QWEN25_05B_INSTRUCT_BF16_CATEGORY_BALANCED_CONFLICT_PROJECTED_SUCCESSOR_S6_CLEAN_ROOM_V2_DESIGN_FREEZE.json": "99bc106c2738a5cdfcba52485d2e4b6f61dac41ea74202bd884f00168070d6d5",
    "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json": "0da7d95e92637e29c0369ce9aa9c8d0fe9d50802cf1144bc79095b129962576d",
}

CATEGORY_GATES = {
    "instruction_following": {"minimum_hard_passes": 7, "total": 8},
    "safety_refusal": {"minimum_hard_passes": 8, "total": 8},
    "formatting": {"minimum_hard_passes": 7, "total": 8},
    "knowledge": {"minimum_hard_passes": 6, "total": 8},
    "reasoning": {"minimum_hard_passes": 6, "total": 8},
    "tool_use": {"minimum_hard_passes": 7, "total": 8},
    "arithmetic_regression": {"minimum_hard_passes": 4, "total": 4},
    "context_memory_regression": {"minimum_hard_passes": 4, "total": 4},
}

SOURCE_CLASSES = [
    "new_s7_synthetic_task_schemas",
    "public_licensed_factual_sources",
    "deterministic_programmatic_generators",
]

MARKER_POLICY_TRAINING_MATERIAL = ["phase_a_sft_rows", "phase_b_dpo_pairs"]

MODEL_VISIBLE_SURFACES = [
    "all system, developer, user, assistant, and tool message content",
    "Phase A SFT prompt and assistant-target bytes and token IDs",
    "Phase B DPO prompt, chosen-target, and rejected-target bytes and token IDs",
    "frozen chat-template rendering and source-model special-token insertion",
]

ALLOWED_MODEL_VISIBLE_CONTENT_ORIGINS = [
    "task_content.system",
    "task_content.developer",
    "task_content.user",
    "task_content.tool_name_description_and_schema",
    "task_content.tool_result",
    "answer_content.sft_target",
    "answer_content.dpo_chosen",
    "answer_content.dpo_rejected",
]

REQUIRED_SIDECAR_ONLY_FIELDS = {
    "record_id",
    "split",
    "stratum",
    "category",
    "subcategory",
    "task_family",
    "source_class",
    "source_uri_or_local_identity",
    "source_revision",
    "source_license",
    "source_sha256",
    "extractor_or_generator_path",
    "extractor_or_generator_sha256",
    "generator_config_sha256",
    "template_id",
    "canonical_content_sha256",
    "deduplication_report_sha256",
    "human_verification_status",
    "score",
    "reward",
    "preference_label",
    "chosen_label",
    "rejected_label",
    "gate",
    "threshold",
    "pass",
    "fail",
    "reviewer",
    "reviewer_id",
    "reviewer_rationale",
    "evaluator",
    "adjudication",
    "probe_id",
    "selector_digest",
    "seed",
}

REQUIRED_MARKER_ATTESTATION_FIELDS = [
    "policy_id",
    "source_manifest_sha256",
    "sft_manifest_sha256",
    "dpo_manifest_sha256",
    "renderer_sha256",
    "tokenizer_sha256",
    "origin_trace_report_sha256",
    "sidecar_canary_report_sha256",
    "rendered_marker_scan_report_sha256",
    "rows_checked",
    "pairs_checked",
    "violations",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def checksum_companion(path: Path) -> tuple[str, str]:
    fields = path.read_text(encoding="utf-8").strip().split()
    require(len(fields) == 2, f"invalid checksum companion: {path}")
    return fields[0], fields[1]


def main() -> int:
    for relative, expected in ANCHORS.items():
        require(sha256(ROOT / relative) == expected, f"source anchor drift: {relative}")

    spec = load_json(SPEC)
    manifest = load_json(MANIFEST)
    audit = load_json(AUDIT)

    require(spec["stage"] == "specification", "S7 is not specification-stage")
    require(
        spec["status"] == "FROZEN_CANDIDATE_PACKAGE_PENDING_FRESH_L2_NO_AUTHORITY",
        "unexpected S7 status",
    )
    evidence = spec["accepted_aggregate_terminal_evidence"]["values"]
    require(evidence["hard_pass_counts"] == [8, 7, 7], "S6 aggregate count drift")
    require(evidence["response_count_per_epoch"] == 28, "S6 response count drift")
    require(evidence["critical_safety_failures_each_epoch"] == [4, 4, 4], "S6 safety aggregate drift")
    require(evidence["category_minima_passed_each_epoch"] == ["arithmetic", "context_memory"], "S6 category aggregate drift")
    require(evidence["selected_checkpoint"] is None, "S6 checkpoint was asserted")
    require(evidence["official_dev_accessed"] is False, "official dev access was asserted")

    distinct = spec["structural_distinctness"]
    mechanism = distinct["s7_mechanism"]
    require("threshold-only change" in distinct["disallowed_retry_classes"], "threshold retry not prohibited")
    require("seed-only change or seed sweep" in distinct["disallowed_retry_classes"], "seed retry not prohibited")
    require("epoch-only extension or checkpoint cherry-pick" in distinct["disallowed_retry_classes"], "epoch retry not prohibited")
    require(mechanism["phase_a_sft"]["adapter"]["type"] == "DoRA", "Phase A is not DoRA")
    require(mechanism["phase_a_sft"]["optimizer"]["name"] == "Adafactor", "Phase A optimizer drift")
    require(mechanism["phase_b_dpo"]["loss"]["name"] == "reference-anchored DPO", "Phase B is not DPO")
    require(mechanism["phase_b_dpo"]["candidate_count"] == 1, "multiple candidates allowed")
    require(mechanism["phase_b_dpo"]["optimizer_steps"] == 32, "Phase B schedule drift")
    require("no router, adapter, special token" in mechanism["final_candidate_compatibility"], "final architecture compatibility absent")

    data = spec["data_provenance_contract"]
    future = data["future_materialization"]
    source_classes = [item["class"] for item in data["allowed_source_classes"]]
    require(source_classes == SOURCE_CLASSES, "allowed source classes drift")
    require(data["rows_created_by_this_package"] == 0, "data rows were asserted")
    require(future["status"] == "NOT_MATERIALIZED", "future data was materialized")
    require(future["sft_row_count"] == 448 and future["preference_pair_count"] == 256, "training counts drift")
    require(len(future["strata"]) == 8, "capability strata drift")
    contamination = data["contamination_controls"]
    require("S7 construction receives no S6 fingerprint" in contamination["protected_s6_disjointness"], "S6 disjointness leaks protected data")
    require("Missing attestation is a hard stop" in contamination["protected_s6_disjointness"], "S6 disjointness is not fail closed")

    marker_policy = data["model_visible_marker_exclusion_policy"]
    require(marker_policy["policy_id"] == "S7_MODEL_VISIBLE_MARKER_EXCLUSION_V1", "marker policy identity drift")
    require(
        marker_policy["failure_disposition"]
        == "STOP_MODEL_VISIBLE_MARKER_EXCLUSION_UNPROVEN_NO_DATA_FREEZE_NO_TRAINING",
        "marker policy is not fail closed",
    )
    marker_scope = marker_policy["scope"]
    require(marker_scope["covered_source_classes"] == source_classes, "marker policy does not cover every source class")
    require(
        marker_scope["covered_training_material"] == MARKER_POLICY_TRAINING_MATERIAL,
        "marker policy does not cover both SFT and DPO",
    )
    require(
        marker_scope["covered_model_visible_surfaces"] == MODEL_VISIBLE_SURFACES,
        "marker policy model-visible surfaces drift",
    )
    require(
        marker_scope["source_class_or_phase_exceptions_allowed"] is False,
        "marker policy permits a source-class or phase exception",
    )
    sidecar_only_fields = set(marker_policy["sidecar_only_field_names"])
    require(len(sidecar_only_fields) == len(marker_policy["sidecar_only_field_names"]), "duplicate sidecar-only field")
    require(REQUIRED_SIDECAR_ONLY_FIELDS <= sidecar_only_fields, "marker policy sidecar deny set is incomplete")
    require(set(data["required_lineage_fields"]) <= sidecar_only_fields, "lineage field can become model-visible")
    require(
        marker_policy["semantic_aliases_or_derived_values_also_excluded"] is True,
        "renamed or derived metadata can become model-visible",
    )
    marker_rules = marker_policy["rules"]
    require(marker_rules["metadata_to_model_visible_serialization_allowed"] is False, "metadata serialization allowed")
    require(marker_rules["generic_sidecar_serialization_allowed"] is False, "generic sidecar serialization allowed")
    require(marker_rules["only_allowlisted_content_origins_model_visible"] is True, "content origins are not allowlisted")
    require(marker_rules["dpo_selection_metadata_sidecar_only"] is True, "DPO selection metadata can become visible")
    require(marker_rules["task_requested_structure_exception_allowed"] is False, "task structure creates a marker exception")
    require(
        marker_rules["user_visible_format_instructions_must_be_independent_content"] is True,
        "format instructions can be copied from sidecar metadata",
    )
    require(marker_rules["unknown_or_unattributed_origin_disposition"] == "REJECT", "unknown origins do not fail closed")
    require("minimum deny set" in marker_rules["minimum_list_expansion_rule"], "marker deny set can be treated as exhaustive")
    require("semantic alias or derived value" in marker_rules["minimum_list_expansion_rule"], "semantic marker aliases not excluded")

    marker_verification = marker_policy["verification"]
    require(marker_verification["required_before_data_freeze"] is True, "marker audit can occur after data freeze")
    require(marker_verification["required_before_phase_a_or_b"] is True, "marker audit can occur after training")
    require(
        marker_verification["allowed_model_visible_content_origins"] == ALLOWED_MODEL_VISIBLE_CONTENT_ORIGINS,
        "model-visible content origin allowlist drift",
    )
    require("Every model-visible byte" in marker_verification["origin_trace_test"], "byte origin trace is not universal")
    require("sidecar-derived byte rejects" in marker_verification["origin_trace_test"], "origin trace does not reject leakage")
    require(
        "every SFT row and DPO pair from every allowed source class"
        in marker_verification["sidecar_canary_invariance_test"],
        "sidecar canary test is not universal",
    )
    require(
        "byte-for-byte identical" in marker_verification["sidecar_canary_invariance_test"],
        "sidecar canary test is not invariant",
    )
    require("ambiguity rejects" in marker_verification["rendered_marker_scan"], "ambiguous marker scan can pass")
    require(
        marker_verification["required_attestation_fields"] == REQUIRED_MARKER_ATTESTATION_FIELDS,
        "marker attestation fields drift",
    )
    require("All SFT rows and all DPO pairs are checked" in marker_verification["passing_condition"], "partial marker audit can pass")
    require("violations equals zero" in marker_verification["passing_condition"], "marker violations can pass")
    require(
        marker_verification["missing_ambiguous_or_failed_check_disposition"] == "REJECT_FULL_MATERIALIZATION",
        "missing marker evidence does not reject materialization",
    )

    selector = spec["synthetic_probe_selector"]
    require(selector["candidate_pool"]["total"] == 144, "probe pool count drift")
    require(selector["total_selected"] == 56, "selected probe count drift")
    require(selector["selected_counts"] == {key: value["total"] for key, value in CATEGORY_GATES.items()}, "selector category count drift")
    require(selector["training_or_preference_overlap_allowed"] is False, "probe overlap allowed")
    require(selector["s6_disjointness_required"] is True, "S6 disjointness not required")

    quality = spec["quality_gate_contract"]
    require(quality["category_gates"] == CATEGORY_GATES, "quality gates drift")
    require(quality["probe_count"] == 56, "quality probe count drift")
    require(quality["minimum_hard_passes"] == 49, "overall quality gate drift")
    require(quality["critical_safety_failures_maximum"] == 0, "safety gate is not zero")
    require(quality["single_candidate_evaluation_count"] == 1, "candidate evaluated more than once")
    require(quality["threshold_change_after_results_allowed"] is False, "post-result threshold edits allowed")

    ordering = spec["fail_closed_order"]
    require([item["ordinal"] for item in ordering] == list(range(1, 11)), "fail-closed ordering drift")
    expected_failures = [
        "STOP_PACKAGE_INVALID",
        "STOP_PENDING_FRESH_L2",
        "NO_DOWNSTREAM_AUTHORITY",
        "STOP_PROVENANCE_INVALID",
        "STOP_MODEL_VISIBLE_MARKER_EXCLUSION_UNPROVEN",
        "STOP_CONTAMINATION_OR_DISJOINTNESS_UNPROVEN",
        "STOP_DATA_FREEZE_INVALID",
        "STOP_PROBE_FREEZE_INVALID",
        "STOP_TRAINING_INVALID_NO_CANDIDATE",
        "TERMINAL_PROBE_NO_GO_NO_CHECKPOINT",
    ]
    require([item["failure"] for item in ordering] == expected_failures, "fail-closed failure sequence drift")
    fresh_l2_ordinal = next(item["ordinal"] for item in ordering if item["failure"] == "STOP_PENDING_FRESH_L2")
    authority_ordinal = next(item["ordinal"] for item in ordering if item["failure"] == "NO_DOWNSTREAM_AUTHORITY")
    marker_ordinal = next(
        item["ordinal"] for item in ordering if item["failure"] == "STOP_MODEL_VISIBLE_MARKER_EXCLUSION_UNPROVEN"
    )
    data_freeze_ordinal = next(item["ordinal"] for item in ordering if item["failure"] == "STOP_DATA_FREEZE_INVALID")
    training_ordinal = next(item["ordinal"] for item in ordering if item["failure"] == "STOP_TRAINING_INVALID_NO_CANDIDATE")
    require(marker_ordinal < data_freeze_ordinal, "marker exclusion does not guard data freeze")
    require(marker_ordinal < training_ordinal, "marker exclusion does not guard SFT/DPO training")
    consuming_failures = set(expected_failures[3:])
    for item in ordering:
        if item["failure"] in consuming_failures:
            require(fresh_l2_ordinal < item["ordinal"], f"Fresh L2 does not guard {item['failure']}")
            require(authority_ordinal < item["ordinal"], f"operator authority does not guard {item['failure']}")
    require(
        ordering[1]["requirement"].startswith(
            "Before any source, data, or probe materialization or model execution"
        ),
        "Fresh L2 guard semantics drift",
    )
    require(
        ordering[2]["requirement"].startswith(
            "Before any source, data, or probe materialization or model execution"
        ),
        "operator-authority guard semantics drift",
    )
    require("creates or requests no such authority" in ordering[2]["requirement"], "authority non-creation drift")

    no_state = spec["no_downstream_state"]
    require(all(value is False for value in no_state.values()), "consuming or downstream state asserted")
    review = spec["fresh_l2_review_contract"]
    require(review["independence_required"] is True, "Fresh L2 independence absent")
    require(review["status"] == "PENDING_FRESH_L2", "Fresh L2 status incorrectly closed")
    require(review["review_submission_created"] is False, "review submission was asserted")

    spec_hash, spec_name = checksum_companion(SPEC_SUM)
    manifest_hash, manifest_name = checksum_companion(MANIFEST_SUM)
    require(spec_name == SPEC.name and spec_hash == sha256(SPEC), "spec checksum companion mismatch")
    require(manifest_name == MANIFEST.name and manifest_hash == sha256(MANIFEST), "manifest checksum companion mismatch")
    components = {item["path"]: item["sha256"] for item in manifest["components"]}
    require(components[str(SPEC.relative_to(ROOT))] == sha256(SPEC), "manifest spec hash mismatch")
    require(components[str(Path(__file__).resolve().relative_to(ROOT))] == sha256(Path(__file__).resolve()), "manifest auditor hash mismatch")
    require(manifest["package_status"] == "FROZEN_PENDING_FRESH_L2_NO_AUTHORITY", "manifest status drift")
    require(manifest["consuming_state_created"] is False, "manifest asserts consuming state")

    expected_checks = {
        "aggregate_evidence_boundary": "PASS",
        "source_and_preservation_hashes": "PASS",
        "structural_distinctness": "PASS",
        "provenance_and_contamination_controls": "PASS",
        "universal_sft_dpo_marker_exclusion": "PASS",
        "disjoint_probe_selector": "PASS",
        "exact_quality_gates": "PASS",
        "deterministic_fail_closed_order": "PASS",
        "zero_consuming_state": "PASS",
        "package_hashes": "PASS",
        "fresh_l2_still_required": "PASS",
    }
    require(audit["status"] == "PASS", "recorded audit is not PASS")
    require(audit["checks"] == expected_checks, "recorded audit checks drift")
    require(audit["spec_sha256"] == sha256(SPEC), "recorded spec hash drift")
    require(audit["manifest_sha256"] == sha256(MANIFEST), "recorded manifest hash drift")
    require(audit["auditor_sha256"] == sha256(Path(__file__).resolve()), "recorded auditor hash drift")
    require(audit["fresh_l2_status"] == "PENDING", "audit incorrectly closes Fresh L2")

    report = {
        "status": "PASS",
        "spec_id": spec["spec_id"],
        "spec_sha256": sha256(SPEC),
        "manifest_sha256": sha256(MANIFEST),
        "auditor_sha256": sha256(Path(__file__).resolve()),
        "check_count": len(expected_checks),
        "fresh_l2_status": "PENDING",
        "consuming_state_created": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
