#!/usr/bin/env python3
"""Bind the reviewed marker-free V8 successor into specification records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "design/BENCHMARK_INTERFACE.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_PRODUCT_V8_EXECUTION_PACKAGE_CONTRACT.json"
FREEZE = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-product-v8/freeze_manifest.json"
MANIFEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8/v8-execution-package-manifest.json"
SELF_TEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8/v8-execution-package-self-test.json"
REVIEW_REQUEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8/execution-package-l2-review-request.json"
REVIEW_DECISION = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8/execution-package-l2-decision.json"
REVIEW_ACCEPTANCE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8/execution-package-l2-acceptance.json"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8"
AUTHORITY = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-full-finetune-product-v8-attempt-operator-authority.json"
RECOVERY_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_PRODUCT_V7_EVAL_RECOVERY_V1_CONTRACT.json"
RECOVERY_REVIEW = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7-eval-recovery-v1/independent-review-acceptance.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def item(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"artifact is absent: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    return companion.is_file() and companion.read_text(encoding="ascii").split() == [sha256(path), path.name]


def append_unique(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def main() -> None:
    require(load(PIPELINE).get("current_stage") == "specification", "current stage is not specification")
    require(not RUN_ROOT.exists(), "V8 official namespace exists")
    require(not AUTHORITY.exists(), "V8 attempt authority exists")
    for path in (CONTRACT, FREEZE, MANIFEST, SELF_TEST, REVIEW_REQUEST, REVIEW_DECISION, REVIEW_ACCEPTANCE):
        require(companion_matches(path), f"checksum companion differs: {path}")

    manifest = load(MANIFEST)
    self_test = load(SELF_TEST)
    acceptance = load(REVIEW_ACCEPTANCE)
    contract = load(CONTRACT)
    require(self_test.get("status") == "PASS" and self_test.get("check_count") == 39, "V8 self-test differs")
    require(all(self_test.get("checks", {}).values()), "V8 self-test has a failed check")
    require(acceptance.get("status") == "ACCEPTED_V8_FULL_FINETUNE_EXECUTION_PACKAGE_NO_ATTEMPT", "V8 review status differs")
    require(acceptance.get("accepted") is True, "V8 review did not accept")
    require(acceptance.get("attempt_authority_granted") is False, "V8 review grants authority")
    require(acceptance.get("detached_launch_authorized") is False, "V8 review grants launch")
    require(acceptance.get("fresh_successor_no_v7_execution_evidence_dependency") is True, "V8 fresh boundary differs")
    require(acceptance.get("score_policy_schema") == "ace2_score_gates_v1", "V8 score schema differs")
    require(acceptance.get("execution_tree_sha256") == manifest.get("tree_sha256"), "V8 review tree differs")
    require(contract.get("future_authorized_lifecycle_order") == [
        "full_training_from_pinned_source_model",
        "ordered_synthetic_probe_selector",
        "single_locked_checkpoint_dev_qualification",
        "selected_bf16_materialization",
        "retention",
        "exactly_once_holdout",
    ], "V8 lifecycle order differs")

    v8 = {
        "attempt_authority_granted": False,
        "attempt_marker_exists": False,
        "candidate_id": "qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8",
        "contract": item(CONTRACT),
        "detached_launch_authorized": False,
        "execution_manifest": {**item(MANIFEST), "tree_sha256": manifest["tree_sha256"]},
        "execution_self_test": {**item(SELF_TEST), "check_count": self_test["check_count"], "status": self_test["status"]},
        "freeze_manifest": item(FREEZE),
        "fresh_successor_no_v7_execution_evidence_dependency": True,
        "future_authorized_lifecycle_order": contract["future_authorized_lifecycle_order"],
        "holdout_accessed": False,
        "independent_l2_acceptance": {**item(REVIEW_ACCEPTANCE), "reviewer_thread_id": acceptance["reviewer_thread_id"], "status": acceptance["status"]},
        "independent_l2_decision": item(REVIEW_DECISION),
        "mechanism": "fresh-source full-parameter supervised fine-tuning; FP32 master weights and optimizer moments with BF16 compute",
        "official_dev_accessed": False,
        "official_namespace_exists": False,
        "qualified_checkpoint_exists": False,
        "retention_executed": False,
        "score_policy_schema": "ace2_score_gates_v1",
        "selected_checkpoint_exists": False,
        "separate_fresh_operator_authority_required_before_launch": True,
        "status": "ACCEPTED_MARKER_FREE_FRESH_SUCCESSOR_PACKAGE_NO_ATTEMPT",
        "teacher": {"mode": "none", "outputs_used": False},
        "training_executed": False,
    }
    recovery = {
        "contract": item(RECOVERY_CONTRACT),
        "independent_review": item(RECOVERY_REVIEW),
        "permitted_as_active_direction": False,
        "reason": "Live operator guidance forbids V7 repair, replay, resume, rescore, checkpoint reuse, and scored-row reuse; preserve as historical non-authoritative evidence only.",
        "status": "HISTORICAL_DISALLOWED_BY_LIVE_SUCCESSOR_BOUNDARY",
    }

    value = load(TARGET)
    value["schema_version"] = 38
    value["mission"] = "productize-local-qwen-chat-demo-v8-marker-free-successor-reviewed-no-attempt"
    value["non_benchmark_statement"] = (
        "No external accelerator benchmark, hidden harness, or golden-output contract applies. "
        "V7 remains consumed and terminal EVALUATOR_EXECUTION_FAILURE with no quality conclusion. "
        "The fresh V8 full-training successor package is independently accepted only as a marker-free pre-execution package using canonical ace2_score_gates_v1 policy and immutable pre-V7 source/data inputs. "
        "No V8 authority, namespace, marker, training, evaluator, checkpoint, W4A8, RTL, chat-demo, FPGA, or U280 result exists."
    )
    value["contract_status"] = "v8_fresh_successor_exact_package_l2_accepted_no_attempt_no_authority_selected_policy_null_downstream_closed"
    value["historical_disallowed_v7_eval_recovery"] = recovery

    internal = value["internal_evaluation_contract"]
    internal["scope"] = "canonical hash-bound dev, retention, and exactly-once holdout policy frozen for the marker-free V8 fresh successor"
    internal["contract"] = CONTRACT.relative_to(ROOT).as_posix()
    internal["contract_sha256"] = sha256(CONTRACT)
    internal["active_marker_free_v8_full_finetune_successor"] = v8

    local = value["local_contract"]
    local["entrypoint_status"] = "base_bound_product_ineligible_v8_marker_free_successor_reviewed_no_qualified_bf16_checkpoint_or_quantized_reference"
    local["marker_free_v8_full_finetune_successor"] = v8
    local["historical_disallowed_v7_eval_recovery"] = recovery

    spec = value["specification_stage_evidence"]
    spec["status"] = "public_interface_closed_v8_fresh_successor_marker_free_package_l2_accepted_no_attempt_selected_policy_null_downstream_closed"
    spec["marker_free_v8_full_finetune_successor"] = v8
    spec["benchmark_interface_closure"]["score_policy_scope"] = "canonical ace2_score_gates_v1 policy over the unchanged internal 56-case dev evaluator, hash-bound retention inputs, and sealed 56-case exactly-once holdout"
    spec["benchmark_interface_closure"]["closure"] = "external_non_benchmark_with_exact_v8_fresh_successor_preexecution_package_l2_accepted_no_attempt"

    for section_name in ("product_compression_gate", "internal_acceptance_contract"):
        section = value[section_name]
        section["marker_free_v8_full_finetune_successor"] = v8
        section["historical_disallowed_v7_eval_recovery"] = recovery
        section["next_candidate_execution_authorized"] = False
        section["next_structurally_distinct_engineer_task_required"] = False
        section["next_candidate_execution_precondition"] = "A fresh operator authority may later authorize exactly one full V8 lifecycle bound to the reviewed tree; Planner must not create authority, launch, namespace, or marker."
        section["replacement_engineer_task_prepared"] = True
        section["frozen_execution_package_ready"] = True
        section["prepared_candidate_id"] = v8["candidate_id"]
        section["prepared_engineer_task_id"] = "v8-full-training-selector-dev-retention-holdout-after-fresh-authority"
        section["authorized_candidate_id"] = None
        section["authorized_engineer_task_id"] = None
        section["operator_direction_required"] = True
        section["planner_execution_permitted"] = False

    value["stage_order"]["stage_1_local_runtime"] = "blocked_pending_fresh_operator_authority_for_exactly_one_full_v8_training_selector_dev_retention_holdout_lifecycle_then_quality_quantization_and_accelerator_demo"
    value["reason"] = (
        f"V7 remains consumed terminal EVALUATOR_EXECUTION_FAILURE with no quality conclusion. Fresh V8 tree {manifest['tree_sha256']} starts from the pinned source model and byte-identical immutable pre-V7 training/probe inputs, uses canonical ace2_score_gates_v1 policy, passes 39/39 non-consuming checks, and is independently accepted with no V7 execution-evidence dependency. The acceptance grants no authority or launch; no V8 namespace, marker, training, evaluator, checkpoint, or quality result exists. selected_policy_id remains null, and quantization, RTL, chat-demo, U280, and product completion remain closed."
    )
    unsupported = value["unsupported_claims"]
    for claim in (
        "v7_eval_recovery_as_active_successor_or_quality_disposition",
        "v8_package_acceptance_as_attempt_or_launch_authority",
        "v8_static_self_test_as_training_quality_latency_or_resource_evidence",
        "v8_package_acceptance_as_qualified_checkpoint_or_quantized_reference",
    ):
        append_unique(unsupported, claim)

    temporary = TARGET.with_name(f".{TARGET.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, TARGET)
    print(f"ACE2_V8_SPECIFICATION_BINDING_PASS tree={manifest['tree_sha256']}")


if __name__ == "__main__":
    main()
