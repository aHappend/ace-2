#!/usr/bin/env python3
"""Bind the independently accepted marker-free V7 package into specification state."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "design/BENCHMARK_INTERFACE.json"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_PRODUCT_V7_EXECUTION_PACKAGE_CONTRACT.json"
FREEZE = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-product-v7/freeze_manifest.json"
MANIFEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7/v7-execution-package-manifest.json"
SELF_TEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7/v7-execution-package-self-test.json"
L2_DECISION = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7/execution-package-l2-decision.json"
L2_ACCEPTANCE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7/execution-package-l2-acceptance.json"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7"
AUTHORITY = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-full-finetune-product-v7-attempt-operator-authority.json"
OFFLINE = MANIFEST.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def item(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    value = json.loads(TARGET.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    self_test = json.loads(SELF_TEST.read_text(encoding="utf-8"))
    acceptance = json.loads(L2_ACCEPTANCE.read_text(encoding="utf-8"))
    if acceptance.get("status") != "ACCEPTED_V7_FULL_FINETUNE_EXECUTION_PACKAGE_NO_ATTEMPT":
        raise RuntimeError("V7 independent L2 acceptance is absent")
    if self_test.get("check_count") != 34 or not all(self_test.get("checks", {}).values()):
        raise RuntimeError("V7 34-check self-test is not accepted")
    if RUN_ROOT.exists() or AUTHORITY.exists():
        raise RuntimeError("V7 authority or official namespace exists")
    execution_names = (
        "v7-detached-launch-intent.json",
        "v7-detached-process.json",
        "v7-launch-terminal-status.json",
        "v7-worker-terminal-status.json",
    )
    if any((OFFLINE / name).exists() for name in execution_names):
        raise RuntimeError("V7 launch or terminal execution state exists")

    v7 = {
        "candidate_id": "qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7",
        "status": "ACCEPTED_MARKER_FREE_EXECUTION_PACKAGE_NO_ATTEMPT",
        "mechanism": "full-parameter supervised fine-tuning; all 494,032,768 parameters use FP32 master weights with BF16 compute",
        "teacher": {"mode": "none", "outputs_used": False},
        "contract": item(CONTRACT),
        "freeze_manifest": item(FREEZE),
        "execution_manifest": {**item(MANIFEST), "tree_sha256": manifest["tree_sha256"]},
        "execution_self_test": {**item(SELF_TEST), "check_count": self_test["check_count"], "status": self_test["status"]},
        "independent_l2_decision": item(L2_DECISION),
        "independent_l2_acceptance": {**item(L2_ACCEPTANCE), "status": acceptance["status"], "reviewer_thread_id": acceptance["reviewer_thread_id"]},
        "attempt_authority_granted": False,
        "detached_launch_authorized": False,
        "official_namespace_exists": False,
        "attempt_marker_exists": False,
        "training_executed": False,
        "official_dev_accessed": False,
        "retention_executed": False,
        "holdout_accessed": False,
        "selected_checkpoint_exists": False,
        "qualified_checkpoint_exists": False,
        "resource_values_are_prerequisites_not_measurements": True,
        "local_cuda_gpu_present_at_freeze": False,
        "rejected_predecessor_trees_preserved": [
            "0fe1918b5fa4352b0ed8b1ab5dc032a4fe28c25148596837d429dbec40094578",
            "24b8cb4346a2197bd9d6856239a694e266cec5b90d7947ae48515b0a1934e27d",
        ],
        "separate_fresh_operator_authority_required_before_launch": True,
    }

    value["schema_version"] = 35
    value["mission"] = "productize-local-qwen-chat-demo-v7-full-finetune-package-accepted-no-attempt"
    value["non_benchmark_statement"] = (
        "No external accelerator benchmark, hidden harness, or golden-output contract applies. "
        "V2, V3, and V4 are terminal dev NO-GO records; V5 is terminal ABORTED_UNAUTHORIZED_PRETRAIN. "
        "The materially distinct V7 full-parameter package is independently accepted only as an exact marker-free pre-execution package. "
        "No V7 authority, launch, namespace, marker, training, official evaluator, checkpoint, quality, quantization, RTL, or U280 result exists."
    )
    value["contract_status"] = "v7_full_finetune_exact_package_l2_accepted_no_attempt_no_authority_selected_policy_null_downstream_closed"
    internal = value["internal_evaluation_contract"]
    internal["scope"] = "unchanged hash-bound dev, retention, and exactly-once holdout policy frozen for the accepted marker-free V7 full-parameter successor"
    internal["contract"] = CONTRACT.relative_to(ROOT).as_posix()
    internal["contract_sha256"] = sha256(CONTRACT)
    local = value["local_contract"]
    local["entrypoint_status"] = "base_bound_product_ineligible_v7_full_finetune_package_accepted_marker_free_pending_separate_operator_authority_and_successful_ordered_quality_then_quantization"
    local["marker_free_v7_full_finetune_successor"] = v7
    spec = value["specification_stage_evidence"]
    spec["status"] = "public_interface_closed_v7_full_finetune_marker_free_package_l2_accepted_no_attempt_selected_policy_null_downstream_closed"
    spec["benchmark_interface_closure"]["score_policy_scope"] = "the unchanged internal 56-case dev evaluator, hash-bound retention inputs, and sealed 56-case exactly-once holdout frozen for any separately authorized V7 attempt"
    spec["benchmark_interface_closure"]["closure"] = "external_non_benchmark_with_exact_v7_full_finetune_preexecution_package_l2_accepted_no_attempt"
    compression = value["product_compression_gate"]
    compression["selection_status"] = "none_v7_full_finetune_package_accepted_no_attempt_no_qualified_checkpoint_quantization_and_rtl_closed"
    compression["marker_free_v7_full_finetune_successor"] = v7
    compression["next_candidate_execution_authorized"] = False
    compression["next_candidate_execution_precondition"] = "A fresh operator authority may authorize exactly one detached V7 attempt; Planner must not create authority, launch, namespace, or marker. V5 remains terminal and prohibited."
    compression["next_structurally_distinct_engineer_task_required"] = False
    compression["replacement_engineer_task_prepared"] = True
    compression["frozen_execution_package_ready"] = True
    compression["prepared_candidate_id"] = v7["candidate_id"]
    compression["prepared_engineer_task_id"] = "v7-detached-full-finetune-attempt-after-fresh-authority"
    compression["authorized_candidate_id"] = None
    compression["authorized_engineer_task_id"] = None
    compression["operator_direction_required"] = True
    compression["planner_execution_permitted"] = False
    internal_gate = value["internal_acceptance_contract"]
    internal_gate["marker_free_v7_full_finetune_successor"] = v7
    internal_gate["next_candidate_execution_authorized"] = False
    internal_gate["next_candidate_execution_precondition"] = compression["next_candidate_execution_precondition"]
    internal_gate["next_structurally_distinct_engineer_task_required"] = False
    internal_gate["replacement_engineer_task_prepared"] = True
    internal_gate["frozen_execution_package_ready"] = True
    internal_gate["prepared_candidate_id"] = v7["candidate_id"]
    internal_gate["prepared_engineer_task_id"] = compression["prepared_engineer_task_id"]
    internal_gate["authorized_candidate_id"] = None
    internal_gate["authorized_engineer_task_id"] = None
    internal_gate["operator_direction_required"] = True
    internal_gate["planner_execution_permitted"] = False
    value["stage_order"]["stage_1_local_runtime"] = "blocked_pending_fresh_operator_authority_for_exactly_one_detached_v7_attempt_then_ordered_quality_quantization_and_accelerator_demo"
    value["reason"] = (
        "V5 remains terminal ABORTED_UNAUTHORIZED_PRETRAIN with no quality outcome and no attempt-0002. "
        "A materially distinct V7 full-parameter BF16-compute package is now frozen at tree "
        f"{manifest['tree_sha256']} and independently accepted after 34/34 semantic checks. "
        "The acceptance grants no authority or launch permission; no V7 namespace, marker, training, evaluator, checkpoint, or quality result exists. "
        "selected_policy_id remains null, and quantization, RTL, U280, and product completion remain closed."
    )
    unsupported = value["unsupported_claims"]
    for claim in (
        "v7_package_acceptance_as_attempt_or_launch_authority",
        "v7_static_self_test_as_training_quality_or_latency_evidence",
        "v7_resource_prerequisites_as_measured_peak_memory_or_wall_time",
        "v7_package_acceptance_as_qualified_checkpoint_or_quantized_reference",
    ):
        if claim not in unsupported:
            unsupported.append(claim)

    temporary = TARGET.with_name(f".{TARGET.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, TARGET)
    print(f"ACE2_V7_SPECIFICATION_BINDING_PASS tree={manifest['tree_sha256']}")


if __name__ == "__main__":
    main()
