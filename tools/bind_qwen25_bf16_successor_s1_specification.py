#!/usr/bin/env python3
"""Bind the reviewed marker-free BF16 successor S1 transport package."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "design/BENCHMARK_INTERFACE.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_SUCCESSOR_S1_TRANSPORT_PACKAGE_CONTRACT.json"
PACKAGE_AUDIT = ROOT / "build/bf16-successor-s1-transport/package-audit.json"
REPAIR_AUDIT = ROOT / "build/bf16-successor-s1-transport-repair/repair-audit.json"
FRESH_REVIEW = ROOT / "research/raw/specification/bf16-successor-s1-transport-fresh-review-20260808.json"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-successor-s1-transport-attempt-operator-authority.json"
INTENT = ROOT / "build/bf16-successor-s1-transport/backend-submission-intent.json"
RESULT = ROOT / "build/bf16-successor-s1-transport/backend-submission-result.json"
RAW = ROOT / "build/bf16-successor-s1-transport/backend-submission-raw.log"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-successor-s1"
PACKAGE_ID = "qwen2.5-0.5b-instruct-ace2-bf16-successor-s1-transport-package-v1"
PACKAGE_IDENTITY = "3b8d5c3b337701bdc318782d01abebd24d7ad5ee050ea76d42bfb6049bdaf7fb"


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


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    return companion.is_file() and companion.read_text(encoding="ascii").split() == [
        sha256(path),
        path.name,
    ]


def item(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def append_unique(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def main() -> None:
    require(load(PIPELINE).get("current_stage") == "specification", "current stage is not specification")
    for path in (CONTRACT, PACKAGE_AUDIT, REPAIR_AUDIT, FRESH_REVIEW):
        require(companion_matches(path), f"checksum companion differs: {path}")

    contract = load(CONTRACT)
    package_audit = load(PACKAGE_AUDIT)
    repair_audit = load(REPAIR_AUDIT)
    fresh_review = load(FRESH_REVIEW)

    require(contract.get("package_id") == PACKAGE_ID, "S1 package identity differs")
    require(contract.get("status") == "MARKER_FREE_PREAUTHORITY_TRANSPORT_PACKAGE", "S1 package status differs")
    require(contract.get("authority", {}).get("granted") is False, "S1 contract grants authority")
    require(package_audit.get("status") == "PASS_MARKER_FREE_SUCCESSOR_PACKAGE", "S1 package audit status differs")
    require(package_audit.get("check_count") == 36, "S1 package audit count differs")
    require(all(package_audit.get("checks", {}).values()), "S1 package audit has a failed check")
    require(package_audit.get("package_identity_sha256") == PACKAGE_IDENTITY, "S1 package digest differs")
    require(repair_audit.get("status") == "PASS_ADDITIVE_S1_V1_REPAIR", "S1 repair audit status differs")
    require(repair_audit.get("check_count") == 22, "S1 repair audit count differs")
    require(all(repair_audit.get("checks", {}).values()), "S1 repair audit has a failed check")
    require(repair_audit.get("frozen_v1", {}).get("package_identity_sha256") == PACKAGE_IDENTITY, "S1 repair identity differs")
    require(repair_audit.get("state_before") == {"authority": False, "intent": False, "raw": False, "result": False}, "S1 repair initial state differs")
    require(repair_audit.get("state_after") == {"authority": False, "intent": False, "raw": False, "result": False}, "S1 repair final state differs")
    require(fresh_review.get("mission_id") == "ffc5e2b1f181", "S1 reviewer mission differs")
    require(fresh_review.get("round") == 2, "S1 reviewer round differs")
    require(fresh_review.get("producer_role") == "reviewer", "S1 reviewer role differs")
    require(fresh_review.get("review", {}).get("status") == "done", "S1 reviewer status differs")
    require(fresh_review.get("sealed_round_sha256") == "d341cc7156b8a736a1633ea7e033f0ec63bfd005ef334547a5d2e2a35dc068c3", "S1 sealed round differs")
    require(not any(path.exists() for path in (AUTHORITY, INTENT, RESULT, RAW, RUN_ROOT)), "S1 consuming state exists")

    s1 = {
        "package_id": PACKAGE_ID,
        "status": "MARKER_FREE_PREAUTHORITY_TRANSPORT_PACKAGE_FRESH_REVIEW_DONE",
        "package_identity_sha256": PACKAGE_IDENTITY,
        "claim_boundary": contract["claim_boundary"],
        "contract": item(CONTRACT),
        "package_audit": {
            **item(PACKAGE_AUDIT),
            "status": package_audit["status"],
            "check_count": package_audit["check_count"],
        },
        "additive_repair_audit": {
            **item(REPAIR_AUDIT),
            "status": repair_audit["status"],
            "check_count": repair_audit["check_count"],
            "amlt_version": repair_audit["preflight"]["amlt_version"],
            "amlt_executable_sha256": repair_audit["preflight"]["amlt_executable_sha256"],
        },
        "fresh_reviewer": {
            **item(FRESH_REVIEW),
            "mission_id": fresh_review["mission_id"],
            "round": fresh_review["round"],
            "status": fresh_review["review"]["status"],
            "sealed_round_sha256": fresh_review["sealed_round_sha256"],
        },
        "transport_only": True,
        "authority_granted": False,
        "submission_intent_exists": False,
        "submission_result_exists": False,
        "raw_submission_output_exists": False,
        "official_namespace_exists": False,
        "backend_experiment_or_job_exists": False,
        "model_execution": False,
        "evaluator_execution": False,
        "selected_checkpoint_exists": False,
        "qualified_checkpoint_exists": False,
        "model_quality_conclusion": None,
        "w4a8_policy_selected": False,
        "quality_or_downstream_authority_opened": False,
        "separate_fresh_operator_authority_required_before_transport_submission": True,
        "full_training_execution_package_ready": False,
        "later_quality_lifecycle_required": [
            "full_training",
            "selector_and_dev_qualification",
            "retention",
            "exactly_once_holdout",
            "quantized_policy_selection",
            "accelerator_agreement",
            "arbitrary_text_chat_demo",
        ],
    }

    value = load(TARGET)
    value["schema_version"] = 40
    value["mission"] = "productize-local-qwen-chat-demo-s1-marker-free-transport-reviewed-no-authority"
    value["non_benchmark_statement"] = (
        "No external accelerator benchmark, hidden harness, or golden-output contract applies. "
        "V7 remains terminal EVALUATOR_EXECUTION_FAILURE and V8 remains terminal BACKEND_CLIENT_INTERACTIVE_PROMPT_NO_EXECUTION. "
        "The explicitly new S1 package is independently accepted only as marker-free, authority-gated transport infrastructure; it has no authority, submission intent, backend job, model/evaluator execution, checkpoint, or quality conclusion."
    )
    value["contract_status"] = "s1_marker_free_transport_reviewed_no_authority_selected_policy_null_downstream_closed"
    value["marker_free_bf16_successor_s1_transport"] = s1

    local = value["local_contract"]
    local["entrypoint_status"] = "base_bound_product_ineligible_s1_transport_reviewed_no_authority_no_qualified_bf16_checkpoint_or_quantized_reference"
    local["marker_free_bf16_successor_s1_transport"] = s1

    specification = value["specification_stage_evidence"]
    specification["status"] = "public_interface_closed_s1_marker_free_transport_reviewed_no_authority_selected_policy_null_downstream_closed"
    specification["marker_free_bf16_successor_s1_transport"] = s1
    specification["benchmark_interface_closure"]["closure"] = "external_non_benchmark_with_s1_marker_free_transport_reviewed_no_authority"

    for section_name in ("product_compression_gate", "internal_acceptance_contract"):
        section = value[section_name]
        section["marker_free_bf16_successor_s1_transport"] = s1
        section["next_candidate_execution_authorized"] = False
        section["next_candidate_execution_precondition"] = (
            "S1 transport v1 and its additive repair are frozen and Fresh-Reviewer accepted, but no operator authority exists. "
            "A later transport submission requires fresh authority and remains inert; a separate full-training execution package and ordered quality lifecycle are still required before quantization or accelerator work."
        )
        section["next_structurally_distinct_engineer_task_required"] = False
        section["replacement_engineer_task_prepared"] = True
        section["frozen_execution_package_ready"] = False
        section["transport_package_ready"] = True
        section["full_training_execution_package_ready"] = False
        section["prepared_candidate_id"] = PACKAGE_ID
        section["prepared_engineer_task_id"] = "s1-marker-free-transport-package-v1"
        section["authorized_candidate_id"] = None
        section["authorized_engineer_task_id"] = None
        section["operator_direction_required"] = True
        section["planner_execution_permitted"] = False

    value["stage_order"]["stage_1_local_runtime"] = (
        "blocked_after_v8_terminal_submission_no_go_with_explicitly_new_planner_owned_bf16_successor_s1_transport_"
        "frozen_reviewed_no_authority_no_full_training_quality_quantization_or_accelerator_demo"
    )
    value["reason"] = (
        "V7 and V8 remain immutable terminal predecessors with no quality conclusion. "
        f"The distinct S1 marker-free transport package {PACKAGE_IDENTITY} passes the frozen 36-check audit and the additive 22-check exact-version/project repair, and Fresh Reviewer mission ffc5e2b1f181 returned done. "
        "No S1 authority, intent, result, raw output, namespace, backend job, model execution, evaluator execution, checkpoint, or quality conclusion exists. selected_policy_id remains null; full training, quantization, accelerator agreement, arbitrary-text chat, U280, and project completion remain closed."
    )
    unsupported = value["unsupported_claims"]
    for claim in (
        "s1_package_or_fresh_review_as_transport_submission_authority",
        "s1_transport_check_as_model_training_or_evaluator_execution",
        "s1_transport_package_as_qualified_checkpoint_quantized_reference_or_product_acceptance",
        "s1_submission_without_fresh_operator_authority",
    ):
        append_unique(unsupported, claim)

    temporary = TARGET.with_name(f".{TARGET.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, TARGET)
    print(f"ACE2_S1_SPECIFICATION_BINDING_PASS identity={PACKAGE_IDENTITY}")


if __name__ == "__main__":
    main()
