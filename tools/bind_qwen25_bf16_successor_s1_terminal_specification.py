#!/usr/bin/env python3
"""Bind the terminal, independently reviewed S1 transport lifecycle to the spec."""

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
PACKAGE_REVIEW = ROOT / "research/raw/specification/bf16-successor-s1-transport-fresh-review-20260808.json"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-successor-s1-transport-attempt-operator-authority.json"
INTENT = ROOT / "build/bf16-successor-s1-transport/backend-submission-intent.json"
CLIENT_RESULT = ROOT / "build/bf16-successor-s1-transport/backend-submission-result.json"
CLIENT_RAW = ROOT / "build/bf16-successor-s1-transport/backend-submission.raw.txt"
TERMINAL_AUDIT = ROOT / "build/bf16-successor-s1-transport/backend-terminal-audit-20260808T111251Z.json"
TERMINAL_REVIEW = ROOT / "research/raw/specification/bf16-successor-s1-transport-terminal-backend-fresh-review-20260808.json"
DOWNLOADED_STDOUT = ROOT / "build/bf16-successor-s1-transport/backend-terminal-download-20260808T111251Z/logs/ace2-bf16-successor-s1-transport-20260808-preauthority/ace2-bf16-successor-s1-transport-gate/stdout.txt"
PACKAGE_ID = "qwen2.5-0.5b-instruct-ace2-bf16-successor-s1-transport-package-v1"
PACKAGE_IDENTITY = "3b8d5c3b337701bdc318782d01abebd24d7ad5ee050ea76d42bfb6049bdaf7fb"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    return companion.is_file() and companion.read_text(encoding="ascii").split() == [sha256(path), path.name]


def item(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def append_unique(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def main() -> int:
    require(load(PIPELINE).get("current_stage") == "specification", "current stage is not specification")
    for path in (
        CONTRACT,
        PACKAGE_AUDIT,
        REPAIR_AUDIT,
        PACKAGE_REVIEW,
        AUTHORITY,
        INTENT,
        CLIENT_RESULT,
        TERMINAL_AUDIT,
        TERMINAL_REVIEW,
    ):
        require(companion_matches(path), f"checksum companion differs: {path}")

    contract = load(CONTRACT)
    package_audit = load(PACKAGE_AUDIT)
    repair_audit = load(REPAIR_AUDIT)
    package_review = load(PACKAGE_REVIEW)
    authority = load(AUTHORITY)
    intent = load(INTENT)
    client_result = load(CLIENT_RESULT)
    terminal_audit = load(TERMINAL_AUDIT)
    terminal_review = load(TERMINAL_REVIEW)

    require(contract.get("package_id") == PACKAGE_ID, "S1 package identity differs")
    require(package_audit.get("package_identity_sha256") == PACKAGE_IDENTITY, "S1 package digest differs")
    require(package_audit.get("check_count") == 36 and all(package_audit.get("checks", {}).values()), "S1 package audit differs")
    require(repair_audit.get("check_count") == 22 and all(repair_audit.get("checks", {}).values()), "S1 repair audit differs")
    require(package_review.get("review", {}).get("status") == "done", "S1 package review is not done")
    require(authority.get("maximum_backend_jobs") == 1, "S1 authority job budget differs")
    require(intent.get("authority_sha256") == sha256(AUTHORITY), "S1 intent authority binding differs")
    require(client_result.get("status") == "TERMINAL_SUBMISSION_NO_GO", "S1 raw client result changed")
    require(client_result.get("prompt_detected") is True, "S1 prompt-detection evidence changed")
    require(terminal_audit.get("status") == "TERMINAL_S1_TRANSPORT_EVIDENCE_COMPLETE", "S1 terminal audit status differs")
    require(terminal_audit.get("check_count") == 34 and terminal_audit.get("failed_check_count") == 0, "S1 terminal audit checks differ")
    require(terminal_audit.get("cardinality", {}).get("authority_count") == 1, "S1 authority count differs")
    require(terminal_audit.get("cardinality", {}).get("intent_count") == 1, "S1 intent count differs")
    require(terminal_audit.get("cardinality", {}).get("backend_job_count") == 1, "S1 job count differs")
    require(terminal_audit.get("cardinality", {}).get("retry_count") == 0, "S1 retry count differs")
    require(terminal_audit.get("backend", {}).get("status") == "failed", "S1 terminal backend state differs")
    require(terminal_audit.get("download", {}).get("stdout", {}).get("sha256") == sha256(DOWNLOADED_STDOUT), "S1 downloaded stdout binding differs")
    require(terminal_review.get("review", {}).get("status") == "done", "S1 terminal Fresh Review is not done")

    s1 = {
        "package_id": PACKAGE_ID,
        "status": "TERMINAL_TRANSPORT_GATE_EXECUTED_FAILED_FRESH_REVIEW_DONE",
        "package_identity_sha256": PACKAGE_IDENTITY,
        "claim_boundary": "The sole authorized S1 transport-only AMLT invocation created one experiment and one job. The inert payload executed, printed the no-training/no-evaluator/no-checkpoint marker, and terminated failed by its configured exit 97. This validates and closes only the transport attempt; it establishes no model, evaluator, quality, W4A8, RTL, synthesis/PPA, FPGA, U280, or project-completion result.",
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
        "package_fresh_reviewer": {
            **item(PACKAGE_REVIEW),
            "mission_id": package_review["mission_id"],
            "round": package_review["round"],
            "status": package_review["review"]["status"],
            "sealed_round_sha256": package_review["sealed_round_sha256"],
        },
        "authority": {
            **item(AUTHORITY),
            "count": 1,
            "consumed": True,
            "maximum_backend_jobs": authority["maximum_backend_jobs"],
        },
        "submission_intent": {
            **item(INTENT),
            "count": 1,
            "consumed": True,
        },
        "raw_client_result": {
            **item(CLIENT_RESULT),
            "raw_output": item(CLIENT_RAW),
            "status": client_result["status"],
            "client_exit_code": client_result["client_exit_code"],
            "prompt_detected": client_result["prompt_detected"],
            "classification": "RAW_CLIENT_RESULT_MISCLASSIFICATION_PRESERVED",
        },
        "terminal_audit": {
            **item(TERMINAL_AUDIT),
            "status": terminal_audit["status"],
            "failure_taxonomy": terminal_audit["failure_taxonomy"],
            "check_count": terminal_audit["check_count"],
            "failed_check_count": terminal_audit["failed_check_count"],
        },
        "terminal_fresh_reviewer": {
            **item(TERMINAL_REVIEW),
            "mission_id": terminal_review["mission_id"],
            "status": terminal_review["review"]["status"],
            "thread_id": terminal_review["review"].get("thread_id"),
        },
        "backend": terminal_audit["backend"],
        "downloaded_stdout": terminal_audit["download"]["stdout"],
        "cardinality": terminal_audit["cardinality"],
        "transport_only": True,
        "transport_payload_executed": True,
        "authority_granted": True,
        "authority_consumed": True,
        "submission_intent_exists": True,
        "submission_result_exists": True,
        "raw_submission_output_exists": True,
        "backend_experiment_exists": True,
        "backend_job_exists": True,
        "backend_job_terminal": True,
        "backend_job_status": "failed",
        "retry_count": 0,
        "model_execution": False,
        "evaluator_execution": False,
        "selected_checkpoint_exists": False,
        "qualified_checkpoint_exists": False,
        "model_quality_conclusion": None,
        "w4a8_policy_selected": False,
        "quality_or_downstream_authority_opened": False,
        "retry_relaunch_resume_cancel_or_second_job_permitted": False,
        "full_training_execution_package_ready": False,
        "later_quality_lifecycle_required": [
            "fresh_separately_authorized_full_training_package",
            "selector_and_dev_qualification",
            "retention",
            "exactly_once_holdout",
            "quantized_policy_selection",
            "accelerator_agreement",
            "arbitrary_text_chat_demo",
        ],
    }

    value = load(TARGET)
    value["schema_version"] = 41
    value["mission"] = "productize-local-qwen-chat-demo-s1-transport-terminal-reviewed"
    value["non_benchmark_statement"] = (
        "No external accelerator benchmark, hidden harness, or golden-output contract applies. "
        "V7 remains terminal EVALUATOR_EXECUTION_FAILURE and V8 remains terminal BACKEND_CLIENT_INTERACTIVE_PROMPT_NO_EXECUTION. "
        "The sole authorized S1 transport-only invocation created one experiment and one job; the inert payload executed and the job terminated failed by configured exit 97. "
        "Its downloaded output and exact one-authority/one-intent/one-job cardinality are independently reviewed. No model/evaluator execution, checkpoint, quality conclusion, W4A8 selection, RTL, FPGA, or U280 authority follows."
    )
    value["contract_status"] = "s1_transport_terminal_reviewed_selected_policy_null_downstream_closed"
    value["marker_free_bf16_successor_s1_transport"] = s1

    local = value["local_contract"]
    local["entrypoint_status"] = "base_bound_product_ineligible_s1_transport_terminal_reviewed_no_qualified_bf16_checkpoint_or_quantized_reference"
    local["marker_free_bf16_successor_s1_transport"] = s1

    specification = value["specification_stage_evidence"]
    specification["status"] = "public_interface_closed_s1_transport_terminal_reviewed_selected_policy_null_downstream_closed"
    specification["marker_free_bf16_successor_s1_transport"] = s1
    specification["benchmark_interface_closure"]["closure"] = "external_non_benchmark_with_s1_transport_terminal_reviewed"

    for section_name in ("product_compression_gate", "internal_acceptance_contract"):
        section = value[section_name]
        section["marker_free_bf16_successor_s1_transport"] = s1
        section["next_candidate_execution_authorized"] = False
        section["next_candidate_execution_precondition"] = (
            "The sole S1 transport attempt is consumed and terminal with independent evidence review. "
            "A new, separately scoped full-training execution package and fresh operator authority are required before any model or evaluator lifecycle; no S1 retry, resume, relaunch, cancel, or second job is legal."
        )
        section["next_structurally_distinct_engineer_task_required"] = True
        section["replacement_engineer_task_prepared"] = False
        section["frozen_execution_package_ready"] = False
        section["transport_package_ready"] = False
        section["full_training_execution_package_ready"] = False
        section["prepared_candidate_id"] = None
        section["prepared_engineer_task_id"] = None
        section["authorized_candidate_id"] = None
        section["authorized_engineer_task_id"] = None
        section["operator_direction_required"] = True
        section["planner_execution_permitted"] = False

    value["stage_order"]["stage_1_local_runtime"] = (
        "blocked_after_terminal_reviewed_s1_transport_gate_no_full_training_quality_quantization_or_accelerator_demo"
    )
    value["reason"] = (
        "V7 and V8 remain immutable terminal predecessors with no quality conclusion. The sole authorized S1 transport-only invocation consumed exactly one authority and one intent, created exactly one experiment and one job, executed only the inert no-training/no-evaluator/no-checkpoint payload, and reached terminal failed state with zero retries. "
        f"The downloaded stdout SHA-256 is {sha256(DOWNLOADED_STDOUT)}; the 34-check terminal audit and Fresh Reviewer disposition are bound. "
        "The raw client TERMINAL_SUBMISSION_NO_GO/prompt_detected result remains preserved as a false-positive classification artifact. selected_policy_id remains null; full training, quantization, accelerator agreement, arbitrary-text chat, U280, and project completion remain closed."
    )
    unsupported = value["unsupported_claims"]
    for claim in (
        "s1_terminal_transport_failure_as_model_quality_failure",
        "s1_retry_resume_relaunch_cancel_or_second_job",
        "s1_terminal_transport_as_full_training_quantized_reference_or_product_acceptance",
        "s1_raw_client_prompt_detection_as_zero_job_evidence",
    ):
        append_unique(unsupported, claim)

    temporary = TARGET.with_name(f".{TARGET.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, TARGET)
    print(f"ACE2_S1_TERMINAL_SPECIFICATION_BINDING_PASS identity={PACKAGE_IDENTITY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
