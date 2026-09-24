#!/usr/bin/env python3
"""Bind the immutable S5 terminal NO-GO into BENCHMARK_INTERFACE.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "design/BENCHMARK_INTERFACE.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
BUILD = ROOT / "build/bf16-full-finetune-successor-s5"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-attempt-operator-authority.json"
INTENT = BUILD / "backend-submission-intent.json"
RESULT = BUILD / "backend-submission-result.json"
RAW = BUILD / "backend-submission.raw.txt"
TERMINAL = BUILD / "backend-terminal-audit-20260808T150208Z.json"
PROVENANCE = BUILD / "submission-client-provenance-audit-20260808T151648Z.json"
FIRST_REVIEW = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-terminal-provenance-fresh-review-20260808.json"
FINAL_REVIEW = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-terminal-provenance-fresh-review-final-20260808.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def main() -> int:
    for path in (TARGET, PIPELINE, AUTHORITY, INTENT, RESULT, RAW, TERMINAL, PROVENANCE, FIRST_REVIEW, FINAL_REVIEW):
        if not path.is_file():
            raise RuntimeError(f"required S5 terminal specification input is absent: {path}")
    pipeline = load(PIPELINE)
    if pipeline.get("current_stage") != "specification":
        raise RuntimeError("pipeline stage differs")
    terminal = load(TERMINAL)
    provenance = load(PROVENANCE)
    final_review = load(FINAL_REVIEW).get("review", {})
    if terminal.get("status") != "TERMINAL_S5_DEV_QUALITY_NO_GO_EVIDENCE_COMPLETE" or terminal.get("check_count") != 60:
        raise RuntimeError("terminal audit differs")
    if provenance.get("status") != "PASS_BOUNDED_SUBMISSION_CLIENT_PROVENANCE_DEVIATION" or provenance.get("check_count") != 22:
        raise RuntimeError("provenance audit differs")
    if final_review.get("status") != "done":
        raise RuntimeError("final Fresh Reviewer status differs")

    data = load(TARGET)
    data["schema_version"] = max(int(data.get("schema_version", 0)), 47)
    data["mission"] = "productize-local-qwen-chat-demo-s5-terminal-quality-no-go"
    data["stage"] = "specification"
    data["applies"] = False
    data["external_contract"] = None
    data["non_benchmark_statement"] = (
        "No external accelerator benchmark, hidden harness, or golden-output contract applies. "
        "S5 consumed exactly one lifecycle and terminated DEV_QUALITY_GATE_FAILURE at 27/56 versus the frozen minimum 48. "
        "Retention and holdout did not run. The submission used a noncanonical AMLT 11.17.0 launcher; the bounded provenance audit preserves both launchers and does not claim transport equivalence. "
        "S5 is immutable and opens no retry, W4A8, RTL, synthesis/PPA, FPGA/U280, stage-advance, or completion authority."
    )
    data["contract_status"] = "s5_terminal_dev_quality_no_go_client_provenance_deviation_selected_policy_null_downstream_closed"

    terminal_binding = {
        "package_id": "qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s5-package-v1",
        "package_identity_sha256": "0487a96540d733eeee88bbef1342b4af5f99204ec334d56c50092be6a5c28a02",
        "execution_tree_sha256": "54602351773e85fc28e0347aacbee988cc787157306f01913bd74ff190f79a0a",
        "status": "TERMINAL_S5_DEV_QUALITY_NO_GO_FRESH_REVIEW_DONE",
        "failure_taxonomy": "DEV_QUALITY_GATE_FAILURE",
        "first_failed_gate": "minimum_hard_passes",
        "authority": record(AUTHORITY),
        "submission_intent": record(INTENT),
        "submission_result": record(RESULT),
        "submission_raw": record(RAW),
        "terminal_audit": {**record(TERMINAL), "check_count": 60, "failed_check_count": 0},
        "client_provenance_audit": {
            **record(PROVENANCE),
            "check_count": 22,
            "failed_check_count": 0,
            "classification": "BOUNDED_SUBMISSION_CLIENT_PROVENANCE_DEVIATION",
            "canonical_executable_used_for_submission": False,
            "transport_equivalence_claimed": False,
        },
        "first_fresh_review": {**record(FIRST_REVIEW), "status": "blocked_checkpoint_write_only", "substantive_rulings_supported": True},
        "final_fresh_review": {**record(FINAL_REVIEW), "status": "done", "thread_id": final_review.get("thread_id")},
        "backend": terminal["backend"],
        "cardinality": terminal["cardinality"],
        "training": {"completed": True, "optimizer_steps": 132, "checkpoint_epochs": [1, 2, 3]},
        "selector": {"selected_epoch": 3, "locked_before_dev": True},
        "dev": {"executed_once": True, "response_count": 56, "hard_pass_count": 27, "minimum_hard_passes": 48, "status": "NO_GO"},
        "selected_bf16_materialized": False,
        "retention_executed": False,
        "holdout_executed": False,
        "model_quality_conclusion": "NO_GO",
        "retry_count": 0,
        "retry_resume_relaunch_resubmit_repair_rescore_or_attempt_0002_allowed": False,
        "quality_or_downstream_authority_opened": False,
        "planner_execution_permitted": False,
        "next_required_event": "distinct marker-free successor design requires separate operator authority; none is granted in this task",
    }
    data.pop("active_marker_free_s5_successor", None)
    data["authoritative_s5_terminal_containment"] = terminal_binding

    spec_evidence = data.setdefault("specification_stage_evidence", {})
    spec_evidence.pop("active_marker_free_s5_successor", None)
    spec_evidence["status"] = "PASS_TERMINAL_S5_SPECIFICATION_CLOSURE"
    spec_evidence["authoritative_s5_terminal_containment"] = {
        "status": terminal_binding["status"],
        "failure_taxonomy": terminal_binding["failure_taxonomy"],
        "terminal_audit_sha256": terminal_binding["terminal_audit"]["sha256"],
        "terminal_audit_check_count": 60,
        "client_provenance_audit_sha256": terminal_binding["client_provenance_audit"]["sha256"],
        "client_provenance_audit_check_count": 22,
        "fresh_reviewer_sha256": terminal_binding["final_fresh_review"]["sha256"],
        "fresh_reviewer_status": "done",
        "dev_hard_pass_count": 27,
        "dev_minimum_hard_passes": 48,
        "retention_executed": False,
        "holdout_executed": False,
        "retry_or_downstream_authority_opened": False,
    }

    successor_precondition = (
        "S5 is consumed terminal DEV_QUALITY_GATE_FAILURE evidence and cannot be retried or repaired in place. "
        "Any next quality route must be a distinct marker-free successor informed only by non-hidden aggregate evidence and requires separate operator authority; no successor is authorized in this task."
    )
    for container_name in ("product_compression_gate", "internal_acceptance_contract"):
        container = data.get(container_name)
        if isinstance(container, dict):
            container["next_candidate_execution_authorized"] = False
            container["next_candidate_execution_precondition"] = successor_precondition
            container["next_structurally_distinct_engineer_task_required"] = True
            container["replacement_engineer_task_prepared"] = False
            container["planner_execution_permitted"] = False
            container["predecessor_fresh_reviewer_adjudication_complete"] = True
            container["s5_terminal_quality_status"] = "DEV_QUALITY_GATE_FAILURE"
            container["s5_client_provenance_status"] = "BOUNDED_SUBMISSION_CLIENT_PROVENANCE_DEVIATION"

    data["reason"] = (
        "S5 consumed exactly one authorized lifecycle: one authority, intent, experiment, and backend job, zero retries. "
        "Training completed 132 steps, epoch 3 was locked before one official dev, and dev returned 27/56 versus the frozen minimum 48. "
        "Retention and holdout did not execute. The 60/60 terminal audit and 22/22 client-provenance audit pass. "
        "Fresh Reviewer returned done and classified the noncanonical AMLT launcher as a bounded provenance deviation without established transport equivalence. "
        "S5 is terminal and immutable; selected_policy_id remains null, and W4A8, accelerator agreement, arbitrary-text chat, U280, and project completion remain closed."
    )
    unsupported = data.setdefault("unsupported_claims", [])
    for claim in (
        "s5_retry_resume_relaunch_resubmit_repair_in_place_rescore_or_attempt_0002",
        "canonical_amlt_executable_used_for_s5_submission",
        "s5_submission_client_transport_equivalence",
        "s5_dev_no_go_as_bf16_quality_pass",
        "s5_terminal_no_go_as_w4a8_rtl_fpga_u280_or_stage_authority",
    ):
        if claim not in unsupported:
            unsupported.append(claim)

    TARGET.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    print(f"ACE2_S5_TERMINAL_BENCHMARK_INTERFACE_BOUND sha256={sha256(TARGET)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
