#!/usr/bin/env python3
"""Audit specification closure for the accepted marker-free V7 package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PIPELINE_SUM = ROOT / "research/PIPELINE_STATE.sha256"
SPEC = ROOT / "design/SPEC.md"
BENCHMARK = ROOT / "design/BENCHMARK_INTERFACE.json"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_PRODUCT_V7_EXECUTION_PACKAGE_CONTRACT.json"
FREEZE = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-product-v7/freeze_manifest.json"
MANIFEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7/v7-execution-package-manifest.json"
SELF_TEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7/v7-execution-package-self-test.json"
REQUEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7/execution-package-l2-review-request.json"
DECISION = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7/execution-package-l2-decision.json"
ACCEPTANCE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7/execution-package-l2-acceptance.json"
OFFLINE = MANIFEST.parent
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v7"
AUTHORITY = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-full-finetune-product-v7-attempt-operator-authority.json"
QUALIFICATION_ATTEMPT = ROOT / "research/qualification/qwen25_full_finetune_v7_one_gpu_20260808/local-evidence/volcano-submit-attempt-0001.json"
QUALIFICATION_REVIEW = ROOT / "research/raw/specification/v7-volcano-one-gpu-qualification-pre-job-review-20260808.json"
OUTPUT = ROOT / "research/diagnostics/qwen25_full_finetune_v7_specification_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    return companion.is_file() and companion.read_text(encoding="ascii").split() == [sha256(path), path.name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    pipeline = load(PIPELINE)
    contract = load(CONTRACT)
    freeze = load(FREEZE)
    manifest = load(MANIFEST)
    self_test = load(SELF_TEST)
    request = load(REQUEST)
    decision = load(DECISION)
    acceptance = load(ACCEPTANCE)
    qualification_attempt = load(QUALIFICATION_ATTEMPT)
    qualification_review = load(QUALIFICATION_REVIEW)
    benchmark = load(BENCHMARK)
    spec = SPEC.read_text(encoding="utf-8")
    ground_truth = GROUND_TRUTH.read_text(encoding="utf-8")
    local_v7 = benchmark["local_contract"]["marker_free_v7_full_finetune_successor"]
    compression_v7 = benchmark["product_compression_gate"]["marker_free_v7_full_finetune_successor"]
    internal_v7 = benchmark["internal_acceptance_contract"]["marker_free_v7_full_finetune_successor"]
    qualification = local_v7["non_consuming_volcano_qualification"]
    execution_records = [
        OFFLINE / name
        for name in (
            "v7-detached-launch-intent.json",
            "v7-detached-process.json",
            "v7-launch-terminal-status.json",
            "v7-worker-terminal-status.json",
        )
    ]
    required_rulings = (
        "v7.full-parameter-distinct: supported",
        "v7.detached-worker-interlocks: supported",
        "v7.marker-free-freeze: supported",
        "v7.resource-claims-honest: supported",
    )
    checks = {
        "pipeline_stage_specification": pipeline.get("current_stage") == "specification",
        "pipeline_checksum_unchanged": PIPELINE_SUM.is_file() and PIPELINE_SUM.read_text(encoding="ascii").split() == [sha256(PIPELINE), PIPELINE.name],
        "contract_sidecar": companion_matches(CONTRACT),
        "freeze_sidecar": companion_matches(FREEZE),
        "manifest_sidecar": companion_matches(MANIFEST),
        "self_test_sidecar": companion_matches(SELF_TEST),
        "request_sidecar": companion_matches(REQUEST),
        "decision_sidecar": companion_matches(DECISION),
        "acceptance_sidecar": companion_matches(ACCEPTANCE),
        "tree_exact": manifest.get("tree_sha256") == acceptance.get("execution_tree_sha256") == "99adb88967a4ebd16f8f6b342147a6dea1d9e8f55b6042a8d6fe88d4139ec2c2",
        "request_tree_exact": request.get("execution_tree_sha256") == manifest.get("tree_sha256"),
        "self_test_34_pass": self_test.get("check_count") == 34 and self_test.get("status") == "PASS" and all(self_test.get("checks", {}).values()),
        "l2_done": decision.get("reviewer_status") == "done" and all(item in decision.get("reason", "").lower() for item in required_rulings),
        "l2_acceptance_exact": acceptance.get("status") == "ACCEPTED_V7_FULL_FINETUNE_EXECUTION_PACKAGE_NO_ATTEMPT" and acceptance.get("accepted") is True,
        "l2_grants_no_authority": acceptance.get("attempt_authority_granted") is False and acceptance.get("detached_launch_authorized") is False,
        "full_parameter_distinct": contract["mechanism"]["all_model_parameters_trainable"] is True and contract["mechanism"]["parameter_efficient_adapter_used"] is False and contract["mechanism"]["trainable_parameter_count"] == 494032768,
        "teacher_none": contract["source_teacher_data_evaluator"]["teacher"]["mode"] == "none" and contract["source_teacher_data_evaluator"]["teacher"]["outputs_used"] is False,
        "detached_interlocks_frozen": all(contract["execution_interlocks"].values()),
        "resource_claim_boundary": "not measured" in contract["resource_assumptions"]["claim_boundary"].lower(),
        "ground_truth_unknown_only": "have not been measured" in ground_truth and "conservative prerequisites" in ground_truth,
        "no_authority": not AUTHORITY.exists(),
        "no_official_namespace": not RUN_ROOT.exists(),
        "no_launch_or_terminal_records": not any(path.exists() for path in execution_records),
        "freeze_reports_no_execution": freeze.get("training_executed") is False and freeze.get("detached_launch_created") is False,
        "spec_behavior_interface": "## Public `ace2_shell` parameter and port contract" in spec,
        "spec_clock_reset_protocol": "### Reset, clock, and CDC rules" in spec,
        "spec_acceptance_matrix": all(f"| {name} |" in spec for name in ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")),
        "spec_v7_binding": "ACCEPTED_V7_FULL_FINETUNE_EXECUTION_PACKAGE_NO_ATTEMPT" in spec and manifest["tree_sha256"] in spec,
        "benchmark_non_external": benchmark.get("applies") is False and benchmark.get("external_contract") is None,
        "benchmark_v7_binding": local_v7 == compression_v7 == internal_v7 and local_v7["execution_manifest"]["tree_sha256"] == manifest["tree_sha256"],
        "benchmark_v7_no_execution": local_v7["attempt_authority_granted"] is False and local_v7["training_executed"] is False and local_v7["official_namespace_exists"] is False,
        "qualification_attempt_sidecar": companion_matches(QUALIFICATION_ATTEMPT),
        "qualification_review_sidecar": companion_matches(QUALIFICATION_REVIEW),
        "qualification_attempt_pre_job_no_backend": qualification_attempt.get("classification") == "PRE_JOB_FAILURE_NO_BACKEND_EXPERIMENT" and qualification_attempt.get("failure_taxonomy") == "pre_job_cli_interactivity_gate" and qualification_attempt.get("submission_exit_code") == 1 and qualification_attempt.get("backend", {}).get("experiment_exists_after_attempt") is False and qualification_attempt.get("backend", {}).get("job_exists_after_attempt") is False and qualification_attempt.get("v7_lifecycle_state_created") is False,
        "qualification_fresh_review_done": qualification_review.get("reviewer_status") == "done" and qualification_review.get("acceptance") == "ACCEPTED_PRE_JOB_VOLCANO_PREREQUISITE" and qualification_review.get("classification") == "PRE_JOB_FAILURE_NO_BACKEND_EXPERIMENT" and qualification_review.get("evidence", {}).get("attempt_record_sha256") == sha256(QUALIFICATION_ATTEMPT),
        "benchmark_qualification_reconciled": qualification.get("status") == "ATTEMPT_CONSUMED_PRE_JOB_FAILURE_NO_BACKEND_EXPERIMENT" and qualification.get("submission_commands_issued") == 1 and qualification.get("submission_retries_issued") == 0 and qualification.get("authorized_submission_budget_consumed") is True and qualification.get("backend_job_exists") is False and qualification.get("runtime_b200_evidence_available") is False and qualification.get("fresh_reviewer_status") == "done" and qualification.get("review_record_sha256") == sha256(QUALIFICATION_REVIEW) and qualification.get("submission_retry_permitted") is False and qualification.get("future_submission_requires_fresh_authority") is True,
        "rejected_tree_round1_preserved": (ROOT / "evidence/specification/qwen25-full-finetune-v7-l2-round1-rejected-tree-0fe1918b5fa4352b").is_dir(),
        "rejected_tree_round2_preserved": (ROOT / "evidence/specification/qwen25-full-finetune-v7-l2-round2-rejected-tree-24b8cb4346a2197b").is_dir(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "check_count": len(checks),
        "checks": dict(sorted(checks.items())),
        "claim_boundary": "Specification-stage audit of the exact accepted marker-free V7 package; no authority, launch, model, evaluator, RTL, synthesis, or U280 execution.",
        "input_bindings": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in (PIPELINE, SPEC, BENCHMARK, GROUND_TRUTH, CONTRACT, FREEZE, MANIFEST, SELF_TEST, REQUEST, DECISION, ACCEPTANCE, QUALIFICATION_ATTEMPT, QUALIFICATION_REVIEW)
        },
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "schema_version": 1,
        "status": status,
    }
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, OUTPUT)
        OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(
            f"{sha256(OUTPUT)}  {OUTPUT.name}\n", encoding="ascii"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
