#!/usr/bin/env python3
"""Bind the one-shot QECR candidate invocation and audit frozen constraints."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "cross_layer_quantization_error_carry_final_output_v1"
LATEST = ROOT / "evidence" / CONTRACT / "latest"
OUTPUT = LATEST / "candidate_discriminator"
RESULTS = OUTPUT / "results.json"
RUN_LOG = OUTPUT / "run.log"
PRECHECK = LATEST / "PRECHECK.json"
BASELINE = LATEST / "BASELINE_REPRODUCIBILITY.json"
RTL_REVIEW = (
    ROOT
    / "evidence"
    / "review"
    / "rtl_checklist_cross_layer_quantization_error_carry_final_output_v1"
    / "decision.json"
)
ARCHITECTURE = ROOT / "evidence" / CONTRACT / "architecture_freeze.json"
PIPELINE = ROOT / "research" / "PIPELINE_STATE.json"
PUBLIC_STATUS = ROOT / "research" / "PUBLIC_STATUS.json"
BASELINE_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/649676d76012/round-0001.json"
)

EXPECTED_RTL_SHA = "0ac1ee58be61260674061e77b6b311ebc6d32fc221b3f549ae993a886fa149c1"
EXPECTED_CANDIDATE_SHA = (
    "1b4f220f646a458df088267c12a543b8a844902bf245b1b71f09bca00d2d44f3"
)
EXPECTED_PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
INVOCATION = [
    ".venv/bin/python",
    "tools/localize_score_to_lm_head.py",
    "--output-dir",
    "evidence/cross_layer_quantization_error_carry_final_output_v1/latest/candidate_discriminator",
    "--diagnostic-rope-mechanism",
    CONTRACT,
    "--capture-token-limit",
    "128",
]


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint-exit-status", type=int, required=True)
    args = parser.parse_args()

    if RESULTS.exists():
        raise SystemExit("candidate results already exist; exact-one binder refuses overwrite")
    if not RUN_LOG.is_file():
        raise SystemExit("candidate run.log is missing")

    precheck = load(PRECHECK)
    baseline = load(BASELINE)
    rtl_review = load(RTL_REVIEW)
    architecture = load(ARCHITECTURE)
    pipeline = load(PIPELINE)
    public_status = load(PUBLIC_STATUS)
    baseline_review = load(BASELINE_REVIEW)
    run_log = RUN_LOG.read_text(encoding="utf-8")

    source_hashes_exact = True
    source_hash_mismatches: list[str] = []
    for relative, expected in precheck.get("source_hashes", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            source_hashes_exact = False
            source_hash_mismatches.append(relative)

    predecessor = architecture["predecessor_no_go"]
    predecessor_path = ROOT / predecessor["path"]
    predecessor_seal_exact = (
        predecessor_path.is_file()
        and sha256(predecessor_path) == predecessor["sha256"]
    )
    frontier = architecture["frontier"]
    selected = public_status["selected_replacement_contract"]
    transition_exact = any(
        item.get("at") == "2026-08-02T05:02:31Z"
        and item.get("direction") == "advance"
        and item.get("from_stage") == "rtl"
        and item.get("to_stage") == "verification"
        for item in pipeline.get("stage_history", [])
    )
    baseline_review_passed = (
        baseline_review.get("review", {}).get("status") == "done"
        and "bit-for-bit" in baseline_review.get("review", {}).get("reason", "")
    )

    prior_candidate_results: list[str] = []
    for root in (ROOT / "evidence", ROOT / "verification"):
        for path in root.rglob("*.json"):
            if path == RESULTS:
                continue
            try:
                value = load(path)
            except (OSError, ValueError, TypeError):
                continue
            if value.get("diagnostic_rope_mechanism") == CONTRACT:
                prior_candidate_results.append(path.relative_to(ROOT).as_posix())

    failed_before_candidate_execution = (
        args.entrypoint_exit_status != 0
        and "FileExistsError" in run_log
        and "output_dir.mkdir(parents=True, exist_ok=False)" in run_log
    )
    preserved_constraints = {
        "baseline_independent_review_passed": baseline_review_passed,
        "baseline_run_count_exactly_one": baseline.get("baseline_run_count") == 1,
        "baseline_candidate_run_count_zero": baseline.get("candidate_run_count") == 0,
        "baseline_candidate_capability_accepted_false": (
            baseline.get("candidate_capability_accepted") is False
        ),
        "rtl_to_verification_advance_exact": transition_exact,
        "current_stage_verification": pipeline.get("current_stage") == "verification",
        "live_rtl_sha_exact": precheck.get("live_rtl_source_sha256") == EXPECTED_RTL_SHA,
        "candidate_aggregate_sha_exact": (
            precheck.get("candidate_rtl_hash") == EXPECTED_CANDIDATE_SHA
            and rtl_review.get("candidate_rtl_hash") == EXPECTED_CANDIDATE_SHA
        ),
        "rtl_checklist_all_three_true": all(
            rtl_review.get("checklist", {}).get(name) is True
            for name in (
                "rtl.contract-traceability",
                "rtl.hardware-discipline",
                "rtl.ip-provenance",
            )
        ),
        "rtl_required_remediation_empty": rtl_review.get("required_remediation") == "",
        "candidate_capability_accepted_false": (
            rtl_review.get("candidate_capability_accepted") is False
        ),
        "precheck_source_hashes_exact": source_hashes_exact,
        "ordered_supported_prefix_exact": (
            frontier.get("ordered_supported_layer_operator_prefix") == EXPECTED_PREFIX
            and selected.get("ordered_supported_layer_operator_prefix") == EXPECTED_PREFIX
        ),
        "first_unsupported_operator_exact": (
            frontier.get("first_unsupported_layer_operator") == "layer_0.rope_q"
            and selected.get("first_unsupported_layer_operator") == "layer_0.rope_q"
        ),
        "area_cap_2mm2_exact": frontier.get("area_cap_non_sram_mm2") == 2.0,
        "frequency_floor_100mhz_exact": frontier.get("frequency_floor_mhz") == 100.0,
        "historical_cells_exact": frontier.get("historical_cells") == 62199,
        "historical_non_sram_area_exact": (
            frontier.get("historical_non_sram_area_mm2") == 0.6108746272
        ),
        "historical_setup_slack_exact": (
            frontier.get("historical_setup_slack_ns_at_100mhz") == 0.1502
        ),
        "streaming_memory_boundary_128_bits": selected.get("memory_boundary_bits") == 128,
        "predecessor_seal_exact": predecessor_seal_exact,
        "no_prior_candidate_result": not prior_candidate_results,
        "no_baseline_in_candidate_invocation": (
            "shared_down_projection_residual_fusion_baseline_v1" not in " ".join(INVOCATION)
        ),
    }

    evidence_mismatches = [
        "entrypoint_exit_status_expected_0_observed_1",
        "candidate_results_payload_not_generated",
        "candidate_mechanism_execution_not_reached",
        "candidate_gate_and_metric_checks_unavailable",
    ]
    if not failed_before_candidate_execution:
        evidence_mismatches.append("failure_point_not_proven_pre_execution")
    for name, passed in preserved_constraints.items():
        if not passed:
            evidence_mismatches.append(f"preserved_constraint_mismatch:{name}")
    evidence_mismatches.extend(
        f"precheck_source_hash_mismatch:{path}" for path in source_hash_mismatches
    )

    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contract_id": CONTRACT,
        "node_key": "verify_candidate_discriminator",
        "stage": "verification",
        "status": "invalid_candidate_evidence_entrypoint_failed",
        "valid_candidate_evidence": False,
        "candidate_capability_accepted": False,
        "candidate_gate_passed": None,
        "candidate_metrics": None,
        "execution_contract": {
            "candidate_entrypoint_invocation_count": 1,
            "candidate_mechanism_execution_count": 0,
            "baseline_invocation_count_in_this_task": 0,
            "exact_one_authorization_consumed": True,
            "rerun_permitted": False,
            "entrypoint_exit_status": args.entrypoint_exit_status,
            "failed_before_candidate_execution": failed_before_candidate_execution,
            "invocation": INVOCATION,
            "environment_overrides": {"PYTHONDONTWRITEBYTECODE": "1"},
        },
        "failure": {
            "exception_type": "FileExistsError" if failed_before_candidate_execution else None,
            "failure_site": (
                "tools/localize_score_to_lm_head.py:run:output_dir.mkdir"
                if failed_before_candidate_execution
                else None
            ),
            "reason": (
                "The real entrypoint requires a nonexistent output directory, but the stage's "
                "existing candidate_discriminator directory was already present and empty."
            ),
            "raw_log": artifact(RUN_LOG),
        },
        "independent_constraint_audit": {
            "all_preserved_constraints_match": all(preserved_constraints.values()),
            "checks": preserved_constraints,
            "mismatches": evidence_mismatches,
            "prior_candidate_results": prior_candidate_results,
        },
        "frozen_bindings": {
            "live_rtl_source_sha256": EXPECTED_RTL_SHA,
            "candidate_rtl_hash": EXPECTED_CANDIDATE_SHA,
            "baseline": artifact(BASELINE),
            "precheck": artifact(PRECHECK),
            "rtl_review": artifact(RTL_REVIEW),
            "architecture_freeze": artifact(ARCHITECTURE),
            "predecessor_no_go": artifact(predecessor_path),
        },
        "claim_boundary": [
            "One candidate-only entrypoint invocation occurred and failed before candidate execution.",
            "No candidate metric, gate result, capability acceptance, or verification PASS is claimed.",
            "No paired, shell, PPA, physical, prototype, benchmark, signoff, tapeout, or silicon run occurred.",
        ],
        "next_required_gate": "fresh independent Reviewer disposition of this invalid one-shot evidence",
    }
    RESULTS.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums = OUTPUT / "SHA256SUMS"
    sums.write_text(
        f"{sha256(RESULTS)}  results.json\n{sha256(RUN_LOG)}  run.log\n",
        encoding="utf-8",
    )
    print(
        "QECR_CANDIDATE_AUDIT "
        f"status={result['status']} preserved_constraints="
        f"{str(all(preserved_constraints.values())).lower()} rerun_permitted=false"
    )


if __name__ == "__main__":
    main()
