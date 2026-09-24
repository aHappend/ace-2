#!/usr/bin/env python3
"""Read-only independent L2 audit of the authoritative DPRF integrity NO_GO."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_down_projection_residual_fusion_verification import check_replays


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_down_projection_residual_fusion_v1"
CANDIDATE_ID = "down_projection_residual_fusion_76ef0dda2e646558"
CANDIDATE_HASH = "76ef0dda2e646558ecb3e2047d3ec00d69f416cbe0a880b3402876614a4a7a28"
RTL_SHA256 = "b3f33cb503aa89f4f7c54b0fb4ffcc814061b28917bf911b35ad335fb2425bdf"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"

LATEST = ROOT / "evidence" / CONTRACT / "latest"
DECISION = LATEST / "VERIFICATION_DECISION.json"
RESULTS = ROOT / "verification/RESULTS.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PLAN = ROOT / "verification/PLAN.md"
CHECKPOINT = ROOT / "CHECKPOINT.md"
OUT = ROOT / "evidence/review/focused_verification_shared_down_projection_residual_fusion_v1/decision.json"
OUT_ARCHIVE = OUT.parent / "archive"
RESULTS_INPUT_ARCHIVE = ROOT / "evidence/shared_down_projection_residual_fusion_v1/archive/independent_l2_input"
FIRST_BASELINE = ROOT / "verification/raw/archive/pre-dprf-42fa63d925d813c8/dprf_focused_baseline_replay/results.json"
FIRST_CANDIDATE = ROOT / "verification/raw/archive/pre-dprf-42fa63d925d813c8/dprf_focused_candidate_replay/results.json"
SECOND_BASELINE = ROOT / "verification/raw/latest/dprf_focused_baseline_replay/results.json"
SECOND_CANDIDATE = ROOT / "verification/raw/latest/dprf_focused_candidate_replay/results.json"
PRIOR_NON_REVIEWER_VERDICTS = (
    (
        "engineer",
        ROOT
        / "evidence/shared_down_projection_residual_fusion_v1/archive/duplicate_execution_integrity_rebind/SELF_ATTESTED_L2_DECISION.b870475fa0082227de4d7d476800fceb20e2673be2fc848e89ba10a5227c2fde.json",
    ),
    (
        "planner",
        ROOT
        / "evidence/shared_down_projection_residual_fusion_v1/archive/duplicate_execution_integrity_rebind/SELF_ATTESTED_L2_DECISION.e70b674fe92c1ae7db195f6b33881f5747a11ad573d6d7daa16949dd469f3608.json",
    ),
    (
        "planner",
        ROOT
        / "evidence/review/focused_verification_shared_down_projection_residual_fusion_v1/archive/decision.d8a0d682689fa097b971b9c2a2254141bc35096b367292d69eee745749ed521f.json",
    ),
    (
        "planner",
        ROOT
        / "evidence/shared_down_projection_residual_fusion_v1/archive/duplicate_execution_integrity_rebind/SELF_ATTESTED_L2_DECISION.9b6c14ff75c46989fccb274e02fa938827289042c98bd53ab6f523c66a742caa.json",
    ),
)


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
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_canonical(value: dict[str, Any], label: str) -> None:
    require(
        value.get("integrity", {}).get("canonical_sha256") == canonical_sha256(value),
        f"{label} canonical SHA-256 differs",
    )


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def strip_execution_time(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_execution_time(item)
            for key, item in value.items()
            if key != "generated_at_utc"
        }
    if isinstance(value, list):
        return [strip_execution_time(item) for item in value]
    return value


def verify_artifact(record: dict[str, Any]) -> None:
    path = ROOT / record["path"]
    require(artifact(path) == record, f"artifact binding differs: {record['path']}")


def update_public(now: str, review_record: dict[str, Any], decision_record: dict[str, Any]) -> None:
    public = load(PUBLIC)
    checklist = {
        "verification.independent-oracle": False,
        "verification.coverage-stress": True,
        "verification.reproducible-green": True,
    }
    status = "integrity_no_go_duplicate_discriminator_execution_independently_confirmed"
    latest = "down_projection_residual_fusion_integrity_no_go_independently_confirmed"
    manager_action = "manager_rollback_to_architecture_for_structurally_distinct_successor"
    public.update(
        {
            "generated_at_utc": now,
            "last_updated_utc": now,
            "latest_decision": latest,
            "routing_status": status,
            "required_operator_action": "none",
            "required_manager_action": manager_action,
            "supported_layer_operator_prefix": PREFIX,
            "ordered_supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "current_mode": "ADVANCE",
            "verification_checklist": checklist,
            "stage_closing": False,
        }
    )
    stage = public.setdefault("stage", {})
    stage.update(
        {
            "current_stage": "verification",
            "current_stage_status": status,
            "current_stage_checklist": checklist,
            "current_stage_evidence": [
                "verification/PLAN.md",
                "verification/RESULTS.json",
                decision_record["path"],
                review_record["path"],
            ],
            "stage_closing": False,
            "planner_may_advance_stage": False,
            "stage_transition_owner": "Manager",
        }
    )
    latest_verification = {
        "status": status,
        "contract_id": CONTRACT,
        "candidate_id": CANDIDATE_ID,
        "candidate_rtl_hash": CANDIDATE_HASH,
        "technical_disposition": "bounded_no_go",
        "execution_integrity": "failed_duplicate_successful_execution",
        "checklist": checklist,
        "decision": decision_record,
        "independent_l2_review": review_record,
        "results": "verification/RESULTS.json",
        "paired_smoke_run": False,
        "ppa_run": False,
    }
    public["latest_verification_stage"] = latest_verification
    public["verification_integrity_no_go_review"] = review_record
    dashboard = public.setdefault("dashboard_fields", {})
    dashboard.update(
        {
            "current_stage": "verification",
            "current_mode": "ADVANCE",
            "latest_decision": latest,
            "candidate_meets_numeric_acceptance": False,
            "supported_layer_operator_prefix": PREFIX,
            "ordered_supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "verification_checklist": checklist,
            "latest_verification_stage": latest_verification,
        }
    )
    dashboard.setdefault("candidate_mechanism", {}).update(
        {
            "contract_id": CONTRACT,
            "exact_scale32_dataset_discriminator_status": status,
            "implementation_authorized": False,
            "status": status,
        }
    )
    dashboard.setdefault("latest_rtl_candidate", {}).update(
        {
            "contract_id": CONTRACT,
            "candidate_id": CANDIDATE_ID,
            "candidate_rtl_hash": CANDIDATE_HASH,
            "status": status,
            "required_manager_action": manager_action,
            "stage_closing": False,
        }
    )
    public.setdefault("implementation_frontier", {}).update(
        {
            "current_stage": "verification",
            "current_mode": "ADVANCE",
            "latest_decision": latest,
            "supported_layer_operator_prefix": PREFIX,
            "ordered_supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "verification_checklist": checklist,
            "latest_verification_stage": latest_verification,
        }
    )
    public.setdefault("integrity", {})["canonical_sha256"] = None
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)


def main() -> int:
    if OUT.exists():
        prior = load(OUT)
        require(
            "verification_results" in prior or "review_tool" not in prior,
            f"independent L2 verdict already exists and is current: {OUT}",
        )
        OUT_ARCHIVE.mkdir(parents=True, exist_ok=True)
        prior_destination = OUT_ARCHIVE / f"decision.{sha256(OUT)}.json"
        if not prior_destination.exists():
            shutil.copy2(OUT, prior_destination)
        OUT.unlink()
    require(load(PIPELINE).get("current_stage") == "verification", "Manager-owned stage is not verification")
    require(sha256(ROOT / "rtl/ace2_down_projection_residual_fusion_core.sv") == RTL_SHA256, "live RTL changed")

    decision = load(DECISION)
    results = load(RESULTS)
    first_baseline = load(FIRST_BASELINE)
    first_candidate = load(FIRST_CANDIDATE)
    second_baseline = load(SECOND_BASELINE)
    second_candidate = load(SECOND_CANDIDATE)
    verify_canonical(decision, "authoritative discriminator decision")
    verify_canonical(results, "verification results")

    require(decision.get("contract_id") == CONTRACT, "decision contract differs")
    require(decision.get("candidate_id") == CANDIDATE_ID, "decision candidate differs")
    require(decision.get("candidate_rtl_hash") == CANDIDATE_HASH, "decision RTL hash differs")
    require(decision.get("status") == "integrity_no_go", "decision status differs")
    require(decision.get("decision") == "integrity_no_go_duplicate_successful_discriminator_execution", "decision disposition differs")
    require(decision.get("stage") == "verification" and decision.get("stage_closing") is False, "stage disposition differs")
    execution = decision["execution_contract"]
    require(execution["required_successful_execution_count"] == 1, "exact-one requirement differs")
    require(execution["observed_successful_execution_count"] == 2, "observed execution count differs")
    require(execution["satisfied"] is False and execution["failure"] == "duplicate_successful_execution", "integrity failure differs")
    require(execution["rerun_permitted"] is False, "rerun prohibition differs")

    for record in (
        execution["first_successful_execution"]["baseline_replay"],
        execution["first_successful_execution"]["candidate_replay"],
        execution["later_duplicate_execution"]["baseline_replay"],
        execution["later_duplicate_execution"]["candidate_replay"],
        decision["precheck"],
        decision["rtl_review"],
        *decision["source_hashes"],
    ):
        verify_artifact(record)

    require(strip_execution_time(first_baseline) == strip_execution_time(second_baseline), "baseline executions are not payload-equivalent")
    require(strip_execution_time(first_candidate) == strip_execution_time(second_candidate), "candidate executions are not payload-equivalent")
    recomputed_quality = check_replays(first_baseline, first_candidate)
    require(recomputed_quality == decision["technical_disposition"]["quality_gate"], "quality recomputation differs")
    require(recomputed_quality["passed"] is False, "technical quality gate unexpectedly passed")
    require(results["decision"] == artifact(DECISION), "verification results decision binding differs")
    require(results["execution_integrity"] == execution, "verification results integrity binding differs")
    require(results["quality_gate"] == recomputed_quality, "verification results quality binding differs")

    prohibited = ("ppa", "prototype", "benchmark", "signoff", "tapeout", "silicon")
    commands = [" ".join(item["command"]).lower() for item in results["commands"]]
    require(not any(token in command for token in prohibited for command in commands), "downstream command recorded")
    require(all(item["exit_status"] == 0 for item in results["commands"]), "retained command failure recorded")
    for item in results["commands"]:
        verify_artifact(item["raw_log"])

    results_input_record = artifact(RESULTS)
    RESULTS_INPUT_ARCHIVE.mkdir(parents=True, exist_ok=True)
    results_input_path = RESULTS_INPUT_ARCHIVE / f"RESULTS.{results_input_record['sha256']}.json"
    if not results_input_path.exists():
        shutil.copy2(RESULTS, results_input_path)
    results_input_record = artifact(results_input_path)

    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "schema_version": 2,
        "reviewed_at_utc": now,
        "reviewer_role": "independent_l2",
        "review_execution_context": "independent_reviewer_round_after_engineer_discriminator_execution",
        "reviewer_status": "done",
        "stage": "verification",
        "scope": "authoritative_retained_discriminator_and_execution_integrity_only",
        "stage_closing": False,
        "manager_stage_transition_owner": "Manager",
        "contract_id": CONTRACT,
        "candidate_id": CANDIDATE_ID,
        "candidate_rtl_hash": CANDIDATE_HASH,
        "live_rtl_source_sha256": RTL_SHA256,
        "decision": "accept_integrity_no_go",
        "technical_disposition": "bounded_no_go",
        "execution_integrity_disposition": "duplicate_successful_execution_exact_one_contract_failed",
        "candidate_capability_accepted": False,
        "quality_discriminator_complete": True,
        "focused_discriminator_passed": False,
        "execution_contract_satisfied": False,
        "reason": (
            "Independent read-only recomputation confirms the retained technical bounded NO_GO and the separate execution-integrity NO_GO. "
            "The first successful execution covers exactly WikiText-2 and C4-en-512, at most 128 captured tokens, all 24 ordered layers, "
            "bit-identical layer-0 inputs, and all-lane scalar-oracle agreement, but eight frozen quality conditions fail. "
            "A second successful execution produced an equivalent payload at a later timestamp, so the exact-one acceptance contract is irreversibly unsatisfied. "
            "Prior Engineer/Planner-generated files that self-labeled as independent L2 are retained only as provenance and are not Reviewer acceptance. "
            "No discriminator or downstream flow was executed by this L2 audit."
        ),
        "provenance_audit": {
            "prior_non_reviewer_self_attestations": [
                {"producer_role": role, "artifact": artifact(path)}
                for role, path in PRIOR_NON_REVIEWER_VERDICTS
            ],
            "disposition": "not_accepted_as_independent_l2",
        },
        "review_tool": artifact(Path(__file__).resolve()),
        "authoritative_discriminator_decision": artifact(DECISION),
        "verification_results_review_input": results_input_record,
        "technical_evidence": {
            "first_baseline_replay": artifact(FIRST_BASELINE),
            "first_candidate_replay": artifact(FIRST_CANDIDATE),
            "later_duplicate_baseline_replay": artifact(SECOND_BASELINE),
            "later_duplicate_candidate_replay": artifact(SECOND_CANDIDATE),
            "payload_equivalent_except_generated_at_utc": True,
            "datasets": ["wikitext2", "c4_en_512"],
            "capture_token_limit": 128,
            "layers": 24,
        },
        "quality_gate_recomputed": recomputed_quality,
        "ordered_trace_captures": decision["ordered_trace_captures"],
        "contract_preservation": {
            "ordered_supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "non_sram_area_cap_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "manager_stage_remains": "verification",
        },
        "downstream_runs": {
            "third_discriminator_run": False,
            "paired_smoke": False,
            "full_shell": False,
            "candidate_ppa": False,
            "prototype": False,
            "benchmark": False,
            "signoff": False,
        },
        "required_manager_action": "rollback_to_architecture_for_structurally_distinct_successor",
        "required_operator_action": "none",
        "claim_boundary": "Independent L2 acceptance of the technical bounded NO_GO and duplicate-execution integrity NO_GO only; no stage transition or downstream acceptance.",
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    payload["integrity"]["canonical_sha256"] = canonical_sha256(payload)
    dump(OUT, payload)
    review_record = artifact(OUT)
    decision_record = artifact(DECISION)

    results["status"] = "integrity_no_go"
    results["independent_l2_review"] = review_record
    results["required_manager_action"] = "manager_rollback_to_architecture_for_structurally_distinct_successor"
    results["integrity"]["canonical_sha256"] = None
    results["integrity"]["canonical_sha256"] = canonical_sha256(results)
    dump(RESULTS, results)

    PLAN.write_text(
        """# ACE-2 down-projection residual-fusion verification plan

This plan is limited to the active `verification` stage and candidate
`shared_down_projection_residual_fusion_v1` at RTL hash
`76ef0dda2e646558ecb3e2047d3ec00d69f416cbe0a880b3402876614a4a7a28`.
It does not run or claim PPA, prototype, benchmark, signoff, tapeout, or silicon evidence.

## Acceptance gates

- Independent oracle includes standalone and all-lane arithmetic checks plus frozen quality constraints.
- Coverage stress includes reset, boundary, stalls/backpressure, illegal/error, randomized,
  single-clock CDC disposition, X/Z, saturation/overflow, formal, and representative datasets.
- Reproducibility requires retained successful commands and non-contradictory hash-bound artifacts.
  The exact-one discriminator execution contract remains binding.

## Authoritative disposition

Technical status: `bounded_no_go`; eight frozen quality conditions fail.
Execution-integrity status: `integrity_no_go`; two successful executions violate exact-one.
An independent read-only L2 audit confirms both dispositions. Stage closing: `false`.
A third discriminator execution is prohibited; the Manager owns rollback or hold routing.
""",
        encoding="utf-8",
    )
    CHECKPOINT.write_text(
        f"""# Goal

Close the bounded verification discriminator and independent L2 disposition for
`shared_down_projection_residual_fusion_v1` without entering downstream stages.

# Current State

The Manager-owned stage remains `verification`. Candidate `{CANDIDATE_ID}` remains bound
to RTL hash `{CANDIDATE_HASH}`. The first successful discriminator execution at
`2026-08-02T00:14:50Z` is authoritative for technical evidence only.

# Result

Status: `integrity_no_go`. Stage closing: `false`.
Independent read-only L2 recomputation confirms the technical `bounded_no_go`: all 24 layers
and scalar-oracle traces are present, but eight frozen quality conditions fail. It also confirms
that a second successful execution at `2026-08-02T00:17:24Z` irreversibly violated the exact-one
contract. The two payloads are equivalent except for generation timestamps; this does not cure
the integrity failure. No third discriminator run was performed.

# Boundaries

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
`layer_0.rope_q`. The 2.0 mm2 non-SRAM cap and 100 MHz floor are unchanged.
No paired smoke, shell admission, PPA, prototype, benchmark, signoff, tapeout, or silicon run
was performed. Only the Manager may change `current_stage`; the required routing is rollback
to architecture for a structurally distinct successor or an explicit Manager hold.
""",
        encoding="utf-8",
    )
    update_public(now, review_record, decision_record)
    print(
        "ACE2_DPRF_INDEPENDENT_L2_INTEGRITY_NO_GO_PASS "
        f"verdict_sha256={sha256(OUT)} failed_conditions={len(recomputed_quality['failed_conditions'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
