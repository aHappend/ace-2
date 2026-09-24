#!/usr/bin/env python3
"""Bind independent acceptance of the Q/K rotation bounded no-go."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "evidence/review/rope_pair_rotation_no_go_l2/decision.json"
RESULTS = ROOT / "evidence/rope_pair_rotation/latest/RESULTS.json"
MANIFEST = ROOT / "design/RTL_MANIFEST.json"
TRACEABILITY = ROOT / "design/RTL_TRACEABILITY.md"
POLICY = ROOT / "design/FAST_LOOP_POLICY.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
LIVE_VIEW = ROOT / ".argus/live-view.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    canonical = json.loads(json.dumps(value))
    canonical.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(canonical, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def refresh_public_integrity(public: dict[str, Any]) -> None:
    tracked = {
        item["path"]
        for item in public.get("artifact_hashes", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    tracked.update(
        path.relative_to(ROOT).as_posix()
        for path in (
            CHECKPOINT,
            MANIFEST,
            TRACEABILITY,
            POLICY,
            PIPELINE,
            RESULTS,
            DECISION,
            LIVE_VIEW,
            ROOT / "tools/run_rope_pair_rotation_no_go_l2.py",
            ROOT / "tools/bind_rope_pair_rotation_no_go_l2.py",
        )
    )
    tracked.discard(PUBLIC.relative_to(ROOT).as_posix())
    public["artifact_hashes"] = [
        artifact(ROOT / relative)
        for relative in sorted(tracked)
        if (ROOT / relative).is_file()
    ]
    public["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonical_sha256": None,
        "canonicalization": (
            "UTF-8, sorted keys, two-space indentation, trailing newline, with "
            "integrity.canonical_sha256 set to null"
        ),
    }
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)


def main() -> None:
    review = load(DECISION)
    result = load(RESULTS)
    pipeline = load(PIPELINE)
    decision = review.get("decision", {})
    stage_name = str(pipeline.get("current_stage") or "")
    require(
        stage_name in {"rtl", "architecture"},
        "Manager-owned stage is neither rtl nor the required architecture rollback",
    )
    rollback_completed = stage_name == "architecture"
    require(review.get("stage_closing") is False, "review is unexpectedly stage-closing")
    require(decision.get("status") == "done", "independent no-go review is not done")
    require(review.get("smoke_gate_passed") is False, "review packet says smoke passed")
    require(review.get("rtl_contract_traceability") is False, "contract gap was lost")
    require(review.get("full_shell_regression_run") is False, "full shell was run")
    require(review.get("canonical_sky130_ppa_run") is False, "PPA was run")
    require(
        review.get("evidence", {}).get("results", {}).get("sha256")
        == sha256_file(RESULTS),
        "review is not bound to the current no-go result",
    )
    require(
        result.get("status") == "bounded_no_go_rope_pair_rotation_c4_regressed",
        "rotation no-go result changed",
    )

    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    review_status = {
        "accepted": True,
        "accepted_scope": "bounded_negative_result_and_expensive_run_blocking_only",
        "candidate_capability_accepted": False,
        "decision": "done",
        "evidence": DECISION.relative_to(ROOT).as_posix(),
        "evidence_sha256": sha256_file(DECISION),
        "reviewed_at_utc": review["reviewed_at_utc"],
        "rtl_contract_traceability": False,
        "routing_authorized": (
            "manager_rollback_completed_operator_contract_required"
            if rollback_completed
            else "manager_rollback_to_architecture_only"
        ),
        "stage_closing": False,
    }

    manifest = load(MANIFEST)
    manifest["candidate_review_binding"] = {
        "decision": "done",
        "level": "independent_l2",
        "result_binding": RESULTS.relative_to(ROOT).as_posix(),
        "review_binding": DECISION.relative_to(ROOT).as_posix(),
        "review_sha256": sha256_file(DECISION),
        "review_status": "bounded_rotation_no_go_accepted_not_capability",
    }
    manifest["independent_reviewer_acceptance"] = False
    manifest["independent_reviewer_verdict"] = (
        "rotation_bounded_no_go_accepted_candidate_capability_rejected"
    )
    manifest["independent_no_go_review"] = review_status
    manifest.setdefault("latest_evidence", {})["rotation_no_go_l2"] = artifact(DECISION)
    manifest["current_stage"] = stage_name
    manifest["architecture_contract_status"] = (
        "awaiting_operator_approved_replacement_after_manager_rollback"
        if rollback_completed
        else "not_satisfied_by_failed_rotation_candidate"
    )
    if rollback_completed:
        gap = manifest.setdefault("traceability", {}).setdefault(
            "architecture_contract_gap", {}
        )
        gap["resolution_owner"] = (
            "operator approves a replacement numerical contract; architecture "
            "must then refreeze compute, interface, risk, and area-reuse plans"
        )
    manifest["generated_at_utc"] = now
    write(MANIFEST, manifest)

    policy = load(POLICY)
    active = policy["active_repair_authorization"]
    active["execution_status"] = "completed_bounded_no_go_independently_accepted"
    active["independent_reviewer_gate"] = review_status
    active["replacement_numerical_contract_authorized"] = False
    active["required_manager_action"] = (
        "completed_rollback_to_architecture"
        if rollback_completed
        else "rollback_to_architecture"
    )
    active["required_operator_action"] = "approve_replacement_numerical_contract"
    active["updated_at_utc"] = now
    write(POLICY, policy)

    trace = TRACEABILITY.read_text(encoding="utf-8").rstrip()
    if rollback_completed:
        trace = trace.replace(
            "Manager rollback to `architecture` is required before an\n"
            "operator-approved replacement numerical contract can be implemented.",
            "Manager rollback to `architecture` is complete. An operator-approved\n"
            "replacement numerical contract is required before architecture can be\n"
            "refrozen or RTL implementation can resume.",
        )
    routing_sentence = (
        "Manager rollback to `architecture` is complete; explicit operator approval "
        "of a replacement numerical contract is now required."
        if rollback_completed
        else "Manager rollback to `architecture` is the only authorized routing action."
    )
    review_section = f"""

## Independent review

Independent L2 accepted only the bounded negative result and the decision to
block full-shell/PPA after the failed two-dataset gate. It did not accept the
rotation candidate as a supported capability, close the RTL stage, or authorize
a replacement numerical contract. Review: `{DECISION.relative_to(ROOT).as_posix()}`
(`{sha256_file(DECISION)}`). The RTL contract-traceability item remains unmet.
{routing_sentence}
"""
    if "## Independent review" in trace:
        trace = trace.split("## Independent review", 1)[0].rstrip()
    TRACEABILITY.write_text(trace + review_section, encoding="utf-8")

    routing_block = (
        "Manager completed the rollback to `architecture`. No replacement numerical "
        "contract is authorized. The operator must explicitly approve the next "
        "architecture/interface/numerical contract before architecture can be "
        "refrozen or RTL implementation can resume."
        if rollback_completed
        else "Manager must roll back to `architecture`; the operator must explicitly "
        "approve any replacement numerical contract."
    )
    CHECKPOINT.write_text(
        f"""# Goal

Advance the complete Qwen2.5-0.5B W4A8 accelerator without relaxing the 1.05x
quality limit, 2.0 mm^2 non-SRAM cap, or 100 MHz SKY130 floor.

# Current State

The Manager-owned stage is `{stage_name}`. The accepted ordered prefix remains
through `layer_0.v_proj`; `layer_0.rope_q` is first unsupported.

Independent L2 accepted the shared +22.5 degree Q/K basis-rotation result only
as a bounded no-go. WikiText-2 improved from
`{result['smoke_gate']['ratios']['wikitext2']['baseline']}` to
`{result['smoke_gate']['ratios']['wikitext2']['candidate']}`, while C4-en
regressed from `{result['smoke_gate']['ratios']['c4_en_512']['baseline']}` to
`{result['smoke_gate']['ratios']['c4_en_512']['candidate']}`. The required joint
gate therefore failed, and completed full-shell regression and canonical SKY130
PPA were correctly not run.

The failed rotation candidate retains scalar Q/K scales and does not implement
the accepted per-head 256-byte metadata-table architecture.
`rtl.contract-traceability` remains unmet. The independent review does not
accept the candidate as capability or authorize a replacement numerical contract.

# Required Routing

No additional RTL numerical candidate is authorized under the consumed repair
contract. {routing_block} Planner must not edit
`research/PIPELINE_STATE.json` or perform environment/RTL/verification/PPA/
prototype/benchmark/signoff work while the replacement architecture contract is
unapproved.

# Relevant Evidence

- `evidence/rope_pair_rotation/latest/RESULTS.json`
- `evidence/review/rope_pair_rotation_no_go_l2/decision.json`
- `evidence/per_head_qk_repair/latest/RESULTS.json`
- `design/RTL_MANIFEST.json`
- `design/RTL_TRACEABILITY.md`
- `research/PUBLIC_STATUS.json`
""",
        encoding="utf-8",
    )

    public = load(PUBLIC)
    public["generated_at_utc"] = now
    public["last_updated_utc"] = now
    public["latest_decision"] = (
        "manager_rolled_back_to_architecture_after_independent_rotation_no_go"
        if rollback_completed
        else "independent_l2_accepted_rotation_no_go_manager_architecture_rollback_required"
    )
    public["rope_pair_rotation_no_go_l2_review"] = review_status
    public["blockers"] = [
        {
            "id": "replacement_numerical_contract_not_operator_approved",
            "stage": stage_name,
            "status": "active",
            "reason": (
                "Both authorized numerical repair directions are bounded no-gos. "
                "The Manager has rolled back to architecture, but no replacement "
                "numerical/interface contract has operator approval."
                if rollback_completed
                else "Both authorized numerical repair directions are bounded no-gos, "
                "and the active failed rotation RTL does not implement the accepted "
                "per-head metadata-table architecture."
            ),
            "required_resolution": (
                "Explicit operator approval of a replacement numerical contract."
                if rollback_completed
                else "Manager rollback to architecture and explicit operator approval "
                "of a replacement numerical contract."
            ),
            "evidence": DECISION.relative_to(ROOT).as_posix(),
        }
    ]
    stage = public.setdefault("stage", {})
    stage["current_stage"] = stage_name
    stage["completed_prior_stages"] = [
        name
        for name, value in pipeline.get("stages", {}).items()
        if isinstance(value, dict) and value.get("status") == "done"
    ]
    stage["current_stage_status"] = (
        "architecture_waiting_operator_approved_replacement_numerical_contract"
        if rollback_completed
        else "rotation_no_go_independently_accepted_rtl_contract_unmet_requires_architecture_rollback"
    )
    stage["current_stage_checklist"] = (
        {
            "architecture.compute-memory-model": False,
            "architecture.interface-control": False,
            "architecture.leverage-risk": False,
            "architecture.area-reuse-plan": False,
        }
        if rollback_completed
        else {
            "rtl.contract-traceability": False,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
        }
    )
    evidence = []
    for relative in (
        "design/ARCHITECTURE.md",
        "design/MEMORY_MODEL.json",
        "design/SPEC.md",
        POLICY.relative_to(ROOT).as_posix(),
        PIPELINE.relative_to(ROOT).as_posix(),
        RESULTS.relative_to(ROOT).as_posix(),
        DECISION.relative_to(ROOT).as_posix(),
        MANIFEST.relative_to(ROOT).as_posix(),
        TRACEABILITY.relative_to(ROOT).as_posix(),
        CHECKPOINT.relative_to(ROOT).as_posix(),
    ):
        if relative not in evidence:
            evidence.append(relative)
    stage["current_stage_evidence"] = evidence
    stage["downstream_locked_until_manager_advance"] = (
        ["environment", "rtl", "verification", "ppa", "prototype", "benchmark", "signoff"]
        if rollback_completed
        else ["verification", "ppa", "prototype", "benchmark", "signoff"]
    )

    for section_name in ("implementation_frontier", "dashboard_fields"):
        section = public.setdefault(section_name, {})
        section["current_stage"] = stage_name
        section["latest_decision"] = public["latest_decision"]
        section["rope_pair_rotation_no_go_l2_review"] = review_status
        section["rtl_contract_traceability"] = False
        section["required_manager_action"] = (
            "completed_rollback_to_architecture"
            if rollback_completed
            else "rollback_to_architecture"
        )
        section["required_operator_action"] = "approve_replacement_numerical_contract"
        section["routing_status"] = (
            "architecture_waiting_operator_approved_replacement_numerical_contract"
            if rollback_completed
            else "rotation_no_go_independently_accepted_awaiting_manager_rollback_and_operator_contract"
        )

    claim = {
        "claim": (
            "independent L2 accepted the +22.5 degree Q/K basis result only as a "
            "bounded no-go and confirmed that the failed joint smoke gate correctly "
            "blocked completed full-shell and canonical SKY130 PPA runs"
        ),
        "evidence": [
            RESULTS.relative_to(ROOT).as_posix(),
            DECISION.relative_to(ROOT).as_posix(),
        ],
    }
    claims = [
        item
        for item in public.get("public_claims", [])
        if not (
            isinstance(item, dict)
            and (
                "independent L2 accepted the +22.5 degree Q/K basis"
                in str(item.get("claim") or "")
                or str(item.get("claim") or "").startswith(
                    "the current Manager-owned stage is "
                )
            )
        )
    ]
    stage_claim = {
        "claim": f"the current Manager-owned stage is {stage_name}",
        "evidence": [
            PIPELINE.relative_to(ROOT).as_posix(),
            PUBLIC.relative_to(ROOT).as_posix(),
        ],
    }
    public["public_claims"] = [*claims, stage_claim, claim]
    write(
        LIVE_VIEW,
        {
            "version": 1,
            "title": (
                "Architecture replan awaiting operator contract"
                if rollback_completed
                else "RTL no-go: architecture replan required"
            ),
            "paths": [
                "research/PIPELINE_STATE.json",
                "evidence/rope_pair_rotation/latest/RESULTS.json",
                "evidence/review/rope_pair_rotation_no_go_l2/decision.json",
                "design/FAST_LOOP_POLICY.json",
                "design/ARCHITECTURE.md",
                "research/PUBLIC_STATUS.json",
            ],
            "reason": (
                "Shows the Manager architecture rollback, independently certified "
                "two-dataset no-go, consumed repair authorization, immutable targets, "
                "and the operator-only replacement-contract blocker."
                if rollback_completed
                else "Shows the certified two-dataset failure, independent review, "
                "traceability mismatch, consumed repair authorization, and required "
                "Manager rollback."
            ),
        },
    )
    refresh_public_integrity(public)
    write(PUBLIC, public)

    print(
        "ACE2_ROTATION_NO_GO_L2_BIND_PASS "
        f"decision_sha256={sha256_file(DECISION)} stage={pipeline['current_stage']}"
    )


if __name__ == "__main__":
    main()
