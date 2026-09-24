#!/usr/bin/env python3
"""Publish the blocked per-head Q/K L2 attempt without claiming acceptance."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "evidence/review/per_head_qk_no_go_l2/decision.json"
PUBLIC_STATUS = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"
RESULTS = ROOT / "evidence/per_head_qk_repair/latest/RESULTS.json"
CHECKPOINT = ROOT / "CHECKPOINT.md"
RECOVERY = ROOT / "evidence/per_head_qk_repair/archive/FOCUSED_EVIDENCE_RECOVERY.json"
RECOVERED_LINT = ROOT / (
    "evidence/per_head_qk_repair/archive/"
    "rtl_lint.8d853ad49d9d869c4a5cc27ec282304a9517058fb36590ae2a97b6beb1ce78fd.log"
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def main() -> None:
    review = load(DECISION)
    pipeline = load(PIPELINE_STATE)
    result = load(RESULTS)
    decision = review.get("decision", {})
    require(pipeline.get("current_stage") == "rtl", "Manager-owned stage changed")
    require(decision.get("status") == "blocked", "review is not blocked")
    require(
        "Evidence supports the bounded no-go and expensive-run blocking"
        in str(decision.get("reason") or ""),
        "review did not substantively support the bounded no-go",
    )
    require(review.get("smoke_gate_passed") is False, "smoke gate is not failed")
    require(review.get("full_shell_regression_run") is False, "full shell was run")
    require(review.get("canonical_sky130_ppa_run") is False, "PPA was run")
    require(
        sha256_file(RESULTS)
        == "b56df2f506ba733af2050500d728ebeefadceb27acc0e014c3de7391684ca027",
        "frozen no-go packet changed",
    )
    require(result.get("smoke_gate", {}).get("passed") is False, "result gate changed")

    timestamp = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    review_status = {
        "accepted": False,
        "attempt_count": 1,
        "attempted_at_utc": review["reviewed_at_utc"],
        "decision": "blocked_checkpoint_handoff_not_formal_acceptance",
        "evidence": DECISION.relative_to(ROOT).as_posix(),
        "evidence_sha256": sha256_file(DECISION),
        "expensive_runs_correctly_blocked_supported": True,
        "negative_result_supported": True,
        "operator_authorization_consumed": "exactly_one_bounded_l2_review",
        "routing_authorized": False,
        "stage_closing": False,
    }

    status = load(PUBLIC_STATUS)
    status["per_head_qk_no_go_l2_review"] = review_status
    for key in ("implementation_frontier", "dashboard_fields"):
        section = status.setdefault(key, {})
        section["per_head_qk_no_go_l2_review"] = review_status
        section["latest_decision"] = (
            "hold_manager_routing_per_head_qk_no_go_l2_blocked_checkpoint_handoff"
        )
        section["routing_status"] = (
            "no_structural_diagnosis_dispatch_without_formal_l2_acceptance"
        )

    existing_blockers = [
        item
        for item in status.get("blockers", [])
        if isinstance(item, dict)
        and item.get("id") != "per_head_qk_no_go_l2_checkpoint_handoff_blocked"
    ]
    status["blockers"] = [
        {
            "evidence": DECISION.relative_to(ROOT).as_posix(),
            "id": "per_head_qk_no_go_l2_checkpoint_handoff_blocked",
            "reason": (
                "The sole authorized L2 call supported the bounded negative result "
                "and expensive-run blocking but returned blocked because its "
                "read-only sandbox could not create CHECKPOINT.md."
            ),
            "required_resolution": (
                "Do not route the structural diagnosis. Fresh operator authorization "
                "is required before any second independent L2 call."
            ),
            "stage": "rtl",
            "status": "active",
        },
        *existing_blockers,
    ]

    claim = {
        "claim": (
            "one independent L2 call found the per-head Q/K bounded no-go and "
            "expensive-run blocking supported, but formal acceptance is absent "
            "because the verdict was blocked on checkpoint write access"
        ),
        "evidence": [
            DECISION.relative_to(ROOT).as_posix(),
            CHECKPOINT.relative_to(ROOT).as_posix(),
        ],
    }
    claims = [
        item
        for item in status.get("public_claims", [])
        if not (
            isinstance(item, dict)
            and "one independent L2 call found the per-head Q/K" in str(item.get("claim") or "")
        )
    ]
    status["public_claims"] = [*claims, claim]
    stage = status.setdefault("stage", {})
    stage["current_stage_status"] = (
        "per_head_qk_no_go_l2_blocked_no_follow_on_routing"
    )
    stage_evidence = list(stage.get("current_stage_evidence", []))
    for relative in (
        RESULTS.relative_to(ROOT).as_posix(),
        DECISION.relative_to(ROOT).as_posix(),
        CHECKPOINT.relative_to(ROOT).as_posix(),
    ):
        if relative not in stage_evidence:
            stage_evidence.append(relative)
    stage["current_stage_evidence"] = stage_evidence

    paths = {
        item["path"]
        for item in status.get("artifact_hashes", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    paths.update(
        path.relative_to(ROOT).as_posix()
        for path in (
            CHECKPOINT,
            DECISION,
            RECOVERY,
            RECOVERED_LINT,
            ROOT / "tools/run_per_head_qk_no_go_l2.py",
            ROOT / "tools/bind_per_head_qk_no_go_l2_block.py",
        )
    )
    paths.discard(PUBLIC_STATUS.relative_to(ROOT).as_posix())
    status["artifact_hashes"] = [
        artifact(ROOT / relative)
        for relative in sorted(paths)
        if (ROOT / relative).is_file()
    ]
    status["generated_at_utc"] = timestamp
    status["last_updated_utc"] = timestamp
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonical_sha256": None,
        "canonicalization": (
            "UTF-8, sorted keys, two-space indentation, trailing newline, with "
            "integrity.canonical_sha256 set to null"
        ),
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    PUBLIC_STATUS.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "ACE2_PER_HEAD_QK_L2_BLOCK_BIND_PASS "
        f"decision_sha256={sha256_file(DECISION)} stage={pipeline['current_stage']}"
    )


if __name__ == "__main__":
    main()
