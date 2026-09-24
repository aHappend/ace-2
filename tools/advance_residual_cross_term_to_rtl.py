#!/usr/bin/env python3
"""Exercise the Manager-only environment-to-RTL transition for the active successor."""

from __future__ import annotations

import copy
import hashlib
import json
import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_qk_residual_cross_term_attention_v1"
NEXT_ACTION = "implement_bounded_reference_rtl_and_run_consolidated_preflight"
RTL_ENTRY_STATUS = "manager_advanced_to_rtl_bounded_reference_rtl_and_preflight_authorized"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"
REVIEW = (
    "evidence/review/"
    "environment_stage_closing_shared_qk_residual_cross_term_attention_v1/decision.json"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: str) -> dict[str, Any]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def dump(path: str, value: dict[str, Any]) -> None:
    (ROOT / path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def artifact(path: str) -> dict[str, Any]:
    item = ROOT / path
    return {"path": path, "bytes": item.stat().st_size, "sha256": sha256(path)}


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def first_unsupported(scope: dict[str, Any]) -> str | None:
    value = scope.get("implementation_frontier", {}).get("first_unsupported_layer_operator")
    if isinstance(value, dict):
        return value.get("identifier")
    return value


def live_record(
    source: dict[str, Any],
    *,
    allow_historical_rtl_started: bool = False,
) -> dict[str, Any]:
    record = copy.deepcopy(source)
    require(record.get("contract_id") == CONTRACT, "active contract changed")
    require(record.get("implementation_authorized") is True, "authorization missing")
    require(record.get("operator_approval_consumed") is False, "authorization consumed")
    if not allow_historical_rtl_started:
        require(record.get("rtl_started") is False, "successor RTL already started")
    require(record.get("stage_closing") is False, "bounded attempt became stage-closing")
    record.update(
        {
            "implementation_authorized": True,
            "operator_approval_consumed": False,
            "required_manager_action": NEXT_ACTION,
            "rtl_started": False,
            "stage_closing": False,
            "status": RTL_ENTRY_STATUS,
        }
    )
    return record


def checkpoint_text() -> str:
    return f"""# Goal

Execute the Manager-only `environment -> rtl` transition for
`{CONTRACT}` while preserving the frozen accelerator contract.

# Current State

The Manager advanced `current_stage` to `rtl` after the independent environment
review returned `done`. Standing `implementation_authorized=true` is active and
unconsumed. Successor RTL has not started.

Only the bounded independent reference/RTL implementation and its consolidated
preflight are authorized next. Paired smoke, shell admission/regression, PPA,
FPGA/prototype, benchmark, signoff, GDS, tapeout, and silicon work remain locked.

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
`layer_0.rope_q`; mode remains `ADVANCE`. The 2.0 mm2 non-SRAM cap, 100 MHz
floor, abstract streaming-memory boundary, and historical PPA frontier are
unchanged.

# Verified Done

- Pre-transition decisive verifier: `python3 tools/check_environment_stage_closing_review.py`
- Independent decision: `{REVIEW}`
- Environment evidence remains sealed as historical input; no raw probe was rerun.
- Manager transition is recorded in `research/PIPELINE_STATE.json`.

# Failed / Dead Ends

None in this transition.

# Open Questions / Blockers

- Bounded reference/RTL implementation and consolidated preflight are pending.
- Independent Reviewer acceptance is required after that bounded increment.

# Relevant Files and Evidence

- `research/PIPELINE_STATE.json`
- `research/ENVIRONMENT_REVIEWER_VERDICT.json`
- `{REVIEW}`
- `evidence/environment_preflight/latest/PACKET.json`
- `design/NUMERICAL_REPLACEMENT_PROPOSAL.md`
- `design/RTL_MANIFEST.json`
"""


def transition(*, reconcile_current_rtl: bool = False) -> None:
    pipeline = load("research/PIPELINE_STATE.json")
    if reconcile_current_rtl:
        require(pipeline.get("current_stage") == "rtl", "Manager stage is not rtl")
        event = pipeline.get("stage_history", [])[-1]
        require(
            event.get("by") == "manager"
            and event.get("from_stage") == "environment"
            and event.get("to_stage") == "rtl",
            "latest Manager history item is not the authorized transition",
        )
    else:
        require(pipeline.get("current_stage") == "environment", "Manager stage is not environment")
    review = load(REVIEW)
    verdict = load("research/ENVIRONMENT_REVIEWER_VERDICT.json")
    audit = load("research/ENVIRONMENT_AUDIT.json")
    preflight = load("evidence/environment_preflight/latest/PACKET.json")
    require(review.get("contract_id") == CONTRACT, "review contract mismatch")
    require(review.get("decision", {}).get("status") == "done", "review is not done")
    require(review.get("evidence", {}).get("unchanged_audit_was_not_rerun") is True,
            "review did not preserve the no-rerun boundary")
    require(verdict.get("contract_id") == CONTRACT and verdict.get("verdict") == "done",
            "canonical environment verdict is not done")
    require(verdict.get("implementation_authorized") is True,
            "canonical verdict lost implementation authorization")
    require(verdict.get("required_repairs_before_manager_advance") == [],
            "environment review has unresolved repairs")
    require(audit.get("readiness_summary", {}).get("raw_capability_artifact_count") == 27,
            "environment capability artifact count changed")
    require(audit.get("readiness_summary", {}).get("raw_probes_rerun") is False,
            "raw probes were rerun")
    require(all(audit.get("stage_gate_items", {}).values()), "environment checklist incomplete")
    require(preflight.get("contract_id") == CONTRACT and all(preflight.get("checklist", {}).values()),
            "consolidated environment preflight is not complete")

    target = load("design/TARGET.json")
    scope = load("design/CHIP_SCOPE.json")
    manifest = load("design/RTL_MANIFEST.json")
    policy = load("design/FAST_LOOP_POLICY.json")
    oracle = load("reference/ORACLE_MANIFEST.json")
    require(target.get("frozen_numeric_targets", {}).get("area_cap_non_sram_mm2") == 2.0,
            "2.0 mm2 non-SRAM cap changed")
    require(target.get("frozen_numeric_targets", {}).get("frequency_floor_mhz") == 100.0,
            "100 MHz floor changed")
    require(scope.get("implementation_frontier", {}).get("ordered_supported_layer_operator_prefix") == PREFIX,
            "accepted prefix changed")
    require(first_unsupported(scope) == FIRST_UNSUPPORTED,
            "first unsupported operator changed")
    historical_rtl = manifest.get("historical_standalone_rtl")
    if historical_rtl:
        require(
            manifest.get("candidate_rtl_hash") == historical_rtl.get("candidate_rtl_hash"),
            "historical standalone RTL hash binding changed",
        )
        require(
            "historical" in str(manifest.get("candidate_rtl_hash_scope") or ""),
            "historical standalone RTL scope changed",
        )
    else:
        require(manifest.get("candidate_rtl_hash") is None, "successor candidate hash already exists")
        require(manifest.get("candidate_rtl_hash_scope") == "successor_rtl_not_started",
                "successor RTL start boundary changed")

    now = utc_now()
    reason = (
        "Both environment checklist items are independently Reviewer-certified for "
        f"{CONTRACT}; all 27 capability artifacts and the consolidated preflight are "
        "hash-bound, with no raw probe rerun and no unresolved repair. Advance to RTL "
        "while preserving standing implementation_authorized=true, stage_closing=false, "
        "ADVANCE mode, the accepted prefix through layer_0.v_proj, first unsupported "
        "layer_0.rope_q, the 2.0 mm2 non-SRAM cap, the 100 MHz floor, and the abstract "
        "streaming-memory boundary. Authorize only the bounded reference/RTL implementation "
        "and consolidated preflight; paired smoke, shell, and PPA remain locked."
    )
    if not reconcile_current_rtl:
        pipeline["current_stage"] = "rtl"
        pipeline.setdefault("stage_history", []).append(
            {
                "at": now,
                "by": "manager",
                "direction": "advance",
                "from_stage": "environment",
                "reason": reason,
                "to_stage": "rtl",
            }
        )
        dump("research/PIPELINE_STATE.json", pipeline)

    active = live_record(
        policy["active_repair_authorization"],
        allow_historical_rtl_started=bool(historical_rtl),
    )
    for key in (
        "active_repair_authorization",
        "architecture_proposal_authorization",
        "selected_replacement_contract",
    ):
        policy[key] = copy.deepcopy(active)
    policy["manager_recommendation"] = NEXT_ACTION
    dump("design/FAST_LOOP_POLICY.json", policy)

    target["current_stage"] = "rtl"
    target["current_architecture_contract"] = copy.deepcopy(active)
    target["fast_loop_contract"]["manager_recommendation"] = NEXT_ACTION
    for key in ("active_repair_authorization", "architecture_proposal_authorization"):
        target["fast_loop_contract"][key] = copy.deepcopy(active)
    dump("design/TARGET.json", target)

    scope["stage"] = {
        "current_stage": "rtl",
        "current_stage_status": RTL_ENTRY_STATUS + "_not_started",
        "downstream_stages_locked_until_manager_advance": [
            "verification", "ppa", "prototype", "benchmark", "signoff"
        ],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    scope["authority_override"]["required_next_action"] = NEXT_ACTION
    scope["authority_override"]["architecture_review_status"] = (
        "architecture_and_environment_independently_accepted_rtl_entry_open"
    )
    scope["authority_override"]["operator_implementation_approval"] = copy.deepcopy(active)
    scope["operator_owned_execution_policy"]["active_successor_contract"] = copy.deepcopy(active)
    scope["implementation_frontier"]["latest_decision"] = (
        "manager_advanced_to_rtl_bounded_reference_rtl_and_preflight_pending"
    )
    dump("design/CHIP_SCOPE.json", scope)

    manifest["stage"] = "rtl"
    manifest["current_stage"] = "rtl"
    manifest["architecture_contract_status"] = RTL_ENTRY_STATUS + "_not_started"
    manifest["candidate_status"] = "bounded_reference_rtl_and_consolidated_preflight_pending"
    manifest["candidate_id"] = None
    manifest["candidate_rtl_hash"] = None
    manifest["candidate_rtl_hash_scope"] = "successor_rtl_not_started"
    manifest["candidate_source_hashes"] = {}
    manifest["candidate_generated_hashes"] = {}
    manifest["candidate_rtl_sources"] = []
    manifest["candidate_generated_sources"] = []
    manifest["candidate_interface"]["status"] = "rtl_entry_open_candidate_not_started"
    manifest["candidate_review_binding"]["status"] = (
        "environment_independently_accepted_manager_advanced_to_rtl_candidate_not_started"
    )
    manifest["proposed_replacement_contract"] = copy.deepcopy(active)
    manifest["traceability"]["architecture_contract_gap"] = {
        "resolution_owner": "bounded implementation then independent RTL Reviewer",
        "status": "rtl_entry_open_bounded_reference_rtl_and_preflight_pending",
    }
    dump("design/RTL_MANIFEST.json", manifest)

    oracle["stage"] = "rtl"
    oracle["current_stage"] = "rtl"
    oracle["architecture_contract_status"] = RTL_ENTRY_STATUS + "_not_started"
    oracle["proposed_architecture_contract"] = copy.deepcopy(active)
    dump("reference/ORACLE_MANIFEST.json", oracle)

    (ROOT / "CHECKPOINT.md").write_text(checkpoint_text(), encoding="utf-8")

    status = load("research/PUBLIC_STATUS.json")
    status["current_mode"] = "ADVANCE"
    status["latest_decision"] = (
        "manager_advanced_to_rtl_bounded_reference_rtl_and_preflight_pending"
    )
    status["architecture_proposal_gate"].update(
        {
            "contract_id": CONTRACT,
            "implementation_authorized": True,
            "required_manager_action": NEXT_ACTION,
            "required_operator_action": "none_standing_authorization_is_current",
            "status": RTL_ENTRY_STATUS,
        }
    )
    status["selected_replacement_contract"].update(copy.deepcopy(active))
    status["stage"] = {
        "completed_prior_stages": ["definition", "architecture", "environment"],
        "current_stage": "rtl",
        "current_stage_checklist": copy.deepcopy(manifest["traceability"]["stage_checklist"]),
        "current_stage_evidence": [
            "research/PIPELINE_STATE.json",
            "research/ENVIRONMENT_REVIEWER_VERDICT.json",
            REVIEW,
            "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
            "design/RTL_MANIFEST.json",
        ],
        "current_stage_status": RTL_ENTRY_STATUS + "_not_started",
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }
    latest_environment = status.get("latest_environment_stage", {})
    latest_environment.update(
        {
            "implementation_authorized": True,
            "reviewer_verdict": "done",
            "stage_transition_completed": True,
            "status": "independent_review_done_manager_advanced_to_rtl",
        }
    )
    status["latest_environment_stage"] = latest_environment
    status["blockers"] = [
        {
            "id": "bounded_reference_rtl_and_consolidated_preflight_pending",
            "stage": "rtl",
            "status": "active",
            "reason": "Manager opened RTL entry; the successor reference/RTL increment has not started.",
            "required_resolution": (
                "Implement only the frozen bounded reference/RTL candidate and run its "
                "consolidated preflight before independent Reviewer acceptance."
            ),
            "evidence": "research/PIPELINE_STATE.json",
        }
    ]
    for name in ("dashboard_fields", "implementation_frontier"):
        container = status[name]
        container["current_stage"] = "rtl"
        container["current_mode"] = "ADVANCE"
        container["latest_decision"] = status["latest_decision"]
        container["required_manager_action"] = NEXT_ACTION
        container["required_operator_action"] = "none_standing_authorization_is_current"
        container["candidate_mechanism"] = copy.deepcopy(active)
        container["operator_policy"] = copy.deepcopy(policy)
        nested_environment = container.get("latest_environment_stage", {})
        nested_environment.update(copy.deepcopy(latest_environment))
        container["latest_environment_stage"] = nested_environment
    status["public_claims"] = [
        {
            "claim": "current Manager-owned stage is rtl",
            "evidence": ["research/PIPELINE_STATE.json", "research/PUBLIC_STATUS.json"],
        },
        {
            "claim": (
                "independent environment review passed and the Manager advanced the active "
                "shared QK residual cross-term contract to rtl"
            ),
            "evidence": ["research/ENVIRONMENT_REVIEWER_VERDICT.json", REVIEW,
                         "research/PIPELINE_STATE.json"],
        },
        {
            "claim": (
                "only bounded reference/RTL implementation and consolidated preflight are "
                "authorized; successor RTL has not started"
            ),
            "evidence": ["design/RTL_MANIFEST.json", "design/FAST_LOOP_POLICY.json"],
        },
        {
            "claim": (
                "the supported prefix, 2.0 mm2 non-SRAM cap, 100 MHz floor, and abstract "
                "streaming-memory boundary are unchanged"
            ),
            "evidence": ["design/CHIP_SCOPE.json", "design/TARGET.json"],
        },
        {
            "claim": (
                "no successor paired smoke, shell admission or regression, PPA, prototype, "
                "benchmark, signoff, GDS, tapeout, or silicon result exists"
            ),
            "evidence": ["design/RTL_MANIFEST.json"],
        },
    ]
    public_paths = {
        item.get("path") for item in status.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path") != "research/PUBLIC_STATUS.json"
    }
    public_paths.update(
        {
            "CHECKPOINT.md",
            "design/CHIP_SCOPE.json",
            "design/FAST_LOOP_POLICY.json",
            "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
            "design/RTL_MANIFEST.json",
            "design/TARGET.json",
            "reference/ORACLE_MANIFEST.json",
            "research/ENVIRONMENT_REVIEWER_VERDICT.json",
            "research/PIPELINE_STATE.json",
            REVIEW,
            "tools/advance_residual_cross_term_to_rtl.py",
        }
    )
    status["artifact_hashes"] = [
        artifact(path) for path in sorted(public_paths) if (ROOT / path).is_file()
    ]
    status["last_updated_utc"] = now
    status["generated_at_utc"] = now
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": (
            "UTF-8 sorted keys two-space indentation trailing newline canonical hash null during hash"
        ),
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump("research/PUBLIC_STATUS.json", status)


def validate() -> None:
    pipeline = load("research/PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "rtl", "Manager stage did not advance")
    event = pipeline.get("stage_history", [])[-1]
    require(
        event.get("by") == "manager"
        and event.get("from_stage") == "environment"
        and event.get("to_stage") == "rtl",
        "latest Manager history item is not the authorized transition",
    )
    target = load("design/TARGET.json")
    scope = load("design/CHIP_SCOPE.json")
    manifest = load("design/RTL_MANIFEST.json")
    policy = load("design/FAST_LOOP_POLICY.json")
    oracle = load("reference/ORACLE_MANIFEST.json")
    status = load("research/PUBLIC_STATUS.json")
    require(target.get("current_stage") == "rtl", "TARGET stage stale")
    require(scope.get("stage", {}).get("current_stage") == "rtl", "CHIP_SCOPE stage stale")
    require(manifest.get("current_stage") == "rtl", "RTL_MANIFEST stage stale")
    require(oracle.get("current_stage") == "rtl", "ORACLE_MANIFEST stage stale")
    require(status.get("stage", {}).get("current_stage") == "rtl", "PUBLIC_STATUS stage stale")
    require(manifest.get("candidate_rtl_hash") is None, "candidate RTL unexpectedly exists")
    require(manifest.get("candidate_rtl_hash_scope") == "successor_rtl_not_started",
            "candidate start boundary stale")
    require(target.get("frozen_numeric_targets", {}).get("area_cap_non_sram_mm2") == 2.0,
            "area target changed")
    require(target.get("frozen_numeric_targets", {}).get("frequency_floor_mhz") == 100.0,
            "frequency target changed")
    require(scope.get("implementation_frontier", {}).get("ordered_supported_layer_operator_prefix") == PREFIX,
            "accepted prefix changed")
    require(first_unsupported(scope) == FIRST_UNSUPPORTED,
            "first unsupported operator changed")
    for key in (
        "active_repair_authorization",
        "architecture_proposal_authorization",
        "selected_replacement_contract",
    ):
        record = policy[key]
        require(record.get("contract_id") == CONTRACT, f"policy contract stale: {key}")
        require(record.get("implementation_authorized") is True, f"authorization stale: {key}")
        require(record.get("operator_approval_consumed") is False, f"approval consumed: {key}")
        require(record.get("rtl_started") is False, f"RTL marked started: {key}")
        require(record.get("stage_closing") is False, f"stage_closing changed: {key}")
        require(record.get("required_manager_action") == NEXT_ACTION, f"routing stale: {key}")
    require(status.get("integrity", {}).get("canonical_sha256") == canonical_sha256(status),
            "PUBLIC_STATUS canonical hash stale")
    require("/home/" not in json.dumps(status, sort_keys=True), "PUBLIC_STATUS leaks private paths")
    for record in status.get("artifact_hashes", []):
        path = str(record.get("path") or "")
        require((ROOT / path).is_file(), f"public artifact missing: {path}")
        require((ROOT / path).stat().st_size == record.get("bytes"),
                f"public artifact byte count stale: {path}")
        require(sha256(path) == record.get("sha256"), f"public artifact hash stale: {path}")
    blocker_ids = {item.get("id") for item in status.get("blockers", [])}
    require(blocker_ids == {"bounded_reference_rtl_and_consolidated_preflight_pending"},
            "post-transition blocker set is not bounded")
    forbidden_locks = policy["active_repair_authorization"].get("forbidden", [])
    for forbidden in ("paired_smoke", "shell", "ppa"):
        require(any(forbidden in item for item in forbidden_locks),
                f"missing downstream lock: {forbidden}")
    checkpoint = (ROOT / "CHECKPOINT.md").read_text(encoding="utf-8")
    require("advanced `current_stage` to `rtl`" in checkpoint, "checkpoint transition missing")


def refresh_public_integrity() -> None:
    status = load("research/PUBLIC_STATUS.json")
    paths = [str(item.get("path")) for item in status.get("artifact_hashes", [])]
    status["artifact_hashes"] = [artifact(path) for path in paths]
    status.setdefault("integrity", {})["canonical_sha256"] = None
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump("research/PUBLIC_STATUS.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--refresh-public-integrity", action="store_true")
    parser.add_argument("--reconcile-current-rtl", action="store_true")
    args = parser.parse_args()
    selected_actions = sum(
        bool(value)
        for value in (args.check, args.refresh_public_integrity, args.reconcile_current_rtl)
    )
    if selected_actions > 1:
        parser.error("choose at most one action")
    if args.refresh_public_integrity:
        refresh_public_integrity()
    elif args.reconcile_current_rtl:
        transition(reconcile_current_rtl=True)
    elif not args.check:
        transition()
    validate()
    print(
        "ACE2_MANAGER_RTL_TRANSITION_PASS "
        f"contract={CONTRACT} stage=rtl rtl_started=false "
        "authorized=bounded_reference_rtl_plus_consolidated_preflight"
    )
