#!/usr/bin/env python3
"""Refresh privacy-filtered public status for the RTL quality rollback."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_STATUS = ROOT / "research" / "PUBLIC_STATUS.json"
PIPELINE_STATE = ROOT / "research" / "PIPELINE_STATE.json"
ROUTING = (
    ROOT
    / "evidence"
    / "diagnostics"
    / "static-scale-contract-routing-20260731T083144Z"
    / "RESULTS.json"
)
REVIEW_VERDICT = ROOT / "evidence" / "review" / "latest" / "ppa_repair_verdict.json"
TIMESTAMP = "2026-07-31T08:41:29Z"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(relative: str) -> dict[str, object]:
    path = ROOT / relative
    return {
        "bytes": path.stat().st_size if path.is_file() else 0,
        "exists": path.is_file(),
        "path": relative,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def canonical_sha256(data: dict[str, object]) -> str:
    canonical = copy.deepcopy(data)
    canonical["integrity"]["canonical_sha256"] = None
    payload = json.dumps(canonical, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    pipeline = json.loads(PIPELINE_STATE.read_text(encoding="utf-8"))
    if pipeline.get("current_stage") != "rtl":
        raise RuntimeError("public rollback status requires Manager-owned current_stage=rtl")
    routing = json.loads(ROUTING.read_text(encoding="utf-8"))
    review = json.loads(REVIEW_VERDICT.read_text(encoding="utf-8"))
    status = json.loads(PUBLIC_STATUS.read_text(encoding="utf-8"))

    candidate = routing["current_rtl_candidate"]
    if review.get("status") != "accepted" or review.get("reviewer_role") != "independent_reviewer":
        raise RuntimeError("RTL/PPA prerequisite lacks an accepted independent review")
    if review.get("candidate_id") != candidate["candidate_id"]:
        raise RuntimeError("independent review is bound to a different RTL candidate")
    if (
        review.get("ordered_source_hash_list_sha256")
        != candidate["ordered_source_hash_list_sha256"]
    ):
        raise RuntimeError("independent review source binding differs from routing evidence")
    reviewed_artifacts = {
        "candidate_packet_sha256": candidate["candidate_evidence"]["path"],
        "full_shell_log_sha256": (
            "evidence/candidates/ppa_repair_6388ec6ae7085bd7/rtl_shell_sim.log"
        ),
        "full_shell_verification_binding_sha256": (
            "evidence/candidates/ppa_repair_6388ec6ae7085bd7/"
            "full_shell_verification_binding.json"
        ),
        "rtl_lint_log_sha256": (
            "evidence/candidates/ppa_repair_6388ec6ae7085bd7/rtl_lint.log"
        ),
        "sky130_sta_log_sha256": (
            "evidence/candidates/ppa_repair_6388ec6ae7085bd7/sky130_sta.log"
        ),
        "sky130_yosys_log_sha256": (
            "evidence/candidates/ppa_repair_6388ec6ae7085bd7/sky130_yosys.log"
        ),
    }
    for field, relative in reviewed_artifacts.items():
        if review["evidence_hashes"].get(field) != sha256_file(ROOT / relative):
            raise RuntimeError(f"independent review artifact hash mismatch: {relative}")
    frozen = routing["quality_observations"]["frozen_max_abs_static_per_tensor"]
    percentile = routing["quality_observations"]["diagnostic_static_percentile_0_999"]
    routing_path = ROUTING.relative_to(ROOT).as_posix()

    status["generated_at_utc"] = TIMESTAMP
    status["last_updated_utc"] = TIMESTAMP
    status["blockers"] = [
        {
            "id": "rtl_quality_contract_requires_architecture_replan",
            "evidence": routing_path,
            "resolution": (
                "Manager rollback to architecture is required to freeze and budget a "
                "structurally different W4A8 activation-scale or metadata mechanism; "
                "the 1.05x quality targets remain unchanged."
            ),
        }
    ]

    dashboard = status["dashboard_fields"]
    dashboard["current_stage"] = "rtl"
    dashboard["latest_decision"] = "request_architecture_replan_for_w4a8_quality_contract"
    dashboard["latest_rtl_candidate"] = {
        "candidate_id": candidate["candidate_id"],
        "candidate_rtl_hash": candidate["ordered_source_hash_list_sha256"],
        "status": "independent_l2_accepted_bounded_prerequisite_quality_no_go",
        "accepted_publication_frontier_changed": False,
        "acceptance_scope": review["acceptance_scope"],
        "area_cap_met": candidate["area_cap_met"],
        "frequency_floor_met": candidate["frequency_floor_met"],
        "non_sram_area_mm2": candidate["sky130_non_sram_area_mm2"],
        "setup_slack_ns_at_100mhz": candidate["sky130_setup_slack_ns_at_100mhz"],
        "evidence": candidate["candidate_evidence"]["path"],
        "independent_l2_verdict": REVIEW_VERDICT.relative_to(ROOT).as_posix(),
    }
    dashboard["latest_quality_diagnostic"] = {
        "status": "failed_quality_target_architecture_replan_required",
        "first_material_divergence": "model.layers.0.score",
        "frozen_wikitext2_perplexity_ratio": frozen["wikitext2_perplexity_ratio"],
        "frozen_c4_en_512_perplexity_ratio": frozen["c4_en_512_perplexity_ratio"],
        "diagnostic_0_999_wikitext2_perplexity_ratio": percentile[
            "wikitext2_perplexity_ratio"
        ],
        "diagnostic_0_999_c4_en_512_perplexity_ratio": percentile[
            "c4_en_512_perplexity_ratio"
        ],
        "quality_target_ratio_max": 1.05,
        "targets_relaxed": False,
        "evidence": routing_path,
    }

    frontier = status["implementation_frontier"]
    frontier["latest_decision"] = "request_architecture_replan_for_w4a8_quality_contract"
    frontier["latest_rtl_candidate"] = dashboard["latest_rtl_candidate"]
    frontier["latest_quality_diagnostic"] = dashboard["latest_quality_diagnostic"]

    status["public_claims"] = [
        {
            "claim": "current Manager-owned stage is rtl",
            "evidence": ["research/PIPELINE_STATE.json", "research/PUBLIC_STATUS.json"],
        },
        {
            "claim": (
                "candidate 6388ec6a passes exact-source full-shell verification and "
                "canonical SKY130 area/timing thresholds and has bounded independent L2 "
                "prerequisite acceptance"
            ),
            "evidence": [
                candidate["candidate_evidence"]["path"],
                REVIEW_VERDICT.relative_to(ROOT).as_posix(),
            ],
        },
        {
            "claim": (
                "the fresh paired smoke fails both immutable 1.05x perplexity limits and "
                "the first material localized divergence is the layer-0 attention score"
            ),
            "evidence": [routing_path],
        },
        {
            "claim": (
                "tested local static-scale substitutions do not close quality, so the next "
                "credible mechanism requires an architecture contract review"
            ),
            "evidence": [routing_path],
        },
        {
            "claim": (
                "the independent RTL/PPA prerequisite acceptance does not change the "
                "accepted publication frontier or relax the full-model quality targets"
            ),
            "evidence": [REVIEW_VERDICT.relative_to(ROOT).as_posix(), routing_path],
        },
    ]

    status["stage"] = {
        "completed_prior_stages": ["definition", "architecture", "environment"],
        "current_stage": "rtl",
        "current_stage_checklist": {
            "rtl.contract-traceability": False,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
        },
        "current_stage_evidence": [
            candidate["candidate_evidence"]["path"],
            REVIEW_VERDICT.relative_to(ROOT).as_posix(),
            routing_path,
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
        ],
        "current_stage_source": "research/PIPELINE_STATE.json",
        "downstream_locked_until_manager_advance": [
            "verification",
            "ppa",
            "prototype",
            "benchmark",
            "signoff",
        ],
        "planner_may_advance_stage": False,
        "stage_transition_owner": "Manager",
    }

    required_artifacts = {
        item["path"] for item in status.get("artifact_hashes", [])
    }
    required_artifacts.update(
        {
            "research/PIPELINE_STATE.json",
            routing_path,
            candidate["candidate_evidence"]["path"],
            frozen["path"],
            percentile["path"],
            routing["quality_observations"]["diagnostic_static_percentile_0_99"]["path"],
            routing["quality_observations"]["first_material_divergence"]["path"],
            routing["bounded_framework_repair"]["source"]["path"],
            routing["bounded_framework_repair"]["test_source"]["path"],
            REVIEW_VERDICT.relative_to(ROOT).as_posix(),
        }
    )
    status["artifact_hashes"] = [
        artifact(relative) for relative in sorted(required_artifacts)
    ]
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": (
            "UTF-8, sorted keys, two-space indentation, trailing newline, with "
            "integrity.canonical_sha256 set to null"
        ),
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    PUBLIC_STATUS.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "ACE2_PUBLIC_STATUS_RTL_ROLLBACK_PASS "
        f"canonical_sha256={status['integrity']['canonical_sha256']}"
    )


if __name__ == "__main__":
    main()
