#!/usr/bin/env python3
"""Reconcile public/current-facing status after the sealed tile-BFP no-go."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PUBLIC_STATUS = ROOT / "research/PUBLIC_STATUS.json"
POLICY = ROOT / "design/FAST_LOOP_POLICY.json"
LIVE_VIEW = ROOT / ".argus/live-view.json"
NO_GO = (
    ROOT
    / "evidence/layer0_tile_bfp_score_attention_v1/latest/BOUNDED_NO_GO.json"
)

CONTRACT = "layer0_tile_bfp_score_attention_v1"
NO_GO_SHA256 = "c4a9084dcfe137451a8e7b226df7990fe8a4fe6dc8f922f5b88879a251ce875c"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {
        "bytes": path.stat().st_size,
        "path": relative,
        "sha256": sha256_file(path),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    candidate = copy.deepcopy(value)
    candidate.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    pipeline = load(PIPELINE)
    policy = load(POLICY)
    no_go = load(NO_GO)

    if pipeline.get("current_stage") != "rtl":
        raise RuntimeError("status repair requires Manager-owned current_stage=rtl")
    if sha256_file(NO_GO) != NO_GO_SHA256:
        raise RuntimeError("sealed tile-BFP no-go hash changed")
    authorization = policy.get("active_repair_authorization", {})
    if (
        authorization.get("contract_id") != CONTRACT
        or authorization.get("implementation_authorized") is not False
        or authorization.get("operator_approval_consumed") is not True
    ):
        raise RuntimeError("operator authorization state does not match sealed no-go")

    result_relative = NO_GO.relative_to(ROOT).as_posix()
    source_list_sha = no_go["source_binding"]["ordered_source_hash_list"]["sha256"]
    proposal_sha = no_go["proposal_sha256"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    live_view = {
        "version": 1,
        "title": "Tile-BFP no-go awaiting Manager architecture reroute",
        "paths": [
            "research/PIPELINE_STATE.json",
            result_relative,
            "research/PUBLIC_STATUS.json",
            "CHECKPOINT.md",
        ],
        "reason": (
            "The tile-BFP candidate is a sealed bounded no-go and its one-time "
            "implementation approval is consumed. The Manager-owned stage remains rtl; "
            "the next legal move is rtl to architecture for a structurally distinct "
            "contract with fresh operator approval."
        ),
    }
    dump(LIVE_VIEW, live_view)

    status = load(PUBLIC_STATUS)
    current_contract = {
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "result_binding": result_relative,
        "result_binding_sha256": NO_GO_SHA256,
        "status": "rejected_unique_paired_smoke",
    }
    architecture_gate = {
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "required_manager_action": (
            "reroute_rtl_to_architecture_for_structurally_distinct_contract"
        ),
        "required_operator_action": (
            "fresh_approval_after_structurally_distinct_architecture_contract"
        ),
        "result_binding": result_relative,
        "result_binding_sha256": NO_GO_SHA256,
        "source": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        "status": "authorization_consumed_rejected_unique_paired_smoke",
    }
    candidate_mechanism = {
        "contract_id": CONTRACT,
        "implementation_authorized": False,
        "mechanism": (
            "wide_tile_extrema_then_signed24_block_floating_score_"
            "hierarchical_softmax"
        ),
        "predecessor_no_go": (
            "evidence/layer0_tile_max_delta_attention_v1/latest/BOUNDED_NO_GO.json"
        ),
        "proposal": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        "proposal_sha256": proposal_sha,
        "result_binding": result_relative,
        "result_binding_sha256": NO_GO_SHA256,
        "stage_closing": False,
        "status": "authorization_consumed_rejected_unique_paired_smoke",
    }

    latest_environment = copy.deepcopy(status["latest_environment_stage"])
    latest_environment.update(
        {
            "contract_binding": CONTRACT,
            "implementation_authorized": False,
            "primary_fast_loop_ready": False,
            "status": "historical_rtl_entry_certification_candidate_now_rejected",
        }
    )

    status["architecture_proposal_gate"] = architecture_gate
    status["selected_replacement_contract"] = current_contract
    status["current_mode"] = "ADVANCE"
    status["first_unsupported_layer_operator"] = "layer_0.rope_q"
    status["supported_layer_operator_prefix"] = PREFIX
    status["latest_decision"] = "tile_bfp_unique_paired_smoke_bounded_no_go"
    status["latest_ppa_frontier_status"] = (
        "historical_frontier_preserved_no_candidate_ppa"
    )
    status["latest_environment_stage"] = latest_environment
    status["blockers"] = [
        {
            "evidence": result_relative,
            "id": "manager_architecture_reroute_required",
            "reason": (
                "The tile-BFP candidate passed focused arithmetic and quality checks "
                "but failed both frozen unique paired-smoke improvement thresholds."
            ),
            "required_resolution": (
                "Manager rolls rtl back to architecture for one structurally distinct "
                "successor; implementation then requires fresh operator approval."
            ),
            "stage": "rtl",
            "status": "active",
        }
    ]

    dispatch = status.get("architecture_successor_dispatch")
    if isinstance(dispatch, dict):
        dispatch.update(
            {
                "current_stage": "rtl",
                "implementation_authorized": False,
                "result_binding": result_relative,
                "result_binding_sha256": NO_GO_SHA256,
                "required_manager_action": (
                    "reroute_rtl_to_architecture_for_structurally_distinct_contract"
                ),
                "status": "historical_dispatch_completed_candidate_rejected",
            }
        )

    for name in ("dashboard_fields", "implementation_frontier"):
        view = status[name]
        view.update(
            {
                "candidate_mechanism": candidate_mechanism,
                "candidate_rtl_hash": source_list_sha,
                "candidate_rtl_hash_scope": (
                    "tile_bfp_reference_vectors_standalone_rtl_and_focused_quality_"
                    "not_shell_admitted"
                ),
                "current_mode": "ADVANCE",
                "current_stage": "rtl",
                "first_unsupported_layer_operator": "layer_0.rope_q",
                "latest_decision": "tile_bfp_unique_paired_smoke_bounded_no_go",
                "latest_environment_stage": latest_environment,
                "latest_ppa_frontier_status": (
                    "historical_frontier_preserved_no_candidate_ppa"
                ),
                "operator_policy": copy.deepcopy(policy),
                "ordered_supported_layer_operator_prefix": PREFIX,
                "routing_status": "manager_architecture_reroute_required",
                "rtl_contract_traceability": False,
                "supported_layer_operator_prefix": PREFIX,
            }
        )
        cycle_impact = view.get("latest_ppa_frontier", {}).get(
            "cycle_or_tokens_per_second_impact"
        )
        if isinstance(cycle_impact, dict):
            cycle_impact["status"] = "not_remeasured_for_rejected_tile_bfp_candidate"

    status["generated_at_utc"] = now
    status["last_updated_utc"] = now

    artifact_paths = {
        item["path"]
        for item in status.get("artifact_hashes", [])
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and (ROOT / item["path"]).is_file()
        and item["path"] != PUBLIC_STATUS.relative_to(ROOT).as_posix()
    }
    artifact_paths.update(
        {
            LIVE_VIEW.relative_to(ROOT).as_posix(),
            NO_GO.relative_to(ROOT).as_posix(),
            POLICY.relative_to(ROOT).as_posix(),
            PIPELINE.relative_to(ROOT).as_posix(),
            Path(__file__).resolve().relative_to(ROOT).as_posix(),
        }
    )
    status["artifact_hashes"] = [
        artifact(relative) for relative in sorted(artifact_paths)
    ]
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": (
            "UTF-8 sorted keys two-space indentation trailing newline; "
            "integrity.canonical_sha256 null during hash"
        ),
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    dump(PUBLIC_STATUS, status)

    print(
        "ACE2_TILE_BFP_PUBLIC_STATUS_PASS "
        f"canonical_sha256={status['integrity']['canonical_sha256']} "
        f"no_go_sha256={NO_GO_SHA256}"
    )


if __name__ == "__main__":
    main()
